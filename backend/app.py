# app.py
"""
Robust Flask app for ParkingVision.
Registers startup handler dynamically (compatible with Flask 2.x and 3.x)
so importing the module never raises AttributeError.
"""
from flask import Flask, jsonify, render_template, Response, request
from flask_cors import CORS
import os
import time
import threading
from pathlib import Path
import cv2
import numpy as np
import pickle

# local import (ensure core/parking_monitor.py exists)
from core.parking_monitor import ParkingMonitor

# -------------------------
# App and folders
# -------------------------
BASE_DIR = Path(__file__).parent.resolve()
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), template_folder=str(TEMPLATES_DIR))
CORS(app, resources={r"/api/*": {"origins": "*"}})

ENABLE_VIDEO = os.getenv("ENABLE_VIDEO", "true").lower() in ("1", "true", "yes")

# -------------------------
# Shared state
# -------------------------
parking_data = {
    "lot1": {"total": 0, "available": 0, "occupied": 0, "history": []},
    "lot2": {"total": 0, "available": 0, "occupied": 0, "history": []},
}
frames = {"1": b"", "2": b""}
frame_locks = {"1": threading.Lock(), "2": threading.Lock()}

video_processors = []
processing_started = False
processing_lock = threading.Lock()

# -------------------------
# Video processing
# -------------------------
class VideoProcessor:
    def __init__(self, video_idx, video_path, pos_file, monitor):
        self.video_idx = int(video_idx)
        self.video_path = Path(video_path)
        self.pos_file = Path(pos_file)
        self.monitor = monitor
        self.positions = []
        self.cap = None
        self.fps = 30.0
        self.frame_delay = 1.0 / 30.0
        self.lot_key = f"lot{self.video_idx + 1}"
        self.lot_id = str(self.video_idx + 1)
        self._setup()

    def _setup(self):
        try:
            if self.pos_file.exists():
                with open(self.pos_file, "rb") as f:
                    self.positions = pickle.load(f)
                parking_data[self.lot_key]["total"] = len(self.positions)
                print(f"Loaded {len(self.positions)} positions for lot {self.video_idx + 1}")
            else:
                print(f"Positions file not found: {self.pos_file}")

            if self.video_path.exists():
                self.cap = cv2.VideoCapture(str(self.video_path))
                if self.cap.isOpened():
                    fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                    self.fps = fps if fps > 0 else 30.0
                    self.frame_delay = 1.0 / self.fps
                    try:
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    print(f"Video {self.video_idx + 1} opened, FPS: {self.fps}")
                else:
                    print(f"Failed to open video: {self.video_path}")
                    self.cap = None
            else:
                print(f"Video file not found: {self.video_path}")

        except Exception as e:
            print(f"Error during VideoProcessor setup for lot {self.video_idx + 1}: {e}")
            self.cap = None

    def _process_frame(self, frame):
        if not self.positions:
            return frame, 0

        processed_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 1)
        thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 16)
        processed = cv2.medianBlur(thresh, 5)
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.dilate(processed, kernel, iterations=1)

        free = 0
        for x, y, w, h in self.positions:
            spot = processed[y:y+h, x:x+w]
            nonzero = cv2.countNonZero(spot)
            if nonzero <= (w * h * self.monitor.OCCUPANCY_THRESHOLD):
                color = (0, 255, 0)
                free += 1
            else:
                color = (0, 0, 255)
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), color, 2)

        status = f"Free: {free}/{len(self.positions)}"
        cv2.putText(processed_frame, status, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2, cv2.LINE_AA)
        return processed_frame, free

    def run(self):
        if not self.cap:
            print(f"No capture for lot {self.video_idx + 1} — thread exiting.")
            return

        last = time.time()
        frame_count = 0
        while True:
            try:
                ok, frame = self.cap.read()
                if not ok:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                annotated, free = self._process_frame(frame)
                total = len(self.positions)
                occupied = max(total - free, 0)
                parking_data[self.lot_key]["available"] = free
                parking_data[self.lot_key]["occupied"] = occupied

                now = time.strftime("%H:%M:%S")
                hist = parking_data[self.lot_key]["history"]
                hist.append({"time": now, "available": free, "occupied": occupied})
                if len(hist) > 50:
                    del hist[:-50]

                ok_enc, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok_enc:
                    with frame_locks[self.lot_id]:
                        frames[self.lot_id] = buf.tobytes()

                elapsed = time.time() - last
                if elapsed < self.frame_delay:
                    time.sleep(self.frame_delay - elapsed)
                last = time.time()

                frame_count += 1
                if frame_count % 200 == 0:
                    print(f"Lot {self.video_idx + 1} streaming - target FPS {self.fps:.2f}")

            except Exception as e:
                print(f"Error in video loop (lot {self.video_idx + 1}): {e}")
                time.sleep(0.1)

# -------------------------
# Setup threads exactly once
# -------------------------
def setup_video_processing():
    global processing_started
    if not ENABLE_VIDEO:
        print("ENABLE_VIDEO=false -> skipping video processing.")
        return

    with processing_lock:
        if processing_started:
            return
        monitor = ParkingMonitor()
        for i in range(min(2, len(monitor.video_paths))):
            p = VideoProcessor(i, monitor.video_paths[i], monitor.pos_files[i], monitor)
            video_processors.append(p)

        for proc in video_processors:
            t = threading.Thread(target=proc.run, daemon=True)
            t.start()

        processing_started = True
        print(f"Started {len(video_processors)} video thread(s).")

# Register start handler dynamically (no decorators that might not exist)
def _start_background():
    try:
        setup_video_processing()
    except Exception as e:
        print(f"Error starting background processing: {e}")

if hasattr(app, "before_serving"):
    # Flask 3.x (and 2.2+): before_serving exists
    app.before_serving(_start_background)
elif hasattr(app, "before_first_request"):
    # older Flask versions
    app.before_first_request(_start_background)
else:
    # No safe hook available — we'll start lazily on first request
    print("No before_serving/before_first_request hook found — will start on first request.")

def ensure_processing_started():
    if not processing_started:
        setup_video_processing()

# -------------------------
# Streaming generator
# -------------------------
def generate_frames(lot_id: str):
    while True:
        try:
            with frame_locks[lot_id]:
                data = frames[lot_id]
                if not data:
                    blank = np.zeros((480, 640, 3), np.uint8)
                    txt = f"Loading Lot {lot_id}..."
                    cv2.putText(blank, txt, (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    _, buf = cv2.imencode(".jpg", blank)
                    data = buf.tobytes()

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            time.sleep(0.033)
        except Exception as e:
            print(f"Stream error (lot {lot_id}): {e}")
            time.sleep(0.1)

# -------------------------
# Routes
# -------------------------
def _template_exists(name: str) -> bool:
    return (TEMPLATES_DIR / name).exists()

@app.route("/health")
def health():
    ensure_processing_started()
    return jsonify({"status": "ok", "enable_video": ENABLE_VIDEO, "processing_started": processing_started})

@app.route("/")
def index():
    ensure_processing_started()
    if _template_exists("index.html"):
        return render_template("index.html")
    return jsonify({
        "message": "ParkingVision backend running",
        "endpoints": ["/health", "/api/parking-data", "/api/parking-details?lot=1", "/api/video-stream?lot=1"]
    })

@app.route("/details.html")
def details():
    ensure_processing_started()
    if _template_exists("details.html"):
        return render_template("details.html")
    return jsonify({"message": "details.html not found in templates"}), 200

@app.route("/api/parking-data")
def get_parking_data():
    ensure_processing_started()
    return jsonify({
        "lot1": {
            "total": parking_data["lot1"]["total"],
            "available": parking_data["lot1"]["available"],
            "occupied": parking_data["lot1"]["occupied"],
        },
        "lot2": {
            "total": parking_data["lot2"]["total"],
            "available": parking_data["lot2"]["available"],
            "occupied": parking_data["lot2"]["occupied"],
        },
    })

@app.route("/api/parking-details")
def get_parking_details():
    ensure_processing_started()
    lot = request.args.get("lot", "1")
    key = f"lot{lot}"
    if key not in parking_data:
        return jsonify({"error": "Invalid lot number"}), 400
    return jsonify({
        "total": parking_data[key]["total"],
        "available": parking_data[key]["available"],
        "occupied": parking_data[key]["occupied"],
        "history": parking_data[key]["history"],
    })

@app.route("/api/video-stream")
def video_stream():
    ensure_processing_started()
    lot = request.args.get("lot", "1")
    if lot not in ("1", "2"):
        return "Invalid lot number", 400
    return Response(generate_frames(lot), mimetype="multipart/x-mixed-replace; boundary=frame")

# -------------------------
# Local run
# -------------------------
if __name__ == "__main__":
    if ENABLE_VIDEO:
        setup_video_processing()
        time.sleep(1)
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
