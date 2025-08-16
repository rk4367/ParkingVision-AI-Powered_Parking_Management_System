# app.py
from flask import Flask, send_from_directory, jsonify, Response, request
from flask_cors import CORS
import os
import cv2
import numpy as np
import pickle
import threading
import time
from pathlib import Path
from core.parking_monitor import ParkingMonitor
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__, static_folder="static", static_url_path="/")
CORS(app)

# Global variables to store parking data
parking_data = {
    'lot1': {'total': 0, 'available': 0, 'occupied': 0, 'history': []},
    'lot2': {'total': 0, 'available': 0, 'occupied': 0, 'history': []}
}

# Create frames to store the latest processed frames for streaming
frames = {
    '1': b'',
    '2': b''
}

# Thread locks for frame updates
frame_locks = {
    '1': threading.Lock(),
    '2': threading.Lock()
}

# Video processing class for better performance
class VideoProcessor:
    def __init__(self, video_idx, video_path, pos_file, monitor):
        self.video_idx = video_idx
        self.video_path = video_path
        self.pos_file = pos_file
        self.monitor = monitor
        self.positions = []
        self.cap = None
        self.fps = 30  # Default FPS
        self.frame_delay = 1.0 / 30  # Default delay between frames
        self.lot_key = f'lot{video_idx + 1}'
        self.lot_id = str(video_idx + 1)
        
        self.setup_video()
    
    def setup_video(self):
        """Initialize video capture and load positions"""
        try:
            if self.pos_file.exists():
                with open(self.pos_file, 'rb') as f:
                    self.positions = pickle.load(f)
                    parking_data[self.lot_key]['total'] = len(self.positions)
                    print(f"Loaded {len(self.positions)} positions for lot {self.video_idx + 1}")
            
            if self.video_path.exists():
                self.cap = cv2.VideoCapture(str(self.video_path))
                if self.cap.isOpened():
                    # Get actual FPS from video
                    self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                    if self.fps <= 0:
                        self.fps = 30  # Fallback FPS
                    
                    self.frame_delay = 1.0 / self.fps
                    print(f"Video {self.video_idx + 1} FPS: {self.fps}")
                    
                    # Set buffer size to reduce latency
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    print(f"Failed to open video {self.video_idx + 1}")
                    self.cap = None
            else:
                print(f"Video file not found: {self.video_path}")
                
        except Exception as e:
            print(f"Error setting up video {self.video_idx + 1}: {e}")
            self.cap = None
    
    def process_single_frame(self, frame):
        """Process a single frame for parking detection"""
        if not self.positions:
            return frame, 0
        
        # Create a copy for processing
        processed_frame = frame.copy()
        
        # Convert to grayscale and apply preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 1)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 25, 16)
        processed = cv2.medianBlur(thresh, 5)
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.dilate(processed, kernel, iterations=1)
        
        # Count free spots
        free_count = 0
        for x, y, w, h in self.positions:
            spot = processed[y:y+h, x:x+w]
            nonzero = cv2.countNonZero(spot)
            
            if nonzero <= (w * h * self.monitor.OCCUPANCY_THRESHOLD):
                color = (0, 255, 0)  # Green - free
                free_count += 1
            else:
                color = (0, 0, 255)  # Red - occupied
            
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), color, 2)
        
        # Add status text
        status = f"Free: {free_count}/{len(self.positions)}"
        cv2.putText(processed_frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 200, 0), 2, cv2.LINE_AA)
        
        return processed_frame, free_count
    
    def run_processing(self):
        """Main processing loop for this video"""
        if not self.cap:
            print(f"No video capture available for lot {self.video_idx + 1}")
            return
        
        frame_count = 0
        last_time = time.time()
        
        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    # Loop video when it ends
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Process frame
                processed_frame, free_count = self.process_single_frame(frame)
                
                # Update parking data
                occupied_count = len(self.positions) - free_count
                parking_data[self.lot_key]['available'] = free_count
                parking_data[self.lot_key]['occupied'] = occupied_count
                
                # Add to history (limit to last 50 entries)
                now = time.strftime("%H:%M:%S")
                history_entry = {
                    'time': now,
                    'available': free_count,
                    'occupied': occupied_count
                }
                
                parking_data[self.lot_key]['history'].append(history_entry)
                if len(parking_data[self.lot_key]['history']) > 50:
                    parking_data[self.lot_key]['history'] = parking_data[self.lot_key]['history'][-50:]
                
                # Encode frame for streaming
                ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    with frame_locks[self.lot_id]:
                        frames[self.lot_id] = buffer.tobytes()
                
                # Frame rate control - maintain video's original speed
                current_time = time.time()
                elapsed = current_time - last_time
                
                if elapsed < self.frame_delay:
                    time.sleep(self.frame_delay - elapsed)
                
                last_time = time.time()
                frame_count += 1
                
                # Print FPS every 100 frames for debugging
                if frame_count % 100 == 0:
                    actual_fps = 100 / (time.time() - (last_time - elapsed))
                    print(f"Lot {self.video_idx + 1} - Target FPS: {self.fps:.1f}, Actual FPS: {actual_fps:.1f}")
                
            except Exception as e:
                print(f"Error processing frame for lot {self.video_idx + 1}: {e}")
                time.sleep(0.1)

# Global video processors
video_processors = []

def setup_video_processing():
    """Setup video processors for all lots"""
    monitor = ParkingMonitor()
    
    for i in range(min(2, len(monitor.video_paths))):
        processor = VideoProcessor(i, monitor.video_paths[i], monitor.pos_files[i], monitor)
        video_processors.append(processor)
    
    # Start processing threads
    threads = []
    for processor in video_processors:
        thread = threading.Thread(target=processor.run_processing)
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    print(f"Started {len(video_processors)} video processing threads")

def generate_frames(lot_id):
    """Generate frames for streaming with minimal delay"""
    while True:
        try:
            with frame_locks[lot_id]:
                if frames[lot_id]:
                    frame_data = frames[lot_id]
                else:
                    # Generate a blank frame if no data available
                    blank_frame = np.zeros((480, 640, 3), np.uint8)
                    text = f"Loading Lot {lot_id}..."
                    cv2.putText(blank_frame, text, (200, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    _, buffer = cv2.imencode('.jpg', blank_frame)
                    frame_data = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            
            # Minimal delay for streaming - adjust this value to control stream speed
            # Lower values = faster stream, higher values = slower stream
            time.sleep(0.033)  # ~30 FPS streaming
            
        except Exception as e:
            print(f"Error generating frame for lot {lot_id}: {e}")
            time.sleep(0.1)

@app.route('/api/parking-data')
def get_parking_data():
    return jsonify({
        'lot1': {
            'total': parking_data['lot1']['total'],
            'available': parking_data['lot1']['available'],
            'occupied': parking_data['lot1']['occupied']
        },
        'lot2': {
            'total': parking_data['lot2']['total'],
            'available': parking_data['lot2']['available'],
            'occupied': parking_data['lot2']['occupied']
        }
    })

@app.route('/api/parking-details')
def get_parking_details():
    lot = request.args.get('lot', '1')
    lot_key = f'lot{lot}'
    
    if lot_key in parking_data:
        return jsonify({
            'total': parking_data[lot_key]['total'],
            'available': parking_data[lot_key]['available'],
            'occupied': parking_data[lot_key]['occupied'],
            'history': parking_data[lot_key]['history']
        })
    else:
        return jsonify({'error': 'Invalid lot number'}), 400

@app.route('/api/video-stream')
def video_stream():
    lot = request.args.get('lot', '1')
    if lot not in ['1', '2']:
        return "Invalid lot number", 400
    
    return Response(generate_frames(lot),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

# --- STATIC FILE SERVING FOR REACT FRONTEND ---

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    # Serve static files from static, else index.html for SPA
    static_folder = str(app.static_folder)
    file_path = os.path.join(static_folder, path)
    if path != "" and os.path.exists(file_path):
        return send_from_directory(static_folder, path)
    else:
        return send_from_directory(static_folder, 'index.html')

if __name__ == '__main__':
    print("Starting Parking Monitor Flask App...")
    
    # Setup video processing
    setup_video_processing()
    
    # Give some time for video processors to initialize
    time.sleep(2)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
