import cv2
import pickle
import numpy as np
from pathlib import Path
import os

class ParkingMonitor:
    def __init__(self):
        self.positions = []
        self.drawing = False
        self.current_idx = -1
        self.start_pos = None
        
        # Configuration
        self.SPOT_WIDTH, self.SPOT_HEIGHT = 90, 30
        self.OCCUPANCY_THRESHOLD = 0.2
        self.FRAME_SKIP = 1  # Process every frame for smooth playback
        
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        # Go up one level to backend, then to assets
        assets_dir = script_dir.parent / "assets"
        
        # File paths - using relative paths from the script location
        self.video_paths = [
            assets_dir / "video-1.mp4",
            assets_dir / "video-3.mp4"
        ]
        self.pos_files = [
            assets_dir / "coordinate-video-1",
            assets_dir / "coordinate-video-3"
        ]
        self.original_frame_sizes = {}  # To store original video dimensions as {video_idx: (width, height)}

    def load_positions(self, pos_file):
        """Load parking positions from file"""
        try:
            if pos_file.exists():
                with open(pos_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error loading positions: {e}")
        return []

    def save_positions(self, positions, pos_file):
        """Save parking positions to file"""
        try:
            # Create directory if it doesn't exist
            pos_file.parent.mkdir(parents=True, exist_ok=True)
            with open(pos_file, 'wb') as f:
                pickle.dump(positions, f)
        except Exception as e:
            print(f"Error saving positions: {e}")

    def get_config_frame(self, video_path, video_idx):
        """Get properly scaled first frame from video"""
        print(f"Trying to open video: {video_path}")
        if not video_path.exists():
            print(f"Video file does not exist: {video_path}")
            return None
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return None
        
        # Get and store original video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_frame_sizes[video_idx] = (width, height)
        print(f"Video dimensions: {width}x{height}")
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Failed to read frame from video")
            return None
            
        # Ensure we get the full frame for both videos
        frame = cv2.resize(frame, (width, height))
        
        return frame

    
    def _mouse_handler(self, event, x, y, flags, param):
        """Mouse callback for spot configuration"""
        pos_file, video_idx = param
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking existing spot
            for i, (px, py, w, h) in enumerate(self.positions):
                if px <= x <= px+w and py <= y <= py+h:
                    self.current_idx = i
                    self.drawing = True
                    self.start_pos = (x, y)
                    return
            
            # Add new spot
            self.positions.append((x, y, self.SPOT_WIDTH, self.SPOT_HEIGHT))
            self.save_positions(self.positions, pos_file)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if self.current_idx >= 0 and self.start_pos:
                dx, dy = x - self.start_pos[0], y - self.start_pos[1]
                px, py, w, h = self.positions[self.current_idx]
                new_w = max(20, w + dx)
                new_h = max(20, h + dy)
                self.positions[self.current_idx] = (px, py, new_w, new_h)
                self.start_pos = (x, y)
                self.save_positions(self.positions, pos_file)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_idx = -1
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, (px, py, w, h) in enumerate(self.positions[:]):
                if px <= x <= px+w and py <= y <= py+h:
                    self.positions.pop(i)
                    self.save_positions(self.positions, pos_file)
                    break

    def configure_spots(self, frame, pos_file, video_idx):
        """Interactive spot configuration with proper scaling"""
        original_frame = frame.copy()
        
        cv2.namedWindow("Configure Parking Spots", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Configure Parking Spots", self._mouse_handler, param=(pos_file, video_idx))
        
        print("\nConfiguration Controls:")
        print("- LEFT click: Add spot")
        print("- LEFT drag: Resize spot")
        print("- RIGHT click: Delete spot")
        print("- Press 'r' to reset")
        print("- ESC: Save and exit")
        
        while True:
            display_frame = original_frame.copy()
            for i, (x, y, w, h) in enumerate(self.positions):
                color = (0, 255, 255) if i == self.current_idx else (0, 255, 0)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display_frame, str(i+1), (x+5, y+20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            cv2.imshow("Configure Parking Spots", display_frame)
            key = cv2.waitKey(1)
            
            if key == 27:  # ESC
                self.save_positions(self.positions, pos_file)
                print(f"Saved {len(self.positions)} spots")
                break
            elif key == ord('r'):  # Reset
                self.positions = []
                print("Reset all spots")
        
        cv2.destroyWindow("Configure Parking Spots")

    def process_frame(self, frame, positions, video_idx):
        """Process frame with proper scaling"""
        if video_idx in self.original_frame_sizes:
            frame = cv2.resize(frame, self.original_frame_sizes[video_idx])
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 1)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 25, 16)
        processed = cv2.medianBlur(thresh, 5)
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.dilate(processed, kernel, iterations=1)
        
        free_count = 0
        for x, y, w, h in positions:
            spot = processed[y:y+h, x:x+w]
            nonzero = cv2.countNonZero(spot)
            
            if nonzero > (w * h * self.OCCUPANCY_THRESHOLD):
                color = (0, 0, 255)  # Red - occupied
                thickness = 2
            else:
                color = (0, 255, 0)  # Green - free
                thickness = 3
                free_count += 1
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            cv2.putText(frame, str(nonzero), (x+5, y+h-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Display status
        status = f"Free: {free_count}/{len(positions)}"
        cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 200, 0), 2, cv2.LINE_AA)
        return frame

    def monitor(self):
        """Monitoring with fast video playback"""
        caps = []
        all_positions = []
        
        # Initialize video captures
        for i, video_path in enumerate(self.video_paths):
            print(f"Loading positions for video {i+1}...")
            positions = self.load_positions(self.pos_files[i])
            if not positions:
                print(f"\nSkipping video {i+1} - no spots configured")
                caps.append(None)
                all_positions.append([])
                continue
            
            print(f"Opening video: {video_path}")
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Failed to open video {i+1}")
                caps.append(None)
            else:
                # Set to original dimensions if available
                if i in self.original_frame_sizes:
                    width, height = self.original_frame_sizes[i]
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                caps.append(cap)
                all_positions.append(positions)
                win_name = f"Parking Lot {i+1}"
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                print(f"Video {i+1} ready with {len(positions)} spots, FPS: {fps}")
        
        # Check if any videos are available
        if not any(cap is not None for cap in caps):
            print("No videos available for monitoring!")
            return
        
        print(f"\nStarting monitoring (ESC to exit)...")
        
        while True:
            for i, (cap, positions) in enumerate(zip(caps, all_positions)):
                if cap is None or not positions:
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    # Loop video when it ends
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                
                # Process frame
                processed_frame = self.process_frame(frame, positions, i)
                win_name = f"Parking Lot {i+1}"
                cv2.imshow(win_name, processed_frame)
            
            # Use a very small delay - this controls the speed
            # Lower values = faster playback, higher values = slower playback
            key = cv2.waitKey(1)  # 1ms delay for very fast playback
            if key == 27:  # ESC
                break
        
        # Cleanup
        for cap in caps:
            if cap is not None:
                cap.release()
        cv2.destroyAllWindows()
        print("Monitoring stopped")

    def run(self):
        """Main execution flow"""
        try:
            print("Parking Monitor Started")
            print(f"Script directory: {Path(__file__).parent}")
            print(f"Assets directory: {Path(__file__).parent.parent / 'assets'}")
            
            # Check if video files exist
            for i, video_path in enumerate(self.video_paths):
                if video_path.exists():
                    print(f"Video {i+1}: ✓ Found at {video_path}")
                else:
                    print(f"Video {i+1}: ✗ NOT found at {video_path}")
            
            # Configuration phase for both videos
            for i in range(len(self.video_paths)):
                print(f"\n=== Video {i+1} Configuration ===")
                
                if not self.video_paths[i].exists():
                    print(f"Video file not found: {self.video_paths[i]}")
                    continue
                
                existing_positions = self.load_positions(self.pos_files[i])
                
                if not existing_positions or input("Configure/reconfigure spots? (y/N): ").lower() == 'y':
                    frame = self.get_config_frame(self.video_paths[i], i)
                    if frame is None:
                        print("Failed to get configuration frame")
                        continue
                        
                    self.positions = existing_positions.copy()
                    self.configure_spots(frame, self.pos_files[i], i)
                else:
                    print(f"Using existing configuration with {len(existing_positions)} spots")
            
            # Monitoring phase
            self.monitor()
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = ParkingMonitor()
    monitor.run()