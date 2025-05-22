import cv2
import numpy as np
from collections import deque, Counter
import threading
from queue import Queue
import time
import os
from datetime import datetime
from filterpy.kalman import KalmanFilter
from scipy.signal import savgol_filter
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

class KalmanTracker:
    def __init__(self, x, y):
        # Kalman filtresi (konum ve hız takibi için)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # Durum geçiş matrisi
        self.kf.F = np.array([
            [1, 0, 1, 0],  # x = x + dx
            [0, 1, 0, 1],  # y = y + dy
            [0, 0, 1, 0],  # dx = dx
            [0, 0, 0, 1]   # dy = dy
        ])
        
        # Ölçüm fonksiyonu
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Ölçüm gürültüsü
        self.kf.R *= 10
        
        # Süreç gürültüsü
        self.kf.Q = np.eye(4) * 0.1
        
        # İlk durum
        self.kf.x = np.array([[x], [y], [0], [0]])
        
        # Hareket geçmişi
        self.positions = deque(maxlen=30)
        self.velocities = deque(maxlen=30)
        self.filtered_velocities = deque(maxlen=30)
        
    def update(self, x, y):
        # Ölçümü güncelle
        self.kf.predict()
        self.kf.update(np.array([[x], [y]]))
        
        # Pozisyon ve hız geçmişini güncelle
        pos = self.kf.x[:2].flatten()
        vel = self.kf.x[2:].flatten()
        
        self.positions.append(pos)
        self.velocities.append(vel)
        
        # Hız verilerini filtrele
        if len(self.velocities) >= 5:
            velocities_array = np.array(list(self.velocities))
            filtered = savgol_filter(velocities_array, 5, 2, axis=0)
            self.filtered_velocities.append(filtered[-1])
            return pos, filtered[-1]
        
        return pos, vel

class MotionPattern:
    def __init__(self):
        self.positions = deque(maxlen=60)  # 2 saniyelik veri (30fps)
        self.velocities = deque(maxlen=60)
        self.directions = deque(maxlen=60)
        self.pattern_type = "BELİRSİZ"
        self.confidence = 0.0
        
    def add_measurement(self, pos, vel):
        self.positions.append(pos)
        self.velocities.append(vel)
        
        if len(vel) >= 2:
            angle = np.arctan2(vel[1], vel[0])
            self.directions.append(angle)
            
    def analyze_pattern(self):
        if len(self.positions) < 30:  # En az 1 saniyelik veri
            return "BELİRSİZ", 0.0
            
        # Pozisyon analizi
        positions = np.array(list(self.positions))
        velocities = np.array(list(self.velocities))
        
        # FFT ile frekans analizi
        fft_x = np.fft.fft(positions[:, 0])
        fft_y = np.fft.fft(positions[:, 1])
        freqs = np.fft.fftfreq(len(positions))
        
        # Baskın frekansları bul
        main_freq_x = abs(freqs[np.argmax(np.abs(fft_x[1:]))+1])
        main_freq_y = abs(freqs[np.argmax(np.abs(fft_y[1:]))+1])
        
        # Hareket yönü analizi
        mean_vel = np.mean(velocities, axis=0)
        vel_magnitude = np.linalg.norm(mean_vel)
        
        # Yön değişimi analizi
        if len(self.directions) >= 2:
            direction_changes = np.diff(list(self.directions))
            direction_changes = np.abs(np.unwrap(direction_changes))
            
            # Periyodik hareket analizi
            is_periodic = np.any(direction_changes > np.pi/2)
            periodicity = np.std(direction_changes)
        else:
            is_periodic = False
            periodicity = 0
            
        # Hareket tipi belirleme
        if is_periodic and periodicity > 1.5:
            # Dönme hareketi
            pattern = "DÖNÜYOR"
            conf = min(periodicity / 3.0, 1.0)
            
        elif main_freq_y > 0.1 and main_freq_y > main_freq_x * 2:
            # Yukarı-aşağı hareket
            y_amplitude = np.ptp(positions[:, 1])
            if y_amplitude > 20:  # Minimum genlik kontrolü
                pattern = "YUKARI-AŞAĞI"
                conf = min(y_amplitude / 100.0, 1.0)
            else:
                pattern = "BELİRSİZ"
                conf = 0.0
                
        elif main_freq_x > 0.1 and main_freq_x > main_freq_y * 2:
            # Sağa-sola hareket
            x_amplitude = np.ptp(positions[:, 0])
            if x_amplitude > 20:
                pattern = "SAĞA-SOLA"
                conf = min(x_amplitude / 100.0, 1.0)
            else:
                pattern = "BELİRSİZ"
                conf = 0.0
                
        elif vel_magnitude > 1.0:
            # Karışık hareket
            pattern = "KARIŞIK"
            conf = min(vel_magnitude / 5.0, 1.0)
            
        else:
            pattern = "BELİRSİZ"
            conf = 0.0
            
        self.pattern_type = pattern
        self.confidence = conf
        return pattern, conf

class MotionRegion:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.tracker = KalmanTracker(x + w/2, y + h/2)
        self.pattern = MotionPattern()
        self.active = True
        self.age = 0
        self.last_update = time.time()
        self.points = []
        self.area = w * h
        
    def update(self, x, y, w, h, points):
        current_time = time.time()
        
        # Bölge güncelleme
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w * h
        self.points = points
        self.age += 1
        
        # Merkez noktayı güncelle
        center_x = x + w/2
        center_y = y + h/2
        
        # Kalman filtresi ile takip
        pos, vel = self.tracker.update(center_x, center_y)
        
        # Hareket örüntüsünü güncelle
        self.pattern.add_measurement(pos, vel)
        
        # Hareket analizi
        pattern_type, confidence = self.pattern.analyze_pattern()
        
        self.last_update = current_time
        return pattern_type, confidence

class MotionAnalyzer:
    def __init__(self, save_dir="detected_motions"):
        self.frame_queue = Queue(maxsize=300)
        self.processing = False
        self.current_frame = None
        self.prev_frame = None
        
        # Motion history
        self.history = deque(maxlen=90)  # 3 seconds history (30fps)
        self.motion_vectors = deque(maxlen=90)
        
        # Save directory for detected motions
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
            useHarrisDetector=True,
            k=0.04
        )
        
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            minEigThreshold=1e-4
        )
        
        # Dense optical flow parameters
        self.dense_flow_params = dict(
            pyr_scale=0.5,
            levels=5,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Motion analysis parameters
        self.min_area = 500
        self.min_motion_threshold = 2.0
        self.direction_threshold = 5.0
        
        # Temporary variables
        self.prev_gray = None
        self.prev_points = None
        self.motion_mask = None
        self.machine_type = "UNKNOWN"
        
        # New variables for motion tracking
        self.last_known_motion = "BELİRSİZ"
        self.motion_counts = Counter()
        self.most_frequent_motion = None
        self.most_frequent_motion_frame = None
        
    def _preprocess_frame(self, frame):
        """Frame preprocessing"""
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        return blurred
        
    def _detect_background_motion(self, frame):
        """Background subtraction based motion detection"""
        motion_mask = self.bg_subtractor.apply(frame)
        motion_mask = cv2.erode(motion_mask, None, iterations=1)
        motion_mask = cv2.dilate(motion_mask, None, iterations=2)
        
        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        motion_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                motion_regions.append({
                    'box': (x, y, x+w, y+h),
                    'area': area,
                    'contour': cnt
                })
        
        return motion_regions, motion_mask
        
    def _detect_optical_flow(self, frame, gray):
        """Optical flow based motion detection"""
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.feature_params
            )
            return None, None
            
        if self.prev_points is None:
            self.prev_points = cv2.goodFeaturesToTrack(
                self.prev_gray, mask=None, **self.feature_params
            )
            if self.prev_points is None:
                return None, None
                
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray,
            self.prev_points, None,
            **self.lk_params
        )
        
        if next_points is None:
            return None, None
            
        good_new = next_points[status == 1]
        good_old = self.prev_points[status == 1]
        
        if len(good_new) < 3:
            return None, None
            
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None, **self.dense_flow_params
        )
        
        motion_vectors = good_new - good_old
        mean_motion = np.mean(motion_vectors, axis=0)
        magnitude = np.linalg.norm(mean_motion)
        
        if magnitude < self.min_motion_threshold:
            return None, None
            
        direction = self._analyze_motion_direction(motion_vectors, flow)
        
        x_min, y_min = np.min(good_new, axis=0)
        x_max, y_max = np.max(good_new, axis=0)
        
        flow_info = {
            'box': (int(x_min), int(y_min), int(x_max), int(y_max)),
            'points': good_new,
            'vectors': motion_vectors,
            'flow': flow,
            'magnitude': magnitude,
            'direction': direction
        }
        
        self.prev_gray = gray
        self.prev_points = good_new.reshape(-1, 1, 2)
        
        return flow_info, motion_vectors
        
    def _analyze_motion_direction(self, motion_vectors, flow):
        """Motion direction analysis"""
        if len(motion_vectors) == 0:
            return "UNCERTAIN"
            
        mean_motion = np.mean(motion_vectors, axis=0)
        dx, dy = mean_motion
        
        flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_magnitude = np.mean(flow_magnitude)
        
        angles = np.arctan2(motion_vectors[:, 1], motion_vectors[:, 0])
        angle_variance = np.var(angles)
        
        if len(self.motion_vectors) >= 30:
            recent_vectors = np.array(list(self.motion_vectors)[-30:])
            freq_x = np.fft.fft(recent_vectors[:, 0])
            freq_y = np.fft.fft(recent_vectors[:, 1])
            main_freq_x = np.abs(freq_x[1:]).max()
            main_freq_y = np.abs(freq_y[1:]).max()
            
            if main_freq_x > 5 or main_freq_y > 5:
                if abs(dy) > abs(dx):
                    return "PRESS"
                else:
                    return "ROTARY"
                    
        if angle_variance > 0.5 and mean_magnitude > 2.0:
            return "ROTATING"
            
        if abs(dx) > self.direction_threshold:
            if abs(dx) > abs(dy) * 1.5:
                return "LEFT-RIGHT"
        elif abs(dy) > self.direction_threshold:
            if abs(dy) > abs(dx) * 1.5:
                return "UP-DOWN"
                
        if mean_magnitude > 2.0:
            return "MIXED"
            
        return "UNCERTAIN"
        
    def _analyze_frame(self, frame):
        """Frame analysis"""
        if self.prev_frame is None:
            self.prev_frame = frame
            return
            
        processed = self._preprocess_frame(frame)
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        motion_regions, motion_mask = self._detect_background_motion(processed)
        flow_info, motion_vectors = self._detect_optical_flow(processed, gray)
        
        if motion_vectors is not None:
            self.motion_vectors.append(motion_vectors.mean(axis=0))
            
        motion_info = None
        if flow_info is not None:
            motion_info = flow_info
            if motion_regions:
                largest_region = max(motion_regions, key=lambda x: x['area'])
                motion_info['bg_box'] = largest_region['box']
                motion_info['bg_area'] = largest_region['area']
                
        if motion_info is not None:
            direction = motion_info['direction']
            
            # Update motion tracking
            if direction != "UNCERTAIN" and direction != "UNKNOWN":
                self.last_known_motion = direction
                self.motion_counts[direction] += 1
                
                # Update most frequent motion
                if self.motion_counts[direction] > self.motion_counts.get(self.most_frequent_motion, 0):
                    self.most_frequent_motion = direction
                    # Save frame for most frequent motion
                    self.most_frequent_motion_frame = frame.copy()
                    # Save the image
                    self._save_most_frequent_motion()
            
            self.history.append(direction)
            
        self.motion_mask = motion_mask
        self.prev_frame = frame
        self.current_frame = frame
        
        return motion_info
        
    def _save_most_frequent_motion(self):
        """Save the frame with the most frequent motion and highlight significant moving regions"""
        if self.most_frequent_motion and self.most_frequent_motion_frame is not None:
            # Create a copy of the frame
            save_frame = self.most_frequent_motion_frame.copy()
            
            # Process the frame to detect motion
            processed = self._preprocess_frame(save_frame)
            current_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            
            # If prev_gray is not set, initialize it
            if self.prev_gray is None:
                self.prev_gray = current_gray
                return
            
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, 
                current_gray,
                None, 
                **self.dense_flow_params
            )
            
            # Calculate flow magnitude
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Normalize magnitude to 0-255 range
            magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Apply threshold to get significant motion areas
            _, magnitude_mask = cv2.threshold(magnitude_norm, 30, 255, cv2.THRESH_BINARY)
            
            # Create motion mask if not available
            if self.motion_mask is None:
                self.motion_mask = self.bg_subtractor.apply(processed)
            
            # Combine with motion mask for better accuracy
            motion_mask = self.motion_mask.copy()
            combined_mask = cv2.bitwise_and(motion_mask, magnitude_mask)
            
            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in the combined mask
            contours, _ = cv2.findContours(
                combined_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Create an empty mask for significant motion
            significant_motion_mask = np.zeros_like(combined_mask)
            
            # Filter and process contours
            if contours:
                # Calculate average magnitude for each contour
                contour_info = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > self.min_area:
                        # Create mask for this contour
                        contour_mask = np.zeros_like(combined_mask)
                        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                        
                        # Calculate average magnitude in this contour
                        mean_magnitude = cv2.mean(magnitude, mask=contour_mask)[0]
                        contour_info.append((contour, area, mean_magnitude))
                
                if contour_info:
                    # Sort contours by magnitude
                    contour_info.sort(key=lambda x: x[2], reverse=True)
                    
                    # Take the top 3 contours with highest magnitude
                    for contour, area, _ in contour_info[:3]:
                        cv2.drawContours(significant_motion_mask, [contour], -1, 255, -1)
            
            # Create a colored overlay for the significant moving regions
            overlay = save_frame.copy()
            overlay[significant_motion_mask > 0] = [0, 255, 0]  # Green color for moving parts
            
            # Blend the original frame with the overlay
            alpha = 0.4  # Transparency for visibility
            save_frame = cv2.addWeighted(overlay, alpha, save_frame, 1 - alpha, 0)
            
            # Draw motion type text with background for better visibility
            text = f"Motion Type: {self.most_frequent_motion}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            color = (0, 255, 0)  # Green color
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Calculate text position (top-left corner)
            x = 10
            y = 40
            
            # Draw black background rectangle for text
            cv2.rectangle(save_frame, 
                        (x - 5, y - text_height - 5),
                        (x + text_width + 5, y + 5),
                        (0, 0, 0),
                        -1)
            
            # Draw text
            cv2.putText(save_frame,
                       text,
                       (x, y),
                       font,
                       font_scale,
                       color,
                       thickness)
            
            # Update prev_gray for next frame
            self.prev_gray = current_gray
            
            # Save the modified frame
            filename = os.path.join(self.save_dir, "most_frequent_motion.jpg")
            cv2.imwrite(filename, save_frame)
        
    def get_visualization(self):
        """Visualization"""
        if self.current_frame is None:
            return None
            
        vis = self.current_frame.copy()
        
        if len(self.history) > 0:
            current_motion = self.history[-1]
            display_motion = current_motion
            
            # If motion is unknown/uncertain, use last known motion
            if current_motion in ["UNCERTAIN", "UNKNOWN"]:
                display_motion = self.last_known_motion
            
            cv2.putText(
                vis,
                f"Motion: {display_motion}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
            
        return vis
        
    def start_processing(self):
        """Start video processing"""
        self.processing = True
        self.process_thread = threading.Thread(target=self._process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()
        
    def stop_processing(self):
        """Stop video processing"""
        self.processing = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
            
    def add_frame(self, frame):
        """Add new frame"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
            
    def _process_frames(self):
        """Frame processing loop"""
        while self.processing:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                self._analyze_frame(frame)
                time.sleep(0.001)  # CPU yükünü azalt 