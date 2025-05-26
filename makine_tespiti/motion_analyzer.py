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
        
        # Increase tracking history with no limits
        self.positions = deque()  # No maxlen limit
        self.velocities = deque()  # No maxlen limit
        self.filtered_velocities = deque()  # No maxlen limit
        
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
        self.positions = deque()
        self.velocities = deque()
        self.directions = deque()
        self.pattern_types = set()  # Tespit edilen tüm hareket tipleri
        self.pattern_confidences = {}  # Her hareket tipi için güven değeri
        
        # Hareket analizi için parametreler
        self.min_rotation_speed = 0.5  # Minimum dönme hızı (radyan/frame)
        self.min_press_distance = 10.0  # Minimum press hareketi mesafesi (piksel)
        self.camera_motion_threshold = 0.7  # Kamera hareketi filtreleme eşiği
        
    def add_measurement(self, pos, vel):
        self.positions.append(pos)
        self.velocities.append(vel)
        
        if len(vel) >= 2:
            angle = np.arctan2(vel[1], vel[0])
            self.directions.append(angle)
            
    def _detect_camera_motion(self, flow_vectors):
        """Kamera hareketini tespit et"""
        if len(flow_vectors) < 10:
            return False
            
        mean_vector = np.mean(flow_vectors, axis=0)
        mean_magnitude = np.linalg.norm(mean_vector)
        
        vector_angles = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0])
        angle_std = np.std(vector_angles)
        
        return angle_std < 0.5 and mean_magnitude > self.camera_motion_threshold
        
    def _analyze_local_motion(self, positions, velocities):
        """Yerel hareket analizi - birden fazla hareket tipi tespit edebilir"""
        if len(positions) < 10:
            return {}
            
        detected_patterns = {}
        
        # Pozisyon değişimlerini analiz et
        pos_diff = np.diff(positions, axis=0)
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        
        # ROTATING hareketi tespiti
        if len(self.directions) >= 5:
            recent_angles = np.array(list(self.directions)[-5:])
            angle_diff = np.diff(recent_angles)
            angle_diff = np.abs(np.unwrap(angle_diff))
            
            total_angle_change = np.sum(angle_diff)
            mean_angular_velocity = total_angle_change / len(angle_diff)
            
            if (mean_angular_velocity > self.min_rotation_speed and
                np.all(angle_diff > 0.1) and
                np.std(pos_diff) < np.mean(vel_magnitudes)):
                detected_patterns["ROTATING"] = min(mean_angular_velocity / np.pi, 1.0)
                
        # PRESS hareketi tespiti
        vertical_movement = np.ptp(positions[:, 1])
        horizontal_movement = np.ptp(positions[:, 0])
        
        if vertical_movement > self.min_press_distance:
            y_diff = np.diff(positions[:, 1])
            direction_changes = np.where(np.diff(np.signbit(y_diff)))[0]
            
            if (len(direction_changes) >= 2 and
                vertical_movement > horizontal_movement * 1.5):
                detected_patterns["PRESS"] = min(vertical_movement / 100.0, 1.0)
                
        return detected_patterns
        
    def analyze_pattern(self):
        if len(self.positions) < 10:
            return "BELİRSİZ", 0.0
            
        positions = np.array(list(self.positions))
        velocities = np.array(list(self.velocities))
        
        # Kamera hareketi kontrolü
        if self._detect_camera_motion(velocities):
            return "BELİRSİZ", 0.0
            
        # Hareket bölgelerini bul
        motion_regions = []
        current_region = []
        
        for i, vel in enumerate(velocities):
            if np.linalg.norm(vel) > 1.0:
                current_region.append(i)
            elif len(current_region) > 0:
                motion_regions.append(current_region)
                current_region = []
                
        if len(current_region) > 0:
            motion_regions.append(current_region)
            
        # Her bölgedeki hareketleri analiz et
        all_patterns = {}
        
        for region in motion_regions:
            if len(region) < 5:
                continue
                
            region_positions = positions[region]
            region_velocities = velocities[region]
            
            # Bölgedeki tüm hareket tiplerini al
            patterns = self._analyze_local_motion(region_positions, region_velocities)
            
            # Her hareket tipini güncelle
            for pattern, confidence in patterns.items():
                if pattern not in all_patterns or confidence > all_patterns[pattern]:
                    all_patterns[pattern] = confidence
                    
        # Sonuçları birleştir
        if not all_patterns:
            return "BELİRSİZ", 0.0
            
        # Tespit edilen tüm hareket tiplerini kaydet
        self.pattern_types = set(all_patterns.keys())
        self.pattern_confidences = all_patterns
        
        # En yüksek güvenilirliğe sahip hareketi döndür
        best_pattern = max(all_patterns.items(), key=lambda x: x[1])
        
        # Eğer birden fazla hareket varsa, birleştir
        if len(all_patterns) > 1:
            patterns = "+".join(sorted(all_patterns.keys()))
            confidence = np.mean(list(all_patterns.values()))
            return patterns, confidence
            
        return best_pattern

    def get_all_patterns(self):
        """Tespit edilen tüm hareket tiplerini ve güven değerlerini döndür"""
        return self.pattern_types, self.pattern_confidences

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
        # GPU kullanılabilirlik kontrolü
        self.use_gpu = False
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.use_gpu = True
                print("GPU detected! Using CUDA acceleration.")
                # GPU stream oluştur
                self.gpu_stream = cv2.cuda.Stream()
                # GPU için gerekli nesneleri oluştur
                self.gpu_bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(
                    history=500,
                    varThreshold=16,
                    detectShadows=False
                )
            else:
                print("No GPU detected. Using CPU.")
        except Exception as e:
            print(f"Error initializing GPU: {e}. Using CPU.")

        # Unlimited frame queue size for any video length
        self.frame_queue = Queue()  # No maxsize limit
        self.processing = False
        self.current_frame = None
        self.prev_frame = None
        
        # Increase motion history size significantly
        self.history = deque()  # No maxlen limit
        self.motion_vectors = deque()  # No maxlen limit
        
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
        
        # Motion detection states and thresholds
        self.current_motion = "BELİRSİZ"
        self.motion_start_time = None
        self.motion_frames = []
        self.motion_confidence_threshold = 0.6
        self.motion_duration_threshold = 15  # Minimum frame count for a valid motion
        self.stable_motion_count = 0
        self.last_saved_motion = None
        
    def _preprocess_frame(self, frame):
        """Frame preprocessing"""
        if self.use_gpu:
            # Frame'i GPU'ya yükle
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # GPU üzerinde blur işlemi
            gpu_blurred = cv2.cuda.GaussianBlur(gpu_frame, (5, 5), 0)
            
            # Sonucu CPU'ya geri al
            blurred = gpu_blurred.download()
            return blurred
        else:
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            return blurred
        
    def _detect_background_motion(self, frame):
        """Background subtraction based motion detection"""
        if self.use_gpu:
            # Frame'i GPU'ya yükle
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # GPU üzerinde arkaplan çıkarma
            gpu_motion_mask = self.gpu_bg_subtractor.apply(gpu_frame, learningRate=-1)
            
            # Maske'yi CPU'ya al
            motion_mask = gpu_motion_mask.download()
            
            # Morfolojik işlemler için GPU
            gpu_mask = cv2.cuda_GpuMat()
            gpu_mask.upload(motion_mask)
            
            # Erosion ve dilation işlemleri
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            gpu_eroded = cv2.cuda.erode(gpu_mask, kernel)
            gpu_dilated = cv2.cuda.dilate(gpu_eroded, kernel, iterations=2)
            
            # Son maske'yi CPU'ya al
            motion_mask = gpu_dilated.download()
        else:
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
            if self.use_gpu:
                self.gpu_prev_gray = cv2.cuda_GpuMat()
                self.gpu_prev_gray.upload(gray)
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

        if self.use_gpu:
            # GPU'ya yükle
            gpu_gray = cv2.cuda_GpuMat()
            gpu_gray.upload(gray)
            
            # GPU üzerinde optik akış hesapla
            gpu_flow = cv2.cuda_SparsePyrLKOpticalFlow.create(
                winSize=(21, 21),
                maxLevel=4,
                iters=10
            )
            
            gpu_prev_points = cv2.cuda_GpuMat()
            gpu_prev_points.upload(self.prev_points)
            
            gpu_next_points = cv2.cuda_GpuMat()
            gpu_status = cv2.cuda_GpuMat()
            
            gpu_flow.calc(self.gpu_prev_gray, gpu_gray, gpu_prev_points, 
                         gpu_next_points, gpu_status, self.gpu_stream)
            
            # CPU'ya geri al
            next_points = gpu_next_points.download()
            status = gpu_status.download().flatten()
            
            # Dense optical flow için
            gpu_flow_dense = cv2.cuda.FarnebackOpticalFlow.create(
                numLevels=5,
                pyrScale=0.5,
                fastPyramids=False,
                winSize=15,
                numIters=3,
                polyN=5,
                polySigma=1.2,
                flags=0
            )
            flow = gpu_flow_dense.calc(self.gpu_prev_gray, gpu_gray, None).download()
            
            # GPU belleği güncelle
            self.gpu_prev_gray = gpu_gray
        else:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray,
                self.prev_points, None,
                **self.lk_params
            )
            
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray,
                None, **self.dense_flow_params
            )
        
        if next_points is None:
            return None, None
            
        good_new = next_points[status == 1]
        good_old = self.prev_points[status == 1]
        
        if len(good_new) < 3:
            return None, None
            
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
                # If this is a new motion pattern
                if self.current_motion != direction:
                    # If we were tracking a previous motion, save it if valid
                    if self.current_motion != "BELİRSİZ" and len(self.motion_frames) >= self.motion_duration_threshold:
                        self._save_motion_sequence()
                    
                    # Start tracking new motion
                    self.current_motion = direction
                    self.motion_start_time = time.time()
                    self.motion_frames = [frame.copy()]
                    self.stable_motion_count = 1
                else:
                    # Continue tracking current motion
                    self.stable_motion_count += 1
                    if len(self.motion_frames) < 100:  # Limit stored frames to prevent memory issues
                        self.motion_frames.append(frame.copy())
            else:
                # If uncertain motion is detected and we were tracking a motion
                if self.current_motion != "BELİRSİZ" and len(self.motion_frames) >= self.motion_duration_threshold:
                    self._save_motion_sequence()
                
                self.current_motion = "BELİRSİZ"
                self.motion_frames = []
                self.stable_motion_count = 0
            
            self.history.append(direction)
            
        self.motion_mask = motion_mask
        self.prev_frame = frame
        self.current_frame = frame
        
        return motion_info
        
    def _save_motion_sequence(self):
        """Save the detected motion sequence"""
        if not self.motion_frames or self.current_motion == "BELİRSİZ":
            return
            
        # Prevent saving the same motion pattern repeatedly
        current_time = time.time()
        if (self.last_saved_motion is not None and 
            self.last_saved_motion[0] == self.current_motion and 
            current_time - self.last_saved_motion[1] < 5):  # 5 second cooldown
            return
            
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Select representative frame (middle frame of sequence)
        mid_idx = len(self.motion_frames) // 2
        representative_frame = self.motion_frames[mid_idx].copy()
        
        # Add text overlay with motion type
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Hareket: {self.current_motion}"
        
        # Get text size
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        
        # Calculate text position
        text_x = 10
        text_y = text_size[1] + 20
        
        # Draw black background for text
        cv2.rectangle(representative_frame, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (0, 0, 0),
                     -1)
        
        # Draw text
        cv2.putText(representative_frame,
                   text,
                   (text_x, text_y),
                   font,
                   1,
                   (0, 255, 0),
                   2)
        
        # Save the frame
        filename = os.path.join(self.save_dir, f"motion_{self.current_motion}_{timestamp}.jpg")
        cv2.imwrite(filename, representative_frame)
        
        # Update last saved motion
        self.last_saved_motion = (self.current_motion, current_time)
        
        # Clear motion frames
        self.motion_frames = []
        
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
            
        # GPU kaynaklarını temizle
        if self.use_gpu:
            try:
                if hasattr(self, 'gpu_stream'):
                    self.gpu_stream.free()
                if hasattr(self, 'gpu_prev_gray'):
                    self.gpu_prev_gray.release()
                cv2.cuda.deviceReset()
                print("GPU resources cleaned up successfully.")
            except Exception as e:
                print(f"Error cleaning up GPU resources: {e}")
            
    def add_frame(self, frame):
        """Add new frame"""
        self.frame_queue.put(frame)
            
    def _process_frames(self):
        """Frame processing loop"""
        while self.processing:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                self._analyze_frame(frame)
                time.sleep(0.001)  # CPU yükünü azalt 