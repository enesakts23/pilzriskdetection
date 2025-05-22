import numpy as np
import cv2
from collections import deque, Counter
from ultralytics import YOLO
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import fft, fftfreq
import numpy.linalg as LA

class MotionTracker:
    def __init__(self, model_type='optical_flow', history_size=90):  # 3 saniyelik geçmiş (30fps)
        self.model_type = model_type
        self.history = deque(maxlen=history_size)
        self.motion_history = deque(maxlen=history_size)  # Ham hareket verileri
        self.magnitude_history = deque(maxlen=history_size)  # Hareket büyüklükleri
        self.frequency_history = deque(maxlen=history_size)  # Frekans analizi
        
        # YOLO modeli
        if 'yolo' in model_type:
            try:
                model_name = {
                    'yolo11n': 'yolo11n.pt',
                    'yolo11s': 'yolo11s.pt',
                    'yolo11m': 'yolo11m.pt',
                    'yolo11l': 'yolo11l.pt',
                    'yolo11x': 'yolo11x.pt'
                }.get(model_type, 'yolo11n.pt')
                
                self.yolo = YOLO(model_name)
                print(f"{model_type} modeli başarıyla yüklendi")
            except Exception as e:
                print(f"YOLO model yükleme hatası: {e}")
                self.yolo = None
        else:
            self.yolo = None
        
        # Dense Optical Flow için parametreler
        self.dense_flow_params = dict(
            pyr_scale=0.5,
            levels=5,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Sparse Optical Flow için parametreler
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            minEigThreshold=1e-4
        )
        
        self.feature_params = dict(
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
            useHarrisDetector=True,
            k=0.04
        )
        
        # Hareket analizi parametreleri
        self.min_motion_threshold = 0.1
        self.rotation_threshold = 0.2
        self.direction_ratio = 1.1
        
        # Filtre parametreleri
        self.butterworth_filter = self._create_butterworth_filter()
        
        # Geçmiş veriler
        self.prev_gray = None
        self.prev_points = None
        self.prev_frame = None
        self.prev_flow = None
        self.motion_accumulator = np.zeros((2,))  # Hareket birikimi
        
    def _create_butterworth_filter(self):
        """Butterworth alçak geçiren filtre oluştur"""
        nyquist = 15.0  # 30fps'nin yarısı
        cutoff = 5.0    # 5Hz kesme frekansı
        order = 4
        b, a = butter(order, cutoff/nyquist, btype='low')
        return (b, a)
        
    def _apply_butterworth_filter(self, data):
        """Veriyi filtrele"""
        if len(data) < 10:
            return data
        return filtfilt(self.butterworth_filter[0], self.butterworth_filter[1], data)
        
    def _analyze_frequency(self, motion_data):
        """Hareket verilerinin frekans analizi"""
        if len(motion_data) < 30:  # En az 1 saniyelik veri
            return None, 0.0
            
        # FFT uygula
        motion_fft = fft(motion_data)
        freqs = fftfreq(len(motion_data), 1/30)  # 30fps
        
        # Pozitif frekansları al
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        amplitudes = np.abs(motion_fft)[pos_mask]
        
        # Baskın frekansları bul
        peaks, properties = find_peaks(amplitudes, height=np.mean(amplitudes))
        
        if len(peaks) == 0:
            return None, 0.0
            
        # En güçlü frekans
        dominant_idx = peaks[np.argmax(properties["peak_heights"])]
        dominant_freq = freqs[dominant_idx]
        strength = properties["peak_heights"][np.argmax(properties["peak_heights"])]
        
        return dominant_freq, strength
        
    def _detect_periodic_motion(self, motion_vectors, time_window=30):
        """Periyodik hareket tespiti"""
        if len(self.motion_history) < time_window:
            return False, 0.0, 0.0
            
        # Son time_window frame'deki hareket verilerini al
        recent_motions = np.array(list(self.motion_history)[-time_window:])
        
        # X ve Y eksenleri için ayrı analiz
        freq_x, strength_x = self._analyze_frequency(recent_motions[:, 0])
        freq_y, strength_y = self._analyze_frequency(recent_motions[:, 1])
        
        # Periyodiklik skoru
        if freq_x is not None and freq_y is not None:
            periodicity = max(strength_x, strength_y)
            frequency = freq_x if strength_x > strength_y else freq_y
            return True, frequency, periodicity
            
        return False, 0.0, 0.0
        
    def _analyze_motion_pattern(self, motion_vectors):
        """Gelişmiş hareket örüntüsü analizi"""
        if len(motion_vectors) == 0:
            return "BELİRSİZ", 0.0
            
        # Ortalama hareket vektörü
        mean_motion = np.mean(motion_vectors, axis=0)
        
        # Hareket büyüklüğü
        magnitude = np.linalg.norm(mean_motion)
        self.magnitude_history.append(magnitude)
        
        # Hareket vektörünü kaydet
        self.motion_history.append(mean_motion)
        
        # Minimum hareket kontrolü
        if magnitude < self.min_motion_threshold:
            return "BELİRSİZ", 0.0
            
        # Hareket birikimini güncelle
        self.motion_accumulator += mean_motion
        
        # Periyodik hareket analizi
        is_periodic, freq, periodicity = self._detect_periodic_motion(motion_vectors)
        
        if is_periodic and periodicity > 0.5:
            # Frekansa göre hareket tipi belirleme
            if 0.5 <= freq <= 2.0:  # Yavaş periyodik hareket
                if abs(mean_motion[1]) > abs(mean_motion[0]):
                    return "PRES", periodicity
                else:
                    return "DÖNER", periodicity
            elif 2.0 < freq <= 5.0:  # Hızlı periyodik hareket
                return "ROBOT", periodicity
                
        # Yön analizi
        dx, dy = mean_motion
        angle = np.arctan2(dy, dx)
        angles = np.arctan2(motion_vectors[:, 1], motion_vectors[:, 0])
        angle_variance = np.var(angles)
        
        # Rotasyon analizi
        if angle_variance > self.rotation_threshold:
            return "DÖNÜYOR", angle_variance
            
        # Yön analizi
        if abs(dx) > abs(dy) * self.direction_ratio:
            return "SAĞA-SOLA", abs(dx/dy) if dy != 0 else abs(dx)
        elif abs(dy) > abs(dx) * self.direction_ratio:
            return "YUKARI-AŞAĞI", abs(dy/dx) if dx != 0 else abs(dy)
            
        return "KARIŞIK", magnitude
        
    def _detect_with_multiple_methods(self, frame):
        """Birden fazla yöntemle hareket tespiti"""
        if self.prev_frame is None:
            self.prev_frame = frame
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return None
            
        # Gri tonlamalı görüntü
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Dense Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, **self.dense_flow_params
        )
        
        # 2. Sparse Optical Flow
        if self.prev_points is None:
            self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, **self.feature_params)
            
        if self.prev_points is not None:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None, **self.lk_params
            )
            
            if next_points is not None:
                # İyi noktaları seç
                good_new = next_points[status == 1]
                good_old = self.prev_points[status == 1]
                
                if len(good_new) >= 3:
                    # Sparse motion vectors
                    sparse_vectors = good_new - good_old
                    
                    # Dense flow'dan hareket vektörleri
                    flow_vectors = []
                    for pt in good_old:
                        x, y = map(int, pt.ravel())
                        if 0 <= y < flow.shape[0] and 0 <= x < flow.shape[1]:
                            flow_vectors.append(flow[y, x])
                            
                    # Her iki yöntemin sonuçlarını birleştir
                    combined_vectors = np.concatenate([sparse_vectors, flow_vectors])
                    
                    # Hareket analizi
                    motion_type, confidence = self._analyze_motion_pattern(combined_vectors)
                    
                    # Sınırlayıcı kutu hesapla
                    x_min, y_min = np.min(good_new, axis=0)
                    x_max, y_max = np.max(good_new, axis=0)
                    
                    detection = [{
                        'box': (int(x_min), int(y_min), int(x_max), int(y_max)),
                        'confidence': confidence,
                        'movement': motion_type,
                        'points': good_new,
                        'motion_vectors': combined_vectors,
                        'magnitude': np.linalg.norm(np.mean(combined_vectors, axis=0))
                    }]
                    
                    # Geçmişi güncelle
                    self.history.append(motion_type)
                    self.prev_points = good_new.reshape(-1, 1, 2)
                    self.prev_gray = gray
                    self.prev_frame = frame
                    self.prev_flow = flow
                    
                    return detection
                    
        # Güncelleme
        self.prev_gray = gray
        self.prev_frame = frame
        self.prev_points = cv2.goodFeaturesToTrack(gray, **self.feature_params)
        return None
        
    def detect_motion(self, frame):
        """Ana hareket tespit fonksiyonu"""
        if self.model_type == 'optical_flow':
            return self._detect_with_multiple_methods(frame)
        else:
            return self._detect_with_yolo(frame)
            
    def get_dominant_motion(self):
        """Baskın hareket tipini belirle"""
        if not self.history:
            return "BELİRSİZ"
            
        # Son 30 frame'deki hareket tiplerini analiz et
        recent_motions = list(self.history)[-30:]
        motion_counts = Counter(recent_motions)
        
        # En yaygın hareket tipi
        return motion_counts.most_common(1)[0][0]
    
    def _compensate_camera_motion(self, prev_frame, curr_frame):
        """Kamera hareketini telafi et"""
        if prev_frame is None:
            return None, None
            
        # Gri tonlamaya çevir
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Arka plan noktalarını bul
        prev_points = cv2.goodFeaturesToTrack(prev_gray, **self.feature_params)
        if prev_points is None:
            return None, None
            
        # Optik akış hesapla
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points, None, **self.lk_params
        )
        
        if curr_points is None:
            return None, None
            
        # İyi noktaları seç
        good_prev = prev_points[status == 1]
        good_curr = curr_points[status == 1]
        
        if len(good_prev) < 4:  # Homografi için minimum 4 nokta gerekli
            return None, None
            
        # Homografi matrisini hesapla
        H, mask = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)
        
        if H is None:
            return None, None
            
        # Kamera hareketini telafi et
        h, w = curr_frame.shape[:2]
        aligned_frame = cv2.warpPerspective(curr_frame, np.linalg.inv(H), (w, h))
        
        return aligned_frame, H
    
    def _detect_with_yolo(self, frame):
        if self.yolo is None:
            return None
            
        # Kamera hareketini telafi et
        if self.prev_frame is not None:
            aligned_frame, H = self._compensate_camera_motion(self.prev_frame, frame)
            if aligned_frame is not None:
                frame = aligned_frame
        
        results = self.yolo(frame, conf=0.2, classes=None)
        if not results or len(results) == 0:
            self.prev_frame = frame.copy()
            return None
            
        result = results[0]
        detections = []
        
        for box in result.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # ROI içinde hareket analizi
            roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            
            # Önceki noktalar varsa, onları ROI'ye göre filtrele
            if self.prev_points is not None:
                mask = ((self.prev_points[:, 0] >= x1) & 
                       (self.prev_points[:, 0] <= x2) & 
                       (self.prev_points[:, 1] >= y1) & 
                       (self.prev_points[:, 1] <= y2))
                prev_roi_points = self.prev_points[mask]
            else:
                prev_roi_points = None
            
            # Yeni noktaları bul
            points = cv2.goodFeaturesToTrack(roi_gray, **self.feature_params)
            
            if points is not None:
                points = points.reshape(-1, 2) + [x1, y1]
                
                if prev_roi_points is not None and len(prev_roi_points) > 0:
                    # Minimum nokta sayısını al
                    min_points = min(len(points), len(prev_roi_points))
                    if min_points >= 3:  # En az 3 nokta gerekli
                        points = points[:min_points]
                        prev_roi_points = prev_roi_points[:min_points]
                        
                        # Hareket vektörlerini hesapla
                        motion_vectors = points - prev_roi_points
                        
                        # Kamera hareketini çıkar
                        if H is not None:
                            # Noktaları homografi ile dönüştür
                            transformed_prev = cv2.perspectiveTransform(
                                prev_roi_points.reshape(-1, 1, 2), H
                            ).reshape(-1, 2)
                            motion_vectors = points - transformed_prev
                        
                        # Hareket büyüklüğünü kontrol et
                        motion_magnitude = np.linalg.norm(motion_vectors, axis=1).mean()
                        if motion_magnitude > 0.3:
                            direction = calculate_motion_direction(motion_vectors)
                            self.history.append(direction)
                            
                            detections.append({
                                'box': (x1, y1, x2, y2),
                                'confidence': conf,
                                'movement': direction,
                                'points': points,
                                'motion_vectors': motion_vectors,
                                'magnitude': motion_magnitude
                            })
                
                self.prev_points = points
        
        self.prev_frame = frame.copy()
        return detections

def calculate_motion_direction(motion_vectors):
    if len(motion_vectors) == 0:
        return "BELİRSİZ"
    
    # Ortalama hareket vektörü
    mean_motion = np.mean(motion_vectors, axis=0)
    dx, dy = mean_motion
    
    # Açısal hareket analizi
    angles = np.arctan2(motion_vectors[:, 1], motion_vectors[:, 0])
    angle_variance = np.var(angles)
    
    # Minimum hareket eşiği - daha hassas
    if np.sqrt(dx*dx + dy*dy) < 0.3:  # Daha düşük hareket eşiği
        return "BELİRSİZ"
    
    # Rotasyon kontrolü - daha hassas
    if angle_variance > 0.3:  # Daha düşük rotasyon eşiği
        return "DÖNÜYOR"
    
    # Yön kontrolü - daha hassas
    if abs(dx) > abs(dy) * 1.2:  # Daha düşük oran
        return "SAĞA-SOLA"
    elif abs(dy) > abs(dx) * 1.2:  # Daha düşük oran
        return "YUKARI-AŞAĞI"
    
    return "KARIŞIK"

def draw_motion_box(frame, detections):
    if not detections:
        return frame
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        movement = det['movement']
        confidence = det['confidence']
        magnitude = det.get('magnitude', 1.0)
        
        # Hareket yoğunluğuna göre renk
        intensity = min(magnitude * 255, 255)
        color = (0, int(intensity), int(255-intensity))
        
        # İnce çerçeve
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        
        # Hareket noktaları ve vektörleri
        if 'points' in det and 'motion_vectors' in det:
            points = det['points']
            vectors = det['motion_vectors']
            for pt, vec in zip(points, vectors):
                # Nokta
                cv2.circle(frame, tuple(map(int, pt)), 2, color, -1)
                # Hareket vektörü
                end_pt = tuple(map(int, pt + vec * 3))  # Vektörü görselleştirmek için ölçekle
                cv2.arrowedLine(frame, tuple(map(int, pt)), end_pt, color, 1)
        
        # Hareket etiketi
        label = f"{movement} ({confidence:.2f})"
        cv2.putText(frame, label, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame 