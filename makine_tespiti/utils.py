import numpy as np
import cv2
from collections import deque, Counter

class MotionTracker:
    def __init__(self, history_size=10):
        self.history = deque(maxlen=history_size)
        
    def add_motion(self, motion_type):
        self.history.append(motion_type)
        
    def get_dominant_motion(self):
        if not self.history:
            return "BELİRSİZ"
        return Counter(self.history).most_common(1)[0][0]

def calculate_motion_direction(motion_vectors):
    """
    Hareket vektörlerinden hareket yönünü hesaplar
    """
    if len(motion_vectors) == 0:
        return "BELİRSİZ"
        
    dxs = motion_vectors[:, 0]
    dys = motion_vectors[:, 1]
    
    mean_dx = np.mean(dxs)
    mean_dy = np.mean(dys)
    
    # Açısal varyans hesaplama
    angles = np.arctan2(dys, dxs)
    angle_variance = np.var(angles)
    
    # Minimum hareket eşiği
    min_movement = 0.5
    if abs(mean_dx) < min_movement and abs(mean_dy) < min_movement:
        return "BELİRSİZ"
    
    # Rotasyon kontrolü
    if angle_variance > 0.5:  # Yüksek açısal varyans rotasyonu gösterir
        return "DÖNÜYOR"
    
    # Lineer hareket yönü kontrolü
    if abs(mean_dx) > abs(mean_dy) * 1.5:
        return "SAĞA-SOLA"
    elif abs(mean_dy) > abs(mean_dx) * 1.5:
        return "YUKARI-AŞAĞI"
    
    return "BELİRSİZ"

def draw_motion_box(frame, points, direction, confidence=None):
    """
    Hareketli bölgeyi çerçeve içine alır ve hareket yönünü yazar
    """
    if len(points) == 0:
        return frame
    
    # Hareketli noktaların sınırlarını bul
    x_min = int(np.min(points[:, 0]))
    x_max = int(np.max(points[:, 0]))
    y_min = int(np.min(points[:, 1]))
    y_max = int(np.max(points[:, 1]))
    
    # Çerçeve çiz
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Hareket yönünü yaz
    text = f"Hareket: {direction}"
    if confidence is not None:
        text += f" ({confidence:.0%})"
    
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame 