import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from utils import calculate_motion_direction, draw_motion_box, MotionTracker
from collections import Counter

class VideoAnalyzerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Makine Hareketi Tespit Sistemi")
        self.root.geometry("500x300")
        
        # Ana frame
        self.main_frame = tk.Frame(self.root, padx=20, pady=20)
        self.main_frame.pack(expand=True, fill='both')
        
        # Başlık
        title_label = tk.Label(
            self.main_frame, 
            text="Makine Hareketi Tespit Sistemi",
            font=('Helvetica', 14, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # FPS Kontrolü
        fps_frame = tk.Frame(self.main_frame)
        fps_frame.pack(fill='x', pady=10)
        
        tk.Label(fps_frame, text="Video Hızı (FPS):").pack(side='left')
        self.fps_scale = ttk.Scale(
            fps_frame,
            from_=1,
            to=60,
            orient='horizontal'
        )
        self.fps_scale.set(30)  # Varsayılan FPS
        self.fps_scale.pack(side='left', fill='x', expand=True, padx=10)
        
        # Video seçme butonu
        self.select_button = tk.Button(
            self.main_frame,
            text="Video Dosyası Seç",
            command=self.select_video,
            width=20,
            height=2
        )
        self.select_button.pack(pady=10)
        
        # Seçilen dosya adı
        self.file_label = tk.Label(
            self.main_frame,
            text="Henüz dosya seçilmedi",
            wraplength=350
        )
        self.file_label.pack(pady=10)
        
        # Son durum etiketi
        self.status_label = tk.Label(
            self.main_frame,
            text="",
            font=('Helvetica', 12),
            wraplength=350
        )
        self.status_label.pack(pady=10)
        
        self.root.mainloop()
    
    def select_video(self):
        """Video dosyası seçme dialog'unu açar"""
        file_path = filedialog.askopenfilename(
            title="Video Dosyası Seç",
            filetypes=[
                ("Video dosyaları", "*.mp4 *.avi *.mov"),
                ("Tüm dosyalar", "*.*")
            ]
        )
        
        if file_path:
            self.file_label.config(text=f"Seçilen dosya: {file_path}")
            self.analyze_video(file_path)
    
    def analyze_video(self, video_path):
        """
        Video üzerinde hareket analizi yapar
        """
        # Video yakalama
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Hata", "Video açılamadı!")
            return

        # İlk kareyi al
        ret, frame1 = cap.read()
        if not ret:
            messagebox.showerror("Hata", "Video karesi okunamadı!")
            return

        # Gri tonlamaya çevir
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        # Lucas-Kanade optik akış parametreleri
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Köşe noktası tespit parametreleri
        feature_params = dict(
            maxCorners=200,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        # İlk karede köşe noktalarını bul
        p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
        if p0 is None:
            messagebox.showerror("Hata", "Takip edilecek nokta bulunamadı!")
            return

        # Hareket takipçisi
        motion_tracker = MotionTracker(history_size=15)
        last_frame = None
        last_direction = "BELİRSİZ"

        while True:
            ret, frame2 = cap.read()
            if not ret:
                break

            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Optik akış hesapla
            p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

            # İyi noktaları seç
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Hareket vektörlerini hesapla
            motion_vectors = good_new - good_old

            # Anlık hareket yönünü belirle
            instant_direction = calculate_motion_direction(motion_vectors)
            
            # Hareket geçmişine ekle
            motion_tracker.add_motion(instant_direction)
            
            # Baskın hareket yönünü al
            dominant_direction = motion_tracker.get_dominant_motion()
            
            # Güven oranını hesapla
            confidence = Counter(motion_tracker.history)[dominant_direction] / len(motion_tracker.history)

            # Görüntüyü işaretle
            frame_with_box = draw_motion_box(frame2.copy(), good_new, dominant_direction, confidence)
            last_frame = frame_with_box.copy()
            last_direction = dominant_direction

            # Görüntüyü göster
            cv2.imshow('Makine Hareketi Tespiti', frame_with_box)

            # Sonraki kare için hazırlık
            gray1 = gray2.copy()
            p0 = good_new.reshape(-1, 1, 2)

            # FPS kontrolü
            delay = int(1000 / self.fps_scale.get())  # milisaniye cinsinden bekleme süresi
            if cv2.waitKey(delay) & 0xFF == 27:
                break

        cap.release()
        
        # Son tespit edilen hareketi göster
        if last_frame is not None:
            cv2.imshow('Son Tespit Edilen Hareket', last_frame)
            cv2.waitKey(3000)  # 3 saniye bekle
            
        cv2.destroyAllWindows()
        
        # GUI'de son durumu göster
        self.status_label.config(
            text=f"Video analizi tamamlandı!\nSon tespit edilen hareket: {last_direction}",
            fg="green"
        )

if __name__ == "__main__":
    app = VideoAnalyzerGUI() 