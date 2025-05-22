import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from utils import draw_motion_box, MotionTracker

class VideoAnalyzerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Makine Hareketi Tespit Sistemi")
        self.root.geometry("600x400")
        
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
        
        # Model Seçimi
        model_frame = tk.Frame(self.main_frame)
        model_frame.pack(fill='x', pady=10)
        
        tk.Label(model_frame, text="Tespit Modeli:").pack(side='left')
        self.model_var = tk.StringVar(value='optical_flow')
        models = [
            ('Optik Akış', 'optical_flow'),
            ('YOLO11-n', 'yolo11n'),
            ('YOLO11-s', 'yolo11s'),
            ('YOLO11-m', 'yolo11m'),
            ('YOLO11-l', 'yolo11l'),
            ('YOLO11-x', 'yolo11x')
        ]
        
        model_select = ttk.Frame(model_frame)
        model_select.pack(side='left', padx=10)
        
        for text, value in models:
            ttk.Radiobutton(
                model_select,
                text=text,
                value=value,
                variable=self.model_var
            ).pack(side='left', padx=5)
        
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
        
        # Hassasiyet Kontrolü
        sensitivity_frame = tk.Frame(self.main_frame)
        sensitivity_frame.pack(fill='x', pady=10)
        
        tk.Label(sensitivity_frame, text="Hassasiyet:").pack(side='left')
        self.sensitivity_scale = ttk.Scale(
            sensitivity_frame,
            from_=0.1,
            to=1.0,
            orient='horizontal'
        )
        self.sensitivity_scale.set(0.25)  # Varsayılan hassasiyet
        self.sensitivity_scale.pack(side='left', fill='x', expand=True, padx=10)
        
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
        """Video üzerinde hareket analizi yapar"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Hata", "Video açılamadı!")
            return
        
        # Seçilen modele göre hareket takipçisi oluştur
        motion_tracker = MotionTracker(
            model_type=self.model_var.get(),
            history_size=15
        )
        motion_tracker.confidence_threshold = self.sensitivity_scale.get()
        
        last_frame = None
        last_direction = "BELİRSİZ"
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Hareket analizi
            detections = motion_tracker.detect_motion(frame)
            
            # Görüntüyü işaretle
            if detections:
                frame = draw_motion_box(frame, detections)
                last_direction = motion_tracker.get_dominant_motion()
            
            # Görüntüyü göster
            cv2.imshow('Makine Hareketi Tespiti', frame)
            last_frame = frame.copy()
            
            # FPS kontrolü
            delay = int(1000 / self.fps_scale.get())
            if cv2.waitKey(delay) & 0xFF == 27:  # ESC tuşu
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