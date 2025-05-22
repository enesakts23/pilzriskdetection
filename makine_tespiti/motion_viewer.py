import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
from motion_analyzer import MotionAnalyzer
import os
import time

class MotionViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Makine Hareket Analizi")
        
        # Ana pencere boyutunu ayarla
        self.root.geometry("1024x768")
        
        # Analiz motoru
        self.analyzer = MotionAnalyzer()
        
        # Video yakalama
        self.cap = None
        self.is_running = False
        self.video_path = None
        self.frame_count = 0
        self.current_frame = 0
        
        # GUI bileşenleri
        self._setup_gui()
        
    def _setup_gui(self):
        # Kontrol paneli
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Video seçme düğmesi
        self.select_button = ttk.Button(
            control_frame, text="Video Seç", command=self._select_video
        )
        self.select_button.pack(side=tk.LEFT, padx=5)
        
        # Başlat/Durdur düğmesi
        self.start_button = ttk.Button(
            control_frame, text="Başlat", command=self._toggle_capture
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.start_button["state"] = "disabled"
        
        # Video ilerleme çubuğu
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Scale(
            control_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.progress_var,
            command=self._on_progress_change
        )
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Video bilgisi
        self.info_label = ttk.Label(control_frame, text="Video seçilmedi")
        self.info_label.pack(side=tk.LEFT, padx=5)
        
        # Görüntü paneli
        self.panel = ttk.Label(self.root)
        self.panel.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Pencere kapatma olayını yakala
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
    def _select_video(self):
        # Video dosyası seç
        video_path = filedialog.askopenfilename(
            title="Video Seç",
            filetypes=[
                ("Video dosyaları", "*.mp4 *.avi *.mkv *.mov"),
                ("Tüm dosyalar", "*.*")
            ]
        )
        
        if video_path:
            try:
                # Video dosyasını aç
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError("Video dosyası açılamadı!")
                    
                # Video bilgilerini al
                self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Bilgileri göster
                duration = self.frame_count / fps
                self.info_label["text"] = f"{os.path.basename(video_path)} - {width}x{height}, {fps:.1f}fps, {duration:.1f}s"
                
                # Video yolu kaydet
                self.video_path = video_path
                self.current_frame = 0
                
                # Başlat düğmesini etkinleştir
                self.start_button["state"] = "normal"
                
                # Progress bar'ı sıfırla
                self.progress_var.set(0)
                
                cap.release()
                
            except Exception as e:
                messagebox.showerror("Hata", str(e))
                
    def _toggle_capture(self):
        if not self.is_running:
            try:
                # Video dosyasını aç
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    raise ValueError("Video dosyası açılamadı!")
                    
                # İstenen frame'e git
                if self.current_frame > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                    
                # Analiz motorunu başlat
                self.analyzer.start_processing()
                
                # Yakalamayı başlat
                self.is_running = True
                self.capture_thread = threading.Thread(target=self._capture_loop)
                self.capture_thread.daemon = True
                self.capture_thread.start()
                
                # Görüntüleme döngüsünü başlat
                self._update_gui()
                
                self.start_button.config(text="Durdur")
                self.select_button["state"] = "disabled"
                
            except Exception as e:
                messagebox.showerror("Hata", str(e))
                self._cleanup()
        else:
            self._cleanup()
            self.start_button.config(text="Başlat")
            self.select_button["state"] = "normal"
            
    def _capture_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                self.is_running = False
                break
                
            # Frame'i analiz motoruna gönder
            self.analyzer.add_frame(frame)
            
            # İlerlemeyi güncelle
            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            progress = (self.current_frame / self.frame_count) * 100
            self.progress_var.set(progress)
            
            # FPS kontrolü (30fps)
            time.sleep(1/30)
            
    def _on_progress_change(self, value):
        if not self.is_running and self.video_path:
            # String'i float'a çevir
            value = float(value)
            # Frame numarasını hesapla
            self.current_frame = int((value / 100) * self.frame_count)
            
    def _update_gui(self):
        if self.is_running:
            # Görselleştirmeyi al
            vis = self.analyzer.get_visualization()
            
            if vis is not None:
                # OpenCV BGR -> RGB
                rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                
                # NumPy array -> PIL Image
                image = Image.fromarray(rgb)
                
                # PIL Image -> Tkinter PhotoImage
                photo = ImageTk.PhotoImage(image)
                
                # Görüntüyü güncelle
                self.panel.config(image=photo)
                self.panel.image = photo  # Referansı koru
                
            # GUI'yi 30ms sonra tekrar güncelle (yaklaşık 30 FPS)
            self.root.after(30, self._update_gui)
            
    def _cleanup(self):
        self.is_running = False
        
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()
            
        if self.cap is not None:
            self.cap.release()
            
        self.analyzer.stop_processing()
        
    def _on_closing(self):
        self._cleanup()
        self.root.destroy()
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    viewer = MotionViewer()
    viewer.run() 