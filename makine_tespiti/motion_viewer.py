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
        
        # Pencereyi ekranın ortasında başlat
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        
        # Analiz motoru
        self.analyzer = MotionAnalyzer()
        
        # Video yakalama
        self.cap = None
        self.is_running = False
        self.video_path = None
        self.frame_count = 0
        self.current_frame = 0
        self.original_video_size = (0, 0)
        
        # GUI bileşenleri
        self._setup_gui()
        
    def _setup_gui(self):
        # Ana container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Video container (scrollable)
        self.canvas = tk.Canvas(self.main_container)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Video frame
        self.video_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.video_frame, anchor="nw")
        
        # Scrollbars
        self.h_scrollbar = ttk.Scrollbar(self.main_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.v_scrollbar = ttk.Scrollbar(self.main_container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)
        
        # Video panel
        self.panel = ttk.Label(self.video_frame)
        self.panel.pack(padx=0, pady=0)
        
        # Kontrol paneli
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # Video seçme düğmesi
        self.select_button = ttk.Button(control_frame, text="Video Seç", command=self._select_video)
        self.select_button.pack(side=tk.LEFT, padx=5)
        
        # Başlat/Durdur düğmesi
        self.start_button = ttk.Button(control_frame, text="Başlat", command=self._toggle_capture)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.start_button["state"] = "disabled"
        
        # Video ilerleme çubuğu
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                variable=self.progress_var, command=self._on_progress_change)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Video bilgisi
        self.info_label = ttk.Label(control_frame, text="Video seçilmedi")
        self.info_label.pack(side=tk.LEFT, padx=5)
        
        # Bind resize event
        self.video_frame.bind('<Configure>', self._on_frame_configure)
        
    def _on_frame_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def _select_video(self):
        video_path = filedialog.askopenfilename(
            title="Video Seç",
            filetypes=[("Video dosyaları", "*.mp4 *.avi *.mkv *.mov"), ("Tüm dosyalar", "*.*")]
        )
        
        if video_path:
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError("Video dosyası açılamadı!")
                
                # Video bilgilerini al
                self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Orijinal video boyutunu sakla
                self.original_video_size = (width, height)
                
                # Video bilgilerini göster
                duration = self.frame_count / fps
                self.info_label["text"] = f"{os.path.basename(video_path)} - {width}x{height}, {fps:.1f}fps, {duration:.1f}s"
                
                # Video yolunu kaydet
                self.video_path = video_path
                self.current_frame = 0
                
                # Başlat düğmesini etkinleştir
                self.start_button["state"] = "normal"
                
                # Progress bar'ı sıfırla
                self.progress_var.set(0)
                
                cap.release()
                
            except Exception as e:
                messagebox.showerror("Hata", str(e))
                
    def _update_gui(self):
        if self.is_running:
            # Görselleştirmeyi al
            vis = self.analyzer.get_visualization()
            
            if vis is not None:
                # OpenCV BGR -> RGB
                rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                
                # Orijinal boyutta göster, kesinlikle yeniden boyutlandırma yapma
                image = Image.fromarray(rgb)
                photo = ImageTk.PhotoImage(image)
                
                # Görüntüyü güncelle
                self.panel.configure(image=photo)
                self.panel.image = photo
                
                # Canvas scroll bölgesini güncelle
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            # Video FPS'ine göre güncelle
            if hasattr(self, 'cap'):
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:  # Geçersiz FPS değeri durumunda
                    fps = 30.0
                update_delay = int(1000.0 / fps)  # ms cinsinden delay
                self.root.after(update_delay, self._update_gui)
            else:
                self.root.after(33, self._update_gui)  # Yaklaşık 30 FPS
                
    def _capture_loop(self):
        while self.is_running:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    # Video bittiğinde başa sar
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Frame'i analiz motoruna gönder
                self.analyzer.add_frame(frame)
                
                # İlerlemeyi güncelle
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                progress = (self.current_frame / self.frame_count) * 100
                self.progress_var.set(progress)
                
                # FPS kontrolü
                if hasattr(self, 'cap'):
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0:
                        fps = 30.0
                    time.sleep(1.0 / fps)
            else:
                time.sleep(0.001)
                
    def _toggle_capture(self):
        if not self.is_running:
            try:
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
            
    def _on_progress_change(self, value):
        if not self.is_running and self.video_path:
            # String'i float'a çevir
            value = float(value)
            # Frame numarasını hesapla
            self.current_frame = int((value / 100) * self.frame_count)
            
    def _cleanup(self):
        self.is_running = False
        
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()
            
        if self.cap is not None:
            self.cap.release()
            
        self.analyzer.stop_processing()
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    viewer = MotionViewer()
    viewer.run() 