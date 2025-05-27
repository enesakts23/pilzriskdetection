import sys
import cv2
from makine_tespiti.motion_analyzer import MotionAnalyzer

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python run_motion_analysis.py <video_path> <result_file>')
        sys.exit(1)
    video_path = sys.argv[1]
    result_file = sys.argv[2]

    cap = cv2.VideoCapture(video_path)
    analyzer = MotionAnalyzer()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detected_motions = []

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        analyzer.add_frame(frame)
        analyzer._analyze_frame(frame)
        # Hareket tipi güncellendi mi kontrol et
        if hasattr(analyzer, 'current_motion'):
            motion = analyzer.current_motion
            if motion and motion != 'BELİRSİZ' and motion not in detected_motions:
                detected_motions.append(motion)
    cap.release()

    # Sadece belirsiz olmayan hareketleri yaz
    detected_motions = [m for m in detected_motions if m != 'BELİRSİZ']
    if detected_motions:
        with open(result_file, 'w') as f:
            f.write(', '.join(detected_motions))
    else:
        with open(result_file, 'w') as f:
            f.write('') 