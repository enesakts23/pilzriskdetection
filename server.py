from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import threading
import time
import uuid
import subprocess

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv', 'mov'}

# Klasörleri oluştur
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__)
# CORS ayarlarını güncelle
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max-limit

# Hareket tipi kodları
MOTION_CODES = {
    "ROTARY": 1,
    "ROTATING": 1,
    "PRESS": 2,
    "MIXED": 3,
    "LEFT-RIGHT": 4,
    "UP-DOWN": 5,
    "BELİRSİZ": 0,
    "UNCERTAIN": 0,
    "UNKNOWN": 0
}

# Basit dosya uzantısı kontrolü
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Dosya bulunamadı'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Dosya seçilmedi'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Geçersiz dosya türü. Sadece video dosyaları kabul edilir.'}), 400
        
        # Güvenli dosya adı oluştur
        filename = str(uuid.uuid4()) + '_' + file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Dosyayı kaydet
        try:
            file.save(filepath)
            return jsonify({'filename': filename}), 200
        except Exception as e:
            return jsonify({'error': f'Dosya kaydedilirken hata oluştu: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Beklenmeyen bir hata oluştu: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    # Analiz işlemini başlat
    result_file = os.path.join(app.config['RESULTS_FOLDER'], filename + '.txt')
    try:
        process = subprocess.run([
            'python3', 'run_motion_analysis.py', filepath, result_file
        ], capture_output=True, text=True, timeout=600)
        if process.returncode != 0:
            return jsonify({'error': 'Analysis failed', 'details': process.stderr}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Sonucu oku
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            result = f.read().strip()
        return jsonify({'result': result}), 200
    else:
        return jsonify({'error': 'Result not found'}), 500

# --- API: Tek adımda video analiz endpoint'i ---
@app.route('/api/analyze_video', methods=['POST'])
def api_analyze_video():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadı!'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya adı boş!'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Geçersiz dosya türü!'}), 400

    # Dosyayı kaydet
    filename = str(uuid.uuid4()) + '_' + file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Analiz et
    result_file = os.path.join(app.config['RESULTS_FOLDER'], filename + '.txt')
    try:
        process = subprocess.run([
            'python3', 'run_motion_analysis.py', filepath, result_file
        ], capture_output=True, text=True, timeout=600)
        
        if process.returncode != 0:
            return jsonify({'error': 'Analiz başarısız', 'details': process.stderr}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Sonucu oku ve kodlu JSON döndür
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            result = f.read().strip()
            
        if not result:  # Eğer sonuç boşsa
            return jsonify({"motions": []}), 200
            
        motions = []
        # Her bir hareketi ayrı ayrı işle
        for motion in [m.strip() for m in result.split(',')]:
            if not motion:  # Boş string kontrolü
                continue
                
            # Hareket tipini kontrol et
            motion_type = motion.upper()
            if motion_type in MOTION_CODES:
                motions.append({
                    "id": MOTION_CODES[motion_type],
                    "name": motion_type
                })
                
        return jsonify({"motions": motions}), 200
    else:
        return jsonify({'error': 'Sonuç bulunamadı'}), 500

@app.route('/')
def serve_index():
    return send_from_directory('web', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port) 