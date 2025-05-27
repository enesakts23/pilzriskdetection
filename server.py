from flask import Flask, request, jsonify, send_from_directory
import os
import threading
import time
import uuid
import subprocess

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv', 'mov'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Basit dosya uzantısı kontrolü
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = str(uuid.uuid4()) + '_' + file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'filename': filename}), 200
    return jsonify({'error': 'Invalid file type'}), 400

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
        # motion_analyzer.py'yi subprocess ile çağırıyoruz
        # motion_analyzer.py'nin sonuna analiz edilen hareket tipini yazan bir kod ekleyeceğiz
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

@app.route('/')
def serve_index():
    return send_from_directory('web', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 