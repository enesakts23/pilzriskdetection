let uploadedFilename = null;

const uploadForm = document.getElementById('uploadForm');
const analyzeSection = document.getElementById('analyzeSection');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultSection = document.getElementById('resultSection');
const resultDiv = document.getElementById('result');
const errorDiv = document.getElementById('error');
const filenameBox = document.getElementById('filenameBox');
const videoFileInput = document.getElementById('videoFile');

// Hareket tipi -> Font Awesome ikon eşleştirmesi
const motionIcons = {
    'ROTARY':  { icon: 'fas fa-sync-alt', color: 'result-icon-rotary', label: 'Dönme' },
    'ROTATING': { icon: 'fas fa-sync-alt', color: 'result-icon-rotary', label: 'Dönme' },
    'PRESS':   { icon: 'fas fa-arrow-down', color: 'result-icon-press', label: 'Basma' },
    'MIXED':   { icon: 'fas fa-random', color: 'result-icon-mixed', label: 'Karma' },
    'LEFT-RIGHT': { icon: 'fas fa-arrows-alt-h', color: 'result-icon-leftright', label: 'Sağ-Sol' },
    'UP-DOWN': { icon: 'fas fa-arrows-alt-v', color: 'result-icon-updown', label: 'Yukarı-Aşağı' },
    'UNCERTAIN': { icon: 'fas fa-question-circle', color: 'result-icon-unknown', label: 'Bilinmeyen' },
    'UNKNOWN': { icon: 'fas fa-question-circle', color: 'result-icon-unknown', label: 'Bilinmeyen' },
};

function getMotionIcon(motionType) {
    let type = motionType.split('+')[0].trim().toUpperCase();
    return motionIcons[type] || { icon: 'fas fa-cogs', color: 'result-icon-unknown', label: motionType };
}

// Video seçildiğinde dosya adını göster
videoFileInput.addEventListener('change', function() {
    if (videoFileInput.files.length > 0) {
        const file = videoFileInput.files[0];
        filenameBox.innerHTML = `<i class='fas fa-file-video'></i> ${file.name}`;
        filenameBox.style.display = 'flex';
    } else {
        filenameBox.style.display = 'none';
        filenameBox.innerHTML = '';
    }
});

uploadForm.addEventListener('submit', function(e) {
    e.preventDefault();
    errorDiv.textContent = '';
    resultSection.style.display = 'none';
    analyzeSection.style.display = 'none';
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files[0];
    if (!file) {
        errorDiv.textContent = 'Lütfen bir video dosyası seçin.';
        return;
    }
    const formData = new FormData();
    formData.append('file', file);
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.filename) {
            uploadedFilename = data.filename;
            analyzeSection.style.display = 'block';
        } else {
            errorDiv.textContent = data.error || 'Yükleme başarısız.';
        }
    })
    .catch(() => {
        errorDiv.textContent = 'Yükleme sırasında hata oluştu.';
    });
});

analyzeBtn.addEventListener('click', function() {
    errorDiv.textContent = '';
    resultSection.style.display = 'none';
    resultDiv.innerHTML = '<span style="color:#2563eb"><i class="fas fa-spinner fa-spin result-icon"></i></span> Analiz ediliyor, lütfen bekleyin...';
    resultSection.style.display = 'block';
    fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: uploadedFilename })
    })
    .then(res => res.json())
    .then(data => {
        if (data.result !== undefined) {
            let result = data.result.trim();
            if (!result || result.toUpperCase().includes('BELİRSİZ')) {
                resultDiv.innerHTML = '<span style="color:#aaa"><i class="fas fa-info-circle result-icon result-icon-unknown"></i></span> Belirgin bir hareket tespit edilmedi.';
            } else {
                let motions = result.split(',').map(m => m.trim()).filter(Boolean);
                let html = '';
                motions.forEach(motion => {
                    let iconData = getMotionIcon(motion);
                    html += `<div><i class="${iconData.icon} result-icon ${iconData.color}"></i> <span>${motion}</span></div>`;
                });
                resultDiv.innerHTML = html;
            }
        } else {
            resultDiv.textContent = '';
            errorDiv.textContent = data.error || 'Analiz sırasında hata oluştu.';
        }
    })
    .catch(() => {
        resultDiv.textContent = '';
        errorDiv.textContent = 'Analiz sırasında hata oluştu.';
    });
}); 