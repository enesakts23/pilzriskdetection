# Makine Hareketi Tespit Sistemi

Bu proje, video görüntülerinden makine hareketlerini tespit eden ve sınıflandıran bir sistemdir.

## Özellikler

- Kullanıcı dostu grafik arayüzü
- Video dosyası seçme ve yükleme
- Hareket tespiti ve takibi
- Hareket türü sınıflandırma:
  - Dönen hareket
  - Yukarı-aşağı hareket
  - Sağa-sola hareket
- Hareketli bölgenin çerçeve içine alınması
- Gerçek zamanlı hareket analizi

## Kurulum

1. Python sanal ortamı oluşturun:
```bash
python3 -m venv venv
```

2. Sanal ortamı aktifleştirin:
```bash
source venv/bin/activate  # Linux/Mac için
```

3. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

## Kullanım

1. Programı başlatın:
```bash
python makine_tespiti/main.py
```

2. Açılan pencereden "Video Dosyası Seç" butonuna tıklayın
3. Analiz etmek istediğiniz video dosyasını seçin
4. Video analizi otomatik olarak başlayacaktır

## Çıktılar

Program çalıştığında:
- Seçilen video üzerinde hareket eden bölge yeşil çerçeve içine alınır
- Ekranın üst kısmında tespit edilen hareket türü gösterilir
- ESC tuşu ile video analizi kapatılabilir

## Hareket Türleri

- **DÖNÜYOR**: Dairesel veya açısal hareket tespit edildiğinde
- **YUKARI-AŞAĞI**: Dikey yönde lineer hareket tespit edildiğinde
- **SAĞA-SOLA**: Yatay yönde lineer hareket tespit edildiğinde
- **BELİRSİZ**: Hareket yönü net olmadığında
