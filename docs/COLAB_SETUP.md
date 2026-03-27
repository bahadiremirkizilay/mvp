# Google Colab ile Batch Validation Nasıl Yapılır?

## 📋 Gereksinimler
- Google Drive hesabı
- UBFC-RPPG dataset (birden fazla subject klasörü)
- Bu projenin dosyaları Drive'da

## 🚀 Adım Adım Kurulum

### 1. Projeyi Google Drive'a Yükle
```
Google Drive/
└── mvp/                          ← Proje klasörü
    ├── colab_validation.ipynb    ← Notebook dosyası
    ├── models/
    │   └── face_landmarker.task  ← MediaPipe modeli
    └── data/
        └── ubfc/                 ← UBFC-RPPG dataset
            ├── subject1/
            │   ├── vid.avi
            │   └── ground_truth.txt
            ├── subject2/
            │   ├── vid.avi
            │   └── ground_truth.txt
            └── ...
```

### 2. Colab'da Aç
1. Google Drive'da `colab_validation.ipynb` dosyasına sağ tık
2. **"Birlikte aç"** → **"Google Colaboratory"** seç
3. Runtime → **"Change runtime type"** → **GPU** seç (opsiyonel, ama daha hızlı)

### 3. Drive Yolunu Ayarla
**Cell 2'de** (MOUNT GOOGLE DRIVE):
```python
PROJECT_PATH = '/content/drive/MyDrive/mvp'      # Proje klasörünüz
DATASET_PATH = f'{PROJECT_PATH}/data/ubfc'        # UBFC dataset yolu
```

### 4. Tüm Cell'leri Çalıştır
- **Runtime** → **Run all** (veya Ctrl+F9)
- İlk çalıştırmada Google Drive izni isteyecek → **"Connect to Google Drive"** tıkla

## 📊 Sonuçları Nerede Bulursunuz?

Colab çalışması bittiğinde Drive'da 3 dosya oluşur:

1. **`batch_validation_results.csv`**
   - Her subject için detaylı metrikler (MAE, RMSE, Correlation, vb.)
   - Excel/Numbers ile açabilirsiniz

2. **`batch_validation_plots.png`**
   - MAE dağılımı
   - Correlation dağılımı
   - MAE vs Correlation scatter plot
   - Sample count vs MAE

3. **`BATCH_VALIDATION_REPORT.txt`**
   - Özet istatistikler
   - Genel performans metrikleri
   - Subject bazında tablo

## 📈 Beklenen Sonuçlar

Tek video test sonuçlarımız (optimizasyon sonrası):
- ✅ **MAE: 6.26 BPM**
- ✅ **RMSE: 7.56 BPM**
- ✅ **Correlation: +0.39**

Batch validation ile (tüm subjects):
- Ortalama MAE **~7-10 BPM** aralığında olmalı
- Correlation **~0.3-0.5** aralığında olmalı
- Bazı subjects çok iyi (MAE <5), bazıları daha kötü olabilir (video kalitesi, ışıklandırma, hareket vb.)

## ⚠️ Önemli Notlar

### UBFC-RPPG Dataset Yapısı
Her subject klasörü **mutlaka** şunları içermeli:
- `vid.avi` - Video dosyası
- `ground_truth.txt` - Ground truth BPM değerleri (3 satırlık format)

### Sık Karşılaşılan Hatalar ve Çözümleri

#### "module 'mediapipe' has no attribute 'solutions'"
**Neden:** MediaPipe kurulumu eksik veya bozuk
**Çözüm:**
1. **Runtime → Restart runtime** yapın
2. Cell 1'i tekrar çalıştırın (MediaPipe'ı yeniden yükler)
3. Hata devam ederse: `Runtime → Factory reset runtime`
4. Tüm cell'leri sırayla çalıştırın

#### "No predictions generated"
**Neden:** Yüz tespit edilemedi veya motion confidence çok düşük
**Çözüm:**
→ Video kalitesini kontrol edin
→ Videoda net yüz görünümü var mı?
→ Aşırı hareket var mı?

#### "Too few aligned samples"
**Neden:** Çok az BPM tahmini yapıldı (<10 örnek)
**Çözüm:**
→ Video çok kısa olabilir (minimum 30+ saniye önerilir)
→ Hareket çok fazla olabilir
→ Aydınlatma çok zayıf olabilir

#### "Failed to load ground truth"
**Neden:** ground_truth.txt formatı yanlış
**Çözüm:**
→ Dosyanın 3 satırdan oluştuğunu kontrol edin:
  - Satır 1: rPPG signal (kullanılmaz)
  - Satır 2: BPM değerleri (boşlukla ayrılmış)
  - Satır 3: Timestamp değerleri (boşlukla ayrılmış)

#### "Session Crashed" / "RAM Tükendi" (Cell 8'de)
**Neden:** Çok fazla video aynı anda işlenmeye çalışılıyor, RAM doldu
**Çözüm:**
1. **Runtime → Restart runtime** yapın
2. Tüm cell'leri tekrar çalıştırın (Cell 1-7)
3. Cell 8 otomatik olarak ilk 15 subject ile sınırlayacak
4. **Alternatif:** Colab Pro kullanın (daha fazla RAM)
5. **Alternatif:** Subject'leri manuel olarak gruplandırın:
   ```python
   # Cell 8'den önce yeni bir cell'de:
   subjects = subjects[0:10]  # İlk 10 subject
   # Sonra Cell 8'i çalıştır
   ```

#### "Out of Memory" Hatası
**Neden:** GPU/RAM yetersiz
**Çözüm:**
→ Runtime → Change runtime type → **CPU seçin** (GPU gerekmez)
→ Runtime → Restart runtime
→ Daha az subject ile test edin (5-10 subject)

### İpuçları
- **GPU Runtime** kullanın (daha hızlı)
- **Test için önce 2-3 subject** ile deneyin
- Tüm dataset için **~30-60 dakika** sürebilir (subject sayısına bağlı)
- **Colab ücretsiz kullanımda 12 saat runtime limiti** var

## 🔧 Parametreleri Değiştirme

**Cell 3'te** (IMPORTS & CONFIGURATION) parametreleri değiştirebilirsiniz:

```python
CONFIG = {
    'rppg': {
        'window_seconds': 7,        # Daha kısa = daha hızlı tepki, daha gürültülü
        'bandpass_low': 0.8,        # Alt frekans limiti (BPM/60)
        'bandpass_high': 2.2,       # Üst frekans limiti (BPM/60)
    },
    'hrv': {
        'min_peak_prominence': 0.10,  # Daha düşük = daha hassas
        'min_peak_distance_sec': 0.35,  # Daha kısa = daha yüksek BPM tespit edebilir
    },
    'visualization': {
        'update_interval': 7         # Daha düşük = daha sık BPM tahmini
    }
}
```

## 📞 Destek

Sorun yaşarsanız:
1. Drive bağlantısını kontrol edin (Cell 2)
2. Dataset yapısını kontrol edin (her klasörde vid.avi + ground_truth.txt)
3. Notebook'u yeniden başlatın: Runtime → Restart runtime
4. Tek bir subject ile test edin (hata mesajlarını görün)

---

**Not:** Bu notebook, optimize edilmiş konfigürasyon ile çalışır (tek video testlerinden elde edilen en iyi parametreler). Farklı dataset'ler için parametreleri ayarlamanız gerekebilir.
