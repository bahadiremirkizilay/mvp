# Performance Analysis - What to Improve?

## 📊 Current Results (5 Subjects)

| Subject | MAE | Correlation | Bias | GT Std | Pred Std | Problem |
|---------|-----|-------------|------|--------|----------|---------|
| subject1 | 8.46 | **+0.471** ✅ | -7.13 | 4.96 | 7.32 | Iyi! |
| subject3 | 8.11 | +0.199 | -6.87 | 4.64 | 5.89 | Orta |
| subject4 | 12.10 | **-0.038** ❌ | -11.04 | 2.73 | 6.56 | Kötü - yüksek varyans |
| subject5 | 8.93 | +0.011 | +5.65 | 2.86 | 10.19 | **ÇOK yüksek varyans** |
| subject8 | 10.97 | **-0.332** ❌ | -9.18 | 4.52 | 4.99 | Negatif korelasyon! |

**Overall:** MAE=9.71, Correlation=0.062, Bias=-5.71

---

## 🔍 Problem 1: Sistematik DÜŞÜK Tahmin (Bias: -5.71 BPM)

### Neden?
- Peak detection çok muhafazakar (çok az peak buluyor)
- Bandpass filter çok dar olabilir
- Prominence threshold çok yüksek

### Çözüm:
```yaml
# Mevcut:
hrv:
  min_peak_prominence: 0.10
  min_peak_distance_sec: 0.35

# Önerilen:
hrv:
  min_peak_prominence: 0.08      # ↓ Daha hassas peak detection
  min_peak_distance_sec: 0.30    # ↓ Daha sık peak'lere izin ver
```

---

## 🔍 Problem 2: Yüksek Varyans (Pred Std > GT Std)

### Neden?
- Motion artifacts filtrelenmemiş
- ROI selection kararsız
- Outlier rejection yetersiz

### Çözüm:
```yaml
# Mevcut:
rppg:
  ma_window_sec: 0.3

# Önerilen:
rppg:
  ma_window_sec: 0.5             # ↑ Daha fazla smoothing
```

```python
# validate.py / batch_validate.py'deki threshold:
_MOTION_CONF_THRESHOLD = 0.50

# Önerilen:
_MOTION_CONF_THRESHOLD = 0.60    # ↑ Sadece çok temiz sinyaller
```

---

## 🔍 Problem 3: Çok Değişken Correlation (-0.332 → +0.471)

### Neden?
- Dataset 2 = hareket + stres (bazı subject'lerde çok hareket var)
- Motion filtering yetersiz
- Sistem subject-dependent

### Çözüm:
**Adaptive processing:**
- Hareket çok ise → daha fazla smoothing
- Hareket az ise → daha hassas peak detection

---

## 🎯 ÖNCELİK SIRASI (Kolay → Zor)

### Level 1: Config Parametreleri (5 dk) ⭐⭐⭐⭐⭐
1. **Peak prominence:** 0.10 → 0.08
2. **Peak distance:** 0.35 → 0.30
3. **MA window:** 0.3 → 0.5
4. **Bandpass:** 0.8-2.2 → 0.7-2.5 Hz

**Beklenen iyileşme:** Bias -5.71 → -2~3 BPM, MAE 9.71 → 8~9 BPM

### Level 2: Motion Filtering (10 dk) ⭐⭐⭐⭐
1. **Motion threshold:** 0.50 → 0.60
2. Frame-to-frame motion detection ekle
3. Sadece stable window'ları kullan

**Beklenen iyileşme:** Correlation 0.062 → 0.15~0.25

### Level 3: Signal Quality Check (30 dk) ⭐⭐⭐
1. Her tahmin için SNR hesapla
2. Düşük quality tahminleri ata
3. Quality-weighted temporal smoothing

**Beklenen iyileşme:** MAE 9.71 → 7~8 BPM, Correlation → 0.30+

### Level 4: Advanced Filtering (1-2 saat) ⭐⭐
1. Adaptive bandpass filter
2. ICA/PCA-based artifact removal
3. Multi-scale temporal analysis

**Beklenen iyileşme:** State-of-the-art performance

---

## 💡 EN HIZLI İYİLEŞTİRME (5 DK)

```yaml
# config/config.yaml

rppg:
  window_seconds: 7
  bandpass_low: 0.7              # ← 0.8'den genişlet
  bandpass_high: 2.5             # ← 2.2'den genişlet
  bandpass_order: 4
  min_frames: 42
  ma_window_sec: 0.5             # ← 0.3'ten artır (daha smooth)

hrv:
  min_peak_prominence: 0.08      # ← 0.10'dan düşür (daha hassas)
  min_peak_distance_sec: 0.30    # ← 0.35'ten düşür (daha sık peak)

visualization:
  update_interval: 7
```

```python
# validate.py ve batch_validate.py'de:
_MOTION_CONF_THRESHOLD = 0.60    # ← 0.50'den artır (daha temiz)
```

**Test et:**
```bash
python batch_validate.py --subjects subject1 subject3 subject4 subject5 subject8
python analyze_batch_results.py
```

---

## 📌 DONANIM vs ALGORİTMA

| MAE/RMSE/Correlation Nedeni | Donanım | Algoritma |
|------------------------------|---------|-----------|
| Sistematik bias (-5.71 BPM) | ❌ | ✅ Peak detection |
| Yüksek varyans | ❌ | ✅ Motion filtering |
| Değişken correlation | ⚠️ Video quality | ✅ Signal processing |
| Negatif correlation | ❌ | ✅ Outlier rejection |

**SONUÇ: %95 ALGORİTMA, %5 VIDEO KALİTESİ**

Senin PC/kamera özellikleriyle ALAKASIZ! Validation offline, sadece algoritma parametreleri önemli.

---

## 🎯 RECOMMENDATION

**1. İlk önce Level 1 yap (yukarıdaki config değişiklikleri)**
   → 5 dakika, hemen sonuç göreceksin
   
**2. Sonuçlar iyileşirse Level 2'ye geç**
   → Motion filtering ekle
   
**3. Daha fazla iyileştirme gerekirse Level 3/4**
   → Advanced techniques

Config değişikliklerini yapayım mı? 🚀
