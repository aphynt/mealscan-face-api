# Face Anti-Spoofing - Panduan Lengkap

## 🔒 Apa itu Face Anti-Spoofing?

Face anti-spoofing adalah teknologi untuk mendeteksi apakah wajah yang terdeteksi adalah **wajah asli** atau **wajah palsu** (foto/video dari handphone, layar komputer, atau media lain).

Dengan fitur ini, sistem face recognition akan **menolak** jika seseorang mencoba absen menggunakan foto dari handphone.

## ✨ Fitur

- ✅ Deteksi wajah asli vs wajah palsu
- ✅ Menggunakan model MiniFASNetV2 (ringan, ~1.7 MB)
- ✅ Berjalan di CPU (tidak butuh GPU)
- ✅ Otomatis terintegrasi dengan face recognition
- ✅ Memberikan confidence score

## 📦 Instalasi

### 1. Download Model Anti-Spoofing

Jalankan script untuk download model ONNX:

```bash
python download_antispoofing_model.py
```

Model akan tersimpan di: `models/MiniFASNetV2.onnx`

**Atau download manual:**
- URL: https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx
- Simpan di folder: `models/MiniFASNetV2.onnx`

### 2. Install Dependencies (Jika Belum)

Dependencies sudah ada di `api/requirements.txt`:

```bash
cd api
pip install -r requirements.txt
```

### 3. Restart API Server

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

Jika berhasil, akan muncul log:
```
🔒 Loading anti-spoofing model...
✅ Anti-spoofing model loaded successfully
```

## 🎯 Cara Kerja

### Saat Registrasi Wajah

1. User upload foto untuk registrasi
2. Sistem deteksi wajah
3. **Sistem cek apakah wajah asli atau palsu**
4. Jika wajah palsu → **DITOLAK** dengan error message
5. Jika wajah asli → Lanjut proses registrasi

### Saat Recognition/Absensi

1. User upload foto untuk absen
2. Sistem deteksi wajah
3. **Sistem cek apakah wajah asli atau palsu**
4. Jika wajah palsu → **DITOLAK** dengan error message
5. Jika wajah asli → Lanjut proses recognition & matching

## 📝 API Response

### Response Berhasil (Wajah Asli)

```json
{
  "success": true,
  "message": "Wajah dikenali",
  "employee_id": "123456",
  "nik": "123456",
  "similarity": 0.85,
  "confidence": 0.98,
  "is_real_face": true,
  "anti_spoofing_score": 0.95
}
```

### Response Gagal (Wajah Palsu)

```json
{
  "success": false,
  "message": "Wajah palsu terdeteksi! Mohon gunakan wajah asli, bukan foto/video dari handphone.",
  "is_real_face": false,
  "anti_spoofing_score": 0.87,
  "confidence": 0.92
}
```

## 🧪 Testing

### Test dengan Wajah Asli
- Gunakan webcam/kamera langsung
- Hasil: **Diterima** ✅

### Test dengan Foto dari Handphone
- Foto wajah di handphone, ambil foto layar handphone
- Hasil: **Ditolak** ❌

### Test dengan Foto di Layar Komputer
- Buka foto wajah di layar komputer, foto layar tersebut
- Hasil: **Ditolak** ❌

## ⚙️ Konfigurasi

### Menonaktifkan Anti-Spoofing (Tidak Disarankan)

Jika ingin menonaktifkan sementara, hapus atau rename file model:

```bash
mv models/MiniFASNetV2.onnx models/MiniFASNetV2.onnx.backup
```

Restart server, akan muncul warning:
```
⚠️ Anti-spoofing model not found
⚠️ System will run WITHOUT anti-spoofing protection
```

### Mengaktifkan Kembali

```bash
mv models/MiniFASNetV2.onnx.backup models/MiniFASNetV2.onnx
```

Restart server.

## 🔍 Troubleshooting

### Model Tidak Loading

**Gejala:**
```
⚠️ Anti-spoofing model not found at: models/MiniFASNetV2.onnx
```

**Solusi:**
1. Pastikan file ada di `models/MiniFASNetV2.onnx`
2. Jalankan: `python download_antispoofing_model.py`

### Error saat Prediksi

**Gejala:**
```
Error in anti-spoofing prediction: ...
```

**Solusi:**
1. Pastikan `onnxruntime` terinstall: `pip install onnxruntime==1.16.3`
2. Cek ukuran file model (harus ~1.7 MB)
3. Download ulang model jika corrupt

### False Positive (Wajah Asli Ditolak)

Ini bisa terjadi karena:
- Pencahayaan terlalu gelap/terang
- Kualitas kamera rendah
- Jarak terlalu jauh/dekat

**Solusi:**
- Coba dengan pencahayaan lebih baik
- Pastikan wajah terlihat jelas
- Jarak ideal: 30-60 cm dari kamera

## 📊 Technical Details

### Model Information
- **Model**: MiniFASNetV2
- **Framework**: ONNX Runtime
- **Size**: ~1.7 MB
- **Input**: RGB image (80x80)
- **Output**: 2 classes (Fake/Real)
- **Reference**: [yakhyo/face-anti-spoofing](https://github.com/yakhyo/face-anti-spoofing)

### Processing Pipeline
1. Face detection (InsightFace)
2. Get face bounding box
3. Crop & resize face region (scale=2.7)
4. Normalize & preprocess
5. ONNX inference
6. Softmax classification
7. Return: Real/Fake + confidence score

## 📚 Referensi

- [MiniFASNet Paper](https://arxiv.org/abs/1906.05403)
- [Silent Face Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
- [ONNX Runtime](https://onnxruntime.ai/)

## 🛡️ Security Notes

1. **Anti-spoofing BUKAN solusi 100% sempurna**
   - Masih bisa di-bypass dengan teknik advanced (3D mask, dll)
   - Untuk keamanan maksimal, kombinasikan dengan liveness detection

2. **Rekomendasi Deployment:**
   - Gunakan kamera berkualitas baik
   - Pastikan pencahayaan cukup
   - Monitor false positive/negative rate
   - Update model secara berkala

## 📞 Support

Jika ada masalah atau pertanyaan:
1. Check logs di terminal server
2. Lihat troubleshooting guide di atas
3. Pastikan semua dependencies terinstall

---

**✅ Selamat! Sistem face recognition Anda sekarang dilindungi dari spoofing attack.**
