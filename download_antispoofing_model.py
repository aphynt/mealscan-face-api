"""
Script untuk download MiniFASNetV2 ONNX model
untuk face anti-spoofing detection
"""
import urllib.request
import os
from pathlib import Path

# Model URL from GitHub releases
MODEL_URL = "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx"
MODEL_NAME = "MiniFASNetV2.onnx"

# Target directory
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / MODEL_NAME

def download_model():
    """Download the anti-spoofing ONNX model"""
    
    print("=" * 60)
    print("📥 Downloading Face Anti-Spoofing Model")
    print("=" * 60)
    
    # Create models directory if not exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    if MODEL_PATH.exists():
        print(f"✅ Model sudah ada di: {MODEL_PATH}")
        print("⚠️ Jika ingin download ulang, hapus file tersebut terlebih dahulu.")
        return
    
    print(f"\n📍 URL: {MODEL_URL}")
    print(f"📂 Target: {MODEL_PATH}")
    print("\n⏳ Downloading... (Ukuran ~1.7 MB)")
    
    try:
        def progress_callback(block_num, block_size, total_size):
            """Show download progress"""
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            bar_length = 40
            filled = int(bar_length * downloaded / total_size)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\r[{bar}] {percent:.1f}% ({downloaded / 1024 / 1024:.2f} MB)", end='', flush=True)
        
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, progress_callback)
        print("\n\n✅ Download selesai!")
        print(f"✅ Model tersimpan di: {MODEL_PATH}")
        
        # Verify file size
        file_size = MODEL_PATH.stat().st_size / 1024 / 1024
        print(f"📊 Ukuran file: {file_size:.2f} MB")
        
        if file_size < 1.0:
            print("⚠️ WARNING: File size terlalu kecil, mungkin download tidak sempurna")
            print("⚠️ Coba hapus file dan download lagi")
        else:
            print("\n" + "=" * 60)
            print("✅ SUKSES! Anti-spoofing model siap digunakan")
            print("=" * 60)
            print("\n📝 Langkah selanjutnya:")
            print("   1. Restart API server (uvicorn)")
            print("   2. Anti-spoofing akan otomatis aktif")
            print("   3. Test dengan foto dari handphone (harus ditolak)")
            print("   4. Test dengan wajah asli (harus diterima)")
            
    except Exception as e:
        print(f"\n\n❌ Error saat download: {e}")
        print("\n💡 Solusi:")
        print("   1. Pastikan koneksi internet stabil")
        print("   2. Atau download manual dari:")
        print(f"      {MODEL_URL}")
        print(f"   3. Simpan di: {MODEL_PATH}")
        
        # Clean up partial download
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()

if __name__ == "__main__":
    download_model()
