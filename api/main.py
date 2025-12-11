"""
FastAPI Backend untuk Sistem Absensi Makan dengan Face Recognition
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import logging

# Import local modules
from utils import FaceRecognitionSystem, validate_image_file
from schemas import FaceRegistrationResponse, FaceRecognitionResponse, MealType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="\n[%(asctime)s] [%(levelname)s] %(message)s\n"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Face Recognition API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

FACES_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Face recognition instance
face_system = None


@app.on_event("startup")
async def startup_event():
    global face_system
    logger.info("🚀 Loading face recognition model...")
    face_system = FaceRecognitionSystem(det_size=(640, 640), similarity_threshold=0.5)
    logger.info("✅ Model loaded successfully")


# ============================
# Root
# ============================
@app.get("/")
async def root():
    return {"message": "Face Recognition API Running"}


# ============================
# Registration
# ============================
@app.post("/api/face/register", response_model=FaceRegistrationResponse)
async def register_face(employee_id: str = Form(...), file: UploadFile = File(...)):
    employee_id = str(employee_id)

    logger.info(f"📝 Registering face for employee: {employee_id}")

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File harus berupa gambar")

    content = await file.read()
    if not validate_image_file(content):
        raise HTTPException(400, "Image rusak / terlalu kecil")

    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    result = face_system.extract_face_embedding_from_array(img)

    if result is None:
        raise HTTPException(400, "Tidak ada wajah terdeteksi")

    # Save embedding
    embedding_path = EMBEDDINGS_DIR / f"{employee_id}.pkl"
    face_system.save_embedding(result["embedding"], str(embedding_path))

    # Save original image
    img_path = FACES_DIR / f"{employee_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(img_path), img)

    logger.info(f"✅ Face registered for: {employee_id}")

    return FaceRegistrationResponse(
        success=True,
        message="Face registered",
        employee_id=employee_id,
        bbox=result["bbox"],
        confidence=result["confidence"]
    )


# ============================
# Recognition (dipanggil Laravel)
# ============================
@app.post("/recognize")
async def recognize_face_simple(file: UploadFile = File(...)):
    logger.info("📸 Received recognition request")

    content = await file.read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)

    result = face_system.extract_face_embedding_from_array(img)
    if result is None:
        logger.info("❌ No face detected")
        return {"success": False, "message": "Tidak ada wajah terdeteksi"}

    embeddings = face_system.load_all_embeddings(str(EMBEDDINGS_DIR))
    logger.info(f"📚 Loaded embeddings: {list(embeddings.keys())}")

    match = face_system.find_matching_face(result["embedding"], embeddings)

    if match is None:
        logger.info("❌ No match found")
        return {
            "success": False,
            "message": "Wajah tidak dikenali",
            "similarity": 0.0,
            "confidence": float(result["confidence"])
        }

    nik, similarity = match
    nik = str(nik)

    logger.info(f"🎯 MATCH FOUND! NIK = {nik}, similarity = {similarity}")

    # ⚠️ Tidak lagi ambil nama ke Laravel, cukup kirim NIK & skor
    response_data = {
        "success": True,
        "message": "Wajah dikenali",
        "employee_id": nik,
        "nik": nik,
        "similarity": float(similarity),
        "confidence": float(result["confidence"])
    }

    logger.info(f"📤 Final Response to Laravel: {response_data}")

    return response_data


# ============================
# Check-in attendance (opsional, kalau mau pakai langsung dari Python)
# ============================
@app.post("/api/attendance/checkin", response_model=FaceRecognitionResponse)
async def attendance_checkin(file: UploadFile = File(...)):
    logger.info("📝 Processing attendance check-in")

    content = await file.read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)

    result = face_system.extract_face_embedding_from_array(img)
    if result is None:
        return FaceRecognitionResponse(success=False, message="Tidak ada wajah terdeteksi")

    embeddings = face_system.load_all_embeddings(str(EMBEDDINGS_DIR))
    match = face_system.find_matching_face(result["embedding"], embeddings)

    if match is None:
        return FaceRecognitionResponse(success=False, message="Wajah tidak dikenali")

    nik, similarity = match
    nik = str(nik)

    # Di sini juga TIDAK panggil Laravel, cukup kirim NIK
    response_data = FaceRecognitionResponse(
        success=True,
        message="Check-in berhasil",
        employee_id=nik,
        employee_name=None,  # Laravel yang akan isi nama berdasarkan NIK
        similarity=similarity,
        confidence=result["confidence"],
        can_attend=True,
        meal_type=MealType.LUNCH,
        attendance_id=1
    )

    logger.info(f"📤 Final Check-in Response: {response_data}")

    return response_data
