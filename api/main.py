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
from utils import FaceRecognitionSystem, AntiSpoofingONNX, validate_image_file
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
MODELS_DIR = BASE_DIR / "models"

FACES_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Face recognition and anti-spoofing instances
face_system = None
anti_spoofing = None


@app.on_event("startup")
async def startup_event():
    global face_system, anti_spoofing
    logger.info("🚀 Loading face recognition model...")
    face_system = FaceRecognitionSystem(det_size=(640, 640), similarity_threshold=0.5)
    logger.info("✅ Face recognition model loaded successfully")
    
    # Load anti-spoofing model
    anti_spoofing_model_path = MODELS_DIR / "MiniFASNetV2.onnx"
    if anti_spoofing_model_path.exists():
        try:
            logger.info("🔒 Loading anti-spoofing model...")
            anti_spoofing = AntiSpoofingONNX(str(anti_spoofing_model_path), scale=2.7)
            logger.info("✅ Anti-spoofing model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Anti-spoofing model failed to load: {e}")
            logger.warning("⚠️ System will run WITHOUT anti-spoofing protection")
            anti_spoofing = None
    else:
        logger.warning(f"⚠️ Anti-spoofing model not found at: {anti_spoofing_model_path}")
        logger.warning("⚠️ System will run WITHOUT anti-spoofing protection")
        anti_spoofing = None


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

    # Anti-spoofing check
    is_real_face = None
    anti_spoofing_score = None
    
    if anti_spoofing is not None:
        spoof_result = anti_spoofing.predict(img, result["bbox"])
        is_real_face = spoof_result["is_real"]
        anti_spoofing_score = spoof_result["score"]
        
        logger.info(f"🔒 Anti-spoofing: {spoof_result['label']} (score: {anti_spoofing_score:.3f})")
        
        if not is_real_face:
            logger.warning(f"⚠️ SPOOFING DETECTED! Wajah palsu terdeteksi untuk: {employee_id}")
            raise HTTPException(400, f"Wajah palsu terdeteksi! Mohon gunakan wajah asli, bukan foto/video dari handphone. Score: {anti_spoofing_score:.2f}")

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
        confidence=result["confidence"],
        is_real_face=is_real_face,
        anti_spoofing_score=anti_spoofing_score
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

    # Anti-spoofing check
    is_real_face = None
    anti_spoofing_score = None
    
    if anti_spoofing is not None:
        spoof_result = anti_spoofing.predict(img, result["bbox"])
        is_real_face = spoof_result["is_real"]
        anti_spoofing_score = spoof_result["score"]
        
        logger.info(f"🔒 Anti-spoofing: {spoof_result['label']} (score: {anti_spoofing_score:.3f})")
        
        if not is_real_face:
            logger.warning(f"⚠️ SPOOFING DETECTED! Wajah palsu terdeteksi")
            # Don't block here, continue to find matching face

    embeddings = face_system.load_all_embeddings(str(EMBEDDINGS_DIR))
    # logger.info(f"📚 Loaded embeddings: {list(embeddings.keys())}")

    match = face_system.find_matching_face(result["embedding"], embeddings)

    if match is None:
        logger.info("❌ No match found")
        return {
            "success": False,
            "message": "Wajah tidak dikenali",
            "similarity": 0.0,
            "confidence": float(result["confidence"]),
            "is_real_face": is_real_face,
            "anti_spoofing_score": float(anti_spoofing_score) if anti_spoofing_score else None
        }

    nik, similarity = match
    nik = str(nik)

    logger.info(f"🎯 MATCH FOUND! NIK = {nik}, similarity = {similarity}")

    # Send data regardless of real/fake status
    message = "Wajah dikenali"
    if is_real_face is False:
        message += " (⚠️ FAKE DETECTED)"
    
    response_data = {
        "success": True,
        "message": message,
        "employee_id": nik,
        "nik": nik,
        "similarity": float(similarity),
        "confidence": float(result["confidence"]),
        "is_real_face": is_real_face,
        "anti_spoofing_score": float(anti_spoofing_score) if anti_spoofing_score else None
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

    # Anti-spoofing check
    is_real_face = None
    anti_spoofing_score = None
    
    if anti_spoofing is not None:
        spoof_result = anti_spoofing.predict(img, result["bbox"])
        is_real_face = spoof_result["is_real"]
        anti_spoofing_score = spoof_result["score"]
        
        logger.info(f"🔒 Anti-spoofing: {spoof_result['label']} (score: {anti_spoofing_score:.3f})")
        
        if not is_real_face:
            logger.warning(f"⚠️ SPOOFING DETECTED! Wajah palsu terdeteksi")
            # Don't block here, continue to find matching face

    embeddings = face_system.load_all_embeddings(str(EMBEDDINGS_DIR))
    match = face_system.find_matching_face(result["embedding"], embeddings)

    if match is None:
        return FaceRecognitionResponse(
            success=False,
            message="Wajah tidak dikenali",
            is_real_face=is_real_face,
            anti_spoofing_score=anti_spoofing_score
        )

    nik, similarity = match
    nik = str(nik)

    # Send message with fake warning if detected
    message = "Check-in berhasil"
    if is_real_face is False:
        message += " (⚠️ FAKE DETECTED)"

    # Di sini juga TIDAK panggil Laravel, cukup kirim NIK
    response_data = FaceRecognitionResponse(
        success=True,
        message=message,
        employee_id=nik,
        employee_name=None,  # Laravel yang akan isi nama berdasarkan NIK
        similarity=similarity,
        confidence=result["confidence"],
        can_attend=True,
        meal_type=MealType.LUNCH,
        attendance_id=1,
        is_real_face=is_real_face,
        anti_spoofing_score=anti_spoofing_score
    )

    logger.info(f"📤 Final Check-in Response: {response_data}")

    return response_data
