"""
Face Recognition Utilities
Menggunakan ArcFace face recognition yang ringan di CPU
"""
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime as ort
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceRecognitionSystem:
    """
    Sistem Face Recognition untuk Absensi Makan Karyawan
    Menggunakan model ArcFace
    """
    
    def __init__(self, 
                 det_size: Tuple[int, int] = (640, 640),
                 similarity_threshold: float = 0.5):
        """
        Initialize Face Recognition System
        
        Args:
            det_size: Size untuk face detection
            similarity_threshold: Threshold untuk face matching (0.5 default, strict)
        """
        self.det_size = det_size
        self.similarity_threshold = similarity_threshold
        self.app = None
        
        logger.info(f"Initializing Face Recognition System...")
        self._load_model()
    
    def _load_model(self):
        """Load InsightFace model"""
        try:
            # Initialize FaceAnalysis without specifying model name (will use default)
            # Default model will be auto-downloaded on first run
            self.app = FaceAnalysis(
                providers=["CPUExecutionProvider"]
            )
            self.app.prepare(ctx_id=0, det_size=self.det_size)
            logger.info(f"✓ Face Recognition model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Error loading model: {e}")
            raise
    
    def extract_face_embedding(self, image_path: str) -> Optional[Dict]:
        """
        Extract face embedding dari image
        
        Args:
            image_path: Path ke image file
            
        Returns:
            Dict dengan keys: embedding, bbox, confidence, atau None jika tidak ada wajah
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Cannot read image: {image_path}")
                return None
            
            # Detect faces
            faces = self.app.get(img)
            
            if len(faces) == 0:
                logger.warning(f"No face detected in: {image_path}")
                return None
            
            if len(faces) > 1:
                logger.warning(f"Multiple faces detected ({len(faces)}), using the largest one")
            
            # Get the largest face (by bbox area)
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            return {
                'embedding': face.embedding,
                'bbox': face.bbox.tolist(),
                'confidence': float(face.det_score),
                'embedding_norm': float(np.linalg.norm(face.embedding))
            }
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
    
    def extract_face_embedding_from_array(self, img_array: np.ndarray) -> Optional[Dict]:
        """
        Extract face embedding dari numpy array (untuk upload via API)
        
        Args:
            img_array: Image sebagai numpy array (BGR format)
            
        Returns:
            Dict dengan keys: embedding, bbox, confidence, atau None jika tidak ada wajah
        """
        try:
            if img_array is None:
                logger.error("Invalid image array")
                return None
            
            # Detect faces
            faces = self.app.get(img_array)
            
            if len(faces) == 0:
                logger.warning("No face detected in image")
                return None
            
            if len(faces) > 1:
                logger.warning(f"Multiple faces detected ({len(faces)}), using the largest one")
            
            # Get the largest face
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            return {
                'embedding': face.embedding,
                'bbox': face.bbox.tolist(),
                'confidence': float(face.det_score),
                'embedding_norm': float(np.linalg.norm(face.embedding))
            }
            
        except Exception as e:
            logger.error(f"Error extracting embedding from array: {e}")
            return None
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Hitung cosine similarity antara 2 embeddings
        
        Args:
            emb1: Embedding pertama
            emb2: Embedding kedua
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def compare_faces(self, emb1: np.ndarray, emb2: np.ndarray) -> Tuple[bool, float]:
        """
        Bandingkan 2 face embeddings
        
        Args:
            emb1: Embedding pertama
            emb2: Embedding kedua
            
        Returns:
            Tuple (is_match, similarity_score)
        """
        similarity = self.cosine_similarity(emb1, emb2)
        is_match = similarity >= self.similarity_threshold
        return is_match, similarity
    
    def find_matching_face(self, 
                          query_embedding: np.ndarray, 
                          database_embeddings: Dict[str, np.ndarray]) -> Optional[Tuple[str, float]]:
        """
        Cari wajah yang cocok dari database
        
        Args:
            query_embedding: Embedding yang ingin dicari
            database_embeddings: Dict {employee_id: embedding}
            
        Returns:
            Tuple (employee_id, similarity) atau None jika tidak ada yang cocok
        """
        best_match = None
        best_similarity = 0.0
        
        for employee_id, db_embedding in database_embeddings.items():
            similarity = self.cosine_similarity(query_embedding, db_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = employee_id
        
        # Return only if above threshold
        if best_similarity >= self.similarity_threshold:
            return best_match, best_similarity
        
        return None
    
    def save_embedding(self, embedding: np.ndarray, file_path: str):
        """Save embedding to file using pickle"""
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(embedding, f)
            logger.info(f"✓ Embedding saved to: {file_path}")
        except Exception as e:
            logger.error(f"✗ Error saving embedding: {e}")
            raise
    
    def load_embedding(self, file_path: str) -> Optional[np.ndarray]:
        """Load embedding from file"""
        try:
            with open(file_path, 'rb') as f:
                embedding = pickle.load(f)
            return embedding
        except Exception as e:
            logger.error(f"✗ Error loading embedding: {e}")
            return None
    
    def load_all_embeddings(self, embeddings_dir: str) -> Dict[str, np.ndarray]:
        """
        Load semua embeddings dari directory
        
        Args:
            embeddings_dir: Path ke directory embeddings
            
        Returns:
            Dict {employee_id: embedding}
        """
        embeddings_db = {}
        embeddings_path = Path(embeddings_dir)
        
        if not embeddings_path.exists():
            logger.warning(f"Embeddings directory not found: {embeddings_dir}")
            return embeddings_db
        
        for emb_file in embeddings_path.glob("*.pkl"):
            employee_id = emb_file.stem  # filename without extension
            embedding = self.load_embedding(str(emb_file))
            if embedding is not None:
                embeddings_db[employee_id] = embedding
        
        logger.info(f"✓ Loaded {len(embeddings_db)} embeddings from database")
        return embeddings_db


def save_image_from_upload(upload_file, save_path: str) -> bool:
    """
    Save uploaded file to disk
    
    Args:
        upload_file: FastAPI UploadFile object
        save_path: Path untuk menyimpan file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save file
        with open(save_path, "wb") as buffer:
            content = upload_file.file.read()
            buffer.write(content)
        
        logger.info(f"✓ Image saved to: {save_path}")
        return True
    except Exception as e:
        logger.error(f"✗ Error saving image: {e}")
        return False


def validate_image_file(file_content: bytes) -> bool:
    """
    Validate apakah file adalah valid image
    
    Args:
        file_content: File content sebagai bytes
        
    Returns:
        True if valid image, False otherwise
    """
    try:
        # Decode image from bytes
        nparr = np.frombuffer(file_content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return False
        
        # Check dimensions
        if img.shape[0] < 50 or img.shape[1] < 50:
            logger.warning("Image too small")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating image: {e}")
        return False


class AntiSpoofingONNX:
    """
    Face Anti-Spoofing menggunakan MiniFASNetV2 ONNX model
    Untuk mendeteksi apakah wajah asli atau fake (foto/video dari handphone)
    """
    
    def __init__(self, model_path: str, scale: float = 2.7):
        """
        Initialize Anti-Spoofing detector
        
        Args:
            model_path: Path ke ONNX model file
            scale: Crop scale factor (2.7 untuk V2, 4.0 untuk V1SE)
        """
        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
            )
            self.scale = scale
            
            # Get input/output names and shapes
            input_cfg = self.session.get_inputs()[0]
            self.input_name = input_cfg.name
            self.input_size = tuple(input_cfg.shape[2:])
            
            output_cfg = self.session.get_outputs()[0]
            self.output_name = output_cfg.name
            
            logger.info(f"✓ Anti-spoofing model loaded: {model_path}")
        except Exception as e:
            logger.error(f"✗ Error loading anti-spoofing model: {e}")
            raise
    
    def _xyxy2xywh(self, bbox: List[float]) -> List[int]:
        """Convert [x1, y1, x2, y2] to [x, y, w, h]"""
        x1, y1, x2, y2 = bbox
        return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
    
    def _crop_face(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Crop and resize face region from image"""
        src_h, src_w = image.shape[:2]
        x, y, box_w, box_h = bbox
        
        scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, self.scale)
        new_w = box_w * scale
        new_h = box_h * scale
        
        center_x = x + box_w / 2
        center_y = y + box_h / 2
        
        x1 = max(0, int(center_x - new_w / 2))
        y1 = max(0, int(center_y - new_h / 2))
        x2 = min(src_w - 1, int(center_x + new_w / 2))
        y2 = min(src_h - 1, int(center_y + new_h / 2))
        
        cropped = image[y1:y2 + 1, x1:x2 + 1]
        return cv2.resize(cropped, self.input_size[::-1])
    
    def _preprocess(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Preprocess face crop for inference"""
        face = self._crop_face(image, bbox)
        face = face.astype(np.float32)
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        return face
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to logits"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
    
    def predict(self, image: np.ndarray, bbox_xyxy: List[float]) -> Dict:
        """
        Predict apakah wajah asli (Real) atau palsu (Fake)
        
        Args:
            image: Input image (BGR format)
            bbox_xyxy: Face bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary dengan keys: label, score, is_real
        """
        try:
            bbox_xywh = self._xyxy2xywh(bbox_xyxy)
            
            input_tensor = self._preprocess(image, bbox_xywh)
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            
            logits = outputs[0]
            probs = self._softmax(logits)
            
            label_idx = int(np.argmax(probs))
            score = float(probs[0, label_idx])
            
            return {
                "label": "Real" if label_idx == 1 else "Fake",
                "score": score,
                "is_real": label_idx == 1
            }
        except Exception as e:
            logger.error(f"Error in anti-spoofing prediction: {e}")
            return {
                "label": "Unknown",
                "score": 0.0,
                "is_real": False
            }

