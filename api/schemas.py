"""
Database Models dan Pydantic Schemas
Untuk sistem absensi makan dengan face recognition
"""
from datetime import datetime, time
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from enum import Enum


# ============= ENUMS =============

class MealType(str, Enum):
    """Jenis waktu makan"""
    BREAKFAST = "breakfast"  # Makan pagi
    LUNCH = "lunch"          # Makan siang
    DINNER = "dinner"        # Makan malam


class AttendanceStatus(str, Enum):
    """Status absensi"""
    PRESENT = "present"      # Hadir
    ABSENT = "absent"        # Tidak hadir
    LATE = "late"           # Terlambat


class UserRole(str, Enum):
    """Role user"""
    ADMIN = "admin"
    EMPLOYEE = "employee"


# ============= PYDANTIC SCHEMAS =============

class EmployeeBase(BaseModel):
    """Base schema untuk Employee"""
    employee_id: str = Field(..., description="ID Karyawan (unique)")
    name: str = Field(..., min_length=2, max_length=100)
    email: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    department: Optional[str] = Field(None, max_length=50)
    position: Optional[str] = Field(None, max_length=50)
    is_active: bool = Field(default=True)


class EmployeeCreate(EmployeeBase):
    """Schema untuk create employee"""
    pass


class EmployeeUpdate(BaseModel):
    """Schema untuk update employee"""
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    department: Optional[str] = Field(None, max_length=50)
    position: Optional[str] = Field(None, max_length=50)
    is_active: Optional[bool] = None


class EmployeeResponse(EmployeeBase):
    """Schema untuk response employee"""
    id: int
    created_at: datetime
    updated_at: Optional[datetime]
    has_face_registered: bool = False
    
    class Config:
        from_attributes = True


class FaceRegistration(BaseModel):
    """Schema untuk register face"""
    employee_id: str
    # File akan di-upload via multipart/form-data


class FaceRegistrationResponse(BaseModel):
    """Response setelah register face"""
    success: bool
    message: str
    employee_id: str
    confidence: float
    bbox: List[float]
    is_real_face: Optional[bool] = None
    anti_spoofing_score: Optional[float] = None


class FaceRecognitionResponse(BaseModel):
    """Response dari face recognition"""
    success: bool
    message: str
    employee_id: Optional[str] = None
    employee_name: Optional[str] = None
    similarity: Optional[float] = None
    confidence: Optional[float] = None
    can_attend: bool = False
    meal_type: Optional[MealType] = None
    attendance_id: Optional[int] = None
    is_real_face: Optional[bool] = None
    anti_spoofing_score: Optional[float] = None


class MealTimeSettingBase(BaseModel):
    """Base schema untuk meal time settings"""
    meal_type: MealType
    start_time: time = Field(..., description="Waktu mulai makan (HH:MM:SS)")
    end_time: time = Field(..., description="Waktu selesai makan (HH:MM:SS)")
    is_active: bool = Field(default=True)
    
    @validator('end_time')
    def validate_time_range(cls, v, values):
        """Validate bahwa end_time > start_time"""
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time harus lebih besar dari start_time')
        return v


class MealTimeSettingCreate(MealTimeSettingBase):
    """Schema untuk create meal time setting"""
    pass


class MealTimeSettingUpdate(BaseModel):
    """Schema untuk update meal time setting"""
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    is_active: Optional[bool] = None


class MealTimeSettingResponse(MealTimeSettingBase):
    """Schema untuk response meal time setting"""
    id: int
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class AttendanceCreate(BaseModel):
    """Schema untuk create attendance (internal use)"""
    employee_id: str
    meal_type: MealType
    status: AttendanceStatus = AttendanceStatus.PRESENT
    similarity_score: float
    confidence_score: float


class AttendanceResponse(BaseModel):
    """Schema untuk response attendance"""
    id: int
    employee_id: str
    employee_name: str
    meal_type: MealType
    status: AttendanceStatus
    attendance_date: datetime
    similarity_score: float
    confidence_score: float
    created_at: datetime
    
    class Config:
        from_attributes = True


class AttendanceListQuery(BaseModel):
    """Query parameters untuk list attendance"""
    employee_id: Optional[str] = None
    meal_type: Optional[MealType] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    status: Optional[AttendanceStatus] = None
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)


class AttendanceStats(BaseModel):
    """Statistics untuk attendance"""
    total_employees: int
    total_present: int
    total_absent: int
    total_late: int
    attendance_rate: float  # percentage
    meal_type: MealType
    date: datetime


class SystemStatus(BaseModel):
    """Status sistem"""
    status: str
    model_loaded: bool
    total_employees: int
    total_registered_faces: int
    total_attendance_today: int
    current_meal_time: Optional[MealType] = None
    meal_time_active: bool = False


class ErrorResponse(BaseModel):
    """Standard error response"""
    success: bool = False
    error: str
    detail: Optional[str] = None


# ============= DATABASE MODELS (untuk referensi struktur table) =============
"""
Di Laravel Migration, buat table-table berikut:

1. employees:
   - id (bigint, primary key)
   - employee_id (varchar, unique, indexed)
   - name (varchar)
   - email (varchar, nullable)
   - phone (varchar, nullable)
   - department (varchar, nullable)
   - position (varchar, nullable)
   - is_active (boolean, default true)
   - created_at (timestamp)
   - updated_at (timestamp)

2. face_embeddings:
   - id (bigint, primary key)
   - employee_id (varchar, foreign key -> employees.employee_id)
   - embedding_path (varchar) - path ke file .pkl
   - face_image_path (varchar) - path ke foto wajah
   - confidence_score (float)
   - bbox (json) - bounding box coordinates
   - created_at (timestamp)
   - updated_at (timestamp)

3. meal_time_settings:
   - id (bigint, primary key)
   - meal_type (enum: breakfast, lunch, dinner)
   - start_time (time)
   - end_time (time)
   - is_active (boolean, default true)
   - created_at (timestamp)
   - updated_at (timestamp)
   - UNIQUE(meal_type) - hanya 1 setting per meal type

4. attendance_logs:
   - id (bigint, primary key)
   - employee_id (varchar, foreign key -> employees.employee_id)
   - meal_type (enum: breakfast, lunch, dinner)
   - status (enum: present, absent, late)
   - attendance_date (date, indexed)
   - attendance_time (timestamp)
   - similarity_score (float)
   - confidence_score (float)
   - created_at (timestamp)
   - UNIQUE(employee_id, meal_type, attendance_date) - 1 absensi per meal per hari

5. api_logs (optional):
   - id (bigint, primary key)
   - endpoint (varchar)
   - method (varchar)
   - status_code (int)
   - response_time (float)
   - ip_address (varchar)
   - created_at (timestamp)
"""
