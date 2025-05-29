from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class DocumentTypeEnum(str, Enum):
    KRA_PIN = "kra_pin"
    BUSINESS_CERT = "business_cert"
    KENYAN_ID = "kenyan_id"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    UNKNOWN = "unknown"

class ProcessingRequest(BaseModel):
    document_id: str
    webhook_url: Optional[str] = None
    priority: int = 1

class ProcessingStatus(BaseModel):
    document_id: str
    status: str
    progress: int
    result: Optional[Dict] = None
    errors: Optional[List[str]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class ValidationResult(BaseModel):
    is_valid: bool
    extracted_data: Dict[str, Any]
    confidence: float
    document_type: str
    errors: List[str]
    processing_time: float

class SupportedDocument(BaseModel):
    type: str
    name: str
    description: str

class BatchProcessResponse(BaseModel):
    batch_id: str
    document_ids: List[str]
    message: str

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str

class WebhookPayload(BaseModel):
    document_id: str
    status: str
    result: Dict[str, Any]
    timestamp: str

class TemplateField(BaseModel):
    name: str
    display_name: str
    region: Dict[str, float]  # x1, y1, x2, y2
    ocr_config: str
    preprocessing: Optional[str] = None
    validation_pattern: Optional[str] = None
    required: bool = True
    data_type: str = "string"
    post_processing: Optional[str] = None

class DocumentTemplateResponse(BaseModel):
    document_type: str
    display_name: str
    description: str
    version: str
    fields: List[TemplateField]
    classification_rules: Dict[str, Any]
    validation_rules: Dict[str, Any]
    created_by: str
    created_at: str
    updated_at: str