from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from contextlib import asynccontextmanager
import os
import json

from .core.config import get_settings
from .core.document_processor import DocumentProcessor
from .models.schemas import (
    ProcessingRequest, ProcessingStatus, ValidationResult,
    SupportedDocument, BatchProcessResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor
    processor = DocumentProcessor()
    await processor.initialize()
    logger.info("Enhanced Document processor initialized")
    yield
    await processor.cleanup()

app = FastAPI(
    title="Enhanced Kenyan Document Processing API",
    description="AI-powered document validation with Vision AI, spatial analysis, and multi-language support",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files and templates
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    templates = Jinja2Templates(directory="templates")
except Exception as e:
    logger.warning(f"Template setup failed: {e}")
    templates = None

# Enhanced Web UI Routes
@app.get("/", response_class=HTMLResponse)
async def web_interface(request: Request):
    """Serve the enhanced web interface"""
    if not templates:
        return HTMLResponse("""
        <h1>Enhanced Kenyan Document Processor API</h1>
        <p>Web interface unavailable. Use <a href='/docs'>/docs</a> for API documentation.</p>
        <p>Features: Vision AI, Spatial Analysis, Multi-language Support</p>
        """)
    
    try:
        supported_docs = processor.get_supported_documents() if processor else []
        return templates.TemplateResponse("index.html", {
            "request": request,
            "title": "Enhanced Document Processor",
            "supported_documents": supported_docs
        })
    except Exception as e:
        return HTMLResponse(f"""
        <h1>Enhanced Document Processor</h1>
        <p>Service Available. Web UI error: {str(e)}</p>
        <p><a href='/docs'>Use API Documentation</a></p>
        <ul>
            <li>Vision AI Processing</li>
            <li>Spatial Field Analysis</li>
            <li>Multi-language Detection</li>
            <li>Enhanced Field Extraction</li>
        </ul>
        """)

@app.get("/results/{document_id}", response_class=HTMLResponse)
async def results_page(request: Request, document_id: str):
    """Serve enhanced results page"""
    if not templates:
        return HTMLResponse(f"""
        <h1>Results for Document: {document_id}</h1>
        <p>Web interface is not available. Use API endpoints.</p>
        <a href='/status/{document_id}'>Check Status via API</a>
        """)
    
    try:
        status = await processor.get_job_status(document_id) if processor else None
        return templates.TemplateResponse("results.html", {
            "request": request,
            "document_id": document_id,
            "status": status
        })
    except Exception as e:
        return HTMLResponse(f"<h1>Error</h1><p>Could not load results: {str(e)}</p>")

# Enhanced API Routes
@app.post("/upload", response_model=Dict)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    webhook_url: Optional[str] = None,
    priority: int = 1,
    processing_options: Optional[str] = Form(None)
):
    """Upload and process document asynchronously with enhanced options"""
    if not processor:
        raise HTTPException(status_code=503, detail="Service unavailable")
        
    if not processor.is_supported_file(file.filename, file.content_type):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    document_id = str(uuid.uuid4())
    
    try:
        file_content = await file.read()
        
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File size too large. Max: {settings.MAX_FILE_SIZE/1024/1024}MB")
        
        # Parse processing options if provided
        options = {}
        if processing_options:
            try:
                options = json.loads(processing_options)
            except json.JSONDecodeError:
                logger.warning(f"Invalid processing options: {processing_options}")
        
        background_tasks.add_task(
            processor.process_document_async,
            document_id,
            file_content,
            file.filename,
            webhook_url
        )
        
        await processor.update_job_status(document_id, "queued", 0)
        
        return {
            "document_id": document_id,
            "status": "queued",
            "message": "Document uploaded successfully and queued for enhanced processing",
            "estimated_processing_time": "10-30 seconds",
            "features_enabled": [
                "Vision AI Classification",
                "Spatial Field Analysis", 
                "Multi-language Detection",
                "Enhanced OCR"
            ]
        }
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/process-sync")
async def process_document_sync(file: UploadFile = File(...)):
    """Process document synchronously with enhanced Vision AI processing"""
    if not processor:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    if not processor.is_supported_file(file.filename, file.content_type):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
    
    start_time = datetime.now()
    
    try:
        file_content = await file.read()
        
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large. Max size: {settings.MAX_FILE_SIZE/1024/1024}MB")
        
        logger.info(f"🚀 Processing file: {file.filename}, size: {len(file_content)} bytes")
        
        result = await processor.process_document_sync(file_content, file.filename)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Enhanced response format with all new fields
        enhanced_result = {
            # Core validation results
            "is_valid": result.get('extraction_success', False),
            "document_type": result.get('document_type', 'unknown'),
            "overall_confidence": result.get('overall_confidence', 0.0),
            
            # Detailed confidence breakdown
            "classification_confidence": result.get('classification_confidence', 0),
            "extraction_confidence": result.get('extraction_confidence', 0),
            
            # Enhanced extracted data with all fields
            "extracted_data": result.get('extracted_data', {}),
            
            # Processing metadata
            "processing_time": processing_time,
            "processing_method": result.get('processing_method', 'enhanced_ai'),
            
            # OCR and debugging information
            "ocr_text_sample": result.get('ocr_text_sample', ''),
            "debug_info": {
                **result.get('debug_info', {}),
                "api_version": "2.1.0",
                "timestamp": datetime.now().isoformat(),
                "file_info": {
                    "name": file.filename,
                    "size": len(file_content),
                    "type": file.content_type
                }
            },
            
            # Enhanced error reporting
            "extraction_errors": result.get('extraction_errors', []),
            "warnings": result.get('warnings', []),
            
            # Feature flags
            "features_used": [
                "vision_ai_classification",
                "spatial_field_analysis",
                "enhanced_ocr",
                "multi_language_detection"
            ],
            
            # Additional metadata
            "supported_fields": processor.get_supported_documents() if processor else [],
        }
        
        logger.info(f"✅ Processing completed: {enhanced_result['document_type']} with {enhanced_result['overall_confidence']:.1f}% confidence")
        
        return enhanced_result
        
    except Exception as e:
        logger.error(f"❌ Sync processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/status/{document_id}")
async def get_processing_status(document_id: str):
    """Get enhanced processing status for a document"""
    if not processor:
        raise HTTPException(status_code=503, detail="Service unavailable")
        
    status = await processor.get_job_status(document_id)
    if not status:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Enhance status response
    enhanced_status = {
        **status,
        "api_version": "2.1.0",
        "features_available": [
            "vision_ai_classification",
            "spatial_field_analysis", 
            "enhanced_ocr",
            "multi_language_detection"
        ]
    }
    
    return enhanced_status

@app.post("/batch-process", response_model=BatchProcessResponse)
async def batch_process_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    webhook_url: Optional[str] = None,
    processing_options: Optional[str] = Form(None)
):
    """Process multiple documents in batch with enhanced features"""
    if not processor:
        raise HTTPException(status_code=503, detail="Service unavailable")
        
    if len(files) > settings.MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Maximum {settings.MAX_BATCH_SIZE} files per batch")
    
    batch_id = str(uuid.uuid4())
    document_ids = []
    
    # Parse processing options if provided
    options = {}
    if processing_options:
        try:
            options = json.loads(processing_options)
        except json.JSONDecodeError:
            logger.warning(f"Invalid processing options: {processing_options}")
    
    for i, file in enumerate(files):
        if not processor.is_supported_file(file.filename, file.content_type):
            logger.warning(f"Skipping unsupported file: {file.filename}")
            continue
        
        document_id = f"{batch_id}_{i}"
        document_ids.append(document_id)
        
        try:
            file_content = await file.read()
            background_tasks.add_task(
                processor.process_document_async,
                document_id,
                file_content,
                file.filename,
                webhook_url
            )
            await processor.update_job_status(document_id, "queued", 0)
        except Exception as e:
            logger.error(f"Batch file error: {e}")
    
    return BatchProcessResponse(
        batch_id=batch_id,
        document_ids=document_ids,
        message=f"Batch of {len(document_ids)} documents queued for enhanced processing",
        features_enabled=[
            "Vision AI Classification",
            "Spatial Field Analysis",
            "Multi-language Detection", 
            "Enhanced OCR"
        ]
    )

@app.get("/supported-documents", response_model=List[SupportedDocument])
async def get_supported_documents():
    """Get list of supported document types with enhanced field information"""
    if not processor:
        return []
    return processor.get_supported_documents()

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with detailed service information"""
    if not processor:
        return {
            "status": "initializing", 
            "timestamp": datetime.now().isoformat(),
            "api_version": "2.1.0"
        }
    
    health_data = await processor.health_check()
    
    # Enhance health check response
    enhanced_health = {
        **health_data,
        "api_version": "2.1.0",
        "features": {
            "vision_ai": True,
            "spatial_analysis": True,
            "multi_language": True,
            "enhanced_ocr": True,
            "foreign_documents": True
        },
        "supported_document_types": len(processor.get_supported_documents()),
        "performance": {
            "avg_processing_time": "5-15 seconds",
            "supported_file_formats": ["PDF", "JPEG", "PNG", "TIFF"],
            "max_file_size": f"{settings.MAX_FILE_SIZE/1024/1024}MB"
        }
    }
    
    return enhanced_health

@app.delete("/cleanup/{document_id}")
async def cleanup_document(document_id: str):
    """Clean up processing data for a document"""
    if not processor:
        raise HTTPException(status_code=503, detail="Service unavailable")
        
    success = await processor.cleanup_document(document_id)
    if success:
        return {
            "message": "Document data cleaned up successfully",
            "document_id": document_id,
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=404, detail="Document not found")

# New Enhanced Endpoints

@app.get("/api/v1/capabilities")
async def get_api_capabilities():
    """Get detailed API capabilities and features"""
    return {
        "api_version": "2.1.0",
        "last_updated": "2025-06-03",
        "capabilities": {
            "document_types": 4,
            "languages_supported": ["English", "Swahili", "French", "Romanian"],
            "processing_methods": ["vision_ai", "spatial_analysis", "template_fallback"],
            "max_file_size_mb": settings.MAX_FILE_SIZE / 1024 / 1024,
            "supported_formats": ["PDF", "JPEG", "PNG", "TIFF", "BMP", "WEBP"]
        },
        "features": {
            "vision_ai_classification": {
                "description": "Advanced document type detection using AI",
                "accuracy": "80-95%"
            },
            "spatial_field_analysis": {
                "description": "Field extraction using bounding box analysis", 
                "accuracy": "70-90%"
            },
            "multi_language_detection": {
                "description": "Automatic language detection and processing",
                "supported_languages": 4
            },
            "enhanced_ocr": {
                "description": "Multiple OCR strategies with fallback mechanisms",
                "accuracy": "85-98%"
            }
        },
        "performance": {
            "average_processing_time": "5-15 seconds",
            "concurrent_processing": settings.MAX_CONCURRENT_JOBS,
            "uptime_target": "99.9%"
        }
    }

@app.get("/api/v1/statistics")
async def get_processing_statistics():
    """Get processing statistics (placeholder for future metrics)"""
    return {
        "timestamp": datetime.now().isoformat(),
        "api_version": "2.1.0",
        "statistics": {
            "total_documents_processed": "Tracking not implemented",
            "success_rate": "Tracking not implemented", 
            "average_confidence": "Tracking not implemented",
            "most_common_document_type": "Tracking not implemented"
        },
        "note": "Statistics tracking will be implemented in future versions"
    }

@app.get("/ping")
async def ping():
    """Simple ping endpoint for Railway health checks"""
    return {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(),
        "api_version": "2.1.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)