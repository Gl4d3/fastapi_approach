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
    try:
        processor = DocumentProcessor()
        await processor.initialize()
        logger.info("Document processor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
    yield
    if processor:
        await processor.cleanup()

app = FastAPI(
    title="Kenyan Document Processing API",
    description="AI-powered document validation and data extraction for Kenyan official documents",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None
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

# Web UI Routes
@app.get("/", response_class=HTMLResponse)
async def web_interface(request: Request):
    """Serve the main web interface"""
    if not templates:
        return HTMLResponse("<h1>Kenyan Document Processor API</h1><p>Web interface unavailable. Use /docs for API.</p>")
    
    try:
        supported_docs = processor.get_supported_documents() if processor else []
        return templates.TemplateResponse("index.html", {
            "request": request,
            "title": "Kenyan Document Processor",
            "supported_documents": supported_docs
        })
    except Exception as e:
        return HTMLResponse(f"<h1>Service Available</h1><p>API working. Web UI error: {str(e)}</p><p><a href='/docs'>Use API Documentation</a></p>")

@app.get("/results/{document_id}", response_class=HTMLResponse)
async def results_page(request: Request, document_id: str):
    """Serve results page"""
    if not templates:
        return HTMLResponse("<h1>Results unavailable</h1><p>Web interface is not available. Use API endpoints.</p>")
    
    try:
        status = await processor.get_job_status(document_id) if processor else None
        return templates.TemplateResponse("results.html", {
            "request": request,
            "document_id": document_id,
            "status": status
        })
    except Exception as e:
        return HTMLResponse(f"<h1>Error</h1><p>Could not load results: {str(e)}</p>")

@app.post("/upload", response_model=Dict)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    webhook_url: Optional[str] = None,
    priority: int = 1
):
    """Upload and process document asynchronously"""
    if not processor:
        raise HTTPException(status_code=503, detail="Service unavailable")
        
    if not processor.is_supported_file(file.filename, file.content_type):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    document_id = str(uuid.uuid4())
    
    try:
        file_content = await file.read()
        
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large. Max size: {settings.MAX_FILE_SIZE/1024/1024}MB")
        
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
            "message": "Document uploaded successfully and queued for processing"
        }
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/process-sync")
async def process_document_sync(file: UploadFile = File(...)):
    """Enhanced document processing with foreign document support"""
    if not processor:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    if not processor.is_supported_file(file.filename, file.content_type):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
    
    try:
        file_content = await file.read()
        
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large. Max size: {settings.MAX_FILE_SIZE/1024/1024}MB")
        
        # Log processing attempt
        logger.info(f"Processing file: {file.filename}, size: {len(file_content)} bytes")
        
        start_time = datetime.now()
        result = await processor.process_document_sync(file_content, file.filename)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Enhanced response format
        return {
            "is_valid": result.get('extraction_success', False),
            "extracted_data": result.get('extracted_data', {}),
            "confidence": result.get('overall_confidence', 0.0),
            "document_type": result.get('document_type', 'unknown'),
            "errors": result.get('extraction_errors', []),
            "processing_time": processing_time,
            "classification_confidence": result.get('classification_confidence', 0),
            "extraction_confidence": result.get('extraction_confidence', 0),
            "ocr_sample": result.get('ocr_text_sample', '')[:100]  # First 100 chars for debugging
        }
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/status/{document_id}")
async def get_processing_status(document_id: str):
    """Get processing status for a document"""
    if not processor:
        raise HTTPException(status_code=503, detail="Service unavailable")
        
    status = await processor.get_job_status(document_id)
    if not status:
        raise HTTPException(status_code=404, detail="Document not found")
    return status

@app.post("/batch-process", response_model=BatchProcessResponse)
async def batch_process_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    webhook_url: Optional[str] = None
):
    """Process multiple documents in batch"""
    if not processor:
        raise HTTPException(status_code=503, detail="Service unavailable")
        
    if len(files) > settings.MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Maximum {settings.MAX_BATCH_SIZE} files per batch")
    
    batch_id = str(uuid.uuid4())
    document_ids = []
    
    for i, file in enumerate(files):
        if not processor.is_supported_file(file.filename, file.content_type):
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
        message=f"Batch of {len(document_ids)} documents queued for processing"
    )

@app.get("/supported-documents", response_model=List[SupportedDocument])
async def get_supported_documents():
    """Get list of supported document types"""
    if not processor:
        return []
    return processor.get_supported_documents()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not processor:
        return {"status": "initializing", "timestamp": datetime.now().isoformat()}
    
    return await processor.health_check()

@app.delete("/cleanup/{document_id}")
async def cleanup_document(document_id: str):
    """Clean up processing data for a document"""
    if not processor:
        raise HTTPException(status_code=503, detail="Service unavailable")
        
    success = await processor.cleanup_document(document_id)
    if success:
        return {"message": "Document data cleaned up successfully"}
    else:
        raise HTTPException(status_code=404, detail="Document not found")

@app.get("/ping")
async def ping():
    """Simple ping endpoint for Railway"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
