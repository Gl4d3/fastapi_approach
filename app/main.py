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
    processor = DocumentProcessor()
    await processor.initialize()
    logger.info("Document processor initialized")
    yield
    await processor.cleanup()

app = FastAPI(
    title="Kenyan Document Processing API",
    description="AI-powered document validation and data extraction for Kenyan official documents",
    version="2.0.0",
    lifespan=lifespan
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
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Web UI Routes
@app.get("/", response_class=HTMLResponse)
async def web_interface(request: Request):
    """Serve the main web interface"""
    supported_docs = processor.get_supported_documents() if processor else []
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Kenyan Document Processor",
        "supported_documents": supported_docs
    })

@app.get("/results/{document_id}", response_class=HTMLResponse)
async def results_page(request: Request, document_id: str):
    """Serve results page"""
    status = await processor.get_job_status(document_id) if processor else None
    return templates.TemplateResponse("results.html", {
        "request": request,
        "document_id": document_id,
        "status": status
    })

@app.post("/upload", response_model=Dict)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    webhook_url: Optional[str] = None,
    priority: int = 1
):
    """Upload and process document asynchronously"""
    if not processor.is_supported_file(file.filename, file.content_type):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    document_id = str(uuid.uuid4())
    
    try:
        file_content = await file.read()
        
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File size too large")
        
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

@app.post("/process-sync", response_model=ValidationResult)
async def process_document_sync(file: UploadFile = File(...)):
    """Process document synchronously and return results immediately"""
    if not processor.is_supported_file(file.filename, file.content_type):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    start_time = datetime.now()
    
    try:
        file_content = await file.read()
        result = await processor.process_document_sync(file_content, file.filename)
        
        return ValidationResult(
            is_valid=result.get('extraction_success', False),
            extracted_data=result.get('extracted_data', {}),
            confidence=result.get('overall_confidence', 0.0),
            document_type=result.get('document_type', 'unknown'),
            errors=result.get('extraction_errors', []),
            processing_time=(datetime.now() - start_time).total_seconds()
        )
        
    except Exception as e:
        logger.error(f"Sync processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/status/{document_id}")
async def get_processing_status(document_id: str):
    """Get processing status for a document"""
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
    return processor.get_supported_documents()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return await processor.health_check()

@app.delete("/cleanup/{document_id}")
async def cleanup_document(document_id: str):
    """Clean up processing data for a document"""
    success = await processor.cleanup_document(document_id)
    if success:
        return {"message": "Document data cleaned up successfully"}
    else:
        raise HTTPException(status_code=404, detail="Document not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
