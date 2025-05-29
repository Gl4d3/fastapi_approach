import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from app.main import app
import io
from PIL import Image
import numpy as np

# Test client
client = TestClient(app)

@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    # Create a simple test image
    img = Image.new('RGB', (800, 600), color='white')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "services" in data

def test_supported_documents():
    """Test the supported documents endpoint"""
    response = client.get("/supported-documents")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

def test_upload_invalid_file():
    """Test uploading an invalid file type"""
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"invalid content", "text/plain")}
    )
    assert response.status_code == 400

def test_process_sync_invalid_file():
    """Test sync processing with invalid file"""
    response = client.post(
        "/process-sync",
        files={"file": ("test.txt", b"invalid content", "text/plain")}
    )
    assert response.status_code == 400

def test_process_sync_valid_image(sample_image):
    """Test sync processing with valid image"""
    response = client.post(
        "/process-sync",
        files={"file": ("test.png", sample_image, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "is_valid" in data
    assert "extracted_data" in data
    assert "confidence" in data
    assert "document_type" in data
    assert "processing_time" in data

def test_upload_valid_image(sample_image):
    """Test async upload with valid image"""
    response = client.post(
        "/upload",
        files={"file": ("test.png", sample_image, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["status"] == "queued"

def test_status_not_found():
    """Test status check for non-existent document"""
    response = client.get("/status/non-existent-id")
    assert response.status_code == 404

def test_cleanup_not_found():
    """Test cleanup for non-existent document"""
    response = client.delete("/cleanup/non-existent-id")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_async_endpoints():
    """Test async endpoints"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200

def test_batch_process_empty():
    """Test batch processing with no files"""
    response = client.post("/batch-process", files=[])
    assert response.status_code == 422  # Validation error

def test_batch_process_valid(sample_image):
    """Test batch processing with valid files"""
    files = [
        ("files", ("test1.png", sample_image, "image/png")),
        ("files", ("test2.png", sample_image, "image/png"))
    ]
    response = client.post("/batch-process", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "batch_id" in data
    assert "document_ids" in data
    assert len(data["document_ids"]) == 2

if __name__ == "__main__":
    pytest.main([__file__])