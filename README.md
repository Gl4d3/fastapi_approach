# Kenyan Document Processing System
A cost-effective, open-source document validation and data extraction system specifically designed for Kenyan government documents including KRA PIN certificates, National IDs, Business Registration certificates, and more.

## 🚀 Features

- **Multi-format Support**: Process images (JPEG, PNG, TIFF, etc.) and PDF documents
- **Template-based Extraction**: Accurate field extraction using document-specific templates
- **Async Processing**: Handle high-volume document processing with background jobs
- **Real-time Status**: Track processing status and receive webhook notifications
- **Batch Processing**: Process multiple documents simultaneously
- **Redis Caching**: Optional Redis integration for job queuing and caching
- **Health Monitoring**: Built-in health checks and monitoring endpoints
- **Docker Ready**: Complete containerization with Docker Compose
- **Production Ready**: Comprehensive testing, logging, and deployment configuration

## 📋 Supported Documents

- **KRA PIN Certificate**: Extract PIN number, taxpayer name, email, certificate date
- **Kenyan National ID**: Extract ID number, full name, date of birth, district of birth
- **Business Registration Certificate**: Extract business name, registration number, business type, registration date
- **Passport**: Extract passport details (template ready - coming soon)
- **Driver's License**: Extract license details (template ready - coming soon)

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python 3.9+)
- **OCR Engine**: Tesseract OCR + EasyOCR
- **Image Processing**: OpenCV, Pillow, scikit-image
- **PDF Processing**: PyMuPDF (fitz)
- **Caching**: Redis (optional)
- **Containerization**: Docker & Docker Compose
- **Configuration**: YAML-based templates
- **Testing**: Pytest with async support
- **Monitoring**: Prometheus metrics, structured logging

## 📁 Project Structure

```
kenyan-doc-processor/
├── app/                       # Main application package
│   ├── __init__.py            # Package initialization
│   ├── main.py                # FastAPI application and endpoints
│   ├── core/                  # Core application modules
│   │   ├── __init__.py        # Core package initialization
│   │   ├── config.py          # Configuration management
│   │   └── document_processor.py # Main document processing logic
│   ├── models/                # Pydantic models and schemas
│   │   ├── __init__.py        # Models package initialization
│   │   └── schemas.py         # API request/response models
│   └── utils/                 # Utility modules
│       ├── __init__.py        # Utils package initialization
│       ├── file_handlers.py   # PDF and image processing utilities
│       └── template_manager.py # Template configuration management
├── templates/                 # Document extraction templates
│   ├── kra_pin.yaml           # KRA PIN certificate template
│   ├── kenyan_id.yaml         # Kenyan National ID template
│   └── business_cert.yaml     # Business registration template
├── docker/                    # Docker configuration
│   ├── Dockerfile             # Application container definition
│   ├── docker-compose.yml     # Multi-service orchestration
│   ├── nginx.conf             # Nginx reverse proxy config
│   └── entrypoint.sh          # Container startup script
├── tests/                     # Test suite
│   ├── __init__.py            # Test package initialization
│   ├── test_main.py           # Main API endpoint tests
│   └── sample_docs/           # Sample documents for testing
├── scripts/                   # Development and deployment scripts
│   ├── setup.sh               # Development environment setup
│   ├── test.sh                # Test runner script
│   ├── deploy.sh              # Deployment script
│   └── make_executable.sh     # Script permissions setup
├── uploads/                   # Uploaded files (created at runtime)
├── logs/                      # Application logs (created at runtime)
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project configuration and metadata
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── .dockerignore              # Docker ignore rules
└── README.md                  # This file
```


## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Tesseract OCR
- Redis (optional)
- Docker & Docker Compose (for containerized deployment)

### Option 1: Docker Deployment (Recommended)

1. **Clone the repository**:
```bash
git clone <repository-url>
cd kenyan-doc-processor