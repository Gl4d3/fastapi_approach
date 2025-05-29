# PowerShell script to fix import issues
Write-Host "Fixing import issues in the codebase..." -ForegroundColor Green

# Stop any running containers
Write-Host "Stopping containers..." -ForegroundColor Yellow
docker-compose -f docker/docker-compose.simple.yml down

# Clean up
docker system prune -f

# Rebuild and start
Write-Host "Rebuilding containers..." -ForegroundColor Yellow
docker-compose -f docker/docker-compose.simple.yml up --build -d

# Wait for startup
Write-Host "Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Test health endpoint
Write-Host "Testing health endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
    Write-Host "✓ Health Check Passed: $($response.status)" -ForegroundColor Green
    Write-Host "Services: $($response.services | ConvertTo-Json)" -ForegroundColor Cyan
} catch {
    Write-Host "✗ Health Check Failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Checking logs..." -ForegroundColor Yellow
    docker-compose -f docker/docker-compose.simple.yml logs api
}

Write-Host "`nAPI Endpoints:" -ForegroundColor Cyan
Write-Host "- Health: http://localhost:8000/health" -ForegroundColor White
Write-Host "- Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "- Supported Docs: http://localhost:8000/supported-documents" -ForegroundColor White