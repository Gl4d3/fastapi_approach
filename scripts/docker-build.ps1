# PowerShell script to build and run Docker containers
Write-Host "Building Kenyan Document Processing System..." -ForegroundColor Green

# Clean up previous containers
Write-Host "Cleaning up previous containers..." -ForegroundColor Yellow
docker-compose -f docker/docker-compose.simple.yml down
docker system prune -f

# Build and start services
Write-Host "Building and starting services..." -ForegroundColor Yellow
docker-compose -f docker/docker-compose.simple.yml up --build -d

# Wait for services to start
Write-Host "Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Check health
Write-Host "Checking service health..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
    Write-Host "✓ API Health Check: $($response.status)" -ForegroundColor Green
} catch {
    Write-Host "✗ API Health Check Failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Show logs
Write-Host "Showing recent logs..." -ForegroundColor Cyan
docker-compose -f docker/docker-compose.simple.yml logs --tail=20

Write-Host "`nServices running:" -ForegroundColor Green
Write-Host "API: http://localhost:8000" -ForegroundColor White
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "Health Check: http://localhost:8000/health" -ForegroundColor White