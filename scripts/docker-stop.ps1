# PowerShell script to stop Docker services
Write-Host "Stopping Kenyan Document Processing System..." -ForegroundColor Yellow

docker-compose -f docker/docker-compose.simple.yml down

Write-Host "All services stopped." -ForegroundColor Green