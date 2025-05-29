# PowerShell script to view logs
param(
    [string]$Service = "api",
    [int]$Lines = 50
)

Write-Host "Showing logs for $Service service (last $Lines lines)..." -ForegroundColor Green
docker-compose -f docker/docker-compose.simple.yml logs --tail=$Lines $Service