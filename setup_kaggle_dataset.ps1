# PowerShell script to download APTOS 2019 dataset from Kaggle
Write-Host "========================================" -ForegroundColor Green
Write-Host "Kaggle Dataset Download Setup" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Check if Kaggle is installed
Write-Host "`n[1/4] Checking Kaggle API..." -ForegroundColor Cyan
$kaggleCheck = pip show kaggle 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[INFO] Installing Kaggle API..." -ForegroundColor Yellow
    pip install kaggle
} else {
    Write-Host "[OK] Kaggle API is installed" -ForegroundColor Green
}

# Check for credentials
Write-Host "`n[2/4] Checking Kaggle credentials..." -ForegroundColor Cyan
$kagglePath = "$env:USERPROFILE\.kaggle\kaggle.json"
if (-not (Test-Path $kagglePath)) {
    Write-Host "[WARNING] Kaggle credentials not found!" -ForegroundColor Yellow
    Write-Host "`nTo download from Kaggle:" -ForegroundColor Yellow
    Write-Host "1. Go to: https://www.kaggle.com/settings" -ForegroundColor White
    Write-Host "2. Click 'Create New API Token'" -ForegroundColor White
    Write-Host "3. Download kaggle.json" -ForegroundColor White
    Write-Host "4. Place in: $kagglePath" -ForegroundColor White
    Write-Host "`nOR download manually from:" -ForegroundColor Yellow
    Write-Host "https://www.kaggle.com/c/aptos2019-blindness-detection/data`n" -ForegroundColor Cyan
} else {
    Write-Host "[OK] Kaggle credentials found" -ForegroundColor Green
}

# Create directories
Write-Host "`n[3/4] Creating directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "data\raw\normal" | Out-Null
New-Item -ItemType Directory -Force -Path "data\raw\disease" | Out-Null
New-Item -ItemType Directory -Force -Path "data\raw\temp_downloads" | Out-Null
Write-Host "[OK] Directories created" -ForegroundColor Green

# Instructions
Write-Host "`n[4/4] Download Instructions:" -ForegroundColor Cyan
Write-Host "`nTo download APTOS 2019 dataset:" -ForegroundColor Yellow
Write-Host "`nOption A - With Kaggle API:" -ForegroundColor White
Write-Host "  kaggle competitions download -c aptos2019-blindness-detection" -ForegroundColor Cyan
Write-Host "  python utils\download_real_dataset.py --organize data\raw\temp_downloads" -ForegroundColor Cyan

Write-Host "`nOption B - Manual Download:" -ForegroundColor White
Write-Host "  1. Visit: https://www.kaggle.com/c/aptos2019-blindness-detection/data" -ForegroundColor Cyan
Write-Host "  2. Download train_images.zip and train.csv" -ForegroundColor Cyan
Write-Host "  3. Extract to data\raw\temp_downloads\" -ForegroundColor Cyan
Write-Host "  4. Run: python utils\download_real_dataset.py --organize data\raw\temp_downloads" -ForegroundColor Cyan

Write-Host "`n========================================" -ForegroundColor Green

