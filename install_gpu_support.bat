@echo off
echo ========================================
echo Installing TensorFlow with GPU Support
echo ========================================
echo.
echo You have NVIDIA GeForce RTX 2050 detected!
echo CUDA Version: 12.7
echo.
echo Installing TensorFlow with CUDA support...
echo.

pip uninstall tensorflow -y
pip install tensorflow[and-cuda]

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Verify GPU support with:
echo   python src\gpu_utils.py
echo.
pause

