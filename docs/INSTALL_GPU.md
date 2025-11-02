# GPU Support Installation Guide

## ✅ GPU Detected!

You have an **NVIDIA GeForce RTX 2050** with CUDA 12.7!

## Quick Install

Run the batch script:
```bash
install_gpu_support.bat
```

Or manually:
```bash
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

## Verify Installation

After installation, check GPU support:
```bash
python src\gpu_utils.py
```

You should see:
```
[SUCCESS] Found 1 GPU device(s)
  GPU 0: /physical_device:GPU:0
```

## Performance Improvement

With GPU support enabled:
- **Training speed**: 10-50x faster than CPU
- **Batch size**: Can use larger batches (32-64)
- **Efficiency**: Better resource utilization

## Current Status

- ✅ **GPU Hardware**: NVIDIA GeForce RTX 2050 (4GB VRAM)
- ✅ **CUDA Driver**: Version 12.7
- ✅ **NVIDIA Drivers**: Installed (566.36)
- ❌ **TensorFlow GPU**: Not installed (need tensorflow[and-cuda])

## After Installation

The training script will automatically:
1. Detect your GPU
2. Configure memory growth
3. Use GPU for training
4. Fall back to CPU if GPU unavailable

No code changes needed!

---

**Note**: The project is already configured to use GPU automatically once TensorFlow with CUDA is installed.

