# GPU Setup Guide

## Automatic GPU Detection

The project now automatically detects and uses GPU if available, otherwise falls back to CPU.

## Current Setup

The `src/gpu_utils.py` module handles:
- ✅ Automatic GPU detection
- ✅ GPU memory growth configuration (prevents OOM errors)
- ✅ Multi-GPU support (MirroredStrategy)
- ✅ CPU fallback if GPU unavailable

## Checking GPU Status

Run:
```bash
python src/gpu_utils.py
```

Or check in training:
```bash
python src/train.py
```
(The training script will automatically show GPU/CPU status)

## Installing GPU Support

If you have an NVIDIA GPU but TensorFlow isn't detecting it:

### Option 1: Install TensorFlow with CUDA (Recommended)
```bash
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

### Option 2: Check CUDA Installation
1. **NVIDIA Drivers**: Must be installed
2. **CUDA Toolkit**: Version 11.8 or 12.x recommended
3. **cuDNN**: Must match CUDA version

### Verify CUDA Installation
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA (if installed)
nvcc --version
```

### Check TensorFlow GPU Support
```python
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

## Performance Tips

### With GPU:
- Training will be **significantly faster** (10-50x speedup)
- Batch size can be increased
- More epochs can be run efficiently

### With CPU:
- Training will work but be slower
- Smaller batch sizes recommended
- Consider reducing epochs or model complexity

## Current Status

Based on the check:
- **TensorFlow Version**: 2.20.0
- **CUDA Built**: No
- **GPU Detected**: 0 devices

**Status**: Running on CPU

## To Enable GPU:

1. Install NVIDIA drivers
2. Install CUDA Toolkit (11.8 or 12.x)
3. Install cuDNN
4. Reinstall TensorFlow with GPU support:
   ```bash
   pip install tensorflow[and-cuda]
   ```

Or use tensorflow-gpu (if available):
```bash
pip install tensorflow-gpu
```

## Testing GPU

After installation, test with:
```bash
python src/gpu_utils.py
```

You should see:
```
[SUCCESS] Found 1 GPU device(s)
  GPU 0: /physical_device:GPU:0
    Device: <your GPU name>
```

---

**Note**: The training script will automatically use GPU if available, no code changes needed!

