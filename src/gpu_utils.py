"""
GPU utility functions for TensorFlow
Automatically detects and configures GPU/CPU usage
"""
import tensorflow as tf
import os

def setup_gpu():
    """
    Setup GPU configuration for TensorFlow
    Returns device info and configures TensorFlow accordingly
    """
    print("=" * 60)
    print("Checking GPU Availability...")
    print("=" * 60)
    
    # Check for GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    cuda_built = tf.test.is_built_with_cuda()
    
    device_info = {
        'gpu_available': False,
        'gpu_count': 0,
        'device_name': 'CPU',
        'cuda_built': cuda_built,
        'gpus': []
    }
    
    if len(gpus) > 0 and cuda_built:
        # GPU is available
        print(f"[SUCCESS] Found {len(gpus)} GPU device(s)")
        
        for i, gpu in enumerate(gpus):
            try:
                # Enable memory growth to avoid allocating all GPU memory
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  GPU {i}: {gpu.name}")
                
                # Get GPU details
                gpu_details = tf.config.experimental.get_device_details(gpu)
                if 'device_name' in gpu_details:
                    print(f"    Device: {gpu_details['device_name']}")
                
                device_info['gpus'].append({
                    'name': gpu.name,
                    'details': gpu_details
                })
            except RuntimeError as e:
                print(f"  [WARNING] Error configuring GPU {i}: {str(e)}")
        
        device_info['gpu_available'] = True
        device_info['gpu_count'] = len(gpus)
        device_info['device_name'] = f"GPU ({len(gpus)} device(s))"
        
        # Set mixed precision for faster training (optional)
        # Uncomment if you want to use mixed precision training
        # policy = tf.keras.mixed_precision.Policy('mixed_float16')
        # tf.keras.mixed_precision.set_global_policy(policy)
        # print("  [INFO] Mixed precision enabled for faster training")
        
        print("\n[INFO] Training will use GPU acceleration")
        print("=" * 60)
        
    else:
        # No GPU available, use CPU
        print("[INFO] No GPU detected or TensorFlow built without CUDA support")
        
        if not cuda_built:
            print("  TensorFlow was not built with CUDA support")
            print("  To enable GPU support, install tensorflow-gpu or tensorflow[and-cuda]")
        else:
            print("  No GPU devices found")
            print("  Check:")
            print("    - NVIDIA drivers installed")
            print("    - CUDA toolkit installed")
            print("    - cuDNN installed")
        
        print("\n[INFO] Training will use CPU")
        print("=" * 60)
    
    return device_info

def get_device_strategy():
    """
    Get the appropriate distribution strategy for training
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if len(gpus) > 1:
        # Multiple GPUs - use MirroredStrategy
        print(f"[INFO] Using MirroredStrategy for {len(gpus)} GPUs")
        return tf.distribute.MirroredStrategy()
    elif len(gpus) == 1:
        # Single GPU - default strategy
        return None
    else:
        # CPU - default strategy
        return None

def print_gpu_info():
    """
    Print detailed GPU information
    """
    print("\n" + "=" * 60)
    print("TensorFlow GPU Information")
    print("=" * 60)
    
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"CUDA Built: {tf.test.is_built_with_cuda()}")
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU Devices Found: {len(gpus)}")
    
    if len(gpus) > 0:
        for i, gpu in enumerate(gpus):
            print(f"\nGPU {i}:")
            print(f"  Name: {gpu.name}")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                for key, value in details.items():
                    print(f"  {key}: {value}")
            except:
                print("  Details: Available")
    
    print("=" * 60 + "\n")

if __name__ == "__main__":
    # Test GPU setup
    device_info = setup_gpu()
    print_gpu_info()
    print(f"\nDevice Info: {device_info}")

