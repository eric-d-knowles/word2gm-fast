#!/usr/bin/env python3
"""
Quick GPU test script
"""
import os

# Set up GPU environment
print("Setting up GPU environment...")
accessible_gpu = None
for i in range(4):
    device_path = f"/dev/nvidia{i}"
    try:
        with open(device_path, 'rb') as f:
            pass
        accessible_gpu = i
        print(f"Found accessible GPU: {i}")
        break
    except (PermissionError, FileNotFoundError):
        continue

if accessible_gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(accessible_gpu)
    print(f"Set CUDA_VISIBLE_DEVICES={accessible_gpu}")

# Import TensorFlow
print("Importing TensorFlow...")
from src.word2gm_fast.utils import import_tensorflow_silently
tf = import_tensorflow_silently(deterministic=False, force_cpu=False)

# Test GPU detection
print(f"\nTensorFlow {tf.__version__}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

gpus = tf.config.list_physical_devices('GPU')
print(f"Physical GPUs detected: {len(gpus)}")

if gpus:
    print("✅ SUCCESS: GPU detected!")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    
    # Test computation
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
        print(f"✅ GPU computation successful: {c.numpy()}")
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
else:
    print("❌ No GPUs detected")
