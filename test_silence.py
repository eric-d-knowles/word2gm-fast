#!/usr/bin/env python3
"""
Test script to verify TensorFlow silencing works.
"""

import os
import sys

# Add src to path
sys.path.insert(0, '/scratch/edk202/word2gm-fast/src')

from word2gm_fast.utils.tf_silence import import_tf_quietly, silence_tensorflow

def test_basic_import():
    """Test basic TensorFlow import silencing."""
    print("Testing basic TensorFlow import...")
    tf = import_tf_quietly(force_cpu=True)
    print(f"✅ TensorFlow imported successfully (version {tf.__version__})")
    return tf

def test_context_silence():
    """Test context manager silencing."""
    print("\nTesting context manager silence...")
    tf = import_tf_quietly(force_cpu=True)
    
    print("Creating dataset operations with silence...")
    with silence_tensorflow():
        # Create a simple dataset that should trigger C++ messages
        dataset = tf.data.Dataset.range(10)
        dataset = dataset.batch(2)
        
        # Iterate through it (this often triggers "OUT_OF_RANGE" messages)
        count = 0
        for batch in dataset:
            count += 1
        
        print(f"✅ Processed {count} batches silently")

def test_pipeline_import():
    """Test importing pipeline components."""
    print("\nTesting pipeline component imports...")
    try:
        from word2gm_fast.dataprep.simple_pipeline import process_single_year
        print("✅ Pipeline imports successful")
    except Exception as e:
        print(f"❌ Pipeline import failed: {e}")

if __name__ == "__main__":
    print("TensorFlow Silence Test")
    print("=" * 40)
    
    test_basic_import()
    test_context_silence()
    test_pipeline_import()
    
    print("\n✅ All tests completed")
