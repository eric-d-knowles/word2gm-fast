#!/usr/bin/env python3
"""
Simple test script for Word2GM training to verify the implementation works.
"""

import os
import sys
from pathlib import Path

# Setup project path
project_root = Path(__file__).parent
os.chdir(project_root)
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print(f"Project root: {project_root}")
print(f"Source path: {src_path}")
print(f"Python path: {sys.path[:3]}...")

# Configure for CPU-only testing to avoid GPU issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        from word2gm_fast.utils.tf_silence import import_tensorflow_silently
        tf = import_tensorflow_silently(force_cpu=True, gpu_memory_growth=False)
        print(f"✓ TensorFlow {tf.__version__} imported successfully")
        
        from word2gm_fast.models.word2gm_model import Word2GMModel
        from word2gm_fast.models.config import Word2GMConfig
        print("✓ Model modules imported successfully")
        
        from word2gm_fast.dataprep.tfrecord_io import load_triplets_from_tfrecord, load_vocab_from_tfrecord
        print("✓ Data I/O modules imported successfully")
        
        from word2gm_fast.training.training_utils import train_step, summarize_dataset_pipeline
        print("✓ Training utilities imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_creation():
    """Test model creation and basic functionality."""
    print("\nTesting model creation...")
    
    try:
        from word2gm_fast.models.config import Word2GMConfig
        from word2gm_fast.models.word2gm_model import Word2GMModel
        from word2gm_fast.utils.tf_silence import import_tensorflow_silently
        
        tf = import_tensorflow_silently(force_cpu=True, gpu_memory_growth=False)
        
        # Create small test model
        config = Word2GMConfig(
            vocab_size=1000,
            embedding_size=50,
            num_mixtures=2,
            batch_size=32,
            epochs_to_train=1
        )
        
        model = Word2GMModel(config)
        print(f"✓ Model created with {model.count_params():,} parameters")
        
        # Test forward pass
        test_word_ids = tf.constant([0, 1, 2])
        test_pos_ids = tf.constant([1, 2, 3])
        test_neg_ids = tf.constant([4, 5, 6])
        
        loss = model((test_word_ids, test_pos_ids, test_neg_ids), training=True)
        print(f"✓ Forward pass successful, loss: {loss:.6f}")
        
        # Test word distribution extraction
        mus, vars, weights = model.get_word_distributions(test_word_ids)
        print(f"✓ Word distributions: μ{mus.shape}, σ²{vars.shape}, π{weights.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """Test training step functionality."""
    print("\nTesting training step...")
    
    try:
        from word2gm_fast.models.config import Word2GMConfig
        from word2gm_fast.models.word2gm_model import Word2GMModel
        from word2gm_fast.training.training_utils import train_step
        from word2gm_fast.utils.tf_silence import import_tensorflow_silently
        
        tf = import_tensorflow_silently(force_cpu=True, gpu_memory_growth=False)
        
        # Create test model and optimizer
        config = Word2GMConfig(vocab_size=100, embedding_size=20, num_mixtures=2)
        model = Word2GMModel(config)
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.05)
        
        # Test data
        word_ids = tf.constant([0, 1, 2, 3])
        pos_ids = tf.constant([1, 2, 3, 4])
        neg_ids = tf.constant([5, 6, 7, 8])
        
        # Training step
        loss, grads = train_step(model, optimizer, word_ids, pos_ids, neg_ids)
        print(f"✓ Training step successful, loss: {loss:.6f}")
        print(f"✓ Gradients computed for {len(grads)} variables")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Word2GM Implementation Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_model_creation,
        test_training_step
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
    
    print(f"\nTest Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Your Word2GM implementation is working correctly.")
        print("\nNext steps:")
        print("1. Run the data preparation notebook: notebooks/prepare_training_dataset.ipynb")
        print("2. Run the training notebook: notebooks/train_word2gm.ipynb")
        print("3. Explore the trained model with the interactive tools")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
