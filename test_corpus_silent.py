#!/usr/bin/env python3
"""
Test corpus_to_dataset with silent TensorFlow import.
"""

import os
import sys

# Add src to path
sys.path.insert(0, '/scratch/edk202/word2gm-fast/src')

def test_corpus_to_dataset_import():
    """Test importing corpus_to_dataset with silent TF."""
    print("Testing corpus_to_dataset import with silent TensorFlow...")
    
    try:
        from word2gm_fast.dataprep.corpus_to_dataset import make_dataset, validate_5gram_line
        print("✅ corpus_to_dataset imported successfully")
        
        # Test that TensorFlow operations work
        from word2gm_fast.utils.tf_silence import import_tf_quietly
        tf = import_tf_quietly(force_cpu=True)
        
        # Test the validation function with a simple tensor
        test_line = tf.constant("word1 word2 center word4 word5")
        line, is_valid = validate_5gram_line(test_line)
        print(f"✅ validate_5gram_line works: is_valid = {is_valid.numpy()}")
        
    except Exception as e:
        print(f"❌ Import or function test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing corpus_to_dataset with silent TensorFlow")
    print("=" * 50)
    
    success = test_corpus_to_dataset_import()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
