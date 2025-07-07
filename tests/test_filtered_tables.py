"""
Test the filtered lookup tables functionality.
"""

import sys
sys.path.append('/scratch/edk202/word2gm-fast/src')

import os
import tempfile
import tensorflow as tf
from word2gm_fast.io.tables import get_unique_indices_from_triplets
from word2gm_fast.io.triplets import write_triplets_to_tfrecord
from word2gm_fast.io.vocab import write_vocab_to_tfrecord


def create_test_data():
    """Create test TFRecord files for testing."""
    
    # Create test vocabulary
    vocab_list = ["UNK", "the", "and", "python", "algorithm", "unused_token"]
    vocab_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(vocab_list),
            values=tf.constant(list(range(len(vocab_list))), dtype=tf.int64)
        ),
        default_value=0
    )
    
    # Create test frequencies
    frequencies = {
        "UNK": 0,
        "the": 1000,
        "and": 800,
        "python": 100,
        "algorithm": 50,
        "unused_token": 5  # This token won't appear in triplets
    }
    
    # Create test triplets (note: "unused_token" index 5 doesn't appear)
    test_triplets = [
        (1, 2, 3),  # "the", "and", "python" 
        (2, 3, 4),  # "and", "python", "algorithm"
        (3, 4, 1),  # "python", "algorithm", "the"
        (4, 1, 2),  # "algorithm", "the", "and"
        (0, 1, 2),  # "UNK", "the", "and"
    ]
    
    return vocab_table, frequencies, test_triplets


def test_unique_indices_extraction():
    """Test extracting unique indices from triplets."""
    
    print("Testing unique indices extraction...")
    
    vocab_table, frequencies, test_triplets = create_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write test triplets
        triplets_path = os.path.join(temp_dir, "test_triplets.tfrecord")
        triplets_ds = tf.data.Dataset.from_tensor_slices(test_triplets)
        count = write_triplets_to_tfrecord(triplets_ds, triplets_path)
        
        print(f"  Created {count} test triplets")
        
        # Extract unique indices
        unique_indices = get_unique_indices_from_triplets(triplets_path)
        
        print(f"  Found unique indices: {sorted(unique_indices)}")
        
        # Verify results
        expected_indices = {0, 1, 2, 3, 4}  # All except 5 (unused_token)
        assert unique_indices == expected_indices, f"Expected {expected_indices}, got {unique_indices}"
        
        print("  ✓ Unique indices extraction test passed!")


def test_table_filtering():
    """Test that tables are properly filtered."""
    
    print("Testing table filtering...")
    
    # Import here to avoid circular imports
    from word2gm_fast.io.tables import create_token_to_index_table, create_index_to_token_table
    
    vocab_table, frequencies, test_triplets = create_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write test data
        vocab_path = os.path.join(temp_dir, "test_vocab.tfrecord")
        triplets_path = os.path.join(temp_dir, "test_triplets.tfrecord")
        
        write_vocab_to_tfrecord(vocab_table, vocab_path, frequencies=frequencies)
        triplets_ds = tf.data.Dataset.from_tensor_slices(test_triplets)
        write_triplets_to_tfrecord(triplets_ds, triplets_path)
        
        # Test unfiltered table
        unfiltered_table = create_token_to_index_table(vocab_path)
        
        # Test filtered table
        filtered_table = create_token_to_index_table(
            vocab_path, 
            triplets_tfrecord_path=triplets_path
        )
        
        # Test lookups
        test_tokens = ["the", "python", "unused_token"]
        
        print("  Token lookup comparison:")
        for token in test_tokens:
            unfiltered_result = unfiltered_table.lookup(tf.constant([token]))
            filtered_result = filtered_table.lookup(tf.constant([token]))
            
            print(f"    '{token}': unfiltered={unfiltered_result.numpy()[0]}, filtered={filtered_result.numpy()[0]}")
            
            # "unused_token" should return default value (0) in filtered table
            if token == "unused_token":
                assert filtered_result.numpy()[0] == 0, f"Expected default value 0 for unused token"
            else:
                # Other tokens should have same index in both tables
                assert unfiltered_result.numpy()[0] == filtered_result.numpy()[0], f"Mismatch for token {token}"
        
        print("  ✓ Table filtering test passed!")


if __name__ == "__main__":
    print("Running filtered lookup tables tests...")
    print("=" * 40)
    
    try:
        test_unique_indices_extraction()
        print()
        test_table_filtering()
        print()
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
