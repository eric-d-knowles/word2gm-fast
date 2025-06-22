"""
Unit tests for TFRecord I/O utilities.

Tests all functionality in src.word2gm_fast.dataprep.tfrecord_io including:
- Triplet serialization and loading
- Vocabulary serialization and loading (including optimized version)
- Pipeline artifact save/load
- Error handling and edge cases
"""

import unittest
import tempfile
import shutil
import os
import tensorflow as tf
import numpy as np
from typing import List, Tuple

# Import the module under test
from src.word2gm_fast.dataprep.tfrecord_io import (
    write_triplets_to_tfrecord,
    load_triplets_from_tfrecord,
    parse_triplet_example,
    write_vocab_to_tfrecord,
    load_vocab_from_tfrecord,
    parse_vocab_example,
    save_pipeline_artifacts,
    load_pipeline_artifacts
)


class TestTFRecordIO(unittest.TestCase):
    """Test cases for TFRecord I/O functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample vocabulary data
        self.vocab_words = ["UNK", "the", "man", "king", "queen", "word"]
        self.vocab_ids = [0, 1, 2, 3, 4, 5]
        
        # Create vocabulary lookup table
        self.vocab_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(self.vocab_words),
                values=tf.constant(self.vocab_ids, dtype=tf.int64)
            ),
            default_value=0
        )
        
        # Create sample triplet data
        self.triplet_data = [
            (1, 2, 3),  # (center, positive, negative)
            (2, 1, 4),
            (3, 4, 5),
            (4, 3, 1),
            (5, 2, 3)
        ]
        
        # Create triplet dataset
        self.triplets_dataset = tf.data.Dataset.from_tensor_slices(
            [tf.constant(triplet, dtype=tf.int64) for triplet in self.triplet_data]
        )
        
        # Create sample text dataset for pipeline testing
        sample_lines = ["the king is great", "the queen rules", "word embeddings"]
        self.text_dataset = tf.data.Dataset.from_tensor_slices(
            [line.encode('utf-8') for line in sample_lines]
        )
    
    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory and all its contents
        shutil.rmtree(self.test_dir)
    
    # Test triplet serialization and loading
    
    def test_write_and_load_triplets_uncompressed(self):
        """Test writing and loading triplets without compression."""
        output_path = os.path.join(self.test_dir, "triplets.tfrecord")
        
        # Write triplets
        write_triplets_to_tfrecord(self.triplets_dataset, output_path, compress=False)
        
        # Verify file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Load triplets
        loaded_dataset = load_triplets_from_tfrecord(output_path, compressed=False)
        
        # Verify content
        loaded_triplets = list(loaded_dataset.as_numpy_iterator())
        self.assertEqual(len(loaded_triplets), len(self.triplet_data))
        
        for original, loaded in zip(self.triplet_data, loaded_triplets):
            center, positive, negative = loaded
            self.assertEqual(original[0], center)
            self.assertEqual(original[1], positive)
            self.assertEqual(original[2], negative)
    
    def test_write_and_load_triplets_compressed(self):
        """Test writing and loading triplets with compression."""
        output_path = os.path.join(self.test_dir, "triplets.tfrecord.gz")
        
        # Write triplets
        write_triplets_to_tfrecord(self.triplets_dataset, output_path, compress=True)
        
        # Verify file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Load triplets (auto-detect compression)
        loaded_dataset = load_triplets_from_tfrecord(output_path)
        
        # Verify content
        loaded_triplets = list(loaded_dataset.as_numpy_iterator())
        self.assertEqual(len(loaded_triplets), len(self.triplet_data))
    
    def test_parse_triplet_example(self):
        """Test parsing individual triplet examples."""
        # Create a test example
        center, positive, negative = 1, 2, 3
        example = tf.train.Example(features=tf.train.Features(feature={
            'center': tf.train.Feature(int64_list=tf.train.Int64List(value=[center])),
            'positive': tf.train.Feature(int64_list=tf.train.Int64List(value=[positive])),
            'negative': tf.train.Feature(int64_list=tf.train.Int64List(value=[negative])),
        }))
        
        # Parse the example
        serialized = example.SerializeToString()
        parsed_center, parsed_positive, parsed_negative = parse_triplet_example(serialized)
        
        # Verify parsed values
        self.assertEqual(center, parsed_center.numpy())
        self.assertEqual(positive, parsed_positive.numpy())
        self.assertEqual(negative, parsed_negative.numpy())
    
    # Test vocabulary serialization and loading
    
    def test_write_and_load_vocab_uncompressed(self):
        """Test writing and loading vocabulary without compression."""
        output_path = os.path.join(self.test_dir, "vocab.tfrecord")
        
        # Write vocabulary
        write_vocab_to_tfrecord(self.vocab_table, output_path, compress=False)
        
        # Verify file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Load vocabulary
        loaded_vocab_table = load_vocab_from_tfrecord(output_path, compressed=False)
        
        # Verify size
        self.assertEqual(self.vocab_table.size().numpy(), loaded_vocab_table.size().numpy())
        
        # Test lookups
        test_words = tf.constant(self.vocab_words)
        original_ids = self.vocab_table.lookup(test_words).numpy()
        loaded_ids = loaded_vocab_table.lookup(test_words).numpy()
        
        np.testing.assert_array_equal(original_ids, loaded_ids)
    
    def test_write_and_load_vocab_compressed(self):
        """Test writing and loading vocabulary with compression."""
        output_path = os.path.join(self.test_dir, "vocab.tfrecord.gz")
        
        # Write vocabulary
        write_vocab_to_tfrecord(self.vocab_table, output_path, compress=True)
        
        # Verify file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Load vocabulary (auto-detect compression)
        loaded_vocab_table = load_vocab_from_tfrecord(output_path)
        
        # Test lookups
        test_words = tf.constant(["UNK", "the", "nonexistentword"])
        original_ids = self.vocab_table.lookup(test_words).numpy()
        loaded_ids = loaded_vocab_table.lookup(test_words).numpy()
        
        np.testing.assert_array_equal(original_ids, loaded_ids)
    
    def test_vocab_optimized_batch_sizes(self):
        """Test optimized vocabulary loading with different batch sizes."""
        output_path = os.path.join(self.test_dir, "vocab_batch_test.tfrecord.gz")
        
        # Write vocabulary
        write_vocab_to_tfrecord(self.vocab_table, output_path, compress=True)
        
        # Test different batch sizes
        for batch_size in [1, 2, 5, 10, 1000]:
            loaded_vocab_table = load_vocab_from_tfrecord(
                output_path, batch_size=batch_size
            )
            
            # Verify correctness
            test_words = tf.constant(self.vocab_words)
            original_ids = self.vocab_table.lookup(test_words).numpy()
            loaded_ids = loaded_vocab_table.lookup(test_words).numpy()
            
            np.testing.assert_array_equal(original_ids, loaded_ids, 
                                        err_msg=f"Failed with batch_size={batch_size}")
    
    def test_parse_vocab_example(self):
        """Test parsing individual vocabulary examples."""
        # Create a test example
        word = "test"
        word_id = 42
        example = tf.train.Example(features=tf.train.Features(feature={
            'word': tf.train.Feature(bytes_list=tf.train.BytesList(value=[word.encode('utf-8')])),
            'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[word_id])),
        }))
        
        # Parse the example
        serialized = example.SerializeToString()
        parsed_word, parsed_id = parse_vocab_example(serialized)
        
        # Verify parsed values
        self.assertEqual(word.encode('utf-8'), parsed_word.numpy())
        self.assertEqual(word_id, parsed_id.numpy())
    
    # Test pipeline artifact save/load
    
    def test_save_and_load_pipeline_artifacts_compressed(self):
        """Test saving and loading complete pipeline artifacts with compression."""
        # Save artifacts
        artifacts = save_pipeline_artifacts(
            self.text_dataset, 
            self.vocab_table, 
            self.triplets_dataset, 
            self.test_dir, 
            compress=True
        )
        
        # Verify return values
        self.assertIn('vocab_path', artifacts)
        self.assertIn('triplets_path', artifacts)
        self.assertIn('vocab_size', artifacts)
        self.assertTrue(artifacts['compressed'])
        
        # Verify files exist
        self.assertTrue(os.path.exists(artifacts['vocab_path']))
        self.assertTrue(os.path.exists(artifacts['triplets_path']))
        
        # Load artifacts
        loaded_artifacts = load_pipeline_artifacts(self.test_dir, compressed=True)
        
        # Verify loaded artifacts
        self.assertIn('vocab_table', loaded_artifacts)
        self.assertIn('triplets_ds', loaded_artifacts)
        self.assertIn('vocab_size', loaded_artifacts)
        
        # Test vocabulary integrity
        test_words = tf.constant(self.vocab_words)
        original_ids = self.vocab_table.lookup(test_words).numpy()
        loaded_ids = loaded_artifacts['vocab_table'].lookup(test_words).numpy()
        np.testing.assert_array_equal(original_ids, loaded_ids)
        
        # Test triplets integrity
        original_triplets = list(self.triplets_dataset.as_numpy_iterator())
        loaded_triplets = list(loaded_artifacts['triplets_ds'].as_numpy_iterator())
        
        self.assertEqual(len(original_triplets), len(loaded_triplets))
        for orig, loaded in zip(original_triplets, loaded_triplets):
            np.testing.assert_array_equal(orig, loaded)
    
    def test_save_and_load_pipeline_artifacts_uncompressed(self):
        """Test saving and loading complete pipeline artifacts without compression."""
        # Save artifacts
        artifacts = save_pipeline_artifacts(
            self.text_dataset, 
            self.vocab_table, 
            self.triplets_dataset, 
            self.test_dir, 
            compress=False
        )
        
        # Verify compression setting
        self.assertFalse(artifacts['compressed'])
        
        # Load artifacts
        loaded_artifacts = load_pipeline_artifacts(self.test_dir, compressed=False)
        
        # Basic integrity check
        self.assertEqual(
            self.vocab_table.size().numpy(), 
            loaded_artifacts['vocab_table'].size().numpy()
        )
    
    def test_load_pipeline_artifacts_auto_detect_compression(self):
        """Test auto-detection of compression when loading pipeline artifacts."""
        # Save compressed artifacts
        save_pipeline_artifacts(
            self.text_dataset, 
            self.vocab_table, 
            self.triplets_dataset, 
            self.test_dir, 
            compress=True
        )
        
        # Load with auto-detection (compressed=None)
        loaded_artifacts = load_pipeline_artifacts(self.test_dir)
        
        # Verify successful load
        self.assertIn('vocab_table', loaded_artifacts)
        self.assertIn('triplets_ds', loaded_artifacts)
    
    # Test error handling and edge cases
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file raises appropriate error."""
        nonexistent_path = os.path.join(self.test_dir, "nonexistent.tfrecord")
        
        with self.assertRaises(tf.errors.NotFoundError):
            list(load_triplets_from_tfrecord(nonexistent_path).take(1))
    
    def test_load_pipeline_artifacts_missing_files(self):
        """Test loading pipeline artifacts when files are missing."""
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir)
        
        with self.assertRaises(FileNotFoundError):
            load_pipeline_artifacts(empty_dir)
    
    def test_vocab_default_values(self):
        """Test vocabulary lookup with default values for unknown words."""
        output_path = os.path.join(self.test_dir, "vocab_default.tfrecord")
        
        # Write and load vocabulary
        write_vocab_to_tfrecord(self.vocab_table, output_path)
        loaded_vocab_table = load_vocab_from_tfrecord(output_path, default_value=99)
        
        # Test unknown word lookup
        unknown_words = tf.constant(["unknown1", "unknown2"])
        ids = loaded_vocab_table.lookup(unknown_words).numpy()
        
        # Should return default value (99) for unknown words
        np.testing.assert_array_equal(ids, [99, 99])
    
    def test_empty_datasets(self):
        """Test handling of empty datasets."""
        # Create empty datasets
        empty_triplets = tf.data.Dataset.from_tensor_slices(
            tf.constant([], dtype=tf.int64, shape=[0, 3])
        )
        
        # Test writing empty triplets (should not crash)
        output_path = os.path.join(self.test_dir, "empty_triplets.tfrecord")
        write_triplets_to_tfrecord(empty_triplets, output_path)
        
        # Load and verify empty
        loaded_empty = load_triplets_from_tfrecord(output_path)
        loaded_list = list(loaded_empty.as_numpy_iterator())
        self.assertEqual(len(loaded_list), 0)
    
    # Performance and optimization tests
    
    def test_large_vocabulary_performance(self):
        """Test performance with larger vocabulary (regression test for optimization)."""
        # Create larger vocabulary
        large_vocab_words = [f"word_{i}" for i in range(1000)]
        large_vocab_ids = list(range(1000))
        
        large_vocab_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(large_vocab_words),
                values=tf.constant(large_vocab_ids, dtype=tf.int64)
            ),
            default_value=0
        )
        
        output_path = os.path.join(self.test_dir, "large_vocab.tfrecord.gz")
        
        # Time the operations
        import time
        
        # Write
        start = time.time()
        write_vocab_to_tfrecord(large_vocab_table, output_path, compress=True)
        write_time = time.time() - start
        
        # Load with optimization (batched)
        start = time.time()
        loaded_table = load_vocab_from_tfrecord(output_path, batch_size=100)
        load_time = time.time() - start
        
        # Verify correctness
        test_sample = tf.constant(large_vocab_words[:10])
        original_ids = large_vocab_table.lookup(test_sample).numpy()
        loaded_ids = loaded_table.lookup(test_sample).numpy()
        np.testing.assert_array_equal(original_ids, loaded_ids)
        
        # Performance should be reasonable (not strict timing test, just sanity check)
        self.assertLess(write_time, 30.0, "Write time too slow")
        self.assertLess(load_time, 5.0, "Load time too slow (optimization may have regressed)")
    
    def test_compression_effectiveness(self):
        """Test that compression actually reduces file size."""
        # Use same data for both
        triplet_path_uncompressed = os.path.join(self.test_dir, "triplets.tfrecord")
        triplet_path_compressed = os.path.join(self.test_dir, "triplets.tfrecord.gz")
        
        # Create larger dataset for better compression test
        large_triplets = tf.data.Dataset.from_tensor_slices([
            tf.constant([i % 10, (i + 1) % 10, (i + 2) % 10], dtype=tf.int64)
            for i in range(1000)
        ])
        
        # Write uncompressed
        write_triplets_to_tfrecord(large_triplets, triplet_path_uncompressed, compress=False)
        
        # Write compressed
        write_triplets_to_tfrecord(large_triplets, triplet_path_compressed, compress=True)
        
        # Check file sizes
        uncompressed_size = os.path.getsize(triplet_path_uncompressed)
        compressed_size = os.path.getsize(triplet_path_compressed)
        
        # Compressed should be smaller
        self.assertLess(compressed_size, uncompressed_size, 
                       "Compression should reduce file size")


if __name__ == '__main__':
    # Configure TensorFlow to reduce log output during tests
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    
    # Run the tests
    unittest.main(verbosity=2)
