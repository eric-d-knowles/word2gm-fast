"""Unit tests for io.triplets module.

Tests TFRecord serialization and deserialization of integer triplets.
"""
import pytest
import tensorflow as tf
import tempfile
import os
from src.word2gm_fast.io.triplets import (
    write_triplets_to_tfrecord,
    load_triplets_from_tfrecord
)


@pytest.fixture
def sample_integer_triplets():
    """Create a sample dataset of integer triplets."""
    triplets = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [1, 3, 5]
    ]
    return tf.data.Dataset.from_tensor_slices(triplets)


@pytest.fixture
def temp_tfrecord_path():
    """Create a temporary TFRecord file path."""
    with tempfile.NamedTemporaryFile(suffix='.tfrecord', delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def temp_compressed_tfrecord_path():
    """Create a temporary compressed TFRecord file path."""
    with tempfile.NamedTemporaryFile(suffix='.tfrecord.gz', delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


class TestWriteTripletsToTFRecord:
    """Test the write_triplets_to_tfrecord function."""
    
    def test_basic_write_uncompressed(self, sample_integer_triplets, temp_tfrecord_path):
        """Test basic writing of triplets to uncompressed TFRecord."""
        triplet_count = write_triplets_to_tfrecord(
            sample_integer_triplets, 
            temp_tfrecord_path, 
            compress=False
        )
        
        # Check that file was created
        assert os.path.exists(temp_tfrecord_path)
        assert os.path.getsize(temp_tfrecord_path) > 0
        
        # Check triplet count
        assert triplet_count == 5
    
    def test_basic_write_compressed(self, sample_integer_triplets, temp_compressed_tfrecord_path):
        """Test basic writing of triplets to compressed TFRecord."""
        triplet_count = write_triplets_to_tfrecord(
            sample_integer_triplets, 
            temp_compressed_tfrecord_path, 
            compress=True
        )
        
        # Check that file was created
        assert os.path.exists(temp_compressed_tfrecord_path)
        assert os.path.getsize(temp_compressed_tfrecord_path) > 0
        
        # Check triplet count
        assert triplet_count == 5
    
    def test_empty_dataset(self, temp_tfrecord_path):
        """Test writing empty dataset."""
        empty_dataset = tf.data.Dataset.from_tensor_slices(tf.constant([], dtype=tf.int32, shape=[0, 3]))
        
        triplet_count = write_triplets_to_tfrecord(
            empty_dataset, 
            temp_tfrecord_path, 
            compress=False
        )
        
        # Should create file but with zero triplets
        assert os.path.exists(temp_tfrecord_path)
        assert triplet_count == 0
    
    def test_large_integers(self, temp_tfrecord_path):
        """Test with large integer values."""
        large_triplets = [
            [999999, 1000000, 1000001],
            [0, 1, 2],  # Include small values too
            [500000, 600000, 700000]
        ]
        dataset = tf.data.Dataset.from_tensor_slices(large_triplets)
        
        triplet_count = write_triplets_to_tfrecord(dataset, temp_tfrecord_path, compress=False)
        assert triplet_count == 3


class TestLoadTripletsFromTFRecord:
    """Test the load_triplets_from_tfrecord function."""
    
    def test_basic_load_uncompressed(self, sample_integer_triplets, temp_tfrecord_path):
        """Test basic loading from uncompressed TFRecord."""
        # First write the data
        write_triplets_to_tfrecord(sample_integer_triplets, temp_tfrecord_path, compress=False)
        
        # Then load it back
        loaded_dataset = load_triplets_from_tfrecord(temp_tfrecord_path, compressed=False)
        
        # Convert to list for comparison
        original_triplets = list(sample_integer_triplets.as_numpy_iterator())
        loaded_triplets = list(loaded_dataset.as_numpy_iterator())
        
        assert len(loaded_triplets) == len(original_triplets)
        
        # Check each triplet
        for orig, loaded in zip(original_triplets, loaded_triplets):
            assert len(loaded) == 3  # Should be (center, positive, negative)
            center, positive, negative = loaded
            assert list(orig) == [center, positive, negative]
    
    def test_basic_load_compressed(self, sample_integer_triplets, temp_compressed_tfrecord_path):
        """Test basic loading from compressed TFRecord."""
        # First write the data
        write_triplets_to_tfrecord(sample_integer_triplets, temp_compressed_tfrecord_path, compress=True)
        
        # Then load it back
        loaded_dataset = load_triplets_from_tfrecord(temp_compressed_tfrecord_path, compressed=True)
        
        # Convert to list for comparison
        original_triplets = list(sample_integer_triplets.as_numpy_iterator())
        loaded_triplets = list(loaded_dataset.as_numpy_iterator())
        
        assert len(loaded_triplets) == len(original_triplets)
    
    def test_load_empty_file(self, temp_tfrecord_path):
        """Test loading from empty TFRecord file."""
        empty_dataset = tf.data.Dataset.from_tensor_slices(tf.constant([], dtype=tf.int32, shape=[0, 3]))
        write_triplets_to_tfrecord(empty_dataset, temp_tfrecord_path, compress=False)
        
        loaded_dataset = load_triplets_from_tfrecord(temp_tfrecord_path, compressed=False)
        loaded_triplets = list(loaded_dataset)
        
        assert len(loaded_triplets) == 0
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        # TFRecordDataset doesn't raise until you try to iterate
        dataset = load_triplets_from_tfrecord("/nonexistent/file.tfrecord", compressed=False)
        
        # The error should occur when we try to iterate
        with pytest.raises(Exception):  # Should raise some kind of file error when iterating
            list(dataset.take(1))


class TestRoundTripConsistency:
    """Test write-then-load consistency."""
    
    def test_roundtrip_uncompressed(self, sample_integer_triplets, temp_tfrecord_path):
        """Test that write->load preserves data exactly."""
        # Write
        original_count = write_triplets_to_tfrecord(
            sample_integer_triplets, temp_tfrecord_path, compress=False
        )
        
        # Load
        loaded_dataset = load_triplets_from_tfrecord(temp_tfrecord_path, compressed=False)
        
        # Compare
        original_triplets = list(sample_integer_triplets.as_numpy_iterator())
        loaded_triplets = list(loaded_dataset.as_numpy_iterator())
        
        assert len(loaded_triplets) == original_count
        assert len(loaded_triplets) == len(original_triplets)
        
        # Check exact equality
        for orig, loaded in zip(original_triplets, loaded_triplets):
            center, positive, negative = loaded
            assert list(orig) == [center, positive, negative]
    
    def test_roundtrip_compressed(self, sample_integer_triplets, temp_compressed_tfrecord_path):
        """Test that compressed write->load preserves data exactly."""
        # Write
        original_count = write_triplets_to_tfrecord(
            sample_integer_triplets, temp_compressed_tfrecord_path, compress=True
        )
        
        # Load
        loaded_dataset = load_triplets_from_tfrecord(temp_compressed_tfrecord_path, compressed=True)
        
        # Compare
        original_triplets = list(sample_integer_triplets.as_numpy_iterator())
        loaded_triplets = list(loaded_dataset.as_numpy_iterator())
        
        assert len(loaded_triplets) == original_count
        assert len(loaded_triplets) == len(original_triplets)
    
    def test_data_types_preserved(self, temp_tfrecord_path):
        """Test that data types are preserved through round-trip."""
        # Create triplets with specific dtypes
        triplets = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices(triplets)
        
        # Write and load
        write_triplets_to_tfrecord(dataset, temp_tfrecord_path, compress=False)
        loaded_dataset = load_triplets_from_tfrecord(temp_tfrecord_path, compressed=False)
        
        # Check data types in loaded dataset
        for triplet in loaded_dataset.take(1):
            center, positive, negative = triplet
            assert center.dtype == tf.int64  # TFRecord typically uses int64
            assert positive.dtype == tf.int64
            assert negative.dtype == tf.int64


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_triplet(self, temp_tfrecord_path):
        """Test with single triplet."""
        single_triplet = tf.data.Dataset.from_tensor_slices([[42, 43, 44]])
        
        count = write_triplets_to_tfrecord(single_triplet, temp_tfrecord_path, compress=False)
        assert count == 1
        
        loaded = load_triplets_from_tfrecord(temp_tfrecord_path, compressed=False)
        triplets = list(loaded)
        assert len(triplets) == 1
        
        center, positive, negative = triplets[0]
        assert [center, positive, negative] == [42, 43, 44]
    
    def test_directory_creation(self):
        """Test that directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "subdir", "triplets.tfrecord")
            
            # Directory doesn't exist yet
            assert not os.path.exists(os.path.dirname(nested_path))
            
            # Write should create directory
            triplets = tf.data.Dataset.from_tensor_slices([[1, 2, 3]])
            write_triplets_to_tfrecord(triplets, nested_path, compress=False)
            
            # Directory should now exist
            assert os.path.exists(nested_path)
