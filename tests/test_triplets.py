"""
Pytest tests for word2gm_fast.io.triplets module.
"""
import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
src_path = PROJECT_ROOT / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import tensorflow as tf
from word2gm_fast.io.triplets import (
    write_triplets_to_tfrecord, 
    load_triplets_from_tfrecord,
    parse_triplet_example
)


class TestTripletsModule:
    """Test class for triplets I/O functionality."""
    
    @pytest.fixture
    def sample_triplets_list(self):
        """Create sample triplets as a Python list."""
        return [
            (1, 2, 3),  # (target, context, negative)
            (2, 3, 4),
            (3, 4, 1),
            (4, 1, 2),
            (1, 3, 4),
        ]
    
    @pytest.fixture
    def sample_triplets_dataset(self, sample_triplets_list):
        """Create sample triplets as a TensorFlow dataset."""
        return tf.data.Dataset.from_tensor_slices([
            tf.constant(triplet, dtype=tf.int64) for triplet in sample_triplets_list
        ])
    
    def test_write_triplets_from_list(self, sample_triplets_list, tmp_path):
        """Test writing triplets from a Python list."""
        triplet_path = tmp_path / "triplets_list.tfrecord"
        
        write_triplets_to_tfrecord(sample_triplets_list, str(triplet_path))
        
        assert triplet_path.exists()
        assert triplet_path.stat().st_size > 0
    
    def test_write_triplets_from_dataset(self, sample_triplets_dataset, tmp_path):
        """Test writing triplets from a TensorFlow dataset."""
        triplet_path = tmp_path / "triplets_dataset.tfrecord"
        
        write_triplets_to_tfrecord(sample_triplets_dataset, str(triplet_path))
        
        assert triplet_path.exists()
        assert triplet_path.stat().st_size > 0
    
    def test_load_triplets_from_tfrecord(self, sample_triplets_list, tmp_path):
        """Test loading triplets from TFRecord."""
        triplet_path = tmp_path / "triplets_load.tfrecord"
        
        # Write triplets
        write_triplets_to_tfrecord(sample_triplets_list, str(triplet_path))
        
        # Load triplets back
        loaded_dataset = load_triplets_from_tfrecord(str(triplet_path))
        
        assert isinstance(loaded_dataset, tf.data.Dataset)
        
        # Convert to list and verify
        loaded_triplets = list(loaded_dataset)
        assert len(loaded_triplets) == len(sample_triplets_list)
    
    def test_parse_triplet_example(self, sample_triplets_list, tmp_path):
        """Test parsing individual triplet examples."""
        triplet_path = tmp_path / "triplets_parse.tfrecord"
        
        # Write triplets
        write_triplets_to_tfrecord(sample_triplets_list, str(triplet_path))
        
        # Read raw TFRecord and parse
        raw_dataset = tf.data.TFRecordDataset(str(triplet_path))
        
        for raw_record in raw_dataset.take(1):
            center, positive, negative = parse_triplet_example(raw_record)
            assert isinstance(center, tf.Tensor)
            assert isinstance(positive, tf.Tensor)
            assert isinstance(negative, tf.Tensor)
            assert center.dtype == tf.int64
            assert positive.dtype == tf.int64
            assert negative.dtype == tf.int64
    
    def test_triplets_roundtrip(self, sample_triplets_list, tmp_path):
        """Test complete roundtrip: write then read triplets."""
        triplet_path = tmp_path / "triplets_roundtrip.tfrecord"
        
        # Write triplets
        write_triplets_to_tfrecord(sample_triplets_list, str(triplet_path))
        
        # Load back and convert to list
        loaded_dataset = load_triplets_from_tfrecord(str(triplet_path))
        loaded_triplets = []
        
        for triplet in loaded_dataset:
            center, positive, negative = triplet
            loaded_triplets.append((
                center.numpy(),
                positive.numpy(),
                negative.numpy()
            ))
        
        # Verify data matches
        assert len(loaded_triplets) == len(sample_triplets_list)
        for original, loaded in zip(sample_triplets_list, loaded_triplets):
            assert original == loaded
    
    def test_triplets_compression(self, sample_triplets_list, tmp_path):
        """Test compressed triplets TFRecord writing."""
        triplet_path = tmp_path / "triplets_compressed.tfrecord"
        
        write_triplets_to_tfrecord(sample_triplets_list, str(triplet_path), compress=True)
        
        # Compressed files have .gz extension added
        gz_path = tmp_path / "triplets_compressed.tfrecord.gz"
        assert gz_path.exists()
    
    def test_load_compressed_triplets(self, sample_triplets_list, tmp_path):
        """Test loading compressed triplets."""
        triplet_path = tmp_path / "triplets_load_compressed.tfrecord"
        
        # Write compressed triplets
        write_triplets_to_tfrecord(sample_triplets_list, str(triplet_path), compress=True)
        
        # Load compressed triplets (should auto-detect compression)
        gz_path = str(triplet_path) + ".gz"
        loaded_dataset = load_triplets_from_tfrecord(gz_path)
        
        # Verify we can read the data
        loaded_triplets = list(loaded_dataset)
        assert len(loaded_triplets) == len(sample_triplets_list)
    
    def test_empty_triplets(self, tmp_path):
        """Test handling of empty triplets."""
        triplet_path = tmp_path / "triplets_empty.tfrecord"
        
        write_triplets_to_tfrecord([], str(triplet_path))
        
        assert triplet_path.exists()
        
        # Should be able to load empty dataset
        loaded_dataset = load_triplets_from_tfrecord(str(triplet_path))
        loaded_triplets = list(loaded_dataset)
        assert len(loaded_triplets) == 0
    
    def test_large_triplets(self, tmp_path):
        """Test handling of large triplet values."""
        large_triplets = [
            (1000000, 2000000, 3000000),
            (4000000, 5000000, 6000000),
        ]
        
        triplet_path = tmp_path / "triplets_large.tfrecord"
        
        write_triplets_to_tfrecord(large_triplets, str(triplet_path))
        
        # Load back and verify
        loaded_dataset = load_triplets_from_tfrecord(str(triplet_path))
        loaded_triplets = []
        
        for triplet in loaded_dataset:
            center, positive, negative = triplet
            loaded_triplets.append((
                center.numpy(),
                positive.numpy(),
                negative.numpy()
            ))
        
        assert loaded_triplets == large_triplets
