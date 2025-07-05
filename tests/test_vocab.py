"""
Pytest tests for word2gm_fast.io.vocab module.
"""
import pytest
import tempfile
import os
import tensorflow as tf
from word2gm_fast.io.vocab import write_vocab_to_tfrecord, parse_vocab_example


class TestVocabModule:
    """Test class for vocab I/O functionality."""
    
    @pytest.fixture
    def sample_vocab_table(self):
        """Create a sample vocabulary table for testing."""
        vocab_words = ["UNK", "the", "quick", "brown", "fox"]
        vocab_indices = list(range(len(vocab_words)))
        
        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(vocab_words),
                values=tf.constant(vocab_indices, dtype=tf.int64)
            ),
            default_value=0
        )
    
    @pytest.fixture
    def sample_frequencies(self):
        """Create sample frequency data."""
        return {
            "UNK": 100.0,
            "the": 90.0,
            "quick": 80.0,
            "brown": 70.0,
            "fox": 60.0
        }
    
    def test_write_vocab_to_tfrecord_basic(self, sample_vocab_table, tmp_path):
        """Test basic vocab writing without frequencies."""
        vocab_path = tmp_path / "vocab.tfrecord"
        
        write_vocab_to_tfrecord(sample_vocab_table, str(vocab_path))
        
        assert vocab_path.exists()
        assert vocab_path.stat().st_size > 0
    
    def test_write_vocab_to_tfrecord_with_frequencies(self, sample_vocab_table, sample_frequencies, tmp_path):
        """Test vocab writing with frequencies."""
        vocab_path = tmp_path / "vocab_freq.tfrecord"
        
        write_vocab_to_tfrecord(sample_vocab_table, str(vocab_path), frequencies=sample_frequencies)
        
        assert vocab_path.exists()
        assert vocab_path.stat().st_size > 0
    
    def test_parse_vocab_example_basic(self, sample_vocab_table, tmp_path):
        """Test parsing vocab examples without frequencies."""
        vocab_path = tmp_path / "vocab_parse.tfrecord"
        write_vocab_to_tfrecord(sample_vocab_table, str(vocab_path))
        
        # Read the TFRecord and parse an example
        dataset = tf.data.TFRecordDataset(str(vocab_path))
        for raw_record in dataset.take(1):
            token, index = parse_vocab_example(raw_record)
            assert isinstance(token, tf.Tensor)
            assert isinstance(index, tf.Tensor)
            assert token.dtype == tf.string
            assert index.dtype == tf.int64
    
    def test_parse_vocab_example_with_frequencies(self, sample_vocab_table, sample_frequencies, tmp_path):
        """Test parsing vocab examples with frequencies."""
        vocab_path = tmp_path / "vocab_parse_freq.tfrecord"
        write_vocab_to_tfrecord(sample_vocab_table, str(vocab_path), frequencies=sample_frequencies)
        
        # Read the TFRecord and parse an example
        dataset = tf.data.TFRecordDataset(str(vocab_path))
        for raw_record in dataset.take(1):
            token, index, frequency = parse_vocab_example(raw_record)
            assert isinstance(token, tf.Tensor)
            assert isinstance(index, tf.Tensor)
            assert isinstance(frequency, tf.Tensor)
            assert token.dtype == tf.string
            assert index.dtype == tf.int64
            assert frequency.dtype == tf.float32
    
    def test_vocab_roundtrip(self, sample_vocab_table, sample_frequencies, tmp_path):
        """Test complete roundtrip: write then read vocab."""
        vocab_path = tmp_path / "vocab_roundtrip.tfrecord"
        
        # Write vocab
        write_vocab_to_tfrecord(sample_vocab_table, str(vocab_path), frequencies=sample_frequencies)
        
        # Read back and verify
        dataset = tf.data.TFRecordDataset(str(vocab_path))
        parsed_data = []
        
        for raw_record in dataset:
            token, index, frequency = parse_vocab_example(raw_record)
            parsed_data.append((
                token.numpy().decode('utf-8'),
                index.numpy(),
                frequency.numpy()
            ))
        
        # Verify we got all expected tokens
        assert len(parsed_data) == 5
        tokens = [item[0] for item in parsed_data]
        assert "UNK" in tokens
        assert "the" in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
    
    def test_vocab_compression(self, sample_vocab_table, tmp_path):
        """Test compressed vocab TFRecord writing."""
        vocab_path = tmp_path / "vocab_compressed.tfrecord"
        
        write_vocab_to_tfrecord(sample_vocab_table, str(vocab_path), compress=True)
        
        assert vocab_path.exists()
        # Compressed files should have .gz extension
        gz_path = tmp_path / "vocab_compressed.tfrecord.gz"
        assert gz_path.exists()
    
    def test_empty_frequencies(self, sample_vocab_table, tmp_path):
        """Test handling of empty frequencies."""
        vocab_path = tmp_path / "vocab_empty_freq.tfrecord"
        
        write_vocab_to_tfrecord(sample_vocab_table, str(vocab_path), frequencies={})
        
        assert vocab_path.exists()
        # Should still work with empty frequencies
        dataset = tf.data.TFRecordDataset(str(vocab_path))
        count = sum(1 for _ in dataset)
        assert count == 5  # All vocab words should be written
