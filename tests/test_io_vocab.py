"""Unit tests for io.vocab module.

Tests TFRecord serialization and deserialization of vocabulary data.
"""
import pytest
import tensorflow as tf
import tempfile
import os
from src.word2gm_fast.io.vocab import (
    write_vocab_to_tfrecord,
    load_vocab_from_tfrecord
)
from src.word2gm_fast.dataprep.index_vocab import build_vocab_table


@pytest.fixture
def sample_vocab_data():
    """Create sample vocabulary data."""
    vocab_list = ["UNK", "the", "quick", "brown", "fox", "jumps"]
    vocab_table = build_vocab_table(vocab_list)
    frequencies = {
        "UNK": 0,
        "the": 100,
        "quick": 80,
        "brown": 60,
        "fox": 40,
        "jumps": 20
    }
    return vocab_table, vocab_list, frequencies


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


class TestWriteVocabToTFRecord:
    """Test the write_vocab_to_tfrecord function."""
    
    def test_basic_write_uncompressed(self, sample_vocab_data, temp_tfrecord_path):
        """Test basic writing of vocabulary to uncompressed TFRecord."""
        vocab_table, vocab_list, frequencies = sample_vocab_data
        
        write_vocab_to_tfrecord(
            vocab_table, 
            temp_tfrecord_path, 
            frequencies=frequencies,
            compress=False
        )
        
        # Check that file was created
        assert os.path.exists(temp_tfrecord_path)
        assert os.path.getsize(temp_tfrecord_path) > 0
    
    def test_basic_write_compressed(self, sample_vocab_data, temp_compressed_tfrecord_path):
        """Test basic writing of vocabulary to compressed TFRecord."""
        vocab_table, vocab_list, frequencies = sample_vocab_data
        
        write_vocab_to_tfrecord(
            vocab_table, 
            temp_compressed_tfrecord_path, 
            frequencies=frequencies,
            compress=True
        )
        
        # Check that file was created
        assert os.path.exists(temp_compressed_tfrecord_path)
        assert os.path.getsize(temp_compressed_tfrecord_path) > 0
    
    def test_write_without_frequencies(self, sample_vocab_data, temp_tfrecord_path):
        """Test writing vocabulary without frequency information."""
        vocab_table, vocab_list, frequencies = sample_vocab_data
        
        write_vocab_to_tfrecord(
            vocab_table, 
            temp_tfrecord_path,
            frequencies=None,
            compress=False
        )
        
        # Should still create a valid file
        assert os.path.exists(temp_tfrecord_path)
        assert os.path.getsize(temp_tfrecord_path) > 0
    
    def test_empty_vocab(self, temp_tfrecord_path):
        """Test writing empty vocabulary."""
        empty_vocab = ["UNK"]  # Minimum valid vocab
        vocab_table = build_vocab_table(empty_vocab)
        
        write_vocab_to_tfrecord(
            vocab_table,
            temp_tfrecord_path,
            frequencies={"UNK": 0},
            compress=False
        )
        
        assert os.path.exists(temp_tfrecord_path)


class TestLoadVocabFromTFRecord:
    """Test the load_vocab_from_tfrecord function."""
    
    def test_basic_load_uncompressed(self, sample_vocab_data, temp_tfrecord_path):
        """Test basic loading from uncompressed TFRecord."""
        vocab_table, vocab_list, frequencies = sample_vocab_data
        
        # First write the data
        write_vocab_to_tfrecord(
            vocab_table, temp_tfrecord_path, 
            frequencies=frequencies, compress=False
        )
        
        # Then load it back
        loaded_vocab_list, loaded_frequencies = load_vocab_from_tfrecord(
            temp_tfrecord_path, compressed=False
        )
        
        # Check vocabulary list
        assert loaded_vocab_list == vocab_list
        
        # Check frequencies
        assert loaded_frequencies == frequencies
    
    def test_basic_load_compressed(self, sample_vocab_data, temp_compressed_tfrecord_path):
        """Test basic loading from compressed TFRecord."""
        vocab_table, vocab_list, frequencies = sample_vocab_data
        
        # First write the data
        write_vocab_to_tfrecord(
            vocab_table, temp_compressed_tfrecord_path, 
            frequencies=frequencies, compress=True
        )
        
        # Then load it back
        loaded_vocab_list, loaded_frequencies = load_vocab_from_tfrecord(
            temp_compressed_tfrecord_path, compressed=True
        )
        
        # Check vocabulary list
        assert loaded_vocab_list == vocab_list
        
        # Check frequencies
        assert loaded_frequencies == frequencies
    
    def test_load_without_frequencies(self, sample_vocab_data, temp_tfrecord_path):
        """Test loading when no frequencies were saved."""
        vocab_table, vocab_list, frequencies = sample_vocab_data
        
        # Write without frequencies
        write_vocab_to_tfrecord(
            vocab_table, temp_tfrecord_path, 
            frequencies=None, compress=False
        )
        
        # Load back
        loaded_vocab_list, loaded_frequencies = load_vocab_from_tfrecord(
            temp_tfrecord_path, compressed=False
        )
        
        # Vocabulary should be preserved
        assert loaded_vocab_list == vocab_list
        
        # Frequencies should all be 0 when None was saved
        expected_frequencies = {token: 0 for token in vocab_list}
        assert loaded_frequencies == expected_frequencies
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        # TFRecordDataset doesn't raise until you try to iterate
        with pytest.raises(Exception):  # Should raise some kind of file error when iterating
            load_vocab_from_tfrecord("/nonexistent/file.tfrecord", compressed=False)


class TestRoundTripConsistency:
    """Test write-then-load consistency."""
    
    def test_roundtrip_with_frequencies(self, sample_vocab_data, temp_tfrecord_path):
        """Test that write->load preserves vocab and frequencies exactly."""
        vocab_table, vocab_list, frequencies = sample_vocab_data
        
        # Write
        write_vocab_to_tfrecord(
            vocab_table, temp_tfrecord_path, 
            frequencies=frequencies, compress=False
        )
        
        # Load
        loaded_vocab_list, loaded_frequencies = load_vocab_from_tfrecord(
            temp_tfrecord_path, compressed=False
        )
        
        # Check exact equality
        assert loaded_vocab_list == vocab_list
        assert loaded_frequencies == frequencies
    
    def test_roundtrip_compressed(self, sample_vocab_data, temp_compressed_tfrecord_path):
        """Test that compressed write->load preserves data exactly."""
        vocab_table, vocab_list, frequencies = sample_vocab_data
        
        # Write
        write_vocab_to_tfrecord(
            vocab_table, temp_compressed_tfrecord_path, 
            frequencies=frequencies, compress=True
        )
        
        # Load
        loaded_vocab_list, loaded_frequencies = load_vocab_from_tfrecord(
            temp_compressed_tfrecord_path, compressed=True
        )
        
        # Check exact equality
        assert loaded_vocab_list == vocab_list
        assert loaded_frequencies == frequencies
    
    def test_vocab_ordering_preserved(self, temp_tfrecord_path):
        """Test that vocabulary ordering is preserved."""
        # Create vocab with specific ordering
        vocab_list = ["UNK", "zebra", "apple", "banana"]  # Not alphabetical
        vocab_table = build_vocab_table(vocab_list)
        frequencies = {"UNK": 0, "zebra": 10, "apple": 20, "banana": 30}
        
        # Write and load
        write_vocab_to_tfrecord(vocab_table, temp_tfrecord_path, frequencies, compress=False)
        loaded_vocab_list, loaded_frequencies = load_vocab_from_tfrecord(temp_tfrecord_path, compressed=False)
        
        # Order should be preserved exactly
        assert loaded_vocab_list == vocab_list
        assert loaded_frequencies == frequencies


class TestLargeVocabulary:
    """Test with large vocabulary sizes."""
    
    def test_large_vocab(self, temp_tfrecord_path):
        """Test with larger vocabulary."""
        # Create a larger vocabulary
        vocab_list = ["UNK"] + [f"word_{i}" for i in range(1000)]
        vocab_table = build_vocab_table(vocab_list)
        frequencies = {token: i for i, token in enumerate(vocab_list)}
        
        # Write and load
        write_vocab_to_tfrecord(vocab_table, temp_tfrecord_path, frequencies, compress=False)
        loaded_vocab_list, loaded_frequencies = load_vocab_from_tfrecord(temp_tfrecord_path, compressed=False)
        
        # Should preserve all data
        assert len(loaded_vocab_list) == 1001
        assert loaded_vocab_list == vocab_list
        assert loaded_frequencies == frequencies


class TestSpecialCharacters:
    """Test with special characters in vocabulary."""
    
    def test_special_characters(self, temp_tfrecord_path):
        """Test vocabulary with special characters."""
        vocab_list = ["UNK", "hello", "world!", "@symbol", "café", "测试"]
        vocab_table = build_vocab_table(vocab_list)
        frequencies = {token: i * 10 for i, token in enumerate(vocab_list)}
        
        # Write and load
        write_vocab_to_tfrecord(vocab_table, temp_tfrecord_path, frequencies, compress=False)
        loaded_vocab_list, loaded_frequencies = load_vocab_from_tfrecord(temp_tfrecord_path, compressed=False)
        
        # Should handle special characters correctly
        assert loaded_vocab_list == vocab_list
        assert loaded_frequencies == frequencies


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_minimal_vocab(self, temp_tfrecord_path):
        """Test with minimal vocabulary (just UNK)."""
        vocab_list = ["UNK"]
        vocab_table = build_vocab_table(vocab_list)
        frequencies = {"UNK": 0}
        
        write_vocab_to_tfrecord(vocab_table, temp_tfrecord_path, frequencies, compress=False)
        loaded_vocab_list, loaded_frequencies = load_vocab_from_tfrecord(temp_tfrecord_path, compressed=False)
        
        assert loaded_vocab_list == vocab_list
        assert loaded_frequencies == frequencies
    
    def test_zero_frequencies(self, temp_tfrecord_path):
        """Test with zero frequencies for all tokens."""
        vocab_list = ["UNK", "token1", "token2"]
        vocab_table = build_vocab_table(vocab_list)
        frequencies = {token: 0 for token in vocab_list}
        
        write_vocab_to_tfrecord(vocab_table, temp_tfrecord_path, frequencies, compress=False)
        loaded_vocab_list, loaded_frequencies = load_vocab_from_tfrecord(temp_tfrecord_path, compressed=False)
        
        assert loaded_vocab_list == vocab_list
        assert loaded_frequencies == frequencies
    
    def test_directory_creation(self):
        """Test that directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "subdir", "vocab.tfrecord")
            
            # Directory doesn't exist yet
            assert not os.path.exists(os.path.dirname(nested_path))
            
            # Write should create directory
            vocab_list = ["UNK", "test"]
            vocab_table = build_vocab_table(vocab_list)
            write_vocab_to_tfrecord(vocab_table, nested_path, compress=False)
            
            # Directory should now exist
            assert os.path.exists(nested_path)
