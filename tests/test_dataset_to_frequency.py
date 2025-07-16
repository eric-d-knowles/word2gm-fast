"""Unit tests for dataset_to_frequency module.

Tests the token frequency counting functionality for TensorFlow datasets.
"""
import pytest
import tensorflow as tf
from src.word2gm_fast.dataprep.dataset_to_frequency import dataset_to_frequency


@pytest.fixture
def sample_5gram_lines():
    """Fixture providing sample 5-gram lines for testing."""
    lines = [
        "the quick brown fox jumps",
        "a quick brown dog runs",
        "the brown fox jumps high",
        "quick brown animals move fast",
        "the fox and dog play"
    ]
    return lines


@pytest.fixture
def sample_dataset(sample_5gram_lines):
    """Create a TensorFlow dataset from sample lines."""
    return tf.data.Dataset.from_tensor_slices(sample_5gram_lines)


class TestDatasetToFrequency:
    """Test the dataset_to_frequency function."""
    
    def test_basic_frequency_counting(self, sample_dataset):
        """Test basic frequency counting functionality."""
        frequency_table = dataset_to_frequency(sample_dataset)
        
        # Check that it returns a dictionary
        assert isinstance(frequency_table, dict)
        
        # Check that all expected tokens are present
        expected_tokens = {'the', 'quick', 'brown', 'fox', 'jumps', 'a', 'dog', 'runs', 'high', 'animals', 'move', 'fast', 'and', 'play'}
        assert set(frequency_table.keys()) == expected_tokens
    
    def test_frequency_counts(self, sample_dataset):
        """Test that frequency counts are correct."""
        frequency_table = dataset_to_frequency(sample_dataset)
        
        # Sample data:
        # "the quick brown fox jumps"     - brown: 1
        # "a quick brown dog runs"        - brown: 1 
        # "the brown fox jumps high"      - brown: 1
        # "quick brown animals move fast" - brown: 1
        # "the fox and dog play"          - (no brown)
        
        # Check specific high-frequency tokens
        assert frequency_table['the'] == 3    # lines 1, 3, 5
        assert frequency_table['brown'] == 4  # lines 1, 2, 3, 4
        assert frequency_table['fox'] == 3    # lines 1, 3, 5
        assert frequency_table['quick'] == 3  # lines 1, 2, 4
        assert frequency_table['jumps'] == 2  # lines 1, 3
        assert frequency_table['dog'] == 2    # lines 2, 5
        
        # Check some single-occurrence tokens
        assert frequency_table['a'] == 1
        assert frequency_table['runs'] == 1
        assert frequency_table['high'] == 1
    
    def test_sorted_by_frequency(self, sample_dataset):
        """Test that results are sorted by frequency in descending order."""
        frequency_table = dataset_to_frequency(sample_dataset)
        
        frequencies = list(frequency_table.values())
        
        # Check that frequencies are in descending order
        assert frequencies == sorted(frequencies, reverse=True)
        
        # Check that the highest frequency tokens come first
        tokens_by_order = list(frequency_table.keys())
        high_freq_tokens = tokens_by_order[:3]  # First 3 should be highest frequency
        
        # Should include tokens with frequency 3
        for token in high_freq_tokens:
            assert frequency_table[token] >= 2  # At least frequency 2
    
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        # Create empty dataset with correct string dtype
        empty_dataset = tf.data.Dataset.from_tensor_slices(tf.constant([], dtype=tf.string))
        frequency_table = dataset_to_frequency(empty_dataset)
        
        assert isinstance(frequency_table, dict)
        assert len(frequency_table) == 0
    
    def test_single_line_dataset(self):
        """Test dataset with single line."""
        single_line_dataset = tf.data.Dataset.from_tensor_slices(["hello world test"])
        frequency_table = dataset_to_frequency(single_line_dataset)
        
        expected = {'hello': 1, 'world': 1, 'test': 1}
        assert frequency_table == expected
    
    def test_repeated_tokens_same_line(self):
        """Test counting repeated tokens within the same line."""
        repeated_dataset = tf.data.Dataset.from_tensor_slices(["the the the fox fox"])
        frequency_table = dataset_to_frequency(repeated_dataset)
        
        assert frequency_table['the'] == 3
        assert frequency_table['fox'] == 2
    
    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        whitespace_dataset = tf.data.Dataset.from_tensor_slices([
            "  hello   world  ",
            "\thello\ttest\t",
            " world test "
        ])
        frequency_table = dataset_to_frequency(whitespace_dataset)
        
        # Should properly strip and split
        assert frequency_table['hello'] == 2
        assert frequency_table['world'] == 2
        assert frequency_table['test'] == 2
    
    def test_unk_tokens(self):
        """Test handling of UNK tokens."""
        unk_dataset = tf.data.Dataset.from_tensor_slices([
            "the UNK brown fox UNK",
            "UNK quick brown UNK jumps"
        ])
        frequency_table = dataset_to_frequency(unk_dataset)
        
        # UNK should be counted like any other token
        assert frequency_table['UNK'] == 4
        assert frequency_table['brown'] == 2


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_special_characters(self):
        """Test handling of special characters in tokens."""
        special_dataset = tf.data.Dataset.from_tensor_slices([
            "hello, world! test.",
            "test-case under_score"
        ])
        frequency_table = dataset_to_frequency(special_dataset)
        
        # Should treat punctuation as part of tokens
        assert 'hello,' in frequency_table
        assert 'world!' in frequency_table
        assert 'test.' in frequency_table
        assert 'test-case' in frequency_table
        assert 'under_score' in frequency_table
    
    def test_unicode_tokens(self):
        """Test handling of unicode characters."""
        unicode_dataset = tf.data.Dataset.from_tensor_slices([
            "hello café naïve résumé",
            "café naïve test"
        ])
        frequency_table = dataset_to_frequency(unicode_dataset)
        
        assert frequency_table['café'] == 2
        assert frequency_table['naïve'] == 2
        assert frequency_table['résumé'] == 1
        assert frequency_table['hello'] == 1
    
    def test_large_frequency_values(self):
        """Test with dataset that produces large frequency values."""
        # Create a dataset with many repetitions
        repeated_lines = ["the quick brown fox"] * 1000
        large_dataset = tf.data.Dataset.from_tensor_slices(repeated_lines)
        frequency_table = dataset_to_frequency(large_dataset)
        
        assert frequency_table['the'] == 1000
        assert frequency_table['quick'] == 1000
        assert frequency_table['brown'] == 1000
        assert frequency_table['fox'] == 1000
