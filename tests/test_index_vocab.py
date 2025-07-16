"""Unit tests for index_vocab module.

Tests vocabulary building, string-to-integer conversion, and lookup table functionality.
"""
import pytest
import tensorflow as tf
from unittest.mock import patch
from src.word2gm_fast.dataprep.index_vocab import (
    build_vocab_table,
    make_vocab,
    triplets_to_integers,
    print_vocab_summary,
    preview_integer_triplets,
    print_dataset_properties
)


@pytest.fixture
def sample_lines_dataset():
    """Create a sample dataset of text lines."""
    lines = [
        "the quick brown fox",
        "a quick brown dog",
        "the fox jumps high",
        "animals move very fast"
    ]
    return tf.data.Dataset.from_tensor_slices(lines)


@pytest.fixture
def sample_string_triplets():
    """Create a sample dataset of string triplets."""
    triplets = [
        ("brown", "the", "dog"),
        ("brown", "quick", "animals"),
        ("fox", "the", "move"),
        ("fox", "jumps", "very"),
        ("move", "animals", "quick"),
        ("move", "very", "brown")
    ]
    return tf.data.Dataset.from_tensor_slices(triplets)


@pytest.fixture
def sample_frequency_table():
    """Create a sample frequency table."""
    return {
        "the": 100,
        "quick": 80,
        "brown": 70,
        "fox": 60,
        "jumps": 50,
        "dog": 30,
        "animals": 25,
        "move": 20,
        "very": 15,
        "fast": 10,
        "high": 8,
        "a": 5
    }


class TestBuildVocabTable:
    """Test the build_vocab_table function."""
    
    def test_basic_vocab_table_creation(self):
        """Test basic vocabulary table creation."""
        vocab_list = ["UNK", "apple", "banana", "cherry"]
        vocab_table = build_vocab_table(vocab_list)
        
        # Check that it's a StaticHashTable
        assert isinstance(vocab_table, tf.lookup.StaticHashTable)
        
        # Test lookups
        test_tokens = tf.constant(["apple", "banana", "cherry", "unknown"])
        results = vocab_table.lookup(test_tokens).numpy()
        
        expected = [1, 2, 3, 0]  # unknown maps to UNK (index 0)
        assert list(results) == expected
    
    def test_unk_token_validation(self):
        """Test that UNK token must be at index 0."""
        invalid_vocab = ["apple", "UNK", "banana"]
        
        with pytest.raises(ValueError, match="UNK token must be at index 0"):
            build_vocab_table(invalid_vocab)
    
    def test_lookup_unknown_tokens(self):
        """Test that unknown tokens map to UNK (index 0)."""
        vocab_list = ["UNK", "known", "token"]
        vocab_table = build_vocab_table(vocab_list)
        
        unknown_tokens = tf.constant(["unknown1", "unknown2", "UNK"])
        results = vocab_table.lookup(unknown_tokens).numpy()
        
        expected = [0, 0, 0]  # All map to UNK
        assert list(results) == expected


class TestMakeVocab:
    """Test the make_vocab function."""
    
    def test_basic_vocab_creation(self, sample_lines_dataset):
        """Test basic vocabulary creation from lines."""
        vocab_table, vocab_list, frequencies = make_vocab(sample_lines_dataset)
        
        # Check return types
        assert isinstance(vocab_table, tf.lookup.StaticHashTable)
        assert isinstance(vocab_list, list)
        assert isinstance(frequencies, dict)
        
        # UNK should be at index 0
        assert vocab_list[0] == "UNK"
        
        # Should contain expected tokens
        expected_tokens = {"a", "animals", "brown", "dog", "fast", "fox", 
                          "high", "jumps", "move", "quick", "the", "very"}
        vocab_set = set(vocab_list[1:])  # Exclude UNK
        assert vocab_set == expected_tokens
    
    def test_vocab_ordering(self, sample_lines_dataset):
        """Test that vocabulary is alphabetically ordered (except UNK)."""
        _, vocab_list, _ = make_vocab(sample_lines_dataset)
        
        # Vocabulary should be alphabetically sorted after UNK
        vocab_without_unk = vocab_list[1:]
        assert vocab_without_unk == sorted(vocab_without_unk)
    
    def test_frequency_counting(self, sample_lines_dataset):
        """Test that frequencies are correctly counted."""
        _, vocab_list, frequencies = make_vocab(sample_lines_dataset)
        
        # Check some expected frequencies
        assert frequencies["the"] == 2  # Appears in 2 lines
        assert frequencies["quick"] == 2  # Appears in 2 lines
        assert frequencies["brown"] == 2  # Appears in 2 lines
        assert frequencies["fox"] == 2   # Appears in 2 lines
        assert frequencies["UNK"] == 0   # UNK always has 0 frequency
        
        # Tokens that appear once
        once_tokens = {"a", "dog", "jumps", "high", "animals", "move", "very", "fast"}
        for token in once_tokens:
            assert frequencies[token] == 1
    
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        empty_dataset = tf.data.Dataset.from_tensor_slices(tf.constant([], dtype=tf.string))
        vocab_table, vocab_list, frequencies = make_vocab(empty_dataset)
        
        # Should still have UNK token
        assert len(vocab_list) == 1
        assert vocab_list[0] == "UNK"
        assert frequencies["UNK"] == 0


class TestTripletsToIntegers:
    """Test the triplets_to_integers function."""
    
    def test_basic_conversion(self, sample_string_triplets):
        """Test basic string-to-integer conversion."""
        integer_dataset, vocab_table, vocab_list, vocab_size, _ = triplets_to_integers(
            sample_string_triplets
        )
        
        # Check return types
        assert isinstance(integer_dataset, tf.data.Dataset)
        assert isinstance(vocab_table, tf.lookup.StaticHashTable)
        assert isinstance(vocab_list, list)
        assert isinstance(vocab_size, int)
        
        # Vocab should start with UNK
        assert vocab_list[0] == "UNK"
        assert vocab_size == len(vocab_list)
        
        # All triplets should be integers
        for triplet in integer_dataset.take(3):
            center, positive, negative = triplet.numpy()
            assert isinstance(int(center), int)
            assert isinstance(int(positive), int)
            assert isinstance(int(negative), int)
    
    def test_vocab_table_consistency(self, sample_string_triplets):
        """Test that vocab table produces consistent mappings."""
        integer_dataset, vocab_table, vocab_list, _, _ = triplets_to_integers(
            sample_string_triplets
        )
        
        # Test that vocab table matches vocab list
        for i, token in enumerate(vocab_list):
            lookup_result = vocab_table.lookup(tf.constant([token])).numpy()[0]
            assert lookup_result == i
    
    def test_frequency_preservation(self, sample_string_triplets, sample_frequency_table):
        """Test that frequency table is used for ordering when provided."""
        _, _, vocab_list_with_freq, _, summary = triplets_to_integers(
            sample_string_triplets,
            frequency_table=sample_frequency_table,
            show_summary=True
        )
        
        _, _, vocab_list_without_freq, _, _ = triplets_to_integers(
            sample_string_triplets
        )
        
        # Should have different orderings (frequency vs alphabetical)
        # Both should start with UNK but differ after that
        assert vocab_list_with_freq[0] == "UNK"
        assert vocab_list_without_freq[0] == "UNK"
        
        # Summary should be returned when show_summary=True
        assert summary is not None
        assert "vocab_size" in summary
    
    def test_caching_functionality(self, sample_string_triplets):
        """Test that caching option works."""
        cached_dataset, _, _, _, _ = triplets_to_integers(
            sample_string_triplets,
            cache=True
        )
        
        # Cached dataset should show cache in string representation
        assert "Cache" in str(cached_dataset)
    
    def test_empty_triplets(self):
        """Test handling of empty triplets dataset."""
        empty_triplets = tf.data.Dataset.from_tensor_slices(tf.constant([], dtype=tf.string, shape=[0, 3]))
        
        integer_dataset, vocab_table, vocab_list, vocab_size, _ = triplets_to_integers(
            empty_triplets
        )
        
        # Should have only UNK token
        assert vocab_size == 1
        assert vocab_list == ["UNK"]
        
        # Dataset should be empty
        triplets = list(integer_dataset)
        assert len(triplets) == 0
    
    def test_token_counting_from_triplets(self, sample_string_triplets):
        """Test that token counts are correctly computed from triplets."""
        _, _, _, _, summary = triplets_to_integers(
            sample_string_triplets,
            show_summary=True
        )
        
        # Should have computed some token statistics
        assert "vocab_size" in summary
        assert "total_tokens" in summary
        assert summary["total_tokens"] > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_token_vocab(self):
        """Test vocabulary with only UNK token."""
        single_line = tf.data.Dataset.from_tensor_slices([""])
        vocab_table, vocab_list, frequencies = make_vocab(single_line)
        
        assert len(vocab_list) == 1
        assert vocab_list[0] == "UNK"
        assert frequencies["UNK"] == 0
    
    def test_duplicate_tokens_in_line(self):
        """Test handling of repeated tokens within a line."""
        repeated_dataset = tf.data.Dataset.from_tensor_slices(["the the the"])
        vocab_table, vocab_list, frequencies = make_vocab(repeated_dataset)
        
        # Should count each occurrence
        assert frequencies["the"] == 3
        assert "the" in vocab_list
    
    def test_unk_token_in_input(self):
        """Test handling when UNK appears in input data."""
        unk_dataset = tf.data.Dataset.from_tensor_slices(["UNK token test"])
        vocab_table, vocab_list, frequencies = make_vocab(unk_dataset)
        
        # UNK should still be at index 0 with frequency 0
        assert vocab_list[0] == "UNK"
        assert frequencies["UNK"] == 0
        
        # The UNK from input should be treated as a regular token
        # But our make_vocab function specifically excludes it from frequency counting
        assert "token" in vocab_list
        assert "test" in vocab_list


class TestDisplayFunctions:
    """Test the display and summary functions (without testing actual display output)."""
    
    def test_print_vocab_summary_structure(self, sample_string_triplets, sample_frequency_table):
        """Test that print_vocab_summary returns proper structure."""
        _, vocab_table, vocab_list, _, _ = triplets_to_integers(sample_string_triplets)
        
        with patch('IPython.display.display_markdown'):  # Mock the display
            summary = print_vocab_summary(vocab_list, vocab_table, sample_frequency_table)
        
        # Check summary structure
        expected_keys = {"vocab_size", "unk_token", "sample_tokens", 
                        "lowest_index", "highest_index", "total_tokens", "most_frequent"}
        assert all(key in summary for key in expected_keys)
        
        assert summary["vocab_size"] == len(vocab_list)
        assert summary["unk_token"] == "UNK"
        assert summary["lowest_index"] == 0
        assert summary["highest_index"] == len(vocab_list) - 1
    
    def test_display_functions_no_errors(self, sample_string_triplets):
        """Test that display functions don't raise errors."""
        integer_dataset, vocab_table, vocab_list, _, _ = triplets_to_integers(
            sample_string_triplets
        )
        
        # Mock display functions to avoid actual output
        with patch('IPython.display.display_markdown'):
            # These should not raise errors
            print_vocab_summary(vocab_list, vocab_table)
            preview_integer_triplets(integer_dataset, 2)
            print_dataset_properties(integer_dataset)
