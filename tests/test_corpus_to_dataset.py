"""Unit tests for corpus_to_dataset module focusing on critical functionality.

Tests the core logic of 5-gram validation, dataset filtering, and summary statistics.
"""
import pytest
import tensorflow as tf
from unittest.mock import patch
from src.word2gm_fast.dataprep.corpus_to_dataset import (
    validate_5gram_line,
    print_dataset_summary,
    make_dataset
)



@pytest.fixture
def test_5gram_data():
    """Fixture providing test data for 5-gram validation."""
    valid_lines = [
        "the quick brown fox jumps",
        "UNK quick brown fox jumps",  # context word present
        "the UNK brown fox jumps",    # context word present
        "the quick brown UNK jumps",  # context word present
        "UNK UNK brown fox jumps",    # at least one context word
        "the quick brown UNK UNK",    # at least one context word
    ]
    
    invalid_lines = [
        "the quick UNK fox jumps",    # center is UNK
        "UNK UNK UNK UNK jumps",      # center is UNK, only one context
        "the UNK UNK UNK UNK",        # all context words are UNK
        "UNK quick UNK UNK UNK",      # center is UNK
    ]
    
    all_lines = valid_lines + invalid_lines
    return all_lines, valid_lines, invalid_lines


@pytest.fixture
def temp_corpus_file(tmp_path, test_5gram_data):
    """Create a temporary corpus file with test data."""
    all_lines, _, _ = test_5gram_data
    temp_file = tmp_path / "test_corpus.txt"
    temp_file.write_text("\n".join(all_lines) + "\n")
    return str(temp_file)


class TestValidate5gramLine:
    """Test the core 5-gram validation logic."""
    
    def test_valid_cases(self, test_5gram_data):
        """Test that valid 5-grams are correctly identified."""
        _, valid_lines, _ = test_5gram_data
        
        for line in valid_lines:
            line_tensor = tf.constant(line)
            _, is_valid = validate_5gram_line(line_tensor)
            assert is_valid.numpy() == True, f"Should be valid: {line}"
    
    def test_invalid_cases(self, test_5gram_data):
        """Test that invalid 5-grams are correctly rejected."""
        _, _, invalid_lines = test_5gram_data
        
        for line in invalid_lines:
            line_tensor = tf.constant(line)
            _, is_valid = validate_5gram_line(line_tensor)
            assert is_valid.numpy() == False, f"Should be invalid: {line}"
    
    def test_validation_rules(self):
        """Test specific validation rules."""
        # Center word cannot be UNK
        line_tensor = tf.constant("the quick UNK fox jumps")
        _, is_valid = validate_5gram_line(line_tensor)
        assert is_valid.numpy() == False
        
        # At least one context word must not be UNK
        line_tensor = tf.constant("UNK UNK brown UNK UNK")
        _, is_valid = validate_5gram_line(line_tensor)
        assert is_valid.numpy() == False
        
        # Valid case: center OK, at least one context OK
        line_tensor = tf.constant("UNK UNK brown fox UNK")
        _, is_valid = validate_5gram_line(line_tensor)
        assert is_valid.numpy() == True


class TestMakeDataset:
    """Test the main dataset creation function."""
    
    def test_filtering_accuracy(self, temp_corpus_file, test_5gram_data):
        """Test that filtering produces correct results."""
        _, valid_lines, invalid_lines = test_5gram_data
        
        dataset, summary = make_dataset(temp_corpus_file, show_summary=True)
        
        # Check filtered results
        result_lines = [line.numpy().decode("utf-8").strip() for line in dataset]
        expected_lines = [line.strip() for line in valid_lines]
        
        assert set(result_lines) == set(expected_lines)
        assert len(result_lines) == len(valid_lines)
    
    def test_summary_statistics(self, temp_corpus_file, test_5gram_data):
        """Test that summary statistics are calculated correctly."""
        _, valid_lines, invalid_lines = test_5gram_data
        
        with patch('src.word2gm_fast.dataprep.corpus_to_dataset.display'):
            dataset, summary = make_dataset(temp_corpus_file, show_summary=True)
        
        assert isinstance(summary, dict)
        assert summary["retained"] == len(valid_lines)
        assert summary["rejected"] == len(invalid_lines)
        assert summary["total"] == len(valid_lines) + len(invalid_lines)
    
    def test_caching_functionality(self, temp_corpus_file):
        """Test that caching option works."""
        dataset_cached, _ = make_dataset(temp_corpus_file, cache=True)
        dataset_uncached, _ = make_dataset(temp_corpus_file, cache=False)
        
        # Both should produce same results
        cached_results = [line.numpy().decode("utf-8") for line in dataset_cached]
        uncached_results = [line.numpy().decode("utf-8") for line in dataset_uncached]
        
        assert set(cached_results) == set(uncached_results)
        
        # Cached dataset should show cache in string representation
        assert "Cache" in str(dataset_cached)
    
    def test_preview_integration(self, temp_corpus_file):
        """Test that preview option integrates correctly."""
        with patch('src.word2gm_fast.dataprep.corpus_to_dataset.preview_dataset') as mock_preview:
            dataset, _ = make_dataset(temp_corpus_file, preview_n=3)
            mock_preview.assert_called_once_with(dataset, 3)
            
            # Test no preview
            mock_preview.reset_mock()
            dataset, _ = make_dataset(temp_corpus_file, preview_n=0)
            mock_preview.assert_not_called()
    
    def test_optional_summary(self, temp_corpus_file):
        """Test summary option behavior."""
        # With summary
        dataset, summary = make_dataset(temp_corpus_file, show_summary=True)
        assert isinstance(summary, dict)
        assert "retained" in summary
        
        # Without summary
        dataset, summary = make_dataset(temp_corpus_file, show_summary=False)
        assert summary is None


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_file(self, tmp_path):
        """Test handling of empty corpus file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        
        dataset, summary = make_dataset(str(empty_file), show_summary=True)
        
        result_lines = list(dataset)
        assert len(result_lines) == 0
        assert summary["retained"] == 0
        assert summary["rejected"] == 0
        assert summary["total"] == 0
    
    def test_all_invalid_lines(self, tmp_path):
        """Test file with only invalid lines."""
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("the quick UNK fox jumps\nUNK UNK UNK UNK UNK\n")
        
        dataset, summary = make_dataset(str(invalid_file), show_summary=True)
        
        result_lines = list(dataset)
        assert len(result_lines) == 0
        assert summary["retained"] == 0
        assert summary["rejected"] == 2
        assert summary["total"] == 2
    
    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        line_with_spaces = tf.constant("  the quick brown fox jumps  ")
        _, is_valid = validate_5gram_line(line_with_spaces)
        assert is_valid.numpy() == True


class TestSummaryFunction:
    """Test the summary calculation function directly."""
    
    def test_summary_calculation(self, temp_corpus_file, test_5gram_data):
        """Test direct summary calculation."""
        _, valid_lines, invalid_lines = test_5gram_data
        dataset, _ = make_dataset(temp_corpus_file)
        
        with patch('src.word2gm_fast.dataprep.corpus_to_dataset.display'):
            summary = print_dataset_summary(dataset, temp_corpus_file)
        
        # Verify counts match expected data
        assert summary["retained"] == len(valid_lines)
        assert summary["rejected"] == len(invalid_lines)
        assert summary["total"] == summary["retained"] + summary["rejected"]

