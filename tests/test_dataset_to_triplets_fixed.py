"""Unit tests for dataset_to_triplets module.

Tests the triplet generation functionality including downsampling and negative sampling.
"""
import pytest
import tensorflow as tf
from unittest.mock import patch
from src.word2gm_fast.dataprep.dataset_to_triplets import (
    dataset_to_triplets,
    print_triplets_summary
)


@pytest.fixture
def sample_5gram_dataset():
    """Create a sample dataset of valid 5-gram lines."""
    lines = [
        "the quick brown fox jumps",
        "a quick brown dog runs",
        "the brown fox jumps high",
        "quick animals move very fast"
    ]
    return tf.data.Dataset.from_tensor_slices(lines)


@pytest.fixture
def sample_frequency_table():
    """Create a sample frequency table for testing."""
    return {
        "the": 100,
        "quick": 80, 
        "brown": 70,
        "fox": 60,
        "jumps": 50,
        "a": 40,
        "dog": 30,
        "runs": 20,
        "high": 15,
        "animals": 10,
        "move": 8,
        "very": 6,
        "fast": 5,
        "UNK": 200  # Should be excluded from negative sampling
    }


class TestDatasetToTriplets:
    """Test the main dataset_to_triplets function."""
    
    def test_basic_triplet_generation(self, sample_5gram_dataset, sample_frequency_table):
        """Test basic triplet generation functionality."""
        triplets_ds, summary = dataset_to_triplets(
            dataset=sample_5gram_dataset,
            frequency_table=sample_frequency_table,
            downsample_threshold=1.0,  # High threshold to keep all tokens
            show_summary=True
        )
        
        # Check that we get a dataset back
        assert isinstance(triplets_ds, tf.data.Dataset)
        assert isinstance(summary, dict)
        
        # Should generate some triplets
        assert summary["total_triplets"] > 0
        
        # Collect all triplets to verify structure
        triplets = list(triplets_ds.take(10))
        assert len(triplets) > 0
        
        # Each triplet should have 3 string elements
        for triplet in triplets:
            center, positive, negative = triplet.numpy()
            assert isinstance(center.decode('utf-8'), str)
            assert isinstance(positive.decode('utf-8'), str)
            assert isinstance(negative.decode('utf-8'), str)
    
    def test_triplet_structure(self, sample_5gram_dataset, sample_frequency_table):
        """Test that triplets have correct structure (center, positive, negative)."""
        triplets_ds, _ = dataset_to_triplets(
            dataset=sample_5gram_dataset,
            frequency_table=sample_frequency_table,
            downsample_threshold=1.0
        )
        
        # Get first few triplets
        triplets = list(triplets_ds.take(5))
        
        for triplet in triplets:
            center, positive, negative = [t.numpy().decode('utf-8') for t in triplet]
            
            # Center should be from position 2 of the 5-grams
            # For our sample data, centers should be: brown, brown, fox, move
            expected_centers = {"brown", "fox", "move"}
            assert center in expected_centers
            
            # Positive should be a context word (not UNK)
            assert positive != "UNK"
            
            # Negative should be from vocabulary (not UNK) 
            assert negative != "UNK"
            assert negative in sample_frequency_table
    
    def test_unk_exclusion(self, sample_frequency_table):
        """Test that UNK tokens are properly excluded from negative sampling."""
        # Create dataset with UNK context tokens
        lines_with_unk = [
            "the quick brown UNK jumps",  # One UNK context
            "UNK UNK brown fox UNK"       # Multiple UNK contexts
        ]
        unk_dataset = tf.data.Dataset.from_tensor_slices(lines_with_unk)
        
        triplets_ds, _ = dataset_to_triplets(
            dataset=unk_dataset,
            frequency_table=sample_frequency_table,
            downsample_threshold=1.0
        )
        
        # Collect all triplets
        triplets = list(triplets_ds)
        
        # Check that no UNK appears in any position
        for triplet in triplets:
            center, positive, negative = [t.numpy().decode('utf-8') for t in triplet]
            assert center != "UNK"
            assert positive != "UNK"
            assert negative != "UNK"
    
    def test_empty_dataset(self, sample_frequency_table):
        """Test handling of empty dataset."""
        empty_dataset = tf.data.Dataset.from_tensor_slices(tf.constant([], dtype=tf.string))
        
        triplets_ds, summary = dataset_to_triplets(
            dataset=empty_dataset,
            frequency_table=sample_frequency_table,
            show_summary=True
        )
        
        assert isinstance(triplets_ds, tf.data.Dataset)
        assert summary["total_triplets"] == 0
        
        # Should be able to iterate without error
        triplets = list(triplets_ds)
        assert len(triplets) == 0
    
    def test_caching_functionality(self, sample_5gram_dataset, sample_frequency_table):
        """Test that caching option works."""
        triplets_cached, _ = dataset_to_triplets(
            dataset=sample_5gram_dataset,
            frequency_table=sample_frequency_table,
            cache=True
        )
        
        # Cached dataset should show cache in string representation
        assert "Cache" in str(triplets_cached)
    
    def test_summary_statistics(self, sample_5gram_dataset, sample_frequency_table):
        """Test that summary statistics are reasonable."""
        triplets_ds, summary = dataset_to_triplets(
            dataset=sample_5gram_dataset,
            frequency_table=sample_frequency_table,
            downsample_threshold=1.0,
            show_summary=True
        )
        
        # Check summary structure
        required_keys = ["total_triplets", "unique_centers", "unique_positives", 
                        "unique_negatives", "total_unique_words"]
        for key in required_keys:
            assert key in summary
            assert isinstance(summary[key], int)
            assert summary[key] >= 0
        
        # Total unique words should be at least as large as any individual category
        assert summary["total_unique_words"] >= summary["unique_centers"]
        assert summary["total_unique_words"] >= summary["unique_positives"]
        assert summary["total_unique_words"] >= summary["unique_negatives"]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_line_dataset(self, sample_frequency_table):
        """Test dataset with single line."""
        single_line_dataset = tf.data.Dataset.from_tensor_slices(["the quick brown fox jumps"])
        
        triplets_ds, summary = dataset_to_triplets(
            dataset=single_line_dataset,
            frequency_table=sample_frequency_table,
            downsample_threshold=1.0,
            show_summary=True
        )
        
        # Should generate some triplets from the single line
        assert summary["total_triplets"] > 0
        
        # All triplets should have "brown" as center (position 2)
        triplets = list(triplets_ds)
        for triplet in triplets:
            center = triplet[0].numpy().decode('utf-8')
            assert center == "brown"
    
    def test_all_unk_context(self, sample_frequency_table):
        """Test line where all context words are UNK."""
        all_unk_context = tf.data.Dataset.from_tensor_slices(["UNK UNK brown UNK UNK"])
        
        triplets_ds, summary = dataset_to_triplets(
            dataset=all_unk_context,
            frequency_table=sample_frequency_table,
            downsample_threshold=1.0,
            show_summary=True
        )
        
        # Should generate no triplets since no valid context
        assert summary["total_triplets"] == 0
        
        triplets = list(triplets_ds)
        assert len(triplets) == 0


class TestDownsampling:
    """Test frequency-based downsampling functionality."""
    
    def test_downsampling_effect(self, sample_frequency_table):
        """Test that downsampling reduces high-frequency words."""
        high_freq_dataset = tf.data.Dataset.from_tensor_slices([
            "the the the the the",  # All high-frequency "the"
            "a a a a a"             # All medium-frequency "a"
        ])
        
        # No downsampling
        triplets_no_ds, summary_no_ds = dataset_to_triplets(
            dataset=high_freq_dataset,
            frequency_table=sample_frequency_table,
            downsample_threshold=1.0,  # No downsampling
            show_summary=True
        )
        
        # Aggressive downsampling
        triplets_ds, summary_ds = dataset_to_triplets(
            dataset=high_freq_dataset,
            frequency_table=sample_frequency_table,
            downsample_threshold=0.001,  # Aggressive downsampling
            show_summary=True
        )
        
        # Downsampling should reduce total triplets
        assert summary_ds["total_triplets"] <= summary_no_ds["total_triplets"]
    
    def test_threshold_zero_keeps_all(self, sample_5gram_dataset, sample_frequency_table):
        """Test that threshold=0 effectively disables downsampling."""
        triplets_ds, summary = dataset_to_triplets(
            dataset=sample_5gram_dataset,
            frequency_table=sample_frequency_table,
            downsample_threshold=0.0,  # Essentially no downsampling
            show_summary=True
        )
        
        # Should have generated some triplets
        assert summary["total_triplets"] > 0
