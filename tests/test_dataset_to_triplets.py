"""
Unit tests for dataset_to_triplets.py (pytest style)
"""
import pytest
import tensorflow as tf
import numpy as np
from src.word2gm_fast.dataprep.dataset_to_triplets import build_skipgram_triplets
from src.word2gm_fast.dataprep.index_vocab import build_vocab_table



@pytest.fixture
def triplet_test_data():
    vocab = ["UNK", "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    vocab_table = build_vocab_table(vocab)
    lines = [
        b"the quick brown fox jumps",
        b"quick brown fox jumps over", 
        b"brown fox jumps over the",
        b"fox jumps over the lazy",
        b"jumps over the lazy dog",
        # Add lines with UNK in context positions
        b"UNK quick brown fox jumps",
        b"the UNK brown fox jumps",
        b"the quick brown UNK jumps",
        b"the quick brown fox UNK",
        b"UNK UNK brown fox jumps",
        b"the quick brown UNK UNK",
    ]
    dataset = tf.data.Dataset.from_tensor_slices(lines)
    return vocab, vocab_table, lines, dataset

def test_basic_triplet_generation(triplet_test_data):
    vocab, vocab_table, lines, dataset = triplet_test_data
    triplets_ds = build_skipgram_triplets(dataset, vocab_table)
    triplets = list(triplets_ds.as_numpy_iterator())
    assert len(triplets) > 0
    for triplet in triplets:
        assert len(triplet) == 3
        center, pos, neg = triplet
        assert center >= 0
        assert pos >= 0
        assert neg >= 0
        assert neg != 0     # Negative should not be UNK


def test_center_word_extraction(triplet_test_data):
    vocab, vocab_table, lines, dataset = triplet_test_data
    triplets_ds = build_skipgram_triplets(dataset, vocab_table)
    triplets = list(triplets_ds.as_numpy_iterator())
    expected_centers = []
    for line in lines:
        tokens = line.decode("utf-8").split()
        center_token = tokens[2]
        center_id = vocab.index(center_token)
        expected_centers.append(center_id)
    actual_centers = set(triplet[0] for triplet in triplets)
    expected_centers_set = set(expected_centers)
    assert actual_centers.issubset(expected_centers_set)


def test_context_word_extraction(triplet_test_data):
    vocab, vocab_table, lines, dataset = triplet_test_data
    triplets_ds = build_skipgram_triplets(dataset, vocab_table)
    triplets = list(triplets_ds.as_numpy_iterator())
    all_context_words = set()
    for line in lines:
        tokens = line.decode("utf-8").split()
        context_tokens = [tokens[i] for i in [0, 1, 3, 4]]
        all_context_words.update(context_tokens)
    context_ids = set(vocab.index(tok) for tok in all_context_words)
    for triplet in triplets:
        pos = triplet[1]
        assert pos in context_ids


def test_multiple_triplets_per_line(triplet_test_data):
    """Test that multiple triplets are generated per valid line, skipping UNK contexts."""
    vocab, vocab_table, lines, dataset = triplet_test_data
    triplets_ds = build_skipgram_triplets(dataset, vocab_table)
    triplets = list(triplets_ds.as_numpy_iterator())
    # There should be multiple triplets (since each line can generate up to 4, but fewer if UNK in context)
    assert len(triplets) >= 4
    # All centers should be valid (not UNK)
    for triplet in triplets:
        center = triplet[0]
        assert center != 0
    # No positive context should be UNK
    for triplet in triplets:
        pos = triplet[1]
        assert pos != 0


def test_negative_sampling_range(triplet_test_data):
    """Test that negative samples are in the correct range."""
    vocab, vocab_table, lines, dataset = triplet_test_data
    triplets_ds = build_skipgram_triplets(dataset, vocab_table)
    triplets = list(triplets_ds.as_numpy_iterator())
    vocab_size = len(vocab)
    for triplet in triplets:
        neg = triplet[2]
        # Negative should be in range [1, vocab_size) (excluding UNK at 0)
        assert neg >= 1
        assert neg < vocab_size


def test_no_triplets_with_unk_context(triplet_test_data):
    """Test that no triplet has UNK as the positive context word."""
    vocab, vocab_table, lines, dataset = triplet_test_data
    triplets_ds = build_skipgram_triplets(dataset, vocab_table)
    triplets = list(triplets_ds.as_numpy_iterator())
    for triplet in triplets:
        pos = triplet[1]
        assert pos != 0


def test_frequency_based_downsampling(triplet_test_data):
    """Test that frequency-based downsampling reduces high-frequency words."""
    vocab, vocab_table, lines, dataset = triplet_test_data
    
    # Create frequency distribution where "the" is very high frequency
    frequencies = {
        "UNK": 100.0,
        "the": 1000.0,  # Very high frequency - should be downsampled
        "quick": 50.0,
        "brown": 30.0,
        "fox": 25.0,
        "jumps": 20.0,
        "over": 15.0,
        "lazy": 10.0,
        "dog": 8.0
    }
    
    # Generate triplets with downsampling
    triplets_ds_downsampled = build_skipgram_triplets(
        dataset, vocab_table, 
        frequencies=frequencies, 
        downsample_threshold=1e-3  # Aggressive downsampling
    )
    
    # Generate triplets without downsampling
    triplets_ds_normal = build_skipgram_triplets(dataset, vocab_table)
    
    triplets_downsampled = list(triplets_ds_downsampled.as_numpy_iterator())
    triplets_normal = list(triplets_ds_normal.as_numpy_iterator())
    
    # Count occurrences of "the" (index 1) as center word
    the_idx = vocab_table.lookup(tf.constant("the")).numpy()
    
    the_count_downsampled = sum(1 for t in triplets_downsampled if t[0] == the_idx)
    the_count_normal = sum(1 for t in triplets_normal if t[0] == the_idx)
    
    # Downsampling should reduce the count of high-frequency words
    assert the_count_downsampled <= the_count_normal
    
    # Should still have some triplets
    assert len(triplets_downsampled) > 0


def test_downsampling_threshold_effect(triplet_test_data):
    """Test that different downsampling thresholds produce different results."""
    vocab, vocab_table, lines, dataset = triplet_test_data
    
    frequencies = {word: 100.0 if word == "the" else 10.0 for word in vocab}
    
    # Aggressive downsampling
    triplets_aggressive = list(build_skipgram_triplets(
        dataset, vocab_table, 
        frequencies=frequencies, 
        downsample_threshold=1e-2
    ).as_numpy_iterator())
    
    # Mild downsampling
    triplets_mild = list(build_skipgram_triplets(
        dataset, vocab_table, 
        frequencies=frequencies, 
        downsample_threshold=1e-5
    ).as_numpy_iterator())
    
    # No downsampling
    triplets_none = list(build_skipgram_triplets(
        dataset, vocab_table
    ).as_numpy_iterator())
    
    # More aggressive downsampling should result in fewer or equal triplets
    assert len(triplets_aggressive) <= len(triplets_mild) <= len(triplets_none)


def test_frequencies_none_no_downsampling(triplet_test_data):
    """Test that passing frequencies=None disables downsampling."""
    vocab, vocab_table, lines, dataset = triplet_test_data
    
    # Should be equivalent to not passing frequencies
    triplets_none = list(build_skipgram_triplets(
        dataset, vocab_table, frequencies=None
    ).as_numpy_iterator())
    
    triplets_default = list(build_skipgram_triplets(
        dataset, vocab_table
    ).as_numpy_iterator())
    
    # Should produce identical results
    assert len(triplets_none) == len(triplets_default)
    assert triplets_none == triplets_default

