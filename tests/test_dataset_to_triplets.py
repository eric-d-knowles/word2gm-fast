"""
Unit tests for dataset_to_triplets.py (pytest style)
"""
import pytest
import tensorflow as tf
import numpy as np
from src.word2gm_fast.dataprep.dataset_to_triplets import build_skipgram_triplets
from src.word2gm_fast.dataprep.index_vocab import build_vocab_table

@pytest.fixture(scope="module")
def summary_collector(request):
    summaries = []
    yield summaries
    def print_summaries():
        for s in summaries:
            print(s)
    request.addfinalizer(print_summaries)

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

def test_basic_triplet_generation(triplet_test_data, summary_collector):
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
    summary_collector.append("[TEST -- dataset_to_triplets] test_basic_triplet_generation: Skip-gram triplets are generated with non-UNK context words")

def test_center_word_extraction(triplet_test_data, summary_collector):
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
    summary_collector.append("[TEST -- dataset_to_triplets] test_center_word_extraction: Center words match expected values")

def test_context_word_extraction(triplet_test_data, summary_collector):
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
    summary_collector.append("[TEST -- dataset_to_triplets] test_context_word_extraction: Positive context tokens are valid")

def test_multiple_triplets_per_line(triplet_test_data, summary_collector):
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
    summary_collector.append("[TEST -- dataset_to_triplets] test_multiple_triplets_per_line: Multiple triplets are generated per line, skipping UNK contexts")

def test_negative_sampling_range(triplet_test_data, summary_collector):
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
    summary_collector.append("[TEST -- dataset_to_triplets] test_negative_sampling_range: Negative samples are in the correct range and not UNK")

def test_no_triplets_with_unk_context(triplet_test_data, summary_collector):
    """Test that no triplet has UNK as the positive context word."""
    vocab, vocab_table, lines, dataset = triplet_test_data
    triplets_ds = build_skipgram_triplets(dataset, vocab_table)
    triplets = list(triplets_ds.as_numpy_iterator())
    for triplet in triplets:
        pos = triplet[1]
        assert pos != 0
    summary_collector.append("[TEST -- dataset_to_triplets] test_no_triplets_with_unk_context: No triplet has UNK as positive context word")
