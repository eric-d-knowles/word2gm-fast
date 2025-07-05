"""
Integration tests for the complete IO pipeline (pytest style).

Tests that all IO modules work together correctly:
- vocab + tables integration
- artifacts + individual modules integration
- End-to-end pipeline data flow
"""

import pytest
import tempfile
import os
import tensorflow as tf
import numpy as np
from src.word2gm_fast.io.vocab import write_vocab_to_tfrecord, parse_vocab_example
from src.word2gm_fast.io.triplets import write_triplets_to_tfrecord, load_triplets_from_tfrecord
from src.word2gm_fast.io.tables import create_token_to_index_table, create_index_to_token_table
from src.word2gm_fast.io.artifacts import save_pipeline_artifacts, load_pipeline_artifacts


@pytest.fixture
def integration_test_data(tmp_path):
    """Create comprehensive test data for integration testing."""
    # Create a realistic vocabulary
    vocab_words = ["UNK", "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "word", "embeddings"]
    vocab_ids = list(range(len(vocab_words)))
    frequencies = {word: 100.0 / (i + 1) for i, word in enumerate(vocab_words)}  # Decreasing frequencies
    
    vocab_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(vocab_words),
            values=tf.constant(vocab_ids, dtype=tf.int64)
        ),
        default_value=0
    )
    
    # Create realistic triplet data
    triplet_data = [
        (1, 2, 5),  # the, quick, jumps
        (2, 3, 6),  # quick, brown, over
        (3, 4, 7),  # brown, fox, lazy
        (4, 5, 8),  # fox, jumps, dog
        (5, 6, 1),  # jumps, over, the
        (6, 7, 2),  # over, lazy, quick
        (7, 8, 3),  # lazy, dog, brown
        (8, 1, 4),  # dog, the, fox
    ]
    
    triplets_dataset = tf.data.Dataset.from_tensor_slices([
        tf.constant(triplet, dtype=tf.int64) for triplet in triplet_data
    ])
    
    # Sample text dataset
    sample_lines = [
        "the quick brown fox jumps over the lazy dog",
        "word embeddings are useful for nlp",
        "the fox is quick and brown"
    ]
    text_dataset = tf.data.Dataset.from_tensor_slices([
        line.encode('utf-8') for line in sample_lines
    ])
    
    return {
        'tmp_dir': tmp_path,
        'vocab_words': vocab_words,
        'vocab_ids': vocab_ids,
        'frequencies': frequencies,
        'vocab_table': vocab_table,
        'triplet_data': triplet_data,
        'triplets_dataset': triplets_dataset,
        'text_dataset': text_dataset
    }


def test_vocab_tables_roundtrip_consistency(integration_test_data):
    """Test that vocab -> TFRecord -> tables produces consistent results."""
    tmp_dir = integration_test_data['tmp_dir']
    vocab_table = integration_test_data['vocab_table']
    frequencies = integration_test_data['frequencies']
    vocab_words = integration_test_data['vocab_words']
    vocab_ids = integration_test_data['vocab_ids']
    
    # Write vocab to TFRecord
    vocab_path = tmp_dir / "integration_vocab.tfrecord"
    write_vocab_to_tfrecord(vocab_table, str(vocab_path), frequencies=frequencies)
    
    # Create both lookup tables
    token_to_index = create_token_to_index_table(str(vocab_path))
    index_to_token = create_index_to_token_table(str(vocab_path))
    
    # Test forward lookup (token -> index)
    for word, expected_idx in zip(vocab_words, vocab_ids):
        actual_idx = token_to_index.lookup(tf.constant(word)).numpy()
        assert actual_idx == expected_idx, f"Token->Index failed for {word}: {actual_idx} vs {expected_idx}"
    
    # Test reverse lookup (index -> token)
    for expected_word, idx in zip(vocab_words, vocab_ids):
        actual_word = index_to_token.lookup(tf.constant(idx, dtype=tf.int64)).numpy().decode('utf-8')
        assert actual_word == expected_word, f"Index->Token failed for {idx}: {actual_word} vs {expected_word}"
    
    # Test round-trip consistency
    for word in vocab_words:
        idx = token_to_index.lookup(tf.constant(word)).numpy()
        recovered_word = index_to_token.lookup(tf.constant(idx, dtype=tf.int64)).numpy().decode('utf-8')
        assert recovered_word == word, f"Round-trip failed: {word} -> {idx} -> {recovered_word}"


def test_frequency_preservation_through_pipeline(integration_test_data):
    """Test that frequencies are correctly preserved through the entire pipeline."""
    tmp_dir = integration_test_data['tmp_dir']
    vocab_table = integration_test_data['vocab_table']
    frequencies = integration_test_data['frequencies']
    
    # Write vocab with frequencies
    vocab_path = tmp_dir / "freq_test_vocab.tfrecord"
    write_vocab_to_tfrecord(vocab_table, str(vocab_path), frequencies=frequencies)
    
    # Read back and parse frequencies
    raw_ds = tf.data.TFRecordDataset(str(vocab_path))
    vocab_ds = raw_ds.map(parse_vocab_example)
    
    loaded_frequencies = {}
    for word_tensor, id_tensor, freq_tensor in vocab_ds:
        word = word_tensor.numpy().decode('utf-8')
        freq = freq_tensor.numpy()
        loaded_frequencies[word] = freq
    
    # Check all frequencies match
    for word, expected_freq in frequencies.items():
        assert word in loaded_frequencies, f"Word {word} missing in loaded frequencies"
        actual_freq = loaded_frequencies[word]
        assert abs(actual_freq - expected_freq) < 1e-6, f"Frequency mismatch for {word}: {actual_freq} vs {expected_freq}"


def test_complete_artifacts_pipeline(integration_test_data):
    """Test the complete save/load artifacts pipeline."""
    tmp_dir = integration_test_data['tmp_dir']
    artifacts_dir = tmp_dir / "complete_test"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Save artifacts
    save_result = save_pipeline_artifacts(
        integration_test_data['text_dataset'],
        integration_test_data['vocab_table'],
        integration_test_data['triplets_dataset'],
        str(artifacts_dir),
        compress=True
    )
    
    # Verify save result
    assert 'vocab_path' in save_result
    assert 'triplets_path' in save_result
    assert 'vocab_size' in save_result
    assert 'triplet_count' in save_result
    assert save_result['compressed'] is True
    
    # Load artifacts
    loaded_artifacts = load_pipeline_artifacts(str(artifacts_dir), compressed=True)
    
    # Verify loaded structure
    required_keys = ['token_to_index_table', 'index_to_token_table', 'triplets_ds', 'vocab_size']
    for key in required_keys:
        assert key in loaded_artifacts, f"Missing key in loaded artifacts: {key}"
    
    # Test loaded vocab tables
    token_to_index = loaded_artifacts['token_to_index_table']
    index_to_token = loaded_artifacts['index_to_token_table']
    
    for word, expected_idx in zip(integration_test_data['vocab_words'], integration_test_data['vocab_ids']):
        # Test token->index
        actual_idx = token_to_index.lookup(tf.constant(word)).numpy()
        assert actual_idx == expected_idx
        
        # Test index->token
        actual_word = index_to_token.lookup(tf.constant(expected_idx, dtype=tf.int64)).numpy().decode('utf-8')
        assert actual_word == word
    
    # Test loaded triplets (note: they're cast to int32)
    loaded_triplets = [(int(c), int(p), int(n)) for c, p, n in loaded_artifacts['triplets_ds']]
    expected_triplets = integration_test_data['triplet_data']
    assert loaded_triplets == expected_triplets
    
    # Test vocab size
    assert loaded_artifacts['vocab_size'] == len(integration_test_data['vocab_words'])


def test_artifacts_compression_consistency(integration_test_data):
    """Test that compressed and uncompressed artifacts produce identical results."""
    tmp_dir = integration_test_data['tmp_dir']
    
    # Save compressed
    compressed_dir = tmp_dir / "compressed"
    os.makedirs(compressed_dir, exist_ok=True)
    save_pipeline_artifacts(
        integration_test_data['text_dataset'],
        integration_test_data['vocab_table'],
        integration_test_data['triplets_dataset'],
        str(compressed_dir),
        compress=True
    )
    
    # Save uncompressed
    uncompressed_dir = tmp_dir / "uncompressed"
    os.makedirs(uncompressed_dir, exist_ok=True)
    save_pipeline_artifacts(
        integration_test_data['text_dataset'],
        integration_test_data['vocab_table'],
        integration_test_data['triplets_dataset'],
        str(uncompressed_dir),
        compress=False
    )
    
    # Load both
    compressed_artifacts = load_pipeline_artifacts(str(compressed_dir), compressed=True)
    uncompressed_artifacts = load_pipeline_artifacts(str(uncompressed_dir), compressed=False)
    
    # Compare vocab tables
    vocab_words = integration_test_data['vocab_words']
    for word in vocab_words:
        idx_compressed = compressed_artifacts['token_to_index_table'].lookup(tf.constant(word)).numpy()
        idx_uncompressed = uncompressed_artifacts['token_to_index_table'].lookup(tf.constant(word)).numpy()
        assert idx_compressed == idx_uncompressed
    
    # Compare triplets
    triplets_compressed = list(compressed_artifacts['triplets_ds'])
    triplets_uncompressed = list(uncompressed_artifacts['triplets_ds'])
    assert len(triplets_compressed) == len(triplets_uncompressed)
    
    for (c1, p1, n1), (c2, p2, n2) in zip(triplets_compressed, triplets_uncompressed):
        assert c1.numpy() == c2.numpy()
        assert p1.numpy() == p2.numpy()
        assert n1.numpy() == n2.numpy()


def test_artifacts_auto_detection(integration_test_data):
    """Test that compression auto-detection works correctly."""
    tmp_dir = integration_test_data['tmp_dir']
    artifacts_dir = tmp_dir / "auto_detect"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Save compressed
    save_pipeline_artifacts(
        integration_test_data['text_dataset'],
        integration_test_data['vocab_table'],
        integration_test_data['triplets_dataset'],
        str(artifacts_dir),
        compress=True
    )
    
    # Load with auto-detection (should detect .gz files)
    artifacts = load_pipeline_artifacts(str(artifacts_dir))  # No compressed parameter
    
    # Should work correctly
    assert 'token_to_index_table' in artifacts
    assert 'triplets_ds' in artifacts
    assert artifacts['vocab_size'] == len(integration_test_data['vocab_words'])


def test_error_handling_missing_files(tmp_path):
    """Test error handling when artifact files are missing."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    with pytest.raises(FileNotFoundError):
        load_pipeline_artifacts(str(empty_dir))
