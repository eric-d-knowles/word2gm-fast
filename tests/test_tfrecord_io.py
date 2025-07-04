"""
Unit tests for the new modular I/O utilities (pytest style).

Tests all functionality in src.word2gm_fast.io including:
- Vocabulary TFRecord operations with frequency support
- Triplet TFRecord operations
- Lookup table creation
- Pipeline artifact save/load
- Error handling and edge cases
"""

import pytest
import tempfile
import shutil
import os
import tensorflow as tf
import numpy as np
from typing import List, Tuple



from src.word2gm_fast.io.triplets import (
    write_triplets_to_tfrecord,
    load_triplets_from_tfrecord,
    parse_triplet_example
)
from src.word2gm_fast.io.vocab import (
    write_vocab_to_tfrecord,
    parse_vocab_example
)
from src.word2gm_fast.io.tables import (
    create_token_to_index_table,
    create_index_to_token_table
)
from src.word2gm_fast.io.artifacts import (
    save_pipeline_artifacts,
    load_pipeline_artifacts
)



@pytest.fixture    
def tfrecord_test_data(tmp_path):
    # Create sample vocabulary data
    vocab_words = ["UNK", "the", "man", "king", "queen", "word"]
    vocab_ids = [0, 1, 2, 3, 4, 5]
    frequencies = {
        "UNK": 100.0,
        "the": 50.0,
        "man": 25.0,
        "king": 15.0,
        "queen": 12.0,
        "word": 8.0
    }
    vocab_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(vocab_words),
            values=tf.constant(vocab_ids, dtype=tf.int64)
        ),
        default_value=0
    )
    triplet_data = [
        (1, 2, 3),
        (2, 1, 4),
        (3, 4, 5),
        (4, 3, 1),
        (5, 2, 3)
    ]
    triplets_dataset = tf.data.Dataset.from_tensor_slices(
        [tf.constant(triplet, dtype=tf.int64) for triplet in triplet_data]
    )
    sample_lines = ["the king is great", "the queen rules", "word embeddings"]
    text_dataset = tf.data.Dataset.from_tensor_slices(
        [line.encode('utf-8') for line in sample_lines]
    )
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

def test_write_and_load_triplets_uncompressed(tfrecord_test_data):
    tmp_dir = tfrecord_test_data['tmp_dir']
    triplets_dataset = tfrecord_test_data['triplets_dataset']
    triplet_path = tmp_dir / "triplets.tfrecord"
    write_triplets_to_tfrecord(triplets_dataset, str(triplet_path))
    loaded_ds = load_triplets_from_tfrecord(str(triplet_path))
    loaded_triplets = [tuple(t.numpy() for t in x) for x in loaded_ds]
    assert loaded_triplets == tfrecord_test_data['triplet_data']


def test_write_and_load_triplets_compressed(tfrecord_test_data):
    tmp_dir = tfrecord_test_data['tmp_dir']
    triplets_dataset = tfrecord_test_data['triplets_dataset']
    triplet_path = tmp_dir / "triplets_compressed.tfrecord.gz"
    write_triplets_to_tfrecord(triplets_dataset, str(triplet_path), compress=True)
    loaded_ds = load_triplets_from_tfrecord(str(triplet_path), compressed=True)
    loaded_triplets = [tuple(t.numpy() for t in x) for x in loaded_ds]
    assert loaded_triplets == tfrecord_test_data['triplet_data']


def test_parse_triplet_example(tfrecord_test_data):
    # Serialize and parse a single example
    ex = tf.train.Example(features=tf.train.Features(feature={
        'center': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
        'positive': tf.train.Feature(int64_list=tf.train.Int64List(value=[2])),
        'negative': tf.train.Feature(int64_list=tf.train.Int64List(value=[3]))
    }))
    ex_bytes = ex.SerializeToString()
    parsed = parse_triplet_example(ex_bytes)
    assert tuple(t.numpy() for t in parsed) == (1, 2, 3)


def test_write_and_load_vocab_uncompressed(tfrecord_test_data):
    tmp_dir = tfrecord_test_data['tmp_dir']
    vocab_table = tfrecord_test_data['vocab_table']
    vocab_words = tfrecord_test_data['vocab_words']
    vocab_ids = tfrecord_test_data['vocab_ids']
    vocab_path = tmp_dir / "vocab.tfrecord"
    write_vocab_to_tfrecord(vocab_table, str(vocab_path))
    loaded_vocab_table = create_token_to_index_table(str(vocab_path))
    # Check that all vocab words map to the correct ids
    for word, idx in zip(vocab_words, vocab_ids):
        assert loaded_vocab_table.lookup(tf.constant(word)).numpy() == idx
    # Check that an OOV word maps to 0 (UNK)
    assert loaded_vocab_table.lookup(tf.constant("notinthevocab")).numpy() == 0


def test_write_and_load_vocab_with_frequencies(tfrecord_test_data):
    """Test that vocabulary TFRecords correctly store and preserve frequency information."""
    tmp_dir = tfrecord_test_data['tmp_dir']
    vocab_table = tfrecord_test_data['vocab_table']
    frequencies = tfrecord_test_data['frequencies']
    vocab_path = tmp_dir / "vocab_with_freq.tfrecord"
    
    # Write vocab with frequencies
    write_vocab_to_tfrecord(vocab_table, str(vocab_path), frequencies=frequencies)
    
    # Load back and verify frequencies are preserved in the TFRecord
    raw_ds = tf.data.TFRecordDataset(str(vocab_path))
    vocab_ds = raw_ds.map(parse_vocab_example)
    
    loaded_data = {}
    for word_tensor, id_tensor, freq_tensor in vocab_ds:
        word = word_tensor.numpy().decode('utf-8')
        freq = freq_tensor.numpy()
        loaded_data[word] = freq
    
    # Check that frequencies match
    for word, expected_freq in frequencies.items():
        assert word in loaded_data
        assert abs(loaded_data[word] - expected_freq) < 1e-6, f"Frequency mismatch for {word}: {loaded_data[word]} vs {expected_freq}"


def test_write_vocab_without_frequencies(tfrecord_test_data):
    """Test that vocabulary TFRecords work correctly when no frequencies are provided."""
    tmp_dir = tfrecord_test_data['tmp_dir']
    vocab_table = tfrecord_test_data['vocab_table']
    vocab_path = tmp_dir / "vocab_no_freq.tfrecord"
    
    # Write vocab without frequencies
    write_vocab_to_tfrecord(vocab_table, str(vocab_path), frequencies=None)
    
    # Load back and verify default frequencies (0.0) are used
    raw_ds = tf.data.TFRecordDataset(str(vocab_path))
    vocab_ds = raw_ds.map(parse_vocab_example)
    
    for word_tensor, id_tensor, freq_tensor in vocab_ds:
        freq = freq_tensor.numpy()
        assert freq == 0.0, f"Expected default frequency 0.0, got {freq}"


def test_write_and_load_vocab_compressed(tfrecord_test_data):
    tmp_dir = tfrecord_test_data['tmp_dir']
    vocab_table = tfrecord_test_data['vocab_table']
    vocab_words = tfrecord_test_data['vocab_words']
    vocab_ids = tfrecord_test_data['vocab_ids']
    vocab_path = tmp_dir / "vocab_compressed.tfrecord.gz"
    write_vocab_to_tfrecord(vocab_table, str(vocab_path), compress=True)
    loaded_vocab_table = create_token_to_index_table(str(vocab_path), compressed=True)
    for word, idx in zip(vocab_words, vocab_ids):
        assert loaded_vocab_table.lookup(tf.constant(word)).numpy() == idx
    assert loaded_vocab_table.lookup(tf.constant("notinthevocab")).numpy() == 0


def test_parse_vocab_example(tfrecord_test_data):
    """Test parsing vocab examples with frequency information."""
    ex = tf.train.Example(features=tf.train.Features(feature={
        'word': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'test'])),
        'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[42])),
        'frequency': tf.train.Feature(float_list=tf.train.FloatList(value=[3.14]))
    }))
    ex_bytes = ex.SerializeToString()
    word, idx, freq = parse_vocab_example(ex_bytes)
    assert word.numpy() == b'test'
    assert idx.numpy() == 42
    assert abs(freq.numpy() - 3.14) < 1e-6


def test_parse_vocab_example_default_frequency(tfrecord_test_data):
    """Test parsing vocab examples with missing frequency (should default to 0.0)."""
    ex = tf.train.Example(features=tf.train.Features(feature={
        'word': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'test'])),
        'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[42]))
        # Note: no frequency field
    }))
    ex_bytes = ex.SerializeToString()
    word, idx, freq = parse_vocab_example(ex_bytes)
    assert word.numpy() == b'test'
    assert idx.numpy() == 42
    assert freq.numpy() == 0.0  # Should default to 0.0


def test_save_and_load_pipeline_artifacts(tfrecord_test_data):
    tmp_dir = tfrecord_test_data['tmp_dir']
    artifacts_dir = tmp_dir / "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    save_pipeline_artifacts(
        tfrecord_test_data['text_dataset'],
        tfrecord_test_data['vocab_table'],
        tfrecord_test_data['triplets_dataset'],
        str(artifacts_dir)
    )
    artifacts = load_pipeline_artifacts(str(artifacts_dir))
    loaded_token_to_index = artifacts['token_to_index_table']
    loaded_index_to_token = artifacts['index_to_token_table']
    loaded_triplets = artifacts['triplets_ds']
    vocab_size = artifacts['vocab_size']
    
    # Check that the lookup table maps each word to the correct id
    for word, expected_id in zip(tfrecord_test_data['vocab_words'], tfrecord_test_data['vocab_ids']):
        assert loaded_token_to_index.lookup(tf.constant(word)).numpy() == expected_id
        assert loaded_index_to_token.lookup(tf.constant(expected_id, dtype=tf.int64)).numpy().decode('utf-8') == word
    
    # Check vocab size
    assert vocab_size == len(tfrecord_test_data['vocab_words'])
    
    # Check triplets (they are cast to int32 in the loaded artifacts)
    loaded_triplets_list = [(int(c), int(p), int(n)) for c, p, n in loaded_triplets]
    assert loaded_triplets_list == tfrecord_test_data['triplet_data']


def test_create_index_to_token_table(tfrecord_test_data):
    """Test that index-to-token lookup tables work correctly."""
    tmp_dir = tfrecord_test_data['tmp_dir']
    vocab_table = tfrecord_test_data['vocab_table']
    vocab_path = tmp_dir / "vocab.tfrecord"
    
    write_vocab_to_tfrecord(vocab_table, str(vocab_path))
    index_to_token_table = create_index_to_token_table(str(vocab_path))
    
    # Test forward and reverse lookup
    for word, idx in zip(tfrecord_test_data['vocab_words'], tfrecord_test_data['vocab_ids']):
        retrieved_word = index_to_token_table.lookup(tf.constant(idx, dtype=tf.int64)).numpy().decode('utf-8')
        assert retrieved_word == word
    
    # Test default value for unknown index
    unknown_word = index_to_token_table.lookup(tf.constant(999, dtype=tf.int64)).numpy().decode('utf-8')
    assert unknown_word == "UNK"


