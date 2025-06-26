"""
Unit tests for TFRecord I/O utilities (pytest style).

Tests all functionality in src.word2gm_fast.dataprep.tfrecord_io including:
- Triplet serialization and loading
- Vocabulary serialization and loading (including optimized version)
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

from src.word2gm_fast.dataprep.tfrecord_io import (
    write_triplets_to_tfrecord,
    load_triplets_from_tfrecord,
    parse_triplet_example,
    write_vocab_to_tfrecord,
    load_vocab_from_tfrecord,
    parse_vocab_example,
    save_pipeline_artifacts,
    load_pipeline_artifacts
)

@pytest.fixture(scope="module")
def summary_collector(request):
    summaries = []
    yield summaries
    def print_summaries():
        for s in summaries:
            print(s)
    request.addfinalizer(print_summaries)

@pytest.fixture
def tfrecord_test_data(tmp_path):
    # Create sample vocabulary data
    vocab_words = ["UNK", "the", "man", "king", "queen", "word"]
    vocab_ids = [0, 1, 2, 3, 4, 5]
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
        'vocab_table': vocab_table,
        'triplet_data': triplet_data,
        'triplets_dataset': triplets_dataset,
        'text_dataset': text_dataset
    }

def test_write_and_load_triplets_uncompressed(tfrecord_test_data, summary_collector):
    tmp_dir = tfrecord_test_data['tmp_dir']
    triplets_dataset = tfrecord_test_data['triplets_dataset']
    triplet_path = tmp_dir / "triplets.tfrecord"
    write_triplets_to_tfrecord(triplets_dataset, str(triplet_path))
    loaded_ds = load_triplets_from_tfrecord(str(triplet_path))
    loaded_triplets = [tuple(t.numpy() for t in x) for x in loaded_ds]
    assert loaded_triplets == tfrecord_test_data['triplet_data']
    summary_collector.append("[TEST -- tfrecord_io] test_write_and_load_triplets_uncompressed: Roundtrip TFRecord I/O for triplets (uncompressed)")
    print("[SUMMARY] test_write_and_load_triplets_uncompressed: Validates roundtrip TFRecord I/O for triplets (uncompressed).")

def test_write_and_load_triplets_compressed(tfrecord_test_data, summary_collector):
    tmp_dir = tfrecord_test_data['tmp_dir']
    triplets_dataset = tfrecord_test_data['triplets_dataset']
    triplet_path = tmp_dir / "triplets_compressed.tfrecord.gz"
    write_triplets_to_tfrecord(triplets_dataset, str(triplet_path), compress=True)
    loaded_ds = load_triplets_from_tfrecord(str(triplet_path), compressed=True)
    loaded_triplets = [tuple(t.numpy() for t in x) for x in loaded_ds]
    assert loaded_triplets == tfrecord_test_data['triplet_data']
    summary_collector.append("[TEST -- tfrecord_io] test_write_and_load_triplets_compressed: Roundtrip TFRecord I/O for triplets (compressed)")
    print("[SUMMARY] test_write_and_load_triplets_compressed: Validates roundtrip TFRecord I/O for triplets (compressed).")

def test_parse_triplet_example(tfrecord_test_data, summary_collector):
    # Serialize and parse a single example
    ex = tf.train.Example(features=tf.train.Features(feature={
        'center': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
        'positive': tf.train.Feature(int64_list=tf.train.Int64List(value=[2])),
        'negative': tf.train.Feature(int64_list=tf.train.Int64List(value=[3]))
    }))
    ex_bytes = ex.SerializeToString()
    parsed = parse_triplet_example(ex_bytes)
    assert tuple(t.numpy() for t in parsed) == (1, 2, 3)
    summary_collector.append("[TEST -- tfrecord_io] test_parse_triplet_example: Correct parsing of serialized triplet examples")
    print("[SUMMARY] test_parse_triplet_example: Checks correct parsing of serialized triplet examples.")

def test_write_and_load_vocab_uncompressed(tfrecord_test_data, summary_collector):
    tmp_dir = tfrecord_test_data['tmp_dir']
    vocab_table = tfrecord_test_data['vocab_table']
    vocab_words = tfrecord_test_data['vocab_words']
    vocab_ids = tfrecord_test_data['vocab_ids']
    vocab_path = tmp_dir / "vocab.tfrecord"
    write_vocab_to_tfrecord(vocab_table, str(vocab_path))
    loaded_vocab_table = load_vocab_from_tfrecord(str(vocab_path))
    # Check that all vocab words map to the correct ids
    for word, idx in zip(vocab_words, vocab_ids):
        assert loaded_vocab_table.lookup(tf.constant(word)).numpy() == idx
    # Check that an OOV word maps to 0 (UNK)
    assert loaded_vocab_table.lookup(tf.constant("notinthevocab")).numpy() == 0
    summary_collector.append("[TEST -- tfrecord_io] test_write_and_load_vocab_uncompressed: Roundtrip TFRecord I/O for vocabulary (uncompressed)")
    print("[SUMMARY] test_write_and_load_vocab_uncompressed: Validates roundtrip TFRecord I/O for vocabulary (uncompressed).");

def test_write_and_load_vocab_compressed(tfrecord_test_data, summary_collector):
    tmp_dir = tfrecord_test_data['tmp_dir']
    vocab_table = tfrecord_test_data['vocab_table']
    vocab_words = tfrecord_test_data['vocab_words']
    vocab_ids = tfrecord_test_data['vocab_ids']
    vocab_path = tmp_dir / "vocab_compressed.tfrecord.gz"
    write_vocab_to_tfrecord(vocab_table, str(vocab_path), compress=True)
    loaded_vocab_table = load_vocab_from_tfrecord(str(vocab_path), compressed=True)
    for word, idx in zip(vocab_words, vocab_ids):
        assert loaded_vocab_table.lookup(tf.constant(word)).numpy() == idx
    assert loaded_vocab_table.lookup(tf.constant("notinthevocab")).numpy() == 0
    summary_collector.append("[TEST -- tfrecord_io] test_write_and_load_vocab_compressed: Roundtrip TFRecord I/O for vocabulary (compressed)")
    print("[SUMMARY] test_write_and_load_vocab_compressed: Validates roundtrip TFRecord I/O for vocabulary (compressed).")

def test_parse_vocab_example(tfrecord_test_data, summary_collector):
    ex = tf.train.Example(features=tf.train.Features(feature={
        'word': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'test'])),
        'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[42]))
    }))
    ex_bytes = ex.SerializeToString()
    word, idx = parse_vocab_example(ex_bytes)
    assert word.numpy() == b'test'
    assert idx.numpy() == 42
    summary_collector.append("[TEST -- tfrecord_io] test_parse_vocab_example: Correct parsing of serialized vocabulary examples")
    print("[SUMMARY] test_parse_vocab_example: Checks correct parsing of serialized vocabulary examples.")

def test_save_and_load_pipeline_artifacts(tfrecord_test_data, summary_collector):
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
    loaded_vocab = artifacts['vocab_table']
    loaded_triplets = artifacts['triplets_ds']
    # Check that the lookup table maps each word to the correct id
    for word, expected_id in zip(tfrecord_test_data['vocab_words'], tfrecord_test_data['vocab_ids']):
        assert loaded_vocab.lookup(tf.constant(word)).numpy() == expected_id
    loaded_triplets_list = [tuple(x) for x in loaded_triplets]
    assert loaded_triplets_list == tfrecord_test_data['triplet_data']
    summary_collector.append("[TEST -- tfrecord_io] test_save_and_load_pipeline_artifacts: Saving and loading of all pipeline artifacts (vocab and triplets) as TFRecords")
    print("[SUMMARY] test_save_and_load_pipeline_artifacts: Validates saving and loading of all pipeline artifacts (vocab and triplets) as TFRecords")
