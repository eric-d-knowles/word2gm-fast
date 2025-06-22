"""
Optimized TFRecord I/O utilities for word2GM skip-gram training data.
Provides faster functions for vocabulary and triplet serialization.
"""

import os
import tensorflow as tf
import time
from typing import Dict, List, Optional, Union
import numpy as np


def write_vocab_to_tfrecord_optimized(
    vocab_table: tf.lookup.StaticHashTable,
    output_path: str,
    compress: bool = False,
    batch_size: int = 1000
) -> None:
    """
    Save vocabulary to TFRecord with batched writing for speed.
    
    Parameters
    ----------
    vocab_table : tf.lookup.StaticHashTable
        The vocabulary lookup table to save.
    output_path : str
        Path for the output TFRecord file.
    compress : bool, optional
        Whether to compress the file with GZIP. Default is False.
    batch_size : int, optional
        Number of vocab entries to batch together. Default is 1000.
    """
    if compress and not output_path.endswith(".gz"):
        output_path += ".gz"

    options = tf.io.TFRecordOptions(compression_type="GZIP") if compress else None

    print(f"Writing vocabulary TFRecord to: {output_path}")
    start = time.perf_counter()

    # Export the vocabulary table once
    vocab_keys, vocab_values = vocab_table.export()
    vocab_keys_np = vocab_keys.numpy()
    vocab_values_np = vocab_values.numpy()

    count = 0
    with tf.io.TFRecordWriter(output_path, options=options) as writer:
        # Process in batches
        for i in range(0, len(vocab_keys_np), batch_size):
            batch_keys = vocab_keys_np[i:i+batch_size]
            batch_values = vocab_values_np[i:i+batch_size]
            
            # Create batch example
            words_batch = [key.decode('utf-8').encode('utf-8') for key in batch_keys]
            ids_batch = [int(val) for val in batch_values]
            
            example = tf.train.Example(features=tf.train.Features(feature={
                'words': tf.train.Feature(bytes_list=tf.train.BytesList(value=words_batch)),
                'ids': tf.train.Feature(int64_list=tf.train.Int64List(value=ids_batch)),
                'batch_size': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(words_batch)])),
            }))
            writer.write(example.SerializeToString())
            count += len(words_batch)

    duration = time.perf_counter() - start
    print(f"Vocabulary write complete. Words written: {count:,}")
    print(f"Write time: {duration:.2f} sec")


def load_vocab_from_tfrecord_optimized(
    tfrecord_path: str,
    compressed: Optional[bool] = None,
    default_value: int = 0
) -> tf.lookup.StaticHashTable:
    """
    Load vocabulary from TFRecord with optimized batch processing.
    
    Parameters
    ----------
    tfrecord_path : str
        Path to the vocabulary TFRecord file.
    compressed : bool, optional
        Whether the file is GZIP compressed. Auto-detected if None.
    default_value : int, optional
        Default value for unknown words. Default is 0.
        
    Returns
    -------
    tf.lookup.StaticHashTable
        Reconstructed vocabulary lookup table.
    """
    if compressed is None:
        compressed = tfrecord_path.endswith(".gz")

    compression_type = "GZIP" if compressed else None

    print(f"Loading vocabulary TFRecord from: {tfrecord_path}")
    start = time.perf_counter()

    # Parse batch example
    def parse_batch_vocab_example(example_proto):
        feature_description = {
            'words': tf.io.VarLenFeature(tf.string),
            'ids': tf.io.VarLenFeature(tf.int64),
            'batch_size': tf.io.FixedLenFeature([1], tf.int64),
        }
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        return (tf.sparse.to_dense(parsed['words']), 
                tf.sparse.to_dense(parsed['ids']))

    # Load and process all batches
    raw_ds = tf.data.TFRecordDataset(tfrecord_path, compression_type=compression_type)
    batch_ds = raw_ds.map(parse_batch_vocab_example)
    
    # Collect all words and ids efficiently
    all_words = []
    all_ids = []
    
    for words_batch, ids_batch in batch_ds:
        all_words.extend(words_batch.numpy())
        all_ids.extend(ids_batch.numpy())
    
    # Create the lookup table
    vocab_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(all_words),
            values=tf.constant(all_ids, dtype=tf.int64)
        ),
        default_value=default_value
    )

    duration = time.perf_counter() - start
    print(f"Vocabulary loaded. Size: {len(all_words):,} words")
    print(f"Load time: {duration:.2f} sec")

    return vocab_table


def write_triplets_to_tfrecord_optimized(
    dataset: tf.data.Dataset,
    output_path: str,
    compress: bool = False,
    batch_size: int = 1000
) -> None:
    """
    Write triplets to TFRecord with batched writing for speed.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Dataset of (center, positive, negative) triplets.
    output_path : str
        Path for the output TFRecord file.
    compress : bool, optional
        Whether to compress the file with GZIP. Default is False.
    batch_size : int, optional
        Number of triplets to batch together. Default is 1000.
    """
    if compress and not output_path.endswith(".gz"):
        output_path += ".gz"

    options = tf.io.TFRecordOptions(compression_type="GZIP") if compress else None

    print(f"Writing TFRecord to: {output_path}")
    start = time.perf_counter()

    count = 0
    with tf.io.TFRecordWriter(output_path, options=options) as writer:
        # Batch the dataset
        batched_ds = dataset.batch(batch_size)
        
        for batch in batched_ds.as_numpy_iterator():
            centers = [int(x) for x in batch[:, 0]]
            positives = [int(x) for x in batch[:, 1]]
            negatives = [int(x) for x in batch[:, 2]]
            
            example = tf.train.Example(features=tf.train.Features(feature={
                'centers': tf.train.Feature(int64_list=tf.train.Int64List(value=centers)),
                'positives': tf.train.Feature(int64_list=tf.train.Int64List(value=positives)),
                'negatives': tf.train.Feature(int64_list=tf.train.Int64List(value=negatives)),
                'batch_size': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(centers)])),
            }))
            writer.write(example.SerializeToString())
            count += len(centers)

    duration = time.perf_counter() - start
    print(f"Write complete. Triplets written: {count:,}")
    print(f"Write time: {duration:.2f} sec")


def load_triplets_from_tfrecord_optimized(
    tfrecord_path: str,
    compressed: Optional[bool] = None,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    unbatch: bool = True
) -> tf.data.Dataset:
    """
    Load triplets from TFRecord with optimized batch processing.

    Parameters
    ----------
    tfrecord_path : str
        Path to the triplet TFRecord file.
    compressed : bool, optional
        Whether the file is GZIP compressed. Auto-detected if None.
    num_parallel_reads : int, optional
        Number of parallel readers. Default is AUTOTUNE.
    unbatch : bool, optional
        Whether to unbatch into individual triplets. Default is True.

    Returns
    -------
    tf.data.Dataset
        Dataset of (center, positive, negative) triplets.
    """
    if compressed is None:
        compressed = tfrecord_path.endswith(".gz")

    compression_type = "GZIP" if compressed else None

    print(f"Loading TFRecord from: {tfrecord_path}")
    start = time.perf_counter()

    def parse_batch_triplet_example(example_proto):
        feature_description = {
            'centers': tf.io.VarLenFeature(tf.int64),
            'positives': tf.io.VarLenFeature(tf.int64),
            'negatives': tf.io.VarLenFeature(tf.int64),
            'batch_size': tf.io.FixedLenFeature([1], tf.int64),
        }
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        
        centers = tf.sparse.to_dense(parsed['centers'])
        positives = tf.sparse.to_dense(parsed['positives'])
        negatives = tf.sparse.to_dense(parsed['negatives'])
        
        # Stack into triplets
        return tf.stack([centers, positives, negatives], axis=1)

    raw_ds = tf.data.TFRecordDataset(
        tfrecord_path,
        buffer_size=64 << 20,  # 64MB buffer
        compression_type=compression_type,
        num_parallel_reads=num_parallel_reads
    )

    batch_ds = raw_ds.map(parse_batch_triplet_example, num_parallel_calls=tf.data.AUTOTUNE)
    
    if unbatch:
        # Unbatch to individual triplets
        dataset = batch_ds.flat_map(lambda batch: tf.data.Dataset.from_tensor_slices(batch))
    else:
        dataset = batch_ds

    duration = time.perf_counter() - start
    print(f"TFRecord loaded and parsed")
    print(f"Load time (lazy initialization): {duration:.3f} sec")

    return dataset


# Alternative: Use TensorFlow's native serialization format
def write_vocab_to_checkpoint(
    vocab_table: tf.lookup.StaticHashTable,
    output_path: str
) -> None:
    """
    Save vocabulary using TensorFlow's checkpoint format (fastest option).
    
    Parameters
    ----------
    vocab_table : tf.lookup.StaticHashTable
        The vocabulary lookup table to save.
    output_path : str
        Path for the output checkpoint directory.
    """
    print(f"Writing vocabulary checkpoint to: {output_path}")
    start = time.perf_counter()
    
    # Create a checkpoint
    checkpoint = tf.train.Checkpoint(vocab_table=vocab_table)
    checkpoint.write(output_path)
    
    duration = time.perf_counter() - start
    vocab_size = vocab_table.size().numpy()
    print(f"Vocabulary checkpoint complete. Words: {vocab_size:,}")
    print(f"Write time: {duration:.2f} sec")


def load_vocab_from_checkpoint(
    checkpoint_path: str,
    default_value: int = 0
) -> tf.lookup.StaticHashTable:
    """
    Load vocabulary from TensorFlow checkpoint (fastest option).
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the vocabulary checkpoint.
    default_value : int, optional
        Default value for unknown words. Default is 0.
        
    Returns
    -------
    tf.lookup.StaticHashTable
        Reconstructed vocabulary lookup table.
    """
    print(f"Loading vocabulary checkpoint from: {checkpoint_path}")
    start = time.perf_counter()
    
    # Create a dummy table for the checkpoint structure
    dummy_vocab = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(['dummy']),
            values=tf.constant([0], dtype=tf.int64)
        ),
        default_value=default_value
    )
    
    # Load from checkpoint
    checkpoint = tf.train.Checkpoint(vocab_table=dummy_vocab)
    checkpoint.read(checkpoint_path)
    
    duration = time.perf_counter() - start
    vocab_size = dummy_vocab.size().numpy()
    print(f"Vocabulary loaded. Size: {vocab_size:,} words")
    print(f"Load time: {duration:.2f} sec")
    
    return dummy_vocab
