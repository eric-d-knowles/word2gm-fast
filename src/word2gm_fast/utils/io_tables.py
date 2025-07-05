"""
Lookup table creation utilities for word2GM skip-gram training data.

Provides functions to create token-to-index and index-to-token lookup tables
from vocabulary TFRecord files with optimized batched processing.
"""

import tensorflow as tf
import time
from typing import Optional
from IPython.display import display, Markdown
from .io_vocab import parse_vocab_example


def create_token_to_index_table(
    tfrecord_path: str,
    compressed: Optional[bool] = None,
    default_value: int = 0,
    batch_size: int = 1000
) -> tf.lookup.StaticHashTable:
    """
    Create a token-to-index lookup table from a TFRecord vocab file.
    
    Parameters
    ----------
    tfrecord_path : str
        Path to the vocabulary TFRecord file.
    compressed : bool, optional
        Whether the file is GZIP compressed. Auto-detected if None.
    default_value : int, optional
        Default value for unknown words (typically 0 for UNK). Default is 0.
    batch_size : int, optional
        Batch size for processing vocabulary entries. Default is 1000.
    Returns
    -------
    tf.lookup.StaticHashTable
        Reconstructed token-to-index lookup table.
    """
    if compressed is None:
        compressed = tfrecord_path.endswith(".gz")

    compression_type = "GZIP" if compressed else None

    display(Markdown(f"<pre>Loading token-to-index vocabulary TFRecord from: {tfrecord_path}</pre>"))
    start = time.perf_counter()

    # Load the raw TFRecord dataset with optimized buffer settings
    raw_ds = tf.data.TFRecordDataset(
        tfrecord_path,
        compression_type=compression_type,
        buffer_size=128 << 20
    )
    
    # Parse the vocabulary entries with batching and prefetching
    vocab_ds = raw_ds.map(
        parse_vocab_example,
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Extract words and IDs using optimized batched processing
    words = []
    ids = []
    for word_batch, id_batch, _ in vocab_ds:
        words.extend(word_batch.numpy())
        ids.extend(id_batch.numpy())
    
    # Create the lookup table
    token_to_index_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(words),
            values=tf.constant(ids, dtype=tf.int64)
        ),
        default_value=default_value
    )

    duration = time.perf_counter() - start

    return token_to_index_table


def create_index_to_token_table(
    tfrecord_path: str,
    compressed: Optional[bool] = None,
    default_value: str = "UNK",
    batch_size: int = 1000
) -> tf.lookup.StaticHashTable:
    """
    Create an index-to-token lookup table from a TFRecord vocab file.
    
    Parameters
    ----------
    tfrecord_path : str
        Path to the vocabulary TFRecord file.
    compressed : bool, optional
        Whether the file is GZIP compressed. Auto-detected if None.
    default_value : str, optional
        Default value for unknown indices (typically 'UNK'). Default is 'UNK'.
    batch_size : int, optional
        Batch size for processing vocabulary entries. Default is 1000.
    Returns
    -------
    tf.lookup.StaticHashTable
        Reconstructed index-to-token lookup table.
    """
    if compressed is None:
        compressed = tfrecord_path.endswith(".gz")

    compression_type = "GZIP" if compressed else None

    display(Markdown(f"<pre>Loading index-to-token vocab TFRecord from: {tfrecord_path}</pre>"))
    start = time.perf_counter()

    # Load the raw TFRecord dataset with optimized buffer settings
    raw_ds = tf.data.TFRecordDataset(
        tfrecord_path,
        compression_type=compression_type,
        buffer_size=128 << 20
    )

    # Parse the vocabulary entries with batching and prefetching
    vocab_ds = raw_ds.map(
        parse_vocab_example,
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Extract words and IDs using optimized batched processing
    words = []
    ids = []
    for word_batch, id_batch, _ in vocab_ds:
        words.extend(word_batch.numpy())
        ids.extend(id_batch.numpy())

    # Create the lookup table (index -> token)
    index_to_token_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(ids, dtype=tf.int64),
            values=tf.constant([w.decode('utf-8') for w in words]),
        ),
        default_value=default_value
    )

    duration = time.perf_counter() - start

    return index_to_token_table
