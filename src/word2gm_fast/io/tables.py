"""
Lookup table creation utilities for word2GM skip-gram training data.

Provides functions to create token-to-index and index-to-token 
lookup tables from vocabulary TFRecord files with optimized batched processing.
"""

import tensorflow as tf
import time
from IPython import display
from .vocab import parse_vocab_example


def create_token_to_index_table(
    tfrecord_path: str,
    compressed: bool = None,
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
        Token-to-index lookup table.
    """
    if compressed is None:
        compressed = tfrecord_path.endswith(".gz")

    compression_type = "GZIP" if compressed else None

    display.display_markdown(
        f"<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>Loading token-to-index vocabulary TFRecord from:<br>&nbsp;&nbsp;{tfrecord_path}</span>",
        raw=True
    )
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
        for word, id_val in zip(word_batch.numpy(), id_batch.numpy()):
            words.append(word)
            ids.append(id_val)
    
    # Create the lookup table
    token_to_index_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(words),
            values=tf.constant(ids, dtype=tf.int64)
        ),
        default_value=tf.constant(default_value, dtype=tf.int64)
    )

    duration = time.perf_counter() - start
    display.display_markdown(
        f"<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>Token-to-index lookup table created successfully.<br>Table contains {len(words)} tokens. Processing time: {duration:.2f}s</span>",
        raw=True
    )

    return token_to_index_table


def create_index_to_token_table(
    tfrecord_path: str,
    compressed: bool = None,
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
        Index-to-token lookup table.
    """
    if compressed is None:
        compressed = tfrecord_path.endswith(".gz")

    compression_type = "GZIP" if compressed else None

    display.display_markdown(
        f"<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>Loading index-to-token vocab TFRecord from:<br>&nbsp;&nbsp;{tfrecord_path}</span>",
        raw=True
    )
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
        for word, id_val in zip(word_batch.numpy(), id_batch.numpy()):
            words.append(word.decode('utf-8'))
            ids.append(id_val)

    # Create the lookup table
    index_to_token_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(ids, dtype=tf.int64),
            values=tf.constant(words),
        ),
        default_value=tf.constant(default_value, dtype=tf.string)
    )

    duration = time.perf_counter() - start
    display.display_markdown(
        f"<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>Index-to-token lookup table created successfully.<br>Table contains {len(words)} tokens. Processing time: {duration:.2f}s</span>",
        raw=True
    )

    return index_to_token_table
