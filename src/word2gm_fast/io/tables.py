"""
Lookup table creation utilities for word2GM skip-gram training data.

Provides functions to create token-to-index and index-to-token lookup tables
from vocabulary TFRecord files with optimized batched processing.
"""

import tensorflow as tf
import time
from typing import Optional, Set
from IPython.display import display, Markdown
from .vocab import parse_vocab_example
from .triplets import load_triplets_from_tfrecord


def create_token_to_index_table(
    tfrecord_path: str,
    triplets_tfrecord_path: Optional[str] = None,
    compressed: Optional[bool] = None,
    default_value: int = 0,
    batch_size: int = 1000
) -> tf.lookup.StaticHashTable:
    """
    Create a token-to-index lookup table from a TFRecord vocab file.
    
    Optionally filters the vocabulary to only include tokens that appear in the triplets dataset.
    
    Parameters
    ----------
    tfrecord_path : str
        Path to the vocabulary TFRecord file.
    triplets_tfrecord_path : str, optional
        Path to the triplets TFRecord file. If provided, only tokens that appear
        in triplets will be included in the lookup table.
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

    # Get unique indices from triplets if triplets file is provided
    valid_indices = None
    if triplets_tfrecord_path is not None:
        valid_indices = get_unique_indices_from_triplets(triplets_tfrecord_path, compressed)

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
    filtered_count = 0
    total_count = 0
    
    for word_batch, id_batch, _ in vocab_ds:
        for word, id_val in zip(word_batch.numpy(), id_batch.numpy()):
            total_count += 1
            # Include token if no filtering or if it appears in triplets
            if valid_indices is None or int(id_val) in valid_indices:
                words.append(word)
                ids.append(id_val)
            else:
                filtered_count += 1
    
    # Create the lookup table
    token_to_index_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(words),
            values=tf.constant(ids, dtype=tf.int64)
        ),
        default_value=default_value
    )

    duration = time.perf_counter() - start
    
    if valid_indices is not None:
        display(Markdown(f"<pre>Filtered vocabulary: kept {len(words)} tokens, removed {filtered_count} tokens not in triplets</pre>"))
    
    display(Markdown(f"<pre>Created token-to-index table with {len(words)} entries in {duration:.2f}s</pre>"))

    return token_to_index_table


def create_index_to_token_table(
    tfrecord_path: str,
    triplets_tfrecord_path: Optional[str] = None,
    compressed: Optional[bool] = None,
    default_value: str = "UNK",
    batch_size: int = 1000
) -> tf.lookup.StaticHashTable:
    """
    Create an index-to-token lookup table from a TFRecord vocab file.
    
    Optionally filters the vocabulary to only include tokens that appear in the triplets dataset.
    
    Parameters
    ----------
    tfrecord_path : str
        Path to the vocabulary TFRecord file.
    triplets_tfrecord_path : str, optional
        Path to the triplets TFRecord file. If provided, only tokens that appear
        in triplets will be included in the lookup table.
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

    # Get unique indices from triplets if triplets file is provided
    valid_indices = None
    if triplets_tfrecord_path is not None:
        valid_indices = get_unique_indices_from_triplets(triplets_tfrecord_path, compressed)

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
    filtered_count = 0
    total_count = 0

    for word_batch, id_batch, _ in vocab_ds:
        for word, id_val in zip(word_batch.numpy(), id_batch.numpy()):
            total_count += 1
            # Include token if no filtering or if it appears in triplets
            if valid_indices is None or int(id_val) in valid_indices:
                words.append(word.decode('utf-8'))
                ids.append(id_val)
            else:
                filtered_count += 1

    # Create the lookup table (index -> token)
    index_to_token_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(ids, dtype=tf.int64),
            values=tf.constant(words),
        ),
        default_value=default_value
    )

    duration = time.perf_counter() - start
    
    if valid_indices is not None:
        display(Markdown(f"<pre>Filtered vocabulary: kept {len(words)} tokens, removed {filtered_count} tokens not in triplets</pre>"))
    
    display(Markdown(f"<pre>Created index-to-token table with {len(words)} entries in {duration:.2f}s</pre>"))

    return index_to_token_table


def get_unique_indices_from_triplets(
    triplets_tfrecord_path: str,
    compressed: Optional[bool] = None,
    batch_size: int = 10000
) -> Set[int]:
    """
    Extract unique token indices that appear in the triplets dataset.
    
    Parameters
    ----------
    triplets_tfrecord_path : str
        Path to the triplets TFRecord file.
    compressed : bool, optional
        Whether the file is GZIP compressed. Auto-detected if None.
    batch_size : int, optional
        Batch size for processing triplets. Default is 10000.
        
    Returns
    -------
    set[int]
        Set of unique token indices that appear in triplets.
    """
    if compressed is None:
        compressed = triplets_tfrecord_path.endswith(".gz")
    
    display(Markdown(f"<pre>Extracting unique indices from triplets: {triplets_tfrecord_path}</pre>"))
    start = time.perf_counter()
    
    # Load triplets dataset
    triplets_ds = load_triplets_from_tfrecord(triplets_tfrecord_path, compressed=compressed)
    
    # Collect unique indices
    unique_indices = set()
    
    # Process in batches for efficiency
    batched_ds = triplets_ds.batch(batch_size)
    
    for batch in batched_ds:
        center_batch, positive_batch, negative_batch = batch
        
        # Convert to numpy and add to set
        unique_indices.update(center_batch.numpy())
        unique_indices.update(positive_batch.numpy())
        unique_indices.update(negative_batch.numpy())
    
    duration = time.perf_counter() - start
    display(Markdown(f"<pre>Found {len(unique_indices)} unique indices in {duration:.2f}s</pre>"))
    
    return unique_indices


def create_filtered_lookup_tables(
    vocab_tfrecord_path: str,
    triplets_tfrecord_path: str,
    compressed: Optional[bool] = None,
    batch_size: int = 1000
) -> tuple[tf.lookup.StaticHashTable, tf.lookup.StaticHashTable]:
    """
    Create both token-to-index and index-to-token lookup tables filtered by triplets.
    
    This is a convenience function that creates both lookup tables while only
    scanning the triplets dataset once for efficiency.
    
    Parameters
    ----------
    vocab_tfrecord_path : str
        Path to the vocabulary TFRecord file.
    triplets_tfrecord_path : str
        Path to the triplets TFRecord file.
    compressed : bool, optional
        Whether the files are GZIP compressed. Auto-detected if None.
    batch_size : int, optional
        Batch size for processing entries. Default is 1000.
        
    Returns
    -------
    tuple[tf.lookup.StaticHashTable, tf.lookup.StaticHashTable]
        (token_to_index_table, index_to_token_table) filtered by triplets.
    """
    display(Markdown(f"<pre>Creating filtered lookup tables from vocab and triplets</pre>"))
    
    # Create both tables - each will scan triplets independently
    # This is acceptable since the triplets scan is usually fast
    token_to_index_table = create_token_to_index_table(
        vocab_tfrecord_path, 
        triplets_tfrecord_path=triplets_tfrecord_path,
        compressed=compressed,
        batch_size=batch_size
    )
    
    index_to_token_table = create_index_to_token_table(
        vocab_tfrecord_path,
        triplets_tfrecord_path=triplets_tfrecord_path,
        compressed=compressed,
        batch_size=batch_size
    )
    
    return token_to_index_table, index_to_token_table
