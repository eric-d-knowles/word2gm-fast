"""
Pipeline artifacts management for word2GM skip-gram training data.

Provides functions to save and load complete pipeline artifacts including 
vocabulary tables and triplets datasets as TFRecord files.
"""

import os
from ..utils.tf_silence import import_tf_quietly
from IPython import display
from typing import Dict, Optional, Union
from .vocab import write_vocab_to_tfrecord
from .triplets import write_triplets_to_tfrecord, load_triplets_from_tfrecord
from .tables import create_token_to_index_table, create_index_to_token_table

# Import TensorFlow silently
tf = import_tf_quietly(force_cpu=True)


def save_pipeline_artifacts(
    dataset: tf.data.Dataset,
    vocab_table: tf.lookup.StaticHashTable,
    triplets_ds: tf.data.Dataset,
    output_dir: str,
    compress: bool = True
) -> Dict[str, Union[str, int, bool]]:
    """
    Save all pipeline artifacts (vocab + triplets) to TFRecord files.
    
    Parameters
    ----------
    dataset : tf.data.Dataset
        The original text dataset (for reference/metadata).
    vocab_table : tf.lookup.StaticHashTable
        The vocabulary lookup table.
    triplets_ds : tf.data.Dataset
        The skip-gram triplets dataset.
    output_dir : str
        Directory to save the TFRecord files.
    compress : bool, optional
        Whether to compress files with GZIP. Default is True.
        
    Returns
    -------
    Dict[str, Union[str, int, bool]]
        Paths to the saved files and metadata including triplet count.
    """
    os.makedirs(output_dir, exist_ok=True)

    ext = ".tfrecord.gz" if compress else ".tfrecord"
    vocab_path = os.path.join(output_dir, f"vocab{ext}")
    triplets_path = os.path.join(output_dir, f"triplets{ext}")
    display.display_markdown(
        f"<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>Saving pipeline artifacts to: {output_dir}</span>",
        raw=True
    )
    
    # Save vocabulary TFRecord
    # Try to get frequencies if available (from vocab_table, vocab_list, frequencies tuple)
    frequencies = None
    if hasattr(vocab_table, 'frequencies'):
        frequencies = vocab_table.frequencies
    elif isinstance(vocab_table, tuple) and len(vocab_table) == 3:
        # If user passed (vocab_table, vocab_list, frequencies)
        _, _, frequencies = vocab_table
        vocab_table = vocab_table[0]
    write_vocab_to_tfrecord(vocab_table, vocab_path, frequencies=frequencies, compress=compress)
    
    # Save triplets and get count in one pass
    triplet_count = write_triplets_to_tfrecord(triplets_ds, triplets_path, compress=compress)
    
    artifacts = {
        'vocab_path': vocab_path,
        'triplets_path': triplets_path,
        'vocab_size': int(vocab_table.size().numpy()),
        'triplet_count': triplet_count,
        'compressed': compress,
        'output_dir': output_dir
    }
    display.display_markdown(
        "<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>All artifacts saved successfully!</span>",
        raw=True
    )
    return artifacts


def load_pipeline_artifacts(
    output_dir: str, 
    compressed: Optional[bool] = None,
    filter_to_triplets: bool = False
) -> Dict[str, Union[tf.lookup.StaticHashTable, tf.data.Dataset, int]]:
    """
    Load all pipeline artifacts from TFRecord files.
    
    Parameters
    ----------
    output_dir : str
        Directory containing the TFRecord files.
    compressed : bool, optional
        Whether files are compressed. Auto-detected if None.
    filter_to_triplets : bool, optional
        If True, filter vocabulary tables to only include tokens that appear
        in the triplets dataset. This prevents querying of heavily downsampled
        tokens and ensures all accessible tokens have reliable embeddings.
        Default is False for backward compatibility.
        
    Returns
    -------
    Dict[str, Union[tf.lookup.StaticHashTable, tf.data.Dataset, int]]
        Loaded vocabulary tables and triplets dataset.
    """
    if compressed is None:
        # Auto-detect based on available files
        if os.path.exists(os.path.join(output_dir, "vocab.tfrecord.gz")):
            compressed = True
        elif os.path.exists(os.path.join(output_dir, "vocab.tfrecord")):
            compressed = False
        else:
            raise FileNotFoundError(f"No vocabulary TFRecord files found in {output_dir}")
    
    ext = ".tfrecord.gz" if compressed else ".tfrecord"
    vocab_path = os.path.join(output_dir, f"vocab{ext}")
    triplets_path = os.path.join(output_dir, f"triplets{ext}")

    if filter_to_triplets:
        display.display_markdown(
            f"<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>Loading filtered pipeline artifacts from: {output_dir}</span>",
            raw=True
        )
        display.display_markdown(
            "<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>Filtering vocabulary to tokens that appear in triplets...</span>",
            raw=True
        )
        
        # Load vocabulary tables with filtering
        token_to_index_table = create_token_to_index_table(
            vocab_path, 
            triplets_tfrecord_path=triplets_path,
            compressed=compressed
        )
        index_to_token_table = create_index_to_token_table(
            vocab_path,
            triplets_tfrecord_path=triplets_path, 
            compressed=compressed
        )
    else:
        display.display_markdown(
            f"<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>Loading pipeline artifacts from: {output_dir}</span>",
            raw=True
        )
        
        # Load vocabulary tables without filtering (original behavior)
        token_to_index_table = create_token_to_index_table(vocab_path, compressed=compressed)
        index_to_token_table = create_index_to_token_table(vocab_path, compressed=compressed)

    # Load triplets and cast to tf.int32 for model compatibility
    triplets_ds = load_triplets_from_tfrecord(triplets_path, compressed=compressed)
    def cast_triplet_to_int32(word_idx, pos_idx, neg_idx):
        return (tf.cast(word_idx, tf.int32),
                tf.cast(pos_idx, tf.int32),
                tf.cast(neg_idx, tf.int32))
    triplets_ds = triplets_ds.map(cast_triplet_to_int32, num_parallel_calls=tf.data.AUTOTUNE)

    artifacts = {
        'token_to_index_table': token_to_index_table,
        'index_to_token_table': index_to_token_table,
        'triplets_ds': triplets_ds,
        'vocab_size': int(token_to_index_table.size().numpy()),
        'filtered_to_triplets': filter_to_triplets
    }

    if filter_to_triplets:
        display.display_markdown(
            "<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>Filtered artifacts loaded successfully!</span>",
            raw=True
        )
    else:
        display.display_markdown(
            "<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>All artifacts loaded successfully!</span>",
            raw=True
        )
    return artifacts
