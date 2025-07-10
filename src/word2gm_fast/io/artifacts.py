"""
Pipeline artifacts management for word2GM skip-gram training data.

Provides functions to save and load complete pipeline artifacts including 
vocabulary tables and triplets datasets as TFRecord files.
"""

import os
import json
import gzip
import tensorflow as tf
from typing import Dict, Optional, Union
from IPython.display import display, Markdown
from .vocab import write_vocab_to_tfrecord
from .triplets import write_triplets_to_tfrecord, load_triplets_from_tfrecord
from .tables import create_token_to_index_table, create_index_to_token_table


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
    display(Markdown(f"<pre>Saving pipeline artifacts to: {output_dir}</pre>"))
    
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
    display(Markdown("<pre>All artifacts saved successfully!</pre>"))
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
        display(Markdown(f"<pre>Loading filtered pipeline artifacts from: {output_dir}</pre>"))
        display(Markdown(f"<pre>Filtering vocabulary to tokens that appear in triplets...</pre>"))
        
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
        display(Markdown(f"<pre>Loading pipeline artifacts from: {output_dir}</pre>"))
        
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
        display(Markdown("<pre>Filtered artifacts loaded successfully!</pre>"))
    else:
        display(Markdown("<pre>All artifacts loaded successfully!</pre>"))
    return artifacts


def save_metadata(metadata: Dict, output_path: str, compress: bool = True) -> str:
    """
    Save metadata dictionary to a JSON file, optionally compressed.
    
    Parameters
    ----------
    metadata : Dict
        Dictionary containing metadata to save.
    output_path : str
        Path to save the metadata file.
    compress : bool, optional
        Whether to compress the file with gzip. Default is True.
        
    Returns
    -------
    str
        The actual path where the file was saved (may include .gz extension).
    """
    if compress and not output_path.endswith('.gz'):
        output_path += '.gz'
    
    metadata_json = json.dumps(metadata, indent=2)
    
    if compress:
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            f.write(metadata_json)
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(metadata_json)
    
    display(Markdown(f"<pre>Metadata saved to: {output_path}</pre>"))
    return output_path


def load_metadata(input_path: str) -> Dict:
    """
    Load metadata from a JSON file, automatically detecting compression.
    
    Parameters
    ----------
    input_path : str
        Path to the metadata file.
        
    Returns
    -------
    Dict
        The loaded metadata dictionary.
    """
    is_compressed = input_path.endswith('.gz')
    
    if is_compressed:
        with gzip.open(input_path, 'rt', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    display(Markdown(f"<pre>Metadata loaded from: {input_path}</pre>"))
    return metadata
