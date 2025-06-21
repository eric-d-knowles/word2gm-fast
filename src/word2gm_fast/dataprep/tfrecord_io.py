"""
TFRecord I/O utilities for word2vec skip-gram training data.

Provides functions to save and load vocabulary tables and triplet datasets
as TFRecord files for efficient training data management.
"""

import os
import tensorflow as tf
import time
from typing import Dict, List, Optional, Union


def write_triplet_dataset_to_tfrecord(
    dataset: tf.data.Dataset,
    output_path: str,
    compress: bool = False
) -> None:
    """
    Stream skip-gram triplets from a tf.data.Dataset into a TFRecord file.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Dataset of (center, positive, negative) triplets as int64 tensors.
    output_path : str
        Path for the output TFRecord file.
    compress : bool, optional
        Whether to compress the file with GZIP. Default is False.
    """
    if compress and not output_path.endswith(".gz"):
        output_path += ".gz"

    options = tf.io.TFRecordOptions(compression_type="GZIP") if compress else None

    print(f"Writing TFRecord to: {output_path}")
    start = time.perf_counter()

    count = 0
    with tf.io.TFRecordWriter(output_path, options=options) as writer:
        for triplet in dataset.as_numpy_iterator():
            center, positive, negative = (int(x) for x in triplet)
            example = tf.train.Example(features=tf.train.Features(feature={
                'center': tf.train.Feature(int64_list=tf.train.Int64List(value=[center])),
                'positive': tf.train.Feature(int64_list=tf.train.Int64List(value=[positive])),
                'negative': tf.train.Feature(int64_list=tf.train.Int64List(value=[negative])),
            }))
            writer.write(example.SerializeToString())
            count += 1

    duration = time.perf_counter() - start
    print(f"Write complete. Triplets written: {count:,}")
    print(f"Write time: {duration:.2f} sec")


def parse_triplet_example(example_proto: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Parse a single triplet example from TFRecord.

    Parameters
    ----------
    example_proto : tf.Tensor
        Serialized tf.train.Example containing triplet data.

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        (center, positive, negative) as int64 tensors.
    """
    feature_description = {
        'center': tf.io.FixedLenFeature([], tf.int64),
        'positive': tf.io.FixedLenFeature([], tf.int64),
        'negative': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    return parsed['center'], parsed['positive'], parsed['negative']


def load_triplets_from_tfrecord(
    tfrecord_path: Union[str, List[str]],
    compressed: Optional[bool] = None,
    num_parallel_reads: Optional[int] = tf.data.AUTOTUNE
) -> tf.data.Dataset:
    """
    Load skip-gram triplets from a TFRecord file (optionally gzipped).

    Parameters
    ----------
    tfrecord_path : str or list of str
        Path to the TFRecord file(s).
    compressed : bool, optional
        Whether the file(s) are GZIP compressed. Auto-detected if None.
    num_parallel_reads : int, optional
        Number of parallel readers for TFRecord files. Default is AUTOTUNE.

    Returns
    -------
    tf.data.Dataset
        Parsed dataset of (center, positive, negative) triplets as int64 tensors.
    """
    if compressed is None:
        # Auto-detect only if a single file is passed
        if isinstance(tfrecord_path, str):
            compressed = tfrecord_path.endswith(".gz")
        else:
            compressed = False  # Assume uncompressed when list is passed

    compression_type = "GZIP" if compressed else None

    print(f"Loading TFRecord from: {tfrecord_path}")
    start = time.perf_counter()

    raw_ds = tf.data.TFRecordDataset(
        tfrecord_path,
        buffer_size=64 << 20,  # 64MB buffer
        compression_type=compression_type,
        num_parallel_reads=num_parallel_reads
    )

    parsed_ds = raw_ds.map(
        parse_triplet_example, 
        num_parallel_calls=tf.data.AUTOTUNE
    )

    duration = time.perf_counter() - start
    print(f"TFRecord loaded and parsed")
    print(f"Load time (lazy initialization): {duration:.3f} sec")

    return parsed_ds


def write_vocab_to_tfrecord(
    vocab_table: tf.lookup.StaticHashTable,
    output_path: str,
    compress: bool = False
) -> None:
    """
    Save a vocabulary lookup table to a TFRecord file.
    
    Parameters
    ----------
    vocab_table : tf.lookup.StaticHashTable
        The vocabulary lookup table to save.
    output_path : str
        Path for the output TFRecord file.
    compress : bool, optional
        Whether to compress the file with GZIP. Default is False.
    """
    if compress and not output_path.endswith(".gz"):
        output_path += ".gz"

    options = tf.io.TFRecordOptions(compression_type="GZIP") if compress else None

    print(f"Writing vocabulary TFRecord to: {output_path}")
    start = time.perf_counter()

    # Export the vocabulary table
    vocab_keys, vocab_values = vocab_table.export()
    vocab_keys_np = vocab_keys.numpy()
    vocab_values_np = vocab_values.numpy()

    count = 0
    with tf.io.TFRecordWriter(output_path, options=options) as writer:
        for word_bytes, word_id in zip(vocab_keys_np, vocab_values_np):
            word = word_bytes.decode('utf-8')
            example = tf.train.Example(features=tf.train.Features(feature={
                'word': tf.train.Feature(bytes_list=tf.train.BytesList(value=[word.encode('utf-8')])),
                'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(word_id)])),
            }))
            writer.write(example.SerializeToString())
            count += 1

    duration = time.perf_counter() - start
    print(f"Vocabulary write complete. Words written: {count:,}")
    print(f"Write time: {duration:.2f} sec")


def parse_vocab_example(example_proto: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Parse a single vocabulary example from TFRecord.

    Parameters
    ----------
    example_proto : tf.Tensor
        Serialized tf.train.Example containing vocabulary data.

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor]
        (word, id) as string and int64 tensors.
    """
    feature_description = {
        'word': tf.io.FixedLenFeature([], tf.string),
        'id': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    return parsed['word'], parsed['id']


def load_vocab_from_tfrecord(
    tfrecord_path: str,
    compressed: Optional[bool] = None,
    default_value: int = 0
) -> tf.lookup.StaticHashTable:
    """
    Load a vocabulary lookup table from a TFRecord file.
    
    Parameters
    ----------
    tfrecord_path : str
        Path to the vocabulary TFRecord file.
    compressed : bool, optional
        Whether the file is GZIP compressed. Auto-detected if None.
    default_value : int, optional
        Default value for unknown words (typically 0 for UNK). Default is 0.
        
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

    # Load the raw TFRecord dataset
    raw_ds = tf.data.TFRecordDataset(
        tfrecord_path,
        compression_type=compression_type
    )
    
    # Parse the vocabulary entries
    vocab_ds = raw_ds.map(parse_vocab_example)
    
    # Extract words and IDs - NOTE: This materializes the dataset
    words = []
    ids = []
    for word, word_id in vocab_ds:
        words.append(word.numpy())
        ids.append(word_id.numpy())
    
    # Create the lookup table
    vocab_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(words),
            values=tf.constant(ids, dtype=tf.int64)
        ),
        default_value=default_value
    )

    duration = time.perf_counter() - start
    print(f"Vocabulary loaded. Size: {len(words):,} words")
    print(f"Load time: {duration:.2f} sec")

    return vocab_table


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
        Paths to the saved files and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ext = ".tfrecord.gz" if compress else ".tfrecord"
    
    vocab_path = os.path.join(output_dir, f"vocab{ext}")
    triplets_path = os.path.join(output_dir, f"triplets{ext}")
    
    print(f"Saving pipeline artifacts to: {output_dir}")
    
    # Save vocabulary
    write_vocab_to_tfrecord(vocab_table, vocab_path, compress=compress)
    
    # Save triplets
    write_triplet_dataset_to_tfrecord(triplets_ds, triplets_path, compress=compress)
    
    artifacts = {
        'vocab_path': vocab_path,
        'triplets_path': triplets_path,
        'vocab_size': int(vocab_table.size().numpy()),
        'compressed': compress,
        'output_dir': output_dir
    }
    
    print("All artifacts saved successfully!")
    return artifacts


def load_pipeline_artifacts(
    output_dir: str, 
    compressed: Optional[bool] = None
) -> Dict[str, Union[tf.lookup.StaticHashTable, tf.data.Dataset, int]]:
    """
    Load all pipeline artifacts from TFRecord files.
    
    Parameters
    ----------
    output_dir : str
        Directory containing the TFRecord files.
    compressed : bool, optional
        Whether files are compressed. Auto-detected if None.
        
    Returns
    -------
    Dict[str, Union[tf.lookup.StaticHashTable, tf.data.Dataset, int]]
        Loaded vocabulary table and triplets dataset.
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
    
    print(f"Loading pipeline artifacts from: {output_dir}")
    
    # Load vocabulary
    vocab_table = load_vocab_from_tfrecord(vocab_path, compressed=compressed)
    
    # Load triplets
    triplets_ds = load_triplets_from_tfrecord(triplets_path, compressed=compressed)
    
    artifacts = {
        'vocab_table': vocab_table,
        'triplets_ds': triplets_ds,
        'vocab_size': int(vocab_table.size().numpy()),
    }
    
    print("All artifacts loaded successfully!")
    return artifacts