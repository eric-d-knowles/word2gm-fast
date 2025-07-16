"""
Vocabulary TFRecord I/O utilities for word2GM skip-gram training data.

Provides functions to save and load vocabulary tables with frequencies as TFRecord files.
"""

import tensorflow as tf
import time
import os
from typing import Optional, Dict, Union, List
from IPython import display
from ..utils.tf_silence import import_tf_quietly

# Import TensorFlow silently
tf = import_tf_quietly(force_cpu=True)


def write_vocab_to_tfrecord(
    vocab_table: tf.lookup.StaticHashTable,
    output_path: str,
    frequencies: Optional[Dict[str, int]] = None,
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
    frequencies : dict, optional
        Dictionary mapping words to their frequencies. If None, frequencies are set to 0.
    compress : bool, optional
        Whether to compress the file with GZIP. Default is False.
    """
    if compress and not output_path.endswith(".gz"):
        output_path += ".gz"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    options = tf.io.TFRecordOptions(compression_type="GZIP") if compress else None
    display.display_markdown(
        f"<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>Writing vocabulary TFRecord to: {output_path}</span>",
        raw=True
    )
    start = time.perf_counter()
    # Export the vocabulary table
    vocab_keys, vocab_values = vocab_table.export()
    vocab_keys_np = vocab_keys.numpy()
    vocab_values_np = vocab_values.numpy()
    count = 0
    with tf.io.TFRecordWriter(output_path, options=options) as writer:
        for word_bytes, word_id in zip(vocab_keys_np, vocab_values_np):
            word = word_bytes.decode('utf-8')
            frequency = frequencies.get(word, 0) if frequencies else 0
            example = tf.train.Example(features=tf.train.Features(feature={
                'word': tf.train.Feature(bytes_list=tf.train.BytesList(value=[word.encode('utf-8')])),
                'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(word_id)])),
                'frequency': tf.train.Feature(int64_list=tf.train.Int64List(value=[frequency])),
            }))
            writer.write(example.SerializeToString())
            count += 1
    duration = time.perf_counter() - start
    display.display_markdown(
        f"<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>Vocabulary write complete. Words written: {count:,}</span>",
        raw=True
    )


def parse_vocab_example(example_proto: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Parse a single vocabulary example from TFRecord.

    Parameters
    ----------
    example_proto : tf.Tensor
        Serialized tf.train.Example containing vocabulary data.

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        (word, id, frequency) as string, int64, and int64 tensors.
    """
    feature_description = {
        'word': tf.io.FixedLenFeature([], tf.string),
        'id': tf.io.FixedLenFeature([], tf.int64),
        'frequency': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    return parsed['word'], parsed['id'], parsed['frequency']


def load_vocab_from_tfrecord(
    tfrecord_path: Union[str, List[str]],
    compressed: Optional[bool] = None,
    num_parallel_reads: Optional[int] = tf.data.AUTOTUNE
) -> tuple[List[str], Dict[str, int]]:
    """
    Load vocabulary data from a TFRecord file (optionally gzipped).

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
    tuple[List[str], Dict[str, int]]
        (vocab_list, frequencies) - vocabulary words and their frequencies.
    """
    if compressed is None:
        # Auto-detect only if a single file is passed
        if isinstance(tfrecord_path, str):
            compressed = tfrecord_path.endswith(".gz")
        else:
            compressed = False  # Assume uncompressed when list is passed

    compression_type = "GZIP" if compressed else None

    display.display_markdown(f"<pre>Loading vocabulary TFRecord from: {tfrecord_path}</pre>")
    start = time.perf_counter()

    raw_ds = tf.data.TFRecordDataset(
        tfrecord_path,
        buffer_size=64 << 20,  # 64MB buffer
        compression_type=compression_type,
        num_parallel_reads=num_parallel_reads
    )

    parsed_ds = raw_ds.map(
        lambda x: parse_vocab_example(x), 
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Convert dataset to lists
    vocab_list = []
    frequencies = {}
    
    for word_tensor, id_tensor, freq_tensor in parsed_ds:
        word = word_tensor.numpy().decode('utf-8')
        word_id = int(id_tensor.numpy())
        frequency = int(freq_tensor.numpy())
        
        vocab_list.append((word_id, word))  # Store as (id, word) for sorting
        frequencies[word] = frequency
    
    # Sort by ID to preserve original vocabulary order
    vocab_list.sort(key=lambda x: x[0])
    vocab_list = [word for _, word in vocab_list]

    duration = time.perf_counter() - start
    display.display_markdown(f"<pre>Vocabulary TFRecord loaded and parsed</pre>")

    return vocab_list, frequencies
