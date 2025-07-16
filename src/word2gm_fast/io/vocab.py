"""
Vocabulary TFRecord I/O utilities for word2GM skip-gram training data.

Provides functions to save and load vocabulary tables with frequencies as TFRecord files.
"""

import tensorflow as tf
import time
from typing import Optional
from IPython import display
from ..utils.tf_silence import import_tf_quietly

# Import TensorFlow silently
tf = import_tf_quietly(force_cpu=True)


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
            example = tf.train.Example(features=tf.train.Features(feature={
                'word': tf.train.Feature(bytes_list=tf.train.BytesList(value=[word.encode('utf-8')])),
                'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(word_id)])),
            }))
            writer.write(example.SerializeToString())
            count += 1
    duration = time.perf_counter() - start
    display.display_markdown(
        f"<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>Vocabulary write complete. Words written: {count:,}</span>",
        raw=True
    )


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
