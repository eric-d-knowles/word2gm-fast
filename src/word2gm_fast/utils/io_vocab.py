"""
Vocabulary TFRecord I/O utilities for word2GM skip-gram training data.

Provides functions to save and load vocabulary tables with frequencies as TFRecord files.
"""

import tensorflow as tf
import time
from typing import Optional
from IPython.display import display, Markdown


def write_vocab_file(vocab_list, filepath):
    """
    Write the vocabulary list to a text file, one word per line, in vocab index order.
    The first word should be 'UNK' at index 0.

    Parameters
    ----------
    vocab_list : list[str]
        List of vocabulary tokens, with 'UNK' at index 0.
    filepath : str
        Path to the output vocab file.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for word in vocab_list:
            f.write(f"{word}\n")


def write_vocab_to_tfrecord(
    vocab_table: tf.lookup.StaticHashTable,
    output_path: str,
    frequencies: Optional[dict] = None,
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
        Mapping from word (str) to frequency (int or float). If None, frequency is set to 0.
    compress : bool, optional
        Whether to compress the file with GZIP. Default is False.
    """
    if compress and not output_path.endswith(".gz"):
        output_path += ".gz"
    options = tf.io.TFRecordOptions(compression_type="GZIP") if compress else None
    display(Markdown(f"Writing vocabulary TFRecord to: {output_path}"))
    start = time.perf_counter()
    # Export the vocabulary table
    vocab_keys, vocab_values = vocab_table.export()
    vocab_keys_np = vocab_keys.numpy()
    vocab_values_np = vocab_values.numpy()
    count = 0
    with tf.io.TFRecordWriter(output_path, options=options) as writer:
        for word_bytes, word_id in zip(vocab_keys_np, vocab_values_np):
            word = word_bytes.decode('utf-8')
            freq = 0
            if frequencies is not None:
                freq = frequencies.get(word, 0)
            example = tf.train.Example(features=tf.train.Features(feature={
                'word': tf.train.Feature(bytes_list=tf.train.BytesList(value=[word.encode('utf-8')])),
                'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(word_id)])),
                'frequency': tf.train.Feature(float_list=tf.train.FloatList(value=[float(freq)])),
            }))
            writer.write(example.SerializeToString())
            count += 1
    duration = time.perf_counter() - start
    display(Markdown(f"<pre>Vocabulary write complete. Words written: {count:,}</pre>"))


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
        (word, id, frequency) as string, int64, and float32 tensors.
    """
    feature_description = {
        'word': tf.io.FixedLenFeature([], tf.string),
        'id': tf.io.FixedLenFeature([], tf.int64),
        'frequency': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    return parsed['word'], parsed['id'], parsed['frequency']
