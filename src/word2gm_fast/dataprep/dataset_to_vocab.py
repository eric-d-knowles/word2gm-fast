# dataprep/dataset_to_vocab.py

"""
Minimal utilities for building a vocabulary and TensorFlow lookup table
from a tf.data.Dataset of text lines. The UNK token is always 'UNK' and
will be at index 0. All other tokens are sorted alphabetically.
"""

import tensorflow as tf


def build_vocab_table(vocab_list: list[str]) -> tf.lookup.StaticHashTable:
    """
    Build a TensorFlow StaticHashTable mapping words to integer IDs.

    The vocabulary list must be ordered by ID (i.e., index = ID), with
    the UNK token ('UNK') at index 0. All other tokens should be sorted
    alphabetically or as desired by the pipeline.

    Parameters
    ----------
    vocab_list : list[str]
        List of vocabulary tokens, with 'UNK' at index 0.

    Returns
    -------
    tf.lookup.StaticHashTable
        A lookup table mapping tokens to integer IDs.
    """
    if vocab_list[0] != "UNK":
        raise ValueError(f"UNK token must be at index 0, got {vocab_list[0]}")
    keys = tf.constant(vocab_list, dtype=tf.string)
    values = tf.range(len(vocab_list), dtype=tf.int32)
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=0
    )


def make_vocab(dataset: tf.data.Dataset) -> tf.lookup.StaticHashTable:
    """
    Build a vocab hash table from a tf.data.Dataset of lines.
    The UNK token is always 'UNK' and will be at index 0.
    All other tokens are sorted alphabetically.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Dataset of lines (strings) to build the vocab from.

    Returns
    -------
    tf.lookup.StaticHashTable
        A lookup table mapping tokens to integer IDs.
    """
    print("Scanning dataset for vocab...")
    vocab_set = set()
    for line in dataset.as_numpy_iterator():
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        tokens = line.strip().split()
        vocab_set.update(tokens)
    vocab = ["UNK"] + sorted(tok for tok in vocab_set if tok != "UNK")
    print(f"Vocab size: {len(vocab)}")
    return build_vocab_table(vocab)


def make_vocab_tf(dataset: tf.data.Dataset) -> tf.lookup.StaticHashTable:
    """
    Build a vocab hash table from a tf.data.Dataset of lines using only
    TensorFlow ops (scalable for large corpora).

    The UNK token is always 'UNK' and will be at index 0. All other tokens
    are sorted alphabetically.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Dataset of lines (strings) to build the vocab from.

    Returns
    -------
    tf.lookup.StaticHashTable
        A lookup table mapping tokens to integer IDs.
    """
    tokens = dataset.flat_map(
        lambda line: tf.data.Dataset.from_tensor_slices(tf.strings.split(line))
    )
    unique_tokens = tokens.apply(tf.data.experimental.unique())
    vocab_bytes = list(unique_tokens.as_numpy_iterator())
    vocab = ["UNK"] + sorted(
        tok.decode("utf-8") for tok in vocab_bytes if tok.decode("utf-8") != "UNK"
    )
    return build_vocab_table(vocab)
