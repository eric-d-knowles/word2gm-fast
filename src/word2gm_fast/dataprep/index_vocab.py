"""
Convert a tf.data.Dataset of lines into a tf.lookup.StaticHashTable containing
the corpus vocabulary. 'UNK' will be at index 0. All other tokens are sorted 
and indexed alphabetically.
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
        raise ValueError(
            f"UNK token must be at index 0, got {vocab_list[0]}"
        )
    keys = tf.constant(vocab_list, dtype=tf.string)
    values = tf.range(len(vocab_list), dtype=tf.int32)
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=0
    )


def make_vocab(dataset: tf.data.Dataset) -> tf.lookup.StaticHashTable:
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
    tokenized = dataset.map(
        lambda line: tf.strings.split(line),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    tokens = tokenized.flat_map(
        lambda tokens: tf.data.Dataset.from_tensor_slices(tokens)
    )
    unique_tokens = tokens.unique()
    vocab_bytes = list(unique_tokens.as_numpy_iterator())
    vocab = ["UNK"] + sorted(
        tok.decode("utf-8") 
        for tok in vocab_bytes 
        if tok.decode("utf-8") != "UNK"
    )
    return build_vocab_table(vocab)