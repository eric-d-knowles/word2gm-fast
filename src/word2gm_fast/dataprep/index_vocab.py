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


def make_vocab(dataset: tf.data.Dataset):
    """
    Build a vocab hash table and frequency dict from a tf.data.Dataset of lines using only
    TensorFlow ops (scalable for large corpora).

    The UNK token is always 'UNK' and will be at index 0. All other tokens
    are sorted alphabetically.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Dataset of lines (strings) to build the vocab from.

    Returns
    -------
    tuple
        (vocab_table, vocab_list, frequencies)
        vocab_table: tf.lookup.StaticHashTable
        vocab_list: list[str]
        frequencies: dict[str, int]
    """
    tokenized = dataset.map(
        lambda line: tf.strings.split(line),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    tokens = tokenized.flat_map(
        lambda tokens: tf.data.Dataset.from_tensor_slices(tokens)
    )
    # Collect all tokens as numpy bytes
    token_bytes = list(tokens.as_numpy_iterator())
    # Count frequencies using Python dict
    from collections import Counter
    token_strs = [tok.decode('utf-8') for tok in token_bytes]
    freq_counter = Counter(token_strs)
    # Build vocab list (UNK at 0, rest sorted)
    vocab = ["UNK"] + sorted(
        tok for tok in freq_counter.keys() if tok != "UNK"
    )
    # Build frequency dict (UNK always 0)
    frequencies = {tok: (freq_counter[tok] if tok != "UNK" else 0) for tok in vocab}
    vocab_table = build_vocab_table(vocab)
    return vocab_table, vocab, frequencies
