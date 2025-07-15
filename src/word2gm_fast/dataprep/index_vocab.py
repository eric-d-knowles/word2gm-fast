"""
Convert a tf.data.Dataset of lines into a tf.lookup.StaticHashTable containing
the corpus vocabulary. Also provides functions to convert string triplets to 
integer triplets using vocabulary mapping.
"""

import tensorflow as tf
from typing import Tuple


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


def triplets_to_integers(
    triplets_dataset: tf.data.Dataset,
    frequency_table: dict = None
) -> Tuple[tf.data.Dataset, tf.lookup.StaticHashTable, list, int]:
    """
    Convert string triplets to integer triplets and create vocabulary mapping.
    
    Collects all triplets into memory, builds vocabulary from collected tokens, 
    then converts to integers.
    
    Parameters
    ----------
    triplets_dataset : tf.data.Dataset
        Dataset of (center, positive, negative) string triplets.
    frequency_table : dict, optional
        Pre-computed frequency table to preserve frequency information.
        If None, frequencies will be computed from triplets.
        
    Returns
    -------
    tuple
        (integer_triplets_dataset, vocab_table, vocab_list, vocab_size)
        integer_triplets_dataset: tf.data.Dataset of (center_id, pos_id, neg_id)
        vocab_table: tf.lookup.StaticHashTable for string->int mapping
        vocab_list: list[str] of vocabulary tokens (UNK at index 0)
        vocab_size: int total vocabulary size
    """
    # Collect ALL triplets into memory and extract unique tokens
    all_triplets = []
    unique_tokens = set()
    
    for triplet in triplets_dataset:
        center, positive, negative = triplet.numpy()
        
        # Decode bytes to strings if needed
        center_str = center.decode('utf-8') if isinstance(center, bytes) else center
        positive_str = positive.decode('utf-8') if isinstance(positive, bytes) else positive  
        negative_str = negative.decode('utf-8') if isinstance(negative, bytes) else negative
        
        # Store triplet and collect unique tokens
        triplet_tuple = (center_str, positive_str, negative_str)
        all_triplets.append(triplet_tuple)
        unique_tokens.update(triplet_tuple)
    
    # Build vocabulary from collected tokens
    vocab_tokens = sorted(unique_tokens - {"UNK"})  # Remove UNK if present
    
    # Sort by frequency if frequency_table is provided, otherwise alphabetically
    if frequency_table:
        vocab_tokens.sort(key=lambda x: frequency_table.get(x, 0), reverse=True)
    
    vocab_list = ["UNK"] + vocab_tokens
    vocab_size = len(vocab_list)
    
    # Create token-to-ID mapping for fast conversion
    token_to_id = {token: idx for idx, token in enumerate(vocab_list)}
    
    # Convert all triplets to integers using in-memory mapping
    integer_triplets = []
    
    for center_str, positive_str, negative_str in all_triplets:
        center_id = token_to_id.get(center_str, 0)  # 0 = UNK
        positive_id = token_to_id.get(positive_str, 0)
        negative_id = token_to_id.get(negative_str, 0)
        
        integer_triplets.append([center_id, positive_id, negative_id])
    
    # Convert to TensorFlow dataset
    integer_dataset = tf.data.Dataset.from_tensor_slices(integer_triplets)
    
    # Create lookup table for compatibility with existing code
    vocab_keys = tf.constant(vocab_list, dtype=tf.string)
    vocab_values = tf.constant(list(range(vocab_size)), dtype=tf.int32)
    vocab_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(vocab_keys, vocab_values),
        default_value=0
    )
    
    return integer_dataset, vocab_table, vocab_list, vocab_size
