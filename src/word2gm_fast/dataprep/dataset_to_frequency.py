"""
Convert a TensorFlow TextLineDataset to a frequency table for all unique tokens.

This module processes TF datasets to extract token frequencies, which can be used
for negative sampling and frequency-based downsampling in skip-gram training.
"""

import tensorflow as tf
from typing import Dict
from ..utils.tf_silence import import_tf_quietly

# Import TensorFlow silently
tf = import_tf_quietly(force_cpu=True)


def dataset_to_frequency(dataset: tf.data.Dataset) -> Dict[str, int]:
    """
    Convert a TextLineDataset to a frequency table for all unique tokens.
        
    Parameters
    ----------
    dataset : tf.data.Dataset
        A TextLineDataset containing 5-gram lines.
        
    Returns
    -------
    Dict[str, int]
        Dictionary mapping tokens to their frequencies, sorted by frequency (descending).
    """
    # Split each line into tokens
    def split_line(line):
        tokens = tf.strings.split(tf.strings.strip(line))
        return tokens
    
    # Collect all tokens and count frequencies
    all_tokens = []
    
    for tokens_tensor in dataset.map(split_line):
        tokens = tokens_tensor.numpy()
        all_tokens.extend([token.decode('utf-8') for token in tokens])
    
    # Count frequencies
    from collections import Counter
    token_counts = Counter(all_tokens)
    
    # Sort by frequency (descending) and convert to dictionary
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    frequency_table = dict(sorted_tokens)
    
    return frequency_table
