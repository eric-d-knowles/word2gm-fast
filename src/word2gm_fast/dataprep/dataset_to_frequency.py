"""
Convert a TensorFlow TextLineDataset to a frequency table for all unique tokens.

This module processes TF datasets to extract token frequencies, which can be used
for negative sampling and frequency-based downsampling in skip-gram training.
"""

import tensorflow as tf
from typing import Dict


def dataset_to_frequency(dataset: tf.data.Dataset) -> Dict[str, int]:
    """
    Convert a TextLineDataset to a frequency table for all unique tokens.
    
    Uses TensorFlow ops for optimized performance and pipeline integration.
    
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
    print("Converting dataset to frequency table...")
    
    all_tokens = []
    line_count = 0
    
    for tokens_tensor in dataset.map(split_line):
        tokens = tokens_tensor.numpy()
        all_tokens.extend([token.decode('utf-8') for token in tokens])
        line_count += 1
        
        if line_count % 1_000_000 == 0:
            print(f"  Processed {line_count:,} lines, collected {len(all_tokens):,} tokens")
    
    print(f"Collected {len(all_tokens):,} total tokens from {line_count:,} lines")
    
    # Count frequencies
    from collections import Counter
    token_counts = Counter(all_tokens)
    
    print(f"Found {len(token_counts):,} unique tokens")
    
    # Sort by frequency (descending)
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Convert to dictionary
    frequency_table = dict(sorted_tokens)
    
    print(f"Frequency table complete: {len(frequency_table):,} tokens")
    if frequency_table:
        most_frequent = sorted_tokens[0]
        least_frequent = sorted_tokens[-1]
        print(f"  Most frequent: '{most_frequent[0]}' ({most_frequent[1]:,} times)")
        print(f"  Least frequent: '{least_frequent[0]}' ({least_frequent[1]:,} times)")
    
    return frequency_table
