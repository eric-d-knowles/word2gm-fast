"""
Triplet Dataset Utilities

Provides utility functions for working with word2gm triplet datasets.
Functions for analyzing, validating, and summarizing triplet datasets.
"""

import time
import tensorflow as tf
from typing import Set, Optional, Tuple
from IPython.display import display, Markdown


def count_unique_triplet_tokens(
    triplets_ds: tf.data.Dataset,
    progress_interval: int = 100_000,
    show_progress: bool = True,
    batch_size: int = 1000
) -> Tuple[int, Set[int]]:
    """
    Count the number of unique token indices in a triplets dataset.
    
    Iterates through all triplets and returns the count of unique token indices
    found across target, context, and negative samples.
    
    Parameters
    ----------
    triplets_ds : tf.data.Dataset
        Dataset of triplets in the format (target, context, negative)
    progress_interval : int, default=100_000
        Show progress after this many triplets are processed
    show_progress : bool, default=True
        Whether to display progress information
    batch_size : int, default=1000
        Batch size for dataset iteration. Larger values may be faster
        but consume more memory.
        
    Returns
    -------
    tuple
        (count, indices_set) where:
        - count is the number of unique token indices
        - indices_set is the set of all unique token indices
        
    Examples
    --------
    >>> from word2gm_fast.io.artifacts import load_pipeline_artifacts
    >>> from word2gm_fast.utils.triplet_utils import count_unique_triplet_tokens
    >>> 
    >>> # Load artifacts
    >>> artifacts = load_pipeline_artifacts('/path/to/artifacts')
    >>> triplets_ds = artifacts['triplets_ds']
    >>> 
    >>> # Count unique tokens
    >>> unique_count, unique_indices = count_unique_triplet_tokens(triplets_ds)
    >>> print(f"Found {unique_count:,} unique token indices in triplets")
    >>> 
    >>> # Check if a specific token index appears in triplets
    >>> token_idx = 1234
    >>> if token_idx in unique_indices:
    >>>     print(f"Token index {token_idx} appears in triplets")
    >>> else:
    >>>     print(f"Token index {token_idx} does NOT appear in triplets")
    """
    start_time = time.perf_counter()
    
    # Prepare batched dataset for efficient processing
    batched_ds = triplets_ds.batch(batch_size)
    
    # Track unique token indices
    unique_indices = set()
    triplet_count = 0
    
    if show_progress:
        display(Markdown(f"<pre>Counting unique token indices in triplets dataset...</pre>"))
    
    # Process dataset in batches
    for batch in batched_ds:
        # Unpack batch
        if isinstance(batch, tuple) and len(batch) == 3:
            # Format: (target, context, negative)
            target, context, negative = batch
        else:
            # Format: batch tensor of shape (batch_size, 3)
            raise ValueError("Expected triplet dataset in format (target, context, negative)")
        
        # Add indices to set (automatically handles uniqueness)
        unique_indices.update(target.numpy())
        unique_indices.update(context.numpy())
        unique_indices.update(negative.numpy())
        
        # Track triplet count for progress reporting
        current_batch_size = len(target)
        triplet_count += current_batch_size
        
        # Show progress at intervals
        if show_progress and triplet_count % progress_interval < current_batch_size:
            elapsed = time.perf_counter() - start_time
            current_unique = len(unique_indices)
            display(Markdown(
                f"<pre>Processed {triplet_count:,} triplets in {elapsed:.1f}s - "
                f"Found {current_unique:,} unique token indices</pre>"
            ))
    
    # Final stats
    duration = time.perf_counter() - start_time
    unique_count = len(unique_indices)
    
    if show_progress:
        tokens_per_second = triplet_count / duration
        display(Markdown(
            f"<pre>✓ Completed analysis of {triplet_count:,} triplets in {duration:.1f}s "
            f"({tokens_per_second:.0f} triplets/s)\n"
            f"✓ Found {unique_count:,} unique token indices</pre>"
        ))
    
    return unique_count, unique_indices
