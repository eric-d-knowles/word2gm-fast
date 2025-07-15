"""
Load a 5-gram corpus, filter out malformed lines, and prepare a TensorFlow dataset
for skip-gram training.
"""

import tensorflow as tf
from typing import Optional, Tuple


def validate_5gram_line(line: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Check if a 5-gram line is valid according to center/context rules.

    Parameters
    ----------
    line : tf.Tensor
        A single line from the corpus as a TensorFlow string tensor.

    Returns
    -------
    tuple
        (line, is_valid_bool):
            line (tf.Tensor): The original input line.
            is_valid_bool (tf.Tensor): Boolean tensor indicating if the line is
            a valid 5-gram.
    """
    tokens = tf.strings.split(tf.strings.strip(line))
    center_valid = tf.not_equal(tokens[2], "UNK")
    context = tf.stack([tokens[0], tokens[1], tokens[3], tokens[4]])
    context_valid = tf.reduce_any(tf.not_equal(context, "UNK"))
    is_valid = tf.logical_and(center_valid, context_valid)
    return line, is_valid


def preview_dataset(
    dataset: tf.data.Dataset, n: int, buffer_size: int = 1000
) -> None:
    """
    Print a preview of n random lines from a dataset.

    Parameters
    ----------
    dataset : tf.data.Dataset
        The dataset to preview.
    n : int
        Number of lines to preview.
    buffer_size : int, optional
        Buffer size for shuffling (default is 1000).
    """
    print(f"\nPreviewing {n} random retained 5-grams:")
    for line in dataset.shuffle(buffer_size).take(n):
        print("  ", line.numpy().decode("utf-8"))


def print_dataset_summary(
    dataset: tf.data.Dataset, filepath: str
) -> dict:
    """
    Print a summary of retained, rejected, and total line counts for a dataset.

    Parameters
    ----------
    dataset : tf.data.Dataset
        The filtered dataset.
    filepath : str
        Path to the original corpus text file.

    Returns
    -------
    summary : dict
        Dictionary of retained, rejected, and total line counts.
    """
    retained_count = dataset.reduce(
        tf.constant(0, tf.int64), lambda x, _: x + 1
    ).numpy()
    raw_dataset = tf.data.TextLineDataset(filepath)
    total_count = raw_dataset.reduce(
        tf.constant(0, tf.int64), lambda x, _: x + 1
    ).numpy()
    rejected_count = total_count - retained_count
    summary = {
        "retained": retained_count,
        "rejected": rejected_count,
        "total": total_count
    }
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


def make_dataset(
    filepath: str,
    preview_n: int = 0,
    cache: bool = False,
    show_summary: bool = False
) -> Tuple[tf.data.Dataset, Optional[dict]]:
    """
    Load and filter a 5-gram corpus with high performance and tracing safety.

    Parameters
    ----------
    filepath : str
        Path to the corpus text file.
    preview_n : int, optional
        Number of retained lines to preview (default is 0).
    cache : bool, optional
        Whether to cache the resulting dataset (default is False).
    show_summary : bool, optional
        Whether to compute and print summary counts (default is False).

    Returns
    -------
    filtered_dataset : tf.data.Dataset
        A dataset of valid 5-gram lines.
    summary : dict or None
        Dictionary of retained, rejected, and total line counts if show_summary
        is True, otherwise None.
    """
    dataset = tf.data.TextLineDataset(filepath)
    dataset = dataset.map(
        validate_5gram_line, num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.filter(lambda line, valid: valid)
    dataset = dataset.map(
        lambda line, _: line, num_parallel_calls=tf.data.AUTOTUNE
    )
    if cache:
        dataset = dataset.cache()
    if preview_n > 0:
        preview_dataset(dataset, preview_n)
    if show_summary:
        summary = print_dataset_summary(dataset, filepath)
        return dataset, summary
    return dataset, None