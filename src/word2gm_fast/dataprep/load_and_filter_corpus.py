# training/load_and_filter_corpus.py

import tensorflow as tf
import time

"""
Load a 5-gram corpus, filter out malformed lines, and prepare a dataset for skip-gram training.
Fast, tracing-safe implementation using TensorFlow ops only.
"""

UNK_TOKEN = "UNK"
EXPECTED_NGRAM_LENGTH = 5

def map_and_mask_line(line: tf.Tensor):
    """
    Returns: (line, is_valid_bool) using tracing-safe TF ops.
    """
    tokens = tf.strings.split(tf.strings.strip(line))

    def check_valid():
        center_valid = tf.not_equal(tokens[2], UNK_TOKEN)
        context = tf.stack([tokens[0], tokens[1], tokens[3], tokens[4]])
        context_valid = tf.reduce_any(tf.not_equal(context, UNK_TOKEN))
        return tf.logical_and(center_valid, context_valid)

    is_len_5 = tf.equal(tf.size(tokens), EXPECTED_NGRAM_LENGTH)
    is_valid = tf.cond(is_len_5, check_valid, lambda: tf.constant(False))

    return line, is_valid

def load_and_filter_corpus(filepath: str, preview_n: int = 0, max_lines: int | None = None, cache: bool = False, show_summary: bool = False):
    """
    Load and filter a 5-gram corpus with high performance and tracing safety.

    Parameters
    ----------
    filepath : str
        Path to corpus text file.
    preview_n : int
        Number of retained lines to preview.
    max_lines : int or None
        Optional limit on number of lines to read.
    cache : bool
        Whether to cache the resulting dataset.
    show_summary : bool
        Whether to compute and print summary counts with timing.

    Returns
    -------
    filtered_dataset : tf.data.Dataset
        A dataset of valid 5-gram lines.
    summary : dict or None
        Dictionary of retained, rejected, and total line counts (if requested).
    """
    with tf.device("/CPU:0"):
        prep_start = time.time()

        dataset = tf.data.TextLineDataset(filepath)
        if max_lines:
            dataset = dataset.take(max_lines)

        dataset = dataset.map(map_and_mask_line, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.filter(lambda line, valid: valid)
        dataset = dataset.map(lambda line, _: line, num_parallel_calls=tf.data.AUTOTUNE)

        if cache:
            dataset = dataset.cache()

        prep_end = time.time()

        if preview_n > 0:
            print(f"\n✅ Previewing {preview_n} retained 5-grams:")
            for line in dataset.take(preview_n):
                print("  ", line.numpy().decode("utf-8"))

        summary = None
        if show_summary:
            count_start = time.time()

            retained_count = dataset.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1).numpy()
            raw_dataset = tf.data.TextLineDataset(filepath)
            if max_lines:
                raw_dataset = raw_dataset.take(max_lines)
            total_count = raw_dataset.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1).numpy()
            rejected_count = total_count - retained_count

            count_end = time.time()

            print("\n⏱️ Benchmark:")
            print(f"  processing_time_sec: {round(prep_end - prep_start, 3)}")
            print(f"  counting_time_sec: {round(count_end - count_start, 3)}")
            print(f"  total_time_sec: {round(count_end - prep_start, 3)}")

            summary = {
                "retained": retained_count,
                "rejected": rejected_count,
                "total": total_count
            }

            print("\n📊 Summary:")
            for k, v in summary.items():
                print(f"  {k}: {v}")

        else:
            print("\n⏱️ Benchmark:")
            print(f"  processing_time_sec: {round(prep_end - prep_start, 3)}")

        return dataset, summary
