# dataprep/corpus_to_dataset.py

"""
Convert a dataset of 5-gram lines into (center, positive, negative) skip-gram triplets
for training. All logic is implemented with TensorFlow ops for tracing safety and speed.
"""

import tensorflow as tf


def build_skipgram_triplets(
    dataset: tf.data.Dataset,
    vocab_table: tf.lookup.StaticHashTable,
    vocab_size: int,
    unk_index: int = 0,
) -> tf.data.Dataset:
    """
    Convert lines of text into (center, positive, negative) skip-gram triplets.

    Filters out triplets where the center word is UNK or where no valid context exists.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Dataset of 5-gram text lines.
    vocab_table : tf.lookup.StaticHashTable
        Lookup table mapping tokens to vocab indices.
    vocab_size : int
        Total vocabulary size (used for uniform negative sampling).
    unk_index : int, optional
        Index assigned to the 'UNK' token (default is 0).

    Returns
    -------
    tf.data.Dataset
        Dataset of (center, positive, negative) training triplets.
    """

    def generate_triplet(line: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Generate a single skip-gram triplet from a 5-gram line.

        Parameters
        ----------
        line : tf.Tensor
            A single 5-gram line as a TensorFlow string tensor.

        Returns
        -------
        tuple
            (center, positive, negative) token indices, or (-1, -1, -1) if invalid.
        """
        tokens = tf.strings.split(tf.strings.strip(line))
        token_ids = vocab_table.lookup(tokens)

        center_id = token_ids[2]
        context_indices = tf.constant([0, 1, 3, 4], dtype=tf.int32)
        context_ids = tf.gather(token_ids, context_indices)

        valid_mask = tf.not_equal(context_ids, unk_index)
        valid_context_ids = tf.boolean_mask(context_ids, valid_mask)

        def skip_line():
            return (
                tf.constant(-1, dtype=tf.int64),
                tf.constant(-1, dtype=tf.int64),
                tf.constant(-1, dtype=tf.int64),
            )

        def select_positive():
            rand_index = tf.random.uniform([], minval=0, maxval=tf.shape(valid_context_ids)[0], dtype=tf.int32)
            pos_id = tf.cast(valid_context_ids[rand_index], tf.int64)
            neg_id = tf.random.uniform([], minval=1, maxval=vocab_size, dtype=tf.int64)
            return tf.cast(center_id, tf.int64), pos_id, neg_id

        return tf.cond(
            tf.shape(valid_context_ids)[0] > 0,
            true_fn=select_positive,
            false_fn=skip_line,
        )

    def is_valid_triplet(center: tf.Tensor, pos: tf.Tensor, neg: tf.Tensor) -> tf.Tensor:
        """
        Check if a triplet is valid (center is not -1 or UNK).

        Parameters
        ----------
        center : tf.Tensor
        pos : tf.Tensor
        neg : tf.Tensor

        Returns
        -------
        tf.Tensor
            Boolean tensor indicating validity.
        """
        return tf.logical_and(
            tf.not_equal(center, -1),
            tf.not_equal(center, unk_index)
        )

    triplet_ds = dataset.map(generate_triplet, num_parallel_calls=tf.data.AUTOTUNE)
    triplet_ds = triplet_ds.filter(is_valid_triplet)

    return triplet_ds