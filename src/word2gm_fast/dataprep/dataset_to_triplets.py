"""
Convert a dataset of 5-gram lines into (center, positive, negative) skip-gram
triplets for training. All logic is implemented with TensorFlow ops for tracing
safety and speed.
"""

import tensorflow as tf


def build_skipgram_triplets(
    dataset: tf.data.Dataset,
    vocab_table: tf.lookup.StaticHashTable,
) -> tf.data.Dataset:
    """
    Convert lines of text into (center, positive, negative) skip-gram triplets.

    Generates multiple triplets per valid 5-gram line (one for each valid
    context word). Only generates valid triplets (center is not UNK).
    UNK is always at index 0.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Dataset of 5-gram text lines.
    vocab_table : tf.lookup.StaticHashTable
        Lookup table mapping tokens to vocab indices.

    Returns
    -------
    tf.data.Dataset
        Dataset of (center, positive, negative) training triplets.
    """
    # Pre-compute constants outside the hot path
    vocab_size = vocab_table.size()
    context_indices = tf.constant([0, 1, 3, 4], dtype=tf.int32)

    def generate_all_triplets(line: tf.Tensor):
        """
        Generate all valid skip-gram triplets from a 5-gram line.
        Returns a dataset of triplets for this line.

        Parameters
        ----------
        line : tf.Tensor
            A single 5-gram line as a TensorFlow string tensor.

        Returns
        -------
        tf.data.Dataset
            Dataset of (center, positive, negative) triplets for this line.
        """
        tokens = tf.strings.split(tf.strings.strip(line))
        token_ids = vocab_table.lookup(tokens)

        center_id = token_ids[2]
        context_ids = tf.gather(token_ids, context_indices)

        # Filter out UNK context words and skip if center is UNK
        valid_mask = tf.not_equal(context_ids, 0)
        valid_context_ids = tf.boolean_mask(context_ids, valid_mask)

        # Skip lines where center is UNK or no valid context exists
        skip_condition = tf.logical_or(
            tf.equal(center_id, 0),  # Center is UNK
            tf.equal(tf.shape(valid_context_ids)[0], 0)  # No valid context
        )

        def create_triplets():
            # Generate negative samples for each positive context
            num_contexts = tf.shape(valid_context_ids)[0]
            negatives = tf.random.uniform(
                [num_contexts], minval=1, maxval=vocab_size, dtype=tf.int32
            )
            
            # Broadcast center to match number of contexts
            centers = tf.fill([num_contexts], center_id)
            
            # Create triplets: (center, positive_context, negative)
            triplets = tf.stack([
                tf.cast(centers, tf.int64),
                tf.cast(valid_context_ids, tf.int64),
                tf.cast(negatives, tf.int64)
            ], axis=1)
            
            return tf.data.Dataset.from_tensor_slices(triplets)

        def empty_dataset():
            # Return empty dataset with correct shape
            empty_triplets = tf.zeros([0, 3], dtype=tf.int64)
            return tf.data.Dataset.from_tensor_slices(empty_triplets)

        return tf.cond(
            skip_condition,
            true_fn=empty_dataset,
            false_fn=create_triplets
        )

    # Use flat_map to generate multiple triplets per line
    triplet_ds = dataset.flat_map(generate_all_triplets)

    return triplet_ds
