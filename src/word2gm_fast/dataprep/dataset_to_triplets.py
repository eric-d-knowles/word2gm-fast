"""
Convert a dataset of 5-gram lines into (center, positive, negative) skip-gram
triplets for training. All logic is implemented with TensorFlow ops for tracing
safety and speed.
"""

import tensorflow as tf


def build_skipgram_triplets(
    dataset: tf.data.Dataset,
    vocab_table: tf.lookup.StaticHashTable,
    frequencies: tf.Tensor = None,
    downsample_threshold: float = 1e-5,
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

    # If frequencies are provided, create a lookup table for index->frequency
    if frequencies is not None:
        # frequencies: shape [vocab_size], dtype float32 or int64
        freq_tensor = tf.convert_to_tensor(frequencies, dtype=tf.float32)
        total_count = tf.reduce_sum(freq_tensor)
        # Compute word probabilities
        word_probs = freq_tensor / total_count
        # Compute downsampling probabilities (per Mikolov et al.)
        # P_keep = min(1, sqrt(threshold / prob) + threshold / prob)
        # For rare words, P_keep ~ 1; for frequent, P_keep < 1
        keep_probs = tf.minimum(1.0, tf.sqrt(downsample_threshold / word_probs) + downsample_threshold / word_probs)
    else:
        keep_probs = None

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

        # Frequency-based downsampling of context words
        if keep_probs is not None:
            # Get keep probabilities for each context word
            context_keep_probs = tf.gather(keep_probs, valid_context_ids)
            # Sample random uniform values for each context
            random_vals = tf.random.uniform(tf.shape(context_keep_probs), minval=0.0, maxval=1.0, dtype=tf.float32)
            # Keep only those where random < keep_prob
            downsample_mask = random_vals < context_keep_probs
            valid_context_ids = tf.boolean_mask(valid_context_ids, downsample_mask)

        # Skip lines where center is UNK or no valid context exists
        skip_condition = tf.logical_or(
            tf.equal(center_id, 0),  # Center is UNK
            tf.equal(tf.shape(valid_context_ids)[0], 0)  # No valid context (after downsampling)
        )

        def create_triplets():
            # Generate negative samples for each positive context
            num_contexts = tf.shape(valid_context_ids)[0]
            # Cast vocab_size to int32 to match minval dtype
            negatives = tf.random.uniform(
                [num_contexts], minval=1, maxval=tf.cast(vocab_size, tf.int32), dtype=tf.int32,
                seed=42  # Add seed for deterministic mode
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
