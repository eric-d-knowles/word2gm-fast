import tensorflow as tf

def build_skipgram_triplets(
    dataset: tf.data.Dataset,
    vocab_table: tf.lookup.StaticHashTable,
    vocab_size: int,
    unk_index: int = 0,
) -> tf.data.Dataset:
    """
    Convert lines of text into (center, positive, negative) skipgram triplets.
    
    Filters out triplets where the center word is UNK.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Dataset of 5-gram text lines.
    vocab_table : tf.lookup.StaticHashTable
        Lookup table mapping tokens to vocab indices.
    vocab_size : int
        Total vocabulary size (used for uniform negative sampling).
    unk_index : int
        Index assigned to 'UNK' token. Positive words with this index are skipped.

    Returns
    -------
    tf.data.Dataset
        Dataset of (center, positive, negative) training triplets.
    """

    def generate_triplet(line):
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

    def is_valid_triplet(center, pos, neg):
        return tf.logical_and(
            tf.not_equal(center, -1),
            tf.not_equal(center, unk_index)  # Filter out UNK as center
        )

    triplet_ds = dataset.map(generate_triplet, num_parallel_calls=tf.data.AUTOTUNE)
    triplet_ds = triplet_ds.filter(is_valid_triplet)

    return triplet_ds
