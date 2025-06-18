import tensorflow as tf
from training.tfrecord_io import load_triplets_from_tfrecord

def get_training_dataset(
    tfrecord_path,
    batch_size=16384,
    shuffle_buffer_size=50000,
    drop_remainder=True,
    compressed=None,
    cache=True,
    prefetch=True
):
    """
    Load and preprocess a skip-gram triplet TFRecord dataset for GPU training.

    Parameters
    ----------
    tfrecord_path : str or list of str
        Path to the TFRecord file(s).
    batch_size : int
        Number of triplets per training batch.
    shuffle_buffer_size : int
        Buffer size for dataset shuffling.
    drop_remainder : bool
        Whether to drop the last incomplete batch.
    cache : bool
        Whether to cache the dataset in memory.
    prefetch : bool
        Whether to prefetch batches to keep GPU fed.

    Returns
    -------
    tf.data.Dataset
        Preprocessed dataset of (center, positive, negative) triplets.
    """
    dataset = load_triplets_from_tfrecord(tfrecord_path)
    
    if cache:
        dataset = dataset.cache()

    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # === Debug print ===
    print("✅ Dataset pipeline configuration:")
    print(f"• Batch size: {batch_size}")
    print(f"• Shuffle buffer: {shuffle_buffer_size}")
    print(f"• Cache enabled: {cache}")
    print(f"• Prefetch enabled: {prefetch}")
    
    return dataset
