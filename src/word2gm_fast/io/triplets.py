"""
Triplets TFRecord I/O utilities for word2GM skip-gram training data.

Provides functions to save and load skip-gram triplets datasets as TFRecord files.
"""

import tensorflow as tf
import time
from typing import List, Optional, Union
from IPython.display import display, Markdown


def write_triplets_to_tfrecord(
    dataset: tf.data.Dataset,
    output_path: str,
    compress: bool = False
) -> int:
    """
    Stream skip-gram triplets from a tf.data.Dataset into a TFRecord file.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Dataset of (center, positive, negative) triplets as int64 tensors.
    output_path : str
        Path for the output TFRecord file.
    compress : bool, optional
        Whether to compress the file with GZIP. Default is False.

    Returns
    -------
    int
        Number of triplets written to the TFRecord file.
    """
    if compress and not output_path.endswith(".gz"):
        output_path += ".gz"
    options = tf.io.TFRecordOptions(compression_type="GZIP") if compress else None
    display(Markdown(f"<pre>Writing TFRecord to: {output_path}</pre>"))
    start = time.perf_counter()
    count = 0
    with tf.io.TFRecordWriter(output_path, options=options) as writer:
        # Handle both TensorFlow datasets and plain Python iterables
        if hasattr(dataset, 'as_numpy_iterator'):
            iterator = dataset.as_numpy_iterator()
        else:
            iterator = dataset
            
        for triplet in iterator:
            center, positive, negative = (int(x) for x in triplet)
            example = tf.train.Example(features=tf.train.Features(feature={
                'center': tf.train.Feature(int64_list=tf.train.Int64List(value=[center])),
                'positive': tf.train.Feature(int64_list=tf.train.Int64List(value=[positive])),
                'negative': tf.train.Feature(int64_list=tf.train.Int64List(value=[negative])),
            }))
            writer.write(example.SerializeToString())
            count += 1
    duration = time.perf_counter() - start
    display(Markdown(f"<pre>Write complete. Triplets written: {count:,}</pre>"))
    return count


def parse_triplet_example(example_proto: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Parse a single triplet example from TFRecord.

    Parameters
    ----------
    example_proto : tf.Tensor
        Serialized tf.train.Example containing triplet data.

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        (center, positive, negative) as int64 tensors.
    """
    feature_description = {
        'center': tf.io.FixedLenFeature([], tf.int64),
        'positive': tf.io.FixedLenFeature([], tf.int64),
        'negative': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    return parsed['center'], parsed['positive'], parsed['negative']


def load_triplets_from_tfrecord(
    tfrecord_path: Union[str, List[str]],
    compressed: Optional[bool] = None,
    num_parallel_reads: Optional[int] = tf.data.AUTOTUNE
) -> tf.data.Dataset:
    """
    Load skip-gram triplets from a TFRecord file (optionally gzipped).

    Parameters
    ----------
    tfrecord_path : str or list of str
        Path to the TFRecord file(s).
    compressed : bool, optional
        Whether the file(s) are GZIP compressed. Auto-detected if None.
    num_parallel_reads : int, optional
        Number of parallel readers for TFRecord files. Default is AUTOTUNE.

    Returns
    -------
    tf.data.Dataset
        Parsed dataset of (center, positive, negative) triplets as int64 tensors.
    """
    if compressed is None:
        # Auto-detect only if a single file is passed
        if isinstance(tfrecord_path, str):
            compressed = tfrecord_path.endswith(".gz")
        else:
            compressed = False  # Assume uncompressed when list is passed

    compression_type = "GZIP" if compressed else None

    display(Markdown(f"<pre>Loading triplet TFRecord from: {tfrecord_path}</pre>"))
    start = time.perf_counter()

    raw_ds = tf.data.TFRecordDataset(
        tfrecord_path,
        buffer_size=64 << 20,  # 64MB buffer
        compression_type=compression_type,
        num_parallel_reads=num_parallel_reads
    )

    parsed_ds = raw_ds.map(
        lambda x: parse_triplet_example(x), 
        num_parallel_calls=tf.data.AUTOTUNE
    )

    duration = time.perf_counter() - start
    display(Markdown(f"<pre>Triplet TFRecord loaded and parsed</pre>"))

    return parsed_ds
