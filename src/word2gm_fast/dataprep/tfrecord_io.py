# word2gm_fast/training/tfrecord_io.py

import tensorflow as tf
import time
from typing import Optional, Union
from termcolor import colored


def write_triplet_dataset_to_tfrecord(
    dataset: tf.data.Dataset,
    output_path: str,
    compress: bool = False
) -> None:
    """
    Stream skip-gram triplets from a tf.data.Dataset into a TFRecord file.
    """
    if compress and not output_path.endswith(".gz"):
        output_path += ".gz"

    options = tf.io.TFRecordOptions(compression_type="GZIP") if compress else None

    print(colored(f"💾 Writing TFRecord to: {output_path}", "yellow"))
    start = time.perf_counter()

    count = 0
    with tf.io.TFRecordWriter(output_path, options=options) as writer:
        for triplet in dataset.as_numpy_iterator():
            c, p, n = (int(x) for x in triplet)
            example = tf.train.Example(features=tf.train.Features(feature={
                'center': tf.train.Feature(int64_list=tf.train.Int64List(value=[c])),
                'positive': tf.train.Feature(int64_list=tf.train.Int64List(value=[p])),
                'negative': tf.train.Feature(int64_list=tf.train.Int64List(value=[n])),
            }))
            writer.write(example.SerializeToString())
            count += 1

    duration = time.perf_counter() - start
    print(colored(f"✅ Write complete. Triplets written: {count}", "green"))
    print(colored(f"🕒 Write time: {duration:.2f} sec", "blue"))


def parse_triplet_example(example_proto):
    feature_description = {
        'center': tf.io.FixedLenFeature([], tf.int64),
        'positive': tf.io.FixedLenFeature([], tf.int64),
        'negative': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    return parsed['center'], parsed['positive'], parsed['negative']


def load_triplets_from_tfrecord(
    tfrecord_path: Union[str, list[str]],
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
        Parsed dataset of (center, context, negative) triplets.
    """
    if compressed is None:
        # Auto-detect only if a single file is passed
        if isinstance(tfrecord_path, str):
            compressed = tfrecord_path.endswith(".gz")
        else:
            compressed = False  # Assume uncompressed when list is passed

    compression_type = "GZIP" if compressed else None

    print(colored(f"📂 Loading TFRecord from: {tfrecord_path}", "cyan"))
    start = time.perf_counter()

    raw_ds = tf.data.TFRecordDataset(
        tfrecord_path,
        buffer_size=64 << 20,
        compression_type=compression_type,
        num_parallel_reads=num_parallel_reads
    )

    parsed_ds = raw_ds.map(parse_triplet_example, num_parallel_calls=tf.data.AUTOTUNE)

    duration = time.perf_counter() - start
    print(colored(f"✅ TFRecord loaded and parsed", "green"))
    print(colored(f"🕒 Load time (init only): {duration:.2f} sec", "blue"))

    return parsed_ds