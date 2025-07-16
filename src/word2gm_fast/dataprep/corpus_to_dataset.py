"""
Load a 5-gram corpus, filter out invalid lines, and prepare a TensorFlow dataset
for skip-gram training.
"""

from typing import Optional, Tuple
from ..utils.tf_silence import import_tf_quietly
from IPython import display

# Import TensorFlow silently
tf = import_tf_quietly(force_cpu=True)


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
    lines = [
        "   " + line.numpy().decode("utf-8")
        for line in dataset.shuffle(buffer_size).take(n)
    ]
    display.display_markdown(
        "<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>\nPreview of {n} random retained 5-grams:<br><br>{lines}<br></span>".format(
            n=n,
            lines="<br>".join(["&nbsp;&nbsp;&nbsp;" + line for line in lines])
        ),
        raw=True
    )


def print_dataset_summary(
    dataset: tf.data.Dataset, filepath: str
) -> dict:
    """
    Print a summary of retained, rejected, and total line counts for a dataset. For large corpora,
    this is time-consuming.

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
    display.display_markdown(
        "<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>\nSummary:<br><br>{lines}<br></span>".format(
            lines="<br>".join([f"- {k.capitalize()}: {v}" for k, v in summary.items()]),
        ),
        raw=True
    )
    return summary


def print_dataset_properties(dataset: tf.data.Dataset, title: str = "Dataset Properties") -> None:
    """
    Print dataset properties using formatted display.

    Parameters
    ----------
    dataset : tf.data.Dataset
        The dataset to inspect.
    title : str, optional
        Title for the properties display (default is "Dataset Properties").
    """
    properties = {}
    
    # Check common dataset properties
    try:
        properties["Element spec"] = str(dataset.element_spec)
    except Exception:
        properties["Element spec"] = "Unknown"
    
    # Check for cardinality (number of elements)
    try:
        cardinality = dataset.cardinality().numpy()
        if cardinality == tf.data.INFINITE_CARDINALITY:
            properties["Cardinality"] = "Infinite"
        elif cardinality == tf.data.UNKNOWN_CARDINALITY:
            properties["Cardinality"] = "Unknown"
        else:
            properties["Cardinality"] = str(cardinality)
    except Exception:
        properties["Cardinality"] = "Unknown"
    
    # Try to get additional properties from the dataset's options
    try:
        options = dataset.options()
        if hasattr(options, 'deterministic') and options.deterministic is not None:
            properties["Deterministic"] = str(options.deterministic)
        elif hasattr(options, 'experimental_deterministic') and options.experimental_deterministic is not None:
            properties["Deterministic"] = str(options.experimental_deterministic)
        
        if hasattr(options, 'threading') and options.threading is not None:
            threading_opts = options.threading
            threading_details = []
            if hasattr(threading_opts, 'max_intra_op_parallelism') and threading_opts.max_intra_op_parallelism is not None:
                threading_details.append(f"intra_op={threading_opts.max_intra_op_parallelism}")
            if hasattr(threading_opts, 'private_threadpool_size') and threading_opts.private_threadpool_size is not None:
                threading_details.append(f"threadpool={threading_opts.private_threadpool_size}")
            
            if threading_details:
                properties["Threading"] = ", ".join(threading_details)
            else:
                properties["Threading"] = "Default settings"
    except Exception:
        pass
    
    # Check if dataset appears to be cached, mapped, etc. by inspecting string representation
    dataset_str = str(dataset)
    transformations = []
    if "MapDataset" in dataset_str:
        transformations.append("Mapped")
    if "FilterDataset" in dataset_str:
        transformations.append("Filtered")
    if "CacheDataset" in dataset_str:
        transformations.append("Cached")
    if "BatchDataset" in dataset_str:
        transformations.append("Batched")
    if "ShuffleDataset" in dataset_str:
        transformations.append("Shuffled")
    if "RepeatDataset" in dataset_str:
        transformations.append("Repeated")
    if "PrefetchDataset" in dataset_str:
        transformations.append("Prefetched")
    
    if transformations:
        properties["Transformations"] = ", ".join(transformations)
    
    # Display the properties
    display.display_markdown(
        "<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>\n{title}:<br><br>{lines}<br></span>".format(
            title=title,
            lines="<br>".join([f"- {k}: {v}" for k, v in properties.items()]),
        ),
        raw=True
    )


def make_dataset(
    filepath: str,
    preview_n: int = 0,
    cache: bool = False,
    show_summary: bool = False,
    show_properties: bool = False
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
    show_properties : bool, optional
        Whether to display dataset properties (default is False).

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
    if show_properties:
        print_dataset_properties(dataset, "Processed Dataset Properties")
    if show_summary:
        summary = print_dataset_summary(dataset, filepath)
        return dataset, summary
    return dataset, None