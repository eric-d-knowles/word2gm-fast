"""
Convert a tf.data.Dataset of lines into a tf.lookup.StaticHashTable containing
the corpus vocabulary. Also provides functions to convert string triplets to 
integer triplets using vocabulary mapping.
"""

import tensorflow as tf
from typing import Tuple, Optional, Dict
from ..utils.tf_silence import import_tf_quietly
from IPython import display

# Import TensorFlow silently
tf = import_tf_quietly(force_cpu=True)


def preview_integer_triplets(
    dataset: tf.data.Dataset, n: int, buffer_size: int = 1000
) -> None:
    """
    Print a preview of n random integer triplets from a dataset.

    Parameters
    ----------
    dataset : tf.data.Dataset
        The integer triplets dataset to preview.
    n : int
        Number of triplets to preview.
    buffer_size : int, optional
        Buffer size for shuffling (default is 1000).
    """
    triplet_lines = []
    for triplet in dataset.shuffle(buffer_size).take(n):
        center, positive, negative = triplet.numpy()
        triplet_lines.append(f"   ({center}, {positive}, {negative})")
    
    display.display_markdown(
        "<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>\nPreview of {n} random integer triplets:<br><br>{lines}<br></span>".format(
            n=n,
            lines="<br>".join(["&nbsp;&nbsp;&nbsp;" + line for line in triplet_lines])
        ),
        raw=True
    )


def print_vocab_summary(
    vocab_list: list, vocab_table: tf.lookup.StaticHashTable, 
    frequency_table: dict = None, title: str = "Vocabulary Summary",
    triplet_token_counts: dict = None
) -> dict:
    """
    Print a summary of the vocabulary and integer mapping.

    Parameters
    ----------
    vocab_list : list
        List of vocabulary tokens.
    vocab_table : tf.lookup.StaticHashTable
        The vocabulary lookup table.
    frequency_table : dict, optional
        Token frequencies if available.
    title : str, optional
        Title for the summary display.
    triplet_token_counts : dict, optional
        Token counts from actual triplets (takes precedence over frequency_table).

    Returns
    -------
    summary : dict
        Dictionary containing vocabulary statistics.
    """
    vocab_size = len(vocab_list)
    
    # Basic vocabulary statistics
    summary = {
        "vocab_size": vocab_size,
        "unk_token": vocab_list[0] if vocab_list else "None",
        "sample_tokens": vocab_list[1:6] if len(vocab_list) > 5 else vocab_list[1:],
        "lowest_index": 0,
        "highest_index": vocab_size - 1 if vocab_size > 0 else 0
    }
    
    # Add frequency statistics if available (for internal use, not displayed)
    token_counts_source = triplet_token_counts if triplet_token_counts else frequency_table
    if token_counts_source:
        # Calculate total tokens based on vocabulary that actually appears in triplets
        vocab_set = set(vocab_list)
        total_tokens = sum(freq for token, freq in token_counts_source.items() if token in vocab_set)
        most_frequent = sorted(
            [(token, freq) for token, freq in token_counts_source.items() if token in vocab_set],
            key=lambda x: x[1], reverse=True
        )[:5]
        summary["total_tokens"] = total_tokens
        summary["most_frequent"] = most_frequent
    
    # Format the summary
    lines = [
        f"- Vocabulary size: {summary['vocab_size']:,}",
        f"- Index range: {summary['lowest_index']} to {summary['highest_index']:,}",
        f"- UNK token: {summary['unk_token']} (index {summary['lowest_index']})",
        f"- Sample tokens: {', '.join(summary['sample_tokens'])}"
    ]
    
    display.display_markdown(
        "<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>\n{title}:<br><br>{lines}<br></span>".format(
            title=title,
            lines="<br>".join(lines)
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
    
    # Check for various transformation types
    if "MapDataset" in dataset_str or "map(" in dataset_str.lower():
        transformations.append("Mapped")
    if "FilterDataset" in dataset_str or "filter(" in dataset_str.lower():
        transformations.append("Filtered")
    if "CacheDataset" in dataset_str or "cache(" in dataset_str.lower():
        transformations.append("Cached")
    if "BatchDataset" in dataset_str or "batch(" in dataset_str.lower():
        transformations.append("Batched")
    if "ShuffleDataset" in dataset_str or "shuffle(" in dataset_str.lower():
        transformations.append("Shuffled")
    if "RepeatDataset" in dataset_str or "repeat(" in dataset_str.lower():
        transformations.append("Repeated")
    if "PrefetchDataset" in dataset_str or "prefetch(" in dataset_str.lower():
        transformations.append("Prefetched")
    if "TensorSliceDataset" in dataset_str or "from_tensor_slices" in dataset_str.lower():
        transformations.append("TensorSlices")
    
    # For integer triplets datasets specifically, we know they are created from tensor slices
    # and may be cached, so ensure we show the underlying transformation
    if len(transformations) == 1 and "Cached" in transformations:
        # This is likely a cached dataset created from tensor slices
        transformations = ["TensorSlices", "Cached"]
    
    if transformations:
        properties["Transformations"] = ", ".join(transformations)
    
    # Display using the same format as other functions
    display.display_markdown(
        "<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>\n{title}:<br><br>{lines}<br></span>".format(
            title=title,
            lines="<br>".join([f"- {k}: {v}" for k, v in properties.items()]),
        ),
        raw=True
    )


def build_vocab_table(vocab_list: list[str]) -> tf.lookup.StaticHashTable:
    """
    Build a TensorFlow StaticHashTable mapping words to integer IDs.

    The vocabulary list must be ordered by ID (i.e., index = ID), with
    the UNK token ('UNK') at index 0. All other tokens should be sorted
    alphabetically or as desired by the pipeline.

    Parameters
    ----------
    vocab_list : list[str]
        List of vocabulary tokens, with 'UNK' at index 0.

    Returns
    -------
    tf.lookup.StaticHashTable
        A lookup table mapping tokens to integer IDs.
    """
    if vocab_list[0] != "UNK":
        raise ValueError(
            f"UNK token must be at index 0, got {vocab_list[0]}"
        )
    keys = tf.constant(vocab_list, dtype=tf.string)
    values = tf.range(len(vocab_list), dtype=tf.int32)
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=0
    )


def make_vocab(dataset: tf.data.Dataset):
    """
    Build a vocab hash table and frequency dict from a tf.data.Dataset of lines using only
    TensorFlow ops (scalable for large corpora).

    The UNK token is always 'UNK' and will be at index 0. All other tokens
    are sorted alphabetically.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Dataset of lines (strings) to build the vocab from.

    Returns
    -------
    tuple
        (vocab_table, vocab_list, frequencies)
        vocab_table: tf.lookup.StaticHashTable
        vocab_list: list[str]
        frequencies: dict[str, int]
    """
    tokenized = dataset.map(
        lambda line: tf.strings.split(line),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    tokens = tokenized.flat_map(
        lambda tokens: tf.data.Dataset.from_tensor_slices(tokens)
    )
    # Collect all tokens as numpy bytes
    token_bytes = list(tokens.as_numpy_iterator())
    # Count frequencies using Python dict
    from collections import Counter
    token_strs = [tok.decode('utf-8') for tok in token_bytes]
    freq_counter = Counter(token_strs)
    # Build vocab list (UNK at 0, rest sorted)
    vocab = ["UNK"] + sorted(
        tok for tok in freq_counter.keys() if tok != "UNK"
    )
    # Build frequency dict (UNK always 0)
    frequencies = {tok: (freq_counter[tok] if tok != "UNK" else 0) for tok in vocab}
    vocab_table = build_vocab_table(vocab)
    return vocab_table, vocab, frequencies


def triplets_to_integers(
    triplets_dataset: tf.data.Dataset,
    frequency_table: dict = None,
    preview_n: int = 0,
    show_summary: bool = False,
    show_properties: bool = False,
    cache: bool = False
) -> Tuple[tf.data.Dataset, tf.lookup.StaticHashTable, list, int, Optional[dict]]:
    """
    Convert string triplets to integer triplets and create vocabulary mapping.
    
    Collects all triplets into memory, builds vocabulary from collected tokens, 
    then converts to integers.
    
    Parameters
    ----------
    triplets_dataset : tf.data.Dataset
        Dataset of (center, positive, negative) string triplets.
    frequency_table : dict, optional
        Pre-computed frequency table to preserve frequency information.
        If None, frequencies will be computed from triplets.
    preview_n : int, optional
        Number of integer triplets to preview (default is 0).
    show_summary : bool, optional
        Whether to display vocabulary summary (default is False).
    show_properties : bool, optional
        Whether to display dataset properties (default is False).
    cache : bool, optional
        Whether to cache the resulting dataset (default is False).
        
    Returns
    -------
    tuple
        (integer_triplets_dataset, vocab_table, vocab_list, vocab_size, summary)
        integer_triplets_dataset: tf.data.Dataset of (center_id, pos_id, neg_id)
        vocab_table: tf.lookup.StaticHashTable for string->int mapping
        vocab_list: list[str] of vocabulary tokens (UNK at index 0)
        vocab_size: int total vocabulary size
        summary: dict or None, vocabulary summary if show_summary is True
    """
    # Collect ALL triplets into memory and extract unique tokens
    all_triplets = []
    unique_tokens = set()
    triplet_token_counts = {}  # Count tokens from actual triplets
    
    for triplet in triplets_dataset:
        center, positive, negative = triplet.numpy()
        
        # Decode bytes to strings if needed
        center_str = center.decode('utf-8') if isinstance(center, bytes) else center
        positive_str = positive.decode('utf-8') if isinstance(positive, bytes) else positive  
        negative_str = negative.decode('utf-8') if isinstance(negative, bytes) else negative
        
        # Store triplet and collect unique tokens
        triplet_tuple = (center_str, positive_str, negative_str)
        all_triplets.append(triplet_tuple)
        unique_tokens.update(triplet_tuple)
        
        # Count token occurrences in triplets
        for token in triplet_tuple:
            triplet_token_counts[token] = triplet_token_counts.get(token, 0) + 1
    
    # Build vocabulary from collected tokens
    vocab_tokens = sorted(unique_tokens - {"UNK"})  # Remove UNK if present
    
    # Sort by frequency if frequency_table is provided, otherwise alphabetically
    if frequency_table:
        vocab_tokens.sort(key=lambda x: frequency_table.get(x, 0), reverse=True)
    
    vocab_list = ["UNK"] + vocab_tokens
    vocab_size = len(vocab_list)
    
    # Create token-to-ID mapping for fast conversion
    token_to_id = {token: idx for idx, token in enumerate(vocab_list)}
    
    # Convert all triplets to integers using in-memory mapping
    integer_triplets = []
    
    for center_str, positive_str, negative_str in all_triplets:
        center_id = token_to_id.get(center_str, 0)  # 0 = UNK
        positive_id = token_to_id.get(positive_str, 0)
        negative_id = token_to_id.get(negative_str, 0)
        
        integer_triplets.append([center_id, positive_id, negative_id])
    
    # Convert to TensorFlow dataset
    integer_dataset = tf.data.Dataset.from_tensor_slices(integer_triplets)
    
    # Optional caching
    if cache:
        integer_dataset = integer_dataset.cache()
    
    # Create lookup table for compatibility with existing code
    vocab_keys = tf.constant(vocab_list, dtype=tf.string)
    vocab_values = tf.constant(list(range(vocab_size)), dtype=tf.int32)
    vocab_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(vocab_keys, vocab_values),
        default_value=0
    )
    
    # Optional preview and summary
    if preview_n > 0:
        preview_integer_triplets(integer_dataset, preview_n)
    
    if show_properties:
        print_dataset_properties(integer_dataset, "Integer Triplets Dataset Properties")
    
    if show_summary:
        summary = print_vocab_summary(
            vocab_list, vocab_table, frequency_table, "Vocabulary Summary", 
            triplet_token_counts
        )
        return integer_dataset, vocab_table, vocab_list, vocab_size, summary
    
    return integer_dataset, vocab_table, vocab_list, vocab_size, None
