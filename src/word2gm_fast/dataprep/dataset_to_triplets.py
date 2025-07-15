"""
Convert a dataset of 5-gram lines into (center, positive, negative) skip-gram
triplets for training. Works with string tokens and uses frequency table for
downsampling.
"""

import tensorflow as tf
from typing import Dict, Optional, Tuple
from ..utils.tf_silence import import_tf_quietly
from IPython import display

# Import TensorFlow silently
tf = import_tf_quietly(force_cpu=True)


def preview_triplets(
    dataset: tf.data.Dataset, n: int, buffer_size: int = 1000
) -> None:
    """
    Print a preview of n random triplets from a dataset.

    Parameters
    ----------
    dataset : tf.data.Dataset
        The triplets dataset to preview.
    n : int
        Number of triplets to preview.
    buffer_size : int, optional
        Buffer size for shuffling (default is 1000).
    """
    triplet_lines = []
    for triplet in dataset.shuffle(buffer_size).take(n):
        center, positive, negative = triplet.numpy()
        center_str = center.decode("utf-8")
        positive_str = positive.decode("utf-8")
        negative_str = negative.decode("utf-8")
        triplet_lines.append(f"   ({center_str}, {positive_str}, {negative_str})")
    
    display.display_markdown(
        "<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>\nPreview of {n} random triplets:<br><br>{lines}<br></span>".format(
            n=n,
            lines="<br>".join(["&nbsp;&nbsp;&nbsp;" + line for line in triplet_lines])
        ),
        raw=True
    )


def print_triplets_summary(
    dataset: tf.data.Dataset, title: str = "Triplets Summary"
) -> dict:
    """
    Print a summary of the triplets dataset.

    Parameters
    ----------
    dataset : tf.data.Dataset
        The triplets dataset to summarize.
    title : str, optional
        Title for the summary display.

    Returns
    -------
    summary : dict
        Dictionary containing dataset statistics.
    """
    # Count total triplets and collect unique values
    total_triplets = 0
    unique_centers = set()
    unique_positives = set()
    unique_negatives = set()
    all_unique_words = set()
    
    # Iterate through entire dataset to get exact counts
    for triplet in dataset:
        total_triplets += 1
        center, positive, negative = triplet.numpy()
        center_str = center.decode("utf-8")
        positive_str = positive.decode("utf-8")
        negative_str = negative.decode("utf-8")
        
        unique_centers.add(center_str)
        unique_positives.add(positive_str)
        unique_negatives.add(negative_str)
        
        # Add all words to combined vocabulary
        all_unique_words.add(center_str)
        all_unique_words.add(positive_str)
        all_unique_words.add(negative_str)
    
    summary = {
        "total_triplets": total_triplets,
        "unique_centers": len(unique_centers),
        "unique_positives": len(unique_positives),
        "unique_negatives": len(unique_negatives),
        "total_unique_words": len(all_unique_words)
    }
    
    # Format the summary
    lines = [
        f"- Total triplets: {summary['total_triplets']:,}",
        f"- Unique centers: {summary['unique_centers']:,}",
        f"- Unique positives: {summary['unique_positives']:,}",
        f"- Unique negatives: {summary['unique_negatives']:,}",
        f"- Total unique words: {summary['total_unique_words']:,}"
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
    if "FlatMapDataset" in dataset_str or "flat_map(" in dataset_str.lower():
        transformations.append("FlatMapped")
    
    # For triplets datasets specifically, we know they involve mapping and flat_map
    # operations even if not visible due to caching
    if len(transformations) == 1 and "Cached" in transformations:
        # This is likely a cached dataset that had other transformations
        transformations = ["Mapped", "FlatMapped", "Cached"]
    
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


def dataset_to_triplets(
    dataset: tf.data.Dataset,
    frequency_table: Dict[str, int],
    downsample_threshold: float = 1e-5,
    preview_n: int = 0,
    show_summary: bool = False,
    show_properties: bool = False,
    cache: bool = False
) -> Tuple[tf.data.Dataset, Optional[dict]]:
    """
    Convert 5-gram lines to (center, positive, negative) string triplets.

    Generates multiple triplets per valid 5-gram line (one for each valid
    context word). Input dataset should already be filtered to exclude
    lines with UNK center tokens (handled by corpus_to_dataset).
    Uses frequency-based downsampling and uniform negative sampling.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Dataset of 5-gram text lines (pre-filtered, no UNK centers).
    frequency_table : Dict[str, int]
        Token frequencies for downsampling decisions.
    downsample_threshold : float, default=1e-5
        Threshold for frequency-based downsampling.
    preview_n : int, optional
        Number of triplets to preview (default is 0).
    show_summary : bool, optional
        Whether to compute and print summary statistics (default is False).
    show_properties : bool, optional
        Whether to display dataset properties (default is False).
    cache : bool, optional
        Whether to cache the resulting triplets dataset (default is False).

    Returns
    -------
    triplet_dataset : tf.data.Dataset
        Dataset of (center, positive, negative) string triplets.
    summary : dict or None
        Dictionary of triplet statistics if show_summary is True, otherwise None.
    """
    # Convert frequency table to TensorFlow lookup table for graph compatibility
    total_count = sum(frequency_table.values())
    
    # Create tensors for vocabulary and keep probabilities (excluding UNK)
    vocab_tokens = [token for token in frequency_table.keys() if token != "UNK"]
    keep_probs_values = []
    
    for token in vocab_tokens:
        freq = frequency_table[token]
        word_prob = freq / total_count
        # Compute downsampling probability (per Mikolov et al.)
        keep_prob = min(1.0, (downsample_threshold / word_prob) ** 0.5 + downsample_threshold / word_prob)
        keep_probs_values.append(keep_prob)
    
    # Create TensorFlow lookup table for keep probabilities (UNK not included)
    vocab_tensor = tf.constant(vocab_tokens, dtype=tf.string)
    keep_probs_tensor = tf.constant(keep_probs_values, dtype=tf.float32)
    
    # Create lookup table: token -> keep_probability
    keep_prob_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(vocab_tensor, keep_probs_tensor),
        default_value=1.0  # Default keep probability for unknown tokens
    )
    
    # Get all non-UNK tokens for uniform negative sampling
    all_tokens = tf.constant(vocab_tokens, dtype=tf.string)
    vocab_size = len(vocab_tokens)

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
            Dataset of (center, positive, negative) string triplets for this line.
        """
        # Split line into tokens
        tokens = tf.strings.split(tf.strings.strip(line))
        
        # Extract center and context tokens
        center_token = tokens[2]
        context_tokens = tf.stack([tokens[0], tokens[1], tokens[3], tokens[4]])
        
        # Filter out UNK context words
        valid_mask = tf.not_equal(context_tokens, "UNK")
        valid_context = tf.boolean_mask(context_tokens, valid_mask)
        
        # Apply frequency-based downsampling to context words (graph-compatible)
        def apply_downsampling():
            num_tokens = tf.shape(valid_context)[0]
            
            # Handle empty case
            def empty_case():
                return tf.zeros([0], dtype=tf.string)
            
            def downsample_case():
                # Look up keep probabilities for each token
                token_keep_probs = keep_prob_table.lookup(valid_context)
                
                # Generate random values for downsampling
                random_vals = tf.random.uniform([num_tokens], minval=0.0, maxval=1.0, dtype=tf.float32)
                
                # Apply downsampling: keep token if random < keep_prob
                keep_mask = random_vals < token_keep_probs
                return tf.boolean_mask(valid_context, keep_mask)
            
            return tf.cond(
                tf.equal(num_tokens, 0),
                true_fn=empty_case,
                false_fn=downsample_case
            )
        
        final_context = apply_downsampling()
        
        # Check if we should skip this line (no valid context after downsampling)
        has_no_context = tf.equal(tf.shape(final_context)[0], 0)
        
        def create_triplets():
            # Generate negative samples for each positive context
            num_contexts = tf.shape(final_context)[0]
            
            # Broadcast center to match number of contexts
            centers = tf.fill([num_contexts], center_token)
            
            # Generate uniform random negative samples
            negative_indices = tf.random.uniform(
                [num_contexts], minval=0, maxval=vocab_size, dtype=tf.int32
            )
            negatives = tf.gather(all_tokens, negative_indices)
            
            # Create triplets: (center, positive_context, negative)
            triplets = tf.stack([centers, final_context, negatives], axis=1)
            
            return tf.data.Dataset.from_tensor_slices(triplets)

        def empty_dataset():
            # Return empty dataset with correct shape
            empty_triplets = tf.zeros([0, 3], dtype=tf.string)
            return tf.data.Dataset.from_tensor_slices(empty_triplets)

        return tf.cond(
            has_no_context,
            true_fn=empty_dataset,
            false_fn=create_triplets
        )

    # Use flat_map to generate multiple triplets per line
    triplet_ds = dataset.flat_map(generate_all_triplets)

    # Optional caching
    if cache:
        triplet_ds = triplet_ds.cache()

    # Optional preview and summary
    if preview_n > 0:
        preview_triplets(triplet_ds, preview_n)
    
    if show_properties:
        print_dataset_properties(triplet_ds, "Triplets Dataset Properties")
    
    if show_summary:
        summary = print_triplets_summary(triplet_ds, "Generated Triplets Summary")
        return triplet_ds, summary
    
    return triplet_ds, None
