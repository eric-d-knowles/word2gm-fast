"""
Convert a dataset of 5-gram lines into (center, positive, negative) skip-gram
triplets for training. Works with string tokens and uses frequency table for
downsampling.
"""

import tensorflow as tf
from typing import Dict


def dataset_to_triplets(
    dataset: tf.data.Dataset,
    frequency_table: Dict[str, int],
    downsample_threshold: float = 1e-5,
) -> tf.data.Dataset:
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

    Returns
    -------
    tf.data.Dataset
        Dataset of (center, positive, negative) string triplets.
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

    return triplet_ds
