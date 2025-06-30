# training_utils.py

import tensorflow as tf

def log_training_metrics(model, grads, step, summary_writer, notebook_display=None):
    """
    Log diagnostic training metrics to TensorBoard and optionally update notebook display.

    Parameters
    ----------
    model : tf.keras.Model
        The Word2GM model instance.
    grads : list of tf.Tensor
        Gradients computed for model parameters.
    step : int
        The current global step for logging.
    summary_writer : tf.summary.SummaryWriter
        TensorBoard summary writer.
    notebook_display : dict or None, optional
        If provided, should be a dict with keys 'display_handle' and 'display_fn' (e.g., IPython.display.display/Markdown),
        to update notebook cell output with current metrics.
    """
    grad_norm = tf.linalg.global_norm(grads)
    mu_norm_avg = tf.reduce_mean(tf.norm(model.mus, axis=-1))
    sigma_norm_avg = tf.reduce_mean(tf.exp(model.logsigmas))
    mixture = tf.nn.softmax(model.mixture, axis=-1)
    entropy = -tf.reduce_sum(mixture * tf.math.log(mixture + 1e-9), axis=-1)
    entropy_mean = tf.reduce_mean(entropy)
    with summary_writer.as_default():
        tf.summary.scalar("gradient_norm", grad_norm, step=step)
        tf.summary.scalar("mu_norm_avg", mu_norm_avg, step=step)
        tf.summary.scalar("sigma_norm_avg", sigma_norm_avg, step=step)
        tf.summary.scalar("mixture_entropy_avg", entropy_mean, step=step)

def train_step(
    model, optimizer, word_idxs, pos_idxs, neg_idxs,
    normclip=False, norm_cap=5.0,
    lower_sig=0.05, upper_sig=1.0,
    wout=False
):
    """
    Perform a single training step for the Word2GM model, with optional norm and
    sigma clipping.

    Parameters
    ----------
    model : tf.keras.Model
        The Word2GM model instance.
    optimizer : tf.keras.optimizers.Optimizer
        The optimizer for training.
    word_idxs, pos_idxs, neg_idxs : tf.Tensor
        Input tensors for anchor, positive, and negative word indices.
    normclip : bool, optional
        Whether to apply norm clipping to means and sigmas (default: False).
    norm_cap : float, optional
        Maximum norm for means if norm clipping is enabled (default: 5.0).
    lower_sig : float, optional
        Lower bound for log-sigma clipping (default: 0.05).
    upper_sig : float, optional
        Upper bound for log-sigma clipping (default: 1.0).
    wout : bool, optional
        Whether to also clip output means/sigmas (if present, default: False).

    Returns
    -------
    loss : tf.Tensor
        The computed loss for the batch.
    grads : list of tf.Tensor
        Gradients for model parameters.
    """
    variables = model.trainable_variables
    #
    with tf.GradientTape() as tape:
        loss = model((word_idxs, pos_idxs, neg_idxs), training=True)
    grads = tape.gradient(loss, variables)
    # Filter out None gradients
    grads_and_vars = [(g, v) for g, v in zip(grads, variables) if g is not None]
    if grads_and_vars:
        optimizer.apply_gradients(grads_and_vars)
    if normclip:
        clipped_mu = tf.clip_by_norm(model.mus, clip_norm=norm_cap, axes=[-1])
        model.mus.assign(clipped_mu)
        log_min = tf.math.log(lower_sig)
        log_max = tf.math.log(upper_sig)
        clipped_sigma = tf.clip_by_value(model.logsigmas, log_min, log_max)
        model.logsigmas.assign(clipped_sigma)
        if wout:
            clipped_mu_out = tf.clip_by_norm(
                model.mus_out, clip_norm=norm_cap, axes=[-1]
            )
            clipped_sigma_out = tf.clip_by_value(
                model.logsigmas_out, log_min, log_max
            )
            model.mus_out.assign(clipped_mu_out)
            model.logsigmas_out.assign(clipped_sigma_out)
    return loss, grads

def summarize_dataset_pipeline(ds, logger=None):
    """
    Recursively print or log the transformation stack of a tf.data.Dataset.

    Parameters
    ----------
    ds : tf.data.Dataset
        The dataset to summarize.
    logger : logging.Logger, optional
        Logger to use for output (if None, prints to stdout).
    """
    def unwrap(ds):
        while hasattr(ds, '_variant_tracker'):
            ds = ds._variant_tracker._dataset
        return ds

    def collect_pipeline(ds, stack=None):
        if stack is None:
            stack = []
        name = type(ds).__name__
        stack.append(name)
        # Recursively collect input datasets
        if hasattr(ds, '_input_dataset'):
            collect_pipeline(ds._input_dataset, stack)
        elif hasattr(ds, '_input_datasets'):
            for sub in ds._input_datasets:
                collect_pipeline(sub, stack)
        return stack

    ds_unwrapped = unwrap(ds)
    pipeline = collect_pipeline(ds_unwrapped, stack=[])
    # The stack is from last to first, so reverse for chronological order
    pipeline = pipeline[::-1]
    if logger:
        logger.info("Dataset pipeline structure (oldest to newest):")
        for i, name in enumerate(pipeline):
            logger.info(f"  [{i}] {name}")
    # If no logger is provided, do nothing (no print for production)
