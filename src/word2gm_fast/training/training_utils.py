# training_utils.py

import tensorflow as tf

def log_training_metrics(model, grads, step, summary_writer):
    """
    Log diagnostic metrics to TensorBoard.
    """
    with summary_writer.as_default():
        # === Gradient norm ===
        grad_norm = tf.linalg.global_norm(grads)
        tf.summary.scalar("gradient_norm", grad_norm, step=step)

        # === Parameter norms ===
        tf.summary.scalar("mu_norm_avg", tf.reduce_mean(tf.norm(model.mus, axis=-1)), step=step)
        tf.summary.scalar("sigma_norm_avg", tf.reduce_mean(tf.exp(model.logsigmas)), step=step)

        # === Mixture weight entropy ===
        mixture = tf.nn.softmax(model.mixture, axis=-1)
        entropy = -tf.reduce_sum(mixture * tf.math.log(mixture + 1e-9), axis=-1)
        entropy_mean = tf.reduce_mean(entropy)
        tf.summary.scalar("mixture_entropy_avg", entropy_mean, step=step)

def train_step(model, optimizer, word_idxs, pos_idxs, neg_idxs,
               normclip=False, norm_cap=5.0,
               lower_sig=0.05, upper_sig=1.0,
               wout=False):
    # Extract variables once to avoid retrace if model signature changes
    variables = model.trainable_variables

    with tf.GradientTape() as tape:
        loss = model((word_idxs, pos_idxs, neg_idxs), training=True)

    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))

    # Norm clipping logic (kept conditional, but safe if flags are constant)
    if normclip:
        clipped_mu = tf.clip_by_norm(model.mus, clip_norm=norm_cap, axes=[-1])
        model.mus.assign(clipped_mu)

        log_min = tf.math.log(lower_sig)
        log_max = tf.math.log(upper_sig)
        clipped_sigma = tf.clip_by_value(model.logsigmas, log_min, log_max)
        model.logsigmas.assign(clipped_sigma)

        if wout:
            clipped_mu_out = tf.clip_by_norm(model.mus_out, clip_norm=norm_cap, axes=[-1])
            clipped_sigma_out = tf.clip_by_value(model.logsigmas_out, log_min, log_max)
            model.mus_out.assign(clipped_mu_out)
            model.logsigmas_out.assign(clipped_sigma_out)

    return loss, grads

def summarize_dataset_pipeline(ds, logger=None):
    """
    Recursively print or log the transformation stack of a tf.data.Dataset.
    """
    def unwrap(ds):
        while hasattr(ds, '_variant_tracker'):
            ds = ds._variant_tracker._dataset
        return ds

    def describe(ds, indent=0):
        name = type(ds).__name__
        line = "  " * indent + f"🔹 {name}"
        if logger:
            logger.info(line)
        else:
            print(line)
        if hasattr(ds, '_input_dataset'):
            describe(ds._input_dataset, indent + 1)
        elif hasattr(ds, '_input_datasets'):
            for sub in ds._input_datasets:
                describe(sub, indent + 1)

    if logger:
        logger.info("🔍 Dataset pipeline structure:")
    else:
        print("🔍 Dataset pipeline structure:")

    ds_unwrapped = unwrap(ds)
    describe(ds_unwrapped)
