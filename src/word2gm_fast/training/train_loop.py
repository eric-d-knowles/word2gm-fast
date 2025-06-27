"""
Word2GM Training Loop

Notebook-friendly training loop for Word2GM embedding model. No CLI functionality;
all configuration is via function arguments. Provides functions for epoch-wise
training, optimizer setup, and TensorBoard logging.
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision


from ..models.word2gm_model import Word2GMModel
from ..models.config import Word2GMConfig
from ..dataprep.tfrecord_io import (
    read_triplets_from_tfrecord,
    read_vocab_from_tfrecord
)
    train_step, 
    log_training_metrics,
    summarize_dataset_pipeline
)

# Import the new ResourceMonitor
from .resource_monitor import ResourceMonitor


# === TensorFlow config ===
tf.config.optimizer.set_jit(True)  # Enable XLA
mixed_precision.set_global_policy('mixed_float16')


def build_optimizer(args):
    """
    Build and return a Keras optimizer based on arguments.

    Parameters
    ----------
    args : object
        Object with optimizer settings (must have .learning_rate, .adagrad, etc).

    Returns
    -------
    tf.keras.optimizers.Optimizer
        The configured optimizer instance.
    """
    return (
        tf.keras.optimizers.Adagrad(learning_rate=args.learning_rate)
        if args.adagrad else
        tf.keras.optimizers.SGD(
            learning_rate=args.learning_rate,
            momentum=0.9,
            nesterov=True
        )
    )


@tf.function
def train_one_epoch(model, optimizer, dataset, summary_writer=None, epoch=0):
    """
    Run one epoch of training for the Word2GM model.

    Parameters
    ----------
    model : tf.keras.Model
        The Word2GM model instance.
    optimizer : tf.keras.optimizers.Optimizer
        The optimizer for training.
    dataset : tf.data.Dataset
        The dataset yielding (word_idxs, pos_idxs, neg_idxs) batches.
    summary_writer : tf.summary.SummaryWriter, optional
        TensorBoard summary writer for logging metrics.
    epoch : int, optional
        The current epoch number (for logging).

    Returns
    -------
    tf.Tensor
        The average loss for the epoch.
    """
    total_loss = 0.0
    num_batches = 0
    nonzero_batches = 0

    for step, (word_idxs, pos_idxs, neg_idxs) in enumerate(dataset):
        with tf.profiler.experimental.Trace("train", step_num=step, _r=1):
            loss, grads = train_step(
                model, optimizer, word_idxs, pos_idxs, neg_idxs,
                normclip=model.config.normclip,
                norm_cap=model.config.norm_cap,
                lower_sig=model.config.lower_sig,
                upper_sig=model.config.upper_sig,
                wout=model.config.wout
            )

        total_loss += loss
        num_batches += 1
        nonzero_batches += tf.cast(loss > 0, tf.int32)

        global_step = epoch * 1_000 + step  # Fallback if summary step is unset
        current_step = tf.summary.experimental.get_step()
        if current_step is None:
            current_step = global_step

        if summary_writer and step % 100 == 0:
            with summary_writer.as_default():
                tf.summary.scalar("batch_loss", loss, step=current_step)
                tf.summary.scalar(
                    "learning_rate",
                    optimizer.learning_rate,
                    step=current_step
                )
                tf.summary.scalar(
                    "nonzero_batches",
                    tf.cast(nonzero_batches, tf.float32),
                    step=current_step
                )

                if step % 500 == 0:
                    log_training_metrics(
                        model, grads, step=current_step,
                        summary_writer=summary_writer
                    )
                    tf.summary.histogram(
                        "center_mean_norms",
                        tf.norm(model.mus, axis=-1),
                        step=current_step
                    )
                    tf.summary.histogram(
                        "covariances",
                        tf.exp(tf.reshape(model.logsigmas, [-1])),
                        step=current_step
                    )
                    tf.summary.histogram(
                        "mixture_weights",
                        tf.reshape(model.mixture, [-1]),
                        step=current_step
                    )

    avg_loss = total_loss / tf.cast(tf.maximum(1, num_batches), tf.float32)

    if summary_writer:
        with summary_writer.as_default():
            tf.summary.scalar("epoch_loss", avg_loss, step=epoch)
            tf.summary.scalar("epoch_nonzero_batches", tf.cast(nonzero_batches, tf.float32), step=epoch)

    return avg_loss


def run_notebook_training(
    training_dataset,
    save_path,
    vocab_size,
    embedding_size=50,
    num_mixtures=2,
    spherical=True,
    learning_rate=0.05,
    epochs=10,
    adagrad=True,
    normclip=True,
    norm_cap=5.0,
    lower_sig=0.05,
    upper_sig=1.0,
    wout=False,
    tensorboard_log_path=None,
    monitor_interval=10,
    profile=False
):
    """
    Run the full training loop for the Word2GM model in a notebook environment.

    Parameters
    ----------
    training_dataset : tf.data.Dataset
        The training dataset yielding (word_idxs, pos_idxs, neg_idxs) batches.
    save_path : str
        Directory to save trained model.
    vocab_size : int
        Vocabulary size.
    embedding_size : int, optional
        Embedding dimension (default: 50).
    num_mixtures : int, optional
        Number of mixture components (default: 2).
    spherical : bool, optional
        Use spherical covariances (default: True).
    learning_rate : float, optional
        Learning rate (default: 0.05).
    epochs : int, optional
        Number of epochs (default: 10).
    adagrad : bool, optional
        Use Adagrad optimizer (default: True).
    normclip : bool, optional
        Enable norm clipping (default: True).
    norm_cap : float, optional
        Norm cap for means (default: 5.0).
    lower_sig : float, optional
        Lower bound for log-sigma clipping (default: 0.05).
    upper_sig : float, optional
        Upper bound for log-sigma clipping (default: 1.0).
    wout : bool, optional
        Whether to also clip output means/sigmas (default: False).
    tensorboard_log_path : str, optional
        Directory for TensorBoard logs.
    monitor_interval : int, optional
        Resource monitor interval (seconds, default: 10).
    benchmark_steps : int, optional
        Number of steps for benchmarking (default: 0).
    profile : bool, optional
        Enable TensorFlow profiler (default: False).

    Returns
    -------
    None
    """
    class Args:
        pass
    args = Args()
    args.training_dataset = training_dataset
    args.save_path = save_path
    args.vocab_size = vocab_size
    args.embedding_size = embedding_size
    args.num_mixtures = num_mixtures
    args.spherical = spherical
    args.learning_rate = learning_rate
    args.epochs_to_train = epochs
    args.adagrad = adagrad
    args.normclip = normclip
    args.norm_cap = norm_cap
    args.lower_sig = lower_sig
    args.upper_sig = upper_sig
    args.wout = wout
    args.tensorboard_log_path = tensorboard_log_path or os.path.join(save_path, "tensorboard")
    args.monitor_interval = monitor_interval
    args.profile = profile

    # === Dataset structure ===
    summarize_dataset_pipeline(args.training_dataset)

    print("\n🚀 Starting Word2GM training")
    start_time = time.time()

    # === Create TensorBoard writer ===
    os.makedirs(args.tensorboard_log_path, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(str(args.tensorboard_log_path))
    print(f"📝 Writing TensorBoard logs to {args.tensorboard_log_path}")

    # === Initialize model and optimizer ===
    model = Word2GMModel(args, args.vocab_size)
    optimizer = build_optimizer(args)

    # === Build model once with dummy input ===
    dummy_input = next(iter(args.training_dataset))
    _ = model((dummy_input[0], dummy_input[1], dummy_input[2]), training=True)
    _ = optimizer.apply_gradients(
        zip(
            [tf.zeros_like(v) for v in model.trainable_variables],
            model.trainable_variables
        )
    )

    # === Start resource monitoring ===
    resource_monitor = ResourceMonitor(
        interval=monitor_interval,
        tensorboard_writer=summary_writer,
        print_to_notebook=True
    )
    resource_monitor.start()

    try:
        # === Training loop ===
        for epoch in range(args.epochs_to_train):
            print(f"\n📘 Epoch {epoch + 1}/{args.epochs_to_train}")
            epoch_start = time.time()

            epoch_loss = train_one_epoch(
                model, optimizer, args.training_dataset,
                summary_writer=summary_writer, epoch=epoch
            )

            epoch_time = time.time() - epoch_start
            print(f"📉 Loss at epoch {epoch + 1}: {epoch_loss.numpy():.5f}")
            print(f"🕒 Epoch time: {epoch_time:.2f} sec")
            # Save model after each epoch
            os.makedirs(args.save_path, exist_ok=True)
            model.save_weights(
                os.path.join(args.save_path, f"model_weights_epoch{epoch+1}.weights.h5")
            )

        # === Report and log total time ===
        total_time = time.time() - start_time
        print(f"⏱️ Total training time: {total_time:.2f} seconds")

        with summary_writer.as_default():
            tf.summary.scalar("total_training_time_seconds", total_time, step=0)

        summary_writer.flush()
    finally:
        # === Stop resource monitoring ===
        resource_monitor.stop()
        resource_monitor.join()
    # ...existing code...