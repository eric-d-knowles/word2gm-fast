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


from ..models.word2gm_model import Word2GMModel
from ..models.config import Word2GMConfig
from ..io.triplets import load_triplets_from_tfrecord
from ..io.artifacts import load_pipeline_artifacts
from .training_utils import train_step


# Import the new ResourceMonitor
from ..utils.resource_monitor import ResourceMonitor


# === TensorFlow config ===
tf.config.optimizer.set_jit(True)  # Enable XLA


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
    # Initialize total_loss with the model's compute dtype or float32
    total_loss = tf.constant(0.0, dtype=getattr(model, 'compute_dtype', tf.float32))
    num_batches = 0
    nonzero_batches = 0

    for step, (word_idxs, pos_idxs, neg_idxs) in enumerate(dataset):
        #
        with tf.profiler.experimental.Trace("train", step_num=step, _r=1):
            loss, grads = train_step(
                model, optimizer, word_idxs, pos_idxs, neg_idxs,
                normclip=model.config.normclip,
                norm_cap=model.config.norm_cap,
                lower_sig=model.config.lower_sig,
                upper_sig=model.config.upper_sig,
                wout=model.config.wout
            )

        if total_loss is None:
            total_loss = tf.cast(0.0, dtype=loss.dtype)
        total_loss += tf.cast(loss, dtype=total_loss.dtype)
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

    avg_loss = total_loss / tf.cast(tf.maximum(1, num_batches), total_loss.dtype)

    if summary_writer:
        with summary_writer.as_default():
            tf.summary.scalar("epoch_loss", avg_loss, step=epoch)
            tf.summary.scalar("epoch_nonzero_batches", tf.cast(nonzero_batches, tf.float32), step=epoch)

    return avg_loss
