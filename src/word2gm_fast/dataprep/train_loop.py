from pathlib import Path
import sys

import os
import time
import numpy as np
import tensorflow as tf
from threading import Event

from models.word2gm_model import Word2GMModel
from config.config import TrainingConfig
from training.tfrecord_io import load_triplets_from_tfrecord
from training.build_vocab import load_vocab
from training.training_utils import (
    train_step, 
    log_training_metrics,
    summarize_dataset_pipeline
)
from utils.logging_utils import get_loggers, color
from utils.benchmark_utils import (
    monitor_resources,
    benchmark_with_resource_log,
    plot_resource_log
)
from utils.path_utils import create_output_paths


# === TensorFlow config ===
tf.config.optimizer.set_jit(True)  # Enable XLA


def build_optimizer(args):
    return (
        tf.keras.optimizers.Adagrad(learning_rate=args.learning_rate)
        if args.adagrad else
        tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=0.9, nesterov=True)
    )


@tf.function
def train_one_epoch(model, optimizer, dataset, summary_writer=None, epoch=0):
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
                tf.summary.scalar("learning_rate", optimizer.learning_rate, step=current_step)
                tf.summary.scalar("nonzero_batches", tf.cast(nonzero_batches, tf.float32), step=current_step)

                if step % 500 == 0:
                    log_training_metrics(model, grads, step=current_step, summary_writer=summary_writer)
                    tf.summary.histogram("center_mean_norms", tf.norm(model.mus, axis=-1), step=current_step)
                    tf.summary.histogram("covariances", tf.exp(tf.reshape(model.logsigmas, [-1])), step=current_step)
                    tf.summary.histogram("mixture_weights", tf.reshape(model.mixture, [-1]), step=current_step)

    avg_loss = total_loss / tf.cast(tf.maximum(1, num_batches), tf.float32)

    if summary_writer:
        with summary_writer.as_default():
            tf.summary.scalar("epoch_loss", avg_loss, step=epoch)
            tf.summary.scalar("epoch_nonzero_batches", tf.cast(nonzero_batches, tf.float32), step=epoch)

    return avg_loss


def run_training(args):
    # === Setup loggers ===
    log, _, log_resource = get_loggers(
        log_path=args.log_file_path,
        resource_log_path=args.resource_log_path
    )

    # === Dataset structure ===
    summarize_dataset_pipeline(args.training_dataset, logger=log)

    # === Log metadata ===
    log.info(color("📋 Run metadata", "magenta"))
    log.info(f"  🗓️ Year: {args.year}")
    log.info(f"  🧠 Embedding size: {args.embedding_size}")
    log.info(f"  🔢 Mixtures: {args.num_mixtures}")
    log.info(f"  🌐 Spherical: {args.spherical}")
    log.info(f"  📈 Learning rate: {args.learning_rate}")
    log.info(f"  ⚙️ Optimizer: {'Adagrad' if args.adagrad else 'SGD'}")
    log.info(f"  🧹 Norm clip: {args.normclip} (cap={args.norm_cap})")
    log.info(f"  🧊 Sigma bounds: [{args.lower_sig}, {args.upper_sig}]")
    log.info(f"  📚 Vocab size: {args.vocab_size}")
    log.info(f"  📦 Save path: {args.save_path}")
    log.info(f"  📊 TensorBoard: {args.tensorboard_log_path}")
    log.info(f"  📁 Training log: {args.log_file_path}")
    log.info(f"  📁 Resource log: {args.resource_log_path}")
    log.info(color("-" * 60, "magenta"))

    log.info(color("🚀 Starting Word2GM training", "green"))
    start_time = time.time()

    # === Create TensorBoard writer ===
    os.makedirs(args.tensorboard_log_path, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(str(args.tensorboard_log_path))
    log.info(color(f"📝 Writing TensorBoard logs to {args.tensorboard_log_path}", "cyan"))

    # === Initialize model and optimizer ===
    model = Word2GMModel(args, args.vocab_size)
    optimizer = build_optimizer(args)

    # === Build model once with dummy input ===
    dummy_input = next(iter(args.training_dataset))
    _ = model((dummy_input[0], dummy_input[1], dummy_input[2]), training=True)
    _ = optimizer.apply_gradients(zip([tf.zeros_like(v) for v in model.trainable_variables], model.trainable_variables))

    # === Start system resource monitor ===
    stop_event = Event()
    monitor_thread = monitor_resources(stop_event, args.monitor_interval, logger=log_resource)

    # === Training loop ===
    for epoch in range(args.epochs_to_train):
        log.info(color(f"📘 Epoch {epoch + 1}/{args.epochs_to_train}", "cyan"))

        epoch_start = time.time()

        if epoch == 1 and args.benchmark_steps > 0:
            benchmark_with_resource_log(
                args.training_dataset, model, optimizer,
                normclip=model.config.normclip,
                norm_cap=model.config.norm_cap,
                lower_sig=model.config.lower_sig,
                upper_sig=model.config.upper_sig,
                wout=model.config.wout,
                resource_log_path=args.resource_log_path,
                steps=args.benchmark_steps
            )
            
        if epoch == 2 and args.profile:
            tf.profiler.experimental.start(str(args.tensorboard_log_path))
            
        epoch_loss = train_one_epoch(
            model, optimizer, args.training_dataset,
            summary_writer=summary_writer, epoch=epoch
        )

        if epoch == 2 and args.profile:
            tf.profiler.experimental.stop()

        epoch_time = time.time() - epoch_start  # ⬅️ Measure here
        log.info(color(f"📉 Loss at epoch {epoch + 1}: {epoch_loss.numpy():.5f}", "yellow"))
        log.info(color(f"🕒 Epoch time: {epoch_time:.2f} sec", "blue"))
        
    # === Save model ===
    os.makedirs(args.save_path, exist_ok=True)
    log.info(color(f"💾 Saving model to: {args.save_path}", "green"))
    model.save_weights(os.path.join(args.save_path, "model_weights.weights.h5"))

    # === Shutdown monitoring ===
    log.info(color("✅ Training complete. Shutting down monitor.", "green"))
    stop_event.set()
    monitor_thread.join()

    # === Report and log total time ===
    total_time = time.time() - start_time
    log.info(color(f"⏱️ Total training time: {total_time:.2f} seconds", "magenta"))

    with summary_writer.as_default():
        tf.summary.scalar("total_training_time_seconds", total_time, step=0)

    # === Plot resource usage ===
    plot_resource_log(args.resource_log_path, save_dir=args.log_dir)

    summary_writer.flush()