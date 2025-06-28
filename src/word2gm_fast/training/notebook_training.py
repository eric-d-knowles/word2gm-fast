"""
Notebook-Oriented Training Entrypoint for Word2GM

High-level training orchestration for Word2GM, including resource monitoring, logging, and model saving. Intended for use in Jupyter/VS Code notebooks.
"""

import os
import time
import tensorflow as tf
from ..models.word2gm_model import Word2GMModel
from ..utils.resource_monitor import ResourceMonitor
from .train_loop import train_one_epoch, build_optimizer
from .training_utils import summarize_dataset_pipeline

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
    args.var_scale = getattr(args, 'var_scale', 0.05)
    args.loss_epsilon = getattr(args, 'loss_epsilon', 1e-8)
    args.max_pe = getattr(args, 'max_pe', False)


    summarize_dataset_pipeline(args.training_dataset)

    from IPython.display import display, Markdown
    display(Markdown("<pre>\nStarting Word2GM training</pre>"))
    start_time = time.time()

    os.makedirs(args.tensorboard_log_path, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(str(args.tensorboard_log_path))
    display(Markdown(f"<pre>Writing TensorBoard logs to {args.tensorboard_log_path}</pre>"))

    model = Word2GMModel(args)
    optimizer = build_optimizer(args)

    dummy_input = next(iter(args.training_dataset))
    _ = model((dummy_input[0], dummy_input[1], dummy_input[2]), training=True)
    _ = optimizer.apply_gradients(
        zip(
            [tf.zeros_like(v) for v in model.trainable_variables],
            model.trainable_variables
        )
    )

    resource_monitor = ResourceMonitor(
        interval=monitor_interval,
        summary_writer=summary_writer
    )
    resource_monitor.print_to_notebook = True
    resource_monitor.start()

    try:
        from IPython.display import display, Markdown
        for epoch in range(args.epochs_to_train):
            epoch_start = time.time()

            epoch_loss = train_one_epoch(
                model, optimizer, args.training_dataset,
                summary_writer=summary_writer, epoch=epoch
            )

            epoch_time = time.time() - epoch_start
            # Print a single summary line per epoch
            display(Markdown(f"<pre>Epoch {epoch + 1}/{args.epochs_to_train} | Loss: {epoch_loss.numpy():.5f} | Time: {epoch_time:.2f} sec</pre>"))
            os.makedirs(args.save_path, exist_ok=True)
            model.save_weights(
                os.path.join(args.save_path, f"model_weights_epoch{epoch+1}.weights.h5")
            )
            # Log resource usage at the end of each epoch for accurate max tracking
            resource_monitor.log_resource_usage()

        total_time = time.time() - start_time
        display(Markdown(f"<pre>Total training time: {total_time:.2f} seconds</pre>"))

        with summary_writer.as_default():
            tf.summary.scalar("total_training_time_seconds", total_time, step=0)

        summary_writer.flush()
    finally:
        resource_monitor.stop()
