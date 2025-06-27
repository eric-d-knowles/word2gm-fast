# tests/test_notebook_training.py
import pytest
import tensorflow as tf
import tempfile
import shutil
from src.word2gm_fast.training.notebook_training import run_notebook_training

class DummyModelConfig:
    def __init__(self, vocab_size=5, embedding_size=2, num_mixtures=1):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_mixtures = num_mixtures
        self.spherical = True
        self.learning_rate = 0.01
        self.epochs_to_train = 1
        self.adagrad = True
        self.normclip = False
        self.norm_cap = 1.0
        self.lower_sig = 0.01
        self.upper_sig = 2.0
        self.wout = False
        self.batch_size = 2

def dummy_dataset():
    # Yields batches of (word_idxs, pos_idxs, neg_idxs)
    data = [
        (tf.constant([0, 1]), tf.constant([1, 2]), tf.constant([2, 3])),
        (tf.constant([1, 2]), tf.constant([2, 3]), tf.constant([3, 4]))
    ]
    return tf.data.Dataset.from_generator(
        lambda: data,
        output_signature=(
            tf.TensorSpec(shape=(2,), dtype=tf.int32),
            tf.TensorSpec(shape=(2,), dtype=tf.int32),
            tf.TensorSpec(shape=(2,), dtype=tf.int32)
        )
    )

def test_run_notebook_training_runs():
    ds = dummy_dataset()
    tmpdir = tempfile.mkdtemp()
    try:
        run_notebook_training(
            training_dataset=ds,
            save_path=tmpdir,
            vocab_size=5,
            embedding_size=2,
            num_mixtures=1,
            spherical=True,
            learning_rate=0.01,
            epochs=1,
            adagrad=True,
            normclip=False,
            norm_cap=1.0,
            lower_sig=0.01,
            upper_sig=2.0,
            wout=False,
            tensorboard_log_path=None,
            monitor_interval=1,
            profile=False
        )
        # Check that at least one weights file was created
        import os
        files = os.listdir(tmpdir)
        assert any(f.endswith('.weights.h5') for f in files)
    finally:
        shutil.rmtree(tmpdir)
