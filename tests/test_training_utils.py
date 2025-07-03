# tests/test_training_utils.py
import pytest
import tensorflow as tf
tf.random.set_seed(1)
from src.word2gm_fast.training import training_utils

class DummyModel(tf.keras.Model):
    def __init__(self, vocab_size=10, embed_dim=4, n_components=2):
        super().__init__()
        self.mus = tf.Variable(tf.random.normal([vocab_size, n_components, embed_dim]), name='mus')
        self.logsigmas = tf.Variable(tf.random.normal([vocab_size, n_components, embed_dim]), name='logsigmas')
        self.mixture = tf.Variable(tf.random.normal([vocab_size, n_components]), name='mixture')
        self.mus_out = tf.Variable(tf.random.normal([vocab_size, n_components, embed_dim]), name='mus_out')
        self.logsigmas_out = tf.Variable(tf.random.normal([vocab_size, n_components, embed_dim]), name='logsigmas_out')
    def call(self, inputs, training=False):
        word_idxs, pos_idxs, neg_idxs = inputs
        mus_sum = tf.reduce_sum(tf.gather(self.mus, word_idxs))
        logsigmas_sum = tf.reduce_sum(tf.gather(self.logsigmas, word_idxs))
        mixture_sum = tf.reduce_sum(tf.gather(self.mixture, word_idxs))
        mus_out_sum = tf.reduce_sum(tf.gather(self.mus_out, word_idxs))
        logsigmas_out_sum = tf.reduce_sum(tf.gather(self.logsigmas_out, word_idxs))
        return mus_sum + logsigmas_sum + mixture_sum + mus_out_sum + logsigmas_out_sum

def test_log_training_metrics(tmp_path):
    model = DummyModel()
    grads = [tf.ones_like(var) for var in model.trainable_variables]
    step = 1
    log_dir = tmp_path / "tb"
    writer = tf.summary.create_file_writer(str(log_dir))
    # Should not raise
    training_utils.log_training_metrics(model, grads, step, writer)

def test_train_step_basic():
    model = DummyModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    word_idxs = tf.constant([0, 1, 2])
    pos_idxs = tf.constant([1, 2, 3])
    neg_idxs = tf.constant([2, 3, 4])
    loss, grads = training_utils.train_step(model, optimizer, word_idxs, pos_idxs, neg_idxs)
    assert tf.is_tensor(loss)
    assert isinstance(grads, list)
    assert all(tf.is_tensor(g) for g in grads)

def test_train_step_with_clipping():
    model = DummyModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    word_idxs = tf.constant([0, 1, 2])
    pos_idxs = tf.constant([1, 2, 3])
    neg_idxs = tf.constant([2, 3, 4])
    # Should not raise with normclip and wout
    loss, grads = training_utils.train_step(
        model, optimizer, word_idxs, pos_idxs, neg_idxs,
        normclip=True, norm_cap=1.0, lower_sig=0.01, upper_sig=2.0, wout=True
    )
    assert tf.is_tensor(loss)
    assert isinstance(grads, list)
    assert all(tf.is_tensor(g) for g in grads)

def test_summarize_dataset_pipeline(capsys):
    ds = tf.data.Dataset.from_tensor_slices([1, 2, 3]).map(lambda x: x + 1)
    # Should print pipeline structure
    training_utils.summarize_dataset_pipeline(ds)
    out = capsys.readouterr().out
    assert "Dataset pipeline structure" in out
    assert "MapDataset" in out or "TensorSliceDataset" in out
