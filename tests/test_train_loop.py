# tests/test_train_loop.py
import pytest
import tensorflow as tf
tf.random.set_seed(1)
from src.word2gm_fast.training.train_loop import train_one_epoch
from src.word2gm_fast.training.training_utils import train_step

class DummyModel(tf.keras.Model):
    def __init__(self, vocab_size=5, embed_dim=2, n_components=1):
        super().__init__()
        self.mus = tf.Variable(tf.random.normal([vocab_size, n_components, embed_dim]), name='mus')
        self.logsigmas = tf.Variable(tf.random.normal([vocab_size, n_components, embed_dim]), name='logsigmas')
        self.mixture = tf.Variable(tf.random.normal([vocab_size, n_components]), name='mixture')
        class Config:
            normclip = False
            norm_cap = 1.0
            lower_sig = 0.01
            upper_sig = 2.0
            wout = False
        self.config = Config()
    def call(self, inputs, training=False):
        word_idxs, pos_idxs, neg_idxs = inputs
        mus_sum = tf.reduce_sum(tf.gather(self.mus, word_idxs))
        logsigmas_sum = tf.reduce_sum(tf.gather(self.logsigmas, word_idxs))
        mixture_sum = tf.reduce_sum(tf.gather(self.mixture, word_idxs))
        return mus_sum + logsigmas_sum + mixture_sum

@pytest.fixture
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

def test_train_one_epoch_runs(dummy_dataset):
    model = DummyModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    avg_loss = train_one_epoch(model, optimizer, dummy_dataset)
    assert tf.is_tensor(avg_loss)
    assert avg_loss.shape == ()
    assert avg_loss.numpy() == avg_loss.numpy()  # is a number
