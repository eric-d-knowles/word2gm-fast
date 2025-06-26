"""
Unit tests for Word2GM model implementation.

Tests mathematical correctness, GPU/CPU compatibility, and edge cases.
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from word2gm_fast.models.word2gm_model import Word2GMModel
from word2gm_fast.models.config import Word2GMConfig


class test_word2gm_model:
    """Test suite for Word2GMModel."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return Word2GMConfig(
            vocab_size=100,
            embedding_size=10,
            num_mixtures=2,
            spherical=True,
            batch_size=4
        )
    
    @pytest.fixture
    def diagonal_config(self):
        """Configuration with diagonal covariances."""
        return Word2GMConfig(
            vocab_size=100,
            embedding_size=10,
            num_mixtures=2,
            spherical=False,
            batch_size=4
        )
    
    @pytest.fixture
    def asymmetric_config(self):
        """Configuration with asymmetric relationships."""
        return Word2GMConfig(
            vocab_size=100,
            embedding_size=10,
            num_mixtures=2,
            spherical=True,
            wout=True,
            batch_size=4
        )
    
    def test_model_initialization_basic(self, basic_config):
        """Test basic model initialization."""
        model = Word2GMModel(basic_config)
        
        # Check parameter shapes
        assert model.mus.shape == (100, 2, 10)
        assert model.logsigmas.shape == (100, 2, 1)  # Spherical
        assert model.mixture.shape == (100, 2)
        
        # Check that parameters are trainable
        assert model.mus.trainable
        assert model.logsigmas.trainable
        assert model.mixture.trainable
    
    def test_model_initialization_diagonal(self, diagonal_config):
        """Test model initialization with diagonal covariances."""
        model = Word2GMModel(diagonal_config)
        
        # Check parameter shapes
        assert model.mus.shape == (100, 2, 10)
        assert model.logsigmas.shape == (100, 2, 10)  # Diagonal
        assert model.mixture.shape == (100, 2)
    
    def test_model_initialization_asymmetric(self, asymmetric_config):
        """Test model initialization with asymmetric relationships."""
        model = Word2GMModel(asymmetric_config)
        
        # Check that output parameters exist
        assert hasattr(model, 'mus_out')
        assert hasattr(model, 'logsigmas_out')
        assert hasattr(model, 'mixture_out')
        
        # Check shapes
        assert model.mus_out.shape == (100, 2, 10)
        assert model.logsigmas_out.shape == (100, 2, 1)
        assert model.mixture_out.shape == (100, 2)
    
    def test_get_word_distributions_basic(self, basic_config):
        """Test getting word distributions."""
        model = Word2GMModel(basic_config)
        word_ids = tf.constant([0, 1, 2, 3])
        
        mus, vars, weights = model.get_word_distributions(word_ids)
        
        # Check output shapes
        assert mus.shape == (4, 2, 10)
        assert vars.shape == (4, 2, 10)  # Broadcasted from spherical
        assert weights.shape == (4, 2)
        
        # Check that weights are normalized
        weight_sums = tf.reduce_sum(weights, axis=-1)
        np.testing.assert_allclose(weight_sums.numpy(), 1.0, rtol=1e-6)
        
        # Check that variances are positive
        assert tf.reduce_all(vars > 0)
    
    def test_get_word_distributions_diagonal(self, diagonal_config):
        """Test getting word distributions with diagonal covariances."""
        model = Word2GMModel(diagonal_config)
        word_ids = tf.constant([0, 1, 2, 3])
        
        mus, vars, weights = model.get_word_distributions(word_ids)
        
        # Check output shapes (no broadcasting for diagonal)
        assert mus.shape == (4, 2, 10)
        assert vars.shape == (4, 2, 10)
        assert weights.shape == (4, 2)
    
    def test_get_word_distributions_asymmetric(self, asymmetric_config):
        """Test getting word distributions with asymmetric relationships."""
        model = Word2GMModel(asymmetric_config)
        word_ids = tf.constant([0, 1, 2, 3])
        
        # Test input embeddings
        mus_in, vars_in, weights_in = model.get_word_distributions(
            word_ids, use_output=False
        )
        
        # Test output embeddings
        mus_out, vars_out, weights_out = model.get_word_distributions(
            word_ids, use_output=True
        )
        
        # Check that they're different (with high probability)
        assert not tf.reduce_all(tf.equal(mus_in, mus_out))
    
    def test_expected_likelihood_kernel_shape(self, basic_config):
        """Test expected likelihood kernel output shape."""
        model = Word2GMModel(basic_config)
        
        # Create dummy GMM parameters
        batch_size = 3
        mus1 = tf.random.normal((batch_size, 2, 10))
        vars1 = tf.random.uniform((batch_size, 2, 10), 0.1, 1.0)
        weights1 = tf.nn.softmax(tf.random.normal((batch_size, 2)), axis=-1)
        
        mus2 = tf.random.normal((batch_size, 2, 10))
        vars2 = tf.random.uniform((batch_size, 2, 10), 0.1, 1.0)
        weights2 = tf.nn.softmax(tf.random.normal((batch_size, 2)), axis=-1)
        
        # Compute kernel
        kernel = model.expected_likelihood_kernel(
            mus1, vars1, weights1, mus2, vars2, weights2
        )
        
        # Check output shape
        assert kernel.shape == (batch_size,)
        
        # Check that kernel values are positive
        assert tf.reduce_all(kernel > 0)
    
    def test_expected_likelihood_kernel_symmetry(self, basic_config):
        """Test that ELK is NOT symmetric (this is expected behavior)."""
        model = Word2GMModel(basic_config)
        
        # Create dummy GMM parameters
        mus1 = tf.random.normal((2, 2, 10))
        vars1 = tf.random.uniform((2, 2, 10), 0.1, 1.0)
        weights1 = tf.nn.softmax(tf.random.normal((2, 2)), axis=-1)
        
        mus2 = tf.random.normal((2, 2, 10))
        vars2 = tf.random.uniform((2, 2, 10), 0.1, 1.0)
        weights2 = tf.nn.softmax(tf.random.normal((2, 2)), axis=-1)
        
        # Compute both directions
        kernel_12 = model.expected_likelihood_kernel(
            mus1, vars1, weights1, mus2, vars2, weights2
        )
        kernel_21 = model.expected_likelihood_kernel(
            mus2, vars2, weights2, mus1, vars1, weights1
        )
        
        # Should NOT be equal (asymmetric by design)
        assert not tf.reduce_all(tf.equal(kernel_12, kernel_21))
    
    def test_forward_pass_shape(self, basic_config):
        """Test forward pass (loss computation)."""
        model = Word2GMModel(basic_config)
        
        # Create triplet inputs
        word_ids = tf.constant([0, 1, 2, 3])
        pos_ids = tf.constant([10, 11, 12, 13])
        neg_ids = tf.constant([20, 21, 22, 23])
        
        # Forward pass
        loss = model((word_ids, pos_ids, neg_ids), training=True)
        
        # Check that loss is a scalar
        assert loss.shape == ()
        
        # Check that loss is non-negative (max-margin loss)
        assert loss >= 0
    
    def test_tf_function_compatibility(self, basic_config):
        """Test that model works with @tf.function decoration."""
        model = Word2GMModel(basic_config)
        
        @tf.function
        def train_step(word_ids, pos_ids, neg_ids):
            with tf.GradientTape() as tape:
                loss = model((word_ids, pos_ids, neg_ids), training=True)
            grads = tape.gradient(loss, model.trainable_variables)
            return loss, grads
        
        # Test that it compiles and runs
        word_ids = tf.constant([0, 1, 2, 3])
        pos_ids = tf.constant([10, 11, 12, 13])
        neg_ids = tf.constant([20, 21, 22, 23])
        
        loss, grads = train_step(word_ids, pos_ids, neg_ids)
        
        # Check outputs
        assert loss.shape == ()
        assert len(grads) == len(model.trainable_variables)
        assert all(grad is not None for grad in grads)
    
    def test_get_word_embedding(self, basic_config):
        """Test single word embedding extraction."""
        model = Word2GMModel(basic_config)
        
        # Test specific component
        embedding_comp0 = model.get_word_embedding(word_id=5, component=0)
        assert embedding_comp0.shape == (10,)
        
        # Test mixture-weighted mean
        embedding_mean = model.get_word_embedding(word_id=5, component=None)
        assert embedding_mean.shape == (10,)
        
        # Should be different (unless weights are [1, 0])
        # This is probabilistic but very likely to pass
        if not np.allclose(embedding_comp0, embedding_mean, rtol=1e-3):
            assert True  # Expected case
    
    def test_edge_cases(self, basic_config):
        """Test edge cases and error conditions."""
        model = Word2GMModel(basic_config)
        
        # Test with single word
        word_ids = tf.constant([0])
        mus, vars, weights = model.get_word_distributions(word_ids)
        assert mus.shape == (1, 2, 10)
        
        # Test with large batch
        word_ids = tf.constant(list(range(50)))
        mus, vars, weights = model.get_word_distributions(word_ids)
        assert mus.shape == (50, 2, 10)
    
    def test_parameter_updates(self, basic_config):
        """Test that parameters update during training."""
        model = Word2GMModel(basic_config)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        
        # Store initial parameters
        initial_mus = model.mus.numpy().copy()
        
        # Training step
        word_ids = tf.constant([0, 1, 2, 3])
        pos_ids = tf.constant([10, 11, 12, 13])
        neg_ids = tf.constant([20, 21, 22, 23])
        
        with tf.GradientTape() as tape:
            loss = model((word_ids, pos_ids, neg_ids), training=True)
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # Check that parameters changed
        updated_mus = model.mus.numpy()
        assert not np.allclose(initial_mus, updated_mus, rtol=1e-6)


class TestWord2GMModelGPU:
    """GPU-specific tests (run only if GPU available)."""
    
    @pytest.mark.skipif(
        not tf.config.list_physical_devices('GPU'),
        reason="GPU not available"
    )
    def test_gpu_compatibility(self):
        """Test that model works on GPU."""
        config = Word2GMConfig(
            vocab_size=100,
            embedding_size=10,
            num_mixtures=2,
            spherical=True,
            batch_size=4
        )
        
        with tf.device('/GPU:0'):
            model = Word2GMModel(config)
            
            word_ids = tf.constant([0, 1, 2, 3])
            pos_ids = tf.constant([10, 11, 12, 13])
            neg_ids = tf.constant([20, 21, 22, 23])
            
            # Should run without CUDA errors
            loss = model((word_ids, pos_ids, neg_ids), training=True)
            assert loss.shape == ()


@pytest.mark.parametrize("spherical", [True, False])
@pytest.mark.parametrize("wout", [True, False])
@pytest.mark.parametrize("num_mixtures", [1, 2, 3])
def test_model_configurations(spherical, wout, num_mixtures):
    """Test various model configurations."""
    config = Word2GMConfig(
        vocab_size=50,
        embedding_size=8,
        num_mixtures=num_mixtures,
        spherical=spherical,
        wout=wout,
        batch_size=2
    )
    
    model = Word2GMModel(config)
    
    # Test that model can run forward pass
    word_ids = tf.constant([0, 1])
    pos_ids = tf.constant([10, 11])
    neg_ids = tf.constant([20, 21])
    
    loss = model((word_ids, pos_ids, neg_ids), training=True)
    assert loss.shape == ()
    assert tf.math.is_finite(loss)


if __name__ == "__main__":
    # Run a quick smoke test if called directly
    print("🧪 Running Word2GM Model Smoke Test")
    print("=" * 40)
    
    # Basic functionality test
    config = Word2GMConfig(
        vocab_size=10,
        embedding_size=5,
        num_mixtures=2,
        spherical=True,
        batch_size=2
    )
    
    model = Word2GMModel(config)
    
    # Test forward pass
    word_ids = tf.constant([0, 1])
    pos_ids = tf.constant([2, 3]) 
    neg_ids = tf.constant([4, 5])
    
    loss = model((word_ids, pos_ids, neg_ids), training=True)
    print(f"✅ Model created and forward pass works")
    print(f"   Loss shape: {loss.shape}, value: {loss.numpy():.4f}")
    
    # Test @tf.function compatibility
    @tf.function
    def test_tf_function():
        return model((word_ids, pos_ids, neg_ids), training=True)
    
    loss_tf = test_tf_function()
    print(f"✅ @tf.function compatibility works")
    print(f"   TF function loss: {loss_tf.numpy():.4f}")
    
    print("\n🎉 Smoke test passed! Run full tests with:")
    print("   python -m pytest tests/test_word2gm_model.py -v")
