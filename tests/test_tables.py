"""
Pytest tests for word2gm_fast.io.tables module.
"""
import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
src_path = PROJECT_ROOT / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import tensorflow as tf
from word2gm_fast.io.tables import create_token_to_index_table, create_index_to_token_table
from word2gm_fast.io.vocab import write_vocab_to_tfrecord


class TestTablesModule:
    """Test class for lookup table creation functionality."""
    
    @pytest.fixture
    def sample_vocab_tfrecord(self, tmp_path):
        """Create a sample vocabulary TFRecord file."""
        vocab_words = ["UNK", "the", "quick", "brown", "fox", "jumps", "over"]
        vocab_indices = list(range(len(vocab_words)))
        frequencies = {word: 100.0 - i*10 for i, word in enumerate(vocab_words)}
        
        vocab_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(vocab_words),
                values=tf.constant(vocab_indices, dtype=tf.int64)
            ),
            default_value=0
        )
        
        vocab_path = tmp_path / "vocab_for_tables.tfrecord"
        write_vocab_to_tfrecord(vocab_table, str(vocab_path), frequencies=frequencies)
        
        return str(vocab_path), vocab_words, vocab_indices, frequencies
    
    def test_create_token_to_index_table(self, sample_vocab_tfrecord):
        """Test creating token-to-index lookup table."""
        vocab_path, vocab_words, vocab_indices, frequencies = sample_vocab_tfrecord
        
        # Create the table
        token_to_index_table = create_token_to_index_table(vocab_path)
        
        # Verify it's a StaticHashTable
        assert isinstance(token_to_index_table, tf.lookup.StaticHashTable)
        
        # Test lookups
        for word, expected_index in zip(vocab_words, vocab_indices):
            actual_index = token_to_index_table.lookup(tf.constant(word))
            assert actual_index.numpy() == expected_index
    
    def test_create_index_to_token_table(self, sample_vocab_tfrecord):
        """Test creating index-to-token lookup table."""
        vocab_path, vocab_words, vocab_indices, frequencies = sample_vocab_tfrecord
        
        # Create the table
        index_to_token_table = create_index_to_token_table(vocab_path)
        
        # Verify it's a StaticHashTable
        assert isinstance(index_to_token_table, tf.lookup.StaticHashTable)
        
        # Test lookups
        for word, index in zip(vocab_words, vocab_indices):
            actual_token = index_to_token_table.lookup(tf.constant([index], dtype=tf.int64))
            assert actual_token.numpy()[0].decode('utf-8') == word
    
    def test_token_to_index_default_value(self, sample_vocab_tfrecord):
        """Test default value behavior for token-to-index table."""
        vocab_path, vocab_words, vocab_indices, frequencies = sample_vocab_tfrecord
        
        token_to_index_table = create_token_to_index_table(vocab_path)
        
        # Test lookup of unknown token
        unknown_index = token_to_index_table.lookup(tf.constant("UNKNOWN_TOKEN"))
        assert unknown_index.numpy() == 0  # Should return UNK index (0)
    
    def test_index_to_token_default_value(self, sample_vocab_tfrecord):
        """Test default value behavior for index-to-token table."""
        vocab_path, vocab_words, vocab_indices, frequencies = sample_vocab_tfrecord
        
        index_to_token_table = create_index_to_token_table(vocab_path)
        
        # Test lookup of unknown index
        unknown_token = index_to_token_table.lookup(tf.constant([999], dtype=tf.int64))
        assert unknown_token.numpy()[0].decode('utf-8') == "UNK"  # Should return UNK token
    
    def test_table_roundtrip(self, sample_vocab_tfrecord):
        """Test roundtrip: token -> index -> token."""
        vocab_path, vocab_words, vocab_indices, frequencies = sample_vocab_tfrecord
        
        token_to_index_table = create_token_to_index_table(vocab_path)
        index_to_token_table = create_index_to_token_table(vocab_path)
        
        # Test roundtrip for each word
        for word in vocab_words:
            # Token -> Index
            index = token_to_index_table.lookup(tf.constant(word))
            
            # Index -> Token
            recovered_token = index_to_token_table.lookup(tf.constant([index], dtype=tf.int64))
            recovered_word = recovered_token.numpy()[0].decode('utf-8')
            
            assert recovered_word == word
    
    def test_batch_lookups(self, sample_vocab_tfrecord):
        """Test batch lookups for both tables."""
        vocab_path, vocab_words, vocab_indices, frequencies = sample_vocab_tfrecord
        
        token_to_index_table = create_token_to_index_table(vocab_path)
        index_to_token_table = create_index_to_token_table(vocab_path)
        
        # Test batch token-to-index lookup
        batch_tokens = tf.constant(vocab_words[:3])  # ["UNK", "the", "quick"]
        batch_indices = token_to_index_table.lookup(batch_tokens)
        
        expected_indices = [0, 1, 2]
        assert batch_indices.numpy().tolist() == expected_indices
        
        # Test batch index-to-token lookup
        batch_indices_input = tf.constant([0, 1, 2], dtype=tf.int64)
        batch_tokens_output = index_to_token_table.lookup(batch_indices_input)
        
        expected_tokens = ["UNK", "the", "quick"]
        actual_tokens = [token.decode('utf-8') for token in batch_tokens_output.numpy()]
        assert actual_tokens == expected_tokens
    
    def test_table_size(self, sample_vocab_tfrecord):
        """Test table size reporting."""
        vocab_path, vocab_words, vocab_indices, frequencies = sample_vocab_tfrecord
        
        token_to_index_table = create_token_to_index_table(vocab_path)
        index_to_token_table = create_index_to_token_table(vocab_path)
        
        # Both tables should have the same size
        token_table_size = token_to_index_table.size()
        index_table_size = index_to_token_table.size()
        
        assert token_table_size.numpy() == len(vocab_words)
        assert index_table_size.numpy() == len(vocab_words)
        assert token_table_size.numpy() == index_table_size.numpy()
    
    def test_tables_with_compressed_vocab(self, tmp_path):
        """Test creating tables from compressed vocab TFRecord."""
        vocab_words = ["UNK", "compressed", "vocab"]
        vocab_indices = list(range(len(vocab_words)))
        
        vocab_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(vocab_words),
                values=tf.constant(vocab_indices, dtype=tf.int64)
            ),
            default_value=0
        )
        
        vocab_path = tmp_path / "compressed_vocab.tfrecord"
        write_vocab_to_tfrecord(vocab_table, str(vocab_path), compress=True)
        
        # Tables should work with compressed files
        gz_path = str(vocab_path) + ".gz"
        token_to_index_table = create_token_to_index_table(gz_path)
        index_to_token_table = create_index_to_token_table(gz_path)
        
        # Test a lookup
        index = token_to_index_table.lookup(tf.constant("compressed"))
        assert index.numpy() == 1
        
        token = index_to_token_table.lookup(tf.constant([1], dtype=tf.int64))
        assert token.numpy()[0].decode('utf-8') == "compressed"
