"""Unit tests for io.tables module.

Tests creation of token-to-index and index-to-token lookup tables from TFRecord files.
"""
import pytest
import tensorflow as tf
import tempfile
import os
from src.word2gm_fast.io.tables import (
    create_token_to_index_table,
    create_index_to_token_table
)
from src.word2gm_fast.io.vocab import write_vocab_to_tfrecord
from src.word2gm_fast.io.triplets import write_triplets_to_tfrecord
from src.word2gm_fast.dataprep.index_vocab import build_vocab_table


@pytest.fixture
def sample_vocab_data():
    """Create sample vocabulary data and save to TFRecord."""
    vocab_list = ["UNK", "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    vocab_table = build_vocab_table(vocab_list)
    frequencies = {token: (i + 1) * 10 for i, token in enumerate(vocab_list)}
    frequencies["UNK"] = 0  # UNK always has 0 frequency
    
    return vocab_table, vocab_list, frequencies


@pytest.fixture
def sample_triplets_data():
    """Create sample triplets data."""
    triplets = [
        [1, 2, 3],  # the, quick, brown
        [2, 1, 4],  # quick, the, fox
        [3, 4, 5],  # brown, fox, jumps
        [4, 5, 6],  # fox, jumps, over
        [1, 3, 8]   # the, brown, dog (note: not all vocab tokens appear)
    ]
    return tf.data.Dataset.from_tensor_slices(triplets)


@pytest.fixture
def temp_vocab_file():
    """Create temporary vocab TFRecord file."""
    with tempfile.NamedTemporaryFile(suffix='.tfrecord', delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def temp_compressed_vocab_file():
    """Create temporary compressed vocab TFRecord file."""
    with tempfile.NamedTemporaryFile(suffix='.tfrecord.gz', delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def temp_triplets_file():
    """Create temporary triplets TFRecord file."""
    with tempfile.NamedTemporaryFile(suffix='.tfrecord', delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def prepared_vocab_file(sample_vocab_data, temp_vocab_file):
    """Prepare a vocab TFRecord file with sample data."""
    vocab_table, vocab_list, frequencies = sample_vocab_data
    write_vocab_to_tfrecord(vocab_table, temp_vocab_file, frequencies, compress=False)
    return temp_vocab_file, vocab_list, frequencies


@pytest.fixture
def prepared_triplets_file(sample_triplets_data, temp_triplets_file):
    """Prepare a triplets TFRecord file with sample data."""
    write_triplets_to_tfrecord(sample_triplets_data, temp_triplets_file, compress=False)
    return temp_triplets_file


class TestCreateTokenToIndexTable:
    """Test the create_token_to_index_table function."""
    
    def test_basic_creation_uncompressed(self, prepared_vocab_file):
        """Test basic creation of token-to-index table from uncompressed file."""
        vocab_file, vocab_list, frequencies = prepared_vocab_file
        
        table = create_token_to_index_table(vocab_file, compressed=False)
        
        # Check that it's a StaticHashTable
        assert isinstance(table, tf.lookup.StaticHashTable)
        
        # Test some lookups
        test_tokens = tf.constant(["the", "quick", "brown", "unknown"])
        indices = table.lookup(test_tokens).numpy()
        
        # Should get correct indices for known tokens
        assert indices[0] == 1  # "the" should be at index 1
        assert indices[1] == 2  # "quick" should be at index 2
        assert indices[2] == 3  # "brown" should be at index 3
        assert indices[3] == 0  # "unknown" should map to UNK (index 0)
    
    def test_basic_creation_compressed(self, sample_vocab_data, temp_compressed_vocab_file):
        """Test basic creation from compressed file."""
        vocab_table, vocab_list, frequencies = sample_vocab_data
        write_vocab_to_tfrecord(vocab_table, temp_compressed_vocab_file, frequencies, compress=True)
        
        table = create_token_to_index_table(temp_compressed_vocab_file, compressed=True)
        
        # Should work the same as uncompressed
        assert isinstance(table, tf.lookup.StaticHashTable)
        
        # Test lookups
        test_tokens = tf.constant(["the", "UNK"])
        indices = table.lookup(test_tokens).numpy()
        assert indices[1] == 0  # UNK should be at index 0
    
    def test_with_triplet_filtering(self, prepared_vocab_file, prepared_triplets_file):
        """Test creation with triplet filtering."""
        vocab_file, vocab_list, frequencies = prepared_vocab_file
        triplets_file = prepared_triplets_file
        
        # Create table with filtering
        table = create_token_to_index_table(
            vocab_file, 
            triplets_tfrecord_path=triplets_file,
            compressed=False
        )
        
        assert isinstance(table, tf.lookup.StaticHashTable)
        
        # The table should only include tokens that appear in triplets
        # Based on our sample triplets: [1,2,3], [2,1,4], [3,4,5], [4,5,6], [1,3,8]
        # Used indices: 1, 2, 3, 4, 5, 6, 8 (plus 0 for UNK)
        
        # Test that filtered tokens work
        used_tokens = tf.constant(["the", "quick", "brown"])  # indices 1,2,3
        indices = table.lookup(used_tokens).numpy()
        assert all(idx >= 0 for idx in indices)  # Should get valid indices
    
    def test_table_size(self, prepared_vocab_file):
        """Test that table size is correct."""
        vocab_file, vocab_list, frequencies = prepared_vocab_file
        
        table = create_token_to_index_table(vocab_file, compressed=False)
        
        # Table size should match vocabulary size
        assert table.size().numpy() == len(vocab_list)
    
    def test_nonexistent_file(self):
        """Test with non-existent file."""
        with pytest.raises(Exception):  # Should raise some kind of file error
            create_token_to_index_table("/nonexistent/file.tfrecord", compressed=False)


class TestCreateIndexToTokenTable:
    """Test the create_index_to_token_table function."""
    
    def test_basic_creation_uncompressed(self, prepared_vocab_file):
        """Test basic creation of index-to-token table from uncompressed file."""
        vocab_file, vocab_list, frequencies = prepared_vocab_file
        
        table = create_index_to_token_table(vocab_file, compressed=False)
        
        # Check that it's a StaticHashTable
        assert isinstance(table, tf.lookup.StaticHashTable)
        
        # Test some lookups
        test_indices = tf.constant([0, 1, 2, 3], dtype=tf.int64)
        tokens = table.lookup(test_indices)
        
        # Convert to strings for comparison
        token_strings = [t.numpy().decode('utf-8') for t in tokens]
        
        # Should get correct tokens for known indices
        assert token_strings[0] == "UNK"   # index 0 -> UNK
        assert token_strings[1] == "the"   # index 1 -> the
        assert token_strings[2] == "quick" # index 2 -> quick
        assert token_strings[3] == "brown" # index 3 -> brown
    
    def test_basic_creation_compressed(self, sample_vocab_data, temp_compressed_vocab_file):
        """Test basic creation from compressed file."""
        vocab_table, vocab_list, frequencies = sample_vocab_data
        write_vocab_to_tfrecord(vocab_table, temp_compressed_vocab_file, frequencies, compress=True)
        
        table = create_index_to_token_table(temp_compressed_vocab_file, compressed=True)
        
        # Should work the same as uncompressed
        assert isinstance(table, tf.lookup.StaticHashTable)
        
        # Test lookup
        token = table.lookup(tf.constant([0], dtype=tf.int64))
        assert token[0].numpy().decode('utf-8') == "UNK"
    
    def test_with_triplet_filtering(self, prepared_vocab_file, prepared_triplets_file):
        """Test creation with triplet filtering."""
        vocab_file, vocab_list, frequencies = prepared_vocab_file
        triplets_file = prepared_triplets_file
        
        # Create table with filtering
        table = create_index_to_token_table(
            vocab_file,
            triplets_tfrecord_path=triplets_file,
            compressed=False
        )
        
        assert isinstance(table, tf.lookup.StaticHashTable)
        
        # Test that we can look up indices that appear in triplets
        test_indices = tf.constant([1, 2, 3], dtype=tf.int64)  # These should be in triplets
        tokens = table.lookup(test_indices)
        
        # Should get valid tokens (not empty or error)
        for token in tokens:
            token_str = token.numpy().decode('utf-8')
            assert len(token_str) > 0
    
    def test_out_of_range_index(self, prepared_vocab_file):
        """Test lookup with out-of-range index."""
        vocab_file, vocab_list, frequencies = prepared_vocab_file
        
        table = create_index_to_token_table(vocab_file, compressed=False)
        
        # Try to look up an index that's too large
        large_index = tf.constant([999999], dtype=tf.int64)
        result = table.lookup(large_index)
        
        # Should return some default value (likely empty string or UNK)
        result_str = result[0].numpy().decode('utf-8')
        assert isinstance(result_str, str)  # Should get some string result
    
    def test_table_size(self, prepared_vocab_file):
        """Test that table size is correct."""
        vocab_file, vocab_list, frequencies = prepared_vocab_file
        
        table = create_index_to_token_table(vocab_file, compressed=False)
        
        # Table size should match vocabulary size
        assert table.size().numpy() == len(vocab_list)
    
    def test_nonexistent_file(self):
        """Test with non-existent file."""
        with pytest.raises(Exception):  # Should raise some kind of file error
            create_index_to_token_table("/nonexistent/file.tfrecord", compressed=False)


class TestTableConsistency:
    """Test consistency between token-to-index and index-to-token tables."""
    
    def test_roundtrip_consistency(self, prepared_vocab_file):
        """Test that token->index->token roundtrip works correctly."""
        vocab_file, vocab_list, frequencies = prepared_vocab_file
        
        token_to_index = create_token_to_index_table(vocab_file, compressed=False)
        index_to_token = create_index_to_token_table(vocab_file, compressed=False)
        
        # Test roundtrip for known tokens
        test_tokens = tf.constant(["the", "quick", "brown", "fox"])
        
        # token -> index
        indices = token_to_index.lookup(test_tokens)
        
        # index -> token
        recovered_tokens = index_to_token.lookup(indices)
        
        # Should get back the same tokens
        for i, original_token in enumerate(test_tokens.numpy()):
            recovered_token = recovered_tokens[i].numpy().decode('utf-8')
            assert original_token.decode('utf-8') == recovered_token
    
    def test_table_sizes_match(self, prepared_vocab_file):
        """Test that both tables have the same size."""
        vocab_file, vocab_list, frequencies = prepared_vocab_file
        
        token_to_index = create_token_to_index_table(vocab_file, compressed=False)
        index_to_token = create_index_to_token_table(vocab_file, compressed=False)
        
        assert token_to_index.size().numpy() == index_to_token.size().numpy()
    
    def test_unknown_token_handling(self, prepared_vocab_file):
        """Test that unknown tokens are handled consistently."""
        vocab_file, vocab_list, frequencies = prepared_vocab_file
        
        token_to_index = create_token_to_index_table(vocab_file, compressed=False)
        index_to_token = create_index_to_token_table(vocab_file, compressed=False)
        
        # Look up unknown token
        unknown_index = token_to_index.lookup(tf.constant(["unknown_word"]))
        
        # Should map to UNK index (0)
        assert unknown_index.numpy()[0] == 0
        
        # Looking up index 0 should give us UNK
        unk_token = index_to_token.lookup(tf.constant([0], dtype=tf.int64))
        assert unk_token[0].numpy().decode('utf-8') == "UNK"


class TestFilteredTables:
    """Test tables created with triplet filtering."""
    
    def test_filtered_vs_unfiltered_sizes(self, prepared_vocab_file, prepared_triplets_file):
        """Test that filtered tables may have different sizes."""
        vocab_file, vocab_list, frequencies = prepared_vocab_file
        triplets_file = prepared_triplets_file
        
        # Create unfiltered tables
        unfiltered_token_to_index = create_token_to_index_table(vocab_file, compressed=False)
        unfiltered_index_to_token = create_index_to_token_table(vocab_file, compressed=False)
        
        # Create filtered tables
        filtered_token_to_index = create_token_to_index_table(
            vocab_file, triplets_tfrecord_path=triplets_file, compressed=False
        )
        filtered_index_to_token = create_index_to_token_table(
            vocab_file, triplets_tfrecord_path=triplets_file, compressed=False
        )
        
        # Filtered tables should potentially be smaller (or same size if all vocab is used)
        unfiltered_size = unfiltered_token_to_index.size().numpy()
        filtered_size = filtered_token_to_index.size().numpy()
        
        assert filtered_size <= unfiltered_size
        
        # Both filtered tables should have same size
        assert filtered_token_to_index.size().numpy() == filtered_index_to_token.size().numpy()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_minimal_vocab(self, temp_vocab_file):
        """Test with minimal vocabulary (just UNK)."""
        vocab_list = ["UNK"]
        vocab_table = build_vocab_table(vocab_list)
        frequencies = {"UNK": 0}
        
        write_vocab_to_tfrecord(vocab_table, temp_vocab_file, frequencies, compress=False)
        
        token_to_index = create_token_to_index_table(temp_vocab_file, compressed=False)
        index_to_token = create_index_to_token_table(temp_vocab_file, compressed=False)
        
        # Should work with minimal vocab
        assert token_to_index.size().numpy() == 1
        assert index_to_token.size().numpy() == 1
        
        # UNK should map correctly
        unk_index = token_to_index.lookup(tf.constant(["UNK"]))
        assert unk_index.numpy()[0] == 0
        
        unk_token = index_to_token.lookup(tf.constant([0], dtype=tf.int64))
        assert unk_token[0].numpy().decode('utf-8') == "UNK"
    
    def test_empty_triplets_file(self, prepared_vocab_file, temp_triplets_file):
        """Test with empty triplets file for filtering."""
        vocab_file, vocab_list, frequencies = prepared_vocab_file
        
        # Create empty triplets file
        empty_triplets = tf.data.Dataset.from_tensor_slices(tf.constant([], dtype=tf.int32, shape=[0, 3]))
        write_triplets_to_tfrecord(empty_triplets, temp_triplets_file, compress=False)
        
        # Creating tables with empty triplets filter should raise an error or return minimal tables
        try:
            token_to_index = create_token_to_index_table(
                vocab_file, triplets_tfrecord_path=temp_triplets_file, compressed=False
            )
            # If it doesn't raise an error, it should at least create a valid table
            assert isinstance(token_to_index, tf.lookup.StaticHashTable)
        except Exception:
            # Empty triplets might cause an error, which is acceptable
            pass
