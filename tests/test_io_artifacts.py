"""Unit tests for io.artifacts module.

Tests pipeline artifacts management for saving/loading complete training data.
"""
import pytest
import tensorflow as tf
import tempfile
import os
from unittest.mock import patch
from src.word2gm_fast.io.artifacts import (
    save_pipeline_artifacts,
    load_pipeline_artifacts
)
from src.word2gm_fast.dataprep.index_vocab import build_vocab_table


@pytest.fixture
def sample_pipeline_data():
    """Create sample pipeline data for testing."""
    # Create sample dataset
    lines = ["the quick brown fox", "a quick brown dog"]
    dataset = tf.data.Dataset.from_tensor_slices(lines)
    
    # Create vocabulary
    vocab_list = ["UNK", "the", "quick", "brown", "fox", "a", "dog"]
    vocab_table = build_vocab_table(vocab_list)
    
    # Create integer triplets
    triplets = [
        [2, 1, 6],  # quick, the, dog
        [3, 2, 5],  # brown, quick, a
        [4, 3, 1],  # fox, brown, the
        [3, 5, 4]   # brown, a, fox
    ]
    triplets_ds = tf.data.Dataset.from_tensor_slices(triplets)
    
    return dataset, vocab_table, triplets_ds


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestSavePipelineArtifacts:
    """Test the save_pipeline_artifacts function."""
    
    def test_basic_save_uncompressed(self, sample_pipeline_data, temp_output_dir):
        """Test basic saving of pipeline artifacts without compression."""
        dataset, vocab_table, triplets_ds = sample_pipeline_data
        
        with patch('IPython.display.display_markdown'):  # Mock display
            artifacts = save_pipeline_artifacts(
                dataset=dataset,
                vocab_table=vocab_table,
                triplets_ds=triplets_ds,
                output_dir=temp_output_dir,
                compress=False
            )
        
        # Check return structure
        assert isinstance(artifacts, dict)
        expected_keys = {'vocab_path', 'triplets_path', 'vocab_size', 'triplet_count', 'compressed', 'output_dir'}
        assert set(artifacts.keys()) == expected_keys
        
        # Check file paths
        assert artifacts['vocab_path'].endswith('.tfrecord')
        assert artifacts['triplets_path'].endswith('.tfrecord')
        assert not artifacts['compressed']
        
        # Check files were created
        assert os.path.exists(artifacts['vocab_path'])
        assert os.path.exists(artifacts['triplets_path'])
        
        # Check sizes
        assert artifacts['vocab_size'] > 0
        assert artifacts['triplet_count'] > 0
    
    def test_basic_save_compressed(self, sample_pipeline_data, temp_output_dir):
        """Test basic saving of pipeline artifacts with compression."""
        dataset, vocab_table, triplets_ds = sample_pipeline_data
        
        with patch('IPython.display.display_markdown'):  # Mock display
            artifacts = save_pipeline_artifacts(
                dataset=dataset,
                vocab_table=vocab_table,
                triplets_ds=triplets_ds,
                output_dir=temp_output_dir,
                compress=True
            )
        
        # Check compressed file extensions
        assert artifacts['vocab_path'].endswith('.tfrecord.gz')
        assert artifacts['triplets_path'].endswith('.tfrecord.gz')
        assert artifacts['compressed']
        
        # Check files were created
        assert os.path.exists(artifacts['vocab_path'])
        assert os.path.exists(artifacts['triplets_path'])
    
    def test_directory_creation(self, sample_pipeline_data):
        """Test that output directory is created if it doesn't exist."""
        dataset, vocab_table, triplets_ds = sample_pipeline_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = os.path.join(temp_dir, "nested", "output")
            
            # Directory doesn't exist yet
            assert not os.path.exists(nested_dir)
            
            with patch('IPython.display.display_markdown'):  # Mock display
                artifacts = save_pipeline_artifacts(
                    dataset=dataset,
                    vocab_table=vocab_table,
                    triplets_ds=triplets_ds,
                    output_dir=nested_dir,
                    compress=False
                )
            
            # Directory should now exist
            assert os.path.exists(nested_dir)
            assert artifacts['output_dir'] == nested_dir
    
    def test_empty_triplets(self, temp_output_dir):
        """Test saving with empty triplets dataset."""
        # Create minimal data
        dataset = tf.data.Dataset.from_tensor_slices(["empty"])
        vocab_list = ["UNK"]
        vocab_table = build_vocab_table(vocab_list)
        empty_triplets = tf.data.Dataset.from_tensor_slices(tf.constant([], dtype=tf.int32, shape=[0, 3]))
        
        with patch('IPython.display.display_markdown'):  # Mock display
            artifacts = save_pipeline_artifacts(
                dataset=dataset,
                vocab_table=vocab_table,
                triplets_ds=empty_triplets,
                output_dir=temp_output_dir,
                compress=False
            )
        
        assert artifacts['triplet_count'] == 0
        assert artifacts['vocab_size'] == 1  # Just UNK


class TestLoadPipelineArtifacts:
    """Test the load_pipeline_artifacts function."""
    
    def test_basic_load_uncompressed(self, sample_pipeline_data, temp_output_dir):
        """Test basic loading of uncompressed pipeline artifacts."""
        dataset, vocab_table, triplets_ds = sample_pipeline_data
        
        # First save the artifacts
        with patch('IPython.display.display_markdown'):  # Mock display
            saved_artifacts = save_pipeline_artifacts(
                dataset=dataset,
                vocab_table=vocab_table,
                triplets_ds=triplets_ds,
                output_dir=temp_output_dir,
                compress=False
            )
        
        # Then load them back
        with patch('IPython.display.display_markdown'):  # Mock display
            loaded_artifacts = load_pipeline_artifacts(
                output_dir=temp_output_dir,
                compressed=False
            )
        
        # Check return structure
        expected_keys = {'token_to_index_table', 'index_to_token_table', 'triplets_ds', 'vocab_size', 'filtered_to_triplets'}
        assert set(loaded_artifacts.keys()) == expected_keys
        
        # Check types
        assert isinstance(loaded_artifacts['token_to_index_table'], tf.lookup.StaticHashTable)
        assert isinstance(loaded_artifacts['index_to_token_table'], tf.lookup.StaticHashTable)
        assert isinstance(loaded_artifacts['triplets_ds'], tf.data.Dataset)
        assert isinstance(loaded_artifacts['vocab_size'], int)
        assert isinstance(loaded_artifacts['filtered_to_triplets'], bool)
        
        # Check vocab size consistency
        assert loaded_artifacts['vocab_size'] == saved_artifacts['vocab_size']
        assert not loaded_artifacts['filtered_to_triplets']
    
    def test_basic_load_compressed(self, sample_pipeline_data, temp_output_dir):
        """Test basic loading of compressed pipeline artifacts."""
        dataset, vocab_table, triplets_ds = sample_pipeline_data
        
        # First save the artifacts
        with patch('IPython.display.display_markdown'):  # Mock display
            saved_artifacts = save_pipeline_artifacts(
                dataset=dataset,
                vocab_table=vocab_table,
                triplets_ds=triplets_ds,
                output_dir=temp_output_dir,
                compress=True
            )
        
        # Then load them back
        with patch('IPython.display.display_markdown'):  # Mock display
            loaded_artifacts = load_pipeline_artifacts(
                output_dir=temp_output_dir,
                compressed=True
            )
        
        # Should work the same as uncompressed
        assert loaded_artifacts['vocab_size'] == saved_artifacts['vocab_size']
    
    def test_auto_detect_compression(self, sample_pipeline_data, temp_output_dir):
        """Test auto-detection of compression."""
        dataset, vocab_table, triplets_ds = sample_pipeline_data
        
        # Save compressed artifacts
        with patch('IPython.display.display_markdown'):  # Mock display
            save_pipeline_artifacts(
                dataset=dataset,
                vocab_table=vocab_table,
                triplets_ds=triplets_ds,
                output_dir=temp_output_dir,
                compress=True
            )
        
        # Load without specifying compression (should auto-detect)
        with patch('IPython.display.display_markdown'):  # Mock display
            loaded_artifacts = load_pipeline_artifacts(
                output_dir=temp_output_dir,
                compressed=None  # Auto-detect
            )
        
        # Should work correctly
        assert loaded_artifacts['vocab_size'] > 0
    
    def test_load_with_filtering(self, sample_pipeline_data, temp_output_dir):
        """Test loading with triplet filtering enabled."""
        dataset, vocab_table, triplets_ds = sample_pipeline_data
        
        # Save artifacts
        with patch('IPython.display.display_markdown'):  # Mock display
            save_pipeline_artifacts(
                dataset=dataset,
                vocab_table=vocab_table,
                triplets_ds=triplets_ds,
                output_dir=temp_output_dir,
                compress=False
            )
        
        # Load with filtering
        with patch('IPython.display.display_markdown'):  # Mock display
            loaded_artifacts = load_pipeline_artifacts(
                output_dir=temp_output_dir,
                compressed=False,
                filter_to_triplets=True
            )
        
        # Should indicate filtering was applied
        assert loaded_artifacts['filtered_to_triplets']
    
    def test_load_nonexistent_directory(self):
        """Test loading from non-existent directory."""
        with pytest.raises(FileNotFoundError):
            load_pipeline_artifacts("/nonexistent/directory")
    
    def test_load_empty_directory(self, temp_output_dir):
        """Test loading from directory with no TFRecord files."""
        with pytest.raises(FileNotFoundError):
            load_pipeline_artifacts(temp_output_dir)


class TestRoundTripConsistency:
    """Test save-then-load consistency."""
    
    def test_roundtrip_uncompressed(self, sample_pipeline_data, temp_output_dir):
        """Test that save->load preserves data correctly."""
        dataset, vocab_table, triplets_ds = sample_pipeline_data
        
        # Save
        with patch('IPython.display.display_markdown'):  # Mock display
            saved_artifacts = save_pipeline_artifacts(
                dataset=dataset,
                vocab_table=vocab_table,
                triplets_ds=triplets_ds,
                output_dir=temp_output_dir,
                compress=False
            )
        
        # Load
        with patch('IPython.display.display_markdown'):  # Mock display
            loaded_artifacts = load_pipeline_artifacts(
                output_dir=temp_output_dir,
                compressed=False
            )
        
        # Check triplet count consistency
        original_triplets = list(triplets_ds.as_numpy_iterator())
        loaded_triplets = list(loaded_artifacts['triplets_ds'].as_numpy_iterator())
        
        assert len(loaded_triplets) == len(original_triplets)
        assert len(loaded_triplets) == saved_artifacts['triplet_count']
        
        # Check vocabulary functionality
        token_table = loaded_artifacts['token_to_index_table']
        index_table = loaded_artifacts['index_to_token_table']
        
        # Test some lookups
        test_tokens = tf.constant(["the", "quick", "unknown"])
        indices = token_table.lookup(test_tokens)
        
        # Should get valid indices (or UNK for unknown)
        assert all(idx >= 0 for idx in indices.numpy())
    
    def test_roundtrip_compressed(self, sample_pipeline_data, temp_output_dir):
        """Test that compressed save->load preserves data correctly."""
        dataset, vocab_table, triplets_ds = sample_pipeline_data
        
        # Save compressed
        with patch('IPython.display.display_markdown'):  # Mock display
            saved_artifacts = save_pipeline_artifacts(
                dataset=dataset,
                vocab_table=vocab_table,
                triplets_ds=triplets_ds,
                output_dir=temp_output_dir,
                compress=True
            )
        
        # Load compressed
        with patch('IPython.display.display_markdown'):  # Mock display
            loaded_artifacts = load_pipeline_artifacts(
                output_dir=temp_output_dir,
                compressed=True
            )
        
        # Check consistency
        assert loaded_artifacts['vocab_size'] == saved_artifacts['vocab_size']
        
        # Check triplet data types are correct (int32)
        for triplet in loaded_artifacts['triplets_ds'].take(1):
            center, positive, negative = triplet
            assert center.dtype == tf.int32
            assert positive.dtype == tf.int32
            assert negative.dtype == tf.int32


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_minimal_data(self, temp_output_dir):
        """Test with minimal data (single triplet, minimal vocab)."""
        # Minimal dataset
        dataset = tf.data.Dataset.from_tensor_slices(["test"])
        vocab_list = ["UNK", "test"]
        vocab_table = build_vocab_table(vocab_list)
        triplets_ds = tf.data.Dataset.from_tensor_slices([[1, 1, 1]])  # Single triplet
        
        # Save and load
        with patch('IPython.display.display_markdown'):  # Mock display
            saved_artifacts = save_pipeline_artifacts(
                dataset=dataset,
                vocab_table=vocab_table,
                triplets_ds=triplets_ds,
                output_dir=temp_output_dir,
                compress=False
            )
            
            loaded_artifacts = load_pipeline_artifacts(
                output_dir=temp_output_dir,
                compressed=False
            )
        
        assert saved_artifacts['triplet_count'] == 1
        assert saved_artifacts['vocab_size'] == 2
        assert loaded_artifacts['vocab_size'] == 2
