"""
Pytest tests for word2gm_fast.io.artifacts module.
"""
import pytest
import tempfile
import os
import json
import gzip
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
src_path = PROJECT_ROOT / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import tensorflow as tf
from word2gm_fast.io.artifacts import (
    save_pipeline_artifacts, 
    load_pipeline_artifacts,
    save_metadata,
    load_metadata
)
from word2gm_fast.io.vocab import write_vocab_to_tfrecord
from word2gm_fast.io.triplets import write_triplets_to_tfrecord


class TestArtifactsModule:
    """Test class for artifacts I/O functionality."""
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return {
            'vocab_size': 1000,
            'total_tokens': 50000,
            'model_config': {
                'embedding_dim': 128,
                'epochs': 10,
                'batch_size': 32
            },
            'metadata': {
                'version': '1.0',
                'timestamp': '2025-01-01T00:00:00Z',
                'author': 'test_user'
            },
            'training_params': {
                'learning_rate': 0.001,
                'negative_samples': 5
            }
        }
    
    @pytest.fixture
    def sample_pipeline_data(self, tmp_path):
        """Create sample pipeline data for testing."""
        # Create vocab data
        vocab_words = ["UNK", "the", "quick", "brown", "fox"]
        vocab_indices = list(range(len(vocab_words)))
        vocab_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(vocab_words),
                values=tf.constant(vocab_indices, dtype=tf.int64)
            ),
            default_value=0
        )
        
        # Create triplets data
        triplets = [(1, 2, 3), (2, 3, 4), (3, 4, 1)]
        triplets_ds = tf.data.Dataset.from_tensor_slices([
            tf.constant(triplet, dtype=tf.int64) for triplet in triplets
        ])
        
        # Create text dataset (mock)
        text_ds = tf.data.Dataset.from_tensor_slices([
            b"the quick brown fox",
            b"brown fox jumps",
            b"quick brown animal"
        ])
        
        return {
            'vocab_table': vocab_table,
            'triplets_ds': triplets_ds,
            'text_ds': text_ds,
            'vocab_words': vocab_words,
            'triplets': triplets
        }
    
    def test_save_metadata_uncompressed(self, sample_metadata, tmp_path):
        """Test saving metadata without compression."""
        metadata_path = tmp_path / "metadata.json"
        
        actual_path = save_metadata(sample_metadata, str(metadata_path), compress=False)
        
        assert os.path.exists(actual_path)
        assert actual_path == str(metadata_path)
        
        # Verify content
        with open(actual_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == sample_metadata
    
    def test_save_metadata_compressed(self, sample_metadata, tmp_path):
        """Test saving metadata with compression."""
        metadata_path = tmp_path / "metadata.json"
        
        actual_path = save_metadata(sample_metadata, str(metadata_path), compress=True)
        
        expected_path = str(metadata_path) + ".gz"
        assert os.path.exists(expected_path)
        assert actual_path == expected_path
        
        # Verify content
        with gzip.open(actual_path, 'rt', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == sample_metadata
    
    def test_load_metadata_uncompressed(self, sample_metadata, tmp_path):
        """Test loading uncompressed metadata."""
        metadata_path = tmp_path / "metadata.json"
        
        # Save first
        save_metadata(sample_metadata, str(metadata_path), compress=False)
        
        # Load and verify
        loaded_data = load_metadata(str(metadata_path))
        assert loaded_data == sample_metadata
    
    def test_load_metadata_compressed(self, sample_metadata, tmp_path):
        """Test loading compressed metadata."""
        metadata_path = tmp_path / "metadata.json"
        
        # Save compressed
        actual_path = save_metadata(sample_metadata, str(metadata_path), compress=True)
        
        # Load and verify
        loaded_data = load_metadata(actual_path)
        assert loaded_data == sample_metadata
    
    def test_metadata_roundtrip(self, sample_metadata, tmp_path):
        """Test complete metadata roundtrip."""
        metadata_path = tmp_path / "metadata_roundtrip.json"
        
        # Save and load
        actual_path = save_metadata(sample_metadata, str(metadata_path))
        loaded_data = load_metadata(actual_path)
        
        # Verify complete equality
        assert loaded_data == sample_metadata
        
        # Verify nested structures
        assert loaded_data['model_config']['embedding_dim'] == 128
        assert loaded_data['metadata']['version'] == '1.0'
        assert loaded_data['training_params']['learning_rate'] == 0.001
    
    def test_save_pipeline_artifacts(self, sample_pipeline_data, tmp_path):
        """Test saving complete pipeline artifacts."""
        output_dir = tmp_path / "pipeline_artifacts"
        output_dir.mkdir()
        
        data = sample_pipeline_data
        
        # Save artifacts
        artifacts_info = save_pipeline_artifacts(
            dataset=data['text_ds'],
            vocab_table=data['vocab_table'],
            triplets_ds=data['triplets_ds'],
            output_dir=str(output_dir)
        )
        
        # Verify return structure
        assert isinstance(artifacts_info, dict)
        assert 'vocab_path' in artifacts_info
        assert 'triplets_path' in artifacts_info
        assert 'triplet_count' in artifacts_info
        
        # Verify files were created
        assert os.path.exists(artifacts_info['vocab_path'])
        assert os.path.exists(artifacts_info['triplets_path'])
        
        # Verify triplet count
        assert artifacts_info['triplet_count'] == 3
    
    def test_load_pipeline_artifacts(self, sample_pipeline_data, tmp_path):
        """Test loading complete pipeline artifacts."""
        output_dir = tmp_path / "pipeline_artifacts"
        output_dir.mkdir()
        
        data = sample_pipeline_data
        
        # Save artifacts first
        artifacts_info = save_pipeline_artifacts(
            dataset=data['text_ds'],
            vocab_table=data['vocab_table'],
            triplets_ds=data['triplets_ds'],
            output_dir=str(output_dir)
        )
        
        # Load artifacts
        loaded_artifacts = load_pipeline_artifacts(str(output_dir))
        
        # Verify loaded structure
        assert isinstance(loaded_artifacts, dict)
        assert 'token_to_index_table' in loaded_artifacts
        assert 'index_to_token_table' in loaded_artifacts
        assert 'triplets_ds' in loaded_artifacts
        assert 'vocab_size' in loaded_artifacts
        
        # Verify vocab size
        assert loaded_artifacts['vocab_size'] == 5
        
        # Test table functionality
        token_table = loaded_artifacts['token_to_index_table']
        index = token_table.lookup(tf.constant("the"))
        assert index.numpy() == 1
    
    def test_pipeline_artifacts_roundtrip(self, sample_pipeline_data, tmp_path):
        """Test complete pipeline artifacts roundtrip."""
        output_dir = tmp_path / "pipeline_roundtrip"
        output_dir.mkdir()
        
        data = sample_pipeline_data
        
        # Save artifacts
        save_pipeline_artifacts(
            dataset=data['text_ds'],
            vocab_table=data['vocab_table'],
            triplets_ds=data['triplets_ds'],
            output_dir=str(output_dir)
        )
        
        # Load artifacts
        loaded_artifacts = load_pipeline_artifacts(str(output_dir))
        
        # Test vocabulary roundtrip
        token_to_index = loaded_artifacts['token_to_index_table']
        index_to_token = loaded_artifacts['index_to_token_table']
        
        for word in data['vocab_words']:
            # Token -> Index -> Token
            index = token_to_index.lookup(tf.constant(word))
            # Convert scalar to tensor with shape [1] for lookup
            index_tensor = tf.expand_dims(index, 0)
            recovered_token = index_to_token.lookup(index_tensor)
            recovered_word = recovered_token.numpy()[0].decode('utf-8')
            assert recovered_word == word
        
        # Test triplets dataset
        triplets_ds = loaded_artifacts['triplets_ds']
        loaded_triplets = []
        for triplet in triplets_ds:
            center, pos, neg = triplet
            loaded_triplets.append((
                center.numpy(),
                pos.numpy(),
                neg.numpy()
            ))
        
        assert len(loaded_triplets) == 3
    
    def test_empty_metadata(self, tmp_path):
        """Test handling of empty metadata."""
        metadata_path = tmp_path / "empty_metadata.json"
        empty_metadata = {}
        
        actual_path = save_metadata(empty_metadata, str(metadata_path))
        loaded_data = load_metadata(actual_path)
        
        assert loaded_data == empty_metadata
    
    def test_large_metadata(self, tmp_path):
        """Test handling of large metadata."""
        metadata_path = tmp_path / "large_metadata.json"
        large_metadata = {
            'large_list': list(range(1000)),
            'large_dict': {f'key_{i}': f'value_{i}' for i in range(100)},
            'nested': {
                'level1': {
                    'level2': {
                        'level3': {
                            'data': list(range(500))
                        }
                    }
                }
            }
        }
        
        actual_path = save_metadata(large_metadata, str(metadata_path))
        loaded_data = load_metadata(actual_path)
        
        assert loaded_data == large_metadata
        assert len(loaded_data['large_list']) == 1000
        assert len(loaded_data['large_dict']) == 100
        assert len(loaded_data['nested']['level1']['level2']['level3']['data']) == 500
