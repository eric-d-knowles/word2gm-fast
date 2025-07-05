"""
Unit tests for the data preparation pipeline (pytest style).

Tests the complete pipeline functionality including:
- Single file processing
- Batch processing
- Multiprocessing
- Error handling
- Year range parsing
- Resource detection
"""

import pytest
import tempfile
import shutil
import os
import time
from unittest.mock import patch, MagicMock
from src.word2gm_fast.dataprep.pipeline import (
    prepare_training_data,
    batch_prepare_training_data,
    get_corpus_years,
    parse_year_range,
    detect_cluster_resources,
    get_safe_worker_count,
    _process_single_year
)


@pytest.fixture
def pipeline_test_setup(tmp_path):
    """Create test corpus files and directory structure."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    
    # Create test corpus files for different years
    test_years = ["2018", "2019", "2020"]
    test_content = "the quick brown fox jumps over the lazy dog\nthe fox is quick and brown\n"
    
    for year in test_years:
        corpus_file = corpus_dir / f"{year}.txt"
        with open(corpus_file, 'w') as f:
            f.write(test_content * 100)  # Make files reasonably sized
    
    # Create some non-year files to test filtering
    (corpus_dir / "readme.txt").write_text("Not a year file")
    (corpus_dir / "202.txt").write_text("Invalid year")
    
    return {
        'corpus_dir': str(corpus_dir),
        'test_years': test_years,
        'test_content': test_content
    }


def test_get_corpus_years(pipeline_test_setup):
    """Test automatic discovery of corpus years."""
    corpus_dir = pipeline_test_setup['corpus_dir']
    expected_years = pipeline_test_setup['test_years']
    
    discovered_years = get_corpus_years(corpus_dir)
    assert discovered_years == sorted(expected_years)


def test_get_corpus_years_empty_directory(tmp_path):
    """Test get_corpus_years with empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    years = get_corpus_years(str(empty_dir))
    assert years == []


def test_get_corpus_years_nonexistent_directory():
    """Test get_corpus_years with non-existent directory."""
    years = get_corpus_years("/nonexistent/directory")
    assert years == []


def test_parse_year_range():
    """Test year range parsing functionality."""
    # Single year
    assert parse_year_range("2019") == ["2019"]
    
    # Simple range
    assert parse_year_range("2018-2020") == ["2018", "2019", "2020"]
    
    # Complex range with commas
    assert parse_year_range("2017,2019-2021,2023") == ["2017", "2019", "2020", "2021", "2023"]
    
    # Single range
    assert parse_year_range("1400-1403") == ["1400", "1401", "1402", "1403"]


def test_parse_year_range_invalid():
    """Test year range parsing with invalid inputs."""
    with pytest.raises(ValueError):
        parse_year_range("2020-2018")  # Invalid range
    
    with pytest.raises(ValueError):
        parse_year_range("not_a_year")  # Invalid year


@patch('src.word2gm_fast.dataprep.pipeline.make_dataset')
@patch('word2gm_fast.dataprep.index_vocab.make_vocab')  
@patch('word2gm_fast.dataprep.dataset_to_triplets.build_skipgram_triplets')
@patch('word2gm_fast.io.triplets.write_triplets_to_tfrecord')
@patch('word2gm_fast.io.vocab.write_vocab_to_tfrecord')
def test_prepare_training_data_success(mock_write_vocab, mock_write_triplets, mock_build_triplets,
                                     mock_make_vocab, mock_make_dataset, pipeline_test_setup):
    """Test successful single file processing."""
    # Mock the pipeline components
    mock_dataset = MagicMock()
    mock_make_dataset.return_value = (mock_dataset, None)
    
    mock_vocab_table = MagicMock()
    mock_vocab_table.export.return_value = (MagicMock(), MagicMock())
    mock_vocab_table.export.return_value[0].numpy.return_value = ["UNK", "the", "fox"]
    mock_vocab_table.size.return_value = MagicMock()
    mock_vocab_table.size.return_value.numpy.return_value = 3
    
    mock_make_vocab.return_value = (mock_vocab_table, ["UNK", "the", "fox"], {"UNK": 100, "the": 50, "fox": 25})
    
    mock_triplets_ds = MagicMock()
    mock_build_triplets.return_value = mock_triplets_ds
    
    mock_write_triplets.return_value = 1000  # Mock triplet count
    
    corpus_dir = pipeline_test_setup['corpus_dir']
    
    # Test the function
    output_dir, summary = prepare_training_data(
        corpus_file="2019.txt",
        corpus_dir=corpus_dir,
        output_subdir="test_artifacts",
        show_progress=False,
        show_summary=False
    )
    
    # Verify calls
    mock_make_dataset.assert_called_once()
    mock_make_vocab.assert_called_once()
    mock_build_triplets.assert_called_once()
    mock_write_vocab.assert_called_once()
    mock_write_triplets.assert_called_once()
    
    # Check output
    assert output_dir.endswith("test_artifacts")
    assert summary['corpus_file'] == "2019.txt"
    assert summary['vocab_size'] == 3
    assert summary['triplet_count'] == 1000


def test_prepare_training_data_file_not_found(pipeline_test_setup):
    """Test error handling when corpus file doesn't exist."""
    corpus_dir = pipeline_test_setup['corpus_dir']
    
    with pytest.raises(FileNotFoundError):
        prepare_training_data(
            corpus_file="nonexistent.txt",
            corpus_dir=corpus_dir
        )


def test_prepare_training_data_invalid_directory():
    """Test error handling when corpus directory doesn't exist."""
    with pytest.raises(FileNotFoundError):
        prepare_training_data(
            corpus_file="test.txt",
            corpus_dir="/nonexistent/directory"
        )


def test_batch_prepare_training_data_auto_discovery(pipeline_test_setup):
    """Test batch processing with automatic year discovery."""
    corpus_dir = pipeline_test_setup['corpus_dir']
    expected_years = pipeline_test_setup['test_years']
    
    with patch('src.word2gm_fast.dataprep.pipeline.prepare_training_data') as mock_prepare:
        mock_prepare.return_value = ("output_dir", {"triplet_count": 1000, "vocab_size": 100, "total_duration_s": 10.0})
        
        results = batch_prepare_training_data(
            corpus_dir=corpus_dir,
            use_multiprocessing=False,  # Use sequential for easier testing
            show_progress=False,
            show_summary=False
        )
        
        # Should have processed all discovered years
        assert len(results) == len(expected_years)
        for year in expected_years:
            assert year in results
            assert 'error' not in results[year]


def test_batch_prepare_training_data_specific_years(pipeline_test_setup):
    """Test batch processing with specific years."""
    corpus_dir = pipeline_test_setup['corpus_dir']
    
    with patch('src.word2gm_fast.dataprep.pipeline.prepare_training_data') as mock_prepare:
        mock_prepare.return_value = ("output_dir", {"triplet_count": 1000, "vocab_size": 100, "total_duration_s": 10.0})
        
        results = batch_prepare_training_data(
            corpus_dir=corpus_dir,
            years=["2019", "2020"],
            use_multiprocessing=False,
            show_progress=False,
            show_summary=False
        )
        
        assert len(results) == 2
        assert "2019" in results
        assert "2020" in results
        assert "2018" not in results


def test_batch_prepare_training_data_year_range(pipeline_test_setup):
    """Test batch processing with year range."""
    corpus_dir = pipeline_test_setup['corpus_dir']
    
    with patch('src.word2gm_fast.dataprep.pipeline.prepare_training_data') as mock_prepare:
        mock_prepare.return_value = ("output_dir", {"triplet_count": 1000, "vocab_size": 100, "total_duration_s": 10.0})
        
        results = batch_prepare_training_data(
            corpus_dir=corpus_dir,
            year_range="2019-2020",
            use_multiprocessing=False,
            show_progress=False,
            show_summary=False
        )
        
        assert len(results) == 2
        assert "2019" in results
        assert "2020" in results
        assert "2018" not in results


def test_batch_prepare_training_data_conflicting_params(pipeline_test_setup):
    """Test error when both years and year_range are specified."""
    corpus_dir = pipeline_test_setup['corpus_dir']
    
    with pytest.raises(ValueError):
        batch_prepare_training_data(
            corpus_dir=corpus_dir,
            years=["2019"],
            year_range="2019-2020"
        )


def test_batch_prepare_training_data_missing_files(pipeline_test_setup):
    """Test handling of missing corpus files."""
    corpus_dir = pipeline_test_setup['corpus_dir']
    
    with patch('src.word2gm_fast.dataprep.pipeline.prepare_training_data') as mock_prepare:
        mock_prepare.return_value = ("output_dir", {"triplet_count": 1000, "vocab_size": 100, "total_duration_s": 10.0})
        
        results = batch_prepare_training_data(
            corpus_dir=corpus_dir,
            years=["2019", "2099"],  # 2099 doesn't exist
            use_multiprocessing=False,
            show_progress=False,
            show_summary=False
        )
        
        # Should only process existing files
        assert len(results) == 1
        assert "2019" in results
        assert "2099" not in results


def test_detect_cluster_resources():
    """Test cluster resource detection."""
    # Test without environment variables
    resources = detect_cluster_resources()
    
    assert 'detected_cpus' in resources
    assert 'allocated_cpus' in resources
    assert 'scheduler' in resources
    assert 'recommended_workers' in resources
    assert resources['detected_cpus'] > 0
    assert resources['recommended_workers'] > 0


@patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '8'})
def test_detect_cluster_resources_slurm():
    """Test cluster resource detection with SLURM."""
    resources = detect_cluster_resources()
    
    assert resources['allocated_cpus'] == 8
    assert resources['scheduler'] == 'SLURM'
    assert resources['recommended_workers'] == 8


def test_get_safe_worker_count():
    """Test safe worker count calculation."""
    # Test with no limit
    workers = get_safe_worker_count()
    assert workers > 0
    
    # Test with limit
    workers = get_safe_worker_count(max_workers=4)
    assert workers <= 4
    assert workers > 0


def test_process_single_year_helper(pipeline_test_setup):
    """Test the multiprocessing helper function."""
    corpus_dir = pipeline_test_setup['corpus_dir']
    
    with patch('src.word2gm_fast.dataprep.pipeline.prepare_training_data') as mock_prepare:
        mock_prepare.return_value = ("output_dir", {"triplet_count": 1000, "vocab_size": 100, "total_duration_s": 10.0})
        
        args = ("2019", corpus_dir, True, False, 1e-5)
        year, success, result = _process_single_year(args)
        
        assert year == "2019"
        assert success is True
        assert isinstance(result, dict)
        assert result['triplet_count'] == 1000


def test_process_single_year_helper_error(pipeline_test_setup):
    """Test the multiprocessing helper function with error."""
    corpus_dir = pipeline_test_setup['corpus_dir']
    
    with patch('src.word2gm_fast.dataprep.pipeline.prepare_training_data') as mock_prepare:
        mock_prepare.side_effect = Exception("Test error")
        
        args = ("2019", corpus_dir, True, False, 1e-5)
        year, success, result = _process_single_year(args)
        
        assert year == "2019"
        assert success is False
        assert "Test error" in result
