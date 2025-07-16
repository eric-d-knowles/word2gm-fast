"""
Unit tests for the data preparation pipeline.

Tests the complete pipeline functionality including:
- Single year processing
- Year range processing with parallel execution
- Year parsing and validation
- Error handling and edge cases
- Integration scenarios
"""

import pytest
import tempfile
import shutil
import os
import time
from unittest.mock import patch, MagicMock
from src.word2gm_fast.dataprep.pipeline import (
    process_single_year,
    process_year_range,
    parse_years,
    get_available_years,
    run_pipeline
)


@pytest.fixture
def temp_corpus_dir():
    """Create temporary directory with sample corpus files."""
    temp_dir = tempfile.mkdtemp()
    
    # Create sample corpus files - need to be larger to generate triplets
    sample_content = {
        "1680.txt": ("the quick brown fox jumps over the lazy dog " * 50 + 
                    "the cat sat on the mat " * 50 + 
                    "machine learning is powerful " * 50),
        "1681.txt": ("hello world this is a test corpus " * 50 + 
                    "with multiple lines of text " * 50 + 
                    "natural language processing " * 50),
        "1682.txt": ("machine learning natural language processing " * 50 + 
                    "word embeddings and neural networks " * 50 + 
                    "deep learning transformer models " * 50),
        "1683.txt": "small corpus file with few words",  # Very small file
        "invalid.txt": "not a year file",  # Invalid name
    }
    
    for filename, content in sample_content.items():
        with open(os.path.join(temp_dir, filename), 'w') as f:
            f.write(content)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def empty_corpus_dir():
    """Create empty temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestProcessSingleYear:
    """Test the process_single_year function."""
    
    def test_successful_processing(self, temp_corpus_dir):
        """Test successful processing of a single year."""
        result = process_single_year(temp_corpus_dir, "1680", compress=False)
        
        # Should return success dictionary
        assert "error" not in result
        assert result["year"] == "1680"
        assert "vocab_size" in result
        assert "triplet_count" in result
        assert "duration" in result
        assert "output_dir" in result
        
        # Check that artifacts were created
        output_dir = result["output_dir"]
        assert os.path.exists(output_dir)
        assert os.path.exists(os.path.join(output_dir, "triplets.tfrecord"))
        assert os.path.exists(os.path.join(output_dir, "vocab.tfrecord"))
        
        # Basic sanity checks
        assert result["vocab_size"] > 0
        assert result["triplet_count"] >= 0  # Allow 0 triplets for small corpus
        assert result["duration"] > 0
    
    def test_compressed_output(self, temp_corpus_dir):
        """Test processing with compression enabled."""
        result = process_single_year(temp_corpus_dir, "1681", compress=True)
        
        assert "error" not in result
        output_dir = result["output_dir"]
        
        # Should create compressed files
        assert os.path.exists(os.path.join(output_dir, "triplets.tfrecord.gz"))
        assert os.path.exists(os.path.join(output_dir, "vocab.tfrecord.gz"))
    
    def test_nonexistent_file(self, temp_corpus_dir):
        """Test processing non-existent corpus file."""
        result = process_single_year(temp_corpus_dir, "9999", compress=False)
        
        assert "error" in result
        assert "not found" in result["error"].lower()
    
    def test_invalid_corpus_directory(self):
        """Test processing with invalid corpus directory."""
        result = process_single_year("/nonexistent/directory", "1680", compress=False)
        
        assert "error" in result
    
    def test_small_corpus_file(self, temp_corpus_dir):
        """Test processing very small corpus file."""
        result = process_single_year(temp_corpus_dir, "1683", compress=False)
        
        # Should still work even with small file
        if "error" not in result:
            assert result["vocab_size"] > 0
            assert result["triplet_count"] >= 0  # Might be 0 for very small corpus


class TestProcessYearRange:
    """Test the process_year_range function."""
    
    def test_single_year_range(self, temp_corpus_dir):
        """Test processing single year via range function."""
        results = process_year_range(temp_corpus_dir, "1680", compress=False, show_progress=False)
        
        assert "1680" in results
        assert "error" not in results["1680"]
        assert results["1680"]["year"] == "1680"
    
    def test_multiple_years_sequential(self, temp_corpus_dir):
        """Test processing multiple years sequentially."""
        results = process_year_range(
            temp_corpus_dir, "1680,1681", 
            compress=False, max_workers=1, show_progress=False
        )
        
        assert len(results) == 2
        assert "1680" in results
        assert "1681" in results
        
        for year, result in results.items():
            assert "error" not in result
            assert result["year"] == year
    
    def test_year_range_dash_notation(self, temp_corpus_dir):
        """Test processing year range with dash notation."""
        results = process_year_range(
            temp_corpus_dir, "1680-1682", 
            compress=False, max_workers=1, show_progress=False
        )
        
        # Should process 1680, 1681, 1682
        assert len(results) == 3
        for year in ["1680", "1681", "1682"]:
            assert year in results
            assert "error" not in results[year]
    
    def test_parallel_processing(self, temp_corpus_dir):
        """Test parallel processing of multiple years."""
        results = process_year_range(
            temp_corpus_dir, "1680,1681", 
            compress=False, max_workers=2, show_progress=False
        )
        
        assert len(results) == 2
        for year, result in results.items():
            assert "error" not in result
    
    def test_missing_files_filtered(self, temp_corpus_dir):
        """Test that missing files are filtered out."""
        results = process_year_range(
            temp_corpus_dir, "1680,9999,1681", 
            compress=False, show_progress=False
        )
        
        # Should only process existing files
        assert len(results) == 2
        assert "1680" in results
        assert "1681" in results
        assert "9999" not in results
    
    def test_empty_directory(self, empty_corpus_dir):
        """Test processing with empty corpus directory."""
        results = process_year_range(
            empty_corpus_dir, "1680-1690", 
            compress=False, show_progress=False
        )
        
        assert len(results) == 0


class TestParseYears:
    """Test the parse_years function."""
    
    def test_single_year(self):
        """Test parsing single year."""
        years = parse_years("1680")
        assert years == ["1680"]
    
    def test_comma_separated_years(self):
        """Test parsing comma-separated years."""
        years = parse_years("1680,1681,1682")
        assert years == ["1680", "1681", "1682"]
    
    def test_year_range_dash(self):
        """Test parsing year range with dash."""
        years = parse_years("1680-1683")
        assert years == ["1680", "1681", "1682", "1683"]
    
    def test_mixed_format(self):
        """Test parsing mixed comma and dash format."""
        years = parse_years("1680,1685-1687,1690")
        expected = ["1680", "1685", "1686", "1687", "1690"]
        assert years == expected
    
    def test_whitespace_handling(self):
        """Test parsing with whitespace."""
        years = parse_years(" 1680 , 1681 - 1683 , 1690 ")
        expected = ["1680", "1681", "1682", "1683", "1690"]
        assert years == expected


class TestGetAvailableYears:
    """Test the get_available_years function."""
    
    def test_get_available_years(self, temp_corpus_dir):
        """Test getting available years from corpus directory."""
        years = get_available_years(temp_corpus_dir)
        
        # Should find valid year files, sorted
        expected = ["1680", "1681", "1682", "1683"]
        assert years == expected
    
    def test_empty_directory(self, empty_corpus_dir):
        """Test getting years from empty directory."""
        years = get_available_years(empty_corpus_dir)
        assert years == []
    
    def test_nonexistent_directory(self):
        """Test getting years from non-existent directory."""
        years = get_available_years("/nonexistent/directory")
        assert years == []


class TestRunPipeline:
    """Test the run_pipeline function (simple API)."""
    
    def test_run_pipeline_basic(self, temp_corpus_dir):
        """Test basic pipeline execution."""
        results = run_pipeline(temp_corpus_dir, "1680", compress=False, show_progress=False)
        
        assert "1680" in results
        assert "error" not in results["1680"]
    
    def test_run_pipeline_with_kwargs(self, temp_corpus_dir):
        """Test pipeline with additional keyword arguments."""
        results = run_pipeline(
            temp_corpus_dir, "1680,1681", 
            compress=True, max_workers=1, show_progress=False
        )
        
        assert len(results) == 2
        for year, result in results.items():
            assert "error" not in result


class TestIntegrationScenarios:
    """Integration tests for complete pipeline scenarios."""
    
    def test_full_pipeline_workflow(self, temp_corpus_dir):
        """Test complete pipeline workflow from start to finish."""
        # 1. Check available years
        available = get_available_years(temp_corpus_dir)
        assert len(available) > 0
        
        # 2. Parse year range
        years_to_process = "1680-1681"
        parsed_years = parse_years(years_to_process)
        assert len(parsed_years) == 2
        
        # 3. Process years
        results = process_year_range(
            temp_corpus_dir, years_to_process,
            compress=False, show_progress=False
        )
        
        # 4. Verify results
        assert len(results) == 2
        for year in parsed_years:
            assert year in results
            result = results[year]
            assert "error" not in result
            assert result["vocab_size"] > 0
            assert result["triplet_count"] >= 0  # Allow 0 triplets for small corpus
            
            # Verify artifacts exist
            output_dir = result["output_dir"]
            assert os.path.exists(os.path.join(output_dir, "triplets.tfrecord"))
            assert os.path.exists(os.path.join(output_dir, "vocab.tfrecord"))
    
    def test_error_resilience(self, temp_corpus_dir):
        """Test pipeline resilience to errors."""
        # Include both valid and invalid years
        mixed_years = "1680,9999,1681,8888"
        
        results = process_year_range(
            temp_corpus_dir, mixed_years,
            compress=False, show_progress=False
        )
        
        # Should process valid years and skip invalid ones
        assert "1680" in results
        assert "1681" in results
        assert "9999" not in results
        assert "8888" not in results
        
        # Valid years should be successful
        for year in ["1680", "1681"]:
            assert "error" not in results[year]
    
    def test_performance_metrics(self, temp_corpus_dir):
        """Test that performance metrics are captured."""
        results = process_year_range(
            temp_corpus_dir, "1680",
            compress=False, show_progress=False
        )
        
        result = results["1680"]
        assert "duration" in result
        assert result["duration"] > 0
        assert "triplet_count" in result
        assert "vocab_size" in result
        
        # Metrics should be reasonable
        assert 0 < result["duration"] < 60  # Should complete within a minute
        assert result["vocab_size"] > 0
        assert result["triplet_count"] >= 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_character_corpus(self, temp_corpus_dir):
        """Test with minimal corpus content."""
        # Create minimal corpus file
        minimal_file = os.path.join(temp_corpus_dir, "1999.txt")
        with open(minimal_file, 'w') as f:
            f.write("a")
        
        try:
            result = process_single_year(temp_corpus_dir, "1999", compress=False)
            
            # Should handle gracefully
            if "error" not in result:
                assert result["vocab_size"] >= 1  # At least UNK token
        finally:
            if os.path.exists(minimal_file):
                os.remove(minimal_file)
    
    def test_very_large_year_range(self, temp_corpus_dir):
        """Test parsing very large year range."""
        # This should parse correctly but most years won't exist
        years = parse_years("1680-1700")
        assert len(years) == 21
        
        # Processing should only handle existing files
        results = process_year_range(
            temp_corpus_dir, "1680-1700",
            compress=False, show_progress=False
        )
        
        # Should only process existing years
        assert len(results) <= 4  # Only 4 files exist
        
        # All processed years should be successful
        for year, result in results.items():
            assert "error" not in result
