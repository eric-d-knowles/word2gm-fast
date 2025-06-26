"""Unit tests for filter_corpus 5-gram filtering logic (pytest style).

This script tests all valid and invalid 5-gram combinations for the
phrase 'the quick brown fox jumps' with any subset of words replaced by
'UNK'. It ensures correct filtering and summary statistics.
"""
import pytest
import tensorflow as tf
from src.word2gm_fast.dataprep.corpus_to_dataset import make_dataset

@pytest.fixture
def all_5gram_lines():
    lines = [
        # Valid cases (at least one context is non-UNK, center is not UNK)
        "the quick brown fox jumps\n",
        "UNK quick brown fox jumps\n",
        "the UNK brown fox jumps\n",
        "the quick brown UNK jumps\n",
        "the quick brown fox UNK\n",
        "UNK UNK brown fox jumps\n",
        "UNK quick brown UNK jumps\n",
        "UNK quick brown fox UNK\n",
        "the UNK brown UNK jumps\n",
        "the UNK brown fox UNK\n",
        "UNK UNK brown UNK jumps\n",
        "UNK UNK brown fox UNK\n",
        "the quick brown UNK UNK\n",
        "UNK quick brown UNK UNK\n",
        "the UNK brown UNK UNK\n",
        # Invalid cases (center is UNK or all context is UNK)
        "the quick UNK fox jumps\n",
        "the UNK UNK fox jumps\n",
        "UNK quick UNK fox jumps\n",
        "the quick UNK UNK jumps\n",
        "the quick UNK fox UNK\n",
        "the quick UNK UNK UNK\n",
        "UNK UNK UNK fox jumps\n",
        "UNK quick UNK UNK jumps\n",
        "UNK quick UNK fox UNK\n",
        "UNK quick UNK UNK UNK\n",
        "the UNK UNK UNK jumps\n",
        "the UNK UNK fox UNK\n",
        "the UNK UNK UNK UNK\n",
        "the quick UNK UNK UNK\n",
        "UNK UNK UNK UNK jumps\n",
        "UNK UNK UNK fox UNK\n",
        "UNK UNK brown UNK UNK\n",
    ]
    valid_lines = [
        "the quick brown fox jumps\n",
        "UNK quick brown fox jumps\n",
        "the UNK brown fox jumps\n",
        "the quick brown UNK jumps\n",
        "the quick brown fox UNK\n",
        "UNK UNK brown fox jumps\n",
        "UNK quick brown UNK jumps\n",
        "UNK quick brown fox UNK\n",
        "the UNK brown UNK jumps\n",
        "the UNK brown fox UNK\n",
        "UNK UNK brown UNK jumps\n",
        "UNK UNK brown fox UNK\n",
        "the quick brown UNK UNK\n",
        "UNK quick brown UNK UNK\n",
        "the UNK brown UNK UNK\n",
    ]
    return lines, valid_lines

def test_corpus_to_dataset(tmp_path, all_5gram_lines):
    """Test that only valid 5-grams are retained and summary is correct."""
    lines, valid_lines = all_5gram_lines
    temp_file = tmp_path / "test_5grams.txt"
    temp_file.write_text("".join(lines))
    dataset, summary = make_dataset(str(temp_file), show_summary=True)
    result_lines = [line.numpy().decode("utf-8").strip() for line in dataset]
    expected_lines = [line.strip() for line in valid_lines]
    assert set(result_lines) == set(expected_lines)
    assert isinstance(summary, dict)
    assert summary["retained"] == 15
    assert summary["rejected"] == 17
    assert summary["total"] == 32
