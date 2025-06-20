"""Unit tests for filter_corpus 5-gram filtering logic.

This script tests all valid and invalid 5-gram combinations for the
phrase 'the quick brown fox jumps' with any subset of words replaced by
'UNK'. It ensures correct filtering and summary statistics.
"""
import unittest
import tempfile
import os
import tensorflow as tf
from src.word2gm_fast.dataprep.corpus_to_dataset import make_dataset

class TestCorpusToDataset(unittest.TestCase):
    """Unit tests for the make_dataset function."""

    def setUp(self) -> None:
        """Create a temporary file with all 32 possible 5-gram lines."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.lines = [
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
        self.valid_lines = [
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
        self.temp_file.writelines(self.lines)
        self.temp_file.flush()

    def tearDown(self) -> None:
        """Remove the temporary file after tests."""
        self.temp_file.close()
        os.unlink(self.temp_file.name)

    def test_corpus_to_dataset(self) -> None:
        """Test that only valid 5-grams are retained and summary is correct."""
        dataset, summary = make_dataset(
            self.temp_file.name, show_summary=True
        )
        result_lines = [line.numpy().decode("utf-8").strip() for line in dataset]
        expected_lines = [line.strip() for line in self.valid_lines]

        # Uncomment for debugging:
        # print("Expected valid lines:", set(self.valid_lines))
        # print("Actual result lines:", set(result_lines))
        # print("Missing from result:", set(self.valid_lines) - set(result_lines))
        # print("Unexpected in result:", set(result_lines) - set(self.valid_lines))

        self.assertCountEqual(result_lines, expected_lines)
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["retained"], 15)
        self.assertEqual(summary["rejected"], 17)
        self.assertEqual(summary["total"], 32)

if __name__ == '__main__':
    unittest.main(buffer=False)
