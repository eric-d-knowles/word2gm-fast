import unittest
import tempfile
import tensorflow as tf
from src.word2gm_fast.dataprep.load_and_filter_corpus import load_and_filter_corpus

class TestLoadAndFilterCorpus(unittest.TestCase):
    def setUp(self):
        # Create a temporary file with test 5-gram lines
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.lines = [
            "the quick brown fox jumps\n",      # valid
            "UNK quick brown fox jumps\n",      # invalid (center is UNK)
            "the quick brown UNK jumps\n",      # valid (context has at least one non-UNK)
            "the quick brown fox UNK\n",        # valid (context has at least one non-UNK)
            "UNK UNK UNK UNK UNK\n"             # invalid (all UNK)
        ]
        self.valid_lines = [
            "the quick brown fox jumps",
            "the quick brown UNK jumps",
            "the quick brown fox UNK"
        ]
        self.temp_file.writelines(self.lines)
        self.temp_file.flush()

    def tearDown(self):
        self.temp_file.close()

    def test_load_and_filter(self):
        dataset, summary = load_and_filter_corpus(self.temp_file.name)
        result_lines = [line.numpy().decode("utf-8") for line in dataset]
        self.assertCountEqual(result_lines, self.valid_lines)
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["retained"], 3)
        self.assertEqual(summary["rejected"], 2)
        self.assertEqual(summary["total"], 5)

if __name__ == '__main__':
    unittest.main()
