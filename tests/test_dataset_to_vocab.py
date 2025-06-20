"""
Unit tests for dataset_to_vocab.py
"""
import unittest
import tensorflow as tf
import numpy as np
from src.word2gm_fast.dataprep.dataset_to_vocab import make_vocab, build_vocab_table

class TestDatasetToVocab(unittest.TestCase):
    def setUp(self):
        # Example dataset with more n-gram variety and some repeated/UNK tokens
        self.lines = [
            b"the quick brown fox jumps",
            b"the quick brown UNK jumps",
            b"UNK quick brown fox jumps",
            b"the quick brown fox jumps",
            b"lazy dog UNK jumps over",
            b"over the lazy dog jumps",
            b"brown fox jumps over the",
            b"quick brown fox jumps over",
            b"fox jumps over the lazy",
            b"jumps over the lazy dog",
            b"UNK dog jumps over the",
            b"the dog jumps over UNK",
            b"dog jumps over the quick",
            b"jumps over the quick brown",
            b"over the quick brown fox",
        ]
        self.dataset = tf.data.Dataset.from_tensor_slices(self.lines)

    def test_make_vocab_table(self):
        table = make_vocab(self.dataset)
        # Check that known tokens map to correct indices
        for token in ["UNK", "the", "quick", "brown", "fox", "jumps", "lazy", "dog", "over"]:
            idx = table.lookup(tf.constant(token)).numpy()
            self.assertIsInstance(idx, (int, np.integer))
        # UNK must be index 0
        self.assertEqual(table.lookup(tf.constant("UNK")).numpy(), 0)

    def test_vocab_table_contents(self):
        # Build vocab list and table separately for direct inspection
        vocab_set = set()
        for line in self.lines:
            tokens = line.decode("utf-8").strip().split()
            vocab_set.update(tokens)
        vocab = ["UNK"] + sorted(tok for tok in vocab_set if tok != "UNK")
        table = build_vocab_table(vocab)
        # Check that all tokens are present and mapped correctly
        for i, token in enumerate(vocab):
            self.assertEqual(table.lookup(tf.constant(token)).numpy(), i)
        # Check that an OOV token maps to 0 (UNK)
        self.assertEqual(table.lookup(tf.constant("notinthevocab")).numpy(), 0)

if __name__ == "__main__":
    unittest.main()
