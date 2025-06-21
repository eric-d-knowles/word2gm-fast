"""
Unit tests for dataset_to_triplets.py
"""
import unittest
import tensorflow as tf
import numpy as np
from src.word2gm_fast.dataprep.dataset_to_triplets import build_skipgram_triplets
from src.word2gm_fast.dataprep.index_vocab import build_vocab_table


class TestDatasetToTriplets(unittest.TestCase):
    def setUp(self):
        """Set up test data with a small vocabulary and valid 5-gram lines."""
        # Create a small test vocabulary
        self.vocab = ["UNK", "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        self.vocab_table = build_vocab_table(self.vocab)
        
        # Test 5-gram lines (should all be valid after corpus filtering)
        self.lines = [
            b"the quick brown fox jumps",
            b"quick brown fox jumps over", 
            b"brown fox jumps over the",
            b"fox jumps over the lazy",
            b"jumps over the lazy dog",
        ]
        self.dataset = tf.data.Dataset.from_tensor_slices(self.lines)

    def test_basic_triplet_generation(self):
        """Test that triplets are generated with correct structure."""
        triplets_ds = build_skipgram_triplets(self.dataset, self.vocab_table)
        
        # Collect all triplets
        triplets = list(triplets_ds.as_numpy_iterator())
        
        # Should have multiple triplets (up to 4 per valid line)
        self.assertGreater(len(triplets), 0)
        
        # Each triplet should have 3 elements: center, positive, negative
        for triplet in triplets:
            self.assertEqual(len(triplet), 3)
            center, pos, neg = triplet
            
            # All indices should be non-negative integers
            self.assertGreaterEqual(center, 0)
            self.assertGreaterEqual(pos, 0)
            self.assertGreaterEqual(neg, 0)
            
            # Center should not be UNK (index 0)
            self.assertNotEqual(center, 0)
            
            # Negative should not be UNK (starts from index 1)
            self.assertNotEqual(neg, 0)

    def test_center_word_extraction(self):
        """Test that center words are correctly extracted (3rd token)."""
        triplets_ds = build_skipgram_triplets(self.dataset, self.vocab_table)
        triplets = list(triplets_ds.as_numpy_iterator())
        
        # Get expected center words from our test lines
        expected_centers = []
        for line in self.lines:
            tokens = line.decode("utf-8").split()
            center_token = tokens[2]  # 3rd token (index 2)
            center_id = self.vocab.index(center_token)
            expected_centers.append(center_id)
        
        # Collect actual center words from triplets
        actual_centers = set(triplet[0] for triplet in triplets)
        expected_centers_set = set(expected_centers)
        
        # All actual centers should be from our expected set
        self.assertTrue(actual_centers.issubset(expected_centers_set))

    def test_context_word_extraction(self):
        """Test that positive words come from context positions."""
        triplets_ds = build_skipgram_triplets(self.dataset, self.vocab_table)
        triplets = list(triplets_ds.as_numpy_iterator())
        
        # Get all possible context words from our test lines
        all_context_words = set()
        for line in self.lines:
            tokens = line.decode("utf-8").split()
            # Context positions are 0, 1, 3, 4 (excluding center at 2)
            context_tokens = [tokens[0], tokens[1], tokens[3], tokens[4]]
            for token in context_tokens:
                if token in self.vocab and token != "UNK":
                    all_context_words.add(self.vocab.index(token))
        
        # Collect actual positive words from triplets
        actual_positives = set(triplet[1] for triplet in triplets)
        
        # All actual positives should be from our context words
        self.assertTrue(actual_positives.issubset(all_context_words))

    def test_no_unk_centers(self):
        """Test that UNK tokens are never used as center words."""
        # Create a dataset with UNK in center position
        unk_lines = [
            b"the quick UNK fox jumps",  # UNK in center
            b"UNK brown fox jumps over",  # UNK not in center (should be valid)
        ]
        unk_dataset = tf.data.Dataset.from_tensor_slices(unk_lines)
        
        triplets_ds = build_skipgram_triplets(unk_dataset, self.vocab_table)
        triplets = list(triplets_ds.as_numpy_iterator())
        
        # Should have some triplets (from the second line)
        self.assertGreater(len(triplets), 0)
        
        # No triplet should have UNK (index 0) as center
        for triplet in triplets:
            center = triplet[0]
            self.assertNotEqual(center, 0, "UNK should not be used as center word")

    def test_multiple_triplets_per_line(self):
        """Test that multiple triplets are generated per valid line."""
        # Use a single line that should generate multiple triplets
        single_line = tf.data.Dataset.from_tensor_slices([b"the quick brown fox jumps"])
        
        triplets_ds = build_skipgram_triplets(single_line, self.vocab_table)
        triplets = list(triplets_ds.as_numpy_iterator())
        
        # Should generate multiple triplets (one for each valid context word)
        # Line "the quick brown fox jumps" has center "brown" and 4 context words
        # All context words are non-UNK, so should generate 4 triplets
        self.assertEqual(len(triplets), 4)
        
        # All should have the same center word ("brown" = index 3)
        brown_index = self.vocab.index("brown")
        for triplet in triplets:
            self.assertEqual(triplet[0], brown_index)

    def test_negative_sampling_range(self):
        """Test that negative samples are in the correct range."""
        triplets_ds = build_skipgram_triplets(self.dataset, self.vocab_table)
        triplets = list(triplets_ds.as_numpy_iterator())
        
        vocab_size = len(self.vocab)
        
        for triplet in triplets:
            neg = triplet[2]
            # Negative should be in range [1, vocab_size) (excluding UNK at 0)
            self.assertGreaterEqual(neg, 1)
            self.assertLess(neg, vocab_size)


if __name__ == "__main__":
    unittest.main()
