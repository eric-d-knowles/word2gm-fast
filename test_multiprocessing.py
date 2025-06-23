#!/usr/bin/env python3
"""
Test script for multiprocessing functionality in the Word2GM pipeline.
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Change to project directory  
sys.path.insert(0, '/scratch/edk202/word2gm-fast')

# Import pipeline modules
from src.word2gm_fast.dataprep.pipeline import batch_prepare_training_data
from src.word2gm_fast.utils import import_tensorflow_silently

# Silence TensorFlow
tf = import_tensorflow_silently(deterministic=False)

def create_test_corpus_files(temp_dir: str, years: list) -> str:
    """Create small test corpus files for testing multiprocessing."""
    for year in years:
        corpus_file = os.path.join(temp_dir, f"{year}.txt")
        with open(corpus_file, 'w') as f:
            # Write simple test data
            for i in range(1000):  # Small corpus for fast testing
                f.write(f"the quick brown fox jumps over the lazy dog number {i}\n")
                f.write(f"hello world this is a test sentence for year {year}\n")
                f.write(f"machine learning natural language processing tensorflow python\n")
    return temp_dir

def test_multiprocessing():
    """Test the multiprocessing functionality."""
    print("🧪 Testing multiprocessing functionality...")
    print("=" * 60)
    
    # Create temporary directory and test files
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Creating test corpus files in: {temp_dir}")
        
        test_years = ["2000", "2001", "2002"]
        create_test_corpus_files(temp_dir, test_years)
        
        print(f"✅ Created {len(test_years)} test corpus files")
        print()
        
        # Test sequential processing
        print("🔄 Testing sequential processing...")
        start_time = time.perf_counter()
        
        results_sequential = batch_prepare_training_data(
            years=test_years,
            corpus_dir=temp_dir,
            compress=False,
            show_progress=True,
            show_summary=False,
            use_multiprocessing=False
        )
        
        sequential_time = time.perf_counter() - start_time
        print(f"⏱️  Sequential time: {sequential_time:.2f}s")
        print()
        
        # Test parallel processing
        print("🚀 Testing parallel processing...")
        start_time = time.perf_counter()
        
        results_parallel = batch_prepare_training_data(
            years=test_years,
            corpus_dir=temp_dir,
            compress=False,
            show_progress=True,
            show_summary=False,
            max_workers=2,
            use_multiprocessing=True
        )
        
        parallel_time = time.perf_counter() - start_time
        print(f"⏱️  Parallel time: {parallel_time:.2f}s")
        print()
        
        # Compare results
        print("📊 COMPARISON RESULTS")
        print("=" * 40)
        
        successful_sequential = [year for year in test_years if year in results_sequential and 'error' not in results_sequential[year]]
        successful_parallel = [year for year in test_years if year in results_parallel and 'error' not in results_parallel[year]]
        
        print(f"Sequential successful: {len(successful_sequential)}/{len(test_years)}")
        print(f"Parallel successful:   {len(successful_parallel)}/{len(test_years)}")
        
        if len(successful_sequential) == len(successful_parallel) == len(test_years):
            speedup = sequential_time / parallel_time if parallel_time > 0 else 1
            print(f"Speedup: {speedup:.2f}x")
            
            # Compare triplet counts
            seq_triplets = sum(results_sequential[year]['triplet_count'] for year in successful_sequential)
            par_triplets = sum(results_parallel[year]['triplet_count'] for year in successful_parallel)
            print(f"Sequential triplets: {seq_triplets:,}")
            print(f"Parallel triplets:   {par_triplets:,}")
            
            if seq_triplets == par_triplets:
                print("✅ Triplet counts match - multiprocessing working correctly!")
            else:
                print("❌ Triplet counts differ - potential issue!")
                
        else:
            print("❌ Different success rates - potential issue!")
        
        print()
        print("🎉 Multiprocessing test completed!")

if __name__ == "__main__":
    test_multiprocessing()
