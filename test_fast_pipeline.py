#!/usr/bin/env python3
"""
Test script to compare standard vs fast mode pipeline performance.

This script demonstrates the time savings from skipping dataset manifestation
by using the optimized pipeline that writes directly to TFRecord.
"""

import sys
import os
sys.path.insert(0, '/scratch/edk202/word2gm-fast/src')

from word2gm_fast.utils import import_tensorflow_silently
tf = import_tensorflow_silently(deterministic=False)

from word2gm_fast.dataprep.pipeline import (
    prepare_training_data, 
    prepare_training_data_fast,
    estimate_fast_mode_savings
)

def test_pipeline_performance():
    """Compare standard vs fast mode performance on a small test corpus."""
    
    # Create test corpus
    test_corpus_dir = "/tmp/test_pipeline"
    os.makedirs(test_corpus_dir, exist_ok=True)
    
    test_corpus_file = "test_corpus.txt"
    test_corpus_path = os.path.join(test_corpus_dir, test_corpus_file)
    
    # Generate test data (5-gram format)
    print("🔧 Creating test corpus...")
    test_lines = []
    for i in range(1000):  # Small test corpus
        words = [f"word{j}" for j in range(i % 10, i % 10 + 5)]
        test_lines.append(" ".join(words))
    
    with open(test_corpus_path, 'w') as f:
        for line in test_lines:
            f.write(line + '\n')
    
    print(f"✅ Created test corpus: {len(test_lines)} lines")
    print()
    
    # Run analysis
    print("📊 Running corpus analysis...")
    try:
        analysis = estimate_fast_mode_savings(test_corpus_file, test_corpus_dir)
        print()
    except Exception as e:
        print(f"Analysis failed: {e}")
        print()
    
    # Test standard mode
    print("🐌 Testing STANDARD mode...")
    print("-" * 40)
    try:
        output_dir1, summary1 = prepare_training_data(
            corpus_file=test_corpus_file,
            corpus_dir=test_corpus_dir,
            output_subdir="test_standard",
            show_progress=True,
            show_summary=True
        )
        print()
    except Exception as e:
        print(f"Standard mode failed: {e}")
        return
    
    # Test fast mode  
    print("⚡ Testing FAST mode...")
    print("-" * 40)
    try:
        output_dir2, summary2 = prepare_training_data_fast(
            corpus_file=test_corpus_file,
            corpus_dir=test_corpus_dir,
            output_subdir="test_fast",
            show_progress=True,
            show_summary=True
        )
        print()
    except Exception as e:
        print(f"Fast mode failed: {e}")
        return
    
    # Compare results
    print("🔍 COMPARISON RESULTS")
    print("=" * 50)
    print(f"Standard mode time:  {summary1['total_duration_s']:.3f}s")
    print(f"Fast mode time:      {summary2['total_duration_s']:.3f}s")
    
    if summary1['total_duration_s'] > 0:
        speedup = summary1['total_duration_s'] / summary2['total_duration_s']
        time_saved = summary1['total_duration_s'] - summary2['total_duration_s']
        percent_saved = (time_saved / summary1['total_duration_s']) * 100
        
        print(f"Time saved:          {time_saved:.3f}s")
        print(f"Speedup:             {speedup:.2f}x")
        print(f"Improvement:         {percent_saved:.1f}%")
    
    # Verify outputs match
    print()
    print("🔍 VERIFICATION")
    print("=" * 50)
    print(f"Vocab size match:    {summary1['vocab_size'] == summary2['vocab_size']}")
    print(f"Triplet count match: {summary1['triplet_count'] == summary2['triplet_count']}")
    
    if summary1['vocab_size'] == summary2['vocab_size'] and summary1['triplet_count'] == summary2['triplet_count']:
        print("✅ Both modes produce identical results!")
    else:
        print("❌ Results differ - investigation needed")
    
    # Cleanup
    import shutil
    try:
        shutil.rmtree(test_corpus_dir)
        print()
        print("🧹 Cleanup complete")
    except:
        pass

if __name__ == "__main__":
    print("🚀 Pipeline Performance Comparison Test")
    print("=" * 50)
    print()
    
    test_pipeline_performance()
