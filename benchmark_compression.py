#!/usr/bin/env python3
"""
Benchmark script to measure TFRecord compression impact on training performance.
Tests both compressed and uncompressed TFRecord files during typical training operations.
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.word2gm_fast.utils import import_tensorflow_silently
# Import TensorFlow silently first
tf = import_tensorflow_silently(deterministic=False)
from src.word2gm_fast.dataprep.corpus_to_dataset import make_dataset
from src.word2gm_fast.dataprep.index_vocab import make_vocab
from src.word2gm_fast.dataprep.dataset_to_triplets import build_skipgram_triplets
from src.word2gm_fast.dataprep.tfrecord_io import (
    write_triplets_to_tfrecord, 
    load_triplets_from_tfrecord
)

# Silence TensorFlow
# tf already imported above

def benchmark_compression_impact():
    """
    Benchmark the impact of TFRecord compression on training data loading performance.
    """
    print("🔬 TFRECORD COMPRESSION IMPACT BENCHMARK")
    print("=" * 60)
    
    # Use a small test corpus
    corpus_file = "1780.txt"
    corpus_dir = "/vast/edk202/NLP_corpora/Google_Books/20200217/eng-fiction/5gram_files/6corpus/yearly_files/data"
    corpus_path = os.path.join(corpus_dir, corpus_file)
    
    if not os.path.exists(corpus_path):
        print(f"❌ Test corpus not found: {corpus_path}")
        return
        
    print(f"📁 Using test corpus: {corpus_file}")
    
    # Create test data
    print("🔄 Generating test triplets...")
    dataset, _ = make_dataset(corpus_path, show_summary=False)
    dataset = dataset.cache()
    vocab_table = make_vocab(dataset)
    triplets_ds = build_skipgram_triplets(dataset, vocab_table)
    
    # Count triplets for benchmarking
    triplet_count = sum(1 for _ in triplets_ds)
    print(f"   Generated {triplet_count:,} triplets")
    
    # Recreate for saving
    triplets_ds = build_skipgram_triplets(dataset, vocab_table)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        uncompressed_path = os.path.join(temp_dir, "triplets_uncompressed.tfrecord")
        compressed_path = os.path.join(temp_dir, "triplets_compressed.tfrecord.gz")
        
        # Save both versions
        print("\n💾 SAVING BENCHMARKS")
        print("-" * 40)
        
        # Save uncompressed
        start_time = time.perf_counter()
        write_triplets_to_tfrecord(triplets_ds, uncompressed_path, compress=False)
        uncompressed_save_time = time.perf_counter() - start_time
        uncompressed_size = os.path.getsize(uncompressed_path) / 1024 / 1024
        
        # Recreate dataset for compressed save
        triplets_ds = build_skipgram_triplets(dataset, vocab_table)
        
        # Save compressed
        start_time = time.perf_counter()
        write_triplets_to_tfrecord(triplets_ds, compressed_path, compress=True)
        compressed_save_time = time.perf_counter() - start_time
        compressed_size = os.path.getsize(compressed_path) / 1024 / 1024
        
        print(f"\n📊 SAVE PERFORMANCE:")
        print(f"   • Uncompressed: {uncompressed_save_time:.2f}s, {uncompressed_size:.2f}MB")
        print(f"   • Compressed:   {compressed_save_time:.2f}s, {compressed_size:.2f}MB")
        print(f"   • Compression ratio: {uncompressed_size/compressed_size:.1f}x")
        print(f"   • Save overhead: {compressed_save_time/uncompressed_save_time:.1f}x")
        
        # Loading benchmarks
        print(f"\n📈 LOADING BENCHMARKS")
        print("-" * 40)
        
        def benchmark_loading(path, label, num_trials=5):
            """Benchmark loading performance"""
            times = []
            for trial in range(num_trials):
                start_time = time.perf_counter()
                ds = load_triplets_from_tfrecord(path)
                # Force evaluation by taking a few samples
                list(ds.take(100))
                load_time = time.perf_counter() - start_time
                times.append(load_time)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"   • {label}: {avg_time:.3f}s avg ({min_time:.3f}-{max_time:.3f}s)")
            return avg_time
        
        # Suppress TFRecord loading messages temporarily
        import io
        from contextlib import redirect_stdout
        
        with redirect_stdout(io.StringIO()):
            uncompressed_time = benchmark_loading(uncompressed_path, "Uncompressed")
            compressed_time = benchmark_loading(compressed_path, "Compressed  ")
        
        loading_overhead = compressed_time / uncompressed_time
        print(f"   • Loading overhead: {loading_overhead:.2f}x")
        
        # Training simulation benchmark
        print(f"\n🎯 TRAINING SIMULATION BENCHMARK")
        print("-" * 40)
        
        def benchmark_training_iteration(path, label, batch_size=1024, num_batches=50):
            """Simulate training data loading"""
            with redirect_stdout(io.StringIO()):
                ds = load_triplets_from_tfrecord(path)
            
            # Typical training pipeline setup
            ds = ds.batch(batch_size)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            
            start_time = time.perf_counter()
            samples_processed = 0
            
            for i, batch in enumerate(ds.take(num_batches)):
                # Simulate some processing work
                center, positive, negative = batch
                _ = tf.nn.embedding_lookup([tf.random.normal([10000, 128])], center)
                samples_processed += batch_size
            
            total_time = time.perf_counter() - start_time
            throughput = samples_processed / total_time
            
            print(f"   • {label}: {total_time:.2f}s, {throughput:,.0f} samples/sec")
            return throughput
        
        uncompressed_throughput = benchmark_training_iteration(uncompressed_path, "Uncompressed")
        compressed_throughput = benchmark_training_iteration(compressed_path, "Compressed  ")
        
        throughput_ratio = compressed_throughput / uncompressed_throughput
        print(f"   • Training throughput ratio: {throughput_ratio:.3f}")
        
        # Summary
        print(f"\n🎯 SUMMARY & RECOMMENDATIONS")
        print("=" * 60)
        print(f"💾 Storage savings: {uncompressed_size/compressed_size:.1f}x smaller files")
        print(f"⏱️  Loading overhead: {loading_overhead:.2f}x slower")
        print(f"🚀 Training impact: {(1-throughput_ratio)*100:.1f}% throughput reduction")
        
        if loading_overhead < 1.5 and throughput_ratio > 0.85:
            print(f"✅ RECOMMENDATION: Use compression (minimal impact, good storage savings)")
        elif loading_overhead < 2.0 and throughput_ratio > 0.75:
            print(f"⚠️  RECOMMENDATION: Consider compression trade-offs based on storage constraints")
        else:
            print(f"❌ RECOMMENDATION: Avoid compression for training (too much performance impact)")
        
        # Disk I/O considerations
        print(f"\n💡 ADDITIONAL CONSIDERATIONS:")
        print(f"   • Compressed files reduce disk I/O bandwidth requirements")
        print(f"   • Modern CPUs can decompress faster than disk reads in many cases")
        print(f"   • Network storage benefits more from compression than local SSD")
        print(f"   • Training from remote storage may favor compression")


if __name__ == "__main__":
    benchmark_compression_impact()
