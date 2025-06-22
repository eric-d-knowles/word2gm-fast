# TFRecord Compression Impact Analysis
# Let's run a quick benchmark to understand the performance trade-offs

import time
import tempfile
import os

def benchmark_compression_impact():
    """Quick benchmark of compression impact on training data loading"""
    
    print("🔬 TFRecord Compression Impact Analysis")
    print("=" * 50)
    
    # Use existing training data
    output_dir = "./training_data"
    uncompressed_path = os.path.join(output_dir, "triplets_uncompressed.tfrecord")  
    compressed_path = os.path.join(output_dir, "triplets.tfrecord.gz")
    
    if not os.path.exists(compressed_path):
        print("❌ No compressed TFRecord found. Please run the data preparation first.")
        return
        
    # Create uncompressed version for comparison
    if not os.path.exists(uncompressed_path):
        print("🔄 Creating uncompressed version for comparison...")
        from src.word2gm_fast.dataprep.tfrecord_io import load_triplets_from_tfrecord, write_triplets_to_tfrecord
        
        # Load compressed and save uncompressed
        triplets_ds = load_triplets_from_tfrecord(compressed_path)
        write_triplets_to_tfrecord(triplets_ds, uncompressed_path, compress=False)
    
    # File size comparison
    if os.path.exists(uncompressed_path):
        compressed_size = os.path.getsize(compressed_path) / 1024 / 1024
        uncompressed_size = os.path.getsize(uncompressed_path) / 1024 / 1024
        compression_ratio = uncompressed_size / compressed_size
        
        print(f"📊 STORAGE COMPARISON:")
        print(f"   • Uncompressed: {uncompressed_size:.2f} MB")
        print(f"   • Compressed:   {compressed_size:.2f} MB") 
        print(f"   • Space savings: {compression_ratio:.1f}x")
    
    # Loading speed comparison
    print(f"\n⏱️  LOADING SPEED COMPARISON:")
    
    def time_loading(path, label, trials=3):
        times = []
        for i in range(trials):
            start = time.perf_counter()
            ds = load_triplets_from_tfrecord(path)
            # Force some evaluation
            samples = list(ds.take(1000))
            duration = time.perf_counter() - start
            times.append(duration)
        
        avg_time = sum(times) / len(times)
        print(f"   • {label}: {avg_time:.3f}s avg")
        return avg_time
    
    # Suppress verbose output
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    
    try:
        sys.stdout = StringIO()
        compressed_time = time_loading(compressed_path, "Compressed")
        if os.path.exists(uncompressed_path):
            uncompressed_time = time_loading(uncompressed_path, "Uncompressed")
            overhead = compressed_time / uncompressed_time
        else:
            overhead = 1.0
    finally:
        sys.stdout = old_stdout
    
    print(f"   • Compressed:   {compressed_time:.3f}s")
    if os.path.exists(uncompressed_path):
        print(f"   • Uncompressed: {uncompressed_time:.3f}s")
        print(f"   • Overhead:     {overhead:.2f}x")
    
    # Training simulation
    print(f"\n🎯 TRAINING SIMULATION:")
    
    def simulate_training(path, label, batch_size=512, num_batches=10):
        with StringIO() as buf:
            old_stdout = sys.stdout
            sys.stdout = buf
            try:
                ds = load_triplets_from_tfrecord(path)
            finally:
                sys.stdout = old_stdout
        
        # Typical training pipeline
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        start = time.perf_counter()
        total_samples = 0
        
        for i, (center, pos, neg) in enumerate(ds.take(num_batches)):
            # Simulate some computation
            _ = tf.reduce_sum(center + pos + neg)
            total_samples += batch_size
            
        duration = time.perf_counter() - start
        throughput = total_samples / duration
        print(f"   • {label}: {throughput:,.0f} samples/sec")
        return throughput
    
    compressed_throughput = simulate_training(compressed_path, "Compressed  ")
    if os.path.exists(uncompressed_path):
        uncompressed_throughput = simulate_training(uncompressed_path, "Uncompressed")
        throughput_ratio = compressed_throughput / uncompressed_throughput
        impact = (1 - throughput_ratio) * 100
        print(f"   • Training impact: {impact:+.1f}%")
    
    # Recommendations
    print(f"\n💡 ANALYSIS:")
    if os.path.exists(uncompressed_path):
        print(f"   • Compression reduces file size by {compression_ratio:.1f}x")
        print(f"   • Loading overhead: {overhead:.2f}x slower") 
        print(f"   • Training impact: {impact:+.1f}%")
        
        if overhead < 1.3 and abs(impact) < 15:
            print(f"✅ RECOMMENDATION: Compression is acceptable for most use cases")
        elif overhead < 2.0 and abs(impact) < 25:
            print(f"⚠️  RECOMMENDATION: Consider trade-offs based on storage/bandwidth needs")  
        else:
            print(f"❌ RECOMMENDATION: Avoid compression for performance-critical training")
    else:
        print(f"   • Analysis limited without uncompressed comparison")
        print(f"   • Compression provides {compression_ratio:.1f}x storage savings")
        
    print(f"\n🔧 FACTORS TO CONSIDER:")
    print(f"   • Network storage benefits more from compression")
    print(f"   • Modern CPUs decompress faster than slow disk I/O")
    print(f"   • Bandwidth-limited environments favor compression")
    print(f"   • Local NVMe storage may not need compression")

# Run the benchmark
benchmark_compression_impact()
