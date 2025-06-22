"""
Complete data preparation pipeline for Word2GM skip-gram training data.

This module provides a high-level interface to process corpus files and generate
TFRecord training artifacts. It handles the entire pipeline from corpus filtering
through TFRecord serialization with optimized performance and clean output.

Usage:
    from src.word2gm_fast.dataprep.pipeline import prepare_training_data
    
    # Process a corpus file and generate training artifacts
    prepare_training_data(
        corpus_file="2019.txt",
        corpus_dir="/vast/edk202/NLP_corpora/...",
        output_subdir="2019_artifacts"  # Optional: creates nested directory
    )
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
from io import StringIO
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Pipeline components will be imported when needed to avoid multiprocessing issues


def prepare_training_data(
    corpus_file: str,
    corpus_dir: str,
    output_subdir: Optional[str] = None,
    compress: bool = True,
    show_progress: bool = True,
    show_summary: bool = True,
    cache_dataset: bool = True
) -> Tuple[str, dict]:
    """
    Complete data preparation pipeline for Word2GM skip-gram training.
    
    Takes a preprocessed corpus file and generates optimized TFRecord artifacts
    for efficient model training. All processing uses TensorFlow-native operations
    for scalability and performance.
    
    Parameters
    ----------
    corpus_file : str
        Name of the corpus file (e.g., "2019.txt")
    corpus_dir : str
        Directory containing the corpus file (e.g., "/vast/edk202/NLP_corpora/...")
    output_subdir : str, optional
        Subdirectory name for artifacts (e.g., "2019_artifacts"). If None,
        artifacts are saved directly in corpus_dir as "training_artifacts"
    compress : bool, default=True
        Whether to compress TFRecord files with GZIP
    show_progress : bool, default=True
        Whether to display progress and summary information
    show_summary : bool, default=True
        Whether to display a detailed summary of the pipeline execution
    cache_dataset : bool, default=True
        Whether to cache the filtered dataset for faster processing
        
    Returns
    -------
    output_dir : str
        Path to the directory containing generated artifacts
    summary : dict
        Dictionary with pipeline statistics and file information
        
    Raises
    ------
    FileNotFoundError
        If the corpus file does not exist
    ValueError
        If corpus_dir is not accessible
        
    Examples
    --------
    >>> # Basic usage - artifacts in corpus_dir/training_artifacts/
    >>> output_dir, summary = prepare_training_data(
    ...     corpus_file="2019.txt",
    ...     corpus_dir="/vast/edk202/NLP_corpora/Google_Books/..."
    ... )
    
    >>> # Organized artifacts - artifacts in corpus_dir/2019_artifacts/
    >>> output_dir, summary = prepare_training_data(
    ...     corpus_file="2019.txt", 
    ...     corpus_dir="/vast/edk202/NLP_corpora/Google_Books/...",
    ...     output_subdir="2019_artifacts"
    ... )
    
    >>> # Access results
    >>> print(f"Artifacts saved to: {output_dir}")
    >>> print(f"Vocabulary size: {summary['vocab_size']:,}")
    >>> print(f"Training triplets: {summary['triplet_count']:,}")
    """
    
    # Import TensorFlow and pipeline components inside function to avoid multiprocessing issues
    from ..utils import import_tensorflow_silently
    tf = import_tensorflow_silently(deterministic=False)
    
    from .corpus_to_dataset import make_dataset
    from .index_vocab import make_vocab
    from .dataset_to_triplets import build_skipgram_triplets
    from .tfrecord_io import save_pipeline_artifacts
    
    # Validate inputs
    corpus_path = os.path.join(corpus_dir, corpus_file)
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    if not os.path.isdir(corpus_dir):
        raise ValueError(f"Corpus directory not accessible: {corpus_dir}")
    
    # Set up output directory
    if output_subdir:
        # Create year-specific subdirectory (e.g., "2019_artifacts")
        output_dir = os.path.join(corpus_dir, output_subdir)
    else:
        # Default to generic subdirectory
        output_dir = os.path.join(corpus_dir, "training_artifacts")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get corpus information
    file_size_mb = os.path.getsize(corpus_path) / 1024 / 1024
    
    if show_progress:
        print(f"🚀 Starting Word2GM data preparation pipeline")
        print(f"📁 Corpus: {corpus_file} ({file_size_mb:.3f} MB)")
        print(f"💾 Output: {output_dir}")
        print()
    
    # Start timing
    start_total = time.perf_counter()
    
    # Step 1: Load and filter corpus
    if show_progress:
        print("🔄 Step 1/4: Loading and filtering corpus...")
    
    step_start = time.perf_counter()
    dataset, _ = make_dataset(corpus_path, show_summary=False)
    if cache_dataset:
        dataset = dataset.cache()
    step_duration = time.perf_counter() - step_start
    
    if show_progress:
        print(f"   ✅ Corpus filtered in {step_duration:.3f}s")
    
    # Step 2: Build vocabulary  
    if show_progress:
        print("🔄 Step 2/4: Building vocabulary...")
        
    step_start = time.perf_counter()
    vocab_table = make_vocab(dataset)
    vocab_export = vocab_table.export()
    vocab_size = len(vocab_export[0].numpy())
    step_duration = time.perf_counter() - step_start
    
    if show_progress:
        print(f"   ✅ Vocabulary built: {vocab_size:,} words in {step_duration:.3f}s")
    
    # Step 3: Generate training triplets
    if show_progress:
        print("🔄 Step 3/4: Generating skip-gram triplets...")
        
    step_start = time.perf_counter()
    triplets_ds = build_skipgram_triplets(dataset, vocab_table)
    step_duration = time.perf_counter() - step_start
    
    if show_progress:
        print(f"   ✅ Generated triplets dataset in {step_duration:.3f}s")
    
    # Step 4: Save TFRecord artifacts (count triplets during writing)
    if show_progress:
        print("🔄 Step 4/4: Saving TFRecord artifacts...")
        
    step_start = time.perf_counter()
    
    # Suppress verbose output during save
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        artifacts = save_pipeline_artifacts(
            dataset=dataset,
            vocab_table=vocab_table,
            triplets_ds=triplets_ds,
            output_dir=output_dir,
            compress=compress
        )
        # Get triplet count from artifacts (counted during TFRecord writing)
        triplet_count = artifacts['triplet_count']
    finally:
        sys.stdout = old_stdout
    
    step_duration = time.perf_counter() - step_start
    total_duration = time.perf_counter() - start_total
    
    # Calculate file sizes
    triplets_file = os.path.join(output_dir, "triplets.tfrecord.gz" if compress else "triplets.tfrecord")
    vocab_file = os.path.join(output_dir, "vocab.tfrecord.gz" if compress else "vocab.tfrecord")
    
    triplets_size_mb = os.path.getsize(triplets_file) / 1024 / 1024 if os.path.exists(triplets_file) else 0
    vocab_size_mb = os.path.getsize(vocab_file) / 1024 / 1024 if os.path.exists(vocab_file) else 0
    total_artifact_size_mb = triplets_size_mb + vocab_size_mb
    
    if show_summary:
        print(f"✅ Artifacts saved in {step_duration:.3f}s")
        print()
        print("📊 PIPELINE SUMMARY")
        print("=" * 50)
        print(f"Corpus processed:   {file_size_mb:.3f} MB")
        print(f"Vocabulary size:    {vocab_size:,} words")
        print(f"Training triplets:  {triplet_count:,}")
        print(f"Artifact size:      {total_artifact_size_mb:.3f} MB")
        print(f"Compression ratio:  {file_size_mb / total_artifact_size_mb:.3f}x")
        print(f"Total time:         {total_duration:.3f}s")
        print(f"Processing rate:    {file_size_mb / total_duration:.3f} MB/s")
        print(f"Triplet rate:       {triplet_count / total_duration:.3f} MB/s")
        print()
        print("📁 Generated files:")
        print(f"   🎯 {os.path.basename(triplets_file)} ({triplets_size_mb:.3f} MB)")
        print(f"   📚 {os.path.basename(vocab_file)} ({vocab_size_mb:.3f} MB)")
        print()
        print("🎉 Pipeline complete! Ready for model training.")
    
    # Create summary dictionary
    summary = {
        'corpus_file': corpus_file,
        'corpus_size_mb': file_size_mb,
        'vocab_size': vocab_size,
        'triplet_count': triplet_count,
        'artifacts_size_mb': total_artifact_size_mb,
        'compression_ratio': file_size_mb / total_artifact_size_mb if total_artifact_size_mb > 0 else 0,
        'total_duration_s': total_duration,
        'processing_rate_mb_s': file_size_mb / total_duration,
        'triplets_file': triplets_file,
        'vocab_file': vocab_file,
        'output_dir': output_dir
    }
    
    return output_dir, summary


def prepare_training_data_fast(
    corpus_file: str,
    corpus_dir: str,
    output_subdir: Optional[str] = None,
    compress: bool = True,
    show_progress: bool = True,
    show_summary: bool = True,
    cache_dataset: bool = True
) -> Tuple[str, dict]:
    """
    Optimized data preparation pipeline that skips triplet manifestation.
    
    This version writes directly to TFRecord without first manifesting the entire
    triplets dataset in memory. For large corpora, this can save significant time
    by avoiding the double iteration over triplets.
    
    Parameters are identical to prepare_training_data().
    
    Returns
    -------
    output_dir : str
        Path to the directory containing generated artifacts
    summary : dict
        Dictionary with pipeline statistics and file information
    """
    
    # Import TensorFlow and pipeline components inside function to avoid multiprocessing issues
    from ..utils import import_tensorflow_silently
    tf = import_tensorflow_silently(deterministic=False)
    
    from .corpus_to_dataset import make_dataset
    from .index_vocab import make_vocab
    from .dataset_to_triplets import build_skipgram_triplets
    from .tfrecord_io import write_triplets_to_tfrecord_silent, write_vocab_to_tfrecord
    
    # Validate inputs (same as original)
    corpus_path = os.path.join(corpus_dir, corpus_file)
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    if not os.path.isdir(corpus_dir):
        raise ValueError(f"Corpus directory not accessible: {corpus_dir}")
    
    # Set up output directory
    if output_subdir:
        output_dir = os.path.join(corpus_dir, output_subdir)
    else:
        output_dir = os.path.join(corpus_dir, "training_artifacts")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get corpus information
    file_size_mb = os.path.getsize(corpus_path) / 1024 / 1024
    
    if show_progress:
        print(f"🚀 Starting Word2GM data preparation pipeline (FAST mode)")
        print(f"📁 Corpus: {corpus_file} ({file_size_mb:.3f} MB)")
        print(f"💾 Output: {output_dir}")
        print("⚡ Skip manifestation: ON")
        print()
    
    # Start timing
    start_total = time.perf_counter()
    
    # Step 1: Load and filter corpus
    if show_progress:
        print("🔄 Step 1/3: Loading and filtering corpus...")
    
    step_start = time.perf_counter()
    dataset, _ = make_dataset(corpus_path, show_summary=False)
    if cache_dataset:
        dataset = dataset.cache()
    step_duration = time.perf_counter() - step_start
    
    if show_progress:
        print(f"   ✅ Corpus filtered in {step_duration:.3f}s")
    
    # Step 2: Build vocabulary  
    if show_progress:
        print("🔄 Step 2/3: Building vocabulary...")
        
    step_start = time.perf_counter()
    vocab_table = make_vocab(dataset)
    vocab_export = vocab_table.export()
    vocab_size = len(vocab_export[0].numpy())
    step_duration = time.perf_counter() - step_start
    
    if show_progress:
        print(f"   ✅ Vocabulary built: {vocab_size:,} words in {step_duration:.3f}s")
    
    # Step 3: Generate triplets and save directly to TFRecord
    if show_progress:
        print("🔄 Step 3/3: Generating triplets and saving to TFRecord...")
        
    step_start = time.perf_counter()
    triplets_ds = build_skipgram_triplets(dataset, vocab_table)
    
    # Save artifacts and get triplet count in one pass
    from .tfrecord_io import write_triplets_to_tfrecord_silent, write_vocab_to_tfrecord
    
    ext = ".tfrecord.gz" if compress else ".tfrecord"
    triplets_path = os.path.join(output_dir, f"triplets{ext}")
    vocab_path = os.path.join(output_dir, f"vocab{ext}")
    
    # Suppress verbose output during save
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        # Save vocab
        write_vocab_to_tfrecord(vocab_table, vocab_path, compress=compress)
        # Save triplets and get count
        triplet_count = write_triplets_to_tfrecord_silent(triplets_ds, triplets_path, compress=compress)
    finally:
        sys.stdout = old_stdout
    
    step_duration = time.perf_counter() - step_start
    total_duration = time.perf_counter() - start_total
    
    # Calculate file sizes
    triplets_size_mb = os.path.getsize(triplets_path) / 1024 / 1024 if os.path.exists(triplets_path) else 0
    vocab_size_mb = os.path.getsize(vocab_path) / 1024 / 1024 if os.path.exists(vocab_path) else 0
    total_artifact_size_mb = triplets_size_mb + vocab_size_mb
    
    if show_summary:
        print(f"✅ Triplets generated and saved in {step_duration:.3f}s")
        print()
        print("📊 PIPELINE SUMMARY (FAST MODE)")
        print("=" * 50)
        print(f"Corpus processed:   {file_size_mb:.3f} MB")
        print(f"Vocabulary size:    {vocab_size:,} words")
        print(f"Training triplets:  {triplet_count:,}")
        print(f"Artifact size:      {total_artifact_size_mb:.3f} MB")
        print(f"Compression ratio:  {file_size_mb / total_artifact_size_mb:.3f}x")
        print(f"Total time:         {total_duration:.3f}s")
        print(f"Processing rate:    {file_size_mb / total_duration:.3f} MB/s")
        print()
        print("📁 Generated files:")
        print(f"   🎯 {os.path.basename(triplets_path)} ({triplets_size_mb:.3f} MB)")
        print(f"   📚 {os.path.basename(vocab_path)} ({vocab_size_mb:.3f} MB)")
        print()
        print("⚡ Fast mode: Skipped dataset manifestation")
        print("🎉 Pipeline complete! Ready for model training.")
    
    # Create summary dictionary
    summary = {
        'corpus_file': corpus_file,
        'corpus_size_mb': file_size_mb,
        'vocab_size': vocab_size,
        'triplet_count': triplet_count,
        'artifacts_size_mb': total_artifact_size_mb,
        'compression_ratio': file_size_mb / total_artifact_size_mb if total_artifact_size_mb > 0 else 0,
        'total_duration_s': total_duration,
        'processing_rate_mb_s': file_size_mb / total_duration,
        'triplets_file': triplets_path,
        'vocab_file': vocab_path,
        'output_dir': output_dir,
        'fast_mode': True
    }
    
    return output_dir, summary


def get_corpus_years(corpus_dir: str) -> list:
    """
    Get available corpus years from a directory.
    
    Parameters
    ----------
    corpus_dir : str
        Directory containing corpus files
        
    Returns
    -------
    list
        Sorted list of available years (as strings)
        
    Examples
    --------
    >>> years = get_corpus_years("/vast/edk202/NLP_corpora/...")
    >>> print(f"Available years: {', '.join(years)}")
    """
    if not os.path.isdir(corpus_dir):
        return []
    
    years = []
    for filename in os.listdir(corpus_dir):
        if filename.endswith('.txt') and filename[:-4].isdigit():
            year = filename[:-4]
            if len(year) == 4:  # Assume 4-digit years
                years.append(year)
    
    return sorted(years)


def _process_single_year(args):
    """
    Helper function for multiprocessing - processes a single year.
    
    Parameters
    ----------
    args : tuple
        (year, corpus_dir, compress, show_progress) tuple
        
    Returns
    -------
    tuple
        (year, success, result) where result is either summary dict or error string
    """
    year, corpus_dir, compress, show_progress = args
    
    try:
        corpus_file = f"{year}.txt"
        output_subdir = f"{year}_artifacts"
        
        output_dir, summary = prepare_training_data(
            corpus_file=corpus_file,
            corpus_dir=corpus_dir,
            output_subdir=output_subdir,
            compress=compress,
            show_progress=show_progress,
            show_summary=False  # Suppress individual summaries in parallel mode
        )
        
        return (year, True, summary)
        
    except Exception as e:
        return (year, False, str(e))


def batch_prepare_training_data(
    years: list,
    corpus_dir: str,
    compress: bool = True,
    show_progress: bool = True,
    show_summary: bool = True,
    max_workers: int = None,
    use_multiprocessing: bool = True
) -> dict:
    """
    Prepare training data for multiple years in batch with optional parallel processing.
    
    Parameters
    ----------
    years : list
        List of years to process (e.g., ["2018", "2019", "2020"])
    corpus_dir : str
        Directory containing corpus files
    compress : bool, default=True
        Whether to compress TFRecord files
    show_progress : bool, default=True
        Whether to show progress for each year
    show_summary : bool, default=True
        Whether to display a summary of the batch processing
    max_workers : int, optional
        Maximum number of parallel workers. If None, uses min(cpu_count(), len(years))
    use_multiprocessing : bool, default=True
        Whether to use multiprocessing for parallel year processing.
        If False, processes years sequentially.
        
    Returns
    -------
    dict
        Dictionary mapping years to their summary information
        
    Examples
    --------
    >>> # Process multiple years in parallel (default)
    >>> results = batch_prepare_training_data(
    ...     years=["2018", "2019", "2020"],
    ...     corpus_dir="/vast/edk202/NLP_corpora/...",
    ...     max_workers=4
    ... )
    >>> 
    >>> # Process sequentially (for debugging)
    >>> results = batch_prepare_training_data(
    ...     years=["2018", "2019"],
    ...     corpus_dir="/vast/edk202/NLP_corpora/...",
    ...     use_multiprocessing=False
    ... )
    """
    results = {}
    
    if not use_multiprocessing or len(years) == 1:
        # Sequential processing (original implementation)
        for i, year in enumerate(years, 1):
            if show_progress:
                print(f"\n{'='*60}")
                print(f"📅 Processing year {year} ({i}/{len(years)})")
                print(f"{'='*60}")
            
            try:
                corpus_file = f"{year}.txt"
                output_subdir = f"{year}_artifacts"
                
                output_dir, summary = prepare_training_data(
                    corpus_file=corpus_file,
                    corpus_dir=corpus_dir,
                    output_subdir=output_subdir,
                    compress=compress,
                    show_progress=show_progress
                )
                
                results[year] = summary
                
            except Exception as e:
                if show_progress:
                    print(f"❌ Error processing {year}: {e}")
                results[year] = {'error': str(e)}
    
    else:
        # Parallel processing using multiprocessing
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(years))
        
        if show_progress:
            print(f"\n{'='*60}")
            print(f"🚀 PARALLEL BATCH PROCESSING")
            print(f"{'='*60}")
            print(f"📅 Processing {len(years)} years: {', '.join(years)}")
            print(f"💻 Using {max_workers} parallel workers")
            print(f"⚡ Estimated speedup: {min(max_workers, len(years)):.1f}x")
            print(f"{'='*60}")
        
        # Prepare arguments for worker processes
        worker_args = [(year, corpus_dir, compress, False) for year in years]  # show_progress=False in workers
        
        # Start parallel processing
        start_time = time.perf_counter()
        completed_count = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_year = {
                executor.submit(_process_single_year, args): args[0] 
                for args in worker_args
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_year):
                year = future_to_year[future]
                completed_count += 1
                
                try:
                    year_result, success, data = future.result()
                    
                    if success:
                        results[year_result] = data
                        if show_progress:
                            triplets = data.get('triplet_count', 0)
                            vocab_size = data.get('vocab_size', 0)
                            duration = data.get('total_duration_s', 0)
                            print(f"✅ {year_result} complete ({completed_count}/{len(years)}): "
                                  f"{triplets:,} triplets, {vocab_size:,} vocab, {duration:.1f}s")
                    else:
                        results[year_result] = {'error': data}
                        if show_progress:
                            print(f"❌ {year_result} failed ({completed_count}/{len(years)}): {data}")
                            
                except Exception as e:
                    results[year] = {'error': str(e)}
                    if show_progress:
                        print(f"❌ {year} failed ({completed_count}/{len(years)}): {e}")
        
        parallel_duration = time.perf_counter() - start_time
        
        if show_progress:
            print(f"\n🎯 Parallel processing completed in {parallel_duration:.1f}s")

    if show_summary:
        print(f"\n{'='*60}")
        print(f"🎉 Batch processing complete!")
        print(f"{'='*60}")
        
        # Summary table
        successful = [year for year, result in results.items() if 'error' not in result]
        failed = [year for year, result in results.items() if 'error' in result]
        
        print(f"✅ Successful: {len(successful)} years")
        if successful:
            total_triplets = sum(results[year]['triplet_count'] for year in successful)
            total_vocab = sum(results[year]['vocab_size'] for year in successful)
            avg_duration = sum(results[year]['total_duration_s'] for year in successful) / len(successful)
            
            print(f"   Total triplets: {total_triplets:,}")
            print(f"   Average vocab size: {total_vocab // len(successful):,}")
            print(f"   Average time per year: {avg_duration:.1f}s")
            
            if use_multiprocessing and len(years) > 1:
                # Calculate parallel efficiency
                sequential_estimate = avg_duration * len(successful)
                if 'parallel_duration' in locals():
                    actual_speedup = sequential_estimate / parallel_duration
                    efficiency = actual_speedup / min(max_workers, len(successful)) * 100
                    print(f"   Parallel speedup: {actual_speedup:.1f}x")
                    print(f"   Parallel efficiency: {efficiency:.1f}%")
        
        if failed:
            print(f"❌ Failed: {len(failed)} years: {', '.join(failed)}")
    
    return results


def estimate_fast_mode_savings(corpus_file: str, corpus_dir: str) -> dict:
    """
    Estimate potential time savings from using fast mode (no manifestation).
    
    This function analyzes a corpus to estimate how much time could be saved
    by skipping dataset manifestation and writing directly to TFRecord.
    
    Parameters
    ----------
    corpus_file : str
        Name of the corpus file to analyze
    corpus_dir : str  
        Directory containing the corpus file
        
    Returns
    -------
    dict
        Estimates including corpus stats and projected time savings
    """
    corpus_path = os.path.join(corpus_dir, corpus_file)
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    # Get basic file info
    file_size_mb = os.path.getsize(corpus_path) / 1024 / 1024
    
    print(f"📊 Analyzing corpus: {corpus_file} ({file_size_mb:.3f} MB)")
    print()
    
    # Quick corpus analysis - sample first 10k lines to estimate
    import time
    
    start = time.perf_counter()
    with open(corpus_path, 'r') as f:
        sample_lines = []
        for i, line in enumerate(f):
            if i >= 10000:  # Sample first 10k lines
                break
            sample_lines.append(line.strip())
    
    # Estimate line stats
    total_lines_estimate = file_size_mb * 1024 * 1024 // 80  # Rough estimate: ~80 bytes/line
    avg_line_length = sum(len(line) for line in sample_lines) / len(sample_lines) if sample_lines else 80
    
    # Quick vocabulary estimate from sample
    from ..utils import import_tensorflow_silently
    tf = import_tensorflow_silently(deterministic=False)
    from .corpus_to_dataset import make_dataset
    from .index_vocab import make_vocab
    from .dataset_to_triplets import build_skipgram_triplets
    
    # Create small sample dataset 
    sample_dataset = tf.data.Dataset.from_tensor_slices(sample_lines[:1000])
    vocab_table = make_vocab(sample_dataset)
    vocab_size_estimate = int(vocab_table.size().numpy())
    
    # Estimate triplets per line (typically 2-4 for 5-grams)
    sample_triplets = build_skipgram_triplets(sample_dataset, vocab_table)
    sample_triplet_count = sum(1 for _ in sample_triplets.take(100).as_numpy_iterator()) 
    triplets_per_line = sample_triplet_count / min(100, len(sample_lines))
    
    # Total estimates
    estimated_total_triplets = int(total_lines_estimate * triplets_per_line)
    
    # Time estimates based on empirical measurements
    # These are rough estimates from typical performance
    manifestation_rate_triplets_per_sec = 50000  # Typical rate for iteration 
    tfrecord_write_rate_triplets_per_sec = 25000  # Typical rate for TFRecord writing
    
    # Time for manifestation step
    manifestation_time_s = estimated_total_triplets / manifestation_rate_triplets_per_sec
    
    # Time for TFRecord writing (happens in both modes)
    tfrecord_write_time_s = estimated_total_triplets / tfrecord_write_rate_triplets_per_sec
    
    # Fast mode skips manifestation
    time_saved_s = manifestation_time_s
    time_saved_percent = (time_saved_s / (manifestation_time_s + tfrecord_write_time_s)) * 100
    
    analysis_time = time.perf_counter() - start
    
    results = {
        'corpus_file': corpus_file,
        'corpus_size_mb': file_size_mb,
        'estimated_total_lines': int(total_lines_estimate),
        'estimated_vocab_size': vocab_size_estimate,
        'estimated_total_triplets': estimated_total_triplets,
        'estimated_manifestation_time_s': manifestation_time_s,
        'estimated_tfrecord_write_time_s': tfrecord_write_time_s,
        'estimated_time_saved_s': time_saved_s,
        'estimated_time_saved_percent': time_saved_percent,
        'analysis_time_s': analysis_time
    }
    
    print("🔍 ANALYSIS RESULTS")
    print("=" * 40)
    print(f"Estimated total lines:    {int(total_lines_estimate):,}")
    print(f"Estimated vocabulary:     {vocab_size_estimate:,} words")
    print(f"Estimated triplets:       {estimated_total_triplets:,}")
    print()
    print("⏱️  TIME ESTIMATES")
    print("=" * 40)
    print(f"Manifestation step:       {manifestation_time_s:.1f}s")
    print(f"TFRecord writing:         {tfrecord_write_time_s:.1f}s")
    print(f"Total (standard mode):    {manifestation_time_s + tfrecord_write_time_s:.1f}s")
    print(f"Total (fast mode):        {tfrecord_write_time_s:.1f}s")
    print()
    print("💾 FAST MODE SAVINGS")
    print("=" * 40)
    print(f"Time saved:               {time_saved_s:.1f}s ({time_saved_s/60:.1f} min)")
    print(f"Speedup:                  {time_saved_percent:.1f}%")
    print()
    print("📝 Note: These are rough estimates based on typical performance.")
    print("   Actual results may vary depending on hardware and corpus characteristics.")
    
    return results
