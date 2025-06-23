"""
Complete data preparation pipeline for Word2GM skip-gram training data.

This module provides a high-level interface to process corpus files and generate
TFRecord training artifacts. It handles the entire pipeline from corpus filtering
through TFRecord serialization with optimized performance and clean output.

The pipeline uses an optimized direct-to-TFRecord approach that avoids unnecessary
dataset manifestation for maximum performance and memory efficiency.

Usage:
    from src.word2gm_fast.dataprep.pipeline import prepare_training_data
    
    # Process a corpus file and generate training artifacts
    prepare_training_data(
        corpus_file="2019.txt",
        corpus_dir="/vast/edk202/NLP_corpora/...",
        output_subdir="2019_artifacts"  # Optional: creates nested directory
    )
    
    # Process multiple years in parallel
    from src.word2gm_fast.dataprep.pipeline import batch_prepare_training_data
    
    # Auto-discover and process all corpus files
    results = batch_prepare_training_data(corpus_dir)
    
    # Process specific years
    results = batch_prepare_training_data(corpus_dir, years=["2018", "2019"])
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
    for efficient model training. Uses optimized direct-to-TFRecord approach
    for maximum performance by skipping unnecessary dataset manifestation.
    
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
    from .tfrecord_io import write_triplets_to_tfrecord_silent, write_vocab_to_tfrecord
    
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
        print(f"Starting Word2GM data preparation pipeline")
        print(f"Corpus: {corpus_file} ({file_size_mb:.3f} MB)")
        print(f"Output: {output_dir}")
        print()
    
    # Step 1: Load and filter corpus
    if show_progress:
        print("Step 1/3: Loading and filtering corpus...")
    
    step_start = time.perf_counter()
    dataset, _ = make_dataset(corpus_path, show_summary=False)
    if cache_dataset:
        dataset = dataset.cache()
    step_duration = time.perf_counter() - step_start
    
    if show_progress:
        print(f"   Corpus filtered in {step_duration:.3f}s")
    
    # Step 2: Build vocabulary  
    if show_progress:
        print("Step 2/3: Building vocabulary...")
        
    step_start = time.perf_counter()
    vocab_table = make_vocab(dataset)
    vocab_export = vocab_table.export()
    vocab_size = len(vocab_export[0].numpy())
    step_duration = time.perf_counter() - step_start
    
    if show_progress:
        print(f"   Vocabulary built: {vocab_size:,} words in {step_duration:.3f}s")
    
    # Step 3: Generate triplets and save directly to TFRecord (optimized approach)
    if show_progress:
        print("Step 3/3: Generating triplets and saving to TFRecord...")
        
    step_start = time.perf_counter()
    triplets_ds = build_skipgram_triplets(dataset, vocab_table)
    
    # Save artifacts and get triplet count in one pass (optimized: no dataset manifestation)
    ext = ".tfrecord.gz" if compress else ".tfrecord"
    triplets_path = os.path.join(output_dir, f"triplets{ext}")
    vocab_path = os.path.join(output_dir, f"vocab{ext}")
    
    # Suppress verbose output during save
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        # Save vocab
        write_vocab_to_tfrecord(vocab_table, vocab_path, compress=compress)
        # Save triplets and get count (direct streaming to TFRecord)
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
        print(f"Triplets generated and saved in {step_duration:.3f}s")
        print()
        print("PIPELINE SUMMARY")
        print("=" * 50)
        print(f"Corpus processed:   {file_size_mb:.3f} MB")
        print(f"Vocabulary size:    {vocab_size:,} words")
        print(f"Training triplets:  {triplet_count:,}")
        print(f"Artifact size:      {total_artifact_size_mb:.3f} MB")
        print(f"Compression ratio:  {file_size_mb / total_artifact_size_mb:.3f}x")
        print(f"Total time:         {total_duration:.3f}s")
        print(f"Processing rate:    {file_size_mb / total_duration:.3f} MB/s")
        print()
        print("Generated files:")
        print(f"   {os.path.basename(triplets_path)} ({triplets_size_mb:.3f} MB)")
        print(f"   {os.path.basename(vocab_path)} ({vocab_size_mb:.3f} MB)")
        print()
        print("Optimized: Direct-to-TFRecord streaming (no dataset manifestation)")
        print("Pipeline complete! Ready for model training.")
    
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
        'output_dir': output_dir
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
    corpus_dir: str,
    years: list = None,
    compress: bool = True,
    show_progress: bool = True,
    show_summary: bool = True,
    max_workers: int = None,
    use_multiprocessing: bool = True
) -> dict:
    """
    Prepare training data for multiple years in batch with optional parallel processing.
    
    If no years are specified, automatically processes all available corpus files in the directory.
    
    Parameters
    ----------
    corpus_dir : str
        Directory containing corpus files
    years : list, optional
        List of years to process (e.g., ["2018", "2019", "2020"]). 
        If None, automatically discovers and processes all available years.
    compress : bool, default=True
        Whether to compress TFRecord files
    show_progress : bool, default=True
        Whether to show progress for each year
    show_summary : bool, default=True
        Whether to display a summary of the batch processing
    max_workers : int, optional
        Maximum number of parallel workers. If None, uses cluster-aware detection
        with get_safe_worker_count() to respect actual resource allocation
    use_multiprocessing : bool, default=True
        Whether to use multiprocessing for parallel year processing.
        If False, processes years sequentially.
        
    Returns
    -------
    dict
        Dictionary mapping years to their summary information
        
    Examples
    --------
    >>> # Process ALL available years in directory (auto-discovery)
    >>> results = batch_prepare_training_data(
    ...     corpus_dir="/vast/edk202/NLP_corpora/...",
    ...     max_workers=8
    ... )
    >>> 
    >>> # Process specific years only
    >>> results = batch_prepare_training_data(
    ...     corpus_dir="/vast/edk202/NLP_corpora/...",
    ...     years=["2018", "2019", "2020"],
    ...     max_workers=4
    ... )
    >>> 
    >>> # Process sequentially (for debugging)
    >>> results = batch_prepare_training_data(
    ...     corpus_dir="/vast/edk202/NLP_corpora/...",
    ...     years=["2018", "2019"],
    ...     use_multiprocessing=False
    ... )
    """
    # Auto-discover years if not specified
    if years is None:
        years = get_corpus_years(corpus_dir)
        if show_progress:
            print(f"Auto-discovered {len(years)} corpus files in directory")
            print(f"Years to process: {', '.join(sorted(years))}")
            print()
    
    # Validate that we have years to process
    if not years:
        print("No corpus files found in directory or no years specified")
        return {}
    
    if show_progress and len(years) > 10:
        print(f"Processing {len(years)} years - this may take a while!")
        print(f"Consider using fewer max_workers or processing in smaller batches for very large datasets")
        print()
    
    results = {}
    
    if not use_multiprocessing or len(years) == 1:
        # Sequential processing (original implementation)
        for i, year in enumerate(years, 1):
            if show_progress:
                print(f"\n{'='*60}")
                print(f"Processing year {year} ({i}/{len(years)})")
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
                    print(f"Error processing {year}: {e}")
                results[year] = {'error': str(e)}
    
    else:
        # Parallel processing using multiprocessing
        if max_workers is None:
            max_workers = min(get_safe_worker_count(), len(years))
        
        if show_progress:
            print(f"\n{'='*60}")
            print(f"PARALLEL BATCH PROCESSING")
            print(f"{'='*60}")
            print(f"Processing {len(years)} years: {', '.join(years)}")
            print(f"Using {max_workers} parallel workers")
            print(f"Estimated speedup: {min(max_workers, len(years)):.1f}x")
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
                            print(f"{year_result} complete ({completed_count}/{len(years)}): "
                                  f"{triplets:,} triplets, {vocab_size:,} vocab, {duration:.1f}s")
                    else:
                        results[year_result] = {'error': data}
                        if show_progress:
                            print(f"{year_result} failed ({completed_count}/{len(years)}): {data}")
                            
                except Exception as e:
                    results[year] = {'error': str(e)}
                    if show_progress:
                        print(f"{year} failed ({completed_count}/{len(years)}): {e}")
        
        parallel_duration = time.perf_counter() - start_time
        
        if show_progress:
            print(f"\nParallel processing completed in {parallel_duration:.1f}s")

    if show_summary:
        print(f"\n{'='*60}")
        print(f"Batch processing complete!")
        print(f"{'='*60}")
        
        # Summary table
        successful = [year for year, result in results.items() if 'error' not in result]
        failed = [year for year, result in results.items() if 'error' in result]
        
        print(f"Successful: {len(successful)} years")
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
            print(f"Failed: {len(failed)} years: {', '.join(failed)}")
    
    return results


def process_all_corpora(
    corpus_dir: str,
    compress: bool = True,
    max_workers: int = None,
    show_progress: bool = True,
    exclude_years: list = None,
    include_patterns: list = None
) -> dict:
    """
    Convenience function to process ALL corpus files in a directory.
    
    This is equivalent to calling batch_prepare_training_data() with years=None,
    but provides additional filtering options for large directories.
    
    Parameters
    ----------
    corpus_dir : str
        Directory containing corpus files
    compress : bool, default=True
        Whether to compress TFRecord files
    max_workers : int, optional
        Maximum number of parallel workers. If None, auto-detects optimal number
        using cluster-aware detection that respects actual resource allocation
    show_progress : bool, default=True
        Whether to show progress updates
    exclude_years : list, optional
        List of years to skip (e.g., ["1800", "1801"] to skip early years)
    include_patterns : list, optional
        List of filename patterns to include (e.g., ["18*", "19*"] for 1800s and 1900s)
        
    Returns
    -------
    dict
        Dictionary mapping years to their summary information
        
    Examples
    --------
    >>> # Process ALL corpus files in directory
    >>> results = process_all_corpora("/vast/edk202/NLP_corpora/.../data")
    
    >>> # Process all except early years
    >>> results = process_all_corpora(
    ...     corpus_dir="/vast/edk202/NLP_corpora/.../data",
    ...     exclude_years=["1800", "1801", "1802"],
    ...     max_workers=8
    ... )
    
    >>> # Process only 1800s and 1900s
    >>> results = process_all_corpora(
    ...     corpus_dir="/vast/edk202/NLP_corpora/.../data",
    ...     include_patterns=["18*", "19*"]
    ... )
    """
    # Get all available years
    all_years = get_corpus_years(corpus_dir)
    
    if not all_years:
        print(f"No corpus files found in {corpus_dir}")
        return {}
    
    # Apply filters
    years_to_process = all_years.copy()
    
    # Apply include patterns
    if include_patterns:
        import fnmatch
        filtered_years = []
        for year in years_to_process:
            if any(fnmatch.fnmatch(year, pattern) for pattern in include_patterns):
                filtered_years.append(year)
        years_to_process = filtered_years
        
        if show_progress:
            print(f"Include patterns {include_patterns} matched {len(years_to_process)} years")
    
    # Apply exclude list
    if exclude_years:
        years_to_process = [year for year in years_to_process if year not in exclude_years]
        
        if show_progress:
            print(f"Excluding {len(exclude_years)} years: {', '.join(exclude_years)}")
    
    if not years_to_process:
        print("No corpus files remain after filtering")
        return {}
    
    # Auto-set reasonable max_workers for large datasets using cluster-aware detection
    if max_workers is None:
        base_workers = get_safe_worker_count()
        
        if len(years_to_process) <= 4:
            max_workers = min(base_workers, len(years_to_process))
        elif len(years_to_process) <= 20:
            max_workers = min(base_workers // 2, 8)  # Conservative for medium datasets
        else:
            max_workers = min(base_workers // 3, 6)  # Very conservative for large datasets
            
        if show_progress:
            print(f"Auto-selected {max_workers} workers for {len(years_to_process)} years")
    
    if show_progress:
        print(f"Processing {len(years_to_process)} corpus files with {max_workers} workers")
        total_estimated_time = len(years_to_process) * 45 / max_workers  # ~45s per year estimate
        print(f"Estimated completion time: {total_estimated_time/60:.1f} minutes")
        print()
    
    # Process the filtered years
    return batch_prepare_training_data(
        corpus_dir=corpus_dir,
        years=years_to_process,
        compress=compress,
        show_progress=show_progress,
        show_summary=True,
        max_workers=max_workers,
        use_multiprocessing=True
    )


def detect_cluster_resources() -> dict:
    """
    Detect actual allocated cluster resources vs hardware capabilities.
    
    Returns
    -------
    dict
        Dictionary with detected and allocated CPU information
    """
    import os
    import multiprocessing as mp
    
    result = {
        'detected_cpus': mp.cpu_count(),
        'allocated_cpus': None,
        'scheduler': None,
        'recommended_workers': None
    }
    
    # Check various cluster schedulers
    if os.environ.get('SLURM_CPUS_PER_TASK'):
        result['allocated_cpus'] = int(os.environ['SLURM_CPUS_PER_TASK'])
        result['scheduler'] = 'SLURM'
    elif os.environ.get('SLURM_NTASKS'):
        result['allocated_cpus'] = int(os.environ['SLURM_NTASKS'])
        result['scheduler'] = 'SLURM'
    elif os.environ.get('PBS_NUM_PPN'):
        result['allocated_cpus'] = int(os.environ['PBS_NUM_PPN'])
        result['scheduler'] = 'PBS/Torque'
    elif os.environ.get('LSB_DJOB_NUMPROC'):
        result['allocated_cpus'] = int(os.environ['LSB_DJOB_NUMPROC'])
        result['scheduler'] = 'LSF'
    
    # Recommend safe number of workers
    if result['allocated_cpus']:
        # Use allocated resources
        result['recommended_workers'] = result['allocated_cpus']
    else:
        # Conservative fallback - assume hyperthreading and leave some headroom
        result['recommended_workers'] = max(1, result['detected_cpus'] // 4)
    
    return result


def get_safe_worker_count(max_workers: int = None) -> int:
    """
    Get a safe number of workers considering cluster allocation.
    
    Parameters
    ----------
    max_workers : int, optional
        Maximum desired workers. If None, uses detected allocation.
        
    Returns
    -------
    int
        Safe number of workers to use
    """
    resources = detect_cluster_resources()
    
    if max_workers is None:
        return resources['recommended_workers']
    else:
        # Don't exceed allocation
        if resources['allocated_cpus']:
            return min(max_workers, resources['allocated_cpus'])
        else:
            # Conservative limit if no allocation detected
            return min(max_workers, resources['recommended_workers'])
