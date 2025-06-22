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

# Import TensorFlow with silencing
from ..utils import import_tensorflow_silently
tf = import_tensorflow_silently(deterministic=False)

# Import pipeline components
from .corpus_to_dataset import make_dataset
from .index_vocab import make_vocab
from .dataset_to_triplets import build_skipgram_triplets
from .tfrecord_io import save_pipeline_artifacts


def prepare_training_data(
    corpus_file: str,
    corpus_dir: str,
    output_subdir: Optional[str] = None,
    compress: bool = True,
    show_progress: bool = True,
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
        print(f"📁 Corpus: {corpus_file} ({file_size_mb:.1f} MB)")
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
        print(f"   ✅ Corpus filtered in {step_duration:.1f}s")
    
    # Step 2: Build vocabulary  
    if show_progress:
        print("🔄 Step 2/4: Building vocabulary...")
        
    step_start = time.perf_counter()
    vocab_table = make_vocab(dataset)
    vocab_export = vocab_table.export()
    vocab_size = len(vocab_export[0].numpy())
    step_duration = time.perf_counter() - step_start
    
    if show_progress:
        print(f"   ✅ Vocabulary built: {vocab_size:,} words in {step_duration:.1f}s")
    
    # Step 3: Generate training triplets
    if show_progress:
        print("🔄 Step 3/4: Generating skip-gram triplets...")
        
    step_start = time.perf_counter()
    triplets_ds = build_skipgram_triplets(dataset, vocab_table)
    # Count triplets (consumes dataset)
    triplet_count = sum(1 for _ in triplets_ds.as_numpy_iterator())
    # Recreate for saving
    triplets_ds = build_skipgram_triplets(dataset, vocab_table)
    step_duration = time.perf_counter() - step_start
    
    if show_progress:
        print(f"   ✅ Generated {triplet_count:,} triplets in {step_duration:.1f}s")
    
    # Step 4: Save TFRecord artifacts
    if show_progress:
        print("🔄 Step 4/4: Saving TFRecord artifacts...")
        
    step_start = time.perf_counter()
    
    # Suppress verbose output during save
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        save_pipeline_artifacts(
            dataset=dataset,
            vocab_table=vocab_table,
            triplets_ds=triplets_ds,
            output_dir=output_dir,
            compress=compress
        )
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
    
    if show_progress:
        print(f"   ✅ Artifacts saved in {step_duration:.1f}s")
        print()
        print("📊 PIPELINE SUMMARY")
        print("=" * 50)
        print(f"Corpus processed:   {file_size_mb:.1f} MB")
        print(f"Vocabulary size:    {vocab_size:,} words")
        print(f"Training triplets:  {triplet_count:,}")
        print(f"Artifact size:      {total_artifact_size_mb:.1f} MB")
        print(f"Compression ratio:  {file_size_mb / total_artifact_size_mb:.1f}x")
        print(f"Total time:         {total_duration:.1f}s")
        print(f"Processing rate:    {file_size_mb / total_duration:.1f} MB/s")
        print()
        print("📁 Generated files:")
        print(f"   🎯 {os.path.basename(triplets_file)} ({triplets_size_mb:.1f} MB)")
        print(f"   📚 {os.path.basename(vocab_file)} ({vocab_size_mb:.1f} MB)")
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


def batch_prepare_training_data(
    years: list,
    corpus_dir: str,
    compress: bool = True,
    show_progress: bool = True
) -> dict:
    """
    Prepare training data for multiple years in batch.
    
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
        
    Returns
    -------
    dict
        Dictionary mapping years to their summary information
        
    Examples
    --------
    >>> # Process multiple years
    >>> results = batch_prepare_training_data(
    ...     years=["2018", "2019", "2020"],
    ...     corpus_dir="/vast/edk202/NLP_corpora/..."
    ... )
    >>> 
    >>> for year, summary in results.items():
    ...     print(f"{year}: {summary['vocab_size']:,} vocab, {summary['triplet_count']:,} triplets")
    """
    results = {}
    
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
    
    if show_progress:
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
            print(f"   Total triplets: {total_triplets:,}")
            print(f"   Average vocab size: {total_vocab // len(successful):,}")
        
        if failed:
            print(f"❌ Failed: {len(failed)} years: {', '.join(failed)}")
    
    return results
