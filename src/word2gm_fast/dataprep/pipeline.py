"""
Simple, clean pipeline for Word2GM data preparation.

This module provides a straightforward interface to process corpus files 
and generate TFRecord training artifacts with minimal complexity.
"""

import os
import time
from typing import Optional, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from IPython import display

def process_single_year(corpus_dir: str, year: str, compress: bool = True, downsample_threshold: float = 1e-5) -> Dict:
    """
    Process a single year's corpus file.
    
    Parameters
    ----------
    corpus_dir : str
        Directory containing corpus files
    year : str
        Year to process (e.g., "1684")
    compress : bool
        Whether to compress output files
    downsample_threshold : float
        Threshold for downsampling frequent words (default: 1e-5)
        
    Returns
    -------
    dict
        Processing results
    """
    # Import inside function to avoid multiprocessing issues
    from .corpus_to_dataset import make_dataset
    from .dataset_to_frequency import dataset_to_frequency
    from .dataset_to_triplets import dataset_to_triplets
    from .index_vocab import triplets_to_integers
    from ..io.triplets import write_triplets_to_tfrecord
    from ..io.vocab import write_vocab_to_tfrecord
    from ..utils.tf_silence import import_tf_quietly, silence_tensorflow
    
    # Silent TensorFlow import
    tf = import_tf_quietly(force_cpu=True)
    
    start_time = time.perf_counter()
    
    try:
        # Set up paths
        corpus_file = f"{year}.txt"
        corpus_path = os.path.join(corpus_dir, corpus_file)
        output_dir = os.path.join(corpus_dir, f"{year}_artifacts")
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(corpus_path):
            return {"error": f"Corpus file not found: {corpus_path}"}
        
        # Wrap all TensorFlow operations in silence context
        with silence_tensorflow():
            # 1. Load corpus
            dataset, _ = make_dataset(corpus_path, show_summary=False)
            
            # 2. Build frequency table
            frequency_table = dataset_to_frequency(dataset)
            
            # 3. Generate string triplets
            string_triplets, _ = dataset_to_triplets(
                dataset=dataset,
                frequency_table=frequency_table,
                downsample_threshold=downsample_threshold,
                show_summary=False
            )
            
            # 4. Convert to integers and build vocab
            integer_triplets, vocab_table, vocab_list, vocab_size, _ = triplets_to_integers(
                triplets_dataset=string_triplets,
                frequency_table=frequency_table,
                show_summary=False
            )
            
            # 5. Save artifacts
            ext = ".tfrecord.gz" if compress else ".tfrecord"
            triplets_path = os.path.join(output_dir, f"triplets{ext}")
            vocab_path = os.path.join(output_dir, f"vocab{ext}")
            
            # Save files
            write_vocab_to_tfrecord(vocab_table, vocab_path, frequencies=frequency_table, compress=compress)
            triplet_count = write_triplets_to_tfrecord(integer_triplets, triplets_path, compress=compress)
        
        duration = time.perf_counter() - start_time
        
        return {
            "year": year,
            "vocab_size": vocab_size,
            "triplet_count": triplet_count,
            "duration": duration,
            "output_dir": output_dir
        }
        
    except Exception as e:
        return {"error": str(e)}


def process_year_range(
    corpus_dir: str,
    year_range: str,
    compress: bool = True,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
    downsample_threshold: float = 1e-5
) -> Dict[str, Dict]:
    """
    Process a range of years in parallel.
    
    Parameters
    ----------
    corpus_dir : str
        Directory containing corpus files
    year_range : str
        Year range (e.g., "1680-1690" or "1684,1685,1690")
    compress : bool
        Whether to compress output files
    max_workers : int, optional
        Number of parallel workers (default: auto-detect)
    show_progress : bool
        Whether to show progress
    downsample_threshold : float
        Threshold for downsampling frequent words (default: 1e-5)
        
    Returns
    -------
    dict
        Results for each year
        
    Examples
    --------
    >>> results = process_year_range("/path/to/corpus", "1680-1690")
    >>> results = process_year_range("/path/to/corpus", "1684,1690,1695")
    """
    
    # Parse year range
    years = parse_years(year_range)
    
    # Filter to existing files
    existing_years = []
    for year in years:
        corpus_path = os.path.join(corpus_dir, f"{year}.txt")
        if os.path.exists(corpus_path):
            existing_years.append(year)
    
    if not existing_years:
        if show_progress:
            display.display_markdown(
                f"<span style='font-family: monospace; font-size: 120%; font-weight: normal; color: orange;'>No corpus files found for years: {years}</span>",
                raw=True
            )
        return {}
    
    if show_progress:
        display.display_markdown(
            f"<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>Processing {len(existing_years)} years: {min(existing_years)}-{max(existing_years)}</span>",
            raw=True
        )
        if len(years) > len(existing_years):
            missing = len(years) - len(existing_years)
            display.display_markdown(
                f"<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>Skipping {missing} missing files</span>",
                raw=True
            )
    
    # Auto-detect workers
    if max_workers is None:
        import multiprocessing as mp
        max_workers = min(mp.cpu_count() // 2, len(existing_years))
    
    results = {}
    start_time = time.perf_counter()
    
    if len(existing_years) == 1 or max_workers == 1:
        # Sequential processing
        for year in existing_years:
            if show_progress:
                display.display_markdown(
                    f"<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>Processing {year}...</span>",
                    raw=True
                )
            result = process_single_year(corpus_dir, year, compress, downsample_threshold)
            results[year] = result
            if show_progress:
                if "error" in result:
                    display.display_markdown(
                        f"<span style='font-family: monospace; font-size: 120%; font-weight: normal; color: red;'>FAILED: {result['error']}</span>",
                        raw=True
                    )
                else:
                    display.display_markdown(
                        f"<span style='font-family: monospace; font-size: 120%; font-weight: normal; color: green;'>OK ({result['triplet_count']:,} triplets, {result['duration']:.1f}s)</span>",
                        raw=True
                    )
    else:
        # Parallel processing
        if show_progress:
            display.display_markdown(
                f"<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>Using {max_workers} parallel workers</span>",
                raw=True
            )
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_year = {
                executor.submit(process_single_year, corpus_dir, year, compress, downsample_threshold): year
                for year in existing_years
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_year):
                year = future_to_year[future]
                completed += 1
                
                try:
                    result = future.result()
                    results[year] = result
                    
                    if show_progress:
                        if "error" in result:
                            display.display_markdown(
                                f"<span style='font-family: monospace; font-size: 120%; font-weight: normal; color: red;'>{year}: FAILED ({completed}/{len(existing_years)})</span>",
                                raw=True
                            )
                        else:
                            triplets = result['triplet_count']
                            duration = result['duration']
                            display.display_markdown(
                                f"<span style='font-family: monospace; font-size: 120%; font-weight: normal; color: green;'>{year}: OK - {triplets:,} triplets ({duration:.1f}s) [{completed}/{len(existing_years)}]</span>",
                                raw=True
                            )
                            
                except Exception as e:
                    results[year] = {"error": str(e)}
                    if show_progress:
                        display.display_markdown(
                            f"<span style='font-family: monospace; font-size: 120%; font-weight: normal; color: red;'>{year}: ERROR - {e} [{completed}/{len(existing_years)}]</span>",
                            raw=True
                        )
    
    total_time = time.perf_counter() - start_time
    
    if show_progress:
        successful = [y for y, r in results.items() if "error" not in r]
        failed = [y for y, r in results.items() if "error" in r]
        
        summary_lines = [f"Completed in {total_time:.1f}s"]
        summary_lines.append(f"Successful: {len(successful)} years")
        if failed:
            summary_lines.append(f"Failed: {len(failed)} years")
        
        if successful:
            total_triplets = sum(results[y]['triplet_count'] for y in successful)
            avg_vocab = sum(results[y]['vocab_size'] for y in successful) // len(successful)
            triplets_per_sec = int(total_triplets / total_time) if total_time > 0 else 0
            summary_lines.append(f"Total triplets: {total_triplets:,}")
            summary_lines.append(f"Average vocab: {avg_vocab:,}")
            summary_lines.append(f"Triplets per second: {triplets_per_sec:,}")
        
        display.display_markdown(
            "<span style='font-family: monospace; font-size: 120%; font-weight: normal;'>Summary:<br><br>{}<br></span>".format(
                "<br>".join([f"- {line}" for line in summary_lines])
            ),
            raw=True
        )
    
    return results


def parse_years(year_range: str) -> List[str]:
    """Parse year range string into list of years."""
    years = []
    for part in year_range.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            start_year = int(start.strip())
            end_year = int(end.strip())
            years.extend([str(y) for y in range(start_year, end_year + 1)])
        else:
            years.append(part)
    return years


def get_available_years(corpus_dir: str) -> List[str]:
    """Get list of available years in corpus directory."""
    if not os.path.isdir(corpus_dir):
        return []
    
    years = []
    for filename in os.listdir(corpus_dir):
        if filename.endswith('.txt') and len(filename) == 8:  # YYYY.txt
            year = filename[:-4]
            if year.isdigit():
                years.append(year)
    
    return sorted(years)


# Simple API
def run_pipeline(corpus_dir: str, years: str, **kwargs):
    """
    Simple API to run the pipeline.
    
    Parameters
    ----------
    corpus_dir : str
        Directory containing corpus files
    years : str
        Years to process (e.g., "1680-1690")
    **kwargs
        Additional options (compress, max_workers, show_progress, downsample_threshold)
    """
    return process_year_range(corpus_dir, years, **kwargs)
