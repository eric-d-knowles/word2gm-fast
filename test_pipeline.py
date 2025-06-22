#!/usr/bin/env python3
"""
Test script for the new data preparation pipeline module.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.word2gm_fast.dataprep.pipeline import prepare_training_data, get_corpus_years

def test_pipeline():
    """Test the pipeline with a sample corpus file"""
    
    corpus_dir = "/vast/edk202/NLP_corpora/Google_Books/20200217/eng-fiction/5gram_files/6corpus/yearly_files/data"
    
    # Test getting available years
    print("🔍 Discovering available corpus years...")
    years = get_corpus_years(corpus_dir)
    print(f"Available years: {', '.join(years[:10])}{'...' if len(years) > 10 else ''}")
    
    if years:
        # Test with the first available year
        test_year = years[0]
        corpus_file = f"{test_year}.txt"
        
        print(f"\n🧪 Testing pipeline with {corpus_file}...")
        
        try:
            output_dir, summary = prepare_training_data(
                corpus_file=corpus_file,
                corpus_dir=corpus_dir,
                output_subdir=f"{test_year}_test_artifacts",
                compress=True,
                show_progress=True,
                cache_dataset=True
            )
            
            print("\n✅ Pipeline test successful!")
            print(f"Artifacts saved to: {output_dir}")
            
        except Exception as e:
            print(f"\n❌ Pipeline test failed: {e}")
    else:
        print("No corpus files found for testing")

if __name__ == "__main__":
    test_pipeline()
