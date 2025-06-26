# Word2GM-Fast: GPU-Friendly TensorFlow Port

A modern, GPU-accelerated TensorFlow 2.x implementation of Word2GM (Word to Gaussian Mixture) embeddings, based on the original [benathi/word2gm](https://github.com/benathi/word2gm) repository.

## Overview

Word2GM represents each word as a **Gaussian Mixture Model** instead of a single point vector, enabling:
- **Multimodal representations**: Words with multiple meanings (e.g., "bank" → financial + riverbank)
- **Uncertainty modeling**: Capture confidence and variability in embeddings
- **Richer semantic relationships**: Better model entailment, similarity, and polysemy

Based on the paper: ["Multimodal Word Distributions"](https://arxiv.org/abs/1704.08424) by Athiwaratkun and Wilson (ACL 2017).

## Features

✅ **GPU-Accelerated Training**: Full TensorFlow 2.x implementation with GPU support  
✅ **TFRecord Pipeline**: Efficient data loading and preprocessing  
✅ **Scalable Architecture**: Handles large vocabularies and corpora  
✅ **Modern Codebase**: Clean, documented, and maintainable Python  
✅ **Multiprocessing**: CPU-optimized data preparation  
✅ **Interactive Notebooks**: Complete training and evaluation workflows  

## Project Structure

```
word2gm-fast/
├── src/word2gm_fast/
│   ├── dataprep/           # Data preprocessing pipeline
│   │   ├── pipeline.py     # Main data preparation pipeline
│   │   └── tfrecord_io.py  # TFRecord I/O utilities
│   ├── models/             # Word2GM model implementation
│   │   ├── word2gm_model.py  # Main model architecture
│   │   └── config.py       # Configuration classes
│   ├── training/           # Training utilities
│   │   ├── training_utils.py # Training step and utilities
│   │   └── train_word2gm.py  # Training script
│   └── utils/              # Utility modules
│       └── tf_silence.py   # TensorFlow configuration
├── notebooks/
│   ├── prepare_training_dataset.ipynb  # Data preparation
│   └── train_word2gm.ipynb            # GPU training & evaluation
└── README.md
```

## Quick Start

### 1. Data Preparation (CPU-Optimized)

Use the data preparation notebook to process text corpora into TFRecord training artifacts:

```bash
# Open the data preparation notebook
jupyter notebook notebooks/prepare_training_dataset.ipynb
```

This creates year-specific artifact directories:
```
corpus_data/
├── 2019.txt
├── 2019_artifacts/
│   ├── triplets.tfrecord
│   └── vocab.tfrecord
```

### 2. Model Training (GPU-Accelerated)

Train the Word2GM model using GPU acceleration:

```bash
# Open the training notebook
jupyter notebook notebooks/train_word2gm.ipynb
```

Or use the standalone training script:
```bash
python src/word2gm_fast/training/train_word2gm.py
```

### 3. Model Usage

```python
from word2gm_fast.models.word2gm_model import Word2GMModel
from word2gm_fast.models.config import Word2GMConfig

# Load trained model
config = Word2GMConfig(vocab_size=vocab_size)
model = Word2GMModel(config)
model.load_weights("path/to/saved/model")

# Get word embedding (mixture-weighted mean)
embedding = model.get_word_embedding(word_id)

# Find nearest neighbors
neighbors = model.get_nearest_neighbors(word_id, k=10)

# Component-specific neighbors (for polysemy analysis)
bank_financial = model.get_nearest_neighbors(bank_id, k=5, component=0)
bank_geography = model.get_nearest_neighbors(bank_id, k=5, component=1)
```

## Model Architecture

Each word `w` is represented as a Gaussian Mixture with `K` components:
- **Means (μ)**: `[vocab_size, K, embedding_dim]` - Component centers
- **Covariances (Σ)**: `[vocab_size, K, embedding_dim]` - Component variances (spherical or diagonal)
- **Mixture Weights (π)**: `[vocab_size, K]` - Component probabilities

**Training Objective**: Max-margin loss using Expected Likelihood Kernel similarity.

## Configuration

Key hyperparameters (see `src/word2gm_fast/models/config.py`):

```python
config = Word2GMConfig(
    vocab_size=50000,        # Vocabulary size
    embedding_size=50,       # Embedding dimensionality  
    num_mixtures=2,          # Number of Gaussian components
    spherical=True,          # Spherical vs diagonal covariance
    learning_rate=0.05,      # Learning rate
    batch_size=128,          # Training batch size
    epochs_to_train=10,      # Number of epochs
    adagrad=True,           # Use Adagrad optimizer
    normclip=True,          # Enable parameter clipping
)
```

## Performance Optimizations

- **GPU Memory Growth**: Dynamic GPU memory allocation
- **TFRecord Streaming**: Efficient data loading with prefetching
- **Batched Processing**: Optimized vocabulary operations (12.6x speedup)
- **Multiprocessing**: Parallel data preprocessing
- **Mixed Precision**: Optional for larger models (configurable)

## Requirements

- Python 3.8+
- TensorFlow 2.8+
- NumPy, Pandas, Matplotlib
- Optional: scikit-learn (for t-SNE visualization)

## GPU Support

The implementation is optimized for NVIDIA GPUs with CUDA support:
- Automatic GPU detection and configuration
- Memory growth to prevent OOM errors
- CPU fallback for data preprocessing
- Multi-GPU support (configurable)

## Evaluation & Visualization

The training notebook includes comprehensive evaluation:
- **Parameter Statistics**: Mean norms, variances, mixture weights
- **Nearest Neighbors**: Word similarity using expected likelihood kernel
- **Polysemy Analysis**: Component-specific neighbors for ambiguous words
- **t-SNE Visualization**: 2D embeddings of mixture components
- **Interactive Exploration**: Query tool for word analysis

## Differences from Original

Key improvements over the original benathi/word2gm:

1. **Modern TensorFlow 2.x**: Native eager execution, tf.function decorators
2. **GPU Acceleration**: Full GPU support with memory optimization
3. **TFRecord Pipeline**: Efficient data format for large-scale training
4. **Clean Architecture**: Modular, documented, and extensible code
5. **Jupyter Integration**: Interactive notebooks for exploration
6. **Performance**: Optimized data loading and training loops

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@InProceedings{athiwilson2017,
  author = {Ben Athiwaratkun and Andrew Gordon Wilson},
  title = {Multimodal Word Distributions},
  booktitle = {Conference of the Association for Computational Linguistics (ACL)},
  year = {2017}
}
```

## License

This project maintains compatibility with the original BSD-3-Clause license.

## Contributing

Contributions are welcome! Areas for improvement:
- Mixed precision training support
- Distributed training for multi-GPU setups
- Additional evaluation metrics and benchmarks
- Integration with HuggingFace transformers
- Pre-trained model releases
