"""
DEPRECATED: TFRecord I/O utilities have been moved to the io package.

This module has been split into focused modules in the io package:
- word2gm_fast.io.vocab: Vocabulary TFRecord operations
- word2gm_fast.io.triplets: Triplets TFRecord operations  
- word2gm_fast.io.tables: Lookup table creation
- word2gm_fast.io.artifacts: Pipeline artifacts management

Please update your imports to use the new modules directly.
"""

import warnings

warnings.warn(
    "The tfrecord_io module has been deprecated and split into focused modules "
    "in the io package. Please update your imports to use the new modules directly: "
    "word2gm_fast.io.vocab, word2gm_fast.io.triplets, word2gm_fast.io.tables, "
    "word2gm_fast.io.artifacts",
    DeprecationWarning,
    stacklevel=2
)