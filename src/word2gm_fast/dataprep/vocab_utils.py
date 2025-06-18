import json
import collections
import tensorflow as tf
from pathlib import Path
from termcolor import colored

UNK_TOKEN = "UNK"

def load_vocab(vocab_path: str) -> list[str]:
    """
    Load a vocab list from a JSON file.
    Assumes index order in the list matches word ID.
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_vocab(vocab_list: list[str], output_path: str, overwrite: bool = True):
    """
    Save vocab list to JSON file.

    Parameters
    ----------
    vocab_list : list of str
        The vocabulary to save.
    output_path : str
        Path to the output JSON file.
    overwrite : bool
        Whether to overwrite an existing file. Defaults to True.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(colored(f"⚠️  File exists and overwrite is False: {output_path}", "red"))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_list, f, indent=2, ensure_ascii=False)
    print(colored(f"💾 Vocab saved to: {output_path}", "yellow"))

def build_vocab_table(vocab_list: list[str]) -> tf.lookup.StaticHashTable:
    """
    Builds a StaticHashTable mapping words (strings) to integer IDs.
    Assumes vocab_list is ordered by ID (i.e., index = ID).
    """
    keys = tf.constant(vocab_list, dtype=tf.string)
    values = tf.range(len(vocab_list), dtype=tf.int32)
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=0  # UNK_ID
    )

def build_vocab(corpus_path, vocab_path, min_count=5, save=True, overwrite=False):
    """
    Build a vocabulary from a corpus.

    Parameters
    ----------
    corpus_path : str
        Path to the corpus file.
    vocab_path : str
        Where to save the vocab JSON file.
    min_count : int
        Minimum frequency to include a token.
    save : bool
        Whether to save the vocab to disk.
    overwrite : bool
        Whether to overwrite the vocab file if it already exists.

    Returns
    -------
    list[str]
        The vocabulary list (with UNK at index 0).
    """
    token_counter = collections.Counter()

    print(colored(f"🔍 Scanning corpus: {corpus_path}", "cyan"))
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            token_counter.update(tokens)

    # Filter out low-frequency tokens
    filtered = {tok: count for tok, count in token_counter.items() if count >= min_count}

    # Sort by frequency descending
    sorted_tokens = sorted(filtered, key=lambda t: -filtered[t])

    # Ensure UNK is index 0 and appears only once
    vocab = [UNK_TOKEN] + [tok for tok in sorted_tokens if tok != UNK_TOKEN]

    print(colored(f"🧾 Vocab size (min_count={min_count}): {len(vocab)}", "green"))

    if save:
        save_vocab(vocab, vocab_path, overwrite=overwrite)

    return vocab
