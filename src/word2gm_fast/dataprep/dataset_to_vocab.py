import json
import tensorflow as tf
from pathlib import Path
from termcolor import colored

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
    The UNK token is always 'UNK' and must be at index 0.
    """
    # Ensure UNK is present and at index 0
    if vocab_list[0] != "UNK":
        raise ValueError(f"UNK token must be at index 0, got {vocab_list[0]}")
    keys = tf.constant(vocab_list, dtype=tf.string)
    values = tf.range(len(vocab_list), dtype=tf.int32)
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=0  # UNK_ID
    )

def make_vocab(
    dataset: tf.data.Dataset,
    vocab_path: str,
    save: bool = True,
    overwrite: bool = False
) -> list[str]:
    """
    Build a vocabulary from a tf.data.Dataset of lines.

    The UNK token is always 'UNK' and will be at index 0. All other tokens
    are sorted alphabetically. Assumes all preprocessing (e.g., filtering
    infrequent tokens) is done upstream.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Dataset of lines (strings) to build the vocab from.
    vocab_path : str
        Path to save the vocab JSON file.
    save : bool, optional
        Whether to save the vocab to disk (default: True).
    overwrite : bool, optional
        Whether to overwrite the vocab file if it already exists (default: False).

    Returns
    -------
    list[str]
        The vocabulary list (with 'UNK' at index 0).
    """
    print(colored("🔍 Scanning dataset for vocab...", "cyan"))
    vocab_set = set()
    for line in dataset.as_numpy_iterator():
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        tokens = line.strip().split()
        vocab_set.update(tokens)

    # Ensure 'UNK' is index 0 and appears only once
    vocab = ["UNK"] + sorted(tok for tok in vocab_set if tok != "UNK")

    print(colored(f"🧾 Vocab size: {len(vocab)}", "green"))

    if save:
        save_vocab(vocab, vocab_path, overwrite=overwrite)

    return vocab
