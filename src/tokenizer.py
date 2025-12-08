"""
Tokenizer for converting text captions to token sequences.
Builds vocabulary from training captions and handles special tokens.
"""

import re
from collections import Counter
from typing import List, Dict


class SimpleTokenizer:
    """
    Simple word-level tokenizer for image captions.

    Special tokens:
    - PAD (0): Padding for variable-length sequences
    - START (1): Start of caption marker
    - END (2): End of caption marker
    - UNK (3): Unknown words (not in vocabulary)
    """

    # Special token definitions (matching src/inference.py)
    PAD_TOKEN = "<PAD>"
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    UNK_TOKEN = "<UNK>"

    PAD_ID = 0
    START_ID = 1
    END_ID = 2
    UNK_ID = 3

    def __init__(self, vocab_size=10000, min_freq=1):
        """
        Args:
            vocab_size: Maximum vocabulary size (including special tokens)
            min_freq: Minimum frequency for a word to be included in vocab
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq

        # Will be populated by build_vocab()
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()

        # Initialize with special tokens
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Initialize special tokens in vocabulary."""
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_ID,
            self.START_TOKEN: self.START_ID,
            self.END_TOKEN: self.END_ID,
            self.UNK_TOKEN: self.UNK_ID,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def basic_tokenize(self, text: str) -> List[str]:
        """
        Convert text to list of words (basic tokenization).

        Args:
            text: Raw caption string

        Returns:
            List of lowercase words
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation except apostrophes (to keep "don't", "it's", etc.)
        text = re.sub(r"[^\w\s']", "", text)

        # Split on whitespace
        words = text.split()

        return words

    def build_vocab(self, captions: List[str]):
        """
        Build vocabulary from list of captions.

        Args:
            captions: List of caption strings
        """
        print(f"Building vocabulary from {len(captions)} captions...")

        # Count all words
        for caption in captions:
            words = self.basic_tokenize(caption)
            self.word_counts.update(words)

        print(f"Found {len(self.word_counts)} unique words")

        # Get most common words (accounting for special tokens already in vocab)
        # vocab_size includes 4 special tokens, so we need vocab_size - 4 regular words
        num_regular_words = self.vocab_size - 4
        most_common = self.word_counts.most_common(num_regular_words)

        # Filter by minimum frequency
        most_common = [(word, count) for word, count in most_common if count >= self.min_freq]

        # Add to vocabulary (starting from index 4, after special tokens)
        next_idx = 4
        for word, count in most_common:
            if word not in self.word2idx:  # Skip if already exists
                self.word2idx[word] = next_idx
                self.idx2word[next_idx] = word
                next_idx += 1

        actual_vocab_size = len(self.word2idx)
        print(f"Final vocabulary size: {actual_vocab_size}")
        print(f"Words mapped to <UNK>: {len(self.word_counts) - (actual_vocab_size - 4)}")

        # Update vocab_size to actual size
        self.vocab_size = actual_vocab_size

    def encode(self, text: str, max_length: int = None, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to sequence of token IDs.

        Args:
            text: Caption string
            max_length: Maximum sequence length (including START/END if added)
            add_special_tokens: Whether to add START and END tokens

        Returns:
            List of token IDs
        """
        # Tokenize
        words = self.basic_tokenize(text)

        # Convert to IDs (use UNK for unknown words)
        token_ids = [self.word2idx.get(word, self.UNK_ID) for word in words]

        # Add special tokens
        if add_special_tokens:
            token_ids = [self.START_ID] + token_ids + [self.END_ID]

        # Truncate if needed
        if max_length is not None and len(token_ids) > max_length:
            if add_special_tokens:
                # Keep START, truncate middle, keep END
                token_ids = [self.START_ID] + token_ids[1:max_length-1] + [self.END_ID]
            else:
                token_ids = token_ids[:max_length]

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert sequence of token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip PAD, START, END tokens

        Returns:
            Decoded caption string
        """
        words = []
        special_ids = {self.PAD_ID, self.START_ID, self.END_ID}

        for idx in token_ids:
            if skip_special_tokens and idx in special_ids:
                continue

            word = self.idx2word.get(idx, self.UNK_TOKEN)
            words.append(word)

        return " ".join(words)

    def pad_sequence(self, token_ids: List[int], max_length: int) -> List[int]:
        """
        Pad sequence to max_length with PAD tokens.

        Args:
            token_ids: List of token IDs
            max_length: Target length

        Returns:
            Padded sequence of length max_length
        """
        if len(token_ids) >= max_length:
            return token_ids[:max_length]
        else:
            padding = [self.PAD_ID] * (max_length - len(token_ids))
            return token_ids + padding

    def __len__(self):
        """Return vocabulary size."""
        return self.vocab_size

    def save(self, filepath: str):
        """Save tokenizer vocabulary to file."""
        import json

        data = {
            "vocab_size": self.vocab_size,
            "min_freq": self.min_freq,
            "word2idx": self.word2idx,
            "word_counts": dict(self.word_counts),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Tokenizer saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load tokenizer from file."""
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data["vocab_size"], min_freq=data["min_freq"])
        tokenizer.word2idx = data["word2idx"]
        tokenizer.idx2word = {v: k for k, v in tokenizer.word2idx.items()}  # Reverse mapping: ID -> word
        tokenizer.word_counts = Counter(data["word_counts"])

        print(f"Tokenizer loaded from {filepath}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")

        return tokenizer


# Demo/test code
if __name__ == "__main__":
    # Example captions
    example_captions = [
        "A child in a pink dress is climbing up a set of stairs.",
        "A black dog and a spotted dog are fighting.",
        "A little girl climbing into a wooden playhouse.",
        "Two dogs of different breeds looking at each other on the road.",
    ]

    # Build vocabulary
    tokenizer = SimpleTokenizer(vocab_size=100, min_freq=1)
    tokenizer.build_vocab(example_captions)

    # Test encoding
    test_caption = "A little girl in a pink dress."
    print(f"\nOriginal: {test_caption}")

    encoded = tokenizer.encode(test_caption, max_length=20)
    print(f"Encoded: {encoded}")

    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    # Test padding
    padded = tokenizer.pad_sequence(encoded, max_length=25)
    print(f"Padded to 25: {padded}")

    # Show vocabulary
    print(f"\nVocabulary size: {len(tokenizer)}")
    print(f"Sample words: {list(tokenizer.word2idx.keys())[:20]}")
