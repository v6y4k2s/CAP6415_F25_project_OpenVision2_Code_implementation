"""
Real dataset loader for Flickr8k image captioning.
Loads images and captions, applies preprocessing, and creates PyTorch DataLoaders.
"""

import os
import csv
import random
from typing import List, Tuple, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    from .tokenizer import SimpleTokenizer
except ImportError:
    # When running as a script
    from tokenizer import SimpleTokenizer


class Flickr8kDataset(Dataset):
    """
    Flickr8k dataset for image captioning.

    Dataset structure:
        data/
        ├── Images/              # Image directory
        │   ├── 1000268201_693b08cb0e.jpg
        │   └── ...
        └── captions.txt         # CSV file with image,caption columns
    """

    def __init__(
        self,
        data_dir: str,
        captions_file: str,
        tokenizer: SimpleTokenizer,
        max_caption_length: int = 50,
        split: str = "train",
        train_split: float = 0.8,
        val_split: float = 0.1,
        seed: int = 42,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            data_dir: Path to data directory (e.g., "data/")
            captions_file: Name of captions file (e.g., "captions.txt")
            tokenizer: Tokenizer instance for encoding captions
            max_caption_length: Maximum caption length (will pad/truncate)
            split: "train", "val", or "test"
            train_split: Fraction of data for training (default 0.8 = 80%)
            val_split: Fraction of data for validation (default 0.1 = 10%)
            seed: Random seed for reproducible splits
            transform: Image transformations (if None, uses default ViT preprocessing)
        """
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "Images")
        self.captions_path = os.path.join(data_dir, captions_file)
        self.tokenizer = tokenizer
        self.max_caption_length = max_caption_length
        self.split = split
        self.seed = seed

        # Load all image-caption pairs
        self.data = self._load_captions()

        # Split dataset into train/val/test
        self.data = self._split_dataset(self.data, train_split, val_split, seed, split)

        # Default transform for ViT (matches timm's preprocessing)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet stats
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transform

        print(f"{split.upper()} dataset: {len(self.data)} samples")

    def _load_captions(self) -> List[Tuple[str, str]]:
        """
        Load image-caption pairs from CSV file.

        Returns:
            List of (image_filename, caption) tuples
        """
        data = []

        with open(self.captions_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = row['image']
                caption = row['caption']
                data.append((image_name, caption))

        print(f"Loaded {len(data)} image-caption pairs from {self.captions_path}")
        return data

    def _split_dataset(
        self,
        data: List[Tuple[str, str]],
        train_split: float,
        val_split: float,
        seed: int,
        split: str,
    ) -> List[Tuple[str, str]]:
        """
        Split dataset into train/val/test sets.

        Important: Each image has 5 captions, so we split by IMAGE, not by caption.
        This ensures all captions for an image stay in the same split.

        Args:
            data: List of (image_name, caption) tuples
            train_split: Fraction for training
            val_split: Fraction for validation
            seed: Random seed
            split: Which split to return ("train", "val", or "test")

        Returns:
            Data for the requested split
        """
        # Group captions by image
        image_to_captions = {}
        for image_name, caption in data:
            if image_name not in image_to_captions:
                image_to_captions[image_name] = []
            image_to_captions[image_name].append(caption)

        # Get unique image names
        image_names = sorted(image_to_captions.keys())

        # Shuffle with seed for reproducibility
        random.seed(seed)
        random.shuffle(image_names)

        # Calculate split indices
        num_images = len(image_names)
        train_end = int(num_images * train_split)
        val_end = train_end + int(num_images * val_split)

        # Split images
        if split == "train":
            split_images = image_names[:train_end]
        elif split == "val":
            split_images = image_names[train_end:val_end]
        elif split == "test":
            split_images = image_names[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        # Collect all captions for images in this split
        split_data = []
        for image_name in split_images:
            for caption in image_to_captions[image_name]:
                split_data.append((image_name, caption))

        return split_data

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            image: Tensor of shape (3, 224, 224)
            caption_tokens: Tensor of shape (max_caption_length,)
        """
        image_name, caption_text = self.data[idx]

        # Load and preprocess image
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Tokenize caption
        caption_tokens = self.tokenizer.encode(
            caption_text,
            max_length=self.max_caption_length,
            add_special_tokens=True,  # Adds START and END tokens
        )

        # Pad to max_caption_length
        caption_tokens = self.tokenizer.pad_sequence(caption_tokens, self.max_caption_length)

        # Convert to tensor
        caption_tokens = torch.tensor(caption_tokens, dtype=torch.long)

        return image, caption_tokens


def create_flickr8k_dataloaders(
    data_dir: str = "data",
    captions_file: str = "captions.txt",
    vocab_size: int = 10000,
    min_freq: int = 2,
    max_caption_length: int = 50,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, SimpleTokenizer]:
    """
    Create train/val/test DataLoaders for Flickr8k dataset.

    Args:
        data_dir: Path to data directory
        captions_file: Name of captions CSV file
        vocab_size: Maximum vocabulary size
        min_freq: Minimum word frequency to include in vocab
        max_caption_length: Maximum caption length
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for data loading
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        seed: Random seed for reproducibility

    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set
        tokenizer: Fitted tokenizer instance
    """
    print("=" * 60)
    print("Creating Flickr8k DataLoaders")
    print("=" * 60)

    # Step 1: Load all captions to build vocabulary
    captions_path = os.path.join(data_dir, captions_file)
    all_captions = []

    with open(captions_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_captions.append(row['caption'])

    # Step 2: Build tokenizer on all captions (or just training captions)
    # Note: For proper evaluation, you should only build vocab on training captions
    # But for simplicity in learning, we'll use all captions
    tokenizer = SimpleTokenizer(vocab_size=vocab_size, min_freq=min_freq)
    tokenizer.build_vocab(all_captions)

    print("\n" + "=" * 60)
    print("Creating datasets...")
    print("=" * 60)

    # Step 3: Create datasets
    train_dataset = Flickr8kDataset(
        data_dir=data_dir,
        captions_file=captions_file,
        tokenizer=tokenizer,
        max_caption_length=max_caption_length,
        split="train",
        train_split=train_split,
        val_split=val_split,
        seed=seed,
    )

    val_dataset = Flickr8kDataset(
        data_dir=data_dir,
        captions_file=captions_file,
        tokenizer=tokenizer,
        max_caption_length=max_caption_length,
        split="val",
        train_split=train_split,
        val_split=val_split,
        seed=seed,
    )

    test_dataset = Flickr8kDataset(
        data_dir=data_dir,
        captions_file=captions_file,
        tokenizer=tokenizer,
        max_caption_length=max_caption_length,
        split="test",
        train_split=train_split,
        val_split=val_split,
        seed=seed,
    )

    # Step 4: Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("\n" + "=" * 60)
    print("DataLoaders created successfully!")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print("=" * 60)

    return train_loader, val_loader, test_loader, tokenizer


# Demo/test code
if __name__ == "__main__":
    print("Testing Flickr8k dataset loader...")

    # Create dataloaders
    train_loader, val_loader, test_loader, tokenizer = create_flickr8k_dataloaders(
        data_dir="data",
        vocab_size=5000,
        batch_size=4,
        num_workers=0,  # Use 0 for debugging
    )

    # Test: Load one batch
    print("\nTesting batch loading...")
    images, captions = next(iter(train_loader))

    print(f"Image batch shape: {images.shape}")  # Expected: (4, 3, 224, 224)
    print(f"Caption batch shape: {captions.shape}")  # Expected: (4, 50)

    # Decode first caption
    print("\nFirst caption (tokenized):", captions[0].tolist()[:20])
    print("First caption (decoded):", tokenizer.decode(captions[0].tolist()))

    print("\nDataset test passed!")
