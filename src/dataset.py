"""
Dataset module for OpenVision Student

This module provides a fake data generator for educational purposes.
Later, this can be extended to load real image-caption datasets.
"""

import torch
from torch.utils.data import Dataset


class FakeImageCaptionDataset(Dataset):
    """
    Fake dataset that generates random images and captions.

    This is useful for:
    1. Testing the training pipeline without downloading real data
    2. Verifying that the model can overfit on small data
    3. Educational purposes - understanding data shapes

    Args:
        num_samples: Number of fake samples to generate
        image_size: Size of square images (default: 224 for ViT)
        caption_length: Maximum length of captions (default: 20 tokens)
        vocab_size: Size of vocabulary (default: 10000 words)

    Returns:
        Dictionary with:
        - 'image': Tensor of shape (3, 224, 224) - random RGB image
        - 'caption': Tensor of shape (caption_length,) - random token IDs
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = 224,
        caption_length: int = 20,
        vocab_size: int = 10000,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.caption_length = caption_length
        self.vocab_size = vocab_size

    def __len__(self):
        """Return the total number of samples"""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generate one fake sample.

        Note: We use the same random seed based on idx to ensure
        reproducibility - the same idx will always return the same data.
        """
        # Use idx as seed for reproducibility
        generator = torch.Generator().manual_seed(idx)

        # Generate random image: (3, 224, 224)
        # Values in [0, 1] range (typical for pretrained ViT)
        image = torch.rand(
            3, self.image_size, self.image_size,
            generator=generator
        )

        # Generate random caption tokens: (caption_length,)
        # Token IDs in range [0, vocab_size)
        caption = torch.randint(
            0, self.vocab_size,
            (self.caption_length,),
            generator=generator
        )

        return {
            'image': image,
            'caption': caption,
        }


def create_dataloaders(
    train_samples: int = 80,
    val_samples: int = 20,
    batch_size: int = 4,
    vocab_size: int = 10000,
    caption_length: int = 20,
    num_workers: int = 0,
):
    """
    Create train and validation dataloaders.

    Args:
        train_samples: Number of training samples
        val_samples: Number of validation samples
        batch_size: Batch size for both train and val
        vocab_size: Vocabulary size
        caption_length: Maximum caption length
        num_workers: Number of workers for data loading (0 = main thread)

    Returns:
        tuple: (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader

    # Create datasets
    train_dataset = FakeImageCaptionDataset(
        num_samples=train_samples,
        vocab_size=vocab_size,
        caption_length=caption_length,
    )

    val_dataset = FakeImageCaptionDataset(
        num_samples=val_samples,
        vocab_size=vocab_size,
        caption_length=caption_length,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for training
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# Example usage and testing
if __name__ == "__main__":
    print("Testing FakeImageCaptionDataset...")
    print("-" * 50)

    # Create a small dataset
    dataset = FakeImageCaptionDataset(num_samples=10)

    print(f"Dataset size: {len(dataset)}")

    # Get first sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Caption shape: {sample['caption'].shape}")
    print(f"  Caption tokens (first 10): {sample['caption'][:10].tolist()}")

    # Verify reproducibility
    sample2 = dataset[0]
    same = torch.allclose(sample['image'], sample2['image'])
    print(f"\nReproducibility check: {same}")

    # Test dataloaders
    print("\n" + "-" * 50)
    print("Testing DataLoaders...")
    print("-" * 50)

    train_loader, val_loader = create_dataloaders(
        train_samples=16,
        val_samples=8,
        batch_size=4,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Get first batch
    batch = next(iter(train_loader))
    print(f"\nFirst training batch:")
    print(f"  Images shape: {batch['image'].shape}")
    print(f"  Captions shape: {batch['caption'].shape}")

    print("\n Dataset module working correctly!")
