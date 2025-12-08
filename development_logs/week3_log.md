# Week 3 Development Log - OpenVision 2 Project
**Date Range:** November 25 - December 1, 2025
**Student:** Sumedh
**Focus:** Training Pipeline, Dataset Management & Inference System

---

## Overview
This week transformed the prototype into a production-ready training system. Implemented proper dataset management, complete training/validation loops, model checkpointing, and inference capabilities.

## Tasks Completed

### 1. Project Restructuring (Nov 25-26)
- **Package Organization:**
  - Created src/__init__.py to make src/ a proper Python package
  - Organized imports for easy access to all model components
  - Enabled clean imports: `from src import OpenVisionStudent`

- **Directory Structure:**
  - Created notebooks/ directory for Jupyter experiments
  - Created assets/ directory for documentation images
  - Set up checkpoints/ directory for model saves
  - Organized codebase for scalability

### 2. Dataset Module Implementation (Nov 26-27)
- **FakeImageCaptionDataset Class:**
  - Built custom PyTorch Dataset for reproducible random data
  - Generates random images: (3, 224, 224) in [0, 1] range
  - Generates random caption tokens with configurable vocab size
  - Uses index-based seeds for reproducibility across runs
  - Useful for testing pipeline before real data integration

- **DataLoader Factory:**
  - Implemented `create_dataloaders()` function
  - Creates separate train and validation datasets
  - Handles batching, shuffling, and parallel loading
  - Returns tuple: (train_loader, val_loader)
  - Configurable batch size and number of workers

- **Technical Details:**
  - Implemented `__len__()` to return dataset size
  - Implemented `__getitem__(idx)` to return (image, caption)
  - Ensured reproducibility with deterministic random seeds

### 3. Complete Training Script (Nov 27-29)
- **train_one_epoch() Function:**
  - Iterates through batches with tqdm progress bars
  - Performs forward pass → compute loss → backward → optimize
  - Applies 50% visual token masking during training
  - Tracks and returns average epoch loss
  - Real-time loss monitoring per batch

- **validate() Function:**
  - Uses `torch.no_grad()` context for memory efficiency
  - No gradient computation (faster, uses less memory)
  - No visual masking (uses all 197 tokens)
  - Monitors validation loss to detect overfitting
  - Returns average validation loss

- **main() Function - Complete Pipeline:**
  - Automatic device selection (CUDA/MPS/CPU)
  - Model initialization with parameter counting
  - Training loop with alternating train/validation
  - Best model checkpointing based on validation loss
  - Final summary with loss progression table
  - Epoch timing and performance metrics

- **Command-Line Interface:**
  - Implemented argparse for full control
  - Data parameters: train-samples, val-samples, batch-size, vocab-size
  - Model parameters: text-dim, num-heads, num-decoder-layers, max-seq-len
  - Training parameters: epochs, lr, seed
  - Output parameters: save-model, output-dir
  - Help text for all arguments

### 4. Training Execution & Results (Nov 29)
- **Test Run Configuration:**
  - Training samples: 16 (fake data)
  - Validation samples: 8 (fake data)
  - Batch size: 4
  - Epochs: 3
  - Device: Apple Silicon GPU (MPS)

- **Model Statistics:**
  - Total parameters: 121.7M
  - Frozen parameters: 85.8M (ViT encoder)
  - Trainable parameters: 35.9M (projection + decoder)

- **Training Results:**
  ```
  Epoch | Train Loss | Val Loss
  ------|------------|----------
      1 |     9.3967 |   8.8529
      2 |     8.7243 |   8.5501
      3 |     8.2313 |   8.1280 ← best
  ```

- **Checkpoint:**
  - Saved best_model.pt (774MB)
  - Stored epoch, model state, optimizer state, losses
  - Can resume training or load for inference

### 5. Architecture Verification Notebook (Nov 30)
- **Created architecture_check.ipynb:**
  - Step-by-step verification of all components
  - Tests Phase 1: Visual Encoder
  - Tests Phase 2: Projection + Decoder
  - Tests Phase 3: Forward pass and loss
  - Educational tool for understanding pipeline
  - Interactive visualization of tensor shapes

### 6. Inference Module Implementation (Dec 1)
- **generate_caption() Function:**
  - Autoregressive generation for single image
  - Greedy decoding: argmax at each step
  - Process: [START] → predict → append → repeat
  - Stops at [END] token or max_length
  - No visual masking (uses all 197 tokens)
  - Temperature control for sampling randomness

- **generate_caption_batch() Function:**
  - Batch inference (processes images sequentially)
  - Returns list of caption token sequences

- **tokens_to_text() Function:**
  - Converts token IDs to human-readable text
  - Requires vocabulary mapping (word_to_id dict)
  - Placeholder for future tokenizer integration

- **Demo Script:**
  - Loads checkpoint from file
  - Generates captions for test images
  - Tests different temperature values (0.8, 1.0, 1.5)
  - Displays generated captions

- **Test Results:**
  - Model loaded successfully (Epoch 2, Val Loss: 8.13)
  - Generated captions: Repetitive token (5567)
  - Expected behavior: Model trained on random data
  - Validates inference pipeline works correctly

## Challenges Faced

1. **DataLoader Configuration:**
   - Initial issues with batch collation
   - **Resolution:** Used default_collate, ensured consistent tensor shapes

2. **Device Management:**
   - MPS (Apple Silicon GPU) compatibility issues
   - **Resolution:** Added device checks, tested on CPU as fallback

3. **Checkpoint Size:**
   - 774MB checkpoint file (large)
   - **Resolution:** Acceptable for educational project, documented in logs

4. **Inference Output Quality:**
   - Repetitive captions from fake data training
   - **Resolution:** Expected behavior, validates architecture; need real data

5. **Progress Tracking:**
   - Wanted real-time training feedback
   - **Resolution:** Integrated tqdm for progress bars

## Code Statistics
- Files created: 4 (dataset.py, train.py, inference.py, __init__.py)
- Total lines: ~680
- Functions implemented: 8+ major functions
- Notebook cells: 15+ (architecture verification)

## Next Week Goals
1. Implement real dataset loader (Flickr8k)
2. Build vocabulary and tokenizer from real captions
3. Train on actual image-caption pairs
4. Evaluate caption quality on real data
5. Generate meaningful captions for test images

## Learning Outcomes
- PyTorch Dataset and DataLoader design patterns
- Training/validation loop best practices
- Model checkpointing strategies
- Autoregressive text generation
- Temperature-based sampling
- Production-ready code organization

## Time Spent
- Project restructuring: 3 hours
- Dataset implementation: 6 hours
- Training script: 10 hours
- Training execution: 4 hours
- Inference implementation: 6 hours
- Testing and debugging: 5 hours
- **Total:** ~34 hours

## Technical Insights

### Loss Interpretation:
- Initial loss ~9.4 ≈ -log(1/10000) for random predictions
- Decreasing loss shows model learning patterns
- Validation loss > train loss is healthy (no overfitting yet)

### Visual Masking Impact:
- Training: 50% masking → robustness
- Validation: 0% masking → full context
- Mimics data augmentation strategy

### Inference Behavior:
- Repetitive output expected from random data
- Validates pipeline functionality
- Real data needed for meaningful captions

---

