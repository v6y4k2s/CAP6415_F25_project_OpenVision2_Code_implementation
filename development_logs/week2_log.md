# Week 2 Development Log - OpenVision 2 Project
**Date Range:** November 18-24, 2025
**Student:** Sumedh
**Focus:** Bridge Layer, Decoder Implementation & Loss Function

---

## Overview
This week focused on building the core components connecting visual and textual modalities: the projection layer and causal decoder. Additionally, implemented the complete forward pass and loss computation for the model.

## Tasks Completed

### 1. Projection Layer Implementation (Nov 18-19)
- **Component Development:**
  - Implemented `ProjectionLayer` class for dimensional transformation
  - Built learned linear transformation: 768 → 512 dimensions
  - Used nn.Linear(768, 512) with 393,216 trainable parameters

- **Technical Understanding:**
  - Learned that projection performs lossy but task-aware compression
  - Not all ViT features are relevant for caption generation
  - Layer learns to preserve caption-relevant information
  - Acts as regularization and reduces decoder computational cost

- **Testing:**
  - Verified input shape: (batch, 197, 768)
  - Confirmed output shape: (batch, 197, 512)
  - Tested gradient flow through projection layer

### 2. Causal Decoder Implementation (Nov 20-21)
- **Component Development:**
  - Implemented `CausalDecoder` class using GPT-style architecture
  - Built using PyTorch's nn.TransformerDecoder
  - Created text embedding layer for caption tokens
  - Implemented positional encoding for sequence ordering
  - Created causal mask generation function

- **Technical Details:**
  - Decoder configuration: 6 layers, 8 attention heads
  - Text dimension: 512, matching projected visual tokens
  - Vocabulary size: 10,000 (for initial testing)
  - Causal mask uses -inf (not 0) because softmax(-inf) = 0

- **Key Concepts Learned:**
  - Autoregressive generation: predict one word at a time
  - Causal masking: position i can only see positions 0 to i-1
  - Cross-attention: decoder attends to all visual tokens
  - Self-attention: decoder attends to previous words only

### 3. Visual Token Masking (Nov 21)
- **Implementation:**
  - Created `apply_visual_token_masking()` function
  - Randomly masks 50% of patch tokens during training
  - Keeps [CLS] token intact (index 0)
  - Zeroes out masked positions

- **Understanding:**
  - OpenVision 2 innovation for robustness
  - Forces model to work with incomplete visual information
  - Prevents overfitting to specific visual patterns
  - Used only during training, not at inference

### 4. Complete Model Integration (Nov 22-24)
- **OpenVisionStudent Class:**
  - Built unified model class combining all components
  - Implemented forward() method for end-to-end inference
  - Implemented compute_loss() for training objective

- **Forward Pass Pipeline:**
  1. Visual Encoder: Extract 197 visual tokens (768-dim)
  2. Projection Layer: Transform to 512-dim
  3. Visual Masking: Randomly mask 50% (training only)
  4. Causal Decoder: Generate caption logits
  5. Output: (batch, caption_length, vocab_size)

- **Loss Function:**
  - Implemented Causal Language Modeling (CLM) objective
  - Next-token prediction task (same as GPT)
  - Input: caption_tokens[:, :-1] (all but last word)
  - Target: caption_tokens[:, 1:] (all but first, shifted)
  - Loss: Cross-entropy between predictions and targets

- **Training Verification:**
  - Created minimal training loop (10 steps)
  - Tested on 4 dummy image-caption pairs
  - **Results:**
    - Initial loss: 9.3852 (random predictions)
    - Final loss: 0.1034 (nearly perfect memorization)
  - Confirmed model can learn (overfitting on small data is expected)

## Challenges Faced

1. **Causal Masking Confusion:**
   - Initially unclear why use -inf instead of 0
   - **Resolution:** Understood that softmax(-inf) = 0, completely blocking attention

2. **Decoder Architecture:**
   - Confused between encoder-decoder and decoder-only models
   - Unclear how visual tokens serve as "memory"
   - **Resolution:** Studied PyTorch TransformerDecoder API, read documentation

3. **Loss Calculation:**
   - Initially confused about input/target shifting
   - **Resolution:** Studied GPT training, understood next-token prediction

4. **Shape Mismatches:**
   - Encountered dimension errors during forward pass
   - **Resolution:** Carefully traced tensor shapes through pipeline, fixed mismatches

## Code Statistics
- Files modified: src/model.py (now ~260 lines)
- Classes implemented: ProjectionLayer, CausalDecoder, OpenVisionStudent
- Functions implemented: 5+ (forward, compute_loss, masking, etc.)
- Test results: Loss decreased from 9.38 → 0.10 (successful overfitting)

## Next Week Goals
1. Restructure project into proper Python package
2. Implement dataset class for reproducible data generation
3. Build complete training script with validation
4. Add model checkpointing functionality
5. Create Jupyter notebook for architecture verification

## Learning Outcomes
- Deep understanding of transformer decoder architecture
- Mastered causal language modeling objective
- Learned visual token masking technique
- Experience with end-to-end model integration
- Understanding of next-token prediction for text generation

## Time Spent
- Projection layer: 4 hours
- Decoder implementation: 10 hours
- Loss function and training: 6 hours
- Testing and debugging: 6 hours
- **Total:** ~26 hours

## Technical Insights
- Projection layer trades off information for computational efficiency
- Causal masking is fundamental for autoregressive generation
- Visual masking acts as data augmentation
- Model architecture verified through successful overfitting test

---

