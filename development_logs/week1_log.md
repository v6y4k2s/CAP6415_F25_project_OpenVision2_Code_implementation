# Week 1 Development Log - OpenVision 2 Project
**Date Range:** November 11-17, 2025
**Student:** Sumedh
**Focus:** Project Setup & Visual Front-End Implementation

---

## Overview
This week marked the beginning of the OpenVision 2 image captioning project. The primary focus was on understanding vision-language models, setting up the development environment, and implementing the visual encoder component.

## Tasks Completed

### 1. Project Initialization (Nov 11-12)
- **Environment Setup:**
  - Created project repository structure
  - Set up Python virtual environment
  - Installed core dependencies: PyTorch, timm, torchvision
  - Configured Apple Silicon GPU (MPS) support

- **Research & Planning:**
  - Studied vision-language model architectures
  - Researched OpenVision 2 architecture (encoder-decoder approach)
  - Analyzed Vision Transformer (ViT) for visual encoding
    

### 2. Visual Encoder Implementation (Nov 13-15)
- **Component Development:**
  - Implemented `VisualEncoder` class in src/model.py
  - Integrated pretrained ViT model from timm library (vit_base_patch16_224)
  - Removed classification head to extract raw patch embeddings
  - Configured model to output visual tokens instead of classification logits

- **Technical Details:**
  - Input shape: (batch, 3, 224, 224) - RGB images
  - Output shape: (batch, 197, 768) - Visual tokens
  - 197 tokens = 196 patch tokens + 1 [CLS] token
  - Each token is 768-dimensional embedding

- **Testing & Validation:**
  - Created test script to verify encoder output shapes
  - Tested with dummy images (batch size of 2)
  - Confirmed visual token extraction works correctly

### 3. Concept Learning & Documentation (Nov 16-17)
- **Key Concepts Mastered:**
  - Visual tokens vs global average pooling
  - Why patch-based tokens preserve spatial information
  - Shape conventions: (batch, sequence_length, hidden_dim)
  - Difference between classification and feature extraction modes

- **Understanding Achieved:**
  - Why OpenVision needs 197 separate tokens (spatial awareness)
  - How patch tokens enable region-aware caption generation
  - Tensor shape interpretation for multi-dimensional data

## Challenges Faced

1. **Initial Confusion:**
   - Understanding difference between visual tokens and single feature vectors
   - Grasping the concept of 197 tokens vs 1 global representation
   - **Resolution:** Studied transformer architecture, visualized patch-based encoding

2. **Library Integration:**
   - Learning timm library API for pretrained models
   - Understanding how to modify model architecture for feature extraction
   - **Resolution:** Read timm documentation, experimented with model structure

## Code Statistics
- Files created: 2 (src/model.py, CLAUDE.md)
- Lines of code: ~120
- Functions implemented: VisualEncoder class
- Tests written: 1 verification script

## Next Week Goals
1. Implement projection layer to bridge visual and text dimensions
2. Build causal decoder for autoregressive caption generation
3. Understand visual token masking technique
4. Create end-to-end forward pass

## Learning Outcomes
- Solid understanding of Vision Transformers
- Experience with timm library for pretrained models
- Clear grasp of visual token concept
- Foundation for encoder-decoder architecture

## Time Spent
- Environment setup: 3 hours
- Research and planning: 5 hours
- Implementation: 8 hours
- Testing and debugging: 4 hours
- **Total:** ~20 hours

---
