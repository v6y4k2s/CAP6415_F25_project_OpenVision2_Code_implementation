import torch
import torch.nn as nn
import timm
import math


# ============================================================
# 1. VISUAL ENCODER (Vision Transformer)
# ============================================================

class VisualEncoder(nn.Module):
    """
    Loads a pretrained Vision Transformer and extracts patch embeddings.
    """

    def __init__(self, model_name="vit_base_patch16_224", freeze=True):
        super().__init__()

        # Load pretrained ViT from timm
        self.vit = timm.create_model(model_name, pretrained=True)

        # Remove classification head → keep only patch embeddings
        self.vit.head = nn.Identity()

        # Optionally freeze encoder weights
        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False

        # Store embedding dimension (e.g., 768 for ViT-B/16)
        self.embed_dim = self.vit.embed_dim

    def forward(self, images):
        """
        images: (batch, 3, 224, 224)
        returns: (batch, num_tokens, embed_dim)
        """
        return self.vit.forward_features(images)


# ============================================================
# 2. PROJECTION LAYER (Visual → Text embedding space)
# ============================================================

class ProjectionLayer(nn.Module):
    """
    Projects visual tokens from ViT embedding dim → text decoder embedding dim.
    """

    def __init__(self, visual_dim=768, text_dim=512):
        super().__init__()
        self.projection = nn.Linear(visual_dim, text_dim)

    def forward(self, visual_tokens):
        """
        visual_tokens: (batch, num_tokens, visual_dim)
        returns: (batch, num_tokens, text_dim)
        """
        return self.projection(visual_tokens)


# ============================================================
# 3. CAUSAL DECODER (GPT-style Transformer)
# ============================================================

class CausalDecoder(nn.Module):
    """
    A GPT-style Transformer decoder for generating captions.
    """

    def __init__(
        self,
        vocab_size=10000,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        max_seq_len=128,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection → vocabulary logits
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, text_tokens, visual_tokens, text_mask=None):
        """
        text_tokens: (batch, seq_len)
        visual_tokens: (batch, num_visual_tokens, hidden_dim)
        """
        batch_size, seq_len = text_tokens.shape

        # Embed tokens and positions
        token_embeds = self.token_embedding(text_tokens)
        positions = torch.arange(seq_len, device=text_tokens.device)
        pos_embeds = self.positional_embedding(positions)
        text_embeds = token_embeds + pos_embeds

        # Generate causal mask if not provided
        if text_mask is None:
            text_mask = self.generate_causal_mask(seq_len, text_tokens.device)

        # Transformer decoding (visual tokens act as memory)
        decoder_output = self.transformer_decoder(
            tgt=text_embeds,
            memory=visual_tokens,
            tgt_mask=text_mask,
        )

        # Predict vocabulary logits
        logits = self.output_projection(decoder_output)
        return logits

    def generate_causal_mask(self, seq_len, device):
        """
        Creates an autoregressive causal mask.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))


# ============================================================
# 4. VISUAL TOKEN MASKING (OpenVision-style)
# ============================================================

def apply_visual_token_masking(visual_tokens, mask_ratio=0.5):
    """
    Randomly masks 50% of visual patch tokens.
    Keeps CLS token at index 0.
    """
    batch_size, num_tokens, hidden_dim = visual_tokens.shape

    num_patches = num_tokens - 1
    num_masked = int(num_patches * mask_ratio)

    masked_tokens = visual_tokens.clone()

    for i in range(batch_size):
        mask_indices = torch.randperm(num_patches)[:num_masked] + 1
        masked_tokens[i, mask_indices] = 0

    return masked_tokens


# ============================================================
# 5. COMPLETE MODEL (OpenVisionStudent)
# ============================================================

class OpenVisionStudent(nn.Module):
    """
    Full OpenVision 2 model for educational purposes.
    Combines: VisualEncoder → ProjectionLayer → CausalDecoder
    """

    def __init__(
        self,
        vocab_size=10000,
        visual_dim=768,
        text_dim=512,
        num_decoder_layers=6,
        num_heads=8,
        max_seq_len=128,
        freeze_visual_encoder=True,
    ):
        super().__init__()

        # Component 1: Visual Encoder (frozen ViT)
        self.visual_encoder = VisualEncoder(freeze=freeze_visual_encoder)

        # Component 2: Projection Layer (768 → 512)
        self.projector = ProjectionLayer(visual_dim=visual_dim, text_dim=text_dim)

        # Component 3: Causal Decoder (GPT-style)
        self.decoder = CausalDecoder(
            vocab_size=vocab_size,
            hidden_dim=text_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
        )

        self.vocab_size = vocab_size

    def forward(self, images, text_tokens, apply_masking=True, mask_ratio=0.5):
        """
        Full forward pass through the model.

        Args:
            images: (batch, 3, 224, 224)
            text_tokens: (batch, seq_len) - caption token IDs
            apply_masking: Whether to apply visual token masking (True during training)
            mask_ratio: Fraction of visual patches to mask (default 0.5)

        Returns:
            logits: (batch, seq_len, vocab_size) - next-token predictions
        """
        # Step 1: Extract visual tokens from images
        visual_tokens = self.visual_encoder(images)  # (batch, 197, 768)

        # Step 2: Project to text embedding space
        visual_tokens = self.projector(visual_tokens)  # (batch, 197, 512)

        # Step 3: Apply visual token masking (training only)
        if apply_masking:
            visual_tokens = apply_visual_token_masking(visual_tokens, mask_ratio)

        # Step 4: Decode with causal attention
        logits = self.decoder(text_tokens, visual_tokens)  # (batch, seq_len, vocab_size)

        return logits

    def compute_loss(self, images, caption_tokens, apply_visual_masking=True, mask_ratio=0.5):
        """
        Computes Causal Language Modeling loss.

        Args:
            images: (batch, 3, 224, 224)
            caption_tokens: (batch, seq_len) - full caption including start/end tokens
            apply_visual_masking: Whether to apply visual token masking (True for training, False for validation)
            mask_ratio: Fraction of visual patches to mask (default 0.5)

        Returns:
            loss: Cross-entropy loss for next-token prediction
        """
        # Separate input and targets
        # Input: all tokens except the last one
        # Target: all tokens except the first one (shifted by 1)
        input_tokens = caption_tokens[:, :-1]  # (batch, seq_len-1)
        target_tokens = caption_tokens[:, 1:]  # (batch, seq_len-1)

        # Forward pass
        logits = self.forward(
            images, input_tokens, apply_masking=apply_visual_masking, mask_ratio=mask_ratio
        )  # (batch, seq_len-1, vocab_size)

        # Flatten for cross-entropy
        # logits: (batch * seq_len-1, vocab_size)
        # targets: (batch * seq_len-1)
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = target_tokens.reshape(-1)

        # Compute cross-entropy loss
        loss = nn.functional.cross_entropy(logits_flat, targets_flat)

        return loss


# ============================================================
# 6. FULL PIPELINE TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3: Testing OpenVisionStudent with Training Loop")
    print("=" * 60)

    # Create the full model
    model = OpenVisionStudent(
        vocab_size=10000,
        visual_dim=768,
        text_dim=512,
        num_decoder_layers=4,  # Smaller for faster testing
        num_heads=8,
        max_seq_len=128,
    )

    # Create dummy data (in real training, this comes from a dataset)
    batch_size = 4
    seq_len = 20
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_captions = torch.randint(0, 10000, (batch_size, seq_len))

    print(f"\nDummy batch:")
    print(f"  Images: {dummy_images.shape}")
    print(f"  Captions: {dummy_captions.shape}")

    # Test forward pass
    print("\n" + "-" * 60)
    print("Testing forward pass...")
    print("-" * 60)
    logits = model(dummy_images, dummy_captions[:, :-1])  # Exclude last token
    print(f"Output logits: {logits.shape}")
    print(f"Expected: ({batch_size}, {seq_len-1}, 10000)")

    # Test loss computation
    print("\n" + "-" * 60)
    print("Testing loss computation...")
    print("-" * 60)
    loss = model.compute_loss(dummy_images, dummy_captions)
    print(f"Initial loss: {loss.item():.4f}")
    print(f"(Random baseline ~ln(10000) = {math.log(10000):.4f})")

    # Minimal training loop (overfitting on dummy data)
    print("\n" + "-" * 60)
    print("Running minimal training loop (10 steps)...")
    print("-" * 60)
    print("\nThis demonstrates the model CAN learn by overfitting on dummy data.\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for step in range(10):
        optimizer.zero_grad()
        loss = model.compute_loss(dummy_images, dummy_captions)
        loss.backward()
        optimizer.step()

        if step % 2 == 0:
            print(f"Step {step:2d} | Loss: {loss.item():.4f}")

    print(f"\nStep  9 | Loss: {loss.item():.4f}")
    print("\n" + "=" * 60)
    print("✅ PHASE 3 COMPLETE!")
    print("=" * 60)
    print("\nKey Observations:")
    print("  1. Loss should decrease from ~9.21 (random) to lower values")
    print("  2. This proves the model is learning (overfitting on dummy data)")
    print("  3. Next step: Train on real image-caption pairs!")
    print("=" * 60)
