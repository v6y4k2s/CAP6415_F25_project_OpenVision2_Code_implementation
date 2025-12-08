**Please note Git LFS needs to be installed on PC while cloning this repository**
This is due to the model in the checkpoints folder which is a huge file as compared to the other files. 

UNOFFICIAL CODE IMPLEMENTATION OF OPENVISION2
OpenVision 2 — Vision-Language Image Captioning Model 

OpenVision 2 is an vision-language model that generates natural language captions from images.
It combines a pretrained Vision Transformer (ViT) with a lightweight transformer decoder for fast, single-GPU training.

Overview
Image captioning requires converting high-dimensional visual data into coherent sentences. OpenVision 2 solves this using a simple and effective encoder–decoder pipeline:
Visual Encoder — Frozen ViT-Base extracts 197 visual patch tokens (768-dim).
Projection Layer — Learns to map visual features into the text embedding space (768 → 512).
Causal Decoder — A 6-layer transformer (GPT-style) generates captions autoregressively using cross-attention.
The model is trained on Flickr8k (8K images, 40K captions) with a causal language-modeling objective.

Architecture
Image → ViT Encoder → Projection Layer → Transformer Decoder → Caption

Key features:
Visual Token Masking (50%) for robustness
Frozen ViT to avoid overfitting on small datasets
Autoregressive decoding with causal masking


Results
Metric	Value
Validation Loss	0.89
Training Loss	0.41
Parameters	116.6M (30.8M trainable)
Vocab Size	5,000

Example captions:

“a little girl in a pink dress is standing on the stairs”

“two dogs are playing in the grass”

Project Structure
cv_project/
 ├── src/
 │   ├── model.py
 │   ├── tokenizer.py
 │   ├── real_dataset.py
 │   ├── train_real.py
 │   └── inference_real.py
 ├── data/
 ├── checkpoints/
 ├── notebooks/
 ├── development_logs/
 └── requirements.txt

Installation
git clone <repo>
cd cv_project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

Dataset (Flickr8k)
Download from Kaggle: https://www.kaggle.com/datasets/adityajn105/flickr8k

Place files as:

data/
 ├── Images/
 └── captions.txt

Training
python src/train_real.py --epochs 10 --batch-size 32 --vocab-size 5000 --save-model

""Inference
python src/inference_real.py \
  --checkpoint checkpoints/best_model_real.pt \
  --tokenizer checkpoints/tokenizer.json \
  --image path/to/image.jpg""

Technical Details

Encoder: ViT-Base/16 (197×768 tokens, frozen)
Decoder: 6-layer transformer, 512-dim embeddings, 8 heads
Training Tricks: token masking, teacher forcing, Adam optimizer (1e-4)

Limitations & Future Work
Word-level vocab (no BPE)
Greedy decoding only
Small dataset (Flickr8k)
Encoder not fine-tuned
Planned improvements: beam search, BLEU/CIDEr evaluation, attention maps, ViT fine-tuning, MS-COCO support.

Acknowledgments
ViT models from timm
Flickr8k dataset
PyTorch ecosystem
