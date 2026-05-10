# DA6401 Assignment 3 — Transformer Machine Translation (DE → EN)

## Overview
Implements the Transformer architecture from *"Attention Is All You Need"* (Vaswani et al., 2017) from scratch using PyTorch. Trains a Neural Machine Translation system on the Multi30k dataset (German → English).

**Test BLEU: 37.20** on Multi30k test set.

---

## Project Structure

```
assignment3/
├── model.py          # Transformer architecture (MHA, PE, Encoder, Decoder)
├── train.py          # Training loop, LabelSmoothingLoss, greedy decode, BLEU eval
├── dataset.py        # Data loading from cached tensors
├── lr_scheduler.py   # Noam warmup scheduler
└── checkpoints/
    └── best.pt       # Best checkpoint by validation loss
```

The notebook (`DA6401_Assignment3_Transformer.ipynb`) is the single entry point — all `.py` files are written by notebook cells and executed in Colab.

---

## Setup

### Requirements
```
torch, datasets, spacy, sacrebleu, wandb, tqdm, matplotlib
```

Install (Cell 2 in notebook):
```bash
pip install datasets wandb evaluate sacrebleu tqdm
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

> **Note:** `torchtext` is incompatible with Python 3.12 + PyTorch 2.10. The vocab is built with a plain `SimpleVocab` class — no torchtext needed.

### First-Time Data Setup
Run **Cell 10A** once. It downloads Multi30k, tokenizes with spaCy, and saves `processed_data.pt` to Google Drive. All subsequent sessions load from the Drive cache automatically.

---

## Model Architecture

| Hyperparameter | Value |
|---|---|
| d_model | 256 |
| Encoder/Decoder layers (N) | 3 |
| Attention heads | 8 |
| d_ff | 512 |
| Dropout | 0.1 |
| Label smoothing (ε) | 0.1 |
| Warmup steps | 4000 |
| Epochs | 20 |
| Batch size | 128 |

- **Positional Encoding:** Sinusoidal (fixed), registered as buffer
- **Attention:** Custom scaled dot-product + multi-head (no `nn.MultiheadAttention`)
- **Norm:** Post-LayerNorm (`Add & Norm`)
- **Optimizer:** Adam (β1=0.9, β2=0.98, ε=1e-9) + Noam scheduler

---

## Training

Open the notebook in Google Colab (GPU runtime recommended — T4 is sufficient).

Run cells in order:
1. Cells 1–2: GPU check, install dependencies
2. Cell 3: Mount Google Drive
3. Cells 5–8: Write `model.py`, `lr_scheduler.py`, `dataset.py`, `train.py`
4. Cell 9: W&B login
5. **Cell 10A** *(first time only)*: Build and save dataset
6. Cell 10: Load dataloaders
7. Cell 11: Initialize model
8. Cell 13: Train baseline (20 epochs, ~6 min on T4)

---

## W&B Experiments

All runs are logged to project `da6401-a3`. Runs are grouped by experiment section for easy chart comparison.

| Section | W&B Group | What's compared |
|---|---|---|
| 2.1 | `2.1_scheduler` | Noam LR vs Fixed LR `1e-4` |
| 2.2 | `2.2_scaling` | With vs without `sqrt(d_k)` scaling; Q/K grad norms logged per step |
| 2.3 | `2.3_attention` | Encoder attention head heatmaps (all 8 heads) |
| 2.4 | `2.4_pos_encoding` | Sinusoidal PE vs Learned PE (nn.Embedding) |
| 2.5 | `2.5_label_smoothing` | ε=0.1 vs ε=0.0; prediction confidence logged per epoch |

---

## Results

| Experiment | Val Loss (best) | Test BLEU |
|---|---|---|
| Baseline (Noam + sinusoidal + smooth=0.1) | 2.63 | **37.20** |
| Fixed LR 1e-4 | higher / diverges early | — |
| No sqrt(d_k) scaling | unstable grad norms | — |
| Learned PE | comparable | run Cell 18 |
| No label smoothing (ε=0.0) | lower loss, overconfident | run Cell 19 |

---

## References

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 — https://arxiv.org/abs/1706.03762
- Multi30k dataset — https://huggingface.co/datasets/bentrevett/multi30k
- Assignment spec — DA6401, IIT Madras