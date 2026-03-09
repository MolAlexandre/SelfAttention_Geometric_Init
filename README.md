# Self-Attention Geometric Init

Reproduction experiments for:

> **The underlying structures of self-attention: symmetry, directionality, and emergent dynamics in Transformer training**  
> [arXiv 2502.10927](https://arxiv.org/abs/2502.10927)

We study how the **symmetry of** $W_{QK} = W_Q W_K^T$ evolves during training and how initialising $W_Q = W_K$ (symmetric init) affects learning dynamics, on two settings:

- **BERT-Mini** on Wikipedia (Masked Language Modeling)
- **ViT-6L** on CIFAR-10 (Image Classification)


## Key Theoretical Results

- **Bidirectional training (encoders, e.g., BERT)**  
  → gradient updates induce **symmetry** in $W_{QK}$ (i.e., $W_{QK} ≈ W_QW_K^T$)

- **Autoregressive training (decoders, e.g., GPT/LLaMA)**  
  → updates induce **directionality** and **column dominance** in $W_{QK}$

These structures are empirically verified on GPT, LLaMA3 as well as vision (ViT).

## Practical Application: Symmetric Initialization

The paper shows that initializing $W_{QK}$ symmetrically ($W_{QK} = W_{Q}W_{K^T}$at t=0) improves encoder model performance on NLP tasks, by aligning the initialization with the geometric structure the model naturally develops.


---

## Repository structure

```
.
├── src/
│   ├── config.py          # All model & training configs + named presets
│   ├── models/
│   │   ├── transformer.py # MultiHeadAttention, FeedForward, EncoderBlock
│   │   ├── embeddings.py  # BERTEmbeddings, VITEmbeddings
│   │   ├── bert.py        # BERTForMLM
│   │   └── vit.py         # VITForClassification
│   ├── data/
│   │   ├── wikipedia.py   # WikipediaDatasetManager + MLMCollator
│   │   └── cifar.py       # CifarDatasetManager
│   ├── training/
│   │   ├── base_trainer.py
│   │   ├── bert_trainer.py
│   │   └── vit_trainer.py
│   └── metrics/
│       └── symmetry.py    # symmetry_score(), compute_model_symmetry()
├── scripts/
│   ├── train_bert.py      # Entry point — BERT MLM
│   └── train_vit.py       # Entry point — ViT CIFAR-10
├── notebooks/
│   └── dir_sym.ipynb      # Exploration & visualisation
├── assets/
│   └── 2502.10927v2.pdf   # Reference paper
└── requirements.txt
```

---

## Quick start

```bash
pip install -r requirements.txt

# BERT (standard vs symmetric init)
python scripts/train_bert.py --model standard
python scripts/train_bert.py --model symmetric

# ViT on CIFAR-10
python scripts/train_vit.py --model standard
python scripts/train_vit.py --model symmetric

# Resume from checkpoint
python scripts/train_bert.py --model symmetric --resume checkpoints/symmetric_best.pt
```

---

## Symmetry score

For a layer's attention matrices $W_Q$ and $W_K$, we define:

$$s(W_{QK}) = \frac{\|W_{QK}^s\|_F^2 - \|W_{QK}^a\|_F^2}{\|W_{QK}\|_F^2} \in [-1, 1]$$

where $W_{QK}^s = \frac{W_{QK} + W_{QK}^T}{2}$ is the symmetric part and $W_{QK}^a$ the skew-symmetric part.

- **+1** → fully symmetric
- **−1** → fully skew-symmetric
- **0** → no preference

The score is logged per layer and per epoch in `metrics_*.csv`.

---

## Configs

Named presets are available in `src/config.py`:

| Class | Architecture | Task |
|---|---|---|
| `BERTMiniConfig` | 4L-256H | MLM |
| `BERTBaseConfig` | 12L-768H | MLM |
| `BERTLargeConfig` | 24L-1024H | MLM |
| `ViT6LayerCIFAR10` | 6L-512H | CIFAR-10 |

All accept a `symmetric_init: bool` argument.
