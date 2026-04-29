# BioBERT Adaptation on PubMedQA

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-ffcc00.svg)](https://huggingface.co/docs/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Biomedical question answering with BioBERT: ternary classification (yes/no/maybe) on the PubMedQA benchmark.**

---

## 📋 Overview

This repository contains the implementation of our project for the **Deep Learning & NLP** module at ENSA of Fez. We propose a **two-phase fine-tuning strategy** for **BioBERT** on the **PubMedQA** dataset.

### Key Results

| Model | Accuracy | Weighted F1 | F1 (maybe) |
|-------|----------|-------------|-------------|
| BioBERT (baseline) | 68% | — | — |
| **Our model (BioBERT + 2 phases)** | **67.5%** | **0.65** | **0.20** |
| Human expert (reference) | 78.0% | — | — |

### Detailed Metrics per Class

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Yes | 0.72 | 0.77 | **0.74** | 103 |
| No | 0.62 | 0.74 | **0.68** | 72 |
| Maybe | 0.60 | 0.12 | **0.20** | 25 |
| **Accuracy** |  |  |**0.675 (67.5%)** | 200 |
| **Macro avg** | 0.65 | 0.54 | 0.54 | 200 |
| **Weighted avg** | 0.67 | 0.68 | 0.65 | 200 |

---


## 🎯 Task Description

Given a biomedical question and a PubMed abstract, the model predicts one of three answers:

- ✅ **yes** — The abstract confirms the question
- ❌ **no** — The abstract rejects the question  
- 🤔 **maybe** — The evidence is inconclusive

**Example:**
> *Question:* "Does vitamin D supplementation reduce mortality in older adults?"  
> *Abstract:* [PubMed article summary]  
> *Prediction:* **Yes** (confidence: 0.91)

---

## 📊 Dataset

| Subset | Size | Labels | Usage |
|--------|------|--------|-------|
| PQA-A (artificial) | 211,300 | yes/no | Phase 1 (adaptation) |
| PQA-L (expert) | 1,000 | yes/no/maybe | Phase 2 (fine-tuning) |
| PQA-U (unlabeled) | 61,200 | — | Not used |

**Data split (PQA-L):**
- Training: 800 samples (80%)
- Validation: 100 samples (10%)
- Test: 100 samples (10%)

---

## 🏗️ Methodology

### Two-Phase Training

**Phase 1 — Adaptation on PQA-A**
- 10,000 samples (subsampled, seed=42)
- 3 epochs
- Learning rate: 2e-5
- Batch size: 16
- Focus on yes/no classification

**Phase 2 — Fine-tuning on PQA-L**
- 800 training samples
- Early stopping (patience = 5)
- Learning rate: 5e-6 with layer-wise decay (0.9x per layer)
- Batch size: 8
- Label smoothing: ε = 0.1

### Learning Curves

**Phase 1 (PQA-A):**

| Step | Train Loss | Eval Loss | Train Acc | Eval Acc |
|------|------------|-----------|-----------|----------|
| 0 | 0.620 | 0.590 | 0.965 | 0.960 |
| 200 | 0.240 | 0.595 | 0.967 | 0.966 |
| 500 | 0.150 | 0.580 | 0.970 | 0.970 |
| 1000 | 0.095 | 0.555 | 0.975 | 0.975 |
| 1500 | 0.070 | 0.530 | 0.980 | 0.980 |

**Phase 2 (PQA-L):** Convergence achieved after ~800 steps, final validation accuracy ~74%.

### Confusion Matrix

| True \ Predicted | Yes | No | Maybe |
|------------------|-----|-----|-------|
| **Yes** (103) | 79 | 23 | 1 |
| **No** (72) | 18 | 53 | 1 |
| **Maybe** (25) | 13 | 9 | 3 |

**Observations:**
- Only 3/25 "maybe" examples correctly classified (12% recall)
- 52% of "maybe" examples confused with "yes"
- 36% of "maybe" examples confused with "no"

---
2. Install dependencies
```
pip install -r requirements.txt
```

3. Run the notebook
```
jupyter notebook BioBERT_PubMedQA.ipynb
```
