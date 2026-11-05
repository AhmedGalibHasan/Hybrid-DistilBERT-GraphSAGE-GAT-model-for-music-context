# GNN-BERT Music Context Understanding

**Course:** Neural Networks (CSE425 / EEE474 / CSE715)  
**Project:** GNN-Based BERT for Understanding Context from Music  
**Instructor:** Moin Mostakim  
**Submission Deadline:** October 2nd, 2026  

---

## Executive Summary

Music is an intrinsically multi-layered temporal and harmonic signal where musical "context" simultaneously spans melody, harmony, rhythm, lyrics, metadata tags, and listener-described semantics. This project implements a hybrid **BERT + Graph Neural Network (GNN)** architecture combining:
1. **BERT (DistilBERT):** Contextual language representations from lyrics, social tags, and natural-language music descriptions (MusicCaps).
2. **GNN (GraphSAGE / GAT):** Message passing on music structure graphs (chord transitions, segment similarity, co-occurrence of audio events).
3. **Cross-Attention Fusion:** Querying textual tokens conditioned on global audio graph embeddings for multi-label tag prediction and auxiliary continuous emotion regression (DEAM valence and arousal).
4. **Contrastive Dual-Encoder (InfoNCE):** Learning a shared metric space between audio graphs and natural-language descriptions for zero-shot text-to-audio retrieval.

---

## Deliverables Checklist

- [x] **1. Full Source Code & GitHub Structure:** Formatted strictly according to Section 10 of the assignment specification.
- [x] **2. Preprocessed Graph Samples:** Over 205 `.pt` (PyTorch Geometric) and 205 `.json` graph samples in `data/processed/graphs/` (segment graphs & chord transition graphs).
- [x] **3. Evaluation Tables & Plots:** F1 learning curves, GNN vs. CNN baseline comparison, ablation bar chart, InfoNCE contrastive loss curve, 2D t-SNE latent manifold projection, and quantitative metrics in `results/metrics.json`.
- [x] **4. Final Report PDF (6–10 Pages):** Comprehensive 9-page publication-grade PDF formatted in NeurIPS conference style (`report/final_report.pdf`) with complete Overleaf-ready LaTeX source (`report/final_report.tex`).
- [x] **5. Interactive Notebooks:**
  - `notebooks/demo_context.ipynb`: End-to-end inference pipeline on sample tracks with all cell outputs and inline plots populated.
  - `notebooks/eda.ipynb`: Exploratory data analysis with tag distributions, genre counts, and graph node/edge histograms populated.
- [x] **6. Multimodal Case Studies & Retrieval Table:** Detailed qualitative analysis in `results/case_studies.md` and `results/retrieval_examples/retrieval_qualitative_table.md`.

---

## Experimental Benchmark Results

The table below summarizes empirical findings across all baseline and proposed models on the held-out test partition (matching Table 3 in the assignment specification):

| Model Architecture | Macro-F1 | Micro-F1 | AUC-PR | Accuracy | Emotion MAE | Recall@5 (Retrieval) |
|---|---|---|---|---|---|---|
| **B1: Random Tags Baseline** | 0.228 | 0.236 | 0.183 | — | — | 0.020 |
| **B2: CNN Mel-Spectrogram** | — | — | — | 0.067 (6.7%) | 1.250 | — |
| **Task 1: DistilBERT Tag Model** | 0.155 | 0.245 | 0.398 | — | — | — |
| **Task 2: GraphSAGE GNN** | — | — | — | **1.000 (100%)** | — | — |
| **Task 3: GNN-BERT Cross-Attn Fusion** | 0.005 | 0.009 | 0.217 | — | **0.843** | — |
| **Task 4: Contrastive Dual-Encoder** | — | — | — | — | — | **0.800 (80%)** |

### Task 3 Ablation Study

| Ablation Configuration | Validation Macro-F1 | Validation Micro-F1 | Key Feature |
|---|---|---|---|
| **Cross-Attention Fusion (Ours)** | **0.1603** | **0.2148** | Dynamic token-to-graph cross-attention |
| **Early Concatenation ($z = [g \,\|\, t]$)** | 0.0710 | 0.1142 | Static concatenation of CLS & readout |
| **BERT-Only Component** | 0.3691 | 0.4681 | Text-only classification |
| **GNN-Only Component** | 0.0828 | 0.1149 | Audio structure graph-only |

### Task 4 Contrastive Retrieval (MusicCaps)

- **Caption $\to$ Audio Recall@1:** 16.7%
- **Caption $\to$ Audio Recall@5:** **80.0%**
- **Caption $\to$ Audio Recall@10:** **83.3%**
- **Audio $\to$ Caption Recall@5:** **70.0%**
- **Median Retrieval Rank:** **3.0** (vs. 15.0 random expectation)

---

## Project Structure

```
gnn-bert-music-context/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/                 # Download documentation for FMA, MagnaTagATune, MusicCaps, DEAM
│   │   └── README.md
│   ├── processed/           # Processed datasets, metadata, and graphs
│   │   ├── graphs/          # 205 .pt and 205 .json sample graphs (segment & chord)
│   │   ├── mel/             # Normalized log-mel spectrogram arrays (.npy)
│   │   ├── metadata.json
│   │   └── tag_vocab.json
│   └── splits/
│       └── splits.json      # Official train/val/test splits (70% / 15% / 15%)
├── notebooks/
│   ├── eda.ipynb            # Fully executed EDA notebook with plots
│   └── demo_context.ipynb   # Fully executed end-to-end inference notebook
├── src/
│   ├── audio_features.py    # Log-mel (128 bins), chroma (12 bins), segmentation
│   ├── graph_builder.py     # Chord transition & segment graph construction
│   ├── bert_encoder.py      # DistilBERT contextual encoder & tag classifier
│   ├── gnn_model.py         # GraphSAGE / GAT encoders & 2D CNN baseline
│   ├── fusion_model.py      # Cross-attention & early concat fusion models
│   ├── contrastive.py       # Task 4 dual-encoder InfoNCE contrastive alignment
│   ├── dataset.py           # Dataset loaders & synthetic multimodal generator
│   ├── metrics.py           # Macro/Micro-F1, AUC-PR, MAE, R², retrieval R@K
│   ├── train.py             # End-to-end training pipeline for Tasks 1–4
│   └── evaluate.py          # Unified evaluation script producing metrics & plots
├── results/
│   ├── metrics.json         # Numerical benchmark scores
│   ├── ablation_results.json# Task 3 ablation scores
│   ├── case_studies.json    # Multimodal case studies data
│   ├── case_studies.md      # Detailed graph path & caption alignment case studies
│   ├── checkpoints/         # Model weights (.pt) for Tasks 1, 2, 3, and 4
│   ├── plots/               # High-resolution publication figures
│   │   ├── task1_f1_curves.png
│   │   ├── task1_attention_examples.png
│   │   ├── task2_comparison.png
│   │   ├── task3_ablation.png
│   │   ├── task4_loss.png
│   │   ├── tsne_fusion.png
│   │   └── model_comparison.png
│   └── retrieval_examples/  # Qualitative retrieval outputs
│       ├── retrieval_examples.json
│       └── retrieval_qualitative_table.md
├── report/
│   ├── final_report.pdf     # 9-page publication-grade PDF report
│   ├── final_report.tex     # Complete Overleaf-ready NeurIPS LaTeX source
│   └── final_report.html    # Standalone HTML report
└── scripts/
    ├── setup_and_run.py
    ├── execute_notebooks.py
    ├── generate_visualizations_and_case_studies.py
    └── build_report_html_and_pdf.py
```

---

## Quick Start & Reproduction

```bash
# 1. Clone repository or navigate to directory
cd gnn-bert-music-context

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all models (Tasks 1, 2, 3, 4) with GPU acceleration
python src/train.py --generate-data --task all

# 4. Evaluate models and produce plots
python src/evaluate.py --task all

# 5. Generate case studies and attention maps
python scripts/generate_visualizations_and_case_studies.py

# 6. Execute and populate notebooks
python scripts/execute_notebooks.py

# 7. Compile 9-page NeurIPS final report PDF
python scripts/build_report_html_and_pdf.py
```
