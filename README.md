# DistilBERT Emotion Classification Pipeline

A comprehensive pipeline for **multi-label emotion classification** using [DistilBERT](https://huggingface.co/distilbert-base-uncased) on the [GoEmotions](https://huggingface.co/datasets/go_emotions) dataset (28 emotion labels).

The project explores multiple training strategies, compares them with statistical rigour, and provides explainability (XAI) via LIME.

---

## Project Structure

```
.
├── configs/
│   └── base_config.yaml           # Centralised YAML configuration
├── src/                            # Shared utility modules
│   ├── config_utils.py             #   YAML + CLI config loading
│   ├── data_utils.py               #   Dataset loading, label detection, tokenisation
│   ├── model_utils.py              #   Model initialisation helper
│   ├── train_utils.py              #   Data collators, WeightedTrainer, training pipeline
│   ├── metrics.py                  #   Trainer-compatible & comprehensive metrics
│   ├── losses.py                   #   SupCon loss for contrastive pretraining
│   ├── viz_utils.py                #   Plots: label distribution, confusion matrix, embeddings
│   └── analysis_utils.py           #   Embedding extraction & hard-negative mining
│
├── fine_tune_emotions.py           # 1. Baseline fine-tuning
├── fine_tune_clean.py              # 2. Fine-tuning on cleaned dataset
├── MLM_pretraining.py              # 3. MLM domain adaptation → fine-tuning
├── weighted_emotions.py            # 4. Class-weighted loss (BCEWithLogitsLoss + pos_weight)
├── partial_freezing.py             # 5. Layer-wise freezing analysis
├── k-fold-cross-validation.py      # 6. K-Fold cross-validation
├── contrastive_pretraining.py      # 7. Supervised Contrastive (SupCon) pretraining
├── clean_dataset.py                # Dataset cleaning via Confident Learning (cleanlab)
│
├── ensemble-emotions.py            # Soft-vote ensemble of all models
├── compare_models.py               # Metrics comparison table + charts
├── evaluate_emotions.py            # Standalone model evaluation
├── evaluate_all_models.py          # Batch evaluation with comprehensive metrics
├── train_all_models.py             # End-to-end training of all variants
├── statystical_significance_analysis.py  # Bootstrap CIs, McNemar's test
│
├── xai_analysis.py                 # LIME explainability analysis
├── test_lime_single_sentence.py    # Quick LIME demo on one sentence
├── run_analysis.py                 # Hard-negative mining demo
├── analyze_predictions.py          # Prediction distribution analysis
│
├── execute_improvements.sh         # Shell script to run the full pipeline
├── requirements.txt                # Python dependencies
└── README.md                       # ← You are here
```

---

## Quick Start

### 1. Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Individual Training Scripts

Every training script reads from `configs/base_config.yaml` by default:

```bash
# Baseline fine-tuning
python fine_tune_emotions.py

# Fine-tuning on cleaned dataset
python fine_tune_clean.py

# MLM domain-adaptation + fine-tuning
python MLM_pretraining.py

# Weighted loss (class-imbalance aware)
python weighted_emotions.py

# Partial freezing layer analysis
python partial_freezing.py

# K-Fold cross-validation
python k-fold-cross-validation.py

# Supervised Contrastive pretraining
python contrastive_pretraining.py
```

Override the config file with `--config`:

```bash
python fine_tune_emotions.py --config configs/my_config.yaml
```

### 3. Run the Full Pipeline

```bash
bash execute_improvements.sh
```

This runs dataset cleaning → all training variants → ensemble → comparison → statistical analysis → XAI.

---

## Training Strategies

| # | Strategy | Script | Key Idea |
|---|----------|--------|----------|
| 1 | **Baseline** | `fine_tune_emotions.py` | Standard fine-tuning with Optuna HP search |
| 2 | **Clean Baseline** | `fine_tune_clean.py` | Same, but on the dataset cleaned by Confident Learning |
| 3 | **MLM Domain Adaptation** | `MLM_pretraining.py` | Continued MLM pretraining on task corpus, then fine-tuning |
| 4 | **Weighted Loss** | `weighted_emotions.py` | `BCEWithLogitsLoss` with per-class `pos_weight` + softening exponent |
| 5 | **Partial Freezing** | `partial_freezing.py` | Freezes lower transformer layers; sweeps 0–6 unfrozen layers |
| 6 | **K-Fold CV** | `k-fold-cross-validation.py` | 4-fold CV for stability estimation, then final model on original split |
| 7 | **SupCon Pretraining** | `contrastive_pretraining.py` | Supervised contrastive learning with Jaccard-based similarity |
| 8 | **Ensemble** | `ensemble-emotions.py` | Soft-vote ensemble with automatic threshold tuning |

---

## Configuration

All training scripts are driven by a single YAML file (`configs/base_config.yaml`):

```yaml
dataset:
  name: "go_emotions"
  subset: "simplified"
  # path: "data/go_emotions_cleaned"   # uncomment to use cleaned data
  text_column: "text"
  label_column: "labels"

model:
  name: "distilbert-base-uncased"
  max_length: 128

training:
  learning_rate: 2.0e-5
  per_device_train_batch_size: 16
  num_train_epochs: 3
  weight_decay: 0.01
  seed: 42

optuna:
  enabled: true
  n_trials: 3
```

---

## Source Modules (`src/`)

| Module | Purpose |
|--------|---------|
| `config_utils.py` | Loads YAML config, parses CLI `--config` / `--output_dir` |
| `data_utils.py` | Loads datasets (remote or local), detects single/multi-label, converts to binary vectors, tokenises |
| `model_utils.py` | `get_model()` — loads DistilBERT for sequence classification with proper `problem_type` |
| `train_utils.py` | `MultiLabelDataCollator`, `WeightedTrainer`, `build_training_args()`, `evaluate_and_save()`, `run_standard_pipeline()` |
| `metrics.py` | `compute_metrics()` (Trainer-compatible), `comprehensive_metrics()` with per-label stats, calibration, distribution diagnostics |
| `losses.py` | `SupConLoss` — supervised contrastive loss with optional Jaccard similarity |
| `viz_utils.py` | Label distribution bar charts, normalised confusion matrices, PCA / t-SNE embedding visualisations |
| `analysis_utils.py` | CLS-embedding extraction, hard-negative mining |

---

## Evaluation & Analysis

```bash
# Compare all models (CSV + charts)
python compare_models.py

# Statistical significance (bootstrap CIs + McNemar)
python statystical_significance_analysis.py

# XAI with LIME
python xai_analysis.py

# Prediction distribution analysis
python analyze_predictions.py
```

---

## Output Directories

Each training script produces outputs in a dedicated directory:

| Script | Output Directory |
|--------|-----------------|
| Baseline | `outputs/go-emotions/` |
| Clean Baseline | `outputs/clean_baseline_go-emotions/` |
| MLM | `outputs/continued_pretraining_go-emotions/` |
| Weighted | `outputs/weighted_model_go-emotions/` |
| Partial Freezing | `layer_analysis_go-emotions/` |
| K-Fold | `kfold_model_go-emotions/` |
| SupCon | `outputs/go-emotions/supcon_pretraining/` |
| Ensemble | `ensemble_go-emotions/` |
| Comparison | `model_comparison_go-emotions/` |
| Statistical | `statistical_analysis_go-emotions/` |

Each directory typically contains:
- `final_model/` — saved model + tokenizer
- `test_metrics.json` — test-set metrics
- `*.png` — visualisation plots
- TensorBoard logs in `logs/` or `*_logs/`

---

## Requirements

- Python ≥ 3.10
- PyTorch
- Transformers (HuggingFace)
- Datasets (HuggingFace)
- scikit-learn, numpy, pandas
- matplotlib, seaborn
- optuna
- lime
- cleanlab
- statsmodels
- accelerate, peft (optional — for LoRA adapters)

Install everything with:

```bash
pip install -r requirements.txt
```

---

## License

This project is for academic / research purposes.
