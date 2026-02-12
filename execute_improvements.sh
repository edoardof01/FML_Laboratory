#!/bin/bash

#execute_improvements.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo "    DISTILBERT COMPLETE TRAINING PIPELINE"
echo "========================================================"
echo "This script runs ALL model variants and compares them."
echo ""

# Activate virtual environment
echo "[0/11] Activating virtual environment..."
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual environment activated: $(which python)"
else
    echo "ERROR: Virtual environment 'venv' not found!"
    echo "Create it with: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# 1. Install Dependencies
echo ""
echo "========================================================"
echo "[1/11] Installing dependencies..."
echo "========================================================"
pip install -r requirements.txt

# 2. Dataset Cleaning (Confident Learning)
echo ""
echo "========================================================"
echo "[2/11] Dataset Quality Analysis (Confident Learning)..."
echo "========================================================"
python clean_dataset.py --config configs/base_config.yaml

# 3. Supervised Contrastive Pretraining
echo ""
echo "========================================================"
echo "[3/11] Supervised Contrastive Pretraining..."
echo "========================================================"
python contrastive_pretraining.py --config configs/base_config.yaml

# 4. MLM Domain Adaptation Pretraining
echo ""
echo "========================================================"
echo "[4/11] MLM Domain Adaptation Pretraining..."
echo "========================================================"
python MLM_pretraining.py --config configs/base_config.yaml

# 5. Standard Fine-tuning (Baseline)
echo ""
echo "========================================================"
echo "[5/11] Standard Fine-tuning (Baseline)..."
echo "========================================================"
python fine_tune_emotions.py --config configs/base_config.yaml

# 6. K-Fold Cross Validation
echo ""
echo "========================================================"
echo "[6/11] K-Fold Cross Validation..."
echo "========================================================"
python k-fold-cross-validation.py --config configs/base_config.yaml

# 7. Weighted Class Training
echo ""
echo "========================================================"
echo "[7/11] Weighted Class Training..."
echo "========================================================"
python weighted_emotions.py --config configs/base_config.yaml

# 8. Partial Layer Freezing
echo ""
echo "========================================================"
echo "[8/11] Partial Layer Freezing..."
echo "========================================================"
python partial_freezing.py --config configs/base_config.yaml

# 9. Ensemble Model
echo ""
echo "========================================================"
echo "[9/11] Ensemble Model Evaluation..."
echo "========================================================"
python ensemble-emotions.py --config configs/base_config.yaml

# 10. XAI Analysis (LIME Explanations)
echo ""
echo "========================================================"
echo "[10/11] Explainability Analysis (LIME)..."
echo "========================================================"
python xai_analysis.py --config configs/base_config.yaml

# 11. Statistical Significance Analysis
echo ""
echo "========================================================"
echo "[12/12] Clean-Dataset Baseline Fine-tuning..."
echo "========================================================"
python fine_tune_clean.py --config configs/base_config.yaml

# Final: Model Comparison Report
echo ""
echo "========================================================"
echo "[FINAL] Generating Model Comparison Report..."
echo "========================================================"
python compare_models.py --config configs/base_config.yaml

echo ""
echo "========================================================"
echo "    COMPLETE PIPELINE FINISHED! 🚀"
echo "========================================================"
echo ""
echo "Results saved in:"
echo "  - ./outputs/go-emotions/ (main outputs)"
echo "  - ./model_comparison_go-emotions/ (comparison report)"
echo "  - ./statistical_analysis_go-emotions/ (significance tests)"
echo ""
