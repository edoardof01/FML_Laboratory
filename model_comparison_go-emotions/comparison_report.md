# Model Comparison Report
## Dataset: go-emotions

### Summary
- **Models Compared**: 8
- **Best Overall Model**: Ensemble

### Comparison Table

| Model                |   F1 Weighted |   F1 Micro |   Accuracy |   Precision |   Recall |   Avg Rank |
|:---------------------|--------------:|-----------:|-----------:|------------:|---------:|-----------:|
| Ensemble             |      0.603341 |   0.61559  |   0.467293 |    0.6117   | 0.619529 |        2.8 |
| Clean Baseline       |      0.581649 |   0.603992 |   0.505804 |    0.631464 | 0.561858 |        2.8 |
| Baseline (Fine-tune) |      0.573756 |   0.587731 |   0.473927 |    0.6414   | 0.530574 |        3.4 |
| Weighted Classes     |      0.573239 |   0.547037 |   0.272342 |    0.480529 | 0.744667 |        5.8 |
| Contrastive (SupCon) |      0.556252 |   0.577927 |   0.464345 |    0.654447 | 0.503871 |        4.6 |
| K-Fold CV            |      0.551562 |   0.581897 |   0.462871 |    0.690095 | 0.490599 |        4.8 |
| MLM Pretraining      |      0.53996  |   0.574599 |   0.454395 |    0.701749 | 0.478591 |        5.6 |
| Partial Freezing     |      0.532508 |   0.571896 |   0.449051 |    0.711289 | 0.46911  |        6.2 |

### Best Model per Metric

| Metric | Best Model | Score |
|--------|------------|-------|
| F1 Weighted | Ensemble | 0.6033 |
| F1 Micro | Ensemble | 0.6156 |
| Accuracy | Clean Baseline | 0.5058 |
| Precision | Partial Freezing | 0.7113 |
| Recall | Weighted Classes | 0.7447 |

### Visualizations

![Bar Chart](model_comparison_bar.png)

![Heatmap](model_comparison_heatmap.png)

![Radar Chart](model_comparison_radar.png)

---
*Generated automatically by compare_models.py*
