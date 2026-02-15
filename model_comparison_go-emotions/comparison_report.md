# Model Comparison Report
## Dataset: go-emotions

### Summary
- **Models Compared**: 8
- **Best Overall Model**: Ensemble

### Comparison Table

| Model                |   F1 Weighted |   F1 Micro |   Accuracy |   Precision |   Recall |   Avg Rank |
|:---------------------|--------------:|-----------:|-----------:|------------:|---------:|-----------:|
| Ensemble             |      0.604887 |   0.613693 |   0.462318 |    0.601853 | 0.626007 |        3   |
| Weighted Classes     |      0.584567 |   0.585764 |   0.43044  |    0.574887 | 0.601359 |        5   |
| Clean Baseline       |      0.581649 |   0.603992 |   0.505804 |    0.631464 | 0.561858 |        3   |
| Baseline (Fine-tune) |      0.573756 |   0.587731 |   0.473927 |    0.6414   | 0.530574 |        3.8 |
| K-Fold CV            |      0.573265 |   0.588729 |   0.47319  |    0.638857 | 0.531522 |        4   |
| Contrastive (SupCon) |      0.556252 |   0.577927 |   0.464345 |    0.654447 | 0.503871 |        5   |
| MLM Pretraining      |      0.53996  |   0.574599 |   0.454395 |    0.701749 | 0.478591 |        5.8 |
| Partial Freezing     |      0.532508 |   0.571896 |   0.449051 |    0.711289 | 0.46911  |        6.4 |

### Best Model per Metric

| Metric | Best Model | Score |
|--------|------------|-------|
| F1 Weighted | Ensemble | 0.6049 |
| F1 Micro | Ensemble | 0.6137 |
| Accuracy | Clean Baseline | 0.5058 |
| Precision | Partial Freezing | 0.7113 |
| Recall | Ensemble | 0.6260 |

### Visualizations

![Bar Chart](model_comparison_bar.png)

![Heatmap](model_comparison_heatmap.png)

![Radar Chart](model_comparison_radar.png)

---
*Generated automatically by compare_models.py*
