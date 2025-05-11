# 🌾 Occupational Risk Analysis for Female Agricultural Workers



Machine learning project predicting occupational accidents among female agricultural workers, comparing Random Forest and optimized XGBoost models.

## 🚀 Key Features

- **Data Pipeline**  
  Automated cleaning of 80+ variables with Pandas/KNNImputer
- **Comparative Modeling**  
  Random Forest vs XGBoost with hyperparameter tuning
- **Explainability**  
  SHAP analysis for risk factor interpretation
- **Safety Recommendations**  
  PPE guidance based on worker profiles

## 📊 Model Performance

### Random Forest
| Metric          | Score               |
|-----------------|---------------------|
| Accuracy        | 0.75               |
| AUC-ROC         | 0.85               |
| Recall (Class 1)| 0.33               |
| Confusion Matrix| [[10 0] [4 2]]     |

### Optimized XGBoost (GridSearchCV)
| Metric          | Score               |
|-----------------|---------------------|
| Accuracy        | 0.88               |
| AUC-ROC         | 0.78               |
| Recall (Class 1)| 0.60               |
| Confusion Matrix| [[11 0] [2 3]]     |

**Key Tradeoffs**:
- XGBoost achieved **17% higher accuracy** but **8% lower AUC-ROC**
- Better recall (0.60 vs 0.33) for accident prediction (Class 1)

## 🛠️ Tech Stack

```python
# Data Processing
Pandas • NumPy • KNNImputer

# Visualization
Seaborn • Plotly • Matplotlib

# Modeling
RandomForest • XGBoost (GridSearchCV) • SMOTE

# Explainability
SHAP • Feature Importance
