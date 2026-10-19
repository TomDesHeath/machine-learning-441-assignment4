# Machine Learning CS441/741 Assignment 4: Ensemble Learning Architectures

**Student Number**: `24888923`  
**Module**: CS441 / RW441 Machine Learning  
**Department**: Computer Science Division, Stellenbosch University  

---

## 1. Overview

This repository contains a complete, pure Python implementation of Ensemble Learning algorithms built from scratch for **Assignment 4: Ensemble Learning** on the UNSW-NB15 network intrusion dataset ($N = 257,673$ instances, $D = 43$ descriptive features, $C = 10$ target categories).

Evaluated Model Families:
- **Baseline Single Decision Tree (CART)** (Gini impurity, cost-sensitive class weighting)
- **Baseline k-Nearest Neighbours (kNN)** (Vectorized Minkowski distance, distance-weighted voting)
- **Random Forest (RF)** (Bootstrap aggregation, random subspace feature selection, out-of-bag error estimation, parallel multi-core fitting)
- **Gradient Boosted Decision Trees (GBDT)** (Multi-class multinomial log-loss, Newton-Raphson leaf updates, shrinkage learning rate)
- **Stacking Classifier** (Heterogeneous meta-learner blending out-of-fold probability predictions from kNN, CART, RF, and GBDT into an $L_2$-regularized multinomial logistic regression meta-model)

---

## 2. Directory Structure

```
assignment_4/
├── README.md                           # System requirements & execution documentation
├── run_all.py                          # Master execution script supporting non-interactive runs
├── 24888923RW441assignment4.tex        # Primary IEEEtran LaTeX report source file
├── 24888923RW441assignment4.pdf        # Compiled deliverable PDF report
├── 24888923RW441assignment4.zip        # Submission archive
├── src/                                # Modular Python implementations from scratch
│   ├── __init__.py
│   ├── data_loader.py                  # UNSW-NB15 dataset loader and cleaner
│   ├── preprocessing.py               # One-hot encoding, Min-Max scaling, feature mapping
│   ├── decision_tree.py                # CART Decision Tree classifier
│   ├── knn.py                          # Vectorized k-Nearest Neighbours classifier
│   ├── random_forest.py                # Parallel Random Forest classifier
│   ├── gradient_boosting.py            # Multi-Class Gradient Boosted Decision Trees
│   ├── stacking.py                     # Two-tier Stacking Ensemble meta-learner
│   ├── metrics.py                      # Multi-class accuracy, F1, Kappa, confusion matrix
│   └── statistical_tests.py            # Friedman test, Iman-Davenport, Nemenyi CD plot
├── datasets/                           # UNSW-NB15 dataset files
│   ├── networkTraffic.csv              # Primary dataset (257,673 instances, 43 features)
│   └── features.csv                    # Feature descriptions
├── figures/                            # High-resolution output plots (300 DPI)
│   ├── fig1_ensemble_tuning_curves.png # Validation Macro F1 and runtime vs. estimator count M
│   ├── fig2_oob_convergence.png        # Out-of-bag error convergence rate
│   ├── fig3_feature_importance.png     # MDI feature importances (Top 15 features)
│   ├── fig4_critical_difference.png    # Nemenyi Critical Difference (CD) diagram
│   ├── fig5_per_class_f1_radar.png     # Per-class attack category F1-score comparison
│   └── fig6_confusion_matrices.png     # Aggregated 5-fold confusion matrices (Single DT vs RF)
└── tables/                             # Output metrics CSV files
    ├── table1_dataset_summary.csv      # UNSW-NB15 class counts and descriptions
    ├── table2_rf_hyperparams.csv       # Random Forest grid search results
    ├── table3_gbdt_hyperparams.csv     # Gradient Boosting grid search results
    ├── table4_model_comparison.csv     # 5-fold cross-validation performance comparison
    ├── table5_per_class_metrics.csv    # Per-class Precision, Recall, F1 for all 10 categories
    └── table6_statistical_ranks.csv    # Friedman average ranks and pairwise fold scores
```

---

## 3. System Requirements & Dependencies

- Python 3.9+
- Packages:
  - `numpy`
  - `pandas`
  - `scipy`
  - `matplotlib`
  - `seaborn`
- TeX Live 2026 (for compiling `pdflatex`)

---

## 4. Execution Commands

To execute the entire experimental pipeline, run:

```bash
# Non-interactive headless execution (saves 300 DPI figures to disk without blocking)
PYTHONPATH=assignment_4 python3 assignment_4/run_all.py --no-show

# Fast validation mode on subsampled dataset
PYTHONPATH=assignment_4 python3 assignment_4/run_all.py --fast-test --no-show
```

To compile the LaTeX PDF report:

```bash
pdflatex assignment_4/24888923RW441assignment4.tex
pdflatex assignment_4/24888923RW441assignment4.tex
```

---

## 5. Submission Packaging Protocol

To create the submission ZIP archive:

```bash
zip -r 24888923RW441assignment4.zip \
    24888923RW441assignment4.pdf \
    24888923RW441assignment4.tex \
    run_all.py \
    README.md \
    src/ \
    datasets/ \
    figures/ \
    tables/
```
