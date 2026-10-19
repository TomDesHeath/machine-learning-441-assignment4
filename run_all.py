"""
Master Execution Script for CS441/741 Assignment 4: Ensemble Learning.
Executes 5-fold Stratified CV, hyperparameter grid search, model benchmarking,
generates output CSV tables, creates 300 DPI publication plots, and reports statistical significance.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src modules can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_raw_dataset, get_feature_column_names, ATTACK_CATEGORY_MAP
from src.preprocessing import UNSWPreprocessor
from src.decision_tree import DecisionTreeClassifier
from src.knn import KNearestNeighbours
from src.random_forest import RandomForestClassifier
from src.gradient_boosting import GradientBoostingClassifier
from src.stacking import StackingClassifier
from src.metrics import get_all_metrics, confusion_matrix, precision_recall_f1_score, accuracy_score, balanced_accuracy_score, cohens_kappa_score
from src.statistical_tests import friedman_test, nemenyi_cd, generate_cd_plot


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.join(script_dir, 'datasets', 'networkTraffic.csv')
    parser = argparse.ArgumentParser(description="Run Assignment 4 Ensemble Learning Pipeline.")
    parser.add_argument('--no-show', action='store_true', help="Do not display interactive matplotlib windows.")
    parser.add_argument('--fast-test', action='store_true', help="Run on a subsample of dataset for rapid validation.")
    parser.add_argument('--data-path', type=str, default=default_data, help="Path to UNSW-NB15 dataset.")
    return parser.parse_args()


def run_pipeline():
    args = parse_args()
    if args.no_show:
        plt.switch_backend('Agg')

    start_time_pipeline = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, 'figures')
    tab_dir = os.path.join(script_dir, 'tables')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    print("================================================================================")
    print("CS441/741 ASSIGNMENT 4: ENSEMBLE LEARNING EXECUTION PIPELINE")
    print("================================================================================")

    # -------------------------------------------------------------------------
    # 1. DATA LOADING & SUMMARY
    # -------------------------------------------------------------------------
    print("\n[Phase 1] Loading UNSW-NB15 Dataset...")
    df = load_raw_dataset(args.data_path)
    nominal_cols, numeric_cols, target_col = get_feature_column_names(df)

    if args.fast_test:
        print("--> FAST TEST MODE ACTIVATED: Subsampling 20,000 instances.")
        df = df.sample(n=min(20000, len(df)), random_state=42).reset_index(drop=True)

    N_total = len(df)
    print(f"Dataset successfully loaded: {N_total} total instances, {len(nominal_cols)} nominal, {len(numeric_cols)} numeric features.")

    # Table 1: Dataset Summary
    print("\nGenerating Table 1: Dataset Class Distribution Summary...")
    class_counts = df[target_col].value_counts().sort_index()
    table1_rows = []
    for c_id, count in class_counts.items():
        c_name = ATTACK_CATEGORY_MAP.get(c_id, f"Class {c_id}")
        pct = (count / N_total) * 100.0
        desc = "Normal Background Traffic" if c_id == 0 else f"Attack Cyber Threat ({c_name})"
        table1_rows.append({
            'Class ID': c_id,
            'Class Name': c_name,
            'Sample Count': count,
            'Percentage (%)': f"{pct:.2f}%",
            'Description': desc
        })

    df_table1 = pd.DataFrame(table1_rows)
    df_table1.to_csv(os.path.join(tab_dir, 'table1_dataset_summary.csv'), index=False)
    print("Saved tables/table1_dataset_summary.csv")

    # Preprocess Feature Matrices
    print("\nPreprocessing Features Matrix...")
    prep_tree = UNSWPreprocessor(nominal_cols, numeric_cols, scale_numeric=False)
    X_tree = prep_tree.fit_transform(df)

    prep_knn = UNSWPreprocessor(nominal_cols, numeric_cols, scale_numeric=True)
    X_knn = prep_knn.fit_transform(df)

    y = df[target_col].values.astype(np.int64)
    feature_names = prep_tree.feature_names_
    n_classes = len(np.unique(y))

    print(f"Transformed tree feature matrix shape: {X_tree.shape}")
    print(f"Transformed kNN feature matrix shape: {X_knn.shape}")

    # -------------------------------------------------------------------------
    # 2. STRATIFIED CROSS-VALIDATION INDEX GENERATION (5 FOLDS)
    # -------------------------------------------------------------------------
    n_splits = 5
    rng = np.random.RandomState(42)
    classes, counts = np.unique(y, return_counts=True)
    class_indices = {c: np.where(y == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(class_indices[c])

    cv_folds = [[] for _ in range(n_splits)]
    for c in classes:
        c_idx = class_indices[c]
        sizes = np.full(n_splits, len(c_idx) // n_splits)
        sizes[: len(c_idx) % n_splits] += 1
        curr = 0
        for k, sz in enumerate(sizes):
            cv_folds[k].extend(c_idx[curr : curr + sz])
            curr += sz

    all_idx = np.arange(N_total)
    splits = []
    for k in range(n_splits):
        val_idx = np.array(cv_folds[k], dtype=np.int64)
        tr_idx = np.setdiff1d(all_idx, val_idx)
        splits.append((tr_idx, val_idx))

    # -------------------------------------------------------------------------
    # 3. HYPERPARAMETER GRID SEARCH FOR RANDOM FOREST & GBDT
    # -------------------------------------------------------------------------
    print("\n[Phase 2] Random Forest Grid Search & Tuning...")
    rf_grid = [
        {'n_estimators': 10, 'max_depth': 10},
        {'n_estimators': 50, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 15},
        {'n_estimators': 200, 'max_depth': 20},
    ]

    table2_rows = []
    tr_idx0, val_idx0 = splits[0]

    for param in rf_grid:
        m_est = param['n_estimators']
        d_max = param['max_depth']
        print(f"  Evaluating RF (M={m_est}, depth={d_max})...")
        t0 = time.time()
        rf_temp = RandomForestClassifier(n_estimators=m_est, max_depth=d_max, class_weight='balanced', random_state=42, n_jobs=-1)
        rf_temp.fit(X_tree[tr_idx0], y[tr_idx0])
        fit_t = time.time() - t0

        tr_preds = rf_temp.predict(X_tree[tr_idx0])
        val_preds = rf_temp.predict(X_tree[val_idx0])

        tr_acc = accuracy_score(y[tr_idx0], tr_preds)
        val_acc = accuracy_score(y[val_idx0], val_preds)
        _, _, val_macro_f1 = precision_recall_f1_score(y[val_idx0], val_preds, average='macro', num_classes=n_classes)
        oob_err = 1.0 - rf_temp.oob_score_ if rf_temp.oob_score_ > 0 else 0.0

        table2_rows.append({
            'Estimators (M)': m_est,
            'Max Depth': d_max,
            'Subspace Features': 'sqrt(D)',
            'Train Acc': f"{tr_acc:.4f}",
            'Val Acc': f"{val_acc:.4f}",
            'Val Macro F1': f"{val_macro_f1:.4f}",
            'OOB Error': f"{oob_err:.4f}",
            'Fit Time (s)': f"{fit_t:.2f}"
        })

    df_table2 = pd.DataFrame(table2_rows)
    df_table2.to_csv(os.path.join(tab_dir, 'table2_rf_hyperparams.csv'), index=False)
    print("Saved tables/table2_rf_hyperparams.csv")

    print("\n[Phase 3] Gradient Boosting Grid Search & Tuning...")
    gbdt_grid = [
        {'n_estimators': 20, 'learning_rate': 0.1, 'max_depth': 3},
        {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3},
        {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 5},
        {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 5},
    ]

    table3_rows = []
    for param in gbdt_grid:
        m_est = param['n_estimators']
        lr = param['learning_rate']
        d_max = param['max_depth']
        print(f"  Evaluating GBDT (M={m_est}, lr={lr}, depth={d_max})...")
        t0 = time.time()
        gb_temp = GradientBoostingClassifier(n_estimators=m_est, learning_rate=lr, max_depth=d_max, random_state=42)
        gb_temp.fit(X_tree[tr_idx0], y[tr_idx0])
        fit_t = time.time() - t0

        tr_preds = gb_temp.predict(X_tree[tr_idx0])
        val_preds = gb_temp.predict(X_tree[val_idx0])

        tr_acc = accuracy_score(y[tr_idx0], tr_preds)
        val_acc = accuracy_score(y[val_idx0], val_preds)
        _, _, val_macro_f1 = precision_recall_f1_score(y[val_idx0], val_preds, average='macro', num_classes=n_classes)

        table3_rows.append({
            'Estimators (M)': m_est,
            'Learning Rate': lr,
            'Max Depth': d_max,
            'Train Acc': f"{tr_acc:.4f}",
            'Val Acc': f"{val_acc:.4f}",
            'Val Macro F1': f"{val_macro_f1:.4f}",
            'Fit Time (s)': f"{fit_t:.2f}"
        })

    df_table3 = pd.DataFrame(table3_rows)
    df_table3.to_csv(os.path.join(tab_dir, 'table3_gbdt_hyperparams.csv'), index=False)
    print("Saved tables/table3_gbdt_hyperparams.csv")

    # -------------------------------------------------------------------------
    # 4. FULL 5-FOLD BENCHMARK COMPARISON ACROSS ALL ALGORITHMS
    # -------------------------------------------------------------------------
    print("\n[Phase 4] Full 5-Fold Cross-Validation Model Evaluation...")

    models_to_eval = [
        ('Single Decision Tree', lambda: DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=42), 'tree'),
        ('k-Nearest Neighbours', lambda: KNearestNeighbours(n_neighbors=5, weights='distance'), 'knn'),
        ('Random Forest', lambda: RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1), 'tree'),
        ('Gradient Boosting', lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42), 'tree'),
    ]

    model_fold_scores = {name: [] for name, _, _ in models_to_eval}
    model_metrics_summary = {}

    cm_dt_accum = np.zeros((n_classes, n_classes), dtype=np.int64)
    cm_rf_accum = np.zeros((n_classes, n_classes), dtype=np.int64)
    best_rf_model = None

    for name, model_factory, data_type in models_to_eval:
        print(f"\nEvaluating Model: {name}")
        fold_accs, fold_bal_accs, fold_f1s, fold_wf1s, fold_kappas = [], [], [], [], []
        fit_times, pred_times = [], []

        for k, (tr_idx, val_idx) in enumerate(splits):
            X_tr = X_knn[tr_idx] if data_type == 'knn' else X_tree[tr_idx]
            X_val = X_knn[val_idx] if data_type == 'knn' else X_tree[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            model = model_factory()

            t0 = time.time()
            model.fit(X_tr, y_tr)
            fit_t = time.time() - t0

            t0 = time.time()
            preds = model.predict(X_val)
            pred_t = time.time() - t0

            fit_times.append(fit_t)
            pred_times.append(pred_t)

            acc = accuracy_score(y_val, preds)
            bal_acc = balanced_accuracy_score(y_val, preds, num_classes=n_classes)
            _, _, macro_f1 = precision_recall_f1_score(y_val, preds, average='macro', num_classes=n_classes)
            _, _, weighted_f1 = precision_recall_f1_score(y_val, preds, average='weighted', num_classes=n_classes)
            kappa = cohens_kappa_score(y_val, preds, num_classes=n_classes)

            fold_accs.append(acc)
            fold_bal_accs.append(bal_acc)
            fold_f1s.append(macro_f1)
            fold_wf1s.append(weighted_f1)
            fold_kappas.append(kappa)

            model_fold_scores[name].append(macro_f1)

            # Accumulate confusion matrices for Q1 baseline DT vs Q2 RF
            if name == 'Single Decision Tree':
                cm_dt_accum += confusion_matrix(y_val, preds, num_classes=n_classes)
            elif name == 'Random Forest':
                cm_rf_accum += confusion_matrix(y_val, preds, num_classes=n_classes)
                if k == 0:
                    best_rf_model = model

            print(f"  Fold {k+1}/5 - Acc: {acc:.4f}, BalAcc: {bal_acc:.4f}, MacroF1: {macro_f1:.4f} (Fit: {fit_t:.1f}s, Pred: {pred_t:.1f}s)")

        model_metrics_summary[name] = {
            'Accuracy': f"{np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}",
            'Balanced Accuracy': f"{np.mean(fold_bal_accs):.4f} +/- {np.std(fold_bal_accs):.4f}",
            'Macro F1': f"{np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}",
            'Weighted F1': f"{np.mean(fold_wf1s):.4f} +/- {np.std(fold_wf1s):.4f}",
            'Cohen Kappa': f"{np.mean(fold_kappas):.4f} +/- {np.std(fold_kappas):.4f}",
            'Fit Time (s)': f"{np.mean(fit_times):.2f}",
            'Pred Time (s)': f"{np.mean(pred_times):.2f}"
        }

    # Stacking Classifier Evaluation
    print("\nEvaluating Model: Stacking Classifier")
    stack_fold_f1s, stack_fit_times, stack_pred_times = [], [], []
    stack_accs, stack_bal_accs, stack_wf1s, stack_kappas = [], [], [], []

    for k, (tr_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X_tree[tr_idx], y[tr_idx]
        X_val, y_val = X_tree[val_idx], y[val_idx]

        base_estimators = [
            ('dt', DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42))
        ]
        stack_model = StackingClassifier(estimators=base_estimators, cv=3, random_state=42)

        t0 = time.time()
        stack_model.fit(X_tr, y_tr)
        fit_t = time.time() - t0

        t0 = time.time()
        preds = stack_model.predict(X_val)
        pred_t = time.time() - t0

        acc = accuracy_score(y_val, preds)
        bal_acc = balanced_accuracy_score(y_val, preds, num_classes=n_classes)
        _, _, macro_f1 = precision_recall_f1_score(y_val, preds, average='macro', num_classes=n_classes)
        _, _, weighted_f1 = precision_recall_f1_score(y_val, preds, average='weighted', num_classes=n_classes)
        kappa = cohens_kappa_score(y_val, preds, num_classes=n_classes)

        stack_accs.append(acc)
        stack_bal_accs.append(bal_acc)
        stack_fold_f1s.append(macro_f1)
        stack_wf1s.append(weighted_f1)
        stack_kappas.append(kappa)
        stack_fit_times.append(fit_t)
        stack_pred_times.append(pred_t)
        print(f"  Fold {k+1}/5 - Acc: {acc:.4f}, BalAcc: {bal_acc:.4f}, MacroF1: {macro_f1:.4f} (Fit: {fit_t:.1f}s)")

    model_fold_scores['Stacking Classifier'] = stack_fold_f1s
    model_metrics_summary['Stacking Classifier'] = {
        'Accuracy': f"{np.mean(stack_accs):.4f} +/- {np.std(stack_accs):.4f}",
        'Balanced Accuracy': f"{np.mean(stack_bal_accs):.4f} +/- {np.std(stack_bal_accs):.4f}",
        'Macro F1': f"{np.mean(stack_fold_f1s):.4f} +/- {np.std(stack_fold_f1s):.4f}",
        'Weighted F1': f"{np.mean(stack_wf1s):.4f} +/- {np.std(stack_wf1s):.4f}",
        'Cohen Kappa': f"{np.mean(stack_kappas):.4f} +/- {np.std(stack_kappas):.4f}",
        'Fit Time (s)': f"{np.mean(stack_fit_times):.2f}",
        'Pred Time (s)': f"{np.mean(stack_pred_times):.2f}"
    }

    # Table 4: Model Comparison Summary
    table4_rows = []
    for m_name, metrics in model_metrics_summary.items():
        table4_rows.append({
            'Model Name': m_name,
            'Accuracy': metrics['Accuracy'],
            'Balanced Accuracy': metrics['Balanced Accuracy'],
            'Macro F1': metrics['Macro F1'],
            'Weighted F1': metrics['Weighted F1'],
            "Cohen's Kappa": metrics['Cohen Kappa'],
            'Fit Time (s)': metrics['Fit Time (s)'],
            'Inference Latency (s)': metrics['Pred Time (s)']
        })
    df_table4 = pd.DataFrame(table4_rows)
    df_table4.to_csv(os.path.join(tab_dir, 'table4_model_comparison.csv'), index=False)
    print("\nSaved tables/table4_model_comparison.csv")

    # Table 5: Per-Class Performance
    print("\nGenerating Table 5: Per-Class Detailed Performance...")
    # Evaluate Single DT vs Random Forest vs GBDT per class on fold 0
    X_tr0, y_tr0 = X_tree[splits[0][0]], y[splits[0][0]]
    X_v0, y_v0 = X_tree[splits[0][1]], y[splits[0][1]]

    dt_eval = DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=42).fit(X_tr0, y_tr0)
    rf_eval = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1).fit(X_tr0, y_tr0)
    gb_eval = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42).fit(X_tr0, y_tr0)

    p_dt, r_dt, f1_dt, sup_dt = precision_recall_f1_score(y_v0, dt_eval.predict(X_v0), average=None, num_classes=n_classes)
    p_rf, r_rf, f1_rf, sup_rf = precision_recall_f1_score(y_v0, rf_eval.predict(X_v0), average=None, num_classes=n_classes)
    p_gb, r_gb, f1_gb, sup_gb = precision_recall_f1_score(y_v0, gb_eval.predict(X_v0), average=None, num_classes=n_classes)

    table5_rows = []
    for c in range(n_classes):
        c_name = ATTACK_CATEGORY_MAP.get(c, f"Class {c}")
        table5_rows.append({'Class ID': c, 'Class Name': c_name, 'Model': 'Single DT', 'Precision': f"{p_dt[c]:.4f}", 'Recall': f"{r_dt[c]:.4f}", 'F1-Score': f"{f1_dt[c]:.4f}", 'Support': int(sup_dt[c])})
        table5_rows.append({'Class ID': c, 'Class Name': c_name, 'Model': 'Random Forest', 'Precision': f"{p_rf[c]:.4f}", 'Recall': f"{r_rf[c]:.4f}", 'F1-Score': f"{f1_rf[c]:.4f}", 'Support': int(sup_rf[c])})
        table5_rows.append({'Class ID': c, 'Class Name': c_name, 'Model': 'Gradient Boosting', 'Precision': f"{p_gb[c]:.4f}", 'Recall': f"{r_gb[c]:.4f}", 'F1-Score': f"{f1_gb[c]:.4f}", 'Support': int(sup_gb[c])})

    df_table5 = pd.DataFrame(table5_rows)
    df_table5.to_csv(os.path.join(tab_dir, 'table5_per_class_metrics.csv'), index=False)
    print("Saved tables/table5_per_class_metrics.csv")

    # Table 6 & Statistical Evaluation
    print("\n[Phase 5] Non-Parametric Statistical Testing (Friedman & Nemenyi)...")
    all_models_order = ['Single Decision Tree', 'k-Nearest Neighbours', 'Random Forest', 'Gradient Boosting', 'Stacking Classifier']
    perf_matrix = np.zeros((n_splits, len(all_models_order)), dtype=np.float64)

    for idx, m_name in enumerate(all_models_order):
        perf_matrix[:, idx] = model_fold_scores[m_name]

    friedman_res = friedman_test(perf_matrix)
    cd_val = nemenyi_cd(len(all_models_order), n_splits, alpha=0.05)

    print(f"Friedman chi2_F: {friedman_res['chi2_F']:.4f} (p-val: {friedman_res['p_chi2']:.4e})")
    print(f"Iman-Davenport F_F: {friedman_res['F_F']:.4f} (p-val: {friedman_res['p_F']:.4e})")
    print(f"Nemenyi Critical Difference (CD) at alpha=0.05: {cd_val:.4f}")

    table6_rows = []
    for idx, m_name in enumerate(all_models_order):
        row = {'Model Name': m_name}
        for k in range(n_splits):
            row[f'Fold {k+1}'] = f"{perf_matrix[k, idx]:.4f}"
        row['Average Rank'] = f"{friedman_res['avg_ranks'][idx]:.2f}"
        table6_rows.append(row)

    df_table6 = pd.DataFrame(table6_rows)
    df_table6.to_csv(os.path.join(tab_dir, 'table6_statistical_ranks.csv'), index=False)
    print("Saved tables/table6_statistical_ranks.csv")

    # -------------------------------------------------------------------------
    # 5. GRAPHIC ARTIFACT GENERATION (FIGURES 1 TO 6)
    # -------------------------------------------------------------------------
    print("\n[Phase 6] Generating High-Resolution Graphic Artifacts (300 DPI)...")

    # Fig 1: Ensemble Tuning Curves
    plt.figure(figsize=(10, 4), dpi=300)
    plt.subplot(1, 2, 1)
    m_vals = [10, 50, 100, 200]
    rf_f1s = [float(r['Val Macro F1']) for r in table2_rows]
    plt.plot(m_vals, rf_f1s, 'o-', color='navy', linewidth=2, label='Random Forest (depth=15)')
    gb_m_vals = [20, 50, 100, 150]
    gb_f1s = [float(r['Val Macro F1']) for r in table3_rows]
    plt.plot(gb_m_vals, gb_f1s, 's--', color='crimson', linewidth=2, label='Gradient Boosting (lr=0.05)')
    plt.xlabel('Number of Estimators (M)', fontweight='bold')
    plt.ylabel('Validation Macro F1-Score', fontweight='bold')
    plt.title('(a) Macro F1 Convergence', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    rf_times = [float(r['Fit Time (s)']) for r in table2_rows]
    gb_times = [float(r['Fit Time (s)']) for r in table3_rows]
    plt.plot(m_vals, rf_times, 'o-', color='navy', linewidth=2, label='Random Forest Fit Time')
    plt.plot(gb_m_vals, gb_times, 's--', color='crimson', linewidth=2, label='Gradient Boosting Fit Time')
    plt.xlabel('Number of Estimators (M)', fontweight='bold')
    plt.ylabel('Training Duration (Seconds)', fontweight='bold')
    plt.title('(b) Algorithmic Runtime Complexity', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig1_ensemble_tuning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved figures/fig1_ensemble_tuning_curves.png")

    # Fig 2: OOB Error Convergence
    plt.figure(figsize=(6, 4.5), dpi=300)
    oob_errs = [float(r['OOB Error']) for r in table2_rows]
    plt.plot(m_vals, oob_errs, 'D-', color='darkgreen', linewidth=2.5, markersize=8, label='Out-of-Bag (OOB) Generalization Error')
    plt.xlabel('Ensemble Estimators (M)', fontweight='bold')
    plt.ylabel('Generalization Error Rate', fontweight='bold')
    plt.title('Random Forest OOB Convergence Profile', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig2_oob_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved figures/fig2_oob_convergence.png")

    # Fig 3: MDI Feature Importance
    if best_rf_model is not None and best_rf_model.feature_importances_ is not None:
        plt.figure(figsize=(8, 6), dpi=300)
        importances = best_rf_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:15]
        top_names = [feature_names[i] for i in sorted_idx]
        top_imps = importances[sorted_idx]

        sns.barplot(x=top_imps, y=top_names, palette='Blues_r')
        plt.xlabel('Mean Decrease in Impurity (MDI)', fontweight='bold')
        plt.ylabel('UNSW-NB15 Descriptive Features', fontweight='bold')
        plt.title('Top 15 Feature Importances (Random Forest Ensemble)', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig3_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved figures/fig3_feature_importance.png")

    # Fig 4: Critical Difference Plot
    generate_cd_plot(friedman_res['avg_ranks'], all_models_order, cd_val, filename=os.path.join(fig_dir, 'fig4_critical_difference.png'))
    print("Saved figures/fig4_critical_difference.png")

    # Fig 5: Per-Class F1 Radar / Grouped Bar Chart
    plt.figure(figsize=(9, 4.5), dpi=300)
    x_cls = np.arange(n_classes)
    width = 0.25

    plt.bar(x_cls - width, f1_dt, width, label='Single Decision Tree', color='gray', edgecolor='black')
    plt.bar(x_cls, f1_rf, width, label='Random Forest', color='navy', edgecolor='black')
    plt.bar(x_cls + width, f1_gb, width, label='Gradient Boosting', color='crimson', edgecolor='black')

    plt.xlabel('Target Class Category', fontweight='bold')
    plt.ylabel('F1-Score', fontweight='bold')
    plt.title('Per-Class Attack Classification F1-Score Comparison', fontweight='bold')
    plt.xticks(x_cls, [ATTACK_CATEGORY_MAP.get(c, f"{c}") for c in range(n_classes)], rotation=30, ha='right')
    plt.ylim(0, 1.05)
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig5_per_class_f1_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved figures/fig5_per_class_f1_radar.png")

    # Fig 6: Aggregated Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    cm_dt_norm = cm_dt_accum.astype('float') / cm_dt_accum.sum(axis=1)[:, np.newaxis]
    cm_rf_norm = cm_rf_accum.astype('float') / cm_rf_accum.sum(axis=1)[:, np.newaxis]

    labels = [ATTACK_CATEGORY_MAP.get(c, f"{c}") for c in range(n_classes)]

    sns.heatmap(cm_dt_norm, ax=axes[0], annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
    axes[0].set_title('(a) Single Decision Tree Confusion Matrix', fontweight='bold')
    axes[0].set_xlabel('Predicted Class', fontweight='bold')
    axes[0].set_ylabel('True Class', fontweight='bold')

    sns.heatmap(cm_rf_norm, ax=axes[1], annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
    axes[1].set_title('(b) Random Forest Ensemble Confusion Matrix', fontweight='bold')
    axes[1].set_xlabel('Predicted Class', fontweight='bold')
    axes[1].set_ylabel('True Class', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig6_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved figures/fig6_confusion_matrices.png")

    elapsed_total = time.time() - start_time_pipeline
    print("\n================================================================================")
    print(f"PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {elapsed_total:.2f} SECONDS")
    print("================================================================================")


if __name__ == '__main__':
    run_pipeline()
