"""
Non-Parametric Statistical Protocol for Machine Learning Algorithm Comparison.
Implements Friedman omnibus test, Iman-Davenport correction, Nemenyi CD calculation, and CD plot generator.
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def compute_ranks(performance_matrix):
    """
    Compute ranks per fold for each model. Higher score gets rank 1 (best).

    Parameters:
    -----------
    performance_matrix : np.ndarray (N_folds, K_models)
        Performance values where higher is better.

    Returns:
    --------
    np.ndarray (N_folds, K_models)
        Rank matrix where 1 is best rank.
    """
    # Negative matrix so scipy.stats.rankdata gives rank 1 to largest value
    ranks = np.zeros_like(performance_matrix, dtype=np.float64)
    for i in range(performance_matrix.shape[0]):
        ranks[i, :] = stats.rankdata(-performance_matrix[i, :], method='average')
    return ranks


def friedman_test(performance_matrix):
    """
    Perform Friedman omnibus test and Iman-Davenport correction.

    Parameters:
    -----------
    performance_matrix : np.ndarray (N_folds, K_models)
        Matrix of metric scores across evaluation folds (rows) for each model (columns).

    Returns:
    --------
    dict containing:
        - 'avg_ranks': np.ndarray of average ranks per model
        - 'chi2_F': Friedman chi-square statistic
        - 'p_chi2': p-value of chi-square test
        - 'F_F': Iman-Davenport F-statistic
        - 'p_F': p-value of F-statistic
        - 'df1', 'df2': degrees of freedom
    """
    N, K = performance_matrix.shape
    ranks = compute_ranks(performance_matrix)
    R_j = np.mean(ranks, axis=0)

    # Friedman statistic chi2_F
    sum_sq_ranks = np.sum((R_j - (K + 1) / 2.0) ** 2)
    chi2_F = (12.0 * N / (K * (K + 1))) * sum_sq_ranks
    p_chi2 = float(stats.chi2.sf(chi2_F, df=K - 1))

    # Iman-Davenport statistic F_F
    num = (N - 1) * chi2_F
    den = N * (K - 1) - chi2_F
    if den > 0:
        F_F = float(num / den)
        df1 = K - 1
        df2 = (K - 1) * (N - 1)
        p_F = float(stats.f.sf(F_F, df1, df2))
    else:
        F_F = float('inf')
        df1 = K - 1
        df2 = (K - 1) * (N - 1)
        p_F = 0.0

    return {
        'avg_ranks': R_j,
        'ranks': ranks,
        'chi2_F': float(chi2_F),
        'p_chi2': p_chi2,
        'F_F': F_F,
        'p_F': p_F,
        'df1': df1,
        'df2': df2,
        'N': N,
        'K': K
    }


def nemenyi_cd(K, N, alpha=0.05):
    """
    Compute Critical Difference (CD) threshold for Nemenyi post-hoc test.

    q_alpha values for alpha=0.05:
    K=2: 1.960, K=3: 2.343, K=4: 2.569, K=5: 2.728, K=6: 2.850, K=7: 2.949, K=8: 3.031, K=9: 3.102, K=10: 3.164
    """
    q_alpha_map = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164
    }
    q_alpha = q_alpha_map.get(K, 2.728)
    cd = q_alpha * np.sqrt((K * (K + 1)) / (6.0 * N))
    return float(cd)


def generate_cd_plot(avg_ranks, model_names, cd, filename='figures/fig4_critical_difference.png'):
    """
    Generate and save a publication-quality Critical Difference (CD) plot.
    """
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=300)
    
    K = len(model_names)
    sorted_indices = np.argsort(avg_ranks)
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_ranks = [avg_ranks[i] for i in sorted_indices]

    # Draw horizontal axis representing ranks (1 to K)
    ax.hlines(0, 1, K, colors='black', linewidth=1.5)
    for r in range(1, K + 1):
        ax.plot([r, r], [-0.05, 0.05], color='black', linewidth=1.5)
        ax.text(r, -0.15, f"{r}", ha='center', va='top', fontsize=10, fontweight='bold')

    ax.text((K + 1) / 2.0, -0.35, "Average Rank (Lower is Better)", ha='center', va='top', fontsize=11, fontweight='bold')

    # Draw CD indicator line top right
    cd_start = 1.0
    ax.hlines(0.6, cd_start, cd_start + cd, colors='crimson', linewidth=2.5)
    ax.text(cd_start + cd / 2.0, 0.65, f"CD = {cd:.3f}", ha='center', va='bottom', color='crimson', fontsize=10, fontweight='bold')

    # Draw model markers and labels (stagger left and right)
    left_models = sorted_names[: (K + 1) // 2]
    left_ranks = sorted_ranks[: (K + 1) // 2]
    right_models = sorted_names[(K + 1) // 2:]
    right_ranks = sorted_ranks[(K + 1) // 2:]

    # Left side (rank 1 to middle)
    y_pos = 0.1
    for name, r in zip(left_models, left_ranks):
        ax.plot([r, r], [0, y_pos], color='navy', linestyle='--', linewidth=1.0)
        ax.plot(r, y_pos, 'o', color='navy', markersize=6)
        ax.text(r, y_pos + 0.05, f"{name} ({r:.2f})", ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=30)
        y_pos += 0.15

    # Right side
    y_pos = 0.1
    for name, r in zip(right_models, right_ranks):
        ax.plot([r, r], [0, y_pos], color='darkred', linestyle='--', linewidth=1.0)
        ax.plot(r, y_pos, 's', color='darkred', markersize=6)
        ax.text(r, y_pos + 0.05, f"{name} ({r:.2f})", ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=-30)
        y_pos += 0.15

    # Draw non-significant difference bars (cliques where rank diff <= CD)
    clique_y = 0.12
    for i in range(K):
        for j in range(i + 1, K):
            if abs(sorted_ranks[i] - sorted_ranks[j]) <= cd:
                ax.hlines(clique_y, sorted_ranks[i], sorted_ranks[j], colors='darkgray', linewidth=3.0, alpha=0.7)
                clique_y += 0.04

    ax.set_xlim(0.5, K + 0.5)
    ax.set_ylim(-0.5, 1.2)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename
