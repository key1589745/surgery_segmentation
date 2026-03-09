import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
import pandas as pd
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ================================
# 1. Core DPP Selection with γ
# ================================

def dpp_selection_with_gamma(
    memory_features: np.ndarray,      # (n_memories, d)
    current_feature: np.ndarray,      # (d,)
    k: int,
    gamma: float,
    selection_method: str = 'greedy'  # 'greedy' or 'exact'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    DPP selection using your S and R formulation with γ.
    
    Args:
        memory_features: Features of memory candidates
        current_feature: Feature of current frame
        k: Number to select
        gamma: Balance parameter (your hyperparameter)
        selection_method: 'greedy' (fast) or 'exact' (slow but optimal)
    
    Returns:
        selected_indices, metadata
    """
    n = len(memory_features)
    
    # 1. Compute similarity matrix S
    S = memory_features @ memory_features.T  # (n, n)
    
    # 2. Compute quality scores using γ
    # q_i = exp(γ * similarity_to_current)
    similarities_to_current = memory_features @ current_feature  # (n,)
    quality_scores = np.exp(gamma * similarities_to_current)  # (n,)
    
    # 3. Construct DPP kernel L = S ⊙ (q q^⊤)
    L = S * np.outer(quality_scores, quality_scores)
    
    # Add small diagonal for numerical stability
    L += np.eye(n) * 1e-8
    
    # 4. Select subset using DPP
    if selection_method == 'greedy':
        selected = greedy_dpp_selection(L, k)
    else:
        selected = exact_dpp_selection(L, k)
    
    # Compute metrics
    selected_features = memory_features[selected]
    selected_qualities = quality_scores[selected]
    
    # Diversity: log determinant of similarity submatrix
    S_selected = selected_features @ selected_features.T
    S_selected += np.eye(k) * 1e-8
    diversity = np.log(np.linalg.det(S_selected))
    
    # Average similarity to current frame
    avg_similarity_to_current = np.mean(similarities_to_current[selected])
    
    # Effective quality-diversity balance
    # Higher γ gives more weight to current frame similarity
    effective_gamma = gamma
    
    metadata = {
        'diversity': diversity,
        'avg_quality': np.mean(selected_qualities),
        'avg_similarity_to_current': avg_similarity_to_current,
        'quality_scores': quality_scores,
        'similarities_to_current': similarities_to_current,
        'L_matrix': L,
        'effective_gamma': effective_gamma
    }
    
    return selected, metadata

def greedy_dpp_selection(L: np.ndarray, k: int) -> np.ndarray:
    """Greedy DPP selection (fast approximation)."""
    n = L.shape[0]
    selected = []
    remaining = list(range(n))
    
    for _ in range(k):
        if not selected:
            # First item: choose with highest diagonal (L_ii = q_i^2)
            scores = np.diag(L)
        else:
            # Greedy: choose item that maximizes log det
            scores = []
            for i in remaining:
                if not selected:
                    scores.append(L[i, i])
                else:
                    # Compute determinant if we add i
                    sel_idx = selected + [i]
                    submatrix = L[np.ix_(sel_idx, sel_idx)]
                    scores.append(np.linalg.det(submatrix))
            scores = np.array(scores)
        
        best_idx = remaining[np.argmax(scores)]
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return np.array(selected)

# ================================
# 2. γ Sensitivity Experiment Design
# ================================

def run_gamma_sensitivity_experiment(
    memory_features: np.ndarray,
    current_features: np.ndarray,  # Multiple frames for robustness
    k: int = 10,
    gamma_values: List[float] = None,
    n_trials: int = 5,
    segmentation_model: callable = None  # Your segmentation model
) -> pd.DataFrame:
    """
    Comprehensive γ sensitivity analysis.
    
    Args:
        memory_features: (n_memories, d)
        current_features: (n_frames, d) - multiple frames
        k: memories to select
        gamma_values: range of γ to test
        n_trials: random trials for statistical significance
        segmentation_model: function that takes (current_frame, selected_memories) → mIoU
    
    Returns:
        DataFrame with results
    """
    if gamma_values is None:
        # Key γ ranges to test:
        # Negative: prefer dissimilar frames (exploration)
        # Small positive: mild preference for similar
        # Large positive: strong preference for similar
        gamma_values = np.concatenate([
            np.linspace(-2, -0.5, 4),   # Negative γ
            np.linspace(-0.4, 0.4, 9),  # Near zero
            np.linspace(0.5, 2, 4),     # Positive
            np.linspace(2.5, 5, 4),     # Strong positive
            [10, 20]                     # Very strong
        ])
    
    results = []
    
    for trial in range(n_trials):
        np.random.seed(42 + trial)
        
        # For each current frame (or average over frames)
        for frame_idx, current_feature in enumerate(current_features[:10]):  # Limit to 10 frames
            for gamma in gamma_values:
                # Select memories using DPP with this γ
                selected, metadata = dpp_selection_with_gamma(
                    memory_features, current_feature, k, gamma, 'greedy'
                )
                
                # Compute segmentation performance
                if segmentation_model is not None:
                    mIoU, mAP = segmentation_model(current_feature, selected)
                else:
                    # Simulated performance (replace with real model)
                    mIoU = simulate_segmentation_performance(
                        memory_features[selected], current_feature, gamma
                    )
                    mAP = mIoU * 0.9  # Simulated
                
                # Compute key metrics
                results.append({
                    'trial': trial,
                    'frame_idx': frame_idx,
                    'gamma': gamma,
                    'selected_indices': selected,
                    'mIoU': mIoU,
                    'mAP': mAP,
                    'diversity': metadata['diversity'],
                    'avg_similarity_to_current': metadata['avg_similarity_to_current'],
                    'avg_quality': metadata['avg_quality'],
                    'quality_std': np.std(metadata['quality_scores']),
                    'selected_similarity_std': np.std(metadata['similarities_to_current'][selected])
                })
    
    return pd.DataFrame(results)

def simulate_segmentation_performance(
    selected_features: np.ndarray,
    current_feature: np.ndarray,
    gamma: float
) -> float:
    """
    Simulate segmentation performance based on selected memories.
    Replace with your actual model evaluation.
    
    Intuition: Performance is good when we have both:
    1. Some similar frames (for context)
    2. Diverse frames (for robustness)
    """
    # Compute similarities between selected memories and current
    similarities = selected_features @ current_feature
    
    # Diversity among selected memories
    selected_similarity = selected_features @ selected_features.T
    selected_similarity += np.eye(len(selected_features)) * 1e-8
    diversity = np.log(np.linalg.det(selected_similarity))
    
    # Balance: want some similarity but not too much
    avg_similarity = np.mean(similarities)
    similarity_penalty = np.exp(-5 * (avg_similarity - 0.5)**2)  # Peak at 0.5
    
    # Simulated performance
    performance = (
        0.6 * similarity_penalty +
        0.4 * np.tanh(diversity / len(selected_features)) +
        np.random.normal(0, 0.05)  # Noise
    )
    
    return np.clip(performance, 0, 1)

# ================================
# 3. Key Visualizations for γ Analysis
# ================================

def plot_gamma_sensitivity_comprehensive(results_df: pd.DataFrame):
    """Create comprehensive visualization suite for γ sensitivity."""
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Main Performance vs γ
    ax1 = plt.subplot(3, 4, 1)
    plot_performance_vs_gamma(results_df, ax1)
    
    # 2. Quality-Diversity Tradeoff
    ax2 = plt.subplot(3, 4, 2)
    plot_quality_diversity_tradeoff(results_df, ax2)
    
    # 3. Similarity Distribution Evolution
    ax3 = plt.subplot(3, 4, 3)
    plot_similarity_distribution(results_df, ax3)
    
    # 4. Effective Temperature Analysis
    ax4 = plt.subplot(3, 4, 4)
    plot_effective_temperature(results_df, ax4)
    
    # 5. Selection Consistency
    ax5 = plt.subplot(3, 4, 5)
    plot_selection_consistency(results_df, ax5)
    
    # 6. γ Response Surface
    ax6 = plt.subplot(3, 4, 6)
    plot_gamma_response_surface(results_df, ax6)
    
    # 7. Statistical Significance
    ax7 = plt.subplot(3, 4, 7)
    plot_statistical_significance(results_df, ax7)
    
    # 8. Optimal γ by Frame Type
    ax8 = plt.subplot(3, 4, 8)
    plot_optimal_gamma_by_frame(results_df, ax8)
    
    # 9. Kernel Spectrum Analysis
    ax9 = plt.subplot(3, 4, 9)
    plot_kernel_spectrum(results_df, ax9)
    
    # 10. Memory Utilization
    ax10 = plt.subplot(3, 4, 10)
    plot_memory_utilization(results_df, ax10)
    
    # 11. Robustness Analysis
    ax11 = plt.subplot(3, 4, 11)
    plot_robustness_analysis(results_df, ax11)
    
    # 12. Recommendations
    ax12 = plt.subplot(3, 4, 12)
    plot_recommendations_summary(results_df, ax12)
    
    plt.suptitle(f'Comprehensive Sensitivity Analysis of Balance Parameter γ', 
                 fontsize=20, y=1.02)
    plt.tight_layout()
    
    return fig

def plot_performance_vs_gamma(results_df, ax):
    """Plot segmentation performance metrics vs γ."""
    
    # Group by gamma and compute statistics
    grouped = results_df.groupby('gamma').agg({
        'mIoU': ['mean', 'std', 'min', 'max'],
        'mAP': ['mean', 'std']
    })
    
    gammas = grouped.index
    mIoU_mean = grouped[('mIoU', 'mean')]
    mIoU_std = grouped[('mIoU', 'std')]
    
    # Plot with confidence intervals
    ax.plot(gammas, mIoU_mean, 'o-', linewidth=2, markersize=6, label='mIoU')
    ax.fill_between(gammas, 
                   mIoU_mean - mIoU_std, 
                   mIoU_mean + mIoU_std, 
                   alpha=0.2)
    
    # Highlight key regions
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='γ=0 (pure diversity)')
    
    # Find optimal γ
    optimal_idx = np.argmax(mIoU_mean.values)
    optimal_gamma = gammas[optimal_idx]
    optimal_mIoU = mIoU_mean.iloc[optimal_idx]
    
    ax.axvline(x=optimal_gamma, color='red', linestyle='--', alpha=0.7)
    ax.plot(optimal_gamma, optimal_mIoU, 'ro', markersize=10)
    
    ax.set_xlabel('γ (Balance Parameter)', fontsize=12)
    ax.set_ylabel('mIoU', fontsize=12)
    ax.set_title('Segmentation Performance vs γ', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate(f'Optimal: γ={optimal_gamma:.2f}\nmIoU={optimal_mIoU:.3f}',
               xy=(optimal_gamma, optimal_mIoU),
               xytext=(optimal_gamma + 0.5, optimal_mIoU - 0.1),
               fontsize=10,
               arrowprops=dict(arrowstyle='->', color='red'))
    
    return optimal_gamma, optimal_mIoU

def plot_quality_diversity_tradeoff(results_df, ax):
    """Plot the fundamental tradeoff controlled by γ."""
    
    # Compute averages
    grouped = results_df.groupby('gamma').mean(numeric_only=True)
    
    # Create scatter with γ as color
    sc = ax.scatter(grouped['diversity'], grouped['avg_similarity_to_current'],
                   c=grouped.index, s=100, cmap='coolwarm', alpha=0.8,
                   edgecolors='black', linewidth=0.5)
    
    # Connect points in γ order
    sorted_idx = np.argsort(grouped.index)
    ax.plot(grouped['diversity'].iloc[sorted_idx], 
            grouped['avg_similarity_to_current'].iloc[sorted_idx],
            'k--', alpha=0.3, linewidth=1)
    
    # Label key γ points
    for gamma in [np.min(grouped.index), 0, np.max(grouped.index)]:
        if gamma in grouped.index:
            idx = grouped.index.get_loc(gamma)
            ax.annotate(f'γ={gamma:.1f}',
                       xy=(grouped['diversity'].iloc[idx], 
                           grouped['avg_similarity_to_current'].iloc[idx]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Diversity (log-det of S)', fontsize=12)
    ax.set_ylabel('Avg Similarity to Current', fontsize=12)
    ax.set_title('Quality-Diversity Tradeoff', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('γ value', fontsize=11)
    
    ax.grid(True, alpha=0.3)

def plot_similarity_distribution(results_df, ax):
    """Show how selected memory similarity distribution changes with γ."""
    
    # Select key γ values
    key_gammas = [-2, -1, 0, 0.5, 1, 2, 5]
    
    for gamma in key_gammas:
        if gamma in results_df['gamma'].values:
            subset = results_df[results_df['gamma'] == gamma]
            
            # Get similarity values
            similarities = []
            for _, row in subset.iterrows():
                similarities.extend([row['avg_similarity_to_current']])
            
            # Plot KDE
            if len(similarities) > 1:
                from scipy.stats import gaussian_kde
                try:
                    kde = gaussian_kde(similarities)
                    x_vals = np.linspace(-1, 1, 100)
                    y_vals = kde(x_vals)
                    ax.plot(x_vals, y_vals, label=f'γ={gamma}', linewidth=2, alpha=0.7)
                except:
                    pass
    
    ax.set_xlabel('Similarity to Current Frame', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Similarity Distribution by γ', fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add vertical line at 0
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

def plot_effective_temperature(results_df, ax):
    """Show γ as effective temperature parameter."""
    
    # Group by gamma
    grouped = results_df.groupby('gamma').agg({
        'quality_std': 'mean',
        'selected_similarity_std': 'mean',
        'mIoU': 'mean'
    })
    
    gammas = grouped.index
    
    # Plot multiple metrics
    ax2 = ax.twinx()
    
    # Line 1: Quality concentration (inverse of std)
    line1, = ax.plot(gammas, 1/(grouped['quality_std'] + 1e-8), 
                    'b-', label='Quality Concentration', linewidth=2)
    
    # Line 2: Performance
    line2, = ax2.plot(gammas, grouped['mIoU'], 
                     'r-', label='mIoU', linewidth=2)
    
    ax.set_xlabel('γ (Effective Temperature)', fontsize=12)
    ax.set_ylabel('Quality Concentration (1/std)', fontsize=12, color='b')
    ax2.set_ylabel('mIoU', fontsize=12, color='r')
    
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    ax.set_title('γ as Temperature Parameter', fontsize=14)
    
    # Add legends
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=10)
    
    ax.grid(True, alpha=0.3)

def plot_selection_consistency(results_df, ax):
    """Plot consistency of selection across trials for the same frame."""
    # Compute Jaccard similarity of selected sets for same frame/gamma across trials
    unique_gammas = sorted(results_df['gamma'].unique())
    consistencies = []
    
    for gamma in unique_gammas:
        subset = results_df[results_df['gamma'] == gamma]
        frame_consistencies = []
        
        for frame_idx in subset['frame_idx'].unique():
            frame_trials = subset[subset['frame_idx'] == frame_idx]
            if len(frame_trials) < 2:
                continue
            
            # Pairwise Jaccard similarity between trials
            sets = [set(idx_list) for idx_list in frame_trials['selected_indices']]
            jaccards = []
            for i in range(len(sets)):
                for j in range(i+1, len(sets)):
                    intersection = len(sets[i].intersection(sets[j]))
                    union = len(sets[i].union(sets[j]))
                    if union > 0:
                        jaccards.append(intersection / union)
            
            if jaccards:
                frame_consistencies.append(np.mean(jaccards))
        
        if frame_consistencies:
            consistencies.append(np.mean(frame_consistencies))
        else:
            consistencies.append(0)
            
    ax.plot(unique_gammas, consistencies, 'g-o', linewidth=2)
    ax.set_xlabel('γ', fontsize=12)
    ax.set_ylabel('Selection Consistency (Jaccard)', fontsize=12)
    ax.set_title('Selection Stability vs γ', fontsize=14)
    ax.grid(True, alpha=0.3)

def plot_gamma_response_surface(results_df, ax):
    """Plot heatmap of mIoU vs Gamma and Diversity."""
    # Bin diversity to create a grid
    results_df['diversity_bin'] = pd.qcut(results_df['diversity'], q=10, duplicates='drop')
    
    pivot = results_df.pivot_table(
        values='mIoU', 
        index='diversity_bin', 
        columns=pd.cut(results_df['gamma'], bins=10),
        aggfunc='mean'
    )
    
    sns.heatmap(pivot, ax=ax, cmap='viridis', cbar_kws={'label': 'mIoU'})
    ax.set_title('Response Surface: mIoU vs γ & Diversity', fontsize=14)
    ax.set_xlabel('γ Bins')
    ax.set_ylabel('Diversity Bins')

def plot_statistical_significance(results_df, ax):
    """Visual summary of statistical significance."""
    grouped = results_df.groupby('gamma')['mIoU']
    means = grouped.mean()
    sems = grouped.sem() # Standard error
    
    # Filter to display a subset of gammas to avoid crowding
    display_gammas = sorted(results_df['gamma'].unique())[::2] 
    
    plot_means = means.loc[display_gammas]
    plot_sems = sems.loc[display_gammas]
    
    ax.bar(range(len(display_gammas)), plot_means, yerr=plot_sems, capsize=5, alpha=0.7)
    ax.set_xticks(range(len(display_gammas)))
    ax.set_xticklabels([f'{g:.1f}' for g in display_gammas], rotation=45)
    ax.set_title('mIoU by γ (Mean ± SE)', fontsize=14)
    ax.set_xlabel('γ')
    ax.set_ylabel('mIoU')

def plot_optimal_gamma_by_frame(results_df, ax):
    """Scatter plot of optimal Gamma per frame."""
    optimal_gammas = []
    frame_indices = sorted(results_df['frame_idx'].unique())
    
    for frame_idx in frame_indices:
        subset = results_df[results_df['frame_idx'] == frame_idx]
        best_row = subset.loc[subset['mIoU'].idxmax()]
        optimal_gammas.append(best_row['gamma'])
        
    ax.scatter(frame_indices, optimal_gammas, c='purple', s=50, alpha=0.7)
    ax.axhline(y=np.mean(optimal_gammas), color='k', linestyle='--', label='Mean Optimal γ')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Optimal γ')
    ax.set_title('Optimal γ per Frame', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_kernel_spectrum(results_df, ax):
    """Placeholder for Kernel Spectrum Analysis."""
    ax.text(0.5, 0.5, "Kernel Spectrum Analysis\n(L matrix not available in results)", 
            ha='center', va='center', fontsize=12)
    ax.axis('off')

def plot_memory_utilization(results_df, ax):
    """Histogram of memory usage frequency."""
    all_selected = np.concatenate(results_df['selected_indices'].values)
    
    ax.hist(all_selected, bins=50, color='orange', alpha=0.7)
    ax.set_xlabel('Memory Index')
    ax.set_ylabel('Selection Count')
    ax.set_title('Memory Utilization Frequency', fontsize=14)
    ax.grid(True, alpha=0.3)

def plot_robustness_analysis(results_df, ax):
    """Plot robustness (std dev of mIoU) vs Gamma."""
    robustness = results_df.groupby('gamma')['mIoU'].std()
    
    ax.plot(robustness.index, robustness.values, 'r-o', linewidth=2)
    ax.set_xlabel('γ')
    ax.set_ylabel('mIoU Std Dev (Lower is Better)')
    ax.set_title('Robustness vs γ', fontsize=14)
    ax.invert_yaxis() # Lower std dev is better
    ax.grid(True, alpha=0.3)

def plot_recommendations_summary(results_df, ax):
    """Text summary of recommendations."""
    grouped = results_df.groupby('gamma').mean(numeric_only=True)
    optimal_gamma = grouped['mIoU'].idxmax()
    
    ax.axis('off')
    text = f"RECOMMENDATIONS\n\n"
    text += f"Optimal γ for mIoU: {optimal_gamma:.2f}\n"
    text += f"Peak mIoU: {grouped['mIoU'].max():.4f}\n\n"
    text += f"Suggested Range:\n"
    
    # Simple range logic
    threshold = 0.98 * grouped['mIoU'].max()
    good_gammas = grouped[grouped['mIoU'] >= threshold].index
    if len(good_gammas) > 0:
        text += f"[{good_gammas.min():.2f}, {good_gammas.max():.2f}]\n"
    else:
        text += f"Around {optimal_gamma:.2f}\n"
        
    ax.text(0.1, 0.5, text, fontsize=12, va='center', family='monospace')
    ax.set_title('Summary', fontsize=14)


# ================================
# 4. Statistical Analysis Functions
# ================================

def analyze_gamma_statistical_significance(results_df: pd.DataFrame):
    """Statistical analysis of γ effects."""
    
    print("=" * 70)
    print("STATISTICAL ANALYSIS OF γ PARAMETER")
    print("=" * 70)
    
    # 1. ANOVA across γ values
    from scipy import stats
    
    # Group mIoU by γ
    grouped_data = []
    gamma_groups = results_df.groupby('gamma')['mIoU']
    
    gamma_values = []
    for gamma_val, group in gamma_groups:
        gamma_values.append(gamma_val)
        grouped_data.append(group.values)
    
    # One-way ANOVA
    f_stat, p_value = stats.f_oneway(*grouped_data)
    
    print(f"\n1. One-way ANOVA for mIoU across γ values:")
    print(f"   F-statistic: {f_stat:.4f}")
    print(f"   p-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print("   → Significant difference found (p < 0.05)")
        
        # Post-hoc Tukey test
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            
            # Prepare data for Tukey
            tukey_data = pd.DataFrame({
                'mIoU': np.concatenate(grouped_data),
                'gamma': np.repeat([f'{g:.2f}' for g in gamma_values], 
                                  [len(g) for g in grouped_data])
            })
            
            tukey_result = pairwise_tukeyhsd(tukey_data['mIoU'], tukey_data['gamma'])
            print("\n2. Post-hoc Tukey HSD Test:")
            print(tukey_result)
            
            # Extract significant pairs
            sig_pairs = tukey_result.reject
            print(f"\n   Significant pairs: {np.sum(sig_pairs)}")
            
        except ImportError:
            print("   Install statsmodels for post-hoc analysis")
    else:
        print("   → No significant difference found")
    
    # 2. Correlation analysis
    print("\n3. Correlation Analysis:")
    
    corr_data = results_df.groupby('gamma').mean(numeric_only=True).reset_index()
    
    correlations = {}
    metrics = ['mIoU', 'avg_similarity_to_current', 'diversity']
    
    for metric in metrics:
        corr = np.corrcoef(corr_data['gamma'], corr_data[metric])[0, 1]
        correlations[metric] = corr
        
        direction = "positive" if corr > 0 else "negative"
        strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
        
        print(f"   {metric:30s}: r = {corr:6.3f} ({strength} {direction} correlation)")
    
    # 3. Optimal γ analysis
    print("\n4. Optimal γ Analysis:")
    
    optimal_by_mIoU = corr_data.loc[corr_data['mIoU'].idxmax()]
    print(f"   Optimal γ for mIoU: {optimal_by_mIoU['gamma']:.3f}")
    print(f"   mIoU at optimal: {optimal_by_mIoU['mIoU']:.4f}")
    print(f"   Similarity at optimal: {optimal_by_mIoU['avg_similarity_to_current']:.3f}")
    print(f"   Diversity at optimal: {optimal_by_mIoU['diversity']:.3f}")
    
    # 4. Sensitivity analysis
    print("\n5. Sensitivity Analysis:")
    
    # Compute derivatives numerically
    gamma_sorted = np.sort(corr_data['gamma'].unique())
    mIoU_sorted = corr_data.set_index('gamma').loc[gamma_sorted]['mIoU']
    
    # Compute gradient
    gradient = np.gradient(mIoU_sorted, gamma_sorted)
    
    # Find regions of high sensitivity
    high_sensitivity_idx = np.where(np.abs(gradient) > np.percentile(np.abs(gradient), 75))[0]
    if len(high_sensitivity_idx) > 0:
        sensitive_gammas = gamma_sorted[high_sensitivity_idx]
        print(f"   High sensitivity regions: γ ∈ [{sensitive_gammas.min():.2f}, {sensitive_gammas.max():.2f}]")
    
    # Find plateau regions
    plateau_idx = np.where(np.abs(gradient) < np.percentile(np.abs(gradient), 25))[0]
    if len(plateau_idx) > 0:
        plateau_gammas = gamma_sorted[plateau_idx]
        print(f"   Plateau regions: γ ∈ [{plateau_gammas.min():.2f}, {plateau_gammas.max():.2f}]")
    
    return {
        'anova': (f_stat, p_value),
        'correlations': correlations,
        'optimal_gamma': optimal_by_mIoU['gamma'],
        'gradient': gradient,
        'gamma_values': gamma_sorted
    }

# ================================
# 5. Practical Recommendations Generator
# ================================

def generate_gamma_recommendations(results_df: pd.DataFrame, 
                                   task_requirements: Dict[str, Any] = None):
    """Generate practical recommendations based on γ sensitivity analysis."""
    
    if task_requirements is None:
        task_requirements = {
            'prioritize_quality': False,  # Prioritize similarity to current
            'need_diversity': True,       # Need diverse memories
            'robustness_important': True, # Need consistent performance
            'operating_range': 'wide'     # 'wide' or 'narrow' γ range
        }
    
    # Analyze results
    grouped = results_df.groupby('gamma').mean(numeric_only=True)
    
    # Find optimal γ for different criteria
    optimal_for_mIoU = grouped['mIoU'].idxmax()
    optimal_for_diversity = grouped['diversity'].idxmax()
    optimal_for_similarity = grouped['avg_similarity_to_current'].idxmax()
    
    # Find robust γ (high performance across frames)
    robustness = results_df.groupby('gamma')['mIoU'].std()
    most_robust = robustness.idxmin()
    
    print("=" * 70)
    print("γ PARAMETER RECOMMENDATIONS FOR VIDEO SEGMENTATION")
    print("=" * 70)
    
    print("\n1. Optimal γ Values for Different Objectives:")
    print(f"   • Max mIoU:                 γ = {optimal_for_mIoU:.3f}")
    print(f"   • Max diversity:            γ = {optimal_for_diversity:.3f}")
    print(f"   • Max similarity to current: γ = {optimal_for_similarity:.3f}")
    print(f"   • Most robust:              γ = {most_robust:.3f}")
    
    print("\n2. Recommended γ Ranges:")
    
    # Find γ where mIoU > 95% of max
    max_mIoU = grouped['mIoU'].max()
    good_gammas = grouped[grouped['mIoU'] > 0.95 * max_mIoU].index
    if len(good_gammas) > 0:
        print(f"   • High performance range:  γ ∈ [{good_gammas.min():.2f}, {good_gammas.max():.2f}]")
    
    # Find γ where performance is stable (±5% of max)
    stable_gammas = grouped[(grouped['mIoU'] > 0.9 * max_mIoU) & 
                           (robustness < 0.1)].index
    if len(stable_gammas) > 0:
        print(f"   • Stable performance range: γ ∈ [{stable_gammas.min():.2f}, {stable_gammas.max():.2f}]")
    
    print("\n3. Task-Specific Recommendations:")
    
    if task_requirements['prioritize_quality']:
        print(f"   • For quality-focused tasks: γ = {optimal_for_similarity:.2f} ± 0.5")
        print("     (Selects memories most similar to current frame)")
    
    if task_requirements['need_diversity']:
        print(f"   • For diversity-focused tasks: γ = {optimal_for_diversity:.2f} ± 0.5")
        print("     (Ensures diverse memory selection)")
    
    if task_requirements['robustness_important']:
        print(f"   • For robust performance: γ = {most_robust:.2f} ± 0.3")
        print("     (Consistent performance across different frames)")
    
    print("\n4. Special Cases:")
    print("   • γ = 0: Pure diversity selection (ignores current frame similarity)")
    print("   • γ → ∞: Only selects most similar frame (becomes k-NN)")
    print("   • γ < 0: Prefers dissimilar frames (useful for anomaly detection)")
    
    print("\n5. Implementation Notes:")
    print("   • Start with γ = 0.5 as default")
    print("   • Tune in range [0, 2] for most video segmentation tasks")
    print("   • Use cross-validation per video sequence")
    print("   • Consider adaptive γ based on scene dynamics")
    
    return {
        'optimal_gammas': {
            'mIoU': optimal_for_mIoU,
            'diversity': optimal_for_diversity,
            'similarity': optimal_for_similarity,
            'robustness': most_robust
        },
        'recommended_ranges': {
            'high_performance': (good_gammas.min(), good_gammas.max()) if len(good_gammas) > 0 else None,
            'stable': (stable_gammas.min(), stable_gammas.max()) if len(stable_gammas) > 0 else None
        }
    }

# ================================
# 6. Main Execution with Example Data
# ================================

def main_experiment():
    """Run complete γ sensitivity analysis with example data."""
    
    print("Running γ Parameter Sensitivity Analysis for DPP Memory Selection")
    print("=" * 80)
    
    # Generate example data (replace with your actual features)
    np.random.seed(42)
    
    n_memories = 500
    feature_dim = 128
    n_current_frames = 20
    
    # 1. Generate memory features with some structure
    memory_features = np.random.randn(n_memories, feature_dim)
    
    # Add cluster structure (memories from different scenes)
    n_clusters = 5
    cluster_centers = np.random.randn(n_clusters, feature_dim) * 2
    cluster_size = n_memories // n_clusters
    
    for i in range(n_clusters):
        start = i * cluster_size
        end = start + cluster_size
        memory_features[start:end] += cluster_centers[i] * 0.5
    
    # Normalize
    memory_features = memory_features / np.linalg.norm(memory_features, axis=1, keepdims=True)
    
    # 2. Generate current frame features (some from each cluster)
    current_features = []
    for i in range(n_current_frames):
        # Sample from different clusters
        cluster_id = i % n_clusters
        current_feature = cluster_centers[cluster_id] + np.random.randn(feature_dim) * 0.3
        current_feature = current_feature / np.linalg.norm(current_feature)
        current_features.append(current_feature)
    
    current_features = np.array(current_features)
    
    print(f"Generated {n_memories} memory features with {n_clusters} clusters")
    print(f"Testing on {n_current_frames} current frames")
    
    # 3. Define γ range to test
    gamma_values = np.concatenate([
        np.linspace(-3, -1, 5),      # Negative: prefer dissimilar
        np.linspace(-0.8, 0.8, 9),   # Near zero
        np.linspace(1, 3, 5),        # Positive
        np.linspace(4, 10, 4),       # Strong positive
        [15, 20]                     # Very strong
    ])
    
    print(f"\nTesting {len(gamma_values)} γ values from {gamma_values[0]:.1f} to {gamma_values[-1]:.1f}")
    
    # 4. Run sensitivity experiment
    results_df = run_gamma_sensitivity_experiment(
        memory_features=memory_features,
        current_features=current_features,
        k=10,  # Select 10 memories
        gamma_values=gamma_values,
        n_trials=3,
        segmentation_model=None  # Using simulated performance
    )
    
    print(f"\nExperiment completed. Collected {len(results_df)} data points.")
    
    # 5. Generate comprehensive visualizations
    print("\nGenerating visualizations...")
    fig = plot_gamma_sensitivity_comprehensive(results_df)
    plt.savefig('gamma_sensitivity_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("✓ Saved comprehensive visualization")
    
    # 6. Statistical analysis
    print("\n" + "="*80)
    stats_results = analyze_gamma_statistical_significance(results_df)
    
    # 7. Generate recommendations
    print("\n" + "="*80)
    recommendations = generate_gamma_recommendations(results_df)
    
    # 8. Save results for further analysis
    results_df.to_csv('gamma_sensitivity_results.csv', index=False)
    print("\n✓ Results saved to CSV for further analysis")
    
    # 9. Create summary report
    create_summary_report(results_df, stats_results, recommendations)
    
    plt.show()
    
    return results_df, stats_results, recommendations

def create_summary_report(results_df, stats_results, recommendations):
    """Create a summary report file."""
    
    with open('gamma_sensitivity_summary.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("γ PARAMETER SENSITIVITY ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("EXPERIMENT SETUP\n")
        f.write("-"*40 + "\n")
        f.write(f"Number of memory items: {len(results_df['selected_indices'].iloc[0]) if len(results_df) > 0 else 'N/A'}\n")
        f.write(f"γ range tested: [{results_df['gamma'].min():.2f}, {results_df['gamma'].max():.2f}]\n")
        f.write(f"Number of trials: {results_df['trial'].nunique()}\n")
        f.write(f"Number of test frames: {results_df['frame_idx'].nunique()}\n\n")
        
        f.write("KEY FINDINGS\n")
        f.write("-"*40 + "\n")
        
        # Optimal γ
        optimal_gamma = recommendations['optimal_gammas']['mIoU']
        f.write(f"1. Optimal γ for segmentation: {optimal_gamma:.3f}\n")
        
        # Performance at optimal
        optimal_perf = results_df[results_df['gamma'] == optimal_gamma]['mIoU'].mean()
        f.write(f"   • mIoU at optimal γ: {optimal_perf:.4f}\n")
        
        # Comparison to extremes
        gamma_0_perf = results_df[results_df['gamma'] == 0]['mIoU'].mean() if 0 in results_df['gamma'].values else None
        if gamma_0_perf:
            improvement = (optimal_perf - gamma_0_perf) / gamma_0_perf * 100
            f.write(f"   • Improvement over γ=0: {improvement:.1f}%\n")
        
        # Sensitivity analysis
        f.write(f"\n2. Sensitivity Analysis:\n")
        f_stat, p_value = stats_results['anova']
        f.write(f"   • ANOVA p-value: {p_value:.6f}\n")
        f.write(f"   • Significant differences: {'YES' if p_value < 0.05 else 'NO'}\n")
        
        # Recommended ranges
        f.write(f"\n3. Recommended γ Ranges:\n")
        if recommendations['recommended_ranges']['high_performance']:
            low, high = recommendations['recommended_ranges']['high_performance']
            f.write(f"   • High performance: [{low:.2f}, {high:.2f}]\n")
        
        f.write("\n4. Practical Recommendations:\n")
        f.write("   • Default starting value: γ = 0.5\n")
        f.write("   • Tuning range: [0, 2] for most tasks\n")
        f.write("   • Use adaptive γ for dynamic scenes\n")
        f.write("   • Validate per sequence for best results\n")
    
    print("✓ Summary report saved to gamma_sensitivity_summary.txt")

if __name__ == "__main__":
    # Run the complete analysis
    results, stats, recommendations = main_experiment()
