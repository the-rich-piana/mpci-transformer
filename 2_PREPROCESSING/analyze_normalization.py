import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

def analyze_activity_matrices(activity_matrix, dff_activity_matrix, timestamps, timestamps_2):
    """
    Comprehensive analysis of raw activity vs ΔF/F corrected activity
    to determine optimal normalization for machine learning
    """
    
    print("="*80)
    print("NEURAL ACTIVITY ANALYSIS FOR ML NORMALIZATION")
    print("="*80)
    
    # Basic shape info
    print(f"\nBASIC STATISTICS:")
    print(f"Raw activity shape: {activity_matrix.shape}")
    print(f"ΔF/F activity shape: {dff_activity_matrix.shape}")
    print(f"Timestamps match: {np.array_equal(timestamps, timestamps_2)}")
    
    # Summary statistics
    print(f"\nRAW ACTIVITY STATISTICS:")
    print(f"  Mean: {np.mean(activity_matrix):.4f}")
    print(f"  Std: {np.std(activity_matrix):.4f}")
    print(f"  Min: {np.min(activity_matrix):.4f}")
    print(f"  Max: {np.max(activity_matrix):.4f}")
    print(f"  Median: {np.median(activity_matrix):.4f}")
    print(f"  25th percentile: {np.percentile(activity_matrix, 25):.4f}")
    print(f"  75th percentile: {np.percentile(activity_matrix, 75):.4f}")
    print(f"  95th percentile: {np.percentile(activity_matrix, 95):.4f}")
    print(f"  99th percentile: {np.percentile(activity_matrix, 99):.4f}")
    
    print(f"\nΔF/F ACTIVITY STATISTICS:")
    print(f"  Mean: {np.mean(dff_activity_matrix):.4f}")
    print(f"  Std: {np.std(dff_activity_matrix):.4f}")
    print(f"  Min: {np.min(dff_activity_matrix):.4f}")
    print(f"  Max: {np.max(dff_activity_matrix):.4f}")
    print(f"  Median: {np.median(dff_activity_matrix):.4f}")
    print(f"  25th percentile: {np.percentile(dff_activity_matrix, 25):.4f}")
    print(f"  75th percentile: {np.percentile(dff_activity_matrix, 75):.4f}")
    print(f"  95th percentile: {np.percentile(dff_activity_matrix, 95):.4f}")
    print(f"  99th percentile: {np.percentile(dff_activity_matrix, 99):.4f}")
    
    # Sparsity analysis
    raw_sparse = np.mean(activity_matrix == 0)
    dff_sparse = np.mean(dff_activity_matrix == 0)
    raw_near_zero = np.mean(np.abs(activity_matrix) < 0.01)
    dff_near_zero = np.mean(np.abs(dff_activity_matrix) < 0.01)
    
    print(f"\nSPARSITY ANALYSIS:")
    print(f"  Raw activity - exact zeros: {raw_sparse:.3f}")
    print(f"  Raw activity - near zeros (<0.01): {raw_near_zero:.3f}")
    print(f"  ΔF/F activity - exact zeros: {dff_sparse:.3f}")
    print(f"  ΔF/F activity - near zeros (<0.01): {dff_near_zero:.3f}")
    
    # Distribution analysis
    print(f"\nDISTRIBUTION ANALYSIS:")
    
    # Test for normality (on a sample due to computational constraints)
    sample_size = min(10000, activity_matrix.size)
    raw_sample = np.random.choice(activity_matrix.flatten(), sample_size, replace=False)
    dff_sample = np.random.choice(dff_activity_matrix.flatten(), sample_size, replace=False)
    
    _, raw_p_norm = stats.normaltest(raw_sample)
    _, dff_p_norm = stats.normaltest(dff_sample)
    
    print(f"  Raw activity normality test p-value: {raw_p_norm:.2e}")
    print(f"  ΔF/F activity normality test p-value: {dff_p_norm:.2e}")
    print(f"  Raw activity is normal: {raw_p_norm > 0.05}")
    print(f"  ΔF/F activity is normal: {dff_p_norm > 0.05}")
    
    # Skewness and kurtosis
    raw_skew = stats.skew(raw_sample)
    dff_skew = stats.skew(dff_sample)
    raw_kurt = stats.kurtosis(raw_sample)
    dff_kurt = stats.kurtosis(dff_sample)
    
    print(f"  Raw activity skewness: {raw_skew:.4f}")
    print(f"  ΔF/F activity skewness: {dff_skew:.4f}")
    print(f"  Raw activity kurtosis: {raw_kurt:.4f}")
    print(f"  ΔF/F activity kurtosis: {dff_kurt:.4f}")
    
    # Per-neuron statistics
    print(f"\nPER-NEURON ANALYSIS:")
    raw_neuron_means = np.mean(activity_matrix, axis=0)
    dff_neuron_means = np.mean(dff_activity_matrix, axis=0)
    raw_neuron_stds = np.std(activity_matrix, axis=0)
    dff_neuron_stds = np.std(dff_activity_matrix, axis=0)
    
    print(f"  Raw - Mean of neuron means: {np.mean(raw_neuron_means):.4f}")
    print(f"  Raw - Std of neuron means: {np.std(raw_neuron_means):.4f}")
    print(f"  Raw - Mean of neuron stds: {np.mean(raw_neuron_stds):.4f}")
    print(f"  Raw - Std of neuron stds: {np.std(raw_neuron_stds):.4f}")
    
    print(f"  ΔF/F - Mean of neuron means: {np.mean(dff_neuron_means):.4f}")
    print(f"  ΔF/F - Std of neuron means: {np.std(dff_neuron_means):.4f}")
    print(f"  ΔF/F - Mean of neuron stds: {np.mean(dff_neuron_stds):.4f}")
    print(f"  ΔF/F - Std of neuron stds: {np.std(dff_neuron_stds):.4f}")
    
    # Correlation analysis
    print(f"\nCORRELATION ANALYSIS:")
    # Sample neurons for correlation analysis
    n_sample_neurons = min(100, activity_matrix.shape[1])
    sample_indices = np.random.choice(activity_matrix.shape[1], n_sample_neurons, replace=False)
    
    raw_corr = np.corrcoef(activity_matrix[:, sample_indices].T)
    dff_corr = np.corrcoef(dff_activity_matrix[:, sample_indices].T)
    
    # Remove diagonal and get upper triangle
    raw_corr_vals = raw_corr[np.triu_indices_from(raw_corr, k=1)]
    dff_corr_vals = dff_corr[np.triu_indices_from(dff_corr, k=1)]
    
    print(f"  Raw activity mean correlation: {np.mean(raw_corr_vals):.4f}")
    print(f"  ΔF/F activity mean correlation: {np.mean(dff_corr_vals):.4f}")
    print(f"  Raw activity correlation std: {np.std(raw_corr_vals):.4f}")
    print(f"  ΔF/F activity correlation std: {np.std(dff_corr_vals):.4f}")
    
    return {
        'raw_stats': {
            'mean': np.mean(activity_matrix),
            'std': np.std(activity_matrix),
            'min': np.min(activity_matrix),
            'max': np.max(activity_matrix),
            'median': np.median(activity_matrix),
            'p95': np.percentile(activity_matrix, 95),
            'p99': np.percentile(activity_matrix, 99),
            'skew': raw_skew,
            'kurtosis': raw_kurt,
            'sparsity': raw_sparse
        },
        'dff_stats': {
            'mean': np.mean(dff_activity_matrix),
            'std': np.std(dff_activity_matrix),
            'min': np.min(dff_activity_matrix),
            'max': np.max(dff_activity_matrix),
            'median': np.median(dff_activity_matrix),
            'p95': np.percentile(dff_activity_matrix, 95),
            'p99': np.percentile(dff_activity_matrix, 99),
            'skew': dff_skew,
            'kurtosis': dff_kurt,
            'sparsity': dff_sparse
        },
        'raw_neuron_means': raw_neuron_means,
        'dff_neuron_means': dff_neuron_means,
        'raw_neuron_stds': raw_neuron_stds,
        'dff_neuron_stds': dff_neuron_stds
    }

def plot_normalization_comparison(activity_matrix, dff_activity_matrix, stats_dict):
    """
    Create visualizations comparing different normalization approaches
    """
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Sample data for plotting (to avoid memory issues)
    sample_size = min(50000, activity_matrix.size)
    raw_sample = np.random.choice(activity_matrix.flatten(), sample_size, replace=False)
    dff_sample = np.random.choice(dff_activity_matrix.flatten(), sample_size, replace=False)
    
    # Row 1: Raw distributions
    axes[0, 0].hist(raw_sample, bins=100, alpha=0.7, density=True, color='blue')
    axes[0, 0].set_title('Raw Activity Distribution')
    axes[0, 0].set_xlabel('Activity Value')
    axes[0, 0].set_ylabel('Density')
    
    axes[0, 1].hist(dff_sample, bins=100, alpha=0.7, density=True, color='red')
    axes[0, 1].set_title('ΔF/F Activity Distribution')
    axes[0, 1].set_xlabel('Activity Value')
    axes[0, 1].set_ylabel('Density')
    
    # Log-scale histograms
    axes[0, 2].hist(raw_sample, bins=100, alpha=0.7, density=True, color='blue')
    axes[0, 2].set_yscale('log')
    axes[0, 2].set_title('Raw Activity (Log Scale)')
    axes[0, 2].set_xlabel('Activity Value')
    axes[0, 2].set_ylabel('Log Density')
    
    axes[0, 3].hist(dff_sample, bins=100, alpha=0.7, density=True, color='red')
    axes[0, 3].set_yscale('log')
    axes[0, 3].set_title('ΔF/F Activity (Log Scale)')
    axes[0, 3].set_xlabel('Activity Value')
    axes[0, 3].set_ylabel('Log Density')
    
    # Row 2: Normalization comparisons
    # Min-max normalization
    raw_minmax = (raw_sample - np.min(raw_sample)) / (np.max(raw_sample) - np.min(raw_sample))
    dff_minmax = (dff_sample - np.min(dff_sample)) / (np.max(dff_sample) - np.min(dff_sample))
    
    axes[1, 0].hist(raw_minmax, bins=50, alpha=0.7, density=True, color='blue', label='Raw')
    axes[1, 0].hist(dff_minmax, bins=50, alpha=0.7, density=True, color='red', label='ΔF/F')
    axes[1, 0].set_title('Min-Max Normalization [0,1]')
    axes[1, 0].set_xlabel('Normalized Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    
    # Z-score normalization
    raw_zscore = (raw_sample - np.mean(raw_sample)) / np.std(raw_sample)
    dff_zscore = (dff_sample - np.mean(dff_sample)) / np.std(dff_sample)
    
    axes[1, 1].hist(raw_zscore, bins=50, alpha=0.7, density=True, color='blue', label='Raw')
    axes[1, 1].hist(dff_zscore, bins=50, alpha=0.7, density=True, color='red', label='ΔF/F')
    axes[1, 1].set_title('Z-Score Normalization')
    axes[1, 1].set_xlabel('Z-Score')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    
    # Percentile normalization (0-99.5th percentile)
    raw_p99 = np.percentile(raw_sample, 99.5)
    dff_p99 = np.percentile(dff_sample, 99.5)
    raw_perc = np.clip(raw_sample / raw_p99, 0, 1)
    dff_perc = np.clip(dff_sample / dff_p99, 0, 1)
    
    axes[1, 2].hist(raw_perc, bins=50, alpha=0.7, density=True, color='blue', label='Raw')
    axes[1, 2].hist(dff_perc, bins=50, alpha=0.7, density=True, color='red', label='ΔF/F')
    axes[1, 2].set_title('Percentile Normalization (0-99.5%)')
    axes[1, 2].set_xlabel('Normalized Value')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].legend()
    
    # Robust scaling (median and IQR)
    raw_median = np.median(raw_sample)
    raw_iqr = np.percentile(raw_sample, 75) - np.percentile(raw_sample, 25)
    dff_median = np.median(dff_sample)
    dff_iqr = np.percentile(dff_sample, 75) - np.percentile(dff_sample, 25)
    
    raw_robust = (raw_sample - raw_median) / raw_iqr
    dff_robust = (dff_sample - dff_median) / dff_iqr
    
    axes[1, 3].hist(raw_robust, bins=50, alpha=0.7, density=True, color='blue', label='Raw')
    axes[1, 3].hist(dff_robust, bins=50, alpha=0.7, density=True, color='red', label='ΔF/F')
    axes[1, 3].set_title('Robust Scaling (Median/IQR)')
    axes[1, 3].set_xlabel('Scaled Value')
    axes[1, 3].set_ylabel('Density')
    axes[1, 3].legend()
    
    # Row 3: Per-neuron statistics
    axes[2, 0].hist(stats_dict['raw_neuron_means'], bins=50, alpha=0.7, color='blue', label='Raw')
    axes[2, 0].hist(stats_dict['dff_neuron_means'], bins=50, alpha=0.7, color='red', label='ΔF/F')
    axes[2, 0].set_title('Per-Neuron Mean Distribution')
    axes[2, 0].set_xlabel('Mean Activity')
    axes[2, 0].set_ylabel('Count')
    axes[2, 0].legend()
    
    axes[2, 1].hist(stats_dict['raw_neuron_stds'], bins=50, alpha=0.7, color='blue', label='Raw')
    axes[2, 1].hist(stats_dict['dff_neuron_stds'], bins=50, alpha=0.7, color='red', label='ΔF/F')
    axes[2, 1].set_title('Per-Neuron Std Distribution')
    axes[2, 1].set_xlabel('Std Activity')
    axes[2, 1].set_ylabel('Count')
    axes[2, 1].legend()
    
    # CV (coefficient of variation) analysis
    raw_cv = stats_dict['raw_neuron_stds'] / (stats_dict['raw_neuron_means'] + 1e-8)
    dff_cv = stats_dict['dff_neuron_stds'] / (np.abs(stats_dict['dff_neuron_means']) + 1e-8)
    
    axes[2, 2].hist(raw_cv, bins=50, alpha=0.7, color='blue', label='Raw')
    axes[2, 2].hist(dff_cv, bins=50, alpha=0.7, color='red', label='ΔF/F')
    axes[2, 2].set_title('Coefficient of Variation')
    axes[2, 2].set_xlabel('CV (Std/Mean)')
    axes[2, 2].set_ylabel('Count')
    axes[2, 2].legend()
    
    # Q-Q plot comparison
    from scipy import stats
    stats.probplot(raw_sample, dist="norm", plot=axes[2, 3])
    axes[2, 3].set_title('Q-Q Plot: Raw Activity vs Normal')
    axes[2, 3].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def recommend_normalization(stats_dict):
    """
    Provide recommendations based on the statistical analysis
    """
    
    print("\\n" + "="*80)
    print("NORMALIZATION RECOMMENDATIONS FOR MACHINE LEARNING")
    print("="*80)
    
    raw_stats = stats_dict['raw_stats']
    dff_stats = stats_dict['dff_stats']
    
    print("\\nANALYSIS SUMMARY:")
    print(f"1. ΔF/F correction significantly changes the data distribution")
    print(f"2. Raw data range: [{raw_stats['min']:.4f}, {raw_stats['max']:.4f}]")
    print(f"3. ΔF/F data range: [{dff_stats['min']:.4f}, {dff_stats['max']:.4f}]")
    print(f"4. Raw data is {'highly' if abs(raw_stats['skew']) > 2 else 'moderately' if abs(raw_stats['skew']) > 0.5 else 'not'} skewed")
    print(f"5. ΔF/F data is {'highly' if abs(dff_stats['skew']) > 2 else 'moderately' if abs(dff_stats['skew']) > 0.5 else 'not'} skewed")
    
    print("\\nRECOMMENDATIONS:")
    
    # Recommendation 1: Data choice
    if abs(dff_stats['skew']) < abs(raw_stats['skew']) and abs(dff_stats['kurtosis']) < abs(raw_stats['kurtosis']):
        print("✓ RECOMMENDATION 1: Use ΔF/F corrected data (better distribution properties)")
    else:
        print("? RECOMMENDATION 1: Consider both raw and ΔF/F data (similar distribution properties)")
    
    # Recommendation 2: Normalization method
    if abs(dff_stats['skew']) < 1 and abs(dff_stats['kurtosis']) < 3:
        print("✓ RECOMMENDATION 2: Z-score normalization (data is reasonably normal)")
    elif dff_stats['min'] >= 0:
        print("✓ RECOMMENDATION 2: Percentile normalization (0-99.5th percentile) for positive data")
    else:
        print("✓ RECOMMENDATION 2: Robust scaling (median/IQR) for skewed data with outliers")
    
    # Recommendation 3: Per-neuron vs global normalization
    neuron_mean_var = np.var(stats_dict['dff_neuron_means'])
    neuron_std_var = np.var(stats_dict['dff_neuron_stds'])
    
    if neuron_mean_var > 0.1 or neuron_std_var > 0.1:
        print("✓ RECOMMENDATION 3: Per-neuron normalization (neurons have different scales)")
    else:
        print("✓ RECOMMENDATION 3: Global normalization acceptable (similar neuron scales)")
    
    print("\\nSUGGESTED PREPROCESSING PIPELINE:")
    print("1. Load session with ΔF/F correction: session.get_dff_activity_matrix(neucoeff=0.7)")
    print("2. Apply per-neuron percentile normalization (0-99.5th percentile)")
    print("3. Clip outliers: np.clip(normalized_data, 0, 1)")
    print("4. Optional: Apply mild smoothing if needed for temporal consistency")
    
    return None

# Usage example (you would call this in your notebook):
# stats_dict = analyze_activity_matrices(activity_matrix, dff_activity_matrix, timestamps, timestamps_2)
# fig = plot_normalization_comparison(activity_matrix, dff_activity_matrix, stats_dict)
# recommend_normalization(stats_dict)