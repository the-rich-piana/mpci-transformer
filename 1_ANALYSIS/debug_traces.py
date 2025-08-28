#!/usr/bin/env python3
"""
Debug script to compare calcium traces between notebook and preprocessor approaches
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Add utils directory to path for imports
utils_dir = os.path.join(parent_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from utils.activity_preprocessor import CalciumDataPreprocessor
from utils.load_meso_session import MesoscopeSession
from one.api import ONE

def debug_trace_differences():
    """Compare traces step by step between notebook and preprocessor"""
    print("="*80)
    print("DEBUGGING CALCIUM TRACE DIFFERENCES")
    print("="*80)
    
    # Load the same session as the notebook
    one = ONE()
    eid = '5ea6bb9b-6163-4e8a-816b-efe7002666b0'
    
    print(f"\n1. Loading session {eid}...")
    session = MesoscopeSession.from_eid(one, eid, raw_activity=False)
    
    print(f"   Session loaded: {session.n_total_neurons} total neurons")
    print(f"   FOVs: {list(session.fovs.keys())}")
    
    # Step 1: Get raw ΔF/F data (exactly like notebook)
    print(f"\n2. Getting raw ΔF/F data (like notebook)...")
    
    # Method 1: Notebook approach
    dff_matrix_notebook, timestamps_notebook = session.get_dff_activity_matrix(100, neucoeff=0.7)
    print(f"   Notebook ΔF/F shape: {dff_matrix_notebook.shape}")
    print(f"   Notebook ΔF/F range: {dff_matrix_notebook.min():.4f} to {dff_matrix_notebook.max():.4f}")
    
    # Method 2: Preprocessor approach (first step only)
    dff_matrix_preprocessor, timestamps_preprocessor = session.get_dff_activity_matrix(neucoeff=0.7)
    print(f"   Preprocessor ΔF/F shape: {dff_matrix_preprocessor.shape}")
    print(f"   Preprocessor ΔF/F range: {dff_matrix_preprocessor.min():.4f} to {dff_matrix_preprocessor.max():.4f}")
    
    # Compare raw data
    print(f"\n3. Comparing raw ΔF/F data...")
    
    # Check if they're identical for first 100 timepoints
    common_length = min(100, dff_matrix_notebook.shape[0], dff_matrix_preprocessor.shape[0])
    notebook_subset = dff_matrix_notebook[:common_length, :]
    preprocessor_subset = dff_matrix_preprocessor[:common_length, :]
    
    if notebook_subset.shape == preprocessor_subset.shape:
        diff = np.abs(notebook_subset - preprocessor_subset)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"   Max difference in raw ΔF/F: {max_diff:.6f}")
        print(f"   Mean difference in raw ΔF/F: {mean_diff:.6f}")
        
        if max_diff < 1e-10:
            print("   ✓ Raw ΔF/F data is identical!")
        else:
            print("   ✗ Raw ΔF/F data differs!")
            
    else:
        print(f"   ✗ Shape mismatch: notebook {notebook_subset.shape} vs preprocessor {preprocessor_subset.shape}")
    
    # Step 2: Apply preprocessor filtering and see where it diverges
    print(f"\n4. Testing preprocessor steps...")
    
    preprocessor = CalciumDataPreprocessor(neucoeff=0.7, temporal_smoothing=True)
    
    # Get the first few steps manually
    print(f"   4.1. Initial ΔF/F loading...")
    raw_dff, raw_timestamps = session.get_dff_activity_matrix(neucoeff=0.7)
    print(f"        Shape: {raw_dff.shape}, Range: {raw_dff.min():.4f} to {raw_dff.max():.4f}")
    
    # Check truncation effect
    print(f"   4.2. Testing truncation...")
    last_stimoff = session.stimOff_times[-1]
    cutoff_idx = np.searchsorted(raw_timestamps, last_stimoff, side='right')
    truncated_dff = raw_dff[:cutoff_idx]
    truncated_timestamps = raw_timestamps[:cutoff_idx]
    print(f"        Truncated at index {cutoff_idx}")
    print(f"        Truncated shape: {truncated_dff.shape}")
    print(f"        Truncated range: {truncated_dff.min():.4f} to {truncated_dff.max():.4f}")
    
    # Check filtering effect
    print(f"   4.3. Testing neuron filtering...")
    filtered_data, neuron_mask = preprocessor._filter_neurons(truncated_dff)
    print(f"        Filtered shape: {filtered_data.shape}")
    print(f"        Filtered range: {filtered_data.min():.4f} to {filtered_data.max():.4f}")
    print(f"        Neurons kept: {neuron_mask.sum()}/{len(neuron_mask)} ({neuron_mask.sum()/len(neuron_mask)*100:.1f}%)")
    
    # Check top neuron selection
    print(f"   4.4. Testing top neuron selection...")
    top_n_neurons = 5000
    if top_n_neurons < filtered_data.shape[1]:
        neuron_activity_metric = np.std(filtered_data, axis=0)
        top_n_relative_indices = np.argsort(neuron_activity_metric)[-top_n_neurons:]
        final_data = filtered_data[:, top_n_relative_indices]
        print(f"        Selected top {top_n_neurons} neurons")
        print(f"        Final shape: {final_data.shape}")
        print(f"        Final range: {final_data.min():.4f} to {final_data.max():.4f}")
    else:
        final_data = filtered_data
        print(f"        Keeping all filtered neurons")
    
    # Check normalization effect
    print(f"   4.5. Testing normalization...")
    normalized_data, norm_params = preprocessor._robust_normalize(final_data)
    print(f"        Normalized shape: {normalized_data.shape}")
    print(f"        Normalized range: {normalized_data.min():.4f} to {normalized_data.max():.4f}")
    
    # Check temporal smoothing effect
    if preprocessor.temporal_smoothing:
        print(f"   4.6. Testing temporal smoothing...")
        smoothed_data = preprocessor._temporal_smooth(normalized_data)
        print(f"        Smoothed shape: {smoothed_data.shape}")
        print(f"        Smoothed range: {smoothed_data.min():.4f} to {smoothed_data.max():.4f}")
        final_processed = smoothed_data
    else:
        final_processed = normalized_data
    
    # Step 3: Compare individual neurons visually
    print(f"\n5. Visual comparison of individual neurons...")
    
    # Get same neurons for comparison
    # Take top 5 most active neurons from notebook data
    neuron_activity_metric_nb = np.std(dff_matrix_notebook, axis=0)
    top_5_indices_nb = np.argsort(neuron_activity_metric_nb)[-5:]
    
    # Create plots
    fig, axes = plt.subplots(5, 3, figsize=(15, 15))
    
    for i, neuron_idx in enumerate(top_5_indices_nb):
        # Notebook data (first 100 timepoints)
        axes[i, 0].plot(dff_matrix_notebook[:100, neuron_idx])
        axes[i, 0].set_title(f'Neuron {neuron_idx} - Notebook Raw ΔF/F')
        axes[i, 0].set_ylabel('ΔF/F')
        
        # Preprocessor raw data (same neuron if it exists)
        if neuron_idx < dff_matrix_preprocessor.shape[1]:
            axes[i, 1].plot(dff_matrix_preprocessor[:100, neuron_idx])
            axes[i, 1].set_title(f'Neuron {neuron_idx} - Preprocessor Raw ΔF/F')
            
        # Try to find corresponding neuron in processed data (if it survived filtering)
        # This is tricky since filtering changes indices
        if i < final_processed.shape[1]:
            axes[i, 2].plot(final_processed[:100, i])
            axes[i, 2].set_title(f'Top {i+1} - Processed')
            
        if i == 4:  # Bottom row
            axes[i, 0].set_xlabel('Time (samples)')
            axes[i, 1].set_xlabel('Time (samples)')
            axes[i, 2].set_xlabel('Time (samples)')
    
    plt.tight_layout()
    plt.savefig('trace_comparison.png', dpi=150, bbox_inches='tight')
    print(f"   Saved comparison plot to trace_comparison.png")
    
    # Step 4: Summary
    print(f"\n6. SUMMARY:")
    print(f"   Raw ΔF/F identical: {max_diff < 1e-10 if 'max_diff' in locals() else 'Unknown'}")
    print(f"   Truncation removes {raw_dff.shape[0] - truncated_dff.shape[0]} timepoints")
    print(f"   Filtering removes {len(neuron_mask) - neuron_mask.sum()} neurons")
    print(f"   Final range: {final_processed.min():.4f} to {final_processed.max():.4f}")
    
    return {
        'notebook_data': dff_matrix_notebook,
        'preprocessor_raw': dff_matrix_preprocessor,
        'preprocessor_final': final_processed,
        'timestamps': timestamps_notebook,
        'neuron_mask': neuron_mask if 'neuron_mask' in locals() else None
    }

if __name__ == "__main__":
    results = debug_trace_differences()