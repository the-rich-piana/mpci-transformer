#!/usr/bin/env python3
"""
Debug the filtering criteria to see why all neurons are being removed
"""
import numpy as np
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Add utils directory to path for imports
utils_dir = os.path.join(parent_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from utils.load_meso_session import MesoscopeSession
from one.api import ONE

def debug_filtering():
    """Debug why filtering removes all neurons"""
    print("="*80)
    print("DEBUGGING NEURON FILTERING")
    print("="*80)
    
    # Load session
    one = ONE()
    eid = '5ea6bb9b-6163-4e8a-816b-efe7002666b0'
    session = MesoscopeSession.from_eid(one, eid, raw_activity=False)
    
    # Get data
    dff_matrix, timestamps = session.get_dff_activity_matrix(neucoeff=0.7)
    
    # Truncate like preprocessor
    last_stimoff = session.stimOff_times[-1]
    cutoff_idx = np.searchsorted(timestamps, last_stimoff, side='right')
    dff_matrix = dff_matrix[:cutoff_idx]
    
    print(f"Data shape: {dff_matrix.shape}")
    print(f"Data range: {dff_matrix.min():.4f} to {dff_matrix.max():.4f}")
    
    # Calculate per-neuron statistics
    neuron_medians = np.median(dff_matrix, axis=0)
    neuron_q99 = np.percentile(dff_matrix, 99, axis=0)
    neuron_q01 = np.percentile(dff_matrix, 1, axis=0)
    neuron_iqr = np.percentile(dff_matrix, 75, axis=0) - np.percentile(dff_matrix, 25, axis=0)
    neuron_stds = np.std(dff_matrix, axis=0)
    negative_fraction = np.mean(dff_matrix < 0, axis=0)
    
    print(f"\nNeuron statistics:")
    print(f"  Medians: {neuron_medians.min():.4f} to {neuron_medians.max():.4f}")
    print(f"  IQR: {neuron_iqr.min():.4f} to {neuron_iqr.max():.4f}")
    print(f"  Stds: {neuron_stds.min():.4f} to {neuron_stds.max():.4f}")
    print(f"  Negative fraction: {negative_fraction.min():.4f} to {negative_fraction.max():.4f}")
    
    # Test each filtering criterion
    print(f"\nTesting filtering criteria:")
    
    # 1. Variability filter
    min_iqr_threshold = np.percentile(neuron_iqr[neuron_iqr > 0], 5)
    iqr_pass = neuron_iqr > min_iqr_threshold
    print(f"  1. IQR > {min_iqr_threshold:.4f}: {iqr_pass.sum()}/{len(iqr_pass)} pass")
    
    # 2. Range filter
    max_range_threshold = np.percentile(neuron_q99 - neuron_q01, 95)
    range_pass = (neuron_q99 - neuron_q01) < max_range_threshold
    print(f"  2. Range < {max_range_threshold:.4f}: {range_pass.sum()}/{len(range_pass)} pass")
    
    # 3. Median filter
    min_median_threshold = np.percentile(neuron_medians, 1)
    median_pass = neuron_medians > min_median_threshold
    print(f"  3. Median > {min_median_threshold:.4f}: {median_pass.sum()}/{len(median_pass)} pass")
    
    # 4. Std filter
    max_std_threshold = np.percentile(neuron_stds, 99)
    std_pass = neuron_stds < max_std_threshold
    print(f"  4. Std < {max_std_threshold:.4f}: {std_pass.sum()}/{len(std_pass)} pass")
    
    # 5. Negative fraction filter
    max_negative_fraction = 0.5
    neg_pass = negative_fraction < max_negative_fraction
    print(f"  5. Neg fraction < {max_negative_fraction}: {neg_pass.sum()}/{len(neg_pass)} pass")
    
    # Combined filter
    combined_pass = iqr_pass & range_pass & median_pass & std_pass & neg_pass
    print(f"\nCombined filter: {combined_pass.sum()}/{len(combined_pass)} pass")
    
    # Show which filter is most restrictive
    print(f"\nMost restrictive filters:")
    filters = [
        ("IQR", iqr_pass),
        ("Range", range_pass), 
        ("Median", median_pass),
        ("Std", std_pass),
        ("Negative", neg_pass)
    ]
    
    for name, filt in sorted(filters, key=lambda x: x[1].sum()):
        print(f"  {name}: {filt.sum()}/{len(filt)} ({filt.sum()/len(filt)*100:.1f}%)")
    
    # Try less restrictive criteria
    print(f"\nTrying less restrictive criteria:")
    
    # Much more lenient filtering
    min_iqr_lenient = np.percentile(neuron_iqr[neuron_iqr > 0], 1)  # 1st percentile instead of 5th
    max_range_lenient = np.percentile(neuron_q99 - neuron_q01, 99)  # 99th percentile instead of 95th
    min_median_lenient = np.percentile(neuron_medians, 0.1)  # 0.1st percentile instead of 1st
    max_std_lenient = np.percentile(neuron_stds, 99.9)  # 99.9th percentile instead of 99th
    max_negative_lenient = 0.8  # Allow 80% negative instead of 50%
    
    lenient_mask = (
        (neuron_iqr > min_iqr_lenient) & 
        ((neuron_q99 - neuron_q01) < max_range_lenient) & 
        (neuron_medians > min_median_lenient) &
        (neuron_stds < max_std_lenient) &
        (negative_fraction < max_negative_lenient)
    )
    
    print(f"  Lenient filter: {lenient_mask.sum()}/{len(lenient_mask)} pass ({lenient_mask.sum()/len(lenient_mask)*100:.1f}%)")

if __name__ == "__main__":
    debug_filtering()