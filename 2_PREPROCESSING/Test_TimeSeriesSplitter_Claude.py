#!/usr/bin/env python3
"""
Quick test of the TimeSeriesSplitter with the new leak-aware logic
"""
import sys
import os
import numpy as np

# Add utils directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_dir = os.path.join(parent_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from data_splitter import TimeSeriesSplitter

def test_leak_aware_splitting():
    """Test the leak-aware splitting logic with a simple example"""
    
    # Create a simple split map: 100 train, 20 val, 30 test
    split_map = np.concatenate([
        np.zeros(100, dtype=int),      # 0-99: train
        np.ones(20, dtype=int),        # 100-119: val  
        np.full(30, 2, dtype=int)      # 120-149: test
    ])
    
    print(f"Split map: {len(split_map)} total points")
    print(f"  Train: 0-99 (100 points)")
    print(f"  Val: 100-119 (20 points)")
    print(f"  Test: 120-149 (30 points)")
    
    # Test with seq_len=10, pred_len=5 (total window = 15)
    seq_len, pred_len, label_len = 10, 5, 3
    
    splitter = TimeSeriesSplitter(split_map, seq_len, pred_len, label_len)
    
    print(f"\nTesting with seq_len={seq_len}, pred_len={pred_len}, label_len={label_len}")
    print(f"Total window length: {seq_len + pred_len} = {splitter.total_window_len}")
    
    # Check some specific examples
    print(f"\n=== Sample Analysis ===")
    
    train_indices = splitter.get_indices('train')
    val_indices = splitter.get_indices('val')  
    test_indices = splitter.get_indices('test')
    
    print(f"Training samples: {len(train_indices)}")
    if len(train_indices) > 0:
        print(f"  First few: {train_indices[:5]}")
        print(f"  Last few: {train_indices[-5:]}")
        
        # Check that training samples stay within training data
        for i in train_indices[:3]:  # Check first 3
            window_end = i + splitter.total_window_len
            window_splits = split_map[i:window_end]
            print(f"  Train sample {i}: window {i}-{window_end-1}, splits: {set(window_splits)}")
    
    print(f"\nValidation samples: {len(val_indices)}")  
    if len(val_indices) > 0:
        print(f"  First few: {val_indices[:5]}")
        print(f"  Last few: {val_indices[-5:]}")
        
        # Check validation samples can span train/val but not test
        for i in val_indices[:3]:
            window_end = i + splitter.total_window_len
            window_splits = split_map[i:window_end]
            print(f"  Val sample {i}: window {i}-{window_end-1}, splits: {set(window_splits)}")
    
    print(f"\nTest samples: {len(test_indices)}")
    if len(test_indices) > 0:
        print(f"  First few: {test_indices[:5]}")
        print(f"  Last few: {test_indices[-5:]}")
        
        # Check test samples
        for i in test_indices[:3]:
            window_end = i + splitter.total_window_len
            window_splits = split_map[i:window_end]
            print(f"  Test sample {i}: window {i}-{window_end-1}, splits: {set(window_splits)}")

    # Summary statistics
    summary = splitter.get_split_summary()
    print(f"\n=== Split Summary ===")
    for key, value in summary.items():
        if 'pct' in key:
            print(f"  {key}: {value:.1f}%")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    test_leak_aware_splitting()