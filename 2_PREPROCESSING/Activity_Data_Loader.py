import os
import sys
import numpy as np
import pandas as pd
import h5py
import warnings
warnings.filterwarnings('ignore')

# Add utils directory to path for TimeSeriesSplitter
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_dir = os.path.join(parent_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from data_splitter import TimeSeriesSplitter

# Simple base class to replace torch Dataset for testing
class SimpleDataset:
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError

class Dataset_Activity(SimpleDataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='activity_raw.csv', # Changed default data_path
                 target='OT', scale=True, timeenc=0, freq='h'): # Target is now a placeholder
        
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        # Fix: label_len should not exceed seq_len for DLinear
        if self.label_len > self.seq_len:
            print(f"Warning: label_len ({self.label_len}) > seq_len ({self.seq_len}). Setting label_len = seq_len // 2 = {self.seq_len // 2}")
            self.label_len = max(1, self.seq_len // 2)
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target # This will be ignored but is kept for API compatibility
        self.scale = scale
        self.timeenc = timeenc # This will be ignored but is kept for API compatibility
        self.freq = freq # This will be ignored but is kept for API compatibility

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        # Read preprocessed HDF5 data
        file_path = os.path.join(self.root_path, self.data_path)
        
        with h5py.File(file_path, 'r') as f:
            # Load the already preprocessed data (ΔF/F normalized and filtered)
            processed_data = f['activity'][:]
            timestamps = f['timestamps'][:]
            
            # Load covariate matrix
            covariate_matrix = f['covariate_matrix'][:]
            feature_names = [name.decode('utf-8') for name in f['covariate_metadata']['feature_names'][:]]
            
            # Load metadata
            n_original_neurons = f['metadata'].attrs['n_original_neurons']
            n_processed_neurons = f['metadata'].attrs['n_processed_neurons']
            
            # Load normalization parameters for inverse transform
            if 'normalization' in f:
                self.norm_q005 = f['normalization']['q005'][:]
                self.norm_q995 = f['normalization']['q995'][:]
                self.norm_range = f['normalization']['data_range'][:]
            else:
                self.norm_q005 = None
                self.norm_q995 = None
                self.norm_range = None
            
            print(f"Loaded preprocessed data: {processed_data.shape}")
            print(f"Loaded covariate matrix: {covariate_matrix.shape}")
            print(f"Original neurons: {n_original_neurons}, Processed neurons: {n_processed_neurons}")
            print(f"Data range: {processed_data.min():.3f} to {processed_data.max():.3f}")
            print(f"Covariate features: {feature_names}")
        
        # === NEW SPLITTING LOGIC USING TIMESERIESPLITTER ===
        # Create stimulus-based split map using covariate matrix
        print("\nCreating stimulus-based splits...")
        split_map = TimeSeriesSplitter.create_stimulus_based_splits(
            covariate_matrix=covariate_matrix,
            train_pct=0.7, 
            val_pct=0.1,
            held_out_stimulus_types=[]  # No held-out types for now
        )
        
        # Create TimeSeriesSplitter to get leak-free sample indices
        splitter = TimeSeriesSplitter(
            split_map=split_map,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            label_len=self.label_len
        )
        
        # Get valid starting indices for the current dataset flag
        flag_map = {0: 'train', 1: 'val', 2: 'test'}
        current_flag = flag_map[self.set_type]
        self.valid_indices = splitter.get_indices(flag=current_flag)
        
        print(f"Split summary:")
        summary = splitter.get_split_summary()
        for key, value in summary.items():
            if 'pct' in key:
                print(f"  {key}: {value:.1f}%")
            else:
                print(f"  {key}: {value}")
        
        if len(self.valid_indices) == 0:
            raise RuntimeError(f"No valid samples found for split '{current_flag}'. Check window sizes and data length.")
        
        # === SELECT TOP NEURONS FROM PREPROCESSED DATA ===
        # Calculate neuron means across entire dataset (no temporal leakage concern)
        neuron_means = np.mean(processed_data, axis=0)
        
        # Select top neurons with highest mean activity
        num_neurons_to_keep = min(5000, processed_data.shape[1])
        top_indices = np.argsort(neuron_means)[-num_neurons_to_keep:]
        
        print(f"Selected top {num_neurons_to_keep} neurons out of {processed_data.shape[1]} preprocessed neurons")
        
        # Apply neuron selection
        data_filtered = processed_data[:, top_indices]
        
        # Store full data for indexing in __getitem__ - no border slicing!
        self.data_x = data_filtered  # Full dataset
        self.data_y = data_filtered  # Full dataset (same as data_x)
        self.covariate_data = covariate_matrix  # Full covariate matrix
        self.feature_names = feature_names
        self.data_stamp = np.zeros((data_filtered.shape[0], 1))
        
        # Store selection info and normalization parameters for selected neurons
        self.top_indices = top_indices
        if self.norm_q005 is not None:
            self.norm_q005 = self.norm_q005[:, top_indices]
            self.norm_q995 = self.norm_q995[:, top_indices]
            self.norm_range = self.norm_range[:, top_indices]
        
        print(f"Full dataset shape: {self.data_x.shape}")
        print(f"Full covariate shape: {self.covariate_data.shape}")
        print(f"Valid {current_flag} samples: {len(self.valid_indices)}")
        print(f"Data statistics - Mean: {self.data_x.mean():.3f}, Std: {self.data_x.std():.3f}")

    def __getitem__(self, index):
        # CRITICAL CHANGE: 'index' now refers to an index into our list of SAFE starting points.
        s_begin = self.valid_indices[index]  # Get actual starting point from valid indices
        print(f"Sample index: {index} -> Data index: {s_begin}")
        
        # Rest of sequence extraction logic remains the same
        s_end = s_begin + self.seq_len  # End of input sequence
        print(f"Input sequence: {s_begin} to {s_end-1} ({self.seq_len} steps)")
        
        # Target sequence start (overlaps with end of input by label_len)
        r_begin = s_end - self.label_len  # Start of target sequence (includes overlap)
        print(f"Target sequence start (r_begin): {r_begin} (overlap of {self.label_len} steps)")
        
        # Target sequence end (target start + label overlap + prediction length)
        r_end = r_begin + self.label_len + self.pred_len  # End of target sequence
        print(f"Target sequence: {r_begin} to {r_end-1} (total target length: {self.label_len + self.pred_len})")
        
        # Bounds checking (should never fail due to TimeSeriesSplitter validation)
        if r_end > len(self.data_x):
            raise IndexError(f"Sequence would exceed data bounds. Data length: {len(self.data_x)}, required end: {r_end}")

        # Extract neural data sequences
        seq_x = self.data_x[s_begin:s_end]  # Input neural sequence
        seq_y = self.data_y[r_begin:r_end]  # Target neural sequence (with overlap)
        
        # Ensure consistent shapes and dtypes
        seq_x = np.array(seq_x, dtype=np.float32)
        seq_y = np.array(seq_y, dtype=np.float32)
        
        # Verify expected shapes
        if seq_x.shape[0] != self.seq_len:
            raise ValueError(f"seq_x has wrong length: {seq_x.shape[0]}, expected: {self.seq_len}")
        if seq_y.shape[0] != (self.label_len + self.pred_len):
            raise ValueError(f"seq_y has wrong length: {seq_y.shape[0]}, expected: {self.label_len + self.pred_len}")
        
        # Extract covariate sequences aligned to neural sequences
        seq_x_mark = self.covariate_data[s_begin:s_end]  # Input covariates (wheel, stimuli, trial phase)
        seq_y_mark = self.covariate_data[r_begin:r_end]  # Target covariates (known future values)
        
        # Ensure consistent dtypes
        seq_x_mark = np.array(seq_x_mark, dtype=np.float32)
        seq_y_mark = np.array(seq_y_mark, dtype=np.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # The length is now the number of valid samples we can create (leak-free)
        return len(self.valid_indices)

    def inverse_transform(self, data):
        """Convert normalized predictions back to original scale"""
        if hasattr(self, 'norm_q005') and self.norm_q005 is not None:
            # Reverse the robust normalization: unnormalized = (normalized * range) + q005
            return (data * self.norm_range) + self.norm_q005
        return data

# Test the dataset
if __name__ == "__main__":
    print("=== TESTING DATASET_ACTIVITY WITH COVARIATES ===")
    
    # Test with your session file
    dataset = Dataset_Activity(
        root_path="../DATA",
        data_path="session_61f260e7-b5d3-4865-a577-bcfc53fda8a8.h5",
        flag='train'
    )
    
    print(f"\nDataset length: {len(dataset)}")
    print(f"Feature names: {dataset.feature_names}")
    
    # Test a few samples
    i=504
    seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[i]
    
    print(f"\nSample {i}:")
    print(f"  seq_x (input neural) shape: {seq_x.shape}")
    print(f"  seq_y (target neural) shape: {seq_y.shape}")  
    print(f"  seq_x_mark (input covariates) shape: {seq_x_mark.shape}")
    print(f"  seq_y_mark (target covariates) shape: {seq_y_mark.shape}")
    #TODO: assert that shapes are correct.
    print(f"  Input wheel velocity range: {seq_x_mark[:, 0].min():.3f} to {seq_x_mark[:, 0].max():.3f}")
    print(f"  Input stimulus activity: {np.sum(seq_x_mark[:, 1:10].sum(axis=1) > 0)} samples with stimulus")
    print(f"  Target wheel velocity range: {seq_y_mark[:, 0].min():.3f} to {seq_y_mark[:, 0].max():.3f}")
    print(f"  Target stimulus activity: {np.sum(seq_y_mark[:, 1:10].sum(axis=1) > 0)} samples with stimulus")

    print("\n✓ Dataset_Activity successfully loads and provides covariate data!")
