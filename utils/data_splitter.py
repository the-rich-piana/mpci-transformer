import numpy as np
from typing import Tuple, List, Optional

class TimeSeriesSplitter:
    """
    Generates leak-free lists of valid starting indices for time series forecasting.

    A sample starting at index `i` is considered valid for a given split (e.g., 'train')
    if its entire required window of data falls completely within that split.
    """
    def __init__(self, split_map, seq_len, pred_len, label_len=48):
        """
        Args:
            split_map (np.ndarray): An array of the same length as the data, where each
                                     element is an integer representing the split
                                     (e.g., 0=train, 1=val, 2=test).
            seq_len (int): Length of the input sequence.
            pred_len (int): Length of the prediction sequence.
            label_len (int): Length of the overlap/decoder seed.
        """
        self.split_map = split_map
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len

        # The total number of consecutive timepoints a single sample needs.
        # This is from the start of the input (s_begin) to the end of the target (r_end).
        self.total_window_len = self.seq_len + self.pred_len
        
        # --- These will store our final, clean lists of indices ---
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        self._generate_safe_indices()

    def _generate_safe_indices(self):
        """The core logic to find all valid starting points with proper leak allowance."""
        print("Generating leak-aware sample indices...")
        
        # Iterate through every possible starting point in the entire dataset
        for i in range(len(self.split_map) - self.total_window_len + 1):
            
            # Define the full window this sample would touch
            window = self.split_map[i : i + self.total_window_len]
            
            # The starting point determines which split this sample belongs to
            start_split_type = window[0]
            
            # TRAINING: Entire window must be in training data
            if start_split_type == 0:  # Starts in train
                if np.all(window == 0):  # Entire window in train
                    self.train_indices.append(i)
            
            # VALIDATION: Can start in train or val, but cannot extend into test
            if (start_split_type == 0 or start_split_type == 1) and not np.any(window == 2):  # Starts in train/val, no test
                if start_split_type == 1 or np.any(window == 1):  # Must touch validation data
                    self.val_indices.append(i)
            
            # TEST: Can start anywhere, but this sample belongs to test if it starts in test
            # OR if it starts in train/val but extends into test data
            if start_split_type == 2:  # Starts in test
                self.test_indices.append(i)
            elif (start_split_type == 0 or start_split_type == 1) and np.any(window == 2):  # Extends into test
                self.test_indices.append(i)
        
        print(f"Found {len(self.train_indices)} valid training samples.")
        print(f"Found {len(self.val_indices)} valid validation samples.")
        print(f"Found {len(self.test_indices)} valid test samples.")

    def get_indices(self, flag='train'):
        """Public method to get the clean index list for a given split."""
        if flag == 'train':
            return self.train_indices
        elif flag == 'val':
            return self.val_indices
        elif flag == 'test':
            return self.test_indices
        else:
            raise ValueError("Flag must be 'train', 'val', or 'test'.")
    
    def get_split_summary(self):
        """Return summary statistics about the splits."""
        total_samples = len(self.train_indices) + len(self.val_indices) + len(self.test_indices)
        return {
            'train_samples': len(self.train_indices),
            'val_samples': len(self.val_indices),
            'test_samples': len(self.test_indices),
            'total_samples': total_samples,
            'train_pct': len(self.train_indices) / total_samples * 100 if total_samples > 0 else 0,
            'val_pct': len(self.val_indices) / total_samples * 100 if total_samples > 0 else 0,
            'test_pct': len(self.test_indices) / total_samples * 100 if total_samples > 0 else 0
        }

    @staticmethod
    def create_stimulus_based_splits(covariate_matrix: np.ndarray, 
                                   train_pct: float = 0.7, val_pct: float = 0.1, 
                                   held_out_stimulus_types: Optional[List[int]] = None) -> np.ndarray:
        """
        Create stimulus-based train/val/test splits using existing covariate matrix.
        
        Args:
            covariate_matrix: [n_timepoints, 11] covariate matrix from preprocessed data
                            Columns 1-9 are stimulus types (catch + 4 left + 4 right)
            train_pct: Percentage for training (default 0.7)
            val_pct: Percentage for validation (default 0.1) 
            held_out_stimulus_types: List of stimulus types to hold out for test set
                                   (e.g., [1] for Left 100% contrast)
                                   
        Returns:
            np.ndarray: Split map where 0=train, 1=val, 2=test
        """
        if held_out_stimulus_types is None:
            held_out_stimulus_types = []
            
        test_pct = 1.0 - train_pct - val_pct
        n_timepoints = covariate_matrix.shape[0]
        
        # Extract stimulus encoding (columns 1-9 are the 9 stimulus types)
        stimulus_onehot = covariate_matrix[:, 1:10]  # [n_timepoints, 9]
        
        # Initialize split map (default to train)
        split_map = np.zeros(n_timepoints, dtype=int)
        
        # For each stimulus type, create balanced splits
        for stim_type in range(9):  # 9 stimulus types (0=catch, 1-4=left, 5-8=right)
            
            # Check if this stimulus type should be held out for testing
            if stim_type in held_out_stimulus_types:
                stimulus_mask = stimulus_onehot[:, stim_type] == 1
                split_map[stimulus_mask] = 2  # Test
                print(f"Held out stimulus type {stim_type} for testing: {stimulus_mask.sum()} timepoints")
                continue
            
            # Find all continuous blocks of this stimulus type
            stimulus_blocks = TimeSeriesSplitter._find_stimulus_blocks(stimulus_onehot[:, stim_type])
            
            if len(stimulus_blocks) == 0:
                continue
                
            # Split blocks into train/val/test based on percentages
            n_blocks = len(stimulus_blocks)
            n_train_blocks = int(n_blocks * train_pct)
            n_val_blocks = int(n_blocks * val_pct)
            
            # Assign blocks to splits (chronological order to preserve temporal structure)
            for i, (start_idx, end_idx) in enumerate(stimulus_blocks):
                if i < n_train_blocks:
                    split_assignment = 0  # Train
                elif i < n_train_blocks + n_val_blocks:
                    split_assignment = 1  # Val
                else:
                    split_assignment = 2  # Test
                    
                split_map[start_idx:end_idx+1] = split_assignment
            
            print(f"Stimulus type {stim_type}: {n_blocks} blocks -> {n_train_blocks} train, {n_val_blocks} val, {n_blocks-n_train_blocks-n_val_blocks} test")
        
        return split_map
    
    @staticmethod
    def _find_stimulus_blocks(stimulus_mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find continuous blocks where a stimulus is active.
        
        Args:
            stimulus_mask: Binary array indicating when stimulus is active [n_timepoints]
            
        Returns:
            List of (start_idx, end_idx) tuples for continuous stimulus blocks
        """
        stimulus_indices = np.where(stimulus_mask)[0]
        
        if len(stimulus_indices) == 0:
            return []
        
        blocks = []
        block_start = stimulus_indices[0]
        
        for i in range(1, len(stimulus_indices)):
            # If there's a gap, end current block and start new one
            if stimulus_indices[i] - stimulus_indices[i-1] > 1:
                blocks.append((block_start, stimulus_indices[i-1]))
                block_start = stimulus_indices[i]
        
        # Add the final block
        blocks.append((block_start, stimulus_indices[-1]))
        
        return blocks