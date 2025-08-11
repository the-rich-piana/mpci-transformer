import numpy as np
from typing import Tuple
from load_meso_session import MesoscopeSession

def build_covariate_matrix(session: MesoscopeSession, timestamps: np.ndarray, aligned_wheel_velocity: np.ndarray) -> Tuple[np.ndarray, list]:
    """
    Build comprehensive covariate matrix aligned to neural timestamps.
    
    Args:
        session: MesoscopeSession object with trial and behavioral data
        timestamps: Neural timestamps to align covariates to [n_timepoints]
        aligned_wheel_velocity: Pre-calculated aligned wheel velocity [n_timepoints]
        
    Returns:
        covariate_matrix: [n_timepoints, 11] matrix with all covariates
        feature_names: List of feature names for documentation
    """
    n_timepoints = len(timestamps)
    
    # Get wheel velocity (passed as parameter)
    wheel_velocity = aligned_wheel_velocity.reshape(-1, 1)
    
    # Get stimulus type encoding (9 types)
    stimulus_encoding = encode_stimulus_types(session, timestamps)
    
    # Get trial phase encoding (stimulus vs inter-trial)
    trial_phase = encode_trial_phase(session, timestamps)
    
    # Concatenate all features
    covariate_matrix = np.concatenate([
        wheel_velocity,          # [n_timepoints, 1]
        stimulus_encoding,       # [n_timepoints, 9] 
        trial_phase             # [n_timepoints, 1]
    ], axis=1)
    
    # Feature names for documentation
    feature_names = [
        'wheel_velocity',
        'stimulus_catch_trial',
        'stimulus_left_100pct',
        'stimulus_left_25pct', 
        'stimulus_left_12.5pct',
        'stimulus_left_6.25pct',
        'stimulus_right_100pct',
        'stimulus_right_25pct',
        'stimulus_right_12.5pct', 
        'stimulus_right_6.25pct',
        'trial_phase'
    ]
    
    assert covariate_matrix.shape == (n_timepoints, 11), f"Expected shape ({n_timepoints}, 11), got {covariate_matrix.shape}"
    
    return covariate_matrix, feature_names

def encode_stimulus_types(session: MesoscopeSession, timestamps: np.ndarray) -> np.ndarray:
    """
    Create one-hot encoding for stimulus types based on contrast combinations.
    
    Stimulus types are defined by contrast combinations:
    - 0: Catch trials - (left=0 & right=NaN) OR (left=NaN & right=0) - no stimulus
    - 1-4: Left side stimuli (100%, 25%, 12.5%, 6.25%) with right=NaN
    - 5-8: Right side stimuli (100%, 25%, 12.5%, 6.25%) with left=NaN
    
    Args:
        session: MesoscopeSession object
        timestamps: 1D array of neural timestamps to encode [n_timepoints]
        
    Returns:
        np.ndarray: One-hot encoded stimulus types [n_timepoints, 9]
    """
    # Define contrast values (excluding 0% as it's catch trial)
    contrast_values = [1.0, 0.25, 0.125, 0.0625]  # 100%, 25%, 12.5%, 6.25%
    
    # Initialize one-hot matrix
    n_timepoints = len(timestamps)
    stimulus_encoding = np.zeros((n_timepoints, 9), dtype=np.float32)
    
    # For each trial, find which timepoints fall within stimulus presentation
    for trial_idx in range(len(session.stimOn_times)):
        stim_on = session.stimOn_times[trial_idx]
        stim_off = session.stimOff_times[trial_idx]
        
        # Skip trials with NaN timing
        if np.isnan(stim_on) or np.isnan(stim_off):
            continue
        
        # Find timepoints within this stimulus period
        mask = (timestamps >= stim_on) & (timestamps <= stim_off)
        
        if not np.any(mask):
            continue
        
        # Get contrasts for this trial
        left_contrast = session.contrastLeft[trial_idx]
        right_contrast = session.contrastRight[trial_idx]
        
        # Determine stimulus type
        stimulus_type = _get_stimulus_type(left_contrast, right_contrast, contrast_values)
        
        # Set one-hot encoding for this time period
        stimulus_encoding[mask, stimulus_type] = 1.0
    
    return stimulus_encoding

def _get_stimulus_type(left_contrast: float, right_contrast: float, contrast_values: list) -> int:
    """
    Map contrasts to stimulus type index.
    
    Stimulus types:
    0: Catch trial (left=0 & right=NaN) OR (left=NaN & right=0) - no stimulus
    1-4: Left stimuli (1.0, 0.25, 0.125, 0.0625) with right=NaN
    5-8: Right stimuli (1.0, 0.25, 0.125, 0.0625) with left=NaN
    
    Returns:
        int: Stimulus type (0-8)
    """
    # Catch trials: no stimulus on either side
    if (left_contrast == 0.0 and np.isnan(right_contrast)) or (np.isnan(left_contrast) and right_contrast == 0.0):
        return 0
    
    # Left stimulus (right = NaN, left has contrast > 0)
    if np.isnan(right_contrast) and not np.isnan(left_contrast) and left_contrast > 0.0:
        try:
            contrast_idx = contrast_values.index(left_contrast)
            return 1 + contrast_idx  # Types 1-4
        except ValueError:
            # Find closest match for floating point precision issues
            distances = [abs(left_contrast - cv) for cv in contrast_values]
            closest_idx = distances.index(min(distances))
            return 1 + closest_idx
    
    # Right stimulus (left = NaN, right has contrast > 0)  
    if np.isnan(left_contrast) and not np.isnan(right_contrast) and right_contrast > 0.0:
        try:
            contrast_idx = contrast_values.index(right_contrast)
            return 5 + contrast_idx  # Types 5-8
        except ValueError:
            # Find closest match for floating point precision issues
            distances = [abs(right_contrast - cv) for cv in contrast_values]
            closest_idx = distances.index(min(distances))
            return 5 + closest_idx
    
    # Default to catch trial if no clear match
    return 0

def encode_trial_phase(session: MesoscopeSession, timestamps: np.ndarray) -> np.ndarray:
    """
    Create binary encoding for trial phase: stimulus period vs inter-trial interval.
    
    Args:
        session: MesoscopeSession object
        timestamps: 1D array of neural timestamps [n_timepoints]
        
    Returns:
        np.ndarray: Binary trial phase encoding [n_timepoints, 1]
                   1.0 = stimulus period, 0.0 = inter-trial interval
    """
    n_timepoints = len(timestamps)
    trial_phase = np.zeros((n_timepoints, 1), dtype=np.float32)
    
    # Mark all timepoints that fall within any stimulus period
    for trial_idx in range(len(session.stimOn_times)):
        stim_on = session.stimOn_times[trial_idx]
        stim_off = session.stimOff_times[trial_idx]
        
        # Skip trials with NaN timing
        if np.isnan(stim_on) or np.isnan(stim_off):
            continue
        
        # Find timepoints within this stimulus period
        mask = (timestamps >= stim_on) & (timestamps <= stim_off)
        trial_phase[mask, 0] = 1.0
    
    return trial_phase

def get_covariate_descriptions() -> dict:
    """
    Get detailed descriptions of each covariate feature for documentation.
    
    Returns:
        dict: Feature descriptions for HDF5 metadata
    """
    descriptions = {
        'wheel_velocity': 'Aligned wheel velocity in rad/s, calculated from wheel position with 5Hz sampling',
        'stimulus_catch_trial': 'Binary indicator for catch trials (no visual stimulus presented)', 
        'stimulus_left_100pct': 'Binary indicator for left side 100% contrast stimulus',
        'stimulus_left_25pct': 'Binary indicator for left side 25% contrast stimulus',
        'stimulus_left_12.5pct': 'Binary indicator for left side 12.5% contrast stimulus',
        'stimulus_left_6.25pct': 'Binary indicator for left side 6.25% contrast stimulus',
        'stimulus_right_100pct': 'Binary indicator for right side 100% contrast stimulus',
        'stimulus_right_25pct': 'Binary indicator for right side 25% contrast stimulus',
        'stimulus_right_12.5pct': 'Binary indicator for right side 12.5% contrast stimulus',
        'stimulus_right_6.25pct': 'Binary indicator for right side 6.25% contrast stimulus',
        'trial_phase': 'Binary indicator for trial phase: 1.0=stimulus period, 0.0=inter-trial interval'
    }
    return descriptions