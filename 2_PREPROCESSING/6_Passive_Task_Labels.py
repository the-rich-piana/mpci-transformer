#Rastermap is in `source_project_env`

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import h5py
from pathlib import Path
from rastermap import Rastermap

# Setup path
parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def main():
    # Path to the passive dataset
    data_path = '/cs/student/projects1/aibh/2024/gcosta/mpci_data/session_5ea6bb9b-6163-4e8a-816b-efe7002666b0_7000_passive.h5'
    
    # Create figures directory
    figures_dir = Path('./figures')
    figures_dir.mkdir(exist_ok=True)
    
    # Load data from H5 file and inspect structure
    print("Loading data from H5 file...")
    with h5py.File(data_path, 'r') as f:
        print("H5 file structure:")
        def print_structure(name, obj):
            print(f"  {name}: {type(obj).__name__}")
            if hasattr(obj, 'shape'):
                print(f"    shape: {obj.shape}")
        f.visititems(print_structure)
        
        # Load main datasets
        activity = f['activity'][:]
        timestamps = f['timestamps'][:]
        covariate_matrix = f['covariate_matrix'][:]
        
        # Try to find stimOff_times
        stimoff_times = None
        if 'trial_data/stimOff_times' in f:
            stimoff_times = f['trial_data/stimOff_times'][:]
            print(f"Found stimOff_times with shape: {stimoff_times.shape}")
        else:
            print("stimOff_times not found in H5 file")
    
    print(f"Activity shape: {activity.shape}")
    print(f"Timestamps shape: {timestamps.shape}")
    print(f"Covariate matrix shape: {covariate_matrix.shape}")
    
    # Extract trial_phase (last column of covariate matrix)
    trial_phase = covariate_matrix[:, -1]  # trial_phase is the last feature
    
    # Calculate actual sampling rate from timestamp data
    duration_seconds = timestamps[-1] - timestamps[0]
    n_timesteps = len(timestamps)
    actual_sampling_rate = (n_timesteps - 1) / duration_seconds  # -1 because we have n-1 intervals
    
    # Convert timestamps to minutes for display purposes only
    timestamps_minutes = (timestamps - timestamps[0]) / 60
    duration_minutes = timestamps_minutes[-1]
    
    print(f"=== SAMPLING RATE ANALYSIS ===")
    print(f"Total timesteps: {n_timesteps}")
    print(f"Duration: {duration_seconds:.3f} seconds ({duration_minutes:.1f} minutes)")
    print(f"Actual sampling rate: {actual_sampling_rate:.3f} Hz (not exactly 5Hz)")
    
    print(f"\n=== TRIAL PHASE ANALYSIS ===")
    print(f"Trial phase values: min={trial_phase.min()}, max={trial_phase.max()}")
    print(f"Trial phase proportion: {trial_phase.mean():.3f} (fraction in trial)")
    
    # Find exact timestep index for last stimOff (PURE TIMESTEP LOGIC)
    if stimoff_times is not None:
        last_stimoff_time_seconds = stimoff_times[-1]
        # Find exact timestep where last stimOff occurs
        last_stimoff_timestep_idx = np.searchsorted(timestamps, last_stimoff_time_seconds, side='right') - 1
        last_stimoff_timestep_idx = max(0, min(last_stimoff_timestep_idx, len(timestamps) - 1))  # Clamp to valid range
        
        print(f"\n=== STIMOFF-BASED SPLIT ANALYSIS ===")
        print(f"Last stimOff time: {last_stimoff_time_seconds:.3f} seconds")
        print(f"Last stimOff timestep index: {last_stimoff_timestep_idx}")
        print(f"Expected timestep (at 4.9Hz): {last_stimoff_time_seconds * 4.9:.0f}")
        print(f"Actual timestep (from searchsorted): {last_stimoff_timestep_idx}")
        print(f"Timestamp at found index: {timestamps[last_stimoff_timestep_idx]:.3f} seconds")
        
        # Use stimOff-based split (pure timestep logic)
        split_timestep_idx = last_stimoff_timestep_idx
        
    else:
        print(f"\n=== FALLBACK: TRIAL_PHASE-BASED SPLIT ===")
        # Fallback: Find the last trial timepoint from trial_phase
        last_trial_idx = np.where(trial_phase == 1)[0][-1] if np.any(trial_phase == 1) else 0
        split_timestep_idx = last_trial_idx
        print(f"Last trial phase timestep index: {split_timestep_idx}")
    
    # Convert split timestep to time units for display only
    split_time_seconds = timestamps[split_timestep_idx] - timestamps[0]
    split_time_minutes = split_time_seconds / 60
    
    print(f"\n=== FINAL SPLIT DECISION ===")
    print(f"Split at timestep index: {split_timestep_idx}")
    print(f"Split at time: {split_time_seconds:.3f} seconds ({split_time_minutes:.2f} minutes)")
    
    # Create comparison plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15), dpi=150)
    
    # Plot 1: Original trial_phase from covariate matrix
    colors = ['blue', 'lightgreen']  # 0=blue (inter-trial), 1=lightgreen (trial)
    for i in range(len(timestamps_minutes)-1):
        phase = int(trial_phase[i])
        ax1.axvspan(timestamps_minutes[i], timestamps_minutes[i+1], 
                   color=colors[phase], alpha=0.7, linewidth=0)
    
    ax1.set_ylabel('Trial Phase (Covariate)', fontsize=12)
    ax1.set_title('Trial Phase from Covariate Matrix\n(Light Green = Trial Period, Blue = Inter-Trial/Passive Period)', fontsize=14)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_yticks([])  # Remove y-axis ticks since legend explains colors
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Hardcoded active/passive split (using pure timestep logic)
    # Everything before split timestep = active (green), everything after = passive (blue)
    ax2.axvspan(0, split_time_minutes, color='lightgreen', alpha=0.7, linewidth=0, label='Active Period')
    ax2.axvspan(split_time_minutes, duration_minutes, color='blue', alpha=0.7, linewidth=0, label='Passive Period')
    
    # Add vertical line at transition
    ax2.axvline(split_time_minutes, color='red', linestyle='--', linewidth=2, label=f'Split at {split_time_minutes:.1f} min')
    
    ax2.set_ylabel('Hardcoded Split', fontsize=12)
    ax2.set_title('Hardcoded Active/Passive Split for Model Training\n(Green = Training/Validation, Blue = Test Set)', fontsize=14)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_yticks([])  # Remove y-axis ticks since legend explains colors
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Set up dual x-axis for first two plots
    for ax in [ax1, ax2]:
        # Primary x-axis (minutes)
        ax.set_xlim(0, duration_minutes)
        ax.set_xlabel('Time (minutes)', fontsize=12)
        
        # Secondary x-axis (timesteps)
        ax2_top = ax.secondary_xaxis('top')
        # Convert minute ticks to timestep indices using actual sampling rate
        minute_ticks = ax.get_xticks()
        timestep_indices = minute_ticks * 60 * actual_sampling_rate  # Use actual sampling rate
        ax2_top.set_xticks(minute_ticks)
        ax2_top.set_xticklabels([f'{int(ts)}' for ts in timestep_indices])
        ax2_top.set_xlabel('Timesteps (indices)', fontsize=12)
    
    # Add legends for first plot
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightgreen', label='Trial Period (trial_phase=1)'),
                      Patch(facecolor='blue', label='Inter-Trial Period (trial_phase=0)')]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Plot 3: Rastermap with behavioral state overlay
    print("\nComputing Rastermap embedding...")
    model = Rastermap(
        n_clusters=100,  # number of clusters to compute
        n_PCs=200,  # number of PCs
        locality=0.5,  # locality in sorting is low here to get more global sorting (this is a value from 0-1)
        time_lag_window=5,  # use future timepoints to compute correlation
        grid_upsample=10,  # default value, 10 is good for large recordings
    ).fit(activity.T)  # activity is [time, neurons] so transpose to [neurons, time]

    y = model.embedding  # neurons x 1
    isort = model.isort
    X_embedding = model.X_embedding  # binned activity for visualization

    # Verify dimensions match
    print(f"Activity shape: {activity.shape}")
    print(f"X_embedding shape: {X_embedding.shape}")
    
    # Check that time dimensions match
    assert X_embedding.shape[1] == activity.shape[0], f"Time dimensions don't match: X_embedding[1]={X_embedding.shape[1]}, activity[0]={activity.shape[0]}"
    
    # Plot rastermap as background (no extent - use natural timestep coordinates)
    ax3.imshow(X_embedding, vmin=0, vmax=0.8, cmap="gray_r", aspect="auto")
    
    # Add colored overlays for behavioral states using exact timestep indices
    total_timesteps = activity.shape[0]
    
    # Active period (green overlay) - from 0 to split timestep
    ax3.axvspan(0, split_timestep_idx, color='lightgreen', alpha=0.2, 
                label='Active Period (Train/Val)')
    
    # Passive period (blue overlay) - from split timestep to end
    ax3.axvspan(split_timestep_idx, total_timesteps, color='blue', alpha=0.2,
                label='Passive Period (Test)')
    
    # Add vertical line at exact split timestep
    ax3.axvline(split_timestep_idx, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    # Formatting for rastermap plot (now using timestep coordinates)
    ax3.set_xlabel('Timesteps (indices)', fontsize=12)
    ax3.set_ylabel('Sorted Neurons', fontsize=12)
    ax3.set_title('Neural Activity Rastermap with Active/Passive Behavioral State Overlay\n(Green = Active Period, Blue = Passive Period)', fontsize=14)
    ax3.set_xlim(0, total_timesteps)
    
    # Add time (minutes) axis on top for rastermap
    ax3_top = ax3.secondary_xaxis('top')
    timestep_ticks = ax3.get_xticks()
    minute_labels = timestep_ticks / (60 * actual_sampling_rate)  # Use actual sampling rate
    ax3_top.set_xticks(timestep_ticks)
    ax3_top.set_xticklabels([f'{min_val:.0f}' for min_val in minute_labels])
    ax3_top.set_xlabel('Time (minutes)', fontsize=12)
    
    # Add legend for rastermap overlay  
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    raster_legend_elements = [
        Patch(facecolor='lightgreen', alpha=0.3, label='Active Period (Train/Val)'),
        Patch(facecolor='blue', alpha=0.3, label='Passive Period (Test)'),
        Line2D([0], [0], color='red', linestyle='--', label=f'Split at timestep {split_timestep_idx}')
    ]
    ax3.legend(handles=raster_legend_elements, loc='upper right')
    
    # Now save the complete figure with all three plots
    plt.tight_layout()
    
    # Save the plot
    output_path = figures_dir / 'trial_phase_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Complete comparison plot with rastermap saved to: {output_path}")
    
    # Print summary statistics (using corrected timestep-based split)
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total recording duration: {duration_minutes:.1f} minutes ({n_timesteps} timesteps)")
    print(f"Active period (train/val): timesteps 0 - {split_timestep_idx} ({split_time_minutes:.1f} min)")
    print(f"Passive period (test): timesteps {split_timestep_idx} - {n_timesteps} ({duration_minutes - split_time_minutes:.1f} min)")
    print(f"Active/Passive ratio: {split_timestep_idx/n_timesteps:.1%} / {(n_timesteps - split_timestep_idx)/n_timesteps:.1%}")
    print(f"Corrected split using actual sampling rate: {actual_sampling_rate:.3f} Hz")
    
    plt.show()

if __name__ == '__main__':
    main()