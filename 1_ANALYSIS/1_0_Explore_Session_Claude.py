# %% [markdown]
# <h1>Load Session

# %%
import sys
sys.executable

# %%
from one.api import ONE
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import seaborn as sns


parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from utils.load_meso_session import MesoscopeSession

one = ONE()
SESSION_INDEX = 0


# %%
session = MesoscopeSession.from_eid(one, '61f260e7-b5d3-4865-a577-bcfc53fda8a8', True)

# %%
session.wheel_position

# %%
activity_matrix, timestamps = session.get_activity_matrix()

# %%
session.wheel_timestamps

# %%
session.wheel_position

# %%
import brainbox.behavior.wheel as wh

# print("Wheel V1")
Fs = 5
pos, t = wh.interpolate_position(session.wheel_timestamps, session.wheel_position, freq=Fs)
# Use lower corner frequency (must be < Fs/2 = 2.5 Hz)
vel, acc = wh.velocity_filtered(pos, Fs, corner_frequency=2)

print(f"Original vel range: {np.min(vel):.3f} to {np.max(vel):.3f} rad/s")

# Clip extreme velocity outliers (IBL data typically ranges -2 to +2 rad/s)
vel_clipped = np.clip(vel, -2.5, 2.5)  # Conservative clipping to ±10 rad/s
print(f"After clipping: {np.min(vel_clipped):.3f} to {np.max(vel_clipped):.3f} rad/s")

# Use clipped velocity for alignment
vel = vel_clipped

print(f"Activity matrix shape: {activity_matrix.shape}")
print(f"Activity timestamps length: {len(timestamps)}")
print(f"Activity duration: {timestamps[0]:.2f}s to {timestamps[-1]:.2f}s")
print(f"Wheel position length: {len(session.wheel_position)}")
print(f"Wheel timestamps length: {len(session.wheel_timestamps)}")
print(f"Wheel duration: {session.wheel_timestamps[0]:.2f}s to {session.wheel_timestamps[-1]:.2f}s")
print(f"Interpolated position length: {len(pos)}")
print(f"Interpolated time range: {t[0]:.2f}s to {t[-1]:.2f}s") 
print(f"Velocity array length: {len(vel)}")

# Create wheel velocity array aligned to neural timestamps
def create_aligned_wheel_velocity(wheel_vel, wheel_timestamps, neural_timestamps):
    """
    Create wheel velocity array aligned to neural timestamps.
    Uses 0 velocity where no wheel data exists (no movement).
    """
    from scipy.interpolate import interp1d
    
    # Initialize with zeros (no movement = zero velocity)
    aligned_velocity = np.zeros(len(neural_timestamps))
    
    # Find neural timestamps that overlap with wheel recording period
    wheel_start = wheel_timestamps[0]
    wheel_end = wheel_timestamps[-1]
    
    # Create mask for neural timestamps within wheel recording period
    overlap_mask = (neural_timestamps >= wheel_start) & (neural_timestamps <= wheel_end)
    
    if np.any(overlap_mask):
        # Interpolate wheel velocity for overlapping timestamps only
        interp_func = interp1d(wheel_timestamps, wheel_vel, 
                              kind='linear', bounds_error=False, fill_value=0)
        aligned_velocity[overlap_mask] = interp_func(neural_timestamps[overlap_mask])
    
    return aligned_velocity

# Create aligned wheel velocity 
aligned_wheel_velocity = create_aligned_wheel_velocity(vel, t, timestamps)

print(f"Aligned wheel velocity shape: {aligned_wheel_velocity.shape}")
print(f"Neural timestamps shape: {timestamps.shape}")
print(f"Non-zero velocity samples: {np.sum(aligned_wheel_velocity != 0)}")
print(f"Wheel recording covers: {t[0]:.1f}s to {t[-1]:.1f}s")
print(f"Neural recording covers: {timestamps[0]:.1f}s to {timestamps[-1]:.1f}s")

# Debug the alignment
print(f"\nDEBUG: Original vel range: {np.min(vel):.3f} to {np.max(vel):.3f}")
print(f"DEBUG: Aligned vel range: {np.min(aligned_wheel_velocity):.3f} to {np.max(aligned_wheel_velocity):.3f}")
print(f"DEBUG: Aligned vel non-zero range: {np.min(aligned_wheel_velocity[aligned_wheel_velocity != 0]):.3f} to {np.max(aligned_wheel_velocity[aligned_wheel_velocity != 0]):.3f}")

vel

# %%
# Plot wheel velocity with behavioral events (with zoom capability)

# Set time window for zooming (in minutes)
start_time_min = 10  # Start at 10 minutes
duration_min = 5     # Show 5 minutes worth of data

# Convert to seconds and find indices
start_time_sec = start_time_min * 60
end_time_sec = start_time_sec + (duration_min * 60)

# Find corresponding indices in neural data
start_idx = np.searchsorted(timestamps, start_time_sec)
end_idx = np.searchsorted(timestamps, end_time_sec)

# Create subset
time_subset = timestamps[start_idx:end_idx]
vel_subset = aligned_wheel_velocity[start_idx:end_idx]

fig, ax = plt.subplots(1, 1, figsize=(20, 6))

# Plot wheel velocity as continuous line
ax.plot(time_subset/60, vel_subset, 'k-', linewidth=0.8, alpha=0.7, label='Wheel Velocity')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Wheel Velocity (rad/s)')
ax.set_title(f'Wheel Velocity with Behavioral Events ({start_time_min}-{start_time_min+duration_min} min)')

# Plot stimulus onset times in the time window
stim_on_times = session.stimOn_times[~np.isnan(session.stimOn_times)]
stim_in_window = stim_on_times[(stim_on_times >= start_time_sec) & (stim_on_times < end_time_sec)]
for i, stim_time in enumerate(stim_in_window):
    ax.axvline(stim_time/60, color='red', alpha=0.6, linewidth=1, 
              label='Stimulus On' if i == 0 else "")

# Plot stimulus off times in the time window
stim_off_times = session.stimOff_times[~np.isnan(session.stimOff_times)]
stim_off_in_window = stim_off_times[(stim_off_times >= start_time_sec) & (stim_off_times < end_time_sec)]
for i, stim_off_time in enumerate(stim_off_in_window):
    ax.axvline(stim_off_time/60, color='orange', alpha=0.6, linewidth=1, linestyle='--',
              label='Stimulus Off' if i == 0 else "")

# Plot feedback times in the time window
feedback_times = session.feedback_times[~np.isnan(session.feedback_times)]
feedback_in_window = feedback_times[(feedback_times >= start_time_sec) & (feedback_times < end_time_sec)]
for i, feedback_time in enumerate(feedback_in_window):
    ax.axvline(feedback_time/60, color='blue', alpha=0.6, linewidth=1,
              label='Feedback' if i == 0 else "")

# Add legend and grid
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(start_time_min, start_time_min + duration_min)

plt.tight_layout()
plt.show()

print(f"Showing {start_time_min}-{start_time_min+duration_min} minutes")
print(f"Stimuli on in window: {len(stim_in_window)}")
print(f"Stimuli off in window: {len(stim_off_in_window)}")
print(f"Feedback events in window: {len(feedback_in_window)}")
print(f"Velocity range in window: {np.min(vel_subset):.3f} to {np.max(vel_subset):.3f} rad/s")

# %% [markdown]
# <h2> Summary Statistics

# %%
# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Get activity matrix
# activity_matrix, timestamps = session.get_activity_matrix()
print(f"Activity matrix shape: {activity_matrix.shape}")
print(f"Session duration: {timestamps[-1]:.1f} seconds ({timestamps[-1]/60:.1f} minutes)")

# 1. DATA OVERVIEW & BASIC STATISTICS
print(f"\n=== DATA OVERVIEW ===")
print(f"Total neurons: {activity_matrix.shape[1]}")
print(f"Total timepoints: {activity_matrix.shape[0]}")
print(f"Sampling rate: {1/np.mean(np.diff(timestamps)):.1f} Hz")

# Basic activity statistics
total_events = np.sum(activity_matrix > 0.1)  # Using 0.1 as threshold for "active"
sparsity = 1 - (total_events / activity_matrix.size)
print(f"Sparsity: {sparsity:.3f} ({100*sparsity:.1f}% zeros)")

# Per-neuron statistics
neuron_firing_rates = np.mean(activity_matrix > 0.5, axis=0)  # Fraction of time active
neuron_mean_activity = np.mean(activity_matrix, axis=0)

print(f"Active neurons (>1% time): {np.sum(neuron_firing_rates > 0.01)}/{len(neuron_firing_rates)}")
print(f"Mean firing rate: {np.mean(neuron_firing_rates):.3f}")

# 2. POPULATION ACTIVITY OVER TIME
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Population sum over time (downsampled for visualization)
downsample_factor = 50  # Show every 50th timepoint
pop_activity = np.sum(activity_matrix > 0.1, axis=1)  # Number of active neurons per timepoint
time_ds = timestamps[::downsample_factor]
pop_ds = pop_activity[::downsample_factor]

axes[0,0].plot(time_ds/60, pop_ds, alpha=0.7, linewidth=0.8)
axes[0,0].set_xlabel('Time (minutes)')
axes[0,0].set_ylabel('Active neurons')
axes[0,0].set_title('Population Activity Over Time')

# Distribution of neuron activity levels
axes[0,1].hist(neuron_mean_activity, bins=50, alpha=0.7, edgecolor='none')
axes[0,1].set_xlabel('Mean activity per neuron')
axes[0,1].set_ylabel('Count')
axes[0,1].set_title('Distribution of Neuron Activity')
axes[0,1].set_yscale('log')

# Firing rate distribution
axes[1,0].hist(neuron_firing_rates, bins=50, alpha=0.7, edgecolor='none')
axes[1,0].set_xlabel('Fraction of time active')
axes[1,0].set_ylabel('Count')
axes[1,0].set_title('Neuron Firing Rate Distribution')

# Activity heatmap (sample of neurons)
n_sample = min(100, activity_matrix.shape[1])  # Sample 100 neurons
sample_idx = np.random.choice(activity_matrix.shape[1], n_sample, replace=False)
sample_idx = sample_idx[np.argsort(neuron_firing_rates[sample_idx])[::-1]]  # Sort by activity

# Downsample time for heatmap
time_sample = slice(0, min(1000, activity_matrix.shape[0]))  # First 1000 timepoints
heatmap_data = activity_matrix[time_sample, sample_idx].T

im = axes[1,1].imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest')
axes[1,1].set_xlabel('Time (samples)')
axes[1,1].set_ylabel('Neurons (sorted by activity)')
axes[1,1].set_title(f'Activity Heatmap (top {n_sample} neurons)')

plt.tight_layout()
plt.show()

# # 3. FOV BREAKDOWN
# print(f"\n=== FOV BREAKDOWN ===")
# for fov_name, fov in session.fovs.items():
#     print(f"{fov_name}: {fov.n_neurons} neurons ({fov.n_neurons/session.n_total_neurons*100:.1f}%)")

# %%
sns.lineplot(activity_matrix[:100,1000])

# %%
session.plot_binary_activity_heatmap(time_window=50, threshold=0.9)

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from scipy.signal import correlate
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)

# Assume activity_matrix, timestamps, and session are already loaded
print(f"Working with {activity_matrix.shape} data (time x neurons)")

# =============================================================================
# 1. INFORMATION CONTENT ANALYSIS
# =============================================================================
print("\n=== INFORMATION CONTENT ANALYSIS ===")

# Calculate entropy for each neuron (measure of information content)
def calculate_neuron_entropy(activity, bins=20):
    """Calculate entropy for each neuron's activity distribution"""
    entropies = []
    for i in range(activity.shape[1]):
        hist, _ = np.histogram(activity[:, i], bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        entropies.append(entropy(hist))
    return np.array(entropies)

neuron_entropies = calculate_neuron_entropy(activity_matrix)
print(f"Mean neuron entropy: {np.mean(neuron_entropies):.3f}")

# Information content per FOV
# fov_entropies = {}
# neuron_idx = 0
# for fov_name, fov in session.fovs.items():
#     fov_activity = activity_matrix[:, neuron_idx:neuron_idx+fov.n_neurons]
#     fov_entropies[fov_name] = np.mean(calculate_neuron_entropy(fov_activity))
    # neuron_idx += fov.n_neurons

# =============================================================================
# 2. DIMENSIONALITY ANALYSIS (PCA)
# =============================================================================
print("\n=== DIMENSIONALITY ANALYSIS ===")

# Standardize data for PCA (sample every 10th timepoint for efficiency)
sample_step = 10
activity_sample = activity_matrix[::sample_step, :]
scaler = StandardScaler()
activity_scaled = scaler.fit_transform(activity_sample)

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(activity_scaled)
explained_var = pca.explained_variance_ratio_

# Find number of components for 80% and 95% variance
cumvar = np.cumsum(explained_var)
n_80 = np.argmax(cumvar >= 0.8) + 1
n_95 = np.argmax(cumvar >= 0.95) + 1

print(f"Components for 80% variance: {n_80}/{activity_matrix.shape[1]} ({n_80/activity_matrix.shape[1]*100:.1f}%)")
print(f"Components for 95% variance: {n_95}/{activity_matrix.shape[1]} ({n_95/activity_matrix.shape[1]*100:.1f}%)")

# =============================================================================
# 3. TIMESCALE ANALYSIS
# =============================================================================
print("\n=== TIMESCALE ANALYSIS ===")

# Calculate population activity for correlation analysis
pop_activity = np.mean(activity_matrix > 0.1, axis=1)  # Fraction of active neurons

# Calculate autocorrelation at different lags
def calculate_autocorr(signal, max_lag_samples):
    """Calculate autocorrelation up to max_lag"""
    autocorr = correlate(signal, signal, mode='full')
    mid = len(autocorr) // 2
    autocorr = autocorr[mid:mid+max_lag_samples+1]
    return autocorr / autocorr[0]  # Normalize

# Define time lags to analyze
sampling_rate = 1 / np.mean(np.diff(timestamps))
lags_seconds = [1, 10, 60]  # 1s, 10s, 1min
max_lag_samples = int(60 * sampling_rate)  # 1 minute worth of samples

autocorr = calculate_autocorr(pop_activity, max_lag_samples)
lag_times = np.arange(len(autocorr)) / sampling_rate

# =============================================================================
# PLOTTING
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Information content by neuron
axes[0,0].hist(neuron_entropies, bins=50, alpha=0.7, edgecolor='none')
axes[0,0].set_xlabel('Entropy (bits)')
axes[0,0].set_ylabel('Number of neurons')
axes[0,0].set_title('Information Content per Neuron')
axes[0,0].axvline(np.mean(neuron_entropies), color='red', linestyle='--', label=f'Mean: {np.mean(neuron_entropies):.2f}')
axes[0,0].legend()

# 2. Information content by FOV
fov_names = list(fov_entropies.keys())
fov_entropy_values = list(fov_entropies.values())
axes[0,1].bar(range(len(fov_names)), fov_entropy_values, alpha=0.7)
axes[0,1].set_xlabel('FOV')
axes[0,1].set_ylabel('Mean Entropy')
axes[0,1].set_title('Information Content by Brain Region')
axes[0,1].set_xticks(range(len(fov_names)))
axes[0,1].set_xticklabels(fov_names, rotation=45)

# 3. PCA explained variance
axes[0,2].plot(np.arange(1, min(100, len(explained_var))+1), 
               cumvar[:min(100, len(explained_var))], 'o-', markersize=3)
axes[0,2].axhline(0.8, color='red', linestyle='--', alpha=0.7, label='80%')
axes[0,2].axhline(0.95, color='orange', linestyle='--', alpha=0.7, label='95%')
axes[0,2].axvline(n_80, color='red', linestyle=':', alpha=0.7)
axes[0,2].axvline(n_95, color='orange', linestyle=':', alpha=0.7)
axes[0,2].set_xlabel('Principal Component')
axes[0,2].set_ylabel('Cumulative Variance Explained')
axes[0,2].set_title('Dimensionality of Neural Activity')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# 4. First few PCs over time
time_sample = timestamps[::sample_step]
for i in range(3):
    axes[1,0].plot(time_sample/60, pca_result[:, i], alpha=0.7, label=f'PC{i+1}')
axes[1,0].set_xlabel('Time (minutes)')
axes[1,0].set_ylabel('PC Score')
axes[1,0].set_title('Principal Components Over Time')
axes[1,0].legend()

# 5. Autocorrelation function
axes[1,1].plot(lag_times, autocorr, linewidth=2)
axes[1,1].set_xlabel('Lag (seconds)')
axes[1,1].set_ylabel('Autocorrelation')
axes[1,1].set_title('Population Activity Autocorrelation')
axes[1,1].grid(True, alpha=0.3)
axes[1,1].set_xlim(0, 60)

# Add vertical lines for key timescales
for lag in lags_seconds:
    if lag <= 60:
        lag_idx = int(lag * sampling_rate)
        if lag_idx < len(autocorr):
            axes[1,1].axvline(lag, color='red', linestyle='--', alpha=0.5)
            axes[1,1].text(lag, autocorr[lag_idx], f'{lag}s\n{autocorr[lag_idx]:.3f}', 
                          verticalalignment='bottom', fontsize=9)

# 6. Effective dimensionality over time windows
window_size = int(60 * sampling_rate)  # 1-minute windows
step_size = int(10 * sampling_rate)    # 10-second steps

def sliding_dimensionality(data, window_size, step_size, var_threshold=0.8):
    """Calculate effective dimensionality in sliding windows"""
    dims = []
    times = []
    
    for start in range(0, data.shape[0] - window_size, step_size):
        window = data[start:start+window_size, :]
        window_scaled = StandardScaler().fit_transform(window)
        
        pca_window = PCA()
        pca_window.fit(window_scaled)
        cumvar_window = np.cumsum(pca_window.explained_variance_ratio_)
        
        dim = np.argmax(cumvar_window >= var_threshold) + 1
        dims.append(dim)
        times.append(timestamps[start + window_size//2])
    
    return np.array(times), np.array(dims)

# Calculate sliding dimensionality (subsample for efficiency)
subsample_factor = 5
data_sub = activity_matrix[::subsample_factor, :]
times_sub = timestamps[::subsample_factor]

window_times, sliding_dims = sliding_dimensionality(
    data_sub, 
    window_size // subsample_factor, 
    step_size // subsample_factor
)

axes[1,2].plot(window_times/60, sliding_dims, alpha=0.7, linewidth=1)
axes[1,2].set_xlabel('Time (minutes)')
axes[1,2].set_ylabel('Effective Dimensionality (80% var)')
axes[1,2].set_title('Dimensionality Over Time')
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print(f"\n=== SUMMARY ===")
# print(f"Most informative FOV: {max(fov_entropies, key=fov_entropies.get)} (entropy: {max(fov_entropies.values()):.3f})")
# print(f"Least informative FOV: {min(fov_entropies, key=fov_entropies.get)} (entropy: {min(fov_entropies.values()):.3f})")
print(f"Effective dimensionality: {n_80}/{activity_matrix.shape[1]} neurons ({n_80/activity_matrix.shape[1]*100:.1f}%)")
print(f"Autocorrelation at 1s: {autocorr[int(1*sampling_rate)]:.3f}")
print(f"Autocorrelation at 10s: {autocorr[int(10*sampling_rate)]:.3f}")
if len(autocorr) > int(60*sampling_rate):
    print(f"Autocorrelation at 1min: {autocorr[int(60*sampling_rate)]:.3f}")
print(f"Mean sliding dimensionality: {np.mean(sliding_dims):.1f} ± {np.std(sliding_dims):.1f}")

# %%
# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("=== DETAILED SUMMARY STATISTICS ===")

# =============================================================================
# 1. ACTIVITY VALUE RANGES
# =============================================================================
print("\n--- Activity Value Ranges ---")
print(f"Global min: {np.min(activity_matrix):.4f}")
print(f"Global max: {np.max(activity_matrix):.4f}")
print(f"Global mean: {np.mean(activity_matrix):.4f}")
print(f"Global median: {np.median(activity_matrix):.4f}")
print(f"Global std: {np.std(activity_matrix):.4f}")


# %%
# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("=== DETAILED SUMMARY STATISTICS ===")

# =============================================================================
# 1. ACTIVITY VALUE RANGES
# =============================================================================
print("\n--- Activity Value Ranges ---")
print(f"Global min: {np.min(activity_matrix):.4f}")
print(f"Global max: {np.max(activity_matrix):.4f}")
print(f"Global mean: {np.mean(activity_matrix):.4f}")
print(f"Global median: {np.median(activity_matrix):.4f}")
print(f"Global std: {np.std(activity_matrix):.4f}")

# Percentiles
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
perc_values = np.percentile(activity_matrix, percentiles)
print(f"\nPercentiles:")
for p, v in zip(percentiles, perc_values):
    print(f"  {p}th: {v:.4f}")

# Fraction of zeros
zero_fraction = np.mean(activity_matrix == 0)
print(f"\nFraction of exact zeros: {zero_fraction:.3f} ({zero_fraction*100:.1f}%)")

# Fraction above different thresholds
thresholds = [0.01, 0.1, 0.5, 1.0, 2.0]
print(f"\nFraction above thresholds:")
for thresh in thresholds:
    frac = np.mean(activity_matrix > thresh)
    print(f"  >{thresh}: {frac:.4f} ({frac*100:.2f}%)")

# =============================================================================
# 2. NEURON-LEVEL STATISTICS
# =============================================================================
print("\n--- Neuron-Level Statistics ---")

# Calculate per-neuron statistics
neuron_means = np.mean(activity_matrix, axis=0)
neuron_maxes = np.max(activity_matrix, axis=0)
neuron_stds = np.std(activity_matrix, axis=0)
neuron_active_fraction = np.mean(activity_matrix > 0.1, axis=0)  # Fraction of time active

print(f"Most active neuron (mean): {np.max(neuron_means):.4f}")
print(f"Least active neuron (mean): {np.min(neuron_means):.4f}")
print(f"Most active neuron (max): {np.max(neuron_maxes):.4f}")
print(f"Most active neuron (% time active): {np.max(neuron_active_fraction)*100:.1f}%")

# Silent neurons
silent_neurons = np.sum(neuron_maxes == 0)
print(f"Completely silent neurons: {silent_neurons}/{len(neuron_means)} ({silent_neurons/len(neuron_means)*100:.1f}%)")

# Very active neurons
very_active = np.sum(neuron_active_fraction > 0.1)  # Active >10% of time
print(f"Very active neurons (>10% time): {very_active}/{len(neuron_means)} ({very_active/len(neuron_means)*100:.1f}%)")

# =============================================================================
# 3. TOP 100 MOST ACTIVE NEURONS BY FOV
# =============================================================================
print("\n--- Top 100 Most Active Neurons by FOV ---")

# Get indices of top 100 most active neurons (by mean activity)
top_100_indices = np.argsort(neuron_means)[-100:][::-1]  # Top 100, sorted descending

# Map neuron indices to FOVs
def get_neuron_fov_mapping(session):
    """Create mapping from neuron index to FOV name"""
    neuron_to_fov = {}
    neuron_idx = 0
    
    for fov_name, fov in session.fovs.items():
        for i in range(fov.n_neurons):
            neuron_to_fov[neuron_idx] = fov_name
            neuron_idx += 1
    
    return neuron_to_fov

neuron_to_fov = get_neuron_fov_mapping(session)

# Count top neurons by FOV
top_100_fovs = [neuron_to_fov[idx] for idx in top_100_indices]
fov_counts = pd.Series(top_100_fovs).value_counts()

print("Top 100 neurons by FOV:")
for fov, count in fov_counts.items():
    total_in_fov = session.fovs[fov].n_neurons
    percentage = count/total_in_fov*100
    print(f"  {fov}: {count}/100 ({count}% of top 100, {percentage:.1f}% of {fov})")

# =============================================================================
# 4. TEMPORAL STATISTICS
# =============================================================================
print("\n--- Temporal Statistics ---")

# Population activity over time
pop_activity = np.sum(activity_matrix > 0.1, axis=1)  # Number of active neurons per timepoint
pop_mean_activity = np.mean(activity_matrix, axis=1)  # Mean activity per timepoint

print(f"Max simultaneous active neurons: {np.max(pop_activity)}/{activity_matrix.shape[1]} ({np.max(pop_activity)/activity_matrix.shape[1]*100:.1f}%)")
print(f"Min simultaneous active neurons: {np.min(pop_activity)}")
print(f"Mean simultaneous active neurons: {np.mean(pop_activity):.1f}")

# Most/least active time periods
most_active_time = timestamps[np.argmax(pop_activity)]
least_active_time = timestamps[np.argmin(pop_activity)]
print(f"Most active timepoint: {most_active_time:.1f}s ({most_active_time/60:.1f}min)")
print(f"Least active timepoint: {least_active_time:.1f}s ({least_active_time/60:.1f}min)")

# =============================================================================
# 5. FOV-LEVEL COMPARISONS
# =============================================================================
print("\n--- FOV-Level Comparisons ---")

fov_stats = {}
neuron_idx = 0

for fov_name, fov in session.fovs.items():
    fov_activity = activity_matrix[:, neuron_idx:neuron_idx+fov.n_neurons]
    
    fov_stats[fov_name] = {
        'n_neurons': fov.n_neurons,
        'mean_activity': np.mean(fov_activity),
        'max_activity': np.max(fov_activity),
        'active_fraction': np.mean(fov_activity > 0.1),
        'silent_neurons': np.sum(np.max(fov_activity, axis=0) == 0),
        'very_active_neurons': np.sum(np.mean(fov_activity > 0.1, axis=0) > 0.1)
    }
    
    neuron_idx += fov.n_neurons

# Create DataFrame for easier viewing
fov_df = pd.DataFrame(fov_stats).T
fov_df['silent_pct'] = fov_df['silent_neurons'] / fov_df['n_neurons'] * 100
fov_df['very_active_pct'] = fov_df['very_active_neurons'] / fov_df['n_neurons'] * 100

print("FOV Statistics:")
print(fov_df.round(4))

# =============================================================================
# 6. VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Activity value distribution (log scale)
axes[0,0].hist(activity_matrix.flatten()[::1000], bins=100, alpha=0.7, edgecolor='none')  # Sample for speed
axes[0,0].set_xlabel('Activity Value')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('Distribution of Activity Values')
axes[0,0].set_yscale('log')

# 2. Neuron activity distribution
axes[0,1].hist(neuron_means, bins=50, alpha=0.7, edgecolor='none')
axes[0,1].set_xlabel('Mean Activity per Neuron')
axes[0,1].set_ylabel('Count')
axes[0,1].set_title('Distribution of Neuron Mean Activities')
axes[0,1].set_yscale('log')

# 3. Top 100 neurons by FOV
axes[0,2].bar(range(len(fov_counts)), fov_counts.values, alpha=0.7)
axes[0,2].set_xlabel('FOV')
axes[0,2].set_ylabel('Number of Top 100 Neurons')
axes[0,2].set_title('Top 100 Most Active Neurons by FOV')
axes[0,2].set_xticks(range(len(fov_counts)))
axes[0,2].set_xticklabels(fov_counts.index, rotation=45)

# 4. Population activity over time (downsampled)
downsample = 100
axes[1,0].plot(timestamps[::downsample]/60, pop_activity[::downsample], alpha=0.7, linewidth=0.8)
axes[1,0].set_xlabel('Time (minutes)')
axes[1,0].set_ylabel('Active Neurons')
axes[1,0].set_title('Population Activity Over Time')

# 5. FOV activity comparison
axes[1,1].bar(range(len(fov_df)), fov_df['mean_activity'], alpha=0.7)
axes[1,1].set_xlabel('FOV')
axes[1,1].set_ylabel('Mean Activity')
axes[1,1].set_title('Mean Activity by FOV')
axes[1,1].set_xticks(range(len(fov_df)))
axes[1,1].set_xticklabels(fov_df.index, rotation=45)

# 6. FOV neuron composition
width = 0.35
x = np.arange(len(fov_df))
axes[1,2].bar(x - width/2, fov_df['silent_pct'], width, label='Silent (%)', alpha=0.7)
axes[1,2].bar(x + width/2, fov_df['very_active_pct'], width, label='Very Active (%)', alpha=0.7)
axes[1,2].set_xlabel('FOV')
axes[1,2].set_ylabel('Percentage of Neurons')
axes[1,2].set_title('Neuron Activity Composition by FOV')
axes[1,2].set_xticks(x)
axes[1,2].set_xticklabels(fov_df.index, rotation=45)
axes[1,2].legend()

plt.tight_layout()
plt.show()

# =============================================================================
# 7. FINAL SUMMARY
# =============================================================================
print(f"\n=== FINAL SUMMARY ===")
print(f"Dataset: {activity_matrix.shape[0]} timepoints × {activity_matrix.shape[1]} neurons")
print(f"Duration: {timestamps[-1]/60:.1f} minutes")
print(f"Most active FOV: {fov_df['mean_activity'].idxmax()} (mean activity: {fov_df['mean_activity'].max():.4f})")
print(f"Most represented in top 100: {fov_counts.index[0]} ({fov_counts.iloc[0]} neurons)")
print(f"Sparsity: {zero_fraction*100:.1f}% exact zeros, {(1-np.mean(activity_matrix > 0.1))*100:.1f}% below threshold")
print(f"Dynamic range: {np.max(activity_matrix)/np.mean(activity_matrix[activity_matrix > 0]):.1f}x above mean non-zero")

# %% [markdown]
# ## Treescope

# %%
import treescope
treescope.basic_interactive_setup(autovisualize_arrays=True)

# %%
activity_matrix.T

# %%
activity_matrix[500:550, 5000:5010].T

# %%
parent_dir = os.path.dirname(os.getcwd())
os.listdir(parent_dir)
pre_dir = os.path.join(parent_dir, '2_PREPROCESSING')
if pre_dir not in sys.path:
    sys.path.append(pre_dir)
    
from activity_preprocessor import CalciumDataPreprocessor

# from utils.load_meso_session import MesoscopeSession

# %% [markdown]
# ## Neuroglancer

# %%
preprocessor = CalciumDataPreprocessor(neucoeff=0.7, temporal_smoothing=True)
df_f_activity = preprocessor.preprocess_session(session, '../DATA/session_000.h5')

# %%
df_f_activity['processed_data'].shape

# %%
import neuroglancer as ng

dimensions = ng.CoordinateSpace(
   names=['time', 'neurons',],
   units='',
   scales=[1, 1, 1],
)
viewer = ng.Viewer()

with viewer.txn() as s:
  s.dimensions = dimensions
  s.layers['raw'] = ng.ImageLayer(
      source=ng.LocalVolume(df_f_activity['processed_data'], dimensions))
  s.layout = 'xy'
viewer

# USE THIS SHADER

# #uicontrol int channel slider(min=0, max=4)
# #uicontrol vec3 color color(default="red")
# #uicontrol float brightness slider(min=-1, max=1)
# #uicontrol float contrast slider(min=-3, max=3, step=0.01)
# void main() {
#   emitRGB(color *
#           (toNormalized(getDataValue(channel)) + brightness) *
#           exp(contrast));
# }

# %%
fig = plt.figure(figsize=(12, 12))
plt.title(f'traces for {session.eid} condition')
im = plt.imshow(df_f_activity['processed_data'].T[:940,:], aspect="auto")
plt.xlabel('timestep')
plt.ylabel('neuron')
cbar = fig.colorbar(im)
cbar.set_label("normalized activity (df/f)")
plt.show();

# %% [markdown]
# ## Rastermap

# %%
df_f_activity['processed_data'].shape

# %%
from rastermap import Rastermap

model = Rastermap(n_clusters=100, # number of clusters to compute
                  n_PCs=200, # number of PCs
                  locality=0.5, # locality in sorting is low here to get more global sorting (this is a value from 0-1)
                  time_lag_window=5, # use future timepoints to compute correlation
                  grid_upsample=10, # default value, 10 is good for large recordings
                  ).fit(df_f_activity['processed_data'].T)

# %%
y = model.embedding # neurons x 1
isort = model.isort

# visualize binning over neurons
X_embedding = model.X_embedding

# plot

fig = plt.figure(figsize=(12,5), dpi=200)
ax = fig.add_subplot(111)
ax.imshow(X_embedding, vmin=0, vmax=0.8, cmap="gray_r", aspect="auto")

# %%
preprocessed_session = MesoscopeSession.from_preprocessed('../DATA/session_000.h5')

# %%
import matplotlib.patches as patches

def plot_behavioral_events(ax, stim_times, feedback_times, timestamps, xmin_idx, xmax_idx):
    """Plot stimulus onset times and feedback times as vertical lines"""
    # Convert time indices to actual time
    xmin_time = timestamps[xmin_idx] if xmin_idx < len(timestamps) else timestamps[-1]
    xmax_time = timestamps[xmax_idx] if xmax_idx < len(timestamps) else timestamps[-1]
    
    # Get number of neurons for y-axis scaling
    nn = X_embedding.shape[0]
    
    # Filter stimulus times to the visible time window
    stim_times_filtered = stim_times[~np.isnan(stim_times)]  # Remove NaN values
    stim_times_visible = stim_times_filtered[
        (stim_times_filtered >= xmin_time) & (stim_times_filtered < xmax_time)
    ]
    
    # Filter feedback times to the visible time window
    feedback_times_filtered = feedback_times[~np.isnan(feedback_times)]  # Remove NaN values
    feedback_times_visible = feedback_times_filtered[
        (feedback_times_filtered >= xmin_time) & (feedback_times_filtered < xmax_time)
    ]
    
    # Convert times to sample indices for plotting
    sampling_rate = 1 / np.mean(np.diff(timestamps))
    
    # Plot stimulus onsets
    for stim_time in stim_times_visible:
        stim_idx = int((stim_time - xmin_time) * sampling_rate)
        if 0 <= stim_idx < (xmax_idx - xmin_idx):
            ax.axvline(x=stim_idx, color='red', linewidth=1.5, alpha=0.8, 
                      label='Stimulus Onset' if stim_time == stim_times_visible[0] else "")
    
    # Plot feedback times
    for feedback_time in feedback_times_visible:
        feedback_idx = int((feedback_time - xmin_time) * sampling_rate)
        if 0 <= feedback_idx < (xmax_idx - xmin_idx):
            ax.axvline(x=feedback_idx, color='blue', linewidth=1.5, alpha=0.8,
                      label='Feedback' if feedback_time == feedback_times_visible[0] else "")
    
    ax.set_ylim([0, nn])
    ax.invert_yaxis()

def plot_wheel_position(ax, wheel_pos, timestamps, xmin_idx, xmax_idx):
    """Plot wheel position trace"""
    # Get wheel position for the time window
    wheel_window = wheel_pos[xmin_idx:xmax_idx]
    time_indices = np.arange(len(wheel_window))
    
    ax.plot(time_indices, wheel_window, color='green', linewidth=1, alpha=0.8)
    ax.set_ylabel('Wheel Position', color='green')
    ax.tick_params(axis='y', labelcolor='green')
    ax.set_xlim(0, len(wheel_window))

# Define time window for visualization (in sample indices)
xmin_idx = 0
xmax_idx = 5000

# Create figure with grid for easy plotting
fig = plt.figure(figsize=(15, 8), dpi=200)
grid = plt.GridSpec(12, 20, figure=fig, wspace=0.1, hspace=0.4)

# Plot rastermap with behavioral events
ax_raster = plt.subplot(grid[2:9, :-1])
ax_raster.imshow(X_embedding[:, xmin_idx:xmax_idx], cmap="gray_r", vmin=0, vmax=0.8, aspect="auto")
ax_raster.set_ylabel("Neurons (sorted)")
ax_raster.set_title("Rastermap with Behavioral Events")

# Add behavioral events
plot_behavioral_events(ax_raster, session.stimOn_times, session.feedback_times, timestamps, xmin_idx, xmax_idx)

# Add colorbar for neuron ordering
ax_cbar = plt.subplot(grid[2:9, -1])
ax_cbar.imshow(np.arange(0, X_embedding.shape[0])[:, np.newaxis], cmap="gist_ncar", aspect="auto")
ax_cbar.axis("off")

# Plot wheel position below the rastermap
ax_wheel = plt.subplot(grid[10:12, :-1])
plot_wheel_position(ax_wheel, session.wheel_position, timestamps, xmin_idx, xmax_idx)
ax_wheel.set_xlabel("Time (samples)")

# Add legend
ax_raster.legend(loc='upper right', fontsize=8)

# plt.tight_layout()
plt.show()


