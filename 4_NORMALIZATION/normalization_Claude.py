# %%
from utils.load_meso_session import FOVData, _load_fov_data
from one.api import ONE
import numpy as np
one = ONE()
import seaborn as sns
import matplotlib.pyplot as plt
from rastermap import Rastermap

FOV_select = "alf/FOV_03"

FOV_dff: FOVData = _load_fov_data(one=one, eid="5ea6bb9b-6163-4e8a-816b-efe7002666b0" ,raw_activity=True, collection=FOV_select)
FOV_dec: FOVData = _load_fov_data(one=one, eid="5ea6bb9b-6163-4e8a-816b-efe7002666b0" ,raw_activity=False, collection=FOV_select)

# %%
def robust_normalize(data):
    """Apply robust per-neuron normalization"""
    # Use 0.5th and 99.5th percentiles for extremely robust scaling
    q005 = np.percentile(data, 0.5, axis=0, keepdims=True)
    q995 = np.percentile(data, 99.5, axis=0, keepdims=True)
    
    # Ensure non-zero range
    data_range = q995 - q005
    data_range = np.maximum(data_range, 0.1)  # Minimum range of 0.1
    
    # Scale to [0, 1] range
    normalized = (data - q005) / data_range
    
    # Clip extreme outliers
    normalized = np.clip(normalized, -0.25, 1.5)
    
    return normalized

# %%
activity, timestamps, mask = FOV_dff.get_fov_dff_activity()
activity_robust = robust_normalize(activity)
fluor, _, _ = FOV_dff.get_fov_fluorescence()
dec, _, _ = FOV_dec.get_neuron_activity()

print(f"Data shapes - Fluor: {fluor.shape}, ΔF/F: {activity.shape}, Deconv: {dec.shape}, Robust: {activity_robust.shape}")


# %%

# Select a representative neuron and time window for visualization
neuron_idx = 2  # Same neuron used in previous plots
time_start = 5000
time_end = 9000  # Show 2000 time points for better detail

# Extract time series for selected neuron and window
time_points = timestamps[time_start:time_end]
fluor_trace = fluor[time_start:time_end, neuron_idx]
dff_trace = activity[time_start:time_end, neuron_idx]
deconv_trace = dec[time_start:time_end, neuron_idx]
robust_trace = activity_robust[time_start:time_end, neuron_idx]

# Create stacked plot
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

# Set seaborn style
sns.set_style("darkgrid")
palette = sns.color_palette("Set1", 4)

# Plot 1: Raw Fluorescence
axes[0].plot(time_points, fluor_trace, color=palette[0], linewidth=1.2)
axes[0].set_ylabel('Raw Fluorescence \n (AU)\n', fontsize=12)
axes[0].set_title(f'Calcium Imaging Traces - Neuron {neuron_idx} - {FOV_select}', fontsize=14, pad=20)
# axes[0].grid(False, alpha=0.3)

# Plot 2: ΔF/F
axes[1].plot(time_points, dff_trace, color=palette[1], linewidth=1.2)
axes[1].set_ylabel('ΔF/F\n', fontsize=12)
# axes[1].grid(False, alpha=0.3)

# Plot 3: Deconvolved
axes[2].plot(time_points, deconv_trace, color=palette[2], linewidth=1.)
axes[2].set_ylabel('Deconvolved\n (AU)', fontsize=12)

# Plot 4: Model Input
axes[3].plot(time_points, robust_trace, color=palette[3], linewidth=1.)
axes[3].set_ylabel('Modul Input \n (ΔF/F robust)', fontsize=12)
axes[3].set_xlabel('Time (seconds)', fontsize=12)
# axes[2].grid(False, alpha=0.3)

# Improve layout
plt.tight_layout()

stats_text = f"""
Data Statistics:
• Raw Fluor mean: {fluor_trace.mean():.1f} ± {fluor_trace.std():.1f}
• ΔF/F mean: {dff_trace.mean():.2f} ± {dff_trace.std():.2f}  
• Deconv mean: {deconv_trace.mean():.2f} ± {deconv_trace.std():.2f}
• Model input mean: {robust_trace.mean():.2f} ± {robust_trace.std():.2f}
"""

fig.text(0.8, -0.10, stats_text, fontsize=10, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.show()

# %%
activity.mean()

# %%
model = Rastermap(n_clusters=100, # number of clusters to compute
                  n_PCs=200, # number of PCs
                  locality=0.5, # locality in sorting is low here to get more global sorting (this is a value from 0-1)
                  time_lag_window=20, # use future timepoints to compute correlation
                  grid_upsample=10, # default value, 10 is good for large recordings
                  ).fit(activity[:, :].T)
y = model.embedding
isort = model.isort
X_embedding_activity = model.X_embedding
print(f"min {X_embedding_activity.min()} max {X_embedding_activity.max()}")

model = Rastermap(n_clusters=100, # number of clusters to compute
                  n_PCs=200, # number of PCs
                  locality=0.5, # locality in sorting is low here to get more global sorting (this is a value from 0-1)
                  time_lag_window=20, # use future timepoints to compute correlation
                  grid_upsample=10, # default value, 10 is good for large recordings
                  ).fit(activity_robust[:, :].T)
y = model.embedding
isort = model.isort
X_embedding_robust = model.X_embedding
print(f"min {X_embedding_robust.min()} max {X_embedding_robust.max()}")

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=200)

# Plot original ΔF/F rastermap
im1 = axes[0].imshow(X_embedding_activity, vmin=-.25, vmax=3, cmap="viridis", aspect="auto")
axes[0].set_title('Rastermap: Original ΔF/F', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Neurons (sorted)', fontsize=10)
axes[0].grid(False)

# Plot robust normalized rastermap
im2 = axes[1].imshow(X_embedding_robust, vmin=-.25, vmax=3, cmap="viridis", aspect="auto")
axes[1].set_title('Rastermap: Robust Normalized ΔF/F', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Time', fontsize=10)
axes[1].set_ylabel('Neurons (sorted)', fontsize=10)
axes[1].grid(False)

# Adjust layout before adding colorbar
plt.subplots_adjust(right=0.85)

# Add colorbar
cbar = plt.colorbar(im2, ax=axes, shrink=0.8, label='Activity')

plt.show()

