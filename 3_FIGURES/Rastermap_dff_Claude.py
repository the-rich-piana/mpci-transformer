# %%
import sys
sys.executable

# %%
# from one.api import ONE
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import seaborn as sns
from rastermap import Rastermap
    

import h5py
from utils.load_meso_session import MesoscopeSession

from one.api import ONE
one = ONE()

SESSION_INDEX = 0


# %%
session = MesoscopeSession.from_eid(one, "5ea6bb9b-6163-4e8a-816b-efe7002666b0")
activity, timestamps = session.get_dff_activity_matrix()

# %%

model = Rastermap(n_clusters=100, # number of clusters to compute
                  n_PCs=200, # number of PCs
                  locality=0.1, # locality in sorting is low here to get more global sorting (this is a value from 0-1)
                  time_lag_window=5, # use future timepoints to compute correlation
                  grid_upsample=10, # default value, 10 is good for large recordings
                  ).fit(activity.T)

# %%
y = model.embedding
isort = model.isort
X_embedding = model.X_embedding

fig = plt.figure(figsize=(12,5), dpi=200)
ax = fig.add_subplot(111)
ax.imshow(X_embedding, vmin=-0.1, vmax=2.6, cmap="viridis", aspect="auto")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 10

# Your data
num_neurons = activity.shape[1]
embedding_min = -0.1
embedding_max = 2.6

# Get dimensions
_, n_timesteps = X_embedding.shape
# total_time_minutes = n_timesteps / 4.9
total_time_minutes = session.duration_seconds / 60

# Calculate 5-minute scale bar width in axes coordinates
minutes_per_timestep = total_time_minutes / n_timesteps
timesteps_per_5min = 5 / minutes_per_timestep
scale_bar_width = timesteps_per_5min / n_timesteps  # Convert to fraction of plot width

# Create figure with standard subplot
fig = plt.figure(figsize=(12, 6), dpi=200)
ax = plt.subplot(111)

# Main raster plot
im = ax.imshow(X_embedding, vmin=embedding_min, vmax=embedding_max, cmap="viridis", aspect="auto")

# Remove default ticks and labels
ax.set_xticks([])
ax.set_yticks([])

ax.text(-0.02, 0.5, f'{num_neurons:,} neurons', transform=ax.transAxes, 
        rotation=90, ha='right', va='center', fontweight='bold', fontsize=12)

# Add 5-minute scale bar using data coordinates (perfectly aligned)
# Note: y-axis is inverted (ylim: 516.5 to -0.5), so +540 puts it below the image
scale_bar_start_x = 0  # Left edge of image data
scale_bar_end_x = timesteps_per_5min  # 5 minutes worth of timesteps
scale_bar_y = 540  # Below the image (image goes from 0 to 516, so +540 is below)

ax.plot([scale_bar_start_x, scale_bar_end_x], [scale_bar_y, scale_bar_y], 'k-', 
        linewidth=6, clip_on=False)
ax.text(scale_bar_end_x/2, scale_bar_y + 15, '5 min', 
        ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add timestep count in bottom center
ax.text(0.5, -0.05, f'{n_timesteps:,} time steps', transform=ax.transAxes, 
        ha='center', va='top', fontweight='bold', fontsize=12)

# Create small colorbar on bottom right using manual positioning
cbar_width = 0.2
cbar_height = 0.02
# Position it on bottom right, aligned with image right edge
ax_pos = ax.get_position()
cbar_left = ax_pos.x1 - cbar_width
cbar_bottom = ax_pos.y0 - 0.08
cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

cbar = plt.colorbar(im, cax=cbar_ax, ticks=[embedding_min, embedding_max], orientation='horizontal')
cbar.outline.set_visible(False)
cbar.ax.tick_params(labelsize=9, width=0, length=0)
cbar.ax.set_xlabel('ΔF/F', fontweight='bold', fontsize=10)
cbar.ax.set_xticklabels(['0.', '.8'])

# Remove spines and clean up
for spine in ax.spines.values():
    spine.set_visible(False)

# Use tight_layout now that we're back to standard subplot
plt.tight_layout()
plt.savefig("../FIGURES/Activity_Traces_Whole.png")

# %%
# Debug information to understand the plot structure
print("=== DEBUG INFORMATION ===")
print(f"X_embedding shape: {X_embedding.shape}")
print(f"Activity shape: {activity.shape}")
print(f"Number of neurons (original): {num_neurons:,}")
print(f"Number of clusters (rastermap): {X_embedding.shape[0]}")
print(f"Number of timesteps: {n_timesteps:,}")
print(f"Total time (minutes): {total_time_minutes:.2f}")
print(f"Minutes per timestep: {minutes_per_timestep:.6f}")
print(f"Timesteps per 5min: {timesteps_per_5min:.2f}")
print(f"Scale bar width (fraction): {scale_bar_width:.6f}")
print(f"Session duration (seconds): {session.duration_seconds}")
print()
print("Timestamps info:")
print(f"Timestamps shape: {timestamps.shape}")
print(f"First timestamp: {timestamps[0]:.3f}")
print(f"Last timestamp: {timestamps[-1]:.3f}")
print(f"Timestamp range: {timestamps[-1] - timestamps[0]:.3f} seconds")
print()
print("Current axes info:")
print(f"Axes position: {ax.get_position()}")
print(f"Axes xlim: {ax.get_xlim()}")
print(f"Axes ylim: {ax.get_ylim()}")

plt.show()