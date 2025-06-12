# IBL Neural Data Analysis: Spike Matrix and Raster Plots
# ==============================================
#
# This notebook demonstrates how to:
# 1. Load neural data from the International Brain Laboratory (IBL) dataset
# 2. Create a cells × time matrix of spike counts
# 3. Generate raster plots to visualize neural activity

import numpy as np
import matplotlib.pyplot as plt
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader, EphysSessionLoader
from iblutil.numerical import bincount2D
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Connect to ONE
print("Connecting to ONE API...")
one = ONE(base_url='https://openalyx.internationalbrainlab.org')

# Define parameters
time_bin_size = 0.05  # 50 ms bins for spike count matrix
depth_bin_size = 10   # 10 µm bins for depth raster

# %% [markdown]
# ## Part 1: Load data from a single probe insertion

# Create a SpikeSortingLoader for a specific probe insertion
# This example uses a specific pid, but you can search for others
pid = 'da8dfec1-d265-44e8-84ce-6ae9c109b8bd'  # Example from documentation
print(f"Loading data for probe insertion: {pid}")

# Load spike sorting data
ssl = SpikeSortingLoader(pid=pid, one=one)
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)

print(f"Loaded {len(spikes['times'])} spikes and {len(np.unique(spikes['clusters']))} clusters")

# %% [markdown]
# ## Part 2: Filter for good quality units (clusters)

# Find the clusters labeled as "good" (label == 1)
good_clusters_idx = clusters['label'] == 1
good_cluster_ids = clusters['cluster_id'][good_clusters_idx]

print(f"Number of good clusters: {len(good_cluster_ids)} out of {len(clusters['cluster_id'])}")

# Get basic statistics about these clusters
if 'firing_rate' in clusters:
    mean_fr = np.mean(clusters['firing_rate'][good_clusters_idx])
    min_fr = np.min(clusters['firing_rate'][good_clusters_idx])
    max_fr = np.max(clusters['firing_rate'][good_clusters_idx])
    print(f"Firing rate statistics for good clusters:")
    print(f"  Mean: {mean_fr:.2f} Hz")
    print(f"  Min: {min_fr:.2f} Hz")
    print(f"  Max: {max_fr:.2f} Hz")

# %% [markdown]
# ## Part 3: Create a spike count matrix (cells × time)

# Set up time bins for the recording
t_start = np.min(spikes['times'])
t_end = np.max(spikes['times'])
time_bins = np.arange(t_start, t_end + time_bin_size, time_bin_size)
n_time_bins = len(time_bins) - 1

print(f"Time range: {t_start:.2f}s to {t_end:.2f}s")
print(f"Number of time bins: {n_time_bins}")

# Initialize the spike count matrix (rows = cells, columns = time bins)
n_cells = len(good_cluster_ids)
spike_matrix = np.zeros((n_cells, n_time_bins))

# For each good cluster, bin its spikes into the time bins
print("Creating spike count matrix...")
for i, cluster_id in enumerate(good_cluster_ids):
    # Get spike times for this cluster
    cluster_spike_times = spikes['times'][spikes['clusters'] == cluster_id]
    
    # Bin the spike times
    spike_counts, _ = np.histogram(cluster_spike_times, bins=time_bins)
    
    # Add to matrix
    spike_matrix[i, :] = spike_counts

print(f"Created spike matrix with shape: {spike_matrix.shape}")

# Convert to firing rates (spikes/second)
firing_rate_matrix = spike_matrix / time_bin_size

# %% [markdown]
# ## Part 4: Visualize the spike count matrix

# Create a figure to visualize a portion of the spike matrix
plt.figure(figsize=(14, 8))
# Only show a subset if the matrix is very large
max_time_bins_to_show = min(500, n_time_bins)
max_cells_to_show = min(100, n_cells)

# Create a custom colormap from white to black
cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'black'])

# Plot the spike matrix
plt.imshow(spike_matrix[:max_cells_to_show, :max_time_bins_to_show], 
           aspect='auto', 
           cmap=cmap,
           interpolation='none')

plt.colorbar(label='Spike Count')
plt.xlabel(f'Time Bins ({time_bin_size*1000:.0f} ms each)')
plt.ylabel('Neuron (Cluster) Index')
plt.title('Spike Count Matrix (Cells × Time)')
plt.tight_layout()

# %% [markdown]
# ## Part 5: Create a depth raster plot (similar to the IBL visualization)

print("Creating depth raster plot...")
# Remove any NaN values
keep_idx = np.bitwise_and(~np.isnan(spikes['times']), ~np.isnan(spikes['depths']))

# Create a 2D histogram of spikes across time and depth
fr, time_edges, depth_edges = bincount2D(
    spikes['times'][keep_idx], 
    spikes['depths'][keep_idx], 
    time_bin_size, 
    depth_bin_size
)

# Convert to firing rate
fr = fr / time_bin_size

# Create a figure
fig, ax = plt.subplots(figsize=(14, 10))
extent = [time_edges[0], time_edges[-1], depth_edges[0], depth_edges[-1]]

# Plot the 2D histogram
img = ax.imshow(fr.T, 
                aspect='auto', 
                origin='lower', 
                cmap='Greys',
                extent=extent)
plt.colorbar(img, ax=ax, label='Firing Rate (Hz)')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Depth (µm)')
ax.set_title(f'Neural Activity Raster Plot\n{len(spikes["times"][keep_idx])} spikes across all depths')

# %% [markdown]
# ## Part 6: Use the built-in IBL raster plotting function

# The IBL toolkit provides a built-in function for creating raster plots
print("Creating IBL raster plot...")
fig, axs = ssl.raster(spikes, channels)

# %% [markdown]
# ## Part 7: Create raster plots for individual neurons

# Select a few example neurons to visualize
num_examples = min(4, len(good_cluster_ids))
example_indices = np.random.choice(len(good_cluster_ids), num_examples, replace=False)
example_cluster_ids = good_cluster_ids[example_indices]

# Create a figure with subplots for each example neuron
fig, axs = plt.subplots(num_examples, 1, figsize=(14, 3*num_examples), sharex=True)

for i, cluster_id in enumerate(example_cluster_ids):
    # Get spike times for this cluster
    cluster_spike_times = spikes['times'][spikes['clusters'] == cluster_id]
    
    # Plot raster
    axs[i].eventplot(cluster_spike_times, lineoffsets=0.5, linelengths=1, linewidths=0.5, color='black')
    axs[i].set_ylabel(f'Cluster {cluster_id}')
    
    # Add text with cluster info
    idx = np.where(clusters['cluster_id'] == cluster_id)[0][0]
    if 'acronym' in clusters:
        region = clusters['acronym'][idx]
    else:
        region = 'Unknown'
    
    if 'firing_rate' in clusters:
        fr = clusters['firing_rate'][idx]
        axs[i].text(0.98, 0.95, f"Region: {region}, FR: {fr:.2f} Hz", 
                 transform=axs[i].transAxes, ha='right', va='top')
    else:
        axs[i].text(0.98, 0.95, f"Region: {region}", 
                 transform=axs[i].transAxes, ha='right', va='top')

plt.xlabel('Time (seconds)')
plt.tight_layout()

# %% [markdown]
# ## Part 8: Load data from an entire session (multiple probes)

# Get a session ID from the probe insertion
eid, pname = one.pid2eid(pid)
print(f"Loading data for session: {eid}")

# Use EphysSessionLoader to load data from all probes in the session
ephys_loader = EphysSessionLoader(eid=eid, one=one)
ephys_loader.load_spike_sorting()

# Get a list of all probes in this session
probes = list(ephys_loader.ephys.keys())
print(f"Probes in this session: {probes}")

# Create a dictionary to hold information about cells across all probes
all_cells_info = []

for probe in probes:
    probe_clusters = ephys_loader.ephys[probe]['clusters']
    probe_spikes = ephys_loader.ephys[probe]['spikes']
    
    # Filter for good clusters
    good_clusters_idx = probe_clusters['label'] == 1
    good_cluster_ids = probe_clusters['cluster_id'][good_clusters_idx]
    
    for j, cluster_id in enumerate(good_cluster_ids):
        # Get basic info about this cluster
        idx = np.where(probe_clusters['cluster_id'] == cluster_id)[0][0]
        
        cell_info = {
            'probe': probe,
            'cluster_id': cluster_id,
            'depth': probe_clusters['depths'][idx] if 'depths' in probe_clusters else np.nan
        }
        
        # Add region information if available
        if 'acronym' in probe_clusters:
            cell_info['region'] = probe_clusters['acronym'][idx]
        
        # Add firing rate information if available
        if 'firing_rate' in probe_clusters:
            cell_info['firing_rate'] = probe_clusters['firing_rate'][idx]
        
        all_cells_info.append(cell_info)

# Convert to DataFrame for easy analysis
cells_df = pd.DataFrame(all_cells_info)

print(f"Total number of good cells across all probes: {len(cells_df)}")
if 'region' in cells_df.columns:
    print("Cells per brain region:")
    print(cells_df['region'].value_counts().head(10))  # Top 10 regions

# %% [markdown]
# ## Part 9: Create full session spike matrix

# Find common time range across all probes
t_start_session = float('inf')
t_end_session = 0

for probe in probes:
    probe_spikes = ephys_loader.ephys[probe]['spikes']
    if len(probe_spikes['times']) > 0:  # Skip empty probes
        t_start_session = min(t_start_session, np.min(probe_spikes['times']))
        t_end_session = max(t_end_session, np.max(probe_spikes['times']))

# Create time bins for the entire session
session_time_bins = np.arange(t_start_session, t_end_session + time_bin_size, time_bin_size)
n_session_time_bins = len(session_time_bins) - 1

print(f"Session time range: {t_start_session:.2f}s to {t_end_session:.2f}s")
print(f"Number of time bins: {n_session_time_bins}")

# Initialize the full session spike matrix
n_total_cells = len(cells_df)
session_spike_matrix = np.zeros((n_total_cells, n_session_time_bins))

# Fill the matrix
print("Creating full session spike matrix...")
for i, (_, cell) in enumerate(cells_df.iterrows()):
    probe = cell['probe']
    cluster_id = cell['cluster_id']
    
    # Get spike times for this cluster
    probe_spikes = ephys_loader.ephys[probe]['spikes']
    cell_spike_times = probe_spikes['times'][probe_spikes['clusters'] == cluster_id]
    
    # Bin the spike times
    spike_counts, _ = np.histogram(cell_spike_times, bins=session_time_bins)
    
    # Add to matrix
    session_spike_matrix[i, :] = spike_counts

print(f"Created full session spike matrix with shape: {session_spike_matrix.shape}")

# %% [markdown]
# ## Part 10: Visualize the full session spike matrix

# Create a figure to visualize a portion of the full session spike matrix
plt.figure(figsize=(14, 8))
# Only show a subset if the matrix is very large
max_time_bins_to_show = min(500, n_session_time_bins)
max_cells_to_show = min(100, n_total_cells)

# Create a custom colormap from white to black
cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'black'])

# Plot the spike matrix
plt.imshow(session_spike_matrix[:max_cells_to_show, :max_time_bins_to_show], 
           aspect='auto', 
           cmap=cmap,
           interpolation='none')

plt.colorbar(label='Spike Count')
plt.xlabel(f'Time Bins ({time_bin_size*1000:.0f} ms each)')
plt.ylabel('Neuron (Cluster) Index')
plt.title('Full Session Spike Count Matrix (Cells × Time)')
plt.tight_layout()

# %% [markdown]
# ## Part 11: Save the spike matrix

# Save the matrices for future use
print("Saving spike matrices...")
np.save('single_probe_spike_matrix.npy', spike_matrix)
np.save('full_session_spike_matrix.npy', session_spike_matrix)

# Also save metadata about the cells
cells_df.to_csv('cells_metadata.csv', index=False)

print("Analysis complete!")