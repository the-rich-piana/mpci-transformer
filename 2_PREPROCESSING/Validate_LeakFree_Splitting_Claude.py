# %% [markdown]
# # Comprehensive Validation of Leak-Free TimeSeriesSplitter
# 
# This notebook validates that our new TimeSeriesSplitter creates proper train/val/test splits 
# with appropriate data flow allowances using the visualization tools from analysis.

# %%
# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from rastermap import Rastermap


parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.data_splitter import TimeSeriesSplitter
from utils.load_meso_session import MesoscopeSession
from Activity_Data_Loader import Dataset_Activity

# %%
print("=== LOADING SESSION DATAK ===")
session_path = "../DATA/session_5ea6bb9b-6163-4e8a-816b-efe7002666b0.h5"

# Load via MesoscopeSession for trial data access
preprocessed_session = MesoscopeSession.from_preprocessed(session_path)
activity, timestamps = preprocessed_session.get_preprocessed_data()

# Load covariate matrix directly from HDF5
with h5py.File(session_path, 'r') as f:
    covariate_matrix = f['covariate_matrix'][:]
    feature_names = [name.decode('utf-8') for name in f['covariate_metadata']['feature_names'][:]]
    covariate_metadata = f['covariate_metadata']
    

print(f"Activity shape: {activity.shape}")
print(f"Covariate shape: {covariate_matrix.shape}")  
print(f"Covariate features: {feature_names}")
# %%
# %%
# Create stimulus-based splits and TimeSeriesSplitter
print("\n=== CREATING STIMULUS-BASED SPLITS ===")

# Map stimulus feature names to stimulus type IDs for clarity
stimulus_feature_mapping = {
    'stimulus_catch_trial': 0,
    'stimulus_left_100pct': 1, 
    'stimulus_left_25pct': 2,
    'stimulus_left_12.5pct': 3,
    'stimulus_left_6.25pct': 4,
    'stimulus_right_100pct': 5,
    'stimulus_right_25pct': 6, 
    'stimulus_right_12.5pct': 7,
    'stimulus_right_6.25pct': 8
}

print("Available stimulus types:")
for name, stimulus_type in stimulus_feature_mapping.items():
    print(f"  {stimulus_type}: {name}")

# Select stimulus type to hold out for testing
held_out_stimulus_names = ['stimulus_left_6.25pct']  # Left 6.25% contrast
held_out_stimulus_types = [stimulus_feature_mapping[name] for name in held_out_stimulus_names]

print(f"\nHolding out for test set: {held_out_stimulus_names} (stimulus types: {held_out_stimulus_types})")

split_map = TimeSeriesSplitter.create_stimulus_based_splits(
    covariate_matrix=covariate_matrix,
    train_pct=0.9,
    val_pct=0.05,
    held_out_stimulus_types=held_out_stimulus_types
)

# Test with realistic DLinear parameters
seq_len, pred_len, label_len = 48, 16, 4

splitter = TimeSeriesSplitter(
    split_map=split_map,
    seq_len=seq_len,
    pred_len=pred_len,
    label_len=label_len
)

summary = splitter.get_split_summary()
print(f"\nSplit Summary:")
for key, value in summary.items():
    if 'pct' in key:
        print(f"  {key}: {value:.1f}%")
    else:
        print(f"  {key}: {value}")

print("\n=== VISUALIZING SPLIT MAP ===")
# %%

# Define time window for visualization (in sample indices)  

# Use a smaller window for better sample visualization
viz_xmin_idx = 8000 
viz_xmax_idx = 10000
xmin_idx = 10000
xmax_idx = 12000  # Show 6000 samples to see multiple split transitions

model = Rastermap(n_clusters=100, # number of clusters to compute
                  n_PCs=200, # number of PCs
                  locality=0.5, # locality in sorting is low here to get more global sorting (this is a value from 0-1)
                  time_lag_window=5, # use future timepoints to compute correlation
                  grid_upsample=10, # default value, 10 is good for large recordings
                  ).fit(activity.T)

y = model.embedding 
isort = model.isort

X_embedding = model.X_embedding

def create_split_overlay(split_map, xmin_idx, xmax_idx, X_embedding_shape):
    """Create a colored overlay for train/val/test splits"""
    # Get split data for the time window
    split_window = split_map[xmin_idx:xmax_idx]
    
    # Define colors for each split type
    colors = [
        [0, 0, 1, 0.3],      # 0: Train - blue
        [1, 0.5, 0, 0.3],    # 1: Val - orange  
        [1, 0, 0, 0.3]       # 2: Test - red
    ]
    
    # Create RGB overlay
    nn = X_embedding_shape[0]  # Number of neurons
    n_timepoints = len(split_window)
    overlay = np.zeros((nn, n_timepoints, 4))  # RGBA
    
    for t, split_type in enumerate(split_window):
        overlay[:, t] = colors[split_type]
    
    return overlay

# Create figure for stimulus-colored rastermap with wheel velocity
fig = plt.figure(figsize=(15, 8), dpi=150)
grid = plt.GridSpec(12, 20, figure=fig, wspace=0.1, hspace=0.4)


# Plot rastermap (keep original scaling)
ax_raster = plt.subplot(grid[2:9, :-1])
ax_raster.imshow(X_embedding[:, xmin_idx:xmax_idx], cmap="gray_r", vmin=0, vmax=0.8, aspect="auto")

# Create and overlay split colors
split_overlay = create_split_overlay(split_map, xmin_idx, xmax_idx, X_embedding.shape)
ax_raster.imshow(split_overlay, aspect="auto")

ax_raster.set_ylabel("Neurons (sorted)")
ax_raster.set_title("Rastermap with Train/Val/Test Split Overlay")

ax_cbar = plt.subplot(grid[2:9, -1])
activity_gradient = np.linspace(0.8, 0, X_embedding.shape[0])[:, np.newaxis]  # High at top
ax_cbar.imshow(activity_gradient, cmap="gray_r", aspect="auto", vmin=0, vmax=0.8)
ax_cbar.yaxis.set_label_position("right")
ax_cbar.set_ylabel("Activity", rotation=270, labelpad=10)
ax_cbar.set_yticks([0, X_embedding.shape[0]//2, X_embedding.shape[0]-1])
ax_cbar.set_yticklabels(['0.80', '0.40', '0.00'])
ax_cbar.set_xticks([])
ax_cbar.yaxis.tick_right()

# Plot wheel velocity below the rastermap (bigger vertical space)
# ax_wheel = plt.subplot(grid[10:12, :-1])
# plot_wheel_velocity(ax_wheel, preprocessed_session.aligned_wheel_velocity, timestamps, xmin_idx, xmax_idx)

# Create legend for split types
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=(0, 0, 1, 0.3), label='Train'),
    Patch(facecolor=(1, 0.5, 0, 0.3), label='Val'),
    Patch(facecolor=(1, 0, 0, 0.3), label='Test'),
]
ax_raster.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)

plt.show()

# %%
# === VISUALIZE ACTUAL DATALOADER SAMPLES ===
print("\n=== TESTING ACTUAL DATALOADER SAMPLES ===")

# Create actual DataLoader instances using the same parameters as the splitter
datasets = {}
for flag in ['train', 'val', 'test']:
    print(f"\nCreating {flag} dataset...")
    datasets[flag] = Dataset_Activity(
        root_path="../DATA",
        data_path="session_5ea6bb9b-6163-4e8a-816b-efe7002666b0.h5",
        flag=flag,
        size=[seq_len, label_len, pred_len]
    )
    print(f"  Dataset length: {len(datasets[flag])}")

# %%
# === VISUALIZE SAMPLE WINDOWS ON RASTERMAP ===
print("\n=== VISUALIZING DATALOADER SAMPLE WINDOWS ===")
# Use a smaller window for better sample visualization
viz_xmin_idx = 0
viz_xmax_idx = 20000

# Create figure for sample window visualization
fig = plt.figure(figsize=(20, 12), dpi=150)
from matplotlib.gridspec import GridSpec
grid = GridSpec(15, 1, figure=fig, hspace=0.4)

# Plot split map as timeline at the top
ax_splits = plt.subplot(grid[0:2, :])
split_window = split_map[viz_xmin_idx:viz_xmax_idx]
split_colors = ['blue', 'orange', 'red']
split_color_array = [split_colors[split] for split in split_window]

for i, (split_val, color) in enumerate(zip(split_window, split_color_array)):
    ax_splits.axvline(i, color=color, alpha=0.8, linewidth=1)

ax_splits.set_xlim(0, len(split_window))
ax_splits.set_ylabel('Splits')
ax_splits.set_yticks([])
ax_splits.set_title(f'Train/Val/Test Timeline (indices {viz_xmin_idx}-{viz_xmax_idx})')

# Add split legend
from matplotlib.patches import Patch
split_legend = [
    Patch(facecolor='blue', label='Train'),
    Patch(facecolor='orange', label='Val'), 
    Patch(facecolor='red', label='Test')
]
ax_splits.legend(handles=split_legend, loc='upper right')

# Plot rastermap with sample windows
ax_raster = plt.subplot(grid[3:12, :])
raster_data = X_embedding[:, viz_xmin_idx:viz_xmax_idx]
ax_raster.imshow(raster_data, cmap="gray_r", vmin=0, vmax=0.8, aspect="auto")

# Sample from each dataset and overlay their windows
sample_colors = {'train': 'blue', 'val': 'orange', 'test': 'red'}
sample_alphas = {'train': 0.3, 'val': 0.5, 'test': 0.4}

for flag in ['train', 'val', 'test']:
    dataset = datasets[flag]
    if len(dataset) == 0:
        continue
        
    # Sample multiple indices from this dataset
    n_samples_to_show = min(100, len(dataset))
    np.random.seed(42)  # Reproducible
    sample_indices = np.random.choice(len(dataset), size=n_samples_to_show, replace=False)
    
    print(f"\n{flag.upper()} samples in visualization window:")
    samples_in_window = 0
    
    for sample_idx in sample_indices:
        # Get actual data indices from the DataLoader
        data_start_idx = dataset.valid_indices[sample_idx]
        
        # Calculate sequence window (same as DataLoader __getitem__)
        s_begin = data_start_idx
        s_end = s_begin + seq_len
        r_begin = s_end - label_len  
        r_end = r_begin + label_len + pred_len
        
        # Check if this sample falls within our visualization window
        if (s_begin >= viz_xmin_idx and r_end <= viz_xmax_idx):
            samples_in_window += 1
            
            # Convert to relative indices within the visualization window
            rel_s_begin = s_begin - viz_xmin_idx
            rel_s_end = s_end - viz_xmin_idx
            rel_r_begin = r_begin - viz_xmin_idx
            rel_r_end = r_end - viz_xmin_idx
            
            # Draw input sequence (seq_x)
            ax_raster.axvspan(rel_s_begin, rel_s_end, alpha=sample_alphas[flag], 
                            color=sample_colors[flag], label=f'{flag} input' if sample_idx == sample_indices[0] else "")
            
            # Draw target sequence (seq_y) with different pattern
            ax_raster.axvspan(rel_r_begin, rel_r_end, alpha=sample_alphas[flag]*0.7, 
                            color=sample_colors[flag], linestyle='--', linewidth=2,
                            label=f'{flag} target' if sample_idx == sample_indices[0] else "")
            
            print(f"  Sample {sample_idx}: data_idx {data_start_idx}, input [{rel_s_begin}:{rel_s_end}], target [{rel_r_begin}:{rel_r_end}]")
    
    print(f"  {samples_in_window}/{n_samples_to_show} samples visible in window")

ax_raster.set_ylabel("Neurons (sorted)")
ax_raster.set_xlabel("Time (relative to window)")
ax_raster.set_title("Rastermap with Actual DataLoader Sample Windows")
ax_raster.legend(loc='upper right')

# Add sample statistics
ax_stats = plt.subplot(grid[13:15, :])
ax_stats.axis('off')

stats_text = f"""
Dataset Statistics:
• Train: {len(datasets['train']):,} samples ({len(datasets['train'])/sum(len(d) for d in datasets.values())*100:.1f}%)
• Val: {len(datasets['val']):,} samples ({len(datasets['val'])/sum(len(d) for d in datasets.values())*100:.1f}%)  
• Test: {len(datasets['test']):,} samples ({len(datasets['test'])/sum(len(d) for d in datasets.values())*100:.1f}%)
• Total: {sum(len(d) for d in datasets.values()):,} samples

Sample Parameters:
• seq_len: {seq_len} (input sequence length)
• pred_len: {pred_len} (prediction length)  
• label_len: {label_len} (decoder overlap length)
• Total window: {seq_len + pred_len} timepoints per sample
"""

ax_stats.text(0.02, 0.5, stats_text, fontsize=10, verticalalignment='center',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

plt.tight_layout()
plt.show()

# %%
# # Validate leak-free property with specific examples
# print("\n=== VALIDATING LEAK-FREE PROPERTY ===")

# def check_window_splits(sample_idx, data_idx, split_type_name):
#     """Check what splits a sample window touches"""
#     window_end = min(data_idx + seq_len + pred_len, len(split_map))
#     window_splits = split_map[data_idx:window_end]
#     unique_splits = set(window_splits)
    
#     print(f"{split_type_name} sample {sample_idx}: data_idx={data_idx}")
#     print(f"  Window: {data_idx} to {window_end-1} (length {window_end-data_idx})")
#     print(f"  Touches splits: {sorted(unique_splits)}")
    
#     # Validation logic
#     if split_type_name == 'Train':
#         valid = unique_splits == {0}
#         print(f"  ✓ Valid: Must only touch train data" if valid else f"  ✗ Invalid: Touches non-train data")
#     elif split_type_name == 'Val':
#         valid = 2 not in unique_splits  # Cannot touch test
#         print(f"  ✓ Valid: No test data contamination" if valid else f"  ✗ Invalid: Touches test data")
#     else:  # Test
#         valid = True  # Test can touch anything
#         print(f"  ✓ Valid: Test samples can touch any data")
    
#     return valid

# # Check specific examples
# print("\nChecking sample windows:")
# all_valid = True

# # Training samples - must stay in train data
# train_samples = [(0, train_indices[0]), (100, train_indices[100]), (1000, train_indices[1000])]
# for sample_idx, data_idx in train_samples:
#     valid = check_window_splits(sample_idx, data_idx, 'Train')
#     all_valid = all_valid and valid
#     print()

# # Validation samples - can span train/val but not test
# val_samples = [(0, val_indices[0]), (50, val_indices[50]), (100, val_indices[100])]
# for sample_idx, data_idx in val_samples:
#     valid = check_window_splits(sample_idx, data_idx, 'Val')
#     all_valid = all_valid and valid
#     print()

# # Test samples - can touch any data
# test_samples = [(0, test_indices[0]), (100, test_indices[100]), (500, test_indices[500])]
# for sample_idx, data_idx in test_samples:
#     valid = check_window_splits(sample_idx, data_idx, 'Test')
#     all_valid = all_valid and valid
#     print()

# print(f"=== OVERALL VALIDATION: {'✅ PASSED' if all_valid else '❌ FAILED'} ===")

# # %%
# # Test actual Dataset_Activity classes
# print("\n=== TESTING DATASET_ACTIVITY CLASSES ===")

# datasets = {}
# for flag in ['train', 'val', 'test']:
#     print(f"\nCreating {flag} dataset...")
#     datasets[flag] = Dataset_Activity(
#         root_path="../DATA",
#         data_path="session_61f260e7-b5d3-4865-a577-bcfc53fda8a8.h5",
#         flag=flag,
#         size=[seq_len, label_len, pred_len]
#     )
#     print(f"  Dataset length: {len(datasets[flag])}")

# # Test sample extraction and verify no data leakage with random sampling
# print(f"\n=== TESTING DATASET_ACTIVITY LEAK-FREE SAMPLING ===")

# def validate_dataset_splits(dataset, flag, split_map, n_samples=100):
#     """Validate that dataset samples don't leak across splits"""
#     print(f"\n{flag.upper()} Dataset Validation ({n_samples} random samples):")
    
#     dataset_length = len(dataset)
#     if dataset_length == 0:
#         print(f"  ❌ Empty dataset!")
#         return False
    
#     # Sample random indices from the dataset
#     np.random.seed(42)  # Reproducible results
#     sample_indices = np.random.choice(dataset_length, size=min(n_samples, dataset_length), replace=False)
    
#     violations = []
#     all_valid = True
    
#     for i, sample_idx in enumerate(sample_indices):
#         # Get the actual sequence from __getitem__ (verify it works)
#         _ = dataset[sample_idx]  # We only care that this doesn't error
            
#         # Get the actual data indices this sample touches
#         actual_data_idx = dataset.valid_indices[sample_idx]  # Starting point in full dataset
        
#         # Calculate the full window this sample covers
#         s_begin = actual_data_idx
#         s_end = s_begin + seq_len
#         r_begin = s_end - label_len  
#         r_end = r_begin + label_len + pred_len
        
#         # Check what splits this window touches
#         if r_end > len(split_map):
#             print(f"  ❌ Sample {sample_idx} extends beyond data bounds!")
#             all_valid = False
#             continue
            
#         window_splits = split_map[s_begin:r_end]
#         unique_splits = set(window_splits)
        
#         # Validate based on split type
#         valid = True
#         if flag == 'train':
#             # Training samples must only touch training data (split 0)
#             if unique_splits != {0}:
#                 valid = False
#                 violations.append((sample_idx, actual_data_idx, unique_splits))
#         elif flag == 'val':
#             # Validation samples cannot touch test data (split 2)
#             if 2 in unique_splits:
#                 valid = False  
#                 violations.append((sample_idx, actual_data_idx, unique_splits))
#         # Test samples can touch any data - no restrictions
        
#         if not valid:
#             all_valid = False
            
#         # Progress indicator for large validations
#         if (i + 1) % 25 == 0 or i == 0:
#             print(f"  Validated {i+1}/{len(sample_indices)} samples...")
    
#     # Report results
#     if all_valid:
#         print(f"  ✅ All {len(sample_indices)} samples passed validation!")
#         print(f"  ✅ No data leakage detected in {flag} dataset")
#     else:
#         print(f"  ❌ Found {len(violations)} violations out of {len(sample_indices)} samples:")
#         for sample_idx, data_idx, splits in violations[:5]:  # Show first 5 violations
#             print(f"    Sample {sample_idx} (data_idx {data_idx}) touches splits: {splits}")
#         if len(violations) > 5:
#             print(f"    ... and {len(violations) - 5} more violations")
    
#     return all_valid

# # Validate each dataset with random sampling
# overall_valid = True
# for flag in ['train', 'val', 'test']:
#     dataset = datasets[flag]
#     valid = validate_dataset_splits(dataset, flag, split_map, n_samples=100)
#     overall_valid = overall_valid and valid

# print(f"\n=== FINAL DATASET VALIDATION: {'✅ PASSED' if overall_valid else '❌ FAILED'} ===")

# # Quick shape verification for a few samples
# print(f"\n=== SAMPLE SHAPE VERIFICATION ===")
# for flag in ['train', 'val', 'test']:
#     if len(datasets[flag]) > 0:
#         seq_x, seq_y, seq_x_mark, seq_y_mark = datasets[flag][0]
#         print(f"{flag}: Neural {seq_x.shape}, {seq_y.shape} | Covariates {seq_x_mark.shape}, {seq_y_mark.shape}")

# # %%
# %%
