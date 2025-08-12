# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a neural time series forecasting project that applies transformer models (specifically Informer) to mesoscope calcium imaging data from the International Brain Laboratory (IBL). The project analyzes neural activity patterns from mouse cortical imaging sessions and trains deep learning models to predict future neural activity.

## Architecture

### Data Pipeline
- **Raw Data**: Mesoscope calcium imaging sessions from IBL stored in HDF5 format
- **Session Management**: `MesoscopeSession` class handles loading, filtering, and preprocessing of multi-FOV neural data
- **Data Storage**: Activity matrices are saved/loaded from HDF5 files in `DATA/` directory

### Key Components

#### 1. Data Loading (`utils/load_meso_session.py`)
- `MesoscopeSession`: Main class for handling neural imaging sessions with dual-mode design
- `FOVData`: Container for individual Field of View data (only used in exploration mode)
- Key methods:
  - `from_eid()`: Load session from experiment ID via ONE API (exploration mode)
  - `from_preprocessed()`: Load session from preprocessed HDF5 file (training mode)
  - `get_activity_matrix()`: Extract combined neural activity across FOVs (exploration only)
  - `get_dff_activity_matrix()`: Extract ΔF/F activity across FOVs (exploration only)
  - `get_preprocessed_data()`: Access preprocessed data (training mode only)

#### 2. Data Preprocessing (`utils/activity_preprocessor.py`)
- `CalciumDataPreprocessor`: Complete preprocessing pipeline for calcium imaging data
- Handles ΔF/F calculation, neuron filtering, robust normalization, and temporal smoothing
- Saves preprocessed data with behavioral data to HDF5 files for training

#### 3. Directory Structure
- `1_ANALYSIS/`: Exploratory data analysis notebooks
- `2_PREPROCESSING/`: Data normalization, preprocessing pipeline, and quality control notebooks
- `3_MODELLING/`: Deep learning model implementations
  - Transformer variants (Informer, Autoformer, DLinear)
  - Baseline models (linear regression, autoregression)
  - PyTorch Lightning implementations
- `DATA/`: HDF5 files containing processed activity matrices
- `TUTORIALS/`: IBL data access and general tutorials
- `utils/`: Utility functions for data loading

#### 4. Model Architecture
- **Primary Model**: Informer transformer for time series forecasting
- **Input**: Neural activity time series [time_points, neurons]
- **Output**: Predicted future neural activity
- **Training**: Custom loss with spike detection weighting and scale regularization

## Common Development Tasks

### Data Loading

#### Exploration Mode (Raw Data Analysis)
```python
from utils.load_meso_session import MesoscopeSession
from one.api import ONE

# Load session directly from IBL via ONE API
one = ONE()
session = MesoscopeSession.from_eid(one, 'your_eid_here', raw_activity=True)

# Access raw activity across all FOVs
activity_matrix, timestamps = session.get_activity_matrix(time_window=1000)

# Get ΔF/F corrected activity (preferred for analysis)
dff_matrix, timestamps = session.get_dff_activity_matrix(neucoeff=0.7)

# Access individual FOV data for detailed analysis
for fov_name, fov_data in session.fovs.items():
    neuron_activity, timestamps, mask = fov_data.get_neuron_dff_activity(neucoeff=0.7)
```

#### Training Mode (Preprocessed Data)
```python
from utils.load_meso_session import MesoscopeSession
from 2_PREPROCESSING.activity_preprocessor import CalciumDataPreprocessor

# First, preprocess raw session data
one = ONE()
session = MesoscopeSession.from_eid(one, 'your_eid_here', raw_activity=True)
preprocessor = CalciumDataPreprocessor(neucoeff=0.7, temporal_smoothing=True)
results = preprocessor.preprocess_session(session, 'processed_data/session.h5')

# Later, load preprocessed data for training
session = MesoscopeSession.from_preprocessed('processed_data/session.h5')
processed_data, timestamps = session.get_preprocessed_data()

# Trial data is still available
stim_times = session.stimOn_times
feedback_times = session.feedback_times
wheel_pos = session.wheel_position
response_times = session.response_times
contrast_left = session.contrastLeft
contrast_right = session.contrastRight
feedback_type = session.feedbackType
trial_intervals = session.intervals
wheel_vel = session.wheel_velocity
```

### Model Training
Primary model training is done in Jupyter notebooks in `3_MODELLING/`:
- `7_2_informer.ipynb`: Main Informer implementation
- Training uses PyTorch with custom loss functions for neural spike detection
- Models saved to `3_MODELLING/informer_calcium_model/`

### Data Preprocessing Pipeline
The `CalciumDataPreprocessor` class provides a complete preprocessing pipeline:

1. **ΔF/F Calculation**: Suite2p neuropil correction (F - neucoeff*Fneu) with configurable coefficient
2. **Neuron Filtering**: Statistical filtering to remove problematic neurons based on:
   - Variability (IQR thresholds)
   - Outlier detection (extreme range filtering)
   - Baseline quality (median activity levels)
   - Signal quality (standard deviation bounds)
3. **Robust Normalization**: Per-neuron percentile-based scaling (0.5th to 99.5th percentile)
4. **Temporal Smoothing**: Optional Gaussian smoothing to reduce noise
5. **Quality Assessment**: Comprehensive metrics on data quality and preprocessing effects

The pipeline saves all results, behavioral data, and metadata to HDF5 files optimized for training.

### Wheel Velocity Integration
The `MesoscopeSession` class includes integrated wheel velocity calculation:

- **`get_aligned_wheel_velocity(neural_timestamps)`**: Calculates wheel velocity from raw position data and aligns it to neural activity timestamps
- **Preprocessing Integration**: `CalciumDataPreprocessor` automatically calculates and saves `aligned_wheel_velocity` in trial_data
- **Velocity Calculation**: Uses brainbox.behavior.wheel with 5Hz sampling, 2Hz low-pass filter, and ±2.5 rad/s clipping
- **Zero-padding**: Missing wheel data periods (no movement) are filled with zeros
- **Dual-mode Support**: Available in both exploration (`get_aligned_wheel_velocity()`) and training modes (`session.aligned_wheel_velocity`)

## Key Dependencies
- PyTorch + transformers library (HuggingFace)
- ONE API for IBL data access
- h5py for HDF5 data storage
- Pydantic for data validation
- scipy for signal processing (preprocessing pipeline)
- Standard scientific Python stack (numpy, pandas, matplotlib, seaborn)

## Important Notes
- No formal package management (requirements.txt/environment.yml) - dependencies managed ad-hoc
- Models are trained on GPU when available, fall back to CPU
- Data files are large (multi-GB HDF5 files) - ensure adequate disk space
- **Dual-mode design**: Use `from_eid()` for exploration/analysis, `from_preprocessed()` for training
- Session data contains multiple FOVs (Fields of View) - only available in exploration mode
- Trial data (stimuli, feedback, wheel, contrasts, response times) is preserved across both modes
- When working with jupyter notebooks, always look for a file that has the same file name but ends in _Claude.py. You are incapable of working directly in jupyter notebooks due to it's JSON structure. As such, I will manually copy and paste your changes from the .py file into ipynb.

## IBL Stimulus Type Encoding System

### Stimulus Types (9 Total)
The project uses a standardized encoding for IBL behavioral trials based on contrast combinations:

**Type 0: Catch Trials (Purple)**
- `contrastLeft=0, contrastRight=NaN` OR `contrastLeft=NaN, contrastRight=0`  
- No visual stimulus presented - tests animal's use of prior knowledge about stimulus probability

**Types 1-4: Left Stimuli (Red Shades)**
- `contrastLeft>0, contrastRight=NaN`
- Type 1: Left 100% contrast (dark red)
- Type 2: Left 25% contrast (medium red)  
- Type 3: Left 12.5% contrast (light red)
- Type 4: Left 6.25% contrast (very light red)

**Types 5-8: Right Stimuli (Blue Shades)**  
- `contrastLeft=NaN, contrastRight>0`
- Type 5: Right 100% contrast (dark blue)
- Type 6: Right 25% contrast (medium blue)
- Type 7: Right 12.5% contrast (light blue)  
- Type 8: Right 6.25% contrast (very light blue)

### Encoding Functions
- **`encode_stimulus_types(session, timestamps)`**: Creates one-hot encoding `[n_timepoints, 9]` for all stimulus types
- **`_get_stimulus_type(left_contrast, right_contrast, contrast_values)`**: Maps individual trial contrasts to stimulus type indices
- **Visualization**: Color-coded overlays on rastermap plots show stimulus timing and type

## Covariate Integration System

### Covariate Builder (`utils/covariate_builder.py`)
The project includes a comprehensive behavioral covariate system for time series forecasting:

- **`build_covariate_matrix(session, timestamps, aligned_wheel_velocity)`**: Creates [n_timepoints, 11] covariate matrix
- **11-feature structure**: wheel_velocity [1] + stimulus_types [9] + trial_phase [1]
- **Temporal alignment**: All covariates precisely aligned to neural timestamps at 5Hz sampling
- **Feature documentation**: Complete metadata with descriptions for each covariate

### Updated HDF5 Schema (Post-Covariate Integration)
```
/activity                   # Neural activity [n_timepoints, n_neurons] (renamed from processed_data)
/covariate_matrix          # Behavioral features [n_timepoints, 11] 
/timestamps                # Neural timestamps
/covariate_metadata/       # Feature names and descriptions
  ├── feature_names        # String array: ['wheel_velocity', 'stimulus_catch_trial', ...]
  └── [feature]_description # Detailed descriptions as HDF5 attributes
/trial_data/              # Original behavioral data (preserved)
/normalization/           # Preprocessing parameters (preserved)  
/metadata/               # Session metadata (preserved)
```

### Covariate Features [n_timepoints, 11]
1. **wheel_velocity**: Aligned wheel velocity (rad/s) calculated from position data
2. **stimulus_catch_trial**: Binary indicator for catch trials (no visual stimulus)
3-6. **stimulus_left_[100|25|12.5|6.25]pct**: Binary indicators for left-side stimuli by contrast
7-10. **stimulus_right_[100|25|12.5|6.25]pct**: Binary indicators for right-side stimuli by contrast
11. **trial_phase**: Binary indicator (1=stimulus period, 0=inter-trial interval)

### Dataset_Activity Integration
Updated Dataset_Activity class now supports covariate loading:
- **Real covariate data**: Replaces dummy `seq_x_mark`/`seq_y_mark` tensors with actual behavioral features
- **Sequence alignment**: Input and target covariates properly aligned with neural sequences
- **Torch-free testing**: `SimpleDataset` base class for testing without PyTorch dependency
- **Clear documentation**: Detailed comments explaining sequence extraction logic (`s_begin`, `s_end`, `r_begin`, `r_end`)

### Validation Suite (`2_PREPROCESSING/4_Validate_Preprocessing_Claude.py`)
Comprehensive validation notebook with 4 consolidated cells:
- **Treescope integration**: `np.column_stack()` arrays for visual data inspection
- **Rastermap visualization**: Stimulus-colored overlays matching analysis style
- **Temporal alignment validation**: Verifies neural-behavioral synchronization
- **Dataset_Activity testing**: Standalone covariate loading verification
- **Ready/not-ready assessment**: Clear status for DLinear integration

### Usage Examples

#### Preprocessing with Covariates
```python
from utils.activity_preprocessor import CalciumDataPreprocessor

# Preprocessing automatically builds and saves covariate matrix
preprocessor = CalciumDataPreprocessor(neucoeff=0.7, temporal_smoothing=True)
results = preprocessor.preprocess_session(session, 'processed_data/session.h5')

# Results include covariate_matrix in trial_data
covariate_matrix = results['trial_data']['covariate_matrix']  # [n_timepoints, 11]
feature_names = results['trial_data']['covariate_feature_names']
```

#### Loading Preprocessed Data with Covariates
```python
# Load session with covariate data
session = MesoscopeSession.from_preprocessed('processed_data/session.h5')

# Access via HDF5 directly  
with h5py.File('processed_data/session.h5', 'r') as f:
    activity = f['activity'][:]                 # Neural data
    covariate_matrix = f['covariate_matrix'][:]  # Behavioral features
    feature_names = [name.decode('utf-8') for name in f['covariate_metadata']['feature_names'][:]]
```

#### Dataset_Activity with Covariates
```python  
from 2_PREPROCESSING.Test_Data_Loader import Dataset_Activity

dataset = Dataset_Activity(root_path="DATA/", data_path="session.h5", flag='train')
seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[0]

# seq_x_mark: [96, 11] input behavioral features
# seq_y_mark: [144, 11] target behavioral features (known future values)
```

## Data Sources
- `good_mesoscope_sessions_final.csv`: Curated list of high-quality IBL sessions
- Raw IBL data accessed via ONE API
- Processed activity matrices stored in `DATA/session_[eid].h5`

## TimeSeriesSplitter System

### Leak-Free Data Splitting (`utils/data_splitter.py`)
The project implements a sophisticated data splitting system that prevents temporal data leakage while allowing appropriate data flow for time series forecasting:

- **`TimeSeriesSplitter`**: Core class that generates leak-aware train/val/test sample indices
- **Leak-aware logic**: Different rules for each split type:
  - **Training**: Entire sequence window (seq_len + pred_len) must stay within training data
  - **Validation**: Can start in train/val data but cannot extend into test data
  - **Test**: Can start anywhere and extend into any data (most permissive)
- **Stimulus-based splits**: Uses existing covariate matrix to create balanced splits per stimulus type

### Updated Dataset_Activity Architecture (`2_PREPROCESSING/Test_Data_Loader.py`)
Complete overhaul of data loading to use leak-free splitting:

- **Replaced chronological splits**: No more border1/border2 approach
- **Valid indices approach**: `__len__()` returns number of valid samples, `__getitem__(index)` uses `valid_indices[index]` 
- **Full data storage**: Stores complete dataset and uses TimeSeriesSplitter to determine safe starting points
- **DLinear compatibility**: Maintains same constructor interface and return shapes

### Splitting Logic Details
```python
from utils.data_splitter import TimeSeriesSplitter

# Create stimulus-based split map
split_map = TimeSeriesSplitter.create_stimulus_based_splits(
    covariate_matrix=covariate_matrix,  # Uses existing [n_timepoints, 11] matrix
    train_pct=0.7, val_pct=0.1,
    held_out_stimulus_types=[]  # Optional held-out types for test
)

# Generate leak-free sample indices  
splitter = TimeSeriesSplitter(split_map, seq_len=96, pred_len=96, label_len=48)
train_indices = splitter.get_indices('train')  # Safe starting points only
```

### Validation Suite (`2_PREPROCESSING/Validate_LeakFree_Splitting_Claude.py`)
Comprehensive testing with visualization:
- **Split visualization**: Timeline plots showing train/val/test boundaries
- **Window validation**: Confirms sample windows respect split rules
- **Dataset integration testing**: All three Dataset_Activity classes (train/val/test)
- **Leak detection**: Explicit validation that no inappropriate data leakage occurs

### Key Results
- **Sample distribution**: ~63.5% train, ~10.3% val, ~26.2% test from 18,890 total leak-free samples
- **Stimulus balancing**: All 9 stimulus types properly distributed across splits
- **Validation passed**: All sample windows respect temporal boundaries correctly
- **Ready for training**: DLinear-compatible interface maintained

### Usage Examples

#### Basic Dataset Loading
```python
from 2_PREPROCESSING.Test_Data_Loader import Dataset_Activity

# Each split uses TimeSeriesSplitter internally
train_dataset = Dataset_Activity(root_path="../DATA", data_path="session.h5", flag='train')
val_dataset = Dataset_Activity(root_path="../DATA", data_path="session.h5", flag='val') 
test_dataset = Dataset_Activity(root_path="../DATA", data_path="session.h5", flag='test')

# Datasets return leak-free samples
seq_x, seq_y, seq_x_mark, seq_y_mark = train_dataset[0]
```

#### Custom Split Configuration
```python
# Create custom held-out test set (e.g., Left 100% contrast)
split_map = TimeSeriesSplitter.create_stimulus_based_splits(
    covariate_matrix, held_out_stimulus_types=[1]  # Hold out Left 100%
)
```

## Memories
- **Covariate Integration Complete**: Implemented comprehensive behavioral covariate system with 11-dimensional feature matrix (wheel velocity + 9 stimulus types + trial phase), integrated into preprocessing pipeline, updated HDF5 schema, and validated with comprehensive testing suite. Ready for DLinear forecasting experiments.
- **TimeSeriesSplitter Complete**: Implemented leak-aware data splitting system that prevents temporal data leakage while allowing appropriate data flow for time series models. Uses stimulus-based balancing, generates 18,890 leak-free samples (63.5% train, 10.3% val, 26.2% test), and maintains DLinear compatibility. Fully validated with comprehensive testing suite showing no data contamination across splits.