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
- `MesoscopeSession`: Main class for handling neural imaging sessions
- `FOVData`: Container for individual Field of View data
- Key methods:
  - `from_eid()`: Load session from experiment ID via ONE API
  - `from_csv()`: Load session from CSV file list
  - `load_activity_matrix()`: Load preprocessed HDF5 activity data
  - `get_activity_matrix()`: Extract combined neural activity across FOVs
  - `save_activity_matrix()`: Save processed data to HDF5

#### 2. Directory Structure
- `1_ANALYSIS/`: Exploratory data analysis notebooks
- `2_PREPROCESSING/`: Data normalization and binning
- `3_MODELLING/`: Deep learning model implementations
  - Transformer variants (Informer, Autoformer, DLinear)
  - Baseline models (linear regression, autoregression)
  - PyTorch Lightning implementations
- `DATA/`: HDF5 files containing processed activity matrices
- `TUTORIALS/`: IBL data access and general tutorials
- `utils/`: Utility functions for data loading

#### 3. Model Architecture
- **Primary Model**: Informer transformer for time series forecasting
- **Input**: Neural activity time series [time_points, neurons]
- **Output**: Predicted future neural activity
- **Training**: Custom loss with spike detection weighting and scale regularization

## Common Development Tasks

### Data Loading
```python
from utils.load_meso_session import MesoscopeSession

# Load from HDF5
activity_matrix, timestamps, eid = MesoscopeSession.load_session_from_hdf5('DATA/activity_raw.h5')

# Load from IBL directly
session = MesoscopeSession.from_csv(one, 'good_mesoscope_sessions_final.csv', index=0)
activity_matrix, timestamps = session.get_activity_matrix(time_window=1000)

# Get ΔF/F corrected activity (preferred for modeling)
dff_matrix, timestamps = session.get_dff_activity_matrix(neucoeff=0.7)
session.save_dff_activity_matrix('DATA/dff_activity.h5', neucoeff=0.7)
```

### Model Training
Primary model training is done in Jupyter notebooks in `3_MODELLING/`:
- `7_2_informer.ipynb`: Main Informer implementation
- Training uses PyTorch with custom loss functions for neural spike detection
- Models saved to `3_MODELLING/informer_calcium_model/`

### Data Preprocessing
Neural activity data requires preprocessing before model training:
- **ΔF/F Correction**: Use `get_dff_activity_matrix()` to apply Suite2p neuropil correction (F - 0.7*Fneu)
- **Normalization**: Per-neuron percentile normalization (0-99.5th percentile) on corrected data
- **Windowing**: Sequence-to-sequence learning with sliding windows
- **Quality control**: Filtering of bad frames and QC metrics

## Key Dependencies
- PyTorch + transformers library (HuggingFace)
- ONE API for IBL data access
- h5py for HDF5 data storage
- Pydantic for data validation
- Standard scientific Python stack (numpy, pandas, matplotlib, seaborn)

## Important Notes
- No formal package management (requirements.txt/environment.yml) - dependencies managed ad-hoc
- Models are trained on GPU when available, fall back to CPU
- Data files are large (multi-GB HDF5 files) - ensure adequate disk space
- Session data contains multiple FOVs (Fields of View) that are concatenated for modeling

## Data Sources
- `good_mesoscope_sessions_final.csv`: Curated list of high-quality IBL sessions
- Raw IBL data accessed via ONE API
- Processed activity matrices stored in `DATA/activity_raw.h5`