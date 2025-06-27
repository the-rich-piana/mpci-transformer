from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from one.api import ONE
import numpy.typing as npt
import h5py
import os

class FOVData(BaseModel):
    """Data for a single Field of View (FOV)"""
    fov_name: str
    collection: str
    roi_activity: np.ndarray = Field(..., description="ROI activity matrix [frames, ROIs]")
    timestamps: np.ndarray = Field(..., description="Timestamps for each frame")
    bad_frames: np.ndarray = Field(..., description="Boolean array marking bad frames")
    frame_qc: np.ndarray = Field(..., description="QC values for frames")
    roi_types: Optional[np.ndarray] = Field(None, description="Types of ROIs (1=neuron)")
    brain_locations: Optional[np.ndarray] = Field(None, description="Brain locations for ROIs")
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def n_frames(self) -> int:
        """Get number of frames"""
        return self.roi_activity.shape[0]
    
    @property
    def n_rois(self) -> int:
        """Get number of ROIs"""
        return self.roi_activity.shape[1]
    
    @property
    def n_neurons(self) -> int:
        """Get number of neurons (ROIs with type 1)"""
        if self.roi_types is None:
            return 0
        return np.sum(np.array(self.roi_types) == 1)
    
    @property
    def good_frame_mask(self) -> np.ndarray:
        """Get mask for good frames"""
        return (self.bad_frames == 0) & (self.frame_qc == 0)
    
    def get_filtered_activity(self, time_window: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get activity filtered by good frames and time window"""
        # Filter by good frames
        good_mask = self.good_frame_mask
        filtered_activity = self.roi_activity[good_mask, :]
        filtered_timestamps = self.timestamps[good_mask]
        
        # Apply time window if specified
        if time_window is not None:
            time_limit = filtered_timestamps[0] + time_window
            time_mask = filtered_timestamps <= time_limit
            filtered_activity = filtered_activity[time_mask, :]
            filtered_timestamps = filtered_timestamps[time_mask]
            
        return filtered_activity, filtered_timestamps
    
    def get_neuron_activity(self, time_window: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get activity only for neurons, filtered by good frames and time window"""
        filtered_activity, filtered_timestamps = self.get_filtered_activity(time_window)
        
        # Filter by ROI type
        if self.roi_types is not None:
            neuron_mask = np.array(self.roi_types) == 1
            neuron_activity = filtered_activity[:, neuron_mask]
            return neuron_activity, filtered_timestamps, neuron_mask
        
        # If no ROI types available, return empty arrays
        return np.array([]).reshape(filtered_activity.shape[0], 0), filtered_timestamps, np.array([])

class MesoscopeSession(BaseModel):
    """Data for a complete mesoscope session with multiple FOVs"""
    eid: str
    subject: str
    date: str
    duration_hours: float
    raw_activity: bool
    task_protocol: str
    fovs: Dict[str, FOVData] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def n_fovs(self) -> int:
        """Get number of FOVs"""
        return len(self.fovs)
    
    @property
    def n_total_rois(self) -> int:
        """Get total number of ROIs across all FOVs"""
        return sum(fov.n_rois for fov in self.fovs.values())
    
    @property
    def n_total_neurons(self) -> int:
        """Get total number of neurons across all FOVs"""
        return sum(fov.n_neurons for fov in self.fovs.values())
        
    @classmethod
    def from_eid(cls, one: ONE, eid: str, raw_activity: bool, max_fovs: Optional[int] = None) -> 'MesoscopeSession':
        """Load mesoscope session data from experiment ID"""
        # Get session details
        session_info = one.get_details(eid, full=True)
        
        # Create session object
        session = cls(
            eid=eid,
            raw_activity=raw_activity,
            subject=session_info.get('subject', 'unknown'),
            date=str(session_info.get('start_time', 'unknown')),
            duration_hours=_calculate_duration(session_info),
            task_protocol=session_info.get('task_protocol', '')
        )
        
        # Find FOV collections
        all_datasets = one.list_datasets(eid)
        fov_collections = set()
        for dataset in all_datasets:
            if 'mpci' in dataset and 'FOV_' in dataset:
                parts = dataset.split('/')
                if len(parts) >= 2:
                    collection = '/'.join(parts[:-1])
                    fov_collections.add(collection)
        
        # Sort FOV collections
        fov_collections = sorted(list(fov_collections))
        
        # Limit the number of FOVs if specified
        if max_fovs is not None:
            fov_collections = fov_collections[:min(max_fovs, len(fov_collections))]
        
        # Load data for each FOV
        for collection in fov_collections:
            fov_name = collection.split('/')[-1]
            fov_data = _load_fov_data(one, eid, raw_activity, collection)
            if fov_data is not None:
                session.fovs[fov_name] = fov_data
        
        return session
    
    @classmethod
    def from_csv(cls, one: ONE, csv_path: str, index: int, raw_activity: bool = False, max_fovs: Optional[int] = None) -> 'MesoscopeSession':
        """Load mesoscope session from a CSV file containing session list"""
        import pandas as pd
        
        # Load sessions
        sessions_df = pd.read_csv(csv_path)
        
        # Get session at the specified index
        selected_session = sessions_df.iloc[index]
        eid = selected_session['eid']
        # Load the session
        return cls.from_eid(one, eid, raw_activity, max_fovs)
    
    def get_activity_matrix(self, time_window: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get a combined matrix of all neuron activity across all FOVs
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (activity_matrix, timestamps)
            - activity_matrix: Combined matrix [time, neurons] of all neurons across FOVs
            - timestamps: Common timeline for the activity
        """
        # List to store neuron activity from each FOV
        all_neuron_activities = []
        common_timestamps = None
        
        # Process each FOV
        for fov_name, fov in self.fovs.items():
            # Get neuron activity, timestamps, and neuron mask
            neuron_activity, timestamps, _ = fov.get_neuron_activity(time_window)
            
            # Skip if no neurons
            if neuron_activity.shape[1] == 0:
                continue
                
            # Store the neuron activity
            all_neuron_activities.append(neuron_activity)
            
            # Store the timestamps (should be the same across FOVs,
            # but we'll use the first valid one)
            if common_timestamps is None and timestamps.size > 0:
                common_timestamps = timestamps
        
        # If no valid FOVs, return empty arrays
        if not all_neuron_activities or common_timestamps is None:
            return np.array([]), np.array([])
        
        # Verify all activities have the same number of time points
        expected_time_points = all_neuron_activities[0].shape[0]
        all_neuron_activities = [act for act in all_neuron_activities 
                                if act.shape[0] == expected_time_points]
        
        # Combine all neuron activities into a single matrix
        if all_neuron_activities:
            # Concatenate along the neuron dimension (axis=1)
            combined_activity = np.concatenate(all_neuron_activities, axis=1)
            return combined_activity, common_timestamps
        
        return np.array([]), common_timestamps
    
    def save_activity_matrix(self, path: str, time_window: Optional[float] = None, compression: str = 'gzip'):
        """Save the combined activity matrix to an HDF5 file
        
        Args:
            path (str): Path to save the HDF5 file (should end with .h5 or .hdf5)
            time_window (Optional[float]): Time window to extract data for (in seconds)
            compression (str): Compression method for HDF5 ('gzip', 'lzf', 'szip', or None)
        """
        activity_matrix, timestamps = self.get_activity_matrix(time_window)
        
        if activity_matrix.size == 0:
            print("Warning: No activity data to save!")
            return
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save to HDF5 file
        with h5py.File(path, 'w') as f:
            # Save the main activity matrix
            f.create_dataset('activity_matrix', 
                            data=activity_matrix, 
                            compression=compression,
                            chunks=True)
            
            # Save timestamps
            f.create_dataset('timestamps', 
                            data=timestamps, 
                            compression=compression)
            
            # Save metadata
            metadata_group = f.create_group('metadata')
            metadata_group.attrs['eid'] = self.eid
            metadata_group.attrs['subject'] = self.subject
            metadata_group.attrs['date'] = self.date
            metadata_group.attrs['duration_hours'] = self.duration_hours
            metadata_group.attrs['task_protocol'] = self.task_protocol
            metadata_group.attrs['n_fovs'] = self.n_fovs
            metadata_group.attrs['n_total_neurons'] = self.n_total_neurons
            metadata_group.attrs['time_window'] = time_window if time_window is not None else "full_session"
            metadata_group.attrs['shape_description'] = "activity_matrix: [time_points, neurons], timestamps: [time_points]"
            
            # Save FOV information
            fov_group = f.create_group('fov_info')
            neuron_count = 0
            for fov_name, fov in self.fovs.items():
                fov_subgroup = fov_group.create_group(fov_name)
                fov_subgroup.attrs['collection'] = fov.collection
                fov_subgroup.attrs['n_neurons'] = fov.n_neurons
                fov_subgroup.attrs['n_rois'] = fov.n_rois
                fov_subgroup.attrs['neuron_start_index'] = neuron_count
                fov_subgroup.attrs['neuron_end_index'] = neuron_count + fov.n_neurons
                neuron_count += fov.n_neurons
        
        print(f"Activity matrix saved to {path}")
        print(f"Shape: {activity_matrix.shape} (time_points x neurons)")
        print(f"File size: {os.path.getsize(path) / (1024**2):.2f} MB")
    
    def plot_binary_activity_heatmap(self, time_window: float = 300, threshold: float = 0.2):
        """Plot raster of neural activity for all FOVs"""
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Count total neurons to set up plot
        total_neurons = self.n_total_neurons
        
        if total_neurons == 0:
            ax.text(0.5, 0.5, "No neurons found in the data!", 
                  ha='center', va='center', transform=ax.transAxes)
            return fig

        # Keep track of the neuron starting index for each FOV
        neuron_start_index = 0
        
        # Process each FOV
        for fov_name, fov in self.fovs.items():
            # Get neuron activity
            neuron_activity, timestamps, neuron_mask = fov.get_neuron_activity(time_window)
            n_neurons = neuron_activity.shape[1]
            
            if n_neurons == 0:
                continue
                
            # Normalize activity per neuron
            normalized_data = np.zeros_like(neuron_activity)
            for r in range(neuron_activity.shape[1]):
                roi_max = np.max(neuron_activity[:, r])
                if roi_max > 0:  # Avoid division by zero
                    normalized_data[:, r] = neuron_activity[:, r] / roi_max
            
            # Find activity above threshold
            binary_activity = normalized_data > threshold
            
            # Skip if no activity above threshold
            if not np.any(binary_activity):
                neuron_start_index += n_neurons  # Still increment the index
                continue
            
            # Find active points
            active_points = np.where(binary_activity.T)
            
            if len(active_points) >= 2 and len(active_points[0]) > 0:
                active_neurons = active_points[0] + neuron_start_index
                active_time_indices = active_points[1]
                
                # Plot raster
                ax.scatter(
                    timestamps[active_time_indices], 
                    active_neurons,
                    s=1, 
                    color='green', 
                    alpha=0.5, 
                    marker='|'
                )
            
            # Update neuron start index
            neuron_start_index += n_neurons
        
        # Add labels and title
        ax.set_title(f"Neuron Activity Raster ({total_neurons} neurons, threshold={threshold:.2f})")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Neuron index")
        ax.set_ylim(-1, total_neurons)
        
        # Add horizontal lines to separate FOVs
        neuron_count = 0
        for fov_name, fov in self.fovs.items():
            n_neurons = fov.n_neurons
            neuron_count += n_neurons
            if neuron_count > 0:
                ax.axhline(neuron_count, color='black', linestyle='--', alpha=0.3)
        
        # Add FOV labels
        neuron_count = 0
        for fov_name, fov in self.fovs.items():
            n_neurons = fov.n_neurons
            if n_neurons > 0:
                middle_pos = neuron_count + n_neurons / 2
                ax.text(-30, middle_pos, fov_name, 
                       verticalalignment='center', fontsize=10)
                neuron_count += n_neurons
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig

# Helper functions
def _calculate_duration(session_info) -> float:
    """Calculate session duration in hours"""
    from datetime import datetime
    
    session_start = session_info.get('start_time')
    session_end = session_info.get('end_time')
    
    if session_start and session_end:
        try:
            if isinstance(session_start, str):
                start_dt = datetime.fromisoformat(session_start.replace('Z', '+00:00'))
            else:
                start_dt = session_start
                
            if isinstance(session_end, str):
                end_dt = datetime.fromisoformat(session_end.replace('Z', '+00:00'))
            else:
                end_dt = session_end
                
            duration_seconds = (end_dt - start_dt).total_seconds()
            return duration_seconds / 3600
        except Exception:
            pass
    
    return 0.0

def _load_fov_data(one: ONE, eid: str, raw_activity: bool, collection: str) -> Optional[FOVData]:
    """Load data for a single FOV"""
    try:
        # Load the key datasets
        if raw_activity:
            roi_activity = one.load_dataset(eid, 'mpci.ROIActivityF', collection=collection)        
        else:
            roi_activity = one.load_dataset(eid, 'mpci.ROIActivityDeconvolved', collection=collection)
        timestamps = one.load_dataset(eid, 'mpci.times', collection=collection)
        bad_frames = one.load_dataset(eid, 'mpci.badFrames', collection=collection)
        frame_qc = one.load_dataset(eid, 'mpci.mpciFrameQC', collection=collection)
        
        # Try to load ROI types
        roi_types = None
        try:
            roi_types = one.load_dataset(eid, 'mpciROIs.mpciROITypes', collection=collection)
        except Exception:
            pass
        
        # Try to load brain locations
        brain_locations = None
        try:
            brain_locations = one.load_dataset(eid, 'mpciROIs.brainLocationIds_ccf_2017_estimate', collection=collection)
        except Exception:
            pass
        
        # Create FOV data object
        fov_name = collection.split('/')[-1]
        return FOVData(
            fov_name=fov_name,
            collection=collection,
            roi_activity=roi_activity,
            timestamps=timestamps,
            bad_frames=bad_frames,
            frame_qc=frame_qc,
            roi_types=roi_types,
            brain_locations=brain_locations
        )
    except Exception as e:
        print(f"Error loading FOV data for {collection}: {str(e)}")
        return None