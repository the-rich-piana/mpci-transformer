from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from one.api import ONE
import numpy.typing as npt
import h5py
import os
import brainbox.behavior.wheel as wh


class FOVData(BaseModel):
    """Data for a single Field of View (FOV)"""
    fov_name: str
    collection: str
    neuropil_activity: Optional[np.ndarray] = Field(None, description="Neuropil activity matrix [frames, ROIs]")    
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
    
    def compute_dff(self, neucoeff: float = 0.7) -> np.ndarray:
        """
        Compute ΔF/F using Suite2p method: F - neucoeff * Fneu
        
        Args:
            neucoeff: Neuropil coefficient (default 0.7 as per Suite2p)
            
        Returns:
            ΔF/F matrix [frames, ROIs]
        """
        if self.neuropil_activity is None:
            raise ValueError(f"Neuropil activity not available for {self.fov_name}")
            
        # Suite2p ΔF/F calculation
        f_corrected = self.roi_activity - neucoeff * self.neuropil_activity
        
        # Compute baseline (using simple percentile method)
        # You can make this more sophisticated later
        f0 = np.percentile(f_corrected, 10, axis=0, keepdims=True)
        f0 = np.maximum(f0, 1.0)  # Avoid division by very small numbers
        
        # Calculate ΔF/F
        dff = (f_corrected - f0) / f0
        
        return dff
    
    def get_dff_activity(self, time_window: Optional[float] = None, neucoeff: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """Get ΔF/F activity filtered by good frames and time window"""
        # Compute ΔF/F
        dff_activity = self.compute_dff(neucoeff=neucoeff)
        
        # Filter by good frames
        good_mask = self.good_frame_mask
        filtered_dff = dff_activity[good_mask, :]
        filtered_timestamps = self.timestamps[good_mask]
        
        # Apply time window if specified
        if time_window is not None:
            time_limit = filtered_timestamps[0] + time_window
            time_mask = filtered_timestamps <= time_limit
            filtered_dff = filtered_dff[time_mask, :]
            filtered_timestamps = filtered_timestamps[time_mask]
            
        return filtered_dff, filtered_timestamps
    
    def get_neuron_dff_activity(self, time_window: Optional[float] = None, neucoeff: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ΔF/F activity only for neurons, filtered by good frames and time window"""
        filtered_dff, filtered_timestamps = self.get_dff_activity(time_window, neucoeff)
        
        # Filter by ROI type
        if self.roi_types is not None:
            neuron_mask = np.array(self.roi_types) == 1
            neuron_dff = filtered_dff[:, neuron_mask]
            return neuron_dff, filtered_timestamps, neuron_mask
        
        # If no ROI types available, return empty arrays
        return np.array([]).reshape(filtered_dff.shape[0], 0), filtered_timestamps, np.array([])

class MesoscopeSession(BaseModel):
    """Data for a complete mesoscope session with multiple FOVs"""
    eid: str
    subject: str
    date: str
    duration_hours: float
    duration_seconds: float
    raw_activity: bool
    task_protocol: str
    fovs: Dict[str, FOVData] = Field(default_factory=dict)
    stimOn_times: np.ndarray = Field(..., description="Stimulus onset times")
    stimOff_times: np.ndarray = Field(..., description="Time in seconds, relative to the session start, of the stimulus offset, as recorded by an external photodiode.")
    feedback_times: np.ndarray = Field(..., description="Feedback times")
    wheel_position: np.ndarray = Field(..., description="Wheel position data")
    response_times: np.ndarray = Field(..., description="Times at  which response was recorded for each trial")
    contrastLeft: np.ndarray = Field(..., description="The contrast of the stimulus that appears on the left side of the screen (-35º azimuth).  When there is a non-zero contrast on the right, contrastLeft == 0, when there is no contrast on either side (a catch trial), contrastLeft == NaN.")
    contrastRight: np.ndarray = Field(..., description="The contrast of the stimulus that appears on the right side of the screen (-35º azimuth).  When there is a non-zero contrast on the right, contrastLeft == 0, when there is no contrast on either side (a catch trial), contrastRight == NaN.")
    feedbackType: np.ndarray = Field(..., description="Feedback type (+1 correct, -1 incorrect)")
    intervals: np.ndarray = Field(..., description="Trial start/end times [n_trials, 2]")
    wheel_timestamps: np.ndarray = Field(..., description="Wheel event timestamps (irregular sampling)")
    aligned_wheel_velocity: Optional[np.ndarray] = Field(None, description="Wheel velocity aligned to neural timestamps")
    
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
    
    def get_aligned_wheel_velocity(self, neural_timestamps: np.ndarray) -> np.ndarray:
        """
        Get wheel velocity aligned to neural timestamps.
        Uses 0 velocity where no wheel data exists (no movement).
        
        Args:
            neural_timestamps: Neural activity timestamps to align to
            
        Returns:
            Aligned wheel velocity array matching neural timestamps
        """
        from scipy.interpolate import interp1d
        
        # Calculate wheel velocity at appropriate frequency
        Fs = 5  # Match typical neural sampling rate
        pos, t = wh.interpolate_position(self.wheel_timestamps, self.wheel_position, freq=Fs)
        vel, acc = wh.velocity_filtered(pos, Fs, corner_frequency=2)
        
        # Clip extreme velocity outliers
        vel_clipped = np.clip(vel, -2.5, 2.5)
        
        # Initialize with zeros (no movement = zero velocity)
        aligned_velocity = np.zeros(len(neural_timestamps))
        
        # Find neural timestamps that overlap with wheel recording period
        wheel_start = t[0]
        wheel_end = t[-1]
        
        # Create mask for neural timestamps within wheel recording period
        overlap_mask = (neural_timestamps >= wheel_start) & (neural_timestamps <= wheel_end)
        
        if np.any(overlap_mask):
            # Interpolate wheel velocity for overlapping timestamps only
            interp_func = interp1d(t, vel_clipped, 
                                  kind='linear', bounds_error=False, fill_value=0)
            aligned_velocity[overlap_mask] = interp_func(neural_timestamps[overlap_mask])
        
        return aligned_velocity
        
    @classmethod
    def from_eid(cls, one: ONE, eid: str, raw_activity: bool = True, max_fovs: Optional[int] = None) -> 'MesoscopeSession':
        """Load mesoscope session data from experiment ID"""
        # Get session details
        session_info = one.get_details(eid, full=True)
        print(session_info)
        
        # Load trial data
        trials = one.load_object(eid, 'trials',)
        stimOn_times = trials['stimOn_times']
        stimOff_times = trials['stimOff_times']
        feedback_times = trials['feedback_times']
        response_times = trials['response_times']
        contrastLeft = trials['contrastLeft']
        contrastRight = trials['contrastRight']
        feedbackType = trials['feedbackType']
        intervals = trials['intervals']
        
        # Load wheel data
        wheel = one.load_object(eid, 'wheel')
        wheel_position = wheel['position']
        wheel_timestamps = wheel['timestamps']
        
        duration_hours, duration_seconds = _calculate_duration(session_info)
        # Create session object
        session = cls(
            eid=eid,
            raw_activity=raw_activity,
            subject=session_info.get('subject', 'unknown'),
            date=str(session_info.get('start_time', 'unknown')),
            duration_seconds=duration_seconds,
            duration_hours=duration_hours,
            task_protocol=session_info.get('task_protocol', ''),
            stimOn_times=stimOn_times,
            stimOff_times=stimOff_times,
            feedback_times=feedback_times,
            wheel_position=wheel_position,
            wheel_timestamps=wheel_timestamps,
            response_times=response_times,
            contrastLeft=contrastLeft,
            contrastRight=contrastRight,
            feedbackType=feedbackType,
            aligned_wheel_velocity=None,
            intervals=intervals,
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
    def from_preprocessed(cls, path: str) -> 'MesoscopeSession':
        """Load mesoscope session from preprocessed HDF5 file
        
        Args:
            path (str): Path to the preprocessed HDF5 file
            
        Returns:
            MesoscopeSession: Session object with preprocessed data loaded
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessed HDF5 file not found: {path}")
        
        with h5py.File(path, 'r') as f:
            # Load metadata
            metadata = {}
            for key in f['metadata'].attrs.keys():
                value = f['metadata'].attrs[key]
                if isinstance(value, bytes):
                    metadata[key] = value.decode('utf-8')
                else:
                    metadata[key] = value
            
            # Load trial data
            trial_data = {}
            if 'trial_data' in f:
                for key in f['trial_data'].keys():
                    trial_data[key] = f['trial_data'][key][:]
            # Create session object with empty FOVs
            session = cls(
                eid=metadata['eid'],
                subject=metadata['subject'], 
                date=metadata['date'],
                duration_hours=metadata['duration_hours'],
                task_protocol=metadata['task_protocol'],
                duration_seconds=metadata['duration_seconds'],
                raw_activity=False,  # Preprocessed data is not raw
                fovs={},  # No FOVs for preprocessed data
                stimOn_times=trial_data.get('stimOn_times', np.array([])),
                stimOff_times=trial_data.get('stimOff_times', np.array([])),                
                feedback_times=trial_data.get('feedback_times', np.array([])),
                wheel_position=trial_data.get('wheel_position', np.array([])),
                response_times=trial_data.get('response_times', np.array([])),
                contrastLeft=trial_data.get('contrastLeft', np.array([])),
                contrastRight=trial_data.get('contrastRight', np.array([])),
                feedbackType=trial_data.get('feedbackType', np.array([])),
                intervals=trial_data.get('intervals', np.array([])),
                wheel_timestamps=trial_data.get('wheel_timestamps', np.array([])),
                aligned_wheel_velocity=trial_data.get('aligned_wheel_velocity', np.array([])),
            )
            
            # Store preprocessed data as instance attributes
            session._activity = f['activity'][:]
            session._timestamps = f['timestamps'][:]
            session._neuron_mask = f['neuron_mask'][:]
            
            # Store normalization parameters
            session._normalization_params = {}
            if 'normalization' in f:
                for key in f['normalization'].keys():
                    session._normalization_params[key] = f['normalization'][key][:]
                for key in f['normalization'].attrs.keys():
                    session._normalization_params[key] = f['normalization'].attrs[key]
        
        print(f"Preprocessed session loaded from {path}")
        print(f"Shape: {session._activity.shape} (time_points x neurons)")
        print(f"EID: {session.eid}, Subject: {session.subject}")
        
        return session
    
    def get_preprocessed_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get preprocessed data (only available for sessions loaded from preprocessed files)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (activity, timestamps)
        """
        if not hasattr(self, '_activity'):
            raise ValueError("No preprocessed data available. Use from_preprocessed() to load preprocessed sessions.")
        
        return self._activity, self._timestamps
    
    @classmethod
    def load_activity_matrix(cls, path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Union[str, float, int]]]:
        """Load activity matrix from HDF5 file
        
        Args:
            path (str): Path to the HDF5 file
            
        Returns:
            Tuple[np.ndarray, np.ndarray, Dict]: (activity_matrix, timestamps, metadata)
            - activity_matrix: Neural activity matrix [time_points, neurons]
            - timestamps: Time points [time_points]
            - metadata: Dictionary containing session metadata
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"HDF5 file not found: {path}")
        
        with h5py.File(path, 'r') as f:
            # Load the activity matrix and timestamps
            activity_matrix = f['activity_matrix'][:]
            timestamps = f['timestamps'][:]
            
            # Load metadata
            metadata = {}
            for key in f.attrs.keys():
                value = f.attrs[key]
                # Handle different data types
                if isinstance(value, bytes):
                    metadata[key] = value.decode('utf-8')
                elif isinstance(value, np.ndarray) and value.size == 1:
                    metadata[key] = value.item()
                else:
                    metadata[key] = value
        
        print(f"Activity matrix loaded from {path}")
        print(f"Shape: {activity_matrix.shape} (time_points x neurons)")
        print(f"Metadata: {metadata}")
        
        return activity_matrix, timestamps, metadata    
    @classmethod
    def load_session_from_hdf5(cls, path: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """Convenience method to load just the essential data from HDF5
        
        Args:
            path (str): Path to the HDF5 file
            
        Returns:
            Tuple[np.ndarray, np.ndarray, str]: (activity_matrix, timestamps, eid)
        """
        activity_matrix, timestamps, metadata = cls.load_activity_matrix(path)
        eid = metadata.get('eid', 'unknown')
        return activity_matrix, timestamps, eid
    
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
    
    def get_dff_activity_matrix(self, time_window: Optional[float] = None, neucoeff: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a combined ΔF/F matrix of all neuron activity across all FOVs
        
        Args:
            time_window: Time window to extract (in seconds)
            neucoeff: Neuropil coefficient for ΔF/F calculation
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (dff_matrix, timestamps)
        """
        # List to store neuron ΔF/F from each FOV
        all_neuron_dff = []
        common_timestamps = None
        
        # Process each FOV
        for fov_name, fov in self.fovs.items():
            # Get neuron ΔF/F activity
            neuron_dff, timestamps, _ = fov.get_neuron_dff_activity(time_window, neucoeff)
            
            # Skip if no neurons
            if neuron_dff.shape[1] == 0:
                continue
                
            # Store the neuron ΔF/F
            all_neuron_dff.append(neuron_dff)
            
            # Store the timestamps
            if common_timestamps is None and timestamps.size > 0:
                common_timestamps = timestamps
        
        # If no valid FOVs, return empty arrays
        if not all_neuron_dff or common_timestamps is None:
            return np.array([]), np.array([])
        
        # Verify all activities have the same number of time points
        expected_time_points = all_neuron_dff[0].shape[0]
        all_neuron_dff = [act for act in all_neuron_dff 
                         if act.shape[0] == expected_time_points]
        
        # Combine all neuron activities into a single matrix
        if all_neuron_dff:
            # Concatenate along the neuron dimension (axis=1)
            combined_dff = np.concatenate(all_neuron_dff, axis=1)
            return combined_dff, common_timestamps
        
        return np.array([]), common_timestamps
    

# Helper functions
def _calculate_duration(session_info) -> Tuple[float, float]:
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
            return duration_seconds / 3600, duration_seconds
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
        
        # Load neuropil activity
        neuropil_activity = None
        try:
            neuropil_activity = one.load_dataset(eid, 'mpci.ROINeuropilActivityF', collection=collection)
        except Exception as e:
            print(f"Warning: Could not load neuropil activity for {collection}: {e}")
            
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
            neuropil_activity=neuropil_activity,
            timestamps=timestamps,
            bad_frames=bad_frames,
            frame_qc=frame_qc,
            roi_types=roi_types,
            brain_locations=brain_locations
        )
    except Exception as e:
        print(f"Error loading FOV data for {collection}: {str(e)}")
        return None