import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from load_meso_session import MesoscopeSession
from covariate_builder import build_covariate_matrix, get_covariate_descriptions
from one.api import ONE

class CalciumDataPreprocessorPseudo:
    """
    Preprocessing pipeline for calcium imaging data
    Handles fluorescence data, outlier removal, neuron filtering, and normalization
    """
    
    def __init__(self):
        self.preprocessing_stats = {}
        
    def preprocess_session(self, session: MesoscopeSession, 
                          output_path: str = None,
                          top_n_neurons: int = 5000,
                          include_passive: bool = False,
                          save_intermediates: bool = False,
                          ) -> dict:
        """
        Complete preprocessing pipeline for a mesoscope session
        
        Args:
            session: MesoscopeSession object
            output_path: Path to save processed data
            top_n_neurons: Number of top neurons to select (default: 5000)
            save_intermediates: Whether to save intermediate processing steps
            
        Returns:
            dict: Contains processed data and metadata
        """
        print("="*80)
        print("CALCIUM IMAGING PREPROCESSING PIPELINE")
        print("="*80)
        
        # Step 1: Load ΔF/F data
        print("\n1. Loading fluorescence data...")
        pseudo_matrix, timestamps = session.get_pseudo_activity_matrix()
        print(f"MEAN: {pseudo_matrix.mean()}")
        
        print(f"   Raw Pseudo shape: {pseudo_matrix.shape}")
        print(f"   Raw Pseudo range: {pseudo_matrix.min():.1f} to {pseudo_matrix.max():.1f}")
        
        
        if include_passive:
            # Step 1.5: Keep entire time series including passive period
            print("\n1.5. Keeping entire time series (including passive period)...")
            original_duration = timestamps[-1] - timestamps[0]
            truncated_duration = original_duration
            
            print(f"   Keeping full duration: {original_duration:.1f}s")
            print(f"   Final shape: {pseudo_matrix.shape}")
            
        else:
            # Step 1.5: Truncate to remove passive video period
            print("\n1.5. Truncating passive video period...")
            last_stimoff = session.stimOff_times[-1]
            cutoff_idx = np.searchsorted(timestamps, last_stimoff, side='right')
            original_duration = timestamps[-1] - timestamps[0]
            
            pseudo_matrix = pseudo_matrix[:cutoff_idx]
            timestamps = timestamps[:cutoff_idx]
            truncated_duration = timestamps[-1] - timestamps[0]
            
            print(f"   Last stimOff at {last_stimoff:.1f}s, truncating at index {cutoff_idx}")
            print(f"   Truncated shape: {pseudo_matrix.shape}")
            print(f"   Duration: {original_duration:.1f}s → {truncated_duration:.1f}s ({truncated_duration/original_duration:.1%} retained)")
        # Step 3: Select first N neurons (no filtering, just selection)
        print(f"\n3. Selecting first {top_n_neurons} neurons...")
        if top_n_neurons is not None and top_n_neurons < pseudo_matrix.shape[1]:
            # Get indices of first N neurons
            top_n_indices = np.arange(top_n_neurons)
            
            # Select final data to be processed
            pseudo_matrix = pseudo_matrix[:, :top_n_neurons]
            print(f"Selected neurons mean: {pseudo_matrix.mean()}")
            
            # Create final mask
            final_neuron_mask = np.zeros(pseudo_matrix.shape[1], dtype=bool)
            final_neuron_mask[top_n_indices] = True
            
            print(f"   Selected first {top_n_neurons} neurons from {pseudo_matrix.shape[1]} total neurons")
            print(f"   Selection method: first N neurons")
            final_data = pseudo_matrix
            
        else:
            # Keep all neurons if top_n not specified or larger than available
            final_data = pseudo_matrix
            final_neuron_mask = np.ones(pseudo_matrix.shape[1], dtype=bool)
            print(f"   Keeping all {pseudo_matrix.shape[1]} neurons (top_n={top_n_neurons})")
        
        
        final_data = final_data
            
        # Step 5: Calculate aligned wheel velocity
        print("\n5. Calculating aligned wheel velocity...")
        aligned_wheel_velocity = session.get_aligned_wheel_velocity(timestamps)
        print(f"   Wheel velocity range: {aligned_wheel_velocity.min():.3f} to {aligned_wheel_velocity.max():.3f} rad/s")
        print(f"   Non-zero velocity samples: {np.sum(aligned_wheel_velocity != 0)}/{len(aligned_wheel_velocity)}")
        
        # Step 6: Build covariate matrix
        print("\n6. Building covariate matrix...")
        covariate_matrix, covariate_feature_names = build_covariate_matrix(session, timestamps, aligned_wheel_velocity)
        print(f"   Covariate matrix shape: {covariate_matrix.shape}")
        print(f"   Features: {len(covariate_feature_names)} total")
        print(f"   Non-zero covariate samples: {np.sum(covariate_matrix != 0, axis=0)}")
        
        # Step 7: Quality assessment
        print("\n7. Quality assessment...")
        
        # Compile results
        results = {
            'activity': final_data,
            'covariate_matrix': covariate_matrix,
            'timestamps': timestamps,
            'neuron_mask': final_neuron_mask,
            'trial_data': {
                'stimOn_times': session.stimOn_times,
                'stimOff_times': session.stimOff_times,                
                'feedback_times': session.feedback_times,
                'wheel_position': session.wheel_position,
                'response_times': session.response_times,
                'contrastLeft': session.contrastLeft,
                'contrastRight': session.contrastRight,
                'feedbackType': session.feedbackType,
                'intervals': session.intervals,
                'wheel_timestamps': session.wheel_timestamps,
                'aligned_wheel_velocity': aligned_wheel_velocity,
                'covariate_feature_names': covariate_feature_names
            },
            'session_metadata': {
                'eid': session.eid,
                'subject': session.subject,
                'date': session.date,
                'duration_seconds': truncated_duration,
                'duration_hours': truncated_duration / 3600,
                'task_protocol': session.task_protocol,
                'n_original_neurons': session.n_total_neurons,
            }
        }
        
        # Save if requested
        if output_path:
            self._save_processed_data(results, output_path, save_intermediates)
            
        return results
    
    def _save_processed_data(self, results, output_path, save_intermediates=False):
        """Save processed data to HDF5 file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n6. Saving processed data to {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # Main neural activity data
            f.create_dataset('activity', data=results['activity'], 
                           compression='gzip', chunks=True)
            
            # Covariate matrix as top-level dataset
            f.create_dataset('covariate_matrix', data=results['covariate_matrix'],
                           compression='gzip', chunks=True)
            f.create_dataset('timestamps', data=results['timestamps'], 
                           compression='gzip')
            f.create_dataset('neuron_mask', data=results['neuron_mask'])
            
            # Trial data
            trial_group = f.create_group('trial_data')
            trial_group.create_dataset('stimOn_times', data=results['trial_data']['stimOn_times'], 
                                      compression='gzip')
            trial_group.create_dataset('stimOff_times', data=results['trial_data']['stimOff_times'], 
                                      compression='gzip')
            trial_group.create_dataset('feedback_times', data=results['trial_data']['feedback_times'], 
                                      compression='gzip')
            trial_group.create_dataset('wheel_position', data=results['trial_data']['wheel_position'], 
                                      compression='gzip')
            trial_group.create_dataset('response_times', data=results['trial_data']['response_times'], 
                                      compression='gzip')
            trial_group.create_dataset('contrastLeft', data=results['trial_data']['contrastLeft'], 
                                      compression='gzip')
            trial_group.create_dataset('contrastRight', data=results['trial_data']['contrastRight'], 
                                      compression='gzip')
            trial_group.create_dataset('feedbackType', data=results['trial_data']['feedbackType'], 
                                      compression='gzip')
            trial_group.create_dataset('intervals', data=results['trial_data']['intervals'], 
                                      compression='gzip')
            trial_group.create_dataset('aligned_wheel_velocity', data=results['trial_data']['aligned_wheel_velocity'], 
                                      compression='gzip')
            trial_group.create_dataset('wheel_timestamps', data=results['trial_data']['wheel_timestamps'], 
                                      compression='gzip')
            
            # Covariate metadata
            covariate_group = f.create_group('covariate_metadata')
            covariate_group.create_dataset('feature_names', data=[name.encode('utf-8') for name in results['trial_data']['covariate_feature_names']])
            
            # Add feature descriptions as attributes
            feature_descriptions = get_covariate_descriptions()
            for feature_name, description in feature_descriptions.items():
                covariate_group.attrs[f'{feature_name}_description'] = description
                        
            # Metadata
            meta_group = f.create_group('metadata')
            for key, value in results['session_metadata'].items():
                meta_group.attrs[key] = value
            
            # Preprocessing stats
            stats_group = f.create_group('preprocessing_stats')
            for stage, stats in self.preprocessing_stats.items():
                stage_group = stats_group.create_group(stage)
                for key, value in stats.items():
                    if isinstance(value, np.ndarray):
                        stage_group.create_dataset(key, data=value)
                    else:
                        stage_group.attrs[key] = value
        
        print(f"   Saved {results['activity'].shape} activity matrix")
        print(f"   Saved {results['covariate_matrix'].shape} covariate matrix")
        print(f"   File size: {output_path.stat().st_size / (1024**2):.2f} MB")
