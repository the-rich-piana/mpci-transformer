import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from load_meso_session import MesoscopeSession
from one.api import ONE

class CalciumDataPreprocessor:
    """
    Preprocessing pipeline for calcium imaging data
    Handles ΔF/F calculation, outlier removal, neuron filtering, and normalization
    """
    
    def __init__(self, neucoeff=0.7, temporal_smoothing=True):
        self.neucoeff = neucoeff
        self.temporal_smoothing = temporal_smoothing
        self.preprocessing_stats = {}
        
    def preprocess_session(self, session: MesoscopeSession, 
                          output_path: str = None,
                          save_intermediates: bool = False) -> dict:
        """
        Complete preprocessing pipeline for a mesoscope session
        
        Args:
            session: MesoscopeSession object
            output_path: Path to save processed data
            save_intermediates: Whether to save intermediate processing steps
            
        Returns:
            dict: Contains processed data and metadata
        """
        print("="*80)
        print("CALCIUM IMAGING PREPROCESSING PIPELINE")
        print("="*80)
        
        # Step 1: Load ΔF/F data
        print("\n1. Loading ΔF/F data...")
        dff_matrix, timestamps = session.get_dff_activity_matrix(neucoeff=self.neucoeff)
        
        print(f"   Raw ΔF/F shape: {dff_matrix.shape}")
        print(f"   Raw ΔF/F range: {dff_matrix.min():.1f} to {dff_matrix.max():.1f}")
        
        # Step 2: Filter problematic neurons
        print("\n2. Filtering neurons...")
        filtered_data, neuron_mask = self._filter_neurons(dff_matrix)
        
        print(f"   Kept {neuron_mask.sum()} neurons out of {len(neuron_mask)}")
        print(f"   Filtered data range: {filtered_data.min():.1f} to {filtered_data.max():.1f}")
        
        # Step 3: Robust normalization
        print("\n3. Applying robust normalization...")
        normalized_data, norm_params = self._robust_normalize(filtered_data)
        
        print(f"   Normalized range: {normalized_data.min():.3f} to {normalized_data.max():.3f}")
        print(f"   Normalized mean: {normalized_data.mean():.3f}, std: {normalized_data.std():.3f}")
        
        # Step 4: Optional temporal smoothing
        if self.temporal_smoothing:
            print("\n4. Applying temporal smoothing...")
            smoothed_data = self._temporal_smooth(normalized_data)
            final_data = smoothed_data
        else:
            final_data = normalized_data
            
        # Step 5: Calculate aligned wheel velocity
        print("\n5. Calculating aligned wheel velocity...")
        aligned_wheel_velocity = session.get_aligned_wheel_velocity(timestamps)
        print(f"   Wheel velocity range: {aligned_wheel_velocity.min():.3f} to {aligned_wheel_velocity.max():.3f} rad/s")
        print(f"   Non-zero velocity samples: {np.sum(aligned_wheel_velocity != 0)}/{len(aligned_wheel_velocity)}")
        
        # Step 6: Quality assessment
        print("\n6. Quality assessment...")
        quality_metrics = self._assess_quality(dff_matrix, final_data, neuron_mask)
        
        # Compile results
        results = {
            'processed_data': final_data,
            'timestamps': timestamps,
            'neuron_mask': neuron_mask,
            'normalization_params': norm_params,
            'quality_metrics': quality_metrics,
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
                'aligned_wheel_velocity': aligned_wheel_velocity
            },
            'session_metadata': {
                'eid': session.eid,
                'subject': session.subject,
                'date': session.date,
                'duration_seconds': session.duration_seconds,
                'duration_hours': session.duration_hours,
                'task_protocol': session.task_protocol,
                'n_original_neurons': session.n_total_neurons,
                'n_processed_neurons': neuron_mask.sum(),
                'neucoeff': self.neucoeff,
                'temporal_smoothing': self.temporal_smoothing
            }
        }
        
        # Save if requested
        if output_path:
            self._save_processed_data(results, output_path, save_intermediates)
            
        return results
    
    def _filter_neurons(self, dff_matrix):
        """Filter out problematic neurons based on statistical criteria"""
        n_frames, n_neurons = dff_matrix.shape
        
        # Calculate per-neuron statistics
        neuron_medians = np.median(dff_matrix, axis=0)
        neuron_q99 = np.percentile(dff_matrix, 99, axis=0)
        neuron_q01 = np.percentile(dff_matrix, 1, axis=0)
        neuron_iqr = np.percentile(dff_matrix, 75, axis=0) - np.percentile(dff_matrix, 25, axis=0)
        neuron_stds = np.std(dff_matrix, axis=0)
        
        # Filtering criteria
        # 1. Must have reasonable variability (not flat)
        min_iqr_threshold = np.percentile(neuron_iqr[neuron_iqr > 0], 5)
        
        # 2. Must not have extreme outliers (likely artifacts)
        max_range_threshold = np.percentile(neuron_q99 - neuron_q01, 95)
        
        # 3. Must have reasonable baseline (not too negative)
        min_median_threshold = np.percentile(neuron_medians, 1)
        
        # 4. Must have reasonable standard deviation
        max_std_threshold = np.percentile(neuron_stds, 99)
        
        # 5. Must not be predominantly negative (calcium should be mostly positive)
        negative_fraction = np.mean(dff_matrix < 0, axis=0)
        max_negative_fraction = 0.5  # Allow up to 50% negative values
        
        # Apply all criteria
        neuron_mask = (
            (neuron_iqr > min_iqr_threshold) & 
            ((neuron_q99 - neuron_q01) < max_range_threshold) & 
            (neuron_medians > min_median_threshold) &
            (neuron_stds < max_std_threshold) &
            (negative_fraction < max_negative_fraction)
        )
        
        # Store filtering stats
        self.preprocessing_stats['filtering'] = {
            'n_original': n_neurons,
            'n_kept': neuron_mask.sum(),
            'filter_rate': neuron_mask.sum() / n_neurons,
            'min_iqr_threshold': min_iqr_threshold,
            'max_range_threshold': max_range_threshold,
            'min_median_threshold': min_median_threshold,
            'max_std_threshold': max_std_threshold,
            'max_negative_fraction': max_negative_fraction
        }
        
        return dff_matrix[:, neuron_mask], neuron_mask
    
    def _robust_normalize(self, data):
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
        normalized = np.clip(normalized, 0, 1)
        
        # Store normalization parameters
        norm_params = {
            'q005': q005,
            'q995': q995,
            'data_range': data_range,
            'method': 'robust_percentile_0.5_99.5'
        }
        
        self.preprocessing_stats['normalization'] = {
            'method': 'robust_percentile_0.5_99.5',
            'final_range': (normalized.min(), normalized.max()),
            'final_mean': normalized.mean(),
            'final_std': normalized.std(),
            'clipping_fraction': np.mean((normalized == 0) | (normalized == 1))
        }
        
        return normalized, norm_params
    
    def _temporal_smooth(self, data, sigma=1.0):
        """Apply mild temporal smoothing to reduce noise"""
        # Very mild Gaussian smoothing along time axis
        smoothed = gaussian_filter1d(data, sigma=sigma, axis=0)
        
        self.preprocessing_stats['temporal_smoothing'] = {
            'applied': True,
            'sigma': sigma,
            'smoothing_effect': np.mean(np.abs(data - smoothed))
        }
        
        return smoothed
    
    def _assess_quality(self, original_data, processed_data, neuron_mask):
        """Assess quality of preprocessing"""
        metrics = {
            'original_shape': original_data.shape,
            'processed_shape': processed_data.shape,
            'neuron_retention_rate': neuron_mask.sum() / len(neuron_mask),
            'original_range': (original_data.min(), original_data.max()),
            'processed_range': (processed_data.min(), processed_data.max()),
            'original_sparsity': np.mean(np.abs(original_data) < 0.01),
            'processed_sparsity': np.mean(processed_data < 0.01),
            'original_outlier_fraction': np.mean(np.abs(original_data) > np.percentile(np.abs(original_data), 99)),
            'processed_outlier_fraction': np.mean(processed_data > 0.99),
        }
        
        return metrics
    
    def _save_processed_data(self, results, output_path, save_intermediates=False):
        """Save processed data to HDF5 file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n6. Saving processed data to {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # Main processed data
            f.create_dataset('processed_data', data=results['processed_data'], 
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
            
            # Normalization parameters
            norm_group = f.create_group('normalization')
            for key, value in results['normalization_params'].items():
                if isinstance(value, np.ndarray):
                    norm_group.create_dataset(key, data=value)
                else:
                    norm_group.attrs[key] = value
            
            # Metadata
            meta_group = f.create_group('metadata')
            for key, value in results['session_metadata'].items():
                meta_group.attrs[key] = value
            
            # Quality metrics
            quality_group = f.create_group('quality_metrics')
            for key, value in results['quality_metrics'].items():
                if isinstance(value, tuple):
                    quality_group.attrs[key] = list(value)
                else:
                    quality_group.attrs[key] = value
            
            # Preprocessing stats
            stats_group = f.create_group('preprocessing_stats')
            for stage, stats in self.preprocessing_stats.items():
                stage_group = stats_group.create_group(stage)
                for key, value in stats.items():
                    if isinstance(value, np.ndarray):
                        stage_group.create_dataset(key, data=value)
                    else:
                        stage_group.attrs[key] = value
        
        print(f"   Saved {results['processed_data'].shape} processed data matrix")
        print(f"   File size: {output_path.stat().st_size / (1024**2):.2f} MB")

# Usage example
def preprocess_session_batch(csv_path: str, output_dir: str, session_indices: list = None):
    """Process multiple sessions from CSV"""
    one = ONE()
    preprocessor = CalciumDataPreprocessor(neucoeff=0.7, temporal_smoothing=True)
    
    # Read session list
    sessions_df = pd.read_csv(csv_path)
    
    if session_indices is None:
        session_indices = range(len(sessions_df))
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_summary = []
    
    for idx in session_indices:
        try:
            print(f"\n{'='*50}")
            print(f"Processing session {idx+1}/{len(sessions_df)}")
            print(f"{'='*50}")
            
            # Load session
            eid = sessions_df.iloc[idx]['eid']
            session = MesoscopeSession.from_eid(one, eid, True)
            
            # Process
            output_path = output_dir / f"session_{idx:03d}_processed.h5"
            results = preprocessor.preprocess_session(session, output_path)
            
            # Store summary
            summary = {
                'session_idx': idx,
                'eid': session.eid,
                'subject': session.subject,
                'n_original_neurons': results['session_metadata']['n_original_neurons'],
                'n_processed_neurons': results['session_metadata']['n_processed_neurons'],
                'retention_rate': results['quality_metrics']['neuron_retention_rate'],
                'final_shape': results['processed_data'].shape,
                'output_path': str(output_path)
            }
            results_summary.append(summary)
            
        except Exception as e:
            print(f"Error processing session {idx}: {e}")
            continue
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(output_dir / 'preprocessing_summary.csv', index=False)
    
    print(f"\n{'='*50}")
    print("PREPROCESSING COMPLETE")
    print(f"Processed {len(results_summary)} sessions")
    print(f"Results saved to {output_dir}")
    print(f"{'='*50}")
    
    return results_summary

# # Example usage:
# if __name__ == "__main__":
#     # Process a single session
#     one = ONE()
#     preprocessor = CalciumDataPreprocessor(neucoeff=0.7, temporal_smoothing=True)
    
#     # Load session by EID
#     eid = "your_eid_here"  # Replace with actual EID
#     session = MesoscopeSession.from_eid(one, eid, True)
#     results = preprocessor.preprocess_session(session, 'processed_data/session_000.h5')
    
#     # Or process multiple sessions
#     # summary = preprocess_session_batch(
#     #     '../good_mesoscope_sessions_final.csv', 
#     #     'processed_data/',
#     #     session_indices=[0, 1, 2]  # Process first 3 sessions
#     # )