import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from load_meso_session import MesoscopeSession
from covariate_builder import build_covariate_matrix, get_covariate_descriptions
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
        print("\n1. Loading ΔF/F data...")
        dff_matrix, timestamps = session.get_dff_activity_matrix(neucoeff=self.neucoeff)
        print(f"MEAN: {dff_matrix.mean()}")
        
        print(f"   Raw ΔF/F shape: {dff_matrix.shape}")
        print(f"   Raw ΔF/F range: {dff_matrix.min():.1f} to {dff_matrix.max():.1f}")
        
        
        if include_passive:
            # Step 1.5: Keep entire time series including passive period
            print("\n1.5. Keeping entire time series (including passive period)...")
            original_duration = timestamps[-1] - timestamps[0]
            truncated_duration = original_duration
            
            print(f"   Keeping full duration: {original_duration:.1f}s")
            print(f"   Final shape: {dff_matrix.shape}")
            
        else:
            # Step 1.5: Truncate to remove passive video period
            print("\n1.5. Truncating passive video period...")
            last_stimoff = session.stimOff_times[-1]
            cutoff_idx = np.searchsorted(timestamps, last_stimoff, side='right')
            original_duration = timestamps[-1] - timestamps[0]
            
            dff_matrix = dff_matrix[:cutoff_idx]
            timestamps = timestamps[:cutoff_idx]
            truncated_duration = timestamps[-1] - timestamps[0]
            
            print(f"   Last stimOff at {last_stimoff:.1f}s, truncating at index {cutoff_idx}")
            print(f"   Truncated shape: {dff_matrix.shape}")
            print(f"   Duration: {original_duration:.1f}s → {truncated_duration:.1f}s ({truncated_duration/original_duration:.1%} retained)")
        
        # Step 2: Robust normalization (applied to full matrix)
        print("\n2. Applying robust normalization...")
        normalized_data, norm_params = self._robust_normalize(dff_matrix)
        
        print(f"   Normalized range: {normalized_data.min():.3f} to {normalized_data.max():.3f}")
        print(f"   Original mean: {dff_matrix.mean():.3f}, std: {dff_matrix.std():.3f}")        
        print(f"   Normalized mean: {normalized_data.mean():.3f}, std: {normalized_data.std():.3f}")
        
        # Step 3: Select first N neurons (no filtering, just selection)
        print(f"\n3. Selecting first {top_n_neurons} neurons...")
        if top_n_neurons is not None and top_n_neurons < dff_matrix.shape[1]:
            # Get indices of first N neurons
            top_n_indices = np.arange(top_n_neurons)
            
            # Select final data to be processed
            final_data_normalized = normalized_data[:, :top_n_neurons]
            print(f"Selected neurons mean: {final_data_normalized.mean()}")
            
            # Create final mask
            final_neuron_mask = np.zeros(dff_matrix.shape[1], dtype=bool)
            final_neuron_mask[top_n_indices] = True
            
            print(f"   Selected first {top_n_neurons} neurons from {dff_matrix.shape[1]} total neurons")
            print(f"   Selection method: first N neurons")
            
        else:
            # Keep all neurons if top_n not specified or larger than available
            final_data_normalized = normalized_data
            final_neuron_mask = np.ones(dff_matrix.shape[1], dtype=bool)
            print(f"   Keeping all {dff_matrix.shape[1]} neurons (top_n={top_n_neurons})")
        
        # Step 4: Optional temporal smoothing
        if self.temporal_smoothing:
            print("\n4. Applying temporal smoothing...")
            smoothed_data = self._temporal_smooth(final_data_normalized)
            final_data = smoothed_data
        else:
            final_data = final_data_normalized
            
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
        quality_metrics = self._assess_quality(dff_matrix, final_data, final_neuron_mask)
        
        # Compile results
        results = {
            'activity': final_data,
            'covariate_matrix': covariate_matrix,
            'timestamps': timestamps,
            'neuron_mask': final_neuron_mask,
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
                'neucoeff': self.neucoeff,
                'temporal_smoothing': self.temporal_smoothing
            }
        }
        
        # Save if requested
        if output_path:
            self._save_processed_data(results, output_path, save_intermediates)
            
        return results
    
    
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
        normalized = np.clip(normalized, -0.25, 1.5)
        
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
        
        print(f"   Saved {results['activity'].shape} activity matrix")
        print(f"   Saved {results['covariate_matrix'].shape} covariate matrix")
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
                'final_shape': results['activity'].shape,
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
