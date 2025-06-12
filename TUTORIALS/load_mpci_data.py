from one.api import ONE
import numpy as np
import pandas as pd
from datetime import datetime
import time
import gc 

def __main__():
    # Connect to the ONE API
    
    MAX_ERROR_PERCENT = 0
    
    good_sessions_df = pd.read_csv('good_mesoscope_sessions.csv')
    print(f"Loaded {len(good_sessions_df)} previously processed sessions from CSV")
    
    one = ONE()
    assert not one.offline, 'ONE must be connect to Alyx for searching imaging sessions'
    # Search for mesoscope sessions using the correct query
    query = 'field_of_view__imaging_type__name,mesoscope'
    eids = one.search(procedures='Imaging', django=query, query_type='remote')

    print(f"Number of mesoscope imaging sessions: {len(eids)}")

    good_sessions = []

    session_limit = len(eids)  # Process all sessions
    batch_size = 50  # Process this many at a time
    pause_seconds = 5  # Pause between batches to avoid rate limiting

    for batch_start in range(0, min(session_limit, len(eids)), batch_size):
        batch_end = min(batch_start + batch_size, session_limit)
        print(f"\nProcessing batch {batch_start//batch_size + 1}, sessions {batch_start+1}-{batch_end}")
        
        for i in range(batch_start, batch_end):
            eid = eids[i]
            if str(eid) in good_sessions_df['eid'].values:
                print(f"- Session {eid} already processed, skipping")
                continue
            try:
                # Get session metadata with full details
                session_info = one.get_details(eid, full=True)
                task_protocol = session_info.get('task_protocol', '')
                
                # Check if the session is biasedChoiceWorld or passiveChoiceWorld
                if not ('biasedChoiceWorld' in task_protocol or 'passiveChoiceWorld' in task_protocol):
                    print("- Not biasedChoiceWorld or passiveChoiceWorld, skipping")
                    continue
                    
                # Get session duration
                session_start = session_info.get('start_time')
                session_end = session_info.get('end_time')
                
                duration_hours = None
                if session_start and session_end:
                    try:
                        # Parse datetime strings
                        start_dt = datetime.fromisoformat(session_start.replace('Z', '+00:00'))
                        end_dt = datetime.fromisoformat(session_end.replace('Z', '+00:00'))
                        
                        # Calculate duration
                        duration_seconds = (end_dt - start_dt).total_seconds()
                        duration_hours = duration_seconds / 3600
                        
                        if duration_hours < 1:
                            print(f"- Session too short ({duration_hours:.2f} hours), skipping")
                            continue
                        else:
                            print(f"Session duration: {duration_hours:.2f} hours")
                    except Exception as e:
                        print(f"- Error parsing session times: {str(e)}, skipping")
                        continue
                else:
                    print("- Missing session start or end time, skipping")
                    continue
            
                # List all datasets to identify FOVs
                all_datasets = one.list_datasets(eid)
                
                # Find all FOVs in this session
                fov_collections = set()
                for dataset in all_datasets:
                    if 'mpci' in dataset and 'FOV_' in dataset:
                        # Extract collection path (e.g., alf/FOV_00)
                        parts = dataset.split('/')
                        if len(parts) >= 2:  # Make sure we have at least collection/dataset
                            collection = '/'.join(parts[:-1])  # Everything except the last part
                            fov_collections.add(collection)
                
                # Sort FOV collections for consistent order
                fov_collections = sorted(list(fov_collections))
                print(f"Found {len(fov_collections)} FOV collections: {fov_collections}")
                
                # Process each FOV collection
                good_fovs = []
                
                for collection in fov_collections:
                    try:
                        print(f"Checking FOV collection: {collection}")
                        
                        # List all datasets in this FOV
                        fov_datasets = one.list_datasets(eid, collection=collection)
                        
                        # Check if necessary datasets exist
                        has_roi_activity = any('mpci.ROIActivityDeconvolved' in ds for ds in fov_datasets)
                        has_times = any('mpci.times' in ds for ds in fov_datasets)
                        has_bad_frames = any('mpci.badFrames' in ds for ds in fov_datasets)
                        has_frame_qc = any('mpci.mpciFrameQC' in ds for ds in fov_datasets)
                        
                        # Print which datasets are present
                        print(f"Required datasets present: ROIActivity={has_roi_activity}, times={has_times}, badFrames={has_bad_frames}, frameQC={has_frame_qc}")
                        
                        if not (has_roi_activity and has_times and has_bad_frames and has_frame_qc):
                            print(f"- Missing required datasets, skipping")
                            continue
                        
                        # Only load QC data to check for issues (much smaller files)
                        bad_frames = one.load_dataset(eid, 'mpci.badFrames', collection=collection)
                        frame_qc = one.load_dataset(eid, 'mpci.mpciFrameQC', collection=collection)
                        
                        if bad_frames is None or frame_qc is None:
                            print(f"- Could not load QC datasets, skipping")
                            continue
                        
                        # Check if there are any bad frames
                        bad_frame_count = np.sum(bad_frames)
                        qc_issue_count = np.sum(frame_qc)
                        
                        # Allow up to 5% bad frames
                        total_frames = len(bad_frames)
                        bad_frame_percentage = (bad_frame_count / total_frames) * 100 if total_frames > 0 else 0
                        qc_issue_percentage = (qc_issue_count / total_frames) * 100 if total_frames > 0 else 0
                        
                        
                        if bad_frame_percentage > MAX_ERROR_PERCENT or qc_issue_percentage > MAX_ERROR_PERCENT:
                            print(f"- {collection} has too many QC issues: {bad_frame_percentage:.1f}% bad frames, {qc_issue_percentage:.1f}% QC issues")
                            continue
                        
                        # For ROI count, we'll use a less data-intensive approach
                        # Just load the ROI Activity shape to get the count
                        try:
                            # This will only load the array metadata, not the full array
                            roi_activity = one.load_dataset(eid, 'mpci.ROIActivityDeconvolved', collection=collection)
                            roi_count = roi_activity.shape[1]  # Number of columns = number of ROIs
                            print(f"Found {roi_count} ROIs in activity matrix")
                            
                            # If there are too few ROIs, skip this FOV
                            if roi_count < 50:  # Adjust this threshold as needed
                                print(f"- Too few ROIs ({roi_count}), skipping")
                                continue
                            
                        except Exception as e:
                            print(f"- Error getting ROI count: {str(e)}")
                            continue
                        
                        # This FOV passed all checks
                        good_fovs.append({
                            'collection': collection,
                            'roi_count': roi_count,
                            'frame_count': total_frames,
                            'bad_frame_percentage': bad_frame_percentage,
                            'qc_issue_percentage': qc_issue_percentage
                        })
                        
                        print(f"+ Good FOV! {collection}: {roi_count} ROIs, {total_frames} frames")
                        
                        # Clean up to free memory
                        # del bad_frames, frame_qc, roi_activity
                        gc.collect()
                        
                    except Exception as e:
                        print(f"- Error processing FOV {collection}: {str(e)}")
                
                # Only add session if it has at least one good FOV
                if good_fovs:
                    subject = session_info.get('subject', 'unknown')
                    date = session_info.get('start_time', 'unknown')
                    
                    # Add session to good sessions list
                    good_sessions.append({
                        'eid': eid,
                        'subject': subject,
                        'date': date,
                        'duration_hours': duration_hours if duration_hours else 0,
                        'task_protocol': task_protocol,
                        'fov_count': len(good_fovs),
                        'total_rois': sum(fov['roi_count'] for fov in good_fovs),
                        'fovs': good_fovs
                    })
                    
                    print(f"+ Good session! {subject}, {duration_hours:.2f} hours, {len(good_fovs)} good FOVs, {sum(fov['roi_count'] for fov in good_fovs)} total ROIs")
                else:
                    print("- No good FOVs found in this session")
                    
            except Exception as e:
                print(f"- Error processing session: {str(e)}")
            
            # Clean up to free memory
            gc.collect()
            
            if len(good_sessions) >= 10:  # Stop once we have found enough good sessions
                print("Found 10 good sessions, stopping search")
                break
        
        # Save progress after each batch
        if good_sessions:
            print(f"\nSaving progress after batch: {len(good_sessions)} good sessions found so far")
            # Save to CSV
            sessions_df = pd.DataFrame([{
                'eid': s['eid'],
                'subject': s['subject'],
                'date': s['date'],
                'duration_hours': s['duration_hours'],
                'task_protocol': s['task_protocol'],
                'good_fov_count': s['fov_count'],
                'total_rois': s['total_rois']
            } for s in good_sessions])
            
            sessions_df.to_csv('good_mesoscope_sessions.csv', index=False)
        
        # Pause between batches to avoid rate limiting
        if batch_end < session_limit and batch_end < len(eids):
            print(f"Pausing for {pause_seconds} seconds before next batch...")
            time.sleep(pause_seconds)
        

    # Create a DataFrame of good sessions
    if good_sessions:
        # Create main sessions dataframe
        sessions_df = pd.DataFrame([{
            'eid': s['eid'],
            'subject': s['subject'],
            'date': s['date'],
            'duration_hours': s['duration_hours'],
            'task_protocol': s['task_protocol'],
            'good_fov_count': s['fov_count'],
            'total_rois': s['total_rois']
        } for s in good_sessions])
        
        print("\nGood sessions found:")
        print(sessions_df)
        
        # Create FOVs dataframe for the first session
        if good_sessions and 'fovs' in good_sessions[0]:
            first_session_fovs = pd.DataFrame(good_sessions[0]['fovs'])
            print(f"\nFOVs in first good session ({good_sessions[0]['eid']}):")
            print(first_session_fovs)
        
        # Save the final results
        sessions_df.to_csv('good_mesoscope_sessions_final.csv', index=False)
    
    else:
        print("No good sessions found. Consider relaxing the criteria.")
        
if __name__ == "__main__":
    __main__()