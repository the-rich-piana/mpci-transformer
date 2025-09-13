#!/usr/bin/env python3
"""
Validation script to check that Brain_Regions.py region labels 
align perfectly with MesoscopeSession neuron order
"""

import sys
import os
sys.path.append('../utils')

from one.api import ONE
import numpy as np
import pandas as pd
from load_meso_session import MesoscopeSession
from iblatlas.atlas import AllenAtlas
from Brain_Regions import embed_meso, export_region_labels_csv

def extract_regions_from_session(eid):
    """Extract region labels in same order as MesoscopeSession.get_dff_activity_matrix()"""
    print("Loading session with MesoscopeSession.from_eid()...")
    one = ONE()
    session = MesoscopeSession.from_eid(one, eid, raw_activity=True)
    
    atlas = AllenAtlas()
    all_region_labels = []
    
    # Get FOV collections in same sorted order as MesoscopeSession
    all_datasets = one.list_datasets(eid)
    fov_collections = set()
    for dataset in all_datasets:
        if 'mpci' in dataset and 'FOV_' in dataset:
            parts = dataset.split('/')
            if len(parts) >= 2:
                collection = '/'.join(parts[:-1])
                fov_collections.add(collection)
    
    fov_collections = sorted(list(fov_collections))
    print(f"Found {len(fov_collections)} FOVs: {[c.split('/')[-1] for c in fov_collections]}")
    
    # Process each FOV in same order as session.get_dff_activity_matrix()
    for collection in fov_collections:
        fov_name = collection.split('/')[-1]
        print(f"Processing {fov_name}...")
        
        # Load ROI data for this FOV
        roi_types = one.load_dataset(eid, 'mpciROIs.mpciROITypes', collection=collection)
        
        # Try to load brain locations
        try:
            brain_location_ids = one.load_dataset(eid, 'mpciROIs.brainLocationIds_ccf_2017_estimate', collection=collection)
        except:
            try:
                brain_location_ids = one.load_dataset(eid, 'mpciROIs.brainLocationsIds_ccf_2017', collection=collection)
            except:
                print(f"Warning: No brain location data found for {fov_name}")
                brain_location_ids = np.zeros(len(roi_types))
        
        # Convert IDs to region labels
        region_labels = atlas.regions.id2acronym(brain_location_ids)
        
        # Filter to only neurons (same as MesoscopeSession)
        neuron_mask = np.array(roi_types) == 1
        neuron_region_labels = region_labels[neuron_mask]
        
        print(f"  {np.sum(neuron_mask)} neurons in {fov_name}")
        all_region_labels.extend(neuron_region_labels)
    
    return np.array(all_region_labels)

def validate_alignment(eid):
    """Compare region labels from both methods"""
    print("="*80)
    print(f"VALIDATING REGION ALIGNMENT FOR EID: {eid}")
    print("="*80)
    
    # Method 1: Extract from MesoscopeSession
    print("\n1. Extracting region labels from MesoscopeSession...")
    session_regions = extract_regions_from_session(eid)
    
    # Method 2: Run Brain_Regions.py and extract CSV
    print("\n2. Running Brain_Regions.py script...")
    embed_meso(eid)  # This creates the .npy file
    df = export_region_labels_csv(eid, f"regions_{eid}_validation.csv")
    brain_regions_labels = df['region_label'].values
    
    print(f"\n3. Comparing results...")
    print(f"   MesoscopeSession method: {len(session_regions)} neurons")
    print(f"   Brain_Regions.py method: {len(brain_regions_labels)} neurons")
    
    # Check alignment
    if len(session_regions) != len(brain_regions_labels):
        print("❌ MISMATCH: Different number of neurons!")
        return False
    
    matches = session_regions == brain_regions_labels
    n_matches = np.sum(matches)
    
    print(f"   Perfect matches: {n_matches}/{len(session_regions)} ({n_matches/len(session_regions)*100:.1f}%)")
    
    if n_matches == len(session_regions):
        print("✅ PERFECT ALIGNMENT: All region labels match!")
        print("\nRegion counts:")
        region_counts = pd.Series(session_regions).value_counts()
        for region, count in region_counts.items():
            print(f"   {region}: {count}")
        return True
    else:
        print("❌ MISALIGNMENT detected!")
        print("First 10 mismatches:")
        mismatch_indices = np.where(~matches)[0][:10]
        for i in mismatch_indices:
            print(f"   Index {i}: MesoscopeSession='{session_regions[i]}' vs Brain_Regions='{brain_regions_labels[i]}'")
        return False

if __name__ == "__main__":
    eid = "5ea6bb9b-6163-4e8a-816b-efe7002666b0"
    success = validate_alignment(eid)
    
    if success:
        print(f"\n🎉 SUCCESS: Region labels are perfectly aligned!")
        print(f"You can now use 'regions_{eid}_validation.csv' for unit embedding visualization")
    else:
        print(f"\n❌ FAILED: Region labels are not aligned. Check FOV processing order.")