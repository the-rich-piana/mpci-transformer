#  cd utils && python output_session_zscore.py --session_id 5ea6bb9b-6163-4e8a-816b-efe7002666b0 --num_neurons 7000
from one.api import ONE
import sys
import os
import argparse

from activity_preprocessor import CalciumDataPreprocessor
from activity_preprocessor_fluor import CalciumDataPreprocessorFluor

parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from load_meso_session import MesoscopeSession

def main():
    parser = argparse.ArgumentParser(description='Preprocess calcium imaging session data')
    parser.add_argument('--session_id', type=str, required=True, 
                       help='Session ID (EID) to preprocess')
    parser.add_argument('--neucoeff', type=float, default=0.7,
                       help='Neuropil coefficient for Suite2p correction (default: 0.7)')
    parser.add_argument('--temporal_smoothing', default=False,
                       help='Apply temporal smoothing to data, defaults to Falses')
    parser.add_argument('--num_neurons', type=int, default=5000,
                       help='Number of neurons to include in the final activity matrix (default: 5000)')    
    parser.add_argument('--include_passive', type=bool, default=False,
                       help='Include the last 15 minutes of passive behavior')
    args = parser.parse_args()
    
    one = ONE()
    session = MesoscopeSession.from_eid(one, args.session_id, True)
    
    preprocessor = CalciumDataPreprocessorFluor(neucoeff=args.neucoeff, temporal_smoothing=args.temporal_smoothing)
    
    data_type_suffix = "_zscore"
    # Update output path to include processing type
    output_path = f'/cs/student/projects1/aibh/2024/gcosta/mpci_data/session_{args.session_id}_{args.num_neurons}{data_type_suffix}{"_passive" if args.include_passive else ""}.h5'
    preprocessor.preprocess_session(session, output_path, top_n_neurons=args.num_neurons, include_passive=args.include_passive)
    
    print(f"Preprocessed Session {args.session_id}")

if __name__ == '__main__':
    print(sys.version)
    print(f'Preprocessing session')
    main()