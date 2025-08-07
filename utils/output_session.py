#Run with: $ cd utils && python output_session.py --session_id 61f260e7-b5d3-4865-a577-bcfc53fda8a8

from one.api import ONE
import sys
import os
import argparse

from activity_preprocessor import CalciumDataPreprocessor

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
    parser.add_argument('--temporal_smoothing', action='store_true',
                       help='Apply temporal smoothing to data')
    
    args = parser.parse_args()
    
    one = ONE()
    session = MesoscopeSession.from_eid(one, args.session_id, True)
    preprocessor = CalciumDataPreprocessor(neucoeff=args.neucoeff, temporal_smoothing=args.temporal_smoothing)
    output_path = f'../DATA/session_{args.session_id}.h5'
    preprocessor.preprocess_session(session, output_path)
    
    print(f"Preprocessed Session {args.session_id}")

if __name__ == '__main__':
    print(sys.version)
    print(f'Preprocessing session')
    main()