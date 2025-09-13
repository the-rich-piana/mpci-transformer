#  cd utils && python output_session_pseudo.py --session_id 5ea6bb9b-6163-4e8a-816b-efe7002666b0 --num_neurons 7000

from one.api import ONE
import sys
import os
import argparse

from activity_preprocessor_pseudo import CalciumDataPreprocessorPseudo

parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from load_meso_session import MesoscopeSession

def main():
    parser = argparse.ArgumentParser(description='Preprocess calcium imaging session data')
    parser.add_argument('--session_id', type=str, required=True, 
                       help='Session ID (EID) to preprocess')
    parser.add_argument('--num_neurons', type=int, default=5000,
                       help='Number of neurons to include in the final activity matrix (default: 5000)')    
    parser.add_argument('--include_passive', type=bool, default=False,
                       help='Include the last 15 minutes of passive behavior')
    args = parser.parse_args()
    
    one = ONE()
    session = MesoscopeSession.from_eid(one, args.session_id, False)
    
    preprocessor = CalciumDataPreprocessorPseudo()
    data_type_suffix = "_pseudo"
    
    # Update output path to include processing type
    output_path = f'/cs/student/projects1/aibh/2024/gcosta/mpci_data/session_{args.session_id}_{args.num_neurons}{data_type_suffix}{"_passive" if args.include_passive else ""}.h5'
    preprocessor.preprocess_session(session, output_path, top_n_neurons=args.num_neurons, include_passive=args.include_passive)
    
    print(f"Preprocessed Session {args.session_id} pseudo")

if __name__ == '__main__':
    print(sys.version)
    print(f'Preprocessing session')
    main()