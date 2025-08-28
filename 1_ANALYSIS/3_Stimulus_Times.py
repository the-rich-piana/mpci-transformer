# %%
import sys
sys.executable

# %%
from one.api import ONE
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import seaborn as sns


parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from utils.load_meso_session import MesoscopeSession

one = ONE()
SESSION_INDEX = 0


# %%
session = MesoscopeSession.from_eid(one, '61f260e7-b5d3-4865-a577-bcfc53fda8a8', True)

# %%
session.stimOn_times

# %%
session.stimOff_times

# %%



