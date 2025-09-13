import numpy as np
from rastermap import Rastermap
import seaborn as sns
import matplotlib.pyplot as plt

def plot_raster(X_embedding: np.ndarray):
    # plot

    fig = plt.figure(figsize=(12,5), dpi=200)
    ax = fig.add_subplot(111)
    ax.imshow(X_embedding, vmin=0, vmax=1.5, cmap="viridis", aspect="auto")