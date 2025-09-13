import matplotlib.pyplot as plt
import numpy as np


def plot_calcium_raster(X_embedding, session, timestamps, 
                       output_path="../FIGURES/Activity_Traces_Whole.png",
                       embedding_min=-0.1, embedding_max=2.6,
                       figsize=(12, 6), dpi=200):
    """
    Plot calcium imaging rastermap with clean axes coordinate positioning.
    
    Simplified approach: main image uses data coordinates (with extent),
    all overlays use axes coordinates for consistent, predictable positioning.
    
    Parameters:
    -----------
    X_embedding : np.ndarray
        Rastermap embedding matrix, shape (n_clusters, n_timesteps)
    session : MesoscopeSession
        Session object containing stimOff_times and other metadata
    timestamps : np.ndarray
        Timestamps array corresponding to the neural data
    output_path : str
        Path to save the figure
    embedding_min, embedding_max : float
        Min/max values for colormap scaling
    figsize : tuple
        Figure size (width, height)
    dpi : int
        Figure DPI for saving
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Set font properties
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 10
    
    # Extract dimensions
    n_clusters, n_timesteps = X_embedding.shape
    duration_seconds = session.duration_seconds
    
    # Calculate active/passive split
    last_stimOff_secs = session.stimOff_times[-1]
    last_stimoff_timestep_idx = np.searchsorted(timestamps, last_stimOff_secs, side='right') - 1
    active_fraction = last_stimoff_timestep_idx / n_timesteps
    
    # Calculate 5-minute scale bar width in axes coordinates  
    total_time_minutes = duration_seconds / 60
    minutes_per_timestep = total_time_minutes / n_timesteps
    timesteps_per_5min = 5 / minutes_per_timestep
    scale_bar_width = timesteps_per_5min / n_timesteps  # Fraction of plot width
    
    # Get neuron count
    num_neurons = getattr(session, 'num_neurons', 11393)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Main raster plot with data coordinates
    extent = (0, n_timesteps, n_clusters, 0)
    im = ax.imshow(X_embedding, vmin=embedding_min, vmax=embedding_max,
                   cmap="viridis", aspect="auto", extent=extent)
    
    # Remove default ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # === ALL OVERLAYS USE AXES COORDINATES (transform=ax.transAxes) ===
    
    # Add neuron count label (left side)
    ax.text(-0.02, 0.5, f'{num_neurons:,} neurons', transform=ax.transAxes, 
            rotation=90, ha='right', va='center', fontweight='bold', fontsize=12)
    
    # Add 5-minute scale bar (bottom left, properly aligned)
    scale_y = -0.06
    ax.plot([0.0, scale_bar_width], [scale_y, scale_y], 'k-', 
            linewidth=6, clip_on=False, transform=ax.transAxes)
    ax.text(scale_bar_width/2, scale_y - 0.02, '5 min', 
            transform=ax.transAxes, ha='center', va='top', fontweight='bold', fontsize=10)
    
    # Add timestep count (bottom center)
    ax.text(0.5, -0.03, f'{n_timesteps:,} time steps', transform=ax.transAxes, 
            ha='center', va='top', fontweight='bold', fontsize=12)
    
    # Add active/passive splits (above train/val/test)
    active_passive_y = -0.10
    colors_ap = {'active': "#FF6B35", 'passive': "#004E89"}
    
    # Active segment
    ax.plot([0, active_fraction], [active_passive_y, active_passive_y], 
            color=colors_ap['active'], linewidth=10, solid_capstyle='butt',
            transform=ax.transAxes, clip_on=False)
    ax.text(active_fraction/2, active_passive_y, 'Active', color='white',
            transform=ax.transAxes, ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Passive segment  
    ax.plot([active_fraction, 1.0], [active_passive_y, active_passive_y],
            color=colors_ap['passive'], linewidth=10, solid_capstyle='butt', 
            transform=ax.transAxes, clip_on=False)
    ax.text(active_fraction + (1.0 - active_fraction)/2, active_passive_y, 'Passive', color='white',
            transform=ax.transAxes, ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Add train/val/test splits (below active/passive)
    splits_y = -0.14
    train_end = 0.7
    val_end = 0.8
    colors_split = {'train': "#9B3496", 'val': "#6D9E2C", 'test': "#B63B36"}
    
    # Train segment
    ax.plot([0, train_end], [splits_y, splits_y], color=colors_split['train'],
            linewidth=12, solid_capstyle='butt', transform=ax.transAxes, clip_on=False)
    ax.text(train_end/2, splits_y, 'Train', color='white',
            transform=ax.transAxes, ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Val segment
    ax.plot([train_end, val_end], [splits_y, splits_y], color=colors_split['val'],
            linewidth=12, solid_capstyle='butt', transform=ax.transAxes, clip_on=False) 
    ax.text(train_end + (val_end - train_end)/2, splits_y, 'Val', color='black',
            transform=ax.transAxes, ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Test segment
    ax.plot([val_end, 1.0], [splits_y, splits_y], color=colors_split['test'],
            linewidth=12, solid_capstyle='butt', transform=ax.transAxes, clip_on=False)
    ax.text(val_end + (1.0 - val_end)/2, splits_y, 'Test', color='white',
            transform=ax.transAxes, ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Add colorbar (bottom right, using figure coordinates for precise control)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="2%", pad=0.25)
    
    # Make colorbar only span right 20% of the plot
    cax_pos = cax.get_position()
    cbar_width = cax_pos.width * 0.2
    cbar_left = cax_pos.x1 - cbar_width
    cax.set_position([cbar_left, cax_pos.y0, cbar_width, cax_pos.height])
    
    cbar = plt.colorbar(im, cax=cax, ticks=[embedding_min, embedding_max], 
                       orientation='horizontal')
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, width=0, length=0)
    cbar.ax.set_xlabel('ΔF/F', fontweight='bold', fontsize=10)
    cbar.ax.set_xticklabels(['0.', '.8'])
    
    # Set axis limits (keep data coordinate limits for main image)
    ax.set_xlim(0, n_timesteps)
    ax.set_ylim(n_clusters, 0)
    
    # Remove spines for clean look
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    
    return fig, ax


# Example usage:
# fig, ax = plot_calcium_raster(X_embedding, session, timestamps)
# plt.show()