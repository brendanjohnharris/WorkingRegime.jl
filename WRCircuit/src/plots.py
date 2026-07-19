import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_spiking_activity(
    FNSnet,
    runner,
    tstart=None,
    tstop=None,
    window_size_ms=10.0,
    ms_per_s=1000,
    fps=30.0,
    dpi=100,
):
    """
    Animate the spiking activity of the network on a 2D heatmap using blitting.

    Parameters
    ----------
    FNSnet : WRCircuit
        The neural network model.
    runner : brainpy.DSRunner
        The simulation runner that has run the simulation and contains spiking data.
    tstart : float
        Start time in milliseconds for the animation.
    tstop : float
        Stop time in milliseconds for the animation.
    window_size_ms : float
        The size of the time window in milliseconds over which to sum spiking activity.
    ms_per_s : float
        The number of milliseconds per second, used to map milliseconds to seconds in the animation.
    fps : float
        Frames per second for the animation.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The animation object.
    """
    # Extract the time stamps and spiking data from the runner
    ts = runner.mon["ts"]  # Time stamps, shape (num_time_steps,)
    E_spikes = runner.mon["E.spike"]  # Shape (num_time_steps, num_exc_neurons)
    I_spikes = runner.mon["I.spike"]  # Shape (num_time_steps, num_inh_neurons)

    if tstart is None:
        tstart = ts[0]
    if tstop is None:
        tstop = ts[-1]

    # Get the positions
    E_positions = np.array(FNSnet.E.positions)
    I_positions = np.array(FNSnet.I.positions)

    # Select the time range
    start_idx = np.searchsorted(ts, tstart)
    stop_idx = np.searchsorted(ts, tstop)

    ts_range = ts[start_idx:stop_idx]
    E_spikes_range = E_spikes[start_idx:stop_idx, :]
    I_spikes_range = I_spikes[start_idx:stop_idx, :]

    # Total time in ms and seconds
    total_time_ms = tstop - tstart
    total_time_seconds = total_time_ms / ms_per_s

    # Compute the number of frames and frame times
    num_frames = int(np.ceil(total_time_seconds * fps))
    frame_times = np.linspace(tstart, tstop, num_frames)

    # Prepare the activity frames
    num_exc_neurons = E_spikes.shape[1]
    num_inh_neurons = I_spikes.shape[1]

    E_activity_frames = np.zeros((num_frames, num_exc_neurons))
    I_activity_frames = np.zeros((num_frames, num_inh_neurons))

    # For each frame, compute the activity (summing spikes over the window)
    for frame_idx, frame_time in enumerate(frame_times):
        window_start = frame_time - window_size_ms

        idx_start = np.searchsorted(ts_range, window_start, side="left")
        idx_end = np.searchsorted(ts_range, frame_time, side="right")

        E_activity_frames[frame_idx, :] = np.sum(
            E_spikes_range[idx_start:idx_end, :], axis=0
        )
        I_activity_frames[frame_idx, :] = np.sum(
            I_spikes_range[idx_start:idx_end, :], axis=0
        )

    # Define the spatial grid
    domain = FNSnet.E.embedding.domain
    size = FNSnet.E.size
    xedges = np.linspace(0, domain[0], size[0] + 1)
    yedges = np.linspace(0, domain[1], size[1] + 1)

    # Precompute all histograms for each frame (here, only for the excitatory population)
    histograms = []
    for frame_idx in range(num_frames):
        hist2d_E, _, _ = np.histogram2d(
            E_positions[:, 0],
            E_positions[:, 1],
            bins=[xedges, yedges],
            weights=E_activity_frames[frame_idx, :],
        )
        # If you want to include the inhibitory population, uncomment and sum:
        # hist2d_I, _, _ = np.histogram2d(
        #     I_positions[:, 0],
        #     I_positions[:, 1],
        #     bins=[xedges, yedges],
        #     weights=I_activity_frames[frame_idx, :],
        # )
        # hist2d = hist2d_E + hist2d_I

        hist2d = hist2d_E  # Only using the excitatory population
        histograms.append(hist2d)

    # Convert histograms to a numpy array: shape (num_frames, xbins, ybins)
    histograms = np.array(histograms)

    # Determine maximum value for consistent color scaling
    max_hist_value = histograms.max()

    # Initialize the figure and axes
    fig, ax = plt.subplots(dpi=dpi)

    # Plot the initial heatmap
    im = ax.imshow(
        histograms[0].T,  # Transpose to match axes orientation
        origin="lower",
        extent=[0, domain[0], 0, domain[1]],
        interpolation="nearest",
        aspect="auto",
        cmap="hot",
        vmin=0,
        vmax=max_hist_value,  # Use consistent scaling across frames
    )

    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Spike Count")

    # Set axis labels
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # Create a title text object and store it so we can update it efficiently
    title = ax.set_title("")

    # --- Blitting: Define the initialization function ---
    def init():
        # Set the initial image and title text
        im.set_data(histograms[0].T)
        title.set_text("")
        # Return the artists that will be updated
        return [im, title]

    # --- Blitting: Define the update function ---
    def update(frame_idx):
        im.set_data(histograms[frame_idx].T)
        title.set_text(f"Time: {frame_times[frame_idx]:.2f} ms")
        # Return the updated artists
        return [im, title]

    # Create the animation using blitting
    interval = 1000 / fps  # Convert fps to interval in milliseconds
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        init_func=init,
        interval=interval,
        blit=False,
    )

    # Close the figure to avoid displaying a static plot in some environments.
    plt.close(fig)

    return ani
