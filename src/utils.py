import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def gelman_rubin(chains):
    """
    Computes the Gelman-Rubin statistic for a chain of samples.
    Uses https://en.wikipedia.org/wiki/Gelman-Rubin_statistic as a reference.
    
    Parameters
    ----------
    chains : np.array
        A 3D array of shape n_chains, n_samples, dim.
    
    Returns
    -------
    gr : float
        The Gelman-Rubin statistic.
    """
    # J is no. of chains, L is no. samples.
    J, L, _ = chains.shape

    # Mean value of chain j.
    x_bar_j = np.mean(chains, axis=1)

    # Mean of the means of all chains.
    x_bar_star = np.mean(x_bar_j, axis=0)

    # Variance of the means of all chains.
    B = (L / (J - 1)) * np.sum((x_bar_j - x_bar_star)**2, axis=0)

    # Averaged variances of the individual chains across all chains.
    W = (1 / J) * np.sum(np.var(chains, axis=1, ddof=1), axis=0)

    # Estimate of the Gelman-Rubin statistic R.
    R = ((L - 1) / L) + (B / (L * W))
    return R


def make_joint_plot(iid_samples):
    assert len(iid_samples.shape) == 2, "iid_samples should be a 2D array"
    alpha_samples, beta_samples = iid_samples[:, 0].flatten(), iid_samples[:, 1].flatten()

    # First, plot the hexbin heatmap as before
    plt.hexbin(alpha_samples, beta_samples, gridsize=100, cmap='magma', bins='log')
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\beta$")
    plt.title(f"Joint distribution of alpha and beta (N={len(iid_samples_1)})")
    plt.show()
    return plt


def make_trace_plot(chains, param_idx, param_name, N_view=1000):
    """
    Generates a trace plot for the given chains.
    
    Parameters
    ----------
    chains : np.ndarray
        An array of chains, expected to have shape (n_chains, n_samples, n_parameters).
    param_idx : int
        The index of the parameter to plot.
    param_name : str
        The name of the parameter to plot.
    N_view : int, optional
        The number of samples to view in the zoomed-in plots.
    
    Returns
    -------
    plt : matplotlib.pyplot
        The plot object.
    """
    # Assuming chains_1 is your list of chains and each chain has a shape of (N, M) where N is the number of samples
    N = chains[0].shape[0]  # Total number of samples in a chain
    middle_start = N//2 - 500  # Starting index for the middle 1000 samples, adjust as necessary

    # Set up the grid
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

    # Main plot
    ax_main = plt.subplot(gs[:, 0])
    for chain in chains:
        ax_main.plot(chain[:, param_idx], alpha=0.5, label='alpha')
    
    ax_main.set_xlabel('Iteration', fontsize=16)
    ax_main.set_ylabel(f'{param_name} value', fontsize=16)
    ax_main.set_title(f'Trace plot of {param_name}', fontsize=20)

    # Subplot 1 - Zoom on first N_view samples
    ax_zoom1 = plt.subplot(gs[0, 1])
    for chain in chains:
        ax_zoom1.plot(chain[:N_view, param_idx], alpha=0.5, label=f'{param_name}  (first 1000)')
    ax_zoom1.set_title(f'Trace of First {N_view} Samples', fontsize=18)

    # Subplot 2 - Zoom on middle N samples
    ax_zoom2 = plt.subplot(gs[1, 1])
    for chain in chains:
        ax_zoom2.plot(np.arange(middle_start, middle_start+N_view),
                      chain[middle_start:middle_start+N_view, param_idx],
                      alpha=0.5, label=f'{param_name} (middle 1000)')
    ax_zoom2.set_title(f'Trace of Middle {N_view} Samples', fontsize=18)

    plt.tight_layout()
    return plt


def make_corner_plot(iid_samples, parameter_names=None):
    """
    Generates a corner plot for the given iid samples.

    Parameters:
    -----------
    iid_samples : np.ndarray
        The iid samples to plot. Shape should be (n_samples, n_parameters).
    parameter_names : List[str], optional
        The names of the parameters. If None, default names will be used.
        
    Returns:
    --------
    plt : matplotlib.pyplot
        The plot object.
    """
    n_parameters = iid_samples.shape[1]
    if parameter_names is None:
        parameter_names = [f"Param {i+1}" for i in range(n_parameters)]

    fig, axes = plt.subplots(n_parameters, n_parameters, figsize=(10, 10))

    for i in range(n_parameters):
        for j in range(n_parameters):
            ax = axes[i, j]
            if i == j:  # Plot the marginal distribution
                ax.hist(iid_samples[:, i], bins=100)
                mean = np.mean(iid_samples[:, i])
                std = np.std(iid_samples[:, i])
                ax.axvline(mean, color='red', linestyle='-')
                ax.axvline(mean + std, color='red', linestyle='--')
                ax.axvline(mean - std, color='red', linestyle='--')
                ax.set_title(f"{parameter_names[i]}"+ f" = {mean:.3f} + $\pm {std:.4f}$", color='red')
            elif i > j:  # Plot the hexbin heatmap for joint distributions
                ax.hexbin(iid_samples[:, j], iid_samples[:, i], gridsize=50, cmap='magma', bins='log')
            else:  # Remove upper triangle
                ax.axis('off')

            # Set labels
            if j == 0 and i > 0:  # Only label the leftmost subplots for clarity
                ax.set_ylabel(parameter_names[i])
            if i == n_parameters-1:  # Only label the bottom subplots for clarity
                ax.set_xlabel(parameter_names[j])
    fig.suptitle(f"Corner plot of the parameters (N={len(iid_samples)})")

    plt.tight_layout()
    return plt