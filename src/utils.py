import numpy as np
import matplotlib.pyplot as plt

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


def make_trace_plot(all_chains):
    # Make nicer.
    n_chains, n_samples, _ = all_chains.shape
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Main Trace plot (zoomed out)
    for chain in all_chains:
        alpha_samples, beta_samples = chain[:, 0], chain[:, 1]
        ax1.plot(alpha_samples, label='alpha')
        ax1.plot(beta_samples, label='beta')
    
    ax1.set_title('Trace plot of alpha and beta (Zoomed out)')
    ax1.legend()
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Parameter value')

    # Zoom in on the burn-in phase
    burn_in_end = int(0.05 * n_samples)  # Adjust as necessary
    for chain in all_chains:
        alpha_samples, beta_samples = chain[:burn_in_end, 0], chain[:burn_in_end, 1]
        ax2.plot(alpha_samples, label='alpha')
        ax2.plot(beta_samples, label='beta')

    ax2.set_title('Burn-in phase')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Parameter value')

    # Zoom in on an equilibrium phase
    equilibrium_start = int(0.45 * n_samples)  # Adjust as necessary
    equilibrium_end = int(0.55 * n_samples)  # Adjust as necessary
    for chain in all_chains:
        alpha_samples, beta_samples = chain[equilibrium_start:equilibrium_end, 0], chain[equilibrium_start:equilibrium_end, 1]
        ax3.plot(alpha_samples, label='alpha')
        ax3.plot(beta_samples, label='beta')

    ax3.set_title('Equilibrium phase')
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Parameter value')

    plt.tight_layout()
    return plt


def make_joint_plot(all_iid_samples):
    fig, ax = plt.subplots(figsize=(10, 10))
    alpha_samples, beta_samples = all_iid_samples[:, 0], all_iid_samples[:, 1]
    ax.scatter(alpha_samples, beta_samples, alpha=0.5)
    ax.set_title('Joint distribution of alpha and beta')
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    return plt