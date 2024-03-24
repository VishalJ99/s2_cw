import numpy as np
from typing import Callable


def metropolis_sampler(target_logPDF: Callable, x0: np.ndarray, n_samples: int,
                       cov_Q: np.ndarray = None):
    """
    Metropolis-Hastings sampler using a multivariate normal proposal
    distribution. Assumes target_logPDF is a callable function that takes a
    1D array of model parameters as input and returns the log of the target
    distribution at those parameters. This function generates samples by
    proposing new states from the multivariate normal distribution centered at
    the current state with a specified covariance matrix.

    Parameters
    ----------
    target_logPDF : Callable
        The log of the target distribution to sample from. This function must
        accept a 1D np.ndarray of parameters and return a float representing
        the log probability density of those parameters.
    x0 : np.ndarray
        The initial state of the chain. Shape should be (dim,), where dim is
        the dimensionality of the parameter space.
    n_samples : int
        The number of samples to generate.
    cov_Q : np.ndarray, optional
        The covariance matrix of the multivariate normal proposal distribution.
        If not provided, the identity matrix is used. Shape should be
        (dim, dim), where dim matches the dimensionality of x0.

    Returns
    -------
    np.ndarray, float
        - The samples from the chain. Shape will be (n_samples, dim), where dim
          is the dimensionality of the parameter space.
        - The acceptance rate as a float, representing the proportion of
        proposed samples that were accepted.

    """
    num_accept = 0
    chain = np.zeros((n_samples, len(x0)))
    if cov_Q is None:
        cov_Q = np.eye(len(x0))

    # Set the initial point.
    chain[0] = x0

    # Run the Metropolis-Hastings algorithm.
    for i in range(n_samples - 1):
        # Get the current point.
        x_current = chain[i]
 
        # Generate a proposed point.
        x_proposed = np.random.multivariate_normal(x_current, cov_Q)

        # Calculate log acceptance ratio.
        log_a = target_logPDF(x_proposed) - target_logPDF(x_current)

        # Accept / reject the proposed point.
        if np.log(np.random.uniform()) < log_a:
            x_new = x_proposed  
            num_accept += 1
        else:
            x_new = x_current

        # Update the chain.
        chain[i + 1] = x_new

    # Calculate acceptance rate.
    acceptance_rate = num_accept / n_samples 
    return chain, acceptance_rate

