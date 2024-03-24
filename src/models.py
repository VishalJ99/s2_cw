import numpy as np
from typing import Callable


def uniform_2d(
    theta: np.ndarray, a: int = -100, b: int = 100, c: int = -100, d: int = 100
) -> float:
    """
    2D Uniform PDF for alpha and beta.
    Upper and lower bounds are a and b for alpha and c and d for beta.

    Parameters
    ----------
    theta : np.ndarray
        The 2D parameter values at which to evaluate the PDF.
    a : float
        Lower bound for alpha.
    b : float
        Upper bound for alpha.
    c : float
        Lower bound for beta.
    d : float
        Upper bound for beta.

    Returns
    -------
    float
        The value of the 2D uniform PDF at theta.
    """
    alpha, beta = theta
    if (a <= alpha <= b) and (c <= beta <= d):
        return 1 / ((b - a) * (d - c))
    else:
        return 0


def cauchy(x: float, alpha: float, beta: float) -> float:
    """
    Likelihood function of the lighthouse problem.

    Parameters
    ----------
    x : float
        The value at which to evaluate the pdf.
    alpha : float
        The location parameter of the Cauchy distribution.
    beta : float
        The scale parameter of the Cauchy distribution.

    Returns
    -------
    float
        The value of the Cauchy distribution at x.
    """
    return beta / (np.pi * (beta**2 + (x - alpha) ** 2))


def log_p(theta: np.ndarray, data: np.ndarray, prior: Callable) -> float:
    """
    Unnormalised log posterior of alpha and beta.

    Parameters
    ----------
    theta : np.ndarray
        The parameter values to calculate the log posterior for.
    data : np.ndarray
        The data to use in the likelihood function.
    prior : Callable
        The prior distribution of the parameters.

    Returns
    -------
    float
        The log posterior probability of the parameter values.
    """
    alpha, beta = theta
    log_likelihood = sum([np.log(cauchy(x, alpha, beta)) for x, _ in data])
    return log_likelihood + np.log(prior(theta))


def log_uniform(x: float, a: float = 1e-5, b: float = 1e3) -> float:
    """
    Log uniform distribution. With support [a, b].

    Parameters
    ----------
    x : float
        The value at which to evaluate the log uniform PDF.
    a : float
        Lower bound of the support.
    b : float
        Upper bound of the support.

    Returns
    -------
    float
        The value of the log uniform PDF at x.
    """
    assert a != 0 or b != 0, "a or b cannot be zero"
    if a <= x <= b:
        return 1 / (x * np.log(b / a))
    else:
        return 0


def log_normal(
    I: float, x: float, alpha: float, beta: float, I0: float, sigma: float = 1
) -> float:
    """
    Log normal PDF. Calculates the mean from the intensity I0 and the distance
    from the lighthouse sqrt(beta^2 + (x - alpha)^2). The standard deviation
    is given by sigma.

    Parameters
    ----------
    I : float
        The value at which to evaluate the log normal PDF.
    x : float
        The detector position x.
    alpha : float
        The location along the coast of the lighthouse.
    beta : float
        The location out to sea of the lighthouse.
    I0 : float
        The absolute intensity of the lighthouse.
    sigma : float
        The standard deviation of the log normal distribution.
    Returns
    -------
    float
        The value of the log normal PDF at I.
    """
    d = np.sqrt(beta**2 + (x - alpha)**2)
    mu = np.log(I0 / d**2)
    return (1 / (I * np.sqrt(2 * np.pi * sigma**2))) * np.exp(
        - (np.log(I) - mu) ** 2 / (2 * sigma**2)
    )


def log_p2(
    theta: np.ndarray, data: np.ndarray, prior_1: Callable, prior_2: Callable
) -> float:
    """
    Un normalised log posterior of alpha, beta and I0. Prior 1 is the prior for
    alpha and beta and prior 2 is the prior for I0. Uses the cauchy distribution
    for the likelihood of x and the log normal distribution for the likelihood of I.
    Assumes data is a list of lists where each sublist is list of floats [x, I].

    Parameters
    ----------
    theta : np.ndarray
        The parameter values (alpha, beta, I0) at which to evaluate
        the log posterior.
    data : np.ndarray
        The data to use in the likelihood function. Each row is a list of floats
        [x, I].
    prior_1 : Callable
        The prior distribution for alpha and beta.
    prior_2 : Callable
        The prior distribution for I0.

    Returns
    -------
    float
        The log posterior probability of the parameter values.
    """
    # Unpack the parameters.
    alpha, beta, I0 = theta

    log_likelihood_x = sum([np.log(cauchy(x, alpha, beta)) for x, _ in data])
    log_likelihood_i = sum([np.log(log_normal(I, x, alpha, beta, I0))
                            for x, I in data])
    log_likelihood = log_likelihood_x + log_likelihood_i

    return log_likelihood + np.log(prior_1((alpha, beta))) + np.log(prior_2(I0))
