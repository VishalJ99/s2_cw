from samplers import metropolis_sampler
from models import uniform_2d, log_p, log_p2, log_uniform
from utils import gelman_rubin, make_trace_plot, make_corner_plot
from scipy.stats import loguniform
import numpy as np
import argparse
from emcee.autocorr import integrated_time, AutocorrError
import matplotlib.pyplot as plt
import random
import warnings
import sys

random.seed(42)
np.random.seed(42)

# Catch some errors related to calling np.log on zero
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main(n_chains, n_samples):
    # Load the data.
    with open("lighthouse_flash_data.txt", "r") as f:
        data = []
        lines = f.readlines()
        for line in lines:
            fmt_line = line.rstrip().split(" ")
            x = float(fmt_line[0])
            I = float(fmt_line[1])
            data.append([x, I])

    # Prior limits for alpha.
    a, b = -100, 100
    # Prior limits for beta.
    c, d = 0, 50
    # Prior limits for I0.
    e, f = 1e-4, 1e2

    # Define pdf lambda functions.
    prior_1 = lambda theta: uniform_2d(theta, a, b, c, d)
    prior_2 = lambda theta: log_uniform(theta, e, f)
    log_posterior_1 = lambda theta: log_p(theta, data, prior_1)
    log_posterior_2 = lambda theta: log_p2(theta, data, prior_1, prior_2)

    # ------------------
    # Code for part v.
    # ------------------
    print("[INFO] Sampling Pr(alpha, beta | data) using Metropolis-Hastings.")
    cov_Q = np.eye(2)
    all_chains = np.zeros((n_chains, n_samples, 2))
    all_iid_samples = []

    for i in range(n_chains):
        print("[INFO] Running chain", i + 1)
        # Draw x0 from the prior.
        alpha = np.random.uniform(a, b)
        beta = np.random.uniform(c, d)

        # Define the initial state.
        x0 = np.asarray([alpha, beta])

        # Sample the posterior.
        samples, _ = metropolis_sampler(log_posterior_1, x0, n_samples, cov_Q)
        all_chains[i] = samples

        # Discard the burn-in.
        samples = samples[int(0.1 * n_samples) :]

        # Thin the chain.
        try:
            tau = max(integrated_time(samples[:, j])
                      for j in range(samples.shape[1]))
        except AutocorrError:
            print("[ERROR] Chain too short to calculate integrated"
                  " auto correlation time.")
            sys.exit(1)
        thinning = int(2 * tau[0])
        iid_samples = samples[::thinning]
        print("[INFO] Integrated autocorrelation time:", tau)
        print("[INFO] Number of i.i.d samples retrieved:", len(iid_samples))
        all_iid_samples.extend(iid_samples)

    # Calculate the Gelman-Rubin statistic.
    gelman_rubin_statistic = gelman_rubin(all_chains)
    print("[INFO] Gelman-Rubin statistic:", gelman_rubin_statistic)

    # Calculate the mean and standard deviation for the parameters.
    mean_params = np.mean(all_iid_samples, axis=0)
    std_params = np.std(all_iid_samples, axis=0)

    print("[INFO] Mean alpha:", mean_params[0])
    print("[INFO] Mean beta:", mean_params[1])
    print("[INFO] Standard deviation alpha:", std_params[0])
    print("[INFO] Standard deviation beta:", std_params[1])

    # Show trace plots.
    plt = make_trace_plot(all_chains, 0, r"$\alpha$")
    plt.show()
    plt = make_trace_plot(all_chains, 1, r"$\beta$")
    plt.show()

    # Show joint plot.
    param_names = [r"$\alpha$", r"$\beta$"]
    plt = make_corner_plot(np.asarray(all_iid_samples), param_names)
    plt.show()

    # ------------------
    # Code for part vii.
    # ------------------
    print("[INFO] Sampling Pr(alpha, beta, I0 | data)"
          " using Metropolis-Hastings.")

    cov_Q = np.asarray([[1, 0, 0], [0, 0.5, 0], [0, 0, 5]])
    all_chains = np.zeros((n_chains, n_samples, 3))
    all_iid_samples = []
    for i in range(n_chains):
        print("[INFO] Running chain", i + 1)
        # Draw alpha, beta from uniform.
        alpha = np.random.uniform(a, b)
        beta = np.random.uniform(c, d)

        # Draw I0 from log uniform.
        I0 = loguniform.rvs(e, f)

        # Define the initial state.
        x0 = np.asarray([alpha, beta, I0])

        # Sample the posterior.
        samples, _ = metropolis_sampler(log_posterior_2, x0, n_samples, cov_Q)
        all_chains[i] = samples

        # Discard the burn-in.
        samples = samples[int(0.1 * n_samples):]
        try:
            tau = max([integrated_time(samples[:, j])
                       for j in range(samples.shape[1])])
        except AutocorrError:
            print("[ERROR] Chain too short to calculate integrated"
                  " auto correlation time.")
            sys.exit(1)

        # Thin the chain.
        thinning = int(2 * tau[0])
        iid_samples = samples[::thinning]
        print("[INFO] Integrated autocorrelation time:", tau)
        print("[INFO] Number of i.i.d samples retrieved:", len(iid_samples))
        all_iid_samples.extend(iid_samples)

    # Calculate the Gelman-Rubin statistic.
    gelman_rubin_statistic = gelman_rubin(all_chains)
    print("[INFO] Gelman-Rubin statistic:", gelman_rubin_statistic)

    # Calculate the mean and standard deviation for the parameters.
    mean_params = np.mean(all_iid_samples, axis=0)
    std_params = np.std(all_iid_samples, axis=0)

    print("[INFO] Mean alpha:", mean_params[0])
    print("[INFO] Mean beta:", mean_params[1])
    print("[INFO] Mean I0:", mean_params[2])

    print("[INFO] Standard deviation alpha:", std_params[0])
    print("[INFO] Standard deviation beta:", std_params[1])
    print("[INFO] Standard deviation I0:", std_params[2])

    # Show trace plots.
    plt = make_trace_plot(all_chains, 0, r"$\alpha$")
    plt.show()
    plt = make_trace_plot(all_chains, 1, r"$\beta$")
    plt.show()
    plt = make_trace_plot(all_chains, 2, r"$I_0$")
    plt.show()

    # Show joint plot.
    param_names = [r"$\alpha$", r"$\beta$", r"$I_0$"]
    plt = make_corner_plot(np.asarray(all_iid_samples), param_names)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LightHouse Metropolis"
                                     " Hasting sampler.")
    parser.add_argument(
        "--n_chains", type=int, default=10, help="Number of chains to run."
        " Default is 10 chains."
    )
    parser.add_argument(
        "--n_samples", type=int, default=100000, help="Number of samples to"
        " generate. Default is 1e5 samples"
    )
    args = parser.parse_args()
    assert (
        args.n_chains > 1
    ), "Number of chains must be greater than 1 to calculate"
    " the Gelman-Rubin statistic."
    main(args.n_chains, args.n_samples)
