
from samplers import metropolis_sampler
from models import uniform_2d, log_p, log_p2, log_uniform
from utils import gelman_rubin, make_trace_plot, make_joint_plot
from scipy.stats import loguniform

import numpy as np
import argparse
import matplotlib.pyplot as plt
from emcee.autocorr import integrated_time
import matplotlib.pyplot as plt
import random
from corner import corner

random.seed(42)
np.random.seed(42)


def main(n_chains, n_samples):
    # Read the positional data
    with open('lighthouse_flash_data.txt','r') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            fmt_line = line.rstrip().split(' ')
            x = float(fmt_line[0])
            i = float(fmt_line[1])
            data.append([x, i])

    # a, b = -100, 100
    # c, d = 0, 50
    # cov_Q = np.eye(2)
    # all_chains = np.zeros((n_chains, n_samples, 2))
    # all_iid_samples = []

    # for i in range(n_chains):
    #     print('[INFO] Running chain', i)
    #     # Draw x0 from the prior.
    #     alpha = np.random.uniform(a, b)
    #     beta = np.random.uniform(c, d)
    #     x0 = np.asarray([alpha, beta])

    #     # Sample the posterior.
    #     samples, acceptance = metropolis_sampler(lambda theta: log_p2(theta, data, uniform_2d), x0, n_samples, cov_Q)
    #     all_chains[i] = samples

    #     # Discard the burn-in.
    #     samples = samples[int(0.1*n_samples):]
    #     tau = max(integrated_time(samples[:, 0]), integrated_time(samples[:, 1]))
    #     thinning = int(2*tau)
    #     iid_samples = samples[::thinning]
    #     print('[INFO] Integrated autocorrelation time:', tau)
    #     print('[INFO] Number of samples:', len(iid_samples))
    #     all_iid_samples.extend(iid_samples)

    # gelman_rubin_statistic = gelman_rubin(all_chains)

    # print('[INFO] Gelman-Rubin statistic:', gelman_rubin_statistic)

    # mean_params = np.mean(all_iid_samples, axis=0)
    # std_params = np.std(all_iid_samples, axis=0)

    # print('[INFO] Mean alpha:', mean_params[0])
    # print('[INFO] Mean beta:', mean_params[1])
    # print('[INFO] Standard deviation alpha:', std_params[0])
    # print('[INFO] Standard deviation beta:', std_params[1])

    # np.save('samples_1.npy', all_iid_samples)
    # ------------------
    # Code for part vii.
    # ------------------

    # Prior limits for alpha.
    a, b = -100, 100
    # Prior limits for beta.
    c, d = 0, 50
    # Prior limits for I0.
    e, f = 1e-5, 1e3

    cov_Q = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    all_chains = np.zeros((n_chains, n_samples, 3))
    all_iid_samples = []
    for i in range(n_chains):
        print('[INFO] Running chain', i)
        # Draw x0 from the prior.
        alpha = np.random.uniform(a, b)
        beta = np.random.uniform(c, d)
        I0 = loguniform.rvs(e, f)

        x0 = np.asarray([alpha, beta, I0])

        # Sample the posterior.
        samples, acceptance = metropolis_sampler(lambda theta: log_p2(theta, data, uniform_2d, log_uniform), x0, n_samples, cov_Q)
        all_chains[i] = samples

        # Discard the burn-in.
        samples = samples[int(0.1*n_samples):]
        tau = max(integrated_time(samples[:, 0]), integrated_time(samples[:, 1]), integrated_time(samples[:, 2]))
        thinning = int(2*tau)
        iid_samples = samples[::thinning]
        print('[INFO] Integrated autocorrelation time:', tau)
        print('[INFO] Number of samples:', len(iid_samples))
        all_iid_samples.extend(iid_samples)

    gelman_rubin_statistic = gelman_rubin(all_chains)
    print('[INFO] Gelman-Rubin statistic:', gelman_rubin_statistic)
    mean_params = np.mean(all_iid_samples, axis=0)
    std_params = np.std(all_iid_samples, axis=0)

    print('[INFO] Mean alpha:', mean_params[0])
    print('[INFO] Mean beta:', mean_params[1])
    print('[INFO] Mean I0:', mean_params[2])

    print('[INFO] Standard deviation alpha:', std_params[0])
    print('[INFO] Standard deviation beta:', std_params[1])
    print('[INFO] Standard deviation I0:', std_params[2])
    np.save('samples_2.npy', all_iid_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Metropolis-Hastings chains.")
    parser.add_argument("--n_chains", type=int, default=5, help="Number of chains to run.")
    parser.add_argument("--n_samples", type=int, default=100000, help="Number of samples to generate.")
    args = parser.parse_args()
    main(args.n_chains, args.n_samples)
