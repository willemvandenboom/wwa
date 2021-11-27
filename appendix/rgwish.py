"""
This script compares the computational speed of the G-Wishart sampler from
Lenkoski (2013, doi:10.1002/sta4.23) with the one from van den Boom et al.
(2021, arXiv:2108.01308) that uses graph decomposition.
"""


import time

import igraph
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

import wwa


def rgwish(G, df, rate, rng, method=0):
    start_time = time.perf_counter()
    
    if method == 0:
        K, max_prime = wwa.rgwish(G, df, rate, rng, get_max_prime=True)
    else:
        K = wwa.rgwish(G, df, rate, rng, decompose=False)
        max_prime = -1
    
    return K, time.perf_counter() - start_time, max_prime


def get_comp_time(p, graph, rng):
    G = igraph.Graph.Ring(p)
    G.add_edges([(0, i) for i in range(2, p - 1)])
    
    # Rate matrix as in Section 2.3 of
    # Dobra et al. (2011, doi:10.1198/jasa.2011.tm10465)
    A = np.eye(p)
    A[0, -1] = 0.4
    A[-1, 0] = A[0, -1]

    for i in range(1, p):
        A[i, i - 1] = 0.5
        A[i - 1, i] = A[i, i - 1]

    rate = np.eye(p) + 100 * np.linalg.inv(A)
    comp_time = np.empty((n_setup, n_rep))
    prime_size = np.empty((n_rep,), dtype=int)
    
    for r in range(n_rep):
        if graph == 1:
            G = igraph.Graph.Erdos_Renyi(n=p, p=0.5, directed=False)
        elif graph == 2:
            G = igraph.Graph.Erdos_Renyi(n=p, p=2.0 / (p - 1), directed=False)
        
        for m in range(n_setup):
            _, comp_time[m, r], max_prime = rgwish(G, df, rate, rng, method=m)
            
            if m == 0:
                prime_size[r] = max_prime
    
    res = np.empty((3, n_setup))
    res[0, :] = np.median(comp_time, axis=1)
    res[1:, :] = np.quantile(a=comp_time, q=[0.025, 0.975], axis=1)
    res_prime_size = np.empty((3,))
    res_prime_size[0] = prime_size.mean()
    res_prime_size[1:] = np.quantile(a=prime_size, q=[0.025, 0.975])
    return res, res_prime_size / p


df = 3.0
n_setup = 2
n_rep = 10**2
n_graph = 3
p_seq = [10, 20, 40, 80]
n_p = len(p_seq)
comp_time = np.zeros((n_p, 3, n_setup))
prime_size = np.empty((n_p, 3))
rng = np.random.Generator(np.random.SFC64(seed=0))
CI_width = 0.6 / n_setup
fig, ax = plt.subplots(nrows=n_graph, ncols=2, figsize=(8, 8))

# Run `rgwish` once to avoid the `cppyy` overhead from the first run in the
# benchmarking.
rgwish(igraph.Graph.Ring(2), df, np.eye(2), rng)

for graph in range(n_graph):
    for ind in range(n_p):
        print("Working on graph =", graph, " p =", p_seq[ind], end="\r")

        comp_time[ind, :, :], prime_size[ind, :] = get_comp_time(
            p=p_seq[ind], graph=graph, rng=rng
        )

    for p_ind in range(n_p):
        for setup_ind in range(n_setup):
            x_offset = 0.8 * (setup_ind - 0.5*(n_setup - 1)) / n_setup
            col = plt.rcParams['axes.prop_cycle'].by_key()['color'][setup_ind]

            # 2.5th and 97.5th quantiles
            quantiles = comp_time[p_ind, 1:, setup_ind]

            ax[graph, 0].add_patch(plt.Rectangle(
                (p_ind + x_offset - 0.5*CI_width, quantiles[0]), CI_width,
                quantiles[1] - quantiles[0],
                # We do a manual alpha (color transparancy) as otherwise extra
                # rectangle lines appear in the PDF output.
                color=1.0 - 0.3*(1.0 - np.array(matplotlib.colors.to_rgb(col)))
            ))

            ls = ["solid", (0, (4, 4))][setup_ind]

            if p_ind == 0:  # Set label only once
                ax[graph, 0].hlines(
                    y=comp_time[p_ind, 0, setup_ind],
                    xmin=p_ind + x_offset - 0.5*CI_width,
                    xmax=p_ind + x_offset + 0.5*CI_width,
                    linestyles=ls, color=col,
                    label=[
                        "With decomposition", "Without decomposition"
                    ][setup_ind]
                )
            else:
                ax[graph, 0].hlines(
                    y=comp_time[p_ind, 0, setup_ind],
                    xmin=p_ind + x_offset - 0.5*CI_width,
                    xmax=p_ind + x_offset + 0.5*CI_width,
                    linestyles=ls, color=col
                )

        # Plot number of nodes of largest prime component.
        col = plt.rcParams['axes.prop_cycle'].by_key()['color'][n_setup]

        # 2.5th and 97.5th quantiles
        quantiles = prime_size[p_ind, 1:]

        ax[graph, 1].add_patch(plt.Rectangle(
            (p_ind - 0.5*CI_width, quantiles[0]), CI_width,
            quantiles[1] - quantiles[0],
            # We do a manual alpha (color transparancy) as otherwise extra
            # rectangle lines appear in the PDF output.
            color=1.0 - 0.3*(1.0 - np.array(matplotlib.colors.to_rgb(col)))
        ))

        ax[graph, 1].hlines(
            y=prime_size[p_ind, 0], xmin=p_ind - 0.5*CI_width,
            xmax=p_ind + 0.5*CI_width, color=col
        )

    for i in range(2):
        ax[graph, i].set_xticks(np.arange(n_p))
        ax[graph, i].set_xticklabels(p_seq)

    ax[graph, 0].set_yscale("log")
    ax[graph, 0].set_ylabel("Cost of a sample (seconds)")
    ax[graph, 1].set_ylabel("Prop. of nodes in largest prime")
    ax[graph, 1].set_ylim((0, 1.1))
    
    ax[graph, 0].set_title([
        "Decomposable graph", r"Random with $\rho = 0.5$",
        r"Random with $\rho = 2/(p-1)$"
    ][graph])


for i in range(2):
    ax[-1, i].set_xlabel("Number of nodes")

ax[0, 0].legend()
fig.tight_layout()
fig.savefig("rgwish.pdf")
