"""
This script produces the results for the application to gene expression data in
van den Boom et al. (2021, arXiv:2108.01308).
"""

import igraph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import p_tqdm
import scipy.special
import sklearn.covariance

import wwa


# Gene expression data from Mohammadi & Wit (2015, doi:10.1214/14-BA889),
# available in the R package `BDgraph` as `geneExpression`.
# The gene expression are already ordered from most to least variable.
data_orig = np.loadtxt(fname="data/geneExpression.csv", delimiter=",")
n, p = data_orig.shape
data = data_orig.copy()

# Quantile-normalize the data to marginally follow a standard Gaussian distribution.
for j in range(p):
    data[:, j] = scipy.special.ndtri(
        pd.Series(data[:, j]).rank(method="average") / (n + 1)
    )

p_list = [50, 100]
G_init = 2 * [None]

for p_ind in range(2):
    p = p_list[p_ind]

    prec = sklearn.covariance.GraphicalLassoCV(
        n_jobs=-1, assume_centered=True
    ).fit(data[:, :p]).precision_

    np.fill_diagonal(prec, 0.0)

    G_init[p_ind] = igraph.Graph.Adjacency(
        matrix=(prec != 0.0).tolist(), mode="undirected"
    )


n_iter = 16000
burnin = 6000
res_mat = np.empty((2, 3), dtype=object)
rng = np.random.Generator(np.random.SFC64(seed=0))
print("Running WWA for p = 50...")

res_mat[0, 0] = wwa.MCMC(
    G_init=G_init[0], n_iter=n_iter, data=data[:, :p_list[0]], rng=rng
)

print("Running WWA for p = 100...")

res_mat[1, 0] = wwa.MCMC(
    G_init=G_init[1], n_iter=n_iter, data=data, rng=rng, delayed_accept=False,
    loc_bal=False
)

par_seed = wwa.random_seed(rng)


def par_func(s):
    """Function with simple counter as argument for use with `p_tqdm`."""
    p_ind = int(s > 0)
    last = s == 2

    return wwa.MCMC(
        G_init=res_mat[1, 0]["last G"] if last else G_init[p_ind],
        n_iter=n_iter - burnin if last else n_iter,
        data=data[:, :p_list[p_ind]], rng=np.random.Generator(np.random.SFC64(
            seed=np.random.SeedSequence(entropy=par_seed, spawn_key=(s,))
        )), DCBF=True
    )


result_list = p_tqdm.p_map(par_func, range(3))

for s in range(3):
    p_ind = int(s > 0)
    last = s == 2
    res_mat[p_ind, 1 + last] = result_list[s]


n_cores = p_tqdm.p_tqdm.cpu_count()
print("Time spent on the first", burnin, "iterations in seconds:")

for p_ind in range(2):
    for res_ind in range(2):
        res = res_mat[p_ind, res_ind]
        par_time = res["par_time"][burnin - 1]

        print(
            res["elapsed_time"][burnin - 1] - par_time + par_time*n_cores/128.0
        )


# Create the trace plot for p = 100.
fig, ax = plt.subplots(figsize=(8, 4))

for res_ind in range(2):
    res = res_mat[1, 1 - res_ind]
    
    ax.plot(
        res["n_edges"],
        ls=[(0, (4, 4)), (0, (0.5, 0.5))][res_ind],
        color=plt.rcParams['axes.prop_cycle'].by_key()['color'][res_ind + 1],
        label=["DCBF", "No delayed acceptence nor informed proposal"][res_ind],
        zorder = 1 - res_ind
    )

ax.legend()
ax.set_ylim((1000, 1260))
ax.set_xlabel("Iteration number")
ax.set_ylabel("Number of edges")
fig.savefig("gene.pdf")


print("Cost of an independent sample in seconds:")

for res in [res_mat[0, 0], res_mat[0, 1], res_mat[1, 0]]:
    print(wwa.CIS(res, n_cores=n_cores, burnin=burnin))

print(wwa.CIS(res=res_mat[1, 2], n_cores=n_cores))


# Save the MCMC chains to compute their R hat using the function `Rhat` from
# the R package `rstan`.
gene_chains = np.empty((4, n_iter - burnin), dtype=int)

for i in range(4):
    gene_chains[i, :] = res_mat[i // 2, i % 2]["n_edges"][burnin:]

np.savetxt(fname="data/gene_chains.csv", X=gene_chains.T, delimiter=",")
