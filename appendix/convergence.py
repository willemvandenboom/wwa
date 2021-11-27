"""
This script compares convergence of the method from van den Boom et al.
(2021, arXiv:2108.01308) across two different approximations for the ratio of
normalizing constants.
"""

import igraph
import matplotlib.pyplot as plt
import numpy as np

import wwa


rng = np.random.Generator(np.random.SFC64(seed=0))
p = 40
K = np.eye(p)
K[0, -1] = 0.4
K[-1, 0] = K[0, -1]

for i in range(1, p):
    K[i, i - 1] = 0.5
    K[i - 1, i] = K[i, i - 1]

data = rng.multivariate_normal(
    mean=np.zeros(p), cov=np.linalg.inv(K), size=3 * p // 2
)

# Initialize at the empty graph.
G_init = igraph.Graph()
G_init.add_vertices(p)
res_mat = np.empty((2,), dtype=object)
rng = np.random.Generator(np.random.SFC64(seed=0))

for ind in range(2):
    print("Running MCMC", ind)
    
    res_mat[ind] = wwa.MCMC(
        G_init=G_init, n_iter=10**3, data=data, rng=rng, Letac=ind == 1
    )

fig, ax = plt.subplots(figsize=(8, 4))

for res_ind in [1, 0]:
    res = res_mat[res_ind]
    
    ax.plot(
        res["n_edges"], ls=[(0, (4, 4)), "solid"][res_ind],
        color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1 - res_ind],
        label=[
            "Approximation equal to one",
            "Approximation from Mohammadi et al. (2021)"
        ][res_ind]
    )

ax.legend()
ax.set_xlabel("Iteration number")
ax.set_ylabel("Number of edges")
ax.set_xscale("log")
fig.savefig("convergence.pdf")
