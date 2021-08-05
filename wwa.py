""""
The G-Wishart weighted proposal algorithm

This Python module contains functions for posterior computation using Markov
chain Monte Carlo (MCMC) for Bayesian Gaussian graphical models. Specifically,
it implements the G-Wishart weighted proposal algorithm
(WWA, van den Boom et al., 2021, arXiv:2108.01308) and the double conditional
Bayes factor (DCBF, Hinne et al., 2014, doi:10.1002/sta4.66) sampler.
"""

import os
import platform
import subprocess
import time

os.environ["EXTRA_CLING_ARGS"] = "-I/usr/local/include"

# Set number of cores used by the BLAS. More than 1 core does not necessarily
# yield a speed up. By not setting "MKL_NUM_THREADS", multithreading is
# automatically disabled by Intel MKL inside an OpenMP parallel environment.
os.environ["MKL_NUM_THREADS"] = "1"

import cppyy
import igraph
import numpy as np
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages


# Load C++ functions.
_platform_name = platform.system()

if _platform_name == "Darwin":
    _compiler_opts = "g++ -march=native"
elif _platform_name == "Linux":
    # `-march=native` yields "architecture not recognized" on the Yale-NUS
    # virtual machine.
    _compiler_opts = "/usr/bin/clang++-10 -march=nocona"
else:
    raise NotImplementedError(
        "Compiling has only been tested on macOS and Linux."
    )

_binary_file_name = "wwa_" + _platform_name + ".so"
print("Compiling `wwa.cpp`...")

subprocess.run(
    _compiler_opts + " -shared -fPIC -I/usr/local/include -std=c++17 " \
        + "-L$CONDA_PREFIX/lib -L/usr/local -Wl,-rpath,$CONDA_PREFIX/lib " \
        + "-lmkl_rt -ligraph -Xpreprocessor -fopenmp -lomp -O2 -DNDEBUG -o " \
        + _binary_file_name + " wwa.cpp", shell=True, check=True
)

print("Finished compiling.")
cppyy.include("wwa.h")
cppyy.load_library(_binary_file_name)

# Reduce Python overhead:
_rgwish_cpp = cppyy.gbl.rgwish_cpp
_rgwish_identity_cpp = cppyy.gbl.rgwish_identity_cpp
_update_G_cpp = cppyy.gbl.update_G_cpp
_update_G_DCBF_cpp = cppyy.gbl.update_G_DCBF_cpp


_large_int = 2**63


def random_seed(rng):
    # The following is faster than `rng.integers(_large_int)`.
    return int(_large_int * rng.random())


def rgwish(G, df, rate, rng):
    """Sample from the G-Wishart distribution."""
    K = np.empty(2 * (G.vcount(),))
    
    try:
        _rgwish_cpp(K, G.__graph_as_capsule(), df, rate, random_seed(rng))
    except:
        print("Error in `_rgwish_cpp`. Retrying...")
        return rgwish(G, df, rate, rng)
    
    return K


def rgwish_identity(G, df, rng):
    """
    Sample from the G-Wishart distribution with an identity scale matrix.
    """
    K = np.empty(2 * (G.vcount(),))
    
    try:
        _rgwish_identity_cpp(K, G.__graph_as_capsule(), df, random_seed(rng))
    except:
        print("Error in `_rgwish_identity_cpp`. Retrying...")
        return rgwish_identity(G, df, rng)
    
    return K


df_0 = 3.0  # Degrees of freedom of the G-Wishart prior


def MCMC_update(
    G, edge_prob_mat, df, rate, rng, delayed_accept=True, loc_bal=True,
    DCBF=False
):
    par_time = 0.0  # Time spent in parallel computations
    
    # MCMC step
    p = G.vcount()
    adj = np.array(G.get_adjacency().data, dtype=int)
    
    if DCBF:
        try:
            _update_G_DCBF_cpp(
                p, adj, edge_prob_mat, df, df_0, rate, random_seed(rng)
            )
        except:
            print("Error in `_update_G_DCBF_cpp`.")
    else:
        try:
            par_time = _update_G_cpp(
                p, adj, edge_prob_mat, df, df_0, rate, p, random_seed(rng),
                False,  # approx
                delayed_accept, loc_bal
            )
        except:
            print("Error in `_update_G_cpp`.")
    
    G_tilde = igraph.Graph.Adjacency(adj.tolist(), mode="undirected")
    return G_tilde, par_time


def MCMC(
    G_init, n_iter, data, edge_prob=0.5,
    rng=np.random.Generator(np.random.SFC64(seed=0)),
    verbose=True, delayed_accept=True, loc_bal=True, DCBF=False
):
    n, p = data.shape
    U = data.T @ data
    df = df_0 + n  # Degrees of freedom of the G-Wishart posterior
    rate = np.eye(p) + U  # Rate matrix of the G-Wishart posterior
    edge_prob_mat = np.full((p, p), fill_value=0.5)
    n_edges = np.empty(n_iter, dtype=int)
    elapsed_time = np.empty(n_iter)
    par_time = np.empty(n_iter)
    G = G_init.copy()
    t0 = time.time()
    
    for s in range(n_iter):
        if verbose and s % 100 == 0:
            print("Iteration", s, end="\r")
        
        G, par_time[s] = MCMC_update(
            G, edge_prob_mat, df, rate, rng, delayed_accept, loc_bal, DCBF
        )

        n_edges[s] = G.ecount()
        elapsed_time[s] = time.time() - t0
    
    return {
        "last G": G,
        "n_edges": n_edges,
        "elapsed_time": elapsed_time,
        "par_time": par_time.cumsum(),
    }


# We compute the integrated autocorrelation time (IAT) using the R package
# `LaplacesDemon`.
rpy2.robjects.numpy2ri.activate()  # Enable passing NumPy arrays to R.

if not rpackages.isinstalled("LaplacesDemon"):
    rpackages.importr("utils").install_packages(
        "LaplacesDemon", repos="https://cloud.r-project.org"
    )


def IAT(vec):
    """Compute the integrated autocorrelation time."""
    return rpackages.importr("LaplacesDemon").IAT(vec)[0]


def CIS(vec, time, par_time=0.0, n_cores=128):
    """
    Compute the cost of an independent sample.

    par_time -- time spent in parallel computations (default 0.0)
    n_cores -- number of CPU cores used for parallel computations (default 128)
    """
    time_128 = time - par_time + par_time*n_cores/128.0
    return IAT(vec) * time_128 / len(vec)


def CIS(res, n_cores=128, burnin=0):
    """
    Compute the cost of an independent sample.

    par_time -- time spent in parallel computations (default 0.0)
    n_cores -- number of CPU cores used for parallel computations (default 128)
    """
    n_edges = res["n_edges"][burnin:]
    time = res["elapsed_time"][-1]
    par_time=res["par_time"][-1]

    if burnin > 0:
        time -= res["elapsed_time"][burnin - 1]
        par_time -= res["par_time"][burnin - 1]

    time_128 = time - par_time + par_time*n_cores/128.0
    return IAT(n_edges) * time_128 / len(n_edges)