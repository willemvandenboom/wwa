"""
This script compares the method from van den Boom et al.
(2021, arXiv:2108.01308) with other MCMC methods for posterior inference in
Gaussian graphical models with the G-Wishart prior.
"""

import time

import igraph
import numpy as np
import p_tqdm
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages
import sklearn.datasets

import wwa


# We use the R package `BDgraph`.
rpy2.robjects.numpy2ri.activate()  # Enable passing NumPy arrays to R.

if not rpackages.isinstalled("BDgraph"):
    print("Installing BDgraph...")
    
    rpackages.importr("utils").install_packages(
        "BDgraph", repos="https://cloud.r-project.org", type="binary"
    )
    
    print("Finished installing BDgraph.")


def G_equal(G0, G1):
    """Check whether graphs `G0` and `G1` are equal"""

    if G0.ecount() != G1.ecount():
        return False
    
    return sorted(G0.get_edgelist()) == sorted(G1.get_edgelist())


def MCMC_update(G, K, df, rate, rng, algorithm="WWA"):
    t0 = time.time()

    if algorithm == "CL":
        G, K = wwa.MCMC_update_CL(G, K, edge_prob_mat, df, rate, rng)
        elapsed_time = time.time() - t0
        par_time = 0.0
    else:

        G, par_time = wwa.MCMC_update(
            G, edge_prob_mat, df, rate, rng, loc_bal=algorithm == "WWA",
            DCBF=algorithm == "DCBF"
        )

        elapsed_time = time.time() - t0
        K = wwa.rgwish(G, df, rate, rng)
    
    return G, K, elapsed_time, par_time


def MCMC(df, rate, burnin, recorded, algorithm="WWA"):
    res = np.zeros((p, p), dtype=int)
    K_hat = np.zeros((p, p))
    n_edges = np.empty(recorded, dtype=int)
    elapsed_time = np.empty(recorded)
    elapsed_time = np.empty(recorded)
    par_time = np.empty(recorded)
    G = igraph.Graph()
    G.add_vertices(p)
    G_cycle = igraph.Graph.Ring(p)
    cycle_prob = 0
    K = wwa.rgwish(G, df, rate, rng)
    print_interval = recorded // 1000
    
    for s in range(burnin):
        if s % print_interval == 0:
            print("Burnin iteration", s, end="\r")
        
        G, K, _, _ = MCMC_update(G, K, df, rate, rng, algorithm)
    
    for s in range(recorded):
        if s % print_interval == 0:
            print("Iteration", s, end="\r")
        
        G, K, elapsed_time[s], par_time[s] = MCMC_update(
            G, K, df, rate, rng, algorithm
        )

        res += np.array(G.get_adjacency(0).data)
        K_hat += K
        n_edges[s] = G.ecount()
        cycle_prob += G_equal(G_cycle, G)
    
    return {
        "Edge probability": res / recorded,
        "K_hat": K_hat / recorded,
        "n_edges": n_edges,
        "elapsed_time": elapsed_time.cumsum(),
        "par_time": par_time.cumsum(),
        "cycle_prob": cycle_prob / recorded
    }


def BDgraph_MCMC(df_0, U, n, burnin, recorded):
    t0 = time.time()
    rpackages.importr("base").set_seed(0)
    BDgraph = rpackages.importr("BDgraph")

    BDgraph_res = BDgraph.bdgraph(
        data=U, n=n, iter=burnin + recorded, burnin=burnin,
        g_prior=edge_prob_mat, df_prior=df_0, jump=1, save=True
    )
    
    elapsed_time = time.time() - t0
    G_cycle = igraph.Graph.Ring(p)

    
    def get_res(name):
        return np.array(BDgraph_res[int(
            np.where(np.array(BDgraph_res.names) == name)[0][0]
        )])
    
    
    # BDgraph string formulation of cycle graph
    graph_str_np = np.zeros(p * (p - 1) // 2, dtype=int)
    ind = 0

    for i in range(p - 1):
        ind += i + 1
        graph_str_np[ind - 1] = 1

        if i == p - 3:
            graph_str_np[ind] = 1
    
    graph_str = ""

    for tmp in graph_str_np.astype(str):
        graph_str += tmp
    
    tmp = np.where(get_res("sample_graphs") == graph_str)[0]
    
    if len(tmp) == 0:
        cycle_prob = 0.0
    else:
        graph_weights = get_res("graph_weights")
        cycle_prob = graph_weights[tmp[0]] / graph_weights.sum()
    
    return {
        "Edge probability": np.array(BDgraph.plinks(
            bdgraph_obj=BDgraph_res, round=5, burnin=0
        )),
        "K_hat": get_res("K_hat"),
        "elapsed_time": elapsed_time,
        "cycle_prob": cycle_prob
    }


algorithms = ["WWA", "WWA_no_loc_bal", "DCBF", "CL", "BDgraph"]


def get_results(burnin, recorded, algs=algorithms):
    U = data.T @ data
    df_0 = 3.0 # Degrees of freedom of the Wishart prior
    rate_0 = np.eye(p) # Rate matrix of the Wishart prior
    df = df_0 + n
    rate = rate_0 + U
    
    res = {}
    
    for algorithm in algs:
        print("Working on " + algorithm + "...")
        
        if algorithm != "BDgraph":
            res[algorithm] = MCMC(df, rate, burnin, recorded, algorithm)
            
            print(
                "Cost of an independent sample (seconds):",
                wwa.CIS(res=res[algorithm], n_cores=p_tqdm.p_tqdm.cpu_count())
            )
        else:
            res["BDgraph"] = BDgraph_MCMC(df_0, U, n, burnin, recorded)
    
    return res


rng = np.random.Generator(np.random.SFC64(seed=0))


print("Iris Virginica data set")
iris = sklearn.datasets.load_iris()

data = iris["data"][(
    iris["target"] == (iris["target_names"] == "virginica").nonzero()[0][0]
).nonzero()[0], :]

data -= data.mean(axis=0)
n, p = data.shape
edge_prob_mat = np.full((p, p), fill_value=0.5)

res_iris = get_results(burnin=10**3, recorded=10**6)


for algorithm in algorithms:
    print("Posterior edge inclusion probabilities for " + algorithm + ":")
    print(res_iris[algorithm]["Edge probability"])


# Cycle graph example from
# (Wang & Li, 2012, Section 6.2, doi:10.1214/12-EJS669).
print("Cycle graph example")

for p in [10, 100]:
    print("Working on p = " + str(p) + "...")
    n = 3 * p // 2
    A = np.eye(p)
    A[0, -1] = 0.4
    A[-1, 0] = A[0, -1]

    for i in range(1, p):
        A[i, i - 1] = 0.5
        A[i - 1, i] = A[i, i - 1]

    data = rng.multivariate_normal(
        mean=np.zeros(p), cov=np.linalg.inv(A), size=n
    )

    edge_prob_mat = np.full((p, p), fill_value=2.0 / (p - 1))
    G_cycle = igraph.Graph.Ring(p)

    if p == 100:
        algs = ["WWA_no_loc_bal", "CL", "BDgraph"]

        res_cycle = get_results(
            burnin=10**4, recorded=10**5, algs=["WWA_no_loc_bal", "BDgraph"]
        )

        res_cycle["CL"] = get_results(
            burnin=10**4, recorded=10**6, algs=["CL"]
        )["CL"]
    else:
        algs = algorithms
        res_cycle = get_results(burnin=10**3, recorded=10**6, algs=algs)


    def get_edgeprob(algorithm):
        return res_cycle[algorithm]["Edge probability"][np.triu_indices(
            p, k=1
        )]


    for algorithm in algs:
        print("\n" + algorithm + ":")
        
        print("Difference from WWA in posterior edge inclusion probabilities:")
        edgeprob = get_edgeprob(algorithm)
        tmp = np.abs(get_edgeprob(algs[0]) - edgeprob)
        print("Max:", tmp.max())
        print("Mean squared difference:", np.mean(tmp**2))
        
        print("Minimum posterior edge inclusion probability for included edges:")

        included = np.array(
            G_cycle.get_adjacency(0).data
        )[np.triu_indices(p, k=1)] == 1

        print(edgeprob[included].min())
                     
        print(
            "Maximum posterior edge inclusion probability for excluded edges:"
        )

        print(edgeprob[~included].max())
        
        print(
            "Posterior probability of the true graph:",
            res_cycle[algorithm]["cycle_prob"]
        )
        
        print(
            "Accuracy of the posterior mean for the precision matrix in " \
                + "terms of Kullback-Leibler divergence:"
        )

        K_hat = res_cycle[algorithm]["K_hat"]
        
        # Equation 16 of Mohammadi & Wit (2015, doi:10.1214/14-BA889)
        print(0.5 * (np.trace(
            A @ K_hat
        ) - p - np.linalg.slogdet(K_hat)[1] - np.linalg.slogdet(A)[1]))

        print(
            "Frobenius norm of the elementwise error in the precision matrix:"
        )

        print(np.linalg.norm(x=K_hat - np.linalg.inv(A), ord="fro"))


print(
    "Cost of an independent sample (seconds) for CL based on the last 10^5 " \
        + "iterations:"
)

res_tmp = res_cycle["CL"]

for key in ["n_edges", "elapsed_time", "par_time"]:
    res_tmp[key] = res_tmp[key][-10**5:]

print(
    "Cost of an independent sample (seconds):",
    wwa.CIS(res=res_tmp, n_cores=32)
)
