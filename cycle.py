"""
This script produces the figure for the simulation study with cycle graphs in
van den Boom et al. (2021, arXiv:2108.01308).
"""

import igraph
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import p_tqdm

import wwa


p_seq = [10, 20, 40]
n_p = len(p_seq)
n_rep = 32

# Generate the simulated data from a cycle graph following Section 6.2 of
# Wang & Li (2012, doi:10.1214/12-ejs669).
data_simul = [np.empty((n_rep, 3 * p // 2, p)) for p in p_seq]
rng = np.random.Generator(np.random.SFC64(seed=0))

for p_ind in range(n_p):
    p = p_seq[p_ind]
    K = np.eye(p)
    K[0, -1] = 0.4
    K[-1, 0] = K[0, -1]

    for i in range(1, p):
        K[i, i - 1] = 0.5
        K[i - 1, i] = K[i, i - 1]
    
    for r in range(n_rep):
        data_simul[p_ind][r, :, :] = rng.multivariate_normal(
            mean=np.zeros(p), cov=np.linalg.inv(K),
            size=data_simul[p_ind][r, :, :].shape[0]
        )


n_setup = 5
res_mat = np.empty((n_p, n_rep, n_setup), dtype=object)


for p_ind in range(n_p):
    p = p_seq[p_ind]
    print("Working on p =", p)


    def MCMC(r, rng, delayed_accept=True, loc_bal=True, DCBF=False):
        data = data_simul[p_ind][r, :, :]

        return wwa.MCMC(
            G_init=igraph.Graph.Ring(p), n_iter=11000, data=data,
            edge_prob=2.0 / (data.shape[1] - 1), rng=rng, verbose=False,
            delayed_accept=delayed_accept, loc_bal=loc_bal, DCBF=DCBF
        )


    par_seed = wwa.random_seed(rng)


    def par_func(s):
        """Function with simple counter as argument for use with `p_tqdm`."""
        setup_ind = s % 3

        return MCMC(
            r=s // 3, rng=np.random.Generator(np.random.SFC64(
                seed=np.random.SeedSequence(entropy=par_seed, spawn_key=(s,))
            )), delayed_accept=setup_ind==2, loc_bal=False, DCBF=setup_ind == 0
        )

    
    # We do not use all CPU cores for p = 40 to avoid running out of memory.
    result_list = p_tqdm.p_map(
        par_func, range(3 * n_rep), num_cpus=4 if p == 40 else None
    )
    
    for r in range(n_rep):
        for setup_ind in range(3):
            res_mat[p_ind, r, 1 + setup_ind] = result_list[3*r + setup_ind]
        
        for setup_ind in [0, 4]:
            delayed_accept = setup_ind==0

            print(
                "Running WWA", "with" if delayed_accept else "without",
                "delayed acceptance for p =", p, "  r =", r
            )

            res_mat[p_ind, r, setup_ind] = MCMC(
                r, rng, delayed_accept=delayed_accept
            )


# Create the plot for this simulation.
CIS_mat = np.empty((n_p, n_rep, n_setup))

for p_ind in range(n_p):
    for r in range(n_rep):
        for setup_ind in range(n_setup):
            CIS_mat[p_ind, r, setup_ind] = wwa.CIS(
                res_mat[p_ind, r, setup_ind],
                n_cores=p_tqdm.p_tqdm.cpu_count(), burnin=1000
            )

CI_width = 0.6 / n_setup
fig, ax = plt.subplots(figsize=(8, 4))

for setup_ind in range(n_setup):
    x_offset = 0.8 * (setup_ind - 0.5*(n_setup - 1)) / n_setup
    col = plt.rcParams['axes.prop_cycle'].by_key()['color'][setup_ind]
    
    # We do a manual alpha as otherwise extra rectangle lines appear in the PDF
    # output.
    col_light = 1.0 - 0.3*(1.0 - np.array(matplotlib.colors.to_rgb(col)))
    
    for p_ind in range(n_p):
        CIS_vec = CIS_mat[p_ind, :, setup_ind]

        # Bootstrapping to get 2.5th and 97.5th percentiles
        CIS_CI = np.percentile(a=rng.choice(
            a=CIS_vec, size=(1000, n_rep), replace=True
        ).mean(axis=1), q=[2.5, 97.5])
        
        ax.add_patch(plt.Rectangle(
            (p_ind + x_offset - 0.5*CI_width, CIS_CI[0]), CI_width,
            CIS_CI[1] - CIS_CI[0],
            # We do a manual alpha (color transparancy) as otherwise extra
            # rectangle lines appear in the PDF output.
            color=1.0 - 0.3*(1.0 - np.array(matplotlib.colors.to_rgb(col)))
        ))

        ls = [
            "solid", (0, (4, 4)), (0, (0.5, 0.5)), (0, (2, 2)),
            (0, (2, 1, 1, 1))
        ][setup_ind]
        
        if p_ind == 0:  # Set label only once
            ax.hlines(
                y=CIS_vec.mean(), xmin=p_ind + x_offset - 0.5*CI_width,
                xmax=p_ind + x_offset + 0.5*CI_width, linestyles=ls, color=col,
                label=[
                    "WWA", "DCBF",
                    "No delayed acceptence nor informed proposal",
                    "No informed proposal", "No delayed acceptance"
                ][setup_ind]
            )
        else:
            ax.hlines(
                y=CIS_vec.mean(), xmin=p_ind + x_offset - 0.5*CI_width,
                xmax=p_ind + x_offset + 0.5*CI_width, linestyles=ls, color=col
            )

ax.set_yscale("log")
ax.legend()
ax.set_xticks(np.arange(n_p))
ax.set_xticklabels(p_seq)
ax.set_xlabel("Number of nodes")
ax.set_ylabel("Cost of an independent sample (seconds)")
fig.savefig("cycle.pdf")


print("Relative efficiency of WWA versus DCBF:")
print(CIS_mat[:, :, 1].mean(axis=1) / CIS_mat[:, :, 0].mean(axis=1))