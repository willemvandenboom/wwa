# wwa

Repository with the code used for the paper "The *G*-Wishart Weighted Proposal
Algorithm: Efficient Posterior Computation for Gaussian Graphical Models" by
Willem van den Boom, Alexandros Beskos and Maria De Iorio
([arXiv:2108.01308](https://arxiv.org/abs/2108.01308) and
[doi:10.1080/10618600.2022.2050250]).

The implementation uses Python and C++ code. For a version using Rcpp that
might be easier to compile, see [here]. Note that the Rcpp version does not use
graph decomposition when sampling from the *G*-Wishart distribution, unlike the
version in this repository.


## Description of files

* [`wwa.py`](wwa.py) is a Python module that provides an implementation of the
*G*-Wishart weighted proposal algorithm (WWA). The other Python scripts import
it. It uses the C++ code in [`wwa.cpp`](wwa.cpp) via the header file
[`wwa.h`](wwa.h).

* [`cycle.py`](cycle.py) produces the figure for the simulation study with
cycle graphs.

* [`uniform.py`](uniform.py) produces the figure for the simulation study with
uniformly sampled graphs.

* [`gene.py`](gene.py) produces the results for the application to the gene
expression data in[`data/geneExpression.csv`](data/geneExpression.csv).

* The scripts in the folder [appendix](appendix/) produce additional empirical
results for in an appendix to the paper.

* [`environment.yml`](environment.yml) details the conda environment used for
the paper. It can be used to [recreate the environment]. The dependencies of
[`wwa.cpp`](wwa.cpp) are detailed preceding the respective include directives.


[doi:10.1080/10618600.2022.2050250]: https://doi.org/10.1080/10618600.2022.2050250
[here]: https://github.com/willemvandenboom/graph-sphere/blob/fbd881fef37003a3eaaa60723a8a73a3d2979b60/graph_sphere_MCMC.R#L40-L73
[recreate the environment]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
