# wwa

Repository with the code used for the paper "The *G*-Wishart Weighted Proposal
Algorithm: Efficient Posterior Computation for Gaussian Graphical Models" by
Willem van den Boom, Alexandros Beskos and Maria De Iorio (in preparation).


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

* [`environment.yml`](environment.yml) details the conda environment used for
the paper. It can be used to [recreate the environment]. The dependencies of
[`wwa.cpp`](wwa.cpp) are detailed preceding the respective include directives.


[recreate the environment]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file