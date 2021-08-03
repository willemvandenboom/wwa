#include <igraph/igraph.h>

void rgwish_cpp(
    double* K_out, igraph_t* G_ptr, double df, double* rate_in, long seed
);

void rgwish_identity_cpp(
    double* K_out, igraph_t* G_ptr, double df, long seed
);

double update_G_cpp(
    int p, long* adj_in, double* edge_prob_mat_in, double df, double df_0,
    double* rate_in, int n_edge, long seed, bool approx = false,
    bool delayed_accept = true, bool loc_bal = true
);

void update_G_DCBF_cpp(
    int p, long* adj_in, double* edge_prob_mat_in, double df, double df_0,
    double* rate_in, long seed
);