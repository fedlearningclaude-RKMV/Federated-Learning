"""FedRF-Diff: Federated Random Forest with tree-level aggregation and decentralized convergence.

Four hospitals (horizontal clients) exchange complete Random Forests as explicit decision trees:
no raw records or latent encodings leave a hospital, the server never averages parameters and
never sees a validation metric.

    1  Initialisation        F_k^(0) per client, G^(0) empty, q_k = 0                  client.py
    2  Client upload         complete forest F_k^(t-1) + schema metadata + ZKP          client.py, zkp.py
    3  Tree-level aggregation  validate, pool U^(t), G^(t) = SelectTrees(U^(t), A, B)    server.py
    4  Broadcast             G^(t), round t and the verified schema                     pipeline.py
    5  Local refinement      F_k^(t) = CapOrPrune(TreePool(G^(t), H_k^(t)), B_k)       client.py, rf_kernel.py
    6  Local convergence     loss stability, prediction stability, G/F agreement;     convergence.py
                             only b_k = [q_k >= P] is sent
    7  Server stopping       all b_k True or t = R_max                                  server.py, pipeline.py

Supporting modules: data.py and alignment.py (hospital split, schema, federated imputation),
privacy.py (Laplace DP on new leaf vectors), metrics.py, report.py, plotting.py, config.py.

Run:   python -m fedrf_diff
"""
