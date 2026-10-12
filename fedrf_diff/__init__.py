"""FedRF-Diff: federated Random Forest with whole-tree sharing and a combined convergence criterion.

Four hospitals (horizontal clients) train local Random Forests on their own patients and send
the server complete, explicit decision trees: no raw records, embeddings, latent codes,
feature vectors or split histograms leave a hospital.

    1  Schema agreement      feature order, missing values, categories, classes   alignment.py
    2  Local preprocessing   federated imputation and one-hot encoding            alignment.py
    3  Local training        m new Random Forest trees per client per round       rf_kernel.py, client.py
    4  Explicit export       trees as nodes, Laplace DP on every leaf vector      rf_kernel.py, privacy.py
    5  Whole-tree transfer   serialized update + Modified Pedersen ZKP            client.py, zkp.py
    6  Server validation     proof, schema, node references, splits, leaves      server.py
    7  Global forest update  concatenate the valid trees (no averaging)          server.py
    8  Global evaluation     clients score G_t and return counts only           client.py, metrics.py
    9  Combined convergence  Delta M <= eps_M and Delta P <= eps_P for H rounds  convergence.py
   10  Stop or continue      until converged or R rounds; report and figures    pipeline.py, report.py, plotting.py

Run:   python -m fedrf_diff
"""
