"""Pure-numpy implementation of the SEER-derived Isolation Forest (no scikit-learn needed)."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_P = np.load(os.path.join(HERE, "model_params.npz"))
META = json.load(open(os.path.join(HERE, "model_meta.json")))
FEATURES = META["features"]

def _c(n):
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    m2 = n > 2
    out[n == 2] = 1.0
    out[m2] = 2.0 * (np.log(n[m2] - 1.0) + np.euler_gamma) - 2.0 * (n[m2] - 1.0) / n[m2]
    return out

def anomaly_score(X):
    """X: (n, 15) raw feature matrix in FEATURES order. Returns IF score (higher = more anomalous)."""
    X = (np.atleast_2d(np.asarray(X, float)) - _P["mean"]) / _P["scale"]
    off = _P["offsets"]; L, R, F, T, N = _P["left"], _P["right"], _P["feature"], _P["threshold"], _P["n_samples"]
    n_trees = len(off) - 1
    depth = np.zeros(len(X))
    for i, x in enumerate(X):
        tot = 0.0
        for k in range(n_trees):
            node, d, o = 0, 0, off[k]
            while L[o + node] != -1:
                node = L[o + node] if x[F[o + node]] <= T[o + node] else R[o + node]
                d += 1
            tot += d + _c([N[o + node]])[0]
        depth[i] = tot
    return 2.0 ** (-depth / (n_trees * _c([_P["max_samples"][0]])[0]))

def probability(score):
    z = META["platt_a"] + META["platt_b"] * (np.asarray(score) - 0.5) * 10
    return 1.0 / (1.0 + np.exp(-z))

def percentile(score):
    return float(np.interp(score, _P["score_percentiles"], np.arange(0, 101)))

def build_features(age, sex, histotype, grade, stage, resection, nodes_assessed, nodes_examined, nodes_positive, rural):
    x = {f: 0.0 for f in FEATURES}
    x["Age"] = age
    x["Male sex"] = 1.0 if sex == "Male" else 0.0
    x["Gastrinoma"] = 1.0 if histotype == "Gastrinoma" else 0.0
    x["Glucagonoma"] = 1.0 if histotype == "Glucagonoma" else 0.0
    x["Other functioning histotype"] = 1.0 if histotype in ("VIPoma", "Somatostatinoma", "ACTH-producing") else 0.0
    x["Grade"] = {"G1": 1, "G2": 2, "G3": 3}[grade]
    x["Stage"] = {"Localized": 1, "Regional": 2, "Distant": 3}[stage]
    x["Partial pancreatectomy"] = 1.0 if resection == "Partial (distal/central) pancreatectomy" else 0.0
    x["Pancreaticoduodenectomy"] = 1.0 if resection == "Pancreaticoduodenectomy" else 0.0
    x["Total/extended pancreatectomy"] = 1.0 if resection == "Total/extended pancreatectomy" else 0.0
    if nodes_assessed:
        x["Lymph nodes examined"] = nodes_examined; x["Positive lymph nodes"] = nodes_positive; x["Nodes not assessed"] = 0.0
    else:
        x["Lymph nodes examined"] = META["median_nodes"]; x["Positive lymph nodes"] = 0.0; x["Nodes not assessed"] = 1.0
    x["Rural hospital area"] = 1.0 if rural else 0.0
    return np.array([[x[f] for f in FEATURES]])
