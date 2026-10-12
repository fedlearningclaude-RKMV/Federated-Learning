"""Laplace differential privacy on the shared leaf vectors.

What is protected. In whole-tree sharing the per-patient statistics a client releases are the
leaf outputs: every leaf carries the class-probability vector of the training patients that
reach it. Before export each leaf vector v, L1-clipped to C, is released as

    v~ = v + Lap(b)^(n_classes),    b = 2C / epsilon

Any change to the patients behind a leaf moves v by at most ||v - v'||_1 <= 2C, so every
released leaf vector is epsilon-DP (pure DP, delta = 0). v~ is then clamped at 0 and
renormalised so the server receives a valid distribution; post-processing costs no privacy.

Accounting. A patient reaches one leaf per tree (the leaves partition the tree's training
sample: parallel composition), so a tree costs epsilon per patient; the m trees a client sends
in a round compose sequentially to m * epsilon, and rounds add up (basic composition). Trees that
are only re-sent (a client's kept trees, the global forest) are post-processing and free.

Scope. Split features and thresholds are chosen on the raw records and are not privatised,
so the shared trees as a whole are not an end-to-end DP release: the guarantee covers the leaf
outputs given the tree structure. The noise comes from numpy's seeded generator for
reproducible experiments, not from a hardened DP sampler.
"""

import numpy as np


def laplace_scale(epsilon, clip):
    return 2 * clip / epsilon


def clip_l1(vectors, clip):
    norms = np.abs(vectors).sum(axis=1, keepdims=True)
    return vectors * np.minimum(1.0, np.divide(clip, norms, out=np.ones_like(norms), where=norms > 0))


def privatize_leaves(value, is_leaf, epsilon, clip, rng):
    """Laplace mechanism on every leaf vector; internal-node rows are never sent and stay as they are."""
    out = value.copy()
    v = clip_l1(value[is_leaf], clip)
    v = np.clip(v + rng.laplace(0.0, laplace_scale(epsilon, clip), v.shape), 0.0, None)
    totals = v.sum(axis=1, keepdims=True)
    out[is_leaf] = np.divide(v, totals, out=np.full_like(v, 1 / v.shape[1]), where=totals > 0)
    return out


def epsilon_schedule(spec, rounds, seed):
    """Per-round epsilon: a constant, uniform draws from a (low, high) range, or None (no DP)."""
    if spec is None:
        return [None] * rounds
    if isinstance(spec, (tuple, list)):
        low, high = spec
        return np.random.default_rng([seed, 2]).uniform(low, high, rounds).tolist()
    return [float(spec)] * rounds


def describe(spec):
    if spec is None:
        return "No DP"
    if isinstance(spec, (tuple, list)):
        return f"U[{spec[0]:g}, {spec[1]:g}]"
    return f"{spec:g}"


class PrivacyAccountant:
    """Per-patient privacy loss of one client under basic composition.

    Only newly grown trees consume budget: re-sending a tree that was already released (in
    F_k^(t-1) or G^(t)) is post-processing of a DP output and costs nothing.
    """

    def __init__(self):
        self.total = 0.0

    def spend(self, epsilon, n_trees):
        """Record the release of n_trees new trees; returns their per-patient epsilon."""
        spent = 0.0 if epsilon is None else n_trees * epsilon
        self.total += spent
        return spent
