"""Hierarchical few-shot transfer (HiFT) model.

Residual of membrane m in solvent j relative to a prior mean f (e.g. a zero-shot GBM):
    r_mj = z_j' (beta + u_m) + e_mj,   u_m ~ N(0, Sigma),   e_mj ~ N(0, s2)
z_j = [1, standardized solvent descriptors]. beta, Sigma, s2 are learned by EM across training membranes;
a new membrane's u_m is updated in closed form from k anchor measurements (empirical Bayes).
"""
import numpy as np


class HierarchicalTransfer:
    def __init__(self, intercept_only=False, n_iter=500, tol=1e-7, prior_strength=None, prior_scale=0.01):
        self.intercept_only = intercept_only
        self.n_iter = n_iter
        self.tol = tol
        self.prior_strength = prior_strength
        self.prior_scale = prior_scale

    def _z(self, Z):
        Z = np.atleast_2d(np.asarray(Z, float))
        return Z[:, :1] if self.intercept_only else Z

    def fit(self, groups):
        """groups: iterable of (Z_m, r_m) with Z_m (n_m, p) and r_m (n_m,)."""
        Zs = [self._z(Z) for Z, _ in groups]
        rs = [np.asarray(r, float) for _, r in groups]
        p = Zs[0].shape[1]
        G = np.stack([Z.T @ Z for Z in Zs])
        h = np.stack([Z.T @ r for Z, r in zip(Zs, rs)])
        rr = np.array([r @ r for r in rs])
        N = sum(len(r) for r in rs)
        M = len(rs)
        nu0 = self.prior_strength if self.prior_strength is not None else p + 2
        psi = np.eye(p) * self.prior_scale * nu0
        beta = np.zeros(p)
        Sigma = np.diag([0.2] + [0.02] * (p - 1))
        s2 = 0.05
        G_inv_sum = np.linalg.pinv(G.sum(0))
        prev = np.inf
        for it in range(self.n_iter):
            Sinv = np.linalg.inv(Sigma)
            V = np.linalg.inv(G / s2 + Sinv)
            resid_h = h - G @ beta
            u = np.einsum("mij,mj->mi", V, resid_h) / s2
            Sigma = (psi + np.einsum("mi,mj->ij", u, u) + V.sum(0)) / (nu0 + M)
            bu = beta + u
            sse = rr - 2 * np.einsum("mi,mi->m", bu, h) + np.einsum("mi,mij,mj->m", bu, G, bu)
            s2 = (sse.sum() + np.einsum("mij,mji->", V, G)) / N
            s2 = max(s2, 1e-6)
            beta = G_inv_sum @ (h - np.einsum("mij,mj->mi", G, u)).sum(0)
            obj = s2 + np.trace(Sigma)
            if abs(prev - obj) < self.tol:
                break
            prev = obj
        self.beta_, self.Sigma_, self.s2_, self.n_iter_ = beta, Sigma, s2, it + 1
        return self

    def posterior(self, Z_a=None, r_a=None):
        if Z_a is None or len(r_a) == 0:
            return np.zeros_like(self.beta_), self.Sigma_.copy()
        Z = self._z(Z_a)
        resid = np.asarray(r_a, float) - Z @ self.beta_
        V = np.linalg.inv(Z.T @ Z / self.s2_ + np.linalg.inv(self.Sigma_))
        return V @ Z.T @ resid / self.s2_, V

    def predict(self, Z_t, Z_a=None, r_a=None):
        u, V = self.posterior(Z_a, r_a)
        Z = self._z(Z_t)
        mean = Z @ (self.beta_ + u)
        var = np.einsum("ij,jk,ik->i", Z, V, Z) + self.s2_
        return mean, np.sqrt(var)

    def expected_variance(self, Z_anchor, Z_targets):
        """Posterior predictive variance at targets after observing anchors (independent of the values)."""
        _, V = self.posterior(Z_anchor, np.zeros(len(np.atleast_2d(Z_anchor))))
        Z = self._z(Z_targets)
        return np.einsum("ij,jk,ik->i", Z, V, Z) + self.s2_
