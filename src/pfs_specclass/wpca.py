"""Weighted PCA with missing data (alternating weighted least squares, GPU).

Model: x_i ≈ mu + c_i @ P, minimising Σ_id w_id (x_id - mu_d - c_i·P_d)^2.
Equivalent in spirit to EMPCA (Bailey 2012) but solves all components jointly.
"""

from __future__ import annotations

import numpy as np
import torch


def _solve_batched(A, b, ridge):
    k = A.shape[-1]
    eye = torch.eye(k, device=A.device, dtype=A.dtype)
    return torch.linalg.solve(A + ridge * eye, b.unsqueeze(-1)).squeeze(-1)


class WeightedPCA:
    def __init__(self, n_components: int = 10, n_iter: int = 30, ridge: float = 1e-6, device: str = "cuda",
                 batch: int = 8192, seed: int = 0):
        self.k = n_components
        self.n_iter = n_iter
        self.ridge = ridge
        self.device = device
        self.batch = batch
        self.seed = seed

    # -- helpers ---------------------------------------------------------------
    def _t(self, a):
        return torch.as_tensor(a, device=self.device, dtype=torch.float32)

    def _coeffs(self, X, W, P, mu):
        """Per-spectrum weighted least squares for the coefficients (batched)."""
        out = []
        for s in range(0, X.shape[0], self.batch):
            x = self._t(X[s:s + self.batch]) - mu
            w = self._t(W[s:s + self.batch])
            A = torch.einsum("kd,nd,ld->nkl", P, w, P)
            b = torch.einsum("kd,nd->nk", P, w * x)
            out.append(_solve_batched(A.double(), b.double(), self.ridge).float())
        return torch.cat(out)

    def _components(self, X, W, C, mu):
        """Per-pixel weighted least squares for the eigenvectors."""
        D = X.shape[1]
        A = torch.zeros(D, self.k, self.k, device=self.device, dtype=torch.float64)
        b = torch.zeros(D, self.k, device=self.device, dtype=torch.float64)
        for s in range(0, X.shape[0], self.batch):
            x = self._t(X[s:s + self.batch]) - mu
            w = self._t(W[s:s + self.batch])
            c = C[s:s + self.batch]
            A += torch.einsum("nd,nk,nl->dkl", w, c, c).double()
            b += torch.einsum("nd,nk->dk", w * x, c).double()
        return _solve_batched(A, b, self.ridge).float().T  # (k, D)

    # -- API -------------------------------------------------------------------
    def fit(self, X, W, verbose: bool = True):
        X = np.asarray(X, np.float32)
        W = np.asarray(W, np.float32)
        wsum = W.sum(0)
        mu = (W * X).sum(0) / np.maximum(wsum, 1e-30)
        self.mean_ = self._t(mu)
        self.pixel_weight_ = wsum

        g = torch.Generator(device="cpu").manual_seed(self.seed)
        P = torch.linalg.qr(torch.randn(X.shape[1], self.k, generator=g))[0].T.to(self.device)
        for it in range(self.n_iter):
            C = self._coeffs(X, W, P, self.mean_)
            P = self._components(X, W, C, self.mean_)
            P = torch.linalg.qr(P.T)[0].T  # orthonormalise; span is what matters
            if verbose and (it % 5 == 4 or it == self.n_iter - 1):
                C = self._coeffs(X, W, P, self.mean_)
                chi2 = self.chi2(X, W, C, P).sum() / max(float((W > 0).sum()), 1)
                print(f"  iter {it + 1}: chi2/pix = {chi2:.4f}", flush=True)

        # rotate so components are ordered by (weighted-fit) coefficient variance
        C = self._coeffs(X, W, P, self.mean_)
        cov = (C.T @ C).double() / C.shape[0]
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)
        R = evecs[:, order].float()
        self.components_ = R.T @ P
        self.explained_variance_ = evals[order].float().cpu().numpy()
        return self

    def transform(self, X, W):
        return self._coeffs(np.asarray(X, np.float32), np.asarray(W, np.float32), self.components_, self.mean_).cpu().numpy()

    def reconstruct(self, C):
        return (self._t(C) @ self.components_ + self.mean_).cpu().numpy()

    def chi2(self, X, W, C=None, P=None):
        """Per-spectrum chi^2 of the reconstruction."""
        P = self.components_ if P is None else P
        if C is None:
            C = self._coeffs(X, W, P, self.mean_)
        C = self._t(C)
        out = []
        for s in range(0, X.shape[0], self.batch):
            x = self._t(X[s:s + self.batch])
            w = self._t(W[s:s + self.batch])
            r = x - (C[s:s + self.batch] @ P + self.mean_)
            out.append((w * r * r).sum(1))
        return torch.cat(out).cpu().numpy()

    def save(self, path):
        np.savez(path, components=self.components_.cpu().numpy(), mean=self.mean_.cpu().numpy(),
                 explained_variance=self.explained_variance_)

    @classmethod
    def load(cls, path, device="cuda"):
        d = np.load(path)
        obj = cls(n_components=d["components"].shape[0], device=device)
        obj.components_ = obj._t(d["components"])
        obj.mean_ = obj._t(d["mean"])
        obj.explained_variance_ = d["explained_variance"]
        return obj
