"""Score new spectra with a trained autoencoder: latent, cluster, kNN label, anomaly.

Anomaly scores are percentiles relative to the reference set (the spectra
encoded at training time) *within the same S/N bin*, so that noisy spectra are
not flagged merely for being noisy.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from . import preprocess as pp
from . import quality as q
from . import visit_quality as vq
from .dataset import WeightConfig, normalise
from .model import SpectrumAutoencoder, chi2_per_spectrum

N_SNR_BINS = 10


def snr_bin_edges(snr, nbins=N_SNR_BINS):
    return np.quantile(snr[np.isfinite(snr)], np.linspace(0, 1, nbins + 1))


def snr_bin(snr, edges):
    return np.clip(np.searchsorted(edges, snr, side="right") - 1, 0, len(edges) - 2)


def percentile_in_bins(score, snr, nbins=N_SNR_BINS):
    """Rank-transform ``score`` within S/N quantile bins -> (0, 1) (1 = most extreme)."""
    out = np.full(len(score), np.nan)
    ok = np.isfinite(score) & np.isfinite(snr)
    b = snr_bin(snr, snr_bin_edges(snr[ok], nbins))
    for k in range(nbins):
        s = ok & (b == k)
        if s.any():
            out[s] = (score[s].argsort().argsort() + 0.5) / s.sum()
    return out


class SpectrumScorer:
    def __init__(self, model_dir: str, device: str = "cuda", k_neighbors: int = 15):
        self.device = device
        ckpt = torch.load(os.path.join(model_dir, "ae.pt"), map_location=device, weights_only=False)
        self.args = ckpt["args"]
        self.cfg = WeightConfig(sky_frac=self.args["sky_frac"], flux_frac=self.args["flux_frac"])
        self.wave = ckpt["wave"]
        self.net = SpectrumAutoencoder(self.wave, n_latent=self.args["latent"]).to(device)
        self.net.load_state_dict(ckpt["state"])
        self.net.eval()
        self.gmm = joblib.load(os.path.join(model_dir, "gmm.joblib"))
        hot_path = os.path.join(model_dir, "hot_fibers.csv")
        self.hot = set()
        if os.path.exists(hot_path):
            hf = pd.read_csv(hot_path)
            self.hot = set(hf.loc[hf["hot"], "fiberId"].astype(int))
        # hot exposure/camera units; spectra from visits not in the table get no HOT_VISIT check
        units_path = os.path.join(model_dir, "visit_units.parquet")
        self.unit_tables = None
        if os.path.exists(units_path):
            u = pd.read_parquet(units_path, columns=["unit", "kind", "hot_bins"])
            self.unit_tables = [u[u["kind"] == "visitset"], u[u["kind"] == "visit"]]

        ref = pd.read_parquet(os.path.join(model_dir, "ae_results.parquet"))
        ref = ref[ref["usable"]]
        self.snr_edges = snr_bin_edges(ref["snrBin"].values)
        b = snr_bin(ref["snrBin"].values, self.snr_edges)
        # sorted reference scores per S/N bin for percentile lookup
        self.ref_chi2 = [np.sort(ref["chi2nu"].values[b == k]) for k in range(N_SNR_BINS)]
        self.ref_nlogp = [np.sort(-ref["latent_logp"].values[b == k]) for k in range(N_SNR_BINS)]
        # kNN reference: training objects with a DRP class (labels are only a hint)
        tr = ref[ref["train"]]
        lat = tr[[f"s{i + 1}" for i in range(self.args["latent"])]].values
        self.knn = NearestNeighbors(n_neighbors=k_neighbors).fit(lat)
        self.knn_labels = np.asarray(tr["classificationName"].fillna("NONE"), dtype=object)

    @torch.no_grad()
    def score_arrays(self, flux, ivar, sky, norm, snr, fiber_ids=None, arms=None, visits=None):
        f, w = normalise(flux, ivar, sky, norm, self.cfg)
        fiber_ids = pd.Series([""] * len(f) if fiber_ids is None else list(fiber_ids))
        qm = q.metrics(self.wave, f, w)
        qflag = q.flags(qm, fiber_ids, self.hot, None if arms is None else pd.Series(list(arms)))
        hot_bins = np.zeros(len(f), np.int64)
        if self.unit_tables is not None and visits is not None:
            hot_bins = vq.lookup_hot_bins(pd.Series(list(visits)), fiber_ids, self.unit_tables)
            qflag = qflag | q.visit_flags(hot_bins).astype(qflag.dtype)
        q.adjust_weights(self.wave, w, qflag, hot_bins)
        x = torch.as_tensor(f, device=self.device)
        wt = torch.as_tensor(w, device=self.device)
        s, model = self.net(x, wt)
        chi2nu = (chi2_per_spectrum(x, wt, model) / (wt > 0).sum(1).clamp_min(1)).cpu().numpy()
        s = s.cpu().numpy()
        logp = self.gmm.score_samples(s)
        b = snr_bin(snr, self.snr_edges)
        pct = lambda ref, v: np.array([np.searchsorted(ref[k], vi) / max(len(ref[k]), 1) for k, vi in zip(b, v)])
        _, nn = self.knn.kneighbors(s)
        votes = self.knn_labels[nn]
        knn_label = [pd.Series(v).value_counts().index[0] for v in votes]
        knn_frac = [pd.Series(v).value_counts().iloc[0] / len(v) for v in votes]
        out = pd.DataFrame({
            "chi2nu": chi2nu, "latent_logp": logp, "gmm_label": self.gmm.predict(s),
            "anom_chi2": pct(self.ref_chi2, chi2nu), "anom_density": pct(self.ref_nlogp, -logp),
            "knn_label": knn_label, "knn_frac": knn_frac,
        })
        out["anom_score"] = np.fmax(out["anom_chi2"], out["anom_density"])
        out["qflag"] = qflag
        out["qflag_names"] = [q.flag_string(v) for v in qflag]
        out["hot_bins"] = hot_bins
        out = pd.concat([out, qm.reset_index(drop=True)], axis=1)
        for i in range(s.shape[1]):
            out[f"s{i + 1}"] = s[:, i]
        return out, model.cpu().numpy()

    def score_files(self, paths):
        specs = [pp.read_pfsobject(p) for p in paths]
        meta = pd.DataFrame([sp.meta for sp in specs])
        flux = np.stack([sp.flux for sp in specs])
        ivar = np.stack([sp.ivar for sp in specs])
        sky = np.stack([sp.sky for sp in specs])
        out, _ = self.score_arrays(flux, ivar, sky, meta["norm"].values.astype(np.float32), meta["snrBin"].values,
                                   meta["fiberIds"], meta["arms"], meta["visits"])
        return pd.concat([meta, out], axis=1)

