"""Load the spectrum cube, normalise spectra and build effective weights."""

from __future__ import annotations

import os
import zlib
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd


@dataclass
class WeightConfig:
    # Systematic sky-subtraction residual: add (sky_frac * sky)^2 to the variance.
    sky_frac: float = 0.04
    # Fractional model-error floor: add (flux_frac * |normalised flux|)^2 to the variance,
    # so that very high-S/N spectra do not dominate the fit.
    flux_frac: float = 0.02
    # Drop this many bins at both ends of every contiguous good segment (arm edges).
    edge_trim: int = 3
    # Around internal gaps of >= gap_min empty bins (e.g. the b+m+n coverage gaps), drop
    # gap_trim bins on each side (~5-8 nm): the arm ends there carry large unmasked spikes.
    gap_min: int = 10
    gap_trim: int = 25


def split_of(cat_id, obj_id, nfold: int = 10) -> np.ndarray:
    """Deterministic fold per object (catId, objId); all combinations share a fold."""
    keys = (f"{c}_{o}".encode() for c, o in zip(cat_id, obj_id))
    return np.fromiter((zlib.crc32(k) % nfold for k in keys), dtype=np.int16, count=len(cat_id))


def load_meta(path_h5: str, quality: bool = True) -> pd.DataFrame:
    """Per-spectrum metadata; joins quality metrics/flags (``qflag``) when available."""
    meta = pd.read_parquet(path_h5.replace(".h5", "_meta.parquet"))
    meta["fold"] = split_of(meta["catId"].values, meta["objId"].values)
    qpath = path_h5.replace(".h5", "_quality.parquet")
    if quality and os.path.exists(qpath):
        qm = pd.read_parquet(qpath)
        meta = meta.join(qm.drop(columns="row"))
    return meta


def trim_edges(ivar: np.ndarray, n: int) -> np.ndarray:
    """Zero ``n`` bins at the edges of each contiguous run of ivar > 0 (row-wise)."""
    if n <= 0:
        return ivar
    good = ivar > 0
    keep = good.copy()
    for s in range(1, n + 1):
        keep[:, s:] &= good[:, :-s]
        keep[:, :-s] &= good[:, s:]
    keep[:, :n] = False
    keep[:, -n:] = False
    return np.where(keep, ivar, 0.0).astype(ivar.dtype)


def trim_gaps(ivar: np.ndarray, gap_min: int, trim: int) -> np.ndarray:
    """Zero ``trim`` bins next to every run of >= ``gap_min`` empty bins (row-wise)."""
    if trim <= 0:
        return ivar
    from scipy.ndimage import maximum_filter1d

    empty = (ivar <= 0).astype(np.int32)
    c = np.concatenate([np.zeros((len(ivar), 1), np.int32), np.cumsum(empty, 1)], 1)
    full = (c[:, gap_min:] - c[:, :-gap_min]) == gap_min  # window [j, j+gap_min) entirely empty
    long_run = np.zeros(ivar.shape, bool)
    long_run[:, : full.shape[1]] = full
    long_run = maximum_filter1d(long_run.view(np.uint8), gap_min, axis=1, origin=(gap_min - 1) // 2).astype(bool)
    near = maximum_filter1d(long_run.view(np.uint8), 2 * trim + 1, axis=1).astype(bool)
    return np.where(near, 0.0, ivar).astype(ivar.dtype)


def load_arrays(path_h5: str, rows=None, cfg: WeightConfig = WeightConfig(), quality: bool = True,
                visit_mask: bool = True):
    """Return (wave, flux_norm, weight, norm) for the requested rows.

    flux_norm = flux / norm, with norm the median flux over good bins (from the
    metadata); weight is the effective inverse variance of flux_norm.  With
    ``quality`` the weights are adjusted according to the quality flags (and, with
    ``visit_mask``, the per-spectrum hot 20-nm bins of hot visits).
    """
    with h5py.File(path_h5, "r") as h:
        wave = h["wave"][:]
        if rows is None:
            flux, ivar, sky = h["flux"][:], h["ivar"][:], h["sky"][:]
        else:
            rows = np.asarray(rows)
            order = np.argsort(rows)
            inv = np.argsort(order)
            sr = rows[order]
            flux, ivar, sky = (h[k][sr][inv] for k in ("flux", "ivar", "sky"))
    meta = load_meta(path_h5)
    sub = meta if rows is None else meta.iloc[rows]
    norm = sub["norm"].values.astype(np.float32)
    f, w = normalise(flux, ivar, sky, norm, cfg)
    if quality and "qflag" in sub:
        from .quality import adjust_weights

        hb = sub["hot_bins"].values if visit_mask and "hot_bins" in sub else None
        adjust_weights(wave, w, sub["qflag"].values, hb)
    return wave, f, w, norm


def normalise(flux, ivar, sky, norm, cfg: WeightConfig):
    ivar = trim_edges(ivar, cfg.edge_trim)
    ivar = trim_gaps(ivar, cfg.gap_min, cfg.gap_trim)
    good = (ivar > 0) & np.isfinite(norm)[:, None] & (norm > 0)[:, None]
    n = np.where(norm > 0, norm, 1.0)[:, None]
    f = np.where(good, flux / n, 0.0).astype(np.float32)
    var = np.where(good, 1.0 / np.where(good, ivar, 1.0), np.inf) / n**2
    var = var + (cfg.sky_frac * sky / n) ** 2 + (cfg.flux_frac * np.abs(f)) ** 2
    w = np.where(good, 1.0 / var, 0.0).astype(np.float32)
    return f, w


def select_train(meta, snr_min, good_min, folds, exclude_flags: int = 0):
    sel = (meta["fold"].isin(folds) & (meta["snrBin"] > snr_min) & (meta["goodFrac"] > good_min)
           & (meta["norm"] > 0)).to_numpy(copy=True)
    if exclude_flags and "qflag" in meta:
        sel &= (meta["qflag"].values & exclude_flags) == 0
    return sel


def best_per_object(meta, mask):
    """Index of the highest-S/N combination per (catId, objId) among ``mask``."""
    m = meta[mask]
    return m.sort_values("snrBin", ascending=False).drop_duplicates(["catId", "objId"]).index.values
