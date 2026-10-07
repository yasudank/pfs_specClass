"""Exposure-level ("hot visit") quality: coherent leave-one-arm-out residuals.

For every spectrum and every arm region, the arm is hidden from the encoder and predicted
from the other arms; the fractional residual (data - prediction) / prediction is averaged in
20-nm coarse bins.  The model's prediction is biased in a way that depends on the kind of
object (and on reddening) and differs between arm configurations (b+r+n vs b+m+n), so from
every residual profile we subtract the median profile of its K nearest neighbours observed in
*other* visits with the *same* arm configuration ("deviation").  Neighbours are found with the
latent vector encoded *with the arm hidden*: a latent from the full spectrum encodes the
artefact itself, and neighbours sharing the artefact would cancel it.

A calibration error of one camera in one exposure is shared by all spectra observed with it,
whereas object-to-object differences average out.  Deviations are therefore aggregated per
*unit*, using two definitions:
  * visit set x spectrograph: spectra with exactly the same visits on the same spectrograph
    (share the same exposures and cameras; most sensitive)
  * single visit x spectrograph: every spectrum that contains the visit (covers ~99% of the
    spectra, but a bad visit is diluted in coadds of many visits)
A unit is hot in a coarse bin if its median deviation is significant (> SIG x its error),
larger than AMP, and an outlier among units of the same arm configuration (> Z x the robust
unit-to-unit spread in that bin, around that configuration's typical deviation).  A spectrum gets the union of the hot bins of all units it belongs to.

The model used to predict the hidden arm cannot reproduce an artefact of that arm (the arm is
hidden), which is why this works even with a model trained on data containing the artefacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from .model import fit_amplitude

ARM_REGIONS = {"b": (380.0, 645.0), "red": (650.0, 940.0), "n": (945.0, 1260.0)}
BIN_WIDTH = 20.0  # nm
FIBERS_PER_SPECTROGRAPH = 651

HOT_AMP = 0.05
HOT_SIG = 4.0
HOT_Z = 4.0
MIN_SPECTRA = 20
K_BASELINE = 40


def coarse_bins():
    """List of (arm, lo, hi) coarse bins (45 bins; fits in an int64 bitmask)."""
    out = []
    for arm, (lo, hi) in ARM_REGIONS.items():
        edges = np.arange(lo, hi + 1e-6, BIN_WIDTH)
        if edges[-1] < hi:
            edges = np.append(edges, hi)
        out += [(arm, a, b) for a, b in zip(edges[:-1], edges[1:])]
    return out


@torch.no_grad()
def encode(net, X, W, device="cuda", batch=2048):
    out = []
    for s in range(0, len(X), batch):
        x = torch.as_tensor(X[s:s + batch], device=device)
        w = torch.as_tensor(W[s:s + batch], device=device)
        out.append(net.encoder(x, w).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def leave_arm_out_residuals(net, wave, X, W, device="cuda", batch=2048):
    """Weighted mean fractional residual per coarse bin with the bin's arm hidden.

    Returns (R, S): R (N, n_bins), NaN where a spectrum has no valid pixels in a bin, and the
    latent vectors with each arm hidden, S (N, n_arms, n_latent).
    """
    bins = coarse_bins()
    arms = list(ARM_REGIONS)
    arm_pix = {a: torch.as_tensor((wave > lo) & (wave < hi), device=device) for a, (lo, hi) in ARM_REGIONS.items()}
    bin_pix = torch.stack([torch.as_tensor((wave > lo) & (wave < hi), device=device) for _, lo, hi in bins]).float()
    bin_arm = np.array([arms.index(a) for a, _, _ in bins])
    R = np.full((len(X), len(bins)), np.nan, np.float32)
    S = None
    for s in range(0, len(X), batch):
        x = torch.as_tensor(X[s:s + batch], device=device)
        w = torch.as_tensor(W[s:s + batch], device=device)
        z0 = torch.zeros(len(x), device=device)
        for ai, a in enumerate(arms):
            w_in = w * (~arm_pix[a]).float()
            lat = net.encoder(x, w_in)
            if S is None:
                S = np.zeros((len(X), len(arms), lat.shape[1]), np.float32)
            S[s:s + batch, ai] = lat.cpu().numpy()
            model = net.transform(net.decoder(lat), z0)
            model = fit_amplitude(x, w_in, model)[:, None] * model
            ok = model.abs() > 0.05
            r = torch.where(ok, (x - model) / torch.where(ok, model, torch.ones_like(model)), torch.zeros_like(x))
            wr = torch.where(ok, w * model**2, torch.zeros_like(w))  # weight of the fractional residual
            k = np.flatnonzero(bin_arm == ai)
            sw = wr @ bin_pix[k].T
            mean = (r * wr) @ bin_pix[k].T / sw.clamp_min(1e-30)
            R[s:s + batch, k] = torch.where(sw > 0, mean, torch.full_like(mean, float("nan"))).cpu().numpy()
    return R, S


def knn_deviation(R, S, visits, sel, groups, k=K_BASELINE, n_query=2 * K_BASELINE):
    """R minus the median R of the k neighbours from other visit sets.

    ``S`` (N, n_arms, n_latent) holds the latents with each arm hidden; the bins of an arm use
    the latent with that arm hidden.  Neighbours are searched among rows ``sel`` with the same
    ``groups`` value (arm configuration).
    """
    D = np.full_like(R, np.nan)
    groups = np.asarray(groups)
    visits = np.asarray(visits)
    bin_arm = np.array([list(ARM_REGIONS).index(a) for a, _, _ in coarse_bins()])
    for g in np.unique(groups[sel]):
        idx = np.flatnonzero(sel & (groups == g))
        if len(idx) <= n_query:
            continue
        vs = visits[idx]
        for ai in range(S.shape[1]):
            cols = np.flatnonzero(bin_arm == ai)
            nn = NearestNeighbors(n_neighbors=n_query).fit(S[idx, ai]).kneighbors(S[idx, ai], return_distance=False)
            Ra = R[np.ix_(idx, cols)]
            with np.errstate(all="ignore"):
                for j in range(len(idx)):
                    cand = nn[j][vs[nn[j]] != vs[j]][:k]
                    D[idx[j], cols] = Ra[j] - np.nanmedian(Ra[cand], 0)
    return D


def arm_group(arms: pd.Series) -> np.ndarray:
    """Arm configuration used for baselines: 'm' if the coadd contains m-arm data, else 'r'."""
    return np.where(arms.fillna("").str.contains("m"), "m", "r")


def spectrograph_of(fiber_ids: pd.Series) -> pd.Series:
    fib = pd.to_numeric(fiber_ids.fillna("").str.split(",").str[0], errors="coerce")
    return ((fib - 1) // FIBERS_PER_SPECTROGRAPH + 1).astype("Int64").astype(str)


def membership(meta: pd.DataFrame, rows: np.ndarray, kind: str) -> pd.DataFrame:
    """(row, unit, group) for ``kind`` in {"visitset", "visit"}; group = arm configuration."""
    sm = spectrograph_of(meta["fiberIds"]).values[rows]
    vis = meta["visits"].fillna("").values[rows]
    grp = arm_group(meta["arms"])[rows] if "arms" in meta else np.full(len(rows), "r")
    x = pd.DataFrame({"row": rows, "visit": vis, "sm": sm, "group": grp})
    if kind == "visit":
        x["visit"] = x["visit"].str.split(",")
        x = x.explode("visit")
    x["unit"] = x["visit"] + "|SM" + x["sm"]
    return x[["row", "unit", "group"]]


def hot_units(D, members: pd.DataFrame) -> pd.DataFrame:
    """Median deviation per unit and its hot bins (bitmask column ``hot_bins``).

    The typical deviation and the unit-to-unit spread are computed per arm configuration
    (the unit's majority ``group``).
    """
    gb = members.groupby("unit")
    groups = pd.DataFrame({"rows": gb["row"].apply(np.array), "group": gb["group"].agg(lambda g: g.mode().iloc[0])})
    units, n, med, sig = [], [], [], []
    for unit, r in groups["rows"].items():
        v = D[r]
        cnt = np.sum(np.isfinite(v), 0)
        with np.errstate(all="ignore"):
            m = np.nanmedian(v, 0) if np.isfinite(v).any() else np.full(v.shape[1], np.nan)
            err = 1.253 * 1.4826 * np.nanmedian(np.abs(v - m), 0) / np.sqrt(np.maximum(cnt, 1))
        units.append(unit)
        n.append(len(r))
        med.append(m)
        sig.append((cnt >= MIN_SPECTRA) & (np.abs(m) > HOT_SIG * err))
    med, sig, n = np.array(med), np.array(sig), np.array(n)
    grp = groups["group"].values
    ok = n >= MIN_SPECTRA
    dev = np.full_like(med, np.nan)
    hot = np.zeros(med.shape, bool)
    for g in np.unique(grp):
        sel = grp == g
        ref = sel & ok if (sel & ok).sum() >= 30 else ok  # fall back to all units if too few
        with np.errstate(all="ignore"):
            centre = np.nanmedian(med[ref], 0)
            spread = 1.4826 * np.nanmedian(np.abs(med[ref] - centre), 0)
        dev[sel] = med[sel] - centre
        hot[sel] = sig[sel] & ok[sel, None] & (np.abs(dev[sel]) > np.maximum(HOT_AMP, HOT_Z * spread))
    tab = pd.DataFrame({"unit": units, "group": grp, "n": n, "n_hot_bins": hot.sum(1), "hot_bins": bits(hot)})
    tab[[f"dev_{k}" for k in range(med.shape[1])]] = dev
    return tab


def bits(mask: np.ndarray) -> np.ndarray:
    """Boolean (n, n_bins) -> int64 bitmask."""
    return (mask.astype(np.int64) << np.arange(mask.shape[1], dtype=np.int64)).sum(1)


def unbits(b, n_bins=None) -> np.ndarray:
    n_bins = len(coarse_bins()) if n_bins is None else n_bins
    b = np.asarray(b, np.int64)
    return ((b[:, None] >> np.arange(n_bins, dtype=np.int64)) & 1).astype(bool)


def spectrum_hot_bins(n_rows: int, members: list[pd.DataFrame], tables: list[pd.DataFrame]) -> np.ndarray:
    """Union (bitwise OR) of the hot bins of all units of every spectrum."""
    out = np.zeros(n_rows, np.int64)
    for mem, tab in zip(members, tables):
        hb = mem["unit"].map(tab.set_index("unit")["hot_bins"]).fillna(0).astype(np.int64).values
        np.bitwise_or.at(out, mem["row"].values, hb)
    return out


def lookup_hot_bins(visits: pd.Series, fiber_ids: pd.Series, tables: list[pd.DataFrame]) -> np.ndarray:
    """Hot-bin bitmask for (new) spectra from saved unit tables (visit set table, visit table)."""
    meta = pd.DataFrame({"visits": pd.Series(visits).values, "fiberIds": pd.Series(fiber_ids).values})
    rows = np.arange(len(meta))
    mems = [membership(meta, rows, "visitset"), membership(meta, rows, "visit")]
    return spectrum_hot_bins(len(meta), mems, tables)


def bins_to_pixels(wave, hot_bins: np.ndarray) -> np.ndarray:
    """Pixel mask (n, n_pix) from an int64 hot-bin bitmask (n,)."""
    m = unbits(hot_bins)
    pix = np.zeros((len(hot_bins), len(wave)), bool)
    for k, (_, lo, hi) in enumerate(coarse_bins()):
        if m[:, k].any():
            pix[np.ix_(m[:, k], (wave > lo) & (wave < hi))] = True
    return pix
