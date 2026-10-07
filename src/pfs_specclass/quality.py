"""Spectrum quality metrics, flags and the corresponding weight adjustments.

Known instrumental problems (see outputs/diagnostics/):
  * "n-arm bump": flux at the r/n junction rises at ~930 nm, peaks at 970-1000 nm and
    settles on an elevated plateau; the rest of the n arm is distorted too.  Strongly
    concentrated on the fibres at the high-fiberId end of each spectrograph and around
    fiberId ~1140 ("hot" fibres, found from the data by a binomial test).
  * "n-arm step": the same fibres (and the high-fiberId end of spectrograph 2) also show a
    jump at the r/n junction without the peak.  A rise from r to n is also produced by real
    red stars, so a step alone is only acted upon on hot fibres.
  * strongly negative flux in the blue (b arm) or the NIR (n arm): failed sky subtraction /
    flux calibration.
  * "r-only box": coadds that mix m-arm and r-arm visits (arms = bmnr) are covered by the r
    arm alone at 656-703 and 896-937 nm (where m has no data).  In a few % of them these two
    ranges are offset from the rest of the spectrum by >15% (relative calibration of the r
    and m visits).  These ranges are dropped for every bmnr spectrum, which makes it look
    like a b+m+n spectrum (the encoder is trained to be invariant to that coverage).

Hot fibres are found per fibre and, to catch extended regions with few spectra per fibre,
also over windows of +-POOL fibres (binomial tests of the rate of BUMP or STEP spectra).

Flags (bitmask):
  NARM_BUMP  peak = f(975-1005) / mean(f(915-935), f(1030-1060)) > PEAK_THRESH
  NARM_STEP  rise = f(1030-1060) / f(915-935) > RISE_THRESH
  HOT_FIBER  spectrum observed (in any visit) on a hot fibre
  NEG_BLUE   mean 380-480 nm flux < -0.05 (normalised) at > 5 sigma
  NEG_NIR    mean 1150-1260 nm flux < -0.05 at > 5 sigma
  MIXED_RM   coadd contains both m-arm and r-arm visits
  HOT_VISIT  observed in an exposure/camera with a coherent calibration error in some 20-nm
             bins (visit_quality.py); the bins are stored per spectrum in ``hot_bins``
  HOT_VISIT_HEAVY  HOT_VISIT with >= HEAVY_HOT_BINS hot bins

Weight policy (applied after normalisation):
  (NARM_BUMP | NARM_STEP) & HOT_FIBER  -> n arm (> 925 nm) weight 0
  NARM_BUMP only         -> 925-1015 nm weight 0 (keeps the rest of the n arm)
  NEG_BLUE / NEG_NIR     -> weights unchanged; excluded from training, reported in outputs
  MIXED_RM               -> R_ONLY_MASK ranges weight 0 (b+m+n gaps incl. the same edge trim)
  HOT_VISIT              -> the spectrum's hot 20-nm bins weight 0

Training uses only spectra without any of EXCLUDE_FROM_TRAINING.  Kept: NARM_STEP alone (mostly
real red stars), MIXED_RM and HOT_VISIT with few hot bins (handled by the weights).  Fibre-based
exclusion does not bias the object population (objects are assigned to fibres at random);
visit-based exclusion would (visits are tied to fields, runs and arm configurations), so only
heavily affected spectra are excluded for hot visits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom

NARM_BUMP = 1
HOT_FIBER = 2
NEG_BLUE = 4
NEG_NIR = 8
NARM_STEP = 16
MIXED_RM = 32
HOT_VISIT = 64
HOT_VISIT_HEAVY = 128
FLAG_NAMES = {NARM_BUMP: "NARM_BUMP", NARM_STEP: "NARM_STEP", HOT_FIBER: "HOT_FIBER", NEG_BLUE: "NEG_BLUE",
              NEG_NIR: "NEG_NIR", MIXED_RM: "MIXED_RM", HOT_VISIT: "HOT_VISIT", HOT_VISIT_HEAVY: "HOT_VISIT_HEAVY"}
EXCLUDE_FROM_TRAINING = NEG_BLUE | NEG_NIR | HOT_FIBER | NARM_BUMP | HOT_VISIT_HEAVY
HEAVY_HOT_BINS = 3
# flags that only change how a spectrum is handled; a spectrum with only these set counts as clean
HANDLING_ONLY = MIXED_RM

PEAK_THRESH = 1.2
RISE_THRESH = 1.5
FIBERS_PER_SPECTROGRAPH = 651
POOL = 3
BUMP_RANGE = (925.0, 1015.0)
NARM_START = 925.0
# b+m+n coverage gaps (655.8-702.7, 895.8-937.5 nm) widened by the 25-bin gap trim of dataset.py
R_ONLY_MASK = ((650.0, 708.5), (888.0, 945.5))

WINDOWS = {"f915": (915, 935), "f975": (975, 1005), "f1030": (1030, 1060), "fblue": (380, 480),
           "fnir": (1150, 1260)}


def _wmean(f, w, wave, lo, hi):
    s = (wave > lo) & (wave < hi)
    ww = w[:, s]
    sw = ww.sum(1)
    mean = (f[:, s] * ww).sum(1) / np.where(sw > 0, sw, np.nan)
    return mean, 1.0 / np.sqrt(np.where(sw > 0, sw, np.nan))


def metrics(wave, f, w) -> pd.DataFrame:
    """Per-spectrum metrics from normalised flux ``f`` and weights ``w``."""
    out = {}
    for k, (lo, hi) in WINDOWS.items():
        out[k], out["e" + k[1:]] = _wmean(f, w, wave, lo, hi)
    m = pd.DataFrame(out)
    m["peak"] = m["f975"] / (0.5 * (m["f915"] + m["f1030"]))
    m["rise"] = m["f1030"] / m["f915"]
    return m[["peak", "rise", "fblue", "eblue", "fnir", "enir"]]


def find_hot_fibers(fiber_ids: pd.Series, hit: np.ndarray, min_n=20, p_max=1e-4, ratio_min=3.0,
                    pool=POOL) -> pd.DataFrame:
    """Fibres whose rate of ``hit`` spectra is significantly above the global rate.

    ``fiber_ids`` holds comma-separated fibre lists per spectrum; each spectrum counts once
    for every fibre it was observed with.  A fibre is hot if it passes the test on its own,
    or if the window of +-``pool`` fibres (same spectrograph) passes it and the fibre's own
    rate is at least twice the global rate.
    """
    x = pd.DataFrame({"fiberId": fiber_ids.str.split(","), "hit": hit}).explode("fiberId")
    x = x[x["fiberId"].notna() & (x["fiberId"] != "")]
    x["fiberId"] = x["fiberId"].astype(int)
    g = x.groupby("fiberId")["hit"].agg(n="size", k="sum")
    p0 = g["k"].sum() / g["n"].sum()
    g["rate"] = g["k"] / g["n"]
    g["p_value"] = binom.sf(g["k"] - 1, g["n"], p0)
    single = (g["n"] >= min_n) & (g["p_value"] < p_max) & (g["rate"] > ratio_min * p0)

    full = g.reindex(np.arange(1, g.index.max() + 1), fill_value=0)
    sm = (full.index.values - 1) // FIBERS_PER_SPECTROGRAPH
    pn = np.zeros(len(full))
    pk = np.zeros(len(full))
    for off in range(-pool, pool + 1):
        same = np.roll(sm, -off) == sm
        if off > 0:
            same[-off:] = False
        elif off < 0:
            same[:-off] = False
        pn += np.where(same, np.roll(full["n"].values, -off), 0)
        pk += np.where(same, np.roll(full["k"].values, -off), 0)
    full["n_pool"], full["k_pool"] = pn, pk
    full["p_pool"] = binom.sf(full["k_pool"] - 1, full["n_pool"].clip(lower=1), p0)
    g = g.join(full[["n_pool", "k_pool", "p_pool"]])
    pooled = (g["p_pool"] < p_max) & (g["k_pool"] / g["n_pool"] > ratio_min * p0) & (g["rate"] > 2 * p0)
    g["hot_single"], g["hot_pooled"] = single, pooled
    g["hot"] = single | pooled
    g.attrs["global_rate"] = p0
    return g


def flags(m: pd.DataFrame, fiber_ids: pd.Series, hot: set[int], arms: pd.Series | None = None) -> np.ndarray:
    fl = np.zeros(len(m), np.int32)
    if arms is not None:
        a = pd.Series(arms).fillna("").values
        mixed = np.array(["m" in x and "r" in x for x in a])
        fl |= np.where(mixed, MIXED_RM, 0).astype(np.int32)
    fl |= np.where(m["peak"].values > PEAK_THRESH, NARM_BUMP, 0).astype(np.int32)
    fl |= np.where(m["rise"].values > RISE_THRESH, NARM_STEP, 0).astype(np.int32)
    on_hot = fiber_ids.fillna("").map(lambda s: any(int(f) in hot for f in s.split(",") if f)).values
    fl |= np.where(on_hot, HOT_FIBER, 0).astype(np.int32)
    fl |= np.where((m["fblue"] < -0.05) & (m["fblue"] / m["eblue"] < -5), NEG_BLUE, 0).astype(np.int32)
    fl |= np.where((m["fnir"] < -0.05) & (m["fnir"] / m["enir"] < -5), NEG_NIR, 0).astype(np.int32)
    return fl


def adjust_weights(wave, w, fl, hot_bins=None):
    """Zero weights according to the quality flags (in place, returns ``w``).

    ``hot_bins``: optional int64 bitmask per spectrum of hot 20-nm bins (visit_quality.py).
    """
    bump = (fl & NARM_BUMP) > 0
    step = (fl & NARM_STEP) > 0
    hot = (fl & HOT_FIBER) > 0
    narm = wave > NARM_START
    local = (wave > BUMP_RANGE[0]) & (wave < BUMP_RANGE[1])
    w[np.ix_((bump | step) & hot, narm)] = 0
    w[np.ix_(bump & ~hot, local)] = 0
    mixed = (fl & MIXED_RM) > 0
    if mixed.any():
        r_only = np.zeros(len(wave), bool)
        for lo, hi in R_ONLY_MASK:
            r_only |= (wave > lo) & (wave < hi)
        w[np.ix_(mixed, r_only)] = 0
    if hot_bins is not None and np.any(hot_bins):
        from .visit_quality import bins_to_pixels

        w[bins_to_pixels(wave, np.asarray(hot_bins, np.int64))] = 0
    return w


def visit_flags(hot_bins) -> np.ndarray:
    """HOT_VISIT / HOT_VISIT_HEAVY bits from per-spectrum hot-bin bitmasks."""
    hb = np.asarray(hot_bins, np.int64)
    nbits = np.array([bin(int(v)).count("1") for v in hb]) if len(hb) else np.zeros(0, int)
    return (np.where(hb != 0, HOT_VISIT, 0) | np.where(nbits >= HEAVY_HOT_BINS, HOT_VISIT_HEAVY, 0)).astype(np.int32)


def is_clean(fl) -> np.ndarray:
    return (np.asarray(fl) & ~HANDLING_ONLY) == 0


def flag_string(v: int) -> str:
    return "|".join(n for b, n in FLAG_NAMES.items() if v & b) or ""
