"""Diagnostic plots shared by the PCA baseline and the autoencoder."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# dataviz reference palette: categorical slots 1-3 and the blue sequential ramp
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
MUTED = "#9a9a9a"
SEQ = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#555555", labelsize=8)


def scatter_panels(xy, meta_sub, out, title, chi2_col="chi2nu"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    kw = dict(s=1.5, linewidths=0, rasterized=True)

    a = axes[0, 0]
    cls = meta_sub["classificationName"].fillna("NONE").values
    a.scatter(*xy[cls == "NONE"].T, c=MUTED, label="no class", **kw)
    for c, col in zip(["STAR", "GALAXY", "QSO"], CAT):
        s = cls == c
        a.scatter(*xy[s].T, c=col, label=f"{c} ({s.sum()})", **kw)
    a.set_title("DRP classification (reference only)", fontsize=10)
    a.legend(markerscale=6, fontsize=8, frameon=False)

    a = axes[0, 1]
    v = np.log10(np.clip(meta_sub["snrBin"].values, 1, None))
    o = np.argsort(v)
    sc = a.scatter(*xy[o].T, c=v[o], cmap=SEQ, **kw)
    fig.colorbar(sc, ax=a, label="log10 median S/N per bin")
    a.set_title("S/N", fontsize=10)

    a = axes[1, 0]
    has_m = meta_sub["arms"].str.contains("m").values
    a.scatter(*xy[~has_m].T, c=CAT[0], label=f"r arm only ({(~has_m).sum()})", **kw)
    a.scatter(*xy[has_m].T, c=CAT[1], label=f"includes m arm ({has_m.sum()})", **kw)
    a.set_title("Arm configuration (should NOT drive structure)", fontsize=10)
    a.legend(markerscale=6, fontsize=8, frameon=False)

    a = axes[1, 1]
    v = np.log10(np.clip(meta_sub[chi2_col].values, 0.1, None))
    o = np.argsort(v)
    sc = a.scatter(*xy[o].T, c=v[o], cmap=SEQ, vmax=np.percentile(v, 99), **kw)
    fig.colorbar(sc, ax=a, label="log10 reduced chi2 of reconstruction")
    a.set_title("Reconstruction chi2", fontsize=10)

    for a in axes.ravel():
        style(a)
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_outliers(wave, X, W, recon, meta, idx, out, chi2_col="chi2nu", extra=""):
    """Plot spectra ``idx`` (rows of X/W/meta) with their reconstructions ``recon`` (same order)."""
    n = len(idx)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.6 * n), sharex=True, squeeze=False)
    for a, i, r in zip(axes[:, 0], idx, recon):
        ok = W[i] > 0
        a.plot(wave[ok], X[i][ok], lw=0.4, color=MUTED)
        a.plot(wave, r, lw=0.6, color=CAT[1])
        lo, hi = np.percentile(X[i][ok], [1, 99]) if ok.any() else (0, 1)
        a.set_ylim(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))
        m = meta.loc[i]
        e = f"  {extra}={m[extra]:.2f}" if extra else ""
        a.set_title(f"{m.catId} {m.objId} {m.combination}  {m.classificationName} z={m.bestRedshift:.3f}  "
                    f"S/N={m.snrBin:.1f}  chi2nu={m[chi2_col]:.1f}{e}", fontsize=8, loc="left")
        style(a)
    axes[-1, 0].set_xlabel("observed wavelength [nm]")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
