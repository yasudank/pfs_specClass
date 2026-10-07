"""Weighted-PCA baseline: fit, project all spectra, UMAP, diagnostic plots.

Usage:
  uv run python scripts/run_wpca.py --data data/pfs_spectra.h5 --out outputs/wpca
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from pfs_specclass.dataset import WeightConfig, best_per_object, load_arrays, load_meta, select_train
from pfs_specclass.plots import CAT, plot_outliers, scatter_panels, style
from pfs_specclass.wpca import WeightedPCA


def plot_components(wave, pca, out):
    k = min(pca.components_.shape[0], 10)
    fig, axes = plt.subplots(k + 1, 1, figsize=(12, 1.4 * (k + 1)), sharex=True)
    axes[0].plot(wave, pca.mean_.cpu().numpy(), lw=0.6, color=CAT[0])
    axes[0].set_ylabel("mean", fontsize=8)
    ev = pca.explained_variance_ / pca.explained_variance_.sum()
    for i in range(k):
        axes[i + 1].plot(wave, pca.components_[i].cpu().numpy(), lw=0.6, color=CAT[0])
        axes[i + 1].set_ylabel(f"PC{i + 1}\n{ev[i] * 100:.1f}%", fontsize=8)
    for a in axes:
        style(a)
    axes[-1].set_xlabel("observed wavelength [nm]")
    fig.suptitle("Weighted PCA: mean spectrum and components (normalised flux)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pfs_spectra.h5")
    ap.add_argument("--out", default="outputs/wpca")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--snr-min", type=float, default=5.0)
    ap.add_argument("--good-min", type=float, default=0.8)
    ap.add_argument("--sky-frac", type=float, default=0.04)
    ap.add_argument("--flux-frac", type=float, default=0.02)
    ap.add_argument("--umap-max", type=int, default=60000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    meta = load_meta(args.data)
    cfg = WeightConfig(sky_frac=args.sky_frac, flux_frac=args.flux_frac)
    wave, X, W, _ = load_arrays(args.data, cfg=cfg)
    print(f"loaded {X.shape} in {time.time() - t0:.0f}s", flush=True)

    train = select_train(meta, args.snr_min, args.good_min, folds=range(8))
    print(f"training spectra: {train.sum()}", flush=True)
    pca = WeightedPCA(n_components=args.k, n_iter=args.n_iter).fit(X[train], W[train])
    pca.save(os.path.join(args.out, "wpca_model.npz"))

    usable = (meta["norm"] > 0).values & ((W > 0).sum(1) > 100)
    C = np.full((len(meta), args.k), np.nan, np.float32)
    C[usable] = pca.transform(X[usable], W[usable])
    chi2 = np.full(len(meta), np.nan)
    chi2[usable] = pca.chi2(X[usable], W[usable], C[usable])
    dof = (W > 0).sum(1) - args.k
    meta["wpca_chi2nu"] = chi2 / np.maximum(dof, 1)
    meta["train"] = train
    meta["usable"] = usable
    for i in range(args.k):
        meta[f"pc{i + 1}"] = C[:, i]
    print(f"projection done {time.time() - t0:.0f}s", flush=True)

    # validation chi2 vs training chi2 (overfitting check)
    val = select_train(meta, args.snr_min, args.good_min, folds=[8, 9])
    print(f"median chi2nu train={np.nanmedian(meta.wpca_chi2nu[train]):.3f} "
          f"val={np.nanmedian(meta.wpca_chi2nu[val]):.3f} all={np.nanmedian(meta.wpca_chi2nu[usable]):.3f}")

    plot_components(wave, pca, os.path.join(args.out, "components.png"))

    # UMAP on one spectrum per object, S/N-selected
    import umap

    sel = best_per_object(meta, train | val)
    rng = np.random.default_rng(0)
    if len(sel) > args.umap_max:
        sel = np.sort(rng.choice(sel, args.umap_max, replace=False))
    emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(C[sel])
    meta.loc[sel, "umap1"] = emb[:, 0]
    meta.loc[sel, "umap2"] = emb[:, 1]
    scatter_panels(emb, meta.loc[sel], os.path.join(args.out, "umap.png"),
                   f"UMAP of weighted-PCA coefficients (k={args.k}, {len(sel)} objects, S/N>{args.snr_min})",
                   chi2_col="wpca_chi2nu")
    scatter_panels(C[sel][:, :2], meta.loc[sel], os.path.join(args.out, "pc1_pc2.png"), "PC1 vs PC2",
                   chi2_col="wpca_chi2nu")
    print(f"umap done {time.time() - t0:.0f}s", flush=True)

    # largest reconstruction residuals among good-S/N spectra
    cand = np.flatnonzero(train | val)
    top = cand[np.argsort(-meta.wpca_chi2nu.values[cand])]
    top = top[:12]
    plot_outliers(wave, X, W, pca.reconstruct(C[top]), meta, top, os.path.join(args.out, "top_chi2.png"),
                  chi2_col="wpca_chi2nu")

    meta.to_parquet(os.path.join(args.out, "wpca_results.parquet"), index=False)
    print(f"all done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
