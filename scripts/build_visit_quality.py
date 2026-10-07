"""Find hot exposure/camera units and flag their spectra (HOT_VISIT + hot 20-nm bins).

Run after build_quality.py (which rewrites qflag); this script is idempotent.

Output:
  data/pfs_spectra_visit_units.parquet   unit tables (kind = visitset | visit) with hot bins
  data/pfs_spectra_quality.parquet       updated: hot_bins column, HOT_VISIT bit in qflag

Usage:
  uv run python scripts/build_visit_quality.py --model outputs/ae_v4
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from pfs_specclass import quality as q
from pfs_specclass import visit_quality as vq
from pfs_specclass.dataset import WeightConfig, load_arrays, load_meta
from pfs_specclass.model import SpectrumAutoencoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pfs_spectra.h5")
    ap.add_argument("--model", default="outputs/ae_v4")
    ap.add_argument("--snr-min", type=float, default=3.0)
    args = ap.parse_args()
    t0 = time.time()

    ck = torch.load(f"{args.model}/ae.pt", map_location="cuda", weights_only=False)
    net = SpectrumAutoencoder(ck["wave"], n_latent=ck["args"]["latent"]).cuda()
    net.load_state_dict(ck["state"])
    net.eval()
    cfg = WeightConfig(sky_frac=ck["args"]["sky_frac"], flux_frac=ck["args"]["flux_frac"])

    meta = load_meta(args.data)
    wave, X, W, _ = load_arrays(args.data, cfg=cfg, visit_mask=False)
    R, S = vq.leave_arm_out_residuals(net, wave, X, W)
    del X, W
    print(f"residuals {time.time() - t0:.0f}s", flush=True)

    sel = (meta["snrBin"] > args.snr_min).values & (meta["norm"] > 0).values & np.isfinite(R).any(1)
    D = vq.knn_deviation(R, S, meta["visits"].fillna("").values, sel, vq.arm_group(meta["arms"]))
    np.savez_compressed(args.data.replace(".h5", "_armres.npz"), R=R, D=D)
    print(f"kNN baseline {time.time() - t0:.0f}s", flush=True)

    rows_sel = np.flatnonzero(sel)
    rows_all = np.arange(len(meta))
    tables, mem_all = [], []
    for kind in ("visitset", "visit"):
        tab = vq.hot_units(D, vq.membership(meta, rows_sel, kind))
        tab["kind"] = kind
        tables.append(tab)
        mem_all.append(vq.membership(meta, rows_all, kind))
        ok = tab["n"] >= vq.MIN_SPECTRA
        print(f"{kind:8s}: units {len(tab)} (>= {vq.MIN_SPECTRA} spectra: {ok.sum()}), hot {(tab.n_hot_bins > 0).sum()}",
              " by arm config:", tab[ok].groupby("group").n_hot_bins.apply(lambda v: f"{(v > 0).mean():.1%}").to_dict())
    pd.concat(tables, ignore_index=True).to_parquet(args.data.replace(".h5", "_visit_units.parquet"), index=False)

    hot_bins = vq.spectrum_hot_bins(len(meta), mem_all, tables)
    qpath = args.data.replace(".h5", "_quality.parquet")
    qm = pd.read_parquet(qpath).sort_values("row").reset_index(drop=True)
    qm["hot_bins"] = hot_bins
    qm["qflag"] = (qm["qflag"].values & ~(q.HOT_VISIT | q.HOT_VISIT_HEAVY)) | q.visit_flags(hot_bins)
    qm.to_parquet(qpath, index=False)

    ok = (meta["snrBin"] > args.snr_min).values
    nb = vq.unbits(hot_bins).sum(1)
    print(f"HOT_VISIT: all {np.mean(hot_bins != 0):.1%}, S/N>{args.snr_min} {np.mean(hot_bins[ok] != 0):.1%}; "
          f"hot bins per flagged spectrum: median {np.median(nb[hot_bins != 0]):.0f}")
    excl = (qm["qflag"].values & q.EXCLUDE_FROM_TRAINING) != 0
    s5 = (meta["snrBin"] > 5).values
    print(f"excluded from training by flags (S/N>5): {np.mean(excl[s5]):.1%};  HOT_VISIT_HEAVY {np.mean((qm.qflag.values[s5] & q.HOT_VISIT_HEAVY) > 0):.1%}")
    hv = (qm["qflag"].values & q.HOT_VISIT) > 0
    print("HOT_VISIT share (S/N>5) by combination:", pd.Series(hv[s5]).groupby(meta["combination"].values[s5]).mean().round(3).to_dict())
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
