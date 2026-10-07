"""Compute quality metrics, hot fibres and flags for every spectrum in the cube.

Output:
  data/pfs_spectra_quality.parquet   one row per cube row (metrics + qflag)
  data/hot_fibers.csv                per-fibre bump statistics, column `hot`

Usage:
  uv run python scripts/build_quality.py --data data/pfs_spectra.h5
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np
import pandas as pd

from pfs_specclass import quality as q
from pfs_specclass.dataset import WeightConfig, load_meta, normalise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pfs_spectra.h5")
    ap.add_argument("--snr-min", type=float, default=3.0, help="S/N cut for the hot-fibre statistics")
    args = ap.parse_args()

    meta = load_meta(args.data, quality=False)
    cfg = WeightConfig()
    parts = []
    with h5py.File(args.data, "r") as h:
        wave = h["wave"][:]
        for a in range(0, len(meta), 20000):
            b = min(a + 20000, len(meta))
            f, w = normalise(h["flux"][a:b], h["ivar"][a:b], h["sky"][a:b],
                             meta["norm"].values[a:b].astype(np.float32), cfg)
            parts.append(q.metrics(wave, f, w))
    m = pd.concat(parts, ignore_index=True)

    sel = (meta["snrBin"] > args.snr_min).values & m["peak"].notna().values
    hit = (m.loc[sel, "peak"] > q.PEAK_THRESH) | (m.loc[sel, "rise"] > q.RISE_THRESH)
    fib = q.find_hot_fibers(meta.loc[sel, "fiberIds"], hit.values)
    hot = set(fib.index[fib["hot"]])
    fib.to_csv(args.data.replace(".h5", "_hot_fibers.csv"))

    m["qflag"] = q.flags(m, meta["fiberIds"], hot, meta["arms"])
    m["row"] = np.arange(len(m))
    m.to_parquet(args.data.replace(".h5", "_quality.parquet"), index=False)

    print(f"global bump|step rate (S/N>{args.snr_min}): {fib.attrs['global_rate']:.2%}; hot fibres: {len(hot)} "
          f"(single {fib['hot_single'].sum()}, pooled {fib['hot_pooled'].sum()})")
    print("hot fibres:", sorted(hot))
    ok = meta["snrBin"] > args.snr_min
    for bit, name in q.FLAG_NAMES.items():
        print(f"{name:10s} all {np.mean(m.qflag & bit > 0):6.2%}   S/N>{args.snr_min} {np.mean(m.qflag[ok] & bit > 0):6.2%}")
    drop = ((m.qflag & (q.NARM_BUMP | q.NARM_STEP)) > 0) & ((m.qflag & q.HOT_FIBER) > 0)
    local = ((m.qflag & q.NARM_BUMP) > 0) & ((m.qflag & q.HOT_FIBER) == 0)
    print(f"n arm dropped: {drop.mean():.2%}   925-1015 nm masked: {local.mean():.2%}")


if __name__ == "__main__":
    main()
