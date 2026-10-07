"""Build the HDF5 spectrum cube from all pfsObject files.

Output (HDF5):
  wave             (NPIX,)     bin centres [nm]
  flux, ivar, sky  (N, NPIX)   float32, resampled onto the common log-lambda grid
and a parquet table with one row per spectrum (key: catId, objId, combination),
joined with the `targets` table of the metadata DB.

Usage:
  uv run python scripts/build_dataset.py --out data/pfs_spectra.h5 [--limit 1000]
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd

from pfs_specclass import preprocess as pp

DEFAULT_FITS_ROOT = "/mnt/ugnas/work/PFS/CommFiller/pfs-target-viewer/extracted_targets/fits"
DEFAULT_DB = "/mnt/ugnas/work/PFS/CommFiller/pfs-target-viewer/pfs_metadata.sqlite3"


def list_files(root: str) -> list[str]:
    files = []
    for cat in sorted(os.listdir(root)):
        for sub in sorted(os.listdir(os.path.join(root, cat))):
            d = os.path.join(root, cat, sub)
            files.extend(os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".fits"))
    return files


def _work(path):
    try:
        return pp.read_pfsobject(path)
    except Exception as e:  # keep going; failures are reported at the end
        return f"{path}: {e!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits-root", default=DEFAULT_FITS_ROOT)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", default="data/pfs_spectra.h5")
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = list_files(args.fits_root)
    if args.limit:
        rng = np.random.default_rng(0)
        files = sorted(rng.choice(files, args.limit, replace=False))
    n = len(files)
    print(f"{n} files, NPIX={pp.NPIX}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    metas, failures = [], []
    t0 = time.time()
    with h5py.File(tmp, "w") as h:
        h["wave"] = pp.WAVE
        h["wave"].attrs.update(lam_min=pp.LAM_MIN, lam_max=pp.LAM_MAX, dloglam=pp.DLOGLAM)
        chunks = (256, pp.NPIX)
        dset = {
            k: h.create_dataset(k, (n, pp.NPIX), "f4", chunks=chunks, maxshape=(None, pp.NPIX))
            for k in ("flux", "ivar", "sky")
        }
        buf = {k: np.zeros((256, pp.NPIX), "f4") for k in dset}
        row = 0  # rows written to file
        nb = 0  # rows in buffer

        def flush():
            nonlocal row, nb
            for k in dset:
                dset[k][row : row + nb] = buf[k][:nb]
            row += nb
            nb = 0

        with Pool(args.nproc) as pool:
            for i, res in enumerate(pool.imap(_work, files, chunksize=32)):
                if isinstance(res, str):
                    failures.append(res)
                    continue
                buf["flux"][nb], buf["ivar"][nb], buf["sky"][nb] = res.flux, res.ivar, res.sky
                metas.append(res.meta)
                nb += 1
                if nb == 256:
                    flush()
                if (i + 1) % 5000 == 0:
                    el = time.time() - t0
                    print(f"{i + 1}/{n}  {el / 60:.1f} min  eta {(n - i - 1) * el / (i + 1) / 60:.1f} min", flush=True)
        flush()
        for k in dset:  # drop rows reserved for failed files
            dset[k].resize((row, pp.NPIX))
    os.replace(tmp, args.out)

    meta = pd.DataFrame(metas)
    meta["row"] = np.arange(len(meta))
    con = sqlite3.connect(args.db)
    tg = pd.read_sql(
        "select catId, objId, combination, classificationName, probaGalaxy, probaStar, probaQSO,"
        " bestRedshift, bestRedshiftError, bestVelocity, bestSubClass, nvisit, exptime from targets",
        con,
    )
    meta = meta.merge(tg, on=["catId", "objId", "combination"], how="left", validate="one_to_one")
    meta_path = os.path.splitext(args.out)[0] + "_meta.parquet"
    meta.to_parquet(meta_path, index=False)

    print(f"done: {row} spectra in {(time.time() - t0) / 60:.1f} min -> {args.out}, {meta_path}")
    print(f"unmatched in DB: {meta['classificationName'].isna().sum()}")
    print(f"failures: {len(failures)}")
    for f in failures[:20]:
        print("  ", f)
    if failures:
        with open(os.path.splitext(args.out)[0] + "_failures.txt", "w") as fh:
            fh.write("\n".join(failures))
    return 0


if __name__ == "__main__":
    sys.exit(main())
