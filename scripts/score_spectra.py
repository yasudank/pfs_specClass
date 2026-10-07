"""Score new pfsObject spectra with a trained model.

Usage:
  uv run python scripts/score_spectra.py --model outputs/ae_v4 --out scores.csv FILE.fits [FILE.fits ...]
  uv run python scripts/score_spectra.py --model outputs/ae_v4 --out scores.csv --list files.txt
"""

from __future__ import annotations

import argparse

import pandas as pd

from pfs_specclass.scoring import SpectrumScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--list", help="text file with one FITS path per line")
    ap.add_argument("--model", default="outputs/ae_v4")
    ap.add_argument("--out", default="scores.csv")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    files = list(args.files)
    if args.list:
        files += [l.strip() for l in open(args.list) if l.strip()]
    scorer = SpectrumScorer(args.model)
    res = pd.concat([scorer.score_files(files[s:s + args.batch]) for s in range(0, len(files), args.batch)])
    res.to_csv(args.out, index=False)
    cols = ["catId", "objId", "combination", "snrBin", "knn_label", "knn_frac", "gmm_label", "chi2nu",
            "anom_score", "qflag_names"]
    print(res[cols].sort_values("anom_score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
