"""Compare representations (weighted PCA vs autoencoder) on held-out objects.

Metrics
  * median reduced chi^2 of the reconstruction (validation/test folds, S/N > 5)
  * repeat-observation consistency: latent distance between two combinations of the
    same object, relative to random pairs (lower = more S/N/mask invariant)
  * arm-configuration leakage: AUC for predicting "includes m arm" from the latent
  * instrument leakage: AUC for predicting the spectrograph (one-vs-rest, macro) and whether the
    spectrum was taken on a hot fibre, from the latent (objects are assigned to fibres at
    random, so 0.5 is ideal)
  * fraction of quality-flagged spectra among the top-N anomalies (lower = anomalies are
    less dominated by known instrumental problems)
  * kNN (k=15) agreement with DRP classification for S/N > 5 objects, folds 8-9 queried
    against folds 0-7 (labels are noisy; this only measures information content)

Usage:
  uv run python scripts/compare.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from pfs_specclass.quality import HOT_FIBER, is_clean


def evaluate(df, cols, chi2_col, snr_min=5.0, qflag=None):
    ok = df["usable"].values & (df["snrBin"].values > snr_min) & (df["goodFrac"].values > 0.8)
    held = ok & df["fold"].isin([8, 9]).values
    res = {"median chi2nu (held-out)": float(np.nanmedian(df.loc[held, chi2_col]))}

    Z = StandardScaler().fit(df.loc[ok, cols].values).transform(df[cols].fillna(0).values)
    g = df[ok].groupby(["catId", "objId"]).indices
    gi = np.flatnonzero(ok)
    pairs = np.array([(gi[ix[0]], gi[ix[1]]) for ix in g.values() if len(ix) >= 2])
    d_same = np.linalg.norm(Z[pairs[:, 0]] - Z[pairs[:, 1]], axis=1)
    r = np.random.default_rng(0).choice(gi, (len(pairs), 2))
    d_rand = np.linalg.norm(Z[r[:, 0]] - Z[r[:, 1]], axis=1)
    res["repeat pairs"] = len(pairs)
    res["same/random latent distance"] = float(np.median(d_same) / np.median(d_rand))

    lab = df["classificationName"].fillna("NONE").values
    tr = ok & df["fold"].isin(range(8)).values & (lab != "NONE")
    te = held & (lab != "NONE")
    knn = KNeighborsClassifier(15).fit(Z[tr], lab[tr])
    pred = knn.predict(Z[te])
    res["kNN acc vs DRP class"] = float((pred == lab[te]).mean())
    for c in ("STAR", "GALAXY", "QSO"):
        m = lab[te] == c
        res[f"  recall {c}"] = float((pred[m] == c).mean()) if m.any() else np.nan

    # nuisance leakage: how well does the latent predict the arm configuration?
    has_m = df["arms"].str.contains("m").values
    for name, clf in (("logistic", LogisticRegression(max_iter=2000)), ("kNN", KNeighborsClassifier(15))):
        clf.fit(Z[tr], has_m[tr])
        res[f"m-arm AUC ({name}, 0.5=ideal)"] = float(roc_auc_score(has_m[te], clf.predict_proba(Z[te])[:, 1]))

    # instrument leakage: spectrograph and hot-fibre membership
    sm = pd.to_numeric(df["fiberIds"].fillna("").str.split(",").str[0], errors="coerce")
    sm = ((sm - 1) // 651 + 1).values
    single = df["spectrographs"].fillna("").str.len().values == 1
    trs, tes = tr & single & np.isfinite(sm), te & single & np.isfinite(sm)
    clf = LogisticRegression(max_iter=2000).fit(Z[trs], sm[trs])
    res["spectrograph AUC (logistic, 0.5=ideal)"] = float(
        roc_auc_score(sm[tes], clf.predict_proba(Z[tes]), multi_class="ovr", average="macro"))
    if qflag is not None:
        hot = (qflag & HOT_FIBER) > 0
        clf = KNeighborsClassifier(15).fit(Z[tr], hot[tr])
        res["hot-fibre AUC (kNN, 0.5=ideal)"] = float(roc_auc_score(hot[te], clf.predict_proba(Z[te])[:, 1]))

    if "anom_score" in df and qflag is not None:
        cand = np.flatnonzero(ok)
        flagged = ~is_clean(qflag[cand])
        res["flagged frac (all S/N>5)"] = float(flagged.mean())
        for col, name in (("chi2nu", "chi2"), ("latent_logp", "low density")):
            v = df[col].values[cand]
            order = np.argsort(-v if col == "chi2nu" else v)
            for n in (100, 1000):
                res[f"flagged frac in top-{n} {name}"] = float(flagged[order[:n]].mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wpca", default="outputs/wpca/wpca_results.parquet")
    ap.add_argument("--ae", nargs="+", default=["outputs/ae/ae_results.parquet"])
    ap.add_argument("--k", type=int, default=10, help="number of PCs to compare (match AE latent dim)")
    ap.add_argument("--quality", default="data/pfs_spectra_quality.parquet")
    args = ap.parse_args()
    qflag = pd.read_parquet(args.quality).sort_values("row")["qflag"].values
    fib = pd.read_parquet(args.quality.replace("_quality", "_meta"), columns=["row", "fiberIds", "spectrographs"])
    fib = fib.set_index("row")

    def with_fibres(df):
        return df.drop(columns=["fiberIds", "spectrographs"], errors="ignore").join(fib, on="row")

    wp = with_fibres(pd.read_parquet(args.wpca))
    rows = {f"wPCA (k={args.k})": evaluate(wp, [f"pc{i + 1}" for i in range(args.k)], "wpca_chi2nu",
                                           qflag=qflag[wp["row"].values])}
    for path in args.ae:
        ae = with_fibres(pd.read_parquet(path))
        lat = [c for c in ae.columns if c.startswith("s") and c[1:].isdigit()]
        rows[path.split("/")[-2]] = evaluate(ae, lat, "chi2nu", qflag=qflag[ae["row"].values])
    print(pd.DataFrame(rows).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
