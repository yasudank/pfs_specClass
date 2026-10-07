"""Train the spectrum autoencoder, encode all spectra, density + anomaly scores.

Usage:
  uv run python scripts/train_ae.py --data data/pfs_spectra.h5 --out outputs/ae
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from pfs_specclass.dataset import WeightConfig, best_per_object, load_arrays, load_meta, select_train
from pfs_specclass.model import SpectrumAutoencoder, augment_mask, chi2_per_spectrum, gap_mask
from pfs_specclass.plots import CAT, plot_outliers, scatter_panels, style
from pfs_specclass.quality import EXCLUDE_FROM_TRAINING, FLAG_NAMES, flag_string, is_clean
from pfs_specclass.scoring import percentile_in_bins


def batches(n, bs, shuffle, gen=None):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for s in range(0, n, bs):
        yield idx[s:s + bs]


def loss_fn(x, w, model):
    """Mean reduced chi^2 over the batch."""
    npix = (w > 0).sum(1).clamp_min(1)
    return (chi2_per_spectrum(x, w, model) / npix).mean()


@torch.no_grad()
def encode(net, X, W, device, bs=1024):
    net.eval()
    S, CHI2 = [], []
    for s in range(0, len(X), bs):
        x = torch.as_tensor(X[s:s + bs], device=device)
        w = torch.as_tensor(W[s:s + bs], device=device)
        lat, model = net(x, w)
        S.append(lat.cpu())
        CHI2.append(chi2_per_spectrum(x, w, model).cpu())
    return torch.cat(S).numpy(), torch.cat(CHI2).numpy()


@torch.no_grad()
def reconstruct(net, X, W, device):
    net.eval()
    x = torch.as_tensor(X, device=device)
    w = torch.as_tensor(W, device=device)
    return net(x, w)[1].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pfs_spectra.h5")
    ap.add_argument("--out", default="outputs/ae")
    ap.add_argument("--latent", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--snr-min", type=float, default=5.0)
    ap.add_argument("--good-min", type=float, default=0.8)
    ap.add_argument("--sky-frac", type=float, default=0.04)
    ap.add_argument("--flux-frac", type=float, default=0.02)
    ap.add_argument("--consistency", type=float, default=1.0,
                    help="weight of the latent consistency loss between two augmented views")
    ap.add_argument("--gmm", type=int, default=32)
    ap.add_argument("--umap-max", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-quality", action="store_true", help="ignore quality flags (weights and selection)")
    ap.add_argument("--exclude", default=None,
                    help="comma-separated flag names excluded from training "
                         "(default: quality.EXCLUDE_FROM_TRAINING); weights are adjusted regardless")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    config = dict(vars(args))
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    t0 = time.time()

    meta = load_meta(args.data)
    cfg = WeightConfig(sky_frac=args.sky_frac, flux_frac=args.flux_frac)
    use_q = not args.no_quality and "qflag" in meta
    if use_q:
        shutil.copy(args.data.replace(".h5", "_hot_fibers.csv"), os.path.join(args.out, "hot_fibers.csv"))
        units = args.data.replace(".h5", "_visit_units.parquet")
        if os.path.exists(units):
            shutil.copy(units, os.path.join(args.out, "visit_units.parquet"))
    if args.exclude is None:
        excl = EXCLUDE_FROM_TRAINING
    else:
        by_name = {v: k for k, v in FLAG_NAMES.items()}
        excl = 0
        for name in filter(None, args.exclude.split(",")):
            excl |= by_name[name.strip()]
    excl = excl if use_q else 0
    json.dump(config | {"exclude_from_training": flag_string(excl)},
              open(os.path.join(args.out, "config.json"), "w"), indent=1)
    wave, X, W, _ = load_arrays(args.data, cfg=cfg, quality=use_q)
    print(f"loaded {X.shape} in {time.time() - t0:.0f}s", flush=True)

    tr = select_train(meta, args.snr_min, args.good_min, folds=range(8), exclude_flags=excl)
    va = select_train(meta, args.snr_min, args.good_min, folds=[8], exclude_flags=excl)
    Xtr, Wtr = (torch.as_tensor(a[tr], device=device) for a in (X, W))
    Xva, Wva = (torch.as_tensor(a[va], device=device) for a in (X, W))
    print(f"train {len(Xtr)}  val {len(Xva)}", flush=True)

    net = SpectrumAutoencoder(wave, n_latent=args.latent).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(Xtr) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    gen = torch.Generator().manual_seed(args.seed)
    gap = gap_mask(wave).to(device)
    hist = []
    for ep in range(args.epochs):
        net.train()
        tl, nb = 0.0, 0
        for b in batches(len(Xtr), args.bs, True, gen):
            x, w = Xtr[b], Wtr[b]
            # encoder sees an augmented mask; loss uses the full set of good pixels
            # two augmented views for the encoder; reconstruction loss uses all good pixels
            s1, model = net(x, w, w_enc=augment_mask(w, gap))
            loss = loss_fn(x, w, model)
            if args.consistency > 0:
                s2 = net.encoder(x, augment_mask(w, gap))
                scale = s1.detach().var(0).mean().clamp_min(1e-6)
                loss = loss + args.consistency * ((s1 - s2) ** 2).mean() / scale
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            tl += loss.item()
            nb += 1
        net.eval()
        with torch.no_grad():
            vl = np.mean([loss_fn(Xva[b], Wva[b], net(Xva[b], Wva[b])[1]).item()
                          for b in batches(len(Xva), 1024, False)])
        hist.append((ep + 1, tl / nb, vl))
        print(f"epoch {ep + 1:3d}  train chi2nu {tl / nb:.4f}  val chi2nu {vl:.4f}  {time.time() - t0:.0f}s", flush=True)
    torch.save({"state": net.state_dict(), "args": vars(args), "wave": wave}, os.path.join(args.out, "ae.pt"))
    del Xtr, Wtr, Xva, Wva

    h = np.array(hist)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(h[:, 0], h[:, 1], color=CAT[0], lw=2, label="train")
    ax.plot(h[:, 0], h[:, 2], color=CAT[1], lw=2, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean reduced chi2")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "loss.png"), dpi=110)
    plt.close(fig)

    # encode everything usable
    usable = (meta["norm"] > 0).values & ((W > 0).sum(1) > 100)
    S = np.full((len(meta), args.latent), np.nan, np.float32)
    chi2 = np.full(len(meta), np.nan)
    S[usable], chi2[usable] = encode(net, X[usable], W[usable], device)
    meta["chi2nu"] = chi2 / np.maximum((W > 0).sum(1), 1)
    for i in range(args.latent):
        meta[f"s{i + 1}"] = S[:, i]
    meta["train"], meta["usable"] = tr, usable
    print(f"encoded {usable.sum()} spectra {time.time() - t0:.0f}s", flush=True)

    # latent density (one spectrum per object in the training set)
    fit_rows = best_per_object(meta, tr)
    gmm = GaussianMixture(args.gmm, covariance_type="full", random_state=args.seed, reg_covar=1e-4)
    gmm.fit(S[fit_rows])
    logp = np.full(len(meta), np.nan)
    logp[usable] = gmm.score_samples(S[usable])
    meta["latent_logp"] = logp
    meta["gmm_label"] = -1
    meta.loc[usable, "gmm_label"] = gmm.predict(S[usable])
    # S/N-conditioned anomaly scores in [0, 1]
    snr = meta["snrBin"].values
    meta["anom_chi2"] = percentile_in_bins(meta["chi2nu"].values, snr)
    meta["anom_density"] = percentile_in_bins(-logp, snr)
    meta["anom_score"] = np.fmax(meta["anom_chi2"], meta["anom_density"])
    joblib.dump(gmm, os.path.join(args.out, "gmm.joblib"))

    # repeat-observation consistency: latent distance between combinations of the same object
    good = usable & (snr > args.snr_min)
    g = meta[good].groupby(["catId", "objId"])
    pairs = [(ix[0], ix[1]) for ix in g.indices.values() if len(ix) >= 2]
    gi = np.flatnonzero(good)
    if pairs:
        p = np.array([(gi[a], gi[b]) for a, b in pairs])
        d_same = np.linalg.norm(S[p[:, 0]] - S[p[:, 1]], axis=1)
        rng = np.random.default_rng(0)
        r = rng.choice(gi, (len(p), 2))
        d_rand = np.linalg.norm(S[r[:, 0]] - S[r[:, 1]], axis=1)
        print(f"repeat pairs: {len(p)}  median latent distance same-object {np.median(d_same):.3f} "
              f"vs random {np.median(d_rand):.3f}")

    # UMAP
    import umap

    va_all = select_train(meta, args.snr_min, args.good_min, folds=range(10), exclude_flags=excl)
    sel = best_per_object(meta, va_all)
    if len(sel) > args.umap_max:
        sel = np.sort(np.random.default_rng(0).choice(sel, args.umap_max, replace=False))
    emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(S[sel])
    meta.loc[sel, "umap1"], meta.loc[sel, "umap2"] = emb[:, 0], emb[:, 1]
    scatter_panels(emb, meta.loc[sel], os.path.join(args.out, "umap.png"),
                   f"UMAP of autoencoder latents (dim={args.latent}, {len(sel)} objects, S/N>{args.snr_min})")
    print(f"umap done {time.time() - t0:.0f}s", flush=True)

    # anomaly galleries among S/N > snr_min: all, and quality-clean only (qflag == 0)
    galleries = {"": va_all}
    if use_q:
        meta["qflag_names"] = meta["qflag"].map(flag_string)
        galleries["_clean"] = va_all & is_clean(meta["qflag"].values)
    for suffix, sel in galleries.items():
        cand = np.flatnonzero(sel)
        for name, col in (("top_chi2", "chi2nu"), ("top_lowdensity", "latent_logp")):
            v = meta[col].values[cand]
            top = cand[np.argsort(-v if col == "chi2nu" else v)][:12]
            plot_outliers(wave, X, W, reconstruct(net, X[top], W[top], device), meta, top,
                          os.path.join(args.out, f"{name}{suffix}.png"), extra="latent_logp")

    meta.to_parquet(os.path.join(args.out, "ae_results.parquet"), index=False)
    print(f"all done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
