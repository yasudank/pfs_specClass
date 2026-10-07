"""Spender-style spectrum autoencoder (Melchior et al. 2022), adapted to PFS.

Encoder: conv stack + attention pooling over wavelength -> latent s.
Decoder: MLP -> rest-frame spectrum on a log-lambda grid, then shifted by the
redshift (pass z=0 to work in the observed frame) and sampled on the observed
grid.  Training minimises the inverse-variance weighted chi^2, so masked pixels
(weight 0) and heteroscedastic noise are handled directly.  The spectrum
amplitude is solved analytically, so the latent describes shape only.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, n_latent=10, channels=(64, 128, 256), kernels=(5, 11, 21), dropout=0.1, n_coarse=32):
        super().__init__()
        layers = []
        cin = 2  # flux, observed-pixel indicator
        for i, (c, k) in enumerate(zip(channels, kernels)):
            layers += [nn.Conv1d(cin, c, k, padding=k // 2), nn.PReLU(c), nn.Dropout(dropout)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool1d(2 if i == 0 else 4))
            cin = c
        # last conv produces 2x channels: half are values, half are attention logits
        layers.append(nn.Conv1d(cin, 2 * cin, 1))
        self.conv = nn.Sequential(*layers)
        self.c = cin
        self.n_coarse = n_coarse
        # coarse continuum (weighted means in n_coarse segments + coverage) goes straight to the MLP
        self.mlp = nn.Sequential(nn.Linear(cin + 2 * n_coarse, 256), nn.PReLU(), nn.Linear(256, 128), nn.PReLU(),
                                 nn.Linear(128, n_latent))

    def coarse(self, x, w):
        B, D = x.shape
        n = self.n_coarse
        pad = (-D) % n
        xs = F.pad(x * w, (0, pad)).view(B, n, -1).sum(-1)
        ws = F.pad(w, (0, pad)).view(B, n, -1).sum(-1)
        mean = torch.where(ws > 0, xs / ws.clamp_min(1e-12), torch.zeros_like(xs))
        cover = F.pad((w > 0).float(), (0, pad)).view(B, n, -1).mean(-1)
        return torch.cat([mean.clamp(-3, 10), cover], 1)

    def forward(self, x, w):
        obs = (w > 0).float()
        # clamp so unmasked sky residual spikes cannot dominate the encoder input
        h = self.conv(torch.stack([x.clamp(-3, 10) * obs, obs], 1))
        v, a = h[:, : self.c], h[:, self.c :]
        a = torch.softmax(a, dim=-1)
        pooled = (v * a).sum(-1)
        return self.mlp(torch.cat([pooled, self.coarse(x, w)], 1))


class Decoder(nn.Module):
    def __init__(self, n_latent, n_rest, hidden=(128, 512, 1024)):
        super().__init__()
        dims = (n_latent, *hidden)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.PReLU(b)]
        layers.append(nn.Linear(dims[-1], n_rest))
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        # softplus keeps the model non-negative-ish without hard clipping
        return F.softplus(self.net(s)) - 0.1


class SpectrumAutoencoder(nn.Module):
    """Autoencoder on a log-lambda grid with optional redshifting."""

    def __init__(self, wave_obs: np.ndarray, n_latent=10, z_max=0.0, rest_oversample=1.0):
        super().__init__()
        loglam = np.log(wave_obs)
        dl = float(np.median(np.diff(loglam))) / rest_oversample
        lo = loglam[0] - np.log1p(z_max) - 2 * dl
        hi = loglam[-1] + 2 * dl
        n_rest = int(np.ceil((hi - lo) / dl)) + 1
        self.register_buffer("loglam_obs", torch.tensor(loglam, dtype=torch.float32))
        self.loglam_rest0 = lo
        self.dloglam_rest = dl
        self.n_rest = n_rest
        self.encoder = Encoder(n_latent)
        self.decoder = Decoder(n_latent, n_rest)

    @property
    def wave_rest(self):
        return np.exp(self.loglam_rest0 + self.dloglam_rest * np.arange(self.n_rest))

    def transform(self, rest, z):
        """Sample the rest-frame model at observed wavelengths for redshift z (linear interp)."""
        pos = (self.loglam_obs[None, :] - torch.log1p(z)[:, None] - self.loglam_rest0) / self.dloglam_rest
        pos = pos.clamp(0, self.n_rest - 1.001)
        i0 = pos.floor().long()
        t = pos - i0
        return torch.gather(rest, 1, i0) * (1 - t) + torch.gather(rest, 1, i0 + 1) * t

    def forward(self, x, w, z=None, w_enc=None):
        """Return (latent, model).  ``w_enc`` optionally overrides the mask the encoder sees."""
        if z is None:
            z = torch.zeros(x.shape[0], device=x.device)
        s = self.encoder(x, w if w_enc is None else w_enc)
        rest = self.decoder(s)
        model = self.transform(rest, z)
        amp = fit_amplitude(x, w, model)
        return s, amp[:, None] * model


def fit_amplitude(x, w, m):
    num = (w * x * m).sum(1)
    den = (w * m * m).sum(1).clamp_min(1e-12)
    return (num / den).clamp_min(1e-3)


def chi2_per_spectrum(x, w, model):
    return (w * (x - model) ** 2).sum(1)


def random_block_mask(w, n_blocks=2, max_frac=0.08, generator=None):
    """Augmentation: zero random contiguous wavelength blocks (simulates arm gaps)."""
    B, D = w.shape
    w = w.clone()
    L = (torch.rand(B, n_blocks, device=w.device) * max_frac * D).long()
    S = (torch.rand(B, n_blocks, device=w.device) * D).long()
    idx = torch.arange(D, device=w.device)[None, None, :]
    m = ((idx >= S[..., None]) & (idx < (S + L)[..., None])).any(1)
    w[m] = 0
    return w


# Wavelength ranges [nm] without data in b+m+n coadds (measured on the cube).
BMN_GAPS = ((655.8, 702.7), (895.8, 937.5))


def gap_mask(wave, gaps=BMN_GAPS):
    m = np.zeros(len(wave), bool)
    for lo, hi in gaps:
        m |= (wave > lo) & (wave < hi)
    return torch.as_tensor(m)


def augment_mask(w, gap, p_gap=0.5):
    """Random blocks plus, with probability p_gap, the b+m+n coverage gaps."""
    w = random_block_mask(w)
    apply = torch.rand(w.shape[0], 1, device=w.device) < p_gap
    return torch.where(apply & gap[None, :].to(w.device), torch.zeros_like(w), w)
