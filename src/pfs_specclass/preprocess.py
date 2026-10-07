"""Read pfsObject FITS files and resample them onto a common log-lambda grid.

All combinations (brn / bmn / selected_*) have different native wavelength
grids (11251, 11501 or 13293 pixels), so every spectrum is resampled onto one
log-lambda grid.  Masked pixels get zero weight; the output carries an inverse
variance (ivar) with ivar == 0 meaning "no information".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits

# Common grid: Δlnλ = 3.5e-4 (~105 km/s, ~0.28 nm at 800 nm).
LAM_MIN = 376.0  # nm
LAM_MAX = 1268.0  # nm
DLOGLAM = 3.5e-4
NPIX = int(np.ceil(np.log(LAM_MAX / LAM_MIN) / DLOGLAM))
LOGLAM_EDGES = np.log(LAM_MIN) + DLOGLAM * np.arange(NPIX + 1)
WAVE = np.exp(0.5 * (LOGLAM_EDGES[:-1] + LOGLAM_EDGES[1:]))  # bin centres [nm]

# Approximate resolving powers of the PFS arms.
R_RED_LOW = 3000.0
R_RED_MED = 5000.0
M_ARM_RANGE = (700.0, 890.0)  # nm, region covered by the medium-resolution arm

FNAME_RE = re.compile(
    r"pfsObject_(?P<catname>[^_]+)_(?:.*_)?(?P<catid>\d+)_(?P<objid>\d+)_(?P<comb>[a-z]+_[A-Za-z0-9]+)\.fits$")


@dataclass
class Spectrum:
    flux: np.ndarray  # (NPIX,) float32
    ivar: np.ndarray  # (NPIX,) float32, 0 = masked
    sky: np.ndarray  # (NPIX,) float32
    meta: dict = field(default_factory=dict)


def _smooth_m_arm(wave, flux, var, good):
    """Degrade the medium-resolution red arm to the low-resolution red arm.

    Normalised Gaussian convolution restricted to good pixels; the variance is
    propagated assuming independent pixels.
    """
    sel = (wave > M_ARM_RANGE[0]) & (wave < M_ARM_RANGE[1])
    if sel.sum() < 10:
        return flux, var
    idx = np.flatnonzero(sel)
    lam = wave[idx]
    dlam = np.median(np.diff(lam))
    lam_c = np.median(lam)
    fwhm = np.sqrt((lam_c / R_RED_LOW) ** 2 - (lam_c / R_RED_MED) ** 2)
    sigma_pix = fwhm / 2.3548 / dlam
    half = int(np.ceil(4 * sigma_pix))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_pix) ** 2)

    w = good[idx].astype(np.float64)
    f = np.where(good[idx], flux[idx], 0.0)
    v = np.where(good[idx], var[idx], 0.0)
    norm = np.convolve(w, k, mode="same")
    fs = np.convolve(f * w, k, mode="same")
    vs = np.convolve(v * w, k**2, mode="same")
    ok = norm > 0.5 * k.sum()
    flux = flux.copy()
    var = var.copy()
    flux[idx[ok]] = fs[ok] / norm[ok]
    var[idx[ok]] = vs[ok] / norm[ok] ** 2
    return flux, var


def resample(wave, flux, var, good, sky=None):
    """Inverse-variance weighted average of native pixels into log-lambda bins."""
    b = np.floor((np.log(wave) - LOGLAM_EDGES[0]) / DLOGLAM).astype(np.int64)
    inside = (b >= 0) & (b < NPIX) & good
    b = b[inside]
    iv = 1.0 / var[inside]
    sum_w = np.bincount(b, weights=iv, minlength=NPIX)
    sum_wf = np.bincount(b, weights=iv * flux[inside], minlength=NPIX)
    out_flux = np.zeros(NPIX)
    np.divide(sum_wf, sum_w, out=out_flux, where=sum_w > 0)
    out_sky = np.zeros(NPIX)
    if sky is not None:
        n = np.bincount(b, minlength=NPIX)
        s = np.bincount(b, weights=sky[inside], minlength=NPIX)
        np.divide(s, n, out=out_sky, where=n > 0)
    return out_flux.astype(np.float32), sum_w.astype(np.float32), out_sky.astype(np.float32)


def read_pfsobject(path: str, mask_bits: int = 0xFFFFFFFF, smooth_m: bool = True) -> Spectrum:
    """Read one pfsObject file.  Pixels with any of ``mask_bits`` set are dropped."""
    with fits.open(path, memmap=False) as h:
        wave = h["WAVELENGTH"].data.astype(np.float64)
        flux = h["FLUX"].data.astype(np.float64)
        mask = h["MASK"].data.astype(np.int64)
        var = h["COVAR"].data[0].astype(np.float64)
        sky = h["SKY"].data.astype(np.float64)
        th = h["TARGET"].header
        obs = h["OBSERVATIONS"].data
        arms = "".join(sorted(set(obs["arm"])))
        visits = sorted({int(v) for v in obs["visit"]})
        # expTime is per (visit, arm); count each visit once
        exp_by_visit = {int(v): float(t) for v, t in zip(obs["visit"], obs["expTime"])}
        fibers = sorted({int(f) for f in obs["fiberId"]})
        spectrographs = sorted({int(x) for x in obs["spectrograph"]})

    good = ((mask & mask_bits) == 0) & (var > 0) & np.isfinite(flux) & np.isfinite(var)
    native_snr = np.median(flux[good] / np.sqrt(var[good])) if good.any() else np.nan

    if smooth_m and "m" in arms:
        flux, var = _smooth_m_arm(wave, flux, var, good)

    f, iv, s = resample(wave, flux, var, good, sky)
    has = iv > 0
    m = FNAME_RE.search(path)
    meta = dict(
        catId=int(th["CATID"]),
        objId=int(th["OBJID"]),
        combination=m["comb"] if m else "",
        catName=m["catname"] if m else "",
        ra=float(th["RA"]),
        dec=float(th["DEC"]),
        arms=arms,
        nVisitObs=len(visits),
        expTimeObs=float(sum(exp_by_visit.values())),
        visits=",".join(map(str, visits)),
        fiberIds=",".join(map(str, fibers)),
        spectrographs=",".join(map(str, spectrographs)),
        snrNative=float(native_snr),
        snrBin=float(np.median(f[has] * np.sqrt(iv[has]))) if has.any() else np.nan,
        goodFrac=float(has.mean()),
        norm=float(np.median(f[has])) if has.any() else np.nan,
        path=path,
    )
    return Spectrum(f, iv, s, meta)
