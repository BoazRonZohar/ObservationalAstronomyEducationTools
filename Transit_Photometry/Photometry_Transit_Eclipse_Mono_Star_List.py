#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photometry_Transit_Eclipse_Mono_Star_List.py  (FWHM-based aperture version)
=======================================================
SCRIPT VERSION: v7 (2026-07-24)
Includes: CCD noise equation (gain/RON/dark), None-overlap crash fix,
flip-detection confidence margin + sparse-field auto-disable,
interactive gain/RON prompts, adaptive anchor stamp sizing,
auto-reject noisy comp stars (std+drift+tail_offset vs. a detected
baseline segment, gap-based tail-cluster detection), comp_stars quality
table + per-comp-star plots.
If you are unsure which copy of this script you are running, search for
this "SCRIPT VERSION" line - if it's missing, you have an OLD copy.

Differential photometry of a variable star + comparison stars across a
large series (tens to 200+) of FITS frames that have NO WCS / astrometric
solution (e.g. frames already stacked/calibrated in AIP4WIN).

METHOD ("aperture method" driven by FWHM, as requested)
--------------------------------------------------------
1. You define pixel positions of the target star (V) and all candidate
   comparison stars (C1, C2, ...) ONCE, on a single reference frame
   (a simple CSV file - see star_list_example.csv).
2. The script measures the FWHM (full width at half maximum) of the stellar
   profile directly from the data, by fitting a 2D Gaussian to each selected
   star on the reference frame.
3. You are then asked (interactively, or via command-line flags for batch/
   unattended use) to choose:
       - which comparison stars to actually use for this run,
       - the aperture radius as a MULTIPLE of the measured FWHM,
       - the inner and outer sky-annulus radii, also as multiples of FWHM.
   (Typical values: aperture = 3 x FWHM, annulus = 4 x FWHM to 6 x FWHM.)
4. For every FITS frame in the input folder:
       a. The anchor comparison star is re-located by cross-correlating a
          small template cut from the reference frame against a search
          window in the new frame (handles the fact there is no WCS and the
          field shifts between frames).
       b. That shift is applied to every other selected star as a first
          guess, then each star is individually re-centered with a local
          centroid (sub-pixel accuracy).
       c. Aperture photometry (source aperture + sky annulus, using the
          radii computed in step 3) is performed at every refined position.
5. Results are written to CSV, Excel (.xlsx), and/or a plain text table,
   including the differential magnitude of the target relative to the
   ensemble of comparison stars used.

INSTALL
-------
    pip install numpy scipy astropy photutils scikit-image pandas openpyxl matplotlib

INPUT FILES
-----------
1. A reference FITS frame (one frame where you already know the pixel
   position of each star).
2. A star list CSV (name,x,y,role) - see star_list_example.csv. role must be
   "target" (exactly one row) or "comp" (any number of rows).
3. A folder with all the science FITS frames to measure.

USAGE (interactive - will ask for FWHM multipliers and which comp stars to use)
--------------------------------------------------------------------------------
    python Photometry_Transit_Eclipse_Mono_Star_List.py \
        --input_dir  /path/to/fits_folder \
        --ref_file   /path/to/reference_frame.fits \
        --star_list  star_list.csv \
        --output results \
        --output_format xlsx,csv,txt

USAGE (fully unattended / batch - no prompts)
-----------------------------------------------
    python Photometry_Transit_Eclipse_Mono_Star_List.py \
        --input_dir /path/to/fits_folder --ref_file ref.fits \
        --star_list star_list.csv \
        --stars C1,C2,C3,C4,C6 \
        --k_aperture 3.0 --k_ann_in 4.0 --k_ann_out 6.0 \
        --output results --output_format xlsx

----------------------------------------------------------------------
Created by: Dr. Boaz Ron Zohar
Affiliation: Kinneret Observatory
Member of the LCO Global Sky Partners programme
Part of Observational Astronomy Education Tools
https://github.com/BoazRonZohar/ObservationalAstronomyEducationTools
"""

import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clipped_stats

from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats
from photutils.centroids import centroid_com

from scipy.optimize import curve_fit
from skimage.feature import match_template

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# Star list handling
# --------------------------------------------------------------------------

def load_star_list(csv_path):
    df = pd.read_csv(csv_path)
    required = {"name", "x", "y", "role"}
    if not required.issubset(df.columns):
        raise ValueError(f"star_list CSV must have columns {required}, got {list(df.columns)}")
    df["role"] = df["role"].str.lower().str.strip()
    if (df["role"] == "target").sum() != 1:
        raise ValueError("star_list must contain exactly one row with role='target'")
    return df.reset_index(drop=True)


def detect_star_list_format(path):
    """Returns 'startool' if the file looks like an AIP4WIN 'Star Data Tool'
    export (a text report with an X/Y/Sigma/FWHM table), or 'csv' if it looks
    like the simple name,x,y,role CSV format."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        head = f.read(4000)
    if "Star Data Tool" in head:
        return "startool"
    if re.search(r"^\s*name\s*,\s*x\s*,\s*y\s*,\s*role", head, re.IGNORECASE | re.MULTILINE):
        return "csv"
    # fallback heuristic: a tab/space-separated table with these column headers
    if re.search(r"\bX\b", head) and re.search(r"\bY\b", head) and re.search(r"\bFWHM\b", head):
        return "startool"
    return "csv"


def parse_star_data_tool(path):
    """Parse an AIP4WIN 'Star Data Tool' text export.
    Returns (meta_dict, DataFrame) where the DataFrame has one row per
    detected star with columns: x, y, sigma, fwhm, star_sky, sky_adu
    (row order preserved, 0-indexed; display as 1-based to the user)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    meta = {}
    header_idx = None
    for i, line in enumerate(lines):
        m = re.search(r"Radius of star diaphragm:\s*([\d.]+)", line)
        if m:
            meta["aperture_radius"] = float(m.group(1))
        m = re.search(r"Sky annulus inner radius:\s*([\d.]+)", line)
        if m:
            meta["ann_in"] = float(m.group(1))
        m = re.search(r"Sky annulus outer radius:\s*([\d.]+)", line)
        if m:
            meta["ann_out"] = float(m.group(1))
        m = re.search(r"Image:\s*(.+)", line)
        if m:
            meta["image"] = m.group(1).strip()
        if re.search(r"\bX\b", line) and re.search(r"\bY\b", line) and re.search(r"FWHM", line):
            header_idx = i

    rows = []
    if header_idx is not None:
        for line in lines[header_idx + 1:]:
            line = line.strip()
            if not line:
                continue
            tokens = re.split(r"\s+", line)
            if len(tokens) < 6:
                continue
            try:
                x, y, sigma, fwhm, star_sky, sky_adu = [float(t) for t in tokens[:6]]
            except ValueError:
                continue
            rows.append(dict(x=x, y=y, sigma=sigma, fwhm=fwhm,
                              star_sky=star_sky, sky_adu=sky_adu))

    if not rows:
        raise ValueError(
            f"Could not find any star rows in {path}. Expected an AIP4WIN "
            f"'Star Data Tool' export with an X/Y/Sigma/FWHM/Star-Sky/Sky ADU table.")

    return meta, pd.DataFrame(rows)


# --------------------------------------------------------------------------
# FITS I/O helpers
# --------------------------------------------------------------------------

def read_fits_data(path):
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                return hdu.data.astype(float), hdu.header
    raise ValueError(f"No 2D image data found in {path}")


def get_obs_time_jd(header, path):
    for key in ("JD", "JD-OBS", "JD_OBS"):
        if key in header:
            try:
                return float(header[key])
            except (TypeError, ValueError):
                pass
    for key in ("MJD-OBS", "MJD"):
        if key in header:
            try:
                return float(header[key]) + 2400000.5
            except (TypeError, ValueError):
                pass
    if "DATE-OBS" in header:
        try:
            t = Time(header["DATE-OBS"], format="isot", scale="utc")
            if "EXPTIME" in header:
                t = t + float(header["EXPTIME"]) / 2.0 / 86400.0
            return t.jd
        except Exception:
            pass
    return None


# --------------------------------------------------------------------------
# FWHM measurement (2D Gaussian fit)
# --------------------------------------------------------------------------

def _gauss2d(coords, amp, x0, y0, sx, sy, offset):
    x, y = coords
    return offset + amp * np.exp(-(((x - x0) ** 2) / (2 * sx ** 2) +
                                    ((y - y0) ** 2) / (2 * sy ** 2)))


def measure_fwhm(data, x, y, box_half=15):
    """Fit a 2D Gaussian to the star at (x, y) and return its FWHM in pixels
    (average of the x and y FWHM), or None if the fit fails."""
    h, w = data.shape
    x0i, x1i = int(x - box_half), int(x + box_half)
    y0i, y1i = int(y - box_half), int(y + box_half)
    if x0i < 0 or y0i < 0 or x1i > w or y1i > h:
        return None

    cutout = data[y0i:y1i, x0i:x1i].astype(float)
    if not np.isfinite(cutout).all():
        cutout = np.nan_to_num(cutout, nan=np.nanmedian(cutout))

    yy, xx = np.mgrid[0:cutout.shape[0], 0:cutout.shape[1]]
    _, med, std = sigma_clipped_stats(cutout, sigma=3.0)

    amp0 = cutout.max() - med
    if amp0 <= 0:
        return None

    p0 = (amp0, cutout.shape[1] / 2, cutout.shape[0] / 2, 2.5, 2.5, med)
    try:
        popt, _ = curve_fit(_gauss2d, (xx.ravel(), yy.ravel()), cutout.ravel(),
                             p0=p0, maxfev=5000)
    except Exception:
        return None

    _, _, _, sx, sy, _ = popt
    sx, sy = abs(sx), abs(sy)
    if sx <= 0 or sy <= 0 or sx > box_half or sy > box_half:
        return None

    fwhm_x = 2.3548 * sx
    fwhm_y = 2.3548 * sy
    return (fwhm_x + fwhm_y) / 2.0


def pair_aperture_limit(sep, ratio, fwhm, contam=0.01):
    """Largest aperture that keeps a close companion's light under `contam`.

    The obvious rule - scale the aperture with the seeing, as every other
    star gets - is exactly backwards for a blended pair. A wider PSF smears
    the companion further INTO the aperture, so as the seeing degrades the
    aperture has to shrink, not grow. Left to grow, it swallows more of the
    companion in the frames where the seeing is worst, and the target appears
    to brighten whenever the night softens: measured on the run that exposed
    this, the contamination ran from 7.3% to 14.0% across 123 frames and put
    82 mmag of seeing-shaped error straight into the target's curve - deeper
    than the eclipse that was being looked for.

    Both fractions are exact for a Gaussian PSF. The light of a star inside a
    circle offset from it by `sep` is the non-central chi-square with two
    degrees of freedom, and the star's own light inside its own circle is the
    central case - no integration and no lookup table.

    Returns (r_max_px, flux_fraction_kept_at_r_max). The second number is the
    one that decides whether the frame is worth keeping: when the seeing is
    wide enough that separating the pair costs most of the target's light,
    that frame did not measure this star and should be dropped rather than
    measured and hoped over.
    """
    from scipy.stats import ncx2, chi2
    s = max(float(fwhm), 0.5) / 2.3548
    lam = (float(sep) / s) ** 2

    def contamination(r):
        u = (r / s) ** 2
        f_t = chi2.cdf(u, 2)                      # the target's own light
        f_c = ncx2.cdf(u, 2, lam)                 # the companion's, leaking in
        tot = f_t + ratio * f_c
        return (ratio * f_c / tot) if tot > 0 else 1.0

    lo, hi = 0.3 * s, 3.0 * float(sep)
    if contamination(lo) > contam:                # even a pinhole is polluted
        return lo, float(chi2.cdf((lo / s) ** 2, 2))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if contamination(mid) > contam:
            hi = mid
        else:
            lo = mid
    return lo, float(chi2.cdf((lo / s) ** 2, 2))


def detect_close_pair(image, x, y, fwhm, max_sep_fwhm=3.0,
                      cost_gain=0.7, min_ratio=0.05):
    """Is the target a pair that the star finder cannot split?

    A star finder reports one source per peak and applies shape cuts, so a
    companion closer than about one FWHM is never reported at all: the pair
    arrives as a single, slightly elongated star and the blend check that
    reads that list has nothing to warn about. That is exactly the case the
    blend check most needs to catch. On Gaia DR3 4477867314376415360 a
    companion 1.7 mag fainter sat 4.9 px from the target with a FWHM of
    2.9 px - inside the 5.25 px aperture the run had chosen, contributing
    10.4% of the light - and nothing was reported, because DAOStarFinder had
    delivered the pair as one star.

    So the pair is looked for in the pixels instead. Fit one Gaussian to the
    target, fit two, and accept the second only if it earns its place three
    times over: it has to remove a real share of the residual, sit far enough
    out that an aperture can be steered around it, and carry enough light to
    matter. On the frame above the three tests separate cleanly - the target
    scores 0.50 / 4.91 px / 0.207 while the worst of twelve single comparison
    stars scores 0.79 / 1.97 px, the fitter merely splitting one PSF in half.

    Needs a high signal-to-noise image. On one 30 s colour plane the same
    target scores 0.88 and is missed; on the three planes summed it is found.
    Give it the deepest image available, and read a null as "not found",
    never as "not there". Returns (separation_px, delta_mag) or None.
    """
    from scipy.optimize import least_squares
    R = int(max(6, round(3.0 * fwhm)))
    xi, yi = int(round(x)), int(round(y))
    if (xi - R < 0 or yi - R < 0
            or yi + R + 1 > image.shape[0] or xi + R + 1 > image.shape[1]):
        return None
    cut = image[yi - R:yi + R + 1, xi - R:xi + R + 1].astype(float)
    if cut.size == 0 or not np.all(np.isfinite(cut)):
        return None
    yy, xx = np.mgrid[0:cut.shape[0], 0:cut.shape[1]]
    s0 = max(float(fwhm), 1.5) / 2.3548
    bg0 = float(np.median(cut))
    a0 = float(np.max(cut)) - bg0
    if not np.isfinite(a0) or a0 <= 0:
        return None
    cx0, cy0 = x - (xi - R), y - (yi - R)

    def one(p):
        a, px, py, s, bg = p
        return a * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * s * s)) + bg

    def two(p):
        a1, x1, y1, a2, x2, y2, s, bg = p
        return (a1 * np.exp(-((xx - x1) ** 2 + (yy - y1) ** 2) / (2 * s * s))
                + a2 * np.exp(-((xx - x2) ** 2 + (yy - y2) ** 2) / (2 * s * s))
                + bg)

    try:
        r1 = least_squares(lambda p: (one(p) - cut).ravel(),
                           [a0, cx0, cy0, s0, bg0], max_nfev=4000)
        # the second component is started from several directions - a single
        # start lands in a local minimum that depends on which way the pair
        # happens to lie on the chip
        best = None
        for dx, dy in ((1.2 * fwhm, 0.0), (-1.2 * fwhm, 0.0),
                       (0.0, 1.2 * fwhm), (0.0, -1.2 * fwhm),
                       (0.9 * fwhm, 0.9 * fwhm), (-0.9 * fwhm, -0.9 * fwhm)):
            r2 = least_squares(
                lambda p: (two(p) - cut).ravel(),
                [a0, cx0, cy0, 0.3 * a0, cx0 + dx, cy0 + dy, s0, bg0],
                max_nfev=6000)
            if best is None or r2.cost < best.cost:
                best = r2
    except Exception:
        return None
    if best is None or not np.isfinite(best.cost) or r1.cost <= 0:
        return None

    a1, x1, y1, a2, x2, y2, _s, _bg = best.x
    a1, a2 = abs(a1), abs(a2)
    if a1 < a2:                       # the brighter component is the target
        a1, a2, x1, y1, x2, y2 = a2, a1, x2, y2, x1, y1
    if a1 <= 0:
        return None
    sep = float(np.hypot(x1 - x2, y1 - y2))
    ratio = a2 / a1                   # one width for both, so this is the flux ratio
    if best.cost > cost_gain * r1.cost:
        return None
    if not (0.5 * fwhm < sep < max_sep_fwhm * fwhm):
        return None
    if ratio < min_ratio:
        return None
    return sep, float(-2.5 * np.log10(ratio))


def measure_reference_fwhm(ref_data, stars, names, box_half=15):
    """Measure FWHM for a list of star names on the reference frame and
    return (per_star_dict, median_fwhm)."""
    results = {}
    for name in names:
        row = stars[stars.name == name]
        if row.empty:
            continue
        x, y = float(row.iloc[0].x), float(row.iloc[0].y)
        fwhm = measure_fwhm(ref_data, x, y, box_half=box_half)
        results[name] = fwhm

    valid = [v for v in results.values() if v is not None]
    median_fwhm = float(np.median(valid)) if valid else None
    return results, median_fwhm


# --------------------------------------------------------------------------
# Star (re-)location between frames
# --------------------------------------------------------------------------

ORIENTATIONS = {
    # name: (function to transform a 2D array, sign applied to (rel_x, rel_y)
    #        of every OTHER star relative to the anchor, once this
    #        orientation is detected)
    "normal":  (lambda a: a,               (1, 1)),
    "flip_ud": (lambda a: a[::-1, :],      (1, -1)),   # image flipped top/bottom
    "flip_lr": (lambda a: a[:, ::-1],      (-1, 1)),   # image flipped left/right
    "rot180":  (lambda a: a[::-1, ::-1],   (-1, -1)),  # both (typical meridian flip)
}


def find_anchor_shift(ref_data, new_data, ax, ay, stamp_half=40, search_margin=100,
                       detect_flips=True, min_corr=0.3, flip_margin=0.08):
    """Locate the anchor star (and, if present, its nearby neighbours) in
    `new_data`. The stamp is deliberately large enough to include a few
    neighbouring stars so the pattern is NOT point-symmetric - a lone star's
    profile looks identical whether the image is flipped or not, so
    orientation can only be detected once the template includes some
    asymmetric context around it.

    To avoid noise-driven false positives (a flipped/rotated orientation
    winning by a hair due to noise rather than a real flip - which happens
    easily when the stamp contains little or no asymmetric context, e.g. a
    sparse field), a non-'normal' orientation is only accepted if its
    correlation peak beats 'normal's peak by at least `flip_margin`.
    Otherwise 'normal' (no flip) is used, since it is the far more common
    case and a tie/near-tie gives no real evidence of an actual flip.

    Returns (dx, dy, peak, orientation) where orientation is one of the keys
    of ORIENTATIONS ('normal' if detect_flips=False), or
    (None, None, None, None) if no reliable match was found.
    """
    h, w = ref_data.shape
    hn, wn = new_data.shape

    x0s, x1s = int(ax - stamp_half), int(ax + stamp_half)
    y0s, y1s = int(ay - stamp_half), int(ay + stamp_half)
    if x0s < 0 or y0s < 0 or x1s > w or y1s > h:
        return None, None, None, None
    stamp = ref_data[y0s:y1s, x0s:x1s]

    x0 = max(0, int(ax - stamp_half - search_margin))
    x1 = min(wn, int(ax + stamp_half + search_margin))
    y0 = max(0, int(ay - stamp_half - search_margin))
    y1 = min(hn, int(ay + stamp_half + search_margin))
    search = new_data[y0:y1, x0:x1]

    if search.shape[0] <= stamp.shape[0] or search.shape[1] <= stamp.shape[1]:
        return None, None, None, None

    orientations_to_try = ORIENTATIONS if detect_flips else {"normal": ORIENTATIONS["normal"]}

    results = {}  # name -> (peak, ix, iy)
    for name, (transform, _sign) in orientations_to_try.items():
        t_stamp = transform(stamp)
        result = match_template(search, t_stamp)
        iy, ix = np.unravel_index(np.argmax(result), result.shape)
        results[name] = (float(result[iy, ix]), ix, iy)

    best_name = max(results, key=lambda k: results[k][0])
    peak, ix, iy = results[best_name]

    # require a clear margin over 'normal' before trusting a flip
    if best_name != "normal" and "normal" in results:
        normal_peak = results["normal"][0]
        if peak - normal_peak < flip_margin:
            best_name = "normal"
            peak, ix, iy = results["normal"]

    orientation = best_name
    if peak < min_corr:
        return None, None, None, None

    found_ax = x0 + ix + stamp_half
    found_ay = y0 + iy + stamp_half
    return found_ax - ax, found_ay - ay, peak, orientation


def refine_centroid(data, x_guess, y_guess, box_half=8):
    h, w = data.shape
    x0, x1 = int(round(x_guess - box_half)), int(round(x_guess + box_half))
    y0, y1 = int(round(y_guess - box_half)), int(round(y_guess + box_half))
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 <= x0 or y1 <= y0:
        return x_guess, y_guess, False

    cutout = data[y0:y1, x0:x1].copy()
    if not np.isfinite(cutout).all():
        cutout = np.nan_to_num(cutout, nan=np.nanmedian(cutout))

    _, med, _ = sigma_clipped_stats(cutout, sigma=3.0)
    cutout = cutout - med
    cutout[cutout < 0] = 0
    if cutout.sum() <= 0:
        return x_guess, y_guess, False

    try:
        cx, cy = centroid_com(cutout)
    except Exception:
        return x_guess, y_guess, False

    if not (np.isfinite(cx) and np.isfinite(cy)):
        return x_guess, y_guess, False

    new_x, new_y = x0 + cx, y0 + cy
    if abs(new_x - x_guess) > box_half or abs(new_y - y_guess) > box_half:
        return x_guess, y_guess, False

    return new_x, new_y, True


# --------------------------------------------------------------------------
# Photometry (aperture method)
# --------------------------------------------------------------------------

def measure_flux(data, x, y, r_aper, r_in, r_out, gain=1.0, ron=0.0, dark=0.0):
    """Aperture photometry with local sky-annulus background subtraction,
    using the full CCD noise equation (Newberry 1991 / Merline & Howell
    1995, as used in AIP4Win / Berry & Burnell's Handbook of Astronomical
    Image Processing) for the flux uncertainty:

        SNR = S* / sqrt[ S* + n_ap*(1 + n_ap/n_sky)*(S_sky + S_dark + RON^2 + (gain/2)^2) ]

    where S* is the target's signal (electrons, sky-subtracted), S_sky and
    S_dark are the sky and dark levels per pixel (electrons), RON is the
    read noise (electrons), and (gain/2)^2 is the ADU quantization term.
    `gain` is in electrons per ADU; `ron` in electrons; `dark` in ADU/pixel
    (matching the AIP4Win convention where dark is already subtracted from
    the reduced frame, so it defaults to 0).

    All ADU quantities (raw_sum, sky_per_pix, dark) are converted to
    electrons via `gain` before combining. The returned flux/flux_err stay
    in ADU (electrons / gain) so the rest of the pipeline is unaffected.
    """
    position = (x, y)
    aperture = CircularAperture(position, r=r_aper)
    annulus = CircularAnnulus(position, r_in=r_in, r_out=r_out)

    ap_stats = ApertureStats(data, aperture)

    ann_mask = annulus.to_mask(method="center")
    if isinstance(ann_mask, (list, tuple)):
        ann_mask = ann_mask[0]
    ann_data = ann_mask.multiply(data)
    if ann_data is None:
        # no overlap at all between the annulus and the frame (e.g. star too
        # close to, or off, the image edge) - can't estimate a local sky
        sky_per_pix = 0.0
    else:
        ann_data = ann_data[ann_mask.data > 0]
        if ann_data.size == 0 or not np.isfinite(ann_data).any():
            sky_per_pix = 0.0
        else:
            _, sky_per_pix, _ = sigma_clipped_stats(ann_data, sigma=3.0)

    raw_sum = float(np.atleast_1d(ap_stats.sum)[0])
    n_pix = float(np.atleast_1d(aperture.area)[0])
    n_sky = float(np.atleast_1d(annulus.area)[0])

    flux = raw_sum - sky_per_pix * n_pix  # ADU, sky-subtracted (unchanged)

    # --- CCD equation, in electrons ---
    star_e = max(flux, 0.0) * gain
    sky_e = max(sky_per_pix, 0.0) * gain
    dark_e = max(dark, 0.0) * gain
    ron_term = ron ** 2
    quant_term = (gain / 2.0) ** 2
    bg_factor = (1.0 + n_pix / n_sky) if n_sky > 0 else 1.0

    variance_e2 = star_e + n_pix * bg_factor * (sky_e + dark_e + ron_term + quant_term)
    flux_err_e = np.sqrt(max(variance_e2, 0.0))
    flux_err = flux_err_e / gain if gain > 0 else flux_err_e  # back to ADU

    return float(flux), float(flux_err), float(sky_per_pix)


# --------------------------------------------------------------------------
# Interactive configuration (FWHM multiples + star selection)
# --------------------------------------------------------------------------

import os as _os_for_uri  # (kept local-name safe; os is already imported above)
from urllib.parse import urlparse, unquote


def _clean_file_uri(raw):
    """If the user pasted a file:// URI (e.g. copied from a browser address
    bar or some file managers), turn it back into a normal OS path.
    'file:///F:/folder/name with spaces/x.fit' -> 'F:/folder/name with spaces/x.fit'
    Leaves ordinary paths untouched.
    """
    if raw.lower().startswith("file:"):
        parsed = urlparse(raw)
        path = unquote(parsed.path)
        # On Windows, urlparse turns 'file:///F:/x' into path='/F:/x' -> drop the leading slash
        if len(path) > 2 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return path
    return raw


def prompt_path(msg, default=None, must_exist=None):
    r"""Ask the user for a path the same way your reduction script does:
        raw = input("Enter path: ").strip().strip('"')
    This also copes with the other common ways people paste a path:
      - wrapped in double quotes:  "C:\Data\WASP-02"
      - wrapped in single quotes: 'C:\Data\WASP-02'
      - trailing slash/backslash: C:\Data\WASP-02\
      - forward or back slashes, mixed
      - a relative path (left as-is; resolved against the current folder)
    must_exist: None -> no check, "dir" -> must be an existing folder,
                "file" -> must be an existing file.
    Re-prompts until a usable path is given (or the default is accepted).
    """
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            raw = input(f"{msg}{suffix}: ")
        except EOFError:
            raw = ""
        raw = raw.strip().strip('"').strip("'").strip()
        if raw == "" and default:
            raw = default
        raw = _clean_file_uri(raw)
        # normalize accidental trailing slash/backslash and mixed separators
        raw = raw.rstrip("/\\")
        raw = os.path.normpath(raw) if raw else raw

        if raw == "":
            print("  ! path cannot be empty, try again")
            continue
        if must_exist == "dir" and not os.path.isdir(raw):
            print(f"  ! folder not found: {raw}  -> try again")
            continue
        if must_exist == "file" and not os.path.isfile(raw):
            print(f"  ! file not found: {raw}  -> try again")
            continue
        return raw


def default_output_dir(input_dir, ref_file):
    """Suggest '<star name>-results' inside the input folder.

    The star name is taken from the reference frame's OBJECT keyword, which is
    what the telescope actually recorded for the target, so the folder is named
    after the star rather than after whatever the frames happen to be called.
    Falls back to the input folder's own name when OBJECT is missing or blank -
    that happens with frames whose header was stripped during reduction.
    Characters Windows forbids in a folder name are replaced with '_'.
    """
    name = ""
    try:
        with fits.open(ref_file) as hdul:
            for hdu in hdul:
                if hdu.header.get("OBJECT"):
                    name = str(hdu.header["OBJECT"]).strip()
                    break
    except Exception:
        name = ""
    if not name:
        name = os.path.basename(os.path.normpath(input_dir))
    for ch in '\\/:*?"<>|':
        name = name.replace(ch, "_")
    # Spaces become underscores too: a path without spaces is easier to pass to
    # other tools and does not need quoting on the command line.
    name = "_".join(name.split()).rstrip(". ")
    if not name:
        name = "target"
    return os.path.join(input_dir, f"{name}_results")


def prompt_float(msg, default):
    try:
        raw = input(f"{msg} [{default}]: ").strip()
    except EOFError:
        raw = ""
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print("  ! not a number, using default")
        return default


def configure_from_csv(ref_data, stars, args):
    """Handles: (1) which comparison stars to use, (2) FWHM measurement,
    (3) aperture / annulus radii as multiples of FWHM. Any value already
    given on the command line skips its prompt (for unattended/batch runs).
    Used for the simple name,x,y,role CSV star-list format."""

    all_comp_names = list(stars[stars.role == "comp"].name)
    target_name = stars[stars.role == "target"].iloc[0]["name"]

    # --- 1. which comparison stars to use ---
    if args.stars:
        chosen = [s.strip() for s in args.stars.split(",") if s.strip()]
        unknown = [s for s in chosen if s not in all_comp_names]
        if unknown:
            raise ValueError(f"Unknown comparison star name(s): {unknown}. "
                              f"Available: {all_comp_names}")
    else:
        print(f"\nAvailable comparison stars in {args.star_list}: {', '.join(all_comp_names)}")
        raw = input("Which comparison stars should be used? "
                    "(comma-separated names, or Enter for ALL): ").strip()
        chosen = all_comp_names if raw == "" else [s.strip() for s in raw.split(",") if s.strip()]

    if args.anchor:
        anchor_name = args.anchor
    else:
        anchor_name = chosen[0]
    if anchor_name not in all_comp_names:
        raise ValueError(f"Anchor star '{anchor_name}' is not in the star list")

    measure_names = list(dict.fromkeys([target_name] + chosen + [anchor_name]))

    # --- 2. measure FWHM on the reference frame ---
    print("\nMeasuring FWHM (2D Gaussian fit) on the reference frame...")
    per_star_fwhm, median_fwhm = measure_reference_fwhm(
        ref_data, stars, measure_names, box_half=args.fwhm_box)

    for name, val in per_star_fwhm.items():
        txt = f"{val:.2f} px" if val is not None else "FIT FAILED"
        print(f"   {name:6s} FWHM = {txt}")

    if median_fwhm is None:
        print("Could not measure FWHM automatically for any star.")
        median_fwhm = prompt_float("Enter FWHM manually (pixels)", 3.0)
    else:
        print(f"-> Median FWHM = {median_fwhm:.2f} px  (used to size the aperture)")
        if args.fwhm_override is not None:
            median_fwhm = args.fwhm_override
            print(f"   (overridden by --fwhm_override = {median_fwhm:.2f} px)")

    # Only this path has the reference image, so only this path can look for
    # a companion in the pixels. The Star Data Tool path takes --pair_sep_px.
    pair = None
    if getattr(args, "pair_cap", True):
        try:
            trow = stars[stars.name == target_name].iloc[0]
            comp_fwhm = [v for k, v in per_star_fwhm.items()
                         if k != target_name and v is not None
                         and 1.5 < v < 30]
            pf = float(np.median(comp_fwhm)) if comp_fwhm else median_fwhm
            pair = detect_close_pair(ref_data, float(trow.x), float(trow.y), pf)
            if pair is not None:
                print("   companion found %.2f px away, %.2f mag fainter"
                      % (pair[0], pair[1]))
        except Exception as e:
            print("   companion check skipped: " + str(e))

    config = _prompt_aperture_radii(median_fwhm, args, pair=pair)
    config.update({
        "target_name": target_name,
        "chosen_comps": chosen,
        "anchor_name": anchor_name,
        "per_star_fwhm": per_star_fwhm,
        "known_mags": get_known_mags(chosen, args),
    })
    return stars, config


def configure_from_startool(rows_df, args, meta):
    """Handles the AIP4WIN 'Star Data Tool' export format: no names/roles
    exist yet, only a numbered list of detected stars with measured X, Y,
    Sigma, FWHM, Star-Sky, Sky ADU. Lets the user pick, by row number, which
    detected star is the target and which are the comparison stars, then
    builds a synthetic name/x/y/role table (V, C1, C2, ...) and reuses the
    FWHM values AIP4WIN already measured (no re-fitting needed)."""

    n = len(rows_df)
    print(f"\n{n} stars detected in the Star Data Tool file"
          + (f" (image: {meta['image']})" if "image" in meta else "") + ":\n")
    print(f"{'idx':>4} {'X':>10} {'Y':>10} {'FWHM':>7} {'Sigma':>7} "
          f"{'Star-Sky':>12} {'Sky ADU':>9}")
    for i, r in rows_df.iterrows():
        print(f"{i+1:>4} {r.x:>10.3f} {r.y:>10.3f} {r.fwhm:>7.2f} {r.sigma:>7.2f} "
              f"{r.star_sky:>12.1f} {r.sky_adu:>9.3f}")
    if {"aperture_radius", "ann_in", "ann_out"} <= meta.keys():
        print(f"\n(For reference, AIP4WIN itself used aperture={meta['aperture_radius']}, "
              f"annulus={meta['ann_in']}-{meta['ann_out']} px when this file was produced.)")

    def _parse_indices(raw, all_indices):
        raw = raw.strip().lower()
        if raw in ("all", ""):
            return list(all_indices)
        out = []
        for tok in raw.split(","):
            tok = tok.strip()
            if tok:
                out.append(int(tok))
        return out

    all_idx = list(range(1, n + 1))

    # --- target: always row 1, comparisons: always all the rest ---
    # (fixed convention - no prompt needed every time)
    target_idx = args.target_index if args.target_index is not None else 1
    remaining = [i for i in all_idx if i != target_idx]
    if args.comp_indices is not None:
        comp_idx = _parse_indices(args.comp_indices, remaining)
    else:
        comp_idx = remaining
    print(f"\nUsing row {target_idx} as the TARGET star, and the remaining "
          f"{len(comp_idx)} rows as comparison stars (C1..C{len(comp_idx)}).")

    # --- build the synthetic star list (V, C1, C2, ...) ---
    target_row = rows_df.iloc[target_idx - 1]
    names = ["V"] + [f"C{k}" for k in range(1, len(comp_idx) + 1)]
    xs = [target_row.x] + [rows_df.iloc[i - 1].x for i in comp_idx]
    ys = [target_row.y] + [rows_df.iloc[i - 1].y for i in comp_idx]
    roles = ["target"] + ["comp"] * len(comp_idx)
    fwhms = [target_row.fwhm] + [rows_df.iloc[i - 1].fwhm for i in comp_idx]

    stars = pd.DataFrame({"name": names, "x": xs, "y": ys, "role": roles})
    per_star_fwhm = dict(zip(names, fwhms))

    # --- anchor (tracking) star ---
    comp_names = names[1:]
    if args.anchor:
        anchor_name = args.anchor
        if anchor_name not in comp_names:
            raise ValueError(f"--anchor '{anchor_name}' is not one of the chosen "
                              f"comparison stars {comp_names}")
    elif args.anchor_index is not None:
        pos = comp_idx.index(args.anchor_index) if args.anchor_index in comp_idx else None
        if pos is None:
            raise ValueError(f"--anchor_index {args.anchor_index} is not one of the "
                              f"chosen comparison row numbers {comp_idx}")
        anchor_name = comp_names[pos]
    else:
        anchor_name = comp_names[0]
        print(f"(Using {anchor_name} = row {comp_idx[0]} as the tracking anchor star; "
              f"override with --anchor_index)")

    median_fwhm = float(np.median(fwhms))
    print(f"\n-> Median FWHM (from the Star Data Tool measurements of the "
          f"chosen stars) = {median_fwhm:.2f} px")
    if args.fwhm_override is not None:
        median_fwhm = args.fwhm_override
        print(f"   (overridden by --fwhm_override = {median_fwhm:.2f} px)")

    config = _prompt_aperture_radii(median_fwhm, args)
    config.update({
        "target_name": "V",
        "chosen_comps": comp_names,
        "anchor_name": anchor_name,
        "per_star_fwhm": per_star_fwhm,
        "known_mags": get_known_mags(comp_names, args),
    })
    return stars, config


def load_known_mags_csv(path):
    """CSV with columns name,mag - the known/catalog magnitude of each
    comparison star, used for absolute ensemble magnitude calibration."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "name" not in cols or "mag" not in cols:
        raise ValueError("known_mags CSV must have columns: name, mag "
                          f"(got {list(df.columns)})")
    return dict(zip(df[cols["name"]], df[cols["mag"]].astype(float)))


def prompt_known_mags_manual(comp_names):
    mags = {}
    print("Enter the known (catalog) magnitude for each comparison star "
          "(press Enter to skip a star - it just won't be used for calibration):")
    for name in comp_names:
        raw = input(f"   {name}: ").strip()
        if raw == "":
            continue
        try:
            mags[name] = float(raw)
        except ValueError:
            print("     ! not a number, skipping this star")
    return mags


def get_known_mags(comp_names, args):
    """Returns a dict {comp_star_name: known_catalog_magnitude}, used to add
    an absolute-calibration magnitude column for the target (in addition to
    the purely relative/differential columns). Returns {} if the user does
    not want this (nothing is added to the output in that case)."""
    if args.known_mags:
        mags = load_known_mags_csv(args.known_mags)
        print(f"Loaded {len(mags)} known magnitudes from {args.known_mags}")
        return {k: v for k, v in mags.items() if k in comp_names}

    if args.skip_calibration:
        return {}

    ans = input("\nAdd an absolute-calibrated magnitude for the target using known "
                "catalog magnitudes of the comparison stars? [y/N]: ").strip().lower()
    if ans not in ("y", "yes"):
        return {}

    raw = input("Enter path to a CSV with columns name,mag "
                "(or press Enter to type the values in manually): ")
    raw = raw.strip().strip('"').strip("'").strip()
    if raw:
        raw = _clean_file_uri(raw)
        raw = os.path.normpath(raw.rstrip("/\\"))
        if os.path.isfile(raw):
            mags = load_known_mags_csv(raw)
            return {k: v for k, v in mags.items() if k in comp_names}
        print(f"  ! file not found: {raw} -- falling back to manual entry")

    return prompt_known_mags_manual(comp_names)


def _prompt_aperture_radii(median_fwhm, args, pair=None):
    """Shared step 3: aperture / annulus radii as multiples of FWHM."""
    k_aperture = args.k_aperture if args.k_aperture is not None else \
        prompt_float("Aperture radius = k x FWHM -> enter k", 3.0)
    k_ann_in = args.k_ann_in if args.k_ann_in is not None else \
        prompt_float("Sky annulus INNER radius = k x FWHM -> enter k", 4.0)
    k_ann_out = args.k_ann_out if args.k_ann_out is not None else \
        prompt_float("Sky annulus OUTER radius = k x FWHM -> enter k", 6.0)

    r_aperture = k_aperture * median_fwhm
    r_ann_in = k_ann_in * median_fwhm
    r_ann_out = k_ann_out * median_fwhm

    # A close companion caps the aperture here too, but the shape of the fix
    # differs from the plate-solved scripts, because this one measures every
    # frame at ONE radius fixed for the whole run instead of resizing it with
    # the seeing. A fixed radius admits a roughly constant share of the
    # companion, which dilutes an eclipse by a constant factor; it does not
    # write the seeing-shaped trend into the curve that a radius growing with
    # the seeing does. So the cap is applied once, at the median seeing, and
    # there is no frame to drop.
    pair_note = ""
    sep = ratio = None
    if getattr(args, "pair_sep_px", None):
        sep, ratio = float(args.pair_sep_px), 0.2
    if pair is not None:
        sep, ratio = pair[0], 10 ** (-0.4 * pair[1])
    if sep and getattr(args, "pair_cap", True):
        contam = getattr(args, "pair_contam", 0.01)
        r_max, kept = pair_aperture_limit(sep, ratio, median_fwhm, contam)
        if r_max < r_aperture:
            pair_note = (f"companion {sep:.2f} px away - aperture capped at "
                         f"{r_max:.2f} px to hold its light under "
                         f"{contam * 100:.0f}% "
                         f"({kept * 100:.0f}% of the target's light kept)")
            print("==> " + pair_note)
            r_aperture = r_max

    print(f"\n==> aperture radius = {r_aperture:.2f} px "
          f"({k_aperture} x FWHM={median_fwhm:.2f})")
    print(f"==> sky annulus     = {r_ann_in:.2f} - {r_ann_out:.2f} px "
          f"({k_ann_in} x / {k_ann_out} x FWHM)\n")

    return {
        "median_fwhm": median_fwhm,
        "k_aperture": k_aperture,
        "k_ann_in": k_ann_in,
        "k_ann_out": k_ann_out,
        "r_aperture": r_aperture,
        "r_ann_in": r_ann_in,
        "r_ann_out": r_ann_out,
        "pair_note": pair_note,
    }


# --------------------------------------------------------------------------
# Main measurement loop
# --------------------------------------------------------------------------

def estimate_adaptive_stamp_half(stars, anchor_name, median_fwhm=None, fwhm_factor=8.0,
                                  margin_factor=1.5, min_half=20, max_half=150):
    """Choose a stamp size (half-width, px) for the anchor template.

    Sized primarily from the star's own FWHM (fwhm_factor x FWHM), the same
    way the aperture/annulus radii already are elsewhere in this script.
    This matters because the stamp is matched against a new frame with
    normalized cross-correlation: a stamp much larger than the star just
    adds background sky (uniform noise, no structure) that dilutes the
    correlation peak, and can push a perfectly-good match below min_corr
    even though the star was found at the right place. A stamp sized to
    the star's own width keeps most of its area as actual signal.

    Falls back to the nearest-neighbour distance (old behaviour) only when
    the FWHM isn't known yet. Either way the result is clipped to
    [min_half, max_half], and the nearest-neighbour distance is still
    returned/checked by the caller to decide whether the stamp happens to
    reach a neighbour (needed for flip/rotation detection) - if it
    doesn't, flip detection is safely disabled elsewhere rather than
    trusting a lone, point-symmetric star.

    Returns (stamp_half, nearest_dist)."""
    anchor_row = stars[stars.name == anchor_name].iloc[0]
    ax, ay = float(anchor_row.x), float(anchor_row.y)
    others = stars[stars.name != anchor_name]
    nearest = None
    if not others.empty:
        dists = np.sqrt((others.x - ax) ** 2 + (others.y - ay) ** 2)
        nearest = float(dists.min())

    if median_fwhm is not None and median_fwhm > 0:
        stamp_half = float(np.clip(fwhm_factor * median_fwhm, min_half, max_half))
    elif nearest is not None:
        stamp_half = float(np.clip(nearest * margin_factor, min_half, max_half))
    else:
        stamp_half = 40.0
    return stamp_half, nearest


def detect_main_segment(df, gap_minutes=10):
    """Returns a boolean mask marking the 'main' contiguous observing run -
    the longest stretch of frames with no time gap larger than
    `gap_minutes` inside it. Any cluster of frames separated from that
    stretch by such a gap (a re-pointing or a restart after a long pause,
    or a stray test exposure taken before the run proper) is excluded from
    the std/drift statistics, and checked separately as 'tail_offset'.
    If there's no such gap, everything counts as 'main'.

    The longest stretch is chosen rather than 'everything before the last
    gap', which is what this used to do. That older rule assumed the odd
    cluster always sits at the END of the night. When it sits at the
    BEGINNING instead - V1111 Cep opened with a single test frame taken 32
    minutes before the real series - the rule kept only that one frame as
    'main', a scatter cannot be computed from a single point, and every
    comparison star came back unmeasurable.
    """
    if "elapsed_min" not in df.columns or not df["elapsed_min"].notna().any():
        return pd.Series(True, index=df.index)
    times = df["elapsed_min"]
    valid = times.dropna().sort_values()
    if len(valid) < 2:
        return pd.Series(True, index=df.index)

    gaps = valid.diff()
    big = gaps > gap_minutes
    if not big.any():
        return pd.Series(True, index=df.index)

    # cut the run into segments at every big gap, then keep the longest one
    seg_id = big.cumsum()
    counts = seg_id.value_counts()
    best = counts.index[0]
    members = valid[seg_id == best]
    lo, hi = float(members.iloc[0]), float(members.iloc[-1])

    n_out = int(len(valid) - len(members))
    if n_out:
        print(f"   (main observing block: {len(members)} frame(s) between "
              f"{lo:.1f} and {hi:.1f} min; {n_out} frame(s) outside it are "
              f"excluded from the std/drift statistics)")
    return ((times >= lo) & (times <= hi)) | times.isna()


def find_baseline_segment(values, min_baseline_points=5, threshold_k=2.0):
    """Finds the initial 'flat' baseline segment of a time-ordered series
    (e.g. a comparison star's magnitude difference from the ensemble
    median) - the stretch at the START before a clear, sustained deviation
    begins (e.g. before a transit-like dip/rise, or a systematic drift). A
    blind 'first 20%' can be wrong if the deviation starts early or late;
    this instead walks forward from the start and stops at the first
    sustained departure from the initial level (more than `threshold_k`
    times the initial point-to-point scatter, over a small window, so a
    single noisy point doesn't trigger it).

    Returns (baseline_level, end_index) - end_index is where the baseline
    segment ends (exclusive)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < min_baseline_points * 2:
        return (float(np.median(values)) if n else np.nan), n

    initial_chunk = values[:min_baseline_points]
    baseline_level = float(np.median(initial_chunk))
    local_mad = float(np.median(np.abs(initial_chunk - baseline_level))) * 1.4826
    local_mad = max(local_mad, 1e-6)

    window = max(3, min_baseline_points // 2)
    end_idx = n
    for i in range(min_baseline_points, n - window + 1):
        chunk = values[i:i + window]
        if np.median(np.abs(chunk - baseline_level)) > threshold_k * local_mad:
            end_idx = i
            break

    baseline_level = float(np.median(values[:end_idx])) if end_idx > 0 else baseline_level
    return baseline_level, end_idx


def evaluate_completeness(df, comp_names, target_name, min_completeness=0.95):
    """For each comparison star, computes what fraction of the frames where
    the TARGET was successfully measured also have a successful measurement
    of this comparison star. The denominator is frames where the target
    succeeded (not all frames in the folder) - a frame where the target
    itself failed indicates a general problem with that frame/observing
    conditions (clouds, tracking, bad seeing, etc.), not something specific
    to any one comparison star, so it isn't held against comparison stars
    either way.

    Returns (completeness, failed) where completeness is
    {name: fraction 0-1} and failed is the list of names below
    min_completeness - these should be hard-rejected regardless of how
    stable they look in the frames where they DO appear, since that subset
    may be a biased sample of only their best moments (e.g. a star that's
    marginally resolved/near the detection threshold will disproportionately
    drop out exactly when conditions are less than perfect)."""
    target_col = f"{target_name}_ok"
    target_ok = df[target_col] if target_col in df.columns else pd.Series(True, index=df.index)
    denom = int(target_ok.sum())

    completeness = {}
    failed = []
    for c in comp_names:
        col = f"{c}_ok"
        if col not in df.columns or denom == 0:
            completeness[c] = 0.0
            failed.append(c)
            continue
        num = int((df[col] & target_ok).sum())
        pct = num / denom
        completeness[c] = pct
        if pct < min_completeness:
            failed.append(c)
    return completeness, failed


def evaluate_comp_stability(df, comp_names, main_mask=None):
    """For each comparison star, computes THREE things, using its
    instrumental magnitude relative to the median of the OTHER comparison
    stars (this cancels out real, shared sky-transparency/airmass changes,
    leaving each star's own behaviour):

      - std: point-to-point scatter over the 'main' segment (catches noisy/
        unstable measurements)
      - drift: the star's level at the END of the main segment (its last
        ~20%) minus its BASELINE level - the genuinely flat stretch at the
        very start, found by find_baseline_segment() rather than a blind
        first-20% (which can be wrong if a dip/rise starts early or late).
        Catches a star that doesn't return to where it started.
      - tail_offset: if there's a trailing anomalous cluster (frames
        excluded from 'main' by a long gap - see detect_main_segment), its
        mean level minus that SAME baseline. A good comparison star should
        sit at the same baseline even after a gap/re-pointing; NaN if there
        is no such cluster, or this star has no valid data in it.

    All three matter: a low std with a large drift or tail_offset (a smooth
    trend, or a cluster that never comes back to baseline) is just as bad a
    comparison star as a high-std/noisy one.
    """
    if main_mask is None:
        main_mask = pd.Series(True, index=df.index)
    tail_mask = ~main_mask

    stability = {}
    drift = {}
    tail_offset = {}
    for c in comp_names:
        others = [o for o in comp_names if o != c]
        if not others:
            stability[c] = np.nan
            drift[c] = np.nan
            tail_offset[c] = np.nan
            continue
        other_mags = df[[f"{o}_mag" for o in others]]
        median_others = other_mags.median(axis=1, skipna=True)
        full_diff = df[f"{c}_mag"] - median_others
        main_diff = full_diff[main_mask].dropna()

        stability[c] = float(main_diff.std()) if len(main_diff) > 1 else np.nan

        baseline_level, base_end = find_baseline_segment(main_diff.values)
        n = len(main_diff)
        if n >= 6 and np.isfinite(baseline_level):
            k = max(1, n // 5)  # last ~20% - should have "returned to baseline" by now
            end_val = float(main_diff.iloc[-k:].mean())
            drift[c] = end_val - baseline_level
        else:
            drift[c] = np.nan

        tail_diff = full_diff[tail_mask].dropna()
        if len(tail_diff) > 0 and np.isfinite(baseline_level):
            tail_offset[c] = float(tail_diff.mean() - baseline_level)
        else:
            tail_offset[c] = np.nan

    return stability, drift, tail_offset


def _p2p_noise(values):
    """Short-timescale noise, from the difference between consecutive points.

    A slow drift barely changes how much NEIGHBOURING points differ, so this
    measures noise while ignoring trend - which is exactly what is needed to
    ask "is this star's trend big compared with its own noise?". Uses the
    median absolute difference rather than the standard deviation so a few
    outlying frames cannot inflate it."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 6:
        return np.nan
    d = np.abs(np.diff(v))
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    return float(np.median(d) * 1.4826 / np.sqrt(2.0))


def _running_median(values, window):
    """Running median. Kills white noise, keeps any smooth trend intact."""
    v = np.asarray(values, float)
    n = len(v)
    half = max(1, int(window) // 2)
    out = np.empty(n, float)
    for i in range(n):
        seg = v[max(0, i - half):min(n, i + half + 1)]
        seg = seg[np.isfinite(seg)]
        out[i] = np.median(seg) if len(seg) else np.nan
    return out


def evaluate_comp_trend(df, comp_names, main_mask=None, smooth_points=15):
    """For each comparison star: how large its SMOOTH trend is compared with
    its own short-timescale noise.

    Why this exists next to evaluate_comp_stability(): that function ranks
    stars by plain standard deviation, and a standard deviation adds noise
    and trend in quadrature. A star that slides 30 mmag across the run while
    carrying 7 mmag of noise lands at sqrt(7^2 + trend_contribution^2) - only
    a little above a star that is merely noisy - so the two failure modes
    become indistinguishable, and a ratio-gap detector over such values finds
    no gap to cut at.

    Measured on WASP-52, 24/8/2026 (250 frames): the ranking by standard
    deviation put the WORST star (12.35 mmag) at a star that was simply
    noisy, while the star actually corrupting the ensemble - a non-pointlike
    source sliding 32 mmag with the seeing - sat tenth. By trend/noise the
    separation is clean: 4.55 and 2.34 for the two bad stars, 1.16 to 1.73
    for the other ten.

    Returns {name: ratio}. A constant star sits near 1: its smoothed curve is
    flat to within the noise. NaN when the star has too few valid points.
    """
    if main_mask is None:
        main_mask = pd.Series(True, index=df.index)

    ratios = {}
    for c in comp_names:
        others = [o for o in comp_names if o != c]
        if not others:
            ratios[c] = np.nan
            continue
        other_mags = df[[f"{o}_mag" for o in others]]
        median_others = other_mags.median(axis=1, skipna=True)
        rel = (df[f"{c}_mag"] - median_others)[main_mask].dropna()
        if len(rel) < max(12, smooth_points):
            ratios[c] = np.nan
            continue
        noise = _p2p_noise(rel.values)
        if not np.isfinite(noise) or noise <= 0:
            ratios[c] = np.nan
            continue
        sm = _running_median(rel.values, smooth_points)
        sm = sm[np.isfinite(sm)]
        if len(sm) < 3:
            ratios[c] = np.nan
            continue
        ratios[c] = float((sm.max() - sm.min()) / noise)
    return ratios


def select_steady_comps(trend_ratios, max_ratio=2.5, min_keep=4):
    """Reject comparison stars whose smooth trend dominates their noise.

    Unlike select_good_comps() this uses an ABSOLUTE threshold, because a
    trend/noise ratio is already normalised - a constant star sits near 1
    whatever its brightness or the night's conditions, so "how far above 1"
    means the same thing in every observation and needs no elbow-finding.

    Never drops below min_keep stars: an ensemble too small to average is a
    worse problem than one drifting member. When more stars exceed the
    threshold than can be spared, the worst offenders go first.
    """
    valid = {k: v for k, v in trend_ratios.items() if np.isfinite(v)}
    if len(valid) <= min_keep:
        return list(trend_ratios.keys()), []
    order = sorted(valid.items(), key=lambda kv: kv[1])
    keep, drop = [], []
    for name, r in order:
        if r > max_ratio and (len(order) - len(drop)) > min_keep:
            drop.append(name)
        else:
            keep.append(name)
    # stars with no number are kept: absence of evidence is not evidence
    keep += [k for k in trend_ratios if k not in valid]
    drop_set = set(drop)
    return [k for k in trend_ratios if k not in drop_set], drop


def select_good_comps(stability, min_gap_ratio=1.5, min_keep=2):
    """Split comparison stars into 'good' and 'rejected' by looking for the
    single largest RATIO jump between consecutive scatter values once
    sorted ascending (a simple, distribution-free 'elbow' detector). This
    adapts to however large the noisy-star cluster happens to be, unlike a
    fixed multiple-of-the-median threshold (which breaks down when close to
    half the comparison stars are noisy, as the median itself gets dragged
    up). If no jump exceeds `min_gap_ratio`, nothing is rejected - a
    reasonably uniform spread of scatter values is not evidence of a bad
    star, just normal statistical variation."""
    valid = {k: v for k, v in stability.items() if np.isfinite(v)}
    items = sorted(valid.items(), key=lambda kv: kv[1])
    names_sorted = [n for n, _ in items]
    vals_sorted = np.array([v for _, v in items])
    n = len(vals_sorted)
    if n <= min_keep:
        return names_sorted, []

    ratios = vals_sorted[1:] / np.clip(vals_sorted[:-1], 1e-12, None)
    best_idx = None
    best_ratio = min_gap_ratio
    for i, r in enumerate(ratios):
        if (i + 1) >= min_keep and r > best_ratio:
            best_ratio = r
            best_idx = i
    if best_idx is None:
        return names_sorted, []
    return names_sorted[:best_idx + 1], names_sorted[best_idx + 1:]


def recompute_ensemble(df, target_name, comp_names, comp_weights=None):
    """(Re-)computes the ensemble columns (n_comps_used, diff_mag,
    mag_comp_minus_target, flux_comp_minus_target, and their _err
    counterparts) using only the given comp_names, in place, for every row
    of df.

    comp_weights, when given, is a fixed {comp_name: weight} dict - normally
    1/measured_stability**2 for each comp, combined in magnitude space. Left
    out, each comp is weighted every frame by 1/formal_flux_error**2, which is
    what the CCD noise equation predicts for that frame.

    That formal error tracks a star's brightness, not how steady it actually
    is. On one real run, the noisiest of five comparison stars (11 mmag
    point-to-point) had the smallest formal error of the five and so was
    given the most weight - nearly a third of the ensemble - and the
    resulting light curve was noisier than every one of the five comparison
    stars measured individually, this noisiest one included. A fixed weight
    from each comp's own measured scatter does not have this failure mode,
    but it needs that scatter measured first, which is why it is only
    available on a second pass, after the quality checks below have run."""
    use_fixed = comp_weights is not None
    n_comps_used, diff_mag, mag_diff, flux_diff = [], [], [], []
    mag_diff_err, flux_diff_err = [], []
    for _, row in df.iterrows():
        target_flux = row.get(f"{target_name}_flux", np.nan)
        target_ok = row.get(f"{target_name}_ok", False)
        target_ferr = row.get(f"{target_name}_ferr", np.nan)
        target_magerr = row.get(f"{target_name}_magerr", np.nan)
        fluxes, ferrs = [], []
        for c in comp_names:
            if row.get(f"{c}_ok", False):
                fluxes.append(row[f"{c}_flux"])
                ferrs.append(row[f"{c}_ferr"])

        # all-or-nothing: only combine this frame if EVERY comp_name
        # succeeded here (see note in the main per-frame loop for why -
        # comp stars sit at very different absolute flux levels, so a
        # changing subset shifts the ensemble's level for reasons unrelated
        # to the target)
        all_comps_ok = len(fluxes) == len(comp_names)

        if target_ok and fluxes and all_comps_ok:
            fluxes_arr = np.array(fluxes)
            ferrs_arr = np.array(ferrs)

            if use_fixed:
                weights = np.array([comp_weights.get(c, np.nan) for c in comp_names],
                                   dtype=float)
                ok_w = np.isfinite(weights) & (weights > 0)
                if not ok_w.any():
                    weights = np.ones(len(comp_names))
                elif not ok_w.all():
                    weights = weights.copy()
                    weights[~ok_w] = np.median(weights[ok_w])
                mags_arr = -2.5 * np.log10(fluxes_arr) + 25.0
                ensemble_mag_w = float(np.sum(weights * mags_arr) / np.sum(weights))
                ensemble_flux_w = float(10 ** (-(ensemble_mag_w - 25.0) / 2.5))
                ensemble_magerr = float(np.sqrt(1.0 / np.sum(weights)))
                ensemble_flux_err = ensemble_magerr * ensemble_flux_w / 1.0857
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    weights = 1.0 / np.clip(ferrs_arr, 1e-6, None) ** 2
                if not np.isfinite(weights).any() or weights.sum() <= 0:
                    weights = np.ones_like(fluxes_arr)
                ensemble_flux_w = float(np.sum(weights * fluxes_arr) / np.sum(weights))
                ensemble_mag_w = -2.5 * np.log10(ensemble_flux_w) + 25.0
                ensemble_flux_err = float(np.sqrt(1.0 / np.sum(weights)))
                ensemble_magerr = 1.0857 * ensemble_flux_err / ensemble_flux_w

            target_mag = row[f"{target_name}_mag"]

            n_comps_used.append(len(fluxes))
            diff_mag.append(-2.5 * np.log10(target_flux / ensemble_flux_w))
            mag_diff.append(ensemble_mag_w - target_mag)
            flux_diff.append(ensemble_flux_w - target_flux)
            mag_diff_err.append(float(np.sqrt(target_magerr ** 2 + ensemble_magerr ** 2))
                                 if np.isfinite(target_magerr) else np.nan)
            flux_diff_err.append(float(np.sqrt(ensemble_flux_err ** 2 + target_ferr ** 2))
                                  if np.isfinite(target_ferr) else np.nan)
        else:
            n_comps_used.append(len(fluxes))
            diff_mag.append(np.nan)
            mag_diff.append(np.nan)
            flux_diff.append(np.nan)
            mag_diff_err.append(np.nan)
            flux_diff_err.append(np.nan)

    df["n_comps_used"] = n_comps_used
    df["diff_mag"] = diff_mag
    df["mag_comp_minus_target"] = mag_diff
    df["flux_comp_minus_target"] = flux_diff
    df["mag_comp_minus_target_err"] = mag_diff_err
    df["flux_comp_minus_target_err"] = flux_diff_err
    return df


def process_all(input_dir, ref_file, star_list_path, args):
    ref_data, ref_header = read_fits_data(ref_file)

    fmt = detect_star_list_format(star_list_path)
    if fmt == "startool":
        print(f"\n(Detected an AIP4WIN 'Star Data Tool' export in {star_list_path})")
        meta, rows_df = parse_star_data_tool(star_list_path)
        stars, config = configure_from_startool(rows_df, args, meta)
    else:
        stars = load_star_list(star_list_path)
        stars, config = configure_from_csv(ref_data, stars, args)

    config["gain"] = args.gain
    config["ron"] = args.ron
    config["dark"] = args.dark

    target_name = config["target_name"]
    use_names = [target_name] + config["chosen_comps"]
    anchor_name = config["anchor_name"]
    r_aper = config["r_aperture"]
    r_in = config["r_ann_in"]
    r_out = config["r_ann_out"]

    anchor_row = stars[stars.name == anchor_name].iloc[0]
    ax0, ay0 = float(anchor_row.x), float(anchor_row.y)

    if args.stamp_half is not None:
        stamp_half = args.stamp_half
        _, nearest = estimate_adaptive_stamp_half(stars, anchor_name,
                                                    median_fwhm=config.get("median_fwhm"))
    else:
        stamp_half, nearest = estimate_adaptive_stamp_half(stars, anchor_name,
                                                             median_fwhm=config.get("median_fwhm"))
        print(f"Auto-selected anchor stamp size = {stamp_half:.0f} px "
              f"(8 x FWHM={config.get('median_fwhm'):.2f}; override with --stamp_half)")
        if nearest is not None and nearest < stamp_half:
            print(f"  (nearest star to anchor {anchor_name} is {nearest:.1f} px away - "
                  f"within the stamp, so flip detection can use it)")

    detect_flips = not args.no_detect_flips
    if detect_flips and (nearest is None or nearest > stamp_half):
        detect_flips = False
        print(f"WARNING: the nearest star to the anchor is "
              f"{'unknown' if nearest is None else f'{nearest:.0f} px'} away, farther than "
              f"the {stamp_half:.0f} px stamp - it contains ONLY the anchor star itself, "
              f"which is point-symmetric and gives NO real evidence of a flip/rotation. "
              f"Disabling automatic flip detection for this run to avoid noise-driven false "
              f"positives (assuming 'normal' orientation throughout). If this field really "
              f"does flip between sessions (e.g. a meridian flip), increase --stamp_half to "
              f"at least {nearest:.0f} px (or more) so it can include that neighbour.")

    files = sorted(glob.glob(os.path.join(input_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No files matching {args.pattern} found in {input_dir}")
    print(f"Found {len(files)} FITS files. Measuring: {use_names} "
          f"(anchor = {anchor_name})\n")

    rows = []
    for i, path in enumerate(files, 1):
        fname = os.path.basename(path)
        try:
            data, header = read_fits_data(path)
        except Exception as e:
            print(f"[{i}/{len(files)}] SKIP {fname}: cannot read ({e})")
            continue

        dx, dy, peak, orientation = find_anchor_shift(
            ref_data, data, ax0, ay0,
            stamp_half=stamp_half, search_margin=args.search_margin,
            detect_flips=detect_flips, min_corr=args.min_corr)

        if dx is None:
            print(f"[{i}/{len(files)}] SKIP {fname}: anchor star/pattern not found "
                  f"(no orientation matched above min_corr={args.min_corr})")
            continue

        found_ax, found_ay = ax0 + dx, ay0 + dy
        sign_x, sign_y = ORIENTATIONS[orientation][1]

        row = {"file": fname, "jd": get_obs_time_jd(header, path),
               "anchor_dx": dx, "anchor_dy": dy, "anchor_corr": peak,
               "orientation": orientation}

        comp_fluxes = []
        comp_ferrs = []
        comp_names_used = []
        for name in use_names:
            star = stars[stars.name == name].iloc[0]
            rel_x, rel_y = star.x - ax0, star.y - ay0
            guess_x = found_ax + sign_x * rel_x
            guess_y = found_ay + sign_y * rel_y
            xr, yr, ok = refine_centroid(data, guess_x, guess_y, box_half=args.centroid_box)
            flux, ferr, sky = measure_flux(data, xr, yr, r_aper, r_in, r_out,
                                            gain=args.gain, ron=args.ron, dark=args.dark)

            good = ok and np.isfinite(flux) and flux > 0
            row[f"{name}_x"] = xr
            row[f"{name}_y"] = yr
            row[f"{name}_flux"] = flux
            row[f"{name}_ferr"] = ferr
            row[f"{name}_sky"] = sky
            row[f"{name}_ok"] = good
            row[f"{name}_mag"] = (-2.5 * np.log10(flux) + 25.0) if good else np.nan
            # sigma_mag from the CCD equation: sigma_mag = 1.0857/SNR = 1.0857*(ferr/flux)
            row[f"{name}_magerr"] = (1.0857 * ferr / flux) if good else np.nan

            if name != target_name and good:
                comp_fluxes.append(flux)
                comp_ferrs.append(ferr)
                comp_names_used.append(name)

        target_flux = row.get(f"{target_name}_flux", np.nan)
        target_ok = row.get(f"{target_name}_ok", False)
        target_magerr = row.get(f"{target_name}_magerr", np.nan)

        # all-or-nothing: only use this frame's ensemble if EVERY chosen
        # comparison star succeeded here, not just whichever subset happens
        # to be available. Different comp stars sit at very different
        # absolute flux levels, so averaging over a changing subset shifts
        # the ensemble's absolute level from frame to frame for reasons that
        # have nothing to do with the target - a single comp star dropping
        # out (e.g. a one-off centroid/edge issue) can otherwise swing the
        # combined result far more than it swings any individual comp star.
        all_comps_ok = len(comp_names_used) == len([n for n in use_names if n != target_name])

        if target_ok and comp_fluxes and all_comps_ok:
            comp_fluxes_arr = np.array(comp_fluxes)
            comp_ferrs_arr = np.array(comp_ferrs)
            # inverse-variance weighted ensemble (stars measured with less
            # photon noise count more) - falls back to a simple mean if all
            # errors are zero/invalid
            with np.errstate(divide="ignore", invalid="ignore"):
                weights = 1.0 / np.clip(comp_ferrs_arr, 1e-6, None) ** 2
            if not np.isfinite(weights).any() or weights.sum() <= 0:
                weights = np.ones_like(comp_fluxes_arr)
            ensemble_flux_w = float(np.sum(weights * comp_fluxes_arr) / np.sum(weights))
            ensemble_mag_w = -2.5 * np.log10(ensemble_flux_w) + 25.0
            target_mag = row[f"{target_name}_mag"]

            # standard inverse-variance weighted-mean error propagation
            ensemble_flux_err = float(np.sqrt(1.0 / np.sum(weights)))
            ensemble_magerr = 1.0857 * ensemble_flux_err / ensemble_flux_w

            row["n_comps_used"] = len(comp_fluxes)
            row["diff_mag"] = -2.5 * np.log10(target_flux / ensemble_flux_w)  # target - ensemble (legacy column)
            row["mag_comp_minus_target"] = ensemble_mag_w - target_mag
            row["flux_comp_minus_target"] = ensemble_flux_w - target_flux
            row["mag_comp_minus_target_err"] = float(
                np.sqrt(target_magerr ** 2 + ensemble_magerr ** 2)) if np.isfinite(target_magerr) else np.nan
            row["flux_comp_minus_target_err"] = float(
                np.sqrt(ensemble_flux_err ** 2 + row.get(f"{target_name}_ferr", np.nan) ** 2))
        else:
            row["n_comps_used"] = len(comp_fluxes)
            row["diff_mag"] = np.nan
            row["mag_comp_minus_target"] = np.nan
            row["flux_comp_minus_target"] = np.nan
            row["mag_comp_minus_target_err"] = np.nan
            row["flux_comp_minus_target_err"] = np.nan

        # --- absolute ensemble calibration (only if known catalog magnitudes
        # were provided for at least one comparison star): m_target =
        # 2.5 * [ log10(sum F_i / F_target) - log10(sum 10^(-0.4 * m_i)) ]
        known_mags = config.get("known_mags", {})
        row["target_mag_calibrated"] = np.nan
        if known_mags and target_ok:
            calib_fluxes = [f for n, f in zip(comp_names_used, comp_fluxes) if n in known_mags]
            calib_mags = [known_mags[n] for n in comp_names_used if n in known_mags]
            if calib_fluxes:
                flux_sum = float(np.sum(calib_fluxes))
                mag_sum_term = float(np.sum(10.0 ** (-0.4 * np.array(calib_mags))))
                row["target_mag_calibrated"] = 2.5 * (
                    np.log10(flux_sum / target_flux) - np.log10(mag_sum_term))

        rows.append(row)
        status = "ok" if target_ok else "TARGET FAILED"
        orient_flag = "" if orientation == "normal" else f" [FLIP: {orientation}]"
        print(f"[{i}/{len(files)}] {fname}: shift=({dx:+.1f},{dy:+.1f}) "
              f"corr={peak:.2f} comps={len(comp_fluxes)}/{len(config['chosen_comps'])} "
              f"target={status}{orient_flag}")

    df = pd.DataFrame(rows)

    # --- elapsed time in minutes, from the first (earliest) timed frame ---
    # (computed BEFORE the stability check below, since that check needs it
    # to exclude a trailing anomalous cluster - e.g. a re-pointing after a
    # long gap - from the drift/scatter statistics)
    if "jd" in df.columns and df["jd"].notna().any():
        t0 = df["jd"].dropna().min()
        df["elapsed_min"] = (df["jd"] - t0) * 1440.0
    else:
        df["elapsed_min"] = np.nan

    # --- auto-reject noisy/drifting/incomplete comparison stars ---
    config["comp_stability"] = {}
    config["comp_drift"] = {}
    config["comp_tail_offset"] = {}
    config["comp_completeness"] = {}
    config["completeness_failed"] = []
    config["rejected_comps"] = []
    config["final_comps"] = list(config["chosen_comps"])
    if args.auto_reject_comps and len(config["chosen_comps"]) > 2:
        # --- step 1: hard completeness gate (measured in >= min_completeness
        # of the frames where the TARGET succeeded - a frame where the
        # target failed reflects a general problem with that frame/those
        # conditions, not a comparison-star-specific one, so it doesn't
        # count against any comparison star either way) ---
        completeness, completeness_failed = evaluate_completeness(
            df, config["chosen_comps"], config["target_name"],
            min_completeness=args.min_completeness)

        print(f"\nComparison-star completeness check (min {args.min_completeness*100:.0f}% "
              f"of frames where the target succeeded):")
        for c in config["chosen_comps"]:
            pct = completeness.get(c, 0.0)
            flag = "FAILED (incomplete)" if c in completeness_failed else "ok"
            print(f"   {c:6s} {pct*100:5.1f}%  -> {flag}")

        survivors = [c for c in config["chosen_comps"] if c not in completeness_failed]

        # --- step 2: std/drift/tail_offset scoring, only on completeness
        # survivors ---
        main_mask = detect_main_segment(df, gap_minutes=args.tail_gap_minutes)
        n_excluded = int((~main_mask).sum())
        if n_excluded > 0:
            print(f"\n(Found a trailing cluster of {n_excluded} frame(s) - gap > "
                  f"{args.tail_gap_minutes} min - excluded from std/drift, but checked "
                  f"separately as 'tail_offset' below; override with --tail_gap_minutes)")

        stability, drift, tail_offset = evaluate_comp_stability(df, survivors, main_mask)
        badness = {}
        for c in survivors:
            s, d, t = stability.get(c, np.nan), drift.get(c, np.nan), tail_offset.get(c, np.nan)
            if not (np.isfinite(s) and np.isfinite(d)):
                continue
            badness[c] = s + abs(d) + (abs(t) if np.isfinite(t) else 0.0)

        if len(survivors) > 2:
            good_comps, badness_rejected = select_good_comps(
                badness, min_gap_ratio=args.reject_gap_ratio)
        else:
            good_comps, badness_rejected = survivors, []

        print("\nComparison-star stability check (spread + drift + tail-offset, "
              "vs. median of the other comps, completeness survivors only):")
        for c in survivors:
            s_val = stability.get(c, float("nan"))
            d_val = drift.get(c, float("nan"))
            t_val = tail_offset.get(c, float("nan"))
            flag = "REJECTED" if c in badness_rejected else "kept"
            t_str = f"{t_val:+.4f}" if np.isfinite(t_val) else "  n/a "
            print(f"   {c:6s} std={s_val:.4f}  drift={d_val:+.4f}  "
                  f"tail_offset={t_str} mag  -> {flag}")

        rejected_comps = list(completeness_failed) + list(badness_rejected)
        if rejected_comps:
            print(f"\n-> Rejected {len(rejected_comps)} comparison star(s) from the "
                  f"ensemble: {', '.join(rejected_comps)} "
                  f"({len(completeness_failed)} incomplete, {len(badness_rejected)} "
                  f"noisy/drifting; disable with --no_auto_reject_comps)")
        else:
            print("-> No comparison stars rejected.")
        # Weighted by each comp's own measured scatter, not by its formal
        # per-frame flux error - see recompute_ensemble(). Run whether or not
        # a comp was rejected: the formal weighting used until now is wrong
        # regardless, and this is the first point in the run where the real
        # scatter of each surviving comp is known.
        comp_weights = {c: 1.0 / stability[c] ** 2 for c in good_comps
                        if np.isfinite(stability.get(c, np.nan)) and stability[c] > 0}
        df = recompute_ensemble(df, config["target_name"], good_comps,
                                comp_weights=comp_weights or None)

        config["comp_stability"] = stability
        config["comp_drift"] = drift
        config["comp_tail_offset"] = tail_offset
        config["comp_completeness"] = completeness
        config["completeness_failed"] = completeness_failed
        config["rejected_comps"] = rejected_comps
        config["final_comps"] = good_comps

    # --- put the 6 requested columns first, keep everything else after ---
    target_mag_col = f"{target_name}_mag"
    target_flux_col = f"{target_name}_flux"
    front_cols = ["jd", "elapsed_min", target_mag_col,
                  "mag_comp_minus_target", target_flux_col,
                  "flux_comp_minus_target", "target_mag_calibrated"]
    front_cols = [c for c in front_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    return df, config


# --------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------

def build_comp_stars_table(config):
    """Builds a DataFrame with one row per originally-chosen comparison star:
    its completeness (fraction of target-successful frames where this star
    was also measured), stability scatter (std, mag), drift (mag, over the
    main segment), tail_offset (mag, trailing anomalous cluster vs.
    baseline, if any), a combined 0-100 quality score (100 = the best
    comparison star in this run among completeness survivors, lower =
    noisier/more drifting - computed as
    100 * best_badness / this_star_badness, where
    badness = std + |drift| + |tail_offset|), a rank (1 = best among
    survivors), and whether it was kept, rejected for incompleteness, or
    rejected for noise/drift."""
    stability = config.get("comp_stability", {})
    drift = config.get("comp_drift", {})
    tail_offset = config.get("comp_tail_offset", {})
    completeness = config.get("comp_completeness", {})
    completeness_failed = set(config.get("completeness_failed", []))
    rejected = set(config.get("rejected_comps", []))

    badness = {}
    for c in config["chosen_comps"]:
        s = stability.get(c, float("nan"))
        d = drift.get(c, float("nan"))
        t = tail_offset.get(c, float("nan"))
        if np.isfinite(s) and np.isfinite(d):
            badness[c] = s + abs(d) + (abs(t) if np.isfinite(t) else 0.0)
        else:
            badness[c] = float("nan")

    valid_badness = [v for v in badness.values() if np.isfinite(v) and v > 0]
    best_badness = min(valid_badness) if valid_badness else None

    rows = []
    for c in config["chosen_comps"]:
        std_val = stability.get(c, float("nan"))
        drift_val = drift.get(c, float("nan"))
        tail_val = tail_offset.get(c, float("nan"))
        completeness_val = completeness.get(c, float("nan"))
        b_val = badness.get(c, float("nan"))
        if best_badness is not None and np.isfinite(b_val) and b_val > 0:
            quality_pct = round(100.0 * best_badness / b_val, 1)
        else:
            quality_pct = None

        if c in completeness_failed:
            status = "REJECTED (incomplete)"
        elif c in rejected:
            status = "REJECTED (noisy/drifting)"
        else:
            status = "used"

        rows.append({
            "comp_star": c,
            "completeness_pct": round(completeness_val * 100, 1) if np.isfinite(completeness_val) else None,
            "stability_std_mag": round(std_val, 5) if np.isfinite(std_val) else None,
            "drift_mag": round(drift_val, 5) if np.isfinite(drift_val) else None,
            "tail_offset_mag": round(tail_val, 5) if np.isfinite(tail_val) else None,
            "quality_pct": quality_pct,
            "status": status,
        })
    table = pd.DataFrame(rows)
    if not table.empty and table["quality_pct"].notna().any():
        table = table.sort_values("quality_pct", ascending=False, na_position="last").reset_index(drop=True)
        table.insert(1, "rank", [i + 1 if q is not None else None
                                  for i, q in enumerate(table["quality_pct"])])
    return table


def write_outputs(df, config, out_base, formats):
    formats = [f.strip().lower() for f in formats]
    comp_table = build_comp_stars_table(config)

    # always write this, regardless of which output formats were chosen -
    # this is the one place to see which comparison stars were kept/rejected
    # and how they compare in quality
    if not comp_table.empty:
        comp_path = f"{out_base}_comp_stars.csv"
        comp_table.to_csv(comp_path, index=False)
        print(f"Wrote {comp_path}")

    if "csv" in formats:
        path = f"{out_base}.csv"
        df.to_csv(path, index=False)
        print(f"Wrote {path}")

    if "txt" in formats:
        path = f"{out_base}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("Photometry configuration\n")
            f.write("-------------------------\n")
            f.write(f"Target star        : {config['target_name']}\n")
            f.write(f"Comparison stars    : {', '.join(config['chosen_comps'])}\n")
            if config.get("rejected_comps"):
                f.write(f"Auto-rejected comps : {', '.join(config['rejected_comps'])}\n")
                f.write(f"Final ensemble comps: {', '.join(config['final_comps'])}\n")
            f.write(f"Anchor star         : {config['anchor_name']}\n")
            f.write(f"Median FWHM         : {config['median_fwhm']:.2f} px\n")
            f.write(f"Aperture radius     : {config['r_aperture']:.2f} px "
                    f"({config['k_aperture']} x FWHM)\n")
            f.write(f"Sky annulus         : {config['r_ann_in']:.2f} - "
                    f"{config['r_ann_out']:.2f} px "
                    f"({config['k_ann_in']} x / {config['k_ann_out']} x FWHM)\n")
            f.write(f"Gain (e-/ADU)       : {config['gain']}\n")
            f.write(f"Read noise (e-)     : {config['ron']}\n")
            f.write(f"Dark current (ADU)  : {config['dark']}\n\n")

            stability = config.get("comp_stability", {})
            if not comp_table.empty:
                f.write("Comparison star completeness, stability & quality ranking\n")
                f.write("-------------------------------------------------------------\n")
                f.write(f"{'rank':>4s} {'name':6s} {'complete%':>9s} {'std (mag)':>10s} "
                        f"{'drift (mag)':>12s} {'tail_offset':>12s} {'quality %':>10s}  status\n")
                for _, r in comp_table.iterrows():
                    rank_val = r.get("rank")
                    rank_str = str(int(rank_val)) if pd.notna(rank_val) else "-"
                    std_val = r["stability_std_mag"]
                    drift_val = r["drift_mag"]
                    tail_val = r["tail_offset_mag"]
                    comp_val = r["completeness_pct"]
                    q_val = r["quality_pct"]
                    tail_str = f"{tail_val:+.4f}" if tail_val is not None else "     n/a"
                    comp_str = f"{comp_val:.1f}" if comp_val is not None else "n/a"
                    q_str = f"{q_val:.1f}%" if q_val is not None else "n/a"
                    f.write(f"{rank_str:>4s} {r['comp_star']:6s} {comp_str:>9s} "
                            f"{std_val if std_val is not None else float('nan'):10.4f} "
                            f"{drift_val if drift_val is not None else float('nan'):+12.4f} "
                            f"{tail_str:>12s} "
                            f"{q_str:>10s}  {r['status']}\n")
                f.write("(complete% = fraction of target-successful frames where this star "
                        "was also\n measured; drift = end-of-main-run minus the initial "
                        "baseline level; tail_offset =\n the trailing anomalous cluster's "
                        "mean minus that same baseline (n/a if no such\n cluster); quality % "
                        "= 100 x best_(std+|drift|+|tail_offset|) / this star's value,\n "
                        "among completeness survivors only - 100% is the best comparison "
                        "star in this run,\n lower means noisier/more drifting)\n\n")

            f.write(df.to_string(index=False))
        print(f"Wrote {path}")

    if "xlsx" in formats:
        path = f"{out_base}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="photometry", index=False)
            cfg_rows = [
                ("Target star", config["target_name"]),
                ("Comparison stars used", ", ".join(config["chosen_comps"])),
                ("Auto-rejected comps", ", ".join(config.get("rejected_comps", [])) or "(none)"),
                ("Final ensemble comps", ", ".join(config.get("final_comps", config["chosen_comps"]))),
                ("Anchor star", config["anchor_name"]),
                ("Median FWHM (px)", round(config["median_fwhm"], 3)),
                ("k_aperture", config["k_aperture"]),
                ("k_ann_in", config["k_ann_in"]),
                ("k_ann_out", config["k_ann_out"]),
                ("Aperture radius (px)", round(config["r_aperture"], 3)),
                ("Annulus inner (px)", round(config["r_ann_in"], 3)),
                ("Annulus outer (px)", round(config["r_ann_out"], 3)),
                ("Gain (e-/ADU)", config["gain"]),
                ("Read noise RON (e-)", config["ron"]),
                ("Dark current (ADU)", config["dark"]),
            ]
            for name, val in config["per_star_fwhm"].items():
                cfg_rows.append((f"FWHM[{name}] (px)", "" if val is None else round(val, 3)))
            pd.DataFrame(cfg_rows, columns=["parameter", "value"]).to_excel(
                writer, sheet_name="config", index=False)

            if not comp_table.empty:
                comp_table.to_excel(writer, sheet_name="comp_stars", index=False)
        print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", default=None,
                     help="Folder with all FITS frames (if omitted, you will be asked for it)")
    ap.add_argument("--ref_file", default=None,
                     help="Reference FITS frame (if omitted, you will be asked for it)")
    ap.add_argument("--star_list", default=None,
                     help="Star list file - either the simple name,x,y,role CSV, "
                          "or an AIP4WIN 'Star Data Tool' text export "
                          "(if omitted, you will be asked for it)")

    ap.add_argument("--stars", default=None,
                     help="[CSV format] Comma-separated comparison-star names to use "
                          "(default: prompt interactively)")
    ap.add_argument("--anchor", default=None,
                     help="Comparison star (by name) used to track the field shift "
                          "(default: first chosen comparison star)")

    ap.add_argument("--target_index", type=int, default=None,
                     help="[Star Data Tool format] row number of the target/variable star "
                          "(default: prompt interactively)")
    ap.add_argument("--comp_indices", default=None,
                     help="[Star Data Tool format] comma-separated row numbers of the "
                          "comparison stars, or 'all' (default: prompt interactively)")
    ap.add_argument("--anchor_index", type=int, default=None,
                     help="[Star Data Tool format] row number (from --comp_indices) to use "
                          "as the tracking anchor star (default: first chosen comparison star)")

    ap.add_argument("--no_pair_cap", dest="pair_cap", action="store_false",
                    help="do not look for a companion too close for a star "
                         "finder to split, and do not cap the aperture on one")
    ap.add_argument("--pair_contam", type=float, default=0.01,
                    help="how much of a close companion's light the target's "
                         "aperture may carry (default 0.01 = 1%%)")
    ap.add_argument("--pair_sep_px", type=float, default=None,
                    help="separation in px of a companion you already know "
                         "about, from Aladin or Gaia. Needed on the Star Data "
                         "Tool path, which has no reference image to search.")
    ap.set_defaults(pair_cap=True)
    ap.add_argument("--k_aperture", type=float, default=None,
                     help="Aperture radius as a multiple of FWHM (default: prompt)")
    ap.add_argument("--k_ann_in", type=float, default=None,
                     help="Sky annulus inner radius as a multiple of FWHM (default: prompt)")
    ap.add_argument("--k_ann_out", type=float, default=None,
                     help="Sky annulus outer radius as a multiple of FWHM (default: prompt)")
    ap.add_argument("--fwhm_override", type=float, default=None,
                     help="Force a specific FWHM value (px) instead of the measured one")
    ap.add_argument("--fwhm_box", type=int, default=15,
                     help="Half-size (px) of the cutout used to fit each star's FWHM")

    ap.add_argument("--auto_reject_comps", action="store_true", default=True,
                     help="Automatically drop noisy/unstable comparison stars from the "
                          "ensemble based on their scatter vs. the other comps (default: on)")
    ap.add_argument("--no_auto_reject_comps", dest="auto_reject_comps", action="store_false",
                     help="Disable automatic rejection of noisy comparison stars - use "
                          "every chosen comparison star in the ensemble regardless of scatter")
    ap.add_argument("--reject_gap_ratio", type=float, default=1.4,
                     help="Minimum ratio jump (sorted scatter values) to treat as a real "
                          "gap between 'good' and 'noisy' comparison stars (default: 1.5)")
    ap.add_argument("--tail_gap_minutes", type=float, default=10.0,
                     help="A time gap larger than this (minutes) with no measurements at "
                          "all marks the start of a trailing anomalous cluster (e.g. after "
                          "a re-pointing). Excluded from std/drift, but checked separately "
                          "as tail_offset vs. the baseline (default: 10.0)")
    ap.add_argument("--min_completeness", type=float, default=0.95,
                     help="A comparison star must be successfully measured in at least this "
                          "fraction of the frames where the target succeeded, or it is hard-"
                          "rejected from the ensemble regardless of how stable it looks in "
                          "the frames where it does appear (default: 0.95)")

    ap.add_argument("--known_mags", default=None,
                     help="CSV file with columns name,mag giving the known/catalog "
                          "magnitude of each comparison star, used to add an "
                          "absolute-calibrated magnitude column for the target "
                          "(default: prompt whether you want to add this)")
    ap.add_argument("--skip_calibration", action="store_true",
                     help="Don't prompt about absolute magnitude calibration "
                          "(use only if --known_mags is also not given)")

    ap.add_argument("--gain", type=float, default=None,
                     help="Camera gain, electrons per ADU, used in the CCD noise "
                          "equation (default: prompt, suggested value 1.0)")
    ap.add_argument("--ron", type=float, default=None,
                     help="Camera read-out noise (RON), in electrons, used in the CCD "
                          "noise equation (default: prompt, suggested value 3.7 e- - a "
                          "COMMUNITY-SOURCED ESTIMATE, not an official datasheet value, "
                          "for the Sony IMX533 sensor (QHY/ZWO ASI533-class camera) at "
                          "GAIN=51, based on published gain-vs-read-noise curves: gain 0 "
                          "~3.8-4.0e-, gain 70 ~3.5e-, gain 100/unity ~1.5e-, gain>100 "
                          "~1.0-1.2e-. Replace with a value measured from your own bias "
                          "frames when available, or your camera's official figure.)")
    ap.add_argument("--dark", type=float, default=0.0,
                     help="Dark current per pixel, in ADU, used in the CCD noise "
                          "equation (default: 0.0 - matches AIP4Win's convention of "
                          "dark already subtracted during reduction)")


    ap.add_argument("--search_margin", type=int, default=100,
                     help="Max expected shift (pixels) to search for the anchor star/pattern")
    ap.add_argument("--stamp_half", type=int, default=None,
                     help="Half-size (px) of the anchor template. If omitted, this is "
                          "chosen AUTOMATICALLY from the star list, sized to just reach "
                          "the nearest neighbouring star to the anchor (x1.5 margin, "
                          "clipped to 20-150 px) - this adapts to how sparse or crowded "
                          "the field is. A lone (point-symmetric) star cannot reveal an "
                          "up-down/left-right flip, so the stamp must include at least "
                          "one neighbour.")
    ap.add_argument("--centroid_box", type=int, default=8)
    ap.add_argument("--no_detect_flips", action="store_true",
                     help="Disable automatic detection of up-down / left-right / 180-degree "
                          "flips between frames (e.g. from a meridian flip). Slightly faster; "
                          "only use if you're sure the frames never flip.")
    ap.add_argument("--min_corr", type=float, default=0.3,
                     help="Minimum normalized cross-correlation to accept an anchor match "
                          "(frame is skipped below this)")
    ap.add_argument("--pattern", default="*.fit*")

    ap.add_argument("--output", default="results",
                     help="Output file base name (no extension)")
    ap.add_argument("--output_dir", default=None,
                     help="Folder to write the results into (created if it does not "
                          "exist). Default: asked interactively, offering the input "
                          "folder - press Enter to accept it.")
    ap.add_argument("--output_format", default="xlsx,csv",
                     help="Comma-separated list: csv,xlsx,txt")
    ap.add_argument("--plot", action="store_true",
                     help="Generate the main light curve plot without asking "
                          "(default: asked interactively at the end of the run, Enter=yes)")
    args = ap.parse_args()

    if args.input_dir is None:
        args.input_dir = prompt_path("Enter path to the folder with the FITS files to measure",
                                      must_exist="dir")
    if args.ref_file is None:
        args.ref_file = prompt_path("Enter path to the reference FITS frame",
                                     must_exist="file")
    if args.star_list is None:
        args.star_list = prompt_path(
            "Enter path to the star list file (name,x,y,role CSV, or an "
            "AIP4WIN 'Star Data Tool' export)",
            must_exist="file")

    # Where the results go. Everything downstream builds its filenames from
    # args.output, so folding the folder into that prefix here is enough - the
    # CSV/xlsx/txt writer and every plot follow automatically.
    if args.output_dir is None:
        args.output_dir = prompt_path(
            "Enter path to the folder where the results should be saved",
            default=default_output_dir(args.input_dir, args.ref_file))
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"  created output folder: {args.output_dir}")
    args.output = os.path.join(args.output_dir, os.path.basename(args.output))

    if args.gain is None:
        args.gain = prompt_float("Camera gain (electrons per ADU) - check your FITS "
                                  "header's EGAIN keyword if present", 1.0)
    if args.ron is None:
        args.ron = prompt_float("Camera Read Out Noise RON (electrons) - check your "
                                 "camera's datasheet/bias frames if known; the suggested "
                                 "default is a community estimate, not an official spec", 3.7)

    print(f"\nUsing gain={args.gain} e-/ADU, RON={args.ron} e-, dark={args.dark} ADU "
          f"for the noise model.")

    df, config = process_all(args.input_dir, args.ref_file, args.star_list, args)

    write_outputs(df, config, args.output, args.output_format.split(","))

    n_ok = df["diff_mag"].notna().sum()
    print(f"\nDone. {n_ok}/{len(df)} frames have a valid differential magnitude.")

    want_main_plot = args.plot
    if not want_main_plot:
        try:
            ans = input("\nGenerate the main light curve plot (comparison ensemble - "
                        "target vs time)? [Y/n]: ").strip().lower()
        except EOFError:
            ans = ""
        want_main_plot = ans in ("", "y", "yes")

    if want_main_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        good = df[df["mag_comp_minus_target"].notna()].copy()
        if good.empty:
            print("Nothing to plot for the main light curve.")
        else:
            has_jd = good["jd"].notna().any()
            xaxis = good["elapsed_min"] if "elapsed_min" in good.columns and good["elapsed_min"].notna().any() else \
                (good["jd"] if has_jd else np.arange(len(good)))
            xlabel = "Elapsed time (minutes)" if "elapsed_min" in good.columns and good["elapsed_min"].notna().any() else \
                ("JD" if has_jd else "Frame #")

            plt.figure(figsize=(10, 4))
            yerr = good["mag_comp_minus_target_err"] if "mag_comp_minus_target_err" in good.columns else None
            plt.errorbar(xaxis, good["mag_comp_minus_target"], yerr=yerr,
                         fmt="o", markersize=4, elinewidth=1, capsize=2)
            plt.gca().invert_yaxis()
            plt.xlabel(xlabel)
            plt.ylabel("Comparison ensemble - target (mag)")
            plt.title(f"Light curve  (aperture={config['r_aperture']:.1f}px, "
                      f"FWHM={config['median_fwhm']:.1f}px, "
                      f"{len(config['final_comps'])} comp star(s) used)")
            plt.tight_layout()
            plt.savefig(f"{args.output}_light_curve.png", dpi=150)
            print(f"Plot written to {args.output}_light_curve.png")

    # --- optional: one difference plot per comparison star, vs the target ---
    try:
        ans = input("\nAlso generate an individual light curve (comp star - target) for "
                    "EACH comparison star, to visually check for noisy stars? [y/N]: ").strip().lower()
    except EOFError:
        ans = "n"
    if ans in ("y", "yes"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        target_name = config["target_name"]
        has_jd = df["jd"].notna().any()
        has_min = "elapsed_min" in df.columns and df["elapsed_min"].notna().any()
        xlabel = "Elapsed time (minutes)" if has_min else ("JD" if has_jd else "Frame #")

        saved = []
        for c in config["chosen_comps"]:
            col = f"{c}_mag"
            if col not in df.columns:
                continue
            sub = df[df[col].notna() & df[f"{target_name}_mag"].notna()].copy()
            if sub.empty:
                continue
            xaxis = sub["elapsed_min"] if has_min else (sub["jd"] if has_jd else np.arange(len(sub)))
            diff = sub[col] - sub[f"{target_name}_mag"]
            diff_err = None
            errcol = f"{c}_magerr"
            target_errcol = f"{target_name}_magerr"
            if errcol in sub.columns and target_errcol in sub.columns:
                diff_err = np.sqrt(sub[errcol] ** 2 + sub[target_errcol] ** 2)

            plt.figure(figsize=(10, 4))
            plt.errorbar(xaxis, diff, yerr=diff_err, fmt="o", markersize=4,
                         elinewidth=1, capsize=2)
            plt.gca().invert_yaxis()
            plt.xlabel(xlabel)
            plt.ylabel(f"{c} - {target_name} (mag)")
            if c in config.get("completeness_failed", []):
                status = "REJECTED (incomplete)"
            elif c in config.get("rejected_comps", []):
                status = "REJECTED (noisy/drifting)"
            else:
                status = "used in ensemble"
            comp_pct = config.get("comp_completeness", {}).get(c, float("nan"))
            std_val = config.get("comp_stability", {}).get(c, float("nan"))
            drift_val = config.get("comp_drift", {}).get(c, float("nan"))
            tail_val = config.get("comp_tail_offset", {}).get(c, float("nan"))
            tail_str = f"{tail_val:+.4f}" if np.isfinite(tail_val) else "n/a"
            comp_str = f"{comp_pct*100:.0f}%" if np.isfinite(comp_pct) else "n/a"
            plt.title(f"{c} vs {target_name}  (complete={comp_str}, std={std_val:.4f}, "
                      f"drift={drift_val:+.4f}, tail_offset={tail_str} mag, {status})")
            plt.tight_layout()
            fname = f"{args.output}_comp_{c}.png"
            plt.savefig(fname, dpi=150)
            plt.close()
            saved.append(fname)

        print(f"Wrote {len(saved)} per-comparison-star plot(s): {', '.join(saved)}")

    print(f"\nAll output files are in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
