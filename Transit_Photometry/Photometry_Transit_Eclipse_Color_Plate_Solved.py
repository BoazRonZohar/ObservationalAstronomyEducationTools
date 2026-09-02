# -*- coding: utf-8 -*-
"""
Photometry_Transit_Eclipse_Color_Plate_Solved.py
==================

One command for a whole night.

Point it at a folder of downloaded frames. It finds every target in there,
measures all three colour channels of each, checks the result against a
second, independent positioning method, and writes ONE summary page telling
you which results you can trust.

What it replaces
----------------
  channel splitting in Fitswork  -> the colour FITS already holds three
                                    separate planes; reading one is a copy,
                                    not a computation (verified bit-for-bit)
  the sky-subtraction step       -> the photometry subtracts a LOCAL sky from
                                    an annulus around each star, which both
                                    supersedes a single global number and
                                    follows gradients across the frame
  preparing a star list by hand  -> the target comes from the coordinates the
                                    telescope recorded, the comparison stars
                                    are chosen automatically. You can still
                                    supply your own list and it will win.

Reliability
-----------
Every channel is measured twice - once positioning stars by the plate
solution in each frame's header, once by the anchor-template method - and the
two are compared. Agreement is reported per channel. A run that fails this
check is flagged rather than quietly written out.

Nothing is ever written into the observation folders: input files are only
read. All output goes to the folder you choose.

This file is self-contained. The measurement engine that used to live in
Photometry_Transit_Eclipse_Mono_Star_List.py is merged in below, so the script runs on its
own with nothing beside it. Only the packages in requirements.txt are needed.

----------------------------------------------------------------------
Created by: Dr. Boaz Ron Zohar
Affiliation: Kinneret Observatory
Member of the LCO Global Sky Partners programme
Part of Observational Astronomy Education Tools
https://github.com/BoazRonZohar/ObservationalAstronomyEducationTools
"""

import io
import os
import re
import sys
import csv
import glob
import math
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")



# ==========================================================================
# the measurement engine, merged in (was Photometry_Transit_Eclipse_Mono_Star_List.py)
# ==========================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photometry_Transit_Eclipse_Color_Plate_Solved.py  (FWHM-based aperture version)
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
    python Photometry_Transit_Eclipse_Color_Plate_Solved.py \
        --input_dir  /path/to/fits_folder \
        --ref_file   /path/to/reference_frame.fits \
        --star_list  star_list.csv \
        --output results \
        --output_format xlsx,csv,txt

USAGE (fully unattended / batch - no prompts)
-----------------------------------------------
    python Photometry_Transit_Eclipse_Color_Plate_Solved.py \
        --input_dir /path/to/fits_folder --ref_file ref.fits \
        --star_list star_list.csv \
        --stars C1,C2,C3,C4,C6 \
        --k_aperture 3.0 --k_ann_in 4.0 --k_ann_out 6.0 \
        --output results --output_format xlsx
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

    config = _prompt_aperture_radii(median_fwhm, args)
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


def _prompt_aperture_radii(median_fwhm, args):
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


def _engine_main():
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




# ==========================================================================
# this tool
# ==========================================================================

from astropy.io import fits                                    # noqa: E402
from astropy.wcs import WCS                                    # noqa: E402
from astropy.coordinates import SkyCoord                       # noqa: E402
from astropy.stats import sigma_clipped_stats                  # noqa: E402
from photutils.detection import DAOStarFinder                  # noqa: E402
import astropy.units as u                                      # noqa: E402

CHANNELS = [(0, "R"), (1, "G"), (2, "B")]


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------

import contextlib


@contextlib.contextmanager
def quiet():
    """Swallow whatever the engine prints while it writes its files.

    The engine narrates every file it writes and every statistic it forms.
    None of it is a decision the observer makes, and it buried the two lines
    that are - which target, which channel. It still all reaches SUMMARY.txt.
    Errors are not swallowed: an exception propagates as it always did."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def build_wcs(header):
    h = header.copy()
    for k in list(h):
        if k.startswith(("TR1_", "TR2_")):
            del h[k]
    h["NAXIS"] = 2
    for k in ("NAXIS3", "NAXIS4"):
        if k in h:
            del h[k]
    return WCS(h)


def has_wcs(h):
    return all(k in h for k in ("CD1_1", "CD2_2", "CRVAL1", "CRPIX1"))


def position_angle(h):
    if "PA" in h:
        try:
            return float(h["PA"])
        except Exception:
            pass
    try:
        return math.degrees(math.atan2(float(h["CD1_2"]), float(h["CD1_1"]))) % 360.0
    except Exception:
        return float("nan")


def safe_name(name):
    """A folder name Windows will accept, built from the object name."""
    out = re.sub('[\\\\/:*?"<>|]', "_", str(name)).strip()
    out = "_".join(out.split()).rstrip(". ")
    return out or "target"


def scan_folder(folder, pattern="*.fts"):
    """Group every colour frame under `folder` by the object it points at."""
    files = []
    for root, _dirs, _f in os.walk(folder):
        files += glob.glob(os.path.join(root, pattern))
        files += glob.glob(os.path.join(root, "*.fit"))
    targets = {}
    skipped = []
    for p in sorted(set(files)):
        try:
            h = fits.getheader(p)
        except Exception as e:
            skipped.append((p, f"unreadable: {e}"))
            continue
        if h.get("NAXIS") != 3 or h.get("NAXIS3") != 3:
            continue                                  # not a 3-plane colour frame
        obj = str(h.get("OBJECT", "")).strip()
        if not obj:
            skipped.append((p, "no OBJECT in header"))
            continue
        # Some reductions write OBJECT as "name:id", a per-exposure id
        # appended to the target name - every frame of the same star then
        # looks like a different target. Ordinary OBJECT values carry no such
        # suffix, so this only ever touches that format.
        obj = re.sub(r":\d+$", "", obj).strip()
        targets.setdefault(obj, []).append(p)
    return targets, skipped


def classify_frames(paths):
    """Sort a target's frames into the ones to measure and the ones to drop.

    A meridian flip is NOT a reason to drop a frame - it is routine on this
    mount and both positioning methods handle it. It is measured and marked,
    because a flip moves the star onto a different part of the sensor and can
    leave a small step in the light curve exactly there.

    A frame pointing at a different field is a genuine reject: it belongs to
    the next target."""
    info = []
    for p in paths:
        h = fits.getheader(p)
        # get_obs_time_jd(), not a direct header read: it already knows to
        # fall back from BJD-OBS/JD through MJD-OBS/MJD to DATE-OBS. Reading
        # only BJD-OBS/JD directly left every frame without one of those two
        # exact keywords - most frames, since the common keywords are
        # MJD-OBS and DATE-OBS - with jd=NaN, which sorts the frames back into
        # whatever order glob happened to return them in.
        jd = get_obs_time_jd(h, p)
        info.append(dict(path=p, header=h, pa=position_angle(h),
                         crval=float(h.get("CRVAL1", np.nan)),
                         jd=float(jd) if jd is not None else np.nan))
    crv = np.array([i["crval"] for i in info], float)
    pas = np.array([i["pa"] for i in info], float)
    med_crv = np.nanmedian(crv)
    med_pa = np.nanmedian(pas)
    keep, drop = [], []
    for i in info:
        if not has_wcs(i["header"]):
            i["why"] = "no plate solution in header"
            drop.append(i)
        elif abs(i["crval"] - med_crv) > 0.5:
            i["why"] = f"different field (CRVAL1={i['crval']:.3f})"
            drop.append(i)
        else:
            d = abs((i["pa"] - med_pa + 180.0) % 360.0 - 180.0)
            i["flipped"] = d > 90.0
            keep.append(i)
    keep.sort(key=lambda i: i["jd"])
    return keep, drop


def choose_reference(frames, plane=1, n_sample=12, verbose=True):
    """Pick which frame everything else is measured against.

    This used to be simply the first frame, and the first frame is often the
    worst one in the run - it is taken before the mount, the focus and the
    guiding have settled. That has now bitten twice: on AC Cnc the first frame
    was the most trailed of the whole night (elongation 1.47 against a median
    of 1.09) and its shape pushed the target out of the green channel
    altogether; on V1111 Cep the first frame was a lone test exposure taken 32
    minutes before the real series, dimmer and softer than the rest, and it
    left the comparison-star stability test with a single point to work from.

    A frame is chosen instead on what it looks like: round stars first, since
    elongation is what breaks both centroiding and the star finder's shape
    cuts, then sharpness, and it must show a normal number of stars. Frames
    sitting alone in time - separated from their neighbours by much more than
    the run's usual cadence - are skipped, because that is what a test
    exposure looks like."""
    n = len(frames)
    if n < 3:
        return 0
    jd = np.array([f["jd"] for f in frames], float)
    t = (jd - np.nanmin(jd)) * 1440.0
    gaps = np.diff(t)
    cadence = np.median(gaps) if len(gaps) else 1.0
    isolated = np.zeros(n, bool)
    for i in range(n):
        near = []
        if i > 0:
            near.append(t[i] - t[i - 1])
        if i < n - 1:
            near.append(t[i + 1] - t[i])
        if near and min(near) > max(5 * cadence, 10.0):
            isolated[i] = True

    idx = [i for i in np.unique(np.linspace(0, n - 1, min(n_sample, n)).astype(int))
           if not isolated[i]]
    if not idx:
        idx = list(np.unique(np.linspace(0, n - 1, min(n_sample, n)).astype(int)))

    rows = []
    for i in idx:
        try:
            img = fits.getdata(frames[i]["path"])[plane].astype(float)[:-8]
            _m, med, std = sigma_clipped_stats(img, sigma=3.0)
            src = DAOStarFinder(fwhm=6.0, threshold=25 * std, exclude_border=True,
                                roundlo=-5, roundhi=5, sharplo=0, sharphi=5)(img - med)
            if src is None or len(src) < 5:
                continue
            a = np.array([[x, y, f, pk] for x, y, f, pk in
                          zip(src["xcentroid"], src["ycentroid"], src["flux"], src["peak"])
                          if pk < 45000])
            if len(a) < 5:
                continue
            a = a[np.argsort(-a[:, 2])][:20]
            E, W = [], []
            for x, y, _f, _pk in a:
                xi, yi = int(round(x)), int(round(y))
                st = img[yi - 12:yi + 13, xi - 12:xi + 13] - med
                if st.shape != (25, 25):
                    continue
                st = np.clip(st, 0, None)
                tot = st.sum()
                if tot <= 0:
                    continue
                yy, xx = np.mgrid[-12:13, -12:13]
                cx = (st * xx).sum() / tot
                cy = (st * yy).sum() / tot
                sxx = (st * (xx - cx) ** 2).sum() / tot
                syy = (st * (yy - cy) ** 2).sum() / tot
                sxy = (st * (xx - cx) * (yy - cy)).sum() / tot
                tr, det = sxx + syy, sxx * syy - sxy ** 2
                d = max(tr * tr / 4 - det, 0) ** 0.5
                l1, l2 = tr / 2 + d, tr / 2 - d
                if l2 <= 0:
                    continue
                E.append((l1 / l2) ** 0.5)
                W.append(2.3548 * (l1 * l2) ** 0.25)
            if E:
                rows.append((i, float(np.median(E)), float(np.median(W)), len(src)))
        except Exception:
            continue

    if not rows:
        return 0
    counts = np.array([r[3] for r in rows], float)
    keep_rows = [r for r in rows if r[3] >= 0.5 * counts.max()]
    if not keep_rows:
        keep_rows = rows
    # roundest first, sharpest as the tie-break
    keep_rows.sort(key=lambda r: (round(r[1], 3), r[2]))
    best = keep_rows[0]
    if verbose:
        worst = max(keep_rows, key=lambda r: r[1])
        print(f"    reference frame: #{best[0] + 1} of {n} "
              f"(elongation {best[1]:.2f}, FWHM {best[2]:.1f})"
              + (f"  [frame 1 was {rows[0][1]:.2f}]" if rows and rows[0][0] == 0
                 and abs(rows[0][1] - best[1]) > 0.02 else "")
              + (f"; {int(isolated.sum())} isolated frame(s) skipped"
                 if isolated.any() else ""))
    return int(best[0])


# --------------------------------------------------------------------------
# optional overrides supplied by the observer
# --------------------------------------------------------------------------

TARGET_FILE_NAMES = ("target.txt", "targets.txt", "coordinates.txt",
                     "target_coords.txt", "\u05db\u05d5\u05db\u05d1.txt",
                     "\u05e7\u05d5\u05d0\u05d5\u05e8\u05d3\u05d9\u05e0\u05d8\u05d5\u05ea.txt")


TARGET_FILE_HINT = """  To place the star yourself, or to set the aperture and sky ring yourself,
  put a file called target.txt in the folder with the frames:

        20:52:54.49 +37:03:58.9      the exact position of the star
        aperture = 4.5               radius in PIXELS      (optional)
        annulus  = 9 14              sky ring in PIXELS    (optional)

  Any line may be left out. With coordinates the star is taken exactly there
  and not moved onto the nearest light - which is what a close pair needs.
  Without the file everything is found automatically, as it is now."""


def _parse_coord(line):
    """Read one line of sky coordinates. Returns (ra_deg, dec_deg) or None.

    Accepts the sexagesimal form the observer already writes, with or without
    colons - "20:52:54.49 +37:03:58.9" and "20 52 54.49 +37 03 58.9" - and a
    plain pair of degrees. Two bare numbers are read as degrees; anything with
    colons, or with more than two fields, is read as hours and degrees."""
    s = line.strip().replace(",", " ")
    if not s:
        return None
    toks = s.split()
    try:
        if ":" in s or len(toks) >= 4:
            c = SkyCoord(s, unit=(u.hourangle, u.deg))
        elif len(toks) == 2:
            c = SkyCoord(float(toks[0]), float(toks[1]), unit=(u.deg, u.deg))
        else:
            return None
    except Exception:
        return None
    return float(c.ra.deg), float(c.dec.deg)


def _norm_name(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def read_target_file(folders, object_name=None, n_targets=1, verbose=True):
    """Read the observer's corrections for this target, if any were left.

    Everything this pipeline decides by itself - where the target is, how big
    the aperture should be - is decided from the data, and that is right for
    an ordinary field. Two cases are not ordinary.

    The first is the recorded pointing. The target is found from OBJCTRA and
    OBJCTDEC, the coordinates the telescope wrote into the header, and those
    are where the mount believed it was looking, not where the star is. On one
    of these nights they sit 15.8 arcsec - 12.6 px - from the star, while the
    plate solution of the same frame is good to a third of a pixel. When the
    observer has looked the star up properly, in Aladin or in Gaia, their
    coordinates are better than the header's and should simply be used.

    The second is a close pair. When another star sits about one FWHM away,
    an aperture sized from the measured width swallows both and a sky annulus
    sized the same way can land on the companion instead of on sky. Radii
    chosen by eye for that one field beat any rule.

    Neither correction is invented here, and neither is guessed. Both are read
    from a small text file the observer puts beside the frames, and if that
    file is absent nothing changes at all - the run behaves exactly as it did
    before this function existed. That matters: a correction is always fitted
    to one night, and a correction applied to every night would break the
    nights that were already right.

    File name: target.txt (also targets.txt, coordinates.txt, or the Hebrew
    names). Format, one block per target, blank lines and # ignored:

        20:52:54.49 +37:03:58.9
        Gaia DR3 1870866489773486080
        aperture = 4.5
        annulus  = 9 14

    The coordinate line and the name line are the file the observer already
    produces. Radii are optional and in PIXELS; write aperture_fwhm and
    annulus_fwhm instead if you prefer multiples of the FWHM. The name is
    matched against the OBJECT keyword and only matters when one folder holds
    more than one target; a single unnamed block applies to the only target
    there is."""
    path = None
    seen = set()
    for folder in folders:
        if not folder or folder in seen:
            continue
        seen.add(folder)
        try:
            entries = os.listdir(folder)
        except OSError:
            continue
        # Notepad appends its own .txt when the save dialog is left on
        # "Text Documents", so a file saved as target.txt lands on disk as
        # target.txt.txt and every match fails silently. That happened on the
        # first real use of this. Fold any repeat of the extension away
        # before matching.
        low = {}
        for e in entries:
            k = e.lower()
            while k.endswith(".txt.txt"):
                k = k[:-4]
            low.setdefault(k, e)
        for want in TARGET_FILE_NAMES:
            if want.lower() in low:
                path = os.path.join(folder, low[want.lower()])
                break
        if path:
            break
        # Nothing matched, but if something in the folder was clearly meant to
        # be this file, say so. A correction that is silently ignored is worse
        # than no correction: the run looks normal and the numbers are the old
        # ones.
        if verbose:
            near = [e for e in entries
                    if re.search(r"target|coord|כוכב", e, re.I)
                    and e.lower().endswith((".txt", ".dat"))]
            for e in near:
                print(f"    NOTE: found \"{e}\" but the name it needs is "
                      f"target.txt - ignored")
    if path is None:
        return None

    blocks, cur = [], {}
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError as e:
        print(f"    could not read {path}: {e}")
        return None

    def flush():
        if cur:
            blocks.append(dict(cur))
            cur.clear()

    for raw in lines:
        line = raw.split("#")[0].strip()
        if not line:
            continue
        # Coordinates are recognised first, so that a colon inside them can
        # never be mistaken for the colon of a "key: value" line.
        rd = _parse_coord(line)
        if rd is not None:
            if "radec" in cur:
                flush()
            cur["radec"] = rd
            continue
        m = re.match(r"^\s*([A-Za-z_]+)\s*[=:]\s*(.+)$", line)
        key, val = (m.group(1).lower(), m.group(2).strip()) if m else ("", "")
        if key in ("object", "target", "name") and val:
            cur["name"] = val
            continue
        if key in ("aperture", "aperture_px", "r_aperture") and val:
            try:
                cur["r_ap"] = float(val.split()[0])
            except ValueError:
                pass
            continue
        if key in ("annulus", "annulus_px", "sky") and val:
            nums = [float(v) for v in re.findall(r"[\d.]+", val)]
            if len(nums) >= 2:
                cur["r_in"], cur["r_out"] = nums[0], nums[1]
            continue
        if key in ("aperture_fwhm", "k_aperture") and val:
            try:
                cur["k_ap"] = float(val.split()[0])
            except ValueError:
                pass
            continue
        if key in ("annulus_fwhm", "k_annulus") and val:
            nums = [float(v) for v in re.findall(r"[\d.]+", val)]
            if len(nums) >= 2:
                cur["k_in"], cur["k_out"] = nums[0], nums[1]
            continue
        # anything left over is the star's name
        if "name" in cur and "radec" in cur:
            flush()
        cur["name"] = line
    flush()

    if not blocks:
        print(f"    {os.path.basename(path)} found but nothing readable in it")
        return None

    # Which block belongs to this target. A block that names a DIFFERENT star
    # is never used for this one: a file left behind in a folder holding two
    # targets would otherwise put one star's coordinates on the other.
    chosen = None
    for b in blocks:
        if not b.get("name"):
            continue
        have, want = _norm_name(b["name"]), _norm_name(object_name or "")
        if have and want and (have in want or want in have):
            chosen = b
            break
        # Gaia designations get written several ways - with DR3, with DR2,
        # with neither. The identifier itself is the long number.
        hn = set(re.findall(r"\d{8,}", b["name"]))
        wn = set(re.findall(r"\d{8,}", object_name or ""))
        if hn and (hn & wn):
            chosen = b
            break
    if chosen is None:
        unnamed = [b for b in blocks if not b.get("name")]
        if len(unnamed) == 1 and len(blocks) == 1:
            chosen = unnamed[0]
        elif len(blocks) == 1 and n_targets == 1:
            # One block, one target: there is nothing it could belong to
            # except this star, whatever it happens to be called.
            chosen = blocks[0]
            if verbose and chosen.get("name"):
                print(f"    {os.path.basename(path)} says "
                      f"\"{chosen['name']}\" and the frames say "
                      f"\"{object_name}\" - same star assumed, "
                      f"there is only one here")
    if chosen is None:
        if verbose:
            print(f"    {os.path.basename(path)} has {len(blocks)} block(s), "
                  f"none of them naming {object_name} - ignored")
        return None
    chosen["source"] = path
    return chosen

# --------------------------------------------------------------------------
# star selection
# --------------------------------------------------------------------------

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


def pick_stars(image, header, n_comps=12, max_radius=1300, sat_level=None,
               max_snap=30.0, radec=None):
    """Find the target and choose the comparison stars automatically.

    Target: the coordinates the telescope recorded for this pointing
    (OBJCTRA/OBJCTDEC), converted to pixels through this frame's own plate
    solution, then snapped to the nearest detected star.

    Comparisons: bright but unsaturated, isolated from neighbours (so no
    other star leaks into the aperture or the sky annulus), within a sensible
    distance of the target, and comparable to it in brightness."""
    w = build_wcs(header)
    fwhm = float(header.get("FWHM", 5.0)) or 5.0
    _m, med, std = sigma_clipped_stats(image[: image.shape[0] - 8], sigma=3.0)
    src = DAOStarFinder(fwhm=max(fwhm, 2.0), threshold=8 * std, exclude_border=True)(image - med)
    if src is None or len(src) == 0:
        raise RuntimeError("no stars detected in the reference frame")
    a = np.array([[x, y, f, pk] for x, y, f, pk in
                  zip(src["xcentroid"], src["ycentroid"], src["flux"], src["peak"])
                  if 40 < y < image.shape[0] - 40 and 40 < x < image.shape[1] - 40])
    if sat_level is None:
        sat_level = 0.85 * float(np.nanmax(image))

    # Where the target is. Normally from the coordinates the telescope
    # recorded; if the observer supplied better ones, from those.
    locked = radec is not None
    if locked:
        c = SkyCoord(radec[0], radec[1], unit=(u.deg, u.deg))
    else:
        c = SkyCoord(header["OBJCTRA"], header["OBJCTDEC"], unit=(u.hourangle, u.deg))
    tx, ty = [float(v) for v in w.all_world2pix(c.ra.deg, c.dec.deg, 0)]

    # The target is NOT looked up in the list above. That list comes from a
    # star finder, and a star finder applies shape cuts: it keeps sources that
    # look like clean round stars and discards the rest. That is right for
    # choosing comparison stars, and wrong for the target - the target is the
    # variable, it may be blended, elongated or in eclipse, and it must be
    # measured whatever it looks like. On AC Cnc the target was thrown out of
    # the green channel for a roundness of -1.034 against a limit of -1.0,
    # while red and blue kept it: the same star, present and 19 sigma above
    # sky, dropped over a hundredth.
    #
    # Its position is already known from the coordinates the telescope
    # recorded, so it only needs to be located, not discovered: search a small
    # box around the prediction with the shape cuts opened up.
    box = int(max(3.0 * max_snap, 60))
    xi, yi = int(round(tx)), int(round(ty))
    y0, y1 = max(yi - box, 0), min(yi + box, image.shape[0])
    x0, x1 = max(xi - box, 0), min(xi + box, image.shape[1])
    cut = image[y0:y1, x0:x1]
    _m2, med2, std2 = sigma_clipped_stats(cut, sigma=3.0)
    tsrc = DAOStarFinder(fwhm=max(fwhm, 2.0), threshold=6 * std2,
                         sharplo=0.0, sharphi=3.0,
                         roundlo=-3.0, roundhi=3.0)(cut - med2)
    if tsrc is None or len(tsrc) == 0:
        if locked:
            # Supplied coordinates are used whether or not a star finder
            # agrees there is something there - that is the whole point of
            # supplying them. Only the brightness scale is lost, and that is
            # only used to choose comparison stars of a similar brightness.
            T = np.array([tx, ty, float(np.nanmedian(a[:, 2])), np.nan])
            tsrc = None
        else:
            raise RuntimeError(
                f"nothing found within {box} px of the recorded coordinates - the "
                f"pointing or the plate solution looks wrong. Supply a star list "
                f"with --star_list for this target.")
    if tsrc is None:
        snap = float("nan")
    else:
        txs = np.array(tsrc["xcentroid"]) + x0
        tys = np.array(tsrc["ycentroid"]) + y0
        tfl = np.array(tsrc["flux"], dtype=float)
        tpk = np.array(tsrc["peak"], dtype=float)
        d = np.hypot(txs - tx, tys - ty)
        near = d <= max_snap
        if near.any():
            # among the candidates close enough to be the target, the brightest
            k = int(np.argmax(np.where(near, tfl, -np.inf)))
        elif locked:
            k = int(np.argmin(d))
        else:
            raise RuntimeError(
                f"the nearest source is {d.min():.1f} px from the recorded "
                f"coordinates, further than the {max_snap:.0f} px allowed. "
                f"Supply a star list with --star_list for this target.")
        if locked:
            # The supplied position is used as given: NOT snapped to the
            # nearest detection. On a close pair the nearest detection is
            # exactly what has to be avoided - the star finder reports the
            # blend, or the companion, and snapping to it measures the wrong
            # star. The distance to the nearest light is still measured and
            # reported, because it tells the observer whether the coordinates
            # landed where they meant them to; it is not acted on.
            T = np.array([tx, ty, tfl[k], tpk[k]])
        else:
            T = np.array([txs[k], tys[k], tfl[k], tpk[k]])
        snap = float(d[k])

    # Is anything else close enough to the target to worry about?
    #
    # Detections closer together than about one FWHM are not two stars: no
    # optic can separate them, and a star finder routinely reports a single
    # slightly elongated source twice, once on each side of its peak. On
    # V1111 Cep that split one target into a "pair" 3.2 px apart with a FWHM
    # of 5.8 and the whole observation was refused over it. Anything that
    # close is therefore treated as part of the target and merged into it.
    #
    # A genuinely separate neighbour, further out but still inside the
    # aperture, is a real problem - its light is being added to the target -
    # but it is not a reason to refuse to measure. It is reported, and the
    # note travels through to the summary so the number is read with it.
    others = np.hypot(a[:, 0] - T[0], a[:, 1] - T[1])
    others = others[others > 1e-3]
    resolved = others[others >= 1.0 * fwhm]
    nn = float(np.min(resolved)) if len(resolved) else np.inf
    blend_note = ""
    if nn < 2.0 * fwhm:
        blend_note = (f"another star {nn:.1f} px from the target "
                      f"(FWHM {fwhm:.1f}) - its light is inside the aperture")
    n_merged = int((others < 1.0 * fwhm).sum())
    if n_merged:
        blend_note = ((blend_note + "; ") if blend_note else "") + \
            f"{n_merged} detection(s) within one FWHM treated as the target itself"

    cand = []
    for s in a:
        if np.hypot(s[0] - T[0], s[1] - T[1]) < 3:
            continue
        if s[3] >= sat_level:
            continue
        if s[2] < 0.2 * T[2] or s[2] > 12 * T[2]:
            continue
        nn = np.sort(np.hypot(a[:, 0] - s[0], a[:, 1] - s[1]))[1]
        if nn < max(5 * fwhm, 25):
            continue
        if np.hypot(s[0] - T[0], s[1] - T[1]) > max_radius:
            continue
        cand.append(s)
    cand.sort(key=lambda s: -s[2])
    cand = cand[:n_comps]
    names = ["V"] + [f"C{i}" for i in range(1, len(cand) + 1)]
    xs = [T[0]] + [s[0] for s in cand]
    ys = [T[1]] + [s[1] for s in cand]
    stars = pd.DataFrame({"name": names, "x": xs, "y": ys,
                          "role": ["target"] + ["comp"] * len(cand)})
    return stars, snap, fwhm, blend_note


def stars_from_list(path):
    if detect_star_list_format(path) == "startool":
        _meta, rows = parse_star_data_tool(path)
        names = ["V"] + [f"C{i}" for i in range(1, len(rows))]
        return pd.DataFrame({"name": names, "x": list(rows.x), "y": list(rows.y),
                             "role": ["target"] + ["comp"] * (len(rows) - 1)})
    return load_star_list(path)


def _bright_stars(image, fwhm, n=400, thresh=15.0):
    _m, med, std = sigma_clipped_stats(image, sigma=3.0)
    src = DAOStarFinder(fwhm=max(fwhm, 2.0), threshold=thresh * std,
                        exclude_border=True)(image - med)
    if src is None or len(src) == 0:
        return np.zeros((0, 3))
    a = np.array([[x, y, f] for x, y, f in
                  zip(src["xcentroid"], src["ycentroid"], src["flux"])])
    return a[np.argsort(-a[:, 2])][:n]


def field_offset(ref_img, ref_wcs, img, wcs, fwhm, max_shift=40.0, tol=1.5):
    """Find the constant pixel shift between a frame and the plate solution.

    These frames are solved BEFORE being cropped, so the solution sits a fixed
    offset from the array. That offset cancels between frames of the same
    orientation, but not across a meridian flip: the crop takes the same rows
    of the array, and a 180 deg rotation puts those rows on the other side of
    the sky, so the two orientations disagree by twice the offset.

    The offset is in the header in plain sight: YORGSUBF=12 rows were cropped
    from a 3028-row frame, NAXIS2 is now 3008, and CRPIX2 is still 1514 - the
    centre of the frame BEFORE the crop. Predicting OBJCTRA/OBJCTDEC through
    the header WCS lands 6-17 px from the star on all six fields measured;
    subtracting YORGSUBF from CRPIX2 first lands within 0.8 px on five of the
    six. So the plate solution is good and the header is 12 rows stale.

    It is deliberately NOT corrected here. The observer reads corrected
    coordinates off these same frames in Aladin, through this same stale
    header, so both ends use the same ruler and agree whether or not the
    ruler is true. Fixing one end alone would move the target by 12 px.
    The shift is therefore measured, from the WHOLE star field and once per
    orientation, and a group whose shift cannot be measured is left
    unmeasured rather than guessed at.

    The shift is measured from the WHOLE star field, not from one star. Every
    predicted position is paired with every detected star nearby and the
    difference vectors are histogrammed; a real shift shows up as hundreds of
    pairs voting for the same bin, while a wrong shift would need hundreds of
    unrelated stars to line up by chance. That is what makes this safe where
    picking "the brightest star nearby" is not - a single-star lock is exactly
    the ambiguity that makes template matching fail in a sparse field.

    Returns (dx, dy, n_votes). (0, 0, 0) if no shift is convincing, in which
    case the caller should fall back to the plain solution rather than guess."""
    R = _bright_stars(ref_img, fwhm)
    S = _bright_stars(img, fwhm)
    if len(R) < 20 or len(S) < 20:
        return 0.0, 0.0, 0
    sky = ref_wcs.all_pix2world(R[:, 0], R[:, 1], 0)
    px, py = wcs.all_world2pix(sky[0], sky[1], 0)
    px = np.atleast_1d(px).astype(float); py = np.atleast_1d(py).astype(float)
    dx = S[:, 0][None, :] - px[:, None]
    dy = S[:, 1][None, :] - py[:, None]
    m = (np.abs(dx) <= max_shift) & (np.abs(dy) <= max_shift)
    if m.sum() < 20:
        return 0.0, 0.0, 0
    vx, vy = dx[m], dy[m]
    bins = np.arange(-max_shift, max_shift + tol, tol)
    H, xe, ye = np.histogram2d(vx, vy, bins=[bins, bins])
    i, j = np.unravel_index(int(H.argmax()), H.shape)
    votes = int(H[i, j])
    if votes < 15:
        return 0.0, 0.0, 0
    cx = 0.5 * (xe[i] + xe[i + 1]); cy = 0.5 * (ye[j] + ye[j + 1])
    near = (np.abs(vx - cx) <= tol) & (np.abs(vy - cy) <= tol)
    return float(np.median(vx[near])), float(np.median(vy[near])), votes


def lock_offset(image, gx, gy, fwhm, search=45.0):
    """Measure the constant shift between the plate solution and this array.

    Looks for the brightest star within `search` px of the predicted position
    of one bright reference star and returns the vector to it. Returns (0, 0)
    if nothing convincing is found, so a failure here degrades to plain WCS
    positioning rather than to a wrong answer."""
    h, w = image.shape
    x0, x1 = int(max(gx - search, 0)), int(min(gx + search, w))
    y0, y1 = int(max(gy - search, 0)), int(min(gy + search, h))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return 0.0, 0.0
    box = image[y0:y1, x0:x1]
    try:
        _m, med, std = sigma_clipped_stats(box, sigma=3.0)
        src = DAOStarFinder(fwhm=max(fwhm, 2.0), threshold=8 * std)(box - med)
    except Exception:
        return 0.0, 0.0
    if src is None or len(src) == 0:
        return 0.0, 0.0
    i = int(np.argmax(src["flux"]))
    return (float(src["xcentroid"][i]) + x0 - gx,
            float(src["ycentroid"][i]) + y0 - gy)


# --------------------------------------------------------------------------
# aperture
# --------------------------------------------------------------------------

def choose_radii(frames, plane, stars, cfg, n_sample=9, verbose=True):
    """Choose the aperture AND the sky annulus by measurement, not by guess.

    Both are found with the same test: try a value, see how much the
    comparison stars - which should be constant - disagree with each other,
    and keep whatever makes them quietest.

    That test responds to the annulus for a concrete reason. The annulus sets
    how much sky is subtracted from each star. Put its inner edge too close and
    some of the star's own light is counted as sky and removed, and how much
    depends on the width of the star that night, which changes frame to frame -
    so the star appears to flicker. Put its outer edge too far out and a
    neighbour or a background gradient leaks into the estimate instead. Either
    way a star that should sit still starts moving, which is what is measured.

    Every combination is evaluated from a SINGLE pass over the sample frames.
    An aperture sum does not depend on the annulus, and a sky level does not
    depend on the aperture, so each frame is read once, a few aperture sums and
    a few sky levels are taken from it, and every pairing after that is
    arithmetic. Scanning them one at a time instead meant re-reading every
    frame for every candidate, which was slower by more than an order of
    magnitude for the same answer.

    The inner edge is never allowed below 2 x FWHM whatever the numbers say, so
    the star's wings stay outside it by construction rather than by luck -
    searching too freely on one night's data is how a setting gets tuned into
    that night's noise."""
    from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
    fwhm = cfg["median_fwhm"]
    AP = [0.7, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5]
    IN = [2.0, 2.5, 3.0, 4.0, 5.0]
    WIDTH = 2.0                     # outer edge = inner + this, in FWHM
    # A close companion puts an absolute ceiling on the aperture, and the scan
    # has to respect it. Left free, the scan judges radii by how quiet the
    # COMPARISON stars are - and a radius that reaches the target's companion
    # does nothing to the comparison stars, so it scores well while quietly
    # diluting the target. The ceiling is the one thing the scan cannot see.
    if cfg.get("pair_sep_px"):
        r_max, _kept = pair_aperture_limit(
            cfg["pair_sep_px"], cfg.get("pair_ratio", 0.2), fwhm,
            cfg.get("pair_contam", 0.01))
        AP = [k for k in AP if k * fwhm <= r_max] or [min(AP)]

    idx = np.unique(np.linspace(0, len(frames) - 1, min(n_sample, len(frames))).astype(int))
    sub = [frames[i] for i in idx]
    ref_w = build_wcs(sub[0]["header"])
    sky = ref_w.all_pix2world([float(v) for v in stars.x], [float(v) for v in stars.y], 0)

    raw = {k: [] for k in AP}       # aperture sums, before sky removal
    bkg = {k: [] for k in IN}       # sky level per pixel
    for fr in sub:
        img = fits.getdata(fr["path"])[plane].astype(float)
        w = build_wcs(fr["header"])
        px, py = w.all_world2pix(sky[0], sky[1], 0)
        pos = []
        for gx, gy in zip(np.atleast_1d(px).astype(float), np.atleast_1d(py).astype(float)):
            xr, yr, _ok = refine_centroid(img, gx, gy, box_half=cfg["centroid_box"])
            pos.append((xr, yr))
        pos = np.array(pos, float)
        ok = np.isfinite(pos[:, 0])
        if ok.sum() < 4:
            continue
        for k in AP:
            ap = CircularAperture(pos[ok], r=k * fwhm)
            v = np.full(len(pos), np.nan)
            v[ok] = np.array(aperture_photometry(img, ap)["aperture_sum"])
            raw[k].append(v)
        for k in IN:
            an = CircularAnnulus(pos[ok], r_in=k * fwhm, r_out=(k + WIDTH) * fwhm)
            v = np.full(len(pos), np.nan)
            v[ok] = [sigma_clipped_stats(m.multiply(img)[m.data > 0], sigma=3.0)[1]
                     for m in an.to_mask(method="center")]
            bkg[k].append(v)

    for d in (raw, bkg):
        for k in d:
            d[k] = np.array(d[k]) if d[k] else None

    def score(k_ap, k_in):
        A, B = raw[k_ap], bkg[k_in]
        if A is None or B is None or A.shape[0] < 4 or A.shape[1] < 4:
            return np.inf
        F = A - B * (np.pi * (k_ap * fwhm) ** 2)
        good = np.all(np.isfinite(F), axis=1) & np.all(F > 0, axis=1)
        if good.sum() < 4:
            return np.inf
        C = F[good][:, 1:]
        sc = []
        for i in range(C.shape[1]):
            o = np.delete(C, i, axis=1).sum(axis=1)
            # the same point-to-point measure the final noise figure uses.
            # Plain scatter would be dominated by a transparency step and the
            # scan would then chase the step instead of the radii - on this
            # night that pushed the red aperture to the top of the grid.
            sc.append(_p2p(-2.5 * np.log10(C[:, i] / o)))
        sc = [v for v in sc if np.isfinite(v)]
        return float(np.median(sc)) if sc else np.inf

    best, k_ap, k_in = np.inf, cfg.get("k_aperture", 1.5), max(cfg.get("k_ann_in", 3.0), 2.0)
    for a in AP:
        for b in IN:
            v = score(a, b)
            if v < best:
                best, k_ap, k_in = v, a, b
    k_out = k_in + WIDTH
    if not np.isfinite(best):
        return (cfg.get("k_aperture", 1.5), max(cfg.get("k_ann_in", 3.0), 2.0),
                cfg.get("k_ann_out", 5.0), float("nan"))
    if verbose:
        print(f"    radii chosen by measurement: aperture {k_ap:.2f} x FWHM, "
              f"sky annulus {k_in:.2f}-{k_out:.2f} x FWHM "
              f"(comparison scatter {best*1000:.1f} mmag)")
    return k_ap, k_in, k_out, best


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def measure_channel(frames, plane, stars, ref_idx, cfg, route):
    """Measure one colour channel of one target, all frames, one route.

    route='wcs'    - positions from each frame's plate solution
    route='anchor' - positions from a template cut around the anchor star,
                     cross-correlated against each frame (the older method)
    """
    ref = frames[ref_idx]
    ref_img = fits.getdata(ref["path"])[plane].astype(float)
    ref_w = build_wcs(ref["header"])
    ref_pa = ref["pa"]
    names = list(stars.name)
    target_name = names[0]
    anchor_name = names[1] if len(names) > 1 else names[0]
    arow = stars[stars.name == anchor_name].iloc[0]
    ax0, ay0 = float(arow.x), float(arow.y)
    sky = ref_w.all_pix2world([float(v) for v in stars.x],
                              [float(v) for v in stars.y], 0)

    stamp_half, _nearest = estimate_adaptive_stamp_half(
        stars, anchor_name, median_fwhm=cfg["median_fwhm"])

    # One constant shift per orientation, measured from the whole star field.
    # Frames sharing the reference's orientation need none (it cancels); frames
    # on the other side of a meridian flip need one, because the crop offset
    # changes sign there. Measured once per group, not per frame.
    group_shift = {False: (0.0, 0.0, -1)}
    if route == "wcs":
        for fr in frames:
            g = bool(fr.get("flipped", False))
            if g in group_shift:
                continue
            gimg = fits.getdata(fr["path"])[plane].astype(float)
            dx, dy, votes = field_offset(ref_img, ref_w, gimg, build_wcs(fr["header"]),
                                         cfg["median_fwhm"])
            group_shift[g] = (dx, dy, votes)
            print(f"      orientation {'flipped' if g else 'normal'}: "
                  f"shift ({dx:+.1f}, {dy:+.1f}) px from {votes} stars"
                  if votes > 0 else
                  f"      orientation {'flipped' if g else 'normal'}: "
                  f"no consistent shift found - these frames will not be measured")

    rows = []
    for fr in frames:
        img = fits.getdata(fr["path"])[plane].astype(float)
        h = fr["header"]
        row = {"file": os.path.basename(fr["path"]),
               "jd": get_obs_time_jd(h, fr["path"]),
               "flipped": bool(fr.get("flipped", False))}

        if route == "wcs":
            w = build_wcs(h)
            px, py = w.all_world2pix(sky[0], sky[1], 0)
            px = np.atleast_1d(px).astype(float)
            py = np.atleast_1d(py).astype(float)
            # Positions come from the plate solution and nothing else. No
            # per-frame search is done here on purpose: an earlier version
            # looked for the brightest star near the predicted position and
            # shifted everything onto it, and in a dense field that "brightest
            # nearby star" is often the wrong one - precisely the failure this
            # route exists to avoid.
            #
            # One consequence is worth knowing. These frames were cropped
            # AFTER being solved (YORGSUBF=12, CRPIX2 still on the uncropped
            # centre), so the solution sits a fixed 12 px from the array.
            # That is harmless while it is the same in every frame, because
            # it cancels between the reference and the frame. It does NOT
            # cancel across a meridian flip: a 180 deg rotation puts the
            # cropped rows on the opposite side of the sky, so frames either
            # side of a flip disagree by twice it (measured here: 24 px).
            # The shift is measured per orientation; an orientation whose
            # shift cannot be measured is not measured at all, rather than
            # quietly mis-measured. See field_offset for why the header is
            # left alone rather than corrected.
            gx_, gy_, votes_ = group_shift.get(bool(fr.get("flipped", False)),
                                               (0.0, 0.0, 0))
            if votes_ == 0:
                continue          # no trustworthy shift for this orientation
            guesses = list(zip(px + gx_, py + gy_))
            row["anchor_corr"] = np.nan
            row["orientation"] = "rot180" if fr.get("flipped") else "normal"
            row["anchor_dx"] = float(guesses[1][0] - ax0) if len(guesses) > 1 else np.nan
            row["anchor_dy"] = float(guesses[1][1] - ay0) if len(guesses) > 1 else np.nan
        else:
            dx, dy, peak, orientation = find_anchor_shift(
                ref_img, img, ax0, ay0, stamp_half=stamp_half,
                search_margin=100, detect_flips=True, min_corr=0.3)
            if dx is None:
                continue
            sx, sy = ORIENTATIONS[orientation][1]
            fx, fy = ax0 + dx, ay0 + dy
            guesses = [(fx + sx * (float(s.x) - ax0), fy + sy * (float(s.y) - ay0))
                       for _i, s in stars.iterrows()]
            row.update(anchor_dx=dx, anchor_dy=dy, anchor_corr=peak,
                       orientation=orientation)

        # Centroid every star first, then size each aperture from that star's
        # own width in THIS frame. Seeing and focus drift during a run, and a
        # star at the edge of the field is wider than one at the centre, so a
        # radius fixed once for the whole run captures a changing fraction of
        # the light and that alone looks like a brightness change.
        centres, widths = [], []
        for gi, (gx, gy) in enumerate(guesses):
            if gi == 0 and cfg.get("lock_target"):
                # The target is held at the position the plate solution puts
                # the supplied coordinates, and not centroided. Centroiding
                # pulls the aperture towards the brightest light in the box,
                # and on a close pair that is the pair's combined centre, or
                # the companion - it walks off the star the observer asked
                # for. The comparison stars are centroided as always.
                xr, yr, ok = float(gx), float(gy), True
            else:
                xr, yr, ok = refine_centroid(img, gx, gy, box_half=cfg["centroid_box"])
            centres.append((xr, yr, ok))
            wv = np.nan
            if cfg.get("per_frame_fwhm", True) and ok:
                try:
                    v = measure_fwhm(img, xr, yr, box_half=cfg.get("fwhm_box", 15))
                    if v is not None and np.isfinite(v) and 1.5 < v < 30:
                        wv = float(v)
                except Exception:
                    pass
            widths.append(wv)
        widths = np.array(widths, float)
        # a star whose own width could not be measured falls back to the
        # frame's median, and that to the reference value - never to nothing
        frame_fwhm = np.nanmedian(widths) if np.isfinite(widths).any() else np.nan
        if not np.isfinite(frame_fwhm):
            frame_fwhm = cfg["median_fwhm"]
        row["frame_fwhm"] = float(frame_fwhm)

        # The ceiling a close companion puts on the aperture, recomputed for
        # THIS frame's seeing, and applied to every star in it - not to the
        # target alone. One radius for the whole frame is what makes the
        # varying flux fraction harmless: every star keeps the same share of
        # its own light, so the share cancels in the difference. Cap only the
        # target and the share stops cancelling, moves with the seeing, and
        # writes a trend into the curve that looks like an eclipse.
        r_pair, pair_drop = float("inf"), False
        if cfg.get("pair_sep_px"):
            r_pair, kept = pair_aperture_limit(
                cfg["pair_sep_px"], cfg.get("pair_ratio", 0.2), frame_fwhm,
                cfg.get("pair_contam", 0.01))
            pair_drop = kept < cfg.get("pair_min_flux", 0.5)
        row["pair_raper_max"] = r_pair
        row["pair_dropped"] = pair_drop

        for (name, (gx, gy)), (xr, yr, ok), wv in zip(zip(names, guesses), centres, widths):
            f_use = wv if np.isfinite(wv) else frame_fwhm
            if not cfg.get("per_frame_fwhm", True):
                f_use = cfg["median_fwhm"]
            r_ap = cfg["k_aperture"] * f_use
            if np.isfinite(r_pair):
                # once the ceiling binds, every star is measured at the frame's
                # radius, so the per-star widths stop mattering and the shares
                # stay equal
                r_ap = min(cfg["k_aperture"] * frame_fwhm, r_pair)
            r_in = cfg["k_ann_in"] * f_use
            r_out = cfg["k_ann_out"] * f_use
            if name == target_name and cfg.get("fixed_radii_px"):
                # Radii the observer chose by eye for this field, in pixels,
                # fixed for the whole run. They apply to the target only: the
                # comparison stars keep the measured radii, which is what a
                # differential measurement needs. A different aperture on the
                # target shifts its zero point by a constant and nothing else.
                r_ap, r_in, r_out = cfg["fixed_radii_px"]
            row[f"{name}_fwhm"] = float(wv) if np.isfinite(wv) else np.nan
            row[f"{name}_raper"] = float(r_ap)
            flux, ferr, skyv = measure_flux(
                img, xr, yr, r_ap, r_in, r_out,
                gain=cfg["gain"], ron=cfg["ron"], dark=cfg["dark"])
            good = bool(ok and np.isfinite(flux) and flux > 0)
            if pair_drop and name == target_name:
                # the seeing this frame is wider than the pair is apart: no
                # aperture separates them without throwing away the star. That
                # is a frame which did not measure this target.
                good = False
            row[f"{name}_x"] = xr
            row[f"{name}_y"] = yr
            row[f"{name}_flux"] = flux
            row[f"{name}_ferr"] = ferr
            row[f"{name}_sky"] = skyv
            row[f"{name}_ok"] = good
            row[f"{name}_mag"] = (-2.5 * np.log10(flux) + 25.0) if good else np.nan
            row[f"{name}_magerr"] = (1.0857 * ferr / flux) if good else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}
    t0 = df["jd"].dropna().min() if df["jd"].notna().any() else np.nan
    df["elapsed_min"] = (df["jd"] - t0) * 1440.0 if np.isfinite(t0) else np.nan

    comps = names[1:]
    df = recompute_ensemble(df, target_name, comps)

    config = dict(cfg)
    config.setdefault("per_star_fwhm", {n: cfg["median_fwhm"] for n in names})
    config.update(target_name=target_name, chosen_comps=comps,
                  anchor_name=anchor_name, final_comps=list(comps),
                  comp_stability={}, comp_drift={}, comp_tail_offset={},
                  comp_completeness={}, completeness_failed=[], rejected_comps=[],
                  comp_trend_ratio={}, drift_rejected=[])

    # the engine's own quality gates, unchanged
    if len(comps) > 2:
        completeness, failed = evaluate_completeness(df, comps, target_name,
                                                         min_completeness=0.95)
        survivors = [c for c in comps if c not in failed]
        main_mask = detect_main_segment(df, gap_minutes=10.0)
        stability, drift, tail = evaluate_comp_stability(df, survivors, main_mask)
        badness = {}
        for c in survivors:
            s, d, t = stability.get(c, np.nan), drift.get(c, np.nan), tail.get(c, np.nan)
            if np.isfinite(s) and np.isfinite(d):
                badness[c] = s + abs(d) + (abs(t) if np.isfinite(t) else 0.0)

        # A star is only ever dropped on evidence that it behaved badly.
        # If the stability test could not produce a number - which happens when
        # the run has an isolated frame separated from the rest, because the
        # segment finder is built for a trailing cluster and a leading one
        # leaves it with a single point - then there is no evidence either way,
        # and the stars are kept. Treating "could not measure" as "failed"
        # emptied the ensemble completely on V1111 Cep and produced a light
        # curve with no comparison stars at all.
        if not badness:
            print("    (the stability test could not run - keeping all "
                  f"{len(survivors)} comparison stars; check the light curve)")
            good, bad = survivors, []
        elif len(survivors) > 2:
            good, bad = select_good_comps(badness, min_gap_ratio=1.4)
            if not good:
                good, bad = survivors, []
        else:
            good, bad = survivors, []
        # Second filter, for the failure mode the first one cannot see: a star
        # that is not especially noisy but slides smoothly across the run.
        # A standard deviation buries such a trend under the noise; a
        # trend-to-noise ratio does not. See evaluate_comp_trend.
        trend = {}
        drifters = []
        if len(good) > 4:
            trend = evaluate_comp_trend(df, good, main_mask)
            good2, drifters = select_steady_comps(trend, max_ratio=2.5,
                                                      min_keep=4)
            if good2:
                good = good2

        rejected = list(failed) + list(bad) + list(drifters)
        # Weighted by each comp's own measured scatter, not by its formal
        # per-frame flux error - see recompute_ensemble(). Run whether or not
        # a comp was rejected: the formal weighting used until now is wrong
        # regardless, and this is the first point where the real scatter of
        # each surviving comp is known.
        comp_weights = {c: 1.0 / stability[c] ** 2 for c in good
                        if np.isfinite(stability.get(c, np.nan)) and stability[c] > 0}
        df = recompute_ensemble(df, target_name, good, comp_weights=comp_weights or None)
        config.update(comp_stability=stability, comp_drift=drift,
                      comp_tail_offset=tail, comp_completeness=completeness,
                      comp_trend_ratio=trend, drift_rejected=drifters,
                      completeness_failed=failed, rejected_comps=rejected,
                      final_comps=good)
    return df, config


# --------------------------------------------------------------------------
# agreement between the two routes
# --------------------------------------------------------------------------

def compare_routes(df_w, df_a, names, limits):
    out = {}
    if df_a.empty or df_w.empty:
        out["verdict"] = "anchor route produced nothing"
        out["ok"] = False
        out["pos_max"] = float("nan")
        out["curve_mmag"] = float("nan")
        return out
    a = df_a.set_index("file")
    w = df_w.set_index("file")
    common = [f for f in w.index if f in a.index]
    diffs = []
    for f in common:
        for n in names:
            if f"{n}_x" not in a.columns:
                continue
            xa, ya = a.at[f, f"{n}_x"], a.at[f, f"{n}_y"]
            xw, yw = w.at[f, f"{n}_x"], w.at[f, f"{n}_y"]
            if all(np.isfinite([xa, ya, xw, yw])):
                diffs.append(float(np.hypot(xa - xw, ya - yw)))
    pos_max = float(np.nanmax(diffs)) if diffs else float("nan")

    ca = np.array([a.at[f, "mag_comp_minus_target"] for f in common], float)
    cw = np.array([w.at[f, "mag_comp_minus_target"] for f in common], float)
    ok = np.isfinite(ca) & np.isfinite(cw)
    if ok.sum() > 2:
        d = (ca[ok] - np.nanmean(ca[ok])) - (cw[ok] - np.nanmean(cw[ok]))
        curve = float(np.std(d, ddof=1) * 1000)
    else:
        curve = float("nan")

    checks = []
    if np.isfinite(pos_max):
        checks.append(pos_max <= limits["pos_px"])
    if np.isfinite(curve):
        checks.append(curve <= limits["curve_mmag"])
    out["pos_max"] = pos_max
    out["curve_mmag"] = curve
    out["n_frames_wcs"] = int(df_w["mag_comp_minus_target"].notna().sum())
    out["n_frames_anchor"] = int(df_a["mag_comp_minus_target"].notna().sum())
    out["ok"] = bool(checks) and all(checks)
    return out


def scatter_mmag(df):
    v = df["mag_comp_minus_target"].to_numpy(float) if "mag_comp_minus_target" in df else np.array([])
    v = v[np.isfinite(v)]
    return float(np.std(v, ddof=1) * 1000) if len(v) > 2 else float("nan")


def throughput(df, comp_names):
    """How much light the whole field delivered in each frame, relative to
    normal. Built from the comparison stars, so it tracks the sky and the
    optics, not the target."""
    cols = [f"{c}_flux" for c in comp_names if f"{c}_flux" in df.columns]
    if len(cols) < 3:
        return None
    F = df[cols].to_numpy(float)
    ok = np.all(np.isfinite(F) & (F > 0), axis=1)
    E = np.full(len(df), np.nan)
    E[ok] = F[ok].sum(axis=1)
    med = np.nanmedian(E)
    return E / med if np.isfinite(med) and med > 0 else None


def find_throughput_steps(rel, min_step=0.15, min_run=5):
    """Split a run where the field's throughput changed abruptly and stayed
    changed - cloud arriving, dew, a dome edge - as opposed to drifting.

    Such a step matters because it does not cancel in differential photometry
    as cleanly as it should: the loss is rarely exactly grey, so the stars do
    not all dim by the same factor and a level offset survives between the
    segments. Measuring the run as one series then reports that offset as
    noise. On the AC Cnc night this turned 23 and 31 mmag into 94.

    Returns a list of (start, end) index ranges. One range means no step."""
    n = len(rel)
    if n < 3 * min_run:
        return [(0, n)]
    v = np.log10(np.where(np.isfinite(rel) & (rel > 0), rel, np.nan))
    bounds = [0]
    start = 0
    while True:
        best, best_gap = None, 0.0
        for i in range(start + min_run, n - min_run + 1):
            a, b = v[start:i], v[i:]
            if np.isfinite(a).sum() < min_run or np.isfinite(b).sum() < min_run:
                continue
            gap = abs(np.nanmedian(b) - np.nanmedian(a))
            spread = max(np.nanstd(a), np.nanstd(b), 1e-4)
            if gap > best_gap and gap > abs(np.log10(1 - min_step)) and gap > 4 * spread:
                best, best_gap = i, gap
        if best is None:
            break
        bounds.append(best)
        start = best
        if n - start < 3 * min_run:
            break
    bounds.append(n)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _p2p(v):
    """Scatter from the difference between consecutive points.

    A level step in the middle of a run, or a slow drift, barely changes how
    much neighbouring points differ from each other, so this measures the
    short-timescale noise and ignores both. That removes the need to detect
    steps at all: measuring the plain standard deviation across a run with a
    transparency step reported 94 mmag where the real noise was 14."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 4:
        return float("nan")
    d = np.diff(v)
    # 1.4826*MAD is a standard-deviation-equivalent that ignores outliers;
    # dividing by sqrt(2) undoes the fact that a difference carries the noise
    # of two points
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad / np.sqrt(2.0))


_NULL_SCALE = {}


def _smooth_spread(v, w):
    s = pd.Series(v).rolling(w, center=True, min_periods=max(2, w // 2)).median()
    return float(np.nanstd(s.to_numpy(), ddof=1))


def _null_scale(n, w, n_sim=600):
    """How far smoothing shrinks pure white noise, for this length and window.

    Measured rather than derived, and cached: it depends only on n and w."""
    key = (n, w)
    if key not in _NULL_SCALE:
        rng = np.random.default_rng(17 + 131 * n + w)
        _NULL_SCALE[key] = float(np.median(
            [_smooth_spread(rng.normal(0.0, 1.0, n), w) for _ in range(n_sim)]))
    return _NULL_SCALE[key] or np.nan


def smoothness(v):
    """How much SMOOTH structure a curve holds, in units of its own noise.

    The curve is put through a running median and the spread of the result is
    measured. Noise averages down under smoothing and a real trend does not,
    so what survives is the signal. Dividing by the spread the same smoothing
    leaves in pure noise of the same level puts a flat star near 1 whatever
    its noise happens to be.

    The noise level is taken from consecutive frames, which an eclipse cannot
    inflate - frames are a minute apart and an eclipse takes hours."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 8:
        return float("nan")
    w = int(np.clip(n // 8, 3, 9)) | 1
    sigma = 1.4826 * np.median(np.abs(np.diff(v))) / np.sqrt(2)
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    k = _null_scale(n, w)
    return _smooth_spread(v, w) / (sigma * k) if np.isfinite(k) else float("nan")


def change_ratio(df, comp_names):
    """Does the target change more than the comparison stars do?

    This replaces a number that was wrong. The old ratio divided the target's
    scatter by the comparison stars' noise, and called anything above 3 a real
    change. But a target can be noisier than the comparison stars for reasons
    that have nothing to do with variability - it may be fainter, or blended
    with a neighbour - and that extra noise went straight into the ratio. On
    six observations it declared five of them variable. One was.

    What is measured instead: how much SMOOTH structure the target's curve
    holds, against how much the comparison stars' own curves hold. Each
    comparison star is measured the same way, against the mean of the others,
    so the comparison set supplies the null from the same frames, the same
    sky and the same night. That matters, because photometric noise is not
    white - air mass, seeing and transparency drift slowly, and a slow drift
    is exactly what a search for smooth structure will find. Against pure
    white noise one of these nights looked like a real change in all three
    channels; against the comparison stars it did not stand out at all,
    because every star in the field was drifting with it.

    Returns (ratio, target_smoothness, comparison_level, n_comps). A ratio
    near 1 means the target does whatever the comparison stars do."""
    if "mag_comp_minus_target" not in df.columns:
        return float("nan"), float("nan"), float("nan"), 0
    r_t = smoothness(df["mag_comp_minus_target"].to_numpy(float))

    mags, used = {}, []
    for c in comp_names:
        col = f"{c}_mag"
        if col in df.columns:
            mags[c] = df[col].to_numpy(float)
            used.append(c)
    r_c = []
    for c in used:
        others = [mags[o] for o in used if o != c]
        if not others:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = mags[c] - np.nanmean(np.vstack(others), axis=0)
        s = smoothness(v)
        if np.isfinite(s):
            r_c.append(s)
    if len(r_c) < 4 or not np.isfinite(r_t):
        # Too few comparison stars to say what "normal" looks like tonight.
        # The white-noise scale is still meaningful on its own, so it is
        # reported, but without a field to measure it against.
        return float("nan"), r_t, float("nan"), len(r_c)
    # The most structured comparison star, not the average one: the question
    # is whether the target does something no ordinary star in the field does.
    level = float(np.max(r_c))
    return (r_t / level if level > 0 else float("nan")), r_t, level, len(r_c)


def noise_mmag(df, comp_names, segments=None):
    """The measurement noise, from the comparison stars.

    NOT from the target. The target is the variable star: its scatter is
    mostly the variation being looked for, so using it as a noise figure
    reports a real eclipse as if it were an error. A comparison star is
    expected to be constant, so how much IT moves against the others is what
    the measurement can actually resolve.

    `segments` is accepted and ignored - kept so older calls keep working.
    The point-to-point estimator does not need the run split up."""
    cols = [f"{c}_flux" for c in comp_names if f"{c}_flux" in df.columns]
    if len(cols) < 2:
        return float("nan")
    F = df[cols].to_numpy(float)
    ok = np.all(np.isfinite(F) & (F > 0), axis=1)
    F = F[ok]
    if len(F) < 6:
        return float("nan")
    out = []
    for i in range(F.shape[1]):
        o = np.delete(F, i, axis=1).sum(axis=1)
        out.append(_p2p(-2.5 * np.log10(F[:, i] / o)) * 1000)
    out = [v for v in out if np.isfinite(v)]
    if not out:
        return float("nan")
    v = float(np.median(out))
    if F.shape[1] == 2:
        v /= np.sqrt(2.0)      # two stars measure each other; split the noise
    return v


def target_depth_mag(df, faint_frac=0.10):
    """How deep the target went, measured robustly.

    Not max minus min: that is decided by exactly two points, so one bad frame
    at each end sets the answer. On this same AC Cnc night the peak-to-peak
    depth moved by 0.22 mag between two runs of the same data purely from a
    different aperture, and it broke the colour ordering that had looked like
    a physical result.

    Instead: the baseline is the median of the brighter half of the points -
    a star spends most of a run out of eclipse - and the bottom is the median
    of the faintest tenth. Both are medians over many points, so neither moves
    when one frame misbehaves."""
    v = df["mag_comp_minus_target"].to_numpy(float) if "mag_comp_minus_target" in df else np.array([])
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return float("nan")
    n_faint = max(3, int(round(faint_frac * len(v))))
    order = np.sort(v)                     # larger value = target fainter
    bottom = float(np.median(order[-n_faint:]))
    baseline = float(np.median(order[: len(order) // 2]))
    return bottom - baseline


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def write_lightcurve_sheet(df, config, out_path, target_name, channel,
                           measured_noise=None):
    """A small spreadsheet holding just the light curve, ready to plot.

    The full table has 145 columns - position, flux, error, sky, FWHM and
    aperture for every one of thirteen stars - and the three columns anyone
    actually plots sat at 139, 142 and 144. Nobody scrolls to EI to draw a
    graph. This file has those three first and nothing else in the way, so the
    columns can be selected straight away and turned into a scatter plot with
    error bars."""
    cols = {}
    if "elapsed_min" in df.columns:
        cols["Time_minutes"] = df["elapsed_min"]
    if "jd" in df.columns:
        cols["BJD"] = df["jd"]
    if "mag_comp_minus_target" in df.columns:
        # Named for exactly what it is. It was called "Magnitude", and that
        # was read as the star's own magnitude - so the column everyone
        # actually plots looked like it was missing from the file it was
        # sitting in. The sign follows the convention too: variable minus
        # comparisons, so an eclipse rises here and dips once the axis is
        # inverted.
        cols["Var_minus_Comp"] = -df["mag_comp_minus_target"]
    err = None
    if "mag_comp_minus_target_err" in df.columns:
        err = df["mag_comp_minus_target_err"].to_numpy(float)
    if measured_noise is not None and np.isfinite(measured_noise):
        base = np.zeros(len(df)) if err is None else np.nan_to_num(err, nan=0.0)
        err = np.maximum(base, measured_noise / 1000.0)
    if err is not None:
        cols["Error"] = err
    out = pd.DataFrame(cols)
    # keep the plotting columns first, then a few worth glancing at
    for extra in ("frame_fwhm", "flipped", "file"):
        if extra in df.columns:
            out[extra] = df[extra].values
    if "Var_minus_Comp" in out:
        out = out[np.isfinite(out["Var_minus_Comp"].to_numpy(float))]

    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as w:
            out.to_excel(w, sheet_name="light curve", index=False)
            sh = w.sheets["light curve"]
            sh.column_dimensions["A"].width = 14
            sh.column_dimensions["B"].width = 16
            sh.column_dimensions["C"].width = 14
            sh.column_dimensions["D"].width = 12
            sh.freeze_panes = "A2"
            n = len(out)
            note_rows = [
                "",
                f"{target_name}   channel {channel}",
                "A = minutes from the first frame",
                "C = Var_minus_Comp: the variable star minus the comparison",
                "    ensemble. This is the light curve - not the star's own",
                "    magnitude. Plot this one.",
                "D = error, from the scatter measured on the comparison stars",
                "To plot: select columns A and C -> Insert -> Scatter.",
                "Then invert the vertical axis (a larger magnitude means fainter),",
                "and add error bars from column D (Custom -> both directions).",
            ]
            for i, txt in enumerate(note_rows):
                sh.cell(row=n + 3 + i, column=1, value=txt)
    except Exception:
        out.to_csv(os.path.splitext(out_path)[0] + ".csv", index=False)
    return out_path


def finder_chart(image, stars, config, out_path, target_name, channel, note=""):
    """One picture showing which stars the run actually used.

    Everything else in the output is numbers about stars the code chose on its
    own; this is the only place you can check that it chose the right ones.
    The target is marked differently from the comparison stars, and stars that
    were selected but then thrown out by the quality tests are marked as
    rejected rather than hidden - it matters whether a star was never picked or
    picked and dropped."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    lo, hi = np.percentile(image[np.isfinite(image)], [30, 99.5])
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(image, origin="lower", cmap="gray_r", vmin=lo, vmax=hi)

    used = set(config.get("final_comps", []))
    rejected = set(config.get("rejected_comps", []))
    r = max(2.5 * config.get("median_fwhm", 6.0), 18)

    for _i, st in stars.iterrows():
        nm = str(st["name"])
        x, y = float(st.x), float(st.y)
        # The target's row is identified by its role, not by comparing its
        # name to target_name: stars.name is always "V" for the target (see
        # pick_stars), while target_name is the real object name. Comparing
        # the two directly meant this branch never matched, and the target
        # was drawn as an ordinary comparison star - or, once it also failed
        # to appear in `used`, as "rejected".
        is_target = str(st.get("role", "")).strip().lower() == "target"
        if is_target:
            col, lw, lab = "#d62728", 2.4, f"{nm}  (target)"
        elif nm in rejected:
            col, lw, lab = "#999999", 1.2, f"{nm}  rejected"
        elif nm in used or not used:
            col, lw, lab = "#1f77b4", 1.8, nm
        else:
            col, lw, lab = "#999999", 1.2, f"{nm}  rejected"
        ax.add_patch(Circle((x, y), r, fill=False, ec=col, lw=lw))
        ax.text(x + r * 1.15, y + r * 0.35, lab, color=col, fontsize=9,
                weight="bold" if is_target else "normal")

    n_used = len(used) if used else max(len(stars) - 1, 0)
    line1 = f"{target_name}   -   channel {channel}"
    line2 = f"target + {n_used} comparison stars used"
    if rejected:
        line2 += f"   ({len(rejected)} rejected by the quality tests)"
    title = line1 + "\n" + line2 + (("\n" + note) if note else "")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_channel(df, config, out_base, title, flip_marks=None, step_marks=None,
                 measured_noise=None, comp_base=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    good = df[df["mag_comp_minus_target"].notna()]
    if good.empty:
        return
    x = good["elapsed_min"]
    # The error bar shows the noise MEASURED on the comparison stars, not the
    # one the CCD equation predicts. The formula came out at 4-7 mmag here
    # while the stars themselves moved by 14 - so the formal bar was both
    # invisible on the plot and smaller than the real uncertainty, which is the
    # worse of the two faults. The larger of the two is used, so the bar is
    # never more confident than the data justify.
    err = good.get("mag_comp_minus_target_err")
    err = err.to_numpy(float) if err is not None else np.full(len(good), np.nan)
    if measured_noise is not None and np.isfinite(measured_noise):
        err = np.maximum(np.nan_to_num(err, nan=0.0), measured_noise / 1000.0)
    # Plotted as VARIABLE minus comparisons, which is the convention, and on
    # an inverted axis - so an eclipse is a dip. The engine stores the other
    # subtraction, comparisons minus variable; combined with the inverted axis
    # that put eclipses UPWARDS, which is how V1111 Cep came out as a hill.
    # The measurement is identical either way, only its sign is written down.
    y = -good["mag_comp_minus_target"]
    plt.figure(figsize=(10, 4))
    plt.errorbar(x, y, yerr=err,
                 fmt="o", markersize=4, elinewidth=1, capsize=2)
    if flip_marks:
        for t in flip_marks:
            plt.axvline(t, color="darkorange", ls="--", lw=1.2)
        plt.plot([], [], color="darkorange", ls="--", lw=1.2, label="meridian flip")
    if step_marks:
        for t in step_marks:
            plt.axvline(t, color="purple", ls=":", lw=1.6)
        plt.plot([], [], color="purple", ls=":", lw=1.6, label="throughput step")
    if flip_marks or step_marks:
        plt.legend(fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel("Elapsed time (minutes)")
    plt.ylabel("Variable - comparison ensemble (mag)")
    n_used = len(config.get("final_comps", []))
    n_rej = len(config.get("rejected_comps", []))
    sub = f"{n_used} comparison stars used"
    if n_rej:
        sub += f"   ({n_rej} rejected)"
    if measured_noise is not None and np.isfinite(measured_noise):
        sub += f"   -   error bars = measured noise, {measured_noise:.1f} mmag"
    plt.title(title + "\n" + sub, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{out_base}_light_curve.png", dpi=150)
    plt.close()

    tname = config["target_name"]
    for c in config["chosen_comps"]:
        col = f"{c}_mag"
        if col not in df.columns:
            continue
        sub = df[df[col].notna() & df[f"{tname}_mag"].notna()]
        if sub.empty:
            continue
        diff = sub[f"{tname}_mag"] - sub[col]      # variable minus this comparison
        err = None
        if f"{c}_magerr" in sub and f"{tname}_magerr" in sub:
            err = np.sqrt(sub[f"{c}_magerr"] ** 2 + sub[f"{tname}_magerr"] ** 2)
        status = ("REJECTED" if c in config.get("rejected_comps", []) else "used")
        plt.figure(figsize=(10, 4))
        plt.errorbar(sub["elapsed_min"], diff, yerr=err, fmt="o", markersize=4,
                     elinewidth=1, capsize=2)
        plt.gca().invert_yaxis()
        plt.xlabel("Elapsed time (minutes)")
        plt.ylabel(f"{tname} - {c} (mag)")
        sd = float(np.std(diff, ddof=1) * 1000) if len(diff) > 2 else float("nan")
        plt.title(f"{c} vs {tname}   scatter={sd:.1f} mmag   ({status})")
        plt.tight_layout()
        plt.savefig(f"{comp_base or out_base}_comp_{c}.png", dpi=150)
        plt.close()


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

SUMMARY_LEGEND = """
Legend
======
measurement noise (mmag)  How much the comparison stars move against each other,
                          from one frame to the next. Measured from the
                          difference between neighbours, so a step in
                          transparency or a slow drift does not inflate it. They
                          are supposed to be constant, so what is measured on
                          them is the noise of the measurement. This is what
                          decides which eclipse depth can be seen: a change
                          smaller than the noise will not be found.
                          Measured on the comparison stars and not on the target
                          - the target is the variable star, and its scatter is
                          mostly the change you are looking for.
target variation (mmag)   How much the target actually moved. On a constant star
                          this will be the size of the noise; on a variable star
                          it is the real change.
eclipse depth (mag)       How deep the star went. The baseline is the median of
                          the brighter half of the points, and the bottom is the
                          median of the faintest tenth - both are medians over
                          many points, so one outlying point does not move them.
                          Not the difference between the two extreme points.
variation vs comparison   The single number that says whether there is anything
                          here. How much smooth change there is in the target's
                          curve, against how much smooth change there is in the
                          curves of the comparison stars themselves, in the same
                          frames.

                          below 1.2    The target does what every star in the
                                       field does. There is no eclipse here.
                          1.2 to 2     Maybe. Not clean enough to claim.
                          above 2      A real change - no comparison star does
                                       this.

                          Why against the comparison stars and not against the
                          noise: photometric noise is not white. Airmass, seeing
                          and transparency wander slowly, and a slow wander is
                          exactly what a search for smooth change will find. In
                          one observation here all three channels looked like a
                          real change against white noise - and against the
                          comparison stars it turned out that the whole field
                          had wandered together, so there was nothing there.

                          The previous measure - target scatter divided by
                          comparison noise - was wrong. A faint target, or one
                          blended with a neighbour, is noisier than the
                          comparison stars for reasons that are not change, and
                          that noise went straight into the ratio. Across six
                          observations it declared five of them variable. One
                          is variable.

                          Needs at least 4 comparison stars. With fewer than
                          that a "-" is written.
frames                    How many frames were measured successfully, out of how
                          many went in.
comparison stars          How many survived the quality tests and entered the
                          ensemble.
position gap              The largest distance between the position set by the
                          plate-solution method and the one set by the anchor
                          method. Above 2 pixels - the methods measured
                          different things.
curve gap                 The scatter of the difference between the two light
                          curves. Zero means the two methods measured the same
                          star.
reliability               RELIABLE = the two methods agree. CHECK = they do not,
                          and the result needs to be checked by hand before it
                          is relied on.
"""


def write_summary(rows, out_dir, dropped_all):
    csv_path = os.path.join(out_dir, "SUMMARY.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["target", "channel", "measurement noise (mmag)",
                     "target variation (mmag)", "eclipse depth (mag)",
                     "variation vs comparison", "frames", "comparison stars",
                     "position gap (pixels)", "curve gap (mmag)",
                     "reliability", "notes"])
        for r in rows:
            wr.writerow([r["target"], r["channel"],
                         f"{r['noise']:.1f}" if np.isfinite(r["noise"]) else "-",
                         f"{r['scatter']:.1f}",
                         f"{r['rng']:.3f}" if np.isfinite(r["rng"]) else "-",
                         f"{r['snr']:.1f}" if np.isfinite(r["snr"]) else "-",
                         f"{r['n_used']}/{r['n_frames']}", r["n_comps"],
                         f"{r['pos_max']:.2f}" if np.isfinite(r["pos_max"]) else "-",
                         f"{r['curve']:.1f}" if np.isfinite(r["curve"]) else "-",
                         "RELIABLE" if r["ok"] else "CHECK", r["note"]])
        wr.writerow([])
        for line in SUMMARY_LEGEND.strip().splitlines():
            wr.writerow([line])

    txt_path = os.path.join(out_dir, "SUMMARY.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("Observation summary\n" + "=" * 70 + "\n\n")
        cur = None
        for r in rows:
            if r["target"] != cur:
                cur = r["target"]
                fh.write(f"\n{cur}\n" + "-" * 70 + "\n")
            nz = f"{r['noise']:.1f}" if np.isfinite(r["noise"]) else "-"
            rg = f"{r['rng']:.3f}" if np.isfinite(r["rng"]) else "-"
            sn = f"{r['snr']:.1f}" if np.isfinite(r["snr"]) else "-"
            fh.write(f"  channel {r['channel']}:  noise {nz} mmag  |  "
                     f"target variation {r['scatter']:.1f} mmag "
                     f"(depth {rg} mag)  |  "
                     f"variation vs comparison {sn}  |  "
                     f"frames {r['n_used']}/{r['n_frames']}  |  "
                     f"{r['n_comps']} comparison stars  |  "
                     f"{'RELIABLE' if r['ok'] else 'CHECK'}")
            if r["note"]:
                fh.write(f"   [{r['note']}]")
            fh.write("\n")
        if dropped_all:
            fh.write("\n\nRejected frames\n" + "-" * 70 + "\n")
            for t, name, why in dropped_all:
                fh.write(f"  {t}: {name}  -  {why}\n")
        fh.write("\n" + SUMMARY_LEGEND)
    return csv_path, txt_path


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Measure a whole night in one command.")
    ap.add_argument("--folder", default=None, help="folder with the downloaded frames")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--star_list", default=None,
                    help="optional: your own star list; overrides the automatic choice")
    ap.add_argument("--n_comps", type=int, default=12)
    ap.add_argument("--k_aperture", type=float, default=None)
    ap.add_argument("--no_pair_cap", dest="pair_cap", action="store_false",
                    help="do not look for a companion too close for the star "
                         "finder to split, and do not cap the aperture on one")
    ap.add_argument("--pair_contam", type=float, default=0.01,
                    help="how much of a close companion's light the target's "
                         "aperture may carry (default 0.01 = 1%%). The "
                         "aperture is resized every frame to hold this.")
    ap.add_argument("--pair_min_flux", type=float, default=0.5,
                    help="drop a frame when holding the contamination limit "
                         "would leave less than this fraction of the target's "
                         "own light inside the aperture (default 0.5)")
    ap.add_argument("--pair_sep_px", type=float, default=None,
                    help="separation in px of a companion you already know "
                         "about (from Aladin or Gaia). The aperture is capped "
                         "at half of it on every star. Use when the frames are "
                         "too shallow for the pair to be found in the pixels.")
    ap.set_defaults(pair_cap=True)
    ap.add_argument("--k_ann_in", type=float, default=None)
    ap.add_argument("--k_ann_out", type=float, default=None)
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--ron", type=float, default=None)
    ap.add_argument("--dark", type=float, default=0.0)
    ap.add_argument("--channels", default="R,G,B")
    ap.add_argument("--auto_radii", action="store_true", default=None,
                    help="measure the best aperture and sky annulus instead of "
                         "assuming them (default: asked once at the start)")
    ap.add_argument("--no_auto_radii", dest="auto_radii", action="store_false")
    ap.add_argument("--per_frame_fwhm", action="store_true", default=True,
                    help="size each star's aperture from its own FWHM measured in "
                         "each frame (default)")
    ap.add_argument("--fixed_fwhm", dest="per_frame_fwhm", action="store_false",
                    help="use one FWHM, from the reference frame, for the whole run")
    ap.add_argument("--limit_pos_px", type=float, default=2.0)
    ap.add_argument("--limit_curve_mmag", type=float, default=10.0)
    args = ap.parse_args()

    print("=" * 70)
    print("Run_Observation - one command for a whole night")
    # Which file is actually running, and when it was last changed. Without
    # this there is no way to tell from the output whether a fix is in the
    # version that just ran - the same script exists in several folders, and
    # an editor or a stale bytecode cache can quietly serve an older one.
    _me = os.path.abspath(__file__)
    try:
        import datetime
        _when = datetime.datetime.fromtimestamp(
            os.path.getmtime(_me)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        _when = "unknown"
    print(f"  running: {_me}")
    print(f"  last changed: {_when}")
    print("=" * 70)

    if args.folder is None:
        args.folder = prompt_path("Enter path to the folder with the frames",
                                      must_exist="dir")
    if args.output_dir is None:
        args.output_dir = prompt_path(
            "Enter path to the folder where the results should be saved",
            default=os.path.join(args.folder, "Results"))
    os.makedirs(args.output_dir, exist_ok=True)

    targets, skipped = scan_folder(args.folder)
    if not targets:
        raise SystemExit("No 3-plane colour FITS frames with an OBJECT keyword found.")
    print(f"\nFound {len(targets)} target(s):")
    for t, ps in sorted(targets.items()):
        print(f"   {t:38s} {len(ps):3d} frames")

    # The note about target.txt belongs HERE, next to the question about the
    # aperture - that is the moment the answer is useful. It first sat further
    # down, after the radii had already been chosen, where it could not
    # possibly help anyone.
    if not any(read_target_file([os.path.dirname(ps[0]), args.folder], t,
                                n_targets=len(targets), verbose=False)
               for t, ps in targets.items()):
        print("\n" + TARGET_FILE_HINT + "\n")

    # One question instead of three. Asking for an aperture and then quietly
    # overriding it with a scan - which is what this used to do - wastes the
    # answer and hides what actually got used.
    if args.auto_radii is None:
        # An answer that is neither yes nor no is asked again, not taken as
        # no. A Hebrew keyboard put a "מ" here and the run silently went
        # to manual radii - the one outcome nobody chose.
        while True:
            try:
                ans = input("Choose the aperture and sky annulus automatically, "
                            "by measuring what works best? [Y/n]: ").strip().lower()
            except EOFError:
                ans = ""
            if ans in ("", "y", "yes"):
                args.auto_radii = True
                break
            if ans in ("n", "no"):
                args.auto_radii = False
                break
            print(f"    '{ans}' is not y or n - please answer y or n "
                  f"(check the keyboard language)")
    if args.auto_radii:
        print("  -> radii will be measured per target and per channel")
        if args.k_aperture is None:
            args.k_aperture = 1.5      # only a starting point for the scan
        if args.k_ann_in is None:
            args.k_ann_in = 3.0
        if args.k_ann_out is None:
            args.k_ann_out = 5.0
    else:
        if args.k_aperture is None:
            args.k_aperture = prompt_float("Aperture radius = k x FWHM -> enter k", 2.0)
        if args.k_ann_in is None:
            args.k_ann_in = prompt_float("Sky annulus INNER radius = k x FWHM -> enter k", 3.0)
        if args.k_ann_out is None:
            args.k_ann_out = prompt_float("Sky annulus OUTER radius = k x FWHM -> enter k", 5.0)
    if args.gain is None:
        args.gain = prompt_float("Camera gain (electrons per ADU)", 1.0)
    if args.ron is None:
        args.ron = prompt_float("Camera Read Out Noise RON (electrons)", 3.7)

    wanted = [c.strip().upper() for c in args.channels.split(",") if c.strip()]
    limits = dict(pos_px=args.limit_pos_px, curve_mmag=args.limit_curve_mmag)
    summary, dropped_all = [], []
    target_dirs = {}


    for tname in sorted(targets):
        print(f"\n{tname}  -  {len(targets[tname])} frames")
        keep, drop = classify_frames(targets[tname])
        for d in drop:
            dropped_all.append((tname, os.path.basename(d["path"]), d["why"]))
            print(f"  DROP {os.path.basename(d['path'])[-30:]}  -  {d['why']}")
        n_flip = sum(1 for k in keep if k.get("flipped"))
        if n_flip:
            print(f"  {n_flip} frame(s) after a meridian flip - measured and marked, not dropped")
        if len(keep) < 5:
            print("  too few usable frames, skipping this target")
            continue

        # Everything a target produces - measurements, light curves and its own
        # summary - goes into ONE folder named after it, so nothing is split
        # between here and there and nothing from two targets can collide.
        tdir = os.path.join(args.output_dir, "Output_" + safe_name(tname))
        os.makedirs(tdir, exist_ok=True)
        prefix = ""
        target_dirs[tname] = tdir

        # Corrections left beside the frames, if any. Absent - the normal
        # case - this is None and nothing below it changes anything.
        over = read_target_file(
            [os.path.dirname(targets[tname][0]), args.folder], tname,
            n_targets=len(targets))
        over_radec, over_note = None, ""
        if over:
            print(f"  reading {os.path.basename(over['source'])}")
            _ = over
            if over.get("radec"):
                over_radec = over["radec"]
                cc = SkyCoord(over_radec[0], over_radec[1], unit=(u.deg, u.deg))
                print(f"    coordinates {cc.to_string('hmsdms', sep=':', precision=2)}"
                      f" - used as given")
                over_note = "coordinates supplied by the observer"

        # Always judged on the green plane, whichever channels are being
        # measured. Green carries the most signal on this camera and is the
        # quietest, so it ranks the frames most reliably - and, more
        # importantly, it makes the choice the same every time: ranking on
        # whichever channel happened to be asked for first meant that running
        # "--channels G" and "--channels R,G,B" picked different reference
        # frames and produced different numbers from identical data.
        # Taking this off the screen must not lose it: which frame the
        # whole run was measured against is worth being able to look up, so
        # it travels into the note and reaches SUMMARY.txt.
        ref_note = ""
        try:
            with quiet() as buf:
                ref_idx = choose_reference(keep, plane=1)
            ref_note = " ".join(buf.getvalue().split()).strip()
        except Exception as e:
            print(f"    could not rank the frames ({e}); using the first")
            ref_idx = 0
        ref_hdr = keep[ref_idx]["header"]
        # The companion is a property of the sky, not of the colour channel,
        # so it is looked for once and the answer serves all three. The three
        # planes are summed for it: on one plane this pair is missed and on
        # the sum it is found (see detect_close_pair).
        pair, deep_img = None, None

        for plane, cname in CHANNELS:
            if cname not in wanted:
                continue
            print(f"  {cname} ...", end="", flush=True)
            ref_img = fits.getdata(keep[ref_idx]["path"])[plane].astype(float)
            try:
                if args.star_list:
                    stars = stars_from_list(args.star_list)
                    snap, fwhm = float("nan"), float(ref_hdr.get("FWHM", 5.0))
                    note_src = "star list supplied"
                else:
                    stars, snap, fwhm, blend_note = pick_stars(
                        ref_img, ref_hdr, n_comps=args.n_comps,
                        radec=over_radec)
                    note_src = blend_note
                    if args.pair_cap and pair is None:
                        if deep_img is None:
                            deep_img = fits.getdata(
                                keep[ref_idx]["path"]).astype(float).sum(axis=0)
                        pair = detect_close_pair(
                            deep_img, float(stars.x[0]), float(stars.y[0]), fwhm)
            except Exception as e:
                print(f"    star selection failed: {e}")
                continue
            if np.isfinite(snap):
                # What this distance means: the 12 rows the header is stale
                # by (YORGSUBF, see field_offset), not a bad pointing and not
                # a bad plate solution. It is the same on every frame of every
                # field, so it cancels in the measurement - the target is
                # snapped onto the star and every frame is placed relative to
                # that. Coordinates the observer read off these same frames
                # carry the same 12 rows and land on the star directly.
                note_src = ((note_src + "; ") if note_src else "") + (
                    f"supplied coordinates used as given, nearest light "
                    f"{snap:.2f} px away" if over_radec is not None else
                    f"target found {snap:.2f} px from the recorded coordinates")

            pair_sep = pair_ratio = None
            if args.pair_sep_px:
                pair_sep, pair_ratio = float(args.pair_sep_px), 0.2
            if pair is not None:
                sep_px, dmag = pair
                pair_sep, pair_ratio = sep_px, 10 ** (-0.4 * dmag)
                r_med, kept_med = pair_aperture_limit(
                    sep_px, pair_ratio, fwhm, args.pair_contam)
                note_src = ((note_src + "; ") if note_src else "") + (
                    f"companion {sep_px:.2f} px away, {dmag:.2f} mag fainter, "
                    f"unresolved by the star finder - aperture follows the "
                    f"seeing under a {args.pair_contam * 100:.0f}% "
                    f"contamination limit ({r_med:.2f} px at the median, "
                    f"{kept_med * 100:.0f}% of the target's light)")
            pair_cfg = dict(pair_sep_px=pair_sep, pair_ratio=pair_ratio,
                            pair_contam=args.pair_contam,
                            pair_min_flux=args.pair_min_flux)
            cfg0 = dict(median_fwhm=fwhm, k_aperture=args.k_aperture,
                        k_ann_in=args.k_ann_in, k_ann_out=args.k_ann_out,
                        centroid_box=8, **pair_cfg)
            k_use, kin_use, kout_use = args.k_aperture, args.k_ann_in, args.k_ann_out
            if args.auto_radii:
                try:
                    k_use, kin_use, kout_use, _v = choose_radii(keep, plane, stars, cfg0)
                except Exception as e:
                    print(f"    radius scan failed ({e}); using the values given")

            cfg = dict(median_fwhm=fwhm, r_aperture=k_use * fwhm,
                       r_ann_in=kin_use * fwhm, r_ann_out=kout_use * fwhm,
                       k_aperture=k_use, k_ann_in=kin_use,
                       k_ann_out=kout_use, gain=args.gain, ron=args.ron,
                       dark=args.dark, centroid_box=8, known_mags={},
                       per_frame_fwhm=args.per_frame_fwhm, fwhm_box=15,
                       **pair_cfg)

            # The observer's corrections, applied last so they win over
            # anything measured, and only for this target. The note is rebuilt
            # per channel - appending to the target's note instead repeated it
            # once in G and twice in B.
            radii_note = ""
            if over_radec is not None:
                cfg["lock_target"] = True
            if over:
                r_ap = over.get("r_ap")
                r_in, r_out = over.get("r_in"), over.get("r_out")
                if over.get("k_ap") is not None:
                    r_ap = over["k_ap"] * fwhm
                if over.get("k_in") is not None and over.get("k_out") is not None:
                    r_in, r_out = over["k_in"] * fwhm, over["k_out"] * fwhm
                # An aperture without an annulus, or the other way round, is
                # a half-instruction; fill the missing half from the measured
                # radii so the pair is always consistent.
                if r_ap is not None or r_in is not None:
                    if r_ap is None:
                        r_ap = k_use * fwhm
                    if r_in is None or r_out is None:
                        r_in, r_out = kin_use * fwhm, kout_use * fwhm
                    if not (0 < r_ap < r_in < r_out):
                        print(f"    radii {r_ap:.1f}/{r_in:.1f}/{r_out:.1f} px are "
                              f"not increasing - ignored, measured radii used")
                    else:
                        cfg["fixed_radii_px"] = (float(r_ap), float(r_in), float(r_out))
                        radii_note = (f"target radii supplied "
                                      f"{r_ap:.1f}/{r_in:.1f}/{r_out:.1f} px")

            with quiet():
                df_w, cfg_w = measure_channel(keep, plane, stars, ref_idx, cfg, "wcs")
                df_a, cfg_a = measure_channel(keep, plane, stars, ref_idx, cfg, "anchor")
            if df_w.empty:
                print("    nothing measured, skipping")
                continue

            # No step detection here any more. It was fragile - on the same
            # night it put the step in three different places in the three
            # channels, once inside the eclipse egress - and the noise figure
            # no longer needs it: measuring scatter between consecutive frames
            # ignores a level step by construction. A step is obvious in the
            # light curve anyway.
            segs = None
            step_note = ""
            tmin_arr = df_w["elapsed_min"].to_numpy(float)

            noise = noise_mmag(df_w, cfg_w.get("final_comps", []), segs)

            # The four things actually looked at - the light curve, the
            # spreadsheet, the star chart and the summary - sit at the top of
            # the folder. Everything else is diagnostic: a plot per comparison
            # star, the second positioning method, the quality tables. Those go
            # into Details, because 57 files in one flat folder buries the 8
            # that matter.
            details = os.path.join(tdir, "Details")
            os.makedirs(details, exist_ok=True)
            base = os.path.join(tdir, f"{prefix}{cname}")
            dbase = os.path.join(details, f"{prefix}{cname}")
            with quiet():
                write_outputs(df_w, cfg_w, dbase + "_wcs", ["csv", "xlsx"])
            try:
                write_lightcurve_sheet(df_w, cfg_w,
                                       base + "_light_curve.xlsx",
                                       tname, cname, measured_noise=noise)
            except Exception as e:
                print(f"    could not write the light-curve sheet: {e}")
            flips = [float(r["elapsed_min"]) for _i, r in df_w.iterrows() if r.get("flipped")]
            step_marks = []
            plot_channel(df_w, cfg_w, base + "_wcs",
                         f"{tname}  -  channel {cname}  (positions from the plate solution)",
                         flip_marks=flips, step_marks=step_marks,
                         measured_noise=noise, comp_base=dbase + "_wcs")
            if not df_a.empty:
                with quiet():
                    write_outputs(df_a, cfg_a, dbase + "_anchor", ["csv"])
            # the engine puts its quality table beside the main one; it belongs
            # with the diagnostics
            qual = base + "_wcs_comp_stars.csv"
            if os.path.exists(qual):
                try:
                    os.replace(qual, dbase + "_wcs_comp_stars.csv")
                except Exception:
                    pass

            # one chart per target showing which stars were actually used, so
            # the automatic choice can be checked by eye. Stars are picked per
            # channel, so the chart says which channel it is drawn for.
            chart = os.path.join(tdir, "stars_used.png")
            if not os.path.exists(chart):
                try:
                    other = [c for _p, c in CHANNELS if c in wanted and c != cname]
                    note = ("stars are chosen per channel; "
                            f"{', '.join(other)} may differ slightly") if other else ""
                    finder_chart(ref_img, stars, cfg_w, chart,
                                 cfg_w["target_name"], cname, note=note)
                except Exception as e:
                    print(f"    could not draw the star chart: {e}")

            cmp_ = compare_routes(df_w, df_a, list(stars.name), limits)
            sc = scatter_mmag(df_w)
            note = note_src
            for extra in (over_note, radii_note, ref_note):
                if extra:
                    note = (note + "; " if note else "") + extra
            if step_note:
                note = (note + "; " if note else "") + step_note
            if n_flip:
                note = (note + "; " if note else "") + f"{n_flip} frames after a flip"
            rng = target_depth_mag(df_w)
            cr, r_t, r_c, n_c = change_ratio(df_w, cfg_w.get("final_comps", []))
            summary.append(dict(target=tname, channel=cname, scatter=sc,
                                noise=noise, rng=rng,
                                snr=cr, smooth_t=r_t, smooth_c=r_c,
                                n_used=cmp_.get("n_frames_wcs", 0), n_frames=len(keep),
                                n_comps=len(cfg_w.get("final_comps", [])),
                                pos_max=cmp_["pos_max"], curve=cmp_["curve_mmag"],
                                ok=cmp_["ok"], note=note))
            print(" done", flush=True)

    if not summary:
        raise SystemExit("Nothing was measured.")
    # each target's summary lives with that target's results
    for tname, tdir in target_dirs.items():
        rows_t = [r for r in summary if r["target"] == tname]
        drops_t = [d for d in dropped_all if d[0] == tname]
        if rows_t:
            write_summary(rows_t, tdir, drops_t)
    # and, only when there is more than one, a combined one above them
    if len(target_dirs) > 1:
        csv_path, txt_path = write_summary(summary, args.output_dir, dropped_all)
    else:
        only = list(target_dirs.values())[0] if target_dirs else args.output_dir
        csv_path = os.path.join(only, "SUMMARY.csv")
        txt_path = os.path.join(only, "SUMMARY.txt")

    # The summary is written, not printed. Every number in it is in
    # SUMMARY.txt and SUMMARY.csv beside the light curves, which is where it
    # gets read from; on the screen it only pushed the two lines that matter -
    # what ran and where it went - off the top.
    print("")
    for tname, tdir in sorted(target_dirs.items()):
        print(f"  {tname}  ->  {os.path.abspath(tdir)}")


if __name__ == "__main__":
    main()