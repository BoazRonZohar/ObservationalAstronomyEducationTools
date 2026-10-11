# -*- coding: utf-8 -*-
"""
Find_Moving_Objects.py
============================

Find faint moving objects by TRACKING them, not by detecting them.

The third of the moving-object tools, and the one built for objects too faint
to stand out in any single frame. The other two ask each frame "is there
something here?" and then try to join the answers. This one asks the whole
night at once.

Why the other two are not enough
--------------------------------

On thirteen 60 s frames from the Faulkes 2 m, an asteroid the observer found
by eye in 2011 sits at a signal-to-noise of 5 to 12 per frame. It is one of
about seven thousand marginal detections. There is no threshold that keeps it
and rejects the noise: at 4 sigma it is buried among 578 spurious detections
per frame, and at 6 sigma it is gone. Both earlier tools failed on it, in
both directions.

The method, and where it came from
----------------------------------

Every step below except the stacking was proposed by the observer, who found
these objects by hand in 2011 and knows how it is actually done.

  1. detect, with a LOCAL background
        A global threshold is set by the brightest part of the frame. Next to
        a saturated star it is far above anything faint - which is exactly
        where this asteroid sits, and why it was invisible to the first
        attempt. Subtracting a background measured on a 48 px grid makes the
        threshold mean the same thing everywhere.

  2. keep only things with the instrument's WIDTH
        A real object has a PSF: it is as wide as every star in the frame. A
        cosmic ray is one pixel wide. A satellite and a meteor are lines with
        no width at all. Measuring the SHORT axis and requiring it to match
        the stars removes nine of every ten detections in one step - and it is
        an absolute test against the instrument, not a shape ratio.

  3. three epochs, not two, and the middle one must agree
        Two points always define a line, so any early detection paired with
        any late one is a "candidate" and millions of them are nonsense. Three
        points test whether the motion is UNIFORM, which is a real property of
        a solar system object over twenty minutes and not a property of noise.
        With ~600 detections per frame the chance a random one lands within
        2 px of the predicted midpoint is about 1 in 500, so this removes
        99.8% of the accidental pairs before any real work is done. It also
        removes cosmic rays at the root: a cosmic ray exists in one frame, so
        it cannot take part in a triple at all.

  4. blink with a STRIDE
        Between neighbouring frames this asteroid moves 1.8 px, less than the
        3.2 px width of a star, so it does not separate from itself. Between
        frames four apart it moves 6 px and separates cleanly.

  5. THEN stack along the velocity the triple implies
        This is the part the earlier tools lacked. Instead of asking "is there
        light at the predicted place in each frame" - a question noise answers
        yes to often enough - shift all the frames by where the object should
        have travelled and add them. A real object piles up: signal-to-noise
        gains the square root of the number of frames, 3.6x here, so 6 becomes
        22. An accidental triple has nothing along the rest of its line, so
        its stack stays empty.

Steps 2 and 3 are what make step 5 affordable. Without them there would be
two thousand blind velocities to try; with them, only the few dozen the data
itself supports.

Known answer
------------

The 13 frames of 2P/Encke from 2011-07-05 (Faulkes North) contain an asteroid
the observer discovered by eye: 18.8 arcsec/hour, present in all 13 frames,
straight to 0.18 arcsec, at RA 343.8985 Dec -8.7898 at MJD 55747.50356. Any
change to this file must still find it.

What you get for each track
---------------------------

The rate and direction, how many frames it appeared in, how far it moved
against its own scatter, and its position at the first frame. Then the
catalogue answer: SkyBoT is asked what known solar system object was at that
place at that moment.

When it names one, the next line is a link straight to that object's entry in
the Minor Planet Center database - orbit, observation history, ephemeris. A
numbered object is linked by its number, which never changes; an unnumbered
one by its provisional designation.

When it names nothing, there is no link, and that is the interesting case.

----------------------------------------------------------------------
Created by: Dr. Boaz Ron Zohar
Affiliation: Kinneret Observatory
Member of the LCO Global Sky Partners programme
Part of Observational Astronomy Education Tools
https://github.com/BoazRonZohar/ObservationalAstronomyEducationTools
"""

import os
import re
import sys
import glob
import time
import argparse
import warnings

import numpy as np

NL = chr(10)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from astropy.io import fits                                      # noqa: E402
from astropy.wcs import WCS, FITSFixedWarning                    # noqa: E402
from astropy.wcs.utils import proj_plane_pixel_scales            # noqa: E402
from astropy.time import Time                                    # noqa: E402
from astropy.coordinates import SkyCoord                         # noqa: E402
import astropy.units as u                                        # noqa: E402
from astropy.utils.exceptions import AstropyWarning              # noqa: E402
from astropy.stats import SigmaClip, sigma_clipped_stats         # noqa: E402
from photutils.background import Background2D, MedianBackground  # noqa: E402
from photutils.detection import DAOStarFinder                    # noqa: E402
from scipy.ndimage import (affine_transform, gaussian_filter,    # noqa: E402
                           label, find_objects)
from scipy.spatial import cKDTree                                # noqa: E402

warnings.filterwarnings("ignore", category=AstropyWarning)
warnings.filterwarnings("ignore", category=FITSFixedWarning)


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------

def image_and_header(path):
    """One 2-D image and its header, whatever the file holds.

    A three-plane colour frame gives up its green plane - half the pixels of
    the Bayer mosaic rather than a quarter, so the sharpest of the three. A
    frame that is already one plane is used as it stands."""
    with fits.open(path, memmap=False) as hd:
        h, d = hd[0].header, hd[0].data
        if d is None and len(hd) > 1:
            h, d = hd[1].header, hd[1].data
        if d is None:
            raise RuntimeError("no image data")
        img = np.asarray(d[1] if d.ndim == 3 else d, dtype=float)
    return img, h


def build_wcs(header, fix_crpix=True):
    """The plate solution, with the stale crop rows taken out of it.

    MaxIm crops YORGSUBF rows after PinPoint has solved the frame and leaves
    CRPIX2 on the uncropped centre. Sky coordinates read straight from such a
    header sit about 15 arcsec from the truth, which does not matter for
    finding a track - the error is the same in every frame - but matters
    completely for saying WHERE the object was, which is the whole point.

    The correction is applied whenever YORGSUBF is not zero, and the run says
    so (see `crop_note`): a frame read out as a subframe and solved AFTER the
    readout already has the right CRPIX2, and on such a frame the correction
    moves every position by YORGSUBF rows. Nothing in the header tells the two
    cases apart, so the observer has to be told which one was assumed."""
    h = header.copy()
    for k in list(h):
        if k.startswith(("TR1_", "TR2_")):
            del h[k]
    h["NAXIS"] = 2
    for k in ("NAXIS3", "NAXIS4"):
        if k in h:
            del h[k]
    if fix_crpix:
        try:
            h["CRPIX2"] = float(h["CRPIX2"]) - float(h.get("YORGSUBF", 0) or 0)
        except (KeyError, TypeError, ValueError):
            pass
    return WCS(h)


def crop_note(header):
    """A line for the log when build_wcs has moved CRPIX2, else None."""
    try:
        y = float(header.get("YORGSUBF", 0) or 0)
    except (TypeError, ValueError):
        return None
    if not y:
        return None
    return (f"CRPIX2 lowered by YORGSUBF = {y:.0f} rows (MaxIm crop after the "
            f"plate solution). If these frames were read out as a subframe "
            f"and solved afterwards, run with --keep_crpix: the positions "
            f"below would otherwise be {y:.0f} rows off.")


def has_plate_solution(header):
    """True when the header holds a celestial WCS in any of its forms.

    Only CD1_1 used to count. A solution written as CDELT and PC, or as CDELT
    and CROTA2, was not recognised: the frames were skipped as unsolved, and
    where one did get through, the pixel scale fell back to 1.0 and every rate
    in arcsec/hour was really in pixels/hour."""
    if not all(k in header for k in ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2")):
        return False
    try:
        return bool(build_wcs(header, fix_crpix=False).has_celestial)
    except Exception:
        return False


def pixel_scale_of(w):
    """Arcseconds per pixel, from any form of the plate solution."""
    try:
        return float(np.mean(proj_plane_pixel_scales(w.celestial)) * 3600.0)
    except Exception:
        return float("nan")


def exposure_s(h):
    for k in ("EXPTIME", "EXPOSURE"):
        try:
            v = float(h[k])
            if v >= 0:
                return v
        except (KeyError, TypeError, ValueError):
            pass
    return 0.0


def frame_mjd(h):
    """The MIDDLE of the exposure, as an MJD in UTC.

    MJD-OBS and DATE-OBS mark the START of the exposure, at LCO and in MaxIm
    alike, while the position measured on the frame is where the object was
    halfway through it. Taking the start put every position half an exposure
    late - for a near-Earth object at 300 arcsec/hour in a 60 s frame, 2.5
    arcsec - and the catalogue was asked about the wrong moment by the same
    amount. DATE-AVG is not used: on the 2011 Faulkes frames it is wrong in 8
    of 13 files."""
    start = float("nan")
    for k in ("MJD-OBS", "MJD_OBS"):
        if k in h:
            try:
                start = float(h[k])
                break
            except (TypeError, ValueError):
                pass
    if not np.isfinite(start):
        for k in ("DATE-OBS", "DATE_OBS"):
            if k not in h:
                continue
            v = str(h[k]).strip()
            # a date with the time in a keyword of its own
            if "T" not in v and len(v) <= 10:
                t = str(h.get("TIME-OBS", h.get("UT", h.get("UTSTART", "")))).strip()
                if t:
                    v = f"{v}T{t}"
            try:
                start = float(Time(v, format="isot", scale="utc").mjd)
                break
            except Exception:
                pass
    if not np.isfinite(start):
        return float("nan")
    return start + 0.5 * exposure_s(h) / 86400.0


def object_name(h):
    """OBJECT, with any per-frame scheduling serial removed."""
    return re.sub(r"\s*:\s*\d+\s*$", "", str(h.get("OBJECT", "")).strip()).strip()


def frame_header(path):
    """The header that goes with the image: the primary, or the first
    extension when the primary is empty (LCO's compressed files)."""
    h = fits.getheader(path)
    if h.get("NAXIS", 0) == 0:
        try:
            h = fits.getheader(path, 1)
        except Exception:
            pass
    return h


def scan_folder(folder, skip=None):
    """The runs in a folder: frames of one target, one camera set-up, one night.

    Frames used to be grouped by OBJECT alone and kept in FILE NAME order.
    Neither is safe. Two nights on the same target became one "run" with a
    gap of a day in it, and a different binning or camera on the same target
    put images of two sizes into one stack, which stopped the run with an
    error. Names that do not sort by time - a counter that wraps, a re-numbered
    copy - put the quick look's "spread through the run" sample out of order
    and drew the pictures with the wrong frame under each time. So frames are
    sorted by their own clock, and a run is split wherever the image size or
    the pixel scale changes or the clock jumps by more than two hours.

    Folders whose name starts with "result" are not entered: that is where
    this tool and the others in the collection write."""
    skip = os.path.abspath(skip) if skip else None
    files = []
    for root, dirs, _f in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.lower().startswith("result")
                   and (skip is None or os.path.abspath(os.path.join(root, d)) != skip)]
        for pat in ("*.fits", "*.fts", "*.fit", "*.FITS", "*.FTS", "*.FIT",
                    "*.fits.fz"):
            files += glob.glob(os.path.join(root, pat))
    rows = {}
    unsolved = 0
    for p in sorted(set(files)):
        try:
            h = frame_header(p)
        except Exception:
            continue
        if not has_plate_solution(h):
            unsolved += 1
            continue
        obj = object_name(h)
        if not obj:
            continue
        mjd = frame_mjd(h)
        if not np.isfinite(mjd):
            continue
        nx, ny = int(h.get("NAXIS1", 0)), int(h.get("NAXIS2", 0))
        scale = pixel_scale_of(build_wcs(h, fix_crpix=False))
        key = (obj, nx, ny, round(scale, 2) if np.isfinite(scale) else 0.0)
        rows.setdefault(key, []).append((p, str(h.get("DATE-OBS", "")), mjd))
    if unsolved:
        print(f"  {unsolved} FITS file(s) without a plate solution were left out")
    runs = {}
    for (obj, nx, ny, scale), items in sorted(rows.items()):
        # one moment appears once: colour planes of the same exposure are merged
        best = {}
        for path, when, mjd in items:
            key = when or path
            base = os.path.basename(path).lower()
            rank = 0 if "green" in base else (1 if not any(
                t in base for t in ("red_of", "blue_of")) else 2)
            if key not in best or rank < best[key][1]:
                best[key] = (path, rank, mjd)
        seq = sorted((m, p) for p, _r, m in best.values())
        pieces, cur = [], [seq[0]]
        for a, b in zip(seq, seq[1:]):
            if (b[0] - a[0]) * 24.0 > 2.0:
                pieces.append(cur)
                cur = []
            cur.append(b)
        pieces.append(cur)
        for piece in pieces:
            runs.setdefault(obj, []).append(
                dict(obj=obj, size=(nx, ny), scale=scale,
                     paths=[p for _m, p in piece],
                     mjds=[m for m, _p in piece]))
    out = {}
    for obj, lst in runs.items():
        for r in lst:
            name = obj
            if len(lst) > 1:
                day = Time(r["mjds"][0], format="mjd").isot[:16].replace("T", " ")
                name = (f"{obj} [{day} UTC, {r['size'][0]}x{r['size'][1]} px, "
                        f"{r['scale']:.2f} arcsec/px]")
            out[name] = r["paths"]
    return out


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def short_axis(im, x, y, psf_px):
    """(long, short) axis of a source, as FWHM in pixels.

    Measured on the frame BEFORE it is smoothed, with second moments weighted
    by a Gaussian of the stars' own width, and the weight then divided back
    out (for a Gaussian source this is exact: 1/s_measured^2 = 1/s_true^2 +
    1/s_weight^2).

    The earlier version took plain moments of a 13x13 box of the SMOOTHED
    frame. Smoothing by one pixel gives a cosmic ray the width of a 2.4 px
    star before anything is measured, and plain moments over a box that size
    are set by the noise in its corners rather than by the source: on a
    synthetic frame with 60 cosmic rays and stars of 3.2 px, all 60 cosmic
    rays passed the width test. With the weighted moments on the unsmoothed
    frame none of them do, and 57 to 59 of 60 stars at S/N 5 to 20 still pass
    - also at a star width of 2.2 px."""
    sw = psf_px / 2.3548
    box = int(np.ceil(3 * sw)) + 1
    xi, yi = int(round(x)), int(round(y))
    if xi - box < 0 or yi - box < 0 or xi + box >= im.shape[1] or yi + box >= im.shape[0]:
        return None
    c = np.asarray(im[yi - box:yi + box + 1, xi - box:xi + box + 1], dtype=float)
    gy, gx = np.mgrid[-box:box + 1, -box:box + 1].astype(float)
    mx, my = x - xi, y - yi
    for _ in range(3):
        w = np.exp(-((gx - mx) ** 2 + (gy - my) ** 2) / (2 * sw * sw))
        s = (c * w).sum()
        if s <= 0:
            return None
        mx, my = (c * w * gx).sum() / s, (c * w * gy).sum() / s
        if abs(mx) > box / 2 or abs(my) > box / 2:
            return None
    xx = (c * w * (gx - mx) ** 2).sum() / s
    yy = (c * w * (gy - my) ** 2).sum() / s
    xy = (c * w * (gx - mx) * (gy - my)).sum() / s
    tr, dt = xx + yy, xx * yy - xy * xy
    r = np.sqrt(max(tr * tr / 4 - dt, 0.0))
    out = []
    for m2 in (tr / 2 + r, tr / 2 - r):
        if m2 <= 0:
            out.append(0.0)
        elif m2 >= 0.98 * sw * sw:          # much wider than the weight
            out.append(7.0 * psf_px)
        else:
            out.append(2.3548 * np.sqrt(m2 * sw * sw / (sw * sw - m2)))
    return out[0], out[1]


def find_sources(img, fwhm, threshold):
    """DAOStarFinder, as (x, y, flux) arrays, on photutils 1.x to 3.x.

    photutils 3 renamed the centroid columns to x_centroid and y_centroid and
    moved the shape limits into ranges; the old names stopped the run with a
    KeyError on the first frame."""
    kw = dict(fwhm=fwhm, threshold=threshold, exclude_border=True)
    try:
        f = DAOStarFinder(sharpness_range=(0.05, 3.0),
                          roundness_range=(-1.5, 1.5), **kw)
    except TypeError:
        f = DAOStarFinder(sharplo=0.05, sharphi=3.0, roundlo=-1.5, roundhi=1.5, **kw)
    t = f(img)
    if t is None or len(t) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    xc = "x_centroid" if "x_centroid" in t.colnames else "xcentroid"
    yc = "y_centroid" if "y_centroid" in t.colnames else "ycentroid"
    return (np.asarray(t[xc], float), np.asarray(t[yc], float),
            np.asarray(t["flux"], float))


def saturation_level(h, img):
    """The level at which a pixel stops counting light.

    The header says it when it can: MAXLIN (MaxIm's linearity limit) or
    SATURATE (LCO's BANZAI). Only without either is the frame's own maximum
    used, as before - which marks a cosmic ray or a hot pixel as a saturated
    star whenever the frame holds no real one."""
    for k in ("MAXLIN", "SATURATE"):
        try:
            v = float(h[k])
            if v > 0:
                return 0.95 * v, k
        except (KeyError, TypeError, ValueError):
            pass
    return 0.85 * float(np.nanmax(img)), "0.85 x the frame maximum"


def saturated_zones(img, level, psf_px):
    """(x, y, radius) of every saturated region, radius scaled to its size.

    The zone used to be 90 px around every pixel above the level, whatever
    the star and whatever the camera: 27 arcsec at 0.3 arcsec/px, 67 at 0.74.
    A faint object passing a bright star was masked for most of the run. The
    zone now reaches as far again as the saturated core and its bleed trail
    extend, plus five star widths for the halo."""
    lab, n = label(img >= level)
    zones = []
    for i, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        ys, xs = sl
        cy, cx = 0.5 * (ys.start + ys.stop - 1), 0.5 * (xs.start + xs.stop - 1)
        half = 0.5 * max(ys.stop - ys.start, xs.stop - xs.start)
        zones.append((cx, cy, 2.0 * half + 5.0 * psf_px))
    return np.array(zones) if zones else np.zeros((0, 3))


def measure_psf(raws, level, guess=3.2):
    """The stars' FWHM in pixels, measured on the first frames themselves.

    It was a fixed 3.2 px - right for the Faulkes frames of 2011 and for no
    other telescope in particular. The width test below is a test AGAINST
    this number, so with a wrong one it either passes everything or rejects
    the stars themselves.

    Only sources found again in another of the first frames count. A first
    version measured everything bright in one frame, and on the 2011 Faulkes
    frames - a sparse field, a 2 m telescope, 60 s - the cosmic rays
    outnumbered the stars: it measured 1.10 px (0.33 arcsec) instead of about
    3.2, the width test then threw the stars away with the rays, and the
    known asteroid was not found. A cosmic ray is never in two frames. A hot
    pixel is, so anything narrower than one pixel is left out as well.

    `raws` is a list of (background-subtracted frame, its noise), all on the
    reference grid, the first one being measured."""
    raw0, rms0 = raws[0]
    x, y, f = find_sources(raw0, guess, 10.0 * rms0)
    if len(x) == 0:
        return float("nan"), 0
    if len(raws) > 1:
        again = np.zeros(len(x), bool)
        for r, rms in raws[1:]:
            xo, yo, _f = find_sources(r, guess, 10.0 * rms)
            if len(xo):
                d, _i = cKDTree(np.c_[xo, yo]).query(np.c_[x, y])
                again |= d <= 1.5
        x, y, f = x[again], y[again], f[again]
        if len(x) == 0:
            return float("nan"), 0
    tree = cKDTree(np.c_[x, y])
    widths = []
    for i in np.argsort(-f)[:300]:
        if len(tree.query_ball_point((x[i], y[i]), 4.0 * guess)) > 1:
            continue                                   # a neighbour
        xi, yi = int(round(x[i])), int(round(y[i]))
        if not (0 <= yi < raw0.shape[0] and 0 <= xi < raw0.shape[1]):
            continue
        pk = raw0[yi, xi]
        if pk < 20 * rms0 or pk > 0.5 * level:
            continue
        sh = short_axis(raw0, x[i], y[i], guess)
        if sh is None or sh[1] < 1.0:
            continue
        widths.append(np.sqrt(sh[0] * sh[1]))
    if len(widths) < 5:
        return float("nan"), len(widths)
    return float(np.median(widths)), len(widths)


def to_reference(w, ref_w, shape):
    """The affine map from the reference frame's pixels to this frame's.

    Frames used to be lined up by a SHIFT alone, taken at the centre. A
    rotation was ignored: 0.05 degrees moves the edge of a 2400 px LCO frame
    by a pixel, and a German mount that flips at the meridian turns the field
    by 180 degrees, after which no star in the second half of the night lands
    on itself and the static-star filter has nothing to recognise. The map is
    fitted to 25 points spread over the frame, through each frame's own plate
    solution, so rotation, a flip and a change of scale all come out of it."""
    ny, nx = shape
    gx, gy = np.meshgrid(np.linspace(0, nx - 1, 5), np.linspace(0, ny - 1, 5))
    gx, gy = gx.ravel(), gy.ravel()
    ra, de = ref_w.all_pix2world(gx, gy, 0)
    fx, fy = w.all_world2pix(ra, de, 0)
    A = np.c_[gx, gy, np.ones_like(gx)]
    cx = np.linalg.lstsq(A, fx, rcond=None)[0]
    cy = np.linalg.lstsq(A, fy, rcond=None)[0]
    return cx, cy


def align(img, cx, cy):
    """Resample a frame onto the reference grid (NaN where it has no data)."""
    # affine_transform works in (row, column): input = M @ output + offset
    M = np.array([[cy[1], cy[0]], [cx[1], cx[0]]])
    off = np.array([cy[2], cx[2]])
    return affine_transform(img, M, offset=off, order=1, mode="constant",
                            cval=np.nan)


def star_offset(ref_xy, xy, reach=250.0):
    """How far this frame's stars sit from the reference frame's, in pixels.

    The answer is the most common difference vector between the two lists of
    bright stars; None when no offset is supported by at least six stars."""
    if len(ref_xy) < 6 or len(xy) < 6:
        return None
    d = (xy[None, :, :] - ref_xy[:, None, :]).reshape(-1, 2)
    d = d[(np.abs(d) <= reach).all(axis=1)]
    if len(d) < 6:
        return None
    bins = np.arange(-reach, reach + 2.0, 2.0)
    hist, ex, ey = np.histogram2d(d[:, 0], d[:, 1], bins=[bins, bins])
    i, j = np.unravel_index(np.argmax(hist), hist.shape)
    guess = np.array([0.5 * (ex[i] + ex[i + 1]), 0.5 * (ey[j] + ey[j + 1])])
    near = d[np.hypot(*(d - guess).T) <= 2.0]
    if len(near) < 6:
        return None
    return np.median(near, axis=0), len(near)


FIX_CRPIX = True


def prepare(paths, psf_px, threshold, min_usable=5, verbose=True):
    """Align every frame, remove its local background, and find its sources.

    Returns the frames on the reference grid, their mid-exposure times, the
    detections, the reference plate solution, the scratch file and the star
    width that was used."""
    import tempfile
    state = dict(ref_w=None, ref_shape=None, level_from=None)
    store, mjds, det = None, [], []
    scratch = os.path.join(tempfile.gettempdir(),
                           f"movtrack_{os.getpid()}_{int(time.time() * 1000)}.dat")
    notes = set()

    def load(n, p):
        """Read, align and background-subtract one frame (None if unusable)."""
        try:
            img, h = image_and_header(p)
            w = build_wcs(h, fix_crpix=FIX_CRPIX)
            mjd = frame_mjd(h)
            if not np.isfinite(mjd):
                return None
        except Exception as e:
            if verbose:
                print(f"    skipped {os.path.basename(p)}: {e}")
            return None
        if FIX_CRPIX:
            note = crop_note(h)
            if note:
                notes.add(note)
        level, state["level_from"] = saturation_level(h, img)
        if state["ref_w"] is None:
            state["ref_w"], state["ref_shape"] = w, img.shape
        if img.shape != state["ref_shape"]:
            if verbose:
                rs = state["ref_shape"]
                print(f"    skipped {os.path.basename(p)}: {img.shape[1]}x"
                      f"{img.shape[0]} px, the run is {rs[1]}x{rs[0]}")
            return None
        cx, cy = to_reference(w, state["ref_w"], state["ref_shape"])
        for _try in range(2):
            aligned = align(img, cx, cy)
            cover = np.isfinite(aligned)
            fill = float(np.nanmedian(aligned)) if cover.any() else 0.0
            aligned = np.where(cover, aligned, fill)
            # A background measured on a grid, not one number for the whole
            # frame. This is the step that made the difference: next to a
            # saturated star a global threshold sits far above anything faint.
            bkg = Background2D(aligned, (48, 48), filter_size=(3, 3),
                               mask=~cover, sigma_clip=SigmaClip(sigma=3.0),
                               bkg_estimator=MedianBackground())
            raw = np.where(cover, aligned - bkg.background, 0.0)
            rms_raw = float(sigma_clipped_stats(raw[cover], sigma=3.0)[2])
            # The plate solutions are checked against the stars themselves.
            # Frames of one night do not all carry an equally good solution:
            # on the 2010 Faulkes comet frames some had a CD matrix and some
            # only a nominal CDELT/CROTA pointing, ~100 px apart, and lined up
            # through their headers the stars of one frame landed nowhere near
            # those of the next. Every star of such a frame then looked like a
            # mover. Each frame's bright stars are compared with the first
            # frame's; when they all sit off by the same amount, the frame is
            # moved onto them.
            gx, gy, gf = find_sources(raw, psf_px or 3.2, 15.0 * rms_raw)
            top = np.argsort(-gf)[:150]
            xy = np.c_[gx[top], gy[top]]
            if state.get("ref_stars") is None:
                state["ref_stars"] = xy
                break
            off = star_offset(state["ref_stars"], xy)
            if off is None or _try:
                if off is None and verbose and len(state["ref_stars"]) >= 6:
                    print(f"\n    {os.path.basename(p)}: its stars could not "
                          f"be matched to the first frame's - lined up by the "
                          f"plate solution alone")
                break
            (ox, oy), nmatch = off
            if np.hypot(ox, oy) <= 0.5:
                break
            if verbose:
                print(f"\n    {os.path.basename(p)}: plate solution "
                      f"{np.hypot(ox, oy):.1f} px off the stars "
                      f"({nmatch} stars agree) - lined up on the stars")
            cx = cx.copy(); cy = cy.copy()
            cx[2] += cx[0] * ox + cx[1] * oy
            cy[2] += cy[0] * ox + cy[1] * oy
        return dict(n=n, img=img, w=w, mjd=mjd, level=level, cx=cx, cy=cy,
                    cover=cover, raw=raw, rms_raw=rms_raw)

    def finish(fr):
        nonlocal store
        raw, cover, img = fr["raw"], fr["cover"], fr["img"]
        cx, cy = fr["cx"], fr["cy"]
        clean = gaussian_filter(raw, 1.0)
        # The threshold has to be measured on the image the finder actually
        # sees. Taking it from the background BEFORE smoothing made "4 sigma"
        # mean 13.5 sigma of the smoothed frame - the known asteroid peaks at
        # 11 sigma there and was thrown away by the very step meant to find it.
        rms = float(sigma_clipped_stats(clean[cover], sigma=3.0)[2])
        # The finder's default shape cuts exist to reject things that are not
        # stars - which includes comets. A coma is exactly the sort of soft,
        # extended source `sharplo` throws away, so the cuts are opened here.
        xs, ys, fl = find_sources(clean, psf_px, threshold * rms)
        # Saturated stars are rejected HERE, by name, and not left to the
        # finder's shape cuts. Those cuts used to remove them as a side effect
        # of removing anything that is not a neat point - and the same cuts
        # removed comets. Opening them for the comet let a single saturated
        # star flood the output with 178 fragments of itself. A bright star's
        # core and spikes are not a shape problem, they are a brightness
        # problem, so they are masked as one. They are found on the frame as
        # it was read, where the level means what the header says it means,
        # and carried to the reference grid.
        zones = saturated_zones(img, fr["level"], psf_px)
        if len(zones):
            # this frame's pixel -> reference pixel: invert the affine map
            A = np.array([[cx[0], cx[1]], [cy[0], cy[1]]])
            ref_xy = (np.linalg.inv(A) @
                      (zones[:, :2] - np.array([cx[2], cy[2]])).T).T
            zones = np.c_[ref_xy, zones[:, 2]]
        edge = int(np.ceil(2 * psf_px))
        k = len(mjds)
        for qx, qy, qf in zip(xs, ys, fl):
            # nothing within two star widths of the edge of the data: a frame
            # turned or shifted onto the grid leaves a border with no pixels
            xi, yi = int(round(qx)), int(round(qy))
            if not cover[max(yi - edge, 0):yi + edge + 1,
                         max(xi - edge, 0):xi + edge + 1].all():
                continue
            sh = short_axis(raw, qx, qy, psf_px)
            if sh is None:
                continue
            L, W = sh
            # It must HAVE a width - that is what separates a real object
            # from a cosmic ray, a satellite or a meteor, all of which are
            # lines with no thickness. There is deliberately NO upper limit
            # on the width: an earlier version rejected anything wider than
            # 2.2 PSF, which is precisely what a comet is. The tool was
            # filtering out the kind of object it was meant to find.
            if W < 0.55 * psf_px or L > 3.0 * max(W, 1e-6):
                continue
            if len(zones) and np.any(np.hypot(zones[:, 0] - qx, zones[:, 1] - qy)
                                     < zones[:, 2]):
                continue                     # inside a saturated star
            det.append((k, qx, qy, qf))
        if store is None:
            store = np.memmap(scratch, dtype=np.float32, mode="w+",
                              shape=(len(paths),) + clean.shape)
        store[k] = clean.astype(np.float32)
        mjds.append(fr["mjd"])
        if verbose:
            print(f"\r    {fr['n'] + 1}/{len(paths)} frames", end="", flush=True)

    pending = []
    for n, p in enumerate(paths):
        fr = load(n, p)
        if fr is None:
            continue
        if psf_px is not None:
            finish(fr)
            continue
        # The star width comes from the first frames, unless it was given:
        # the first frame is measured, and the next two say which of its
        # sources are stars (a cosmic ray is never in two frames).
        pending.append(fr)
        if len(pending) < 3 and n < len(paths) - 1:
            continue
        fw, nst = measure_psf([(q["raw"], q["rms_raw"]) for q in pending],
                              pending[0]["level"])
        if np.isfinite(fw):
            psf_px = fw
            if verbose:
                print(f"\n    star width (FWHM) measured on {nst} stars of "
                      f"the first frame: {fw:.2f} px "
                      f"= {fw * pixel_scale_of(pending[0]['w']):.2f} arcsec")
        else:
            psf_px = 3.2
            if verbose:
                print(f"\n    too few clean stars to measure their width "
                      f"({nst}); using 3.2 px - give --psf_px if that is "
                      f"wrong for this telescope")
        for q in pending:
            finish(q)
        pending = []
    if pending:                                   # fewer frames than that
        psf_px = 3.2
        for q in pending:
            finish(q)
    if verbose:
        print()
        if state["level_from"]:
            print(f"    saturation: {state['level_from']}")
        for note in sorted(notes):
            print(f"    NOTE: {note}")
    if len(mjds) < min_usable:
        raise SystemExit("Too few usable frames.")
    frames = store[:len(mjds)]
    return (frames, np.array(mjds), np.array(det).reshape(-1, 4),
            state["ref_w"], scratch, psf_px)


def cap_per_frame(det, budget, n_frames):
    """Keep only the `budget` brightest detections of each frame.

    The ORDER of this against drop_static matters more than the number does.
    It used to run inside `prepare`, on everything the finder produced, which
    spent the whole budget on the brightest things in the frame - and in a
    crowded field the brightest things are all static stars, deleted by the
    very next step. The budget was spent on objects guaranteed to be thrown
    away, and a faint mover never got near the cut.

    Measured on one degree of sky holding 5,139 sources per frame: capping
    first left drop_static able to recognise 359 stars, because 400 detections
    per frame is not enough repetition to learn a field. Capping afterwards let
    it recognise 6,092. What reaches the search stopped being marginal stars
    and started being things that are not where the sky keeps putting them.

    It is OFF by default, and that is deliberate. Once the static stars are
    gone, what remains is faint by nature - and so is an asteroid. Keeping the
    brightest of the leftovers throws away the faint end, which is the end the
    object being looked for lives at: on that same field it dropped 196 of the
    596 candidates in a frame. A budget here only bounds the running time, and
    it buys that by discarding exactly what the search exists to find."""
    if not budget or len(det) == 0:
        return det
    out = []
    for k in range(n_frames):
        m = det[det[:, 0] == k]
        if len(m) > budget:
            m = m[np.argsort(-m[:, 3])][:budget]
        out.append(m)
    return np.vstack(out) if out else det


def drop_static(det, n_frames, tol=2.5, min_frames=None, psf_px=3.2):
    """Set aside every position the sky keeps returning to."""
    if min_frames is None:
        min_frames = max(3, int(0.35 * n_frames))
    xy = det[:, 1:3]
    tree = cKDTree(xy)
    pairs = tree.query_ball_tree(tree, r=tol)
    lab = -np.ones(len(xy), int)
    n = 0
    for i in range(len(xy)):
        if lab[i] >= 0:
            continue
        st = [i]
        while st:
            j = st.pop()
            if lab[j] >= 0:
                continue
            lab[j] = n
            st.extend(pairs[j])
        n += 1
    keep = np.ones(len(xy), bool)
    centres, spreads = [], []
    for k in range(n):
        m = lab == k
        if len(np.unique(det[m, 0])) < min_frames:
            continue
        # A star's detections sit on top of each other. A SLOW MOVER's do not:
        # this asteroid shifts 1.8 px between neighbouring frames, inside the
        # 2.5 px linking radius, so single-link clustering chained all thirteen
        # of its detections into one group that looked exactly like a star
        # present in every frame - and the star filter deleted the very object
        # the tool exists to find. A cluster is only a star if it is COMPACT.
        p = xy[m]
        if np.hypot(*(p.max(axis=0) - p.min(axis=0))) > 2.0 * tol:
            continue
        # Compact is not enough either. An object seen in only four frames,
        # moving 1.4 px between them, spans 4.2 px - inside the 5 px limit -
        # and was deleted as a star. A star's detections scatter around one
        # place; a mover's march steadily in one direction. So the cluster's
        # positions are fitted against time (frame order), and a cluster that
        # travels more than half a star width along that fit is not a star.
        f = det[m, 0]
        if np.ptp(f) > 0:
            sx = np.polyfit(f, p[:, 0], 1)[0]
            sy = np.polyfit(f, p[:, 1], 1)[0]
            if np.hypot(sx, sy) * np.ptp(f) > max(1.5, 0.5 * psf_px):
                continue
        keep[m] = False
        centres.append(p.mean(axis=0))
        spreads.append(float(np.sqrt(np.mean(
            np.sum((p - p.mean(axis=0)) ** 2, axis=1)))))
    precision = float(np.median(spreads)) if spreads else float("nan")
    return (det[keep], np.array(centres) if centres else np.zeros((0, 2)),
            precision)


# --------------------------------------------------------------------------
# triples and stacking
# --------------------------------------------------------------------------

def triples(det, hours, n_frames, psf_px, max_rate_px_h, mid_tol=2.0,
            min_move_px=3.0, min_middles=2, min_arc_frac=0.25, verbose=True):
    """Velocities supported by three or more epochs moving at ONE rate.

    Two points define a line and prove nothing. A third has to land where the
    other two say it should - a real property of anything orbiting the Sun
    over twenty minutes, and one noise satisfies about once in five hundred
    tries.

    WHICH three is not fixed, and that matters. A first version asked for the
    first, middle and last frames; the known asteroid is detected in the first
    and missing from both of the others, so no triple could ever form and the
    search returned nothing. Detection at this brightness is intermittent by
    nature. So any early detection is paired with any late one, and the line
    between them is then offered to EVERY frame in between - the object has to
    be found in at least `min_middles` of them, not in two particular ones."""
    # ANY two detections far enough apart in time - not "one early, one late".
    # The known asteroid is detected in frames 0,1,2,3,4 and 7 and in none of
    # the last four, so an early-times-late scheme had nothing to pair it with
    # and could never find it however good the rest of the method was.
    t0, t1 = hours[0], hours[-1]
    early = det
    late = det
    if len(det) == 0:
        return []
    by_frame = {k: cKDTree(det[det[:, 0] == k][:, 1:3])
                for k in range(n_frames) if (det[:, 0] == k).any()}
    # frames actually used for the middle test: a dozen spread through the run
    probe = sorted(set(int(round(v)) for v in
                       np.linspace(0, n_frames - 1, min(14, n_frames))))
    probe = [k for k in probe if k in by_frame]
    need = max(2, int(round(min_middles * len(probe) / max(n_frames - 2, 1))))
    need = min(need, max(2, len(probe) - 2))
    out = []
    span_all = t1 - t0
    max_move = max_rate_px_h * span_all
    by_frame_pts = {k: det[det[:, 0] == k] for k in by_frame}
    # This loop is where a crowded field spends its minutes, and it used to
    # spend them in silence: the only line it printed came after it had
    # finished, so there was no way to tell a long job from a hung one. The
    # count is written over itself every few hundred detections.
    step = max(1, len(early) // 200)
    t_start = time.time()
    for ai in range(len(early)):
        if verbose and ai % step == 0 and ai:
            done = ai / len(early)
            left_s = (time.time() - t_start) * (1 - done) / max(done, 1e-9)
            print(f"\r    {ai}/{len(early)} detections followed, "
                  f"{len(out)} lines so far, about {left_s:.0f} s left   ",
                  end="", flush=True)
        ka = int(early[ai, 0]); ax, ay = early[ai, 1], early[ai, 2]
        ta = hours[ka]
        for kc, tree_c in by_frame.items():
            tc = hours[kc]
            dt = tc - ta
            # How much of the run the two anchor points must span. This was a
            # flat 0.45, and that number quietly decided which objects could
            # be found at all: an asteroid that enters the field, is seen for
            # a third of the night and leaves has no pair of its own
            # detections separated by 45% of the run, so no line could ever be
            # drawn through it and it was invisible to the method - not faint,
            # not rejected, never considered.
            #
            # Halving the stretch used to be the way round it. Cutting the run
            # in half halves span_all too, so a short arc suddenly cleared the
            # bar. That was never a search strategy, it was a workaround for
            # this line, and when the quick look stopped halving the workaround
            # went with it. The right answer is to lower the bar rather than
            # shorten the run, because shortening the run costs the arc length
            # that everything else here is measured against.
            if dt <= min_arc_frac * span_all:
                continue
            pts = by_frame_pts[kc]
            near = tree_c.query_ball_point((ax, ay), max_rate_px_h * dt)
            for ci in near:
                cx, cy = pts[ci, 1], pts[ci, 2]
                move = np.hypot(cx - ax, cy - ay)
                if move < min_move_px:
                    continue
                vx, vy = (cx - ax) / dt, (cy - ay) / dt
                seen = 0
                for k in probe:
                    if k in (ka, kc):
                        continue
                    px = ax + vx * (hours[k] - ta)
                    py = ay + vy * (hours[k] - ta)
                    if by_frame[k].query_ball_point((px, py), mid_tol):
                        seen += 1
                if seen >= need:
                    out.append((vx, vy, ax, ay, ta, seen + 2))
    if verbose:
        print(f"\r    {len(out)} lines that at least {need + 2} of "
              f"{len(probe)} sampled epochs agree on" + " " * 30)
    return out


def static_sky(frames, rows=64):
    """The median of all frames: what does not move.

    Read in strips, so that a stack of a few hundred large frames on disk is
    never in memory at once."""
    n, ny, nx = frames.shape
    out = np.zeros((ny, nx), np.float32)
    for r0 in range(0, ny, rows):
        out[r0:r0 + rows] = np.median(np.asarray(frames[:, r0:r0 + rows, :]),
                                      axis=0)
    return out


def stack_along(frames, hours, vx, vy, x0, y0, t0, half=22, core=3,
                static=None):
    """Add the frames up along one velocity, and see if anything piles up.

    Only a small box is stacked: the velocity already says where the object
    should be in every frame, so there is nothing to gain from moving whole
    images around.

    The frames are stacked twice: as they are, and with the sky that does not
    move (the median of all frames) taken out first. With the stars left in,
    every star in the box is smeared into a streak by the stacking, the
    streaks raise the measured noise, and an object detected cleanly in all
    13 frames stacked to S/N 3.6 - under the threshold - because its box held
    a bright star. With the stars out, the same object stacked far above it.
    The stars-in stack stays because a very slow object sits in the median
    itself and is partly taken out with it. The better of the two counts. The
    noise is a robust one (median absolute deviation) in both."""
    n, ny, nx = frames.shape
    acc = np.zeros((2 * half + 1, 2 * half + 1))
    acc_s = np.zeros_like(acc) if static is not None else None
    used = 0
    for k in range(n):
        X = x0 + vx * (hours[k] - t0)
        Y = y0 + vy * (hours[k] - t0)
        xi, yi = int(round(X)), int(round(Y))
        if xi - half < 0 or yi - half < 0 or xi + half >= nx or yi + half >= ny:
            continue
        b = frames[k][yi - half:yi + half + 1, xi - half:xi + half + 1]
        acc += b
        if acc_s is not None:
            acc_s += b - static[yi - half:yi + half + 1, xi - half:xi + half + 1]
        used += 1
    if used < 5:
        return None
    best = np.nan
    for a in (acc, acc_s):
        if a is None:
            continue
        a = a / used
        med = float(np.median(a))
        rms = 1.4826 * float(np.median(np.abs(a - med)))
        c = a[half - core:half + core + 1, half - core:half + core + 1]
        snr = float(c.max() - med) / rms if rms > 0 else np.nan
        if np.isfinite(snr) and not (snr <= best):
            best = snr
    return best, used, acc / used



# --------------------------------------------------------------------------
# the quick look
# --------------------------------------------------------------------------

class Tee:
    """Write to the screen and to a file at the same time.

    The console is where the answer appears and also where it disappears: it
    scrolls, and a session that took an hour leaves nothing behind. Everything
    printed goes to run_log.txt as well, including the questions asked and the
    answers given, so a run can be read back exactly as it happened."""

    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, "a", encoding="utf-8")
        import datetime
        self.fh.write(chr(10) + "=" * 70 + chr(10) +
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
                      chr(10) + "=" * 70 + chr(10))

    def write(self, s):
        self.stream.write(s)
        # A progress counter rewrites one screen line hundreds of times with
        # a carriage return. Written to the file as it came, it made a single
        # line of thousands of counts; only what the line finally said is kept.
        self.line = getattr(self, "line", "") + s
        *done, self.line = self.line.split(chr(10))
        try:
            for ln in done:
                self.fh.write(ln.split(chr(13))[-1] + chr(10))
            self.fh.flush()
        except Exception:
            pass
        return len(s)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


def parse_skybot(body):
    """The objects in a SkyBoT text reply, with the columns found BY NAME.

    The reply's header line names its columns:
        Num | Name | RA(h) | DE(deg) | Class | Mv | Err(arcsec) | d(arcsec) |
        dRA(arcsec/h) | dDEC(arcsec/h) | Dg(ua) | Dh(ua)
    The separation used to be taken as the smallest number among the 7th to
    9th fields - Err, d and dRA. Err is the uncertainty of the ephemeris, a
    fraction of an arcsec for a well-observed asteroid, so it was almost
    always the "separation", and every object within the one-arcminute search
    cone was reported as a match a fraction of an arcsec away. A westward
    dRA, which is negative, won outright. The distance is column d."""
    cols = None
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            f = [c.strip().lower() for c in line.lstrip("#").split("|")]
            if len(f) > 5 and f[0].startswith("num"):
                cols = f
            continue
        f = [c.strip() for c in line.split("|")]
        if len(f) < 8:
            continue

        def col(prefix, default):
            if cols:
                for i, c in enumerate(cols):
                    if c.startswith(prefix) and i < len(f):
                        return f[i]
            return f[default] if default < len(f) else ""

        def num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        sep = num(col("d(", 7))
        if sep is None:
            continue
        rows.append({"number": f[0] or "-", "name": f[1], "sep": sep,
                     "mag": num(col("mv", 5)), "ra": f[2], "dec": f[3],
                     "kind": col("class", 4),
                     "dra": num(col("dra", 8)), "ddec": num(col("ddec", 9))})
    rows.sort(key=lambda z: z["sep"])
    return rows


def identify(ra_deg, dec_deg, mjd, radius_arcsec=30.0, site="500", timeout=25):
    """Ask SkyBoT what known solar system object is at this place and time.

    IMCCE's SkyBoT holds the orbits of everything catalogued and will say what
    was in a given patch of sky at a given moment. That turns a track into a
    name - or, much more interestingly, fails to, which is the only way an
    unknown object announces itself.

    The default observer is the geocentre. For a main-belt asteroid that is
    wrong by a few arcseconds - Earth's radius over a couple of astronomical
    units - so the match radius allows for it. Give the observatory's MPC code
    as `site` and the residual drops to the measurement error.

    `mjd` must be the moment the position belongs to - the middle of the first
    exposure, which is what the tracks are now referred to.

    Never fatal: no internet, a slow service or an unexpected reply all return
    an error, and the run carries on."""
    import urllib.request
    import urllib.parse
    url = "https://ssp.imcce.fr/webservices/skybot/api/conesearch.php?" + \
        urllib.parse.urlencode({
            "-ep": f"{mjd + 2400000.5:.6f}", "-ra": f"{ra_deg:.6f}",
            "-dec": f"{dec_deg:.6f}", "-rm": f"{max(radius_arcsec / 60.0, 1.0):.2f}",
            "-mime": "text", "-output": "object", "-loc": str(site),
            "-filter": "0", "-objFilter": "111", "-refsys": "EQJ2000",
            "-from": "FindMovingObjects"})
    # SkyBoT reports its own failures INSIDE an ordinary reply: a header line
    # "# Flag: -1" and a message, with no rows. Read as an empty answer, that
    # became "NO known solar system object - worth a second look" for
    # (21246) 1995 YF1 on the WASP-52 frames, which the same query named
    # 3.5 arcsec away a minute later. A failed reply is retried, and if it
    # keeps failing it is reported as a failure, not as an empty sky.
    err = "no reply"
    for attempt in range(3):
        if attempt:
            time.sleep(3.0 * attempt)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                body = r.read().decode("utf-8", "replace")
        except Exception as e:
            err = str(e)
            continue
        m = re.search(r"^#\s*Flag\s*:\s*(-?\d+)", body, re.M)
        if m and int(m.group(1)) < 0:
            line = next((ln.strip("# ").strip() for ln in body.splitlines()
                         if ln.strip("# ").strip()
                         and not ln.lstrip("# ").lower().startswith(("flag", "ticket"))),
                        "")
            err = f"SkyBoT reported an error: {line or 'Flag -1'}"
            continue
        return {"objects": parse_skybot(body)}
    return {"error": err}


def mpc_url(number, name):
    """The object's page in the Minor Planet Center database.

    The number is used when the object has one: a numbered object keeps its
    number for good, while names and provisional designations change. An
    unnumbered object has only its designation, so that is what goes in.

    Returns None when there is nothing to build a link from, rather than a
    link that leads nowhere.
    """
    import urllib.parse
    num = str(number or "").strip().strip("()").strip()
    ident = num if num not in ("-", "") else str(name or "").strip()
    if not ident:
        return None
    return ("https://www.minorplanetcenter.net/db_search/show_object?object_id="
            + urllib.parse.quote(ident))


def name_it(ra, dec, mjd, args, motion=None):
    """What the catalogue thinks this is, as a list of lines to print.

    A list rather than one string because an identified object also gets a
    link to its MPC entry, and the callers indent each line themselves.

    `motion` is the measured (dRA*cos(Dec), dDec) in arcsec/hour. When it is
    given, the catalogued object's own motion is printed beside it: a name
    found at the right place but moving at a different rate or in another
    direction is a coincidence, not an identification.
    """
    if getattr(args, "no_catalogue", False):
        return ["not checked against the catalogue"]
    r = identify(ra, dec, mjd, radius_arcsec=args.match_arcsec,
                 site=args.site_code)
    if "error" in r:
        return [f"catalogue check FAILED ({r['error']}) - this is not a "
                f"\"no known object\"; the position above can be pasted into "
                f"SkyBoT by hand"]
    objs = r.get("objects", [])
    near = [o for o in objs if o["sep"] <= args.match_arcsec]
    if not near:
        n = len(objs)
        return [f"NO known solar system object within {args.match_arcsec:.0f} "
                f"arcsec ({n} catalogued further out) - worth a second look"]
    o = near[0]
    num = f"({o['number']}) " if o["number"] not in ("-", "") else ""
    mag = f", magnitude {o['mag']:.1f}" if o["mag"] else ""
    extra = ""
    if len(near) > 1:
        extra = f"  [{len(near) - 1} other catalogued object(s) also within " \
                f"{args.match_arcsec:.0f} arcsec]"
    lines = [f"identified: {num}{o['name']} - {o['sep']:.1f} arcsec from the "
             f"measured position{mag}{extra}"]
    if motion is not None and o.get("dra") is not None and o.get("ddec") is not None:
        cat = np.hypot(o["dra"], o["ddec"])
        diff = np.hypot(o["dra"] - motion[0], o["ddec"] - motion[1])
        verdict = "agrees" if diff <= max(0.2 * cat, 5.0) else \
            "DOES NOT agree - check before trusting the name"
        lines.append(f"catalogue motion {cat:.1f} arcsec/hour "
                     f"(dRA*cosDec {o['dra']:+.1f}, dDec {o['ddec']:+.1f}), "
                     f"measured {np.hypot(*motion):.1f} "
                     f"({motion[0]:+.1f}, {motion[1]:+.1f}): {verdict}")
    link = mpc_url(o["number"], o["name"])
    if link:
        lines.append(f"MPC entry: {link}")
    return lines


def draw_objects(objs, frames, hours, px, stem, tag=""):
    """One picture per object: the same field, three times, circle moving.

    A window that follows the object shows a dot standing still and proves
    nothing. The stars have to be in the picture, and staying put, for the
    motion to be visible at all - which is also why the field is several times
    the length of the track rather than a tight crop."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  could not draw the cut-outs: {e}")
        return
    for k, o in enumerate(objs, 1):
        try:
            cx0 = int(round(0.5 * (o["X"].min() + o["X"].max())))
            cy0 = int(round(0.5 * (o["Y"].min() + o["Y"].max())))
            half = int(max(170, 4.0 * max(np.ptp(o["X"]), np.ptp(o["Y"]))))
            pick = [0, len(frames) // 2, len(frames) - 1]
            # Four panels. The first is the WHOLE frame with the object marked,
            # because a close-up on its own says what it does but never says
            # where it is - and where it is, is the first thing anyone asks.
            fig, ax = plt.subplots(1, 4, figsize=(21.5, 6.0))
            full = frames[0]
            flo, fhi = np.percentile(full[::4, ::4], [30, 99.5])
            ax[0].imshow(full, cmap="gray_r", vmin=flo, vmax=fhi, origin="lower")
            ax[0].plot(o["X"], o["Y"], "-", color="red", lw=1.2)
            ax[0].plot(o["x0"], o["y0"], "o", ms=30, mfc="none",
                       mec="red", mew=1.4)
            ax[0].add_patch(plt.Rectangle(
                (cx0 - half, cy0 - half), 2 * half, 2 * half,
                fill=False, ec="red", lw=0.8, ls="--"))
            ax[0].set_xticks([]); ax[0].set_yticks([])
            ax[0].set_title(f"the whole frame, {full.shape[1]}x{full.shape[0]} px"
                            f"\nobject at pixel {o['x0']:.0f},{o['y0']:.0f}",
                            fontsize=10)
            for n, i in enumerate(pick, start=1):
                y0, y1 = max(cy0 - half, 0), min(cy0 + half, frames.shape[1])
                x0, x1 = max(cx0 - half, 0), min(cx0 + half, frames.shape[2])
                c = np.asarray(frames[i][y0:y1, x0:x1])
                # The grey scale is set by the NOISE, the same way in every
                # close-up: from 2 sigma below the sky to 12 above. It was the
                # 15th to 99.5th percentile of each crop, which in a crowded
                # field is set by the stars, and an object at S/N 10 sat at
                # white in one panel and showed in the next.
                if c.size:
                    _m, med, sd = sigma_clipped_stats(c, sigma=3.0)
                    lo, hi = med - 2.0 * sd, med + 12.0 * sd
                else:
                    lo, hi = 0, 1
                ax[n].imshow(c, cmap="gray_r", vmin=lo, vmax=hi,
                             origin="lower", extent=[x0, x1, y0, y1])
                xp = np.polyval([o["vx"], o["x0"]], hours[i])
                yp = np.polyval([o["vy"], o["y0"]], hours[i])
                ax[n].plot(xp, yp, "o", ms=26, mfc="none", mec="red", mew=1.4)
                ax[n].set_xticks([]); ax[n].set_yticks([])
                ax[n].set_title(f"{hours[i] * 60:.0f} min", fontsize=11)
            rate = float(np.hypot(o["vx"], o["vy"]) * px)
            fig.suptitle(f"object {k}{tag}  -  {rate:.1f} arcsec/hour  -  "
                         f"same field in the three close-ups, only the circle moves",
                         fontsize=12)
            fig.tight_layout()
            fig.savefig(f"{stem}_object{k}.png", dpi=110)
            plt.close(fig)
        except Exception as e:
            print(f"  could not draw object {k}: {e}")


def quick_look(paths, idx, args, stem=None, px_scale=1.0, verbose_depth=True):
    """A real search, on a handful of frames spread through the stretch.

    The first version of this read exactly three - first, middle, last - which
    is the classic blink and is right in principle. In practice it failed on
    the one night whose answer is known, and for a reason worth keeping in
    view: at this brightness a detection is intermittent. The 2011 asteroid is
    found in frames 0,1,2,3,4 and 7 of thirteen and missing from the rest, so
    no triple built from fixed positions contained it three times. Three frames
    are enough for an object that is plainly there in all of them, and this one
    is not.

    So a sample is read instead of three, and the full method runs on ALL of
    it. Fourteen frames out of a hundred and sixty is still a ninth of the
    work, and it asks a fair question rather than a lucky one."""
    sub = [paths[i] for i in idx]
    # Progress is shown here too. Reading fourteen frames off an external disk
    # is a minute or two of silence, and silence is indistinguishable from a
    # hung run - which is exactly the question this tool kept being asked.
    print("")
    frames, mjds, det, ref_w, scratch, psf = prepare(
        sub, args.psf_px, args.threshold, min_usable=3, verbose=True)
    try:
        # Everything read is searched. There used to be a cull here: the
        # frames were taken in adjacent PAIRS and only the deeper of each was
        # kept, so fourteen frames off the disk bought a search over seven.
        # That was a fair trade when one pass was all there was and the only
        # question was which seven to spend it on. It is the wrong trade now
        # that the next pass doubles the sampling anyway - half of what has
        # already been paid for is thrown away and nothing is bought with it.
        # It also cost a real detection: on a thirteen frame night the cull
        # left seven, too few of them inside the first third where that
        # asteroid is visible to make a triple. Halving the stretch looked
        # like the fix, but all halving really did was switch the cull off.
        counts = [int((det[:, 0] == k).sum()) for k in range(len(mjds))]
        if verbose_depth:
            print(f" [{len(mjds)} frames searched: "
                  f"{min(counts)}-{max(counts)} sources each]")
        # The search says what it is doing here too. A quick look used to run
        # it silently, so a crowded field went several minutes with nothing on
        # the screen and no way to tell a long job from a hung one - and when
        # it ended in NOT FOUND there was no way to tell "looked and it is not
        # there" from "nothing was left to look at".
        objs = search_prepared(frames, mjds, det,
                               psf_px=psf, min_snr=args.min_snr,
                               min_significance=args.min_significance,
                               max_wander=args.max_wander, px_scale=px_scale,
                               max_rate_arcsec_h=args.max_rate, verbose=True,
                               max_per_frame=args.max_per_frame)
        hrs = hours_of(mjds)
        # The pictures are drawn HERE, while the frames still exist. A quick
        # look used to leave nothing behind but a line on the screen, which is
        # the one thing that cannot be checked afterwards.
        if stem and objs:
            draw_objects(objs, frames, hrs, px_scale, stem,
                         tag=" (quick look)")
        return objs, hrs, np.sort(mjds), ref_w, frames.shape[0]
    finally:
        del frames
        try:
            os.remove(scratch)
        except OSError:
            pass


def quick_session(paths, args, px_scale, out_stem):
    """Look at a sample, say what is there, and let the observer ask for more.

    Each pass DOUBLES the sampling over the WHOLE run - fourteen frames, then
    twenty-seven, then fifty-three - until the sample is the run itself.

    It used to halve the stretch and look at one half instead. That buys a
    denser look at a piece of the night by throwing the rest of the night
    away, and what a track is measured against is exactly how far it moved:
    significance is travel over scatter, and travel grows with the span. Two
    passes cost the same reading either way - fourteen frames across half the
    run, or twenty-seven across all of it, both reach the same spacing - but
    only the second keeps the whole arc. A hundred and twenty frames spanning
    twenty-three minutes once found nothing at all, and that is the failure
    halving walks towards, one press at a time.

    The frames added by each pass fall midway between the ones already
    sampled, so every pass contains the pass before it whole."""
    n = min(args.quick_frames, len(paths))
    while True:
        idx = sorted(set(int(round(v))
                         for v in np.linspace(0, len(paths) - 1, n)))
        print(f"\n  reading {len(idx)} frames of {len(paths)}, spread over "
              f"the whole run ...", end="", flush=True)
        stem = f"{out_stem}_quick_{len(idx)}frames"
        try:
            objs, hours, mjds, ref_w, n_read = quick_look(
                paths, idx, args, stem=stem, px_scale=px_scale)
        except SystemExit as e:
            print(f" {e}")
            return
        span = hours[-1] - hours[0]
        print(f"  {span * 60:.0f} minutes apart")
        if objs:
            print(f"  FOUND {len(objs)} thing(s) that moved:")
            lines = report_objects(objs, ref_w, mjds[0], n_read, args)
            for ln in lines:
                print("  " + ln)
            with open(stem + "_tracks.txt", "w", encoding="utf-8") as fh:
                fh.write(f"quick look, {len(idx)} frames of {len(paths)} "
                         f"spread over the whole run" + NL +
                         f"{len(objs)} object(s), times are mid-exposure UTC, "
                         f"first frame MJD {mjds[0]:.6f}" + NL + NL)
                fh.write(NL.join(lines) + NL)
            print(f"  -> {stem}_tracks.txt  and a picture per object")
            print("  (a sample of frames - run again and answer n to the quick "
                  "look to measure them properly)")
        else:
            print("  NOT FOUND in this sample")
        # The ladder ends where the sample IS the run. There is no denser
        # look left to offer, and by then the quick look has quietly become
        # the full search.
        if len(idx) >= len(paths):
            print("  that was every frame in the run - nothing left to add")
            return
        nxt = min(2 * n - 1, len(paths))
        try:
            ans = input(f"  look again with {nxt} frames instead of "
                        f"{len(idx)}, or stop? [y/n]: ").strip().lower()
        except EOFError:
            return
        if not ans.startswith("y"):
            return
        # 2n-1, not 2n: the new frames fall exactly midway between the ones
        # already read, so the previous sample survives whole inside the next.
        n = nxt


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def search(paths, psf_px=None, threshold=4.0, min_snr=8.0,
           max_per_frame=0, min_significance=5.0, max_wander=5.0,
           max_rate_arcsec_h=400.0, verbose=True):
    frames, mjds, det, ref_w, scratch, psf_px = prepare(
        paths, psf_px, threshold, verbose=verbose)
    px = pixel_scale_of(ref_w)
    if not np.isfinite(px):
        px = 1.0
    objs = search_prepared(frames, mjds, det, psf_px, min_snr,
                           min_significance, max_wander, px,
                           max_rate_arcsec_h, verbose,
                           max_per_frame=max_per_frame)
    order = np.argsort(mjds)
    return objs, frames, hours_of(mjds), mjds[order], ref_w, scratch


def hours_of(mjds):
    m = np.sort(np.asarray(mjds, float))
    return (m - m[0]) * 24.0


def follow(frames, hours, cx, cy, psf_px, passes=3):
    """Measure one object in every frame, along its line, against an EMPTY sky.

    The earlier measurement took the brightest pixel in a 19x19 box around
    the predicted place. Any star inside that box is brighter than a faint
    asteroid, so the box locked onto the star in every frame and the "track"
    came out standing still: on a synthetic night with an S/N 7 asteroid at
    40 arcsec/hour the line the triples had found (62 px/hour, correct) was
    re-measured as 0.15 px/hour and thrown away as not moving.

    Now each frame's box has the sky behind the object taken out first: the
    median of the same box in the frames where the object is somewhere else,
    at least 2 star widths away (a frame that has no three such frames is not
    used). What a star contributes is the same in all
    of them and cancels; the object does not, because it was elsewhere. Then
    the position is a Gaussian-weighted centroid starting from the predicted
    place, and a frame counts only if the object stands at least 3 sigma above
    that empty sky. "Seen in" is the number of such frames - it used to be the
    number of frames in which the box fitted inside the image, whether there
    was anything in it or not."""
    n, ny, nx = frames.shape
    sw = psf_px / 2.3548
    half = int(np.ceil(4 * psf_px)) + 2
    gy, gx = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    X = Y = T = SN = None
    in_field = 0
    for _pass in range(passes):
        reach = (2.0 if _pass == 0 else 1.0) * psf_px
        px_ = np.polyval(cx, hours)
        py_ = np.polyval(cy, hours)
        xs, ys, ts, sn = [], [], [], []
        in_field = 0
        for k in range(n):
            gxk, gyk = px_[k], py_[k]
            xi, yi = int(round(gxk)), int(round(gyk))
            if xi - half < 0 or yi - half < 0 or xi + half >= nx or yi + half >= ny:
                continue
            in_field += 1
            box = np.asarray(frames[k][yi - half:yi + half + 1,
                                       xi - half:xi + half + 1], float)
            away = np.where(np.hypot(px_ - gxk, py_ - gyk) > 2.0 * psf_px)[0]
            sig_map = None
            # Without at least three frames in which the object is elsewhere
            # there is no empty sky to measure against, and whatever star sits
            # in the box would be measured instead. Such a frame is not used.
            if len(away) < 3:
                continue
            if len(away) >= 3:
                pick = away[np.linspace(0, len(away) - 1,
                                        min(12, len(away))).round().astype(int)]
                others = np.array([frames[j][yi - half:yi + half + 1,
                                             xi - half:xi + half + 1]
                                   for j in pick], dtype=float)
                sky = np.median(others, axis=0)
                box = box - sky
                if len(pick) >= 5:
                    # how much each pixel changes from frame to frame: a star
                    # under the object leaves its photon noise behind even
                    # when its light is taken out, and a pixel there must
                    # count for less
                    sig_map = 1.4826 * np.median(np.abs(others - sky), axis=0)
            noise = 1.4826 * float(np.median(np.abs(box - np.median(box))))
            if noise <= 0:
                continue
            sig_map = (np.full(box.shape, noise) if sig_map is None
                       else np.maximum(sig_map, noise))
            inv_var = (noise / sig_map) ** 2
            mx, my = gxk - xi, gyk - yi
            ok = True
            for _ in range(4):
                w = np.exp(-((gx - mx) ** 2 + (gy - my) ** 2) / (2 * sw * sw))
                wb = np.clip(box, 0, None) * w * inv_var
                t = wb.sum()
                if t <= 0:
                    ok = False
                    break
                mx, my = (wb * gx).sum() / t, (wb * gy).sum() / t
                if np.hypot(xi + mx - gxk, yi + my - gyk) > reach:
                    ok = False
                    break
            if not ok:
                continue
            j0, i0 = int(round(my)) + half, int(round(mx)) + half
            peak = float(box[j0, i0])
            if peak < 3.0 * sig_map[j0, i0]:
                continue
            xs.append(xi + mx)
            ys.append(yi + my)
            ts.append(hours[k])
            sn.append(peak / sig_map[j0, i0])
        if len(xs) < 4:
            return None
        X, Y, T, SN = np.array(xs), np.array(ys), np.array(ts), np.array(sn)
        # A frame in which the "object" is many times brighter than in the
        # others is a cosmic ray or a satellite crossing the box, not the
        # object: a real one keeps roughly its brightness through the night.
        bright = SN > max(3.0 * float(np.median(SN)), 10.0)
        if bright.any() and (~bright).sum() >= 4:
            X, Y, T, SN = X[~bright], Y[~bright], T[~bright], SN[~bright]
        cx, cy = np.polyfit(T, X, 1), np.polyfit(T, Y, 1)
        # One frame where a cosmic ray landed on the object, or beside it,
        # pulls its centroid by a pixel or more. Left in, that one frame set
        # the track's scatter, and the track was thrown out as wandering more
        # than the stars do. A frame far off the line of all the others is
        # set aside, and the line refitted without it.
        for _clip in range(2):
            res = np.hypot(X - np.polyval(cx, T), Y - np.polyval(cy, T))
            lim = max(3.0 * 1.4826 * float(np.median(res)), 0.3 * psf_px)
            keep = res <= lim
            if keep.all() or keep.sum() < 4:
                break
            X, Y, T, SN = X[keep], Y[keep], T[keep], SN[keep]
            cx, cy = np.polyfit(T, X, 1), np.polyfit(T, Y, 1)
    return X, Y, T, SN, cx, cy, in_field


def search_prepared(frames, mjds, det, psf_px=3.2, min_snr=8.0,
                    min_significance=5.0, max_wander=5.0, px_scale=1.0,
                    max_rate_arcsec_h=400.0, verbose=True,
                    max_per_frame=0):
    """Everything after the frames have been read and their sources found."""
    order = np.argsort(mjds, kind="stable")
    if np.any(order != np.arange(len(order))):
        # Only when the frames really are out of order. Indexing a disk-backed
        # stack with a list reads ALL of it into memory - 250 frames of
        # 2400x2400 px are 5.8 GB - and it used to happen on every run, sorted
        # or not. The frames now arrive sorted, so this is the rare case.
        frames, mjds = frames[order], mjds[order]
        remap = np.empty(len(order), int)
        remap[order] = np.arange(len(order))
        det[:, 0] = remap[det[:, 0].astype(int)]
    hours = (mjds - mjds[0]) * 24.0
    if verbose:
        print(f"    {len(det)} detections with a star's width, "
              f"{len(frames)} frames over {hours[-1]:.2f} h")
    if len(det) == 0:
        return []

    left, stat, precision = drop_static(det, len(frames),
                                        tol=max(2.5, 0.8 * psf_px),
                                        psf_px=psf_px)
    if verbose:
        print(f"    {len(stat)} static stars, {len(left)} detections left over"
              + (f", positions repeat to {precision:.2f} px"
                 if np.isfinite(precision) else ""))
    # The budget is spent HERE, on what is left after the sky has been taken
    # out, and not on the frame as it arrived.
    n_before = len(left)
    left = cap_per_frame(left, max_per_frame, len(frames))
    if verbose and len(left) != n_before:
        print(f"    {len(left)} of those kept, at most {max_per_frame} per "
              f"frame - {n_before - len(left)} of the FAINTEST discarded")

    # The ceiling is a RATE, not a distance. It used to be sixty pixels of
    # total travel however long the night was, which says a slow object on a
    # long run and a fast one on a short run are the same thing - they are not.
    # An object at 29.5 arcsec/hour crossed 83 px in two hours of WASP-52 and
    # was refused by that ceiling, while the same object inside a shorter
    # stretch passed. What is physical is the rate: a main-belt asteroid moves
    # around 30 arcsec/hour, and a near-Earth object a few hundred, so the
    # limit is set in those units and converted with the pixel scale.
    max_rate_px_h = max_rate_arcsec_h / max(px_scale, 1e-6)
    # The tolerances follow the star width: 2 px and 3 px were set for 3.2 px
    # stars, and on a telescope with 6 px stars a faint object's centroid
    # scatters further than 2 px from frame to frame.
    cands = triples(left, hours, len(frames), psf_px,
                    max_rate_px_h=max_rate_px_h,
                    mid_tol=max(2.0, 0.6 * psf_px),
                    min_move_px=max(3.0, psf_px), verbose=verbose)

    # The second place the run goes quiet: every candidate line has the frames
    # added up along it, and there can be tens of thousands of them.
    seen, hits = set(), []
    c_step = max(1, len(cands) // 100)
    s_half, s_core = max(22, int(6 * psf_px)), max(3, int(round(psf_px)))
    static = static_sky(frames) if len(frames) >= 5 and cands else None
    # Measured on ONE frame, not on the median: the median of many frames is
    # so quiet that every faint star in it stands 5 of its own sigmas high,
    # while a star matters to the measurement only when it is a few times the
    # noise of the frame the object is measured in.
    frame_rms = (float(sigma_clipped_stats(
        np.asarray(frames[len(frames) // 2]), sigma=3.0)[2])
        if static is not None else np.nan)
    for ci, (vx, vy, x0, y0, t0, nep) in enumerate(cands):
        if verbose and ci % c_step == 0 and ci:
            print(f"\r    {ci}/{len(cands)} lines stacked, "
                  f"{len(hits)} strong enough   ", end="", flush=True)
        r = stack_along(frames, hours, vx, vy, x0, y0, t0,
                        half=s_half, core=s_core, static=static)
        if r is None:
            continue
        snr, used, acc = r
        if not np.isfinite(snr) or snr < min_snr:
            continue
        key = (int(round(x0 / 12)), int(round(y0 / 12)),
               int(round(vx / 4)), int(round(vy / 4)))
        if key in seen:
            continue
        seen.add(key)
        hits.append(dict(snr=snr, vx=vx, vy=vy, x0=x0, y0=y0, t0=t0,
                         used=used, epochs=nep))
    if verbose and cands:
        print(f"\r    {len(cands)} lines stacked, {len(hits)} strong enough"
              + " " * 20)
    hits.sort(key=lambda h: -h["snr"])

    # One object arrives many times over, because many different pairs of its
    # own detections imply nearly the same line. Reporting eight lines that are
    # really two objects is how the comet came to be called "not found" when it
    # was sitting in the list twice. Two tracks are the same object if they
    # occupy the same place at the beginning, the middle and the end.
    def at(h, t):
        return h["x0"] + h["vx"] * (t - h["t0"]), h["y0"] + h["vy"] * (t - h["t0"])
    probes = (hours[0], hours[len(hours) // 2], hours[-1])
    objects = []
    for h in hits:
        for g in objects:
            if all(np.hypot(*(np.subtract(at(h, t), at(g[0], t)))) < 10.0
                   for t in probes):
                g.append(h)
                break
        else:
            objects.append([h])

    # measure each distinct object properly: follow it frame by frame from its
    # own predicted path, and fit the track to what is actually there
    # MO_DEBUG=1 prints why each candidate that was stacked strongly enough
    # was then turned down - for checking a known object that is not reported.
    debug = bool(os.environ.get("MO_DEBUG"))

    def _why(h, reason):
        if debug:
            print(f"    [debug] rejected line at ({h['x0']:.0f},{h['y0']:.0f}) "
                  f"px, {np.hypot(h['vx'], h['vy']):.1f} px/h, stack S/N "
                  f"{h['snr']:.1f}: {reason}")

    out = []
    for g in objects:
        h = g[0]
        # Measured against the LINE, not against the previous frame. Chaining
        # frame to frame let the box wander: one frame locked onto a
        # neighbouring star and every later frame followed that star, which is
        # how this object came out at 28 arcsec/hour instead of 18.8. The line
        # is refitted twice, and each frame is re-measured within a star width
        # of where the line says the object is - so a single bad frame cannot
        # drag the rest of the track with it.
        cx = np.array([h["vx"], h["x0"] - h["vx"] * h["t0"]])
        cy = np.array([h["vy"], h["y0"] - h["vy"] * h["t0"]])
        r = follow(frames, hours, cx, cy, psf_px)
        if r is None:
            _why(h, "could not be measured in 4 frames")
            continue
        X, Y, t, SN, cx, cy, in_field = r
        # Seen in at least half of the frames it was inside - not 60% of the
        # whole run. An object that crosses the edge of the field halfway
        # through the night is inside for half of it; the old count could
        # never let it through, whatever the triples found.
        if len(X) < max(5, int(np.ceil(0.5 * in_field))):
            _why(h, f"seen in {len(X)} of {in_field} frames")
            continue
        # And what it was seen with has to be more than the noise's own best
        # effort. Each frame counts from S/N 3, which a box of pure noise
        # reaches somewhere often enough; on an empty synthetic night with
        # 6 px stars a "track" of such peaks, median S/N 3.3, went through
        # every later test. The 2011 asteroid is at S/N 5 to 12. The floor is
        # 5, not 4: on the 2P/Encke frames three "objects" at a median of 4.0
        # to 4.5 were a diffraction spike and empty sky, while the two real
        # ones measure 6.6 and 8.1. This S/N is the peak over the local noise,
        # about 1.5 times the S/N of the whole image of the object - a
        # synthetic asteroid put in at S/N 5 measures 7.6 here.
        if float(np.median(SN)) < 5.0:
            _why(h, f"median S/N {np.median(SN):.1f}")
            continue
        res = np.hypot(X - np.polyval(cx, t), Y - np.polyval(cy, t))
        rms_px = float(np.sqrt(np.mean(res ** 2)))
        moved_px = float(np.hypot(cx[0], cy[0]) * (t[-1] - t[0]))
        if rms_px <= 0 or moved_px / rms_px < min_significance:
            _why(h, f"moved {moved_px:.1f} px, scatter {rms_px:.2f} px")
            continue
        # Less than three star widths over the whole run is not a motion this
        # method can confirm: the "track" of a faint star measured at S/N 4
        # wanders a star width by itself, and the centroid of a bright star
        # drifted by 3 px over three hours of Kelt-16. Below three widths
        # there is also no empty sky to measure the middle of the track
        # against. On the 2011 Faulkes frames the known asteroid moves 7.
        if moved_px < 3.0 * psf_px:
            _why(h, f"moved {moved_px:.1f} px < 3 star widths")
            continue
        # Only the frames in which the object is on EMPTY sky count. On the
        # Kelt-16 frames (RASA 11, 3 hours) the centroids of bright stars
        # drifted by 3 px over the night and every such star came out as a
        # mover at 2 arcsec/hour; on 2P/Encke "tracks" sat on a saturated
        # star, ran along its diffraction spike, or began on a star and went
        # on through noise. What is under each position is read from the
        # median of all frames, which holds the stars, the saturated cores and
        # their spikes and nothing that moves. A real object crosses a star
        # now and then; it must still be seen often enough, and well enough,
        # where nothing else is.
        if static is not None:
            xi_ = np.clip(np.round(X).astype(int), 0, static.shape[1] - 1)
            yi_ = np.clip(np.round(Y).astype(int), 0, static.shape[0] - 1)
            # A star counts only where it is as bright as the object itself:
            # a faint one under a brighter object barely moves its centroid,
            # and treating every faint star as "on a star" cost the comet of
            # the 2010 Faulkes frames half of its frames.
            off = static[yi_, xi_] < np.maximum(3.0, SN) * frame_rms
            # Most of the frames it WAS seen in, not the full count again: a
            # frame that lands on a star is not evidence against the object.
            # Asked for five of five, the comet C/2007 S2 - seen in 5 of 10
            # frames, one of them on a star - was refused at stack S/N 37,
            # while what this test exists to remove (a saturated star, the
            # centroid of a star drifting) is on empty sky in no frame at all.
            if (off.sum() < max(4, int(np.ceil(0.5 * len(X))))
                    or float(np.median(SN[off])) < 5.0):
                _why(h, f"on empty sky in {int(off.sum())} of {len(X)} frames"
                        f" (in field {in_field})")
                continue
        # Five times, not three. A comet is not a point: its centroid is
        # measured less precisely than a star's however good the frames are,
        # and at three times the stellar scatter the 2011 comet was thrown out
        # while the asteroid beside it survived. The allowance has to fit the
        # kind of object being looked for, not only the instrument.
        #
        # And the allowance grows for a FAINT object. The stars' scatter is
        # that of bright stars, a few hundredths of a pixel; a centroid at
        # S/N 7 is uncertain by about 0.9 star widths / S/N - on 3.2 px stars,
        # 0.4 px - however well the frames are aligned. Held to the bright
        # stars' standard, an object at S/N 5 per frame was found by the
        # triples and then rejected for scattering as much as its own
        # brightness makes it scatter.
        expected = 0.9 * psf_px / max(float(np.median(SN)), 1.0)
        allow = max(max_wander * max(precision, 0.15), 3.0 * expected) \
            if np.isfinite(precision) else np.inf
        if rms_px > allow:
            _why(h, f"scatter {rms_px:.2f} px > allowed {allow:.2f} px")
            continue                      # wanders further than it should
        out.append(dict(n=len(X), n_field=in_field, vx=cx[0], vy=cy[0],
                        x0=np.polyval(cx, 0.0), y0=np.polyval(cy, 0.0),
                        rms_px=rms_px, snr=h["snr"],
                        snr_frame=float(np.median(SN)), SN=SN,
                        lines=len(g), X=X, Y=Y, T=t,
                        moved_px=moved_px, sig=moved_px / rms_px))
    # Merge AGAIN, now that every track has been refitted: two candidates that
    # started apart routinely converge onto the same object, and reporting it
    # twice is how a list of two objects looked like a list of three.
    t_a, t_b = hours[0], hours[-1]

    def ends(o):
        return (o["x0"] + o["vx"] * t_a, o["y0"] + o["vy"] * t_a,
                o["x0"] + o["vx"] * t_b, o["y0"] + o["vy"] * t_b)

    merged = []
    for o in sorted(out, key=lambda z: -z["sig"]):
        ax, ay, bx, by = ends(o)
        for m in merged:
            mx, my, nx_, ny_ = ends(m)
            if np.hypot(ax - mx, ay - my) < 9.0 and np.hypot(bx - nx_, by - ny_) < 9.0:
                break
        else:
            merged.append(o)
    # A line through two different objects is not a third object. On the
    # 2P/Encke frames two "tracks" at 35 and 38 arcsec/hour took some frames
    # from the comet and some from 2011 NV1, both moving at 20. Their ends do
    # not match either real track, so the merge above kept them. They are
    # recognised by what they MEASURED: a track that shares its positions,
    # frame for frame, with a better one, is dropped. Better is seen in more
    # frames, then with a higher S/N.
    kept = []
    for o in sorted(merged, key=lambda z: (-z["n"], -z["snr_frame"])):
        shared = 0
        for m in kept:
            same_t = np.isin(o["T"], m["T"])
            if not same_t.any():
                continue
            j = np.searchsorted(m["T"], o["T"][same_t])
            shared += int(np.sum(np.hypot(o["X"][same_t] - m["X"][j],
                                          o["Y"][same_t] - m["Y"][j]) < psf_px))
        if shared >= 2 and shared >= 0.25 * o["n"]:
            if debug:
                print(f"    [debug] dropped a track at {np.hypot(o['vx'], o['vy']):.1f}"
                      f" px/h: {shared} of its {o['n']} points belong to a "
                      f"better track")
            continue
        kept.append(o)
    return sorted(kept, key=lambda z: -z["sig"])


def on_sky(o, ref_w):
    """Position, rate and direction of a track, measured on the sky.

    The direction used to be the angle of the motion in PIXELS, printed as
    "towards ... deg" as if it were a position angle. How pixels sit on the
    sky depends on the camera's rotation and on whether the image is
    mirrored, so that number meant nothing outside the one frame. It is now
    the position angle of the motion, from north through east, and the rate is
    the angular distance travelled in an hour, both through the plate
    solution."""
    a = ref_w.pixel_to_world(o["x0"], o["y0"])
    b = ref_w.pixel_to_world(o["x0"] + o["vx"], o["y0"] + o["vy"])
    a = SkyCoord(a.ra, a.dec, frame="icrs")
    b = SkyCoord(b.ra, b.dec, frame="icrs")
    rate = float(a.separation(b).to(u.arcsec).value)
    pa = float(a.position_angle(b).to(u.deg).value) % 360.0
    d_ra = float((b.ra - a.ra).wrap_at(180 * u.deg).to(u.arcsec).value
                 * np.cos(a.dec.radian))
    d_de = float((b.dec - a.dec).to(u.arcsec).value)
    return float(a.ra.deg), float(a.dec.deg), rate, pa, d_ra, d_de


def per_frame_lines(o, ref_w, mjd0, indent="          "):
    """One line per frame in which the object was measured."""
    ra, de = ref_w.all_pix2world(o["X"], o["Y"], 0)
    lines = [indent + "UTC (mid-exposure)          MJD            RA (deg)    "
             "Dec (deg)    x (px)   y (px)   S/N"]
    for i in range(len(o["X"])):
        m = mjd0 + o["T"][i] / 24.0
        lines.append(indent + f"{Time(m, format='mjd').isot[:22]:<26}  "
                     f"{m:.6f}  {float(ra[i]):10.6f}  {float(de[i]):+10.6f}  "
                     f"{o['X'][i]:7.1f}  {o['Y'][i]:7.1f}  {o['SN'][i]:5.1f}")
    return lines


def report_objects(objs, ref_w, mjd0, n_frames, args):
    """The lines that describe each track, for the screen and the file alike."""
    px = pixel_scale_of(ref_w)
    lines = []
    for k, o in enumerate(objs, 1):
        ra, dec, rate, pa, d_ra, d_de = on_sky(o, ref_w)
        lines.append(
            f"object {k}: {rate:5.1f} arcsec/hour towards position angle "
            f"{pa:5.1f} deg (N through E) | seen in {o['n']} of {n_frames} "
            f"frames ({o['n_field']} with it inside the field) | moved "
            f"{o['moved_px'] * px:.1f} arcsec, {o['sig']:.0f}x its own "
            f"{o['rms_px'] * px:.2f} arcsec scatter | S/N {o['snr_frame']:.1f} "
            f"per frame, {o['snr']:.0f} stacked")
        lines.append(f"          start RA {ra:.6f} Dec {dec:+.6f} at "
                     f"{Time(mjd0, format='mjd').isot[:22]} UTC = MJD {mjd0:.6f}"
                     f" (middle of the first exposure)   "
                     f"(pixel {o['x0']:.1f},{o['y0']:.1f} of the first frame)")
        for ln in name_it(ra, dec, mjd0, args, motion=(d_ra, d_de)):
            lines.append("          " + ln)
        lines.append("          measured in each frame:")
        lines += per_frame_lines(o, ref_w, mjd0, indent="            ")
        lines.append("")
    return lines


def main():
    global FIX_CRPIX
    ap = argparse.ArgumentParser(
        description="Track faint moving objects by stacking along their motion.")
    ap.add_argument("--folder", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="look at a few frames first, and ask before doing more")
    ap.add_argument("--quick_frames", type=int, default=14,
                    help="how many frames the FIRST quick pass searches; each "
                         "pass after it doubles, on request, up to the whole "
                         "run")
    ap.add_argument("--psf_px", type=float, default=None,
                    help="star width (FWHM) in pixels. Measured on the first "
                         "frame when not given")
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--keep_crpix", action="store_true",
                    help="use CRPIX2 as the header gives it, without taking "
                         "YORGSUBF off (frames read out as a subframe and "
                         "plate-solved afterwards)")
    ap.add_argument("--no_catalogue", action="store_true",
                    help="skip the SkyBoT check (no internet, or not wanted)")
    ap.add_argument("--match_arcsec", type=float, default=30.0,
                    help="how close a catalogued object must be to count as "
                         "the same one")
    ap.add_argument("--site_code", default="500",
                    help="MPC observatory code. 500 is the geocentre, which "
                         "for a main-belt asteroid is a few arcsec out")
    ap.add_argument("--max_rate", type=float, default=400.0,
                    help="fastest motion looked for, arcsec/hour. A main-belt "
                         "asteroid moves about 30; a near-Earth object can "
                         "reach several hundred")
    ap.add_argument("--max_wander", type=float, default=5.0,
                    help="how many times the stars' own scatter a track may "
                         "wander off its line")
    ap.add_argument("--min_significance", type=float, default=5.0,
                    help="how many times its own scatter a track must travel")
    ap.add_argument("--max_per_frame", type=int, default=0,
                    help="0 searches every candidate, which is the point. A "
                         "number here caps how many per frame reach the "
                         "search and keeps the BRIGHTEST of them - it bounds "
                         "the running time by throwing away the faint end, "
                         "which is where an asteroid is")
    ap.add_argument("--min_snr", type=float, default=8.0,
                    help="how strong the stacked object must be")
    args = ap.parse_args()
    FIX_CRPIX = not args.keep_crpix

    print("=" * 70)
    print("Find_Moving_Objects - stack along the motion, then look")
    me = os.path.abspath(__file__)
    import datetime
    try:
        when = datetime.datetime.fromtimestamp(
            os.path.getmtime(me)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        when = "unknown"
    print(f"  running: {me}")
    print(f"  last changed: {when}")
    print("=" * 70)

    if args.folder is None:
        args.folder = input("Enter path to the folder with the frames: ").strip().strip('"')
    if args.output_dir is None:
        d = os.path.join(args.folder, "Moving_Objects")
        v = input(f"Enter path for the results [{d}]: ").strip().strip('"')
        args.output_dir = v or d
    # Everything this writes goes into a sub-folder called result_<date_time>,
    # a new one for every run. Pointing the tool at the folder of frames is
    # the natural thing to do, and without the sub-folder the text files and
    # pictures land among the frames themselves; with ONE fixed sub-folder, as
    # before, the second run wrote over the tracks and pictures of the first.
    base_out = args.output_dir
    args.output_dir = os.path.join(
        base_out, "result_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.output_dir, exist_ok=True)
    # Everything the screen shows is written down as well. A console scrolls
    # away and cannot be gone back to, and the identification - the one line
    # that says what the object actually is - was only ever on the screen.
    log_path = os.path.join(args.output_dir, "run_log.txt")
    sys.stdout = Tee(sys.stdout, log_path)
    print(f"  results will be written to {os.path.abspath(args.output_dir)}")

    targets = scan_folder(args.folder, skip=base_out)
    if not targets:
        raise SystemExit("No plate-solved frames with an OBJECT keyword found.")
    for t, ps in sorted(targets.items()):
        print(f"\n{t}  -  {len(ps)} frames")
        if len(ps) < 6:
            print("  too few frames"); continue
        # Asked, not switched. --quick was a command-line flag, and this is run
        # from an editor with no arguments, so the one control that saves an
        # hour of reading was unreachable in practice. Everything else this
        # family of tools needs is asked for; so is this.
        quick = args.quick
        if not quick and len(ps) > args.quick_frames:
            while True:
                try:
                    ans = input(f"  A quick look first, on {args.quick_frames} "
                                f"frames instead of {len(ps)}? [Y/n]: ").strip().lower()
                except EOFError:
                    ans = ""
                if ans in ("", "y", "yes"):
                    quick = True
                    break
                if ans in ("n", "no"):
                    break
                print(f"    '{ans}' is not y or n - please answer y or n "
                      f"(check the keyboard language)")
        stem = os.path.join(args.output_dir, re.sub(r"[^A-Za-z0-9_]+", "_", t).strip("_"))
        if quick:
            try:
                pxs = pixel_scale_of(build_wcs(frame_header(ps[0]), fix_crpix=False))
                if not np.isfinite(pxs):
                    pxs = 1.0
            except Exception:
                pxs = 1.0
            quick_session(ps, args, pxs, stem)
            continue
        objs, frames, hours, mjds, ref_w, scratch = search(
            ps, psf_px=args.psf_px, threshold=args.threshold,
            min_snr=args.min_snr, max_per_frame=args.max_per_frame,
            min_significance=args.min_significance, max_wander=args.max_wander,
            max_rate_arcsec_h=args.max_rate)
        try:
            px = pixel_scale_of(ref_w)
            if not objs:
                print("  nothing moved in a straight line at a steady rate")
            lines = report_objects(objs, ref_w, mjds[0], len(frames), args)
            for ln in lines:
                print("  " + ln)
            with open(stem + "_tracks.txt", "w", encoding="utf-8") as fh:
                fh.write(f"{t}" + NL + f"{len(frames)} frames over "
                         f"{hours[-1]:.3f} h, times are mid-exposure UTC, "
                         f"first frame MJD {mjds[0]:.6f}" + NL + NL)
                fh.write(NL.join(lines) + NL)
            draw_objects(objs, frames, hours, px, stem)
            print(f"  -> {stem}_tracks.txt")
        finally:
            del frames
            try:
                os.remove(scratch)
            except OSError:
                pass


if __name__ == "__main__":
    main()
