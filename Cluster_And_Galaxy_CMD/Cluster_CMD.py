# -*- coding: utf-8 -*-
"""
Created by: Dr. Boaz Ron Zohar
https://github.com/BoazRonZohar/ObservationalAstronomyEducationTools/blob/main/Cluster_And_Galaxy_CMD/Cluster_CMD.py
Affiliation: Kinneret Observatory
Member of the LCO Global Sky Partners programme
Date: September 2025

Written for student projects on star clusters.

WHAT IT DOES

Takes one B-band and one V (or G) band image of a star cluster and produces
its colour-magnitude diagram.

A cluster is the closest astronomy comes to a controlled experiment: the
stars formed together, sit at the same distance, and are seen through the
same dust. So their differences are real differences between stars. Plot
colour against brightness and they fall along a main sequence that bends at
a point set by the cluster's age - which is how a cluster is dated.

WHAT YOU NEED

  two FITS images of the same cluster, one in B and one in V (or G), plate-solved

TWO KINDS OF CLUSTER, and the difference matters

  Open (O)       Membership comes from Cantat-Gaudin & Anders 2020
                 (Vizier J/A+A/640/A1), which used Gaia proper motions and
                 parallaxes to decide which stars actually belong. Field
                 stars that merely lie in the same direction are excluded.
                 Give the catalogue's name for the cluster - M6 is NGC_6405
                 there. Open_Cluster_Name_Resolver.py translates for you.

  Globular (G)   No such membership catalogue exists for these, so members
                 are taken geometrically: everything inside a radius you
                 give, in pixels. You will be asked for that radius.

WHAT IT ASKS YOU

  cluster type, O or G
  cluster name
  cluster radius in pixels        globular clusters only
  distance in parsecs             turns apparent magnitude into absolute
  colour excess E(B-V)            corrects both the colour and the brightness.
                                  A_V is not asked for separately: it is
                                  3.1 x E(B-V), so asking for both would let
                                  an impossible pair through unnoticed
  the two FITS paths

HOW IT WORKS

  1. finds sources in both images and measures the FWHM of each one
  2. gives every star its own aperture, 1.2 x its own FWHM, rather than one
     radius for the whole frame - a bright star and a faint one do not have
     the same profile
  3. matches the B and G detections to each other
  4. calibrates against APASS9 reference stars pulled from Vizier
  5. selects the members, by catalogue or by radius
  6. corrects for extinction and distance, and plots the diagram

WHAT YOU GET

Written into a sub-folder called result_CMD, created next to your images. The
folder holding your data comes out of a run exactly as it went in.

Note that these names are FIXED - they do not carry the cluster's name - so a
second cluster run against the same images overwrites the first. Give each
cluster its own folder.

  A by-product, kept because it is what everything is measured on:

    <input>_bgsub.fits              each image with its background subtracted.

  Every source in the frame

    fluxes_XY_FWHM_Ap.csv           every detection: position, its own FWHM,
                                    the aperture radius that FWHM gave it,
                                    and the flux in B and G. Instrumental,
                                    not yet calibrated.
    calib_stars_apass.csv           the APASS9 stars found in the field, with
                                    their catalogue magnitudes. These set the
                                    zero point. If the calibration looks
                                    wrong, look here first.
    fluxes_calibrated.csv           the same sources in real magnitudes
    fluxes_calibrated_galactic.csv  and after extinction and distance are
                                    taken out, so the values are absolute
    fluxes_calibrated_galactic_CMD.png
                                    the diagram for the whole frame, members
                                    and field stars together

  Cluster members only

    Everything above still contains stars that merely lie in the same
    direction. These files are what is left after the membership step -
    Gaia astrometry for an open cluster, the radius you gave for a globular.
    THIS IS THE RESULT.

    fluxes_cluster_only.csv                    the members
    fluxes_cluster_only_calibrated.csv         in real magnitudes
    fluxes_cluster_only_galactic.csv           extinction and distance removed
    fluxes_cluster_only_galactic_CMD.png       the cluster's own diagram
    cluster_with_stars.png                     the second-band image with the
                                               members marked, so you can see
                                               whether the selection makes sense

  If you look at three files, look at these:

    calib_stars_apass.csv                 did the calibration have anything
                                          to work with
    cluster_with_stars.png                 did the membership step pick the
                                          cluster and not the field
    fluxes_cluster_only_galactic_CMD.png  the diagram, in absolute magnitude

  Comparing the two diagrams is the point. The whole-frame one is a scatter;
  the members-only one should show a main sequence. If it does not, the
  membership step is where to look, not the photometry.

The parameters at the top of this file are tuned for the Kinneret frames
these projects use. On very different data, the detection threshold and the
matching tolerances are the first things to look at.

Usage: run it. Every question has a default; press Enter to accept it.
"""

import os
import time
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from astropy.wcs import WCS
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.spatial import cKDTree
import astroalign as aa
import matplotlib.pyplot as plt


# ------------------- Photometry config -------------------
DETECTION_SIGMA = 4.0   # detection threshold in units of background sigma (controls how many stars are detected)
FWHM_HINT = 3.0         # fallback FWHM in pixels (used only if per-star FWHM estimation fails)
K_APERTURE = 2       # aperture radius scaling factor relative to each star's FWHM
K_ANNULUS_IN = 2      # inner radius of background annulus, in units of FWHM
K_ANNULUS_OUT = 4     # outer radius of background annulus, in units of FWHM
GAIN_E_PER_ADU = 1.0    # CCD gain: electrons per ADU (used for noise estimation if needed)
READ_NOISE_E = 5.0      # detector read noise in electrons (used for noise estimation if needed)

R_V = 3.1               # A_V / E(B-V) for dust in our own galaxy. A_V is
                        # derived from the colour excess rather than asked for
                        # separately - see where E(B-V) is read in.

XY_MATCH_TOLERANCE = 15.0  # pixels

# Matching the B and G detections to each other, after registering B onto G's
# pixel grid - see the note beside where this is used. On real M67 frames,
# going through each frame's own WCS left a few pixels of residual that grew
# with distance from the field centre - about 1 degree of real, uncorrected
# rotation between the two exposures, more than a simple TAN plate solution
# captures reliably over a 2400 px field. Registering directly from the star
# patterns (astroalign, below) removes that; this tolerance is what is left
# over after that - mostly centroiding noise, not registration error.
BG_MATCH_TOLERANCE = 5.0  # pixels, in G's own pixel grid

# Sanity check on the astroalign fit: B and G come off the same telescope, so
# the pixel scale between them should be the same to a fraction of a percent.
# A fitted scale far from 1 means astroalign found too few real star pairs to
# trust - usually a sparse field - and the WCS-based fallback is used instead.
ASTROALIGN_MAX_SCALE_ERROR = 0.05  # 5%

# --- Compute RA/DEC for every star using WCS with RA_Dec_TOLERANCE arcsec tolerance ---
RA_Dec_TOLERANCE = 60    # arcsec

# How close a detected star has to be to an APASS catalogue entry to be
# treated as that star, in get_apass_calib_stars(). This used to default to
# RA_Dec_TOLERANCE (60 arcsec - about 81 px on these frames), which is not a
# match radius for one star, it is most of the frame: on M67, 8 of the 10
# "calibration stars" this produced shared one of just two catalogue
# positions, at detected pixel positions up to 130 px apart from each other -
# different real stars in the image, all credited with the catalogue
# magnitude of whichever bright APASS star happened to be within 81 px of
# them. A few arcsec is generous for how well two independent measurements of
# the same star agree.
APASS_MATCH_RADIUS = 3.0  # arcsec


# ------------------- Interactive user input -------------------
def _ask(prompt, default, cast=str, upper=False):
    s = input(f"{prompt} [{default}]: ").strip()
    if s == "" or s == "0":
        return default
    try:
        v = cast(s)
        return v.upper() if (upper and isinstance(v, str)) else v
    except Exception:
        return default
    
# The Cantat-Gaudin membership catalogue is about 86 MB and takes a good few
# seconds to arrive. It is needed twice - once to check the cluster's name and
# once to pick out its members - so it is fetched once and kept.
_CG_CATALOGUE = None


def cantat_gaudin_members():
    """The Cantat-Gaudin & Anders 2020 membership table, downloaded once."""
    global _CG_CATALOGUE
    if _CG_CATALOGUE is None:
        print("Downloading the Cantat-Gaudin cluster catalogue "
              "(Vizier J/A+A/640/A1, about 86 MB). This takes a few seconds "
              "the first time, and is then reused for the rest of the run.",
              flush=True)
        t0 = time.time()
        Vizier.ROW_LIMIT = -1
        _CG_CATALOGUE = Vizier.get_catalogs("J/A+A/640/A1")[1]
        print(f"   got it in {time.time() - t0:.0f} s - "
              f"{len(set(_CG_CATALOGUE['Cluster'])):,} clusters, "
              f"{len(_CG_CATALOGUE):,} member stars", flush=True)
    return _CG_CATALOGUE


def normalize_cluster_name(user_input: str) -> str:
    # --- Cluster name resolver ---
    members_table = cantat_gaudin_members()
    catalog_clusters = set(members_table["Cluster"])
    messier_to_catalog = {
        "M6":   "NGC_6405", "M7":   "NGC_6475", "M11":  "NGC_6705", "M18":  "NGC_6613",
        "M21":  "NGC_6531", "M23":  "NGC_6494", "M25":  "IC_4725",  "M26":  "NGC_6694",
        "M29":  "NGC_6913", "M34":  "NGC_1039", "M35":  "NGC_2168", "M36":  "NGC_1960",
        "M37":  "NGC_2099", "M38":  "NGC_1912", "M39":  "NGC_7092", "M41":  "NGC_2287",
        "M44":  "NGC_2632", "M45":  "Melotte_22","M46": "NGC_2437","M47": "NGC_2422",
        "M48":  "NGC_2548", "M50":  "NGC_2323", "M52":  "NGC_7654","M67": "NGC_2682",
        "M93":  "NGC_2447", "M103": "NGC_581",
    }
    name = user_input.strip().upper().replace(" ", "")
    if name in messier_to_catalog:
        candidate = messier_to_catalog[name]
    else:
        candidate = name
        for prefix in ["NGC", "IC", "COLLINDER", "MELOTTE"]:
            if candidate.startswith(prefix) and "_" not in candidate:
                head = ''.join([c for c in candidate if not c.isdigit()])
                tail = ''.join([c for c in candidate if c.isdigit()])
                if head and tail:
                    candidate = head + "_" + tail
                break
    if candidate in catalog_clusters:
        return candidate
    else:
        raise RuntimeError(f"'{user_input}' is not an open cluster in this catalog.")

# ------------------- Cluster name resolver -------------------

def _norm_path(p: str) -> str:
    p = p.strip().strip('"').strip("'")
    return os.path.normpath(p)



# ---------- Background subtraction (new block) ----------
def subtract_background_and_save(path):
    data, hdr = fits.getdata(path, header=True)
    median_val = np.nanmedian(data)
    data_sub = data - median_val
    # into result_CMD, not next to the frames: the folder holding your data
    # should come out of a run exactly as it went in.
    out_dir = os.path.join(os.path.dirname(path) or os.getcwd(), "result_CMD")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, stem + "_bgsub.fits")
    fits.writeto(out_path, data_sub, hdr, overwrite=True)
    print(f"[bgsub] wrote {out_path} (median={median_val:.3f})")
    return out_path


# ------------------- Core photometry helpers -------------------
def estimate_fwhm_moments(img, x, y, box=15, r_bg_in=8, r_bg_out=12):
    h, w = img.shape
    x0 = int(round(x)); y0 = int(round(y))
    x1 = max(0, x0 - box//2); x2 = min(w, x0 + box//2 + 1)
    y1 = max(0, y0 - box//2); y2 = min(h, y0 + box//2 + 1)
    if x2 <= x1+2 or y2 <= y1+2:
        return np.nan
    cut = img[y1:y2, x1:x2].astype(float)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    r = np.hypot(yy - y, xx - x)
    ann = (r >= r_bg_in) & (r <= r_bg_out)
    if np.sum(ann) >= 20:
        bg_mean, _, _ = sigma_clipped_stats(cut[ann], sigma=3.0, maxiters=5)
    else:
        bg_mean = float(np.nanmedian(cut))
    cut_bs = cut - bg_mean
    cut_bs[cut_bs < 0] = 0.0
    flux = cut_bs.sum()
    if flux <= 0:
        return np.nan
    x_mean = (cut_bs * (xx - x)).sum() / flux
    y_mean = (cut_bs * (yy - y)).sum() / flux
    x2 = (cut_bs * (xx - x - x_mean)**2).sum() / flux
    y2 = (cut_bs * (yy - y - y_mean)**2).sum() / flux
    sigma = float(np.sqrt(max(1e-12, 0.5 * (x2 + y2))))
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma

def _cutout(img, x, y, r_out):
    h, w = img.shape
    R = int(np.ceil(r_out)) + 2
    x0 = int(round(x)); y0 = int(round(y))
    x1 = max(0, x0 - R); x2 = min(w, x0 + R + 1)
    y1 = max(0, y0 - R); y2 = min(h, y0 + R + 1)
    cut = img[y1:y2, x1:x2].astype(float)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    return cut, xx, yy

def local_background_cutout(img, x, y, r_in, r_out):
    cut, xx, yy = _cutout(img, x, y, r_out)
    r = np.hypot(yy - y, xx - x)
    ann = (r >= r_in) & (r <= r_out)
    vals = cut[ann]
    if vals.size < 30:
        mean = float(np.nanmedian(vals)) if vals.size > 0 else float(np.nanmedian(cut))
        std = float(np.nanstd(vals)) if vals.size > 0 else float(np.nanstd(cut))
        return mean, std
    mean, median, std = sigma_clipped_stats(vals, sigma=3.0, maxiters=5)
    return float(mean), float(std)

def aperture_photometry_single_fast(img, x, y, r_ap, r_in, r_out):
    cut, xx, yy = _cutout(img, x, y, r_out)
    r = np.hypot(yy - y, xx - x)
    ap = (r <= r_ap)
    bkg_mean, bkg_std = local_background_cutout(img, x, y, r_in, r_out)
    ap_vals = cut[ap].astype(float)
    npix_ap = ap_vals.size
    flux_ap_adu = float(np.nansum(ap_vals) - bkg_mean * npix_ap)
    return flux_ap_adu

# ------------------- Main photometry routine -------------------
def run_photometry(filename, band):
    data = fits.getdata(filename, ext=0).astype(float)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    daofind = DAOStarFinder(fwhm=FWHM_HINT, threshold=DETECTION_SIGMA * std)
    tbl = daofind(data - median)
    if tbl is None or len(tbl) == 0:
        raise RuntimeError(f"No sources detected in {band}")
    tbl = tbl[tbl['peak'] > 10 * std]
    positions = [(float(r['xcentroid']), float(r['ycentroid'])) for r in tbl]

    rows = []
    # A frame can hold thousands of sources and each one is measured on its
    # own, so this is where a run goes quiet for a while. Report progress on
    # one line that rewrites itself, so it is clear the run is alive.
    n_total = len(positions)
    # About ten updates whatever the count, each on its own line. A carriage
    # return that rewrites one line looks tidier in a terminal, but Spyder's
    # console does not always honour it and then prints every update on one
    # very long line - which is worse than a few extra lines.
    step = max(1, n_total // 10)
    print(f"Measuring {n_total:,} sources in the {band} image...", flush=True)
    for i, (x, y) in enumerate(positions, 1):
        if i % step == 0 or i == n_total:
            print(f"   {i:,} of {n_total:,}", flush=True)
        fwhm_star = estimate_fwhm_moments(data, x, y, box=15, r_bg_in=8, r_bg_out=12)
        if not np.isfinite(fwhm_star) or fwhm_star <= 0:
            continue
        r_ap = K_APERTURE * fwhm_star
        r_in = K_ANNULUS_IN * fwhm_star
        r_out = K_ANNULUS_OUT * fwhm_star
        flux = aperture_photometry_single_fast(data, x, y, r_ap, r_in, r_out)
        rows.append([x, y, fwhm_star, r_ap, flux])

    df = pd.DataFrame(rows, columns=["X", "Y", "FWHM", "Aperture_Radius", f"Flux_{band}"])
    return df


# ------------------- Sources that cannot be measured cleanly -------------------

# How far a source's width may sit from the frame's typical width before it is
# not a single star. A star's width does vary a little across a field - it grows
# towards the corners - but not by a factor of two, so anything outside these
# bounds is something else: two stars the detector did not separate, or a cosmic
# ray, which is narrower than the optics can make anything.
WIDTH_MIN_FACTOR = 0.5
WIDTH_MAX_FACTOR = 2.0


def crowded_or_misshapen(df, quiet=False):
    """Which sources can be measured cleanly. Returns a mask of the good ones.

    WHY THIS IS HERE
    Aperture photometry sums the light in a circle. That is exact for a star with
    nothing near it and meaningless for two stars sharing the circle, because the
    sum is then both of them and there is no way to tell how much belongs to
    which. Nothing downstream can notice: the flux is a perfectly ordinary
    number, it becomes a perfectly ordinary magnitude, and it lands in the diagram
    next to the real stars.

    Measured on an M12 frame, where the centre is crowded: source widths ran from
    2.2 to 13.9 pixels around a typical 6.3, and the colour spread that came out
    of it ran from -3 to +5, where a globular cluster spans about -0.2 to +1.5.
    The structure a colour-magnitude diagram is drawn for had disappeared into it.

    Two tests, because each catches what the other misses.

    THE WIDTH TEST catches a blend the detector reported as one source: two stars
    close enough to merge are measured as one object about twice as wide. It also
    catches the opposite - a cosmic ray or a hot pixel, narrower than the optics
    can make any real source.

    THE NEIGHBOUR TEST catches a blend the detector did separate. Two stars ten
    pixels apart can each be measured at a normal width, and each one's aperture
    still swallows the other, because the aperture is drawn from the width and is
    wider than the gap. This looks at the actual distance to the nearest source
    rather than at the shape.

    Neither test repairs anything. A source that fails is dropped, which loses
    the crowded centre of a globular and leaves what remains trustworthy.
    """
    if df is None or not len(df) or "FWHM" not in df.columns:
        return None

    n_start = len(df)
    width = df["FWHM"].to_numpy(dtype=float)
    typical = float(np.nanmedian(width))
    if not np.isfinite(typical) or typical <= 0:
        return None

    too_narrow = width < WIDTH_MIN_FACTOR * typical
    too_wide = width > WIDTH_MAX_FACTOR * typical

    # The nearest other source, from the same list. cKDTree is asked for the two
    # closest neighbours because the closest one is always the source itself.
    xy = df[["X", "Y"]].to_numpy(dtype=float)
    crowded = np.zeros(n_start, dtype=bool)
    if n_start > 1 and "Aperture_Radius" in df.columns:
        distance, _ = cKDTree(xy).query(xy, k=2)
        crowded = distance[:, 1] < df["Aperture_Radius"].to_numpy(dtype=float)

    keep = ~(too_narrow | too_wide | crowded)
    if not quiet:
        print(f"   typical source width: {typical:.1f} px", flush=True)
        print(f"   {int(too_wide.sum()):,} too wide to be one star, "
              f"{int(too_narrow.sum()):,} too narrow to be a star at all, "
              f"{int(crowded.sum()):,} with another source inside their aperture",
              flush=True)
        print(f"   {int(keep.sum()):,} of {n_start:,} survive all three",
              flush=True)
    return keep


def write_cmd(subset, plot_path, name, what, distance, ebv, av):
    """One colour-magnitude diagram, drawn and saved.

    The heading is the file's own name. A figure copied into a document or
    printed out is then still traceable to the file it came from, and two
    figures that differ only in scope or in filter cannot be mistaken for each
    other, because the words that separate them are the same words in both
    places. The cluster and the numbers used sit on the second line.
    """
    plt.figure()
    plt.scatter(subset["Color_index_corr"], subset["Mag_V_corr_copy"],
                s=10, color="black")
    plt.gca().invert_yaxis()          # brighter = up
    plt.xlim(0, 2)                    # the cluster sequence sits in here; a few
                                      # stray points far outside it should not
                                      # squeeze the real spread down to nothing
    plt.xlabel("Color index (B−V)")
    plt.ylabel("V magnitude (corrected)")
    stem = os.path.splitext(os.path.basename(plot_path))[0]
    plt.title(f"{stem}\n"
              f"{name}    d = {distance:,.0f} pc    "
              f"E(B−V) = {ebv:.3f}    A_V = {av:.2f}", fontsize=11)
    plt.savefig(plot_path, dpi=150)
    plt.show()


def write_star_overlay(image, vmin, vmax, subset, plot_path, name, what):
    """The frame with a circle on every source in this subset.

    The companion to the diagram above. Two of them side by side answer the
    question the diagram alone cannot: where in the sky the sources that were set
    aside actually were. If they are the crowded centre, it is visible at once.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    plt.scatter(subset["X"], subset["Y"], s=30, edgecolor="red",
                facecolor="none", lw=1)
    stem = os.path.splitext(os.path.basename(plot_path))[0]
    plt.title(f"{stem}\n{name}", fontsize=11)
    plt.xlabel("X [pix]")
    plt.ylabel("Y [pix]")
    plt.savefig(plot_path, dpi=150)
    plt.show()

# ------------------- Calibration stars from APASS -------------------


def get_apass_calib_stars(fits_file, n_stars=10, radius_deg=0.3, match_radius_arcsec = APASS_MATCH_RADIUS):
    """
    Extract calibration stars from APASS DR9 for a given FITS image.

    Parameters
    ----------
    fits_file : str
        Path to FITS image (must contain WCS in header).
    n_stars : int
        Number of bright stars to return.
    radius_deg : float
        Search radius around the image center (in degrees).
    match_radius_arcsec : float
        Maximum allowed separation between detected star and APASS source [arcsec].
        Default = 3 arcsec.

    Returns
    -------
    calib_df : pandas.DataFrame
        Table with columns: X, Y, RA, DEC, Mag_B, Mag_V
    """

    # load image + WCS
    hdr = fits.getheader(fits_file, ext=0)
    w = WCS(hdr)

    # detect stars in image
    data = fits.getdata(fits_file, ext=0).astype(float)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    daofind = DAOStarFinder(fwhm=FWHM_HINT, threshold=DETECTION_SIGMA * std)
    tbl = daofind(data - median)
    if tbl is None or len(tbl) == 0:
        raise RuntimeError("No stars detected in image.")

    # image center
    ny, nx = data.shape
    ra_c, dec_c = w.all_pix2world(nx/2, ny/2, 0)

    # query APASS DR9
    #
    # row_limit has to be passed to the constructor, not set on the class
    # afterwards: "Vizier.ROW_LIMIT = -1" here left the new instance's own
    # ROW_LIMIT at astroquery's default of 50, silently - not an error, just
    # 50 rows returned instead of however many APASS actually has. On a
    # crowded globular cluster field this mattered: the real count was 301,
    # and with only 50 of them to try to match against, 2,913 detected stars
    # in a small field found exactly one clean, unambiguous pairing.
    catalog = "II/336/apass9"
    result = Vizier(columns=["RAJ2000","DEJ2000","Bmag","Vmag"], row_limit=-1).query_region(
        SkyCoord(ra=ra_c*u.deg, dec=dec_c*u.deg),
        radius=radius_deg*u.deg,
        catalog=catalog
    )
    if len(result) == 0:
        raise RuntimeError("No APASS sources found in region.")
    apass = result[0].to_pandas()
    apass.rename(columns={"RAJ2000":"RA","DEJ2000":"DEC","Bmag":"Mag_B","Vmag":"Mag_V"}, inplace=True)

    # match detected stars to APASS
    stars = []
    for row in tbl:
        x, y = float(row['xcentroid']), float(row['ycentroid'])
        ra, dec = w.all_pix2world(x, y, 0)
        c_img = SkyCoord(ra*u.deg, dec*u.deg)
        c_cat = SkyCoord(apass["RA"].values*u.deg, apass["DEC"].values*u.deg)
        idx, sep2d, _ = c_img.match_to_catalog_sky(c_cat)
        if sep2d.arcsec < match_radius_arcsec:  # default 3 arcsec
            stars.append({
                "X": x,
                "Y": y,
                "RA": apass.loc[idx,"RA"],
                "DEC": apass.loc[idx,"DEC"],
                "Mag_B": apass.loc[idx,"Mag_B"],
                "Mag_V": apass.loc[idx,"Mag_V"]
            })

    if len(stars) == 0:
        raise RuntimeError("No matches between image and APASS catalog.")

    calib_df = pd.DataFrame(stars)
    calib_df = calib_df.sort_values("Mag_V").head(n_stars).reset_index(drop=True)
    return calib_df


# ------------------- Calibration using N APASS stars -------------------


def calibrate_with_apass(df, calib_df, flux_col, cat_mag_col, out_mag_col):
    """
    Apply N-star calibration formula to compute calibrated magnitudes.

    Parameters
    ----------
    df : pandas.DataFrame
        Table of measured stars with fluxes.
    calib_df : pandas.DataFrame
        Calibration stars table, must include [flux_col, cat_mag_col].
    flux_col : str
        Column in df with measured flux.
    cat_mag_col : str
        Column in calib_df with catalog magnitudes.
    out_mag_col : str
        Name of new output column for calibrated magnitudes.

    Returns
    -------
    df_out : pandas.DataFrame
        Copy of df with new column [out_mag_col].
    """
    F_cal = np.array(calib_df[flux_col], dtype=float)
    M_cal = np.array(calib_df[cat_mag_col], dtype=float)
    denom = np.sum(10**(-0.4 * M_cal))

    mags = []
    for F_spot in df[flux_col].values:
        if F_spot <= 0 or denom <= 0:
            mags.append(np.nan)
            continue
        ratio = np.sum(F_cal / F_spot) / denom
        m_spot = 2.5 * np.log10(ratio)
        mags.append(m_spot)

    df_out = df.copy()
    df_out[out_mag_col] = mags
    return df_out

# ------------------- Cluster membership filtering (open + globular) -------------------



def get_cluster_members(cluster_name, fits_file, cluster_type="O",
                        match_radius_arcmin=1.0, radius_px=200):
    """
    Get cluster members:
    - Open clusters (O): Cantat-Gaudin & Anders 2020 (Gaia DR2 membership).
    - Globular clusters (G): select stars within radius_px from center (X,Y mean).
    """

    # Load WCS for RA/DEC transforms
    hdr = fits.getheader(fits_file, ext=0)
    w = WCS(hdr)

    if cluster_type == "O":
        # Open cluster: Cantat-Gaudin & Anders 2020, already in memory from
        # the name check rather than downloaded a second time
        members_table = cantat_gaudin_members()
        cluster_members = members_table[members_table["Cluster"] == cluster_name]

        if len(cluster_members) == 0:
            raise RuntimeError(f"No members found for open cluster {cluster_name}")

        # Detect probability column name
        if "proba" in cluster_members.colnames:
            prob_col = "proba"
        elif "Prob" in cluster_members.colnames:
            prob_col = "Prob"
        elif "Pmem" in cluster_members.colnames:
            prob_col = "Pmem"
        else:
            prob_col = None

        skycoords = SkyCoord(cluster_members["RA_ICRS"],
                             cluster_members["DE_ICRS"], unit="deg")
        xpix, ypix = w.world_to_pixel(skycoords)

        members_df = pd.DataFrame({
            "RA": cluster_members["RA_ICRS"],
            "DEC": cluster_members["DE_ICRS"],
            "X": xpix,
            "Y": ypix
        })
        if prob_col:
            members_df["Prob"] = cluster_members[prob_col]
        else:
            members_df["Prob"] = np.nan

    elif cluster_type == "G":
        # Globular cluster: select stars from photometry CSV
        flux_csv = os.path.join(os.path.dirname(fits_file), "fluxes_XY_FWHM_Ap.csv")
        if not os.path.exists(flux_csv):
            raise RuntimeError("fluxes_XY_FWHM_Ap.csv not found (run photometry first)")
        df_flux = pd.read_csv(flux_csv)

        # Cluster center estimated as mean X,Y
        cx = df_flux["X"].mean()
        cy = df_flux["Y"].mean()

        dx = df_flux["X"] - cx
        dy = df_flux["Y"] - cy
        r = np.sqrt(dx**2 + dy**2)

        members_df = df_flux[r <= radius_px].copy()
        members_df["Prob"] = 1.0
        members_df["ClusterCenterX"] = cx
        members_df["ClusterCenterY"] = cy

    else:
        raise ValueError("cluster_type must be 'O' or 'G'")

    members_df["MatchRadius_arcsec"] = match_radius_arcmin * 60.0
    return members_df


# ------------------- Registering B onto G -------------------

def register_B_onto_G(B_img, G_img, xB, yB, fits_file_B, fits_file_G):
    """Where each B-image position (xB, yB) lands in G's pixel grid.

    B and G are separate exposures - the telescope moves between them, if
    only by a little - so their pixel grids do not coincide, and by how much
    is not knowable in advance: it depends on the mount, the dither, how long
    the filter change took. On one real pair of M67 frames it was about 18 px
    of shift plus roughly a degree of rotation - enough that not one of over
    a thousand detected stars shared a rounded pixel between the two images.

    astroalign finds the registration directly from the star patterns in
    these two specific images - it does not assume any telescope, field, or
    plate scale, and does not depend on how good either WCS solution is. That
    matters here: going through the WCS on the same M67 frames left several
    pixels of residual that grew toward the field edges, because a plain TAN
    plate solution does not capture a real degree of rotation as cleanly as
    matching the star pattern itself does.

    Falls back to each frame's own WCS only if astroalign cannot find enough
    matching stars - a very sparse field - or its fitted scale is not
    plausible for two frames from the same telescope, which is a sign the fit
    is not trustworthy rather than a sign the telescope changed."""
    try:
        transf, (src_list, _tgt_list) = aa.find_transform(B_img, G_img)
        if abs(transf.scale - 1.0) > ASTROALIGN_MAX_SCALE_ERROR:
            raise ValueError(f"fitted scale {transf.scale:.3f} is not plausible "
                             f"for two frames from the same telescope")
        xy = transf(np.column_stack([xB, yB]))
        print(f"[out] registered B onto G from {len(src_list)} matched star "
             f"pairs (astroalign): shift ({transf.translation[0]:+.1f}, "
             f"{transf.translation[1]:+.1f}) px, rotation "
             f"{np.degrees(transf.rotation):+.2f} deg, scale {transf.scale:.4f}")
        return xy[:, 0], xy[:, 1]
    except Exception as e:
        print(f"[out] astroalign registration failed ({e}) - "
             f"falling back to each frame's own WCS")
        wcs_B = WCS(fits.getheader(fits_file_B))
        wcs_G = WCS(fits.getheader(fits_file_G))
        ra_B, dec_B = wcs_B.all_pix2world(xB, yB, 0)
        return wcs_G.all_world2pix(ra_B, dec_B, 0)


# ------------------- Reading a folder instead of asking -------------------

def frames_in_folder(folder):
    """Which file in this folder is B and which is V, read from the headers.

    WHY NOT ASK
    The filter is written into every frame the telescope produces. Asking a
    person to type it invites the one mistake that cannot be caught later: two
    paths swapped, and every colour in the diagram comes out with its sign
    reversed while the run looks perfectly normal throughout.

    Returns (b_path, v_path, object_name). Anything it cannot work out comes
    back as None and is asked for in the usual way.
    """
    import glob as _glob

    found = {}
    names = []
    for pattern in ("*.fits", "*.fit", "*.fts"):
        for path in sorted(_glob.glob(os.path.join(folder, pattern))):
            try:
                head = fits.getheader(path)
            except Exception:
                continue
            filt = str(head.get("FILTER", "")).strip().upper()
            if not filt:
                # No FILTER keyword: fall back to the file name, which is how
                # frames combined by other software usually carry it.
                stem = os.path.basename(path).upper()
                if "_B" in stem or " B" in stem:
                    filt = "B"
                elif "_V" in stem or " V" in stem:
                    filt = "V"
                elif "_G" in stem or " G" in stem:
                    filt = "G"
            if head.get("OBJECT"):
                names.append(str(head["OBJECT"]).strip())
            key = None
            if filt.startswith("B"):
                key = "B"
            elif filt.startswith("V") or filt.startswith("G"):
                key = "V"
            if key:
                found.setdefault(key, []).append(path)

    name = None
    if names:
        name = max(set(names), key=names.count)
    return found, name


def describe_choice(found):
    """Say which frame was taken for each filter, and what else was there.

    WHY THE REST ARE NAMED
    Because a folder often holds more than one frame of a filter and only one of
    them is used. The M13 folder here holds three in B: two single exposures from
    2023 and one combined frame from 2025. Taking the first in alphabetical order
    means taking a single 2023 exposure and leaving the deeper combined one
    unused - which is a reasonable thing to do by accident once, and not a
    reasonable thing to do twenty times without noticing.

    Nothing is chosen differently here. The others are simply named, so that the
    choice can be seen and the folder tidied if it was the wrong one.
    """
    import os as _os

    chosen = {}
    for key in ("B", "V"):
        paths = found.get(key, [])
        if not paths:
            print("   {}: not found in the folder".format(key))
            chosen[key] = None
            continue
        chosen[key] = paths[0]
        print("   {}: {}".format(key, _os.path.basename(paths[0])))
        if len(paths) > 1:
            others = ", ".join(_os.path.basename(p) for p in paths[1:])
            print("      {} more in this filter, not used: {}".format(
                len(paths) - 1, others))
    return chosen["B"], chosen["V"]


def cluster_type_from_simbad(name):
    """Open or globular, from the object type SIMBAD publishes. 'O', 'G' or None.

    This is a fact about the cluster, not a judgement: SIMBAD returns OpC for an
    open cluster and GlC for a globular. Asking for it is asking the person to
    repeat what the catalogue already knows.
    """
    try:
        from astroquery.simbad import Simbad

        query = Simbad()
        query.add_votable_fields("otype")
        result = query.query_object(name)
        if result is None or not len(result):
            return None
        otype = str(result["otype"][0]).strip()
        if "GlC" in otype:
            return "G"
        if "OpC" in otype or otype == "Cl*":
            return "O"
    except Exception as exc:
        print(f"[lookup] SIMBAD unavailable ({exc})", flush=True)
    return None


def globular_from_harris(name):
    """Distance in pc, E(B-V) and half-light radius for a globular.

    Harris (1996, 2010 edition) is the standard compilation for the Milky Way's
    globulars: 147 of them, a distance for 145 and a reddening for every one.
    That is essentially every globular a telescope on the ground can reach.
    """
    try:
        tables = Vizier(columns=["**"]).query_object(name, catalog="VII/202")
        if not tables:
            return None, None, None
        distance = ebv = half_light = None
        for table in tables:
            row = table[0]
            if distance is None and "Rsun" in table.colnames:
                distance = float(row["Rsun"]) * 1000.0
            if ebv is None and "E(B-V)" in table.colnames:
                ebv = float(row["E(B-V)"])
            if half_light is None and "Rh" in table.colnames:
                half_light = float(row["Rh"])
        return distance, ebv, half_light
    except Exception as exc:
        print(f"[lookup] Harris catalogue unavailable ({exc})", flush=True)
        return None, None, None


def open_cluster_from_dias(name):
    """Distance in pc, E(B-V) and diameter for an open cluster.

    WHY NOT THE DUST MAPS
    Because they answer a different question. IRSA gives the extinction through
    the whole Galaxy in a direction, which is right for something seen behind all
    of it and wrong for a cluster sitting inside it. On M67 and M12 the
    full-line-of-sight value came out 20-30 percent below the published cluster
    values.

    Dias et al. is a compilation from the literature rather than one uniform
    measurement, and its numbers do not always agree with catalogues built from
    Gaia alone: for M67 it gives 808 pc where Cantat-Gaudin gives 859. That is
    why what it returns is put on the screen before it is used.
    """
    try:
        tables = Vizier(columns=["**"]).query_object(name, catalog="B/ocl")
        if not tables:
            return None, None, None
        table = tables[0]
        row = table[0]
        distance = float(row["Dist"]) if "Dist" in table.colnames else None
        ebv = float(row["E(B-V)"]) if "E(B-V)" in table.colnames else None
        diameter = float(row["Diam"]) if "Diam" in table.colnames else None
        return distance, ebv, diameter
    except Exception as exc:
        print(f"[lookup] Dias catalogue unavailable ({exc})", flush=True)
        return None, None, None


def pixel_scale_of(path):
    """Arcseconds per pixel, from the plate solution, or None.

    WHY THE DETERMINANT AND NOT THE DIAGONAL
    The CD matrix carries the scale and the rotation together. When north is up
    the scale sits on its diagonal, and reading the diagonal is right. When the
    frame is rotated the scale moves off the diagonal, and what is left there is
    whatever the rotation happened to leave behind.

    On a 2023 M67 frame, rotated by about ninety degrees, the diagonal gave 0.096
    arcsec per pixel where the true scale is 1.25 - wrong by a factor of thirteen.

    The area of one pixel on the sky is the determinant of the matrix whatever
    the rotation, because a rotation does not change area. Its square root is the
    scale, and it is right for every frame from every telescope without the code
    needing to know which telescope that was.
    """
    try:
        head = fits.getheader(path)
        wcs = WCS(head)
        if wcs.has_celestial:
            area = abs(float(np.linalg.det(wcs.pixel_scale_matrix)))
            if area > 0:
                return 3600.0 * float(np.sqrt(area))
        # No plate solution at all. Some capture software writes the focal length
        # and the physical pixel size instead, and the scale follows from those.
        if head.get("PIXSCALE"):
            return float(head["PIXSCALE"])
        focal, pixel_um = head.get("FOCALLEN"), head.get("XPIXSZ")
        if focal and pixel_um:
            return 206.265 * float(pixel_um) / float(focal)
    except Exception:
        pass
    return None


def offered(label, value, source, unit=""):
    """Show a looked-up number and let it be overridden, or ask when there is none.

    Every number that goes into the diagram is printed with where it came from
    before it is used. A distance and a reddening set the whole vertical scale of
    a colour-magnitude diagram, and a wrong one moves every star together - which
    is exactly the kind of error that looks like a result.
    """
    if value is None:
        print(f"   {label}: not found in the catalogue")
        return float(input(f"   enter {label}{unit}: "))
    typed = input(
        f"   {label} = {value:g}{unit}  ({source})  [Enter to accept]: ").strip()
    return float(typed) if typed else float(value)


# ------------------- Run -------------------
print("=== Cluster Photometry Interactive Input ===")

folder = _norm_path(_ask(
    "Path to the folder holding the frames (Enter to name the two files instead)",
    "", str))

fits_file_B = fits_file_G = None
cluster_name = None
if folder and os.path.isdir(folder):
    _found, object_name = frames_in_folder(folder)
    fits_file_B, fits_file_G = describe_choice(_found)
    if object_name:
        cluster_name = object_name
        print(f"   cluster: {cluster_name}   (from the OBJECT keyword)")
elif folder:
    print(f"[input] there is no folder at {folder} - naming the two files instead")

if cluster_name is None:
    cluster_name = _ask("Cluster name", "M13", str)

cluster_type = cluster_type_from_simbad(cluster_name)
if cluster_type:
    kind = "globular" if cluster_type == "G" else "open"
    print(f"   type: {kind}   (SIMBAD)")
else:
    cluster_type = _ask("Cluster type [O=open, G=globular]", "O", str, upper=True)

print("Looking the cluster up...", flush=True)
if cluster_type == "G":
    _dist, _ebv, _size = globular_from_harris(cluster_name)
    _source = "Harris catalogue"
else:
    _dist, _ebv, _size = open_cluster_from_dias(cluster_name)
    _source = "Dias catalogue"

Cluster_distance = offered("distance", _dist, _source, " pc")
E_BV = offered("E(B-V)", _ebv, _source)

if cluster_type == "G":
    # The radius is still asked for. What the catalogue publishes is the
    # half-light radius, which by definition holds half the cluster's light and
    # is therefore too small to select its members; the tidal radius would be the
    # right quantity and is not in this table. The catalogue value is shown as a
    # yardstick - a globular usually reaches several times it - and the choice
    # stays with the person making it, because it decides which stars are members.
    _scale = pixel_scale_of(fits_file_G or fits_file_B or "")
    if _size and _scale:
        print(f"   for scale: the half-light radius is {_size:g} arcmin"
              f" = {_size * 60.0 / _scale:.0f} px in this frame;"
              f" a globular usually reaches several times that")
    Cluster_radius_px = float(input("Enter cluster radius in pixels (EXP: 200): "))
else:
    try:
        cluster_name = normalize_cluster_name(cluster_name)
    except RuntimeError as e:
        print(e)
        exit(1)
    Cluster_radius_px = None

A_V = R_V * E_BV
print(f"Using distance {Cluster_distance:g} pc, E(B-V) = {E_BV}, "
      f"A_V = R_V x E(B-V) = {A_V:.3f}")

if fits_file_B is None:
    fits_file_B = _ask("Path to B-band FITS image", "D:/example_B.fts", str)
if fits_file_G is None:
    fits_file_G = _ask("Path to V (or G) band FITS image", "D:/example_G.fts", str)
fits_file_B = _norm_path(fits_file_B)
fits_file_G = _norm_path(fits_file_G)

# Everything this script writes goes into one sub-folder beside the images,
# rather than being scattered through the folder holding your data. The frames
# stay where they are and are only read.
#
# Worked out BEFORE the background step, because that step now returns a path
# inside result_CMD; deriving the output folder from it afterwards would nest
# result_CMD inside itself.
_datadir = os.path.dirname(fits_file_G) if os.path.dirname(fits_file_G) else os.getcwd()
_outdir = os.path.join(_datadir, "result_CMD")
os.makedirs(_outdir, exist_ok=True)
print(f"[out] writing everything to {_outdir}")

fits_file_B = subtract_background_and_save(fits_file_B)
fits_file_G = subtract_background_and_save(fits_file_G)
out_csv = os.path.join(_outdir, "fluxes_XY_FWHM_Ap.csv")
dfG = run_photometry(fits_file_G, "G")
dfB = run_photometry(fits_file_B, "B")

# Match B detections to G detections by where they actually fall, not by an
# identical rounded pixel - the two images are separate exposures and their
# pixel grids essentially never coincide exactly. On one real pair of M67
# frames, taken 11 minutes apart, the two grids were about 18 px apart at the
# field centre plus roughly a degree of rotation: not one of over a thousand
# detected stars in either band then shared a rounded pixel with its
# counterpart, and the matched table came out with zero rows.
#
# register_B_onto_G() finds the true registration from the star patterns
# themselves (astroalign), falling back to each frame's own WCS only if that
# is not possible. wcs_G is kept: it is also used further down, to place the
# APASS calibration stars and the cluster's own catalogue members onto this
# same pixel grid.
wcs_G = WCS(fits.getheader(fits_file_G))
B_img = fits.getdata(fits_file_B, ext=0).astype(float)
G_img = fits.getdata(fits_file_G, ext=0).astype(float)
xB_in_G, yB_in_G = register_B_onto_G(B_img, G_img, dfB["X"].values, dfB["Y"].values,
                                     fits_file_B, fits_file_G)

tree = cKDTree(dfG[["X", "Y"]].values)
dist, idx = tree.query(np.column_stack([xB_in_G, yB_in_G]),
                       distance_upper_bound=BG_MATCH_TOLERANCE)

# If more than one B star lands within tolerance of the same G star - which
# happens in a crowded field - only the closer of the two is kept, so that G
# star's flux is not duplicated onto two rows.
best_for_g = {}
for i in range(len(dfB)):
    if dist[i] == np.inf:
        continue
    j = int(idx[i])
    if j not in best_for_g or dist[i] < best_for_g[j][1]:
        best_for_g[j] = (i, dist[i])

rows = []
for j, (i, _d) in best_for_g.items():
    g, b = dfG.iloc[j], dfB.iloc[i]
    rows.append([g["X"], g["Y"], g["FWHM"], g["Aperture_Radius"],
                b["Flux_B"], g["Flux_G"]])
df = pd.DataFrame(rows, columns=["X", "Y", "FWHM", "Aperture_Radius",
                                 "Flux_B", "Flux_G"])
print(f"[out] matched {len(df):,} stars between the {len(dfG):,} found in G "
     f"and the {len(dfB):,} found in B (within {BG_MATCH_TOLERANCE:.0f} px "
     f"of each other, in G's pixel grid, after registration)")

df.to_csv(out_csv, index=False)
print(f"[out] wrote {out_csv}")
   
# Run calibration star extraction on the B-band image
try:
    calib_df = get_apass_calib_stars(fits_file_B, n_stars=10)
    calib_csv = os.path.join(_outdir, "calib_stars_apass.csv")
    calib_df.to_csv(calib_csv, index=False)
    print(f"[calib] wrote {calib_csv}")
except Exception as e:
    print(f"[calib] error: {e}")

# Run calibration with XY tolerance matching
try:
    flux_csv = os.path.join(_outdir, "fluxes_XY_FWHM_Ap.csv")
    calib_csv = os.path.join(_outdir, "calib_stars_apass.csv")

    if os.path.exists(flux_csv) and os.path.exists(calib_csv):
        df_flux = pd.read_csv(flux_csv)
        df_calib = pd.read_csv(calib_csv)

        # calib_stars_apass.csv carries each calibration star's real sky
        # position (RA, DEC) from the APASS match, and also an X, Y - but that
        # X, Y is in B's own pixel grid, since get_apass_calib_stars() detects
        # stars on the B image only. df_flux's X, Y are in G's pixel grid (see
        # the sky-based B/G match above). Recomputing X, Y from each
        # calibration star's own RA/DEC through G's WCS puts both tables in
        # the same frame it is actually being matched against.
        calX, calY = wcs_G.all_world2pix(df_calib["RA"].values, df_calib["DEC"].values, 0)

        flux_coords = df_flux[["X","Y"]].values
        calib_coords = np.column_stack([calX, calY])
        tree = cKDTree(flux_coords)
        dist, idx = tree.query(calib_coords, distance_upper_bound = XY_MATCH_TOLERANCE)

        matches = []
        for i, j in enumerate(idx):
            if j < len(df_flux) and dist[i] != np.inf:
                row = df_calib.iloc[i].copy()
                row["Flux_B"] = df_flux.iloc[j]["Flux_B"]
                row["Flux_G"] = df_flux.iloc[j]["Flux_G"]
                matches.append(row)

        calib_matched = pd.DataFrame(matches)
        n_calib_used = len(calib_matched)

        # apply calibration using matched stars
        merged = df_flux.copy()
        merged = calibrate_with_apass(merged, calib_matched, "Flux_B", "Mag_B", "Mag_B_cal")
        merged = calibrate_with_apass(merged, calib_matched, "Flux_G", "Mag_V", "Mag_V_cal")
        merged["N_calib_used"] = n_calib_used

        out_cal_csv = os.path.join(_outdir, "fluxes_calibrated.csv")
        merged.to_csv(out_cal_csv, index=False)
        print(f"[calibN] wrote {out_cal_csv} using {n_calib_used} calibration stars (XY match ±5 px)")
    else:
        print("[calibN] required CSV files not found, skipping calibration")
except Exception as e:
    print(f"[calibN] error: {e}")   # ------------------- Galactic extinction correction -------------------   
cal_csv = os.path.join(_outdir, "fluxes_calibrated.csv")
if os.path.exists(cal_csv):
    df_cal = pd.read_csv(cal_csv)

    # --- Compute RA/DEC for every star using WCS with RA_Dec_TOLERANCE arcsec tolerance ---
    hdr = fits.getheader(fits_file_B, ext=0)
    w = WCS(hdr)
    ra, dec = w.all_pix2world(df_cal["X"].values, df_cal["Y"].values, 0)

    tol_deg = RA_Dec_TOLERANCE/ 3600.0  # RA_Dec_TOLERANCE arcsec in degrees
    ra = (ra / tol_deg).round() * tol_deg
    dec = (dec / tol_deg).round() * tol_deg

    df_cal["RA"] = ra
    df_cal["DEC"] = dec

    # extinction-corrected values
    df_cal["Mag_V_corr"] = df_cal["Mag_V_cal"] - A_V
    df_cal["Mag_B_corr"] = df_cal["Mag_B_cal"] - (A_V + E_BV)
    df_cal["BV_corr"]    = df_cal["Mag_B_corr"] - df_cal["Mag_V_corr"]

    # record the extinction parameters
    df_cal["A_V_used"]  = A_V
    df_cal["E_BV_used"] = E_BV

    out_corr_csv = os.path.join(_outdir, "fluxes_calibrated_galactic.csv")

    desired_order = [
        "X", "Y", "RA", "DEC", "Prob",
        "FWHM", "Aperture_Radius",
        "Flux_B", "Flux_G",
        "Mag_B_cal", "Mag_V_cal", "N_calib_used",
        "Mag_V_corr", "Mag_B_corr",
        "A_V_used", "E_BV_used",
        "BV_corr", "Mag_V_corr_copy"
    ]
    
    cols = [c for c in desired_order if c in df_cal.columns]
    df_cal[cols].to_csv(out_corr_csv, index=False)
    
    print(f"[galactic] wrote {out_corr_csv}")

else:
    print("[galactic] fluxes_calibrated.csv not found, skipping galactic correction")


# ------------------- Run cluster membership selection -------------------

try:
    members_df = get_cluster_members(cluster_name,
                                     fits_file_B,
                                     cluster_type=cluster_type,
                                     match_radius_arcmin=1.0,
                                     radius_px=Cluster_radius_px if Cluster_radius_px else 200)

    out_cluster_csv = os.path.join(_outdir, "fluxes_cluster_only.csv")
    members_df.to_csv(out_cluster_csv, index=False)
    print(f"[cluster] wrote {out_cluster_csv} with {len(members_df)} members of {cluster_name}")

except Exception as e:
    print(f"[cluster] error: {e}")
   

# ------------------- Cluster calibrated photometry -------------------
try:
    cal_csv = os.path.join(_outdir, "fluxes_calibrated.csv")
    cluster_csv = os.path.join(_outdir, "fluxes_cluster_only.csv")

    if os.path.exists(cal_csv) and os.path.exists(cluster_csv):
        df_cal = pd.read_csv(cal_csv)
        df_cluster = pd.read_csv(cluster_csv)
        
        # --- Compute RA/DEC for every star using WCS with RA_Dec_TOLERANCE arcsec tolerance ---
        # df_cal's X, Y are in G's pixel grid (it descends from fluxes_XY_FWHM_Ap.csv,
        # see the sky-based B/G match above) - so this has to go through G's WCS,
        # not B's, or every position comes out shifted by the B/G frame offset.
        ra, dec = wcs_G.all_pix2world(df_cal["X"].values, df_cal["Y"].values, 0)
        
        # round RA/DEC to nearest RA_Dec_TOLERANCE arcsec (i.e. 1/60 degree)
        tol_deg = RA_Dec_TOLERANCE / 3600.0  # 60 arcsec in degrees
        ra = (ra / tol_deg).round() * tol_deg
        dec = (dec / tol_deg).round() * tol_deg
        
        df_cal["RA"] = ra
        df_cal["DEC"] = dec


        # Match cluster-only stars to calibrated stars using KDTree with ±5 px tolerance
        cal_coords = df_cal[["X", "Y"]].values
        cluster_coords = df_cluster[["X", "Y"]].values

        tree = cKDTree(cal_coords)
        dist, idx = tree.query(cluster_coords, distance_upper_bound=XY_MATCH_TOLERANCE)

        # If two catalogue members (a crowded field can have close pairs) both
        # land within tolerance of the same measured star, only the closer of
        # the two is kept - otherwise that one star's photometry is written
        # out twice, as two "different" members with identical flux.
        best_for_j = {}
        for i, j in enumerate(idx):
            if j < len(df_cal) and dist[i] != np.inf:
                if j not in best_for_j or dist[i] < best_for_j[j][1]:
                    best_for_j[j] = (i, dist[i])

        matches = []
        for j, (i, _d) in best_for_j.items():
            row = df_cal.iloc[j].copy()
            # Preserve extra info from cluster-only file (RA, DEC, Prob)
            #row["RA"] = df_cluster.iloc[i].get("RA", np.nan)
            #row["DEC"] = df_cluster.iloc[i].get("DEC", np.nan)
            row["Prob"] = df_cluster.iloc[i].get("Prob", np.nan)
            matches.append(row)

        if matches:
            cluster_calib = pd.DataFrame(matches)
            out_cluster_calib_csv = os.path.join(_outdir, "fluxes_cluster_only_calibrated.csv")
            cluster_calib.to_csv(out_cluster_calib_csv, index=False)
            print(f"[cluster_calib] wrote {out_cluster_calib_csv} with {len(cluster_calib)} calibrated members")
        else:
            print("[cluster_calib] no matches found between cluster members and calibrated photometry")
    else:
        print("[cluster_calib] required CSV files not found, skipping calibrated cluster output")

except Exception as e:
    print(f"[cluster_calib] error: {e}")

# ------------------- Cluster galactic-corrected photometry -------------------
try:
    gal_csv = os.path.join(_outdir, "fluxes_calibrated_galactic.csv")
    cluster_csv = os.path.join(_outdir, "fluxes_cluster_only.csv")

    if os.path.exists(gal_csv) and os.path.exists(cluster_csv):
        df_gal = pd.read_csv(gal_csv)
        df_cluster = pd.read_csv(cluster_csv)

        # --- Compute RA/DEC for every star using WCS with RA_Dec_TOLERANCE arcsec tolerance ---
        # df_gal's X, Y are in G's pixel grid, same reasoning as the block above.
        ra, dec = wcs_G.all_pix2world(df_gal["X"].values, df_gal["Y"].values, 0)

        # round RA/DEC to nearest RA_Dec_TOLERANCE arcsec (i.e. 1/60 degree)
        tol_deg = RA_Dec_TOLERANCE  / 3600.0  # 60 arcsec in degrees
        ra = (ra / tol_deg).round() * tol_deg
        dec = (dec / tol_deg).round() * tol_deg

        df_gal["RA"] = ra
        df_gal["DEC"] = dec


# ----Match cluster-only stars to galactic-corrected calibrated stars using KDTree------
        gal_coords = df_gal[["X", "Y"]].values
        cluster_coords = df_cluster[["X", "Y"]].values

        tree = cKDTree(gal_coords)
        dist, idx = tree.query(cluster_coords, distance_upper_bound=XY_MATCH_TOLERANCE)

        # If two catalogue members (a crowded field can have close pairs) both
        # land within tolerance of the same measured star, only the closer of
        # the two is kept - otherwise that one star's photometry is written
        # out twice, as two "different" members with identical flux.
        best_for_j = {}
        for i, j in enumerate(idx):
            if j < len(df_gal) and dist[i] != np.inf:
                if j not in best_for_j or dist[i] < best_for_j[j][1]:
                    best_for_j[j] = (i, dist[i])

        matches = []
        for j, (i, _d) in best_for_j.items():
            row = df_gal.iloc[j].copy()
            # Preserve RA, DEC, Prob from cluster-only file
            #row["RA"] = df_cluster.iloc[i].get("RA", np.nan)
            #row["DEC"] = df_cluster.iloc[i].get("DEC", np.nan)
            row["Prob"] = df_cluster.iloc[i].get("Prob", np.nan)
            matches.append(row)

        if matches:
            cluster_gal = pd.DataFrame(matches)

            # Drop unused APASS columns that often remain empty
            drop_cols = ["Mag_B", "Mag_V"]
            cluster_gal = cluster_gal.drop(columns=drop_cols, errors="ignore")

            out_cluster_gal_csv = os.path.join(_outdir, "fluxes_cluster_only_galactic.csv")

            desired_order = [
                "X", "Y", "RA", "DEC", "Prob",
                "FWHM", "Aperture_Radius",
                "Flux_B", "Flux_G",
                "Mag_B_cal", "Mag_V_cal", "N_calib_used",
                "Mag_V_corr", "Mag_B_corr",
                "A_V_used", "E_BV_used",
                "BV_corr", "Mag_V_corr_copy"
            ]
            
            cols = [c for c in desired_order if c in cluster_gal.columns]
            cluster_gal[cols].to_csv(out_cluster_gal_csv, index=False)
            
            print(f"[cluster_gal] wrote {out_cluster_gal_csv} with {len(cluster_gal)} galactic-corrected members")

        else:
            print("[cluster_gal] no matches found between cluster members and galactic-corrected photometry")
    else:
        print("[cluster_gal] required CSV files not found, skipping galactic-corrected cluster output")

except Exception as e:
    print(f"[cluster_gal] error: {e}")

# ------------------- Add color index and CMD plots -------------------
try:
    for fname in ["fluxes_calibrated_galactic.csv", "fluxes_cluster_only_galactic.csv"]:
        fpath = os.path.join(_outdir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)

            if "Mag_B_corr" in df.columns and "Mag_V_corr" in df.columns:
                # add new columns
                df["Color_index_corr"] = df["Mag_B_corr"] - df["Mag_V_corr"]
                df["Mag_V_corr_copy"] = df["Mag_V_corr"]

                # save back to CSV (overwrite)
                df.to_csv(fpath, index=False)
                print(f"[color_index] updated {fname} with new columns")

                # Two diagrams from the same table: everything that was
                # measured, and what is left once sources with a neighbour
                # inside their own aperture are set aside.
                #
                # The names say which is which. "fluxes_cluster_only_galactic"
                # described where the numbers came from, which matters to nobody
                # opening the folder a year later; scope, filter and count are
                # what actually distinguish one picture from another.
                scope = ("cluster_sources" if "cluster_only" in fname
                         else "whole_frame_sources")
                keep_mask = crowded_or_misshapen(df)
                versions = [("all", df)]
                if keep_mask is not None and keep_mask.any() and not keep_mask.all():
                    versions.append(("no_overlap", df.loc[keep_mask]))
                    clean_csv = os.path.join(
                        _outdir, f"{scope}_no_overlap_{int(keep_mask.sum())}.csv")
                    df.loc[keep_mask].to_csv(clean_csv, index=False)
                    print(f"[color_index] wrote {os.path.basename(clean_csv)}")

                for kind, subset in versions:
                    where = ("cluster sources" if scope == "cluster_sources"
                             else "whole frame")
                    what = (f"{where}, all" if kind == "all"
                            else f"{where}, no overlap")
                    plot_path = os.path.join(
                        _outdir, f"CMD_{scope}_{kind}_{len(subset)}.png")
                    write_cmd(subset, plot_path, cluster_name, what,
                              Cluster_distance, E_BV, A_V)
                    print(f"[color_index] saved CMD plot to {plot_path}")
            else:
                print(f"[color_index] {fname} missing Mag_B_corr or Mag_V_corr columns")
        else:
            print(f"[color_index] file {fname} not found, skipped")

except Exception as e:
    print(f"[color_index] error: {e}")

# ------------------- Plot G-band image with measured stars -------------------
try:
    # Load G-band data
    data_G = fits.getdata(fits_file_G, ext=0)

    # Stretch for better visibility
    from astropy.visualization import ZScaleInterval
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data_G)

    # Read measured stars (flux file)
    flux_csv = os.path.join(_outdir, "fluxes_cluster_only_galactic.csv")
    if os.path.exists(flux_csv):
        df_flux = pd.read_csv(flux_csv)

        keep_mask = crowded_or_misshapen(df_flux, quiet=True)
        maps = [("all", df_flux)]
        if keep_mask is not None and keep_mask.any() and not keep_mask.all():
            maps.append(("no_overlap", df_flux.loc[keep_mask]))

        for kind, subset in maps:
            what = ("cluster sources, all" if kind == "all"
                    else "cluster sources, no overlap")
            out_img_path = os.path.join(
                _outdir, f"map_cluster_sources_{kind}_{len(subset)}.png")
            write_star_overlay(data_G, vmin, vmax, subset, out_img_path,
                               cluster_name, what)
            print(f"[plot] wrote {out_img_path}")
    else:
        print("[plot] flux file not found, skipping star overlay plot")

except Exception as e:
    print(f"[plot] error: {e}")
# ---------------------------------------------------------------------




