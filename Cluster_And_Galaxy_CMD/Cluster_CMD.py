# -*- coding: utf-8 -*-
"""
Created by: Dr. Boaz Ron Zohar
https://github.com/BoazRonZohar/ObservationalAstronomyEducationTools/blob/main/Cluster_And_Galaxy_CMD/Cluster_CMD.py
Affiliation: Kinneret Observatory
Member of the LCO Global Sky Partners programme
Date: September 2025

Written for student projects on star clusters.

WHAT IT DOES

Takes one B-band and one G-band image of a star cluster and produces its
colour-magnitude diagram.

A cluster is the closest astronomy comes to a controlled experiment: the
stars formed together, sit at the same distance, and are seen through the
same dust. So their differences are real differences between stars. Plot
colour against brightness and they fall along a main sequence that bends at
a point set by the cluster's age - which is how a cluster is dated.

WHAT YOU NEED

  two FITS images of the same cluster, one in B and one in G, plate-solved

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
  galactic extinction A_V         corrects the brightness
  colour excess E(B-V)            corrects the colour
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
    cluster_G_with_stars.png                   the G image with the members
                                               marked, so you can see whether
                                               the selection makes sense

  If you look at three files, look at these:

    calib_stars_apass.csv                 did the calibration have anything
                                          to work with
    cluster_G_with_stars.png              did the membership step pick the
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
import matplotlib.pyplot as plt


# ------------------- Photometry config -------------------
DETECTION_SIGMA = 4.0   # detection threshold in units of background sigma (controls how many stars are detected)
FWHM_HINT = 3.0         # fallback FWHM in pixels (used only if per-star FWHM estimation fails)
K_APERTURE = 2       # aperture radius scaling factor relative to each star's FWHM
K_ANNULUS_IN = 2      # inner radius of background annulus, in units of FWHM
K_ANNULUS_OUT = 4     # outer radius of background annulus, in units of FWHM
GAIN_E_PER_ADU = 1.0    # CCD gain: electrons per ADU (used for noise estimation if needed)
READ_NOISE_E = 5.0      # detector read noise in electrons (used for noise estimation if needed)

XY_MATCH_TOLERANCE = 15.0  # pixels

# --- Compute RA/DEC for every star using WCS with RA_Dec_TOLERANCE arcsec tolerance ---
RA_Dec_TOLERANCE = 60    # arcsec


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

# ------------------- Calibration stars from APASS -------------------


def get_apass_calib_stars(fits_file, n_stars=10, radius_deg=0.3, match_radius_arcsec = RA_Dec_TOLERANCE):
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
        Default = 60 arcsec (1 arcmin).

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
    Vizier.ROW_LIMIT = -1
    catalog = "II/336/apass9"
    result = Vizier(columns=["RAJ2000","DEJ2000","Bmag","Vmag"]).query_region(
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
        if sep2d.arcsec < match_radius_arcsec:  # default 60 arcsec = 1 arcmin
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



# ------------------- Run -------------------
print("=== Cluster Photometry Interactive Input ===") 
cluster_type = _ask("Cluster type [O=open, G=globular]", "O", str, upper=True)

if cluster_type == "G":
    # For globular clusters: ask raw name and radius
    cluster_name = _ask("Cluster name", "M13", str)
    Cluster_radius_px = float(input("Enter cluster radius in pixels (EXP: 200): "))

else:
    # For open clusters: ask name and normalize
    raw_name = _ask("Cluster name", "NGC_2682", str)
    try:
        cluster_name = normalize_cluster_name(raw_name)
    except RuntimeError as e:
        print(e)
        exit(1)
    Cluster_radius_px = None

        
Cluster_distance = float(input("Enter Cluster Distance (PC): (EXP: 850) "))
# ask user for galactic extinction coefficients
A_V = float(input("Enter Galactic extinction coefficient A_V (mag): (EXP: 0.13) "))
E_BV = float(input("Enter Galactic color excess E(B-V): (EXP: 0.041) "))
fits_file_B      = _ask("Path to B-band FITS image", r"D:\example_B.fts", str)
fits_file_G      = _ask("Path to G-band FITS image", r"D:\example_G.fts", str)
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

# join on nearest pixel
dfG["X_int"] = dfG["X"].round().astype(int)
dfG["Y_int"] = dfG["Y"].round().astype(int)
dfB["X_int"] = dfB["X"].round().astype(int)
dfB["Y_int"] = dfB["Y"].round().astype(int)

df = pd.merge(dfG[["X_int","Y_int","X","Y","FWHM","Aperture_Radius","Flux_G"]],
              dfB[["X_int","Y_int","Flux_B"]],
              on=["X_int","Y_int"], how="inner")

df = df[["X","Y","FWHM","Aperture_Radius","Flux_B","Flux_G"]]

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

        # KDTree matching on X,Y with tolerance 5 pixels
        flux_coords = df_flux[["X","Y"]].values
        calib_coords = df_calib[["X","Y"]].values
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
        hdr = fits.getheader(fits_file_B, ext=0)
        w = WCS(hdr)
        ra, dec = w.all_pix2world(df_cal["X"].values, df_cal["Y"].values, 0)
        
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

        matches = []
        for i, j in enumerate(idx):
            if j < len(df_cal) and dist[i] != np.inf:
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
        hdr = fits.getheader(fits_file_B, ext=0)  # use B-band header with WCS
        w = WCS(hdr)
        ra, dec = w.all_pix2world(df_gal["X"].values, df_gal["Y"].values, 0)

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

        matches = []
        for i, j in enumerate(idx):
            if j < len(df_gal) and dist[i] != np.inf:
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

                # make CMD plot
                
                plt.figure()
                plt.scatter(df["Color_index_corr"], df["Mag_V_corr_copy"], s=10, color="black")
                plt.gca().invert_yaxis()  # brighter = up
                plt.xlabel("Color index (B−V)")
                plt.ylabel("V magnitude (corrected)")
                plt.title(f"{cluster_name} ({cluster_type}), d={Cluster_distance} pc, A_V={A_V}, E(B−V)={E_BV}")

                # save and show plot
                plot_path = fpath.replace(".csv", "_CMD.png")
                plt.savefig(plot_path, dpi=150)
                #plt.savefig(plot_path, dpi=150)
                plt.show()
                #plt.close()

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

        plt.figure(figsize=(8,8))
        plt.imshow(data_G, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)

        # overlay measured stars
        plt.scatter(df_flux["X"], df_flux["Y"], 
                    s=30, edgecolor="red", facecolor="none", lw=1)

        plt.title(f"{cluster_name} - G-band with measured stars")
        plt.xlabel("X [pix]")
        plt.ylabel("Y [pix]")

        out_img_path = os.path.join(_outdir, "cluster_G_with_stars.png")
        plt.savefig(out_img_path, dpi=150)
        plt.show()

        print(f"[plot] wrote {out_img_path}")
    else:
        print("[plot] flux file not found, skipping star overlay plot")

except Exception as e:
    print(f"[plot] error: {e}")
# ---------------------------------------------------------------------




