# -*- coding: utf-8 -*-
"""
Created by: Dr. Boaz Ron Zohar
https://github.com/BoazRonZohar/ObservationalAstronomyEducationTools/blob/main/Cluster_CMD/Cluster_CMD.py
Affiliation: Kinneret Observatory
Member of the LCO Global Sky Partners programme
Date: September 2025, revised October 2026

Written for student projects on star clusters.

WHAT IT DOES

Takes a B-band and a V-band image of a star cluster - or the blue and green
channels of a colour camera - and produces its colour-magnitude diagram, in
absolute magnitude.

A cluster is the closest astronomy comes to a controlled experiment: the
stars formed together, sit at the same distance, and are seen through the
same dust. So their differences are real differences between stars. Plot
colour against brightness and they fall along a main sequence that bends at
a point set by the cluster's age - which is how a cluster is dated.

WHAT YOU NEED

  plate-solved FITS images of the cluster, in one of three forms:

  1. mono frames through B and V filters (the FILTER keyword says which).
  2. colour-camera channels already split into separate files - one each for
     B, G and R, recognised from FILTER or from the file name
     (..._B_..., ..._G_..., ..._R_...). G is the camera's GREEN channel, not
     Johnson V; the calibration's colour term takes care of the difference.
  3. raw colour-camera frames, three planes (R, G, B) in one FITS file. The
     frames are grouped by night and exposure time, the group with the most
     total exposure is aligned and median-combined, and its blue and green
     planes are measured as in case 2.

TWO KINDS OF CLUSTER, and the difference matters

  Open (O)       Membership comes from Cantat-Gaudin & Anders 2020
                 (Vizier J/A+A/640/A1), which used Gaia proper motions and
                 parallaxes to decide which stars actually belong. Field
                 stars that merely lie in the same direction are excluded.
                 Only stars with membership probability at least
                 OPEN_CLUSTER_PROB_MIN (0.5) are kept. The catalogue's own
                 name for the cluster is looked up for you - M6 is NGC_6405
                 there.

  Globular (G)   Two ways, you choose:
                   P  proper motions: Vasiliev & Baumgardt 2021 (Vizier
                      J/MNRAS/505/5978) give each globular's mean Gaia EDR3
                      proper motion and parallax; Gaia DR3 stars in the frame
                      that move with the cluster are taken as members. (Their
                      per-star membership list is not on Vizier, so it is
                      rebuilt here from the cluster's mean motion.)
                   R  radius: everything within a radius you give, in pixels,
                      of the cluster centre. The centre is SIMBAD's catalogue
                      position put through the plate solution, not the mean
                      of the detections.

WHAT IT ASKS YOU

  a folder of frames              which file is which is read from FILTER (or
                                  the file name, or the three-plane layout),
                                  the cluster name from OBJECT. Press Enter
                                  instead to name the files and the cluster
                                  yourself
  cluster type, O or G            only if SIMBAD does not say
  distance in parsecs             offered from the Harris catalogue for a
                                  globular, Dias for an open cluster, with its
                                  source shown; Enter accepts, or type your own
  colour excess E(B-V)            offered the same way. It corrects both the
                                  colour and the brightness. A_V is not asked
                                  for separately: it is 3.1 x E(B-V), so asking
                                  for both would let an impossible pair through
  membership method, and the      globular clusters only
  cluster radius in pixels        (the radius only for the radius method; the
                                  catalogue's half-light radius is shown in
                                  pixels as a yardstick)

  Every question can also be answered on the command line, so a run can be
  repeated exactly:

    python Cluster_CMD.py --folder "C:/data/M67" --yes --no-show
    python Cluster_CMD.py --help            (the full list)

  --yes accepts every default without asking; --no-show saves the figures
  without opening windows.

HOW IT WORKS

  1. measures a typical star width (FWHM) on each image from bright,
     isolated, unsaturated stars, and detects sources with that width
  2. measures every star through ONE aperture, K_APERTURE (2) x that typical
     FWHM, with the sky from a ring around it. One radius for the whole frame
     means every star loses the same fraction of its light, so the fraction
     cancels in the calibration. Each magnitude gets an error from the photon
     noise (detector gain) and the measured sky noise. Stars whose peak comes
     near the detector's saturation level are flagged and kept out of the
     calibration and the diagram
  3. matches the B and G detections to each other (registered with astroalign)
  4. calibrates against APASS DR9 (default) or Gaia DR3 synthetic Johnson
     photometry, using every catalogue star in the field that is matched,
     unsaturated and well measured, with sigma-clipped straight-line fits
     that include colour terms:
         (b-v)_instrumental = alpha + beta (B-V), inverted to
         (B-V) = a + c (b-v)_instrumental      (c = 1/beta; see fit_calibration
                                                for why this way round)
         V     = v_instrumental + ZP_V + k_V (B-V)
     If APASS gives too few stars, Gaia is used instead. If neither gives at
     least CALIB_MIN_STARS, the run stops rather than produce a diagram on a
     calibration it does not have
  5. checks that the colours point the right way: instrumental b-v must rise
     with catalogue B-V. If it falls, a colour camera's B and R channels were
     swapped (some software flips the image before separating the colours) -
     the R channel is then measured as blue, loudly. For filter frames the
     run stops: the files are probably mislabelled
  6. selects the members, by catalogue, by proper motion or by radius
  7. corrects for extinction and distance - M_V = V - A_V - (5 log10 d - 5) -
     and plots the diagram

WHAT YOU GET

Written into a NEW sub-folder for every run, next to your images:

    result_CMD_<cluster>_<YYYYMMDD_HHMMSS>

so nothing from an earlier run is overwritten, and no step reads a file from
an earlier run. The folder holding your data comes out of a run exactly as it
went in.

  A by-product, kept because it is what everything is measured on:

    <input>_bgsub.fits              each image with its background subtracted
    stack_<R|G|B>_....fits          (three-plane frames only) the aligned,
                                    median-combined colour planes

  Every source in the frame

    fluxes_XY_FWHM_Ap.csv           every detection matched in both bands:
                                    position, width, aperture, flux, its
                                    error, instrumental magnitude, peak pixel
                                    and saturation flag. Not yet calibrated.
    calib_stars_<catalogue>.csv     the catalogue stars used for the zero
                                    point and colour term, with catalogue and
                                    instrumental magnitudes, calibrated
                                    magnitudes and residuals, and whether each
                                    survived the sigma clipping. If the
                                    calibration looks wrong, look here first.
    fluxes_calibrated.csv           the same sources in real magnitudes, with
                                    errors
    fluxes_calibrated_galactic.csv  and after extinction is taken out
                                    (Mag_V_corr, BV_corr: apparent, dereddened)
                                    and the distance too (Abs_Mag_V = M_V)
    CMD_whole_frame_sources_all_<N>.png
                                    the diagram for the whole frame, members
                                    and field stars together

  Cluster members only

    fluxes_cluster_only.csv                    the members from the catalogue
                                               (or the radius)
    fluxes_cluster_only_calibrated.csv         matched to the photometry
    fluxes_cluster_only_galactic.csv           extinction and distance removed
    CMD_cluster_sources_all_<N>.png            the cluster's own diagram
    map_cluster_sources_all_<N>.png            the second-band image with the
                                               members marked

  Twice over, crowded sources set aside

    A source with another source inside its own aperture, or one too narrow or
    too wide to be a single star, is not measured honestly. Each diagram and
    map above is written again without them, as ..._no_overlap_<N>.png, with
    the table beside it as <scope>_no_overlap_<N>.csv. <N> is the number of
    sources in the picture.

    run_summary.txt                 every number the run used and found

  Saturated stars are in the tables (column Saturated = True) but not in the
  diagrams, whose <N> counts only the stars actually drawn. Typical error bars
  per 1-magnitude bin are drawn at the left edge of each diagram.

  If you look at three files, look at these:

    calib_stars_*.csv                     did the calibration have anything
                                          to work with
    map_cluster_sources_*.png             did the membership step pick the
                                          cluster and not the field
    CMD_cluster_sources_*.png             the diagram, in absolute magnitude

The parameters at the top of this file are tuned for the Kinneret and LCO
frames these projects use. On very different data, the detection threshold and
the matching tolerances are the first things to look at.

Usage: run it. Every question has a default; press Enter to accept it.
"""

import os
import re
import sys
import glob
import time
import shutil
import argparse
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.detection import DAOStarFinder
from photutils.aperture import (CircularAperture, CircularAnnulus,
                                aperture_photometry, ApertureStats)
from astroquery.vizier import Vizier
from scipy.spatial import cKDTree
import astroalign as aa
import matplotlib


# ------------------- Photometry config -------------------
DETECTION_SIGMA = 4.0   # detection threshold in units of background sigma (controls how many stars are detected)
PEAK_MIN_SIGMA = 10.0   # a detection is kept only if its peak is this many background sigma high
FWHM_HINT = 3.0         # starting guess for the FWHM in pixels; the real value is measured
K_APERTURE = 2.0        # aperture radius = K_APERTURE x the image's median FWHM, the same for every star
K_ANNULUS_IN = 3.0      # inner radius of the sky ring, in units of the median FWHM
K_ANNULUS_OUT = 5.0     # outer radius of the sky ring, in units of the median FWHM
FWHM_BOX_FACTOR = 5.0   # the box a star's width is measured in is at least this many FWHM across
GAIN_E_PER_ADU = 1.0    # CCD gain: electrons per ADU, used only if the header has no EGAIN/GAIN
READ_NOISE_E = 5.0      # read noise in electrons, used only if the header has no RDNOISE

# Saturation. A star whose brightest pixel comes near the level where the
# detector stops counting is not measured: its core is flattened and its
# flux is too low by an unknown amount. The level is SATURATE (or MAXLIN) from
# the header when there is one; otherwise it is found from the image itself -
# saturated stars pile up at one pixel value, which shows as a bump at the top
# of the pixel histogram. A star is flagged when its peak exceeds
# SATURATION_FRACTION of that level, because a detector stops being linear a
# little before it stops counting altogether.
SATURATION_FRACTION = 0.8

# Pixels this close to the frame edge are not used: colour-camera software
# leaves a few rows of nonsense there (values of millions on the Kinneret
# frames), and a star cut by the edge cannot be measured anyway.
EDGE_MARGIN_PX = 30

R_V = 3.1               # A_V / E(B-V) for dust in our own galaxy. A_V is
                        # derived from the colour excess rather than asked for
                        # separately - see where E(B-V) is read in.

# Matching a catalogue position (calibration star or cluster member) to a
# measured star, in G's pixel grid. Two independent measurements of the same
# star's position agree to about a pixel; 3 px is generous for that and still
# small next to the distance between stars. (This was 15 px, which in a
# crowded field picks up the wrong star as often as the right one.)
XY_MATCH_TOLERANCE = 3.0  # pixels

# Matching the B and G detections to each other, after registering B onto G's
# pixel grid with astroalign - see register_B_onto_G().
BG_MATCH_TOLERANCE = 3.0  # pixels, in G's own pixel grid

# Sanity check on the astroalign fit: B and G come off the same telescope, so
# the pixel scale between them should be the same to a fraction of a percent.
ASTROALIGN_MAX_SCALE_ERROR = 0.05  # 5%

# ------------------- Calibration config -------------------
# How close a detected star has to be to a catalogue entry, on the sky, to be
# treated as that star. A few arcsec is generous for how well two independent
# measurements of the same star agree.
CALIB_MATCH_RADIUS = 3.0  # arcsec
CALIB_CATALOGUE = "APASS"  # default; "GAIA" for Gaia DR3 synthetic B, V
CALIB_MIN_STARS = 5        # fewer than this and the run stops
CALIB_MAX_INST_ERR = 0.05  # mag; only well-measured stars set the calibration
CALIB_MAX_CAT_ERR = 0.10   # mag; and only well-measured catalogue entries
CALIB_CLIP_SIGMA = 3.0     # outliers beyond this many sigma are left out of the fit

# ------------------- Membership config -------------------
OPEN_CLUSTER_PROB_MIN = 0.5   # Cantat-Gaudin membership probability threshold
GC_PM_NSIGMA = 3.0            # how far from the cluster's mean proper motion a member may be
GC_PM_INTRINSIC = 0.1         # mas/yr, internal motions of the cluster plus systematics
GC_GAIA_GMAG_LIMIT = 20.0     # Gaia stars fainter than this are not fetched

# Stars whose calibrated B-V error is larger than this are left out of the
# diagrams (they stay in the tables). On a colour camera the instrumental
# colour is multiplied by about 3 in the calibration, and so is its noise, so
# the faintest stars would otherwise smear across the whole colour axis.
CMD_MAX_BV_ERR = 0.3

# The distance modulus mu = 5 log10(d) - 5 is applied so the diagram is in
# absolute magnitude M_V. Set False to plot apparent (dereddened) V instead.
APPLY_DISTANCE_MODULUS = True


# ------------------- Interactive user input -------------------
# The command-line arguments, if any. None means a plain interactive run.
ARGS = None


def _accept_defaults():
    return ARGS is not None and getattr(ARGS, "yes", False)


def _ask(prompt, default, cast=str, upper=False, choices=None):
    """Ask one question. Enter accepts the default shown in brackets.

    Only an empty answer means "the default". "0" used to be treated as the
    default too, which made it impossible to answer 0 - for instance an
    E(B-V) of 0. A default of None means there is none: an answer is needed.
    With choices, any other answer is refused and the question asked again.
    """
    if _accept_defaults():
        if default is None:
            sys.exit(f"[input] '{prompt}' has no default, so --yes cannot "
                     f"answer it - give it on the command line")
        print(f"{prompt} [{default}]: (accepted, --yes)")
        return default
    shown = "" if default is None else f" [{default}]"
    while True:
        s = input(f"{prompt}{shown}: ").strip()
        if s == "":
            if default is None:
                print("   an answer is needed here")
                continue
            return default
        try:
            v = cast(s)
        except Exception:
            print(f"   could not read '{s}' - try again")
            continue
        if upper and isinstance(v, str):
            v = v.upper()
        if choices and v not in choices:
            print(f"   answer one of: {', '.join(choices)}")
            continue
        return v


def _norm_path(p: str) -> str:
    p = p.strip().strip('"').strip("'")
    return os.path.normpath(p) if p else ""


# ------------------- Names and catalogues -------------------

def _squash(name):
    """A name reduced to letters and digits, for comparing catalogue names:
    'NGC 2682', 'NGC_2682' and 'ngc2682' all become 'NGC2682'."""
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def simbad_aliases(name):
    """Every identifier SIMBAD knows for this object, squashed. Used to check
    that a catalogue row really is the cluster asked for, and not a neighbour
    that happened to lie within the search radius."""
    aliases = {_squash(name)}
    try:
        from astroquery.simbad import Simbad
        ids = Simbad.query_objectids(name)
        if ids is not None and len(ids):
            col = ids.colnames[0]
            aliases.update(_squash(i) for i in ids[col])
    except Exception as exc:
        print(f"[lookup] SIMBAD identifiers unavailable ({exc})", flush=True)
    return aliases


def simbad_info(name):
    """Type ('O', 'G' or None) and catalogue position (SkyCoord or None).

    The type is a fact about the cluster, not a judgement: SIMBAD returns OpC
    for an open cluster and GlC for a globular. Asking for it is asking the
    person to repeat what the catalogue already knows. The position is used as
    the centre of a globular cluster.
    """
    try:
        from astroquery.simbad import Simbad
        query = Simbad()
        query.add_votable_fields("otype")
        result = query.query_object(name)
        if result is None or not len(result):
            return None, None
        cols = result.colnames
        otype = str(result["otype"][0]).strip() if "otype" in cols else ""
        kind = None
        if "GlC" in otype:
            kind = "G"
        elif "OpC" in otype or otype == "Cl*":
            kind = "O"
        centre = None
        if "ra" in cols and "dec" in cols:
            centre = SkyCoord(float(result["ra"][0]) * u.deg,
                              float(result["dec"][0]) * u.deg)
        elif "RA" in cols and "DEC" in cols:
            centre = SkyCoord(str(result["RA"][0]), str(result["DEC"][0]),
                              unit=(u.hourangle, u.deg))
        return kind, centre
    except Exception as exc:
        print(f"[lookup] SIMBAD unavailable ({exc})", flush=True)
    return None, None


def _matching_row(table, columns, aliases):
    """The first row of a catalogue table whose name, in any of these columns,
    is one of the cluster's names. None if there is no such row."""
    for row in table:
        for col in columns:
            if col in table.colnames and _squash(row[col]) in aliases:
                return row
    return None


def globular_from_harris(name, aliases):
    """Distance in pc, E(B-V) and half-light radius (arcmin) for a globular.

    Harris (1996, 2010 edition) is the standard compilation for the Milky Way's
    globulars: 147 of them, a distance for 145 and a reddening for every one.
    The row used is the one whose name matches the cluster - a search around a
    position can also return a neighbour.
    """
    try:
        tables = Vizier(columns=["**"]).query_object(name, catalog="VII/202")
        if not tables:
            return None, None, None
        distance = ebv = half_light = None
        found = False
        for table in tables:
            row = _matching_row(table, ("ID", "Name"), aliases)
            if row is None:
                continue
            found = True
            if distance is None and "Rsun" in table.colnames:
                distance = float(row["Rsun"]) * 1000.0
            if ebv is None and "E(B-V)" in table.colnames:
                ebv = float(row["E(B-V)"])
            if half_light is None and "Rh" in table.colnames:
                half_light = float(row["Rh"])
        if not found:
            print(f"[lookup] the Harris catalogue has no row named {name}",
                  flush=True)
        return distance, ebv, half_light
    except Exception as exc:
        print(f"[lookup] Harris catalogue unavailable ({exc})", flush=True)
        return None, None, None


def open_cluster_from_dias(name, aliases):
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
        row = _matching_row(table, ("Cluster",), aliases)
        if row is None:
            print(f"[lookup] the Dias catalogue has no row named {name}",
                  flush=True)
            return None, None, None

        def _num(col):
            try:
                v = float(row[col]) if col in table.colnames else None
                return v if v is not None and np.isfinite(v) else None
            except Exception:
                return None
        return _num("Dist"), _num("E(B-V)"), _num("Diam")
    except Exception as exc:
        print(f"[lookup] Dias catalogue unavailable ({exc})", flush=True)
        return None, None, None


MESSIER_TO_CG = {
    "M6":   "NGC_6405", "M7":   "NGC_6475", "M11":  "NGC_6705", "M18":  "NGC_6613",
    "M21":  "NGC_6531", "M23":  "NGC_6494", "M25":  "IC_4725",  "M26":  "NGC_6694",
    "M29":  "NGC_6913", "M34":  "NGC_1039", "M35":  "NGC_2168", "M36":  "NGC_1960",
    "M37":  "NGC_2099", "M38":  "NGC_1912", "M39":  "NGC_7092", "M41":  "NGC_2287",
    "M44":  "NGC_2632", "M45":  "Melotte_22", "M46": "NGC_2437", "M47": "NGC_2422",
    "M48":  "NGC_2548", "M50":  "NGC_2323", "M52":  "NGC_7654", "M67": "NGC_2682",
    "M93":  "NGC_2447", "M103": "NGC_581",
}


def normalize_cluster_name(user_input, aliases):
    """The Cantat-Gaudin catalogue's own name for an open cluster.

    Only the small table of cluster names (about 2,000 rows) is fetched for
    this, not the 86 MB table of member stars.
    """
    clusters = Vizier(columns=["Cluster"], row_limit=-1).get_catalogs(
        "J/A+A/640/A1/table1")[0]
    names = [str(c) for c in clusters["Cluster"]]
    by_squash = {_squash(c): c for c in names}

    name = user_input.strip().upper().replace(" ", "")
    if name in MESSIER_TO_CG:
        return MESSIER_TO_CG[name]
    if _squash(name) in by_squash:
        return by_squash[_squash(name)]
    for alias in aliases:
        if alias in by_squash:
            return by_squash[alias]
    raise RuntimeError(f"'{user_input}' is not an open cluster in the "
                       f"Cantat-Gaudin catalogue.")


def cantat_gaudin_members(cg_name):
    """The Cantat-Gaudin & Anders 2020 members of ONE cluster.

    Only this cluster's rows are asked for. The whole membership table is
    about 86 MB; it is downloaded only if the filtered query fails.
    """
    cols = ["RA_ICRS", "DE_ICRS", "Gmag", "proba", "Cluster"]
    print(f"Fetching the Cantat-Gaudin members of {cg_name}...", flush=True)
    t0 = time.time()
    try:
        res = Vizier(columns=cols, column_filters={"Cluster": f"={cg_name}"},
                     row_limit=-1).query_constraints(catalog="J/A+A/640/A1/nodup")
        table = res[0] if len(res) else None
    except Exception as exc:
        print(f"   filtered query failed ({exc})", flush=True)
        table = None
    if table is None or not len(table):
        print("   falling back to the whole membership table (about 86 MB)...",
              flush=True)
        full = Vizier(columns=cols, row_limit=-1).get_catalogs(
            "J/A+A/640/A1/nodup")[0]
        table = full[full["Cluster"] == cg_name]
    table = table[np.array([str(c) == cg_name for c in table["Cluster"]])]
    print(f"   {len(table):,} stars in {time.time() - t0:.0f} s", flush=True)
    return table


def vasiliev_baumgardt_row(aliases):
    """Mean proper motion and parallax of a globular from Vasiliev & Baumgardt
    2021 (J/MNRAS/505/5978, table A1), or None."""
    try:
        table = Vizier(columns=["**"], row_limit=-1).get_catalogs(
            "J/MNRAS/505/5978/tablea1")[0]
        return _matching_row(table, ("Name", "OName", "SimbadName"), aliases)
    except Exception as exc:
        print(f"[lookup] Vasiliev & Baumgardt catalogue unavailable ({exc})",
              flush=True)
        return None


# ------------------- Reading images -------------------

def read_image(path):
    """(data as float, header) from the first HDU that holds an image."""
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                return hdu.data.astype(np.float32), hdu.header.copy()
    raise RuntimeError(f"no image in {path}")


def is_three_plane(header):
    return header.get("NAXIS") == 3 and header.get("NAXIS3") == 3


def detector_gain(header):
    """Electrons per ADU of THIS file, and where the number came from.

    EGAIN is preferred: on the Kinneret colour camera GAIN is the camera's gain
    SETTING (51), not electrons per ADU. A file that is the median of N frames
    has less noise than one frame - as if the gain were N x 2/pi higher - and
    that is taken into account when the header says how many were combined.
    A sum of N frames keeps the gain of one frame: its counts are electrons
    divided by the same gain.
    """
    gain, source = None, None
    for key in ("EGAIN", "GAIN"):
        v = header.get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if 0 < v < 20:          # a real e-/ADU value, not a camera setting
            gain, source = v, key
            break
    if gain is None:
        gain, source = GAIN_E_PER_ADU, "default constant"
    n, how = combined_count(header)
    if n > 1 and how == "sum":
        return gain, f"{source} (sum of {n} frames)"
    if n > 1:
        factor = n * (2.0 / np.pi if how == "median" else 1.0)
        return gain * factor, f"{source} x {factor:.1f} ({how} of {n} frames)"
    return gain, source


def combined_count(header):
    """How many frames went into this file, and how they were combined."""
    n = header.get("NCOMBINE")
    how = str(header.get("COMBTYPE", "median")).lower()
    if n:
        return int(n), _combine_kind(how)
    for line in header.get("HISTORY", []):
        m = re.search(r"(median|average|mean|sum)\s+of\s+(\d+)\s+images",
                      str(line), re.I)
        if m:
            return int(m.group(2)), _combine_kind(m.group(1))
    return 1, "single"


def _combine_kind(word):
    word = str(word).lower()
    return "median" if "med" in word else "sum" if "sum" in word else "mean"


def read_noise(header):
    for key in ("RDNOISE", "READNOIS", "RDNOISE1", "RON"):
        try:
            v = float(header.get(key))
            if v > 0:
                return v, key
        except (TypeError, ValueError):
            continue
    return READ_NOISE_E, "default constant"


def saturation_level(raw, header):
    """The pixel value at which a star counts as saturated, and how it was found.

    From the header's SATURATE or MAXLIN when there is one. Otherwise from the
    image: saturated stars pile up at the top of the pixel histogram. The
    Kinneret colour frames show this clearly - after colour conversion the
    clipping level is not the largest pixel value but sits at 70-90 percent of
    it, so simply taking the maximum would miss most saturated stars.
    """
    for key in ("SATURATE", "MAXLIN", "SATLEVEL"):
        try:
            v = float(header.get(key))
        except (TypeError, ValueError):
            continue
        if v > 0:
            return SATURATION_FRACTION * v, f"{SATURATION_FRACTION:g} x {key} ({v:g})"

    m = EDGE_MARGIN_PX
    inner = raw[m:-m, m:-m] if min(raw.shape) > 4 * m else raw
    inner = inner[np.isfinite(inner)]
    top = float(np.sort(inner)[-20])          # robust to a few hot pixels
    high = inner[inner > 0.4 * top]
    level = top
    how = "the image's maximum"
    if high.size > 200:
        counts, edges = np.histogram(high, bins=60, range=(0.4 * top, top))
        for i in range(10, len(counts)):
            before = np.median(counts[i - 10:i])
            if counts[i] >= 20 and counts[i] > 2.0 * max(before, 1.0):
                level = float(edges[i])
                how = "the pile-up of saturated pixels in the histogram"
                break
    return SATURATION_FRACTION * level, f"{SATURATION_FRACTION:g} x {how} ({level:.0f})"


# ------------------- Recognising the frames -------------------

_BAND_WORDS = {"B": "B", "BLUE": "B", "V": "V", "G": "G", "GREEN": "G",
               "R": "R", "RED": "R", "RP": "R", "RC": "R"}


def band_of(path, header):
    """'B', 'V', 'G', 'R', 'RGB' (three planes) or None.

    From FILTER first. Without it, from the file name - but only from a whole
    word of it, separated by _ - space or dot, so that 'M 67_B_13.fts' is blue
    while '..._Bin2...' and '...-B1-color...' (binning 1) are not.
    """
    if is_three_plane(header):
        return "RGB"
    filt = str(header.get("FILTER", "")).strip().upper().replace("'", "P")
    if filt:
        filt = re.sub(r"^(JOHNSON|BESSELL|BESSEL)[\s_-]*", "", filt)
        if filt in _BAND_WORDS:
            return _BAND_WORDS[filt]
        if filt[:1] in ("B", "V") and len(filt) <= 2:
            return filt[:1]
    stem = os.path.splitext(os.path.basename(path))[0].upper()
    tokens = [t for t in re.split(r"[\s_.\-@]+", stem) if t]
    for t in reversed(tokens):          # the band is usually near the end
        if t in _BAND_WORDS and t != "RP":
            return _BAND_WORDS[t]
    return None


def frames_in_folder(folder):
    """The frames in this folder, sorted by band, and the cluster name.

    WHY NOT ASK
    The filter is written into every frame the telescope produces. Asking a
    person to type it invites the one mistake that cannot be caught later: two
    paths swapped, and every colour in the diagram comes out with its sign
    reversed while the run looks perfectly normal throughout. (The colour check
    after the calibration would now catch that - but better not to need it.)

    Returns ({band: [paths]}, object_name).
    """
    found, names = {}, []
    paths = []
    for pattern in ("*.fits", "*.fit", "*.fts"):
        paths += glob.glob(os.path.join(folder, pattern))
    for path in sorted(set(paths)):
        try:
            head = fits.getheader(path)
            if head.get("NAXIS", 0) == 0 and "XTENSION" not in head:
                with fits.open(path) as hdul:
                    for hdu in hdul:
                        if hdu.data is not None and hdu.data.ndim >= 2:
                            head = hdu.header
                            break
        except Exception:
            continue
        band = band_of(path, head)
        if head.get("OBJECT"):
            names.append(" ".join(str(head["OBJECT"]).split()).upper())
        if band:
            found.setdefault(band, []).append(path)
    name = max(set(names), key=names.count) if names else None
    return found, name


def describe_choice(found, keys):
    """Say which frame was taken for each band, and what else was there.

    WHY THE REST ARE NAMED
    Because a folder often holds more than one frame of a filter and only one of
    them is used. The others are simply named, so that the choice can be seen
    and the folder tidied if it was the wrong one.
    """
    chosen = {}
    for key in keys:
        paths = found.get(key, [])
        if not paths:
            chosen[key] = None
            continue
        chosen[key] = paths[0]
        print(f"   {key}: {os.path.basename(paths[0])}")
        if len(paths) > 1:
            others = ", ".join(os.path.basename(p) for p in paths[1:])
            print(f"      {len(paths) - 1} more in this band, not used: {others}")
    return chosen


def flip_notes(headers):
    """Header lines saying the image was flipped or mirrored, or naming the
    Bayer pattern / row order - the usual reason a colour camera's R and B
    come out swapped. Returns a list of short strings."""
    notes = []
    for h in headers:
        for line in h.get("HISTORY", []):
            if re.search(r"flip|mirror", str(line), re.I):
                notes.append(f"HISTORY '{str(line).strip()}'")
        for key in ("FLIPSTAT", "ROWORDER", "BAYERPAT"):
            if key in h:
                notes.append(f"{key} = '{str(h[key]).strip()}'")
    seen, out = set(), []
    for n in notes:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


# ------------------- Three-plane colour frames -------------------

def night_of(header):
    """The date the night STARTED, so a night crossing midnight stays one."""
    d = str(header.get("DATE-OBS", ""))[:19]
    try:
        t = datetime.fromisoformat(d)
        return (t - timedelta(hours=12)).date().isoformat()
    except ValueError:
        return d[:10] or "unknown"


def plane_order(header):
    """Which colour each of the three planes holds. R, G, B (MaxIm DL, most
    capture software) unless the header names the planes otherwise."""
    order = ["R", "G", "B"]
    named = {}
    for key in header:
        m = re.match(r"^(PLANE|CHAN|CHANNEL|LAYER|COLOR)_?(\d)$", key)
        if m:
            val = str(header[key]).strip().upper()
            if val[:1] in ("R", "G", "B"):
                named[int(m.group(2))] = val[:1]
    if len(named) == 3:
        start = min(named)
        order = [named[start + i] for i in range(3)]
    return order


def choose_colour_group(paths):
    """Group three-plane frames by night and exposure time, and pick the group
    with the largest total exposure. Different nights have different sky and
    focus, and frames of different length saturate at different magnitudes,
    so they are not mixed."""
    groups = {}
    for p in paths:
        h = fits.getheader(p)
        key = (night_of(h), float(h.get("EXPTIME", h.get("EXPOSURE", 0)) or 0))
        groups.setdefault(key, []).append(p)
    ranked = sorted(groups.items(),
                    key=lambda kv: (kv[0][1] * len(kv[1]), len(kv[1])),
                    reverse=True)
    print(f"   {len(paths)} three-plane colour frames, in {len(groups)} groups "
          f"(night, exposure):")
    for i, ((night, exp), ps) in enumerate(ranked):
        mark = "  <- used" if i == 0 else ""
        print(f"      night of {night}, {exp:g} s x {len(ps)} = "
              f"{exp * len(ps):g} s{mark}")
    return ranked[0][1]


def stack_colour_frames(paths, out_dir):
    """Align and median-combine three-plane colour frames.

    Each frame is registered onto the first from its green plane (astroalign);
    the same transformation is applied to all three planes. The median is
    taken plane by plane through a temporary file on disk, so a dozen 108 MB
    frames fit in a few hundred MB of memory. Returns {'R': path, 'G': path,
    'B': path} of three 2-D FITS files written into out_dir, and the headers
    of the frames used.
    """
    n = len(paths)
    ref, ref_hdr = read_image(paths[0])
    order = plane_order(ref_hdr)
    if order != ["R", "G", "B"]:
        print(f"   the header names the planes {order}")
    else:
        print("   planes taken as R, G, B (the header does not say otherwise)")
    g_index = order.index("G")
    _, ny, nx = ref.shape
    m = EDGE_MARGIN_PX

    def _clean_edges(plane):
        # the frame edge holds junk from the colour conversion; overwrite it
        # with the sky level so it neither confuses the alignment nor the stack
        med = float(np.nanmedian(plane[m:-m, m:-m]))
        plane[:m, :] = med
        plane[-m:, :] = med
        plane[:, :m] = med
        plane[:, -m:] = med
        return np.nan_to_num(plane, nan=med)

    tmp = tempfile.mkdtemp(prefix="cluster_cmd_stack_")
    headers = [ref_hdr]
    try:
        cubes = [np.lib.format.open_memmap(os.path.join(tmp, f"plane{p}.npy"),
                                           mode="w+", dtype=np.float32,
                                           shape=(n, ny, nx)) for p in range(3)]
        ref_G = _clean_edges(ref[g_index].copy())
        for p in range(3):
            cubes[p][0] = _clean_edges(ref[p].copy())
        del ref
        for k, path in enumerate(paths[1:], start=1):
            data, hdr = read_image(path)
            headers.append(hdr)
            planes = [_clean_edges(data[p].copy()) for p in range(3)]
            del data
            transf, (src, _tgt) = aa.find_transform(planes[g_index], ref_G)
            if abs(transf.scale - 1.0) > ASTROALIGN_MAX_SCALE_ERROR:
                raise RuntimeError(f"alignment of {os.path.basename(path)} "
                                   f"failed (scale {transf.scale:.3f})")
            for p in range(3):
                aligned, _fp = aa.apply_transform(transf, planes[p], ref_G,
                                                  fill_value=np.nan)
                cubes[p][k] = aligned.astype(np.float32)
            print(f"   aligned {k + 1} of {n}: {os.path.basename(path)} - "
                  f"{len(src)} stars, shift ({transf.translation[0]:+.1f}, "
                  f"{transf.translation[1]:+.1f}) px, rotation "
                  f"{np.degrees(transf.rotation):+.2f} deg", flush=True)

        # a 2-D header with the reference frame's plate solution
        hdr2 = ref_hdr.copy()
        for key in ("NAXIS3", "BZERO", "BSCALE"):
            hdr2.remove(key, ignore_missing=True, remove_all=True)
        hdr2["NAXIS"] = 2
        hdr2["NCOMBINE"] = (n, "frames median-combined by Cluster_CMD")
        hdr2["COMBTYPE"] = "median"
        hdr2.add_history(f"Cluster_CMD: aligned (astroalign) and median of "
                         f"{n} images: " + ", ".join(os.path.basename(p) for p in paths))

        exp = float(ref_hdr.get("EXPTIME", 0) or 0)
        out = {}
        for p, colour in enumerate(order):
            stack = np.empty((ny, nx), dtype=np.float32)
            for r0 in range(0, ny, 256):
                stack[r0:r0 + 256] = np.nanmedian(cubes[p][:, r0:r0 + 256], axis=0)
            med = float(np.nanmedian(stack))
            stack = np.nan_to_num(stack, nan=med)
            h = hdr2.copy()
            h["FILTER"] = (colour, "colour-camera channel (not a Johnson filter)")
            path_out = os.path.join(out_dir, f"stack_{colour}_{n}x{exp:g}s.fits")
            fits.writeto(path_out, stack, h)
            out[colour] = path_out
            print(f"[stack] wrote {path_out}", flush=True)
        del cubes
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out, headers


# ------------------- Background subtraction -------------------
def subtract_background_and_save(path, out_dir):
    raw, hdr = read_image(path)
    median_val = float(np.nanmedian(raw))
    data_sub = raw - median_val
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, stem + "_bgsub.fits")
    hdr = hdr.copy()
    for key in ("BZERO", "BSCALE"):
        hdr.remove(key, ignore_missing=True)
    fits.writeto(out_path, data_sub.astype(np.float32), hdr)
    print(f"[bgsub] wrote {out_path} (median={median_val:.3f})")
    return out_path, raw, data_sub, hdr


# ------------------- Core photometry helpers -------------------
def fwhm_box(fwhm):
    """The side of the box a star's width is measured in: at least
    FWHM_BOX_FACTOR x FWHM, odd, and never under 15 px. A fixed 15 px box cut
    off the wings of the stars on frames with a 5-6 px FWHM and so made every
    star look narrower than it was."""
    b = int(np.ceil(max(15, FWHM_BOX_FACTOR * fwhm)))
    return b + 1 if b % 2 == 0 else b


def estimate_fwhm_moments(img, x, y, box=15, fwhm_w=None):
    """A star's FWHM from its Gaussian-weighted second moments.

    Plain moments of a background-subtracted cutout grow with the size of the
    box: the noise in the outer pixels (clipped at zero, so it cannot average
    out) is weighted by the square of its distance from the centre. On a 1 s
    LCO frame that took a 3.5 px star to 10-20 px. Weighting the pixels with a
    Gaussian of width fwhm_w (the image's typical FWHM) keeps the outer noise
    out, and for a Gaussian star the weighting can be undone exactly:
        sigma_star^2 = M sigma_w^2 / (sigma_w^2 - M)
    where M is the measured weighted second moment.
    """
    h, w = img.shape
    x0 = int(round(x)); y0 = int(round(y))
    x1 = max(0, x0 - box//2); x2 = min(w, x0 + box//2 + 1)
    y1 = max(0, y0 - box//2); y2 = min(h, y0 + box//2 + 1)
    if x2 <= x1+2 or y2 <= y1+2:
        return np.nan
    cut = img[y1:y2, x1:x2].astype(float)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    r2 = (yy - y) ** 2 + (xx - x) ** 2
    # the sky from the outer part of the box, beyond 0.4 x its side
    ann = r2 >= (0.4 * box) ** 2
    if np.sum(ann) >= 20:
        bg_mean, _, _ = sigma_clipped_stats(cut[ann], sigma=3.0, maxiters=5)
    else:
        bg_mean = float(np.nanmedian(cut))
    cut_bs = cut - bg_mean
    if fwhm_w is None:
        fwhm_w = box / FWHM_BOX_FACTOR
    sw2 = (fwhm_w / 2.3548) ** 2
    wgt = cut_bs * np.exp(-0.5 * r2 / sw2)
    flux = wgt.sum()
    if not np.isfinite(flux) or flux <= 0:
        return np.nan
    x_mean = (wgt * (xx - x)).sum() / flux
    y_mean = (wgt * (yy - y)).sum() / flux
    m2 = 0.5 * (wgt * ((xx - x - x_mean) ** 2 + (yy - y - y_mean) ** 2)).sum() / flux
    if not np.isfinite(m2) or m2 <= 0 or m2 >= 0.95 * sw2:
        return np.nan
    return 2.3548 * float(np.sqrt(m2 * sw2 / (sw2 - m2)))


def _detect(data, std, fwhm, edge):
    tbl = DAOStarFinder(fwhm=fwhm, threshold=DETECTION_SIGMA * std)(data)
    if tbl is None or len(tbl) == 0:
        return None
    tbl = tbl[tbl["peak"] > PEAK_MIN_SIGMA * std]
    ny, nx = data.shape
    x = np.asarray(tbl["xcentroid"], float)
    y = np.asarray(tbl["ycentroid"], float)
    ok = (x > edge) & (x < nx - 1 - edge) & (y > edge) & (y < ny - 1 - edge)
    return tbl[ok]


def typical_fwhm(data, raw, std, sat_level, band):
    """The median FWHM of bright, isolated, unsaturated stars, in pixels.

    The weighted-moment width depends a little on the weight's own width, so
    the measurement is repeated with the weight set to the last answer until
    it settles (three or four rounds).
    """
    fwhm = FWHM_HINT
    tbl = _detect(data, std, FWHM_HINT, EDGE_MARGIN_PX)
    if tbl is None or len(tbl) < 5:
        print(f"   too few stars to measure the FWHM in {band}; using {fwhm} px")
        return fwhm
    x = np.asarray(tbl["xcentroid"], float)
    y = np.asarray(tbl["ycentroid"], float)
    peak = np.asarray(tbl["peak"], float)
    nearest = cKDTree(np.column_stack([x, y])).query(
        np.column_stack([x, y]), k=2)[0][:, 1]
    raw_peak = np.array([np.nanmax(raw[int(round(yy)) - 1:int(round(yy)) + 2,
                                       int(round(xx)) - 1:int(round(xx)) + 2])
                         for xx, yy in zip(x, y)])
    good = (nearest > 25) & (peak > 30 * std) & (raw_peak < sat_level)
    idx = np.where(good)[0]
    if len(idx) < 5:
        idx = np.where(raw_peak < sat_level)[0]
    idx = idx[np.argsort(peak[idx])[::-1][:200]]
    for _ in range(6):
        box = fwhm_box(fwhm)
        widths = np.array([estimate_fwhm_moments(data, x[i], y[i], box, fwhm)
                           for i in idx])
        widths = widths[np.isfinite(widths) & (widths > 0.8)]
        if len(widths) < 5:
            break
        new = float(np.median(widths))
        done = abs(new - fwhm) < 0.02 * fwhm
        fwhm = new
        if done:
            break
    print(f"   typical FWHM in {band}: {fwhm:.2f} px (from {len(idx)} bright isolated stars)",
          flush=True)
    return fwhm


# ------------------- Main photometry routine -------------------
def run_photometry(path, band, out_dir):
    """Detect and measure every star in one image.

    Returns a table (X, Y, FWHM, Aperture_Radius, Flux, Flux_err, Mag_inst,
    Mag_inst_err, Peak, Saturated - the last five with the band appended) and
    a dict describing the image.
    """
    bgsub_path, raw, data, hdr = subtract_background_and_save(path, out_dir)
    _mean, _median, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    data = data - _median
    sat_level, sat_src = saturation_level(raw, hdr)
    gain, gain_src = detector_gain(hdr)
    rn, rn_src = read_noise(hdr)
    print(f"   {band}: saturation level {sat_level:.0f} ADU ({sat_src}); gain "
          f"{gain:.2f} e-/ADU ({gain_src}); read noise {rn:g} e- ({rn_src})",
          flush=True)

    fwhm = typical_fwhm(data, raw, std, sat_level, band)
    r_ap = K_APERTURE * fwhm
    r_in, r_out = K_ANNULUS_IN * fwhm, K_ANNULUS_OUT * fwhm
    edge = max(EDGE_MARGIN_PX, int(np.ceil(r_out)) + 2)
    tbl = _detect(data, std, fwhm, edge)
    if tbl is None or len(tbl) == 0:
        raise RuntimeError(f"No sources detected in {band}")
    # One star found two or three times over. A trailed or elongated star -
    # the 120 s M67 colour frames drifted about 10 px during each exposure -
    # is split by the detector into a row of "sources" 2-3 px apart, each of
    # which would then be measured as a star of its own. Two real stars closer
    # than one FWHM cannot be measured apart anyway, so only the brightest
    # detection within one FWHM is kept.
    tbl.sort("flux", reverse=True)
    x = np.asarray(tbl["xcentroid"], float)
    y = np.asarray(tbl["ycentroid"], float)
    keep = np.ones(len(x), dtype=bool)
    tree = cKDTree(np.column_stack([x, y]))
    for i in range(len(x)):
        if keep[i]:
            for j in tree.query_ball_point([x[i], y[i]], r=fwhm):
                if j > i:
                    keep[j] = False
    if (~keep).sum():
        print(f"   {int((~keep).sum()):,} detections within one FWHM of a brighter "
              f"one dropped as duplicates", flush=True)
    x, y = x[keep], y[keep]
    positions = np.column_stack([x, y])
    print(f"Measuring {len(x):,} sources in the {band} image "
          f"(aperture {r_ap:.1f} px = {K_APERTURE:g} x FWHM, sky ring "
          f"{r_in:.1f}-{r_out:.1f} px)...", flush=True)

    ap = CircularAperture(positions, r=r_ap)
    ann = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
    sky = ApertureStats(data, ann, sigma_clip=SigmaClip(sigma=3.0, maxiters=5))
    sky_med = np.asarray(sky.median, float)
    sky_std = np.asarray(sky.std, float)
    n_sky = np.asarray(sky.sum_aper_area.value, float)
    phot = aperture_photometry(data, ap)
    n_ap = float(ap.area)
    flux = np.asarray(phot["aperture_sum"], float) - sky_med * n_ap

    # Noise: photons from the star, plus the sky (and read) noise measured in
    # the ring, which counts once per aperture pixel and once more through the
    # uncertainty of the sky level itself. The measured ring scatter already
    # contains the read noise; the header value is a floor under it.
    pix_var = np.maximum(sky_std ** 2, (rn / gain) ** 2)
    var = np.clip(flux, 0, None) / gain + n_ap * pix_var * (1.0 + n_ap / np.maximum(n_sky, 1))
    flux_err = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(flux > 0, -2.5 * np.log10(flux), np.nan)
        mag_err = np.where(flux > 0, 1.0857 * flux_err / flux, np.nan)

    # the brightest pixel within one FWHM, in the ORIGINAL frame
    peak = np.asarray(ApertureStats(raw, CircularAperture(positions, r=max(1.5, fwhm))).max, float)
    saturated = peak > sat_level

    box = fwhm_box(fwhm)
    widths = np.array([estimate_fwhm_moments(data, xx, yy, box, fwhm) for xx, yy in zip(x, y)])

    df = pd.DataFrame({
        "X": x, "Y": y, "FWHM": widths, "Aperture_Radius": r_ap,
        f"Flux_{band}": flux, f"Flux_{band}_err": flux_err,
        f"Mag_inst_{band}": mag, f"Mag_inst_{band}_err": mag_err,
        f"Peak_{band}": peak, f"Saturated_{band}": saturated,
    })
    df = df[np.isfinite(df[f"Mag_inst_{band}"])].reset_index(drop=True)
    print(f"   {len(df):,} measured, {int(df[f'Saturated_{band}'].sum()):,} "
          f"saturated", flush=True)
    info = {"path": path, "bgsub": bgsub_path, "fwhm": fwhm, "r_ap": r_ap,
            "sat_level": sat_level, "sat_src": sat_src, "gain": gain,
            "gain_src": gain_src, "header": hdr}
    return df, info, data


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

    Two tests, because each catches what the other misses.

    THE WIDTH TEST catches a blend the detector reported as one source: two stars
    close enough to merge are measured as one object about twice as wide. It also
    catches the opposite - a cosmic ray or a hot pixel, narrower than the optics
    can make any real source.

    THE NEIGHBOUR TEST catches a blend the detector did separate. Two stars ten
    pixels apart can each be measured at a normal width, and each one's aperture
    still swallows the other, because the aperture is wider than the gap. This
    looks at the actual distance to the nearest source rather than at the shape.

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

    if "Nearest_px" in df.columns:
        crowded = df["Nearest_px"].to_numpy(float) < df["Aperture_Radius"].to_numpy(float)
    else:
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


# ------------------- Registering B onto G -------------------

def register_B_onto_G(B_img, G_img, xB, yB, fits_file_B, fits_file_G):
    """Where each B-image position (xB, yB) lands in G's pixel grid.

    B and G are separate exposures - the telescope moves between them, if
    only by a little - so their pixel grids do not coincide, and by how much
    is not knowable in advance. On one real pair of M67 frames it was about
    18 px of shift plus roughly a degree of rotation.

    astroalign finds the registration directly from the star patterns in
    these two specific images - it does not assume any telescope, field, or
    plate scale, and does not depend on how good either WCS solution is.
    (For the channels of one colour exposure it simply finds no shift.)

    Falls back to each frame's own WCS only if astroalign cannot find enough
    matching stars, or its fitted scale is not plausible for two frames from
    the same telescope."""
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
        wcs_B = WCS(fits.getheader(fits_file_B), naxis=2)
        wcs_G = WCS(fits.getheader(fits_file_G), naxis=2)
        ra_B, dec_B = wcs_B.all_pix2world(xB, yB, 0)
        return wcs_G.all_world2pix(ra_B, dec_B, 0)


def match_B_to_G(dfB, dfG, B_img, G_img, file_B, file_G):
    """One row per star found in both bands, in G's pixel grid."""
    xB_in_G, yB_in_G = register_B_onto_G(B_img, G_img, dfB["X"].values,
                                         dfB["Y"].values, file_B, file_G)
    tree = cKDTree(dfG[["X", "Y"]].values)
    dist, idx = tree.query(np.column_stack([xB_in_G, yB_in_G]),
                           distance_upper_bound=BG_MATCH_TOLERANCE)
    # If more than one B star lands within tolerance of the same G star - which
    # happens in a crowded field - only the closer of the two is kept, so that G
    # star's flux is not duplicated onto two rows.
    best_for_g = {}
    for i in range(len(dfB)):
        if not np.isfinite(dist[i]):
            continue
        j = int(idx[i])
        if j not in best_for_g or dist[i] < best_for_g[j][1]:
            best_for_g[j] = (i, dist[i])
    gi = np.array([j for j in best_for_g], dtype=int)
    bi = np.array([best_for_g[j][0] for j in best_for_g], dtype=int)
    g = dfG.iloc[gi].reset_index(drop=True)
    b = dfB.iloc[bi].reset_index(drop=True)
    df = g[["X", "Y", "FWHM", "Aperture_Radius"]].copy()
    for col in ("Flux_B", "Flux_B_err", "Mag_inst_B", "Mag_inst_B_err",
                "Peak_B", "Saturated_B"):
        df[col] = b[col].values
    for col in ("Flux_G", "Flux_G_err", "Mag_inst_G", "Mag_inst_G_err",
                "Peak_G", "Saturated_G"):
        df[col] = g[col].values
    df["Saturated"] = df["Saturated_B"] | df["Saturated_G"]
    # the distance to the nearest OTHER detection in G - used to recognise
    # stars that share their aperture with a neighbour
    xyG = dfG[["X", "Y"]].values
    d2, _ = cKDTree(xyG).query(df[["X", "Y"]].values, k=2)
    df["Nearest_px"] = d2[:, 1]
    print(f"[out] matched {len(df):,} stars between the {len(dfG):,} found in G "
          f"and the {len(dfB):,} found in B (within {BG_MATCH_TOLERANCE:.0f} px "
          f"of each other, in G's pixel grid, after registration)")
    return df


# ------------------- Checking the plate solution -------------------

# What fraction of the brightest detections must have a Gaia star within
# CALIB_MATCH_RADIUS for the plate solution to be trusted as it is.
WCS_CHECK_MIN_FRACTION = 0.5


def verify_plate_solution(df_G, wcs, shape):
    """The plate solution of the G image, checked against Gaia and repaired if
    it does not fit.

    WHY
    Everything that uses the sky - the calibration stars, the cluster members -
    goes through this solution. Software that combines or re-registers frames
    sometimes keeps the header of a DIFFERENT frame: the split M67 channel
    files from AIP4Win carry a solution that is off by about 45 px (55 arcsec)
    from the image they sit on, so only 4 of 2,000 APASS stars matched and
    the calibration had nothing to work with. And the raw Kinneret colour
    frames were cropped by MaxIm DL AFTER PinPoint solved them, which leaves
    the solution about 10 px (12 arcsec) off in y.

    HOW
    The 100 brightest unsaturated detections are put through the solution and looked up
    in Gaia DR3. If at least half have a Gaia star within a few arcsec, the
    solution is used as it is. Otherwise the pattern of bright Gaia stars,
    projected through the header's solution (whose scale and rotation are
    usually right even when its position is not), is matched to the pattern
    of detections with astroalign - trying the mirror image too - and a new
    TAN solution is fitted to every star that then pairs up.
    """
    from astropy.wcs.utils import fit_wcs_from_points
    ny, nx = shape
    centre, radius = field_centre_and_radius(wcs, shape)
    # unsaturated stars only: a saturated star's centre is poorly defined and
    # a big one is often detected two or three times over
    bright = df_G[~df_G["Saturated_G"].astype(bool)].sort_values("Flux_G", ascending=False)
    xy = bright[["X", "Y"]].to_numpy(float)
    try:
        res = Vizier(columns=["RA_ICRS", "DE_ICRS", "Gmag"],
                     column_filters={"Gmag": "<17"}, row_limit=-1).query_region(
            SkyCoord(centre.ra, centre.dec), radius=1.3 * radius * u.deg,
            catalog="I/355/gaiadr3")
        gaia = res[0]
    except Exception as exc:
        print(f"[wcs] Gaia unavailable ({exc}) - the plate solution is not checked")
        return wcs, "not checked"
    gaia.sort("Gmag")
    gsky = SkyCoord(np.asarray(gaia["RA_ICRS"], float) * u.deg,
                    np.asarray(gaia["DE_ICRS"], float) * u.deg)

    def fraction(w, n=100):
        ra, dec = w.all_pix2world(xy[:n, 0], xy[:n, 1], 0)
        _i, sep, _ = SkyCoord(ra * u.deg, dec * u.deg).match_to_catalog_sky(gsky)
        ok = sep.arcsec < CALIB_MATCH_RADIUS
        return ok.mean(), (np.median(sep.arcsec[ok]) if ok.any() else np.nan)

    f0, s0 = fraction(wcs)
    print(f"[wcs] {100 * f0:.0f}% of the 100 brightest stars have a Gaia star "
          f"within {CALIB_MATCH_RADIUS:g} arcsec (median {s0:.2f} arcsec)", flush=True)
    if f0 >= WCS_CHECK_MIN_FRACTION:
        return wcs, f"checked against Gaia: {100 * f0:.0f}% of bright stars match"

    print("[wcs] the plate solution in the header does not fit this image - "
          "repairing it from the star pattern", flush=True)
    gx, gy = wcs.all_world2pix(gsky.ra.deg, gsky.dec.deg, 0)
    near = (gx > -0.05 * nx) & (gx < 1.05 * nx) & (gy > -0.05 * ny) & (gy < 1.05 * ny)
    src_all = np.column_stack([gx, gy])
    det_tree = cKDTree(xy[:600])
    max_off = 0.25 * max(nx, ny)
    best = None
    for parity in (1, -1):
        src = src_all.copy()
        if parity < 0:
            src[:, 0] = (nx - 1) - src[:, 0]
        cand = src[near][:600]
        # First the simple case: the solution's scale and rotation are right
        # and only its position is off. Every catalogue star then sits at the
        # same offset from its detection, which shows as one sharp peak among
        # the offsets of all catalogue-detection pairs.
        dxy = []
        for k, nb in enumerate(det_tree.query_ball_point(cand, r=max_off)):
            if nb:
                dxy.append(xy[nb] - cand[k])
        if dxy:
            dxy = np.vstack(dxy)
            bins = np.arange(-max_off, max_off + 4, 4.0)
            H, ex, ey = np.histogram2d(dxy[:, 0], dxy[:, 1], bins=[bins, bins])
            i, j = np.unravel_index(np.argmax(H), H.shape)
            peak = H[i, j]
            if peak >= 10 and peak > 8 * max(np.median(H[H > 0]), 1):
                shift = np.array([0.5 * (ex[i] + ex[i + 1]), 0.5 * (ey[j] + ey[j + 1])])
                T = lambda p, s_=shift: p + s_
                if best is None or peak > best[2]:
                    best = (parity, T, int(peak))
                continue
        # Otherwise match the star patterns, which allows for rotation too.
        for n_cat, n_det in ((150, 150), (200, 200), (300, 300)):
            try:
                T, (s, _t) = aa.find_transform(cand[:n_cat], xy[:n_det])
            except Exception:
                continue
            if abs(T.scale - 1.0) > 0.1:
                continue
            if best is None or len(s) > best[2]:
                best = (parity, T, len(s))
            break
    if best is None:
        stop("PLATE SOLUTION: the header's solution does not fit the image and "
             "could not be repaired. Plate-solve the G/V image again (e.g. "
             "astrometry.net) and rerun.")
    parity, T, nctl = best
    src = src_all.copy()
    if parity < 0:
        src[:, 0] = (nx - 1) - src[:, 0]
    guess = T(src)
    tree = cKDTree(df_G[["X", "Y"]].to_numpy(float))
    new = wcs
    tol = 5.0
    for _ in range(3):
        d, j = tree.query(guess, distance_upper_bound=tol)
        ok = np.isfinite(d)
        if ok.sum() < 10:
            break
        px = df_G[["X", "Y"]].to_numpy(float)[j[ok]]
        new = fit_wcs_from_points((px[:, 0], px[:, 1]), gsky[ok],
                                  proj_point="center", projection="TAN")
        gx2, gy2 = new.all_world2pix(gsky.ra.deg, gsky.dec.deg, 0)
        guess = np.column_stack([gx2, gy2])
        tol = XY_MATCH_TOLERANCE
    f1, s1 = fraction(new)
    shift = np.hypot(*(np.array(new.all_world2pix(centre.ra.deg, centre.dec.deg, 0))
                       - np.array([nx / 2, ny / 2])))
    print(f"[wcs] repaired ({'mirrored, ' if parity < 0 else ''}{nctl} pattern "
          f"stars, {int(ok.sum())} in the fit): the field centre moved {shift:.0f} px; "
          f"now {100 * f1:.0f}% of bright stars match Gaia (median {s1:.2f} arcsec)",
          flush=True)
    if f1 < WCS_CHECK_MIN_FRACTION:
        stop("PLATE SOLUTION: the repaired solution still does not fit. "
             "Plate-solve the G/V image again and rerun.")
    return new, (f"REPAIRED from Gaia: header solution matched {100 * f0:.0f}% of bright "
                 f"stars, repaired {100 * f1:.0f}%, centre moved {shift:.0f} px")


# ------------------- Calibration -------------------

def field_centre_and_radius(wcs, shape):
    ny, nx = shape
    centre = wcs.pixel_to_world(nx / 2, ny / 2)
    corners = wcs.pixel_to_world(np.array([0, nx, 0, nx]), np.array([0, 0, ny, ny]))
    radius = float(np.max(centre.separation(corners).deg))
    return centre, radius


def query_calibration_catalogue(which, centre, radius_deg):
    """Reference stars with Johnson B and V: RA, DEC, Mag_B, Mag_V, e_B, e_V.

    APASS DR9 (II/336/apass9) is the default. Gaia DR3 synthetic photometry
    (I/360/syntphot) is computed from Gaia's own low-resolution spectra and
    standardised to the Johnson-Kron-Cousins system: much deeper, uniform
    across the sky, and resolved down to about an arcsecond - useful where
    APASS is thin or blended. Only entries Gaia flags as good are used.

    row_limit has to be passed to the constructor: "Vizier.ROW_LIMIT = -1" set
    on the class afterwards left astroquery's default of 50 rows, silently.
    """
    coord = SkyCoord(centre.ra, centre.dec)
    if which == "APASS":
        cols = ["RAJ2000", "DEJ2000", "Bmag", "Vmag", "e_Bmag", "e_Vmag"]
        res = Vizier(columns=cols, row_limit=-1).query_region(
            coord, radius=radius_deg * u.deg, catalog="II/336/apass9")
        if len(res) == 0:
            return pd.DataFrame(columns=["RA", "DEC", "Mag_B", "Mag_V", "e_B", "e_V"])
        t = res[0].to_pandas()
        out = pd.DataFrame({"RA": t["RAJ2000"], "DEC": t["DEJ2000"],
                            "Mag_B": t["Bmag"], "Mag_V": t["Vmag"],
                            "e_B": t["e_Bmag"], "e_V": t["e_Vmag"]})
    else:
        cols = ["Source", "RA_ICRS", "DE_ICRS", "Bmag", "Vmag", "FB", "e_FB",
                "FV", "e_FV", "BFlag", "VFlag"]
        res = Vizier(columns=cols, row_limit=-1).query_region(
            coord, radius=radius_deg * u.deg, catalog="I/360/syntphot")
        if len(res) == 0:
            return pd.DataFrame(columns=["RA", "DEC", "Mag_B", "Mag_V", "e_B", "e_V"])
        t = res[0].to_pandas()
        good = (t["BFlag"] == 1) & (t["VFlag"] == 1)
        t = t[good]
        out = pd.DataFrame({"RA": t["RA_ICRS"], "DEC": t["DE_ICRS"],
                            "Mag_B": t["Bmag"], "Mag_V": t["Vmag"],
                            "e_B": 1.0857 * t["e_FB"] / t["FB"],
                            "e_V": 1.0857 * t["e_FV"] / t["FV"]})
    return out.reset_index(drop=True)


def match_catalogue(df, wcs_G, cat, shape):
    """Each catalogue star matched to the closest measured star, once.

    A match must be within CALIB_MATCH_RADIUS on the sky and XY_MATCH_TOLERANCE
    in G's pixel grid. Each measured star is used for at most one catalogue
    star (the closest), and each catalogue star for at most one measured star.
    """
    cat = cat[np.isfinite(cat["Mag_B"]) & np.isfinite(cat["Mag_V"])].reset_index(drop=True)
    if not len(cat):
        return pd.DataFrame()
    cx, cy = wcs_G.all_world2pix(cat["RA"].values, cat["DEC"].values, 0)
    ny, nx = shape
    inside = (cx > 0) & (cx < nx) & (cy > 0) & (cy < ny)
    cat = cat[inside].reset_index(drop=True)
    cx, cy = cx[inside], cy[inside]
    if not len(cat):
        return pd.DataFrame()
    dist, idx = cKDTree(df[["X", "Y"]].values).query(
        np.column_stack([cx, cy]), distance_upper_bound=XY_MATCH_TOLERANCE)
    best = {}
    for i in range(len(cat)):
        if not np.isfinite(dist[i]):
            continue
        j = int(idx[i])
        if j not in best or dist[i] < best[j][1]:
            best[j] = (i, dist[i])
    rows = []
    for j, (i, d) in best.items():
        star = df.iloc[j]
        sep = SkyCoord(star["RA"] * u.deg, star["DEC"] * u.deg).separation(
            SkyCoord(cat.loc[i, "RA"] * u.deg, cat.loc[i, "DEC"] * u.deg)).arcsec
        if sep > CALIB_MATCH_RADIUS:
            continue
        r = star.to_dict()
        r.update({"row": j, "Cat_RA": cat.loc[i, "RA"], "Cat_DEC": cat.loc[i, "DEC"],
                  "Cat_B": cat.loc[i, "Mag_B"], "Cat_V": cat.loc[i, "Mag_V"],
                  "Cat_e_B": cat.loc[i, "e_B"], "Cat_e_V": cat.loc[i, "e_V"],
                  "Sep_arcsec": sep, "Sep_px": d})
        rows.append(r)
    return pd.DataFrame(rows)


def clipped_line(x, y, nsig=CALIB_CLIP_SIGMA, maxiter=10):
    """Straight-line fit y = intercept + slope x, with points further than
    nsig robust sigmas from the line left out, repeated until nothing changes.
    Returns (intercept, slope, mask_used, rms_of_used)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    use = np.isfinite(x) & np.isfinite(y)
    for _ in range(maxiter):
        if use.sum() < 3:
            break
        slope, intercept = np.polyfit(x[use], y[use], 1)
        resid = y - (intercept + slope * x)
        sig = 1.4826 * np.median(np.abs(resid[use] - np.median(resid[use])))
        sig = max(sig, 1e-3)
        new = np.isfinite(resid) & (np.abs(resid) < nsig * sig)
        if np.array_equal(new, use):
            break
        use = new
    slope, intercept = np.polyfit(x[use], y[use], 1)
    resid = y - (intercept + slope * x)
    return intercept, slope, use, float(np.std(resid[use]))


def calibration_candidates(df, wcs_G, shape, which):
    """Catalogue stars matched to good measurements: unsaturated, isolated,
    well measured, with finite catalogue magnitudes."""
    centre, radius = field_centre_and_radius(wcs_G, shape)
    print(f"[calib] querying {'APASS DR9' if which == 'APASS' else 'Gaia DR3 synthetic photometry'} "
          f"within {radius * 60:.0f} arcmin of the field centre...", flush=True)
    cat = query_calibration_catalogue(which, centre, radius)
    print(f"   {len(cat):,} catalogue stars in the region", flush=True)
    m = match_catalogue(df, wcs_G, cat, shape)
    if not len(m):
        return m, {"catalogue": len(cat), "matched": 0}
    n_matched = len(m)
    keep = ~m["Saturated"].astype(bool)
    n_sat = int((~keep).sum())
    crowded = m["Nearest_px"] < m["Aperture_Radius"]
    keep &= ~crowded
    good_err = (m["Mag_inst_B_err"] < CALIB_MAX_INST_ERR) & (m["Mag_inst_G_err"] < CALIB_MAX_INST_ERR)
    keep &= good_err
    cat_err_ok = ~((m["Cat_e_B"] > CALIB_MAX_CAT_ERR) | (m["Cat_e_V"] > CALIB_MAX_CAT_ERR))
    keep &= cat_err_ok
    stats = {"catalogue": len(cat), "matched": n_matched, "saturated": n_sat,
             "crowded": int(crowded.sum()), "noisy": int((~good_err).sum()),
             "bad_catalogue_error": int((~cat_err_ok).sum()),
             "median_sep_arcsec": float(np.median(m["Sep_arcsec"]))}
    m["Candidate"] = keep.values
    return m, stats


def fit_calibration(cand):
    """Zero points and colour terms from the candidate stars.

        (b-v)_inst      = alpha + beta * (B-V)_cat      (the colour transformation)
            so (B-V)    = a + c * (b-v)_inst   with c = 1/beta, a = -alpha/beta
        V_cat - v_inst  = ZP_V + k_V * (B-V)_cat
        B_cat - b_inst  = ZP_B + k_B * (B-V)_cat   (shown, and a cross-check)

    WHY THE COLOUR IS FITTED THIS WAY ROUND
    A straight-line fit treats its x as exact and puts all the scatter in y.
    Fitting catalogue colour against instrumental colour therefore assumes the
    instrumental colour is exact - and on a colour camera it is far from it:
    the blue and green channels overlap, so the instrumental colour spans only
    about a third of the real one, and the debayering adds a scatter of a few
    hundredths that depends on where each star falls on the colour-filter
    grid. That scatter is small in magnitudes but large next to the tiny
    range, and the fit came out badly flattened: on M15 c = 1.9 where the
    bright, precisely measured stars alone give about 2.7, which left every
    red giant 0.2 mag too blue.

    So the instrumental colour is fitted against the catalogue colour - the
    usual form of a photometric transformation - and inverted. Checked
    against Gaia's synthetic colours, which are precise to a hundredth or
    two: on M15 this gives c = 3.0 and the Gaia colours give 2.9-3.0; on the
    split M67 channels 2.8 against 3.1; on the LCO filter frames 1.00 against
    1.03 (where every way of fitting agrees). Correcting beta for APASS's
    quoted colour errors made it worse, because those errors are overstated
    - on the LCO frames the whole B-V residual is 0.05 mag while APASS
    claims 0.08 for its colours alone - so no such correction is applied.

    Each fit is sigma-clipped; a star clipped from either the colour or the V
    fit is marked as not used.
    """
    c = cand[cand["Candidate"]].copy()
    bv_inst = c["Mag_inst_B"] - c["Mag_inst_G"]
    bv_cat = c["Cat_B"] - c["Cat_V"]
    alpha, beta, use_c, _ = clipped_line(bv_cat, bv_inst)
    cterm, a = 1.0 / beta, -alpha / beta
    rms_c = float(np.std((a + cterm * bv_inst - bv_cat)[use_c]))
    zpv, kv, use_v, rms_v = clipped_line(bv_cat, c["Cat_V"] - c["Mag_inst_G"])
    zpb, kb, use_b, rms_b = clipped_line(bv_cat, c["Cat_B"] - c["Mag_inst_B"])
    used = use_c & use_v
    corr = float(np.corrcoef(bv_inst, bv_cat)[0, 1]) if len(c) > 2 else np.nan
    return {"a": a, "c": cterm, "ZP_V": zpv, "k_V": kv, "ZP_B": zpb, "k_B": kb,
            "rms_colour_fit": rms_c, "rms_V_fit": rms_v, "rms_B_fit": rms_b,
            "N_candidates": len(c), "N_used": int(used.sum()),
            "correlation": corr, "used_index": c.index[used]}


def apply_calibration(df, cal):
    """Calibrated B, V, B-V and their errors for every star in df."""
    b, v = df["Mag_inst_B"].to_numpy(float), df["Mag_inst_G"].to_numpy(float)
    eb, ev = df["Mag_inst_B_err"].to_numpy(float), df["Mag_inst_G_err"].to_numpy(float)
    a, c, zp, k = cal["a"], cal["c"], cal["ZP_V"], cal["k_V"]
    bv = a + c * (b - v)
    V = v + zp + k * bv
    B = V + bv
    # V = (1 - k c) v + k c b + const ; B = (1 - (1+k) c) v + (1+k) c b + const
    e_bv = np.abs(c) * np.hypot(eb, ev)
    e_V = np.hypot((1 - k * c) * ev, k * c * eb)
    e_B = np.hypot((1 - (1 + k) * c) * ev, (1 + k) * c * eb)
    out = df.copy()
    out["Mag_B_cal"], out["Mag_V_cal"], out["BV_cal"] = B, V, bv
    out["Mag_B_cal_err"], out["Mag_V_cal_err"], out["BV_cal_err"] = e_B, e_V, e_bv
    return out


# ------------------- Cluster membership -------------------

def match_members(df, members, wcs_G):
    """Measured stars that are catalogue members, each once (closest)."""
    mx, my = wcs_G.all_world2pix(members["RA"].values, members["DEC"].values, 0)
    dist, idx = cKDTree(df[["X", "Y"]].values).query(
        np.column_stack([mx, my]), distance_upper_bound=XY_MATCH_TOLERANCE)
    # If two catalogue members (a crowded field can have close pairs) both
    # land within tolerance of the same measured star, only the closer of the
    # two is kept - otherwise that one star's photometry is written out twice.
    best = {}
    for i in range(len(members)):
        if not np.isfinite(dist[i]):
            continue
        j = int(idx[i])
        if j not in best or dist[i] < best[j][1]:
            best[j] = (i, dist[i])
    rows = sorted(best)
    out = df.iloc[rows].copy()
    out["Prob"] = [members["Prob"].iloc[best[j][0]] for j in rows]
    return out


def open_cluster_members(cg_name, prob_min):
    t = cantat_gaudin_members(cg_name)
    members = pd.DataFrame({"RA": np.asarray(t["RA_ICRS"], float),
                            "DEC": np.asarray(t["DE_ICRS"], float),
                            "Gmag": np.asarray(t["Gmag"], float),
                            "Prob": np.asarray(t["proba"], float)})
    n_all = len(members)
    members = members[members["Prob"] >= prob_min].reset_index(drop=True)
    print(f"   {len(members):,} of {n_all:,} catalogue members have membership "
          f"probability >= {prob_min:g}", flush=True)
    if not len(members):
        raise RuntimeError(f"No members found for open cluster {cg_name}")
    return members


def globular_pm_members(vb_row, wcs_G, shape):
    """Gaia DR3 stars in the frame moving with the cluster.

    A star is a member when its proper motion is within GC_PM_NSIGMA of the
    cluster's mean (Vasiliev & Baumgardt 2021), counting the star's own Gaia
    error, the error of the mean and the cluster's internal motions
    (GC_PM_INTRINSIC), and its parallax agrees with the cluster's within the
    same number of sigmas. Field stars at the distance of the Sun's
    neighbourhood have much larger proper motions and parallaxes than a
    globular 5-15 kpc away, so this removes most of them.
    """
    centre, radius = field_centre_and_radius(wcs_G, shape)
    cols = ["RA_ICRS", "DE_ICRS", "pmRA", "pmDE", "e_pmRA", "e_pmDE", "Plx",
            "e_Plx", "Gmag"]
    print(f"[cluster] fetching Gaia DR3 stars (G < {GC_GAIA_GMAG_LIMIT:g}) within "
          f"{radius * 60:.0f} arcmin...", flush=True)
    res = Vizier(columns=cols, column_filters={"Gmag": f"<{GC_GAIA_GMAG_LIMIT}"},
                 row_limit=-1).query_region(SkyCoord(centre.ra, centre.dec),
                                            radius=radius * u.deg,
                                            catalog="I/355/gaiadr3")
    if not len(res):
        raise RuntimeError("no Gaia DR3 stars returned")
    g = res[0].to_pandas()
    pmra0, pmde0 = float(vb_row["pmRA"]), float(vb_row["pmDE"])
    e_pmra0, e_pmde0 = float(vb_row["e_pmRA"]), float(vb_row["e_pmDE"])
    plx0, e_plx0 = float(vb_row["plx"]), float(vb_row["e_plx"])
    s_ra = np.sqrt(g["e_pmRA"] ** 2 + e_pmra0 ** 2 + GC_PM_INTRINSIC ** 2)
    s_de = np.sqrt(g["e_pmDE"] ** 2 + e_pmde0 ** 2 + GC_PM_INTRINSIC ** 2)
    chi = np.sqrt(((g["pmRA"] - pmra0) / s_ra) ** 2 + ((g["pmDE"] - pmde0) / s_de) ** 2)
    plx_ok = np.abs(g["Plx"] - plx0) < GC_PM_NSIGMA * np.sqrt(g["e_Plx"] ** 2 + e_plx0 ** 2)
    member = (chi < GC_PM_NSIGMA) & plx_ok
    print(f"   cluster mean motion ({pmra0:+.2f}, {pmde0:+.2f}) mas/yr, parallax "
          f"{plx0:.3f} mas; {int(member.sum()):,} of {len(g):,} Gaia stars "
          f"(with proper motions: {int(np.isfinite(chi).sum()):,}) move with it",
          flush=True)
    m = g[member]
    return pd.DataFrame({"RA": m["RA_ICRS"].values, "DEC": m["DE_ICRS"].values,
                         "Gmag": m["Gmag"].values,
                         "Prob": np.clip(1.0 - (chi[member].values / GC_PM_NSIGMA) ** 2, 0, 1)})


# ------------------- Plots -------------------

def write_cmd(subset, plot_path, name, distance, ebv, av, mu, show):
    """One colour-magnitude diagram, drawn and saved.

    The heading is the file's own name, so a figure copied into a document is
    still traceable to the file it came from. The cluster and the numbers used
    sit on the second line. The colour axis follows the data (1st to 99th
    percentile, padded) but always covers -0.5 to 2. The error bars at the
    left are the median errors of the stars in each 1-magnitude bin.
    """
    import matplotlib.pyplot as plt
    ycol = "Abs_Mag_V" if APPLY_DISTANCE_MODULUS else "Mag_V_corr"
    x = subset["BV_corr"].to_numpy(float)
    y = subset[ycol].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    ex = subset["BV_cal_err"].to_numpy(float)[ok]
    ey = subset["Mag_V_cal_err"].to_numpy(float)[ok]

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.scatter(x, y, s=6, color="black")
    if len(x):
        lo, hi = np.percentile(x, [1, 99])
        pad = 0.15 * (hi - lo) + 0.1
        # follow the data, but never narrower than -0.5..2 and never wider
        # than -1..3: a handful of noisy faint stars should not squeeze the
        # real sequence into a sliver
        xmin = max(min(lo - pad, -0.5), -1.0)
        xmax = min(max(hi + pad, 2.0), 3.0)
    else:
        xmin, xmax = -0.5, 2.0
    ax.set_xlim(xmin, xmax)
    # typical error bars per 1-magnitude bin, at the left edge
    if len(y):
        for m0 in np.arange(np.floor(np.nanmin(y)), np.ceil(np.nanmax(y)), 1.0):
            inbin = (y >= m0) & (y < m0 + 1)
            if inbin.sum() >= 5:
                ax.errorbar(xmin + 0.08 * (xmax - xmin), m0 + 0.5,
                            xerr=np.nanmedian(ex[inbin]), yerr=np.nanmedian(ey[inbin]),
                            color="tab:red", capsize=2, lw=1)
    ax.invert_yaxis()          # brighter = up
    ax.set_xlabel("(B−V)₀  colour, dereddened")
    if APPLY_DISTANCE_MODULUS:
        ax.set_ylabel("M_V  absolute magnitude")
        # the observed V on the right, for comparison with the literature
        sec = ax.secondary_yaxis("right", functions=(lambda M: M + mu + av,
                                                     lambda V: V - mu - av))
        sec.set_ylabel("V  apparent (as observed)")
    else:
        ax.set_ylabel("V₀  apparent magnitude, extinction removed")
    stem = os.path.splitext(os.path.basename(plot_path))[0]
    ax.set_title(f"{stem}\n"
                 f"{name}    d = {distance:,.0f} pc  (μ = {mu:.2f})    "
                 f"E(B−V) = {ebv:.3f}    A_V = {av:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def write_star_overlay(image, vmin, vmax, subset, plot_path, name, show, centre=None):
    """The frame with a circle on every source in this subset - where in the
    sky the sources in the diagram actually are."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    plt.scatter(subset["X"], subset["Y"], s=30, edgecolor="red",
                facecolor="none", lw=1)
    if centre is not None:
        plt.plot(centre[0], centre[1], "c+", ms=20, mew=2)
    stem = os.path.splitext(os.path.basename(plot_path))[0]
    plt.title(f"{stem}\n{name}", fontsize=11)
    plt.xlabel("X [pix]")
    plt.ylabel("Y [pix]")
    fig.savefig(plot_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def offered(label, value, source, unit="", override=None):
    """Show a looked-up number and let it be overridden, or ask when there is none.

    Every number that goes into the diagram is printed with where it came from
    before it is used. A distance and a reddening set the whole vertical scale of
    a colour-magnitude diagram, and a wrong one moves every star together - which
    is exactly the kind of error that looks like a result.
    """
    if override is not None:
        print(f"   {label} = {override:g}{unit}  (command line)")
        return float(override)
    if value is None:
        print(f"   {label}: not found in the catalogue")
        return float(_ask(f"   enter {label}{unit}", None, float))
    value = float(f"{value:.4g}")   # catalogue floats arrive as 0.09000000357627869
    return float(_ask(f"   {label} = {value:g}{unit}  ({source}), Enter to accept",
                      value, float))


def pixel_scale_of(path):
    """Arcseconds per pixel, from the plate solution (the square root of the
    determinant of the CD matrix, right whatever the rotation), or None."""
    try:
        head = fits.getheader(path)
        wcs = WCS(head, naxis=2)
        if wcs.has_celestial:
            area = abs(float(np.linalg.det(wcs.pixel_scale_matrix)))
            if area > 0:
                return 3600.0 * float(np.sqrt(area))
        if head.get("PIXSCALE"):
            return float(head["PIXSCALE"])
        focal, pixel_um = head.get("FOCALLEN"), head.get("XPIXSZ")
        if focal and pixel_um:
            return 206.265 * float(pixel_um) / float(focal)
    except Exception:
        pass
    return None


def stop(message):
    print("\n" + "!" * 78)
    print(message)
    print("!" * 78)
    sys.exit(1)


def loud(message):
    print("\n" + "*" * 78)
    for line in message.splitlines():
        print("*  " + line)
    print("*" * 78 + "\n", flush=True)


# ------------------- Run -------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Colour-magnitude diagram of a star cluster. "
                                "Every option is optional; anything not given is asked.")
    p.add_argument("--folder", help="folder holding the frames")
    p.add_argument("--b", help="B (or blue channel) FITS file")
    p.add_argument("--v", help="V (or green channel) FITS file")
    p.add_argument("--r", help="red channel FITS file (colour cameras; used if B and R prove swapped)")
    p.add_argument("--cluster", help="cluster name, e.g. M67")
    p.add_argument("--type", choices=["O", "G"], help="O = open, G = globular")
    p.add_argument("--distance", type=float, help="distance in pc")
    p.add_argument("--ebv", type=float, help="colour excess E(B-V)")
    p.add_argument("--membership", choices=["P", "R"],
                   help="globulars: P = Gaia proper motions, R = radius")
    p.add_argument("--radius-px", type=float, help="globulars, radius method: radius in pixels")
    p.add_argument("--prob-min", type=float, help="open clusters: minimum membership probability")
    p.add_argument("--catalogue", choices=["APASS", "GAIA"], help="calibration catalogue")
    p.add_argument("--yes", action="store_true", help="accept every default without asking")
    p.add_argument("--no-show", action="store_true", help="save figures without opening windows")
    args, _unknown = p.parse_known_args(argv)   # Spyder may add its own arguments
    return args


def main(argv=None):
    global ARGS
    ARGS = parse_args(argv)
    show = not ARGS.no_show
    if not show:
        matplotlib.use("Agg")

    summary = []

    def note(line):
        summary.append(line)

    print("=== Cluster Photometry Interactive Input ===")

    folder = ARGS.folder
    if folder is None and not (ARGS.b and ARGS.v):
        folder = _ask("Path to the folder holding the frames (Enter to name the files instead)",
                      "", str)
    folder = _norm_path(folder or "")

    found, object_name = {}, None
    if folder and os.path.isdir(folder):
        found, object_name = frames_in_folder(folder)
    elif folder:
        print(f"[input] there is no folder at {folder} - naming the files instead")
    for key, val in (("B", ARGS.b), ("V", ARGS.v), ("R", ARGS.r)):
        if val:
            path = _norm_path(val)
            band = band_of(path, fits.getheader(path))
            if key == "V" and band == "G":
                key = "G"
            found[key] = [path] + [p for p in found.get(key, []) if p != path]

    # Which kind of data is this?
    if found.get("B") and found.get("V"):
        mode = "mono"
    elif found.get("B") and found.get("G"):
        mode = "split"
    elif found.get("RGB"):
        mode = "rgb3"
    else:
        print("[input] could not find a B and a V/G frame in the folder")
        found["B"] = [_norm_path(_ask("Path to B-band (or blue channel) FITS image", None, str))]
        vpath = _norm_path(_ask("Path to V-band (or green channel) FITS image", None, str))
        vband = band_of(vpath, fits.getheader(vpath))
        if vband == "G":
            found["G"] = [vpath]
            mode = "split"
        else:
            found["V"] = [vpath]
            mode = "mono"
        if mode == "split" and not found.get("R"):
            rp = _ask("Path to the red channel (Enter if none)", "", str)
            if rp:
                found["R"] = [_norm_path(rp)]
        if not object_name:
            object_name = str(fits.getheader(found["B"][0]).get("OBJECT", "")).strip().upper() or None

    if not object_name:
        for key in ("G", "V", "B", "RGB"):
            if found.get(key):
                obj = fits.getheader(found[key][0]).get("OBJECT")
                if obj:
                    object_name = " ".join(str(obj).split()).upper()
                    break
    cluster_name = ARGS.cluster or object_name
    if cluster_name:
        print(f"   cluster: {cluster_name}   "
              f"({'command line' if ARGS.cluster else 'from the OBJECT keyword'})")
    else:
        cluster_name = _ask("Cluster name (e.g. M67, NGC 6205)", None, str)
    cluster_name = " ".join(cluster_name.split())

    # The output folder, new for every run, next to the images.
    data_dir = folder if folder and os.path.isdir(folder) else os.path.dirname(
        (found.get("G") or found.get("V") or found.get("B") or found.get("RGB"))[0])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9]+", "_", cluster_name).strip("_")
    out_dir = os.path.join(data_dir, f"result_CMD_{safe}_{stamp}")
    os.makedirs(out_dir, exist_ok=False)
    print(f"[out] writing everything to {out_dir}")
    note(f"Cluster_CMD run {stamp}")
    note(f"cluster: {cluster_name}")
    note(f"data folder: {data_dir}")

    # What the files are, and which are used.
    if mode == "mono":
        print("   B and V filter frames:")
        ch = describe_choice(found, ("B", "V"))
        file_B, file_G, file_R = ch["B"], ch["V"], None
        note("data: mono filter frames, B and V")
    elif mode == "split":
        print("   colour-camera channels, already split into files:")
        ch = describe_choice(found, ("B", "G", "R"))
        file_B, file_G, file_R = ch["B"], ch["G"], ch.get("R")
        print("   NOTE: G is the camera's GREEN channel, not Johnson V. It is "
              "calibrated onto V with a colour term (k_V below).")
        note("data: colour-camera channel files B, G (green, not V), R")
    else:
        print("   raw colour-camera frames, three planes each:")
        used = choose_colour_group(found["RGB"])
        planes, colour_headers = stack_colour_frames(used, out_dir)
        file_B, file_G, file_R = planes["B"], planes["G"], planes["R"]
        print("   NOTE: G is the camera's GREEN channel, not Johnson V. It is "
              "calibrated onto V with a colour term (k_V below).")
        note(f"data: three-plane colour frames, median of {len(used)}: "
             + ", ".join(os.path.basename(p) for p in used))
    colour_camera = mode in ("split", "rgb3")
    note(f"B file: {file_B}")
    note(f"G/V file: {file_G}")

    # Cluster type and position
    kind, centre_sky = simbad_info(cluster_name)
    aliases = simbad_aliases(cluster_name)
    cluster_type = ARGS.type or kind
    if cluster_type:
        src = "command line" if ARGS.type else "SIMBAD"
        print(f"   type: {'globular' if cluster_type == 'G' else 'open'}   ({src})")
    else:
        cluster_type = _ask("Cluster type [O=open, G=globular]", "O", str, upper=True,
                            choices=("O", "G"))

    print("Looking the cluster up...", flush=True)
    if cluster_type == "G":
        _dist, _ebv, _size = globular_from_harris(cluster_name, aliases)
        _source = "Harris catalogue"
    else:
        _dist, _ebv, _size = open_cluster_from_dias(cluster_name, aliases)
        _source = "Dias catalogue"

    Cluster_distance = offered("distance", _dist, _source, " pc", ARGS.distance)
    E_BV = offered("E(B-V)", _ebv, _source, "", ARGS.ebv)
    A_V = R_V * E_BV
    mu = 5.0 * np.log10(Cluster_distance) - 5.0
    print(f"Using distance {Cluster_distance:g} pc (distance modulus {mu:.2f}), "
          f"E(B-V) = {E_BV}, A_V = R_V x E(B-V) = {A_V:.3f}")
    note(f"distance {Cluster_distance:g} pc, mu = {mu:.3f}, E(B-V) = {E_BV}, A_V = {A_V:.3f}")

    membership, radius_px, cg_name, vb_row = None, None, None, None
    prob_min = ARGS.prob_min if ARGS.prob_min is not None else OPEN_CLUSTER_PROB_MIN
    if cluster_type == "G":
        vb_row = vasiliev_baumgardt_row(aliases)
        default_m = "P" if vb_row is not None else "R"
        if vb_row is None:
            print("   no Vasiliev & Baumgardt entry - members by radius")
        membership = ARGS.membership or _ask(
            "Membership: P = Gaia proper motions (Vasiliev & Baumgardt 2021), "
            "R = within a radius", default_m, str, upper=True, choices=("P", "R"))
        if membership == "P" and vb_row is None:
            membership = "R"
        if membership == "R":
            # What the catalogue publishes is the half-light radius, which by
            # definition holds half the cluster's light and is too small to
            # select its members. It is shown as a yardstick - a globular
            # usually reaches several times it.
            _scale = pixel_scale_of(file_G)
            default_r = None
            if _size and _scale:
                rh_px = _size * 60.0 / _scale
                default_r = round(3 * rh_px)
                print(f"   for scale: the half-light radius is {_size:g} arcmin"
                      f" = {rh_px:.0f} px in this frame;"
                      f" a globular usually reaches several times that")
            radius_px = ARGS.radius_px if ARGS.radius_px else float(
                _ask("Cluster radius in pixels", default_r, float))
            note(f"membership: within {radius_px:g} px of the SIMBAD centre")
        else:
            note("membership: Gaia DR3 proper motions vs Vasiliev & Baumgardt 2021 mean")
    else:
        try:
            cg_name = normalize_cluster_name(cluster_name, aliases)
            print(f"   Cantat-Gaudin name: {cg_name}")
        except RuntimeError as e:
            stop(str(e))
        note(f"membership: Cantat-Gaudin & Anders 2020, {cg_name}, prob >= {prob_min:g}")

    # ---------------- Photometry ----------------
    dfG, infoG, G_img = run_photometry(file_G, "G", out_dir)
    dfB, infoB, B_img = run_photometry(file_B, "B", out_dir)
    shape = G_img.shape
    wcs_G, wcs_note = verify_plate_solution(dfG, WCS(fits.getheader(file_G), naxis=2), shape)
    note(f"plate solution: {wcs_note}")

    def matched_table(dfB_, B_img_, file_B_):
        d = match_B_to_G(dfB_, dfG, B_img_, G_img, file_B_, file_G)
        ra, dec = wcs_G.all_pix2world(d["X"].values, d["Y"].values, 0)
        d.insert(2, "RA", ra)
        d.insert(3, "DEC", dec)
        return d

    df = matched_table(dfB, B_img, file_B)
    note(f"FWHM: G {infoG['fwhm']:.2f} px, B {infoB['fwhm']:.2f} px; aperture "
         f"{infoG['r_ap']:.2f} px (G), {infoB['r_ap']:.2f} px (B)")
    note(f"saturation: G {infoG['sat_level']:.0f} ({infoG['sat_src']}), "
         f"B {infoB['sat_level']:.0f} ({infoB['sat_src']})")

    # ---------------- Calibration ----------------
    def calibrate(df_):
        first = (ARGS.catalogue or CALIB_CATALOGUE).upper()
        order = [first] + [c for c in ("APASS", "GAIA") if c != first]
        for which in order:
            cand, stats = calibration_candidates(df_, wcs_G, shape, which)
            n = int(cand["Candidate"].sum()) if len(cand) else 0
            print(f"   {stats.get('matched', 0)} matched to measured stars "
                  f"(median separation {stats.get('median_sep_arcsec', np.nan):.2f} arcsec); "
                  f"left out: {stats.get('saturated', 0)} saturated, "
                  f"{stats.get('crowded', 0)} with a neighbour in the aperture, "
                  f"{stats.get('noisy', 0)} too noisy, "
                  f"{stats.get('bad_catalogue_error', 0)} with large catalogue errors; "
                  f"{n} candidates", flush=True)
            if n >= CALIB_MIN_STARS:
                cal = fit_calibration(cand)
                if cal["N_used"] >= CALIB_MIN_STARS:
                    cal["catalogue"] = which
                    return cand, cal
            print(f"   too few calibration stars from {which} "
                  f"(need {CALIB_MIN_STARS})", flush=True)
        stop("CALIBRATION FAILED: fewer than "
             f"{CALIB_MIN_STARS} usable reference stars in APASS and in Gaia.\n"
             "Is the image plate-solved (WCS in the header)? Is the field mostly "
             "saturated?\nThe run stops here rather than draw a diagram on a "
             "calibration it does not have.")

    cand, cal = calibrate(df)

    # ---------------- Colour sanity check ----------------
    swapped = False
    print(f"[check] correlation of instrumental b-v with catalogue B-V over "
          f"{cal['N_candidates']} calibration stars: r = {cal['correlation']:+.2f}",
          flush=True)
    if not (cal["correlation"] > 0):
        if colour_camera:
            heads = colour_headers if mode == "rgb3" else [fits.getheader(file_B)]
            notes = flip_notes(heads)
            why = ("\n".join("   " + n for n in notes) if notes
                   else "   (the header says nothing about a flip)")
            if not file_R:
                stop("COLOUR CHECK FAILED: blue and red channels look swapped "
                     "(instrumental colour falls as the real colour rises), and "
                     "there is no red channel to swap in.")
            loud("COLOUR CHECK: B AND R ARE SWAPPED.\n"
                 "Instrumental b-v FALLS as catalogue B-V rises - the channel "
                 "labelled B is really red.\n"
                 "This happens when the image is flipped/mirrored before the "
                 "colours are separated, so the\nBayer pattern is read the wrong "
                 "way round. The header says:\n" + why + "\n"
                 f"The channel labelled R ({os.path.basename(file_R)}) is now "
                 "measured as BLUE.")
            note("COLOUR CHECK: B and R swapped (correlation "
                 f"{cal['correlation']:+.2f}); header notes: " + "; ".join(notes))
            dfB, infoB, B_img = run_photometry(file_R, "B", out_dir)
            file_B = file_R
            note(f"B file after the swap: {file_B}; FWHM {infoB['fwhm']:.2f} px, "
                 f"aperture {infoB['r_ap']:.2f} px, saturation "
                 f"{infoB['sat_level']:.0f} ({infoB['sat_src']})")
            df = matched_table(dfB, B_img, file_B)
            cand, cal = calibrate(df)
            print(f"[check] after the swap: r = {cal['correlation']:+.2f}", flush=True)
            if not (cal["correlation"] > 0):
                stop("COLOUR CHECK FAILED even after swapping B and R. The colours "
                     "cannot be trusted; check the files.")
            swapped = True
        else:
            stop("COLOUR CHECK FAILED: instrumental b-v FALLS as catalogue B-V "
                 f"rises (r = {cal['correlation']:+.2f}).\nFor filter frames this "
                 "means the files are almost certainly mislabelled - the frame "
                 "called B is not blue,\nor the one called V is. Check the FILTER "
                 "keywords of:\n   " + file_B + "\n   " + file_G)
    elif cal["correlation"] < 0.5:
        print("   WARNING: the correlation is weak - the colours are noisy", flush=True)
    if not swapped:
        note(f"colour check: OK, correlation {cal['correlation']:+.2f}, no swap")

    # Save the calibration table with residuals
    calibrated_cand = apply_calibration(cand, cal)
    calibrated_cand["Cat_BV"] = calibrated_cand["Cat_B"] - calibrated_cand["Cat_V"]
    calibrated_cand["Resid_V"] = calibrated_cand["Mag_V_cal"] - calibrated_cand["Cat_V"]
    calibrated_cand["Resid_B"] = calibrated_cand["Mag_B_cal"] - calibrated_cand["Cat_B"]
    calibrated_cand["Resid_BV"] = calibrated_cand["BV_cal"] - calibrated_cand["Cat_BV"]
    calibrated_cand["Used"] = calibrated_cand.index.isin(cal["used_index"])
    calib_csv = os.path.join(out_dir, f"calib_stars_{cal['catalogue'].lower()}.csv")
    calibrated_cand.drop(columns=["row"], errors="ignore").to_csv(calib_csv, index=False)
    u_ = calibrated_cand[calibrated_cand["Used"]]
    rms_V, rms_B, rms_BV = (float(np.std(u_[c])) for c in ("Resid_V", "Resid_B", "Resid_BV"))
    print(f"[calib] {cal['catalogue']}: {cal['N_used']} of {cal['N_candidates']} "
          f"candidate stars used after {CALIB_CLIP_SIGMA:g}-sigma clipping")
    print(f"   (B-V) = {cal['a']:+.3f} + {cal['c']:.3f} x (b-v)_inst")
    print(f"   V     = v_inst {cal['ZP_V']:+.3f} {cal['k_V']:+.3f} x (B-V)")
    print(f"   B     = b_inst {cal['ZP_B']:+.3f} {cal['k_B']:+.3f} x (B-V)   "
          f"(blue channel colour term 1/c = {1 / cal['c']:.3f})")
    print(f"   rms scatter of the used stars: V {rms_V:.3f}, B {rms_B:.3f}, "
          f"B-V {rms_BV:.3f} mag")
    print(f"[calib] wrote {calib_csv}")
    note(f"calibration: {cal['catalogue']}, {cal['N_used']} of {cal['N_candidates']} stars used")
    note(f"  (B-V) = {cal['a']:+.4f} + {cal['c']:.4f} (b-v)_inst")
    note(f"  V = v_inst {cal['ZP_V']:+.4f} {cal['k_V']:+.4f} (B-V)")
    note(f"  B = b_inst {cal['ZP_B']:+.4f} {cal['k_B']:+.4f} (B-V)")
    note(f"  rms: V {rms_V:.4f}, B {rms_B:.4f}, B-V {rms_BV:.4f}")
    note(f"  median residuals: V {np.median(u_['Resid_V']):+.4f}, "
         f"B-V {np.median(u_['Resid_BV']):+.4f}")

    flux_csv = os.path.join(out_dir, "fluxes_XY_FWHM_Ap.csv")
    df.to_csv(flux_csv, index=False)
    print(f"[out] wrote {flux_csv}")

    # ---------------- Calibrated and corrected tables ----------------
    df_cal = apply_calibration(df, cal)
    df_cal["N_calib_used"] = cal["N_used"]
    out_cal_csv = os.path.join(out_dir, "fluxes_calibrated.csv")
    df_cal.to_csv(out_cal_csv, index=False)
    print(f"[calib] wrote {out_cal_csv} ({len(df_cal):,} stars)")

    # extinction-corrected values (apparent), then the distance
    df_cal["Mag_V_corr"] = df_cal["Mag_V_cal"] - A_V
    df_cal["Mag_B_corr"] = df_cal["Mag_B_cal"] - (A_V + E_BV)
    df_cal["BV_corr"] = df_cal["Mag_B_corr"] - df_cal["Mag_V_corr"]
    df_cal["Abs_Mag_V"] = df_cal["Mag_V_corr"] - mu
    df_cal["A_V_used"] = A_V
    df_cal["E_BV_used"] = E_BV
    df_cal["DistMod_used"] = mu
    out_corr_csv = os.path.join(out_dir, "fluxes_calibrated_galactic.csv")
    df_cal.to_csv(out_corr_csv, index=False)
    print(f"[galactic] wrote {out_corr_csv}")

    # ---------------- Members ----------------
    centre_px = None
    if centre_sky is not None:
        cx, cy = wcs_G.world_to_pixel(centre_sky)
        centre_px = (float(cx), float(cy))
    if cluster_type == "G":
        if centre_px is None:
            centre_px = (shape[1] / 2, shape[0] / 2)
            print("   SIMBAD gave no position - the frame centre is used as the "
                  "cluster centre")
        print(f"   cluster centre (SIMBAD through the plate solution): "
              f"x = {centre_px[0]:.0f}, y = {centre_px[1]:.0f}")
        note(f"cluster centre px: {centre_px[0]:.1f}, {centre_px[1]:.1f}")
        if membership == "P":
            members = globular_pm_members(vb_row, wcs_G, shape)
            members.to_csv(os.path.join(out_dir, "fluxes_cluster_only.csv"), index=False)
            cluster = match_members(df_cal, members, wcs_G)
        else:
            r = np.hypot(df_cal["X"] - centre_px[0], df_cal["Y"] - centre_px[1])
            cluster = df_cal[r <= radius_px].copy()
            cluster["Prob"] = 1.0
            cluster[["X", "Y", "RA", "DEC", "Prob"]].to_csv(
                os.path.join(out_dir, "fluxes_cluster_only.csv"), index=False)
        cluster["Dist_from_centre_px"] = np.hypot(cluster["X"] - centre_px[0],
                                                  cluster["Y"] - centre_px[1])
    else:
        members = open_cluster_members(cg_name, prob_min)
        mx, my = wcs_G.all_world2pix(members["RA"].values, members["DEC"].values, 0)
        members["X"], members["Y"] = mx, my
        members.to_csv(os.path.join(out_dir, "fluxes_cluster_only.csv"), index=False)
        cluster = match_members(df_cal, members, wcs_G)
    print(f"[cluster] {len(cluster):,} measured stars are members of {cluster_name}")
    cal_cols = [c for c in df_cal.columns if not c.endswith(("_corr", "_used"))
                and c != "Abs_Mag_V"] + ["Prob"]
    cluster[[c for c in cal_cols if c in cluster.columns]].to_csv(
        os.path.join(out_dir, "fluxes_cluster_only_calibrated.csv"), index=False)
    cluster.to_csv(os.path.join(out_dir, "fluxes_cluster_only_galactic.csv"), index=False)
    note(f"members measured: {len(cluster)}")

    # ---------------- Diagrams and maps ----------------
    from astropy.visualization import ZScaleInterval
    vmin, vmax = ZScaleInterval().get_limits(G_img)
    for scope, table in (("whole_frame_sources", df_cal), ("cluster_sources", cluster)):
        unsat = table[~table["Saturated"].astype(bool)]
        n_sat = len(table) - len(unsat)
        unsat = unsat[unsat["BV_cal_err"] <= CMD_MAX_BV_ERR]
        print(f"[cmd] {scope}: {len(table):,} stars; left out of the diagram: "
              f"{n_sat:,} saturated, {len(table) - n_sat - len(unsat):,} with a "
              f"B-V error above {CMD_MAX_BV_ERR:g}")
        keep_mask = crowded_or_misshapen(unsat)
        versions = [("all", unsat)]
        if keep_mask is not None and keep_mask.any() and not keep_mask.all():
            versions.append(("no_overlap", unsat.loc[keep_mask]))
            clean_csv = os.path.join(out_dir, f"{scope}_no_overlap_{int(keep_mask.sum())}.csv")
            unsat.loc[keep_mask].to_csv(clean_csv, index=False)
        for kind_, subset in versions:
            plot_path = os.path.join(out_dir, f"CMD_{scope}_{kind_}_{len(subset)}.png")
            write_cmd(subset, plot_path, cluster_name, Cluster_distance, E_BV,
                      A_V, mu, show)
            print(f"[cmd] saved {plot_path}")
            note(f"plot {os.path.basename(plot_path)}: {len(subset)} stars")
            if scope == "cluster_sources":
                map_path = os.path.join(out_dir, f"map_cluster_sources_{kind_}_{len(subset)}.png")
                write_star_overlay(G_img, vmin, vmax, subset, map_path,
                                   cluster_name, show,
                                   centre_px if cluster_type == "G" else None)
                print(f"[plot] wrote {map_path}")

    with open(os.path.join(out_dir, "run_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary) + "\n")
    print(f"\nDone. Everything is in {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
