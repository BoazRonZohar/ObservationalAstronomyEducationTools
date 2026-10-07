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
absolute magnitude, and - as far as the stars allow - the cluster's age and
distance from theoretical isochrones.

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
  metallicity [M/H]               for the age and distance only, and only when
                                  the isochrone table is beside the script.
                                  Offered from Harris for a globular ([Fe/H]
                                  plus 0.21, for the extra oxygen and the like
                                  in a globular's stars - see
                                  ALPHA_ENHANCEMENT_MH), from Dias for an open
                                  cluster, or solar when Dias has none
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
  8. fits PARSEC isochrones to the members, for the age and the distance -
     see fit_age_and_distance(). An age only when the members reach the
     turnoff, the one part of the diagram that changes with age; otherwise
     a distance only - from the main sequence at the catalogue age for an
     open cluster, from the horizontal branch for a globular - with the
     reason. The isochrones are the file isochrones_PARSEC_UBV.csv.gz beside
     this script (PARSEC: Bressan et al. 2012; the other references are in
     isochrones_PARSEC_UBV_README.txt). Without it this step is left out and
     everything else runs

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

  Age and distance (when the isochrone table is beside this script)

    CMD_isochrone_fit_<N>.png       the members with the best-fitting isochrone
                                    and those at the 16th and 84th percentiles
                                    of the age; V0 on the left, M_V at the
                                    fitted distance on the right. <N> is the
                                    number of stars in the fit
    isochrone_fit_map.png           how well every age and distance tried
                                    fits, and where the resamplings landed
    isochrone_fit_stars.csv         the members, whether each was in the fit,
                                    and how far it lies from the isochrone
    isochrone_best.csv              the isochrone drawn, in absolute and in
                                    observed magnitudes
    isochrone_fit_grid.csv          the fit statistic for every age and
                                    distance tried

    The age and distance with their ranges, the distance from the Gaia
    parallaxes and the catalogue values are printed at the end of the run.

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

# ------------------- Age and distance config -------------------
# The age and distance come from sliding theoretical isochrones (the Padova
# PARSEC models, from the table that sits beside this script) over the
# members' diagram - see fit_age_and_distance(). Set False to skip that step.
FIT_ISOCHRONES = True
ISOCHRONE_FILE = "isochrones_PARSEC_UBV.csv.gz"   # looked for next to this script
FIT_MAX_BV_ERR = 0.10     # stars in the fit: B-V error at most this (mag)
FIT_MAX_V_ERR = 0.05      # and V error at most this
FIT_FAINTEST_MV = 8.0     # and no fainter than this M_V at the catalogue distance:
                          # the models' colours are least certain for the coolest dwarfs
FIT_MIN_STARS = 30        # fewer usable members than this: no fit
FIT_COLOUR_FLOOR = 0.02   # mag, added in quadrature to every star's B-V error
FIT_MAG_FLOOR = 0.02      # mag, likewise for V: calibration and model errors
FIT_D2_CAP = 9.0          # a star further than 3 sigma from the isochrone counts as 3 sigma
FIT_MU_RANGE = 1.0        # distance moduli tried: the catalogue's value +- this
FIT_BOOTSTRAP = 200       # resamplings of the stars, for the uncertainties
FIT_SEED = 1              # the resamplings are random, but the same on every run
FIT_EBV_STEP = 0.02       # the fit is repeated with E(B-V) +- this, to show its effect
FIT_CURVE_STEP = 0.01     # mag between points filled in along an isochrone
FIT_NEIGHBOURS = 16       # isochrone points looked at for each star
FIT_FLASH_GAP = 0.3       # mag; see _joined()
TURNOFF_MIN_STARS = 10    # members within 0.5 mag of the turnoff needed for an age
GC_MIN_LOGAGE = 9.7       # globulars: only ages from 5 Gyr up are tried
GC_DEFAULT_LOGAGE = 10.08 # 12 Gyr: drawn for a globular whose age cannot be fitted

# PARSEC's stars have the Sun's mix of elements. A globular cluster's stars
# have about twice as much oxygen, magnesium and the like for each iron atom
# ([alpha/Fe] = +0.3), and a scaled-solar isochrone mimics that with a higher
# overall metallicity: [M/H] = [Fe/H] + 0.21 (Salaris, Chieffi & Straniero 1993).
ALPHA_ENHANCEMENT_MH = 0.21

# The horizontal branch as a standard candle, for a globular whose turnoff is
# too faint to reach: M_V(HB) = 0.16 [Fe/H] + 0.84 (Harris 1996, 2010 edition).
HB_MV_SLOPE = 0.16
HB_MV_ZERO = 0.84
HB_MIN_STARS = 5
HB_WINDOW = 1.0           # mag either side of where the catalogue puts the branch
HB_HALF_WIDTH = 0.5       # mag either side of the branch itself, once it is found
HB_MAX_BV_ERR = 0.25      # looser than for the fit: the branch is blue, where a
HB_MAX_V_ERR = 0.10       # colour camera's blue channel is weakest

# Open clusters' Gaia DR3 parallaxes: they come out too small by 0.017 mas on
# average (Lindegren et al. 2021), and errors common to stars a few arcminutes
# apart do not average away; 0.011 mas is the size Vasiliev & Baumgardt (2021)
# found for them. (A globular's mean parallax is taken from Vasiliev &
# Baumgardt as published, with both already in it - see globular_parallax().)
GAIA_PLX_ZERO_POINT = -0.017   # mas
GAIA_PLX_SYSTEMATIC = 0.011    # mas
GAIA_RUWE_MAX = 1.4            # Gaia's own limit for a well-behaved single-star solution


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


def _value(table, row, *names):
    """A number from a catalogue row, from the first column whose name matches
    one of names when both are squashed ('[Fe/H]' matches 'FeH'), or None if
    there is no such column or it is empty."""
    wanted = {_squash(n) for n in names}
    for col in table.colnames:
        if _squash(col) in wanted:
            try:
                v = row[col]
                if np.ma.is_masked(v):
                    return None
                v = float(v)
                return v if np.isfinite(v) else None
            except (TypeError, ValueError):
                return None
    return None


def globular_from_harris(name, aliases):
    """Distance in pc, E(B-V), half-light radius (arcmin) and a dict of other
    values ("feh": [Fe/H]) for a globular.

    Harris (1996, 2010 edition) is the standard compilation for the Milky Way's
    globulars: 147 of them, a distance for 145 and a reddening for every one.
    The row used is the one whose name matches the cluster - a search around a
    position can also return a neighbour.
    """
    try:
        tables = Vizier(columns=["**"]).query_object(name, catalog="VII/202")
        if not tables:
            return None, None, None, {}
        distance = ebv = half_light = feh = None
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
            if feh is None:
                feh = _value(table, row, "[Fe/H]")
        if not found:
            print(f"[lookup] the Harris catalogue has no row named {name}",
                  flush=True)
        return distance, ebv, half_light, ({"feh": feh} if feh is not None else {})
    except Exception as exc:
        print(f"[lookup] Harris catalogue unavailable ({exc})", flush=True)
        return None, None, None, {}


def open_cluster_from_dias(name, aliases):
    """Distance in pc, E(B-V), diameter and a dict of other values ("feh":
    [Fe/H], "log_age": log10 of the age in years, where the catalogue has
    them) for an open cluster.

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
            return None, None, None, {}
        table = tables[0]
        row = _matching_row(table, ("Cluster",), aliases)
        if row is None:
            print(f"[lookup] the Dias catalogue has no row named {name}",
                  flush=True)
            return None, None, None, {}

        def _num(col):
            try:
                v = float(row[col]) if col in table.colnames else None
                return v if v is not None and np.isfinite(v) else None
            except Exception:
                return None
        extra = {"feh": _value(table, row, "[Fe/H]"), "log_age": _value(table, row, "Age")}
        return (_num("Dist"), _num("E(B-V)"), _num("Diam"),
                {k: v for k, v in extra.items() if v is not None})
    except Exception as exc:
        print(f"[lookup] Dias catalogue unavailable ({exc})", flush=True)
        return None, None, None, {}


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


def cantat_gaudin_cluster(cg_name):
    """Age, distance and extinction of an open cluster from Cantat-Gaudin et
    al. 2020 (J/A+A/640/A1, table 1 - the catalogue the members come from),
    who estimated them for every cluster from its Gaia photometry and
    parallaxes: a dict with "log_age", "distance" (pc) and "A_V", empty if
    the catalogue cannot be reached."""
    try:
        res = Vizier(columns=["Cluster", "AgeNN", "AVNN", "DMNN", "DistPc"],
                     column_filters={"Cluster": f"={cg_name}"},
                     row_limit=-1).query_constraints(catalog="J/A+A/640/A1/table1")
        if not len(res):
            return {}
        t = res[0]
        rows = [r for r in t if str(r["Cluster"]) == cg_name]
        if not rows:
            return {}
        row = rows[0]
        out = {"log_age": _value(t, row, "AgeNN"), "A_V": _value(t, row, "AVNN"),
               "distance": _value(t, row, "DistPc")}
        if out["distance"] is None:
            dm = _value(t, row, "DMNN")
            out["distance"] = None if dm is None else 10.0 ** (dm / 5.0 + 1.0)
        return {k: v for k, v in out.items() if v is not None}
    except Exception as exc:
        print(f"[lookup] Cantat-Gaudin cluster parameters unavailable ({exc})",
              flush=True)
        return {}


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


# ------------------- Age and distance from isochrones -------------------
#
# An isochrone is the line that stars of one age and one composition make in
# the diagram, from the faintest dwarfs to the most evolved giants: what the
# models say a cluster of that age should look like. The distance is how far
# down the magnitude axis it has to be slid to lie on the members, and the age
# is the isochrone that lies on them best.

def _script_folder():
    """The folder this script is in. The isochrone table is looked for here,
    not in the folder the run was started from, so it is found however the
    script is started."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:          # pasted into an interactive session
        return os.getcwd()


def load_isochrones():
    """The PARSEC isochrone table that sits beside this script, or None.

    Without it only the age and distance step is left out, with a message
    saying which file is missing and where it should be; everything else runs.
    """
    path = os.path.join(_script_folder(), ISOCHRONE_FILE)
    if not os.path.isfile(path):
        loud("AGE AND DISTANCE LEFT OUT: the isochrone table is not here:\n"
             f"   {path}\n"
             f"{ISOCHRONE_FILE} comes with this script, in the Cluster_CMD folder of\n"
             "the repository. Put it next to Cluster_CMD.py to get the age and the\n"
             "distance; the rest of the run goes ahead without it.")
        return None
    try:
        iso = pd.read_csv(path, comment="#")
    except Exception as exc:
        loud(f"AGE AND DISTANCE LEFT OUT: {path} could not be read ({exc})")
        return None
    missing = [c for c in ("MH", "logAge", "label", "MV", "BV") if c not in iso.columns]
    if missing:
        loud(f"AGE AND DISTANCE LEFT OUT: {path} has no column {', '.join(missing)}")
        return None
    print(f"[age] isochrones: {os.path.basename(path)}, {iso['MH'].nunique()} "
          f"metallicities x {iso['logAge'].nunique()} ages", flush=True)
    return iso


def isochrones_at(iso, mh):
    """The isochrones of one metallicity, {logAge: (BV, MV, label)} in the
    table's own order (up in initial mass), for the grid metallicity nearest
    to mh, and that metallicity. Metallicities are not interpolated: the
    grid steps by 0.2, so the one used is at most 0.1 from the one asked."""
    grid = np.unique(iso["MH"].to_numpy(float))
    mh_used = float(grid[np.argmin(np.abs(grid - mh))])
    if mh < grid[0] - 0.1 or mh > grid[-1] + 0.1:
        print(f"   [M/H] = {mh:+.2f} is outside the models ({grid[0]:+.2f} to "
              f"{grid[-1]:+.2f}); the nearest, {mh_used:+.2f}, is used", flush=True)
    sub = iso[np.isclose(iso["MH"].to_numpy(float), mh_used)]
    curves = {}
    for age, g in sub.groupby("logAge", sort=True):
        curves[round(float(age), 3)] = (g["BV"].to_numpy(float), g["MV"].to_numpy(float),
                                        g["label"].to_numpy(int))
    return curves, mh_used


def _joined(bv, mv, label):
    """For each pair of neighbouring points of an isochrone, whether the line
    runs on between them.

    The points go up in initial mass, and neighbours are joined by the stars
    of the masses in between - except where the evolution jumps. A star of
    about the Sun's mass leaves the top of the red giant branch (PARSEC's
    stage 3) in the helium flash and reappears on the horizontal branch
    (stage 4), far away in the diagram, with nothing in between. So
    neighbours are joined when they are in the same stage or the next one,
    and the step from 3 to 4 only when it is shorter than FIT_FLASH_GAP.
    Stages 0 and 1 count as one: PARSEC switches between them back and forth
    along the main sequence.
    """
    if len(bv) < 2:
        return np.zeros(0, bool)
    step = np.diff(label)
    gap = np.hypot(np.diff(bv), np.diff(mv))
    flash = (label[:-1] == 3) & (label[1:] == 4) & (gap > FIT_FLASH_GAP)
    main_sequence = (label[:-1] <= 1) & (label[1:] <= 1)
    return (((step == 0) | (step == 1)) & ~flash) | main_sequence


def _segments(bv, mv, label):
    """The isochrone cut where it does not run on, as arrays of indices - so
    it is drawn without lines across the jumps."""
    return np.split(np.arange(len(bv)), np.where(~_joined(bv, mv, label))[0] + 1)


def _curve_points(bv, mv, label):
    """The isochrone's points, filled in to no more than FIT_CURVE_STEP apart
    wherever the line runs on, so that the distance from a star to the
    nearest point is the distance to the line."""
    join = _joined(bv, mv, label)
    if not join.any():
        return bv.copy(), mv.copy()
    d = np.hypot(np.diff(bv), np.diff(mv))
    n = np.where(join, np.ceil(d / FIT_CURVE_STEP), 1).astype(int)
    extra = np.maximum(n - 1, 0)
    seg = np.repeat(np.arange(len(n)), extra)
    k = np.arange(len(seg)) - np.repeat(np.cumsum(extra) - extra, extra) + 1
    t = k / n[seg]
    x = bv[seg] + t * (bv[seg + 1] - bv[seg])
    y = mv[seg] + t * (mv[seg + 1] - mv[seg])
    return np.concatenate([bv, x]), np.concatenate([mv, y])


def _fit_arrays(stars, ebv_shift=0.0):
    """Dereddened colour and magnitude of the stars, and their errors with
    the floors added - with E(B-V) ebv_shift higher than the value used."""
    c = stars["BV_corr"].to_numpy(float) - ebv_shift
    m = stars["Mag_V_corr"].to_numpy(float) - R_V * ebv_shift
    sc = np.hypot(stars["BV_cal_err"].to_numpy(float), FIT_COLOUR_FLOOR)
    sm = np.hypot(stars["Mag_V_cal_err"].to_numpy(float), FIT_MAG_FLOOR)
    return c, m, sc, sm


def _distances2(points, c, m, sc, sm, mus):
    """How far every star is from an isochrone slid to each distance modulus
    in mus: the squared distance to its nearest point, counted in the star's
    own errors and capped at FIT_D2_CAP. An array (len(mus), number of stars).

    The isochrone's points nearest each star are found in a tree, in units of
    the typical errors; the distance is then measured, in the star's own
    errors, to each of the FIT_NEIGHBOURS nearest, and the smallest kept.
    """
    px, py = points
    gc, gm = float(np.median(sc)), float(np.median(sm))
    tree = cKDTree(np.column_stack([px / gc, py / gm]))
    k = min(FIT_NEIGHBOURS, len(px))
    mus = np.atleast_1d(np.asarray(mus, float))
    n = len(c)
    out = np.empty((len(mus), n))
    per = max(1, 2_000_000 // max(n * k, 1))     # distance moduli at a time, to bound memory
    for i0 in range(0, len(mus), per):
        mu = mus[i0:i0 + per]
        # the stars slid up instead of the isochrone down: V0 - mu against M_V
        qx = np.tile(c, len(mu))
        qy = (m[None, :] - mu[:, None]).ravel()
        q = np.column_stack([qx / gc, qy / gm])
        try:
            _, j = tree.query(q, k=k, workers=-1)
        except TypeError:                          # scipy before 1.6
            _, j = tree.query(q, k=k)
        j = np.asarray(j).reshape(len(qx), k)
        dx = (px[j] - qx[:, None]) / np.tile(sc, len(mu))[:, None]
        dy = (py[j] - qy[:, None]) / np.tile(sm, len(mu))[:, None]
        out[i0:i0 + len(mu)] = np.min(dx * dx + dy * dy, axis=1).reshape(len(mu), n)
    return np.minimum(out, FIT_D2_CAP)


def _grid(points, ages, c, m, sc, sm, mus, counts=None):
    """S[age, mu], the capped squared distances summed over the stars, for
    every age and distance modulus; with resampling counts also
    Sb[age, mu, resampling], the same sum with each star counted as often
    as it was drawn."""
    S = np.empty((len(ages), len(mus)))
    Sb = None if counts is None else np.empty((len(ages), len(mus), len(counts)))
    for i, a in enumerate(ages):
        D2 = _distances2(points[a], c, m, sc, sm, mus)
        S[i] = D2.sum(axis=1)
        if counts is not None:
            Sb[i] = D2 @ counts.T
    return S, Sb


def grid_fit(points, ages, c, m, sc, sm, mu_cat, counts=None):
    """The age and distance modulus with the smallest S: every age given,
    distance moduli within FIT_MU_RANGE of the catalogue's - first in steps
    of 0.05 mag, then of 0.01 mag around what that first pass found, for the
    stars as measured and for each resampling of them."""
    ages = np.asarray(ages, float)
    coarse = mu_cat + np.arange(-FIT_MU_RANGE, FIT_MU_RANGE + 1e-9, 0.05)
    S, Sb = _grid(points, ages, c, m, sc, sm, coarse, counts)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    lo_i = hi_i = i
    lo_mu = hi_mu = coarse[j]
    if Sb is not None:
        bi, bj = np.unravel_index(Sb.reshape(-1, Sb.shape[2]).argmin(axis=0), S.shape)
        lo_i, hi_i = min(lo_i, bi.min()), max(hi_i, bi.max())
        lo_mu, hi_mu = min(lo_mu, coarse[bj].min()), max(hi_mu, coarse[bj].max())
    sel = np.arange(max(lo_i - 2, 0), min(hi_i + 3, len(ages)))
    fine = np.arange(max(lo_mu - 0.1, coarse[0]), min(hi_mu + 0.1, coarse[-1]) + 1e-9, 0.01)
    Sf, Sbf = _grid(points, ages[sel], c, m, sc, sm, fine, counts)
    i, j = np.unravel_index(np.argmin(Sf), Sf.shape)
    out = {"log_age": float(ages[sel][i]), "mu": float(fine[j]), "S_min": float(Sf[i, j]),
           "age_edge": len(ages) > 1 and ages[sel][i] in (ages[0], ages[-1]),
           "mu_edge": abs(fine[j] - mu_cat) > FIT_MU_RANGE - 0.03,
           "map": (ages, coarse, S)}
    if Sbf is not None:
        bi, bj = np.unravel_index(Sbf.reshape(-1, Sbf.shape[2]).argmin(axis=0), Sf.shape)
        out["boot_log_age"] = ages[sel][bi]
        out["boot_mu"] = fine[bj]
    return out


def turnoff(bv, mv, label):
    """The turnoff of an isochrone, (B-V, M_V): its bluest main-sequence point,
    where the stars that are running out of hydrogen in their cores leave
    the main sequence. None if it has no main sequence."""
    ms = np.where(label == 1)[0]
    if not len(ms):
        return None
    i = ms[np.argmin(bv[ms])]
    return float(bv[i]), float(mv[i])


def turnoff_coverage(v0, v0_to):
    """Whether these stars can give an age, for a turnoff at V0 = v0_to:
    'good', 'weak' or 'none', and a dict of the numbers with, for 'weak'
    and 'none', why.

    The diagram changes with age only around the turnoff, so an age needs
    stars there: TURNOFF_MIN_STARS within half a magnitude of it. It is
    'good' when there are also stars on the main sequence below it and
    stars past it (subgiants and giants), and 'weak' when one side is
    nearly empty.
    """
    info = {"v0_to": float(v0_to), "why": "",
            "near": int(np.sum(np.abs(v0 - v0_to) <= 0.5)),
            "below": int(np.sum((v0 > v0_to + 0.5) & (v0 <= v0_to + 2.0))),
            "past": int(np.sum(v0 < v0_to - 0.5))}
    if info["near"] < TURNOFF_MIN_STARS:
        if v0.max() < v0_to - 0.5:
            info["why"] = "fainter than every star measured"
        elif v0.min() > v0_to + 0.5:
            info["why"] = "brighter than every star measured - its stars are probably saturated"
        else:
            info["why"] = f"reached by only {info['near']} stars, fewer than {TURNOFF_MIN_STARS}"
        return "none", info
    if info["below"] < 5 or info["past"] < 3:
        info["why"] = f"few stars {'below' if info['below'] < 5 else 'past'} it"
        return "weak", info
    return "good", info


def _turnoff_text(info, full=True):
    text = f"V0 = {info['v0_to']:.2f}, {info['near']} stars within 0.5 mag of it"
    if full:
        text += f", {info['below']} on the main sequence below it, {info['past']} brighter"
    return text


def fit_selection(cluster, mu_cat):
    """The members, renumbered; which of them go into the fit: unsaturated,
    clear of neighbours (crowded_or_misshapen), measured to FIT_MAX_BV_ERR in
    B-V and FIT_MAX_V_ERR in V, and no fainter than FIT_FAINTEST_MV at the
    catalogue distance; which are measured well enough to look for the
    horizontal branch among (the same, but to HB_MAX_BV_ERR and HB_MAX_V_ERR);
    and a line saying how many were left out of the fit, and why."""
    t = cluster.reset_index(drop=True)
    finite = (np.isfinite(t["BV_corr"].to_numpy(float))
              & np.isfinite(t["Mag_V_corr"].to_numpy(float)))
    unsat = ~t["Saturated"].astype(bool).to_numpy() & finite
    keep = crowded_or_misshapen(t, quiet=True)
    clean = unsat & (np.ones(len(t), bool) if keep is None else np.asarray(keep, bool))
    precise = (clean & (t["BV_cal_err"].to_numpy(float) <= FIT_MAX_BV_ERR)
               & (t["Mag_V_cal_err"].to_numpy(float) <= FIT_MAX_V_ERR))
    use = precise & (t["Mag_V_corr"].to_numpy(float) - mu_cat <= FIT_FAINTEST_MV)
    branch = (clean & (t["BV_cal_err"].to_numpy(float) <= HB_MAX_BV_ERR)
              & (t["Mag_V_cal_err"].to_numpy(float) <= HB_MAX_V_ERR))
    why = (f"{len(t)} members; left out: {int((~unsat).sum())} saturated or unmeasured, "
           f"{int((unsat & ~clean).sum())} crowded or misshapen, "
           f"{int((clean & ~precise).sum())} with a B-V error above {FIT_MAX_BV_ERR:g} "
           f"or a V error above {FIT_MAX_V_ERR:g}, {int((precise & ~use).sum())} fainter "
           f"than M_V = {FIT_FAINTEST_MV:g}; {int(use.sum())} in the fit")
    return t, use, branch, why


def horizontal_branch_distance(t, sample, feh, mu_cat, rgb_bv):
    """Distance modulus from the horizontal branch, for a globular whose
    turnoff is out of reach, or None.

    Horizontal-branch stars burn helium in their cores, and all have about
    the same brightness in V where the branch is level - from the RR Lyrae
    gap redward - at M_V(HB) = HB_MV_SLOPE [Fe/H] + HB_MV_ZERO. This is the
    standard candle the Harris catalogue's own distances come from, so this
    is a check of these measurements against it rather than an independent
    distance.

    The stars looked at are the members in sample that are bluer than the
    giant branch (by 0.15 mag, at the branch's brightness) and within
    HB_WINDOW of where the catalogue distance puts the branch. That cut
    still lets in a few giants and asymptotic-branch stars, so the branch is
    found as the median V0, taken again over the stars within HB_HALF_WIDTH
    of it until it stops moving. Blueward of B-V0 = 0 the branch turns down
    - hotter stars are fainter in V - so V0(HB) is then the same median over
    the stars at B-V0 >= 0, if there are HB_MIN_STARS of them; otherwise
    over all of them, with a warning. The error given is the error of the
    median.
    """
    c = t["BV_corr"].to_numpy(float)
    v0 = t["Mag_V_corr"].to_numpy(float)
    mv_hb = HB_MV_SLOPE * feh + HB_MV_ZERO
    expected = mu_cat + mv_hb
    red_edge = rgb_bv - 0.15 if rgb_bv is not None and np.isfinite(rgb_bv) else 0.5
    cand = sample & (c > -0.4) & (c < red_edge) & (np.abs(v0 - expected) < HB_WINDOW)

    def settle(mask, v=None):
        """The median V0 of the stars in mask within HB_HALF_WIDTH of it."""
        if mask.sum() < HB_MIN_STARS:
            return None, mask
        v = float(np.median(v0[mask])) if v is None else v
        for _ in range(20):
            near = mask & (np.abs(v0 - v) < HB_HALF_WIDTH)
            if near.sum() < HB_MIN_STARS:
                return None, near
            v_new = float(np.median(v0[near]))
            if abs(v_new - v) < 1e-4:
                break
            v = v_new
        return v_new, near

    v_all, branch = settle(cand)
    if v_all is None:
        return None
    v_hb, chosen = settle(cand & (c >= 0.0), v_all)
    caveat = ""
    if v_hb is None:
        v_hb, chosen = v_all, branch
        caveat = ("too few stars at B-V0 >= 0: the bluer stars used are fainter in V "
                  "than the level part of the branch, so this distance comes out too large")
    vals = v0[chosen]
    err = 1.2533 * float(np.std(vals, ddof=1)) / np.sqrt(len(vals))
    return {"mu": v_hb - mv_hb, "err": err, "v0_hb": v_hb, "mv_hb": mv_hb,
            "n": int(chosen.sum()), "mask": chosen, "caveat": caveat}


def _distance_pc(mu):
    return 10.0 ** (np.asarray(mu, float) / 5.0 + 1.0)


def gaia_parallax_distance(members, wcs_G, shape):
    """The cluster's parallax from Gaia DR3, from its catalogue members in the
    frame: those with a Gaia star within 1 arcsec and a RUWE below
    GAIA_RUWE_MAX, averaged with weights 1/error^2 and 3-sigma clipping, and
    corrected for Gaia's zero point. A dict (plx, err in mas; n), or None."""
    centre, radius = field_centre_and_radius(wcs_G, shape)
    try:
        res = Vizier(columns=["RA_ICRS", "DE_ICRS", "Plx", "e_Plx", "RUWE", "Gmag"],
                     column_filters={"Gmag": "<18"}, row_limit=-1).query_region(
            SkyCoord(centre.ra, centre.dec), radius=radius * u.deg, catalog="I/355/gaiadr3")
        g = res[0].to_pandas()
        gsky = SkyCoord(g["RA_ICRS"].to_numpy(float) * u.deg, g["DE_ICRS"].to_numpy(float) * u.deg)
        msky = SkyCoord(members["RA"].to_numpy(float) * u.deg,
                        members["DEC"].to_numpy(float) * u.deg)
        j, sep, _ = msky.match_to_catalog_sky(gsky)
        p = g.iloc[j[sep.arcsec < 1.0]]
        p = p[np.isfinite(p["Plx"]) & (p["e_Plx"] > 0) & (p["RUWE"] < GAIA_RUWE_MAX)]
    except Exception as exc:
        print(f"[age] Gaia parallaxes unavailable ({exc})", flush=True)
        return None
    x, e = p["Plx"].to_numpy(float), p["e_Plx"].to_numpy(float)
    use = np.ones(len(x), bool)
    for _ in range(10):
        if use.sum() < 5:
            return None
        w = 1.0 / e[use] ** 2
        mean = float(np.sum(w * x[use]) / np.sum(w))
        new = np.abs(x - mean) < 3.0 * e
        if np.array_equal(new, use):
            break
        use = new
    stat = float(1.0 / np.sqrt(np.sum(1.0 / e[use] ** 2)))
    return {"plx": mean - GAIA_PLX_ZERO_POINT, "err": float(np.hypot(stat, GAIA_PLX_SYSTEMATIC)),
            "n": int(use.sum()),
            "source": f"Gaia DR3 parallaxes of {int(use.sum())} members"}


def globular_parallax(vb_row):
    """The cluster's mean Gaia EDR3 parallax from Vasiliev & Baumgardt 2021,
    as a dict like gaia_parallax_distance's, or None.

    Used as published: they corrected each star's parallax for the zero point
    (Lindegren et al. 2021) before averaging, and their uncertainty already
    includes the errors common to neighbouring stars. They also found those
    corrected parallaxes about 0.01 mas too large on average; that is not
    applied here, but for a distant globular it is a tenth of the parallax."""
    try:
        plx, err = float(vb_row["plx"]), float(vb_row["e_plx"])
    except Exception:
        return None
    if not (np.isfinite(plx) and np.isfinite(err)) or plx <= 0:
        return None
    return {"plx": plx, "err": err, "n": None,
            "source": "Gaia EDR3 mean parallax, Vasiliev & Baumgardt 2021"}


def _plx_distance(gaia):
    """Distance and its 1-sigma range, in pc, from a parallax dict."""
    plx, err = gaia["plx"], gaia["err"]
    hi = 1000.0 / (plx - err) if plx > err else np.inf
    return 1000.0 / plx, 1000.0 / (plx + err), hi


def _age_text(log_age):
    gyr = 10.0 ** (log_age - 9.0)
    return f"{1000 * gyr:.0f} Myr" if gyr < 1 else f"{gyr:.1f} Gyr"


def fit_age_and_distance(cluster, cluster_type, iso, mh, ebv, mu_cat, lit, gaia,
                         out_dir, name, show, note):
    """The cluster's age and distance from the PARSEC isochrones, as far as
    these stars allow. Writes its figures and tables into out_dir and its
    numbers into the run summary, and returns them (or None).

    WHAT IS COMPARED
    Each isochrone, for the cluster's metallicity, is slid down the magnitude
    axis by a trial distance modulus mu and laid over the members in the
    dereddened diagram, (B-V)0 against V0. For every star the distance to the
    nearest point of the isochrone is counted in the star's own errors, with
    a floor (FIT_COLOUR_FLOOR, FIT_MAG_FLOOR) for the calibration and the
    models, squared, and capped at FIT_D2_CAP. The cap is what makes this
    work on a real cluster: a field star, an unresolved binary or a blue
    straggler then costs the same whichever isochrone is tried, and cannot
    drag the fit far towards itself. The sum over the stars, S, is smallest
    for the age and distance that fit best. Unresolved binaries lying close
    above the main sequence still pull it a little: on simulated clusters
    with 20-30 percent binaries the distance came out 1-3 percent short.

    WHEN THERE IS AN AGE
    The diagram changes with age only around the turnoff, where the stars
    that are running out of hydrogen leave the main sequence. So an age is
    given only when the members reach the turnoff - see turnoff_coverage().
    Otherwise, saying why:
      an open cluster gets a distance only, from its main sequence, fitted
      at the catalogue age (Cantat-Gaudin et al. 2020) - below the turnoff
      the main sequence hardly depends on the age
      a globular gets a distance only, from its horizontal branch (see
      horizontal_branch_distance()), with a 12 Gyr isochrone drawn for
      comparison, not fitted
    Globulars are tried only from GC_MIN_LOGAGE up, and only on the main
    sequence, subgiant and giant branches: the colour of their horizontal
    branch depends on how much mass each star lost, which the models fix.

    THE UNCERTAINTIES
    The fit is repeated on FIT_BOOTSTRAP resamplings of the stars - the same
    number of stars, drawn with repetition - and the 16th to 84th percentiles
    of the ages and distances found are the ranges given: the scatter from
    which stars happen to have been measured. E(B-V) is not fitted but held
    at the value given; the fit is repeated with it FIT_EBV_STEP higher and
    lower to show how much that matters. The metallicity is the model grid's
    nearest to the one given.
    """
    import matplotlib.pyplot as plt
    print("\n[age] age and distance from the isochrones", flush=True)
    t, use, branch_ok, why = fit_selection(cluster, mu_cat)
    print("   " + why, flush=True)
    note("age and distance - stars: " + why)
    n = int(use.sum())
    if n < FIT_MIN_STARS:
        msg = (f"only {n} members are measured well enough for the fit (it needs "
               f"{FIT_MIN_STARS}) - no age or distance")
        loud("AGE AND DISTANCE: " + msg)
        note("age and distance: " + msg)
        return None

    curves, mh_used = isochrones_at(iso, mh)
    labels = (0, 1, 2, 3) if cluster_type == "G" else tuple(range(8))

    def part(a):
        bv, mv, lab = curves[a]
        k = np.isin(lab, labels)
        return bv[k], mv[k], lab[k]

    points = {a: _curve_points(*part(a)) for a in curves if len(part(a)[0]) >= 2}
    all_ages = np.array(sorted(points))
    ages = all_ages[all_ages >= GC_MIN_LOGAGE - 1e-6] if cluster_type == "G" else all_ages

    def nearest_age(a):
        return float(all_ages[np.argmin(np.abs(all_ages - a))])

    stars = t[use]
    c, m, sc, sm = _fit_arrays(stars)
    rng = np.random.default_rng(FIT_SEED)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=FIT_BOOTSTRAP)
    res = {"mh": mh, "mh_used": mh_used, "ebv": ebv, "n": n, "method": None,
           "age_status": "none", "draw": [], "hb": None, "warnings": []}
    print(f"   PARSEC isochrones at [M/H] = {mh_used:+.2f} (asked {mh:+.2f}), "
          f"E(B-V) = {ebv:g} held fixed; {n} stars, {FIT_BOOTSTRAP} resamplings",
          flush=True)

    # Can these stars give an age at all? A globular's turnoff is so faint
    # that it is checked first, at 12 Gyr and the catalogue distance.
    try_age, why_not, age_notes = True, "", []
    if cluster_type == "G":
        to = turnoff(*curves[nearest_age(GC_DEFAULT_LOGAGE)])
        status, info = turnoff_coverage(m, mu_cat + to[1])
        if status == "none":
            try_age, why_not = False, "the turnoff is " + info["why"]
            age_notes.append("at 12 Gyr and the catalogue distance: turnoff at "
                             + _turnoff_text(info, full=False))

    if try_age:
        t0 = time.time()
        fit = grid_fit(points, ages, c, m, sc, sm, mu_cat, counts)
        print(f"   {len(ages)} ages x {int(round(2 * FIT_MU_RANGE / 0.05)) + 1} distances, "
              f"then a finer grid: {time.time() - t0:.0f} s", flush=True)
        to = turnoff(*curves[fit["log_age"]])
        status, info = turnoff_coverage(m, fit["mu"] + to[1])
        if status != "none":
            res.update(method="age and distance", age_status=status, fit=fit,
                       age_why="at " + _turnoff_text(info)
                       + (f" - {info['why']}, so the age is less certain"
                          if status == "weak" else ""),
                       log_age=fit["log_age"], mu=fit["mu"], turnoff=to,
                       log_age_range=tuple(np.percentile(fit["boot_log_age"], [16, 84])),
                       mu_range=tuple(np.percentile(fit["boot_mu"], [16, 84])))
            if fit["age_edge"]:
                res["warnings"].append("the best age is at the edge of the ages tried")
        else:
            # The best isochrone puts its turnoff where there are no stars:
            # with no turnoff to hold it, the fit can slide to any age whose
            # main sequence passes through the stars - typically a young one,
            # whose main sequence also runs up through the blue stragglers.
            why_not = "these stars do not pin it down"
            age_notes.append(f"best free fit: {_age_text(fit['log_age'])}, turnoff at "
                             + _turnoff_text(info, full=False))

    if res["method"] is None and cluster_type != "G":
        a_lit, a_src = lit.get("log_age"), lit.get("log_age_source", "")
        if a_lit is None:
            a_fix, a_src = fit["log_age"], "the best free fit"
        else:
            a_fix = nearest_age(a_lit)
            if abs(a_fix - a_lit) > 0.005:
                a_src = f"the model age nearest the {_age_text(a_lit)} of {a_src}"
        fit = grid_fit(points, [a_fix], c, m, sc, sm, mu_cat, counts)
        _, info = turnoff_coverage(m, fit["mu"] + turnoff(*curves[a_fix])[1])
        age_notes.append(f"at {_age_text(a_fix)}: turnoff at " + _turnoff_text(info, full=False))
        res.update(method="distance at a fixed age", age_why=why_not, fit=fit,
                   log_age=a_fix, age_source=a_src + ", age held fixed", mu=fit["mu"],
                   mu_range=tuple(np.percentile(fit["boot_mu"], [16, 84])))
    res["age_notes"] = age_notes

    if cluster_type == "G":
        # the giant branch's colour at the horizontal branch's brightness, from
        # the 12 Gyr isochrone, sets the red edge of the horizontal branch
        bv12, mv12, lab12 = curves[nearest_age(GC_DEFAULT_LOGAGE)]
        feh = mh - ALPHA_ENHANCEMENT_MH
        mv_hb = HB_MV_SLOPE * feh + HB_MV_ZERO
        rgb_bv = None
        r = np.where(lab12 == 3)[0]
        if len(r) > 1:
            r = r[:np.argmin(mv12[r]) + 1]          # up to the tip of the branch
            o = np.argsort(mv12[r])
            rgb_bv = float(np.interp(mv_hb, mv12[r][o], bv12[r][o]))
        hb = horizontal_branch_distance(t, branch_ok, feh, mu_cat, rgb_bv)
        res["hb"] = hb
        if res["method"] is None:
            if hb is None:
                msg = (f"no age - {why_not} ({'; '.join(age_notes)}). No distance either: "
                       f"fewer than {HB_MIN_STARS} horizontal-branch stars were found")
                loud("AGE AND DISTANCE: " + msg)
                note("age and distance: " + msg)
                return None
            res.update(method="distance from the horizontal branch", age_why=why_not,
                       log_age=nearest_age(GC_DEFAULT_LOGAGE), mu=hb["mu"],
                       mu_range=(hb["mu"] - hb["err"], hb["mu"] + hb["err"]),
                       age_source="for comparison only, not fitted")

    # E(B-V) a little higher and lower
    shifts = {}
    for sgn in (+1, -1):
        dE = sgn * FIT_EBV_STEP
        if res["method"] == "distance from the horizontal branch":
            shifts[dE] = (None, res["mu"] - R_V * dE)
            continue
        cs, ms_, scs, sms = _fit_arrays(stars, dE)
        a_try = ages if res["method"] == "age and distance" else [res["log_age"]]
        f2 = grid_fit(points, a_try, cs, ms_, scs, sms, mu_cat)
        shifts[dE] = (f2["log_age"] if res["method"] == "age and distance" else None, f2["mu"])
    res["ebv_shifts"] = shifts
    if res.get("fit") is not None and res["fit"]["mu_edge"]:
        res["warnings"].append(f"the best distance is at the edge of those tried (catalogue "
                               f"distance modulus +- {FIT_MU_RANGE:g})")

    # ---- the numbers, on the screen and in the run summary ----
    res["distance"] = float(_distance_pc(res["mu"]))
    res["distance_range"] = tuple(float(x) for x in _distance_pc(res["mu_range"]))
    lines = [f"AGE AND DISTANCE ({res['method']})",
             f"PARSEC isochrones, [M/H] = {mh_used:+.2f} (asked {mh:+.2f}); "
             f"E(B-V) = {ebv:g} held fixed; {n} stars in the fit"]
    d, (d1, d2) = res["distance"], res["distance_range"]
    if res["method"] == "age and distance":
        a1, a2 = res["log_age_range"]
        lines.append(f"age       {_age_text(res['log_age'])}   ({_age_text(a1)} - {_age_text(a2)})"
                     f"   log t = {res['log_age']:.2f}"
                     + ("   WEAK - see the turnoff below" if res["age_status"] == "weak" else ""))
    else:
        lines.append(f"age       not fitted - {res['age_why']}")
        lines += ["          " + a for a in res["age_notes"]]
        lines.append(f"          isochrone drawn at {_age_text(res['log_age'])} "
                     f"({res.get('age_source', '')})")
    lines.append(f"distance  {d:,.0f} pc   ({d1:,.0f} - {d2:,.0f})   mu = {res['mu']:.2f}")
    if res["method"] == "age and distance":
        lines.append(f"          (ranges: 16th-84th percentiles of {FIT_BOOTSTRAP} resamplings)")
        lines.append("turnoff   " + res["age_why"])
    elif res["method"] == "distance from the horizontal branch":
        hb = res["hb"]
        lines.append(f"          {hb['n']} horizontal-branch stars, V0 = {hb['v0_hb']:.2f}, "
                     f"M_V(HB) = {hb['mv_hb']:.2f}; error of the median only")
        if hb["caveat"]:
            lines.append("          WARNING: " + hb["caveat"])
    for dE, (a_s, mu_s) in sorted(res["ebv_shifts"].items(), reverse=True):
        lines.append(f"with E(B-V) {ebv + dE:.3f}: "
                     + (f"{_age_text(a_s)}, " if a_s is not None else "")
                     + f"{_distance_pc(mu_s):,.0f} pc")
    if cluster_type == "G" and res["method"] == "age and distance" and res["hb"] is not None:
        hb = res["hb"]
        lines.append(f"horizontal branch, as a check: {_distance_pc(hb['mu']):,.0f} pc "
                     f"({hb['n']} stars, V0 = {hb['v0_hb']:.2f})")
    if gaia is not None:
        gd, g1, g2 = _plx_distance(gaia)
        lines.append(f"Gaia      {gd:,.0f} pc   ({g1:,.0f} - {g2:,.0f})   parallax "
                     f"{gaia['plx']:.3f} +- {gaia['err']:.3f} mas, {gaia['source']}")
    lines += lit.get("lines", [])
    for w in res["warnings"]:
        lines.append("WARNING: " + w)
    loud("\n".join(lines))
    for line in lines:
        note("  " + line)

    # ---- tables ----
    a_draw, mu_draw = res["log_age"], res["mu"]
    bvd, mvd, labd = part(a_draw)
    shown = np.isfinite(t["BV_corr"].to_numpy(float)) & np.isfinite(t["Mag_V_corr"].to_numpy(float))
    ca, ma, sca, sma = _fit_arrays(t)
    dist2 = np.full(len(t), np.nan)
    ok = shown & np.isfinite(sca) & np.isfinite(sma)
    if ok.any():
        dist2[ok] = _distances2(_curve_points(bvd, mvd, labd), ca[ok], ma[ok], sca[ok],
                                sma[ok], [mu_draw])[0]
    table = t.copy()
    table["Fit_used"] = use
    table["Fit_d2_capped"] = dist2
    if res["hb"] is not None:
        table["HB_star"] = res["hb"]["mask"]
    table.to_csv(os.path.join(out_dir, "isochrone_fit_stars.csv"), index=False)
    best = iso[np.isclose(iso["MH"].to_numpy(float), mh_used)
               & np.isclose(iso["logAge"].to_numpy(float), a_draw)].copy()
    best["V0"] = best["MV"] + mu_draw
    best["V_observed"] = best["V0"] + R_V * ebv
    best["BV_observed"] = best["BV"] + ebv
    best.to_csv(os.path.join(out_dir, "isochrone_best.csv"), index=False)
    if res.get("fit") is not None:
        g_ages, g_mus, g_S = res["fit"]["map"]
        A, MU = np.meshgrid(g_ages, g_mus, indexing="ij")
        pd.DataFrame({"logAge": A.ravel(), "mu": MU.ravel(),
                      "distance_pc": _distance_pc(MU.ravel()), "S": g_S.ravel()}).to_csv(
            os.path.join(out_dir, "isochrone_fit_grid.csv"), index=False)

    # ---- the diagram ----
    c_all, v_all = t["BV_corr"].to_numpy(float), t["Mag_V_corr"].to_numpy(float)
    drawn = shown & ~t["Saturated"].astype(bool).to_numpy() & (
        t["BV_cal_err"].to_numpy(float) <= CMD_MAX_BV_ERR)
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.scatter(c_all[drawn & ~use], v_all[drawn & ~use], s=5, color="0.7",
               label="members not in the fit")
    ax.scatter(c_all[use], v_all[use], s=7, color="black", label=f"in the fit ({n})")
    if res["hb"] is not None:
        hb = res["hb"]
        ax.scatter(c_all[hb["mask"]], v_all[hb["mask"]], s=30, facecolor="none",
                   edgecolor="tab:blue", lw=1, label=f"horizontal branch ({hb['n']})")
        ax.axhline(hb["v0_hb"], color="tab:blue", ls=":", lw=1)

    def draw(a, mu, last_stage=7, **kw):
        bv, mv, lab = part(a)
        keep = lab <= last_stage
        bv, mv, lab = bv[keep], mv[keep], lab[keep]
        for k, seg in enumerate(_segments(bv, mv, lab)):
            ax.plot(bv[seg], mv[seg] + mu, **(kw if k == 0 else {**kw, "label": None}))

    if res["method"] == "age and distance":
        f = res["fit"]
        for q, colour in ((0, "tab:blue"), (1, "tab:orange")):
            a_q = nearest_age(res["log_age_range"][q])
            if a_q != res["log_age"]:
                # each drawn at the distance that suits it best
                S_a = _grid(points, [a_q], c, m, sc, sm,
                            mu_cat + np.arange(-FIT_MU_RANGE, FIT_MU_RANGE + 1e-9, 0.01))[0][0]
                mu_q = mu_cat - FIT_MU_RANGE + 0.01 * int(np.argmin(S_a))
                # only up to the subgiant branch: the part that changes with age
                draw(a_q, mu_q, last_stage=2, color=colour, ls="--", lw=1,
                     label=f"{_age_text(a_q)} ({'16th' if q == 0 else '84th'} percentile)")
        draw(res["log_age"], res["mu"], color="tab:red", lw=1.6,
             label=f"best: {_age_text(res['log_age'])}, {res['distance']:,.0f} pc")
        ax.plot(res["turnoff"][0], res["turnoff"][1] + res["mu"], marker="*", ms=14,
                color="gold", mec="black", ls="none", label="turnoff")
    elif res["method"] == "distance at a fixed age":
        draw(res["log_age"], res["mu"], color="tab:red", lw=1.6,
             label=f"{_age_text(res['log_age'])} (age fixed), {res['distance']:,.0f} pc")
    else:
        draw(res["log_age"], res["mu"], color="tab:red", lw=1.2, ls="--",
             label=f"{_age_text(res['log_age'])}, not fitted, at {res['distance']:,.0f} pc")
    if drawn.any():
        lo, hi = np.percentile(c_all[drawn], [1, 99])
        pad = 0.15 * (hi - lo) + 0.1
        ax.set_xlim(max(min(lo - pad, -0.5), -1.0), min(max(hi + pad, 2.0), 3.0))
        ax.set_ylim(np.nanmax(v_all[drawn]) + 0.3, np.nanmin(v_all[drawn]) - 0.5)
    else:
        ax.invert_yaxis()
    ax.set_xlabel("(B−V)₀  colour, dereddened")
    ax.set_ylabel("V₀  apparent magnitude, extinction removed")
    sec = ax.secondary_yaxis("right", functions=(lambda V, mu_=res["mu"]: V - mu_,
                                                 lambda M, mu_=res["mu"]: M + mu_))
    sec.set_ylabel(f"M_V  at the fitted distance (μ = {res['mu']:.2f})")
    head = (f"{_age_text(res['log_age'])}, " if res["method"] == "age and distance" else "")
    plot_path = os.path.join(out_dir, f"CMD_isochrone_fit_{n}.png")
    stem = os.path.splitext(os.path.basename(plot_path))[0]
    ax.set_title(f"{stem}\n{name}    {head}{res['distance']:,.0f} pc    "
                 f"[M/H] = {mh_used:+.2f}    E(B−V) = {ebv:.3f}", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"[age] saved {plot_path}")
    note(f"plot {os.path.basename(plot_path)}")

    # ---- how well every age and distance fits ----
    if res.get("fit") is not None:
        g_ages, g_mus, g_S = res["fit"]["map"]
        f = res["fit"]
        fig, ax = plt.subplots(figsize=(7, 5.5))
        if len(g_ages) > 1:
            dS = g_S - g_S.min()
            pcm = ax.pcolormesh(_distance_pc(g_mus), 10.0 ** (g_ages - 9.0), dS,
                                shading="nearest", cmap="viridis_r", vmin=0,
                                vmax=max(float(np.percentile(dS, 50)), 1.0))
            fig.colorbar(pcm, ax=ax, label="S − S_min   (0 = the best fit)")
            ax.set_yscale("log")
            pairs, k = np.unique(np.column_stack([f["boot_log_age"], f["boot_mu"]]),
                                 axis=0, return_counts=True)
            ax.scatter(_distance_pc(pairs[:, 1]), 10.0 ** (pairs[:, 0] - 9.0), s=6 * k,
                       facecolor="none", edgecolor="black", lw=0.6,
                       label="best fit of each resampling")
            ax.plot(res["distance"], 10.0 ** (res["log_age"] - 9.0), marker="*", ms=14,
                    color="red", mec="black", ls="none", label="best fit")
            ax.set_ylabel("age [Gyr]")
        else:
            ax.plot(_distance_pc(g_mus), g_S[0] - g_S[0].min(), color="black")
            ax.axvspan(*res["distance_range"], color="0.85",
                       label="16th-84th percentile of the resamplings")
            ax.axvline(res["distance"], color="tab:red", label="best fit")
            ax.set_ylabel("S − S_min   (0 = the best fit)")
        if gaia is not None:
            ax.axvline(_plx_distance(gaia)[0], color="tab:green", ls="--", lw=1,
                       label="Gaia parallax")
        ax.set_xlabel("distance [pc]")
        map_path = os.path.join(out_dir, "isochrone_fit_map.png")
        ax.set_title(f"isochrone_fit_map\n{name}    [M/H] = {mh_used:+.2f}    "
                     f"E(B−V) = {ebv:.3f}", fontsize=10)
        # under the axes, so it covers nothing
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8,
                  frameon=False)
        fig.tight_layout()
        fig.savefig(map_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        print(f"[age] saved {map_path}")
    return res


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
    p.add_argument("--mh", type=float, help="metallicity [M/H] for the isochrones")
    p.add_argument("--no-fit", action="store_true", help="leave out the age and distance fit")
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
        _dist, _ebv, _size, _extra = globular_from_harris(cluster_name, aliases)
        _source = "Harris catalogue"
    else:
        _dist, _ebv, _size, _extra = open_cluster_from_dias(cluster_name, aliases)
        _source = "Dias catalogue"

    Cluster_distance = offered("distance", _dist, _source, " pc", ARGS.distance)
    E_BV = offered("E(B-V)", _ebv, _source, "", ARGS.ebv)
    A_V = R_V * E_BV
    mu = 5.0 * np.log10(Cluster_distance) - 5.0
    print(f"Using distance {Cluster_distance:g} pc (distance modulus {mu:.2f}), "
          f"E(B-V) = {E_BV}, A_V = R_V x E(B-V) = {A_V:.3f}")
    note(f"distance {Cluster_distance:g} pc, mu = {mu:.3f}, E(B-V) = {E_BV}, A_V = {A_V:.3f}")

    # The metallicity, for the age and distance fit - asked only when the
    # isochrone table is there to use it.
    iso, MH = None, None
    if FIT_ISOCHRONES and not ARGS.no_fit:
        iso = load_isochrones()
    if iso is not None:
        feh = _extra.get("feh")
        if cluster_type == "G":
            mh_value = None if feh is None else feh + ALPHA_ENHANCEMENT_MH
            mh_source = ("" if feh is None else
                         f"Harris catalogue [Fe/H] = {feh:+.2f}, plus {ALPHA_ENHANCEMENT_MH:g} "
                         "for a globular's extra alpha elements")
        elif feh is not None:
            mh_value, mh_source = feh, "Dias catalogue [Fe/H]"
        else:
            mh_value, mh_source = 0.0, "solar - the Dias catalogue gives none"
        if mh_value is None and ARGS.mh is None and _accept_defaults():
            print("   metallicity [M/H]: not in the catalogue - the age and distance fit "
                  "is left out (give --mh to include it)")
            note("age and distance: left out - no metallicity")
            iso = None
        else:
            MH = offered("metallicity [M/H]", mh_value, mh_source, "", ARGS.mh)
            as_offered = (ARGS.mh is None and mh_value is not None
                          and abs(MH - float(f"{mh_value:.4g}")) < 1e-9)
            note(f"metallicity for the isochrones: [M/H] = {MH:g} "
                 f"({mh_source if as_offered else 'given'})")

    membership, radius_px, cg_name, vb_row = None, None, None, None
    cg_params = {}
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
        if iso is not None:
            cg_params = cantat_gaudin_cluster(cg_name)

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

    # ---------------- Age and distance ----------------
    if iso is not None:
        lit = {"lines": []}
        if cluster_type != "G":
            if cg_params.get("log_age") is not None:
                lit["log_age"], lit["log_age_source"] = (cg_params["log_age"],
                                                         "Cantat-Gaudin et al. 2020")
            elif _extra.get("log_age") is not None:
                lit["log_age"], lit["log_age_source"] = _extra["log_age"], "Dias catalogue"
            bits = ([_age_text(cg_params["log_age"])] if "log_age" in cg_params else []) + \
                   ([f"{cg_params['distance']:,.0f} pc"] if "distance" in cg_params else [])
            if bits:
                lit["lines"].append("catalogue Cantat-Gaudin et al. 2020: " + ", ".join(bits))
            bits = ([_age_text(_extra["log_age"])] if "log_age" in _extra else []) + \
                   ([f"{_dist:,.0f} pc"] if _dist is not None else [])
            if bits:
                lit["lines"].append("catalogue Dias: " + ", ".join(bits))
        elif _dist is not None:
            lit["lines"].append(f"catalogue Harris: {_dist:,.0f} pc")
        lit["lines"].append(f"the other diagrams use {Cluster_distance:,.0f} pc, as given above")
        try:
            if cluster_type == "G":
                gaia = globular_parallax(vb_row) if vb_row is not None else None
            else:
                gaia = gaia_parallax_distance(members, wcs_G, shape)
            fit_age_and_distance(cluster, cluster_type, iso, MH, E_BV, mu, lit, gaia,
                                 out_dir, cluster_name, show, note)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            loud(f"AGE AND DISTANCE FAILED ({exc}).\nEverything before it is saved.")
            note(f"age and distance: failed ({exc})")

    with open(os.path.join(out_dir, "run_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary) + "\n")
    print(f"\nDone. Everything is in {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
