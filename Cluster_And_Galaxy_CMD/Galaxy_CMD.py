# -*- coding: utf-8 -*-
"""
Created by: Dr. Boaz Ron Zohar
https://github.com/BoazRonZohar/ObservationalAstronomyEducationTools/blob/main/Cluster_And_Galaxy_CMD/Galaxy_CMD.py
Affiliation: Kinneret Observatory
Member of the LCO Global Sky Partners programme
Date: September 2025

Written for student projects on spiral galaxies.

WHAT IT DOES

Takes one B-band and one V-band image of a galaxy and produces its
colour-magnitude diagram, together with the radial density profile of the
blue knots - the young star-forming regions in the arms.

The colour of a source says how hot and how young it is. Plotting colour
against brightness for every source in the galaxy separates the blue knots
from the older population, and the radial profile then shows how they are
distributed with distance from the centre - which is what a spiral arm
looks like in numbers.

WHERE THE METHOD COMES FROM

Identifying star-forming regions by their colour in ordinary broad-band
images, and then treating their distribution as a measurable property of the
galaxy, follows

    Brosch, N. (1992). Star formation systematics from colour images.
    Astrophysics and Space Science, 188, 289-298.
    doi:10.1007/BF00644916

The appeal for teaching is that it asks nothing exotic. Two broad-band frames
of a galaxy, B and V, are within reach of a school-accessible telescope, and
the young regions separate out on colour alone - no spectroscopy, no narrow-
band filter. A student with one night of data can ask a real question about
where a galaxy is forming stars, and answer it with a number.

This script is the pipeline for that, and has been used across many student
projects.

WHAT YOU NEED

  two FITS images of the same galaxy, one in B and one in V, plate-solved
  (the reference stars are pulled from a catalogue by sky position)

WHAT IT ASKS YOU

  galaxy name                    used to name the output files, and to look
                                 the next two up
  distance in Mpc                converts pixels to parsecs in the profile.
                                 Offered from Cosmicflows-3 (Tully+ 2016)
  colour excess E(B-V)           corrects brightness and colour for the dust
                                 of our own galaxy. Offered from the Schlafly
                                 & Finkbeiner (2011) dust maps via IRSA.
                                 A_V is not asked for: it is 3.1 x E(B-V)
  the two FITS paths
  how many reference stars to calibrate against

Every one of these has a default, and the two catalogue values are fetched by
name. Press Enter to accept them or type your own.

HOW IT WORKS

  1. subtracts the background from each image and finds sources in both
  2. measures each source with an aperture sized from its own FWHM
  3. matches the B and V detections to each other by position
  4. pulls reference stars from APASS9 through Vizier, matches them to the
     detections, and calibrates the instrumental magnitudes against them
  5. removes the foreground stars, leaving the galaxy's own sources
  6. plots the colour-magnitude diagram, and the radial density profile of
     what is left, in pixels and again in parsecs

WHAT YOU GET

Around twenty files, all named after the galaxy. They go into a sub-folder
called result_CMD, created next to your images - the folder holding your data
comes out of a run exactly as it went in, with the frames only read.

That is a lot of files, so here is what each one is and which you actually
want to look at. Below, NAME stands for the galaxy name you typed.

  A by-product, kept because it is what everything is measured on:

    <input>_bgsub.fits            each input image with its background
                                  subtracted. Useful if you want to check a
                                  source by eye.

  Stage 1 - everything that was measured

    NAME_photometry_results.csv   every source detected in BOTH images and
                                  matched between them: position, FWHM,
                                  aperture radius, flux in B and in V.
                                  Instrumental values, not yet calibrated.
    NAME_reference_stars.csv      the APASS9 stars found in the field, with
                                  their catalogue B and V. These set the zero
                                  point. If the calibration looks wrong, look
                                  here first - are there enough of them, and
                                  are they spread across the frame?

  Stage 2 - calibrated, still with foreground stars in it

    NAME_calibrated_photometry.csv            real magnitudes
    NAME_calibrated_photometry_with_color.csv the same, with B-V added
    NAME_calibrated_photometry_CMD.png        colour-magnitude diagram
    NAME_V_sources.png                        the V image with every measured
                                              source circled
    NAME_B_sources.png                        the same for B. Compare the two:
                                              they should find the same things
                                              in the same places

  Stage 3 - foreground stars removed

    Stars of our own galaxy lie in front of the target and are not part of
    it. They are matched against the catalogue and taken out.

    NAME_calibrated_photometry_no_stars.csv
    NAME_calibrated_photometry_no_stars_with_color.csv
    NAME_calibrated_photometry_no_stars_CMD.png
    NAME_V_sources_no_stars.png

  Stage 4 - blue knots only

    What is left is cut on colour, keeping the blue sources: the young
    star-forming regions in the arms. THIS IS THE RESULT the projects are
    usually after.

    NAME_calibrated_photometry_no_stars_color_filtered.csv
    NAME_calibrated_photometry_no_stars_color_filtered_with_color.csv
    NAME_calibrated_photometry_no_stars_color_filtered_CMD.png
    NAME_V_sources_color_filtered.png
    NAME_calibrated_photometry_no_stars_color_filtered_with_radius.csv
                                    the same, plus each knot's distance
                                    from the centre of the galaxy

  Stage 5 - how the knots are distributed

    NAME_radial_density_profile.csv       knots per square pixel, by radius
    NAME_radial_density_profile_pc.csv    the same in parsecs, using the
                                          distance you gave
    NAME_radial_density_profile_step.csv     the same as a step function
    NAME_radial_density_profile_step_pc.csv  the step function in parsecs
    NAME_radial_density_profile_step.png       the plot, in pixels
    NAME_radial_density_profile_step_pc.png    the plot, in parsecs

  If you look at four files, look at these:

    NAME_reference_stars.csv                   did the calibration have
                                               anything to work with
    NAME_V_sources_color_filtered.png          did it find the arms
    NAME_..._color_filtered_CMD.png            the diagram
    NAME_radial_density_profile_step_pc.png    the arms, in numbers

  A _with_color file is its parent file with the B-V column added. If you
  only want one, take the _with_color one.

The parameters at the top of this file are tuned for the Kinneret frames
these projects use. On very different data - a much smaller telescope, a
much fainter galaxy - the detection threshold and the matching tolerances
are the first things to look at.

Usage: run it. Every question has a default; press Enter to accept it.
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.background import Background2D, MedianBackground
from astropy.stats import sigma_clipped_stats, SigmaClip
from scipy.ndimage import gaussian_filter
from astropy.wcs import WCS
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
import astroalign as aa


# ===================== PARAMETERS =====================
# Aperture/annulus scaling relative to FWHM
APERTURE_SCALE = 2
ANNULUS_INNER_SCALE = 2.5
ANNULUS_OUTER_SCALE = 3
# FWHM measurement window (pixels from center)
FWHM_WINDOW_SIZE = 10

# Detection parameters
#
# The width of a star is measured from your own frames, not assumed: see
# estimate_seeing(). A fixed guess is the wrong thing here. DAOStarFinder
# convolves the image with a kernel of the width it is given, so a star three
# times narrower than that guess fails the sharpness test and is never found -
# and it fails more often in the sharper of two filters, which puts a colour
# error into every source in the frame.
DAOFIND_FWHM = None              # None = measure it from the data (recommended)
SEEING_FALLBACK = 4.0            # Used only if the measurement cannot be made
SEEING_MIN, SEEING_MAX = 1.5, 15.0   # Sanity range for the measured value, pixels

SIGMA_CLIP = 3.0                 # Sigma clipping level for background statistics
DETECTION_THRESHOLD_SIGMA = 3.0  # Detection threshold in units of background sigma
PEAK_MIN_STD = 3.0               # Minimum peak signal-to-noise ratio for a detection

# Background for detection. One median for a whole frame cannot describe a
# galaxy: across these frames the local sky runs from -2 to +62 counts, so a
# single global threshold is far too high in the empty corners and far too low
# in the disc. A coarse grid follows the diffuse light instead.
BACKGROUND_BOX_SIZE = 48         # Grid cell for the local background, pixels
BACKGROUND_FILTER_SIZE = 3       # Median filter applied across the grid

# Filled in by process_fits() once the frames are read, and written into the
# output so a run can be reproduced.
SEEING_MEASURED = {}

# Background subtraction
BG_SUB_FUNC = np.nanmedian       # Function used to compute and subtract background level

# Matching a detection in B to the same object in V. Like the catalogue
# tolerance below, this is tied to the measured seeing rather than fixed. At a
# flat 15 px - nearly four times the width of a star on these frames - noise in
# one band pairs with unrelated noise in the other: such pairs carried three
# times the colour scatter of real ones and made up a quarter of the catalogue.
MATCH_SCALE = 1.5                # Tolerance as a multiple of the measured FWHM
MATCH_TOLERANCE_MIN = 3.0        # ...but never tighter than this, in pixels

# B is registered onto V's pixel grid before any matching happens (see
# register_B_onto_V) - B and V are separate exposures and generally do not
# share a pixel grid at all. On one real pair of galaxy frames the shift
# turned out to be under 2 px and this made no visible difference; on a pair
# of star-cluster frames from the same kind of pipeline it was 18 px plus a
# degree of rotation, and every match tolerance above was then measuring the
# wrong thing. A sanity check on the fitted scale catches a registration
# astroalign could not really trust: B and V come off the same telescope, so
# the pixel scale between them should agree to a fraction of a percent.
ASTROALIGN_MAX_SCALE_ERROR = 0.05  # 5%

# Reference star parameters
REF_MAG_LIMIT = 20.0             # Catalog magnitude limit for selecting reference stars
REF_CATALOG = "II/336/apass9"    # Reference star catalog to use (APASS9 with B,V magnitudes)

# Matching a catalogue star to what was actually measured. This used to be a
# flat 20 pixels, which is 15 arcsec on these frames: wide enough that when a
# star was missed, the nearest noise blob was accepted in its place without a
# word. The tolerance now scales with the measured seeing, and two further
# checks below reject a pairing that does not hold up.
CATALOG_FLUX_MATCH_SCALE = 1.5   # Tolerance as a multiple of the measured FWHM
CATALOG_FLUX_MATCH_MIN = 3.0     # ...but never tighter than this, in pixels
REF_CROSS_BAND_TOLERANCE = 2.0   # B and V must land on the same object, within
                                 # this multiple of the FWHM, or the star is
                                 # dropped from the calibration
REF_ZP_OUTLIER_SIGMA = 3.0       # Reject a reference star whose implied zero
                                 # point is this far from the robust value

# Calibration parameters
CALIB_NUM_STARS = None           # Number of reference stars for calibration (user will be asked)

# CMD plot range for color index (B-V)
CMD_COLOR_MIN = -0.5   # left limit of x-axis
CMD_COLOR_MAX = 0.5    # right limit of x-axis

# === Parameters ===
REMOVE_TOL = 10.0  # pixel tolerance for matching sources to catalog stars

Step_size = 10    # Resolution of The Blue Knots Density Diagram 

# ---------------- PATH NORMALIZATION ----------------
def _norm_path(p: str) -> str:
    """Remove extra quotes and normalize filesystem path."""
    p = p.strip().strip('"').strip("'")
    return os.path.normpath(p)

# ---------------- USER INPUT ----------------
def _ask(prompt, default, cast=str):
    """Ask user for input with default and type casting."""
    s = input(f"{prompt} [{default}]: ").strip()
    if not s:
        return default
    try:
        return cast(s)
    except Exception:
        return default

# ---------- Background subtraction ----------
def subtract_background_and_save(path):
    """Subtract background (median or other function) from FITS and save new file."""
    data, hdr = fits.getdata(path, header=True)
    bg_val = BG_SUB_FUNC(data)
    data_sub = data - bg_val
    # into result_CMD, not next to the frames: the folder holding your data
    # should come out of a run exactly as it went in.
    out_dir = os.path.join(os.path.dirname(path) or os.getcwd(), "result_CMD")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, stem + "_bgsub.fits")
    fits.writeto(out_path, data_sub, hdr, overwrite=True)
    print(f"[bgsub] wrote {out_path} (bg={bg_val:.3f})")
    return out_path


# ------------Measure FWHM around a light source-----------
def compute_fwhm(data, x, y, size=FWHM_WINDOW_SIZE):
    """Measure FWHM around a light source."""
    x_min, x_max = int(x-size), int(x+size)
    y_min, y_max = int(y-size), int(y+size)
    if x_min < 0 or y_min < 0 or x_max >= data.shape[1] or y_max >= data.shape[0]:
        # A source too close to the edge for a full measuring box. There are
        # hundreds of these on a typical frame; printing one line each buried
        # everything else, so they are counted and reported once at the end.
        return None

    sub_image = data[y_min:y_max, x_min:x_max]
    smoothed = gaussian_filter(sub_image, sigma=2)
    peak = np.max(smoothed)
    half_max = peak / 2
    above_half_max = smoothed > half_max
    indices = np.argwhere(above_half_max)
    if indices.size > 0:
        min_x, max_x = indices[:, 1].min(), indices[:, 1].max()
        min_y, max_y = indices[:, 0].min(), indices[:, 0].max()
        fwhm_x = max_x - min_x
        fwhm_y = max_y - min_y
        return np.mean([fwhm_x, fwhm_y])
    return None

def local_background(data):
    """Background and noise on a coarse grid, so a galaxy's own light does not
    set the detection threshold for the empty sky around it."""
    return Background2D(data, (BACKGROUND_BOX_SIZE, BACKGROUND_BOX_SIZE),
                        filter_size=(BACKGROUND_FILTER_SIZE, BACKGROUND_FILTER_SIZE),
                        sigma_clip=SigmaClip(sigma=SIGMA_CLIP),
                        bkg_estimator=MedianBackground())


def star_width(data, x, y, half=12):
    """Width at half maximum of one source, in pixels, measured on the data as
    it is. compute_fwhm() smooths first, which suits aperture sizing but adds
    about 3 pixels; a detection kernel needs the true width."""
    x, y = int(round(x)), int(round(y))
    if x - half < 0 or y - half < 0 or x + half >= data.shape[1] or y + half >= data.shape[0]:
        return np.nan
    cut = data[y - half:y + half, x - half:x + half]
    peak = cut.max()
    if not np.isfinite(peak) or peak <= 0:
        return np.nan
    idx = np.argwhere(cut > peak / 2)
    if not len(idx):
        return np.nan
    return np.mean([np.ptp(idx[:, 1]) + 1, np.ptp(idx[:, 0]) + 1])


def estimate_seeing(data, background, rms, band):
    """Measure how wide a star actually is in this frame.

    Detect once at a high threshold to get bright, unambiguous stars, measure
    each one, and take the median. Nothing about the telescope is assumed and
    no header keyword is required, so this works on any frame.
    """
    try:
        found = DAOStarFinder(fwhm=SEEING_FALLBACK, threshold=20.0 * rms)(data - background)
        if found is not None and len(found) >= 5:
            brightest = found[np.argsort(found['flux'])][-200:]
            widths = [star_width(data - background, s['xcentroid'], s['ycentroid'])
                      for s in brightest]
            widths = np.asarray(widths, dtype=float)
            seeing = np.nanmedian(widths)
            if np.isfinite(seeing) and SEEING_MIN <= seeing <= SEEING_MAX:
                print(f"   {band}: stars are {seeing:.2f} px wide"
                      f" (measured on {int(np.isfinite(widths).sum())} bright stars)", flush=True)
                return float(seeing)
        print(f"   {band}: could not measure the star width, using"
              f" {SEEING_FALLBACK} px", flush=True)
    except Exception as exc:
        print(f"   {band}: star width measurement failed ({exc}), using"
              f" {SEEING_FALLBACK} px", flush=True)
    return float(SEEING_FALLBACK)


def process_fits(filename, band):
    """Detect sources, perform aperture photometry and return results."""
    hdul = fits.open(filename)
    data = hdul[0].data
    hdul.close()

    print(f"Finding sources in the {band} image...", flush=True)
    bkg = local_background(data)
    rms = float(np.median(bkg.background_rms))
    print(f"   {band}: local sky runs from {bkg.background.min():.1f}"
          f" to {bkg.background.max():.1f} counts, noise {rms:.2f}", flush=True)

    fwhm_detect = DAOFIND_FWHM if DAOFIND_FWHM else estimate_seeing(
        data, bkg.background, rms, band)
    SEEING_MEASURED[band] = fwhm_detect

    threshold = DETECTION_THRESHOLD_SIGMA * rms
    daofind = DAOStarFinder(fwhm=fwhm_detect, threshold=threshold)
    sources = daofind(data - bkg.background)
    if sources is None or len(sources) == 0:
        print(f"   {band}: no sources found", flush=True)
        return []
    sources = sources[sources['peak'] > PEAK_MIN_STD * rms]

    # The frame with the smooth galaxy light removed, used for measuring widths.
    flat = data - bkg.background

    # Measuring every source takes the longest of any step here and used to run
    # in silence. Report about ten times, each on its own line: Spyder's console
    # does not reliably honour a carriage return, and a rewritten line then
    # becomes one very long one.
    n_total = len(sources)
    step = max(1, n_total // 10)
    print(f"Measuring {n_total:,} sources in the {band} image...", flush=True)

    n_edge = n_nosignal = n_negative = 0
    results = []
    for i, source in enumerate(sources, 1):
        if i % step == 0 or i == n_total:
            print(f"   {i:,} of {n_total:,}", flush=True)
        x, y = source['xcentroid'], source['ycentroid']
        # Width is measured on the frame with the diffuse light taken out, and
        # the flux on the frame as it is.
        #
        # compute_fwhm asks which pixels sit above half of the brightest one. On
        # a galaxy that question has no useful answer: inside a box on the disc
        # the counts run from about 50 to 63, half of the peak is 31, and not one
        # pixel falls below it - so every source came back as wide as the box,
        # and its aperture was sized by the box rather than by itself. Taking the
        # smooth galaxy light out first puts the floor back at zero, and half the
        # peak becomes a threshold that separates the source from what surrounds
        # it. A star on empty sky is unaffected: its floor was already zero.
        fwhm = compute_fwhm(flat, x, y)
        # fwhm is None       - too close to the edge for a full measuring box
        # fwhm is zero/NaN   - a detection that turned out to be nothing. On a
        #                      background-subtracted image the local peak of a
        #                      noise ripple can be 0.006 counts; half of that is
        #                      a meaningless threshold, one pixel clears it, and
        #                      the width between that pixel and itself is zero.
        #                      Passing it on gives photutils a radius of zero
        #                      and the run stops.
        if fwhm is None:
            n_edge += 1
            continue
        if not (np.isfinite(fwhm) and fwhm > 0):
            n_nosignal += 1
            continue

        radius = APERTURE_SCALE * fwhm
        aperture = CircularAperture((x, y), r=radius)
        annulus_inner_radius = radius * ANNULUS_INNER_SCALE
        annulus_outer_radius = radius * ANNULUS_OUTER_SCALE
        annulus = CircularAnnulus((x, y), r_in=annulus_inner_radius, r_out=annulus_outer_radius)

        phot_table = aperture_photometry(data, [aperture, annulus])
        background_mean = phot_table['aperture_sum_1'][0] / annulus.area
        background_subtracted_flux = phot_table['aperture_sum_0'][0] - background_mean * aperture.area

        if background_subtracted_flux < 0:
            n_negative += 1
            continue

        results.append([x, y, fwhm, radius, background_subtracted_flux,
                        band, annulus_inner_radius, annulus_outer_radius])

    print(f"   {len(results):,} measured"
          f"   |   skipped: {n_edge:,} at the frame edge,"
          f" {n_nosignal:,} with no measurable width,"
          f" {n_negative:,} with negative flux", flush=True)
    return results

# ---------------- REGISTERING B ONTO V ----------------
def register_B_onto_V(B_img, V_img, xB, yB, fits_file_B, fits_file_V):
    """Where each B-image position (xB, yB) lands in V's pixel grid.

    B and V are separate exposures, and nothing guarantees their pixel grids
    coincide - the telescope can move between them by an amount that depends
    on the mount, the dither, how long the filter change took. Every match
    tolerance in this file is only correct once that is corrected for.

    astroalign finds the registration directly from the star patterns in
    these two specific images - it does not assume any telescope, field, or
    plate scale, and does not depend on how good either WCS solution is.
    Falls back to each frame's own WCS only if astroalign cannot find enough
    matching stars, or its fitted scale is not plausible for two frames from
    the same telescope."""
    try:
        transf, (src_list, _tgt_list) = aa.find_transform(B_img, V_img)
        if abs(transf.scale - 1.0) > ASTROALIGN_MAX_SCALE_ERROR:
            raise ValueError(f"fitted scale {transf.scale:.3f} is not plausible "
                             f"for two frames from the same telescope")
        xy = transf(np.column_stack([xB, yB]))
        print(f"   registered B onto V from {len(src_list)} matched star "
             f"pairs (astroalign): shift ({transf.translation[0]:+.1f}, "
             f"{transf.translation[1]:+.1f}) px, rotation "
             f"{np.degrees(transf.rotation):+.2f} deg, scale {transf.scale:.4f}",
             flush=True)
        return xy[:, 0], xy[:, 1]
    except Exception as e:
        print(f"   astroalign registration failed ({e}) - "
             f"falling back to each frame's own WCS", flush=True)
        wcs_B = WCS(fits.getheader(fits_file_B))
        wcs_V = WCS(fits.getheader(fits_file_V))
        ra_B, dec_B = wcs_B.all_pix2world(xB, yB, 0)
        return wcs_V.all_world2pix(ra_B, dec_B, 0)


# ---------------- MATCHING FUNCTION ----------------
def match_sources(df_B, df_V, tol, xB_reg=None, yB_reg=None):
    """Match B and V sources by nearest (X,Y) within tolerance.

    xB_reg/yB_reg, when given, are each B source's position after
    register_B_onto_V() - what the matching distance is measured on. df_B's
    own X/Y are still what gets stored in the output, since aperture
    photometry on the B image needs B's own pixel coordinates, not V's."""
    if xB_reg is None:
        xB_reg = df_B["X"].values
    if yB_reg is None:
        yB_reg = df_B["Y"].values
    matched_rows = []
    used_V = set()
    for i, (_, rowB) in enumerate(df_B.iterrows()):
        xB, yB = xB_reg[i], yB_reg[i]
        dists = np.sqrt((df_V["X"] - xB)**2 + (df_V["Y"] - yB)**2)
        min_idx = dists.idxmin()
        if dists[min_idx] <= tol and min_idx not in used_V:
            rowV = df_V.loc[min_idx]
            merged = {
                "X_B": rowB["X"], "Y_B": rowB["Y"],
                "FWHM_B": rowB["FWHM"], "Flux_B": rowB["Flux"],
                "X_V": rowV["X"], "Y_V": rowV["Y"],
                "FWHM_V": rowV["FWHM"], "Flux_V": rowV["Flux"]
            }
            matched_rows.append(merged)
            used_V.add(min_idx)
    return pd.DataFrame(matched_rows)

# ---------------- REFERENCE STARS (Vizier) ----------------
def extract_reference_stars(fits_file, df_B, df_V, xB_reg=None, yB_reg=None,
                            mag_limit=15.0,
                            catalog="II/336/apass9",
                            fwhm_hint=4.0):
    """Query Vizier and return reference stars with catalog mags, measured fluxes and positions (B,V).

    fits_file is V's - catalogue positions come out in V's pixel grid. df_V's
    own X/Y are already in that same grid, but df_B's are not (B is a
    separate exposure), so xB_reg/yB_reg - B's positions after
    register_B_onto_V() - are what the catalogue is actually matched against
    in B. The native df_B X/Y are still what gets stored, for photometry on
    the B image itself."""
    if xB_reg is None:
        xB_reg = df_B["X"].values
    if yB_reg is None:
        yB_reg = df_B["Y"].values
    # How far from the catalogue position a detection may be and still be
    # accepted as that star. Tied to the seeing, so it stays a fraction of a
    # star's width rather than a flat number of pixels.
    flux_tol = max(CATALOG_FLUX_MATCH_MIN, CATALOG_FLUX_MATCH_SCALE * fwhm_hint)
    print(f"   matching catalogue stars to detections within {flux_tol:.1f} px", flush=True)
    hdr = fits.getheader(fits_file)
    wcs = WCS(hdr)

    ra_center, dec_center = wcs.wcs.crval
    naxis1, naxis2 = hdr["NAXIS1"], hdr["NAXIS2"]
    scale_deg = np.mean(np.abs(wcs.pixel_scale_matrix.diagonal()))
    fov_ra = naxis1 * scale_deg
    fov_dec = naxis2 * scale_deg

    # row_limit has to be passed to the constructor, not set on the class
    # afterwards - see the same fix in Cluster_CMD.py's get_apass_calib_stars
    # for why: doing it on the class leaves a freshly constructed instance at
    # astroquery's default of 50 rows, silently.
    v = Vizier(columns=["RAJ2000","DEJ2000","Bmag","Vmag"],
               column_filters={"Vmag":"<%.2f" % mag_limit}, row_limit=-1)
    result = v.query_region(
        SkyCoord(ra_center, dec_center, unit="deg"),
        width=f"{fov_ra}d", height=f"{fov_dec}d",
        catalog=catalog
    )

    if len(result) == 0:
        print("No reference stars found in Vizier catalog.")
        return pd.DataFrame()

    stars = result[0]
    coords = SkyCoord(stars["RAJ2000"], stars["DEJ2000"], unit="deg")
    x_pix, y_pix = wcs.world_to_pixel(coords)

    df_ref = pd.DataFrame({
        "RA": stars["RAJ2000"],
        "Dec": stars["DEJ2000"],
        "Bmag": stars["Bmag"],
        "Vmag": stars["Vmag"],
        "X_pix": x_pix,
        "Y_pix": y_pix
    })

    flux_B, flux_V = [], []
    XB_meas, YB_meas, XV_meas, YV_meas = [], [], [], []
    XB_reg_meas, YB_reg_meas = [], []

    for _, row in df_ref.iterrows():
        dB = np.sqrt((xB_reg - row["X_pix"])**2 + (yB_reg - row["Y_pix"])**2)
        dV = np.sqrt((df_V["X"] - row["X_pix"])**2 + (df_V["Y"] - row["Y_pix"])**2)

        if dB.min() <= flux_tol:
            idxB = int(np.argmin(dB))
            fB = df_B.iloc[idxB]["Flux"]
            XB, YB = df_B.iloc[idxB]["X"], df_B.iloc[idxB]["Y"]
            XB_reg, YB_reg = xB_reg[idxB], yB_reg[idxB]
        else:
            fB, XB, YB, XB_reg, YB_reg = np.nan, np.nan, np.nan, np.nan, np.nan

        if dV.min() <= flux_tol:
            idxV = dV.idxmin()
            fV = df_V.loc[idxV, "Flux"]
            XV, YV = df_V.loc[idxV, "X"], df_V.loc[idxV, "Y"]
        else:
            fV, XV, YV = np.nan, np.nan, np.nan

        flux_B.append(fB)
        flux_V.append(fV)
        XB_meas.append(XB)
        YB_meas.append(YB)
        XV_meas.append(XV)
        YV_meas.append(YV)
        XB_reg_meas.append(XB_reg)
        YB_reg_meas.append(YB_reg)

    df_ref["Flux_B_measured"] = flux_B
    df_ref["Flux_V_measured"] = flux_V
    df_ref["X_B"] = XB_meas
    df_ref["Y_B"] = YB_meas
    df_ref["X_V"] = XV_meas
    df_ref["Y_V"] = YV_meas

    # filter only stars with valid flux in both bands
    n_found = len(df_ref)
    df_ref = df_ref.dropna(subset=["Flux_B_measured","Flux_V_measured"])
    n_bothbands = len(df_ref)
    XB_reg_meas = np.array(XB_reg_meas)[df_ref.index] if n_bothbands else np.array([])
    YB_reg_meas = np.array(YB_reg_meas)[df_ref.index] if n_bothbands else np.array([])

    # How far each band's measurement ended up from where the catalogue says the
    # star is, and from the other band. Kept in the file: when a calibration
    # goes wrong these two columns say so at a glance.
    #
    # Both distances use B's position after registration onto V's grid, not
    # its native pixel position - X_pix/Y_pix and X_V/Y_V are already in that
    # grid, and comparing them against B's own, unregistered pixels would
    # report an offset that is really just the two frames' own difference.
    df_ref["Offset_B"] = np.hypot(XB_reg_meas - df_ref["X_pix"],
                                  YB_reg_meas - df_ref["Y_pix"])
    df_ref["Offset_V"] = np.hypot(df_ref["X_V"] - df_ref["X_pix"],
                                  df_ref["Y_V"] - df_ref["Y_pix"])
    df_ref["Separation_B_V"] = np.hypot(XB_reg_meas - df_ref["X_V"],
                                        YB_reg_meas - df_ref["Y_V"])

    # A star is only usable if both bands measured the same object. When one
    # band misses the star, the nearest blob is accepted instead and the two
    # bands drift apart - which is exactly the pairing that ruins a zero point.
    cross_tol = REF_CROSS_BAND_TOLERANCE * max(fwhm_hint, 1.0)
    same_object = df_ref["Separation_B_V"] <= cross_tol
    n_split = int((~same_object).sum())
    df_ref = df_ref[same_object]

    print(f"   reference stars: {n_found} in the catalogue,"
          f" {n_bothbands} measured in both bands,"
          f" {n_split} dropped for landing on different objects in B and V"
          f" (more than {cross_tol:.1f} px apart),"
          f" {len(df_ref)} usable", flush=True)

    return df_ref.reset_index(drop=True)


# ---------------- CALIBRATION FUNCTION ----------------
def compute_zero_point(fluxes, mags, label=""):
    """Zero point from reference stars, combined so that one bad star cannot
    carry the frame.

    This was a plain mean, and a mean is the wrong tool. A reference star whose
    flux was measured on the wrong object pairs a real catalogue magnitude with
    a meaningless flux; the pair can be several magnitudes out, and a single one
    moves the mean enough to shift every magnitude in the output. Taking the
    median first and then clipping about it makes the bad stars visible and
    harmless: they are reported, not averaged in.
    """
    fluxes = np.asarray(fluxes, dtype=float)
    mags = np.asarray(mags, dtype=float)
    mask = np.isfinite(fluxes) & np.isfinite(mags) & (fluxes > 0)
    if not mask.any():
        return np.nan, 0, 0

    zp = mags[mask] + 2.5 * np.log10(fluxes[mask])
    keep = np.ones(len(zp), dtype=bool)
    if len(zp) >= 4:
        spread = np.median(np.abs(zp - np.median(zp))) * 1.4826  # robust sigma
        if spread > 0:
            keep = np.abs(zp - np.median(zp)) <= REF_ZP_OUTLIER_SIGMA * spread
    value = float(np.median(zp[keep])) if keep.any() else float(np.median(zp))
    n_used, n_cut = int(keep.sum()), int((~keep).sum())
    if label:
        print(f"   zero point {label} = {value:.3f}"
              f"   from {n_used} stars, {n_cut} rejected as outliers"
              f"   (scatter {np.std(zp[keep]):.3f} mag)", flush=True)
    return value, n_used, n_cut


# ----- Add color index (B-V) and create CMD for both full and cleaned catalogs -----

def make_cmd(df, base_file, label):
    if "Mag_B" in df.columns and "Mag_V" in df.columns:
        # Compute color index
        df = df.copy()
        df["B-V"] = df["Mag_B"] - df["Mag_V"]

        # Reorder columns: place B-V immediately before Mag_V
        cols = list(df.columns)
        if "B-V" in cols and "Mag_V" in cols:
            cols.remove("B-V")
            cols.remove("Mag_V")
            # put everything else first, then B-V, then Mag_V
            cols = cols + ["B-V", "Mag_V"]
            df = df[cols]

        # Save new file with color index
        # base_file already carries whichever stage this is - "_no_stars",
        # "_no_stars_color_filtered" - so the stage is not appended again.
        # It used to be, which produced names like
        # ..._no_stars_color_filtered_with_color_no_stars_color_filtered.csv
        out_file_color = base_file.replace(".csv", "_with_color.csv")
        df.to_csv(out_file_color, index=False)
        print(f"File with color index saved to {out_file_color}")

        # Create CMD plot
        plt.figure(figsize=(8, 10))
        plt.scatter(df["B-V"], df["Mag_V"],
                    s=30, edgecolor="black", facecolor="cyan", alpha=0.7)

        plt.gca().invert_yaxis()  # brighter objects at the top
        plt.xlabel("B - V (Color Index)")
        plt.ylabel("V magnitude")
        plt.title(f"CMD of {obj_name} {label} (B-V, V)")

        # >>> Control of x-axis (color index) range <<<
        plt.xlim(CMD_COLOR_MIN, CMD_COLOR_MAX)

        out_cmd = base_file.replace(".csv", "_CMD.png")
        plt.savefig(out_cmd, dpi=150)
        plt.close()
        print(f"CMD diagram saved to {out_cmd}")
    else:
        print(f"Mag_B or Mag_V not found in {label} catalog. CMD not created.")

# ---------------- LOOK THE GALAXY UP ----------------
# Two of the three numbers this script needs about the galaxy are published,
# and asking a student to type them from memory is how a run ends up assuming
# M101 sits at 1 Mpc. They are fetched by name and offered as the default; you
# can still type your own.

R_V = 3.1                        # A_V / E(B-V) for dust in our own galaxy
DISTANCE_CATALOG = "J/AJ/152/50"  # Cosmicflows-3, Tully et al. 2016


def lookup_distance(name):
    """Redshift-independent distance in Mpc from Cosmicflows-3, or None.

    Redshift is no use for a galaxy this close - M101's recession velocity puts
    it at about 3 Mpc, roughly half its real distance - so a catalogue of direct
    measurements is what is wanted.
    """
    try:
        coord = SkyCoord.from_name(name)
        hit = Vizier(columns=["**"], row_limit=5).query_region(
            coord, radius=2 * u.arcmin, catalog=DISTANCE_CATALOG)
        if not hit:
            return None
        table = hit[0]
        for col in ("Dist", "<Dist>"):
            if col in table.colnames:
                value = float(table[col][0])
                if np.isfinite(value) and value > 0:
                    print(f"[lookup] {name}: {value:.2f} Mpc"
                          f" (Cosmicflows-3)", flush=True)
                    return value
    except Exception as exc:
        print(f"[lookup] distance unavailable ({exc})", flush=True)
    return None


def lookup_center_pixel(name, header):
    """Where the galaxy actually sits in this frame, in pixels, or None.

    The radial profile used to assume the galaxy sits exactly at the frame's
    own geometric centre - true only if the telescope was pointed dead-on. On
    a real pair of M101 frames the pointing was off by 72 px, about 1.8 kpc at
    that distance: not huge next to the whole profile, but enough to bias the
    inner rings, where 72 px is a large fraction of the radius, and to shift
    the whole radius scale. The galaxy's own catalogued position, converted
    through this frame's WCS, does not depend on how well the pointing landed.
    """
    try:
        coord = SkyCoord.from_name(name)
        w = WCS(header)
        x, y = w.world_to_pixel(coord)
        x, y = float(x), float(y)
        nx, ny = header["NAXIS1"], header["NAXIS2"]
        if 0 <= x < nx and 0 <= y < ny:
            print(f"[lookup] {name}: centred at pixel ({x:.0f}, {y:.0f})", flush=True)
            return x, y
        print(f"[lookup] {name}'s catalogued position falls outside this frame "
             f"- using the frame's own centre for the radial profile instead", flush=True)
    except Exception as exc:
        print(f"[lookup] galaxy centre unavailable ({exc}) - using the frame's "
             f"own centre for the radial profile instead", flush=True)
    return None, None


def lookup_extinction(name):
    """Galactic reddening E(B-V) towards the galaxy, or None.

    Schlafly & Finkbeiner (2011) recalibration of the Schlegel dust maps, served
    by IRSA. A_V follows as R_V x E(B-V); the two are not free to be set apart
    from each other, which is why only one of them is asked for.
    """
    try:
        from astroquery.ipac.irsa.irsa_dust import IrsaDust
        table = IrsaDust.get_query_table(name, section="ebv")
        value = float(table["ext SandF mean"][0])
        if np.isfinite(value) and value >= 0:
            print(f"[lookup] {name}: E(B-V) = {value:.4f},"
                  f" so A_V = {R_V * value:.3f} (Schlafly & Finkbeiner 2011)",
                  flush=True)
            return value
    except Exception as exc:
        print(f"[lookup] extinction unavailable ({exc})", flush=True)
    return None


# ---------------- RUN ANALYSIS ----------------
print("=== Photometry Input ===")
obj_name = _ask("Galaxy name", "M101", str)

print("Looking the galaxy up...", flush=True)
_dist_default = lookup_distance(obj_name)
_ebv_default = lookup_extinction(obj_name)
if _dist_default is None:
    print("[lookup] no published distance found - please supply one", flush=True)
if _ebv_default is None:
    print("[lookup] no reddening found - please supply one", flush=True)

distance = _ask("Distance (Mpc)", _dist_default if _dist_default else 10.0, float)
E_BV     = _ask("Galactic color excess E(B-V)",
                _ebv_default if _ebv_default is not None else 0.0, float)
# A_V is not asked for separately: it is R_V x E(B-V) by definition, and asking
# for both invites a pair that cannot both be true. The defaults this script
# used to offer, A_V 0.1 with E(B-V) 0.05, imply R_V = 2.0.
A_V = R_V * E_BV
print(f"Using distance {distance:.2f} Mpc,"
      f" E(B-V) = {E_BV:.4f}, A_V = {A_V:.3f}", flush=True)
fits_file_B = _norm_path(_ask("Path to B-band FITS", r"D:\example_B.fts", str))
fits_file_V = _norm_path(_ask("Path to V-band FITS", r"D:\example_V.fts", str))

# Everything this script writes goes into one sub-folder beside the images,
# rather than being scattered through the folder holding your data. The frames
# stay where they are and are only read.
_datadir = os.path.dirname(fits_file_B) if os.path.dirname(fits_file_B) else os.getcwd()
_outdir = os.path.join(_datadir, "result_CMD")
os.makedirs(_outdir, exist_ok=True)
print(f"[out] writing everything to {_outdir}")
fits_file_B = subtract_background_and_save(fits_file_B)
fits_file_V = subtract_background_and_save(fits_file_V)

results_B = process_fits(fits_file_B, "B")
results_V = process_fits(fits_file_V, "V")

df_B = pd.DataFrame(results_B, columns=[
    "X", "Y", "FWHM", "Aperture Radius", "Flux",
    "Band", "Annulus Inner Radius", "Annulus Outer Radius"
])
df_V = pd.DataFrame(results_V, columns=[
    "X", "Y", "FWHM", "Aperture Radius", "Flux",
    "Band", "Annulus Inner Radius", "Annulus Outer Radius"
])

_fwhm_hint = max(SEEING_MEASURED.get("B", SEEING_FALLBACK),
                 SEEING_MEASURED.get("V", SEEING_FALLBACK))

# B is registered onto V's pixel grid once, here, before any matching -
# against V's own detections and against the reference-star catalogue alike.
print("Registering B onto V...", flush=True)
_B_img = fits.getdata(fits_file_B, ext=0).astype(float)
_V_img = fits.getdata(fits_file_V, ext=0).astype(float)
_xB_reg, _yB_reg = register_B_onto_V(_B_img, _V_img, df_B["X"].values, df_B["Y"].values,
                                     fits_file_B, fits_file_V)

_match_tol = max(MATCH_TOLERANCE_MIN, MATCH_SCALE * _fwhm_hint)
print(f"Matching B to V within {_match_tol:.1f} px"
      f" (stars are {_fwhm_hint:.1f} px wide)", flush=True)
df_matched = match_sources(df_B, df_V, tol=_match_tol, xB_reg=_xB_reg, yB_reg=_yB_reg)
print(f"   {len(df_matched):,} sources measured in both bands", flush=True)

csv_filename = os.path.join(_outdir, f"{obj_name}_photometry_results.csv")
df_matched.to_csv(csv_filename, index=False)
print(f"Data saved to {csv_filename}")

# ----- Run reference star extraction -----
df_ref = extract_reference_stars(fits_file_V, df_B, df_V, xB_reg=_xB_reg, yB_reg=_yB_reg,
                                 mag_limit=REF_MAG_LIMIT,
                                 catalog=REF_CATALOG,
                                 fwhm_hint=_fwhm_hint)
if not df_ref.empty:
    csv_ref = os.path.join(_outdir, f"{obj_name}_reference_stars.csv")
    df_ref.to_csv(csv_ref, index=False)
    print(f"Reference stars saved to {csv_ref}")

    # ----- Calibration using N reference stars -----
    print(f"{len(df_ref)} reference stars available.")
    N = _ask("How many to calibrate against", len(df_ref), int)
    N = max(1, min(int(N), len(df_ref)))
    df_calib = df_ref.head(N)

    # compute zero points
    zp_B, n_zp_B, n_cut_B = compute_zero_point(
        df_calib["Flux_B_measured"].values, df_calib["Bmag"].values, label="B")
    zp_V, n_zp_V, n_cut_V = compute_zero_point(
        df_calib["Flux_V_measured"].values, df_calib["Vmag"].values, label="V")

    # add aperture radii
    df_matched["Aperture_Radius_B"] = df_matched["FWHM_B"] * APERTURE_SCALE
    df_matched["Aperture_Radius_V"] = df_matched["FWHM_V"] * APERTURE_SCALE

    # Calibrated magnitudes, then the foreground dust of our own galaxy taken
    # out. A_V dims the V band; B is dimmed by A_V + E(B-V), which is what the
    # colour excess means. Subtracting both leaves the colour reddened by
    # exactly E(B-V), so that is what comes off B-V.
    df_matched["Mag_B"] = zp_B - 2.5 * np.log10(df_matched["Flux_B"]) - (A_V + E_BV)
    df_matched["Mag_V"] = zp_V - 2.5 * np.log10(df_matched["Flux_V"]) - A_V

    # What this run assumed, carried in the data itself. Without these columns
    # there is no way to tell afterwards which distance or extinction produced
    # a given set of numbers.
    df_matched["ZP_B_used"] = zp_B
    df_matched["ZP_V_used"] = zp_V
    df_matched["N_ref_used"] = n_zp_B + n_zp_V
    df_matched["A_V_used"] = A_V
    df_matched["E_BV_used"] = E_BV
    df_matched["Distance_Mpc_used"] = distance
    df_matched["FWHM_measured_B"] = SEEING_MEASURED.get("B", np.nan)
    df_matched["FWHM_measured_V"] = SEEING_MEASURED.get("V", np.nan)

    # save calibrated photometry
    csv_calib = os.path.join(_outdir, f"{obj_name}_calibrated_photometry.csv")
    df_matched.to_csv(csv_calib, index=False)
    print(f"Calibrated photometry saved to {csv_calib}")
       
    # ----- Create V-band image with detected sources -----
data_V, hdr_V = fits.getdata(fits_file_V, header=True)

plt.figure(figsize=(10, 10))
plt.imshow(data_V, cmap="gray", origin="lower", vmin=np.percentile(data_V, 5), vmax=np.percentile(data_V, 99))
plt.colorbar(label="Counts")

# overlay measured sources from df_V
plt.scatter(df_V["X"], df_V["Y"], s=40, edgecolor="green", facecolor="none", label="Measured sources")

plt.title(f"{obj_name} - V band with detected sources")
plt.xlabel("X [pixels]")
plt.ylabel("Y [pixels]")
plt.legend()

out_png = os.path.join(_outdir, f"{obj_name}_V_sources.png")
plt.savefig(out_png, dpi=150)
plt.close()
print(f"V-band source map saved to {out_png}")

# The same for B. There used to be no B map, and without one there is no way to
# see by eye that the two filters found the same things - which is precisely the
# failure that put a colour error into every source here.
data_B_disp = fits.getdata(fits_file_B)
plt.figure(figsize=(10, 10))
plt.imshow(data_B_disp, cmap="gray", origin="lower",
           vmin=np.percentile(data_B_disp, 5), vmax=np.percentile(data_B_disp, 99))
plt.colorbar(label="Counts")
plt.scatter(df_B["X"], df_B["Y"], s=40, edgecolor="blue", facecolor="none",
            label="Measured sources")
plt.title(f"{obj_name} - B band with detected sources")
plt.xlabel("X [pixels]")
plt.ylabel("Y [pixels]")
plt.legend()
out_png_B = os.path.join(_outdir, f"{obj_name}_B_sources.png")
plt.savefig(out_png_B, dpi=150)
plt.close()
print(f"B-band source map saved to {out_png_B}")

# This block loads the calibrated photometry results and the reference stars list
# It compares the X,Y positions of all measured sources with the reference stars
# Any source within a pixel tolerance (REMOVE_TOL) of a reference star is flagged as a star
# Those flagged sources are removed from the calibrated photometry table
# A new cleaned file is saved with the suffix "_calibrated_photometry_no_stars.csv"

# === Input files ===
calib_file = csv_calib
ref_file   = csv_ref

# === Load data ===
df_calib = pd.read_csv(calib_file)
df_ref   = pd.read_csv(ref_file)

# Ensure numeric coords
for col in ["X_B","Y_B","X_V","Y_V"]:
    if col in df_ref.columns:
        df_ref[col] = pd.to_numeric(df_ref[col], errors="coerce")

# === Filter out reference stars ===
mask_remove = []
for i, row in df_calib.iterrows():
    xb, yb = row.get("X_B", np.nan), row.get("Y_B", np.nan)
    xv, yv = row.get("X_V", np.nan), row.get("Y_V", np.nan)

    # check distance to all reference stars
    dB = np.sqrt((df_ref["X_B"] - xb)**2 + (df_ref["Y_B"] - yb)**2)
    dV = np.sqrt((df_ref["X_V"] - xv)**2 + (df_ref["Y_V"] - yv)**2)

    if (dB.min() <= REMOVE_TOL) or (dV.min() <= REMOVE_TOL):
        mask_remove.append(True)
    else:
        mask_remove.append(False)

df_clean = df_calib.loc[~pd.Series(mask_remove)].reset_index(drop=True)

# === Save new file ===
out_file = calib_file.replace("_calibrated_photometry.csv",
                              "_calibrated_photometry_no_stars.csv")
df_clean.to_csv(out_file, index=False)
print(f"Cleaned file saved to {out_file}")


# ----- Create V-band image with all measured sources (already exists above) No stars-----
# {obj_name}_V_sources.png is saved earlier in the code

# ----- Create V-band image with cleaned sources (no catalog stars) -----
plt.figure(figsize=(10, 10))
plt.imshow(data_V, cmap="gray", origin="lower",
           vmin=np.percentile(data_V, 5), vmax=np.percentile(data_V, 99))
plt.colorbar(label="Counts")

# overlay cleaned detections from the filtered catalog
plt.scatter(df_clean["X_V"], df_clean["Y_V"],
            s=40, edgecolor="green", facecolor="none", label="Cleaned sources (no stars)")

plt.title(f"{obj_name} - V band with detected sources (no stars)")
plt.xlabel("X [pixels]")
plt.ylabel("Y [pixels]")
plt.legend()

out_png_clean = os.path.join(_outdir, f"{obj_name}_V_sources_no_stars.png")
plt.savefig(out_png_clean, dpi=150)
plt.close()
print(f"V-band cleaned source map saved to {out_png_clean}")

# Run CMD creation for full calibrated photometry
make_cmd(df_calib, calib_file, "(all sources)")

# Run CMD creation for cleaned catalog (no stars)
make_cmd(df_clean, out_file, "(no stars)")


# ----- Create V-band image and CMD with cleaned sources filtered by color index (B-V range) -----
if "Mag_B" in df_clean.columns and "Mag_V" in df_clean.columns:
    df_clean_color = df_clean.copy()
    df_clean_color["B-V"] = df_clean_color["Mag_B"] - df_clean_color["Mag_V"]

    # filter by CMD_COLOR_MIN / CMD_COLOR_MAX
    mask_color = (df_clean_color["B-V"] >= CMD_COLOR_MIN) & (df_clean_color["B-V"] <= CMD_COLOR_MAX)
    df_color_filtered = df_clean_color[mask_color].reset_index(drop=True)

    # --- save filtered catalog as CSV ---
    out_csv_color = os.path.join(_outdir, f"{obj_name}_calibrated_photometry_no_stars_color_filtered.csv")
    df_color_filtered.to_csv(out_csv_color, index=False)
    print(f"Color-filtered photometry saved to {out_csv_color}")

    # --- create and save galaxy image with color-filtered sources ---
    plt.figure(figsize=(10, 10))
    plt.imshow(data_V, cmap="gray", origin="lower",
               vmin=np.percentile(data_V, 5), vmax=np.percentile(data_V, 99))
    plt.colorbar(label="Counts")

    plt.scatter(df_color_filtered["X_V"], df_color_filtered["Y_V"],
                s=40, edgecolor="green", facecolor="none",
                label=f"Sources in color range ({CMD_COLOR_MIN} ≤ B-V ≤ {CMD_COLOR_MAX})")

    plt.title(f"{obj_name} - V band with color-filtered sources (no stars)")
    plt.xlabel("X [pixels]")
    plt.ylabel("Y [pixels]")
    plt.legend()

    out_png_color = os.path.join(_outdir, f"{obj_name}_V_sources_color_filtered.png")
    plt.savefig(out_png_color, dpi=150)
    plt.close()
    print(f"V-band color-filtered source map saved to {out_png_color}")

    # --- create CMD for the color-filtered catalog ---
    make_cmd(df_color_filtered, out_csv_color, "(no stars, color filtered)")

else:
    print("Mag_B or Mag_V not found in cleaned catalog. Color-filtered map not created.")
    

# ----- Add radial distances in px and pc and create radial profiles -----
if not df_color_filtered.empty:
    # Galaxy center in pixels - the galaxy's own catalogued position when it
    # can be looked up, the frame's geometric centre otherwise.
    x_center, y_center = lookup_center_pixel(obj_name, hdr_V)
    if x_center is None:
        x_center = data_V.shape[1] / 2.0
        y_center = data_V.shape[0] / 2.0

    # Radial distance in pixels. X_V/Y_V, not X_B/Y_B: x_center/y_center are in
    # V's pixel grid (V's own WCS, or V's frame centre), and B's native pixels
    # are not generally the same grid at all - see register_B_onto_V().
    df_color_filtered["Radial_Distance_px"] = np.sqrt(
        (df_color_filtered["X_V"] - x_center)**2 +
        (df_color_filtered["Y_V"] - y_center)**2
    )

    # Pixel scale from WCS (deg/pixel -> rad/pixel)
    wcs_V = WCS(hdr_V)
    pixscale_deg = np.mean(np.abs(wcs_V.pixel_scale_matrix.diagonal()))
    pixscale_rad = np.deg2rad(pixscale_deg)

    # Distance in parsec
    distance_pc = distance * 1.0e6

    # Conversion factor (pc/pixel)
    px_to_pc = pixscale_rad * distance_pc

    # Add distances in pc
    df_color_filtered["Radial_Distance_pc"] = df_color_filtered["Radial_Distance_px"] * px_to_pc

    # Sort by radial distance
    df_color_filtered = df_color_filtered.sort_values("Radial_Distance_px").reset_index(drop=True)

    # Save CSV with all distances
    out_csv_color_rad = os.path.join(
        _outdir,
        f"{obj_name}_calibrated_photometry_no_stars_color_filtered_with_radius.csv"
    )
    df_color_filtered.to_csv(out_csv_color_rad, index=False)
    print(f"Color-filtered photometry with radial distances saved to {out_csv_color_rad}")

    # -----------------------------------------------------------
    # Radial density profile in pixels
    # -----------------------------------------------------------
    bin_width_px = Step_size
    max_r_px = df_color_filtered["Radial_Distance_px"].max()
    bins_px = np.arange(0, max_r_px + bin_width_px, bin_width_px)

    counts_px, edges_px = np.histogram(df_color_filtered["Radial_Distance_px"], bins=bins_px)
    areas_px2 = np.pi * (edges_px[1:]**2 - edges_px[:-1]**2)
    densities_px = counts_px / areas_px2
    bin_centers_px = 0.5 * (edges_px[1:] + edges_px[:-1])

    # Save profile (px)
    df_profile_px = pd.DataFrame({
        "R_inner_px": edges_px[:-1],
        "R_outer_px": edges_px[1:],
        "R_center_px": bin_centers_px,
        "N_sources": counts_px,
        "Annulus_area_px2": areas_px2,
        "Density_per_px2": densities_px
    })
    out_csv_profile_px = os.path.join(_outdir, f"{obj_name}_radial_density_profile.csv")
    df_profile_px.to_csv(out_csv_profile_px, index=False)
    print(f"Radial density profile (px) saved to {out_csv_profile_px}")

    # Step profile (px)
    step_r_px, step_density_px, step_counts_px = [], [], []
    for i in range(len(densities_px)):
        r_in, r_out = edges_px[i], edges_px[i+1]
        step_r_px.extend([r_in, r_out])
        step_density_px.extend([densities_px[i], densities_px[i]])
        step_counts_px.extend([counts_px[i], counts_px[i]])

    df_step_px = pd.DataFrame({
        "R_step_px": step_r_px,
        "Density_step_per_px2": step_density_px,
        "N_sources_step": step_counts_px
    })
    out_csv_step_px = os.path.join(_outdir, f"{obj_name}_radial_density_profile_step.csv")
    df_step_px.to_csv(out_csv_step_px, index=False)
    print(f"Radial density step profile (px) saved to {out_csv_step_px}")

    plt.figure(figsize=(8,6))
    plt.step(step_r_px, step_density_px, where="post", color="blue", linewidth=2)
    plt.xlabel("Radial distance [pixels]")
    plt.ylabel("Source density [1/pixel²]")
    plt.title(f"Radial density step profile of {obj_name} (pixels)")
    out_png_step_px = os.path.join(_outdir, f"{obj_name}_radial_density_profile_step.png")
    plt.savefig(out_png_step_px, dpi=150)
    plt.close()
    print(f"Radial density step plot (px) saved to {out_png_step_px}")

    # -----------------------------------------------------------
    # Radial density profile in pc
    # -----------------------------------------------------------
    bin_width_pc = bin_width_px * px_to_pc
    max_r_pc = df_color_filtered["Radial_Distance_pc"].max()
    bins_pc = np.arange(0, max_r_pc + bin_width_pc, bin_width_pc)

    counts_pc, edges_pc = np.histogram(df_color_filtered["Radial_Distance_pc"], bins=bins_pc)
    areas_pc2 = np.pi * (edges_pc[1:]**2 - edges_pc[:-1]**2)
    densities_pc = counts_pc / areas_pc2
    bin_centers_pc = 0.5 * (edges_pc[1:] + edges_pc[:-1])

    # Save profile (pc)
    df_profile_pc = pd.DataFrame({
        "R_inner_pc": edges_pc[:-1],
        "R_outer_pc": edges_pc[1:],
        "R_center_pc": bin_centers_pc,
        "N_sources": counts_pc,
        "Annulus_area_pc2": areas_pc2,
        "Density_per_pc2": densities_pc
    })
    out_csv_profile_pc = os.path.join(_outdir, f"{obj_name}_radial_density_profile_pc.csv")
    df_profile_pc.to_csv(out_csv_profile_pc, index=False)
    print(f"Radial density profile (pc) saved to {out_csv_profile_pc}")

    # Step profile (pc)
    step_r_pc, step_density_pc, step_counts_pc = [], [], []
    for i in range(len(densities_pc)):
        r_in, r_out = edges_pc[i], edges_pc[i+1]
        step_r_pc.extend([r_in, r_out])
        step_density_pc.extend([densities_pc[i], densities_pc[i]])
        step_counts_pc.extend([counts_pc[i], counts_pc[i]])

    df_step_pc = pd.DataFrame({
        "R_step_pc": step_r_pc,
        "Density_step_per_pc2": step_density_pc,
        "N_sources_step": step_counts_pc
    })
    out_csv_step_pc = os.path.join(_outdir, f"{obj_name}_radial_density_profile_step_pc.csv")
    df_step_pc.to_csv(out_csv_step_pc, index=False)
    print(f"Radial density step profile (pc) saved to {out_csv_step_pc}")

    plt.figure(figsize=(8,6))
    plt.step(step_r_pc, step_density_pc, where="post", color="green", linewidth=2)
    plt.xlabel("Radial distance [pc]")
    plt.ylabel("Source density [1/pc²]")
    plt.title(f"Radial density step profile of {obj_name} (pc)")
    out_png_step_pc = os.path.join(_outdir, f"{obj_name}_radial_density_profile_step_pc.png")
    plt.savefig(out_png_step_pc, dpi=150)
    plt.close()
    print(f"Radial density step plot (pc) saved to {out_png_step_pc}")

else:
    print("No color-filtered sources available to compute radial distances.")
