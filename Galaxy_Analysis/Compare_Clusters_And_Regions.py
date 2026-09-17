"""
Compare_Clusters_And_Regions.py - the two tracers compared, from finished runs.

WHAT IT IS FOR
Star_Formation_All_In_One.py measures both tracers itself and compares them in
one pass. This does only the comparison, from output folders that already exist:
one from Blue_Clusters_From_Images.py and one from HII_From_Images.py. No frames
are read for measurement, nothing is detected, nothing is subtracted.

That makes it useful for two things the full tool cannot do. Old results can be
re-examined - a pair of runs from last year, or a student's, with no need to still
have the frames. And each side can be measured on its own terms: the full tool has
to put every frame on one grid, because it measures both tracers in the same pass,
and that resampling costs something. Here each run kept its own grid and the
comparison is made on the sky instead.

THE PROBLEM THIS HAS TO SOLVE, AND HOW
Two separate runs leave two catalogues in two different pixel grids. The blue
knots are in the V frame of their run; the HII regions are in the H-alpha frame of
theirs. Subtracting one set of pixel coordinates from the other would measure the
offset between two grids, not the distance between two objects - and that offset
is of the same size as the answer.

So no grid is used at all. Both catalogues are converted to sky coordinates
through the plate solution each run wrote into its own FITS file, and the
separation between a knot and a region is measured as an angle. An angle times a
distance is a length, and that is the number wanted. Nothing is resampled, nothing
is shifted, and neither run's grid is preferred over the other's.

WHAT IT NEEDS
    the output folder of Blue_Clusters_From_Images.py
    the output folder of HII_From_Images.py

Each must still contain the catalogue CSV and at least one FITS frame carrying a
plate solution - both tools write those.

USAGE
    python Compare_Clusters_And_Regions.py
    python Compare_Clusters_And_Regions.py --clusters DIR --regions DIR --out DIR
"""



from __future__ import annotations

import os
import sys
import math
import numpy as np
import textwrap
import matplotlib
import matplotlib.pyplot as plt
import argparse
import csv
import glob
import warnings
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u


# --------------------------------------------------------------------------
# shared module interactive_input.py
# --------------------------------------------------------------------------

def nothing_was_passed(argv=None):
    """True when the tool was started with no arguments at all.

    That is the only case where questions are the right response. A tool started
    with some arguments but not all of them has a mistake in it, and argparse
    saying which one is missing is more useful than a questionnaire.
    """
    if argv is None:
        return len(sys.argv) <= 1
    return len(argv) == 0

def normalise_path(text):
    """A path as a person actually pastes it.

    Windows Explorer's "Copy as path" wraps the path in double quotes, and a path
    dragged into a terminal often arrives with a space on the end. Both are meant
    as the same path, so both are accepted.
    """
    return os.path.normpath(text.strip().strip('"').strip("'"))

def ask(prompt, default=None, cast=str):
    """One question. The default is shown in brackets and a bare Enter takes it.

    A question with no default is asked again until it is answered, and an answer
    that will not convert is refused on the spot rather than three steps later,
    where the reason would no longer be obvious.
    """
    while True:
        shown = ("{} [{}]: ".format(prompt, default) if default is not None
                 else "{}: ".format(prompt))
        text = input(shown).strip()
        if not text:
            if default is None:
                print("    this one has no default - please answer it")
                continue
            # The default goes through the same conversion as a typed answer.
            # Returning it raw looks harmless because most defaults are strings,
            # but ask("Which one", "1", int) then hands back "1" and the caller
            # compares a string with a number several lines later - which is
            # exactly how a bare Enter on the frame-choice prompt crashed a run.
            try:
                return cast(default)
            except (TypeError, ValueError):
                return default
        try:
            return cast(text)
        except (TypeError, ValueError):
            print("    that is not a {} - try again".format(
                getattr(cast, "__name__", "valid value")))

def ask_file(prompt):
    """A path to a file that exists, asked again until it does.

    Checking here is worth the two lines: the alternative is a run that reads three
    frames, spends a minute on the fourth, and then dies on a typo in the first.
    """
    while True:
        path = normalise_path(ask(prompt))
        if os.path.isfile(path):
            return path
        print("    there is no file at " + path)

def ask_folder(prompt):
    """A path to a folder that exists, asked again until it does."""
    while True:
        path = normalise_path(ask(prompt))
        if os.path.isdir(path):
            return path
        print("    there is no folder at " + path)

def ask_optional_float(prompt, blank_note="blank to skip"):
    """A number, or nothing. Returns None when the answer is left blank.

    What blank means is the caller's business, so the caller says it. Blank used
    to mean "do without", and now it usually means "go and look it up" - a prompt
    that still promised the first would be lying about what the tool does.
    """
    while True:
        text = input("{} [{}]: ".format(prompt, blank_note)).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            print("    that is not a number - try again, or press Enter to skip")

def ask_yes_no(prompt, default=False):
    """A yes or no question. Anything starting with y or n counts."""
    hint = "Y/n" if default else "y/N"
    while True:
        text = input("{} [{}]: ".format(prompt, hint)).strip().lower()
        if not text:
            return default
        if text[0] in "yn":
            return text[0] == "y"
        print("    please answer yes or no")


# --------------------------------------------------------------------------
# shared module deprojection.py
# --------------------------------------------------------------------------

INC_MIN_DEG = 30.0

INC_MAX_DEG = 70.0

Q0_DEFAULT = 0.13

Q0_BY_TYPE = {"early": 0.20, "mid": 0.13, "late": 0.10}

def q0_for_type(mtype: str | None) -> float:
    """Pick a disk-thickness constant from the morphological type string.

    Falls back to the mid-type value when the type is unknown or unparsable,
    which is the safe choice: it sits between the two extremes.
    """
    if not mtype:
        return Q0_DEFAULT
    t = str(mtype).strip().upper()
    if t.startswith(("S0", "SA", "E")):
        return Q0_BY_TYPE["early"]
    if t.startswith(("SD", "SM", "IM", "I")):
        return Q0_BY_TYPE["late"]
    return Q0_BY_TYPE["mid"]

def inclination_from_axis_ratio(b_over_a: float, q0: float = Q0_DEFAULT) -> float:
    """Inclination in degrees from the observed minor/major axis ratio.

    Uses the Hubble formula, which accounts for the disk having real thickness:

        cos^2(i) = (q^2 - q0^2) / (1 - q0^2)

    At moderate inclinations this barely differs from the naive acos(b/a). At high
    inclination it matters a lot: for q = 0.2 the naive value is 78.5 deg while this
    gives 81.2 deg, which changes the stretch factor from 5.0 to 6.5.

    Returns 90.0 when the galaxy is thinner than q0, i.e. seen edge-on.
    """
    q = float(b_over_a)
    if q <= q0:
        return 90.0
    cos2 = (q * q - q0 * q0) / (1.0 - q0 * q0)
    cos2 = min(max(cos2, 0.0), 1.0)
    return math.degrees(math.acos(math.sqrt(cos2)))

def stretch_factor(inc_deg: float) -> float:
    """How much the minor-axis direction has to be stretched: 1 / cos(i)."""
    return 1.0 / math.cos(math.radians(inc_deg))

def inclination_is_usable(inc_deg: float) -> tuple[bool, str]:
    """Decide whether deprojecting at this inclination is worth doing.

    Returns (ok, reason). The reason is meant to be printed and to end up in the
    plot, so that a skipped correction is visible rather than silent.
    """
    if inc_deg < INC_MIN_DEG:
        return False, (
            f"inclination {inc_deg:.1f} deg is below {INC_MIN_DEG:.0f} deg - the galaxy is "
            "close to face-on, the correction would be smaller than its own uncertainty, "
            "and the position angle is poorly defined. Not corrected."
        )
    if inc_deg > INC_MAX_DEG:
        return False, (
            f"inclination {inc_deg:.1f} deg is above {INC_MAX_DEG:.0f} deg - the correction "
            "would be large but unreliable, and dust biases it systematically. Not corrected."
        )
    return True, f"inclination {inc_deg:.1f} deg is inside the usable window."

def sky_pa_to_image_angle(pa_sky_deg: float,
                          north_angle_deg: float = 90.0,
                          mirrored: bool = False) -> float:
    """Convert a catalogue position angle into an angle in image pixel coordinates.

    Catalogues give the position angle measured on the sky, from North towards East.
    Our object positions are in image pixels, where the major axis has some angle
    measured from the +x axis. These are not the same number and confusing them
    rotates the whole deprojection.

        north_angle_deg  angle of North in the image, counterclockwise from +x.
                         90 for the usual "north up" orientation.
        mirrored         True if the image parity is flipped, so that East runs
                         clockwise from North instead of counterclockwise.

    If you already know the major-axis angle directly in image coordinates, skip
    this function and pass that angle straight to deproject().
    """
    sign = -1.0 if mirrored else 1.0
    return (north_angle_deg + sign * pa_sky_deg) % 180.0

def deproject(dx, dy, major_axis_angle_deg: float, inc_deg: float):
    """Stretch centred positions from the plane of the sky into the galaxy plane.

    dx, dy must already be measured FROM THE GALAXY CENTRE. This matters: the
    transform is linear, so applying it to uncentred coordinates leaves the shape
    correct but moves the centre, and every radius measured from the old centre
    then comes out wrong.

    major_axis_angle_deg is measured in the same frame as dx, dy - counterclockwise
    from the +x axis. Use sky_pa_to_image_angle() if you are starting from a
    catalogue position angle.

    The transform rotates the major axis onto x, stretches the perpendicular
    direction by 1/cos(i), and rotates back. The resulting matrix is symmetric,
    as a pure stretch along an axis must be.
    """
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)

    th = math.radians(major_axis_angle_deg)
    c, s = math.cos(th), math.sin(th)
    k = stretch_factor(inc_deg)

    m11 = c * c + k * s * s
    m12 = (1.0 - k) * s * c
    m22 = s * s + k * c * c

    return dx * m11 + dy * m12, dx * m12 + dy * m22

def effective_ring_areas(edges,
                         frame_corners,
                         centre_xy,
                         major_axis_angle_deg: float | None,
                         inc_deg: float | None,
                         n_samples: int = 400_000,
                         rng_seed: int = 12345):
    """Area of each annulus that actually falls inside the imaged frame.

    WHY THIS EXISTS
    A ring drawn around the galaxy centre is only partly covered by the image: at
    large radii the corners of the frame cut it. Dividing counts by the full
    geometric ring area therefore makes the outer bins look emptier than they are,
    and produces a radial decline that is pure geometry.

    This is not hypothetical. In M33 a catalogue covering only the central 23'x23'
    produced "an artificially large decrease at radii larger than the largest circle
    inscribed in the area", and the published exponential slope was wrong by a
    factor of three until a size cut and a radial truncation were applied.

    METHOD
    Scatter random points uniformly over the frame, put them through exactly the
    same deprojection as the real objects, and histogram their radii. The fraction
    landing in each ring, times the frame area, is that ring's effective area.
    This handles any frame shape and any deprojection without special-casing.

    frame_corners is (xmin, xmax, ymin, ymax) in the same pixel units as the data.
    Pass major_axis_angle_deg=None to skip deprojection (the uncorrected case).
    """
    xmin, xmax, ymin, ymax = frame_corners
    rng = np.random.default_rng(rng_seed)

    sx = rng.uniform(xmin, xmax, n_samples)
    sy = rng.uniform(ymin, ymax, n_samples)
    frame_area = (xmax - xmin) * (ymax - ymin)

    sdx = sx - centre_xy[0]
    sdy = sy - centre_xy[1]
    if major_axis_angle_deg is not None and inc_deg is not None:
        sdx, sdy = deproject(sdx, sdy, major_axis_angle_deg, inc_deg)

    sr = np.hypot(sdx, sdy)
    counts, _ = np.histogram(sr, bins=edges)
    return counts / float(n_samples) * frame_area

def radial_density(dx, dy,
                   n_rings: int = 20,
                   r_max: float | None = None,
                   frame_corners=None,
                   centre_xy=None,
                   major_axis_angle_deg: float | None = None,
                   inc_deg: float | None = None,
                   min_area_fraction: float = 0.15):
    """Surface density of objects in equal-width concentric rings.

    Equal-width rings out to the outermost object, following the standard practice
    in this literature (20 rings, each 0.05 of the maximum radius).

    Rings whose effective area is a small fraction of their full geometric area are
    marked incomplete rather than plotted, because their density is dominated by how
    the frame happens to cut them. min_area_fraction sets that threshold.

    Returns a dict with ring centres, counts, areas, densities and Poisson errors.
    """
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    r = np.hypot(dx, dy)

    if r_max is None:
        r_max = float(r.max()) if r.size else 1.0
    edges = np.linspace(0.0, r_max, n_rings + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    counts, _ = np.histogram(r, bins=edges)
    geometric = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)

    if frame_corners is not None and centre_xy is not None:
        area = effective_ring_areas(edges, frame_corners, centre_xy,
                                    major_axis_angle_deg, inc_deg)
    else:
        area = geometric

    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.where(area > 0, counts / area, np.nan)
        err = np.where(area > 0, np.sqrt(np.maximum(counts, 0)) / area, np.nan)
        frac = np.where(geometric > 0, area / geometric, 0.0)

    complete = frac >= min_area_fraction

    return {
        "edges": edges,
        "r": centres,
        "counts": counts,
        "area": area,
        "area_fraction": frac,
        "density": density,
        "density_err": err,
        "complete": complete,
        "r_max": r_max,
    }

def sector_dispersion(dx, dy, major_axis_angle_deg, inc_deg, n_sectors=20):
    """How uneven the object counts are once the field is cut into equal wedges.

    THE IDEA
    Slice the galaxy into equal sectors, like slices of a pie. If the disk really is
    round and we are looking straight down on it, every slice holds roughly the same
    number of objects. Seen at an angle it does not. So the tilt that evens the
    slices out is the tilt that undoes the projection.

    Opposite slices are added together before the dispersion is measured. A galaxy
    that is lopsided - a brighter arm on one side, a companion pulling at it - would
    otherwise push the answer around; adding a slice to the one facing it cancels
    that kind of asymmetry.

    Returns std / mean of the folded counts. Smaller is better.
    """
    dxp, dyp = deproject(dx, dy, major_axis_angle_deg, inc_deg)
    theta = np.arctan2(dyp, dxp)
    idx = np.floor((theta % (2.0 * math.pi)) / (2.0 * math.pi) * n_sectors).astype(int)
    idx = np.clip(idx, 0, n_sectors - 1)
    counts = np.bincount(idx, minlength=n_sectors).astype(float)

    half = n_sectors // 2
    folded = counts[:half] + counts[half:]
    mean = folded.mean()
    if mean <= 0:
        return np.inf
    return float(folded.std() / mean)

def fit_sector_method(dx, dy,
                      n_sectors=20,
                      angle_step=5.0,
                      inc_step=5.0,
                      inc_range=(20.0, 80.0),
                      r_max=None):
    """Search for the deprojection angles that make the object counts most even.

    Scans a grid of position angle and inclination and returns the pair that
    minimises sector_dispersion(). This is the second of the two methods introduced
    by Garcia-Gomez and Athanassoula, and in their comparison it is the one that
    agrees most closely with kinematic determinations - 0.96 in position angle and
    0.90 in cos(inclination), better than their Fourier method and better than
    photometry.

    WHEN IT FAILS - read this before trusting the answer
      * Below roughly 150-200 objects the surface being minimised often has several
        minima rather than one, and the deepest is not always the right one. The
        returned dict reports the runners-up so you can look.
      * If the objects trace the spiral arms or a bar instead of filling the disk,
        the method circularises the arms or the bar rather than the disk. In the
        published notes this is exactly what went wrong for several barred galaxies.
      * A ring of objects is circularised as if it were the disk. An outer ring with
        an intrinsic axis ratio of 0.87 fakes an inclination of about 12 degrees in a
        galaxy that is really face-on.
      * It works better the more inclined the galaxy is, and poorly near face-on,
        where the position angle is barely defined in the first place.

    r_max optionally restricts the fit to objects inside that radius, which helps
    when a few far-flung objects dominate the outer sectors.
    """
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)

    if r_max is not None:
        keep = np.hypot(dx, dy) <= r_max
        dx, dy = dx[keep], dy[keep]

    angles = np.arange(0.0, 180.0, angle_step)
    incs = np.arange(inc_range[0], inc_range[1] + 1e-9, inc_step)

    grid = np.full((len(incs), len(angles)), np.inf)
    for i, inc in enumerate(incs):
        for j, ang in enumerate(angles):
            grid[i, j] = sector_dispersion(dx, dy, float(ang), float(inc), n_sectors)

    flat = np.argsort(grid, axis=None)
    best_i, best_j = np.unravel_index(flat[0], grid.shape)

    # Report distinct runners-up: local minima well separated from the best one, so
    # that a shallow global minimum does not hide a competing solution.
    runners = []
    for k in flat[1:]:
        i, j = np.unravel_index(k, grid.shape)
        sep_ang = min(abs(angles[j] - angles[best_j]),
                      180.0 - abs(angles[j] - angles[best_j]))
        if sep_ang < 20.0 and abs(incs[i] - incs[best_i]) < 15.0:
            continue
        if any(min(abs(angles[j] - a), 180.0 - abs(angles[j] - a)) < 20.0
               and abs(incs[i] - c) < 15.0 for a, c, _ in runners):
            continue
        runners.append((float(angles[j]), float(incs[i]), float(grid[i, j])))
        if len(runners) >= 3:
            break

    return {
        "angle": float(angles[best_j]),
        "inc": float(incs[best_i]),
        "dispersion": float(grid[best_i, best_j]),
        "n_objects": int(dx.size),
        "n_sectors": n_sectors,
        "angles": angles,
        "incs": incs,
        "grid": grid,
        "runners_up": runners,
        "few_objects": bool(dx.size < 200),
    }

def distribution_shape(dx, dy):
    """The axis ratio and position angle of a set of positions.

    From the second moments: the shape of the cloud of points, not of any one of
    them. Returns (axis_ratio, angle_deg), the angle measured from +x the same way
    the deprojection measures it.
    """
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    ok = np.isfinite(dx) & np.isfinite(dy)
    if ok.sum() < 20:
        return float("nan"), float("nan")
    cov = np.cov(dx[ok], dy[ok])
    values, vectors = np.linalg.eigh(cov)
    if values[1] <= 0:
        return float("nan"), float("nan")
    ratio = float(np.sqrt(max(values[0], 0.0) / values[1]))
    angle = float(np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1])) % 180.0)
    return ratio, angle

def check_against_catalogue(dx, dy, expected_ratio, expected_angle,
                            ratio_tolerance=0.15, angle_tolerance=25.0):
    """Compare the shape of what was detected with what the catalogue predicts.

    WHY THIS IS WORTH DOING
    A tilted disk seen from here is an ellipse, and the catalogue says which
    ellipse. So the tracers detected in it have a shape that can be predicted
    before they are measured - and comparing the two costs nothing and needs no
    extra data.

    It is a real test because it can fail. On a set of M33 frames the tool
    detected 2844 regions and drew a smooth, entirely plausible radial profile;
    the detections came out round, axis ratio 0.96, while the catalogue says 0.59.
    They were not tracing the galaxy at all - they were subtraction residuals
    spread evenly over the field - and nothing else in the output showed it.

    When the two agree it is evidence the detection is finding the galaxy. When
    they disagree, something is wrong upstream and the profile below should not be
    believed. The check does not say which of the two is at fault.

    A galaxy too round for its own position angle to be defined is not tested on
    angle, for the same reason the catalogue declines to publish one.
    """
    ratio, angle = distribution_shape(dx, dy)
    if not np.isfinite(ratio):
        return {"ok": None, "reason": "too few positions to measure a shape"}

    out = {"measured_ratio": ratio, "measured_angle": angle,
           "expected_ratio": expected_ratio, "expected_angle": expected_angle}

    problems = []
    if expected_ratio is not None and np.isfinite(expected_ratio):
        if abs(ratio - expected_ratio) > ratio_tolerance:
            problems.append(
                "the detections are shaped {:.2f} where the catalogue says {:.2f}"
                .format(ratio, expected_ratio))
    # A round galaxy has no meaningful major axis, so its angle is not tested.
    if (expected_angle is not None and np.isfinite(expected_angle)
            and expected_ratio is not None and expected_ratio < 0.85):
        gap = abs((angle - expected_angle + 90.0) % 180.0 - 90.0)
        out["angle_gap"] = gap
        if gap > angle_tolerance:
            problems.append(
                "their long axis lies at {:.0f} deg where the catalogue says {:.0f}"
                .format(angle, expected_angle))

    out["ok"] = not problems
    out["problems"] = problems
    return out

def describe_check(result):
    """The self-check, in a form worth printing."""
    if result.get("ok") is None:
        return "  shape check: " + result.get("reason", "not done")
    head = ("  shape check: the detections are shaped {:.2f} with their long axis "
            "at {:.0f} deg".format(result["measured_ratio"], result["measured_angle"]))
    if result.get("expected_ratio") is not None:
        head += "; the catalogue predicts {:.2f}".format(result["expected_ratio"])
        if result.get("expected_angle") is not None:
            head += " at {:.0f} deg".format(result["expected_angle"])
    if result["ok"]:
        return head + "\n    they agree - the detections do trace this galaxy"
    return (head + "\n    THEY DISAGREE: " + "; ".join(result["problems"]) +
            "\n    Something upstream is wrong. Detections that do not follow the "
            "galaxy's\n    shape are usually subtraction residuals spread over the "
            "whole field, and\n    the radial profile below will look smooth and "
            "mean nothing.")


# --------------------------------------------------------------------------
# shared module galaxy_catalogue.py
# --------------------------------------------------------------------------

pass  # from deprojection, inlined above

HYPERLEDA_VIZIER_ID = "VII/237"

def hyperleda_geometry(galaxy_name: str, search_radius_arcmin: float = 0.5) -> dict:
    """Fetch position angle, axis ratio and inclination for one galaxy.

    Returns a dict with:
        name_query, pgc, mtype  - identification
        log_r25, b_over_a       - the observed axis ratio
        pa_sky                  - position angle on the sky, or None if not published
        inc_naive               - acos(b/a), ignoring disk thickness
        inc                     - the Hubble-corrected inclination we actually use
        q0                      - the thickness constant chosen for this type
        d25_arcmin              - isophotal diameter, useful for normalising radii
        source                  - human-readable provenance

    Raises LookupError when the name resolves to nothing, so the caller can fall
    back to manually supplied angles instead of silently continuing.
    """
    from astroquery.vizier import Vizier
    import astropy.units as u

    v = Vizier(catalog=HYPERLEDA_VIZIER_ID, row_limit=-1)
    result = v.query_object(galaxy_name, radius=search_radius_arcmin * u.arcmin)
    if not result or len(result[0]) == 0:
        raise LookupError(
            f"HyperLEDA returned no row for '{galaxy_name}'. Check the name, or "
            "supply the angles manually."
        )

    row = result[0][0]

    def _get(col):
        try:
            val = row[col]
        except KeyError:
            return None
        try:
            if hasattr(val, "mask") and val.mask:
                return None
        except Exception:
            pass
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(f) else f

    log_r25 = _get("logR25")
    if log_r25 is None:
        raise LookupError(
            f"HyperLEDA has a row for '{galaxy_name}' but no axis ratio (logR25), "
            "so the inclination cannot be derived. Supply the angles manually."
        )

    b_over_a = 10.0 ** (-log_r25)
    mtype = None
    try:
        mtype = str(row["MType"]).strip() or None
    except KeyError:
        pass

    q0 = q0_for_type(mtype)
    log_d25 = _get("logD25")

    return {
        "name_query": galaxy_name,
        "pgc": _get("PGC"),
        "mtype": mtype,
        "log_r25": log_r25,
        "b_over_a": b_over_a,
        "pa_sky": _get("PA"),
        "inc_naive": math.degrees(math.acos(min(max(b_over_a, 0.0), 1.0))),
        "inc": inclination_from_axis_ratio(b_over_a, q0),
        "q0": q0,
        "d25_arcmin": (0.1 * 10.0 ** log_d25) if log_d25 is not None else None,
        "source": f"HyperLEDA via VizieR {HYPERLEDA_VIZIER_ID}",
    }

def describe(geom: dict) -> str:
    """One readable block summarising what the catalogue gave us."""
    lines = [
        f"  catalogue      : {geom['source']}",
        f"  PGC            : {geom['pgc']}",
        f"  type           : {geom['mtype']}",
        f"  b/a            : {geom['b_over_a']:.3f}   (logR25 = {geom['log_r25']:.3f})",
        f"  inclination    : {geom['inc']:.1f} deg   "
        f"(naive acos(b/a) = {geom['inc_naive']:.1f} deg, q0 = {geom['q0']:.2f})",
    ]
    if geom["pa_sky"] is None:
        lines.append("  position angle : NOT PUBLISHED - the galaxy is too round for the "
                     "major axis to be defined")
    else:
        lines.append(f"  position angle : {geom['pa_sky']:.1f} deg on the sky (N through E)")
    if geom["d25_arcmin"]:
        lines.append(f"  D25            : {geom['d25_arcmin']:.1f} arcmin")
    return "\n".join(lines)

R_V = 3.1                          # A_V / E(B-V) for dust in our own galaxy

DISTANCE_CATALOG = "J/AJ/152/50"   # Cosmicflows-3, Tully et al. 2016

def lookup_distance(name):
    """Redshift-independent distance in Mpc from Cosmicflows-3, or None.

    Every length this tool reports is this number times an angle, so a wrong
    distance rescales the whole answer and nothing in the output looks amiss.
    Redshift is no help for a galaxy this close - the recession velocity of a
    few-Mpc galaxy is dominated by its own motion through the local group, and
    M101's puts it at about half its true distance - so a catalogue of direct
    measurements is what is wanted.

    It is offered, never imposed: the caller is expected to show it and let the
    observer overrule it.
    """
    try:
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier
        import astropy.units as u

        coord = SkyCoord.from_name(name)
        hit = Vizier(columns=["**"], row_limit=5).query_region(
            coord, radius=2 * u.arcmin, catalog=DISTANCE_CATALOG)
        if not hit:
            return None
        table = hit[0]
        for col in ("Dist", "<Dist>"):
            if col in table.colnames:
                value = float(table[col][0])
                if value == value and value > 0:
                    return value
    except Exception:
        return None
    return None

def lookup_extinction(name):
    """Galactic reddening E(B-V) towards the galaxy, or None.

    The Schlafly & Finkbeiner (2011) recalibration of the Schlegel dust maps,
    served by IRSA. Needed wherever a colour is turned into a statement about a
    stellar population: dust in our own galaxy reddens everything behind it, and
    without the correction every source is shifted towards the red, which moves
    the line between a young blue cluster and an ordinary star.

    Where it cancels, and this is worth knowing so it does not get "fixed" later:
    fitting the continuum subtraction factor against an instrumental colour
    measured from the same pair of frames. Reddening shifts every star by the same
    amount there, the fitted line shifts with them, and the value read off it for
    any given pixel is unchanged.
    """
    try:
        from astroquery.ipac.irsa.irsa_dust import IrsaDust
        table = IrsaDust.get_query_table(name, section="ebv")
        value = float(table["ext SandF mean"][0])
        if value == value and value >= 0:
            return value
    except Exception:
        return None
    return None

def describe_lookups(name, distance_mpc, ebv, distance_source="", ebv_source=""):
    """What was looked up, shown before it is used."""
    lines = []
    if distance_mpc:
        lines.append("  distance    {:.2f} Mpc   {}".format(
            distance_mpc, distance_source or "from Cosmicflows-3 (Tully et al. 2016)"))
        lines.append("              every length below scales with this number")
    else:
        lines.append("  distance    not found - radii will be in arcsec, not parsecs")
    if ebv is not None:
        lines.append("  reddening   E(B-V) = {:.4f}, so A_V = {:.3f}   {}".format(
            ebv, R_V * ebv,
            ebv_source or "from Schlafly & Finkbeiner (2011)"))
    return "\n".join(lines)

def gaia_foreground_stars(wcs, shape, g_limit=19.0, significance=3.0,
                          max_rows=50000):
    """Positions, in pixels, of stars that are in front of the galaxy.

    WHY A CATALOGUE AND NOT THE IMAGE
    Everything downstream that needs "a star" needs a foreground star: something
    whose light is pure continuum, so that scaling one filter against another and
    subtracting makes it vanish. Detecting point sources in the image finds those,
    but in a nearby galaxy it also finds the galaxy's own stars - and those are not
    pure continuum. Some are red giants, some are hot young stars sitting in the
    gas that is being measured.

    How badly this matters depends on distance, which is why it went unnoticed for
    a long time. At a few Mpc a galaxy's own stars are not resolved and every point
    source really is foreground. At M33's 0.9 Mpc they are resolved in their
    thousands: measured in this field, of 4209 Gaia sources brighter than G=19,
    only 1590 are foreground. Setting the subtraction factor from the mixture gave
    a star-to-star scatter of 66 percent - which is to say no factor at all.

    THE TEST
    Parallax and proper motion. A star a few hundred parsecs away in our own
    galaxy has both, measurably. A star at 0.9 Mpc has neither, at any precision
    Gaia can reach. So a significant parallax, or a significant proper motion, is
    a physical statement that the star is in front - not a brightness cut or a
    shape cut that happens to correlate with it.

    Returns (positions, diagnostics). Positions is an (N, 2) array of x, y in the
    frame; an empty array means the query failed or found nothing, and the caller
    is expected to fall back on detecting point sources.
    """
    import numpy as np

    diag = {"source": "Gaia DR3", "n_catalogue": 0, "n_foreground": 0,
            "n_in_frame": 0, "why": ""}
    if wcs is None:
        diag["why"] = "no plate solution, so the catalogue cannot be placed"
        return np.empty((0, 2)), diag

    try:
        from astroquery.gaia import Gaia
        import astropy.units as u

        ny, nx = shape[0], shape[1]
        centre = wcs.pixel_to_world(nx / 2.0, ny / 2.0)
        corner = wcs.pixel_to_world(0.0, 0.0)
        radius = float(centre.separation(corner).to(u.deg).value) * 1.05

        query = (
            "SELECT ra, dec, phot_g_mean_mag FROM gaiadr3.gaia_source "
            "WHERE 1=CONTAINS(POINT(ra, dec), CIRCLE({:.6f}, {:.6f}, {:.4f})) "
            "AND phot_g_mean_mag < {:.2f} "
            "AND (parallax/parallax_error > {:.1f} OR "
            "     sqrt(pmra*pmra + pmdec*pmdec) / "
            "     sqrt(pmra_error*pmra_error + pmdec_error*pmdec_error) > {:.1f}) "
            "ORDER BY phot_g_mean_mag"
        ).format(centre.ra.deg, centre.dec.deg, radius, g_limit,
                 significance, significance)

        job = Gaia.launch_job_async(query) if max_rows > 2000 else Gaia.launch_job(query)
        table = job.get_results()
        diag["n_foreground"] = len(table)
        if not len(table):
            diag["why"] = "no foreground stars in the field"
            return np.empty((0, 2)), diag

        x, y = wcs.world_to_pixel_values(np.asarray(table["ra"], dtype=float),
                                         np.asarray(table["dec"], dtype=float))
        inside = (x > 5) & (x < nx - 6) & (y > 5) & (y < ny - 6)
        diag["n_in_frame"] = int(inside.sum())
        # Brightest first. Whoever takes "the first two hundred" of this list must
        # get the two hundred worth measuring: a star near the catalogue limit is
        # a few counts through a narrow filter, and a ratio built on it is noise.
        # Handing back an unordered list cost a factor of two in the scatter of the
        # subtraction factor before this line existed.
        diag["g_range"] = (float(np.nanmin(table["phot_g_mean_mag"])),
                           float(np.nanmax(table["phot_g_mean_mag"])))
        return np.column_stack([x[inside], y[inside]]), diag

    except Exception as exc:
        diag["why"] = "{}: {}".format(type(exc).__name__, exc)
        return np.empty((0, 2)), diag

def describe_foreground(diag):
    """What the catalogue lookup found, in a line."""
    if diag.get("n_in_frame"):
        return ("  {} foreground stars from {} (parallax or proper motion "
                "significant); the galaxy's own stars are not among them".format(
                    diag["n_in_frame"], diag["source"]))
    return ("  no catalogue stars available ({}) - falling back on point sources "
            "found in the image, which in a nearby galaxy includes its own "
            "stars".format(diag.get("why") or "unknown"))

describe__galaxy_catalogue = describe   # this module's own; the plain name gets shadowed below


# --------------------------------------------------------------------------
# shared module tracer_matching.py
# --------------------------------------------------------------------------

def nearest_neighbour(knots_xy, regions_xy):
    """For every blue knot, the nearest HII region and the distance to it.

    Returns (index_of_nearest, distance). Both arrays are as long as knots_xy.
    A knot whose nearest region is the same one as another knot's is perfectly
    normal - one HII complex can produce several clusters, and the student
    projects saw exactly that.
    """
    knots_xy = np.asarray(knots_xy, dtype=float)
    regions_xy = np.asarray(regions_xy, dtype=float)
    if knots_xy.size == 0 or regions_xy.size == 0:
        return np.empty(0, dtype=int), np.empty(0)

    d = np.hypot(knots_xy[:, None, 0] - regions_xy[None, :, 0],
                 knots_xy[:, None, 1] - regions_xy[None, :, 1])
    idx = np.argmin(d, axis=1)
    return idx, d[np.arange(len(idx)), idx]

def local_surface_density(radii, profile_r, profile_density):
    """The HII surface density at each knot's radius, read off the measured profile.

    Linear interpolation, held flat outside the measured range rather than
    extrapolated - an extrapolated density would drive the null model somewhere the
    data never went.
    """
    good = np.isfinite(profile_density) & (profile_density > 0)
    if good.sum() < 2:
        return np.full(np.shape(radii), np.nan)
    return np.interp(radii, np.asarray(profile_r)[good],
                     np.asarray(profile_density)[good],
                     left=profile_density[good][0], right=profile_density[good][-1])

def expected_separation(surface_density):
    """Mean distance to the nearest point in a random field of this density.

    For points scattered at random with surface density n, the distance to the
    nearest one averages 1 / (2 * sqrt(n)). This is the separation you would measure
    from a galaxy in which nothing had drifted at all, and it is the line the real
    measurement has to beat before any drift can be claimed.
    """
    n = np.asarray(surface_density, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 0, 0.5 / np.sqrt(n), np.nan)

def match_report(knots_xy, regions_xy, centre, profile_r, profile_density,
                 scale=1.0, unit="px"):
    """Everything the separation plot needs, measured and expected side by side.

    profile_r and profile_density must be the HII radial profile in the SAME units
    the scale converts to, because the null model is built from them.
    """
    knots_xy = np.asarray(knots_xy, dtype=float)
    regions_xy = np.asarray(regions_xy, dtype=float)

    idx, sep_px = nearest_neighbour(knots_xy, regions_xy)
    r_px = np.hypot(knots_xy[:, 0] - centre[0], knots_xy[:, 1] - centre[1])

    r_u = r_px * scale
    sep_u = sep_px * scale

    n_local = local_surface_density(r_u, profile_r, profile_density)
    sep_expected = expected_separation(n_local)

    with np.errstate(divide="ignore", invalid="ignore"):
        excess = np.where(np.isfinite(sep_expected) & (sep_expected > 0),
                          sep_u / sep_expected, np.nan)

    out = {
        "nearest_index": idx,
        "separation": sep_u,
        "radius": r_u,
        "expected": sep_expected,
        "excess_ratio": excess,
        "unit": unit,
        "n_knots": int(knots_xy.shape[0]),
        "n_regions": int(regions_xy.shape[0]),
        "mean_separation": float(np.nanmean(sep_u)) if sep_u.size else float("nan"),
        "median_separation": float(np.nanmedian(sep_u)) if sep_u.size else float("nan"),
        "mean_expected": float(np.nanmean(sep_expected)) if sep_u.size else float("nan"),
    }

    # How many HII regions ended up as somebody's nearest neighbour, and how often
    # the same one was chosen. Several knots sharing a parent region is expected.
    if idx.size:
        counts = np.bincount(idx, minlength=regions_xy.shape[0])
        out["regions_used"] = int((counts > 0).sum())
        out["max_knots_per_region"] = int(counts.max())
    else:
        out["regions_used"] = 0
        out["max_knots_per_region"] = 0

    # A straight line through separation against radius, the way the student
    # projects reported it, plus the same line for the null model. If the two
    # slopes agree, the trend is telling you about density and not about drift.
    ok = np.isfinite(r_u) & np.isfinite(sep_u)
    if ok.sum() >= 3:
        p = np.polyfit(r_u[ok], sep_u[ok], 1)
        out["fit_slope"], out["fit_intercept"] = float(p[0]), float(p[1])
        resid = sep_u[ok] - np.polyval(p, r_u[ok])
        ss_tot = float(np.sum((sep_u[ok] - sep_u[ok].mean()) ** 2))
        out["fit_r2"] = float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    else:
        out["fit_slope"] = out["fit_intercept"] = out["fit_r2"] = float("nan")

    ok2 = ok & np.isfinite(sep_expected)
    if ok2.sum() >= 3:
        q = np.polyfit(r_u[ok2], sep_expected[ok2], 1)
        out["null_slope"], out["null_intercept"] = float(q[0]), float(q[1])
    else:
        out["null_slope"] = out["null_intercept"] = float("nan")

    return out

def describe(rep):
    """A readable summary, including the comparison that matters."""
    u = rep["unit"]
    lines = [
        "  {} blue knots matched to {} HII regions".format(rep["n_knots"], rep["n_regions"]),
        "  {} regions were somebody's nearest; the busiest parented {} knots".format(
            rep["regions_used"], rep["max_knots_per_region"]),
        "  mean separation   {:.1f} {}   (median {:.1f})".format(
            rep["mean_separation"], u, rep["median_separation"]),
        "  expected from density alone   {:.1f} {}".format(rep["mean_expected"], u),
    ]
    if np.isfinite(rep["mean_expected"]) and rep["mean_expected"] > 0:
        lines.append("  ratio measured / expected   {:.2f}".format(
            rep["mean_separation"] / rep["mean_expected"]))
    lines.append("  separation against radius: slope {:.4f}, R2 = {:.3f}".format(
        rep["fit_slope"], rep["fit_r2"]))
    lines.append("  the same slope for a random field of the same density: {:.4f}".format(
        rep["null_slope"]))
    if np.isfinite(rep["fit_slope"]) and np.isfinite(rep["null_slope"]):
        if rep["null_slope"] != 0 and abs(rep["fit_slope"] / rep["null_slope"] - 1.0) < 0.35:
            lines.append("  -> the measured trend is close to what thinning HII regions "
                         "alone would produce. Do not read it as drift without more.")
        elif rep["fit_slope"] > rep["null_slope"]:
            lines.append("  -> the measured trend is steeper than density alone explains.")
        else:
            lines.append("  -> the measured trend is shallower than density alone "
                         "would predict.")
    return "\n".join(lines)

describe__tracer_matching = describe   # this module's own; the plain name gets shadowed below


# --------------------------------------------------------------------------
# shared module radial_profiles.py
# --------------------------------------------------------------------------

matplotlib.use("Agg")

pass  # from deprojection, inlined above

COLOURS = ["#444444", "#a3283c", "#2b5a89", "#3f7d52", "#8a5a2b"]

def treatment_none():
    return {"key": "none", "label": "As measured, no tilt correction", "angle": None, "inc": None,
            "note": "Positions exactly as measured on the image. Control panel."}

def treatment_catalogue(geom, north_angle=90.0, mirrored=False):
    """Deprojection using the position angle and axis ratio from HyperLEDA."""
    if geom is None:
        return None
    pa_sky, inc = geom.get("pa_sky"), geom.get("inc")
    if pa_sky is None:
        return {"key": "leda", "label": "Tilt correction not possible", "angle": None, "inc": None,
                "note": "HyperLEDA publishes no position angle for this galaxy - it is "
                        "too round for a major axis to be defined. Not corrected."}
    ok, reason = inclination_is_usable(inc)
    if not ok:
        return {"key": "leda", "label": "Tilt correction not possible", "angle": None,
                "inc": None, "note": reason}
    ang = sky_pa_to_image_angle(pa_sky, north_angle, mirrored)
    return {
        "key": "leda", "label": "Corrected for a tilt of {:.0f} deg (HyperLEDA)".format(inc),
        "angle": ang, "inc": inc,
        "note": ("PA {:.1f} deg on sky -> {:.1f} deg in image (north at {:.0f} deg{}). "
                 "Stretch {:.2f}x along the minor axis.").format(
                     pa_sky, ang, north_angle, ", mirrored" if mirrored else "",
                     stretch_factor(inc)),
    }

def treatment_manual(inc, pa_image=None, pa_sky=None, north_angle=90.0, mirrored=False):
    """Deprojection using angles the user supplied, from the literature or by eye."""
    if inc is None:
        return None
    ok, reason = inclination_is_usable(inc)
    if pa_image is not None:
        ang, src = pa_image % 180.0, "given directly in image coordinates"
    elif pa_sky is not None:
        ang = sky_pa_to_image_angle(pa_sky, north_angle, mirrored)
        src = "from sky PA {:.1f} deg".format(pa_sky)
    else:
        return {"key": "manual", "label": "Tilt correction incomplete", "angle": None,
                "inc": None, "note": "An inclination was given but no position angle."}
    if not ok:
        return {"key": "manual", "label": "Tilt correction not possible", "angle": None,
                "inc": None, "note": reason}
    return {"key": "manual", "label": "Corrected for a tilt of {:.0f} deg (yours)".format(inc),
            "angle": ang, "inc": inc,
            "note": ("User-supplied geometry, {}. Major axis at {:.1f} deg in the "
                     "image. Stretch {:.2f}x.").format(src, ang, stretch_factor(inc))}

def treatment_sector(dx, dy, n_sectors=20, step=5.0):
    """Deprojection derived from the objects themselves, with its caveats attached."""
    fit = fit_sector_method(dx, dy, n_sectors=n_sectors,
                            angle_step=step, inc_step=step)
    ok, reason = inclination_is_usable(fit["inc"])

    warn = ""
    if fit["few_objects"]:
        warn += (" Only {} objects - below the 150 to 200 needed for a single clean "
                 "minimum, so treat this with suspicion.".format(fit["n_objects"]))
    if fit["runners_up"]:
        a, c, d = fit["runners_up"][0]
        margin = (d - fit["dispersion"]) / max(fit["dispersion"], 1e-9)
        warn += (" Closest rival solution: PA {:.0f} deg, i {:.0f} deg, worse by only "
                 "{:.0%}.".format(a, c, margin))

    if not ok:
        return {"key": "sector", "label": "Tilt from the data - not possible", "angle": None,
                "inc": None,
                "note": ("Sector method found PA {:.0f} deg, i {:.0f} deg, but "
                         .format(fit["angle"], fit["inc"]) + reason + warn)}
    return {"key": "sector", "label": "Corrected for a tilt of {:.0f} deg (measured here)".format(fit["inc"]),
            "angle": fit["angle"], "inc": fit["inc"],
            "note": ("Angles derived from the objects themselves by evening out the "
                     "counts in {} wedges: PA {:.0f} deg in image, i {:.0f} deg, "
                     "stretch {:.2f}x.").format(n_sectors, fit["angle"], fit["inc"],
                                                stretch_factor(fit["inc"])) + warn}

def compute_profiles(x, y, centre, frame, treatments, rings=20):
    """Deproject and bin the objects once per treatment."""
    dx0, dy0 = np.asarray(x, float) - centre[0], np.asarray(y, float) - centre[1]
    results = []
    for t in treatments:
        if t["angle"] is None:
            dxp, dyp = dx0, dy0
        else:
            dxp, dyp = deproject(dx0, dy0, t["angle"], t["inc"])
        res = radial_density(dxp, dyp, n_rings=rings, frame_corners=frame,
                             centre_xy=centre, major_axis_angle_deg=t["angle"],
                             inc_deg=t["inc"])
        res["dx"], res["dy"] = dxp, dyp
        results.append(res)
    return results

def geometric_areas(edges):
    """The full mathematical ring areas, for comparison with the observed ones."""
    return math.pi * (np.asarray(edges)[1:] ** 2 - np.asarray(edges)[:-1] ** 2)

def write_profiles_csv(path, treatments, results, scale=1.0, unit="px"):
    """Save every profile, including both the observed and the full ring areas.

    Keeping the full geometric area alongside the observed one lets anyone compare
    against older analyses, which divided by the full area and so under-reported the
    density of any ring the frame had clipped.
    """
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("treatment,major_axis_angle_deg,inclination_deg,"
                 "r_inner_{u},r_outer_{u},r_centre_{u},n_objects,"
                 "observed_area_{u}2,full_ring_area_{u}2,area_fraction,"
                 "density_per_{u}2,density_err,ring_complete\n".format(u=unit))
        for t, res in zip(treatments, results):
            geo = geometric_areas(res["edges"])
            for k in range(len(res["r"])):
                fh.write("{},{},{},{:.4f},{:.4f},{:.4f},{},{:.4f},{:.4f},{:.4f},"
                         "{:.6e},{:.6e},{}\n".format(
                             t["label"],
                             "" if t["angle"] is None else "{:.1f}".format(t["angle"]),
                             "" if t["inc"] is None else "{:.1f}".format(t["inc"]),
                             res["edges"][k] * scale, res["edges"][k + 1] * scale,
                             res["r"][k] * scale, int(res["counts"][k]),
                             res["area"][k] * scale ** 2, geo[k] * scale ** 2,
                             res["area_fraction"][k],
                             res["density"][k] / scale ** 2,
                             res["density_err"][k] / scale ** 2,
                             int(res["complete"][k])))

def step_profile(ax, edges, values, colour, lw=1.7, alpha=1.0, ls="-", label=None):
    """Draw a density profile as steps, flat across each ring.

    A surface density is not a measurement at a point. It is one number for a whole
    ring - the count in it divided by its area - and it says nothing about where
    inside that ring the objects sat. Drawing it as a dot at the ring's centre
    joined to the next dot by a sloping line invites exactly the reading the data
    does not support: that the density varies smoothly and that the value at the
    centre of the ring is somehow better measured than at its edge.

    Steps say what was actually measured, and it is how the tool in the repository
    has always drawn this, so the two can be laid side by side.
    """
    y = np.append(values, values[-1])
    ax.step(edges, y, where="post", color=colour, lw=lw, alpha=alpha, ls=ls,
            label=label)

def profile_figure(treatments, results, unit_label, title, scale=1.0):
    """One row per treatment - scatter and profile - then all profiles overlaid."""
    n = len(treatments)
    fig = plt.figure(figsize=(11.0, 4.1 * n + 3.4))
    gs = fig.add_gridspec(n + 1, 2, height_ratios=[1] * n + [1.25],
                          hspace=0.85, wspace=0.30)

    for i, (t, res) in enumerate(zip(treatments, results)):
        col = COLOURS[i % len(COLOURS)]
        dxu, dyu = res["dx"] * scale, res["dy"] * scale
        ru = res["r"] * scale
        du = res["density"] / scale ** 2
        eu = res["density_err"] / scale ** 2

        ax = fig.add_subplot(gs[i, 0])
        ax.scatter(dxu, dyu, s=7, c=col, alpha=0.65, linewidths=0)
        ax.plot(0, 0, "+", c="k", ms=11, mew=1.6)
        lim = 1.06 * max(np.abs(dxu).max(), np.abs(dyu).max())
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(t["label"], fontsize=10.5, loc="left")
        ax.set_xlabel("x from centre [" + unit_label + "]", fontsize=8.5)
        ax.set_ylabel("y from centre [" + unit_label + "]", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.15, lw=0.6)

        ax2 = fig.add_subplot(gs[i, 1])
        g = res["complete"]
        edges = res["edges"] * scale
        # The step is the measurement. The bars on top of it are the counting
        # uncertainty, drawn at the middle of each ring only because they have to
        # be drawn somewhere.
        step_profile(ax2, edges, np.where(g, du, np.nan), col)
        if g.any():
            ax2.errorbar(ru[g], du[g], yerr=eu[g], fmt="none", ecolor=col,
                         elinewidth=1.0, capsize=2, alpha=0.75)
        if (~g).any():
            step_profile(ax2, edges, np.where(~g, du, np.nan), col,
                         lw=1.2, alpha=0.45, ls="--",
                         label="ring clipped by the frame")
            ax2.legend(fontsize=7, frameon=False, loc="lower left")
        ax2.set_yscale("log")
        ax2.set_xlabel("deprojected radius [" + unit_label + "]", fontsize=8.5)
        ax2.set_ylabel("surface density [n per " + unit_label + "^2]", fontsize=8.5)
        ax2.tick_params(labelsize=7.5)
        ax2.grid(alpha=0.15, lw=0.6)
        ax2.set_title("N = " + str(int(res["counts"].sum())), fontsize=9, loc="right")

        ax.text(0.0, -0.30, "\n".join(textwrap.wrap(t["note"], width=125)),
                transform=ax.transAxes, fontsize=7.6, color="#4a4a4a",
                va="top", ha="left", linespacing=1.5)

    axc = fig.add_subplot(gs[n, :])
    for i, (t, res) in enumerate(zip(treatments, results)):
        g = res["complete"]
        if g.any():
            step_profile(axc, res["edges"] * scale,
                         np.where(g, res["density"] / scale ** 2, np.nan),
                         COLOURS[i % len(COLOURS)], label=t["label"])
    axc.set_yscale("log")
    axc.set_xlabel("deprojected radius [" + unit_label + "]")
    axc.set_ylabel("surface density [n per " + unit_label + "^2]")
    axc.grid(alpha=0.18, lw=0.6)
    axc.legend(fontsize=8.5, frameon=False)
    axc.set_title("All treatments overlaid - the spread between them is the "
                  "uncertainty on the radii", fontsize=10, loc="left")

    fig.suptitle(title, fontsize=13, y=0.995)
    return fig

def _profile_rows(result, scale, unit):
    """The profile exactly as the picture draws it, ring by ring."""
    edges = result["edges"] * scale
    head = ["ring_inner_" + unit, "ring_outer_" + unit, "r_centre_" + unit,
            "n_objects", "density_per_" + unit + "2", "density_error",
            "ring_complete"]
    rows = []
    for k in range(len(result["r"])):
        rows.append([float(edges[k]), float(edges[k + 1]),
                     float(result["r"][k] * scale),
                     int(result["counts"][k]),
                     float(result["density"][k] / scale ** 2),
                     float(result["density_err"][k] / scale ** 2),
                     int(bool(result["complete"][k]))])
    return head, rows

def _position_rows(result, scale, unit):
    """Every object, where the picture puts it: deprojected, centred, in `unit`."""
    head = ["x_from_centre_" + unit, "y_from_centre_" + unit,
            "radius_" + unit]
    dx = np.asarray(result["dx"], dtype=float) * scale
    dy = np.asarray(result["dy"], dtype=float) * scale
    rows = [[float(a), float(b), float(np.hypot(a, b))] for a, b in zip(dx, dy)]
    return head, rows

def _write_tables(path, sheets):
    """Write each table as its own CSV, named after the picture it belongs to.

    WHY ONE FILE PER TABLE AND NOT ONE PER PICTURE
    Because the two tables behind a picture have different shapes - one row per
    ring in the profile, one row per object in the positions - and putting them in
    one file means one of them starts halfway down, which is exactly the layout
    that makes a spreadsheet awkward to chart from. Two files, each a clean
    rectangle with a header row, can be charted by selecting two columns.

    CSV rather than a spreadsheet format on purpose: it opens in Excel, it opens
    in anything else, and it needs no library that might not be installed on the
    machine this ends up running on.

    Returns the paths written.
    """
    import csv as _csv

    stem = os.path.splitext(path)[0]
    written = []
    for name, (head, rows) in sheets.items():
        slug = "_".join(p for p in
                        "".join(c if c.isalnum() else "_" for c in name).split("_")
                        if p).lower()
        out = "{}_{}.csv".format(stem, slug)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = _csv.writer(fh)
            writer.writerow(head)
            writer.writerows(rows)
        written.append(out)
    return written

def write_treatment_workbook(path, treatment, result, scale, unit):
    """The two tables behind one treatment's picture."""
    return _write_tables(path, {
        "profile": _profile_rows(result, scale, unit),
        "positions": _position_rows(result, scale, unit),
    })

def write_overlay_workbook(path, treatments, results, scale, unit):
    """One table per curve in the overlay."""
    sheets = {}
    for treatment, result in zip(treatments, results):
        name = "".join(c if (c.isalnum() or c == " ") else " "
                       for c in treatment["label"]).strip()
        sheets[name[:31] or "treatment"] = _profile_rows(result, scale, unit)
    return _write_tables(path, sheets)

def one_treatment_figure(treatment, result, unit_label, title, scale=1.0,
                         colour_index=0):
    """One treatment on its own sheet: where the objects are, and their profile.

    WHY THE SAME THING IS DRAWN TWICE
    The combined sheet is for comparing the treatments against each other, which
    is what it is for and what it is good at. It is not what you put in front of
    somebody, or into a report: at four inches high a panel is too small to read a
    radius off it, and two treatments side by side compete for the eye when only
    one of them is under discussion. So each is written again on its own, full
    size, and the overlay too. The combined sheet is unchanged; these are extra.
    """
    col = COLOURS[colour_index % len(COLOURS)]
    fig = plt.figure(figsize=(13.0, 6.2))
    gs = fig.add_gridspec(1, 2, wspace=0.28)

    dxu, dyu = result["dx"] * scale, result["dy"] * scale
    ru = result["r"] * scale
    du = result["density"] / scale ** 2
    eu = result["density_err"] / scale ** 2

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(dxu, dyu, s=14, c=col, alpha=0.7, linewidths=0)
    ax.plot(0, 0, "+", c="k", ms=14, mew=1.8)
    lim = 1.06 * max(np.abs(dxu).max(), np.abs(dyu).max())
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x from centre [" + unit_label + "]")
    ax.set_ylabel("y from centre [" + unit_label + "]")
    ax.grid(alpha=0.15, lw=0.6)
    ax.set_title("where they are", fontsize=11, loc="left")

    ax2 = fig.add_subplot(gs[0, 1])
    g = result["complete"]
    edges = result["edges"] * scale
    step_profile(ax2, edges, np.where(g, du, np.nan), col)
    if g.any():
        ax2.errorbar(ru[g], du[g], yerr=eu[g], fmt="none", ecolor=col,
                     elinewidth=1.2, capsize=3, alpha=0.8)
    if (~g).any():
        step_profile(ax2, edges, np.where(~g, du, np.nan), col,
                     lw=1.3, alpha=0.45, ls="--",
                     label="ring clipped by the frame")
        ax2.legend(fontsize=9, frameon=False, loc="lower left")
    ax2.set_yscale("log")
    ax2.set_xlabel("deprojected radius [" + unit_label + "]")
    ax2.set_ylabel("surface density [n per " + unit_label + "^2]")
    ax2.grid(alpha=0.15, lw=0.6)
    ax2.set_title("N = " + str(int(result["counts"].sum())), fontsize=10,
                  loc="right")
    ax2.set_title("how many per unit area, against radius", fontsize=11,
                  loc="left")

    fig.suptitle("{} - {}".format(title, treatment["label"]), fontsize=13)
    fig.text(0.09, 0.015,
             chr(10).join(textwrap.wrap(treatment["note"], width=150)),
             fontsize=9, color="#4a4a4a", va="bottom")
    fig.subplots_adjust(bottom=0.17, top=0.90)
    return fig

def overlay_figure(treatments, results, unit_label, title, scale=1.0):
    """The treatments on one pair of axes, on a sheet of its own."""
    fig, ax = plt.subplots(figsize=(13.0, 6.2))
    for i, (t, res) in enumerate(zip(treatments, results)):
        g = res["complete"]
        if g.any():
            step_profile(ax, res["edges"] * scale,
                         np.where(g, res["density"] / scale ** 2, np.nan),
                         COLOURS[i % len(COLOURS)], label=t["label"])
    ax.set_yscale("log")
    ax.set_xlabel("deprojected radius [" + unit_label + "]")
    ax.set_ylabel("surface density [n per " + unit_label + "^2]")
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(fontsize=10, frameon=False)
    ax.set_title("{} - all treatments overlaid; the spread between them is the "
                 "uncertainty on the radii".format(title), fontsize=12, loc="left")
    fig.tight_layout()
    return fig

def run_profiles(x, y, centre, frame, out_prefix, title,
                 galaxy=None, geom=None, north_angle=90.0, mirrored=False,
                 manual_inc=None, manual_pa_image=None, manual_pa_sky=None,
                 sector_method=False, sectors=20, sector_step=5.0,
                 rings=20, scale=1.0, unit="px", verbose=True):
    """Deproject, bin, plot and save. Returns (treatments, results, paths).

    out_prefix is a path stem; this writes <prefix>_profiles.csv and
    <prefix>_profiles.png next to it.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Too many rings for too few objects does not add detail, it manufactures
    # noise: rings holding one object or none, error bars taller than the profile,
    # and a jagged line that invites reading structure into counting statistics.
    # Seen on a real run - 181 blue knots spread over 30 rings, whole rings empty.
    if rings and len(x) and len(x) / float(rings) < 8.0:
        suggested = max(4, int(len(x) / 12))
        print("  note: {} objects over {} rings is about {:.1f} per ring, so the "
              "profile below".format(len(x), rings, len(x) / float(rings)))
        print("        will be dominated by counting noise. About {} rings would "
              "be readable.".format(suggested))

    if geom is None and galaxy:
        try:
            describe = describe__galaxy_catalogue  # from galaxy_catalogue, inlined above
            geom = hyperleda_geometry(galaxy)
            if verbose:
                print("\nHyperLEDA geometry for " + galaxy + ":")
                print(describe(geom))
        except Exception as exc:
            if verbose:
                print("\nCatalogue lookup failed ({}: {}). Continuing without it."
                      .format(type(exc).__name__, exc))

    treatments = [treatment_none()]
    t = treatment_catalogue(geom, north_angle, mirrored)
    if t:
        treatments.append(t)
    t = treatment_manual(manual_inc, manual_pa_image, manual_pa_sky,
                         north_angle, mirrored)
    if t:
        treatments.append(t)
    if sector_method:
        treatments.append(treatment_sector(x - centre[0], y - centre[1],
                                           sectors, sector_step))

    results = compute_profiles(x, y, centre, frame, treatments, rings=rings)

    if verbose:
        print("\nTreatments:")
        for t, res in zip(treatments, results):
            print("  - {}: r_max = {:.1f} {}, {} of {} rings clipped by the frame"
                  .format(t["label"], res["r_max"] * scale, unit,
                          int((~res["complete"]).sum()), rings))
            print("      " + t["note"])

    csv_path = out_prefix + "_profiles.csv"
    png_path = out_prefix + "_profiles.png"
    write_profiles_csv(csv_path, treatments, results, scale, unit)
    fig = profile_figure(treatments, results, unit, title, scale)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    combined_books = write_overlay_workbook(
        os.path.splitext(png_path)[0] + ".csv", treatments, results, scale, unit)

    if verbose:
        print("\nWrote " + csv_path)
        print("Wrote " + png_path)
    # And each part again on a sheet of its own, full size. The combined one
    # above is untouched; these are in addition to it.
    extra_paths = []
    for i, (treat, res) in enumerate(zip(treatments, results)):
        slug = ''.join(c if c.isalnum() else '_' for c in treat['label'])
        slug = '_'.join(p for p in slug.split('_') if p).lower()
        one_path = '{}_profile_{}.png'.format(out_prefix, slug)
        one = one_treatment_figure(treat, res, unit, title, scale,
                                   colour_index=i)
        one.savefig(one_path, dpi=150, bbox_inches='tight')
        plt.close(one)
        extra_paths.append(one_path)
        extra_paths.extend(write_treatment_workbook(
            one_path[:-4] + '.csv', treat, res, scale, unit))
    over_path = out_prefix + '_profile_all_overlaid.png'
    over = overlay_figure(treatments, results, unit, title, scale)
    over.savefig(over_path, dpi=150, bbox_inches='tight')
    plt.close(over)
    extra_paths.append(over_path)
    extra_paths.extend(write_overlay_workbook(
        over_path[:-4] + '.csv', treatments, results, scale, unit))
    extra_paths[:0] = combined_books
    if verbose:
        for p in extra_paths:
            print('Wrote ' + p)
    return treatments, results, tuple([csv_path, png_path] + extra_paths)


# --------------------------------------------------------------------------
# Compare_Clusters_And_Regions.py
# --------------------------------------------------------------------------

matplotlib.use("Agg")

warnings.filterwarnings("ignore")

pass  # from radial_profiles, inlined above

describe_match = describe__tracer_matching  # from tracer_matching, inlined above

pass  # from galaxy_catalogue, inlined above

describe_geometry = describe__galaxy_catalogue  # from galaxy_catalogue, inlined above

pass  # from deprojection, inlined above

pass  # from interactive_input, inlined above

ARCSEC_PER_RADIAN = 206264.806

def read_table(path):
    """A CSV as a header list and a list of rows, with comment lines skipped."""
    rows = [r for r in csv.reader(open(path, newline="", encoding="utf-8-sig"))
            if r and not r[0].lstrip().startswith("#")]
    if not rows:
        raise ValueError(path + " is empty")
    return [h.strip() for h in rows[0]], rows[1:]

def columns_of(header, rows, *names):
    """Pull named columns out as float arrays, or say which one is missing."""
    index = {h.lower(): i for i, h in enumerate(header)}
    picked = []
    for name in names:
        if name.lower() not in index:
            raise KeyError("no column called {} - found {}".format(name, header))
        picked.append(index[name.lower()])
    out = [[] for _ in names]
    for row in rows:
        try:
            values = [float(row[i]) for i in picked]
        except (ValueError, IndexError):
            continue
        for k, v in enumerate(values):
            out[k].append(v)
    return [np.array(v, dtype=float) for v in out]

def first_match(folder, patterns):
    """The first file matching any of these patterns, searched recursively."""
    for pattern in patterns:
        hits = sorted(glob.glob(os.path.join(folder, "**", pattern),
                                recursive=True))
        if hits:
            return hits[0]
    return None

def frame_with_a_solution(folder, prefer=()):
    """A FITS file in this folder that carries a usable plate solution.

    Preferred names are tried first - the frame the catalogue was measured on is
    the right one to convert it through - and anything else only as a fallback,
    because every frame a run writes shares one grid.
    """
    ordered = list(prefer) + ["*.fits", "*.fts"]
    seen = []
    for pattern in ordered:
        for path in sorted(glob.glob(os.path.join(folder, "**", pattern),
                                     recursive=True)):
            if path in seen:
                continue
            seen.append(path)
            try:
                header = fits.getheader(path)
                wcs = WCS(header)
                if wcs.has_celestial:
                    return path, wcs, header
            except Exception:
                continue
    return None, None, None

def load_clusters(folder):
    """Blue-knot positions on the sky, from a finished cluster run."""
    table = first_match(folder, ["*color_filtered_with_radius.csv",
                                 "*no_stars_color_filtered.csv",
                                 "*calibrated_photometry_no_stars.csv"])
    if table is None:
        raise SystemExit("No blue-knot catalogue in " + folder
                         + "\n  expected a file ending in "
                           "_color_filtered_with_radius.csv")
    header, rows = read_table(table)
    x, y = columns_of(header, rows, "X_V", "Y_V")

    path, wcs, head = frame_with_a_solution(
        folder, ["*_V_on_V_grid.fits", "*_V_combined.fits", "*_V*.fits"])
    if wcs is None:
        raise SystemExit("No frame with a plate solution in " + folder
                         + "\n  the knot positions cannot be put on the sky "
                           "without one")
    sky = wcs.pixel_to_world(x, y)
    return sky, os.path.basename(table), os.path.basename(path), head

def load_regions(folder):
    """HII region positions on the sky, from a finished region run."""
    table = first_match(folder, ["*_regions.csv"])
    if table is None:
        raise SystemExit("No HII region catalogue in " + folder
                         + "\n  expected a file ending in _regions.csv")
    header, rows = read_table(table)
    x, y = columns_of(header, rows, "x", "y")

    path, wcs, head = frame_with_a_solution(
        folder, ["*_Ha_minus_continuum.fits", "*_subtracted.fits", "*.fits"])
    if wcs is None:
        raise SystemExit("No frame with a plate solution in " + folder)
    sky = wcs.pixel_to_world(x, y)
    shape = (head["NAXIS2"], head["NAXIS1"])
    return sky, os.path.basename(table), os.path.basename(path), head, wcs, shape

def offsets_arcsec(centre, coords):
    """Positions as offsets from the centre, in arcseconds, east and north.

    A tangent plane about the galaxy's own centre. Over a field of a few tenths of
    a degree the difference between this and the sphere is far below a pixel, and
    working in it means the rest of the analysis - rings, deprojection, nearest
    neighbours - is ordinary flat geometry with no grid behind it.
    """
    d_lon, d_lat = centre.spherical_offsets_to(coords)
    return (d_lon.to(u.arcsec).value, d_lat.to(u.arcsec).value)

def frame_extent_arcsec(centre, wcs, shape):
    """How far the region frame reaches, in the same arcsecond frame.

    The ring areas are corrected for the part of each ring that falls outside the
    frame, so the analysis has to know where the frame ends. The corners are
    enough: the edges between them are straight to well under a pixel here.
    """
    ny, nx = shape
    xs = [0, nx - 1, 0, nx - 1]
    ys = [0, 0, ny - 1, ny - 1]
    corners = wcs.pixel_to_world(np.array(xs, dtype=float),
                                 np.array(ys, dtype=float))
    cx, cy = offsets_arcsec(centre, corners)
    return (float(cx.min()), float(cx.max()), float(cy.min()), float(cy.max()))

def both_populations_figure(knots, regions, pairs, unit, title):
    """Both tracers on one map, with an arrow from each knot to its parent region.

    Two panels for the same reason the full tool uses two: the galaxy is thousands
    of parsecs across and the separation being drawn is of order a hundred, so at
    the scale that fits the galaxy a typical arrow is shorter than its own
    arrowhead. The right panel is a tenth of the width, placed where the pairs
    actually are, and it is the one the arrows can be read in.
    """
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(15.5, 8.2))
    kx, ky = knots[:, 0], knots[:, 1]
    rx, ry = regions[:, 0], regions[:, 1]
    idx = pairs["nearest_index"]

    def draw(target, lw, size_r, size_k):
        target.scatter(rx, ry, s=size_r, c="#a3283c", alpha=0.55, linewidths=0,
                       label="HII regions ({})".format(len(rx)))
        target.scatter(kx, ky, s=size_k, facecolors="none", edgecolors="#2b5a89",
                       linewidths=1.1, label="blue knots ({})".format(len(kx)))
        # The arrow points the way the cluster travelled: from the gas that lit
        # up when its stars switched on, to where the cluster is now. Drawn the
        # other way round it would read as the cluster moving into the gas, which
        # is the reverse of what happened.
        for i in range(len(kx)):
            j = idx[i]
            target.annotate("", xy=(kx[i], ky[i]), xytext=(rx[j], ry[j]),
                            arrowprops=dict(arrowstyle="->", color="#333333",
                                            lw=lw, alpha=0.85, shrinkA=0,
                                            shrinkB=0, mutation_scale=8))

    draw(ax, 0.7, 16, 26)
    ax.plot(0, 0, "+", c="k", ms=15, mew=2)
    lim = 1.05 * max(np.abs(np.concatenate([kx, rx])).max(),
                     np.abs(np.concatenate([ky, ry])).max())
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("east of centre [" + unit + "]")
    ax.set_ylabel("north of centre [" + unit + "]")
    ax.grid(alpha=0.15, lw=0.6)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_title(title, fontsize=12, loc="left")

    half = 0.10 * lim
    best, most = (0.0, 0.0), -1
    for cxx in np.linspace(-lim + half, lim - half, 25):
        for cyy in np.linspace(-lim + half, lim - half, 25):
            here = int(((np.abs(kx - cxx) < half) & (np.abs(ky - cyy) < half)).sum())
            if here > most:
                best, most = (float(cxx), float(cyy)), here

    draw(zoom, 1.3, 90, 150)
    zoom.set_xlim(best[0] - half, best[0] + half)
    zoom.set_ylim(best[1] - half, best[1] + half)
    zoom.set_aspect("equal", adjustable="box")
    zoom.set_xlabel("east of centre [" + unit + "]")
    zoom.grid(alpha=0.15, lw=0.6)
    zoom.set_title("the busiest {:.0f} {} of it - {} knots".format(
        2 * half, unit, most), fontsize=11, loc="left")
    ax.add_patch(plt.Rectangle((best[0] - half, best[1] - half), 2 * half,
                               2 * half, fill=False, edgecolor="#333333",
                               lw=1.2, linestyle="--"))

    note = ("Each arrow runs from an HII region to the blue knot nearest it - "
            "from where the cluster formed to where it is now. Mean separation "
            "{:.0f} {}, median {:.0f}. Measured on the sky, not on either run's "
            "pixel grid.").format(pairs["mean_separation"], unit,
                                  float(np.nanmedian(pairs["separation"])))
    fig.text(0.06, 0.05, "\n".join(textwrap.wrap(note, width=150)),
             fontsize=8.5, color="#4a4a4a", va="top")
    fig.subplots_adjust(bottom=0.16)
    return fig

def separation_figure(pairs, unit, title):
    """Separation against galactocentric radius, against the density null model."""
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    r, s, e = pairs["radius"], pairs["separation"], pairs["expected"]

    ax.scatter(r, s, s=30, c="#2b5a89", alpha=0.75, linewidths=0,
               label="measured separation")
    order = np.argsort(r)
    if np.isfinite(pairs.get("fit_slope", np.nan)):
        ax.plot(r[order], np.polyval([pairs["fit_slope"], pairs["fit_intercept"]],
                                     r[order]), "-", lw=1.8, c="#2b5a89",
                label="fit: slope {:.3f}, R2 = {:.2f}".format(
                    pairs["fit_slope"], pairs["fit_r2"]))
    good = np.isfinite(e)
    if good.any():
        ax.plot(r[order][np.isfinite(e[order])], e[order][np.isfinite(e[order])],
                "--", lw=2.0, c="#a3283c",
                label="expected from HII density alone")
    ax.set_xlabel("galactocentric radius [" + unit + "]")
    ax.set_ylabel("separation to the nearest HII region [" + unit + "]")
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title(title, fontsize=12, loc="left")
    fig.tight_layout()
    return fig

def ask_for_arguments():
    """Two folders and somewhere to write, when nothing was passed."""
    print("No arguments were given, so I will ask instead.")
    print("Press Enter to accept the value shown in brackets.")
    print("")
    clusters = ask_folder("Output folder of the blue-cluster run")
    regions = ask_folder("Output folder of the HII-region run")
    out = normalise_path(ask("Where to write the comparison", clusters, str))
    galaxy = ask("Galaxy name [blank and it is read from the frames]", "", str)
    distance = ask("Distance in Mpc [blank and it is looked up]", "", str)
    argv = ["--clusters", clusters, "--regions", regions, "--out", out]
    if galaxy:
        argv += ["--galaxy", galaxy]
    if distance:
        argv += ["--distance-mpc", distance]
    return argv

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clusters", help="output folder of the blue-cluster run")
    ap.add_argument("--regions", help="output folder of the HII-region run")
    ap.add_argument("--out", default=None, help="where to write the comparison")
    ap.add_argument("--galaxy", default=None)
    ap.add_argument("--distance-mpc", type=float, default=None)
    ap.add_argument("--rings", type=int, default=20)
    args = ap.parse_args(ask_for_arguments() if nothing_was_passed(argv) else argv)

    if not (args.clusters and args.regions):
        ap.error("give both --clusters and --regions")

    knot_sky, knot_table, knot_frame, knot_head = load_clusters(args.clusters)
    region_sky, region_table, region_frame, region_head, region_wcs, region_shape = \
        load_regions(args.regions)

    galaxy = args.galaxy or region_head.get("OBJECT") or knot_head.get("OBJECT")
    if not galaxy:
        ap.error("neither run's frames name a galaxy - give --galaxy")
    galaxy = str(galaxy).strip()

    print("")
    print("Blue knots : {:5d}  from {}".format(len(knot_sky), knot_table))
    print("             positions put on the sky through " + knot_frame)
    print("HII regions: {:5d}  from {}".format(len(region_sky), region_table))
    print("             positions put on the sky through " + region_frame)
    print("Galaxy     : " + galaxy)

    distance = args.distance_mpc
    if distance is None:
        distance = lookup_distance(galaxy)
    if not distance:
        ap.error("no distance for " + galaxy + " - give --distance-mpc")
    distance = float(distance)

    # The galaxy's own centre, from its catalogued position through one plate
    # solution. Taken once and used for both populations: each run worked out a
    # centre of its own, and two centres a pixel apart would put the two tracers
    # on two slightly different radial scales.
    centre_sky = SkyCoord.from_name(galaxy)
    print("Centre     : {:.5f} {:+.5f} deg, from the catalogue".format(
        centre_sky.ra.deg, centre_sky.dec.deg))

    geom = hyperleda_geometry(galaxy)
    if geom:
        print("")
        print(describe_geometry(geom))

    kx, ky = offsets_arcsec(centre_sky, knot_sky)
    rx, ry = offsets_arcsec(centre_sky, region_sky)
    frame = frame_extent_arcsec(centre_sky, region_wcs, region_shape)

    # Arcseconds to parsecs. An angle in radians times a distance is a length;
    # everything below is reported in parsecs and nothing is ever in pixels.
    scale = distance * 1.0e6 / ARCSEC_PER_RADIAN
    unit = "pc"
    print("")
    print("  {:.2f} Mpc, so one arcsecond is {:.1f} pc".format(distance, scale))
    print("  measured on the sky - neither run's pixel grid is used")

    # North is up and east is left in this frame by construction, so a position
    # angle measured on the sky needs no rotation to reach it.
    north_angle = 90.0

    print("")
    print("=== radial analysis: HII regions ===")
    out_dir = args.out or args.clusters
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, galaxy.replace(" ", "_"))

    _, hii_res, _ = run_profiles(
        rx, ry, (0.0, 0.0), frame, stem + "_HII",
        "HII regions - " + galaxy, geom=geom, north_angle=north_angle,
        mirrored=False, rings=args.rings, scale=scale, unit=unit)

    print("")
    print("=== radial analysis: blue knots ===")
    _, knot_res, _ = run_profiles(
        kx, ky, (0.0, 0.0), frame, stem + "_knots",
        "Blue knots - " + galaxy, geom=geom, north_angle=north_angle,
        mirrored=False, rings=args.rings, scale=scale, unit=unit)

    print("")
    print("=== pairing blue knots with HII regions ===")
    profile = hii_res[0]
    good = profile["complete"]
    pairs = match_report(np.column_stack([kx, ky]), np.column_stack([rx, ry]),
                         (0.0, 0.0), profile["r"][good] * scale,
                         profile["density"][good] / scale ** 2,
                         scale=scale, unit=unit)
    print(describe_match(pairs))

    # Pairs far enough apart that the association is unlikely to be real. Not
    # removed - a cut here would quietly decide the answer - but counted, because
    # two catalogues from two separate runs can pair objects that have nothing to
    # do with each other, and the mean is what such a pair moves.
    far = np.asarray(pairs["separation"]) > 1000.0
    if far.any():
        print("  {} of {} knots are more than 1 kpc from any region - kept, but "
              "they are unlikely to be real associations".format(
                  int(far.sum()), len(far)))
        kept = np.asarray(pairs["separation"])[~far]
        if kept.size:
            print("     without them the mean separation would be {:.0f} {} "
                  "instead of {:.0f}".format(float(np.nanmean(kept)), unit,
                                             pairs["mean_separation"]))

    path = stem + "_matches.csv"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# blue knots paired with their nearest HII region\n")
        fh.write("# both catalogues converted to the sky and compared there;\n")
        fh.write("# offsets are from the catalogued centre of {}, in {}\n".format(
            galaxy, unit))
        fh.write("knot_east,knot_north,hii_east,hii_north,separation,"
                 "knot_radius,expected_from_density,excess_ratio\n")
        for i in range(len(kx)):
            j = pairs["nearest_index"][i]
            fh.write("{:.3f},{:.3f},{:.3f},{:.3f},{:.4f},{:.4f},{:.4f},"
                     "{:.4f}\n".format(
                         kx[i] * scale, ky[i] * scale, rx[j] * scale,
                         ry[j] * scale, pairs["separation"][i],
                         pairs["radius"][i], pairs["expected"][i],
                         pairs["excess_ratio"][i]))
    print("Wrote " + path)

    fig = both_populations_figure(
        np.column_stack([kx * scale, ky * scale]),
        np.column_stack([rx * scale, ry * scale]),
        pairs, unit, "Both tracers - " + galaxy)
    p1 = stem + "_both_populations.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote " + p1)

    fig = separation_figure(pairs, unit,
                            "Separation against radius - " + galaxy)
    p2 = stem + "_separation_vs_radius.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote " + p2)
    return 0

if __name__ == "__main__":
    sys.exit(main())
