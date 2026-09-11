"""
Star_Formation_Full_Analysis.py - both tracers of star formation, and the link between them.

WHAT IT DOES
Takes all four frames of a galaxy - B, V, R and H-alpha - and produces everything
the two single-tracer tools produce, plus the thing neither of them can do alone:
it pairs each young blue cluster with the HII region it most likely came from, and
asks how that separation behaves across the galaxy.

WHY THE PAIRING IS INTERESTING
An HII region is gas glowing around stars that have just switched on. A blue knot
is the cluster those stars belong to, seen about ten million years later, by which
time it has drifted a little from where it formed. The distance between them is
that drift made visible. Student projects on three galaxies measured a few hundred
parsecs each time - 527, 525 and 384 - which is what OB stars moving at about ten
kilometres a second for ten million years would give.

THE ONE GRID RULE
The four frames are not taken at the same instant and the telescope drifts between
them. On the M74 set used to develop this tool, the same catalogued sky position
lands at pixel 952 in B and at pixel 965 in R - thirteen pixels apart, about 190
parsecs at that galaxy's distance. Since the whole measurement here is a separation
of a few hundred parsecs, comparing positions measured on two different frames
without registering them first would put a large fraction of the answer into the
bookkeeping.

So everything is resampled onto one reference grid, the V frame, before any
distance is measured. The blue knots are measured on V already; H-alpha and R are
brought to it through both WCS solutions.

WHAT IT NEEDS
    four plate-solved FITS frames of the same galaxy: B, V, R, H-alpha

USAGE
    python Star_Formation_Full_Analysis.py --folder "D:/NGC4212" --galaxy "NGC 4212"
        --distance-mpc 16.7

    The folder holds the frames; which file is which is read from the headers, and
    what was chosen is printed before anything is measured. Naming the four files
    directly still works:

    python Star_Formation_Full_Analysis.py --b B.fits --v V.fits --red R.fits --halpha Ha.fits
        --galaxy "NGC 4212" --distance-mpc 16.7

    # if the blue knots have already been measured by Blue_Clusters_From_Images.py
    python Star_Formation_Full_Analysis.py --b B.fits --v V.fits --red R.fits --halpha Ha.fits
        --galaxy "NGC 4212" --knots-csv result_CMD/NGC4212_..._with_radius.csv
"""

# ===========================================================================
# GENERATED FILE - do not edit.
#
# Built by build.py on 2026-09-11 10:51 from:
#   Star_Formation_Full_Analysis.py
#   interactive_input.py
#   frame_inventory.py
#   anchor_registration.py
#   galaxy_window.py
#   galaxy_extent.py
#   deprojection.py
#   galaxy_catalogue.py
#   tracer_matching.py
#   continuum_subtraction.py
#   radial_profiles.py
#   HII_From_Images.py  (image handling)
#
# This file is standalone on purpose: it needs nothing else from the
# project, only the usual third-party packages. The cost of that is this
# warning - a fix made here is overwritten by the next build. Change the
# source above instead, then run build.py again.
# ===========================================================================


from __future__ import annotations

import os
import sys
import re
import numpy as np
import math
import textwrap
import matplotlib
import matplotlib.pyplot as plt
import argparse
import warnings
import glob
import subprocess


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
# shared module frame_inventory.py
# --------------------------------------------------------------------------

ROLES = ("B", "V", "R", "Ha")

ROLE_MEANING = {
    "B": "blue",
    "V": "visual - the reference grid",
    "R": "red continuum",
    "Ha": "H-alpha - the emission line",
}

_PATTERNS = {
    "Ha": re.compile(r"(?<![a-z0-9])h[\s_-]*al(?:pha|fa)(?![a-z0-9])"
                     r"|(?<![a-z0-9])ha(?![a-z0-9])"),
    "B": re.compile(r"(?<![a-z0-9])(?:b|bessell[\s_-]*b|bp)(?![a-z0-9])"),
    "V": re.compile(r"(?<![a-z0-9])(?:v|bessell[\s_-]*v)(?![a-z0-9])"),
    "R": re.compile(r"(?<![a-z0-9])(?:r|rp|rc|rprime|sloan[\s_-]*r|bessell[\s_-]*r)"
                    r"(?![a-z0-9])"),
}

_COMBINED_WORDS = re.compile(r"\b(median|average|mean|sum)\s+of\b|\bcombin|\bstack",
                             re.I)

FITS_SUFFIXES = (".fits", ".fit", ".fts", ".fits.fz")

OUTPUT_FOLDERS = {"result_cmd", "result", "results", "output", "out"}

def roles_in_text(text):
    """Every filter role named in a piece of text."""
    if not text:
        return set()
    low = str(text).lower()
    return {role for role, pattern in _PATTERNS.items() if pattern.search(low)}

def role_from_header(header):
    """The role named by the filter keyword, or None when there is no such keyword."""
    for key in ("FILTER", "FILTER1", "FILTNAME", "INSFILTE"):
        value = header.get(key)
        if value is None or str(value).strip().upper() in ("", "NOTPRESENT", "NONE"):
            continue
        found = roles_in_text(value)
        if len(found) == 1:
            return found.pop()
    return None

def history_lines(header):
    """The history lines of a header, as plain strings."""
    try:
        return [str(v) for v in header["HISTORY"]]
    except (KeyError, TypeError):
        return []

def is_combined(header):
    """True when the header says this frame is several frames combined."""
    return any(_COMBINED_WORDS.search(line) for line in history_lines(header))

def combine_note(header):
    """The history line that says how the frame was combined, for the summary."""
    for line in history_lines(header):
        if _COMBINED_WORDS.search(line):
            return line.strip()
    return ""

def has_wcs(header):
    """True when the header carries a plate solution the tools can use."""
    try:
        from astropy.wcs import WCS
        return WCS(header).has_celestial
    except Exception:
        return False

def inspect(path):
    """Everything the tools need to know about one file."""
    from astropy.io import fits

    header = fits.getheader(path)
    role = role_from_header(header)
    source = "the filter keyword in the header"

    if role is None:
        found = roles_in_text(os.path.basename(path))
        source = "the file name"
        if len(found) == 1:
            role = next(iter(found))

    return {
        "path": path,
        "name": os.path.basename(path),
        "role": role,
        "role_source": source if role else "",
        "combined": is_combined(header),
        "combine_note": combine_note(header),
        "wcs": has_wcs(header),
        "exptime": header.get("EXPTIME"),
        "object": (str(header.get("OBJECT")).strip()
                   if header.get("OBJECT") else ""),
        "shape": (header.get("NAXIS2"), header.get("NAXIS1")),
    }

def scan_folder(folder, recursive=True):
    """Inspect every FITS file in a folder. Returns a list of records."""
    found = []
    for root, dirs, files in os.walk(folder):
        # Do not walk into what an earlier run wrote.
        dirs[:] = [d for d in dirs if d.lower() not in OUTPUT_FOLDERS]
        for name in sorted(files):
            if not name.lower().endswith(FITS_SUFFIXES):
                continue
            path = os.path.join(root, name)
            try:
                found.append(inspect(path))
            except Exception as exc:
                found.append({"path": path, "name": name, "role": None,
                              "role_source": "", "combined": False,
                              "combine_note": "", "wcs": False, "exptime": None,
                              "object": "",
                              "shape": (None, None),
                              "error": "{}: {}".format(type(exc).__name__, exc)})
        if not recursive:
            break
    return found

def galaxy_from_records(records, folder=None):
    """The galaxy's name, taken from the frames themselves.

    The telescope wrote it into every exposure it took, so there is no reason to
    ask for it - and no reason to read it off a folder name, which is written by a
    person and carries whatever else they wanted to remember that day. A combined
    frame has usually lost the keyword along with the rest of its header, but it
    only takes one exposure in the folder that still has it.

    Returns (name, where it came from). The folder name is the fallback, cleaned of
    the notes people append to it.
    """
    import collections

    votes = collections.Counter(r.get("object", "") for r in records
                                if r.get("object"))
    if votes:
        name, n = votes.most_common(1)[0]
        return name, "the OBJECT keyword, in {} of {} frames".format(n, len(records))

    if folder:
        stem = os.path.basename(os.path.normpath(folder))
        # "M33 0.4 meter" is a note to self, not a catalogue name; the name is the
        # first word that looks like one.
        m = re.match(r"\s*([A-Za-z]{1,4}\s*\d+[A-Za-z]?)", stem)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip(), "the folder name"
        return stem, "the folder name"

    return None, ""

def choose(records, roles=ROLES, ask=None):
    """Pick one frame per role: the combined one, when there is one.

    A combined frame is deeper than any single exposure, which is the reason it
    was made, so it wins whenever one exists. When there is none but several
    exposures, all of them are returned to be combined rather than one being
    picked - that is what the observer went out and took.

    Returns (chosen, members, notes): the representative frame per role, the
    exposures to combine where there are several, and anything worth saying.
    """
    chosen, members, notes = {}, {}, []

    for role in roles:
        mine = [r for r in records if r.get("role") == role and not r.get("error")]
        if not mine:
            notes.append("{}: nothing in the folder identifies itself as {}".format(
                role, ROLE_MEANING[role]))
            continue

        combined = [r for r in mine if r["combined"]]
        if len(combined) == 1:
            chosen[role] = combined[0]
            continue
        if len(combined) > 1:
            if ask is None:
                raise RuntimeError("{}: {} combined frames to choose from".format(
                    role, len(combined)))
            chosen[role] = ask(role, combined)
            continue

        singles = [r for r in mine if r["wcs"]] or mine
        if len(singles) == 1:
            chosen[role] = singles[0]
            notes.append("{}: no combined frame, using the only exposure".format(role))
        else:
            # Several exposures and nothing combined. Picking one of them and
            # discarding the rest throws away most of the observation - with six
            # H-alpha frames it costs a factor of 2.4 in noise - so they are all
            # kept and the caller combines them. The first is named as the
            # representative because the output has to be labelled with something.
            chosen[role] = singles[0]
            members[role] = singles
            notes.append("{}: no combined frame; {} exposures to combine".format(
                role, len(singles)))

    return chosen, members, notes

def describe(records, chosen, notes):
    """The summary a person reads before agreeing to the run."""
    lines = ["", "What the folder holds:"]
    by_role = {}
    for r in records:
        by_role.setdefault(r.get("role") or "?", []).append(r)

    for role in list(ROLES) + ["?"]:
        group = by_role.get(role)
        if not group:
            continue
        lines.append("  " + ("{} ({})".format(role, ROLE_MEANING[role])
                             if role in ROLE_MEANING else "not identified"))
        for r in sorted(group, key=lambda x: x["name"]):
            mark = "->" if chosen.get(role) is r else "  "
            if r.get("error"):
                bits = ["unreadable: " + r["error"]]
            else:
                bits = ["combined" if r["combined"] else "single",
                        "solved" if r["wcs"] else "NO WCS"]
            lines.append("    {} {:<42} {}".format(mark, r["name"], ", ".join(bits)))

    lines.append("")
    lines.append("Chosen for the measurement:")
    for role in ROLES:
        r = chosen.get(role)
        lines.append("    {:<3} {}".format(role, r["name"] if r else "nothing"))
    for n in notes:
        lines.append("    note: " + n)
    lines.append("")
    return "\n".join(lines)

def wcs_from_reference(image, reference_image, reference_wcs,
                       detection_sigma=6.0, max_control_points=60):
    """Fit a plate solution for an unsolved frame by matching its stars.

    A combined frame and the single exposures it was built from show the same
    stars in almost the same places, so a solved exposure of the same field can
    lend its solution: match the star patterns, read each matched star's sky
    position from the solved frame, and fit a solution to those positions in the
    unsolved frame's own pixels.

    This is not copying a header across. The result is fitted to matched stars and
    comes with the residual of that fit, which is the honest measure of how well
    it did: a few hundredths of a pixel means the two frames really are on the
    same grid, and a residual approaching a pixel means something moved.

    Returns (wcs, diagnostics).
    """
    import numpy as np
    import astroalign as aa
    from astropy.wcs.utils import fit_wcs_from_points

    transform, (src, dst) = aa.find_transform(
        image, reference_image,
        max_control_points=max_control_points,
        detection_sigma=detection_sigma)

    sky = reference_wcs.pixel_to_world(dst[:, 0], dst[:, 1])
    fitted = fit_wcs_from_points((src[:, 0], src[:, 1]), sky, proj_point="center")

    back_x, back_y = fitted.world_to_pixel(sky)
    residual = np.hypot(back_x - src[:, 0], back_y - src[:, 1])

    return fitted, {
        "n_stars": int(len(src)),
        "median_residual_px": float(np.median(residual)),
        "worst_residual_px": float(residual.max()),
        "shift_px": (float(transform.translation[0]), float(transform.translation[1])),
        "rotation_deg": float(np.degrees(transform.rotation)),
        "scale": float(transform.scale),
    }

def describe_wcs_fit(name, diag):
    """One readable paragraph about a solution lent from another frame."""
    return (
        "  {}: solved by matching {} stars against an exposure that has a solution\n"
        "    shift ({:+.1f}, {:+.1f}) px, rotation {:+.3f} deg, scale {:.5f}\n"
        "    fit residual: median {:.2f} px, worst {:.2f} px".format(
            name, diag["n_stars"], diag["shift_px"][0], diag["shift_px"][1],
            diag["rotation_deg"], diag["scale"],
            diag["median_residual_px"], diag["worst_residual_px"]))

def donor_for(record, records):
    """A solved exposure that can lend its plate solution to an unsolved frame.

    An exposure through the same filter is preferred - same stars, same depth,
    the closest thing to the frame that needs solving - but any solved exposure
    of the field will do, since the match is made on star positions and those do
    not care which filter they were seen through.
    """
    if record is None or record.get("wcs"):
        return None
    same_filter = [r for r in records
                   if r.get("wcs") and r.get("role") == record.get("role")]
    if same_filter:
        return same_filter[0]
    solved = [r for r in records if r.get("wcs")]
    return solved[0] if solved else None

def plan(folder, roles=ROLES, ask=None, recursive=True):
    """Everything a tool needs from one folder path.

    Returns (records, chosen, members, donors, notes): what is in the folder, the
    frame chosen for each role, the several exposures to combine when there was no
    combined frame, which solved exposure will lend a plate solution to a frame
    that lacks one, and anything worth saying out loud.
    """
    records = scan_folder(folder, recursive=recursive)
    if not records:
        raise RuntimeError("no FITS files in " + str(folder))
    chosen, members, notes = choose(records, roles=roles, ask=ask)
    donors = {role: donor_for(rec, records) for role, rec in chosen.items()}
    for role, donor in donors.items():
        if donor is not None:
            notes.append("{}: has no plate solution; will borrow one from {}".format(
                role, donor["name"]))
        elif role in chosen and not chosen[role]["wcs"]:
            notes.append("{}: has no plate solution and there is no solved "
                         "exposure in the folder to take one from".format(role))
    return records, chosen, members, donors, notes

describe__frame_inventory = describe   # this module's own; the plain name gets shadowed below


# --------------------------------------------------------------------------
# shared module anchor_registration.py
# --------------------------------------------------------------------------

def pick_anchors(positions, shape, n_anchors=4, edge_margin=40):
    """Choose a few widely separated stars from a list of candidate positions.

    Spread is the whole point. The rotation angle is measured as a difference of
    directions between two stars, so its precision is the centring precision
    divided by the distance between them: anchors close together measure the
    angle badly no matter how well each one is centred.

    The frame is divided into quadrants and one star is taken from each, as far
    from the centre as it can be found. That guarantees the spread instead of
    hoping for it.
    """
    positions = np.asarray(positions, dtype=float)
    if len(positions) == 0:
        return positions

    ny, nx = shape
    inside = ((positions[:, 0] > edge_margin) & (positions[:, 0] < nx - edge_margin)
              & (positions[:, 1] > edge_margin) & (positions[:, 1] < ny - edge_margin))
    positions = positions[inside]
    if len(positions) <= n_anchors:
        return positions

    cx, cy = nx / 2.0, ny / 2.0
    dx, dy = positions[:, 0] - cx, positions[:, 1] - cy
    radius = np.hypot(dx, dy)

    chosen = []
    quadrants = ((dx >= 0) & (dy >= 0), (dx < 0) & (dy >= 0),
                 (dx < 0) & (dy < 0), (dx >= 0) & (dy < 0))
    for q in quadrants:
        if q.any():
            chosen.append(positions[np.where(q)[0][np.argmax(radius[q])]])

    # A field with an empty quadrant still gets its four, taken from wherever the
    # stars actually are.
    if len(chosen) < n_anchors:
        order = np.argsort(radius)[::-1]
        for i in order:
            p = positions[i]
            if not any(np.allclose(p, c) for c in chosen):
                chosen.append(p)
            if len(chosen) >= n_anchors:
                break

    return np.array(chosen[:n_anchors])

def centre_on(image, positions, box=15):
    """Where each star actually sits in this frame, to a fraction of a pixel."""
    from photutils.centroids import centroid_sources, centroid_com

    positions = np.asarray(positions, dtype=float)
    if not len(positions):
        return np.empty((0, 2))
    try:
        x, y = centroid_sources(np.nan_to_num(image),
                                positions[:, 0], positions[:, 1],
                                box_size=box, centroid_func=centroid_com)
    except Exception:
        return np.empty((0, 2))
    return np.column_stack([x, y])

def anchor_transform(reference, image, positions, box=15, max_move_px=8.0,
                     n_anchors=4):
    """Fit shift, rotation and scale from a few anchor stars.

    Returns (transform, diagnostics) where transform maps a position in `image`
    to the matching position in `reference`, or (None, diagnostics) when the
    anchors could not be measured well enough to trust.

    The odd one out is dropped: each anchor's residual against the fit is
    measured, and if one stands far above the rest it is removed and the fit
    redone. That is what the fourth anchor is for.
    """
    anchors = pick_anchors(positions, reference.shape, n_anchors=n_anchors)
    diag = {"n_anchors": len(anchors), "dropped": 0}
    if len(anchors) < 2:
        diag["why"] = "fewer than two usable anchor stars"
        return None, diag

    a = centre_on(reference, anchors, box=box)
    b = centre_on(image, anchors, box=box)
    if len(a) != len(b) or not len(a):
        diag["why"] = "the anchors could not be centred"
        return None, diag

    ok = (np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
          & (np.hypot(*(b - a).T) < max_move_px))
    a, b = a[ok], b[ok]
    if len(a) < 2:
        diag["why"] = "the anchors moved further than a frame like this should"
        return None, diag

    def fit(p, q):
        """Least-squares similarity transform taking q onto p."""
        pc, qc = p.mean(axis=0), q.mean(axis=0)
        p0, q0 = p - pc, q - qc
        denom = (q0 ** 2).sum()
        if denom <= 0:
            return None
        # The rotation-and-scale part, as one complex multiplication.
        num = complex((p0[:, 0] * q0[:, 0] + p0[:, 1] * q0[:, 1]).sum(),
                      (p0[:, 1] * q0[:, 0] - p0[:, 0] * q0[:, 1]).sum())
        z = num / denom
        return z, pc, qc

    got = fit(a, b)
    if got is None:
        diag["why"] = "the anchors are on top of each other"
        return None, diag
    z, pc, qc = got

    if len(a) >= 4:
        pred = _apply(b, z, pc, qc)
        resid = np.hypot(*(pred - a).T)
        worst = int(np.argmax(resid))
        others = np.delete(resid, worst)
        if resid[worst] > max(3.0 * np.median(others), 0.5):
            a, b = np.delete(a, worst, 0), np.delete(b, worst, 0)
            diag["dropped"] = 1
            z, pc, qc = fit(a, b)

    pred = _apply(b, z, pc, qc)
    resid = np.hypot(*(pred - a).T)
    diag.update({
        "n_used": len(a),
        "shift_px": (float(pc[0] - qc[0] * z.real + qc[1] * z.imag),
                     float(pc[1] - qc[0] * z.imag - qc[1] * z.real)),
        "rotation_deg": float(np.degrees(np.arctan2(z.imag, z.real))),
        "scale": float(abs(z)),
        "residual_px": float(np.max(resid)),
        "separation_px": float(np.hypot(*(a.max(axis=0) - a.min(axis=0)))),
    })
    return (z, pc, qc), diag

def _apply(points, z, pc, qc):
    """Map points through a fitted similarity transform."""
    q0 = np.asarray(points, dtype=float) - qc
    x = q0[:, 0] * z.real - q0[:, 1] * z.imag + pc[0]
    y = q0[:, 0] * z.imag + q0[:, 1] * z.real + pc[1]
    return np.column_stack([x, y])

def apply_transform(image, transform, shape=None):
    """Resample a frame through a transform fitted from the anchors.

    One affine interpolation, and no astrometric arithmetic at all. That is where
    the time goes: the version that went through both plate solutions had to turn
    sixty-one million pixel positions into sky positions and back again before it
    could interpolate anything.
    """
    from scipy.ndimage import affine_transform

    z, pc, qc = transform
    shape = shape or image.shape

    # affine_transform pulls: for each output pixel it asks where in the input to
    # look, so the inverse of the fitted map is what it needs, written in the
    # (row, column) order it expects.
    inv = 1.0 / z
    m = np.array([[inv.real, inv.imag], [-inv.imag, inv.real]], dtype=float)
    offset = np.array([qc[1], qc[0]]) - m.dot(np.array([pc[1], pc[0]]))

    known = np.isfinite(image)
    filled = np.where(known, image, 0.0).astype(np.float32)
    out = affine_transform(filled, m, offset=offset, output_shape=shape,
                           order=1, mode="constant", cval=np.nan)
    seen = affine_transform(known.astype(np.float32), m, offset=offset,
                            output_shape=shape, order=1, mode="constant", cval=0.0)
    out[seen < 0.5] = np.nan
    return out

def describe(diag):
    """The registration, in a line worth printing."""
    if diag.get("why"):
        return "  anchors unusable ({}); falling back".format(diag["why"])
    return ("  {} anchor stars{}, {:.0f} px apart: shift ({:+.2f}, {:+.2f}) px, "
            "rotation {:+.4f} deg, scale {:.6f}; worst anchor off by {:.2f} px"
            .format(diag["n_used"],
                    " (one dropped)" if diag.get("dropped") else "",
                    diag["separation_px"], diag["shift_px"][0], diag["shift_px"][1],
                    diag["rotation_deg"], diag["scale"], diag["residual_px"]))

describe__anchor_registration = describe   # this module's own; the plain name gets shadowed below


# --------------------------------------------------------------------------
# shared module galaxy_window.py
# --------------------------------------------------------------------------

def galaxy_window(shape, centre, r25_px, margin=1.5):
    """The slice of a frame worth working on, as (y0, y1, x0, x1).

    Returns None when there is no radius to work from, or when the window would
    cover essentially the whole frame anyway - in which case cropping would add a
    bookkeeping step and save nothing.
    """
    if not r25_px or r25_px <= 0:
        return None

    ny, nx = shape
    reach = margin * r25_px
    x0 = int(max(0, np.floor(centre[0] - reach)))
    x1 = int(min(nx, np.ceil(centre[0] + reach)))
    y0 = int(max(0, np.floor(centre[1] - reach)))
    y1 = int(min(ny, np.ceil(centre[1] + reach)))

    if x1 - x0 < 64 or y1 - y0 < 64:
        return None
    if (x1 - x0) * (y1 - y0) > 0.9 * nx * ny:
        return None
    return (y0, y1, x0, x1)

def cut(data, window):
    """The window's worth of a frame."""
    if window is None or data is None:
        return data
    y0, y1, x0, x1 = window
    return data[y0:y1, x0:x1]

def shift_centre(centre, window):
    """Where the galaxy's centre sits once the frame has been cut."""
    if window is None:
        return centre
    y0, _, x0, _ = window
    return (centre[0] - x0, centre[1] - y0)

def shift_positions(positions, window, shape=None):
    """Star positions moved into the cut frame, keeping only those inside it."""
    if window is None or positions is None or not len(positions):
        return positions
    y0, y1, x0, x1 = window
    p = np.asarray(positions, dtype=float)
    inside = ((p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1))
    return np.column_stack([p[inside, 0] - x0, p[inside, 1] - y0])

def restore_positions(x, y, window):
    """Positions measured in the cut frame, put back where they belong.

    Everything the tool reports - catalogues, plots, the pairing with the blue
    knots - is in the coordinates of the original frame, so measurements made in
    the window have to come back out of it. Forgetting this would shift every
    position by the size of the crop, which is the kind of error that looks like
    a discovery.
    """
    if window is None:
        return x, y
    y0, _, x0, _ = window
    return np.asarray(x, dtype=float) + x0, np.asarray(y, dtype=float) + y0

def describe(window, shape, margin):
    """What was cut, and what it saved."""
    if window is None:
        return ("  working on the whole frame - either no isophotal radius is "
                "known, or the galaxy fills it")
    y0, y1, x0, x1 = window
    kept = (y1 - y0) * (x1 - x0)
    whole = shape[0] * shape[1]
    return ("  working on {} x {} pixels around the galaxy, {:.0f}% of the frame "
            "({:.1f} x R25); the rest is sky that would be discarded anyway"
            .format(x1 - x0, y1 - y0, 100.0 * kept / whole, margin))

describe__galaxy_window = describe   # this module's own; the plain name gets shadowed below


# --------------------------------------------------------------------------
# shared module galaxy_extent.py
# --------------------------------------------------------------------------

def surface_brightness_profile(data, centre, r_max=None, n_rings=40, step=4,
                               percentile=90.0):
    """How much light each ring holds, above what an empty ring holds.

    WHY A HIGH PERCENTILE AND NOT THE MEDIAN
    Because of what is being measured. A stellar disk is smooth, and its median
    describes it. Line emission is not smooth: it lives in discrete regions with
    empty sky between them, so a ring holding a dozen bright HII regions and
    twenty thousand empty pixels has a median of zero. Measured on M83, the median
    of the subtracted frame reached the noise at 256 px while the regions
    themselves went on to 556 - and using the median cut the catalogue from 335
    regions to 122.

    The ninetieth percentile asks the useful question instead: is there anything
    bright in this ring. It works for a smooth tracer too, where it simply tracks
    the median with an offset.

    WHY THE FAR FIELD IS SUBTRACTED
    A high percentile of pure noise is not zero - it is about 1.3 times the noise,
    because it is picking the brightest tenth of the sample. Comparing against
    zero would call empty sky "still galaxy" for ever. So the same statistic is
    measured far out, where there is nothing, and that floor is taken off.

    Sampled every few pixels rather than every one: this is a shape measurement
    over thousands of pixels a ring, and reading a quarter of them changes nothing
    while making it quick enough for a 61-megapixel frame.
    """
    from astropy.stats import sigma_clipped_stats

    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny:step, 0:nx:step]
    r = np.hypot(xx - centre[0], yy - centre[1]).ravel()
    v = np.asarray(data[::step, ::step], dtype=float).ravel()
    ok = np.isfinite(v)
    r, v = r[ok], v[ok]
    if not len(r):
        return None

    if r_max is None:
        r_max = float(np.percentile(r, 99))
    far = r > 0.8 * r.max()
    if far.sum() < 500:
        far = r > np.percentile(r, 80)
    sky = float(np.median(v[far]))
    noise = float(sigma_clipped_stats(v[far], sigma=3.0)[2])
    if not np.isfinite(noise) or noise <= 0:
        return None

    # What this statistic reads on empty sky. Everything is measured against it.
    floor = float(np.percentile(v[far], percentile))

    edges = np.linspace(0.0, r_max, n_rings + 1)
    centres, excess = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if m.sum() < 50:
            continue
        centres.append(0.5 * (a + b))
        excess.append((float(np.percentile(v[m], percentile)) - floor) / noise)
    return np.array(centres), np.array(excess), sky, noise

def visible_radius(data, centre, threshold=0.5, r_max=None, percentile=90.0):
    """Where the galaxy's light falls below a fraction of the sky noise.

    `threshold` is in units of that noise, and it is the one real choice here.
    Half a sigma is the default because the two errors are not symmetric: too
    large a radius lets empty sky into the measurement and dilutes it, which is
    what was happening, while too small a radius deletes the outer regions - and
    the outskirts are where a radial profile is most sensitive, since the density
    is lowest there. Half a sigma sits where the profile has clearly flattened but
    has not yet reached zero.

    Returns (radius in pixels, diagnostics), or (None, diagnostics) when the
    profile never rises above the threshold at all - a frame with no galaxy in it,
    or a centre in the wrong place.
    """
    got = surface_brightness_profile(data, centre, r_max=r_max,
                                     percentile=percentile)
    if got is None:
        return None, {"why": "could not measure a background to compare against"}
    radii, excess, sky, noise = got

    above = excess > threshold
    if not above.any():
        return None, {"why": "nothing rises above {:.1f} sigma of the sky noise "
                             "anywhere".format(threshold),
                      "sky": sky, "noise": noise}

    # The last ring above the threshold, walking outwards - not the first one
    # below it, so that a single noisy ring in the middle does not cut the galaxy
    # in half.
    last = int(np.where(above)[0].max())
    radius = float(radii[last])
    return radius, {"sky": sky, "noise": noise, "threshold": threshold,
                    "central_excess": float(excess[0]) if len(excess) else float("nan"),
                    "radii": radii, "excess": excess}

def describe(radius, diag, r25_px=None, label=""):
    """What was measured, next to what the catalogue says."""
    if radius is None:
        return "  {}: could not measure an extent ({})".format(
            label, diag.get("why", "unknown"))
    line = ("  {}: the light falls below {:.1f} x the sky noise at {:.0f} px"
            .format(label, diag.get("threshold", 0.5), radius))
    if r25_px:
        line += "; the catalogue's R25 is {:.0f} px".format(r25_px)
        if radius < 0.8 * r25_px:
            line += ("\n    the exposure does not reach the catalogue radius, so "
                     "the measured one is used")
        elif radius > 1.2 * r25_px:
            line += ("\n    the light reaches past the catalogue radius; the "
                     "catalogue one is used, being the more standard")
    return line

def working_radius(measured, r25_px):
    """The radius to actually use: the smaller of what is there and what is published.

    Smaller of the two on purpose. If the exposure does not reach R25 then the
    space between is empty sky and including it only adds noise. If the light
    reaches past R25 - which happens for a bright inner disk measured generously -
    then R25 is the standard definition and there is no reason to invent a wider
    one.
    """
    if measured and r25_px:
        return min(measured, r25_px)
    return measured or r25_px

describe__galaxy_extent = describe   # this module's own; the plain name gets shadowed below


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
# shared module continuum_subtraction.py
# --------------------------------------------------------------------------

def _aperture_fluxes(frames, positions, r=6.0, r_in=10.0, r_out=16.0):
    """Background-subtracted aperture flux at the same positions in several frames."""
    from photutils.aperture import (CircularAperture, CircularAnnulus,
                                    aperture_photometry)

    ap = CircularAperture(positions, r=r)
    ann = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
    out = []
    for img in frames:
        phot = aperture_photometry(img, [ap, ann])
        bkg_mean = phot["aperture_sum_1"] / ann.area
        out.append(np.asarray(phot["aperture_sum_0"] - bkg_mean * ap.area, dtype=float))
    return out

def find_reference_stars(frame, mask=None, n_stars=300, threshold_sigma=25.0):
    """Bright, isolated point sources to calibrate against."""
    from photutils.detection import DAOStarFinder
    from astropy.stats import sigma_clipped_stats

    mean, median, std = sigma_clipped_stats(frame, sigma=3.0, mask=mask)
    finder = DAOStarFinder(fwhm=4.0, threshold=threshold_sigma * std)
    tbl = finder(frame - median, mask=mask)
    if tbl is None or len(tbl) == 0:
        return np.empty((0, 2))
    tbl.sort("flux")
    tbl = tbl[::-1][:n_stars]
    return np.transpose((np.asarray(tbl["xcentroid"], dtype=float),
                         np.asarray(tbl["ycentroid"], dtype=float)))

def constant_factor(ha, red, mask=None, n_stars=200, positions=None):
    """One factor for the whole frame, taken from the stars.

    Returns the factor to multiply H-alpha by before subtracting R, and the
    star-to-star scatter, which is the honest uncertainty on it.
    """
    pos = (np.asarray(positions, dtype=float)[:n_stars]
           if positions is not None and len(positions) >= 10
           else find_reference_stars(red, mask=mask, n_stars=n_stars))
    if len(pos) < 5:
        raise RuntimeError("too few stars found to set the continuum factor")

    f_ha, f_r = _aperture_fluxes([ha, red], pos)
    ok = np.isfinite(f_ha) & np.isfinite(f_r) & (f_ha > 0)
    if ok.sum() < 5:
        raise RuntimeError("no usable stars for the continuum factor")

    ratio = f_r[ok] / f_ha[ok]
    med = float(np.median(ratio))
    spread = 1.4826 * np.median(np.abs(ratio - med))
    if spread > 0:
        ratio = ratio[np.abs(ratio - med) < 3.0 * spread]

    factor = float(np.median(ratio))
    scatter = float(1.4826 * np.median(np.abs(ratio - factor)))
    return factor, {
        "n_stars": int(ratio.size),
        "scatter": scatter,
        "relative_scatter": scatter / factor if factor else float("nan"),
        "method": "one constant for the whole frame",
    }

def colour_map(b, v, smooth_px=6.0, min_signal=None):
    """An instrumental B minus V value for every pixel, smoothed to be usable.

    Colour is a ratio of two faint things, so it is far noisier than either frame
    on its own. Both frames are smoothed before the ratio is taken - the colour of
    a star-forming complex varies on scales much larger than a pixel, so this costs
    nothing real and makes the map usable.

    Returns the colour map and a boolean array marking where it can be trusted.
    """
    from scipy.ndimage import gaussian_filter
    from astropy.stats import sigma_clipped_stats

    bs = gaussian_filter(np.nan_to_num(b, nan=0.0), smooth_px)
    vs = gaussian_filter(np.nan_to_num(v, nan=0.0), smooth_px)

    if min_signal is None:
        _, _, std_v = sigma_clipped_stats(v, sigma=3.0)
        min_signal = 3.0 * std_v / max(smooth_px, 1.0)

    usable = (bs > min_signal) & (vs > min_signal) & np.isfinite(bs) & np.isfinite(vs)
    with np.errstate(divide="ignore", invalid="ignore"):
        colour = np.where(usable, -2.5 * np.log10(np.where(usable, bs / vs, 1.0)), np.nan)
    return colour, usable

def colour_dependent_factor(ha, red, b, v, mask=None, n_stars=300, smooth_px=6.0,
                            positions=None):
    """A factor that varies across the frame with the colour of what is there.

    Follows the recommendation of Spector, Finkelman and Brosch (2012): measure the
    factor star by star, measure each star's colour, fit a straight line through
    them, and then read the factor off that line for every part of the image
    according to its own colour.

    WHERE IT FALLS BACK
    The colour of a pixel is only measurable where both broad bands have signal.
    In the empty sky between the arms they do not, and there the constant factor is
    used instead. The diagnostics report what fraction of the frame got which, so a
    run where the colour map barely applied does not look like one where it did.

    Returns (factor_map, diagnostics). factor_map has the same shape as the frames.
    """
    # Catalogue positions when they are available: the relation being fitted is
    # the one foreground stars follow, and a nearby galaxy's own stars sitting in
    # the sample are what makes the fit fail rather than merely scatter.
    pos = (np.asarray(positions, dtype=float)[:n_stars]
           if positions is not None and len(positions) >= 10
           else find_reference_stars(v, mask=mask, n_stars=n_stars))
    if len(pos) < 8:
        raise RuntimeError("too few stars to fit a colour relation")

    f_ha, f_r, f_b, f_v = _aperture_fluxes([ha, red, b, v], pos)
    ok = (np.isfinite(f_ha) & np.isfinite(f_r) & np.isfinite(f_b) & np.isfinite(f_v)
          & (f_ha > 0) & (f_b > 0) & (f_v > 0))
    if ok.sum() < 8:
        raise RuntimeError("too few stars with good flux in all four frames")

    ratio = f_r[ok] / f_ha[ok]
    colour = -2.5 * np.log10(f_b[ok] / f_v[ok])

    # Clip obvious outliers in both directions before fitting: saturated stars, and
    # any star that does have line emission and so is not pure continuum.
    good = np.isfinite(ratio) & np.isfinite(colour)
    for _ in range(3):
        if good.sum() < 6:
            break
        med_r = np.median(ratio[good])
        spread = 1.4826 * np.median(np.abs(ratio[good] - med_r))
        if spread <= 0:
            break
        good &= np.abs(ratio - med_r) < 3.0 * spread

    if good.sum() < 6:
        raise RuntimeError("not enough clean stars left to fit the colour relation")

    slope, intercept = np.polyfit(colour[good], ratio[good], 1)
    predicted = np.polyval([slope, intercept], colour[good])
    resid = ratio[good] - predicted
    ss_tot = float(np.sum((ratio[good] - ratio[good].mean()) ** 2))
    r2 = float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan")

    const = float(np.median(ratio[good]))
    const_scatter = float(1.4826 * np.median(np.abs(ratio[good] - const)))
    fit_scatter = float(1.4826 * np.median(np.abs(resid)))

    c_lo, c_hi = np.percentile(colour[good], [2, 98])

    # Only use the fitted relation if it actually describes the stars better than a
    # single number does. Fitting a slope through a handful of stars spanning a
    # narrow range of colour can easily come out worse than the constant it was
    # meant to improve on, and a method that is better in principle but measurably
    # worse on this data should not be the one that runs.
    improves = (fit_scatter < const_scatter) and (r2 >= 0.30) and (good.sum() >= 12)
    reason = ""
    if not improves:
        bits = []
        if good.sum() < 12:
            bits.append("only {} clean stars".format(int(good.sum())))
        if r2 < 0.30:
            bits.append("the fit explains little of the scatter (R2 = {:.2f})".format(r2))
        if fit_scatter >= const_scatter:
            bits.append("the fitted line scatters more than a constant "
                        "({:.4f} against {:.4f})".format(fit_scatter, const_scatter))
        if c_hi - c_lo < 0.5:
            bits.append("the stars span only {:.2f} mag of colour".format(c_hi - c_lo))
        reason = "; ".join(bits)

    cmap, usable = colour_map(b, v, smooth_px=smooth_px)
    c_clipped = np.clip(cmap, c_lo, c_hi)

    factor_map = np.full(ha.shape, const, dtype=float)
    applied = np.zeros(ha.shape, dtype=bool)
    if improves:
        # Never extrapolate the relation beyond the colours the stars actually
        # spanned; outside that range the straight line is a guess, not a measurement.
        applied = usable & np.isfinite(c_clipped)
        factor_map[applied] = np.polyval([slope, intercept], c_clipped[applied])

    return factor_map, {
        "method": ("factor fitted against colour, per Spector et al. 2012"
                   if improves else
                   "one constant - the colour fit was tried and rejected"),
        "used_colour": bool(improves),
        "rejected_because": reason,
        "n_stars": int(good.sum()),
        "slope": float(slope),
        "intercept": float(intercept),
        "fit_r2": r2,
        "colour_range": (float(c_lo), float(c_hi)),
        "constant_factor": const,
        "constant_scatter": const_scatter,
        "relative_scatter": const_scatter / const if const else float("nan"),
        "fit_scatter": fit_scatter,
        "scatter_reduction": (const_scatter / fit_scatter) if fit_scatter > 0 else float("nan"),
        "fraction_colour_applied": float(applied.mean()),
        "factor_min": float(np.nanmin(factor_map)),
        "factor_max": float(np.nanmax(factor_map)),
    }

def describe(diag):
    """A readable summary of whichever method was used."""
    lines = ["  method: " + diag.get("method", "?"),
             "  stars used: {}".format(diag.get("n_stars"))]
    if diag.get("rejected_because"):
        lines.append("  the colour fit was rejected because " + diag["rejected_because"])
        lines.append("  falling back to a constant factor of {:.3f}".format(
            diag["constant_factor"]))
    if "slope" in diag:
        lines += [
            "  factor = {:.4f} + {:.4f} x (B-V instrumental),  R2 = {:.2f}".format(
                diag["intercept"], diag["slope"], diag["fit_r2"]),
            "  colours spanned by the stars: {:.2f} to {:.2f}".format(*diag["colour_range"]),
            "  factor varies across the frame from {:.3f} to {:.3f}".format(
                diag["factor_min"], diag["factor_max"]),
            "  a single constant would have been {:.3f}".format(diag["constant_factor"]),
            "  scatter about a constant {:.4f}, about the fitted line {:.4f} "
            "- the line is {:.0%} {}".format(
                diag["constant_scatter"], diag["fit_scatter"],
                abs(diag["fit_scatter"] / diag["constant_scatter"] - 1.0)
                if diag["constant_scatter"] else float("nan"),
                "tighter" if diag["fit_scatter"] < diag["constant_scatter"] else "wider"),
            "  the colour relation was applied to {:.0%} of the frame; the rest fell "
            "back to the constant".format(diag["fraction_colour_applied"]),
        ]
    else:
        lines.append("  factor {:.4f}, star-to-star scatter {:.1%}".format(
            diag.get("constant_factor", float("nan")),
            diag.get("relative_scatter", float("nan"))))
    return "\n".join(lines)

describe__continuum_subtraction = describe   # this module's own; the plain name gets shadowed below


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
# borrowed from HII_From_Images.py
# --------------------------------------------------------------------------

matplotlib.use("Agg")

warnings.filterwarnings("ignore")

pass  # from radial_profiles, inlined above

pass  # from interactive_input, inlined above

plan_folder = plan; describe_folder = describe__frame_inventory  # from frame_inventory, inlined above

plan_folder = plan; describe_folder = describe__frame_inventory  # from frame_inventory, inlined above

pass  # from galaxy_catalogue, inlined above

pass  # from deprojection, inlined above

pass  # from anchor_registration, inlined above

pass  # from galaxy_window, inlined above

describe_window = describe__galaxy_window  # from galaxy_window, inlined above

pass  # from galaxy_extent, inlined above

describe_extent = describe__galaxy_extent  # from galaxy_extent, inlined above

describe_anchors = describe__anchor_registration  # from anchor_registration, inlined above

def frames_from_folder(folder, roles):
    """Work out which file in a folder plays each part, and say so before starting.

    Returns three things per role: the chosen frame, the solved exposure that will
    lend it a plate solution if it has none, and the list of exposures to combine
    where the folder held several and nothing combined.
    """
    def pick(role, options):
        print("")
        print("More than one frame could be the " + role + ":")
        for i, opt in enumerate(options, 1):
            print("  {}. {}".format(i, opt["name"]))
        while True:
            answer = ask("Which one", "1", int)
            if 1 <= answer <= len(options):
                return options[answer - 1]
            print("    pick a number from the list")

    records, chosen, members, donors, notes = plan_folder(folder, roles=roles,
                                                          ask=pick)
    print(describe_folder(records, chosen, notes))

    missing = [r for r in roles if r not in chosen]
    if missing:
        raise SystemExit("Nothing in the folder is the " + " or the ".join(missing))

    # Where the folder holds several exposures of a filter and nothing combined,
    # say what combining them is worth and let the person decide. The gain is a
    # real number, so it is put on the screen rather than hidden behind a flag
    # nobody will remember a year from now.
    to_combine = {}
    for role in roles:
        group = members.get(role) or []
        if len(group) < 2:
            continue
        if ask_yes_no("There are {} {} exposures and no combined frame. Combining "
                      "them lowers the noise by about a factor of {:.1f}. "
                      "Combine?".format(len(group), role, len(group) ** 0.5),
                      True):
            to_combine[role] = [r["path"] for r in group]
        else:
            print("  using {} on its own".format(group[0]["name"]))

    name, name_source = galaxy_from_records(records, folder)
    if name:
        print("Galaxy: {}   (from {})".format(name, name_source))

    return ({role: chosen[role]["path"] for role in roles},
            {role: (donors[role]["path"] if donors.get(role) else None)
             for role in roles},
            to_combine, name)

def load_frame(path, donor_path=None):
    """Open a FITS file and return the science image, its WCS and a bad-pixel mask.

    A frame that came out of a median combine usually has no plate solution left
    in it, because the combining step wrote a new file and kept almost nothing.
    When `donor_path` names a solved exposure of the same field, a solution is
    fitted for this frame by matching star patterns against that one, and the
    residual of that fit is printed so it can be judged rather than trusted.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    hdul = fits.open(path)
    sci = None
    for hdu in hdul:
        if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
            sci = hdu
            break
    if sci is None:
        hdul.close()
        raise ValueError(path + ": no 2-D image found")

    # Single precision on purpose. Camera data arrives as 16-bit integers, so
    # double precision doubles the memory for no information at all: seven
    # significant figures against pixel noise of tens of counts. On a 61-megapixel
    # frame the difference is 245 MB against 489, and four filters of those was
    # what drove this machine into swapping.
    data = np.asarray(sci.data, dtype=np.float32)
    header = sci.header

    bpm = None
    for hdu in hdul:
        if hdu.name.upper() == "BPM" and hdu.data is not None:
            bpm = np.asarray(hdu.data) > 0
            break

    wcs = None
    try:
        w = WCS(header)
        if w.has_celestial:
            wcs = w.celestial
    except Exception:
        pass

    info = {
        "filter": str(header.get("FILTER", "?")),
        "exptime": float(header.get("EXPTIME", 1.0) or 1.0),
        "pixel_scale": header.get("PIXSCALE", None),
        "object": str(header.get("OBJECT", "?")),
    }
    hdul.close()

    if wcs is None and donor_path:
        reference, ref_wcs, _, ref_info = load_frame(donor_path)
        if ref_wcs is not None:
            wcs, diag = wcs_from_reference(data, reference, ref_wcs)
            print(describe_wcs_fit(os.path.basename(path), diag))
            if info["pixel_scale"] is None:
                info["pixel_scale"] = ref_info["pixel_scale"]
            if info["filter"] == "?":
                info["filter"] = ref_info["filter"]

    return data, wcs, bpm, info

def _centroid_locally(data, positions, box):
    """Centre of light in a small box, with that box's own sky taken off first.

    The one place this is done, because getting it wrong is invisible. The sky is
    part of the weight in a centre of light, and in a box of a couple of hundred
    pixels it is most of the weight, so leaving it in returns the middle of the box
    dressed up as a measurement. Every caller here wants the star, not the box.

    Returns an array the same length as `positions`, with NaN where a position sits
    too near an edge or the stamp holds nothing above its own sky.
    """
    data = np.asarray(data)
    pos = np.asarray(positions, dtype=float)
    half = box // 2
    ny, nx = data.shape
    grid_y, grid_x = np.mgrid[0:box, 0:box]
    out = np.full(pos.shape, np.nan)
    for i, (xc, yc) in enumerate(pos):
        if not (np.isfinite(xc) and np.isfinite(yc)):
            continue
        xi, yi = int(round(xc)), int(round(yc))
        if not (half <= xi < nx - half and half <= yi < ny - half):
            continue
        stamp = np.nan_to_num(
            data[yi - half:yi + half + 1, xi - half:xi + half + 1]).astype(float)
        if stamp.shape != (box, box):
            continue
        stamp = stamp - np.median(stamp)
        stamp[stamp < 0.0] = 0.0        # negative noise must not pull the centre
        total = stamp.sum()
        if total <= 0.0:
            continue
        out[i] = [xi - half + (stamp * grid_x).sum() / total,
                  yi - half + (stamp * grid_y).sum() / total]
    return out

def centre_on_stars(data, positions, box=15, max_move_px=6.0):
    """Move catalogue positions onto the stars as this frame actually shows them.

    WHY THIS IS NEEDED
    A catalogue position and a star in a frame are not the same thing to better
    than a pixel or two. The plate solution has its own error, the catalogue is at
    a different epoch, and a combined frame carries whatever its registration left
    behind. Measured on an M33 R frame: the real stars sat a median of 2.4 pixels
    away from where Gaia said they were, and not by a constant amount - the spread
    was 2.2 pixels.

    That matters because most of what is done with these positions is aperture
    photometry, and an aperture is not moved by being wrong. A circle of radius 6
    placed 2.4 pixels off centre misses a good part of the star, and misses a
    different part of every star. The subtraction factor built from those fluxes
    is wrong, and the star widths measured around those points are not the widths
    of stars. Both of those were wrong here, and both showed in the picture.

    The one thing that was right was the frame registration, and only because it
    centroids inside a box before fitting - which is exactly what this does, once,
    for everyone else.

    WHY THE SKY IS TAKEN OFF FIRST
    A centre of light is a weighted average, and the sky is part of the weight. In
    a 15 pixel box on a frame where the sky sits at a few hundred counts, the two
    hundred sky pixels outweigh the star, and their centre of light is the middle
    of the box. So the answer comes back as very nearly the position it was given,
    whatever the star is doing - it looks like a successful measurement and is not
    one. Measured on M33: centroiding the raw stamp moved the positions 0.6 px and
    left the stacked star flat-topped, its brightest ring at radius 4; taking the
    local sky off the stamp first moved them 4.3 px and gave a star that falls from
    102 counts at the centre to 15 at radius 7, which is what a star looks like.

    The stamp is floored at zero after the sky comes off so that negative noise
    cannot pull the centre around.

    Positions that cannot be centred, or that move further than a plate solution
    plausibly errs, are dropped rather than trusted.
    """
    positions = np.asarray(positions, dtype=float)
    if not len(positions):
        return positions, {"n_in": 0, "n_out": 0}

    ny, nx = data.shape
    half = box // 2
    edge = box
    inside = ((positions[:, 0] > edge) & (positions[:, 0] < nx - edge) &
              (positions[:, 1] > edge) & (positions[:, 1] < ny - edge))
    p = positions[inside]
    if not len(p):
        return np.empty((0, 2)), {"n_in": len(positions), "n_out": 0,
                                  "why": "none of them fall inside the frame"}

    found = _centroid_locally(data, p, box)

    dx, dy = found[:, 0] - p[:, 0], found[:, 1] - p[:, 1]
    moved = np.hypot(dx, dy)
    good = np.isfinite(moved) & (moved < max_move_px)
    out = found[good]
    return out, {"n_in": int(len(positions)), "n_out": int(good.sum()),
                 "median_move": float(np.median(moved[good])) if good.any() else 0.0,
                 "median_dx": float(np.median(dx[good])) if good.any() else 0.0,
                 "median_dy": float(np.median(dy[good])) if good.any() else 0.0}

def describe_centring(diag):
    """What the centring moved, in a line."""
    if diag.get("why"):
        return "  catalogue positions left as they are ({})".format(diag["why"])
    return ("  {} of {} catalogue stars found and centred; they sat a median of "
            "{:.2f} px from the catalogue position ({:+.2f}, {:+.2f})".format(
                diag["n_out"], diag["n_in"], diag["median_move"],
                diag["median_dx"], diag["median_dy"]))

def combine_exposures(paths, donor_path=None, label="", band_rows=512,
                      positions=None):
    """Put several exposures of one filter onto one grid and take their median.

    WHY COMBINE AT ALL
    A night's observing of one filter is several exposures. Taking one and
    discarding the rest throws away most of what was gathered: six frames combined
    have noise lower by roughly the square root of six, a factor of 2.4, and it is
    the faintest regions - the ones near the detection limit, where the answer is
    actually decided - that appear or vanish on exactly that difference.

    WHY A MEDIAN
    Because it is robust, cheap, and the alternative is not worth what it costs.
    Measured on this data:

        six frames        noise    time     working memory
        median             1.16    0.45 s     1.4x the stack
        clipped mean       1.10    0.97 s     3.3x the stack
        plain average      2.60*   0.02 s     0.2x the stack
                           (* with one cosmic ray in a hundred pixels)

    The clipped mean buys five to seven percent in noise - about what a seventh
    exposure would give - for twice the time and two and a half times the memory.
    A plain average is not an option at all: one cosmic ray ruins the pixel. So
    the median it is, and there is no switch, because there is no case here where
    the other answer is the right one.

    Done a band of rows at a time, so the memory needed does not grow with the
    number of exposures: twenty frames combine in the same space as three.

    The first exposure sets the grid; the rest are brought to it through their
    plate solutions and then have the leftover sub-pixel shift measured from the
    stars and taken out, exactly as the separate filters are.
    """
    import numpy as np

    reference, ref_wcs, ref_bpm, ref_info = load_frame(paths[0], donor_path)
    if len(paths) == 1:
        return reference, ref_wcs, ref_bpm, ref_info

    print("Combining {} {} exposures".format(len(paths), label or "frames"))
    stack = [reference]
    total_exptime = float(ref_info.get("exptime") or 0.0)

    for path in paths[1:]:
        data, wcs, _, info = load_frame(path, donor_path)
        name = os.path.basename(path)

        # Four bright, widely separated stars fix the shift, the rotation and the
        # scale between two frames of the same field. Going through both plate
        # solutions instead costs a minute a frame and arrives at the same place.
        placed = False
        if positions is not None and len(positions):
            transform, adiag = anchor_transform(reference, data, positions)
            if transform is not None:
                data = apply_transform(data, transform, reference.shape)
                print("  " + name)
                print("  " + describe_anchors(adiag))
                placed = True

        if not placed:
            if wcs is not None and ref_wcs is not None:
                data = align_to(ref_wcs, reference.shape, data, wcs)
            elif data.shape != reference.shape:
                print("  skipping {} - no plate solution and a different "
                      "shape".format(name))
                continue
            data = refine_alignment(reference, data, "  " + name,
                                    positions=positions)

        stack.append(data)
        total_exptime += float(info.get("exptime") or 0.0)

    ny, nx = reference.shape
    combined = np.empty((ny, nx), dtype=np.float32)
    for y0 in range(0, ny, band_rows):
        y1 = min(y0 + band_rows, ny)
        band = np.stack([frame[y0:y1] for frame in stack], axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            combined[y0:y1] = np.nanmedian(band, axis=0)

    print("  median of {} exposures, total exposure {:.0f}s, noise lower by about "
          "a factor of {:.1f}".format(len(stack), total_exptime, len(stack) ** 0.5))

    info = dict(ref_info)
    info["exptime"] = total_exptime
    return combined, ref_wcs, ref_bpm, info

def output_prefix(given, fallback_dir, galaxy):
    """Turn what the person typed into a stem for the output files.

    A folder is accepted as well as a stem, because a folder is what people type:
    asked where to put the output, three runs in a row were given the path of the
    frames' own folder, and the files landed beside it named after it. If what was
    given is a directory, the files go inside it, named after the galaxy.
    """
    name = (galaxy or "galaxy").replace(" ", "_")
    if given:
        given = str(given).strip().strip('"').strip("'")
        if os.path.isdir(given):
            return os.path.join(given, name)
        parent = os.path.dirname(given)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        return given
    return os.path.join(fallback_dir or ".", name)

def save_frame(path, data, wcs=None, info=None, history=()):
    """Write a frame this tool made, with a header worth having.

    The continuum-subtracted image is the one the whole measurement rests on, and
    until now it existed only inside a plot. Anyone wanting to check the
    subtraction by eye - which is exactly what the tool asks for when the stars
    disagree about the factor - had no file to open. It carries the plate solution
    so it lines up with the originals in any viewer, and a history saying what it
    is and how it was made.
    WHAT IS DONE TO THE PIXELS BEFORE THEY ARE WRITTEN
    Blank pixels are written as zero rather than as NaN. NaN is legal FITS and
    astropy reads it happily, but it is a floating-point value that older readers
    predate: AIP4Win would not open a frame written from here, and a subtracted
    M83 frame carried nine thousand of them, at the edges the resampling could not
    reach. A file that only this pipeline can open is not much of an output.

    WHY THE PLATE SOLUTION IS WRITTEN TWICE
    astropy writes a rotation as a PC matrix with a separate CDELT scale. That is
    the current convention; the older one is a single CD matrix, and a reader that
    knows only CD sees a frame with no usable coordinates. Both are written - they
    say the same thing, and CD = PC x CDELT is exact - so either kind of reader
    finds what it expects.
    """
    from astropy.io import fits

    header = wcs.to_header() if wcs is not None else fits.Header()
    if wcs is not None:
        try:
            cd = np.asarray(wcs.pixel_scale_matrix, dtype=float)
            for i in (1, 2):
                for j in (1, 2):
                    header["CD{}_{}".format(i, j)] = float(cd[i - 1, j - 1])
        except Exception:
            pass
    if info:
        if info.get("filter"):
            header["FILTER"] = info["filter"]
        if info.get("exptime"):
            header["EXPTIME"] = float(info["exptime"])
        if info.get("object"):
            header["OBJECT"] = info["object"]
        if info.get("pixel_scale"):
            header["PIXSCALE"] = float(info["pixel_scale"])
    pixels = np.asarray(data, dtype="float32")
    blank = ~np.isfinite(pixels)
    if blank.any():
        pixels = np.where(blank, np.float32(0.0), pixels)
        header["HISTORY"] = ("{} blank pixels written as zero, not NaN, so that "
                             "older readers can open this".format(int(blank.sum())))
    for line in history:
        header["HISTORY"] = line

    fits.PrimaryHDU(data=pixels, header=header).writeto(path, overwrite=True)
    return path

def image_orientation(wcs):
    """Angle of North in the image, and whether the parity is flipped.

    Reading this from the WCS removes a whole class of silent error. Assuming
    north is up when it is not rotates the deprojection by an unknown amount, and
    nothing in the output looks wrong.
    """
    cd = wcs.pixel_scale_matrix
    north = math.degrees(math.atan2(cd[1, 1], cd[0, 1])) % 360.0
    east = math.degrees(math.atan2(cd[1, 0], cd[0, 0])) % 360.0
    mirrored = ((east - north) % 360.0) > 180.0
    return north, mirrored

def frame_geometry(path, donor_path=None):
    """Where a frame points and how big it is, without reading its pixels.

    The catalogue lookups all need this - the galaxy's centre in this frame, the
    stars in front of it, the isophotal radius in pixels - and they are needed
    before the frames are combined, not after. Reading a header is instant, while
    reading a 61-megapixel image is not, so the header is tried first. A frame
    that lost its plate solution in a combining step has to be opened properly,
    since its solution is fitted from a donor exposure.

    Returns (wcs, shape, info).
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    try:
        with fits.open(path) as hdul:
            for hdu in hdul:
                header = hdu.header
                if header.get("NAXIS") == 2:
                    w = WCS(header)
                    if w.has_celestial:
                        return (w.celestial,
                                (header["NAXIS2"], header["NAXIS1"]),
                                {"pixel_scale": header.get("PIXSCALE", None),
                                 "filter": str(header.get("FILTER", "?")),
                                 "object": str(header.get("OBJECT", "?"))})
    except Exception:
        pass

    data, wcs, _, info = load_frame(path, donor_path)
    return wcs, data.shape, info

def same_grid(wcs_a, wcs_b, shape, tolerance_px=1.5):
    """True when two plate solutions put the sky in the same pixels already.

    WHY ASK
    Resampling a frame is the most expensive thing this tool does to it: two
    astrometric transforms of every pixel position, tens of millions of them. It
    is worth it when the frames really sit on different grids. It is pure waste
    when they do not - and between consecutive exposures of one filter, they
    almost never do. Measured on a night of M33: the exposures of a filter sat
    within 0.05 to 0.7 of a pixel of each other.

    So the question is asked before the work is done. A few corner and centre
    points are pushed through both solutions; if none of them moves by more than
    about a pixel, the grids are the same and a sub-pixel shift is all that is
    needed. The full resampling stays for the case it was written for.
    """
    import numpy as np

    if wcs_a is None or wcs_b is None:
        return False
    ny, nx = shape
    xs = np.array([0, nx - 1, 0, nx - 1, nx // 2, nx // 4, 3 * nx // 4],
                  dtype=float)
    ys = np.array([0, 0, ny - 1, ny - 1, ny // 2, 3 * ny // 4, ny // 4],
                  dtype=float)
    try:
        sky = wcs_a.pixel_to_world(xs, ys)
        bx, by = wcs_b.world_to_pixel(sky)
    except Exception:
        return False
    moved = np.hypot(bx - xs, by - ys)
    return bool(np.all(np.isfinite(moved)) and moved.max() <= tolerance_px)

def align_to(ref_wcs, ref_shape, data, wcs, band_rows=512):
    """Resample one frame onto another frame's pixel grid using both WCS solutions.

    Both frames are plate-solved, so every output pixel can be mapped to its true
    sky position and the input frame read there. Exact, and it needs no matching of
    stars between the two images.

    Done a few hundred rows at a time. The whole-frame version built five arrays
    the size of the image to hold the coordinates - about 2.5 GB on a 61-megapixel
    frame, for every exposure - which is memory spent on scratch paper. The
    arithmetic is identical; only the order changed.
    """
    from scipy.ndimage import map_coordinates

    ny, nx = ref_shape
    out = np.empty((ny, nx), dtype=np.float32)
    for y0 in range(0, ny, band_rows):
        y1 = min(y0 + band_rows, ny)
        yy, xx = np.mgrid[y0:y1, 0:nx]
        sky = ref_wcs.pixel_to_world(xx, yy)
        sx, sy = wcs.world_to_pixel(sky)
        out[y0:y1] = map_coordinates(data, [sy, sx], order=1,
                                     mode="constant", cval=np.nan)
    return out

def measure_fwhm(data, mask=None, max_sources=200, positions=None):
    """Width of the median star in a frame, in pixels, at half its height.

    WHY THE MEDIAN STAR AND NOT THE MEDIAN OF THE STARS
    Because a single faint star is mostly noise, and every way of reducing one to
    a number is pulled about by that noise. Stacking two hundred of them first
    gives one clean star to measure, and its width is the width the frame has.

    WHY NOT THE SECOND MOMENT
    That is what this did before, and it gave the wrong answer in the way that
    matters most: it got the order of the two filters backwards. A second moment
    weights each pixel by the square of its distance, so the outer ring of a 19
    pixel stamp counts about a hundred and seventy times as heavily as the centre.
    Whatever is out there that is not the star - the galaxy's own light under it,
    a neighbour, sky left over because only the frame's global level had been
    taken off - is therefore most of the answer. On M33 that made the H-alpha
    frame measure 5.53 px against R's 5.77, so the tool blurred H-alpha; stacking
    the same stars shows H-alpha at 4.91 and R at 3.14, so H-alpha was already the
    broader of the two and blurring it made the mismatch worse. Every star then
    subtracted to a black core inside a bright ring, which is exactly what the
    frames looked like.

    Half-maximum is read straight off the radial profile of the stack. It is a
    property of the core, where the star is, and it does not care what the stamp's
    outskirts contain.

    THREE THINGS THE STACK NEEDS
    The local sky comes off each stamp separately, taken from its border ring, so
    a star on a bright arm and a star on empty sky contribute the same star.
    Each stamp is shifted onto a common centre before stacking, to a fraction of a
    pixel, or the stack would be smeared by up to half a pixel of rounding - and
    smeared differently in each filter, which is the error being looked for.
    Saturated stars are dropped: a flat top is wider than the star that made it,
    and it is the brightest stars that saturate, so keeping them would bias the
    frame with the deeper exposure. A core with three or more pixels within two
    percent of its peak is flat, and a star's core is not.
    """
    from astropy.stats import sigma_clipped_stats
    from scipy.ndimage import shift as _shift

    half = 10
    ny, nx = data.shape
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    rr = np.hypot(xx, yy)
    border = rr > half - 1.5

    if positions is not None and len(positions) >= 10:
        candidates = np.asarray(positions, dtype=float)
    else:
        from photutils.detection import DAOStarFinder
        median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)[1:]
        finder = DAOStarFinder(fwhm=4.0, threshold=20.0 * std)
        tbl = finder(data - median, mask=mask)
        if tbl is None or len(tbl) == 0:
            return None
        tbl.sort("flux")
        tbl = tbl[::-1][:max_sources]
        candidates = np.column_stack([tbl["xcentroid"], tbl["ycentroid"]])

    found = []
    for x, y in candidates:
        xi, yi = int(round(x)), int(round(y))
        if not (half < xi < nx - half and half < yi < ny - half):
            continue
        stamp = np.nan_to_num(
            data[yi - half:yi + half + 1, xi - half:xi + half + 1]).astype(float)
        stamp = stamp - np.median(stamp[border])
        core = stamp[half - 1:half + 2, half - 1:half + 2]
        peak = float(core.max())
        if peak <= 0 or (core >= 0.98 * peak).sum() >= 3:
            continue
        found.append((peak, _shift(stamp, (yi - y, xi - x), order=1, mode="nearest")))
    if len(found) < 10:
        return None

    found.sort(key=lambda pair: -pair[0])
    stacked = np.median(np.stack([s for _, s in found[:max_sources]]), axis=0)
    profile = np.array([stacked[(rr >= q) & (rr < q + 1)].mean()
                        for q in range(half + 1)])
    if profile[0] <= 0:
        return None
    for q in range(1, half + 1):
        if profile[q] < 0.5 * profile[0]:
            below, above = profile[q - 1], profile[q]
            return float(2.0 * (q - 1 + (0.5 * profile[0] - below) / (above - below)))
    return None

def match_psf(sharp, fwhm_sharp, fwhm_blunt):
    """Blur the sharper frame so both frames have the same star size.

    Subtracting frames of different sharpness leaves a ring around every star - a
    bright core with a dark halo, or the reverse - and those rings are then detected
    as HII regions. Matching the widths first is what makes stars disappear cleanly.
    """
    from scipy.ndimage import gaussian_filter

    if fwhm_blunt is None or fwhm_sharp is None or fwhm_blunt <= fwhm_sharp:
        return sharp, 0.0
    sigma = math.sqrt(max(fwhm_blunt ** 2 - fwhm_sharp ** 2, 0.0)) / 2.3548
    return gaussian_filter(sharp, sigma), sigma

def _stack_stars(data, positions, max_sources=200, half=10):
    """The median star of a frame: local sky off, sub-pixel aligned, no saturation.

    One star is mostly noise. Two hundred of them stacked is one clean star, and
    every question about the shape of this frame's stars is asked of that.
    """
    from scipy.ndimage import shift as _shift

    data = np.asarray(data)
    pos = np.asarray(positions, dtype=float)
    ny, nx = data.shape
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    border = np.hypot(xx, yy) > half - 1.5

    found = []
    for x, y in pos:
        xi, yi = int(round(x)), int(round(y))
        if not (half < xi < nx - half and half < yi < ny - half):
            continue
        stamp = np.nan_to_num(
            data[yi - half:yi + half + 1, xi - half:xi + half + 1]).astype(float)
        stamp = stamp - np.median(stamp[border])
        core = stamp[half - 1:half + 2, half - 1:half + 2]
        peak = float(core.max())
        if peak <= 0 or (core >= 0.98 * peak).sum() >= 3:
            continue
        found.append((peak, _shift(stamp, (yi - y, xi - x), order=1, mode="nearest")))
    if len(found) < 10:
        return None
    found.sort(key=lambda pair: -pair[0])
    return np.median(np.stack([s for _, s in found[:max_sources]]), axis=0)

def match_psf_on_stars(ha, red, positions, max_sigma=3.0, step=0.02, half=10):
    """Blur one frame until its stars have the same shape as the other's.

    WHY NOT MEASURE TWO WIDTHS AND SUBTRACT THEM IN QUADRATURE
    Because that is what match_psf does and it does not work here, for two reasons
    that compound. A width is one number squeezed out of a whole profile, so it
    throws away exactly the part that matters - a star can have the same half-width
    as another and a quite different core and skirt. And the quadrature formula is
    only exact for Gaussians; a real star has broader wings than a Gaussian, so the
    blur it prescribes is not the blur that matches the two.

    Measured on M83: the widths came out H-alpha 3.78 px against R 4.22, which
    prescribed blurring H-alpha by sigma 0.80. Afterwards the stacked star left a
    negative core inside a positive ring - the signature of H-alpha now being the
    broader of the two. The prescription had the sign of the answer wrong.

    WHAT THIS DOES INSTEAD
    It asks the question the picture asks. Both frames' stars are stacked, each
    profile is divided by its own light inside six pixels so that brightness drops
    out of the comparison and only shape is left, and then one frame is blurred by
    every sigma in a range while the two profiles are compared ring by ring. The
    sigma that makes them agree is the answer, and it is an answer about the whole
    profile rather than about one point on it.

    A positive sigma blurs H-alpha, a negative one blurs R; the search covers both
    so it cannot get the direction wrong.

    Returns (ha, red, sigma, diagnostics).
    """
    from scipy.ndimage import gaussian_filter

    if positions is None or len(positions) < 10:
        return ha, red, 0.0, {"why": "no star positions to measure on"}
    A0 = _stack_stars(ha, positions, half=half)
    R0 = _stack_stars(red, positions, half=half)
    if A0 is None or R0 is None:
        return ha, red, 0.0, {"why": "not enough usable stars to stack"}

    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    rr = np.hypot(xx, yy)
    rings = [(rr >= q) & (rr < q + 1) for q in range(half)]
    inner = rr < 6

    def shape_of(stamp):
        """The radial profile, divided by the light in the core - shape only."""
        total = stamp[inner].sum()
        if total <= 0:
            return None
        return np.array([stamp[m].mean() for m in rings]) / total

    def mismatch(sigma):
        a = gaussian_filter(A0, sigma) if sigma > 0 else A0
        r = gaussian_filter(R0, -sigma) if sigma < 0 else R0
        sa, sr = shape_of(a), shape_of(r)
        if sa is None or sr is None:
            return np.inf
        return float(((sa - sr) ** 2).sum())

    grid = np.arange(-max_sigma, max_sigma + step, step)
    scores = np.array([mismatch(s) for s in grid])
    best = float(grid[int(np.argmin(scores))])
    if abs(best) < step:
        best = 0.0

    diag = {"sigma": best,
            "mismatch_before": mismatch(0.0),
            "mismatch_after": mismatch(best),
            "blurred": ("H-alpha" if best > 0 else "R" if best < 0 else "neither")}
    if best > 0:
        ha = gaussian_filter(ha, best)
    elif best < 0:
        red = gaussian_filter(red, -best)
    return ha, red, abs(best), diag

def describe_psf_match(diag):
    """What the search settled on, and by how much it improved things."""
    if diag.get("why"):
        return "  point spread functions left alone ({})".format(diag["why"])
    if diag["blurred"] == "neither":
        return "  the stars already have the same shape in both frames"
    before, after = diag["mismatch_before"], diag["mismatch_after"]
    line = ("  blurred the {} frame by sigma = {:.2f} px, chosen by comparing the "
            "stacked star ring by ring".format(diag["blurred"], abs(diag["sigma"])))
    if before > 0:
        line += ("\n    the disagreement between the two profiles fell to {:.0f}% "
                 "of what it was".format(100.0 * after / before))
    return line

def continuum_factor(ha, red, mask=None, n_stars=200, positions=None):
    """How much to brighten the narrow frame before subtracting the broad one.

    A foreground star emits no H-alpha line worth speaking of, so whatever it puts
    into the narrow frame is pure continuum, and the ratio of its brightness in the
    two frames is the factor we want.

    "Foreground" is the word that matters, and it is why `positions` exists. Point
    sources found in the image are foreground only when the galaxy is far enough
    away for its own stars not to be resolved. In a nearby galaxy they are, in
    their thousands, and they are not pure continuum - measured on M33 the mixture
    gave a star-to-star scatter of 66 percent, which is no factor at all. Passing
    catalogue positions of stars with a measurable parallax settles the question
    physically instead of hoping distance settles it.

    The median of the per-star ratio is used rather than a fit, because the median
    is not dragged around by the few stars that do have line emission, or are
    saturated, or sit on a bright piece of the galaxy.

    Returns the factor and diagnostics - in particular the star-to-star scatter,
    which is the honest uncertainty on the subtraction.
    """
    from photutils.detection import DAOStarFinder
    from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
    from astropy.stats import sigma_clipped_stats

    if positions is not None and len(positions) >= 10:
        # A generous slice of the catalogue, then the choice is made on what these
        # frames actually recorded rather than on what the catalogue says. Gaia's
        # G magnitude is not this telescope's R, and a star that is respectable in
        # the catalogue can be a handful of counts through a narrow filter - where
        # the ratio it gives is not a measurement but a division by noise.
        pos = np.asarray(positions, dtype=float)[:4 * n_stars]
    else:
        mean_r, med_r, std_r = sigma_clipped_stats(red, sigma=3.0, mask=mask)
        finder = DAOStarFinder(fwhm=4.0, threshold=25.0 * std_r)
        stars = finder(red - med_r, mask=mask)
        if stars is None or len(stars) < 5:
            raise RuntimeError("too few stars found to set the continuum factor")

        stars.sort("flux")
        stars = stars[::-1][:n_stars]
        pos = np.transpose((stars["xcentroid"], stars["ycentroid"]))

    ap = CircularAperture(pos, r=6.0)
    ann = CircularAnnulus(pos, r_in=10.0, r_out=16.0)

    def net_flux(img):
        phot = aperture_photometry(img, [ap, ann])
        bkg_mean = phot["aperture_sum_1"] / ann.area
        return np.asarray(phot["aperture_sum_0"] - bkg_mean * ap.area, dtype=float)

    f_ha, f_r = net_flux(ha), net_flux(red)

    ok = np.isfinite(f_ha) & np.isfinite(f_r) & (f_ha > 0) & (f_r > 0)
    if ok.sum() < 5:
        raise RuntimeError("no usable stars for the continuum factor")

    # Keep the brightest, measured here. With a catalogue list this is what turns
    # a set of stars that are merely real into a set that is worth measuring.
    if ok.sum() > n_stars:
        order = np.argsort(np.where(ok, f_r, -np.inf))[::-1][:n_stars]
        keep_bright = np.zeros(ok.shape, dtype=bool)
        keep_bright[order] = True
        ok &= keep_bright

    ratio = f_r[ok] / f_ha[ok]
    med = float(np.median(ratio))
    spread = 1.4826 * np.median(np.abs(ratio - med))
    if spread > 0:
        ratio = ratio[np.abs(ratio - med) < 3.0 * spread]

    factor = float(np.median(ratio))
    scatter = float(1.4826 * np.median(np.abs(ratio - factor)))
    return factor, {
        "n_stars": int(ratio.size),
        "scatter": scatter,
        "relative_scatter": scatter / factor if factor else float("nan"),
    }

def background_box(shape, requested=None, r25_px=None):
    """How large a square to estimate the background in.

    This is not a free parameter. The background map is meant to follow the sky
    and the galaxy's own smooth light, and to leave the regions sitting on top of
    it alone. A box the size of a spiral arm does the opposite: it treats the arm
    as background and subtracts away the very thing the faint regions stand on.
    Measured on the M83 frames, going from a 64-pixel box to a 200-pixel one found
    79 more regions inside the galaxy while the count out in the empty field, where
    nothing real can be, did not move at all - so those 79 were real and were being
    erased.

    A twelfth of the frame is wide enough to be well clear of any single region and
    small enough to still follow a gradient across the field.
    """
    if requested:
        return int(requested)
    # Tied to the galaxy, not to the frame. A box set as a fraction of the image
    # makes the answer depend on how much empty sky happened to surround the
    # galaxy that night - and it showed: cropping the frame to the galaxy, which
    # should have changed nothing but the running time, moved the region count on
    # M83 from 348 to 282 purely because the box shrank with the frame.
    #
    # A third of R25 is what the measurement pointed to. On M83, R25 is 560 px and
    # the box that worked best was 200.
    if r25_px:
        return int(min(max(r25_px / 3.0, 64), 400))
    return int(min(max(min(shape) // 12, 64), 256))

def detect_regions(net, threshold_sigma=4.0, min_pixels=6, mask=None, deblend=True,
                   box_size=None, nlevels=64, contrast=1e-4, r25_px=None):
    """Find HII regions in the continuum-subtracted frame.

    HII regions are extended blobs, not points, so connected groups of pixels above
    a threshold are grown rather than stellar profiles fitted. Touching blobs are
    separated by deblending, which is what stops one bright complex being counted
    once where a person clicking by eye would have counted three.

    threshold_sigma is in units of the local background noise, so the same number
    means the same thing on any frame. It is the most consequential choice in the
    whole pipeline, which is why it is written into every output file.
    """
    from photutils.background import Background2D, MedianBackground
    from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog
    from astropy.convolution import Gaussian2DKernel, convolve

    box = background_box(net.shape, box_size, r25_px)
    bkg = Background2D(net, box_size=box, filter_size=3,
                       bkg_estimator=MedianBackground(), mask=mask)
    resid = net - bkg.background

    kernel = Gaussian2DKernel(2.0)
    smoothed = convolve(resid, kernel, mask=mask)

    # The threshold has to be measured on the image it is applied to. Smoothing
    # averages neighbouring pixels, so it lowers the noise by about a factor of
    # seven for this kernel; comparing the smoothed image against the unsmoothed
    # noise - which is what the first version of this did - made a nominal "3
    # sigma" behave like twenty, and quietly threw away most of the faint regions.
    smooth_bkg = Background2D(smoothed, box_size=box, filter_size=3,
                              bkg_estimator=MedianBackground(), mask=mask)
    threshold = threshold_sigma * smooth_bkg.background_rms

    seg = detect_sources(smoothed, threshold, npixels=min_pixels, mask=mask)
    if seg is None:
        return None, bkg, resid

    if deblend:
        # Split rather than lump. A large complex is usually several regions that
        # happen to touch, and at this distance one blob can be hundreds of parsecs
        # across. Splitting it changes the count and the size distribution; it
        # barely moves the radial profile, because the pieces sit where the whole
        # sat. On the M83 frames these settings turned 277 blobs into 364 without
        # adding anything to the empty field.
        seg = deblend_sources(smoothed, seg, npixels=min_pixels,
                              nlevels=nlevels, contrast=contrast,
                              progress_bar=False)

    cat = SourceCatalog(resid, seg, mask=mask)
    return cat, bkg, resid

def offset_from_catalogue(reference, image, positions, box=11, max_shift_px=4.0):
    """The shift between two frames, measured on one agreed list of stars.

    WHY NOT JUST DETECT STARS IN BOTH
    Because the brightest three hundred sources in a narrow-band frame are not the
    brightest three hundred in a broad-band one - different filters see different
    stars - so two independently detected lists overlap poorly and the pairs that
    do match are the ones that happened to survive. On an M33 pair this left 21
    usable stars out of six hundred, with a scatter of two pixels.

    Given a list of positions from a catalogue, the question disappears: the same
    star is centroided in both frames, and every one of them contributes.

    WHY THE SKY COMES OFF EACH STAMP FIRST
    Same reason as in centre_on_stars, and the consequence here is worse. A centre
    of light computed with the sky still in the stamp is dominated by the sky, so
    it lands near the middle of the box whatever the star does - in both frames.
    The two answers then agree beautifully and their difference is close to zero,
    so this function reported "already aligned" on a pair that was a full pixel
    apart. On M83 that pixel is what left every star as a dark core beside a bright
    crescent in the subtracted frame.
    """
    if positions is None or len(positions) < 10:
        return None

    pos = np.asarray(positions, dtype=float)
    ca = _centroid_locally(reference, pos, box)
    cb = _centroid_locally(image, pos, box)
    if ca is None or cb is None:
        return None

    dx, dy = cb[:, 0] - ca[:, 0], cb[:, 1] - ca[:, 1]
    ok = (np.isfinite(dx) & np.isfinite(dy)
          & (np.abs(dx) < max_shift_px) & (np.abs(dy) < max_shift_px))
    if ok.sum() < 10:
        return None

    mx, my = float(np.median(dx[ok])), float(np.median(dy[ok]))
    return mx, my, {
        "n_pairs": int(ok.sum()),
        "scatter_x": float(1.4826 * np.median(np.abs(dx[ok] - mx))),
        "scatter_y": float(1.4826 * np.median(np.abs(dy[ok] - my))),
        "from_catalogue": True,
    }

def measure_offset(reference, image, fwhm=5.5, threshold_sigma=20.0, n_stars=300,
                   max_match_px=4.0):
    """The shift still left between two frames after they were put on one grid.

    WHY THIS IS NEEDED AFTER A WCS ALIGNMENT
    Resampling through two plate solutions is only as good as those solutions. Each
    was fitted separately, on a different exposure, and each carries its own error
    of a fraction of an arcsecond. What survives is a small systematic shift - on
    the frames this was written for, 0.6 of a pixel - and it is visible: every star
    leaves a dark-and-bright pair in the subtracted image instead of vanishing, and
    the detector reports those pairs as regions.

    A shift is measurable to far better than a pixel because it is common to every
    star in the frame. Here it is the median over a few hundred of them, and the
    scatter about that median says whether it really is one rigid shift or whether
    something else is wrong.

    Returns (dx, dy, diagnostics). dx, dy are how far `image` sits from
    `reference`, so shifting it by the negative of them brings the two together.
    """
    import numpy as np
    from photutils.detection import DAOStarFinder
    from astropy.stats import sigma_clipped_stats
    from scipy.spatial import cKDTree

    def bright_stars(img):
        _, med, sd = sigma_clipped_stats(img, sigma=3.0)
        table = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * sd)(img - med)
        if table is None or len(table) == 0:
            return np.empty((0, 2))
        table.sort("flux")
        table = table[::-1][:n_stars]
        return np.transpose((np.asarray(table["xcentroid"], dtype=float),
                             np.asarray(table["ycentroid"], dtype=float)))

    a, b = bright_stars(reference), bright_stars(image)
    if len(a) < 10 or len(b) < 10:
        return 0.0, 0.0, {"n_pairs": 0,
                          "why": "too few stars to measure a shift"}

    distance, index = cKDTree(b).query(a, k=1)
    ok = distance < max_match_px
    if ok.sum() < 10:
        return 0.0, 0.0, {"n_pairs": int(ok.sum()),
                          "why": "too few stars close enough to pair up"}

    dx = b[index[ok], 0] - a[ok, 0]
    dy = b[index[ok], 1] - a[ok, 1]
    mx, my = float(np.median(dx)), float(np.median(dy))
    return mx, my, {
        "n_pairs": int(ok.sum()),
        "scatter_x": float(1.4826 * np.median(np.abs(dx - mx))),
        "scatter_y": float(1.4826 * np.median(np.abs(dy - my))),
    }

def shift_frame(image, dx, dy):
    """Move a frame by a fraction of a pixel, to undo a measured offset.

    The blank margin left by an earlier resampling is not a number, and a cubic
    spline run over it would spread that nothingness across the whole frame - the
    first version of this did exactly that, and every measurement downstream came
    back empty. So the gaps are filled before the shift and put back after, which
    keeps them exactly where they were.
    """
    import numpy as np
    from scipy.ndimage import shift as ndshift

    known = np.isfinite(image)
    moved = ndshift(np.where(known, image, 0.0), (-dy, -dx), order=3, mode="nearest")
    still_known = ndshift(known.astype(float), (-dy, -dx), order=1, mode="nearest")
    moved[still_known < 0.5] = np.nan
    return moved

def refine_alignment(reference, image, label, min_shift_px=0.05, positions=None):
    """Measure the leftover shift against a reference frame and take it out.

    When a list of catalogue star positions is available the shift is measured on
    those, the same stars in both frames. Otherwise the frames are asked to supply
    their own stars, which works between exposures through one filter and works
    poorly between two different ones.
    """
    from_catalogue = offset_from_catalogue(reference, image, positions)
    if from_catalogue is not None:
        dx, dy, diag = from_catalogue
    else:
        dx, dy, diag = measure_offset(reference, image)
    if diag.get("n_pairs", 0) < 10:
        print("  {}: could not check the alignment - {}".format(
            label, diag.get("why", "not enough stars")))
        return image
    size = (dx * dx + dy * dy) ** 0.5
    if size < min_shift_px:
        print("  {}: already aligned to {:.2f} px, nothing to correct".format(
            label, size))
        return image
    print("  {}: {}{} stars say it sits ({:+.2f}, {:+.2f}) px from the reference "
          "(scatter {:.2f}, {:.2f}); shifted back".format(
              label, "catalogue " if diag.get("from_catalogue") else "",
              diag["n_pairs"], dx, dy,
              diag["scatter_x"], diag["scatter_y"]))
    return shift_frame(image, dx, dy)

def star_positions(red, mask=None, n_stars=400, threshold_sigma=15.0):
    """Positions of the bright point sources in the broad-band frame.

    Even after matching and subtraction a star rarely vanishes perfectly: a small
    error in the factor or the blurring leaves a residual exactly where the star
    was. Those residuals are compact and bright, so the detector reports them as
    HII regions. Knowing where the stars were lets us throw them out.
    """
    from photutils.detection import DAOStarFinder
    from astropy.stats import sigma_clipped_stats

    mean, median, std = sigma_clipped_stats(red, sigma=3.0, mask=mask)
    finder = DAOStarFinder(fwhm=4.0, threshold=threshold_sigma * std)
    tbl = finder(red - median, mask=mask)
    if tbl is None or len(tbl) == 0:
        return np.empty((0, 2))
    tbl.sort("flux")
    tbl = tbl[::-1][:n_stars]
    return np.transpose((np.asarray(tbl["xcentroid"], dtype=float),
                         np.asarray(tbl["ycentroid"], dtype=float)))

def subtract_continuum(ha, red, factor, mask=None):
    """factor x H-alpha - R, with each frame's own sky taken off first.

    WHY THE SKY COMES OFF BEFORE THE MULTIPLICATION
    Because the factor is meant to scale the starlight, and the frame holds two
    things: starlight and sky. Multiplying the whole frame multiplies the sky with
    it. On an M33 pair the H-alpha sky was 12.6 counts and the R sky 95.4, and the
    factor 25.7 - so the result sat on a pedestal of 25.7 x 12.6 - 95.4 = 229
    counts, seventeen times the noise of the image.

    Detection survived that, because it estimates a background and removes it
    before looking for anything. What did not survive was the picture: the whole
    dynamic range went into the pedestal, faint emission vanished into a flat grey,
    and any difference in flat-fielding between the two filters was amplified by
    the factor along with everything else. Anyone judging the subtraction by eye -
    which is what this tool asks for whenever the factor looks doubtful - was
    looking at an image ruined before they saw it.

    Sky is measured with clipping, so the galaxy and the stars do not drag it.
    """
    from astropy.stats import sigma_clipped_stats

    sky_ha = float(sigma_clipped_stats(ha, sigma=3.0, mask=mask)[1])
    sky_red = float(sigma_clipped_stats(red, sigma=3.0, mask=mask)[1])
    net = factor * (ha - sky_ha) - (red - sky_red)
    return net, {"sky_ha": sky_ha, "sky_red": sky_red,
                 "pedestal_avoided": factor * sky_ha - sky_red}

def line_excess(net, red, x, y, radius_px=5.0):
    """How much line emission each detection has, next to its own continuum.

    The ratio of what is left after the subtraction to what the continuum frame
    holds at the same place. A foreground star has nothing left once the
    subtraction is right, so its ratio sits near zero. An HII region is emission
    with only a modest continuum under it, so its ratio is of order one or more.
    """
    from photutils.aperture import CircularAperture, aperture_photometry

    positions = np.column_stack([np.asarray(x, dtype=float),
                                 np.asarray(y, dtype=float)])
    if not len(positions):
        return np.zeros(0)
    ap = CircularAperture(positions, r=radius_px)
    f_net = np.asarray(aperture_photometry(np.nan_to_num(net), ap)["aperture_sum"],
                       dtype=float)
    f_red = np.asarray(aperture_photometry(np.nan_to_num(red), ap)["aperture_sum"],
                       dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return f_net / np.where(np.abs(f_red) > 1.0, np.abs(f_red), np.nan)

def filter_detections(x, y, centre, r_max_px=None, star_xy=None, star_radius_px=6.0,
                      excess=None, min_excess=0.5,
                      minor_axis=None, elongation=None,
                      min_width_px=1.5, max_elongation=2.5):
    """Drop detections that cannot be HII regions belonging to this galaxy.

    Two cuts, both blunt and both necessary.

    Outside the galaxy: HII regions live in the disk, a frame is usually much wider
    than the galaxy, and everything found out in the empty corners is noise or an
    artefact. Cutting at the isophotal radius R25 removes them without touching the
    disk, and R25 is a published number rather than a judgement call.

    On top of a star: subtraction residuals sit exactly where the stars were.

    WHY THE STAR CUT NEEDS THE SECOND TEST
    A bright HII region contains the young cluster that lights it, and a young
    cluster is a point source in the continuum frame - so a cut that removes
    everything sitting on a point source removes the brightest regions in the
    galaxy along with the foreground stars. On the M83 frames this was measured:
    of forty detections the star cut threw away inside the galaxy, twenty-nine had
    real line emission. Their emission is what separates them - the regions kept
    by the rest of the pipeline showed a ratio around 2.3, and the point sources
    out in the empty field, which really are stars, around 0.17. So a detection
    sitting on a point source is still dropped, unless it is emitting.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.ones(x.size, dtype=bool)
    report = {}

    # Shape. An HII region is a body of gas, so it has width; what is left over
    # from an imperfect subtraction is a thread - a pixel or two across, often
    # curved, following the edge of whatever did not cancel. Measured on the M83
    # frames, blobs inside the galaxy are typically 1.9 px wide and blobs out in
    # the empty field, where nothing real can be, 0.5 px. That one number does
    # more to clean the catalogue than any brightness cut.
    if minor_axis is not None:
        wide_enough = np.isfinite(minor_axis) & (np.asarray(minor_axis) >= min_width_px)
        report["too thin to be a region"] = int((~wide_enough & keep).sum())
        keep &= wide_enough

    if elongation is not None:
        round_enough = np.isfinite(elongation) & (np.asarray(elongation) <= max_elongation)
        report["a streak, not a region"] = int((~round_enough & keep).sum())
        keep &= round_enough

    if r_max_px is not None:
        inside = np.hypot(x - centre[0], y - centre[1]) <= r_max_px
        report["outside the galaxy"] = int((~inside & keep).sum())
        keep &= inside

    if star_xy is not None and len(star_xy):
        d = np.hypot(x[:, None] - star_xy[None, :, 0], y[:, None] - star_xy[None, :, 1])
        on_star = d.min(axis=1) < star_radius_px
        if excess is not None:
            emitting = np.isfinite(excess) & (excess > min_excess)
            rescued = int((on_star & emitting & keep).sum())
            if rescued:
                report["on a star but emitting, kept"] = rescued
            on_star = on_star & ~emitting
        report["on a star"] = int((on_star & keep).sum())
        keep &= ~on_star

    return keep, report

def completeness_vs_radius(resid, centre, detect_kwargs, r_edges,
                           n_per_ring=30, peak_counts=1.0, fwhm_px=6.0, rng_seed=11):
    """What fraction of injected regions comes back, ring by ring.

    WHY BOTHER
    A radial profile is the measurement most easily faked by a detection limit that
    changes with radius. The centre of a galaxy is bright and crowded, the outskirts
    dark and empty, so the faintest thing still visible is not the same in both. If
    detection gets harder inwards, the profile grows a central hole that belongs to
    the search, not to the galaxy.

    HOW
    Drop artificial regions of known peak brightness into the real image, run
    exactly the same detection on it, and count how many come back, ring by ring.

    peak_counts is the peak height of the injected region and should be set in units
    of the background noise. Scaling injected regions to the brightness of an
    average real one puts them hundreds of sigma above the threshold, where
    everything is found everywhere and the test measures nothing.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(rng_seed)
    sigma = fwhm_px / 2.3548
    ny, nx = resid.shape
    recovered = np.zeros(len(r_edges) - 1)
    injected = np.zeros(len(r_edges) - 1)

    for k in range(len(r_edges) - 1):
        r_lo, r_hi = r_edges[k], r_edges[k + 1]
        fake = np.zeros_like(resid)
        truth = []
        for _ in range(n_per_ring):
            rad = math.sqrt(rng.uniform(r_lo ** 2, r_hi ** 2))
            th = rng.uniform(0.0, 2.0 * math.pi)
            px = centre[0] + rad * math.cos(th)
            py = centre[1] + rad * math.sin(th)
            if not (8 < px < nx - 8 and 8 < py < ny - 8):
                continue
            fake[int(round(py)), int(round(px))] += peak_counts
            truth.append((px, py))
        if not truth:
            continue

        # A delta smoothed by a Gaussian has peak amplitude/(2 pi sigma^2);
        # multiplying it back gives a region whose PEAK is peak_counts.
        fake = gaussian_filter(fake, sigma) * (2.0 * math.pi * sigma ** 2)
        cat, _, _ = detect_regions(resid + fake, **detect_kwargs)

        if cat is None:
            injected[k] += len(truth)
            continue

        fx = np.asarray(cat.xcentroid, dtype=float)
        fy = np.asarray(cat.ycentroid, dtype=float)
        # Segments touching the mask come back with a centroid of NaN. Left in, a
        # single NaN makes every distance comparison below false and the measured
        # completeness collapses to zero everywhere.
        finite = np.isfinite(fx) & np.isfinite(fy)
        fx, fy = fx[finite], fy[finite]

        for (tx, ty) in truth:
            injected[k] += 1
            if fx.size and np.min(np.hypot(fx - tx, fy - ty)) < 3.0:
                recovered[k] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(injected > 0, recovered / injected, np.nan)
    return frac, injected, recovered

def choose_subtraction(ha, red, positions, prefix, centre=None, r25_px=None,
                       factors=None, blurs=None, ask=None):
    """Offer three subtractions as pictures and let the person pick one.

    WHY A PERSON DECIDES THIS
    Because every automatic criterion tried here disagreed with the eye, and the
    eye was right. Matching star widths by their measured FWHM prescribed a blur
    with the sign wrong. Choosing the factor as the mean over stars left twice as
    many visible star residuals as a hand-tuned value a few percent lower. The
    quantity that actually matters - whether a star has vanished into the sky or
    still shows as a spot that a detector will call an HII region - is something a
    person reads off a frame in a second and no single number here has captured.

    WHY THREE AND NOT A NUMBER ON THE SCREEN
    A number cannot be judged without knowing what it looks like. Three frames
    written to disk, differing enough to tell apart, and one question.

    HOW THE THREE ARE PICKED
    The whole grid of factor and blur is scored on stars that lie outside the
    galaxy, where nothing but sky and the star is in the aperture. The score is how
    flat what is left of the median star is - a good subtraction leaves no core and
    no ring. The best is offered, with one candidate either side of it in factor,
    so that the three span visibly less and visibly more subtraction rather than
    three versions of the same answer.

    Returns (ha, red, factor, diagnostics). Without a question to ask - a
    non-interactive run - the middle one is taken and said so.
    """
    from scipy.ndimage import gaussian_filter
    from astropy.stats import sigma_clipped_stats

    diag = {}
    if positions is None or len(positions) < 30:
        return ha, red, None, {"why": "too few stars to judge a subtraction on"}

    pos = np.asarray(positions, dtype=float)
    if centre is not None and r25_px:
        outside = np.hypot(pos[:, 0] - centre[0], pos[:, 1] - centre[1]) > 1.3 * r25_px
        if outside.sum() >= 30:
            pos = pos[outside]
    diag["n_stars"] = int(len(pos))

    # WHERE THE RANGE COMES FROM
    # From this pair of frames, never from a number typed in. The factor that
    # cancels a star is measured on the stars themselves and the grid is laid out
    # around it, so a galaxy whose factor is twenty is searched near twenty and one
    # whose factor is twelve is searched near twelve. A grid written by hand would
    # have been the range of whichever galaxy it was written for.
    if factors is None:
        pass  # from continuum_subtraction, inlined above
        try:
            centre_factor = float(constant_factor(ha, red, positions=pos)[0])
        except Exception:
            centre_factor = None
        if not centre_factor or not np.isfinite(centre_factor) or centre_factor <= 0:
            return ha, red, None, {"why": "could not measure a factor to search around"}
        diag["measured_factor"] = centre_factor

    h = 8
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    rr = np.hypot(xx, yy)
    edge = rr > h - 1.5
    ny, nx = ha.shape
    sel = [(int(round(x)), int(round(y))) for x, y in pos
           if h < int(round(x)) < nx - h and h < int(round(y)) < ny - h]
    if len(sel) < 30:
        return ha, red, None, {"why": "the stars all fall too near the frame edge"}

    def stamps(img):
        out = np.empty((len(sel), 2 * h + 1, 2 * h + 1), dtype=np.float32)
        for i, (x, y) in enumerate(sel):
            s = np.nan_to_num(img[y - h:y + h + 1, x - h:x + h + 1])
            out[i] = s - np.median(s[edge])
        return out

    # THERE IS NO RANGE
    # An upper and lower bound on the search is a guess about the answer, and the
    # answer is what is being looked for. So the search starts small and grows: as
    # long as the best point it has found sits on an edge of what has been tried,
    # that edge moves out and it looks again. It stops when the best point has
    # neighbours on all sides, which is the only evidence that it is a minimum
    # rather than the end of the paper.
    f_step = 0.025 * centre_factor if factors is None else None
    b_step = 0.25
    if factors is None:
        f_lo, f_hi = -2, 2                  # in steps, relative to the measured factor
    else:
        f_lo = f_hi = None
    b_lo, b_hi = -2, 2

    cache = {}
    stamp_cache = {}

    def score(fi, bi):
        key = (fi, bi)
        if key in cache:
            return cache[key]
        blur = bi * b_step
        if bi not in stamp_cache:
            A = gaussian_filter(ha, blur) if blur > 0 else ha
            R = gaussian_filter(red, -blur) if blur < 0 else red
            stamp_cache[bi] = (stamps(A), stamps(R))
        SA, SR = stamp_cache[bi]
        f = (centre_factor + fi * f_step) if factors is None else float(factors[fi])
        M = np.median(f * SA - SR, axis=0)
        prof = [M[(rr >= q) & (rr < q + 1)].mean() for q in range(6)]
        cache[key] = (float(np.abs(prof).max()), float(f), float(blur))
        return cache[key]

    if factors is not None:
        f_lo, f_hi = 0, len(factors) - 1
    if blurs is not None:
        b_step = 1.0
        b_lo, b_hi = 0, len(blurs) - 1

    for _round in range(40):
        grid = [score(fi, bi)
                for fi in range(f_lo, f_hi + 1)
                for bi in range(b_lo, b_hi + 1)]
        best_left, best_f, best_blur = min(grid)
        bi_best = int(round(best_blur / b_step)) if b_step else 0
        fi_best = (int(round((best_f - centre_factor) / f_step))
                   if factors is None else None)
        grew = False
        if factors is None:
            if fi_best <= f_lo and centre_factor + (f_lo - 1) * f_step > 0:
                f_lo -= 1; grew = True
            if fi_best >= f_hi:
                f_hi += 1; grew = True
        if bi_best <= b_lo:
            b_lo -= 1; grew = True
        if bi_best >= b_hi:
            b_hi += 1; grew = True
        if not grew:
            break

    scored = sorted(cache.values())
    best_left, best_f, best_blur = scored[0]
    diag["best"] = (best_f, best_blur, best_left)
    diag["searched"] = {
        "factor": (min(s[1] for s in scored), max(s[1] for s in scored)),
        "blur": (min(s[2] for s in scored), max(s[2] for s in scored)),
        "points": len(scored)}
    print("    searched factor {:.2f} to {:.2f} and blur {:+.2f} to {:+.2f} px "
          "over {} combinations, until the best had neighbours on every side"
          .format(diag["searched"]["factor"][0], diag["searched"]["factor"][1],
                  diag["searched"]["blur"][0], diag["searched"]["blur"][1],
                  len(scored)))

    # One candidate either side in factor, each with its own best blur, so the
    # three differ by something the eye can see.
    def best_at(f):
        here = [s for s in scored if abs(s[1] - f) < 1e-6]
        return min(here) if here else None

    others = sorted({f for _, f, _ in scored})
    below = [f for f in others if f < best_f]
    above = [f for f in others if f > best_f]
    picks = [best_at(below[-1]) if below else None,
             (best_left, best_f, best_blur),
             best_at(above[0]) if above else None]
    picks = [p for p in picks if p is not None]
    if len(picks) < 3:                       # the best sat at an end of the range
        extra_pool = [s for s in scored if s[1] not in {p[1] for p in picks}]
        picks += sorted(extra_pool)[:3 - len(picks)]
    picks.sort(key=lambda p: p[1])

    return _write_choices(ha, red, picks, prefix, diag, ask)

def show_pictures(paths, title=None):
    """Put pictures already written to disk on the screen, without blocking.

    WHY IT SHOWS THE SAVED FILE AND NOT THE FIGURE
    Because the figure was drawn under Agg, the backend that writes to a file and
    has no window to attach to, and moving a live figure between backends is not
    something matplotlib supports. Reading back the PNG that was just written
    sidesteps that entirely, and has the advantage that what appears on the screen
    is exactly the file - if they ever disagreed it would be the file that is wrong,
    and the file is what gets kept.

    An interactive backend is borrowed and given back. On a machine with no display
    nothing is shown and nothing fails; the files are still there.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    paths = [p for p in paths if p and os.path.exists(p)]
    if not paths:
        return False

    started = matplotlib.get_backend()
    if started.lower() == "agg":
        for candidate in ("TkAgg", "QtAgg", "Qt5Agg", "MacOSX"):
            try:
                plt.switch_backend(candidate)
                break
            except Exception:
                continue
    if matplotlib.get_backend().lower() == "agg":
        return False

    try:
        import matplotlib.image as mpimg
        for path in paths:
            image = mpimg.imread(path)
            height, width = image.shape[0], image.shape[1]
            side = 9.5
            fig = plt.figure(figsize=(side, side * height / float(width)))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(image)
            ax.set_axis_off()
            fig.canvas.manager.set_window_title(
                title or os.path.basename(path))
        plt.show(block=False)
        plt.pause(0.5)
        return True
    except Exception:
        return False
    finally:
        try:
            plt.switch_backend(started)
        except Exception:
            pass

def _show_choices_window(frames, nz, step):
    """Put the three candidates side by side in one window and leave it open.

    WHY A WINDOW AND NOT THREE FILES
    Because the judgement is a comparison. Three files opened one after another in
    a picture viewer are three separate impressions a few seconds apart; side by
    side they are one look, and the difference between them - which is small, a few
    percent in the factor - is only visible that way.

    The backend is the difficulty. Both tools set Agg at import, which draws to a
    file and never to a screen, and that is the right default for a run that
    produces figures unattended. So an interactive backend is borrowed for this one
    window and given back afterwards, and if none of them can be had - a machine
    with no display, a session over ssh - nothing is shown and the files written
    next to the output are the fallback.

    The window is opened without blocking, so the question can be answered in the
    terminal while it is still on screen.

    Returns True if a window actually appeared.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    def block_mean(a, n):
        k0 = (a.shape[0] // n) * n
        k1 = (a.shape[1] // n) * n
        return a[:k0, :k1].reshape(k0 // n, n, k1 // n, n).mean(axis=(1, 3))

    started = matplotlib.get_backend()
    if started.lower() == "agg":
        for candidate in ("TkAgg", "QtAgg", "Qt5Agg", "MacOSX"):
            try:
                plt.switch_backend(candidate)
                break
            except Exception:
                continue
    if matplotlib.get_backend().lower() == "agg":
        return False                     # nothing on this machine can show a window

    try:
        fig, axes = plt.subplots(1, len(frames), figsize=(5.2 * len(frames), 5.8))
        if len(frames) == 1:
            axes = [axes]
        for ax, (i, (f, blur, net)) in zip(axes, enumerate(frames, 1)):
            small = block_mean(net, step) if step > 1 else net
            ax.imshow(small, origin="lower", cmap="gray_r",
                      vmin=-2 * nz, vmax=8 * nz)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("choice {}\nfactor {:.2f}   {}".format(
                i, f, _blur_words(blur)), fontsize=12)
        fig.suptitle("Which subtraction is best? Look at the stars and at the "
                     "empty sky, then answer in the terminal.", fontsize=13)
        fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)
        return True
    except Exception:
        return False
    finally:
        try:
            plt.switch_backend(started)
        except Exception:
            pass

def _blur_words(blur):
    """The blur as a phrase, so the picture says which frame was blurred."""
    if blur > 0:
        return "H-alpha blurred by {:.2f} px".format(blur)
    if blur < 0:
        return "R blurred by {:.2f} px".format(-blur)
    return "neither frame blurred"

def _write_choices(ha, red, picks, prefix, diag, ask):
    """Write one picture per candidate, ask, and apply the answer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from astropy.stats import sigma_clipped_stats

    def block_mean(a, n):
        k0 = (a.shape[0] // n) * n
        k1 = (a.shape[1] // n) * n
        return a[:k0, :k1].reshape(k0 // n, n, k1 // n, n).mean(axis=(1, 3))

    step = max(1, int(round(max(ha.shape) / 1400.0)))
    frames, paths = [], []
    for i, (left, f, blur) in enumerate(picks, 1):
        A = gaussian_filter(ha, blur) if blur > 0 else ha
        R = gaussian_filter(red, -blur) if blur < 0 else red
        frames.append((f, blur, f * A - R))

    # One stretch for all three, or the eye compares three different stretches.
    middle = frames[len(frames) // 2][2]
    nz = float(sigma_clipped_stats(middle, sigma=3.0)[2])

    for i, (f, blur, net) in enumerate(frames, 1):
        small = block_mean(net, step) if step > 1 else net
        fig, ax = plt.subplots(figsize=(9, 9))
        # Dark on light, the way this is looked at by eye.
        ax.imshow(small, origin="lower", cmap="gray_r", vmin=-2 * nz, vmax=8 * nz)
        ax.set_xticks([]); ax.set_yticks([])
        # The numbers belong on the picture. A choice made from three unlabelled
        # frames cannot be written down, repeated, or argued with afterwards.
        ax.set_title("choice {}\nfactor {:.2f}   {}".format(i, f, _blur_words(blur)),
                     fontsize=14)
        fig.tight_layout()
        path = "{}_choice_{}.png".format(prefix, i)
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    diag["paths"] = paths
    diag["picks"] = [(f, blur) for f, blur, _ in frames]

    # WHY THERE IS NO WINDOW
    # There was one, and it did not work. A figure opened without blocking, so
    # that the question could still be answered in the terminal, is a window with
    # nothing driving its event loop: it appears as an empty grey rectangle and
    # only draws itself once it is closed, which is after the answer has been
    # given. A picture that can only be seen after it is no longer needed is worse
    # than no picture, because it also freezes the screen.
    #
    # So the three are written to disk and their names are printed. Open them in
    # whatever you normally look at images with, or do not - the numbers are on
    # the screen either way, and the middle one is the default.
    shown = False

    print("")
    print("  Three subtractions of the same frames, written next to the other")
    print("  output. Look at the stars and at the empty sky in them, then say")
    print("  which one is best. Enter takes the middle one.")
    print("    in " + (os.path.dirname(paths[0]) or "."))
    for i, (p, (f, blur, _)) in enumerate(zip(paths, frames), 1):
        print("    {}. {:<24} factor {:.2f}, {}".format(
            i, os.path.basename(p), f, _blur_words(blur)))

    answer = 2 if len(paths) >= 2 else 1
    if ask is not None:
        while True:
            got = ask("Which one", str(answer), int)
            if 1 <= got <= len(paths):
                answer = got
                break
            print("    a number from the list, please")
    else:
        print("    nothing to ask with, so {} is used".format(answer))

    if shown:
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass

    f, blur, _ = frames[answer - 1]
    if blur > 0:
        ha = gaussian_filter(ha, blur)
    elif blur < 0:
        red = gaussian_filter(red, -blur)
    diag["chosen"] = answer
    diag["factor"] = f
    diag["blur"] = blur
    return ha, red, f, diag

def describe_choice(diag):
    """What was chosen, in a line."""
    if diag.get("why"):
        return "  no choice offered ({})".format(diag["why"])
    f, blur = diag["factor"], diag["blur"]
    what = ("H-alpha blurred by {:.2f} px".format(blur) if blur > 0 else
            "R blurred by {:.2f} px".format(-blur) if blur < 0 else
            "neither frame blurred")
    return ("  choice {} taken: factor {:.2f}, {}\n"
            "    judged on {} stars outside the galaxy"
            .format(diag["chosen"], f, what, diag["n_stars"]))


# --------------------------------------------------------------------------
# Star_Formation_Full_Analysis.py
# --------------------------------------------------------------------------

matplotlib.use("Agg")

warnings.filterwarnings("ignore")

pass  # from HII_From_Images, inlined above

pass  # from radial_profiles, inlined above

describe_match = describe__tracer_matching  # from tracer_matching, inlined above

pass  # from galaxy_catalogue, inlined above

pass  # from deprojection, inlined above

pass  # from anchor_registration, inlined above

pass  # from galaxy_window, inlined above

describe_window = describe__galaxy_window  # from galaxy_window, inlined above

pass  # from galaxy_extent, inlined above

describe_extent = describe__galaxy_extent  # from galaxy_extent, inlined above

describe_anchors = describe__anchor_registration  # from anchor_registration, inlined above

pass  # from interactive_input, inlined above

def run_blue_cluster_tool(b_path, v_path, galaxy, distance_mpc, script_dir,
                          ebv=None):
    """Run Blue_Clusters_From_Images.py and return the path to its knot catalogue.

    That tool asks its questions interactively, so the answers are piped in. It is
    run as a separate process rather than imported because it is written as a
    script: importing it would execute the whole pipeline at import time, before
    this tool had a chance to say anything.
    """
    tool = os.path.join(script_dir, "Blue_Clusters_From_Images.py")
    if not os.path.exists(tool):
        tool = _unpack_cluster_tool()

    # Galaxy, distance, reddening, then the two frames. The reddening is passed
    # rather than left blank so that the number this run reports is the number it
    # actually used; leaving it blank lets that tool look it up again and it may
    # come back with a different one.
    answers = "{}\n{}\n{}\n{}\n{}\n\n".format(
        galaxy, distance_mpc if distance_mpc else "",
        "{:.4f}".format(ebv) if ebv is not None else "",
        b_path, v_path)

    print("Running Blue_Clusters_From_Images.py for the blue knots")
    print("  (this is the long part - photometry, catalogue calibration, colour cut)")
    proc = subprocess.run([sys.executable, tool], input=answers, text=True,
                          capture_output=True, cwd=script_dir)
    for line in proc.stdout.splitlines():
        if any(k in line for k in ("saved to", "[lookup]", "Wrote", "sources",
                                   "Error", "error")):
            print("    " + line)
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
        raise RuntimeError("the blue-cluster tool exited with an error")

    outdir = os.path.join(os.path.dirname(os.path.abspath(v_path)) or ".", "result_CMD")
    hits = glob.glob(os.path.join(
        outdir, "*color_filtered_with_radius.csv"))
    if not hits:
        raise FileNotFoundError(
            "the blue-cluster tool ran but produced no colour-filtered catalogue in "
            + outdir)
    return max(hits, key=os.path.getmtime)

def output_folders(prefix):
    """Three folders, named for what is in them, and the file stem to use inside.

    WHY NOT ONE FLAT FOLDER
    Because this tool produces three separate pieces of work - the emission-line
    side, the blue-cluster side, and the comparison between them - and a reader
    coming back to a run months later has to be able to tell which file belongs to
    which without opening it. A single folder of twenty files all beginning with
    the galaxy's name does not say that; three named folders do.

    Returns (stem, hii_dir, clusters_dir, comparison_dir), where stem is the
    galaxy's name with no directory attached.
    """
    root = os.path.dirname(os.path.abspath(prefix)) or "."
    stem = os.path.basename(prefix)
    folders = []
    for name in ("HII_regions", "Blue_clusters", "Comparison"):
        path = os.path.join(root, name)
        os.makedirs(path, exist_ok=True)
        folders.append(path)
    return (stem,) + tuple(folders)

def write_registered(folder, galaxy, b_data, v_data, v_wcs, b_info, v_info):
    """Write the B and V frames as this tool holds them, onto the V grid.

    These are not copies of the input files. They are the frames after being
    resampled onto the reference grid and after the leftover sub-pixel shift was
    taken out, and they carry the V frame's plate solution, which the combined
    files on disk had lost. That is what makes them safe to hand to another tool:
    whatever it measures on them is already in the coordinate system everything
    else in this run uses.

    The filter and exposure keywords are put back too, since the tool downstream
    prints them and a combined frame no longer has them.

    These go through save_frame like every other frame this pipeline writes, so
    they get the same treatment: no NaN in the file, and the plate solution
    written in both the PC and the CD convention. They were written directly
    through astropy before, which is how they ended up being the one pair of
    outputs an older reader could not open.
    """
    os.makedirs(folder, exist_ok=True)
    stem = (galaxy or "galaxy").replace(" ", "_")
    written = []
    for name, data, info in (("B", b_data, b_info), ("V", v_data, v_info)):
        path = os.path.join(folder, "{}_{}_on_V_grid.fits".format(stem, name))
        written.append(save_frame(
            path, data, v_wcs,
            {"filter": info.get("filter", name),
             "exptime": info.get("exptime", 1.0),
             "object": galaxy or "galaxy",
             "pixel_scale": info.get("pixel_scale")},
            history=("Resampled onto the V grid by Star_Formation_Full_Analysis",)))
    return written[0], written[1]

def read_knots(path):
    """Read blue-knot positions from the colour-filtered catalogue.

    The columns are named X_V and Y_V because they are measured on the V frame,
    which is exactly the reference grid this tool works on.
    """
    import csv

    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip() for h in rows[0]]
    low = [h.lower() for h in header]
    try:
        ix, iy = low.index("x_v"), low.index("y_v")
    except ValueError:
        raise ValueError(path + ": expected columns X_V and Y_V, found " + str(header))

    xs, ys = [], []
    for r in rows[1:]:
        try:
            xs.append(float(r[ix]))
            ys.append(float(r[iy]))
        except (TypeError, ValueError, IndexError):
            continue
    return np.array(xs), np.array(ys)

def both_populations_figure(knots, regions, pairs, centre, scale, unit, title):
    """Both tracers on one map, with an arrow from each knot to its parent region."""
    fig, ax = plt.subplots(figsize=(9.2, 9.2))

    kx = (knots[:, 0] - centre[0]) * scale
    ky = (knots[:, 1] - centre[1]) * scale
    rx = (regions[:, 0] - centre[0]) * scale
    ry = (regions[:, 1] - centre[1]) * scale

    ax.scatter(rx, ry, s=16, c="#a3283c", alpha=0.55, linewidths=0,
               label="HII regions ({})".format(len(rx)))
    ax.scatter(kx, ky, s=26, facecolors="none", edgecolors="#2b5a89", linewidths=1.1,
               label="blue knots ({})".format(len(kx)))

    idx = pairs["nearest_index"]
    for i in range(len(kx)):
        j = idx[i]
        ax.annotate("", xy=(rx[j], ry[j]), xytext=(kx[i], ky[i]),
                    arrowprops=dict(arrowstyle="->", color="#555555",
                                    lw=0.7, alpha=0.75, shrinkA=2, shrinkB=2))

    ax.plot(0, 0, "+", c="k", ms=15, mew=2)
    lim = 1.05 * max(np.abs(np.concatenate([kx, rx])).max(),
                     np.abs(np.concatenate([ky, ry])).max())
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x from centre [" + unit + "]")
    ax.set_ylabel("y from centre [" + unit + "]")
    ax.grid(alpha=0.15, lw=0.6)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_title(title, fontsize=12, loc="left")

    note = ("Each arrow runs from a blue knot to the nearest HII region, which is "
            "taken to be where it formed. Mean separation {:.0f} {}. Several knots "
            "sharing one region is expected - one complex can make more than one "
            "cluster.").format(pairs["mean_separation"], unit)
    fig.text(0.09, 0.045, "\n".join(textwrap.wrap(note, width=104)),
             fontsize=8, color="#4a4a4a", va="top")
    fig.subplots_adjust(bottom=0.14)
    return fig

def separation_figure(pairs, unit, title):
    """Separation against galactocentric radius, against the density null model."""
    fig, ax = plt.subplots(figsize=(10.5, 6.4))

    r, s, e = pairs["radius"], pairs["separation"], pairs["expected"]

    ax.scatter(r, s, s=30, c="#2b5a89", alpha=0.75, linewidths=0,
               label="measured separation")

    order = np.argsort(r)
    if np.isfinite(pairs["fit_slope"]):
        ax.plot(r[order], np.polyval([pairs["fit_slope"], pairs["fit_intercept"]],
                                     r[order]),
                "-", lw=1.8, c="#2b5a89",
                label="fit: slope {:.3f}, R2 = {:.2f}".format(
                    pairs["fit_slope"], pairs["fit_r2"]))

    good = np.isfinite(e)
    if good.any():
        ax.plot(r[order][np.isfinite(e[order])], e[order][np.isfinite(e[order])],
                "--", lw=2.0, c="#a3283c",
                label="expected from HII density alone")

    ax.set_xlabel("distance of the blue knot from the centre [" + unit + "]")
    ax.set_ylabel("distance to the nearest HII region [" + unit + "]")
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title(title, fontsize=12, loc="left")

    note = ("The dashed line is the separation a galaxy would show if nothing had "
            "drifted at all: in a random field of surface density n the nearest "
            "neighbour sits at 1/(2*sqrt(n)), and HII regions thin out towards the "
            "edge of the disk. Only the gap between the points and that line can be "
            "read as drift. Measured slope {:.4f} against {:.4f} for the null "
            "model.").format(pairs["fit_slope"], pairs["null_slope"])
    fig.text(0.08, 0.03, "\n".join(textwrap.wrap(note, width=118)),
             fontsize=8, color="#4a4a4a", va="top")
    fig.subplots_adjust(bottom=0.22)
    return fig

def ask_for_arguments():
    """Ask for what the command line would have carried, and hand back that list.

    The four frames and the galaxy name are what the tool cannot run without. The
    rest keeps its default here and stays available as a flag, so the questions
    are short enough to be read.
    """
    print("No arguments were given, so I will ask instead.")
    print("Press Enter to accept the value shown in brackets.")
    print("")

    # The galaxy is not asked for: it is written into every frame the telescope
    # took, and the folder scan reads it back. Asking would mean asking the person
    # to retype something the data already says, which is how a typo gets in.
    argv = ["--folder", ask_folder("Path to the folder holding the frames")]

    distance = ask_optional_float(
        "Distance in Mpc", "blank and it is looked up")
    if distance is not None:
        argv += ["--distance-mpc", repr(distance)]

    # Finding the blue knots is the slow half of this tool. Anyone who has already
    # run the blue-cluster tool on these frames has that catalogue sitting in
    # result_CMD, and reusing it turns a run of several minutes into one of about
    # thirty seconds.
    if ask_yes_no("Do you already have a blue-knot catalogue from the "
                  "blue-cluster tool?", False):
        argv += ["--knots-csv", ask_file("Path to that CSV")]

    argv += ["--threshold", repr(ask("Detection threshold for the HII regions, in "
                                     "units of the background noise", 4.0, float))]
    argv += ["--rings", repr(ask("Number of radial rings", 20, int))]

    prefix = ask("Where to write the output, name without extension "
                 "(blank to write beside the V frame)", "")
    if prefix:
        argv += ["--out-prefix", normalise_path(prefix)]

    print("")
    return argv

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Blue knots, HII regions, and the link between them.")
    ap.add_argument("--folder", default=None,
                    help="a folder of frames; the four files are worked out from "
                         "the headers, and this is the usual way to run the tool")
    ap.add_argument("--b", default=None, help="B-band FITS, when naming files yourself")
    ap.add_argument("--v", default=None, help="V-band FITS - the reference grid")
    ap.add_argument("--red", default=None, help="R-band FITS")
    ap.add_argument("--halpha", default=None, help="H-alpha FITS")
    ap.add_argument("--galaxy", default=None,
                    help="only needed when the frames do not name it themselves")
    ap.add_argument("--distance-mpc", type=float, default=None)
    ap.add_argument("--knots-csv", default=None,
                    help="skip the blue-cluster stage and use this catalogue")
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--min-pixels", type=int, default=6)
    ap.add_argument("--extent-sigma", type=float, default=0.5,
                    help="how far above the sky noise the galaxy's light must be "
                         "to count as still present")
    ap.add_argument("--window-margin", type=float, default=1.5,
                    help="how far beyond R25 to look for regions, as a multiple")
    ap.add_argument("--background-box", type=int, default=None,
                    help="size of the square the background is measured in, in px; "
                         "the default is a twelfth of the frame")
    ap.add_argument("--min-width", type=float, default=1.2,
                    help="narrowest a blob may be and still count as a region, in px")
    ap.add_argument("--max-elongation", type=float, default=2.5,
                    help="how far from round a region may be before it is a streak")
    ap.add_argument("--no-deblend", action="store_true")
    ap.add_argument("--deblend-levels", type=int, default=64,
                    help="how finely a touching group is sliced apart")
    ap.add_argument("--deblend-contrast", type=float, default=1e-4,
                    help="how faint a peak may be and still become its own region")
    ap.add_argument("--max-radius-r25", type=float, default=1.0)
    ap.add_argument("--star-radius", type=float, default=6.0)
    ap.add_argument("--no-star-cut", action="store_true")
    ap.add_argument("--rings", type=int, default=20)
    ap.add_argument("--sector-method", action="store_true")
    ap.add_argument("--colour-smooth", type=float, default=6.0,
                    help="pixels of smoothing used to build the B-V colour map")
    ap.add_argument("--constant-factor", action="store_true",
                    help="force one constant subtraction factor instead of "
                         "fitting it against colour")
    ap.add_argument("--out-prefix", default=None)

    # Started with nothing at all - from Spyder, or by double-clicking the file.
    # That is not a mistake, it is someone who has not been told the flags yet.
    if nothing_was_passed(argv):
        argv = ask_for_arguments()
    args = ap.parse_args(argv)

    _donors, _groups, _ebv = {}, {}, None
    if args.folder:
        _chosen, _donors, _groups, _found = frames_from_folder(
            args.folder, ("B", "V", "R", "Ha"))
        if not args.galaxy and _found:
            args.galaxy = _found
        args.b, args.v = _chosen["B"], _chosen["V"]
        args.red, args.halpha = _chosen["R"], _chosen["Ha"]
    if not (args.b and args.v and args.red and args.halpha):
        ap.error("give either --folder, or all four of --b --v --red --halpha")
    if not args.galaxy:
        ap.error("the frames do not name a galaxy in their headers - give --galaxy")

    # Both published numbers this run needs, looked up and shown before use. The
    # distance scales every length reported. The reddening is needed wherever a
    # colour becomes a statement about a stellar population - which is what
    # deciding "this is a young blue cluster" is - and this tool measures colour
    # itself rather than assuming somebody else already corrected it.
    if args.galaxy:
        if not args.distance_mpc:
            found = lookup_distance(args.galaxy)
            if found:
                args.distance_mpc = found
        _ebv = lookup_extinction(args.galaxy)
        summary = describe_lookups(args.galaxy, args.distance_mpc, _ebv)
        if summary:
            print("")
            print(summary)
            print("")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- the reference grid -------------------------------------------------
    # The anchor stars are wanted by the very first thing that happens - combining
    # the exposures of each filter - so the catalogue is asked before a single
    # frame is read. Without them that step falls back on scanning every pixel of
    # every exposure for stars, which is a minute an exposure on a large field.
    _probe_wcs, _probe_shape, _ = frame_geometry(
        (_groups.get("V") or [args.v])[0], _donors.get("V"))
    print("Looking up the foreground stars in this field")
    fg_stars, fg_diag = gaia_foreground_stars(_probe_wcs, _probe_shape)
    print(describe_foreground(fg_diag))
    _anchors = fg_stars if len(fg_stars) else None

    # The geometry belongs here too, for the same reason: the detection stage
    # wants to know where the galaxy ends before it starts, not after.
    geom = None
    try:
        describe = describe__galaxy_catalogue  # from galaxy_catalogue, inlined above
        geom = hyperleda_geometry(args.galaxy)
        print("")
        print("HyperLEDA geometry for " + args.galaxy + ":")
        print(describe(geom))
    except Exception as exc:
        print("Catalogue lookup failed: {}".format(exc))

    centre = (_probe_shape[1] / 2.0, _probe_shape[0] / 2.0)
    try:
        from astropy.coordinates import SkyCoord
        c = SkyCoord.from_name(args.galaxy)
        px, py = _probe_wcs.world_to_pixel(c)
        centre = (float(px), float(py))
        print("Galaxy centre on the V grid: ({:.1f}, {:.1f})".format(*centre))
    except Exception as exc:
        print("WARNING: could not resolve the centre ({}); using the frame centre"
              .format(exc))

    _probe_scale = None
    try:
        _probe_scale = frame_geometry((_groups.get("V") or [args.v])[0],
                                      _donors.get("V"))[2].get("pixel_scale")
    except Exception:
        pass
    r25_px = None
    if geom and geom.get("d25_arcmin") and _probe_scale:
        r25_px = 0.5 * geom["d25_arcmin"] * 60.0 / float(_probe_scale)

    print("Reading the four frames; V is the reference grid")
    v_data, v_wcs, v_bpm, v_info = combine_exposures(
        _groups.get("V") or [args.v], _donors.get("V"), "V", positions=_anchors)
    b_data, b_wcs, b_bpm, b_info = combine_exposures(
        _groups.get("B") or [args.b], _donors.get("B"), "B", positions=_anchors)
    ha, ha_wcs, ha_bpm, ha_info = combine_exposures(
        _groups.get("Ha") or [args.halpha], _donors.get("Ha"), "H-alpha",
        positions=_anchors)
    red, red_wcs, red_bpm, red_info = combine_exposures(
        _groups.get("R") or [args.red], _donors.get("R"), "R", positions=_anchors)
    for nm, inf, pth in (("B", b_info, args.b), ("V", v_info, args.v),
                         ("R", red_info, args.red), ("H-alpha", ha_info, args.halpha)):
        if inf:
            print("  {:<8} {:<9} {:>6.0f}s  {}".format(
                nm, inf["filter"], inf["exptime"], os.path.basename(pth)))

    pixel_scale = v_info["pixel_scale"] or ha_info["pixel_scale"]
    if pixel_scale:
        print("  pixel scale {:.4f} arcsec/px".format(float(pixel_scale)))

    if v_wcs is None:
        print("The V frame has no WCS, so there is no reference grid to work on.")
        return 1

    north_angle, mirrored = image_orientation(v_wcs)
    print("  orientation from the V WCS: north {:.1f} deg from +x, {}".format(
        north_angle, "mirrored" if mirrored else "not mirrored"))

    print("Registering B, H-alpha and R onto the V grid")

    def onto_v(data, wcs, name):
        """Anchor stars first; the plate solutions are the fallback."""
        if len(fg_stars):
            transform, adiag = anchor_transform(v_data, data, fg_stars)
            if transform is not None:
                print("  " + name)
                print("  " + describe_anchors(adiag))
                return apply_transform(data, transform, v_data.shape)
        if wcs is not None and v_wcs is not None:
            return align_to(v_wcs, v_data.shape, data, wcs)
        return data

    b_data = onto_v(b_data, b_wcs, "B")
    ha = onto_v(ha, ha_wcs, "H-alpha")
    red = onto_v(red, red_wcs, "R")

    # Everything below measures something at these positions - the star widths,
    # the subtraction factor, the leftover shift - and a catalogue position is not
    # a star. On M83 they sat a few pixels off, and every one of those measurements
    # was made around a point with no star at it.
    if len(fg_stars):
        fg_stars, _cdiag = centre_on_stars(v_data, fg_stars)
        print(describe_centring(_cdiag))

    # Each plate solution was fitted on its own frame and carries its own error of
    # a fraction of an arcsecond, so resampling through two of them leaves a shift
    # of well under a pixel behind. It is small, but this tool measures separations
    # of a few hundred parsecs - a pixel here is sixteen of them - and the same
    mask = None
    for bp, w in ((ha_bpm, ha_wcs), (red_bpm, red_wcs)):
        if bp is None:
            continue
        if w is not None:
            bp = align_to(v_wcs, v_data.shape, bp.astype(float), w) > 0.5
        mask = bp if mask is None else (mask | bp)
    if v_bpm is not None:
        mask = v_bpm if mask is None else (mask | v_bpm)

    # ---- the HII regions ----------------------------------------------------
    # The anchor fit above put the frames on one grid, but a similarity transform
    # fitted on four stars still leaves a fraction of a pixel, and the subtraction
    # is unforgiving about it: half a pixel on a star three pixels wide leaves a
    # third of that star behind, as a dark core beside a bright crescent. It has to
    # come out before the widths are matched, because a shifted pair also measures
    # its widths wrong.
    print("Checking what the registration left behind")
    _pos = fg_stars if len(fg_stars) else None
    ha = refine_alignment(red, ha, "H-alpha against R", positions=_pos)

    _fw_stars = fg_stars if len(fg_stars) else None
    f_ha = measure_fwhm(ha, mask=mask, positions=_fw_stars)
    f_r = measure_fwhm(red, mask=mask, positions=_fw_stars)
    print("  FWHM  H-alpha {:.2f} px   red {:.2f} px  (reported, not acted on)"
          .format(f_ha or float("nan"), f_r or float("nan")))

    # Everything this run writes, and where. Settled here because the first files
    # go out a few lines below.
    prefix = output_prefix(args.out_prefix,
                           os.path.dirname(os.path.abspath(args.v)), args.galaxy)
    _stem, _dir_hii, _dir_clusters, _dir_compare = output_folders(prefix)
    hii_prefix = os.path.join(_dir_hii, _stem)
    cluster_prefix = os.path.join(_dir_clusters, _stem)
    compare_prefix = os.path.join(_dir_compare, _stem)

    # The two frames the subtraction is about, exactly as they enter it: registered
    # onto the V grid, the leftover shift taken out, the widths matched. Everything
    # afterwards is one arithmetic step from these two, so keeping them means the
    # factor or the blur can be reconsidered later without repeating the hour of
    # combining and registering that produced them.
    _prefix_now = hii_prefix
    for _name, _frame in (("Ha", ha), ("R", red)):
        save_frame("{}_{}_on_V_grid.fits".format(_prefix_now, _name), _frame, v_wcs,
                   {"filter": _name, "object": args.galaxy,
                    "exptime": (ha_info if _name == "Ha" else red_info).get("exptime")},
                   history=("Registered onto the V grid, shift removed, widths matched",
                            "The frame the continuum subtraction works on"))
    print("Kept the two frames the subtraction works on, on the V grid:")
    print("  {}_Ha_on_V_grid.fits and {}_R_on_V_grid.fits"
          .format(os.path.basename(_prefix_now), os.path.basename(_prefix_now)))

    # The blur and the factor are decided together, and by eye. Every automatic
    # rule tried here disagreed with what the frames look like.
    _pick_prefix = hii_prefix
    ha, red, _picked, _chdiag = choose_subtraction(
        ha, red, _fw_stars, _pick_prefix, centre=centre, r25_px=r25_px,
        ask=(None if args.constant_factor else ask))
    if _picked is not None:
        print(describe_choice(_chdiag))

    print("Setting the subtraction factor")
    pass  # from continuum_subtraction, inlined above
    describe_continuum = describe__continuum_subtraction  # from continuum_subtraction, inlined above

    # The factor is measured between the two frames it applies to, R and H-alpha,
    # and nothing else goes into it. B and V are read by this tool for the blue
    # clusters; they have no part in deciding how much continuum to take out.
    if _picked is not None:
        factor = float(_picked)
        factor_note = "factor {:.3f}, chosen from the three pictures".format(factor)
        print("  factor {:.3f}, as chosen above".format(factor))
    else:
        factor, cdiag = constant_factor(
            ha, red, mask=mask,
            positions=fg_stars if len(fg_stars) else None)
        print(describe_continuum(dict(cdiag, constant_factor=factor)))
        factor_note = "one factor, {:.3f}, measured on R and H-alpha".format(factor)
    mean_factor = float(factor)

    net, sky_diag = subtract_continuum(ha, red, factor, mask=mask)
    print("  sky removed first: {:.1f} counts from H-alpha, {:.1f} from R"
          .format(sky_diag["sky_ha"], sky_diag["sky_red"]))

    # Named for what it is, not for what was done to it. "_subtracted" sits in a
    # folder beside a dozen other files and says nothing about which frame it is
    # or which operation produced it.
    _sub_path = hii_prefix + "_Ha_minus_continuum.fits"
    save_frame(_sub_path, net, v_wcs, ha_info, history=(
        "H-alpha with the continuum subtracted, on the V grid",
        "Made by Star_Formation_Full_Analysis.py"))
    print("Wrote the H-alpha frame with the continuum taken out:")
    print("  " + _sub_path)

    # The same frame as a picture. A FITS file has to be opened in something that
    # reads FITS; this one opens anywhere, and it is the frame everything below
    # depends on, so it is the one worth looking at before trusting any of it.
    _look_path = hii_prefix + "_Ha_minus_continuum.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        from astropy.stats import sigma_clipped_stats as _stats
        _nz = float(_stats(net, sigma=3.0)[2])
        _fig, _ax = _plt.subplots(figsize=(10, 10))
        _ax.imshow(net, origin="lower", cmap="gray", vmin=-3 * _nz, vmax=10 * _nz)
        _ax.set_xticks([]); _ax.set_yticks([])
        _ax.set_title("{}  -  H-alpha minus the continuum\n"
                      "black is negative; the scale runs from -3 to +10 times "
                      "the sky noise".format(args.galaxy or "galaxy"), fontsize=12)
        _fig.tight_layout()
        _fig.savefig(_look_path, dpi=110, bbox_inches="tight")
        _plt.close(_fig)
        print("  " + _look_path + "   (the same frame as a picture)")
    except Exception as _exc:
        print("  could not write the picture: {}".format(_exc))

    # Two tracers, two boundaries, and they are not the same number. Emission is
    # counted on the subtracted frame, so that frame says where the emission
    # stops. Blue clusters are found in B, so B says where they stop being
    # findable - and old starlight reaches further out than the gas still forming
    # stars, so measuring both from one image would put the line in the wrong
    # place for at least one of them.
    hii_r, hii_diag = visible_radius(net, centre, threshold=args.extent_sigma)
    print(describe_extent(hii_r, hii_diag, r25_px, "H-alpha emission"))
    knot_r, knot_diag = visible_radius(b_data, centre, threshold=args.extent_sigma)
    print(describe_extent(knot_r, knot_diag, r25_px, "B-band light"))

    # Both reported, neither applied yet - see the note in the other tool. What
    # they are good for is telling how deep the exposure went.
    r25_hii = r25_px
    r25_knots = r25_px

    window = galaxy_window(net.shape, centre, r25_hii, margin=args.window_margin)
    print(describe_window(window, net.shape, args.window_margin))
    net_w = cut(net, window)
    mask_w = cut(mask, window) if mask is not None else None

    det_kwargs = dict(threshold_sigma=args.threshold, min_pixels=args.min_pixels,
                      mask=mask_w, deblend=not args.no_deblend,
                      box_size=args.background_box,
                      nlevels=args.deblend_levels, contrast=args.deblend_contrast,
                      r25_px=r25_px)
    cat, bkg, resid = detect_regions(net_w, **det_kwargs)
    if cat is None or len(cat) == 0:
        print("No HII regions found. Try a lower --threshold.")
        return 1
    # Straight back into the coordinates of the V grid, which is the language
    # everything downstream speaks - the knots, the pairing, the plots.
    hx, hy = restore_positions(np.asarray(cat.xcentroid, dtype=float),
                               np.asarray(cat.ycentroid, dtype=float), window)
    # Shape, straight from the segmentation: a region has a body, a subtraction
    # residual is a thread a pixel or two across.
    _minor = np.asarray(cat.semiminor_sigma.value, dtype=float)
    _major = np.asarray(cat.semimajor_sigma.value, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        _elong = _major / np.where(_minor > 1e-6, _minor, np.nan)
    ok = np.isfinite(hx) & np.isfinite(hy)
    hx, hy = hx[ok], hy[ok]
    _minor, _elong = _minor[ok], _elong[ok]
    print("Detected {} HII regions at {:.1f} sigma".format(len(hx), args.threshold))

    r_max_px = args.max_radius_r25 * r25_hii if (args.max_radius_r25 and r25_hii) else None
    if r_max_px:
        print("Keeping detections inside {:.2f} x R25 = {:.0f} px".format(
            args.max_radius_r25, r_max_px))

    if args.no_star_cut:
        star_xy = None
    elif len(fg_stars):
        star_xy = fg_stars
    else:
        star_xy = star_positions(red, mask=mask)
    excess = line_excess(net, red, hx, hy)
    keep, report = filter_detections(hx, hy, centre, r_max_px, star_xy,
                                     args.star_radius, excess=excess,
                                     minor_axis=_minor, elongation=_elong,
                                     min_width_px=args.min_width,
                                     max_elongation=args.max_elongation)
    hx, hy = hx[keep], hy[keep]
    print("Filtered to {} HII regions ({})".format(
        len(hx), ", ".join("{} {}".format(v, k) for k, v in report.items())
        or "nothing removed"))

    # The catalogue predicts the shape a tilted disk's tracers should have, so it
    # can be compared with the shape actually detected. This is the only check
    # here that catches a detection which has gone wrong while still drawing a
    # perfectly believable profile.
    if geom and geom.get("b_over_a") and len(hx):
        _expected_angle = None
        if geom.get("pa_sky") is not None:
            _expected_angle = sky_pa_to_image_angle(geom["pa_sky"], north_angle,
                                                    mirrored)
        print(describe_check(check_against_catalogue(
            hx - centre[0], hy - centre[1], geom["b_over_a"], _expected_angle)))

    # Where everything this run writes will go. Settled here rather than further
    # down because the blue-cluster stage needs somewhere to put the registered
    # frames it is given.


    # ---- the blue knots -----------------------------------------------------
    if args.knots_csv:
        knots_csv = args.knots_csv
        print("Using the blue-knot catalogue you gave: " + os.path.basename(knots_csv))
    else:
        # Hand the blue-cluster tool the frames as this tool is holding them, not
        # the files on disk. Two reasons, and the second is the important one.
        #
        # A combined frame usually has no plate solution left in it, so the file on
        # disk cannot even be read by a tool that needs one - it fails on the first
        # coordinate conversion.
        #
        # And these copies have already been put on the V grid and had the leftover
        # sub-pixel shift taken out. Measuring the knots on the raw files instead
        # would put that shift straight into their positions, and this tool's whole
        # measurement is a separation of a few hundred parsecs - about forty pixels
        # here. The one grid rule has to hold for the knots too, not only for the
        # regions.
        prep = _dir_clusters
        b_path, v_path = write_registered(prep, args.galaxy, b_data, v_data, v_wcs,
                                          b_info, v_info)
        print("Wrote the registered B and V frames for the blue-cluster tool:")
        print("  " + os.path.basename(b_path) + " and " + os.path.basename(v_path))
        knots_csv = run_blue_cluster_tool(b_path, v_path, args.galaxy,
                                          args.distance_mpc, script_dir,
                                          ebv=_ebv)
        print("  knot catalogue: " + os.path.basename(knots_csv))
    kx, ky = read_knots(knots_csv)
    print("Read {} blue knots, already on the V grid".format(len(kx)))

    # The same limit the HII regions get. The blue-cluster tool measures every blue
    # source in the frame, and a frame is much wider than a galaxy: on the M83 set
    # the knots reached 25 kpc from the centre while the HII regions stopped at 9,
    # which is to say most of the outermost ones are foreground stars and background
    # galaxies rather than clusters in this galaxy at all. Left in, they pair with
    # whatever region happens to be nearest - kiloparsecs away - and drag the mean
    # separation to fifteen times the median while manufacturing a steep trend with
    # radius out of nothing.
    knot_max_px = (args.max_radius_r25 * r25_knots
                   if (args.max_radius_r25 and r25_knots) else None)
    if knot_max_px:
        inside = np.hypot(kx - centre[0], ky - centre[1]) <= knot_max_px
        if (~inside).any():
            print("  {} of them lie beyond {:.0f} px from the centre, outside the "
                  "galaxy, and are dropped".format(int((~inside).sum()), knot_max_px))
        kx, ky = kx[inside], ky[inside]

    if len(kx) == 0:
        print("No blue knots to match. Stopping here.")
        return 1

    # ---- units and outputs --------------------------------------------------
    scale, unit = 1.0, "px"
    if pixel_scale:
        scale, unit = float(pixel_scale), "arcsec"
        if args.distance_mpc:
            scale = float(pixel_scale) * args.distance_mpc * 1e6 / 206264.806
            unit = "pc"

    frame = (0.0, float(v_data.shape[1]), 0.0, float(v_data.shape[0]))

    print("")
    print("=== radial analysis: HII regions ===")
    _, hii_res, _ = run_profiles(
        hx, hy, centre, frame, hii_prefix + "_HII",
        "HII regions - " + args.galaxy, geom=geom, north_angle=north_angle,
        mirrored=mirrored, sector_method=args.sector_method, rings=args.rings,
        scale=scale, unit=unit)

    print("")
    print("=== radial analysis: blue knots ===")
    _, knot_res, _ = run_profiles(
        kx, ky, centre, frame, cluster_prefix + "_knots",
        "Blue knots - " + args.galaxy, geom=geom, north_angle=north_angle,
        mirrored=mirrored, sector_method=args.sector_method, rings=args.rings,
        scale=scale, unit=unit)

    # ---- the pairing --------------------------------------------------------
    print("")
    print("=== pairing blue knots with HII regions ===")
    hii_profile = hii_res[0]
    good = hii_profile["complete"]
    pairs = match_report(
        np.column_stack([kx, ky]), np.column_stack([hx, hy]), centre,
        hii_profile["r"][good] * scale,
        hii_profile["density"][good] / scale ** 2,
        scale=scale, unit=unit)
    print(describe_match(pairs))

    match_path = compare_prefix + "_matches.csv"
    with open(match_path, "w", encoding="utf-8") as fh:
        fh.write("# blue knots paired with their nearest HII region, both measured "
                 "on the V grid\n")
        fh.write("# separations in {}\n".format(unit))
        fh.write("knot_x,knot_y,hii_x,hii_y,separation,knot_radius,"
                 "expected_from_density,excess_ratio\n")
        for i in range(len(kx)):
            j = pairs["nearest_index"][i]
            fh.write("{:.3f},{:.3f},{:.3f},{:.3f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                kx[i], ky[i], hx[j], hy[j], pairs["separation"][i],
                pairs["radius"][i], pairs["expected"][i], pairs["excess_ratio"][i]))
    print("Wrote " + match_path)

    fig = both_populations_figure(
        np.column_stack([kx, ky]), np.column_stack([hx, hy]), pairs, centre,
        scale, unit, "Both tracers - " + args.galaxy)
    p1 = compare_prefix + "_both_populations.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote " + p1)

    fig = separation_figure(
        pairs, unit, "Knot to HII separation against radius - " + args.galaxy)
    p2 = compare_prefix + "_separation_vs_radius.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote " + p2)
    return 0

# --------------------------------------------------------------------------- #
# The blue-cluster tool, carried inside this file
# --------------------------------------------------------------------------- #

_CLUSTER_TOOL_SOURCE = "IyAtKi0gY29kaW5nOiB1dGYtOCAtKi0NCiIiIg0KQ3JlYXRlZCBieTogRHIuIEJvYXogUm9uIFpvaGFyDQpodHRwczovL2dpdGh1Yi5jb20vQm9helJvblpvaGFyL09ic2VydmF0aW9uYWxBc3Ryb25vbXlFZHVjYXRpb25Ub29scy9ibG9iL21haW4vQ2x1c3Rlcl9BbmRfR2FsYXh5X0NNRC9HYWxheHlfQ01ELnB5DQpBZmZpbGlhdGlvbjogS2lubmVyZXQgT2JzZXJ2YXRvcnkNCk1lbWJlciBvZiB0aGUgTENPIEdsb2JhbCBTa3kgUGFydG5lcnMgcHJvZ3JhbW1lDQpEYXRlOiBTZXB0ZW1iZXIgMjAyNQ0KDQpXcml0dGVuIGZvciBzdHVkZW50IHByb2plY3RzIG9uIHNwaXJhbCBnYWxheGllcy4NCg0KV0hBVCBJVCBET0VTDQoNClRha2VzIG9uZSBCLWJhbmQgYW5kIG9uZSBWLWJhbmQgaW1hZ2Ugb2YgYSBnYWxheHkgYW5kIHByb2R1Y2VzIGl0cw0KY29sb3VyLW1hZ25pdHVkZSBkaWFncmFtLCB0b2dldGhlciB3aXRoIHRoZSByYWRpYWwgZGVuc2l0eSBwcm9maWxlIG9mIHRoZQ0KYmx1ZSBrbm90cyAtIHRoZSB5b3VuZyBzdGFyLWZvcm1pbmcgcmVnaW9ucyBpbiB0aGUgYXJtcy4NCg0KVGhlIGNvbG91ciBvZiBhIHNvdXJjZSBzYXlzIGhvdyBob3QgYW5kIGhvdyB5b3VuZyBpdCBpcy4gUGxvdHRpbmcgY29sb3VyDQphZ2FpbnN0IGJyaWdodG5lc3MgZm9yIGV2ZXJ5IHNvdXJjZSBpbiB0aGUgZ2FsYXh5IHNlcGFyYXRlcyB0aGUgYmx1ZSBrbm90cw0KZnJvbSB0aGUgb2xkZXIgcG9wdWxhdGlvbiwgYW5kIHRoZSByYWRpYWwgcHJvZmlsZSB0aGVuIHNob3dzIGhvdyB0aGV5IGFyZQ0KZGlzdHJpYnV0ZWQgd2l0aCBkaXN0YW5jZSBmcm9tIHRoZSBjZW50cmUgLSB3aGljaCBpcyB3aGF0IGEgc3BpcmFsIGFybQ0KbG9va3MgbGlrZSBpbiBudW1iZXJzLg0KDQpXSEVSRSBUSEUgTUVUSE9EIENPTUVTIEZST00NCg0KSWRlbnRpZnlpbmcgc3Rhci1mb3JtaW5nIHJlZ2lvbnMgYnkgdGhlaXIgY29sb3VyIGluIG9yZGluYXJ5IGJyb2FkLWJhbmQNCmltYWdlcywgYW5kIHRoZW4gdHJlYXRpbmcgdGhlaXIgZGlzdHJpYnV0aW9uIGFzIGEgbWVhc3VyYWJsZSBwcm9wZXJ0eSBvZiB0aGUNCmdhbGF4eSwgZm9sbG93cw0KDQogICAgQnJvc2NoLCBOLiAoMTk5MikuIFN0YXIgZm9ybWF0aW9uIHN5c3RlbWF0aWNzIGZyb20gY29sb3VyIGltYWdlcy4NCiAgICBBc3Ryb3BoeXNpY3MgYW5kIFNwYWNlIFNjaWVuY2UsIDE4OCwgMjg5LTI5OC4NCiAgICBkb2k6MTAuMTAwNy9CRjAwNjQ0OTE2DQoNClRoZSBhcHBlYWwgZm9yIHRlYWNoaW5nIGlzIHRoYXQgaXQgYXNrcyBub3RoaW5nIGV4b3RpYy4gVHdvIGJyb2FkLWJhbmQgZnJhbWVzDQpvZiBhIGdhbGF4eSwgQiBhbmQgViwgYXJlIHdpdGhpbiByZWFjaCBvZiBhIHNjaG9vbC1hY2Nlc3NpYmxlIHRlbGVzY29wZSwgYW5kDQp0aGUgeW91bmcgcmVnaW9ucyBzZXBhcmF0ZSBvdXQgb24gY29sb3VyIGFsb25lIC0gbm8gc3BlY3Ryb3Njb3B5LCBubyBuYXJyb3ctDQpiYW5kIGZpbHRlci4gQSBzdHVkZW50IHdpdGggb25lIG5pZ2h0IG9mIGRhdGEgY2FuIGFzayBhIHJlYWwgcXVlc3Rpb24gYWJvdXQNCndoZXJlIGEgZ2FsYXh5IGlzIGZvcm1pbmcgc3RhcnMsIGFuZCBhbnN3ZXIgaXQgd2l0aCBhIG51bWJlci4NCg0KVGhpcyBzY3JpcHQgaXMgdGhlIHBpcGVsaW5lIGZvciB0aGF0LCBhbmQgaGFzIGJlZW4gdXNlZCBhY3Jvc3MgbWFueSBzdHVkZW50DQpwcm9qZWN0cy4NCg0KV0hBVCBZT1UgTkVFRA0KDQogIHR3byBGSVRTIGltYWdlcyBvZiB0aGUgc2FtZSBnYWxheHksIG9uZSBpbiBCIGFuZCBvbmUgaW4gViwgcGxhdGUtc29sdmVkDQogICh0aGUgcmVmZXJlbmNlIHN0YXJzIGFyZSBwdWxsZWQgZnJvbSBhIGNhdGFsb2d1ZSBieSBza3kgcG9zaXRpb24pDQoNCldIQVQgSVQgQVNLUyBZT1UNCg0KICBnYWxheHkgbmFtZSAgICAgICAgICAgICAgICAgICAgdXNlZCB0byBuYW1lIHRoZSBvdXRwdXQgZmlsZXMsIGFuZCB0byBsb29rDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGUgbmV4dCB0d28gdXANCiAgZGlzdGFuY2UgaW4gTXBjICAgICAgICAgICAgICAgIGNvbnZlcnRzIHBpeGVscyB0byBwYXJzZWNzIGluIHRoZSBwcm9maWxlLg0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgT2ZmZXJlZCBmcm9tIENvc21pY2Zsb3dzLTMgKFR1bGx5KyAyMDE2KQ0KICBjb2xvdXIgZXhjZXNzIEUoQi1WKSAgICAgICAgICAgY29ycmVjdHMgYnJpZ2h0bmVzcyBhbmQgY29sb3VyIGZvciB0aGUgZHVzdA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgb2Ygb3VyIG93biBnYWxheHkuIE9mZmVyZWQgZnJvbSB0aGUgU2NobGFmbHkNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICYgRmlua2JlaW5lciAoMjAxMSkgZHVzdCBtYXBzIHZpYSBJUlNBLg0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgQV9WIGlzIG5vdCBhc2tlZCBmb3I6IGl0IGlzIDMuMSB4IEUoQi1WKQ0KICB0aGUgdHdvIEZJVFMgcGF0aHMNCiAgaG93IG1hbnkgcmVmZXJlbmNlIHN0YXJzIHRvIGNhbGlicmF0ZSBhZ2FpbnN0DQoNCkV2ZXJ5IG9uZSBvZiB0aGVzZSBoYXMgYSBkZWZhdWx0LCBhbmQgdGhlIHR3byBjYXRhbG9ndWUgdmFsdWVzIGFyZSBmZXRjaGVkIGJ5DQpuYW1lLiBQcmVzcyBFbnRlciB0byBhY2NlcHQgdGhlbSBvciB0eXBlIHlvdXIgb3duLg0KDQpIT1cgSVQgV09SS1MNCg0KICAxLiBzdWJ0cmFjdHMgdGhlIGJhY2tncm91bmQgZnJvbSBlYWNoIGltYWdlIGFuZCBmaW5kcyBzb3VyY2VzIGluIGJvdGgNCiAgMi4gbWVhc3VyZXMgZWFjaCBzb3VyY2Ugd2l0aCBhbiBhcGVydHVyZSBzaXplZCBmcm9tIGl0cyBvd24gRldITQ0KICAzLiBtYXRjaGVzIHRoZSBCIGFuZCBWIGRldGVjdGlvbnMgdG8gZWFjaCBvdGhlciBieSBwb3NpdGlvbg0KICA0LiBwdWxscyByZWZlcmVuY2Ugc3RhcnMgZnJvbSBBUEFTUzkgdGhyb3VnaCBWaXppZXIsIG1hdGNoZXMgdGhlbSB0byB0aGUNCiAgICAgZGV0ZWN0aW9ucywgYW5kIGNhbGlicmF0ZXMgdGhlIGluc3RydW1lbnRhbCBtYWduaXR1ZGVzIGFnYWluc3QgdGhlbQ0KICA1LiByZW1vdmVzIHRoZSBmb3JlZ3JvdW5kIHN0YXJzLCBsZWF2aW5nIHRoZSBnYWxheHkncyBvd24gc291cmNlcw0KICA2LiBwbG90cyB0aGUgY29sb3VyLW1hZ25pdHVkZSBkaWFncmFtLCBhbmQgdGhlIHJhZGlhbCBkZW5zaXR5IHByb2ZpbGUgb2YNCiAgICAgd2hhdCBpcyBsZWZ0LCBpbiBwaXhlbHMgYW5kIGFnYWluIGluIHBhcnNlY3MNCg0KV0hBVCBZT1UgR0VUDQoNCkFyb3VuZCB0d2VudHkgZmlsZXMsIGFsbCBuYW1lZCBhZnRlciB0aGUgZ2FsYXh5LiBUaGV5IGdvIGludG8gYSBzdWItZm9sZGVyDQpjYWxsZWQgcmVzdWx0X0NNRCwgY3JlYXRlZCBuZXh0IHRvIHlvdXIgaW1hZ2VzIC0gdGhlIGZvbGRlciBob2xkaW5nIHlvdXIgZGF0YQ0KY29tZXMgb3V0IG9mIGEgcnVuIGV4YWN0bHkgYXMgaXQgd2VudCBpbiwgd2l0aCB0aGUgZnJhbWVzIG9ubHkgcmVhZC4NCg0KVGhhdCBpcyBhIGxvdCBvZiBmaWxlcywgc28gaGVyZSBpcyB3aGF0IGVhY2ggb25lIGlzIGFuZCB3aGljaCB5b3UgYWN0dWFsbHkNCndhbnQgdG8gbG9vayBhdC4gQmVsb3csIE5BTUUgc3RhbmRzIGZvciB0aGUgZ2FsYXh5IG5hbWUgeW91IHR5cGVkLg0KDQogIEEgYnktcHJvZHVjdCwga2VwdCBiZWNhdXNlIGl0IGlzIHdoYXQgZXZlcnl0aGluZyBpcyBtZWFzdXJlZCBvbjoNCg0KICAgIDxpbnB1dD5fYmdzdWIuZml0cyAgICAgICAgICAgIGVhY2ggaW5wdXQgaW1hZ2Ugd2l0aCBpdHMgYmFja2dyb3VuZA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHN1YnRyYWN0ZWQuIFVzZWZ1bCBpZiB5b3Ugd2FudCB0byBjaGVjayBhDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgc291cmNlIGJ5IGV5ZS4NCg0KICBTdGFnZSAxIC0gZXZlcnl0aGluZyB0aGF0IHdhcyBtZWFzdXJlZA0KDQogICAgTkFNRV9waG90b21ldHJ5X3Jlc3VsdHMuY3N2ICAgZXZlcnkgc291cmNlIGRldGVjdGVkIGluIEJPVEggaW1hZ2VzIGFuZA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1hdGNoZWQgYmV0d2VlbiB0aGVtOiBwb3NpdGlvbiwgRldITSwNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBhcGVydHVyZSByYWRpdXMsIGZsdXggaW4gQiBhbmQgaW4gVi4NCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBJbnN0cnVtZW50YWwgdmFsdWVzLCBub3QgeWV0IGNhbGlicmF0ZWQuDQogICAgTkFNRV9yZWZlcmVuY2Vfc3RhcnMuY3N2ICAgICAgdGhlIEFQQVNTOSBzdGFycyBmb3VuZCBpbiB0aGUgZmllbGQsIHdpdGgNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGVpciBjYXRhbG9ndWUgQiBhbmQgVi4gVGhlc2Ugc2V0IHRoZSB6ZXJvDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcG9pbnQuIElmIHRoZSBjYWxpYnJhdGlvbiBsb29rcyB3cm9uZywgbG9vaw0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGhlcmUgZmlyc3QgLSBhcmUgdGhlcmUgZW5vdWdoIG9mIHRoZW0sIGFuZA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFyZSB0aGV5IHNwcmVhZCBhY3Jvc3MgdGhlIGZyYW1lPw0KDQogIFN0YWdlIDIgLSBjYWxpYnJhdGVkLCBzdGlsbCB3aXRoIGZvcmVncm91bmQgc3RhcnMgaW4gaXQNCg0KICAgIE5BTUVfY2FsaWJyYXRlZF9waG90b21ldHJ5LmNzdiAgICAgICAgICAgIHJlYWwgbWFnbml0dWRlcw0KICAgIE5BTUVfY2FsaWJyYXRlZF9waG90b21ldHJ5X3dpdGhfY29sb3IuY3N2IHRoZSBzYW1lLCB3aXRoIEItViBhZGRlZA0KICAgIE5BTUVfY2FsaWJyYXRlZF9waG90b21ldHJ5X0NNRC5wbmcgICAgICAgIGNvbG91ci1tYWduaXR1ZGUgZGlhZ3JhbQ0KICAgIE5BTUVfVl9zb3VyY2VzLnBuZyAgICAgICAgICAgICAgICAgICAgICAgIHRoZSBWIGltYWdlIHdpdGggZXZlcnkgbWVhc3VyZWQNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBzb3VyY2UgY2lyY2xlZA0KICAgIE5BTUVfQl9zb3VyY2VzLnBuZyAgICAgICAgICAgICAgICAgICAgICAgIHRoZSBzYW1lIGZvciBCLiBDb21wYXJlIHRoZSB0d286DQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhleSBzaG91bGQgZmluZCB0aGUgc2FtZSB0aGluZ3MNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpbiB0aGUgc2FtZSBwbGFjZXMNCg0KICBTdGFnZSAzIC0gZm9yZWdyb3VuZCBzdGFycyByZW1vdmVkDQoNCiAgICBTdGFycyBvZiBvdXIgb3duIGdhbGF4eSBsaWUgaW4gZnJvbnQgb2YgdGhlIHRhcmdldCBhbmQgYXJlIG5vdCBwYXJ0IG9mDQogICAgaXQuIFRoZXkgYXJlIG1hdGNoZWQgYWdhaW5zdCB0aGUgY2F0YWxvZ3VlIGFuZCB0YWtlbiBvdXQuDQoNCiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFycy5jc3YNCiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFyc193aXRoX2NvbG9yLmNzdg0KICAgIE5BTUVfY2FsaWJyYXRlZF9waG90b21ldHJ5X25vX3N0YXJzX0NNRC5wbmcNCiAgICBOQU1FX1Zfc291cmNlc19ub19zdGFycy5wbmcNCg0KICBTdGFnZSA0IC0gYmx1ZSBrbm90cyBvbmx5DQoNCiAgICBXaGF0IGlzIGxlZnQgaXMgY3V0IG9uIGNvbG91ciwga2VlcGluZyB0aGUgYmx1ZSBzb3VyY2VzOiB0aGUgeW91bmcNCiAgICBzdGFyLWZvcm1pbmcgcmVnaW9ucyBpbiB0aGUgYXJtcy4gVEhJUyBJUyBUSEUgUkVTVUxUIHRoZSBwcm9qZWN0cyBhcmUNCiAgICB1c3VhbGx5IGFmdGVyLg0KDQogICAgTkFNRV9jYWxpYnJhdGVkX3Bob3RvbWV0cnlfbm9fc3RhcnNfY29sb3JfZmlsdGVyZWQuY3N2DQogICAgTkFNRV9jYWxpYnJhdGVkX3Bob3RvbWV0cnlfbm9fc3RhcnNfY29sb3JfZmlsdGVyZWRfd2l0aF9jb2xvci5jc3YNCiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFyc19jb2xvcl9maWx0ZXJlZF9DTUQucG5nDQogICAgTkFNRV9WX3NvdXJjZXNfY29sb3JfZmlsdGVyZWQucG5nDQogICAgTkFNRV9jYWxpYnJhdGVkX3Bob3RvbWV0cnlfbm9fc3RhcnNfY29sb3JfZmlsdGVyZWRfd2l0aF9yYWRpdXMuY3N2DQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGUgc2FtZSwgcGx1cyBlYWNoIGtub3QncyBkaXN0YW5jZQ0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZnJvbSB0aGUgY2VudHJlIG9mIHRoZSBnYWxheHkNCg0KICBTdGFnZSA1IC0gaG93IHRoZSBrbm90cyBhcmUgZGlzdHJpYnV0ZWQNCg0KICAgIE5BTUVfcmFkaWFsX2RlbnNpdHlfcHJvZmlsZS5jc3YgICAgICAga25vdHMgcGVyIHNxdWFyZSBwaXhlbCwgYnkgcmFkaXVzDQogICAgTkFNRV9yYWRpYWxfZGVuc2l0eV9wcm9maWxlX3BjLmNzdiAgICB0aGUgc2FtZSBpbiBwYXJzZWNzLCB1c2luZyB0aGUNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRpc3RhbmNlIHlvdSBnYXZlDQogICAgTkFNRV9yYWRpYWxfZGVuc2l0eV9wcm9maWxlX3N0ZXAuY3N2ICAgICB0aGUgc2FtZSBhcyBhIHN0ZXAgZnVuY3Rpb24NCiAgICBOQU1FX3JhZGlhbF9kZW5zaXR5X3Byb2ZpbGVfc3RlcF9wYy5jc3YgIHRoZSBzdGVwIGZ1bmN0aW9uIGluIHBhcnNlY3MNCiAgICBOQU1FX3JhZGlhbF9kZW5zaXR5X3Byb2ZpbGVfc3RlcC5wbmcgICAgICAgdGhlIHBsb3QsIGluIHBpeGVscw0KICAgIE5BTUVfcmFkaWFsX2RlbnNpdHlfcHJvZmlsZV9zdGVwX3BjLnBuZyAgICB0aGUgcGxvdCwgaW4gcGFyc2Vjcw0KDQogIElmIHlvdSBsb29rIGF0IGZvdXIgZmlsZXMsIGxvb2sgYXQgdGhlc2U6DQoNCiAgICBOQU1FX3JlZmVyZW5jZV9zdGFycy5jc3YgICAgICAgICAgICAgICAgICAgZGlkIHRoZSBjYWxpYnJhdGlvbiBoYXZlDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFueXRoaW5nIHRvIHdvcmsgd2l0aA0KICAgIE5BTUVfVl9zb3VyY2VzX2NvbG9yX2ZpbHRlcmVkLnBuZyAgICAgICAgICBkaWQgaXQgZmluZCB0aGUgYXJtcw0KICAgIE5BTUVfLi4uX2NvbG9yX2ZpbHRlcmVkX0NNRC5wbmcgICAgICAgICAgICB0aGUgZGlhZ3JhbQ0KICAgIE5BTUVfcmFkaWFsX2RlbnNpdHlfcHJvZmlsZV9zdGVwX3BjLnBuZyAgICB0aGUgYXJtcywgaW4gbnVtYmVycw0KDQogIEEgX3dpdGhfY29sb3IgZmlsZSBpcyBpdHMgcGFyZW50IGZpbGUgd2l0aCB0aGUgQi1WIGNvbHVtbiBhZGRlZC4gSWYgeW91DQogIG9ubHkgd2FudCBvbmUsIHRha2UgdGhlIF93aXRoX2NvbG9yIG9uZS4NCg0KVGhlIHBhcmFtZXRlcnMgYXQgdGhlIHRvcCBvZiB0aGlzIGZpbGUgYXJlIHR1bmVkIGZvciB0aGUgS2lubmVyZXQgZnJhbWVzDQp0aGVzZSBwcm9qZWN0cyB1c2UuIE9uIHZlcnkgZGlmZmVyZW50IGRhdGEgLSBhIG11Y2ggc21hbGxlciB0ZWxlc2NvcGUsIGENCm11Y2ggZmFpbnRlciBnYWxheHkgLSB0aGUgZGV0ZWN0aW9uIHRocmVzaG9sZCBhbmQgdGhlIG1hdGNoaW5nIHRvbGVyYW5jZXMNCmFyZSB0aGUgZmlyc3QgdGhpbmdzIHRvIGxvb2sgYXQuDQoNClVzYWdlOiBydW4gaXQuIEV2ZXJ5IHF1ZXN0aW9uIGhhcyBhIGRlZmF1bHQ7IHByZXNzIEVudGVyIHRvIGFjY2VwdCBpdC4NCiIiIg0KDQojID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQ0KIyBHRU5FUkFURUQgRklMRSAtIGRvIG5vdCBlZGl0Lg0KIw0KIyBCdWlsdCBieSBidWlsZC5weSBvbiAyMDI2LTA5LTExIDEwOjUxIGZyb206DQojICAgQmx1ZV9DbHVzdGVyc19Gcm9tX0ltYWdlcy5weQ0KIyAgIGdhbGF4eV93aW5kb3cucHkNCiMgICBkZXByb2plY3Rpb24ucHkNCiMgICBnYWxheHlfY2F0YWxvZ3VlLnB5DQojICAgcmFkaWFsX3Byb2ZpbGVzLnB5DQojDQojIFRoaXMgZmlsZSBpcyBzdGFuZGFsb25lIG9uIHB1cnBvc2U6IGl0IG5lZWRzIG5vdGhpbmcgZWxzZSBmcm9tIHRoZQ0KIyBwcm9qZWN0LCBvbmx5IHRoZSB1c3VhbCB0aGlyZC1wYXJ0eSBwYWNrYWdlcy4gVGhlIGNvc3Qgb2YgdGhhdCBpcyB0aGlzDQojIHdhcm5pbmcgLSBhIGZpeCBtYWRlIGhlcmUgaXMgb3ZlcndyaXR0ZW4gYnkgdGhlIG5leHQgYnVpbGQuIENoYW5nZSB0aGUNCiMgc291cmNlIGFib3ZlIGluc3RlYWQsIHRoZW4gcnVuIGJ1aWxkLnB5IGFnYWluLg0KIyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0NCg0KDQpmcm9tIF9fZnV0dXJlX18gaW1wb3J0IGFubm90YXRpb25zDQoNCmltcG9ydCBudW1weSBhcyBucA0KaW1wb3J0IG1hdGgNCmltcG9ydCBvcw0KaW1wb3J0IHRleHR3cmFwDQppbXBvcnQgbWF0cGxvdGxpYg0KaW1wb3J0IG1hdHBsb3RsaWIucHlwbG90IGFzIHBsdA0KaW1wb3J0IHBhbmRhcyBhcyBwZA0KZnJvbSBhc3Ryb3B5LmlvIGltcG9ydCBmaXRzDQpmcm9tIHBob3R1dGlscy5kZXRlY3Rpb24gaW1wb3J0IERBT1N0YXJGaW5kZXINCmZyb20gcGhvdHV0aWxzLmFwZXJ0dXJlIGltcG9ydCBDaXJjdWxhckFwZXJ0dXJlLCBDaXJjdWxhckFubnVsdXMsIGFwZXJ0dXJlX3Bob3RvbWV0cnkNCmZyb20gcGhvdHV0aWxzLmJhY2tncm91bmQgaW1wb3J0IEJhY2tncm91bmQyRCwgTWVkaWFuQmFja2dyb3VuZA0KZnJvbSBhc3Ryb3B5LnN0YXRzIGltcG9ydCBzaWdtYV9jbGlwcGVkX3N0YXRzLCBTaWdtYUNsaXANCmZyb20gc2NpcHkubmRpbWFnZSBpbXBvcnQgZ2F1c3NpYW5fZmlsdGVyDQpmcm9tIGFzdHJvcHkud2NzIGltcG9ydCBXQ1MNCmZyb20gYXN0cm9xdWVyeS52aXppZXIgaW1wb3J0IFZpemllcg0KZnJvbSBhc3Ryb3B5LmNvb3JkaW5hdGVzIGltcG9ydCBTa3lDb29yZA0KaW1wb3J0IGFzdHJvcHkudW5pdHMgYXMgdQ0KaW1wb3J0IGFzdHJvYWxpZ24gYXMgYWENCg0KDQojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tDQojIHNoYXJlZCBtb2R1bGUgZ2FsYXh5X3dpbmRvdy5weQ0KIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQ0KDQpkZWYgZ2FsYXh5X3dpbmRvdyhzaGFwZSwgY2VudHJlLCByMjVfcHgsIG1hcmdpbj0xLjUpOg0KICAgICIiIlRoZSBzbGljZSBvZiBhIGZyYW1lIHdvcnRoIHdvcmtpbmcgb24sIGFzICh5MCwgeTEsIHgwLCB4MSkuDQoNCiAgICBSZXR1cm5zIE5vbmUgd2hlbiB0aGVyZSBpcyBubyByYWRpdXMgdG8gd29yayBmcm9tLCBvciB3aGVuIHRoZSB3aW5kb3cgd291bGQNCiAgICBjb3ZlciBlc3NlbnRpYWxseSB0aGUgd2hvbGUgZnJhbWUgYW55d2F5IC0gaW4gd2hpY2ggY2FzZSBjcm9wcGluZyB3b3VsZCBhZGQgYQ0KICAgIGJvb2trZWVwaW5nIHN0ZXAgYW5kIHNhdmUgbm90aGluZy4NCiAgICAiIiINCiAgICBpZiBub3QgcjI1X3B4IG9yIHIyNV9weCA8PSAwOg0KICAgICAgICByZXR1cm4gTm9uZQ0KDQogICAgbnksIG54ID0gc2hhcGUNCiAgICByZWFjaCA9IG1hcmdpbiAqIHIyNV9weA0KICAgIHgwID0gaW50KG1heCgwLCBucC5mbG9vcihjZW50cmVbMF0gLSByZWFjaCkpKQ0KICAgIHgxID0gaW50KG1pbihueCwgbnAuY2VpbChjZW50cmVbMF0gKyByZWFjaCkpKQ0KICAgIHkwID0gaW50KG1heCgwLCBucC5mbG9vcihjZW50cmVbMV0gLSByZWFjaCkpKQ0KICAgIHkxID0gaW50KG1pbihueSwgbnAuY2VpbChjZW50cmVbMV0gKyByZWFjaCkpKQ0KDQogICAgaWYgeDEgLSB4MCA8IDY0IG9yIHkxIC0geTAgPCA2NDoNCiAgICAgICAgcmV0dXJuIE5vbmUNCiAgICBpZiAoeDEgLSB4MCkgKiAoeTEgLSB5MCkgPiAwLjkgKiBueCAqIG55Og0KICAgICAgICByZXR1cm4gTm9uZQ0KICAgIHJldHVybiAoeTAsIHkxLCB4MCwgeDEpDQoNCmRlZiBjdXQoZGF0YSwgd2luZG93KToNCiAgICAiIiJUaGUgd2luZG93J3Mgd29ydGggb2YgYSBmcmFtZS4iIiINCiAgICBpZiB3aW5kb3cgaXMgTm9uZSBvciBkYXRhIGlzIE5vbmU6DQogICAgICAgIHJldHVybiBkYXRhDQogICAgeTAsIHkxLCB4MCwgeDEgPSB3aW5kb3cNCiAgICByZXR1cm4gZGF0YVt5MDp5MSwgeDA6eDFdDQoNCmRlZiBzaGlmdF9jZW50cmUoY2VudHJlLCB3aW5kb3cpOg0KICAgICIiIldoZXJlIHRoZSBnYWxheHkncyBjZW50cmUgc2l0cyBvbmNlIHRoZSBmcmFtZSBoYXMgYmVlbiBjdXQuIiIiDQogICAgaWYgd2luZG93IGlzIE5vbmU6DQogICAgICAgIHJldHVybiBjZW50cmUNCiAgICB5MCwgXywgeDAsIF8gPSB3aW5kb3cNCiAgICByZXR1cm4gKGNlbnRyZVswXSAtIHgwLCBjZW50cmVbMV0gLSB5MCkNCg0KZGVmIHNoaWZ0X3Bvc2l0aW9ucyhwb3NpdGlvbnMsIHdpbmRvdywgc2hhcGU9Tm9uZSk6DQogICAgIiIiU3RhciBwb3NpdGlvbnMgbW92ZWQgaW50byB0aGUgY3V0IGZyYW1lLCBrZWVwaW5nIG9ubHkgdGhvc2UgaW5zaWRlIGl0LiIiIg0KICAgIGlmIHdpbmRvdyBpcyBOb25lIG9yIHBvc2l0aW9ucyBpcyBOb25lIG9yIG5vdCBsZW4ocG9zaXRpb25zKToNCiAgICAgICAgcmV0dXJuIHBvc2l0aW9ucw0KICAgIHkwLCB5MSwgeDAsIHgxID0gd2luZG93DQogICAgcCA9IG5wLmFzYXJyYXkocG9zaXRpb25zLCBkdHlwZT1mbG9hdCkNCiAgICBpbnNpZGUgPSAoKHBbOiwgMF0gPj0geDApICYgKHBbOiwgMF0gPCB4MSkgJiAocFs6LCAxXSA+PSB5MCkgJiAocFs6LCAxXSA8IHkxKSkNCiAgICByZXR1cm4gbnAuY29sdW1uX3N0YWNrKFtwW2luc2lkZSwgMF0gLSB4MCwgcFtpbnNpZGUsIDFdIC0geTBdKQ0KDQpkZWYgcmVzdG9yZV9wb3NpdGlvbnMoeCwgeSwgd2luZG93KToNCiAgICAiIiJQb3NpdGlvbnMgbWVhc3VyZWQgaW4gdGhlIGN1dCBmcmFtZSwgcHV0IGJhY2sgd2hlcmUgdGhleSBiZWxvbmcuDQoNCiAgICBFdmVyeXRoaW5nIHRoZSB0b29sIHJlcG9ydHMgLSBjYXRhbG9ndWVzLCBwbG90cywgdGhlIHBhaXJpbmcgd2l0aCB0aGUgYmx1ZQ0KICAgIGtub3RzIC0gaXMgaW4gdGhlIGNvb3JkaW5hdGVzIG9mIHRoZSBvcmlnaW5hbCBmcmFtZSwgc28gbWVhc3VyZW1lbnRzIG1hZGUgaW4NCiAgICB0aGUgd2luZG93IGhhdmUgdG8gY29tZSBiYWNrIG91dCBvZiBpdC4gRm9yZ2V0dGluZyB0aGlzIHdvdWxkIHNoaWZ0IGV2ZXJ5DQogICAgcG9zaXRpb24gYnkgdGhlIHNpemUgb2YgdGhlIGNyb3AsIHdoaWNoIGlzIHRoZSBraW5kIG9mIGVycm9yIHRoYXQgbG9va3MgbGlrZQ0KICAgIGEgZGlzY292ZXJ5Lg0KICAgICIiIg0KICAgIGlmIHdpbmRvdyBpcyBOb25lOg0KICAgICAgICByZXR1cm4geCwgeQ0KICAgIHkwLCBfLCB4MCwgXyA9IHdpbmRvdw0KICAgIHJldHVybiBucC5hc2FycmF5KHgsIGR0eXBlPWZsb2F0KSArIHgwLCBucC5hc2FycmF5KHksIGR0eXBlPWZsb2F0KSArIHkwDQoNCmRlZiBkZXNjcmliZSh3aW5kb3csIHNoYXBlLCBtYXJnaW4pOg0KICAgICIiIldoYXQgd2FzIGN1dCwgYW5kIHdoYXQgaXQgc2F2ZWQuIiIiDQogICAgaWYgd2luZG93IGlzIE5vbmU6DQogICAgICAgIHJldHVybiAoIiAgd29ya2luZyBvbiB0aGUgd2hvbGUgZnJhbWUgLSBlaXRoZXIgbm8gaXNvcGhvdGFsIHJhZGl1cyBpcyAiDQogICAgICAgICAgICAgICAgImtub3duLCBvciB0aGUgZ2FsYXh5IGZpbGxzIGl0IikNCiAgICB5MCwgeTEsIHgwLCB4MSA9IHdpbmRvdw0KICAgIGtlcHQgPSAoeTEgLSB5MCkgKiAoeDEgLSB4MCkNCiAgICB3aG9sZSA9IHNoYXBlWzBdICogc2hhcGVbMV0NCiAgICByZXR1cm4gKCIgIHdvcmtpbmcgb24ge30geCB7fSBwaXhlbHMgYXJvdW5kIHRoZSBnYWxheHksIHs6LjBmfSUgb2YgdGhlIGZyYW1lICINCiAgICAgICAgICAgICIoezouMWZ9IHggUjI1KTsgdGhlIHJlc3QgaXMgc2t5IHRoYXQgd291bGQgYmUgZGlzY2FyZGVkIGFueXdheSINCiAgICAgICAgICAgIC5mb3JtYXQoeDEgLSB4MCwgeTEgLSB5MCwgMTAwLjAgKiBrZXB0IC8gd2hvbGUsIG1hcmdpbikpDQoNCmRlc2NyaWJlX19nYWxheHlfd2luZG93ID0gZGVzY3JpYmUgICAjIHRoaXMgbW9kdWxlJ3Mgb3duOyB0aGUgcGxhaW4gbmFtZSBnZXRzIHNoYWRvd2VkIGJlbG93DQoNCg0KIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQ0KIyBzaGFyZWQgbW9kdWxlIGRlcHJvamVjdGlvbi5weQ0KIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQ0KDQpJTkNfTUlOX0RFRyA9IDMwLjANCg0KSU5DX01BWF9ERUcgPSA3MC4wDQoNClEwX0RFRkFVTFQgPSAwLjEzDQoNClEwX0JZX1RZUEUgPSB7ImVhcmx5IjogMC4yMCwgIm1pZCI6IDAuMTMsICJsYXRlIjogMC4xMH0NCg0KZGVmIHEwX2Zvcl90eXBlKG10eXBlOiBzdHIgfCBOb25lKSAtPiBmbG9hdDoNCiAgICAiIiJQaWNrIGEgZGlzay10aGlja25lc3MgY29uc3RhbnQgZnJvbSB0aGUgbW9ycGhvbG9naWNhbCB0eXBlIHN0cmluZy4NCg0KICAgIEZhbGxzIGJhY2sgdG8gdGhlIG1pZC10eXBlIHZhbHVlIHdoZW4gdGhlIHR5cGUgaXMgdW5rbm93biBvciB1bnBhcnNhYmxlLA0KICAgIHdoaWNoIGlzIHRoZSBzYWZlIGNob2ljZTogaXQgc2l0cyBiZXR3ZWVuIHRoZSB0d28gZXh0cmVtZXMuDQogICAgIiIiDQogICAgaWYgbm90IG10eXBlOg0KICAgICAgICByZXR1cm4gUTBfREVGQVVMVA0KICAgIHQgPSBzdHIobXR5cGUpLnN0cmlwKCkudXBwZXIoKQ0KICAgIGlmIHQuc3RhcnRzd2l0aCgoIlMwIiwgIlNBIiwgIkUiKSk6DQogICAgICAgIHJldHVybiBRMF9CWV9UWVBFWyJlYXJseSJdDQogICAgaWYgdC5zdGFydHN3aXRoKCgiU0QiLCAiU00iLCAiSU0iLCAiSSIpKToNCiAgICAgICAgcmV0dXJuIFEwX0JZX1RZUEVbImxhdGUiXQ0KICAgIHJldHVybiBRMF9CWV9UWVBFWyJtaWQiXQ0KDQpkZWYgaW5jbGluYXRpb25fZnJvbV9heGlzX3JhdGlvKGJfb3Zlcl9hOiBmbG9hdCwgcTA6IGZsb2F0ID0gUTBfREVGQVVMVCkgLT4gZmxvYXQ6DQogICAgIiIiSW5jbGluYXRpb24gaW4gZGVncmVlcyBmcm9tIHRoZSBvYnNlcnZlZCBtaW5vci9tYWpvciBheGlzIHJhdGlvLg0KDQogICAgVXNlcyB0aGUgSHViYmxlIGZvcm11bGEsIHdoaWNoIGFjY291bnRzIGZvciB0aGUgZGlzayBoYXZpbmcgcmVhbCB0aGlja25lc3M6DQoNCiAgICAgICAgY29zXjIoaSkgPSAocV4yIC0gcTBeMikgLyAoMSAtIHEwXjIpDQoNCiAgICBBdCBtb2RlcmF0ZSBpbmNsaW5hdGlvbnMgdGhpcyBiYXJlbHkgZGlmZmVycyBmcm9tIHRoZSBuYWl2ZSBhY29zKGIvYSkuIEF0IGhpZ2gNCiAgICBpbmNsaW5hdGlvbiBpdCBtYXR0ZXJzIGEgbG90OiBmb3IgcSA9IDAuMiB0aGUgbmFpdmUgdmFsdWUgaXMgNzguNSBkZWcgd2hpbGUgdGhpcw0KICAgIGdpdmVzIDgxLjIgZGVnLCB3aGljaCBjaGFuZ2VzIHRoZSBzdHJldGNoIGZhY3RvciBmcm9tIDUuMCB0byA2LjUuDQoNCiAgICBSZXR1cm5zIDkwLjAgd2hlbiB0aGUgZ2FsYXh5IGlzIHRoaW5uZXIgdGhhbiBxMCwgaS5lLiBzZWVuIGVkZ2Utb24uDQogICAgIiIiDQogICAgcSA9IGZsb2F0KGJfb3Zlcl9hKQ0KICAgIGlmIHEgPD0gcTA6DQogICAgICAgIHJldHVybiA5MC4wDQogICAgY29zMiA9IChxICogcSAtIHEwICogcTApIC8gKDEuMCAtIHEwICogcTApDQogICAgY29zMiA9IG1pbihtYXgoY29zMiwgMC4wKSwgMS4wKQ0KICAgIHJldHVybiBtYXRoLmRlZ3JlZXMobWF0aC5hY29zKG1hdGguc3FydChjb3MyKSkpDQoNCmRlZiBzdHJldGNoX2ZhY3RvcihpbmNfZGVnOiBmbG9hdCkgLT4gZmxvYXQ6DQogICAgIiIiSG93IG11Y2ggdGhlIG1pbm9yLWF4aXMgZGlyZWN0aW9uIGhhcyB0byBiZSBzdHJldGNoZWQ6IDEgLyBjb3MoaSkuIiIiDQogICAgcmV0dXJuIDEuMCAvIG1hdGguY29zKG1hdGgucmFkaWFucyhpbmNfZGVnKSkNCg0KZGVmIGluY2xpbmF0aW9uX2lzX3VzYWJsZShpbmNfZGVnOiBmbG9hdCkgLT4gdHVwbGVbYm9vbCwgc3RyXToNCiAgICAiIiJEZWNpZGUgd2hldGhlciBkZXByb2plY3RpbmcgYXQgdGhpcyBpbmNsaW5hdGlvbiBpcyB3b3J0aCBkb2luZy4NCg0KICAgIFJldHVybnMgKG9rLCByZWFzb24pLiBUaGUgcmVhc29uIGlzIG1lYW50IHRvIGJlIHByaW50ZWQgYW5kIHRvIGVuZCB1cCBpbiB0aGUNCiAgICBwbG90LCBzbyB0aGF0IGEgc2tpcHBlZCBjb3JyZWN0aW9uIGlzIHZpc2libGUgcmF0aGVyIHRoYW4gc2lsZW50Lg0KICAgICIiIg0KICAgIGlmIGluY19kZWcgPCBJTkNfTUlOX0RFRzoNCiAgICAgICAgcmV0dXJuIEZhbHNlLCAoDQogICAgICAgICAgICBmImluY2xpbmF0aW9uIHtpbmNfZGVnOi4xZn0gZGVnIGlzIGJlbG93IHtJTkNfTUlOX0RFRzouMGZ9IGRlZyAtIHRoZSBnYWxheHkgaXMgIg0KICAgICAgICAgICAgImNsb3NlIHRvIGZhY2Utb24sIHRoZSBjb3JyZWN0aW9uIHdvdWxkIGJlIHNtYWxsZXIgdGhhbiBpdHMgb3duIHVuY2VydGFpbnR5LCAiDQogICAgICAgICAgICAiYW5kIHRoZSBwb3NpdGlvbiBhbmdsZSBpcyBwb29ybHkgZGVmaW5lZC4gTm90IGNvcnJlY3RlZC4iDQogICAgICAgICkNCiAgICBpZiBpbmNfZGVnID4gSU5DX01BWF9ERUc6DQogICAgICAgIHJldHVybiBGYWxzZSwgKA0KICAgICAgICAgICAgZiJpbmNsaW5hdGlvbiB7aW5jX2RlZzouMWZ9IGRlZyBpcyBhYm92ZSB7SU5DX01BWF9ERUc6LjBmfSBkZWcgLSB0aGUgY29ycmVjdGlvbiAiDQogICAgICAgICAgICAid291bGQgYmUgbGFyZ2UgYnV0IHVucmVsaWFibGUsIGFuZCBkdXN0IGJpYXNlcyBpdCBzeXN0ZW1hdGljYWxseS4gTm90IGNvcnJlY3RlZC4iDQogICAgICAgICkNCiAgICByZXR1cm4gVHJ1ZSwgZiJpbmNsaW5hdGlvbiB7aW5jX2RlZzouMWZ9IGRlZyBpcyBpbnNpZGUgdGhlIHVzYWJsZSB3aW5kb3cuIg0KDQpkZWYgc2t5X3BhX3RvX2ltYWdlX2FuZ2xlKHBhX3NreV9kZWc6IGZsb2F0LA0KICAgICAgICAgICAgICAgICAgICAgICAgICBub3J0aF9hbmdsZV9kZWc6IGZsb2F0ID0gOTAuMCwNCiAgICAgICAgICAgICAgICAgICAgICAgICAgbWlycm9yZWQ6IGJvb2wgPSBGYWxzZSkgLT4gZmxvYXQ6DQogICAgIiIiQ29udmVydCBhIGNhdGFsb2d1ZSBwb3NpdGlvbiBhbmdsZSBpbnRvIGFuIGFuZ2xlIGluIGltYWdlIHBpeGVsIGNvb3JkaW5hdGVzLg0KDQogICAgQ2F0YWxvZ3VlcyBnaXZlIHRoZSBwb3NpdGlvbiBhbmdsZSBtZWFzdXJlZCBvbiB0aGUgc2t5LCBmcm9tIE5vcnRoIHRvd2FyZHMgRWFzdC4NCiAgICBPdXIgb2JqZWN0IHBvc2l0aW9ucyBhcmUgaW4gaW1hZ2UgcGl4ZWxzLCB3aGVyZSB0aGUgbWFqb3IgYXhpcyBoYXMgc29tZSBhbmdsZQ0KICAgIG1lYXN1cmVkIGZyb20gdGhlICt4IGF4aXMuIFRoZXNlIGFyZSBub3QgdGhlIHNhbWUgbnVtYmVyIGFuZCBjb25mdXNpbmcgdGhlbQ0KICAgIHJvdGF0ZXMgdGhlIHdob2xlIGRlcHJvamVjdGlvbi4NCg0KICAgICAgICBub3J0aF9hbmdsZV9kZWcgIGFuZ2xlIG9mIE5vcnRoIGluIHRoZSBpbWFnZSwgY291bnRlcmNsb2Nrd2lzZSBmcm9tICt4Lg0KICAgICAgICAgICAgICAgICAgICAgICAgIDkwIGZvciB0aGUgdXN1YWwgIm5vcnRoIHVwIiBvcmllbnRhdGlvbi4NCiAgICAgICAgbWlycm9yZWQgICAgICAgICBUcnVlIGlmIHRoZSBpbWFnZSBwYXJpdHkgaXMgZmxpcHBlZCwgc28gdGhhdCBFYXN0IHJ1bnMNCiAgICAgICAgICAgICAgICAgICAgICAgICBjbG9ja3dpc2UgZnJvbSBOb3J0aCBpbnN0ZWFkIG9mIGNvdW50ZXJjbG9ja3dpc2UuDQoNCiAgICBJZiB5b3UgYWxyZWFkeSBrbm93IHRoZSBtYWpvci1heGlzIGFuZ2xlIGRpcmVjdGx5IGluIGltYWdlIGNvb3JkaW5hdGVzLCBza2lwDQogICAgdGhpcyBmdW5jdGlvbiBhbmQgcGFzcyB0aGF0IGFuZ2xlIHN0cmFpZ2h0IHRvIGRlcHJvamVjdCgpLg0KICAgICIiIg0KICAgIHNpZ24gPSAtMS4wIGlmIG1pcnJvcmVkIGVsc2UgMS4wDQogICAgcmV0dXJuIChub3J0aF9hbmdsZV9kZWcgKyBzaWduICogcGFfc2t5X2RlZykgJSAxODAuMA0KDQpkZWYgZGVwcm9qZWN0KGR4LCBkeSwgbWFqb3JfYXhpc19hbmdsZV9kZWc6IGZsb2F0LCBpbmNfZGVnOiBmbG9hdCk6DQogICAgIiIiU3RyZXRjaCBjZW50cmVkIHBvc2l0aW9ucyBmcm9tIHRoZSBwbGFuZSBvZiB0aGUgc2t5IGludG8gdGhlIGdhbGF4eSBwbGFuZS4NCg0KICAgIGR4LCBkeSBtdXN0IGFscmVhZHkgYmUgbWVhc3VyZWQgRlJPTSBUSEUgR0FMQVhZIENFTlRSRS4gVGhpcyBtYXR0ZXJzOiB0aGUNCiAgICB0cmFuc2Zvcm0gaXMgbGluZWFyLCBzbyBhcHBseWluZyBpdCB0byB1bmNlbnRyZWQgY29vcmRpbmF0ZXMgbGVhdmVzIHRoZSBzaGFwZQ0KICAgIGNvcnJlY3QgYnV0IG1vdmVzIHRoZSBjZW50cmUsIGFuZCBldmVyeSByYWRpdXMgbWVhc3VyZWQgZnJvbSB0aGUgb2xkIGNlbnRyZQ0KICAgIHRoZW4gY29tZXMgb3V0IHdyb25nLg0KDQogICAgbWFqb3JfYXhpc19hbmdsZV9kZWcgaXMgbWVhc3VyZWQgaW4gdGhlIHNhbWUgZnJhbWUgYXMgZHgsIGR5IC0gY291bnRlcmNsb2Nrd2lzZQ0KICAgIGZyb20gdGhlICt4IGF4aXMuIFVzZSBza3lfcGFfdG9faW1hZ2VfYW5nbGUoKSBpZiB5b3UgYXJlIHN0YXJ0aW5nIGZyb20gYQ0KICAgIGNhdGFsb2d1ZSBwb3NpdGlvbiBhbmdsZS4NCg0KICAgIFRoZSB0cmFuc2Zvcm0gcm90YXRlcyB0aGUgbWFqb3IgYXhpcyBvbnRvIHgsIHN0cmV0Y2hlcyB0aGUgcGVycGVuZGljdWxhcg0KICAgIGRpcmVjdGlvbiBieSAxL2NvcyhpKSwgYW5kIHJvdGF0ZXMgYmFjay4gVGhlIHJlc3VsdGluZyBtYXRyaXggaXMgc3ltbWV0cmljLA0KICAgIGFzIGEgcHVyZSBzdHJldGNoIGFsb25nIGFuIGF4aXMgbXVzdCBiZS4NCiAgICAiIiINCiAgICBkeCA9IG5wLmFzYXJyYXkoZHgsIGR0eXBlPWZsb2F0KQ0KICAgIGR5ID0gbnAuYXNhcnJheShkeSwgZHR5cGU9ZmxvYXQpDQoNCiAgICB0aCA9IG1hdGgucmFkaWFucyhtYWpvcl9heGlzX2FuZ2xlX2RlZykNCiAgICBjLCBzID0gbWF0aC5jb3ModGgpLCBtYXRoLnNpbih0aCkNCiAgICBrID0gc3RyZXRjaF9mYWN0b3IoaW5jX2RlZykNCg0KICAgIG0xMSA9IGMgKiBjICsgayAqIHMgKiBzDQogICAgbTEyID0gKDEuMCAtIGspICogcyAqIGMNCiAgICBtMjIgPSBzICogcyArIGsgKiBjICogYw0KDQogICAgcmV0dXJuIGR4ICogbTExICsgZHkgKiBtMTIsIGR4ICogbTEyICsgZHkgKiBtMjINCg0KZGVmIGVmZmVjdGl2ZV9yaW5nX2FyZWFzKGVkZ2VzLA0KICAgICAgICAgICAgICAgICAgICAgICAgIGZyYW1lX2Nvcm5lcnMsDQogICAgICAgICAgICAgICAgICAgICAgICAgY2VudHJlX3h5LA0KICAgICAgICAgICAgICAgICAgICAgICAgIG1ham9yX2F4aXNfYW5nbGVfZGVnOiBmbG9hdCB8IE5vbmUsDQogICAgICAgICAgICAgICAgICAgICAgICAgaW5jX2RlZzogZmxvYXQgfCBOb25lLA0KICAgICAgICAgICAgICAgICAgICAgICAgIG5fc2FtcGxlczogaW50ID0gNDAwXzAwMCwNCiAgICAgICAgICAgICAgICAgICAgICAgICBybmdfc2VlZDogaW50ID0gMTIzNDUpOg0KICAgICIiIkFyZWEgb2YgZWFjaCBhbm51bHVzIHRoYXQgYWN0dWFsbHkgZmFsbHMgaW5zaWRlIHRoZSBpbWFnZWQgZnJhbWUuDQoNCiAgICBXSFkgVEhJUyBFWElTVFMNCiAgICBBIHJpbmcgZHJhd24gYXJvdW5kIHRoZSBnYWxheHkgY2VudHJlIGlzIG9ubHkgcGFydGx5IGNvdmVyZWQgYnkgdGhlIGltYWdlOiBhdA0KICAgIGxhcmdlIHJhZGlpIHRoZSBjb3JuZXJzIG9mIHRoZSBmcmFtZSBjdXQgaXQuIERpdmlkaW5nIGNvdW50cyBieSB0aGUgZnVsbA0KICAgIGdlb21ldHJpYyByaW5nIGFyZWEgdGhlcmVmb3JlIG1ha2VzIHRoZSBvdXRlciBiaW5zIGxvb2sgZW1wdGllciB0aGFuIHRoZXkgYXJlLA0KICAgIGFuZCBwcm9kdWNlcyBhIHJhZGlhbCBkZWNsaW5lIHRoYXQgaXMgcHVyZSBnZW9tZXRyeS4NCg0KICAgIFRoaXMgaXMgbm90IGh5cG90aGV0aWNhbC4gSW4gTTMzIGEgY2F0YWxvZ3VlIGNvdmVyaW5nIG9ubHkgdGhlIGNlbnRyYWwgMjMneDIzJw0KICAgIHByb2R1Y2VkICJhbiBhcnRpZmljaWFsbHkgbGFyZ2UgZGVjcmVhc2UgYXQgcmFkaWkgbGFyZ2VyIHRoYW4gdGhlIGxhcmdlc3QgY2lyY2xlDQogICAgaW5zY3JpYmVkIGluIHRoZSBhcmVhIiwgYW5kIHRoZSBwdWJsaXNoZWQgZXhwb25lbnRpYWwgc2xvcGUgd2FzIHdyb25nIGJ5IGENCiAgICBmYWN0b3Igb2YgdGhyZWUgdW50aWwgYSBzaXplIGN1dCBhbmQgYSByYWRpYWwgdHJ1bmNhdGlvbiB3ZXJlIGFwcGxpZWQuDQoNCiAgICBNRVRIT0QNCiAgICBTY2F0dGVyIHJhbmRvbSBwb2ludHMgdW5pZm9ybWx5IG92ZXIgdGhlIGZyYW1lLCBwdXQgdGhlbSB0aHJvdWdoIGV4YWN0bHkgdGhlDQogICAgc2FtZSBkZXByb2plY3Rpb24gYXMgdGhlIHJlYWwgb2JqZWN0cywgYW5kIGhpc3RvZ3JhbSB0aGVpciByYWRpaS4gVGhlIGZyYWN0aW9uDQogICAgbGFuZGluZyBpbiBlYWNoIHJpbmcsIHRpbWVzIHRoZSBmcmFtZSBhcmVhLCBpcyB0aGF0IHJpbmcncyBlZmZlY3RpdmUgYXJlYS4NCiAgICBUaGlzIGhhbmRsZXMgYW55IGZyYW1lIHNoYXBlIGFuZCBhbnkgZGVwcm9qZWN0aW9uIHdpdGhvdXQgc3BlY2lhbC1jYXNpbmcuDQoNCiAgICBmcmFtZV9jb3JuZXJzIGlzICh4bWluLCB4bWF4LCB5bWluLCB5bWF4KSBpbiB0aGUgc2FtZSBwaXhlbCB1bml0cyBhcyB0aGUgZGF0YS4NCiAgICBQYXNzIG1ham9yX2F4aXNfYW5nbGVfZGVnPU5vbmUgdG8gc2tpcCBkZXByb2plY3Rpb24gKHRoZSB1bmNvcnJlY3RlZCBjYXNlKS4NCiAgICAiIiINCiAgICB4bWluLCB4bWF4LCB5bWluLCB5bWF4ID0gZnJhbWVfY29ybmVycw0KICAgIHJuZyA9IG5wLnJhbmRvbS5kZWZhdWx0X3JuZyhybmdfc2VlZCkNCg0KICAgIHN4ID0gcm5nLnVuaWZvcm0oeG1pbiwgeG1heCwgbl9zYW1wbGVzKQ0KICAgIHN5ID0gcm5nLnVuaWZvcm0oeW1pbiwgeW1heCwgbl9zYW1wbGVzKQ0KICAgIGZyYW1lX2FyZWEgPSAoeG1heCAtIHhtaW4pICogKHltYXggLSB5bWluKQ0KDQogICAgc2R4ID0gc3ggLSBjZW50cmVfeHlbMF0NCiAgICBzZHkgPSBzeSAtIGNlbnRyZV94eVsxXQ0KICAgIGlmIG1ham9yX2F4aXNfYW5nbGVfZGVnIGlzIG5vdCBOb25lIGFuZCBpbmNfZGVnIGlzIG5vdCBOb25lOg0KICAgICAgICBzZHgsIHNkeSA9IGRlcHJvamVjdChzZHgsIHNkeSwgbWFqb3JfYXhpc19hbmdsZV9kZWcsIGluY19kZWcpDQoNCiAgICBzciA9IG5wLmh5cG90KHNkeCwgc2R5KQ0KICAgIGNvdW50cywgXyA9IG5wLmhpc3RvZ3JhbShzciwgYmlucz1lZGdlcykNCiAgICByZXR1cm4gY291bnRzIC8gZmxvYXQobl9zYW1wbGVzKSAqIGZyYW1lX2FyZWENCg0KZGVmIHJhZGlhbF9kZW5zaXR5KGR4LCBkeSwNCiAgICAgICAgICAgICAgICAgICBuX3JpbmdzOiBpbnQgPSAyMCwNCiAgICAgICAgICAgICAgICAgICByX21heDogZmxvYXQgfCBOb25lID0gTm9uZSwNCiAgICAgICAgICAgICAgICAgICBmcmFtZV9jb3JuZXJzPU5vbmUsDQogICAgICAgICAgICAgICAgICAgY2VudHJlX3h5PU5vbmUsDQogICAgICAgICAgICAgICAgICAgbWFqb3JfYXhpc19hbmdsZV9kZWc6IGZsb2F0IHwgTm9uZSA9IE5vbmUsDQogICAgICAgICAgICAgICAgICAgaW5jX2RlZzogZmxvYXQgfCBOb25lID0gTm9uZSwNCiAgICAgICAgICAgICAgICAgICBtaW5fYXJlYV9mcmFjdGlvbjogZmxvYXQgPSAwLjE1KToNCiAgICAiIiJTdXJmYWNlIGRlbnNpdHkgb2Ygb2JqZWN0cyBpbiBlcXVhbC13aWR0aCBjb25jZW50cmljIHJpbmdzLg0KDQogICAgRXF1YWwtd2lkdGggcmluZ3Mgb3V0IHRvIHRoZSBvdXRlcm1vc3Qgb2JqZWN0LCBmb2xsb3dpbmcgdGhlIHN0YW5kYXJkIHByYWN0aWNlDQogICAgaW4gdGhpcyBsaXRlcmF0dXJlICgyMCByaW5ncywgZWFjaCAwLjA1IG9mIHRoZSBtYXhpbXVtIHJhZGl1cykuDQoNCiAgICBSaW5ncyB3aG9zZSBlZmZlY3RpdmUgYXJlYSBpcyBhIHNtYWxsIGZyYWN0aW9uIG9mIHRoZWlyIGZ1bGwgZ2VvbWV0cmljIGFyZWEgYXJlDQogICAgbWFya2VkIGluY29tcGxldGUgcmF0aGVyIHRoYW4gcGxvdHRlZCwgYmVjYXVzZSB0aGVpciBkZW5zaXR5IGlzIGRvbWluYXRlZCBieSBob3cNCiAgICB0aGUgZnJhbWUgaGFwcGVucyB0byBjdXQgdGhlbS4gbWluX2FyZWFfZnJhY3Rpb24gc2V0cyB0aGF0IHRocmVzaG9sZC4NCg0KICAgIFJldHVybnMgYSBkaWN0IHdpdGggcmluZyBjZW50cmVzLCBjb3VudHMsIGFyZWFzLCBkZW5zaXRpZXMgYW5kIFBvaXNzb24gZXJyb3JzLg0KICAgICIiIg0KICAgIGR4ID0gbnAuYXNhcnJheShkeCwgZHR5cGU9ZmxvYXQpDQogICAgZHkgPSBucC5hc2FycmF5KGR5LCBkdHlwZT1mbG9hdCkNCiAgICByID0gbnAuaHlwb3QoZHgsIGR5KQ0KDQogICAgaWYgcl9tYXggaXMgTm9uZToNCiAgICAgICAgcl9tYXggPSBmbG9hdChyLm1heCgpKSBpZiByLnNpemUgZWxzZSAxLjANCiAgICBlZGdlcyA9IG5wLmxpbnNwYWNlKDAuMCwgcl9tYXgsIG5fcmluZ3MgKyAxKQ0KICAgIGNlbnRyZXMgPSAwLjUgKiAoZWRnZXNbOi0xXSArIGVkZ2VzWzE6XSkNCg0KICAgIGNvdW50cywgXyA9IG5wLmhpc3RvZ3JhbShyLCBiaW5zPWVkZ2VzKQ0KICAgIGdlb21ldHJpYyA9IG1hdGgucGkgKiAoZWRnZXNbMTpdICoqIDIgLSBlZGdlc1s6LTFdICoqIDIpDQoNCiAgICBpZiBmcmFtZV9jb3JuZXJzIGlzIG5vdCBOb25lIGFuZCBjZW50cmVfeHkgaXMgbm90IE5vbmU6DQogICAgICAgIGFyZWEgPSBlZmZlY3RpdmVfcmluZ19hcmVhcyhlZGdlcywgZnJhbWVfY29ybmVycywgY2VudHJlX3h5LA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbWFqb3JfYXhpc19hbmdsZV9kZWcsIGluY19kZWcpDQogICAgZWxzZToNCiAgICAgICAgYXJlYSA9IGdlb21ldHJpYw0KDQogICAgd2l0aCBucC5lcnJzdGF0ZShkaXZpZGU9Imlnbm9yZSIsIGludmFsaWQ9Imlnbm9yZSIpOg0KICAgICAgICBkZW5zaXR5ID0gbnAud2hlcmUoYXJlYSA+IDAsIGNvdW50cyAvIGFyZWEsIG5wLm5hbikNCiAgICAgICAgZXJyID0gbnAud2hlcmUoYXJlYSA+IDAsIG5wLnNxcnQobnAubWF4aW11bShjb3VudHMsIDApKSAvIGFyZWEsIG5wLm5hbikNCiAgICAgICAgZnJhYyA9IG5wLndoZXJlKGdlb21ldHJpYyA+IDAsIGFyZWEgLyBnZW9tZXRyaWMsIDAuMCkNCg0KICAgIGNvbXBsZXRlID0gZnJhYyA+PSBtaW5fYXJlYV9mcmFjdGlvbg0KDQogICAgcmV0dXJuIHsNCiAgICAgICAgImVkZ2VzIjogZWRnZXMsDQogICAgICAgICJyIjogY2VudHJlcywNCiAgICAgICAgImNvdW50cyI6IGNvdW50cywNCiAgICAgICAgImFyZWEiOiBhcmVhLA0KICAgICAgICAiYXJlYV9mcmFjdGlvbiI6IGZyYWMsDQogICAgICAgICJkZW5zaXR5IjogZGVuc2l0eSwNCiAgICAgICAgImRlbnNpdHlfZXJyIjogZXJyLA0KICAgICAgICAiY29tcGxldGUiOiBjb21wbGV0ZSwNCiAgICAgICAgInJfbWF4Ijogcl9tYXgsDQogICAgfQ0KDQpkZWYgc2VjdG9yX2Rpc3BlcnNpb24oZHgsIGR5LCBtYWpvcl9heGlzX2FuZ2xlX2RlZywgaW5jX2RlZywgbl9zZWN0b3JzPTIwKToNCiAgICAiIiJIb3cgdW5ldmVuIHRoZSBvYmplY3QgY291bnRzIGFyZSBvbmNlIHRoZSBmaWVsZCBpcyBjdXQgaW50byBlcXVhbCB3ZWRnZXMuDQoNCiAgICBUSEUgSURFQQ0KICAgIFNsaWNlIHRoZSBnYWxheHkgaW50byBlcXVhbCBzZWN0b3JzLCBsaWtlIHNsaWNlcyBvZiBhIHBpZS4gSWYgdGhlIGRpc2sgcmVhbGx5IGlzDQogICAgcm91bmQgYW5kIHdlIGFyZSBsb29raW5nIHN0cmFpZ2h0IGRvd24gb24gaXQsIGV2ZXJ5IHNsaWNlIGhvbGRzIHJvdWdobHkgdGhlIHNhbWUNCiAgICBudW1iZXIgb2Ygb2JqZWN0cy4gU2VlbiBhdCBhbiBhbmdsZSBpdCBkb2VzIG5vdC4gU28gdGhlIHRpbHQgdGhhdCBldmVucyB0aGUNCiAgICBzbGljZXMgb3V0IGlzIHRoZSB0aWx0IHRoYXQgdW5kb2VzIHRoZSBwcm9qZWN0aW9uLg0KDQogICAgT3Bwb3NpdGUgc2xpY2VzIGFyZSBhZGRlZCB0b2dldGhlciBiZWZvcmUgdGhlIGRpc3BlcnNpb24gaXMgbWVhc3VyZWQuIEEgZ2FsYXh5DQogICAgdGhhdCBpcyBsb3BzaWRlZCAtIGEgYnJpZ2h0ZXIgYXJtIG9uIG9uZSBzaWRlLCBhIGNvbXBhbmlvbiBwdWxsaW5nIGF0IGl0IC0gd291bGQNCiAgICBvdGhlcndpc2UgcHVzaCB0aGUgYW5zd2VyIGFyb3VuZDsgYWRkaW5nIGEgc2xpY2UgdG8gdGhlIG9uZSBmYWNpbmcgaXQgY2FuY2Vscw0KICAgIHRoYXQga2luZCBvZiBhc3ltbWV0cnkuDQoNCiAgICBSZXR1cm5zIHN0ZCAvIG1lYW4gb2YgdGhlIGZvbGRlZCBjb3VudHMuIFNtYWxsZXIgaXMgYmV0dGVyLg0KICAgICIiIg0KICAgIGR4cCwgZHlwID0gZGVwcm9qZWN0KGR4LCBkeSwgbWFqb3JfYXhpc19hbmdsZV9kZWcsIGluY19kZWcpDQogICAgdGhldGEgPSBucC5hcmN0YW4yKGR5cCwgZHhwKQ0KICAgIGlkeCA9IG5wLmZsb29yKCh0aGV0YSAlICgyLjAgKiBtYXRoLnBpKSkgLyAoMi4wICogbWF0aC5waSkgKiBuX3NlY3RvcnMpLmFzdHlwZShpbnQpDQogICAgaWR4ID0gbnAuY2xpcChpZHgsIDAsIG5fc2VjdG9ycyAtIDEpDQogICAgY291bnRzID0gbnAuYmluY291bnQoaWR4LCBtaW5sZW5ndGg9bl9zZWN0b3JzKS5hc3R5cGUoZmxvYXQpDQoNCiAgICBoYWxmID0gbl9zZWN0b3JzIC8vIDINCiAgICBmb2xkZWQgPSBjb3VudHNbOmhhbGZdICsgY291bnRzW2hhbGY6XQ0KICAgIG1lYW4gPSBmb2xkZWQubWVhbigpDQogICAgaWYgbWVhbiA8PSAwOg0KICAgICAgICByZXR1cm4gbnAuaW5mDQogICAgcmV0dXJuIGZsb2F0KGZvbGRlZC5zdGQoKSAvIG1lYW4pDQoNCmRlZiBmaXRfc2VjdG9yX21ldGhvZChkeCwgZHksDQogICAgICAgICAgICAgICAgICAgICAgbl9zZWN0b3JzPTIwLA0KICAgICAgICAgICAgICAgICAgICAgIGFuZ2xlX3N0ZXA9NS4wLA0KICAgICAgICAgICAgICAgICAgICAgIGluY19zdGVwPTUuMCwNCiAgICAgICAgICAgICAgICAgICAgICBpbmNfcmFuZ2U9KDIwLjAsIDgwLjApLA0KICAgICAgICAgICAgICAgICAgICAgIHJfbWF4PU5vbmUpOg0KICAgICIiIlNlYXJjaCBmb3IgdGhlIGRlcHJvamVjdGlvbiBhbmdsZXMgdGhhdCBtYWtlIHRoZSBvYmplY3QgY291bnRzIG1vc3QgZXZlbi4NCg0KICAgIFNjYW5zIGEgZ3JpZCBvZiBwb3NpdGlvbiBhbmdsZSBhbmQgaW5jbGluYXRpb24gYW5kIHJldHVybnMgdGhlIHBhaXIgdGhhdA0KICAgIG1pbmltaXNlcyBzZWN0b3JfZGlzcGVyc2lvbigpLiBUaGlzIGlzIHRoZSBzZWNvbmQgb2YgdGhlIHR3byBtZXRob2RzIGludHJvZHVjZWQNCiAgICBieSBHYXJjaWEtR29tZXogYW5kIEF0aGFuYXNzb3VsYSwgYW5kIGluIHRoZWlyIGNvbXBhcmlzb24gaXQgaXMgdGhlIG9uZSB0aGF0DQogICAgYWdyZWVzIG1vc3QgY2xvc2VseSB3aXRoIGtpbmVtYXRpYyBkZXRlcm1pbmF0aW9ucyAtIDAuOTYgaW4gcG9zaXRpb24gYW5nbGUgYW5kDQogICAgMC45MCBpbiBjb3MoaW5jbGluYXRpb24pLCBiZXR0ZXIgdGhhbiB0aGVpciBGb3VyaWVyIG1ldGhvZCBhbmQgYmV0dGVyIHRoYW4NCiAgICBwaG90b21ldHJ5Lg0KDQogICAgV0hFTiBJVCBGQUlMUyAtIHJlYWQgdGhpcyBiZWZvcmUgdHJ1c3RpbmcgdGhlIGFuc3dlcg0KICAgICAgKiBCZWxvdyByb3VnaGx5IDE1MC0yMDAgb2JqZWN0cyB0aGUgc3VyZmFjZSBiZWluZyBtaW5pbWlzZWQgb2Z0ZW4gaGFzIHNldmVyYWwNCiAgICAgICAgbWluaW1hIHJhdGhlciB0aGFuIG9uZSwgYW5kIHRoZSBkZWVwZXN0IGlzIG5vdCBhbHdheXMgdGhlIHJpZ2h0IG9uZS4gVGhlDQogICAgICAgIHJldHVybmVkIGRpY3QgcmVwb3J0cyB0aGUgcnVubmVycy11cCBzbyB5b3UgY2FuIGxvb2suDQogICAgICAqIElmIHRoZSBvYmplY3RzIHRyYWNlIHRoZSBzcGlyYWwgYXJtcyBvciBhIGJhciBpbnN0ZWFkIG9mIGZpbGxpbmcgdGhlIGRpc2ssDQogICAgICAgIHRoZSBtZXRob2QgY2lyY3VsYXJpc2VzIHRoZSBhcm1zIG9yIHRoZSBiYXIgcmF0aGVyIHRoYW4gdGhlIGRpc2suIEluIHRoZQ0KICAgICAgICBwdWJsaXNoZWQgbm90ZXMgdGhpcyBpcyBleGFjdGx5IHdoYXQgd2VudCB3cm9uZyBmb3Igc2V2ZXJhbCBiYXJyZWQgZ2FsYXhpZXMuDQogICAgICAqIEEgcmluZyBvZiBvYmplY3RzIGlzIGNpcmN1bGFyaXNlZCBhcyBpZiBpdCB3ZXJlIHRoZSBkaXNrLiBBbiBvdXRlciByaW5nIHdpdGgNCiAgICAgICAgYW4gaW50cmluc2ljIGF4aXMgcmF0aW8gb2YgMC44NyBmYWtlcyBhbiBpbmNsaW5hdGlvbiBvZiBhYm91dCAxMiBkZWdyZWVzIGluIGENCiAgICAgICAgZ2FsYXh5IHRoYXQgaXMgcmVhbGx5IGZhY2Utb24uDQogICAgICAqIEl0IHdvcmtzIGJldHRlciB0aGUgbW9yZSBpbmNsaW5lZCB0aGUgZ2FsYXh5IGlzLCBhbmQgcG9vcmx5IG5lYXIgZmFjZS1vbiwNCiAgICAgICAgd2hlcmUgdGhlIHBvc2l0aW9uIGFuZ2xlIGlzIGJhcmVseSBkZWZpbmVkIGluIHRoZSBmaXJzdCBwbGFjZS4NCg0KICAgIHJfbWF4IG9wdGlvbmFsbHkgcmVzdHJpY3RzIHRoZSBmaXQgdG8gb2JqZWN0cyBpbnNpZGUgdGhhdCByYWRpdXMsIHdoaWNoIGhlbHBzDQogICAgd2hlbiBhIGZldyBmYXItZmx1bmcgb2JqZWN0cyBkb21pbmF0ZSB0aGUgb3V0ZXIgc2VjdG9ycy4NCiAgICAiIiINCiAgICBkeCA9IG5wLmFzYXJyYXkoZHgsIGR0eXBlPWZsb2F0KQ0KICAgIGR5ID0gbnAuYXNhcnJheShkeSwgZHR5cGU9ZmxvYXQpDQoNCiAgICBpZiByX21heCBpcyBub3QgTm9uZToNCiAgICAgICAga2VlcCA9IG5wLmh5cG90KGR4LCBkeSkgPD0gcl9tYXgNCiAgICAgICAgZHgsIGR5ID0gZHhba2VlcF0sIGR5W2tlZXBdDQoNCiAgICBhbmdsZXMgPSBucC5hcmFuZ2UoMC4wLCAxODAuMCwgYW5nbGVfc3RlcCkNCiAgICBpbmNzID0gbnAuYXJhbmdlKGluY19yYW5nZVswXSwgaW5jX3JhbmdlWzFdICsgMWUtOSwgaW5jX3N0ZXApDQoNCiAgICBncmlkID0gbnAuZnVsbCgobGVuKGluY3MpLCBsZW4oYW5nbGVzKSksIG5wLmluZikNCiAgICBmb3IgaSwgaW5jIGluIGVudW1lcmF0ZShpbmNzKToNCiAgICAgICAgZm9yIGosIGFuZyBpbiBlbnVtZXJhdGUoYW5nbGVzKToNCiAgICAgICAgICAgIGdyaWRbaSwgal0gPSBzZWN0b3JfZGlzcGVyc2lvbihkeCwgZHksIGZsb2F0KGFuZyksIGZsb2F0KGluYyksIG5fc2VjdG9ycykNCg0KICAgIGZsYXQgPSBucC5hcmdzb3J0KGdyaWQsIGF4aXM9Tm9uZSkNCiAgICBiZXN0X2ksIGJlc3RfaiA9IG5wLnVucmF2ZWxfaW5kZXgoZmxhdFswXSwgZ3JpZC5zaGFwZSkNCg0KICAgICMgUmVwb3J0IGRpc3RpbmN0IHJ1bm5lcnMtdXA6IGxvY2FsIG1pbmltYSB3ZWxsIHNlcGFyYXRlZCBmcm9tIHRoZSBiZXN0IG9uZSwgc28NCiAgICAjIHRoYXQgYSBzaGFsbG93IGdsb2JhbCBtaW5pbXVtIGRvZXMgbm90IGhpZGUgYSBjb21wZXRpbmcgc29sdXRpb24uDQogICAgcnVubmVycyA9IFtdDQogICAgZm9yIGsgaW4gZmxhdFsxOl06DQogICAgICAgIGksIGogPSBucC51bnJhdmVsX2luZGV4KGssIGdyaWQuc2hhcGUpDQogICAgICAgIHNlcF9hbmcgPSBtaW4oYWJzKGFuZ2xlc1tqXSAtIGFuZ2xlc1tiZXN0X2pdKSwNCiAgICAgICAgICAgICAgICAgICAgICAxODAuMCAtIGFicyhhbmdsZXNbal0gLSBhbmdsZXNbYmVzdF9qXSkpDQogICAgICAgIGlmIHNlcF9hbmcgPCAyMC4wIGFuZCBhYnMoaW5jc1tpXSAtIGluY3NbYmVzdF9pXSkgPCAxNS4wOg0KICAgICAgICAgICAgY29udGludWUNCiAgICAgICAgaWYgYW55KG1pbihhYnMoYW5nbGVzW2pdIC0gYSksIDE4MC4wIC0gYWJzKGFuZ2xlc1tqXSAtIGEpKSA8IDIwLjANCiAgICAgICAgICAgICAgIGFuZCBhYnMoaW5jc1tpXSAtIGMpIDwgMTUuMCBmb3IgYSwgYywgXyBpbiBydW5uZXJzKToNCiAgICAgICAgICAgIGNvbnRpbnVlDQogICAgICAgIHJ1bm5lcnMuYXBwZW5kKChmbG9hdChhbmdsZXNbal0pLCBmbG9hdChpbmNzW2ldKSwgZmxvYXQoZ3JpZFtpLCBqXSkpKQ0KICAgICAgICBpZiBsZW4ocnVubmVycykgPj0gMzoNCiAgICAgICAgICAgIGJyZWFrDQoNCiAgICByZXR1cm4gew0KICAgICAgICAiYW5nbGUiOiBmbG9hdChhbmdsZXNbYmVzdF9qXSksDQogICAgICAgICJpbmMiOiBmbG9hdChpbmNzW2Jlc3RfaV0pLA0KICAgICAgICAiZGlzcGVyc2lvbiI6IGZsb2F0KGdyaWRbYmVzdF9pLCBiZXN0X2pdKSwNCiAgICAgICAgIm5fb2JqZWN0cyI6IGludChkeC5zaXplKSwNCiAgICAgICAgIm5fc2VjdG9ycyI6IG5fc2VjdG9ycywNCiAgICAgICAgImFuZ2xlcyI6IGFuZ2xlcywNCiAgICAgICAgImluY3MiOiBpbmNzLA0KICAgICAgICAiZ3JpZCI6IGdyaWQsDQogICAgICAgICJydW5uZXJzX3VwIjogcnVubmVycywNCiAgICAgICAgImZld19vYmplY3RzIjogYm9vbChkeC5zaXplIDwgMjAwKSwNCiAgICB9DQoNCmRlZiBkaXN0cmlidXRpb25fc2hhcGUoZHgsIGR5KToNCiAgICAiIiJUaGUgYXhpcyByYXRpbyBhbmQgcG9zaXRpb24gYW5nbGUgb2YgYSBzZXQgb2YgcG9zaXRpb25zLg0KDQogICAgRnJvbSB0aGUgc2Vjb25kIG1vbWVudHM6IHRoZSBzaGFwZSBvZiB0aGUgY2xvdWQgb2YgcG9pbnRzLCBub3Qgb2YgYW55IG9uZSBvZg0KICAgIHRoZW0uIFJldHVybnMgKGF4aXNfcmF0aW8sIGFuZ2xlX2RlZyksIHRoZSBhbmdsZSBtZWFzdXJlZCBmcm9tICt4IHRoZSBzYW1lIHdheQ0KICAgIHRoZSBkZXByb2plY3Rpb24gbWVhc3VyZXMgaXQuDQogICAgIiIiDQogICAgZHggPSBucC5hc2FycmF5KGR4LCBkdHlwZT1mbG9hdCkNCiAgICBkeSA9IG5wLmFzYXJyYXkoZHksIGR0eXBlPWZsb2F0KQ0KICAgIG9rID0gbnAuaXNmaW5pdGUoZHgpICYgbnAuaXNmaW5pdGUoZHkpDQogICAgaWYgb2suc3VtKCkgPCAyMDoNCiAgICAgICAgcmV0dXJuIGZsb2F0KCJuYW4iKSwgZmxvYXQoIm5hbiIpDQogICAgY292ID0gbnAuY292KGR4W29rXSwgZHlbb2tdKQ0KICAgIHZhbHVlcywgdmVjdG9ycyA9IG5wLmxpbmFsZy5laWdoKGNvdikNCiAgICBpZiB2YWx1ZXNbMV0gPD0gMDoNCiAgICAgICAgcmV0dXJuIGZsb2F0KCJuYW4iKSwgZmxvYXQoIm5hbiIpDQogICAgcmF0aW8gPSBmbG9hdChucC5zcXJ0KG1heCh2YWx1ZXNbMF0sIDAuMCkgLyB2YWx1ZXNbMV0pKQ0KICAgIGFuZ2xlID0gZmxvYXQobnAuZGVncmVlcyhucC5hcmN0YW4yKHZlY3RvcnNbMSwgMV0sIHZlY3RvcnNbMCwgMV0pKSAlIDE4MC4wKQ0KICAgIHJldHVybiByYXRpbywgYW5nbGUNCg0KZGVmIGNoZWNrX2FnYWluc3RfY2F0YWxvZ3VlKGR4LCBkeSwgZXhwZWN0ZWRfcmF0aW8sIGV4cGVjdGVkX2FuZ2xlLA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJhdGlvX3RvbGVyYW5jZT0wLjE1LCBhbmdsZV90b2xlcmFuY2U9MjUuMCk6DQogICAgIiIiQ29tcGFyZSB0aGUgc2hhcGUgb2Ygd2hhdCB3YXMgZGV0ZWN0ZWQgd2l0aCB3aGF0IHRoZSBjYXRhbG9ndWUgcHJlZGljdHMuDQoNCiAgICBXSFkgVEhJUyBJUyBXT1JUSCBET0lORw0KICAgIEEgdGlsdGVkIGRpc2sgc2VlbiBmcm9tIGhlcmUgaXMgYW4gZWxsaXBzZSwgYW5kIHRoZSBjYXRhbG9ndWUgc2F5cyB3aGljaA0KICAgIGVsbGlwc2UuIFNvIHRoZSB0cmFjZXJzIGRldGVjdGVkIGluIGl0IGhhdmUgYSBzaGFwZSB0aGF0IGNhbiBiZSBwcmVkaWN0ZWQNCiAgICBiZWZvcmUgdGhleSBhcmUgbWVhc3VyZWQgLSBhbmQgY29tcGFyaW5nIHRoZSB0d28gY29zdHMgbm90aGluZyBhbmQgbmVlZHMgbm8NCiAgICBleHRyYSBkYXRhLg0KDQogICAgSXQgaXMgYSByZWFsIHRlc3QgYmVjYXVzZSBpdCBjYW4gZmFpbC4gT24gYSBzZXQgb2YgTTMzIGZyYW1lcyB0aGUgdG9vbA0KICAgIGRldGVjdGVkIDI4NDQgcmVnaW9ucyBhbmQgZHJldyBhIHNtb290aCwgZW50aXJlbHkgcGxhdXNpYmxlIHJhZGlhbCBwcm9maWxlOw0KICAgIHRoZSBkZXRlY3Rpb25zIGNhbWUgb3V0IHJvdW5kLCBheGlzIHJhdGlvIDAuOTYsIHdoaWxlIHRoZSBjYXRhbG9ndWUgc2F5cyAwLjU5Lg0KICAgIFRoZXkgd2VyZSBub3QgdHJhY2luZyB0aGUgZ2FsYXh5IGF0IGFsbCAtIHRoZXkgd2VyZSBzdWJ0cmFjdGlvbiByZXNpZHVhbHMNCiAgICBzcHJlYWQgZXZlbmx5IG92ZXIgdGhlIGZpZWxkIC0gYW5kIG5vdGhpbmcgZWxzZSBpbiB0aGUgb3V0cHV0IHNob3dlZCBpdC4NCg0KICAgIFdoZW4gdGhlIHR3byBhZ3JlZSBpdCBpcyBldmlkZW5jZSB0aGUgZGV0ZWN0aW9uIGlzIGZpbmRpbmcgdGhlIGdhbGF4eS4gV2hlbg0KICAgIHRoZXkgZGlzYWdyZWUsIHNvbWV0aGluZyBpcyB3cm9uZyB1cHN0cmVhbSBhbmQgdGhlIHByb2ZpbGUgYmVsb3cgc2hvdWxkIG5vdCBiZQ0KICAgIGJlbGlldmVkLiBUaGUgY2hlY2sgZG9lcyBub3Qgc2F5IHdoaWNoIG9mIHRoZSB0d28gaXMgYXQgZmF1bHQuDQoNCiAgICBBIGdhbGF4eSB0b28gcm91bmQgZm9yIGl0cyBvd24gcG9zaXRpb24gYW5nbGUgdG8gYmUgZGVmaW5lZCBpcyBub3QgdGVzdGVkIG9uDQogICAgYW5nbGUsIGZvciB0aGUgc2FtZSByZWFzb24gdGhlIGNhdGFsb2d1ZSBkZWNsaW5lcyB0byBwdWJsaXNoIG9uZS4NCiAgICAiIiINCiAgICByYXRpbywgYW5nbGUgPSBkaXN0cmlidXRpb25fc2hhcGUoZHgsIGR5KQ0KICAgIGlmIG5vdCBucC5pc2Zpbml0ZShyYXRpbyk6DQogICAgICAgIHJldHVybiB7Im9rIjogTm9uZSwgInJlYXNvbiI6ICJ0b28gZmV3IHBvc2l0aW9ucyB0byBtZWFzdXJlIGEgc2hhcGUifQ0KDQogICAgb3V0ID0geyJtZWFzdXJlZF9yYXRpbyI6IHJhdGlvLCAibWVhc3VyZWRfYW5nbGUiOiBhbmdsZSwNCiAgICAgICAgICAgImV4cGVjdGVkX3JhdGlvIjogZXhwZWN0ZWRfcmF0aW8sICJleHBlY3RlZF9hbmdsZSI6IGV4cGVjdGVkX2FuZ2xlfQ0KDQogICAgcHJvYmxlbXMgPSBbXQ0KICAgIGlmIGV4cGVjdGVkX3JhdGlvIGlzIG5vdCBOb25lIGFuZCBucC5pc2Zpbml0ZShleHBlY3RlZF9yYXRpbyk6DQogICAgICAgIGlmIGFicyhyYXRpbyAtIGV4cGVjdGVkX3JhdGlvKSA+IHJhdGlvX3RvbGVyYW5jZToNCiAgICAgICAgICAgIHByb2JsZW1zLmFwcGVuZCgNCiAgICAgICAgICAgICAgICAidGhlIGRldGVjdGlvbnMgYXJlIHNoYXBlZCB7Oi4yZn0gd2hlcmUgdGhlIGNhdGFsb2d1ZSBzYXlzIHs6LjJmfSINCiAgICAgICAgICAgICAgICAuZm9ybWF0KHJhdGlvLCBleHBlY3RlZF9yYXRpbykpDQogICAgIyBBIHJvdW5kIGdhbGF4eSBoYXMgbm8gbWVhbmluZ2Z1bCBtYWpvciBheGlzLCBzbyBpdHMgYW5nbGUgaXMgbm90IHRlc3RlZC4NCiAgICBpZiAoZXhwZWN0ZWRfYW5nbGUgaXMgbm90IE5vbmUgYW5kIG5wLmlzZmluaXRlKGV4cGVjdGVkX2FuZ2xlKQ0KICAgICAgICAgICAgYW5kIGV4cGVjdGVkX3JhdGlvIGlzIG5vdCBOb25lIGFuZCBleHBlY3RlZF9yYXRpbyA8IDAuODUpOg0KICAgICAgICBnYXAgPSBhYnMoKGFuZ2xlIC0gZXhwZWN0ZWRfYW5nbGUgKyA5MC4wKSAlIDE4MC4wIC0gOTAuMCkNCiAgICAgICAgb3V0WyJhbmdsZV9nYXAiXSA9IGdhcA0KICAgICAgICBpZiBnYXAgPiBhbmdsZV90b2xlcmFuY2U6DQogICAgICAgICAgICBwcm9ibGVtcy5hcHBlbmQoDQogICAgICAgICAgICAgICAgInRoZWlyIGxvbmcgYXhpcyBsaWVzIGF0IHs6LjBmfSBkZWcgd2hlcmUgdGhlIGNhdGFsb2d1ZSBzYXlzIHs6LjBmfSINCiAgICAgICAgICAgICAgICAuZm9ybWF0KGFuZ2xlLCBleHBlY3RlZF9hbmdsZSkpDQoNCiAgICBvdXRbIm9rIl0gPSBub3QgcHJvYmxlbXMNCiAgICBvdXRbInByb2JsZW1zIl0gPSBwcm9ibGVtcw0KICAgIHJldHVybiBvdXQNCg0KZGVmIGRlc2NyaWJlX2NoZWNrKHJlc3VsdCk6DQogICAgIiIiVGhlIHNlbGYtY2hlY2ssIGluIGEgZm9ybSB3b3J0aCBwcmludGluZy4iIiINCiAgICBpZiByZXN1bHQuZ2V0KCJvayIpIGlzIE5vbmU6DQogICAgICAgIHJldHVybiAiICBzaGFwZSBjaGVjazogIiArIHJlc3VsdC5nZXQoInJlYXNvbiIsICJub3QgZG9uZSIpDQogICAgaGVhZCA9ICgiICBzaGFwZSBjaGVjazogdGhlIGRldGVjdGlvbnMgYXJlIHNoYXBlZCB7Oi4yZn0gd2l0aCB0aGVpciBsb25nIGF4aXMgIg0KICAgICAgICAgICAgImF0IHs6LjBmfSBkZWciLmZvcm1hdChyZXN1bHRbIm1lYXN1cmVkX3JhdGlvIl0sIHJlc3VsdFsibWVhc3VyZWRfYW5nbGUiXSkpDQogICAgaWYgcmVzdWx0LmdldCgiZXhwZWN0ZWRfcmF0aW8iKSBpcyBub3QgTm9uZToNCiAgICAgICAgaGVhZCArPSAiOyB0aGUgY2F0YWxvZ3VlIHByZWRpY3RzIHs6LjJmfSIuZm9ybWF0KHJlc3VsdFsiZXhwZWN0ZWRfcmF0aW8iXSkNCiAgICAgICAgaWYgcmVzdWx0LmdldCgiZXhwZWN0ZWRfYW5nbGUiKSBpcyBub3QgTm9uZToNCiAgICAgICAgICAgIGhlYWQgKz0gIiBhdCB7Oi4wZn0gZGVnIi5mb3JtYXQocmVzdWx0WyJleHBlY3RlZF9hbmdsZSJdKQ0KICAgIGlmIHJlc3VsdFsib2siXToNCiAgICAgICAgcmV0dXJuIGhlYWQgKyAiXG4gICAgdGhleSBhZ3JlZSAtIHRoZSBkZXRlY3Rpb25zIGRvIHRyYWNlIHRoaXMgZ2FsYXh5Ig0KICAgIHJldHVybiAoaGVhZCArICJcbiAgICBUSEVZIERJU0FHUkVFOiAiICsgIjsgIi5qb2luKHJlc3VsdFsicHJvYmxlbXMiXSkgKw0KICAgICAgICAgICAgIlxuICAgIFNvbWV0aGluZyB1cHN0cmVhbSBpcyB3cm9uZy4gRGV0ZWN0aW9ucyB0aGF0IGRvIG5vdCBmb2xsb3cgdGhlICINCiAgICAgICAgICAgICJnYWxheHknc1xuICAgIHNoYXBlIGFyZSB1c3VhbGx5IHN1YnRyYWN0aW9uIHJlc2lkdWFscyBzcHJlYWQgb3ZlciB0aGUgIg0KICAgICAgICAgICAgIndob2xlIGZpZWxkLCBhbmRcbiAgICB0aGUgcmFkaWFsIHByb2ZpbGUgYmVsb3cgd2lsbCBsb29rIHNtb290aCBhbmQgIg0KICAgICAgICAgICAgIm1lYW4gbm90aGluZy4iKQ0KDQoNCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0NCiMgc2hhcmVkIG1vZHVsZSBnYWxheHlfY2F0YWxvZ3VlLnB5DQojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tDQoNCnBhc3MgICMgZnJvbSBkZXByb2plY3Rpb24sIGlubGluZWQgYWJvdmUNCg0KSFlQRVJMRURBX1ZJWklFUl9JRCA9ICJWSUkvMjM3Ig0KDQpkZWYgaHlwZXJsZWRhX2dlb21ldHJ5KGdhbGF4eV9uYW1lOiBzdHIsIHNlYXJjaF9yYWRpdXNfYXJjbWluOiBmbG9hdCA9IDAuNSkgLT4gZGljdDoNCiAgICAiIiJGZXRjaCBwb3NpdGlvbiBhbmdsZSwgYXhpcyByYXRpbyBhbmQgaW5jbGluYXRpb24gZm9yIG9uZSBnYWxheHkuDQoNCiAgICBSZXR1cm5zIGEgZGljdCB3aXRoOg0KICAgICAgICBuYW1lX3F1ZXJ5LCBwZ2MsIG10eXBlICAtIGlkZW50aWZpY2F0aW9uDQogICAgICAgIGxvZ19yMjUsIGJfb3Zlcl9hICAgICAgIC0gdGhlIG9ic2VydmVkIGF4aXMgcmF0aW8NCiAgICAgICAgcGFfc2t5ICAgICAgICAgICAgICAgICAgLSBwb3NpdGlvbiBhbmdsZSBvbiB0aGUgc2t5LCBvciBOb25lIGlmIG5vdCBwdWJsaXNoZWQNCiAgICAgICAgaW5jX25haXZlICAgICAgICAgICAgICAgLSBhY29zKGIvYSksIGlnbm9yaW5nIGRpc2sgdGhpY2tuZXNzDQogICAgICAgIGluYyAgICAgICAgICAgICAgICAgICAgIC0gdGhlIEh1YmJsZS1jb3JyZWN0ZWQgaW5jbGluYXRpb24gd2UgYWN0dWFsbHkgdXNlDQogICAgICAgIHEwICAgICAgICAgICAgICAgICAgICAgIC0gdGhlIHRoaWNrbmVzcyBjb25zdGFudCBjaG9zZW4gZm9yIHRoaXMgdHlwZQ0KICAgICAgICBkMjVfYXJjbWluICAgICAgICAgICAgICAtIGlzb3Bob3RhbCBkaWFtZXRlciwgdXNlZnVsIGZvciBub3JtYWxpc2luZyByYWRpaQ0KICAgICAgICBzb3VyY2UgICAgICAgICAgICAgICAgICAtIGh1bWFuLXJlYWRhYmxlIHByb3ZlbmFuY2UNCg0KICAgIFJhaXNlcyBMb29rdXBFcnJvciB3aGVuIHRoZSBuYW1lIHJlc29sdmVzIHRvIG5vdGhpbmcsIHNvIHRoZSBjYWxsZXIgY2FuIGZhbGwNCiAgICBiYWNrIHRvIG1hbnVhbGx5IHN1cHBsaWVkIGFuZ2xlcyBpbnN0ZWFkIG9mIHNpbGVudGx5IGNvbnRpbnVpbmcuDQogICAgIiIiDQogICAgZnJvbSBhc3Ryb3F1ZXJ5LnZpemllciBpbXBvcnQgVml6aWVyDQogICAgaW1wb3J0IGFzdHJvcHkudW5pdHMgYXMgdQ0KDQogICAgdiA9IFZpemllcihjYXRhbG9nPUhZUEVSTEVEQV9WSVpJRVJfSUQsIHJvd19saW1pdD0tMSkNCiAgICByZXN1bHQgPSB2LnF1ZXJ5X29iamVjdChnYWxheHlfbmFtZSwgcmFkaXVzPXNlYXJjaF9yYWRpdXNfYXJjbWluICogdS5hcmNtaW4pDQogICAgaWYgbm90IHJlc3VsdCBvciBsZW4ocmVzdWx0WzBdKSA9PSAwOg0KICAgICAgICByYWlzZSBMb29rdXBFcnJvcigNCiAgICAgICAgICAgIGYiSHlwZXJMRURBIHJldHVybmVkIG5vIHJvdyBmb3IgJ3tnYWxheHlfbmFtZX0nLiBDaGVjayB0aGUgbmFtZSwgb3IgIg0KICAgICAgICAgICAgInN1cHBseSB0aGUgYW5nbGVzIG1hbnVhbGx5LiINCiAgICAgICAgKQ0KDQogICAgcm93ID0gcmVzdWx0WzBdWzBdDQoNCiAgICBkZWYgX2dldChjb2wpOg0KICAgICAgICB0cnk6DQogICAgICAgICAgICB2YWwgPSByb3dbY29sXQ0KICAgICAgICBleGNlcHQgS2V5RXJyb3I6DQogICAgICAgICAgICByZXR1cm4gTm9uZQ0KICAgICAgICB0cnk6DQogICAgICAgICAgICBpZiBoYXNhdHRyKHZhbCwgIm1hc2siKSBhbmQgdmFsLm1hc2s6DQogICAgICAgICAgICAgICAgcmV0dXJuIE5vbmUNCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoNCiAgICAgICAgICAgIHBhc3MNCiAgICAgICAgdHJ5Og0KICAgICAgICAgICAgZiA9IGZsb2F0KHZhbCkNCiAgICAgICAgZXhjZXB0IChUeXBlRXJyb3IsIFZhbHVlRXJyb3IpOg0KICAgICAgICAgICAgcmV0dXJuIE5vbmUNCiAgICAgICAgcmV0dXJuIE5vbmUgaWYgbWF0aC5pc25hbihmKSBlbHNlIGYNCg0KICAgIGxvZ19yMjUgPSBfZ2V0KCJsb2dSMjUiKQ0KICAgIGlmIGxvZ19yMjUgaXMgTm9uZToNCiAgICAgICAgcmFpc2UgTG9va3VwRXJyb3IoDQogICAgICAgICAgICBmIkh5cGVyTEVEQSBoYXMgYSByb3cgZm9yICd7Z2FsYXh5X25hbWV9JyBidXQgbm8gYXhpcyByYXRpbyAobG9nUjI1KSwgIg0KICAgICAgICAgICAgInNvIHRoZSBpbmNsaW5hdGlvbiBjYW5ub3QgYmUgZGVyaXZlZC4gU3VwcGx5IHRoZSBhbmdsZXMgbWFudWFsbHkuIg0KICAgICAgICApDQoNCiAgICBiX292ZXJfYSA9IDEwLjAgKiogKC1sb2dfcjI1KQ0KICAgIG10eXBlID0gTm9uZQ0KICAgIHRyeToNCiAgICAgICAgbXR5cGUgPSBzdHIocm93WyJNVHlwZSJdKS5zdHJpcCgpIG9yIE5vbmUNCiAgICBleGNlcHQgS2V5RXJyb3I6DQogICAgICAgIHBhc3MNCg0KICAgIHEwID0gcTBfZm9yX3R5cGUobXR5cGUpDQogICAgbG9nX2QyNSA9IF9nZXQoImxvZ0QyNSIpDQoNCiAgICByZXR1cm4gew0KICAgICAgICAibmFtZV9xdWVyeSI6IGdhbGF4eV9uYW1lLA0KICAgICAgICAicGdjIjogX2dldCgiUEdDIiksDQogICAgICAgICJtdHlwZSI6IG10eXBlLA0KICAgICAgICAibG9nX3IyNSI6IGxvZ19yMjUsDQogICAgICAgICJiX292ZXJfYSI6IGJfb3Zlcl9hLA0KICAgICAgICAicGFfc2t5IjogX2dldCgiUEEiKSwNCiAgICAgICAgImluY19uYWl2ZSI6IG1hdGguZGVncmVlcyhtYXRoLmFjb3MobWluKG1heChiX292ZXJfYSwgMC4wKSwgMS4wKSkpLA0KICAgICAgICAiaW5jIjogaW5jbGluYXRpb25fZnJvbV9heGlzX3JhdGlvKGJfb3Zlcl9hLCBxMCksDQogICAgICAgICJxMCI6IHEwLA0KICAgICAgICAiZDI1X2FyY21pbiI6ICgwLjEgKiAxMC4wICoqIGxvZ19kMjUpIGlmIGxvZ19kMjUgaXMgbm90IE5vbmUgZWxzZSBOb25lLA0KICAgICAgICAic291cmNlIjogZiJIeXBlckxFREEgdmlhIFZpemllUiB7SFlQRVJMRURBX1ZJWklFUl9JRH0iLA0KICAgIH0NCg0KZGVmIGRlc2NyaWJlKGdlb206IGRpY3QpIC0+IHN0cjoNCiAgICAiIiJPbmUgcmVhZGFibGUgYmxvY2sgc3VtbWFyaXNpbmcgd2hhdCB0aGUgY2F0YWxvZ3VlIGdhdmUgdXMuIiIiDQogICAgbGluZXMgPSBbDQogICAgICAgIGYiICBjYXRhbG9ndWUgICAgICA6IHtnZW9tWydzb3VyY2UnXX0iLA0KICAgICAgICBmIiAgUEdDICAgICAgICAgICAgOiB7Z2VvbVsncGdjJ119IiwNCiAgICAgICAgZiIgIHR5cGUgICAgICAgICAgIDoge2dlb21bJ210eXBlJ119IiwNCiAgICAgICAgZiIgIGIvYSAgICAgICAgICAgIDoge2dlb21bJ2Jfb3Zlcl9hJ106LjNmfSAgIChsb2dSMjUgPSB7Z2VvbVsnbG9nX3IyNSddOi4zZn0pIiwNCiAgICAgICAgZiIgIGluY2xpbmF0aW9uICAgIDoge2dlb21bJ2luYyddOi4xZn0gZGVnICAgIg0KICAgICAgICBmIihuYWl2ZSBhY29zKGIvYSkgPSB7Z2VvbVsnaW5jX25haXZlJ106LjFmfSBkZWcsIHEwID0ge2dlb21bJ3EwJ106LjJmfSkiLA0KICAgIF0NCiAgICBpZiBnZW9tWyJwYV9za3kiXSBpcyBOb25lOg0KICAgICAgICBsaW5lcy5hcHBlbmQoIiAgcG9zaXRpb24gYW5nbGUgOiBOT1QgUFVCTElTSEVEIC0gdGhlIGdhbGF4eSBpcyB0b28gcm91bmQgZm9yIHRoZSAiDQogICAgICAgICAgICAgICAgICAgICAibWFqb3IgYXhpcyB0byBiZSBkZWZpbmVkIikNCiAgICBlbHNlOg0KICAgICAgICBsaW5lcy5hcHBlbmQoZiIgIHBvc2l0aW9uIGFuZ2xlIDoge2dlb21bJ3BhX3NreSddOi4xZn0gZGVnIG9uIHRoZSBza3kgKE4gdGhyb3VnaCBFKSIpDQogICAgaWYgZ2VvbVsiZDI1X2FyY21pbiJdOg0KICAgICAgICBsaW5lcy5hcHBlbmQoZiIgIEQyNSAgICAgICAgICAgIDoge2dlb21bJ2QyNV9hcmNtaW4nXTouMWZ9IGFyY21pbiIpDQogICAgcmV0dXJuICJcbiIuam9pbihsaW5lcykNCg0KUl9WID0gMy4xICAgICAgICAgICAgICAgICAgICAgICAgICAjIEFfViAvIEUoQi1WKSBmb3IgZHVzdCBpbiBvdXIgb3duIGdhbGF4eQ0KDQpESVNUQU5DRV9DQVRBTE9HID0gIkovQUovMTUyLzUwIiAgICMgQ29zbWljZmxvd3MtMywgVHVsbHkgZXQgYWwuIDIwMTYNCg0KZGVmIGxvb2t1cF9kaXN0YW5jZShuYW1lKToNCiAgICAiIiJSZWRzaGlmdC1pbmRlcGVuZGVudCBkaXN0YW5jZSBpbiBNcGMgZnJvbSBDb3NtaWNmbG93cy0zLCBvciBOb25lLg0KDQogICAgRXZlcnkgbGVuZ3RoIHRoaXMgdG9vbCByZXBvcnRzIGlzIHRoaXMgbnVtYmVyIHRpbWVzIGFuIGFuZ2xlLCBzbyBhIHdyb25nDQogICAgZGlzdGFuY2UgcmVzY2FsZXMgdGhlIHdob2xlIGFuc3dlciBhbmQgbm90aGluZyBpbiB0aGUgb3V0cHV0IGxvb2tzIGFtaXNzLg0KICAgIFJlZHNoaWZ0IGlzIG5vIGhlbHAgZm9yIGEgZ2FsYXh5IHRoaXMgY2xvc2UgLSB0aGUgcmVjZXNzaW9uIHZlbG9jaXR5IG9mIGENCiAgICBmZXctTXBjIGdhbGF4eSBpcyBkb21pbmF0ZWQgYnkgaXRzIG93biBtb3Rpb24gdGhyb3VnaCB0aGUgbG9jYWwgZ3JvdXAsIGFuZA0KICAgIE0xMDEncyBwdXRzIGl0IGF0IGFib3V0IGhhbGYgaXRzIHRydWUgZGlzdGFuY2UgLSBzbyBhIGNhdGFsb2d1ZSBvZiBkaXJlY3QNCiAgICBtZWFzdXJlbWVudHMgaXMgd2hhdCBpcyB3YW50ZWQuDQoNCiAgICBJdCBpcyBvZmZlcmVkLCBuZXZlciBpbXBvc2VkOiB0aGUgY2FsbGVyIGlzIGV4cGVjdGVkIHRvIHNob3cgaXQgYW5kIGxldCB0aGUNCiAgICBvYnNlcnZlciBvdmVycnVsZSBpdC4NCiAgICAiIiINCiAgICB0cnk6DQogICAgICAgIGZyb20gYXN0cm9weS5jb29yZGluYXRlcyBpbXBvcnQgU2t5Q29vcmQNCiAgICAgICAgZnJvbSBhc3Ryb3F1ZXJ5LnZpemllciBpbXBvcnQgVml6aWVyDQogICAgICAgIGltcG9ydCBhc3Ryb3B5LnVuaXRzIGFzIHUNCg0KICAgICAgICBjb29yZCA9IFNreUNvb3JkLmZyb21fbmFtZShuYW1lKQ0KICAgICAgICBoaXQgPSBWaXppZXIoY29sdW1ucz1bIioqIl0sIHJvd19saW1pdD01KS5xdWVyeV9yZWdpb24oDQogICAgICAgICAgICBjb29yZCwgcmFkaXVzPTIgKiB1LmFyY21pbiwgY2F0YWxvZz1ESVNUQU5DRV9DQVRBTE9HKQ0KICAgICAgICBpZiBub3QgaGl0Og0KICAgICAgICAgICAgcmV0dXJuIE5vbmUNCiAgICAgICAgdGFibGUgPSBoaXRbMF0NCiAgICAgICAgZm9yIGNvbCBpbiAoIkRpc3QiLCAiPERpc3Q+Iik6DQogICAgICAgICAgICBpZiBjb2wgaW4gdGFibGUuY29sbmFtZXM6DQogICAgICAgICAgICAgICAgdmFsdWUgPSBmbG9hdCh0YWJsZVtjb2xdWzBdKQ0KICAgICAgICAgICAgICAgIGlmIHZhbHVlID09IHZhbHVlIGFuZCB2YWx1ZSA+IDA6DQogICAgICAgICAgICAgICAgICAgIHJldHVybiB2YWx1ZQ0KICAgIGV4Y2VwdCBFeGNlcHRpb246DQogICAgICAgIHJldHVybiBOb25lDQogICAgcmV0dXJuIE5vbmUNCg0KZGVmIGxvb2t1cF9leHRpbmN0aW9uKG5hbWUpOg0KICAgICIiIkdhbGFjdGljIHJlZGRlbmluZyBFKEItVikgdG93YXJkcyB0aGUgZ2FsYXh5LCBvciBOb25lLg0KDQogICAgVGhlIFNjaGxhZmx5ICYgRmlua2JlaW5lciAoMjAxMSkgcmVjYWxpYnJhdGlvbiBvZiB0aGUgU2NobGVnZWwgZHVzdCBtYXBzLA0KICAgIHNlcnZlZCBieSBJUlNBLiBOZWVkZWQgd2hlcmV2ZXIgYSBjb2xvdXIgaXMgdHVybmVkIGludG8gYSBzdGF0ZW1lbnQgYWJvdXQgYQ0KICAgIHN0ZWxsYXIgcG9wdWxhdGlvbjogZHVzdCBpbiBvdXIgb3duIGdhbGF4eSByZWRkZW5zIGV2ZXJ5dGhpbmcgYmVoaW5kIGl0LCBhbmQNCiAgICB3aXRob3V0IHRoZSBjb3JyZWN0aW9uIGV2ZXJ5IHNvdXJjZSBpcyBzaGlmdGVkIHRvd2FyZHMgdGhlIHJlZCwgd2hpY2ggbW92ZXMNCiAgICB0aGUgbGluZSBiZXR3ZWVuIGEgeW91bmcgYmx1ZSBjbHVzdGVyIGFuZCBhbiBvcmRpbmFyeSBzdGFyLg0KDQogICAgV2hlcmUgaXQgY2FuY2VscywgYW5kIHRoaXMgaXMgd29ydGgga25vd2luZyBzbyBpdCBkb2VzIG5vdCBnZXQgImZpeGVkIiBsYXRlcjoNCiAgICBmaXR0aW5nIHRoZSBjb250aW51dW0gc3VidHJhY3Rpb24gZmFjdG9yIGFnYWluc3QgYW4gaW5zdHJ1bWVudGFsIGNvbG91cg0KICAgIG1lYXN1cmVkIGZyb20gdGhlIHNhbWUgcGFpciBvZiBmcmFtZXMuIFJlZGRlbmluZyBzaGlmdHMgZXZlcnkgc3RhciBieSB0aGUgc2FtZQ0KICAgIGFtb3VudCB0aGVyZSwgdGhlIGZpdHRlZCBsaW5lIHNoaWZ0cyB3aXRoIHRoZW0sIGFuZCB0aGUgdmFsdWUgcmVhZCBvZmYgaXQgZm9yDQogICAgYW55IGdpdmVuIHBpeGVsIGlzIHVuY2hhbmdlZC4NCiAgICAiIiINCiAgICB0cnk6DQogICAgICAgIGZyb20gYXN0cm9xdWVyeS5pcGFjLmlyc2EuaXJzYV9kdXN0IGltcG9ydCBJcnNhRHVzdA0KICAgICAgICB0YWJsZSA9IElyc2FEdXN0LmdldF9xdWVyeV90YWJsZShuYW1lLCBzZWN0aW9uPSJlYnYiKQ0KICAgICAgICB2YWx1ZSA9IGZsb2F0KHRhYmxlWyJleHQgU2FuZEYgbWVhbiJdWzBdKQ0KICAgICAgICBpZiB2YWx1ZSA9PSB2YWx1ZSBhbmQgdmFsdWUgPj0gMDoNCiAgICAgICAgICAgIHJldHVybiB2YWx1ZQ0KICAgIGV4Y2VwdCBFeGNlcHRpb246DQogICAgICAgIHJldHVybiBOb25lDQogICAgcmV0dXJuIE5vbmUNCg0KZGVmIGRlc2NyaWJlX2xvb2t1cHMobmFtZSwgZGlzdGFuY2VfbXBjLCBlYnYsIGRpc3RhbmNlX3NvdXJjZT0iIiwgZWJ2X3NvdXJjZT0iIik6DQogICAgIiIiV2hhdCB3YXMgbG9va2VkIHVwLCBzaG93biBiZWZvcmUgaXQgaXMgdXNlZC4iIiINCiAgICBsaW5lcyA9IFtdDQogICAgaWYgZGlzdGFuY2VfbXBjOg0KICAgICAgICBsaW5lcy5hcHBlbmQoIiAgZGlzdGFuY2UgICAgezouMmZ9IE1wYyAgIHt9Ii5mb3JtYXQoDQogICAgICAgICAgICBkaXN0YW5jZV9tcGMsIGRpc3RhbmNlX3NvdXJjZSBvciAiZnJvbSBDb3NtaWNmbG93cy0zIChUdWxseSBldCBhbC4gMjAxNikiKSkNCiAgICAgICAgbGluZXMuYXBwZW5kKCIgICAgICAgICAgICAgIGV2ZXJ5IGxlbmd0aCBiZWxvdyBzY2FsZXMgd2l0aCB0aGlzIG51bWJlciIpDQogICAgZWxzZToNCiAgICAgICAgbGluZXMuYXBwZW5kKCIgIGRpc3RhbmNlICAgIG5vdCBmb3VuZCAtIHJhZGlpIHdpbGwgYmUgaW4gYXJjc2VjLCBub3QgcGFyc2VjcyIpDQogICAgaWYgZWJ2IGlzIG5vdCBOb25lOg0KICAgICAgICBsaW5lcy5hcHBlbmQoIiAgcmVkZGVuaW5nICAgRShCLVYpID0gezouNGZ9LCBzbyBBX1YgPSB7Oi4zZn0gICB7fSIuZm9ybWF0KA0KICAgICAgICAgICAgZWJ2LCBSX1YgKiBlYnYsDQogICAgICAgICAgICBlYnZfc291cmNlIG9yICJmcm9tIFNjaGxhZmx5ICYgRmlua2JlaW5lciAoMjAxMSkiKSkNCiAgICByZXR1cm4gIlxuIi5qb2luKGxpbmVzKQ0KDQpkZWYgZ2FpYV9mb3JlZ3JvdW5kX3N0YXJzKHdjcywgc2hhcGUsIGdfbGltaXQ9MTkuMCwgc2lnbmlmaWNhbmNlPTMuMCwNCiAgICAgICAgICAgICAgICAgICAgICAgICAgbWF4X3Jvd3M9NTAwMDApOg0KICAgICIiIlBvc2l0aW9ucywgaW4gcGl4ZWxzLCBvZiBzdGFycyB0aGF0IGFyZSBpbiBmcm9udCBvZiB0aGUgZ2FsYXh5Lg0KDQogICAgV0hZIEEgQ0FUQUxPR1VFIEFORCBOT1QgVEhFIElNQUdFDQogICAgRXZlcnl0aGluZyBkb3duc3RyZWFtIHRoYXQgbmVlZHMgImEgc3RhciIgbmVlZHMgYSBmb3JlZ3JvdW5kIHN0YXI6IHNvbWV0aGluZw0KICAgIHdob3NlIGxpZ2h0IGlzIHB1cmUgY29udGludXVtLCBzbyB0aGF0IHNjYWxpbmcgb25lIGZpbHRlciBhZ2FpbnN0IGFub3RoZXIgYW5kDQogICAgc3VidHJhY3RpbmcgbWFrZXMgaXQgdmFuaXNoLiBEZXRlY3RpbmcgcG9pbnQgc291cmNlcyBpbiB0aGUgaW1hZ2UgZmluZHMgdGhvc2UsDQogICAgYnV0IGluIGEgbmVhcmJ5IGdhbGF4eSBpdCBhbHNvIGZpbmRzIHRoZSBnYWxheHkncyBvd24gc3RhcnMgLSBhbmQgdGhvc2UgYXJlIG5vdA0KICAgIHB1cmUgY29udGludXVtLiBTb21lIGFyZSByZWQgZ2lhbnRzLCBzb21lIGFyZSBob3QgeW91bmcgc3RhcnMgc2l0dGluZyBpbiB0aGUNCiAgICBnYXMgdGhhdCBpcyBiZWluZyBtZWFzdXJlZC4NCg0KICAgIEhvdyBiYWRseSB0aGlzIG1hdHRlcnMgZGVwZW5kcyBvbiBkaXN0YW5jZSwgd2hpY2ggaXMgd2h5IGl0IHdlbnQgdW5ub3RpY2VkIGZvcg0KICAgIGEgbG9uZyB0aW1lLiBBdCBhIGZldyBNcGMgYSBnYWxheHkncyBvd24gc3RhcnMgYXJlIG5vdCByZXNvbHZlZCBhbmQgZXZlcnkgcG9pbnQNCiAgICBzb3VyY2UgcmVhbGx5IGlzIGZvcmVncm91bmQuIEF0IE0zMydzIDAuOSBNcGMgdGhleSBhcmUgcmVzb2x2ZWQgaW4gdGhlaXINCiAgICB0aG91c2FuZHM6IG1lYXN1cmVkIGluIHRoaXMgZmllbGQsIG9mIDQyMDkgR2FpYSBzb3VyY2VzIGJyaWdodGVyIHRoYW4gRz0xOSwNCiAgICBvbmx5IDE1OTAgYXJlIGZvcmVncm91bmQuIFNldHRpbmcgdGhlIHN1YnRyYWN0aW9uIGZhY3RvciBmcm9tIHRoZSBtaXh0dXJlIGdhdmUNCiAgICBhIHN0YXItdG8tc3RhciBzY2F0dGVyIG9mIDY2IHBlcmNlbnQgLSB3aGljaCBpcyB0byBzYXkgbm8gZmFjdG9yIGF0IGFsbC4NCg0KICAgIFRIRSBURVNUDQogICAgUGFyYWxsYXggYW5kIHByb3BlciBtb3Rpb24uIEEgc3RhciBhIGZldyBodW5kcmVkIHBhcnNlY3MgYXdheSBpbiBvdXIgb3duDQogICAgZ2FsYXh5IGhhcyBib3RoLCBtZWFzdXJhYmx5LiBBIHN0YXIgYXQgMC45IE1wYyBoYXMgbmVpdGhlciwgYXQgYW55IHByZWNpc2lvbg0KICAgIEdhaWEgY2FuIHJlYWNoLiBTbyBhIHNpZ25pZmljYW50IHBhcmFsbGF4LCBvciBhIHNpZ25pZmljYW50IHByb3BlciBtb3Rpb24sIGlzDQogICAgYSBwaHlzaWNhbCBzdGF0ZW1lbnQgdGhhdCB0aGUgc3RhciBpcyBpbiBmcm9udCAtIG5vdCBhIGJyaWdodG5lc3MgY3V0IG9yIGENCiAgICBzaGFwZSBjdXQgdGhhdCBoYXBwZW5zIHRvIGNvcnJlbGF0ZSB3aXRoIGl0Lg0KDQogICAgUmV0dXJucyAocG9zaXRpb25zLCBkaWFnbm9zdGljcykuIFBvc2l0aW9ucyBpcyBhbiAoTiwgMikgYXJyYXkgb2YgeCwgeSBpbiB0aGUNCiAgICBmcmFtZTsgYW4gZW1wdHkgYXJyYXkgbWVhbnMgdGhlIHF1ZXJ5IGZhaWxlZCBvciBmb3VuZCBub3RoaW5nLCBhbmQgdGhlIGNhbGxlcg0KICAgIGlzIGV4cGVjdGVkIHRvIGZhbGwgYmFjayBvbiBkZXRlY3RpbmcgcG9pbnQgc291cmNlcy4NCiAgICAiIiINCiAgICBpbXBvcnQgbnVtcHkgYXMgbnANCg0KICAgIGRpYWcgPSB7InNvdXJjZSI6ICJHYWlhIERSMyIsICJuX2NhdGFsb2d1ZSI6IDAsICJuX2ZvcmVncm91bmQiOiAwLA0KICAgICAgICAgICAgIm5faW5fZnJhbWUiOiAwLCAid2h5IjogIiJ9DQogICAgaWYgd2NzIGlzIE5vbmU6DQogICAgICAgIGRpYWdbIndoeSJdID0gIm5vIHBsYXRlIHNvbHV0aW9uLCBzbyB0aGUgY2F0YWxvZ3VlIGNhbm5vdCBiZSBwbGFjZWQiDQogICAgICAgIHJldHVybiBucC5lbXB0eSgoMCwgMikpLCBkaWFnDQoNCiAgICB0cnk6DQogICAgICAgIGZyb20gYXN0cm9xdWVyeS5nYWlhIGltcG9ydCBHYWlhDQogICAgICAgIGltcG9ydCBhc3Ryb3B5LnVuaXRzIGFzIHUNCg0KICAgICAgICBueSwgbnggPSBzaGFwZVswXSwgc2hhcGVbMV0NCiAgICAgICAgY2VudHJlID0gd2NzLnBpeGVsX3RvX3dvcmxkKG54IC8gMi4wLCBueSAvIDIuMCkNCiAgICAgICAgY29ybmVyID0gd2NzLnBpeGVsX3RvX3dvcmxkKDAuMCwgMC4wKQ0KICAgICAgICByYWRpdXMgPSBmbG9hdChjZW50cmUuc2VwYXJhdGlvbihjb3JuZXIpLnRvKHUuZGVnKS52YWx1ZSkgKiAxLjA1DQoNCiAgICAgICAgcXVlcnkgPSAoDQogICAgICAgICAgICAiU0VMRUNUIHJhLCBkZWMsIHBob3RfZ19tZWFuX21hZyBGUk9NIGdhaWFkcjMuZ2FpYV9zb3VyY2UgIg0KICAgICAgICAgICAgIldIRVJFIDE9Q09OVEFJTlMoUE9JTlQocmEsIGRlYyksIENJUkNMRSh7Oi42Zn0sIHs6LjZmfSwgezouNGZ9KSkgIg0KICAgICAgICAgICAgIkFORCBwaG90X2dfbWVhbl9tYWcgPCB7Oi4yZn0gIg0KICAgICAgICAgICAgIkFORCAocGFyYWxsYXgvcGFyYWxsYXhfZXJyb3IgPiB7Oi4xZn0gT1IgIg0KICAgICAgICAgICAgIiAgICAgc3FydChwbXJhKnBtcmEgKyBwbWRlYypwbWRlYykgLyAiDQogICAgICAgICAgICAiICAgICBzcXJ0KHBtcmFfZXJyb3IqcG1yYV9lcnJvciArIHBtZGVjX2Vycm9yKnBtZGVjX2Vycm9yKSA+IHs6LjFmfSkgIg0KICAgICAgICAgICAgIk9SREVSIEJZIHBob3RfZ19tZWFuX21hZyINCiAgICAgICAgKS5mb3JtYXQoY2VudHJlLnJhLmRlZywgY2VudHJlLmRlYy5kZWcsIHJhZGl1cywgZ19saW1pdCwNCiAgICAgICAgICAgICAgICAgc2lnbmlmaWNhbmNlLCBzaWduaWZpY2FuY2UpDQoNCiAgICAgICAgam9iID0gR2FpYS5sYXVuY2hfam9iX2FzeW5jKHF1ZXJ5KSBpZiBtYXhfcm93cyA+IDIwMDAgZWxzZSBHYWlhLmxhdW5jaF9qb2IocXVlcnkpDQogICAgICAgIHRhYmxlID0gam9iLmdldF9yZXN1bHRzKCkNCiAgICAgICAgZGlhZ1sibl9mb3JlZ3JvdW5kIl0gPSBsZW4odGFibGUpDQogICAgICAgIGlmIG5vdCBsZW4odGFibGUpOg0KICAgICAgICAgICAgZGlhZ1sid2h5Il0gPSAibm8gZm9yZWdyb3VuZCBzdGFycyBpbiB0aGUgZmllbGQiDQogICAgICAgICAgICByZXR1cm4gbnAuZW1wdHkoKDAsIDIpKSwgZGlhZw0KDQogICAgICAgIHgsIHkgPSB3Y3Mud29ybGRfdG9fcGl4ZWxfdmFsdWVzKG5wLmFzYXJyYXkodGFibGVbInJhIl0sIGR0eXBlPWZsb2F0KSwNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbnAuYXNhcnJheSh0YWJsZVsiZGVjIl0sIGR0eXBlPWZsb2F0KSkNCiAgICAgICAgaW5zaWRlID0gKHggPiA1KSAmICh4IDwgbnggLSA2KSAmICh5ID4gNSkgJiAoeSA8IG55IC0gNikNCiAgICAgICAgZGlhZ1sibl9pbl9mcmFtZSJdID0gaW50KGluc2lkZS5zdW0oKSkNCiAgICAgICAgIyBCcmlnaHRlc3QgZmlyc3QuIFdob2V2ZXIgdGFrZXMgInRoZSBmaXJzdCB0d28gaHVuZHJlZCIgb2YgdGhpcyBsaXN0IG11c3QNCiAgICAgICAgIyBnZXQgdGhlIHR3byBodW5kcmVkIHdvcnRoIG1lYXN1cmluZzogYSBzdGFyIG5lYXIgdGhlIGNhdGFsb2d1ZSBsaW1pdCBpcw0KICAgICAgICAjIGEgZmV3IGNvdW50cyB0aHJvdWdoIGEgbmFycm93IGZpbHRlciwgYW5kIGEgcmF0aW8gYnVpbHQgb24gaXQgaXMgbm9pc2UuDQogICAgICAgICMgSGFuZGluZyBiYWNrIGFuIHVub3JkZXJlZCBsaXN0IGNvc3QgYSBmYWN0b3Igb2YgdHdvIGluIHRoZSBzY2F0dGVyIG9mIHRoZQ0KICAgICAgICAjIHN1YnRyYWN0aW9uIGZhY3RvciBiZWZvcmUgdGhpcyBsaW5lIGV4aXN0ZWQuDQogICAgICAgIGRpYWdbImdfcmFuZ2UiXSA9IChmbG9hdChucC5uYW5taW4odGFibGVbInBob3RfZ19tZWFuX21hZyJdKSksDQogICAgICAgICAgICAgICAgICAgICAgICAgICBmbG9hdChucC5uYW5tYXgodGFibGVbInBob3RfZ19tZWFuX21hZyJdKSkpDQogICAgICAgIHJldHVybiBucC5jb2x1bW5fc3RhY2soW3hbaW5zaWRlXSwgeVtpbnNpZGVdXSksIGRpYWcNCg0KICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOg0KICAgICAgICBkaWFnWyJ3aHkiXSA9ICJ7fToge30iLmZvcm1hdCh0eXBlKGV4YykuX19uYW1lX18sIGV4YykNCiAgICAgICAgcmV0dXJuIG5wLmVtcHR5KCgwLCAyKSksIGRpYWcNCg0KZGVmIGRlc2NyaWJlX2ZvcmVncm91bmQoZGlhZyk6DQogICAgIiIiV2hhdCB0aGUgY2F0YWxvZ3VlIGxvb2t1cCBmb3VuZCwgaW4gYSBsaW5lLiIiIg0KICAgIGlmIGRpYWcuZ2V0KCJuX2luX2ZyYW1lIik6DQogICAgICAgIHJldHVybiAoIiAge30gZm9yZWdyb3VuZCBzdGFycyBmcm9tIHt9IChwYXJhbGxheCBvciBwcm9wZXIgbW90aW9uICINCiAgICAgICAgICAgICAgICAic2lnbmlmaWNhbnQpOyB0aGUgZ2FsYXh5J3Mgb3duIHN0YXJzIGFyZSBub3QgYW1vbmcgdGhlbSIuZm9ybWF0KA0KICAgICAgICAgICAgICAgICAgICBkaWFnWyJuX2luX2ZyYW1lIl0sIGRpYWdbInNvdXJjZSJdKSkNCiAgICByZXR1cm4gKCIgIG5vIGNhdGFsb2d1ZSBzdGFycyBhdmFpbGFibGUgKHt9KSAtIGZhbGxpbmcgYmFjayBvbiBwb2ludCBzb3VyY2VzICINCiAgICAgICAgICAgICJmb3VuZCBpbiB0aGUgaW1hZ2UsIHdoaWNoIGluIGEgbmVhcmJ5IGdhbGF4eSBpbmNsdWRlcyBpdHMgb3duICINCiAgICAgICAgICAgICJzdGFycyIuZm9ybWF0KGRpYWcuZ2V0KCJ3aHkiKSBvciAidW5rbm93biIpKQ0KDQpkZXNjcmliZV9fZ2FsYXh5X2NhdGFsb2d1ZSA9IGRlc2NyaWJlICAgIyB0aGlzIG1vZHVsZSdzIG93bjsgdGhlIHBsYWluIG5hbWUgZ2V0cyBzaGFkb3dlZCBiZWxvdw0KDQoNCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0NCiMgc2hhcmVkIG1vZHVsZSByYWRpYWxfcHJvZmlsZXMucHkNCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0NCg0KbWF0cGxvdGxpYi51c2UoIkFnZyIpDQoNCnBhc3MgICMgZnJvbSBkZXByb2plY3Rpb24sIGlubGluZWQgYWJvdmUNCg0KQ09MT1VSUyA9IFsiIzQ0NDQ0NCIsICIjYTMyODNjIiwgIiMyYjVhODkiLCAiIzNmN2Q1MiIsICIjOGE1YTJiIl0NCg0KZGVmIHRyZWF0bWVudF9ub25lKCk6DQogICAgcmV0dXJuIHsia2V5IjogIm5vbmUiLCAibGFiZWwiOiAiQXMgbWVhc3VyZWQsIG5vIHRpbHQgY29ycmVjdGlvbiIsICJhbmdsZSI6IE5vbmUsICJpbmMiOiBOb25lLA0KICAgICAgICAgICAgIm5vdGUiOiAiUG9zaXRpb25zIGV4YWN0bHkgYXMgbWVhc3VyZWQgb24gdGhlIGltYWdlLiBDb250cm9sIHBhbmVsLiJ9DQoNCmRlZiB0cmVhdG1lbnRfY2F0YWxvZ3VlKGdlb20sIG5vcnRoX2FuZ2xlPTkwLjAsIG1pcnJvcmVkPUZhbHNlKToNCiAgICAiIiJEZXByb2plY3Rpb24gdXNpbmcgdGhlIHBvc2l0aW9uIGFuZ2xlIGFuZCBheGlzIHJhdGlvIGZyb20gSHlwZXJMRURBLiIiIg0KICAgIGlmIGdlb20gaXMgTm9uZToNCiAgICAgICAgcmV0dXJuIE5vbmUNCiAgICBwYV9za3ksIGluYyA9IGdlb20uZ2V0KCJwYV9za3kiKSwgZ2VvbS5nZXQoImluYyIpDQogICAgaWYgcGFfc2t5IGlzIE5vbmU6DQogICAgICAgIHJldHVybiB7ImtleSI6ICJsZWRhIiwgImxhYmVsIjogIlRpbHQgY29ycmVjdGlvbiBub3QgcG9zc2libGUiLCAiYW5nbGUiOiBOb25lLCAiaW5jIjogTm9uZSwNCiAgICAgICAgICAgICAgICAibm90ZSI6ICJIeXBlckxFREEgcHVibGlzaGVzIG5vIHBvc2l0aW9uIGFuZ2xlIGZvciB0aGlzIGdhbGF4eSAtIGl0IGlzICINCiAgICAgICAgICAgICAgICAgICAgICAgICJ0b28gcm91bmQgZm9yIGEgbWFqb3IgYXhpcyB0byBiZSBkZWZpbmVkLiBOb3QgY29ycmVjdGVkLiJ9DQogICAgb2ssIHJlYXNvbiA9IGluY2xpbmF0aW9uX2lzX3VzYWJsZShpbmMpDQogICAgaWYgbm90IG9rOg0KICAgICAgICByZXR1cm4geyJrZXkiOiAibGVkYSIsICJsYWJlbCI6ICJUaWx0IGNvcnJlY3Rpb24gbm90IHBvc3NpYmxlIiwgImFuZ2xlIjogTm9uZSwNCiAgICAgICAgICAgICAgICAiaW5jIjogTm9uZSwgIm5vdGUiOiByZWFzb259DQogICAgYW5nID0gc2t5X3BhX3RvX2ltYWdlX2FuZ2xlKHBhX3NreSwgbm9ydGhfYW5nbGUsIG1pcnJvcmVkKQ0KICAgIHJldHVybiB7DQogICAgICAgICJrZXkiOiAibGVkYSIsICJsYWJlbCI6ICJDb3JyZWN0ZWQgZm9yIGEgdGlsdCBvZiB7Oi4wZn0gZGVnIChIeXBlckxFREEpIi5mb3JtYXQoaW5jKSwNCiAgICAgICAgImFuZ2xlIjogYW5nLCAiaW5jIjogaW5jLA0KICAgICAgICAibm90ZSI6ICgiUEEgezouMWZ9IGRlZyBvbiBza3kgLT4gezouMWZ9IGRlZyBpbiBpbWFnZSAobm9ydGggYXQgezouMGZ9IGRlZ3t9KS4gIg0KICAgICAgICAgICAgICAgICAiU3RyZXRjaCB7Oi4yZn14IGFsb25nIHRoZSBtaW5vciBheGlzLiIpLmZvcm1hdCgNCiAgICAgICAgICAgICAgICAgICAgIHBhX3NreSwgYW5nLCBub3J0aF9hbmdsZSwgIiwgbWlycm9yZWQiIGlmIG1pcnJvcmVkIGVsc2UgIiIsDQogICAgICAgICAgICAgICAgICAgICBzdHJldGNoX2ZhY3RvcihpbmMpKSwNCiAgICB9DQoNCmRlZiB0cmVhdG1lbnRfbWFudWFsKGluYywgcGFfaW1hZ2U9Tm9uZSwgcGFfc2t5PU5vbmUsIG5vcnRoX2FuZ2xlPTkwLjAsIG1pcnJvcmVkPUZhbHNlKToNCiAgICAiIiJEZXByb2plY3Rpb24gdXNpbmcgYW5nbGVzIHRoZSB1c2VyIHN1cHBsaWVkLCBmcm9tIHRoZSBsaXRlcmF0dXJlIG9yIGJ5IGV5ZS4iIiINCiAgICBpZiBpbmMgaXMgTm9uZToNCiAgICAgICAgcmV0dXJuIE5vbmUNCiAgICBvaywgcmVhc29uID0gaW5jbGluYXRpb25faXNfdXNhYmxlKGluYykNCiAgICBpZiBwYV9pbWFnZSBpcyBub3QgTm9uZToNCiAgICAgICAgYW5nLCBzcmMgPSBwYV9pbWFnZSAlIDE4MC4wLCAiZ2l2ZW4gZGlyZWN0bHkgaW4gaW1hZ2UgY29vcmRpbmF0ZXMiDQogICAgZWxpZiBwYV9za3kgaXMgbm90IE5vbmU6DQogICAgICAgIGFuZyA9IHNreV9wYV90b19pbWFnZV9hbmdsZShwYV9za3ksIG5vcnRoX2FuZ2xlLCBtaXJyb3JlZCkNCiAgICAgICAgc3JjID0gImZyb20gc2t5IFBBIHs6LjFmfSBkZWciLmZvcm1hdChwYV9za3kpDQogICAgZWxzZToNCiAgICAgICAgcmV0dXJuIHsia2V5IjogIm1hbnVhbCIsICJsYWJlbCI6ICJUaWx0IGNvcnJlY3Rpb24gaW5jb21wbGV0ZSIsICJhbmdsZSI6IE5vbmUsDQogICAgICAgICAgICAgICAgImluYyI6IE5vbmUsICJub3RlIjogIkFuIGluY2xpbmF0aW9uIHdhcyBnaXZlbiBidXQgbm8gcG9zaXRpb24gYW5nbGUuIn0NCiAgICBpZiBub3Qgb2s6DQogICAgICAgIHJldHVybiB7ImtleSI6ICJtYW51YWwiLCAibGFiZWwiOiAiVGlsdCBjb3JyZWN0aW9uIG5vdCBwb3NzaWJsZSIsICJhbmdsZSI6IE5vbmUsDQogICAgICAgICAgICAgICAgImluYyI6IE5vbmUsICJub3RlIjogcmVhc29ufQ0KICAgIHJldHVybiB7ImtleSI6ICJtYW51YWwiLCAibGFiZWwiOiAiQ29ycmVjdGVkIGZvciBhIHRpbHQgb2YgezouMGZ9IGRlZyAoeW91cnMpIi5mb3JtYXQoaW5jKSwNCiAgICAgICAgICAgICJhbmdsZSI6IGFuZywgImluYyI6IGluYywNCiAgICAgICAgICAgICJub3RlIjogKCJVc2VyLXN1cHBsaWVkIGdlb21ldHJ5LCB7fS4gTWFqb3IgYXhpcyBhdCB7Oi4xZn0gZGVnIGluIHRoZSAiDQogICAgICAgICAgICAgICAgICAgICAiaW1hZ2UuIFN0cmV0Y2ggezouMmZ9eC4iKS5mb3JtYXQoc3JjLCBhbmcsIHN0cmV0Y2hfZmFjdG9yKGluYykpfQ0KDQpkZWYgdHJlYXRtZW50X3NlY3RvcihkeCwgZHksIG5fc2VjdG9ycz0yMCwgc3RlcD01LjApOg0KICAgICIiIkRlcHJvamVjdGlvbiBkZXJpdmVkIGZyb20gdGhlIG9iamVjdHMgdGhlbXNlbHZlcywgd2l0aCBpdHMgY2F2ZWF0cyBhdHRhY2hlZC4iIiINCiAgICBmaXQgPSBmaXRfc2VjdG9yX21ldGhvZChkeCwgZHksIG5fc2VjdG9ycz1uX3NlY3RvcnMsDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgYW5nbGVfc3RlcD1zdGVwLCBpbmNfc3RlcD1zdGVwKQ0KICAgIG9rLCByZWFzb24gPSBpbmNsaW5hdGlvbl9pc191c2FibGUoZml0WyJpbmMiXSkNCg0KICAgIHdhcm4gPSAiIg0KICAgIGlmIGZpdFsiZmV3X29iamVjdHMiXToNCiAgICAgICAgd2FybiArPSAoIiBPbmx5IHt9IG9iamVjdHMgLSBiZWxvdyB0aGUgMTUwIHRvIDIwMCBuZWVkZWQgZm9yIGEgc2luZ2xlIGNsZWFuICINCiAgICAgICAgICAgICAgICAgIm1pbmltdW0sIHNvIHRyZWF0IHRoaXMgd2l0aCBzdXNwaWNpb24uIi5mb3JtYXQoZml0WyJuX29iamVjdHMiXSkpDQogICAgaWYgZml0WyJydW5uZXJzX3VwIl06DQogICAgICAgIGEsIGMsIGQgPSBmaXRbInJ1bm5lcnNfdXAiXVswXQ0KICAgICAgICBtYXJnaW4gPSAoZCAtIGZpdFsiZGlzcGVyc2lvbiJdKSAvIG1heChmaXRbImRpc3BlcnNpb24iXSwgMWUtOSkNCiAgICAgICAgd2FybiArPSAoIiBDbG9zZXN0IHJpdmFsIHNvbHV0aW9uOiBQQSB7Oi4wZn0gZGVnLCBpIHs6LjBmfSBkZWcsIHdvcnNlIGJ5IG9ubHkgIg0KICAgICAgICAgICAgICAgICAiezouMCV9LiIuZm9ybWF0KGEsIGMsIG1hcmdpbikpDQoNCiAgICBpZiBub3Qgb2s6DQogICAgICAgIHJldHVybiB7ImtleSI6ICJzZWN0b3IiLCAibGFiZWwiOiAiVGlsdCBmcm9tIHRoZSBkYXRhIC0gbm90IHBvc3NpYmxlIiwgImFuZ2xlIjogTm9uZSwNCiAgICAgICAgICAgICAgICAiaW5jIjogTm9uZSwNCiAgICAgICAgICAgICAgICAibm90ZSI6ICgiU2VjdG9yIG1ldGhvZCBmb3VuZCBQQSB7Oi4wZn0gZGVnLCBpIHs6LjBmfSBkZWcsIGJ1dCAiDQogICAgICAgICAgICAgICAgICAgICAgICAgLmZvcm1hdChmaXRbImFuZ2xlIl0sIGZpdFsiaW5jIl0pICsgcmVhc29uICsgd2Fybil9DQogICAgcmV0dXJuIHsia2V5IjogInNlY3RvciIsICJsYWJlbCI6ICJDb3JyZWN0ZWQgZm9yIGEgdGlsdCBvZiB7Oi4wZn0gZGVnIChtZWFzdXJlZCBoZXJlKSIuZm9ybWF0KGZpdFsiaW5jIl0pLA0KICAgICAgICAgICAgImFuZ2xlIjogZml0WyJhbmdsZSJdLCAiaW5jIjogZml0WyJpbmMiXSwNCiAgICAgICAgICAgICJub3RlIjogKCJBbmdsZXMgZGVyaXZlZCBmcm9tIHRoZSBvYmplY3RzIHRoZW1zZWx2ZXMgYnkgZXZlbmluZyBvdXQgdGhlICINCiAgICAgICAgICAgICAgICAgICAgICJjb3VudHMgaW4ge30gd2VkZ2VzOiBQQSB7Oi4wZn0gZGVnIGluIGltYWdlLCBpIHs6LjBmfSBkZWcsICINCiAgICAgICAgICAgICAgICAgICAgICJzdHJldGNoIHs6LjJmfXguIikuZm9ybWF0KG5fc2VjdG9ycywgZml0WyJhbmdsZSJdLCBmaXRbImluYyJdLA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgc3RyZXRjaF9mYWN0b3IoZml0WyJpbmMiXSkpICsgd2Fybn0NCg0KZGVmIGNvbXB1dGVfcHJvZmlsZXMoeCwgeSwgY2VudHJlLCBmcmFtZSwgdHJlYXRtZW50cywgcmluZ3M9MjApOg0KICAgICIiIkRlcHJvamVjdCBhbmQgYmluIHRoZSBvYmplY3RzIG9uY2UgcGVyIHRyZWF0bWVudC4iIiINCiAgICBkeDAsIGR5MCA9IG5wLmFzYXJyYXkoeCwgZmxvYXQpIC0gY2VudHJlWzBdLCBucC5hc2FycmF5KHksIGZsb2F0KSAtIGNlbnRyZVsxXQ0KICAgIHJlc3VsdHMgPSBbXQ0KICAgIGZvciB0IGluIHRyZWF0bWVudHM6DQogICAgICAgIGlmIHRbImFuZ2xlIl0gaXMgTm9uZToNCiAgICAgICAgICAgIGR4cCwgZHlwID0gZHgwLCBkeTANCiAgICAgICAgZWxzZToNCiAgICAgICAgICAgIGR4cCwgZHlwID0gZGVwcm9qZWN0KGR4MCwgZHkwLCB0WyJhbmdsZSJdLCB0WyJpbmMiXSkNCiAgICAgICAgcmVzID0gcmFkaWFsX2RlbnNpdHkoZHhwLCBkeXAsIG5fcmluZ3M9cmluZ3MsIGZyYW1lX2Nvcm5lcnM9ZnJhbWUsDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNlbnRyZV94eT1jZW50cmUsIG1ham9yX2F4aXNfYW5nbGVfZGVnPXRbImFuZ2xlIl0sDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgIGluY19kZWc9dFsiaW5jIl0pDQogICAgICAgIHJlc1siZHgiXSwgcmVzWyJkeSJdID0gZHhwLCBkeXANCiAgICAgICAgcmVzdWx0cy5hcHBlbmQocmVzKQ0KICAgIHJldHVybiByZXN1bHRzDQoNCmRlZiBnZW9tZXRyaWNfYXJlYXMoZWRnZXMpOg0KICAgICIiIlRoZSBmdWxsIG1hdGhlbWF0aWNhbCByaW5nIGFyZWFzLCBmb3IgY29tcGFyaXNvbiB3aXRoIHRoZSBvYnNlcnZlZCBvbmVzLiIiIg0KICAgIHJldHVybiBtYXRoLnBpICogKG5wLmFzYXJyYXkoZWRnZXMpWzE6XSAqKiAyIC0gbnAuYXNhcnJheShlZGdlcylbOi0xXSAqKiAyKQ0KDQpkZWYgd3JpdGVfcHJvZmlsZXNfY3N2KHBhdGgsIHRyZWF0bWVudHMsIHJlc3VsdHMsIHNjYWxlPTEuMCwgdW5pdD0icHgiKToNCiAgICAiIiJTYXZlIGV2ZXJ5IHByb2ZpbGUsIGluY2x1ZGluZyBib3RoIHRoZSBvYnNlcnZlZCBhbmQgdGhlIGZ1bGwgcmluZyBhcmVhcy4NCg0KICAgIEtlZXBpbmcgdGhlIGZ1bGwgZ2VvbWV0cmljIGFyZWEgYWxvbmdzaWRlIHRoZSBvYnNlcnZlZCBvbmUgbGV0cyBhbnlvbmUgY29tcGFyZQ0KICAgIGFnYWluc3Qgb2xkZXIgYW5hbHlzZXMsIHdoaWNoIGRpdmlkZWQgYnkgdGhlIGZ1bGwgYXJlYSBhbmQgc28gdW5kZXItcmVwb3J0ZWQgdGhlDQogICAgZGVuc2l0eSBvZiBhbnkgcmluZyB0aGUgZnJhbWUgaGFkIGNsaXBwZWQuDQogICAgIiIiDQogICAgd2l0aCBvcGVuKHBhdGgsICJ3IiwgZW5jb2Rpbmc9InV0Zi04IikgYXMgZmg6DQogICAgICAgIGZoLndyaXRlKCJ0cmVhdG1lbnQsbWFqb3JfYXhpc19hbmdsZV9kZWcsaW5jbGluYXRpb25fZGVnLCINCiAgICAgICAgICAgICAgICAgInJfaW5uZXJfe3V9LHJfb3V0ZXJfe3V9LHJfY2VudHJlX3t1fSxuX29iamVjdHMsIg0KICAgICAgICAgICAgICAgICAib2JzZXJ2ZWRfYXJlYV97dX0yLGZ1bGxfcmluZ19hcmVhX3t1fTIsYXJlYV9mcmFjdGlvbiwiDQogICAgICAgICAgICAgICAgICJkZW5zaXR5X3Blcl97dX0yLGRlbnNpdHlfZXJyLHJpbmdfY29tcGxldGVcbiIuZm9ybWF0KHU9dW5pdCkpDQogICAgICAgIGZvciB0LCByZXMgaW4gemlwKHRyZWF0bWVudHMsIHJlc3VsdHMpOg0KICAgICAgICAgICAgZ2VvID0gZ2VvbWV0cmljX2FyZWFzKHJlc1siZWRnZXMiXSkNCiAgICAgICAgICAgIGZvciBrIGluIHJhbmdlKGxlbihyZXNbInIiXSkpOg0KICAgICAgICAgICAgICAgIGZoLndyaXRlKCJ7fSx7fSx7fSx7Oi40Zn0sezouNGZ9LHs6LjRmfSx7fSx7Oi40Zn0sezouNGZ9LHs6LjRmfSwiDQogICAgICAgICAgICAgICAgICAgICAgICAgIns6LjZlfSx7Oi42ZX0se31cbiIuZm9ybWF0KA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0WyJsYWJlbCJdLA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAiIiBpZiB0WyJhbmdsZSJdIGlzIE5vbmUgZWxzZSAiezouMWZ9Ii5mb3JtYXQodFsiYW5nbGUiXSksDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICIiIGlmIHRbImluYyJdIGlzIE5vbmUgZWxzZSAiezouMWZ9Ii5mb3JtYXQodFsiaW5jIl0pLA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXNbImVkZ2VzIl1ba10gKiBzY2FsZSwgcmVzWyJlZGdlcyJdW2sgKyAxXSAqIHNjYWxlLA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXNbInIiXVtrXSAqIHNjYWxlLCBpbnQocmVzWyJjb3VudHMiXVtrXSksDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJlc1siYXJlYSJdW2tdICogc2NhbGUgKiogMiwgZ2VvW2tdICogc2NhbGUgKiogMiwNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVzWyJhcmVhX2ZyYWN0aW9uIl1ba10sDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJlc1siZGVuc2l0eSJdW2tdIC8gc2NhbGUgKiogMiwNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVzWyJkZW5zaXR5X2VyciJdW2tdIC8gc2NhbGUgKiogMiwNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgaW50KHJlc1siY29tcGxldGUiXVtrXSkpKQ0KDQpkZWYgc3RlcF9wcm9maWxlKGF4LCBlZGdlcywgdmFsdWVzLCBjb2xvdXIsIGx3PTEuNywgYWxwaGE9MS4wLCBscz0iLSIsIGxhYmVsPU5vbmUpOg0KICAgICIiIkRyYXcgYSBkZW5zaXR5IHByb2ZpbGUgYXMgc3RlcHMsIGZsYXQgYWNyb3NzIGVhY2ggcmluZy4NCg0KICAgIEEgc3VyZmFjZSBkZW5zaXR5IGlzIG5vdCBhIG1lYXN1cmVtZW50IGF0IGEgcG9pbnQuIEl0IGlzIG9uZSBudW1iZXIgZm9yIGEgd2hvbGUNCiAgICByaW5nIC0gdGhlIGNvdW50IGluIGl0IGRpdmlkZWQgYnkgaXRzIGFyZWEgLSBhbmQgaXQgc2F5cyBub3RoaW5nIGFib3V0IHdoZXJlDQogICAgaW5zaWRlIHRoYXQgcmluZyB0aGUgb2JqZWN0cyBzYXQuIERyYXdpbmcgaXQgYXMgYSBkb3QgYXQgdGhlIHJpbmcncyBjZW50cmUNCiAgICBqb2luZWQgdG8gdGhlIG5leHQgZG90IGJ5IGEgc2xvcGluZyBsaW5lIGludml0ZXMgZXhhY3RseSB0aGUgcmVhZGluZyB0aGUgZGF0YQ0KICAgIGRvZXMgbm90IHN1cHBvcnQ6IHRoYXQgdGhlIGRlbnNpdHkgdmFyaWVzIHNtb290aGx5IGFuZCB0aGF0IHRoZSB2YWx1ZSBhdCB0aGUNCiAgICBjZW50cmUgb2YgdGhlIHJpbmcgaXMgc29tZWhvdyBiZXR0ZXIgbWVhc3VyZWQgdGhhbiBhdCBpdHMgZWRnZS4NCg0KICAgIFN0ZXBzIHNheSB3aGF0IHdhcyBhY3R1YWxseSBtZWFzdXJlZCwgYW5kIGl0IGlzIGhvdyB0aGUgdG9vbCBpbiB0aGUgcmVwb3NpdG9yeQ0KICAgIGhhcyBhbHdheXMgZHJhd24gdGhpcywgc28gdGhlIHR3byBjYW4gYmUgbGFpZCBzaWRlIGJ5IHNpZGUuDQogICAgIiIiDQogICAgeSA9IG5wLmFwcGVuZCh2YWx1ZXMsIHZhbHVlc1stMV0pDQogICAgYXguc3RlcChlZGdlcywgeSwgd2hlcmU9InBvc3QiLCBjb2xvcj1jb2xvdXIsIGx3PWx3LCBhbHBoYT1hbHBoYSwgbHM9bHMsDQogICAgICAgICAgICBsYWJlbD1sYWJlbCkNCg0KZGVmIHByb2ZpbGVfZmlndXJlKHRyZWF0bWVudHMsIHJlc3VsdHMsIHVuaXRfbGFiZWwsIHRpdGxlLCBzY2FsZT0xLjApOg0KICAgICIiIk9uZSByb3cgcGVyIHRyZWF0bWVudCAtIHNjYXR0ZXIgYW5kIHByb2ZpbGUgLSB0aGVuIGFsbCBwcm9maWxlcyBvdmVybGFpZC4iIiINCiAgICBuID0gbGVuKHRyZWF0bWVudHMpDQogICAgZmlnID0gcGx0LmZpZ3VyZShmaWdzaXplPSgxMS4wLCA0LjEgKiBuICsgMy40KSkNCiAgICBncyA9IGZpZy5hZGRfZ3JpZHNwZWMobiArIDEsIDIsIGhlaWdodF9yYXRpb3M9WzFdICogbiArIFsxLjI1XSwNCiAgICAgICAgICAgICAgICAgICAgICAgICAgaHNwYWNlPTAuODUsIHdzcGFjZT0wLjMwKQ0KDQogICAgZm9yIGksICh0LCByZXMpIGluIGVudW1lcmF0ZSh6aXAodHJlYXRtZW50cywgcmVzdWx0cykpOg0KICAgICAgICBjb2wgPSBDT0xPVVJTW2kgJSBsZW4oQ09MT1VSUyldDQogICAgICAgIGR4dSwgZHl1ID0gcmVzWyJkeCJdICogc2NhbGUsIHJlc1siZHkiXSAqIHNjYWxlDQogICAgICAgIHJ1ID0gcmVzWyJyIl0gKiBzY2FsZQ0KICAgICAgICBkdSA9IHJlc1siZGVuc2l0eSJdIC8gc2NhbGUgKiogMg0KICAgICAgICBldSA9IHJlc1siZGVuc2l0eV9lcnIiXSAvIHNjYWxlICoqIDINCg0KICAgICAgICBheCA9IGZpZy5hZGRfc3VicGxvdChnc1tpLCAwXSkNCiAgICAgICAgYXguc2NhdHRlcihkeHUsIGR5dSwgcz03LCBjPWNvbCwgYWxwaGE9MC42NSwgbGluZXdpZHRocz0wKQ0KICAgICAgICBheC5wbG90KDAsIDAsICIrIiwgYz0iayIsIG1zPTExLCBtZXc9MS42KQ0KICAgICAgICBsaW0gPSAxLjA2ICogbWF4KG5wLmFicyhkeHUpLm1heCgpLCBucC5hYnMoZHl1KS5tYXgoKSkNCiAgICAgICAgYXguc2V0X3hsaW0oLWxpbSwgbGltKTsgYXguc2V0X3lsaW0oLWxpbSwgbGltKQ0KICAgICAgICBheC5zZXRfYXNwZWN0KCJlcXVhbCIsIGFkanVzdGFibGU9ImJveCIpDQogICAgICAgIGF4LnNldF90aXRsZSh0WyJsYWJlbCJdLCBmb250c2l6ZT0xMC41LCBsb2M9ImxlZnQiKQ0KICAgICAgICBheC5zZXRfeGxhYmVsKCJ4IGZyb20gY2VudHJlIFsiICsgdW5pdF9sYWJlbCArICJdIiwgZm9udHNpemU9OC41KQ0KICAgICAgICBheC5zZXRfeWxhYmVsKCJ5IGZyb20gY2VudHJlIFsiICsgdW5pdF9sYWJlbCArICJdIiwgZm9udHNpemU9OC41KQ0KICAgICAgICBheC50aWNrX3BhcmFtcyhsYWJlbHNpemU9Ny41KQ0KICAgICAgICBheC5ncmlkKGFscGhhPTAuMTUsIGx3PTAuNikNCg0KICAgICAgICBheDIgPSBmaWcuYWRkX3N1YnBsb3QoZ3NbaSwgMV0pDQogICAgICAgIGcgPSByZXNbImNvbXBsZXRlIl0NCiAgICAgICAgZWRnZXMgPSByZXNbImVkZ2VzIl0gKiBzY2FsZQ0KICAgICAgICAjIFRoZSBzdGVwIGlzIHRoZSBtZWFzdXJlbWVudC4gVGhlIGJhcnMgb24gdG9wIG9mIGl0IGFyZSB0aGUgY291bnRpbmcNCiAgICAgICAgIyB1bmNlcnRhaW50eSwgZHJhd24gYXQgdGhlIG1pZGRsZSBvZiBlYWNoIHJpbmcgb25seSBiZWNhdXNlIHRoZXkgaGF2ZSB0bw0KICAgICAgICAjIGJlIGRyYXduIHNvbWV3aGVyZS4NCiAgICAgICAgc3RlcF9wcm9maWxlKGF4MiwgZWRnZXMsIG5wLndoZXJlKGcsIGR1LCBucC5uYW4pLCBjb2wpDQogICAgICAgIGlmIGcuYW55KCk6DQogICAgICAgICAgICBheDIuZXJyb3JiYXIocnVbZ10sIGR1W2ddLCB5ZXJyPWV1W2ddLCBmbXQ9Im5vbmUiLCBlY29sb3I9Y29sLA0KICAgICAgICAgICAgICAgICAgICAgICAgIGVsaW5ld2lkdGg9MS4wLCBjYXBzaXplPTIsIGFscGhhPTAuNzUpDQogICAgICAgIGlmICh+ZykuYW55KCk6DQogICAgICAgICAgICBzdGVwX3Byb2ZpbGUoYXgyLCBlZGdlcywgbnAud2hlcmUofmcsIGR1LCBucC5uYW4pLCBjb2wsDQogICAgICAgICAgICAgICAgICAgICAgICAgbHc9MS4yLCBhbHBoYT0wLjQ1LCBscz0iLS0iLA0KICAgICAgICAgICAgICAgICAgICAgICAgIGxhYmVsPSJyaW5nIGNsaXBwZWQgYnkgdGhlIGZyYW1lIikNCiAgICAgICAgICAgIGF4Mi5sZWdlbmQoZm9udHNpemU9NywgZnJhbWVvbj1GYWxzZSwgbG9jPSJsb3dlciBsZWZ0IikNCiAgICAgICAgYXgyLnNldF95c2NhbGUoImxvZyIpDQogICAgICAgIGF4Mi5zZXRfeGxhYmVsKCJkZXByb2plY3RlZCByYWRpdXMgWyIgKyB1bml0X2xhYmVsICsgIl0iLCBmb250c2l6ZT04LjUpDQogICAgICAgIGF4Mi5zZXRfeWxhYmVsKCJzdXJmYWNlIGRlbnNpdHkgW24gcGVyICIgKyB1bml0X2xhYmVsICsgIl4yXSIsIGZvbnRzaXplPTguNSkNCiAgICAgICAgYXgyLnRpY2tfcGFyYW1zKGxhYmVsc2l6ZT03LjUpDQogICAgICAgIGF4Mi5ncmlkKGFscGhhPTAuMTUsIGx3PTAuNikNCiAgICAgICAgYXgyLnNldF90aXRsZSgiTiA9ICIgKyBzdHIoaW50KHJlc1siY291bnRzIl0uc3VtKCkpKSwgZm9udHNpemU9OSwgbG9jPSJyaWdodCIpDQoNCiAgICAgICAgYXgudGV4dCgwLjAsIC0wLjMwLCAiXG4iLmpvaW4odGV4dHdyYXAud3JhcCh0WyJub3RlIl0sIHdpZHRoPTEyNSkpLA0KICAgICAgICAgICAgICAgIHRyYW5zZm9ybT1heC50cmFuc0F4ZXMsIGZvbnRzaXplPTcuNiwgY29sb3I9IiM0YTRhNGEiLA0KICAgICAgICAgICAgICAgIHZhPSJ0b3AiLCBoYT0ibGVmdCIsIGxpbmVzcGFjaW5nPTEuNSkNCg0KICAgIGF4YyA9IGZpZy5hZGRfc3VicGxvdChnc1tuLCA6XSkNCiAgICBmb3IgaSwgKHQsIHJlcykgaW4gZW51bWVyYXRlKHppcCh0cmVhdG1lbnRzLCByZXN1bHRzKSk6DQogICAgICAgIGcgPSByZXNbImNvbXBsZXRlIl0NCiAgICAgICAgaWYgZy5hbnkoKToNCiAgICAgICAgICAgIHN0ZXBfcHJvZmlsZShheGMsIHJlc1siZWRnZXMiXSAqIHNjYWxlLA0KICAgICAgICAgICAgICAgICAgICAgICAgIG5wLndoZXJlKGcsIHJlc1siZGVuc2l0eSJdIC8gc2NhbGUgKiogMiwgbnAubmFuKSwNCiAgICAgICAgICAgICAgICAgICAgICAgICBDT0xPVVJTW2kgJSBsZW4oQ09MT1VSUyldLCBsYWJlbD10WyJsYWJlbCJdKQ0KICAgIGF4Yy5zZXRfeXNjYWxlKCJsb2ciKQ0KICAgIGF4Yy5zZXRfeGxhYmVsKCJkZXByb2plY3RlZCByYWRpdXMgWyIgKyB1bml0X2xhYmVsICsgIl0iKQ0KICAgIGF4Yy5zZXRfeWxhYmVsKCJzdXJmYWNlIGRlbnNpdHkgW24gcGVyICIgKyB1bml0X2xhYmVsICsgIl4yXSIpDQogICAgYXhjLmdyaWQoYWxwaGE9MC4xOCwgbHc9MC42KQ0KICAgIGF4Yy5sZWdlbmQoZm9udHNpemU9OC41LCBmcmFtZW9uPUZhbHNlKQ0KICAgIGF4Yy5zZXRfdGl0bGUoIkFsbCB0cmVhdG1lbnRzIG92ZXJsYWlkIC0gdGhlIHNwcmVhZCBiZXR3ZWVuIHRoZW0gaXMgdGhlICINCiAgICAgICAgICAgICAgICAgICJ1bmNlcnRhaW50eSBvbiB0aGUgcmFkaWkiLCBmb250c2l6ZT0xMCwgbG9jPSJsZWZ0IikNCg0KICAgIGZpZy5zdXB0aXRsZSh0aXRsZSwgZm9udHNpemU9MTMsIHk9MC45OTUpDQogICAgcmV0dXJuIGZpZw0KDQpkZWYgX3Byb2ZpbGVfcm93cyhyZXN1bHQsIHNjYWxlLCB1bml0KToNCiAgICAiIiJUaGUgcHJvZmlsZSBleGFjdGx5IGFzIHRoZSBwaWN0dXJlIGRyYXdzIGl0LCByaW5nIGJ5IHJpbmcuIiIiDQogICAgZWRnZXMgPSByZXN1bHRbImVkZ2VzIl0gKiBzY2FsZQ0KICAgIGhlYWQgPSBbInJpbmdfaW5uZXJfIiArIHVuaXQsICJyaW5nX291dGVyXyIgKyB1bml0LCAicl9jZW50cmVfIiArIHVuaXQsDQogICAgICAgICAgICAibl9vYmplY3RzIiwgImRlbnNpdHlfcGVyXyIgKyB1bml0ICsgIjIiLCAiZGVuc2l0eV9lcnJvciIsDQogICAgICAgICAgICAicmluZ19jb21wbGV0ZSJdDQogICAgcm93cyA9IFtdDQogICAgZm9yIGsgaW4gcmFuZ2UobGVuKHJlc3VsdFsiciJdKSk6DQogICAgICAgIHJvd3MuYXBwZW5kKFtmbG9hdChlZGdlc1trXSksIGZsb2F0KGVkZ2VzW2sgKyAxXSksDQogICAgICAgICAgICAgICAgICAgICBmbG9hdChyZXN1bHRbInIiXVtrXSAqIHNjYWxlKSwNCiAgICAgICAgICAgICAgICAgICAgIGludChyZXN1bHRbImNvdW50cyJdW2tdKSwNCiAgICAgICAgICAgICAgICAgICAgIGZsb2F0KHJlc3VsdFsiZGVuc2l0eSJdW2tdIC8gc2NhbGUgKiogMiksDQogICAgICAgICAgICAgICAgICAgICBmbG9hdChyZXN1bHRbImRlbnNpdHlfZXJyIl1ba10gLyBzY2FsZSAqKiAyKSwNCiAgICAgICAgICAgICAgICAgICAgIGludChib29sKHJlc3VsdFsiY29tcGxldGUiXVtrXSkpXSkNCiAgICByZXR1cm4gaGVhZCwgcm93cw0KDQpkZWYgX3Bvc2l0aW9uX3Jvd3MocmVzdWx0LCBzY2FsZSwgdW5pdCk6DQogICAgIiIiRXZlcnkgb2JqZWN0LCB3aGVyZSB0aGUgcGljdHVyZSBwdXRzIGl0OiBkZXByb2plY3RlZCwgY2VudHJlZCwgaW4gYHVuaXRgLiIiIg0KICAgIGhlYWQgPSBbInhfZnJvbV9jZW50cmVfIiArIHVuaXQsICJ5X2Zyb21fY2VudHJlXyIgKyB1bml0LA0KICAgICAgICAgICAgInJhZGl1c18iICsgdW5pdF0NCiAgICBkeCA9IG5wLmFzYXJyYXkocmVzdWx0WyJkeCJdLCBkdHlwZT1mbG9hdCkgKiBzY2FsZQ0KICAgIGR5ID0gbnAuYXNhcnJheShyZXN1bHRbImR5Il0sIGR0eXBlPWZsb2F0KSAqIHNjYWxlDQogICAgcm93cyA9IFtbZmxvYXQoYSksIGZsb2F0KGIpLCBmbG9hdChucC5oeXBvdChhLCBiKSldIGZvciBhLCBiIGluIHppcChkeCwgZHkpXQ0KICAgIHJldHVybiBoZWFkLCByb3dzDQoNCmRlZiBfd3JpdGVfdGFibGVzKHBhdGgsIHNoZWV0cyk6DQogICAgIiIiV3JpdGUgZWFjaCB0YWJsZSBhcyBpdHMgb3duIENTViwgbmFtZWQgYWZ0ZXIgdGhlIHBpY3R1cmUgaXQgYmVsb25ncyB0by4NCg0KICAgIFdIWSBPTkUgRklMRSBQRVIgVEFCTEUgQU5EIE5PVCBPTkUgUEVSIFBJQ1RVUkUNCiAgICBCZWNhdXNlIHRoZSB0d28gdGFibGVzIGJlaGluZCBhIHBpY3R1cmUgaGF2ZSBkaWZmZXJlbnQgc2hhcGVzIC0gb25lIHJvdyBwZXINCiAgICByaW5nIGluIHRoZSBwcm9maWxlLCBvbmUgcm93IHBlciBvYmplY3QgaW4gdGhlIHBvc2l0aW9ucyAtIGFuZCBwdXR0aW5nIHRoZW0gaW4NCiAgICBvbmUgZmlsZSBtZWFucyBvbmUgb2YgdGhlbSBzdGFydHMgaGFsZndheSBkb3duLCB3aGljaCBpcyBleGFjdGx5IHRoZSBsYXlvdXQNCiAgICB0aGF0IG1ha2VzIGEgc3ByZWFkc2hlZXQgYXdrd2FyZCB0byBjaGFydCBmcm9tLiBUd28gZmlsZXMsIGVhY2ggYSBjbGVhbg0KICAgIHJlY3RhbmdsZSB3aXRoIGEgaGVhZGVyIHJvdywgY2FuIGJlIGNoYXJ0ZWQgYnkgc2VsZWN0aW5nIHR3byBjb2x1bW5zLg0KDQogICAgQ1NWIHJhdGhlciB0aGFuIGEgc3ByZWFkc2hlZXQgZm9ybWF0IG9uIHB1cnBvc2U6IGl0IG9wZW5zIGluIEV4Y2VsLCBpdCBvcGVucw0KICAgIGluIGFueXRoaW5nIGVsc2UsIGFuZCBpdCBuZWVkcyBubyBsaWJyYXJ5IHRoYXQgbWlnaHQgbm90IGJlIGluc3RhbGxlZCBvbiB0aGUNCiAgICBtYWNoaW5lIHRoaXMgZW5kcyB1cCBydW5uaW5nIG9uLg0KDQogICAgUmV0dXJucyB0aGUgcGF0aHMgd3JpdHRlbi4NCiAgICAiIiINCiAgICBpbXBvcnQgY3N2IGFzIF9jc3YNCg0KICAgIHN0ZW0gPSBvcy5wYXRoLnNwbGl0ZXh0KHBhdGgpWzBdDQogICAgd3JpdHRlbiA9IFtdDQogICAgZm9yIG5hbWUsIChoZWFkLCByb3dzKSBpbiBzaGVldHMuaXRlbXMoKToNCiAgICAgICAgc2x1ZyA9ICJfIi5qb2luKHAgZm9yIHAgaW4NCiAgICAgICAgICAgICAgICAgICAgICAgICIiLmpvaW4oYyBpZiBjLmlzYWxudW0oKSBlbHNlICJfIiBmb3IgYyBpbiBuYW1lKS5zcGxpdCgiXyIpDQogICAgICAgICAgICAgICAgICAgICAgICBpZiBwKS5sb3dlcigpDQogICAgICAgIG91dCA9ICJ7fV97fS5jc3YiLmZvcm1hdChzdGVtLCBzbHVnKQ0KICAgICAgICB3aXRoIG9wZW4ob3V0LCAidyIsIG5ld2xpbmU9IiIsIGVuY29kaW5nPSJ1dGYtOCIpIGFzIGZoOg0KICAgICAgICAgICAgd3JpdGVyID0gX2Nzdi53cml0ZXIoZmgpDQogICAgICAgICAgICB3cml0ZXIud3JpdGVyb3coaGVhZCkNCiAgICAgICAgICAgIHdyaXRlci53cml0ZXJvd3Mocm93cykNCiAgICAgICAgd3JpdHRlbi5hcHBlbmQob3V0KQ0KICAgIHJldHVybiB3cml0dGVuDQoNCmRlZiB3cml0ZV90cmVhdG1lbnRfd29ya2Jvb2socGF0aCwgdHJlYXRtZW50LCByZXN1bHQsIHNjYWxlLCB1bml0KToNCiAgICAiIiJUaGUgdHdvIHRhYmxlcyBiZWhpbmQgb25lIHRyZWF0bWVudCdzIHBpY3R1cmUuIiIiDQogICAgcmV0dXJuIF93cml0ZV90YWJsZXMocGF0aCwgew0KICAgICAgICAicHJvZmlsZSI6IF9wcm9maWxlX3Jvd3MocmVzdWx0LCBzY2FsZSwgdW5pdCksDQogICAgICAgICJwb3NpdGlvbnMiOiBfcG9zaXRpb25fcm93cyhyZXN1bHQsIHNjYWxlLCB1bml0KSwNCiAgICB9KQ0KDQpkZWYgd3JpdGVfb3ZlcmxheV93b3JrYm9vayhwYXRoLCB0cmVhdG1lbnRzLCByZXN1bHRzLCBzY2FsZSwgdW5pdCk6DQogICAgIiIiT25lIHRhYmxlIHBlciBjdXJ2ZSBpbiB0aGUgb3ZlcmxheS4iIiINCiAgICBzaGVldHMgPSB7fQ0KICAgIGZvciB0cmVhdG1lbnQsIHJlc3VsdCBpbiB6aXAodHJlYXRtZW50cywgcmVzdWx0cyk6DQogICAgICAgIG5hbWUgPSAiIi5qb2luKGMgaWYgKGMuaXNhbG51bSgpIG9yIGMgPT0gIiAiKSBlbHNlICIgIg0KICAgICAgICAgICAgICAgICAgICAgICBmb3IgYyBpbiB0cmVhdG1lbnRbImxhYmVsIl0pLnN0cmlwKCkNCiAgICAgICAgc2hlZXRzW25hbWVbOjMxXSBvciAidHJlYXRtZW50Il0gPSBfcHJvZmlsZV9yb3dzKHJlc3VsdCwgc2NhbGUsIHVuaXQpDQogICAgcmV0dXJuIF93cml0ZV90YWJsZXMocGF0aCwgc2hlZXRzKQ0KDQpkZWYgb25lX3RyZWF0bWVudF9maWd1cmUodHJlYXRtZW50LCByZXN1bHQsIHVuaXRfbGFiZWwsIHRpdGxlLCBzY2FsZT0xLjAsDQogICAgICAgICAgICAgICAgICAgICAgICAgY29sb3VyX2luZGV4PTApOg0KICAgICIiIk9uZSB0cmVhdG1lbnQgb24gaXRzIG93biBzaGVldDogd2hlcmUgdGhlIG9iamVjdHMgYXJlLCBhbmQgdGhlaXIgcHJvZmlsZS4NCg0KICAgIFdIWSBUSEUgU0FNRSBUSElORyBJUyBEUkFXTiBUV0lDRQ0KICAgIFRoZSBjb21iaW5lZCBzaGVldCBpcyBmb3IgY29tcGFyaW5nIHRoZSB0cmVhdG1lbnRzIGFnYWluc3QgZWFjaCBvdGhlciwgd2hpY2gNCiAgICBpcyB3aGF0IGl0IGlzIGZvciBhbmQgd2hhdCBpdCBpcyBnb29kIGF0LiBJdCBpcyBub3Qgd2hhdCB5b3UgcHV0IGluIGZyb250IG9mDQogICAgc29tZWJvZHksIG9yIGludG8gYSByZXBvcnQ6IGF0IGZvdXIgaW5jaGVzIGhpZ2ggYSBwYW5lbCBpcyB0b28gc21hbGwgdG8gcmVhZCBhDQogICAgcmFkaXVzIG9mZiBpdCwgYW5kIHR3byB0cmVhdG1lbnRzIHNpZGUgYnkgc2lkZSBjb21wZXRlIGZvciB0aGUgZXllIHdoZW4gb25seQ0KICAgIG9uZSBvZiB0aGVtIGlzIHVuZGVyIGRpc2N1c3Npb24uIFNvIGVhY2ggaXMgd3JpdHRlbiBhZ2FpbiBvbiBpdHMgb3duLCBmdWxsDQogICAgc2l6ZSwgYW5kIHRoZSBvdmVybGF5IHRvby4gVGhlIGNvbWJpbmVkIHNoZWV0IGlzIHVuY2hhbmdlZDsgdGhlc2UgYXJlIGV4dHJhLg0KICAgICIiIg0KICAgIGNvbCA9IENPTE9VUlNbY29sb3VyX2luZGV4ICUgbGVuKENPTE9VUlMpXQ0KICAgIGZpZyA9IHBsdC5maWd1cmUoZmlnc2l6ZT0oMTMuMCwgNi4yKSkNCiAgICBncyA9IGZpZy5hZGRfZ3JpZHNwZWMoMSwgMiwgd3NwYWNlPTAuMjgpDQoNCiAgICBkeHUsIGR5dSA9IHJlc3VsdFsiZHgiXSAqIHNjYWxlLCByZXN1bHRbImR5Il0gKiBzY2FsZQ0KICAgIHJ1ID0gcmVzdWx0WyJyIl0gKiBzY2FsZQ0KICAgIGR1ID0gcmVzdWx0WyJkZW5zaXR5Il0gLyBzY2FsZSAqKiAyDQogICAgZXUgPSByZXN1bHRbImRlbnNpdHlfZXJyIl0gLyBzY2FsZSAqKiAyDQoNCiAgICBheCA9IGZpZy5hZGRfc3VicGxvdChnc1swLCAwXSkNCiAgICBheC5zY2F0dGVyKGR4dSwgZHl1LCBzPTE0LCBjPWNvbCwgYWxwaGE9MC43LCBsaW5ld2lkdGhzPTApDQogICAgYXgucGxvdCgwLCAwLCAiKyIsIGM9ImsiLCBtcz0xNCwgbWV3PTEuOCkNCiAgICBsaW0gPSAxLjA2ICogbWF4KG5wLmFicyhkeHUpLm1heCgpLCBucC5hYnMoZHl1KS5tYXgoKSkNCiAgICBheC5zZXRfeGxpbSgtbGltLCBsaW0pDQogICAgYXguc2V0X3lsaW0oLWxpbSwgbGltKQ0KICAgIGF4LnNldF9hc3BlY3QoImVxdWFsIiwgYWRqdXN0YWJsZT0iYm94IikNCiAgICBheC5zZXRfeGxhYmVsKCJ4IGZyb20gY2VudHJlIFsiICsgdW5pdF9sYWJlbCArICJdIikNCiAgICBheC5zZXRfeWxhYmVsKCJ5IGZyb20gY2VudHJlIFsiICsgdW5pdF9sYWJlbCArICJdIikNCiAgICBheC5ncmlkKGFscGhhPTAuMTUsIGx3PTAuNikNCiAgICBheC5zZXRfdGl0bGUoIndoZXJlIHRoZXkgYXJlIiwgZm9udHNpemU9MTEsIGxvYz0ibGVmdCIpDQoNCiAgICBheDIgPSBmaWcuYWRkX3N1YnBsb3QoZ3NbMCwgMV0pDQogICAgZyA9IHJlc3VsdFsiY29tcGxldGUiXQ0KICAgIGVkZ2VzID0gcmVzdWx0WyJlZGdlcyJdICogc2NhbGUNCiAgICBzdGVwX3Byb2ZpbGUoYXgyLCBlZGdlcywgbnAud2hlcmUoZywgZHUsIG5wLm5hbiksIGNvbCkNCiAgICBpZiBnLmFueSgpOg0KICAgICAgICBheDIuZXJyb3JiYXIocnVbZ10sIGR1W2ddLCB5ZXJyPWV1W2ddLCBmbXQ9Im5vbmUiLCBlY29sb3I9Y29sLA0KICAgICAgICAgICAgICAgICAgICAgZWxpbmV3aWR0aD0xLjIsIGNhcHNpemU9MywgYWxwaGE9MC44KQ0KICAgIGlmICh+ZykuYW55KCk6DQogICAgICAgIHN0ZXBfcHJvZmlsZShheDIsIGVkZ2VzLCBucC53aGVyZSh+ZywgZHUsIG5wLm5hbiksIGNvbCwNCiAgICAgICAgICAgICAgICAgICAgIGx3PTEuMywgYWxwaGE9MC40NSwgbHM9Ii0tIiwNCiAgICAgICAgICAgICAgICAgICAgIGxhYmVsPSJyaW5nIGNsaXBwZWQgYnkgdGhlIGZyYW1lIikNCiAgICAgICAgYXgyLmxlZ2VuZChmb250c2l6ZT05LCBmcmFtZW9uPUZhbHNlLCBsb2M9Imxvd2VyIGxlZnQiKQ0KICAgIGF4Mi5zZXRfeXNjYWxlKCJsb2ciKQ0KICAgIGF4Mi5zZXRfeGxhYmVsKCJkZXByb2plY3RlZCByYWRpdXMgWyIgKyB1bml0X2xhYmVsICsgIl0iKQ0KICAgIGF4Mi5zZXRfeWxhYmVsKCJzdXJmYWNlIGRlbnNpdHkgW24gcGVyICIgKyB1bml0X2xhYmVsICsgIl4yXSIpDQogICAgYXgyLmdyaWQoYWxwaGE9MC4xNSwgbHc9MC42KQ0KICAgIGF4Mi5zZXRfdGl0bGUoIk4gPSAiICsgc3RyKGludChyZXN1bHRbImNvdW50cyJdLnN1bSgpKSksIGZvbnRzaXplPTEwLA0KICAgICAgICAgICAgICAgICAgbG9jPSJyaWdodCIpDQogICAgYXgyLnNldF90aXRsZSgiaG93IG1hbnkgcGVyIHVuaXQgYXJlYSwgYWdhaW5zdCByYWRpdXMiLCBmb250c2l6ZT0xMSwNCiAgICAgICAgICAgICAgICAgIGxvYz0ibGVmdCIpDQoNCiAgICBmaWcuc3VwdGl0bGUoInt9IC0ge30iLmZvcm1hdCh0aXRsZSwgdHJlYXRtZW50WyJsYWJlbCJdKSwgZm9udHNpemU9MTMpDQogICAgZmlnLnRleHQoMC4wOSwgMC4wMTUsDQogICAgICAgICAgICAgY2hyKDEwKS5qb2luKHRleHR3cmFwLndyYXAodHJlYXRtZW50WyJub3RlIl0sIHdpZHRoPTE1MCkpLA0KICAgICAgICAgICAgIGZvbnRzaXplPTksIGNvbG9yPSIjNGE0YTRhIiwgdmE9ImJvdHRvbSIpDQogICAgZmlnLnN1YnBsb3RzX2FkanVzdChib3R0b209MC4xNywgdG9wPTAuOTApDQogICAgcmV0dXJuIGZpZw0KDQpkZWYgb3ZlcmxheV9maWd1cmUodHJlYXRtZW50cywgcmVzdWx0cywgdW5pdF9sYWJlbCwgdGl0bGUsIHNjYWxlPTEuMCk6DQogICAgIiIiVGhlIHRyZWF0bWVudHMgb24gb25lIHBhaXIgb2YgYXhlcywgb24gYSBzaGVldCBvZiBpdHMgb3duLiIiIg0KICAgIGZpZywgYXggPSBwbHQuc3VicGxvdHMoZmlnc2l6ZT0oMTMuMCwgNi4yKSkNCiAgICBmb3IgaSwgKHQsIHJlcykgaW4gZW51bWVyYXRlKHppcCh0cmVhdG1lbnRzLCByZXN1bHRzKSk6DQogICAgICAgIGcgPSByZXNbImNvbXBsZXRlIl0NCiAgICAgICAgaWYgZy5hbnkoKToNCiAgICAgICAgICAgIHN0ZXBfcHJvZmlsZShheCwgcmVzWyJlZGdlcyJdICogc2NhbGUsDQogICAgICAgICAgICAgICAgICAgICAgICAgbnAud2hlcmUoZywgcmVzWyJkZW5zaXR5Il0gLyBzY2FsZSAqKiAyLCBucC5uYW4pLA0KICAgICAgICAgICAgICAgICAgICAgICAgIENPTE9VUlNbaSAlIGxlbihDT0xPVVJTKV0sIGxhYmVsPXRbImxhYmVsIl0pDQogICAgYXguc2V0X3lzY2FsZSgibG9nIikNCiAgICBheC5zZXRfeGxhYmVsKCJkZXByb2plY3RlZCByYWRpdXMgWyIgKyB1bml0X2xhYmVsICsgIl0iKQ0KICAgIGF4LnNldF95bGFiZWwoInN1cmZhY2UgZGVuc2l0eSBbbiBwZXIgIiArIHVuaXRfbGFiZWwgKyAiXjJdIikNCiAgICBheC5ncmlkKGFscGhhPTAuMTgsIGx3PTAuNikNCiAgICBheC5sZWdlbmQoZm9udHNpemU9MTAsIGZyYW1lb249RmFsc2UpDQogICAgYXguc2V0X3RpdGxlKCJ7fSAtIGFsbCB0cmVhdG1lbnRzIG92ZXJsYWlkOyB0aGUgc3ByZWFkIGJldHdlZW4gdGhlbSBpcyB0aGUgIg0KICAgICAgICAgICAgICAgICAidW5jZXJ0YWludHkgb24gdGhlIHJhZGlpIi5mb3JtYXQodGl0bGUpLCBmb250c2l6ZT0xMiwgbG9jPSJsZWZ0IikNCiAgICBmaWcudGlnaHRfbGF5b3V0KCkNCiAgICByZXR1cm4gZmlnDQoNCmRlZiBydW5fcHJvZmlsZXMoeCwgeSwgY2VudHJlLCBmcmFtZSwgb3V0X3ByZWZpeCwgdGl0bGUsDQogICAgICAgICAgICAgICAgIGdhbGF4eT1Ob25lLCBnZW9tPU5vbmUsIG5vcnRoX2FuZ2xlPTkwLjAsIG1pcnJvcmVkPUZhbHNlLA0KICAgICAgICAgICAgICAgICBtYW51YWxfaW5jPU5vbmUsIG1hbnVhbF9wYV9pbWFnZT1Ob25lLCBtYW51YWxfcGFfc2t5PU5vbmUsDQogICAgICAgICAgICAgICAgIHNlY3Rvcl9tZXRob2Q9RmFsc2UsIHNlY3RvcnM9MjAsIHNlY3Rvcl9zdGVwPTUuMCwNCiAgICAgICAgICAgICAgICAgcmluZ3M9MjAsIHNjYWxlPTEuMCwgdW5pdD0icHgiLCB2ZXJib3NlPVRydWUpOg0KICAgICIiIkRlcHJvamVjdCwgYmluLCBwbG90IGFuZCBzYXZlLiBSZXR1cm5zICh0cmVhdG1lbnRzLCByZXN1bHRzLCBwYXRocykuDQoNCiAgICBvdXRfcHJlZml4IGlzIGEgcGF0aCBzdGVtOyB0aGlzIHdyaXRlcyA8cHJlZml4Pl9wcm9maWxlcy5jc3YgYW5kDQogICAgPHByZWZpeD5fcHJvZmlsZXMucG5nIG5leHQgdG8gaXQuDQogICAgIiIiDQogICAgeCA9IG5wLmFzYXJyYXkoeCwgZHR5cGU9ZmxvYXQpDQogICAgeSA9IG5wLmFzYXJyYXkoeSwgZHR5cGU9ZmxvYXQpDQoNCiAgICAjIFRvbyBtYW55IHJpbmdzIGZvciB0b28gZmV3IG9iamVjdHMgZG9lcyBub3QgYWRkIGRldGFpbCwgaXQgbWFudWZhY3R1cmVzDQogICAgIyBub2lzZTogcmluZ3MgaG9sZGluZyBvbmUgb2JqZWN0IG9yIG5vbmUsIGVycm9yIGJhcnMgdGFsbGVyIHRoYW4gdGhlIHByb2ZpbGUsDQogICAgIyBhbmQgYSBqYWdnZWQgbGluZSB0aGF0IGludml0ZXMgcmVhZGluZyBzdHJ1Y3R1cmUgaW50byBjb3VudGluZyBzdGF0aXN0aWNzLg0KICAgICMgU2VlbiBvbiBhIHJlYWwgcnVuIC0gMTgxIGJsdWUga25vdHMgc3ByZWFkIG92ZXIgMzAgcmluZ3MsIHdob2xlIHJpbmdzIGVtcHR5Lg0KICAgIGlmIHJpbmdzIGFuZCBsZW4oeCkgYW5kIGxlbih4KSAvIGZsb2F0KHJpbmdzKSA8IDguMDoNCiAgICAgICAgc3VnZ2VzdGVkID0gbWF4KDQsIGludChsZW4oeCkgLyAxMikpDQogICAgICAgIHByaW50KCIgIG5vdGU6IHt9IG9iamVjdHMgb3ZlciB7fSByaW5ncyBpcyBhYm91dCB7Oi4xZn0gcGVyIHJpbmcsIHNvIHRoZSAiDQogICAgICAgICAgICAgICJwcm9maWxlIGJlbG93Ii5mb3JtYXQobGVuKHgpLCByaW5ncywgbGVuKHgpIC8gZmxvYXQocmluZ3MpKSkNCiAgICAgICAgcHJpbnQoIiAgICAgICAgd2lsbCBiZSBkb21pbmF0ZWQgYnkgY291bnRpbmcgbm9pc2UuIEFib3V0IHt9IHJpbmdzIHdvdWxkICINCiAgICAgICAgICAgICAgImJlIHJlYWRhYmxlLiIuZm9ybWF0KHN1Z2dlc3RlZCkpDQoNCiAgICBpZiBnZW9tIGlzIE5vbmUgYW5kIGdhbGF4eToNCiAgICAgICAgdHJ5Og0KICAgICAgICAgICAgZGVzY3JpYmUgPSBkZXNjcmliZV9fZ2FsYXh5X2NhdGFsb2d1ZSAgIyBmcm9tIGdhbGF4eV9jYXRhbG9ndWUsIGlubGluZWQgYWJvdmUNCiAgICAgICAgICAgIGdlb20gPSBoeXBlcmxlZGFfZ2VvbWV0cnkoZ2FsYXh5KQ0KICAgICAgICAgICAgaWYgdmVyYm9zZToNCiAgICAgICAgICAgICAgICBwcmludCgiXG5IeXBlckxFREEgZ2VvbWV0cnkgZm9yICIgKyBnYWxheHkgKyAiOiIpDQogICAgICAgICAgICAgICAgcHJpbnQoZGVzY3JpYmUoZ2VvbSkpDQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOg0KICAgICAgICAgICAgaWYgdmVyYm9zZToNCiAgICAgICAgICAgICAgICBwcmludCgiXG5DYXRhbG9ndWUgbG9va3VwIGZhaWxlZCAoe306IHt9KS4gQ29udGludWluZyB3aXRob3V0IGl0LiINCiAgICAgICAgICAgICAgICAgICAgICAuZm9ybWF0KHR5cGUoZXhjKS5fX25hbWVfXywgZXhjKSkNCg0KICAgIHRyZWF0bWVudHMgPSBbdHJlYXRtZW50X25vbmUoKV0NCiAgICB0ID0gdHJlYXRtZW50X2NhdGFsb2d1ZShnZW9tLCBub3J0aF9hbmdsZSwgbWlycm9yZWQpDQogICAgaWYgdDoNCiAgICAgICAgdHJlYXRtZW50cy5hcHBlbmQodCkNCiAgICB0ID0gdHJlYXRtZW50X21hbnVhbChtYW51YWxfaW5jLCBtYW51YWxfcGFfaW1hZ2UsIG1hbnVhbF9wYV9za3ksDQogICAgICAgICAgICAgICAgICAgICAgICAgbm9ydGhfYW5nbGUsIG1pcnJvcmVkKQ0KICAgIGlmIHQ6DQogICAgICAgIHRyZWF0bWVudHMuYXBwZW5kKHQpDQogICAgaWYgc2VjdG9yX21ldGhvZDoNCiAgICAgICAgdHJlYXRtZW50cy5hcHBlbmQodHJlYXRtZW50X3NlY3Rvcih4IC0gY2VudHJlWzBdLCB5IC0gY2VudHJlWzFdLA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHNlY3RvcnMsIHNlY3Rvcl9zdGVwKSkNCg0KICAgIHJlc3VsdHMgPSBjb21wdXRlX3Byb2ZpbGVzKHgsIHksIGNlbnRyZSwgZnJhbWUsIHRyZWF0bWVudHMsIHJpbmdzPXJpbmdzKQ0KDQogICAgaWYgdmVyYm9zZToNCiAgICAgICAgcHJpbnQoIlxuVHJlYXRtZW50czoiKQ0KICAgICAgICBmb3IgdCwgcmVzIGluIHppcCh0cmVhdG1lbnRzLCByZXN1bHRzKToNCiAgICAgICAgICAgIHByaW50KCIgIC0ge306IHJfbWF4ID0gezouMWZ9IHt9LCB7fSBvZiB7fSByaW5ncyBjbGlwcGVkIGJ5IHRoZSBmcmFtZSINCiAgICAgICAgICAgICAgICAgIC5mb3JtYXQodFsibGFiZWwiXSwgcmVzWyJyX21heCJdICogc2NhbGUsIHVuaXQsDQogICAgICAgICAgICAgICAgICAgICAgICAgIGludCgofnJlc1siY29tcGxldGUiXSkuc3VtKCkpLCByaW5ncykpDQogICAgICAgICAgICBwcmludCgiICAgICAgIiArIHRbIm5vdGUiXSkNCg0KICAgIGNzdl9wYXRoID0gb3V0X3ByZWZpeCArICJfcHJvZmlsZXMuY3N2Ig0KICAgIHBuZ19wYXRoID0gb3V0X3ByZWZpeCArICJfcHJvZmlsZXMucG5nIg0KICAgIHdyaXRlX3Byb2ZpbGVzX2Nzdihjc3ZfcGF0aCwgdHJlYXRtZW50cywgcmVzdWx0cywgc2NhbGUsIHVuaXQpDQogICAgZmlnID0gcHJvZmlsZV9maWd1cmUodHJlYXRtZW50cywgcmVzdWx0cywgdW5pdCwgdGl0bGUsIHNjYWxlKQ0KICAgIGZpZy5zYXZlZmlnKHBuZ19wYXRoLCBkcGk9MTUwLCBiYm94X2luY2hlcz0idGlnaHQiKQ0KICAgIHBsdC5jbG9zZShmaWcpDQogICAgY29tYmluZWRfYm9va3MgPSB3cml0ZV9vdmVybGF5X3dvcmtib29rKA0KICAgICAgICBvcy5wYXRoLnNwbGl0ZXh0KHBuZ19wYXRoKVswXSArICIuY3N2IiwgdHJlYXRtZW50cywgcmVzdWx0cywgc2NhbGUsIHVuaXQpDQoNCiAgICBpZiB2ZXJib3NlOg0KICAgICAgICBwcmludCgiXG5Xcm90ZSAiICsgY3N2X3BhdGgpDQogICAgICAgIHByaW50KCJXcm90ZSAiICsgcG5nX3BhdGgpDQogICAgIyBBbmQgZWFjaCBwYXJ0IGFnYWluIG9uIGEgc2hlZXQgb2YgaXRzIG93biwgZnVsbCBzaXplLiBUaGUgY29tYmluZWQgb25lDQogICAgIyBhYm92ZSBpcyB1bnRvdWNoZWQ7IHRoZXNlIGFyZSBpbiBhZGRpdGlvbiB0byBpdC4NCiAgICBleHRyYV9wYXRocyA9IFtdDQogICAgZm9yIGksICh0cmVhdCwgcmVzKSBpbiBlbnVtZXJhdGUoemlwKHRyZWF0bWVudHMsIHJlc3VsdHMpKToNCiAgICAgICAgc2x1ZyA9ICcnLmpvaW4oYyBpZiBjLmlzYWxudW0oKSBlbHNlICdfJyBmb3IgYyBpbiB0cmVhdFsnbGFiZWwnXSkNCiAgICAgICAgc2x1ZyA9ICdfJy5qb2luKHAgZm9yIHAgaW4gc2x1Zy5zcGxpdCgnXycpIGlmIHApLmxvd2VyKCkNCiAgICAgICAgb25lX3BhdGggPSAne31fcHJvZmlsZV97fS5wbmcnLmZvcm1hdChvdXRfcHJlZml4LCBzbHVnKQ0KICAgICAgICBvbmUgPSBvbmVfdHJlYXRtZW50X2ZpZ3VyZSh0cmVhdCwgcmVzLCB1bml0LCB0aXRsZSwgc2NhbGUsDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvbG91cl9pbmRleD1pKQ0KICAgICAgICBvbmUuc2F2ZWZpZyhvbmVfcGF0aCwgZHBpPTE1MCwgYmJveF9pbmNoZXM9J3RpZ2h0JykNCiAgICAgICAgcGx0LmNsb3NlKG9uZSkNCiAgICAgICAgZXh0cmFfcGF0aHMuYXBwZW5kKG9uZV9wYXRoKQ0KICAgICAgICBleHRyYV9wYXRocy5leHRlbmQod3JpdGVfdHJlYXRtZW50X3dvcmtib29rKA0KICAgICAgICAgICAgb25lX3BhdGhbOi00XSArICcuY3N2JywgdHJlYXQsIHJlcywgc2NhbGUsIHVuaXQpKQ0KICAgIG92ZXJfcGF0aCA9IG91dF9wcmVmaXggKyAnX3Byb2ZpbGVfYWxsX292ZXJsYWlkLnBuZycNCiAgICBvdmVyID0gb3ZlcmxheV9maWd1cmUodHJlYXRtZW50cywgcmVzdWx0cywgdW5pdCwgdGl0bGUsIHNjYWxlKQ0KICAgIG92ZXIuc2F2ZWZpZyhvdmVyX3BhdGgsIGRwaT0xNTAsIGJib3hfaW5jaGVzPSd0aWdodCcpDQogICAgcGx0LmNsb3NlKG92ZXIpDQogICAgZXh0cmFfcGF0aHMuYXBwZW5kKG92ZXJfcGF0aCkNCiAgICBleHRyYV9wYXRocy5leHRlbmQod3JpdGVfb3ZlcmxheV93b3JrYm9vaygNCiAgICAgICAgb3Zlcl9wYXRoWzotNF0gKyAnLmNzdicsIHRyZWF0bWVudHMsIHJlc3VsdHMsIHNjYWxlLCB1bml0KSkNCiAgICBleHRyYV9wYXRoc1s6MF0gPSBjb21iaW5lZF9ib29rcw0KICAgIGlmIHZlcmJvc2U6DQogICAgICAgIGZvciBwIGluIGV4dHJhX3BhdGhzOg0KICAgICAgICAgICAgcHJpbnQoJ1dyb3RlICcgKyBwKQ0KICAgIHJldHVybiB0cmVhdG1lbnRzLCByZXN1bHRzLCB0dXBsZShbY3N2X3BhdGgsIHBuZ19wYXRoXSArIGV4dHJhX3BhdGhzKQ0KDQoNCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0NCiMgQmx1ZV9DbHVzdGVyc19Gcm9tX0ltYWdlcy5weQ0KIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQ0KDQpBUEVSVFVSRV9TQ0FMRSA9IDINCg0KQU5OVUxVU19JTk5FUl9TQ0FMRSA9IDIuNQ0KDQpBTk5VTFVTX09VVEVSX1NDQUxFID0gMw0KDQpGV0hNX1dJTkRPV19TSVpFID0gMTANCg0KREFPRklORF9GV0hNID0gTm9uZSAgICAgICAgICAgICAgIyBOb25lID0gbWVhc3VyZSBpdCBmcm9tIHRoZSBkYXRhIChyZWNvbW1lbmRlZCkNCg0KU0VFSU5HX0ZBTExCQUNLID0gNC4wICAgICAgICAgICAgIyBVc2VkIG9ubHkgaWYgdGhlIG1lYXN1cmVtZW50IGNhbm5vdCBiZSBtYWRlDQoNClNFRUlOR19NSU4sIFNFRUlOR19NQVggPSAxLjUsIDE1LjAgICAjIFNhbml0eSByYW5nZSBmb3IgdGhlIG1lYXN1cmVkIHZhbHVlLCBwaXhlbHMNCg0KU0lHTUFfQ0xJUCA9IDMuMCAgICAgICAgICAgICAgICAgIyBTaWdtYSBjbGlwcGluZyBsZXZlbCBmb3IgYmFja2dyb3VuZCBzdGF0aXN0aWNzDQoNCkRFVEVDVElPTl9USFJFU0hPTERfU0lHTUEgPSAzLjAgICMgRGV0ZWN0aW9uIHRocmVzaG9sZCBpbiB1bml0cyBvZiBiYWNrZ3JvdW5kIHNpZ21hDQoNClBFQUtfTUlOX1NURCA9IDMuMCAgICAgICAgICAgICAgICMgTWluaW11bSBwZWFrIHNpZ25hbC10by1ub2lzZSByYXRpbyBmb3IgYSBkZXRlY3Rpb24NCg0KQkFDS0dST1VORF9CT1hfU0laRSA9IDQ4ICAgICAgICAgIyBHcmlkIGNlbGwgZm9yIHRoZSBsb2NhbCBiYWNrZ3JvdW5kLCBwaXhlbHMNCg0KQkFDS0dST1VORF9GSUxURVJfU0laRSA9IDMgICAgICAgIyBNZWRpYW4gZmlsdGVyIGFwcGxpZWQgYWNyb3NzIHRoZSBncmlkDQoNClNFRUlOR19NRUFTVVJFRCA9IHt9DQoNCkJHX1NVQl9GVU5DID0gbnAubmFubWVkaWFuICAgICAgICMgRnVuY3Rpb24gdXNlZCB0byBjb21wdXRlIGFuZCBzdWJ0cmFjdCBiYWNrZ3JvdW5kIGxldmVsDQoNCk1BVENIX1NDQUxFID0gMS41ICAgICAgICAgICAgICAgICMgVG9sZXJhbmNlIGFzIGEgbXVsdGlwbGUgb2YgdGhlIG1lYXN1cmVkIEZXSE0NCg0KTUFUQ0hfVE9MRVJBTkNFX01JTiA9IDMuMCAgICAgICAgIyAuLi5idXQgbmV2ZXIgdGlnaHRlciB0aGFuIHRoaXMsIGluIHBpeGVscw0KDQpBU1RST0FMSUdOX01BWF9TQ0FMRV9FUlJPUiA9IDAuMDUgICMgNSUNCg0KUkVGX01BR19MSU1JVCA9IDIwLjAgICAgICAgICAgICAgIyBDYXRhbG9nIG1hZ25pdHVkZSBsaW1pdCBmb3Igc2VsZWN0aW5nIHJlZmVyZW5jZSBzdGFycw0KDQpSRUZfQ0FUQUxPRyA9ICJJSS8zMzYvYXBhc3M5IiAgICAjIFJlZmVyZW5jZSBzdGFyIGNhdGFsb2cgdG8gdXNlIChBUEFTUzkgd2l0aCBCLFYgbWFnbml0dWRlcykNCg0KQ0FUQUxPR19GTFVYX01BVENIX1NDQUxFID0gMS41ICAgIyBUb2xlcmFuY2UgYXMgYSBtdWx0aXBsZSBvZiB0aGUgbWVhc3VyZWQgRldITQ0KDQpDQVRBTE9HX0ZMVVhfTUFUQ0hfTUlOID0gMy4wICAgICAjIC4uLmJ1dCBuZXZlciB0aWdodGVyIHRoYW4gdGhpcywgaW4gcGl4ZWxzDQoNClJFRl9DUk9TU19CQU5EX1RPTEVSQU5DRSA9IDIuMCAgICMgQiBhbmQgViBtdXN0IGxhbmQgb24gdGhlIHNhbWUgb2JqZWN0LCB3aXRoaW4NCg0KUkVGX1pQX09VVExJRVJfU0lHTUEgPSAzLjAgICAgICAgIyBSZWplY3QgYSByZWZlcmVuY2Ugc3RhciB3aG9zZSBpbXBsaWVkIHplcm8NCg0KQ0FMSUJfTlVNX1NUQVJTID0gTm9uZSAgICAgICAgICAgIyBOdW1iZXIgb2YgcmVmZXJlbmNlIHN0YXJzIGZvciBjYWxpYnJhdGlvbiAodXNlciB3aWxsIGJlIGFza2VkKQ0KDQpDTURfQ09MT1JfTUlOID0gLTAuNSAgICMgbGVmdCBsaW1pdCBvZiB4LWF4aXMNCg0KQ01EX0NPTE9SX01BWCA9IDAuNSAgICAjIHJpZ2h0IGxpbWl0IG9mIHgtYXhpcw0KDQpSRU1PVkVfVE9MID0gMTAuMCAgIyBwaXhlbCB0b2xlcmFuY2UgZm9yIG1hdGNoaW5nIHNvdXJjZXMgdG8gY2F0YWxvZyBzdGFycw0KDQpGT1JFR1JPVU5EX1RPTCA9IDMuMA0KDQpTdGVwX3NpemUgPSAxMCAgICAjIFJlc29sdXRpb24gb2YgVGhlIEJsdWUgS25vdHMgRGVuc2l0eSBEaWFncmFtIA0KDQpkZWYgX25vcm1fcGF0aChwOiBzdHIpIC0+IHN0cjoNCiAgICAiIiJSZW1vdmUgZXh0cmEgcXVvdGVzIGFuZCBub3JtYWxpemUgZmlsZXN5c3RlbSBwYXRoLiIiIg0KICAgIHAgPSBwLnN0cmlwKCkuc3RyaXAoJyInKS5zdHJpcCgiJyIpDQogICAgcmV0dXJuIG9zLnBhdGgubm9ybXBhdGgocCkNCg0KZGVmIF9hc2socHJvbXB0LCBkZWZhdWx0LCBjYXN0PXN0cik6DQogICAgIiIiQXNrIHVzZXIgZm9yIGlucHV0IHdpdGggZGVmYXVsdCBhbmQgdHlwZSBjYXN0aW5nLiIiIg0KICAgIHMgPSBpbnB1dChmIntwcm9tcHR9IFt7ZGVmYXVsdH1dOiAiKS5zdHJpcCgpDQogICAgaWYgbm90IHM6DQogICAgICAgIHJldHVybiBkZWZhdWx0DQogICAgdHJ5Og0KICAgICAgICByZXR1cm4gY2FzdChzKQ0KICAgIGV4Y2VwdCBFeGNlcHRpb246DQogICAgICAgIHJldHVybiBkZWZhdWx0DQoNCmRlZiBzdWJ0cmFjdF9iYWNrZ3JvdW5kX2FuZF9zYXZlKHBhdGgpOg0KICAgICIiIlN1YnRyYWN0IGJhY2tncm91bmQgKG1lZGlhbiBvciBvdGhlciBmdW5jdGlvbikgZnJvbSBGSVRTIGFuZCBzYXZlIG5ldyBmaWxlLiIiIg0KICAgIGRhdGEsIGhkciA9IGZpdHMuZ2V0ZGF0YShwYXRoLCBoZWFkZXI9VHJ1ZSkNCiAgICBiZ192YWwgPSBCR19TVUJfRlVOQyhkYXRhKQ0KICAgIGRhdGFfc3ViID0gZGF0YSAtIGJnX3ZhbA0KICAgICMgaW50byByZXN1bHRfQ01ELCBub3QgbmV4dCB0byB0aGUgZnJhbWVzOiB0aGUgZm9sZGVyIGhvbGRpbmcgeW91ciBkYXRhDQogICAgIyBzaG91bGQgY29tZSBvdXQgb2YgYSBydW4gZXhhY3RseSBhcyBpdCB3ZW50IGluLg0KICAgIG91dF9kaXIgPSBvcy5wYXRoLmpvaW4ob3MucGF0aC5kaXJuYW1lKHBhdGgpIG9yIG9zLmdldGN3ZCgpLCAicmVzdWx0X0NNRCIpDQogICAgb3MubWFrZWRpcnMob3V0X2RpciwgZXhpc3Rfb2s9VHJ1ZSkNCiAgICBzdGVtID0gb3MucGF0aC5zcGxpdGV4dChvcy5wYXRoLmJhc2VuYW1lKHBhdGgpKVswXQ0KICAgIG91dF9wYXRoID0gb3MucGF0aC5qb2luKG91dF9kaXIsIHN0ZW0gKyAiX2Jnc3ViLmZpdHMiKQ0KICAgIGZpdHMud3JpdGV0byhvdXRfcGF0aCwgZGF0YV9zdWIsIGhkciwgb3ZlcndyaXRlPVRydWUpDQogICAgcHJpbnQoZiJbYmdzdWJdIHdyb3RlIHtvdXRfcGF0aH0gKGJnPXtiZ192YWw6LjNmfSkiKQ0KICAgIHJldHVybiBvdXRfcGF0aA0KDQpkZWYgY29tcHV0ZV9md2htKGRhdGEsIHgsIHksIHNpemU9RldITV9XSU5ET1dfU0laRSk6DQogICAgIiIiTWVhc3VyZSBGV0hNIGFyb3VuZCBhIGxpZ2h0IHNvdXJjZS4iIiINCiAgICB4X21pbiwgeF9tYXggPSBpbnQoeC1zaXplKSwgaW50KHgrc2l6ZSkNCiAgICB5X21pbiwgeV9tYXggPSBpbnQoeS1zaXplKSwgaW50KHkrc2l6ZSkNCiAgICBpZiB4X21pbiA8IDAgb3IgeV9taW4gPCAwIG9yIHhfbWF4ID49IGRhdGEuc2hhcGVbMV0gb3IgeV9tYXggPj0gZGF0YS5zaGFwZVswXToNCiAgICAgICAgIyBBIHNvdXJjZSB0b28gY2xvc2UgdG8gdGhlIGVkZ2UgZm9yIGEgZnVsbCBtZWFzdXJpbmcgYm94LiBUaGVyZSBhcmUNCiAgICAgICAgIyBodW5kcmVkcyBvZiB0aGVzZSBvbiBhIHR5cGljYWwgZnJhbWU7IHByaW50aW5nIG9uZSBsaW5lIGVhY2ggYnVyaWVkDQogICAgICAgICMgZXZlcnl0aGluZyBlbHNlLCBzbyB0aGV5IGFyZSBjb3VudGVkIGFuZCByZXBvcnRlZCBvbmNlIGF0IHRoZSBlbmQuDQogICAgICAgIHJldHVybiBOb25lDQoNCiAgICBzdWJfaW1hZ2UgPSBkYXRhW3lfbWluOnlfbWF4LCB4X21pbjp4X21heF0NCiAgICBzbW9vdGhlZCA9IGdhdXNzaWFuX2ZpbHRlcihzdWJfaW1hZ2UsIHNpZ21hPTIpDQogICAgcGVhayA9IG5wLm1heChzbW9vdGhlZCkNCiAgICBoYWxmX21heCA9IHBlYWsgLyAyDQogICAgYWJvdmVfaGFsZl9tYXggPSBzbW9vdGhlZCA+IGhhbGZfbWF4DQogICAgaW5kaWNlcyA9IG5wLmFyZ3doZXJlKGFib3ZlX2hhbGZfbWF4KQ0KICAgIGlmIGluZGljZXMuc2l6ZSA+IDA6DQogICAgICAgIG1pbl94LCBtYXhfeCA9IGluZGljZXNbOiwgMV0ubWluKCksIGluZGljZXNbOiwgMV0ubWF4KCkNCiAgICAgICAgbWluX3ksIG1heF95ID0gaW5kaWNlc1s6LCAwXS5taW4oKSwgaW5kaWNlc1s6LCAwXS5tYXgoKQ0KICAgICAgICBmd2htX3ggPSBtYXhfeCAtIG1pbl94DQogICAgICAgIGZ3aG1feSA9IG1heF95IC0gbWluX3kNCiAgICAgICAgcmV0dXJuIG5wLm1lYW4oW2Z3aG1feCwgZndobV95XSkNCiAgICByZXR1cm4gTm9uZQ0KDQpkZWYgbG9jYWxfYmFja2dyb3VuZChkYXRhKToNCiAgICAiIiJCYWNrZ3JvdW5kIGFuZCBub2lzZSBvbiBhIGNvYXJzZSBncmlkLCBzbyBhIGdhbGF4eSdzIG93biBsaWdodCBkb2VzIG5vdA0KICAgIHNldCB0aGUgZGV0ZWN0aW9uIHRocmVzaG9sZCBmb3IgdGhlIGVtcHR5IHNreSBhcm91bmQgaXQuIiIiDQogICAgcmV0dXJuIEJhY2tncm91bmQyRChkYXRhLCAoQkFDS0dST1VORF9CT1hfU0laRSwgQkFDS0dST1VORF9CT1hfU0laRSksDQogICAgICAgICAgICAgICAgICAgICAgICBmaWx0ZXJfc2l6ZT0oQkFDS0dST1VORF9GSUxURVJfU0laRSwgQkFDS0dST1VORF9GSUxURVJfU0laRSksDQogICAgICAgICAgICAgICAgICAgICAgICBzaWdtYV9jbGlwPVNpZ21hQ2xpcChzaWdtYT1TSUdNQV9DTElQKSwNCiAgICAgICAgICAgICAgICAgICAgICAgIGJrZ19lc3RpbWF0b3I9TWVkaWFuQmFja2dyb3VuZCgpKQ0KDQpkZWYgc3Rhcl93aWR0aChkYXRhLCB4LCB5LCBoYWxmPTEyKToNCiAgICAiIiJXaWR0aCBhdCBoYWxmIG1heGltdW0gb2Ygb25lIHNvdXJjZSwgaW4gcGl4ZWxzLCBtZWFzdXJlZCBvbiB0aGUgZGF0YSBhcw0KICAgIGl0IGlzLiBjb21wdXRlX2Z3aG0oKSBzbW9vdGhzIGZpcnN0LCB3aGljaCBzdWl0cyBhcGVydHVyZSBzaXppbmcgYnV0IGFkZHMNCiAgICBhYm91dCAzIHBpeGVsczsgYSBkZXRlY3Rpb24ga2VybmVsIG5lZWRzIHRoZSB0cnVlIHdpZHRoLiIiIg0KICAgIHgsIHkgPSBpbnQocm91bmQoeCkpLCBpbnQocm91bmQoeSkpDQogICAgaWYgeCAtIGhhbGYgPCAwIG9yIHkgLSBoYWxmIDwgMCBvciB4ICsgaGFsZiA+PSBkYXRhLnNoYXBlWzFdIG9yIHkgKyBoYWxmID49IGRhdGEuc2hhcGVbMF06DQogICAgICAgIHJldHVybiBucC5uYW4NCiAgICBjdXQgPSBkYXRhW3kgLSBoYWxmOnkgKyBoYWxmLCB4IC0gaGFsZjp4ICsgaGFsZl0NCiAgICBwZWFrID0gY3V0Lm1heCgpDQogICAgaWYgbm90IG5wLmlzZmluaXRlKHBlYWspIG9yIHBlYWsgPD0gMDoNCiAgICAgICAgcmV0dXJuIG5wLm5hbg0KICAgIGlkeCA9IG5wLmFyZ3doZXJlKGN1dCA+IHBlYWsgLyAyKQ0KICAgIGlmIG5vdCBsZW4oaWR4KToNCiAgICAgICAgcmV0dXJuIG5wLm5hbg0KICAgIHJldHVybiBucC5tZWFuKFtucC5wdHAoaWR4WzosIDFdKSArIDEsIG5wLnB0cChpZHhbOiwgMF0pICsgMV0pDQoNCmRlZiBlc3RpbWF0ZV9zZWVpbmcoZGF0YSwgYmFja2dyb3VuZCwgcm1zLCBiYW5kKToNCiAgICAiIiJNZWFzdXJlIGhvdyB3aWRlIGEgc3RhciBhY3R1YWxseSBpcyBpbiB0aGlzIGZyYW1lLg0KDQogICAgRGV0ZWN0IG9uY2UgYXQgYSBoaWdoIHRocmVzaG9sZCB0byBnZXQgYnJpZ2h0LCB1bmFtYmlndW91cyBzdGFycywgbWVhc3VyZQ0KICAgIGVhY2ggb25lLCBhbmQgdGFrZSB0aGUgbWVkaWFuLiBOb3RoaW5nIGFib3V0IHRoZSB0ZWxlc2NvcGUgaXMgYXNzdW1lZCBhbmQNCiAgICBubyBoZWFkZXIga2V5d29yZCBpcyByZXF1aXJlZCwgc28gdGhpcyB3b3JrcyBvbiBhbnkgZnJhbWUuDQogICAgIiIiDQogICAgdHJ5Og0KICAgICAgICBmb3VuZCA9IERBT1N0YXJGaW5kZXIoZndobT1TRUVJTkdfRkFMTEJBQ0ssIHRocmVzaG9sZD0yMC4wICogcm1zKShkYXRhIC0gYmFja2dyb3VuZCkNCiAgICAgICAgaWYgZm91bmQgaXMgbm90IE5vbmUgYW5kIGxlbihmb3VuZCkgPj0gNToNCiAgICAgICAgICAgIGJyaWdodGVzdCA9IGZvdW5kW25wLmFyZ3NvcnQoZm91bmRbJ2ZsdXgnXSldWy0yMDA6XQ0KICAgICAgICAgICAgd2lkdGhzID0gW3N0YXJfd2lkdGgoZGF0YSAtIGJhY2tncm91bmQsIHNbJ3hjZW50cm9pZCddLCBzWyd5Y2VudHJvaWQnXSkNCiAgICAgICAgICAgICAgICAgICAgICBmb3IgcyBpbiBicmlnaHRlc3RdDQogICAgICAgICAgICB3aWR0aHMgPSBucC5hc2FycmF5KHdpZHRocywgZHR5cGU9ZmxvYXQpDQogICAgICAgICAgICBzZWVpbmcgPSBucC5uYW5tZWRpYW4od2lkdGhzKQ0KICAgICAgICAgICAgaWYgbnAuaXNmaW5pdGUoc2VlaW5nKSBhbmQgU0VFSU5HX01JTiA8PSBzZWVpbmcgPD0gU0VFSU5HX01BWDoNCiAgICAgICAgICAgICAgICBwcmludChmIiAgIHtiYW5kfTogc3RhcnMgYXJlIHtzZWVpbmc6LjJmfSBweCB3aWRlIg0KICAgICAgICAgICAgICAgICAgICAgIGYiIChtZWFzdXJlZCBvbiB7aW50KG5wLmlzZmluaXRlKHdpZHRocykuc3VtKCkpfSBicmlnaHQgc3RhcnMpIiwgZmx1c2g9VHJ1ZSkNCiAgICAgICAgICAgICAgICByZXR1cm4gZmxvYXQoc2VlaW5nKQ0KICAgICAgICBwcmludChmIiAgIHtiYW5kfTogY291bGQgbm90IG1lYXN1cmUgdGhlIHN0YXIgd2lkdGgsIHVzaW5nIg0KICAgICAgICAgICAgICBmIiB7U0VFSU5HX0ZBTExCQUNLfSBweCIsIGZsdXNoPVRydWUpDQogICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBleGM6DQogICAgICAgIHByaW50KGYiICAge2JhbmR9OiBzdGFyIHdpZHRoIG1lYXN1cmVtZW50IGZhaWxlZCAoe2V4Y30pLCB1c2luZyINCiAgICAgICAgICAgICAgZiIge1NFRUlOR19GQUxMQkFDS30gcHgiLCBmbHVzaD1UcnVlKQ0KICAgIHJldHVybiBmbG9hdChTRUVJTkdfRkFMTEJBQ0spDQoNCmRlZiBwcm9jZXNzX2ZpdHMoZmlsZW5hbWUsIGJhbmQsIHdpbmRvdz1Ob25lKToNCiAgICAiIiJEZXRlY3Qgc291cmNlcywgcGVyZm9ybSBhcGVydHVyZSBwaG90b21ldHJ5IGFuZCByZXR1cm4gcmVzdWx0cy4NCg0KICAgIGB3aW5kb3dgIGlzICh5MCwgeTEsIHgwLCB4MSk6IHRoZSBwYXJ0IG9mIHRoZSBmcmFtZSB0aGUgZ2FsYXh5IG9jY3VwaWVzLiBCb3RoDQogICAgb2YgdGhlIGV4cGVuc2l2ZSBzdGVwcyBoZXJlIGFyZSBwYWlkIGJ5IHRoZSBwaXhlbCBhbmQgYnkgdGhlIHNvdXJjZSAtIGZpbmRpbmcNCiAgICBzb3VyY2VzIHJ1bnMgYSBkZXRlY3RvciBhY3Jvc3MgZXZlcnkgcGl4ZWwsIGFuZCBtZWFzdXJpbmcgdGhlbSBpcyBhIGxvb3Agd2l0aA0KICAgIG9uZSB0dXJuIHBlciBzb3VyY2UgLSBzbyB3b3JraW5nIG91dHNpZGUgdGhlIGdhbGF4eSBjb3N0cyB0d2ljZSBhbmQgcmV0dXJucw0KICAgIG5vdGhpbmcuIE9uIGEgNjEtbWVnYXBpeGVsIGZpZWxkIG9mIE0zMyBvbmx5IGFib3V0IGEgdGhpcmQgb2YgdGhlIGZyYW1lIGhvbGRzDQogICAgdGhlIGdhbGF4eSBhdCBhbGwuDQoNCiAgICBQb3NpdGlvbnMgY29tZSBiYWNrIGluIHRoZSBjb29yZGluYXRlcyBvZiB0aGUgd2hvbGUgZnJhbWUgcmVnYXJkbGVzcywgc28NCiAgICBub3RoaW5nIGRvd25zdHJlYW0gbmVlZHMgdG8ga25vdyBhIHdpbmRvdyB3YXMgdXNlZC4NCiAgICAiIiINCiAgICBoZHVsID0gZml0cy5vcGVuKGZpbGVuYW1lKQ0KICAgIGRhdGEgPSBoZHVsWzBdLmRhdGENCiAgICBoZHVsLmNsb3NlKCkNCg0KICAgIHhfb2Zmc2V0ID0geV9vZmZzZXQgPSAwDQogICAgaWYgd2luZG93IGlzIG5vdCBOb25lOg0KICAgICAgICB5MCwgeTEsIHgwLCB4MSA9IHdpbmRvdw0KICAgICAgICBkYXRhID0gZGF0YVt5MDp5MSwgeDA6eDFdDQogICAgICAgIHhfb2Zmc2V0LCB5X29mZnNldCA9IHgwLCB5MA0KICAgICAgICBwcmludChmIiAgIHtiYW5kfTogd29ya2luZyBvbiB7eDEgLSB4MH0geCB7eTEgLSB5MH0gcGl4ZWxzIGFyb3VuZCB0aGUgIg0KICAgICAgICAgICAgICBmImdhbGF4eSIsIGZsdXNoPVRydWUpDQoNCiAgICBwcmludChmIkZpbmRpbmcgc291cmNlcyBpbiB0aGUge2JhbmR9IGltYWdlLi4uIiwgZmx1c2g9VHJ1ZSkNCiAgICBia2cgPSBsb2NhbF9iYWNrZ3JvdW5kKGRhdGEpDQogICAgcm1zID0gZmxvYXQobnAubWVkaWFuKGJrZy5iYWNrZ3JvdW5kX3JtcykpDQogICAgcHJpbnQoZiIgICB7YmFuZH06IGxvY2FsIHNreSBydW5zIGZyb20ge2JrZy5iYWNrZ3JvdW5kLm1pbigpOi4xZn0iDQogICAgICAgICAgZiIgdG8ge2JrZy5iYWNrZ3JvdW5kLm1heCgpOi4xZn0gY291bnRzLCBub2lzZSB7cm1zOi4yZn0iLCBmbHVzaD1UcnVlKQ0KDQogICAgZndobV9kZXRlY3QgPSBEQU9GSU5EX0ZXSE0gaWYgREFPRklORF9GV0hNIGVsc2UgZXN0aW1hdGVfc2VlaW5nKA0KICAgICAgICBkYXRhLCBia2cuYmFja2dyb3VuZCwgcm1zLCBiYW5kKQ0KICAgIFNFRUlOR19NRUFTVVJFRFtiYW5kXSA9IGZ3aG1fZGV0ZWN0DQoNCiAgICB0aHJlc2hvbGQgPSBERVRFQ1RJT05fVEhSRVNIT0xEX1NJR01BICogcm1zDQogICAgZGFvZmluZCA9IERBT1N0YXJGaW5kZXIoZndobT1md2htX2RldGVjdCwgdGhyZXNob2xkPXRocmVzaG9sZCkNCiAgICBzb3VyY2VzID0gZGFvZmluZChkYXRhIC0gYmtnLmJhY2tncm91bmQpDQogICAgaWYgc291cmNlcyBpcyBOb25lIG9yIGxlbihzb3VyY2VzKSA9PSAwOg0KICAgICAgICBwcmludChmIiAgIHtiYW5kfTogbm8gc291cmNlcyBmb3VuZCIsIGZsdXNoPVRydWUpDQogICAgICAgIHJldHVybiBbXQ0KICAgIHNvdXJjZXMgPSBzb3VyY2VzW3NvdXJjZXNbJ3BlYWsnXSA+IFBFQUtfTUlOX1NURCAqIHJtc10NCg0KICAgICMgVGhlIGZyYW1lIHdpdGggdGhlIHNtb290aCBnYWxheHkgbGlnaHQgcmVtb3ZlZCwgdXNlZCBmb3IgbWVhc3VyaW5nIHdpZHRocy4NCiAgICBmbGF0ID0gZGF0YSAtIGJrZy5iYWNrZ3JvdW5kDQoNCiAgICAjIE1lYXN1cmluZyBldmVyeSBzb3VyY2UgdGFrZXMgdGhlIGxvbmdlc3Qgb2YgYW55IHN0ZXAgaGVyZSBhbmQgdXNlZCB0byBydW4NCiAgICAjIGluIHNpbGVuY2UuIFJlcG9ydCBhYm91dCB0ZW4gdGltZXMsIGVhY2ggb24gaXRzIG93biBsaW5lOiBTcHlkZXIncyBjb25zb2xlDQogICAgIyBkb2VzIG5vdCByZWxpYWJseSBob25vdXIgYSBjYXJyaWFnZSByZXR1cm4sIGFuZCBhIHJld3JpdHRlbiBsaW5lIHRoZW4NCiAgICAjIGJlY29tZXMgb25lIHZlcnkgbG9uZyBvbmUuDQogICAgbl90b3RhbCA9IGxlbihzb3VyY2VzKQ0KICAgIHN0ZXAgPSBtYXgoMSwgbl90b3RhbCAvLyAxMCkNCiAgICBwcmludChmIk1lYXN1cmluZyB7bl90b3RhbDosfSBzb3VyY2VzIGluIHRoZSB7YmFuZH0gaW1hZ2UuLi4iLCBmbHVzaD1UcnVlKQ0KDQogICAgbl9lZGdlID0gbl9ub3NpZ25hbCA9IG5fbmVnYXRpdmUgPSAwDQogICAgcmVzdWx0cyA9IFtdDQogICAgZm9yIGksIHNvdXJjZSBpbiBlbnVtZXJhdGUoc291cmNlcywgMSk6DQogICAgICAgIGlmIGkgJSBzdGVwID09IDAgb3IgaSA9PSBuX3RvdGFsOg0KICAgICAgICAgICAgcHJpbnQoZiIgICB7aTosfSBvZiB7bl90b3RhbDosfSIsIGZsdXNoPVRydWUpDQogICAgICAgIHgsIHkgPSBzb3VyY2VbJ3hjZW50cm9pZCddLCBzb3VyY2VbJ3ljZW50cm9pZCddDQogICAgICAgICMgV2lkdGggaXMgbWVhc3VyZWQgb24gdGhlIGZyYW1lIHdpdGggdGhlIGRpZmZ1c2UgbGlnaHQgdGFrZW4gb3V0LCBhbmQNCiAgICAgICAgIyB0aGUgZmx1eCBvbiB0aGUgZnJhbWUgYXMgaXQgaXMuDQogICAgICAgICMNCiAgICAgICAgIyBjb21wdXRlX2Z3aG0gYXNrcyB3aGljaCBwaXhlbHMgc2l0IGFib3ZlIGhhbGYgb2YgdGhlIGJyaWdodGVzdCBvbmUuIE9uDQogICAgICAgICMgYSBnYWxheHkgdGhhdCBxdWVzdGlvbiBoYXMgbm8gdXNlZnVsIGFuc3dlcjogaW5zaWRlIGEgYm94IG9uIHRoZSBkaXNjDQogICAgICAgICMgdGhlIGNvdW50cyBydW4gZnJvbSBhYm91dCA1MCB0byA2MywgaGFsZiBvZiB0aGUgcGVhayBpcyAzMSwgYW5kIG5vdCBvbmUNCiAgICAgICAgIyBwaXhlbCBmYWxscyBiZWxvdyBpdCAtIHNvIGV2ZXJ5IHNvdXJjZSBjYW1lIGJhY2sgYXMgd2lkZSBhcyB0aGUgYm94LA0KICAgICAgICAjIGFuZCBpdHMgYXBlcnR1cmUgd2FzIHNpemVkIGJ5IHRoZSBib3ggcmF0aGVyIHRoYW4gYnkgaXRzZWxmLiBUYWtpbmcgdGhlDQogICAgICAgICMgc21vb3RoIGdhbGF4eSBsaWdodCBvdXQgZmlyc3QgcHV0cyB0aGUgZmxvb3IgYmFjayBhdCB6ZXJvLCBhbmQgaGFsZiB0aGUNCiAgICAgICAgIyBwZWFrIGJlY29tZXMgYSB0aHJlc2hvbGQgdGhhdCBzZXBhcmF0ZXMgdGhlIHNvdXJjZSBmcm9tIHdoYXQgc3Vycm91bmRzDQogICAgICAgICMgaXQuIEEgc3RhciBvbiBlbXB0eSBza3kgaXMgdW5hZmZlY3RlZDogaXRzIGZsb29yIHdhcyBhbHJlYWR5IHplcm8uDQogICAgICAgIGZ3aG0gPSBjb21wdXRlX2Z3aG0oZmxhdCwgeCwgeSkNCiAgICAgICAgIyBmd2htIGlzIE5vbmUgICAgICAgLSB0b28gY2xvc2UgdG8gdGhlIGVkZ2UgZm9yIGEgZnVsbCBtZWFzdXJpbmcgYm94DQogICAgICAgICMgZndobSBpcyB6ZXJvL05hTiAgIC0gYSBkZXRlY3Rpb24gdGhhdCB0dXJuZWQgb3V0IHRvIGJlIG5vdGhpbmcuIE9uIGENCiAgICAgICAgIyAgICAgICAgICAgICAgICAgICAgICBiYWNrZ3JvdW5kLXN1YnRyYWN0ZWQgaW1hZ2UgdGhlIGxvY2FsIHBlYWsgb2YgYQ0KICAgICAgICAjICAgICAgICAgICAgICAgICAgICAgIG5vaXNlIHJpcHBsZSBjYW4gYmUgMC4wMDYgY291bnRzOyBoYWxmIG9mIHRoYXQgaXMNCiAgICAgICAgIyAgICAgICAgICAgICAgICAgICAgICBhIG1lYW5pbmdsZXNzIHRocmVzaG9sZCwgb25lIHBpeGVsIGNsZWFycyBpdCwgYW5kDQogICAgICAgICMgICAgICAgICAgICAgICAgICAgICAgdGhlIHdpZHRoIGJldHdlZW4gdGhhdCBwaXhlbCBhbmQgaXRzZWxmIGlzIHplcm8uDQogICAgICAgICMgICAgICAgICAgICAgICAgICAgICAgUGFzc2luZyBpdCBvbiBnaXZlcyBwaG90dXRpbHMgYSByYWRpdXMgb2YgemVybw0KICAgICAgICAjICAgICAgICAgICAgICAgICAgICAgIGFuZCB0aGUgcnVuIHN0b3BzLg0KICAgICAgICBpZiBmd2htIGlzIE5vbmU6DQogICAgICAgICAgICBuX2VkZ2UgKz0gMQ0KICAgICAgICAgICAgY29udGludWUNCiAgICAgICAgaWYgbm90IChucC5pc2Zpbml0ZShmd2htKSBhbmQgZndobSA+IDApOg0KICAgICAgICAgICAgbl9ub3NpZ25hbCArPSAxDQogICAgICAgICAgICBjb250aW51ZQ0KDQogICAgICAgIHJhZGl1cyA9IEFQRVJUVVJFX1NDQUxFICogZndobQ0KDQogICAgICAgICMgQSBjaGVhcCBsb29rIGJlZm9yZSB0aGUgZXhwZW5zaXZlIG9uZS4gTW9yZSB0aGFuIGhhbGYgb2Ygd2hhdCB0aGUNCiAgICAgICAgIyBkZXRlY3RvciBoYW5kcyBvdmVyIGhlcmUgaXMgbm9pc2U6IG9uIGFuIE0zMyBmcmFtZSwgMjAsNzI2IG9mIDM5LDQ5MQ0KICAgICAgICAjIHNvdXJjZXMgZW5kZWQgdXAgd2l0aCBuZWdhdGl2ZSBmbHV4IGFuZCB3ZXJlIHRocm93biBhd2F5IC0gYWZ0ZXIgYmVpbmcNCiAgICAgICAgIyBtZWFzdXJlZC4gQXBlcnR1cmUgcGhvdG9tZXRyeSBvbiBhIHNpbmdsZSBzb3VyY2UgY29zdHMgYSB0aG91c2FuZCB0aW1lcw0KICAgICAgICAjIHdoYXQgc3VtbWluZyBhIHNtYWxsIHBhdGNoIG9mIHRoZSBhcnJheSBkb2VzLCBzbyB0aGUgaG9wZWxlc3MgY2FzZXMgYXJlDQogICAgICAgICMgZHJvcHBlZCBmaXJzdC4NCiAgICAgICAgIw0KICAgICAgICAjIFRoZSB0ZXN0IGlzIGRlbGliZXJhdGVseSB3ZWFrZXIgdGhhbiB0aGUgcmVhbCBvbmUuIEl0IHJlamVjdHMgb25seSB3aGF0DQogICAgICAgICMgdGhlIHJlYWwgdGVzdCB3b3VsZCBjZXJ0YWlubHkgcmVqZWN0IHRvbzogYSBwYXRjaCB3aG9zZSB0b3RhbCBzaXRzIHdlbGwNCiAgICAgICAgIyBiZWxvdyB6ZXJvIG9uIGEgZnJhbWUgdGhhdCBhbHJlYWR5IGhhcyBpdHMgYmFja2dyb3VuZCByZW1vdmVkLiBBbnl0aGluZw0KICAgICAgICAjIG5lYXIgdGhlIGxpbmUgZ29lcyB0aHJvdWdoIGFuZCBpcyBtZWFzdXJlZCBwcm9wZXJseSwgc28gdGhlIGNhdGFsb2d1ZQ0KICAgICAgICAjIHRoYXQgY29tZXMgb3V0IGlzIHRoZSBzYW1lIGNhdGFsb2d1ZS4NCiAgICAgICAgX2ggPSBpbnQobWF4KDMsIHJvdW5kKHJhZGl1cykpKQ0KICAgICAgICBfeWksIF94aSA9IGludChyb3VuZCh5KSksIGludChyb3VuZCh4KSkNCiAgICAgICAgaWYgKF9oIDw9IF95aSA8IGZsYXQuc2hhcGVbMF0gLSBfaCkgYW5kIChfaCA8PSBfeGkgPCBmbGF0LnNoYXBlWzFdIC0gX2gpOg0KICAgICAgICAgICAgX3BhdGNoID0gZmxhdFtfeWkgLSBfaDpfeWkgKyBfaCArIDEsIF94aSAtIF9oOl94aSArIF9oICsgMV0NCiAgICAgICAgICAgIGlmIF9wYXRjaC5zdW0oKSA8IC0zLjAgKiBybXMgKiBfcGF0Y2guc2l6ZSAqKiAwLjU6DQogICAgICAgICAgICAgICAgbl9uZWdhdGl2ZSArPSAxDQogICAgICAgICAgICAgICAgY29udGludWUNCg0KICAgICAgICBhcGVydHVyZSA9IENpcmN1bGFyQXBlcnR1cmUoKHgsIHkpLCByPXJhZGl1cykNCiAgICAgICAgYW5udWx1c19pbm5lcl9yYWRpdXMgPSByYWRpdXMgKiBBTk5VTFVTX0lOTkVSX1NDQUxFDQogICAgICAgIGFubnVsdXNfb3V0ZXJfcmFkaXVzID0gcmFkaXVzICogQU5OVUxVU19PVVRFUl9TQ0FMRQ0KICAgICAgICBhbm51bHVzID0gQ2lyY3VsYXJBbm51bHVzKCh4LCB5KSwgcl9pbj1hbm51bHVzX2lubmVyX3JhZGl1cywgcl9vdXQ9YW5udWx1c19vdXRlcl9yYWRpdXMpDQoNCiAgICAgICAgcGhvdF90YWJsZSA9IGFwZXJ0dXJlX3Bob3RvbWV0cnkoZGF0YSwgW2FwZXJ0dXJlLCBhbm51bHVzXSkNCiAgICAgICAgYmFja2dyb3VuZF9tZWFuID0gcGhvdF90YWJsZVsnYXBlcnR1cmVfc3VtXzEnXVswXSAvIGFubnVsdXMuYXJlYQ0KICAgICAgICBiYWNrZ3JvdW5kX3N1YnRyYWN0ZWRfZmx1eCA9IHBob3RfdGFibGVbJ2FwZXJ0dXJlX3N1bV8wJ11bMF0gLSBiYWNrZ3JvdW5kX21lYW4gKiBhcGVydHVyZS5hcmVhDQoNCiAgICAgICAgaWYgYmFja2dyb3VuZF9zdWJ0cmFjdGVkX2ZsdXggPCAwOg0KICAgICAgICAgICAgbl9uZWdhdGl2ZSArPSAxDQogICAgICAgICAgICBjb250aW51ZQ0KDQogICAgICAgIHJlc3VsdHMuYXBwZW5kKFt4LCB5LCBmd2htLCByYWRpdXMsIGJhY2tncm91bmRfc3VidHJhY3RlZF9mbHV4LA0KICAgICAgICAgICAgICAgICAgICAgICAgYmFuZCwgYW5udWx1c19pbm5lcl9yYWRpdXMsIGFubnVsdXNfb3V0ZXJfcmFkaXVzXSkNCg0KICAgIHByaW50KGYiICAge2xlbihyZXN1bHRzKTosfSBtZWFzdXJlZCINCiAgICAgICAgICBmIiAgIHwgICBza2lwcGVkOiB7bl9lZGdlOix9IGF0IHRoZSBmcmFtZSBlZGdlLCINCiAgICAgICAgICBmIiB7bl9ub3NpZ25hbDosfSB3aXRoIG5vIG1lYXN1cmFibGUgd2lkdGgsIg0KICAgICAgICAgIGYiIHtuX25lZ2F0aXZlOix9IHdpdGggbmVnYXRpdmUgZmx1eCIsIGZsdXNoPVRydWUpDQoNCiAgICAjIEJhY2sgaW50byB0aGUgY29vcmRpbmF0ZXMgb2YgdGhlIHdob2xlIGZyYW1lLiBFdmVyeXRoaW5nIGFmdGVyIHRoaXMgcG9pbnQgLQ0KICAgICMgcmVnaXN0ZXJpbmcgQiBvbnRvIFYsIG1hdGNoaW5nIHRoZSBjYXRhbG9ndWUgdGhyb3VnaCB0aGUgV0NTLCB0aGUgcmFkaWFsDQogICAgIyBwcm9maWxlIC0gaXMgd3JpdHRlbiBpbiB0aG9zZSwgYW5kIGEgd2luZG93IHRoYXQgbGVha2VkIGl0cyBvd24gY29vcmRpbmF0ZXMNCiAgICAjIG91dHdhcmRzIHdvdWxkIHNoaWZ0IGV2ZXJ5IHBvc2l0aW9uIGJ5IHRoZSBzaXplIG9mIHRoZSBjcm9wLg0KICAgIGlmIHhfb2Zmc2V0IG9yIHlfb2Zmc2V0Og0KICAgICAgICBmb3Igcm93IGluIHJlc3VsdHM6DQogICAgICAgICAgICByb3dbMF0gKz0geF9vZmZzZXQNCiAgICAgICAgICAgIHJvd1sxXSArPSB5X29mZnNldA0KDQogICAgcmV0dXJuIHJlc3VsdHMNCg0KZGVmIHJlZ2lzdGVyX0Jfb250b19WKEJfaW1nLCBWX2ltZywgeEIsIHlCLCBmaXRzX2ZpbGVfQiwgZml0c19maWxlX1YpOg0KICAgICIiIldoZXJlIGVhY2ggQi1pbWFnZSBwb3NpdGlvbiAoeEIsIHlCKSBsYW5kcyBpbiBWJ3MgcGl4ZWwgZ3JpZC4NCg0KICAgIEIgYW5kIFYgYXJlIHNlcGFyYXRlIGV4cG9zdXJlcywgYW5kIG5vdGhpbmcgZ3VhcmFudGVlcyB0aGVpciBwaXhlbCBncmlkcw0KICAgIGNvaW5jaWRlIC0gdGhlIHRlbGVzY29wZSBjYW4gbW92ZSBiZXR3ZWVuIHRoZW0gYnkgYW4gYW1vdW50IHRoYXQgZGVwZW5kcw0KICAgIG9uIHRoZSBtb3VudCwgdGhlIGRpdGhlciwgaG93IGxvbmcgdGhlIGZpbHRlciBjaGFuZ2UgdG9vay4gRXZlcnkgbWF0Y2gNCiAgICB0b2xlcmFuY2UgaW4gdGhpcyBmaWxlIGlzIG9ubHkgY29ycmVjdCBvbmNlIHRoYXQgaXMgY29ycmVjdGVkIGZvci4NCg0KICAgIGFzdHJvYWxpZ24gZmluZHMgdGhlIHJlZ2lzdHJhdGlvbiBkaXJlY3RseSBmcm9tIHRoZSBzdGFyIHBhdHRlcm5zIGluDQogICAgdGhlc2UgdHdvIHNwZWNpZmljIGltYWdlcyAtIGl0IGRvZXMgbm90IGFzc3VtZSBhbnkgdGVsZXNjb3BlLCBmaWVsZCwgb3INCiAgICBwbGF0ZSBzY2FsZSwgYW5kIGRvZXMgbm90IGRlcGVuZCBvbiBob3cgZ29vZCBlaXRoZXIgV0NTIHNvbHV0aW9uIGlzLg0KICAgIEZhbGxzIGJhY2sgdG8gZWFjaCBmcmFtZSdzIG93biBXQ1Mgb25seSBpZiBhc3Ryb2FsaWduIGNhbm5vdCBmaW5kIGVub3VnaA0KICAgIG1hdGNoaW5nIHN0YXJzLCBvciBpdHMgZml0dGVkIHNjYWxlIGlzIG5vdCBwbGF1c2libGUgZm9yIHR3byBmcmFtZXMgZnJvbQ0KICAgIHRoZSBzYW1lIHRlbGVzY29wZS4iIiINCiAgICB0cnk6DQogICAgICAgIHRyYW5zZiwgKHNyY19saXN0LCBfdGd0X2xpc3QpID0gYWEuZmluZF90cmFuc2Zvcm0oQl9pbWcsIFZfaW1nKQ0KICAgICAgICBpZiBhYnModHJhbnNmLnNjYWxlIC0gMS4wKSA+IEFTVFJPQUxJR05fTUFYX1NDQUxFX0VSUk9SOg0KICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmImZpdHRlZCBzY2FsZSB7dHJhbnNmLnNjYWxlOi4zZn0gaXMgbm90IHBsYXVzaWJsZSAiDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgIGYiZm9yIHR3byBmcmFtZXMgZnJvbSB0aGUgc2FtZSB0ZWxlc2NvcGUiKQ0KICAgICAgICB4eSA9IHRyYW5zZihucC5jb2x1bW5fc3RhY2soW3hCLCB5Ql0pKQ0KICAgICAgICBwcmludChmIiAgIHJlZ2lzdGVyZWQgQiBvbnRvIFYgZnJvbSB7bGVuKHNyY19saXN0KX0gbWF0Y2hlZCBzdGFyICINCiAgICAgICAgICAgICBmInBhaXJzIChhc3Ryb2FsaWduKTogc2hpZnQgKHt0cmFuc2YudHJhbnNsYXRpb25bMF06Ky4xZn0sICINCiAgICAgICAgICAgICBmInt0cmFuc2YudHJhbnNsYXRpb25bMV06Ky4xZn0pIHB4LCByb3RhdGlvbiAiDQogICAgICAgICAgICAgZiJ7bnAuZGVncmVlcyh0cmFuc2Yucm90YXRpb24pOisuMmZ9IGRlZywgc2NhbGUge3RyYW5zZi5zY2FsZTouNGZ9IiwNCiAgICAgICAgICAgICBmbHVzaD1UcnVlKQ0KICAgICAgICByZXR1cm4geHlbOiwgMF0sIHh5WzosIDFdDQogICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOg0KICAgICAgICBwcmludChmIiAgIGFzdHJvYWxpZ24gcmVnaXN0cmF0aW9uIGZhaWxlZCAoe2V9KSAtICINCiAgICAgICAgICAgICBmImZhbGxpbmcgYmFjayB0byBlYWNoIGZyYW1lJ3Mgb3duIFdDUyIsIGZsdXNoPVRydWUpDQogICAgICAgIHdjc19CID0gV0NTKGZpdHMuZ2V0aGVhZGVyKGZpdHNfZmlsZV9CKSkNCiAgICAgICAgd2NzX1YgPSBXQ1MoZml0cy5nZXRoZWFkZXIoZml0c19maWxlX1YpKQ0KICAgICAgICByYV9CLCBkZWNfQiA9IHdjc19CLmFsbF9waXgyd29ybGQoeEIsIHlCLCAwKQ0KICAgICAgICByZXR1cm4gd2NzX1YuYWxsX3dvcmxkMnBpeChyYV9CLCBkZWNfQiwgMCkNCg0KZGVmIG1hdGNoX3NvdXJjZXMoZGZfQiwgZGZfViwgdG9sLCB4Ql9yZWc9Tm9uZSwgeUJfcmVnPU5vbmUpOg0KICAgICIiIk1hdGNoIEIgYW5kIFYgc291cmNlcyBieSBuZWFyZXN0IChYLFkpIHdpdGhpbiB0b2xlcmFuY2UuDQoNCiAgICB4Ql9yZWcveUJfcmVnLCB3aGVuIGdpdmVuLCBhcmUgZWFjaCBCIHNvdXJjZSdzIHBvc2l0aW9uIGFmdGVyDQogICAgcmVnaXN0ZXJfQl9vbnRvX1YoKSAtIHdoYXQgdGhlIG1hdGNoaW5nIGRpc3RhbmNlIGlzIG1lYXN1cmVkIG9uLiBkZl9CJ3MNCiAgICBvd24gWC9ZIGFyZSBzdGlsbCB3aGF0IGdldHMgc3RvcmVkIGluIHRoZSBvdXRwdXQsIHNpbmNlIGFwZXJ0dXJlDQogICAgcGhvdG9tZXRyeSBvbiB0aGUgQiBpbWFnZSBuZWVkcyBCJ3Mgb3duIHBpeGVsIGNvb3JkaW5hdGVzLCBub3QgVidzLiIiIg0KICAgIGlmIHhCX3JlZyBpcyBOb25lOg0KICAgICAgICB4Ql9yZWcgPSBkZl9CWyJYIl0udmFsdWVzDQogICAgaWYgeUJfcmVnIGlzIE5vbmU6DQogICAgICAgIHlCX3JlZyA9IGRmX0JbIlkiXS52YWx1ZXMNCiAgICBtYXRjaGVkX3Jvd3MgPSBbXQ0KICAgIHVzZWRfViA9IHNldCgpDQogICAgZm9yIGksIChfLCByb3dCKSBpbiBlbnVtZXJhdGUoZGZfQi5pdGVycm93cygpKToNCiAgICAgICAgeEIsIHlCID0geEJfcmVnW2ldLCB5Ql9yZWdbaV0NCiAgICAgICAgZGlzdHMgPSBucC5zcXJ0KChkZl9WWyJYIl0gLSB4QikqKjIgKyAoZGZfVlsiWSJdIC0geUIpKioyKQ0KICAgICAgICBtaW5faWR4ID0gZGlzdHMuaWR4bWluKCkNCiAgICAgICAgaWYgZGlzdHNbbWluX2lkeF0gPD0gdG9sIGFuZCBtaW5faWR4IG5vdCBpbiB1c2VkX1Y6DQogICAgICAgICAgICByb3dWID0gZGZfVi5sb2NbbWluX2lkeF0NCiAgICAgICAgICAgIG1lcmdlZCA9IHsNCiAgICAgICAgICAgICAgICAiWF9CIjogcm93QlsiWCJdLCAiWV9CIjogcm93QlsiWSJdLA0KICAgICAgICAgICAgICAgICJGV0hNX0IiOiByb3dCWyJGV0hNIl0sICJGbHV4X0IiOiByb3dCWyJGbHV4Il0sDQogICAgICAgICAgICAgICAgIlhfViI6IHJvd1ZbIlgiXSwgIllfViI6IHJvd1ZbIlkiXSwNCiAgICAgICAgICAgICAgICAiRldITV9WIjogcm93VlsiRldITSJdLCAiRmx1eF9WIjogcm93VlsiRmx1eCJdDQogICAgICAgICAgICB9DQogICAgICAgICAgICBtYXRjaGVkX3Jvd3MuYXBwZW5kKG1lcmdlZCkNCiAgICAgICAgICAgIHVzZWRfVi5hZGQobWluX2lkeCkNCiAgICByZXR1cm4gcGQuRGF0YUZyYW1lKG1hdGNoZWRfcm93cykNCg0KZGVmIGV4dHJhY3RfcmVmZXJlbmNlX3N0YXJzKGZpdHNfZmlsZSwgZGZfQiwgZGZfViwgeEJfcmVnPU5vbmUsIHlCX3JlZz1Ob25lLA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1hZ19saW1pdD0xNS4wLA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNhdGFsb2c9IklJLzMzNi9hcGFzczkiLA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZ3aG1faGludD00LjApOg0KICAgICIiIlF1ZXJ5IFZpemllciBhbmQgcmV0dXJuIHJlZmVyZW5jZSBzdGFycyB3aXRoIGNhdGFsb2cgbWFncywgbWVhc3VyZWQgZmx1eGVzIGFuZCBwb3NpdGlvbnMgKEIsVikuDQoNCiAgICBmaXRzX2ZpbGUgaXMgVidzIC0gY2F0YWxvZ3VlIHBvc2l0aW9ucyBjb21lIG91dCBpbiBWJ3MgcGl4ZWwgZ3JpZC4gZGZfVidzDQogICAgb3duIFgvWSBhcmUgYWxyZWFkeSBpbiB0aGF0IHNhbWUgZ3JpZCwgYnV0IGRmX0IncyBhcmUgbm90IChCIGlzIGENCiAgICBzZXBhcmF0ZSBleHBvc3VyZSksIHNvIHhCX3JlZy95Ql9yZWcgLSBCJ3MgcG9zaXRpb25zIGFmdGVyDQogICAgcmVnaXN0ZXJfQl9vbnRvX1YoKSAtIGFyZSB3aGF0IHRoZSBjYXRhbG9ndWUgaXMgYWN0dWFsbHkgbWF0Y2hlZCBhZ2FpbnN0DQogICAgaW4gQi4gVGhlIG5hdGl2ZSBkZl9CIFgvWSBhcmUgc3RpbGwgd2hhdCBnZXRzIHN0b3JlZCwgZm9yIHBob3RvbWV0cnkgb24NCiAgICB0aGUgQiBpbWFnZSBpdHNlbGYuIiIiDQogICAgaWYgeEJfcmVnIGlzIE5vbmU6DQogICAgICAgIHhCX3JlZyA9IGRmX0JbIlgiXS52YWx1ZXMNCiAgICBpZiB5Ql9yZWcgaXMgTm9uZToNCiAgICAgICAgeUJfcmVnID0gZGZfQlsiWSJdLnZhbHVlcw0KICAgICMgSG93IGZhciBmcm9tIHRoZSBjYXRhbG9ndWUgcG9zaXRpb24gYSBkZXRlY3Rpb24gbWF5IGJlIGFuZCBzdGlsbCBiZQ0KICAgICMgYWNjZXB0ZWQgYXMgdGhhdCBzdGFyLiBUaWVkIHRvIHRoZSBzZWVpbmcsIHNvIGl0IHN0YXlzIGEgZnJhY3Rpb24gb2YgYQ0KICAgICMgc3RhcidzIHdpZHRoIHJhdGhlciB0aGFuIGEgZmxhdCBudW1iZXIgb2YgcGl4ZWxzLg0KICAgIGZsdXhfdG9sID0gbWF4KENBVEFMT0dfRkxVWF9NQVRDSF9NSU4sIENBVEFMT0dfRkxVWF9NQVRDSF9TQ0FMRSAqIGZ3aG1faGludCkNCiAgICBwcmludChmIiAgIG1hdGNoaW5nIGNhdGFsb2d1ZSBzdGFycyB0byBkZXRlY3Rpb25zIHdpdGhpbiB7Zmx1eF90b2w6LjFmfSBweCIsIGZsdXNoPVRydWUpDQogICAgaGRyID0gZml0cy5nZXRoZWFkZXIoZml0c19maWxlKQ0KICAgIHdjcyA9IFdDUyhoZHIpDQoNCiAgICByYV9jZW50ZXIsIGRlY19jZW50ZXIgPSB3Y3Mud2NzLmNydmFsDQogICAgbmF4aXMxLCBuYXhpczIgPSBoZHJbIk5BWElTMSJdLCBoZHJbIk5BWElTMiJdDQogICAgc2NhbGVfZGVnID0gbnAubWVhbihucC5hYnMod2NzLnBpeGVsX3NjYWxlX21hdHJpeC5kaWFnb25hbCgpKSkNCiAgICBmb3ZfcmEgPSBuYXhpczEgKiBzY2FsZV9kZWcNCiAgICBmb3ZfZGVjID0gbmF4aXMyICogc2NhbGVfZGVnDQoNCiAgICAjIHJvd19saW1pdCBoYXMgdG8gYmUgcGFzc2VkIHRvIHRoZSBjb25zdHJ1Y3Rvciwgbm90IHNldCBvbiB0aGUgY2xhc3MNCiAgICAjIGFmdGVyd2FyZHMgLSBzZWUgdGhlIHNhbWUgZml4IGluIENsdXN0ZXJfQ01ELnB5J3MgZ2V0X2FwYXNzX2NhbGliX3N0YXJzDQogICAgIyBmb3Igd2h5OiBkb2luZyBpdCBvbiB0aGUgY2xhc3MgbGVhdmVzIGEgZnJlc2hseSBjb25zdHJ1Y3RlZCBpbnN0YW5jZSBhdA0KICAgICMgYXN0cm9xdWVyeSdzIGRlZmF1bHQgb2YgNTAgcm93cywgc2lsZW50bHkuDQogICAgdiA9IFZpemllcihjb2x1bW5zPVsiUkFKMjAwMCIsIkRFSjIwMDAiLCJCbWFnIiwiVm1hZyJdLA0KICAgICAgICAgICAgICAgY29sdW1uX2ZpbHRlcnM9eyJWbWFnIjoiPCUuMmYiICUgbWFnX2xpbWl0fSwgcm93X2xpbWl0PS0xKQ0KICAgIHJlc3VsdCA9IHYucXVlcnlfcmVnaW9uKA0KICAgICAgICBTa3lDb29yZChyYV9jZW50ZXIsIGRlY19jZW50ZXIsIHVuaXQ9ImRlZyIpLA0KICAgICAgICB3aWR0aD1mIntmb3ZfcmF9ZCIsIGhlaWdodD1mIntmb3ZfZGVjfWQiLA0KICAgICAgICBjYXRhbG9nPWNhdGFsb2cNCiAgICApDQoNCiAgICBpZiBsZW4ocmVzdWx0KSA9PSAwOg0KICAgICAgICBwcmludCgiTm8gcmVmZXJlbmNlIHN0YXJzIGZvdW5kIGluIFZpemllciBjYXRhbG9nLiIpDQogICAgICAgIHJldHVybiBwZC5EYXRhRnJhbWUoKQ0KDQogICAgc3RhcnMgPSByZXN1bHRbMF0NCiAgICBjb29yZHMgPSBTa3lDb29yZChzdGFyc1siUkFKMjAwMCJdLCBzdGFyc1siREVKMjAwMCJdLCB1bml0PSJkZWciKQ0KICAgIHhfcGl4LCB5X3BpeCA9IHdjcy53b3JsZF90b19waXhlbChjb29yZHMpDQoNCiAgICBkZl9yZWYgPSBwZC5EYXRhRnJhbWUoew0KICAgICAgICAiUkEiOiBzdGFyc1siUkFKMjAwMCJdLA0KICAgICAgICAiRGVjIjogc3RhcnNbIkRFSjIwMDAiXSwNCiAgICAgICAgIkJtYWciOiBzdGFyc1siQm1hZyJdLA0KICAgICAgICAiVm1hZyI6IHN0YXJzWyJWbWFnIl0sDQogICAgICAgICJYX3BpeCI6IHhfcGl4LA0KICAgICAgICAiWV9waXgiOiB5X3BpeA0KICAgIH0pDQoNCiAgICBmbHV4X0IsIGZsdXhfViA9IFtdLCBbXQ0KICAgIFhCX21lYXMsIFlCX21lYXMsIFhWX21lYXMsIFlWX21lYXMgPSBbXSwgW10sIFtdLCBbXQ0KICAgIFhCX3JlZ19tZWFzLCBZQl9yZWdfbWVhcyA9IFtdLCBbXQ0KDQogICAgZm9yIF8sIHJvdyBpbiBkZl9yZWYuaXRlcnJvd3MoKToNCiAgICAgICAgZEIgPSBucC5zcXJ0KCh4Ql9yZWcgLSByb3dbIlhfcGl4Il0pKioyICsgKHlCX3JlZyAtIHJvd1siWV9waXgiXSkqKjIpDQogICAgICAgIGRWID0gbnAuc3FydCgoZGZfVlsiWCJdIC0gcm93WyJYX3BpeCJdKSoqMiArIChkZl9WWyJZIl0gLSByb3dbIllfcGl4Il0pKioyKQ0KDQogICAgICAgIGlmIGRCLm1pbigpIDw9IGZsdXhfdG9sOg0KICAgICAgICAgICAgaWR4QiA9IGludChucC5hcmdtaW4oZEIpKQ0KICAgICAgICAgICAgZkIgPSBkZl9CLmlsb2NbaWR4Ql1bIkZsdXgiXQ0KICAgICAgICAgICAgWEIsIFlCID0gZGZfQi5pbG9jW2lkeEJdWyJYIl0sIGRmX0IuaWxvY1tpZHhCXVsiWSJdDQogICAgICAgICAgICBYQl9yZWcsIFlCX3JlZyA9IHhCX3JlZ1tpZHhCXSwgeUJfcmVnW2lkeEJdDQogICAgICAgIGVsc2U6DQogICAgICAgICAgICBmQiwgWEIsIFlCLCBYQl9yZWcsIFlCX3JlZyA9IG5wLm5hbiwgbnAubmFuLCBucC5uYW4sIG5wLm5hbiwgbnAubmFuDQoNCiAgICAgICAgaWYgZFYubWluKCkgPD0gZmx1eF90b2w6DQogICAgICAgICAgICBpZHhWID0gZFYuaWR4bWluKCkNCiAgICAgICAgICAgIGZWID0gZGZfVi5sb2NbaWR4ViwgIkZsdXgiXQ0KICAgICAgICAgICAgWFYsIFlWID0gZGZfVi5sb2NbaWR4ViwgIlgiXSwgZGZfVi5sb2NbaWR4ViwgIlkiXQ0KICAgICAgICBlbHNlOg0KICAgICAgICAgICAgZlYsIFhWLCBZViA9IG5wLm5hbiwgbnAubmFuLCBucC5uYW4NCg0KICAgICAgICBmbHV4X0IuYXBwZW5kKGZCKQ0KICAgICAgICBmbHV4X1YuYXBwZW5kKGZWKQ0KICAgICAgICBYQl9tZWFzLmFwcGVuZChYQikNCiAgICAgICAgWUJfbWVhcy5hcHBlbmQoWUIpDQogICAgICAgIFhWX21lYXMuYXBwZW5kKFhWKQ0KICAgICAgICBZVl9tZWFzLmFwcGVuZChZVikNCiAgICAgICAgWEJfcmVnX21lYXMuYXBwZW5kKFhCX3JlZykNCiAgICAgICAgWUJfcmVnX21lYXMuYXBwZW5kKFlCX3JlZykNCg0KICAgIGRmX3JlZlsiRmx1eF9CX21lYXN1cmVkIl0gPSBmbHV4X0INCiAgICBkZl9yZWZbIkZsdXhfVl9tZWFzdXJlZCJdID0gZmx1eF9WDQogICAgZGZfcmVmWyJYX0IiXSA9IFhCX21lYXMNCiAgICBkZl9yZWZbIllfQiJdID0gWUJfbWVhcw0KICAgIGRmX3JlZlsiWF9WIl0gPSBYVl9tZWFzDQogICAgZGZfcmVmWyJZX1YiXSA9IFlWX21lYXMNCg0KICAgICMgZmlsdGVyIG9ubHkgc3RhcnMgd2l0aCB2YWxpZCBmbHV4IGluIGJvdGggYmFuZHMNCiAgICBuX2ZvdW5kID0gbGVuKGRmX3JlZikNCiAgICBkZl9yZWYgPSBkZl9yZWYuZHJvcG5hKHN1YnNldD1bIkZsdXhfQl9tZWFzdXJlZCIsIkZsdXhfVl9tZWFzdXJlZCJdKQ0KICAgIG5fYm90aGJhbmRzID0gbGVuKGRmX3JlZikNCiAgICBYQl9yZWdfbWVhcyA9IG5wLmFycmF5KFhCX3JlZ19tZWFzKVtkZl9yZWYuaW5kZXhdIGlmIG5fYm90aGJhbmRzIGVsc2UgbnAuYXJyYXkoW10pDQogICAgWUJfcmVnX21lYXMgPSBucC5hcnJheShZQl9yZWdfbWVhcylbZGZfcmVmLmluZGV4XSBpZiBuX2JvdGhiYW5kcyBlbHNlIG5wLmFycmF5KFtdKQ0KDQogICAgIyBIb3cgZmFyIGVhY2ggYmFuZCdzIG1lYXN1cmVtZW50IGVuZGVkIHVwIGZyb20gd2hlcmUgdGhlIGNhdGFsb2d1ZSBzYXlzIHRoZQ0KICAgICMgc3RhciBpcywgYW5kIGZyb20gdGhlIG90aGVyIGJhbmQuIEtlcHQgaW4gdGhlIGZpbGU6IHdoZW4gYSBjYWxpYnJhdGlvbg0KICAgICMgZ29lcyB3cm9uZyB0aGVzZSB0d28gY29sdW1ucyBzYXkgc28gYXQgYSBnbGFuY2UuDQogICAgIw0KICAgICMgQm90aCBkaXN0YW5jZXMgdXNlIEIncyBwb3NpdGlvbiBhZnRlciByZWdpc3RyYXRpb24gb250byBWJ3MgZ3JpZCwgbm90DQogICAgIyBpdHMgbmF0aXZlIHBpeGVsIHBvc2l0aW9uIC0gWF9waXgvWV9waXggYW5kIFhfVi9ZX1YgYXJlIGFscmVhZHkgaW4gdGhhdA0KICAgICMgZ3JpZCwgYW5kIGNvbXBhcmluZyB0aGVtIGFnYWluc3QgQidzIG93biwgdW5yZWdpc3RlcmVkIHBpeGVscyB3b3VsZA0KICAgICMgcmVwb3J0IGFuIG9mZnNldCB0aGF0IGlzIHJlYWxseSBqdXN0IHRoZSB0d28gZnJhbWVzJyBvd24gZGlmZmVyZW5jZS4NCiAgICBkZl9yZWZbIk9mZnNldF9CIl0gPSBucC5oeXBvdChYQl9yZWdfbWVhcyAtIGRmX3JlZlsiWF9waXgiXSwNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBZQl9yZWdfbWVhcyAtIGRmX3JlZlsiWV9waXgiXSkNCiAgICBkZl9yZWZbIk9mZnNldF9WIl0gPSBucC5oeXBvdChkZl9yZWZbIlhfViJdIC0gZGZfcmVmWyJYX3BpeCJdLA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRmX3JlZlsiWV9WIl0gLSBkZl9yZWZbIllfcGl4Il0pDQogICAgZGZfcmVmWyJTZXBhcmF0aW9uX0JfViJdID0gbnAuaHlwb3QoWEJfcmVnX21lYXMgLSBkZl9yZWZbIlhfViJdLA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFlCX3JlZ19tZWFzIC0gZGZfcmVmWyJZX1YiXSkNCg0KICAgICMgQSBzdGFyIGlzIG9ubHkgdXNhYmxlIGlmIGJvdGggYmFuZHMgbWVhc3VyZWQgdGhlIHNhbWUgb2JqZWN0LiBXaGVuIG9uZQ0KICAgICMgYmFuZCBtaXNzZXMgdGhlIHN0YXIsIHRoZSBuZWFyZXN0IGJsb2IgaXMgYWNjZXB0ZWQgaW5zdGVhZCBhbmQgdGhlIHR3bw0KICAgICMgYmFuZHMgZHJpZnQgYXBhcnQgLSB3aGljaCBpcyBleGFjdGx5IHRoZSBwYWlyaW5nIHRoYXQgcnVpbnMgYSB6ZXJvIHBvaW50Lg0KICAgIGNyb3NzX3RvbCA9IFJFRl9DUk9TU19CQU5EX1RPTEVSQU5DRSAqIG1heChmd2htX2hpbnQsIDEuMCkNCiAgICBzYW1lX29iamVjdCA9IGRmX3JlZlsiU2VwYXJhdGlvbl9CX1YiXSA8PSBjcm9zc190b2wNCiAgICBuX3NwbGl0ID0gaW50KCh+c2FtZV9vYmplY3QpLnN1bSgpKQ0KICAgIGRmX3JlZiA9IGRmX3JlZltzYW1lX29iamVjdF0NCg0KICAgIHByaW50KGYiICAgcmVmZXJlbmNlIHN0YXJzOiB7bl9mb3VuZH0gaW4gdGhlIGNhdGFsb2d1ZSwiDQogICAgICAgICAgZiIge25fYm90aGJhbmRzfSBtZWFzdXJlZCBpbiBib3RoIGJhbmRzLCINCiAgICAgICAgICBmIiB7bl9zcGxpdH0gZHJvcHBlZCBmb3IgbGFuZGluZyBvbiBkaWZmZXJlbnQgb2JqZWN0cyBpbiBCIGFuZCBWIg0KICAgICAgICAgIGYiIChtb3JlIHRoYW4ge2Nyb3NzX3RvbDouMWZ9IHB4IGFwYXJ0KSwiDQogICAgICAgICAgZiIge2xlbihkZl9yZWYpfSB1c2FibGUiLCBmbHVzaD1UcnVlKQ0KDQogICAgcmV0dXJuIGRmX3JlZi5yZXNldF9pbmRleChkcm9wPVRydWUpDQoNCmRlZiBjb21wdXRlX3plcm9fcG9pbnQoZmx1eGVzLCBtYWdzLCBsYWJlbD0iIik6DQogICAgIiIiWmVybyBwb2ludCBmcm9tIHJlZmVyZW5jZSBzdGFycywgY29tYmluZWQgc28gdGhhdCBvbmUgYmFkIHN0YXIgY2Fubm90DQogICAgY2FycnkgdGhlIGZyYW1lLg0KDQogICAgVGhpcyB3YXMgYSBwbGFpbiBtZWFuLCBhbmQgYSBtZWFuIGlzIHRoZSB3cm9uZyB0b29sLiBBIHJlZmVyZW5jZSBzdGFyIHdob3NlDQogICAgZmx1eCB3YXMgbWVhc3VyZWQgb24gdGhlIHdyb25nIG9iamVjdCBwYWlycyBhIHJlYWwgY2F0YWxvZ3VlIG1hZ25pdHVkZSB3aXRoDQogICAgYSBtZWFuaW5nbGVzcyBmbHV4OyB0aGUgcGFpciBjYW4gYmUgc2V2ZXJhbCBtYWduaXR1ZGVzIG91dCwgYW5kIGEgc2luZ2xlIG9uZQ0KICAgIG1vdmVzIHRoZSBtZWFuIGVub3VnaCB0byBzaGlmdCBldmVyeSBtYWduaXR1ZGUgaW4gdGhlIG91dHB1dC4gVGFraW5nIHRoZQ0KICAgIG1lZGlhbiBmaXJzdCBhbmQgdGhlbiBjbGlwcGluZyBhYm91dCBpdCBtYWtlcyB0aGUgYmFkIHN0YXJzIHZpc2libGUgYW5kDQogICAgaGFybWxlc3M6IHRoZXkgYXJlIHJlcG9ydGVkLCBub3QgYXZlcmFnZWQgaW4uDQogICAgIiIiDQogICAgZmx1eGVzID0gbnAuYXNhcnJheShmbHV4ZXMsIGR0eXBlPWZsb2F0KQ0KICAgIG1hZ3MgPSBucC5hc2FycmF5KG1hZ3MsIGR0eXBlPWZsb2F0KQ0KICAgIG1hc2sgPSBucC5pc2Zpbml0ZShmbHV4ZXMpICYgbnAuaXNmaW5pdGUobWFncykgJiAoZmx1eGVzID4gMCkNCiAgICBpZiBub3QgbWFzay5hbnkoKToNCiAgICAgICAgcmV0dXJuIG5wLm5hbiwgMCwgMA0KDQogICAgenAgPSBtYWdzW21hc2tdICsgMi41ICogbnAubG9nMTAoZmx1eGVzW21hc2tdKQ0KICAgIGtlZXAgPSBucC5vbmVzKGxlbih6cCksIGR0eXBlPWJvb2wpDQogICAgaWYgbGVuKHpwKSA+PSA0Og0KICAgICAgICBzcHJlYWQgPSBucC5tZWRpYW4obnAuYWJzKHpwIC0gbnAubWVkaWFuKHpwKSkpICogMS40ODI2ICAjIHJvYnVzdCBzaWdtYQ0KICAgICAgICBpZiBzcHJlYWQgPiAwOg0KICAgICAgICAgICAga2VlcCA9IG5wLmFicyh6cCAtIG5wLm1lZGlhbih6cCkpIDw9IFJFRl9aUF9PVVRMSUVSX1NJR01BICogc3ByZWFkDQogICAgdmFsdWUgPSBmbG9hdChucC5tZWRpYW4oenBba2VlcF0pKSBpZiBrZWVwLmFueSgpIGVsc2UgZmxvYXQobnAubWVkaWFuKHpwKSkNCiAgICBuX3VzZWQsIG5fY3V0ID0gaW50KGtlZXAuc3VtKCkpLCBpbnQoKH5rZWVwKS5zdW0oKSkNCiAgICBpZiBsYWJlbDoNCiAgICAgICAgcHJpbnQoZiIgICB6ZXJvIHBvaW50IHtsYWJlbH0gPSB7dmFsdWU6LjNmfSINCiAgICAgICAgICAgICAgZiIgICBmcm9tIHtuX3VzZWR9IHN0YXJzLCB7bl9jdXR9IHJlamVjdGVkIGFzIG91dGxpZXJzIg0KICAgICAgICAgICAgICBmIiAgIChzY2F0dGVyIHtucC5zdGQoenBba2VlcF0pOi4zZn0gbWFnKSIsIGZsdXNoPVRydWUpDQogICAgcmV0dXJuIHZhbHVlLCBuX3VzZWQsIG5fY3V0DQoNCmRlZiBtYWtlX2NtZChkZiwgYmFzZV9maWxlLCBsYWJlbCk6DQogICAgaWYgIk1hZ19CIiBpbiBkZi5jb2x1bW5zIGFuZCAiTWFnX1YiIGluIGRmLmNvbHVtbnM6DQogICAgICAgICMgQ29tcHV0ZSBjb2xvciBpbmRleA0KICAgICAgICBkZiA9IGRmLmNvcHkoKQ0KICAgICAgICBkZlsiQi1WIl0gPSBkZlsiTWFnX0IiXSAtIGRmWyJNYWdfViJdDQoNCiAgICAgICAgIyBSZW9yZGVyIGNvbHVtbnM6IHBsYWNlIEItViBpbW1lZGlhdGVseSBiZWZvcmUgTWFnX1YNCiAgICAgICAgY29scyA9IGxpc3QoZGYuY29sdW1ucykNCiAgICAgICAgaWYgIkItViIgaW4gY29scyBhbmQgIk1hZ19WIiBpbiBjb2xzOg0KICAgICAgICAgICAgY29scy5yZW1vdmUoIkItViIpDQogICAgICAgICAgICBjb2xzLnJlbW92ZSgiTWFnX1YiKQ0KICAgICAgICAgICAgIyBwdXQgZXZlcnl0aGluZyBlbHNlIGZpcnN0LCB0aGVuIEItViwgdGhlbiBNYWdfVg0KICAgICAgICAgICAgY29scyA9IGNvbHMgKyBbIkItViIsICJNYWdfViJdDQogICAgICAgICAgICBkZiA9IGRmW2NvbHNdDQoNCiAgICAgICAgIyBTYXZlIG5ldyBmaWxlIHdpdGggY29sb3IgaW5kZXgNCiAgICAgICAgIyBiYXNlX2ZpbGUgYWxyZWFkeSBjYXJyaWVzIHdoaWNoZXZlciBzdGFnZSB0aGlzIGlzIC0gIl9ub19zdGFycyIsDQogICAgICAgICMgIl9ub19zdGFyc19jb2xvcl9maWx0ZXJlZCIgLSBzbyB0aGUgc3RhZ2UgaXMgbm90IGFwcGVuZGVkIGFnYWluLg0KICAgICAgICAjIEl0IHVzZWQgdG8gYmUsIHdoaWNoIHByb2R1Y2VkIG5hbWVzIGxpa2UNCiAgICAgICAgIyAuLi5fbm9fc3RhcnNfY29sb3JfZmlsdGVyZWRfd2l0aF9jb2xvcl9ub19zdGFyc19jb2xvcl9maWx0ZXJlZC5jc3YNCiAgICAgICAgb3V0X2ZpbGVfY29sb3IgPSBiYXNlX2ZpbGUucmVwbGFjZSgiLmNzdiIsICJfd2l0aF9jb2xvci5jc3YiKQ0KICAgICAgICBkZi50b19jc3Yob3V0X2ZpbGVfY29sb3IsIGluZGV4PUZhbHNlKQ0KICAgICAgICBwcmludChmIkZpbGUgd2l0aCBjb2xvciBpbmRleCBzYXZlZCB0byB7b3V0X2ZpbGVfY29sb3J9IikNCg0KICAgICAgICAjIENyZWF0ZSBDTUQgcGxvdA0KICAgICAgICBwbHQuZmlndXJlKGZpZ3NpemU9KDgsIDEwKSkNCiAgICAgICAgcGx0LnNjYXR0ZXIoZGZbIkItViJdLCBkZlsiTWFnX1YiXSwNCiAgICAgICAgICAgICAgICAgICAgcz0zMCwgZWRnZWNvbG9yPSJibGFjayIsIGZhY2Vjb2xvcj0iY3lhbiIsIGFscGhhPTAuNykNCg0KICAgICAgICBwbHQuZ2NhKCkuaW52ZXJ0X3lheGlzKCkgICMgYnJpZ2h0ZXIgb2JqZWN0cyBhdCB0aGUgdG9wDQogICAgICAgIHBsdC54bGFiZWwoIkIgLSBWIChDb2xvciBJbmRleCkiKQ0KICAgICAgICBwbHQueWxhYmVsKCJWIG1hZ25pdHVkZSIpDQogICAgICAgIHBsdC50aXRsZShmIkNNRCBvZiB7b2JqX25hbWV9IHtsYWJlbH0gKEItViwgVikiKQ0KDQogICAgICAgICMgPj4+IENvbnRyb2wgb2YgeC1heGlzIChjb2xvciBpbmRleCkgcmFuZ2UgPDw8DQogICAgICAgIHBsdC54bGltKENNRF9DT0xPUl9NSU4sIENNRF9DT0xPUl9NQVgpDQoNCiAgICAgICAgb3V0X2NtZCA9IGJhc2VfZmlsZS5yZXBsYWNlKCIuY3N2IiwgIl9DTUQucG5nIikNCiAgICAgICAgcGx0LnNhdmVmaWcob3V0X2NtZCwgZHBpPTE1MCkNCiAgICAgICAgcGx0LmNsb3NlKCkNCiAgICAgICAgcHJpbnQoZiJDTUQgZGlhZ3JhbSBzYXZlZCB0byB7b3V0X2NtZH0iKQ0KICAgIGVsc2U6DQogICAgICAgIHByaW50KGYiTWFnX0Igb3IgTWFnX1Ygbm90IGZvdW5kIGluIHtsYWJlbH0gY2F0YWxvZy4gQ01EIG5vdCBjcmVhdGVkLiIpDQoNClJfViA9IDMuMSAgICAgICAgICAgICAgICAgICAgICAgICMgQV9WIC8gRShCLVYpIGZvciBkdXN0IGluIG91ciBvd24gZ2FsYXh5DQoNCkRJU1RBTkNFX0NBVEFMT0cgPSAiSi9BSi8xNTIvNTAiICAjIENvc21pY2Zsb3dzLTMsIFR1bGx5IGV0IGFsLiAyMDE2DQoNCmRlZiBsb29rdXBfZGlzdGFuY2UobmFtZSk6DQogICAgIiIiUmVkc2hpZnQtaW5kZXBlbmRlbnQgZGlzdGFuY2UgaW4gTXBjIGZyb20gQ29zbWljZmxvd3MtMywgb3IgTm9uZS4NCg0KICAgIFJlZHNoaWZ0IGlzIG5vIHVzZSBmb3IgYSBnYWxheHkgdGhpcyBjbG9zZSAtIE0xMDEncyByZWNlc3Npb24gdmVsb2NpdHkgcHV0cw0KICAgIGl0IGF0IGFib3V0IDMgTXBjLCByb3VnaGx5IGhhbGYgaXRzIHJlYWwgZGlzdGFuY2UgLSBzbyBhIGNhdGFsb2d1ZSBvZiBkaXJlY3QNCiAgICBtZWFzdXJlbWVudHMgaXMgd2hhdCBpcyB3YW50ZWQuDQogICAgIiIiDQogICAgdHJ5Og0KICAgICAgICBjb29yZCA9IFNreUNvb3JkLmZyb21fbmFtZShuYW1lKQ0KICAgICAgICBoaXQgPSBWaXppZXIoY29sdW1ucz1bIioqIl0sIHJvd19saW1pdD01KS5xdWVyeV9yZWdpb24oDQogICAgICAgICAgICBjb29yZCwgcmFkaXVzPTIgKiB1LmFyY21pbiwgY2F0YWxvZz1ESVNUQU5DRV9DQVRBTE9HKQ0KICAgICAgICBpZiBub3QgaGl0Og0KICAgICAgICAgICAgcmV0dXJuIE5vbmUNCiAgICAgICAgdGFibGUgPSBoaXRbMF0NCiAgICAgICAgZm9yIGNvbCBpbiAoIkRpc3QiLCAiPERpc3Q+Iik6DQogICAgICAgICAgICBpZiBjb2wgaW4gdGFibGUuY29sbmFtZXM6DQogICAgICAgICAgICAgICAgdmFsdWUgPSBmbG9hdCh0YWJsZVtjb2xdWzBdKQ0KICAgICAgICAgICAgICAgIGlmIG5wLmlzZmluaXRlKHZhbHVlKSBhbmQgdmFsdWUgPiAwOg0KICAgICAgICAgICAgICAgICAgICBwcmludChmIltsb29rdXBdIHtuYW1lfToge3ZhbHVlOi4yZn0gTXBjIg0KICAgICAgICAgICAgICAgICAgICAgICAgICBmIiAoQ29zbWljZmxvd3MtMykiLCBmbHVzaD1UcnVlKQ0KICAgICAgICAgICAgICAgICAgICByZXR1cm4gdmFsdWUNCiAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGV4YzoNCiAgICAgICAgcHJpbnQoZiJbbG9va3VwXSBkaXN0YW5jZSB1bmF2YWlsYWJsZSAoe2V4Y30pIiwgZmx1c2g9VHJ1ZSkNCiAgICByZXR1cm4gTm9uZQ0KDQpkZWYgbG9va3VwX2NlbnRlcl9waXhlbChuYW1lLCBoZWFkZXIpOg0KICAgICIiIldoZXJlIHRoZSBnYWxheHkgYWN0dWFsbHkgc2l0cyBpbiB0aGlzIGZyYW1lLCBpbiBwaXhlbHMsIG9yIE5vbmUuDQoNCiAgICBUaGUgcmFkaWFsIHByb2ZpbGUgdXNlZCB0byBhc3N1bWUgdGhlIGdhbGF4eSBzaXRzIGV4YWN0bHkgYXQgdGhlIGZyYW1lJ3MNCiAgICBvd24gZ2VvbWV0cmljIGNlbnRyZSAtIHRydWUgb25seSBpZiB0aGUgdGVsZXNjb3BlIHdhcyBwb2ludGVkIGRlYWQtb24uIE9uDQogICAgYSByZWFsIHBhaXIgb2YgTTEwMSBmcmFtZXMgdGhlIHBvaW50aW5nIHdhcyBvZmYgYnkgNzIgcHgsIGFib3V0IDEuOCBrcGMgYXQNCiAgICB0aGF0IGRpc3RhbmNlOiBub3QgaHVnZSBuZXh0IHRvIHRoZSB3aG9sZSBwcm9maWxlLCBidXQgZW5vdWdoIHRvIGJpYXMgdGhlDQogICAgaW5uZXIgcmluZ3MsIHdoZXJlIDcyIHB4IGlzIGEgbGFyZ2UgZnJhY3Rpb24gb2YgdGhlIHJhZGl1cywgYW5kIHRvIHNoaWZ0DQogICAgdGhlIHdob2xlIHJhZGl1cyBzY2FsZS4gVGhlIGdhbGF4eSdzIG93biBjYXRhbG9ndWVkIHBvc2l0aW9uLCBjb252ZXJ0ZWQNCiAgICB0aHJvdWdoIHRoaXMgZnJhbWUncyBXQ1MsIGRvZXMgbm90IGRlcGVuZCBvbiBob3cgd2VsbCB0aGUgcG9pbnRpbmcgbGFuZGVkLg0KICAgICIiIg0KICAgIHRyeToNCiAgICAgICAgY29vcmQgPSBTa3lDb29yZC5mcm9tX25hbWUobmFtZSkNCiAgICAgICAgdyA9IFdDUyhoZWFkZXIpDQogICAgICAgIHgsIHkgPSB3LndvcmxkX3RvX3BpeGVsKGNvb3JkKQ0KICAgICAgICB4LCB5ID0gZmxvYXQoeCksIGZsb2F0KHkpDQogICAgICAgIG54LCBueSA9IGhlYWRlclsiTkFYSVMxIl0sIGhlYWRlclsiTkFYSVMyIl0NCiAgICAgICAgaWYgMCA8PSB4IDwgbnggYW5kIDAgPD0geSA8IG55Og0KICAgICAgICAgICAgcHJpbnQoZiJbbG9va3VwXSB7bmFtZX06IGNlbnRyZWQgYXQgcGl4ZWwgKHt4Oi4wZn0sIHt5Oi4wZn0pIiwgZmx1c2g9VHJ1ZSkNCiAgICAgICAgICAgIHJldHVybiB4LCB5DQogICAgICAgIHByaW50KGYiW2xvb2t1cF0ge25hbWV9J3MgY2F0YWxvZ3VlZCBwb3NpdGlvbiBmYWxscyBvdXRzaWRlIHRoaXMgZnJhbWUgIg0KICAgICAgICAgICAgIGYiLSB1c2luZyB0aGUgZnJhbWUncyBvd24gY2VudHJlIGZvciB0aGUgcmFkaWFsIHByb2ZpbGUgaW5zdGVhZCIsIGZsdXNoPVRydWUpDQogICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBleGM6DQogICAgICAgIHByaW50KGYiW2xvb2t1cF0gZ2FsYXh5IGNlbnRyZSB1bmF2YWlsYWJsZSAoe2V4Y30pIC0gdXNpbmcgdGhlIGZyYW1lJ3MgIg0KICAgICAgICAgICAgIGYib3duIGNlbnRyZSBmb3IgdGhlIHJhZGlhbCBwcm9maWxlIGluc3RlYWQiLCBmbHVzaD1UcnVlKQ0KICAgIHJldHVybiBOb25lLCBOb25lDQoNCmRlZiBsb29rdXBfZXh0aW5jdGlvbihuYW1lKToNCiAgICAiIiJHYWxhY3RpYyByZWRkZW5pbmcgRShCLVYpIHRvd2FyZHMgdGhlIGdhbGF4eSwgb3IgTm9uZS4NCg0KICAgIFNjaGxhZmx5ICYgRmlua2JlaW5lciAoMjAxMSkgcmVjYWxpYnJhdGlvbiBvZiB0aGUgU2NobGVnZWwgZHVzdCBtYXBzLCBzZXJ2ZWQNCiAgICBieSBJUlNBLiBBX1YgZm9sbG93cyBhcyBSX1YgeCBFKEItVik7IHRoZSB0d28gYXJlIG5vdCBmcmVlIHRvIGJlIHNldCBhcGFydA0KICAgIGZyb20gZWFjaCBvdGhlciwgd2hpY2ggaXMgd2h5IG9ubHkgb25lIG9mIHRoZW0gaXMgYXNrZWQgZm9yLg0KICAgICIiIg0KICAgIHRyeToNCiAgICAgICAgZnJvbSBhc3Ryb3F1ZXJ5LmlwYWMuaXJzYS5pcnNhX2R1c3QgaW1wb3J0IElyc2FEdXN0DQogICAgICAgIHRhYmxlID0gSXJzYUR1c3QuZ2V0X3F1ZXJ5X3RhYmxlKG5hbWUsIHNlY3Rpb249ImVidiIpDQogICAgICAgIHZhbHVlID0gZmxvYXQodGFibGVbImV4dCBTYW5kRiBtZWFuIl1bMF0pDQogICAgICAgIGlmIG5wLmlzZmluaXRlKHZhbHVlKSBhbmQgdmFsdWUgPj0gMDoNCiAgICAgICAgICAgIHByaW50KGYiW2xvb2t1cF0ge25hbWV9OiBFKEItVikgPSB7dmFsdWU6LjRmfSwiDQogICAgICAgICAgICAgICAgICBmIiBzbyBBX1YgPSB7Ul9WICogdmFsdWU6LjNmfSAoU2NobGFmbHkgJiBGaW5rYmVpbmVyIDIwMTEpIiwNCiAgICAgICAgICAgICAgICAgIGZsdXNoPVRydWUpDQogICAgICAgICAgICByZXR1cm4gdmFsdWUNCiAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGV4YzoNCiAgICAgICAgcHJpbnQoZiJbbG9va3VwXSBleHRpbmN0aW9uIHVuYXZhaWxhYmxlICh7ZXhjfSkiLCBmbHVzaD1UcnVlKQ0KICAgIHJldHVybiBOb25lDQoNCnByaW50KCI9PT0gUGhvdG9tZXRyeSBJbnB1dCA9PT0iKQ0KDQpvYmpfbmFtZSA9IF9hc2soIkdhbGF4eSBuYW1lIiwgIk0xMDEiLCBzdHIpDQoNCnByaW50KCJMb29raW5nIHRoZSBnYWxheHkgdXAuLi4iLCBmbHVzaD1UcnVlKQ0KDQpfZGlzdF9kZWZhdWx0ID0gbG9va3VwX2Rpc3RhbmNlKG9ial9uYW1lKQ0KDQpfZWJ2X2RlZmF1bHQgPSBsb29rdXBfZXh0aW5jdGlvbihvYmpfbmFtZSkNCg0KaWYgX2Rpc3RfZGVmYXVsdCBpcyBOb25lOg0KICAgIHByaW50KCJbbG9va3VwXSBubyBwdWJsaXNoZWQgZGlzdGFuY2UgZm91bmQgLSBwbGVhc2Ugc3VwcGx5IG9uZSIsIGZsdXNoPVRydWUpDQoNCmlmIF9lYnZfZGVmYXVsdCBpcyBOb25lOg0KICAgIHByaW50KCJbbG9va3VwXSBubyByZWRkZW5pbmcgZm91bmQgLSBwbGVhc2Ugc3VwcGx5IG9uZSIsIGZsdXNoPVRydWUpDQoNCmRpc3RhbmNlID0gX2FzaygiRGlzdGFuY2UgKE1wYykiLCBfZGlzdF9kZWZhdWx0IGlmIF9kaXN0X2RlZmF1bHQgZWxzZSAxMC4wLCBmbG9hdCkNCg0KRV9CViAgICAgPSBfYXNrKCJHYWxhY3RpYyBjb2xvciBleGNlc3MgRShCLVYpIiwNCiAgICAgICAgICAgICAgICBfZWJ2X2RlZmF1bHQgaWYgX2Vidl9kZWZhdWx0IGlzIG5vdCBOb25lIGVsc2UgMC4wLCBmbG9hdCkNCg0KQV9WID0gUl9WICogRV9CVg0KDQpwcmludChmIlVzaW5nIGRpc3RhbmNlIHtkaXN0YW5jZTouMmZ9IE1wYywiDQogICAgICBmIiBFKEItVikgPSB7RV9CVjouNGZ9LCBBX1YgPSB7QV9WOi4zZn0iLCBmbHVzaD1UcnVlKQ0KDQpmaXRzX2ZpbGVfQiA9IF9ub3JtX3BhdGgoX2FzaygiUGF0aCB0byBCLWJhbmQgRklUUyIsIHIiRDpcZXhhbXBsZV9CLmZ0cyIsIHN0cikpDQoNCmZpdHNfZmlsZV9WID0gX25vcm1fcGF0aChfYXNrKCJQYXRoIHRvIFYtYmFuZCBGSVRTIiwgciJEOlxleGFtcGxlX1YuZnRzIiwgc3RyKSkNCg0KX2RhdGFkaXIgPSBvcy5wYXRoLmRpcm5hbWUoZml0c19maWxlX0IpIGlmIG9zLnBhdGguZGlybmFtZShmaXRzX2ZpbGVfQikgZWxzZSBvcy5nZXRjd2QoKQ0KDQpfb3V0ZGlyID0gb3MucGF0aC5qb2luKF9kYXRhZGlyLCAicmVzdWx0X0NNRCIpDQoNCm9zLm1ha2VkaXJzKF9vdXRkaXIsIGV4aXN0X29rPVRydWUpDQoNCnByaW50KGYiW291dF0gd3JpdGluZyBldmVyeXRoaW5nIHRvIHtfb3V0ZGlyfSIpDQoNCmZpdHNfZmlsZV9CID0gc3VidHJhY3RfYmFja2dyb3VuZF9hbmRfc2F2ZShmaXRzX2ZpbGVfQikNCg0KZml0c19maWxlX1YgPSBzdWJ0cmFjdF9iYWNrZ3JvdW5kX2FuZF9zYXZlKGZpdHNfZmlsZV9WKQ0KDQpfd2luZG93ID0gTm9uZQ0KDQp0cnk6DQogICAgcGFzcyAgIyBmcm9tIGdhbGF4eV93aW5kb3csIGlubGluZWQgYWJvdmUNCiAgICBwYXNzICAjIGZyb20gZ2FsYXh5X2NhdGFsb2d1ZSwgaW5saW5lZCBhYm92ZQ0KDQogICAgX2hkcl9wcm9iZSA9IGZpdHMuZ2V0aGVhZGVyKGZpdHNfZmlsZV9WKQ0KICAgIF9zaGFwZV9wcm9iZSA9IChfaGRyX3Byb2JlWyJOQVhJUzIiXSwgX2hkcl9wcm9iZVsiTkFYSVMxIl0pDQogICAgX2N4LCBfY3kgPSBsb29rdXBfY2VudGVyX3BpeGVsKG9ial9uYW1lLCBfaGRyX3Byb2JlKQ0KICAgIF9nZW9tID0gaHlwZXJsZWRhX2dlb21ldHJ5KG9ial9uYW1lKQ0KICAgIF9zY2FsZSA9IF9oZHJfcHJvYmUuZ2V0KCJQSVhTQ0FMRSIpDQogICAgaWYgX3NjYWxlIGlzIE5vbmU6DQogICAgICAgIF93cHJvYmUgPSBXQ1MoX2hkcl9wcm9iZSkNCiAgICAgICAgX3NjYWxlID0gMzYwMC4wICogZmxvYXQobnAubWVhbihucC5hYnMoX3dwcm9iZS5waXhlbF9zY2FsZV9tYXRyaXguZGlhZ29uYWwoKSkpKQ0KICAgIGlmIF9nZW9tIGFuZCBfZ2VvbS5nZXQoImQyNV9hcmNtaW4iKSBhbmQgX3NjYWxlOg0KICAgICAgICBfcjI1ID0gMC41ICogX2dlb21bImQyNV9hcmNtaW4iXSAqIDYwLjAgLyBmbG9hdChfc2NhbGUpDQogICAgICAgIF93aW5kb3cgPSBnYWxheHlfd2luZG93KF9zaGFwZV9wcm9iZSwgKF9jeCwgX2N5KSwgX3IyNSwgbWFyZ2luPTEuNSkNCiAgICAgICAgaWYgX3dpbmRvdyBpcyBOb25lOg0KICAgICAgICAgICAgcHJpbnQoIlt3aW5kb3ddIHRoZSBnYWxheHkgZmlsbHMgdGhpcyBmcmFtZSAtIHdvcmtpbmcgb24gYWxsIG9mIGl0IiwNCiAgICAgICAgICAgICAgICAgIGZsdXNoPVRydWUpDQogICAgICAgIGVsc2U6DQogICAgICAgICAgICBfeTAsIF95MSwgX3gwLCBfeDEgPSBfd2luZG93DQogICAgICAgICAgICBwcmludChmIlt3aW5kb3ddIHRoZSBnYWxheHkgb2NjdXBpZXMge194MSAtIF94MH0geCB7X3kxIC0gX3kwfSBvZiAiDQogICAgICAgICAgICAgICAgICBmIntfc2hhcGVfcHJvYmVbMV19IHgge19zaGFwZV9wcm9iZVswXX0gcGl4ZWxzOyB0aGUgcmVzdCBpcyBza3kiLA0KICAgICAgICAgICAgICAgICAgZmx1c2g9VHJ1ZSkNCiAgICBlbHNlOg0KICAgICAgICBwcmludCgiW3dpbmRvd10gbm8gaXNvcGhvdGFsIHJhZGl1cyBhdmFpbGFibGUgLSB3b3JraW5nIG9uIHRoZSB3aG9sZSBmcmFtZSIsDQogICAgICAgICAgICAgIGZsdXNoPVRydWUpDQpleGNlcHQgRXhjZXB0aW9uIGFzIF9leGM6DQogICAgcHJpbnQoZiJbd2luZG93XSBjb3VsZCBub3Qgd29yayBvdXQgd2hlcmUgdGhlIGdhbGF4eSBlbmRzICh7X2V4Y30pIC0gd29ya2luZyAiDQogICAgICAgICAgZiJvbiB0aGUgd2hvbGUgZnJhbWUiLCBmbHVzaD1UcnVlKQ0KDQpyZXN1bHRzX0IgPSBwcm9jZXNzX2ZpdHMoZml0c19maWxlX0IsICJCIiwgd2luZG93PV93aW5kb3cpDQoNCnJlc3VsdHNfViA9IHByb2Nlc3NfZml0cyhmaXRzX2ZpbGVfViwgIlYiLCB3aW5kb3c9X3dpbmRvdykNCg0KZGZfQiA9IHBkLkRhdGFGcmFtZShyZXN1bHRzX0IsIGNvbHVtbnM9Ww0KICAgICJYIiwgIlkiLCAiRldITSIsICJBcGVydHVyZSBSYWRpdXMiLCAiRmx1eCIsDQogICAgIkJhbmQiLCAiQW5udWx1cyBJbm5lciBSYWRpdXMiLCAiQW5udWx1cyBPdXRlciBSYWRpdXMiDQpdKQ0KDQpkZl9WID0gcGQuRGF0YUZyYW1lKHJlc3VsdHNfViwgY29sdW1ucz1bDQogICAgIlgiLCAiWSIsICJGV0hNIiwgIkFwZXJ0dXJlIFJhZGl1cyIsICJGbHV4IiwNCiAgICAiQmFuZCIsICJBbm51bHVzIElubmVyIFJhZGl1cyIsICJBbm51bHVzIE91dGVyIFJhZGl1cyINCl0pDQoNCl9md2htX2hpbnQgPSBtYXgoU0VFSU5HX01FQVNVUkVELmdldCgiQiIsIFNFRUlOR19GQUxMQkFDSyksDQogICAgICAgICAgICAgICAgIFNFRUlOR19NRUFTVVJFRC5nZXQoIlYiLCBTRUVJTkdfRkFMTEJBQ0spKQ0KDQpwcmludCgiUmVnaXN0ZXJpbmcgQiBvbnRvIFYuLi4iLCBmbHVzaD1UcnVlKQ0KDQpfQl9pbWcgPSBmaXRzLmdldGRhdGEoZml0c19maWxlX0IsIGV4dD0wKS5hc3R5cGUoZmxvYXQpDQoNCl9WX2ltZyA9IGZpdHMuZ2V0ZGF0YShmaXRzX2ZpbGVfViwgZXh0PTApLmFzdHlwZShmbG9hdCkNCg0KX3hCX3JlZywgX3lCX3JlZyA9IHJlZ2lzdGVyX0Jfb250b19WKF9CX2ltZywgX1ZfaW1nLCBkZl9CWyJYIl0udmFsdWVzLCBkZl9CWyJZIl0udmFsdWVzLA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZpdHNfZmlsZV9CLCBmaXRzX2ZpbGVfVikNCg0KX21hdGNoX3RvbCA9IG1heChNQVRDSF9UT0xFUkFOQ0VfTUlOLCBNQVRDSF9TQ0FMRSAqIF9md2htX2hpbnQpDQoNCnByaW50KGYiTWF0Y2hpbmcgQiB0byBWIHdpdGhpbiB7X21hdGNoX3RvbDouMWZ9IHB4Ig0KICAgICAgZiIgKHN0YXJzIGFyZSB7X2Z3aG1faGludDouMWZ9IHB4IHdpZGUpIiwgZmx1c2g9VHJ1ZSkNCg0KZGZfbWF0Y2hlZCA9IG1hdGNoX3NvdXJjZXMoZGZfQiwgZGZfViwgdG9sPV9tYXRjaF90b2wsIHhCX3JlZz1feEJfcmVnLCB5Ql9yZWc9X3lCX3JlZykNCg0KcHJpbnQoZiIgICB7bGVuKGRmX21hdGNoZWQpOix9IHNvdXJjZXMgbWVhc3VyZWQgaW4gYm90aCBiYW5kcyIsIGZsdXNoPVRydWUpDQoNCmNzdl9maWxlbmFtZSA9IG9zLnBhdGguam9pbihfb3V0ZGlyLCBmIntvYmpfbmFtZX1fcGhvdG9tZXRyeV9yZXN1bHRzLmNzdiIpDQoNCmRmX21hdGNoZWQudG9fY3N2KGNzdl9maWxlbmFtZSwgaW5kZXg9RmFsc2UpDQoNCnByaW50KGYiRGF0YSBzYXZlZCB0byB7Y3N2X2ZpbGVuYW1lfSIpDQoNCmRmX3JlZiA9IGV4dHJhY3RfcmVmZXJlbmNlX3N0YXJzKGZpdHNfZmlsZV9WLCBkZl9CLCBkZl9WLCB4Ql9yZWc9X3hCX3JlZywgeUJfcmVnPV95Ql9yZWcsDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYWdfbGltaXQ9UkVGX01BR19MSU1JVCwNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNhdGFsb2c9UkVGX0NBVEFMT0csDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBmd2htX2hpbnQ9X2Z3aG1faGludCkNCg0KaWYgbm90IGRmX3JlZi5lbXB0eToNCiAgICBjc3ZfcmVmID0gb3MucGF0aC5qb2luKF9vdXRkaXIsIGYie29ial9uYW1lfV9yZWZlcmVuY2Vfc3RhcnMuY3N2IikNCiAgICBkZl9yZWYudG9fY3N2KGNzdl9yZWYsIGluZGV4PUZhbHNlKQ0KICAgIHByaW50KGYiUmVmZXJlbmNlIHN0YXJzIHNhdmVkIHRvIHtjc3ZfcmVmfSIpDQoNCiAgICAjIC0tLS0tIENhbGlicmF0aW9uIHVzaW5nIE4gcmVmZXJlbmNlIHN0YXJzIC0tLS0tDQogICAgcHJpbnQoZiJ7bGVuKGRmX3JlZil9IHJlZmVyZW5jZSBzdGFycyBhdmFpbGFibGUuIikNCiAgICBOID0gX2FzaygiSG93IG1hbnkgdG8gY2FsaWJyYXRlIGFnYWluc3QiLCBsZW4oZGZfcmVmKSwgaW50KQ0KICAgIE4gPSBtYXgoMSwgbWluKGludChOKSwgbGVuKGRmX3JlZikpKQ0KICAgIGRmX2NhbGliID0gZGZfcmVmLmhlYWQoTikNCg0KICAgICMgY29tcHV0ZSB6ZXJvIHBvaW50cw0KICAgIHpwX0IsIG5fenBfQiwgbl9jdXRfQiA9IGNvbXB1dGVfemVyb19wb2ludCgNCiAgICAgICAgZGZfY2FsaWJbIkZsdXhfQl9tZWFzdXJlZCJdLnZhbHVlcywgZGZfY2FsaWJbIkJtYWciXS52YWx1ZXMsIGxhYmVsPSJCIikNCiAgICB6cF9WLCBuX3pwX1YsIG5fY3V0X1YgPSBjb21wdXRlX3plcm9fcG9pbnQoDQogICAgICAgIGRmX2NhbGliWyJGbHV4X1ZfbWVhc3VyZWQiXS52YWx1ZXMsIGRmX2NhbGliWyJWbWFnIl0udmFsdWVzLCBsYWJlbD0iViIpDQoNCiAgICAjIGFkZCBhcGVydHVyZSByYWRpaQ0KICAgIGRmX21hdGNoZWRbIkFwZXJ0dXJlX1JhZGl1c19CIl0gPSBkZl9tYXRjaGVkWyJGV0hNX0IiXSAqIEFQRVJUVVJFX1NDQUxFDQogICAgZGZfbWF0Y2hlZFsiQXBlcnR1cmVfUmFkaXVzX1YiXSA9IGRmX21hdGNoZWRbIkZXSE1fViJdICogQVBFUlRVUkVfU0NBTEUNCg0KICAgICMgQ2FsaWJyYXRlZCBtYWduaXR1ZGVzLCB0aGVuIHRoZSBmb3JlZ3JvdW5kIGR1c3Qgb2Ygb3VyIG93biBnYWxheHkgdGFrZW4NCiAgICAjIG91dC4gQV9WIGRpbXMgdGhlIFYgYmFuZDsgQiBpcyBkaW1tZWQgYnkgQV9WICsgRShCLVYpLCB3aGljaCBpcyB3aGF0IHRoZQ0KICAgICMgY29sb3VyIGV4Y2VzcyBtZWFucy4gU3VidHJhY3RpbmcgYm90aCBsZWF2ZXMgdGhlIGNvbG91ciByZWRkZW5lZCBieQ0KICAgICMgZXhhY3RseSBFKEItViksIHNvIHRoYXQgaXMgd2hhdCBjb21lcyBvZmYgQi1WLg0KICAgIGRmX21hdGNoZWRbIk1hZ19CIl0gPSB6cF9CIC0gMi41ICogbnAubG9nMTAoZGZfbWF0Y2hlZFsiRmx1eF9CIl0pIC0gKEFfViArIEVfQlYpDQogICAgZGZfbWF0Y2hlZFsiTWFnX1YiXSA9IHpwX1YgLSAyLjUgKiBucC5sb2cxMChkZl9tYXRjaGVkWyJGbHV4X1YiXSkgLSBBX1YNCg0KICAgICMgV2hhdCB0aGlzIHJ1biBhc3N1bWVkLCBjYXJyaWVkIGluIHRoZSBkYXRhIGl0c2VsZi4gV2l0aG91dCB0aGVzZSBjb2x1bW5zDQogICAgIyB0aGVyZSBpcyBubyB3YXkgdG8gdGVsbCBhZnRlcndhcmRzIHdoaWNoIGRpc3RhbmNlIG9yIGV4dGluY3Rpb24gcHJvZHVjZWQNCiAgICAjIGEgZ2l2ZW4gc2V0IG9mIG51bWJlcnMuDQogICAgZGZfbWF0Y2hlZFsiWlBfQl91c2VkIl0gPSB6cF9CDQogICAgZGZfbWF0Y2hlZFsiWlBfVl91c2VkIl0gPSB6cF9WDQogICAgZGZfbWF0Y2hlZFsiTl9yZWZfdXNlZCJdID0gbl96cF9CICsgbl96cF9WDQogICAgZGZfbWF0Y2hlZFsiQV9WX3VzZWQiXSA9IEFfVg0KICAgIGRmX21hdGNoZWRbIkVfQlZfdXNlZCJdID0gRV9CVg0KICAgIGRmX21hdGNoZWRbIkRpc3RhbmNlX01wY191c2VkIl0gPSBkaXN0YW5jZQ0KICAgIGRmX21hdGNoZWRbIkZXSE1fbWVhc3VyZWRfQiJdID0gU0VFSU5HX01FQVNVUkVELmdldCgiQiIsIG5wLm5hbikNCiAgICBkZl9tYXRjaGVkWyJGV0hNX21lYXN1cmVkX1YiXSA9IFNFRUlOR19NRUFTVVJFRC5nZXQoIlYiLCBucC5uYW4pDQoNCiAgICAjIHNhdmUgY2FsaWJyYXRlZCBwaG90b21ldHJ5DQogICAgY3N2X2NhbGliID0gb3MucGF0aC5qb2luKF9vdXRkaXIsIGYie29ial9uYW1lfV9jYWxpYnJhdGVkX3Bob3RvbWV0cnkuY3N2IikNCiAgICBkZl9tYXRjaGVkLnRvX2Nzdihjc3ZfY2FsaWIsIGluZGV4PUZhbHNlKQ0KICAgIHByaW50KGYiQ2FsaWJyYXRlZCBwaG90b21ldHJ5IHNhdmVkIHRvIHtjc3ZfY2FsaWJ9IikNCg0KZGF0YV9WLCBoZHJfViA9IGZpdHMuZ2V0ZGF0YShmaXRzX2ZpbGVfViwgaGVhZGVyPVRydWUpDQoNCnBsdC5maWd1cmUoZmlnc2l6ZT0oMTAsIDEwKSkNCg0KcGx0Lmltc2hvdyhkYXRhX1YsIGNtYXA9ImdyYXkiLCBvcmlnaW49Imxvd2VyIiwgdm1pbj1ucC5wZXJjZW50aWxlKGRhdGFfViwgNSksIHZtYXg9bnAucGVyY2VudGlsZShkYXRhX1YsIDk5KSkNCg0KcGx0LmNvbG9yYmFyKGxhYmVsPSJDb3VudHMiKQ0KDQpwbHQuc2NhdHRlcihkZl9WWyJYIl0sIGRmX1ZbIlkiXSwgcz00MCwgZWRnZWNvbG9yPSJncmVlbiIsIGZhY2Vjb2xvcj0ibm9uZSIsIGxhYmVsPSJNZWFzdXJlZCBzb3VyY2VzIikNCg0KcGx0LnRpdGxlKGYie29ial9uYW1lfSAtIFYgYmFuZCB3aXRoIGRldGVjdGVkIHNvdXJjZXMiKQ0KDQpwbHQueGxhYmVsKCJYIFtwaXhlbHNdIikNCg0KcGx0LnlsYWJlbCgiWSBbcGl4ZWxzXSIpDQoNCnBsdC5sZWdlbmQoKQ0KDQpvdXRfcG5nID0gb3MucGF0aC5qb2luKF9vdXRkaXIsIGYie29ial9uYW1lfV9WX3NvdXJjZXMucG5nIikNCg0KcGx0LnNhdmVmaWcob3V0X3BuZywgZHBpPTE1MCkNCg0KcGx0LmNsb3NlKCkNCg0KcHJpbnQoZiJWLWJhbmQgc291cmNlIG1hcCBzYXZlZCB0byB7b3V0X3BuZ30iKQ0KDQpkYXRhX0JfZGlzcCA9IGZpdHMuZ2V0ZGF0YShmaXRzX2ZpbGVfQikNCg0KcGx0LmZpZ3VyZShmaWdzaXplPSgxMCwgMTApKQ0KDQpwbHQuaW1zaG93KGRhdGFfQl9kaXNwLCBjbWFwPSJncmF5Iiwgb3JpZ2luPSJsb3dlciIsDQogICAgICAgICAgIHZtaW49bnAucGVyY2VudGlsZShkYXRhX0JfZGlzcCwgNSksIHZtYXg9bnAucGVyY2VudGlsZShkYXRhX0JfZGlzcCwgOTkpKQ0KDQpwbHQuY29sb3JiYXIobGFiZWw9IkNvdW50cyIpDQoNCnBsdC5zY2F0dGVyKGRmX0JbIlgiXSwgZGZfQlsiWSJdLCBzPTQwLCBlZGdlY29sb3I9ImJsdWUiLCBmYWNlY29sb3I9Im5vbmUiLA0KICAgICAgICAgICAgbGFiZWw9Ik1lYXN1cmVkIHNvdXJjZXMiKQ0KDQpwbHQudGl0bGUoZiJ7b2JqX25hbWV9IC0gQiBiYW5kIHdpdGggZGV0ZWN0ZWQgc291cmNlcyIpDQoNCnBsdC54bGFiZWwoIlggW3BpeGVsc10iKQ0KDQpwbHQueWxhYmVsKCJZIFtwaXhlbHNdIikNCg0KcGx0LmxlZ2VuZCgpDQoNCm91dF9wbmdfQiA9IG9zLnBhdGguam9pbihfb3V0ZGlyLCBmIntvYmpfbmFtZX1fQl9zb3VyY2VzLnBuZyIpDQoNCnBsdC5zYXZlZmlnKG91dF9wbmdfQiwgZHBpPTE1MCkNCg0KcGx0LmNsb3NlKCkNCg0KcHJpbnQoZiJCLWJhbmQgc291cmNlIG1hcCBzYXZlZCB0byB7b3V0X3BuZ19CfSIpDQoNCmNhbGliX2ZpbGUgPSBjc3ZfY2FsaWINCg0KcmVmX2ZpbGUgICA9IGNzdl9yZWYNCg0KZGZfY2FsaWIgPSBwZC5yZWFkX2NzdihjYWxpYl9maWxlKQ0KDQpkZl9yZWYgICA9IHBkLnJlYWRfY3N2KHJlZl9maWxlKQ0KDQpmb3IgY29sIGluIFsiWF9CIiwiWV9CIiwiWF9WIiwiWV9WIl06DQogICAgaWYgY29sIGluIGRmX3JlZi5jb2x1bW5zOg0KICAgICAgICBkZl9yZWZbY29sXSA9IHBkLnRvX251bWVyaWMoZGZfcmVmW2NvbF0sIGVycm9ycz0iY29lcmNlIikNCg0KbWFza19yZW1vdmUgPSBbXQ0KDQpmb3IgaSwgcm93IGluIGRmX2NhbGliLml0ZXJyb3dzKCk6DQogICAgeGIsIHliID0gcm93LmdldCgiWF9CIiwgbnAubmFuKSwgcm93LmdldCgiWV9CIiwgbnAubmFuKQ0KICAgIHh2LCB5diA9IHJvdy5nZXQoIlhfViIsIG5wLm5hbiksIHJvdy5nZXQoIllfViIsIG5wLm5hbikNCg0KICAgICMgY2hlY2sgZGlzdGFuY2UgdG8gYWxsIHJlZmVyZW5jZSBzdGFycw0KICAgIGRCID0gbnAuc3FydCgoZGZfcmVmWyJYX0IiXSAtIHhiKSoqMiArIChkZl9yZWZbIllfQiJdIC0geWIpKioyKQ0KICAgIGRWID0gbnAuc3FydCgoZGZfcmVmWyJYX1YiXSAtIHh2KSoqMiArIChkZl9yZWZbIllfViJdIC0geXYpKioyKQ0KDQogICAgaWYgKGRCLm1pbigpIDw9IFJFTU9WRV9UT0wpIG9yIChkVi5taW4oKSA8PSBSRU1PVkVfVE9MKToNCiAgICAgICAgbWFza19yZW1vdmUuYXBwZW5kKFRydWUpDQogICAgZWxzZToNCiAgICAgICAgbWFza19yZW1vdmUuYXBwZW5kKEZhbHNlKQ0KDQpkZl9jbGVhbiA9IGRmX2NhbGliLmxvY1t+cGQuU2VyaWVzKG1hc2tfcmVtb3ZlKV0ucmVzZXRfaW5kZXgoZHJvcD1UcnVlKQ0KDQpfZ2FpYV9yZW1vdmVkID0gMA0KDQp0cnk6DQogICAgcGFzcyAgIyBmcm9tIGdhbGF4eV9jYXRhbG9ndWUsIGlubGluZWQgYWJvdmUNCg0KICAgIF93ViA9IFdDUyhoZHJfVikNCiAgICBfZmcsIF9mZ2RpYWcgPSBnYWlhX2ZvcmVncm91bmRfc3RhcnMoX3dWLCAoaGRyX1ZbIk5BWElTMiJdLCBoZHJfVlsiTkFYSVMxIl0pKQ0KICAgIGlmIF9mZyBpcyBub3QgTm9uZSBhbmQgbGVuKF9mZyk6DQogICAgICAgIF9meCA9IG5wLmFzYXJyYXkoX2ZnWzosIDBdLCBkdHlwZT1mbG9hdCkNCiAgICAgICAgX2Z5ID0gbnAuYXNhcnJheShfZmdbOiwgMV0sIGR0eXBlPWZsb2F0KQ0KICAgICAgICBfa2VlcCA9IFtdDQogICAgICAgIGZvciBfLCBfcm93IGluIGRmX2NsZWFuLml0ZXJyb3dzKCk6DQogICAgICAgICAgICBfZCA9IG5wLmh5cG90KF9meCAtIGZsb2F0KF9yb3dbIlhfViJdKSwgX2Z5IC0gZmxvYXQoX3Jvd1siWV9WIl0pKQ0KICAgICAgICAgICAgX2tlZXAuYXBwZW5kKGJvb2woX2QubWluKCkgPiBGT1JFR1JPVU5EX1RPTCkpDQogICAgICAgIF9nYWlhX3JlbW92ZWQgPSBpbnQobGVuKGRmX2NsZWFuKSAtIHN1bShfa2VlcCkpDQogICAgICAgIGRmX2NsZWFuID0gZGZfY2xlYW4ubG9jW3BkLlNlcmllcyhfa2VlcCkudmFsdWVzXS5yZXNldF9pbmRleChkcm9wPVRydWUpDQogICAgICAgIHByaW50KGYiICAgIHtfZ2FpYV9yZW1vdmVkfSBzb3VyY2VzIHNpdCBvbiBhIEdhaWEgZm9yZWdyb3VuZCBzdGFyICINCiAgICAgICAgICAgICAgZiIocGFyYWxsYXggb3IgcHJvcGVyIG1vdGlvbiBzaWduaWZpY2FudCkgYW5kIGFyZSBkcm9wcGVkOyAiDQogICAgICAgICAgICAgIGYie2xlbihkZl9jbGVhbil9IGxlZnQiLCBmbHVzaD1UcnVlKQ0KZXhjZXB0IEV4Y2VwdGlvbiBhcyBfZXhjOg0KICAgIHByaW50KGYiICAgIGNvdWxkIG5vdCBjaGVjayBhZ2FpbnN0IEdhaWEgKHtfZXhjfSk7IGZvcmVncm91bmQgc3RhcnMgb3RoZXIgdGhhbiAiDQogICAgICAgICAgZiJ0aGUgY2FsaWJyYXRpb24gcmVmZXJlbmNlcyByZW1haW4gaW4gdGhlIGNhdGFsb2d1ZSIsIGZsdXNoPVRydWUpDQoNCnRyeToNCiAgICBpZiBfcjI1IGFuZCBfY3ggaXMgbm90IE5vbmUgYW5kIF9jeSBpcyBub3QgTm9uZToNCiAgICAgICAgX2QgPSBucC5oeXBvdChkZl9jbGVhblsiWF9WIl0uYXN0eXBlKGZsb2F0KSAtIGZsb2F0KF9jeCksDQogICAgICAgICAgICAgICAgICAgICAgZGZfY2xlYW5bIllfViJdLmFzdHlwZShmbG9hdCkgLSBmbG9hdChfY3kpKQ0KICAgICAgICBfb3V0c2lkZSA9IGludCgoX2QgPiBfcjI1KS5zdW0oKSkNCiAgICAgICAgZGZfY2xlYW4gPSBkZl9jbGVhbi5sb2NbKF9kIDw9IF9yMjUpLnZhbHVlc10ucmVzZXRfaW5kZXgoZHJvcD1UcnVlKQ0KICAgICAgICBwcmludChmIiAgICB7X291dHNpZGV9IHNvdXJjZXMgbGllIGJleW9uZCBSMjUgKHtfcjI1Oi4wZn0gcHgpLCBvdXRzaWRlIHRoZSAiDQogICAgICAgICAgICAgIGYiZ2FsYXh5LCBhbmQgYXJlIGRyb3BwZWQ7IHtsZW4oZGZfY2xlYW4pfSBsZWZ0IiwgZmx1c2g9VHJ1ZSkNCmV4Y2VwdCAoTmFtZUVycm9yLCBUeXBlRXJyb3IsIFZhbHVlRXJyb3IpIGFzIF9leGM6DQogICAgcHJpbnQoZiIgICAgbm8gZ2FsYXh5IHJhZGl1cyBhdmFpbGFibGUgKHtfZXhjfSk7IHRoZSBjYXRhbG9ndWUgaXMgbm90IHRyaW1tZWQgIg0KICAgICAgICAgIGYidG8gdGhlIGdhbGF4eSIsIGZsdXNoPVRydWUpDQoNCm91dF9maWxlID0gY2FsaWJfZmlsZS5yZXBsYWNlKCJfY2FsaWJyYXRlZF9waG90b21ldHJ5LmNzdiIsDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAiX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFycy5jc3YiKQ0KDQpkZl9jbGVhbi50b19jc3Yob3V0X2ZpbGUsIGluZGV4PUZhbHNlKQ0KDQpwcmludChmIkNsZWFuZWQgZmlsZSBzYXZlZCB0byB7b3V0X2ZpbGV9IikNCg0KcGx0LmZpZ3VyZShmaWdzaXplPSgxMCwgMTApKQ0KDQpwbHQuaW1zaG93KGRhdGFfViwgY21hcD0iZ3JheSIsIG9yaWdpbj0ibG93ZXIiLA0KICAgICAgICAgICB2bWluPW5wLnBlcmNlbnRpbGUoZGF0YV9WLCA1KSwgdm1heD1ucC5wZXJjZW50aWxlKGRhdGFfViwgOTkpKQ0KDQpwbHQuY29sb3JiYXIobGFiZWw9IkNvdW50cyIpDQoNCnBsdC5zY2F0dGVyKGRmX2NsZWFuWyJYX1YiXSwgZGZfY2xlYW5bIllfViJdLA0KICAgICAgICAgICAgcz00MCwgZWRnZWNvbG9yPSJncmVlbiIsIGZhY2Vjb2xvcj0ibm9uZSIsIGxhYmVsPSJDbGVhbmVkIHNvdXJjZXMgKG5vIHN0YXJzKSIpDQoNCnBsdC50aXRsZShmIntvYmpfbmFtZX0gLSBWIGJhbmQgd2l0aCBkZXRlY3RlZCBzb3VyY2VzIChubyBzdGFycykiKQ0KDQpwbHQueGxhYmVsKCJYIFtwaXhlbHNdIikNCg0KcGx0LnlsYWJlbCgiWSBbcGl4ZWxzXSIpDQoNCnBsdC5sZWdlbmQoKQ0KDQpvdXRfcG5nX2NsZWFuID0gb3MucGF0aC5qb2luKF9vdXRkaXIsIGYie29ial9uYW1lfV9WX3NvdXJjZXNfbm9fc3RhcnMucG5nIikNCg0KcGx0LnNhdmVmaWcob3V0X3BuZ19jbGVhbiwgZHBpPTE1MCkNCg0KcGx0LmNsb3NlKCkNCg0KcHJpbnQoZiJWLWJhbmQgY2xlYW5lZCBzb3VyY2UgbWFwIHNhdmVkIHRvIHtvdXRfcG5nX2NsZWFufSIpDQoNCm1ha2VfY21kKGRmX2NhbGliLCBjYWxpYl9maWxlLCAiKGFsbCBzb3VyY2VzKSIpDQoNCm1ha2VfY21kKGRmX2NsZWFuLCBvdXRfZmlsZSwgIihubyBzdGFycykiKQ0KDQppZiAiTWFnX0IiIGluIGRmX2NsZWFuLmNvbHVtbnMgYW5kICJNYWdfViIgaW4gZGZfY2xlYW4uY29sdW1uczoNCiAgICBkZl9jbGVhbl9jb2xvciA9IGRmX2NsZWFuLmNvcHkoKQ0KICAgIGRmX2NsZWFuX2NvbG9yWyJCLVYiXSA9IGRmX2NsZWFuX2NvbG9yWyJNYWdfQiJdIC0gZGZfY2xlYW5fY29sb3JbIk1hZ19WIl0NCg0KICAgICMgZmlsdGVyIGJ5IENNRF9DT0xPUl9NSU4gLyBDTURfQ09MT1JfTUFYDQogICAgbWFza19jb2xvciA9IChkZl9jbGVhbl9jb2xvclsiQi1WIl0gPj0gQ01EX0NPTE9SX01JTikgJiAoZGZfY2xlYW5fY29sb3JbIkItViJdIDw9IENNRF9DT0xPUl9NQVgpDQogICAgZGZfY29sb3JfZmlsdGVyZWQgPSBkZl9jbGVhbl9jb2xvclttYXNrX2NvbG9yXS5yZXNldF9pbmRleChkcm9wPVRydWUpDQoNCiAgICAjIC0tLSBzYXZlIGZpbHRlcmVkIGNhdGFsb2cgYXMgQ1NWIC0tLQ0KICAgIG91dF9jc3ZfY29sb3IgPSBvcy5wYXRoLmpvaW4oX291dGRpciwgZiJ7b2JqX25hbWV9X2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFyc19jb2xvcl9maWx0ZXJlZC5jc3YiKQ0KICAgIGRmX2NvbG9yX2ZpbHRlcmVkLnRvX2NzdihvdXRfY3N2X2NvbG9yLCBpbmRleD1GYWxzZSkNCiAgICBwcmludChmIkNvbG9yLWZpbHRlcmVkIHBob3RvbWV0cnkgc2F2ZWQgdG8ge291dF9jc3ZfY29sb3J9IikNCg0KICAgICMgLS0tIGNyZWF0ZSBhbmQgc2F2ZSBnYWxheHkgaW1hZ2Ugd2l0aCBjb2xvci1maWx0ZXJlZCBzb3VyY2VzIC0tLQ0KICAgIHBsdC5maWd1cmUoZmlnc2l6ZT0oMTAsIDEwKSkNCiAgICBwbHQuaW1zaG93KGRhdGFfViwgY21hcD0iZ3JheSIsIG9yaWdpbj0ibG93ZXIiLA0KICAgICAgICAgICAgICAgdm1pbj1ucC5wZXJjZW50aWxlKGRhdGFfViwgNSksIHZtYXg9bnAucGVyY2VudGlsZShkYXRhX1YsIDk5KSkNCiAgICBwbHQuY29sb3JiYXIobGFiZWw9IkNvdW50cyIpDQoNCiAgICBwbHQuc2NhdHRlcihkZl9jb2xvcl9maWx0ZXJlZFsiWF9WIl0sIGRmX2NvbG9yX2ZpbHRlcmVkWyJZX1YiXSwNCiAgICAgICAgICAgICAgICBzPTQwLCBlZGdlY29sb3I9ImdyZWVuIiwgZmFjZWNvbG9yPSJub25lIiwNCiAgICAgICAgICAgICAgICBsYWJlbD1mIlNvdXJjZXMgaW4gY29sb3IgcmFuZ2UgKHtDTURfQ09MT1JfTUlOfSDiiaQgQi1WIOKJpCB7Q01EX0NPTE9SX01BWH0pIikNCg0KICAgIHBsdC50aXRsZShmIntvYmpfbmFtZX0gLSBWIGJhbmQgd2l0aCBjb2xvci1maWx0ZXJlZCBzb3VyY2VzIChubyBzdGFycykiKQ0KICAgIHBsdC54bGFiZWwoIlggW3BpeGVsc10iKQ0KICAgIHBsdC55bGFiZWwoIlkgW3BpeGVsc10iKQ0KICAgIHBsdC5sZWdlbmQoKQ0KDQogICAgIyBPbmUgbGV2ZWwgdXAsIG5vdCBpbnRvIHJlc3VsdF9DTUQgd2l0aCB0aGUgcmVzdC4gVGhpcyBpcyB0aGUgcGljdHVyZSB0aGF0DQogICAgIyBhbnN3ZXJzIHdoZXRoZXIgdGhlIHdob2xlIHJ1biB3b3JrZWQgLSB3aGV0aGVyIHdoYXQgd2FzIGZvdW5kIHRyYWNlcyB0aGUNCiAgICAjIGFybXMgb2YgdGhlIGdhbGF4eSAtIGFuZCBpdCBzaG91bGQgbm90IGhhdmUgdG8gYmUgZHVnIG91dCBmcm9tIGFtb25nIHRoaXJ0eQ0KICAgICMgaW50ZXJtZWRpYXRlIGZpbGVzIHRvIGJlIGxvb2tlZCBhdC4NCiAgICBfb25lX3VwID0gb3MucGF0aC5kaXJuYW1lKF9vdXRkaXIpIG9yIF9vdXRkaXINCiAgICBvdXRfcG5nX2NvbG9yID0gb3MucGF0aC5qb2luKF9vbmVfdXAsIGYie29ial9uYW1lfV9WX3NvdXJjZXNfY29sb3JfZmlsdGVyZWQucG5nIikNCiAgICBwbHQuc2F2ZWZpZyhvdXRfcG5nX2NvbG9yLCBkcGk9MTUwKQ0KICAgIHBsdC5jbG9zZSgpDQogICAgcHJpbnQoZiJWLWJhbmQgY29sb3ItZmlsdGVyZWQgc291cmNlIG1hcCBzYXZlZCB0byB7b3V0X3BuZ19jb2xvcn0iKQ0KDQogICAgIyAtLS0gY3JlYXRlIENNRCBmb3IgdGhlIGNvbG9yLWZpbHRlcmVkIGNhdGFsb2cgLS0tDQogICAgbWFrZV9jbWQoZGZfY29sb3JfZmlsdGVyZWQsIG91dF9jc3ZfY29sb3IsICIobm8gc3RhcnMsIGNvbG9yIGZpbHRlcmVkKSIpDQoNCmVsc2U6DQogICAgcHJpbnQoIk1hZ19CIG9yIE1hZ19WIG5vdCBmb3VuZCBpbiBjbGVhbmVkIGNhdGFsb2cuIENvbG9yLWZpbHRlcmVkIG1hcCBub3QgY3JlYXRlZC4iKQ0KDQppZiBub3QgZGZfY29sb3JfZmlsdGVyZWQuZW1wdHk6DQogICAgIyBHYWxheHkgY2VudGVyIGluIHBpeGVscyAtIHRoZSBnYWxheHkncyBvd24gY2F0YWxvZ3VlZCBwb3NpdGlvbiB3aGVuIGl0DQogICAgIyBjYW4gYmUgbG9va2VkIHVwLCB0aGUgZnJhbWUncyBnZW9tZXRyaWMgY2VudHJlIG90aGVyd2lzZS4NCiAgICB4X2NlbnRlciwgeV9jZW50ZXIgPSBsb29rdXBfY2VudGVyX3BpeGVsKG9ial9uYW1lLCBoZHJfVikNCiAgICBpZiB4X2NlbnRlciBpcyBOb25lOg0KICAgICAgICB4X2NlbnRlciA9IGRhdGFfVi5zaGFwZVsxXSAvIDIuMA0KICAgICAgICB5X2NlbnRlciA9IGRhdGFfVi5zaGFwZVswXSAvIDIuMA0KDQogICAgIyBSYWRpYWwgZGlzdGFuY2UgaW4gcGl4ZWxzLiBYX1YvWV9WLCBub3QgWF9CL1lfQjogeF9jZW50ZXIveV9jZW50ZXIgYXJlIGluDQogICAgIyBWJ3MgcGl4ZWwgZ3JpZCAoVidzIG93biBXQ1MsIG9yIFYncyBmcmFtZSBjZW50cmUpLCBhbmQgQidzIG5hdGl2ZSBwaXhlbHMNCiAgICAjIGFyZSBub3QgZ2VuZXJhbGx5IHRoZSBzYW1lIGdyaWQgYXQgYWxsIC0gc2VlIHJlZ2lzdGVyX0Jfb250b19WKCkuDQogICAgZGZfY29sb3JfZmlsdGVyZWRbIlJhZGlhbF9EaXN0YW5jZV9weCJdID0gbnAuc3FydCgNCiAgICAgICAgKGRmX2NvbG9yX2ZpbHRlcmVkWyJYX1YiXSAtIHhfY2VudGVyKSoqMiArDQogICAgICAgIChkZl9jb2xvcl9maWx0ZXJlZFsiWV9WIl0gLSB5X2NlbnRlcikqKjINCiAgICApDQoNCiAgICAjIFBpeGVsIHNjYWxlIGZyb20gV0NTIChkZWcvcGl4ZWwgLT4gcmFkL3BpeGVsKQ0KICAgIHdjc19WID0gV0NTKGhkcl9WKQ0KICAgIHBpeHNjYWxlX2RlZyA9IG5wLm1lYW4obnAuYWJzKHdjc19WLnBpeGVsX3NjYWxlX21hdHJpeC5kaWFnb25hbCgpKSkNCiAgICBwaXhzY2FsZV9yYWQgPSBucC5kZWcycmFkKHBpeHNjYWxlX2RlZykNCg0KICAgICMgRGlzdGFuY2UgaW4gcGFyc2VjDQogICAgZGlzdGFuY2VfcGMgPSBkaXN0YW5jZSAqIDEuMGU2DQoNCiAgICAjIENvbnZlcnNpb24gZmFjdG9yIChwYy9waXhlbCkNCiAgICBweF90b19wYyA9IHBpeHNjYWxlX3JhZCAqIGRpc3RhbmNlX3BjDQoNCiAgICAjIEFkZCBkaXN0YW5jZXMgaW4gcGMNCiAgICBkZl9jb2xvcl9maWx0ZXJlZFsiUmFkaWFsX0Rpc3RhbmNlX3BjIl0gPSBkZl9jb2xvcl9maWx0ZXJlZFsiUmFkaWFsX0Rpc3RhbmNlX3B4Il0gKiBweF90b19wYw0KDQogICAgIyBTb3J0IGJ5IHJhZGlhbCBkaXN0YW5jZQ0KICAgIGRmX2NvbG9yX2ZpbHRlcmVkID0gZGZfY29sb3JfZmlsdGVyZWQuc29ydF92YWx1ZXMoIlJhZGlhbF9EaXN0YW5jZV9weCIpLnJlc2V0X2luZGV4KGRyb3A9VHJ1ZSkNCg0KICAgICMgU2F2ZSBDU1Ygd2l0aCBhbGwgZGlzdGFuY2VzDQogICAgb3V0X2Nzdl9jb2xvcl9yYWQgPSBvcy5wYXRoLmpvaW4oDQogICAgICAgIF9vdXRkaXIsDQogICAgICAgIGYie29ial9uYW1lfV9jYWxpYnJhdGVkX3Bob3RvbWV0cnlfbm9fc3RhcnNfY29sb3JfZmlsdGVyZWRfd2l0aF9yYWRpdXMuY3N2Ig0KICAgICkNCiAgICBkZl9jb2xvcl9maWx0ZXJlZC50b19jc3Yob3V0X2Nzdl9jb2xvcl9yYWQsIGluZGV4PUZhbHNlKQ0KICAgIHByaW50KGYiQ29sb3ItZmlsdGVyZWQgcGhvdG9tZXRyeSB3aXRoIHJhZGlhbCBkaXN0YW5jZXMgc2F2ZWQgdG8ge291dF9jc3ZfY29sb3JfcmFkfSIpDQoNCiAgICAjIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tDQogICAgIyBSYWRpYWwgZGVuc2l0eSBwcm9maWxlIGluIHBpeGVscw0KICAgICMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0NCiAgICBiaW5fd2lkdGhfcHggPSBTdGVwX3NpemUNCiAgICBtYXhfcl9weCA9IGRmX2NvbG9yX2ZpbHRlcmVkWyJSYWRpYWxfRGlzdGFuY2VfcHgiXS5tYXgoKQ0KICAgIGJpbnNfcHggPSBucC5hcmFuZ2UoMCwgbWF4X3JfcHggKyBiaW5fd2lkdGhfcHgsIGJpbl93aWR0aF9weCkNCg0KICAgIGNvdW50c19weCwgZWRnZXNfcHggPSBucC5oaXN0b2dyYW0oZGZfY29sb3JfZmlsdGVyZWRbIlJhZGlhbF9EaXN0YW5jZV9weCJdLCBiaW5zPWJpbnNfcHgpDQogICAgYXJlYXNfcHgyID0gbnAucGkgKiAoZWRnZXNfcHhbMTpdKioyIC0gZWRnZXNfcHhbOi0xXSoqMikNCiAgICBkZW5zaXRpZXNfcHggPSBjb3VudHNfcHggLyBhcmVhc19weDINCiAgICBiaW5fY2VudGVyc19weCA9IDAuNSAqIChlZGdlc19weFsxOl0gKyBlZGdlc19weFs6LTFdKQ0KDQogICAgIyBTYXZlIHByb2ZpbGUgKHB4KQ0KICAgIGRmX3Byb2ZpbGVfcHggPSBwZC5EYXRhRnJhbWUoew0KICAgICAgICAiUl9pbm5lcl9weCI6IGVkZ2VzX3B4WzotMV0sDQogICAgICAgICJSX291dGVyX3B4IjogZWRnZXNfcHhbMTpdLA0KICAgICAgICAiUl9jZW50ZXJfcHgiOiBiaW5fY2VudGVyc19weCwNCiAgICAgICAgIk5fc291cmNlcyI6IGNvdW50c19weCwNCiAgICAgICAgIkFubnVsdXNfYXJlYV9weDIiOiBhcmVhc19weDIsDQogICAgICAgICJEZW5zaXR5X3Blcl9weDIiOiBkZW5zaXRpZXNfcHgNCiAgICB9KQ0KICAgIG91dF9jc3ZfcHJvZmlsZV9weCA9IG9zLnBhdGguam9pbihfb3V0ZGlyLCBmIntvYmpfbmFtZX1fcmFkaWFsX2RlbnNpdHlfcHJvZmlsZS5jc3YiKQ0KICAgIGRmX3Byb2ZpbGVfcHgudG9fY3N2KG91dF9jc3ZfcHJvZmlsZV9weCwgaW5kZXg9RmFsc2UpDQogICAgcHJpbnQoZiJSYWRpYWwgZGVuc2l0eSBwcm9maWxlIChweCkgc2F2ZWQgdG8ge291dF9jc3ZfcHJvZmlsZV9weH0iKQ0KDQogICAgIyBTdGVwIHByb2ZpbGUgKHB4KQ0KICAgIHN0ZXBfcl9weCwgc3RlcF9kZW5zaXR5X3B4LCBzdGVwX2NvdW50c19weCA9IFtdLCBbXSwgW10NCiAgICBmb3IgaSBpbiByYW5nZShsZW4oZGVuc2l0aWVzX3B4KSk6DQogICAgICAgIHJfaW4sIHJfb3V0ID0gZWRnZXNfcHhbaV0sIGVkZ2VzX3B4W2krMV0NCiAgICAgICAgc3RlcF9yX3B4LmV4dGVuZChbcl9pbiwgcl9vdXRdKQ0KICAgICAgICBzdGVwX2RlbnNpdHlfcHguZXh0ZW5kKFtkZW5zaXRpZXNfcHhbaV0sIGRlbnNpdGllc19weFtpXV0pDQogICAgICAgIHN0ZXBfY291bnRzX3B4LmV4dGVuZChbY291bnRzX3B4W2ldLCBjb3VudHNfcHhbaV1dKQ0KDQogICAgZGZfc3RlcF9weCA9IHBkLkRhdGFGcmFtZSh7DQogICAgICAgICJSX3N0ZXBfcHgiOiBzdGVwX3JfcHgsDQogICAgICAgICJEZW5zaXR5X3N0ZXBfcGVyX3B4MiI6IHN0ZXBfZGVuc2l0eV9weCwNCiAgICAgICAgIk5fc291cmNlc19zdGVwIjogc3RlcF9jb3VudHNfcHgNCiAgICB9KQ0KICAgIG91dF9jc3Zfc3RlcF9weCA9IG9zLnBhdGguam9pbihfb3V0ZGlyLCBmIntvYmpfbmFtZX1fcmFkaWFsX2RlbnNpdHlfcHJvZmlsZV9zdGVwLmNzdiIpDQogICAgZGZfc3RlcF9weC50b19jc3Yob3V0X2Nzdl9zdGVwX3B4LCBpbmRleD1GYWxzZSkNCiAgICBwcmludChmIlJhZGlhbCBkZW5zaXR5IHN0ZXAgcHJvZmlsZSAocHgpIHNhdmVkIHRvIHtvdXRfY3N2X3N0ZXBfcHh9IikNCg0KICAgIHBsdC5maWd1cmUoZmlnc2l6ZT0oOCw2KSkNCiAgICBwbHQuc3RlcChzdGVwX3JfcHgsIHN0ZXBfZGVuc2l0eV9weCwgd2hlcmU9InBvc3QiLCBjb2xvcj0iYmx1ZSIsIGxpbmV3aWR0aD0yKQ0KICAgIHBsdC54bGFiZWwoIlJhZGlhbCBkaXN0YW5jZSBbcGl4ZWxzXSIpDQogICAgcGx0LnlsYWJlbCgiU291cmNlIGRlbnNpdHkgWzEvcGl4ZWzCsl0iKQ0KICAgIHBsdC50aXRsZShmIlJhZGlhbCBkZW5zaXR5IHN0ZXAgcHJvZmlsZSBvZiB7b2JqX25hbWV9IChwaXhlbHMpIikNCiAgICBvdXRfcG5nX3N0ZXBfcHggPSBvcy5wYXRoLmpvaW4oX291dGRpciwgZiJ7b2JqX25hbWV9X3JhZGlhbF9kZW5zaXR5X3Byb2ZpbGVfc3RlcC5wbmciKQ0KICAgIHBsdC5zYXZlZmlnKG91dF9wbmdfc3RlcF9weCwgZHBpPTE1MCkNCiAgICBwbHQuY2xvc2UoKQ0KICAgIHByaW50KGYiUmFkaWFsIGRlbnNpdHkgc3RlcCBwbG90IChweCkgc2F2ZWQgdG8ge291dF9wbmdfc3RlcF9weH0iKQ0KDQogICAgIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQ0KICAgICMgUmFkaWFsIGRlbnNpdHkgcHJvZmlsZSBpbiBwYw0KICAgICMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0NCiAgICBiaW5fd2lkdGhfcGMgPSBiaW5fd2lkdGhfcHggKiBweF90b19wYw0KICAgIG1heF9yX3BjID0gZGZfY29sb3JfZmlsdGVyZWRbIlJhZGlhbF9EaXN0YW5jZV9wYyJdLm1heCgpDQogICAgYmluc19wYyA9IG5wLmFyYW5nZSgwLCBtYXhfcl9wYyArIGJpbl93aWR0aF9wYywgYmluX3dpZHRoX3BjKQ0KDQogICAgY291bnRzX3BjLCBlZGdlc19wYyA9IG5wLmhpc3RvZ3JhbShkZl9jb2xvcl9maWx0ZXJlZFsiUmFkaWFsX0Rpc3RhbmNlX3BjIl0sIGJpbnM9Ymluc19wYykNCiAgICBhcmVhc19wYzIgPSBucC5waSAqIChlZGdlc19wY1sxOl0qKjIgLSBlZGdlc19wY1s6LTFdKioyKQ0KICAgIGRlbnNpdGllc19wYyA9IGNvdW50c19wYyAvIGFyZWFzX3BjMg0KICAgIGJpbl9jZW50ZXJzX3BjID0gMC41ICogKGVkZ2VzX3BjWzE6XSArIGVkZ2VzX3BjWzotMV0pDQoNCiAgICAjIFNhdmUgcHJvZmlsZSAocGMpDQogICAgZGZfcHJvZmlsZV9wYyA9IHBkLkRhdGFGcmFtZSh7DQogICAgICAgICJSX2lubmVyX3BjIjogZWRnZXNfcGNbOi0xXSwNCiAgICAgICAgIlJfb3V0ZXJfcGMiOiBlZGdlc19wY1sxOl0sDQogICAgICAgICJSX2NlbnRlcl9wYyI6IGJpbl9jZW50ZXJzX3BjLA0KICAgICAgICAiTl9zb3VyY2VzIjogY291bnRzX3BjLA0KICAgICAgICAiQW5udWx1c19hcmVhX3BjMiI6IGFyZWFzX3BjMiwNCiAgICAgICAgIkRlbnNpdHlfcGVyX3BjMiI6IGRlbnNpdGllc19wYw0KICAgIH0pDQogICAgb3V0X2Nzdl9wcm9maWxlX3BjID0gb3MucGF0aC5qb2luKF9vdXRkaXIsIGYie29ial9uYW1lfV9yYWRpYWxfZGVuc2l0eV9wcm9maWxlX3BjLmNzdiIpDQogICAgZGZfcHJvZmlsZV9wYy50b19jc3Yob3V0X2Nzdl9wcm9maWxlX3BjLCBpbmRleD1GYWxzZSkNCiAgICBwcmludChmIlJhZGlhbCBkZW5zaXR5IHByb2ZpbGUgKHBjKSBzYXZlZCB0byB7b3V0X2Nzdl9wcm9maWxlX3BjfSIpDQoNCiAgICAjIFN0ZXAgcHJvZmlsZSAocGMpDQogICAgc3RlcF9yX3BjLCBzdGVwX2RlbnNpdHlfcGMsIHN0ZXBfY291bnRzX3BjID0gW10sIFtdLCBbXQ0KICAgIGZvciBpIGluIHJhbmdlKGxlbihkZW5zaXRpZXNfcGMpKToNCiAgICAgICAgcl9pbiwgcl9vdXQgPSBlZGdlc19wY1tpXSwgZWRnZXNfcGNbaSsxXQ0KICAgICAgICBzdGVwX3JfcGMuZXh0ZW5kKFtyX2luLCByX291dF0pDQogICAgICAgIHN0ZXBfZGVuc2l0eV9wYy5leHRlbmQoW2RlbnNpdGllc19wY1tpXSwgZGVuc2l0aWVzX3BjW2ldXSkNCiAgICAgICAgc3RlcF9jb3VudHNfcGMuZXh0ZW5kKFtjb3VudHNfcGNbaV0sIGNvdW50c19wY1tpXV0pDQoNCiAgICBkZl9zdGVwX3BjID0gcGQuRGF0YUZyYW1lKHsNCiAgICAgICAgIlJfc3RlcF9wYyI6IHN0ZXBfcl9wYywNCiAgICAgICAgIkRlbnNpdHlfc3RlcF9wZXJfcGMyIjogc3RlcF9kZW5zaXR5X3BjLA0KICAgICAgICAiTl9zb3VyY2VzX3N0ZXAiOiBzdGVwX2NvdW50c19wYw0KICAgIH0pDQogICAgb3V0X2Nzdl9zdGVwX3BjID0gb3MucGF0aC5qb2luKF9vdXRkaXIsIGYie29ial9uYW1lfV9yYWRpYWxfZGVuc2l0eV9wcm9maWxlX3N0ZXBfcGMuY3N2IikNCiAgICBkZl9zdGVwX3BjLnRvX2NzdihvdXRfY3N2X3N0ZXBfcGMsIGluZGV4PUZhbHNlKQ0KICAgIHByaW50KGYiUmFkaWFsIGRlbnNpdHkgc3RlcCBwcm9maWxlIChwYykgc2F2ZWQgdG8ge291dF9jc3Zfc3RlcF9wY30iKQ0KDQogICAgcGx0LmZpZ3VyZShmaWdzaXplPSg4LDYpKQ0KICAgIHBsdC5zdGVwKHN0ZXBfcl9wYywgc3RlcF9kZW5zaXR5X3BjLCB3aGVyZT0icG9zdCIsIGNvbG9yPSJncmVlbiIsIGxpbmV3aWR0aD0yKQ0KICAgIHBsdC54bGFiZWwoIlJhZGlhbCBkaXN0YW5jZSBbcGNdIikNCiAgICBwbHQueWxhYmVsKCJTb3VyY2UgZGVuc2l0eSBbMS9wY8KyXSIpDQogICAgcGx0LnRpdGxlKGYiUmFkaWFsIGRlbnNpdHkgc3RlcCBwcm9maWxlIG9mIHtvYmpfbmFtZX0gKHBjKSIpDQogICAgb3V0X3BuZ19zdGVwX3BjID0gb3MucGF0aC5qb2luKF9vdXRkaXIsIGYie29ial9uYW1lfV9yYWRpYWxfZGVuc2l0eV9wcm9maWxlX3N0ZXBfcGMucG5nIikNCiAgICBwbHQuc2F2ZWZpZyhvdXRfcG5nX3N0ZXBfcGMsIGRwaT0xNTApDQogICAgcGx0LmNsb3NlKCkNCiAgICBwcmludChmIlJhZGlhbCBkZW5zaXR5IHN0ZXAgcGxvdCAocGMpIHNhdmVkIHRvIHtvdXRfcG5nX3N0ZXBfcGN9IikNCg0KDQogICAgIyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQ0KICAgICMgU3RhZ2UgNiAtIGRlcHJvamVjdGlvbiBhbmQgdGhlIHNoYXJlZCByYWRpYWwgYW5hbHlzaXMNCiAgICAjID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09DQogICAgIw0KICAgICMgRXZlcnl0aGluZyBhYm92ZSBtZWFzdXJlZCB0aGUga25vdHMgYXMgdGhleSBhcHBlYXIgb24gdGhlIHNreS4gQnV0IGEgc3BpcmFsDQogICAgIyBnYWxheHkgaXMgYSBmbGF0IGRpc2sgc2VlbiBhdCBhbiBhbmdsZSwgc28gYSBkaXN0YW5jZSBtZWFzdXJlZCBpbiB0aGUgaW1hZ2UNCiAgICAjIGlzIG5vdCB0aGUgZGlzdGFuY2UgaW5zaWRlIHRoZSBnYWxheHkuIFRpbHRlZCwgYSByb3VuZCBkaXNrIHByb2plY3RzIHRvIGFuDQogICAgIyBlbGxpcHNlLCBhbmQgdGhlIGRpcmVjdGlvbiBhY3Jvc3MgdGhlIGVsbGlwc2UgaXMgc3F1YXNoZWQgd2hpbGUgdGhlIGRpcmVjdGlvbg0KICAgICMgYWxvbmcgaXQgaXMgbm90Lg0KICAgICMNCiAgICAjIFRoaXMgc3RhZ2Ugc3RyZXRjaGVzIHRoZSBzcXVhc2hlZCBkaXJlY3Rpb24gYmFjayBvdXQgYW5kIHJlY29tcHV0ZXMgdGhlDQogICAgIyBwcm9maWxlLiBJdCBkb2VzIHNvIHNldmVyYWwgZGlmZmVyZW50IHdheXMgLSB1bmNvcnJlY3RlZCwgZnJvbSB0aGUgSHlwZXJMRURBDQogICAgIyBjYXRhbG9ndWUsIGFuZCBmcm9tIHRoZSBrbm90cyB0aGVtc2VsdmVzIC0gYmVjYXVzZSB0aGUgcHVibGlzaGVkIGNvbXBhcmlzb24NCiAgICAjIG9mIGRlcHJvamVjdGlvbiBtZXRob2RzIGZvdW5kIG5vbmUgdG8gYmUgcmVsaWFibHkgYmV0dGVyIHRoYW4gdGhlIG90aGVycyBhbmQNCiAgICAjIHJlY29tbWVuZGVkIGFwcGx5aW5nIHNldmVyYWwgYW5kIGNvbXBhcmluZy4gVGhlIHNwcmVhZCBiZXR3ZWVuIHRoZSBjdXJ2ZXMgaXMNCiAgICAjIHRoZSB1bmNlcnRhaW50eSBvbiB0aGUgcmFkaWkuDQogICAgIw0KICAgICMgSXQgYWxzbyByZWNvbXB1dGVzIHRoZSByaW5nIGFyZWFzIGhvbmVzdGx5LiBUaGUgcHJvZmlsZSBhYm92ZSBkaXZpZGVzIGVhY2gNCiAgICAjIGNvdW50IGJ5IHRoZSBmdWxsIG1hdGhlbWF0aWNhbCBhcmVhIG9mIGl0cyByaW5nLCBwaSoocl9vdXReMiAtIHJfaW5eMikuIEEgcmluZw0KICAgICMgd2lkZXIgdGhhbiB0aGUgZnJhbWUgaXMgb25seSBwYXJ0bHkgY292ZXJlZCBieSB0aGUgaW1hZ2UsIHNvIGRpdmlkaW5nIGJ5IHRoZQ0KICAgICMgZnVsbCBhcmVhIG1ha2VzIHRoZSBvdXRlciByaW5ncyBsb29rIGVtcHRpZXIgdGhhbiB0aGV5IGFyZS4gVGhlIG5ldyBwcm9maWxlDQogICAgIyBtZWFzdXJlcyB0aGUgYXJlYSBhY3R1YWxseSBvYnNlcnZlZCBhbmQgZmxhZ3MgYW55IHJpbmcgdGhlIGZyYW1lIGhhcyBjbGlwcGVkLg0KICAgICMgQm90aCBhcmVhcyBhcmUgd3JpdHRlbiB0byB0aGUgQ1NWIHNvIHRoZSB0d28gY2FuIGJlIGNvbXBhcmVkLg0KDQogICAgdHJ5Og0KICAgICAgICBwYXNzICAjIGZyb20gcmFkaWFsX3Byb2ZpbGVzLCBpbmxpbmVkIGFib3ZlDQogICAgICAgIGZyb20gYXN0cm9weS53Y3MgaW1wb3J0IFdDUyBhcyBfV0NTDQoNCiAgICAgICAgIyBPcmllbnRhdGlvbiBzdHJhaWdodCBmcm9tIHRoZSBXQ1MsIHJhdGhlciB0aGFuIGFzc3VtaW5nIG5vcnRoIGlzIHVwLg0KICAgICAgICAjIEFzc3VtaW5nIGl0IHdoZW4gaXQgaXMgZmFsc2Ugcm90YXRlcyB0aGUgd2hvbGUgZGVwcm9qZWN0aW9uIHNpbGVudGx5Lg0KICAgICAgICBfbm9ydGgsIF9taXJyb3JlZCA9IDkwLjAsIEZhbHNlDQogICAgICAgIHRyeToNCiAgICAgICAgICAgIF9jZCA9IF9XQ1MoaGRyX1YpLmNlbGVzdGlhbC5waXhlbF9zY2FsZV9tYXRyaXgNCiAgICAgICAgICAgIF9ub3J0aCA9IGZsb2F0KG5wLmRlZ3JlZXMobnAuYXJjdGFuMihfY2RbMSwgMV0sIF9jZFswLCAxXSkpKSAlIDM2MC4wDQogICAgICAgICAgICBfZWFzdCA9IGZsb2F0KG5wLmRlZ3JlZXMobnAuYXJjdGFuMihfY2RbMSwgMF0sIF9jZFswLCAwXSkpKSAlIDM2MC4wDQogICAgICAgICAgICBfbWlycm9yZWQgPSAoKF9lYXN0IC0gX25vcnRoKSAlIDM2MC4wKSA+IDE4MC4wDQogICAgICAgICAgICBwcmludChmIltkZXByb2plY3RdIG9yaWVudGF0aW9uIGZyb20gV0NTOiBub3J0aCB7X25vcnRoOi4xZn0gZGVnIGZyb20gK3gsICINCiAgICAgICAgICAgICAgICAgIGYieydtaXJyb3JlZCcgaWYgX21pcnJvcmVkIGVsc2UgJ25vdCBtaXJyb3JlZCd9IikNCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBfZXhjOg0KICAgICAgICAgICAgcHJpbnQoZiJbZGVwcm9qZWN0XSBjb3VsZCBub3QgcmVhZCB0aGUgb3JpZW50YXRpb24gZnJvbSB0aGUgV0NTICh7X2V4Y30pOyAiDQogICAgICAgICAgICAgICAgICBmImFzc3VtaW5nIG5vcnRoIGlzIHVwIikNCg0KICAgICAgICBfZnJhbWUgPSAoMC4wLCBmbG9hdChkYXRhX1Yuc2hhcGVbMV0pLCAwLjAsIGZsb2F0KGRhdGFfVi5zaGFwZVswXSkpDQogICAgICAgIF9wcmVmaXggPSBvcy5wYXRoLmpvaW4oX291dGRpciwgZiJ7b2JqX25hbWV9X2RlcHJvamVjdGVkIikNCg0KICAgICAgICBwcmludCgiIikNCiAgICAgICAgcHJpbnQoIkRlcHJvamVjdGluZyBhbmQgcmVidWlsZGluZyB0aGUgcmFkaWFsIHByb2ZpbGUuLi4iKQ0KICAgICAgICBydW5fcHJvZmlsZXMoDQogICAgICAgICAgICBkZl9jb2xvcl9maWx0ZXJlZFsiWF9WIl0udG9fbnVtcHkoZHR5cGU9ZmxvYXQpLA0KICAgICAgICAgICAgZGZfY29sb3JfZmlsdGVyZWRbIllfViJdLnRvX251bXB5KGR0eXBlPWZsb2F0KSwNCiAgICAgICAgICAgICh4X2NlbnRlciwgeV9jZW50ZXIpLCBfZnJhbWUsIF9wcmVmaXgsDQogICAgICAgICAgICBmIkJsdWUga25vdHMgLSB7b2JqX25hbWV9IiwNCiAgICAgICAgICAgIGdhbGF4eT1vYmpfbmFtZSwNCiAgICAgICAgICAgIG5vcnRoX2FuZ2xlPV9ub3J0aCwgbWlycm9yZWQ9X21pcnJvcmVkLA0KICAgICAgICAgICAgc2VjdG9yX21ldGhvZD1UcnVlLA0KICAgICAgICAgICAgcmluZ3M9MjAsIHNjYWxlPXB4X3RvX3BjLCB1bml0PSJwYyIpDQogICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBfZXhjOg0KICAgICAgICBwcmludChmIltkZXByb2plY3RdIHRoZSBkZXByb2plY3Rpb24gc3RhZ2UgZmFpbGVkICh7dHlwZShfZXhjKS5fX25hbWVfX306ICINCiAgICAgICAgICAgICAgZiJ7X2V4Y30pLiBFdmVyeXRoaW5nIGFib3ZlIHdhcyBzdGlsbCB3cml0dGVuLiIpDQoNCg0KZWxzZToNCiAgICBwcmludCgiTm8gY29sb3ItZmlsdGVyZWQgc291cmNlcyBhdmFpbGFibGUgdG8gY29tcHV0ZSByYWRpYWwgZGlzdGFuY2VzLiIpDQo="


def _unpack_cluster_tool():
    """Write the carried copy of the cluster tool out, and return its path.

    It goes to a temporary folder of its own that lasts as long as this process,
    so nothing is left behind and nothing beside this file is needed. Its outputs
    are unaffected: that tool writes next to the frames it is given, not next to
    itself.
    """
    import atexit
    import base64
    import shutil
    import tempfile

    folder = tempfile.mkdtemp(prefix="blue_clusters_")
    atexit.register(shutil.rmtree, folder, True)
    path = os.path.join(folder, "Blue_Clusters_From_Images.py")
    with open(path, "wb") as handle:
        handle.write(base64.b64decode(_CLUSTER_TOOL_SOURCE))
    print("    (using the copy of the cluster tool carried inside this file)",
          flush=True)
    return path


if __name__ == "__main__":
    sys.exit(main())
