"""
Star_Formation_All_In_One.py - both tracers of star formation, and the link between them.

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
    python Star_Formation_All_In_One.py --folder "D:/NGC4212" --galaxy "NGC 4212"
        --distance-mpc 16.7

    The folder holds the frames; which file is which is read from the headers, and
    what was chosen is printed before anything is measured. Naming the four files
    directly still works:

    python Star_Formation_All_In_One.py --b B.fits --v V.fits --red R.fits --halpha Ha.fits
        --galaxy "NGC 4212" --distance-mpc 16.7

    # if the blue knots have already been measured by Blue_Clusters_From_Images.py
    python Star_Formation_All_In_One.py --b B.fits --v V.fits --red R.fits --halpha Ha.fits
        --galaxy "NGC 4212" --knots-csv result_CMD_NGC_4212_<date>/NGC_4212_..._with_radius.csv
"""



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

_DIFFERENCE = re.compile(
    r"(?:h[\s_-]*al(?:pha|fa)|(?<![a-z0-9])ha)[^/\\]*?-\s*(?:r|rp|rc)(?![a-z0-9])"
    r"|minus|cont[\s_-]*sub", re.I)

def is_difference_name(name):
    """True when a file name says the frame is one filter minus another, such
    as H-alpha with the R continuum taken off ("M83_HaX12-R.fts")."""
    return bool(_DIFFERENCE.search(str(name)))

_PRODUCT_HISTORY = re.compile(
    r"multiplied\s+by|subtracted\b.*\.(?:fits?|fts)\b", re.I)

def is_product_header(header):
    """True when a HISTORY line says the frame was scaled by a factor or had
    another FITS file subtracted from it. A constant sky ("Subtracted 9.0") does
    not count."""
    return any(_PRODUCT_HISTORY.search(line) for line in history_lines(header))

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

    if is_product_header(header):
        # The header says it is this frame scaled, or another frame taken off
        # it ("Multiplied by 11.5", "Subtracted ...R_SUB_Median_of_3"): not a
        # frame through one filter, whatever its name or FILTER keyword says.
        role = None
        source = "its history (a scaled or subtracted product - not used)"
    elif role is None and is_difference_name(os.path.basename(path)):
        # "HaX12-R" is H-alpha with 12 x R taken off: a product, not a frame
        # through either filter. Naming it R would put it in as the continuum.
        source = "the file name (a difference of two frames - not used)"
    elif role is None:
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
        # Every run now writes a folder of its own - result_CMD_<galaxy>_<date>,
        # result_HII_..., result_StarFormation_... - so anything whose name
        # starts with "result" is output, not data.
        dirs[:] = [d for d in dirs if d.lower() not in OUTPUT_FOLDERS
                   and not d.lower().startswith("result")]
        for name in sorted(files):
            if not name.lower().endswith(FITS_SUFFIXES):
                continue
            # What these tools write next to their results - a background-
            # subtracted copy, a frame with a borrowed solution - is not data.
            if re.search(r"_(bgsub|solved)\.fits?$", name, re.I):
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
        # "_r" is VizieR's own distance of each row from the position asked
        # about. The rows come back in catalogue order, not nearest first, so
        # without it the first row could be a neighbouring galaxy inside the
        # search circle and its distance would be used without a word.
        hit = Vizier(columns=["**", "_r"], row_limit=20).query_region(
            coord, radius=2 * u.arcmin, catalog=DISTANCE_CATALOG)
        if not hit:
            return None
        table = hit[0]
        if "_r" in table.colnames and len(table) > 1:
            table = table[np.argsort(np.asarray(table["_r"], dtype=float))]
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

def header_number(header, keys):
    """The first of these header keywords that holds a positive number, or None."""
    for key in keys:
        try:
            value = float(header.get(key))
        except (TypeError, ValueError):
            continue
        if value == value and value > 0:
            return value
    return None

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
        # What the photometry needs to know about the detector, carried along
        # so that a frame this pipeline writes still says it.
        "saturate": header_number(header, ("SATURATE", "MAXLIN", "SATLEVEL")),
        # Electrons per ADU. EGAIN first: on some cameras GAIN is the gain
        # SETTING (51 on the Kinneret colour camera), not electrons per ADU.
        "gain": next((g for g in (header_number(header, (k,)) for k in
                                  ("EGAIN", "GAIN")) if g and g < 20), None),
        "rdnoise": header_number(header, ("RDNOISE", "READNOIS", "RON")),
        "ncombine": int(header_number(header, ("NCOMBINE",)) or 1),
        "combtype": str(header.get("COMBTYPE", "")),
        "date": str(header.get("DATE-OBS", "")),
    }
    hdul.close()

    if wcs is None and donor_path:
        reference, ref_wcs, _, ref_info = load_frame(donor_path)
        if ref_wcs is not None:
            try:
                wcs, diag = wcs_from_reference(data, reference, ref_wcs)
            except Exception as exc:
                # astroalign finds no star pattern in common - typically a frame
                # with the stars taken out, such as H-alpha minus continuum.
                print("  {}: could not fit a plate solution against {} ({}: {})".format(
                    os.path.basename(path), os.path.basename(donor_path),
                    type(exc).__name__, exc), flush=True)
                return data, None, bpm, info
            print(describe_wcs_fit(os.path.basename(path), diag))
            if info["pixel_scale"] is None:
                info["pixel_scale"] = ref_info["pixel_scale"]
            if info["filter"] == "?":
                info["filter"] = ref_info["filter"]
            # A combined frame written by another program often keeps no header
            # at all. The camera and the exposure length are those of the solved
            # exposure it came from, so they are taken from there, and said so.
            borrowed = []
            for key, card in (("exptime", "EXPTIME"), ("saturate", "SATURATE"),
                              ("gain", "GAIN"), ("rdnoise", "RDNOISE"),
                              ("date", "DATE-OBS"), ("object", "OBJECT")):
                if card in header or (key in ("saturate", "gain", "rdnoise")
                                      and info[key] is not None):
                    continue
                if key == "object" and info[key] != "?":
                    continue
                if ref_info.get(key) not in (None, "", "?"):
                    info[key] = ref_info[key]
                    borrowed.append(card)
            if borrowed:
                print("  {}: no {} in its header; taken from {}".format(
                    os.path.basename(path), ", ".join(borrowed),
                    os.path.basename(donor_path)), flush=True)

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
        return positions, {"n_in": 0, "n_out": 0, "why": "no stars to centre"}

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
    # Every exposure is scaled to the exposure time of the first before the
    # median is taken. A median of a 60 s and a 120 s frame that were not
    # scaled is neither of them: each pixel takes whichever value happens to
    # fall in the middle. Before scaling, pixels at the detector's saturation
    # level are blanked, so that a star saturated in the long exposures is
    # measured from the short ones.
    ref_exptime = float(ref_info.get("exptime") or 1.0)
    sat_levels, sat_masks = [], []

    def scaled(frame, info, name):
        exptime = float(info.get("exptime") or ref_exptime)
        ratio = ref_exptime / exptime
        sat = info.get("saturate")
        if sat:
            frame = np.array(frame, dtype=np.float32, copy=True)
            with np.errstate(invalid="ignore"):
                hit = frame >= SATURATION_FRACTION * sat
            frame[hit] = np.nan
            sat_masks.append(hit)
            sat_levels.append(sat * ratio)
        else:
            sat_levels.append(None)
        if abs(ratio - 1.0) > 1e-6:
            print("  {}: {:.0f} s, scaled by {:.3f} to {:.0f} s".format(
                name, exptime, ratio, ref_exptime))
            frame = frame * np.float32(ratio)
        return frame

    stack = [scaled(reference, ref_info, os.path.basename(paths[0]))]
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

        stack.append(scaled(data, info, name))
        total_exptime += float(info.get("exptime") or 0.0)

    ny, nx = reference.shape
    combined = np.empty((ny, nx), dtype=np.float32)
    for y0 in range(0, ny, band_rows):
        y1 = min(y0 + band_rows, ny)
        band = np.stack([frame[y0:y1] for frame in stack], axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            combined[y0:y1] = np.nanmedian(band, axis=0)
            # A pixel saturated in every exposure has nothing left to take the
            # median of. It is set to the saturation level, so that it stays a
            # bright pixel the photometry flags as saturated, not a hole.
            if sat_masks:
                hole = ~np.isfinite(combined[y0:y1])
                hole &= np.any(np.stack([m[y0:y1] for m in sat_masks], axis=0), axis=0)
                if hole.any():
                    combined[y0:y1][hole] = max(s for s in sat_levels if s)

    print("  median of {} exposures, total exposure {:.0f}s, noise lower by about "
          "a factor of {:.1f}".format(len(stack), total_exptime, len(stack) ** 0.5))

    # The median sits at the level of ONE exposure of the first frame's length,
    # so that is what EXPTIME says. It used to say the total, which made the
    # header claim several times the exposure the pixel values have. The total
    # is kept in TOTEXP, and NCOMBINE says how many frames went in - the noise
    # of a median of N frames is that of one frame with N x 2/pi times the gain.
    info = dict(ref_info)
    info["exptime"] = ref_exptime
    info["total_exptime"] = total_exptime
    info["ncombine"] = len(stack)
    info["combtype"] = "median"
    if all(s for s in sat_levels):
        # Pixels below the level of the shortest exposure came from at least
        # one unsaturated frame.
        info["saturate"] = max(sat_levels)
    else:
        info["saturate"] = None
    # Scaling a longer exposure down lowers its noise along with its counts,
    # as a higher gain would. Written so that GAIN x NCOMBINE is the gain of the
    # total exposure expressed at this frame's level.
    if info.get("gain"):
        info["gain"] = float(info["gain"]) * total_exptime / ref_exptime / len(stack)
    return combined, ref_wcs, ref_bpm, info

SATURATION_FRACTION = 0.8   # a pixel above this fraction of SATURATE counts as saturated

def new_run_folder(base, tool, galaxy):
    """A folder of its own for this run: result_<tool>_<galaxy>_<date>_<time>.

    Every run used to write into the same place under the same names, so a
    second run overwrote the first, and a run that stopped half way left the
    previous run's files sitting beside its own, looking current. A new folder
    each time means nothing a run reads or shows can be left over from another.
    """
    import datetime
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", (galaxy or "galaxy").strip()) or "galaxy"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base or os.getcwd(), "result_{}_{}_{}".format(tool, name, stamp))
    os.makedirs(path, exist_ok=True)
    return path

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
        for key, word in (("saturate", "SATURATE"), ("gain", "GAIN"),
                          ("rdnoise", "RDNOISE"), ("total_exptime", "TOTEXP")):
            if info.get(key):
                header[word] = float(info[key])
        if (info.get("ncombine") or 1) > 1:
            header["NCOMBINE"] = int(info["ncombine"])
            header["COMBTYPE"] = str(info.get("combtype") or "median")
        if info.get("date"):
            header["DATE-OBS"] = str(info["date"])
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
# Star_Formation_All_In_One.py
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
                          ebv=None, out_dir=None):
    """Run Blue_Clusters_From_Images.py and return the path to its knot catalogue.

    It is run as a separate process rather than imported, and given everything on
    its command line. Its answers used to be piped in as typed replies in a fixed
    order - galaxy, distance, reddening, B, V - but that tool's first question is
    the folder of frames, and the distance and reddening are only asked when the
    lookup fails. The replies landed on the wrong questions: the galaxy's name was
    taken as a folder, the distance as the galaxy's name, and the run failed or
    measured the wrong thing. Named flags cannot be taken for one another.
    """
    tool = os.path.join(script_dir, "Blue_Clusters_From_Images.py")
    if not os.path.exists(tool):
        tool = _unpack_cluster_tool()

    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(v_path)),
                                      "Blue_clusters")
    cmd = [sys.executable, tool, "--b", b_path, "--v", v_path, "--galaxy", galaxy,
           "--out-dir", out_dir, "--yes"]
    if distance_mpc:
        cmd += ["--distance-mpc", repr(float(distance_mpc))]
    # The reddening is passed rather than looked up again, so that the number this
    # run reports is the number that tool actually used.
    if ebv is not None:
        cmd += ["--ebv", "{:.4f}".format(ebv)]

    print("Running Blue_Clusters_From_Images.py for the blue knots")
    print("  (this is the long part - photometry, catalogue calibration, colour cut)")
    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=script_dir)
    for line in proc.stdout.splitlines():
        if any(k in line for k in ("[calib]", "B-V =", "V   =", "[wcs]", "[lookup]",
                                   "Blue knots", "STOPPED", "WARNING", "Error",
                                   "error")):
            print("    " + line)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-2000:])
        raise RuntimeError("the blue-cluster tool exited with an error")

    hits = glob.glob(os.path.join(out_dir, "*color_filtered_with_radius.csv"))
    if not hits:
        raise FileNotFoundError(
            "the blue-cluster tool ran but produced no colour-filtered catalogue in "
            + out_dir)
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
             "pixel_scale": info.get("pixel_scale"),
             # what the photometry needs to set its errors and find the
             # saturated stars
             "saturate": info.get("saturate"), "gain": info.get("gain"),
             "rdnoise": info.get("rdnoise"), "ncombine": info.get("ncombine"),
             "combtype": info.get("combtype"), "date": info.get("date")},
            history=("Resampled onto the V grid by Star_Formation_All_In_One",)))
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
    # its result_CMD_<galaxy>_<date> folder, and reusing it turns a run of several minutes into one of about
    # thirty seconds.
    if ask_yes_no("Do you already have a blue-knot catalogue from the "
                  "blue-cluster tool?", False):
        argv += ["--knots-csv", ask_file("Path to that CSV")]

    argv += ["--threshold", repr(ask("Detection threshold for the HII regions, in "
                                     "units of the background noise", 4.0, float))]
    argv += ["--rings", repr(ask("Number of radial rings", 20, int))]

    prefix = ask("Where to write the output, name without extension "
                 "(blank for a new result_StarFormation_<galaxy>_<date> folder "
                 "beside the frames)", "")
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
            n = int(inf.get("ncombine") or 1)
            if n > 1:
                print("  {:<8} {:<9} {:>6.0f}s each, median of {} ({:.0f}s in all)".format(
                    nm, inf["filter"], inf["exptime"], n,
                    inf.get("total_exptime") or n * inf["exptime"]))
            else:
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
    # A new folder for every run, beside the frames, unless --out-prefix says
    # otherwise: result_StarFormation_<galaxy>_<date>_<time>, with the three
    # named folders inside it. They used to be created in the folder holding the
    # frames itself, and every run overwrote the last.
    _run_dir = (None if args.out_prefix else new_run_folder(
        args.folder or os.path.dirname(os.path.abspath(args.v)), "StarFormation",
        args.galaxy))
    if _run_dir:
        print("Writing everything to " + _run_dir)
    prefix = output_prefix(args.out_prefix, _run_dir, args.galaxy)
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
        "Made by Star_Formation_All_In_One.py"))
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
                                          ebv=_ebv, out_dir=_dir_clusters)
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

_CLUSTER_TOOL_SOURCE = "IyAtKi0gY29kaW5nOiB1dGYtOCAtKi0KIiIiCkNyZWF0ZWQgYnk6IERyLiBCb2F6IFJvbiBab2hhcgpodHRwczovL2dpdGh1Yi5jb20vQm9helJvblpvaGFyL09ic2VydmF0aW9uYWxBc3Ryb25vbXlFZHVjYXRpb25Ub29scy9ibG9iL21haW4vR2FsYXh5X0FuYWx5c2lzL0JsdWVfQ2x1c3RlcnNfRnJvbV9JbWFnZXMucHkKQWZmaWxpYXRpb246IEtpbm5lcmV0IE9ic2VydmF0b3J5Ck1lbWJlciBvZiB0aGUgTENPIEdsb2JhbCBTa3kgUGFydG5lcnMgcHJvZ3JhbW1lCkRhdGU6IFNlcHRlbWJlciAyMDI1CgpXcml0dGVuIGZvciBzdHVkZW50IHByb2plY3RzIG9uIHNwaXJhbCBnYWxheGllcy4KCldIQVQgSVQgRE9FUwoKVGFrZXMgb25lIEItYmFuZCBhbmQgb25lIFYtYmFuZCBpbWFnZSBvZiBhIGdhbGF4eSBhbmQgcHJvZHVjZXMgaXRzCmNvbG91ci1tYWduaXR1ZGUgZGlhZ3JhbSwgdG9nZXRoZXIgd2l0aCB0aGUgcmFkaWFsIGRlbnNpdHkgcHJvZmlsZSBvZiB0aGUKYmx1ZSBrbm90cyAtIHRoZSB5b3VuZyBzdGFyLWZvcm1pbmcgcmVnaW9ucyBpbiB0aGUgYXJtcy4KClRoZSBjb2xvdXIgb2YgYSBzb3VyY2Ugc2F5cyBob3cgaG90IGFuZCBob3cgeW91bmcgaXQgaXMuIFBsb3R0aW5nIGNvbG91cgphZ2FpbnN0IGJyaWdodG5lc3MgZm9yIGV2ZXJ5IHNvdXJjZSBpbiB0aGUgZ2FsYXh5IHNlcGFyYXRlcyB0aGUgYmx1ZSBrbm90cwpmcm9tIHRoZSBvbGRlciBwb3B1bGF0aW9uLCBhbmQgdGhlIHJhZGlhbCBwcm9maWxlIHRoZW4gc2hvd3MgaG93IHRoZXkgYXJlCmRpc3RyaWJ1dGVkIHdpdGggZGlzdGFuY2UgZnJvbSB0aGUgY2VudHJlIC0gd2hpY2ggaXMgd2hhdCBhIHNwaXJhbCBhcm0KbG9va3MgbGlrZSBpbiBudW1iZXJzLgoKV0hFUkUgVEhFIE1FVEhPRCBDT01FUyBGUk9NCgpJZGVudGlmeWluZyBzdGFyLWZvcm1pbmcgcmVnaW9ucyBieSB0aGVpciBjb2xvdXIgaW4gb3JkaW5hcnkgYnJvYWQtYmFuZAppbWFnZXMsIGFuZCB0aGVuIHRyZWF0aW5nIHRoZWlyIGRpc3RyaWJ1dGlvbiBhcyBhIG1lYXN1cmFibGUgcHJvcGVydHkgb2YgdGhlCmdhbGF4eSwgZm9sbG93cwoKICAgIEJyb3NjaCwgTi4gKDE5OTIpLiBTdGFyIGZvcm1hdGlvbiBzeXN0ZW1hdGljcyBmcm9tIGNvbG91ciBpbWFnZXMuCiAgICBBc3Ryb3BoeXNpY3MgYW5kIFNwYWNlIFNjaWVuY2UsIDE4OCwgMjg5LTI5OC4KICAgIGRvaToxMC4xMDA3L0JGMDA2NDQ5MTYKClRoZSBhcHBlYWwgZm9yIHRlYWNoaW5nIGlzIHRoYXQgaXQgYXNrcyBub3RoaW5nIGV4b3RpYy4gVHdvIGJyb2FkLWJhbmQgZnJhbWVzCm9mIGEgZ2FsYXh5LCBCIGFuZCBWLCBhcmUgd2l0aGluIHJlYWNoIG9mIGEgc2Nob29sLWFjY2Vzc2libGUgdGVsZXNjb3BlLCBhbmQKdGhlIHlvdW5nIHJlZ2lvbnMgc2VwYXJhdGUgb3V0IG9uIGNvbG91ciBhbG9uZSAtIG5vIHNwZWN0cm9zY29weSwgbm8gbmFycm93LQpiYW5kIGZpbHRlci4gQSBzdHVkZW50IHdpdGggb25lIG5pZ2h0IG9mIGRhdGEgY2FuIGFzayBhIHJlYWwgcXVlc3Rpb24gYWJvdXQKd2hlcmUgYSBnYWxheHkgaXMgZm9ybWluZyBzdGFycywgYW5kIGFuc3dlciBpdCB3aXRoIGEgbnVtYmVyLgoKVGhpcyBzY3JpcHQgaXMgdGhlIHBpcGVsaW5lIGZvciB0aGF0LCBhbmQgaGFzIGJlZW4gdXNlZCBhY3Jvc3MgbWFueSBzdHVkZW50CnByb2plY3RzLgoKV0hBVCBZT1UgTkVFRAoKICBhIEIgYW5kIGEgViBmcmFtZSBvZiB0aGUgc2FtZSBnYWxheHkgKG9yIGEgZm9sZGVyIGhvbGRpbmcgc2V2ZXJhbCBvZiBlYWNoKSwKICBwbGF0ZS1zb2x2ZWQ6IHRoZSByZWZlcmVuY2Ugc3RhcnMgYXJlIHB1bGxlZCBmcm9tIGEgY2F0YWxvZ3VlIGJ5IHNreQogIHBvc2l0aW9uLiBUaGUgcGxhdGUgc29sdXRpb24gaXMgY2hlY2tlZCBhZ2FpbnN0IEdhaWEgYW5kIHJlcGFpcmVkIGlmIGl0CiAgZG9lcyBub3QgZml0IHRoZSBpbWFnZS4KCldIQVQgSVQgQVNLUyBZT1UKCiAgdGhlIGZvbGRlciBob2xkaW5nIHRoZSBmcmFtZXMgIHdoaWNoIGZpbGUgaXMgQiBhbmQgd2hpY2ggaXMgViBpcyByZWFkIGZyb20KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhlIEZJTFRFUiBrZXl3b3JkIG9yIHRoZSBmaWxlIG5hbWUsIGFuZCB0aGUKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZ2FsYXh5J3MgbmFtZSBmcm9tIE9CSkVDVAogIGRpc3RhbmNlIGluIE1wYyAgICAgICAgICAgICAgICBzaG93biBmcm9tIENvc21pY2Zsb3dzLTMgKFR1bGx5KyAyMDE2KTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgRW50ZXIgYWNjZXB0cyBpdC4gQ29udmVydHMgcGl4ZWxzIHRvIHBhcnNlY3MKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYW5kIGdpdmVzIHRoZSBhYnNvbHV0ZSBtYWduaXR1ZGUgTV9WCiAgY29sb3VyIGV4Y2VzcyBFKEItVikgICAgICAgICAgIHNob3duIGZyb20gdGhlIFNjaGxhZmx5ICYgRmlua2JlaW5lciAoMjAxMSkKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZHVzdCBtYXBzIHZpYSBJUlNBOyBFbnRlciBhY2NlcHRzIGl0LgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBBX1YgaXMgMy4xIHggRShCLVYpCgpFdmVyeXRoaW5nIGNhbiBhbHNvIGJlIGdpdmVuIG9uIHRoZSBjb21tYW5kIGxpbmUsIGZvciBleGFtcGxlCgogIHB5dGhvbiBCbHVlX0NsdXN0ZXJzX0Zyb21fSW1hZ2VzLnB5IC0tZm9sZGVyICJEOi9NODMiIC0teWVzCiAgcHl0aG9uIEJsdWVfQ2x1c3RlcnNfRnJvbV9JbWFnZXMucHkgLS1iIEIuZml0cyAtLXYgVi5maXRzIC0tZ2FsYXh5IE04MwogICAgICAtLWRpc3RhbmNlLW1wYyA0LjcgLS1lYnYgMC4wNiAtLWNhdGFsb2cgR0FJQQoKSE9XIElUIFdPUktTCgogIDEuIHN1YnRyYWN0cyB0aGUgYmFja2dyb3VuZCBmcm9tIGVhY2ggZnJhbWUsIG1lYXN1cmVzIGhvdyB3aWRlIGEgc3RhciBpcwogICAgIGluIGVhY2ggKGZyb20gYnJpZ2h0IHN0YXJzIHRoYXQgYXJlIG5vdCBzYXR1cmF0ZWQpLCBhbmQgZmluZHMgdGhlCiAgICAgc291cmNlcyBpbiBib3RoCiAgMi4gcmVnaXN0ZXJzIEIgb250byBWLCBhbmQgbWFrZXMgb25lIGxpc3Qgb2Ygc291cmNlcyBvbiB0aGUgViBncmlkCiAgMy4gbWVhc3VyZXMgZXZlcnkgc291cmNlIGluIGJvdGggYmFuZHMgYXQgdGhlIHNhbWUgcGxhY2Ugb24gdGhlIHNreSwKICAgICB0aHJvdWdoIE9ORSBhcGVydHVyZTogdHdpY2UgdGhlIEZXSE0gb2YgdGhlIHdpZGVyIG9mIHRoZSB0d28gZnJhbWVzLgogICAgIEVhY2ggZmx1eCBnZXRzIGFuIGVycm9yIGZyb20gdGhlIHBob3RvbiBub2lzZSBhbmQgdGhlIG1lYXN1cmVkIHNreQogICAgIG5vaXNlLCBhbmQgZWFjaCBzb3VyY2UgYSBzYXR1cmF0aW9uIGZsYWcKICA0LiBjaGVja3MgdGhlIHBsYXRlIHNvbHV0aW9uIGFnYWluc3QgR2FpYSwgdGhlbiBjYWxpYnJhdGVzIGFnYWluc3QgZXZlcnkKICAgICBjYXRhbG9ndWUgc3RhciB0aGF0IGlzIG1hdGNoZWQgb25jZSwgbm90IHNhdHVyYXRlZCwgbm90IGNyb3dkZWQgYW5kCiAgICAgd2VsbCBtZWFzdXJlZCAoQVBBU1M5LCBvciBHYWlhIERSMyBzeW50aGV0aWMgcGhvdG9tZXRyeSwgd2hpY2ggaXMgYWxzbwogICAgIHRoZSBmYWxsYmFjayk6IGEgc3RyYWlnaHQtbGluZSBmaXQgd2l0aCBhIGNvbG91ciB0ZXJtIGFuZCBvdXRsaWVyCiAgICAgcmVqZWN0aW9uLAogICAgICAgICBCLVYgPSBhICsgYyB4IChiLXYpICAgICAgICBWID0gdiArIFpQICsgayB4IChCLVYpCiAgICAgVGhlIGZpdCBpcyBwcmludGVkIHdpdGggaXRzIHNjYXR0ZXIsIGFuZCB0aGUgcnVuIHN0b3BzIGlmIGZld2VyIHRoYW4KICAgICA1IHN0YXJzIHJlbWFpbiBvciBpZiB0aGUgY29sb3VycyBydW4gdGhlIHdyb25nIHdheSAoQiBhbmQgViBzd2FwcGVkKQogIDUuIHJlbW92ZXMgdGhlIGR1c3Qgb2Ygb3VyIG93biBnYWxheHksIGFuZCB0aGUgZm9yZWdyb3VuZCBzdGFycwogIDYuIGtlZXBzIGFzIGJsdWUga25vdHMgdGhlIHNvdXJjZXMgd2l0aCAtMC41IDw9IChCLVYpMCA8PSAwLjUgYW5kIGEgQi1WCiAgICAgZXJyb3Igb2YgYXQgbW9zdCAwLjIgbWFnLCBhbmQgcGxvdHMgdGhlIGNvbG91ci1tYWduaXR1ZGUgZGlhZ3JhbXMgYW5kCiAgICAgdGhlIHJhZGlhbCBkZW5zaXR5IHByb2ZpbGUKCldIQVQgWU9VIEdFVAoKRXZlcnkgcnVuIHdyaXRlcyBhIG5ldyBmb2xkZXIgYmVzaWRlIHRoZSBmcmFtZXMsCnJlc3VsdF9DTURfPGdhbGF4eT5fPGRhdGU+Xzx0aW1lPiwgc28gYSBzZWNvbmQgcnVuIG5ldmVyIG92ZXJ3cml0ZXMgdGhlIGZpcnN0CmFuZCBub3RoaW5nIGZyb20gYW4gZWFybGllciBydW4gY2FuIGJlIG1pc3Rha2VuIGZvciB0aGlzIG9uZS4gVGhlIGZvbGRlcgpob2xkaW5nIHlvdXIgZGF0YSBpcyBvdGhlcndpc2UgbGVmdCBleGFjdGx5IGFzIGl0IHdhcy4KClRoYXQgaXMgYSBsb3Qgb2YgZmlsZXMsIHNvIGhlcmUgaXMgd2hhdCBlYWNoIG9uZSBpcyBhbmQgd2hpY2ggeW91IGFjdHVhbGx5CndhbnQgdG8gbG9vayBhdC4gQmVsb3csIE5BTUUgc3RhbmRzIGZvciB0aGUgZ2FsYXh5IG5hbWUuCgogIHJ1bl9zdW1tYXJ5LnR4dCAgICAgICAgICAgICAgICAgZXZlcnkgbnVtYmVyIHRoZSBydW4gdXNlZCBhbmQgZm91bmQKCiAgQnktcHJvZHVjdHMsIGtlcHQgYmVjYXVzZSB0aGV5IGFyZSB3aGF0IGV2ZXJ5dGhpbmcgaXMgbWVhc3VyZWQgb246CgogICAgPGlucHV0Pl9iZ3N1Yi5maXRzICAgICAgICAgICAgZWFjaCBpbnB1dCBmcmFtZSB3aXRoIGl0cyBtZWRpYW4gbGV2ZWwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHN1YnRyYWN0ZWQKICAgIE5BTUVfQl9jb21iaW5lZC5maXRzICAgICAgICAgIHdoZW4gdGhlIGZvbGRlciBoZWxkIHNldmVyYWwgZXhwb3N1cmVzCiAgICBOQU1FX1ZfY29tYmluZWQuZml0cyAgICAgICAgICBvZiBhIGZpbHRlcjogdGhlaXIgbWVkaWFuCgogIFN0YWdlIDEgLSBldmVyeXRoaW5nIHRoYXQgd2FzIG1lYXN1cmVkCgogICAgTkFNRV9waG90b21ldHJ5X3Jlc3VsdHMuY3N2ICAgZXZlcnkgc291cmNlIG1lYXN1cmVkIGluIGJvdGggYmFuZHM6CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBwb3NpdGlvbiBpbiBlYWNoIGZyYW1lLCBSQSBhbmQgRGVjLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZmx1eGVzIGFuZCBpbnN0cnVtZW50YWwgbWFnbml0dWRlcyB3aXRoCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBlcnJvcnMsIHBlYWsgcGl4ZWwsIHNhdHVyYXRpb24gZmxhZwogICAgTkFNRV9yZWZlcmVuY2Vfc3RhcnMuY3N2ICAgICAgdGhlIGNhdGFsb2d1ZSBzdGFycyBtYXRjaGVkIHRvIHNvdXJjZXMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3aGV0aGVyIGVhY2ggd2FzIHVzZWQgaW4gdGhlIGNhbGlicmF0aW9uLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYW5kIGl0cyByZXNpZHVhbC4gSWYgdGhlIGNhbGlicmF0aW9uIGxvb2tzCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3cm9uZywgbG9vayBoZXJlIGZpcnN0CgogIFN0YWdlIDIgLSBjYWxpYnJhdGVkLCBzdGlsbCB3aXRoIGZvcmVncm91bmQgc3RhcnMgaW4gaXQKCiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeS5jc3YgICAgICAgICAgICBCLCBWLCBCLVYgd2l0aCBlcnJvcnMsIGJlZm9yZQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKF9vYnMpIGFuZCBhZnRlciB0aGUgR2FsYWN0aWMKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGR1c3QgY29ycmVjdGlvbiwgYW5kIE1fVgogICAgTkFNRV9jYWxpYnJhdGVkX3Bob3RvbWV0cnlfd2l0aF9jb2xvci5jc3YgdGhlIHNhbWUsIHdpdGggQi1WIGFkZGVkCiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9DTUQucG5nICAgICAgICBjb2xvdXItbWFnbml0dWRlIGRpYWdyYW0KICAgIE5BTUVfVl9zb3VyY2VzLnBuZyAgICAgICAgICAgICAgICAgICAgICAgIHRoZSBWIGltYWdlIHdpdGggZXZlcnkgbWVhc3VyZWQKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHNvdXJjZSBjaXJjbGVkCiAgICBOQU1FX0Jfc291cmNlcy5wbmcgICAgICAgICAgICAgICAgICAgICAgICB0aGUgc2FtZSBmb3IgQiwgYXQgdGhlIHNhbWUKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHNvdXJjZXMKCiAgU3RhZ2UgMyAtIGZvcmVncm91bmQgc3RhcnMgcmVtb3ZlZAoKICAgIENhdGFsb2d1ZSBzdGFycywgR2FpYSBzdGFycyB3aXRoIGEgc2lnbmlmaWNhbnQgcGFyYWxsYXggb3IgcHJvcGVyIG1vdGlvbiwKICAgIGFuZCBldmVyeXRoaW5nIGJleW9uZCBSMjUgYXJlIHRha2VuIG91dC4KCiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFycy5jc3YKICAgIE5BTUVfY2FsaWJyYXRlZF9waG90b21ldHJ5X25vX3N0YXJzX3dpdGhfY29sb3IuY3N2CiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFyc19DTUQucG5nCiAgICBOQU1FX1Zfc291cmNlc19ub19zdGFycy5wbmcKCiAgU3RhZ2UgNCAtIGJsdWUga25vdHMgb25seQoKICAgIFRISVMgSVMgVEhFIFJFU1VMVCB0aGUgcHJvamVjdHMgYXJlIHVzdWFsbHkgYWZ0ZXIuCgogICAgTkFNRV9jYWxpYnJhdGVkX3Bob3RvbWV0cnlfbm9fc3RhcnNfY29sb3JfZmlsdGVyZWQuY3N2CiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFyc19jb2xvcl9maWx0ZXJlZF93aXRoX2NvbG9yLmNzdgogICAgTkFNRV9jYWxpYnJhdGVkX3Bob3RvbWV0cnlfbm9fc3RhcnNfY29sb3JfZmlsdGVyZWRfQ01ELnBuZwogICAgTkFNRV9WX3NvdXJjZXNfY29sb3JfZmlsdGVyZWQucG5nCiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFyc19jb2xvcl9maWx0ZXJlZF93aXRoX3JhZGl1cy5jc3YKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhlIHNhbWUsIHBsdXMgZWFjaCBrbm90J3MgZGlzdGFuY2UKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZnJvbSB0aGUgY2VudHJlIG9mIHRoZSBnYWxheHkKCiAgU3RhZ2UgNSAtIGhvdyB0aGUga25vdHMgYXJlIGRpc3RyaWJ1dGVkCgogICAgTkFNRV9yYWRpYWxfZGVuc2l0eV9wcm9maWxlLmNzdiAgICAgICBrbm90cyBwZXIgc3F1YXJlIHBpeGVsLCBieSByYWRpdXMKICAgIE5BTUVfcmFkaWFsX2RlbnNpdHlfcHJvZmlsZV9wYy5jc3YgICAgdGhlIHNhbWUgaW4gcGFyc2VjcwogICAgTkFNRV9yYWRpYWxfZGVuc2l0eV9wcm9maWxlX3N0ZXAuY3N2ICAgICB0aGUgc2FtZSBhcyBhIHN0ZXAgZnVuY3Rpb24KICAgIE5BTUVfcmFkaWFsX2RlbnNpdHlfcHJvZmlsZV9zdGVwX3BjLmNzdiAgdGhlIHN0ZXAgZnVuY3Rpb24gaW4gcGFyc2VjcwogICAgTkFNRV9yYWRpYWxfZGVuc2l0eV9wcm9maWxlX3N0ZXAucG5nICAgICAgIHRoZSBwbG90LCBpbiBwaXhlbHMKICAgIE5BTUVfcmFkaWFsX2RlbnNpdHlfcHJvZmlsZV9zdGVwX3BjLnBuZyAgICB0aGUgcGxvdCwgaW4gcGFyc2VjcwogICAgTkFNRV9kZXByb2plY3RlZF8uLi4gICAgICAgICAgICAgICAgICAgICAgIHRoZSBwcm9maWxlIGRlcHJvamVjdGVkIHNldmVyYWwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3YXlzLCB3aXRoIHRoZSBmcmFtZSdzIGNvdmVyYWdlCgogIElmIHlvdSBsb29rIGF0IGZvdXIgZmlsZXMsIGxvb2sgYXQgdGhlc2U6CgogICAgcnVuX3N1bW1hcnkudHh0ICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRpZCB0aGUgY2FsaWJyYXRpb24gd29yawogICAgTkFNRV9WX3NvdXJjZXNfY29sb3JfZmlsdGVyZWQucG5nICAgICAgICAgIGRpZCBpdCBmaW5kIHRoZSBhcm1zCiAgICBOQU1FXy4uLl9jb2xvcl9maWx0ZXJlZF9DTUQucG5nICAgICAgICAgICAgdGhlIGRpYWdyYW0KICAgIE5BTUVfcmFkaWFsX2RlbnNpdHlfcHJvZmlsZV9zdGVwX3BjLnBuZyAgICB0aGUgYXJtcywgaW4gbnVtYmVycwoKICBJbiB0aGUgZGlhZ3JhbXMsIHNhdHVyYXRlZCBzb3VyY2VzIGFuZCBzb3VyY2VzIHdpdGggYSBCLVYgZXJyb3IgYWJvdmUgMC4zCiAgbWFnIGFyZSBub3QgZHJhd247IHRoZXkgc3RheSBpbiB0aGUgdGFibGVzLgoKVGhlIHBhcmFtZXRlcnMgYXQgdGhlIHRvcCBvZiB0aGUgQmx1ZV9DbHVzdGVycyBzZWN0aW9uIG9mIHRoaXMgZmlsZSBhcmUKdHVuZWQgZm9yIHRoZSBLaW5uZXJldCBhbmQgTENPIGZyYW1lcyB0aGVzZSBwcm9qZWN0cyB1c2UuIE9uIHZlcnkgZGlmZmVyZW50CmRhdGEgLSBhIG11Y2ggc21hbGxlciB0ZWxlc2NvcGUsIGEgbXVjaCBmYWludGVyIGdhbGF4eSAtIHRoZSBkZXRlY3Rpb24KdGhyZXNob2xkIGlzIHRoZSBmaXJzdCB0aGluZyB0byBsb29rIGF0LgoKVXNhZ2U6IHJ1biBpdCwgYW5kIGFuc3dlciB0aGUgcXVlc3Rpb25zOyBFbnRlciBhY2NlcHRzIHdoYXQgaXMgc2hvd24uCiIiIgoKCgpmcm9tIF9fZnV0dXJlX18gaW1wb3J0IGFubm90YXRpb25zCgppbXBvcnQgb3MKaW1wb3J0IHN5cwppbXBvcnQgcmUKaW1wb3J0IG51bXB5IGFzIG5wCmltcG9ydCBtYXRoCmltcG9ydCB0ZXh0d3JhcAppbXBvcnQgbWF0cGxvdGxpYgppbXBvcnQgbWF0cGxvdGxpYi5weXBsb3QgYXMgcGx0CmltcG9ydCBhcmdwYXJzZQppbXBvcnQgd2FybmluZ3MKaW1wb3J0IHBhbmRhcyBhcyBwZApmcm9tIGFzdHJvcHkuaW8gaW1wb3J0IGZpdHMKZnJvbSBwaG90dXRpbHMuZGV0ZWN0aW9uIGltcG9ydCBEQU9TdGFyRmluZGVyCmZyb20gcGhvdHV0aWxzLmFwZXJ0dXJlIGltcG9ydCBDaXJjdWxhckFwZXJ0dXJlLCBDaXJjdWxhckFubnVsdXMsIGFwZXJ0dXJlX3Bob3RvbWV0cnkKZnJvbSBwaG90dXRpbHMuYmFja2dyb3VuZCBpbXBvcnQgQmFja2dyb3VuZDJELCBNZWRpYW5CYWNrZ3JvdW5kCmZyb20gYXN0cm9weS5zdGF0cyBpbXBvcnQgc2lnbWFfY2xpcHBlZF9zdGF0cywgU2lnbWFDbGlwCmZyb20gc2NpcHkubmRpbWFnZSBpbXBvcnQgZ2F1c3NpYW5fZmlsdGVyCmZyb20gYXN0cm9weS53Y3MgaW1wb3J0IFdDUwpmcm9tIGFzdHJvcXVlcnkudml6aWVyIGltcG9ydCBWaXppZXIKZnJvbSBhc3Ryb3B5LmNvb3JkaW5hdGVzIGltcG9ydCBTa3lDb29yZAppbXBvcnQgYXN0cm9weS51bml0cyBhcyB1CmltcG9ydCBhc3Ryb2FsaWduIGFzIGFhCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIHNoYXJlZCBtb2R1bGUgaW50ZXJhY3RpdmVfaW5wdXQucHkKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQoKZGVmIG5vdGhpbmdfd2FzX3Bhc3NlZChhcmd2PU5vbmUpOgogICAgIiIiVHJ1ZSB3aGVuIHRoZSB0b29sIHdhcyBzdGFydGVkIHdpdGggbm8gYXJndW1lbnRzIGF0IGFsbC4KCiAgICBUaGF0IGlzIHRoZSBvbmx5IGNhc2Ugd2hlcmUgcXVlc3Rpb25zIGFyZSB0aGUgcmlnaHQgcmVzcG9uc2UuIEEgdG9vbCBzdGFydGVkCiAgICB3aXRoIHNvbWUgYXJndW1lbnRzIGJ1dCBub3QgYWxsIG9mIHRoZW0gaGFzIGEgbWlzdGFrZSBpbiBpdCwgYW5kIGFyZ3BhcnNlCiAgICBzYXlpbmcgd2hpY2ggb25lIGlzIG1pc3NpbmcgaXMgbW9yZSB1c2VmdWwgdGhhbiBhIHF1ZXN0aW9ubmFpcmUuCiAgICAiIiIKICAgIGlmIGFyZ3YgaXMgTm9uZToKICAgICAgICByZXR1cm4gbGVuKHN5cy5hcmd2KSA8PSAxCiAgICByZXR1cm4gbGVuKGFyZ3YpID09IDAKCmRlZiBub3JtYWxpc2VfcGF0aCh0ZXh0KToKICAgICIiIkEgcGF0aCBhcyBhIHBlcnNvbiBhY3R1YWxseSBwYXN0ZXMgaXQuCgogICAgV2luZG93cyBFeHBsb3JlcidzICJDb3B5IGFzIHBhdGgiIHdyYXBzIHRoZSBwYXRoIGluIGRvdWJsZSBxdW90ZXMsIGFuZCBhIHBhdGgKICAgIGRyYWdnZWQgaW50byBhIHRlcm1pbmFsIG9mdGVuIGFycml2ZXMgd2l0aCBhIHNwYWNlIG9uIHRoZSBlbmQuIEJvdGggYXJlIG1lYW50CiAgICBhcyB0aGUgc2FtZSBwYXRoLCBzbyBib3RoIGFyZSBhY2NlcHRlZC4KICAgICIiIgogICAgcmV0dXJuIG9zLnBhdGgubm9ybXBhdGgodGV4dC5zdHJpcCgpLnN0cmlwKCciJykuc3RyaXAoIiciKSkKCmRlZiBhc2socHJvbXB0LCBkZWZhdWx0PU5vbmUsIGNhc3Q9c3RyKToKICAgICIiIk9uZSBxdWVzdGlvbi4gVGhlIGRlZmF1bHQgaXMgc2hvd24gaW4gYnJhY2tldHMgYW5kIGEgYmFyZSBFbnRlciB0YWtlcyBpdC4KCiAgICBBIHF1ZXN0aW9uIHdpdGggbm8gZGVmYXVsdCBpcyBhc2tlZCBhZ2FpbiB1bnRpbCBpdCBpcyBhbnN3ZXJlZCwgYW5kIGFuIGFuc3dlcgogICAgdGhhdCB3aWxsIG5vdCBjb252ZXJ0IGlzIHJlZnVzZWQgb24gdGhlIHNwb3QgcmF0aGVyIHRoYW4gdGhyZWUgc3RlcHMgbGF0ZXIsCiAgICB3aGVyZSB0aGUgcmVhc29uIHdvdWxkIG5vIGxvbmdlciBiZSBvYnZpb3VzLgogICAgIiIiCiAgICB3aGlsZSBUcnVlOgogICAgICAgIHNob3duID0gKCJ7fSBbe31dOiAiLmZvcm1hdChwcm9tcHQsIGRlZmF1bHQpIGlmIGRlZmF1bHQgaXMgbm90IE5vbmUKICAgICAgICAgICAgICAgICBlbHNlICJ7fTogIi5mb3JtYXQocHJvbXB0KSkKICAgICAgICB0ZXh0ID0gaW5wdXQoc2hvd24pLnN0cmlwKCkKICAgICAgICBpZiBub3QgdGV4dDoKICAgICAgICAgICAgaWYgZGVmYXVsdCBpcyBOb25lOgogICAgICAgICAgICAgICAgcHJpbnQoIiAgICB0aGlzIG9uZSBoYXMgbm8gZGVmYXVsdCAtIHBsZWFzZSBhbnN3ZXIgaXQiKQogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgIyBUaGUgZGVmYXVsdCBnb2VzIHRocm91Z2ggdGhlIHNhbWUgY29udmVyc2lvbiBhcyBhIHR5cGVkIGFuc3dlci4KICAgICAgICAgICAgIyBSZXR1cm5pbmcgaXQgcmF3IGxvb2tzIGhhcm1sZXNzIGJlY2F1c2UgbW9zdCBkZWZhdWx0cyBhcmUgc3RyaW5ncywKICAgICAgICAgICAgIyBidXQgYXNrKCJXaGljaCBvbmUiLCAiMSIsIGludCkgdGhlbiBoYW5kcyBiYWNrICIxIiBhbmQgdGhlIGNhbGxlcgogICAgICAgICAgICAjIGNvbXBhcmVzIGEgc3RyaW5nIHdpdGggYSBudW1iZXIgc2V2ZXJhbCBsaW5lcyBsYXRlciAtIHdoaWNoIGlzCiAgICAgICAgICAgICMgZXhhY3RseSBob3cgYSBiYXJlIEVudGVyIG9uIHRoZSBmcmFtZS1jaG9pY2UgcHJvbXB0IGNyYXNoZWQgYSBydW4uCiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHJldHVybiBjYXN0KGRlZmF1bHQpCiAgICAgICAgICAgIGV4Y2VwdCAoVHlwZUVycm9yLCBWYWx1ZUVycm9yKToKICAgICAgICAgICAgICAgIHJldHVybiBkZWZhdWx0CiAgICAgICAgdHJ5OgogICAgICAgICAgICByZXR1cm4gY2FzdCh0ZXh0KQogICAgICAgIGV4Y2VwdCAoVHlwZUVycm9yLCBWYWx1ZUVycm9yKToKICAgICAgICAgICAgcHJpbnQoIiAgICB0aGF0IGlzIG5vdCBhIHt9IC0gdHJ5IGFnYWluIi5mb3JtYXQoCiAgICAgICAgICAgICAgICBnZXRhdHRyKGNhc3QsICJfX25hbWVfXyIsICJ2YWxpZCB2YWx1ZSIpKSkKCmRlZiBhc2tfZmlsZShwcm9tcHQpOgogICAgIiIiQSBwYXRoIHRvIGEgZmlsZSB0aGF0IGV4aXN0cywgYXNrZWQgYWdhaW4gdW50aWwgaXQgZG9lcy4KCiAgICBDaGVja2luZyBoZXJlIGlzIHdvcnRoIHRoZSB0d28gbGluZXM6IHRoZSBhbHRlcm5hdGl2ZSBpcyBhIHJ1biB0aGF0IHJlYWRzIHRocmVlCiAgICBmcmFtZXMsIHNwZW5kcyBhIG1pbnV0ZSBvbiB0aGUgZm91cnRoLCBhbmQgdGhlbiBkaWVzIG9uIGEgdHlwbyBpbiB0aGUgZmlyc3QuCiAgICAiIiIKICAgIHdoaWxlIFRydWU6CiAgICAgICAgcGF0aCA9IG5vcm1hbGlzZV9wYXRoKGFzayhwcm9tcHQpKQogICAgICAgIGlmIG9zLnBhdGguaXNmaWxlKHBhdGgpOgogICAgICAgICAgICByZXR1cm4gcGF0aAogICAgICAgIHByaW50KCIgICAgdGhlcmUgaXMgbm8gZmlsZSBhdCAiICsgcGF0aCkKCmRlZiBhc2tfZm9sZGVyKHByb21wdCk6CiAgICAiIiJBIHBhdGggdG8gYSBmb2xkZXIgdGhhdCBleGlzdHMsIGFza2VkIGFnYWluIHVudGlsIGl0IGRvZXMuIiIiCiAgICB3aGlsZSBUcnVlOgogICAgICAgIHBhdGggPSBub3JtYWxpc2VfcGF0aChhc2socHJvbXB0KSkKICAgICAgICBpZiBvcy5wYXRoLmlzZGlyKHBhdGgpOgogICAgICAgICAgICByZXR1cm4gcGF0aAogICAgICAgIHByaW50KCIgICAgdGhlcmUgaXMgbm8gZm9sZGVyIGF0ICIgKyBwYXRoKQoKZGVmIGFza19vcHRpb25hbF9mbG9hdChwcm9tcHQsIGJsYW5rX25vdGU9ImJsYW5rIHRvIHNraXAiKToKICAgICIiIkEgbnVtYmVyLCBvciBub3RoaW5nLiBSZXR1cm5zIE5vbmUgd2hlbiB0aGUgYW5zd2VyIGlzIGxlZnQgYmxhbmsuCgogICAgV2hhdCBibGFuayBtZWFucyBpcyB0aGUgY2FsbGVyJ3MgYnVzaW5lc3MsIHNvIHRoZSBjYWxsZXIgc2F5cyBpdC4gQmxhbmsgdXNlZAogICAgdG8gbWVhbiAiZG8gd2l0aG91dCIsIGFuZCBub3cgaXQgdXN1YWxseSBtZWFucyAiZ28gYW5kIGxvb2sgaXQgdXAiIC0gYSBwcm9tcHQKICAgIHRoYXQgc3RpbGwgcHJvbWlzZWQgdGhlIGZpcnN0IHdvdWxkIGJlIGx5aW5nIGFib3V0IHdoYXQgdGhlIHRvb2wgZG9lcy4KICAgICIiIgogICAgd2hpbGUgVHJ1ZToKICAgICAgICB0ZXh0ID0gaW5wdXQoInt9IFt7fV06ICIuZm9ybWF0KHByb21wdCwgYmxhbmtfbm90ZSkpLnN0cmlwKCkKICAgICAgICBpZiBub3QgdGV4dDoKICAgICAgICAgICAgcmV0dXJuIE5vbmUKICAgICAgICB0cnk6CiAgICAgICAgICAgIHJldHVybiBmbG9hdCh0ZXh0KQogICAgICAgIGV4Y2VwdCBWYWx1ZUVycm9yOgogICAgICAgICAgICBwcmludCgiICAgIHRoYXQgaXMgbm90IGEgbnVtYmVyIC0gdHJ5IGFnYWluLCBvciBwcmVzcyBFbnRlciB0byBza2lwIikKCmRlZiBhc2tfeWVzX25vKHByb21wdCwgZGVmYXVsdD1GYWxzZSk6CiAgICAiIiJBIHllcyBvciBubyBxdWVzdGlvbi4gQW55dGhpbmcgc3RhcnRpbmcgd2l0aCB5IG9yIG4gY291bnRzLiIiIgogICAgaGludCA9ICJZL24iIGlmIGRlZmF1bHQgZWxzZSAieS9OIgogICAgd2hpbGUgVHJ1ZToKICAgICAgICB0ZXh0ID0gaW5wdXQoInt9IFt7fV06ICIuZm9ybWF0KHByb21wdCwgaGludCkpLnN0cmlwKCkubG93ZXIoKQogICAgICAgIGlmIG5vdCB0ZXh0OgogICAgICAgICAgICByZXR1cm4gZGVmYXVsdAogICAgICAgIGlmIHRleHRbMF0gaW4gInluIjoKICAgICAgICAgICAgcmV0dXJuIHRleHRbMF0gPT0gInkiCiAgICAgICAgcHJpbnQoIiAgICBwbGVhc2UgYW5zd2VyIHllcyBvciBubyIpCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIHNoYXJlZCBtb2R1bGUgZnJhbWVfaW52ZW50b3J5LnB5CiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KClJPTEVTID0gKCJCIiwgIlYiLCAiUiIsICJIYSIpCgpST0xFX01FQU5JTkcgPSB7CiAgICAiQiI6ICJibHVlIiwKICAgICJWIjogInZpc3VhbCAtIHRoZSByZWZlcmVuY2UgZ3JpZCIsCiAgICAiUiI6ICJyZWQgY29udGludXVtIiwKICAgICJIYSI6ICJILWFscGhhIC0gdGhlIGVtaXNzaW9uIGxpbmUiLAp9CgpfUEFUVEVSTlMgPSB7CiAgICAiSGEiOiByZS5jb21waWxlKHIiKD88IVthLXowLTldKWhbXHNfLV0qYWwoPzpwaGF8ZmEpKD8hW2EtejAtOV0pIgogICAgICAgICAgICAgICAgICAgICByInwoPzwhW2EtejAtOV0paGEoPyFbYS16MC05XSkiKSwKICAgICJCIjogcmUuY29tcGlsZShyIig/PCFbYS16MC05XSkoPzpifGJlc3NlbGxbXHNfLV0qYnxicCkoPyFbYS16MC05XSkiKSwKICAgICJWIjogcmUuY29tcGlsZShyIig/PCFbYS16MC05XSkoPzp2fGJlc3NlbGxbXHNfLV0qdikoPyFbYS16MC05XSkiKSwKICAgICJSIjogcmUuY29tcGlsZShyIig/PCFbYS16MC05XSkoPzpyfHJwfHJjfHJwcmltZXxzbG9hbltcc18tXSpyfGJlc3NlbGxbXHNfLV0qcikiCiAgICAgICAgICAgICAgICAgICAgciIoPyFbYS16MC05XSkiKSwKfQoKX0NPTUJJTkVEX1dPUkRTID0gcmUuY29tcGlsZShyIlxiKG1lZGlhbnxhdmVyYWdlfG1lYW58c3VtKVxzK29mXGJ8XGJjb21iaW58XGJzdGFjayIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmUuSSkKCkZJVFNfU1VGRklYRVMgPSAoIi5maXRzIiwgIi5maXQiLCAiLmZ0cyIsICIuZml0cy5meiIpCgpPVVRQVVRfRk9MREVSUyA9IHsicmVzdWx0X2NtZCIsICJyZXN1bHQiLCAicmVzdWx0cyIsICJvdXRwdXQiLCAib3V0In0KCmRlZiByb2xlc19pbl90ZXh0KHRleHQpOgogICAgIiIiRXZlcnkgZmlsdGVyIHJvbGUgbmFtZWQgaW4gYSBwaWVjZSBvZiB0ZXh0LiIiIgogICAgaWYgbm90IHRleHQ6CiAgICAgICAgcmV0dXJuIHNldCgpCiAgICBsb3cgPSBzdHIodGV4dCkubG93ZXIoKQogICAgcmV0dXJuIHtyb2xlIGZvciByb2xlLCBwYXR0ZXJuIGluIF9QQVRURVJOUy5pdGVtcygpIGlmIHBhdHRlcm4uc2VhcmNoKGxvdyl9CgpfRElGRkVSRU5DRSA9IHJlLmNvbXBpbGUoCiAgICByIig/OmhbXHNfLV0qYWwoPzpwaGF8ZmEpfCg/PCFbYS16MC05XSloYSlbXi9cXF0qPy1ccyooPzpyfHJwfHJjKSg/IVthLXowLTldKSIKICAgIHIifG1pbnVzfGNvbnRbXHNfLV0qc3ViIiwgcmUuSSkKCmRlZiBpc19kaWZmZXJlbmNlX25hbWUobmFtZSk6CiAgICAiIiJUcnVlIHdoZW4gYSBmaWxlIG5hbWUgc2F5cyB0aGUgZnJhbWUgaXMgb25lIGZpbHRlciBtaW51cyBhbm90aGVyLCBzdWNoCiAgICBhcyBILWFscGhhIHdpdGggdGhlIFIgY29udGludXVtIHRha2VuIG9mZiAoIk04M19IYVgxMi1SLmZ0cyIpLiIiIgogICAgcmV0dXJuIGJvb2woX0RJRkZFUkVOQ0Uuc2VhcmNoKHN0cihuYW1lKSkpCgpfUFJPRFVDVF9ISVNUT1JZID0gcmUuY29tcGlsZSgKICAgIHIibXVsdGlwbGllZFxzK2J5fHN1YnRyYWN0ZWRcYi4qXC4oPzpmaXRzP3xmdHMpXGIiLCByZS5JKQoKZGVmIGlzX3Byb2R1Y3RfaGVhZGVyKGhlYWRlcik6CiAgICAiIiJUcnVlIHdoZW4gYSBISVNUT1JZIGxpbmUgc2F5cyB0aGUgZnJhbWUgd2FzIHNjYWxlZCBieSBhIGZhY3RvciBvciBoYWQKICAgIGFub3RoZXIgRklUUyBmaWxlIHN1YnRyYWN0ZWQgZnJvbSBpdC4gQSBjb25zdGFudCBza3kgKCJTdWJ0cmFjdGVkIDkuMCIpIGRvZXMKICAgIG5vdCBjb3VudC4iIiIKICAgIHJldHVybiBhbnkoX1BST0RVQ1RfSElTVE9SWS5zZWFyY2gobGluZSkgZm9yIGxpbmUgaW4gaGlzdG9yeV9saW5lcyhoZWFkZXIpKQoKZGVmIHJvbGVfZnJvbV9oZWFkZXIoaGVhZGVyKToKICAgICIiIlRoZSByb2xlIG5hbWVkIGJ5IHRoZSBmaWx0ZXIga2V5d29yZCwgb3IgTm9uZSB3aGVuIHRoZXJlIGlzIG5vIHN1Y2gga2V5d29yZC4iIiIKICAgIGZvciBrZXkgaW4gKCJGSUxURVIiLCAiRklMVEVSMSIsICJGSUxUTkFNRSIsICJJTlNGSUxURSIpOgogICAgICAgIHZhbHVlID0gaGVhZGVyLmdldChrZXkpCiAgICAgICAgaWYgdmFsdWUgaXMgTm9uZSBvciBzdHIodmFsdWUpLnN0cmlwKCkudXBwZXIoKSBpbiAoIiIsICJOT1RQUkVTRU5UIiwgIk5PTkUiKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICBmb3VuZCA9IHJvbGVzX2luX3RleHQodmFsdWUpCiAgICAgICAgaWYgbGVuKGZvdW5kKSA9PSAxOgogICAgICAgICAgICByZXR1cm4gZm91bmQucG9wKCkKICAgIHJldHVybiBOb25lCgpkZWYgaGlzdG9yeV9saW5lcyhoZWFkZXIpOgogICAgIiIiVGhlIGhpc3RvcnkgbGluZXMgb2YgYSBoZWFkZXIsIGFzIHBsYWluIHN0cmluZ3MuIiIiCiAgICB0cnk6CiAgICAgICAgcmV0dXJuIFtzdHIodikgZm9yIHYgaW4gaGVhZGVyWyJISVNUT1JZIl1dCiAgICBleGNlcHQgKEtleUVycm9yLCBUeXBlRXJyb3IpOgogICAgICAgIHJldHVybiBbXQoKZGVmIGlzX2NvbWJpbmVkKGhlYWRlcik6CiAgICAiIiJUcnVlIHdoZW4gdGhlIGhlYWRlciBzYXlzIHRoaXMgZnJhbWUgaXMgc2V2ZXJhbCBmcmFtZXMgY29tYmluZWQuIiIiCiAgICByZXR1cm4gYW55KF9DT01CSU5FRF9XT1JEUy5zZWFyY2gobGluZSkgZm9yIGxpbmUgaW4gaGlzdG9yeV9saW5lcyhoZWFkZXIpKQoKZGVmIGNvbWJpbmVfbm90ZShoZWFkZXIpOgogICAgIiIiVGhlIGhpc3RvcnkgbGluZSB0aGF0IHNheXMgaG93IHRoZSBmcmFtZSB3YXMgY29tYmluZWQsIGZvciB0aGUgc3VtbWFyeS4iIiIKICAgIGZvciBsaW5lIGluIGhpc3RvcnlfbGluZXMoaGVhZGVyKToKICAgICAgICBpZiBfQ09NQklORURfV09SRFMuc2VhcmNoKGxpbmUpOgogICAgICAgICAgICByZXR1cm4gbGluZS5zdHJpcCgpCiAgICByZXR1cm4gIiIKCmRlZiBoYXNfd2NzKGhlYWRlcik6CiAgICAiIiJUcnVlIHdoZW4gdGhlIGhlYWRlciBjYXJyaWVzIGEgcGxhdGUgc29sdXRpb24gdGhlIHRvb2xzIGNhbiB1c2UuIiIiCiAgICB0cnk6CiAgICAgICAgZnJvbSBhc3Ryb3B5LndjcyBpbXBvcnQgV0NTCiAgICAgICAgcmV0dXJuIFdDUyhoZWFkZXIpLmhhc19jZWxlc3RpYWwKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcmV0dXJuIEZhbHNlCgpkZWYgaW5zcGVjdChwYXRoKToKICAgICIiIkV2ZXJ5dGhpbmcgdGhlIHRvb2xzIG5lZWQgdG8ga25vdyBhYm91dCBvbmUgZmlsZS4iIiIKICAgIGZyb20gYXN0cm9weS5pbyBpbXBvcnQgZml0cwoKICAgIGhlYWRlciA9IGZpdHMuZ2V0aGVhZGVyKHBhdGgpCiAgICByb2xlID0gcm9sZV9mcm9tX2hlYWRlcihoZWFkZXIpCiAgICBzb3VyY2UgPSAidGhlIGZpbHRlciBrZXl3b3JkIGluIHRoZSBoZWFkZXIiCgogICAgaWYgaXNfcHJvZHVjdF9oZWFkZXIoaGVhZGVyKToKICAgICAgICAjIFRoZSBoZWFkZXIgc2F5cyBpdCBpcyB0aGlzIGZyYW1lIHNjYWxlZCwgb3IgYW5vdGhlciBmcmFtZSB0YWtlbiBvZmYKICAgICAgICAjIGl0ICgiTXVsdGlwbGllZCBieSAxMS41IiwgIlN1YnRyYWN0ZWQgLi4uUl9TVUJfTWVkaWFuX29mXzMiKTogbm90IGEKICAgICAgICAjIGZyYW1lIHRocm91Z2ggb25lIGZpbHRlciwgd2hhdGV2ZXIgaXRzIG5hbWUgb3IgRklMVEVSIGtleXdvcmQgc2F5cy4KICAgICAgICByb2xlID0gTm9uZQogICAgICAgIHNvdXJjZSA9ICJpdHMgaGlzdG9yeSAoYSBzY2FsZWQgb3Igc3VidHJhY3RlZCBwcm9kdWN0IC0gbm90IHVzZWQpIgogICAgZWxpZiByb2xlIGlzIE5vbmUgYW5kIGlzX2RpZmZlcmVuY2VfbmFtZShvcy5wYXRoLmJhc2VuYW1lKHBhdGgpKToKICAgICAgICAjICJIYVgxMi1SIiBpcyBILWFscGhhIHdpdGggMTIgeCBSIHRha2VuIG9mZjogYSBwcm9kdWN0LCBub3QgYSBmcmFtZQogICAgICAgICMgdGhyb3VnaCBlaXRoZXIgZmlsdGVyLiBOYW1pbmcgaXQgUiB3b3VsZCBwdXQgaXQgaW4gYXMgdGhlIGNvbnRpbnV1bS4KICAgICAgICBzb3VyY2UgPSAidGhlIGZpbGUgbmFtZSAoYSBkaWZmZXJlbmNlIG9mIHR3byBmcmFtZXMgLSBub3QgdXNlZCkiCiAgICBlbGlmIHJvbGUgaXMgTm9uZToKICAgICAgICBmb3VuZCA9IHJvbGVzX2luX3RleHQob3MucGF0aC5iYXNlbmFtZShwYXRoKSkKICAgICAgICBzb3VyY2UgPSAidGhlIGZpbGUgbmFtZSIKICAgICAgICBpZiBsZW4oZm91bmQpID09IDE6CiAgICAgICAgICAgIHJvbGUgPSBuZXh0KGl0ZXIoZm91bmQpKQoKICAgIHJldHVybiB7CiAgICAgICAgInBhdGgiOiBwYXRoLAogICAgICAgICJuYW1lIjogb3MucGF0aC5iYXNlbmFtZShwYXRoKSwKICAgICAgICAicm9sZSI6IHJvbGUsCiAgICAgICAgInJvbGVfc291cmNlIjogc291cmNlIGlmIHJvbGUgZWxzZSAiIiwKICAgICAgICAiY29tYmluZWQiOiBpc19jb21iaW5lZChoZWFkZXIpLAogICAgICAgICJjb21iaW5lX25vdGUiOiBjb21iaW5lX25vdGUoaGVhZGVyKSwKICAgICAgICAid2NzIjogaGFzX3djcyhoZWFkZXIpLAogICAgICAgICJleHB0aW1lIjogaGVhZGVyLmdldCgiRVhQVElNRSIpLAogICAgICAgICJvYmplY3QiOiAoc3RyKGhlYWRlci5nZXQoIk9CSkVDVCIpKS5zdHJpcCgpCiAgICAgICAgICAgICAgICAgICBpZiBoZWFkZXIuZ2V0KCJPQkpFQ1QiKSBlbHNlICIiKSwKICAgICAgICAic2hhcGUiOiAoaGVhZGVyLmdldCgiTkFYSVMyIiksIGhlYWRlci5nZXQoIk5BWElTMSIpKSwKICAgIH0KCmRlZiBzY2FuX2ZvbGRlcihmb2xkZXIsIHJlY3Vyc2l2ZT1UcnVlKToKICAgICIiIkluc3BlY3QgZXZlcnkgRklUUyBmaWxlIGluIGEgZm9sZGVyLiBSZXR1cm5zIGEgbGlzdCBvZiByZWNvcmRzLiIiIgogICAgZm91bmQgPSBbXQogICAgZm9yIHJvb3QsIGRpcnMsIGZpbGVzIGluIG9zLndhbGsoZm9sZGVyKToKICAgICAgICAjIERvIG5vdCB3YWxrIGludG8gd2hhdCBhbiBlYXJsaWVyIHJ1biB3cm90ZS4KICAgICAgICAjIEV2ZXJ5IHJ1biBub3cgd3JpdGVzIGEgZm9sZGVyIG9mIGl0cyBvd24gLSByZXN1bHRfQ01EXzxnYWxheHk+XzxkYXRlPiwKICAgICAgICAjIHJlc3VsdF9ISUlfLi4uLCByZXN1bHRfU3RhckZvcm1hdGlvbl8uLi4gLSBzbyBhbnl0aGluZyB3aG9zZSBuYW1lCiAgICAgICAgIyBzdGFydHMgd2l0aCAicmVzdWx0IiBpcyBvdXRwdXQsIG5vdCBkYXRhLgogICAgICAgIGRpcnNbOl0gPSBbZCBmb3IgZCBpbiBkaXJzIGlmIGQubG93ZXIoKSBub3QgaW4gT1VUUFVUX0ZPTERFUlMKICAgICAgICAgICAgICAgICAgIGFuZCBub3QgZC5sb3dlcigpLnN0YXJ0c3dpdGgoInJlc3VsdCIpXQogICAgICAgIGZvciBuYW1lIGluIHNvcnRlZChmaWxlcyk6CiAgICAgICAgICAgIGlmIG5vdCBuYW1lLmxvd2VyKCkuZW5kc3dpdGgoRklUU19TVUZGSVhFUyk6CiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICAjIFdoYXQgdGhlc2UgdG9vbHMgd3JpdGUgbmV4dCB0byB0aGVpciByZXN1bHRzIC0gYSBiYWNrZ3JvdW5kLQogICAgICAgICAgICAjIHN1YnRyYWN0ZWQgY29weSwgYSBmcmFtZSB3aXRoIGEgYm9ycm93ZWQgc29sdXRpb24gLSBpcyBub3QgZGF0YS4KICAgICAgICAgICAgaWYgcmUuc2VhcmNoKHIiXyhiZ3N1Ynxzb2x2ZWQpXC5maXRzPyQiLCBuYW1lLCByZS5JKToKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgIHBhdGggPSBvcy5wYXRoLmpvaW4ocm9vdCwgbmFtZSkKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgZm91bmQuYXBwZW5kKGluc3BlY3QocGF0aCkpCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOgogICAgICAgICAgICAgICAgZm91bmQuYXBwZW5kKHsicGF0aCI6IHBhdGgsICJuYW1lIjogbmFtZSwgInJvbGUiOiBOb25lLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAicm9sZV9zb3VyY2UiOiAiIiwgImNvbWJpbmVkIjogRmFsc2UsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICJjb21iaW5lX25vdGUiOiAiIiwgIndjcyI6IEZhbHNlLCAiZXhwdGltZSI6IE5vbmUsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICJvYmplY3QiOiAiIiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgInNoYXBlIjogKE5vbmUsIE5vbmUpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAiZXJyb3IiOiAie306IHt9Ii5mb3JtYXQodHlwZShleGMpLl9fbmFtZV9fLCBleGMpfSkKICAgICAgICBpZiBub3QgcmVjdXJzaXZlOgogICAgICAgICAgICBicmVhawogICAgcmV0dXJuIGZvdW5kCgpkZWYgZ2FsYXh5X2Zyb21fcmVjb3JkcyhyZWNvcmRzLCBmb2xkZXI9Tm9uZSk6CiAgICAiIiJUaGUgZ2FsYXh5J3MgbmFtZSwgdGFrZW4gZnJvbSB0aGUgZnJhbWVzIHRoZW1zZWx2ZXMuCgogICAgVGhlIHRlbGVzY29wZSB3cm90ZSBpdCBpbnRvIGV2ZXJ5IGV4cG9zdXJlIGl0IHRvb2ssIHNvIHRoZXJlIGlzIG5vIHJlYXNvbiB0bwogICAgYXNrIGZvciBpdCAtIGFuZCBubyByZWFzb24gdG8gcmVhZCBpdCBvZmYgYSBmb2xkZXIgbmFtZSwgd2hpY2ggaXMgd3JpdHRlbiBieSBhCiAgICBwZXJzb24gYW5kIGNhcnJpZXMgd2hhdGV2ZXIgZWxzZSB0aGV5IHdhbnRlZCB0byByZW1lbWJlciB0aGF0IGRheS4gQSBjb21iaW5lZAogICAgZnJhbWUgaGFzIHVzdWFsbHkgbG9zdCB0aGUga2V5d29yZCBhbG9uZyB3aXRoIHRoZSByZXN0IG9mIGl0cyBoZWFkZXIsIGJ1dCBpdAogICAgb25seSB0YWtlcyBvbmUgZXhwb3N1cmUgaW4gdGhlIGZvbGRlciB0aGF0IHN0aWxsIGhhcyBpdC4KCiAgICBSZXR1cm5zIChuYW1lLCB3aGVyZSBpdCBjYW1lIGZyb20pLiBUaGUgZm9sZGVyIG5hbWUgaXMgdGhlIGZhbGxiYWNrLCBjbGVhbmVkIG9mCiAgICB0aGUgbm90ZXMgcGVvcGxlIGFwcGVuZCB0byBpdC4KICAgICIiIgogICAgaW1wb3J0IGNvbGxlY3Rpb25zCgogICAgdm90ZXMgPSBjb2xsZWN0aW9ucy5Db3VudGVyKHIuZ2V0KCJvYmplY3QiLCAiIikgZm9yIHIgaW4gcmVjb3JkcwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIHIuZ2V0KCJvYmplY3QiKSkKICAgIGlmIHZvdGVzOgogICAgICAgIG5hbWUsIG4gPSB2b3Rlcy5tb3N0X2NvbW1vbigxKVswXQogICAgICAgIHJldHVybiBuYW1lLCAidGhlIE9CSkVDVCBrZXl3b3JkLCBpbiB7fSBvZiB7fSBmcmFtZXMiLmZvcm1hdChuLCBsZW4ocmVjb3JkcykpCgogICAgaWYgZm9sZGVyOgogICAgICAgIHN0ZW0gPSBvcy5wYXRoLmJhc2VuYW1lKG9zLnBhdGgubm9ybXBhdGgoZm9sZGVyKSkKICAgICAgICAjICJNMzMgMC40IG1ldGVyIiBpcyBhIG5vdGUgdG8gc2VsZiwgbm90IGEgY2F0YWxvZ3VlIG5hbWU7IHRoZSBuYW1lIGlzIHRoZQogICAgICAgICMgZmlyc3Qgd29yZCB0aGF0IGxvb2tzIGxpa2Ugb25lLgogICAgICAgIG0gPSByZS5tYXRjaChyIlxzKihbQS1aYS16XXsxLDR9XHMqXGQrW0EtWmEtel0/KSIsIHN0ZW0pCiAgICAgICAgaWYgbToKICAgICAgICAgICAgcmV0dXJuIHJlLnN1YihyIlxzKyIsICIgIiwgbS5ncm91cCgxKSkuc3RyaXAoKSwgInRoZSBmb2xkZXIgbmFtZSIKICAgICAgICByZXR1cm4gc3RlbSwgInRoZSBmb2xkZXIgbmFtZSIKCiAgICByZXR1cm4gTm9uZSwgIiIKCmRlZiBjaG9vc2UocmVjb3Jkcywgcm9sZXM9Uk9MRVMsIGFzaz1Ob25lKToKICAgICIiIlBpY2sgb25lIGZyYW1lIHBlciByb2xlOiB0aGUgY29tYmluZWQgb25lLCB3aGVuIHRoZXJlIGlzIG9uZS4KCiAgICBBIGNvbWJpbmVkIGZyYW1lIGlzIGRlZXBlciB0aGFuIGFueSBzaW5nbGUgZXhwb3N1cmUsIHdoaWNoIGlzIHRoZSByZWFzb24gaXQKICAgIHdhcyBtYWRlLCBzbyBpdCB3aW5zIHdoZW5ldmVyIG9uZSBleGlzdHMuIFdoZW4gdGhlcmUgaXMgbm9uZSBidXQgc2V2ZXJhbAogICAgZXhwb3N1cmVzLCBhbGwgb2YgdGhlbSBhcmUgcmV0dXJuZWQgdG8gYmUgY29tYmluZWQgcmF0aGVyIHRoYW4gb25lIGJlaW5nCiAgICBwaWNrZWQgLSB0aGF0IGlzIHdoYXQgdGhlIG9ic2VydmVyIHdlbnQgb3V0IGFuZCB0b29rLgoKICAgIFJldHVybnMgKGNob3NlbiwgbWVtYmVycywgbm90ZXMpOiB0aGUgcmVwcmVzZW50YXRpdmUgZnJhbWUgcGVyIHJvbGUsIHRoZQogICAgZXhwb3N1cmVzIHRvIGNvbWJpbmUgd2hlcmUgdGhlcmUgYXJlIHNldmVyYWwsIGFuZCBhbnl0aGluZyB3b3J0aCBzYXlpbmcuCiAgICAiIiIKICAgIGNob3NlbiwgbWVtYmVycywgbm90ZXMgPSB7fSwge30sIFtdCgogICAgZm9yIHJvbGUgaW4gcm9sZXM6CiAgICAgICAgbWluZSA9IFtyIGZvciByIGluIHJlY29yZHMgaWYgci5nZXQoInJvbGUiKSA9PSByb2xlIGFuZCBub3Qgci5nZXQoImVycm9yIildCiAgICAgICAgaWYgbm90IG1pbmU6CiAgICAgICAgICAgIG5vdGVzLmFwcGVuZCgie306IG5vdGhpbmcgaW4gdGhlIGZvbGRlciBpZGVudGlmaWVzIGl0c2VsZiBhcyB7fSIuZm9ybWF0KAogICAgICAgICAgICAgICAgcm9sZSwgUk9MRV9NRUFOSU5HW3JvbGVdKSkKICAgICAgICAgICAgY29udGludWUKCiAgICAgICAgY29tYmluZWQgPSBbciBmb3IgciBpbiBtaW5lIGlmIHJbImNvbWJpbmVkIl1dCiAgICAgICAgaWYgbGVuKGNvbWJpbmVkKSA9PSAxOgogICAgICAgICAgICBjaG9zZW5bcm9sZV0gPSBjb21iaW5lZFswXQogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGlmIGxlbihjb21iaW5lZCkgPiAxOgogICAgICAgICAgICBpZiBhc2sgaXMgTm9uZToKICAgICAgICAgICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigie306IHt9IGNvbWJpbmVkIGZyYW1lcyB0byBjaG9vc2UgZnJvbSIuZm9ybWF0KAogICAgICAgICAgICAgICAgICAgIHJvbGUsIGxlbihjb21iaW5lZCkpKQogICAgICAgICAgICBjaG9zZW5bcm9sZV0gPSBhc2socm9sZSwgY29tYmluZWQpCiAgICAgICAgICAgIGNvbnRpbnVlCgogICAgICAgIHNpbmdsZXMgPSBbciBmb3IgciBpbiBtaW5lIGlmIHJbIndjcyJdXSBvciBtaW5lCiAgICAgICAgaWYgbGVuKHNpbmdsZXMpID09IDE6CiAgICAgICAgICAgIGNob3Nlbltyb2xlXSA9IHNpbmdsZXNbMF0KICAgICAgICAgICAgbm90ZXMuYXBwZW5kKCJ7fTogbm8gY29tYmluZWQgZnJhbWUsIHVzaW5nIHRoZSBvbmx5IGV4cG9zdXJlIi5mb3JtYXQocm9sZSkpCiAgICAgICAgZWxzZToKICAgICAgICAgICAgIyBTZXZlcmFsIGV4cG9zdXJlcyBhbmQgbm90aGluZyBjb21iaW5lZC4gUGlja2luZyBvbmUgb2YgdGhlbSBhbmQKICAgICAgICAgICAgIyBkaXNjYXJkaW5nIHRoZSByZXN0IHRocm93cyBhd2F5IG1vc3Qgb2YgdGhlIG9ic2VydmF0aW9uIC0gd2l0aCBzaXgKICAgICAgICAgICAgIyBILWFscGhhIGZyYW1lcyBpdCBjb3N0cyBhIGZhY3RvciBvZiAyLjQgaW4gbm9pc2UgLSBzbyB0aGV5IGFyZSBhbGwKICAgICAgICAgICAgIyBrZXB0IGFuZCB0aGUgY2FsbGVyIGNvbWJpbmVzIHRoZW0uIFRoZSBmaXJzdCBpcyBuYW1lZCBhcyB0aGUKICAgICAgICAgICAgIyByZXByZXNlbnRhdGl2ZSBiZWNhdXNlIHRoZSBvdXRwdXQgaGFzIHRvIGJlIGxhYmVsbGVkIHdpdGggc29tZXRoaW5nLgogICAgICAgICAgICBjaG9zZW5bcm9sZV0gPSBzaW5nbGVzWzBdCiAgICAgICAgICAgIG1lbWJlcnNbcm9sZV0gPSBzaW5nbGVzCiAgICAgICAgICAgIG5vdGVzLmFwcGVuZCgie306IG5vIGNvbWJpbmVkIGZyYW1lOyB7fSBleHBvc3VyZXMgdG8gY29tYmluZSIuZm9ybWF0KAogICAgICAgICAgICAgICAgcm9sZSwgbGVuKHNpbmdsZXMpKSkKCiAgICByZXR1cm4gY2hvc2VuLCBtZW1iZXJzLCBub3RlcwoKZGVmIGRlc2NyaWJlKHJlY29yZHMsIGNob3Nlbiwgbm90ZXMpOgogICAgIiIiVGhlIHN1bW1hcnkgYSBwZXJzb24gcmVhZHMgYmVmb3JlIGFncmVlaW5nIHRvIHRoZSBydW4uIiIiCiAgICBsaW5lcyA9IFsiIiwgIldoYXQgdGhlIGZvbGRlciBob2xkczoiXQogICAgYnlfcm9sZSA9IHt9CiAgICBmb3IgciBpbiByZWNvcmRzOgogICAgICAgIGJ5X3JvbGUuc2V0ZGVmYXVsdChyLmdldCgicm9sZSIpIG9yICI/IiwgW10pLmFwcGVuZChyKQoKICAgIGZvciByb2xlIGluIGxpc3QoUk9MRVMpICsgWyI/Il06CiAgICAgICAgZ3JvdXAgPSBieV9yb2xlLmdldChyb2xlKQogICAgICAgIGlmIG5vdCBncm91cDoKICAgICAgICAgICAgY29udGludWUKICAgICAgICBsaW5lcy5hcHBlbmQoIiAgIiArICgie30gKHt9KSIuZm9ybWF0KHJvbGUsIFJPTEVfTUVBTklOR1tyb2xlXSkKICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiByb2xlIGluIFJPTEVfTUVBTklORyBlbHNlICJub3QgaWRlbnRpZmllZCIpKQogICAgICAgIGZvciByIGluIHNvcnRlZChncm91cCwga2V5PWxhbWJkYSB4OiB4WyJuYW1lIl0pOgogICAgICAgICAgICBtYXJrID0gIi0+IiBpZiBjaG9zZW4uZ2V0KHJvbGUpIGlzIHIgZWxzZSAiICAiCiAgICAgICAgICAgIGlmIHIuZ2V0KCJlcnJvciIpOgogICAgICAgICAgICAgICAgYml0cyA9IFsidW5yZWFkYWJsZTogIiArIHJbImVycm9yIl1dCiAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICBiaXRzID0gWyJjb21iaW5lZCIgaWYgclsiY29tYmluZWQiXSBlbHNlICJzaW5nbGUiLAogICAgICAgICAgICAgICAgICAgICAgICAic29sdmVkIiBpZiByWyJ3Y3MiXSBlbHNlICJOTyBXQ1MiXQogICAgICAgICAgICBsaW5lcy5hcHBlbmQoIiAgICB7fSB7Ojw0Mn0ge30iLmZvcm1hdChtYXJrLCByWyJuYW1lIl0sICIsICIuam9pbihiaXRzKSkpCgogICAgbGluZXMuYXBwZW5kKCIiKQogICAgbGluZXMuYXBwZW5kKCJDaG9zZW4gZm9yIHRoZSBtZWFzdXJlbWVudDoiKQogICAgZm9yIHJvbGUgaW4gUk9MRVM6CiAgICAgICAgciA9IGNob3Nlbi5nZXQocm9sZSkKICAgICAgICBsaW5lcy5hcHBlbmQoIiAgICB7OjwzfSB7fSIuZm9ybWF0KHJvbGUsIHJbIm5hbWUiXSBpZiByIGVsc2UgIm5vdGhpbmciKSkKICAgIGZvciBuIGluIG5vdGVzOgogICAgICAgIGxpbmVzLmFwcGVuZCgiICAgIG5vdGU6ICIgKyBuKQogICAgbGluZXMuYXBwZW5kKCIiKQogICAgcmV0dXJuICJcbiIuam9pbihsaW5lcykKCmRlZiB3Y3NfZnJvbV9yZWZlcmVuY2UoaW1hZ2UsIHJlZmVyZW5jZV9pbWFnZSwgcmVmZXJlbmNlX3djcywKICAgICAgICAgICAgICAgICAgICAgICBkZXRlY3Rpb25fc2lnbWE9Ni4wLCBtYXhfY29udHJvbF9wb2ludHM9NjApOgogICAgIiIiRml0IGEgcGxhdGUgc29sdXRpb24gZm9yIGFuIHVuc29sdmVkIGZyYW1lIGJ5IG1hdGNoaW5nIGl0cyBzdGFycy4KCiAgICBBIGNvbWJpbmVkIGZyYW1lIGFuZCB0aGUgc2luZ2xlIGV4cG9zdXJlcyBpdCB3YXMgYnVpbHQgZnJvbSBzaG93IHRoZSBzYW1lCiAgICBzdGFycyBpbiBhbG1vc3QgdGhlIHNhbWUgcGxhY2VzLCBzbyBhIHNvbHZlZCBleHBvc3VyZSBvZiB0aGUgc2FtZSBmaWVsZCBjYW4KICAgIGxlbmQgaXRzIHNvbHV0aW9uOiBtYXRjaCB0aGUgc3RhciBwYXR0ZXJucywgcmVhZCBlYWNoIG1hdGNoZWQgc3RhcidzIHNreQogICAgcG9zaXRpb24gZnJvbSB0aGUgc29sdmVkIGZyYW1lLCBhbmQgZml0IGEgc29sdXRpb24gdG8gdGhvc2UgcG9zaXRpb25zIGluIHRoZQogICAgdW5zb2x2ZWQgZnJhbWUncyBvd24gcGl4ZWxzLgoKICAgIFRoaXMgaXMgbm90IGNvcHlpbmcgYSBoZWFkZXIgYWNyb3NzLiBUaGUgcmVzdWx0IGlzIGZpdHRlZCB0byBtYXRjaGVkIHN0YXJzIGFuZAogICAgY29tZXMgd2l0aCB0aGUgcmVzaWR1YWwgb2YgdGhhdCBmaXQsIHdoaWNoIGlzIHRoZSBob25lc3QgbWVhc3VyZSBvZiBob3cgd2VsbAogICAgaXQgZGlkOiBhIGZldyBodW5kcmVkdGhzIG9mIGEgcGl4ZWwgbWVhbnMgdGhlIHR3byBmcmFtZXMgcmVhbGx5IGFyZSBvbiB0aGUKICAgIHNhbWUgZ3JpZCwgYW5kIGEgcmVzaWR1YWwgYXBwcm9hY2hpbmcgYSBwaXhlbCBtZWFucyBzb21ldGhpbmcgbW92ZWQuCgogICAgUmV0dXJucyAod2NzLCBkaWFnbm9zdGljcykuCiAgICAiIiIKICAgIGltcG9ydCBudW1weSBhcyBucAogICAgaW1wb3J0IGFzdHJvYWxpZ24gYXMgYWEKICAgIGZyb20gYXN0cm9weS53Y3MudXRpbHMgaW1wb3J0IGZpdF93Y3NfZnJvbV9wb2ludHMKCiAgICB0cmFuc2Zvcm0sIChzcmMsIGRzdCkgPSBhYS5maW5kX3RyYW5zZm9ybSgKICAgICAgICBpbWFnZSwgcmVmZXJlbmNlX2ltYWdlLAogICAgICAgIG1heF9jb250cm9sX3BvaW50cz1tYXhfY29udHJvbF9wb2ludHMsCiAgICAgICAgZGV0ZWN0aW9uX3NpZ21hPWRldGVjdGlvbl9zaWdtYSkKCiAgICBza3kgPSByZWZlcmVuY2Vfd2NzLnBpeGVsX3RvX3dvcmxkKGRzdFs6LCAwXSwgZHN0WzosIDFdKQogICAgZml0dGVkID0gZml0X3djc19mcm9tX3BvaW50cygoc3JjWzosIDBdLCBzcmNbOiwgMV0pLCBza3ksIHByb2pfcG9pbnQ9ImNlbnRlciIpCgogICAgYmFja194LCBiYWNrX3kgPSBmaXR0ZWQud29ybGRfdG9fcGl4ZWwoc2t5KQogICAgcmVzaWR1YWwgPSBucC5oeXBvdChiYWNrX3ggLSBzcmNbOiwgMF0sIGJhY2tfeSAtIHNyY1s6LCAxXSkKCiAgICByZXR1cm4gZml0dGVkLCB7CiAgICAgICAgIm5fc3RhcnMiOiBpbnQobGVuKHNyYykpLAogICAgICAgICJtZWRpYW5fcmVzaWR1YWxfcHgiOiBmbG9hdChucC5tZWRpYW4ocmVzaWR1YWwpKSwKICAgICAgICAid29yc3RfcmVzaWR1YWxfcHgiOiBmbG9hdChyZXNpZHVhbC5tYXgoKSksCiAgICAgICAgInNoaWZ0X3B4IjogKGZsb2F0KHRyYW5zZm9ybS50cmFuc2xhdGlvblswXSksIGZsb2F0KHRyYW5zZm9ybS50cmFuc2xhdGlvblsxXSkpLAogICAgICAgICJyb3RhdGlvbl9kZWciOiBmbG9hdChucC5kZWdyZWVzKHRyYW5zZm9ybS5yb3RhdGlvbikpLAogICAgICAgICJzY2FsZSI6IGZsb2F0KHRyYW5zZm9ybS5zY2FsZSksCiAgICB9CgpkZWYgZGVzY3JpYmVfd2NzX2ZpdChuYW1lLCBkaWFnKToKICAgICIiIk9uZSByZWFkYWJsZSBwYXJhZ3JhcGggYWJvdXQgYSBzb2x1dGlvbiBsZW50IGZyb20gYW5vdGhlciBmcmFtZS4iIiIKICAgIHJldHVybiAoCiAgICAgICAgIiAge306IHNvbHZlZCBieSBtYXRjaGluZyB7fSBzdGFycyBhZ2FpbnN0IGFuIGV4cG9zdXJlIHRoYXQgaGFzIGEgc29sdXRpb25cbiIKICAgICAgICAiICAgIHNoaWZ0ICh7OisuMWZ9LCB7OisuMWZ9KSBweCwgcm90YXRpb24gezorLjNmfSBkZWcsIHNjYWxlIHs6LjVmfVxuIgogICAgICAgICIgICAgZml0IHJlc2lkdWFsOiBtZWRpYW4gezouMmZ9IHB4LCB3b3JzdCB7Oi4yZn0gcHgiLmZvcm1hdCgKICAgICAgICAgICAgbmFtZSwgZGlhZ1sibl9zdGFycyJdLCBkaWFnWyJzaGlmdF9weCJdWzBdLCBkaWFnWyJzaGlmdF9weCJdWzFdLAogICAgICAgICAgICBkaWFnWyJyb3RhdGlvbl9kZWciXSwgZGlhZ1sic2NhbGUiXSwKICAgICAgICAgICAgZGlhZ1sibWVkaWFuX3Jlc2lkdWFsX3B4Il0sIGRpYWdbIndvcnN0X3Jlc2lkdWFsX3B4Il0pKQoKZGVmIGRvbm9yX2ZvcihyZWNvcmQsIHJlY29yZHMpOgogICAgIiIiQSBzb2x2ZWQgZXhwb3N1cmUgdGhhdCBjYW4gbGVuZCBpdHMgcGxhdGUgc29sdXRpb24gdG8gYW4gdW5zb2x2ZWQgZnJhbWUuCgogICAgQW4gZXhwb3N1cmUgdGhyb3VnaCB0aGUgc2FtZSBmaWx0ZXIgaXMgcHJlZmVycmVkIC0gc2FtZSBzdGFycywgc2FtZSBkZXB0aCwKICAgIHRoZSBjbG9zZXN0IHRoaW5nIHRvIHRoZSBmcmFtZSB0aGF0IG5lZWRzIHNvbHZpbmcgLSBidXQgYW55IHNvbHZlZCBleHBvc3VyZQogICAgb2YgdGhlIGZpZWxkIHdpbGwgZG8sIHNpbmNlIHRoZSBtYXRjaCBpcyBtYWRlIG9uIHN0YXIgcG9zaXRpb25zIGFuZCB0aG9zZSBkbwogICAgbm90IGNhcmUgd2hpY2ggZmlsdGVyIHRoZXkgd2VyZSBzZWVuIHRocm91Z2guCiAgICAiIiIKICAgIGlmIHJlY29yZCBpcyBOb25lIG9yIHJlY29yZC5nZXQoIndjcyIpOgogICAgICAgIHJldHVybiBOb25lCiAgICBzYW1lX2ZpbHRlciA9IFtyIGZvciByIGluIHJlY29yZHMKICAgICAgICAgICAgICAgICAgIGlmIHIuZ2V0KCJ3Y3MiKSBhbmQgci5nZXQoInJvbGUiKSA9PSByZWNvcmQuZ2V0KCJyb2xlIildCiAgICBpZiBzYW1lX2ZpbHRlcjoKICAgICAgICByZXR1cm4gc2FtZV9maWx0ZXJbMF0KICAgIHNvbHZlZCA9IFtyIGZvciByIGluIHJlY29yZHMgaWYgci5nZXQoIndjcyIpXQogICAgcmV0dXJuIHNvbHZlZFswXSBpZiBzb2x2ZWQgZWxzZSBOb25lCgpkZWYgcGxhbihmb2xkZXIsIHJvbGVzPVJPTEVTLCBhc2s9Tm9uZSwgcmVjdXJzaXZlPVRydWUpOgogICAgIiIiRXZlcnl0aGluZyBhIHRvb2wgbmVlZHMgZnJvbSBvbmUgZm9sZGVyIHBhdGguCgogICAgUmV0dXJucyAocmVjb3JkcywgY2hvc2VuLCBtZW1iZXJzLCBkb25vcnMsIG5vdGVzKTogd2hhdCBpcyBpbiB0aGUgZm9sZGVyLCB0aGUKICAgIGZyYW1lIGNob3NlbiBmb3IgZWFjaCByb2xlLCB0aGUgc2V2ZXJhbCBleHBvc3VyZXMgdG8gY29tYmluZSB3aGVuIHRoZXJlIHdhcyBubwogICAgY29tYmluZWQgZnJhbWUsIHdoaWNoIHNvbHZlZCBleHBvc3VyZSB3aWxsIGxlbmQgYSBwbGF0ZSBzb2x1dGlvbiB0byBhIGZyYW1lCiAgICB0aGF0IGxhY2tzIG9uZSwgYW5kIGFueXRoaW5nIHdvcnRoIHNheWluZyBvdXQgbG91ZC4KICAgICIiIgogICAgcmVjb3JkcyA9IHNjYW5fZm9sZGVyKGZvbGRlciwgcmVjdXJzaXZlPXJlY3Vyc2l2ZSkKICAgIGlmIG5vdCByZWNvcmRzOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigibm8gRklUUyBmaWxlcyBpbiAiICsgc3RyKGZvbGRlcikpCiAgICBjaG9zZW4sIG1lbWJlcnMsIG5vdGVzID0gY2hvb3NlKHJlY29yZHMsIHJvbGVzPXJvbGVzLCBhc2s9YXNrKQogICAgZG9ub3JzID0ge3JvbGU6IGRvbm9yX2ZvcihyZWMsIHJlY29yZHMpIGZvciByb2xlLCByZWMgaW4gY2hvc2VuLml0ZW1zKCl9CiAgICBmb3Igcm9sZSwgZG9ub3IgaW4gZG9ub3JzLml0ZW1zKCk6CiAgICAgICAgaWYgZG9ub3IgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIG5vdGVzLmFwcGVuZCgie306IGhhcyBubyBwbGF0ZSBzb2x1dGlvbjsgd2lsbCBib3Jyb3cgb25lIGZyb20ge30iLmZvcm1hdCgKICAgICAgICAgICAgICAgIHJvbGUsIGRvbm9yWyJuYW1lIl0pKQogICAgICAgIGVsaWYgcm9sZSBpbiBjaG9zZW4gYW5kIG5vdCBjaG9zZW5bcm9sZV1bIndjcyJdOgogICAgICAgICAgICBub3Rlcy5hcHBlbmQoInt9OiBoYXMgbm8gcGxhdGUgc29sdXRpb24gYW5kIHRoZXJlIGlzIG5vIHNvbHZlZCAiCiAgICAgICAgICAgICAgICAgICAgICAgICAiZXhwb3N1cmUgaW4gdGhlIGZvbGRlciB0byB0YWtlIG9uZSBmcm9tIi5mb3JtYXQocm9sZSkpCiAgICByZXR1cm4gcmVjb3JkcywgY2hvc2VuLCBtZW1iZXJzLCBkb25vcnMsIG5vdGVzCgpkZXNjcmliZV9fZnJhbWVfaW52ZW50b3J5ID0gZGVzY3JpYmUgICAjIHRoaXMgbW9kdWxlJ3Mgb3duOyB0aGUgcGxhaW4gbmFtZSBnZXRzIHNoYWRvd2VkIGJlbG93CgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIHNoYXJlZCBtb2R1bGUgYW5jaG9yX3JlZ2lzdHJhdGlvbi5weQojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCgpkZWYgcGlja19hbmNob3JzKHBvc2l0aW9ucywgc2hhcGUsIG5fYW5jaG9ycz00LCBlZGdlX21hcmdpbj00MCk6CiAgICAiIiJDaG9vc2UgYSBmZXcgd2lkZWx5IHNlcGFyYXRlZCBzdGFycyBmcm9tIGEgbGlzdCBvZiBjYW5kaWRhdGUgcG9zaXRpb25zLgoKICAgIFNwcmVhZCBpcyB0aGUgd2hvbGUgcG9pbnQuIFRoZSByb3RhdGlvbiBhbmdsZSBpcyBtZWFzdXJlZCBhcyBhIGRpZmZlcmVuY2Ugb2YKICAgIGRpcmVjdGlvbnMgYmV0d2VlbiB0d28gc3RhcnMsIHNvIGl0cyBwcmVjaXNpb24gaXMgdGhlIGNlbnRyaW5nIHByZWNpc2lvbgogICAgZGl2aWRlZCBieSB0aGUgZGlzdGFuY2UgYmV0d2VlbiB0aGVtOiBhbmNob3JzIGNsb3NlIHRvZ2V0aGVyIG1lYXN1cmUgdGhlCiAgICBhbmdsZSBiYWRseSBubyBtYXR0ZXIgaG93IHdlbGwgZWFjaCBvbmUgaXMgY2VudHJlZC4KCiAgICBUaGUgZnJhbWUgaXMgZGl2aWRlZCBpbnRvIHF1YWRyYW50cyBhbmQgb25lIHN0YXIgaXMgdGFrZW4gZnJvbSBlYWNoLCBhcyBmYXIKICAgIGZyb20gdGhlIGNlbnRyZSBhcyBpdCBjYW4gYmUgZm91bmQuIFRoYXQgZ3VhcmFudGVlcyB0aGUgc3ByZWFkIGluc3RlYWQgb2YKICAgIGhvcGluZyBmb3IgaXQuCiAgICAiIiIKICAgIHBvc2l0aW9ucyA9IG5wLmFzYXJyYXkocG9zaXRpb25zLCBkdHlwZT1mbG9hdCkKICAgIGlmIGxlbihwb3NpdGlvbnMpID09IDA6CiAgICAgICAgcmV0dXJuIHBvc2l0aW9ucwoKICAgIG55LCBueCA9IHNoYXBlCiAgICBpbnNpZGUgPSAoKHBvc2l0aW9uc1s6LCAwXSA+IGVkZ2VfbWFyZ2luKSAmIChwb3NpdGlvbnNbOiwgMF0gPCBueCAtIGVkZ2VfbWFyZ2luKQogICAgICAgICAgICAgICYgKHBvc2l0aW9uc1s6LCAxXSA+IGVkZ2VfbWFyZ2luKSAmIChwb3NpdGlvbnNbOiwgMV0gPCBueSAtIGVkZ2VfbWFyZ2luKSkKICAgIHBvc2l0aW9ucyA9IHBvc2l0aW9uc1tpbnNpZGVdCiAgICBpZiBsZW4ocG9zaXRpb25zKSA8PSBuX2FuY2hvcnM6CiAgICAgICAgcmV0dXJuIHBvc2l0aW9ucwoKICAgIGN4LCBjeSA9IG54IC8gMi4wLCBueSAvIDIuMAogICAgZHgsIGR5ID0gcG9zaXRpb25zWzosIDBdIC0gY3gsIHBvc2l0aW9uc1s6LCAxXSAtIGN5CiAgICByYWRpdXMgPSBucC5oeXBvdChkeCwgZHkpCgogICAgY2hvc2VuID0gW10KICAgIHF1YWRyYW50cyA9ICgoZHggPj0gMCkgJiAoZHkgPj0gMCksIChkeCA8IDApICYgKGR5ID49IDApLAogICAgICAgICAgICAgICAgIChkeCA8IDApICYgKGR5IDwgMCksIChkeCA+PSAwKSAmIChkeSA8IDApKQogICAgZm9yIHEgaW4gcXVhZHJhbnRzOgogICAgICAgIGlmIHEuYW55KCk6CiAgICAgICAgICAgIGNob3Nlbi5hcHBlbmQocG9zaXRpb25zW25wLndoZXJlKHEpWzBdW25wLmFyZ21heChyYWRpdXNbcV0pXV0pCgogICAgIyBBIGZpZWxkIHdpdGggYW4gZW1wdHkgcXVhZHJhbnQgc3RpbGwgZ2V0cyBpdHMgZm91ciwgdGFrZW4gZnJvbSB3aGVyZXZlciB0aGUKICAgICMgc3RhcnMgYWN0dWFsbHkgYXJlLgogICAgaWYgbGVuKGNob3NlbikgPCBuX2FuY2hvcnM6CiAgICAgICAgb3JkZXIgPSBucC5hcmdzb3J0KHJhZGl1cylbOjotMV0KICAgICAgICBmb3IgaSBpbiBvcmRlcjoKICAgICAgICAgICAgcCA9IHBvc2l0aW9uc1tpXQogICAgICAgICAgICBpZiBub3QgYW55KG5wLmFsbGNsb3NlKHAsIGMpIGZvciBjIGluIGNob3Nlbik6CiAgICAgICAgICAgICAgICBjaG9zZW4uYXBwZW5kKHApCiAgICAgICAgICAgIGlmIGxlbihjaG9zZW4pID49IG5fYW5jaG9yczoKICAgICAgICAgICAgICAgIGJyZWFrCgogICAgcmV0dXJuIG5wLmFycmF5KGNob3Nlbls6bl9hbmNob3JzXSkKCmRlZiBjZW50cmVfb24oaW1hZ2UsIHBvc2l0aW9ucywgYm94PTE1KToKICAgICIiIldoZXJlIGVhY2ggc3RhciBhY3R1YWxseSBzaXRzIGluIHRoaXMgZnJhbWUsIHRvIGEgZnJhY3Rpb24gb2YgYSBwaXhlbC4iIiIKICAgIGZyb20gcGhvdHV0aWxzLmNlbnRyb2lkcyBpbXBvcnQgY2VudHJvaWRfc291cmNlcywgY2VudHJvaWRfY29tCgogICAgcG9zaXRpb25zID0gbnAuYXNhcnJheShwb3NpdGlvbnMsIGR0eXBlPWZsb2F0KQogICAgaWYgbm90IGxlbihwb3NpdGlvbnMpOgogICAgICAgIHJldHVybiBucC5lbXB0eSgoMCwgMikpCiAgICB0cnk6CiAgICAgICAgeCwgeSA9IGNlbnRyb2lkX3NvdXJjZXMobnAubmFuX3RvX251bShpbWFnZSksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcG9zaXRpb25zWzosIDBdLCBwb3NpdGlvbnNbOiwgMV0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYm94X3NpemU9Ym94LCBjZW50cm9pZF9mdW5jPWNlbnRyb2lkX2NvbSkKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcmV0dXJuIG5wLmVtcHR5KCgwLCAyKSkKICAgIHJldHVybiBucC5jb2x1bW5fc3RhY2soW3gsIHldKQoKZGVmIGFuY2hvcl90cmFuc2Zvcm0ocmVmZXJlbmNlLCBpbWFnZSwgcG9zaXRpb25zLCBib3g9MTUsIG1heF9tb3ZlX3B4PTguMCwKICAgICAgICAgICAgICAgICAgICAgbl9hbmNob3JzPTQpOgogICAgIiIiRml0IHNoaWZ0LCByb3RhdGlvbiBhbmQgc2NhbGUgZnJvbSBhIGZldyBhbmNob3Igc3RhcnMuCgogICAgUmV0dXJucyAodHJhbnNmb3JtLCBkaWFnbm9zdGljcykgd2hlcmUgdHJhbnNmb3JtIG1hcHMgYSBwb3NpdGlvbiBpbiBgaW1hZ2VgCiAgICB0byB0aGUgbWF0Y2hpbmcgcG9zaXRpb24gaW4gYHJlZmVyZW5jZWAsIG9yIChOb25lLCBkaWFnbm9zdGljcykgd2hlbiB0aGUKICAgIGFuY2hvcnMgY291bGQgbm90IGJlIG1lYXN1cmVkIHdlbGwgZW5vdWdoIHRvIHRydXN0LgoKICAgIFRoZSBvZGQgb25lIG91dCBpcyBkcm9wcGVkOiBlYWNoIGFuY2hvcidzIHJlc2lkdWFsIGFnYWluc3QgdGhlIGZpdCBpcwogICAgbWVhc3VyZWQsIGFuZCBpZiBvbmUgc3RhbmRzIGZhciBhYm92ZSB0aGUgcmVzdCBpdCBpcyByZW1vdmVkIGFuZCB0aGUgZml0CiAgICByZWRvbmUuIFRoYXQgaXMgd2hhdCB0aGUgZm91cnRoIGFuY2hvciBpcyBmb3IuCiAgICAiIiIKICAgIGFuY2hvcnMgPSBwaWNrX2FuY2hvcnMocG9zaXRpb25zLCByZWZlcmVuY2Uuc2hhcGUsIG5fYW5jaG9ycz1uX2FuY2hvcnMpCiAgICBkaWFnID0geyJuX2FuY2hvcnMiOiBsZW4oYW5jaG9ycyksICJkcm9wcGVkIjogMH0KICAgIGlmIGxlbihhbmNob3JzKSA8IDI6CiAgICAgICAgZGlhZ1sid2h5Il0gPSAiZmV3ZXIgdGhhbiB0d28gdXNhYmxlIGFuY2hvciBzdGFycyIKICAgICAgICByZXR1cm4gTm9uZSwgZGlhZwoKICAgIGEgPSBjZW50cmVfb24ocmVmZXJlbmNlLCBhbmNob3JzLCBib3g9Ym94KQogICAgYiA9IGNlbnRyZV9vbihpbWFnZSwgYW5jaG9ycywgYm94PWJveCkKICAgIGlmIGxlbihhKSAhPSBsZW4oYikgb3Igbm90IGxlbihhKToKICAgICAgICBkaWFnWyJ3aHkiXSA9ICJ0aGUgYW5jaG9ycyBjb3VsZCBub3QgYmUgY2VudHJlZCIKICAgICAgICByZXR1cm4gTm9uZSwgZGlhZwoKICAgIG9rID0gKG5wLmlzZmluaXRlKGEpLmFsbChheGlzPTEpICYgbnAuaXNmaW5pdGUoYikuYWxsKGF4aXM9MSkKICAgICAgICAgICYgKG5wLmh5cG90KCooYiAtIGEpLlQpIDwgbWF4X21vdmVfcHgpKQogICAgYSwgYiA9IGFbb2tdLCBiW29rXQogICAgaWYgbGVuKGEpIDwgMjoKICAgICAgICBkaWFnWyJ3aHkiXSA9ICJ0aGUgYW5jaG9ycyBtb3ZlZCBmdXJ0aGVyIHRoYW4gYSBmcmFtZSBsaWtlIHRoaXMgc2hvdWxkIgogICAgICAgIHJldHVybiBOb25lLCBkaWFnCgogICAgZGVmIGZpdChwLCBxKToKICAgICAgICAiIiJMZWFzdC1zcXVhcmVzIHNpbWlsYXJpdHkgdHJhbnNmb3JtIHRha2luZyBxIG9udG8gcC4iIiIKICAgICAgICBwYywgcWMgPSBwLm1lYW4oYXhpcz0wKSwgcS5tZWFuKGF4aXM9MCkKICAgICAgICBwMCwgcTAgPSBwIC0gcGMsIHEgLSBxYwogICAgICAgIGRlbm9tID0gKHEwICoqIDIpLnN1bSgpCiAgICAgICAgaWYgZGVub20gPD0gMDoKICAgICAgICAgICAgcmV0dXJuIE5vbmUKICAgICAgICAjIFRoZSByb3RhdGlvbi1hbmQtc2NhbGUgcGFydCwgYXMgb25lIGNvbXBsZXggbXVsdGlwbGljYXRpb24uCiAgICAgICAgbnVtID0gY29tcGxleCgocDBbOiwgMF0gKiBxMFs6LCAwXSArIHAwWzosIDFdICogcTBbOiwgMV0pLnN1bSgpLAogICAgICAgICAgICAgICAgICAgICAgKHAwWzosIDFdICogcTBbOiwgMF0gLSBwMFs6LCAwXSAqIHEwWzosIDFdKS5zdW0oKSkKICAgICAgICB6ID0gbnVtIC8gZGVub20KICAgICAgICByZXR1cm4geiwgcGMsIHFjCgogICAgZ290ID0gZml0KGEsIGIpCiAgICBpZiBnb3QgaXMgTm9uZToKICAgICAgICBkaWFnWyJ3aHkiXSA9ICJ0aGUgYW5jaG9ycyBhcmUgb24gdG9wIG9mIGVhY2ggb3RoZXIiCiAgICAgICAgcmV0dXJuIE5vbmUsIGRpYWcKICAgIHosIHBjLCBxYyA9IGdvdAoKICAgIGlmIGxlbihhKSA+PSA0OgogICAgICAgIHByZWQgPSBfYXBwbHkoYiwgeiwgcGMsIHFjKQogICAgICAgIHJlc2lkID0gbnAuaHlwb3QoKihwcmVkIC0gYSkuVCkKICAgICAgICB3b3JzdCA9IGludChucC5hcmdtYXgocmVzaWQpKQogICAgICAgIG90aGVycyA9IG5wLmRlbGV0ZShyZXNpZCwgd29yc3QpCiAgICAgICAgaWYgcmVzaWRbd29yc3RdID4gbWF4KDMuMCAqIG5wLm1lZGlhbihvdGhlcnMpLCAwLjUpOgogICAgICAgICAgICBhLCBiID0gbnAuZGVsZXRlKGEsIHdvcnN0LCAwKSwgbnAuZGVsZXRlKGIsIHdvcnN0LCAwKQogICAgICAgICAgICBkaWFnWyJkcm9wcGVkIl0gPSAxCiAgICAgICAgICAgIHosIHBjLCBxYyA9IGZpdChhLCBiKQoKICAgIHByZWQgPSBfYXBwbHkoYiwgeiwgcGMsIHFjKQogICAgcmVzaWQgPSBucC5oeXBvdCgqKHByZWQgLSBhKS5UKQogICAgZGlhZy51cGRhdGUoewogICAgICAgICJuX3VzZWQiOiBsZW4oYSksCiAgICAgICAgInNoaWZ0X3B4IjogKGZsb2F0KHBjWzBdIC0gcWNbMF0gKiB6LnJlYWwgKyBxY1sxXSAqIHouaW1hZyksCiAgICAgICAgICAgICAgICAgICAgIGZsb2F0KHBjWzFdIC0gcWNbMF0gKiB6LmltYWcgLSBxY1sxXSAqIHoucmVhbCkpLAogICAgICAgICJyb3RhdGlvbl9kZWciOiBmbG9hdChucC5kZWdyZWVzKG5wLmFyY3RhbjIoei5pbWFnLCB6LnJlYWwpKSksCiAgICAgICAgInNjYWxlIjogZmxvYXQoYWJzKHopKSwKICAgICAgICAicmVzaWR1YWxfcHgiOiBmbG9hdChucC5tYXgocmVzaWQpKSwKICAgICAgICAic2VwYXJhdGlvbl9weCI6IGZsb2F0KG5wLmh5cG90KCooYS5tYXgoYXhpcz0wKSAtIGEubWluKGF4aXM9MCkpKSksCiAgICB9KQogICAgcmV0dXJuICh6LCBwYywgcWMpLCBkaWFnCgpkZWYgX2FwcGx5KHBvaW50cywgeiwgcGMsIHFjKToKICAgICIiIk1hcCBwb2ludHMgdGhyb3VnaCBhIGZpdHRlZCBzaW1pbGFyaXR5IHRyYW5zZm9ybS4iIiIKICAgIHEwID0gbnAuYXNhcnJheShwb2ludHMsIGR0eXBlPWZsb2F0KSAtIHFjCiAgICB4ID0gcTBbOiwgMF0gKiB6LnJlYWwgLSBxMFs6LCAxXSAqIHouaW1hZyArIHBjWzBdCiAgICB5ID0gcTBbOiwgMF0gKiB6LmltYWcgKyBxMFs6LCAxXSAqIHoucmVhbCArIHBjWzFdCiAgICByZXR1cm4gbnAuY29sdW1uX3N0YWNrKFt4LCB5XSkKCmRlZiBhcHBseV90cmFuc2Zvcm0oaW1hZ2UsIHRyYW5zZm9ybSwgc2hhcGU9Tm9uZSk6CiAgICAiIiJSZXNhbXBsZSBhIGZyYW1lIHRocm91Z2ggYSB0cmFuc2Zvcm0gZml0dGVkIGZyb20gdGhlIGFuY2hvcnMuCgogICAgT25lIGFmZmluZSBpbnRlcnBvbGF0aW9uLCBhbmQgbm8gYXN0cm9tZXRyaWMgYXJpdGhtZXRpYyBhdCBhbGwuIFRoYXQgaXMgd2hlcmUKICAgIHRoZSB0aW1lIGdvZXM6IHRoZSB2ZXJzaW9uIHRoYXQgd2VudCB0aHJvdWdoIGJvdGggcGxhdGUgc29sdXRpb25zIGhhZCB0byB0dXJuCiAgICBzaXh0eS1vbmUgbWlsbGlvbiBwaXhlbCBwb3NpdGlvbnMgaW50byBza3kgcG9zaXRpb25zIGFuZCBiYWNrIGFnYWluIGJlZm9yZSBpdAogICAgY291bGQgaW50ZXJwb2xhdGUgYW55dGhpbmcuCiAgICAiIiIKICAgIGZyb20gc2NpcHkubmRpbWFnZSBpbXBvcnQgYWZmaW5lX3RyYW5zZm9ybQoKICAgIHosIHBjLCBxYyA9IHRyYW5zZm9ybQogICAgc2hhcGUgPSBzaGFwZSBvciBpbWFnZS5zaGFwZQoKICAgICMgYWZmaW5lX3RyYW5zZm9ybSBwdWxsczogZm9yIGVhY2ggb3V0cHV0IHBpeGVsIGl0IGFza3Mgd2hlcmUgaW4gdGhlIGlucHV0IHRvCiAgICAjIGxvb2ssIHNvIHRoZSBpbnZlcnNlIG9mIHRoZSBmaXR0ZWQgbWFwIGlzIHdoYXQgaXQgbmVlZHMsIHdyaXR0ZW4gaW4gdGhlCiAgICAjIChyb3csIGNvbHVtbikgb3JkZXIgaXQgZXhwZWN0cy4KICAgIGludiA9IDEuMCAvIHoKICAgIG0gPSBucC5hcnJheShbW2ludi5yZWFsLCBpbnYuaW1hZ10sIFstaW52LmltYWcsIGludi5yZWFsXV0sIGR0eXBlPWZsb2F0KQogICAgb2Zmc2V0ID0gbnAuYXJyYXkoW3FjWzFdLCBxY1swXV0pIC0gbS5kb3QobnAuYXJyYXkoW3BjWzFdLCBwY1swXV0pKQoKICAgIGtub3duID0gbnAuaXNmaW5pdGUoaW1hZ2UpCiAgICBmaWxsZWQgPSBucC53aGVyZShrbm93biwgaW1hZ2UsIDAuMCkuYXN0eXBlKG5wLmZsb2F0MzIpCiAgICBvdXQgPSBhZmZpbmVfdHJhbnNmb3JtKGZpbGxlZCwgbSwgb2Zmc2V0PW9mZnNldCwgb3V0cHV0X3NoYXBlPXNoYXBlLAogICAgICAgICAgICAgICAgICAgICAgICAgICBvcmRlcj0xLCBtb2RlPSJjb25zdGFudCIsIGN2YWw9bnAubmFuKQogICAgc2VlbiA9IGFmZmluZV90cmFuc2Zvcm0oa25vd24uYXN0eXBlKG5wLmZsb2F0MzIpLCBtLCBvZmZzZXQ9b2Zmc2V0LAogICAgICAgICAgICAgICAgICAgICAgICAgICAgb3V0cHV0X3NoYXBlPXNoYXBlLCBvcmRlcj0xLCBtb2RlPSJjb25zdGFudCIsIGN2YWw9MC4wKQogICAgb3V0W3NlZW4gPCAwLjVdID0gbnAubmFuCiAgICByZXR1cm4gb3V0CgpkZWYgZGVzY3JpYmUoZGlhZyk6CiAgICAiIiJUaGUgcmVnaXN0cmF0aW9uLCBpbiBhIGxpbmUgd29ydGggcHJpbnRpbmcuIiIiCiAgICBpZiBkaWFnLmdldCgid2h5Iik6CiAgICAgICAgcmV0dXJuICIgIGFuY2hvcnMgdW51c2FibGUgKHt9KTsgZmFsbGluZyBiYWNrIi5mb3JtYXQoZGlhZ1sid2h5Il0pCiAgICByZXR1cm4gKCIgIHt9IGFuY2hvciBzdGFyc3t9LCB7Oi4wZn0gcHggYXBhcnQ6IHNoaWZ0ICh7OisuMmZ9LCB7OisuMmZ9KSBweCwgIgogICAgICAgICAgICAicm90YXRpb24gezorLjRmfSBkZWcsIHNjYWxlIHs6LjZmfTsgd29yc3QgYW5jaG9yIG9mZiBieSB7Oi4yZn0gcHgiCiAgICAgICAgICAgIC5mb3JtYXQoZGlhZ1sibl91c2VkIl0sCiAgICAgICAgICAgICAgICAgICAgIiAob25lIGRyb3BwZWQpIiBpZiBkaWFnLmdldCgiZHJvcHBlZCIpIGVsc2UgIiIsCiAgICAgICAgICAgICAgICAgICAgZGlhZ1sic2VwYXJhdGlvbl9weCJdLCBkaWFnWyJzaGlmdF9weCJdWzBdLCBkaWFnWyJzaGlmdF9weCJdWzFdLAogICAgICAgICAgICAgICAgICAgIGRpYWdbInJvdGF0aW9uX2RlZyJdLCBkaWFnWyJzY2FsZSJdLCBkaWFnWyJyZXNpZHVhbF9weCJdKSkKCmRlc2NyaWJlX19hbmNob3JfcmVnaXN0cmF0aW9uID0gZGVzY3JpYmUgICAjIHRoaXMgbW9kdWxlJ3Mgb3duOyB0aGUgcGxhaW4gbmFtZSBnZXRzIHNoYWRvd2VkIGJlbG93CgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIHNoYXJlZCBtb2R1bGUgZ2FsYXh5X3dpbmRvdy5weQojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCgpkZWYgZ2FsYXh5X3dpbmRvdyhzaGFwZSwgY2VudHJlLCByMjVfcHgsIG1hcmdpbj0xLjUpOgogICAgIiIiVGhlIHNsaWNlIG9mIGEgZnJhbWUgd29ydGggd29ya2luZyBvbiwgYXMgKHkwLCB5MSwgeDAsIHgxKS4KCiAgICBSZXR1cm5zIE5vbmUgd2hlbiB0aGVyZSBpcyBubyByYWRpdXMgdG8gd29yayBmcm9tLCBvciB3aGVuIHRoZSB3aW5kb3cgd291bGQKICAgIGNvdmVyIGVzc2VudGlhbGx5IHRoZSB3aG9sZSBmcmFtZSBhbnl3YXkgLSBpbiB3aGljaCBjYXNlIGNyb3BwaW5nIHdvdWxkIGFkZCBhCiAgICBib29ra2VlcGluZyBzdGVwIGFuZCBzYXZlIG5vdGhpbmcuCiAgICAiIiIKICAgIGlmIG5vdCByMjVfcHggb3IgcjI1X3B4IDw9IDA6CiAgICAgICAgcmV0dXJuIE5vbmUKCiAgICBueSwgbnggPSBzaGFwZQogICAgcmVhY2ggPSBtYXJnaW4gKiByMjVfcHgKICAgIHgwID0gaW50KG1heCgwLCBucC5mbG9vcihjZW50cmVbMF0gLSByZWFjaCkpKQogICAgeDEgPSBpbnQobWluKG54LCBucC5jZWlsKGNlbnRyZVswXSArIHJlYWNoKSkpCiAgICB5MCA9IGludChtYXgoMCwgbnAuZmxvb3IoY2VudHJlWzFdIC0gcmVhY2gpKSkKICAgIHkxID0gaW50KG1pbihueSwgbnAuY2VpbChjZW50cmVbMV0gKyByZWFjaCkpKQoKICAgIGlmIHgxIC0geDAgPCA2NCBvciB5MSAtIHkwIDwgNjQ6CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIGlmICh4MSAtIHgwKSAqICh5MSAtIHkwKSA+IDAuOSAqIG54ICogbnk6CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIHJldHVybiAoeTAsIHkxLCB4MCwgeDEpCgpkZWYgY3V0KGRhdGEsIHdpbmRvdyk6CiAgICAiIiJUaGUgd2luZG93J3Mgd29ydGggb2YgYSBmcmFtZS4iIiIKICAgIGlmIHdpbmRvdyBpcyBOb25lIG9yIGRhdGEgaXMgTm9uZToKICAgICAgICByZXR1cm4gZGF0YQogICAgeTAsIHkxLCB4MCwgeDEgPSB3aW5kb3cKICAgIHJldHVybiBkYXRhW3kwOnkxLCB4MDp4MV0KCmRlZiBzaGlmdF9jZW50cmUoY2VudHJlLCB3aW5kb3cpOgogICAgIiIiV2hlcmUgdGhlIGdhbGF4eSdzIGNlbnRyZSBzaXRzIG9uY2UgdGhlIGZyYW1lIGhhcyBiZWVuIGN1dC4iIiIKICAgIGlmIHdpbmRvdyBpcyBOb25lOgogICAgICAgIHJldHVybiBjZW50cmUKICAgIHkwLCBfLCB4MCwgXyA9IHdpbmRvdwogICAgcmV0dXJuIChjZW50cmVbMF0gLSB4MCwgY2VudHJlWzFdIC0geTApCgpkZWYgc2hpZnRfcG9zaXRpb25zKHBvc2l0aW9ucywgd2luZG93LCBzaGFwZT1Ob25lKToKICAgICIiIlN0YXIgcG9zaXRpb25zIG1vdmVkIGludG8gdGhlIGN1dCBmcmFtZSwga2VlcGluZyBvbmx5IHRob3NlIGluc2lkZSBpdC4iIiIKICAgIGlmIHdpbmRvdyBpcyBOb25lIG9yIHBvc2l0aW9ucyBpcyBOb25lIG9yIG5vdCBsZW4ocG9zaXRpb25zKToKICAgICAgICByZXR1cm4gcG9zaXRpb25zCiAgICB5MCwgeTEsIHgwLCB4MSA9IHdpbmRvdwogICAgcCA9IG5wLmFzYXJyYXkocG9zaXRpb25zLCBkdHlwZT1mbG9hdCkKICAgIGluc2lkZSA9ICgocFs6LCAwXSA+PSB4MCkgJiAocFs6LCAwXSA8IHgxKSAmIChwWzosIDFdID49IHkwKSAmIChwWzosIDFdIDwgeTEpKQogICAgcmV0dXJuIG5wLmNvbHVtbl9zdGFjayhbcFtpbnNpZGUsIDBdIC0geDAsIHBbaW5zaWRlLCAxXSAtIHkwXSkKCmRlZiByZXN0b3JlX3Bvc2l0aW9ucyh4LCB5LCB3aW5kb3cpOgogICAgIiIiUG9zaXRpb25zIG1lYXN1cmVkIGluIHRoZSBjdXQgZnJhbWUsIHB1dCBiYWNrIHdoZXJlIHRoZXkgYmVsb25nLgoKICAgIEV2ZXJ5dGhpbmcgdGhlIHRvb2wgcmVwb3J0cyAtIGNhdGFsb2d1ZXMsIHBsb3RzLCB0aGUgcGFpcmluZyB3aXRoIHRoZSBibHVlCiAgICBrbm90cyAtIGlzIGluIHRoZSBjb29yZGluYXRlcyBvZiB0aGUgb3JpZ2luYWwgZnJhbWUsIHNvIG1lYXN1cmVtZW50cyBtYWRlIGluCiAgICB0aGUgd2luZG93IGhhdmUgdG8gY29tZSBiYWNrIG91dCBvZiBpdC4gRm9yZ2V0dGluZyB0aGlzIHdvdWxkIHNoaWZ0IGV2ZXJ5CiAgICBwb3NpdGlvbiBieSB0aGUgc2l6ZSBvZiB0aGUgY3JvcCwgd2hpY2ggaXMgdGhlIGtpbmQgb2YgZXJyb3IgdGhhdCBsb29rcyBsaWtlCiAgICBhIGRpc2NvdmVyeS4KICAgICIiIgogICAgaWYgd2luZG93IGlzIE5vbmU6CiAgICAgICAgcmV0dXJuIHgsIHkKICAgIHkwLCBfLCB4MCwgXyA9IHdpbmRvdwogICAgcmV0dXJuIG5wLmFzYXJyYXkoeCwgZHR5cGU9ZmxvYXQpICsgeDAsIG5wLmFzYXJyYXkoeSwgZHR5cGU9ZmxvYXQpICsgeTAKCmRlZiBkZXNjcmliZSh3aW5kb3csIHNoYXBlLCBtYXJnaW4pOgogICAgIiIiV2hhdCB3YXMgY3V0LCBhbmQgd2hhdCBpdCBzYXZlZC4iIiIKICAgIGlmIHdpbmRvdyBpcyBOb25lOgogICAgICAgIHJldHVybiAoIiAgd29ya2luZyBvbiB0aGUgd2hvbGUgZnJhbWUgLSBlaXRoZXIgbm8gaXNvcGhvdGFsIHJhZGl1cyBpcyAiCiAgICAgICAgICAgICAgICAia25vd24sIG9yIHRoZSBnYWxheHkgZmlsbHMgaXQiKQogICAgeTAsIHkxLCB4MCwgeDEgPSB3aW5kb3cKICAgIGtlcHQgPSAoeTEgLSB5MCkgKiAoeDEgLSB4MCkKICAgIHdob2xlID0gc2hhcGVbMF0gKiBzaGFwZVsxXQogICAgcmV0dXJuICgiICB3b3JraW5nIG9uIHt9IHgge30gcGl4ZWxzIGFyb3VuZCB0aGUgZ2FsYXh5LCB7Oi4wZn0lIG9mIHRoZSBmcmFtZSAiCiAgICAgICAgICAgICIoezouMWZ9IHggUjI1KTsgdGhlIHJlc3QgaXMgc2t5IHRoYXQgd291bGQgYmUgZGlzY2FyZGVkIGFueXdheSIKICAgICAgICAgICAgLmZvcm1hdCh4MSAtIHgwLCB5MSAtIHkwLCAxMDAuMCAqIGtlcHQgLyB3aG9sZSwgbWFyZ2luKSkKCmRlc2NyaWJlX19nYWxheHlfd2luZG93ID0gZGVzY3JpYmUgICAjIHRoaXMgbW9kdWxlJ3Mgb3duOyB0aGUgcGxhaW4gbmFtZSBnZXRzIHNoYWRvd2VkIGJlbG93CgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIHNoYXJlZCBtb2R1bGUgZ2FsYXh5X2V4dGVudC5weQojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCgpkZWYgc3VyZmFjZV9icmlnaHRuZXNzX3Byb2ZpbGUoZGF0YSwgY2VudHJlLCByX21heD1Ob25lLCBuX3JpbmdzPTQwLCBzdGVwPTQsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBwZXJjZW50aWxlPTkwLjApOgogICAgIiIiSG93IG11Y2ggbGlnaHQgZWFjaCByaW5nIGhvbGRzLCBhYm92ZSB3aGF0IGFuIGVtcHR5IHJpbmcgaG9sZHMuCgogICAgV0hZIEEgSElHSCBQRVJDRU5USUxFIEFORCBOT1QgVEhFIE1FRElBTgogICAgQmVjYXVzZSBvZiB3aGF0IGlzIGJlaW5nIG1lYXN1cmVkLiBBIHN0ZWxsYXIgZGlzayBpcyBzbW9vdGgsIGFuZCBpdHMgbWVkaWFuCiAgICBkZXNjcmliZXMgaXQuIExpbmUgZW1pc3Npb24gaXMgbm90IHNtb290aDogaXQgbGl2ZXMgaW4gZGlzY3JldGUgcmVnaW9ucyB3aXRoCiAgICBlbXB0eSBza3kgYmV0d2VlbiB0aGVtLCBzbyBhIHJpbmcgaG9sZGluZyBhIGRvemVuIGJyaWdodCBISUkgcmVnaW9ucyBhbmQKICAgIHR3ZW50eSB0aG91c2FuZCBlbXB0eSBwaXhlbHMgaGFzIGEgbWVkaWFuIG9mIHplcm8uIE1lYXN1cmVkIG9uIE04MywgdGhlIG1lZGlhbgogICAgb2YgdGhlIHN1YnRyYWN0ZWQgZnJhbWUgcmVhY2hlZCB0aGUgbm9pc2UgYXQgMjU2IHB4IHdoaWxlIHRoZSByZWdpb25zCiAgICB0aGVtc2VsdmVzIHdlbnQgb24gdG8gNTU2IC0gYW5kIHVzaW5nIHRoZSBtZWRpYW4gY3V0IHRoZSBjYXRhbG9ndWUgZnJvbSAzMzUKICAgIHJlZ2lvbnMgdG8gMTIyLgoKICAgIFRoZSBuaW5ldGlldGggcGVyY2VudGlsZSBhc2tzIHRoZSB1c2VmdWwgcXVlc3Rpb24gaW5zdGVhZDogaXMgdGhlcmUgYW55dGhpbmcKICAgIGJyaWdodCBpbiB0aGlzIHJpbmcuIEl0IHdvcmtzIGZvciBhIHNtb290aCB0cmFjZXIgdG9vLCB3aGVyZSBpdCBzaW1wbHkgdHJhY2tzCiAgICB0aGUgbWVkaWFuIHdpdGggYW4gb2Zmc2V0LgoKICAgIFdIWSBUSEUgRkFSIEZJRUxEIElTIFNVQlRSQUNURUQKICAgIEEgaGlnaCBwZXJjZW50aWxlIG9mIHB1cmUgbm9pc2UgaXMgbm90IHplcm8gLSBpdCBpcyBhYm91dCAxLjMgdGltZXMgdGhlIG5vaXNlLAogICAgYmVjYXVzZSBpdCBpcyBwaWNraW5nIHRoZSBicmlnaHRlc3QgdGVudGggb2YgdGhlIHNhbXBsZS4gQ29tcGFyaW5nIGFnYWluc3QKICAgIHplcm8gd291bGQgY2FsbCBlbXB0eSBza3kgInN0aWxsIGdhbGF4eSIgZm9yIGV2ZXIuIFNvIHRoZSBzYW1lIHN0YXRpc3RpYyBpcwogICAgbWVhc3VyZWQgZmFyIG91dCwgd2hlcmUgdGhlcmUgaXMgbm90aGluZywgYW5kIHRoYXQgZmxvb3IgaXMgdGFrZW4gb2ZmLgoKICAgIFNhbXBsZWQgZXZlcnkgZmV3IHBpeGVscyByYXRoZXIgdGhhbiBldmVyeSBvbmU6IHRoaXMgaXMgYSBzaGFwZSBtZWFzdXJlbWVudAogICAgb3ZlciB0aG91c2FuZHMgb2YgcGl4ZWxzIGEgcmluZywgYW5kIHJlYWRpbmcgYSBxdWFydGVyIG9mIHRoZW0gY2hhbmdlcyBub3RoaW5nCiAgICB3aGlsZSBtYWtpbmcgaXQgcXVpY2sgZW5vdWdoIGZvciBhIDYxLW1lZ2FwaXhlbCBmcmFtZS4KICAgICIiIgogICAgZnJvbSBhc3Ryb3B5LnN0YXRzIGltcG9ydCBzaWdtYV9jbGlwcGVkX3N0YXRzCgogICAgbnksIG54ID0gZGF0YS5zaGFwZQogICAgeXksIHh4ID0gbnAubWdyaWRbMDpueTpzdGVwLCAwOm54OnN0ZXBdCiAgICByID0gbnAuaHlwb3QoeHggLSBjZW50cmVbMF0sIHl5IC0gY2VudHJlWzFdKS5yYXZlbCgpCiAgICB2ID0gbnAuYXNhcnJheShkYXRhWzo6c3RlcCwgOjpzdGVwXSwgZHR5cGU9ZmxvYXQpLnJhdmVsKCkKICAgIG9rID0gbnAuaXNmaW5pdGUodikKICAgIHIsIHYgPSByW29rXSwgdltva10KICAgIGlmIG5vdCBsZW4ocik6CiAgICAgICAgcmV0dXJuIE5vbmUKCiAgICBpZiByX21heCBpcyBOb25lOgogICAgICAgIHJfbWF4ID0gZmxvYXQobnAucGVyY2VudGlsZShyLCA5OSkpCiAgICBmYXIgPSByID4gMC44ICogci5tYXgoKQogICAgaWYgZmFyLnN1bSgpIDwgNTAwOgogICAgICAgIGZhciA9IHIgPiBucC5wZXJjZW50aWxlKHIsIDgwKQogICAgc2t5ID0gZmxvYXQobnAubWVkaWFuKHZbZmFyXSkpCiAgICBub2lzZSA9IGZsb2F0KHNpZ21hX2NsaXBwZWRfc3RhdHModltmYXJdLCBzaWdtYT0zLjApWzJdKQogICAgaWYgbm90IG5wLmlzZmluaXRlKG5vaXNlKSBvciBub2lzZSA8PSAwOgogICAgICAgIHJldHVybiBOb25lCgogICAgIyBXaGF0IHRoaXMgc3RhdGlzdGljIHJlYWRzIG9uIGVtcHR5IHNreS4gRXZlcnl0aGluZyBpcyBtZWFzdXJlZCBhZ2FpbnN0IGl0LgogICAgZmxvb3IgPSBmbG9hdChucC5wZXJjZW50aWxlKHZbZmFyXSwgcGVyY2VudGlsZSkpCgogICAgZWRnZXMgPSBucC5saW5zcGFjZSgwLjAsIHJfbWF4LCBuX3JpbmdzICsgMSkKICAgIGNlbnRyZXMsIGV4Y2VzcyA9IFtdLCBbXQogICAgZm9yIGEsIGIgaW4gemlwKGVkZ2VzWzotMV0sIGVkZ2VzWzE6XSk6CiAgICAgICAgbSA9IChyID49IGEpICYgKHIgPCBiKQogICAgICAgIGlmIG0uc3VtKCkgPCA1MDoKICAgICAgICAgICAgY29udGludWUKICAgICAgICBjZW50cmVzLmFwcGVuZCgwLjUgKiAoYSArIGIpKQogICAgICAgIGV4Y2Vzcy5hcHBlbmQoKGZsb2F0KG5wLnBlcmNlbnRpbGUodlttXSwgcGVyY2VudGlsZSkpIC0gZmxvb3IpIC8gbm9pc2UpCiAgICByZXR1cm4gbnAuYXJyYXkoY2VudHJlcyksIG5wLmFycmF5KGV4Y2VzcyksIHNreSwgbm9pc2UKCmRlZiB2aXNpYmxlX3JhZGl1cyhkYXRhLCBjZW50cmUsIHRocmVzaG9sZD0wLjUsIHJfbWF4PU5vbmUsIHBlcmNlbnRpbGU9OTAuMCk6CiAgICAiIiJXaGVyZSB0aGUgZ2FsYXh5J3MgbGlnaHQgZmFsbHMgYmVsb3cgYSBmcmFjdGlvbiBvZiB0aGUgc2t5IG5vaXNlLgoKICAgIGB0aHJlc2hvbGRgIGlzIGluIHVuaXRzIG9mIHRoYXQgbm9pc2UsIGFuZCBpdCBpcyB0aGUgb25lIHJlYWwgY2hvaWNlIGhlcmUuCiAgICBIYWxmIGEgc2lnbWEgaXMgdGhlIGRlZmF1bHQgYmVjYXVzZSB0aGUgdHdvIGVycm9ycyBhcmUgbm90IHN5bW1ldHJpYzogdG9vCiAgICBsYXJnZSBhIHJhZGl1cyBsZXRzIGVtcHR5IHNreSBpbnRvIHRoZSBtZWFzdXJlbWVudCBhbmQgZGlsdXRlcyBpdCwgd2hpY2ggaXMKICAgIHdoYXQgd2FzIGhhcHBlbmluZywgd2hpbGUgdG9vIHNtYWxsIGEgcmFkaXVzIGRlbGV0ZXMgdGhlIG91dGVyIHJlZ2lvbnMgLSBhbmQKICAgIHRoZSBvdXRza2lydHMgYXJlIHdoZXJlIGEgcmFkaWFsIHByb2ZpbGUgaXMgbW9zdCBzZW5zaXRpdmUsIHNpbmNlIHRoZSBkZW5zaXR5CiAgICBpcyBsb3dlc3QgdGhlcmUuIEhhbGYgYSBzaWdtYSBzaXRzIHdoZXJlIHRoZSBwcm9maWxlIGhhcyBjbGVhcmx5IGZsYXR0ZW5lZCBidXQKICAgIGhhcyBub3QgeWV0IHJlYWNoZWQgemVyby4KCiAgICBSZXR1cm5zIChyYWRpdXMgaW4gcGl4ZWxzLCBkaWFnbm9zdGljcyksIG9yIChOb25lLCBkaWFnbm9zdGljcykgd2hlbiB0aGUKICAgIHByb2ZpbGUgbmV2ZXIgcmlzZXMgYWJvdmUgdGhlIHRocmVzaG9sZCBhdCBhbGwgLSBhIGZyYW1lIHdpdGggbm8gZ2FsYXh5IGluIGl0LAogICAgb3IgYSBjZW50cmUgaW4gdGhlIHdyb25nIHBsYWNlLgogICAgIiIiCiAgICBnb3QgPSBzdXJmYWNlX2JyaWdodG5lc3NfcHJvZmlsZShkYXRhLCBjZW50cmUsIHJfbWF4PXJfbWF4LAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcGVyY2VudGlsZT1wZXJjZW50aWxlKQogICAgaWYgZ290IGlzIE5vbmU6CiAgICAgICAgcmV0dXJuIE5vbmUsIHsid2h5IjogImNvdWxkIG5vdCBtZWFzdXJlIGEgYmFja2dyb3VuZCB0byBjb21wYXJlIGFnYWluc3QifQogICAgcmFkaWksIGV4Y2Vzcywgc2t5LCBub2lzZSA9IGdvdAoKICAgIGFib3ZlID0gZXhjZXNzID4gdGhyZXNob2xkCiAgICBpZiBub3QgYWJvdmUuYW55KCk6CiAgICAgICAgcmV0dXJuIE5vbmUsIHsid2h5IjogIm5vdGhpbmcgcmlzZXMgYWJvdmUgezouMWZ9IHNpZ21hIG9mIHRoZSBza3kgbm9pc2UgIgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICJhbnl3aGVyZSIuZm9ybWF0KHRocmVzaG9sZCksCiAgICAgICAgICAgICAgICAgICAgICAic2t5Ijogc2t5LCAibm9pc2UiOiBub2lzZX0KCiAgICAjIFRoZSBsYXN0IHJpbmcgYWJvdmUgdGhlIHRocmVzaG9sZCwgd2Fsa2luZyBvdXR3YXJkcyAtIG5vdCB0aGUgZmlyc3Qgb25lCiAgICAjIGJlbG93IGl0LCBzbyB0aGF0IGEgc2luZ2xlIG5vaXN5IHJpbmcgaW4gdGhlIG1pZGRsZSBkb2VzIG5vdCBjdXQgdGhlIGdhbGF4eQogICAgIyBpbiBoYWxmLgogICAgbGFzdCA9IGludChucC53aGVyZShhYm92ZSlbMF0ubWF4KCkpCiAgICByYWRpdXMgPSBmbG9hdChyYWRpaVtsYXN0XSkKICAgIHJldHVybiByYWRpdXMsIHsic2t5Ijogc2t5LCAibm9pc2UiOiBub2lzZSwgInRocmVzaG9sZCI6IHRocmVzaG9sZCwKICAgICAgICAgICAgICAgICAgICAiY2VudHJhbF9leGNlc3MiOiBmbG9hdChleGNlc3NbMF0pIGlmIGxlbihleGNlc3MpIGVsc2UgZmxvYXQoIm5hbiIpLAogICAgICAgICAgICAgICAgICAgICJyYWRpaSI6IHJhZGlpLCAiZXhjZXNzIjogZXhjZXNzfQoKZGVmIGRlc2NyaWJlKHJhZGl1cywgZGlhZywgcjI1X3B4PU5vbmUsIGxhYmVsPSIiKToKICAgICIiIldoYXQgd2FzIG1lYXN1cmVkLCBuZXh0IHRvIHdoYXQgdGhlIGNhdGFsb2d1ZSBzYXlzLiIiIgogICAgaWYgcmFkaXVzIGlzIE5vbmU6CiAgICAgICAgcmV0dXJuICIgIHt9OiBjb3VsZCBub3QgbWVhc3VyZSBhbiBleHRlbnQgKHt9KSIuZm9ybWF0KAogICAgICAgICAgICBsYWJlbCwgZGlhZy5nZXQoIndoeSIsICJ1bmtub3duIikpCiAgICBsaW5lID0gKCIgIHt9OiB0aGUgbGlnaHQgZmFsbHMgYmVsb3cgezouMWZ9IHggdGhlIHNreSBub2lzZSBhdCB7Oi4wZn0gcHgiCiAgICAgICAgICAgIC5mb3JtYXQobGFiZWwsIGRpYWcuZ2V0KCJ0aHJlc2hvbGQiLCAwLjUpLCByYWRpdXMpKQogICAgaWYgcjI1X3B4OgogICAgICAgIGxpbmUgKz0gIjsgdGhlIGNhdGFsb2d1ZSdzIFIyNSBpcyB7Oi4wZn0gcHgiLmZvcm1hdChyMjVfcHgpCiAgICAgICAgaWYgcmFkaXVzIDwgMC44ICogcjI1X3B4OgogICAgICAgICAgICBsaW5lICs9ICgiXG4gICAgdGhlIGV4cG9zdXJlIGRvZXMgbm90IHJlYWNoIHRoZSBjYXRhbG9ndWUgcmFkaXVzLCBzbyAiCiAgICAgICAgICAgICAgICAgICAgICJ0aGUgbWVhc3VyZWQgb25lIGlzIHVzZWQiKQogICAgICAgIGVsaWYgcmFkaXVzID4gMS4yICogcjI1X3B4OgogICAgICAgICAgICBsaW5lICs9ICgiXG4gICAgdGhlIGxpZ2h0IHJlYWNoZXMgcGFzdCB0aGUgY2F0YWxvZ3VlIHJhZGl1czsgdGhlICIKICAgICAgICAgICAgICAgICAgICAgImNhdGFsb2d1ZSBvbmUgaXMgdXNlZCwgYmVpbmcgdGhlIG1vcmUgc3RhbmRhcmQiKQogICAgcmV0dXJuIGxpbmUKCmRlZiB3b3JraW5nX3JhZGl1cyhtZWFzdXJlZCwgcjI1X3B4KToKICAgICIiIlRoZSByYWRpdXMgdG8gYWN0dWFsbHkgdXNlOiB0aGUgc21hbGxlciBvZiB3aGF0IGlzIHRoZXJlIGFuZCB3aGF0IGlzIHB1Ymxpc2hlZC4KCiAgICBTbWFsbGVyIG9mIHRoZSB0d28gb24gcHVycG9zZS4gSWYgdGhlIGV4cG9zdXJlIGRvZXMgbm90IHJlYWNoIFIyNSB0aGVuIHRoZQogICAgc3BhY2UgYmV0d2VlbiBpcyBlbXB0eSBza3kgYW5kIGluY2x1ZGluZyBpdCBvbmx5IGFkZHMgbm9pc2UuIElmIHRoZSBsaWdodAogICAgcmVhY2hlcyBwYXN0IFIyNSAtIHdoaWNoIGhhcHBlbnMgZm9yIGEgYnJpZ2h0IGlubmVyIGRpc2sgbWVhc3VyZWQgZ2VuZXJvdXNseSAtCiAgICB0aGVuIFIyNSBpcyB0aGUgc3RhbmRhcmQgZGVmaW5pdGlvbiBhbmQgdGhlcmUgaXMgbm8gcmVhc29uIHRvIGludmVudCBhIHdpZGVyCiAgICBvbmUuCiAgICAiIiIKICAgIGlmIG1lYXN1cmVkIGFuZCByMjVfcHg6CiAgICAgICAgcmV0dXJuIG1pbihtZWFzdXJlZCwgcjI1X3B4KQogICAgcmV0dXJuIG1lYXN1cmVkIG9yIHIyNV9weAoKZGVzY3JpYmVfX2dhbGF4eV9leHRlbnQgPSBkZXNjcmliZSAgICMgdGhpcyBtb2R1bGUncyBvd247IHRoZSBwbGFpbiBuYW1lIGdldHMgc2hhZG93ZWQgYmVsb3cKCgojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCiMgc2hhcmVkIG1vZHVsZSBkZXByb2plY3Rpb24ucHkKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQoKSU5DX01JTl9ERUcgPSAzMC4wCgpJTkNfTUFYX0RFRyA9IDcwLjAKClEwX0RFRkFVTFQgPSAwLjEzCgpRMF9CWV9UWVBFID0geyJlYXJseSI6IDAuMjAsICJtaWQiOiAwLjEzLCAibGF0ZSI6IDAuMTB9CgpkZWYgcTBfZm9yX3R5cGUobXR5cGU6IHN0ciB8IE5vbmUpIC0+IGZsb2F0OgogICAgIiIiUGljayBhIGRpc2stdGhpY2tuZXNzIGNvbnN0YW50IGZyb20gdGhlIG1vcnBob2xvZ2ljYWwgdHlwZSBzdHJpbmcuCgogICAgRmFsbHMgYmFjayB0byB0aGUgbWlkLXR5cGUgdmFsdWUgd2hlbiB0aGUgdHlwZSBpcyB1bmtub3duIG9yIHVucGFyc2FibGUsCiAgICB3aGljaCBpcyB0aGUgc2FmZSBjaG9pY2U6IGl0IHNpdHMgYmV0d2VlbiB0aGUgdHdvIGV4dHJlbWVzLgogICAgIiIiCiAgICBpZiBub3QgbXR5cGU6CiAgICAgICAgcmV0dXJuIFEwX0RFRkFVTFQKICAgIHQgPSBzdHIobXR5cGUpLnN0cmlwKCkudXBwZXIoKQogICAgaWYgdC5zdGFydHN3aXRoKCgiUzAiLCAiU0EiLCAiRSIpKToKICAgICAgICByZXR1cm4gUTBfQllfVFlQRVsiZWFybHkiXQogICAgaWYgdC5zdGFydHN3aXRoKCgiU0QiLCAiU00iLCAiSU0iLCAiSSIpKToKICAgICAgICByZXR1cm4gUTBfQllfVFlQRVsibGF0ZSJdCiAgICByZXR1cm4gUTBfQllfVFlQRVsibWlkIl0KCmRlZiBpbmNsaW5hdGlvbl9mcm9tX2F4aXNfcmF0aW8oYl9vdmVyX2E6IGZsb2F0LCBxMDogZmxvYXQgPSBRMF9ERUZBVUxUKSAtPiBmbG9hdDoKICAgICIiIkluY2xpbmF0aW9uIGluIGRlZ3JlZXMgZnJvbSB0aGUgb2JzZXJ2ZWQgbWlub3IvbWFqb3IgYXhpcyByYXRpby4KCiAgICBVc2VzIHRoZSBIdWJibGUgZm9ybXVsYSwgd2hpY2ggYWNjb3VudHMgZm9yIHRoZSBkaXNrIGhhdmluZyByZWFsIHRoaWNrbmVzczoKCiAgICAgICAgY29zXjIoaSkgPSAocV4yIC0gcTBeMikgLyAoMSAtIHEwXjIpCgogICAgQXQgbW9kZXJhdGUgaW5jbGluYXRpb25zIHRoaXMgYmFyZWx5IGRpZmZlcnMgZnJvbSB0aGUgbmFpdmUgYWNvcyhiL2EpLiBBdCBoaWdoCiAgICBpbmNsaW5hdGlvbiBpdCBtYXR0ZXJzIGEgbG90OiBmb3IgcSA9IDAuMiB0aGUgbmFpdmUgdmFsdWUgaXMgNzguNSBkZWcgd2hpbGUgdGhpcwogICAgZ2l2ZXMgODEuMiBkZWcsIHdoaWNoIGNoYW5nZXMgdGhlIHN0cmV0Y2ggZmFjdG9yIGZyb20gNS4wIHRvIDYuNS4KCiAgICBSZXR1cm5zIDkwLjAgd2hlbiB0aGUgZ2FsYXh5IGlzIHRoaW5uZXIgdGhhbiBxMCwgaS5lLiBzZWVuIGVkZ2Utb24uCiAgICAiIiIKICAgIHEgPSBmbG9hdChiX292ZXJfYSkKICAgIGlmIHEgPD0gcTA6CiAgICAgICAgcmV0dXJuIDkwLjAKICAgIGNvczIgPSAocSAqIHEgLSBxMCAqIHEwKSAvICgxLjAgLSBxMCAqIHEwKQogICAgY29zMiA9IG1pbihtYXgoY29zMiwgMC4wKSwgMS4wKQogICAgcmV0dXJuIG1hdGguZGVncmVlcyhtYXRoLmFjb3MobWF0aC5zcXJ0KGNvczIpKSkKCmRlZiBzdHJldGNoX2ZhY3RvcihpbmNfZGVnOiBmbG9hdCkgLT4gZmxvYXQ6CiAgICAiIiJIb3cgbXVjaCB0aGUgbWlub3ItYXhpcyBkaXJlY3Rpb24gaGFzIHRvIGJlIHN0cmV0Y2hlZDogMSAvIGNvcyhpKS4iIiIKICAgIHJldHVybiAxLjAgLyBtYXRoLmNvcyhtYXRoLnJhZGlhbnMoaW5jX2RlZykpCgpkZWYgaW5jbGluYXRpb25faXNfdXNhYmxlKGluY19kZWc6IGZsb2F0KSAtPiB0dXBsZVtib29sLCBzdHJdOgogICAgIiIiRGVjaWRlIHdoZXRoZXIgZGVwcm9qZWN0aW5nIGF0IHRoaXMgaW5jbGluYXRpb24gaXMgd29ydGggZG9pbmcuCgogICAgUmV0dXJucyAob2ssIHJlYXNvbikuIFRoZSByZWFzb24gaXMgbWVhbnQgdG8gYmUgcHJpbnRlZCBhbmQgdG8gZW5kIHVwIGluIHRoZQogICAgcGxvdCwgc28gdGhhdCBhIHNraXBwZWQgY29ycmVjdGlvbiBpcyB2aXNpYmxlIHJhdGhlciB0aGFuIHNpbGVudC4KICAgICIiIgogICAgaWYgaW5jX2RlZyA8IElOQ19NSU5fREVHOgogICAgICAgIHJldHVybiBGYWxzZSwgKAogICAgICAgICAgICBmImluY2xpbmF0aW9uIHtpbmNfZGVnOi4xZn0gZGVnIGlzIGJlbG93IHtJTkNfTUlOX0RFRzouMGZ9IGRlZyAtIHRoZSBnYWxheHkgaXMgIgogICAgICAgICAgICAiY2xvc2UgdG8gZmFjZS1vbiwgdGhlIGNvcnJlY3Rpb24gd291bGQgYmUgc21hbGxlciB0aGFuIGl0cyBvd24gdW5jZXJ0YWludHksICIKICAgICAgICAgICAgImFuZCB0aGUgcG9zaXRpb24gYW5nbGUgaXMgcG9vcmx5IGRlZmluZWQuIE5vdCBjb3JyZWN0ZWQuIgogICAgICAgICkKICAgIGlmIGluY19kZWcgPiBJTkNfTUFYX0RFRzoKICAgICAgICByZXR1cm4gRmFsc2UsICgKICAgICAgICAgICAgZiJpbmNsaW5hdGlvbiB7aW5jX2RlZzouMWZ9IGRlZyBpcyBhYm92ZSB7SU5DX01BWF9ERUc6LjBmfSBkZWcgLSB0aGUgY29ycmVjdGlvbiAiCiAgICAgICAgICAgICJ3b3VsZCBiZSBsYXJnZSBidXQgdW5yZWxpYWJsZSwgYW5kIGR1c3QgYmlhc2VzIGl0IHN5c3RlbWF0aWNhbGx5LiBOb3QgY29ycmVjdGVkLiIKICAgICAgICApCiAgICByZXR1cm4gVHJ1ZSwgZiJpbmNsaW5hdGlvbiB7aW5jX2RlZzouMWZ9IGRlZyBpcyBpbnNpZGUgdGhlIHVzYWJsZSB3aW5kb3cuIgoKZGVmIHNreV9wYV90b19pbWFnZV9hbmdsZShwYV9za3lfZGVnOiBmbG9hdCwKICAgICAgICAgICAgICAgICAgICAgICAgICBub3J0aF9hbmdsZV9kZWc6IGZsb2F0ID0gOTAuMCwKICAgICAgICAgICAgICAgICAgICAgICAgICBtaXJyb3JlZDogYm9vbCA9IEZhbHNlKSAtPiBmbG9hdDoKICAgICIiIkNvbnZlcnQgYSBjYXRhbG9ndWUgcG9zaXRpb24gYW5nbGUgaW50byBhbiBhbmdsZSBpbiBpbWFnZSBwaXhlbCBjb29yZGluYXRlcy4KCiAgICBDYXRhbG9ndWVzIGdpdmUgdGhlIHBvc2l0aW9uIGFuZ2xlIG1lYXN1cmVkIG9uIHRoZSBza3ksIGZyb20gTm9ydGggdG93YXJkcyBFYXN0LgogICAgT3VyIG9iamVjdCBwb3NpdGlvbnMgYXJlIGluIGltYWdlIHBpeGVscywgd2hlcmUgdGhlIG1ham9yIGF4aXMgaGFzIHNvbWUgYW5nbGUKICAgIG1lYXN1cmVkIGZyb20gdGhlICt4IGF4aXMuIFRoZXNlIGFyZSBub3QgdGhlIHNhbWUgbnVtYmVyIGFuZCBjb25mdXNpbmcgdGhlbQogICAgcm90YXRlcyB0aGUgd2hvbGUgZGVwcm9qZWN0aW9uLgoKICAgICAgICBub3J0aF9hbmdsZV9kZWcgIGFuZ2xlIG9mIE5vcnRoIGluIHRoZSBpbWFnZSwgY291bnRlcmNsb2Nrd2lzZSBmcm9tICt4LgogICAgICAgICAgICAgICAgICAgICAgICAgOTAgZm9yIHRoZSB1c3VhbCAibm9ydGggdXAiIG9yaWVudGF0aW9uLgogICAgICAgIG1pcnJvcmVkICAgICAgICAgVHJ1ZSBpZiB0aGUgaW1hZ2UgcGFyaXR5IGlzIGZsaXBwZWQsIHNvIHRoYXQgRWFzdCBydW5zCiAgICAgICAgICAgICAgICAgICAgICAgICBjbG9ja3dpc2UgZnJvbSBOb3J0aCBpbnN0ZWFkIG9mIGNvdW50ZXJjbG9ja3dpc2UuCgogICAgSWYgeW91IGFscmVhZHkga25vdyB0aGUgbWFqb3ItYXhpcyBhbmdsZSBkaXJlY3RseSBpbiBpbWFnZSBjb29yZGluYXRlcywgc2tpcAogICAgdGhpcyBmdW5jdGlvbiBhbmQgcGFzcyB0aGF0IGFuZ2xlIHN0cmFpZ2h0IHRvIGRlcHJvamVjdCgpLgogICAgIiIiCiAgICBzaWduID0gLTEuMCBpZiBtaXJyb3JlZCBlbHNlIDEuMAogICAgcmV0dXJuIChub3J0aF9hbmdsZV9kZWcgKyBzaWduICogcGFfc2t5X2RlZykgJSAxODAuMAoKZGVmIGRlcHJvamVjdChkeCwgZHksIG1ham9yX2F4aXNfYW5nbGVfZGVnOiBmbG9hdCwgaW5jX2RlZzogZmxvYXQpOgogICAgIiIiU3RyZXRjaCBjZW50cmVkIHBvc2l0aW9ucyBmcm9tIHRoZSBwbGFuZSBvZiB0aGUgc2t5IGludG8gdGhlIGdhbGF4eSBwbGFuZS4KCiAgICBkeCwgZHkgbXVzdCBhbHJlYWR5IGJlIG1lYXN1cmVkIEZST00gVEhFIEdBTEFYWSBDRU5UUkUuIFRoaXMgbWF0dGVyczogdGhlCiAgICB0cmFuc2Zvcm0gaXMgbGluZWFyLCBzbyBhcHBseWluZyBpdCB0byB1bmNlbnRyZWQgY29vcmRpbmF0ZXMgbGVhdmVzIHRoZSBzaGFwZQogICAgY29ycmVjdCBidXQgbW92ZXMgdGhlIGNlbnRyZSwgYW5kIGV2ZXJ5IHJhZGl1cyBtZWFzdXJlZCBmcm9tIHRoZSBvbGQgY2VudHJlCiAgICB0aGVuIGNvbWVzIG91dCB3cm9uZy4KCiAgICBtYWpvcl9heGlzX2FuZ2xlX2RlZyBpcyBtZWFzdXJlZCBpbiB0aGUgc2FtZSBmcmFtZSBhcyBkeCwgZHkgLSBjb3VudGVyY2xvY2t3aXNlCiAgICBmcm9tIHRoZSAreCBheGlzLiBVc2Ugc2t5X3BhX3RvX2ltYWdlX2FuZ2xlKCkgaWYgeW91IGFyZSBzdGFydGluZyBmcm9tIGEKICAgIGNhdGFsb2d1ZSBwb3NpdGlvbiBhbmdsZS4KCiAgICBUaGUgdHJhbnNmb3JtIHJvdGF0ZXMgdGhlIG1ham9yIGF4aXMgb250byB4LCBzdHJldGNoZXMgdGhlIHBlcnBlbmRpY3VsYXIKICAgIGRpcmVjdGlvbiBieSAxL2NvcyhpKSwgYW5kIHJvdGF0ZXMgYmFjay4gVGhlIHJlc3VsdGluZyBtYXRyaXggaXMgc3ltbWV0cmljLAogICAgYXMgYSBwdXJlIHN0cmV0Y2ggYWxvbmcgYW4gYXhpcyBtdXN0IGJlLgogICAgIiIiCiAgICBkeCA9IG5wLmFzYXJyYXkoZHgsIGR0eXBlPWZsb2F0KQogICAgZHkgPSBucC5hc2FycmF5KGR5LCBkdHlwZT1mbG9hdCkKCiAgICB0aCA9IG1hdGgucmFkaWFucyhtYWpvcl9heGlzX2FuZ2xlX2RlZykKICAgIGMsIHMgPSBtYXRoLmNvcyh0aCksIG1hdGguc2luKHRoKQogICAgayA9IHN0cmV0Y2hfZmFjdG9yKGluY19kZWcpCgogICAgbTExID0gYyAqIGMgKyBrICogcyAqIHMKICAgIG0xMiA9ICgxLjAgLSBrKSAqIHMgKiBjCiAgICBtMjIgPSBzICogcyArIGsgKiBjICogYwoKICAgIHJldHVybiBkeCAqIG0xMSArIGR5ICogbTEyLCBkeCAqIG0xMiArIGR5ICogbTIyCgpkZWYgZWZmZWN0aXZlX3JpbmdfYXJlYXMoZWRnZXMsCiAgICAgICAgICAgICAgICAgICAgICAgICBmcmFtZV9jb3JuZXJzLAogICAgICAgICAgICAgICAgICAgICAgICAgY2VudHJlX3h5LAogICAgICAgICAgICAgICAgICAgICAgICAgbWFqb3JfYXhpc19hbmdsZV9kZWc6IGZsb2F0IHwgTm9uZSwKICAgICAgICAgICAgICAgICAgICAgICAgIGluY19kZWc6IGZsb2F0IHwgTm9uZSwKICAgICAgICAgICAgICAgICAgICAgICAgIG5fc2FtcGxlczogaW50ID0gNDAwXzAwMCwKICAgICAgICAgICAgICAgICAgICAgICAgIHJuZ19zZWVkOiBpbnQgPSAxMjM0NSk6CiAgICAiIiJBcmVhIG9mIGVhY2ggYW5udWx1cyB0aGF0IGFjdHVhbGx5IGZhbGxzIGluc2lkZSB0aGUgaW1hZ2VkIGZyYW1lLgoKICAgIFdIWSBUSElTIEVYSVNUUwogICAgQSByaW5nIGRyYXduIGFyb3VuZCB0aGUgZ2FsYXh5IGNlbnRyZSBpcyBvbmx5IHBhcnRseSBjb3ZlcmVkIGJ5IHRoZSBpbWFnZTogYXQKICAgIGxhcmdlIHJhZGlpIHRoZSBjb3JuZXJzIG9mIHRoZSBmcmFtZSBjdXQgaXQuIERpdmlkaW5nIGNvdW50cyBieSB0aGUgZnVsbAogICAgZ2VvbWV0cmljIHJpbmcgYXJlYSB0aGVyZWZvcmUgbWFrZXMgdGhlIG91dGVyIGJpbnMgbG9vayBlbXB0aWVyIHRoYW4gdGhleSBhcmUsCiAgICBhbmQgcHJvZHVjZXMgYSByYWRpYWwgZGVjbGluZSB0aGF0IGlzIHB1cmUgZ2VvbWV0cnkuCgogICAgVGhpcyBpcyBub3QgaHlwb3RoZXRpY2FsLiBJbiBNMzMgYSBjYXRhbG9ndWUgY292ZXJpbmcgb25seSB0aGUgY2VudHJhbCAyMyd4MjMnCiAgICBwcm9kdWNlZCAiYW4gYXJ0aWZpY2lhbGx5IGxhcmdlIGRlY3JlYXNlIGF0IHJhZGlpIGxhcmdlciB0aGFuIHRoZSBsYXJnZXN0IGNpcmNsZQogICAgaW5zY3JpYmVkIGluIHRoZSBhcmVhIiwgYW5kIHRoZSBwdWJsaXNoZWQgZXhwb25lbnRpYWwgc2xvcGUgd2FzIHdyb25nIGJ5IGEKICAgIGZhY3RvciBvZiB0aHJlZSB1bnRpbCBhIHNpemUgY3V0IGFuZCBhIHJhZGlhbCB0cnVuY2F0aW9uIHdlcmUgYXBwbGllZC4KCiAgICBNRVRIT0QKICAgIFNjYXR0ZXIgcmFuZG9tIHBvaW50cyB1bmlmb3JtbHkgb3ZlciB0aGUgZnJhbWUsIHB1dCB0aGVtIHRocm91Z2ggZXhhY3RseSB0aGUKICAgIHNhbWUgZGVwcm9qZWN0aW9uIGFzIHRoZSByZWFsIG9iamVjdHMsIGFuZCBoaXN0b2dyYW0gdGhlaXIgcmFkaWkuIFRoZSBmcmFjdGlvbgogICAgbGFuZGluZyBpbiBlYWNoIHJpbmcsIHRpbWVzIHRoZSBmcmFtZSBhcmVhLCBpcyB0aGF0IHJpbmcncyBlZmZlY3RpdmUgYXJlYS4KICAgIFRoaXMgaGFuZGxlcyBhbnkgZnJhbWUgc2hhcGUgYW5kIGFueSBkZXByb2plY3Rpb24gd2l0aG91dCBzcGVjaWFsLWNhc2luZy4KCiAgICBmcmFtZV9jb3JuZXJzIGlzICh4bWluLCB4bWF4LCB5bWluLCB5bWF4KSBpbiB0aGUgc2FtZSBwaXhlbCB1bml0cyBhcyB0aGUgZGF0YS4KICAgIFBhc3MgbWFqb3JfYXhpc19hbmdsZV9kZWc9Tm9uZSB0byBza2lwIGRlcHJvamVjdGlvbiAodGhlIHVuY29ycmVjdGVkIGNhc2UpLgogICAgIiIiCiAgICB4bWluLCB4bWF4LCB5bWluLCB5bWF4ID0gZnJhbWVfY29ybmVycwogICAgcm5nID0gbnAucmFuZG9tLmRlZmF1bHRfcm5nKHJuZ19zZWVkKQoKICAgIHN4ID0gcm5nLnVuaWZvcm0oeG1pbiwgeG1heCwgbl9zYW1wbGVzKQogICAgc3kgPSBybmcudW5pZm9ybSh5bWluLCB5bWF4LCBuX3NhbXBsZXMpCiAgICBmcmFtZV9hcmVhID0gKHhtYXggLSB4bWluKSAqICh5bWF4IC0geW1pbikKCiAgICBzZHggPSBzeCAtIGNlbnRyZV94eVswXQogICAgc2R5ID0gc3kgLSBjZW50cmVfeHlbMV0KICAgIGlmIG1ham9yX2F4aXNfYW5nbGVfZGVnIGlzIG5vdCBOb25lIGFuZCBpbmNfZGVnIGlzIG5vdCBOb25lOgogICAgICAgIHNkeCwgc2R5ID0gZGVwcm9qZWN0KHNkeCwgc2R5LCBtYWpvcl9heGlzX2FuZ2xlX2RlZywgaW5jX2RlZykKCiAgICBzciA9IG5wLmh5cG90KHNkeCwgc2R5KQogICAgY291bnRzLCBfID0gbnAuaGlzdG9ncmFtKHNyLCBiaW5zPWVkZ2VzKQogICAgcmV0dXJuIGNvdW50cyAvIGZsb2F0KG5fc2FtcGxlcykgKiBmcmFtZV9hcmVhCgpkZWYgcmFkaWFsX2RlbnNpdHkoZHgsIGR5LAogICAgICAgICAgICAgICAgICAgbl9yaW5nczogaW50ID0gMjAsCiAgICAgICAgICAgICAgICAgICByX21heDogZmxvYXQgfCBOb25lID0gTm9uZSwKICAgICAgICAgICAgICAgICAgIGZyYW1lX2Nvcm5lcnM9Tm9uZSwKICAgICAgICAgICAgICAgICAgIGNlbnRyZV94eT1Ob25lLAogICAgICAgICAgICAgICAgICAgbWFqb3JfYXhpc19hbmdsZV9kZWc6IGZsb2F0IHwgTm9uZSA9IE5vbmUsCiAgICAgICAgICAgICAgICAgICBpbmNfZGVnOiBmbG9hdCB8IE5vbmUgPSBOb25lLAogICAgICAgICAgICAgICAgICAgbWluX2FyZWFfZnJhY3Rpb246IGZsb2F0ID0gMC4xNSk6CiAgICAiIiJTdXJmYWNlIGRlbnNpdHkgb2Ygb2JqZWN0cyBpbiBlcXVhbC13aWR0aCBjb25jZW50cmljIHJpbmdzLgoKICAgIEVxdWFsLXdpZHRoIHJpbmdzIG91dCB0byB0aGUgb3V0ZXJtb3N0IG9iamVjdCwgZm9sbG93aW5nIHRoZSBzdGFuZGFyZCBwcmFjdGljZQogICAgaW4gdGhpcyBsaXRlcmF0dXJlICgyMCByaW5ncywgZWFjaCAwLjA1IG9mIHRoZSBtYXhpbXVtIHJhZGl1cykuCgogICAgUmluZ3Mgd2hvc2UgZWZmZWN0aXZlIGFyZWEgaXMgYSBzbWFsbCBmcmFjdGlvbiBvZiB0aGVpciBmdWxsIGdlb21ldHJpYyBhcmVhIGFyZQogICAgbWFya2VkIGluY29tcGxldGUgcmF0aGVyIHRoYW4gcGxvdHRlZCwgYmVjYXVzZSB0aGVpciBkZW5zaXR5IGlzIGRvbWluYXRlZCBieSBob3cKICAgIHRoZSBmcmFtZSBoYXBwZW5zIHRvIGN1dCB0aGVtLiBtaW5fYXJlYV9mcmFjdGlvbiBzZXRzIHRoYXQgdGhyZXNob2xkLgoKICAgIFJldHVybnMgYSBkaWN0IHdpdGggcmluZyBjZW50cmVzLCBjb3VudHMsIGFyZWFzLCBkZW5zaXRpZXMgYW5kIFBvaXNzb24gZXJyb3JzLgogICAgIiIiCiAgICBkeCA9IG5wLmFzYXJyYXkoZHgsIGR0eXBlPWZsb2F0KQogICAgZHkgPSBucC5hc2FycmF5KGR5LCBkdHlwZT1mbG9hdCkKICAgIHIgPSBucC5oeXBvdChkeCwgZHkpCgogICAgaWYgcl9tYXggaXMgTm9uZToKICAgICAgICByX21heCA9IGZsb2F0KHIubWF4KCkpIGlmIHIuc2l6ZSBlbHNlIDEuMAogICAgZWRnZXMgPSBucC5saW5zcGFjZSgwLjAsIHJfbWF4LCBuX3JpbmdzICsgMSkKICAgIGNlbnRyZXMgPSAwLjUgKiAoZWRnZXNbOi0xXSArIGVkZ2VzWzE6XSkKCiAgICBjb3VudHMsIF8gPSBucC5oaXN0b2dyYW0ociwgYmlucz1lZGdlcykKICAgIGdlb21ldHJpYyA9IG1hdGgucGkgKiAoZWRnZXNbMTpdICoqIDIgLSBlZGdlc1s6LTFdICoqIDIpCgogICAgaWYgZnJhbWVfY29ybmVycyBpcyBub3QgTm9uZSBhbmQgY2VudHJlX3h5IGlzIG5vdCBOb25lOgogICAgICAgIGFyZWEgPSBlZmZlY3RpdmVfcmluZ19hcmVhcyhlZGdlcywgZnJhbWVfY29ybmVycywgY2VudHJlX3h5LAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYWpvcl9heGlzX2FuZ2xlX2RlZywgaW5jX2RlZykKICAgIGVsc2U6CiAgICAgICAgYXJlYSA9IGdlb21ldHJpYwoKICAgIHdpdGggbnAuZXJyc3RhdGUoZGl2aWRlPSJpZ25vcmUiLCBpbnZhbGlkPSJpZ25vcmUiKToKICAgICAgICBkZW5zaXR5ID0gbnAud2hlcmUoYXJlYSA+IDAsIGNvdW50cyAvIGFyZWEsIG5wLm5hbikKICAgICAgICBlcnIgPSBucC53aGVyZShhcmVhID4gMCwgbnAuc3FydChucC5tYXhpbXVtKGNvdW50cywgMCkpIC8gYXJlYSwgbnAubmFuKQogICAgICAgIGZyYWMgPSBucC53aGVyZShnZW9tZXRyaWMgPiAwLCBhcmVhIC8gZ2VvbWV0cmljLCAwLjApCgogICAgY29tcGxldGUgPSBmcmFjID49IG1pbl9hcmVhX2ZyYWN0aW9uCgogICAgcmV0dXJuIHsKICAgICAgICAiZWRnZXMiOiBlZGdlcywKICAgICAgICAiciI6IGNlbnRyZXMsCiAgICAgICAgImNvdW50cyI6IGNvdW50cywKICAgICAgICAiYXJlYSI6IGFyZWEsCiAgICAgICAgImFyZWFfZnJhY3Rpb24iOiBmcmFjLAogICAgICAgICJkZW5zaXR5IjogZGVuc2l0eSwKICAgICAgICAiZGVuc2l0eV9lcnIiOiBlcnIsCiAgICAgICAgImNvbXBsZXRlIjogY29tcGxldGUsCiAgICAgICAgInJfbWF4Ijogcl9tYXgsCiAgICB9CgpkZWYgc2VjdG9yX2Rpc3BlcnNpb24oZHgsIGR5LCBtYWpvcl9heGlzX2FuZ2xlX2RlZywgaW5jX2RlZywgbl9zZWN0b3JzPTIwKToKICAgICIiIkhvdyB1bmV2ZW4gdGhlIG9iamVjdCBjb3VudHMgYXJlIG9uY2UgdGhlIGZpZWxkIGlzIGN1dCBpbnRvIGVxdWFsIHdlZGdlcy4KCiAgICBUSEUgSURFQQogICAgU2xpY2UgdGhlIGdhbGF4eSBpbnRvIGVxdWFsIHNlY3RvcnMsIGxpa2Ugc2xpY2VzIG9mIGEgcGllLiBJZiB0aGUgZGlzayByZWFsbHkgaXMKICAgIHJvdW5kIGFuZCB3ZSBhcmUgbG9va2luZyBzdHJhaWdodCBkb3duIG9uIGl0LCBldmVyeSBzbGljZSBob2xkcyByb3VnaGx5IHRoZSBzYW1lCiAgICBudW1iZXIgb2Ygb2JqZWN0cy4gU2VlbiBhdCBhbiBhbmdsZSBpdCBkb2VzIG5vdC4gU28gdGhlIHRpbHQgdGhhdCBldmVucyB0aGUKICAgIHNsaWNlcyBvdXQgaXMgdGhlIHRpbHQgdGhhdCB1bmRvZXMgdGhlIHByb2plY3Rpb24uCgogICAgT3Bwb3NpdGUgc2xpY2VzIGFyZSBhZGRlZCB0b2dldGhlciBiZWZvcmUgdGhlIGRpc3BlcnNpb24gaXMgbWVhc3VyZWQuIEEgZ2FsYXh5CiAgICB0aGF0IGlzIGxvcHNpZGVkIC0gYSBicmlnaHRlciBhcm0gb24gb25lIHNpZGUsIGEgY29tcGFuaW9uIHB1bGxpbmcgYXQgaXQgLSB3b3VsZAogICAgb3RoZXJ3aXNlIHB1c2ggdGhlIGFuc3dlciBhcm91bmQ7IGFkZGluZyBhIHNsaWNlIHRvIHRoZSBvbmUgZmFjaW5nIGl0IGNhbmNlbHMKICAgIHRoYXQga2luZCBvZiBhc3ltbWV0cnkuCgogICAgUmV0dXJucyBzdGQgLyBtZWFuIG9mIHRoZSBmb2xkZWQgY291bnRzLiBTbWFsbGVyIGlzIGJldHRlci4KICAgICIiIgogICAgZHhwLCBkeXAgPSBkZXByb2plY3QoZHgsIGR5LCBtYWpvcl9heGlzX2FuZ2xlX2RlZywgaW5jX2RlZykKICAgIHRoZXRhID0gbnAuYXJjdGFuMihkeXAsIGR4cCkKICAgIGlkeCA9IG5wLmZsb29yKCh0aGV0YSAlICgyLjAgKiBtYXRoLnBpKSkgLyAoMi4wICogbWF0aC5waSkgKiBuX3NlY3RvcnMpLmFzdHlwZShpbnQpCiAgICBpZHggPSBucC5jbGlwKGlkeCwgMCwgbl9zZWN0b3JzIC0gMSkKICAgIGNvdW50cyA9IG5wLmJpbmNvdW50KGlkeCwgbWlubGVuZ3RoPW5fc2VjdG9ycykuYXN0eXBlKGZsb2F0KQoKICAgIGhhbGYgPSBuX3NlY3RvcnMgLy8gMgogICAgZm9sZGVkID0gY291bnRzWzpoYWxmXSArIGNvdW50c1toYWxmOl0KICAgIG1lYW4gPSBmb2xkZWQubWVhbigpCiAgICBpZiBtZWFuIDw9IDA6CiAgICAgICAgcmV0dXJuIG5wLmluZgogICAgcmV0dXJuIGZsb2F0KGZvbGRlZC5zdGQoKSAvIG1lYW4pCgpkZWYgZml0X3NlY3Rvcl9tZXRob2QoZHgsIGR5LAogICAgICAgICAgICAgICAgICAgICAgbl9zZWN0b3JzPTIwLAogICAgICAgICAgICAgICAgICAgICAgYW5nbGVfc3RlcD01LjAsCiAgICAgICAgICAgICAgICAgICAgICBpbmNfc3RlcD01LjAsCiAgICAgICAgICAgICAgICAgICAgICBpbmNfcmFuZ2U9KDIwLjAsIDgwLjApLAogICAgICAgICAgICAgICAgICAgICAgcl9tYXg9Tm9uZSk6CiAgICAiIiJTZWFyY2ggZm9yIHRoZSBkZXByb2plY3Rpb24gYW5nbGVzIHRoYXQgbWFrZSB0aGUgb2JqZWN0IGNvdW50cyBtb3N0IGV2ZW4uCgogICAgU2NhbnMgYSBncmlkIG9mIHBvc2l0aW9uIGFuZ2xlIGFuZCBpbmNsaW5hdGlvbiBhbmQgcmV0dXJucyB0aGUgcGFpciB0aGF0CiAgICBtaW5pbWlzZXMgc2VjdG9yX2Rpc3BlcnNpb24oKS4gVGhpcyBpcyB0aGUgc2Vjb25kIG9mIHRoZSB0d28gbWV0aG9kcyBpbnRyb2R1Y2VkCiAgICBieSBHYXJjaWEtR29tZXogYW5kIEF0aGFuYXNzb3VsYSwgYW5kIGluIHRoZWlyIGNvbXBhcmlzb24gaXQgaXMgdGhlIG9uZSB0aGF0CiAgICBhZ3JlZXMgbW9zdCBjbG9zZWx5IHdpdGgga2luZW1hdGljIGRldGVybWluYXRpb25zIC0gMC45NiBpbiBwb3NpdGlvbiBhbmdsZSBhbmQKICAgIDAuOTAgaW4gY29zKGluY2xpbmF0aW9uKSwgYmV0dGVyIHRoYW4gdGhlaXIgRm91cmllciBtZXRob2QgYW5kIGJldHRlciB0aGFuCiAgICBwaG90b21ldHJ5LgoKICAgIFdIRU4gSVQgRkFJTFMgLSByZWFkIHRoaXMgYmVmb3JlIHRydXN0aW5nIHRoZSBhbnN3ZXIKICAgICAgKiBCZWxvdyByb3VnaGx5IDE1MC0yMDAgb2JqZWN0cyB0aGUgc3VyZmFjZSBiZWluZyBtaW5pbWlzZWQgb2Z0ZW4gaGFzIHNldmVyYWwKICAgICAgICBtaW5pbWEgcmF0aGVyIHRoYW4gb25lLCBhbmQgdGhlIGRlZXBlc3QgaXMgbm90IGFsd2F5cyB0aGUgcmlnaHQgb25lLiBUaGUKICAgICAgICByZXR1cm5lZCBkaWN0IHJlcG9ydHMgdGhlIHJ1bm5lcnMtdXAgc28geW91IGNhbiBsb29rLgogICAgICAqIElmIHRoZSBvYmplY3RzIHRyYWNlIHRoZSBzcGlyYWwgYXJtcyBvciBhIGJhciBpbnN0ZWFkIG9mIGZpbGxpbmcgdGhlIGRpc2ssCiAgICAgICAgdGhlIG1ldGhvZCBjaXJjdWxhcmlzZXMgdGhlIGFybXMgb3IgdGhlIGJhciByYXRoZXIgdGhhbiB0aGUgZGlzay4gSW4gdGhlCiAgICAgICAgcHVibGlzaGVkIG5vdGVzIHRoaXMgaXMgZXhhY3RseSB3aGF0IHdlbnQgd3JvbmcgZm9yIHNldmVyYWwgYmFycmVkIGdhbGF4aWVzLgogICAgICAqIEEgcmluZyBvZiBvYmplY3RzIGlzIGNpcmN1bGFyaXNlZCBhcyBpZiBpdCB3ZXJlIHRoZSBkaXNrLiBBbiBvdXRlciByaW5nIHdpdGgKICAgICAgICBhbiBpbnRyaW5zaWMgYXhpcyByYXRpbyBvZiAwLjg3IGZha2VzIGFuIGluY2xpbmF0aW9uIG9mIGFib3V0IDEyIGRlZ3JlZXMgaW4gYQogICAgICAgIGdhbGF4eSB0aGF0IGlzIHJlYWxseSBmYWNlLW9uLgogICAgICAqIEl0IHdvcmtzIGJldHRlciB0aGUgbW9yZSBpbmNsaW5lZCB0aGUgZ2FsYXh5IGlzLCBhbmQgcG9vcmx5IG5lYXIgZmFjZS1vbiwKICAgICAgICB3aGVyZSB0aGUgcG9zaXRpb24gYW5nbGUgaXMgYmFyZWx5IGRlZmluZWQgaW4gdGhlIGZpcnN0IHBsYWNlLgoKICAgIHJfbWF4IG9wdGlvbmFsbHkgcmVzdHJpY3RzIHRoZSBmaXQgdG8gb2JqZWN0cyBpbnNpZGUgdGhhdCByYWRpdXMsIHdoaWNoIGhlbHBzCiAgICB3aGVuIGEgZmV3IGZhci1mbHVuZyBvYmplY3RzIGRvbWluYXRlIHRoZSBvdXRlciBzZWN0b3JzLgogICAgIiIiCiAgICBkeCA9IG5wLmFzYXJyYXkoZHgsIGR0eXBlPWZsb2F0KQogICAgZHkgPSBucC5hc2FycmF5KGR5LCBkdHlwZT1mbG9hdCkKCiAgICBpZiByX21heCBpcyBub3QgTm9uZToKICAgICAgICBrZWVwID0gbnAuaHlwb3QoZHgsIGR5KSA8PSByX21heAogICAgICAgIGR4LCBkeSA9IGR4W2tlZXBdLCBkeVtrZWVwXQoKICAgIGFuZ2xlcyA9IG5wLmFyYW5nZSgwLjAsIDE4MC4wLCBhbmdsZV9zdGVwKQogICAgaW5jcyA9IG5wLmFyYW5nZShpbmNfcmFuZ2VbMF0sIGluY19yYW5nZVsxXSArIDFlLTksIGluY19zdGVwKQoKICAgIGdyaWQgPSBucC5mdWxsKChsZW4oaW5jcyksIGxlbihhbmdsZXMpKSwgbnAuaW5mKQogICAgZm9yIGksIGluYyBpbiBlbnVtZXJhdGUoaW5jcyk6CiAgICAgICAgZm9yIGosIGFuZyBpbiBlbnVtZXJhdGUoYW5nbGVzKToKICAgICAgICAgICAgZ3JpZFtpLCBqXSA9IHNlY3Rvcl9kaXNwZXJzaW9uKGR4LCBkeSwgZmxvYXQoYW5nKSwgZmxvYXQoaW5jKSwgbl9zZWN0b3JzKQoKICAgIGZsYXQgPSBucC5hcmdzb3J0KGdyaWQsIGF4aXM9Tm9uZSkKICAgIGJlc3RfaSwgYmVzdF9qID0gbnAudW5yYXZlbF9pbmRleChmbGF0WzBdLCBncmlkLnNoYXBlKQoKICAgICMgUmVwb3J0IGRpc3RpbmN0IHJ1bm5lcnMtdXA6IGxvY2FsIG1pbmltYSB3ZWxsIHNlcGFyYXRlZCBmcm9tIHRoZSBiZXN0IG9uZSwgc28KICAgICMgdGhhdCBhIHNoYWxsb3cgZ2xvYmFsIG1pbmltdW0gZG9lcyBub3QgaGlkZSBhIGNvbXBldGluZyBzb2x1dGlvbi4KICAgIHJ1bm5lcnMgPSBbXQogICAgZm9yIGsgaW4gZmxhdFsxOl06CiAgICAgICAgaSwgaiA9IG5wLnVucmF2ZWxfaW5kZXgoaywgZ3JpZC5zaGFwZSkKICAgICAgICBzZXBfYW5nID0gbWluKGFicyhhbmdsZXNbal0gLSBhbmdsZXNbYmVzdF9qXSksCiAgICAgICAgICAgICAgICAgICAgICAxODAuMCAtIGFicyhhbmdsZXNbal0gLSBhbmdsZXNbYmVzdF9qXSkpCiAgICAgICAgaWYgc2VwX2FuZyA8IDIwLjAgYW5kIGFicyhpbmNzW2ldIC0gaW5jc1tiZXN0X2ldKSA8IDE1LjA6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgaWYgYW55KG1pbihhYnMoYW5nbGVzW2pdIC0gYSksIDE4MC4wIC0gYWJzKGFuZ2xlc1tqXSAtIGEpKSA8IDIwLjAKICAgICAgICAgICAgICAgYW5kIGFicyhpbmNzW2ldIC0gYykgPCAxNS4wIGZvciBhLCBjLCBfIGluIHJ1bm5lcnMpOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHJ1bm5lcnMuYXBwZW5kKChmbG9hdChhbmdsZXNbal0pLCBmbG9hdChpbmNzW2ldKSwgZmxvYXQoZ3JpZFtpLCBqXSkpKQogICAgICAgIGlmIGxlbihydW5uZXJzKSA+PSAzOgogICAgICAgICAgICBicmVhawoKICAgIHJldHVybiB7CiAgICAgICAgImFuZ2xlIjogZmxvYXQoYW5nbGVzW2Jlc3Rfal0pLAogICAgICAgICJpbmMiOiBmbG9hdChpbmNzW2Jlc3RfaV0pLAogICAgICAgICJkaXNwZXJzaW9uIjogZmxvYXQoZ3JpZFtiZXN0X2ksIGJlc3Rfal0pLAogICAgICAgICJuX29iamVjdHMiOiBpbnQoZHguc2l6ZSksCiAgICAgICAgIm5fc2VjdG9ycyI6IG5fc2VjdG9ycywKICAgICAgICAiYW5nbGVzIjogYW5nbGVzLAogICAgICAgICJpbmNzIjogaW5jcywKICAgICAgICAiZ3JpZCI6IGdyaWQsCiAgICAgICAgInJ1bm5lcnNfdXAiOiBydW5uZXJzLAogICAgICAgICJmZXdfb2JqZWN0cyI6IGJvb2woZHguc2l6ZSA8IDIwMCksCiAgICB9CgpkZWYgZGlzdHJpYnV0aW9uX3NoYXBlKGR4LCBkeSk6CiAgICAiIiJUaGUgYXhpcyByYXRpbyBhbmQgcG9zaXRpb24gYW5nbGUgb2YgYSBzZXQgb2YgcG9zaXRpb25zLgoKICAgIEZyb20gdGhlIHNlY29uZCBtb21lbnRzOiB0aGUgc2hhcGUgb2YgdGhlIGNsb3VkIG9mIHBvaW50cywgbm90IG9mIGFueSBvbmUgb2YKICAgIHRoZW0uIFJldHVybnMgKGF4aXNfcmF0aW8sIGFuZ2xlX2RlZyksIHRoZSBhbmdsZSBtZWFzdXJlZCBmcm9tICt4IHRoZSBzYW1lIHdheQogICAgdGhlIGRlcHJvamVjdGlvbiBtZWFzdXJlcyBpdC4KICAgICIiIgogICAgZHggPSBucC5hc2FycmF5KGR4LCBkdHlwZT1mbG9hdCkKICAgIGR5ID0gbnAuYXNhcnJheShkeSwgZHR5cGU9ZmxvYXQpCiAgICBvayA9IG5wLmlzZmluaXRlKGR4KSAmIG5wLmlzZmluaXRlKGR5KQogICAgaWYgb2suc3VtKCkgPCAyMDoKICAgICAgICByZXR1cm4gZmxvYXQoIm5hbiIpLCBmbG9hdCgibmFuIikKICAgIGNvdiA9IG5wLmNvdihkeFtva10sIGR5W29rXSkKICAgIHZhbHVlcywgdmVjdG9ycyA9IG5wLmxpbmFsZy5laWdoKGNvdikKICAgIGlmIHZhbHVlc1sxXSA8PSAwOgogICAgICAgIHJldHVybiBmbG9hdCgibmFuIiksIGZsb2F0KCJuYW4iKQogICAgcmF0aW8gPSBmbG9hdChucC5zcXJ0KG1heCh2YWx1ZXNbMF0sIDAuMCkgLyB2YWx1ZXNbMV0pKQogICAgYW5nbGUgPSBmbG9hdChucC5kZWdyZWVzKG5wLmFyY3RhbjIodmVjdG9yc1sxLCAxXSwgdmVjdG9yc1swLCAxXSkpICUgMTgwLjApCiAgICByZXR1cm4gcmF0aW8sIGFuZ2xlCgpkZWYgY2hlY2tfYWdhaW5zdF9jYXRhbG9ndWUoZHgsIGR5LCBleHBlY3RlZF9yYXRpbywgZXhwZWN0ZWRfYW5nbGUsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICByYXRpb190b2xlcmFuY2U9MC4xNSwgYW5nbGVfdG9sZXJhbmNlPTI1LjApOgogICAgIiIiQ29tcGFyZSB0aGUgc2hhcGUgb2Ygd2hhdCB3YXMgZGV0ZWN0ZWQgd2l0aCB3aGF0IHRoZSBjYXRhbG9ndWUgcHJlZGljdHMuCgogICAgV0hZIFRISVMgSVMgV09SVEggRE9JTkcKICAgIEEgdGlsdGVkIGRpc2sgc2VlbiBmcm9tIGhlcmUgaXMgYW4gZWxsaXBzZSwgYW5kIHRoZSBjYXRhbG9ndWUgc2F5cyB3aGljaAogICAgZWxsaXBzZS4gU28gdGhlIHRyYWNlcnMgZGV0ZWN0ZWQgaW4gaXQgaGF2ZSBhIHNoYXBlIHRoYXQgY2FuIGJlIHByZWRpY3RlZAogICAgYmVmb3JlIHRoZXkgYXJlIG1lYXN1cmVkIC0gYW5kIGNvbXBhcmluZyB0aGUgdHdvIGNvc3RzIG5vdGhpbmcgYW5kIG5lZWRzIG5vCiAgICBleHRyYSBkYXRhLgoKICAgIEl0IGlzIGEgcmVhbCB0ZXN0IGJlY2F1c2UgaXQgY2FuIGZhaWwuIE9uIGEgc2V0IG9mIE0zMyBmcmFtZXMgdGhlIHRvb2wKICAgIGRldGVjdGVkIDI4NDQgcmVnaW9ucyBhbmQgZHJldyBhIHNtb290aCwgZW50aXJlbHkgcGxhdXNpYmxlIHJhZGlhbCBwcm9maWxlOwogICAgdGhlIGRldGVjdGlvbnMgY2FtZSBvdXQgcm91bmQsIGF4aXMgcmF0aW8gMC45Niwgd2hpbGUgdGhlIGNhdGFsb2d1ZSBzYXlzIDAuNTkuCiAgICBUaGV5IHdlcmUgbm90IHRyYWNpbmcgdGhlIGdhbGF4eSBhdCBhbGwgLSB0aGV5IHdlcmUgc3VidHJhY3Rpb24gcmVzaWR1YWxzCiAgICBzcHJlYWQgZXZlbmx5IG92ZXIgdGhlIGZpZWxkIC0gYW5kIG5vdGhpbmcgZWxzZSBpbiB0aGUgb3V0cHV0IHNob3dlZCBpdC4KCiAgICBXaGVuIHRoZSB0d28gYWdyZWUgaXQgaXMgZXZpZGVuY2UgdGhlIGRldGVjdGlvbiBpcyBmaW5kaW5nIHRoZSBnYWxheHkuIFdoZW4KICAgIHRoZXkgZGlzYWdyZWUsIHNvbWV0aGluZyBpcyB3cm9uZyB1cHN0cmVhbSBhbmQgdGhlIHByb2ZpbGUgYmVsb3cgc2hvdWxkIG5vdCBiZQogICAgYmVsaWV2ZWQuIFRoZSBjaGVjayBkb2VzIG5vdCBzYXkgd2hpY2ggb2YgdGhlIHR3byBpcyBhdCBmYXVsdC4KCiAgICBBIGdhbGF4eSB0b28gcm91bmQgZm9yIGl0cyBvd24gcG9zaXRpb24gYW5nbGUgdG8gYmUgZGVmaW5lZCBpcyBub3QgdGVzdGVkIG9uCiAgICBhbmdsZSwgZm9yIHRoZSBzYW1lIHJlYXNvbiB0aGUgY2F0YWxvZ3VlIGRlY2xpbmVzIHRvIHB1Ymxpc2ggb25lLgogICAgIiIiCiAgICByYXRpbywgYW5nbGUgPSBkaXN0cmlidXRpb25fc2hhcGUoZHgsIGR5KQogICAgaWYgbm90IG5wLmlzZmluaXRlKHJhdGlvKToKICAgICAgICByZXR1cm4geyJvayI6IE5vbmUsICJyZWFzb24iOiAidG9vIGZldyBwb3NpdGlvbnMgdG8gbWVhc3VyZSBhIHNoYXBlIn0KCiAgICBvdXQgPSB7Im1lYXN1cmVkX3JhdGlvIjogcmF0aW8sICJtZWFzdXJlZF9hbmdsZSI6IGFuZ2xlLAogICAgICAgICAgICJleHBlY3RlZF9yYXRpbyI6IGV4cGVjdGVkX3JhdGlvLCAiZXhwZWN0ZWRfYW5nbGUiOiBleHBlY3RlZF9hbmdsZX0KCiAgICBwcm9ibGVtcyA9IFtdCiAgICBpZiBleHBlY3RlZF9yYXRpbyBpcyBub3QgTm9uZSBhbmQgbnAuaXNmaW5pdGUoZXhwZWN0ZWRfcmF0aW8pOgogICAgICAgIGlmIGFicyhyYXRpbyAtIGV4cGVjdGVkX3JhdGlvKSA+IHJhdGlvX3RvbGVyYW5jZToKICAgICAgICAgICAgcHJvYmxlbXMuYXBwZW5kKAogICAgICAgICAgICAgICAgInRoZSBkZXRlY3Rpb25zIGFyZSBzaGFwZWQgezouMmZ9IHdoZXJlIHRoZSBjYXRhbG9ndWUgc2F5cyB7Oi4yZn0iCiAgICAgICAgICAgICAgICAuZm9ybWF0KHJhdGlvLCBleHBlY3RlZF9yYXRpbykpCiAgICAjIEEgcm91bmQgZ2FsYXh5IGhhcyBubyBtZWFuaW5nZnVsIG1ham9yIGF4aXMsIHNvIGl0cyBhbmdsZSBpcyBub3QgdGVzdGVkLgogICAgaWYgKGV4cGVjdGVkX2FuZ2xlIGlzIG5vdCBOb25lIGFuZCBucC5pc2Zpbml0ZShleHBlY3RlZF9hbmdsZSkKICAgICAgICAgICAgYW5kIGV4cGVjdGVkX3JhdGlvIGlzIG5vdCBOb25lIGFuZCBleHBlY3RlZF9yYXRpbyA8IDAuODUpOgogICAgICAgIGdhcCA9IGFicygoYW5nbGUgLSBleHBlY3RlZF9hbmdsZSArIDkwLjApICUgMTgwLjAgLSA5MC4wKQogICAgICAgIG91dFsiYW5nbGVfZ2FwIl0gPSBnYXAKICAgICAgICBpZiBnYXAgPiBhbmdsZV90b2xlcmFuY2U6CiAgICAgICAgICAgIHByb2JsZW1zLmFwcGVuZCgKICAgICAgICAgICAgICAgICJ0aGVpciBsb25nIGF4aXMgbGllcyBhdCB7Oi4wZn0gZGVnIHdoZXJlIHRoZSBjYXRhbG9ndWUgc2F5cyB7Oi4wZn0iCiAgICAgICAgICAgICAgICAuZm9ybWF0KGFuZ2xlLCBleHBlY3RlZF9hbmdsZSkpCgogICAgb3V0WyJvayJdID0gbm90IHByb2JsZW1zCiAgICBvdXRbInByb2JsZW1zIl0gPSBwcm9ibGVtcwogICAgcmV0dXJuIG91dAoKZGVmIGRlc2NyaWJlX2NoZWNrKHJlc3VsdCk6CiAgICAiIiJUaGUgc2VsZi1jaGVjaywgaW4gYSBmb3JtIHdvcnRoIHByaW50aW5nLiIiIgogICAgaWYgcmVzdWx0LmdldCgib2siKSBpcyBOb25lOgogICAgICAgIHJldHVybiAiICBzaGFwZSBjaGVjazogIiArIHJlc3VsdC5nZXQoInJlYXNvbiIsICJub3QgZG9uZSIpCiAgICBoZWFkID0gKCIgIHNoYXBlIGNoZWNrOiB0aGUgZGV0ZWN0aW9ucyBhcmUgc2hhcGVkIHs6LjJmfSB3aXRoIHRoZWlyIGxvbmcgYXhpcyAiCiAgICAgICAgICAgICJhdCB7Oi4wZn0gZGVnIi5mb3JtYXQocmVzdWx0WyJtZWFzdXJlZF9yYXRpbyJdLCByZXN1bHRbIm1lYXN1cmVkX2FuZ2xlIl0pKQogICAgaWYgcmVzdWx0LmdldCgiZXhwZWN0ZWRfcmF0aW8iKSBpcyBub3QgTm9uZToKICAgICAgICBoZWFkICs9ICI7IHRoZSBjYXRhbG9ndWUgcHJlZGljdHMgezouMmZ9Ii5mb3JtYXQocmVzdWx0WyJleHBlY3RlZF9yYXRpbyJdKQogICAgICAgIGlmIHJlc3VsdC5nZXQoImV4cGVjdGVkX2FuZ2xlIikgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIGhlYWQgKz0gIiBhdCB7Oi4wZn0gZGVnIi5mb3JtYXQocmVzdWx0WyJleHBlY3RlZF9hbmdsZSJdKQogICAgaWYgcmVzdWx0WyJvayJdOgogICAgICAgIHJldHVybiBoZWFkICsgIlxuICAgIHRoZXkgYWdyZWUgLSB0aGUgZGV0ZWN0aW9ucyBkbyB0cmFjZSB0aGlzIGdhbGF4eSIKICAgIHJldHVybiAoaGVhZCArICJcbiAgICBUSEVZIERJU0FHUkVFOiAiICsgIjsgIi5qb2luKHJlc3VsdFsicHJvYmxlbXMiXSkgKwogICAgICAgICAgICAiXG4gICAgU29tZXRoaW5nIHVwc3RyZWFtIGlzIHdyb25nLiBEZXRlY3Rpb25zIHRoYXQgZG8gbm90IGZvbGxvdyB0aGUgIgogICAgICAgICAgICAiZ2FsYXh5J3NcbiAgICBzaGFwZSBhcmUgdXN1YWxseSBzdWJ0cmFjdGlvbiByZXNpZHVhbHMgc3ByZWFkIG92ZXIgdGhlICIKICAgICAgICAgICAgIndob2xlIGZpZWxkLCBhbmRcbiAgICB0aGUgcmFkaWFsIHByb2ZpbGUgYmVsb3cgd2lsbCBsb29rIHNtb290aCBhbmQgIgogICAgICAgICAgICAibWVhbiBub3RoaW5nLiIpCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIHNoYXJlZCBtb2R1bGUgZ2FsYXh5X2NhdGFsb2d1ZS5weQojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCgpwYXNzICAjIGZyb20gZGVwcm9qZWN0aW9uLCBpbmxpbmVkIGFib3ZlCgpIWVBFUkxFREFfVklaSUVSX0lEID0gIlZJSS8yMzciCgpkZWYgaHlwZXJsZWRhX2dlb21ldHJ5KGdhbGF4eV9uYW1lOiBzdHIsIHNlYXJjaF9yYWRpdXNfYXJjbWluOiBmbG9hdCA9IDAuNSkgLT4gZGljdDoKICAgICIiIkZldGNoIHBvc2l0aW9uIGFuZ2xlLCBheGlzIHJhdGlvIGFuZCBpbmNsaW5hdGlvbiBmb3Igb25lIGdhbGF4eS4KCiAgICBSZXR1cm5zIGEgZGljdCB3aXRoOgogICAgICAgIG5hbWVfcXVlcnksIHBnYywgbXR5cGUgIC0gaWRlbnRpZmljYXRpb24KICAgICAgICBsb2dfcjI1LCBiX292ZXJfYSAgICAgICAtIHRoZSBvYnNlcnZlZCBheGlzIHJhdGlvCiAgICAgICAgcGFfc2t5ICAgICAgICAgICAgICAgICAgLSBwb3NpdGlvbiBhbmdsZSBvbiB0aGUgc2t5LCBvciBOb25lIGlmIG5vdCBwdWJsaXNoZWQKICAgICAgICBpbmNfbmFpdmUgICAgICAgICAgICAgICAtIGFjb3MoYi9hKSwgaWdub3JpbmcgZGlzayB0aGlja25lc3MKICAgICAgICBpbmMgICAgICAgICAgICAgICAgICAgICAtIHRoZSBIdWJibGUtY29ycmVjdGVkIGluY2xpbmF0aW9uIHdlIGFjdHVhbGx5IHVzZQogICAgICAgIHEwICAgICAgICAgICAgICAgICAgICAgIC0gdGhlIHRoaWNrbmVzcyBjb25zdGFudCBjaG9zZW4gZm9yIHRoaXMgdHlwZQogICAgICAgIGQyNV9hcmNtaW4gICAgICAgICAgICAgIC0gaXNvcGhvdGFsIGRpYW1ldGVyLCB1c2VmdWwgZm9yIG5vcm1hbGlzaW5nIHJhZGlpCiAgICAgICAgc291cmNlICAgICAgICAgICAgICAgICAgLSBodW1hbi1yZWFkYWJsZSBwcm92ZW5hbmNlCgogICAgUmFpc2VzIExvb2t1cEVycm9yIHdoZW4gdGhlIG5hbWUgcmVzb2x2ZXMgdG8gbm90aGluZywgc28gdGhlIGNhbGxlciBjYW4gZmFsbAogICAgYmFjayB0byBtYW51YWxseSBzdXBwbGllZCBhbmdsZXMgaW5zdGVhZCBvZiBzaWxlbnRseSBjb250aW51aW5nLgogICAgIiIiCiAgICBmcm9tIGFzdHJvcXVlcnkudml6aWVyIGltcG9ydCBWaXppZXIKICAgIGltcG9ydCBhc3Ryb3B5LnVuaXRzIGFzIHUKCiAgICB2ID0gVml6aWVyKGNhdGFsb2c9SFlQRVJMRURBX1ZJWklFUl9JRCwgcm93X2xpbWl0PS0xKQogICAgcmVzdWx0ID0gdi5xdWVyeV9vYmplY3QoZ2FsYXh5X25hbWUsIHJhZGl1cz1zZWFyY2hfcmFkaXVzX2FyY21pbiAqIHUuYXJjbWluKQogICAgaWYgbm90IHJlc3VsdCBvciBsZW4ocmVzdWx0WzBdKSA9PSAwOgogICAgICAgIHJhaXNlIExvb2t1cEVycm9yKAogICAgICAgICAgICBmIkh5cGVyTEVEQSByZXR1cm5lZCBubyByb3cgZm9yICd7Z2FsYXh5X25hbWV9Jy4gQ2hlY2sgdGhlIG5hbWUsIG9yICIKICAgICAgICAgICAgInN1cHBseSB0aGUgYW5nbGVzIG1hbnVhbGx5LiIKICAgICAgICApCgogICAgcm93ID0gcmVzdWx0WzBdWzBdCgogICAgZGVmIF9nZXQoY29sKToKICAgICAgICB0cnk6CiAgICAgICAgICAgIHZhbCA9IHJvd1tjb2xdCiAgICAgICAgZXhjZXB0IEtleUVycm9yOgogICAgICAgICAgICByZXR1cm4gTm9uZQogICAgICAgIHRyeToKICAgICAgICAgICAgaWYgaGFzYXR0cih2YWwsICJtYXNrIikgYW5kIHZhbC5tYXNrOgogICAgICAgICAgICAgICAgcmV0dXJuIE5vbmUKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBwYXNzCiAgICAgICAgdHJ5OgogICAgICAgICAgICBmID0gZmxvYXQodmFsKQogICAgICAgIGV4Y2VwdCAoVHlwZUVycm9yLCBWYWx1ZUVycm9yKToKICAgICAgICAgICAgcmV0dXJuIE5vbmUKICAgICAgICByZXR1cm4gTm9uZSBpZiBtYXRoLmlzbmFuKGYpIGVsc2UgZgoKICAgIGxvZ19yMjUgPSBfZ2V0KCJsb2dSMjUiKQogICAgaWYgbG9nX3IyNSBpcyBOb25lOgogICAgICAgIHJhaXNlIExvb2t1cEVycm9yKAogICAgICAgICAgICBmIkh5cGVyTEVEQSBoYXMgYSByb3cgZm9yICd7Z2FsYXh5X25hbWV9JyBidXQgbm8gYXhpcyByYXRpbyAobG9nUjI1KSwgIgogICAgICAgICAgICAic28gdGhlIGluY2xpbmF0aW9uIGNhbm5vdCBiZSBkZXJpdmVkLiBTdXBwbHkgdGhlIGFuZ2xlcyBtYW51YWxseS4iCiAgICAgICAgKQoKICAgIGJfb3Zlcl9hID0gMTAuMCAqKiAoLWxvZ19yMjUpCiAgICBtdHlwZSA9IE5vbmUKICAgIHRyeToKICAgICAgICBtdHlwZSA9IHN0cihyb3dbIk1UeXBlIl0pLnN0cmlwKCkgb3IgTm9uZQogICAgZXhjZXB0IEtleUVycm9yOgogICAgICAgIHBhc3MKCiAgICBxMCA9IHEwX2Zvcl90eXBlKG10eXBlKQogICAgbG9nX2QyNSA9IF9nZXQoImxvZ0QyNSIpCgogICAgcmV0dXJuIHsKICAgICAgICAibmFtZV9xdWVyeSI6IGdhbGF4eV9uYW1lLAogICAgICAgICJwZ2MiOiBfZ2V0KCJQR0MiKSwKICAgICAgICAibXR5cGUiOiBtdHlwZSwKICAgICAgICAibG9nX3IyNSI6IGxvZ19yMjUsCiAgICAgICAgImJfb3Zlcl9hIjogYl9vdmVyX2EsCiAgICAgICAgInBhX3NreSI6IF9nZXQoIlBBIiksCiAgICAgICAgImluY19uYWl2ZSI6IG1hdGguZGVncmVlcyhtYXRoLmFjb3MobWluKG1heChiX292ZXJfYSwgMC4wKSwgMS4wKSkpLAogICAgICAgICJpbmMiOiBpbmNsaW5hdGlvbl9mcm9tX2F4aXNfcmF0aW8oYl9vdmVyX2EsIHEwKSwKICAgICAgICAicTAiOiBxMCwKICAgICAgICAiZDI1X2FyY21pbiI6ICgwLjEgKiAxMC4wICoqIGxvZ19kMjUpIGlmIGxvZ19kMjUgaXMgbm90IE5vbmUgZWxzZSBOb25lLAogICAgICAgICJzb3VyY2UiOiBmIkh5cGVyTEVEQSB2aWEgVml6aWVSIHtIWVBFUkxFREFfVklaSUVSX0lEfSIsCiAgICB9CgpkZWYgZGVzY3JpYmUoZ2VvbTogZGljdCkgLT4gc3RyOgogICAgIiIiT25lIHJlYWRhYmxlIGJsb2NrIHN1bW1hcmlzaW5nIHdoYXQgdGhlIGNhdGFsb2d1ZSBnYXZlIHVzLiIiIgogICAgbGluZXMgPSBbCiAgICAgICAgZiIgIGNhdGFsb2d1ZSAgICAgIDoge2dlb21bJ3NvdXJjZSddfSIsCiAgICAgICAgZiIgIFBHQyAgICAgICAgICAgIDoge2dlb21bJ3BnYyddfSIsCiAgICAgICAgZiIgIHR5cGUgICAgICAgICAgIDoge2dlb21bJ210eXBlJ119IiwKICAgICAgICBmIiAgYi9hICAgICAgICAgICAgOiB7Z2VvbVsnYl9vdmVyX2EnXTouM2Z9ICAgKGxvZ1IyNSA9IHtnZW9tWydsb2dfcjI1J106LjNmfSkiLAogICAgICAgIGYiICBpbmNsaW5hdGlvbiAgICA6IHtnZW9tWydpbmMnXTouMWZ9IGRlZyAgICIKICAgICAgICBmIihuYWl2ZSBhY29zKGIvYSkgPSB7Z2VvbVsnaW5jX25haXZlJ106LjFmfSBkZWcsIHEwID0ge2dlb21bJ3EwJ106LjJmfSkiLAogICAgXQogICAgaWYgZ2VvbVsicGFfc2t5Il0gaXMgTm9uZToKICAgICAgICBsaW5lcy5hcHBlbmQoIiAgcG9zaXRpb24gYW5nbGUgOiBOT1QgUFVCTElTSEVEIC0gdGhlIGdhbGF4eSBpcyB0b28gcm91bmQgZm9yIHRoZSAiCiAgICAgICAgICAgICAgICAgICAgICJtYWpvciBheGlzIHRvIGJlIGRlZmluZWQiKQogICAgZWxzZToKICAgICAgICBsaW5lcy5hcHBlbmQoZiIgIHBvc2l0aW9uIGFuZ2xlIDoge2dlb21bJ3BhX3NreSddOi4xZn0gZGVnIG9uIHRoZSBza3kgKE4gdGhyb3VnaCBFKSIpCiAgICBpZiBnZW9tWyJkMjVfYXJjbWluIl06CiAgICAgICAgbGluZXMuYXBwZW5kKGYiICBEMjUgICAgICAgICAgICA6IHtnZW9tWydkMjVfYXJjbWluJ106LjFmfSBhcmNtaW4iKQogICAgcmV0dXJuICJcbiIuam9pbihsaW5lcykKClJfViA9IDMuMSAgICAgICAgICAgICAgICAgICAgICAgICAgIyBBX1YgLyBFKEItVikgZm9yIGR1c3QgaW4gb3VyIG93biBnYWxheHkKCkRJU1RBTkNFX0NBVEFMT0cgPSAiSi9BSi8xNTIvNTAiICAgIyBDb3NtaWNmbG93cy0zLCBUdWxseSBldCBhbC4gMjAxNgoKZGVmIGxvb2t1cF9kaXN0YW5jZShuYW1lKToKICAgICIiIlJlZHNoaWZ0LWluZGVwZW5kZW50IGRpc3RhbmNlIGluIE1wYyBmcm9tIENvc21pY2Zsb3dzLTMsIG9yIE5vbmUuCgogICAgRXZlcnkgbGVuZ3RoIHRoaXMgdG9vbCByZXBvcnRzIGlzIHRoaXMgbnVtYmVyIHRpbWVzIGFuIGFuZ2xlLCBzbyBhIHdyb25nCiAgICBkaXN0YW5jZSByZXNjYWxlcyB0aGUgd2hvbGUgYW5zd2VyIGFuZCBub3RoaW5nIGluIHRoZSBvdXRwdXQgbG9va3MgYW1pc3MuCiAgICBSZWRzaGlmdCBpcyBubyBoZWxwIGZvciBhIGdhbGF4eSB0aGlzIGNsb3NlIC0gdGhlIHJlY2Vzc2lvbiB2ZWxvY2l0eSBvZiBhCiAgICBmZXctTXBjIGdhbGF4eSBpcyBkb21pbmF0ZWQgYnkgaXRzIG93biBtb3Rpb24gdGhyb3VnaCB0aGUgbG9jYWwgZ3JvdXAsIGFuZAogICAgTTEwMSdzIHB1dHMgaXQgYXQgYWJvdXQgaGFsZiBpdHMgdHJ1ZSBkaXN0YW5jZSAtIHNvIGEgY2F0YWxvZ3VlIG9mIGRpcmVjdAogICAgbWVhc3VyZW1lbnRzIGlzIHdoYXQgaXMgd2FudGVkLgoKICAgIEl0IGlzIG9mZmVyZWQsIG5ldmVyIGltcG9zZWQ6IHRoZSBjYWxsZXIgaXMgZXhwZWN0ZWQgdG8gc2hvdyBpdCBhbmQgbGV0IHRoZQogICAgb2JzZXJ2ZXIgb3ZlcnJ1bGUgaXQuCiAgICAiIiIKICAgIHRyeToKICAgICAgICBmcm9tIGFzdHJvcHkuY29vcmRpbmF0ZXMgaW1wb3J0IFNreUNvb3JkCiAgICAgICAgZnJvbSBhc3Ryb3F1ZXJ5LnZpemllciBpbXBvcnQgVml6aWVyCiAgICAgICAgaW1wb3J0IGFzdHJvcHkudW5pdHMgYXMgdQoKICAgICAgICBjb29yZCA9IFNreUNvb3JkLmZyb21fbmFtZShuYW1lKQogICAgICAgICMgIl9yIiBpcyBWaXppZVIncyBvd24gZGlzdGFuY2Ugb2YgZWFjaCByb3cgZnJvbSB0aGUgcG9zaXRpb24gYXNrZWQKICAgICAgICAjIGFib3V0LiBUaGUgcm93cyBjb21lIGJhY2sgaW4gY2F0YWxvZ3VlIG9yZGVyLCBub3QgbmVhcmVzdCBmaXJzdCwgc28KICAgICAgICAjIHdpdGhvdXQgaXQgdGhlIGZpcnN0IHJvdyBjb3VsZCBiZSBhIG5laWdoYm91cmluZyBnYWxheHkgaW5zaWRlIHRoZQogICAgICAgICMgc2VhcmNoIGNpcmNsZSBhbmQgaXRzIGRpc3RhbmNlIHdvdWxkIGJlIHVzZWQgd2l0aG91dCBhIHdvcmQuCiAgICAgICAgaGl0ID0gVml6aWVyKGNvbHVtbnM9WyIqKiIsICJfciJdLCByb3dfbGltaXQ9MjApLnF1ZXJ5X3JlZ2lvbigKICAgICAgICAgICAgY29vcmQsIHJhZGl1cz0yICogdS5hcmNtaW4sIGNhdGFsb2c9RElTVEFOQ0VfQ0FUQUxPRykKICAgICAgICBpZiBub3QgaGl0OgogICAgICAgICAgICByZXR1cm4gTm9uZQogICAgICAgIHRhYmxlID0gaGl0WzBdCiAgICAgICAgaWYgIl9yIiBpbiB0YWJsZS5jb2xuYW1lcyBhbmQgbGVuKHRhYmxlKSA+IDE6CiAgICAgICAgICAgIHRhYmxlID0gdGFibGVbbnAuYXJnc29ydChucC5hc2FycmF5KHRhYmxlWyJfciJdLCBkdHlwZT1mbG9hdCkpXQogICAgICAgIGZvciBjb2wgaW4gKCJEaXN0IiwgIjxEaXN0PiIpOgogICAgICAgICAgICBpZiBjb2wgaW4gdGFibGUuY29sbmFtZXM6CiAgICAgICAgICAgICAgICB2YWx1ZSA9IGZsb2F0KHRhYmxlW2NvbF1bMF0pCiAgICAgICAgICAgICAgICBpZiB2YWx1ZSA9PSB2YWx1ZSBhbmQgdmFsdWUgPiAwOgogICAgICAgICAgICAgICAgICAgIHJldHVybiB2YWx1ZQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICByZXR1cm4gTm9uZQogICAgcmV0dXJuIE5vbmUKCmRlZiBsb29rdXBfZXh0aW5jdGlvbihuYW1lKToKICAgICIiIkdhbGFjdGljIHJlZGRlbmluZyBFKEItVikgdG93YXJkcyB0aGUgZ2FsYXh5LCBvciBOb25lLgoKICAgIFRoZSBTY2hsYWZseSAmIEZpbmtiZWluZXIgKDIwMTEpIHJlY2FsaWJyYXRpb24gb2YgdGhlIFNjaGxlZ2VsIGR1c3QgbWFwcywKICAgIHNlcnZlZCBieSBJUlNBLiBOZWVkZWQgd2hlcmV2ZXIgYSBjb2xvdXIgaXMgdHVybmVkIGludG8gYSBzdGF0ZW1lbnQgYWJvdXQgYQogICAgc3RlbGxhciBwb3B1bGF0aW9uOiBkdXN0IGluIG91ciBvd24gZ2FsYXh5IHJlZGRlbnMgZXZlcnl0aGluZyBiZWhpbmQgaXQsIGFuZAogICAgd2l0aG91dCB0aGUgY29ycmVjdGlvbiBldmVyeSBzb3VyY2UgaXMgc2hpZnRlZCB0b3dhcmRzIHRoZSByZWQsIHdoaWNoIG1vdmVzCiAgICB0aGUgbGluZSBiZXR3ZWVuIGEgeW91bmcgYmx1ZSBjbHVzdGVyIGFuZCBhbiBvcmRpbmFyeSBzdGFyLgoKICAgIFdoZXJlIGl0IGNhbmNlbHMsIGFuZCB0aGlzIGlzIHdvcnRoIGtub3dpbmcgc28gaXQgZG9lcyBub3QgZ2V0ICJmaXhlZCIgbGF0ZXI6CiAgICBmaXR0aW5nIHRoZSBjb250aW51dW0gc3VidHJhY3Rpb24gZmFjdG9yIGFnYWluc3QgYW4gaW5zdHJ1bWVudGFsIGNvbG91cgogICAgbWVhc3VyZWQgZnJvbSB0aGUgc2FtZSBwYWlyIG9mIGZyYW1lcy4gUmVkZGVuaW5nIHNoaWZ0cyBldmVyeSBzdGFyIGJ5IHRoZSBzYW1lCiAgICBhbW91bnQgdGhlcmUsIHRoZSBmaXR0ZWQgbGluZSBzaGlmdHMgd2l0aCB0aGVtLCBhbmQgdGhlIHZhbHVlIHJlYWQgb2ZmIGl0IGZvcgogICAgYW55IGdpdmVuIHBpeGVsIGlzIHVuY2hhbmdlZC4KICAgICIiIgogICAgdHJ5OgogICAgICAgIGZyb20gYXN0cm9xdWVyeS5pcGFjLmlyc2EuaXJzYV9kdXN0IGltcG9ydCBJcnNhRHVzdAogICAgICAgIHRhYmxlID0gSXJzYUR1c3QuZ2V0X3F1ZXJ5X3RhYmxlKG5hbWUsIHNlY3Rpb249ImVidiIpCiAgICAgICAgdmFsdWUgPSBmbG9hdCh0YWJsZVsiZXh0IFNhbmRGIG1lYW4iXVswXSkKICAgICAgICBpZiB2YWx1ZSA9PSB2YWx1ZSBhbmQgdmFsdWUgPj0gMDoKICAgICAgICAgICAgcmV0dXJuIHZhbHVlCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHJldHVybiBOb25lCiAgICByZXR1cm4gTm9uZQoKZGVmIGRlc2NyaWJlX2xvb2t1cHMobmFtZSwgZGlzdGFuY2VfbXBjLCBlYnYsIGRpc3RhbmNlX3NvdXJjZT0iIiwgZWJ2X3NvdXJjZT0iIik6CiAgICAiIiJXaGF0IHdhcyBsb29rZWQgdXAsIHNob3duIGJlZm9yZSBpdCBpcyB1c2VkLiIiIgogICAgbGluZXMgPSBbXQogICAgaWYgZGlzdGFuY2VfbXBjOgogICAgICAgIGxpbmVzLmFwcGVuZCgiICBkaXN0YW5jZSAgICB7Oi4yZn0gTXBjICAge30iLmZvcm1hdCgKICAgICAgICAgICAgZGlzdGFuY2VfbXBjLCBkaXN0YW5jZV9zb3VyY2Ugb3IgImZyb20gQ29zbWljZmxvd3MtMyAoVHVsbHkgZXQgYWwuIDIwMTYpIikpCiAgICAgICAgbGluZXMuYXBwZW5kKCIgICAgICAgICAgICAgIGV2ZXJ5IGxlbmd0aCBiZWxvdyBzY2FsZXMgd2l0aCB0aGlzIG51bWJlciIpCiAgICBlbHNlOgogICAgICAgIGxpbmVzLmFwcGVuZCgiICBkaXN0YW5jZSAgICBub3QgZm91bmQgLSByYWRpaSB3aWxsIGJlIGluIGFyY3NlYywgbm90IHBhcnNlY3MiKQogICAgaWYgZWJ2IGlzIG5vdCBOb25lOgogICAgICAgIGxpbmVzLmFwcGVuZCgiICByZWRkZW5pbmcgICBFKEItVikgPSB7Oi40Zn0sIHNvIEFfViA9IHs6LjNmfSAgIHt9Ii5mb3JtYXQoCiAgICAgICAgICAgIGVidiwgUl9WICogZWJ2LAogICAgICAgICAgICBlYnZfc291cmNlIG9yICJmcm9tIFNjaGxhZmx5ICYgRmlua2JlaW5lciAoMjAxMSkiKSkKICAgIHJldHVybiAiXG4iLmpvaW4obGluZXMpCgpkZWYgZ2FpYV9mb3JlZ3JvdW5kX3N0YXJzKHdjcywgc2hhcGUsIGdfbGltaXQ9MTkuMCwgc2lnbmlmaWNhbmNlPTMuMCwKICAgICAgICAgICAgICAgICAgICAgICAgICBtYXhfcm93cz01MDAwMCk6CiAgICAiIiJQb3NpdGlvbnMsIGluIHBpeGVscywgb2Ygc3RhcnMgdGhhdCBhcmUgaW4gZnJvbnQgb2YgdGhlIGdhbGF4eS4KCiAgICBXSFkgQSBDQVRBTE9HVUUgQU5EIE5PVCBUSEUgSU1BR0UKICAgIEV2ZXJ5dGhpbmcgZG93bnN0cmVhbSB0aGF0IG5lZWRzICJhIHN0YXIiIG5lZWRzIGEgZm9yZWdyb3VuZCBzdGFyOiBzb21ldGhpbmcKICAgIHdob3NlIGxpZ2h0IGlzIHB1cmUgY29udGludXVtLCBzbyB0aGF0IHNjYWxpbmcgb25lIGZpbHRlciBhZ2FpbnN0IGFub3RoZXIgYW5kCiAgICBzdWJ0cmFjdGluZyBtYWtlcyBpdCB2YW5pc2guIERldGVjdGluZyBwb2ludCBzb3VyY2VzIGluIHRoZSBpbWFnZSBmaW5kcyB0aG9zZSwKICAgIGJ1dCBpbiBhIG5lYXJieSBnYWxheHkgaXQgYWxzbyBmaW5kcyB0aGUgZ2FsYXh5J3Mgb3duIHN0YXJzIC0gYW5kIHRob3NlIGFyZSBub3QKICAgIHB1cmUgY29udGludXVtLiBTb21lIGFyZSByZWQgZ2lhbnRzLCBzb21lIGFyZSBob3QgeW91bmcgc3RhcnMgc2l0dGluZyBpbiB0aGUKICAgIGdhcyB0aGF0IGlzIGJlaW5nIG1lYXN1cmVkLgoKICAgIEhvdyBiYWRseSB0aGlzIG1hdHRlcnMgZGVwZW5kcyBvbiBkaXN0YW5jZSwgd2hpY2ggaXMgd2h5IGl0IHdlbnQgdW5ub3RpY2VkIGZvcgogICAgYSBsb25nIHRpbWUuIEF0IGEgZmV3IE1wYyBhIGdhbGF4eSdzIG93biBzdGFycyBhcmUgbm90IHJlc29sdmVkIGFuZCBldmVyeSBwb2ludAogICAgc291cmNlIHJlYWxseSBpcyBmb3JlZ3JvdW5kLiBBdCBNMzMncyAwLjkgTXBjIHRoZXkgYXJlIHJlc29sdmVkIGluIHRoZWlyCiAgICB0aG91c2FuZHM6IG1lYXN1cmVkIGluIHRoaXMgZmllbGQsIG9mIDQyMDkgR2FpYSBzb3VyY2VzIGJyaWdodGVyIHRoYW4gRz0xOSwKICAgIG9ubHkgMTU5MCBhcmUgZm9yZWdyb3VuZC4gU2V0dGluZyB0aGUgc3VidHJhY3Rpb24gZmFjdG9yIGZyb20gdGhlIG1peHR1cmUgZ2F2ZQogICAgYSBzdGFyLXRvLXN0YXIgc2NhdHRlciBvZiA2NiBwZXJjZW50IC0gd2hpY2ggaXMgdG8gc2F5IG5vIGZhY3RvciBhdCBhbGwuCgogICAgVEhFIFRFU1QKICAgIFBhcmFsbGF4IGFuZCBwcm9wZXIgbW90aW9uLiBBIHN0YXIgYSBmZXcgaHVuZHJlZCBwYXJzZWNzIGF3YXkgaW4gb3VyIG93bgogICAgZ2FsYXh5IGhhcyBib3RoLCBtZWFzdXJhYmx5LiBBIHN0YXIgYXQgMC45IE1wYyBoYXMgbmVpdGhlciwgYXQgYW55IHByZWNpc2lvbgogICAgR2FpYSBjYW4gcmVhY2guIFNvIGEgc2lnbmlmaWNhbnQgcGFyYWxsYXgsIG9yIGEgc2lnbmlmaWNhbnQgcHJvcGVyIG1vdGlvbiwgaXMKICAgIGEgcGh5c2ljYWwgc3RhdGVtZW50IHRoYXQgdGhlIHN0YXIgaXMgaW4gZnJvbnQgLSBub3QgYSBicmlnaHRuZXNzIGN1dCBvciBhCiAgICBzaGFwZSBjdXQgdGhhdCBoYXBwZW5zIHRvIGNvcnJlbGF0ZSB3aXRoIGl0LgoKICAgIFJldHVybnMgKHBvc2l0aW9ucywgZGlhZ25vc3RpY3MpLiBQb3NpdGlvbnMgaXMgYW4gKE4sIDIpIGFycmF5IG9mIHgsIHkgaW4gdGhlCiAgICBmcmFtZTsgYW4gZW1wdHkgYXJyYXkgbWVhbnMgdGhlIHF1ZXJ5IGZhaWxlZCBvciBmb3VuZCBub3RoaW5nLCBhbmQgdGhlIGNhbGxlcgogICAgaXMgZXhwZWN0ZWQgdG8gZmFsbCBiYWNrIG9uIGRldGVjdGluZyBwb2ludCBzb3VyY2VzLgogICAgIiIiCiAgICBpbXBvcnQgbnVtcHkgYXMgbnAKCiAgICBkaWFnID0geyJzb3VyY2UiOiAiR2FpYSBEUjMiLCAibl9jYXRhbG9ndWUiOiAwLCAibl9mb3JlZ3JvdW5kIjogMCwKICAgICAgICAgICAgIm5faW5fZnJhbWUiOiAwLCAid2h5IjogIiJ9CiAgICBpZiB3Y3MgaXMgTm9uZToKICAgICAgICBkaWFnWyJ3aHkiXSA9ICJubyBwbGF0ZSBzb2x1dGlvbiwgc28gdGhlIGNhdGFsb2d1ZSBjYW5ub3QgYmUgcGxhY2VkIgogICAgICAgIHJldHVybiBucC5lbXB0eSgoMCwgMikpLCBkaWFnCgogICAgdHJ5OgogICAgICAgIGZyb20gYXN0cm9xdWVyeS5nYWlhIGltcG9ydCBHYWlhCiAgICAgICAgaW1wb3J0IGFzdHJvcHkudW5pdHMgYXMgdQoKICAgICAgICBueSwgbnggPSBzaGFwZVswXSwgc2hhcGVbMV0KICAgICAgICBjZW50cmUgPSB3Y3MucGl4ZWxfdG9fd29ybGQobnggLyAyLjAsIG55IC8gMi4wKQogICAgICAgIGNvcm5lciA9IHdjcy5waXhlbF90b193b3JsZCgwLjAsIDAuMCkKICAgICAgICByYWRpdXMgPSBmbG9hdChjZW50cmUuc2VwYXJhdGlvbihjb3JuZXIpLnRvKHUuZGVnKS52YWx1ZSkgKiAxLjA1CgogICAgICAgIHF1ZXJ5ID0gKAogICAgICAgICAgICAiU0VMRUNUIHJhLCBkZWMsIHBob3RfZ19tZWFuX21hZyBGUk9NIGdhaWFkcjMuZ2FpYV9zb3VyY2UgIgogICAgICAgICAgICAiV0hFUkUgMT1DT05UQUlOUyhQT0lOVChyYSwgZGVjKSwgQ0lSQ0xFKHs6LjZmfSwgezouNmZ9LCB7Oi40Zn0pKSAiCiAgICAgICAgICAgICJBTkQgcGhvdF9nX21lYW5fbWFnIDwgezouMmZ9ICIKICAgICAgICAgICAgIkFORCAocGFyYWxsYXgvcGFyYWxsYXhfZXJyb3IgPiB7Oi4xZn0gT1IgIgogICAgICAgICAgICAiICAgICBzcXJ0KHBtcmEqcG1yYSArIHBtZGVjKnBtZGVjKSAvICIKICAgICAgICAgICAgIiAgICAgc3FydChwbXJhX2Vycm9yKnBtcmFfZXJyb3IgKyBwbWRlY19lcnJvcipwbWRlY19lcnJvcikgPiB7Oi4xZn0pICIKICAgICAgICAgICAgIk9SREVSIEJZIHBob3RfZ19tZWFuX21hZyIKICAgICAgICApLmZvcm1hdChjZW50cmUucmEuZGVnLCBjZW50cmUuZGVjLmRlZywgcmFkaXVzLCBnX2xpbWl0LAogICAgICAgICAgICAgICAgIHNpZ25pZmljYW5jZSwgc2lnbmlmaWNhbmNlKQoKICAgICAgICBqb2IgPSBHYWlhLmxhdW5jaF9qb2JfYXN5bmMocXVlcnkpIGlmIG1heF9yb3dzID4gMjAwMCBlbHNlIEdhaWEubGF1bmNoX2pvYihxdWVyeSkKICAgICAgICB0YWJsZSA9IGpvYi5nZXRfcmVzdWx0cygpCiAgICAgICAgZGlhZ1sibl9mb3JlZ3JvdW5kIl0gPSBsZW4odGFibGUpCiAgICAgICAgaWYgbm90IGxlbih0YWJsZSk6CiAgICAgICAgICAgIGRpYWdbIndoeSJdID0gIm5vIGZvcmVncm91bmQgc3RhcnMgaW4gdGhlIGZpZWxkIgogICAgICAgICAgICByZXR1cm4gbnAuZW1wdHkoKDAsIDIpKSwgZGlhZwoKICAgICAgICB4LCB5ID0gd2NzLndvcmxkX3RvX3BpeGVsX3ZhbHVlcyhucC5hc2FycmF5KHRhYmxlWyJyYSJdLCBkdHlwZT1mbG9hdCksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbnAuYXNhcnJheSh0YWJsZVsiZGVjIl0sIGR0eXBlPWZsb2F0KSkKICAgICAgICBpbnNpZGUgPSAoeCA+IDUpICYgKHggPCBueCAtIDYpICYgKHkgPiA1KSAmICh5IDwgbnkgLSA2KQogICAgICAgIGRpYWdbIm5faW5fZnJhbWUiXSA9IGludChpbnNpZGUuc3VtKCkpCiAgICAgICAgIyBCcmlnaHRlc3QgZmlyc3QuIFdob2V2ZXIgdGFrZXMgInRoZSBmaXJzdCB0d28gaHVuZHJlZCIgb2YgdGhpcyBsaXN0IG11c3QKICAgICAgICAjIGdldCB0aGUgdHdvIGh1bmRyZWQgd29ydGggbWVhc3VyaW5nOiBhIHN0YXIgbmVhciB0aGUgY2F0YWxvZ3VlIGxpbWl0IGlzCiAgICAgICAgIyBhIGZldyBjb3VudHMgdGhyb3VnaCBhIG5hcnJvdyBmaWx0ZXIsIGFuZCBhIHJhdGlvIGJ1aWx0IG9uIGl0IGlzIG5vaXNlLgogICAgICAgICMgSGFuZGluZyBiYWNrIGFuIHVub3JkZXJlZCBsaXN0IGNvc3QgYSBmYWN0b3Igb2YgdHdvIGluIHRoZSBzY2F0dGVyIG9mIHRoZQogICAgICAgICMgc3VidHJhY3Rpb24gZmFjdG9yIGJlZm9yZSB0aGlzIGxpbmUgZXhpc3RlZC4KICAgICAgICBkaWFnWyJnX3JhbmdlIl0gPSAoZmxvYXQobnAubmFubWluKHRhYmxlWyJwaG90X2dfbWVhbl9tYWciXSkpLAogICAgICAgICAgICAgICAgICAgICAgICAgICBmbG9hdChucC5uYW5tYXgodGFibGVbInBob3RfZ19tZWFuX21hZyJdKSkpCiAgICAgICAgcmV0dXJuIG5wLmNvbHVtbl9zdGFjayhbeFtpbnNpZGVdLCB5W2luc2lkZV1dKSwgZGlhZwoKICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOgogICAgICAgIGRpYWdbIndoeSJdID0gInt9OiB7fSIuZm9ybWF0KHR5cGUoZXhjKS5fX25hbWVfXywgZXhjKQogICAgICAgIHJldHVybiBucC5lbXB0eSgoMCwgMikpLCBkaWFnCgpkZWYgZGVzY3JpYmVfZm9yZWdyb3VuZChkaWFnKToKICAgICIiIldoYXQgdGhlIGNhdGFsb2d1ZSBsb29rdXAgZm91bmQsIGluIGEgbGluZS4iIiIKICAgIGlmIGRpYWcuZ2V0KCJuX2luX2ZyYW1lIik6CiAgICAgICAgcmV0dXJuICgiICB7fSBmb3JlZ3JvdW5kIHN0YXJzIGZyb20ge30gKHBhcmFsbGF4IG9yIHByb3BlciBtb3Rpb24gIgogICAgICAgICAgICAgICAgInNpZ25pZmljYW50KTsgdGhlIGdhbGF4eSdzIG93biBzdGFycyBhcmUgbm90IGFtb25nIHRoZW0iLmZvcm1hdCgKICAgICAgICAgICAgICAgICAgICBkaWFnWyJuX2luX2ZyYW1lIl0sIGRpYWdbInNvdXJjZSJdKSkKICAgIHJldHVybiAoIiAgbm8gY2F0YWxvZ3VlIHN0YXJzIGF2YWlsYWJsZSAoe30pIC0gZmFsbGluZyBiYWNrIG9uIHBvaW50IHNvdXJjZXMgIgogICAgICAgICAgICAiZm91bmQgaW4gdGhlIGltYWdlLCB3aGljaCBpbiBhIG5lYXJieSBnYWxheHkgaW5jbHVkZXMgaXRzIG93biAiCiAgICAgICAgICAgICJzdGFycyIuZm9ybWF0KGRpYWcuZ2V0KCJ3aHkiKSBvciAidW5rbm93biIpKQoKZGVzY3JpYmVfX2dhbGF4eV9jYXRhbG9ndWUgPSBkZXNjcmliZSAgICMgdGhpcyBtb2R1bGUncyBvd247IHRoZSBwbGFpbiBuYW1lIGdldHMgc2hhZG93ZWQgYmVsb3cKCgojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCiMgc2hhcmVkIG1vZHVsZSBjb250aW51dW1fc3VidHJhY3Rpb24ucHkKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQoKZGVmIF9hcGVydHVyZV9mbHV4ZXMoZnJhbWVzLCBwb3NpdGlvbnMsIHI9Ni4wLCByX2luPTEwLjAsIHJfb3V0PTE2LjApOgogICAgIiIiQmFja2dyb3VuZC1zdWJ0cmFjdGVkIGFwZXJ0dXJlIGZsdXggYXQgdGhlIHNhbWUgcG9zaXRpb25zIGluIHNldmVyYWwgZnJhbWVzLiIiIgogICAgZnJvbSBwaG90dXRpbHMuYXBlcnR1cmUgaW1wb3J0IChDaXJjdWxhckFwZXJ0dXJlLCBDaXJjdWxhckFubnVsdXMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFwZXJ0dXJlX3Bob3RvbWV0cnkpCgogICAgYXAgPSBDaXJjdWxhckFwZXJ0dXJlKHBvc2l0aW9ucywgcj1yKQogICAgYW5uID0gQ2lyY3VsYXJBbm51bHVzKHBvc2l0aW9ucywgcl9pbj1yX2luLCByX291dD1yX291dCkKICAgIG91dCA9IFtdCiAgICBmb3IgaW1nIGluIGZyYW1lczoKICAgICAgICBwaG90ID0gYXBlcnR1cmVfcGhvdG9tZXRyeShpbWcsIFthcCwgYW5uXSkKICAgICAgICBia2dfbWVhbiA9IHBob3RbImFwZXJ0dXJlX3N1bV8xIl0gLyBhbm4uYXJlYQogICAgICAgIG91dC5hcHBlbmQobnAuYXNhcnJheShwaG90WyJhcGVydHVyZV9zdW1fMCJdIC0gYmtnX21lYW4gKiBhcC5hcmVhLCBkdHlwZT1mbG9hdCkpCiAgICByZXR1cm4gb3V0CgpkZWYgZmluZF9yZWZlcmVuY2Vfc3RhcnMoZnJhbWUsIG1hc2s9Tm9uZSwgbl9zdGFycz0zMDAsIHRocmVzaG9sZF9zaWdtYT0yNS4wKToKICAgICIiIkJyaWdodCwgaXNvbGF0ZWQgcG9pbnQgc291cmNlcyB0byBjYWxpYnJhdGUgYWdhaW5zdC4iIiIKICAgIGZyb20gcGhvdHV0aWxzLmRldGVjdGlvbiBpbXBvcnQgREFPU3RhckZpbmRlcgogICAgZnJvbSBhc3Ryb3B5LnN0YXRzIGltcG9ydCBzaWdtYV9jbGlwcGVkX3N0YXRzCgogICAgbWVhbiwgbWVkaWFuLCBzdGQgPSBzaWdtYV9jbGlwcGVkX3N0YXRzKGZyYW1lLCBzaWdtYT0zLjAsIG1hc2s9bWFzaykKICAgIGZpbmRlciA9IERBT1N0YXJGaW5kZXIoZndobT00LjAsIHRocmVzaG9sZD10aHJlc2hvbGRfc2lnbWEgKiBzdGQpCiAgICB0YmwgPSBmaW5kZXIoZnJhbWUgLSBtZWRpYW4sIG1hc2s9bWFzaykKICAgIGlmIHRibCBpcyBOb25lIG9yIGxlbih0YmwpID09IDA6CiAgICAgICAgcmV0dXJuIG5wLmVtcHR5KCgwLCAyKSkKICAgIHRibC5zb3J0KCJmbHV4IikKICAgIHRibCA9IHRibFs6Oi0xXVs6bl9zdGFyc10KICAgIHJldHVybiBucC50cmFuc3Bvc2UoKG5wLmFzYXJyYXkodGJsWyJ4Y2VudHJvaWQiXSwgZHR5cGU9ZmxvYXQpLAogICAgICAgICAgICAgICAgICAgICAgICAgbnAuYXNhcnJheSh0YmxbInljZW50cm9pZCJdLCBkdHlwZT1mbG9hdCkpKQoKZGVmIGNvbnN0YW50X2ZhY3RvcihoYSwgcmVkLCBtYXNrPU5vbmUsIG5fc3RhcnM9MjAwLCBwb3NpdGlvbnM9Tm9uZSk6CiAgICAiIiJPbmUgZmFjdG9yIGZvciB0aGUgd2hvbGUgZnJhbWUsIHRha2VuIGZyb20gdGhlIHN0YXJzLgoKICAgIFJldHVybnMgdGhlIGZhY3RvciB0byBtdWx0aXBseSBILWFscGhhIGJ5IGJlZm9yZSBzdWJ0cmFjdGluZyBSLCBhbmQgdGhlCiAgICBzdGFyLXRvLXN0YXIgc2NhdHRlciwgd2hpY2ggaXMgdGhlIGhvbmVzdCB1bmNlcnRhaW50eSBvbiBpdC4KICAgICIiIgogICAgcG9zID0gKG5wLmFzYXJyYXkocG9zaXRpb25zLCBkdHlwZT1mbG9hdClbOm5fc3RhcnNdCiAgICAgICAgICAgaWYgcG9zaXRpb25zIGlzIG5vdCBOb25lIGFuZCBsZW4ocG9zaXRpb25zKSA+PSAxMAogICAgICAgICAgIGVsc2UgZmluZF9yZWZlcmVuY2Vfc3RhcnMocmVkLCBtYXNrPW1hc2ssIG5fc3RhcnM9bl9zdGFycykpCiAgICBpZiBsZW4ocG9zKSA8IDU6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCJ0b28gZmV3IHN0YXJzIGZvdW5kIHRvIHNldCB0aGUgY29udGludXVtIGZhY3RvciIpCgogICAgZl9oYSwgZl9yID0gX2FwZXJ0dXJlX2ZsdXhlcyhbaGEsIHJlZF0sIHBvcykKICAgIG9rID0gbnAuaXNmaW5pdGUoZl9oYSkgJiBucC5pc2Zpbml0ZShmX3IpICYgKGZfaGEgPiAwKQogICAgaWYgb2suc3VtKCkgPCA1OgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigibm8gdXNhYmxlIHN0YXJzIGZvciB0aGUgY29udGludXVtIGZhY3RvciIpCgogICAgcmF0aW8gPSBmX3Jbb2tdIC8gZl9oYVtva10KICAgIG1lZCA9IGZsb2F0KG5wLm1lZGlhbihyYXRpbykpCiAgICBzcHJlYWQgPSAxLjQ4MjYgKiBucC5tZWRpYW4obnAuYWJzKHJhdGlvIC0gbWVkKSkKICAgIGlmIHNwcmVhZCA+IDA6CiAgICAgICAgcmF0aW8gPSByYXRpb1tucC5hYnMocmF0aW8gLSBtZWQpIDwgMy4wICogc3ByZWFkXQoKICAgIGZhY3RvciA9IGZsb2F0KG5wLm1lZGlhbihyYXRpbykpCiAgICBzY2F0dGVyID0gZmxvYXQoMS40ODI2ICogbnAubWVkaWFuKG5wLmFicyhyYXRpbyAtIGZhY3RvcikpKQogICAgcmV0dXJuIGZhY3RvciwgewogICAgICAgICJuX3N0YXJzIjogaW50KHJhdGlvLnNpemUpLAogICAgICAgICJzY2F0dGVyIjogc2NhdHRlciwKICAgICAgICAicmVsYXRpdmVfc2NhdHRlciI6IHNjYXR0ZXIgLyBmYWN0b3IgaWYgZmFjdG9yIGVsc2UgZmxvYXQoIm5hbiIpLAogICAgICAgICJtZXRob2QiOiAib25lIGNvbnN0YW50IGZvciB0aGUgd2hvbGUgZnJhbWUiLAogICAgfQoKZGVmIGNvbG91cl9tYXAoYiwgdiwgc21vb3RoX3B4PTYuMCwgbWluX3NpZ25hbD1Ob25lKToKICAgICIiIkFuIGluc3RydW1lbnRhbCBCIG1pbnVzIFYgdmFsdWUgZm9yIGV2ZXJ5IHBpeGVsLCBzbW9vdGhlZCB0byBiZSB1c2FibGUuCgogICAgQ29sb3VyIGlzIGEgcmF0aW8gb2YgdHdvIGZhaW50IHRoaW5ncywgc28gaXQgaXMgZmFyIG5vaXNpZXIgdGhhbiBlaXRoZXIgZnJhbWUKICAgIG9uIGl0cyBvd24uIEJvdGggZnJhbWVzIGFyZSBzbW9vdGhlZCBiZWZvcmUgdGhlIHJhdGlvIGlzIHRha2VuIC0gdGhlIGNvbG91ciBvZgogICAgYSBzdGFyLWZvcm1pbmcgY29tcGxleCB2YXJpZXMgb24gc2NhbGVzIG11Y2ggbGFyZ2VyIHRoYW4gYSBwaXhlbCwgc28gdGhpcyBjb3N0cwogICAgbm90aGluZyByZWFsIGFuZCBtYWtlcyB0aGUgbWFwIHVzYWJsZS4KCiAgICBSZXR1cm5zIHRoZSBjb2xvdXIgbWFwIGFuZCBhIGJvb2xlYW4gYXJyYXkgbWFya2luZyB3aGVyZSBpdCBjYW4gYmUgdHJ1c3RlZC4KICAgICIiIgogICAgZnJvbSBzY2lweS5uZGltYWdlIGltcG9ydCBnYXVzc2lhbl9maWx0ZXIKICAgIGZyb20gYXN0cm9weS5zdGF0cyBpbXBvcnQgc2lnbWFfY2xpcHBlZF9zdGF0cwoKICAgIGJzID0gZ2F1c3NpYW5fZmlsdGVyKG5wLm5hbl90b19udW0oYiwgbmFuPTAuMCksIHNtb290aF9weCkKICAgIHZzID0gZ2F1c3NpYW5fZmlsdGVyKG5wLm5hbl90b19udW0odiwgbmFuPTAuMCksIHNtb290aF9weCkKCiAgICBpZiBtaW5fc2lnbmFsIGlzIE5vbmU6CiAgICAgICAgXywgXywgc3RkX3YgPSBzaWdtYV9jbGlwcGVkX3N0YXRzKHYsIHNpZ21hPTMuMCkKICAgICAgICBtaW5fc2lnbmFsID0gMy4wICogc3RkX3YgLyBtYXgoc21vb3RoX3B4LCAxLjApCgogICAgdXNhYmxlID0gKGJzID4gbWluX3NpZ25hbCkgJiAodnMgPiBtaW5fc2lnbmFsKSAmIG5wLmlzZmluaXRlKGJzKSAmIG5wLmlzZmluaXRlKHZzKQogICAgd2l0aCBucC5lcnJzdGF0ZShkaXZpZGU9Imlnbm9yZSIsIGludmFsaWQ9Imlnbm9yZSIpOgogICAgICAgIGNvbG91ciA9IG5wLndoZXJlKHVzYWJsZSwgLTIuNSAqIG5wLmxvZzEwKG5wLndoZXJlKHVzYWJsZSwgYnMgLyB2cywgMS4wKSksIG5wLm5hbikKICAgIHJldHVybiBjb2xvdXIsIHVzYWJsZQoKZGVmIGNvbG91cl9kZXBlbmRlbnRfZmFjdG9yKGhhLCByZWQsIGIsIHYsIG1hc2s9Tm9uZSwgbl9zdGFycz0zMDAsIHNtb290aF9weD02LjAsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBwb3NpdGlvbnM9Tm9uZSk6CiAgICAiIiJBIGZhY3RvciB0aGF0IHZhcmllcyBhY3Jvc3MgdGhlIGZyYW1lIHdpdGggdGhlIGNvbG91ciBvZiB3aGF0IGlzIHRoZXJlLgoKICAgIEZvbGxvd3MgdGhlIHJlY29tbWVuZGF0aW9uIG9mIFNwZWN0b3IsIEZpbmtlbG1hbiBhbmQgQnJvc2NoICgyMDEyKTogbWVhc3VyZSB0aGUKICAgIGZhY3RvciBzdGFyIGJ5IHN0YXIsIG1lYXN1cmUgZWFjaCBzdGFyJ3MgY29sb3VyLCBmaXQgYSBzdHJhaWdodCBsaW5lIHRocm91Z2gKICAgIHRoZW0sIGFuZCB0aGVuIHJlYWQgdGhlIGZhY3RvciBvZmYgdGhhdCBsaW5lIGZvciBldmVyeSBwYXJ0IG9mIHRoZSBpbWFnZQogICAgYWNjb3JkaW5nIHRvIGl0cyBvd24gY29sb3VyLgoKICAgIFdIRVJFIElUIEZBTExTIEJBQ0sKICAgIFRoZSBjb2xvdXIgb2YgYSBwaXhlbCBpcyBvbmx5IG1lYXN1cmFibGUgd2hlcmUgYm90aCBicm9hZCBiYW5kcyBoYXZlIHNpZ25hbC4KICAgIEluIHRoZSBlbXB0eSBza3kgYmV0d2VlbiB0aGUgYXJtcyB0aGV5IGRvIG5vdCwgYW5kIHRoZXJlIHRoZSBjb25zdGFudCBmYWN0b3IgaXMKICAgIHVzZWQgaW5zdGVhZC4gVGhlIGRpYWdub3N0aWNzIHJlcG9ydCB3aGF0IGZyYWN0aW9uIG9mIHRoZSBmcmFtZSBnb3Qgd2hpY2gsIHNvIGEKICAgIHJ1biB3aGVyZSB0aGUgY29sb3VyIG1hcCBiYXJlbHkgYXBwbGllZCBkb2VzIG5vdCBsb29rIGxpa2Ugb25lIHdoZXJlIGl0IGRpZC4KCiAgICBSZXR1cm5zIChmYWN0b3JfbWFwLCBkaWFnbm9zdGljcykuIGZhY3Rvcl9tYXAgaGFzIHRoZSBzYW1lIHNoYXBlIGFzIHRoZSBmcmFtZXMuCiAgICAiIiIKICAgICMgQ2F0YWxvZ3VlIHBvc2l0aW9ucyB3aGVuIHRoZXkgYXJlIGF2YWlsYWJsZTogdGhlIHJlbGF0aW9uIGJlaW5nIGZpdHRlZCBpcwogICAgIyB0aGUgb25lIGZvcmVncm91bmQgc3RhcnMgZm9sbG93LCBhbmQgYSBuZWFyYnkgZ2FsYXh5J3Mgb3duIHN0YXJzIHNpdHRpbmcgaW4KICAgICMgdGhlIHNhbXBsZSBhcmUgd2hhdCBtYWtlcyB0aGUgZml0IGZhaWwgcmF0aGVyIHRoYW4gbWVyZWx5IHNjYXR0ZXIuCiAgICBwb3MgPSAobnAuYXNhcnJheShwb3NpdGlvbnMsIGR0eXBlPWZsb2F0KVs6bl9zdGFyc10KICAgICAgICAgICBpZiBwb3NpdGlvbnMgaXMgbm90IE5vbmUgYW5kIGxlbihwb3NpdGlvbnMpID49IDEwCiAgICAgICAgICAgZWxzZSBmaW5kX3JlZmVyZW5jZV9zdGFycyh2LCBtYXNrPW1hc2ssIG5fc3RhcnM9bl9zdGFycykpCiAgICBpZiBsZW4ocG9zKSA8IDg6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCJ0b28gZmV3IHN0YXJzIHRvIGZpdCBhIGNvbG91ciByZWxhdGlvbiIpCgogICAgZl9oYSwgZl9yLCBmX2IsIGZfdiA9IF9hcGVydHVyZV9mbHV4ZXMoW2hhLCByZWQsIGIsIHZdLCBwb3MpCiAgICBvayA9IChucC5pc2Zpbml0ZShmX2hhKSAmIG5wLmlzZmluaXRlKGZfcikgJiBucC5pc2Zpbml0ZShmX2IpICYgbnAuaXNmaW5pdGUoZl92KQogICAgICAgICAgJiAoZl9oYSA+IDApICYgKGZfYiA+IDApICYgKGZfdiA+IDApKQogICAgaWYgb2suc3VtKCkgPCA4OgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigidG9vIGZldyBzdGFycyB3aXRoIGdvb2QgZmx1eCBpbiBhbGwgZm91ciBmcmFtZXMiKQoKICAgIHJhdGlvID0gZl9yW29rXSAvIGZfaGFbb2tdCiAgICBjb2xvdXIgPSAtMi41ICogbnAubG9nMTAoZl9iW29rXSAvIGZfdltva10pCgogICAgIyBDbGlwIG9idmlvdXMgb3V0bGllcnMgaW4gYm90aCBkaXJlY3Rpb25zIGJlZm9yZSBmaXR0aW5nOiBzYXR1cmF0ZWQgc3RhcnMsIGFuZAogICAgIyBhbnkgc3RhciB0aGF0IGRvZXMgaGF2ZSBsaW5lIGVtaXNzaW9uIGFuZCBzbyBpcyBub3QgcHVyZSBjb250aW51dW0uCiAgICBnb29kID0gbnAuaXNmaW5pdGUocmF0aW8pICYgbnAuaXNmaW5pdGUoY29sb3VyKQogICAgZm9yIF8gaW4gcmFuZ2UoMyk6CiAgICAgICAgaWYgZ29vZC5zdW0oKSA8IDY6CiAgICAgICAgICAgIGJyZWFrCiAgICAgICAgbWVkX3IgPSBucC5tZWRpYW4ocmF0aW9bZ29vZF0pCiAgICAgICAgc3ByZWFkID0gMS40ODI2ICogbnAubWVkaWFuKG5wLmFicyhyYXRpb1tnb29kXSAtIG1lZF9yKSkKICAgICAgICBpZiBzcHJlYWQgPD0gMDoKICAgICAgICAgICAgYnJlYWsKICAgICAgICBnb29kICY9IG5wLmFicyhyYXRpbyAtIG1lZF9yKSA8IDMuMCAqIHNwcmVhZAoKICAgIGlmIGdvb2Quc3VtKCkgPCA2OgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigibm90IGVub3VnaCBjbGVhbiBzdGFycyBsZWZ0IHRvIGZpdCB0aGUgY29sb3VyIHJlbGF0aW9uIikKCiAgICBzbG9wZSwgaW50ZXJjZXB0ID0gbnAucG9seWZpdChjb2xvdXJbZ29vZF0sIHJhdGlvW2dvb2RdLCAxKQogICAgcHJlZGljdGVkID0gbnAucG9seXZhbChbc2xvcGUsIGludGVyY2VwdF0sIGNvbG91cltnb29kXSkKICAgIHJlc2lkID0gcmF0aW9bZ29vZF0gLSBwcmVkaWN0ZWQKICAgIHNzX3RvdCA9IGZsb2F0KG5wLnN1bSgocmF0aW9bZ29vZF0gLSByYXRpb1tnb29kXS5tZWFuKCkpICoqIDIpKQogICAgcjIgPSBmbG9hdCgxLjAgLSBucC5zdW0ocmVzaWQgKiogMikgLyBzc190b3QpIGlmIHNzX3RvdCA+IDAgZWxzZSBmbG9hdCgibmFuIikKCiAgICBjb25zdCA9IGZsb2F0KG5wLm1lZGlhbihyYXRpb1tnb29kXSkpCiAgICBjb25zdF9zY2F0dGVyID0gZmxvYXQoMS40ODI2ICogbnAubWVkaWFuKG5wLmFicyhyYXRpb1tnb29kXSAtIGNvbnN0KSkpCiAgICBmaXRfc2NhdHRlciA9IGZsb2F0KDEuNDgyNiAqIG5wLm1lZGlhbihucC5hYnMocmVzaWQpKSkKCiAgICBjX2xvLCBjX2hpID0gbnAucGVyY2VudGlsZShjb2xvdXJbZ29vZF0sIFsyLCA5OF0pCgogICAgIyBPbmx5IHVzZSB0aGUgZml0dGVkIHJlbGF0aW9uIGlmIGl0IGFjdHVhbGx5IGRlc2NyaWJlcyB0aGUgc3RhcnMgYmV0dGVyIHRoYW4gYQogICAgIyBzaW5nbGUgbnVtYmVyIGRvZXMuIEZpdHRpbmcgYSBzbG9wZSB0aHJvdWdoIGEgaGFuZGZ1bCBvZiBzdGFycyBzcGFubmluZyBhCiAgICAjIG5hcnJvdyByYW5nZSBvZiBjb2xvdXIgY2FuIGVhc2lseSBjb21lIG91dCB3b3JzZSB0aGFuIHRoZSBjb25zdGFudCBpdCB3YXMKICAgICMgbWVhbnQgdG8gaW1wcm92ZSBvbiwgYW5kIGEgbWV0aG9kIHRoYXQgaXMgYmV0dGVyIGluIHByaW5jaXBsZSBidXQgbWVhc3VyYWJseQogICAgIyB3b3JzZSBvbiB0aGlzIGRhdGEgc2hvdWxkIG5vdCBiZSB0aGUgb25lIHRoYXQgcnVucy4KICAgIGltcHJvdmVzID0gKGZpdF9zY2F0dGVyIDwgY29uc3Rfc2NhdHRlcikgYW5kIChyMiA+PSAwLjMwKSBhbmQgKGdvb2Quc3VtKCkgPj0gMTIpCiAgICByZWFzb24gPSAiIgogICAgaWYgbm90IGltcHJvdmVzOgogICAgICAgIGJpdHMgPSBbXQogICAgICAgIGlmIGdvb2Quc3VtKCkgPCAxMjoKICAgICAgICAgICAgYml0cy5hcHBlbmQoIm9ubHkge30gY2xlYW4gc3RhcnMiLmZvcm1hdChpbnQoZ29vZC5zdW0oKSkpKQogICAgICAgIGlmIHIyIDwgMC4zMDoKICAgICAgICAgICAgYml0cy5hcHBlbmQoInRoZSBmaXQgZXhwbGFpbnMgbGl0dGxlIG9mIHRoZSBzY2F0dGVyIChSMiA9IHs6LjJmfSkiLmZvcm1hdChyMikpCiAgICAgICAgaWYgZml0X3NjYXR0ZXIgPj0gY29uc3Rfc2NhdHRlcjoKICAgICAgICAgICAgYml0cy5hcHBlbmQoInRoZSBmaXR0ZWQgbGluZSBzY2F0dGVycyBtb3JlIHRoYW4gYSBjb25zdGFudCAiCiAgICAgICAgICAgICAgICAgICAgICAgICIoezouNGZ9IGFnYWluc3QgezouNGZ9KSIuZm9ybWF0KGZpdF9zY2F0dGVyLCBjb25zdF9zY2F0dGVyKSkKICAgICAgICBpZiBjX2hpIC0gY19sbyA8IDAuNToKICAgICAgICAgICAgYml0cy5hcHBlbmQoInRoZSBzdGFycyBzcGFuIG9ubHkgezouMmZ9IG1hZyBvZiBjb2xvdXIiLmZvcm1hdChjX2hpIC0gY19sbykpCiAgICAgICAgcmVhc29uID0gIjsgIi5qb2luKGJpdHMpCgogICAgY21hcCwgdXNhYmxlID0gY29sb3VyX21hcChiLCB2LCBzbW9vdGhfcHg9c21vb3RoX3B4KQogICAgY19jbGlwcGVkID0gbnAuY2xpcChjbWFwLCBjX2xvLCBjX2hpKQoKICAgIGZhY3Rvcl9tYXAgPSBucC5mdWxsKGhhLnNoYXBlLCBjb25zdCwgZHR5cGU9ZmxvYXQpCiAgICBhcHBsaWVkID0gbnAuemVyb3MoaGEuc2hhcGUsIGR0eXBlPWJvb2wpCiAgICBpZiBpbXByb3ZlczoKICAgICAgICAjIE5ldmVyIGV4dHJhcG9sYXRlIHRoZSByZWxhdGlvbiBiZXlvbmQgdGhlIGNvbG91cnMgdGhlIHN0YXJzIGFjdHVhbGx5CiAgICAgICAgIyBzcGFubmVkOyBvdXRzaWRlIHRoYXQgcmFuZ2UgdGhlIHN0cmFpZ2h0IGxpbmUgaXMgYSBndWVzcywgbm90IGEgbWVhc3VyZW1lbnQuCiAgICAgICAgYXBwbGllZCA9IHVzYWJsZSAmIG5wLmlzZmluaXRlKGNfY2xpcHBlZCkKICAgICAgICBmYWN0b3JfbWFwW2FwcGxpZWRdID0gbnAucG9seXZhbChbc2xvcGUsIGludGVyY2VwdF0sIGNfY2xpcHBlZFthcHBsaWVkXSkKCiAgICByZXR1cm4gZmFjdG9yX21hcCwgewogICAgICAgICJtZXRob2QiOiAoImZhY3RvciBmaXR0ZWQgYWdhaW5zdCBjb2xvdXIsIHBlciBTcGVjdG9yIGV0IGFsLiAyMDEyIgogICAgICAgICAgICAgICAgICAgaWYgaW1wcm92ZXMgZWxzZQogICAgICAgICAgICAgICAgICAgIm9uZSBjb25zdGFudCAtIHRoZSBjb2xvdXIgZml0IHdhcyB0cmllZCBhbmQgcmVqZWN0ZWQiKSwKICAgICAgICAidXNlZF9jb2xvdXIiOiBib29sKGltcHJvdmVzKSwKICAgICAgICAicmVqZWN0ZWRfYmVjYXVzZSI6IHJlYXNvbiwKICAgICAgICAibl9zdGFycyI6IGludChnb29kLnN1bSgpKSwKICAgICAgICAic2xvcGUiOiBmbG9hdChzbG9wZSksCiAgICAgICAgImludGVyY2VwdCI6IGZsb2F0KGludGVyY2VwdCksCiAgICAgICAgImZpdF9yMiI6IHIyLAogICAgICAgICJjb2xvdXJfcmFuZ2UiOiAoZmxvYXQoY19sbyksIGZsb2F0KGNfaGkpKSwKICAgICAgICAiY29uc3RhbnRfZmFjdG9yIjogY29uc3QsCiAgICAgICAgImNvbnN0YW50X3NjYXR0ZXIiOiBjb25zdF9zY2F0dGVyLAogICAgICAgICJyZWxhdGl2ZV9zY2F0dGVyIjogY29uc3Rfc2NhdHRlciAvIGNvbnN0IGlmIGNvbnN0IGVsc2UgZmxvYXQoIm5hbiIpLAogICAgICAgICJmaXRfc2NhdHRlciI6IGZpdF9zY2F0dGVyLAogICAgICAgICJzY2F0dGVyX3JlZHVjdGlvbiI6IChjb25zdF9zY2F0dGVyIC8gZml0X3NjYXR0ZXIpIGlmIGZpdF9zY2F0dGVyID4gMCBlbHNlIGZsb2F0KCJuYW4iKSwKICAgICAgICAiZnJhY3Rpb25fY29sb3VyX2FwcGxpZWQiOiBmbG9hdChhcHBsaWVkLm1lYW4oKSksCiAgICAgICAgImZhY3Rvcl9taW4iOiBmbG9hdChucC5uYW5taW4oZmFjdG9yX21hcCkpLAogICAgICAgICJmYWN0b3JfbWF4IjogZmxvYXQobnAubmFubWF4KGZhY3Rvcl9tYXApKSwKICAgIH0KCmRlZiBkZXNjcmliZShkaWFnKToKICAgICIiIkEgcmVhZGFibGUgc3VtbWFyeSBvZiB3aGljaGV2ZXIgbWV0aG9kIHdhcyB1c2VkLiIiIgogICAgbGluZXMgPSBbIiAgbWV0aG9kOiAiICsgZGlhZy5nZXQoIm1ldGhvZCIsICI/IiksCiAgICAgICAgICAgICAiICBzdGFycyB1c2VkOiB7fSIuZm9ybWF0KGRpYWcuZ2V0KCJuX3N0YXJzIikpXQogICAgaWYgZGlhZy5nZXQoInJlamVjdGVkX2JlY2F1c2UiKToKICAgICAgICBsaW5lcy5hcHBlbmQoIiAgdGhlIGNvbG91ciBmaXQgd2FzIHJlamVjdGVkIGJlY2F1c2UgIiArIGRpYWdbInJlamVjdGVkX2JlY2F1c2UiXSkKICAgICAgICBsaW5lcy5hcHBlbmQoIiAgZmFsbGluZyBiYWNrIHRvIGEgY29uc3RhbnQgZmFjdG9yIG9mIHs6LjNmfSIuZm9ybWF0KAogICAgICAgICAgICBkaWFnWyJjb25zdGFudF9mYWN0b3IiXSkpCiAgICBpZiAic2xvcGUiIGluIGRpYWc6CiAgICAgICAgbGluZXMgKz0gWwogICAgICAgICAgICAiICBmYWN0b3IgPSB7Oi40Zn0gKyB7Oi40Zn0geCAoQi1WIGluc3RydW1lbnRhbCksICBSMiA9IHs6LjJmfSIuZm9ybWF0KAogICAgICAgICAgICAgICAgZGlhZ1siaW50ZXJjZXB0Il0sIGRpYWdbInNsb3BlIl0sIGRpYWdbImZpdF9yMiJdKSwKICAgICAgICAgICAgIiAgY29sb3VycyBzcGFubmVkIGJ5IHRoZSBzdGFyczogezouMmZ9IHRvIHs6LjJmfSIuZm9ybWF0KCpkaWFnWyJjb2xvdXJfcmFuZ2UiXSksCiAgICAgICAgICAgICIgIGZhY3RvciB2YXJpZXMgYWNyb3NzIHRoZSBmcmFtZSBmcm9tIHs6LjNmfSB0byB7Oi4zZn0iLmZvcm1hdCgKICAgICAgICAgICAgICAgIGRpYWdbImZhY3Rvcl9taW4iXSwgZGlhZ1siZmFjdG9yX21heCJdKSwKICAgICAgICAgICAgIiAgYSBzaW5nbGUgY29uc3RhbnQgd291bGQgaGF2ZSBiZWVuIHs6LjNmfSIuZm9ybWF0KGRpYWdbImNvbnN0YW50X2ZhY3RvciJdKSwKICAgICAgICAgICAgIiAgc2NhdHRlciBhYm91dCBhIGNvbnN0YW50IHs6LjRmfSwgYWJvdXQgdGhlIGZpdHRlZCBsaW5lIHs6LjRmfSAiCiAgICAgICAgICAgICItIHRoZSBsaW5lIGlzIHs6LjAlfSB7fSIuZm9ybWF0KAogICAgICAgICAgICAgICAgZGlhZ1siY29uc3RhbnRfc2NhdHRlciJdLCBkaWFnWyJmaXRfc2NhdHRlciJdLAogICAgICAgICAgICAgICAgYWJzKGRpYWdbImZpdF9zY2F0dGVyIl0gLyBkaWFnWyJjb25zdGFudF9zY2F0dGVyIl0gLSAxLjApCiAgICAgICAgICAgICAgICBpZiBkaWFnWyJjb25zdGFudF9zY2F0dGVyIl0gZWxzZSBmbG9hdCgibmFuIiksCiAgICAgICAgICAgICAgICAidGlnaHRlciIgaWYgZGlhZ1siZml0X3NjYXR0ZXIiXSA8IGRpYWdbImNvbnN0YW50X3NjYXR0ZXIiXSBlbHNlICJ3aWRlciIpLAogICAgICAgICAgICAiICB0aGUgY29sb3VyIHJlbGF0aW9uIHdhcyBhcHBsaWVkIHRvIHs6LjAlfSBvZiB0aGUgZnJhbWU7IHRoZSByZXN0IGZlbGwgIgogICAgICAgICAgICAiYmFjayB0byB0aGUgY29uc3RhbnQiLmZvcm1hdChkaWFnWyJmcmFjdGlvbl9jb2xvdXJfYXBwbGllZCJdKSwKICAgICAgICBdCiAgICBlbHNlOgogICAgICAgIGxpbmVzLmFwcGVuZCgiICBmYWN0b3IgezouNGZ9LCBzdGFyLXRvLXN0YXIgc2NhdHRlciB7Oi4xJX0iLmZvcm1hdCgKICAgICAgICAgICAgZGlhZy5nZXQoImNvbnN0YW50X2ZhY3RvciIsIGZsb2F0KCJuYW4iKSksCiAgICAgICAgICAgIGRpYWcuZ2V0KCJyZWxhdGl2ZV9zY2F0dGVyIiwgZmxvYXQoIm5hbiIpKSkpCiAgICByZXR1cm4gIlxuIi5qb2luKGxpbmVzKQoKZGVzY3JpYmVfX2NvbnRpbnV1bV9zdWJ0cmFjdGlvbiA9IGRlc2NyaWJlICAgIyB0aGlzIG1vZHVsZSdzIG93bjsgdGhlIHBsYWluIG5hbWUgZ2V0cyBzaGFkb3dlZCBiZWxvdwoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KIyBzaGFyZWQgbW9kdWxlIHJhZGlhbF9wcm9maWxlcy5weQojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCgptYXRwbG90bGliLnVzZSgiQWdnIikKCnBhc3MgICMgZnJvbSBkZXByb2plY3Rpb24sIGlubGluZWQgYWJvdmUKCkNPTE9VUlMgPSBbIiM0NDQ0NDQiLCAiI2EzMjgzYyIsICIjMmI1YTg5IiwgIiMzZjdkNTIiLCAiIzhhNWEyYiJdCgpkZWYgdHJlYXRtZW50X25vbmUoKToKICAgIHJldHVybiB7ImtleSI6ICJub25lIiwgImxhYmVsIjogIkFzIG1lYXN1cmVkLCBubyB0aWx0IGNvcnJlY3Rpb24iLCAiYW5nbGUiOiBOb25lLCAiaW5jIjogTm9uZSwKICAgICAgICAgICAgIm5vdGUiOiAiUG9zaXRpb25zIGV4YWN0bHkgYXMgbWVhc3VyZWQgb24gdGhlIGltYWdlLiBDb250cm9sIHBhbmVsLiJ9CgpkZWYgdHJlYXRtZW50X2NhdGFsb2d1ZShnZW9tLCBub3J0aF9hbmdsZT05MC4wLCBtaXJyb3JlZD1GYWxzZSk6CiAgICAiIiJEZXByb2plY3Rpb24gdXNpbmcgdGhlIHBvc2l0aW9uIGFuZ2xlIGFuZCBheGlzIHJhdGlvIGZyb20gSHlwZXJMRURBLiIiIgogICAgaWYgZ2VvbSBpcyBOb25lOgogICAgICAgIHJldHVybiBOb25lCiAgICBwYV9za3ksIGluYyA9IGdlb20uZ2V0KCJwYV9za3kiKSwgZ2VvbS5nZXQoImluYyIpCiAgICBpZiBwYV9za3kgaXMgTm9uZToKICAgICAgICByZXR1cm4geyJrZXkiOiAibGVkYSIsICJsYWJlbCI6ICJUaWx0IGNvcnJlY3Rpb24gbm90IHBvc3NpYmxlIiwgImFuZ2xlIjogTm9uZSwgImluYyI6IE5vbmUsCiAgICAgICAgICAgICAgICAibm90ZSI6ICJIeXBlckxFREEgcHVibGlzaGVzIG5vIHBvc2l0aW9uIGFuZ2xlIGZvciB0aGlzIGdhbGF4eSAtIGl0IGlzICIKICAgICAgICAgICAgICAgICAgICAgICAgInRvbyByb3VuZCBmb3IgYSBtYWpvciBheGlzIHRvIGJlIGRlZmluZWQuIE5vdCBjb3JyZWN0ZWQuIn0KICAgIG9rLCByZWFzb24gPSBpbmNsaW5hdGlvbl9pc191c2FibGUoaW5jKQogICAgaWYgbm90IG9rOgogICAgICAgIHJldHVybiB7ImtleSI6ICJsZWRhIiwgImxhYmVsIjogIlRpbHQgY29ycmVjdGlvbiBub3QgcG9zc2libGUiLCAiYW5nbGUiOiBOb25lLAogICAgICAgICAgICAgICAgImluYyI6IE5vbmUsICJub3RlIjogcmVhc29ufQogICAgYW5nID0gc2t5X3BhX3RvX2ltYWdlX2FuZ2xlKHBhX3NreSwgbm9ydGhfYW5nbGUsIG1pcnJvcmVkKQogICAgcmV0dXJuIHsKICAgICAgICAia2V5IjogImxlZGEiLCAibGFiZWwiOiAiQ29ycmVjdGVkIGZvciBhIHRpbHQgb2YgezouMGZ9IGRlZyAoSHlwZXJMRURBKSIuZm9ybWF0KGluYyksCiAgICAgICAgImFuZ2xlIjogYW5nLCAiaW5jIjogaW5jLAogICAgICAgICJub3RlIjogKCJQQSB7Oi4xZn0gZGVnIG9uIHNreSAtPiB7Oi4xZn0gZGVnIGluIGltYWdlIChub3J0aCBhdCB7Oi4wZn0gZGVne30pLiAiCiAgICAgICAgICAgICAgICAgIlN0cmV0Y2ggezouMmZ9eCBhbG9uZyB0aGUgbWlub3IgYXhpcy4iKS5mb3JtYXQoCiAgICAgICAgICAgICAgICAgICAgIHBhX3NreSwgYW5nLCBub3J0aF9hbmdsZSwgIiwgbWlycm9yZWQiIGlmIG1pcnJvcmVkIGVsc2UgIiIsCiAgICAgICAgICAgICAgICAgICAgIHN0cmV0Y2hfZmFjdG9yKGluYykpLAogICAgfQoKZGVmIHRyZWF0bWVudF9tYW51YWwoaW5jLCBwYV9pbWFnZT1Ob25lLCBwYV9za3k9Tm9uZSwgbm9ydGhfYW5nbGU9OTAuMCwgbWlycm9yZWQ9RmFsc2UpOgogICAgIiIiRGVwcm9qZWN0aW9uIHVzaW5nIGFuZ2xlcyB0aGUgdXNlciBzdXBwbGllZCwgZnJvbSB0aGUgbGl0ZXJhdHVyZSBvciBieSBleWUuIiIiCiAgICBpZiBpbmMgaXMgTm9uZToKICAgICAgICByZXR1cm4gTm9uZQogICAgb2ssIHJlYXNvbiA9IGluY2xpbmF0aW9uX2lzX3VzYWJsZShpbmMpCiAgICBpZiBwYV9pbWFnZSBpcyBub3QgTm9uZToKICAgICAgICBhbmcsIHNyYyA9IHBhX2ltYWdlICUgMTgwLjAsICJnaXZlbiBkaXJlY3RseSBpbiBpbWFnZSBjb29yZGluYXRlcyIKICAgIGVsaWYgcGFfc2t5IGlzIG5vdCBOb25lOgogICAgICAgIGFuZyA9IHNreV9wYV90b19pbWFnZV9hbmdsZShwYV9za3ksIG5vcnRoX2FuZ2xlLCBtaXJyb3JlZCkKICAgICAgICBzcmMgPSAiZnJvbSBza3kgUEEgezouMWZ9IGRlZyIuZm9ybWF0KHBhX3NreSkKICAgIGVsc2U6CiAgICAgICAgcmV0dXJuIHsia2V5IjogIm1hbnVhbCIsICJsYWJlbCI6ICJUaWx0IGNvcnJlY3Rpb24gaW5jb21wbGV0ZSIsICJhbmdsZSI6IE5vbmUsCiAgICAgICAgICAgICAgICAiaW5jIjogTm9uZSwgIm5vdGUiOiAiQW4gaW5jbGluYXRpb24gd2FzIGdpdmVuIGJ1dCBubyBwb3NpdGlvbiBhbmdsZS4ifQogICAgaWYgbm90IG9rOgogICAgICAgIHJldHVybiB7ImtleSI6ICJtYW51YWwiLCAibGFiZWwiOiAiVGlsdCBjb3JyZWN0aW9uIG5vdCBwb3NzaWJsZSIsICJhbmdsZSI6IE5vbmUsCiAgICAgICAgICAgICAgICAiaW5jIjogTm9uZSwgIm5vdGUiOiByZWFzb259CiAgICByZXR1cm4geyJrZXkiOiAibWFudWFsIiwgImxhYmVsIjogIkNvcnJlY3RlZCBmb3IgYSB0aWx0IG9mIHs6LjBmfSBkZWcgKHlvdXJzKSIuZm9ybWF0KGluYyksCiAgICAgICAgICAgICJhbmdsZSI6IGFuZywgImluYyI6IGluYywKICAgICAgICAgICAgIm5vdGUiOiAoIlVzZXItc3VwcGxpZWQgZ2VvbWV0cnksIHt9LiBNYWpvciBheGlzIGF0IHs6LjFmfSBkZWcgaW4gdGhlICIKICAgICAgICAgICAgICAgICAgICAgImltYWdlLiBTdHJldGNoIHs6LjJmfXguIikuZm9ybWF0KHNyYywgYW5nLCBzdHJldGNoX2ZhY3RvcihpbmMpKX0KCmRlZiB0cmVhdG1lbnRfc2VjdG9yKGR4LCBkeSwgbl9zZWN0b3JzPTIwLCBzdGVwPTUuMCk6CiAgICAiIiJEZXByb2plY3Rpb24gZGVyaXZlZCBmcm9tIHRoZSBvYmplY3RzIHRoZW1zZWx2ZXMsIHdpdGggaXRzIGNhdmVhdHMgYXR0YWNoZWQuIiIiCiAgICBmaXQgPSBmaXRfc2VjdG9yX21ldGhvZChkeCwgZHksIG5fc2VjdG9ycz1uX3NlY3RvcnMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBhbmdsZV9zdGVwPXN0ZXAsIGluY19zdGVwPXN0ZXApCiAgICBvaywgcmVhc29uID0gaW5jbGluYXRpb25faXNfdXNhYmxlKGZpdFsiaW5jIl0pCgogICAgd2FybiA9ICIiCiAgICBpZiBmaXRbImZld19vYmplY3RzIl06CiAgICAgICAgd2FybiArPSAoIiBPbmx5IHt9IG9iamVjdHMgLSBiZWxvdyB0aGUgMTUwIHRvIDIwMCBuZWVkZWQgZm9yIGEgc2luZ2xlIGNsZWFuICIKICAgICAgICAgICAgICAgICAibWluaW11bSwgc28gdHJlYXQgdGhpcyB3aXRoIHN1c3BpY2lvbi4iLmZvcm1hdChmaXRbIm5fb2JqZWN0cyJdKSkKICAgIGlmIGZpdFsicnVubmVyc191cCJdOgogICAgICAgIGEsIGMsIGQgPSBmaXRbInJ1bm5lcnNfdXAiXVswXQogICAgICAgIG1hcmdpbiA9IChkIC0gZml0WyJkaXNwZXJzaW9uIl0pIC8gbWF4KGZpdFsiZGlzcGVyc2lvbiJdLCAxZS05KQogICAgICAgIHdhcm4gKz0gKCIgQ2xvc2VzdCByaXZhbCBzb2x1dGlvbjogUEEgezouMGZ9IGRlZywgaSB7Oi4wZn0gZGVnLCB3b3JzZSBieSBvbmx5ICIKICAgICAgICAgICAgICAgICAiezouMCV9LiIuZm9ybWF0KGEsIGMsIG1hcmdpbikpCgogICAgaWYgbm90IG9rOgogICAgICAgIHJldHVybiB7ImtleSI6ICJzZWN0b3IiLCAibGFiZWwiOiAiVGlsdCBmcm9tIHRoZSBkYXRhIC0gbm90IHBvc3NpYmxlIiwgImFuZ2xlIjogTm9uZSwKICAgICAgICAgICAgICAgICJpbmMiOiBOb25lLAogICAgICAgICAgICAgICAgIm5vdGUiOiAoIlNlY3RvciBtZXRob2QgZm91bmQgUEEgezouMGZ9IGRlZywgaSB7Oi4wZn0gZGVnLCBidXQgIgogICAgICAgICAgICAgICAgICAgICAgICAgLmZvcm1hdChmaXRbImFuZ2xlIl0sIGZpdFsiaW5jIl0pICsgcmVhc29uICsgd2Fybil9CiAgICByZXR1cm4geyJrZXkiOiAic2VjdG9yIiwgImxhYmVsIjogIkNvcnJlY3RlZCBmb3IgYSB0aWx0IG9mIHs6LjBmfSBkZWcgKG1lYXN1cmVkIGhlcmUpIi5mb3JtYXQoZml0WyJpbmMiXSksCiAgICAgICAgICAgICJhbmdsZSI6IGZpdFsiYW5nbGUiXSwgImluYyI6IGZpdFsiaW5jIl0sCiAgICAgICAgICAgICJub3RlIjogKCJBbmdsZXMgZGVyaXZlZCBmcm9tIHRoZSBvYmplY3RzIHRoZW1zZWx2ZXMgYnkgZXZlbmluZyBvdXQgdGhlICIKICAgICAgICAgICAgICAgICAgICAgImNvdW50cyBpbiB7fSB3ZWRnZXM6IFBBIHs6LjBmfSBkZWcgaW4gaW1hZ2UsIGkgezouMGZ9IGRlZywgIgogICAgICAgICAgICAgICAgICAgICAic3RyZXRjaCB7Oi4yZn14LiIpLmZvcm1hdChuX3NlY3RvcnMsIGZpdFsiYW5nbGUiXSwgZml0WyJpbmMiXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgc3RyZXRjaF9mYWN0b3IoZml0WyJpbmMiXSkpICsgd2Fybn0KCmRlZiBjb21wdXRlX3Byb2ZpbGVzKHgsIHksIGNlbnRyZSwgZnJhbWUsIHRyZWF0bWVudHMsIHJpbmdzPTIwKToKICAgICIiIkRlcHJvamVjdCBhbmQgYmluIHRoZSBvYmplY3RzIG9uY2UgcGVyIHRyZWF0bWVudC4iIiIKICAgIGR4MCwgZHkwID0gbnAuYXNhcnJheSh4LCBmbG9hdCkgLSBjZW50cmVbMF0sIG5wLmFzYXJyYXkoeSwgZmxvYXQpIC0gY2VudHJlWzFdCiAgICByZXN1bHRzID0gW10KICAgIGZvciB0IGluIHRyZWF0bWVudHM6CiAgICAgICAgaWYgdFsiYW5nbGUiXSBpcyBOb25lOgogICAgICAgICAgICBkeHAsIGR5cCA9IGR4MCwgZHkwCiAgICAgICAgZWxzZToKICAgICAgICAgICAgZHhwLCBkeXAgPSBkZXByb2plY3QoZHgwLCBkeTAsIHRbImFuZ2xlIl0sIHRbImluYyJdKQogICAgICAgIHJlcyA9IHJhZGlhbF9kZW5zaXR5KGR4cCwgZHlwLCBuX3JpbmdzPXJpbmdzLCBmcmFtZV9jb3JuZXJzPWZyYW1lLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNlbnRyZV94eT1jZW50cmUsIG1ham9yX2F4aXNfYW5nbGVfZGVnPXRbImFuZ2xlIl0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgaW5jX2RlZz10WyJpbmMiXSkKICAgICAgICByZXNbImR4Il0sIHJlc1siZHkiXSA9IGR4cCwgZHlwCiAgICAgICAgcmVzdWx0cy5hcHBlbmQocmVzKQogICAgcmV0dXJuIHJlc3VsdHMKCmRlZiBnZW9tZXRyaWNfYXJlYXMoZWRnZXMpOgogICAgIiIiVGhlIGZ1bGwgbWF0aGVtYXRpY2FsIHJpbmcgYXJlYXMsIGZvciBjb21wYXJpc29uIHdpdGggdGhlIG9ic2VydmVkIG9uZXMuIiIiCiAgICByZXR1cm4gbWF0aC5waSAqIChucC5hc2FycmF5KGVkZ2VzKVsxOl0gKiogMiAtIG5wLmFzYXJyYXkoZWRnZXMpWzotMV0gKiogMikKCmRlZiB3cml0ZV9wcm9maWxlc19jc3YocGF0aCwgdHJlYXRtZW50cywgcmVzdWx0cywgc2NhbGU9MS4wLCB1bml0PSJweCIpOgogICAgIiIiU2F2ZSBldmVyeSBwcm9maWxlLCBpbmNsdWRpbmcgYm90aCB0aGUgb2JzZXJ2ZWQgYW5kIHRoZSBmdWxsIHJpbmcgYXJlYXMuCgogICAgS2VlcGluZyB0aGUgZnVsbCBnZW9tZXRyaWMgYXJlYSBhbG9uZ3NpZGUgdGhlIG9ic2VydmVkIG9uZSBsZXRzIGFueW9uZSBjb21wYXJlCiAgICBhZ2FpbnN0IG9sZGVyIGFuYWx5c2VzLCB3aGljaCBkaXZpZGVkIGJ5IHRoZSBmdWxsIGFyZWEgYW5kIHNvIHVuZGVyLXJlcG9ydGVkIHRoZQogICAgZGVuc2l0eSBvZiBhbnkgcmluZyB0aGUgZnJhbWUgaGFkIGNsaXBwZWQuCiAgICAiIiIKICAgIHdpdGggb3BlbihwYXRoLCAidyIsIGVuY29kaW5nPSJ1dGYtOCIpIGFzIGZoOgogICAgICAgIGZoLndyaXRlKCJ0cmVhdG1lbnQsbWFqb3JfYXhpc19hbmdsZV9kZWcsaW5jbGluYXRpb25fZGVnLCIKICAgICAgICAgICAgICAgICAicl9pbm5lcl97dX0scl9vdXRlcl97dX0scl9jZW50cmVfe3V9LG5fb2JqZWN0cywiCiAgICAgICAgICAgICAgICAgIm9ic2VydmVkX2FyZWFfe3V9MixmdWxsX3JpbmdfYXJlYV97dX0yLGFyZWFfZnJhY3Rpb24sIgogICAgICAgICAgICAgICAgICJkZW5zaXR5X3Blcl97dX0yLGRlbnNpdHlfZXJyLHJpbmdfY29tcGxldGVcbiIuZm9ybWF0KHU9dW5pdCkpCiAgICAgICAgZm9yIHQsIHJlcyBpbiB6aXAodHJlYXRtZW50cywgcmVzdWx0cyk6CiAgICAgICAgICAgIGdlbyA9IGdlb21ldHJpY19hcmVhcyhyZXNbImVkZ2VzIl0pCiAgICAgICAgICAgIGZvciBrIGluIHJhbmdlKGxlbihyZXNbInIiXSkpOgogICAgICAgICAgICAgICAgZmgud3JpdGUoInt9LHt9LHt9LHs6LjRmfSx7Oi40Zn0sezouNGZ9LHt9LHs6LjRmfSx7Oi40Zn0sezouNGZ9LCIKICAgICAgICAgICAgICAgICAgICAgICAgICJ7Oi42ZX0sezouNmV9LHt9XG4iLmZvcm1hdCgKICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0WyJsYWJlbCJdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICIiIGlmIHRbImFuZ2xlIl0gaXMgTm9uZSBlbHNlICJ7Oi4xZn0iLmZvcm1hdCh0WyJhbmdsZSJdKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAiIiBpZiB0WyJpbmMiXSBpcyBOb25lIGVsc2UgIns6LjFmfSIuZm9ybWF0KHRbImluYyJdKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXNbImVkZ2VzIl1ba10gKiBzY2FsZSwgcmVzWyJlZGdlcyJdW2sgKyAxXSAqIHNjYWxlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJlc1siciJdW2tdICogc2NhbGUsIGludChyZXNbImNvdW50cyJdW2tdKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXNbImFyZWEiXVtrXSAqIHNjYWxlICoqIDIsIGdlb1trXSAqIHNjYWxlICoqIDIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVzWyJhcmVhX2ZyYWN0aW9uIl1ba10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVzWyJkZW5zaXR5Il1ba10gLyBzY2FsZSAqKiAyLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJlc1siZGVuc2l0eV9lcnIiXVtrXSAvIHNjYWxlICoqIDIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgaW50KHJlc1siY29tcGxldGUiXVtrXSkpKQoKZGVmIHN0ZXBfcHJvZmlsZShheCwgZWRnZXMsIHZhbHVlcywgY29sb3VyLCBsdz0xLjcsIGFscGhhPTEuMCwgbHM9Ii0iLCBsYWJlbD1Ob25lKToKICAgICIiIkRyYXcgYSBkZW5zaXR5IHByb2ZpbGUgYXMgc3RlcHMsIGZsYXQgYWNyb3NzIGVhY2ggcmluZy4KCiAgICBBIHN1cmZhY2UgZGVuc2l0eSBpcyBub3QgYSBtZWFzdXJlbWVudCBhdCBhIHBvaW50LiBJdCBpcyBvbmUgbnVtYmVyIGZvciBhIHdob2xlCiAgICByaW5nIC0gdGhlIGNvdW50IGluIGl0IGRpdmlkZWQgYnkgaXRzIGFyZWEgLSBhbmQgaXQgc2F5cyBub3RoaW5nIGFib3V0IHdoZXJlCiAgICBpbnNpZGUgdGhhdCByaW5nIHRoZSBvYmplY3RzIHNhdC4gRHJhd2luZyBpdCBhcyBhIGRvdCBhdCB0aGUgcmluZydzIGNlbnRyZQogICAgam9pbmVkIHRvIHRoZSBuZXh0IGRvdCBieSBhIHNsb3BpbmcgbGluZSBpbnZpdGVzIGV4YWN0bHkgdGhlIHJlYWRpbmcgdGhlIGRhdGEKICAgIGRvZXMgbm90IHN1cHBvcnQ6IHRoYXQgdGhlIGRlbnNpdHkgdmFyaWVzIHNtb290aGx5IGFuZCB0aGF0IHRoZSB2YWx1ZSBhdCB0aGUKICAgIGNlbnRyZSBvZiB0aGUgcmluZyBpcyBzb21laG93IGJldHRlciBtZWFzdXJlZCB0aGFuIGF0IGl0cyBlZGdlLgoKICAgIFN0ZXBzIHNheSB3aGF0IHdhcyBhY3R1YWxseSBtZWFzdXJlZCwgYW5kIGl0IGlzIGhvdyB0aGUgdG9vbCBpbiB0aGUgcmVwb3NpdG9yeQogICAgaGFzIGFsd2F5cyBkcmF3biB0aGlzLCBzbyB0aGUgdHdvIGNhbiBiZSBsYWlkIHNpZGUgYnkgc2lkZS4KICAgICIiIgogICAgeSA9IG5wLmFwcGVuZCh2YWx1ZXMsIHZhbHVlc1stMV0pCiAgICBheC5zdGVwKGVkZ2VzLCB5LCB3aGVyZT0icG9zdCIsIGNvbG9yPWNvbG91ciwgbHc9bHcsIGFscGhhPWFscGhhLCBscz1scywKICAgICAgICAgICAgbGFiZWw9bGFiZWwpCgpkZWYgcHJvZmlsZV9maWd1cmUodHJlYXRtZW50cywgcmVzdWx0cywgdW5pdF9sYWJlbCwgdGl0bGUsIHNjYWxlPTEuMCk6CiAgICAiIiJPbmUgcm93IHBlciB0cmVhdG1lbnQgLSBzY2F0dGVyIGFuZCBwcm9maWxlIC0gdGhlbiBhbGwgcHJvZmlsZXMgb3ZlcmxhaWQuIiIiCiAgICBuID0gbGVuKHRyZWF0bWVudHMpCiAgICBmaWcgPSBwbHQuZmlndXJlKGZpZ3NpemU9KDExLjAsIDQuMSAqIG4gKyAzLjQpKQogICAgZ3MgPSBmaWcuYWRkX2dyaWRzcGVjKG4gKyAxLCAyLCBoZWlnaHRfcmF0aW9zPVsxXSAqIG4gKyBbMS4yNV0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgaHNwYWNlPTAuODUsIHdzcGFjZT0wLjMwKQoKICAgIGZvciBpLCAodCwgcmVzKSBpbiBlbnVtZXJhdGUoemlwKHRyZWF0bWVudHMsIHJlc3VsdHMpKToKICAgICAgICBjb2wgPSBDT0xPVVJTW2kgJSBsZW4oQ09MT1VSUyldCiAgICAgICAgZHh1LCBkeXUgPSByZXNbImR4Il0gKiBzY2FsZSwgcmVzWyJkeSJdICogc2NhbGUKICAgICAgICBydSA9IHJlc1siciJdICogc2NhbGUKICAgICAgICBkdSA9IHJlc1siZGVuc2l0eSJdIC8gc2NhbGUgKiogMgogICAgICAgIGV1ID0gcmVzWyJkZW5zaXR5X2VyciJdIC8gc2NhbGUgKiogMgoKICAgICAgICBheCA9IGZpZy5hZGRfc3VicGxvdChnc1tpLCAwXSkKICAgICAgICBheC5zY2F0dGVyKGR4dSwgZHl1LCBzPTcsIGM9Y29sLCBhbHBoYT0wLjY1LCBsaW5ld2lkdGhzPTApCiAgICAgICAgYXgucGxvdCgwLCAwLCAiKyIsIGM9ImsiLCBtcz0xMSwgbWV3PTEuNikKICAgICAgICBsaW0gPSAxLjA2ICogbWF4KG5wLmFicyhkeHUpLm1heCgpLCBucC5hYnMoZHl1KS5tYXgoKSkKICAgICAgICBheC5zZXRfeGxpbSgtbGltLCBsaW0pOyBheC5zZXRfeWxpbSgtbGltLCBsaW0pCiAgICAgICAgYXguc2V0X2FzcGVjdCgiZXF1YWwiLCBhZGp1c3RhYmxlPSJib3giKQogICAgICAgIGF4LnNldF90aXRsZSh0WyJsYWJlbCJdLCBmb250c2l6ZT0xMC41LCBsb2M9ImxlZnQiKQogICAgICAgIGF4LnNldF94bGFiZWwoInggZnJvbSBjZW50cmUgWyIgKyB1bml0X2xhYmVsICsgIl0iLCBmb250c2l6ZT04LjUpCiAgICAgICAgYXguc2V0X3lsYWJlbCgieSBmcm9tIGNlbnRyZSBbIiArIHVuaXRfbGFiZWwgKyAiXSIsIGZvbnRzaXplPTguNSkKICAgICAgICBheC50aWNrX3BhcmFtcyhsYWJlbHNpemU9Ny41KQogICAgICAgIGF4LmdyaWQoYWxwaGE9MC4xNSwgbHc9MC42KQoKICAgICAgICBheDIgPSBmaWcuYWRkX3N1YnBsb3QoZ3NbaSwgMV0pCiAgICAgICAgZyA9IHJlc1siY29tcGxldGUiXQogICAgICAgIGVkZ2VzID0gcmVzWyJlZGdlcyJdICogc2NhbGUKICAgICAgICAjIFRoZSBzdGVwIGlzIHRoZSBtZWFzdXJlbWVudC4gVGhlIGJhcnMgb24gdG9wIG9mIGl0IGFyZSB0aGUgY291bnRpbmcKICAgICAgICAjIHVuY2VydGFpbnR5LCBkcmF3biBhdCB0aGUgbWlkZGxlIG9mIGVhY2ggcmluZyBvbmx5IGJlY2F1c2UgdGhleSBoYXZlIHRvCiAgICAgICAgIyBiZSBkcmF3biBzb21ld2hlcmUuCiAgICAgICAgc3RlcF9wcm9maWxlKGF4MiwgZWRnZXMsIG5wLndoZXJlKGcsIGR1LCBucC5uYW4pLCBjb2wpCiAgICAgICAgaWYgZy5hbnkoKToKICAgICAgICAgICAgYXgyLmVycm9yYmFyKHJ1W2ddLCBkdVtnXSwgeWVycj1ldVtnXSwgZm10PSJub25lIiwgZWNvbG9yPWNvbCwKICAgICAgICAgICAgICAgICAgICAgICAgIGVsaW5ld2lkdGg9MS4wLCBjYXBzaXplPTIsIGFscGhhPTAuNzUpCiAgICAgICAgaWYgKH5nKS5hbnkoKToKICAgICAgICAgICAgc3RlcF9wcm9maWxlKGF4MiwgZWRnZXMsIG5wLndoZXJlKH5nLCBkdSwgbnAubmFuKSwgY29sLAogICAgICAgICAgICAgICAgICAgICAgICAgbHc9MS4yLCBhbHBoYT0wLjQ1LCBscz0iLS0iLAogICAgICAgICAgICAgICAgICAgICAgICAgbGFiZWw9InJpbmcgY2xpcHBlZCBieSB0aGUgZnJhbWUiKQogICAgICAgICAgICBheDIubGVnZW5kKGZvbnRzaXplPTcsIGZyYW1lb249RmFsc2UsIGxvYz0ibG93ZXIgbGVmdCIpCiAgICAgICAgYXgyLnNldF95c2NhbGUoImxvZyIpCiAgICAgICAgYXgyLnNldF94bGFiZWwoImRlcHJvamVjdGVkIHJhZGl1cyBbIiArIHVuaXRfbGFiZWwgKyAiXSIsIGZvbnRzaXplPTguNSkKICAgICAgICBheDIuc2V0X3lsYWJlbCgic3VyZmFjZSBkZW5zaXR5IFtuIHBlciAiICsgdW5pdF9sYWJlbCArICJeMl0iLCBmb250c2l6ZT04LjUpCiAgICAgICAgYXgyLnRpY2tfcGFyYW1zKGxhYmVsc2l6ZT03LjUpCiAgICAgICAgYXgyLmdyaWQoYWxwaGE9MC4xNSwgbHc9MC42KQogICAgICAgIGF4Mi5zZXRfdGl0bGUoIk4gPSAiICsgc3RyKGludChyZXNbImNvdW50cyJdLnN1bSgpKSksIGZvbnRzaXplPTksIGxvYz0icmlnaHQiKQoKICAgICAgICBheC50ZXh0KDAuMCwgLTAuMzAsICJcbiIuam9pbih0ZXh0d3JhcC53cmFwKHRbIm5vdGUiXSwgd2lkdGg9MTI1KSksCiAgICAgICAgICAgICAgICB0cmFuc2Zvcm09YXgudHJhbnNBeGVzLCBmb250c2l6ZT03LjYsIGNvbG9yPSIjNGE0YTRhIiwKICAgICAgICAgICAgICAgIHZhPSJ0b3AiLCBoYT0ibGVmdCIsIGxpbmVzcGFjaW5nPTEuNSkKCiAgICBheGMgPSBmaWcuYWRkX3N1YnBsb3QoZ3NbbiwgOl0pCiAgICBmb3IgaSwgKHQsIHJlcykgaW4gZW51bWVyYXRlKHppcCh0cmVhdG1lbnRzLCByZXN1bHRzKSk6CiAgICAgICAgZyA9IHJlc1siY29tcGxldGUiXQogICAgICAgIGlmIGcuYW55KCk6CiAgICAgICAgICAgIHN0ZXBfcHJvZmlsZShheGMsIHJlc1siZWRnZXMiXSAqIHNjYWxlLAogICAgICAgICAgICAgICAgICAgICAgICAgbnAud2hlcmUoZywgcmVzWyJkZW5zaXR5Il0gLyBzY2FsZSAqKiAyLCBucC5uYW4pLAogICAgICAgICAgICAgICAgICAgICAgICAgQ09MT1VSU1tpICUgbGVuKENPTE9VUlMpXSwgbGFiZWw9dFsibGFiZWwiXSkKICAgIGF4Yy5zZXRfeXNjYWxlKCJsb2ciKQogICAgYXhjLnNldF94bGFiZWwoImRlcHJvamVjdGVkIHJhZGl1cyBbIiArIHVuaXRfbGFiZWwgKyAiXSIpCiAgICBheGMuc2V0X3lsYWJlbCgic3VyZmFjZSBkZW5zaXR5IFtuIHBlciAiICsgdW5pdF9sYWJlbCArICJeMl0iKQogICAgYXhjLmdyaWQoYWxwaGE9MC4xOCwgbHc9MC42KQogICAgYXhjLmxlZ2VuZChmb250c2l6ZT04LjUsIGZyYW1lb249RmFsc2UpCiAgICBheGMuc2V0X3RpdGxlKCJBbGwgdHJlYXRtZW50cyBvdmVybGFpZCAtIHRoZSBzcHJlYWQgYmV0d2VlbiB0aGVtIGlzIHRoZSAiCiAgICAgICAgICAgICAgICAgICJ1bmNlcnRhaW50eSBvbiB0aGUgcmFkaWkiLCBmb250c2l6ZT0xMCwgbG9jPSJsZWZ0IikKCiAgICBmaWcuc3VwdGl0bGUodGl0bGUsIGZvbnRzaXplPTEzLCB5PTAuOTk1KQogICAgcmV0dXJuIGZpZwoKZGVmIF9wcm9maWxlX3Jvd3MocmVzdWx0LCBzY2FsZSwgdW5pdCk6CiAgICAiIiJUaGUgcHJvZmlsZSBleGFjdGx5IGFzIHRoZSBwaWN0dXJlIGRyYXdzIGl0LCByaW5nIGJ5IHJpbmcuIiIiCiAgICBlZGdlcyA9IHJlc3VsdFsiZWRnZXMiXSAqIHNjYWxlCiAgICBoZWFkID0gWyJyaW5nX2lubmVyXyIgKyB1bml0LCAicmluZ19vdXRlcl8iICsgdW5pdCwgInJfY2VudHJlXyIgKyB1bml0LAogICAgICAgICAgICAibl9vYmplY3RzIiwgImRlbnNpdHlfcGVyXyIgKyB1bml0ICsgIjIiLCAiZGVuc2l0eV9lcnJvciIsCiAgICAgICAgICAgICJyaW5nX2NvbXBsZXRlIl0KICAgIHJvd3MgPSBbXQogICAgZm9yIGsgaW4gcmFuZ2UobGVuKHJlc3VsdFsiciJdKSk6CiAgICAgICAgcm93cy5hcHBlbmQoW2Zsb2F0KGVkZ2VzW2tdKSwgZmxvYXQoZWRnZXNbayArIDFdKSwKICAgICAgICAgICAgICAgICAgICAgZmxvYXQocmVzdWx0WyJyIl1ba10gKiBzY2FsZSksCiAgICAgICAgICAgICAgICAgICAgIGludChyZXN1bHRbImNvdW50cyJdW2tdKSwKICAgICAgICAgICAgICAgICAgICAgZmxvYXQocmVzdWx0WyJkZW5zaXR5Il1ba10gLyBzY2FsZSAqKiAyKSwKICAgICAgICAgICAgICAgICAgICAgZmxvYXQocmVzdWx0WyJkZW5zaXR5X2VyciJdW2tdIC8gc2NhbGUgKiogMiksCiAgICAgICAgICAgICAgICAgICAgIGludChib29sKHJlc3VsdFsiY29tcGxldGUiXVtrXSkpXSkKICAgIHJldHVybiBoZWFkLCByb3dzCgpkZWYgX3Bvc2l0aW9uX3Jvd3MocmVzdWx0LCBzY2FsZSwgdW5pdCk6CiAgICAiIiJFdmVyeSBvYmplY3QsIHdoZXJlIHRoZSBwaWN0dXJlIHB1dHMgaXQ6IGRlcHJvamVjdGVkLCBjZW50cmVkLCBpbiBgdW5pdGAuIiIiCiAgICBoZWFkID0gWyJ4X2Zyb21fY2VudHJlXyIgKyB1bml0LCAieV9mcm9tX2NlbnRyZV8iICsgdW5pdCwKICAgICAgICAgICAgInJhZGl1c18iICsgdW5pdF0KICAgIGR4ID0gbnAuYXNhcnJheShyZXN1bHRbImR4Il0sIGR0eXBlPWZsb2F0KSAqIHNjYWxlCiAgICBkeSA9IG5wLmFzYXJyYXkocmVzdWx0WyJkeSJdLCBkdHlwZT1mbG9hdCkgKiBzY2FsZQogICAgcm93cyA9IFtbZmxvYXQoYSksIGZsb2F0KGIpLCBmbG9hdChucC5oeXBvdChhLCBiKSldIGZvciBhLCBiIGluIHppcChkeCwgZHkpXQogICAgcmV0dXJuIGhlYWQsIHJvd3MKCmRlZiBfd3JpdGVfdGFibGVzKHBhdGgsIHNoZWV0cyk6CiAgICAiIiJXcml0ZSBlYWNoIHRhYmxlIGFzIGl0cyBvd24gQ1NWLCBuYW1lZCBhZnRlciB0aGUgcGljdHVyZSBpdCBiZWxvbmdzIHRvLgoKICAgIFdIWSBPTkUgRklMRSBQRVIgVEFCTEUgQU5EIE5PVCBPTkUgUEVSIFBJQ1RVUkUKICAgIEJlY2F1c2UgdGhlIHR3byB0YWJsZXMgYmVoaW5kIGEgcGljdHVyZSBoYXZlIGRpZmZlcmVudCBzaGFwZXMgLSBvbmUgcm93IHBlcgogICAgcmluZyBpbiB0aGUgcHJvZmlsZSwgb25lIHJvdyBwZXIgb2JqZWN0IGluIHRoZSBwb3NpdGlvbnMgLSBhbmQgcHV0dGluZyB0aGVtIGluCiAgICBvbmUgZmlsZSBtZWFucyBvbmUgb2YgdGhlbSBzdGFydHMgaGFsZndheSBkb3duLCB3aGljaCBpcyBleGFjdGx5IHRoZSBsYXlvdXQKICAgIHRoYXQgbWFrZXMgYSBzcHJlYWRzaGVldCBhd2t3YXJkIHRvIGNoYXJ0IGZyb20uIFR3byBmaWxlcywgZWFjaCBhIGNsZWFuCiAgICByZWN0YW5nbGUgd2l0aCBhIGhlYWRlciByb3csIGNhbiBiZSBjaGFydGVkIGJ5IHNlbGVjdGluZyB0d28gY29sdW1ucy4KCiAgICBDU1YgcmF0aGVyIHRoYW4gYSBzcHJlYWRzaGVldCBmb3JtYXQgb24gcHVycG9zZTogaXQgb3BlbnMgaW4gRXhjZWwsIGl0IG9wZW5zCiAgICBpbiBhbnl0aGluZyBlbHNlLCBhbmQgaXQgbmVlZHMgbm8gbGlicmFyeSB0aGF0IG1pZ2h0IG5vdCBiZSBpbnN0YWxsZWQgb24gdGhlCiAgICBtYWNoaW5lIHRoaXMgZW5kcyB1cCBydW5uaW5nIG9uLgoKICAgIFJldHVybnMgdGhlIHBhdGhzIHdyaXR0ZW4uCiAgICAiIiIKICAgIGltcG9ydCBjc3YgYXMgX2NzdgoKICAgIHN0ZW0gPSBvcy5wYXRoLnNwbGl0ZXh0KHBhdGgpWzBdCiAgICB3cml0dGVuID0gW10KICAgIGZvciBuYW1lLCAoaGVhZCwgcm93cykgaW4gc2hlZXRzLml0ZW1zKCk6CiAgICAgICAgc2x1ZyA9ICJfIi5qb2luKHAgZm9yIHAgaW4KICAgICAgICAgICAgICAgICAgICAgICAgIiIuam9pbihjIGlmIGMuaXNhbG51bSgpIGVsc2UgIl8iIGZvciBjIGluIG5hbWUpLnNwbGl0KCJfIikKICAgICAgICAgICAgICAgICAgICAgICAgaWYgcCkubG93ZXIoKQogICAgICAgIG91dCA9ICJ7fV97fS5jc3YiLmZvcm1hdChzdGVtLCBzbHVnKQogICAgICAgIHdpdGggb3BlbihvdXQsICJ3IiwgbmV3bGluZT0iIiwgZW5jb2Rpbmc9InV0Zi04IikgYXMgZmg6CiAgICAgICAgICAgIHdyaXRlciA9IF9jc3Yud3JpdGVyKGZoKQogICAgICAgICAgICB3cml0ZXIud3JpdGVyb3coaGVhZCkKICAgICAgICAgICAgd3JpdGVyLndyaXRlcm93cyhyb3dzKQogICAgICAgIHdyaXR0ZW4uYXBwZW5kKG91dCkKICAgIHJldHVybiB3cml0dGVuCgpkZWYgd3JpdGVfdHJlYXRtZW50X3dvcmtib29rKHBhdGgsIHRyZWF0bWVudCwgcmVzdWx0LCBzY2FsZSwgdW5pdCk6CiAgICAiIiJUaGUgdHdvIHRhYmxlcyBiZWhpbmQgb25lIHRyZWF0bWVudCdzIHBpY3R1cmUuIiIiCiAgICByZXR1cm4gX3dyaXRlX3RhYmxlcyhwYXRoLCB7CiAgICAgICAgInByb2ZpbGUiOiBfcHJvZmlsZV9yb3dzKHJlc3VsdCwgc2NhbGUsIHVuaXQpLAogICAgICAgICJwb3NpdGlvbnMiOiBfcG9zaXRpb25fcm93cyhyZXN1bHQsIHNjYWxlLCB1bml0KSwKICAgIH0pCgpkZWYgd3JpdGVfb3ZlcmxheV93b3JrYm9vayhwYXRoLCB0cmVhdG1lbnRzLCByZXN1bHRzLCBzY2FsZSwgdW5pdCk6CiAgICAiIiJPbmUgdGFibGUgcGVyIGN1cnZlIGluIHRoZSBvdmVybGF5LiIiIgogICAgc2hlZXRzID0ge30KICAgIGZvciB0cmVhdG1lbnQsIHJlc3VsdCBpbiB6aXAodHJlYXRtZW50cywgcmVzdWx0cyk6CiAgICAgICAgbmFtZSA9ICIiLmpvaW4oYyBpZiAoYy5pc2FsbnVtKCkgb3IgYyA9PSAiICIpIGVsc2UgIiAiCiAgICAgICAgICAgICAgICAgICAgICAgZm9yIGMgaW4gdHJlYXRtZW50WyJsYWJlbCJdKS5zdHJpcCgpCiAgICAgICAgc2hlZXRzW25hbWVbOjMxXSBvciAidHJlYXRtZW50Il0gPSBfcHJvZmlsZV9yb3dzKHJlc3VsdCwgc2NhbGUsIHVuaXQpCiAgICByZXR1cm4gX3dyaXRlX3RhYmxlcyhwYXRoLCBzaGVldHMpCgpkZWYgb25lX3RyZWF0bWVudF9maWd1cmUodHJlYXRtZW50LCByZXN1bHQsIHVuaXRfbGFiZWwsIHRpdGxlLCBzY2FsZT0xLjAsCiAgICAgICAgICAgICAgICAgICAgICAgICBjb2xvdXJfaW5kZXg9MCk6CiAgICAiIiJPbmUgdHJlYXRtZW50IG9uIGl0cyBvd24gc2hlZXQ6IHdoZXJlIHRoZSBvYmplY3RzIGFyZSwgYW5kIHRoZWlyIHByb2ZpbGUuCgogICAgV0hZIFRIRSBTQU1FIFRISU5HIElTIERSQVdOIFRXSUNFCiAgICBUaGUgY29tYmluZWQgc2hlZXQgaXMgZm9yIGNvbXBhcmluZyB0aGUgdHJlYXRtZW50cyBhZ2FpbnN0IGVhY2ggb3RoZXIsIHdoaWNoCiAgICBpcyB3aGF0IGl0IGlzIGZvciBhbmQgd2hhdCBpdCBpcyBnb29kIGF0LiBJdCBpcyBub3Qgd2hhdCB5b3UgcHV0IGluIGZyb250IG9mCiAgICBzb21lYm9keSwgb3IgaW50byBhIHJlcG9ydDogYXQgZm91ciBpbmNoZXMgaGlnaCBhIHBhbmVsIGlzIHRvbyBzbWFsbCB0byByZWFkIGEKICAgIHJhZGl1cyBvZmYgaXQsIGFuZCB0d28gdHJlYXRtZW50cyBzaWRlIGJ5IHNpZGUgY29tcGV0ZSBmb3IgdGhlIGV5ZSB3aGVuIG9ubHkKICAgIG9uZSBvZiB0aGVtIGlzIHVuZGVyIGRpc2N1c3Npb24uIFNvIGVhY2ggaXMgd3JpdHRlbiBhZ2FpbiBvbiBpdHMgb3duLCBmdWxsCiAgICBzaXplLCBhbmQgdGhlIG92ZXJsYXkgdG9vLiBUaGUgY29tYmluZWQgc2hlZXQgaXMgdW5jaGFuZ2VkOyB0aGVzZSBhcmUgZXh0cmEuCiAgICAiIiIKICAgIGNvbCA9IENPTE9VUlNbY29sb3VyX2luZGV4ICUgbGVuKENPTE9VUlMpXQogICAgZmlnID0gcGx0LmZpZ3VyZShmaWdzaXplPSgxMy4wLCA2LjIpKQogICAgZ3MgPSBmaWcuYWRkX2dyaWRzcGVjKDEsIDIsIHdzcGFjZT0wLjI4KQoKICAgIGR4dSwgZHl1ID0gcmVzdWx0WyJkeCJdICogc2NhbGUsIHJlc3VsdFsiZHkiXSAqIHNjYWxlCiAgICBydSA9IHJlc3VsdFsiciJdICogc2NhbGUKICAgIGR1ID0gcmVzdWx0WyJkZW5zaXR5Il0gLyBzY2FsZSAqKiAyCiAgICBldSA9IHJlc3VsdFsiZGVuc2l0eV9lcnIiXSAvIHNjYWxlICoqIDIKCiAgICBheCA9IGZpZy5hZGRfc3VicGxvdChnc1swLCAwXSkKICAgIGF4LnNjYXR0ZXIoZHh1LCBkeXUsIHM9MTQsIGM9Y29sLCBhbHBoYT0wLjcsIGxpbmV3aWR0aHM9MCkKICAgIGF4LnBsb3QoMCwgMCwgIisiLCBjPSJrIiwgbXM9MTQsIG1ldz0xLjgpCiAgICBsaW0gPSAxLjA2ICogbWF4KG5wLmFicyhkeHUpLm1heCgpLCBucC5hYnMoZHl1KS5tYXgoKSkKICAgIGF4LnNldF94bGltKC1saW0sIGxpbSkKICAgIGF4LnNldF95bGltKC1saW0sIGxpbSkKICAgIGF4LnNldF9hc3BlY3QoImVxdWFsIiwgYWRqdXN0YWJsZT0iYm94IikKICAgIGF4LnNldF94bGFiZWwoInggZnJvbSBjZW50cmUgWyIgKyB1bml0X2xhYmVsICsgIl0iKQogICAgYXguc2V0X3lsYWJlbCgieSBmcm9tIGNlbnRyZSBbIiArIHVuaXRfbGFiZWwgKyAiXSIpCiAgICBheC5ncmlkKGFscGhhPTAuMTUsIGx3PTAuNikKICAgIGF4LnNldF90aXRsZSgid2hlcmUgdGhleSBhcmUiLCBmb250c2l6ZT0xMSwgbG9jPSJsZWZ0IikKCiAgICBheDIgPSBmaWcuYWRkX3N1YnBsb3QoZ3NbMCwgMV0pCiAgICBnID0gcmVzdWx0WyJjb21wbGV0ZSJdCiAgICBlZGdlcyA9IHJlc3VsdFsiZWRnZXMiXSAqIHNjYWxlCiAgICBzdGVwX3Byb2ZpbGUoYXgyLCBlZGdlcywgbnAud2hlcmUoZywgZHUsIG5wLm5hbiksIGNvbCkKICAgIGlmIGcuYW55KCk6CiAgICAgICAgYXgyLmVycm9yYmFyKHJ1W2ddLCBkdVtnXSwgeWVycj1ldVtnXSwgZm10PSJub25lIiwgZWNvbG9yPWNvbCwKICAgICAgICAgICAgICAgICAgICAgZWxpbmV3aWR0aD0xLjIsIGNhcHNpemU9MywgYWxwaGE9MC44KQogICAgaWYgKH5nKS5hbnkoKToKICAgICAgICBzdGVwX3Byb2ZpbGUoYXgyLCBlZGdlcywgbnAud2hlcmUofmcsIGR1LCBucC5uYW4pLCBjb2wsCiAgICAgICAgICAgICAgICAgICAgIGx3PTEuMywgYWxwaGE9MC40NSwgbHM9Ii0tIiwKICAgICAgICAgICAgICAgICAgICAgbGFiZWw9InJpbmcgY2xpcHBlZCBieSB0aGUgZnJhbWUiKQogICAgICAgIGF4Mi5sZWdlbmQoZm9udHNpemU9OSwgZnJhbWVvbj1GYWxzZSwgbG9jPSJsb3dlciBsZWZ0IikKICAgIGF4Mi5zZXRfeXNjYWxlKCJsb2ciKQogICAgYXgyLnNldF94bGFiZWwoImRlcHJvamVjdGVkIHJhZGl1cyBbIiArIHVuaXRfbGFiZWwgKyAiXSIpCiAgICBheDIuc2V0X3lsYWJlbCgic3VyZmFjZSBkZW5zaXR5IFtuIHBlciAiICsgdW5pdF9sYWJlbCArICJeMl0iKQogICAgYXgyLmdyaWQoYWxwaGE9MC4xNSwgbHc9MC42KQogICAgYXgyLnNldF90aXRsZSgiTiA9ICIgKyBzdHIoaW50KHJlc3VsdFsiY291bnRzIl0uc3VtKCkpKSwgZm9udHNpemU9MTAsCiAgICAgICAgICAgICAgICAgIGxvYz0icmlnaHQiKQogICAgYXgyLnNldF90aXRsZSgiaG93IG1hbnkgcGVyIHVuaXQgYXJlYSwgYWdhaW5zdCByYWRpdXMiLCBmb250c2l6ZT0xMSwKICAgICAgICAgICAgICAgICAgbG9jPSJsZWZ0IikKCiAgICBmaWcuc3VwdGl0bGUoInt9IC0ge30iLmZvcm1hdCh0aXRsZSwgdHJlYXRtZW50WyJsYWJlbCJdKSwgZm9udHNpemU9MTMpCiAgICBmaWcudGV4dCgwLjA5LCAwLjAxNSwKICAgICAgICAgICAgIGNocigxMCkuam9pbih0ZXh0d3JhcC53cmFwKHRyZWF0bWVudFsibm90ZSJdLCB3aWR0aD0xNTApKSwKICAgICAgICAgICAgIGZvbnRzaXplPTksIGNvbG9yPSIjNGE0YTRhIiwgdmE9ImJvdHRvbSIpCiAgICBmaWcuc3VicGxvdHNfYWRqdXN0KGJvdHRvbT0wLjE3LCB0b3A9MC45MCkKICAgIHJldHVybiBmaWcKCmRlZiBvdmVybGF5X2ZpZ3VyZSh0cmVhdG1lbnRzLCByZXN1bHRzLCB1bml0X2xhYmVsLCB0aXRsZSwgc2NhbGU9MS4wKToKICAgICIiIlRoZSB0cmVhdG1lbnRzIG9uIG9uZSBwYWlyIG9mIGF4ZXMsIG9uIGEgc2hlZXQgb2YgaXRzIG93bi4iIiIKICAgIGZpZywgYXggPSBwbHQuc3VicGxvdHMoZmlnc2l6ZT0oMTMuMCwgNi4yKSkKICAgIGZvciBpLCAodCwgcmVzKSBpbiBlbnVtZXJhdGUoemlwKHRyZWF0bWVudHMsIHJlc3VsdHMpKToKICAgICAgICBnID0gcmVzWyJjb21wbGV0ZSJdCiAgICAgICAgaWYgZy5hbnkoKToKICAgICAgICAgICAgc3RlcF9wcm9maWxlKGF4LCByZXNbImVkZ2VzIl0gKiBzY2FsZSwKICAgICAgICAgICAgICAgICAgICAgICAgIG5wLndoZXJlKGcsIHJlc1siZGVuc2l0eSJdIC8gc2NhbGUgKiogMiwgbnAubmFuKSwKICAgICAgICAgICAgICAgICAgICAgICAgIENPTE9VUlNbaSAlIGxlbihDT0xPVVJTKV0sIGxhYmVsPXRbImxhYmVsIl0pCiAgICBheC5zZXRfeXNjYWxlKCJsb2ciKQogICAgYXguc2V0X3hsYWJlbCgiZGVwcm9qZWN0ZWQgcmFkaXVzIFsiICsgdW5pdF9sYWJlbCArICJdIikKICAgIGF4LnNldF95bGFiZWwoInN1cmZhY2UgZGVuc2l0eSBbbiBwZXIgIiArIHVuaXRfbGFiZWwgKyAiXjJdIikKICAgIGF4LmdyaWQoYWxwaGE9MC4xOCwgbHc9MC42KQogICAgYXgubGVnZW5kKGZvbnRzaXplPTEwLCBmcmFtZW9uPUZhbHNlKQogICAgYXguc2V0X3RpdGxlKCJ7fSAtIGFsbCB0cmVhdG1lbnRzIG92ZXJsYWlkOyB0aGUgc3ByZWFkIGJldHdlZW4gdGhlbSBpcyB0aGUgIgogICAgICAgICAgICAgICAgICJ1bmNlcnRhaW50eSBvbiB0aGUgcmFkaWkiLmZvcm1hdCh0aXRsZSksIGZvbnRzaXplPTEyLCBsb2M9ImxlZnQiKQogICAgZmlnLnRpZ2h0X2xheW91dCgpCiAgICByZXR1cm4gZmlnCgpkZWYgcnVuX3Byb2ZpbGVzKHgsIHksIGNlbnRyZSwgZnJhbWUsIG91dF9wcmVmaXgsIHRpdGxlLAogICAgICAgICAgICAgICAgIGdhbGF4eT1Ob25lLCBnZW9tPU5vbmUsIG5vcnRoX2FuZ2xlPTkwLjAsIG1pcnJvcmVkPUZhbHNlLAogICAgICAgICAgICAgICAgIG1hbnVhbF9pbmM9Tm9uZSwgbWFudWFsX3BhX2ltYWdlPU5vbmUsIG1hbnVhbF9wYV9za3k9Tm9uZSwKICAgICAgICAgICAgICAgICBzZWN0b3JfbWV0aG9kPUZhbHNlLCBzZWN0b3JzPTIwLCBzZWN0b3Jfc3RlcD01LjAsCiAgICAgICAgICAgICAgICAgcmluZ3M9MjAsIHNjYWxlPTEuMCwgdW5pdD0icHgiLCB2ZXJib3NlPVRydWUpOgogICAgIiIiRGVwcm9qZWN0LCBiaW4sIHBsb3QgYW5kIHNhdmUuIFJldHVybnMgKHRyZWF0bWVudHMsIHJlc3VsdHMsIHBhdGhzKS4KCiAgICBvdXRfcHJlZml4IGlzIGEgcGF0aCBzdGVtOyB0aGlzIHdyaXRlcyA8cHJlZml4Pl9wcm9maWxlcy5jc3YgYW5kCiAgICA8cHJlZml4Pl9wcm9maWxlcy5wbmcgbmV4dCB0byBpdC4KICAgICIiIgogICAgeCA9IG5wLmFzYXJyYXkoeCwgZHR5cGU9ZmxvYXQpCiAgICB5ID0gbnAuYXNhcnJheSh5LCBkdHlwZT1mbG9hdCkKCiAgICAjIFRvbyBtYW55IHJpbmdzIGZvciB0b28gZmV3IG9iamVjdHMgZG9lcyBub3QgYWRkIGRldGFpbCwgaXQgbWFudWZhY3R1cmVzCiAgICAjIG5vaXNlOiByaW5ncyBob2xkaW5nIG9uZSBvYmplY3Qgb3Igbm9uZSwgZXJyb3IgYmFycyB0YWxsZXIgdGhhbiB0aGUgcHJvZmlsZSwKICAgICMgYW5kIGEgamFnZ2VkIGxpbmUgdGhhdCBpbnZpdGVzIHJlYWRpbmcgc3RydWN0dXJlIGludG8gY291bnRpbmcgc3RhdGlzdGljcy4KICAgICMgU2VlbiBvbiBhIHJlYWwgcnVuIC0gMTgxIGJsdWUga25vdHMgc3ByZWFkIG92ZXIgMzAgcmluZ3MsIHdob2xlIHJpbmdzIGVtcHR5LgogICAgaWYgcmluZ3MgYW5kIGxlbih4KSBhbmQgbGVuKHgpIC8gZmxvYXQocmluZ3MpIDwgOC4wOgogICAgICAgIHN1Z2dlc3RlZCA9IG1heCg0LCBpbnQobGVuKHgpIC8gMTIpKQogICAgICAgIHByaW50KCIgIG5vdGU6IHt9IG9iamVjdHMgb3ZlciB7fSByaW5ncyBpcyBhYm91dCB7Oi4xZn0gcGVyIHJpbmcsIHNvIHRoZSAiCiAgICAgICAgICAgICAgInByb2ZpbGUgYmVsb3ciLmZvcm1hdChsZW4oeCksIHJpbmdzLCBsZW4oeCkgLyBmbG9hdChyaW5ncykpKQogICAgICAgIHByaW50KCIgICAgICAgIHdpbGwgYmUgZG9taW5hdGVkIGJ5IGNvdW50aW5nIG5vaXNlLiBBYm91dCB7fSByaW5ncyB3b3VsZCAiCiAgICAgICAgICAgICAgImJlIHJlYWRhYmxlLiIuZm9ybWF0KHN1Z2dlc3RlZCkpCgogICAgaWYgZ2VvbSBpcyBOb25lIGFuZCBnYWxheHk6CiAgICAgICAgdHJ5OgogICAgICAgICAgICBkZXNjcmliZSA9IGRlc2NyaWJlX19nYWxheHlfY2F0YWxvZ3VlICAjIGZyb20gZ2FsYXh5X2NhdGFsb2d1ZSwgaW5saW5lZCBhYm92ZQogICAgICAgICAgICBnZW9tID0gaHlwZXJsZWRhX2dlb21ldHJ5KGdhbGF4eSkKICAgICAgICAgICAgaWYgdmVyYm9zZToKICAgICAgICAgICAgICAgIHByaW50KCJcbkh5cGVyTEVEQSBnZW9tZXRyeSBmb3IgIiArIGdhbGF4eSArICI6IikKICAgICAgICAgICAgICAgIHByaW50KGRlc2NyaWJlKGdlb20pKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOgogICAgICAgICAgICBpZiB2ZXJib3NlOgogICAgICAgICAgICAgICAgcHJpbnQoIlxuQ2F0YWxvZ3VlIGxvb2t1cCBmYWlsZWQgKHt9OiB7fSkuIENvbnRpbnVpbmcgd2l0aG91dCBpdC4iCiAgICAgICAgICAgICAgICAgICAgICAuZm9ybWF0KHR5cGUoZXhjKS5fX25hbWVfXywgZXhjKSkKCiAgICB0cmVhdG1lbnRzID0gW3RyZWF0bWVudF9ub25lKCldCiAgICB0ID0gdHJlYXRtZW50X2NhdGFsb2d1ZShnZW9tLCBub3J0aF9hbmdsZSwgbWlycm9yZWQpCiAgICBpZiB0OgogICAgICAgIHRyZWF0bWVudHMuYXBwZW5kKHQpCiAgICB0ID0gdHJlYXRtZW50X21hbnVhbChtYW51YWxfaW5jLCBtYW51YWxfcGFfaW1hZ2UsIG1hbnVhbF9wYV9za3ksCiAgICAgICAgICAgICAgICAgICAgICAgICBub3J0aF9hbmdsZSwgbWlycm9yZWQpCiAgICBpZiB0OgogICAgICAgIHRyZWF0bWVudHMuYXBwZW5kKHQpCiAgICBpZiBzZWN0b3JfbWV0aG9kOgogICAgICAgIHRyZWF0bWVudHMuYXBwZW5kKHRyZWF0bWVudF9zZWN0b3IoeCAtIGNlbnRyZVswXSwgeSAtIGNlbnRyZVsxXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHNlY3RvcnMsIHNlY3Rvcl9zdGVwKSkKCiAgICByZXN1bHRzID0gY29tcHV0ZV9wcm9maWxlcyh4LCB5LCBjZW50cmUsIGZyYW1lLCB0cmVhdG1lbnRzLCByaW5ncz1yaW5ncykKCiAgICBpZiB2ZXJib3NlOgogICAgICAgIHByaW50KCJcblRyZWF0bWVudHM6IikKICAgICAgICBmb3IgdCwgcmVzIGluIHppcCh0cmVhdG1lbnRzLCByZXN1bHRzKToKICAgICAgICAgICAgcHJpbnQoIiAgLSB7fTogcl9tYXggPSB7Oi4xZn0ge30sIHt9IG9mIHt9IHJpbmdzIGNsaXBwZWQgYnkgdGhlIGZyYW1lIgogICAgICAgICAgICAgICAgICAuZm9ybWF0KHRbImxhYmVsIl0sIHJlc1sicl9tYXgiXSAqIHNjYWxlLCB1bml0LAogICAgICAgICAgICAgICAgICAgICAgICAgIGludCgofnJlc1siY29tcGxldGUiXSkuc3VtKCkpLCByaW5ncykpCiAgICAgICAgICAgIHByaW50KCIgICAgICAiICsgdFsibm90ZSJdKQoKICAgIGNzdl9wYXRoID0gb3V0X3ByZWZpeCArICJfcHJvZmlsZXMuY3N2IgogICAgcG5nX3BhdGggPSBvdXRfcHJlZml4ICsgIl9wcm9maWxlcy5wbmciCiAgICB3cml0ZV9wcm9maWxlc19jc3YoY3N2X3BhdGgsIHRyZWF0bWVudHMsIHJlc3VsdHMsIHNjYWxlLCB1bml0KQogICAgZmlnID0gcHJvZmlsZV9maWd1cmUodHJlYXRtZW50cywgcmVzdWx0cywgdW5pdCwgdGl0bGUsIHNjYWxlKQogICAgZmlnLnNhdmVmaWcocG5nX3BhdGgsIGRwaT0xNTAsIGJib3hfaW5jaGVzPSJ0aWdodCIpCiAgICBwbHQuY2xvc2UoZmlnKQogICAgY29tYmluZWRfYm9va3MgPSB3cml0ZV9vdmVybGF5X3dvcmtib29rKAogICAgICAgIG9zLnBhdGguc3BsaXRleHQocG5nX3BhdGgpWzBdICsgIi5jc3YiLCB0cmVhdG1lbnRzLCByZXN1bHRzLCBzY2FsZSwgdW5pdCkKCiAgICBpZiB2ZXJib3NlOgogICAgICAgIHByaW50KCJcbldyb3RlICIgKyBjc3ZfcGF0aCkKICAgICAgICBwcmludCgiV3JvdGUgIiArIHBuZ19wYXRoKQogICAgIyBBbmQgZWFjaCBwYXJ0IGFnYWluIG9uIGEgc2hlZXQgb2YgaXRzIG93biwgZnVsbCBzaXplLiBUaGUgY29tYmluZWQgb25lCiAgICAjIGFib3ZlIGlzIHVudG91Y2hlZDsgdGhlc2UgYXJlIGluIGFkZGl0aW9uIHRvIGl0LgogICAgZXh0cmFfcGF0aHMgPSBbXQogICAgZm9yIGksICh0cmVhdCwgcmVzKSBpbiBlbnVtZXJhdGUoemlwKHRyZWF0bWVudHMsIHJlc3VsdHMpKToKICAgICAgICBzbHVnID0gJycuam9pbihjIGlmIGMuaXNhbG51bSgpIGVsc2UgJ18nIGZvciBjIGluIHRyZWF0WydsYWJlbCddKQogICAgICAgIHNsdWcgPSAnXycuam9pbihwIGZvciBwIGluIHNsdWcuc3BsaXQoJ18nKSBpZiBwKS5sb3dlcigpCiAgICAgICAgb25lX3BhdGggPSAne31fcHJvZmlsZV97fS5wbmcnLmZvcm1hdChvdXRfcHJlZml4LCBzbHVnKQogICAgICAgIG9uZSA9IG9uZV90cmVhdG1lbnRfZmlndXJlKHRyZWF0LCByZXMsIHVuaXQsIHRpdGxlLCBzY2FsZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjb2xvdXJfaW5kZXg9aSkKICAgICAgICBvbmUuc2F2ZWZpZyhvbmVfcGF0aCwgZHBpPTE1MCwgYmJveF9pbmNoZXM9J3RpZ2h0JykKICAgICAgICBwbHQuY2xvc2Uob25lKQogICAgICAgIGV4dHJhX3BhdGhzLmFwcGVuZChvbmVfcGF0aCkKICAgICAgICBleHRyYV9wYXRocy5leHRlbmQod3JpdGVfdHJlYXRtZW50X3dvcmtib29rKAogICAgICAgICAgICBvbmVfcGF0aFs6LTRdICsgJy5jc3YnLCB0cmVhdCwgcmVzLCBzY2FsZSwgdW5pdCkpCiAgICBvdmVyX3BhdGggPSBvdXRfcHJlZml4ICsgJ19wcm9maWxlX2FsbF9vdmVybGFpZC5wbmcnCiAgICBvdmVyID0gb3ZlcmxheV9maWd1cmUodHJlYXRtZW50cywgcmVzdWx0cywgdW5pdCwgdGl0bGUsIHNjYWxlKQogICAgb3Zlci5zYXZlZmlnKG92ZXJfcGF0aCwgZHBpPTE1MCwgYmJveF9pbmNoZXM9J3RpZ2h0JykKICAgIHBsdC5jbG9zZShvdmVyKQogICAgZXh0cmFfcGF0aHMuYXBwZW5kKG92ZXJfcGF0aCkKICAgIGV4dHJhX3BhdGhzLmV4dGVuZCh3cml0ZV9vdmVybGF5X3dvcmtib29rKAogICAgICAgIG92ZXJfcGF0aFs6LTRdICsgJy5jc3YnLCB0cmVhdG1lbnRzLCByZXN1bHRzLCBzY2FsZSwgdW5pdCkpCiAgICBleHRyYV9wYXRoc1s6MF0gPSBjb21iaW5lZF9ib29rcwogICAgaWYgdmVyYm9zZToKICAgICAgICBmb3IgcCBpbiBleHRyYV9wYXRoczoKICAgICAgICAgICAgcHJpbnQoJ1dyb3RlICcgKyBwKQogICAgcmV0dXJuIHRyZWF0bWVudHMsIHJlc3VsdHMsIHR1cGxlKFtjc3ZfcGF0aCwgcG5nX3BhdGhdICsgZXh0cmFfcGF0aHMpCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIGJvcnJvd2VkIGZyb20gSElJX0Zyb21fSW1hZ2VzLnB5CiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KCm1hdHBsb3RsaWIudXNlKCJBZ2ciKQoKd2FybmluZ3MuZmlsdGVyd2FybmluZ3MoImlnbm9yZSIpCgpwYXNzICAjIGZyb20gcmFkaWFsX3Byb2ZpbGVzLCBpbmxpbmVkIGFib3ZlCgpwYXNzICAjIGZyb20gaW50ZXJhY3RpdmVfaW5wdXQsIGlubGluZWQgYWJvdmUKCnBsYW5fZm9sZGVyID0gcGxhbjsgZGVzY3JpYmVfZm9sZGVyID0gZGVzY3JpYmVfX2ZyYW1lX2ludmVudG9yeSAgIyBmcm9tIGZyYW1lX2ludmVudG9yeSwgaW5saW5lZCBhYm92ZQoKcGxhbl9mb2xkZXIgPSBwbGFuOyBkZXNjcmliZV9mb2xkZXIgPSBkZXNjcmliZV9fZnJhbWVfaW52ZW50b3J5ICAjIGZyb20gZnJhbWVfaW52ZW50b3J5LCBpbmxpbmVkIGFib3ZlCgpwYXNzICAjIGZyb20gZ2FsYXh5X2NhdGFsb2d1ZSwgaW5saW5lZCBhYm92ZQoKcGFzcyAgIyBmcm9tIGRlcHJvamVjdGlvbiwgaW5saW5lZCBhYm92ZQoKcGFzcyAgIyBmcm9tIGFuY2hvcl9yZWdpc3RyYXRpb24sIGlubGluZWQgYWJvdmUKCnBhc3MgICMgZnJvbSBnYWxheHlfd2luZG93LCBpbmxpbmVkIGFib3ZlCgpkZXNjcmliZV93aW5kb3cgPSBkZXNjcmliZV9fZ2FsYXh5X3dpbmRvdyAgIyBmcm9tIGdhbGF4eV93aW5kb3csIGlubGluZWQgYWJvdmUKCnBhc3MgICMgZnJvbSBnYWxheHlfZXh0ZW50LCBpbmxpbmVkIGFib3ZlCgpkZXNjcmliZV9leHRlbnQgPSBkZXNjcmliZV9fZ2FsYXh5X2V4dGVudCAgIyBmcm9tIGdhbGF4eV9leHRlbnQsIGlubGluZWQgYWJvdmUKCmRlc2NyaWJlX2FuY2hvcnMgPSBkZXNjcmliZV9fYW5jaG9yX3JlZ2lzdHJhdGlvbiAgIyBmcm9tIGFuY2hvcl9yZWdpc3RyYXRpb24sIGlubGluZWQgYWJvdmUKCmRlZiBkb25vcl9iZXNpZGUocGF0aCwgcm9sZSk6CiAgICAiIiJBIHNvbHZlZCBleHBvc3VyZSBpbiB0aGUgc2FtZSBmb2xkZXIgYXMgYHBhdGhgLCB3aGVuIGBwYXRoYCBoYXMgbm8gc29sdXRpb24uCgogICAgUmV0dXJucyBpdHMgcGF0aCwgb3IgTm9uZSB3aGVuIGBwYXRoYCBpcyBzb2x2ZWQgYWxyZWFkeSBvciBub3RoaW5nIGJlc2lkZSBpdAogICAgaXMuIFNhbWUgZmlsdGVyIGZpcnN0LCB0aGVuIGFueSBzb2x2ZWQgZnJhbWUgb2YgdGhlIGZpZWxkLgogICAgIiIiCiAgICB0cnk6CiAgICAgICAgcmVjID0gaW5zcGVjdChwYXRoKQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICByZXR1cm4gTm9uZQogICAgaWYgcmVjLmdldCgid2NzIik6CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIHJlY1sicm9sZSJdID0gcmVjLmdldCgicm9sZSIpIG9yIHJvbGUKICAgIGhlcmUgPSBvcy5wYXRoLmFic3BhdGgocGF0aCkKICAgIG90aGVycyA9IFtyIGZvciByIGluIHNjYW5fZm9sZGVyKG9zLnBhdGguZGlybmFtZShoZXJlKSwgcmVjdXJzaXZlPUZhbHNlKQogICAgICAgICAgICAgIGlmIG9zLnBhdGguYWJzcGF0aChyWyJwYXRoIl0pICE9IGhlcmUgYW5kIG5vdCByLmdldCgiZXJyb3IiKV0KICAgIGRvbm9yID0gZG9ub3JfZm9yKHJlYywgb3RoZXJzKQogICAgcmV0dXJuIGRvbm9yWyJwYXRoIl0gaWYgZG9ub3IgZWxzZSBOb25lCgpkZWYgZnJhbWVzX2Zyb21fZm9sZGVyKGZvbGRlciwgcm9sZXMpOgogICAgIiIiV29yayBvdXQgd2hpY2ggZmlsZSBpbiBhIGZvbGRlciBwbGF5cyBlYWNoIHBhcnQsIGFuZCBzYXkgc28gYmVmb3JlIHN0YXJ0aW5nLgoKICAgIFJldHVybnMgdGhyZWUgdGhpbmdzIHBlciByb2xlOiB0aGUgY2hvc2VuIGZyYW1lLCB0aGUgc29sdmVkIGV4cG9zdXJlIHRoYXQgd2lsbAogICAgbGVuZCBpdCBhIHBsYXRlIHNvbHV0aW9uIGlmIGl0IGhhcyBub25lLCBhbmQgdGhlIGxpc3Qgb2YgZXhwb3N1cmVzIHRvIGNvbWJpbmUKICAgIHdoZXJlIHRoZSBmb2xkZXIgaGVsZCBzZXZlcmFsIGFuZCBub3RoaW5nIGNvbWJpbmVkLgogICAgIiIiCiAgICBkZWYgcGljayhyb2xlLCBvcHRpb25zKToKICAgICAgICBwcmludCgiIikKICAgICAgICBwcmludCgiTW9yZSB0aGFuIG9uZSBmcmFtZSBjb3VsZCBiZSB0aGUgIiArIHJvbGUgKyAiOiIpCiAgICAgICAgZm9yIGksIG9wdCBpbiBlbnVtZXJhdGUob3B0aW9ucywgMSk6CiAgICAgICAgICAgIHByaW50KCIgIHt9LiB7fSIuZm9ybWF0KGksIG9wdFsibmFtZSJdKSkKICAgICAgICB3aGlsZSBUcnVlOgogICAgICAgICAgICBhbnN3ZXIgPSBhc2soIldoaWNoIG9uZSIsICIxIiwgaW50KQogICAgICAgICAgICBpZiAxIDw9IGFuc3dlciA8PSBsZW4ob3B0aW9ucyk6CiAgICAgICAgICAgICAgICByZXR1cm4gb3B0aW9uc1thbnN3ZXIgLSAxXQogICAgICAgICAgICBwcmludCgiICAgIHBpY2sgYSBudW1iZXIgZnJvbSB0aGUgbGlzdCIpCgogICAgcmVjb3JkcywgY2hvc2VuLCBtZW1iZXJzLCBkb25vcnMsIG5vdGVzID0gcGxhbl9mb2xkZXIoZm9sZGVyLCByb2xlcz1yb2xlcywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFzaz1waWNrKQogICAgcHJpbnQoZGVzY3JpYmVfZm9sZGVyKHJlY29yZHMsIGNob3Nlbiwgbm90ZXMpKQoKICAgIG1pc3NpbmcgPSBbciBmb3IgciBpbiByb2xlcyBpZiByIG5vdCBpbiBjaG9zZW5dCiAgICBpZiBtaXNzaW5nOgogICAgICAgIHJhaXNlIFN5c3RlbUV4aXQoIk5vdGhpbmcgaW4gdGhlIGZvbGRlciBpcyB0aGUgIiArICIgb3IgdGhlICIuam9pbihtaXNzaW5nKSkKCiAgICAjIFdoZXJlIHRoZSBmb2xkZXIgaG9sZHMgc2V2ZXJhbCBleHBvc3VyZXMgb2YgYSBmaWx0ZXIgYW5kIG5vdGhpbmcgY29tYmluZWQsCiAgICAjIHNheSB3aGF0IGNvbWJpbmluZyB0aGVtIGlzIHdvcnRoIGFuZCBsZXQgdGhlIHBlcnNvbiBkZWNpZGUuIFRoZSBnYWluIGlzIGEKICAgICMgcmVhbCBudW1iZXIsIHNvIGl0IGlzIHB1dCBvbiB0aGUgc2NyZWVuIHJhdGhlciB0aGFuIGhpZGRlbiBiZWhpbmQgYSBmbGFnCiAgICAjIG5vYm9keSB3aWxsIHJlbWVtYmVyIGEgeWVhciBmcm9tIG5vdy4KICAgIHRvX2NvbWJpbmUgPSB7fQogICAgZm9yIHJvbGUgaW4gcm9sZXM6CiAgICAgICAgZ3JvdXAgPSBtZW1iZXJzLmdldChyb2xlKSBvciBbXQogICAgICAgIGlmIGxlbihncm91cCkgPCAyOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGlmIGFza195ZXNfbm8oIlRoZXJlIGFyZSB7fSB7fSBleHBvc3VyZXMgYW5kIG5vIGNvbWJpbmVkIGZyYW1lLiBDb21iaW5pbmcgIgogICAgICAgICAgICAgICAgICAgICAgInRoZW0gbG93ZXJzIHRoZSBub2lzZSBieSBhYm91dCBhIGZhY3RvciBvZiB7Oi4xZn0uICIKICAgICAgICAgICAgICAgICAgICAgICJDb21iaW5lPyIuZm9ybWF0KGxlbihncm91cCksIHJvbGUsIGxlbihncm91cCkgKiogMC41KSwKICAgICAgICAgICAgICAgICAgICAgIFRydWUpOgogICAgICAgICAgICB0b19jb21iaW5lW3JvbGVdID0gW3JbInBhdGgiXSBmb3IgciBpbiBncm91cF0KICAgICAgICBlbHNlOgogICAgICAgICAgICBwcmludCgiICB1c2luZyB7fSBvbiBpdHMgb3duIi5mb3JtYXQoZ3JvdXBbMF1bIm5hbWUiXSkpCgogICAgbmFtZSwgbmFtZV9zb3VyY2UgPSBnYWxheHlfZnJvbV9yZWNvcmRzKHJlY29yZHMsIGZvbGRlcikKICAgIGlmIG5hbWU6CiAgICAgICAgcHJpbnQoIkdhbGF4eToge30gICAoZnJvbSB7fSkiLmZvcm1hdChuYW1lLCBuYW1lX3NvdXJjZSkpCgogICAgcmV0dXJuICh7cm9sZTogY2hvc2VuW3JvbGVdWyJwYXRoIl0gZm9yIHJvbGUgaW4gcm9sZXN9LAogICAgICAgICAgICB7cm9sZTogKGRvbm9yc1tyb2xlXVsicGF0aCJdIGlmIGRvbm9ycy5nZXQocm9sZSkgZWxzZSBOb25lKQogICAgICAgICAgICAgZm9yIHJvbGUgaW4gcm9sZXN9LAogICAgICAgICAgICB0b19jb21iaW5lLCBuYW1lKQoKZGVmIGhlYWRlcl9udW1iZXIoaGVhZGVyLCBrZXlzKToKICAgICIiIlRoZSBmaXJzdCBvZiB0aGVzZSBoZWFkZXIga2V5d29yZHMgdGhhdCBob2xkcyBhIHBvc2l0aXZlIG51bWJlciwgb3IgTm9uZS4iIiIKICAgIGZvciBrZXkgaW4ga2V5czoKICAgICAgICB0cnk6CiAgICAgICAgICAgIHZhbHVlID0gZmxvYXQoaGVhZGVyLmdldChrZXkpKQogICAgICAgIGV4Y2VwdCAoVHlwZUVycm9yLCBWYWx1ZUVycm9yKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICBpZiB2YWx1ZSA9PSB2YWx1ZSBhbmQgdmFsdWUgPiAwOgogICAgICAgICAgICByZXR1cm4gdmFsdWUKICAgIHJldHVybiBOb25lCgpkZWYgbG9hZF9mcmFtZShwYXRoLCBkb25vcl9wYXRoPU5vbmUpOgogICAgIiIiT3BlbiBhIEZJVFMgZmlsZSBhbmQgcmV0dXJuIHRoZSBzY2llbmNlIGltYWdlLCBpdHMgV0NTIGFuZCBhIGJhZC1waXhlbCBtYXNrLgoKICAgIEEgZnJhbWUgdGhhdCBjYW1lIG91dCBvZiBhIG1lZGlhbiBjb21iaW5lIHVzdWFsbHkgaGFzIG5vIHBsYXRlIHNvbHV0aW9uIGxlZnQKICAgIGluIGl0LCBiZWNhdXNlIHRoZSBjb21iaW5pbmcgc3RlcCB3cm90ZSBhIG5ldyBmaWxlIGFuZCBrZXB0IGFsbW9zdCBub3RoaW5nLgogICAgV2hlbiBgZG9ub3JfcGF0aGAgbmFtZXMgYSBzb2x2ZWQgZXhwb3N1cmUgb2YgdGhlIHNhbWUgZmllbGQsIGEgc29sdXRpb24gaXMKICAgIGZpdHRlZCBmb3IgdGhpcyBmcmFtZSBieSBtYXRjaGluZyBzdGFyIHBhdHRlcm5zIGFnYWluc3QgdGhhdCBvbmUsIGFuZCB0aGUKICAgIHJlc2lkdWFsIG9mIHRoYXQgZml0IGlzIHByaW50ZWQgc28gaXQgY2FuIGJlIGp1ZGdlZCByYXRoZXIgdGhhbiB0cnVzdGVkLgogICAgIiIiCiAgICBmcm9tIGFzdHJvcHkuaW8gaW1wb3J0IGZpdHMKICAgIGZyb20gYXN0cm9weS53Y3MgaW1wb3J0IFdDUwoKICAgIGhkdWwgPSBmaXRzLm9wZW4ocGF0aCkKICAgIHNjaSA9IE5vbmUKICAgIGZvciBoZHUgaW4gaGR1bDoKICAgICAgICBpZiBoZHUuZGF0YSBpcyBub3QgTm9uZSBhbmQgZ2V0YXR0cihoZHUuZGF0YSwgIm5kaW0iLCAwKSA9PSAyOgogICAgICAgICAgICBzY2kgPSBoZHUKICAgICAgICAgICAgYnJlYWsKICAgIGlmIHNjaSBpcyBOb25lOgogICAgICAgIGhkdWwuY2xvc2UoKQogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IocGF0aCArICI6IG5vIDItRCBpbWFnZSBmb3VuZCIpCgogICAgIyBTaW5nbGUgcHJlY2lzaW9uIG9uIHB1cnBvc2UuIENhbWVyYSBkYXRhIGFycml2ZXMgYXMgMTYtYml0IGludGVnZXJzLCBzbwogICAgIyBkb3VibGUgcHJlY2lzaW9uIGRvdWJsZXMgdGhlIG1lbW9yeSBmb3Igbm8gaW5mb3JtYXRpb24gYXQgYWxsOiBzZXZlbgogICAgIyBzaWduaWZpY2FudCBmaWd1cmVzIGFnYWluc3QgcGl4ZWwgbm9pc2Ugb2YgdGVucyBvZiBjb3VudHMuIE9uIGEgNjEtbWVnYXBpeGVsCiAgICAjIGZyYW1lIHRoZSBkaWZmZXJlbmNlIGlzIDI0NSBNQiBhZ2FpbnN0IDQ4OSwgYW5kIGZvdXIgZmlsdGVycyBvZiB0aG9zZSB3YXMKICAgICMgd2hhdCBkcm92ZSB0aGlzIG1hY2hpbmUgaW50byBzd2FwcGluZy4KICAgIGRhdGEgPSBucC5hc2FycmF5KHNjaS5kYXRhLCBkdHlwZT1ucC5mbG9hdDMyKQogICAgaGVhZGVyID0gc2NpLmhlYWRlcgoKICAgIGJwbSA9IE5vbmUKICAgIGZvciBoZHUgaW4gaGR1bDoKICAgICAgICBpZiBoZHUubmFtZS51cHBlcigpID09ICJCUE0iIGFuZCBoZHUuZGF0YSBpcyBub3QgTm9uZToKICAgICAgICAgICAgYnBtID0gbnAuYXNhcnJheShoZHUuZGF0YSkgPiAwCiAgICAgICAgICAgIGJyZWFrCgogICAgd2NzID0gTm9uZQogICAgdHJ5OgogICAgICAgIHcgPSBXQ1MoaGVhZGVyKQogICAgICAgIGlmIHcuaGFzX2NlbGVzdGlhbDoKICAgICAgICAgICAgd2NzID0gdy5jZWxlc3RpYWwKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcGFzcwoKICAgIGluZm8gPSB7CiAgICAgICAgImZpbHRlciI6IHN0cihoZWFkZXIuZ2V0KCJGSUxURVIiLCAiPyIpKSwKICAgICAgICAiZXhwdGltZSI6IGZsb2F0KGhlYWRlci5nZXQoIkVYUFRJTUUiLCAxLjApIG9yIDEuMCksCiAgICAgICAgInBpeGVsX3NjYWxlIjogaGVhZGVyLmdldCgiUElYU0NBTEUiLCBOb25lKSwKICAgICAgICAib2JqZWN0Ijogc3RyKGhlYWRlci5nZXQoIk9CSkVDVCIsICI/IikpLAogICAgICAgICMgV2hhdCB0aGUgcGhvdG9tZXRyeSBuZWVkcyB0byBrbm93IGFib3V0IHRoZSBkZXRlY3RvciwgY2FycmllZCBhbG9uZwogICAgICAgICMgc28gdGhhdCBhIGZyYW1lIHRoaXMgcGlwZWxpbmUgd3JpdGVzIHN0aWxsIHNheXMgaXQuCiAgICAgICAgInNhdHVyYXRlIjogaGVhZGVyX251bWJlcihoZWFkZXIsICgiU0FUVVJBVEUiLCAiTUFYTElOIiwgIlNBVExFVkVMIikpLAogICAgICAgICMgRWxlY3Ryb25zIHBlciBBRFUuIEVHQUlOIGZpcnN0OiBvbiBzb21lIGNhbWVyYXMgR0FJTiBpcyB0aGUgZ2FpbgogICAgICAgICMgU0VUVElORyAoNTEgb24gdGhlIEtpbm5lcmV0IGNvbG91ciBjYW1lcmEpLCBub3QgZWxlY3Ryb25zIHBlciBBRFUuCiAgICAgICAgImdhaW4iOiBuZXh0KChnIGZvciBnIGluIChoZWFkZXJfbnVtYmVyKGhlYWRlciwgKGssKSkgZm9yIGsgaW4KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICgiRUdBSU4iLCAiR0FJTiIpKSBpZiBnIGFuZCBnIDwgMjApLCBOb25lKSwKICAgICAgICAicmRub2lzZSI6IGhlYWRlcl9udW1iZXIoaGVhZGVyLCAoIlJETk9JU0UiLCAiUkVBRE5PSVMiLCAiUk9OIikpLAogICAgICAgICJuY29tYmluZSI6IGludChoZWFkZXJfbnVtYmVyKGhlYWRlciwgKCJOQ09NQklORSIsKSkgb3IgMSksCiAgICAgICAgImNvbWJ0eXBlIjogc3RyKGhlYWRlci5nZXQoIkNPTUJUWVBFIiwgIiIpKSwKICAgICAgICAiZGF0ZSI6IHN0cihoZWFkZXIuZ2V0KCJEQVRFLU9CUyIsICIiKSksCiAgICB9CiAgICBoZHVsLmNsb3NlKCkKCiAgICBpZiB3Y3MgaXMgTm9uZSBhbmQgZG9ub3JfcGF0aDoKICAgICAgICByZWZlcmVuY2UsIHJlZl93Y3MsIF8sIHJlZl9pbmZvID0gbG9hZF9mcmFtZShkb25vcl9wYXRoKQogICAgICAgIGlmIHJlZl93Y3MgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHdjcywgZGlhZyA9IHdjc19mcm9tX3JlZmVyZW5jZShkYXRhLCByZWZlcmVuY2UsIHJlZl93Y3MpCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOgogICAgICAgICAgICAgICAgIyBhc3Ryb2FsaWduIGZpbmRzIG5vIHN0YXIgcGF0dGVybiBpbiBjb21tb24gLSB0eXBpY2FsbHkgYSBmcmFtZQogICAgICAgICAgICAgICAgIyB3aXRoIHRoZSBzdGFycyB0YWtlbiBvdXQsIHN1Y2ggYXMgSC1hbHBoYSBtaW51cyBjb250aW51dW0uCiAgICAgICAgICAgICAgICBwcmludCgiICB7fTogY291bGQgbm90IGZpdCBhIHBsYXRlIHNvbHV0aW9uIGFnYWluc3Qge30gKHt9OiB7fSkiLmZvcm1hdCgKICAgICAgICAgICAgICAgICAgICBvcy5wYXRoLmJhc2VuYW1lKHBhdGgpLCBvcy5wYXRoLmJhc2VuYW1lKGRvbm9yX3BhdGgpLAogICAgICAgICAgICAgICAgICAgIHR5cGUoZXhjKS5fX25hbWVfXywgZXhjKSwgZmx1c2g9VHJ1ZSkKICAgICAgICAgICAgICAgIHJldHVybiBkYXRhLCBOb25lLCBicG0sIGluZm8KICAgICAgICAgICAgcHJpbnQoZGVzY3JpYmVfd2NzX2ZpdChvcy5wYXRoLmJhc2VuYW1lKHBhdGgpLCBkaWFnKSkKICAgICAgICAgICAgaWYgaW5mb1sicGl4ZWxfc2NhbGUiXSBpcyBOb25lOgogICAgICAgICAgICAgICAgaW5mb1sicGl4ZWxfc2NhbGUiXSA9IHJlZl9pbmZvWyJwaXhlbF9zY2FsZSJdCiAgICAgICAgICAgIGlmIGluZm9bImZpbHRlciJdID09ICI/IjoKICAgICAgICAgICAgICAgIGluZm9bImZpbHRlciJdID0gcmVmX2luZm9bImZpbHRlciJdCiAgICAgICAgICAgICMgQSBjb21iaW5lZCBmcmFtZSB3cml0dGVuIGJ5IGFub3RoZXIgcHJvZ3JhbSBvZnRlbiBrZWVwcyBubyBoZWFkZXIKICAgICAgICAgICAgIyBhdCBhbGwuIFRoZSBjYW1lcmEgYW5kIHRoZSBleHBvc3VyZSBsZW5ndGggYXJlIHRob3NlIG9mIHRoZSBzb2x2ZWQKICAgICAgICAgICAgIyBleHBvc3VyZSBpdCBjYW1lIGZyb20sIHNvIHRoZXkgYXJlIHRha2VuIGZyb20gdGhlcmUsIGFuZCBzYWlkIHNvLgogICAgICAgICAgICBib3Jyb3dlZCA9IFtdCiAgICAgICAgICAgIGZvciBrZXksIGNhcmQgaW4gKCgiZXhwdGltZSIsICJFWFBUSU1FIiksICgic2F0dXJhdGUiLCAiU0FUVVJBVEUiKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKCJnYWluIiwgIkdBSU4iKSwgKCJyZG5vaXNlIiwgIlJETk9JU0UiKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKCJkYXRlIiwgIkRBVEUtT0JTIiksICgib2JqZWN0IiwgIk9CSkVDVCIpKToKICAgICAgICAgICAgICAgIGlmIGNhcmQgaW4gaGVhZGVyIG9yIChrZXkgaW4gKCJzYXR1cmF0ZSIsICJnYWluIiwgInJkbm9pc2UiKQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFuZCBpbmZvW2tleV0gaXMgbm90IE5vbmUpOgogICAgICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgICAgICBpZiBrZXkgPT0gIm9iamVjdCIgYW5kIGluZm9ba2V5XSAhPSAiPyI6CiAgICAgICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgICAgIGlmIHJlZl9pbmZvLmdldChrZXkpIG5vdCBpbiAoTm9uZSwgIiIsICI/Iik6CiAgICAgICAgICAgICAgICAgICAgaW5mb1trZXldID0gcmVmX2luZm9ba2V5XQogICAgICAgICAgICAgICAgICAgIGJvcnJvd2VkLmFwcGVuZChjYXJkKQogICAgICAgICAgICBpZiBib3Jyb3dlZDoKICAgICAgICAgICAgICAgIHByaW50KCIgIHt9OiBubyB7fSBpbiBpdHMgaGVhZGVyOyB0YWtlbiBmcm9tIHt9Ii5mb3JtYXQoCiAgICAgICAgICAgICAgICAgICAgb3MucGF0aC5iYXNlbmFtZShwYXRoKSwgIiwgIi5qb2luKGJvcnJvd2VkKSwKICAgICAgICAgICAgICAgICAgICBvcy5wYXRoLmJhc2VuYW1lKGRvbm9yX3BhdGgpKSwgZmx1c2g9VHJ1ZSkKCiAgICByZXR1cm4gZGF0YSwgd2NzLCBicG0sIGluZm8KCmRlZiBfY2VudHJvaWRfbG9jYWxseShkYXRhLCBwb3NpdGlvbnMsIGJveCk6CiAgICAiIiJDZW50cmUgb2YgbGlnaHQgaW4gYSBzbWFsbCBib3gsIHdpdGggdGhhdCBib3gncyBvd24gc2t5IHRha2VuIG9mZiBmaXJzdC4KCiAgICBUaGUgb25lIHBsYWNlIHRoaXMgaXMgZG9uZSwgYmVjYXVzZSBnZXR0aW5nIGl0IHdyb25nIGlzIGludmlzaWJsZS4gVGhlIHNreSBpcwogICAgcGFydCBvZiB0aGUgd2VpZ2h0IGluIGEgY2VudHJlIG9mIGxpZ2h0LCBhbmQgaW4gYSBib3ggb2YgYSBjb3VwbGUgb2YgaHVuZHJlZAogICAgcGl4ZWxzIGl0IGlzIG1vc3Qgb2YgdGhlIHdlaWdodCwgc28gbGVhdmluZyBpdCBpbiByZXR1cm5zIHRoZSBtaWRkbGUgb2YgdGhlIGJveAogICAgZHJlc3NlZCB1cCBhcyBhIG1lYXN1cmVtZW50LiBFdmVyeSBjYWxsZXIgaGVyZSB3YW50cyB0aGUgc3Rhciwgbm90IHRoZSBib3guCgogICAgUmV0dXJucyBhbiBhcnJheSB0aGUgc2FtZSBsZW5ndGggYXMgYHBvc2l0aW9uc2AsIHdpdGggTmFOIHdoZXJlIGEgcG9zaXRpb24gc2l0cwogICAgdG9vIG5lYXIgYW4gZWRnZSBvciB0aGUgc3RhbXAgaG9sZHMgbm90aGluZyBhYm92ZSBpdHMgb3duIHNreS4KICAgICIiIgogICAgZGF0YSA9IG5wLmFzYXJyYXkoZGF0YSkKICAgIHBvcyA9IG5wLmFzYXJyYXkocG9zaXRpb25zLCBkdHlwZT1mbG9hdCkKICAgIGhhbGYgPSBib3ggLy8gMgogICAgbnksIG54ID0gZGF0YS5zaGFwZQogICAgZ3JpZF95LCBncmlkX3ggPSBucC5tZ3JpZFswOmJveCwgMDpib3hdCiAgICBvdXQgPSBucC5mdWxsKHBvcy5zaGFwZSwgbnAubmFuKQogICAgZm9yIGksICh4YywgeWMpIGluIGVudW1lcmF0ZShwb3MpOgogICAgICAgIGlmIG5vdCAobnAuaXNmaW5pdGUoeGMpIGFuZCBucC5pc2Zpbml0ZSh5YykpOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHhpLCB5aSA9IGludChyb3VuZCh4YykpLCBpbnQocm91bmQoeWMpKQogICAgICAgIGlmIG5vdCAoaGFsZiA8PSB4aSA8IG54IC0gaGFsZiBhbmQgaGFsZiA8PSB5aSA8IG55IC0gaGFsZik6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgc3RhbXAgPSBucC5uYW5fdG9fbnVtKAogICAgICAgICAgICBkYXRhW3lpIC0gaGFsZjp5aSArIGhhbGYgKyAxLCB4aSAtIGhhbGY6eGkgKyBoYWxmICsgMV0pLmFzdHlwZShmbG9hdCkKICAgICAgICBpZiBzdGFtcC5zaGFwZSAhPSAoYm94LCBib3gpOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHN0YW1wID0gc3RhbXAgLSBucC5tZWRpYW4oc3RhbXApCiAgICAgICAgc3RhbXBbc3RhbXAgPCAwLjBdID0gMC4wICAgICAgICAjIG5lZ2F0aXZlIG5vaXNlIG11c3Qgbm90IHB1bGwgdGhlIGNlbnRyZQogICAgICAgIHRvdGFsID0gc3RhbXAuc3VtKCkKICAgICAgICBpZiB0b3RhbCA8PSAwLjA6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgb3V0W2ldID0gW3hpIC0gaGFsZiArIChzdGFtcCAqIGdyaWRfeCkuc3VtKCkgLyB0b3RhbCwKICAgICAgICAgICAgICAgICAgeWkgLSBoYWxmICsgKHN0YW1wICogZ3JpZF95KS5zdW0oKSAvIHRvdGFsXQogICAgcmV0dXJuIG91dAoKZGVmIGNlbnRyZV9vbl9zdGFycyhkYXRhLCBwb3NpdGlvbnMsIGJveD0xNSwgbWF4X21vdmVfcHg9Ni4wKToKICAgICIiIk1vdmUgY2F0YWxvZ3VlIHBvc2l0aW9ucyBvbnRvIHRoZSBzdGFycyBhcyB0aGlzIGZyYW1lIGFjdHVhbGx5IHNob3dzIHRoZW0uCgogICAgV0hZIFRISVMgSVMgTkVFREVECiAgICBBIGNhdGFsb2d1ZSBwb3NpdGlvbiBhbmQgYSBzdGFyIGluIGEgZnJhbWUgYXJlIG5vdCB0aGUgc2FtZSB0aGluZyB0byBiZXR0ZXIKICAgIHRoYW4gYSBwaXhlbCBvciB0d28uIFRoZSBwbGF0ZSBzb2x1dGlvbiBoYXMgaXRzIG93biBlcnJvciwgdGhlIGNhdGFsb2d1ZSBpcyBhdAogICAgYSBkaWZmZXJlbnQgZXBvY2gsIGFuZCBhIGNvbWJpbmVkIGZyYW1lIGNhcnJpZXMgd2hhdGV2ZXIgaXRzIHJlZ2lzdHJhdGlvbiBsZWZ0CiAgICBiZWhpbmQuIE1lYXN1cmVkIG9uIGFuIE0zMyBSIGZyYW1lOiB0aGUgcmVhbCBzdGFycyBzYXQgYSBtZWRpYW4gb2YgMi40IHBpeGVscwogICAgYXdheSBmcm9tIHdoZXJlIEdhaWEgc2FpZCB0aGV5IHdlcmUsIGFuZCBub3QgYnkgYSBjb25zdGFudCBhbW91bnQgLSB0aGUgc3ByZWFkCiAgICB3YXMgMi4yIHBpeGVscy4KCiAgICBUaGF0IG1hdHRlcnMgYmVjYXVzZSBtb3N0IG9mIHdoYXQgaXMgZG9uZSB3aXRoIHRoZXNlIHBvc2l0aW9ucyBpcyBhcGVydHVyZQogICAgcGhvdG9tZXRyeSwgYW5kIGFuIGFwZXJ0dXJlIGlzIG5vdCBtb3ZlZCBieSBiZWluZyB3cm9uZy4gQSBjaXJjbGUgb2YgcmFkaXVzIDYKICAgIHBsYWNlZCAyLjQgcGl4ZWxzIG9mZiBjZW50cmUgbWlzc2VzIGEgZ29vZCBwYXJ0IG9mIHRoZSBzdGFyLCBhbmQgbWlzc2VzIGEKICAgIGRpZmZlcmVudCBwYXJ0IG9mIGV2ZXJ5IHN0YXIuIFRoZSBzdWJ0cmFjdGlvbiBmYWN0b3IgYnVpbHQgZnJvbSB0aG9zZSBmbHV4ZXMKICAgIGlzIHdyb25nLCBhbmQgdGhlIHN0YXIgd2lkdGhzIG1lYXN1cmVkIGFyb3VuZCB0aG9zZSBwb2ludHMgYXJlIG5vdCB0aGUgd2lkdGhzCiAgICBvZiBzdGFycy4gQm90aCBvZiB0aG9zZSB3ZXJlIHdyb25nIGhlcmUsIGFuZCBib3RoIHNob3dlZCBpbiB0aGUgcGljdHVyZS4KCiAgICBUaGUgb25lIHRoaW5nIHRoYXQgd2FzIHJpZ2h0IHdhcyB0aGUgZnJhbWUgcmVnaXN0cmF0aW9uLCBhbmQgb25seSBiZWNhdXNlIGl0CiAgICBjZW50cm9pZHMgaW5zaWRlIGEgYm94IGJlZm9yZSBmaXR0aW5nIC0gd2hpY2ggaXMgZXhhY3RseSB3aGF0IHRoaXMgZG9lcywgb25jZSwKICAgIGZvciBldmVyeW9uZSBlbHNlLgoKICAgIFdIWSBUSEUgU0tZIElTIFRBS0VOIE9GRiBGSVJTVAogICAgQSBjZW50cmUgb2YgbGlnaHQgaXMgYSB3ZWlnaHRlZCBhdmVyYWdlLCBhbmQgdGhlIHNreSBpcyBwYXJ0IG9mIHRoZSB3ZWlnaHQuIEluCiAgICBhIDE1IHBpeGVsIGJveCBvbiBhIGZyYW1lIHdoZXJlIHRoZSBza3kgc2l0cyBhdCBhIGZldyBodW5kcmVkIGNvdW50cywgdGhlIHR3bwogICAgaHVuZHJlZCBza3kgcGl4ZWxzIG91dHdlaWdoIHRoZSBzdGFyLCBhbmQgdGhlaXIgY2VudHJlIG9mIGxpZ2h0IGlzIHRoZSBtaWRkbGUKICAgIG9mIHRoZSBib3guIFNvIHRoZSBhbnN3ZXIgY29tZXMgYmFjayBhcyB2ZXJ5IG5lYXJseSB0aGUgcG9zaXRpb24gaXQgd2FzIGdpdmVuLAogICAgd2hhdGV2ZXIgdGhlIHN0YXIgaXMgZG9pbmcgLSBpdCBsb29rcyBsaWtlIGEgc3VjY2Vzc2Z1bCBtZWFzdXJlbWVudCBhbmQgaXMgbm90CiAgICBvbmUuIE1lYXN1cmVkIG9uIE0zMzogY2VudHJvaWRpbmcgdGhlIHJhdyBzdGFtcCBtb3ZlZCB0aGUgcG9zaXRpb25zIDAuNiBweCBhbmQKICAgIGxlZnQgdGhlIHN0YWNrZWQgc3RhciBmbGF0LXRvcHBlZCwgaXRzIGJyaWdodGVzdCByaW5nIGF0IHJhZGl1cyA0OyB0YWtpbmcgdGhlCiAgICBsb2NhbCBza3kgb2ZmIHRoZSBzdGFtcCBmaXJzdCBtb3ZlZCB0aGVtIDQuMyBweCBhbmQgZ2F2ZSBhIHN0YXIgdGhhdCBmYWxscyBmcm9tCiAgICAxMDIgY291bnRzIGF0IHRoZSBjZW50cmUgdG8gMTUgYXQgcmFkaXVzIDcsIHdoaWNoIGlzIHdoYXQgYSBzdGFyIGxvb2tzIGxpa2UuCgogICAgVGhlIHN0YW1wIGlzIGZsb29yZWQgYXQgemVybyBhZnRlciB0aGUgc2t5IGNvbWVzIG9mZiBzbyB0aGF0IG5lZ2F0aXZlIG5vaXNlCiAgICBjYW5ub3QgcHVsbCB0aGUgY2VudHJlIGFyb3VuZC4KCiAgICBQb3NpdGlvbnMgdGhhdCBjYW5ub3QgYmUgY2VudHJlZCwgb3IgdGhhdCBtb3ZlIGZ1cnRoZXIgdGhhbiBhIHBsYXRlIHNvbHV0aW9uCiAgICBwbGF1c2libHkgZXJycywgYXJlIGRyb3BwZWQgcmF0aGVyIHRoYW4gdHJ1c3RlZC4KICAgICIiIgogICAgcG9zaXRpb25zID0gbnAuYXNhcnJheShwb3NpdGlvbnMsIGR0eXBlPWZsb2F0KQogICAgaWYgbm90IGxlbihwb3NpdGlvbnMpOgogICAgICAgIHJldHVybiBwb3NpdGlvbnMsIHsibl9pbiI6IDAsICJuX291dCI6IDAsICJ3aHkiOiAibm8gc3RhcnMgdG8gY2VudHJlIn0KCiAgICBueSwgbnggPSBkYXRhLnNoYXBlCiAgICBoYWxmID0gYm94IC8vIDIKICAgIGVkZ2UgPSBib3gKICAgIGluc2lkZSA9ICgocG9zaXRpb25zWzosIDBdID4gZWRnZSkgJiAocG9zaXRpb25zWzosIDBdIDwgbnggLSBlZGdlKSAmCiAgICAgICAgICAgICAgKHBvc2l0aW9uc1s6LCAxXSA+IGVkZ2UpICYgKHBvc2l0aW9uc1s6LCAxXSA8IG55IC0gZWRnZSkpCiAgICBwID0gcG9zaXRpb25zW2luc2lkZV0KICAgIGlmIG5vdCBsZW4ocCk6CiAgICAgICAgcmV0dXJuIG5wLmVtcHR5KCgwLCAyKSksIHsibl9pbiI6IGxlbihwb3NpdGlvbnMpLCAibl9vdXQiOiAwLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIndoeSI6ICJub25lIG9mIHRoZW0gZmFsbCBpbnNpZGUgdGhlIGZyYW1lIn0KCiAgICBmb3VuZCA9IF9jZW50cm9pZF9sb2NhbGx5KGRhdGEsIHAsIGJveCkKCiAgICBkeCwgZHkgPSBmb3VuZFs6LCAwXSAtIHBbOiwgMF0sIGZvdW5kWzosIDFdIC0gcFs6LCAxXQogICAgbW92ZWQgPSBucC5oeXBvdChkeCwgZHkpCiAgICBnb29kID0gbnAuaXNmaW5pdGUobW92ZWQpICYgKG1vdmVkIDwgbWF4X21vdmVfcHgpCiAgICBvdXQgPSBmb3VuZFtnb29kXQogICAgcmV0dXJuIG91dCwgeyJuX2luIjogaW50KGxlbihwb3NpdGlvbnMpKSwgIm5fb3V0IjogaW50KGdvb2Quc3VtKCkpLAogICAgICAgICAgICAgICAgICJtZWRpYW5fbW92ZSI6IGZsb2F0KG5wLm1lZGlhbihtb3ZlZFtnb29kXSkpIGlmIGdvb2QuYW55KCkgZWxzZSAwLjAsCiAgICAgICAgICAgICAgICAgIm1lZGlhbl9keCI6IGZsb2F0KG5wLm1lZGlhbihkeFtnb29kXSkpIGlmIGdvb2QuYW55KCkgZWxzZSAwLjAsCiAgICAgICAgICAgICAgICAgIm1lZGlhbl9keSI6IGZsb2F0KG5wLm1lZGlhbihkeVtnb29kXSkpIGlmIGdvb2QuYW55KCkgZWxzZSAwLjB9CgpkZWYgZGVzY3JpYmVfY2VudHJpbmcoZGlhZyk6CiAgICAiIiJXaGF0IHRoZSBjZW50cmluZyBtb3ZlZCwgaW4gYSBsaW5lLiIiIgogICAgaWYgZGlhZy5nZXQoIndoeSIpOgogICAgICAgIHJldHVybiAiICBjYXRhbG9ndWUgcG9zaXRpb25zIGxlZnQgYXMgdGhleSBhcmUgKHt9KSIuZm9ybWF0KGRpYWdbIndoeSJdKQogICAgcmV0dXJuICgiICB7fSBvZiB7fSBjYXRhbG9ndWUgc3RhcnMgZm91bmQgYW5kIGNlbnRyZWQ7IHRoZXkgc2F0IGEgbWVkaWFuIG9mICIKICAgICAgICAgICAgIns6LjJmfSBweCBmcm9tIHRoZSBjYXRhbG9ndWUgcG9zaXRpb24gKHs6Ky4yZn0sIHs6Ky4yZn0pIi5mb3JtYXQoCiAgICAgICAgICAgICAgICBkaWFnWyJuX291dCJdLCBkaWFnWyJuX2luIl0sIGRpYWdbIm1lZGlhbl9tb3ZlIl0sCiAgICAgICAgICAgICAgICBkaWFnWyJtZWRpYW5fZHgiXSwgZGlhZ1sibWVkaWFuX2R5Il0pKQoKZGVmIGNvbWJpbmVfZXhwb3N1cmVzKHBhdGhzLCBkb25vcl9wYXRoPU5vbmUsIGxhYmVsPSIiLCBiYW5kX3Jvd3M9NTEyLAogICAgICAgICAgICAgICAgICAgICAgcG9zaXRpb25zPU5vbmUpOgogICAgIiIiUHV0IHNldmVyYWwgZXhwb3N1cmVzIG9mIG9uZSBmaWx0ZXIgb250byBvbmUgZ3JpZCBhbmQgdGFrZSB0aGVpciBtZWRpYW4uCgogICAgV0hZIENPTUJJTkUgQVQgQUxMCiAgICBBIG5pZ2h0J3Mgb2JzZXJ2aW5nIG9mIG9uZSBmaWx0ZXIgaXMgc2V2ZXJhbCBleHBvc3VyZXMuIFRha2luZyBvbmUgYW5kCiAgICBkaXNjYXJkaW5nIHRoZSByZXN0IHRocm93cyBhd2F5IG1vc3Qgb2Ygd2hhdCB3YXMgZ2F0aGVyZWQ6IHNpeCBmcmFtZXMgY29tYmluZWQKICAgIGhhdmUgbm9pc2UgbG93ZXIgYnkgcm91Z2hseSB0aGUgc3F1YXJlIHJvb3Qgb2Ygc2l4LCBhIGZhY3RvciBvZiAyLjQsIGFuZCBpdCBpcwogICAgdGhlIGZhaW50ZXN0IHJlZ2lvbnMgLSB0aGUgb25lcyBuZWFyIHRoZSBkZXRlY3Rpb24gbGltaXQsIHdoZXJlIHRoZSBhbnN3ZXIgaXMKICAgIGFjdHVhbGx5IGRlY2lkZWQgLSB0aGF0IGFwcGVhciBvciB2YW5pc2ggb24gZXhhY3RseSB0aGF0IGRpZmZlcmVuY2UuCgogICAgV0hZIEEgTUVESUFOCiAgICBCZWNhdXNlIGl0IGlzIHJvYnVzdCwgY2hlYXAsIGFuZCB0aGUgYWx0ZXJuYXRpdmUgaXMgbm90IHdvcnRoIHdoYXQgaXQgY29zdHMuCiAgICBNZWFzdXJlZCBvbiB0aGlzIGRhdGE6CgogICAgICAgIHNpeCBmcmFtZXMgICAgICAgIG5vaXNlICAgIHRpbWUgICAgIHdvcmtpbmcgbWVtb3J5CiAgICAgICAgbWVkaWFuICAgICAgICAgICAgIDEuMTYgICAgMC40NSBzICAgICAxLjR4IHRoZSBzdGFjawogICAgICAgIGNsaXBwZWQgbWVhbiAgICAgICAxLjEwICAgIDAuOTcgcyAgICAgMy4zeCB0aGUgc3RhY2sKICAgICAgICBwbGFpbiBhdmVyYWdlICAgICAgMi42MCogICAwLjAyIHMgICAgIDAuMnggdGhlIHN0YWNrCiAgICAgICAgICAgICAgICAgICAgICAgICAgICgqIHdpdGggb25lIGNvc21pYyByYXkgaW4gYSBodW5kcmVkIHBpeGVscykKCiAgICBUaGUgY2xpcHBlZCBtZWFuIGJ1eXMgZml2ZSB0byBzZXZlbiBwZXJjZW50IGluIG5vaXNlIC0gYWJvdXQgd2hhdCBhIHNldmVudGgKICAgIGV4cG9zdXJlIHdvdWxkIGdpdmUgLSBmb3IgdHdpY2UgdGhlIHRpbWUgYW5kIHR3byBhbmQgYSBoYWxmIHRpbWVzIHRoZSBtZW1vcnkuCiAgICBBIHBsYWluIGF2ZXJhZ2UgaXMgbm90IGFuIG9wdGlvbiBhdCBhbGw6IG9uZSBjb3NtaWMgcmF5IHJ1aW5zIHRoZSBwaXhlbC4gU28KICAgIHRoZSBtZWRpYW4gaXQgaXMsIGFuZCB0aGVyZSBpcyBubyBzd2l0Y2gsIGJlY2F1c2UgdGhlcmUgaXMgbm8gY2FzZSBoZXJlIHdoZXJlCiAgICB0aGUgb3RoZXIgYW5zd2VyIGlzIHRoZSByaWdodCBvbmUuCgogICAgRG9uZSBhIGJhbmQgb2Ygcm93cyBhdCBhIHRpbWUsIHNvIHRoZSBtZW1vcnkgbmVlZGVkIGRvZXMgbm90IGdyb3cgd2l0aCB0aGUKICAgIG51bWJlciBvZiBleHBvc3VyZXM6IHR3ZW50eSBmcmFtZXMgY29tYmluZSBpbiB0aGUgc2FtZSBzcGFjZSBhcyB0aHJlZS4KCiAgICBUaGUgZmlyc3QgZXhwb3N1cmUgc2V0cyB0aGUgZ3JpZDsgdGhlIHJlc3QgYXJlIGJyb3VnaHQgdG8gaXQgdGhyb3VnaCB0aGVpcgogICAgcGxhdGUgc29sdXRpb25zIGFuZCB0aGVuIGhhdmUgdGhlIGxlZnRvdmVyIHN1Yi1waXhlbCBzaGlmdCBtZWFzdXJlZCBmcm9tIHRoZQogICAgc3RhcnMgYW5kIHRha2VuIG91dCwgZXhhY3RseSBhcyB0aGUgc2VwYXJhdGUgZmlsdGVycyBhcmUuCiAgICAiIiIKICAgIGltcG9ydCBudW1weSBhcyBucAoKICAgIHJlZmVyZW5jZSwgcmVmX3djcywgcmVmX2JwbSwgcmVmX2luZm8gPSBsb2FkX2ZyYW1lKHBhdGhzWzBdLCBkb25vcl9wYXRoKQogICAgaWYgbGVuKHBhdGhzKSA9PSAxOgogICAgICAgIHJldHVybiByZWZlcmVuY2UsIHJlZl93Y3MsIHJlZl9icG0sIHJlZl9pbmZvCgogICAgcHJpbnQoIkNvbWJpbmluZyB7fSB7fSBleHBvc3VyZXMiLmZvcm1hdChsZW4ocGF0aHMpLCBsYWJlbCBvciAiZnJhbWVzIikpCiAgICAjIEV2ZXJ5IGV4cG9zdXJlIGlzIHNjYWxlZCB0byB0aGUgZXhwb3N1cmUgdGltZSBvZiB0aGUgZmlyc3QgYmVmb3JlIHRoZQogICAgIyBtZWRpYW4gaXMgdGFrZW4uIEEgbWVkaWFuIG9mIGEgNjAgcyBhbmQgYSAxMjAgcyBmcmFtZSB0aGF0IHdlcmUgbm90CiAgICAjIHNjYWxlZCBpcyBuZWl0aGVyIG9mIHRoZW06IGVhY2ggcGl4ZWwgdGFrZXMgd2hpY2hldmVyIHZhbHVlIGhhcHBlbnMgdG8KICAgICMgZmFsbCBpbiB0aGUgbWlkZGxlLiBCZWZvcmUgc2NhbGluZywgcGl4ZWxzIGF0IHRoZSBkZXRlY3RvcidzIHNhdHVyYXRpb24KICAgICMgbGV2ZWwgYXJlIGJsYW5rZWQsIHNvIHRoYXQgYSBzdGFyIHNhdHVyYXRlZCBpbiB0aGUgbG9uZyBleHBvc3VyZXMgaXMKICAgICMgbWVhc3VyZWQgZnJvbSB0aGUgc2hvcnQgb25lcy4KICAgIHJlZl9leHB0aW1lID0gZmxvYXQocmVmX2luZm8uZ2V0KCJleHB0aW1lIikgb3IgMS4wKQogICAgc2F0X2xldmVscywgc2F0X21hc2tzID0gW10sIFtdCgogICAgZGVmIHNjYWxlZChmcmFtZSwgaW5mbywgbmFtZSk6CiAgICAgICAgZXhwdGltZSA9IGZsb2F0KGluZm8uZ2V0KCJleHB0aW1lIikgb3IgcmVmX2V4cHRpbWUpCiAgICAgICAgcmF0aW8gPSByZWZfZXhwdGltZSAvIGV4cHRpbWUKICAgICAgICBzYXQgPSBpbmZvLmdldCgic2F0dXJhdGUiKQogICAgICAgIGlmIHNhdDoKICAgICAgICAgICAgZnJhbWUgPSBucC5hcnJheShmcmFtZSwgZHR5cGU9bnAuZmxvYXQzMiwgY29weT1UcnVlKQogICAgICAgICAgICB3aXRoIG5wLmVycnN0YXRlKGludmFsaWQ9Imlnbm9yZSIpOgogICAgICAgICAgICAgICAgaGl0ID0gZnJhbWUgPj0gU0FUVVJBVElPTl9GUkFDVElPTiAqIHNhdAogICAgICAgICAgICBmcmFtZVtoaXRdID0gbnAubmFuCiAgICAgICAgICAgIHNhdF9tYXNrcy5hcHBlbmQoaGl0KQogICAgICAgICAgICBzYXRfbGV2ZWxzLmFwcGVuZChzYXQgKiByYXRpbykKICAgICAgICBlbHNlOgogICAgICAgICAgICBzYXRfbGV2ZWxzLmFwcGVuZChOb25lKQogICAgICAgIGlmIGFicyhyYXRpbyAtIDEuMCkgPiAxZS02OgogICAgICAgICAgICBwcmludCgiICB7fTogezouMGZ9IHMsIHNjYWxlZCBieSB7Oi4zZn0gdG8gezouMGZ9IHMiLmZvcm1hdCgKICAgICAgICAgICAgICAgIG5hbWUsIGV4cHRpbWUsIHJhdGlvLCByZWZfZXhwdGltZSkpCiAgICAgICAgICAgIGZyYW1lID0gZnJhbWUgKiBucC5mbG9hdDMyKHJhdGlvKQogICAgICAgIHJldHVybiBmcmFtZQoKICAgIHN0YWNrID0gW3NjYWxlZChyZWZlcmVuY2UsIHJlZl9pbmZvLCBvcy5wYXRoLmJhc2VuYW1lKHBhdGhzWzBdKSldCiAgICB0b3RhbF9leHB0aW1lID0gZmxvYXQocmVmX2luZm8uZ2V0KCJleHB0aW1lIikgb3IgMC4wKQoKICAgIGZvciBwYXRoIGluIHBhdGhzWzE6XToKICAgICAgICBkYXRhLCB3Y3MsIF8sIGluZm8gPSBsb2FkX2ZyYW1lKHBhdGgsIGRvbm9yX3BhdGgpCiAgICAgICAgbmFtZSA9IG9zLnBhdGguYmFzZW5hbWUocGF0aCkKCiAgICAgICAgIyBGb3VyIGJyaWdodCwgd2lkZWx5IHNlcGFyYXRlZCBzdGFycyBmaXggdGhlIHNoaWZ0LCB0aGUgcm90YXRpb24gYW5kIHRoZQogICAgICAgICMgc2NhbGUgYmV0d2VlbiB0d28gZnJhbWVzIG9mIHRoZSBzYW1lIGZpZWxkLiBHb2luZyB0aHJvdWdoIGJvdGggcGxhdGUKICAgICAgICAjIHNvbHV0aW9ucyBpbnN0ZWFkIGNvc3RzIGEgbWludXRlIGEgZnJhbWUgYW5kIGFycml2ZXMgYXQgdGhlIHNhbWUgcGxhY2UuCiAgICAgICAgcGxhY2VkID0gRmFsc2UKICAgICAgICBpZiBwb3NpdGlvbnMgaXMgbm90IE5vbmUgYW5kIGxlbihwb3NpdGlvbnMpOgogICAgICAgICAgICB0cmFuc2Zvcm0sIGFkaWFnID0gYW5jaG9yX3RyYW5zZm9ybShyZWZlcmVuY2UsIGRhdGEsIHBvc2l0aW9ucykKICAgICAgICAgICAgaWYgdHJhbnNmb3JtIGlzIG5vdCBOb25lOgogICAgICAgICAgICAgICAgZGF0YSA9IGFwcGx5X3RyYW5zZm9ybShkYXRhLCB0cmFuc2Zvcm0sIHJlZmVyZW5jZS5zaGFwZSkKICAgICAgICAgICAgICAgIHByaW50KCIgICIgKyBuYW1lKQogICAgICAgICAgICAgICAgcHJpbnQoIiAgIiArIGRlc2NyaWJlX2FuY2hvcnMoYWRpYWcpKQogICAgICAgICAgICAgICAgcGxhY2VkID0gVHJ1ZQoKICAgICAgICBpZiBub3QgcGxhY2VkOgogICAgICAgICAgICBpZiB3Y3MgaXMgbm90IE5vbmUgYW5kIHJlZl93Y3MgaXMgbm90IE5vbmU6CiAgICAgICAgICAgICAgICBkYXRhID0gYWxpZ25fdG8ocmVmX3djcywgcmVmZXJlbmNlLnNoYXBlLCBkYXRhLCB3Y3MpCiAgICAgICAgICAgIGVsaWYgZGF0YS5zaGFwZSAhPSByZWZlcmVuY2Uuc2hhcGU6CiAgICAgICAgICAgICAgICBwcmludCgiICBza2lwcGluZyB7fSAtIG5vIHBsYXRlIHNvbHV0aW9uIGFuZCBhIGRpZmZlcmVudCAiCiAgICAgICAgICAgICAgICAgICAgICAic2hhcGUiLmZvcm1hdChuYW1lKSkKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgIGRhdGEgPSByZWZpbmVfYWxpZ25tZW50KHJlZmVyZW5jZSwgZGF0YSwgIiAgIiArIG5hbWUsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHBvc2l0aW9ucz1wb3NpdGlvbnMpCgogICAgICAgIHN0YWNrLmFwcGVuZChzY2FsZWQoZGF0YSwgaW5mbywgbmFtZSkpCiAgICAgICAgdG90YWxfZXhwdGltZSArPSBmbG9hdChpbmZvLmdldCgiZXhwdGltZSIpIG9yIDAuMCkKCiAgICBueSwgbnggPSByZWZlcmVuY2Uuc2hhcGUKICAgIGNvbWJpbmVkID0gbnAuZW1wdHkoKG55LCBueCksIGR0eXBlPW5wLmZsb2F0MzIpCiAgICBmb3IgeTAgaW4gcmFuZ2UoMCwgbnksIGJhbmRfcm93cyk6CiAgICAgICAgeTEgPSBtaW4oeTAgKyBiYW5kX3Jvd3MsIG55KQogICAgICAgIGJhbmQgPSBucC5zdGFjayhbZnJhbWVbeTA6eTFdIGZvciBmcmFtZSBpbiBzdGFja10sIGF4aXM9MCkKICAgICAgICB3aXRoIHdhcm5pbmdzLmNhdGNoX3dhcm5pbmdzKCk6CiAgICAgICAgICAgIHdhcm5pbmdzLnNpbXBsZWZpbHRlcigiaWdub3JlIikKICAgICAgICAgICAgY29tYmluZWRbeTA6eTFdID0gbnAubmFubWVkaWFuKGJhbmQsIGF4aXM9MCkKICAgICAgICAgICAgIyBBIHBpeGVsIHNhdHVyYXRlZCBpbiBldmVyeSBleHBvc3VyZSBoYXMgbm90aGluZyBsZWZ0IHRvIHRha2UgdGhlCiAgICAgICAgICAgICMgbWVkaWFuIG9mLiBJdCBpcyBzZXQgdG8gdGhlIHNhdHVyYXRpb24gbGV2ZWwsIHNvIHRoYXQgaXQgc3RheXMgYQogICAgICAgICAgICAjIGJyaWdodCBwaXhlbCB0aGUgcGhvdG9tZXRyeSBmbGFncyBhcyBzYXR1cmF0ZWQsIG5vdCBhIGhvbGUuCiAgICAgICAgICAgIGlmIHNhdF9tYXNrczoKICAgICAgICAgICAgICAgIGhvbGUgPSB+bnAuaXNmaW5pdGUoY29tYmluZWRbeTA6eTFdKQogICAgICAgICAgICAgICAgaG9sZSAmPSBucC5hbnkobnAuc3RhY2soW21beTA6eTFdIGZvciBtIGluIHNhdF9tYXNrc10sIGF4aXM9MCksIGF4aXM9MCkKICAgICAgICAgICAgICAgIGlmIGhvbGUuYW55KCk6CiAgICAgICAgICAgICAgICAgICAgY29tYmluZWRbeTA6eTFdW2hvbGVdID0gbWF4KHMgZm9yIHMgaW4gc2F0X2xldmVscyBpZiBzKQoKICAgIHByaW50KCIgIG1lZGlhbiBvZiB7fSBleHBvc3VyZXMsIHRvdGFsIGV4cG9zdXJlIHs6LjBmfXMsIG5vaXNlIGxvd2VyIGJ5IGFib3V0ICIKICAgICAgICAgICJhIGZhY3RvciBvZiB7Oi4xZn0iLmZvcm1hdChsZW4oc3RhY2spLCB0b3RhbF9leHB0aW1lLCBsZW4oc3RhY2spICoqIDAuNSkpCgogICAgIyBUaGUgbWVkaWFuIHNpdHMgYXQgdGhlIGxldmVsIG9mIE9ORSBleHBvc3VyZSBvZiB0aGUgZmlyc3QgZnJhbWUncyBsZW5ndGgsCiAgICAjIHNvIHRoYXQgaXMgd2hhdCBFWFBUSU1FIHNheXMuIEl0IHVzZWQgdG8gc2F5IHRoZSB0b3RhbCwgd2hpY2ggbWFkZSB0aGUKICAgICMgaGVhZGVyIGNsYWltIHNldmVyYWwgdGltZXMgdGhlIGV4cG9zdXJlIHRoZSBwaXhlbCB2YWx1ZXMgaGF2ZS4gVGhlIHRvdGFsCiAgICAjIGlzIGtlcHQgaW4gVE9URVhQLCBhbmQgTkNPTUJJTkUgc2F5cyBob3cgbWFueSBmcmFtZXMgd2VudCBpbiAtIHRoZSBub2lzZQogICAgIyBvZiBhIG1lZGlhbiBvZiBOIGZyYW1lcyBpcyB0aGF0IG9mIG9uZSBmcmFtZSB3aXRoIE4geCAyL3BpIHRpbWVzIHRoZSBnYWluLgogICAgaW5mbyA9IGRpY3QocmVmX2luZm8pCiAgICBpbmZvWyJleHB0aW1lIl0gPSByZWZfZXhwdGltZQogICAgaW5mb1sidG90YWxfZXhwdGltZSJdID0gdG90YWxfZXhwdGltZQogICAgaW5mb1sibmNvbWJpbmUiXSA9IGxlbihzdGFjaykKICAgIGluZm9bImNvbWJ0eXBlIl0gPSAibWVkaWFuIgogICAgaWYgYWxsKHMgZm9yIHMgaW4gc2F0X2xldmVscyk6CiAgICAgICAgIyBQaXhlbHMgYmVsb3cgdGhlIGxldmVsIG9mIHRoZSBzaG9ydGVzdCBleHBvc3VyZSBjYW1lIGZyb20gYXQgbGVhc3QKICAgICAgICAjIG9uZSB1bnNhdHVyYXRlZCBmcmFtZS4KICAgICAgICBpbmZvWyJzYXR1cmF0ZSJdID0gbWF4KHNhdF9sZXZlbHMpCiAgICBlbHNlOgogICAgICAgIGluZm9bInNhdHVyYXRlIl0gPSBOb25lCiAgICAjIFNjYWxpbmcgYSBsb25nZXIgZXhwb3N1cmUgZG93biBsb3dlcnMgaXRzIG5vaXNlIGFsb25nIHdpdGggaXRzIGNvdW50cywKICAgICMgYXMgYSBoaWdoZXIgZ2FpbiB3b3VsZC4gV3JpdHRlbiBzbyB0aGF0IEdBSU4geCBOQ09NQklORSBpcyB0aGUgZ2FpbiBvZiB0aGUKICAgICMgdG90YWwgZXhwb3N1cmUgZXhwcmVzc2VkIGF0IHRoaXMgZnJhbWUncyBsZXZlbC4KICAgIGlmIGluZm8uZ2V0KCJnYWluIik6CiAgICAgICAgaW5mb1siZ2FpbiJdID0gZmxvYXQoaW5mb1siZ2FpbiJdKSAqIHRvdGFsX2V4cHRpbWUgLyByZWZfZXhwdGltZSAvIGxlbihzdGFjaykKICAgIHJldHVybiBjb21iaW5lZCwgcmVmX3djcywgcmVmX2JwbSwgaW5mbwoKU0FUVVJBVElPTl9GUkFDVElPTiA9IDAuOCAgICMgYSBwaXhlbCBhYm92ZSB0aGlzIGZyYWN0aW9uIG9mIFNBVFVSQVRFIGNvdW50cyBhcyBzYXR1cmF0ZWQKCmRlZiBuZXdfcnVuX2ZvbGRlcihiYXNlLCB0b29sLCBnYWxheHkpOgogICAgIiIiQSBmb2xkZXIgb2YgaXRzIG93biBmb3IgdGhpcyBydW46IHJlc3VsdF88dG9vbD5fPGdhbGF4eT5fPGRhdGU+Xzx0aW1lPi4KCiAgICBFdmVyeSBydW4gdXNlZCB0byB3cml0ZSBpbnRvIHRoZSBzYW1lIHBsYWNlIHVuZGVyIHRoZSBzYW1lIG5hbWVzLCBzbyBhCiAgICBzZWNvbmQgcnVuIG92ZXJ3cm90ZSB0aGUgZmlyc3QsIGFuZCBhIHJ1biB0aGF0IHN0b3BwZWQgaGFsZiB3YXkgbGVmdCB0aGUKICAgIHByZXZpb3VzIHJ1bidzIGZpbGVzIHNpdHRpbmcgYmVzaWRlIGl0cyBvd24sIGxvb2tpbmcgY3VycmVudC4gQSBuZXcgZm9sZGVyCiAgICBlYWNoIHRpbWUgbWVhbnMgbm90aGluZyBhIHJ1biByZWFkcyBvciBzaG93cyBjYW4gYmUgbGVmdCBvdmVyIGZyb20gYW5vdGhlci4KICAgICIiIgogICAgaW1wb3J0IGRhdGV0aW1lCiAgICBuYW1lID0gcmUuc3ViKHIiW15BLVphLXowLTlfLi1dKyIsICJfIiwgKGdhbGF4eSBvciAiZ2FsYXh5Iikuc3RyaXAoKSkgb3IgImdhbGF4eSIKICAgIHN0YW1wID0gZGF0ZXRpbWUuZGF0ZXRpbWUubm93KCkuc3RyZnRpbWUoIiVZJW0lZF8lSCVNJVMiKQogICAgcGF0aCA9IG9zLnBhdGguam9pbihiYXNlIG9yIG9zLmdldGN3ZCgpLCAicmVzdWx0X3t9X3t9X3t9Ii5mb3JtYXQodG9vbCwgbmFtZSwgc3RhbXApKQogICAgb3MubWFrZWRpcnMocGF0aCwgZXhpc3Rfb2s9VHJ1ZSkKICAgIHJldHVybiBwYXRoCgpkZWYgb3V0cHV0X3ByZWZpeChnaXZlbiwgZmFsbGJhY2tfZGlyLCBnYWxheHkpOgogICAgIiIiVHVybiB3aGF0IHRoZSBwZXJzb24gdHlwZWQgaW50byBhIHN0ZW0gZm9yIHRoZSBvdXRwdXQgZmlsZXMuCgogICAgQSBmb2xkZXIgaXMgYWNjZXB0ZWQgYXMgd2VsbCBhcyBhIHN0ZW0sIGJlY2F1c2UgYSBmb2xkZXIgaXMgd2hhdCBwZW9wbGUgdHlwZToKICAgIGFza2VkIHdoZXJlIHRvIHB1dCB0aGUgb3V0cHV0LCB0aHJlZSBydW5zIGluIGEgcm93IHdlcmUgZ2l2ZW4gdGhlIHBhdGggb2YgdGhlCiAgICBmcmFtZXMnIG93biBmb2xkZXIsIGFuZCB0aGUgZmlsZXMgbGFuZGVkIGJlc2lkZSBpdCBuYW1lZCBhZnRlciBpdC4gSWYgd2hhdCB3YXMKICAgIGdpdmVuIGlzIGEgZGlyZWN0b3J5LCB0aGUgZmlsZXMgZ28gaW5zaWRlIGl0LCBuYW1lZCBhZnRlciB0aGUgZ2FsYXh5LgogICAgIiIiCiAgICBuYW1lID0gKGdhbGF4eSBvciAiZ2FsYXh5IikucmVwbGFjZSgiICIsICJfIikKICAgIGlmIGdpdmVuOgogICAgICAgIGdpdmVuID0gc3RyKGdpdmVuKS5zdHJpcCgpLnN0cmlwKCciJykuc3RyaXAoIiciKQogICAgICAgIGlmIG9zLnBhdGguaXNkaXIoZ2l2ZW4pOgogICAgICAgICAgICByZXR1cm4gb3MucGF0aC5qb2luKGdpdmVuLCBuYW1lKQogICAgICAgIHBhcmVudCA9IG9zLnBhdGguZGlybmFtZShnaXZlbikKICAgICAgICBpZiBwYXJlbnQgYW5kIG5vdCBvcy5wYXRoLmlzZGlyKHBhcmVudCk6CiAgICAgICAgICAgIG9zLm1ha2VkaXJzKHBhcmVudCwgZXhpc3Rfb2s9VHJ1ZSkKICAgICAgICByZXR1cm4gZ2l2ZW4KICAgIHJldHVybiBvcy5wYXRoLmpvaW4oZmFsbGJhY2tfZGlyIG9yICIuIiwgbmFtZSkKCmRlZiBzYXZlX2ZyYW1lKHBhdGgsIGRhdGEsIHdjcz1Ob25lLCBpbmZvPU5vbmUsIGhpc3Rvcnk9KCkpOgogICAgIiIiV3JpdGUgYSBmcmFtZSB0aGlzIHRvb2wgbWFkZSwgd2l0aCBhIGhlYWRlciB3b3J0aCBoYXZpbmcuCgogICAgVGhlIGNvbnRpbnV1bS1zdWJ0cmFjdGVkIGltYWdlIGlzIHRoZSBvbmUgdGhlIHdob2xlIG1lYXN1cmVtZW50IHJlc3RzIG9uLCBhbmQKICAgIHVudGlsIG5vdyBpdCBleGlzdGVkIG9ubHkgaW5zaWRlIGEgcGxvdC4gQW55b25lIHdhbnRpbmcgdG8gY2hlY2sgdGhlCiAgICBzdWJ0cmFjdGlvbiBieSBleWUgLSB3aGljaCBpcyBleGFjdGx5IHdoYXQgdGhlIHRvb2wgYXNrcyBmb3Igd2hlbiB0aGUgc3RhcnMKICAgIGRpc2FncmVlIGFib3V0IHRoZSBmYWN0b3IgLSBoYWQgbm8gZmlsZSB0byBvcGVuLiBJdCBjYXJyaWVzIHRoZSBwbGF0ZSBzb2x1dGlvbgogICAgc28gaXQgbGluZXMgdXAgd2l0aCB0aGUgb3JpZ2luYWxzIGluIGFueSB2aWV3ZXIsIGFuZCBhIGhpc3Rvcnkgc2F5aW5nIHdoYXQgaXQKICAgIGlzIGFuZCBob3cgaXQgd2FzIG1hZGUuCiAgICBXSEFUIElTIERPTkUgVE8gVEhFIFBJWEVMUyBCRUZPUkUgVEhFWSBBUkUgV1JJVFRFTgogICAgQmxhbmsgcGl4ZWxzIGFyZSB3cml0dGVuIGFzIHplcm8gcmF0aGVyIHRoYW4gYXMgTmFOLiBOYU4gaXMgbGVnYWwgRklUUyBhbmQKICAgIGFzdHJvcHkgcmVhZHMgaXQgaGFwcGlseSwgYnV0IGl0IGlzIGEgZmxvYXRpbmctcG9pbnQgdmFsdWUgdGhhdCBvbGRlciByZWFkZXJzCiAgICBwcmVkYXRlOiBBSVA0V2luIHdvdWxkIG5vdCBvcGVuIGEgZnJhbWUgd3JpdHRlbiBmcm9tIGhlcmUsIGFuZCBhIHN1YnRyYWN0ZWQKICAgIE04MyBmcmFtZSBjYXJyaWVkIG5pbmUgdGhvdXNhbmQgb2YgdGhlbSwgYXQgdGhlIGVkZ2VzIHRoZSByZXNhbXBsaW5nIGNvdWxkIG5vdAogICAgcmVhY2guIEEgZmlsZSB0aGF0IG9ubHkgdGhpcyBwaXBlbGluZSBjYW4gb3BlbiBpcyBub3QgbXVjaCBvZiBhbiBvdXRwdXQuCgogICAgV0hZIFRIRSBQTEFURSBTT0xVVElPTiBJUyBXUklUVEVOIFRXSUNFCiAgICBhc3Ryb3B5IHdyaXRlcyBhIHJvdGF0aW9uIGFzIGEgUEMgbWF0cml4IHdpdGggYSBzZXBhcmF0ZSBDREVMVCBzY2FsZS4gVGhhdCBpcwogICAgdGhlIGN1cnJlbnQgY29udmVudGlvbjsgdGhlIG9sZGVyIG9uZSBpcyBhIHNpbmdsZSBDRCBtYXRyaXgsIGFuZCBhIHJlYWRlciB0aGF0CiAgICBrbm93cyBvbmx5IENEIHNlZXMgYSBmcmFtZSB3aXRoIG5vIHVzYWJsZSBjb29yZGluYXRlcy4gQm90aCBhcmUgd3JpdHRlbiAtIHRoZXkKICAgIHNheSB0aGUgc2FtZSB0aGluZywgYW5kIENEID0gUEMgeCBDREVMVCBpcyBleGFjdCAtIHNvIGVpdGhlciBraW5kIG9mIHJlYWRlcgogICAgZmluZHMgd2hhdCBpdCBleHBlY3RzLgogICAgIiIiCiAgICBmcm9tIGFzdHJvcHkuaW8gaW1wb3J0IGZpdHMKCiAgICBoZWFkZXIgPSB3Y3MudG9faGVhZGVyKCkgaWYgd2NzIGlzIG5vdCBOb25lIGVsc2UgZml0cy5IZWFkZXIoKQogICAgaWYgd2NzIGlzIG5vdCBOb25lOgogICAgICAgIHRyeToKICAgICAgICAgICAgY2QgPSBucC5hc2FycmF5KHdjcy5waXhlbF9zY2FsZV9tYXRyaXgsIGR0eXBlPWZsb2F0KQogICAgICAgICAgICBmb3IgaSBpbiAoMSwgMik6CiAgICAgICAgICAgICAgICBmb3IgaiBpbiAoMSwgMik6CiAgICAgICAgICAgICAgICAgICAgaGVhZGVyWyJDRHt9X3t9Ii5mb3JtYXQoaSwgaildID0gZmxvYXQoY2RbaSAtIDEsIGogLSAxXSkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBwYXNzCiAgICBpZiBpbmZvOgogICAgICAgIGlmIGluZm8uZ2V0KCJmaWx0ZXIiKToKICAgICAgICAgICAgaGVhZGVyWyJGSUxURVIiXSA9IGluZm9bImZpbHRlciJdCiAgICAgICAgaWYgaW5mby5nZXQoImV4cHRpbWUiKToKICAgICAgICAgICAgaGVhZGVyWyJFWFBUSU1FIl0gPSBmbG9hdChpbmZvWyJleHB0aW1lIl0pCiAgICAgICAgaWYgaW5mby5nZXQoIm9iamVjdCIpOgogICAgICAgICAgICBoZWFkZXJbIk9CSkVDVCJdID0gaW5mb1sib2JqZWN0Il0KICAgICAgICBpZiBpbmZvLmdldCgicGl4ZWxfc2NhbGUiKToKICAgICAgICAgICAgaGVhZGVyWyJQSVhTQ0FMRSJdID0gZmxvYXQoaW5mb1sicGl4ZWxfc2NhbGUiXSkKICAgICAgICBmb3Iga2V5LCB3b3JkIGluICgoInNhdHVyYXRlIiwgIlNBVFVSQVRFIiksICgiZ2FpbiIsICJHQUlOIiksCiAgICAgICAgICAgICAgICAgICAgICAgICAgKCJyZG5vaXNlIiwgIlJETk9JU0UiKSwgKCJ0b3RhbF9leHB0aW1lIiwgIlRPVEVYUCIpKToKICAgICAgICAgICAgaWYgaW5mby5nZXQoa2V5KToKICAgICAgICAgICAgICAgIGhlYWRlclt3b3JkXSA9IGZsb2F0KGluZm9ba2V5XSkKICAgICAgICBpZiAoaW5mby5nZXQoIm5jb21iaW5lIikgb3IgMSkgPiAxOgogICAgICAgICAgICBoZWFkZXJbIk5DT01CSU5FIl0gPSBpbnQoaW5mb1sibmNvbWJpbmUiXSkKICAgICAgICAgICAgaGVhZGVyWyJDT01CVFlQRSJdID0gc3RyKGluZm8uZ2V0KCJjb21idHlwZSIpIG9yICJtZWRpYW4iKQogICAgICAgIGlmIGluZm8uZ2V0KCJkYXRlIik6CiAgICAgICAgICAgIGhlYWRlclsiREFURS1PQlMiXSA9IHN0cihpbmZvWyJkYXRlIl0pCiAgICBwaXhlbHMgPSBucC5hc2FycmF5KGRhdGEsIGR0eXBlPSJmbG9hdDMyIikKICAgIGJsYW5rID0gfm5wLmlzZmluaXRlKHBpeGVscykKICAgIGlmIGJsYW5rLmFueSgpOgogICAgICAgIHBpeGVscyA9IG5wLndoZXJlKGJsYW5rLCBucC5mbG9hdDMyKDAuMCksIHBpeGVscykKICAgICAgICBoZWFkZXJbIkhJU1RPUlkiXSA9ICgie30gYmxhbmsgcGl4ZWxzIHdyaXR0ZW4gYXMgemVybywgbm90IE5hTiwgc28gdGhhdCAiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIm9sZGVyIHJlYWRlcnMgY2FuIG9wZW4gdGhpcyIuZm9ybWF0KGludChibGFuay5zdW0oKSkpKQogICAgZm9yIGxpbmUgaW4gaGlzdG9yeToKICAgICAgICBoZWFkZXJbIkhJU1RPUlkiXSA9IGxpbmUKCiAgICBmaXRzLlByaW1hcnlIRFUoZGF0YT1waXhlbHMsIGhlYWRlcj1oZWFkZXIpLndyaXRldG8ocGF0aCwgb3ZlcndyaXRlPVRydWUpCiAgICByZXR1cm4gcGF0aAoKZGVmIGltYWdlX29yaWVudGF0aW9uKHdjcyk6CiAgICAiIiJBbmdsZSBvZiBOb3J0aCBpbiB0aGUgaW1hZ2UsIGFuZCB3aGV0aGVyIHRoZSBwYXJpdHkgaXMgZmxpcHBlZC4KCiAgICBSZWFkaW5nIHRoaXMgZnJvbSB0aGUgV0NTIHJlbW92ZXMgYSB3aG9sZSBjbGFzcyBvZiBzaWxlbnQgZXJyb3IuIEFzc3VtaW5nCiAgICBub3J0aCBpcyB1cCB3aGVuIGl0IGlzIG5vdCByb3RhdGVzIHRoZSBkZXByb2plY3Rpb24gYnkgYW4gdW5rbm93biBhbW91bnQsIGFuZAogICAgbm90aGluZyBpbiB0aGUgb3V0cHV0IGxvb2tzIHdyb25nLgogICAgIiIiCiAgICBjZCA9IHdjcy5waXhlbF9zY2FsZV9tYXRyaXgKICAgIG5vcnRoID0gbWF0aC5kZWdyZWVzKG1hdGguYXRhbjIoY2RbMSwgMV0sIGNkWzAsIDFdKSkgJSAzNjAuMAogICAgZWFzdCA9IG1hdGguZGVncmVlcyhtYXRoLmF0YW4yKGNkWzEsIDBdLCBjZFswLCAwXSkpICUgMzYwLjAKICAgIG1pcnJvcmVkID0gKChlYXN0IC0gbm9ydGgpICUgMzYwLjApID4gMTgwLjAKICAgIHJldHVybiBub3J0aCwgbWlycm9yZWQKCmRlZiBmcmFtZV9nZW9tZXRyeShwYXRoLCBkb25vcl9wYXRoPU5vbmUpOgogICAgIiIiV2hlcmUgYSBmcmFtZSBwb2ludHMgYW5kIGhvdyBiaWcgaXQgaXMsIHdpdGhvdXQgcmVhZGluZyBpdHMgcGl4ZWxzLgoKICAgIFRoZSBjYXRhbG9ndWUgbG9va3VwcyBhbGwgbmVlZCB0aGlzIC0gdGhlIGdhbGF4eSdzIGNlbnRyZSBpbiB0aGlzIGZyYW1lLCB0aGUKICAgIHN0YXJzIGluIGZyb250IG9mIGl0LCB0aGUgaXNvcGhvdGFsIHJhZGl1cyBpbiBwaXhlbHMgLSBhbmQgdGhleSBhcmUgbmVlZGVkCiAgICBiZWZvcmUgdGhlIGZyYW1lcyBhcmUgY29tYmluZWQsIG5vdCBhZnRlci4gUmVhZGluZyBhIGhlYWRlciBpcyBpbnN0YW50LCB3aGlsZQogICAgcmVhZGluZyBhIDYxLW1lZ2FwaXhlbCBpbWFnZSBpcyBub3QsIHNvIHRoZSBoZWFkZXIgaXMgdHJpZWQgZmlyc3QuIEEgZnJhbWUKICAgIHRoYXQgbG9zdCBpdHMgcGxhdGUgc29sdXRpb24gaW4gYSBjb21iaW5pbmcgc3RlcCBoYXMgdG8gYmUgb3BlbmVkIHByb3Blcmx5LAogICAgc2luY2UgaXRzIHNvbHV0aW9uIGlzIGZpdHRlZCBmcm9tIGEgZG9ub3IgZXhwb3N1cmUuCgogICAgUmV0dXJucyAod2NzLCBzaGFwZSwgaW5mbykuCiAgICAiIiIKICAgIGZyb20gYXN0cm9weS5pbyBpbXBvcnQgZml0cwogICAgZnJvbSBhc3Ryb3B5LndjcyBpbXBvcnQgV0NTCgogICAgdHJ5OgogICAgICAgIHdpdGggZml0cy5vcGVuKHBhdGgpIGFzIGhkdWw6CiAgICAgICAgICAgIGZvciBoZHUgaW4gaGR1bDoKICAgICAgICAgICAgICAgIGhlYWRlciA9IGhkdS5oZWFkZXIKICAgICAgICAgICAgICAgIGlmIGhlYWRlci5nZXQoIk5BWElTIikgPT0gMjoKICAgICAgICAgICAgICAgICAgICB3ID0gV0NTKGhlYWRlcikKICAgICAgICAgICAgICAgICAgICBpZiB3Lmhhc19jZWxlc3RpYWw6CiAgICAgICAgICAgICAgICAgICAgICAgIHJldHVybiAody5jZWxlc3RpYWwsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKGhlYWRlclsiTkFYSVMyIl0sIGhlYWRlclsiTkFYSVMxIl0pLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHsicGl4ZWxfc2NhbGUiOiBoZWFkZXIuZ2V0KCJQSVhTQ0FMRSIsIE5vbmUpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAiZmlsdGVyIjogc3RyKGhlYWRlci5nZXQoIkZJTFRFUiIsICI/IikpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAib2JqZWN0Ijogc3RyKGhlYWRlci5nZXQoIk9CSkVDVCIsICI/IikpfSkKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcGFzcwoKICAgIGRhdGEsIHdjcywgXywgaW5mbyA9IGxvYWRfZnJhbWUocGF0aCwgZG9ub3JfcGF0aCkKICAgIHJldHVybiB3Y3MsIGRhdGEuc2hhcGUsIGluZm8KCmRlZiBzYW1lX2dyaWQod2NzX2EsIHdjc19iLCBzaGFwZSwgdG9sZXJhbmNlX3B4PTEuNSk6CiAgICAiIiJUcnVlIHdoZW4gdHdvIHBsYXRlIHNvbHV0aW9ucyBwdXQgdGhlIHNreSBpbiB0aGUgc2FtZSBwaXhlbHMgYWxyZWFkeS4KCiAgICBXSFkgQVNLCiAgICBSZXNhbXBsaW5nIGEgZnJhbWUgaXMgdGhlIG1vc3QgZXhwZW5zaXZlIHRoaW5nIHRoaXMgdG9vbCBkb2VzIHRvIGl0OiB0d28KICAgIGFzdHJvbWV0cmljIHRyYW5zZm9ybXMgb2YgZXZlcnkgcGl4ZWwgcG9zaXRpb24sIHRlbnMgb2YgbWlsbGlvbnMgb2YgdGhlbS4gSXQKICAgIGlzIHdvcnRoIGl0IHdoZW4gdGhlIGZyYW1lcyByZWFsbHkgc2l0IG9uIGRpZmZlcmVudCBncmlkcy4gSXQgaXMgcHVyZSB3YXN0ZQogICAgd2hlbiB0aGV5IGRvIG5vdCAtIGFuZCBiZXR3ZWVuIGNvbnNlY3V0aXZlIGV4cG9zdXJlcyBvZiBvbmUgZmlsdGVyLCB0aGV5CiAgICBhbG1vc3QgbmV2ZXIgZG8uIE1lYXN1cmVkIG9uIGEgbmlnaHQgb2YgTTMzOiB0aGUgZXhwb3N1cmVzIG9mIGEgZmlsdGVyIHNhdAogICAgd2l0aGluIDAuMDUgdG8gMC43IG9mIGEgcGl4ZWwgb2YgZWFjaCBvdGhlci4KCiAgICBTbyB0aGUgcXVlc3Rpb24gaXMgYXNrZWQgYmVmb3JlIHRoZSB3b3JrIGlzIGRvbmUuIEEgZmV3IGNvcm5lciBhbmQgY2VudHJlCiAgICBwb2ludHMgYXJlIHB1c2hlZCB0aHJvdWdoIGJvdGggc29sdXRpb25zOyBpZiBub25lIG9mIHRoZW0gbW92ZXMgYnkgbW9yZSB0aGFuCiAgICBhYm91dCBhIHBpeGVsLCB0aGUgZ3JpZHMgYXJlIHRoZSBzYW1lIGFuZCBhIHN1Yi1waXhlbCBzaGlmdCBpcyBhbGwgdGhhdCBpcwogICAgbmVlZGVkLiBUaGUgZnVsbCByZXNhbXBsaW5nIHN0YXlzIGZvciB0aGUgY2FzZSBpdCB3YXMgd3JpdHRlbiBmb3IuCiAgICAiIiIKICAgIGltcG9ydCBudW1weSBhcyBucAoKICAgIGlmIHdjc19hIGlzIE5vbmUgb3Igd2NzX2IgaXMgTm9uZToKICAgICAgICByZXR1cm4gRmFsc2UKICAgIG55LCBueCA9IHNoYXBlCiAgICB4cyA9IG5wLmFycmF5KFswLCBueCAtIDEsIDAsIG54IC0gMSwgbnggLy8gMiwgbnggLy8gNCwgMyAqIG54IC8vIDRdLAogICAgICAgICAgICAgICAgICBkdHlwZT1mbG9hdCkKICAgIHlzID0gbnAuYXJyYXkoWzAsIDAsIG55IC0gMSwgbnkgLSAxLCBueSAvLyAyLCAzICogbnkgLy8gNCwgbnkgLy8gNF0sCiAgICAgICAgICAgICAgICAgIGR0eXBlPWZsb2F0KQogICAgdHJ5OgogICAgICAgIHNreSA9IHdjc19hLnBpeGVsX3RvX3dvcmxkKHhzLCB5cykKICAgICAgICBieCwgYnkgPSB3Y3NfYi53b3JsZF90b19waXhlbChza3kpCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHJldHVybiBGYWxzZQogICAgbW92ZWQgPSBucC5oeXBvdChieCAtIHhzLCBieSAtIHlzKQogICAgcmV0dXJuIGJvb2wobnAuYWxsKG5wLmlzZmluaXRlKG1vdmVkKSkgYW5kIG1vdmVkLm1heCgpIDw9IHRvbGVyYW5jZV9weCkKCmRlZiBhbGlnbl90byhyZWZfd2NzLCByZWZfc2hhcGUsIGRhdGEsIHdjcywgYmFuZF9yb3dzPTUxMik6CiAgICAiIiJSZXNhbXBsZSBvbmUgZnJhbWUgb250byBhbm90aGVyIGZyYW1lJ3MgcGl4ZWwgZ3JpZCB1c2luZyBib3RoIFdDUyBzb2x1dGlvbnMuCgogICAgQm90aCBmcmFtZXMgYXJlIHBsYXRlLXNvbHZlZCwgc28gZXZlcnkgb3V0cHV0IHBpeGVsIGNhbiBiZSBtYXBwZWQgdG8gaXRzIHRydWUKICAgIHNreSBwb3NpdGlvbiBhbmQgdGhlIGlucHV0IGZyYW1lIHJlYWQgdGhlcmUuIEV4YWN0LCBhbmQgaXQgbmVlZHMgbm8gbWF0Y2hpbmcgb2YKICAgIHN0YXJzIGJldHdlZW4gdGhlIHR3byBpbWFnZXMuCgogICAgRG9uZSBhIGZldyBodW5kcmVkIHJvd3MgYXQgYSB0aW1lLiBUaGUgd2hvbGUtZnJhbWUgdmVyc2lvbiBidWlsdCBmaXZlIGFycmF5cwogICAgdGhlIHNpemUgb2YgdGhlIGltYWdlIHRvIGhvbGQgdGhlIGNvb3JkaW5hdGVzIC0gYWJvdXQgMi41IEdCIG9uIGEgNjEtbWVnYXBpeGVsCiAgICBmcmFtZSwgZm9yIGV2ZXJ5IGV4cG9zdXJlIC0gd2hpY2ggaXMgbWVtb3J5IHNwZW50IG9uIHNjcmF0Y2ggcGFwZXIuIFRoZQogICAgYXJpdGhtZXRpYyBpcyBpZGVudGljYWw7IG9ubHkgdGhlIG9yZGVyIGNoYW5nZWQuCiAgICAiIiIKICAgIGZyb20gc2NpcHkubmRpbWFnZSBpbXBvcnQgbWFwX2Nvb3JkaW5hdGVzCgogICAgbnksIG54ID0gcmVmX3NoYXBlCiAgICBvdXQgPSBucC5lbXB0eSgobnksIG54KSwgZHR5cGU9bnAuZmxvYXQzMikKICAgIGZvciB5MCBpbiByYW5nZSgwLCBueSwgYmFuZF9yb3dzKToKICAgICAgICB5MSA9IG1pbih5MCArIGJhbmRfcm93cywgbnkpCiAgICAgICAgeXksIHh4ID0gbnAubWdyaWRbeTA6eTEsIDA6bnhdCiAgICAgICAgc2t5ID0gcmVmX3djcy5waXhlbF90b193b3JsZCh4eCwgeXkpCiAgICAgICAgc3gsIHN5ID0gd2NzLndvcmxkX3RvX3BpeGVsKHNreSkKICAgICAgICBvdXRbeTA6eTFdID0gbWFwX2Nvb3JkaW5hdGVzKGRhdGEsIFtzeSwgc3hdLCBvcmRlcj0xLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbW9kZT0iY29uc3RhbnQiLCBjdmFsPW5wLm5hbikKICAgIHJldHVybiBvdXQKCmRlZiBtZWFzdXJlX2Z3aG0oZGF0YSwgbWFzaz1Ob25lLCBtYXhfc291cmNlcz0yMDAsIHBvc2l0aW9ucz1Ob25lKToKICAgICIiIldpZHRoIG9mIHRoZSBtZWRpYW4gc3RhciBpbiBhIGZyYW1lLCBpbiBwaXhlbHMsIGF0IGhhbGYgaXRzIGhlaWdodC4KCiAgICBXSFkgVEhFIE1FRElBTiBTVEFSIEFORCBOT1QgVEhFIE1FRElBTiBPRiBUSEUgU1RBUlMKICAgIEJlY2F1c2UgYSBzaW5nbGUgZmFpbnQgc3RhciBpcyBtb3N0bHkgbm9pc2UsIGFuZCBldmVyeSB3YXkgb2YgcmVkdWNpbmcgb25lIHRvCiAgICBhIG51bWJlciBpcyBwdWxsZWQgYWJvdXQgYnkgdGhhdCBub2lzZS4gU3RhY2tpbmcgdHdvIGh1bmRyZWQgb2YgdGhlbSBmaXJzdAogICAgZ2l2ZXMgb25lIGNsZWFuIHN0YXIgdG8gbWVhc3VyZSwgYW5kIGl0cyB3aWR0aCBpcyB0aGUgd2lkdGggdGhlIGZyYW1lIGhhcy4KCiAgICBXSFkgTk9UIFRIRSBTRUNPTkQgTU9NRU5UCiAgICBUaGF0IGlzIHdoYXQgdGhpcyBkaWQgYmVmb3JlLCBhbmQgaXQgZ2F2ZSB0aGUgd3JvbmcgYW5zd2VyIGluIHRoZSB3YXkgdGhhdAogICAgbWF0dGVycyBtb3N0OiBpdCBnb3QgdGhlIG9yZGVyIG9mIHRoZSB0d28gZmlsdGVycyBiYWNrd2FyZHMuIEEgc2Vjb25kIG1vbWVudAogICAgd2VpZ2h0cyBlYWNoIHBpeGVsIGJ5IHRoZSBzcXVhcmUgb2YgaXRzIGRpc3RhbmNlLCBzbyB0aGUgb3V0ZXIgcmluZyBvZiBhIDE5CiAgICBwaXhlbCBzdGFtcCBjb3VudHMgYWJvdXQgYSBodW5kcmVkIGFuZCBzZXZlbnR5IHRpbWVzIGFzIGhlYXZpbHkgYXMgdGhlIGNlbnRyZS4KICAgIFdoYXRldmVyIGlzIG91dCB0aGVyZSB0aGF0IGlzIG5vdCB0aGUgc3RhciAtIHRoZSBnYWxheHkncyBvd24gbGlnaHQgdW5kZXIgaXQsCiAgICBhIG5laWdoYm91ciwgc2t5IGxlZnQgb3ZlciBiZWNhdXNlIG9ubHkgdGhlIGZyYW1lJ3MgZ2xvYmFsIGxldmVsIGhhZCBiZWVuCiAgICB0YWtlbiBvZmYgLSBpcyB0aGVyZWZvcmUgbW9zdCBvZiB0aGUgYW5zd2VyLiBPbiBNMzMgdGhhdCBtYWRlIHRoZSBILWFscGhhCiAgICBmcmFtZSBtZWFzdXJlIDUuNTMgcHggYWdhaW5zdCBSJ3MgNS43Nywgc28gdGhlIHRvb2wgYmx1cnJlZCBILWFscGhhOyBzdGFja2luZwogICAgdGhlIHNhbWUgc3RhcnMgc2hvd3MgSC1hbHBoYSBhdCA0LjkxIGFuZCBSIGF0IDMuMTQsIHNvIEgtYWxwaGEgd2FzIGFscmVhZHkgdGhlCiAgICBicm9hZGVyIG9mIHRoZSB0d28gYW5kIGJsdXJyaW5nIGl0IG1hZGUgdGhlIG1pc21hdGNoIHdvcnNlLiBFdmVyeSBzdGFyIHRoZW4KICAgIHN1YnRyYWN0ZWQgdG8gYSBibGFjayBjb3JlIGluc2lkZSBhIGJyaWdodCByaW5nLCB3aGljaCBpcyBleGFjdGx5IHdoYXQgdGhlCiAgICBmcmFtZXMgbG9va2VkIGxpa2UuCgogICAgSGFsZi1tYXhpbXVtIGlzIHJlYWQgc3RyYWlnaHQgb2ZmIHRoZSByYWRpYWwgcHJvZmlsZSBvZiB0aGUgc3RhY2suIEl0IGlzIGEKICAgIHByb3BlcnR5IG9mIHRoZSBjb3JlLCB3aGVyZSB0aGUgc3RhciBpcywgYW5kIGl0IGRvZXMgbm90IGNhcmUgd2hhdCB0aGUgc3RhbXAncwogICAgb3V0c2tpcnRzIGNvbnRhaW4uCgogICAgVEhSRUUgVEhJTkdTIFRIRSBTVEFDSyBORUVEUwogICAgVGhlIGxvY2FsIHNreSBjb21lcyBvZmYgZWFjaCBzdGFtcCBzZXBhcmF0ZWx5LCB0YWtlbiBmcm9tIGl0cyBib3JkZXIgcmluZywgc28KICAgIGEgc3RhciBvbiBhIGJyaWdodCBhcm0gYW5kIGEgc3RhciBvbiBlbXB0eSBza3kgY29udHJpYnV0ZSB0aGUgc2FtZSBzdGFyLgogICAgRWFjaCBzdGFtcCBpcyBzaGlmdGVkIG9udG8gYSBjb21tb24gY2VudHJlIGJlZm9yZSBzdGFja2luZywgdG8gYSBmcmFjdGlvbiBvZiBhCiAgICBwaXhlbCwgb3IgdGhlIHN0YWNrIHdvdWxkIGJlIHNtZWFyZWQgYnkgdXAgdG8gaGFsZiBhIHBpeGVsIG9mIHJvdW5kaW5nIC0gYW5kCiAgICBzbWVhcmVkIGRpZmZlcmVudGx5IGluIGVhY2ggZmlsdGVyLCB3aGljaCBpcyB0aGUgZXJyb3IgYmVpbmcgbG9va2VkIGZvci4KICAgIFNhdHVyYXRlZCBzdGFycyBhcmUgZHJvcHBlZDogYSBmbGF0IHRvcCBpcyB3aWRlciB0aGFuIHRoZSBzdGFyIHRoYXQgbWFkZSBpdCwKICAgIGFuZCBpdCBpcyB0aGUgYnJpZ2h0ZXN0IHN0YXJzIHRoYXQgc2F0dXJhdGUsIHNvIGtlZXBpbmcgdGhlbSB3b3VsZCBiaWFzIHRoZQogICAgZnJhbWUgd2l0aCB0aGUgZGVlcGVyIGV4cG9zdXJlLiBBIGNvcmUgd2l0aCB0aHJlZSBvciBtb3JlIHBpeGVscyB3aXRoaW4gdHdvCiAgICBwZXJjZW50IG9mIGl0cyBwZWFrIGlzIGZsYXQsIGFuZCBhIHN0YXIncyBjb3JlIGlzIG5vdC4KICAgICIiIgogICAgZnJvbSBhc3Ryb3B5LnN0YXRzIGltcG9ydCBzaWdtYV9jbGlwcGVkX3N0YXRzCiAgICBmcm9tIHNjaXB5Lm5kaW1hZ2UgaW1wb3J0IHNoaWZ0IGFzIF9zaGlmdAoKICAgIGhhbGYgPSAxMAogICAgbnksIG54ID0gZGF0YS5zaGFwZQogICAgeXksIHh4ID0gbnAubWdyaWRbLWhhbGY6aGFsZiArIDEsIC1oYWxmOmhhbGYgKyAxXQogICAgcnIgPSBucC5oeXBvdCh4eCwgeXkpCiAgICBib3JkZXIgPSByciA+IGhhbGYgLSAxLjUKCiAgICBpZiBwb3NpdGlvbnMgaXMgbm90IE5vbmUgYW5kIGxlbihwb3NpdGlvbnMpID49IDEwOgogICAgICAgIGNhbmRpZGF0ZXMgPSBucC5hc2FycmF5KHBvc2l0aW9ucywgZHR5cGU9ZmxvYXQpCiAgICBlbHNlOgogICAgICAgIGZyb20gcGhvdHV0aWxzLmRldGVjdGlvbiBpbXBvcnQgREFPU3RhckZpbmRlcgogICAgICAgIG1lZGlhbiwgc3RkID0gc2lnbWFfY2xpcHBlZF9zdGF0cyhkYXRhLCBzaWdtYT0zLjAsIG1hc2s9bWFzaylbMTpdCiAgICAgICAgZmluZGVyID0gREFPU3RhckZpbmRlcihmd2htPTQuMCwgdGhyZXNob2xkPTIwLjAgKiBzdGQpCiAgICAgICAgdGJsID0gZmluZGVyKGRhdGEgLSBtZWRpYW4sIG1hc2s9bWFzaykKICAgICAgICBpZiB0YmwgaXMgTm9uZSBvciBsZW4odGJsKSA9PSAwOgogICAgICAgICAgICByZXR1cm4gTm9uZQogICAgICAgIHRibC5zb3J0KCJmbHV4IikKICAgICAgICB0YmwgPSB0YmxbOjotMV1bOm1heF9zb3VyY2VzXQogICAgICAgIGNhbmRpZGF0ZXMgPSBucC5jb2x1bW5fc3RhY2soW3RibFsieGNlbnRyb2lkIl0sIHRibFsieWNlbnRyb2lkIl1dKQoKICAgIGZvdW5kID0gW10KICAgIGZvciB4LCB5IGluIGNhbmRpZGF0ZXM6CiAgICAgICAgeGksIHlpID0gaW50KHJvdW5kKHgpKSwgaW50KHJvdW5kKHkpKQogICAgICAgIGlmIG5vdCAoaGFsZiA8IHhpIDwgbnggLSBoYWxmIGFuZCBoYWxmIDwgeWkgPCBueSAtIGhhbGYpOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHN0YW1wID0gbnAubmFuX3RvX251bSgKICAgICAgICAgICAgZGF0YVt5aSAtIGhhbGY6eWkgKyBoYWxmICsgMSwgeGkgLSBoYWxmOnhpICsgaGFsZiArIDFdKS5hc3R5cGUoZmxvYXQpCiAgICAgICAgc3RhbXAgPSBzdGFtcCAtIG5wLm1lZGlhbihzdGFtcFtib3JkZXJdKQogICAgICAgIGNvcmUgPSBzdGFtcFtoYWxmIC0gMTpoYWxmICsgMiwgaGFsZiAtIDE6aGFsZiArIDJdCiAgICAgICAgcGVhayA9IGZsb2F0KGNvcmUubWF4KCkpCiAgICAgICAgaWYgcGVhayA8PSAwIG9yIChjb3JlID49IDAuOTggKiBwZWFrKS5zdW0oKSA+PSAzOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGZvdW5kLmFwcGVuZCgocGVhaywgX3NoaWZ0KHN0YW1wLCAoeWkgLSB5LCB4aSAtIHgpLCBvcmRlcj0xLCBtb2RlPSJuZWFyZXN0IikpKQogICAgaWYgbGVuKGZvdW5kKSA8IDEwOgogICAgICAgIHJldHVybiBOb25lCgogICAgZm91bmQuc29ydChrZXk9bGFtYmRhIHBhaXI6IC1wYWlyWzBdKQogICAgc3RhY2tlZCA9IG5wLm1lZGlhbihucC5zdGFjayhbcyBmb3IgXywgcyBpbiBmb3VuZFs6bWF4X3NvdXJjZXNdXSksIGF4aXM9MCkKICAgIHByb2ZpbGUgPSBucC5hcnJheShbc3RhY2tlZFsocnIgPj0gcSkgJiAocnIgPCBxICsgMSldLm1lYW4oKQogICAgICAgICAgICAgICAgICAgICAgICBmb3IgcSBpbiByYW5nZShoYWxmICsgMSldKQogICAgaWYgcHJvZmlsZVswXSA8PSAwOgogICAgICAgIHJldHVybiBOb25lCiAgICBmb3IgcSBpbiByYW5nZSgxLCBoYWxmICsgMSk6CiAgICAgICAgaWYgcHJvZmlsZVtxXSA8IDAuNSAqIHByb2ZpbGVbMF06CiAgICAgICAgICAgIGJlbG93LCBhYm92ZSA9IHByb2ZpbGVbcSAtIDFdLCBwcm9maWxlW3FdCiAgICAgICAgICAgIHJldHVybiBmbG9hdCgyLjAgKiAocSAtIDEgKyAoMC41ICogcHJvZmlsZVswXSAtIGJlbG93KSAvIChhYm92ZSAtIGJlbG93KSkpCiAgICByZXR1cm4gTm9uZQoKZGVmIG1hdGNoX3BzZihzaGFycCwgZndobV9zaGFycCwgZndobV9ibHVudCk6CiAgICAiIiJCbHVyIHRoZSBzaGFycGVyIGZyYW1lIHNvIGJvdGggZnJhbWVzIGhhdmUgdGhlIHNhbWUgc3RhciBzaXplLgoKICAgIFN1YnRyYWN0aW5nIGZyYW1lcyBvZiBkaWZmZXJlbnQgc2hhcnBuZXNzIGxlYXZlcyBhIHJpbmcgYXJvdW5kIGV2ZXJ5IHN0YXIgLSBhCiAgICBicmlnaHQgY29yZSB3aXRoIGEgZGFyayBoYWxvLCBvciB0aGUgcmV2ZXJzZSAtIGFuZCB0aG9zZSByaW5ncyBhcmUgdGhlbiBkZXRlY3RlZAogICAgYXMgSElJIHJlZ2lvbnMuIE1hdGNoaW5nIHRoZSB3aWR0aHMgZmlyc3QgaXMgd2hhdCBtYWtlcyBzdGFycyBkaXNhcHBlYXIgY2xlYW5seS4KICAgICIiIgogICAgZnJvbSBzY2lweS5uZGltYWdlIGltcG9ydCBnYXVzc2lhbl9maWx0ZXIKCiAgICBpZiBmd2htX2JsdW50IGlzIE5vbmUgb3IgZndobV9zaGFycCBpcyBOb25lIG9yIGZ3aG1fYmx1bnQgPD0gZndobV9zaGFycDoKICAgICAgICByZXR1cm4gc2hhcnAsIDAuMAogICAgc2lnbWEgPSBtYXRoLnNxcnQobWF4KGZ3aG1fYmx1bnQgKiogMiAtIGZ3aG1fc2hhcnAgKiogMiwgMC4wKSkgLyAyLjM1NDgKICAgIHJldHVybiBnYXVzc2lhbl9maWx0ZXIoc2hhcnAsIHNpZ21hKSwgc2lnbWEKCmRlZiBfc3RhY2tfc3RhcnMoZGF0YSwgcG9zaXRpb25zLCBtYXhfc291cmNlcz0yMDAsIGhhbGY9MTApOgogICAgIiIiVGhlIG1lZGlhbiBzdGFyIG9mIGEgZnJhbWU6IGxvY2FsIHNreSBvZmYsIHN1Yi1waXhlbCBhbGlnbmVkLCBubyBzYXR1cmF0aW9uLgoKICAgIE9uZSBzdGFyIGlzIG1vc3RseSBub2lzZS4gVHdvIGh1bmRyZWQgb2YgdGhlbSBzdGFja2VkIGlzIG9uZSBjbGVhbiBzdGFyLCBhbmQKICAgIGV2ZXJ5IHF1ZXN0aW9uIGFib3V0IHRoZSBzaGFwZSBvZiB0aGlzIGZyYW1lJ3Mgc3RhcnMgaXMgYXNrZWQgb2YgdGhhdC4KICAgICIiIgogICAgZnJvbSBzY2lweS5uZGltYWdlIGltcG9ydCBzaGlmdCBhcyBfc2hpZnQKCiAgICBkYXRhID0gbnAuYXNhcnJheShkYXRhKQogICAgcG9zID0gbnAuYXNhcnJheShwb3NpdGlvbnMsIGR0eXBlPWZsb2F0KQogICAgbnksIG54ID0gZGF0YS5zaGFwZQogICAgeXksIHh4ID0gbnAubWdyaWRbLWhhbGY6aGFsZiArIDEsIC1oYWxmOmhhbGYgKyAxXQogICAgYm9yZGVyID0gbnAuaHlwb3QoeHgsIHl5KSA+IGhhbGYgLSAxLjUKCiAgICBmb3VuZCA9IFtdCiAgICBmb3IgeCwgeSBpbiBwb3M6CiAgICAgICAgeGksIHlpID0gaW50KHJvdW5kKHgpKSwgaW50KHJvdW5kKHkpKQogICAgICAgIGlmIG5vdCAoaGFsZiA8IHhpIDwgbnggLSBoYWxmIGFuZCBoYWxmIDwgeWkgPCBueSAtIGhhbGYpOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHN0YW1wID0gbnAubmFuX3RvX251bSgKICAgICAgICAgICAgZGF0YVt5aSAtIGhhbGY6eWkgKyBoYWxmICsgMSwgeGkgLSBoYWxmOnhpICsgaGFsZiArIDFdKS5hc3R5cGUoZmxvYXQpCiAgICAgICAgc3RhbXAgPSBzdGFtcCAtIG5wLm1lZGlhbihzdGFtcFtib3JkZXJdKQogICAgICAgIGNvcmUgPSBzdGFtcFtoYWxmIC0gMTpoYWxmICsgMiwgaGFsZiAtIDE6aGFsZiArIDJdCiAgICAgICAgcGVhayA9IGZsb2F0KGNvcmUubWF4KCkpCiAgICAgICAgaWYgcGVhayA8PSAwIG9yIChjb3JlID49IDAuOTggKiBwZWFrKS5zdW0oKSA+PSAzOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGZvdW5kLmFwcGVuZCgocGVhaywgX3NoaWZ0KHN0YW1wLCAoeWkgLSB5LCB4aSAtIHgpLCBvcmRlcj0xLCBtb2RlPSJuZWFyZXN0IikpKQogICAgaWYgbGVuKGZvdW5kKSA8IDEwOgogICAgICAgIHJldHVybiBOb25lCiAgICBmb3VuZC5zb3J0KGtleT1sYW1iZGEgcGFpcjogLXBhaXJbMF0pCiAgICByZXR1cm4gbnAubWVkaWFuKG5wLnN0YWNrKFtzIGZvciBfLCBzIGluIGZvdW5kWzptYXhfc291cmNlc11dKSwgYXhpcz0wKQoKZGVmIG1hdGNoX3BzZl9vbl9zdGFycyhoYSwgcmVkLCBwb3NpdGlvbnMsIG1heF9zaWdtYT0zLjAsIHN0ZXA9MC4wMiwgaGFsZj0xMCk6CiAgICAiIiJCbHVyIG9uZSBmcmFtZSB1bnRpbCBpdHMgc3RhcnMgaGF2ZSB0aGUgc2FtZSBzaGFwZSBhcyB0aGUgb3RoZXIncy4KCiAgICBXSFkgTk9UIE1FQVNVUkUgVFdPIFdJRFRIUyBBTkQgU1VCVFJBQ1QgVEhFTSBJTiBRVUFEUkFUVVJFCiAgICBCZWNhdXNlIHRoYXQgaXMgd2hhdCBtYXRjaF9wc2YgZG9lcyBhbmQgaXQgZG9lcyBub3Qgd29yayBoZXJlLCBmb3IgdHdvIHJlYXNvbnMKICAgIHRoYXQgY29tcG91bmQuIEEgd2lkdGggaXMgb25lIG51bWJlciBzcXVlZXplZCBvdXQgb2YgYSB3aG9sZSBwcm9maWxlLCBzbyBpdAogICAgdGhyb3dzIGF3YXkgZXhhY3RseSB0aGUgcGFydCB0aGF0IG1hdHRlcnMgLSBhIHN0YXIgY2FuIGhhdmUgdGhlIHNhbWUgaGFsZi13aWR0aAogICAgYXMgYW5vdGhlciBhbmQgYSBxdWl0ZSBkaWZmZXJlbnQgY29yZSBhbmQgc2tpcnQuIEFuZCB0aGUgcXVhZHJhdHVyZSBmb3JtdWxhIGlzCiAgICBvbmx5IGV4YWN0IGZvciBHYXVzc2lhbnM7IGEgcmVhbCBzdGFyIGhhcyBicm9hZGVyIHdpbmdzIHRoYW4gYSBHYXVzc2lhbiwgc28gdGhlCiAgICBibHVyIGl0IHByZXNjcmliZXMgaXMgbm90IHRoZSBibHVyIHRoYXQgbWF0Y2hlcyB0aGUgdHdvLgoKICAgIE1lYXN1cmVkIG9uIE04MzogdGhlIHdpZHRocyBjYW1lIG91dCBILWFscGhhIDMuNzggcHggYWdhaW5zdCBSIDQuMjIsIHdoaWNoCiAgICBwcmVzY3JpYmVkIGJsdXJyaW5nIEgtYWxwaGEgYnkgc2lnbWEgMC44MC4gQWZ0ZXJ3YXJkcyB0aGUgc3RhY2tlZCBzdGFyIGxlZnQgYQogICAgbmVnYXRpdmUgY29yZSBpbnNpZGUgYSBwb3NpdGl2ZSByaW5nIC0gdGhlIHNpZ25hdHVyZSBvZiBILWFscGhhIG5vdyBiZWluZyB0aGUKICAgIGJyb2FkZXIgb2YgdGhlIHR3by4gVGhlIHByZXNjcmlwdGlvbiBoYWQgdGhlIHNpZ24gb2YgdGhlIGFuc3dlciB3cm9uZy4KCiAgICBXSEFUIFRISVMgRE9FUyBJTlNURUFECiAgICBJdCBhc2tzIHRoZSBxdWVzdGlvbiB0aGUgcGljdHVyZSBhc2tzLiBCb3RoIGZyYW1lcycgc3RhcnMgYXJlIHN0YWNrZWQsIGVhY2gKICAgIHByb2ZpbGUgaXMgZGl2aWRlZCBieSBpdHMgb3duIGxpZ2h0IGluc2lkZSBzaXggcGl4ZWxzIHNvIHRoYXQgYnJpZ2h0bmVzcyBkcm9wcwogICAgb3V0IG9mIHRoZSBjb21wYXJpc29uIGFuZCBvbmx5IHNoYXBlIGlzIGxlZnQsIGFuZCB0aGVuIG9uZSBmcmFtZSBpcyBibHVycmVkIGJ5CiAgICBldmVyeSBzaWdtYSBpbiBhIHJhbmdlIHdoaWxlIHRoZSB0d28gcHJvZmlsZXMgYXJlIGNvbXBhcmVkIHJpbmcgYnkgcmluZy4gVGhlCiAgICBzaWdtYSB0aGF0IG1ha2VzIHRoZW0gYWdyZWUgaXMgdGhlIGFuc3dlciwgYW5kIGl0IGlzIGFuIGFuc3dlciBhYm91dCB0aGUgd2hvbGUKICAgIHByb2ZpbGUgcmF0aGVyIHRoYW4gYWJvdXQgb25lIHBvaW50IG9uIGl0LgoKICAgIEEgcG9zaXRpdmUgc2lnbWEgYmx1cnMgSC1hbHBoYSwgYSBuZWdhdGl2ZSBvbmUgYmx1cnMgUjsgdGhlIHNlYXJjaCBjb3ZlcnMgYm90aAogICAgc28gaXQgY2Fubm90IGdldCB0aGUgZGlyZWN0aW9uIHdyb25nLgoKICAgIFJldHVybnMgKGhhLCByZWQsIHNpZ21hLCBkaWFnbm9zdGljcykuCiAgICAiIiIKICAgIGZyb20gc2NpcHkubmRpbWFnZSBpbXBvcnQgZ2F1c3NpYW5fZmlsdGVyCgogICAgaWYgcG9zaXRpb25zIGlzIE5vbmUgb3IgbGVuKHBvc2l0aW9ucykgPCAxMDoKICAgICAgICByZXR1cm4gaGEsIHJlZCwgMC4wLCB7IndoeSI6ICJubyBzdGFyIHBvc2l0aW9ucyB0byBtZWFzdXJlIG9uIn0KICAgIEEwID0gX3N0YWNrX3N0YXJzKGhhLCBwb3NpdGlvbnMsIGhhbGY9aGFsZikKICAgIFIwID0gX3N0YWNrX3N0YXJzKHJlZCwgcG9zaXRpb25zLCBoYWxmPWhhbGYpCiAgICBpZiBBMCBpcyBOb25lIG9yIFIwIGlzIE5vbmU6CiAgICAgICAgcmV0dXJuIGhhLCByZWQsIDAuMCwgeyJ3aHkiOiAibm90IGVub3VnaCB1c2FibGUgc3RhcnMgdG8gc3RhY2sifQoKICAgIHl5LCB4eCA9IG5wLm1ncmlkWy1oYWxmOmhhbGYgKyAxLCAtaGFsZjpoYWxmICsgMV0KICAgIHJyID0gbnAuaHlwb3QoeHgsIHl5KQogICAgcmluZ3MgPSBbKHJyID49IHEpICYgKHJyIDwgcSArIDEpIGZvciBxIGluIHJhbmdlKGhhbGYpXQogICAgaW5uZXIgPSByciA8IDYKCiAgICBkZWYgc2hhcGVfb2Yoc3RhbXApOgogICAgICAgICIiIlRoZSByYWRpYWwgcHJvZmlsZSwgZGl2aWRlZCBieSB0aGUgbGlnaHQgaW4gdGhlIGNvcmUgLSBzaGFwZSBvbmx5LiIiIgogICAgICAgIHRvdGFsID0gc3RhbXBbaW5uZXJdLnN1bSgpCiAgICAgICAgaWYgdG90YWwgPD0gMDoKICAgICAgICAgICAgcmV0dXJuIE5vbmUKICAgICAgICByZXR1cm4gbnAuYXJyYXkoW3N0YW1wW21dLm1lYW4oKSBmb3IgbSBpbiByaW5nc10pIC8gdG90YWwKCiAgICBkZWYgbWlzbWF0Y2goc2lnbWEpOgogICAgICAgIGEgPSBnYXVzc2lhbl9maWx0ZXIoQTAsIHNpZ21hKSBpZiBzaWdtYSA+IDAgZWxzZSBBMAogICAgICAgIHIgPSBnYXVzc2lhbl9maWx0ZXIoUjAsIC1zaWdtYSkgaWYgc2lnbWEgPCAwIGVsc2UgUjAKICAgICAgICBzYSwgc3IgPSBzaGFwZV9vZihhKSwgc2hhcGVfb2YocikKICAgICAgICBpZiBzYSBpcyBOb25lIG9yIHNyIGlzIE5vbmU6CiAgICAgICAgICAgIHJldHVybiBucC5pbmYKICAgICAgICByZXR1cm4gZmxvYXQoKChzYSAtIHNyKSAqKiAyKS5zdW0oKSkKCiAgICBncmlkID0gbnAuYXJhbmdlKC1tYXhfc2lnbWEsIG1heF9zaWdtYSArIHN0ZXAsIHN0ZXApCiAgICBzY29yZXMgPSBucC5hcnJheShbbWlzbWF0Y2gocykgZm9yIHMgaW4gZ3JpZF0pCiAgICBiZXN0ID0gZmxvYXQoZ3JpZFtpbnQobnAuYXJnbWluKHNjb3JlcykpXSkKICAgIGlmIGFicyhiZXN0KSA8IHN0ZXA6CiAgICAgICAgYmVzdCA9IDAuMAoKICAgIGRpYWcgPSB7InNpZ21hIjogYmVzdCwKICAgICAgICAgICAgIm1pc21hdGNoX2JlZm9yZSI6IG1pc21hdGNoKDAuMCksCiAgICAgICAgICAgICJtaXNtYXRjaF9hZnRlciI6IG1pc21hdGNoKGJlc3QpLAogICAgICAgICAgICAiYmx1cnJlZCI6ICgiSC1hbHBoYSIgaWYgYmVzdCA+IDAgZWxzZSAiUiIgaWYgYmVzdCA8IDAgZWxzZSAibmVpdGhlciIpfQogICAgaWYgYmVzdCA+IDA6CiAgICAgICAgaGEgPSBnYXVzc2lhbl9maWx0ZXIoaGEsIGJlc3QpCiAgICBlbGlmIGJlc3QgPCAwOgogICAgICAgIHJlZCA9IGdhdXNzaWFuX2ZpbHRlcihyZWQsIC1iZXN0KQogICAgcmV0dXJuIGhhLCByZWQsIGFicyhiZXN0KSwgZGlhZwoKZGVmIGRlc2NyaWJlX3BzZl9tYXRjaChkaWFnKToKICAgICIiIldoYXQgdGhlIHNlYXJjaCBzZXR0bGVkIG9uLCBhbmQgYnkgaG93IG11Y2ggaXQgaW1wcm92ZWQgdGhpbmdzLiIiIgogICAgaWYgZGlhZy5nZXQoIndoeSIpOgogICAgICAgIHJldHVybiAiICBwb2ludCBzcHJlYWQgZnVuY3Rpb25zIGxlZnQgYWxvbmUgKHt9KSIuZm9ybWF0KGRpYWdbIndoeSJdKQogICAgaWYgZGlhZ1siYmx1cnJlZCJdID09ICJuZWl0aGVyIjoKICAgICAgICByZXR1cm4gIiAgdGhlIHN0YXJzIGFscmVhZHkgaGF2ZSB0aGUgc2FtZSBzaGFwZSBpbiBib3RoIGZyYW1lcyIKICAgIGJlZm9yZSwgYWZ0ZXIgPSBkaWFnWyJtaXNtYXRjaF9iZWZvcmUiXSwgZGlhZ1sibWlzbWF0Y2hfYWZ0ZXIiXQogICAgbGluZSA9ICgiICBibHVycmVkIHRoZSB7fSBmcmFtZSBieSBzaWdtYSA9IHs6LjJmfSBweCwgY2hvc2VuIGJ5IGNvbXBhcmluZyB0aGUgIgogICAgICAgICAgICAic3RhY2tlZCBzdGFyIHJpbmcgYnkgcmluZyIuZm9ybWF0KGRpYWdbImJsdXJyZWQiXSwgYWJzKGRpYWdbInNpZ21hIl0pKSkKICAgIGlmIGJlZm9yZSA+IDA6CiAgICAgICAgbGluZSArPSAoIlxuICAgIHRoZSBkaXNhZ3JlZW1lbnQgYmV0d2VlbiB0aGUgdHdvIHByb2ZpbGVzIGZlbGwgdG8gezouMGZ9JSAiCiAgICAgICAgICAgICAgICAgIm9mIHdoYXQgaXQgd2FzIi5mb3JtYXQoMTAwLjAgKiBhZnRlciAvIGJlZm9yZSkpCiAgICByZXR1cm4gbGluZQoKZGVmIGNvbnRpbnV1bV9mYWN0b3IoaGEsIHJlZCwgbWFzaz1Ob25lLCBuX3N0YXJzPTIwMCwgcG9zaXRpb25zPU5vbmUpOgogICAgIiIiSG93IG11Y2ggdG8gYnJpZ2h0ZW4gdGhlIG5hcnJvdyBmcmFtZSBiZWZvcmUgc3VidHJhY3RpbmcgdGhlIGJyb2FkIG9uZS4KCiAgICBBIGZvcmVncm91bmQgc3RhciBlbWl0cyBubyBILWFscGhhIGxpbmUgd29ydGggc3BlYWtpbmcgb2YsIHNvIHdoYXRldmVyIGl0IHB1dHMKICAgIGludG8gdGhlIG5hcnJvdyBmcmFtZSBpcyBwdXJlIGNvbnRpbnV1bSwgYW5kIHRoZSByYXRpbyBvZiBpdHMgYnJpZ2h0bmVzcyBpbiB0aGUKICAgIHR3byBmcmFtZXMgaXMgdGhlIGZhY3RvciB3ZSB3YW50LgoKICAgICJGb3JlZ3JvdW5kIiBpcyB0aGUgd29yZCB0aGF0IG1hdHRlcnMsIGFuZCBpdCBpcyB3aHkgYHBvc2l0aW9uc2AgZXhpc3RzLiBQb2ludAogICAgc291cmNlcyBmb3VuZCBpbiB0aGUgaW1hZ2UgYXJlIGZvcmVncm91bmQgb25seSB3aGVuIHRoZSBnYWxheHkgaXMgZmFyIGVub3VnaAogICAgYXdheSBmb3IgaXRzIG93biBzdGFycyBub3QgdG8gYmUgcmVzb2x2ZWQuIEluIGEgbmVhcmJ5IGdhbGF4eSB0aGV5IGFyZSwgaW4KICAgIHRoZWlyIHRob3VzYW5kcywgYW5kIHRoZXkgYXJlIG5vdCBwdXJlIGNvbnRpbnV1bSAtIG1lYXN1cmVkIG9uIE0zMyB0aGUgbWl4dHVyZQogICAgZ2F2ZSBhIHN0YXItdG8tc3RhciBzY2F0dGVyIG9mIDY2IHBlcmNlbnQsIHdoaWNoIGlzIG5vIGZhY3RvciBhdCBhbGwuIFBhc3NpbmcKICAgIGNhdGFsb2d1ZSBwb3NpdGlvbnMgb2Ygc3RhcnMgd2l0aCBhIG1lYXN1cmFibGUgcGFyYWxsYXggc2V0dGxlcyB0aGUgcXVlc3Rpb24KICAgIHBoeXNpY2FsbHkgaW5zdGVhZCBvZiBob3BpbmcgZGlzdGFuY2Ugc2V0dGxlcyBpdC4KCiAgICBUaGUgbWVkaWFuIG9mIHRoZSBwZXItc3RhciByYXRpbyBpcyB1c2VkIHJhdGhlciB0aGFuIGEgZml0LCBiZWNhdXNlIHRoZSBtZWRpYW4KICAgIGlzIG5vdCBkcmFnZ2VkIGFyb3VuZCBieSB0aGUgZmV3IHN0YXJzIHRoYXQgZG8gaGF2ZSBsaW5lIGVtaXNzaW9uLCBvciBhcmUKICAgIHNhdHVyYXRlZCwgb3Igc2l0IG9uIGEgYnJpZ2h0IHBpZWNlIG9mIHRoZSBnYWxheHkuCgogICAgUmV0dXJucyB0aGUgZmFjdG9yIGFuZCBkaWFnbm9zdGljcyAtIGluIHBhcnRpY3VsYXIgdGhlIHN0YXItdG8tc3RhciBzY2F0dGVyLAogICAgd2hpY2ggaXMgdGhlIGhvbmVzdCB1bmNlcnRhaW50eSBvbiB0aGUgc3VidHJhY3Rpb24uCiAgICAiIiIKICAgIGZyb20gcGhvdHV0aWxzLmRldGVjdGlvbiBpbXBvcnQgREFPU3RhckZpbmRlcgogICAgZnJvbSBwaG90dXRpbHMuYXBlcnR1cmUgaW1wb3J0IENpcmN1bGFyQXBlcnR1cmUsIENpcmN1bGFyQW5udWx1cywgYXBlcnR1cmVfcGhvdG9tZXRyeQogICAgZnJvbSBhc3Ryb3B5LnN0YXRzIGltcG9ydCBzaWdtYV9jbGlwcGVkX3N0YXRzCgogICAgaWYgcG9zaXRpb25zIGlzIG5vdCBOb25lIGFuZCBsZW4ocG9zaXRpb25zKSA+PSAxMDoKICAgICAgICAjIEEgZ2VuZXJvdXMgc2xpY2Ugb2YgdGhlIGNhdGFsb2d1ZSwgdGhlbiB0aGUgY2hvaWNlIGlzIG1hZGUgb24gd2hhdCB0aGVzZQogICAgICAgICMgZnJhbWVzIGFjdHVhbGx5IHJlY29yZGVkIHJhdGhlciB0aGFuIG9uIHdoYXQgdGhlIGNhdGFsb2d1ZSBzYXlzLiBHYWlhJ3MKICAgICAgICAjIEcgbWFnbml0dWRlIGlzIG5vdCB0aGlzIHRlbGVzY29wZSdzIFIsIGFuZCBhIHN0YXIgdGhhdCBpcyByZXNwZWN0YWJsZSBpbgogICAgICAgICMgdGhlIGNhdGFsb2d1ZSBjYW4gYmUgYSBoYW5kZnVsIG9mIGNvdW50cyB0aHJvdWdoIGEgbmFycm93IGZpbHRlciAtIHdoZXJlCiAgICAgICAgIyB0aGUgcmF0aW8gaXQgZ2l2ZXMgaXMgbm90IGEgbWVhc3VyZW1lbnQgYnV0IGEgZGl2aXNpb24gYnkgbm9pc2UuCiAgICAgICAgcG9zID0gbnAuYXNhcnJheShwb3NpdGlvbnMsIGR0eXBlPWZsb2F0KVs6NCAqIG5fc3RhcnNdCiAgICBlbHNlOgogICAgICAgIG1lYW5fciwgbWVkX3IsIHN0ZF9yID0gc2lnbWFfY2xpcHBlZF9zdGF0cyhyZWQsIHNpZ21hPTMuMCwgbWFzaz1tYXNrKQogICAgICAgIGZpbmRlciA9IERBT1N0YXJGaW5kZXIoZndobT00LjAsIHRocmVzaG9sZD0yNS4wICogc3RkX3IpCiAgICAgICAgc3RhcnMgPSBmaW5kZXIocmVkIC0gbWVkX3IsIG1hc2s9bWFzaykKICAgICAgICBpZiBzdGFycyBpcyBOb25lIG9yIGxlbihzdGFycykgPCA1OgogICAgICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoInRvbyBmZXcgc3RhcnMgZm91bmQgdG8gc2V0IHRoZSBjb250aW51dW0gZmFjdG9yIikKCiAgICAgICAgc3RhcnMuc29ydCgiZmx1eCIpCiAgICAgICAgc3RhcnMgPSBzdGFyc1s6Oi0xXVs6bl9zdGFyc10KICAgICAgICBwb3MgPSBucC50cmFuc3Bvc2UoKHN0YXJzWyJ4Y2VudHJvaWQiXSwgc3RhcnNbInljZW50cm9pZCJdKSkKCiAgICBhcCA9IENpcmN1bGFyQXBlcnR1cmUocG9zLCByPTYuMCkKICAgIGFubiA9IENpcmN1bGFyQW5udWx1cyhwb3MsIHJfaW49MTAuMCwgcl9vdXQ9MTYuMCkKCiAgICBkZWYgbmV0X2ZsdXgoaW1nKToKICAgICAgICBwaG90ID0gYXBlcnR1cmVfcGhvdG9tZXRyeShpbWcsIFthcCwgYW5uXSkKICAgICAgICBia2dfbWVhbiA9IHBob3RbImFwZXJ0dXJlX3N1bV8xIl0gLyBhbm4uYXJlYQogICAgICAgIHJldHVybiBucC5hc2FycmF5KHBob3RbImFwZXJ0dXJlX3N1bV8wIl0gLSBia2dfbWVhbiAqIGFwLmFyZWEsIGR0eXBlPWZsb2F0KQoKICAgIGZfaGEsIGZfciA9IG5ldF9mbHV4KGhhKSwgbmV0X2ZsdXgocmVkKQoKICAgIG9rID0gbnAuaXNmaW5pdGUoZl9oYSkgJiBucC5pc2Zpbml0ZShmX3IpICYgKGZfaGEgPiAwKSAmIChmX3IgPiAwKQogICAgaWYgb2suc3VtKCkgPCA1OgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigibm8gdXNhYmxlIHN0YXJzIGZvciB0aGUgY29udGludXVtIGZhY3RvciIpCgogICAgIyBLZWVwIHRoZSBicmlnaHRlc3QsIG1lYXN1cmVkIGhlcmUuIFdpdGggYSBjYXRhbG9ndWUgbGlzdCB0aGlzIGlzIHdoYXQgdHVybnMKICAgICMgYSBzZXQgb2Ygc3RhcnMgdGhhdCBhcmUgbWVyZWx5IHJlYWwgaW50byBhIHNldCB0aGF0IGlzIHdvcnRoIG1lYXN1cmluZy4KICAgIGlmIG9rLnN1bSgpID4gbl9zdGFyczoKICAgICAgICBvcmRlciA9IG5wLmFyZ3NvcnQobnAud2hlcmUob2ssIGZfciwgLW5wLmluZikpWzo6LTFdWzpuX3N0YXJzXQogICAgICAgIGtlZXBfYnJpZ2h0ID0gbnAuemVyb3Mob2suc2hhcGUsIGR0eXBlPWJvb2wpCiAgICAgICAga2VlcF9icmlnaHRbb3JkZXJdID0gVHJ1ZQogICAgICAgIG9rICY9IGtlZXBfYnJpZ2h0CgogICAgcmF0aW8gPSBmX3Jbb2tdIC8gZl9oYVtva10KICAgIG1lZCA9IGZsb2F0KG5wLm1lZGlhbihyYXRpbykpCiAgICBzcHJlYWQgPSAxLjQ4MjYgKiBucC5tZWRpYW4obnAuYWJzKHJhdGlvIC0gbWVkKSkKICAgIGlmIHNwcmVhZCA+IDA6CiAgICAgICAgcmF0aW8gPSByYXRpb1tucC5hYnMocmF0aW8gLSBtZWQpIDwgMy4wICogc3ByZWFkXQoKICAgIGZhY3RvciA9IGZsb2F0KG5wLm1lZGlhbihyYXRpbykpCiAgICBzY2F0dGVyID0gZmxvYXQoMS40ODI2ICogbnAubWVkaWFuKG5wLmFicyhyYXRpbyAtIGZhY3RvcikpKQogICAgcmV0dXJuIGZhY3RvciwgewogICAgICAgICJuX3N0YXJzIjogaW50KHJhdGlvLnNpemUpLAogICAgICAgICJzY2F0dGVyIjogc2NhdHRlciwKICAgICAgICAicmVsYXRpdmVfc2NhdHRlciI6IHNjYXR0ZXIgLyBmYWN0b3IgaWYgZmFjdG9yIGVsc2UgZmxvYXQoIm5hbiIpLAogICAgfQoKZGVmIGJhY2tncm91bmRfYm94KHNoYXBlLCByZXF1ZXN0ZWQ9Tm9uZSwgcjI1X3B4PU5vbmUpOgogICAgIiIiSG93IGxhcmdlIGEgc3F1YXJlIHRvIGVzdGltYXRlIHRoZSBiYWNrZ3JvdW5kIGluLgoKICAgIFRoaXMgaXMgbm90IGEgZnJlZSBwYXJhbWV0ZXIuIFRoZSBiYWNrZ3JvdW5kIG1hcCBpcyBtZWFudCB0byBmb2xsb3cgdGhlIHNreQogICAgYW5kIHRoZSBnYWxheHkncyBvd24gc21vb3RoIGxpZ2h0LCBhbmQgdG8gbGVhdmUgdGhlIHJlZ2lvbnMgc2l0dGluZyBvbiB0b3Agb2YKICAgIGl0IGFsb25lLiBBIGJveCB0aGUgc2l6ZSBvZiBhIHNwaXJhbCBhcm0gZG9lcyB0aGUgb3Bwb3NpdGU6IGl0IHRyZWF0cyB0aGUgYXJtCiAgICBhcyBiYWNrZ3JvdW5kIGFuZCBzdWJ0cmFjdHMgYXdheSB0aGUgdmVyeSB0aGluZyB0aGUgZmFpbnQgcmVnaW9ucyBzdGFuZCBvbi4KICAgIE1lYXN1cmVkIG9uIHRoZSBNODMgZnJhbWVzLCBnb2luZyBmcm9tIGEgNjQtcGl4ZWwgYm94IHRvIGEgMjAwLXBpeGVsIG9uZSBmb3VuZAogICAgNzkgbW9yZSByZWdpb25zIGluc2lkZSB0aGUgZ2FsYXh5IHdoaWxlIHRoZSBjb3VudCBvdXQgaW4gdGhlIGVtcHR5IGZpZWxkLCB3aGVyZQogICAgbm90aGluZyByZWFsIGNhbiBiZSwgZGlkIG5vdCBtb3ZlIGF0IGFsbCAtIHNvIHRob3NlIDc5IHdlcmUgcmVhbCBhbmQgd2VyZSBiZWluZwogICAgZXJhc2VkLgoKICAgIEEgdHdlbGZ0aCBvZiB0aGUgZnJhbWUgaXMgd2lkZSBlbm91Z2ggdG8gYmUgd2VsbCBjbGVhciBvZiBhbnkgc2luZ2xlIHJlZ2lvbiBhbmQKICAgIHNtYWxsIGVub3VnaCB0byBzdGlsbCBmb2xsb3cgYSBncmFkaWVudCBhY3Jvc3MgdGhlIGZpZWxkLgogICAgIiIiCiAgICBpZiByZXF1ZXN0ZWQ6CiAgICAgICAgcmV0dXJuIGludChyZXF1ZXN0ZWQpCiAgICAjIFRpZWQgdG8gdGhlIGdhbGF4eSwgbm90IHRvIHRoZSBmcmFtZS4gQSBib3ggc2V0IGFzIGEgZnJhY3Rpb24gb2YgdGhlIGltYWdlCiAgICAjIG1ha2VzIHRoZSBhbnN3ZXIgZGVwZW5kIG9uIGhvdyBtdWNoIGVtcHR5IHNreSBoYXBwZW5lZCB0byBzdXJyb3VuZCB0aGUKICAgICMgZ2FsYXh5IHRoYXQgbmlnaHQgLSBhbmQgaXQgc2hvd2VkOiBjcm9wcGluZyB0aGUgZnJhbWUgdG8gdGhlIGdhbGF4eSwgd2hpY2gKICAgICMgc2hvdWxkIGhhdmUgY2hhbmdlZCBub3RoaW5nIGJ1dCB0aGUgcnVubmluZyB0aW1lLCBtb3ZlZCB0aGUgcmVnaW9uIGNvdW50IG9uCiAgICAjIE04MyBmcm9tIDM0OCB0byAyODIgcHVyZWx5IGJlY2F1c2UgdGhlIGJveCBzaHJhbmsgd2l0aCB0aGUgZnJhbWUuCiAgICAjCiAgICAjIEEgdGhpcmQgb2YgUjI1IGlzIHdoYXQgdGhlIG1lYXN1cmVtZW50IHBvaW50ZWQgdG8uIE9uIE04MywgUjI1IGlzIDU2MCBweCBhbmQKICAgICMgdGhlIGJveCB0aGF0IHdvcmtlZCBiZXN0IHdhcyAyMDAuCiAgICBpZiByMjVfcHg6CiAgICAgICAgcmV0dXJuIGludChtaW4obWF4KHIyNV9weCAvIDMuMCwgNjQpLCA0MDApKQogICAgcmV0dXJuIGludChtaW4obWF4KG1pbihzaGFwZSkgLy8gMTIsIDY0KSwgMjU2KSkKCmRlZiBkZXRlY3RfcmVnaW9ucyhuZXQsIHRocmVzaG9sZF9zaWdtYT00LjAsIG1pbl9waXhlbHM9NiwgbWFzaz1Ob25lLCBkZWJsZW5kPVRydWUsCiAgICAgICAgICAgICAgICAgICBib3hfc2l6ZT1Ob25lLCBubGV2ZWxzPTY0LCBjb250cmFzdD0xZS00LCByMjVfcHg9Tm9uZSk6CiAgICAiIiJGaW5kIEhJSSByZWdpb25zIGluIHRoZSBjb250aW51dW0tc3VidHJhY3RlZCBmcmFtZS4KCiAgICBISUkgcmVnaW9ucyBhcmUgZXh0ZW5kZWQgYmxvYnMsIG5vdCBwb2ludHMsIHNvIGNvbm5lY3RlZCBncm91cHMgb2YgcGl4ZWxzIGFib3ZlCiAgICBhIHRocmVzaG9sZCBhcmUgZ3Jvd24gcmF0aGVyIHRoYW4gc3RlbGxhciBwcm9maWxlcyBmaXR0ZWQuIFRvdWNoaW5nIGJsb2JzIGFyZQogICAgc2VwYXJhdGVkIGJ5IGRlYmxlbmRpbmcsIHdoaWNoIGlzIHdoYXQgc3RvcHMgb25lIGJyaWdodCBjb21wbGV4IGJlaW5nIGNvdW50ZWQKICAgIG9uY2Ugd2hlcmUgYSBwZXJzb24gY2xpY2tpbmcgYnkgZXllIHdvdWxkIGhhdmUgY291bnRlZCB0aHJlZS4KCiAgICB0aHJlc2hvbGRfc2lnbWEgaXMgaW4gdW5pdHMgb2YgdGhlIGxvY2FsIGJhY2tncm91bmQgbm9pc2UsIHNvIHRoZSBzYW1lIG51bWJlcgogICAgbWVhbnMgdGhlIHNhbWUgdGhpbmcgb24gYW55IGZyYW1lLiBJdCBpcyB0aGUgbW9zdCBjb25zZXF1ZW50aWFsIGNob2ljZSBpbiB0aGUKICAgIHdob2xlIHBpcGVsaW5lLCB3aGljaCBpcyB3aHkgaXQgaXMgd3JpdHRlbiBpbnRvIGV2ZXJ5IG91dHB1dCBmaWxlLgogICAgIiIiCiAgICBmcm9tIHBob3R1dGlscy5iYWNrZ3JvdW5kIGltcG9ydCBCYWNrZ3JvdW5kMkQsIE1lZGlhbkJhY2tncm91bmQKICAgIGZyb20gcGhvdHV0aWxzLnNlZ21lbnRhdGlvbiBpbXBvcnQgZGV0ZWN0X3NvdXJjZXMsIGRlYmxlbmRfc291cmNlcywgU291cmNlQ2F0YWxvZwogICAgZnJvbSBhc3Ryb3B5LmNvbnZvbHV0aW9uIGltcG9ydCBHYXVzc2lhbjJES2VybmVsLCBjb252b2x2ZQoKICAgIGJveCA9IGJhY2tncm91bmRfYm94KG5ldC5zaGFwZSwgYm94X3NpemUsIHIyNV9weCkKICAgIGJrZyA9IEJhY2tncm91bmQyRChuZXQsIGJveF9zaXplPWJveCwgZmlsdGVyX3NpemU9MywKICAgICAgICAgICAgICAgICAgICAgICBia2dfZXN0aW1hdG9yPU1lZGlhbkJhY2tncm91bmQoKSwgbWFzaz1tYXNrKQogICAgcmVzaWQgPSBuZXQgLSBia2cuYmFja2dyb3VuZAoKICAgIGtlcm5lbCA9IEdhdXNzaWFuMkRLZXJuZWwoMi4wKQogICAgc21vb3RoZWQgPSBjb252b2x2ZShyZXNpZCwga2VybmVsLCBtYXNrPW1hc2spCgogICAgIyBUaGUgdGhyZXNob2xkIGhhcyB0byBiZSBtZWFzdXJlZCBvbiB0aGUgaW1hZ2UgaXQgaXMgYXBwbGllZCB0by4gU21vb3RoaW5nCiAgICAjIGF2ZXJhZ2VzIG5laWdoYm91cmluZyBwaXhlbHMsIHNvIGl0IGxvd2VycyB0aGUgbm9pc2UgYnkgYWJvdXQgYSBmYWN0b3Igb2YKICAgICMgc2V2ZW4gZm9yIHRoaXMga2VybmVsOyBjb21wYXJpbmcgdGhlIHNtb290aGVkIGltYWdlIGFnYWluc3QgdGhlIHVuc21vb3RoZWQKICAgICMgbm9pc2UgLSB3aGljaCBpcyB3aGF0IHRoZSBmaXJzdCB2ZXJzaW9uIG9mIHRoaXMgZGlkIC0gbWFkZSBhIG5vbWluYWwgIjMKICAgICMgc2lnbWEiIGJlaGF2ZSBsaWtlIHR3ZW50eSwgYW5kIHF1aWV0bHkgdGhyZXcgYXdheSBtb3N0IG9mIHRoZSBmYWludCByZWdpb25zLgogICAgc21vb3RoX2JrZyA9IEJhY2tncm91bmQyRChzbW9vdGhlZCwgYm94X3NpemU9Ym94LCBmaWx0ZXJfc2l6ZT0zLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICBia2dfZXN0aW1hdG9yPU1lZGlhbkJhY2tncm91bmQoKSwgbWFzaz1tYXNrKQogICAgdGhyZXNob2xkID0gdGhyZXNob2xkX3NpZ21hICogc21vb3RoX2JrZy5iYWNrZ3JvdW5kX3JtcwoKICAgIHNlZyA9IGRldGVjdF9zb3VyY2VzKHNtb290aGVkLCB0aHJlc2hvbGQsIG5waXhlbHM9bWluX3BpeGVscywgbWFzaz1tYXNrKQogICAgaWYgc2VnIGlzIE5vbmU6CiAgICAgICAgcmV0dXJuIE5vbmUsIGJrZywgcmVzaWQKCiAgICBpZiBkZWJsZW5kOgogICAgICAgICMgU3BsaXQgcmF0aGVyIHRoYW4gbHVtcC4gQSBsYXJnZSBjb21wbGV4IGlzIHVzdWFsbHkgc2V2ZXJhbCByZWdpb25zIHRoYXQKICAgICAgICAjIGhhcHBlbiB0byB0b3VjaCwgYW5kIGF0IHRoaXMgZGlzdGFuY2Ugb25lIGJsb2IgY2FuIGJlIGh1bmRyZWRzIG9mIHBhcnNlY3MKICAgICAgICAjIGFjcm9zcy4gU3BsaXR0aW5nIGl0IGNoYW5nZXMgdGhlIGNvdW50IGFuZCB0aGUgc2l6ZSBkaXN0cmlidXRpb247IGl0CiAgICAgICAgIyBiYXJlbHkgbW92ZXMgdGhlIHJhZGlhbCBwcm9maWxlLCBiZWNhdXNlIHRoZSBwaWVjZXMgc2l0IHdoZXJlIHRoZSB3aG9sZQogICAgICAgICMgc2F0LiBPbiB0aGUgTTgzIGZyYW1lcyB0aGVzZSBzZXR0aW5ncyB0dXJuZWQgMjc3IGJsb2JzIGludG8gMzY0IHdpdGhvdXQKICAgICAgICAjIGFkZGluZyBhbnl0aGluZyB0byB0aGUgZW1wdHkgZmllbGQuCiAgICAgICAgc2VnID0gZGVibGVuZF9zb3VyY2VzKHNtb290aGVkLCBzZWcsIG5waXhlbHM9bWluX3BpeGVscywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbmxldmVscz1ubGV2ZWxzLCBjb250cmFzdD1jb250cmFzdCwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcHJvZ3Jlc3NfYmFyPUZhbHNlKQoKICAgIGNhdCA9IFNvdXJjZUNhdGFsb2cocmVzaWQsIHNlZywgbWFzaz1tYXNrKQogICAgcmV0dXJuIGNhdCwgYmtnLCByZXNpZAoKZGVmIG9mZnNldF9mcm9tX2NhdGFsb2d1ZShyZWZlcmVuY2UsIGltYWdlLCBwb3NpdGlvbnMsIGJveD0xMSwgbWF4X3NoaWZ0X3B4PTQuMCk6CiAgICAiIiJUaGUgc2hpZnQgYmV0d2VlbiB0d28gZnJhbWVzLCBtZWFzdXJlZCBvbiBvbmUgYWdyZWVkIGxpc3Qgb2Ygc3RhcnMuCgogICAgV0hZIE5PVCBKVVNUIERFVEVDVCBTVEFSUyBJTiBCT1RICiAgICBCZWNhdXNlIHRoZSBicmlnaHRlc3QgdGhyZWUgaHVuZHJlZCBzb3VyY2VzIGluIGEgbmFycm93LWJhbmQgZnJhbWUgYXJlIG5vdCB0aGUKICAgIGJyaWdodGVzdCB0aHJlZSBodW5kcmVkIGluIGEgYnJvYWQtYmFuZCBvbmUgLSBkaWZmZXJlbnQgZmlsdGVycyBzZWUgZGlmZmVyZW50CiAgICBzdGFycyAtIHNvIHR3byBpbmRlcGVuZGVudGx5IGRldGVjdGVkIGxpc3RzIG92ZXJsYXAgcG9vcmx5IGFuZCB0aGUgcGFpcnMgdGhhdAogICAgZG8gbWF0Y2ggYXJlIHRoZSBvbmVzIHRoYXQgaGFwcGVuZWQgdG8gc3Vydml2ZS4gT24gYW4gTTMzIHBhaXIgdGhpcyBsZWZ0IDIxCiAgICB1c2FibGUgc3RhcnMgb3V0IG9mIHNpeCBodW5kcmVkLCB3aXRoIGEgc2NhdHRlciBvZiB0d28gcGl4ZWxzLgoKICAgIEdpdmVuIGEgbGlzdCBvZiBwb3NpdGlvbnMgZnJvbSBhIGNhdGFsb2d1ZSwgdGhlIHF1ZXN0aW9uIGRpc2FwcGVhcnM6IHRoZSBzYW1lCiAgICBzdGFyIGlzIGNlbnRyb2lkZWQgaW4gYm90aCBmcmFtZXMsIGFuZCBldmVyeSBvbmUgb2YgdGhlbSBjb250cmlidXRlcy4KCiAgICBXSFkgVEhFIFNLWSBDT01FUyBPRkYgRUFDSCBTVEFNUCBGSVJTVAogICAgU2FtZSByZWFzb24gYXMgaW4gY2VudHJlX29uX3N0YXJzLCBhbmQgdGhlIGNvbnNlcXVlbmNlIGhlcmUgaXMgd29yc2UuIEEgY2VudHJlCiAgICBvZiBsaWdodCBjb21wdXRlZCB3aXRoIHRoZSBza3kgc3RpbGwgaW4gdGhlIHN0YW1wIGlzIGRvbWluYXRlZCBieSB0aGUgc2t5LCBzbwogICAgaXQgbGFuZHMgbmVhciB0aGUgbWlkZGxlIG9mIHRoZSBib3ggd2hhdGV2ZXIgdGhlIHN0YXIgZG9lcyAtIGluIGJvdGggZnJhbWVzLgogICAgVGhlIHR3byBhbnN3ZXJzIHRoZW4gYWdyZWUgYmVhdXRpZnVsbHkgYW5kIHRoZWlyIGRpZmZlcmVuY2UgaXMgY2xvc2UgdG8gemVybywKICAgIHNvIHRoaXMgZnVuY3Rpb24gcmVwb3J0ZWQgImFscmVhZHkgYWxpZ25lZCIgb24gYSBwYWlyIHRoYXQgd2FzIGEgZnVsbCBwaXhlbAogICAgYXBhcnQuIE9uIE04MyB0aGF0IHBpeGVsIGlzIHdoYXQgbGVmdCBldmVyeSBzdGFyIGFzIGEgZGFyayBjb3JlIGJlc2lkZSBhIGJyaWdodAogICAgY3Jlc2NlbnQgaW4gdGhlIHN1YnRyYWN0ZWQgZnJhbWUuCiAgICAiIiIKICAgIGlmIHBvc2l0aW9ucyBpcyBOb25lIG9yIGxlbihwb3NpdGlvbnMpIDwgMTA6CiAgICAgICAgcmV0dXJuIE5vbmUKCiAgICBwb3MgPSBucC5hc2FycmF5KHBvc2l0aW9ucywgZHR5cGU9ZmxvYXQpCiAgICBjYSA9IF9jZW50cm9pZF9sb2NhbGx5KHJlZmVyZW5jZSwgcG9zLCBib3gpCiAgICBjYiA9IF9jZW50cm9pZF9sb2NhbGx5KGltYWdlLCBwb3MsIGJveCkKICAgIGlmIGNhIGlzIE5vbmUgb3IgY2IgaXMgTm9uZToKICAgICAgICByZXR1cm4gTm9uZQoKICAgIGR4LCBkeSA9IGNiWzosIDBdIC0gY2FbOiwgMF0sIGNiWzosIDFdIC0gY2FbOiwgMV0KICAgIG9rID0gKG5wLmlzZmluaXRlKGR4KSAmIG5wLmlzZmluaXRlKGR5KQogICAgICAgICAgJiAobnAuYWJzKGR4KSA8IG1heF9zaGlmdF9weCkgJiAobnAuYWJzKGR5KSA8IG1heF9zaGlmdF9weCkpCiAgICBpZiBvay5zdW0oKSA8IDEwOgogICAgICAgIHJldHVybiBOb25lCgogICAgbXgsIG15ID0gZmxvYXQobnAubWVkaWFuKGR4W29rXSkpLCBmbG9hdChucC5tZWRpYW4oZHlbb2tdKSkKICAgIHJldHVybiBteCwgbXksIHsKICAgICAgICAibl9wYWlycyI6IGludChvay5zdW0oKSksCiAgICAgICAgInNjYXR0ZXJfeCI6IGZsb2F0KDEuNDgyNiAqIG5wLm1lZGlhbihucC5hYnMoZHhbb2tdIC0gbXgpKSksCiAgICAgICAgInNjYXR0ZXJfeSI6IGZsb2F0KDEuNDgyNiAqIG5wLm1lZGlhbihucC5hYnMoZHlbb2tdIC0gbXkpKSksCiAgICAgICAgImZyb21fY2F0YWxvZ3VlIjogVHJ1ZSwKICAgIH0KCmRlZiBtZWFzdXJlX29mZnNldChyZWZlcmVuY2UsIGltYWdlLCBmd2htPTUuNSwgdGhyZXNob2xkX3NpZ21hPTIwLjAsIG5fc3RhcnM9MzAwLAogICAgICAgICAgICAgICAgICAgbWF4X21hdGNoX3B4PTQuMCk6CiAgICAiIiJUaGUgc2hpZnQgc3RpbGwgbGVmdCBiZXR3ZWVuIHR3byBmcmFtZXMgYWZ0ZXIgdGhleSB3ZXJlIHB1dCBvbiBvbmUgZ3JpZC4KCiAgICBXSFkgVEhJUyBJUyBORUVERUQgQUZURVIgQSBXQ1MgQUxJR05NRU5UCiAgICBSZXNhbXBsaW5nIHRocm91Z2ggdHdvIHBsYXRlIHNvbHV0aW9ucyBpcyBvbmx5IGFzIGdvb2QgYXMgdGhvc2Ugc29sdXRpb25zLiBFYWNoCiAgICB3YXMgZml0dGVkIHNlcGFyYXRlbHksIG9uIGEgZGlmZmVyZW50IGV4cG9zdXJlLCBhbmQgZWFjaCBjYXJyaWVzIGl0cyBvd24gZXJyb3IKICAgIG9mIGEgZnJhY3Rpb24gb2YgYW4gYXJjc2Vjb25kLiBXaGF0IHN1cnZpdmVzIGlzIGEgc21hbGwgc3lzdGVtYXRpYyBzaGlmdCAtIG9uCiAgICB0aGUgZnJhbWVzIHRoaXMgd2FzIHdyaXR0ZW4gZm9yLCAwLjYgb2YgYSBwaXhlbCAtIGFuZCBpdCBpcyB2aXNpYmxlOiBldmVyeSBzdGFyCiAgICBsZWF2ZXMgYSBkYXJrLWFuZC1icmlnaHQgcGFpciBpbiB0aGUgc3VidHJhY3RlZCBpbWFnZSBpbnN0ZWFkIG9mIHZhbmlzaGluZywgYW5kCiAgICB0aGUgZGV0ZWN0b3IgcmVwb3J0cyB0aG9zZSBwYWlycyBhcyByZWdpb25zLgoKICAgIEEgc2hpZnQgaXMgbWVhc3VyYWJsZSB0byBmYXIgYmV0dGVyIHRoYW4gYSBwaXhlbCBiZWNhdXNlIGl0IGlzIGNvbW1vbiB0byBldmVyeQogICAgc3RhciBpbiB0aGUgZnJhbWUuIEhlcmUgaXQgaXMgdGhlIG1lZGlhbiBvdmVyIGEgZmV3IGh1bmRyZWQgb2YgdGhlbSwgYW5kIHRoZQogICAgc2NhdHRlciBhYm91dCB0aGF0IG1lZGlhbiBzYXlzIHdoZXRoZXIgaXQgcmVhbGx5IGlzIG9uZSByaWdpZCBzaGlmdCBvciB3aGV0aGVyCiAgICBzb21ldGhpbmcgZWxzZSBpcyB3cm9uZy4KCiAgICBSZXR1cm5zIChkeCwgZHksIGRpYWdub3N0aWNzKS4gZHgsIGR5IGFyZSBob3cgZmFyIGBpbWFnZWAgc2l0cyBmcm9tCiAgICBgcmVmZXJlbmNlYCwgc28gc2hpZnRpbmcgaXQgYnkgdGhlIG5lZ2F0aXZlIG9mIHRoZW0gYnJpbmdzIHRoZSB0d28gdG9nZXRoZXIuCiAgICAiIiIKICAgIGltcG9ydCBudW1weSBhcyBucAogICAgZnJvbSBwaG90dXRpbHMuZGV0ZWN0aW9uIGltcG9ydCBEQU9TdGFyRmluZGVyCiAgICBmcm9tIGFzdHJvcHkuc3RhdHMgaW1wb3J0IHNpZ21hX2NsaXBwZWRfc3RhdHMKICAgIGZyb20gc2NpcHkuc3BhdGlhbCBpbXBvcnQgY0tEVHJlZQoKICAgIGRlZiBicmlnaHRfc3RhcnMoaW1nKToKICAgICAgICBfLCBtZWQsIHNkID0gc2lnbWFfY2xpcHBlZF9zdGF0cyhpbWcsIHNpZ21hPTMuMCkKICAgICAgICB0YWJsZSA9IERBT1N0YXJGaW5kZXIoZndobT1md2htLCB0aHJlc2hvbGQ9dGhyZXNob2xkX3NpZ21hICogc2QpKGltZyAtIG1lZCkKICAgICAgICBpZiB0YWJsZSBpcyBOb25lIG9yIGxlbih0YWJsZSkgPT0gMDoKICAgICAgICAgICAgcmV0dXJuIG5wLmVtcHR5KCgwLCAyKSkKICAgICAgICB0YWJsZS5zb3J0KCJmbHV4IikKICAgICAgICB0YWJsZSA9IHRhYmxlWzo6LTFdWzpuX3N0YXJzXQogICAgICAgIHJldHVybiBucC50cmFuc3Bvc2UoKG5wLmFzYXJyYXkodGFibGVbInhjZW50cm9pZCJdLCBkdHlwZT1mbG9hdCksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbnAuYXNhcnJheSh0YWJsZVsieWNlbnRyb2lkIl0sIGR0eXBlPWZsb2F0KSkpCgogICAgYSwgYiA9IGJyaWdodF9zdGFycyhyZWZlcmVuY2UpLCBicmlnaHRfc3RhcnMoaW1hZ2UpCiAgICBpZiBsZW4oYSkgPCAxMCBvciBsZW4oYikgPCAxMDoKICAgICAgICByZXR1cm4gMC4wLCAwLjAsIHsibl9wYWlycyI6IDAsCiAgICAgICAgICAgICAgICAgICAgICAgICAgIndoeSI6ICJ0b28gZmV3IHN0YXJzIHRvIG1lYXN1cmUgYSBzaGlmdCJ9CgogICAgZGlzdGFuY2UsIGluZGV4ID0gY0tEVHJlZShiKS5xdWVyeShhLCBrPTEpCiAgICBvayA9IGRpc3RhbmNlIDwgbWF4X21hdGNoX3B4CiAgICBpZiBvay5zdW0oKSA8IDEwOgogICAgICAgIHJldHVybiAwLjAsIDAuMCwgeyJuX3BhaXJzIjogaW50KG9rLnN1bSgpKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAid2h5IjogInRvbyBmZXcgc3RhcnMgY2xvc2UgZW5vdWdoIHRvIHBhaXIgdXAifQoKICAgIGR4ID0gYltpbmRleFtva10sIDBdIC0gYVtvaywgMF0KICAgIGR5ID0gYltpbmRleFtva10sIDFdIC0gYVtvaywgMV0KICAgIG14LCBteSA9IGZsb2F0KG5wLm1lZGlhbihkeCkpLCBmbG9hdChucC5tZWRpYW4oZHkpKQogICAgcmV0dXJuIG14LCBteSwgewogICAgICAgICJuX3BhaXJzIjogaW50KG9rLnN1bSgpKSwKICAgICAgICAic2NhdHRlcl94IjogZmxvYXQoMS40ODI2ICogbnAubWVkaWFuKG5wLmFicyhkeCAtIG14KSkpLAogICAgICAgICJzY2F0dGVyX3kiOiBmbG9hdCgxLjQ4MjYgKiBucC5tZWRpYW4obnAuYWJzKGR5IC0gbXkpKSksCiAgICB9CgpkZWYgc2hpZnRfZnJhbWUoaW1hZ2UsIGR4LCBkeSk6CiAgICAiIiJNb3ZlIGEgZnJhbWUgYnkgYSBmcmFjdGlvbiBvZiBhIHBpeGVsLCB0byB1bmRvIGEgbWVhc3VyZWQgb2Zmc2V0LgoKICAgIFRoZSBibGFuayBtYXJnaW4gbGVmdCBieSBhbiBlYXJsaWVyIHJlc2FtcGxpbmcgaXMgbm90IGEgbnVtYmVyLCBhbmQgYSBjdWJpYwogICAgc3BsaW5lIHJ1biBvdmVyIGl0IHdvdWxkIHNwcmVhZCB0aGF0IG5vdGhpbmduZXNzIGFjcm9zcyB0aGUgd2hvbGUgZnJhbWUgLSB0aGUKICAgIGZpcnN0IHZlcnNpb24gb2YgdGhpcyBkaWQgZXhhY3RseSB0aGF0LCBhbmQgZXZlcnkgbWVhc3VyZW1lbnQgZG93bnN0cmVhbSBjYW1lCiAgICBiYWNrIGVtcHR5LiBTbyB0aGUgZ2FwcyBhcmUgZmlsbGVkIGJlZm9yZSB0aGUgc2hpZnQgYW5kIHB1dCBiYWNrIGFmdGVyLCB3aGljaAogICAga2VlcHMgdGhlbSBleGFjdGx5IHdoZXJlIHRoZXkgd2VyZS4KICAgICIiIgogICAgaW1wb3J0IG51bXB5IGFzIG5wCiAgICBmcm9tIHNjaXB5Lm5kaW1hZ2UgaW1wb3J0IHNoaWZ0IGFzIG5kc2hpZnQKCiAgICBrbm93biA9IG5wLmlzZmluaXRlKGltYWdlKQogICAgbW92ZWQgPSBuZHNoaWZ0KG5wLndoZXJlKGtub3duLCBpbWFnZSwgMC4wKSwgKC1keSwgLWR4KSwgb3JkZXI9MywgbW9kZT0ibmVhcmVzdCIpCiAgICBzdGlsbF9rbm93biA9IG5kc2hpZnQoa25vd24uYXN0eXBlKGZsb2F0KSwgKC1keSwgLWR4KSwgb3JkZXI9MSwgbW9kZT0ibmVhcmVzdCIpCiAgICBtb3ZlZFtzdGlsbF9rbm93biA8IDAuNV0gPSBucC5uYW4KICAgIHJldHVybiBtb3ZlZAoKZGVmIHJlZmluZV9hbGlnbm1lbnQocmVmZXJlbmNlLCBpbWFnZSwgbGFiZWwsIG1pbl9zaGlmdF9weD0wLjA1LCBwb3NpdGlvbnM9Tm9uZSk6CiAgICAiIiJNZWFzdXJlIHRoZSBsZWZ0b3ZlciBzaGlmdCBhZ2FpbnN0IGEgcmVmZXJlbmNlIGZyYW1lIGFuZCB0YWtlIGl0IG91dC4KCiAgICBXaGVuIGEgbGlzdCBvZiBjYXRhbG9ndWUgc3RhciBwb3NpdGlvbnMgaXMgYXZhaWxhYmxlIHRoZSBzaGlmdCBpcyBtZWFzdXJlZCBvbgogICAgdGhvc2UsIHRoZSBzYW1lIHN0YXJzIGluIGJvdGggZnJhbWVzLiBPdGhlcndpc2UgdGhlIGZyYW1lcyBhcmUgYXNrZWQgdG8gc3VwcGx5CiAgICB0aGVpciBvd24gc3RhcnMsIHdoaWNoIHdvcmtzIGJldHdlZW4gZXhwb3N1cmVzIHRocm91Z2ggb25lIGZpbHRlciBhbmQgd29ya3MKICAgIHBvb3JseSBiZXR3ZWVuIHR3byBkaWZmZXJlbnQgb25lcy4KICAgICIiIgogICAgZnJvbV9jYXRhbG9ndWUgPSBvZmZzZXRfZnJvbV9jYXRhbG9ndWUocmVmZXJlbmNlLCBpbWFnZSwgcG9zaXRpb25zKQogICAgaWYgZnJvbV9jYXRhbG9ndWUgaXMgbm90IE5vbmU6CiAgICAgICAgZHgsIGR5LCBkaWFnID0gZnJvbV9jYXRhbG9ndWUKICAgIGVsc2U6CiAgICAgICAgZHgsIGR5LCBkaWFnID0gbWVhc3VyZV9vZmZzZXQocmVmZXJlbmNlLCBpbWFnZSkKICAgIGlmIGRpYWcuZ2V0KCJuX3BhaXJzIiwgMCkgPCAxMDoKICAgICAgICBwcmludCgiICB7fTogY291bGQgbm90IGNoZWNrIHRoZSBhbGlnbm1lbnQgLSB7fSIuZm9ybWF0KAogICAgICAgICAgICBsYWJlbCwgZGlhZy5nZXQoIndoeSIsICJub3QgZW5vdWdoIHN0YXJzIikpKQogICAgICAgIHJldHVybiBpbWFnZQogICAgc2l6ZSA9IChkeCAqIGR4ICsgZHkgKiBkeSkgKiogMC41CiAgICBpZiBzaXplIDwgbWluX3NoaWZ0X3B4OgogICAgICAgIHByaW50KCIgIHt9OiBhbHJlYWR5IGFsaWduZWQgdG8gezouMmZ9IHB4LCBub3RoaW5nIHRvIGNvcnJlY3QiLmZvcm1hdCgKICAgICAgICAgICAgbGFiZWwsIHNpemUpKQogICAgICAgIHJldHVybiBpbWFnZQogICAgcHJpbnQoIiAge306IHt9e30gc3RhcnMgc2F5IGl0IHNpdHMgKHs6Ky4yZn0sIHs6Ky4yZn0pIHB4IGZyb20gdGhlIHJlZmVyZW5jZSAiCiAgICAgICAgICAiKHNjYXR0ZXIgezouMmZ9LCB7Oi4yZn0pOyBzaGlmdGVkIGJhY2siLmZvcm1hdCgKICAgICAgICAgICAgICBsYWJlbCwgImNhdGFsb2d1ZSAiIGlmIGRpYWcuZ2V0KCJmcm9tX2NhdGFsb2d1ZSIpIGVsc2UgIiIsCiAgICAgICAgICAgICAgZGlhZ1sibl9wYWlycyJdLCBkeCwgZHksCiAgICAgICAgICAgICAgZGlhZ1sic2NhdHRlcl94Il0sIGRpYWdbInNjYXR0ZXJfeSJdKSkKICAgIHJldHVybiBzaGlmdF9mcmFtZShpbWFnZSwgZHgsIGR5KQoKZGVmIHN0YXJfcG9zaXRpb25zKHJlZCwgbWFzaz1Ob25lLCBuX3N0YXJzPTQwMCwgdGhyZXNob2xkX3NpZ21hPTE1LjApOgogICAgIiIiUG9zaXRpb25zIG9mIHRoZSBicmlnaHQgcG9pbnQgc291cmNlcyBpbiB0aGUgYnJvYWQtYmFuZCBmcmFtZS4KCiAgICBFdmVuIGFmdGVyIG1hdGNoaW5nIGFuZCBzdWJ0cmFjdGlvbiBhIHN0YXIgcmFyZWx5IHZhbmlzaGVzIHBlcmZlY3RseTogYSBzbWFsbAogICAgZXJyb3IgaW4gdGhlIGZhY3RvciBvciB0aGUgYmx1cnJpbmcgbGVhdmVzIGEgcmVzaWR1YWwgZXhhY3RseSB3aGVyZSB0aGUgc3RhcgogICAgd2FzLiBUaG9zZSByZXNpZHVhbHMgYXJlIGNvbXBhY3QgYW5kIGJyaWdodCwgc28gdGhlIGRldGVjdG9yIHJlcG9ydHMgdGhlbSBhcwogICAgSElJIHJlZ2lvbnMuIEtub3dpbmcgd2hlcmUgdGhlIHN0YXJzIHdlcmUgbGV0cyB1cyB0aHJvdyB0aGVtIG91dC4KICAgICIiIgogICAgZnJvbSBwaG90dXRpbHMuZGV0ZWN0aW9uIGltcG9ydCBEQU9TdGFyRmluZGVyCiAgICBmcm9tIGFzdHJvcHkuc3RhdHMgaW1wb3J0IHNpZ21hX2NsaXBwZWRfc3RhdHMKCiAgICBtZWFuLCBtZWRpYW4sIHN0ZCA9IHNpZ21hX2NsaXBwZWRfc3RhdHMocmVkLCBzaWdtYT0zLjAsIG1hc2s9bWFzaykKICAgIGZpbmRlciA9IERBT1N0YXJGaW5kZXIoZndobT00LjAsIHRocmVzaG9sZD10aHJlc2hvbGRfc2lnbWEgKiBzdGQpCiAgICB0YmwgPSBmaW5kZXIocmVkIC0gbWVkaWFuLCBtYXNrPW1hc2spCiAgICBpZiB0YmwgaXMgTm9uZSBvciBsZW4odGJsKSA9PSAwOgogICAgICAgIHJldHVybiBucC5lbXB0eSgoMCwgMikpCiAgICB0Ymwuc29ydCgiZmx1eCIpCiAgICB0YmwgPSB0YmxbOjotMV1bOm5fc3RhcnNdCiAgICByZXR1cm4gbnAudHJhbnNwb3NlKChucC5hc2FycmF5KHRibFsieGNlbnRyb2lkIl0sIGR0eXBlPWZsb2F0KSwKICAgICAgICAgICAgICAgICAgICAgICAgIG5wLmFzYXJyYXkodGJsWyJ5Y2VudHJvaWQiXSwgZHR5cGU9ZmxvYXQpKSkKCmRlZiBzdWJ0cmFjdF9jb250aW51dW0oaGEsIHJlZCwgZmFjdG9yLCBtYXNrPU5vbmUpOgogICAgIiIiZmFjdG9yIHggSC1hbHBoYSAtIFIsIHdpdGggZWFjaCBmcmFtZSdzIG93biBza3kgdGFrZW4gb2ZmIGZpcnN0LgoKICAgIFdIWSBUSEUgU0tZIENPTUVTIE9GRiBCRUZPUkUgVEhFIE1VTFRJUExJQ0FUSU9OCiAgICBCZWNhdXNlIHRoZSBmYWN0b3IgaXMgbWVhbnQgdG8gc2NhbGUgdGhlIHN0YXJsaWdodCwgYW5kIHRoZSBmcmFtZSBob2xkcyB0d28KICAgIHRoaW5nczogc3RhcmxpZ2h0IGFuZCBza3kuIE11bHRpcGx5aW5nIHRoZSB3aG9sZSBmcmFtZSBtdWx0aXBsaWVzIHRoZSBza3kgd2l0aAogICAgaXQuIE9uIGFuIE0zMyBwYWlyIHRoZSBILWFscGhhIHNreSB3YXMgMTIuNiBjb3VudHMgYW5kIHRoZSBSIHNreSA5NS40LCBhbmQgdGhlCiAgICBmYWN0b3IgMjUuNyAtIHNvIHRoZSByZXN1bHQgc2F0IG9uIGEgcGVkZXN0YWwgb2YgMjUuNyB4IDEyLjYgLSA5NS40ID0gMjI5CiAgICBjb3VudHMsIHNldmVudGVlbiB0aW1lcyB0aGUgbm9pc2Ugb2YgdGhlIGltYWdlLgoKICAgIERldGVjdGlvbiBzdXJ2aXZlZCB0aGF0LCBiZWNhdXNlIGl0IGVzdGltYXRlcyBhIGJhY2tncm91bmQgYW5kIHJlbW92ZXMgaXQKICAgIGJlZm9yZSBsb29raW5nIGZvciBhbnl0aGluZy4gV2hhdCBkaWQgbm90IHN1cnZpdmUgd2FzIHRoZSBwaWN0dXJlOiB0aGUgd2hvbGUKICAgIGR5bmFtaWMgcmFuZ2Ugd2VudCBpbnRvIHRoZSBwZWRlc3RhbCwgZmFpbnQgZW1pc3Npb24gdmFuaXNoZWQgaW50byBhIGZsYXQgZ3JleSwKICAgIGFuZCBhbnkgZGlmZmVyZW5jZSBpbiBmbGF0LWZpZWxkaW5nIGJldHdlZW4gdGhlIHR3byBmaWx0ZXJzIHdhcyBhbXBsaWZpZWQgYnkKICAgIHRoZSBmYWN0b3IgYWxvbmcgd2l0aCBldmVyeXRoaW5nIGVsc2UuIEFueW9uZSBqdWRnaW5nIHRoZSBzdWJ0cmFjdGlvbiBieSBleWUgLQogICAgd2hpY2ggaXMgd2hhdCB0aGlzIHRvb2wgYXNrcyBmb3Igd2hlbmV2ZXIgdGhlIGZhY3RvciBsb29rcyBkb3VidGZ1bCAtIHdhcwogICAgbG9va2luZyBhdCBhbiBpbWFnZSBydWluZWQgYmVmb3JlIHRoZXkgc2F3IGl0LgoKICAgIFNreSBpcyBtZWFzdXJlZCB3aXRoIGNsaXBwaW5nLCBzbyB0aGUgZ2FsYXh5IGFuZCB0aGUgc3RhcnMgZG8gbm90IGRyYWcgaXQuCiAgICAiIiIKICAgIGZyb20gYXN0cm9weS5zdGF0cyBpbXBvcnQgc2lnbWFfY2xpcHBlZF9zdGF0cwoKICAgIHNreV9oYSA9IGZsb2F0KHNpZ21hX2NsaXBwZWRfc3RhdHMoaGEsIHNpZ21hPTMuMCwgbWFzaz1tYXNrKVsxXSkKICAgIHNreV9yZWQgPSBmbG9hdChzaWdtYV9jbGlwcGVkX3N0YXRzKHJlZCwgc2lnbWE9My4wLCBtYXNrPW1hc2spWzFdKQogICAgbmV0ID0gZmFjdG9yICogKGhhIC0gc2t5X2hhKSAtIChyZWQgLSBza3lfcmVkKQogICAgcmV0dXJuIG5ldCwgeyJza3lfaGEiOiBza3lfaGEsICJza3lfcmVkIjogc2t5X3JlZCwKICAgICAgICAgICAgICAgICAicGVkZXN0YWxfYXZvaWRlZCI6IGZhY3RvciAqIHNreV9oYSAtIHNreV9yZWR9CgpkZWYgbGluZV9leGNlc3MobmV0LCByZWQsIHgsIHksIHJhZGl1c19weD01LjApOgogICAgIiIiSG93IG11Y2ggbGluZSBlbWlzc2lvbiBlYWNoIGRldGVjdGlvbiBoYXMsIG5leHQgdG8gaXRzIG93biBjb250aW51dW0uCgogICAgVGhlIHJhdGlvIG9mIHdoYXQgaXMgbGVmdCBhZnRlciB0aGUgc3VidHJhY3Rpb24gdG8gd2hhdCB0aGUgY29udGludXVtIGZyYW1lCiAgICBob2xkcyBhdCB0aGUgc2FtZSBwbGFjZS4gQSBmb3JlZ3JvdW5kIHN0YXIgaGFzIG5vdGhpbmcgbGVmdCBvbmNlIHRoZQogICAgc3VidHJhY3Rpb24gaXMgcmlnaHQsIHNvIGl0cyByYXRpbyBzaXRzIG5lYXIgemVyby4gQW4gSElJIHJlZ2lvbiBpcyBlbWlzc2lvbgogICAgd2l0aCBvbmx5IGEgbW9kZXN0IGNvbnRpbnV1bSB1bmRlciBpdCwgc28gaXRzIHJhdGlvIGlzIG9mIG9yZGVyIG9uZSBvciBtb3JlLgogICAgIiIiCiAgICBmcm9tIHBob3R1dGlscy5hcGVydHVyZSBpbXBvcnQgQ2lyY3VsYXJBcGVydHVyZSwgYXBlcnR1cmVfcGhvdG9tZXRyeQoKICAgIHBvc2l0aW9ucyA9IG5wLmNvbHVtbl9zdGFjayhbbnAuYXNhcnJheSh4LCBkdHlwZT1mbG9hdCksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5wLmFzYXJyYXkoeSwgZHR5cGU9ZmxvYXQpXSkKICAgIGlmIG5vdCBsZW4ocG9zaXRpb25zKToKICAgICAgICByZXR1cm4gbnAuemVyb3MoMCkKICAgIGFwID0gQ2lyY3VsYXJBcGVydHVyZShwb3NpdGlvbnMsIHI9cmFkaXVzX3B4KQogICAgZl9uZXQgPSBucC5hc2FycmF5KGFwZXJ0dXJlX3Bob3RvbWV0cnkobnAubmFuX3RvX251bShuZXQpLCBhcClbImFwZXJ0dXJlX3N1bSJdLAogICAgICAgICAgICAgICAgICAgICAgIGR0eXBlPWZsb2F0KQogICAgZl9yZWQgPSBucC5hc2FycmF5KGFwZXJ0dXJlX3Bob3RvbWV0cnkobnAubmFuX3RvX251bShyZWQpLCBhcClbImFwZXJ0dXJlX3N1bSJdLAogICAgICAgICAgICAgICAgICAgICAgIGR0eXBlPWZsb2F0KQogICAgd2l0aCBucC5lcnJzdGF0ZShkaXZpZGU9Imlnbm9yZSIsIGludmFsaWQ9Imlnbm9yZSIpOgogICAgICAgIHJldHVybiBmX25ldCAvIG5wLndoZXJlKG5wLmFicyhmX3JlZCkgPiAxLjAsIG5wLmFicyhmX3JlZCksIG5wLm5hbikKCmRlZiBmaWx0ZXJfZGV0ZWN0aW9ucyh4LCB5LCBjZW50cmUsIHJfbWF4X3B4PU5vbmUsIHN0YXJfeHk9Tm9uZSwgc3Rhcl9yYWRpdXNfcHg9Ni4wLAogICAgICAgICAgICAgICAgICAgICAgZXhjZXNzPU5vbmUsIG1pbl9leGNlc3M9MC41LAogICAgICAgICAgICAgICAgICAgICAgbWlub3JfYXhpcz1Ob25lLCBlbG9uZ2F0aW9uPU5vbmUsCiAgICAgICAgICAgICAgICAgICAgICBtaW5fd2lkdGhfcHg9MS41LCBtYXhfZWxvbmdhdGlvbj0yLjUpOgogICAgIiIiRHJvcCBkZXRlY3Rpb25zIHRoYXQgY2Fubm90IGJlIEhJSSByZWdpb25zIGJlbG9uZ2luZyB0byB0aGlzIGdhbGF4eS4KCiAgICBUd28gY3V0cywgYm90aCBibHVudCBhbmQgYm90aCBuZWNlc3NhcnkuCgogICAgT3V0c2lkZSB0aGUgZ2FsYXh5OiBISUkgcmVnaW9ucyBsaXZlIGluIHRoZSBkaXNrLCBhIGZyYW1lIGlzIHVzdWFsbHkgbXVjaCB3aWRlcgogICAgdGhhbiB0aGUgZ2FsYXh5LCBhbmQgZXZlcnl0aGluZyBmb3VuZCBvdXQgaW4gdGhlIGVtcHR5IGNvcm5lcnMgaXMgbm9pc2Ugb3IgYW4KICAgIGFydGVmYWN0LiBDdXR0aW5nIGF0IHRoZSBpc29waG90YWwgcmFkaXVzIFIyNSByZW1vdmVzIHRoZW0gd2l0aG91dCB0b3VjaGluZyB0aGUKICAgIGRpc2ssIGFuZCBSMjUgaXMgYSBwdWJsaXNoZWQgbnVtYmVyIHJhdGhlciB0aGFuIGEganVkZ2VtZW50IGNhbGwuCgogICAgT24gdG9wIG9mIGEgc3Rhcjogc3VidHJhY3Rpb24gcmVzaWR1YWxzIHNpdCBleGFjdGx5IHdoZXJlIHRoZSBzdGFycyB3ZXJlLgoKICAgIFdIWSBUSEUgU1RBUiBDVVQgTkVFRFMgVEhFIFNFQ09ORCBURVNUCiAgICBBIGJyaWdodCBISUkgcmVnaW9uIGNvbnRhaW5zIHRoZSB5b3VuZyBjbHVzdGVyIHRoYXQgbGlnaHRzIGl0LCBhbmQgYSB5b3VuZwogICAgY2x1c3RlciBpcyBhIHBvaW50IHNvdXJjZSBpbiB0aGUgY29udGludXVtIGZyYW1lIC0gc28gYSBjdXQgdGhhdCByZW1vdmVzCiAgICBldmVyeXRoaW5nIHNpdHRpbmcgb24gYSBwb2ludCBzb3VyY2UgcmVtb3ZlcyB0aGUgYnJpZ2h0ZXN0IHJlZ2lvbnMgaW4gdGhlCiAgICBnYWxheHkgYWxvbmcgd2l0aCB0aGUgZm9yZWdyb3VuZCBzdGFycy4gT24gdGhlIE04MyBmcmFtZXMgdGhpcyB3YXMgbWVhc3VyZWQ6CiAgICBvZiBmb3J0eSBkZXRlY3Rpb25zIHRoZSBzdGFyIGN1dCB0aHJldyBhd2F5IGluc2lkZSB0aGUgZ2FsYXh5LCB0d2VudHktbmluZSBoYWQKICAgIHJlYWwgbGluZSBlbWlzc2lvbi4gVGhlaXIgZW1pc3Npb24gaXMgd2hhdCBzZXBhcmF0ZXMgdGhlbSAtIHRoZSByZWdpb25zIGtlcHQKICAgIGJ5IHRoZSByZXN0IG9mIHRoZSBwaXBlbGluZSBzaG93ZWQgYSByYXRpbyBhcm91bmQgMi4zLCBhbmQgdGhlIHBvaW50IHNvdXJjZXMKICAgIG91dCBpbiB0aGUgZW1wdHkgZmllbGQsIHdoaWNoIHJlYWxseSBhcmUgc3RhcnMsIGFyb3VuZCAwLjE3LiBTbyBhIGRldGVjdGlvbgogICAgc2l0dGluZyBvbiBhIHBvaW50IHNvdXJjZSBpcyBzdGlsbCBkcm9wcGVkLCB1bmxlc3MgaXQgaXMgZW1pdHRpbmcuCiAgICAiIiIKICAgIHggPSBucC5hc2FycmF5KHgsIGR0eXBlPWZsb2F0KQogICAgeSA9IG5wLmFzYXJyYXkoeSwgZHR5cGU9ZmxvYXQpCiAgICBrZWVwID0gbnAub25lcyh4LnNpemUsIGR0eXBlPWJvb2wpCiAgICByZXBvcnQgPSB7fQoKICAgICMgU2hhcGUuIEFuIEhJSSByZWdpb24gaXMgYSBib2R5IG9mIGdhcywgc28gaXQgaGFzIHdpZHRoOyB3aGF0IGlzIGxlZnQgb3ZlcgogICAgIyBmcm9tIGFuIGltcGVyZmVjdCBzdWJ0cmFjdGlvbiBpcyBhIHRocmVhZCAtIGEgcGl4ZWwgb3IgdHdvIGFjcm9zcywgb2Z0ZW4KICAgICMgY3VydmVkLCBmb2xsb3dpbmcgdGhlIGVkZ2Ugb2Ygd2hhdGV2ZXIgZGlkIG5vdCBjYW5jZWwuIE1lYXN1cmVkIG9uIHRoZSBNODMKICAgICMgZnJhbWVzLCBibG9icyBpbnNpZGUgdGhlIGdhbGF4eSBhcmUgdHlwaWNhbGx5IDEuOSBweCB3aWRlIGFuZCBibG9icyBvdXQgaW4KICAgICMgdGhlIGVtcHR5IGZpZWxkLCB3aGVyZSBub3RoaW5nIHJlYWwgY2FuIGJlLCAwLjUgcHguIFRoYXQgb25lIG51bWJlciBkb2VzCiAgICAjIG1vcmUgdG8gY2xlYW4gdGhlIGNhdGFsb2d1ZSB0aGFuIGFueSBicmlnaHRuZXNzIGN1dC4KICAgIGlmIG1pbm9yX2F4aXMgaXMgbm90IE5vbmU6CiAgICAgICAgd2lkZV9lbm91Z2ggPSBucC5pc2Zpbml0ZShtaW5vcl9heGlzKSAmIChucC5hc2FycmF5KG1pbm9yX2F4aXMpID49IG1pbl93aWR0aF9weCkKICAgICAgICByZXBvcnRbInRvbyB0aGluIHRvIGJlIGEgcmVnaW9uIl0gPSBpbnQoKH53aWRlX2Vub3VnaCAmIGtlZXApLnN1bSgpKQogICAgICAgIGtlZXAgJj0gd2lkZV9lbm91Z2gKCiAgICBpZiBlbG9uZ2F0aW9uIGlzIG5vdCBOb25lOgogICAgICAgIHJvdW5kX2Vub3VnaCA9IG5wLmlzZmluaXRlKGVsb25nYXRpb24pICYgKG5wLmFzYXJyYXkoZWxvbmdhdGlvbikgPD0gbWF4X2Vsb25nYXRpb24pCiAgICAgICAgcmVwb3J0WyJhIHN0cmVhaywgbm90IGEgcmVnaW9uIl0gPSBpbnQoKH5yb3VuZF9lbm91Z2ggJiBrZWVwKS5zdW0oKSkKICAgICAgICBrZWVwICY9IHJvdW5kX2Vub3VnaAoKICAgIGlmIHJfbWF4X3B4IGlzIG5vdCBOb25lOgogICAgICAgIGluc2lkZSA9IG5wLmh5cG90KHggLSBjZW50cmVbMF0sIHkgLSBjZW50cmVbMV0pIDw9IHJfbWF4X3B4CiAgICAgICAgcmVwb3J0WyJvdXRzaWRlIHRoZSBnYWxheHkiXSA9IGludCgofmluc2lkZSAmIGtlZXApLnN1bSgpKQogICAgICAgIGtlZXAgJj0gaW5zaWRlCgogICAgaWYgc3Rhcl94eSBpcyBub3QgTm9uZSBhbmQgbGVuKHN0YXJfeHkpOgogICAgICAgIGQgPSBucC5oeXBvdCh4WzosIE5vbmVdIC0gc3Rhcl94eVtOb25lLCA6LCAwXSwgeVs6LCBOb25lXSAtIHN0YXJfeHlbTm9uZSwgOiwgMV0pCiAgICAgICAgb25fc3RhciA9IGQubWluKGF4aXM9MSkgPCBzdGFyX3JhZGl1c19weAogICAgICAgIGlmIGV4Y2VzcyBpcyBub3QgTm9uZToKICAgICAgICAgICAgZW1pdHRpbmcgPSBucC5pc2Zpbml0ZShleGNlc3MpICYgKGV4Y2VzcyA+IG1pbl9leGNlc3MpCiAgICAgICAgICAgIHJlc2N1ZWQgPSBpbnQoKG9uX3N0YXIgJiBlbWl0dGluZyAmIGtlZXApLnN1bSgpKQogICAgICAgICAgICBpZiByZXNjdWVkOgogICAgICAgICAgICAgICAgcmVwb3J0WyJvbiBhIHN0YXIgYnV0IGVtaXR0aW5nLCBrZXB0Il0gPSByZXNjdWVkCiAgICAgICAgICAgIG9uX3N0YXIgPSBvbl9zdGFyICYgfmVtaXR0aW5nCiAgICAgICAgcmVwb3J0WyJvbiBhIHN0YXIiXSA9IGludCgob25fc3RhciAmIGtlZXApLnN1bSgpKQogICAgICAgIGtlZXAgJj0gfm9uX3N0YXIKCiAgICByZXR1cm4ga2VlcCwgcmVwb3J0CgpkZWYgY29tcGxldGVuZXNzX3ZzX3JhZGl1cyhyZXNpZCwgY2VudHJlLCBkZXRlY3Rfa3dhcmdzLCByX2VkZ2VzLAogICAgICAgICAgICAgICAgICAgICAgICAgICBuX3Blcl9yaW5nPTMwLCBwZWFrX2NvdW50cz0xLjAsIGZ3aG1fcHg9Ni4wLCBybmdfc2VlZD0xMSk6CiAgICAiIiJXaGF0IGZyYWN0aW9uIG9mIGluamVjdGVkIHJlZ2lvbnMgY29tZXMgYmFjaywgcmluZyBieSByaW5nLgoKICAgIFdIWSBCT1RIRVIKICAgIEEgcmFkaWFsIHByb2ZpbGUgaXMgdGhlIG1lYXN1cmVtZW50IG1vc3QgZWFzaWx5IGZha2VkIGJ5IGEgZGV0ZWN0aW9uIGxpbWl0IHRoYXQKICAgIGNoYW5nZXMgd2l0aCByYWRpdXMuIFRoZSBjZW50cmUgb2YgYSBnYWxheHkgaXMgYnJpZ2h0IGFuZCBjcm93ZGVkLCB0aGUgb3V0c2tpcnRzCiAgICBkYXJrIGFuZCBlbXB0eSwgc28gdGhlIGZhaW50ZXN0IHRoaW5nIHN0aWxsIHZpc2libGUgaXMgbm90IHRoZSBzYW1lIGluIGJvdGguIElmCiAgICBkZXRlY3Rpb24gZ2V0cyBoYXJkZXIgaW53YXJkcywgdGhlIHByb2ZpbGUgZ3Jvd3MgYSBjZW50cmFsIGhvbGUgdGhhdCBiZWxvbmdzIHRvCiAgICB0aGUgc2VhcmNoLCBub3QgdG8gdGhlIGdhbGF4eS4KCiAgICBIT1cKICAgIERyb3AgYXJ0aWZpY2lhbCByZWdpb25zIG9mIGtub3duIHBlYWsgYnJpZ2h0bmVzcyBpbnRvIHRoZSByZWFsIGltYWdlLCBydW4KICAgIGV4YWN0bHkgdGhlIHNhbWUgZGV0ZWN0aW9uIG9uIGl0LCBhbmQgY291bnQgaG93IG1hbnkgY29tZSBiYWNrLCByaW5nIGJ5IHJpbmcuCgogICAgcGVha19jb3VudHMgaXMgdGhlIHBlYWsgaGVpZ2h0IG9mIHRoZSBpbmplY3RlZCByZWdpb24gYW5kIHNob3VsZCBiZSBzZXQgaW4gdW5pdHMKICAgIG9mIHRoZSBiYWNrZ3JvdW5kIG5vaXNlLiBTY2FsaW5nIGluamVjdGVkIHJlZ2lvbnMgdG8gdGhlIGJyaWdodG5lc3Mgb2YgYW4KICAgIGF2ZXJhZ2UgcmVhbCBvbmUgcHV0cyB0aGVtIGh1bmRyZWRzIG9mIHNpZ21hIGFib3ZlIHRoZSB0aHJlc2hvbGQsIHdoZXJlCiAgICBldmVyeXRoaW5nIGlzIGZvdW5kIGV2ZXJ5d2hlcmUgYW5kIHRoZSB0ZXN0IG1lYXN1cmVzIG5vdGhpbmcuCiAgICAiIiIKICAgIGZyb20gc2NpcHkubmRpbWFnZSBpbXBvcnQgZ2F1c3NpYW5fZmlsdGVyCgogICAgcm5nID0gbnAucmFuZG9tLmRlZmF1bHRfcm5nKHJuZ19zZWVkKQogICAgc2lnbWEgPSBmd2htX3B4IC8gMi4zNTQ4CiAgICBueSwgbnggPSByZXNpZC5zaGFwZQogICAgcmVjb3ZlcmVkID0gbnAuemVyb3MobGVuKHJfZWRnZXMpIC0gMSkKICAgIGluamVjdGVkID0gbnAuemVyb3MobGVuKHJfZWRnZXMpIC0gMSkKCiAgICBmb3IgayBpbiByYW5nZShsZW4ocl9lZGdlcykgLSAxKToKICAgICAgICByX2xvLCByX2hpID0gcl9lZGdlc1trXSwgcl9lZGdlc1trICsgMV0KICAgICAgICBmYWtlID0gbnAuemVyb3NfbGlrZShyZXNpZCkKICAgICAgICB0cnV0aCA9IFtdCiAgICAgICAgZm9yIF8gaW4gcmFuZ2Uobl9wZXJfcmluZyk6CiAgICAgICAgICAgIHJhZCA9IG1hdGguc3FydChybmcudW5pZm9ybShyX2xvICoqIDIsIHJfaGkgKiogMikpCiAgICAgICAgICAgIHRoID0gcm5nLnVuaWZvcm0oMC4wLCAyLjAgKiBtYXRoLnBpKQogICAgICAgICAgICBweCA9IGNlbnRyZVswXSArIHJhZCAqIG1hdGguY29zKHRoKQogICAgICAgICAgICBweSA9IGNlbnRyZVsxXSArIHJhZCAqIG1hdGguc2luKHRoKQogICAgICAgICAgICBpZiBub3QgKDggPCBweCA8IG54IC0gOCBhbmQgOCA8IHB5IDwgbnkgLSA4KToKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgIGZha2VbaW50KHJvdW5kKHB5KSksIGludChyb3VuZChweCkpXSArPSBwZWFrX2NvdW50cwogICAgICAgICAgICB0cnV0aC5hcHBlbmQoKHB4LCBweSkpCiAgICAgICAgaWYgbm90IHRydXRoOgogICAgICAgICAgICBjb250aW51ZQoKICAgICAgICAjIEEgZGVsdGEgc21vb3RoZWQgYnkgYSBHYXVzc2lhbiBoYXMgcGVhayBhbXBsaXR1ZGUvKDIgcGkgc2lnbWFeMik7CiAgICAgICAgIyBtdWx0aXBseWluZyBpdCBiYWNrIGdpdmVzIGEgcmVnaW9uIHdob3NlIFBFQUsgaXMgcGVha19jb3VudHMuCiAgICAgICAgZmFrZSA9IGdhdXNzaWFuX2ZpbHRlcihmYWtlLCBzaWdtYSkgKiAoMi4wICogbWF0aC5waSAqIHNpZ21hICoqIDIpCiAgICAgICAgY2F0LCBfLCBfID0gZGV0ZWN0X3JlZ2lvbnMocmVzaWQgKyBmYWtlLCAqKmRldGVjdF9rd2FyZ3MpCgogICAgICAgIGlmIGNhdCBpcyBOb25lOgogICAgICAgICAgICBpbmplY3RlZFtrXSArPSBsZW4odHJ1dGgpCiAgICAgICAgICAgIGNvbnRpbnVlCgogICAgICAgIGZ4ID0gbnAuYXNhcnJheShjYXQueGNlbnRyb2lkLCBkdHlwZT1mbG9hdCkKICAgICAgICBmeSA9IG5wLmFzYXJyYXkoY2F0LnljZW50cm9pZCwgZHR5cGU9ZmxvYXQpCiAgICAgICAgIyBTZWdtZW50cyB0b3VjaGluZyB0aGUgbWFzayBjb21lIGJhY2sgd2l0aCBhIGNlbnRyb2lkIG9mIE5hTi4gTGVmdCBpbiwgYQogICAgICAgICMgc2luZ2xlIE5hTiBtYWtlcyBldmVyeSBkaXN0YW5jZSBjb21wYXJpc29uIGJlbG93IGZhbHNlIGFuZCB0aGUgbWVhc3VyZWQKICAgICAgICAjIGNvbXBsZXRlbmVzcyBjb2xsYXBzZXMgdG8gemVybyBldmVyeXdoZXJlLgogICAgICAgIGZpbml0ZSA9IG5wLmlzZmluaXRlKGZ4KSAmIG5wLmlzZmluaXRlKGZ5KQogICAgICAgIGZ4LCBmeSA9IGZ4W2Zpbml0ZV0sIGZ5W2Zpbml0ZV0KCiAgICAgICAgZm9yICh0eCwgdHkpIGluIHRydXRoOgogICAgICAgICAgICBpbmplY3RlZFtrXSArPSAxCiAgICAgICAgICAgIGlmIGZ4LnNpemUgYW5kIG5wLm1pbihucC5oeXBvdChmeCAtIHR4LCBmeSAtIHR5KSkgPCAzLjA6CiAgICAgICAgICAgICAgICByZWNvdmVyZWRba10gKz0gMQoKICAgIHdpdGggbnAuZXJyc3RhdGUoaW52YWxpZD0iaWdub3JlIiwgZGl2aWRlPSJpZ25vcmUiKToKICAgICAgICBmcmFjID0gbnAud2hlcmUoaW5qZWN0ZWQgPiAwLCByZWNvdmVyZWQgLyBpbmplY3RlZCwgbnAubmFuKQogICAgcmV0dXJuIGZyYWMsIGluamVjdGVkLCByZWNvdmVyZWQKCmRlZiBjaG9vc2Vfc3VidHJhY3Rpb24oaGEsIHJlZCwgcG9zaXRpb25zLCBwcmVmaXgsIGNlbnRyZT1Ob25lLCByMjVfcHg9Tm9uZSwKICAgICAgICAgICAgICAgICAgICAgICBmYWN0b3JzPU5vbmUsIGJsdXJzPU5vbmUsIGFzaz1Ob25lKToKICAgICIiIk9mZmVyIHRocmVlIHN1YnRyYWN0aW9ucyBhcyBwaWN0dXJlcyBhbmQgbGV0IHRoZSBwZXJzb24gcGljayBvbmUuCgogICAgV0hZIEEgUEVSU09OIERFQ0lERVMgVEhJUwogICAgQmVjYXVzZSBldmVyeSBhdXRvbWF0aWMgY3JpdGVyaW9uIHRyaWVkIGhlcmUgZGlzYWdyZWVkIHdpdGggdGhlIGV5ZSwgYW5kIHRoZQogICAgZXllIHdhcyByaWdodC4gTWF0Y2hpbmcgc3RhciB3aWR0aHMgYnkgdGhlaXIgbWVhc3VyZWQgRldITSBwcmVzY3JpYmVkIGEgYmx1cgogICAgd2l0aCB0aGUgc2lnbiB3cm9uZy4gQ2hvb3NpbmcgdGhlIGZhY3RvciBhcyB0aGUgbWVhbiBvdmVyIHN0YXJzIGxlZnQgdHdpY2UgYXMKICAgIG1hbnkgdmlzaWJsZSBzdGFyIHJlc2lkdWFscyBhcyBhIGhhbmQtdHVuZWQgdmFsdWUgYSBmZXcgcGVyY2VudCBsb3dlci4gVGhlCiAgICBxdWFudGl0eSB0aGF0IGFjdHVhbGx5IG1hdHRlcnMgLSB3aGV0aGVyIGEgc3RhciBoYXMgdmFuaXNoZWQgaW50byB0aGUgc2t5IG9yCiAgICBzdGlsbCBzaG93cyBhcyBhIHNwb3QgdGhhdCBhIGRldGVjdG9yIHdpbGwgY2FsbCBhbiBISUkgcmVnaW9uIC0gaXMgc29tZXRoaW5nIGEKICAgIHBlcnNvbiByZWFkcyBvZmYgYSBmcmFtZSBpbiBhIHNlY29uZCBhbmQgbm8gc2luZ2xlIG51bWJlciBoZXJlIGhhcyBjYXB0dXJlZC4KCiAgICBXSFkgVEhSRUUgQU5EIE5PVCBBIE5VTUJFUiBPTiBUSEUgU0NSRUVOCiAgICBBIG51bWJlciBjYW5ub3QgYmUganVkZ2VkIHdpdGhvdXQga25vd2luZyB3aGF0IGl0IGxvb2tzIGxpa2UuIFRocmVlIGZyYW1lcwogICAgd3JpdHRlbiB0byBkaXNrLCBkaWZmZXJpbmcgZW5vdWdoIHRvIHRlbGwgYXBhcnQsIGFuZCBvbmUgcXVlc3Rpb24uCgogICAgSE9XIFRIRSBUSFJFRSBBUkUgUElDS0VECiAgICBUaGUgd2hvbGUgZ3JpZCBvZiBmYWN0b3IgYW5kIGJsdXIgaXMgc2NvcmVkIG9uIHN0YXJzIHRoYXQgbGllIG91dHNpZGUgdGhlCiAgICBnYWxheHksIHdoZXJlIG5vdGhpbmcgYnV0IHNreSBhbmQgdGhlIHN0YXIgaXMgaW4gdGhlIGFwZXJ0dXJlLiBUaGUgc2NvcmUgaXMgaG93CiAgICBmbGF0IHdoYXQgaXMgbGVmdCBvZiB0aGUgbWVkaWFuIHN0YXIgaXMgLSBhIGdvb2Qgc3VidHJhY3Rpb24gbGVhdmVzIG5vIGNvcmUgYW5kCiAgICBubyByaW5nLiBUaGUgYmVzdCBpcyBvZmZlcmVkLCB3aXRoIG9uZSBjYW5kaWRhdGUgZWl0aGVyIHNpZGUgb2YgaXQgaW4gZmFjdG9yLAogICAgc28gdGhhdCB0aGUgdGhyZWUgc3BhbiB2aXNpYmx5IGxlc3MgYW5kIHZpc2libHkgbW9yZSBzdWJ0cmFjdGlvbiByYXRoZXIgdGhhbgogICAgdGhyZWUgdmVyc2lvbnMgb2YgdGhlIHNhbWUgYW5zd2VyLgoKICAgIFJldHVybnMgKGhhLCByZWQsIGZhY3RvciwgZGlhZ25vc3RpY3MpLiBXaXRob3V0IGEgcXVlc3Rpb24gdG8gYXNrIC0gYQogICAgbm9uLWludGVyYWN0aXZlIHJ1biAtIHRoZSBtaWRkbGUgb25lIGlzIHRha2VuIGFuZCBzYWlkIHNvLgogICAgIiIiCiAgICBmcm9tIHNjaXB5Lm5kaW1hZ2UgaW1wb3J0IGdhdXNzaWFuX2ZpbHRlcgogICAgZnJvbSBhc3Ryb3B5LnN0YXRzIGltcG9ydCBzaWdtYV9jbGlwcGVkX3N0YXRzCgogICAgZGlhZyA9IHt9CiAgICBpZiBwb3NpdGlvbnMgaXMgTm9uZSBvciBsZW4ocG9zaXRpb25zKSA8IDMwOgogICAgICAgIHJldHVybiBoYSwgcmVkLCBOb25lLCB7IndoeSI6ICJ0b28gZmV3IHN0YXJzIHRvIGp1ZGdlIGEgc3VidHJhY3Rpb24gb24ifQoKICAgIHBvcyA9IG5wLmFzYXJyYXkocG9zaXRpb25zLCBkdHlwZT1mbG9hdCkKICAgIGlmIGNlbnRyZSBpcyBub3QgTm9uZSBhbmQgcjI1X3B4OgogICAgICAgIG91dHNpZGUgPSBucC5oeXBvdChwb3NbOiwgMF0gLSBjZW50cmVbMF0sIHBvc1s6LCAxXSAtIGNlbnRyZVsxXSkgPiAxLjMgKiByMjVfcHgKICAgICAgICBpZiBvdXRzaWRlLnN1bSgpID49IDMwOgogICAgICAgICAgICBwb3MgPSBwb3Nbb3V0c2lkZV0KICAgIGRpYWdbIm5fc3RhcnMiXSA9IGludChsZW4ocG9zKSkKCiAgICAjIFdIRVJFIFRIRSBSQU5HRSBDT01FUyBGUk9NCiAgICAjIEZyb20gdGhpcyBwYWlyIG9mIGZyYW1lcywgbmV2ZXIgZnJvbSBhIG51bWJlciB0eXBlZCBpbi4gVGhlIGZhY3RvciB0aGF0CiAgICAjIGNhbmNlbHMgYSBzdGFyIGlzIG1lYXN1cmVkIG9uIHRoZSBzdGFycyB0aGVtc2VsdmVzIGFuZCB0aGUgZ3JpZCBpcyBsYWlkIG91dAogICAgIyBhcm91bmQgaXQsIHNvIGEgZ2FsYXh5IHdob3NlIGZhY3RvciBpcyB0d2VudHkgaXMgc2VhcmNoZWQgbmVhciB0d2VudHkgYW5kIG9uZQogICAgIyB3aG9zZSBmYWN0b3IgaXMgdHdlbHZlIGlzIHNlYXJjaGVkIG5lYXIgdHdlbHZlLiBBIGdyaWQgd3JpdHRlbiBieSBoYW5kIHdvdWxkCiAgICAjIGhhdmUgYmVlbiB0aGUgcmFuZ2Ugb2Ygd2hpY2hldmVyIGdhbGF4eSBpdCB3YXMgd3JpdHRlbiBmb3IuCiAgICBpZiBmYWN0b3JzIGlzIE5vbmU6CiAgICAgICAgcGFzcyAgIyBmcm9tIGNvbnRpbnV1bV9zdWJ0cmFjdGlvbiwgaW5saW5lZCBhYm92ZQogICAgICAgIHRyeToKICAgICAgICAgICAgY2VudHJlX2ZhY3RvciA9IGZsb2F0KGNvbnN0YW50X2ZhY3RvcihoYSwgcmVkLCBwb3NpdGlvbnM9cG9zKVswXSkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBjZW50cmVfZmFjdG9yID0gTm9uZQogICAgICAgIGlmIG5vdCBjZW50cmVfZmFjdG9yIG9yIG5vdCBucC5pc2Zpbml0ZShjZW50cmVfZmFjdG9yKSBvciBjZW50cmVfZmFjdG9yIDw9IDA6CiAgICAgICAgICAgIHJldHVybiBoYSwgcmVkLCBOb25lLCB7IndoeSI6ICJjb3VsZCBub3QgbWVhc3VyZSBhIGZhY3RvciB0byBzZWFyY2ggYXJvdW5kIn0KICAgICAgICBkaWFnWyJtZWFzdXJlZF9mYWN0b3IiXSA9IGNlbnRyZV9mYWN0b3IKCiAgICBoID0gOAogICAgeXksIHh4ID0gbnAubWdyaWRbLWg6aCArIDEsIC1oOmggKyAxXQogICAgcnIgPSBucC5oeXBvdCh4eCwgeXkpCiAgICBlZGdlID0gcnIgPiBoIC0gMS41CiAgICBueSwgbnggPSBoYS5zaGFwZQogICAgc2VsID0gWyhpbnQocm91bmQoeCkpLCBpbnQocm91bmQoeSkpKSBmb3IgeCwgeSBpbiBwb3MKICAgICAgICAgICBpZiBoIDwgaW50KHJvdW5kKHgpKSA8IG54IC0gaCBhbmQgaCA8IGludChyb3VuZCh5KSkgPCBueSAtIGhdCiAgICBpZiBsZW4oc2VsKSA8IDMwOgogICAgICAgIHJldHVybiBoYSwgcmVkLCBOb25lLCB7IndoeSI6ICJ0aGUgc3RhcnMgYWxsIGZhbGwgdG9vIG5lYXIgdGhlIGZyYW1lIGVkZ2UifQoKICAgIGRlZiBzdGFtcHMoaW1nKToKICAgICAgICBvdXQgPSBucC5lbXB0eSgobGVuKHNlbCksIDIgKiBoICsgMSwgMiAqIGggKyAxKSwgZHR5cGU9bnAuZmxvYXQzMikKICAgICAgICBmb3IgaSwgKHgsIHkpIGluIGVudW1lcmF0ZShzZWwpOgogICAgICAgICAgICBzID0gbnAubmFuX3RvX251bShpbWdbeSAtIGg6eSArIGggKyAxLCB4IC0gaDp4ICsgaCArIDFdKQogICAgICAgICAgICBvdXRbaV0gPSBzIC0gbnAubWVkaWFuKHNbZWRnZV0pCiAgICAgICAgcmV0dXJuIG91dAoKICAgICMgVEhFUkUgSVMgTk8gUkFOR0UKICAgICMgQW4gdXBwZXIgYW5kIGxvd2VyIGJvdW5kIG9uIHRoZSBzZWFyY2ggaXMgYSBndWVzcyBhYm91dCB0aGUgYW5zd2VyLCBhbmQgdGhlCiAgICAjIGFuc3dlciBpcyB3aGF0IGlzIGJlaW5nIGxvb2tlZCBmb3IuIFNvIHRoZSBzZWFyY2ggc3RhcnRzIHNtYWxsIGFuZCBncm93czogYXMKICAgICMgbG9uZyBhcyB0aGUgYmVzdCBwb2ludCBpdCBoYXMgZm91bmQgc2l0cyBvbiBhbiBlZGdlIG9mIHdoYXQgaGFzIGJlZW4gdHJpZWQsCiAgICAjIHRoYXQgZWRnZSBtb3ZlcyBvdXQgYW5kIGl0IGxvb2tzIGFnYWluLiBJdCBzdG9wcyB3aGVuIHRoZSBiZXN0IHBvaW50IGhhcwogICAgIyBuZWlnaGJvdXJzIG9uIGFsbCBzaWRlcywgd2hpY2ggaXMgdGhlIG9ubHkgZXZpZGVuY2UgdGhhdCBpdCBpcyBhIG1pbmltdW0KICAgICMgcmF0aGVyIHRoYW4gdGhlIGVuZCBvZiB0aGUgcGFwZXIuCiAgICBmX3N0ZXAgPSAwLjAyNSAqIGNlbnRyZV9mYWN0b3IgaWYgZmFjdG9ycyBpcyBOb25lIGVsc2UgTm9uZQogICAgYl9zdGVwID0gMC4yNQogICAgaWYgZmFjdG9ycyBpcyBOb25lOgogICAgICAgIGZfbG8sIGZfaGkgPSAtMiwgMiAgICAgICAgICAgICAgICAgICMgaW4gc3RlcHMsIHJlbGF0aXZlIHRvIHRoZSBtZWFzdXJlZCBmYWN0b3IKICAgIGVsc2U6CiAgICAgICAgZl9sbyA9IGZfaGkgPSBOb25lCiAgICBiX2xvLCBiX2hpID0gLTIsIDIKCiAgICBjYWNoZSA9IHt9CiAgICBzdGFtcF9jYWNoZSA9IHt9CgogICAgZGVmIHNjb3JlKGZpLCBiaSk6CiAgICAgICAga2V5ID0gKGZpLCBiaSkKICAgICAgICBpZiBrZXkgaW4gY2FjaGU6CiAgICAgICAgICAgIHJldHVybiBjYWNoZVtrZXldCiAgICAgICAgYmx1ciA9IGJpICogYl9zdGVwCiAgICAgICAgaWYgYmkgbm90IGluIHN0YW1wX2NhY2hlOgogICAgICAgICAgICBBID0gZ2F1c3NpYW5fZmlsdGVyKGhhLCBibHVyKSBpZiBibHVyID4gMCBlbHNlIGhhCiAgICAgICAgICAgIFIgPSBnYXVzc2lhbl9maWx0ZXIocmVkLCAtYmx1cikgaWYgYmx1ciA8IDAgZWxzZSByZWQKICAgICAgICAgICAgc3RhbXBfY2FjaGVbYmldID0gKHN0YW1wcyhBKSwgc3RhbXBzKFIpKQogICAgICAgIFNBLCBTUiA9IHN0YW1wX2NhY2hlW2JpXQogICAgICAgIGYgPSAoY2VudHJlX2ZhY3RvciArIGZpICogZl9zdGVwKSBpZiBmYWN0b3JzIGlzIE5vbmUgZWxzZSBmbG9hdChmYWN0b3JzW2ZpXSkKICAgICAgICBNID0gbnAubWVkaWFuKGYgKiBTQSAtIFNSLCBheGlzPTApCiAgICAgICAgcHJvZiA9IFtNWyhyciA+PSBxKSAmIChyciA8IHEgKyAxKV0ubWVhbigpIGZvciBxIGluIHJhbmdlKDYpXQogICAgICAgIGNhY2hlW2tleV0gPSAoZmxvYXQobnAuYWJzKHByb2YpLm1heCgpKSwgZmxvYXQoZiksIGZsb2F0KGJsdXIpKQogICAgICAgIHJldHVybiBjYWNoZVtrZXldCgogICAgaWYgZmFjdG9ycyBpcyBub3QgTm9uZToKICAgICAgICBmX2xvLCBmX2hpID0gMCwgbGVuKGZhY3RvcnMpIC0gMQogICAgaWYgYmx1cnMgaXMgbm90IE5vbmU6CiAgICAgICAgYl9zdGVwID0gMS4wCiAgICAgICAgYl9sbywgYl9oaSA9IDAsIGxlbihibHVycykgLSAxCgogICAgZm9yIF9yb3VuZCBpbiByYW5nZSg0MCk6CiAgICAgICAgZ3JpZCA9IFtzY29yZShmaSwgYmkpCiAgICAgICAgICAgICAgICBmb3IgZmkgaW4gcmFuZ2UoZl9sbywgZl9oaSArIDEpCiAgICAgICAgICAgICAgICBmb3IgYmkgaW4gcmFuZ2UoYl9sbywgYl9oaSArIDEpXQogICAgICAgIGJlc3RfbGVmdCwgYmVzdF9mLCBiZXN0X2JsdXIgPSBtaW4oZ3JpZCkKICAgICAgICBiaV9iZXN0ID0gaW50KHJvdW5kKGJlc3RfYmx1ciAvIGJfc3RlcCkpIGlmIGJfc3RlcCBlbHNlIDAKICAgICAgICBmaV9iZXN0ID0gKGludChyb3VuZCgoYmVzdF9mIC0gY2VudHJlX2ZhY3RvcikgLyBmX3N0ZXApKQogICAgICAgICAgICAgICAgICAgaWYgZmFjdG9ycyBpcyBOb25lIGVsc2UgTm9uZSkKICAgICAgICBncmV3ID0gRmFsc2UKICAgICAgICBpZiBmYWN0b3JzIGlzIE5vbmU6CiAgICAgICAgICAgIGlmIGZpX2Jlc3QgPD0gZl9sbyBhbmQgY2VudHJlX2ZhY3RvciArIChmX2xvIC0gMSkgKiBmX3N0ZXAgPiAwOgogICAgICAgICAgICAgICAgZl9sbyAtPSAxOyBncmV3ID0gVHJ1ZQogICAgICAgICAgICBpZiBmaV9iZXN0ID49IGZfaGk6CiAgICAgICAgICAgICAgICBmX2hpICs9IDE7IGdyZXcgPSBUcnVlCiAgICAgICAgaWYgYmlfYmVzdCA8PSBiX2xvOgogICAgICAgICAgICBiX2xvIC09IDE7IGdyZXcgPSBUcnVlCiAgICAgICAgaWYgYmlfYmVzdCA+PSBiX2hpOgogICAgICAgICAgICBiX2hpICs9IDE7IGdyZXcgPSBUcnVlCiAgICAgICAgaWYgbm90IGdyZXc6CiAgICAgICAgICAgIGJyZWFrCgogICAgc2NvcmVkID0gc29ydGVkKGNhY2hlLnZhbHVlcygpKQogICAgYmVzdF9sZWZ0LCBiZXN0X2YsIGJlc3RfYmx1ciA9IHNjb3JlZFswXQogICAgZGlhZ1siYmVzdCJdID0gKGJlc3RfZiwgYmVzdF9ibHVyLCBiZXN0X2xlZnQpCiAgICBkaWFnWyJzZWFyY2hlZCJdID0gewogICAgICAgICJmYWN0b3IiOiAobWluKHNbMV0gZm9yIHMgaW4gc2NvcmVkKSwgbWF4KHNbMV0gZm9yIHMgaW4gc2NvcmVkKSksCiAgICAgICAgImJsdXIiOiAobWluKHNbMl0gZm9yIHMgaW4gc2NvcmVkKSwgbWF4KHNbMl0gZm9yIHMgaW4gc2NvcmVkKSksCiAgICAgICAgInBvaW50cyI6IGxlbihzY29yZWQpfQogICAgcHJpbnQoIiAgICBzZWFyY2hlZCBmYWN0b3IgezouMmZ9IHRvIHs6LjJmfSBhbmQgYmx1ciB7OisuMmZ9IHRvIHs6Ky4yZn0gcHggIgogICAgICAgICAgIm92ZXIge30gY29tYmluYXRpb25zLCB1bnRpbCB0aGUgYmVzdCBoYWQgbmVpZ2hib3VycyBvbiBldmVyeSBzaWRlIgogICAgICAgICAgLmZvcm1hdChkaWFnWyJzZWFyY2hlZCJdWyJmYWN0b3IiXVswXSwgZGlhZ1sic2VhcmNoZWQiXVsiZmFjdG9yIl1bMV0sCiAgICAgICAgICAgICAgICAgIGRpYWdbInNlYXJjaGVkIl1bImJsdXIiXVswXSwgZGlhZ1sic2VhcmNoZWQiXVsiYmx1ciJdWzFdLAogICAgICAgICAgICAgICAgICBsZW4oc2NvcmVkKSkpCgogICAgIyBPbmUgY2FuZGlkYXRlIGVpdGhlciBzaWRlIGluIGZhY3RvciwgZWFjaCB3aXRoIGl0cyBvd24gYmVzdCBibHVyLCBzbyB0aGUKICAgICMgdGhyZWUgZGlmZmVyIGJ5IHNvbWV0aGluZyB0aGUgZXllIGNhbiBzZWUuCiAgICBkZWYgYmVzdF9hdChmKToKICAgICAgICBoZXJlID0gW3MgZm9yIHMgaW4gc2NvcmVkIGlmIGFicyhzWzFdIC0gZikgPCAxZS02XQogICAgICAgIHJldHVybiBtaW4oaGVyZSkgaWYgaGVyZSBlbHNlIE5vbmUKCiAgICBvdGhlcnMgPSBzb3J0ZWQoe2YgZm9yIF8sIGYsIF8gaW4gc2NvcmVkfSkKICAgIGJlbG93ID0gW2YgZm9yIGYgaW4gb3RoZXJzIGlmIGYgPCBiZXN0X2ZdCiAgICBhYm92ZSA9IFtmIGZvciBmIGluIG90aGVycyBpZiBmID4gYmVzdF9mXQogICAgcGlja3MgPSBbYmVzdF9hdChiZWxvd1stMV0pIGlmIGJlbG93IGVsc2UgTm9uZSwKICAgICAgICAgICAgIChiZXN0X2xlZnQsIGJlc3RfZiwgYmVzdF9ibHVyKSwKICAgICAgICAgICAgIGJlc3RfYXQoYWJvdmVbMF0pIGlmIGFib3ZlIGVsc2UgTm9uZV0KICAgIHBpY2tzID0gW3AgZm9yIHAgaW4gcGlja3MgaWYgcCBpcyBub3QgTm9uZV0KICAgIGlmIGxlbihwaWNrcykgPCAzOiAgICAgICAgICAgICAgICAgICAgICAgIyB0aGUgYmVzdCBzYXQgYXQgYW4gZW5kIG9mIHRoZSByYW5nZQogICAgICAgIGV4dHJhX3Bvb2wgPSBbcyBmb3IgcyBpbiBzY29yZWQgaWYgc1sxXSBub3QgaW4ge3BbMV0gZm9yIHAgaW4gcGlja3N9XQogICAgICAgIHBpY2tzICs9IHNvcnRlZChleHRyYV9wb29sKVs6MyAtIGxlbihwaWNrcyldCiAgICBwaWNrcy5zb3J0KGtleT1sYW1iZGEgcDogcFsxXSkKCiAgICByZXR1cm4gX3dyaXRlX2Nob2ljZXMoaGEsIHJlZCwgcGlja3MsIHByZWZpeCwgZGlhZywgYXNrKQoKZGVmIHNob3dfcGljdHVyZXMocGF0aHMsIHRpdGxlPU5vbmUpOgogICAgIiIiUHV0IHBpY3R1cmVzIGFscmVhZHkgd3JpdHRlbiB0byBkaXNrIG9uIHRoZSBzY3JlZW4sIHdpdGhvdXQgYmxvY2tpbmcuCgogICAgV0hZIElUIFNIT1dTIFRIRSBTQVZFRCBGSUxFIEFORCBOT1QgVEhFIEZJR1VSRQogICAgQmVjYXVzZSB0aGUgZmlndXJlIHdhcyBkcmF3biB1bmRlciBBZ2csIHRoZSBiYWNrZW5kIHRoYXQgd3JpdGVzIHRvIGEgZmlsZSBhbmQKICAgIGhhcyBubyB3aW5kb3cgdG8gYXR0YWNoIHRvLCBhbmQgbW92aW5nIGEgbGl2ZSBmaWd1cmUgYmV0d2VlbiBiYWNrZW5kcyBpcyBub3QKICAgIHNvbWV0aGluZyBtYXRwbG90bGliIHN1cHBvcnRzLiBSZWFkaW5nIGJhY2sgdGhlIFBORyB0aGF0IHdhcyBqdXN0IHdyaXR0ZW4KICAgIHNpZGVzdGVwcyB0aGF0IGVudGlyZWx5LCBhbmQgaGFzIHRoZSBhZHZhbnRhZ2UgdGhhdCB3aGF0IGFwcGVhcnMgb24gdGhlIHNjcmVlbgogICAgaXMgZXhhY3RseSB0aGUgZmlsZSAtIGlmIHRoZXkgZXZlciBkaXNhZ3JlZWQgaXQgd291bGQgYmUgdGhlIGZpbGUgdGhhdCBpcyB3cm9uZywKICAgIGFuZCB0aGUgZmlsZSBpcyB3aGF0IGdldHMga2VwdC4KCiAgICBBbiBpbnRlcmFjdGl2ZSBiYWNrZW5kIGlzIGJvcnJvd2VkIGFuZCBnaXZlbiBiYWNrLiBPbiBhIG1hY2hpbmUgd2l0aCBubyBkaXNwbGF5CiAgICBub3RoaW5nIGlzIHNob3duIGFuZCBub3RoaW5nIGZhaWxzOyB0aGUgZmlsZXMgYXJlIHN0aWxsIHRoZXJlLgogICAgIiIiCiAgICBpbXBvcnQgbWF0cGxvdGxpYgogICAgaW1wb3J0IG1hdHBsb3RsaWIucHlwbG90IGFzIHBsdAoKICAgIHBhdGhzID0gW3AgZm9yIHAgaW4gcGF0aHMgaWYgcCBhbmQgb3MucGF0aC5leGlzdHMocCldCiAgICBpZiBub3QgcGF0aHM6CiAgICAgICAgcmV0dXJuIEZhbHNlCgogICAgc3RhcnRlZCA9IG1hdHBsb3RsaWIuZ2V0X2JhY2tlbmQoKQogICAgaWYgc3RhcnRlZC5sb3dlcigpID09ICJhZ2ciOgogICAgICAgIGZvciBjYW5kaWRhdGUgaW4gKCJUa0FnZyIsICJRdEFnZyIsICJRdDVBZ2ciLCAiTWFjT1NYIik6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHBsdC5zd2l0Y2hfYmFja2VuZChjYW5kaWRhdGUpCiAgICAgICAgICAgICAgICBicmVhawogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICAgICAgY29udGludWUKICAgIGlmIG1hdHBsb3RsaWIuZ2V0X2JhY2tlbmQoKS5sb3dlcigpID09ICJhZ2ciOgogICAgICAgIHJldHVybiBGYWxzZQoKICAgIHRyeToKICAgICAgICBpbXBvcnQgbWF0cGxvdGxpYi5pbWFnZSBhcyBtcGltZwogICAgICAgIGZvciBwYXRoIGluIHBhdGhzOgogICAgICAgICAgICBpbWFnZSA9IG1waW1nLmltcmVhZChwYXRoKQogICAgICAgICAgICBoZWlnaHQsIHdpZHRoID0gaW1hZ2Uuc2hhcGVbMF0sIGltYWdlLnNoYXBlWzFdCiAgICAgICAgICAgIHNpZGUgPSA5LjUKICAgICAgICAgICAgZmlnID0gcGx0LmZpZ3VyZShmaWdzaXplPShzaWRlLCBzaWRlICogaGVpZ2h0IC8gZmxvYXQod2lkdGgpKSkKICAgICAgICAgICAgYXggPSBmaWcuYWRkX2F4ZXMoWzAsIDAsIDEsIDFdKQogICAgICAgICAgICBheC5pbXNob3coaW1hZ2UpCiAgICAgICAgICAgIGF4LnNldF9heGlzX29mZigpCiAgICAgICAgICAgIGZpZy5jYW52YXMubWFuYWdlci5zZXRfd2luZG93X3RpdGxlKAogICAgICAgICAgICAgICAgdGl0bGUgb3Igb3MucGF0aC5iYXNlbmFtZShwYXRoKSkKICAgICAgICBwbHQuc2hvdyhibG9jaz1GYWxzZSkKICAgICAgICBwbHQucGF1c2UoMC41KQogICAgICAgIHJldHVybiBUcnVlCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHJldHVybiBGYWxzZQogICAgZmluYWxseToKICAgICAgICB0cnk6CiAgICAgICAgICAgIHBsdC5zd2l0Y2hfYmFja2VuZChzdGFydGVkKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHBhc3MKCmRlZiBfc2hvd19jaG9pY2VzX3dpbmRvdyhmcmFtZXMsIG56LCBzdGVwKToKICAgICIiIlB1dCB0aGUgdGhyZWUgY2FuZGlkYXRlcyBzaWRlIGJ5IHNpZGUgaW4gb25lIHdpbmRvdyBhbmQgbGVhdmUgaXQgb3Blbi4KCiAgICBXSFkgQSBXSU5ET1cgQU5EIE5PVCBUSFJFRSBGSUxFUwogICAgQmVjYXVzZSB0aGUganVkZ2VtZW50IGlzIGEgY29tcGFyaXNvbi4gVGhyZWUgZmlsZXMgb3BlbmVkIG9uZSBhZnRlciBhbm90aGVyIGluCiAgICBhIHBpY3R1cmUgdmlld2VyIGFyZSB0aHJlZSBzZXBhcmF0ZSBpbXByZXNzaW9ucyBhIGZldyBzZWNvbmRzIGFwYXJ0OyBzaWRlIGJ5CiAgICBzaWRlIHRoZXkgYXJlIG9uZSBsb29rLCBhbmQgdGhlIGRpZmZlcmVuY2UgYmV0d2VlbiB0aGVtIC0gd2hpY2ggaXMgc21hbGwsIGEgZmV3CiAgICBwZXJjZW50IGluIHRoZSBmYWN0b3IgLSBpcyBvbmx5IHZpc2libGUgdGhhdCB3YXkuCgogICAgVGhlIGJhY2tlbmQgaXMgdGhlIGRpZmZpY3VsdHkuIEJvdGggdG9vbHMgc2V0IEFnZyBhdCBpbXBvcnQsIHdoaWNoIGRyYXdzIHRvIGEKICAgIGZpbGUgYW5kIG5ldmVyIHRvIGEgc2NyZWVuLCBhbmQgdGhhdCBpcyB0aGUgcmlnaHQgZGVmYXVsdCBmb3IgYSBydW4gdGhhdAogICAgcHJvZHVjZXMgZmlndXJlcyB1bmF0dGVuZGVkLiBTbyBhbiBpbnRlcmFjdGl2ZSBiYWNrZW5kIGlzIGJvcnJvd2VkIGZvciB0aGlzIG9uZQogICAgd2luZG93IGFuZCBnaXZlbiBiYWNrIGFmdGVyd2FyZHMsIGFuZCBpZiBub25lIG9mIHRoZW0gY2FuIGJlIGhhZCAtIGEgbWFjaGluZQogICAgd2l0aCBubyBkaXNwbGF5LCBhIHNlc3Npb24gb3ZlciBzc2ggLSBub3RoaW5nIGlzIHNob3duIGFuZCB0aGUgZmlsZXMgd3JpdHRlbgogICAgbmV4dCB0byB0aGUgb3V0cHV0IGFyZSB0aGUgZmFsbGJhY2suCgogICAgVGhlIHdpbmRvdyBpcyBvcGVuZWQgd2l0aG91dCBibG9ja2luZywgc28gdGhlIHF1ZXN0aW9uIGNhbiBiZSBhbnN3ZXJlZCBpbiB0aGUKICAgIHRlcm1pbmFsIHdoaWxlIGl0IGlzIHN0aWxsIG9uIHNjcmVlbi4KCiAgICBSZXR1cm5zIFRydWUgaWYgYSB3aW5kb3cgYWN0dWFsbHkgYXBwZWFyZWQuCiAgICAiIiIKICAgIGltcG9ydCBtYXRwbG90bGliCiAgICBpbXBvcnQgbWF0cGxvdGxpYi5weXBsb3QgYXMgcGx0CgogICAgZGVmIGJsb2NrX21lYW4oYSwgbik6CiAgICAgICAgazAgPSAoYS5zaGFwZVswXSAvLyBuKSAqIG4KICAgICAgICBrMSA9IChhLnNoYXBlWzFdIC8vIG4pICogbgogICAgICAgIHJldHVybiBhWzprMCwgOmsxXS5yZXNoYXBlKGswIC8vIG4sIG4sIGsxIC8vIG4sIG4pLm1lYW4oYXhpcz0oMSwgMykpCgogICAgc3RhcnRlZCA9IG1hdHBsb3RsaWIuZ2V0X2JhY2tlbmQoKQogICAgaWYgc3RhcnRlZC5sb3dlcigpID09ICJhZ2ciOgogICAgICAgIGZvciBjYW5kaWRhdGUgaW4gKCJUa0FnZyIsICJRdEFnZyIsICJRdDVBZ2ciLCAiTWFjT1NYIik6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHBsdC5zd2l0Y2hfYmFja2VuZChjYW5kaWRhdGUpCiAgICAgICAgICAgICAgICBicmVhawogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICAgICAgY29udGludWUKICAgIGlmIG1hdHBsb3RsaWIuZ2V0X2JhY2tlbmQoKS5sb3dlcigpID09ICJhZ2ciOgogICAgICAgIHJldHVybiBGYWxzZSAgICAgICAgICAgICAgICAgICAgICMgbm90aGluZyBvbiB0aGlzIG1hY2hpbmUgY2FuIHNob3cgYSB3aW5kb3cKCiAgICB0cnk6CiAgICAgICAgZmlnLCBheGVzID0gcGx0LnN1YnBsb3RzKDEsIGxlbihmcmFtZXMpLCBmaWdzaXplPSg1LjIgKiBsZW4oZnJhbWVzKSwgNS44KSkKICAgICAgICBpZiBsZW4oZnJhbWVzKSA9PSAxOgogICAgICAgICAgICBheGVzID0gW2F4ZXNdCiAgICAgICAgZm9yIGF4LCAoaSwgKGYsIGJsdXIsIG5ldCkpIGluIHppcChheGVzLCBlbnVtZXJhdGUoZnJhbWVzLCAxKSk6CiAgICAgICAgICAgIHNtYWxsID0gYmxvY2tfbWVhbihuZXQsIHN0ZXApIGlmIHN0ZXAgPiAxIGVsc2UgbmV0CiAgICAgICAgICAgIGF4Lmltc2hvdyhzbWFsbCwgb3JpZ2luPSJsb3dlciIsIGNtYXA9ImdyYXlfciIsCiAgICAgICAgICAgICAgICAgICAgICB2bWluPS0yICogbnosIHZtYXg9OCAqIG56KQogICAgICAgICAgICBheC5zZXRfeHRpY2tzKFtdKTsgYXguc2V0X3l0aWNrcyhbXSkKICAgICAgICAgICAgYXguc2V0X3RpdGxlKCJjaG9pY2Uge31cbmZhY3RvciB7Oi4yZn0gICB7fSIuZm9ybWF0KAogICAgICAgICAgICAgICAgaSwgZiwgX2JsdXJfd29yZHMoYmx1cikpLCBmb250c2l6ZT0xMikKICAgICAgICBmaWcuc3VwdGl0bGUoIldoaWNoIHN1YnRyYWN0aW9uIGlzIGJlc3Q/IExvb2sgYXQgdGhlIHN0YXJzIGFuZCBhdCB0aGUgIgogICAgICAgICAgICAgICAgICAgICAiZW1wdHkgc2t5LCB0aGVuIGFuc3dlciBpbiB0aGUgdGVybWluYWwuIiwgZm9udHNpemU9MTMpCiAgICAgICAgZmlnLnRpZ2h0X2xheW91dCgpCiAgICAgICAgcGx0LnNob3coYmxvY2s9RmFsc2UpCiAgICAgICAgcGx0LnBhdXNlKDAuNSkKICAgICAgICByZXR1cm4gVHJ1ZQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICByZXR1cm4gRmFsc2UKICAgIGZpbmFsbHk6CiAgICAgICAgdHJ5OgogICAgICAgICAgICBwbHQuc3dpdGNoX2JhY2tlbmQoc3RhcnRlZCkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBwYXNzCgpkZWYgX2JsdXJfd29yZHMoYmx1cik6CiAgICAiIiJUaGUgYmx1ciBhcyBhIHBocmFzZSwgc28gdGhlIHBpY3R1cmUgc2F5cyB3aGljaCBmcmFtZSB3YXMgYmx1cnJlZC4iIiIKICAgIGlmIGJsdXIgPiAwOgogICAgICAgIHJldHVybiAiSC1hbHBoYSBibHVycmVkIGJ5IHs6LjJmfSBweCIuZm9ybWF0KGJsdXIpCiAgICBpZiBibHVyIDwgMDoKICAgICAgICByZXR1cm4gIlIgYmx1cnJlZCBieSB7Oi4yZn0gcHgiLmZvcm1hdCgtYmx1cikKICAgIHJldHVybiAibmVpdGhlciBmcmFtZSBibHVycmVkIgoKZGVmIF93cml0ZV9jaG9pY2VzKGhhLCByZWQsIHBpY2tzLCBwcmVmaXgsIGRpYWcsIGFzayk6CiAgICAiIiJXcml0ZSBvbmUgcGljdHVyZSBwZXIgY2FuZGlkYXRlLCBhc2ssIGFuZCBhcHBseSB0aGUgYW5zd2VyLiIiIgogICAgaW1wb3J0IG1hdHBsb3RsaWIKICAgIG1hdHBsb3RsaWIudXNlKCJBZ2ciKQogICAgaW1wb3J0IG1hdHBsb3RsaWIucHlwbG90IGFzIHBsdAogICAgZnJvbSBzY2lweS5uZGltYWdlIGltcG9ydCBnYXVzc2lhbl9maWx0ZXIKICAgIGZyb20gYXN0cm9weS5zdGF0cyBpbXBvcnQgc2lnbWFfY2xpcHBlZF9zdGF0cwoKICAgIGRlZiBibG9ja19tZWFuKGEsIG4pOgogICAgICAgIGswID0gKGEuc2hhcGVbMF0gLy8gbikgKiBuCiAgICAgICAgazEgPSAoYS5zaGFwZVsxXSAvLyBuKSAqIG4KICAgICAgICByZXR1cm4gYVs6azAsIDprMV0ucmVzaGFwZShrMCAvLyBuLCBuLCBrMSAvLyBuLCBuKS5tZWFuKGF4aXM9KDEsIDMpKQoKICAgIHN0ZXAgPSBtYXgoMSwgaW50KHJvdW5kKG1heChoYS5zaGFwZSkgLyAxNDAwLjApKSkKICAgIGZyYW1lcywgcGF0aHMgPSBbXSwgW10KICAgIGZvciBpLCAobGVmdCwgZiwgYmx1cikgaW4gZW51bWVyYXRlKHBpY2tzLCAxKToKICAgICAgICBBID0gZ2F1c3NpYW5fZmlsdGVyKGhhLCBibHVyKSBpZiBibHVyID4gMCBlbHNlIGhhCiAgICAgICAgUiA9IGdhdXNzaWFuX2ZpbHRlcihyZWQsIC1ibHVyKSBpZiBibHVyIDwgMCBlbHNlIHJlZAogICAgICAgIGZyYW1lcy5hcHBlbmQoKGYsIGJsdXIsIGYgKiBBIC0gUikpCgogICAgIyBPbmUgc3RyZXRjaCBmb3IgYWxsIHRocmVlLCBvciB0aGUgZXllIGNvbXBhcmVzIHRocmVlIGRpZmZlcmVudCBzdHJldGNoZXMuCiAgICBtaWRkbGUgPSBmcmFtZXNbbGVuKGZyYW1lcykgLy8gMl1bMl0KICAgIG56ID0gZmxvYXQoc2lnbWFfY2xpcHBlZF9zdGF0cyhtaWRkbGUsIHNpZ21hPTMuMClbMl0pCgogICAgZm9yIGksIChmLCBibHVyLCBuZXQpIGluIGVudW1lcmF0ZShmcmFtZXMsIDEpOgogICAgICAgIHNtYWxsID0gYmxvY2tfbWVhbihuZXQsIHN0ZXApIGlmIHN0ZXAgPiAxIGVsc2UgbmV0CiAgICAgICAgZmlnLCBheCA9IHBsdC5zdWJwbG90cyhmaWdzaXplPSg5LCA5KSkKICAgICAgICAjIERhcmsgb24gbGlnaHQsIHRoZSB3YXkgdGhpcyBpcyBsb29rZWQgYXQgYnkgZXllLgogICAgICAgIGF4Lmltc2hvdyhzbWFsbCwgb3JpZ2luPSJsb3dlciIsIGNtYXA9ImdyYXlfciIsIHZtaW49LTIgKiBueiwgdm1heD04ICogbnopCiAgICAgICAgYXguc2V0X3h0aWNrcyhbXSk7IGF4LnNldF95dGlja3MoW10pCiAgICAgICAgIyBUaGUgbnVtYmVycyBiZWxvbmcgb24gdGhlIHBpY3R1cmUuIEEgY2hvaWNlIG1hZGUgZnJvbSB0aHJlZSB1bmxhYmVsbGVkCiAgICAgICAgIyBmcmFtZXMgY2Fubm90IGJlIHdyaXR0ZW4gZG93biwgcmVwZWF0ZWQsIG9yIGFyZ3VlZCB3aXRoIGFmdGVyd2FyZHMuCiAgICAgICAgYXguc2V0X3RpdGxlKCJjaG9pY2Uge31cbmZhY3RvciB7Oi4yZn0gICB7fSIuZm9ybWF0KGksIGYsIF9ibHVyX3dvcmRzKGJsdXIpKSwKICAgICAgICAgICAgICAgICAgICAgZm9udHNpemU9MTQpCiAgICAgICAgZmlnLnRpZ2h0X2xheW91dCgpCiAgICAgICAgcGF0aCA9ICJ7fV9jaG9pY2Vfe30ucG5nIi5mb3JtYXQocHJlZml4LCBpKQogICAgICAgIGZpZy5zYXZlZmlnKHBhdGgsIGRwaT0xMDAsIGJib3hfaW5jaGVzPSJ0aWdodCIpCiAgICAgICAgcGx0LmNsb3NlKGZpZykKICAgICAgICBwYXRocy5hcHBlbmQocGF0aCkKICAgIGRpYWdbInBhdGhzIl0gPSBwYXRocwogICAgZGlhZ1sicGlja3MiXSA9IFsoZiwgYmx1cikgZm9yIGYsIGJsdXIsIF8gaW4gZnJhbWVzXQoKICAgICMgV0hZIFRIRVJFIElTIE5PIFdJTkRPVwogICAgIyBUaGVyZSB3YXMgb25lLCBhbmQgaXQgZGlkIG5vdCB3b3JrLiBBIGZpZ3VyZSBvcGVuZWQgd2l0aG91dCBibG9ja2luZywgc28KICAgICMgdGhhdCB0aGUgcXVlc3Rpb24gY291bGQgc3RpbGwgYmUgYW5zd2VyZWQgaW4gdGhlIHRlcm1pbmFsLCBpcyBhIHdpbmRvdyB3aXRoCiAgICAjIG5vdGhpbmcgZHJpdmluZyBpdHMgZXZlbnQgbG9vcDogaXQgYXBwZWFycyBhcyBhbiBlbXB0eSBncmV5IHJlY3RhbmdsZSBhbmQKICAgICMgb25seSBkcmF3cyBpdHNlbGYgb25jZSBpdCBpcyBjbG9zZWQsIHdoaWNoIGlzIGFmdGVyIHRoZSBhbnN3ZXIgaGFzIGJlZW4KICAgICMgZ2l2ZW4uIEEgcGljdHVyZSB0aGF0IGNhbiBvbmx5IGJlIHNlZW4gYWZ0ZXIgaXQgaXMgbm8gbG9uZ2VyIG5lZWRlZCBpcyB3b3JzZQogICAgIyB0aGFuIG5vIHBpY3R1cmUsIGJlY2F1c2UgaXQgYWxzbyBmcmVlemVzIHRoZSBzY3JlZW4uCiAgICAjCiAgICAjIFNvIHRoZSB0aHJlZSBhcmUgd3JpdHRlbiB0byBkaXNrIGFuZCB0aGVpciBuYW1lcyBhcmUgcHJpbnRlZC4gT3BlbiB0aGVtIGluCiAgICAjIHdoYXRldmVyIHlvdSBub3JtYWxseSBsb29rIGF0IGltYWdlcyB3aXRoLCBvciBkbyBub3QgLSB0aGUgbnVtYmVycyBhcmUgb24KICAgICMgdGhlIHNjcmVlbiBlaXRoZXIgd2F5LCBhbmQgdGhlIG1pZGRsZSBvbmUgaXMgdGhlIGRlZmF1bHQuCiAgICBzaG93biA9IEZhbHNlCgogICAgcHJpbnQoIiIpCiAgICBwcmludCgiICBUaHJlZSBzdWJ0cmFjdGlvbnMgb2YgdGhlIHNhbWUgZnJhbWVzLCB3cml0dGVuIG5leHQgdG8gdGhlIG90aGVyIikKICAgIHByaW50KCIgIG91dHB1dC4gTG9vayBhdCB0aGUgc3RhcnMgYW5kIGF0IHRoZSBlbXB0eSBza3kgaW4gdGhlbSwgdGhlbiBzYXkiKQogICAgcHJpbnQoIiAgd2hpY2ggb25lIGlzIGJlc3QuIEVudGVyIHRha2VzIHRoZSBtaWRkbGUgb25lLiIpCiAgICBwcmludCgiICAgIGluICIgKyAob3MucGF0aC5kaXJuYW1lKHBhdGhzWzBdKSBvciAiLiIpKQogICAgZm9yIGksIChwLCAoZiwgYmx1ciwgXykpIGluIGVudW1lcmF0ZSh6aXAocGF0aHMsIGZyYW1lcyksIDEpOgogICAgICAgIHByaW50KCIgICAge30uIHs6PDI0fSBmYWN0b3IgezouMmZ9LCB7fSIuZm9ybWF0KAogICAgICAgICAgICBpLCBvcy5wYXRoLmJhc2VuYW1lKHApLCBmLCBfYmx1cl93b3JkcyhibHVyKSkpCgogICAgYW5zd2VyID0gMiBpZiBsZW4ocGF0aHMpID49IDIgZWxzZSAxCiAgICBpZiBhc2sgaXMgbm90IE5vbmU6CiAgICAgICAgd2hpbGUgVHJ1ZToKICAgICAgICAgICAgZ290ID0gYXNrKCJXaGljaCBvbmUiLCBzdHIoYW5zd2VyKSwgaW50KQogICAgICAgICAgICBpZiAxIDw9IGdvdCA8PSBsZW4ocGF0aHMpOgogICAgICAgICAgICAgICAgYW5zd2VyID0gZ290CiAgICAgICAgICAgICAgICBicmVhawogICAgICAgICAgICBwcmludCgiICAgIGEgbnVtYmVyIGZyb20gdGhlIGxpc3QsIHBsZWFzZSIpCiAgICBlbHNlOgogICAgICAgIHByaW50KCIgICAgbm90aGluZyB0byBhc2sgd2l0aCwgc28ge30gaXMgdXNlZCIuZm9ybWF0KGFuc3dlcikpCgogICAgaWYgc2hvd246CiAgICAgICAgdHJ5OgogICAgICAgICAgICBpbXBvcnQgbWF0cGxvdGxpYi5weXBsb3QgYXMgcGx0CiAgICAgICAgICAgIHBsdC5jbG9zZSgiYWxsIikKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBwYXNzCgogICAgZiwgYmx1ciwgXyA9IGZyYW1lc1thbnN3ZXIgLSAxXQogICAgaWYgYmx1ciA+IDA6CiAgICAgICAgaGEgPSBnYXVzc2lhbl9maWx0ZXIoaGEsIGJsdXIpCiAgICBlbGlmIGJsdXIgPCAwOgogICAgICAgIHJlZCA9IGdhdXNzaWFuX2ZpbHRlcihyZWQsIC1ibHVyKQogICAgZGlhZ1siY2hvc2VuIl0gPSBhbnN3ZXIKICAgIGRpYWdbImZhY3RvciJdID0gZgogICAgZGlhZ1siYmx1ciJdID0gYmx1cgogICAgcmV0dXJuIGhhLCByZWQsIGYsIGRpYWcKCmRlZiBkZXNjcmliZV9jaG9pY2UoZGlhZyk6CiAgICAiIiJXaGF0IHdhcyBjaG9zZW4sIGluIGEgbGluZS4iIiIKICAgIGlmIGRpYWcuZ2V0KCJ3aHkiKToKICAgICAgICByZXR1cm4gIiAgbm8gY2hvaWNlIG9mZmVyZWQgKHt9KSIuZm9ybWF0KGRpYWdbIndoeSJdKQogICAgZiwgYmx1ciA9IGRpYWdbImZhY3RvciJdLCBkaWFnWyJibHVyIl0KICAgIHdoYXQgPSAoIkgtYWxwaGEgYmx1cnJlZCBieSB7Oi4yZn0gcHgiLmZvcm1hdChibHVyKSBpZiBibHVyID4gMCBlbHNlCiAgICAgICAgICAgICJSIGJsdXJyZWQgYnkgezouMmZ9IHB4Ii5mb3JtYXQoLWJsdXIpIGlmIGJsdXIgPCAwIGVsc2UKICAgICAgICAgICAgIm5laXRoZXIgZnJhbWUgYmx1cnJlZCIpCiAgICByZXR1cm4gKCIgIGNob2ljZSB7fSB0YWtlbjogZmFjdG9yIHs6LjJmfSwge31cbiIKICAgICAgICAgICAgIiAgICBqdWRnZWQgb24ge30gc3RhcnMgb3V0c2lkZSB0aGUgZ2FsYXh5IgogICAgICAgICAgICAuZm9ybWF0KGRpYWdbImNob3NlbiJdLCBmLCB3aGF0LCBkaWFnWyJuX3N0YXJzIl0pKQoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KIyBCbHVlX0NsdXN0ZXJzX0Zyb21fSW1hZ2VzLnB5CiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KCmZyb20gc2NpcHkuc3BhdGlhbCBpbXBvcnQgY0tEVHJlZQpmcm9tIHBob3R1dGlscy5hcGVydHVyZSBpbXBvcnQgQXBlcnR1cmVTdGF0cwoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tIFBhcmFtZXRlcnMgLS0tLS0tLS0tLS0tLS0tLS0tLQojIE1lYXN1cmluZyB0aGUgc291cmNlcwpBUEVSVFVSRV9TQ0FMRSA9IDIgICAgICAgICAgICAgICAjIGFwZXJ0dXJlIHJhZGl1cyA9IHRoaXMgeCB0aGUgdHlwaWNhbCBzdGFyIEZXSE0gb2YgdGhlCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICMgZnJhbWVzIC0gT05FIHJhZGl1cyBmb3IgZXZlcnkgc291cmNlLCBpbiBib3RoIGJhbmRzCkFOTlVMVVNfSU5ORVJfU0NBTEUgPSAyLjUgICAgICAgICMgdGhlIHNreSByaW5nIGFyb3VuZCBlYWNoIHNvdXJjZSwgaW4gdW5pdHMgb2YgdGhlCkFOTlVMVVNfT1VURVJfU0NBTEUgPSAzICAgICAgICAgICMgYXBlcnR1cmUgcmFkaXVzCkZXSE1fV0lORE9XX1NJWkUgPSAxMCAgICAgICAgICAgICMgaGFsZi1zaXplIG9mIHRoZSBib3ggYSBzb3VyY2UncyB3aWR0aCBpcyBtZWFzdXJlZCBpbgpEQU9GSU5EX0ZXSE0gPSBOb25lICAgICAgICAgICAgICAjIE5vbmUgPSBtZWFzdXJlIGl0IGZyb20gdGhlIGRhdGEgKHJlY29tbWVuZGVkKQpTRUVJTkdfRkFMTEJBQ0sgPSA0LjAgICAgICAgICAgICAjIFVzZWQgb25seSBpZiB0aGUgbWVhc3VyZW1lbnQgY2Fubm90IGJlIG1hZGUKU0VFSU5HX01JTiwgU0VFSU5HX01BWCA9IDEuNSwgMTUuMCAgICMgU2FuaXR5IHJhbmdlIGZvciB0aGUgbWVhc3VyZWQgdmFsdWUsIHBpeGVscwpTSUdNQV9DTElQID0gMy4wICAgICAgICAgICAgICAgICAjIFNpZ21hIGNsaXBwaW5nIGxldmVsIGZvciBiYWNrZ3JvdW5kIHN0YXRpc3RpY3MKREVURUNUSU9OX1RIUkVTSE9MRF9TSUdNQSA9IDMuMCAgIyBEZXRlY3Rpb24gdGhyZXNob2xkIGluIHVuaXRzIG9mIGJhY2tncm91bmQgc2lnbWEKUEVBS19NSU5fU1REID0gMy4wICAgICAgICAgICAgICAgIyBNaW5pbXVtIHBlYWsgc2lnbmFsLXRvLW5vaXNlIHJhdGlvIGZvciBhIGRldGVjdGlvbgpCQUNLR1JPVU5EX0JPWF9TSVpFID0gNDggICAgICAgICAjIEdyaWQgY2VsbCBmb3IgdGhlIGxvY2FsIGJhY2tncm91bmQsIHBpeGVscwpCQUNLR1JPVU5EX0ZJTFRFUl9TSVpFID0gMyAgICAgICAjIE1lZGlhbiBmaWx0ZXIgYXBwbGllZCBhY3Jvc3MgdGhlIGdyaWQKU0VFSU5HX01FQVNVUkVEID0ge30KQkdfU1VCX0ZVTkMgPSBucC5uYW5tZWRpYW4gICAgICAgIyBGdW5jdGlvbiB1c2VkIHRvIGNvbXB1dGUgYW5kIHN1YnRyYWN0IGJhY2tncm91bmQgbGV2ZWwKQVNUUk9BTElHTl9NQVhfU0NBTEVfRVJST1IgPSAwLjA1ICAjIDUlCkdBSU5fRV9QRVJfQURVID0gMS4wICAgICAgICAgICAgICMgZWxlY3Ryb25zIHBlciBBRFUsIHVzZWQgb25seSBpZiB0aGUgaGVhZGVyIGhhcyBub25lClJFQURfTk9JU0VfRSA9IDUuMCAgICAgICAgICAgICAgICMgcmVhZCBub2lzZSBpbiBlbGVjdHJvbnMsIHVzZWQgb25seSBpZiB0aGUgaGVhZGVyIGhhcyBub25lCkVER0VfTUFSR0lOX1BYID0gMzAKCiMgQ2FsaWJyYXRpb24KQ0FMSUJfQ0FUQUxPR1VFID0gIkFQQVNTIiAgICAgICAgIyBkZWZhdWx0OyAiR0FJQSIgZm9yIEdhaWEgRFIzIHN5bnRoZXRpYyBCIGFuZCBWCkNBTElCX01BVENIX1JBRElVUyA9IDMuMCAgICAgICAgICMgYXJjc2VjOiBhIGNhdGFsb2d1ZSBzdGFyIGFuZCBhIHNvdXJjZSB0aGlzIGNsb3NlIGFyZSBvbmUKWFlfTUFUQ0hfVE9MRVJBTkNFID0gMy4wICAgICAgICAgIyBwaXhlbHMsIG9uIHRoZSBWIGdyaWQsIGFmdGVyIHRoZSBwbGF0ZSBzb2x1dGlvbiBpcyBjaGVja2VkCkNBTElCX01JTl9TVEFSUyA9IDUgICAgICAgICAgICAgICMgZmV3ZXIgY2FsaWJyYXRpb24gc3RhcnMgdGhhbiB0aGlzIGFuZCB0aGUgcnVuIHN0b3BzCkNBTElCX01BWF9JTlNUX0VSUiA9IDAuMDUgICAgICAgICMgbWFnOyBvbmx5IHdlbGwtbWVhc3VyZWQgc3RhcnMgc2V0IHRoZSBjYWxpYnJhdGlvbgpDQUxJQl9NQVhfQ0FUX0VSUiA9IDAuMTAgICAgICAgICAjIG1hZzsgYW5kIG9ubHkgd2VsbC1tZWFzdXJlZCBjYXRhbG9ndWUgZW50cmllcwpDQUxJQl9DTElQX1NJR01BID0gMy4wICAgICAgICAgICAjIHN0YXJzIGZ1cnRoZXIgdGhhbiB0aGlzIGZyb20gdGhlIGZpdCBhcmUgbGVmdCBvdXQKQ0FMSUJfTUlOX09VVFNJREUgPSAxMCAgICAgICAgICAgIyB1c2Ugb25seSBzdGFycyBiZXlvbmQgUjI1IHdoZW4gYXQgbGVhc3QgdGhpcyBtYW55IGFyZQpXQ1NfQ0hFQ0tfTUlOX0ZSQUNUSU9OID0gMC41ICAgICAjIHNoYXJlIG9mIGJyaWdodCBzdGFycyB0aGF0IG11c3QgaGF2ZSBhIEdhaWEgc3RhciBuZWFyYnkKCiMgVGhlIGJsdWUga25vdHMKQ01EX0NPTE9SX01JTiA9IC0wLjUgICAgICAgICAgICAgIyBhIGJsdWUga25vdCBoYXMgKEItVikwIGJldHdlZW4gdGhlc2UgdHdvIHZhbHVlcwpDTURfQ09MT1JfTUFYID0gMC41CktOT1RfTUFYX0JWX0VSUiA9IDAuMiAgICAgICAgICAgICMgYW5kIGEgQi1WIGVycm9yIG5vIGxhcmdlciB0aGFuIHRoaXMsIGluIG1hZwpDTURfTUFYX0JWX0VSUiA9IDAuMyAgICAgICAgICAgICAjIHNvdXJjZXMgbm9pc2llciB0aGFuIHRoaXMgYXJlIGxlZnQgb3V0IG9mIHRoZSBkaWFncmFtcwpGT1JFR1JPVU5EX1RPTCA9IDMuMCAgICAgICAgICAgICAjIHBpeGVsczogYSBzb3VyY2UgdGhpcyBjbG9zZSB0byBhIEdhaWEgZm9yZWdyb3VuZCBzdGFyIGlzIGl0ClN0ZXBfc2l6ZSA9IDEwICAgICAgICAgICAgICAgICAgICMgUmVzb2x1dGlvbiBvZiBUaGUgQmx1ZSBLbm90cyBEZW5zaXR5IERpYWdyYW0KClJfViA9IDMuMSAgICAgICAgICAgICAgICAgICAgICAgICMgQV9WIC8gRShCLVYpIGZvciBkdXN0IGluIG91ciBvd24gZ2FsYXh5CgpBUkdTID0gTm9uZSAgICAgICAgICAgICAgICAgICAgICAjIHRoZSBjb21tYW5kLWxpbmUgYXJndW1lbnRzLCBpZiBhbnkKCgojIC0tLS0tLS0tLS0tLS0tLS0tLS0gUXVlc3Rpb25zIC0tLS0tLS0tLS0tLS0tLS0tLS0KCmRlZiBfYWNjZXB0X2RlZmF1bHRzKCk6CiAgICByZXR1cm4gQVJHUyBpcyBub3QgTm9uZSBhbmQgZ2V0YXR0cihBUkdTLCAieWVzIiwgRmFsc2UpCgoKZGVmIF9hc2socHJvbXB0LCBkZWZhdWx0LCBjYXN0PXN0ciwgY2hvaWNlcz1Ob25lKToKICAgICIiIkFzayBvbmUgcXVlc3Rpb24uIEVudGVyIGFjY2VwdHMgdGhlIGRlZmF1bHQgc2hvd24gaW4gYnJhY2tldHMuCgogICAgT25seSBhbiBlbXB0eSBhbnN3ZXIgbWVhbnMgInRoZSBkZWZhdWx0Ii4gQW4gYW5zd2VyIHRoYXQgY2Fubm90IGJlIHJlYWQgaXMKICAgIHJlZnVzZWQgYW5kIHRoZSBxdWVzdGlvbiBhc2tlZCBhZ2FpbiAtIGl0IHVzZWQgdG8gYmUgcmVwbGFjZWQgYnkgdGhlCiAgICBkZWZhdWx0IHdpdGhvdXQgYSB3b3JkLCBzbyBhIG1pc3R5cGVkIGRpc3RhbmNlIGJlY2FtZSAxMCBNcGMuIEEgZGVmYXVsdCBvZgogICAgTm9uZSBtZWFucyB0aGVyZSBpcyBub25lOiBhbiBhbnN3ZXIgaXMgbmVlZGVkLgogICAgIiIiCiAgICBpZiBfYWNjZXB0X2RlZmF1bHRzKCk6CiAgICAgICAgaWYgZGVmYXVsdCBpcyBOb25lOgogICAgICAgICAgICBzeXMuZXhpdChmIltpbnB1dF0gJ3twcm9tcHR9JyBoYXMgbm8gZGVmYXVsdCwgc28gLS15ZXMgY2Fubm90IGFuc3dlciAiCiAgICAgICAgICAgICAgICAgICAgIGYiaXQgLSBnaXZlIGl0IG9uIHRoZSBjb21tYW5kIGxpbmUiKQogICAgICAgIHByaW50KGYie3Byb21wdH0gW3tkZWZhdWx0fV06IChhY2NlcHRlZCwgLS15ZXMpIikKICAgICAgICByZXR1cm4gZGVmYXVsdAogICAgc2hvd24gPSAiIiBpZiBkZWZhdWx0IGlzIE5vbmUgZWxzZSBmIiBbe2RlZmF1bHR9XSIKICAgIHdoaWxlIFRydWU6CiAgICAgICAgcyA9IGlucHV0KGYie3Byb21wdH17c2hvd259OiAiKS5zdHJpcCgpCiAgICAgICAgaWYgcyA9PSAiIjoKICAgICAgICAgICAgaWYgZGVmYXVsdCBpcyBOb25lOgogICAgICAgICAgICAgICAgcHJpbnQoIiAgIGFuIGFuc3dlciBpcyBuZWVkZWQgaGVyZSIpCiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICByZXR1cm4gZGVmYXVsdAogICAgICAgIHRyeToKICAgICAgICAgICAgdiA9IGNhc3QocykKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBwcmludChmIiAgIGNvdWxkIG5vdCByZWFkICd7c30nIC0gdHJ5IGFnYWluIikKICAgICAgICAgICAgY29udGludWUKICAgICAgICBpZiBjaG9pY2VzIGFuZCB2IG5vdCBpbiBjaG9pY2VzOgogICAgICAgICAgICBwcmludChmIiAgIGFuc3dlciBvbmUgb2Y6IHsnLCAnLmpvaW4oY2hvaWNlcyl9IikKICAgICAgICAgICAgY29udGludWUKICAgICAgICByZXR1cm4gdgoKCmRlZiBfbm9ybV9wYXRoKHA6IHN0cikgLT4gc3RyOgogICAgIiIiUmVtb3ZlIGV4dHJhIHF1b3RlcyBhbmQgbm9ybWFsaXplIGZpbGVzeXN0ZW0gcGF0aC4iIiIKICAgIHAgPSBwLnN0cmlwKCkuc3RyaXAoJyInKS5zdHJpcCgiJyIpCiAgICByZXR1cm4gb3MucGF0aC5ub3JtcGF0aChwKSBpZiBwIGVsc2UgIiIKCgpkZWYgb2ZmZXJlZChsYWJlbCwgdmFsdWUsIHNvdXJjZSwgdW5pdD0iIiwgb3ZlcnJpZGU9Tm9uZSk6CiAgICAiIiJBIGNhdGFsb2d1ZSB2YWx1ZSwgc2hvd24gd2l0aCB3aGVyZSBpdCBjYW1lIGZyb20sIGFuZCBhY2NlcHRlZCB3aXRoCiAgICBFbnRlciBvciByZXBsYWNlZCBieSB0eXBpbmcgYW5vdGhlci4gQSB2YWx1ZSBnaXZlbiBvbiB0aGUgY29tbWFuZCBsaW5lCiAgICBpcyB1c2VkIGFzIGl0IGlzLiIiIgogICAgaWYgb3ZlcnJpZGUgaXMgbm90IE5vbmU6CiAgICAgICAgcHJpbnQoZiJ7bGFiZWx9OiB7b3ZlcnJpZGV9e3VuaXR9IChmcm9tIHRoZSBjb21tYW5kIGxpbmUpIikKICAgICAgICByZXR1cm4gZmxvYXQob3ZlcnJpZGUpCiAgICBpZiB2YWx1ZSBpcyBOb25lOgogICAgICAgIHByaW50KGYiW2xvb2t1cF0gbm8ge2xhYmVsLmxvd2VyKCl9IGZvdW5kIGluIHRoZSBjYXRhbG9ndWUgLSBwbGVhc2UgZ2l2ZSBvbmUiKQogICAgICAgIHJldHVybiBmbG9hdChfYXNrKGxhYmVsLCBOb25lLCBmbG9hdCkpCiAgICBwcmludChmIltsb29rdXBdIHtsYWJlbH06IHt2YWx1ZTpnfXt1bml0fSAoe3NvdXJjZX0pIikKICAgIHJldHVybiBmbG9hdChfYXNrKGYie2xhYmVsfSAtIEVudGVyIHRvIGFjY2VwdCwgb3IgdHlwZSBhbm90aGVyIiwgdmFsdWUsIGZsb2F0KSkKCgpkZWYgc3RvcChtZXNzYWdlKToKICAgICIiIlN0b3AgdGhlIHJ1biB3aXRoIGEgbWVzc2FnZSB0aGF0IHNheXMgd2hhdCB0byBkby4iIiIKICAgIHByaW50KCIiKQogICAgcHJpbnQoIioiICogNzgpCiAgICBwcmludCgiU1RPUFBFRDogIiArIG1lc3NhZ2UpCiAgICBwcmludCgiKiIgKiA3OCkKICAgIHJhaXNlIFN5c3RlbUV4aXQoMSkKCgpkZWYgbG91ZChtZXNzYWdlKToKICAgIHByaW50KCIiKQogICAgcHJpbnQoIiEiICogNzgpCiAgICBwcmludChtZXNzYWdlKQogICAgcHJpbnQoIiEiICogNzgpCiAgICBwcmludCgiIikKCgojIC0tLS0tLS0tLS0tLS0tLS0tLS0gUmVhZGluZyBmcmFtZXMgYW5kIHRoZSBkZXRlY3RvciAtLS0tLS0tLS0tLS0tLS0tLS0tCgpkZWYgcmVhZF9pbWFnZShwYXRoKToKICAgICIiIihkYXRhIGFzIGZsb2F0MzIsIGhlYWRlcikgZnJvbSB0aGUgZmlyc3QgSERVIHRoYXQgaG9sZHMgYSAyLUQgaW1hZ2UuCgogICAgVGhlIGZpcnN0IEhEVSBpcyBub3QgYWx3YXlzIGl0OiBhbiBMQ08gLmZpdHMuZnogZnJhbWUga2VlcHMgaXRzIGltYWdlIGluCiAgICB0aGUgc2Vjb25kLCBhbmQgcmVhZGluZyB0aGUgZmlyc3QgcmV0dXJuZWQgbm90aGluZy4KICAgICIiIgogICAgd2l0aCBmaXRzLm9wZW4ocGF0aCkgYXMgaGR1bDoKICAgICAgICBmb3IgaGR1IGluIGhkdWw6CiAgICAgICAgICAgIGlmIGhkdS5kYXRhIGlzIG5vdCBOb25lIGFuZCBnZXRhdHRyKGhkdS5kYXRhLCAibmRpbSIsIDApID09IDI6CiAgICAgICAgICAgICAgICBoZWFkZXIgPSBoZHUuaGVhZGVyLmNvcHkoKQogICAgICAgICAgICAgICAgZm9yIGtleSBpbiAoIlhURU5TSU9OIiwgIlBDT1VOVCIsICJHQ09VTlQiLCAiQlpFUk8iLCAiQlNDQUxFIiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICJCTEFOSyIsICJaSU1BR0UiLCAiWkJJVFBJWCIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAiWk5BWElTIiwgIlpOQVhJUzEiLCAiWk5BWElTMiIsICJaVElMRTEiLCAiWlRJTEUyIiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICJaQ01QVFlQRSIsICJaUVVBTlRJWiIsICJaRElUSEVSMCIsICJaU0lNUExFIiwgIlpFWFRFTkQiKToKICAgICAgICAgICAgICAgICAgICBoZWFkZXIucmVtb3ZlKGtleSwgaWdub3JlX21pc3Npbmc9VHJ1ZSwgcmVtb3ZlX2FsbD1UcnVlKQogICAgICAgICAgICAgICAgcmV0dXJuIG5wLmFzYXJyYXkoaGR1LmRhdGEsIGR0eXBlPW5wLmZsb2F0MzIpLCBoZWFkZXIKICAgIHJhaXNlIFJ1bnRpbWVFcnJvcihmIm5vIDItRCBpbWFnZSBpbiB7cGF0aH0iKQoKCmRlZiBjb21iaW5lZF9jb3VudChoZWFkZXIpOgogICAgIiIiSG93IG1hbnkgZnJhbWVzIHdlbnQgaW50byB0aGlzIGZpbGUsIGFuZCBob3cgdGhleSB3ZXJlIGNvbWJpbmVkLiIiIgogICAgbiA9IGhlYWRlci5nZXQoIk5DT01CSU5FIikKICAgIGhvdyA9IHN0cihoZWFkZXIuZ2V0KCJDT01CVFlQRSIsICJtZWRpYW4iKSkubG93ZXIoKQogICAgdHJ5OgogICAgICAgIGlmIG4gYW5kIGludChuKSA+IDE6CiAgICAgICAgICAgIHJldHVybiBpbnQobiksIF9jb21iaW5lX2tpbmQoaG93KQogICAgZXhjZXB0IChUeXBlRXJyb3IsIFZhbHVlRXJyb3IpOgogICAgICAgIHBhc3MKICAgIGZvciBsaW5lIGluIGhlYWRlci5nZXQoIkhJU1RPUlkiLCBbXSk6CiAgICAgICAgbSA9IHJlLnNlYXJjaChyIihtZWRpYW58YXZlcmFnZXxtZWFufHN1bSlccytvZlxzKyhcZCspXHMrIgogICAgICAgICAgICAgICAgICAgICAgciIoPzppbWFnZXN8ZXhwb3N1cmVzfGZyYW1lcykiLCBzdHIobGluZSksIHJlLkkpCiAgICAgICAgaWYgbToKICAgICAgICAgICAgcmV0dXJuIGludChtLmdyb3VwKDIpKSwgX2NvbWJpbmVfa2luZChtLmdyb3VwKDEpKQogICAgcmV0dXJuIDEsICJzaW5nbGUiCgoKZGVmIF9jb21iaW5lX2tpbmQod29yZCk6CiAgICB3b3JkID0gc3RyKHdvcmQpLmxvd2VyKCkKICAgIHJldHVybiAibWVkaWFuIiBpZiAibWVkIiBpbiB3b3JkIGVsc2UgInN1bSIgaWYgInN1bSIgaW4gd29yZCBlbHNlICJtZWFuIgoKCmRlZiBkZXRlY3Rvcl9nYWluKGhlYWRlcik6CiAgICAiIiJFbGVjdHJvbnMgcGVyIEFEVSBvZiBUSElTIGZpbGUsIGFuZCB3aGVyZSB0aGUgbnVtYmVyIGNhbWUgZnJvbS4KCiAgICBFR0FJTiBpcyBwcmVmZXJyZWQ6IG9uIHNvbWUgY2FtZXJhcyBHQUlOIGlzIHRoZSBnYWluIFNFVFRJTkcgKDUxIG9uIHRoZQogICAgS2lubmVyZXQgY29sb3VyIGNhbWVyYSksIG5vdCBlbGVjdHJvbnMgcGVyIEFEVS4gQSBtZWRpYW4gb2YgTiBmcmFtZXMgaGFzCiAgICB0aGUgbm9pc2Ugb2Ygb25lIGZyYW1lIHdpdGggTiB4IDIvcGkgdGltZXMgdGhlIGdhaW4sIGEgbWVhbiBvZiBOIGZyYW1lcwogICAgTiB0aW1lczsgYSBzdW0ga2VlcHMgdGhlIGdhaW4gb2Ygb25lIGZyYW1lLgogICAgIiIiCiAgICBnYWluLCBzb3VyY2UgPSBOb25lLCBOb25lCiAgICBmb3Iga2V5IGluICgiRUdBSU4iLCAiR0FJTiIpOgogICAgICAgIHRyeToKICAgICAgICAgICAgdiA9IGZsb2F0KGhlYWRlci5nZXQoa2V5KSkKICAgICAgICBleGNlcHQgKFR5cGVFcnJvciwgVmFsdWVFcnJvcik6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgaWYgMCA8IHYgPCAyMDoKICAgICAgICAgICAgZ2Fpbiwgc291cmNlID0gdiwga2V5CiAgICAgICAgICAgIGJyZWFrCiAgICBpZiBnYWluIGlzIE5vbmU6CiAgICAgICAgZ2Fpbiwgc291cmNlID0gR0FJTl9FX1BFUl9BRFUsICJkZWZhdWx0IGNvbnN0YW50IgogICAgbiwgaG93ID0gY29tYmluZWRfY291bnQoaGVhZGVyKQogICAgaWYgbiA+IDEgYW5kIGhvdyA9PSAic3VtIjoKICAgICAgICByZXR1cm4gZ2FpbiwgZiJ7c291cmNlfSAoc3VtIG9mIHtufSBmcmFtZXMpIgogICAgaWYgbiA+IDE6CiAgICAgICAgZmFjdG9yID0gbiAqICgyLjAgLyBucC5waSBpZiBob3cgPT0gIm1lZGlhbiIgZWxzZSAxLjApCiAgICAgICAgcmV0dXJuIGdhaW4gKiBmYWN0b3IsIGYie3NvdXJjZX0geCB7ZmFjdG9yOi4xZn0gKHtob3d9IG9mIHtufSBmcmFtZXMpIgogICAgcmV0dXJuIGdhaW4sIHNvdXJjZQoKCmRlZiByZWFkX25vaXNlKGhlYWRlcik6CiAgICBmb3Iga2V5IGluICgiUkROT0lTRSIsICJSRUFETk9JUyIsICJSRE5PSVNFMSIsICJST04iKToKICAgICAgICB0cnk6CiAgICAgICAgICAgIHYgPSBmbG9hdChoZWFkZXIuZ2V0KGtleSkpCiAgICAgICAgICAgIGlmIHYgPiAwOgogICAgICAgICAgICAgICAgcmV0dXJuIHYsIGtleQogICAgICAgIGV4Y2VwdCAoVHlwZUVycm9yLCBWYWx1ZUVycm9yKToKICAgICAgICAgICAgY29udGludWUKICAgIHJldHVybiBSRUFEX05PSVNFX0UsICJkZWZhdWx0IGNvbnN0YW50IgoKCmRlZiBzYXR1cmF0aW9uX2xldmVsKHJhdywgaGVhZGVyKToKICAgICIiIlRoZSBwaXhlbCB2YWx1ZSBhYm92ZSB3aGljaCBhIHNvdXJjZSBjb3VudHMgYXMgc2F0dXJhdGVkLCBhbmQgaG93IGl0CiAgICB3YXMgZm91bmQ6IGZyb20gU0FUVVJBVEUgKG9yIE1BWExJTikgaW4gdGhlIGhlYWRlciB3aGVuIHRoZXJlIGlzIG9uZSwKICAgIG90aGVyd2lzZSBmcm9tIHRoZSBwaWxlLXVwIG9mIHNhdHVyYXRlZCBwaXhlbHMgYXQgdGhlIHRvcCBvZiB0aGUgaW1hZ2UncwogICAgaGlzdG9ncmFtLiIiIgogICAgZm9yIGtleSBpbiAoIlNBVFVSQVRFIiwgIk1BWExJTiIsICJTQVRMRVZFTCIpOgogICAgICAgIHRyeToKICAgICAgICAgICAgdiA9IGZsb2F0KGhlYWRlci5nZXQoa2V5KSkKICAgICAgICBleGNlcHQgKFR5cGVFcnJvciwgVmFsdWVFcnJvcik6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgaWYgdiA+IDA6CiAgICAgICAgICAgIHJldHVybiBTQVRVUkFUSU9OX0ZSQUNUSU9OICogdiwgZiJ7U0FUVVJBVElPTl9GUkFDVElPTjpnfSB4IHtrZXl9ICh7djpnfSkiCiAgICBtID0gRURHRV9NQVJHSU5fUFgKICAgIGlubmVyID0gcmF3W206LW0sIG06LW1dIGlmIG1pbihyYXcuc2hhcGUpID4gNCAqIG0gZWxzZSByYXcKICAgIGlubmVyID0gaW5uZXJbbnAuaXNmaW5pdGUoaW5uZXIpXQogICAgdG9wID0gZmxvYXQobnAuc29ydChpbm5lcilbLTIwXSkKICAgIGhpZ2ggPSBpbm5lcltpbm5lciA+IDAuNCAqIHRvcF0KICAgIGxldmVsLCBob3cgPSB0b3AsICJ0aGUgaW1hZ2UncyBtYXhpbXVtIgogICAgaWYgaGlnaC5zaXplID4gMjAwOgogICAgICAgIGNvdW50cywgZWRnZXMgPSBucC5oaXN0b2dyYW0oaGlnaCwgYmlucz02MCwgcmFuZ2U9KDAuNCAqIHRvcCwgdG9wKSkKICAgICAgICBmb3IgaSBpbiByYW5nZSgxMCwgbGVuKGNvdW50cykpOgogICAgICAgICAgICBiZWZvcmUgPSBucC5tZWRpYW4oY291bnRzW2kgLSAxMDppXSkKICAgICAgICAgICAgaWYgY291bnRzW2ldID49IDIwIGFuZCBjb3VudHNbaV0gPiAyLjAgKiBtYXgoYmVmb3JlLCAxLjApOgogICAgICAgICAgICAgICAgbGV2ZWwgPSBmbG9hdChlZGdlc1tpXSkKICAgICAgICAgICAgICAgIGhvdyA9ICJ0aGUgcGlsZS11cCBvZiBzYXR1cmF0ZWQgcGl4ZWxzIGluIHRoZSBoaXN0b2dyYW0iCiAgICAgICAgICAgICAgICBicmVhawogICAgcmV0dXJuIFNBVFVSQVRJT05fRlJBQ1RJT04gKiBsZXZlbCwgZiJ7U0FUVVJBVElPTl9GUkFDVElPTjpnfSB4IHtob3d9ICh7bGV2ZWw6LjBmfSkiCgoKZGVmIHN1YnRyYWN0X2JhY2tncm91bmRfYW5kX3NhdmUocGF0aCwgb3V0X2Rpcik6CiAgICAiIiJTdWJ0cmFjdCB0aGUgZnJhbWUncyBtZWRpYW4gbGV2ZWwgYW5kIHNhdmUgdGhlIHJlc3VsdCBpbiBvdXRfZGlyLgoKICAgIFJldHVybnMgKHBhdGggd3JpdHRlbiwgcmF3IHBpeGVscywgYmFja2dyb3VuZC1zdWJ0cmFjdGVkIHBpeGVscywgaGVhZGVyKS4KICAgICIiIgogICAgcmF3LCBoZHIgPSByZWFkX2ltYWdlKHBhdGgpCiAgICBiZ192YWwgPSBmbG9hdChCR19TVUJfRlVOQyhyYXcpKQogICAgZGF0YV9zdWIgPSByYXcgLSBucC5mbG9hdDMyKGJnX3ZhbCkKICAgIHN0ZW0gPSBvcy5wYXRoLmJhc2VuYW1lKHBhdGgpCiAgICBmb3Igc3VmZml4IGluICgiLmZpdHMuZnoiLCAiLmZpdHMiLCAiLmZpdCIsICIuZnRzIik6CiAgICAgICAgaWYgc3RlbS5sb3dlcigpLmVuZHN3aXRoKHN1ZmZpeCk6CiAgICAgICAgICAgIHN0ZW0gPSBzdGVtWzotbGVuKHN1ZmZpeCldCiAgICAgICAgICAgIGJyZWFrCiAgICBvdXRfcGF0aCA9IG9zLnBhdGguam9pbihvdXRfZGlyLCBzdGVtICsgIl9iZ3N1Yi5maXRzIikKICAgIGZpdHMuUHJpbWFyeUhEVShkYXRhPWRhdGFfc3ViLCBoZWFkZXI9aGRyKS53cml0ZXRvKG91dF9wYXRoLCBvdmVyd3JpdGU9VHJ1ZSkKICAgIHByaW50KGYiW2Jnc3ViXSB3cm90ZSB7b3V0X3BhdGh9IChiZz17YmdfdmFsOi4zZn0pIikKICAgIHJldHVybiBvdXRfcGF0aCwgcmF3LCBkYXRhX3N1YiwgaGRyCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tIEZpbmRpbmcgYW5kIG1lYXN1cmluZyB0aGUgc291cmNlcyAtLS0tLS0tLS0tLS0tLS0tLS0tCgpkZWYgbG9jYWxfYmFja2dyb3VuZChkYXRhKToKICAgICIiIkJhY2tncm91bmQgYW5kIG5vaXNlIG9uIGEgY29hcnNlIGdyaWQsIHNvIGEgZ2FsYXh5J3Mgb3duIGxpZ2h0IGRvZXMgbm90CiAgICBzZXQgdGhlIGRldGVjdGlvbiB0aHJlc2hvbGQgZm9yIHRoZSBlbXB0eSBza3kgYXJvdW5kIGl0LiIiIgogICAgcmV0dXJuIEJhY2tncm91bmQyRChkYXRhLCAoQkFDS0dST1VORF9CT1hfU0laRSwgQkFDS0dST1VORF9CT1hfU0laRSksCiAgICAgICAgICAgICAgICAgICAgICAgIGZpbHRlcl9zaXplPShCQUNLR1JPVU5EX0ZJTFRFUl9TSVpFLCBCQUNLR1JPVU5EX0ZJTFRFUl9TSVpFKSwKICAgICAgICAgICAgICAgICAgICAgICAgc2lnbWFfY2xpcD1TaWdtYUNsaXAoc2lnbWE9U0lHTUFfQ0xJUCksCiAgICAgICAgICAgICAgICAgICAgICAgIGJrZ19lc3RpbWF0b3I9TWVkaWFuQmFja2dyb3VuZCgpKQoKCmRlZiBzdGFyX3dpZHRoKGRhdGEsIHgsIHksIGhhbGY9MTIsIHNreT0wLjApOgogICAgIiIiV2lkdGggYXQgaGFsZiBtYXhpbXVtIG9mIG9uZSBzb3VyY2UsIGluIHBpeGVscywgbWVhc3VyZWQgb24gdGhlIGRhdGEgYXMKICAgIGl0IGlzLCBhZnRlciBpdHMgbG9jYWwgc2t5IGlzIHRha2VuIG9mZi4KCiAgICBBIHR3by1kaW1lbnNpb25hbCBHYXVzc2lhbiBpcyBmaXR0ZWQgdG8gdGhlIHN0YW1wLCBhbmQgdGhlIEZXSE0gaXMKICAgIDIuMzU1IHggc3FydChzaWdtYV94IHNpZ21hX3kpLiBDb3VudGluZyB0aGUgcGl4ZWxzIGFib3ZlIGhhbGYgdGhlIHBlYWssIGFzCiAgICB0aGlzIGRpZCBiZWZvcmUsIGNhbiBvbmx5IHJldHVybiB3aG9sZSBhbmQgaGFsZiBwaXhlbHM6IGEgbWVkaWFuIG92ZXIgYQogICAgaHVuZHJlZCBzdGFycyB0aGVuIHNpdHMgb24gNC4wMCB3aGV0aGVyIHRoZSBzZWVpbmcgaXMgMy42IG9yIDQuNCBweC4gVGhlCiAgICBwaXhlbCBjb3VudCBpcyBzdGlsbCB0aGUgYW5zd2VyIHdoZW4gdGhlIGZpdCBmYWlscy4KICAgICIiIgogICAgeCwgeSA9IGludChyb3VuZCh4KSksIGludChyb3VuZCh5KSkKICAgIGlmIHggLSBoYWxmIDwgMCBvciB5IC0gaGFsZiA8IDAgb3IgeCArIGhhbGYgPj0gZGF0YS5zaGFwZVsxXSBvciB5ICsgaGFsZiA+PSBkYXRhLnNoYXBlWzBdOgogICAgICAgIHJldHVybiBucC5uYW4KICAgIGN1dCA9IG5wLmFzYXJyYXkoZGF0YVt5IC0gaGFsZjp5ICsgaGFsZiwgeCAtIGhhbGY6eCArIGhhbGZdLCBmbG9hdCkgLSBza3kKICAgIHBlYWsgPSBjdXQubWF4KCkKICAgIGlmIG5vdCBucC5pc2Zpbml0ZShwZWFrKSBvciBwZWFrIDw9IDA6CiAgICAgICAgcmV0dXJuIG5wLm5hbgogICAgaWR4ID0gbnAuYXJnd2hlcmUoY3V0ID4gcGVhayAvIDIpCiAgICBpZiBub3QgbGVuKGlkeCk6CiAgICAgICAgcmV0dXJuIG5wLm5hbgogICAgY29hcnNlID0gbnAubWVhbihbbnAucHRwKGlkeFs6LCAxXSkgKyAxLCBucC5wdHAoaWR4WzosIDBdKSArIDFdKQogICAgdHJ5OgogICAgICAgIGZyb20gYXN0cm9weS5tb2RlbGluZyBpbXBvcnQgbW9kZWxzLCBmaXR0aW5nCiAgICAgICAgeXksIHh4ID0gbnAubWdyaWRbMDpjdXQuc2hhcGVbMF0sIDA6Y3V0LnNoYXBlWzFdXQogICAgICAgIHB5LCBweCA9IG5wLnVucmF2ZWxfaW5kZXgobnAuYXJnbWF4KGN1dCksIGN1dC5zaGFwZSkKICAgICAgICBzMCA9IG1heChjb2Fyc2UgLyAyLjM1NSwgMC41KQogICAgICAgIGcgPSBtb2RlbHMuR2F1c3NpYW4yRChwZWFrLCBweCwgcHksIHMwLCBzMCwgMC4wKSArIG1vZGVscy5Db25zdDJEKDAuMCkKICAgICAgICB3aXRoIHdhcm5pbmdzLmNhdGNoX3dhcm5pbmdzKCk6CiAgICAgICAgICAgIHdhcm5pbmdzLnNpbXBsZWZpbHRlcigiaWdub3JlIikKICAgICAgICAgICAgZml0ID0gZml0dGluZy5MZXZNYXJMU1FGaXR0ZXIoKShnLCB4eCwgeXksIG5wLm5hbl90b19udW0oY3V0KSkKICAgICAgICBzeCwgc3kgPSBhYnMoZml0Lnhfc3RkZGV2XzAudmFsdWUpLCBhYnMoZml0Lnlfc3RkZGV2XzAudmFsdWUpCiAgICAgICAgZndobSA9IDIuMzU0OCAqIG5wLnNxcnQoc3ggKiBzeSkKICAgICAgICBpZiAobnAuaXNmaW5pdGUoZndobSkgYW5kIGZpdC5hbXBsaXR1ZGVfMC52YWx1ZSA+IDAgYW5kIDAuNSA8IGZ3aG0gPCAyICogaGFsZgogICAgICAgICAgICAgICAgYW5kIGFicyhmaXQueF9tZWFuXzAudmFsdWUgLSBweCkgPCAyIGFuZCBhYnMoZml0LnlfbWVhbl8wLnZhbHVlIC0gcHkpIDwgMik6CiAgICAgICAgICAgIHJldHVybiBmbG9hdChmd2htKQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICBwYXNzCiAgICByZXR1cm4gZmxvYXQoY29hcnNlKQoKCmRlZiBfeHlfb2YodGFibGUpOgogICAgIiIieCBhbmQgeSBvZiBhIERBT1N0YXJGaW5kZXIgdGFibGUuIHBob3R1dGlscyAzLjAgcmVuYW1lZCB4Y2VudHJvaWQgYW5kCiAgICB5Y2VudHJvaWQgdG8geF9jZW50cm9pZCBhbmQgeV9jZW50cm9pZDsgYm90aCBhcmUgYWNjZXB0ZWQuIiIiCiAgICBuYW1lcyA9IHRhYmxlLmNvbG5hbWVzCiAgICB4cyA9ICJ4X2NlbnRyb2lkIiBpZiAieF9jZW50cm9pZCIgaW4gbmFtZXMgZWxzZSAieGNlbnRyb2lkIgogICAgeXMgPSAieV9jZW50cm9pZCIgaWYgInlfY2VudHJvaWQiIGluIG5hbWVzIGVsc2UgInljZW50cm9pZCIKICAgIHJldHVybiBucC5hc2FycmF5KHRhYmxlW3hzXSwgZmxvYXQpLCBucC5hc2FycmF5KHRhYmxlW3lzXSwgZmxvYXQpCgoKZGVmIGVzdGltYXRlX3NlZWluZyhkYXRhLCBiYWNrZ3JvdW5kLCBybXMsIGJhbmQsIHJhdz1Ob25lLCBzYXRfbGV2ZWw9Tm9uZSk6CiAgICAiIiJIb3cgd2lkZSBhIHN0YXIgaXMgaW4gdGhpcyBmcmFtZTogdGhlIG1lZGlhbiB3aWR0aCBvZiBicmlnaHQgc3RhcnMgdGhhdAogICAgYXJlIE5PVCBzYXR1cmF0ZWQuIEEgc2F0dXJhdGVkIHN0YXIgaGFzIGEgZmxhdCB0b3AsIHNvIGl0cyB3aWR0aCBhdCBoYWxmCiAgICB0aGUgZmxhdCB0b3Agc2F5cyBub3RoaW5nIGFib3V0IHRoZSBzZWVpbmcgLSBhbmQgdGhlIGFwZXJ0dXJlIG9mIGV2ZXJ5CiAgICBzb3VyY2UgaXMgc2l6ZWQgZnJvbSB0aGlzIG51bWJlci4iIiIKICAgIHRyeToKICAgICAgICBmb3VuZCA9IERBT1N0YXJGaW5kZXIoZndobT1TRUVJTkdfRkFMTEJBQ0ssIHRocmVzaG9sZD0yMC4wICogcm1zKShkYXRhIC0gYmFja2dyb3VuZCkKICAgICAgICBpZiBmb3VuZCBpcyBub3QgTm9uZSBhbmQgbGVuKGZvdW5kKSA+PSA1OgogICAgICAgICAgICBmeCwgZnkgPSBfeHlfb2YoZm91bmQpCiAgICAgICAgICAgIGZsdXggPSBucC5hc2FycmF5KGZvdW5kWydmbHV4J10sIGZsb2F0KQogICAgICAgICAgICBpZiByYXcgaXMgbm90IE5vbmUgYW5kIHNhdF9sZXZlbCBpcyBub3QgTm9uZToKICAgICAgICAgICAgICAgIHBlYWtzID0gbnAuYXJyYXkoW25wLm5hbm1heChyYXdbbWF4KGludChyb3VuZCh5KSkgLSAxLCAwKTppbnQocm91bmQoeSkpICsgMiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbWF4KGludChyb3VuZCh4KSkgLSAxLCAwKTppbnQocm91bmQoeCkpICsgMl0pCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBmb3IgeCwgeSBpbiB6aXAoZngsIGZ5KV0pCiAgICAgICAgICAgICAgICBvayA9IHBlYWtzIDwgc2F0X2xldmVsCiAgICAgICAgICAgICAgICBmeCwgZnksIGZsdXggPSBmeFtva10sIGZ5W29rXSwgZmx1eFtva10KICAgICAgICAgICAgaWYgbGVuKGZ4KSA+PSA1OgogICAgICAgICAgICAgICAgb3JkZXIgPSBucC5hcmdzb3J0KGZsdXgpWy0yMDA6XQogICAgICAgICAgICAgICAgd2lkdGhzID0gbnAuYXNhcnJheShbc3Rhcl93aWR0aChkYXRhIC0gYmFja2dyb3VuZCwgZnhbaV0sIGZ5W2ldKQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZm9yIGkgaW4gb3JkZXJdLCBkdHlwZT1mbG9hdCkKICAgICAgICAgICAgICAgIHNlZWluZyA9IG5wLm5hbm1lZGlhbih3aWR0aHMpCiAgICAgICAgICAgICAgICBpZiBucC5pc2Zpbml0ZShzZWVpbmcpIGFuZCBTRUVJTkdfTUlOIDw9IHNlZWluZyA8PSBTRUVJTkdfTUFYOgogICAgICAgICAgICAgICAgICAgIHByaW50KGYiICAge2JhbmR9OiBzdGFycyBhcmUge3NlZWluZzouMmZ9IHB4IHdpZGUgKG1lZGlhbiBvZiAiCiAgICAgICAgICAgICAgICAgICAgICAgICAgZiJ7aW50KG5wLmlzZmluaXRlKHdpZHRocykuc3VtKCkpfSBicmlnaHQgdW5zYXR1cmF0ZWQgc3RhcnMpIiwKICAgICAgICAgICAgICAgICAgICAgICAgICBmbHVzaD1UcnVlKQogICAgICAgICAgICAgICAgICAgIHJldHVybiBmbG9hdChzZWVpbmcpCiAgICAgICAgcHJpbnQoZiIgICB7YmFuZH06IGNvdWxkIG5vdCBtZWFzdXJlIHRoZSBzdGFyIHdpZHRoLCB1c2luZyIKICAgICAgICAgICAgICBmIiB7U0VFSU5HX0ZBTExCQUNLfSBweCIsIGZsdXNoPVRydWUpCiAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGV4YzoKICAgICAgICBwcmludChmIiAgIHtiYW5kfTogc3RhciB3aWR0aCBtZWFzdXJlbWVudCBmYWlsZWQgKHtleGN9KSwgdXNpbmciCiAgICAgICAgICAgICAgZiIge1NFRUlOR19GQUxMQkFDS30gcHgiLCBmbHVzaD1UcnVlKQogICAgcmV0dXJuIGZsb2F0KFNFRUlOR19GQUxMQkFDSykKCgpkZWYgZGV0ZWN0X3NvdXJjZXMoZGF0YSwgcmF3LCBiYW5kLCBzYXRfbGV2ZWwsIHdpbmRvdz1Ob25lKToKICAgICIiIkZpbmQgdGhlIHNvdXJjZXMgaW4gb25lIGZyYW1lLiBSZXR1cm5zICh4LCB5LCBmbHV4IGVzdGltYXRlKSBpbiB0aGUKICAgIGNvb3JkaW5hdGVzIG9mIHRoZSB3aG9sZSBmcmFtZSwgYW5kIHRoZSBmcmFtZSdzIHR5cGljYWwgc3RhciB3aWR0aC4KCiAgICBgd2luZG93YCBpcyAoeTAsIHkxLCB4MCwgeDEpOiB0aGUgcGFydCBvZiB0aGUgZnJhbWUgdGhlIGdhbGF4eSBvY2N1cGllcy4KICAgIEZpbmRpbmcgc291cmNlcyBpcyBwYWlkIGZvciBieSB0aGUgcGl4ZWwsIHNvIHdvcmtpbmcgb3V0c2lkZSB0aGUgZ2FsYXh5CiAgICBjb3N0cyB0aW1lIGFuZCByZXR1cm5zIG5vdGhpbmcuCiAgICAiIiIKICAgIHhfb2Zmc2V0ID0geV9vZmZzZXQgPSAwCiAgICB3b3JrLCB3b3JrX3JhdyA9IGRhdGEsIHJhdwogICAgaWYgd2luZG93IGlzIG5vdCBOb25lOgogICAgICAgIHkwLCB5MSwgeDAsIHgxID0gd2luZG93CiAgICAgICAgd29yaywgd29ya19yYXcgPSBkYXRhW3kwOnkxLCB4MDp4MV0sIHJhd1t5MDp5MSwgeDA6eDFdCiAgICAgICAgeF9vZmZzZXQsIHlfb2Zmc2V0ID0geDAsIHkwCiAgICAgICAgcHJpbnQoZiIgICB7YmFuZH06IHdvcmtpbmcgb24ge3gxIC0geDB9IHgge3kxIC0geTB9IHBpeGVscyBhcm91bmQgdGhlICIKICAgICAgICAgICAgICBmImdhbGF4eSIsIGZsdXNoPVRydWUpCgogICAgcHJpbnQoZiJGaW5kaW5nIHNvdXJjZXMgaW4gdGhlIHtiYW5kfSBpbWFnZS4uLiIsIGZsdXNoPVRydWUpCiAgICBia2cgPSBsb2NhbF9iYWNrZ3JvdW5kKHdvcmspCiAgICBybXMgPSBmbG9hdChucC5tZWRpYW4oYmtnLmJhY2tncm91bmRfcm1zKSkKICAgIHByaW50KGYiICAge2JhbmR9OiBsb2NhbCBza3kgcnVucyBmcm9tIHtia2cuYmFja2dyb3VuZC5taW4oKTouMWZ9IgogICAgICAgICAgZiIgdG8ge2JrZy5iYWNrZ3JvdW5kLm1heCgpOi4xZn0gY291bnRzLCBub2lzZSB7cm1zOi4yZn0iLCBmbHVzaD1UcnVlKQoKICAgIGZ3aG0gPSBEQU9GSU5EX0ZXSE0gaWYgREFPRklORF9GV0hNIGVsc2UgZXN0aW1hdGVfc2VlaW5nKAogICAgICAgIHdvcmssIGJrZy5iYWNrZ3JvdW5kLCBybXMsIGJhbmQsIHdvcmtfcmF3LCBzYXRfbGV2ZWwpCiAgICBTRUVJTkdfTUVBU1VSRURbYmFuZF0gPSBmd2htCgogICAgc291cmNlcyA9IERBT1N0YXJGaW5kZXIoZndobT1md2htLCB0aHJlc2hvbGQ9REVURUNUSU9OX1RIUkVTSE9MRF9TSUdNQSAqIHJtcykoCiAgICAgICAgd29yayAtIGJrZy5iYWNrZ3JvdW5kKQogICAgaWYgc291cmNlcyBpcyBOb25lIG9yIGxlbihzb3VyY2VzKSA9PSAwOgogICAgICAgIHByaW50KGYiICAge2JhbmR9OiBubyBzb3VyY2VzIGZvdW5kIiwgZmx1c2g9VHJ1ZSkKICAgICAgICByZXR1cm4gbnAuZW1wdHkoMCksIG5wLmVtcHR5KDApLCBucC5lbXB0eSgwKSwgZndobQogICAgc291cmNlcyA9IHNvdXJjZXNbc291cmNlc1sncGVhayddID4gUEVBS19NSU5fU1REICogcm1zXQogICAgeCwgeSA9IF94eV9vZihzb3VyY2VzKQogICAgZiA9IG5wLmFzYXJyYXkoc291cmNlc1siZmx1eCJdLCBmbG9hdCkKICAgIG9yZGVyID0gbnAuYXJnc29ydChmKVs6Oi0xXQogICAgeCwgeSwgZiA9IHhbb3JkZXJdICsgeF9vZmZzZXQsIHlbb3JkZXJdICsgeV9vZmZzZXQsIGZbb3JkZXJdCgogICAgIyBPbmUgc291cmNlIGZvdW5kIHR3byBvciB0aHJlZSB0aW1lcyBvdmVyIC0gYSBzYXR1cmF0ZWQgc3RhciB3aXRoIGEgZmxhdAogICAgIyB0b3AsIG9yIGEga25vdCB3aXRoIHR3byBwZWFrcyBpbnNpZGUgb25lIHNlZWluZyBkaXNjIC0gd291bGQgb3RoZXJ3aXNlIGJlCiAgICAjIG1lYXN1cmVkIGFzIHNldmVyYWwgc291cmNlcyBzaGFyaW5nIG9uZSBhcGVydHVyZS4gVHdvIHNvdXJjZXMgY2xvc2VyIHRoYW4KICAgICMgb25lIEZXSE0gY2Fubm90IGJlIG1lYXN1cmVkIGFwYXJ0LCBzbyBvbmx5IHRoZSBicmlnaHRlc3QgaXMga2VwdC4KICAgIGtlZXAgPSBucC5vbmVzKGxlbih4KSwgZHR5cGU9Ym9vbCkKICAgIHRyZWUgPSBjS0RUcmVlKG5wLmNvbHVtbl9zdGFjayhbeCwgeV0pKQogICAgZm9yIGkgaW4gcmFuZ2UobGVuKHgpKToKICAgICAgICBpZiBrZWVwW2ldOgogICAgICAgICAgICBmb3IgaiBpbiB0cmVlLnF1ZXJ5X2JhbGxfcG9pbnQoW3hbaV0sIHlbaV1dLCByPWZ3aG0pOgogICAgICAgICAgICAgICAgaWYgaiA+IGk6CiAgICAgICAgICAgICAgICAgICAga2VlcFtqXSA9IEZhbHNlCiAgICBwcmludChmIiAgIHtiYW5kfToge2xlbih4KTosfSBkZXRlY3Rpb25zLCB7aW50KCh+a2VlcCkuc3VtKCkpOix9IG9mIHRoZW0gIgogICAgICAgICAgZiJ3aXRoaW4gb25lIEZXSE0gb2YgYSBicmlnaHRlciBvbmUgYW5kIGRyb3BwZWQiLCBmbHVzaD1UcnVlKQogICAgcmV0dXJuIHhba2VlcF0sIHlba2VlcF0sIGZba2VlcF0sIGZ3aG0KCgpkZWYgcmVnaXN0ZXJfQl9vbnRvX1YoQl9pbWcsIFZfaW1nLCBmaXRzX2ZpbGVfQiwgZml0c19maWxlX1YpOgogICAgIiIiVGhlIG1hcHBpbmcgYmV0d2VlbiBCJ3MgcGl4ZWwgZ3JpZCBhbmQgVidzLCBib3RoIHdheXMuCgogICAgQiBhbmQgViBhcmUgc2VwYXJhdGUgZXhwb3N1cmVzIGFuZCBub3RoaW5nIGd1YXJhbnRlZXMgdGhlaXIgZ3JpZHMKICAgIGNvaW5jaWRlLiBhc3Ryb2FsaWduIGZpbmRzIHRoZSByZWdpc3RyYXRpb24gZnJvbSB0aGUgc3RhciBwYXR0ZXJucyBvZgogICAgdGhlc2UgdHdvIGltYWdlczsgZWFjaCBmcmFtZSdzIG93biBwbGF0ZSBzb2x1dGlvbiBpcyB0aGUgZmFsbGJhY2suCgogICAgUmV0dXJucyAoYl90b192LCB2X3RvX2IsIGRlc2NyaXB0aW9uKTsgZWFjaCB0YWtlcyBhbiAoTiwgMikgYXJyYXkuCiAgICAiIiIKICAgIHRyeToKICAgICAgICB0cmFuc2YsIChzcmNfbGlzdCwgX3RndF9saXN0KSA9IGFhLmZpbmRfdHJhbnNmb3JtKAogICAgICAgICAgICBucC5uYW5fdG9fbnVtKEJfaW1nKSwgbnAubmFuX3RvX251bShWX2ltZykpCiAgICAgICAgaWYgYWJzKHRyYW5zZi5zY2FsZSAtIDEuMCkgPiBBU1RST0FMSUdOX01BWF9TQ0FMRV9FUlJPUjoKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmImZpdHRlZCBzY2FsZSB7dHJhbnNmLnNjYWxlOi4zZn0gaXMgbm90IHBsYXVzaWJsZSAiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZiJmb3IgdHdvIGZyYW1lcyBmcm9tIHRoZSBzYW1lIHRlbGVzY29wZSIpCiAgICAgICAgdGV4dCA9IChmImZyb20ge2xlbihzcmNfbGlzdCl9IG1hdGNoZWQgc3RhciBwYWlycyAoYXN0cm9hbGlnbik6IHNoaWZ0ICIKICAgICAgICAgICAgICAgIGYiKHt0cmFuc2YudHJhbnNsYXRpb25bMF06Ky4xZn0sIHt0cmFuc2YudHJhbnNsYXRpb25bMV06Ky4xZn0pIHB4LCAiCiAgICAgICAgICAgICAgICBmInJvdGF0aW9uIHtucC5kZWdyZWVzKHRyYW5zZi5yb3RhdGlvbik6Ky4yZn0gZGVnLCBzY2FsZSAiCiAgICAgICAgICAgICAgICBmInt0cmFuc2Yuc2NhbGU6LjRmfSIpCiAgICAgICAgcHJpbnQoIiAgIHJlZ2lzdGVyZWQgQiBvbnRvIFYgIiArIHRleHQsIGZsdXNoPVRydWUpCiAgICAgICAgaW52ZXJzZSA9IHRyYW5zZi5pbnZlcnNlCiAgICAgICAgcmV0dXJuIChsYW1iZGEgeHk6IHRyYW5zZihucC5hc2FycmF5KHh5LCBmbG9hdCkpLAogICAgICAgICAgICAgICAgbGFtYmRhIHh5OiBpbnZlcnNlKG5wLmFzYXJyYXkoeHksIGZsb2F0KSksIHRleHQpCiAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICAgICAgcHJpbnQoZiIgICBhc3Ryb2FsaWduIHJlZ2lzdHJhdGlvbiBmYWlsZWQgKHtlfSkgLSAiCiAgICAgICAgICAgICAgZiJmYWxsaW5nIGJhY2sgdG8gZWFjaCBmcmFtZSdzIG93biBXQ1MiLCBmbHVzaD1UcnVlKQogICAgICAgIHdjc19CID0gV0NTKGZpdHMuZ2V0aGVhZGVyKGZpdHNfZmlsZV9CKSkuY2VsZXN0aWFsCiAgICAgICAgd2NzX1YgPSBXQ1MoZml0cy5nZXRoZWFkZXIoZml0c19maWxlX1YpKS5jZWxlc3RpYWwKCiAgICAgICAgZGVmIGJfdG9fdih4eSk6CiAgICAgICAgICAgIHh5ID0gbnAuYXNhcnJheSh4eSwgZmxvYXQpCiAgICAgICAgICAgIHJhLCBkZWMgPSB3Y3NfQi5hbGxfcGl4MndvcmxkKHh5WzosIDBdLCB4eVs6LCAxXSwgMCkKICAgICAgICAgICAgcmV0dXJuIG5wLmNvbHVtbl9zdGFjayh3Y3NfVi5hbGxfd29ybGQycGl4KHJhLCBkZWMsIDApKQoKICAgICAgICBkZWYgdl90b19iKHh5KToKICAgICAgICAgICAgeHkgPSBucC5hc2FycmF5KHh5LCBmbG9hdCkKICAgICAgICAgICAgcmEsIGRlYyA9IHdjc19WLmFsbF9waXgyd29ybGQoeHlbOiwgMF0sIHh5WzosIDFdLCAwKQogICAgICAgICAgICByZXR1cm4gbnAuY29sdW1uX3N0YWNrKHdjc19CLmFsbF93b3JsZDJwaXgocmEsIGRlYywgMCkpCiAgICAgICAgcmV0dXJuIGJfdG9fdiwgdl90b19iLCAidGhyb3VnaCB0aGUgdHdvIHBsYXRlIHNvbHV0aW9ucyIKCgpkZWYgYXBlcnR1cmVfcGhvdG9tZXRyeV9hdChkYXRhLCByYXcsIHh5LCByX2FwLCByX2luLCByX291dCwgZ2Fpbiwgcm4pOgogICAgIiIiRmx1eCwgaXRzIGVycm9yLCB0aGUgYnJpZ2h0ZXN0IHJhdyBwaXhlbCwgYW5kIHRoZSBsb2NhbCBza3ksIGZvciBldmVyeQogICAgcG9zaXRpb24gaW4geHkgLSBhbGwgdGhyb3VnaCBvbmUgYXBlcnR1cmUgcmFkaXVzLgoKICAgIFRoZSBza3kgaXMgdGhlIHNpZ21hLWNsaXBwZWQgbWVkaWFuIG9mIGEgcmluZyBhcm91bmQgdGhlIHNvdXJjZSwgc28gdGhlCiAgICBzbW9vdGggbGlnaHQgb2YgdGhlIGdhbGF4eSB1bmRlciBhIGtub3QgaXMgdGFrZW4gb2ZmIHdpdGggaXQuIFRoZSBlcnJvcgogICAgY291bnRzIHRoZSBwaG90b25zIG9mIHRoZSBzb3VyY2UsIHRoZSBza3kgbm9pc2UgbWVhc3VyZWQgaW4gdGhlIHJpbmcgKG9uY2UKICAgIHBlciBhcGVydHVyZSBwaXhlbCwgYW5kIG9uY2UgbW9yZSB0aHJvdWdoIHRoZSB1bmNlcnRhaW50eSBvZiB0aGUgc2t5IGxldmVsCiAgICBpdHNlbGYpLCB3aXRoIHRoZSBoZWFkZXIncyByZWFkIG5vaXNlIGFzIGEgZmxvb3IuCiAgICAiIiIKICAgIHh5ID0gbnAuYXNhcnJheSh4eSwgZmxvYXQpCiAgICBhcCA9IENpcmN1bGFyQXBlcnR1cmUoeHksIHI9cl9hcCkKICAgIGFubiA9IENpcmN1bGFyQW5udWx1cyh4eSwgcl9pbj1yX2luLCByX291dD1yX291dCkKICAgIHNreSA9IEFwZXJ0dXJlU3RhdHMoZGF0YSwgYW5uLCBzaWdtYV9jbGlwPVNpZ21hQ2xpcChzaWdtYT0zLjAsIG1heGl0ZXJzPTUpKQogICAgc2t5X21lZCA9IG5wLmF0bGVhc3RfMWQobnAuYXNhcnJheShza3kubWVkaWFuLCBmbG9hdCkpCiAgICBza3lfc3RkID0gbnAuYXRsZWFzdF8xZChucC5hc2FycmF5KHNreS5zdGQsIGZsb2F0KSkKICAgIG5fc2t5ID0gbnAuYXRsZWFzdF8xZChucC5hc2FycmF5KHNreS5zdW1fYXBlcl9hcmVhLnZhbHVlLCBmbG9hdCkpCiAgICB0b3RhbCA9IG5wLmFzYXJyYXkoYXBlcnR1cmVfcGhvdG9tZXRyeShkYXRhLCBhcClbImFwZXJ0dXJlX3N1bSJdLCBmbG9hdCkKICAgIG5fYXAgPSBmbG9hdChhcC5hcmVhKQogICAgZmx1eCA9IHRvdGFsIC0gc2t5X21lZCAqIG5fYXAKICAgIHBpeF92YXIgPSBucC5tYXhpbXVtKHNreV9zdGQgKiogMiwgKHJuIC8gZ2FpbikgKiogMikKICAgIHZhciA9IChucC5jbGlwKGZsdXgsIDAsIE5vbmUpIC8gZ2FpbgogICAgICAgICAgICsgbl9hcCAqIHBpeF92YXIgKiAoMS4wICsgbl9hcCAvIG5wLm1heGltdW0obl9za3ksIDEpKSkKICAgIHBlYWsgPSBucC5hdGxlYXN0XzFkKG5wLmFzYXJyYXkoCiAgICAgICAgQXBlcnR1cmVTdGF0cyhyYXcsIENpcmN1bGFyQXBlcnR1cmUoeHksIHI9bWF4KDEuNSwgMC41ICogcl9hcCkpKS5tYXgsIGZsb2F0KSkKICAgIHJldHVybiBmbHV4LCBucC5zcXJ0KHZhciksIHBlYWssIHNreV9tZWQKCgpkZWYgaW5zdHJ1bWVudGFsKGZsdXgsIGVycik6CiAgICB3aXRoIG5wLmVycnN0YXRlKGRpdmlkZT0iaWdub3JlIiwgaW52YWxpZD0iaWdub3JlIik6CiAgICAgICAgbWFnID0gbnAud2hlcmUoZmx1eCA+IDAsIC0yLjUgKiBucC5sb2cxMChmbHV4KSwgbnAubmFuKQogICAgICAgIG1hZ19lcnIgPSBucC53aGVyZShmbHV4ID4gMCwgMS4wODU3ICogZXJyIC8gZmx1eCwgbnAubmFuKQogICAgcmV0dXJuIG1hZywgbWFnX2VycgoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLSBUaGUgcGxhdGUgc29sdXRpb24sIGNoZWNrZWQgYWdhaW5zdCBHYWlhIC0tLS0tLS0tLS0tLS0tLS0tLS0KCmRlZiBmaWVsZF9jZW50cmVfYW5kX3JhZGl1cyh3Y3MsIHNoYXBlKToKICAgIG55LCBueCA9IHNoYXBlCiAgICBjZW50cmUgPSB3Y3MucGl4ZWxfdG9fd29ybGQobnggLyAyLCBueSAvIDIpCiAgICBjb3JuZXJzID0gd2NzLnBpeGVsX3RvX3dvcmxkKG5wLmFycmF5KFswLCBueCwgMCwgbnhdKSwgbnAuYXJyYXkoWzAsIDAsIG55LCBueV0pKQogICAgcmV0dXJuIGNlbnRyZSwgZmxvYXQobnAubWF4KGNlbnRyZS5zZXBhcmF0aW9uKGNvcm5lcnMpLmRlZykpCgoKZGVmIHZlcmlmeV9wbGF0ZV9zb2x1dGlvbih4eV9icmlnaHQsIGFsbF94eSwgd2NzLCBzaGFwZSk6CiAgICAiIiJUaGUgcGxhdGUgc29sdXRpb24gb2YgdGhlIFYgZnJhbWUsIGNoZWNrZWQgYWdhaW5zdCBHYWlhIGFuZCByZXBhaXJlZCBpZgogICAgaXQgZG9lcyBub3QgZml0LgoKICAgIEV2ZXJ5dGhpbmcgdGhhdCB1c2VzIHRoZSBza3kgZ29lcyB0aHJvdWdoIGl0OiB0aGUgY2FsaWJyYXRpb24gc3RhcnMsIHRoZQogICAgZm9yZWdyb3VuZCBzdGFycywgdGhlIGdhbGF4eSdzIGNlbnRyZS4gU29mdHdhcmUgdGhhdCBjb21iaW5lcyBvciBjcm9wcwogICAgZnJhbWVzIHNvbWV0aW1lcyBrZWVwcyBhIHNvbHV0aW9uIHRoYXQgYmVsb25ncyB0byBhbm90aGVyIGZyYW1lIC0gaW4gdGhlCiAgICBDbHVzdGVyX0NNRCB0ZXN0cyBhIHNwbGl0LWNoYW5uZWwgZmlsZSBjYXJyaWVkIG9uZSA0NSBweCBvZmYsIGFuZCBvbmx5IDQgb2YKICAgIDIsMDAwIGNhdGFsb2d1ZSBzdGFycyBtYXRjaGVkLgoKICAgIFRoZSAxMDAgYnJpZ2h0ZXN0IHVuc2F0dXJhdGVkIHNvdXJjZXMgYXJlIHB1dCB0aHJvdWdoIHRoZSBzb2x1dGlvbiBhbmQKICAgIGxvb2tlZCB1cCBpbiBHYWlhIERSMy4gSWYgYXQgbGVhc3QgaGFsZiBoYXZlIGEgR2FpYSBzdGFyIHdpdGhpbgogICAgQ0FMSUJfTUFUQ0hfUkFESVVTIHRoZSBzb2x1dGlvbiBpcyBrZXB0LiBPdGhlcndpc2UgdGhlIG9mZnNldCBiZXR3ZWVuIHRoZQogICAgR2FpYSBzdGFycyBhbmQgdGhlIHNvdXJjZXMgaXMgZm91bmQgKG9yIHRoZSBzdGFyIHBhdHRlcm5zIG1hdGNoZWQsIHdpdGgKICAgIHRoZSBtaXJyb3IgaW1hZ2UgdHJpZWQgdG9vKSBhbmQgYSBuZXcgc29sdXRpb24gZml0dGVkLgogICAgIiIiCiAgICBmcm9tIGFzdHJvcHkud2NzLnV0aWxzIGltcG9ydCBmaXRfd2NzX2Zyb21fcG9pbnRzCiAgICBueSwgbnggPSBzaGFwZQogICAgY2VudHJlLCByYWRpdXMgPSBmaWVsZF9jZW50cmVfYW5kX3JhZGl1cyh3Y3MsIHNoYXBlKQogICAgdHJ5OgogICAgICAgIHJlcyA9IFZpemllcihjb2x1bW5zPVsiUkFfSUNSUyIsICJERV9JQ1JTIiwgIkdtYWciXSwKICAgICAgICAgICAgICAgICAgICAgY29sdW1uX2ZpbHRlcnM9eyJHbWFnIjogIjwxNyJ9LCByb3dfbGltaXQ9LTEpLnF1ZXJ5X3JlZ2lvbigKICAgICAgICAgICAgU2t5Q29vcmQoY2VudHJlLnJhLCBjZW50cmUuZGVjKSwgcmFkaXVzPTEuMyAqIHJhZGl1cyAqIHUuZGVnLAogICAgICAgICAgICBjYXRhbG9nPSJJLzM1NS9nYWlhZHIzIikKICAgICAgICBnYWlhID0gcmVzWzBdCiAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGV4YzoKICAgICAgICBwcmludChmIlt3Y3NdIEdhaWEgdW5hdmFpbGFibGUgKHtleGN9KSAtIHRoZSBwbGF0ZSBzb2x1dGlvbiBpcyBub3QgY2hlY2tlZCIpCiAgICAgICAgcmV0dXJuIHdjcywgIm5vdCBjaGVja2VkIChHYWlhIHVuYXZhaWxhYmxlKSIKICAgIGdhaWEuc29ydCgiR21hZyIpCiAgICBnc2t5ID0gU2t5Q29vcmQobnAuYXNhcnJheShnYWlhWyJSQV9JQ1JTIl0sIGZsb2F0KSAqIHUuZGVnLAogICAgICAgICAgICAgICAgICAgIG5wLmFzYXJyYXkoZ2FpYVsiREVfSUNSUyJdLCBmbG9hdCkgKiB1LmRlZykKCiAgICBkZWYgZnJhY3Rpb24odywgbj0xMDApOgogICAgICAgIHJhLCBkZWMgPSB3LmFsbF9waXgyd29ybGQoeHlfYnJpZ2h0WzpuLCAwXSwgeHlfYnJpZ2h0WzpuLCAxXSwgMCkKICAgICAgICBfaSwgc2VwLCBfID0gU2t5Q29vcmQocmEgKiB1LmRlZywgZGVjICogdS5kZWcpLm1hdGNoX3RvX2NhdGFsb2dfc2t5KGdza3kpCiAgICAgICAgb2sgPSBzZXAuYXJjc2VjIDwgQ0FMSUJfTUFUQ0hfUkFESVVTCiAgICAgICAgcmV0dXJuIG9rLm1lYW4oKSwgKG5wLm1lZGlhbihzZXAuYXJjc2VjW29rXSkgaWYgb2suYW55KCkgZWxzZSBucC5uYW4pCgogICAgZjAsIHMwID0gZnJhY3Rpb24od2NzKQogICAgcHJpbnQoZiJbd2NzXSB7MTAwICogZjA6LjBmfSUgb2YgdGhlIDEwMCBicmlnaHRlc3Qgc291cmNlcyBoYXZlIGEgR2FpYSBzdGFyICIKICAgICAgICAgIGYid2l0aGluIHtDQUxJQl9NQVRDSF9SQURJVVM6Z30gYXJjc2VjIChtZWRpYW4ge3MwOi4yZn0gYXJjc2VjKSIsIGZsdXNoPVRydWUpCiAgICBpZiBmMCA+PSBXQ1NfQ0hFQ0tfTUlOX0ZSQUNUSU9OOgogICAgICAgIHJldHVybiB3Y3MsIGYiY2hlY2tlZCBhZ2FpbnN0IEdhaWE6IHsxMDAgKiBmMDouMGZ9JSBvZiBicmlnaHQgc291cmNlcyBtYXRjaCIKCiAgICBwcmludCgiW3djc10gdGhlIHBsYXRlIHNvbHV0aW9uIGluIHRoZSBoZWFkZXIgZG9lcyBub3QgZml0IHRoaXMgaW1hZ2UgLSAiCiAgICAgICAgICAicmVwYWlyaW5nIGl0IGZyb20gdGhlIHN0YXIgcGF0dGVybiIsIGZsdXNoPVRydWUpCiAgICBneCwgZ3kgPSB3Y3MuYWxsX3dvcmxkMnBpeChnc2t5LnJhLmRlZywgZ3NreS5kZWMuZGVnLCAwKQogICAgbmVhciA9IChneCA+IC0wLjA1ICogbngpICYgKGd4IDwgMS4wNSAqIG54KSAmIChneSA+IC0wLjA1ICogbnkpICYgKGd5IDwgMS4wNSAqIG55KQogICAgc3JjX2FsbCA9IG5wLmNvbHVtbl9zdGFjayhbZ3gsIGd5XSkKICAgIGRldF90cmVlID0gY0tEVHJlZSh4eV9icmlnaHRbOjYwMF0pCiAgICBtYXhfb2ZmID0gMC4yNSAqIG1heChueCwgbnkpCiAgICBiZXN0ID0gTm9uZQogICAgZm9yIHBhcml0eSBpbiAoMSwgLTEpOgogICAgICAgIHNyYyA9IHNyY19hbGwuY29weSgpCiAgICAgICAgaWYgcGFyaXR5IDwgMDoKICAgICAgICAgICAgc3JjWzosIDBdID0gKG54IC0gMSkgLSBzcmNbOiwgMF0KICAgICAgICBjYW5kID0gc3JjW25lYXJdWzo2MDBdCiAgICAgICAgZHh5ID0gW10KICAgICAgICBmb3IgaywgbmIgaW4gZW51bWVyYXRlKGRldF90cmVlLnF1ZXJ5X2JhbGxfcG9pbnQoY2FuZCwgcj1tYXhfb2ZmKSk6CiAgICAgICAgICAgIGlmIG5iOgogICAgICAgICAgICAgICAgZHh5LmFwcGVuZCh4eV9icmlnaHRbOjYwMF1bbmJdIC0gY2FuZFtrXSkKICAgICAgICBpZiBkeHk6CiAgICAgICAgICAgIGR4eSA9IG5wLnZzdGFjayhkeHkpCiAgICAgICAgICAgIGJpbnMgPSBucC5hcmFuZ2UoLW1heF9vZmYsIG1heF9vZmYgKyA0LCA0LjApCiAgICAgICAgICAgIEgsIGV4LCBleSA9IG5wLmhpc3RvZ3JhbTJkKGR4eVs6LCAwXSwgZHh5WzosIDFdLCBiaW5zPVtiaW5zLCBiaW5zXSkKICAgICAgICAgICAgaSwgaiA9IG5wLnVucmF2ZWxfaW5kZXgobnAuYXJnbWF4KEgpLCBILnNoYXBlKQogICAgICAgICAgICBwZWFrID0gSFtpLCBqXQogICAgICAgICAgICBpZiBwZWFrID49IDEwIGFuZCBwZWFrID4gOCAqIG1heChucC5tZWRpYW4oSFtIID4gMF0pLCAxKToKICAgICAgICAgICAgICAgIHNoaWZ0ID0gbnAuYXJyYXkoWzAuNSAqIChleFtpXSArIGV4W2kgKyAxXSksIDAuNSAqIChleVtqXSArIGV5W2ogKyAxXSldKQogICAgICAgICAgICAgICAgVCA9IGxhbWJkYSBwLCBzXz1zaGlmdDogcCArIHNfCiAgICAgICAgICAgICAgICBpZiBiZXN0IGlzIE5vbmUgb3IgcGVhayA+IGJlc3RbMl06CiAgICAgICAgICAgICAgICAgICAgYmVzdCA9IChwYXJpdHksIFQsIGludChwZWFrKSkKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgZm9yIG5fY2F0LCBuX2RldCBpbiAoKDE1MCwgMTUwKSwgKDIwMCwgMjAwKSwgKDMwMCwgMzAwKSk6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIFQsIChzLCBfdCkgPSBhYS5maW5kX3RyYW5zZm9ybShjYW5kWzpuX2NhdF0sIHh5X2JyaWdodFs6bl9kZXRdKQogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgaWYgYWJzKFQuc2NhbGUgLSAxLjApID4gMC4xOgogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgaWYgYmVzdCBpcyBOb25lIG9yIGxlbihzKSA+IGJlc3RbMl06CiAgICAgICAgICAgICAgICBiZXN0ID0gKHBhcml0eSwgVCwgbGVuKHMpKQogICAgICAgICAgICBicmVhawogICAgaWYgYmVzdCBpcyBOb25lOgogICAgICAgIHN0b3AoIlBMQVRFIFNPTFVUSU9OOiB0aGUgaGVhZGVyJ3Mgc29sdXRpb24gZG9lcyBub3QgZml0IHRoZSBWIGltYWdlIGFuZCAiCiAgICAgICAgICAgICAiY291bGQgbm90IGJlIHJlcGFpcmVkLiBQbGF0ZS1zb2x2ZSB0aGUgViBmcmFtZSBhZ2FpbiAoZm9yIGV4YW1wbGUgIgogICAgICAgICAgICAgIndpdGggYXN0cm9tZXRyeS5uZXQpIGFuZCByZXJ1bi4iKQogICAgcGFyaXR5LCBULCBuY3RsID0gYmVzdAogICAgc3JjID0gc3JjX2FsbC5jb3B5KCkKICAgIGlmIHBhcml0eSA8IDA6CiAgICAgICAgc3JjWzosIDBdID0gKG54IC0gMSkgLSBzcmNbOiwgMF0KICAgIGd1ZXNzID0gVChzcmMpCiAgICB0cmVlID0gY0tEVHJlZShhbGxfeHkpCiAgICBuZXcgPSB3Y3MKICAgIHRvbCA9IDUuMAogICAgb2sgPSBucC56ZXJvcyhsZW4oZ3Vlc3MpLCBib29sKQogICAgZm9yIF8gaW4gcmFuZ2UoMyk6CiAgICAgICAgZCwgaiA9IHRyZWUucXVlcnkoZ3Vlc3MsIGRpc3RhbmNlX3VwcGVyX2JvdW5kPXRvbCkKICAgICAgICBvayA9IG5wLmlzZmluaXRlKGQpCiAgICAgICAgaWYgb2suc3VtKCkgPCAxMDoKICAgICAgICAgICAgYnJlYWsKICAgICAgICBweCA9IGFsbF94eVtqW29rXV0KICAgICAgICBuZXcgPSBmaXRfd2NzX2Zyb21fcG9pbnRzKChweFs6LCAwXSwgcHhbOiwgMV0pLCBnc2t5W29rXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHByb2pfcG9pbnQ9ImNlbnRlciIsIHByb2plY3Rpb249IlRBTiIpCiAgICAgICAgZ3gyLCBneTIgPSBuZXcuYWxsX3dvcmxkMnBpeChnc2t5LnJhLmRlZywgZ3NreS5kZWMuZGVnLCAwKQogICAgICAgIGd1ZXNzID0gbnAuY29sdW1uX3N0YWNrKFtneDIsIGd5Ml0pCiAgICAgICAgdG9sID0gWFlfTUFUQ0hfVE9MRVJBTkNFCiAgICBmMSwgczEgPSBmcmFjdGlvbihuZXcpCiAgICBzaGlmdCA9IG5wLmh5cG90KCoobnAuYXJyYXkobmV3LmFsbF93b3JsZDJwaXgoY2VudHJlLnJhLmRlZywgY2VudHJlLmRlYy5kZWcsIDApKQogICAgICAgICAgICAgICAgICAgICAgIC0gbnAuYXJyYXkoW254IC8gMiwgbnkgLyAyXSkpKQogICAgcHJpbnQoZiJbd2NzXSByZXBhaXJlZCAoeydtaXJyb3JlZCwgJyBpZiBwYXJpdHkgPCAwIGVsc2UgJyd9e25jdGx9IHBhdHRlcm4gIgogICAgICAgICAgZiJzdGFycywge2ludChvay5zdW0oKSl9IGluIHRoZSBmaXQpOiB0aGUgZmllbGQgY2VudHJlIG1vdmVkIHtzaGlmdDouMGZ9IHB4OyAiCiAgICAgICAgICBmIm5vdyB7MTAwICogZjE6LjBmfSUgb2YgYnJpZ2h0IHNvdXJjZXMgbWF0Y2ggR2FpYSAobWVkaWFuIHtzMTouMmZ9IGFyY3NlYykiLAogICAgICAgICAgZmx1c2g9VHJ1ZSkKICAgIGlmIGYxIDwgV0NTX0NIRUNLX01JTl9GUkFDVElPTjoKICAgICAgICBzdG9wKCJQTEFURSBTT0xVVElPTjogdGhlIHJlcGFpcmVkIHNvbHV0aW9uIHN0aWxsIGRvZXMgbm90IGZpdC4gIgogICAgICAgICAgICAgIlBsYXRlLXNvbHZlIHRoZSBWIGZyYW1lIGFnYWluIGFuZCByZXJ1bi4iKQogICAgcmV0dXJuIG5ldywgKGYiUkVQQUlSRUQgZnJvbSBHYWlhOiBoZWFkZXIgc29sdXRpb24gbWF0Y2hlZCB7MTAwICogZjA6LjBmfSUgb2YgIgogICAgICAgICAgICAgICAgIGYiYnJpZ2h0IHNvdXJjZXMsIHJlcGFpcmVkIHsxMDAgKiBmMTouMGZ9JSwgY2VudHJlIG1vdmVkIHtzaGlmdDouMGZ9IHB4IikKCgojIC0tLS0tLS0tLS0tLS0tLS0tLS0gQ2FsaWJyYXRpb24gLS0tLS0tLS0tLS0tLS0tLS0tLQoKZGVmIHF1ZXJ5X2NhbGlicmF0aW9uX2NhdGFsb2d1ZSh3aGljaCwgY2VudHJlLCByYWRpdXNfZGVnKToKICAgICIiIlJlZmVyZW5jZSBzdGFycyB3aXRoIEpvaG5zb24gQiBhbmQgVjogUkEsIERFQywgTWFnX0IsIE1hZ19WLCBlX0IsIGVfVi4KCiAgICBBUEFTUyBEUjkgKElJLzMzNi9hcGFzczkpIGlzIHRoZSBkZWZhdWx0LiBHYWlhIERSMyBzeW50aGV0aWMgcGhvdG9tZXRyeQogICAgKEkvMzYwL3N5bnRwaG90KSBpcyBjb21wdXRlZCBmcm9tIEdhaWEncyBvd24gbG93LXJlc29sdXRpb24gc3BlY3RyYSBhbmQKICAgIHN0YW5kYXJkaXNlZCB0byB0aGUgSm9obnNvbi1Lcm9uLUNvdXNpbnMgc3lzdGVtOiBkZWVwZXIsIHVuaWZvcm0gYWNyb3NzCiAgICB0aGUgc2t5LCBhbmQgcmVzb2x2ZWQgZG93biB0byBhYm91dCBhbiBhcmNzZWNvbmQuIE9ubHkgZW50cmllcyBHYWlhIGZsYWdzCiAgICBhcyBnb29kIGFyZSB1c2VkLgoKICAgIHJvd19saW1pdCBoYXMgdG8gYmUgcGFzc2VkIHRvIHRoZSBjb25zdHJ1Y3Rvcjogc2V0IG9uIHRoZSBjbGFzcyBhZnRlcndhcmRzCiAgICBpdCBsZWZ0IGFzdHJvcXVlcnkncyBkZWZhdWx0IG9mIDUwIHJvd3MsIHNpbGVudGx5LgogICAgIiIiCiAgICBjb29yZCA9IFNreUNvb3JkKGNlbnRyZS5yYSwgY2VudHJlLmRlYykKICAgIGVtcHR5ID0gcGQuRGF0YUZyYW1lKGNvbHVtbnM9WyJSQSIsICJERUMiLCAiTWFnX0IiLCAiTWFnX1YiLCAiZV9CIiwgImVfViJdKQogICAgaWYgd2hpY2ggPT0gIkFQQVNTIjoKICAgICAgICBjb2xzID0gWyJSQUoyMDAwIiwgIkRFSjIwMDAiLCAiQm1hZyIsICJWbWFnIiwgImVfQm1hZyIsICJlX1ZtYWciXQogICAgICAgIHJlcyA9IFZpemllcihjb2x1bW5zPWNvbHMsIHJvd19saW1pdD0tMSkucXVlcnlfcmVnaW9uKAogICAgICAgICAgICBjb29yZCwgcmFkaXVzPXJhZGl1c19kZWcgKiB1LmRlZywgY2F0YWxvZz0iSUkvMzM2L2FwYXNzOSIpCiAgICAgICAgaWYgbGVuKHJlcykgPT0gMDoKICAgICAgICAgICAgcmV0dXJuIGVtcHR5CiAgICAgICAgdCA9IHJlc1swXS50b19wYW5kYXMoKQogICAgICAgIG91dCA9IHBkLkRhdGFGcmFtZSh7IlJBIjogdFsiUkFKMjAwMCJdLCAiREVDIjogdFsiREVKMjAwMCJdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIk1hZ19CIjogdFsiQm1hZyJdLCAiTWFnX1YiOiB0WyJWbWFnIl0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAiZV9CIjogdFsiZV9CbWFnIl0sICJlX1YiOiB0WyJlX1ZtYWciXX0pCiAgICBlbHNlOgogICAgICAgIGNvbHMgPSBbIlNvdXJjZSIsICJSQV9JQ1JTIiwgIkRFX0lDUlMiLCAiQm1hZyIsICJWbWFnIiwgIkZCIiwgImVfRkIiLAogICAgICAgICAgICAgICAgIkZWIiwgImVfRlYiLCAiQkZsYWciLCAiVkZsYWciXQogICAgICAgIHJlcyA9IFZpemllcihjb2x1bW5zPWNvbHMsIHJvd19saW1pdD0tMSkucXVlcnlfcmVnaW9uKAogICAgICAgICAgICBjb29yZCwgcmFkaXVzPXJhZGl1c19kZWcgKiB1LmRlZywgY2F0YWxvZz0iSS8zNjAvc3ludHBob3QiKQogICAgICAgIGlmIGxlbihyZXMpID09IDA6CiAgICAgICAgICAgIHJldHVybiBlbXB0eQogICAgICAgIHQgPSByZXNbMF0udG9fcGFuZGFzKCkKICAgICAgICB0ID0gdFsodFsiQkZsYWciXSA9PSAxKSAmICh0WyJWRmxhZyJdID09IDEpXQogICAgICAgIG91dCA9IHBkLkRhdGFGcmFtZSh7IlJBIjogdFsiUkFfSUNSUyJdLCAiREVDIjogdFsiREVfSUNSUyJdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIk1hZ19CIjogdFsiQm1hZyJdLCAiTWFnX1YiOiB0WyJWbWFnIl0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAiZV9CIjogMS4wODU3ICogdFsiZV9GQiJdIC8gdFsiRkIiXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICJlX1YiOiAxLjA4NTcgKiB0WyJlX0ZWIl0gLyB0WyJGViJdfSkKICAgIHJldHVybiBvdXQucmVzZXRfaW5kZXgoZHJvcD1UcnVlKQoKCmRlZiBtYXRjaF9jYXRhbG9ndWUoZGYsIHdjcywgY2F0LCBzaGFwZSk6CiAgICAiIiJFYWNoIGNhdGFsb2d1ZSBzdGFyIG1hdGNoZWQgdG8gdGhlIGNsb3Nlc3Qgc291cmNlLCBvbmNlLgoKICAgIEEgbWF0Y2ggbXVzdCBiZSB3aXRoaW4gWFlfTUFUQ0hfVE9MRVJBTkNFIG9uIHRoZSBWIGdyaWQgYW5kCiAgICBDQUxJQl9NQVRDSF9SQURJVVMgb24gdGhlIHNreS4gRWFjaCBzb3VyY2UgaXMgdXNlZCBmb3IgYXQgbW9zdCBvbmUKICAgIGNhdGFsb2d1ZSBzdGFyICh0aGUgY2xvc2VzdCkgLSBhIGJyaWdodCBzdGFyIGZvdW5kIHNldmVyYWwgdGltZXMgb3ZlcgogICAgdXNlZCB0byBiZSBjb3VudGVkIG9uY2UgcGVyIGRldGVjdGlvbiwgYW5kIG9uZSBzdGFyIHRoZW4gb3V0d2VpZ2hlZCB0aGUKICAgIHJlc3Qgb2YgdGhlIGNhbGlicmF0aW9uLgogICAgIiIiCiAgICBjYXQgPSBjYXRbbnAuaXNmaW5pdGUoY2F0WyJNYWdfQiJdKSAmIG5wLmlzZmluaXRlKGNhdFsiTWFnX1YiXSldLnJlc2V0X2luZGV4KGRyb3A9VHJ1ZSkKICAgIGlmIG5vdCBsZW4oY2F0KSBvciBub3QgbGVuKGRmKToKICAgICAgICByZXR1cm4gcGQuRGF0YUZyYW1lKCkKICAgIGN4LCBjeSA9IHdjcy5hbGxfd29ybGQycGl4KGNhdFsiUkEiXS52YWx1ZXMsIGNhdFsiREVDIl0udmFsdWVzLCAwKQogICAgbnksIG54ID0gc2hhcGUKICAgIGluc2lkZSA9IChjeCA+IDApICYgKGN4IDwgbngpICYgKGN5ID4gMCkgJiAoY3kgPCBueSkKICAgIGNhdCA9IGNhdFtpbnNpZGVdLnJlc2V0X2luZGV4KGRyb3A9VHJ1ZSkKICAgIGN4LCBjeSA9IGN4W2luc2lkZV0sIGN5W2luc2lkZV0KICAgIGlmIG5vdCBsZW4oY2F0KToKICAgICAgICByZXR1cm4gcGQuRGF0YUZyYW1lKCkKICAgIGRpc3QsIGlkeCA9IGNLRFRyZWUoZGZbWyJYX1YiLCAiWV9WIl1dLnZhbHVlcykucXVlcnkoCiAgICAgICAgbnAuY29sdW1uX3N0YWNrKFtjeCwgY3ldKSwgZGlzdGFuY2VfdXBwZXJfYm91bmQ9WFlfTUFUQ0hfVE9MRVJBTkNFKQogICAgYmVzdCA9IHt9CiAgICBmb3IgaSBpbiByYW5nZShsZW4oY2F0KSk6CiAgICAgICAgaWYgbm90IG5wLmlzZmluaXRlKGRpc3RbaV0pOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGogPSBpbnQoaWR4W2ldKQogICAgICAgIGlmIGogbm90IGluIGJlc3Qgb3IgZGlzdFtpXSA8IGJlc3Rbal1bMV06CiAgICAgICAgICAgIGJlc3Rbal0gPSAoaSwgZGlzdFtpXSkKICAgIHJvd3MgPSBbXQogICAgZm9yIGosIChpLCBkKSBpbiBiZXN0Lml0ZW1zKCk6CiAgICAgICAgc3JjID0gZGYuaWxvY1tqXQogICAgICAgIHNlcCA9IFNreUNvb3JkKHNyY1siUkEiXSAqIHUuZGVnLCBzcmNbIkRFQyJdICogdS5kZWcpLnNlcGFyYXRpb24oCiAgICAgICAgICAgIFNreUNvb3JkKGNhdC5sb2NbaSwgIlJBIl0gKiB1LmRlZywgY2F0LmxvY1tpLCAiREVDIl0gKiB1LmRlZykpLmFyY3NlYwogICAgICAgIGlmIHNlcCA+IENBTElCX01BVENIX1JBRElVUzoKICAgICAgICAgICAgY29udGludWUKICAgICAgICByID0gc3JjLnRvX2RpY3QoKQogICAgICAgIHIudXBkYXRlKHsicm93IjogaiwgIkNhdF9SQSI6IGNhdC5sb2NbaSwgIlJBIl0sICJDYXRfREVDIjogY2F0LmxvY1tpLCAiREVDIl0sCiAgICAgICAgICAgICAgICAgICJDYXRfQiI6IGNhdC5sb2NbaSwgIk1hZ19CIl0sICJDYXRfViI6IGNhdC5sb2NbaSwgIk1hZ19WIl0sCiAgICAgICAgICAgICAgICAgICJDYXRfZV9CIjogY2F0LmxvY1tpLCAiZV9CIl0sICJDYXRfZV9WIjogY2F0LmxvY1tpLCAiZV9WIl0sCiAgICAgICAgICAgICAgICAgICJTZXBfYXJjc2VjIjogc2VwLCAiU2VwX3B4IjogZH0pCiAgICAgICAgcm93cy5hcHBlbmQocikKICAgIHJldHVybiBwZC5EYXRhRnJhbWUocm93cykKCgpkZWYgY2xpcHBlZF9saW5lKHgsIHksIG5zaWc9Q0FMSUJfQ0xJUF9TSUdNQSwgbWF4aXRlcj0xMCk6CiAgICAiIiJTdHJhaWdodC1saW5lIGZpdCB5ID0gaW50ZXJjZXB0ICsgc2xvcGUgeCwgd2l0aCBwb2ludHMgZnVydGhlciB0aGFuCiAgICBuc2lnIHJvYnVzdCBzaWdtYXMgZnJvbSB0aGUgbGluZSBsZWZ0IG91dCwgcmVwZWF0ZWQgdW50aWwgbm90aGluZyBjaGFuZ2VzLgogICAgUmV0dXJucyAoaW50ZXJjZXB0LCBzbG9wZSwgbWFza191c2VkLCBybXNfb2ZfdXNlZCkuIiIiCiAgICB4ID0gbnAuYXNhcnJheSh4LCBmbG9hdCkKICAgIHkgPSBucC5hc2FycmF5KHksIGZsb2F0KQogICAgdXNlID0gbnAuaXNmaW5pdGUoeCkgJiBucC5pc2Zpbml0ZSh5KQogICAgZm9yIF8gaW4gcmFuZ2UobWF4aXRlcik6CiAgICAgICAgaWYgdXNlLnN1bSgpIDwgMzoKICAgICAgICAgICAgYnJlYWsKICAgICAgICBzbG9wZSwgaW50ZXJjZXB0ID0gbnAucG9seWZpdCh4W3VzZV0sIHlbdXNlXSwgMSkKICAgICAgICByZXNpZCA9IHkgLSAoaW50ZXJjZXB0ICsgc2xvcGUgKiB4KQogICAgICAgIHNpZyA9IG1heCgxLjQ4MjYgKiBucC5tZWRpYW4obnAuYWJzKHJlc2lkW3VzZV0gLSBucC5tZWRpYW4ocmVzaWRbdXNlXSkpKSwgMWUtMykKICAgICAgICBuZXcgPSBucC5pc2Zpbml0ZShyZXNpZCkgJiAobnAuYWJzKHJlc2lkKSA8IG5zaWcgKiBzaWcpCiAgICAgICAgaWYgbnAuYXJyYXlfZXF1YWwobmV3LCB1c2UpOgogICAgICAgICAgICBicmVhawogICAgICAgIHVzZSA9IG5ldwogICAgc2xvcGUsIGludGVyY2VwdCA9IG5wLnBvbHlmaXQoeFt1c2VdLCB5W3VzZV0sIDEpCiAgICByZXNpZCA9IHkgLSAoaW50ZXJjZXB0ICsgc2xvcGUgKiB4KQogICAgcmV0dXJuIGludGVyY2VwdCwgc2xvcGUsIHVzZSwgZmxvYXQobnAuc3RkKHJlc2lkW3VzZV0pKQoKCmRlZiBjYWxpYnJhdGlvbl9jYW5kaWRhdGVzKGRmLCB3Y3MsIHNoYXBlLCB3aGljaCwgY2VudHJlX3B4PU5vbmUsIHIyNV9weD1Ob25lKToKICAgICIiIkNhdGFsb2d1ZSBzdGFycyBtYXRjaGVkIHRvIHNvdXJjZXMgdGhhdCBjYW4gc2V0IGEgY2FsaWJyYXRpb246CiAgICB1bnNhdHVyYXRlZCBpbiBib3RoIGJhbmRzLCB3aXRoIG5vIG90aGVyIHNvdXJjZSBpbnNpZGUgdGhlIGFwZXJ0dXJlLCB3ZWxsCiAgICBtZWFzdXJlZCwgd2l0aCBmaW5pdGUgYW5kIHdlbGwtbWVhc3VyZWQgY2F0YWxvZ3VlIG1hZ25pdHVkZXMuIFdoZXJlIGVub3VnaAogICAgb2YgdGhlbSBsaWUgYmV5b25kIFIyNSBvbmx5IHRob3NlIGFyZSB1c2VkOiB0aGUgc2t5IHJpbmcgb2YgYSBzdGFyIG9uIHRoZQogICAgZGlzYyBpcyBtZWFzdXJpbmcgdGhlIGRpc2MuIiIiCiAgICBjZW50cmUsIHJhZGl1cyA9IGZpZWxkX2NlbnRyZV9hbmRfcmFkaXVzKHdjcywgc2hhcGUpCiAgICBuYW1lID0gIkFQQVNTIERSOSIgaWYgd2hpY2ggPT0gIkFQQVNTIiBlbHNlICJHYWlhIERSMyBzeW50aGV0aWMgcGhvdG9tZXRyeSIKICAgIHByaW50KGYiW2NhbGliXSBxdWVyeWluZyB7bmFtZX0gd2l0aGluIHtyYWRpdXMgKiA2MDouMGZ9IGFyY21pbiBvZiB0aGUgZmllbGQgIgogICAgICAgICAgZiJjZW50cmUuLi4iLCBmbHVzaD1UcnVlKQogICAgY2F0ID0gcXVlcnlfY2FsaWJyYXRpb25fY2F0YWxvZ3VlKHdoaWNoLCBjZW50cmUsIHJhZGl1cykKICAgIHByaW50KGYiICAge2xlbihjYXQpOix9IGNhdGFsb2d1ZSBzdGFycyBpbiB0aGUgcmVnaW9uIiwgZmx1c2g9VHJ1ZSkKICAgIG0gPSBtYXRjaF9jYXRhbG9ndWUoZGYsIHdjcywgY2F0LCBzaGFwZSkKICAgIHN0YXRzID0geyJjYXRhbG9ndWUiOiBsZW4oY2F0KSwgIm1hdGNoZWQiOiBsZW4obSl9CiAgICBpZiBub3QgbGVuKG0pOgogICAgICAgIHJldHVybiBtLCBzdGF0cwogICAga2VlcCA9IH5tWyJTYXR1cmF0ZWQiXS5hc3R5cGUoYm9vbCkKICAgIGNyb3dkZWQgPSBtWyJOZWFyZXN0X3B4Il0gPCBtWyJBcGVydHVyZV9SYWRpdXMiXQogICAgZ29vZF9lcnIgPSAoKG1bIk1hZ19pbnN0X0JfZXJyIl0gPCBDQUxJQl9NQVhfSU5TVF9FUlIpCiAgICAgICAgICAgICAgICAmIChtWyJNYWdfaW5zdF9WX2VyciJdIDwgQ0FMSUJfTUFYX0lOU1RfRVJSKSkKICAgIGNhdF9vayA9IH4oKG1bIkNhdF9lX0IiXSA+IENBTElCX01BWF9DQVRfRVJSKSB8IChtWyJDYXRfZV9WIl0gPiBDQUxJQl9NQVhfQ0FUX0VSUikpCiAgICBrZWVwICY9IH5jcm93ZGVkICYgZ29vZF9lcnIgJiBjYXRfb2sKICAgIHN0YXRzLnVwZGF0ZSh7InNhdHVyYXRlZCI6IGludChtWyJTYXR1cmF0ZWQiXS5hc3R5cGUoYm9vbCkuc3VtKCkpLAogICAgICAgICAgICAgICAgICAiY3Jvd2RlZCI6IGludChjcm93ZGVkLnN1bSgpKSwgIm5vaXN5IjogaW50KCh+Z29vZF9lcnIpLnN1bSgpKSwKICAgICAgICAgICAgICAgICAgImJhZF9jYXRhbG9ndWVfZXJyb3IiOiBpbnQoKH5jYXRfb2spLnN1bSgpKSwKICAgICAgICAgICAgICAgICAgIm1lZGlhbl9zZXBfYXJjc2VjIjogZmxvYXQobnAubWVkaWFuKG1bIlNlcF9hcmNzZWMiXSkpfSkKICAgIGlmIGNlbnRyZV9weCBpcyBub3QgTm9uZSBhbmQgcjI1X3B4OgogICAgICAgIG91dHNpZGUgPSBucC5oeXBvdChtWyJYX1YiXSAtIGNlbnRyZV9weFswXSwgbVsiWV9WIl0gLSBjZW50cmVfcHhbMV0pID4gcjI1X3B4CiAgICAgICAgaWYgaW50KChrZWVwICYgb3V0c2lkZSkuc3VtKCkpID49IENBTElCX01JTl9PVVRTSURFOgogICAgICAgICAgICBzdGF0c1siaW5zaWRlX2dhbGF4eV9sZWZ0X291dCJdID0gaW50KChrZWVwICYgfm91dHNpZGUpLnN1bSgpKQogICAgICAgICAgICBrZWVwICY9IG91dHNpZGUKICAgIG1bIkNhbmRpZGF0ZSJdID0ga2VlcC52YWx1ZXMKICAgIHJldHVybiBtLCBzdGF0cwoKCmRlZiBmaXRfY2FsaWJyYXRpb24oY2FuZCk6CiAgICAiIiJaZXJvIHBvaW50cyBhbmQgY29sb3VyIHRlcm1zIGZyb20gdGhlIGNhbmRpZGF0ZSBzdGFycy4KCiAgICAgICAgKGItdilfaW5zdCAgICAgID0gYWxwaGEgKyBiZXRhICogKEItVilfY2F0ICAgICAgKHRoZSBjb2xvdXIgdHJhbnNmb3JtYXRpb24pCiAgICAgICAgICAgIHNvIChCLVYpICAgID0gYSArIGMgKiAoYi12KV9pbnN0ICAgd2l0aCBjID0gMS9iZXRhLCBhID0gLWFscGhhL2JldGEKICAgICAgICBWX2NhdCAtIHZfaW5zdCAgPSBaUF9WICsga19WICogKEItVilfY2F0CiAgICAgICAgQl9jYXQgLSBiX2luc3QgID0gWlBfQiArIGtfQiAqIChCLVYpX2NhdCAgIChzaG93biwgYW5kIGEgY3Jvc3MtY2hlY2spCgogICAgVGhlIGNvbG91ciBpcyBmaXR0ZWQgdGhpcyB3YXkgcm91bmQgYmVjYXVzZSBhIHN0cmFpZ2h0LWxpbmUgZml0IHRyZWF0cyB4CiAgICBhcyBleGFjdDogdGhlIGNhdGFsb2d1ZSBjb2xvdXIgaXMgdGhlIHByZWNpc2Ugb25lLiBFdmVyeSBmaXQgaXMKICAgIHNpZ21hLWNsaXBwZWQ7IGEgc3RhciBjbGlwcGVkIGZyb20gdGhlIGNvbG91ciBvciB0aGUgViBmaXQgaXMgbWFya2VkIGFzCiAgICBub3QgdXNlZC4gVGhpcyByZXBsYWNlcyBvbmUgemVybyBwb2ludCBwZXIgYmFuZCBmcm9tIHRoZSBicmlnaHRlc3Qgc3RhcnMsCiAgICB3aGljaCBsZWZ0IHRoZSBjb2xvdXIgb24gdGhlIGNhbWVyYSdzIG93biBzY2FsZSAtIHdpdGggZmlsdGVycyB0aGF0IGFyZQogICAgbm90IGV4YWN0bHkgSm9obnNvbiBCIGFuZCBWLCBldmVyeSBCLVYgd2FzIG9mZiBieSBhIGZyYWN0aW9uIG9mIGl0c2VsZi4KICAgICIiIgogICAgYyA9IGNhbmRbY2FuZFsiQ2FuZGlkYXRlIl1dLmNvcHkoKQogICAgYnZfaW5zdCA9IGNbIk1hZ19pbnN0X0IiXSAtIGNbIk1hZ19pbnN0X1YiXQogICAgYnZfY2F0ID0gY1siQ2F0X0IiXSAtIGNbIkNhdF9WIl0KICAgIGFscGhhLCBiZXRhLCB1c2VfYywgXyA9IGNsaXBwZWRfbGluZShidl9jYXQsIGJ2X2luc3QpCiAgICBjdGVybSwgYSA9IDEuMCAvIGJldGEsIC1hbHBoYSAvIGJldGEKICAgIHJtc19jID0gZmxvYXQobnAuc3RkKChhICsgY3Rlcm0gKiBidl9pbnN0IC0gYnZfY2F0KVt1c2VfY10pKQogICAgenB2LCBrdiwgdXNlX3YsIHJtc192ID0gY2xpcHBlZF9saW5lKGJ2X2NhdCwgY1siQ2F0X1YiXSAtIGNbIk1hZ19pbnN0X1YiXSkKICAgIHpwYiwga2IsIHVzZV9iLCBybXNfYiA9IGNsaXBwZWRfbGluZShidl9jYXQsIGNbIkNhdF9CIl0gLSBjWyJNYWdfaW5zdF9CIl0pCiAgICB1c2VkID0gdXNlX2MgJiB1c2VfdgogICAgY29yciA9IGZsb2F0KG5wLmNvcnJjb2VmKGJ2X2luc3QsIGJ2X2NhdClbMCwgMV0pIGlmIGxlbihjKSA+IDIgZWxzZSBucC5uYW4KICAgIHJldHVybiB7ImEiOiBhLCAiYyI6IGN0ZXJtLCAiWlBfViI6IHpwdiwgImtfViI6IGt2LCAiWlBfQiI6IHpwYiwgImtfQiI6IGtiLAogICAgICAgICAgICAicm1zX2NvbG91cl9maXQiOiBybXNfYywgInJtc19WX2ZpdCI6IHJtc192LCAicm1zX0JfZml0Ijogcm1zX2IsCiAgICAgICAgICAgICJOX2NhbmRpZGF0ZXMiOiBsZW4oYyksICJOX3VzZWQiOiBpbnQodXNlZC5zdW0oKSksCiAgICAgICAgICAgICJjb3JyZWxhdGlvbiI6IGNvcnIsICJ1c2VkX2luZGV4IjogYy5pbmRleFt1c2VkXX0KCgpkZWYgYXBwbHlfY2FsaWJyYXRpb24oZGYsIGNhbCk6CiAgICAiIiJDYWxpYnJhdGVkIEIsIFYsIEItViBhbmQgdGhlaXIgZXJyb3JzIGZvciBldmVyeSBzb3VyY2UgaW4gZGYuIiIiCiAgICBiLCB2ID0gZGZbIk1hZ19pbnN0X0IiXS50b19udW1weShmbG9hdCksIGRmWyJNYWdfaW5zdF9WIl0udG9fbnVtcHkoZmxvYXQpCiAgICBlYiwgZXYgPSBkZlsiTWFnX2luc3RfQl9lcnIiXS50b19udW1weShmbG9hdCksIGRmWyJNYWdfaW5zdF9WX2VyciJdLnRvX251bXB5KGZsb2F0KQogICAgYSwgYywgenAsIGsgPSBjYWxbImEiXSwgY2FsWyJjIl0sIGNhbFsiWlBfViJdLCBjYWxbImtfViJdCiAgICBidiA9IGEgKyBjICogKGIgLSB2KQogICAgViA9IHYgKyB6cCArIGsgKiBidgogICAgQiA9IFYgKyBidgogICAgIyBWID0gKDEgLSBrIGMpIHYgKyBrIGMgYiArIGNvbnN0IDsgQiA9ICgxIC0gKDEraykgYykgdiArICgxK2spIGMgYiArIGNvbnN0CiAgICBvdXQgPSBkZi5jb3B5KCkKICAgIG91dFsiTWFnX0Jfb2JzIl0sIG91dFsiTWFnX1Zfb2JzIl0sIG91dFsiQlZfb2JzIl0gPSBCLCBWLCBidgogICAgb3V0WyJNYWdfQl9lcnIiXSA9IG5wLmh5cG90KCgxIC0gKDEgKyBrKSAqIGMpICogZXYsICgxICsgaykgKiBjICogZWIpCiAgICBvdXRbIk1hZ19WX2VyciJdID0gbnAuaHlwb3QoKDEgLSBrICogYykgKiBldiwgayAqIGMgKiBlYikKICAgIG91dFsiQlZfZXJyIl0gPSBucC5hYnMoYykgKiBucC5oeXBvdChlYiwgZXYpCiAgICByZXR1cm4gb3V0CgoKZGVmIGNhbGlicmF0ZShkZiwgd2NzLCBzaGFwZSwgd2hpY2gsIGNlbnRyZV9weCwgcjI1X3B4LCBvdXRfZGlyLCBzYWZlKToKICAgICIiIlJ1biB0aGUgd2hvbGUgY2FsaWJyYXRpb24sIHdpdGggR2FpYSBhcyB0aGUgZmFsbGJhY2sgY2F0YWxvZ3VlLCB0aGUKICAgIGNoZWNrcyB0aGF0IHN0b3AgYSBydW4gdGhhdCBjYW5ub3QgYmUgcmlnaHQsIGFuZCB0aGUgY2FsaWJyYXRpb24gdGFibGUKICAgIHdyaXR0ZW4gb3V0LiBSZXR1cm5zIChjYWwsIGNhdGFsb2d1ZSB1c2VkLCBzdGF0cywgbWF0Y2hlZCB0YWJsZSkuIiIiCiAgICBjYW5kLCBzdGF0cyA9IGNhbGlicmF0aW9uX2NhbmRpZGF0ZXMoZGYsIHdjcywgc2hhcGUsIHdoaWNoLCBjZW50cmVfcHgsIHIyNV9weCkKICAgIG5fY2FuZCA9IGludChjYW5kWyJDYW5kaWRhdGUiXS5zdW0oKSkgaWYgbGVuKGNhbmQpIGVsc2UgMAogICAgaWYgbl9jYW5kIDwgQ0FMSUJfTUlOX1NUQVJTIGFuZCB3aGljaCA9PSAiQVBBU1MiOgogICAgICAgIHByaW50KGYiW2NhbGliXSBvbmx5IHtuX2NhbmR9IHVzYWJsZSBBUEFTUyBzdGFycyAtIHRyeWluZyBHYWlhIERSMyBzeW50aGV0aWMgIgogICAgICAgICAgICAgIGYicGhvdG9tZXRyeSBpbnN0ZWFkIiwgZmx1c2g9VHJ1ZSkKICAgICAgICB3aGljaCA9ICJHQUlBIgogICAgICAgIGNhbmQsIHN0YXRzID0gY2FsaWJyYXRpb25fY2FuZGlkYXRlcyhkZiwgd2NzLCBzaGFwZSwgd2hpY2gsIGNlbnRyZV9weCwgcjI1X3B4KQogICAgICAgIG5fY2FuZCA9IGludChjYW5kWyJDYW5kaWRhdGUiXS5zdW0oKSkgaWYgbGVuKGNhbmQpIGVsc2UgMAogICAgaWYgbl9jYW5kIDwgQ0FMSUJfTUlOX1NUQVJTOgogICAgICAgIHN0b3AoZiJDQUxJQlJBVElPTjogb25seSB7bl9jYW5kfSB1c2FibGUgcmVmZXJlbmNlIHN0YXJzIChhdCBsZWFzdCAiCiAgICAgICAgICAgICBmIntDQUxJQl9NSU5fU1RBUlN9IGFyZSBuZWVkZWQpLiBNYXRjaGVkIHtzdGF0cy5nZXQoJ21hdGNoZWQnLCAwKX0sIG9mICIKICAgICAgICAgICAgIGYid2hpY2ggc2F0dXJhdGVkIHtzdGF0cy5nZXQoJ3NhdHVyYXRlZCcsIDApfSwgY3Jvd2RlZCAiCiAgICAgICAgICAgICBmIntzdGF0cy5nZXQoJ2Nyb3dkZWQnLCAwKX0sIHRvbyBub2lzeSB7c3RhdHMuZ2V0KCdub2lzeScsIDApfS4iKQogICAgcHJpbnQoZiJbY2FsaWJdIHtzdGF0c1snbWF0Y2hlZCddfSBjYXRhbG9ndWUgc3RhcnMgbWF0Y2hlZCAobWVkaWFuIHNlcGFyYXRpb24gIgogICAgICAgICAgZiJ7c3RhdHNbJ21lZGlhbl9zZXBfYXJjc2VjJ106LjJmfSBhcmNzZWMpOyBsZWZ0IG91dDogIgogICAgICAgICAgZiJ7c3RhdHNbJ3NhdHVyYXRlZCddfSBzYXR1cmF0ZWQsIHtzdGF0c1snY3Jvd2RlZCddfSBjcm93ZGVkLCAiCiAgICAgICAgICBmIntzdGF0c1snbm9pc3knXX0gdG9vIG5vaXN5LCB7c3RhdHNbJ2JhZF9jYXRhbG9ndWVfZXJyb3InXX0gd2l0aCBsYXJnZSAiCiAgICAgICAgICBmImNhdGFsb2d1ZSBlcnJvcnMiCiAgICAgICAgICArIChmIiwge3N0YXRzWydpbnNpZGVfZ2FsYXh5X2xlZnRfb3V0J119IG9uIHRoZSBnYWxheHkncyBkaXNjIgogICAgICAgICAgICAgaWYgc3RhdHMuZ2V0KCJpbnNpZGVfZ2FsYXh5X2xlZnRfb3V0IikgZWxzZSAiIiksIGZsdXNoPVRydWUpCiAgICBjYWwgPSBmaXRfY2FsaWJyYXRpb24oY2FuZCkKICAgIGlmIGNhbFsiTl91c2VkIl0gPCBDQUxJQl9NSU5fU1RBUlM6CiAgICAgICAgc3RvcChmIkNBTElCUkFUSU9OOiBvbmx5IHtjYWxbJ05fdXNlZCddfSBzdGFycyBzdXJ2aXZlZCB0aGUgb3V0bGllciAiCiAgICAgICAgICAgICBmInJlamVjdGlvbiAoYXQgbGVhc3Qge0NBTElCX01JTl9TVEFSU30gYXJlIG5lZWRlZCkuIikKCiAgICAjIFRoZSBjb2xvdXIgY2hlY2suIFRoZSBpbnN0cnVtZW50YWwgY29sb3VyIG11c3QgcmlzZSB3aXRoIHRoZSBjYXRhbG9ndWUKICAgICMgY29sb3VyOyBpZiBpdCBmYWxscywgdGhlIGZyYW1lIGxhYmVsbGVkIEIgaXMgbm90IHRoZSBibHVlIG9uZS4gTm90aGluZwogICAgIyBlbHNlIGluIHRoZSBvdXRwdXQgd291bGQgc2hvdyBpdDogdGhlIGRpYWdyYW0gd291bGQgbG9vayBub3JtYWwsIHdpdGgKICAgICMgZXZlcnkgY29sb3VyIHJldmVyc2VkLgogICAgaWYgbm90IChjYWxbImNvcnJlbGF0aW9uIl0gPiAwKToKICAgICAgICBzdG9wKGYiQ09MT1VSIENIRUNLOiB0aGUgaW5zdHJ1bWVudGFsIGNvbG91ciBiLXYgRkFMTFMgYXMgdGhlIGNhdGFsb2d1ZSAiCiAgICAgICAgICAgICBmIkItViByaXNlcyAoY29ycmVsYXRpb24ge2NhbFsnY29ycmVsYXRpb24nXTorLjJmfSBvdmVyICIKICAgICAgICAgICAgIGYie2NhbFsnTl9jYW5kaWRhdGVzJ119IHN0YXJzKS4gVGhlIGZyYW1lIHVzZWQgYXMgQiBpcyBwcm9iYWJseSB0aGUgIgogICAgICAgICAgICAgZiJWIChvciBSKSBmcmFtZSBhbmQgdGhlIG90aGVyIHdheSByb3VuZCAtIGNoZWNrIHRoZSBGSUxURVIga2V5d29yZHMgIgogICAgICAgICAgICAgZiJhbmQgdGhlIGZpbGUgbmFtZXMuIikKICAgIGlmIGNhbFsiY29ycmVsYXRpb24iXSA8IDAuNToKICAgICAgICBsb3VkKGYiV0FSTklORzogdGhlIGluc3RydW1lbnRhbCBhbmQgY2F0YWxvZ3VlIGNvbG91cnMgYWdyZWUgb25seSB3ZWFrbHkgIgogICAgICAgICAgICAgZiIoY29ycmVsYXRpb24ge2NhbFsnY29ycmVsYXRpb24nXTouMmZ9KS4gVGhlIGNvbG91cnMgYmVsb3cgYXJlICIKICAgICAgICAgICAgIGYidW5jZXJ0YWluLiIpCiAgICBwcmludChmIltjYWxpYl0ge2NhbFsnTl91c2VkJ119IG9mIHtjYWxbJ05fY2FuZGlkYXRlcyddfSBzdGFycyB1c2VkICIKICAgICAgICAgIGYiKHsnQVBBU1MgRFI5JyBpZiB3aGljaCA9PSAnQVBBU1MnIGVsc2UgJ0dhaWEgRFIzIHN5bnRoZXRpYyd9KToiLCBmbHVzaD1UcnVlKQogICAgcHJpbnQoZiIgICBCLVYgPSB7Y2FsWydhJ106Ky4zZn0gKyB7Y2FsWydjJ106LjNmfSB4IChiLXYpICAgICAgc2NhdHRlciAiCiAgICAgICAgICBmIntjYWxbJ3Jtc19jb2xvdXJfZml0J106LjNmfSBtYWciLCBmbHVzaD1UcnVlKQogICAgcHJpbnQoZiIgICBWICAgPSB2ICsge2NhbFsnWlBfViddOi4zZn0ge2NhbFsna19WJ106Ky4zZn0geCAoQi1WKSAgIHNjYXR0ZXIgIgogICAgICAgICAgZiJ7Y2FsWydybXNfVl9maXQnXTouM2Z9IG1hZyIsIGZsdXNoPVRydWUpCiAgICBwcmludChmIiAgIEIgICA9IGIgKyB7Y2FsWydaUF9CJ106LjNmfSB7Y2FsWydrX0InXTorLjNmfSB4IChCLVYpICAgc2NhdHRlciAiCiAgICAgICAgICBmIntjYWxbJ3Jtc19CX2ZpdCddOi4zZn0gbWFnIiwgZmx1c2g9VHJ1ZSkKICAgIHByaW50KGYiICAgY29sb3VyIGNvcnJlbGF0aW9uIHtjYWxbJ2NvcnJlbGF0aW9uJ106Ky4yZn0iLCBmbHVzaD1UcnVlKQoKICAgIHRhYmxlID0gY2FuZC5jb3B5KCkKICAgIHRhYmxlWyJVc2VkIl0gPSBGYWxzZQogICAgdGFibGUubG9jW2NhbFsidXNlZF9pbmRleCJdLCAiVXNlZCJdID0gVHJ1ZQogICAgYnZfZml0ID0gY2FsWyJhIl0gKyBjYWxbImMiXSAqICh0YWJsZVsiTWFnX2luc3RfQiJdIC0gdGFibGVbIk1hZ19pbnN0X1YiXSkKICAgIHRhYmxlWyJSZXNpZF9CViJdID0gYnZfZml0IC0gKHRhYmxlWyJDYXRfQiJdIC0gdGFibGVbIkNhdF9WIl0pCiAgICB0YWJsZVsiUmVzaWRfViJdID0gKHRhYmxlWyJNYWdfaW5zdF9WIl0gKyBjYWxbIlpQX1YiXSArIGNhbFsia19WIl0gKiBidl9maXQKICAgICAgICAgICAgICAgICAgICAgICAgLSB0YWJsZVsiQ2F0X1YiXSkKICAgIHBhdGggPSBvcy5wYXRoLmpvaW4ob3V0X2RpciwgZiJ7c2FmZX1fcmVmZXJlbmNlX3N0YXJzLmNzdiIpCiAgICB0YWJsZS50b19jc3YocGF0aCwgaW5kZXg9RmFsc2UpCiAgICBwcmludChmIlJlZmVyZW5jZSBzdGFycywgd2l0aCBlYWNoIG9uZSdzIHJlc2lkdWFsLCBzYXZlZCB0byB7cGF0aH0iKQogICAgcmV0dXJuIGNhbCwgd2hpY2gsIHN0YXRzLCBjYW5kCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tIERpYWdyYW1zIC0tLS0tLS0tLS0tLS0tLS0tLS0KCmRlZiBtYWtlX2NtZChkZiwgYmFzZV9maWxlLCBsYWJlbCwgbmFtZSwgbXU9Tm9uZSwgd2luZG93PU5vbmUpOgogICAgIiIiQ29sb3VyLW1hZ25pdHVkZSBkaWFncmFtIG9mIHRoZSBzb3VyY2VzIGluIGRmLCBzYXZlZCBuZXh0IHRvIGJhc2VfZmlsZS4KCiAgICBUaGUgY29sb3VyIGF4aXMgZm9sbG93cyB0aGUgZGF0YSByYXRoZXIgdGhhbiBhIGZpeGVkIHJhbmdlIC0gdGhlIG9sZCBmaXhlZAogICAgLTAuNSB0byAwLjUgY3V0IGV2ZXJ5IHJlZCBzb3VyY2Ugb3V0IG9mIHRoZSBkaWFncmFtIG9mIGFsbCBzb3VyY2VzLCB3aGljaAogICAgaXMgdGhlIGRpYWdyYW0gdGhhdCBzaG93cyB3aGVyZSB0aGUgYmx1ZSBrbm90cyBzdGFuZCBhZ2FpbnN0IHRoZSByZXN0LgogICAgU2F0dXJhdGVkIHNvdXJjZXMgYW5kIHNvdXJjZXMgd2l0aCBhIEItViBlcnJvciBhYm92ZSBDTURfTUFYX0JWX0VSUiBhcmUgbm90CiAgICBkcmF3biAodGhleSBzdGF5IGluIHRoZSB0YWJsZXMpLiBXaXRoIGEgZGlzdGFuY2UsIHRoZSByaWdodC1oYW5kIGF4aXMgaXMKICAgIHRoZSBhYnNvbHV0ZSBtYWduaXR1ZGUgTV9WLiBCYXJzIG9uIHRoZSBsZWZ0IHNob3cgdGhlIHR5cGljYWwgZXJyb3IgYXQKICAgIGVhY2ggbWFnbml0dWRlLgogICAgIiIiCiAgICBpZiBub3QgeyJNYWdfQiIsICJNYWdfViJ9IDw9IHNldChkZi5jb2x1bW5zKToKICAgICAgICBwcmludChmIk1hZ19CIG9yIE1hZ19WIG5vdCBmb3VuZCBpbiB7bGFiZWx9IGNhdGFsb2cuIENNRCBub3QgY3JlYXRlZC4iKQogICAgICAgIHJldHVybgogICAgZGYgPSBkZi5jb3B5KCkKICAgIGRmWyJCLVYiXSA9IGRmWyJNYWdfQiJdIC0gZGZbIk1hZ19WIl0KICAgIGNvbHMgPSBbYyBmb3IgYyBpbiBkZi5jb2x1bW5zIGlmIGMgbm90IGluICgiQi1WIiwgIk1hZ19WIildICsgWyJCLVYiLCAiTWFnX1YiXQogICAgZGYgPSBkZltjb2xzXQogICAgb3V0X2ZpbGVfY29sb3IgPSBiYXNlX2ZpbGUucmVwbGFjZSgiLmNzdiIsICJfd2l0aF9jb2xvci5jc3YiKQogICAgZGYudG9fY3N2KG91dF9maWxlX2NvbG9yLCBpbmRleD1GYWxzZSkKICAgIHByaW50KGYiRmlsZSB3aXRoIGNvbG9yIGluZGV4IHNhdmVkIHRvIHtvdXRfZmlsZV9jb2xvcn0iKQoKICAgIHNob3cgPSBucC5pc2Zpbml0ZShkZlsiQi1WIl0pICYgbnAuaXNmaW5pdGUoZGZbIk1hZ19WIl0pCiAgICBpZiAiU2F0dXJhdGVkIiBpbiBkZi5jb2x1bW5zOgogICAgICAgIHNob3cgJj0gfmRmWyJTYXR1cmF0ZWQiXS5hc3R5cGUoYm9vbCkKICAgIGlmICJCVl9lcnIiIGluIGRmLmNvbHVtbnM6CiAgICAgICAgc2hvdyAmPSBkZlsiQlZfZXJyIl0gPD0gQ01EX01BWF9CVl9FUlIKICAgIGQgPSBkZltzaG93XQogICAgaWYgbm90IGxlbihkKToKICAgICAgICBwcmludChmIk5vIHNvdXJjZXMgdG8gZHJhdyBpbiB0aGUge2xhYmVsfSBkaWFncmFtLiIpCiAgICAgICAgcmV0dXJuCgogICAgZmlnLCBheCA9IHBsdC5zdWJwbG90cyhmaWdzaXplPSg4LCAxMCkpCiAgICBheC5zY2F0dGVyKGRbIkItViJdLCBkWyJNYWdfViJdLCBzPTE4LCBlZGdlY29sb3I9ImJsYWNrIiwgZmFjZWNvbG9yPSJjeWFuIiwKICAgICAgICAgICAgICAgYWxwaGE9MC43LCBsaW5ld2lkdGhzPTAuNCkKICAgIGxvLCBoaSA9IG5wLnBlcmNlbnRpbGUoZFsiQi1WIl0sIFsxLCA5OV0pCiAgICBsbywgaGkgPSBtaW4obG8sIHdpbmRvd1swXSBpZiB3aW5kb3cgZWxzZSBsbyksIG1heChoaSwgd2luZG93WzFdIGlmIHdpbmRvdyBlbHNlIGhpKQogICAgcGFkID0gMC4xICogbWF4KGhpIC0gbG8sIDAuNSkKICAgIGF4LnNldF94bGltKGxvIC0gcGFkLCBoaSArIHBhZCkKICAgIHltaW4sIHltYXggPSBucC5uYW5taW4oZFsiTWFnX1YiXSksIG5wLm5hbm1heChkWyJNYWdfViJdKQogICAgYXguc2V0X3lsaW0oeW1heCArIDAuMywgeW1pbiAtIDAuMykKICAgIGlmIHdpbmRvdzoKICAgICAgICBmb3IgeHcgaW4gd2luZG93OgogICAgICAgICAgICBheC5heHZsaW5lKHh3LCBjb2xvcj0idGFiOmJsdWUiLCBscz0iLS0iLCBsdz0xKQogICAgaWYgeyJNYWdfVl9lcnIiLCAiQlZfZXJyIn0gPD0gc2V0KGQuY29sdW1ucyk6CiAgICAgICAgeF9iYXIgPSBsbyAtIDAuNSAqIHBhZAogICAgICAgIGZvciBtMCBpbiBucC5hcmFuZ2UobnAuZmxvb3IoeW1pbiksIG5wLmNlaWwoeW1heCksIDEuMCk6CiAgICAgICAgICAgIHNlbCA9IChkWyJNYWdfViJdID49IG0wKSAmIChkWyJNYWdfViJdIDwgbTAgKyAxKQogICAgICAgICAgICBpZiBzZWwuc3VtKCkgPj0gMzoKICAgICAgICAgICAgICAgIGF4LmVycm9yYmFyKHhfYmFyLCBtMCArIDAuNSwgeGVycj1ucC5tZWRpYW4oZC5sb2Nbc2VsLCAiQlZfZXJyIl0pLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgeWVycj1ucC5tZWRpYW4oZC5sb2Nbc2VsLCAiTWFnX1ZfZXJyIl0pLCBjb2xvcj0iMC4zIiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNhcHNpemU9MiwgbHc9MSkKICAgIGF4LnNldF94bGFiZWwoIihCIC0gVikkXzAkICBbbWFnXSwgY29ycmVjdGVkIGZvciBHYWxhY3RpYyByZWRkZW5pbmciKQogICAgYXguc2V0X3lsYWJlbCgiViRfMCQgIFttYWddLCBhcHBhcmVudCwgY29ycmVjdGVkIGZvciBHYWxhY3RpYyBleHRpbmN0aW9uIikKICAgIGlmIG11IGlzIG5vdCBOb25lIGFuZCBucC5pc2Zpbml0ZShtdSk6CiAgICAgICAgc2VjID0gYXguc2Vjb25kYXJ5X3lheGlzKCJyaWdodCIsIGZ1bmN0aW9ucz0obGFtYmRhIG06IG0gLSBtdSwgbGFtYmRhIE06IE0gKyBtdSkpCiAgICAgICAgc2VjLnNldF95bGFiZWwoZiJNJF9WJCAgW21hZ10gICAoZGlzdGFuY2UgbW9kdWx1cyB7bXU6LjJmfSkiKQogICAgbGVmdF9vdXQgPSBpbnQobGVuKGRmKSAtIGxlbihkKSkKICAgIGF4LnNldF90aXRsZShmIkNNRCBvZiB7bmFtZX0ge2xhYmVsfVxue2xlbihkKX0gb2Yge2xlbihkZil9IHNvdXJjZXMgZHJhd24iCiAgICAgICAgICAgICAgICAgKyAoZiIgKHtsZWZ0X291dH0gc2F0dXJhdGVkIG9yIHdpdGggYSBCLVYgZXJyb3IgYWJvdmUgIgogICAgICAgICAgICAgICAgICAgIGYie0NNRF9NQVhfQlZfRVJSOmd9IG1hZyBsZWZ0IG91dCkiIGlmIGxlZnRfb3V0IGVsc2UgIiIpCiAgICAgICAgICAgICAgICAgKyAoZiI7IGRhc2hlZCBsaW5lczogdGhlIGJsdWUta25vdCBjb2xvdXIgd2luZG93IiBpZiB3aW5kb3cgZWxzZSAiIikpCiAgICBvdXRfY21kID0gYmFzZV9maWxlLnJlcGxhY2UoIi5jc3YiLCAiX0NNRC5wbmciKQogICAgZmlnLnNhdmVmaWcob3V0X2NtZCwgZHBpPTE1MCwgYmJveF9pbmNoZXM9InRpZ2h0IikKICAgIHBsdC5jbG9zZShmaWcpCiAgICBwcmludChmIkNNRCBkaWFncmFtIHNhdmVkIHRvIHtvdXRfY21kfSIpCgoKZGVmIHNvdXJjZV9tYXAoZGF0YSwgeCwgeSwgdGl0bGUsIHBhdGgsIGNvbG91cj0iZ3JlZW4iLCBsYWJlbD0iTWVhc3VyZWQgc291cmNlcyIpOgogICAgcGx0LmZpZ3VyZShmaWdzaXplPSgxMCwgMTApKQogICAgZmluaXRlID0gZGF0YVtucC5pc2Zpbml0ZShkYXRhKV0KICAgIHBsdC5pbXNob3coZGF0YSwgY21hcD0iZ3JheSIsIG9yaWdpbj0ibG93ZXIiLAogICAgICAgICAgICAgICB2bWluPW5wLnBlcmNlbnRpbGUoZmluaXRlLCA1KSwgdm1heD1ucC5wZXJjZW50aWxlKGZpbml0ZSwgOTkpKQogICAgcGx0LmNvbG9yYmFyKGxhYmVsPSJDb3VudHMiKQogICAgcGx0LnNjYXR0ZXIoeCwgeSwgcz00MCwgZWRnZWNvbG9yPWNvbG91ciwgZmFjZWNvbG9yPSJub25lIiwgbGFiZWw9bGFiZWwpCiAgICBwbHQudGl0bGUodGl0bGUpCiAgICBwbHQueGxhYmVsKCJYIFtwaXhlbHNdIikKICAgIHBsdC55bGFiZWwoIlkgW3BpeGVsc10iKQogICAgcGx0LmxlZ2VuZCgpCiAgICBwbHQuc2F2ZWZpZyhwYXRoLCBkcGk9MTUwKQogICAgcGx0LmNsb3NlKCkKICAgIHByaW50KGYiU291cmNlIG1hcCBzYXZlZCB0byB7cGF0aH0iKQoKCmRlZiBnYWxheHlfY2VudHJlX3BpeGVsKG5hbWUsIHdjcywgc2hhcGUpOgogICAgIiIiV2hlcmUgdGhlIGdhbGF4eSBzaXRzIGluIHRoaXMgZnJhbWUsIGluIHBpeGVscywgZnJvbSBpdHMgY2F0YWxvZ3VlZAogICAgcG9zaXRpb24gdGhyb3VnaCB0aGUgKGNoZWNrZWQpIHBsYXRlIHNvbHV0aW9uOyBOb25lIHdoZW4gdW5rbm93biBvcgogICAgb3V0c2lkZSB0aGUgZnJhbWUuIiIiCiAgICB0cnk6CiAgICAgICAgeCwgeSA9IHdjcy53b3JsZF90b19waXhlbChTa3lDb29yZC5mcm9tX25hbWUobmFtZSkpCiAgICAgICAgeCwgeSA9IGZsb2F0KHgpLCBmbG9hdCh5KQogICAgICAgIGlmIDAgPD0geCA8IHNoYXBlWzFdIGFuZCAwIDw9IHkgPCBzaGFwZVswXToKICAgICAgICAgICAgcHJpbnQoZiJbbG9va3VwXSB7bmFtZX06IGNlbnRyZWQgYXQgcGl4ZWwgKHt4Oi4wZn0sIHt5Oi4wZn0pIiwgZmx1c2g9VHJ1ZSkKICAgICAgICAgICAgcmV0dXJuIHgsIHkKICAgICAgICBwcmludChmIltsb29rdXBdIHtuYW1lfSdzIGNhdGFsb2d1ZWQgcG9zaXRpb24gZmFsbHMgb3V0c2lkZSB0aGlzIGZyYW1lIiwKICAgICAgICAgICAgICBmbHVzaD1UcnVlKQogICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBleGM6CiAgICAgICAgcHJpbnQoZiJbbG9va3VwXSBnYWxheHkgY2VudHJlIHVuYXZhaWxhYmxlICh7ZXhjfSkiLCBmbHVzaD1UcnVlKQogICAgcmV0dXJuIE5vbmUKCgojIC0tLS0tLS0tLS0tLS0tLS0tLS0gVGhlIHJ1biAtLS0tLS0tLS0tLS0tLS0tLS0tCgpkZWYgcGFyc2VfYXJncyhhcmd2PU5vbmUpOgogICAgYXAgPSBhcmdwYXJzZS5Bcmd1bWVudFBhcnNlcigKICAgICAgICBkZXNjcmlwdGlvbj0iQmx1ZSBrbm90cyBvZiBhIGdhbGF4eSBmcm9tIGEgQiBhbmQgYSBWIGZyYW1lOiBjYWxpYnJhdGVkICIKICAgICAgICAgICAgICAgICAgICAicGhvdG9tZXRyeSwgY29sb3VyLW1hZ25pdHVkZSBkaWFncmFtLCByYWRpYWwgcHJvZmlsZS4gIgogICAgICAgICAgICAgICAgICAgICJTdGFydGVkIHdpdGhvdXQgYXJndW1lbnRzIGl0IGFza3MgaXRzIHF1ZXN0aW9ucyBpbnN0ZWFkLiIpCiAgICBhcC5hZGRfYXJndW1lbnQoIi0tZm9sZGVyIiwgaGVscD0iZm9sZGVyIGhvbGRpbmcgdGhlIGZyYW1lczsgd2hpY2ggaXMgQiBhbmQgd2hpY2ggIgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgImlzIFYgaXMgcmVhZCBmcm9tIHRoZSBoZWFkZXJzIikKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1iIiwgaGVscD0idGhlIEIgZnJhbWUsIHdoZW4gbmFtaW5nIHRoZSBmaWxlcyB5b3Vyc2VsZiIpCiAgICBhcC5hZGRfYXJndW1lbnQoIi0tdiIsIGhlbHA9InRoZSBWIGZyYW1lIikKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1nYWxheHkiLCBoZWxwPSJnYWxheHkgbmFtZSAob3RoZXJ3aXNlIGZyb20gdGhlIE9CSkVDVCBrZXl3b3JkKSIpCiAgICBhcC5hZGRfYXJndW1lbnQoIi0tZGlzdGFuY2UtbXBjIiwgdHlwZT1mbG9hdCwgaGVscD0iZGlzdGFuY2U7IG90aGVyd2lzZSBsb29rZWQgdXAiKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLWVidiIsIHR5cGU9ZmxvYXQsIGhlbHA9IkdhbGFjdGljIEUoQi1WKTsgb3RoZXJ3aXNlIGxvb2tlZCB1cCIpCiAgICBhcC5hZGRfYXJndW1lbnQoIi0tY2F0YWxvZyIsIGNob2ljZXM9WyJBUEFTUyIsICJHQUlBIl0sIGRlZmF1bHQ9Q0FMSUJfQ0FUQUxPR1VFLAogICAgICAgICAgICAgICAgICAgIGhlbHA9InJlZmVyZW5jZSBzdGFycyBmb3IgdGhlIGNhbGlicmF0aW9uIChkZWZhdWx0IEFQQVNTOyBHYWlhIERSMyAiCiAgICAgICAgICAgICAgICAgICAgICAgICAic3ludGhldGljIHBob3RvbWV0cnkgaXMgYWxzbyB0aGUgYXV0b21hdGljIGZhbGxiYWNrKSIpCiAgICBhcC5hZGRfYXJndW1lbnQoIi0tbWF4LWJ2LWVyciIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9S05PVF9NQVhfQlZfRVJSLAogICAgICAgICAgICAgICAgICAgIGhlbHA9Imxhcmdlc3QgQi1WIGVycm9yIGEgYmx1ZSBrbm90IG1heSBoYXZlLCBpbiBtYWciKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLW91dC1kaXIiLCBoZWxwPSJ3cml0ZSBldmVyeXRoaW5nIGludG8gdGhpcyBmb2xkZXIgKGRlZmF1bHQ6IGEgIgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICJuZXcgZm9sZGVyIHJlc3VsdF9DTURfPGdhbGF4eT5fPGRhdGU+Xzx0aW1lPiAiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgImJlc2lkZSB0aGUgZnJhbWVzKSIpCiAgICBhcC5hZGRfYXJndW1lbnQoIi0teWVzIiwgYWN0aW9uPSJzdG9yZV90cnVlIiwKICAgICAgICAgICAgICAgICAgICBoZWxwPSJhY2NlcHQgZXZlcnkgZGVmYXVsdCB3aXRob3V0IGFza2luZyIpCiAgICByZXR1cm4gYXAucGFyc2VfYXJncyhhcmd2KQoKCmRlZiBtYWluKGFyZ3Y9Tm9uZSk6CiAgICBnbG9iYWwgQVJHUywgYXNrLCBhc2tfeWVzX25vCiAgICBBUkdTID0gcGFyc2VfYXJncyhhcmd2KQogICAgaWYgQVJHUy55ZXM6CiAgICAgICAgIyBUaGUgc2hhcmVkIGZvbGRlciBxdWVzdGlvbnMgKHdoaWNoIGZyYW1lLCB3aGV0aGVyIHRvIGNvbWJpbmUpIHRha2UKICAgICAgICAjIHRoZWlyIGRlZmF1bHRzIHRvby4KICAgICAgICBhc2sgPSBsYW1iZGEgcHJvbXB0LCBkZWZhdWx0PU5vbmUsIGNhc3Q9c3RyOiBjYXN0KGRlZmF1bHQpCiAgICAgICAgYXNrX3llc19ubyA9IGxhbWJkYSBwcm9tcHQsIGRlZmF1bHQ9RmFsc2U6IGRlZmF1bHQKCiAgICBwcmludCgiPT09IFBob3RvbWV0cnkgSW5wdXQgPT09IikKICAgIGZvbGRlciA9IEFSR1MuZm9sZGVyCiAgICBpZiBub3QgZm9sZGVyIGFuZCBub3QgKEFSR1MuYiBhbmQgQVJHUy52KToKICAgICAgICBmb2xkZXIgPSBfbm9ybV9wYXRoKF9hc2soCiAgICAgICAgICAgICJQYXRoIHRvIHRoZSBmb2xkZXIgaG9sZGluZyB0aGUgZnJhbWVzIChFbnRlciB0byBuYW1lIHRoZSB0d28gZmlsZXMgaW5zdGVhZCkiLAogICAgICAgICAgICAiIiwgc3RyKSkKICAgIGZpdHNfZmlsZV9CLCBmaXRzX2ZpbGVfViA9IEFSR1MuYiwgQVJHUy52CiAgICBvYmpfbmFtZSA9IEFSR1MuZ2FsYXh5CiAgICBncm91cHMsIGRvbm9ycyA9IHt9LCB7fQogICAgaWYgZm9sZGVyIGFuZCBvcy5wYXRoLmlzZGlyKGZvbGRlcik6CiAgICAgICAgY2hvc2VuLCBkb25vcnMsIGdyb3VwcywgZm91bmQgPSBmcmFtZXNfZnJvbV9mb2xkZXIoZm9sZGVyLCAoIkIiLCAiViIpKQogICAgICAgIGZpdHNfZmlsZV9CLCBmaXRzX2ZpbGVfViA9IGNob3NlblsiQiJdLCBjaG9zZW5bIlYiXQogICAgICAgIG9ial9uYW1lID0gb2JqX25hbWUgb3IgZm91bmQKICAgIGVsaWYgZm9sZGVyOgogICAgICAgIHByaW50KCJbaW5wdXRdIHRoZXJlIGlzIG5vIGZvbGRlciBhdCAiICsgZm9sZGVyICsgIiAtIG5hbWluZyB0aGUgdHdvIGZpbGVzICIKICAgICAgICAgICAgICAiaW5zdGVhZCIsIGZsdXNoPVRydWUpCiAgICAgICAgZm9sZGVyID0gTm9uZQogICAgaWYgbm90IGZpdHNfZmlsZV9COgogICAgICAgIGZpdHNfZmlsZV9CID0gX25vcm1fcGF0aChfYXNrKCJQYXRoIHRvIEItYmFuZCBGSVRTIiwgTm9uZSwgc3RyKSkKICAgIGlmIG5vdCBmaXRzX2ZpbGVfVjoKICAgICAgICBmaXRzX2ZpbGVfViA9IF9ub3JtX3BhdGgoX2FzaygiUGF0aCB0byBWLWJhbmQgRklUUyIsIE5vbmUsIHN0cikpCiAgICBmb3IgcCBpbiAoZml0c19maWxlX0IsIGZpdHNfZmlsZV9WKToKICAgICAgICBpZiBub3Qgb3MucGF0aC5pc2ZpbGUocCk6CiAgICAgICAgICAgIHN0b3AoZiJ0aGVyZSBpcyBubyBmaWxlIGF0IHtwfSIpCiAgICBpZiBub3Qgb2JqX25hbWU6CiAgICAgICAgdHJ5OgogICAgICAgICAgICBvYmpfbmFtZSA9IHN0cihmaXRzLmdldGhlYWRlcihmaXRzX2ZpbGVfVikuZ2V0KCJPQkpFQ1QiLCAiIikpLnN0cmlwKCkgb3IgTm9uZQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIG9ial9uYW1lID0gTm9uZQogICAgaWYgbm90IG9ial9uYW1lOgogICAgICAgIG9ial9uYW1lID0gX2FzaygiR2FsYXh5IG5hbWUiLCBOb25lLCBzdHIpCiAgICBzYWZlID0gcmUuc3ViKHIiW15BLVphLXowLTlfLi1dKyIsICJfIiwgb2JqX25hbWUuc3RyaXAoKSkKCiAgICBkYXRhZGlyID0gZm9sZGVyIG9yIG9zLnBhdGguZGlybmFtZShvcy5wYXRoLmFic3BhdGgoZml0c19maWxlX1YpKQogICAgaWYgQVJHUy5vdXRfZGlyOgogICAgICAgIG91dF9kaXIgPSBvcy5wYXRoLmFic3BhdGgoQVJHUy5vdXRfZGlyKQogICAgICAgIG9zLm1ha2VkaXJzKG91dF9kaXIsIGV4aXN0X29rPVRydWUpCiAgICBlbHNlOgogICAgICAgIG91dF9kaXIgPSBuZXdfcnVuX2ZvbGRlcihkYXRhZGlyLCAiQ01EIiwgb2JqX25hbWUpCiAgICBwcmludChmIltvdXRdIHdyaXRpbmcgZXZlcnl0aGluZyB0byB7b3V0X2Rpcn0iKQoKICAgIGNvbWJpbmVkX25vdGUgPSBbXQogICAgZm9yIHJvbGUgaW4gKCJCIiwgIlYiKToKICAgICAgICBwYXRocyA9IGdyb3Vwcy5nZXQocm9sZSkKICAgICAgICBpZiBub3QgcGF0aHMgYW5kIGRvbm9ycy5nZXQocm9sZSk6CiAgICAgICAgICAgICMgT25lIGV4cG9zdXJlIHdpdGggbm8gcGxhdGUgc29sdXRpb246IGl0IGJvcnJvd3Mgb25lIGZyb20gYSBzb2x2ZWQKICAgICAgICAgICAgIyBleHBvc3VyZSBvZiB0aGUgc2FtZSBmaWVsZCwgYW5kIHRoZSBzb2x2ZWQgY29weSBpcyB3aGF0IGlzIG1lYXN1cmVkLgogICAgICAgICAgICBwYXRoID0gZml0c19maWxlX0IgaWYgcm9sZSA9PSAiQiIgZWxzZSBmaXRzX2ZpbGVfVgogICAgICAgICAgICBkYXRhX2MsIHdjc19jLCBfYnBtLCBpbmZvX2MgPSBsb2FkX2ZyYW1lKHBhdGgsIGRvbm9yc1tyb2xlXSkKICAgICAgICAgICAgb3V0ID0gb3MucGF0aC5qb2luKG91dF9kaXIsIGYie3NhZmV9X3tyb2xlfV9zb2x2ZWQuZml0cyIpCiAgICAgICAgICAgIHNhdmVfZnJhbWUob3V0LCBkYXRhX2MsIHdjc19jLCBpbmZvX2MsIGhpc3Rvcnk9KAogICAgICAgICAgICAgICAgIlBsYXRlIHNvbHV0aW9uIGZpdHRlZCBhZ2FpbnN0ICIgKyBvcy5wYXRoLmJhc2VuYW1lKGRvbm9yc1tyb2xlXSksKSkKICAgICAgICAgICAgY29tYmluZWRfbm90ZS5hcHBlbmQoZiJ7cm9sZX06IHBsYXRlIHNvbHV0aW9uIGJvcnJvd2VkIGZyb20gIgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBmIntvcy5wYXRoLmJhc2VuYW1lKGRvbm9yc1tyb2xlXSl9IikKICAgICAgICAgICAgaWYgcm9sZSA9PSAiQiI6CiAgICAgICAgICAgICAgICBmaXRzX2ZpbGVfQiA9IG91dAogICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgZml0c19maWxlX1YgPSBvdXQKICAgICAgICAgICAgY29udGludWUKICAgICAgICBpZiBub3QgcGF0aHM6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgZGF0YV9jLCB3Y3NfYywgX2JwbSwgaW5mb19jID0gY29tYmluZV9leHBvc3VyZXMocGF0aHMsIGRvbm9ycy5nZXQocm9sZSksIHJvbGUpCiAgICAgICAgb3V0ID0gb3MucGF0aC5qb2luKG91dF9kaXIsIGYie3NhZmV9X3tyb2xlfV9jb21iaW5lZC5maXRzIikKICAgICAgICBzYXZlX2ZyYW1lKG91dCwgZGF0YV9jLCB3Y3NfYywgaW5mb19jLAogICAgICAgICAgICAgICAgICAgaGlzdG9yeT0oZiJNZWRpYW4gb2Yge2xlbihwYXRocyl9IGV4cG9zdXJlcyIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAiQ29tYmluZWQgYnkgQmx1ZV9DbHVzdGVyc19Gcm9tX0ltYWdlcy5weSIpKQogICAgICAgIGNvbWJpbmVkX25vdGUuYXBwZW5kKGYie3JvbGV9OiBtZWRpYW4gb2Yge2xlbihwYXRocyl9IGV4cG9zdXJlcyIpCiAgICAgICAgaWYgcm9sZSA9PSAiQiI6CiAgICAgICAgICAgIGZpdHNfZmlsZV9CID0gb3V0CiAgICAgICAgZWxzZToKICAgICAgICAgICAgZml0c19maWxlX1YgPSBvdXQKICAgICAgICBkZWwgZGF0YV9jCgogICAgIyBBIGZpbGUgbmFtZWQgb24gdGhlIGNvbW1hbmQgbGluZSB0aGF0IGhhcyBubyBwbGF0ZSBzb2x1dGlvbiAtIHR5cGljYWxseSBhCiAgICAjIG1lZGlhbiB3cml0dGVuIGJ5IGFub3RoZXIgcHJvZ3JhbSwgd2hpY2gga2VwdCBubyBoZWFkZXIgLSBib3Jyb3dzIG9uZSBmcm9tCiAgICAjIGEgc29sdmVkIGV4cG9zdXJlIGluIGl0cyBvd24gZm9sZGVyLCBleGFjdGx5IGFzIGluIHRoZSBmb2xkZXIgbW9kZS4KICAgIGZvciByb2xlIGluICgiQiIsICJWIik6CiAgICAgICAgcGF0aCA9IGZpdHNfZmlsZV9CIGlmIHJvbGUgPT0gIkIiIGVsc2UgZml0c19maWxlX1YKICAgICAgICBpZiBncm91cHMuZ2V0KHJvbGUpIG9yIGRvbm9ycy5nZXQocm9sZSk6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgZG9ub3IgPSBkb25vcl9iZXNpZGUocGF0aCwgcm9sZSkKICAgICAgICBpZiBkb25vciBpcyBOb25lOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHByaW50KGYiW2lucHV0XSB7cm9sZX06IHtvcy5wYXRoLmJhc2VuYW1lKHBhdGgpfSBoYXMgbm8gcGxhdGUgc29sdXRpb247ICIKICAgICAgICAgICAgICBmImZpdHRpbmcgb25lIGFnYWluc3Qge29zLnBhdGguYmFzZW5hbWUoZG9ub3IpfSIsIGZsdXNoPVRydWUpCiAgICAgICAgZGF0YV9jLCB3Y3NfYywgX2JwbSwgaW5mb19jID0gbG9hZF9mcmFtZShwYXRoLCBkb25vcikKICAgICAgICBpZiB3Y3NfYyBpcyBOb25lOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgICMgVGhlIG51bWJlciBvZiBmcmFtZXMgY29tYmluZWQgYmVsb25ncyB0byB0aGlzIGZpbGUsIG5vdCB0byB0aGUgZG9ub3I7CiAgICAgICAgIyBpdCBpcyBrbm93biBvbmx5IGlmIHRoaXMgZmlsZSdzIG93biBoZWFkZXIgc2F5cyBpdC4KICAgICAgICBuX2NvbWIsIGhvdyA9IGNvbWJpbmVkX2NvdW50KGZpdHMuZ2V0aGVhZGVyKHBhdGgpKQogICAgICAgIGluZm9fY1sibmNvbWJpbmUiXSwgaW5mb19jWyJjb21idHlwZSJdID0gbl9jb21iLCAoIiIgaWYgbl9jb21iID09IDEgZWxzZSBob3cpCiAgICAgICAgb3V0ID0gb3MucGF0aC5qb2luKG91dF9kaXIsIGYie3NhZmV9X3tyb2xlfV9zb2x2ZWQuZml0cyIpCiAgICAgICAgc2F2ZV9mcmFtZShvdXQsIGRhdGFfYywgd2NzX2MsIGluZm9fYywgaGlzdG9yeT0oCiAgICAgICAgICAgICJQbGF0ZSBzb2x1dGlvbiBmaXR0ZWQgYWdhaW5zdCAiICsgb3MucGF0aC5iYXNlbmFtZShkb25vciksKSkKICAgICAgICBjb21iaW5lZF9ub3RlLmFwcGVuZChmIntyb2xlfTogcGxhdGUgc29sdXRpb24gYm9ycm93ZWQgZnJvbSAiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZiJ7b3MucGF0aC5iYXNlbmFtZShkb25vcil9IikKICAgICAgICBpZiByb2xlID09ICJCIjoKICAgICAgICAgICAgZml0c19maWxlX0IgPSBvdXQKICAgICAgICBlbHNlOgogICAgICAgICAgICBmaXRzX2ZpbGVfViA9IG91dAogICAgICAgIGRlbCBkYXRhX2MKCiAgICAjIC0tLS0gdGhlIHB1Ymxpc2hlZCBudW1iZXJzLCBzaG93biBhbmQgY29uZmlybWVkIGJlZm9yZSB1c2UgLS0tLS0tLS0tLS0tCiAgICBwcmludCgiTG9va2luZyB0aGUgZ2FsYXh5IHVwLi4uIiwgZmx1c2g9VHJ1ZSkKICAgIGRpc3RhbmNlID0gb2ZmZXJlZCgiRGlzdGFuY2UgKE1wYykiLCBsb29rdXBfZGlzdGFuY2Uob2JqX25hbWUpLAogICAgICAgICAgICAgICAgICAgICAgICJDb3NtaWNmbG93cy0zLCBUdWxseSBldCBhbC4gMjAxNiIsICIgTXBjIiwgQVJHUy5kaXN0YW5jZV9tcGMpCiAgICBFX0JWID0gb2ZmZXJlZCgiR2FsYWN0aWMgY29sb3VyIGV4Y2VzcyBFKEItVikiLCBsb29rdXBfZXh0aW5jdGlvbihvYmpfbmFtZSksCiAgICAgICAgICAgICAgICAgICAiU2NobGFmbHkgJiBGaW5rYmVpbmVyIDIwMTEsIHZpYSBJUlNBIiwgIiIsIEFSR1MuZWJ2KQogICAgaWYgZGlzdGFuY2UgPD0gMCBvciBFX0JWIDwgMDoKICAgICAgICBzdG9wKCJ0aGUgZGlzdGFuY2UgbXVzdCBiZSBwb3NpdGl2ZSBhbmQgRShCLVYpIG5vdCBuZWdhdGl2ZSIpCiAgICBBX1YgPSBSX1YgKiBFX0JWCiAgICBtdSA9IDUuMCAqIG5wLmxvZzEwKGRpc3RhbmNlICogMS4wZTYpIC0gNS4wCiAgICBwcmludChmIlVzaW5nIGRpc3RhbmNlIHtkaXN0YW5jZTouMmZ9IE1wYyAoZGlzdGFuY2UgbW9kdWx1cyB7bXU6LjJmfSksIgogICAgICAgICAgZiIgRShCLVYpID0ge0VfQlY6LjRmfSwgQV9WID0ge0FfVjouM2Z9IiwgZmx1c2g9VHJ1ZSkKICAgIGZvciBsaW5lIGluIGNvbWJpbmVkX25vdGU6CiAgICAgICAgcHJpbnQoIltpbnB1dF0gIiArIGxpbmUsIGZsdXNoPVRydWUpCgogICAgIyAtLS0tIGZyYW1lcyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQogICAgcGF0aF9CLCByYXdfQiwgZGF0YV9CLCBoZHJfQiA9IHN1YnRyYWN0X2JhY2tncm91bmRfYW5kX3NhdmUoZml0c19maWxlX0IsIG91dF9kaXIpCiAgICBwYXRoX1YsIHJhd19WLCBkYXRhX1YsIGhkcl9WID0gc3VidHJhY3RfYmFja2dyb3VuZF9hbmRfc2F2ZShmaXRzX2ZpbGVfViwgb3V0X2RpcikKICAgIHNoYXBlX1YgPSBkYXRhX1Yuc2hhcGUKICAgIHdjc19WID0gV0NTKGhkcl9WKS5jZWxlc3RpYWwKICAgIGlmIG5vdCB3Y3NfVi5oYXNfY2VsZXN0aWFsOgogICAgICAgIHN0b3AoInRoZSBWIGZyYW1lIGhhcyBubyBwbGF0ZSBzb2x1dGlvbjsgcGxhdGUtc29sdmUgaXQgYW5kIHJlcnVuIikKICAgIHNjYWxlX2FyY3NlYyA9IDM2MDAuMCAqIGZsb2F0KG5wLnNxcnQoYWJzKG5wLmxpbmFsZy5kZXQod2NzX1YucGl4ZWxfc2NhbGVfbWF0cml4KSkpKQoKICAgIGluZm8gPSB7fQogICAgZm9yIGJhbmQsIHJhdywgaGRyIGluICgoIkIiLCByYXdfQiwgaGRyX0IpLCAoIlYiLCByYXdfViwgaGRyX1YpKToKICAgICAgICBzYXQsIHNhdF9zcmMgPSBzYXR1cmF0aW9uX2xldmVsKHJhdywgaGRyKQogICAgICAgIGdhaW4sIGdhaW5fc3JjID0gZGV0ZWN0b3JfZ2FpbihoZHIpCiAgICAgICAgcm4sIHJuX3NyYyA9IHJlYWRfbm9pc2UoaGRyKQogICAgICAgIGluZm9bYmFuZF0gPSB7InNhdCI6IHNhdCwgImdhaW4iOiBnYWluLCAicm4iOiBybn0KICAgICAgICBwcmludChmIiAgIHtiYW5kfTogc2F0dXJhdGlvbiBsZXZlbCB7c2F0Oi4wZn0gQURVICh7c2F0X3NyY30pOyBnYWluICIKICAgICAgICAgICAgICBmIntnYWluOi4yZn0gZS0vQURVICh7Z2Fpbl9zcmN9KTsgcmVhZCBub2lzZSB7cm46Z30gZS0gKHtybl9zcmN9KSIsCiAgICAgICAgICAgICAgZmx1c2g9VHJ1ZSkKCiAgICAjIFdoZXJlIHRoZSBnYWxheHkgaXMgYW5kIGhvdyBmYXIgaXQgcmVhY2hlcywgZm9yIHRoZSBkZXRlY3Rpb24gd2luZG93LCB0aGUKICAgICMgY2FsaWJyYXRpb24gc3RhcnMgYW5kIHRoZSB0cmltbWluZyBvZiB0aGUgY2F0YWxvZ3VlLgogICAgY2VudHJlX3B4ID0gZ2FsYXh5X2NlbnRyZV9waXhlbChvYmpfbmFtZSwgd2NzX1YsIHNoYXBlX1YpCiAgICByMjUgPSBOb25lCiAgICBnZW9tID0gTm9uZQogICAgd2luZG93ID0gTm9uZQogICAgdHJ5OgogICAgICAgIGdlb20gPSBoeXBlcmxlZGFfZ2VvbWV0cnkob2JqX25hbWUpCiAgICAgICAgaWYgZ2VvbSBhbmQgZ2VvbS5nZXQoImQyNV9hcmNtaW4iKToKICAgICAgICAgICAgcjI1ID0gMC41ICogZ2VvbVsiZDI1X2FyY21pbiJdICogNjAuMCAvIHNjYWxlX2FyY3NlYwogICAgICAgICAgICBpZiBjZW50cmVfcHggaXMgbm90IE5vbmU6CiAgICAgICAgICAgICAgICB3aW5kb3cgPSBnYWxheHlfd2luZG93KHNoYXBlX1YsIGNlbnRyZV9weCwgcjI1LCBtYXJnaW49MS41KQogICAgICAgICAgICAgICAgaWYgd2luZG93IGlzIE5vbmU6CiAgICAgICAgICAgICAgICAgICAgcHJpbnQoIlt3aW5kb3ddIHRoZSBnYWxheHkgZmlsbHMgdGhpcyBmcmFtZSAtIHdvcmtpbmcgb24gYWxsIG9mIGl0IiwKICAgICAgICAgICAgICAgICAgICAgICAgICBmbHVzaD1UcnVlKQogICAgICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgICAgICB5MCwgeTEsIHgwLCB4MSA9IHdpbmRvdwogICAgICAgICAgICAgICAgICAgIHByaW50KGYiW3dpbmRvd10gdGhlIGdhbGF4eSBvY2N1cGllcyB7eDEgLSB4MH0geCB7eTEgLSB5MH0gb2YgIgogICAgICAgICAgICAgICAgICAgICAgICAgIGYie3NoYXBlX1ZbMV19IHgge3NoYXBlX1ZbMF19IHBpeGVsczsgdGhlIHJlc3QgaXMgc2t5IiwKICAgICAgICAgICAgICAgICAgICAgICAgICBmbHVzaD1UcnVlKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIHByaW50KCJbd2luZG93XSBubyBpc29waG90YWwgcmFkaXVzIGF2YWlsYWJsZSAtIHdvcmtpbmcgb24gdGhlIHdob2xlIGZyYW1lIiwKICAgICAgICAgICAgICAgICAgZmx1c2g9VHJ1ZSkKICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOgogICAgICAgIHByaW50KGYiW3dpbmRvd10gY291bGQgbm90IHdvcmsgb3V0IHdoZXJlIHRoZSBnYWxheHkgZW5kcyAoe2V4Y30pIC0gd29ya2luZyAiCiAgICAgICAgICAgICAgZiJvbiB0aGUgd2hvbGUgZnJhbWUiLCBmbHVzaD1UcnVlKQoKICAgICMgLS0tLSBkZXRlY3Rpb24gLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQogICAgcHJpbnQoIlJlZ2lzdGVyaW5nIEIgb250byBWLi4uIiwgZmx1c2g9VHJ1ZSkKICAgIGJfdG9fdiwgdl90b19iLCByZWdfdGV4dCA9IHJlZ2lzdGVyX0Jfb250b19WKGRhdGFfQiwgZGF0YV9WLCBwYXRoX0IsIHBhdGhfVikKICAgIHdpbmRvd19CID0gTm9uZQogICAgaWYgd2luZG93IGlzIG5vdCBOb25lOgogICAgICAgICMgVGhlIHNhbWUgcGFydCBvZiB0aGUgc2t5IGluIEIncyBvd24gZ3JpZCwgd2l0aCBhIG1hcmdpbiBmb3Igcm90YXRpb24uCiAgICAgICAgeTAsIHkxLCB4MCwgeDEgPSB3aW5kb3cKICAgICAgICBjb3JuZXJzID0gdl90b19iKG5wLmFycmF5KFtbeDAsIHkwXSwgW3gxLCB5MF0sIFt4MCwgeTFdLCBbeDEsIHkxXV0sIGZsb2F0KSkKICAgICAgICBieDAsIGJ5MCA9IG5wLmZsb29yKGNvcm5lcnMubWluKGF4aXM9MCkpLmFzdHlwZShpbnQpIC0gMjAKICAgICAgICBieDEsIGJ5MSA9IG5wLmNlaWwoY29ybmVycy5tYXgoYXhpcz0wKSkuYXN0eXBlKGludCkgKyAyMAogICAgICAgIHdpbmRvd19CID0gKG1heChieTAsIDApLCBtaW4oYnkxLCBkYXRhX0Iuc2hhcGVbMF0pLAogICAgICAgICAgICAgICAgICAgIG1heChieDAsIDApLCBtaW4oYngxLCBkYXRhX0Iuc2hhcGVbMV0pKQogICAgeEIsIHlCLCBmQiwgZndobV9CID0gZGV0ZWN0X3NvdXJjZXMoZGF0YV9CLCByYXdfQiwgIkIiLCBpbmZvWyJCIl1bInNhdCJdLCB3aW5kb3dfQikKICAgIHhWLCB5ViwgZlYsIGZ3aG1fViA9IGRldGVjdF9zb3VyY2VzKGRhdGFfViwgcmF3X1YsICJWIiwgaW5mb1siViJdWyJzYXQiXSwgd2luZG93KQogICAgZndobSA9IG1heChmd2htX0IsIGZ3aG1fVikKCiAgICAjIE9uZSBsaXN0IG9mIHNvdXJjZXMgb24gdGhlIFYgZ3JpZDogZXZlcnl0aGluZyBmb3VuZCBpbiBWLCBhbmQgd2hhdGV2ZXIgd2FzCiAgICAjIGZvdW5kIGluIEIgd2l0aCBub3RoaW5nIGluIFYgd2l0aGluIG9uZSBGV0hNIC0gYSB2ZXJ5IGJsdWUga25vdCBjYW4gYmUKICAgICMgY2xlYXIgaW4gQiBhbmQgbG9zdCBpbiBWLiBFYWNoIGlzIHRoZW4gbWVhc3VyZWQgaW4gYm90aCBmcmFtZXMgYXQgdGhlCiAgICAjIHNhbWUgcGxhY2Ugb24gdGhlIHNreSB0aHJvdWdoIHRoZSBzYW1lIGFwZXJ0dXJlLiBCZWZvcmUsIGVhY2ggYmFuZCB3YXMKICAgICMgbWVhc3VyZWQgd2hlcmUgaXQgZm91bmQgdGhlIHNvdXJjZSwgdGhyb3VnaCBhbiBhcGVydHVyZSBzaXplZCBmcm9tIHRoYXQKICAgICMgc291cmNlJ3Mgd2lkdGggaW4gdGhhdCBiYW5kOiB0aGUgQiBhbmQgViBhcGVydHVyZXMgb2Ygb25lIGtub3QgZGlmZmVyZWQsCiAgICAjIGFuZCBzbyBpdHMgY29sb3VyIGRlcGVuZGVkIG9uIGl0cyBzaXplIGFzIG11Y2ggYXMgb24gaXRzIGxpZ2h0LgogICAgeHlCX29uX1YgPSBiX3RvX3YobnAuY29sdW1uX3N0YWNrKFt4QiwgeUJdKSkgaWYgbGVuKHhCKSBlbHNlIG5wLmVtcHR5KCgwLCAyKSkKICAgIHh5ViA9IG5wLmNvbHVtbl9zdGFjayhbeFYsIHlWXSkgaWYgbGVuKHhWKSBlbHNlIG5wLmVtcHR5KCgwLCAyKSkKICAgIGlmIGxlbih4eVYpIGFuZCBsZW4oeHlCX29uX1YpOgogICAgICAgIGQsIF8gPSBjS0RUcmVlKHh5VikucXVlcnkoeHlCX29uX1YsIGs9MSkKICAgICAgICBiX29ubHkgPSBkID4gZndobQogICAgZWxzZToKICAgICAgICBiX29ubHkgPSBucC5vbmVzKGxlbih4eUJfb25fViksIGR0eXBlPWJvb2wpCiAgICB4eSA9IG5wLnZzdGFjayhbeHlWLCB4eUJfb25fVltiX29ubHldXSkKICAgIGZvdW5kX2luID0gbnAuYXJyYXkoWyJWIl0gKiBsZW4oeHlWKSArIFsiQiJdICogaW50KGJfb25seS5zdW0oKSksIGR0eXBlPSI8VTIiKQogICAgaWYgbGVuKHh5VikgYW5kIGxlbih4eUJfb25fVik6CiAgICAgICAgZFYsIF8gPSBjS0RUcmVlKHh5Ql9vbl9WKS5xdWVyeSh4eVYsIGs9MSkKICAgICAgICBmb3VuZF9pbls6bGVuKHh5VildW2RWIDw9IGZ3aG1dID0gIkJWIgoKICAgIHJfYXAgPSBBUEVSVFVSRV9TQ0FMRSAqIGZ3aG0KICAgIHJfaW4sIHJfb3V0ID0gQU5OVUxVU19JTk5FUl9TQ0FMRSAqIHJfYXAsIEFOTlVMVVNfT1VURVJfU0NBTEUgKiByX2FwCiAgICB4eV9pbl9CID0gdl90b19iKHh5KSBpZiBsZW4oeHkpIGVsc2UgeHkKICAgIGVkZ2UgPSByX291dCArIDIKICAgIGluc2lkZSA9ICgoeHlbOiwgMF0gPiBlZGdlKSAmICh4eVs6LCAwXSA8IHNoYXBlX1ZbMV0gLSBlZGdlKQogICAgICAgICAgICAgICYgKHh5WzosIDFdID4gZWRnZSkgJiAoeHlbOiwgMV0gPCBzaGFwZV9WWzBdIC0gZWRnZSkKICAgICAgICAgICAgICAmICh4eV9pbl9CWzosIDBdID4gZWRnZSkgJiAoeHlfaW5fQls6LCAwXSA8IGRhdGFfQi5zaGFwZVsxXSAtIGVkZ2UpCiAgICAgICAgICAgICAgJiAoeHlfaW5fQls6LCAxXSA+IGVkZ2UpICYgKHh5X2luX0JbOiwgMV0gPCBkYXRhX0Iuc2hhcGVbMF0gLSBlZGdlKSkKICAgIHh5LCB4eV9pbl9CLCBmb3VuZF9pbiA9IHh5W2luc2lkZV0sIHh5X2luX0JbaW5zaWRlXSwgZm91bmRfaW5baW5zaWRlXQogICAgaWYgbm90IGxlbih4eSk6CiAgICAgICAgc3RvcCgibm8gc291cmNlcyB3ZXJlIGZvdW5kIGluIGJvdGggZnJhbWVzIikKICAgIHByaW50KGYiTWVhc3VyaW5nIHtsZW4oeHkpOix9IHNvdXJjZXMgaW4gYm90aCBiYW5kcyB0aHJvdWdoIG9uZSBhcGVydHVyZSBvZiAiCiAgICAgICAgICBmIntyX2FwOi4xZn0gcHggKHtBUEVSVFVSRV9TQ0FMRTpnfSB4IHRoZSBGV0hNIG9mIHtmd2htOi4yZn0gcHg7IHNreSByaW5nICIKICAgICAgICAgIGYie3JfaW46LjFmfS17cl9vdXQ6LjFmfSBweCkgIFtmb3VuZCBpbiBib3RoOiB7aW50KChmb3VuZF9pbiA9PSAnQlYnKS5zdW0oKSk6LH0sIgogICAgICAgICAgZiIgViBvbmx5OiB7aW50KChmb3VuZF9pbiA9PSAnVicpLnN1bSgpKTosfSwgQiBvbmx5OiAiCiAgICAgICAgICBmIntpbnQoKGZvdW5kX2luID09ICdCJykuc3VtKCkpOix9XSIsIGZsdXNoPVRydWUpCgogICAgZmx1eFYsIGVyclYsIHBlYWtWLCBza3lWID0gYXBlcnR1cmVfcGhvdG9tZXRyeV9hdCgKICAgICAgICBkYXRhX1YsIHJhd19WLCB4eSwgcl9hcCwgcl9pbiwgcl9vdXQsIGluZm9bIlYiXVsiZ2FpbiJdLCBpbmZvWyJWIl1bInJuIl0pCiAgICBmbHV4QiwgZXJyQiwgcGVha0IsIHNreUIgPSBhcGVydHVyZV9waG90b21ldHJ5X2F0KAogICAgICAgIGRhdGFfQiwgcmF3X0IsIHh5X2luX0IsIHJfYXAsIHJfaW4sIHJfb3V0LCBpbmZvWyJCIl1bImdhaW4iXSwgaW5mb1siQiJdWyJybiJdKQogICAgbWFnViwgbWFnVl9lcnIgPSBpbnN0cnVtZW50YWwoZmx1eFYsIGVyclYpCiAgICBtYWdCLCBtYWdCX2VyciA9IGluc3RydW1lbnRhbChmbHV4QiwgZXJyQikKICAgIHdpZHRoViA9IG5wLmFycmF5KFtzdGFyX3dpZHRoKGRhdGFfViwgeCwgeSwgRldITV9XSU5ET1dfU0laRSwgcykKICAgICAgICAgICAgICAgICAgICAgICBmb3IgKHgsIHkpLCBzIGluIHppcCh4eSwgc2t5VildKQogICAgd2lkdGhCID0gbnAuYXJyYXkoW3N0YXJfd2lkdGgoZGF0YV9CLCB4LCB5LCBGV0hNX1dJTkRPV19TSVpFLCBzKQogICAgICAgICAgICAgICAgICAgICAgIGZvciAoeCwgeSksIHMgaW4gemlwKHh5X2luX0IsIHNreUIpXSkKICAgIHJhLCBkZWMgPSB3Y3NfVi5hbGxfcGl4MndvcmxkKHh5WzosIDBdLCB4eVs6LCAxXSwgMCkKCiAgICBkZiA9IHBkLkRhdGFGcmFtZSh7CiAgICAgICAgIlhfQiI6IHh5X2luX0JbOiwgMF0sICJZX0IiOiB4eV9pbl9CWzosIDFdLCAiRldITV9CIjogd2lkdGhCLAogICAgICAgICJGbHV4X0IiOiBmbHV4QiwgIkZsdXhfQl9lcnIiOiBlcnJCLAogICAgICAgICJYX1YiOiB4eVs6LCAwXSwgIllfViI6IHh5WzosIDFdLCAiRldITV9WIjogd2lkdGhWLAogICAgICAgICJGbHV4X1YiOiBmbHV4ViwgIkZsdXhfVl9lcnIiOiBlcnJWLAogICAgICAgICJNYWdfaW5zdF9CIjogbWFnQiwgIk1hZ19pbnN0X0JfZXJyIjogbWFnQl9lcnIsCiAgICAgICAgIk1hZ19pbnN0X1YiOiBtYWdWLCAiTWFnX2luc3RfVl9lcnIiOiBtYWdWX2VyciwKICAgICAgICAiUGVha19CIjogcGVha0IsICJQZWFrX1YiOiBwZWFrViwKICAgICAgICAiU2F0dXJhdGVkIjogKHBlYWtCID4gaW5mb1siQiJdWyJzYXQiXSkgfCAocGVha1YgPiBpbmZvWyJWIl1bInNhdCJdKSwKICAgICAgICAiRm91bmRfaW4iOiBmb3VuZF9pbiwgIkFwZXJ0dXJlX1JhZGl1cyI6IHJfYXAsCiAgICB9KQogICAgbl9hbGwgPSBsZW4oZGYpCiAgICBkZiA9IGRmWyhkZlsiRmx1eF9CIl0gPiAwKSAmIChkZlsiRmx1eF9WIl0gPiAwKV0ucmVzZXRfaW5kZXgoZHJvcD1UcnVlKQogICAgcHJpbnQoZiIgICB7bGVuKGRmKTosfSBvZiB7bl9hbGw6LH0gaGF2ZSBwb3NpdGl2ZSBmbHV4IGluIGJvdGggYmFuZHM7ICIKICAgICAgICAgIGYie2ludChkZlsnU2F0dXJhdGVkJ10uc3VtKCkpOix9IG9mIHRoZW0gc2F0dXJhdGVkIiwgZmx1c2g9VHJ1ZSkKICAgIGlmIGxlbihkZikgPCAyOgogICAgICAgIHN0b3AoImZld2VyIHRoYW4gdHdvIHNvdXJjZXMgaGF2ZSBwb3NpdGl2ZSBmbHV4IGluIGJvdGggYmFuZHMiKQogICAgbmVhcmVzdCA9IGNLRFRyZWUoZGZbWyJYX1YiLCAiWV9WIl1dLnZhbHVlcykucXVlcnkoZGZbWyJYX1YiLCAiWV9WIl1dLnZhbHVlcywgaz0yKVswXVs6LCAxXQogICAgZGZbIk5lYXJlc3RfcHgiXSA9IG5lYXJlc3QKCiAgICAjIC0tLS0gdGhlIHBsYXRlIHNvbHV0aW9uLCBjaGVja2VkLCB0aGVuIHRoZSBza3kgcG9zaXRpb25zIC0tLS0tLS0tLS0tLS0tLQogICAgYnJpZ2h0ID0gZGZbfmRmWyJTYXR1cmF0ZWQiXV0uc29ydF92YWx1ZXMoIkZsdXhfViIsIGFzY2VuZGluZz1GYWxzZSkKICAgIHdjc19WLCB3Y3Nfbm90ZSA9IHZlcmlmeV9wbGF0ZV9zb2x1dGlvbihicmlnaHRbWyJYX1YiLCAiWV9WIl1dLnRvX251bXB5KGZsb2F0KSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBkZltbIlhfViIsICJZX1YiXV0udG9fbnVtcHkoZmxvYXQpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHdjc19WLCBzaGFwZV9WKQogICAgcmEsIGRlYyA9IHdjc19WLmFsbF9waXgyd29ybGQoZGZbIlhfViJdLnZhbHVlcywgZGZbIllfViJdLnZhbHVlcywgMCkKICAgIGRmWyJSQSJdLCBkZlsiREVDIl0gPSByYSwgZGVjCiAgICBpZiAiUkVQQUlSRUQiIGluIHdjc19ub3RlOgogICAgICAgICMgVGhlIGdhbGF4eSdzIGNlbnRyZSB3YXMgcGxhY2VkIHRocm91Z2ggdGhlIG9sZCBzb2x1dGlvbi4KICAgICAgICBjZW50cmVfcHggPSBnYWxheHlfY2VudHJlX3BpeGVsKG9ial9uYW1lLCB3Y3NfViwgc2hhcGVfVikKICAgICAgICAjIEFuZCB0aGUgViBmcmFtZSB3cml0dGVuIHRvIHRoZSBvdXRwdXQgY2FycmllcyB0aGUgcmVwYWlyZWQgc29sdXRpb24sIHNvCiAgICAgICAgIyB0aGF0IGEgdG9vbCByZWFkaW5nIHBvc2l0aW9ucyB0aHJvdWdoIGl0IGxhdGVyICh0aGUgY29tcGFyaXNvbiB3aXRoIHRoZQogICAgICAgICMgSElJIHJlZ2lvbnMpIHB1dHMgdGhlbSB3aGVyZSB0aGV5IGFyZS4KICAgICAgICB3aXRoIGZpdHMub3BlbihwYXRoX1YsIG1vZGU9InVwZGF0ZSIpIGFzIGhkdWw6CiAgICAgICAgICAgIGggPSBoZHVsWzBdLmhlYWRlcgogICAgICAgICAgICBmb3Iga2V5IGluIGxpc3QoaC5rZXlzKCkpOgogICAgICAgICAgICAgICAgaWYgcmUubWF0Y2gociJeKENUWVBFfENSVkFMfENSUElYfENERUxUfENVTklUfENEXGRfXGR8UENcZF9cZHwiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICByIlBWXGRfXGQrfEFffEJffEFQX3xCUF98TE9OUE9MRXxMQVRQT0xFfFdDU0FYRVN8IgogICAgICAgICAgICAgICAgICAgICAgICAgICAgciJSQURFU1lTfEVRVUlOT1gpIiwga2V5KToKICAgICAgICAgICAgICAgICAgICBoLnJlbW92ZShrZXksIGlnbm9yZV9taXNzaW5nPVRydWUsIHJlbW92ZV9hbGw9VHJ1ZSkKICAgICAgICAgICAgaC51cGRhdGUod2NzX1YudG9faGVhZGVyKHJlbGF4PVRydWUpKQogICAgICAgICAgICBoWyJISVNUT1JZIl0gPSAiUGxhdGUgc29sdXRpb24gcmVwYWlyZWQgYWdhaW5zdCBHYWlhIERSMyBieSBCbHVlX0NsdXN0ZXJzIgoKICAgIGNzdl9maWxlbmFtZSA9IG9zLnBhdGguam9pbihvdXRfZGlyLCBmIntzYWZlfV9waG90b21ldHJ5X3Jlc3VsdHMuY3N2IikKICAgIGRmLnRvX2Nzdihjc3ZfZmlsZW5hbWUsIGluZGV4PUZhbHNlKQogICAgcHJpbnQoZiJJbnN0cnVtZW50YWwgcGhvdG9tZXRyeSBzYXZlZCB0byB7Y3N2X2ZpbGVuYW1lfSIpCgogICAgIyAtLS0tIGNhbGlicmF0aW9uIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCiAgICBjYWwsIGNhdGFsb2d1ZSwgY2FsX3N0YXRzLCBtYXRjaGVkID0gY2FsaWJyYXRlKAogICAgICAgIGRmLCB3Y3NfViwgc2hhcGVfViwgQVJHUy5jYXRhbG9nLCBjZW50cmVfcHgsIHIyNSwgb3V0X2Rpciwgc2FmZSkKICAgIGRmID0gYXBwbHlfY2FsaWJyYXRpb24oZGYsIGNhbCkKCiAgICAjIFRoZSBkdXN0IG9mIG91ciBvd24gZ2FsYXh5IHRha2VuIG91dC4gQV9WIGRpbXMgVjsgQiBpcyBkaW1tZWQgYnkKICAgICMgQV9WICsgRShCLVYpLCB3aGljaCBpcyB3aGF0IHRoZSBjb2xvdXIgZXhjZXNzIG1lYW5zLgogICAgZGZbIk1hZ19CIl0gPSBkZlsiTWFnX0Jfb2JzIl0gLSAoQV9WICsgRV9CVikKICAgIGRmWyJNYWdfViJdID0gZGZbIk1hZ19WX29icyJdIC0gQV9WCiAgICBkZlsiTV9WIl0gPSBkZlsiTWFnX1YiXSAtIG11CiAgICBkZlsiWlBfVl91c2VkIl0sIGRmWyJrX1ZfdXNlZCJdID0gY2FsWyJaUF9WIl0sIGNhbFsia19WIl0KICAgIGRmWyJDb2xvdXJfYV91c2VkIl0sIGRmWyJDb2xvdXJfY191c2VkIl0gPSBjYWxbImEiXSwgY2FsWyJjIl0KICAgIGRmWyJOX3JlZl91c2VkIl0gPSBjYWxbIk5fdXNlZCJdCiAgICBkZlsiQ2FsaWJyYXRpb25fY2F0YWxvZ3VlIl0gPSBjYXRhbG9ndWUKICAgIGRmWyJBX1ZfdXNlZCJdLCBkZlsiRV9CVl91c2VkIl0gPSBBX1YsIEVfQlYKICAgIGRmWyJEaXN0YW5jZV9NcGNfdXNlZCJdLCBkZlsiRGlzdGFuY2VfbW9kdWx1c191c2VkIl0gPSBkaXN0YW5jZSwgbXUKICAgIGRmWyJGV0hNX21lYXN1cmVkX0IiXSwgZGZbIkZXSE1fbWVhc3VyZWRfViJdID0gZndobV9CLCBmd2htX1YKCiAgICBjYWxpYl9maWxlID0gb3MucGF0aC5qb2luKG91dF9kaXIsIGYie3NhZmV9X2NhbGlicmF0ZWRfcGhvdG9tZXRyeS5jc3YiKQogICAgZGYudG9fY3N2KGNhbGliX2ZpbGUsIGluZGV4PUZhbHNlKQogICAgcHJpbnQoZiJDYWxpYnJhdGVkIHBob3RvbWV0cnkgc2F2ZWQgdG8ge2NhbGliX2ZpbGV9IikKCiAgICBzb3VyY2VfbWFwKGRhdGFfViwgeHlbOiwgMF0sIHh5WzosIDFdLCBmIntvYmpfbmFtZX0gLSBWIGJhbmQgd2l0aCBtZWFzdXJlZCBzb3VyY2VzIiwKICAgICAgICAgICAgICAgb3MucGF0aC5qb2luKG91dF9kaXIsIGYie3NhZmV9X1Zfc291cmNlcy5wbmciKSkKICAgIHNvdXJjZV9tYXAoZGF0YV9CLCB4eV9pbl9CWzosIDBdLCB4eV9pbl9CWzosIDFdLAogICAgICAgICAgICAgICBmIntvYmpfbmFtZX0gLSBCIGJhbmQgd2l0aCBtZWFzdXJlZCBzb3VyY2VzIiwKICAgICAgICAgICAgICAgb3MucGF0aC5qb2luKG91dF9kaXIsIGYie3NhZmV9X0Jfc291cmNlcy5wbmciKSwgY29sb3VyPSJibHVlIikKCiAgICAjIC0tLS0gZm9yZWdyb3VuZCBzdGFycyBvdXQgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCiAgICAjIEV2ZXJ5IHNvdXJjZSB0aGF0IGlzIGEgY2F0YWxvZ3VlIHN0YXIgKG1hdGNoZWQgd2l0aGluIENBTElCX01BVENIX1JBRElVUyksCiAgICAjIGFuZCBldmVyeSBzb3VyY2Ugb24gYSBHYWlhIHN0YXIgd2l0aCBhIHNpZ25pZmljYW50IHBhcmFsbGF4IG9yIHByb3BlcgogICAgIyBtb3Rpb24sIGlzIGluIGZyb250IG9mIHRoZSBnYWxheHkuCiAgICBpc19zdGFyID0gbnAuemVyb3MobGVuKGRmKSwgZHR5cGU9Ym9vbCkKICAgIGlmIGxlbihtYXRjaGVkKToKICAgICAgICBpc19zdGFyW21hdGNoZWRbInJvdyJdLmFzdHlwZShpbnQpLnZhbHVlc10gPSBUcnVlCiAgICBuX2NhdF9zdGFycyA9IGludChpc19zdGFyLnN1bSgpKQogICAgbl9nYWlhID0gMAogICAgdHJ5OgogICAgICAgIGZnLCBfZmdkaWFnID0gZ2FpYV9mb3JlZ3JvdW5kX3N0YXJzKHdjc19WLCBzaGFwZV9WKQogICAgICAgIGlmIGZnIGlzIG5vdCBOb25lIGFuZCBsZW4oZmcpOgogICAgICAgICAgICBkLCBfID0gY0tEVHJlZShucC5hc2FycmF5KGZnLCBmbG9hdCkpLnF1ZXJ5KGRmW1siWF9WIiwgIllfViJdXS52YWx1ZXMsIGs9MSkKICAgICAgICAgICAgb25fZmcgPSAoZCA8PSBGT1JFR1JPVU5EX1RPTCkgJiB+aXNfc3RhcgogICAgICAgICAgICBuX2dhaWEgPSBpbnQob25fZmcuc3VtKCkpCiAgICAgICAgICAgIGlzX3N0YXIgfD0gb25fZmcKICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOgogICAgICAgIHByaW50KGYiICAgIGNvdWxkIG5vdCBjaGVjayBhZ2FpbnN0IEdhaWEgKHtleGN9KTsgZm9yZWdyb3VuZCBzdGFycyBvdGhlciB0aGFuICIKICAgICAgICAgICAgICBmInRoZSBjYXRhbG9ndWUgc3RhcnMgcmVtYWluIGluIHRoZSBjYXRhbG9ndWUiLCBmbHVzaD1UcnVlKQogICAgZGZfY2xlYW4gPSBkZlt+aXNfc3Rhcl0ucmVzZXRfaW5kZXgoZHJvcD1UcnVlKQogICAgcHJpbnQoZiIgICAge25fY2F0X3N0YXJzfSBzb3VyY2VzIGFyZSBjYXRhbG9ndWUgc3RhcnMgYW5kIHtuX2dhaWF9IG1vcmUgc2l0IG9uIGEgIgogICAgICAgICAgZiJHYWlhIGZvcmVncm91bmQgc3RhciAocGFyYWxsYXggb3IgcHJvcGVyIG1vdGlvbiBzaWduaWZpY2FudCk7ICIKICAgICAgICAgIGYie2xlbihkZl9jbGVhbil9IGxlZnQiLCBmbHVzaD1UcnVlKQogICAgbl9vdXRzaWRlID0gMAogICAgaWYgcjI1IGFuZCBjZW50cmVfcHggaXMgbm90IE5vbmU6CiAgICAgICAgZCA9IG5wLmh5cG90KGRmX2NsZWFuWyJYX1YiXSAtIGNlbnRyZV9weFswXSwgZGZfY2xlYW5bIllfViJdIC0gY2VudHJlX3B4WzFdKQogICAgICAgIG5fb3V0c2lkZSA9IGludCgoZCA+IHIyNSkuc3VtKCkpCiAgICAgICAgZGZfY2xlYW4gPSBkZl9jbGVhblsoZCA8PSByMjUpLnZhbHVlc10ucmVzZXRfaW5kZXgoZHJvcD1UcnVlKQogICAgICAgIHByaW50KGYiICAgIHtuX291dHNpZGV9IHNvdXJjZXMgbGllIGJleW9uZCBSMjUgKHtyMjU6LjBmfSBweCksIG91dHNpZGUgdGhlICIKICAgICAgICAgICAgICBmImdhbGF4eSwgYW5kIGFyZSBkcm9wcGVkOyB7bGVuKGRmX2NsZWFuKX0gbGVmdCIsIGZsdXNoPVRydWUpCiAgICBlbHNlOgogICAgICAgIHByaW50KCIgICAgbm8gZ2FsYXh5IHJhZGl1cyBhdmFpbGFibGU7IHRoZSBjYXRhbG9ndWUgaXMgbm90IHRyaW1tZWQgdG8gdGhlICIKICAgICAgICAgICAgICAiZ2FsYXh5IiwgZmx1c2g9VHJ1ZSkKCiAgICBvdXRfZmlsZSA9IGNhbGliX2ZpbGUucmVwbGFjZSgiX2NhbGlicmF0ZWRfcGhvdG9tZXRyeS5jc3YiLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIl9jYWxpYnJhdGVkX3Bob3RvbWV0cnlfbm9fc3RhcnMuY3N2IikKICAgIGRmX2NsZWFuLnRvX2NzdihvdXRfZmlsZSwgaW5kZXg9RmFsc2UpCiAgICBwcmludChmIkNsZWFuZWQgZmlsZSBzYXZlZCB0byB7b3V0X2ZpbGV9IikKICAgIHNvdXJjZV9tYXAoZGF0YV9WLCBkZl9jbGVhblsiWF9WIl0sIGRmX2NsZWFuWyJZX1YiXSwKICAgICAgICAgICAgICAgZiJ7b2JqX25hbWV9IC0gViBiYW5kIHdpdGggZGV0ZWN0ZWQgc291cmNlcyAobm8gc3RhcnMpIiwKICAgICAgICAgICAgICAgb3MucGF0aC5qb2luKG91dF9kaXIsIGYie3NhZmV9X1Zfc291cmNlc19ub19zdGFycy5wbmciKSwKICAgICAgICAgICAgICAgbGFiZWw9IkNsZWFuZWQgc291cmNlcyAobm8gc3RhcnMpIikKCiAgICBtYWtlX2NtZChkZiwgY2FsaWJfZmlsZSwgIihhbGwgc291cmNlcykiLCBvYmpfbmFtZSwgbXUpCiAgICBtYWtlX2NtZChkZl9jbGVhbiwgb3V0X2ZpbGUsICIobm8gc3RhcnMpIiwgb2JqX25hbWUsIG11LAogICAgICAgICAgICAgd2luZG93PShDTURfQ09MT1JfTUlOLCBDTURfQ09MT1JfTUFYKSkKCiAgICAjIC0tLS0gdGhlIGJsdWUga25vdHMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KICAgIGJ2MCA9IGRmX2NsZWFuWyJNYWdfQiJdIC0gZGZfY2xlYW5bIk1hZ19WIl0KICAgIGluX3dpbmRvdyA9IChidjAgPj0gQ01EX0NPTE9SX01JTikgJiAoYnYwIDw9IENNRF9DT0xPUl9NQVgpCiAgICBwcmVjaXNlID0gZGZfY2xlYW5bIkJWX2VyciJdIDw9IEFSR1MubWF4X2J2X2VycgogICAgdW5zYXQgPSB+ZGZfY2xlYW5bIlNhdHVyYXRlZCJdLmFzdHlwZShib29sKQogICAgZGZfY29sb3JfZmlsdGVyZWQgPSBkZl9jbGVhbltpbl93aW5kb3cgJiBwcmVjaXNlICYgdW5zYXRdLmNvcHkoKQogICAgZGZfY29sb3JfZmlsdGVyZWRbIkItViJdID0gYnYwW2luX3dpbmRvdyAmIHByZWNpc2UgJiB1bnNhdF0KICAgIGRmX2NvbG9yX2ZpbHRlcmVkID0gZGZfY29sb3JfZmlsdGVyZWQucmVzZXRfaW5kZXgoZHJvcD1UcnVlKQogICAgcHJpbnQoZiJCbHVlIGtub3RzOiB7bGVuKGRmX2NvbG9yX2ZpbHRlcmVkKX0gc291cmNlcyB3aXRoIHtDTURfQ09MT1JfTUlOfSA8PSAiCiAgICAgICAgICBmIihCLVYpMCA8PSB7Q01EX0NPTE9SX01BWH0gYW5kIGEgQi1WIGVycm9yIG9mIGF0IG1vc3Qge0FSR1MubWF4X2J2X2Vycn0gbWFnIgogICAgICAgICAgZiIgICh7aW50KChpbl93aW5kb3cgJiB+cHJlY2lzZSkuc3VtKCkpfSBtb3JlIGluIHRoZSBjb2xvdXIgd2luZG93IHdlcmUgdG9vICIKICAgICAgICAgIGYibm9pc3kgdG8gY2xhc3NpZnksIHtpbnQoKGluX3dpbmRvdyAmIH51bnNhdCkuc3VtKCkpfSBzYXR1cmF0ZWQpIiwgZmx1c2g9VHJ1ZSkKICAgIG91dF9jc3ZfY29sb3IgPSBvcy5wYXRoLmpvaW4ob3V0X2RpciwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZiJ7c2FmZX1fY2FsaWJyYXRlZF9waG90b21ldHJ5X25vX3N0YXJzX2NvbG9yX2ZpbHRlcmVkLmNzdiIpCiAgICBkZl9jb2xvcl9maWx0ZXJlZC50b19jc3Yob3V0X2Nzdl9jb2xvciwgaW5kZXg9RmFsc2UpCiAgICBwcmludChmIkNvbG9yLWZpbHRlcmVkIHBob3RvbWV0cnkgc2F2ZWQgdG8ge291dF9jc3ZfY29sb3J9IikKICAgIHNvdXJjZV9tYXAoZGF0YV9WLCBkZl9jb2xvcl9maWx0ZXJlZFsiWF9WIl0sIGRmX2NvbG9yX2ZpbHRlcmVkWyJZX1YiXSwKICAgICAgICAgICAgICAgZiJ7b2JqX25hbWV9IC0gViBiYW5kIHdpdGggdGhlIGJsdWUga25vdHMiLAogICAgICAgICAgICAgICBvcy5wYXRoLmpvaW4ob3V0X2RpciwgZiJ7c2FmZX1fVl9zb3VyY2VzX2NvbG9yX2ZpbHRlcmVkLnBuZyIpLAogICAgICAgICAgICAgICBsYWJlbD1mIkJsdWUga25vdHMgKHtDTURfQ09MT1JfTUlOfSA8PSAoQi1WKTAgPD0ge0NNRF9DT0xPUl9NQVh9KSIpCiAgICBtYWtlX2NtZChkZl9jb2xvcl9maWx0ZXJlZCwgb3V0X2Nzdl9jb2xvciwgIihibHVlIGtub3RzKSIsIG9ial9uYW1lLCBtdSwKICAgICAgICAgICAgIHdpbmRvdz0oQ01EX0NPTE9SX01JTiwgQ01EX0NPTE9SX01BWCkpCgogICAgd3JpdGVfc3VtbWFyeShvcy5wYXRoLmpvaW4ob3V0X2RpciwgInJ1bl9zdW1tYXJ5LnR4dCIpLCBsb2NhbHMoKSkKCiAgICBpZiBkZl9jb2xvcl9maWx0ZXJlZC5lbXB0eToKICAgICAgICBwcmludCgiTm8gY29sb3ItZmlsdGVyZWQgc291cmNlcyBhdmFpbGFibGUgdG8gY29tcHV0ZSByYWRpYWwgZGlzdGFuY2VzLiIpCiAgICAgICAgcmV0dXJuIDAKCiAgICAjIC0tLS0gcmFkaWFsIHByb2ZpbGUgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KICAgIGlmIGNlbnRyZV9weCBpcyBOb25lOgogICAgICAgIGNlbnRyZV9weCA9IChzaGFwZV9WWzFdIC8gMi4wLCBzaGFwZV9WWzBdIC8gMi4wKQogICAgICAgIHByaW50KCJbcHJvZmlsZV0gdGhlIGdhbGF4eSdzIGNlbnRyZSBpcyB1bmtub3duIC0gdXNpbmcgdGhlIGZyYW1lJ3MgY2VudHJlIikKICAgIHhfY2VudGVyLCB5X2NlbnRlciA9IGNlbnRyZV9weAogICAgZGZfY29sb3JfZmlsdGVyZWRbIlJhZGlhbF9EaXN0YW5jZV9weCJdID0gbnAuaHlwb3QoCiAgICAgICAgZGZfY29sb3JfZmlsdGVyZWRbIlhfViJdIC0geF9jZW50ZXIsIGRmX2NvbG9yX2ZpbHRlcmVkWyJZX1YiXSAtIHlfY2VudGVyKQogICAgcHhfdG9fcGMgPSBucC5kZWcycmFkKHNjYWxlX2FyY3NlYyAvIDM2MDAuMCkgKiBkaXN0YW5jZSAqIDEuMGU2CiAgICBkZl9jb2xvcl9maWx0ZXJlZFsiUmFkaWFsX0Rpc3RhbmNlX3BjIl0gPSBkZl9jb2xvcl9maWx0ZXJlZFsiUmFkaWFsX0Rpc3RhbmNlX3B4Il0gKiBweF90b19wYwogICAgZGZfY29sb3JfZmlsdGVyZWQgPSBkZl9jb2xvcl9maWx0ZXJlZC5zb3J0X3ZhbHVlcygiUmFkaWFsX0Rpc3RhbmNlX3B4IikucmVzZXRfaW5kZXgoZHJvcD1UcnVlKQogICAgb3V0X2Nzdl9jb2xvcl9yYWQgPSBvcy5wYXRoLmpvaW4oCiAgICAgICAgb3V0X2RpciwgZiJ7c2FmZX1fY2FsaWJyYXRlZF9waG90b21ldHJ5X25vX3N0YXJzX2NvbG9yX2ZpbHRlcmVkX3dpdGhfcmFkaXVzLmNzdiIpCiAgICBkZl9jb2xvcl9maWx0ZXJlZC50b19jc3Yob3V0X2Nzdl9jb2xvcl9yYWQsIGluZGV4PUZhbHNlKQogICAgcHJpbnQoZiJDb2xvci1maWx0ZXJlZCBwaG90b21ldHJ5IHdpdGggcmFkaWFsIGRpc3RhbmNlcyBzYXZlZCB0byB7b3V0X2Nzdl9jb2xvcl9yYWR9IikKCiAgICBmb3IgdW5pdCwgc2NhbGVfdSwgY29sb3VyIGluICgoInB4IiwgMS4wLCAiYmx1ZSIpLCAoInBjIiwgcHhfdG9fcGMsICJncmVlbiIpKToKICAgICAgICByYWRpaSA9IGRmX2NvbG9yX2ZpbHRlcmVkWyJSYWRpYWxfRGlzdGFuY2VfcHgiXSAqIHNjYWxlX3UKICAgICAgICB3aWR0aCA9IFN0ZXBfc2l6ZSAqIHNjYWxlX3UKICAgICAgICBlZGdlcyA9IG5wLmFyYW5nZSgwLCByYWRpaS5tYXgoKSArIHdpZHRoLCB3aWR0aCkKICAgICAgICBjb3VudHMsIGVkZ2VzID0gbnAuaGlzdG9ncmFtKHJhZGlpLCBiaW5zPWVkZ2VzKQogICAgICAgIGFyZWFzID0gbnAucGkgKiAoZWRnZXNbMTpdICoqIDIgLSBlZGdlc1s6LTFdICoqIDIpCiAgICAgICAgZGVucyA9IGNvdW50cyAvIGFyZWFzCiAgICAgICAgc3VmZml4ID0gIiIgaWYgdW5pdCA9PSAicHgiIGVsc2UgIl9wYyIKICAgICAgICBwZC5EYXRhRnJhbWUoe2YiUl9pbm5lcl97dW5pdH0iOiBlZGdlc1s6LTFdLCBmIlJfb3V0ZXJfe3VuaXR9IjogZWRnZXNbMTpdLAogICAgICAgICAgICAgICAgICAgICAgZiJSX2NlbnRlcl97dW5pdH0iOiAwLjUgKiAoZWRnZXNbMTpdICsgZWRnZXNbOi0xXSksCiAgICAgICAgICAgICAgICAgICAgICAiTl9zb3VyY2VzIjogY291bnRzLCBmIkFubnVsdXNfYXJlYV97dW5pdH0yIjogYXJlYXMsCiAgICAgICAgICAgICAgICAgICAgICBmIkRlbnNpdHlfcGVyX3t1bml0fTIiOiBkZW5zfSkudG9fY3N2KAogICAgICAgICAgICBvcy5wYXRoLmpvaW4ob3V0X2RpciwgZiJ7c2FmZX1fcmFkaWFsX2RlbnNpdHlfcHJvZmlsZXtzdWZmaXh9LmNzdiIpLCBpbmRleD1GYWxzZSkKICAgICAgICBzdGVwX3IgPSBucC5jb2x1bW5fc3RhY2soW2VkZ2VzWzotMV0sIGVkZ2VzWzE6XV0pLnJhdmVsKCkKICAgICAgICBzdGVwX2QgPSBucC5yZXBlYXQoZGVucywgMikKICAgICAgICBzdGVwX24gPSBucC5yZXBlYXQoY291bnRzLCAyKQogICAgICAgIHBkLkRhdGFGcmFtZSh7ZiJSX3N0ZXBfe3VuaXR9Ijogc3RlcF9yLCBmIkRlbnNpdHlfc3RlcF9wZXJfe3VuaXR9MiI6IHN0ZXBfZCwKICAgICAgICAgICAgICAgICAgICAgICJOX3NvdXJjZXNfc3RlcCI6IHN0ZXBfbn0pLnRvX2NzdigKICAgICAgICAgICAgb3MucGF0aC5qb2luKG91dF9kaXIsIGYie3NhZmV9X3JhZGlhbF9kZW5zaXR5X3Byb2ZpbGVfc3RlcHtzdWZmaXh9LmNzdiIpLAogICAgICAgICAgICBpbmRleD1GYWxzZSkKICAgICAgICBwbHQuZmlndXJlKGZpZ3NpemU9KDgsIDYpKQogICAgICAgIHBsdC5zdGVwKHN0ZXBfciwgc3RlcF9kLCB3aGVyZT0icG9zdCIsIGNvbG9yPWNvbG91ciwgbGluZXdpZHRoPTIpCiAgICAgICAgcGx0LnhsYWJlbCgiUmFkaWFsIGRpc3RhbmNlIFtwaXhlbHNdIiBpZiB1bml0ID09ICJweCIgZWxzZSAiUmFkaWFsIGRpc3RhbmNlIFtwY10iKQogICAgICAgIHBsdC55bGFiZWwoIlNvdXJjZSBkZW5zaXR5IFsxL3BpeGVswrJdIiBpZiB1bml0ID09ICJweCIgZWxzZSAiU291cmNlIGRlbnNpdHkgWzEvcGPCsl0iKQogICAgICAgIHBsdC50aXRsZShmIlJhZGlhbCBkZW5zaXR5IHN0ZXAgcHJvZmlsZSBvZiB7b2JqX25hbWV9ICIKICAgICAgICAgICAgICAgICAgZiIoeydwaXhlbHMnIGlmIHVuaXQgPT0gJ3B4JyBlbHNlICdwYyd9KSIpCiAgICAgICAgcG5nID0gb3MucGF0aC5qb2luKG91dF9kaXIsIGYie3NhZmV9X3JhZGlhbF9kZW5zaXR5X3Byb2ZpbGVfc3RlcHtzdWZmaXh9LnBuZyIpCiAgICAgICAgcGx0LnNhdmVmaWcocG5nLCBkcGk9MTUwKQogICAgICAgIHBsdC5jbG9zZSgpCiAgICAgICAgcHJpbnQoZiJSYWRpYWwgZGVuc2l0eSBzdGVwIHByb2ZpbGUgKHt1bml0fSkgc2F2ZWQgdG8ge3BuZ30iKQoKICAgICMgPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KICAgICMgU3RhZ2UgNiAtIGRlcHJvamVjdGlvbiBhbmQgdGhlIHNoYXJlZCByYWRpYWwgYW5hbHlzaXMKICAgICMgPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KICAgICMKICAgICMgQSBzcGlyYWwgZ2FsYXh5IGlzIGEgZmxhdCBkaXNrIHNlZW4gYXQgYW4gYW5nbGUsIHNvIGEgZGlzdGFuY2UgbWVhc3VyZWQgaW4KICAgICMgdGhlIGltYWdlIGlzIG5vdCB0aGUgZGlzdGFuY2UgaW5zaWRlIHRoZSBnYWxheHkuIFRoaXMgc3RhZ2Ugc3RyZXRjaGVzIHRoZQogICAgIyBzcXVhc2hlZCBkaXJlY3Rpb24gYmFjayBvdXQgYW5kIHJlY29tcHV0ZXMgdGhlIHByb2ZpbGUgc2V2ZXJhbCB3YXlzIC0KICAgICMgdW5jb3JyZWN0ZWQsIGZyb20gdGhlIEh5cGVyTEVEQSBjYXRhbG9ndWUsIGFuZCBmcm9tIHRoZSBrbm90cyB0aGVtc2VsdmVzIC0KICAgICMgYmVjYXVzZSBubyBkZXByb2plY3Rpb24gbWV0aG9kIGhhcyBiZWVuIGZvdW5kIHJlbGlhYmx5IGJldHRlciB0aGFuIHRoZQogICAgIyBvdGhlcnMuIFRoZSBzcHJlYWQgYmV0d2VlbiB0aGUgY3VydmVzIGlzIHRoZSB1bmNlcnRhaW50eSBvbiB0aGUgcmFkaWkuIEl0CiAgICAjIGFsc28gdXNlcyB0aGUgcmluZyBhcmVhIGFjdHVhbGx5IGNvdmVyZWQgYnkgdGhlIGZyYW1lLgogICAgdHJ5OgogICAgICAgIG5vcnRoLCBtaXJyb3JlZCA9IGltYWdlX29yaWVudGF0aW9uKHdjc19WKQogICAgICAgIHByaW50KGYiW2RlcHJvamVjdF0gb3JpZW50YXRpb24gZnJvbSBXQ1M6IG5vcnRoIHtub3J0aDouMWZ9IGRlZyBmcm9tICt4LCAiCiAgICAgICAgICAgICAgZiJ7J21pcnJvcmVkJyBpZiBtaXJyb3JlZCBlbHNlICdub3QgbWlycm9yZWQnfSIpCiAgICAgICAgcHJpbnQoIiIpCiAgICAgICAgcHJpbnQoIkRlcHJvamVjdGluZyBhbmQgcmVidWlsZGluZyB0aGUgcmFkaWFsIHByb2ZpbGUuLi4iKQogICAgICAgIHJ1bl9wcm9maWxlcygKICAgICAgICAgICAgZGZfY29sb3JfZmlsdGVyZWRbIlhfViJdLnRvX251bXB5KGR0eXBlPWZsb2F0KSwKICAgICAgICAgICAgZGZfY29sb3JfZmlsdGVyZWRbIllfViJdLnRvX251bXB5KGR0eXBlPWZsb2F0KSwKICAgICAgICAgICAgKHhfY2VudGVyLCB5X2NlbnRlciksICgwLjAsIGZsb2F0KHNoYXBlX1ZbMV0pLCAwLjAsIGZsb2F0KHNoYXBlX1ZbMF0pKSwKICAgICAgICAgICAgb3MucGF0aC5qb2luKG91dF9kaXIsIGYie3NhZmV9X2RlcHJvamVjdGVkIiksCiAgICAgICAgICAgIGYiQmx1ZSBrbm90cyAtIHtvYmpfbmFtZX0iLCBnYWxheHk9b2JqX25hbWUsCiAgICAgICAgICAgIG5vcnRoX2FuZ2xlPW5vcnRoLCBtaXJyb3JlZD1taXJyb3JlZCwgc2VjdG9yX21ldGhvZD1UcnVlLAogICAgICAgICAgICByaW5ncz0yMCwgc2NhbGU9cHhfdG9fcGMsIHVuaXQ9InBjIikKICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOgogICAgICAgIHByaW50KGYiW2RlcHJvamVjdF0gdGhlIGRlcHJvamVjdGlvbiBzdGFnZSBmYWlsZWQgKHt0eXBlKGV4YykuX19uYW1lX199OiAiCiAgICAgICAgICAgICAgZiJ7ZXhjfSkuIEV2ZXJ5dGhpbmcgYWJvdmUgd2FzIHN0aWxsIHdyaXR0ZW4uIikKICAgIHJldHVybiAwCgoKZGVmIHdyaXRlX3N1bW1hcnkocGF0aCwgdik6CiAgICAiIiJFdmVyeSBudW1iZXIgdGhpcyBydW4gdXNlZCBhbmQgZm91bmQsIGluIG9uZSBmaWxlIGJlc2lkZSBpdHMgb3V0cHV0LiIiIgogICAgY2FsLCBzdCA9IHZbImNhbCJdLCB2WyJjYWxfc3RhdHMiXQogICAgbGluZXMgPSBbCiAgICAgICAgZiJCbHVlX0NsdXN0ZXJzX0Zyb21fSW1hZ2VzLnB5IHJ1biBzdW1tYXJ5IiwKICAgICAgICBmImdhbGF4eSAgICAgICAgICAgICAge3ZbJ29ial9uYW1lJ119IiwKICAgICAgICBmIkIgZnJhbWUgICAgICAgICAgICAge3ZbJ2ZpdHNfZmlsZV9CJ119IiwKICAgICAgICBmIlYgZnJhbWUgICAgICAgICAgICAge3ZbJ2ZpdHNfZmlsZV9WJ119IiwKICAgIF0gKyBbZiJpbnB1dCAgICAgICAgICAgICAgIHtufSIgZm9yIG4gaW4gdlsiY29tYmluZWRfbm90ZSJdXSArIFsKICAgICAgICBmImRpc3RhbmNlICAgICAgICAgICAge3ZbJ2Rpc3RhbmNlJ106LjNmfSBNcGMgKGRpc3RhbmNlIG1vZHVsdXMge3ZbJ211J106LjNmfSkiLAogICAgICAgIGYiRShCLVYpICAgICAgICAgICAgICB7dlsnRV9CViddOi40Zn0gICBBX1YgPSB7dlsnQV9WJ106LjNmfSIsCiAgICAgICAgZiJwbGF0ZSBzb2x1dGlvbiAgICAgIHt2Wyd3Y3Nfbm90ZSddfSIsCiAgICAgICAgZiJyZWdpc3RyYXRpb24gICAgICAgIEIgb250byBWIHt2WydyZWdfdGV4dCddfSIsCiAgICAgICAgZiJzdGFyIEZXSE0gICAgICAgICAgIEIge3ZbJ2Z3aG1fQiddOi4yZn0gcHgsIFYge3ZbJ2Z3aG1fViddOi4yZn0gcHgiLAogICAgICAgIGYiYXBlcnR1cmUgICAgICAgICAgICB7dlsncl9hcCddOi4yZn0gcHggcmFkaXVzLCBza3kgcmluZyB7dlsncl9pbiddOi4xZn0tIgogICAgICAgIGYie3ZbJ3Jfb3V0J106LjFmfSBweCwgdGhlIHNhbWUgaW4gYm90aCBiYW5kcyIsCiAgICAgICAgZiJzYXR1cmF0aW9uICAgICAgICAgIEIge3ZbJ2luZm8nXVsnQiddWydzYXQnXTouMGZ9IEFEVSwgViB7dlsnaW5mbyddWydWJ11bJ3NhdCddOi4wZn0gQURVIiwKICAgICAgICBmImdhaW4gICAgICAgICAgICAgICAgQiB7dlsnaW5mbyddWydCJ11bJ2dhaW4nXTouMmZ9LCBWIHt2WydpbmZvJ11bJ1YnXVsnZ2FpbiddOi4yZn0gZS0vQURVIiwKICAgICAgICBmInNvdXJjZXMgbWVhc3VyZWQgICAge2xlbih2WydkZiddKX0gKHBvc2l0aXZlIGZsdXggaW4gYm90aCBiYW5kcykiLAogICAgICAgIGYiY2FsaWJyYXRpb24gICAgICAgICB7dlsnY2F0YWxvZ3VlJ119OiB7Y2FsWydOX3VzZWQnXX0gb2Yge2NhbFsnTl9jYW5kaWRhdGVzJ119ICIKICAgICAgICBmInN0YXJzIHVzZWQgKHtzdC5nZXQoJ21hdGNoZWQnLCAwKX0gbWF0Y2hlZCkiLAogICAgICAgIGYiICAgQi1WID0ge2NhbFsnYSddOisuNGZ9ICsge2NhbFsnYyddOi40Zn0geCAoYi12KSAgIHNjYXR0ZXIge2NhbFsncm1zX2NvbG91cl9maXQnXTouM2Z9IiwKICAgICAgICBmIiAgIFYgICA9IHYgKyB7Y2FsWydaUF9WJ106LjRmfSB7Y2FsWydrX1YnXTorLjRmfSB4IChCLVYpICAgc2NhdHRlciB7Y2FsWydybXNfVl9maXQnXTouM2Z9IiwKICAgICAgICBmIiAgIEIgICA9IGIgKyB7Y2FsWydaUF9CJ106LjRmfSB7Y2FsWydrX0InXTorLjRmfSB4IChCLVYpICAgc2NhdHRlciB7Y2FsWydybXNfQl9maXQnXTouM2Z9IiwKICAgICAgICBmIiAgIGNvbG91ciBjb3JyZWxhdGlvbiB7Y2FsWydjb3JyZWxhdGlvbiddOisuM2Z9IiwKICAgICAgICBmImZvcmVncm91bmQgcmVtb3ZlZCAge3ZbJ25fY2F0X3N0YXJzJ119IGNhdGFsb2d1ZSBzdGFycywge3ZbJ25fZ2FpYSddfSBHYWlhICIKICAgICAgICBmImZvcmVncm91bmQgc3RhcnMsIHt2WyduX291dHNpZGUnXX0gYmV5b25kIFIyNSIsCiAgICAgICAgZiJzb3VyY2VzIGxlZnQgICAgICAgIHtsZW4odlsnZGZfY2xlYW4nXSl9IiwKICAgICAgICBmImJsdWUga25vdHMgICAgICAgICAge2xlbih2WydkZl9jb2xvcl9maWx0ZXJlZCddKX0gICh7Q01EX0NPTE9SX01JTn0gPD0gKEItVikwIDw9ICIKICAgICAgICBmIntDTURfQ09MT1JfTUFYfSwgQi1WIGVycm9yIDw9IHtBUkdTLm1heF9idl9lcnJ9LCBub3Qgc2F0dXJhdGVkKSIsCiAgICBdCiAgICB3aXRoIG9wZW4ocGF0aCwgInciLCBlbmNvZGluZz0idXRmLTgiKSBhcyBmaDoKICAgICAgICBmaC53cml0ZSgiXG4iLmpvaW4obGluZXMpICsgIlxuIikKICAgIHByaW50KGYiUnVuIHN1bW1hcnkgc2F2ZWQgdG8ge3BhdGh9IikKCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgogICAgc3lzLmV4aXQobWFpbigpKQo="


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
