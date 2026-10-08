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

_CLUSTER_TOOL_SOURCE = "IyAtKi0gY29kaW5nOiB1dGYtOCAtKi0KIiIiCkNyZWF0ZWQgYnk6IERyLiBCb2F6IFJvbiBab2hhcgpodHRwczovL2dpdGh1Yi5jb20vQm9helJvblpvaGFyL09ic2VydmF0aW9uYWxBc3Ryb25vbXlFZHVjYXRpb25Ub29scy9ibG9iL21haW4vR2FsYXh5X0FuYWx5c2lzL0JsdWVfQ2x1c3RlcnNfRnJvbV9JbWFnZXMucHkKQWZmaWxpYXRpb246IEtpbm5lcmV0IE9ic2VydmF0b3J5Ck1lbWJlciBvZiB0aGUgTENPIEdsb2JhbCBTa3kgUGFydG5lcnMgcHJvZ3JhbW1lCkRhdGU6IFNlcHRlbWJlciAyMDI1CgpXcml0dGVuIGZvciBzdHVkZW50IHByb2plY3RzIG9uIHNwaXJhbCBnYWxheGllcy4KCldIQVQgSVQgRE9FUwoKVGFrZXMgb25lIEItYmFuZCBhbmQgb25lIFYtYmFuZCBpbWFnZSBvZiBhIGdhbGF4eSBhbmQgcHJvZHVjZXMgaXRzCmNvbG91ci1tYWduaXR1ZGUgZGlhZ3JhbSwgdG9nZXRoZXIgd2l0aCB0aGUgcmFkaWFsIGRlbnNpdHkgcHJvZmlsZSBvZiB0aGUKYmx1ZSBrbm90cyAtIHRoZSB5b3VuZyBzdGFyLWZvcm1pbmcgcmVnaW9ucyBpbiB0aGUgYXJtcy4KClRoZSBjb2xvdXIgb2YgYSBzb3VyY2Ugc2F5cyBob3cgaG90IGFuZCBob3cgeW91bmcgaXQgaXMuIFBsb3R0aW5nIGNvbG91cgphZ2FpbnN0IGJyaWdodG5lc3MgZm9yIGV2ZXJ5IHNvdXJjZSBpbiB0aGUgZ2FsYXh5IHNlcGFyYXRlcyB0aGUgYmx1ZSBrbm90cwpmcm9tIHRoZSBvbGRlciBwb3B1bGF0aW9uLCBhbmQgdGhlIHJhZGlhbCBwcm9maWxlIHRoZW4gc2hvd3MgaG93IHRoZXkgYXJlCmRpc3RyaWJ1dGVkIHdpdGggZGlzdGFuY2UgZnJvbSB0aGUgY2VudHJlIC0gd2hpY2ggaXMgd2hhdCBhIHNwaXJhbCBhcm0KbG9va3MgbGlrZSBpbiBudW1iZXJzLgoKV0hFUkUgVEhFIE1FVEhPRCBDT01FUyBGUk9NCgpJZGVudGlmeWluZyBzdGFyLWZvcm1pbmcgcmVnaW9ucyBieSB0aGVpciBjb2xvdXIgaW4gb3JkaW5hcnkgYnJvYWQtYmFuZAppbWFnZXMsIGFuZCB0aGVuIHRyZWF0aW5nIHRoZWlyIGRpc3RyaWJ1dGlvbiBhcyBhIG1lYXN1cmFibGUgcHJvcGVydHkgb2YgdGhlCmdhbGF4eSwgZm9sbG93cwoKICAgIEJyb3NjaCwgTi4gKDE5OTIpLiBTdGFyIGZvcm1hdGlvbiBzeXN0ZW1hdGljcyBmcm9tIGNvbG91ciBpbWFnZXMuCiAgICBBc3Ryb3BoeXNpY3MgYW5kIFNwYWNlIFNjaWVuY2UsIDE4OCwgMjg5LTI5OC4KICAgIGRvaToxMC4xMDA3L0JGMDA2NDQ5MTYKClRoZSBhcHBlYWwgZm9yIHRlYWNoaW5nIGlzIHRoYXQgaXQgYXNrcyBub3RoaW5nIGV4b3RpYy4gVHdvIGJyb2FkLWJhbmQgZnJhbWVzCm9mIGEgZ2FsYXh5LCBCIGFuZCBWLCBhcmUgd2l0aGluIHJlYWNoIG9mIGEgc2Nob29sLWFjY2Vzc2libGUgdGVsZXNjb3BlLCBhbmQKdGhlIHlvdW5nIHJlZ2lvbnMgc2VwYXJhdGUgb3V0IG9uIGNvbG91ciBhbG9uZSAtIG5vIHNwZWN0cm9zY29weSwgbm8gbmFycm93LQpiYW5kIGZpbHRlci4gQSBzdHVkZW50IHdpdGggb25lIG5pZ2h0IG9mIGRhdGEgY2FuIGFzayBhIHJlYWwgcXVlc3Rpb24gYWJvdXQKd2hlcmUgYSBnYWxheHkgaXMgZm9ybWluZyBzdGFycywgYW5kIGFuc3dlciBpdCB3aXRoIGEgbnVtYmVyLgoKVGhpcyBzY3JpcHQgaXMgdGhlIHBpcGVsaW5lIGZvciB0aGF0LCBhbmQgaGFzIGJlZW4gdXNlZCBhY3Jvc3MgbWFueSBzdHVkZW50CnByb2plY3RzLgoKV0hBVCBZT1UgTkVFRAoKICBhIEIgYW5kIGEgViBmcmFtZSBvZiB0aGUgc2FtZSBnYWxheHkgKG9yIGEgZm9sZGVyIGhvbGRpbmcgc2V2ZXJhbCBvZiBlYWNoKSwKICBwbGF0ZS1zb2x2ZWQ6IHRoZSByZWZlcmVuY2Ugc3RhcnMgYXJlIHB1bGxlZCBmcm9tIGEgY2F0YWxvZ3VlIGJ5IHNreQogIHBvc2l0aW9uLiBUaGUgcGxhdGUgc29sdXRpb24gaXMgY2hlY2tlZCBhZ2FpbnN0IEdhaWEgYW5kIHJlcGFpcmVkIGlmIGl0CiAgZG9lcyBub3QgZml0IHRoZSBpbWFnZS4KCldIQVQgSVQgQVNLUyBZT1UKCiAgdGhlIGZvbGRlciBob2xkaW5nIHRoZSBmcmFtZXMgIHdoaWNoIGZpbGUgaXMgQiBhbmQgd2hpY2ggaXMgViBpcyByZWFkIGZyb20KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhlIEZJTFRFUiBrZXl3b3JkIG9yIHRoZSBmaWxlIG5hbWUsIGFuZCB0aGUKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZ2FsYXh5J3MgbmFtZSBmcm9tIE9CSkVDVAogIGRpc3RhbmNlIGluIE1wYyAgICAgICAgICAgICAgICBzaG93biBmcm9tIENvc21pY2Zsb3dzLTMgKFR1bGx5KyAyMDE2KTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgRW50ZXIgYWNjZXB0cyBpdC4gQ29udmVydHMgcGl4ZWxzIHRvIHBhcnNlY3MKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYW5kIGdpdmVzIHRoZSBhYnNvbHV0ZSBtYWduaXR1ZGUgTV9WCiAgY29sb3VyIGV4Y2VzcyBFKEItVikgICAgICAgICAgIHNob3duIGZyb20gdGhlIFNjaGxhZmx5ICYgRmlua2JlaW5lciAoMjAxMSkKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZHVzdCBtYXBzIHZpYSBJUlNBOyBFbnRlciBhY2NlcHRzIGl0LgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBBX1YgaXMgMy4xIHggRShCLVYpCgpFdmVyeXRoaW5nIGNhbiBhbHNvIGJlIGdpdmVuIG9uIHRoZSBjb21tYW5kIGxpbmUsIGZvciBleGFtcGxlCgogIHB5dGhvbiBCbHVlX0NsdXN0ZXJzX0Zyb21fSW1hZ2VzLnB5IC0tZm9sZGVyICJEOi9NODMiIC0teWVzCiAgcHl0aG9uIEJsdWVfQ2x1c3RlcnNfRnJvbV9JbWFnZXMucHkgLS1iIEIuZml0cyAtLXYgVi5maXRzIC0tZ2FsYXh5IE04MwogICAgICAtLWRpc3RhbmNlLW1wYyA0LjcgLS1lYnYgMC4wNiAtLWNhdGFsb2cgR0FJQQoKSE9XIElUIFdPUktTCgogIDEuIHN1YnRyYWN0cyB0aGUgYmFja2dyb3VuZCBmcm9tIGVhY2ggZnJhbWUsIG1lYXN1cmVzIGhvdyB3aWRlIGEgc3RhciBpcwogICAgIGluIGVhY2ggKGZyb20gYnJpZ2h0IHN0YXJzIHRoYXQgYXJlIG5vdCBzYXR1cmF0ZWQpLCBhbmQgZmluZHMgdGhlCiAgICAgc291cmNlcyBpbiBib3RoCiAgMi4gcmVnaXN0ZXJzIEIgb250byBWLCBhbmQgbWFrZXMgb25lIGxpc3Qgb2Ygc291cmNlcyBvbiB0aGUgViBncmlkCiAgMy4gbWVhc3VyZXMgZXZlcnkgc291cmNlIGluIGJvdGggYmFuZHMgYXQgdGhlIHNhbWUgcGxhY2Ugb24gdGhlIHNreSwKICAgICB0aHJvdWdoIE9ORSBhcGVydHVyZTogdHdpY2UgdGhlIEZXSE0gb2YgdGhlIHdpZGVyIG9mIHRoZSB0d28gZnJhbWVzLgogICAgIEVhY2ggZmx1eCBnZXRzIGFuIGVycm9yIGZyb20gdGhlIHBob3RvbiBub2lzZSBhbmQgdGhlIG1lYXN1cmVkIHNreQogICAgIG5vaXNlLCBhbmQgZWFjaCBzb3VyY2UgYSBzYXR1cmF0aW9uIGZsYWcKICA0LiBjaGVja3MgdGhlIHBsYXRlIHNvbHV0aW9uIGFnYWluc3QgR2FpYSwgdGhlbiBjYWxpYnJhdGVzIGFnYWluc3QgZXZlcnkKICAgICBjYXRhbG9ndWUgc3RhciB0aGF0IGlzIG1hdGNoZWQgb25jZSwgbm90IHNhdHVyYXRlZCwgbm90IGNyb3dkZWQgYW5kCiAgICAgd2VsbCBtZWFzdXJlZCAoQVBBU1M5LCBvciBHYWlhIERSMyBzeW50aGV0aWMgcGhvdG9tZXRyeSwgd2hpY2ggaXMgYWxzbwogICAgIHRoZSBmYWxsYmFjayk6IGEgc3RyYWlnaHQtbGluZSBmaXQgd2l0aCBhIGNvbG91ciB0ZXJtIGFuZCBvdXRsaWVyCiAgICAgcmVqZWN0aW9uLAogICAgICAgICBCLVYgPSBhICsgYyB4IChiLXYpICAgICAgICBWID0gdiArIFpQICsgayB4IChCLVYpCiAgICAgVGhlIGZpdCBpcyBwcmludGVkIHdpdGggaXRzIHNjYXR0ZXIsIGFuZCB0aGUgcnVuIHN0b3BzIGlmIGZld2VyIHRoYW4KICAgICA1IHN0YXJzIHJlbWFpbiBvciBpZiB0aGUgY29sb3VycyBydW4gdGhlIHdyb25nIHdheSAoQiBhbmQgViBzd2FwcGVkKQogIDUuIHJlbW92ZXMgdGhlIGR1c3Qgb2Ygb3VyIG93biBnYWxheHksIGFuZCB0aGUgZm9yZWdyb3VuZCBzdGFycwogIDYuIGtlZXBzIGFzIGJsdWUga25vdHMgdGhlIHNvdXJjZXMgd2l0aCAtMC41IDw9IChCLVYpMCA8PSAwLjUgYW5kIGEgQi1WCiAgICAgZXJyb3Igb2YgYXQgbW9zdCAwLjIgbWFnLCBhbmQgcGxvdHMgdGhlIGNvbG91ci1tYWduaXR1ZGUgZGlhZ3JhbXMgYW5kCiAgICAgdGhlIHJhZGlhbCBkZW5zaXR5IHByb2ZpbGUKCldIQVQgWU9VIEdFVAoKRXZlcnkgcnVuIHdyaXRlcyBhIG5ldyBmb2xkZXIgYmVzaWRlIHRoZSBmcmFtZXMsCnJlc3VsdF9DTURfPGdhbGF4eT5fPGRhdGU+Xzx0aW1lPiwgc28gYSBzZWNvbmQgcnVuIG5ldmVyIG92ZXJ3cml0ZXMgdGhlIGZpcnN0CmFuZCBub3RoaW5nIGZyb20gYW4gZWFybGllciBydW4gY2FuIGJlIG1pc3Rha2VuIGZvciB0aGlzIG9uZS4gVGhlIGZvbGRlcgpob2xkaW5nIHlvdXIgZGF0YSBpcyBvdGhlcndpc2UgbGVmdCBleGFjdGx5IGFzIGl0IHdhcy4KClRoYXQgaXMgYSBsb3Qgb2YgZmlsZXMsIHNvIGhlcmUgaXMgd2hhdCBlYWNoIG9uZSBpcyBhbmQgd2hpY2ggeW91IGFjdHVhbGx5CndhbnQgdG8gbG9vayBhdC4gQmVsb3csIE5BTUUgc3RhbmRzIGZvciB0aGUgZ2FsYXh5IG5hbWUuCgogIHJ1bl9zdW1tYXJ5LnR4dCAgICAgICAgICAgICAgICAgZXZlcnkgbnVtYmVyIHRoZSBydW4gdXNlZCBhbmQgZm91bmQKCiAgQnktcHJvZHVjdHMsIGtlcHQgYmVjYXVzZSB0aGV5IGFyZSB3aGF0IGV2ZXJ5dGhpbmcgaXMgbWVhc3VyZWQgb246CgogICAgPGlucHV0Pl9iZ3N1Yi5maXRzICAgICAgICAgICAgZWFjaCBpbnB1dCBmcmFtZSB3aXRoIGl0cyBtZWRpYW4gbGV2ZWwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHN1YnRyYWN0ZWQKICAgIE5BTUVfQl9jb21iaW5lZC5maXRzICAgICAgICAgIHdoZW4gdGhlIGZvbGRlciBoZWxkIHNldmVyYWwgZXhwb3N1cmVzCiAgICBOQU1FX1ZfY29tYmluZWQuZml0cyAgICAgICAgICBvZiBhIGZpbHRlcjogdGhlaXIgbWVkaWFuCgogIFN0YWdlIDEgLSBldmVyeXRoaW5nIHRoYXQgd2FzIG1lYXN1cmVkCgogICAgTkFNRV9waG90b21ldHJ5X3Jlc3VsdHMuY3N2ICAgZXZlcnkgc291cmNlIG1lYXN1cmVkIGluIGJvdGggYmFuZHM6CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBwb3NpdGlvbiBpbiBlYWNoIGZyYW1lLCBSQSBhbmQgRGVjLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZmx1eGVzIGFuZCBpbnN0cnVtZW50YWwgbWFnbml0dWRlcyB3aXRoCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBlcnJvcnMsIHBlYWsgcGl4ZWwsIHNhdHVyYXRpb24gZmxhZwogICAgTkFNRV9yZWZlcmVuY2Vfc3RhcnMuY3N2ICAgICAgdGhlIGNhdGFsb2d1ZSBzdGFycyBtYXRjaGVkIHRvIHNvdXJjZXMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3aGV0aGVyIGVhY2ggd2FzIHVzZWQgaW4gdGhlIGNhbGlicmF0aW9uLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYW5kIGl0cyByZXNpZHVhbC4gSWYgdGhlIGNhbGlicmF0aW9uIGxvb2tzCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3cm9uZywgbG9vayBoZXJlIGZpcnN0CgogIFN0YWdlIDIgLSBjYWxpYnJhdGVkLCBzdGlsbCB3aXRoIGZvcmVncm91bmQgc3RhcnMgaW4gaXQKCiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeS5jc3YgICAgICAgICAgICBCLCBWLCBCLVYgd2l0aCBlcnJvcnMsIGJlZm9yZQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKF9vYnMpIGFuZCBhZnRlciB0aGUgR2FsYWN0aWMKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGR1c3QgY29ycmVjdGlvbiwgYW5kIE1fVgogICAgTkFNRV9jYWxpYnJhdGVkX3Bob3RvbWV0cnlfd2l0aF9jb2xvci5jc3YgdGhlIHNhbWUsIHdpdGggQi1WIGFkZGVkCiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9DTUQucG5nICAgICAgICBjb2xvdXItbWFnbml0dWRlIGRpYWdyYW0KICAgIE5BTUVfVl9zb3VyY2VzLnBuZyAgICAgICAgICAgICAgICAgICAgICAgIHRoZSBWIGltYWdlIHdpdGggZXZlcnkgbWVhc3VyZWQKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHNvdXJjZSBjaXJjbGVkCiAgICBOQU1FX0Jfc291cmNlcy5wbmcgICAgICAgICAgICAgICAgICAgICAgICB0aGUgc2FtZSBmb3IgQiwgYXQgdGhlIHNhbWUKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHNvdXJjZXMKCiAgU3RhZ2UgMyAtIGZvcmVncm91bmQgc3RhcnMgcmVtb3ZlZAoKICAgIENhdGFsb2d1ZSBzdGFycywgR2FpYSBzdGFycyB3aXRoIGEgc2lnbmlmaWNhbnQgcGFyYWxsYXggb3IgcHJvcGVyIG1vdGlvbiwKICAgIGFuZCBldmVyeXRoaW5nIGJleW9uZCBSMjUgYXJlIHRha2VuIG91dC4KCiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFycy5jc3YKICAgIE5BTUVfY2FsaWJyYXRlZF9waG90b21ldHJ5X25vX3N0YXJzX3dpdGhfY29sb3IuY3N2CiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFyc19DTUQucG5nCiAgICBOQU1FX1Zfc291cmNlc19ub19zdGFycy5wbmcKCiAgU3RhZ2UgNCAtIGJsdWUga25vdHMgb25seQoKICAgIFRISVMgSVMgVEhFIFJFU1VMVCB0aGUgcHJvamVjdHMgYXJlIHVzdWFsbHkgYWZ0ZXIuCgogICAgTkFNRV9jYWxpYnJhdGVkX3Bob3RvbWV0cnlfbm9fc3RhcnNfY29sb3JfZmlsdGVyZWQuY3N2CiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFyc19jb2xvcl9maWx0ZXJlZF93aXRoX2NvbG9yLmNzdgogICAgTkFNRV9jYWxpYnJhdGVkX3Bob3RvbWV0cnlfbm9fc3RhcnNfY29sb3JfZmlsdGVyZWRfQ01ELnBuZwogICAgTkFNRV9WX3NvdXJjZXNfY29sb3JfZmlsdGVyZWQucG5nCiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFyc19jb2xvcl9maWx0ZXJlZF93aXRoX3JhZGl1cy5jc3YKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhlIHNhbWUsIHBsdXMgZWFjaCBrbm90J3MgZGlzdGFuY2UKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZnJvbSB0aGUgY2VudHJlIG9mIHRoZSBnYWxheHkKCiAgU3RhZ2UgNSAtIGhvdyB0aGUga25vdHMgYXJlIGRpc3RyaWJ1dGVkCgogICAgTkFNRV9yYWRpYWxfZGVuc2l0eV9wcm9maWxlLmNzdiAgICAgICBrbm90cyBwZXIgc3F1YXJlIHBpeGVsLCBieSByYWRpdXMKICAgIE5BTUVfcmFkaWFsX2RlbnNpdHlfcHJvZmlsZV9wYy5jc3YgICAgdGhlIHNhbWUgaW4gcGFyc2VjcwogICAgTkFNRV9yYWRpYWxfZGVuc2l0eV9wcm9maWxlX3N0ZXAuY3N2ICAgICB0aGUgc2FtZSBhcyBhIHN0ZXAgZnVuY3Rpb24KICAgIE5BTUVfcmFkaWFsX2RlbnNpdHlfcHJvZmlsZV9zdGVwX3BjLmNzdiAgdGhlIHN0ZXAgZnVuY3Rpb24gaW4gcGFyc2VjcwogICAgTkFNRV9yYWRpYWxfZGVuc2l0eV9wcm9maWxlX3N0ZXAucG5nICAgICAgIHRoZSBwbG90LCBpbiBwaXhlbHMKICAgIE5BTUVfcmFkaWFsX2RlbnNpdHlfcHJvZmlsZV9zdGVwX3BjLnBuZyAgICB0aGUgcGxvdCwgaW4gcGFyc2VjcwogICAgTkFNRV9kZXByb2plY3RlZF8uLi4gICAgICAgICAgICAgICAgICAgICAgIHRoZSBwcm9maWxlIGRlcHJvamVjdGVkIHNldmVyYWwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3YXlzLCB3aXRoIHRoZSBmcmFtZSdzIGNvdmVyYWdlCgogIElmIHlvdSBsb29rIGF0IGZvdXIgZmlsZXMsIGxvb2sgYXQgdGhlc2U6CgogICAgcnVuX3N1bW1hcnkudHh0ICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRpZCB0aGUgY2FsaWJyYXRpb24gd29yawogICAgTkFNRV9WX3NvdXJjZXNfY29sb3JfZmlsdGVyZWQucG5nICAgICAgICAgIGRpZCBpdCBmaW5kIHRoZSBhcm1zCiAgICBOQU1FXy4uLl9jb2xvcl9maWx0ZXJlZF9DTUQucG5nICAgICAgICAgICAgdGhlIGRpYWdyYW0KICAgIE5BTUVfcmFkaWFsX2RlbnNpdHlfcHJvZmlsZV9zdGVwX3BjLnBuZyAgICB0aGUgYXJtcywgaW4gbnVtYmVycwoKICBJbiB0aGUgZGlhZ3JhbXMsIHNhdHVyYXRlZCBzb3VyY2VzIGFuZCBzb3VyY2VzIHdpdGggYSBCLVYgZXJyb3IgYWJvdmUgMC4zCiAgbWFnIGFyZSBub3QgZHJhd247IHRoZXkgc3RheSBpbiB0aGUgdGFibGVzLgoKVGhlIHBhcmFtZXRlcnMgYXQgdGhlIHRvcCBvZiB0aGUgQmx1ZV9DbHVzdGVycyBzZWN0aW9uIG9mIHRoaXMgZmlsZSBhcmUKdHVuZWQgZm9yIHRoZSBLaW5uZXJldCBhbmQgTENPIGZyYW1lcyB0aGVzZSBwcm9qZWN0cyB1c2UuIE9uIHZlcnkgZGlmZmVyZW50CmRhdGEgLSBhIG11Y2ggc21hbGxlciB0ZWxlc2NvcGUsIGEgbXVjaCBmYWludGVyIGdhbGF4eSAtIHRoZSBkZXRlY3Rpb24KdGhyZXNob2xkIGlzIHRoZSBmaXJzdCB0aGluZyB0byBsb29rIGF0LgoKVXNhZ2U6IHJ1biBpdCwgYW5kIGFuc3dlciB0aGUgcXVlc3Rpb25zOyBFbnRlciBhY2NlcHRzIHdoYXQgaXMgc2hvd24uCiIiIgoKCgpmcm9tIF9fZnV0dXJlX18gaW1wb3J0IGFubm90YXRpb25zCgppbXBvcnQgb3MKaW1wb3J0IHN5cwppbXBvcnQgcmUKaW1wb3J0IG51bXB5IGFzIG5wCmltcG9ydCBtYXRoCmltcG9ydCB0ZXh0d3JhcAppbXBvcnQgbWF0cGxvdGxpYgppbXBvcnQgbWF0cGxvdGxpYi5weXBsb3QgYXMgcGx0CmltcG9ydCBhcmdwYXJzZQppbXBvcnQgd2FybmluZ3MKaW1wb3J0IHBhbmRhcyBhcyBwZApmcm9tIGFzdHJvcHkuaW8gaW1wb3J0IGZpdHMKZnJvbSBwaG90dXRpbHMuZGV0ZWN0aW9uIGltcG9ydCBEQU9TdGFyRmluZGVyCmZyb20gcGhvdHV0aWxzLmFwZXJ0dXJlIGltcG9ydCBDaXJjdWxhckFwZXJ0dXJlLCBDaXJjdWxhckFubnVsdXMsIGFwZXJ0dXJlX3Bob3RvbWV0cnkKZnJvbSBwaG90dXRpbHMuYmFja2dyb3VuZCBpbXBvcnQgQmFja2dyb3VuZDJELCBNZWRpYW5CYWNrZ3JvdW5kCmZyb20gYXN0cm9weS5zdGF0cyBpbXBvcnQgc2lnbWFfY2xpcHBlZF9zdGF0cywgU2lnbWFDbGlwCmZyb20gc2NpcHkubmRpbWFnZSBpbXBvcnQgZ2F1c3NpYW5fZmlsdGVyCmZyb20gYXN0cm9weS53Y3MgaW1wb3J0IFdDUwpmcm9tIGFzdHJvcXVlcnkudml6aWVyIGltcG9ydCBWaXppZXIKZnJvbSBhc3Ryb3B5LmNvb3JkaW5hdGVzIGltcG9ydCBTa3lDb29yZAppbXBvcnQgYXN0cm9weS51bml0cyBhcyB1CmltcG9ydCBhc3Ryb2FsaWduIGFzIGFhCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIHNoYXJlZCBtb2R1bGUgaW50ZXJhY3RpdmVfaW5wdXQucHkKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQoKZGVmIG5vdGhpbmdfd2FzX3Bhc3NlZChhcmd2PU5vbmUpOgogICAgIiIiVHJ1ZSB3aGVuIHRoZSB0b29sIHdhcyBzdGFydGVkIHdpdGggbm8gYXJndW1lbnRzIGF0IGFsbC4KCiAgICBUaGF0IGlzIHRoZSBvbmx5IGNhc2Ugd2hlcmUgcXVlc3Rpb25zIGFyZSB0aGUgcmlnaHQgcmVzcG9uc2UuIEEgdG9vbCBzdGFydGVkCiAgICB3aXRoIHNvbWUgYXJndW1lbnRzIGJ1dCBub3QgYWxsIG9mIHRoZW0gaGFzIGEgbWlzdGFrZSBpbiBpdCwgYW5kIGFyZ3BhcnNlCiAgICBzYXlpbmcgd2hpY2ggb25lIGlzIG1pc3NpbmcgaXMgbW9yZSB1c2VmdWwgdGhhbiBhIHF1ZXN0aW9ubmFpcmUuCiAgICAiIiIKICAgIGlmIGFyZ3YgaXMgTm9uZToKICAgICAgICByZXR1cm4gbGVuKHN5cy5hcmd2KSA8PSAxCiAgICByZXR1cm4gbGVuKGFyZ3YpID09IDAKCmRlZiBub3JtYWxpc2VfcGF0aCh0ZXh0KToKICAgICIiIkEgcGF0aCBhcyBhIHBlcnNvbiBhY3R1YWxseSBwYXN0ZXMgaXQuCgogICAgV2luZG93cyBFeHBsb3JlcidzICJDb3B5IGFzIHBhdGgiIHdyYXBzIHRoZSBwYXRoIGluIGRvdWJsZSBxdW90ZXMsIGFuZCBhIHBhdGgKICAgIGRyYWdnZWQgaW50byBhIHRlcm1pbmFsIG9mdGVuIGFycml2ZXMgd2l0aCBhIHNwYWNlIG9uIHRoZSBlbmQuIEJvdGggYXJlIG1lYW50CiAgICBhcyB0aGUgc2FtZSBwYXRoLCBzbyBib3RoIGFyZSBhY2NlcHRlZC4KICAgICIiIgogICAgcmV0dXJuIG9zLnBhdGgubm9ybXBhdGgodGV4dC5zdHJpcCgpLnN0cmlwKCciJykuc3RyaXAoIiciKSkKCmRlZiBhc2socHJvbXB0LCBkZWZhdWx0PU5vbmUsIGNhc3Q9c3RyKToKICAgICIiIk9uZSBxdWVzdGlvbi4gVGhlIGRlZmF1bHQgaXMgc2hvd24gaW4gYnJhY2tldHMgYW5kIGEgYmFyZSBFbnRlciB0YWtlcyBpdC4KCiAgICBBIHF1ZXN0aW9uIHdpdGggbm8gZGVmYXVsdCBpcyBhc2tlZCBhZ2FpbiB1bnRpbCBpdCBpcyBhbnN3ZXJlZCwgYW5kIGFuIGFuc3dlcgogICAgdGhhdCB3aWxsIG5vdCBjb252ZXJ0IGlzIHJlZnVzZWQgb24gdGhlIHNwb3QgcmF0aGVyIHRoYW4gdGhyZWUgc3RlcHMgbGF0ZXIsCiAgICB3aGVyZSB0aGUgcmVhc29uIHdvdWxkIG5vIGxvbmdlciBiZSBvYnZpb3VzLgogICAgIiIiCiAgICB3aGlsZSBUcnVlOgogICAgICAgIHNob3duID0gKCJ7fSBbe31dOiAiLmZvcm1hdChwcm9tcHQsIGRlZmF1bHQpIGlmIGRlZmF1bHQgaXMgbm90IE5vbmUKICAgICAgICAgICAgICAgICBlbHNlICJ7fTogIi5mb3JtYXQocHJvbXB0KSkKICAgICAgICB0ZXh0ID0gaW5wdXQoc2hvd24pLnN0cmlwKCkKICAgICAgICBpZiBub3QgdGV4dDoKICAgICAgICAgICAgaWYgZGVmYXVsdCBpcyBOb25lOgogICAgICAgICAgICAgICAgcHJpbnQoIiAgICB0aGlzIG9uZSBoYXMgbm8gZGVmYXVsdCAtIHBsZWFzZSBhbnN3ZXIgaXQiKQogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgIyBUaGUgZGVmYXVsdCBnb2VzIHRocm91Z2ggdGhlIHNhbWUgY29udmVyc2lvbiBhcyBhIHR5cGVkIGFuc3dlci4KICAgICAgICAgICAgIyBSZXR1cm5pbmcgaXQgcmF3IGxvb2tzIGhhcm1sZXNzIGJlY2F1c2UgbW9zdCBkZWZhdWx0cyBhcmUgc3RyaW5ncywKICAgICAgICAgICAgIyBidXQgYXNrKCJXaGljaCBvbmUiLCAiMSIsIGludCkgdGhlbiBoYW5kcyBiYWNrICIxIiBhbmQgdGhlIGNhbGxlcgogICAgICAgICAgICAjIGNvbXBhcmVzIGEgc3RyaW5nIHdpdGggYSBudW1iZXIgc2V2ZXJhbCBsaW5lcyBsYXRlciAtIHdoaWNoIGlzCiAgICAgICAgICAgICMgZXhhY3RseSBob3cgYSBiYXJlIEVudGVyIG9uIHRoZSBmcmFtZS1jaG9pY2UgcHJvbXB0IGNyYXNoZWQgYSBydW4uCiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHJldHVybiBjYXN0KGRlZmF1bHQpCiAgICAgICAgICAgIGV4Y2VwdCAoVHlwZUVycm9yLCBWYWx1ZUVycm9yKToKICAgICAgICAgICAgICAgIHJldHVybiBkZWZhdWx0CiAgICAgICAgdHJ5OgogICAgICAgICAgICByZXR1cm4gY2FzdCh0ZXh0KQogICAgICAgIGV4Y2VwdCAoVHlwZUVycm9yLCBWYWx1ZUVycm9yKToKICAgICAgICAgICAgcHJpbnQoIiAgICB0aGF0IGlzIG5vdCBhIHt9IC0gdHJ5IGFnYWluIi5mb3JtYXQoCiAgICAgICAgICAgICAgICBnZXRhdHRyKGNhc3QsICJfX25hbWVfXyIsICJ2YWxpZCB2YWx1ZSIpKSkKCmRlZiBhc2tfZmlsZShwcm9tcHQpOgogICAgIiIiQSBwYXRoIHRvIGEgZmlsZSB0aGF0IGV4aXN0cywgYXNrZWQgYWdhaW4gdW50aWwgaXQgZG9lcy4KCiAgICBDaGVja2luZyBoZXJlIGlzIHdvcnRoIHRoZSB0d28gbGluZXM6IHRoZSBhbHRlcm5hdGl2ZSBpcyBhIHJ1biB0aGF0IHJlYWRzIHRocmVlCiAgICBmcmFtZXMsIHNwZW5kcyBhIG1pbnV0ZSBvbiB0aGUgZm91cnRoLCBhbmQgdGhlbiBkaWVzIG9uIGEgdHlwbyBpbiB0aGUgZmlyc3QuCiAgICAiIiIKICAgIHdoaWxlIFRydWU6CiAgICAgICAgcGF0aCA9IG5vcm1hbGlzZV9wYXRoKGFzayhwcm9tcHQpKQogICAgICAgIGlmIG9zLnBhdGguaXNmaWxlKHBhdGgpOgogICAgICAgICAgICByZXR1cm4gcGF0aAogICAgICAgIHByaW50KCIgICAgdGhlcmUgaXMgbm8gZmlsZSBhdCAiICsgcGF0aCkKCmRlZiBhc2tfZm9sZGVyKHByb21wdCk6CiAgICAiIiJBIHBhdGggdG8gYSBmb2xkZXIgdGhhdCBleGlzdHMsIGFza2VkIGFnYWluIHVudGlsIGl0IGRvZXMuIiIiCiAgICB3aGlsZSBUcnVlOgogICAgICAgIHBhdGggPSBub3JtYWxpc2VfcGF0aChhc2socHJvbXB0KSkKICAgICAgICBpZiBvcy5wYXRoLmlzZGlyKHBhdGgpOgogICAgICAgICAgICByZXR1cm4gcGF0aAogICAgICAgIHByaW50KCIgICAgdGhlcmUgaXMgbm8gZm9sZGVyIGF0ICIgKyBwYXRoKQoKZGVmIGFza19vcHRpb25hbF9mbG9hdChwcm9tcHQsIGJsYW5rX25vdGU9ImJsYW5rIHRvIHNraXAiKToKICAgICIiIkEgbnVtYmVyLCBvciBub3RoaW5nLiBSZXR1cm5zIE5vbmUgd2hlbiB0aGUgYW5zd2VyIGlzIGxlZnQgYmxhbmsuCgogICAgV2hhdCBibGFuayBtZWFucyBpcyB0aGUgY2FsbGVyJ3MgYnVzaW5lc3MsIHNvIHRoZSBjYWxsZXIgc2F5cyBpdC4gQmxhbmsgdXNlZAogICAgdG8gbWVhbiAiZG8gd2l0aG91dCIsIGFuZCBub3cgaXQgdXN1YWxseSBtZWFucyAiZ28gYW5kIGxvb2sgaXQgdXAiIC0gYSBwcm9tcHQKICAgIHRoYXQgc3RpbGwgcHJvbWlzZWQgdGhlIGZpcnN0IHdvdWxkIGJlIGx5aW5nIGFib3V0IHdoYXQgdGhlIHRvb2wgZG9lcy4KICAgICIiIgogICAgd2hpbGUgVHJ1ZToKICAgICAgICB0ZXh0ID0gaW5wdXQoInt9IFt7fV06ICIuZm9ybWF0KHByb21wdCwgYmxhbmtfbm90ZSkpLnN0cmlwKCkKICAgICAgICBpZiBub3QgdGV4dDoKICAgICAgICAgICAgcmV0dXJuIE5vbmUKICAgICAgICB0cnk6CiAgICAgICAgICAgIHJldHVybiBmbG9hdCh0ZXh0KQogICAgICAgIGV4Y2VwdCBWYWx1ZUVycm9yOgogICAgICAgICAgICBwcmludCgiICAgIHRoYXQgaXMgbm90IGEgbnVtYmVyIC0gdHJ5IGFnYWluLCBvciBwcmVzcyBFbnRlciB0byBza2lwIikKCmRlZiBhc2tfeWVzX25vKHByb21wdCwgZGVmYXVsdD1GYWxzZSk6CiAgICAiIiJBIHllcyBvciBubyBxdWVzdGlvbi4gQW55dGhpbmcgc3RhcnRpbmcgd2l0aCB5IG9yIG4gY291bnRzLiIiIgogICAgaGludCA9ICJZL24iIGlmIGRlZmF1bHQgZWxzZSAieS9OIgogICAgd2hpbGUgVHJ1ZToKICAgICAgICB0ZXh0ID0gaW5wdXQoInt9IFt7fV06ICIuZm9ybWF0KHByb21wdCwgaGludCkpLnN0cmlwKCkubG93ZXIoKQogICAgICAgIGlmIG5vdCB0ZXh0OgogICAgICAgICAgICByZXR1cm4gZGVmYXVsdAogICAgICAgIGlmIHRleHRbMF0gaW4gInluIjoKICAgICAgICAgICAgcmV0dXJuIHRleHRbMF0gPT0gInkiCiAgICAgICAgcHJpbnQoIiAgICBwbGVhc2UgYW5zd2VyIHllcyBvciBubyIpCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIHNoYXJlZCBtb2R1bGUgZnJhbWVfaW52ZW50b3J5LnB5CiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KClJPTEVTID0gKCJCIiwgIlYiLCAiUiIsICJIYSIpCgpST0xFX01FQU5JTkcgPSB7CiAgICAiQiI6ICJibHVlIiwKICAgICJWIjogInZpc3VhbCAtIHRoZSByZWZlcmVuY2UgZ3JpZCIsCiAgICAiUiI6ICJyZWQgY29udGludXVtIiwKICAgICJIYSI6ICJILWFscGhhIC0gdGhlIGVtaXNzaW9uIGxpbmUiLAp9CgpfUEFUVEVSTlMgPSB7CiAgICAiSGEiOiByZS5jb21waWxlKHIiKD88IVthLXowLTldKWhbXHNfLV0qYWwoPzpwaGF8ZmEpKD8hW2EtejAtOV0pIgogICAgICAgICAgICAgICAgICAgICByInwoPzwhW2EtejAtOV0paGEoPyFbYS16MC05XSkiKSwKICAgICJCIjogcmUuY29tcGlsZShyIig/PCFbYS16MC05XSkoPzpifGJlc3NlbGxbXHNfLV0qYnxicCkoPyFbYS16MC05XSkiKSwKICAgICJWIjogcmUuY29tcGlsZShyIig/PCFbYS16MC05XSkoPzp2fGJlc3NlbGxbXHNfLV0qdikoPyFbYS16MC05XSkiKSwKICAgICJSIjogcmUuY29tcGlsZShyIig/PCFbYS16MC05XSkoPzpyfHJwfHJjfHJwcmltZXxzbG9hbltcc18tXSpyfGJlc3NlbGxbXHNfLV0qcikiCiAgICAgICAgICAgICAgICAgICAgciIoPyFbYS16MC05XSkiKSwKfQoKX0NPTUJJTkVEX1dPUkRTID0gcmUuY29tcGlsZShyIlxiKG1lZGlhbnxhdmVyYWdlfG1lYW58c3VtKVxzK29mXGJ8XGJjb21iaW58XGJzdGFjayIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmUuSSkKCkZJVFNfU1VGRklYRVMgPSAoIi5maXRzIiwgIi5maXQiLCAiLmZ0cyIsICIuZml0cy5meiIpCgpPVVRQVVRfRk9MREVSUyA9IHsicmVzdWx0X2NtZCIsICJyZXN1bHQiLCAicmVzdWx0cyIsICJvdXRwdXQiLCAib3V0In0KCmRlZiByb2xlc19pbl90ZXh0KHRleHQpOgogICAgIiIiRXZlcnkgZmlsdGVyIHJvbGUgbmFtZWQgaW4gYSBwaWVjZSBvZiB0ZXh0LiIiIgogICAgaWYgbm90IHRleHQ6CiAgICAgICAgcmV0dXJuIHNldCgpCiAgICBsb3cgPSBzdHIodGV4dCkubG93ZXIoKQogICAgcmV0dXJuIHtyb2xlIGZvciByb2xlLCBwYXR0ZXJuIGluIF9QQVRURVJOUy5pdGVtcygpIGlmIHBhdHRlcm4uc2VhcmNoKGxvdyl9CgpfRElGRkVSRU5DRSA9IHJlLmNvbXBpbGUoCiAgICByIig/OmhbXHNfLV0qYWwoPzpwaGF8ZmEpfCg/PCFbYS16MC05XSloYSlbXi9cXF0qPy1ccyooPzpyfHJwfHJjKSg/IVthLXowLTldKSIKICAgIHIifG1pbnVzfGNvbnRbXHNfLV0qc3ViIiwgcmUuSSkKCmRlZiBpc19kaWZmZXJlbmNlX25hbWUobmFtZSk6CiAgICAiIiJUcnVlIHdoZW4gYSBmaWxlIG5hbWUgc2F5cyB0aGUgZnJhbWUgaXMgb25lIGZpbHRlciBtaW51cyBhbm90aGVyLCBzdWNoCiAgICBhcyBILWFscGhhIHdpdGggdGhlIFIgY29udGludXVtIHRha2VuIG9mZiAoIk04M19IYVgxMi1SLmZ0cyIpLiIiIgogICAgcmV0dXJuIGJvb2woX0RJRkZFUkVOQ0Uuc2VhcmNoKHN0cihuYW1lKSkpCgpfUFJPRFVDVF9ISVNUT1JZID0gcmUuY29tcGlsZSgKICAgIHIibXVsdGlwbGllZFxzK2J5fHN1YnRyYWN0ZWRcYi4qXC4oPzpmaXRzP3xmdHMpXGIiLCByZS5JKQoKZGVmIGlzX3Byb2R1Y3RfaGVhZGVyKGhlYWRlcik6CiAgICAiIiJUcnVlIHdoZW4gYSBISVNUT1JZIGxpbmUgc2F5cyB0aGUgZnJhbWUgd2FzIHNjYWxlZCBieSBhIGZhY3RvciBvciBoYWQKICAgIGFub3RoZXIgRklUUyBmaWxlIHN1YnRyYWN0ZWQgZnJvbSBpdC4gQSBjb25zdGFudCBza3kgKCJTdWJ0cmFjdGVkIDkuMCIpIGRvZXMKICAgIG5vdCBjb3VudC4iIiIKICAgIHJldHVybiBhbnkoX1BST0RVQ1RfSElTVE9SWS5zZWFyY2gobGluZSkgZm9yIGxpbmUgaW4gaGlzdG9yeV9saW5lcyhoZWFkZXIpKQoKZGVmIHJvbGVfZnJvbV9oZWFkZXIoaGVhZGVyKToKICAgICIiIlRoZSByb2xlIG5hbWVkIGJ5IHRoZSBmaWx0ZXIga2V5d29yZCwgb3IgTm9uZSB3aGVuIHRoZXJlIGlzIG5vIHN1Y2gga2V5d29yZC4iIiIKICAgIGZvciBrZXkgaW4gKCJGSUxURVIiLCAiRklMVEVSMSIsICJGSUxUTkFNRSIsICJJTlNGSUxURSIpOgogICAgICAgIHZhbHVlID0gaGVhZGVyLmdldChrZXkpCiAgICAgICAgaWYgdmFsdWUgaXMgTm9uZSBvciBzdHIodmFsdWUpLnN0cmlwKCkudXBwZXIoKSBpbiAoIiIsICJOT1RQUkVTRU5UIiwgIk5PTkUiKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICBmb3VuZCA9IHJvbGVzX2luX3RleHQodmFsdWUpCiAgICAgICAgaWYgbGVuKGZvdW5kKSA9PSAxOgogICAgICAgICAgICByZXR1cm4gZm91bmQucG9wKCkKICAgIHJldHVybiBOb25lCgpkZWYgaGlzdG9yeV9saW5lcyhoZWFkZXIpOgogICAgIiIiVGhlIGhpc3RvcnkgbGluZXMgb2YgYSBoZWFkZXIsIGFzIHBsYWluIHN0cmluZ3MuIiIiCiAgICB0cnk6CiAgICAgICAgcmV0dXJuIFtzdHIodikgZm9yIHYgaW4gaGVhZGVyWyJISVNUT1JZIl1dCiAgICBleGNlcHQgKEtleUVycm9yLCBUeXBlRXJyb3IpOgogICAgICAgIHJldHVybiBbXQoKZGVmIGlzX2NvbWJpbmVkKGhlYWRlcik6CiAgICAiIiJUcnVlIHdoZW4gdGhlIGhlYWRlciBzYXlzIHRoaXMgZnJhbWUgaXMgc2V2ZXJhbCBmcmFtZXMgY29tYmluZWQuIiIiCiAgICByZXR1cm4gYW55KF9DT01CSU5FRF9XT1JEUy5zZWFyY2gobGluZSkgZm9yIGxpbmUgaW4gaGlzdG9yeV9saW5lcyhoZWFkZXIpKQoKZGVmIGNvbWJpbmVfbm90ZShoZWFkZXIpOgogICAgIiIiVGhlIGhpc3RvcnkgbGluZSB0aGF0IHNheXMgaG93IHRoZSBmcmFtZSB3YXMgY29tYmluZWQsIGZvciB0aGUgc3VtbWFyeS4iIiIKICAgIGZvciBsaW5lIGluIGhpc3RvcnlfbGluZXMoaGVhZGVyKToKICAgICAgICBpZiBfQ09NQklORURfV09SRFMuc2VhcmNoKGxpbmUpOgogICAgICAgICAgICByZXR1cm4gbGluZS5zdHJpcCgpCiAgICByZXR1cm4gIiIKCmRlZiBoYXNfd2NzKGhlYWRlcik6CiAgICAiIiJUcnVlIHdoZW4gdGhlIGhlYWRlciBjYXJyaWVzIGEgcGxhdGUgc29sdXRpb24gdGhlIHRvb2xzIGNhbiB1c2UuIiIiCiAgICB0cnk6CiAgICAgICAgZnJvbSBhc3Ryb3B5LndjcyBpbXBvcnQgV0NTCiAgICAgICAgcmV0dXJuIFdDUyhoZWFkZXIpLmhhc19jZWxlc3RpYWwKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcmV0dXJuIEZhbHNlCgpkZWYgaW5zcGVjdChwYXRoKToKICAgICIiIkV2ZXJ5dGhpbmcgdGhlIHRvb2xzIG5lZWQgdG8ga25vdyBhYm91dCBvbmUgZmlsZS4iIiIKICAgIGZyb20gYXN0cm9weS5pbyBpbXBvcnQgZml0cwoKICAgIGhlYWRlciA9IGZpdHMuZ2V0aGVhZGVyKHBhdGgpCiAgICByb2xlID0gcm9sZV9mcm9tX2hlYWRlcihoZWFkZXIpCiAgICBzb3VyY2UgPSAidGhlIGZpbHRlciBrZXl3b3JkIGluIHRoZSBoZWFkZXIiCgogICAgaWYgaXNfcHJvZHVjdF9oZWFkZXIoaGVhZGVyKToKICAgICAgICAjIFRoZSBoZWFkZXIgc2F5cyBpdCBpcyB0aGlzIGZyYW1lIHNjYWxlZCwgb3IgYW5vdGhlciBmcmFtZSB0YWtlbiBvZmYKICAgICAgICAjIGl0ICgiTXVsdGlwbGllZCBieSAxMS41IiwgIlN1YnRyYWN0ZWQgLi4uUl9TVUJfTWVkaWFuX29mXzMiKTogbm90IGEKICAgICAgICAjIGZyYW1lIHRocm91Z2ggb25lIGZpbHRlciwgd2hhdGV2ZXIgaXRzIG5hbWUgb3IgRklMVEVSIGtleXdvcmQgc2F5cy4KICAgICAgICByb2xlID0gTm9uZQogICAgICAgIHNvdXJjZSA9ICJpdHMgaGlzdG9yeSAoYSBzY2FsZWQgb3Igc3VidHJhY3RlZCBwcm9kdWN0IC0gbm90IHVzZWQpIgogICAgZWxpZiByb2xlIGlzIE5vbmUgYW5kIGlzX2RpZmZlcmVuY2VfbmFtZShvcy5wYXRoLmJhc2VuYW1lKHBhdGgpKToKICAgICAgICAjICJIYVgxMi1SIiBpcyBILWFscGhhIHdpdGggMTIgeCBSIHRha2VuIG9mZjogYSBwcm9kdWN0LCBub3QgYSBmcmFtZQogICAgICAgICMgdGhyb3VnaCBlaXRoZXIgZmlsdGVyLiBOYW1pbmcgaXQgUiB3b3VsZCBwdXQgaXQgaW4gYXMgdGhlIGNvbnRpbnV1bS4KICAgICAgICBzb3VyY2UgPSAidGhlIGZpbGUgbmFtZSAoYSBkaWZmZXJlbmNlIG9mIHR3byBmcmFtZXMgLSBub3QgdXNlZCkiCiAgICBlbGlmIHJvbGUgaXMgTm9uZToKICAgICAgICBmb3VuZCA9IHJvbGVzX2luX3RleHQob3MucGF0aC5iYXNlbmFtZShwYXRoKSkKICAgICAgICBzb3VyY2UgPSAidGhlIGZpbGUgbmFtZSIKICAgICAgICBpZiBsZW4oZm91bmQpID09IDE6CiAgICAgICAgICAgIHJvbGUgPSBuZXh0KGl0ZXIoZm91bmQpKQoKICAgIHJldHVybiB7CiAgICAgICAgInBhdGgiOiBwYXRoLAogICAgICAgICJuYW1lIjogb3MucGF0aC5iYXNlbmFtZShwYXRoKSwKICAgICAgICAicm9sZSI6IHJvbGUsCiAgICAgICAgInJvbGVfc291cmNlIjogc291cmNlIGlmIHJvbGUgZWxzZSAiIiwKICAgICAgICAiY29tYmluZWQiOiBpc19jb21iaW5lZChoZWFkZXIpLAogICAgICAgICJjb21iaW5lX25vdGUiOiBjb21iaW5lX25vdGUoaGVhZGVyKSwKICAgICAgICAid2NzIjogaGFzX3djcyhoZWFkZXIpLAogICAgICAgICJleHB0aW1lIjogaGVhZGVyLmdldCgiRVhQVElNRSIpLAogICAgICAgICJvYmplY3QiOiAoc3RyKGhlYWRlci5nZXQoIk9CSkVDVCIpKS5zdHJpcCgpCiAgICAgICAgICAgICAgICAgICBpZiBoZWFkZXIuZ2V0KCJPQkpFQ1QiKSBlbHNlICIiKSwKICAgICAgICAic2hhcGUiOiAoaGVhZGVyLmdldCgiTkFYSVMyIiksIGhlYWRlci5nZXQoIk5BWElTMSIpKSwKICAgIH0KCmRlZiBzY2FuX2ZvbGRlcihmb2xkZXIsIHJlY3Vyc2l2ZT1UcnVlKToKICAgICIiIkluc3BlY3QgZXZlcnkgRklUUyBmaWxlIGluIGEgZm9sZGVyLiBSZXR1cm5zIGEgbGlzdCBvZiByZWNvcmRzLiIiIgogICAgZm91bmQgPSBbXQogICAgZm9yIHJvb3QsIGRpcnMsIGZpbGVzIGluIG9zLndhbGsoZm9sZGVyKToKICAgICAgICAjIERvIG5vdCB3YWxrIGludG8gd2hhdCBhbiBlYXJsaWVyIHJ1biB3cm90ZS4KICAgICAgICAjIEV2ZXJ5IHJ1biBub3cgd3JpdGVzIGEgZm9sZGVyIG9mIGl0cyBvd24gLSByZXN1bHRfQ01EXzxnYWxheHk+XzxkYXRlPiwKICAgICAgICAjIHJlc3VsdF9ISUlfLi4uLCByZXN1bHRfU3RhckZvcm1hdGlvbl8uLi4gLSBzbyBhbnl0aGluZyB3aG9zZSBuYW1lCiAgICAgICAgIyBzdGFydHMgd2l0aCAicmVzdWx0IiBpcyBvdXRwdXQsIG5vdCBkYXRhLgogICAgICAgIGRpcnNbOl0gPSBbZCBmb3IgZCBpbiBkaXJzIGlmIGQubG93ZXIoKSBub3QgaW4gT1VUUFVUX0ZPTERFUlMKICAgICAgICAgICAgICAgICAgIGFuZCBub3QgZC5sb3dlcigpLnN0YXJ0c3dpdGgoInJlc3VsdCIpXQogICAgICAgIGZvciBuYW1lIGluIHNvcnRlZChmaWxlcyk6CiAgICAgICAgICAgIGlmIG5vdCBuYW1lLmxvd2VyKCkuZW5kc3dpdGgoRklUU19TVUZGSVhFUyk6CiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICAjIFdoYXQgdGhlc2UgdG9vbHMgd3JpdGUgbmV4dCB0byB0aGVpciByZXN1bHRzIC0gYSBiYWNrZ3JvdW5kLQogICAgICAgICAgICAjIHN1YnRyYWN0ZWQgY29weSwgYSBmcmFtZSB3aXRoIGEgYm9ycm93ZWQgc29sdXRpb24gLSBpcyBub3QgZGF0YS4KICAgICAgICAgICAgaWYgcmUuc2VhcmNoKHIiXyhiZ3N1Ynxzb2x2ZWQpXC5maXRzPyQiLCBuYW1lLCByZS5JKToKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgIHBhdGggPSBvcy5wYXRoLmpvaW4ocm9vdCwgbmFtZSkKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgZm91bmQuYXBwZW5kKGluc3BlY3QocGF0aCkpCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOgogICAgICAgICAgICAgICAgZm91bmQuYXBwZW5kKHsicGF0aCI6IHBhdGgsICJuYW1lIjogbmFtZSwgInJvbGUiOiBOb25lLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAicm9sZV9zb3VyY2UiOiAiIiwgImNvbWJpbmVkIjogRmFsc2UsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICJjb21iaW5lX25vdGUiOiAiIiwgIndjcyI6IEZhbHNlLCAiZXhwdGltZSI6IE5vbmUsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICJvYmplY3QiOiAiIiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgInNoYXBlIjogKE5vbmUsIE5vbmUpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAiZXJyb3IiOiAie306IHt9Ii5mb3JtYXQodHlwZShleGMpLl9fbmFtZV9fLCBleGMpfSkKICAgICAgICBpZiBub3QgcmVjdXJzaXZlOgogICAgICAgICAgICBicmVhawogICAgcmV0dXJuIGZvdW5kCgpkZWYgZ2FsYXh5X2Zyb21fcmVjb3JkcyhyZWNvcmRzLCBmb2xkZXI9Tm9uZSk6CiAgICAiIiJUaGUgZ2FsYXh5J3MgbmFtZSwgdGFrZW4gZnJvbSB0aGUgZnJhbWVzIHRoZW1zZWx2ZXMuCgogICAgVGhlIHRlbGVzY29wZSB3cm90ZSBpdCBpbnRvIGV2ZXJ5IGV4cG9zdXJlIGl0IHRvb2ssIHNvIHRoZXJlIGlzIG5vIHJlYXNvbiB0bwogICAgYXNrIGZvciBpdCAtIGFuZCBubyByZWFzb24gdG8gcmVhZCBpdCBvZmYgYSBmb2xkZXIgbmFtZSwgd2hpY2ggaXMgd3JpdHRlbiBieSBhCiAgICBwZXJzb24gYW5kIGNhcnJpZXMgd2hhdGV2ZXIgZWxzZSB0aGV5IHdhbnRlZCB0byByZW1lbWJlciB0aGF0IGRheS4gQSBjb21iaW5lZAogICAgZnJhbWUgaGFzIHVzdWFsbHkgbG9zdCB0aGUga2V5d29yZCBhbG9uZyB3aXRoIHRoZSByZXN0IG9mIGl0cyBoZWFkZXIsIGJ1dCBpdAogICAgb25seSB0YWtlcyBvbmUgZXhwb3N1cmUgaW4gdGhlIGZvbGRlciB0aGF0IHN0aWxsIGhhcyBpdC4KCiAgICBSZXR1cm5zIChuYW1lLCB3aGVyZSBpdCBjYW1lIGZyb20pLiBUaGUgZm9sZGVyIG5hbWUgaXMgdGhlIGZhbGxiYWNrLCBjbGVhbmVkIG9mCiAgICB0aGUgbm90ZXMgcGVvcGxlIGFwcGVuZCB0byBpdC4KICAgICIiIgogICAgaW1wb3J0IGNvbGxlY3Rpb25zCgogICAgdm90ZXMgPSBjb2xsZWN0aW9ucy5Db3VudGVyKHIuZ2V0KCJvYmplY3QiLCAiIikgZm9yIHIgaW4gcmVjb3JkcwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIHIuZ2V0KCJvYmplY3QiKSkKICAgIGlmIHZvdGVzOgogICAgICAgIG5hbWUsIG4gPSB2b3Rlcy5tb3N0X2NvbW1vbigxKVswXQogICAgICAgIHJldHVybiBuYW1lLCAidGhlIE9CSkVDVCBrZXl3b3JkLCBpbiB7fSBvZiB7fSBmcmFtZXMiLmZvcm1hdChuLCBsZW4ocmVjb3JkcykpCgogICAgaWYgZm9sZGVyOgogICAgICAgIHN0ZW0gPSBvcy5wYXRoLmJhc2VuYW1lKG9zLnBhdGgubm9ybXBhdGgoZm9sZGVyKSkKICAgICAgICAjICJNMzMgMC40IG1ldGVyIiBpcyBhIG5vdGUgdG8gc2VsZiwgbm90IGEgY2F0YWxvZ3VlIG5hbWU7IHRoZSBuYW1lIGlzIHRoZQogICAgICAgICMgZmlyc3Qgd29yZCB0aGF0IGxvb2tzIGxpa2Ugb25lLgogICAgICAgIG0gPSByZS5tYXRjaChyIlxzKihbQS1aYS16XXsxLDR9XHMqXGQrW0EtWmEtel0/KSIsIHN0ZW0pCiAgICAgICAgaWYgbToKICAgICAgICAgICAgcmV0dXJuIHJlLnN1YihyIlxzKyIsICIgIiwgbS5ncm91cCgxKSkuc3RyaXAoKSwgInRoZSBmb2xkZXIgbmFtZSIKICAgICAgICByZXR1cm4gc3RlbSwgInRoZSBmb2xkZXIgbmFtZSIKCiAgICByZXR1cm4gTm9uZSwgIiIKCmRlZiBjaG9vc2UocmVjb3Jkcywgcm9sZXM9Uk9MRVMsIGFzaz1Ob25lKToKICAgICIiIlBpY2sgb25lIGZyYW1lIHBlciByb2xlOiB0aGUgY29tYmluZWQgb25lLCB3aGVuIHRoZXJlIGlzIG9uZS4KCiAgICBBIGNvbWJpbmVkIGZyYW1lIGlzIGRlZXBlciB0aGFuIGFueSBzaW5nbGUgZXhwb3N1cmUsIHdoaWNoIGlzIHRoZSByZWFzb24gaXQKICAgIHdhcyBtYWRlLCBzbyBpdCB3aW5zIHdoZW5ldmVyIG9uZSBleGlzdHMuIFdoZW4gdGhlcmUgaXMgbm9uZSBidXQgc2V2ZXJhbAogICAgZXhwb3N1cmVzLCBhbGwgb2YgdGhlbSBhcmUgcmV0dXJuZWQgdG8gYmUgY29tYmluZWQgcmF0aGVyIHRoYW4gb25lIGJlaW5nCiAgICBwaWNrZWQgLSB0aGF0IGlzIHdoYXQgdGhlIG9ic2VydmVyIHdlbnQgb3V0IGFuZCB0b29rLgoKICAgIFJldHVybnMgKGNob3NlbiwgbWVtYmVycywgbm90ZXMpOiB0aGUgcmVwcmVzZW50YXRpdmUgZnJhbWUgcGVyIHJvbGUsIHRoZQogICAgZXhwb3N1cmVzIHRvIGNvbWJpbmUgd2hlcmUgdGhlcmUgYXJlIHNldmVyYWwsIGFuZCBhbnl0aGluZyB3b3J0aCBzYXlpbmcuCiAgICAiIiIKICAgIGNob3NlbiwgbWVtYmVycywgbm90ZXMgPSB7fSwge30sIFtdCgogICAgZm9yIHJvbGUgaW4gcm9sZXM6CiAgICAgICAgbWluZSA9IFtyIGZvciByIGluIHJlY29yZHMgaWYgci5nZXQoInJvbGUiKSA9PSByb2xlIGFuZCBub3Qgci5nZXQoImVycm9yIildCiAgICAgICAgaWYgbm90IG1pbmU6CiAgICAgICAgICAgIG5vdGVzLmFwcGVuZCgie306IG5vdGhpbmcgaW4gdGhlIGZvbGRlciBpZGVudGlmaWVzIGl0c2VsZiBhcyB7fSIuZm9ybWF0KAogICAgICAgICAgICAgICAgcm9sZSwgUk9MRV9NRUFOSU5HW3JvbGVdKSkKICAgICAgICAgICAgY29udGludWUKCiAgICAgICAgY29tYmluZWQgPSBbciBmb3IgciBpbiBtaW5lIGlmIHJbImNvbWJpbmVkIl1dCiAgICAgICAgaWYgbGVuKGNvbWJpbmVkKSA9PSAxOgogICAgICAgICAgICBjaG9zZW5bcm9sZV0gPSBjb21iaW5lZFswXQogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGlmIGxlbihjb21iaW5lZCkgPiAxOgogICAgICAgICAgICBpZiBhc2sgaXMgTm9uZToKICAgICAgICAgICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigie306IHt9IGNvbWJpbmVkIGZyYW1lcyB0byBjaG9vc2UgZnJvbSIuZm9ybWF0KAogICAgICAgICAgICAgICAgICAgIHJvbGUsIGxlbihjb21iaW5lZCkpKQogICAgICAgICAgICBjaG9zZW5bcm9sZV0gPSBhc2socm9sZSwgY29tYmluZWQpCiAgICAgICAgICAgIGNvbnRpbnVlCgogICAgICAgIHNpbmdsZXMgPSBbciBmb3IgciBpbiBtaW5lIGlmIHJbIndjcyJdXSBvciBtaW5lCiAgICAgICAgaWYgbGVuKHNpbmdsZXMpID09IDE6CiAgICAgICAgICAgIGNob3Nlbltyb2xlXSA9IHNpbmdsZXNbMF0KICAgICAgICAgICAgbm90ZXMuYXBwZW5kKCJ7fTogbm8gY29tYmluZWQgZnJhbWUsIHVzaW5nIHRoZSBvbmx5IGV4cG9zdXJlIi5mb3JtYXQocm9sZSkpCiAgICAgICAgZWxzZToKICAgICAgICAgICAgIyBTZXZlcmFsIGV4cG9zdXJlcyBhbmQgbm90aGluZyBjb21iaW5lZC4gUGlja2luZyBvbmUgb2YgdGhlbSBhbmQKICAgICAgICAgICAgIyBkaXNjYXJkaW5nIHRoZSByZXN0IHRocm93cyBhd2F5IG1vc3Qgb2YgdGhlIG9ic2VydmF0aW9uIC0gd2l0aCBzaXgKICAgICAgICAgICAgIyBILWFscGhhIGZyYW1lcyBpdCBjb3N0cyBhIGZhY3RvciBvZiAyLjQgaW4gbm9pc2UgLSBzbyB0aGV5IGFyZSBhbGwKICAgICAgICAgICAgIyBrZXB0IGFuZCB0aGUgY2FsbGVyIGNvbWJpbmVzIHRoZW0uIFRoZSBmaXJzdCBpcyBuYW1lZCBhcyB0aGUKICAgICAgICAgICAgIyByZXByZXNlbnRhdGl2ZSBiZWNhdXNlIHRoZSBvdXRwdXQgaGFzIHRvIGJlIGxhYmVsbGVkIHdpdGggc29tZXRoaW5nLgogICAgICAgICAgICBjaG9zZW5bcm9sZV0gPSBzaW5nbGVzWzBdCiAgICAgICAgICAgIG1lbWJlcnNbcm9sZV0gPSBzaW5nbGVzCiAgICAgICAgICAgIG5vdGVzLmFwcGVuZCgie306IG5vIGNvbWJpbmVkIGZyYW1lOyB7fSBleHBvc3VyZXMgdG8gY29tYmluZSIuZm9ybWF0KAogICAgICAgICAgICAgICAgcm9sZSwgbGVuKHNpbmdsZXMpKSkKCiAgICByZXR1cm4gY2hvc2VuLCBtZW1iZXJzLCBub3RlcwoKZGVmIGRlc2NyaWJlKHJlY29yZHMsIGNob3Nlbiwgbm90ZXMpOgogICAgIiIiVGhlIHN1bW1hcnkgYSBwZXJzb24gcmVhZHMgYmVmb3JlIGFncmVlaW5nIHRvIHRoZSBydW4uIiIiCiAgICBsaW5lcyA9IFsiIiwgIldoYXQgdGhlIGZvbGRlciBob2xkczoiXQogICAgYnlfcm9sZSA9IHt9CiAgICBmb3IgciBpbiByZWNvcmRzOgogICAgICAgIGJ5X3JvbGUuc2V0ZGVmYXVsdChyLmdldCgicm9sZSIpIG9yICI/IiwgW10pLmFwcGVuZChyKQoKICAgIGZvciByb2xlIGluIGxpc3QoUk9MRVMpICsgWyI/Il06CiAgICAgICAgZ3JvdXAgPSBieV9yb2xlLmdldChyb2xlKQogICAgICAgIGlmIG5vdCBncm91cDoKICAgICAgICAgICAgY29udGludWUKICAgICAgICBsaW5lcy5hcHBlbmQoIiAgIiArICgie30gKHt9KSIuZm9ybWF0KHJvbGUsIFJPTEVfTUVBTklOR1tyb2xlXSkKICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiByb2xlIGluIFJPTEVfTUVBTklORyBlbHNlICJub3QgaWRlbnRpZmllZCIpKQogICAgICAgIGZvciByIGluIHNvcnRlZChncm91cCwga2V5PWxhbWJkYSB4OiB4WyJuYW1lIl0pOgogICAgICAgICAgICBtYXJrID0gIi0+IiBpZiBjaG9zZW4uZ2V0KHJvbGUpIGlzIHIgZWxzZSAiICAiCiAgICAgICAgICAgIGlmIHIuZ2V0KCJlcnJvciIpOgogICAgICAgICAgICAgICAgYml0cyA9IFsidW5yZWFkYWJsZTogIiArIHJbImVycm9yIl1dCiAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICBiaXRzID0gWyJjb21iaW5lZCIgaWYgclsiY29tYmluZWQiXSBlbHNlICJzaW5nbGUiLAogICAgICAgICAgICAgICAgICAgICAgICAic29sdmVkIiBpZiByWyJ3Y3MiXSBlbHNlICJOTyBXQ1MiXQogICAgICAgICAgICBsaW5lcy5hcHBlbmQoIiAgICB7fSB7Ojw0Mn0ge30iLmZvcm1hdChtYXJrLCByWyJuYW1lIl0sICIsICIuam9pbihiaXRzKSkpCgogICAgbGluZXMuYXBwZW5kKCIiKQogICAgbGluZXMuYXBwZW5kKCJDaG9zZW4gZm9yIHRoZSBtZWFzdXJlbWVudDoiKQogICAgZm9yIHJvbGUgaW4gUk9MRVM6CiAgICAgICAgciA9IGNob3Nlbi5nZXQocm9sZSkKICAgICAgICBsaW5lcy5hcHBlbmQoIiAgICB7OjwzfSB7fSIuZm9ybWF0KHJvbGUsIHJbIm5hbWUiXSBpZiByIGVsc2UgIm5vdGhpbmciKSkKICAgIGZvciBuIGluIG5vdGVzOgogICAgICAgIGxpbmVzLmFwcGVuZCgiICAgIG5vdGU6ICIgKyBuKQogICAgbGluZXMuYXBwZW5kKCIiKQogICAgcmV0dXJuICJcbiIuam9pbihsaW5lcykKCmRlZiB3Y3NfZnJvbV9yZWZlcmVuY2UoaW1hZ2UsIHJlZmVyZW5jZV9pbWFnZSwgcmVmZXJlbmNlX3djcywKICAgICAgICAgICAgICAgICAgICAgICBkZXRlY3Rpb25fc2lnbWE9Ni4wLCBtYXhfY29udHJvbF9wb2ludHM9NjApOgogICAgIiIiRml0IGEgcGxhdGUgc29sdXRpb24gZm9yIGFuIHVuc29sdmVkIGZyYW1lIGJ5IG1hdGNoaW5nIGl0cyBzdGFycy4KCiAgICBBIGNvbWJpbmVkIGZyYW1lIGFuZCB0aGUgc2luZ2xlIGV4cG9zdXJlcyBpdCB3YXMgYnVpbHQgZnJvbSBzaG93IHRoZSBzYW1lCiAgICBzdGFycyBpbiBhbG1vc3QgdGhlIHNhbWUgcGxhY2VzLCBzbyBhIHNvbHZlZCBleHBvc3VyZSBvZiB0aGUgc2FtZSBmaWVsZCBjYW4KICAgIGxlbmQgaXRzIHNvbHV0aW9uOiBtYXRjaCB0aGUgc3RhciBwYXR0ZXJucywgcmVhZCBlYWNoIG1hdGNoZWQgc3RhcidzIHNreQogICAgcG9zaXRpb24gZnJvbSB0aGUgc29sdmVkIGZyYW1lLCBhbmQgZml0IGEgc29sdXRpb24gdG8gdGhvc2UgcG9zaXRpb25zIGluIHRoZQogICAgdW5zb2x2ZWQgZnJhbWUncyBvd24gcGl4ZWxzLgoKICAgIFRoaXMgaXMgbm90IGNvcHlpbmcgYSBoZWFkZXIgYWNyb3NzLiBUaGUgcmVzdWx0IGlzIGZpdHRlZCB0byBtYXRjaGVkIHN0YXJzIGFuZAogICAgY29tZXMgd2l0aCB0aGUgcmVzaWR1YWwgb2YgdGhhdCBmaXQsIHdoaWNoIGlzIHRoZSBob25lc3QgbWVhc3VyZSBvZiBob3cgd2VsbAogICAgaXQgZGlkOiBhIGZldyBodW5kcmVkdGhzIG9mIGEgcGl4ZWwgbWVhbnMgdGhlIHR3byBmcmFtZXMgcmVhbGx5IGFyZSBvbiB0aGUKICAgIHNhbWUgZ3JpZCwgYW5kIGEgcmVzaWR1YWwgYXBwcm9hY2hpbmcgYSBwaXhlbCBtZWFucyBzb21ldGhpbmcgbW92ZWQuCgogICAgUmV0dXJucyAod2NzLCBkaWFnbm9zdGljcykuCiAgICAiIiIKICAgIGltcG9ydCBudW1weSBhcyBucAogICAgaW1wb3J0IGFzdHJvYWxpZ24gYXMgYWEKICAgIGZyb20gYXN0cm9weS53Y3MudXRpbHMgaW1wb3J0IGZpdF93Y3NfZnJvbV9wb2ludHMKCiAgICB0cmFuc2Zvcm0sIChzcmMsIGRzdCkgPSBhYS5maW5kX3RyYW5zZm9ybSgKICAgICAgICBpbWFnZSwgcmVmZXJlbmNlX2ltYWdlLAogICAgICAgIG1heF9jb250cm9sX3BvaW50cz1tYXhfY29udHJvbF9wb2ludHMsCiAgICAgICAgZGV0ZWN0aW9uX3NpZ21hPWRldGVjdGlvbl9zaWdtYSkKCiAgICBza3kgPSByZWZlcmVuY2Vfd2NzLnBpeGVsX3RvX3dvcmxkKGRzdFs6LCAwXSwgZHN0WzosIDFdKQogICAgZml0dGVkID0gZml0X3djc19mcm9tX3BvaW50cygoc3JjWzosIDBdLCBzcmNbOiwgMV0pLCBza3ksIHByb2pfcG9pbnQ9ImNlbnRlciIpCgogICAgYmFja194LCBiYWNrX3kgPSBmaXR0ZWQud29ybGRfdG9fcGl4ZWwoc2t5KQogICAgcmVzaWR1YWwgPSBucC5oeXBvdChiYWNrX3ggLSBzcmNbOiwgMF0sIGJhY2tfeSAtIHNyY1s6LCAxXSkKCiAgICByZXR1cm4gZml0dGVkLCB7CiAgICAgICAgIm5fc3RhcnMiOiBpbnQobGVuKHNyYykpLAogICAgICAgICJtZWRpYW5fcmVzaWR1YWxfcHgiOiBmbG9hdChucC5tZWRpYW4ocmVzaWR1YWwpKSwKICAgICAgICAid29yc3RfcmVzaWR1YWxfcHgiOiBmbG9hdChyZXNpZHVhbC5tYXgoKSksCiAgICAgICAgInNoaWZ0X3B4IjogKGZsb2F0KHRyYW5zZm9ybS50cmFuc2xhdGlvblswXSksIGZsb2F0KHRyYW5zZm9ybS50cmFuc2xhdGlvblsxXSkpLAogICAgICAgICJyb3RhdGlvbl9kZWciOiBmbG9hdChucC5kZWdyZWVzKHRyYW5zZm9ybS5yb3RhdGlvbikpLAogICAgICAgICJzY2FsZSI6IGZsb2F0KHRyYW5zZm9ybS5zY2FsZSksCiAgICB9CgpkZWYgZGVzY3JpYmVfd2NzX2ZpdChuYW1lLCBkaWFnKToKICAgICIiIk9uZSByZWFkYWJsZSBwYXJhZ3JhcGggYWJvdXQgYSBzb2x1dGlvbiBsZW50IGZyb20gYW5vdGhlciBmcmFtZS4iIiIKICAgIHJldHVybiAoCiAgICAgICAgIiAge306IHNvbHZlZCBieSBtYXRjaGluZyB7fSBzdGFycyBhZ2FpbnN0IGFuIGV4cG9zdXJlIHRoYXQgaGFzIGEgc29sdXRpb25cbiIKICAgICAgICAiICAgIHNoaWZ0ICh7OisuMWZ9LCB7OisuMWZ9KSBweCwgcm90YXRpb24gezorLjNmfSBkZWcsIHNjYWxlIHs6LjVmfVxuIgogICAgICAgICIgICAgZml0IHJlc2lkdWFsOiBtZWRpYW4gezouMmZ9IHB4LCB3b3JzdCB7Oi4yZn0gcHgiLmZvcm1hdCgKICAgICAgICAgICAgbmFtZSwgZGlhZ1sibl9zdGFycyJdLCBkaWFnWyJzaGlmdF9weCJdWzBdLCBkaWFnWyJzaGlmdF9weCJdWzFdLAogICAgICAgICAgICBkaWFnWyJyb3RhdGlvbl9kZWciXSwgZGlhZ1sic2NhbGUiXSwKICAgICAgICAgICAgZGlhZ1sibWVkaWFuX3Jlc2lkdWFsX3B4Il0sIGRpYWdbIndvcnN0X3Jlc2lkdWFsX3B4Il0pKQoKZGVmIGRvbm9yX2ZvcihyZWNvcmQsIHJlY29yZHMpOgogICAgIiIiQSBzb2x2ZWQgZXhwb3N1cmUgdGhhdCBjYW4gbGVuZCBpdHMgcGxhdGUgc29sdXRpb24gdG8gYW4gdW5zb2x2ZWQgZnJhbWUuCgogICAgQW4gZXhwb3N1cmUgdGhyb3VnaCB0aGUgc2FtZSBmaWx0ZXIgaXMgcHJlZmVycmVkIC0gc2FtZSBzdGFycywgc2FtZSBkZXB0aCwKICAgIHRoZSBjbG9zZXN0IHRoaW5nIHRvIHRoZSBmcmFtZSB0aGF0IG5lZWRzIHNvbHZpbmcgLSBidXQgYW55IHNvbHZlZCBleHBvc3VyZQogICAgb2YgdGhlIGZpZWxkIHdpbGwgZG8sIHNpbmNlIHRoZSBtYXRjaCBpcyBtYWRlIG9uIHN0YXIgcG9zaXRpb25zIGFuZCB0aG9zZSBkbwogICAgbm90IGNhcmUgd2hpY2ggZmlsdGVyIHRoZXkgd2VyZSBzZWVuIHRocm91Z2guCiAgICAiIiIKICAgIGlmIHJlY29yZCBpcyBOb25lIG9yIHJlY29yZC5nZXQoIndjcyIpOgogICAgICAgIHJldHVybiBOb25lCiAgICBzYW1lX2ZpbHRlciA9IFtyIGZvciByIGluIHJlY29yZHMKICAgICAgICAgICAgICAgICAgIGlmIHIuZ2V0KCJ3Y3MiKSBhbmQgci5nZXQoInJvbGUiKSA9PSByZWNvcmQuZ2V0KCJyb2xlIildCiAgICBpZiBzYW1lX2ZpbHRlcjoKICAgICAgICByZXR1cm4gc2FtZV9maWx0ZXJbMF0KICAgIHNvbHZlZCA9IFtyIGZvciByIGluIHJlY29yZHMgaWYgci5nZXQoIndjcyIpXQogICAgcmV0dXJuIHNvbHZlZFswXSBpZiBzb2x2ZWQgZWxzZSBOb25lCgpkZWYgcGxhbihmb2xkZXIsIHJvbGVzPVJPTEVTLCBhc2s9Tm9uZSwgcmVjdXJzaXZlPVRydWUpOgogICAgIiIiRXZlcnl0aGluZyBhIHRvb2wgbmVlZHMgZnJvbSBvbmUgZm9sZGVyIHBhdGguCgogICAgUmV0dXJucyAocmVjb3JkcywgY2hvc2VuLCBtZW1iZXJzLCBkb25vcnMsIG5vdGVzKTogd2hhdCBpcyBpbiB0aGUgZm9sZGVyLCB0aGUKICAgIGZyYW1lIGNob3NlbiBmb3IgZWFjaCByb2xlLCB0aGUgc2V2ZXJhbCBleHBvc3VyZXMgdG8gY29tYmluZSB3aGVuIHRoZXJlIHdhcyBubwogICAgY29tYmluZWQgZnJhbWUsIHdoaWNoIHNvbHZlZCBleHBvc3VyZSB3aWxsIGxlbmQgYSBwbGF0ZSBzb2x1dGlvbiB0byBhIGZyYW1lCiAgICB0aGF0IGxhY2tzIG9uZSwgYW5kIGFueXRoaW5nIHdvcnRoIHNheWluZyBvdXQgbG91ZC4KICAgICIiIgogICAgcmVjb3JkcyA9IHNjYW5fZm9sZGVyKGZvbGRlciwgcmVjdXJzaXZlPXJlY3Vyc2l2ZSkKICAgIGlmIG5vdCByZWNvcmRzOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigibm8gRklUUyBmaWxlcyBpbiAiICsgc3RyKGZvbGRlcikpCiAgICBjaG9zZW4sIG1lbWJlcnMsIG5vdGVzID0gY2hvb3NlKHJlY29yZHMsIHJvbGVzPXJvbGVzLCBhc2s9YXNrKQogICAgZG9ub3JzID0ge3JvbGU6IGRvbm9yX2ZvcihyZWMsIHJlY29yZHMpIGZvciByb2xlLCByZWMgaW4gY2hvc2VuLml0ZW1zKCl9CiAgICBmb3Igcm9sZSwgZG9ub3IgaW4gZG9ub3JzLml0ZW1zKCk6CiAgICAgICAgaWYgZG9ub3IgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIG5vdGVzLmFwcGVuZCgie306IGhhcyBubyBwbGF0ZSBzb2x1dGlvbjsgd2lsbCBib3Jyb3cgb25lIGZyb20ge30iLmZvcm1hdCgKICAgICAgICAgICAgICAgIHJvbGUsIGRvbm9yWyJuYW1lIl0pKQogICAgICAgIGVsaWYgcm9sZSBpbiBjaG9zZW4gYW5kIG5vdCBjaG9zZW5bcm9sZV1bIndjcyJdOgogICAgICAgICAgICBub3Rlcy5hcHBlbmQoInt9OiBoYXMgbm8gcGxhdGUgc29sdXRpb24gYW5kIHRoZXJlIGlzIG5vIHNvbHZlZCAiCiAgICAgICAgICAgICAgICAgICAgICAgICAiZXhwb3N1cmUgaW4gdGhlIGZvbGRlciB0byB0YWtlIG9uZSBmcm9tIi5mb3JtYXQocm9sZSkpCiAgICByZXR1cm4gcmVjb3JkcywgY2hvc2VuLCBtZW1iZXJzLCBkb25vcnMsIG5vdGVzCgpkZXNjcmliZV9fZnJhbWVfaW52ZW50b3J5ID0gZGVzY3JpYmUgICAjIHRoaXMgbW9kdWxlJ3Mgb3duOyB0aGUgcGxhaW4gbmFtZSBnZXRzIHNoYWRvd2VkIGJlbG93CgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIHNoYXJlZCBtb2R1bGUgYW5jaG9yX3JlZ2lzdHJhdGlvbi5weQojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCgpkZWYgcGlja19hbmNob3JzKHBvc2l0aW9ucywgc2hhcGUsIG5fYW5jaG9ycz00LCBlZGdlX21hcmdpbj00MCk6CiAgICAiIiJDaG9vc2UgYSBmZXcgd2lkZWx5IHNlcGFyYXRlZCBzdGFycyBmcm9tIGEgbGlzdCBvZiBjYW5kaWRhdGUgcG9zaXRpb25zLgoKICAgIFNwcmVhZCBpcyB0aGUgd2hvbGUgcG9pbnQuIFRoZSByb3RhdGlvbiBhbmdsZSBpcyBtZWFzdXJlZCBhcyBhIGRpZmZlcmVuY2Ugb2YKICAgIGRpcmVjdGlvbnMgYmV0d2VlbiB0d28gc3RhcnMsIHNvIGl0cyBwcmVjaXNpb24gaXMgdGhlIGNlbnRyaW5nIHByZWNpc2lvbgogICAgZGl2aWRlZCBieSB0aGUgZGlzdGFuY2UgYmV0d2VlbiB0aGVtOiBhbmNob3JzIGNsb3NlIHRvZ2V0aGVyIG1lYXN1cmUgdGhlCiAgICBhbmdsZSBiYWRseSBubyBtYXR0ZXIgaG93IHdlbGwgZWFjaCBvbmUgaXMgY2VudHJlZC4KCiAgICBUaGUgZnJhbWUgaXMgZGl2aWRlZCBpbnRvIHF1YWRyYW50cyBhbmQgb25lIHN0YXIgaXMgdGFrZW4gZnJvbSBlYWNoLCBhcyBmYXIKICAgIGZyb20gdGhlIGNlbnRyZSBhcyBpdCBjYW4gYmUgZm91bmQuIFRoYXQgZ3VhcmFudGVlcyB0aGUgc3ByZWFkIGluc3RlYWQgb2YKICAgIGhvcGluZyBmb3IgaXQuCiAgICAiIiIKICAgIHBvc2l0aW9ucyA9IG5wLmFzYXJyYXkocG9zaXRpb25zLCBkdHlwZT1mbG9hdCkKICAgIGlmIGxlbihwb3NpdGlvbnMpID09IDA6CiAgICAgICAgcmV0dXJuIHBvc2l0aW9ucwoKICAgIG55LCBueCA9IHNoYXBlCiAgICBpbnNpZGUgPSAoKHBvc2l0aW9uc1s6LCAwXSA+IGVkZ2VfbWFyZ2luKSAmIChwb3NpdGlvbnNbOiwgMF0gPCBueCAtIGVkZ2VfbWFyZ2luKQogICAgICAgICAgICAgICYgKHBvc2l0aW9uc1s6LCAxXSA+IGVkZ2VfbWFyZ2luKSAmIChwb3NpdGlvbnNbOiwgMV0gPCBueSAtIGVkZ2VfbWFyZ2luKSkKICAgIHBvc2l0aW9ucyA9IHBvc2l0aW9uc1tpbnNpZGVdCiAgICBpZiBsZW4ocG9zaXRpb25zKSA8PSBuX2FuY2hvcnM6CiAgICAgICAgcmV0dXJuIHBvc2l0aW9ucwoKICAgIGN4LCBjeSA9IG54IC8gMi4wLCBueSAvIDIuMAogICAgZHgsIGR5ID0gcG9zaXRpb25zWzosIDBdIC0gY3gsIHBvc2l0aW9uc1s6LCAxXSAtIGN5CiAgICByYWRpdXMgPSBucC5oeXBvdChkeCwgZHkpCgogICAgY2hvc2VuID0gW10KICAgIHF1YWRyYW50cyA9ICgoZHggPj0gMCkgJiAoZHkgPj0gMCksIChkeCA8IDApICYgKGR5ID49IDApLAogICAgICAgICAgICAgICAgIChkeCA8IDApICYgKGR5IDwgMCksIChkeCA+PSAwKSAmIChkeSA8IDApKQogICAgZm9yIHEgaW4gcXVhZHJhbnRzOgogICAgICAgIGlmIHEuYW55KCk6CiAgICAgICAgICAgIGNob3Nlbi5hcHBlbmQocG9zaXRpb25zW25wLndoZXJlKHEpWzBdW25wLmFyZ21heChyYWRpdXNbcV0pXV0pCgogICAgIyBBIGZpZWxkIHdpdGggYW4gZW1wdHkgcXVhZHJhbnQgc3RpbGwgZ2V0cyBpdHMgZm91ciwgdGFrZW4gZnJvbSB3aGVyZXZlciB0aGUKICAgICMgc3RhcnMgYWN0dWFsbHkgYXJlLgogICAgaWYgbGVuKGNob3NlbikgPCBuX2FuY2hvcnM6CiAgICAgICAgb3JkZXIgPSBucC5hcmdzb3J0KHJhZGl1cylbOjotMV0KICAgICAgICBmb3IgaSBpbiBvcmRlcjoKICAgICAgICAgICAgcCA9IHBvc2l0aW9uc1tpXQogICAgICAgICAgICBpZiBub3QgYW55KG5wLmFsbGNsb3NlKHAsIGMpIGZvciBjIGluIGNob3Nlbik6CiAgICAgICAgICAgICAgICBjaG9zZW4uYXBwZW5kKHApCiAgICAgICAgICAgIGlmIGxlbihjaG9zZW4pID49IG5fYW5jaG9yczoKICAgICAgICAgICAgICAgIGJyZWFrCgogICAgcmV0dXJuIG5wLmFycmF5KGNob3Nlbls6bl9hbmNob3JzXSkKCmRlZiBjZW50cmVfb24oaW1hZ2UsIHBvc2l0aW9ucywgYm94PTE1KToKICAgICIiIldoZXJlIGVhY2ggc3RhciBhY3R1YWxseSBzaXRzIGluIHRoaXMgZnJhbWUsIHRvIGEgZnJhY3Rpb24gb2YgYSBwaXhlbC4iIiIKICAgIGZyb20gcGhvdHV0aWxzLmNlbnRyb2lkcyBpbXBvcnQgY2VudHJvaWRfc291cmNlcywgY2VudHJvaWRfY29tCgogICAgcG9zaXRpb25zID0gbnAuYXNhcnJheShwb3NpdGlvbnMsIGR0eXBlPWZsb2F0KQogICAgaWYgbm90IGxlbihwb3NpdGlvbnMpOgogICAgICAgIHJldHVybiBucC5lbXB0eSgoMCwgMikpCiAgICB0cnk6CiAgICAgICAgeCwgeSA9IGNlbnRyb2lkX3NvdXJjZXMobnAubmFuX3RvX251bShpbWFnZSksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcG9zaXRpb25zWzosIDBdLCBwb3NpdGlvbnNbOiwgMV0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYm94X3NpemU9Ym94LCBjZW50cm9pZF9mdW5jPWNlbnRyb2lkX2NvbSkKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcmV0dXJuIG5wLmVtcHR5KCgwLCAyKSkKICAgIHJldHVybiBucC5jb2x1bW5fc3RhY2soW3gsIHldKQoKZGVmIGFuY2hvcl90cmFuc2Zvcm0ocmVmZXJlbmNlLCBpbWFnZSwgcG9zaXRpb25zLCBib3g9MTUsIG1heF9tb3ZlX3B4PTguMCwKICAgICAgICAgICAgICAgICAgICAgbl9hbmNob3JzPTQpOgogICAgIiIiRml0IHNoaWZ0LCByb3RhdGlvbiBhbmQgc2NhbGUgZnJvbSBhIGZldyBhbmNob3Igc3RhcnMuCgogICAgUmV0dXJucyAodHJhbnNmb3JtLCBkaWFnbm9zdGljcykgd2hlcmUgdHJhbnNmb3JtIG1hcHMgYSBwb3NpdGlvbiBpbiBgaW1hZ2VgCiAgICB0byB0aGUgbWF0Y2hpbmcgcG9zaXRpb24gaW4gYHJlZmVyZW5jZWAsIG9yIChOb25lLCBkaWFnbm9zdGljcykgd2hlbiB0aGUKICAgIGFuY2hvcnMgY291bGQgbm90IGJlIG1lYXN1cmVkIHdlbGwgZW5vdWdoIHRvIHRydXN0LgoKICAgIFRoZSBvZGQgb25lIG91dCBpcyBkcm9wcGVkOiBlYWNoIGFuY2hvcidzIHJlc2lkdWFsIGFnYWluc3QgdGhlIGZpdCBpcwogICAgbWVhc3VyZWQsIGFuZCBpZiBvbmUgc3RhbmRzIGZhciBhYm92ZSB0aGUgcmVzdCBpdCBpcyByZW1vdmVkIGFuZCB0aGUgZml0CiAgICByZWRvbmUuIFRoYXQgaXMgd2hhdCB0aGUgZm91cnRoIGFuY2hvciBpcyBmb3IuCiAgICAiIiIKICAgIGFuY2hvcnMgPSBwaWNrX2FuY2hvcnMocG9zaXRpb25zLCByZWZlcmVuY2Uuc2hhcGUsIG5fYW5jaG9ycz1uX2FuY2hvcnMpCiAgICBkaWFnID0geyJuX2FuY2hvcnMiOiBsZW4oYW5jaG9ycyksICJkcm9wcGVkIjogMH0KICAgIGlmIGxlbihhbmNob3JzKSA8IDI6CiAgICAgICAgZGlhZ1sid2h5Il0gPSAiZmV3ZXIgdGhhbiB0d28gdXNhYmxlIGFuY2hvciBzdGFycyIKICAgICAgICByZXR1cm4gTm9uZSwgZGlhZwoKICAgIGEgPSBjZW50cmVfb24ocmVmZXJlbmNlLCBhbmNob3JzLCBib3g9Ym94KQogICAgYiA9IGNlbnRyZV9vbihpbWFnZSwgYW5jaG9ycywgYm94PWJveCkKICAgIGlmIGxlbihhKSAhPSBsZW4oYikgb3Igbm90IGxlbihhKToKICAgICAgICBkaWFnWyJ3aHkiXSA9ICJ0aGUgYW5jaG9ycyBjb3VsZCBub3QgYmUgY2VudHJlZCIKICAgICAgICByZXR1cm4gTm9uZSwgZGlhZwoKICAgIG9rID0gKG5wLmlzZmluaXRlKGEpLmFsbChheGlzPTEpICYgbnAuaXNmaW5pdGUoYikuYWxsKGF4aXM9MSkKICAgICAgICAgICYgKG5wLmh5cG90KCooYiAtIGEpLlQpIDwgbWF4X21vdmVfcHgpKQogICAgYSwgYiA9IGFbb2tdLCBiW29rXQogICAgaWYgbGVuKGEpIDwgMjoKICAgICAgICBkaWFnWyJ3aHkiXSA9ICJ0aGUgYW5jaG9ycyBtb3ZlZCBmdXJ0aGVyIHRoYW4gYSBmcmFtZSBsaWtlIHRoaXMgc2hvdWxkIgogICAgICAgIHJldHVybiBOb25lLCBkaWFnCgogICAgZGVmIGZpdChwLCBxKToKICAgICAgICAiIiJMZWFzdC1zcXVhcmVzIHNpbWlsYXJpdHkgdHJhbnNmb3JtIHRha2luZyBxIG9udG8gcC4iIiIKICAgICAgICBwYywgcWMgPSBwLm1lYW4oYXhpcz0wKSwgcS5tZWFuKGF4aXM9MCkKICAgICAgICBwMCwgcTAgPSBwIC0gcGMsIHEgLSBxYwogICAgICAgIGRlbm9tID0gKHEwICoqIDIpLnN1bSgpCiAgICAgICAgaWYgZGVub20gPD0gMDoKICAgICAgICAgICAgcmV0dXJuIE5vbmUKICAgICAgICAjIFRoZSByb3RhdGlvbi1hbmQtc2NhbGUgcGFydCwgYXMgb25lIGNvbXBsZXggbXVsdGlwbGljYXRpb24uCiAgICAgICAgbnVtID0gY29tcGxleCgocDBbOiwgMF0gKiBxMFs6LCAwXSArIHAwWzosIDFdICogcTBbOiwgMV0pLnN1bSgpLAogICAgICAgICAgICAgICAgICAgICAgKHAwWzosIDFdICogcTBbOiwgMF0gLSBwMFs6LCAwXSAqIHEwWzosIDFdKS5zdW0oKSkKICAgICAgICB6ID0gbnVtIC8gZGVub20KICAgICAgICByZXR1cm4geiwgcGMsIHFjCgogICAgZ290ID0gZml0KGEsIGIpCiAgICBpZiBnb3QgaXMgTm9uZToKICAgICAgICBkaWFnWyJ3aHkiXSA9ICJ0aGUgYW5jaG9ycyBhcmUgb24gdG9wIG9mIGVhY2ggb3RoZXIiCiAgICAgICAgcmV0dXJuIE5vbmUsIGRpYWcKICAgIHosIHBjLCBxYyA9IGdvdAoKICAgIGlmIGxlbihhKSA+PSA0OgogICAgICAgIHByZWQgPSBfYXBwbHkoYiwgeiwgcGMsIHFjKQogICAgICAgIHJlc2lkID0gbnAuaHlwb3QoKihwcmVkIC0gYSkuVCkKICAgICAgICB3b3JzdCA9IGludChucC5hcmdtYXgocmVzaWQpKQogICAgICAgIG90aGVycyA9IG5wLmRlbGV0ZShyZXNpZCwgd29yc3QpCiAgICAgICAgaWYgcmVzaWRbd29yc3RdID4gbWF4KDMuMCAqIG5wLm1lZGlhbihvdGhlcnMpLCAwLjUpOgogICAgICAgICAgICBhLCBiID0gbnAuZGVsZXRlKGEsIHdvcnN0LCAwKSwgbnAuZGVsZXRlKGIsIHdvcnN0LCAwKQogICAgICAgICAgICBkaWFnWyJkcm9wcGVkIl0gPSAxCiAgICAgICAgICAgIHosIHBjLCBxYyA9IGZpdChhLCBiKQoKICAgIHByZWQgPSBfYXBwbHkoYiwgeiwgcGMsIHFjKQogICAgcmVzaWQgPSBucC5oeXBvdCgqKHByZWQgLSBhKS5UKQogICAgZGlhZy51cGRhdGUoewogICAgICAgICJuX3VzZWQiOiBsZW4oYSksCiAgICAgICAgInNoaWZ0X3B4IjogKGZsb2F0KHBjWzBdIC0gcWNbMF0gKiB6LnJlYWwgKyBxY1sxXSAqIHouaW1hZyksCiAgICAgICAgICAgICAgICAgICAgIGZsb2F0KHBjWzFdIC0gcWNbMF0gKiB6LmltYWcgLSBxY1sxXSAqIHoucmVhbCkpLAogICAgICAgICJyb3RhdGlvbl9kZWciOiBmbG9hdChucC5kZWdyZWVzKG5wLmFyY3RhbjIoei5pbWFnLCB6LnJlYWwpKSksCiAgICAgICAgInNjYWxlIjogZmxvYXQoYWJzKHopKSwKICAgICAgICAicmVzaWR1YWxfcHgiOiBmbG9hdChucC5tYXgocmVzaWQpKSwKICAgICAgICAic2VwYXJhdGlvbl9weCI6IGZsb2F0KG5wLmh5cG90KCooYS5tYXgoYXhpcz0wKSAtIGEubWluKGF4aXM9MCkpKSksCiAgICB9KQogICAgcmV0dXJuICh6LCBwYywgcWMpLCBkaWFnCgpkZWYgX2FwcGx5KHBvaW50cywgeiwgcGMsIHFjKToKICAgICIiIk1hcCBwb2ludHMgdGhyb3VnaCBhIGZpdHRlZCBzaW1pbGFyaXR5IHRyYW5zZm9ybS4iIiIKICAgIHEwID0gbnAuYXNhcnJheShwb2ludHMsIGR0eXBlPWZsb2F0KSAtIHFjCiAgICB4ID0gcTBbOiwgMF0gKiB6LnJlYWwgLSBxMFs6LCAxXSAqIHouaW1hZyArIHBjWzBdCiAgICB5ID0gcTBbOiwgMF0gKiB6LmltYWcgKyBxMFs6LCAxXSAqIHoucmVhbCArIHBjWzFdCiAgICByZXR1cm4gbnAuY29sdW1uX3N0YWNrKFt4LCB5XSkKCmRlZiBhcHBseV90cmFuc2Zvcm0oaW1hZ2UsIHRyYW5zZm9ybSwgc2hhcGU9Tm9uZSk6CiAgICAiIiJSZXNhbXBsZSBhIGZyYW1lIHRocm91Z2ggYSB0cmFuc2Zvcm0gZml0dGVkIGZyb20gdGhlIGFuY2hvcnMuCgogICAgT25lIGFmZmluZSBpbnRlcnBvbGF0aW9uLCBhbmQgbm8gYXN0cm9tZXRyaWMgYXJpdGhtZXRpYyBhdCBhbGwuIFRoYXQgaXMgd2hlcmUKICAgIHRoZSB0aW1lIGdvZXM6IHRoZSB2ZXJzaW9uIHRoYXQgd2VudCB0aHJvdWdoIGJvdGggcGxhdGUgc29sdXRpb25zIGhhZCB0byB0dXJuCiAgICBzaXh0eS1vbmUgbWlsbGlvbiBwaXhlbCBwb3NpdGlvbnMgaW50byBza3kgcG9zaXRpb25zIGFuZCBiYWNrIGFnYWluIGJlZm9yZSBpdAogICAgY291bGQgaW50ZXJwb2xhdGUgYW55dGhpbmcuCiAgICAiIiIKICAgIGZyb20gc2NpcHkubmRpbWFnZSBpbXBvcnQgYWZmaW5lX3RyYW5zZm9ybQoKICAgIHosIHBjLCBxYyA9IHRyYW5zZm9ybQogICAgc2hhcGUgPSBzaGFwZSBvciBpbWFnZS5zaGFwZQoKICAgICMgYWZmaW5lX3RyYW5zZm9ybSBwdWxsczogZm9yIGVhY2ggb3V0cHV0IHBpeGVsIGl0IGFza3Mgd2hlcmUgaW4gdGhlIGlucHV0IHRvCiAgICAjIGxvb2ssIHNvIHRoZSBpbnZlcnNlIG9mIHRoZSBmaXR0ZWQgbWFwIGlzIHdoYXQgaXQgbmVlZHMsIHdyaXR0ZW4gaW4gdGhlCiAgICAjIChyb3csIGNvbHVtbikgb3JkZXIgaXQgZXhwZWN0cy4KICAgIGludiA9IDEuMCAvIHoKICAgIG0gPSBucC5hcnJheShbW2ludi5yZWFsLCBpbnYuaW1hZ10sIFstaW52LmltYWcsIGludi5yZWFsXV0sIGR0eXBlPWZsb2F0KQogICAgb2Zmc2V0ID0gbnAuYXJyYXkoW3FjWzFdLCBxY1swXV0pIC0gbS5kb3QobnAuYXJyYXkoW3BjWzFdLCBwY1swXV0pKQoKICAgIGtub3duID0gbnAuaXNmaW5pdGUoaW1hZ2UpCiAgICBmaWxsZWQgPSBucC53aGVyZShrbm93biwgaW1hZ2UsIDAuMCkuYXN0eXBlKG5wLmZsb2F0MzIpCiAgICBvdXQgPSBhZmZpbmVfdHJhbnNmb3JtKGZpbGxlZCwgbSwgb2Zmc2V0PW9mZnNldCwgb3V0cHV0X3NoYXBlPXNoYXBlLAogICAgICAgICAgICAgICAgICAgICAgICAgICBvcmRlcj0xLCBtb2RlPSJjb25zdGFudCIsIGN2YWw9bnAubmFuKQogICAgc2VlbiA9IGFmZmluZV90cmFuc2Zvcm0oa25vd24uYXN0eXBlKG5wLmZsb2F0MzIpLCBtLCBvZmZzZXQ9b2Zmc2V0LAogICAgICAgICAgICAgICAgICAgICAgICAgICAgb3V0cHV0X3NoYXBlPXNoYXBlLCBvcmRlcj0xLCBtb2RlPSJjb25zdGFudCIsIGN2YWw9MC4wKQogICAgb3V0W3NlZW4gPCAwLjVdID0gbnAubmFuCiAgICByZXR1cm4gb3V0CgpkZWYgZGVzY3JpYmUoZGlhZyk6CiAgICAiIiJUaGUgcmVnaXN0cmF0aW9uLCBpbiBhIGxpbmUgd29ydGggcHJpbnRpbmcuIiIiCiAgICBpZiBkaWFnLmdldCgid2h5Iik6CiAgICAgICAgcmV0dXJuICIgIGFuY2hvcnMgdW51c2FibGUgKHt9KTsgZmFsbGluZyBiYWNrIi5mb3JtYXQoZGlhZ1sid2h5Il0pCiAgICByZXR1cm4gKCIgIHt9IGFuY2hvciBzdGFyc3t9LCB7Oi4wZn0gcHggYXBhcnQ6IHNoaWZ0ICh7OisuMmZ9LCB7OisuMmZ9KSBweCwgIgogICAgICAgICAgICAicm90YXRpb24gezorLjRmfSBkZWcsIHNjYWxlIHs6LjZmfTsgd29yc3QgYW5jaG9yIG9mZiBieSB7Oi4yZn0gcHgiCiAgICAgICAgICAgIC5mb3JtYXQoZGlhZ1sibl91c2VkIl0sCiAgICAgICAgICAgICAgICAgICAgIiAob25lIGRyb3BwZWQpIiBpZiBkaWFnLmdldCgiZHJvcHBlZCIpIGVsc2UgIiIsCiAgICAgICAgICAgICAgICAgICAgZGlhZ1sic2VwYXJhdGlvbl9weCJdLCBkaWFnWyJzaGlmdF9weCJdWzBdLCBkaWFnWyJzaGlmdF9weCJdWzFdLAogICAgICAgICAgICAgICAgICAgIGRpYWdbInJvdGF0aW9uX2RlZyJdLCBkaWFnWyJzY2FsZSJdLCBkaWFnWyJyZXNpZHVhbF9weCJdKSkKCmRlc2NyaWJlX19hbmNob3JfcmVnaXN0cmF0aW9uID0gZGVzY3JpYmUgICAjIHRoaXMgbW9kdWxlJ3Mgb3duOyB0aGUgcGxhaW4gbmFtZSBnZXRzIHNoYWRvd2VkIGJlbG93CgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIHNoYXJlZCBtb2R1bGUgZ2FsYXh5X3dpbmRvdy5weQojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCgpkZWYgZ2FsYXh5X3dpbmRvdyhzaGFwZSwgY2VudHJlLCByMjVfcHgsIG1hcmdpbj0xLjUpOgogICAgIiIiVGhlIHNsaWNlIG9mIGEgZnJhbWUgd29ydGggd29ya2luZyBvbiwgYXMgKHkwLCB5MSwgeDAsIHgxKS4KCiAgICBSZXR1cm5zIE5vbmUgd2hlbiB0aGVyZSBpcyBubyByYWRpdXMgdG8gd29yayBmcm9tLCBvciB3aGVuIHRoZSB3aW5kb3cgd291bGQKICAgIGNvdmVyIGVzc2VudGlhbGx5IHRoZSB3aG9sZSBmcmFtZSBhbnl3YXkgLSBpbiB3aGljaCBjYXNlIGNyb3BwaW5nIHdvdWxkIGFkZCBhCiAgICBib29ra2VlcGluZyBzdGVwIGFuZCBzYXZlIG5vdGhpbmcuCiAgICAiIiIKICAgIGlmIG5vdCByMjVfcHggb3IgcjI1X3B4IDw9IDA6CiAgICAgICAgcmV0dXJuIE5vbmUKCiAgICBueSwgbnggPSBzaGFwZQogICAgcmVhY2ggPSBtYXJnaW4gKiByMjVfcHgKICAgIHgwID0gaW50KG1heCgwLCBucC5mbG9vcihjZW50cmVbMF0gLSByZWFjaCkpKQogICAgeDEgPSBpbnQobWluKG54LCBucC5jZWlsKGNlbnRyZVswXSArIHJlYWNoKSkpCiAgICB5MCA9IGludChtYXgoMCwgbnAuZmxvb3IoY2VudHJlWzFdIC0gcmVhY2gpKSkKICAgIHkxID0gaW50KG1pbihueSwgbnAuY2VpbChjZW50cmVbMV0gKyByZWFjaCkpKQoKICAgIGlmIHgxIC0geDAgPCA2NCBvciB5MSAtIHkwIDwgNjQ6CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIGlmICh4MSAtIHgwKSAqICh5MSAtIHkwKSA+IDAuOSAqIG54ICogbnk6CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIHJldHVybiAoeTAsIHkxLCB4MCwgeDEpCgpkZWYgY3V0KGRhdGEsIHdpbmRvdyk6CiAgICAiIiJUaGUgd2luZG93J3Mgd29ydGggb2YgYSBmcmFtZS4iIiIKICAgIGlmIHdpbmRvdyBpcyBOb25lIG9yIGRhdGEgaXMgTm9uZToKICAgICAgICByZXR1cm4gZGF0YQogICAgeTAsIHkxLCB4MCwgeDEgPSB3aW5kb3cKICAgIHJldHVybiBkYXRhW3kwOnkxLCB4MDp4MV0KCmRlZiBzaGlmdF9jZW50cmUoY2VudHJlLCB3aW5kb3cpOgogICAgIiIiV2hlcmUgdGhlIGdhbGF4eSdzIGNlbnRyZSBzaXRzIG9uY2UgdGhlIGZyYW1lIGhhcyBiZWVuIGN1dC4iIiIKICAgIGlmIHdpbmRvdyBpcyBOb25lOgogICAgICAgIHJldHVybiBjZW50cmUKICAgIHkwLCBfLCB4MCwgXyA9IHdpbmRvdwogICAgcmV0dXJuIChjZW50cmVbMF0gLSB4MCwgY2VudHJlWzFdIC0geTApCgpkZWYgc2hpZnRfcG9zaXRpb25zKHBvc2l0aW9ucywgd2luZG93LCBzaGFwZT1Ob25lKToKICAgICIiIlN0YXIgcG9zaXRpb25zIG1vdmVkIGludG8gdGhlIGN1dCBmcmFtZSwga2VlcGluZyBvbmx5IHRob3NlIGluc2lkZSBpdC4iIiIKICAgIGlmIHdpbmRvdyBpcyBOb25lIG9yIHBvc2l0aW9ucyBpcyBOb25lIG9yIG5vdCBsZW4ocG9zaXRpb25zKToKICAgICAgICByZXR1cm4gcG9zaXRpb25zCiAgICB5MCwgeTEsIHgwLCB4MSA9IHdpbmRvdwogICAgcCA9IG5wLmFzYXJyYXkocG9zaXRpb25zLCBkdHlwZT1mbG9hdCkKICAgIGluc2lkZSA9ICgocFs6LCAwXSA+PSB4MCkgJiAocFs6LCAwXSA8IHgxKSAmIChwWzosIDFdID49IHkwKSAmIChwWzosIDFdIDwgeTEpKQogICAgcmV0dXJuIG5wLmNvbHVtbl9zdGFjayhbcFtpbnNpZGUsIDBdIC0geDAsIHBbaW5zaWRlLCAxXSAtIHkwXSkKCmRlZiByZXN0b3JlX3Bvc2l0aW9ucyh4LCB5LCB3aW5kb3cpOgogICAgIiIiUG9zaXRpb25zIG1lYXN1cmVkIGluIHRoZSBjdXQgZnJhbWUsIHB1dCBiYWNrIHdoZXJlIHRoZXkgYmVsb25nLgoKICAgIEV2ZXJ5dGhpbmcgdGhlIHRvb2wgcmVwb3J0cyAtIGNhdGFsb2d1ZXMsIHBsb3RzLCB0aGUgcGFpcmluZyB3aXRoIHRoZSBibHVlCiAgICBrbm90cyAtIGlzIGluIHRoZSBjb29yZGluYXRlcyBvZiB0aGUgb3JpZ2luYWwgZnJhbWUsIHNvIG1lYXN1cmVtZW50cyBtYWRlIGluCiAgICB0aGUgd2luZG93IGhhdmUgdG8gY29tZSBiYWNrIG91dCBvZiBpdC4gRm9yZ2V0dGluZyB0aGlzIHdvdWxkIHNoaWZ0IGV2ZXJ5CiAgICBwb3NpdGlvbiBieSB0aGUgc2l6ZSBvZiB0aGUgY3JvcCwgd2hpY2ggaXMgdGhlIGtpbmQgb2YgZXJyb3IgdGhhdCBsb29rcyBsaWtlCiAgICBhIGRpc2NvdmVyeS4KICAgICIiIgogICAgaWYgd2luZG93IGlzIE5vbmU6CiAgICAgICAgcmV0dXJuIHgsIHkKICAgIHkwLCBfLCB4MCwgXyA9IHdpbmRvdwogICAgcmV0dXJuIG5wLmFzYXJyYXkoeCwgZHR5cGU9ZmxvYXQpICsgeDAsIG5wLmFzYXJyYXkoeSwgZHR5cGU9ZmxvYXQpICsgeTAKCmRlZiBkZXNjcmliZSh3aW5kb3csIHNoYXBlLCBtYXJnaW4pOgogICAgIiIiV2hhdCB3YXMgY3V0LCBhbmQgd2hhdCBpdCBzYXZlZC4iIiIKICAgIGlmIHdpbmRvdyBpcyBOb25lOgogICAgICAgIHJldHVybiAoIiAgd29ya2luZyBvbiB0aGUgd2hvbGUgZnJhbWUgLSBlaXRoZXIgbm8gaXNvcGhvdGFsIHJhZGl1cyBpcyAiCiAgICAgICAgICAgICAgICAia25vd24sIG9yIHRoZSBnYWxheHkgZmlsbHMgaXQiKQogICAgeTAsIHkxLCB4MCwgeDEgPSB3aW5kb3cKICAgIGtlcHQgPSAoeTEgLSB5MCkgKiAoeDEgLSB4MCkKICAgIHdob2xlID0gc2hhcGVbMF0gKiBzaGFwZVsxXQogICAgcmV0dXJuICgiICB3b3JraW5nIG9uIHt9IHgge30gcGl4ZWxzIGFyb3VuZCB0aGUgZ2FsYXh5LCB7Oi4wZn0lIG9mIHRoZSBmcmFtZSAiCiAgICAgICAgICAgICIoezouMWZ9IHggUjI1KTsgdGhlIHJlc3QgaXMgc2t5IHRoYXQgd291bGQgYmUgZGlzY2FyZGVkIGFueXdheSIKICAgICAgICAgICAgLmZvcm1hdCh4MSAtIHgwLCB5MSAtIHkwLCAxMDAuMCAqIGtlcHQgLyB3aG9sZSwgbWFyZ2luKSkKCmRlc2NyaWJlX19nYWxheHlfd2luZG93ID0gZGVzY3JpYmUgICAjIHRoaXMgbW9kdWxlJ3Mgb3duOyB0aGUgcGxhaW4gbmFtZSBnZXRzIHNoYWRvd2VkIGJlbG93CgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIHNoYXJlZCBtb2R1bGUgZ2FsYXh5X2V4dGVudC5weQojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCgpkZWYgc3VyZmFjZV9icmlnaHRuZXNzX3Byb2ZpbGUoZGF0YSwgY2VudHJlLCByX21heD1Ob25lLCBuX3JpbmdzPTQwLCBzdGVwPTQsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBwZXJjZW50aWxlPTkwLjApOgogICAgIiIiSG93IG11Y2ggbGlnaHQgZWFjaCByaW5nIGhvbGRzLCBhYm92ZSB3aGF0IGFuIGVtcHR5IHJpbmcgaG9sZHMuCgogICAgV0hZIEEgSElHSCBQRVJDRU5USUxFIEFORCBOT1QgVEhFIE1FRElBTgogICAgQmVjYXVzZSBvZiB3aGF0IGlzIGJlaW5nIG1lYXN1cmVkLiBBIHN0ZWxsYXIgZGlzayBpcyBzbW9vdGgsIGFuZCBpdHMgbWVkaWFuCiAgICBkZXNjcmliZXMgaXQuIExpbmUgZW1pc3Npb24gaXMgbm90IHNtb290aDogaXQgbGl2ZXMgaW4gZGlzY3JldGUgcmVnaW9ucyB3aXRoCiAgICBlbXB0eSBza3kgYmV0d2VlbiB0aGVtLCBzbyBhIHJpbmcgaG9sZGluZyBhIGRvemVuIGJyaWdodCBISUkgcmVnaW9ucyBhbmQKICAgIHR3ZW50eSB0aG91c2FuZCBlbXB0eSBwaXhlbHMgaGFzIGEgbWVkaWFuIG9mIHplcm8uIE1lYXN1cmVkIG9uIE04MywgdGhlIG1lZGlhbgogICAgb2YgdGhlIHN1YnRyYWN0ZWQgZnJhbWUgcmVhY2hlZCB0aGUgbm9pc2UgYXQgMjU2IHB4IHdoaWxlIHRoZSByZWdpb25zCiAgICB0aGVtc2VsdmVzIHdlbnQgb24gdG8gNTU2IC0gYW5kIHVzaW5nIHRoZSBtZWRpYW4gY3V0IHRoZSBjYXRhbG9ndWUgZnJvbSAzMzUKICAgIHJlZ2lvbnMgdG8gMTIyLgoKICAgIFRoZSBuaW5ldGlldGggcGVyY2VudGlsZSBhc2tzIHRoZSB1c2VmdWwgcXVlc3Rpb24gaW5zdGVhZDogaXMgdGhlcmUgYW55dGhpbmcKICAgIGJyaWdodCBpbiB0aGlzIHJpbmcuIEl0IHdvcmtzIGZvciBhIHNtb290aCB0cmFjZXIgdG9vLCB3aGVyZSBpdCBzaW1wbHkgdHJhY2tzCiAgICB0aGUgbWVkaWFuIHdpdGggYW4gb2Zmc2V0LgoKICAgIFdIWSBUSEUgRkFSIEZJRUxEIElTIFNVQlRSQUNURUQKICAgIEEgaGlnaCBwZXJjZW50aWxlIG9mIHB1cmUgbm9pc2UgaXMgbm90IHplcm8gLSBpdCBpcyBhYm91dCAxLjMgdGltZXMgdGhlIG5vaXNlLAogICAgYmVjYXVzZSBpdCBpcyBwaWNraW5nIHRoZSBicmlnaHRlc3QgdGVudGggb2YgdGhlIHNhbXBsZS4gQ29tcGFyaW5nIGFnYWluc3QKICAgIHplcm8gd291bGQgY2FsbCBlbXB0eSBza3kgInN0aWxsIGdhbGF4eSIgZm9yIGV2ZXIuIFNvIHRoZSBzYW1lIHN0YXRpc3RpYyBpcwogICAgbWVhc3VyZWQgZmFyIG91dCwgd2hlcmUgdGhlcmUgaXMgbm90aGluZywgYW5kIHRoYXQgZmxvb3IgaXMgdGFrZW4gb2ZmLgoKICAgIFNhbXBsZWQgZXZlcnkgZmV3IHBpeGVscyByYXRoZXIgdGhhbiBldmVyeSBvbmU6IHRoaXMgaXMgYSBzaGFwZSBtZWFzdXJlbWVudAogICAgb3ZlciB0aG91c2FuZHMgb2YgcGl4ZWxzIGEgcmluZywgYW5kIHJlYWRpbmcgYSBxdWFydGVyIG9mIHRoZW0gY2hhbmdlcyBub3RoaW5nCiAgICB3aGlsZSBtYWtpbmcgaXQgcXVpY2sgZW5vdWdoIGZvciBhIDYxLW1lZ2FwaXhlbCBmcmFtZS4KICAgICIiIgogICAgZnJvbSBhc3Ryb3B5LnN0YXRzIGltcG9ydCBzaWdtYV9jbGlwcGVkX3N0YXRzCgogICAgbnksIG54ID0gZGF0YS5zaGFwZQogICAgeXksIHh4ID0gbnAubWdyaWRbMDpueTpzdGVwLCAwOm54OnN0ZXBdCiAgICByID0gbnAuaHlwb3QoeHggLSBjZW50cmVbMF0sIHl5IC0gY2VudHJlWzFdKS5yYXZlbCgpCiAgICB2ID0gbnAuYXNhcnJheShkYXRhWzo6c3RlcCwgOjpzdGVwXSwgZHR5cGU9ZmxvYXQpLnJhdmVsKCkKICAgIG9rID0gbnAuaXNmaW5pdGUodikKICAgIHIsIHYgPSByW29rXSwgdltva10KICAgIGlmIG5vdCBsZW4ocik6CiAgICAgICAgcmV0dXJuIE5vbmUKCiAgICBpZiByX21heCBpcyBOb25lOgogICAgICAgIHJfbWF4ID0gZmxvYXQobnAucGVyY2VudGlsZShyLCA5OSkpCiAgICBmYXIgPSByID4gMC44ICogci5tYXgoKQogICAgaWYgZmFyLnN1bSgpIDwgNTAwOgogICAgICAgIGZhciA9IHIgPiBucC5wZXJjZW50aWxlKHIsIDgwKQogICAgc2t5ID0gZmxvYXQobnAubWVkaWFuKHZbZmFyXSkpCiAgICBub2lzZSA9IGZsb2F0KHNpZ21hX2NsaXBwZWRfc3RhdHModltmYXJdLCBzaWdtYT0zLjApWzJdKQogICAgaWYgbm90IG5wLmlzZmluaXRlKG5vaXNlKSBvciBub2lzZSA8PSAwOgogICAgICAgIHJldHVybiBOb25lCgogICAgIyBXaGF0IHRoaXMgc3RhdGlzdGljIHJlYWRzIG9uIGVtcHR5IHNreS4gRXZlcnl0aGluZyBpcyBtZWFzdXJlZCBhZ2FpbnN0IGl0LgogICAgZmxvb3IgPSBmbG9hdChucC5wZXJjZW50aWxlKHZbZmFyXSwgcGVyY2VudGlsZSkpCgogICAgZWRnZXMgPSBucC5saW5zcGFjZSgwLjAsIHJfbWF4LCBuX3JpbmdzICsgMSkKICAgIGNlbnRyZXMsIGV4Y2VzcyA9IFtdLCBbXQogICAgZm9yIGEsIGIgaW4gemlwKGVkZ2VzWzotMV0sIGVkZ2VzWzE6XSk6CiAgICAgICAgbSA9IChyID49IGEpICYgKHIgPCBiKQogICAgICAgIGlmIG0uc3VtKCkgPCA1MDoKICAgICAgICAgICAgY29udGludWUKICAgICAgICBjZW50cmVzLmFwcGVuZCgwLjUgKiAoYSArIGIpKQogICAgICAgIGV4Y2Vzcy5hcHBlbmQoKGZsb2F0KG5wLnBlcmNlbnRpbGUodlttXSwgcGVyY2VudGlsZSkpIC0gZmxvb3IpIC8gbm9pc2UpCiAgICByZXR1cm4gbnAuYXJyYXkoY2VudHJlcyksIG5wLmFycmF5KGV4Y2VzcyksIHNreSwgbm9pc2UKCmRlZiB2aXNpYmxlX3JhZGl1cyhkYXRhLCBjZW50cmUsIHRocmVzaG9sZD0wLjUsIHJfbWF4PU5vbmUsIHBlcmNlbnRpbGU9OTAuMCk6CiAgICAiIiJXaGVyZSB0aGUgZ2FsYXh5J3MgbGlnaHQgZmFsbHMgYmVsb3cgYSBmcmFjdGlvbiBvZiB0aGUgc2t5IG5vaXNlLgoKICAgIGB0aHJlc2hvbGRgIGlzIGluIHVuaXRzIG9mIHRoYXQgbm9pc2UsIGFuZCBpdCBpcyB0aGUgb25lIHJlYWwgY2hvaWNlIGhlcmUuCiAgICBIYWxmIGEgc2lnbWEgaXMgdGhlIGRlZmF1bHQgYmVjYXVzZSB0aGUgdHdvIGVycm9ycyBhcmUgbm90IHN5bW1ldHJpYzogdG9vCiAgICBsYXJnZSBhIHJhZGl1cyBsZXRzIGVtcHR5IHNreSBpbnRvIHRoZSBtZWFzdXJlbWVudCBhbmQgZGlsdXRlcyBpdCwgd2hpY2ggaXMKICAgIHdoYXQgd2FzIGhhcHBlbmluZywgd2hpbGUgdG9vIHNtYWxsIGEgcmFkaXVzIGRlbGV0ZXMgdGhlIG91dGVyIHJlZ2lvbnMgLSBhbmQKICAgIHRoZSBvdXRza2lydHMgYXJlIHdoZXJlIGEgcmFkaWFsIHByb2ZpbGUgaXMgbW9zdCBzZW5zaXRpdmUsIHNpbmNlIHRoZSBkZW5zaXR5CiAgICBpcyBsb3dlc3QgdGhlcmUuIEhhbGYgYSBzaWdtYSBzaXRzIHdoZXJlIHRoZSBwcm9maWxlIGhhcyBjbGVhcmx5IGZsYXR0ZW5lZCBidXQKICAgIGhhcyBub3QgeWV0IHJlYWNoZWQgemVyby4KCiAgICBSZXR1cm5zIChyYWRpdXMgaW4gcGl4ZWxzLCBkaWFnbm9zdGljcyksIG9yIChOb25lLCBkaWFnbm9zdGljcykgd2hlbiB0aGUKICAgIHByb2ZpbGUgbmV2ZXIgcmlzZXMgYWJvdmUgdGhlIHRocmVzaG9sZCBhdCBhbGwgLSBhIGZyYW1lIHdpdGggbm8gZ2FsYXh5IGluIGl0LAogICAgb3IgYSBjZW50cmUgaW4gdGhlIHdyb25nIHBsYWNlLgogICAgIiIiCiAgICBnb3QgPSBzdXJmYWNlX2JyaWdodG5lc3NfcHJvZmlsZShkYXRhLCBjZW50cmUsIHJfbWF4PXJfbWF4LAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcGVyY2VudGlsZT1wZXJjZW50aWxlKQogICAgaWYgZ290IGlzIE5vbmU6CiAgICAgICAgcmV0dXJuIE5vbmUsIHsid2h5IjogImNvdWxkIG5vdCBtZWFzdXJlIGEgYmFja2dyb3VuZCB0byBjb21wYXJlIGFnYWluc3QifQogICAgcmFkaWksIGV4Y2Vzcywgc2t5LCBub2lzZSA9IGdvdAoKICAgIGFib3ZlID0gZXhjZXNzID4gdGhyZXNob2xkCiAgICBpZiBub3QgYWJvdmUuYW55KCk6CiAgICAgICAgcmV0dXJuIE5vbmUsIHsid2h5IjogIm5vdGhpbmcgcmlzZXMgYWJvdmUgezouMWZ9IHNpZ21hIG9mIHRoZSBza3kgbm9pc2UgIgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICJhbnl3aGVyZSIuZm9ybWF0KHRocmVzaG9sZCksCiAgICAgICAgICAgICAgICAgICAgICAic2t5Ijogc2t5LCAibm9pc2UiOiBub2lzZX0KCiAgICAjIFRoZSBsYXN0IHJpbmcgYWJvdmUgdGhlIHRocmVzaG9sZCwgd2Fsa2luZyBvdXR3YXJkcyAtIG5vdCB0aGUgZmlyc3Qgb25lCiAgICAjIGJlbG93IGl0LCBzbyB0aGF0IGEgc2luZ2xlIG5vaXN5IHJpbmcgaW4gdGhlIG1pZGRsZSBkb2VzIG5vdCBjdXQgdGhlIGdhbGF4eQogICAgIyBpbiBoYWxmLgogICAgbGFzdCA9IGludChucC53aGVyZShhYm92ZSlbMF0ubWF4KCkpCiAgICByYWRpdXMgPSBmbG9hdChyYWRpaVtsYXN0XSkKICAgIHJldHVybiByYWRpdXMsIHsic2t5Ijogc2t5LCAibm9pc2UiOiBub2lzZSwgInRocmVzaG9sZCI6IHRocmVzaG9sZCwKICAgICAgICAgICAgICAgICAgICAiY2VudHJhbF9leGNlc3MiOiBmbG9hdChleGNlc3NbMF0pIGlmIGxlbihleGNlc3MpIGVsc2UgZmxvYXQoIm5hbiIpLAogICAgICAgICAgICAgICAgICAgICJyYWRpaSI6IHJhZGlpLCAiZXhjZXNzIjogZXhjZXNzfQoKZGVmIGRlc2NyaWJlKHJhZGl1cywgZGlhZywgcjI1X3B4PU5vbmUsIGxhYmVsPSIiKToKICAgICIiIldoYXQgd2FzIG1lYXN1cmVkLCBuZXh0IHRvIHdoYXQgdGhlIGNhdGFsb2d1ZSBzYXlzLiIiIgogICAgaWYgcmFkaXVzIGlzIE5vbmU6CiAgICAgICAgcmV0dXJuICIgIHt9OiBjb3VsZCBub3QgbWVhc3VyZSBhbiBleHRlbnQgKHt9KSIuZm9ybWF0KAogICAgICAgICAgICBsYWJlbCwgZGlhZy5nZXQoIndoeSIsICJ1bmtub3duIikpCiAgICBsaW5lID0gKCIgIHt9OiB0aGUgbGlnaHQgZmFsbHMgYmVsb3cgezouMWZ9IHggdGhlIHNreSBub2lzZSBhdCB7Oi4wZn0gcHgiCiAgICAgICAgICAgIC5mb3JtYXQobGFiZWwsIGRpYWcuZ2V0KCJ0aHJlc2hvbGQiLCAwLjUpLCByYWRpdXMpKQogICAgaWYgcjI1X3B4OgogICAgICAgIGxpbmUgKz0gIjsgdGhlIGNhdGFsb2d1ZSdzIFIyNSBpcyB7Oi4wZn0gcHgiLmZvcm1hdChyMjVfcHgpCiAgICAgICAgaWYgcmFkaXVzIDwgMC44ICogcjI1X3B4OgogICAgICAgICAgICBsaW5lICs9ICgiXG4gICAgdGhlIGV4cG9zdXJlIGRvZXMgbm90IHJlYWNoIHRoZSBjYXRhbG9ndWUgcmFkaXVzLCBzbyAiCiAgICAgICAgICAgICAgICAgICAgICJ0aGUgbWVhc3VyZWQgb25lIGlzIHVzZWQiKQogICAgICAgIGVsaWYgcmFkaXVzID4gMS4yICogcjI1X3B4OgogICAgICAgICAgICBsaW5lICs9ICgiXG4gICAgdGhlIGxpZ2h0IHJlYWNoZXMgcGFzdCB0aGUgY2F0YWxvZ3VlIHJhZGl1czsgdGhlICIKICAgICAgICAgICAgICAgICAgICAgImNhdGFsb2d1ZSBvbmUgaXMgdXNlZCwgYmVpbmcgdGhlIG1vcmUgc3RhbmRhcmQiKQogICAgcmV0dXJuIGxpbmUKCmRlZiB3b3JraW5nX3JhZGl1cyhtZWFzdXJlZCwgcjI1X3B4KToKICAgICIiIlRoZSByYWRpdXMgdG8gYWN0dWFsbHkgdXNlOiB0aGUgc21hbGxlciBvZiB3aGF0IGlzIHRoZXJlIGFuZCB3aGF0IGlzIHB1Ymxpc2hlZC4KCiAgICBTbWFsbGVyIG9mIHRoZSB0d28gb24gcHVycG9zZS4gSWYgdGhlIGV4cG9zdXJlIGRvZXMgbm90IHJlYWNoIFIyNSB0aGVuIHRoZQogICAgc3BhY2UgYmV0d2VlbiBpcyBlbXB0eSBza3kgYW5kIGluY2x1ZGluZyBpdCBvbmx5IGFkZHMgbm9pc2UuIElmIHRoZSBsaWdodAogICAgcmVhY2hlcyBwYXN0IFIyNSAtIHdoaWNoIGhhcHBlbnMgZm9yIGEgYnJpZ2h0IGlubmVyIGRpc2sgbWVhc3VyZWQgZ2VuZXJvdXNseSAtCiAgICB0aGVuIFIyNSBpcyB0aGUgc3RhbmRhcmQgZGVmaW5pdGlvbiBhbmQgdGhlcmUgaXMgbm8gcmVhc29uIHRvIGludmVudCBhIHdpZGVyCiAgICBvbmUuCiAgICAiIiIKICAgIGlmIG1lYXN1cmVkIGFuZCByMjVfcHg6CiAgICAgICAgcmV0dXJuIG1pbihtZWFzdXJlZCwgcjI1X3B4KQogICAgcmV0dXJuIG1lYXN1cmVkIG9yIHIyNV9weAoKZGVzY3JpYmVfX2dhbGF4eV9leHRlbnQgPSBkZXNjcmliZSAgICMgdGhpcyBtb2R1bGUncyBvd247IHRoZSBwbGFpbiBuYW1lIGdldHMgc2hhZG93ZWQgYmVsb3cKCgojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCiMgc2hhcmVkIG1vZHVsZSBkZXByb2plY3Rpb24ucHkKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQoKSU5DX01JTl9ERUcgPSAzMC4wCgpJTkNfTUFYX0RFRyA9IDcwLjAKClEwX0RFRkFVTFQgPSAwLjEzCgpRMF9CWV9UWVBFID0geyJlYXJseSI6IDAuMjAsICJtaWQiOiAwLjEzLCAibGF0ZSI6IDAuMTB9CgpkZWYgcTBfZm9yX3R5cGUobXR5cGU6IHN0ciB8IE5vbmUpIC0+IGZsb2F0OgogICAgIiIiUGljayBhIGRpc2stdGhpY2tuZXNzIGNvbnN0YW50IGZyb20gdGhlIG1vcnBob2xvZ2ljYWwgdHlwZSBzdHJpbmcuCgogICAgRmFsbHMgYmFjayB0byB0aGUgbWlkLXR5cGUgdmFsdWUgd2hlbiB0aGUgdHlwZSBpcyB1bmtub3duIG9yIHVucGFyc2FibGUsCiAgICB3aGljaCBpcyB0aGUgc2FmZSBjaG9pY2U6IGl0IHNpdHMgYmV0d2VlbiB0aGUgdHdvIGV4dHJlbWVzLgogICAgIiIiCiAgICBpZiBub3QgbXR5cGU6CiAgICAgICAgcmV0dXJuIFEwX0RFRkFVTFQKICAgIHQgPSBzdHIobXR5cGUpLnN0cmlwKCkudXBwZXIoKQogICAgaWYgdC5zdGFydHN3aXRoKCgiUzAiLCAiU0EiLCAiRSIpKToKICAgICAgICByZXR1cm4gUTBfQllfVFlQRVsiZWFybHkiXQogICAgaWYgdC5zdGFydHN3aXRoKCgiU0QiLCAiU00iLCAiSU0iLCAiSSIpKToKICAgICAgICByZXR1cm4gUTBfQllfVFlQRVsibGF0ZSJdCiAgICByZXR1cm4gUTBfQllfVFlQRVsibWlkIl0KCmRlZiBpbmNsaW5hdGlvbl9mcm9tX2F4aXNfcmF0aW8oYl9vdmVyX2E6IGZsb2F0LCBxMDogZmxvYXQgPSBRMF9ERUZBVUxUKSAtPiBmbG9hdDoKICAgICIiIkluY2xpbmF0aW9uIGluIGRlZ3JlZXMgZnJvbSB0aGUgb2JzZXJ2ZWQgbWlub3IvbWFqb3IgYXhpcyByYXRpby4KCiAgICBVc2VzIHRoZSBIdWJibGUgZm9ybXVsYSwgd2hpY2ggYWNjb3VudHMgZm9yIHRoZSBkaXNrIGhhdmluZyByZWFsIHRoaWNrbmVzczoKCiAgICAgICAgY29zXjIoaSkgPSAocV4yIC0gcTBeMikgLyAoMSAtIHEwXjIpCgogICAgQXQgbW9kZXJhdGUgaW5jbGluYXRpb25zIHRoaXMgYmFyZWx5IGRpZmZlcnMgZnJvbSB0aGUgbmFpdmUgYWNvcyhiL2EpLiBBdCBoaWdoCiAgICBpbmNsaW5hdGlvbiBpdCBtYXR0ZXJzIGEgbG90OiBmb3IgcSA9IDAuMiB0aGUgbmFpdmUgdmFsdWUgaXMgNzguNSBkZWcgd2hpbGUgdGhpcwogICAgZ2l2ZXMgODEuMiBkZWcsIHdoaWNoIGNoYW5nZXMgdGhlIHN0cmV0Y2ggZmFjdG9yIGZyb20gNS4wIHRvIDYuNS4KCiAgICBSZXR1cm5zIDkwLjAgd2hlbiB0aGUgZ2FsYXh5IGlzIHRoaW5uZXIgdGhhbiBxMCwgaS5lLiBzZWVuIGVkZ2Utb24uCiAgICAiIiIKICAgIHEgPSBmbG9hdChiX292ZXJfYSkKICAgIGlmIHEgPD0gcTA6CiAgICAgICAgcmV0dXJuIDkwLjAKICAgIGNvczIgPSAocSAqIHEgLSBxMCAqIHEwKSAvICgxLjAgLSBxMCAqIHEwKQogICAgY29zMiA9IG1pbihtYXgoY29zMiwgMC4wKSwgMS4wKQogICAgcmV0dXJuIG1hdGguZGVncmVlcyhtYXRoLmFjb3MobWF0aC5zcXJ0KGNvczIpKSkKCmRlZiBzdHJldGNoX2ZhY3RvcihpbmNfZGVnOiBmbG9hdCkgLT4gZmxvYXQ6CiAgICAiIiJIb3cgbXVjaCB0aGUgbWlub3ItYXhpcyBkaXJlY3Rpb24gaGFzIHRvIGJlIHN0cmV0Y2hlZDogMSAvIGNvcyhpKS4iIiIKICAgIHJldHVybiAxLjAgLyBtYXRoLmNvcyhtYXRoLnJhZGlhbnMoaW5jX2RlZykpCgpkZWYgaW5jbGluYXRpb25faXNfdXNhYmxlKGluY19kZWc6IGZsb2F0KSAtPiB0dXBsZVtib29sLCBzdHJdOgogICAgIiIiRGVjaWRlIHdoZXRoZXIgZGVwcm9qZWN0aW5nIGF0IHRoaXMgaW5jbGluYXRpb24gaXMgd29ydGggZG9pbmcuCgogICAgUmV0dXJucyAob2ssIHJlYXNvbikuIFRoZSByZWFzb24gaXMgbWVhbnQgdG8gYmUgcHJpbnRlZCBhbmQgdG8gZW5kIHVwIGluIHRoZQogICAgcGxvdCwgc28gdGhhdCBhIHNraXBwZWQgY29ycmVjdGlvbiBpcyB2aXNpYmxlIHJhdGhlciB0aGFuIHNpbGVudC4KICAgICIiIgogICAgaWYgaW5jX2RlZyA8IElOQ19NSU5fREVHOgogICAgICAgIHJldHVybiBGYWxzZSwgKAogICAgICAgICAgICBmImluY2xpbmF0aW9uIHtpbmNfZGVnOi4xZn0gZGVnIGlzIGJlbG93IHtJTkNfTUlOX0RFRzouMGZ9IGRlZyAtIHRoZSBnYWxheHkgaXMgIgogICAgICAgICAgICAiY2xvc2UgdG8gZmFjZS1vbiwgdGhlIGNvcnJlY3Rpb24gd291bGQgYmUgc21hbGxlciB0aGFuIGl0cyBvd24gdW5jZXJ0YWludHksICIKICAgICAgICAgICAgImFuZCB0aGUgcG9zaXRpb24gYW5nbGUgaXMgcG9vcmx5IGRlZmluZWQuIE5vdCBjb3JyZWN0ZWQuIgogICAgICAgICkKICAgIGlmIGluY19kZWcgPiBJTkNfTUFYX0RFRzoKICAgICAgICByZXR1cm4gRmFsc2UsICgKICAgICAgICAgICAgZiJpbmNsaW5hdGlvbiB7aW5jX2RlZzouMWZ9IGRlZyBpcyBhYm92ZSB7SU5DX01BWF9ERUc6LjBmfSBkZWcgLSB0aGUgY29ycmVjdGlvbiAiCiAgICAgICAgICAgICJ3b3VsZCBiZSBsYXJnZSBidXQgdW5yZWxpYWJsZSwgYW5kIGR1c3QgYmlhc2VzIGl0IHN5c3RlbWF0aWNhbGx5LiBOb3QgY29ycmVjdGVkLiIKICAgICAgICApCiAgICByZXR1cm4gVHJ1ZSwgZiJpbmNsaW5hdGlvbiB7aW5jX2RlZzouMWZ9IGRlZyBpcyBpbnNpZGUgdGhlIHVzYWJsZSB3aW5kb3cuIgoKZGVmIHNreV9wYV90b19pbWFnZV9hbmdsZShwYV9za3lfZGVnOiBmbG9hdCwKICAgICAgICAgICAgICAgICAgICAgICAgICBub3J0aF9hbmdsZV9kZWc6IGZsb2F0ID0gOTAuMCwKICAgICAgICAgICAgICAgICAgICAgICAgICBtaXJyb3JlZDogYm9vbCA9IEZhbHNlKSAtPiBmbG9hdDoKICAgICIiIkNvbnZlcnQgYSBjYXRhbG9ndWUgcG9zaXRpb24gYW5nbGUgaW50byBhbiBhbmdsZSBpbiBpbWFnZSBwaXhlbCBjb29yZGluYXRlcy4KCiAgICBDYXRhbG9ndWVzIGdpdmUgdGhlIHBvc2l0aW9uIGFuZ2xlIG1lYXN1cmVkIG9uIHRoZSBza3ksIGZyb20gTm9ydGggdG93YXJkcyBFYXN0LgogICAgT3VyIG9iamVjdCBwb3NpdGlvbnMgYXJlIGluIGltYWdlIHBpeGVscywgd2hlcmUgdGhlIG1ham9yIGF4aXMgaGFzIHNvbWUgYW5nbGUKICAgIG1lYXN1cmVkIGZyb20gdGhlICt4IGF4aXMuIFRoZXNlIGFyZSBub3QgdGhlIHNhbWUgbnVtYmVyIGFuZCBjb25mdXNpbmcgdGhlbQogICAgcm90YXRlcyB0aGUgd2hvbGUgZGVwcm9qZWN0aW9uLgoKICAgICAgICBub3J0aF9hbmdsZV9kZWcgIGFuZ2xlIG9mIE5vcnRoIGluIHRoZSBpbWFnZSwgY291bnRlcmNsb2Nrd2lzZSBmcm9tICt4LgogICAgICAgICAgICAgICAgICAgICAgICAgOTAgZm9yIHRoZSB1c3VhbCAibm9ydGggdXAiIG9yaWVudGF0aW9uLgogICAgICAgIG1pcnJvcmVkICAgICAgICAgVHJ1ZSBpZiB0aGUgaW1hZ2UgcGFyaXR5IGlzIGZsaXBwZWQsIHNvIHRoYXQgRWFzdCBydW5zCiAgICAgICAgICAgICAgICAgICAgICAgICBjbG9ja3dpc2UgZnJvbSBOb3J0aCBpbnN0ZWFkIG9mIGNvdW50ZXJjbG9ja3dpc2UuCgogICAgSWYgeW91IGFscmVhZHkga25vdyB0aGUgbWFqb3ItYXhpcyBhbmdsZSBkaXJlY3RseSBpbiBpbWFnZSBjb29yZGluYXRlcywgc2tpcAogICAgdGhpcyBmdW5jdGlvbiBhbmQgcGFzcyB0aGF0IGFuZ2xlIHN0cmFpZ2h0IHRvIGRlcHJvamVjdCgpLgogICAgIiIiCiAgICBzaWduID0gLTEuMCBpZiBtaXJyb3JlZCBlbHNlIDEuMAogICAgcmV0dXJuIChub3J0aF9hbmdsZV9kZWcgKyBzaWduICogcGFfc2t5X2RlZykgJSAxODAuMAoKZGVmIGRlcHJvamVjdChkeCwgZHksIG1ham9yX2F4aXNfYW5nbGVfZGVnOiBmbG9hdCwgaW5jX2RlZzogZmxvYXQpOgogICAgIiIiU3RyZXRjaCBjZW50cmVkIHBvc2l0aW9ucyBmcm9tIHRoZSBwbGFuZSBvZiB0aGUgc2t5IGludG8gdGhlIGdhbGF4eSBwbGFuZS4KCiAgICBkeCwgZHkgbXVzdCBhbHJlYWR5IGJlIG1lYXN1cmVkIEZST00gVEhFIEdBTEFYWSBDRU5UUkUuIFRoaXMgbWF0dGVyczogdGhlCiAgICB0cmFuc2Zvcm0gaXMgbGluZWFyLCBzbyBhcHBseWluZyBpdCB0byB1bmNlbnRyZWQgY29vcmRpbmF0ZXMgbGVhdmVzIHRoZSBzaGFwZQogICAgY29ycmVjdCBidXQgbW92ZXMgdGhlIGNlbnRyZSwgYW5kIGV2ZXJ5IHJhZGl1cyBtZWFzdXJlZCBmcm9tIHRoZSBvbGQgY2VudHJlCiAgICB0aGVuIGNvbWVzIG91dCB3cm9uZy4KCiAgICBtYWpvcl9heGlzX2FuZ2xlX2RlZyBpcyBtZWFzdXJlZCBpbiB0aGUgc2FtZSBmcmFtZSBhcyBkeCwgZHkgLSBjb3VudGVyY2xvY2t3aXNlCiAgICBmcm9tIHRoZSAreCBheGlzLiBVc2Ugc2t5X3BhX3RvX2ltYWdlX2FuZ2xlKCkgaWYgeW91IGFyZSBzdGFydGluZyBmcm9tIGEKICAgIGNhdGFsb2d1ZSBwb3NpdGlvbiBhbmdsZS4KCiAgICBUaGUgdHJhbnNmb3JtIHJvdGF0ZXMgdGhlIG1ham9yIGF4aXMgb250byB4LCBzdHJldGNoZXMgdGhlIHBlcnBlbmRpY3VsYXIKICAgIGRpcmVjdGlvbiBieSAxL2NvcyhpKSwgYW5kIHJvdGF0ZXMgYmFjay4gVGhlIHJlc3VsdGluZyBtYXRyaXggaXMgc3ltbWV0cmljLAogICAgYXMgYSBwdXJlIHN0cmV0Y2ggYWxvbmcgYW4gYXhpcyBtdXN0IGJlLgogICAgIiIiCiAgICBkeCA9IG5wLmFzYXJyYXkoZHgsIGR0eXBlPWZsb2F0KQogICAgZHkgPSBucC5hc2FycmF5KGR5LCBkdHlwZT1mbG9hdCkKCiAgICB0aCA9IG1hdGgucmFkaWFucyhtYWpvcl9heGlzX2FuZ2xlX2RlZykKICAgIGMsIHMgPSBtYXRoLmNvcyh0aCksIG1hdGguc2luKHRoKQogICAgayA9IHN0cmV0Y2hfZmFjdG9yKGluY19kZWcpCgogICAgbTExID0gYyAqIGMgKyBrICogcyAqIHMKICAgIG0xMiA9ICgxLjAgLSBrKSAqIHMgKiBjCiAgICBtMjIgPSBzICogcyArIGsgKiBjICogYwoKICAgIHJldHVybiBkeCAqIG0xMSArIGR5ICogbTEyLCBkeCAqIG0xMiArIGR5ICogbTIyCgpkZWYgZWZmZWN0aXZlX3JpbmdfYXJlYXMoZWRnZXMsCiAgICAgICAgICAgICAgICAgICAgICAgICBmcmFtZV9jb3JuZXJzLAogICAgICAgICAgICAgICAgICAgICAgICAgY2VudHJlX3h5LAogICAgICAgICAgICAgICAgICAgICAgICAgbWFqb3JfYXhpc19hbmdsZV9kZWc6IGZsb2F0IHwgTm9uZSwKICAgICAgICAgICAgICAgICAgICAgICAgIGluY19kZWc6IGZsb2F0IHwgTm9uZSwKICAgICAgICAgICAgICAgICAgICAgICAgIG5fc2FtcGxlczogaW50ID0gNDAwXzAwMCwKICAgICAgICAgICAgICAgICAgICAgICAgIHJuZ19zZWVkOiBpbnQgPSAxMjM0NSk6CiAgICAiIiJBcmVhIG9mIGVhY2ggYW5udWx1cyB0aGF0IGFjdHVhbGx5IGZhbGxzIGluc2lkZSB0aGUgaW1hZ2VkIGZyYW1lLgoKICAgIFdIWSBUSElTIEVYSVNUUwogICAgQSByaW5nIGRyYXduIGFyb3VuZCB0aGUgZ2FsYXh5IGNlbnRyZSBpcyBvbmx5IHBhcnRseSBjb3ZlcmVkIGJ5IHRoZSBpbWFnZTogYXQKICAgIGxhcmdlIHJhZGlpIHRoZSBjb3JuZXJzIG9mIHRoZSBmcmFtZSBjdXQgaXQuIERpdmlkaW5nIGNvdW50cyBieSB0aGUgZnVsbAogICAgZ2VvbWV0cmljIHJpbmcgYXJlYSB0aGVyZWZvcmUgbWFrZXMgdGhlIG91dGVyIGJpbnMgbG9vayBlbXB0aWVyIHRoYW4gdGhleSBhcmUsCiAgICBhbmQgcHJvZHVjZXMgYSByYWRpYWwgZGVjbGluZSB0aGF0IGlzIHB1cmUgZ2VvbWV0cnkuCgogICAgVGhpcyBpcyBub3QgaHlwb3RoZXRpY2FsLiBJbiBNMzMgYSBjYXRhbG9ndWUgY292ZXJpbmcgb25seSB0aGUgY2VudHJhbCAyMyd4MjMnCiAgICBwcm9kdWNlZCAiYW4gYXJ0aWZpY2lhbGx5IGxhcmdlIGRlY3JlYXNlIGF0IHJhZGlpIGxhcmdlciB0aGFuIHRoZSBsYXJnZXN0IGNpcmNsZQogICAgaW5zY3JpYmVkIGluIHRoZSBhcmVhIiwgYW5kIHRoZSBwdWJsaXNoZWQgZXhwb25lbnRpYWwgc2xvcGUgd2FzIHdyb25nIGJ5IGEKICAgIGZhY3RvciBvZiB0aHJlZSB1bnRpbCBhIHNpemUgY3V0IGFuZCBhIHJhZGlhbCB0cnVuY2F0aW9uIHdlcmUgYXBwbGllZC4KCiAgICBNRVRIT0QKICAgIFNjYXR0ZXIgcmFuZG9tIHBvaW50cyB1bmlmb3JtbHkgb3ZlciB0aGUgZnJhbWUsIHB1dCB0aGVtIHRocm91Z2ggZXhhY3RseSB0aGUKICAgIHNhbWUgZGVwcm9qZWN0aW9uIGFzIHRoZSByZWFsIG9iamVjdHMsIGFuZCBoaXN0b2dyYW0gdGhlaXIgcmFkaWkuIFRoZSBmcmFjdGlvbgogICAgbGFuZGluZyBpbiBlYWNoIHJpbmcsIHRpbWVzIHRoZSBmcmFtZSBhcmVhLCBpcyB0aGF0IHJpbmcncyBlZmZlY3RpdmUgYXJlYS4KICAgIFRoaXMgaGFuZGxlcyBhbnkgZnJhbWUgc2hhcGUgYW5kIGFueSBkZXByb2plY3Rpb24gd2l0aG91dCBzcGVjaWFsLWNhc2luZy4KCiAgICBmcmFtZV9jb3JuZXJzIGlzICh4bWluLCB4bWF4LCB5bWluLCB5bWF4KSBpbiB0aGUgc2FtZSBwaXhlbCB1bml0cyBhcyB0aGUgZGF0YS4KICAgIFBhc3MgbWFqb3JfYXhpc19hbmdsZV9kZWc9Tm9uZSB0byBza2lwIGRlcHJvamVjdGlvbiAodGhlIHVuY29ycmVjdGVkIGNhc2UpLgogICAgIiIiCiAgICB4bWluLCB4bWF4LCB5bWluLCB5bWF4ID0gZnJhbWVfY29ybmVycwogICAgcm5nID0gbnAucmFuZG9tLmRlZmF1bHRfcm5nKHJuZ19zZWVkKQoKICAgIHN4ID0gcm5nLnVuaWZvcm0oeG1pbiwgeG1heCwgbl9zYW1wbGVzKQogICAgc3kgPSBybmcudW5pZm9ybSh5bWluLCB5bWF4LCBuX3NhbXBsZXMpCiAgICBmcmFtZV9hcmVhID0gKHhtYXggLSB4bWluKSAqICh5bWF4IC0geW1pbikKCiAgICBzZHggPSBzeCAtIGNlbnRyZV94eVswXQogICAgc2R5ID0gc3kgLSBjZW50cmVfeHlbMV0KICAgIGlmIG1ham9yX2F4aXNfYW5nbGVfZGVnIGlzIG5vdCBOb25lIGFuZCBpbmNfZGVnIGlzIG5vdCBOb25lOgogICAgICAgIHNkeCwgc2R5ID0gZGVwcm9qZWN0KHNkeCwgc2R5LCBtYWpvcl9heGlzX2FuZ2xlX2RlZywgaW5jX2RlZykKCiAgICBzciA9IG5wLmh5cG90KHNkeCwgc2R5KQogICAgY291bnRzLCBfID0gbnAuaGlzdG9ncmFtKHNyLCBiaW5zPWVkZ2VzKQogICAgcmV0dXJuIGNvdW50cyAvIGZsb2F0KG5fc2FtcGxlcykgKiBmcmFtZV9hcmVhCgpkZWYgcmFkaWFsX2RlbnNpdHkoZHgsIGR5LAogICAgICAgICAgICAgICAgICAgbl9yaW5nczogaW50ID0gMjAsCiAgICAgICAgICAgICAgICAgICByX21heDogZmxvYXQgfCBOb25lID0gTm9uZSwKICAgICAgICAgICAgICAgICAgIGZyYW1lX2Nvcm5lcnM9Tm9uZSwKICAgICAgICAgICAgICAgICAgIGNlbnRyZV94eT1Ob25lLAogICAgICAgICAgICAgICAgICAgbWFqb3JfYXhpc19hbmdsZV9kZWc6IGZsb2F0IHwgTm9uZSA9IE5vbmUsCiAgICAgICAgICAgICAgICAgICBpbmNfZGVnOiBmbG9hdCB8IE5vbmUgPSBOb25lLAogICAgICAgICAgICAgICAgICAgbWluX2FyZWFfZnJhY3Rpb246IGZsb2F0ID0gMC4xNSk6CiAgICAiIiJTdXJmYWNlIGRlbnNpdHkgb2Ygb2JqZWN0cyBpbiBlcXVhbC13aWR0aCBjb25jZW50cmljIHJpbmdzLgoKICAgIEVxdWFsLXdpZHRoIHJpbmdzIG91dCB0byB0aGUgb3V0ZXJtb3N0IG9iamVjdCwgZm9sbG93aW5nIHRoZSBzdGFuZGFyZCBwcmFjdGljZQogICAgaW4gdGhpcyBsaXRlcmF0dXJlICgyMCByaW5ncywgZWFjaCAwLjA1IG9mIHRoZSBtYXhpbXVtIHJhZGl1cykuCgogICAgUmluZ3Mgd2hvc2UgZWZmZWN0aXZlIGFyZWEgaXMgYSBzbWFsbCBmcmFjdGlvbiBvZiB0aGVpciBmdWxsIGdlb21ldHJpYyBhcmVhIGFyZQogICAgbWFya2VkIGluY29tcGxldGUgcmF0aGVyIHRoYW4gcGxvdHRlZCwgYmVjYXVzZSB0aGVpciBkZW5zaXR5IGlzIGRvbWluYXRlZCBieSBob3cKICAgIHRoZSBmcmFtZSBoYXBwZW5zIHRvIGN1dCB0aGVtLiBtaW5fYXJlYV9mcmFjdGlvbiBzZXRzIHRoYXQgdGhyZXNob2xkLgoKICAgIFJldHVybnMgYSBkaWN0IHdpdGggcmluZyBjZW50cmVzLCBjb3VudHMsIGFyZWFzLCBkZW5zaXRpZXMgYW5kIFBvaXNzb24gZXJyb3JzLgogICAgIiIiCiAgICBkeCA9IG5wLmFzYXJyYXkoZHgsIGR0eXBlPWZsb2F0KQogICAgZHkgPSBucC5hc2FycmF5KGR5LCBkdHlwZT1mbG9hdCkKICAgIHIgPSBucC5oeXBvdChkeCwgZHkpCgogICAgaWYgcl9tYXggaXMgTm9uZToKICAgICAgICByX21heCA9IGZsb2F0KHIubWF4KCkpIGlmIHIuc2l6ZSBlbHNlIDEuMAogICAgZWRnZXMgPSBucC5saW5zcGFjZSgwLjAsIHJfbWF4LCBuX3JpbmdzICsgMSkKICAgIGNlbnRyZXMgPSAwLjUgKiAoZWRnZXNbOi0xXSArIGVkZ2VzWzE6XSkKCiAgICBjb3VudHMsIF8gPSBucC5oaXN0b2dyYW0ociwgYmlucz1lZGdlcykKICAgIGdlb21ldHJpYyA9IG1hdGgucGkgKiAoZWRnZXNbMTpdICoqIDIgLSBlZGdlc1s6LTFdICoqIDIpCgogICAgaWYgZnJhbWVfY29ybmVycyBpcyBub3QgTm9uZSBhbmQgY2VudHJlX3h5IGlzIG5vdCBOb25lOgogICAgICAgIGFyZWEgPSBlZmZlY3RpdmVfcmluZ19hcmVhcyhlZGdlcywgZnJhbWVfY29ybmVycywgY2VudHJlX3h5LAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYWpvcl9heGlzX2FuZ2xlX2RlZywgaW5jX2RlZykKICAgIGVsc2U6CiAgICAgICAgYXJlYSA9IGdlb21ldHJpYwoKICAgIHdpdGggbnAuZXJyc3RhdGUoZGl2aWRlPSJpZ25vcmUiLCBpbnZhbGlkPSJpZ25vcmUiKToKICAgICAgICBkZW5zaXR5ID0gbnAud2hlcmUoYXJlYSA+IDAsIGNvdW50cyAvIGFyZWEsIG5wLm5hbikKICAgICAgICBlcnIgPSBucC53aGVyZShhcmVhID4gMCwgbnAuc3FydChucC5tYXhpbXVtKGNvdW50cywgMCkpIC8gYXJlYSwgbnAubmFuKQogICAgICAgIGZyYWMgPSBucC53aGVyZShnZW9tZXRyaWMgPiAwLCBhcmVhIC8gZ2VvbWV0cmljLCAwLjApCgogICAgY29tcGxldGUgPSBmcmFjID49IG1pbl9hcmVhX2ZyYWN0aW9uCgogICAgcmV0dXJuIHsKICAgICAgICAiZWRnZXMiOiBlZGdlcywKICAgICAgICAiciI6IGNlbnRyZXMsCiAgICAgICAgImNvdW50cyI6IGNvdW50cywKICAgICAgICAiYXJlYSI6IGFyZWEsCiAgICAgICAgImFyZWFfZnJhY3Rpb24iOiBmcmFjLAogICAgICAgICJkZW5zaXR5IjogZGVuc2l0eSwKICAgICAgICAiZGVuc2l0eV9lcnIiOiBlcnIsCiAgICAgICAgImNvbXBsZXRlIjogY29tcGxldGUsCiAgICAgICAgInJfbWF4Ijogcl9tYXgsCiAgICB9CgpkZWYgc2VjdG9yX2Rpc3BlcnNpb24oZHgsIGR5LCBtYWpvcl9heGlzX2FuZ2xlX2RlZywgaW5jX2RlZywgbl9zZWN0b3JzPTIwKToKICAgICIiIkhvdyB1bmV2ZW4gdGhlIG9iamVjdCBjb3VudHMgYXJlIG9uY2UgdGhlIGZpZWxkIGlzIGN1dCBpbnRvIGVxdWFsIHdlZGdlcy4KCiAgICBUSEUgSURFQQogICAgU2xpY2UgdGhlIGdhbGF4eSBpbnRvIGVxdWFsIHNlY3RvcnMsIGxpa2Ugc2xpY2VzIG9mIGEgcGllLiBJZiB0aGUgZGlzayByZWFsbHkgaXMKICAgIHJvdW5kIGFuZCB3ZSBhcmUgbG9va2luZyBzdHJhaWdodCBkb3duIG9uIGl0LCBldmVyeSBzbGljZSBob2xkcyByb3VnaGx5IHRoZSBzYW1lCiAgICBudW1iZXIgb2Ygb2JqZWN0cy4gU2VlbiBhdCBhbiBhbmdsZSBpdCBkb2VzIG5vdC4gU28gdGhlIHRpbHQgdGhhdCBldmVucyB0aGUKICAgIHNsaWNlcyBvdXQgaXMgdGhlIHRpbHQgdGhhdCB1bmRvZXMgdGhlIHByb2plY3Rpb24uCgogICAgT3Bwb3NpdGUgc2xpY2VzIGFyZSBhZGRlZCB0b2dldGhlciBiZWZvcmUgdGhlIGRpc3BlcnNpb24gaXMgbWVhc3VyZWQuIEEgZ2FsYXh5CiAgICB0aGF0IGlzIGxvcHNpZGVkIC0gYSBicmlnaHRlciBhcm0gb24gb25lIHNpZGUsIGEgY29tcGFuaW9uIHB1bGxpbmcgYXQgaXQgLSB3b3VsZAogICAgb3RoZXJ3aXNlIHB1c2ggdGhlIGFuc3dlciBhcm91bmQ7IGFkZGluZyBhIHNsaWNlIHRvIHRoZSBvbmUgZmFjaW5nIGl0IGNhbmNlbHMKICAgIHRoYXQga2luZCBvZiBhc3ltbWV0cnkuCgogICAgUmV0dXJucyBzdGQgLyBtZWFuIG9mIHRoZSBmb2xkZWQgY291bnRzLiBTbWFsbGVyIGlzIGJldHRlci4KICAgICIiIgogICAgZHhwLCBkeXAgPSBkZXByb2plY3QoZHgsIGR5LCBtYWpvcl9heGlzX2FuZ2xlX2RlZywgaW5jX2RlZykKICAgIHRoZXRhID0gbnAuYXJjdGFuMihkeXAsIGR4cCkKICAgIGlkeCA9IG5wLmZsb29yKCh0aGV0YSAlICgyLjAgKiBtYXRoLnBpKSkgLyAoMi4wICogbWF0aC5waSkgKiBuX3NlY3RvcnMpLmFzdHlwZShpbnQpCiAgICBpZHggPSBucC5jbGlwKGlkeCwgMCwgbl9zZWN0b3JzIC0gMSkKICAgIGNvdW50cyA9IG5wLmJpbmNvdW50KGlkeCwgbWlubGVuZ3RoPW5fc2VjdG9ycykuYXN0eXBlKGZsb2F0KQoKICAgIGhhbGYgPSBuX3NlY3RvcnMgLy8gMgogICAgZm9sZGVkID0gY291bnRzWzpoYWxmXSArIGNvdW50c1toYWxmOl0KICAgIG1lYW4gPSBmb2xkZWQubWVhbigpCiAgICBpZiBtZWFuIDw9IDA6CiAgICAgICAgcmV0dXJuIG5wLmluZgogICAgcmV0dXJuIGZsb2F0KGZvbGRlZC5zdGQoKSAvIG1lYW4pCgpkZWYgZml0X3NlY3Rvcl9tZXRob2QoZHgsIGR5LAogICAgICAgICAgICAgICAgICAgICAgbl9zZWN0b3JzPTIwLAogICAgICAgICAgICAgICAgICAgICAgYW5nbGVfc3RlcD01LjAsCiAgICAgICAgICAgICAgICAgICAgICBpbmNfc3RlcD01LjAsCiAgICAgICAgICAgICAgICAgICAgICBpbmNfcmFuZ2U9KDIwLjAsIDgwLjApLAogICAgICAgICAgICAgICAgICAgICAgcl9tYXg9Tm9uZSk6CiAgICAiIiJTZWFyY2ggZm9yIHRoZSBkZXByb2plY3Rpb24gYW5nbGVzIHRoYXQgbWFrZSB0aGUgb2JqZWN0IGNvdW50cyBtb3N0IGV2ZW4uCgogICAgU2NhbnMgYSBncmlkIG9mIHBvc2l0aW9uIGFuZ2xlIGFuZCBpbmNsaW5hdGlvbiBhbmQgcmV0dXJucyB0aGUgcGFpciB0aGF0CiAgICBtaW5pbWlzZXMgc2VjdG9yX2Rpc3BlcnNpb24oKS4gVGhpcyBpcyB0aGUgc2Vjb25kIG9mIHRoZSB0d28gbWV0aG9kcyBpbnRyb2R1Y2VkCiAgICBieSBHYXJjaWEtR29tZXogYW5kIEF0aGFuYXNzb3VsYSwgYW5kIGluIHRoZWlyIGNvbXBhcmlzb24gaXQgaXMgdGhlIG9uZSB0aGF0CiAgICBhZ3JlZXMgbW9zdCBjbG9zZWx5IHdpdGgga2luZW1hdGljIGRldGVybWluYXRpb25zIC0gMC45NiBpbiBwb3NpdGlvbiBhbmdsZSBhbmQKICAgIDAuOTAgaW4gY29zKGluY2xpbmF0aW9uKSwgYmV0dGVyIHRoYW4gdGhlaXIgRm91cmllciBtZXRob2QgYW5kIGJldHRlciB0aGFuCiAgICBwaG90b21ldHJ5LgoKICAgIFdIRU4gSVQgRkFJTFMgLSByZWFkIHRoaXMgYmVmb3JlIHRydXN0aW5nIHRoZSBhbnN3ZXIKICAgICAgKiBCZWxvdyByb3VnaGx5IDE1MC0yMDAgb2JqZWN0cyB0aGUgc3VyZmFjZSBiZWluZyBtaW5pbWlzZWQgb2Z0ZW4gaGFzIHNldmVyYWwKICAgICAgICBtaW5pbWEgcmF0aGVyIHRoYW4gb25lLCBhbmQgdGhlIGRlZXBlc3QgaXMgbm90IGFsd2F5cyB0aGUgcmlnaHQgb25lLiBUaGUKICAgICAgICByZXR1cm5lZCBkaWN0IHJlcG9ydHMgdGhlIHJ1bm5lcnMtdXAgc28geW91IGNhbiBsb29rLgogICAgICAqIElmIHRoZSBvYmplY3RzIHRyYWNlIHRoZSBzcGlyYWwgYXJtcyBvciBhIGJhciBpbnN0ZWFkIG9mIGZpbGxpbmcgdGhlIGRpc2ssCiAgICAgICAgdGhlIG1ldGhvZCBjaXJjdWxhcmlzZXMgdGhlIGFybXMgb3IgdGhlIGJhciByYXRoZXIgdGhhbiB0aGUgZGlzay4gSW4gdGhlCiAgICAgICAgcHVibGlzaGVkIG5vdGVzIHRoaXMgaXMgZXhhY3RseSB3aGF0IHdlbnQgd3JvbmcgZm9yIHNldmVyYWwgYmFycmVkIGdhbGF4aWVzLgogICAgICAqIEEgcmluZyBvZiBvYmplY3RzIGlzIGNpcmN1bGFyaXNlZCBhcyBpZiBpdCB3ZXJlIHRoZSBkaXNrLiBBbiBvdXRlciByaW5nIHdpdGgKICAgICAgICBhbiBpbnRyaW5zaWMgYXhpcyByYXRpbyBvZiAwLjg3IGZha2VzIGFuIGluY2xpbmF0aW9uIG9mIGFib3V0IDEyIGRlZ3JlZXMgaW4gYQogICAgICAgIGdhbGF4eSB0aGF0IGlzIHJlYWxseSBmYWNlLW9uLgogICAgICAqIEl0IHdvcmtzIGJldHRlciB0aGUgbW9yZSBpbmNsaW5lZCB0aGUgZ2FsYXh5IGlzLCBhbmQgcG9vcmx5IG5lYXIgZmFjZS1vbiwKICAgICAgICB3aGVyZSB0aGUgcG9zaXRpb24gYW5nbGUgaXMgYmFyZWx5IGRlZmluZWQgaW4gdGhlIGZpcnN0IHBsYWNlLgoKICAgIHJfbWF4IG9wdGlvbmFsbHkgcmVzdHJpY3RzIHRoZSBmaXQgdG8gb2JqZWN0cyBpbnNpZGUgdGhhdCByYWRpdXMsIHdoaWNoIGhlbHBzCiAgICB3aGVuIGEgZmV3IGZhci1mbHVuZyBvYmplY3RzIGRvbWluYXRlIHRoZSBvdXRlciBzZWN0b3JzLgogICAgIiIiCiAgICBkeCA9IG5wLmFzYXJyYXkoZHgsIGR0eXBlPWZsb2F0KQogICAgZHkgPSBucC5hc2FycmF5KGR5LCBkdHlwZT1mbG9hdCkKCiAgICBpZiByX21heCBpcyBub3QgTm9uZToKICAgICAgICBrZWVwID0gbnAuaHlwb3QoZHgsIGR5KSA8PSByX21heAogICAgICAgIGR4LCBkeSA9IGR4W2tlZXBdLCBkeVtrZWVwXQoKICAgIGFuZ2xlcyA9IG5wLmFyYW5nZSgwLjAsIDE4MC4wLCBhbmdsZV9zdGVwKQogICAgaW5jcyA9IG5wLmFyYW5nZShpbmNfcmFuZ2VbMF0sIGluY19yYW5nZVsxXSArIDFlLTksIGluY19zdGVwKQoKICAgIGdyaWQgPSBucC5mdWxsKChsZW4oaW5jcyksIGxlbihhbmdsZXMpKSwgbnAuaW5mKQogICAgZm9yIGksIGluYyBpbiBlbnVtZXJhdGUoaW5jcyk6CiAgICAgICAgZm9yIGosIGFuZyBpbiBlbnVtZXJhdGUoYW5nbGVzKToKICAgICAgICAgICAgZ3JpZFtpLCBqXSA9IHNlY3Rvcl9kaXNwZXJzaW9uKGR4LCBkeSwgZmxvYXQoYW5nKSwgZmxvYXQoaW5jKSwgbl9zZWN0b3JzKQoKICAgIGZsYXQgPSBucC5hcmdzb3J0KGdyaWQsIGF4aXM9Tm9uZSkKICAgIGJlc3RfaSwgYmVzdF9qID0gbnAudW5yYXZlbF9pbmRleChmbGF0WzBdLCBncmlkLnNoYXBlKQoKICAgICMgUmVwb3J0IGRpc3RpbmN0IHJ1bm5lcnMtdXA6IGxvY2FsIG1pbmltYSB3ZWxsIHNlcGFyYXRlZCBmcm9tIHRoZSBiZXN0IG9uZSwgc28KICAgICMgdGhhdCBhIHNoYWxsb3cgZ2xvYmFsIG1pbmltdW0gZG9lcyBub3QgaGlkZSBhIGNvbXBldGluZyBzb2x1dGlvbi4KICAgIHJ1bm5lcnMgPSBbXQogICAgZm9yIGsgaW4gZmxhdFsxOl06CiAgICAgICAgaSwgaiA9IG5wLnVucmF2ZWxfaW5kZXgoaywgZ3JpZC5zaGFwZSkKICAgICAgICBzZXBfYW5nID0gbWluKGFicyhhbmdsZXNbal0gLSBhbmdsZXNbYmVzdF9qXSksCiAgICAgICAgICAgICAgICAgICAgICAxODAuMCAtIGFicyhhbmdsZXNbal0gLSBhbmdsZXNbYmVzdF9qXSkpCiAgICAgICAgaWYgc2VwX2FuZyA8IDIwLjAgYW5kIGFicyhpbmNzW2ldIC0gaW5jc1tiZXN0X2ldKSA8IDE1LjA6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgaWYgYW55KG1pbihhYnMoYW5nbGVzW2pdIC0gYSksIDE4MC4wIC0gYWJzKGFuZ2xlc1tqXSAtIGEpKSA8IDIwLjAKICAgICAgICAgICAgICAgYW5kIGFicyhpbmNzW2ldIC0gYykgPCAxNS4wIGZvciBhLCBjLCBfIGluIHJ1bm5lcnMpOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHJ1bm5lcnMuYXBwZW5kKChmbG9hdChhbmdsZXNbal0pLCBmbG9hdChpbmNzW2ldKSwgZmxvYXQoZ3JpZFtpLCBqXSkpKQogICAgICAgIGlmIGxlbihydW5uZXJzKSA+PSAzOgogICAgICAgICAgICBicmVhawoKICAgIHJldHVybiB7CiAgICAgICAgImFuZ2xlIjogZmxvYXQoYW5nbGVzW2Jlc3Rfal0pLAogICAgICAgICJpbmMiOiBmbG9hdChpbmNzW2Jlc3RfaV0pLAogICAgICAgICJkaXNwZXJzaW9uIjogZmxvYXQoZ3JpZFtiZXN0X2ksIGJlc3Rfal0pLAogICAgICAgICJuX29iamVjdHMiOiBpbnQoZHguc2l6ZSksCiAgICAgICAgIm5fc2VjdG9ycyI6IG5fc2VjdG9ycywKICAgICAgICAiYW5nbGVzIjogYW5nbGVzLAogICAgICAgICJpbmNzIjogaW5jcywKICAgICAgICAiZ3JpZCI6IGdyaWQsCiAgICAgICAgInJ1bm5lcnNfdXAiOiBydW5uZXJzLAogICAgICAgICJmZXdfb2JqZWN0cyI6IGJvb2woZHguc2l6ZSA8IDIwMCksCiAgICB9CgpkZWYgZGlzdHJpYnV0aW9uX3NoYXBlKGR4LCBkeSk6CiAgICAiIiJUaGUgYXhpcyByYXRpbyBhbmQgcG9zaXRpb24gYW5nbGUgb2YgYSBzZXQgb2YgcG9zaXRpb25zLgoKICAgIEZyb20gdGhlIHNlY29uZCBtb21lbnRzOiB0aGUgc2hhcGUgb2YgdGhlIGNsb3VkIG9mIHBvaW50cywgbm90IG9mIGFueSBvbmUgb2YKICAgIHRoZW0uIFJldHVybnMgKGF4aXNfcmF0aW8sIGFuZ2xlX2RlZyksIHRoZSBhbmdsZSBtZWFzdXJlZCBmcm9tICt4IHRoZSBzYW1lIHdheQogICAgdGhlIGRlcHJvamVjdGlvbiBtZWFzdXJlcyBpdC4KICAgICIiIgogICAgZHggPSBucC5hc2FycmF5KGR4LCBkdHlwZT1mbG9hdCkKICAgIGR5ID0gbnAuYXNhcnJheShkeSwgZHR5cGU9ZmxvYXQpCiAgICBvayA9IG5wLmlzZmluaXRlKGR4KSAmIG5wLmlzZmluaXRlKGR5KQogICAgaWYgb2suc3VtKCkgPCAyMDoKICAgICAgICByZXR1cm4gZmxvYXQoIm5hbiIpLCBmbG9hdCgibmFuIikKICAgIGNvdiA9IG5wLmNvdihkeFtva10sIGR5W29rXSkKICAgIHZhbHVlcywgdmVjdG9ycyA9IG5wLmxpbmFsZy5laWdoKGNvdikKICAgIGlmIHZhbHVlc1sxXSA8PSAwOgogICAgICAgIHJldHVybiBmbG9hdCgibmFuIiksIGZsb2F0KCJuYW4iKQogICAgcmF0aW8gPSBmbG9hdChucC5zcXJ0KG1heCh2YWx1ZXNbMF0sIDAuMCkgLyB2YWx1ZXNbMV0pKQogICAgYW5nbGUgPSBmbG9hdChucC5kZWdyZWVzKG5wLmFyY3RhbjIodmVjdG9yc1sxLCAxXSwgdmVjdG9yc1swLCAxXSkpICUgMTgwLjApCiAgICByZXR1cm4gcmF0aW8sIGFuZ2xlCgpkZWYgY2hlY2tfYWdhaW5zdF9jYXRhbG9ndWUoZHgsIGR5LCBleHBlY3RlZF9yYXRpbywgZXhwZWN0ZWRfYW5nbGUsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICByYXRpb190b2xlcmFuY2U9MC4xNSwgYW5nbGVfdG9sZXJhbmNlPTI1LjApOgogICAgIiIiQ29tcGFyZSB0aGUgc2hhcGUgb2Ygd2hhdCB3YXMgZGV0ZWN0ZWQgd2l0aCB3aGF0IHRoZSBjYXRhbG9ndWUgcHJlZGljdHMuCgogICAgV0hZIFRISVMgSVMgV09SVEggRE9JTkcKICAgIEEgdGlsdGVkIGRpc2sgc2VlbiBmcm9tIGhlcmUgaXMgYW4gZWxsaXBzZSwgYW5kIHRoZSBjYXRhbG9ndWUgc2F5cyB3aGljaAogICAgZWxsaXBzZS4gU28gdGhlIHRyYWNlcnMgZGV0ZWN0ZWQgaW4gaXQgaGF2ZSBhIHNoYXBlIHRoYXQgY2FuIGJlIHByZWRpY3RlZAogICAgYmVmb3JlIHRoZXkgYXJlIG1lYXN1cmVkIC0gYW5kIGNvbXBhcmluZyB0aGUgdHdvIGNvc3RzIG5vdGhpbmcgYW5kIG5lZWRzIG5vCiAgICBleHRyYSBkYXRhLgoKICAgIEl0IGlzIGEgcmVhbCB0ZXN0IGJlY2F1c2UgaXQgY2FuIGZhaWwuIE9uIGEgc2V0IG9mIE0zMyBmcmFtZXMgdGhlIHRvb2wKICAgIGRldGVjdGVkIDI4NDQgcmVnaW9ucyBhbmQgZHJldyBhIHNtb290aCwgZW50aXJlbHkgcGxhdXNpYmxlIHJhZGlhbCBwcm9maWxlOwogICAgdGhlIGRldGVjdGlvbnMgY2FtZSBvdXQgcm91bmQsIGF4aXMgcmF0aW8gMC45Niwgd2hpbGUgdGhlIGNhdGFsb2d1ZSBzYXlzIDAuNTkuCiAgICBUaGV5IHdlcmUgbm90IHRyYWNpbmcgdGhlIGdhbGF4eSBhdCBhbGwgLSB0aGV5IHdlcmUgc3VidHJhY3Rpb24gcmVzaWR1YWxzCiAgICBzcHJlYWQgZXZlbmx5IG92ZXIgdGhlIGZpZWxkIC0gYW5kIG5vdGhpbmcgZWxzZSBpbiB0aGUgb3V0cHV0IHNob3dlZCBpdC4KCiAgICBXaGVuIHRoZSB0d28gYWdyZWUgaXQgaXMgZXZpZGVuY2UgdGhlIGRldGVjdGlvbiBpcyBmaW5kaW5nIHRoZSBnYWxheHkuIFdoZW4KICAgIHRoZXkgZGlzYWdyZWUsIHNvbWV0aGluZyBpcyB3cm9uZyB1cHN0cmVhbSBhbmQgdGhlIHByb2ZpbGUgYmVsb3cgc2hvdWxkIG5vdCBiZQogICAgYmVsaWV2ZWQuIFRoZSBjaGVjayBkb2VzIG5vdCBzYXkgd2hpY2ggb2YgdGhlIHR3byBpcyBhdCBmYXVsdC4KCiAgICBBIGdhbGF4eSB0b28gcm91bmQgZm9yIGl0cyBvd24gcG9zaXRpb24gYW5nbGUgdG8gYmUgZGVmaW5lZCBpcyBub3QgdGVzdGVkIG9uCiAgICBhbmdsZSwgZm9yIHRoZSBzYW1lIHJlYXNvbiB0aGUgY2F0YWxvZ3VlIGRlY2xpbmVzIHRvIHB1Ymxpc2ggb25lLgogICAgIiIiCiAgICByYXRpbywgYW5nbGUgPSBkaXN0cmlidXRpb25fc2hhcGUoZHgsIGR5KQogICAgaWYgbm90IG5wLmlzZmluaXRlKHJhdGlvKToKICAgICAgICByZXR1cm4geyJvayI6IE5vbmUsICJyZWFzb24iOiAidG9vIGZldyBwb3NpdGlvbnMgdG8gbWVhc3VyZSBhIHNoYXBlIn0KCiAgICBvdXQgPSB7Im1lYXN1cmVkX3JhdGlvIjogcmF0aW8sICJtZWFzdXJlZF9hbmdsZSI6IGFuZ2xlLAogICAgICAgICAgICJleHBlY3RlZF9yYXRpbyI6IGV4cGVjdGVkX3JhdGlvLCAiZXhwZWN0ZWRfYW5nbGUiOiBleHBlY3RlZF9hbmdsZX0KCiAgICBwcm9ibGVtcyA9IFtdCiAgICBpZiBleHBlY3RlZF9yYXRpbyBpcyBub3QgTm9uZSBhbmQgbnAuaXNmaW5pdGUoZXhwZWN0ZWRfcmF0aW8pOgogICAgICAgIGlmIGFicyhyYXRpbyAtIGV4cGVjdGVkX3JhdGlvKSA+IHJhdGlvX3RvbGVyYW5jZToKICAgICAgICAgICAgcHJvYmxlbXMuYXBwZW5kKAogICAgICAgICAgICAgICAgInRoZSBkZXRlY3Rpb25zIGFyZSBzaGFwZWQgezouMmZ9IHdoZXJlIHRoZSBjYXRhbG9ndWUgc2F5cyB7Oi4yZn0iCiAgICAgICAgICAgICAgICAuZm9ybWF0KHJhdGlvLCBleHBlY3RlZF9yYXRpbykpCiAgICAjIEEgcm91bmQgZ2FsYXh5IGhhcyBubyBtZWFuaW5nZnVsIG1ham9yIGF4aXMsIHNvIGl0cyBhbmdsZSBpcyBub3QgdGVzdGVkLgogICAgaWYgKGV4cGVjdGVkX2FuZ2xlIGlzIG5vdCBOb25lIGFuZCBucC5pc2Zpbml0ZShleHBlY3RlZF9hbmdsZSkKICAgICAgICAgICAgYW5kIGV4cGVjdGVkX3JhdGlvIGlzIG5vdCBOb25lIGFuZCBleHBlY3RlZF9yYXRpbyA8IDAuODUpOgogICAgICAgIGdhcCA9IGFicygoYW5nbGUgLSBleHBlY3RlZF9hbmdsZSArIDkwLjApICUgMTgwLjAgLSA5MC4wKQogICAgICAgIG91dFsiYW5nbGVfZ2FwIl0gPSBnYXAKICAgICAgICBpZiBnYXAgPiBhbmdsZV90b2xlcmFuY2U6CiAgICAgICAgICAgIHByb2JsZW1zLmFwcGVuZCgKICAgICAgICAgICAgICAgICJ0aGVpciBsb25nIGF4aXMgbGllcyBhdCB7Oi4wZn0gZGVnIHdoZXJlIHRoZSBjYXRhbG9ndWUgc2F5cyB7Oi4wZn0iCiAgICAgICAgICAgICAgICAuZm9ybWF0KGFuZ2xlLCBleHBlY3RlZF9hbmdsZSkpCgogICAgb3V0WyJvayJdID0gbm90IHByb2JsZW1zCiAgICBvdXRbInByb2JsZW1zIl0gPSBwcm9ibGVtcwogICAgcmV0dXJuIG91dAoKZGVmIGRlc2NyaWJlX2NoZWNrKHJlc3VsdCk6CiAgICAiIiJUaGUgc2VsZi1jaGVjaywgaW4gYSBmb3JtIHdvcnRoIHByaW50aW5nLiIiIgogICAgaWYgcmVzdWx0LmdldCgib2siKSBpcyBOb25lOgogICAgICAgIHJldHVybiAiICBzaGFwZSBjaGVjazogIiArIHJlc3VsdC5nZXQoInJlYXNvbiIsICJub3QgZG9uZSIpCiAgICBoZWFkID0gKCIgIHNoYXBlIGNoZWNrOiB0aGUgZGV0ZWN0aW9ucyBhcmUgc2hhcGVkIHs6LjJmfSB3aXRoIHRoZWlyIGxvbmcgYXhpcyAiCiAgICAgICAgICAgICJhdCB7Oi4wZn0gZGVnIi5mb3JtYXQocmVzdWx0WyJtZWFzdXJlZF9yYXRpbyJdLCByZXN1bHRbIm1lYXN1cmVkX2FuZ2xlIl0pKQogICAgaWYgcmVzdWx0LmdldCgiZXhwZWN0ZWRfcmF0aW8iKSBpcyBub3QgTm9uZToKICAgICAgICBoZWFkICs9ICI7IHRoZSBjYXRhbG9ndWUgcHJlZGljdHMgezouMmZ9Ii5mb3JtYXQocmVzdWx0WyJleHBlY3RlZF9yYXRpbyJdKQogICAgICAgIGlmIHJlc3VsdC5nZXQoImV4cGVjdGVkX2FuZ2xlIikgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIGhlYWQgKz0gIiBhdCB7Oi4wZn0gZGVnIi5mb3JtYXQocmVzdWx0WyJleHBlY3RlZF9hbmdsZSJdKQogICAgaWYgcmVzdWx0WyJvayJdOgogICAgICAgIHJldHVybiBoZWFkICsgIlxuICAgIHRoZXkgYWdyZWUgLSB0aGUgZGV0ZWN0aW9ucyBkbyB0cmFjZSB0aGlzIGdhbGF4eSIKICAgIHJldHVybiAoaGVhZCArICJcbiAgICBUSEVZIERJU0FHUkVFOiAiICsgIjsgIi5qb2luKHJlc3VsdFsicHJvYmxlbXMiXSkgKwogICAgICAgICAgICAiXG4gICAgU29tZXRoaW5nIHVwc3RyZWFtIGlzIHdyb25nLiBEZXRlY3Rpb25zIHRoYXQgZG8gbm90IGZvbGxvdyB0aGUgIgogICAgICAgICAgICAiZ2FsYXh5J3NcbiAgICBzaGFwZSBhcmUgdXN1YWxseSBzdWJ0cmFjdGlvbiByZXNpZHVhbHMgc3ByZWFkIG92ZXIgdGhlICIKICAgICAgICAgICAgIndob2xlIGZpZWxkLCBhbmRcbiAgICB0aGUgcmFkaWFsIHByb2ZpbGUgYmVsb3cgd2lsbCBsb29rIHNtb290aCBhbmQgIgogICAgICAgICAgICAibWVhbiBub3RoaW5nLiIpCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIHNoYXJlZCBtb2R1bGUgZ2FsYXh5X2NhdGFsb2d1ZS5weQojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCgpwYXNzICAjIGZyb20gZGVwcm9qZWN0aW9uLCBpbmxpbmVkIGFib3ZlCgpIWVBFUkxFREFfVklaSUVSX0lEID0gIlZJSS8yMzciCgpkZWYgaHlwZXJsZWRhX2dlb21ldHJ5KGdhbGF4eV9uYW1lOiBzdHIsIHNlYXJjaF9yYWRpdXNfYXJjbWluOiBmbG9hdCA9IDAuNSkgLT4gZGljdDoKICAgICIiIkZldGNoIHBvc2l0aW9uIGFuZ2xlLCBheGlzIHJhdGlvIGFuZCBpbmNsaW5hdGlvbiBmb3Igb25lIGdhbGF4eS4KCiAgICBSZXR1cm5zIGEgZGljdCB3aXRoOgogICAgICAgIG5hbWVfcXVlcnksIHBnYywgbXR5cGUgIC0gaWRlbnRpZmljYXRpb24KICAgICAgICBsb2dfcjI1LCBiX292ZXJfYSAgICAgICAtIHRoZSBvYnNlcnZlZCBheGlzIHJhdGlvCiAgICAgICAgcGFfc2t5ICAgICAgICAgICAgICAgICAgLSBwb3NpdGlvbiBhbmdsZSBvbiB0aGUgc2t5LCBvciBOb25lIGlmIG5vdCBwdWJsaXNoZWQKICAgICAgICBpbmNfbmFpdmUgICAgICAgICAgICAgICAtIGFjb3MoYi9hKSwgaWdub3JpbmcgZGlzayB0aGlja25lc3MKICAgICAgICBpbmMgICAgICAgICAgICAgICAgICAgICAtIHRoZSBIdWJibGUtY29ycmVjdGVkIGluY2xpbmF0aW9uIHdlIGFjdHVhbGx5IHVzZQogICAgICAgIHEwICAgICAgICAgICAgICAgICAgICAgIC0gdGhlIHRoaWNrbmVzcyBjb25zdGFudCBjaG9zZW4gZm9yIHRoaXMgdHlwZQogICAgICAgIGQyNV9hcmNtaW4gICAgICAgICAgICAgIC0gaXNvcGhvdGFsIGRpYW1ldGVyLCB1c2VmdWwgZm9yIG5vcm1hbGlzaW5nIHJhZGlpCiAgICAgICAgc291cmNlICAgICAgICAgICAgICAgICAgLSBodW1hbi1yZWFkYWJsZSBwcm92ZW5hbmNlCgogICAgUmFpc2VzIExvb2t1cEVycm9yIHdoZW4gdGhlIG5hbWUgcmVzb2x2ZXMgdG8gbm90aGluZywgc28gdGhlIGNhbGxlciBjYW4gZmFsbAogICAgYmFjayB0byBtYW51YWxseSBzdXBwbGllZCBhbmdsZXMgaW5zdGVhZCBvZiBzaWxlbnRseSBjb250aW51aW5nLgogICAgIiIiCiAgICBmcm9tIGFzdHJvcXVlcnkudml6aWVyIGltcG9ydCBWaXppZXIKICAgIGltcG9ydCBhc3Ryb3B5LnVuaXRzIGFzIHUKCiAgICB2ID0gVml6aWVyKGNhdGFsb2c9SFlQRVJMRURBX1ZJWklFUl9JRCwgcm93X2xpbWl0PS0xKQogICAgcmVzdWx0ID0gdi5xdWVyeV9vYmplY3QoZ2FsYXh5X25hbWUsIHJhZGl1cz1zZWFyY2hfcmFkaXVzX2FyY21pbiAqIHUuYXJjbWluKQogICAgaWYgbm90IHJlc3VsdCBvciBsZW4ocmVzdWx0WzBdKSA9PSAwOgogICAgICAgIHJhaXNlIExvb2t1cEVycm9yKAogICAgICAgICAgICBmIkh5cGVyTEVEQSByZXR1cm5lZCBubyByb3cgZm9yICd7Z2FsYXh5X25hbWV9Jy4gQ2hlY2sgdGhlIG5hbWUsIG9yICIKICAgICAgICAgICAgInN1cHBseSB0aGUgYW5nbGVzIG1hbnVhbGx5LiIKICAgICAgICApCgogICAgcm93ID0gcmVzdWx0WzBdWzBdCgogICAgZGVmIF9nZXQoY29sKToKICAgICAgICB0cnk6CiAgICAgICAgICAgIHZhbCA9IHJvd1tjb2xdCiAgICAgICAgZXhjZXB0IEtleUVycm9yOgogICAgICAgICAgICByZXR1cm4gTm9uZQogICAgICAgIHRyeToKICAgICAgICAgICAgaWYgaGFzYXR0cih2YWwsICJtYXNrIikgYW5kIHZhbC5tYXNrOgogICAgICAgICAgICAgICAgcmV0dXJuIE5vbmUKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBwYXNzCiAgICAgICAgdHJ5OgogICAgICAgICAgICBmID0gZmxvYXQodmFsKQogICAgICAgIGV4Y2VwdCAoVHlwZUVycm9yLCBWYWx1ZUVycm9yKToKICAgICAgICAgICAgcmV0dXJuIE5vbmUKICAgICAgICByZXR1cm4gTm9uZSBpZiBtYXRoLmlzbmFuKGYpIGVsc2UgZgoKICAgIGxvZ19yMjUgPSBfZ2V0KCJsb2dSMjUiKQogICAgaWYgbG9nX3IyNSBpcyBOb25lOgogICAgICAgIHJhaXNlIExvb2t1cEVycm9yKAogICAgICAgICAgICBmIkh5cGVyTEVEQSBoYXMgYSByb3cgZm9yICd7Z2FsYXh5X25hbWV9JyBidXQgbm8gYXhpcyByYXRpbyAobG9nUjI1KSwgIgogICAgICAgICAgICAic28gdGhlIGluY2xpbmF0aW9uIGNhbm5vdCBiZSBkZXJpdmVkLiBTdXBwbHkgdGhlIGFuZ2xlcyBtYW51YWxseS4iCiAgICAgICAgKQoKICAgIGJfb3Zlcl9hID0gMTAuMCAqKiAoLWxvZ19yMjUpCiAgICBtdHlwZSA9IE5vbmUKICAgIHRyeToKICAgICAgICBtdHlwZSA9IHN0cihyb3dbIk1UeXBlIl0pLnN0cmlwKCkgb3IgTm9uZQogICAgZXhjZXB0IEtleUVycm9yOgogICAgICAgIHBhc3MKCiAgICBxMCA9IHEwX2Zvcl90eXBlKG10eXBlKQogICAgbG9nX2QyNSA9IF9nZXQoImxvZ0QyNSIpCgogICAgcmV0dXJuIHsKICAgICAgICAibmFtZV9xdWVyeSI6IGdhbGF4eV9uYW1lLAogICAgICAgICJwZ2MiOiBfZ2V0KCJQR0MiKSwKICAgICAgICAibXR5cGUiOiBtdHlwZSwKICAgICAgICAibG9nX3IyNSI6IGxvZ19yMjUsCiAgICAgICAgImJfb3Zlcl9hIjogYl9vdmVyX2EsCiAgICAgICAgInBhX3NreSI6IF9nZXQoIlBBIiksCiAgICAgICAgImluY19uYWl2ZSI6IG1hdGguZGVncmVlcyhtYXRoLmFjb3MobWluKG1heChiX292ZXJfYSwgMC4wKSwgMS4wKSkpLAogICAgICAgICJpbmMiOiBpbmNsaW5hdGlvbl9mcm9tX2F4aXNfcmF0aW8oYl9vdmVyX2EsIHEwKSwKICAgICAgICAicTAiOiBxMCwKICAgICAgICAiZDI1X2FyY21pbiI6ICgwLjEgKiAxMC4wICoqIGxvZ19kMjUpIGlmIGxvZ19kMjUgaXMgbm90IE5vbmUgZWxzZSBOb25lLAogICAgICAgICJzb3VyY2UiOiBmIkh5cGVyTEVEQSB2aWEgVml6aWVSIHtIWVBFUkxFREFfVklaSUVSX0lEfSIsCiAgICB9CgpkZWYgZGVzY3JpYmUoZ2VvbTogZGljdCkgLT4gc3RyOgogICAgIiIiT25lIHJlYWRhYmxlIGJsb2NrIHN1bW1hcmlzaW5nIHdoYXQgdGhlIGNhdGFsb2d1ZSBnYXZlIHVzLiIiIgogICAgbGluZXMgPSBbCiAgICAgICAgZiIgIGNhdGFsb2d1ZSAgICAgIDoge2dlb21bJ3NvdXJjZSddfSIsCiAgICAgICAgZiIgIFBHQyAgICAgICAgICAgIDoge2dlb21bJ3BnYyddfSIsCiAgICAgICAgZiIgIHR5cGUgICAgICAgICAgIDoge2dlb21bJ210eXBlJ119IiwKICAgICAgICBmIiAgYi9hICAgICAgICAgICAgOiB7Z2VvbVsnYl9vdmVyX2EnXTouM2Z9ICAgKGxvZ1IyNSA9IHtnZW9tWydsb2dfcjI1J106LjNmfSkiLAogICAgICAgIGYiICBpbmNsaW5hdGlvbiAgICA6IHtnZW9tWydpbmMnXTouMWZ9IGRlZyAgICIKICAgICAgICBmIihuYWl2ZSBhY29zKGIvYSkgPSB7Z2VvbVsnaW5jX25haXZlJ106LjFmfSBkZWcsIHEwID0ge2dlb21bJ3EwJ106LjJmfSkiLAogICAgXQogICAgaWYgZ2VvbVsicGFfc2t5Il0gaXMgTm9uZToKICAgICAgICBsaW5lcy5hcHBlbmQoIiAgcG9zaXRpb24gYW5nbGUgOiBOT1QgUFVCTElTSEVEIC0gdGhlIGdhbGF4eSBpcyB0b28gcm91bmQgZm9yIHRoZSAiCiAgICAgICAgICAgICAgICAgICAgICJtYWpvciBheGlzIHRvIGJlIGRlZmluZWQiKQogICAgZWxzZToKICAgICAgICBsaW5lcy5hcHBlbmQoZiIgIHBvc2l0aW9uIGFuZ2xlIDoge2dlb21bJ3BhX3NreSddOi4xZn0gZGVnIG9uIHRoZSBza3kgKE4gdGhyb3VnaCBFKSIpCiAgICBpZiBnZW9tWyJkMjVfYXJjbWluIl06CiAgICAgICAgbGluZXMuYXBwZW5kKGYiICBEMjUgICAgICAgICAgICA6IHtnZW9tWydkMjVfYXJjbWluJ106LjFmfSBhcmNtaW4iKQogICAgcmV0dXJuICJcbiIuam9pbihsaW5lcykKClJfViA9IDMuMSAgICAgICAgICAgICAgICAgICAgICAgICAgIyBBX1YgLyBFKEItVikgZm9yIGR1c3QgaW4gb3VyIG93biBnYWxheHkKCkRJU1RBTkNFX0NBVEFMT0cgPSAiSi9BSi8xNTIvNTAiICAgIyBDb3NtaWNmbG93cy0zLCBUdWxseSBldCBhbC4gMjAxNgoKZGVmIGxvb2t1cF9kaXN0YW5jZShuYW1lKToKICAgICIiIlJlZHNoaWZ0LWluZGVwZW5kZW50IGRpc3RhbmNlIGluIE1wYyBmcm9tIENvc21pY2Zsb3dzLTMsIG9yIE5vbmUuCgogICAgRXZlcnkgbGVuZ3RoIHRoaXMgdG9vbCByZXBvcnRzIGlzIHRoaXMgbnVtYmVyIHRpbWVzIGFuIGFuZ2xlLCBzbyBhIHdyb25nCiAgICBkaXN0YW5jZSByZXNjYWxlcyB0aGUgd2hvbGUgYW5zd2VyIGFuZCBub3RoaW5nIGluIHRoZSBvdXRwdXQgbG9va3MgYW1pc3MuCiAgICBSZWRzaGlmdCBpcyBubyBoZWxwIGZvciBhIGdhbGF4eSB0aGlzIGNsb3NlIC0gdGhlIHJlY2Vzc2lvbiB2ZWxvY2l0eSBvZiBhCiAgICBmZXctTXBjIGdhbGF4eSBpcyBkb21pbmF0ZWQgYnkgaXRzIG93biBtb3Rpb24gdGhyb3VnaCB0aGUgbG9jYWwgZ3JvdXAsIGFuZAogICAgTTEwMSdzIHB1dHMgaXQgYXQgYWJvdXQgaGFsZiBpdHMgdHJ1ZSBkaXN0YW5jZSAtIHNvIGEgY2F0YWxvZ3VlIG9mIGRpcmVjdAogICAgbWVhc3VyZW1lbnRzIGlzIHdoYXQgaXMgd2FudGVkLgoKICAgIEl0IGlzIG9mZmVyZWQsIG5ldmVyIGltcG9zZWQ6IHRoZSBjYWxsZXIgaXMgZXhwZWN0ZWQgdG8gc2hvdyBpdCBhbmQgbGV0IHRoZQogICAgb2JzZXJ2ZXIgb3ZlcnJ1bGUgaXQuCiAgICAiIiIKICAgIHRyeToKICAgICAgICBmcm9tIGFzdHJvcHkuY29vcmRpbmF0ZXMgaW1wb3J0IFNreUNvb3JkCiAgICAgICAgZnJvbSBhc3Ryb3F1ZXJ5LnZpemllciBpbXBvcnQgVml6aWVyCiAgICAgICAgaW1wb3J0IGFzdHJvcHkudW5pdHMgYXMgdQoKICAgICAgICBjb29yZCA9IFNreUNvb3JkLmZyb21fbmFtZShuYW1lKQogICAgICAgICMgIl9yIiBpcyBWaXppZVIncyBvd24gZGlzdGFuY2Ugb2YgZWFjaCByb3cgZnJvbSB0aGUgcG9zaXRpb24gYXNrZWQKICAgICAgICAjIGFib3V0LiBUaGUgcm93cyBjb21lIGJhY2sgaW4gY2F0YWxvZ3VlIG9yZGVyLCBub3QgbmVhcmVzdCBmaXJzdCwgc28KICAgICAgICAjIHdpdGhvdXQgaXQgdGhlIGZpcnN0IHJvdyBjb3VsZCBiZSBhIG5laWdoYm91cmluZyBnYWxheHkgaW5zaWRlIHRoZQogICAgICAgICMgc2VhcmNoIGNpcmNsZSBhbmQgaXRzIGRpc3RhbmNlIHdvdWxkIGJlIHVzZWQgd2l0aG91dCBhIHdvcmQuCiAgICAgICAgaGl0ID0gVml6aWVyKGNvbHVtbnM9WyIqKiIsICJfciJdLCByb3dfbGltaXQ9MjApLnF1ZXJ5X3JlZ2lvbigKICAgICAgICAgICAgY29vcmQsIHJhZGl1cz0yICogdS5hcmNtaW4sIGNhdGFsb2c9RElTVEFOQ0VfQ0FUQUxPRykKICAgICAgICBpZiBub3QgaGl0OgogICAgICAgICAgICByZXR1cm4gTm9uZQogICAgICAgIHRhYmxlID0gaGl0WzBdCiAgICAgICAgaWYgIl9yIiBpbiB0YWJsZS5jb2xuYW1lcyBhbmQgbGVuKHRhYmxlKSA+IDE6CiAgICAgICAgICAgIHRhYmxlID0gdGFibGVbbnAuYXJnc29ydChucC5hc2FycmF5KHRhYmxlWyJfciJdLCBkdHlwZT1mbG9hdCkpXQogICAgICAgIGZvciBjb2wgaW4gKCJEaXN0IiwgIjxEaXN0PiIpOgogICAgICAgICAgICBpZiBjb2wgaW4gdGFibGUuY29sbmFtZXM6CiAgICAgICAgICAgICAgICB2YWx1ZSA9IGZsb2F0KHRhYmxlW2NvbF1bMF0pCiAgICAgICAgICAgICAgICBpZiB2YWx1ZSA9PSB2YWx1ZSBhbmQgdmFsdWUgPiAwOgogICAgICAgICAgICAgICAgICAgIHJldHVybiB2YWx1ZQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICByZXR1cm4gTm9uZQogICAgcmV0dXJuIE5vbmUKCmRlZiBsb29rdXBfZXh0aW5jdGlvbihuYW1lKToKICAgICIiIkdhbGFjdGljIHJlZGRlbmluZyBFKEItVikgdG93YXJkcyB0aGUgZ2FsYXh5LCBvciBOb25lLgoKICAgIFRoZSBTY2hsYWZseSAmIEZpbmtiZWluZXIgKDIwMTEpIHJlY2FsaWJyYXRpb24gb2YgdGhlIFNjaGxlZ2VsIGR1c3QgbWFwcywKICAgIHNlcnZlZCBieSBJUlNBLiBOZWVkZWQgd2hlcmV2ZXIgYSBjb2xvdXIgaXMgdHVybmVkIGludG8gYSBzdGF0ZW1lbnQgYWJvdXQgYQogICAgc3RlbGxhciBwb3B1bGF0aW9uOiBkdXN0IGluIG91ciBvd24gZ2FsYXh5IHJlZGRlbnMgZXZlcnl0aGluZyBiZWhpbmQgaXQsIGFuZAogICAgd2l0aG91dCB0aGUgY29ycmVjdGlvbiBldmVyeSBzb3VyY2UgaXMgc2hpZnRlZCB0b3dhcmRzIHRoZSByZWQsIHdoaWNoIG1vdmVzCiAgICB0aGUgbGluZSBiZXR3ZWVuIGEgeW91bmcgYmx1ZSBjbHVzdGVyIGFuZCBhbiBvcmRpbmFyeSBzdGFyLgoKICAgIFdoZXJlIGl0IGNhbmNlbHMsIGFuZCB0aGlzIGlzIHdvcnRoIGtub3dpbmcgc28gaXQgZG9lcyBub3QgZ2V0ICJmaXhlZCIgbGF0ZXI6CiAgICBmaXR0aW5nIHRoZSBjb250aW51dW0gc3VidHJhY3Rpb24gZmFjdG9yIGFnYWluc3QgYW4gaW5zdHJ1bWVudGFsIGNvbG91cgogICAgbWVhc3VyZWQgZnJvbSB0aGUgc2FtZSBwYWlyIG9mIGZyYW1lcy4gUmVkZGVuaW5nIHNoaWZ0cyBldmVyeSBzdGFyIGJ5IHRoZSBzYW1lCiAgICBhbW91bnQgdGhlcmUsIHRoZSBmaXR0ZWQgbGluZSBzaGlmdHMgd2l0aCB0aGVtLCBhbmQgdGhlIHZhbHVlIHJlYWQgb2ZmIGl0IGZvcgogICAgYW55IGdpdmVuIHBpeGVsIGlzIHVuY2hhbmdlZC4KICAgICIiIgogICAgdHJ5OgogICAgICAgIGZyb20gYXN0cm9xdWVyeS5pcGFjLmlyc2EuaXJzYV9kdXN0IGltcG9ydCBJcnNhRHVzdAogICAgICAgIHRhYmxlID0gSXJzYUR1c3QuZ2V0X3F1ZXJ5X3RhYmxlKG5hbWUsIHNlY3Rpb249ImVidiIpCiAgICAgICAgdmFsdWUgPSBmbG9hdCh0YWJsZVsiZXh0IFNhbmRGIG1lYW4iXVswXSkKICAgICAgICBpZiB2YWx1ZSA9PSB2YWx1ZSBhbmQgdmFsdWUgPj0gMDoKICAgICAgICAgICAgcmV0dXJuIHZhbHVlCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHJldHVybiBOb25lCiAgICByZXR1cm4gTm9uZQoKZGVmIGRlc2NyaWJlX2xvb2t1cHMobmFtZSwgZGlzdGFuY2VfbXBjLCBlYnYsIGRpc3RhbmNlX3NvdXJjZT0iIiwgZWJ2X3NvdXJjZT0iIik6CiAgICAiIiJXaGF0IHdhcyBsb29rZWQgdXAsIHNob3duIGJlZm9yZSBpdCBpcyB1c2VkLiIiIgogICAgbGluZXMgPSBbXQogICAgaWYgZGlzdGFuY2VfbXBjOgogICAgICAgIGxpbmVzLmFwcGVuZCgiICBkaXN0YW5jZSAgICB7Oi4yZn0gTXBjICAge30iLmZvcm1hdCgKICAgICAgICAgICAgZGlzdGFuY2VfbXBjLCBkaXN0YW5jZV9zb3VyY2Ugb3IgImZyb20gQ29zbWljZmxvd3MtMyAoVHVsbHkgZXQgYWwuIDIwMTYpIikpCiAgICAgICAgbGluZXMuYXBwZW5kKCIgICAgICAgICAgICAgIGV2ZXJ5IGxlbmd0aCBiZWxvdyBzY2FsZXMgd2l0aCB0aGlzIG51bWJlciIpCiAgICBlbHNlOgogICAgICAgIGxpbmVzLmFwcGVuZCgiICBkaXN0YW5jZSAgICBub3QgZm91bmQgLSByYWRpaSB3aWxsIGJlIGluIGFyY3NlYywgbm90IHBhcnNlY3MiKQogICAgaWYgZWJ2IGlzIG5vdCBOb25lOgogICAgICAgIGxpbmVzLmFwcGVuZCgiICByZWRkZW5pbmcgICBFKEItVikgPSB7Oi40Zn0sIHNvIEFfViA9IHs6LjNmfSAgIHt9Ii5mb3JtYXQoCiAgICAgICAgICAgIGVidiwgUl9WICogZWJ2LAogICAgICAgICAgICBlYnZfc291cmNlIG9yICJmcm9tIFNjaGxhZmx5ICYgRmlua2JlaW5lciAoMjAxMSkiKSkKICAgIHJldHVybiAiXG4iLmpvaW4obGluZXMpCgpkZWYgZ2FpYV9mb3JlZ3JvdW5kX3N0YXJzKHdjcywgc2hhcGUsIGdfbGltaXQ9MTkuMCwgc2lnbmlmaWNhbmNlPTMuMCwKICAgICAgICAgICAgICAgICAgICAgICAgICBtYXhfcm93cz01MDAwMCk6CiAgICAiIiJQb3NpdGlvbnMsIGluIHBpeGVscywgb2Ygc3RhcnMgdGhhdCBhcmUgaW4gZnJvbnQgb2YgdGhlIGdhbGF4eS4KCiAgICBXSFkgQSBDQVRBTE9HVUUgQU5EIE5PVCBUSEUgSU1BR0UKICAgIEV2ZXJ5dGhpbmcgZG93bnN0cmVhbSB0aGF0IG5lZWRzICJhIHN0YXIiIG5lZWRzIGEgZm9yZWdyb3VuZCBzdGFyOiBzb21ldGhpbmcKICAgIHdob3NlIGxpZ2h0IGlzIHB1cmUgY29udGludXVtLCBzbyB0aGF0IHNjYWxpbmcgb25lIGZpbHRlciBhZ2FpbnN0IGFub3RoZXIgYW5kCiAgICBzdWJ0cmFjdGluZyBtYWtlcyBpdCB2YW5pc2guIERldGVjdGluZyBwb2ludCBzb3VyY2VzIGluIHRoZSBpbWFnZSBmaW5kcyB0aG9zZSwKICAgIGJ1dCBpbiBhIG5lYXJieSBnYWxheHkgaXQgYWxzbyBmaW5kcyB0aGUgZ2FsYXh5J3Mgb3duIHN0YXJzIC0gYW5kIHRob3NlIGFyZSBub3QKICAgIHB1cmUgY29udGludXVtLiBTb21lIGFyZSByZWQgZ2lhbnRzLCBzb21lIGFyZSBob3QgeW91bmcgc3RhcnMgc2l0dGluZyBpbiB0aGUKICAgIGdhcyB0aGF0IGlzIGJlaW5nIG1lYXN1cmVkLgoKICAgIEhvdyBiYWRseSB0aGlzIG1hdHRlcnMgZGVwZW5kcyBvbiBkaXN0YW5jZSwgd2hpY2ggaXMgd2h5IGl0IHdlbnQgdW5ub3RpY2VkIGZvcgogICAgYSBsb25nIHRpbWUuIEF0IGEgZmV3IE1wYyBhIGdhbGF4eSdzIG93biBzdGFycyBhcmUgbm90IHJlc29sdmVkIGFuZCBldmVyeSBwb2ludAogICAgc291cmNlIHJlYWxseSBpcyBmb3JlZ3JvdW5kLiBBdCBNMzMncyAwLjkgTXBjIHRoZXkgYXJlIHJlc29sdmVkIGluIHRoZWlyCiAgICB0aG91c2FuZHM6IG1lYXN1cmVkIGluIHRoaXMgZmllbGQsIG9mIDQyMDkgR2FpYSBzb3VyY2VzIGJyaWdodGVyIHRoYW4gRz0xOSwKICAgIG9ubHkgMTU5MCBhcmUgZm9yZWdyb3VuZC4gU2V0dGluZyB0aGUgc3VidHJhY3Rpb24gZmFjdG9yIGZyb20gdGhlIG1peHR1cmUgZ2F2ZQogICAgYSBzdGFyLXRvLXN0YXIgc2NhdHRlciBvZiA2NiBwZXJjZW50IC0gd2hpY2ggaXMgdG8gc2F5IG5vIGZhY3RvciBhdCBhbGwuCgogICAgVEhFIFRFU1QKICAgIFBhcmFsbGF4IGFuZCBwcm9wZXIgbW90aW9uLiBBIHN0YXIgYSBmZXcgaHVuZHJlZCBwYXJzZWNzIGF3YXkgaW4gb3VyIG93bgogICAgZ2FsYXh5IGhhcyBib3RoLCBtZWFzdXJhYmx5LiBBIHN0YXIgYXQgMC45IE1wYyBoYXMgbmVpdGhlciwgYXQgYW55IHByZWNpc2lvbgogICAgR2FpYSBjYW4gcmVhY2guIFNvIGEgc2lnbmlmaWNhbnQgcGFyYWxsYXgsIG9yIGEgc2lnbmlmaWNhbnQgcHJvcGVyIG1vdGlvbiwgaXMKICAgIGEgcGh5c2ljYWwgc3RhdGVtZW50IHRoYXQgdGhlIHN0YXIgaXMgaW4gZnJvbnQgLSBub3QgYSBicmlnaHRuZXNzIGN1dCBvciBhCiAgICBzaGFwZSBjdXQgdGhhdCBoYXBwZW5zIHRvIGNvcnJlbGF0ZSB3aXRoIGl0LgoKICAgIFJldHVybnMgKHBvc2l0aW9ucywgZGlhZ25vc3RpY3MpLiBQb3NpdGlvbnMgaXMgYW4gKE4sIDIpIGFycmF5IG9mIHgsIHkgaW4gdGhlCiAgICBmcmFtZTsgYW4gZW1wdHkgYXJyYXkgbWVhbnMgdGhlIHF1ZXJ5IGZhaWxlZCBvciBmb3VuZCBub3RoaW5nLCBhbmQgdGhlIGNhbGxlcgogICAgaXMgZXhwZWN0ZWQgdG8gZmFsbCBiYWNrIG9uIGRldGVjdGluZyBwb2ludCBzb3VyY2VzLgogICAgIiIiCiAgICBpbXBvcnQgbnVtcHkgYXMgbnAKCiAgICBkaWFnID0geyJzb3VyY2UiOiAiR2FpYSBEUjMiLCAibl9jYXRhbG9ndWUiOiAwLCAibl9mb3JlZ3JvdW5kIjogMCwKICAgICAgICAgICAgIm5faW5fZnJhbWUiOiAwLCAid2h5IjogIiJ9CiAgICBpZiB3Y3MgaXMgTm9uZToKICAgICAgICBkaWFnWyJ3aHkiXSA9ICJubyBwbGF0ZSBzb2x1dGlvbiwgc28gdGhlIGNhdGFsb2d1ZSBjYW5ub3QgYmUgcGxhY2VkIgogICAgICAgIHJldHVybiBucC5lbXB0eSgoMCwgMikpLCBkaWFnCgogICAgdHJ5OgogICAgICAgIGZyb20gYXN0cm9xdWVyeS5nYWlhIGltcG9ydCBHYWlhCiAgICAgICAgaW1wb3J0IGFzdHJvcHkudW5pdHMgYXMgdQoKICAgICAgICBueSwgbnggPSBzaGFwZVswXSwgc2hhcGVbMV0KICAgICAgICBjZW50cmUgPSB3Y3MucGl4ZWxfdG9fd29ybGQobnggLyAyLjAsIG55IC8gMi4wKQogICAgICAgIGNvcm5lciA9IHdjcy5waXhlbF90b193b3JsZCgwLjAsIDAuMCkKICAgICAgICByYWRpdXMgPSBmbG9hdChjZW50cmUuc2VwYXJhdGlvbihjb3JuZXIpLnRvKHUuZGVnKS52YWx1ZSkgKiAxLjA1CgogICAgICAgIHF1ZXJ5ID0gKAogICAgICAgICAgICAiU0VMRUNUIHJhLCBkZWMsIHBob3RfZ19tZWFuX21hZyBGUk9NIGdhaWFkcjMuZ2FpYV9zb3VyY2UgIgogICAgICAgICAgICAiV0hFUkUgMT1DT05UQUlOUyhQT0lOVChyYSwgZGVjKSwgQ0lSQ0xFKHs6LjZmfSwgezouNmZ9LCB7Oi40Zn0pKSAiCiAgICAgICAgICAgICJBTkQgcGhvdF9nX21lYW5fbWFnIDwgezouMmZ9ICIKICAgICAgICAgICAgIkFORCAocGFyYWxsYXgvcGFyYWxsYXhfZXJyb3IgPiB7Oi4xZn0gT1IgIgogICAgICAgICAgICAiICAgICBzcXJ0KHBtcmEqcG1yYSArIHBtZGVjKnBtZGVjKSAvICIKICAgICAgICAgICAgIiAgICAgc3FydChwbXJhX2Vycm9yKnBtcmFfZXJyb3IgKyBwbWRlY19lcnJvcipwbWRlY19lcnJvcikgPiB7Oi4xZn0pICIKICAgICAgICAgICAgIk9SREVSIEJZIHBob3RfZ19tZWFuX21hZyIKICAgICAgICApLmZvcm1hdChjZW50cmUucmEuZGVnLCBjZW50cmUuZGVjLmRlZywgcmFkaXVzLCBnX2xpbWl0LAogICAgICAgICAgICAgICAgIHNpZ25pZmljYW5jZSwgc2lnbmlmaWNhbmNlKQoKICAgICAgICBqb2IgPSBHYWlhLmxhdW5jaF9qb2JfYXN5bmMocXVlcnkpIGlmIG1heF9yb3dzID4gMjAwMCBlbHNlIEdhaWEubGF1bmNoX2pvYihxdWVyeSkKICAgICAgICB0YWJsZSA9IGpvYi5nZXRfcmVzdWx0cygpCiAgICAgICAgZGlhZ1sibl9mb3JlZ3JvdW5kIl0gPSBsZW4odGFibGUpCiAgICAgICAgaWYgbm90IGxlbih0YWJsZSk6CiAgICAgICAgICAgIGRpYWdbIndoeSJdID0gIm5vIGZvcmVncm91bmQgc3RhcnMgaW4gdGhlIGZpZWxkIgogICAgICAgICAgICByZXR1cm4gbnAuZW1wdHkoKDAsIDIpKSwgZGlhZwoKICAgICAgICB4LCB5ID0gd2NzLndvcmxkX3RvX3BpeGVsX3ZhbHVlcyhucC5hc2FycmF5KHRhYmxlWyJyYSJdLCBkdHlwZT1mbG9hdCksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbnAuYXNhcnJheSh0YWJsZVsiZGVjIl0sIGR0eXBlPWZsb2F0KSkKICAgICAgICBpbnNpZGUgPSAoeCA+IDUpICYgKHggPCBueCAtIDYpICYgKHkgPiA1KSAmICh5IDwgbnkgLSA2KQogICAgICAgIGRpYWdbIm5faW5fZnJhbWUiXSA9IGludChpbnNpZGUuc3VtKCkpCiAgICAgICAgIyBCcmlnaHRlc3QgZmlyc3QuIFdob2V2ZXIgdGFrZXMgInRoZSBmaXJzdCB0d28gaHVuZHJlZCIgb2YgdGhpcyBsaXN0IG11c3QKICAgICAgICAjIGdldCB0aGUgdHdvIGh1bmRyZWQgd29ydGggbWVhc3VyaW5nOiBhIHN0YXIgbmVhciB0aGUgY2F0YWxvZ3VlIGxpbWl0IGlzCiAgICAgICAgIyBhIGZldyBjb3VudHMgdGhyb3VnaCBhIG5hcnJvdyBmaWx0ZXIsIGFuZCBhIHJhdGlvIGJ1aWx0IG9uIGl0IGlzIG5vaXNlLgogICAgICAgICMgSGFuZGluZyBiYWNrIGFuIHVub3JkZXJlZCBsaXN0IGNvc3QgYSBmYWN0b3Igb2YgdHdvIGluIHRoZSBzY2F0dGVyIG9mIHRoZQogICAgICAgICMgc3VidHJhY3Rpb24gZmFjdG9yIGJlZm9yZSB0aGlzIGxpbmUgZXhpc3RlZC4KICAgICAgICBkaWFnWyJnX3JhbmdlIl0gPSAoZmxvYXQobnAubmFubWluKHRhYmxlWyJwaG90X2dfbWVhbl9tYWciXSkpLAogICAgICAgICAgICAgICAgICAgICAgICAgICBmbG9hdChucC5uYW5tYXgodGFibGVbInBob3RfZ19tZWFuX21hZyJdKSkpCiAgICAgICAgcmV0dXJuIG5wLmNvbHVtbl9zdGFjayhbeFtpbnNpZGVdLCB5W2luc2lkZV1dKSwgZGlhZwoKICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOgogICAgICAgIGRpYWdbIndoeSJdID0gInt9OiB7fSIuZm9ybWF0KHR5cGUoZXhjKS5fX25hbWVfXywgZXhjKQogICAgICAgIHJldHVybiBucC5lbXB0eSgoMCwgMikpLCBkaWFnCgpkZWYgZGVzY3JpYmVfZm9yZWdyb3VuZChkaWFnKToKICAgICIiIldoYXQgdGhlIGNhdGFsb2d1ZSBsb29rdXAgZm91bmQsIGluIGEgbGluZS4iIiIKICAgIGlmIGRpYWcuZ2V0KCJuX2luX2ZyYW1lIik6CiAgICAgICAgcmV0dXJuICgiICB7fSBmb3JlZ3JvdW5kIHN0YXJzIGZyb20ge30gKHBhcmFsbGF4IG9yIHByb3BlciBtb3Rpb24gIgogICAgICAgICAgICAgICAgInNpZ25pZmljYW50KTsgdGhlIGdhbGF4eSdzIG93biBzdGFycyBhcmUgbm90IGFtb25nIHRoZW0iLmZvcm1hdCgKICAgICAgICAgICAgICAgICAgICBkaWFnWyJuX2luX2ZyYW1lIl0sIGRpYWdbInNvdXJjZSJdKSkKICAgIHJldHVybiAoIiAgbm8gY2F0YWxvZ3VlIHN0YXJzIGF2YWlsYWJsZSAoe30pIC0gZmFsbGluZyBiYWNrIG9uIHBvaW50IHNvdXJjZXMgIgogICAgICAgICAgICAiZm91bmQgaW4gdGhlIGltYWdlLCB3aGljaCBpbiBhIG5lYXJieSBnYWxheHkgaW5jbHVkZXMgaXRzIG93biAiCiAgICAgICAgICAgICJzdGFycyIuZm9ybWF0KGRpYWcuZ2V0KCJ3aHkiKSBvciAidW5rbm93biIpKQoKZGVzY3JpYmVfX2dhbGF4eV9jYXRhbG9ndWUgPSBkZXNjcmliZSAgICMgdGhpcyBtb2R1bGUncyBvd247IHRoZSBwbGFpbiBuYW1lIGdldHMgc2hhZG93ZWQgYmVsb3cKCgojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCiMgc2hhcmVkIG1vZHVsZSBjb250aW51dW1fc3VidHJhY3Rpb24ucHkKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQoKZGVmIF9hcGVydHVyZV9mbHV4ZXMoZnJhbWVzLCBwb3NpdGlvbnMsIHI9Ni4wLCByX2luPTEwLjAsIHJfb3V0PTE2LjApOgogICAgIiIiQmFja2dyb3VuZC1zdWJ0cmFjdGVkIGFwZXJ0dXJlIGZsdXggYXQgdGhlIHNhbWUgcG9zaXRpb25zIGluIHNldmVyYWwgZnJhbWVzLiIiIgogICAgZnJvbSBwaG90dXRpbHMuYXBlcnR1cmUgaW1wb3J0IChDaXJjdWxhckFwZXJ0dXJlLCBDaXJjdWxhckFubnVsdXMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFwZXJ0dXJlX3Bob3RvbWV0cnkpCgogICAgYXAgPSBDaXJjdWxhckFwZXJ0dXJlKHBvc2l0aW9ucywgcj1yKQogICAgYW5uID0gQ2lyY3VsYXJBbm51bHVzKHBvc2l0aW9ucywgcl9pbj1yX2luLCByX291dD1yX291dCkKICAgIG91dCA9IFtdCiAgICBmb3IgaW1nIGluIGZyYW1lczoKICAgICAgICBwaG90ID0gYXBlcnR1cmVfcGhvdG9tZXRyeShpbWcsIFthcCwgYW5uXSkKICAgICAgICBia2dfbWVhbiA9IHBob3RbImFwZXJ0dXJlX3N1bV8xIl0gLyBhbm4uYXJlYQogICAgICAgIG91dC5hcHBlbmQobnAuYXNhcnJheShwaG90WyJhcGVydHVyZV9zdW1fMCJdIC0gYmtnX21lYW4gKiBhcC5hcmVhLCBkdHlwZT1mbG9hdCkpCiAgICByZXR1cm4gb3V0CgpkZWYgZmluZF9yZWZlcmVuY2Vfc3RhcnMoZnJhbWUsIG1hc2s9Tm9uZSwgbl9zdGFycz0zMDAsIHRocmVzaG9sZF9zaWdtYT0yNS4wKToKICAgICIiIkJyaWdodCwgaXNvbGF0ZWQgcG9pbnQgc291cmNlcyB0byBjYWxpYnJhdGUgYWdhaW5zdC4iIiIKICAgIGZyb20gcGhvdHV0aWxzLmRldGVjdGlvbiBpbXBvcnQgREFPU3RhckZpbmRlcgogICAgZnJvbSBhc3Ryb3B5LnN0YXRzIGltcG9ydCBzaWdtYV9jbGlwcGVkX3N0YXRzCgogICAgbWVhbiwgbWVkaWFuLCBzdGQgPSBzaWdtYV9jbGlwcGVkX3N0YXRzKGZyYW1lLCBzaWdtYT0zLjAsIG1hc2s9bWFzaykKICAgIGZpbmRlciA9IERBT1N0YXJGaW5kZXIoZndobT00LjAsIHRocmVzaG9sZD10aHJlc2hvbGRfc2lnbWEgKiBzdGQpCiAgICB0YmwgPSBmaW5kZXIoZnJhbWUgLSBtZWRpYW4sIG1hc2s9bWFzaykKICAgIGlmIHRibCBpcyBOb25lIG9yIGxlbih0YmwpID09IDA6CiAgICAgICAgcmV0dXJuIG5wLmVtcHR5KCgwLCAyKSkKICAgIHRibC5zb3J0KCJmbHV4IikKICAgIHRibCA9IHRibFs6Oi0xXVs6bl9zdGFyc10KICAgIHJldHVybiBucC50cmFuc3Bvc2UoKG5wLmFzYXJyYXkodGJsWyJ4Y2VudHJvaWQiXSwgZHR5cGU9ZmxvYXQpLAogICAgICAgICAgICAgICAgICAgICAgICAgbnAuYXNhcnJheSh0YmxbInljZW50cm9pZCJdLCBkdHlwZT1mbG9hdCkpKQoKZGVmIGNvbnN0YW50X2ZhY3RvcihoYSwgcmVkLCBtYXNrPU5vbmUsIG5fc3RhcnM9MjAwLCBwb3NpdGlvbnM9Tm9uZSk6CiAgICAiIiJPbmUgZmFjdG9yIGZvciB0aGUgd2hvbGUgZnJhbWUsIHRha2VuIGZyb20gdGhlIHN0YXJzLgoKICAgIFJldHVybnMgdGhlIGZhY3RvciB0byBtdWx0aXBseSBILWFscGhhIGJ5IGJlZm9yZSBzdWJ0cmFjdGluZyBSLCBhbmQgdGhlCiAgICBzdGFyLXRvLXN0YXIgc2NhdHRlciwgd2hpY2ggaXMgdGhlIGhvbmVzdCB1bmNlcnRhaW50eSBvbiBpdC4KICAgICIiIgogICAgcG9zID0gKG5wLmFzYXJyYXkocG9zaXRpb25zLCBkdHlwZT1mbG9hdClbOm5fc3RhcnNdCiAgICAgICAgICAgaWYgcG9zaXRpb25zIGlzIG5vdCBOb25lIGFuZCBsZW4ocG9zaXRpb25zKSA+PSAxMAogICAgICAgICAgIGVsc2UgZmluZF9yZWZlcmVuY2Vfc3RhcnMocmVkLCBtYXNrPW1hc2ssIG5fc3RhcnM9bl9zdGFycykpCiAgICBpZiBsZW4ocG9zKSA8IDU6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCJ0b28gZmV3IHN0YXJzIGZvdW5kIHRvIHNldCB0aGUgY29udGludXVtIGZhY3RvciIpCgogICAgZl9oYSwgZl9yID0gX2FwZXJ0dXJlX2ZsdXhlcyhbaGEsIHJlZF0sIHBvcykKICAgIG9rID0gbnAuaXNmaW5pdGUoZl9oYSkgJiBucC5pc2Zpbml0ZShmX3IpICYgKGZfaGEgPiAwKQogICAgaWYgb2suc3VtKCkgPCA1OgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigibm8gdXNhYmxlIHN0YXJzIGZvciB0aGUgY29udGludXVtIGZhY3RvciIpCgogICAgcmF0aW8gPSBmX3Jbb2tdIC8gZl9oYVtva10KICAgIG1lZCA9IGZsb2F0KG5wLm1lZGlhbihyYXRpbykpCiAgICBzcHJlYWQgPSAxLjQ4MjYgKiBucC5tZWRpYW4obnAuYWJzKHJhdGlvIC0gbWVkKSkKICAgIGlmIHNwcmVhZCA+IDA6CiAgICAgICAgcmF0aW8gPSByYXRpb1tucC5hYnMocmF0aW8gLSBtZWQpIDwgMy4wICogc3ByZWFkXQoKICAgIGZhY3RvciA9IGZsb2F0KG5wLm1lZGlhbihyYXRpbykpCiAgICBzY2F0dGVyID0gZmxvYXQoMS40ODI2ICogbnAubWVkaWFuKG5wLmFicyhyYXRpbyAtIGZhY3RvcikpKQogICAgcmV0dXJuIGZhY3RvciwgewogICAgICAgICJuX3N0YXJzIjogaW50KHJhdGlvLnNpemUpLAogICAgICAgICJzY2F0dGVyIjogc2NhdHRlciwKICAgICAgICAicmVsYXRpdmVfc2NhdHRlciI6IHNjYXR0ZXIgLyBmYWN0b3IgaWYgZmFjdG9yIGVsc2UgZmxvYXQoIm5hbiIpLAogICAgICAgICJtZXRob2QiOiAib25lIGNvbnN0YW50IGZvciB0aGUgd2hvbGUgZnJhbWUiLAogICAgfQoKZGVmIGNvbG91cl9tYXAoYiwgdiwgc21vb3RoX3B4PTYuMCwgbWluX3NpZ25hbD1Ob25lKToKICAgICIiIkFuIGluc3RydW1lbnRhbCBCIG1pbnVzIFYgdmFsdWUgZm9yIGV2ZXJ5IHBpeGVsLCBzbW9vdGhlZCB0byBiZSB1c2FibGUuCgogICAgQ29sb3VyIGlzIGEgcmF0aW8gb2YgdHdvIGZhaW50IHRoaW5ncywgc28gaXQgaXMgZmFyIG5vaXNpZXIgdGhhbiBlaXRoZXIgZnJhbWUKICAgIG9uIGl0cyBvd24uIEJvdGggZnJhbWVzIGFyZSBzbW9vdGhlZCBiZWZvcmUgdGhlIHJhdGlvIGlzIHRha2VuIC0gdGhlIGNvbG91ciBvZgogICAgYSBzdGFyLWZvcm1pbmcgY29tcGxleCB2YXJpZXMgb24gc2NhbGVzIG11Y2ggbGFyZ2VyIHRoYW4gYSBwaXhlbCwgc28gdGhpcyBjb3N0cwogICAgbm90aGluZyByZWFsIGFuZCBtYWtlcyB0aGUgbWFwIHVzYWJsZS4KCiAgICBSZXR1cm5zIHRoZSBjb2xvdXIgbWFwIGFuZCBhIGJvb2xlYW4gYXJyYXkgbWFya2luZyB3aGVyZSBpdCBjYW4gYmUgdHJ1c3RlZC4KICAgICIiIgogICAgZnJvbSBzY2lweS5uZGltYWdlIGltcG9ydCBnYXVzc2lhbl9maWx0ZXIKICAgIGZyb20gYXN0cm9weS5zdGF0cyBpbXBvcnQgc2lnbWFfY2xpcHBlZF9zdGF0cwoKICAgIGJzID0gZ2F1c3NpYW5fZmlsdGVyKG5wLm5hbl90b19udW0oYiwgbmFuPTAuMCksIHNtb290aF9weCkKICAgIHZzID0gZ2F1c3NpYW5fZmlsdGVyKG5wLm5hbl90b19udW0odiwgbmFuPTAuMCksIHNtb290aF9weCkKCiAgICBpZiBtaW5fc2lnbmFsIGlzIE5vbmU6CiAgICAgICAgXywgXywgc3RkX3YgPSBzaWdtYV9jbGlwcGVkX3N0YXRzKHYsIHNpZ21hPTMuMCkKICAgICAgICBtaW5fc2lnbmFsID0gMy4wICogc3RkX3YgLyBtYXgoc21vb3RoX3B4LCAxLjApCgogICAgdXNhYmxlID0gKGJzID4gbWluX3NpZ25hbCkgJiAodnMgPiBtaW5fc2lnbmFsKSAmIG5wLmlzZmluaXRlKGJzKSAmIG5wLmlzZmluaXRlKHZzKQogICAgd2l0aCBucC5lcnJzdGF0ZShkaXZpZGU9Imlnbm9yZSIsIGludmFsaWQ9Imlnbm9yZSIpOgogICAgICAgIGNvbG91ciA9IG5wLndoZXJlKHVzYWJsZSwgLTIuNSAqIG5wLmxvZzEwKG5wLndoZXJlKHVzYWJsZSwgYnMgLyB2cywgMS4wKSksIG5wLm5hbikKICAgIHJldHVybiBjb2xvdXIsIHVzYWJsZQoKZGVmIGNvbG91cl9kZXBlbmRlbnRfZmFjdG9yKGhhLCByZWQsIGIsIHYsIG1hc2s9Tm9uZSwgbl9zdGFycz0zMDAsIHNtb290aF9weD02LjAsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBwb3NpdGlvbnM9Tm9uZSk6CiAgICAiIiJBIGZhY3RvciB0aGF0IHZhcmllcyBhY3Jvc3MgdGhlIGZyYW1lIHdpdGggdGhlIGNvbG91ciBvZiB3aGF0IGlzIHRoZXJlLgoKICAgIEZvbGxvd3MgdGhlIHJlY29tbWVuZGF0aW9uIG9mIFNwZWN0b3IsIEZpbmtlbG1hbiBhbmQgQnJvc2NoICgyMDEyKTogbWVhc3VyZSB0aGUKICAgIGZhY3RvciBzdGFyIGJ5IHN0YXIsIG1lYXN1cmUgZWFjaCBzdGFyJ3MgY29sb3VyLCBmaXQgYSBzdHJhaWdodCBsaW5lIHRocm91Z2gKICAgIHRoZW0sIGFuZCB0aGVuIHJlYWQgdGhlIGZhY3RvciBvZmYgdGhhdCBsaW5lIGZvciBldmVyeSBwYXJ0IG9mIHRoZSBpbWFnZQogICAgYWNjb3JkaW5nIHRvIGl0cyBvd24gY29sb3VyLgoKICAgIFdIRVJFIElUIEZBTExTIEJBQ0sKICAgIFRoZSBjb2xvdXIgb2YgYSBwaXhlbCBpcyBvbmx5IG1lYXN1cmFibGUgd2hlcmUgYm90aCBicm9hZCBiYW5kcyBoYXZlIHNpZ25hbC4KICAgIEluIHRoZSBlbXB0eSBza3kgYmV0d2VlbiB0aGUgYXJtcyB0aGV5IGRvIG5vdCwgYW5kIHRoZXJlIHRoZSBjb25zdGFudCBmYWN0b3IgaXMKICAgIHVzZWQgaW5zdGVhZC4gVGhlIGRpYWdub3N0aWNzIHJlcG9ydCB3aGF0IGZyYWN0aW9uIG9mIHRoZSBmcmFtZSBnb3Qgd2hpY2gsIHNvIGEKICAgIHJ1biB3aGVyZSB0aGUgY29sb3VyIG1hcCBiYXJlbHkgYXBwbGllZCBkb2VzIG5vdCBsb29rIGxpa2Ugb25lIHdoZXJlIGl0IGRpZC4KCiAgICBSZXR1cm5zIChmYWN0b3JfbWFwLCBkaWFnbm9zdGljcykuIGZhY3Rvcl9tYXAgaGFzIHRoZSBzYW1lIHNoYXBlIGFzIHRoZSBmcmFtZXMuCiAgICAiIiIKICAgICMgQ2F0YWxvZ3VlIHBvc2l0aW9ucyB3aGVuIHRoZXkgYXJlIGF2YWlsYWJsZTogdGhlIHJlbGF0aW9uIGJlaW5nIGZpdHRlZCBpcwogICAgIyB0aGUgb25lIGZvcmVncm91bmQgc3RhcnMgZm9sbG93LCBhbmQgYSBuZWFyYnkgZ2FsYXh5J3Mgb3duIHN0YXJzIHNpdHRpbmcgaW4KICAgICMgdGhlIHNhbXBsZSBhcmUgd2hhdCBtYWtlcyB0aGUgZml0IGZhaWwgcmF0aGVyIHRoYW4gbWVyZWx5IHNjYXR0ZXIuCiAgICBwb3MgPSAobnAuYXNhcnJheShwb3NpdGlvbnMsIGR0eXBlPWZsb2F0KVs6bl9zdGFyc10KICAgICAgICAgICBpZiBwb3NpdGlvbnMgaXMgbm90IE5vbmUgYW5kIGxlbihwb3NpdGlvbnMpID49IDEwCiAgICAgICAgICAgZWxzZSBmaW5kX3JlZmVyZW5jZV9zdGFycyh2LCBtYXNrPW1hc2ssIG5fc3RhcnM9bl9zdGFycykpCiAgICBpZiBsZW4ocG9zKSA8IDg6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCJ0b28gZmV3IHN0YXJzIHRvIGZpdCBhIGNvbG91ciByZWxhdGlvbiIpCgogICAgZl9oYSwgZl9yLCBmX2IsIGZfdiA9IF9hcGVydHVyZV9mbHV4ZXMoW2hhLCByZWQsIGIsIHZdLCBwb3MpCiAgICBvayA9IChucC5pc2Zpbml0ZShmX2hhKSAmIG5wLmlzZmluaXRlKGZfcikgJiBucC5pc2Zpbml0ZShmX2IpICYgbnAuaXNmaW5pdGUoZl92KQogICAgICAgICAgJiAoZl9oYSA+IDApICYgKGZfYiA+IDApICYgKGZfdiA+IDApKQogICAgaWYgb2suc3VtKCkgPCA4OgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigidG9vIGZldyBzdGFycyB3aXRoIGdvb2QgZmx1eCBpbiBhbGwgZm91ciBmcmFtZXMiKQoKICAgIHJhdGlvID0gZl9yW29rXSAvIGZfaGFbb2tdCiAgICBjb2xvdXIgPSAtMi41ICogbnAubG9nMTAoZl9iW29rXSAvIGZfdltva10pCgogICAgIyBDbGlwIG9idmlvdXMgb3V0bGllcnMgaW4gYm90aCBkaXJlY3Rpb25zIGJlZm9yZSBmaXR0aW5nOiBzYXR1cmF0ZWQgc3RhcnMsIGFuZAogICAgIyBhbnkgc3RhciB0aGF0IGRvZXMgaGF2ZSBsaW5lIGVtaXNzaW9uIGFuZCBzbyBpcyBub3QgcHVyZSBjb250aW51dW0uCiAgICBnb29kID0gbnAuaXNmaW5pdGUocmF0aW8pICYgbnAuaXNmaW5pdGUoY29sb3VyKQogICAgZm9yIF8gaW4gcmFuZ2UoMyk6CiAgICAgICAgaWYgZ29vZC5zdW0oKSA8IDY6CiAgICAgICAgICAgIGJyZWFrCiAgICAgICAgbWVkX3IgPSBucC5tZWRpYW4ocmF0aW9bZ29vZF0pCiAgICAgICAgc3ByZWFkID0gMS40ODI2ICogbnAubWVkaWFuKG5wLmFicyhyYXRpb1tnb29kXSAtIG1lZF9yKSkKICAgICAgICBpZiBzcHJlYWQgPD0gMDoKICAgICAgICAgICAgYnJlYWsKICAgICAgICBnb29kICY9IG5wLmFicyhyYXRpbyAtIG1lZF9yKSA8IDMuMCAqIHNwcmVhZAoKICAgIGlmIGdvb2Quc3VtKCkgPCA2OgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigibm90IGVub3VnaCBjbGVhbiBzdGFycyBsZWZ0IHRvIGZpdCB0aGUgY29sb3VyIHJlbGF0aW9uIikKCiAgICBzbG9wZSwgaW50ZXJjZXB0ID0gbnAucG9seWZpdChjb2xvdXJbZ29vZF0sIHJhdGlvW2dvb2RdLCAxKQogICAgcHJlZGljdGVkID0gbnAucG9seXZhbChbc2xvcGUsIGludGVyY2VwdF0sIGNvbG91cltnb29kXSkKICAgIHJlc2lkID0gcmF0aW9bZ29vZF0gLSBwcmVkaWN0ZWQKICAgIHNzX3RvdCA9IGZsb2F0KG5wLnN1bSgocmF0aW9bZ29vZF0gLSByYXRpb1tnb29kXS5tZWFuKCkpICoqIDIpKQogICAgcjIgPSBmbG9hdCgxLjAgLSBucC5zdW0ocmVzaWQgKiogMikgLyBzc190b3QpIGlmIHNzX3RvdCA+IDAgZWxzZSBmbG9hdCgibmFuIikKCiAgICBjb25zdCA9IGZsb2F0KG5wLm1lZGlhbihyYXRpb1tnb29kXSkpCiAgICBjb25zdF9zY2F0dGVyID0gZmxvYXQoMS40ODI2ICogbnAubWVkaWFuKG5wLmFicyhyYXRpb1tnb29kXSAtIGNvbnN0KSkpCiAgICBmaXRfc2NhdHRlciA9IGZsb2F0KDEuNDgyNiAqIG5wLm1lZGlhbihucC5hYnMocmVzaWQpKSkKCiAgICBjX2xvLCBjX2hpID0gbnAucGVyY2VudGlsZShjb2xvdXJbZ29vZF0sIFsyLCA5OF0pCgogICAgIyBPbmx5IHVzZSB0aGUgZml0dGVkIHJlbGF0aW9uIGlmIGl0IGFjdHVhbGx5IGRlc2NyaWJlcyB0aGUgc3RhcnMgYmV0dGVyIHRoYW4gYQogICAgIyBzaW5nbGUgbnVtYmVyIGRvZXMuIEZpdHRpbmcgYSBzbG9wZSB0aHJvdWdoIGEgaGFuZGZ1bCBvZiBzdGFycyBzcGFubmluZyBhCiAgICAjIG5hcnJvdyByYW5nZSBvZiBjb2xvdXIgY2FuIGVhc2lseSBjb21lIG91dCB3b3JzZSB0aGFuIHRoZSBjb25zdGFudCBpdCB3YXMKICAgICMgbWVhbnQgdG8gaW1wcm92ZSBvbiwgYW5kIGEgbWV0aG9kIHRoYXQgaXMgYmV0dGVyIGluIHByaW5jaXBsZSBidXQgbWVhc3VyYWJseQogICAgIyB3b3JzZSBvbiB0aGlzIGRhdGEgc2hvdWxkIG5vdCBiZSB0aGUgb25lIHRoYXQgcnVucy4KICAgIGltcHJvdmVzID0gKGZpdF9zY2F0dGVyIDwgY29uc3Rfc2NhdHRlcikgYW5kIChyMiA+PSAwLjMwKSBhbmQgKGdvb2Quc3VtKCkgPj0gMTIpCiAgICByZWFzb24gPSAiIgogICAgaWYgbm90IGltcHJvdmVzOgogICAgICAgIGJpdHMgPSBbXQogICAgICAgIGlmIGdvb2Quc3VtKCkgPCAxMjoKICAgICAgICAgICAgYml0cy5hcHBlbmQoIm9ubHkge30gY2xlYW4gc3RhcnMiLmZvcm1hdChpbnQoZ29vZC5zdW0oKSkpKQogICAgICAgIGlmIHIyIDwgMC4zMDoKICAgICAgICAgICAgYml0cy5hcHBlbmQoInRoZSBmaXQgZXhwbGFpbnMgbGl0dGxlIG9mIHRoZSBzY2F0dGVyIChSMiA9IHs6LjJmfSkiLmZvcm1hdChyMikpCiAgICAgICAgaWYgZml0X3NjYXR0ZXIgPj0gY29uc3Rfc2NhdHRlcjoKICAgICAgICAgICAgYml0cy5hcHBlbmQoInRoZSBmaXR0ZWQgbGluZSBzY2F0dGVycyBtb3JlIHRoYW4gYSBjb25zdGFudCAiCiAgICAgICAgICAgICAgICAgICAgICAgICIoezouNGZ9IGFnYWluc3QgezouNGZ9KSIuZm9ybWF0KGZpdF9zY2F0dGVyLCBjb25zdF9zY2F0dGVyKSkKICAgICAgICBpZiBjX2hpIC0gY19sbyA8IDAuNToKICAgICAgICAgICAgYml0cy5hcHBlbmQoInRoZSBzdGFycyBzcGFuIG9ubHkgezouMmZ9IG1hZyBvZiBjb2xvdXIiLmZvcm1hdChjX2hpIC0gY19sbykpCiAgICAgICAgcmVhc29uID0gIjsgIi5qb2luKGJpdHMpCgogICAgY21hcCwgdXNhYmxlID0gY29sb3VyX21hcChiLCB2LCBzbW9vdGhfcHg9c21vb3RoX3B4KQogICAgY19jbGlwcGVkID0gbnAuY2xpcChjbWFwLCBjX2xvLCBjX2hpKQoKICAgIGZhY3Rvcl9tYXAgPSBucC5mdWxsKGhhLnNoYXBlLCBjb25zdCwgZHR5cGU9ZmxvYXQpCiAgICBhcHBsaWVkID0gbnAuemVyb3MoaGEuc2hhcGUsIGR0eXBlPWJvb2wpCiAgICBpZiBpbXByb3ZlczoKICAgICAgICAjIE5ldmVyIGV4dHJhcG9sYXRlIHRoZSByZWxhdGlvbiBiZXlvbmQgdGhlIGNvbG91cnMgdGhlIHN0YXJzIGFjdHVhbGx5CiAgICAgICAgIyBzcGFubmVkOyBvdXRzaWRlIHRoYXQgcmFuZ2UgdGhlIHN0cmFpZ2h0IGxpbmUgaXMgYSBndWVzcywgbm90IGEgbWVhc3VyZW1lbnQuCiAgICAgICAgYXBwbGllZCA9IHVzYWJsZSAmIG5wLmlzZmluaXRlKGNfY2xpcHBlZCkKICAgICAgICBmYWN0b3JfbWFwW2FwcGxpZWRdID0gbnAucG9seXZhbChbc2xvcGUsIGludGVyY2VwdF0sIGNfY2xpcHBlZFthcHBsaWVkXSkKCiAgICByZXR1cm4gZmFjdG9yX21hcCwgewogICAgICAgICJtZXRob2QiOiAoImZhY3RvciBmaXR0ZWQgYWdhaW5zdCBjb2xvdXIsIHBlciBTcGVjdG9yIGV0IGFsLiAyMDEyIgogICAgICAgICAgICAgICAgICAgaWYgaW1wcm92ZXMgZWxzZQogICAgICAgICAgICAgICAgICAgIm9uZSBjb25zdGFudCAtIHRoZSBjb2xvdXIgZml0IHdhcyB0cmllZCBhbmQgcmVqZWN0ZWQiKSwKICAgICAgICAidXNlZF9jb2xvdXIiOiBib29sKGltcHJvdmVzKSwKICAgICAgICAicmVqZWN0ZWRfYmVjYXVzZSI6IHJlYXNvbiwKICAgICAgICAibl9zdGFycyI6IGludChnb29kLnN1bSgpKSwKICAgICAgICAic2xvcGUiOiBmbG9hdChzbG9wZSksCiAgICAgICAgImludGVyY2VwdCI6IGZsb2F0KGludGVyY2VwdCksCiAgICAgICAgImZpdF9yMiI6IHIyLAogICAgICAgICJjb2xvdXJfcmFuZ2UiOiAoZmxvYXQoY19sbyksIGZsb2F0KGNfaGkpKSwKICAgICAgICAiY29uc3RhbnRfZmFjdG9yIjogY29uc3QsCiAgICAgICAgImNvbnN0YW50X3NjYXR0ZXIiOiBjb25zdF9zY2F0dGVyLAogICAgICAgICJyZWxhdGl2ZV9zY2F0dGVyIjogY29uc3Rfc2NhdHRlciAvIGNvbnN0IGlmIGNvbnN0IGVsc2UgZmxvYXQoIm5hbiIpLAogICAgICAgICJmaXRfc2NhdHRlciI6IGZpdF9zY2F0dGVyLAogICAgICAgICJzY2F0dGVyX3JlZHVjdGlvbiI6IChjb25zdF9zY2F0dGVyIC8gZml0X3NjYXR0ZXIpIGlmIGZpdF9zY2F0dGVyID4gMCBlbHNlIGZsb2F0KCJuYW4iKSwKICAgICAgICAiZnJhY3Rpb25fY29sb3VyX2FwcGxpZWQiOiBmbG9hdChhcHBsaWVkLm1lYW4oKSksCiAgICAgICAgImZhY3Rvcl9taW4iOiBmbG9hdChucC5uYW5taW4oZmFjdG9yX21hcCkpLAogICAgICAgICJmYWN0b3JfbWF4IjogZmxvYXQobnAubmFubWF4KGZhY3Rvcl9tYXApKSwKICAgIH0KCmRlZiBkZXNjcmliZShkaWFnKToKICAgICIiIkEgcmVhZGFibGUgc3VtbWFyeSBvZiB3aGljaGV2ZXIgbWV0aG9kIHdhcyB1c2VkLiIiIgogICAgbGluZXMgPSBbIiAgbWV0aG9kOiAiICsgZGlhZy5nZXQoIm1ldGhvZCIsICI/IiksCiAgICAgICAgICAgICAiICBzdGFycyB1c2VkOiB7fSIuZm9ybWF0KGRpYWcuZ2V0KCJuX3N0YXJzIikpXQogICAgaWYgZGlhZy5nZXQoInJlamVjdGVkX2JlY2F1c2UiKToKICAgICAgICBsaW5lcy5hcHBlbmQoIiAgdGhlIGNvbG91ciBmaXQgd2FzIHJlamVjdGVkIGJlY2F1c2UgIiArIGRpYWdbInJlamVjdGVkX2JlY2F1c2UiXSkKICAgICAgICBsaW5lcy5hcHBlbmQoIiAgZmFsbGluZyBiYWNrIHRvIGEgY29uc3RhbnQgZmFjdG9yIG9mIHs6LjNmfSIuZm9ybWF0KAogICAgICAgICAgICBkaWFnWyJjb25zdGFudF9mYWN0b3IiXSkpCiAgICBpZiAic2xvcGUiIGluIGRpYWc6CiAgICAgICAgbGluZXMgKz0gWwogICAgICAgICAgICAiICBmYWN0b3IgPSB7Oi40Zn0gKyB7Oi40Zn0geCAoQi1WIGluc3RydW1lbnRhbCksICBSMiA9IHs6LjJmfSIuZm9ybWF0KAogICAgICAgICAgICAgICAgZGlhZ1siaW50ZXJjZXB0Il0sIGRpYWdbInNsb3BlIl0sIGRpYWdbImZpdF9yMiJdKSwKICAgICAgICAgICAgIiAgY29sb3VycyBzcGFubmVkIGJ5IHRoZSBzdGFyczogezouMmZ9IHRvIHs6LjJmfSIuZm9ybWF0KCpkaWFnWyJjb2xvdXJfcmFuZ2UiXSksCiAgICAgICAgICAgICIgIGZhY3RvciB2YXJpZXMgYWNyb3NzIHRoZSBmcmFtZSBmcm9tIHs6LjNmfSB0byB7Oi4zZn0iLmZvcm1hdCgKICAgICAgICAgICAgICAgIGRpYWdbImZhY3Rvcl9taW4iXSwgZGlhZ1siZmFjdG9yX21heCJdKSwKICAgICAgICAgICAgIiAgYSBzaW5nbGUgY29uc3RhbnQgd291bGQgaGF2ZSBiZWVuIHs6LjNmfSIuZm9ybWF0KGRpYWdbImNvbnN0YW50X2ZhY3RvciJdKSwKICAgICAgICAgICAgIiAgc2NhdHRlciBhYm91dCBhIGNvbnN0YW50IHs6LjRmfSwgYWJvdXQgdGhlIGZpdHRlZCBsaW5lIHs6LjRmfSAiCiAgICAgICAgICAgICItIHRoZSBsaW5lIGlzIHs6LjAlfSB7fSIuZm9ybWF0KAogICAgICAgICAgICAgICAgZGlhZ1siY29uc3RhbnRfc2NhdHRlciJdLCBkaWFnWyJmaXRfc2NhdHRlciJdLAogICAgICAgICAgICAgICAgYWJzKGRpYWdbImZpdF9zY2F0dGVyIl0gLyBkaWFnWyJjb25zdGFudF9zY2F0dGVyIl0gLSAxLjApCiAgICAgICAgICAgICAgICBpZiBkaWFnWyJjb25zdGFudF9zY2F0dGVyIl0gZWxzZSBmbG9hdCgibmFuIiksCiAgICAgICAgICAgICAgICAidGlnaHRlciIgaWYgZGlhZ1siZml0X3NjYXR0ZXIiXSA8IGRpYWdbImNvbnN0YW50X3NjYXR0ZXIiXSBlbHNlICJ3aWRlciIpLAogICAgICAgICAgICAiICB0aGUgY29sb3VyIHJlbGF0aW9uIHdhcyBhcHBsaWVkIHRvIHs6LjAlfSBvZiB0aGUgZnJhbWU7IHRoZSByZXN0IGZlbGwgIgogICAgICAgICAgICAiYmFjayB0byB0aGUgY29uc3RhbnQiLmZvcm1hdChkaWFnWyJmcmFjdGlvbl9jb2xvdXJfYXBwbGllZCJdKSwKICAgICAgICBdCiAgICBlbHNlOgogICAgICAgIGxpbmVzLmFwcGVuZCgiICBmYWN0b3IgezouNGZ9LCBzdGFyLXRvLXN0YXIgc2NhdHRlciB7Oi4xJX0iLmZvcm1hdCgKICAgICAgICAgICAgZGlhZy5nZXQoImNvbnN0YW50X2ZhY3RvciIsIGZsb2F0KCJuYW4iKSksCiAgICAgICAgICAgIGRpYWcuZ2V0KCJyZWxhdGl2ZV9zY2F0dGVyIiwgZmxvYXQoIm5hbiIpKSkpCiAgICByZXR1cm4gIlxuIi5qb2luKGxpbmVzKQoKZGVzY3JpYmVfX2NvbnRpbnV1bV9zdWJ0cmFjdGlvbiA9IGRlc2NyaWJlICAgIyB0aGlzIG1vZHVsZSdzIG93bjsgdGhlIHBsYWluIG5hbWUgZ2V0cyBzaGFkb3dlZCBiZWxvdwoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KIyBzaGFyZWQgbW9kdWxlIHJhZGlhbF9wcm9maWxlcy5weQojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCgptYXRwbG90bGliLnVzZSgiQWdnIikKCnBhc3MgICMgZnJvbSBkZXByb2plY3Rpb24sIGlubGluZWQgYWJvdmUKCkNPTE9VUlMgPSBbIiM0NDQ0NDQiLCAiI2EzMjgzYyIsICIjMmI1YTg5IiwgIiMzZjdkNTIiLCAiIzhhNWEyYiJdCgpkZWYgdHJlYXRtZW50X25vbmUoKToKICAgIHJldHVybiB7ImtleSI6ICJub25lIiwgImxhYmVsIjogIkFzIG1lYXN1cmVkLCBubyB0aWx0IGNvcnJlY3Rpb24iLCAiYW5nbGUiOiBOb25lLCAiaW5jIjogTm9uZSwKICAgICAgICAgICAgIm5vdGUiOiAiUG9zaXRpb25zIGV4YWN0bHkgYXMgbWVhc3VyZWQgb24gdGhlIGltYWdlLiBDb250cm9sIHBhbmVsLiJ9CgpkZWYgdHJlYXRtZW50X2NhdGFsb2d1ZShnZW9tLCBub3J0aF9hbmdsZT05MC4wLCBtaXJyb3JlZD1GYWxzZSk6CiAgICAiIiJEZXByb2plY3Rpb24gdXNpbmcgdGhlIHBvc2l0aW9uIGFuZ2xlIGFuZCBheGlzIHJhdGlvIGZyb20gSHlwZXJMRURBLiIiIgogICAgaWYgZ2VvbSBpcyBOb25lOgogICAgICAgIHJldHVybiBOb25lCiAgICBwYV9za3ksIGluYyA9IGdlb20uZ2V0KCJwYV9za3kiKSwgZ2VvbS5nZXQoImluYyIpCiAgICBpZiBwYV9za3kgaXMgTm9uZToKICAgICAgICByZXR1cm4geyJrZXkiOiAibGVkYSIsICJsYWJlbCI6ICJUaWx0IGNvcnJlY3Rpb24gbm90IHBvc3NpYmxlIiwgImFuZ2xlIjogTm9uZSwgImluYyI6IE5vbmUsCiAgICAgICAgICAgICAgICAibm90ZSI6ICJIeXBlckxFREEgcHVibGlzaGVzIG5vIHBvc2l0aW9uIGFuZ2xlIGZvciB0aGlzIGdhbGF4eSAtIGl0IGlzICIKICAgICAgICAgICAgICAgICAgICAgICAgInRvbyByb3VuZCBmb3IgYSBtYWpvciBheGlzIHRvIGJlIGRlZmluZWQuIE5vdCBjb3JyZWN0ZWQuIn0KICAgIG9rLCByZWFzb24gPSBpbmNsaW5hdGlvbl9pc191c2FibGUoaW5jKQogICAgaWYgbm90IG9rOgogICAgICAgIHJldHVybiB7ImtleSI6ICJsZWRhIiwgImxhYmVsIjogIlRpbHQgY29ycmVjdGlvbiBub3QgcG9zc2libGUiLCAiYW5nbGUiOiBOb25lLAogICAgICAgICAgICAgICAgImluYyI6IE5vbmUsICJub3RlIjogcmVhc29ufQogICAgYW5nID0gc2t5X3BhX3RvX2ltYWdlX2FuZ2xlKHBhX3NreSwgbm9ydGhfYW5nbGUsIG1pcnJvcmVkKQogICAgcmV0dXJuIHsKICAgICAgICAia2V5IjogImxlZGEiLCAibGFiZWwiOiAiQ29ycmVjdGVkIGZvciBhIHRpbHQgb2YgezouMGZ9IGRlZyAoSHlwZXJMRURBKSIuZm9ybWF0KGluYyksCiAgICAgICAgImFuZ2xlIjogYW5nLCAiaW5jIjogaW5jLAogICAgICAgICJub3RlIjogKCJQQSB7Oi4xZn0gZGVnIG9uIHNreSAtPiB7Oi4xZn0gZGVnIGluIGltYWdlIChub3J0aCBhdCB7Oi4wZn0gZGVne30pLiAiCiAgICAgICAgICAgICAgICAgIlN0cmV0Y2ggezouMmZ9eCBhbG9uZyB0aGUgbWlub3IgYXhpcy4iKS5mb3JtYXQoCiAgICAgICAgICAgICAgICAgICAgIHBhX3NreSwgYW5nLCBub3J0aF9hbmdsZSwgIiwgbWlycm9yZWQiIGlmIG1pcnJvcmVkIGVsc2UgIiIsCiAgICAgICAgICAgICAgICAgICAgIHN0cmV0Y2hfZmFjdG9yKGluYykpLAogICAgfQoKZGVmIHRyZWF0bWVudF9tYW51YWwoaW5jLCBwYV9pbWFnZT1Ob25lLCBwYV9za3k9Tm9uZSwgbm9ydGhfYW5nbGU9OTAuMCwgbWlycm9yZWQ9RmFsc2UpOgogICAgIiIiRGVwcm9qZWN0aW9uIHVzaW5nIGFuZ2xlcyB0aGUgdXNlciBzdXBwbGllZCwgZnJvbSB0aGUgbGl0ZXJhdHVyZSBvciBieSBleWUuIiIiCiAgICBpZiBpbmMgaXMgTm9uZToKICAgICAgICByZXR1cm4gTm9uZQogICAgb2ssIHJlYXNvbiA9IGluY2xpbmF0aW9uX2lzX3VzYWJsZShpbmMpCiAgICBpZiBwYV9pbWFnZSBpcyBub3QgTm9uZToKICAgICAgICBhbmcsIHNyYyA9IHBhX2ltYWdlICUgMTgwLjAsICJnaXZlbiBkaXJlY3RseSBpbiBpbWFnZSBjb29yZGluYXRlcyIKICAgIGVsaWYgcGFfc2t5IGlzIG5vdCBOb25lOgogICAgICAgIGFuZyA9IHNreV9wYV90b19pbWFnZV9hbmdsZShwYV9za3ksIG5vcnRoX2FuZ2xlLCBtaXJyb3JlZCkKICAgICAgICBzcmMgPSAiZnJvbSBza3kgUEEgezouMWZ9IGRlZyIuZm9ybWF0KHBhX3NreSkKICAgIGVsc2U6CiAgICAgICAgcmV0dXJuIHsia2V5IjogIm1hbnVhbCIsICJsYWJlbCI6ICJUaWx0IGNvcnJlY3Rpb24gaW5jb21wbGV0ZSIsICJhbmdsZSI6IE5vbmUsCiAgICAgICAgICAgICAgICAiaW5jIjogTm9uZSwgIm5vdGUiOiAiQW4gaW5jbGluYXRpb24gd2FzIGdpdmVuIGJ1dCBubyBwb3NpdGlvbiBhbmdsZS4ifQogICAgaWYgbm90IG9rOgogICAgICAgIHJldHVybiB7ImtleSI6ICJtYW51YWwiLCAibGFiZWwiOiAiVGlsdCBjb3JyZWN0aW9uIG5vdCBwb3NzaWJsZSIsICJhbmdsZSI6IE5vbmUsCiAgICAgICAgICAgICAgICAiaW5jIjogTm9uZSwgIm5vdGUiOiByZWFzb259CiAgICByZXR1cm4geyJrZXkiOiAibWFudWFsIiwgImxhYmVsIjogIkNvcnJlY3RlZCBmb3IgYSB0aWx0IG9mIHs6LjBmfSBkZWcgKHlvdXJzKSIuZm9ybWF0KGluYyksCiAgICAgICAgICAgICJhbmdsZSI6IGFuZywgImluYyI6IGluYywKICAgICAgICAgICAgIm5vdGUiOiAoIlVzZXItc3VwcGxpZWQgZ2VvbWV0cnksIHt9LiBNYWpvciBheGlzIGF0IHs6LjFmfSBkZWcgaW4gdGhlICIKICAgICAgICAgICAgICAgICAgICAgImltYWdlLiBTdHJldGNoIHs6LjJmfXguIikuZm9ybWF0KHNyYywgYW5nLCBzdHJldGNoX2ZhY3RvcihpbmMpKX0KCmRlZiB0cmVhdG1lbnRfc2VjdG9yKGR4LCBkeSwgbl9zZWN0b3JzPTIwLCBzdGVwPTUuMCk6CiAgICAiIiJEZXByb2plY3Rpb24gZGVyaXZlZCBmcm9tIHRoZSBvYmplY3RzIHRoZW1zZWx2ZXMsIHdpdGggaXRzIGNhdmVhdHMgYXR0YWNoZWQuIiIiCiAgICBmaXQgPSBmaXRfc2VjdG9yX21ldGhvZChkeCwgZHksIG5fc2VjdG9ycz1uX3NlY3RvcnMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBhbmdsZV9zdGVwPXN0ZXAsIGluY19zdGVwPXN0ZXApCiAgICBvaywgcmVhc29uID0gaW5jbGluYXRpb25faXNfdXNhYmxlKGZpdFsiaW5jIl0pCgogICAgd2FybiA9ICIiCiAgICBpZiBmaXRbImZld19vYmplY3RzIl06CiAgICAgICAgd2FybiArPSAoIiBPbmx5IHt9IG9iamVjdHMgLSBiZWxvdyB0aGUgMTUwIHRvIDIwMCBuZWVkZWQgZm9yIGEgc2luZ2xlIGNsZWFuICIKICAgICAgICAgICAgICAgICAibWluaW11bSwgc28gdHJlYXQgdGhpcyB3aXRoIHN1c3BpY2lvbi4iLmZvcm1hdChmaXRbIm5fb2JqZWN0cyJdKSkKICAgIGlmIGZpdFsicnVubmVyc191cCJdOgogICAgICAgIGEsIGMsIGQgPSBmaXRbInJ1bm5lcnNfdXAiXVswXQogICAgICAgIG1hcmdpbiA9IChkIC0gZml0WyJkaXNwZXJzaW9uIl0pIC8gbWF4KGZpdFsiZGlzcGVyc2lvbiJdLCAxZS05KQogICAgICAgIHdhcm4gKz0gKCIgQ2xvc2VzdCByaXZhbCBzb2x1dGlvbjogUEEgezouMGZ9IGRlZywgaSB7Oi4wZn0gZGVnLCB3b3JzZSBieSBvbmx5ICIKICAgICAgICAgICAgICAgICAiezouMCV9LiIuZm9ybWF0KGEsIGMsIG1hcmdpbikpCgogICAgaWYgbm90IG9rOgogICAgICAgIHJldHVybiB7ImtleSI6ICJzZWN0b3IiLCAibGFiZWwiOiAiVGlsdCBmcm9tIHRoZSBkYXRhIC0gbm90IHBvc3NpYmxlIiwgImFuZ2xlIjogTm9uZSwKICAgICAgICAgICAgICAgICJpbmMiOiBOb25lLAogICAgICAgICAgICAgICAgIm5vdGUiOiAoIlNlY3RvciBtZXRob2QgZm91bmQgUEEgezouMGZ9IGRlZywgaSB7Oi4wZn0gZGVnLCBidXQgIgogICAgICAgICAgICAgICAgICAgICAgICAgLmZvcm1hdChmaXRbImFuZ2xlIl0sIGZpdFsiaW5jIl0pICsgcmVhc29uICsgd2Fybil9CiAgICByZXR1cm4geyJrZXkiOiAic2VjdG9yIiwgImxhYmVsIjogIkNvcnJlY3RlZCBmb3IgYSB0aWx0IG9mIHs6LjBmfSBkZWcgKG1lYXN1cmVkIGhlcmUpIi5mb3JtYXQoZml0WyJpbmMiXSksCiAgICAgICAgICAgICJhbmdsZSI6IGZpdFsiYW5nbGUiXSwgImluYyI6IGZpdFsiaW5jIl0sCiAgICAgICAgICAgICJub3RlIjogKCJBbmdsZXMgZGVyaXZlZCBmcm9tIHRoZSBvYmplY3RzIHRoZW1zZWx2ZXMgYnkgZXZlbmluZyBvdXQgdGhlICIKICAgICAgICAgICAgICAgICAgICAgImNvdW50cyBpbiB7fSB3ZWRnZXM6IFBBIHs6LjBmfSBkZWcgaW4gaW1hZ2UsIGkgezouMGZ9IGRlZywgIgogICAgICAgICAgICAgICAgICAgICAic3RyZXRjaCB7Oi4yZn14LiIpLmZvcm1hdChuX3NlY3RvcnMsIGZpdFsiYW5nbGUiXSwgZml0WyJpbmMiXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgc3RyZXRjaF9mYWN0b3IoZml0WyJpbmMiXSkpICsgd2Fybn0KCmRlZiBjb21wdXRlX3Byb2ZpbGVzKHgsIHksIGNlbnRyZSwgZnJhbWUsIHRyZWF0bWVudHMsIHJpbmdzPTIwKToKICAgICIiIkRlcHJvamVjdCBhbmQgYmluIHRoZSBvYmplY3RzIG9uY2UgcGVyIHRyZWF0bWVudC4iIiIKICAgIGR4MCwgZHkwID0gbnAuYXNhcnJheSh4LCBmbG9hdCkgLSBjZW50cmVbMF0sIG5wLmFzYXJyYXkoeSwgZmxvYXQpIC0gY2VudHJlWzFdCiAgICByZXN1bHRzID0gW10KICAgIGZvciB0IGluIHRyZWF0bWVudHM6CiAgICAgICAgaWYgdFsiYW5nbGUiXSBpcyBOb25lOgogICAgICAgICAgICBkeHAsIGR5cCA9IGR4MCwgZHkwCiAgICAgICAgZWxzZToKICAgICAgICAgICAgZHhwLCBkeXAgPSBkZXByb2plY3QoZHgwLCBkeTAsIHRbImFuZ2xlIl0sIHRbImluYyJdKQogICAgICAgIHJlcyA9IHJhZGlhbF9kZW5zaXR5KGR4cCwgZHlwLCBuX3JpbmdzPXJpbmdzLCBmcmFtZV9jb3JuZXJzPWZyYW1lLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNlbnRyZV94eT1jZW50cmUsIG1ham9yX2F4aXNfYW5nbGVfZGVnPXRbImFuZ2xlIl0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgaW5jX2RlZz10WyJpbmMiXSkKICAgICAgICByZXNbImR4Il0sIHJlc1siZHkiXSA9IGR4cCwgZHlwCiAgICAgICAgcmVzdWx0cy5hcHBlbmQocmVzKQogICAgcmV0dXJuIHJlc3VsdHMKCmRlZiBnZW9tZXRyaWNfYXJlYXMoZWRnZXMpOgogICAgIiIiVGhlIGZ1bGwgbWF0aGVtYXRpY2FsIHJpbmcgYXJlYXMsIGZvciBjb21wYXJpc29uIHdpdGggdGhlIG9ic2VydmVkIG9uZXMuIiIiCiAgICByZXR1cm4gbWF0aC5waSAqIChucC5hc2FycmF5KGVkZ2VzKVsxOl0gKiogMiAtIG5wLmFzYXJyYXkoZWRnZXMpWzotMV0gKiogMikKCmRlZiB3cml0ZV9wcm9maWxlc19jc3YocGF0aCwgdHJlYXRtZW50cywgcmVzdWx0cywgc2NhbGU9MS4wLCB1bml0PSJweCIpOgogICAgIiIiU2F2ZSBldmVyeSBwcm9maWxlLCBpbmNsdWRpbmcgYm90aCB0aGUgb2JzZXJ2ZWQgYW5kIHRoZSBmdWxsIHJpbmcgYXJlYXMuCgogICAgS2VlcGluZyB0aGUgZnVsbCBnZW9tZXRyaWMgYXJlYSBhbG9uZ3NpZGUgdGhlIG9ic2VydmVkIG9uZSBsZXRzIGFueW9uZSBjb21wYXJlCiAgICBhZ2FpbnN0IG9sZGVyIGFuYWx5c2VzLCB3aGljaCBkaXZpZGVkIGJ5IHRoZSBmdWxsIGFyZWEgYW5kIHNvIHVuZGVyLXJlcG9ydGVkIHRoZQogICAgZGVuc2l0eSBvZiBhbnkgcmluZyB0aGUgZnJhbWUgaGFkIGNsaXBwZWQuCiAgICAiIiIKICAgIHdpdGggb3BlbihwYXRoLCAidyIsIGVuY29kaW5nPSJ1dGYtOCIpIGFzIGZoOgogICAgICAgIGZoLndyaXRlKCJ0cmVhdG1lbnQsbWFqb3JfYXhpc19hbmdsZV9kZWcsaW5jbGluYXRpb25fZGVnLCIKICAgICAgICAgICAgICAgICAicl9pbm5lcl97dX0scl9vdXRlcl97dX0scl9jZW50cmVfe3V9LG5fb2JqZWN0cywiCiAgICAgICAgICAgICAgICAgIm9ic2VydmVkX2FyZWFfe3V9MixmdWxsX3JpbmdfYXJlYV97dX0yLGFyZWFfZnJhY3Rpb24sIgogICAgICAgICAgICAgICAgICJkZW5zaXR5X3Blcl97dX0yLGRlbnNpdHlfZXJyLHJpbmdfY29tcGxldGVcbiIuZm9ybWF0KHU9dW5pdCkpCiAgICAgICAgZm9yIHQsIHJlcyBpbiB6aXAodHJlYXRtZW50cywgcmVzdWx0cyk6CiAgICAgICAgICAgIGdlbyA9IGdlb21ldHJpY19hcmVhcyhyZXNbImVkZ2VzIl0pCiAgICAgICAgICAgIGZvciBrIGluIHJhbmdlKGxlbihyZXNbInIiXSkpOgogICAgICAgICAgICAgICAgZmgud3JpdGUoInt9LHt9LHt9LHs6LjRmfSx7Oi40Zn0sezouNGZ9LHt9LHs6LjRmfSx7Oi40Zn0sezouNGZ9LCIKICAgICAgICAgICAgICAgICAgICAgICAgICJ7Oi42ZX0sezouNmV9LHt9XG4iLmZvcm1hdCgKICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0WyJsYWJlbCJdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICIiIGlmIHRbImFuZ2xlIl0gaXMgTm9uZSBlbHNlICJ7Oi4xZn0iLmZvcm1hdCh0WyJhbmdsZSJdKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAiIiBpZiB0WyJpbmMiXSBpcyBOb25lIGVsc2UgIns6LjFmfSIuZm9ybWF0KHRbImluYyJdKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXNbImVkZ2VzIl1ba10gKiBzY2FsZSwgcmVzWyJlZGdlcyJdW2sgKyAxXSAqIHNjYWxlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJlc1siciJdW2tdICogc2NhbGUsIGludChyZXNbImNvdW50cyJdW2tdKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXNbImFyZWEiXVtrXSAqIHNjYWxlICoqIDIsIGdlb1trXSAqIHNjYWxlICoqIDIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVzWyJhcmVhX2ZyYWN0aW9uIl1ba10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVzWyJkZW5zaXR5Il1ba10gLyBzY2FsZSAqKiAyLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJlc1siZGVuc2l0eV9lcnIiXVtrXSAvIHNjYWxlICoqIDIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgaW50KHJlc1siY29tcGxldGUiXVtrXSkpKQoKZGVmIHN0ZXBfcHJvZmlsZShheCwgZWRnZXMsIHZhbHVlcywgY29sb3VyLCBsdz0xLjcsIGFscGhhPTEuMCwgbHM9Ii0iLCBsYWJlbD1Ob25lKToKICAgICIiIkRyYXcgYSBkZW5zaXR5IHByb2ZpbGUgYXMgc3RlcHMsIGZsYXQgYWNyb3NzIGVhY2ggcmluZy4KCiAgICBBIHN1cmZhY2UgZGVuc2l0eSBpcyBub3QgYSBtZWFzdXJlbWVudCBhdCBhIHBvaW50LiBJdCBpcyBvbmUgbnVtYmVyIGZvciBhIHdob2xlCiAgICByaW5nIC0gdGhlIGNvdW50IGluIGl0IGRpdmlkZWQgYnkgaXRzIGFyZWEgLSBhbmQgaXQgc2F5cyBub3RoaW5nIGFib3V0IHdoZXJlCiAgICBpbnNpZGUgdGhhdCByaW5nIHRoZSBvYmplY3RzIHNhdC4gRHJhd2luZyBpdCBhcyBhIGRvdCBhdCB0aGUgcmluZydzIGNlbnRyZQogICAgam9pbmVkIHRvIHRoZSBuZXh0IGRvdCBieSBhIHNsb3BpbmcgbGluZSBpbnZpdGVzIGV4YWN0bHkgdGhlIHJlYWRpbmcgdGhlIGRhdGEKICAgIGRvZXMgbm90IHN1cHBvcnQ6IHRoYXQgdGhlIGRlbnNpdHkgdmFyaWVzIHNtb290aGx5IGFuZCB0aGF0IHRoZSB2YWx1ZSBhdCB0aGUKICAgIGNlbnRyZSBvZiB0aGUgcmluZyBpcyBzb21laG93IGJldHRlciBtZWFzdXJlZCB0aGFuIGF0IGl0cyBlZGdlLgoKICAgIFN0ZXBzIHNheSB3aGF0IHdhcyBhY3R1YWxseSBtZWFzdXJlZCwgYW5kIGl0IGlzIGhvdyB0aGUgdG9vbCBpbiB0aGUgcmVwb3NpdG9yeQogICAgaGFzIGFsd2F5cyBkcmF3biB0aGlzLCBzbyB0aGUgdHdvIGNhbiBiZSBsYWlkIHNpZGUgYnkgc2lkZS4KICAgICIiIgogICAgeSA9IG5wLmFwcGVuZCh2YWx1ZXMsIHZhbHVlc1stMV0pCiAgICBheC5zdGVwKGVkZ2VzLCB5LCB3aGVyZT0icG9zdCIsIGNvbG9yPWNvbG91ciwgbHc9bHcsIGFscGhhPWFscGhhLCBscz1scywKICAgICAgICAgICAgbGFiZWw9bGFiZWwpCgpkZWYgcHJvZmlsZV9maWd1cmUodHJlYXRtZW50cywgcmVzdWx0cywgdW5pdF9sYWJlbCwgdGl0bGUsIHNjYWxlPTEuMCk6CiAgICAiIiJPbmUgcm93IHBlciB0cmVhdG1lbnQgLSBzY2F0dGVyIGFuZCBwcm9maWxlIC0gdGhlbiBhbGwgcHJvZmlsZXMgb3ZlcmxhaWQuIiIiCiAgICBuID0gbGVuKHRyZWF0bWVudHMpCiAgICBmaWcgPSBwbHQuZmlndXJlKGZpZ3NpemU9KDExLjAsIDQuMSAqIG4gKyAzLjQpKQogICAgZ3MgPSBmaWcuYWRkX2dyaWRzcGVjKG4gKyAxLCAyLCBoZWlnaHRfcmF0aW9zPVsxXSAqIG4gKyBbMS4yNV0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgaHNwYWNlPTAuODUsIHdzcGFjZT0wLjMwKQoKICAgIGZvciBpLCAodCwgcmVzKSBpbiBlbnVtZXJhdGUoemlwKHRyZWF0bWVudHMsIHJlc3VsdHMpKToKICAgICAgICBjb2wgPSBDT0xPVVJTW2kgJSBsZW4oQ09MT1VSUyldCiAgICAgICAgZHh1LCBkeXUgPSByZXNbImR4Il0gKiBzY2FsZSwgcmVzWyJkeSJdICogc2NhbGUKICAgICAgICBydSA9IHJlc1siciJdICogc2NhbGUKICAgICAgICBkdSA9IHJlc1siZGVuc2l0eSJdIC8gc2NhbGUgKiogMgogICAgICAgIGV1ID0gcmVzWyJkZW5zaXR5X2VyciJdIC8gc2NhbGUgKiogMgoKICAgICAgICBheCA9IGZpZy5hZGRfc3VicGxvdChnc1tpLCAwXSkKICAgICAgICBheC5zY2F0dGVyKGR4dSwgZHl1LCBzPTcsIGM9Y29sLCBhbHBoYT0wLjY1LCBsaW5ld2lkdGhzPTApCiAgICAgICAgYXgucGxvdCgwLCAwLCAiKyIsIGM9ImsiLCBtcz0xMSwgbWV3PTEuNikKICAgICAgICBsaW0gPSAxLjA2ICogbWF4KG5wLmFicyhkeHUpLm1heCgpLCBucC5hYnMoZHl1KS5tYXgoKSkKICAgICAgICBheC5zZXRfeGxpbSgtbGltLCBsaW0pOyBheC5zZXRfeWxpbSgtbGltLCBsaW0pCiAgICAgICAgYXguc2V0X2FzcGVjdCgiZXF1YWwiLCBhZGp1c3RhYmxlPSJib3giKQogICAgICAgIGF4LnNldF90aXRsZSh0WyJsYWJlbCJdLCBmb250c2l6ZT0xMC41LCBsb2M9ImxlZnQiKQogICAgICAgIGF4LnNldF94bGFiZWwoInggZnJvbSBjZW50cmUgWyIgKyB1bml0X2xhYmVsICsgIl0iLCBmb250c2l6ZT04LjUpCiAgICAgICAgYXguc2V0X3lsYWJlbCgieSBmcm9tIGNlbnRyZSBbIiArIHVuaXRfbGFiZWwgKyAiXSIsIGZvbnRzaXplPTguNSkKICAgICAgICBheC50aWNrX3BhcmFtcyhsYWJlbHNpemU9Ny41KQogICAgICAgIGF4LmdyaWQoYWxwaGE9MC4xNSwgbHc9MC42KQoKICAgICAgICBheDIgPSBmaWcuYWRkX3N1YnBsb3QoZ3NbaSwgMV0pCiAgICAgICAgZyA9IHJlc1siY29tcGxldGUiXQogICAgICAgIGVkZ2VzID0gcmVzWyJlZGdlcyJdICogc2NhbGUKICAgICAgICAjIFRoZSBzdGVwIGlzIHRoZSBtZWFzdXJlbWVudC4gVGhlIGJhcnMgb24gdG9wIG9mIGl0IGFyZSB0aGUgY291bnRpbmcKICAgICAgICAjIHVuY2VydGFpbnR5LCBkcmF3biBhdCB0aGUgbWlkZGxlIG9mIGVhY2ggcmluZyBvbmx5IGJlY2F1c2UgdGhleSBoYXZlIHRvCiAgICAgICAgIyBiZSBkcmF3biBzb21ld2hlcmUuCiAgICAgICAgc3RlcF9wcm9maWxlKGF4MiwgZWRnZXMsIG5wLndoZXJlKGcsIGR1LCBucC5uYW4pLCBjb2wpCiAgICAgICAgaWYgZy5hbnkoKToKICAgICAgICAgICAgYXgyLmVycm9yYmFyKHJ1W2ddLCBkdVtnXSwgeWVycj1ldVtnXSwgZm10PSJub25lIiwgZWNvbG9yPWNvbCwKICAgICAgICAgICAgICAgICAgICAgICAgIGVsaW5ld2lkdGg9MS4wLCBjYXBzaXplPTIsIGFscGhhPTAuNzUpCiAgICAgICAgaWYgKH5nKS5hbnkoKToKICAgICAgICAgICAgc3RlcF9wcm9maWxlKGF4MiwgZWRnZXMsIG5wLndoZXJlKH5nLCBkdSwgbnAubmFuKSwgY29sLAogICAgICAgICAgICAgICAgICAgICAgICAgbHc9MS4yLCBhbHBoYT0wLjQ1LCBscz0iLS0iLAogICAgICAgICAgICAgICAgICAgICAgICAgbGFiZWw9InJpbmcgY2xpcHBlZCBieSB0aGUgZnJhbWUiKQogICAgICAgICAgICBheDIubGVnZW5kKGZvbnRzaXplPTcsIGZyYW1lb249RmFsc2UsIGxvYz0ibG93ZXIgbGVmdCIpCiAgICAgICAgYXgyLnNldF95c2NhbGUoImxvZyIpCiAgICAgICAgYXgyLnNldF94bGFiZWwoImRlcHJvamVjdGVkIHJhZGl1cyBbIiArIHVuaXRfbGFiZWwgKyAiXSIsIGZvbnRzaXplPTguNSkKICAgICAgICBheDIuc2V0X3lsYWJlbCgic3VyZmFjZSBkZW5zaXR5IFtuIHBlciAiICsgdW5pdF9sYWJlbCArICJeMl0iLCBmb250c2l6ZT04LjUpCiAgICAgICAgYXgyLnRpY2tfcGFyYW1zKGxhYmVsc2l6ZT03LjUpCiAgICAgICAgYXgyLmdyaWQoYWxwaGE9MC4xNSwgbHc9MC42KQogICAgICAgIGF4Mi5zZXRfdGl0bGUoIk4gPSAiICsgc3RyKGludChyZXNbImNvdW50cyJdLnN1bSgpKSksIGZvbnRzaXplPTksIGxvYz0icmlnaHQiKQoKICAgICAgICBheC50ZXh0KDAuMCwgLTAuMzAsICJcbiIuam9pbih0ZXh0d3JhcC53cmFwKHRbIm5vdGUiXSwgd2lkdGg9MTI1KSksCiAgICAgICAgICAgICAgICB0cmFuc2Zvcm09YXgudHJhbnNBeGVzLCBmb250c2l6ZT03LjYsIGNvbG9yPSIjNGE0YTRhIiwKICAgICAgICAgICAgICAgIHZhPSJ0b3AiLCBoYT0ibGVmdCIsIGxpbmVzcGFjaW5nPTEuNSkKCiAgICBheGMgPSBmaWcuYWRkX3N1YnBsb3QoZ3NbbiwgOl0pCiAgICBmb3IgaSwgKHQsIHJlcykgaW4gZW51bWVyYXRlKHppcCh0cmVhdG1lbnRzLCByZXN1bHRzKSk6CiAgICAgICAgZyA9IHJlc1siY29tcGxldGUiXQogICAgICAgIGlmIGcuYW55KCk6CiAgICAgICAgICAgIHN0ZXBfcHJvZmlsZShheGMsIHJlc1siZWRnZXMiXSAqIHNjYWxlLAogICAgICAgICAgICAgICAgICAgICAgICAgbnAud2hlcmUoZywgcmVzWyJkZW5zaXR5Il0gLyBzY2FsZSAqKiAyLCBucC5uYW4pLAogICAgICAgICAgICAgICAgICAgICAgICAgQ09MT1VSU1tpICUgbGVuKENPTE9VUlMpXSwgbGFiZWw9dFsibGFiZWwiXSkKICAgIGF4Yy5zZXRfeXNjYWxlKCJsb2ciKQogICAgYXhjLnNldF94bGFiZWwoImRlcHJvamVjdGVkIHJhZGl1cyBbIiArIHVuaXRfbGFiZWwgKyAiXSIpCiAgICBheGMuc2V0X3lsYWJlbCgic3VyZmFjZSBkZW5zaXR5IFtuIHBlciAiICsgdW5pdF9sYWJlbCArICJeMl0iKQogICAgYXhjLmdyaWQoYWxwaGE9MC4xOCwgbHc9MC42KQogICAgYXhjLmxlZ2VuZChmb250c2l6ZT04LjUsIGZyYW1lb249RmFsc2UpCiAgICBheGMuc2V0X3RpdGxlKCJBbGwgdHJlYXRtZW50cyBvdmVybGFpZCAtIHRoZSBzcHJlYWQgYmV0d2VlbiB0aGVtIGlzIHRoZSAiCiAgICAgICAgICAgICAgICAgICJ1bmNlcnRhaW50eSBvbiB0aGUgcmFkaWkiLCBmb250c2l6ZT0xMCwgbG9jPSJsZWZ0IikKCiAgICBmaWcuc3VwdGl0bGUodGl0bGUsIGZvbnRzaXplPTEzLCB5PTAuOTk1KQogICAgcmV0dXJuIGZpZwoKZGVmIF9wcm9maWxlX3Jvd3MocmVzdWx0LCBzY2FsZSwgdW5pdCk6CiAgICAiIiJUaGUgcHJvZmlsZSBleGFjdGx5IGFzIHRoZSBwaWN0dXJlIGRyYXdzIGl0LCByaW5nIGJ5IHJpbmcuIiIiCiAgICBlZGdlcyA9IHJlc3VsdFsiZWRnZXMiXSAqIHNjYWxlCiAgICBoZWFkID0gWyJyaW5nX2lubmVyXyIgKyB1bml0LCAicmluZ19vdXRlcl8iICsgdW5pdCwgInJfY2VudHJlXyIgKyB1bml0LAogICAgICAgICAgICAibl9vYmplY3RzIiwgImRlbnNpdHlfcGVyXyIgKyB1bml0ICsgIjIiLCAiZGVuc2l0eV9lcnJvciIsCiAgICAgICAgICAgICJyaW5nX2NvbXBsZXRlIl0KICAgIHJvd3MgPSBbXQogICAgZm9yIGsgaW4gcmFuZ2UobGVuKHJlc3VsdFsiciJdKSk6CiAgICAgICAgcm93cy5hcHBlbmQoW2Zsb2F0KGVkZ2VzW2tdKSwgZmxvYXQoZWRnZXNbayArIDFdKSwKICAgICAgICAgICAgICAgICAgICAgZmxvYXQocmVzdWx0WyJyIl1ba10gKiBzY2FsZSksCiAgICAgICAgICAgICAgICAgICAgIGludChyZXN1bHRbImNvdW50cyJdW2tdKSwKICAgICAgICAgICAgICAgICAgICAgZmxvYXQocmVzdWx0WyJkZW5zaXR5Il1ba10gLyBzY2FsZSAqKiAyKSwKICAgICAgICAgICAgICAgICAgICAgZmxvYXQocmVzdWx0WyJkZW5zaXR5X2VyciJdW2tdIC8gc2NhbGUgKiogMiksCiAgICAgICAgICAgICAgICAgICAgIGludChib29sKHJlc3VsdFsiY29tcGxldGUiXVtrXSkpXSkKICAgIHJldHVybiBoZWFkLCByb3dzCgpkZWYgX3Bvc2l0aW9uX3Jvd3MocmVzdWx0LCBzY2FsZSwgdW5pdCk6CiAgICAiIiJFdmVyeSBvYmplY3QsIHdoZXJlIHRoZSBwaWN0dXJlIHB1dHMgaXQ6IGRlcHJvamVjdGVkLCBjZW50cmVkLCBpbiBgdW5pdGAuIiIiCiAgICBoZWFkID0gWyJ4X2Zyb21fY2VudHJlXyIgKyB1bml0LCAieV9mcm9tX2NlbnRyZV8iICsgdW5pdCwKICAgICAgICAgICAgInJhZGl1c18iICsgdW5pdF0KICAgIGR4ID0gbnAuYXNhcnJheShyZXN1bHRbImR4Il0sIGR0eXBlPWZsb2F0KSAqIHNjYWxlCiAgICBkeSA9IG5wLmFzYXJyYXkocmVzdWx0WyJkeSJdLCBkdHlwZT1mbG9hdCkgKiBzY2FsZQogICAgcm93cyA9IFtbZmxvYXQoYSksIGZsb2F0KGIpLCBmbG9hdChucC5oeXBvdChhLCBiKSldIGZvciBhLCBiIGluIHppcChkeCwgZHkpXQogICAgcmV0dXJuIGhlYWQsIHJvd3MKCmRlZiBfd3JpdGVfdGFibGVzKHBhdGgsIHNoZWV0cyk6CiAgICAiIiJXcml0ZSBlYWNoIHRhYmxlIGFzIGl0cyBvd24gQ1NWLCBuYW1lZCBhZnRlciB0aGUgcGljdHVyZSBpdCBiZWxvbmdzIHRvLgoKICAgIFdIWSBPTkUgRklMRSBQRVIgVEFCTEUgQU5EIE5PVCBPTkUgUEVSIFBJQ1RVUkUKICAgIEJlY2F1c2UgdGhlIHR3byB0YWJsZXMgYmVoaW5kIGEgcGljdHVyZSBoYXZlIGRpZmZlcmVudCBzaGFwZXMgLSBvbmUgcm93IHBlcgogICAgcmluZyBpbiB0aGUgcHJvZmlsZSwgb25lIHJvdyBwZXIgb2JqZWN0IGluIHRoZSBwb3NpdGlvbnMgLSBhbmQgcHV0dGluZyB0aGVtIGluCiAgICBvbmUgZmlsZSBtZWFucyBvbmUgb2YgdGhlbSBzdGFydHMgaGFsZndheSBkb3duLCB3aGljaCBpcyBleGFjdGx5IHRoZSBsYXlvdXQKICAgIHRoYXQgbWFrZXMgYSBzcHJlYWRzaGVldCBhd2t3YXJkIHRvIGNoYXJ0IGZyb20uIFR3byBmaWxlcywgZWFjaCBhIGNsZWFuCiAgICByZWN0YW5nbGUgd2l0aCBhIGhlYWRlciByb3csIGNhbiBiZSBjaGFydGVkIGJ5IHNlbGVjdGluZyB0d28gY29sdW1ucy4KCiAgICBDU1YgcmF0aGVyIHRoYW4gYSBzcHJlYWRzaGVldCBmb3JtYXQgb24gcHVycG9zZTogaXQgb3BlbnMgaW4gRXhjZWwsIGl0IG9wZW5zCiAgICBpbiBhbnl0aGluZyBlbHNlLCBhbmQgaXQgbmVlZHMgbm8gbGlicmFyeSB0aGF0IG1pZ2h0IG5vdCBiZSBpbnN0YWxsZWQgb24gdGhlCiAgICBtYWNoaW5lIHRoaXMgZW5kcyB1cCBydW5uaW5nIG9uLgoKICAgIFJldHVybnMgdGhlIHBhdGhzIHdyaXR0ZW4uCiAgICAiIiIKICAgIGltcG9ydCBjc3YgYXMgX2NzdgoKICAgIHN0ZW0gPSBvcy5wYXRoLnNwbGl0ZXh0KHBhdGgpWzBdCiAgICB3cml0dGVuID0gW10KICAgIGZvciBuYW1lLCAoaGVhZCwgcm93cykgaW4gc2hlZXRzLml0ZW1zKCk6CiAgICAgICAgc2x1ZyA9ICJfIi5qb2luKHAgZm9yIHAgaW4KICAgICAgICAgICAgICAgICAgICAgICAgIiIuam9pbihjIGlmIGMuaXNhbG51bSgpIGVsc2UgIl8iIGZvciBjIGluIG5hbWUpLnNwbGl0KCJfIikKICAgICAgICAgICAgICAgICAgICAgICAgaWYgcCkubG93ZXIoKQogICAgICAgIG91dCA9ICJ7fV97fS5jc3YiLmZvcm1hdChzdGVtLCBzbHVnKQogICAgICAgIHdpdGggb3BlbihvdXQsICJ3IiwgbmV3bGluZT0iIiwgZW5jb2Rpbmc9InV0Zi04IikgYXMgZmg6CiAgICAgICAgICAgIHdyaXRlciA9IF9jc3Yud3JpdGVyKGZoKQogICAgICAgICAgICB3cml0ZXIud3JpdGVyb3coaGVhZCkKICAgICAgICAgICAgd3JpdGVyLndyaXRlcm93cyhyb3dzKQogICAgICAgIHdyaXR0ZW4uYXBwZW5kKG91dCkKICAgIHJldHVybiB3cml0dGVuCgpkZWYgd3JpdGVfdHJlYXRtZW50X3dvcmtib29rKHBhdGgsIHRyZWF0bWVudCwgcmVzdWx0LCBzY2FsZSwgdW5pdCk6CiAgICAiIiJUaGUgdHdvIHRhYmxlcyBiZWhpbmQgb25lIHRyZWF0bWVudCdzIHBpY3R1cmUuIiIiCiAgICByZXR1cm4gX3dyaXRlX3RhYmxlcyhwYXRoLCB7CiAgICAgICAgInByb2ZpbGUiOiBfcHJvZmlsZV9yb3dzKHJlc3VsdCwgc2NhbGUsIHVuaXQpLAogICAgICAgICJwb3NpdGlvbnMiOiBfcG9zaXRpb25fcm93cyhyZXN1bHQsIHNjYWxlLCB1bml0KSwKICAgIH0pCgpkZWYgd3JpdGVfb3ZlcmxheV93b3JrYm9vayhwYXRoLCB0cmVhdG1lbnRzLCByZXN1bHRzLCBzY2FsZSwgdW5pdCk6CiAgICAiIiJPbmUgdGFibGUgcGVyIGN1cnZlIGluIHRoZSBvdmVybGF5LiIiIgogICAgc2hlZXRzID0ge30KICAgIGZvciB0cmVhdG1lbnQsIHJlc3VsdCBpbiB6aXAodHJlYXRtZW50cywgcmVzdWx0cyk6CiAgICAgICAgbmFtZSA9ICIiLmpvaW4oYyBpZiAoYy5pc2FsbnVtKCkgb3IgYyA9PSAiICIpIGVsc2UgIiAiCiAgICAgICAgICAgICAgICAgICAgICAgZm9yIGMgaW4gdHJlYXRtZW50WyJsYWJlbCJdKS5zdHJpcCgpCiAgICAgICAgc2hlZXRzW25hbWVbOjMxXSBvciAidHJlYXRtZW50Il0gPSBfcHJvZmlsZV9yb3dzKHJlc3VsdCwgc2NhbGUsIHVuaXQpCiAgICByZXR1cm4gX3dyaXRlX3RhYmxlcyhwYXRoLCBzaGVldHMpCgpkZWYgb25lX3RyZWF0bWVudF9maWd1cmUodHJlYXRtZW50LCByZXN1bHQsIHVuaXRfbGFiZWwsIHRpdGxlLCBzY2FsZT0xLjAsCiAgICAgICAgICAgICAgICAgICAgICAgICBjb2xvdXJfaW5kZXg9MCk6CiAgICAiIiJPbmUgdHJlYXRtZW50IG9uIGl0cyBvd24gc2hlZXQ6IHdoZXJlIHRoZSBvYmplY3RzIGFyZSwgYW5kIHRoZWlyIHByb2ZpbGUuCgogICAgV0hZIFRIRSBTQU1FIFRISU5HIElTIERSQVdOIFRXSUNFCiAgICBUaGUgY29tYmluZWQgc2hlZXQgaXMgZm9yIGNvbXBhcmluZyB0aGUgdHJlYXRtZW50cyBhZ2FpbnN0IGVhY2ggb3RoZXIsIHdoaWNoCiAgICBpcyB3aGF0IGl0IGlzIGZvciBhbmQgd2hhdCBpdCBpcyBnb29kIGF0LiBJdCBpcyBub3Qgd2hhdCB5b3UgcHV0IGluIGZyb250IG9mCiAgICBzb21lYm9keSwgb3IgaW50byBhIHJlcG9ydDogYXQgZm91ciBpbmNoZXMgaGlnaCBhIHBhbmVsIGlzIHRvbyBzbWFsbCB0byByZWFkIGEKICAgIHJhZGl1cyBvZmYgaXQsIGFuZCB0d28gdHJlYXRtZW50cyBzaWRlIGJ5IHNpZGUgY29tcGV0ZSBmb3IgdGhlIGV5ZSB3aGVuIG9ubHkKICAgIG9uZSBvZiB0aGVtIGlzIHVuZGVyIGRpc2N1c3Npb24uIFNvIGVhY2ggaXMgd3JpdHRlbiBhZ2FpbiBvbiBpdHMgb3duLCBmdWxsCiAgICBzaXplLCBhbmQgdGhlIG92ZXJsYXkgdG9vLiBUaGUgY29tYmluZWQgc2hlZXQgaXMgdW5jaGFuZ2VkOyB0aGVzZSBhcmUgZXh0cmEuCiAgICAiIiIKICAgIGNvbCA9IENPTE9VUlNbY29sb3VyX2luZGV4ICUgbGVuKENPTE9VUlMpXQogICAgZmlnID0gcGx0LmZpZ3VyZShmaWdzaXplPSgxMy4wLCA2LjIpKQogICAgZ3MgPSBmaWcuYWRkX2dyaWRzcGVjKDEsIDIsIHdzcGFjZT0wLjI4KQoKICAgIGR4dSwgZHl1ID0gcmVzdWx0WyJkeCJdICogc2NhbGUsIHJlc3VsdFsiZHkiXSAqIHNjYWxlCiAgICBydSA9IHJlc3VsdFsiciJdICogc2NhbGUKICAgIGR1ID0gcmVzdWx0WyJkZW5zaXR5Il0gLyBzY2FsZSAqKiAyCiAgICBldSA9IHJlc3VsdFsiZGVuc2l0eV9lcnIiXSAvIHNjYWxlICoqIDIKCiAgICBheCA9IGZpZy5hZGRfc3VicGxvdChnc1swLCAwXSkKICAgIGF4LnNjYXR0ZXIoZHh1LCBkeXUsIHM9MTQsIGM9Y29sLCBhbHBoYT0wLjcsIGxpbmV3aWR0aHM9MCkKICAgIGF4LnBsb3QoMCwgMCwgIisiLCBjPSJrIiwgbXM9MTQsIG1ldz0xLjgpCiAgICBsaW0gPSAxLjA2ICogbWF4KG5wLmFicyhkeHUpLm1heCgpLCBucC5hYnMoZHl1KS5tYXgoKSkKICAgIGF4LnNldF94bGltKC1saW0sIGxpbSkKICAgIGF4LnNldF95bGltKC1saW0sIGxpbSkKICAgIGF4LnNldF9hc3BlY3QoImVxdWFsIiwgYWRqdXN0YWJsZT0iYm94IikKICAgIGF4LnNldF94bGFiZWwoInggZnJvbSBjZW50cmUgWyIgKyB1bml0X2xhYmVsICsgIl0iKQogICAgYXguc2V0X3lsYWJlbCgieSBmcm9tIGNlbnRyZSBbIiArIHVuaXRfbGFiZWwgKyAiXSIpCiAgICBheC5ncmlkKGFscGhhPTAuMTUsIGx3PTAuNikKICAgIGF4LnNldF90aXRsZSgid2hlcmUgdGhleSBhcmUiLCBmb250c2l6ZT0xMSwgbG9jPSJsZWZ0IikKCiAgICBheDIgPSBmaWcuYWRkX3N1YnBsb3QoZ3NbMCwgMV0pCiAgICBnID0gcmVzdWx0WyJjb21wbGV0ZSJdCiAgICBlZGdlcyA9IHJlc3VsdFsiZWRnZXMiXSAqIHNjYWxlCiAgICBzdGVwX3Byb2ZpbGUoYXgyLCBlZGdlcywgbnAud2hlcmUoZywgZHUsIG5wLm5hbiksIGNvbCkKICAgIGlmIGcuYW55KCk6CiAgICAgICAgYXgyLmVycm9yYmFyKHJ1W2ddLCBkdVtnXSwgeWVycj1ldVtnXSwgZm10PSJub25lIiwgZWNvbG9yPWNvbCwKICAgICAgICAgICAgICAgICAgICAgZWxpbmV3aWR0aD0xLjIsIGNhcHNpemU9MywgYWxwaGE9MC44KQogICAgaWYgKH5nKS5hbnkoKToKICAgICAgICBzdGVwX3Byb2ZpbGUoYXgyLCBlZGdlcywgbnAud2hlcmUofmcsIGR1LCBucC5uYW4pLCBjb2wsCiAgICAgICAgICAgICAgICAgICAgIGx3PTEuMywgYWxwaGE9MC40NSwgbHM9Ii0tIiwKICAgICAgICAgICAgICAgICAgICAgbGFiZWw9InJpbmcgY2xpcHBlZCBieSB0aGUgZnJhbWUiKQogICAgICAgIGF4Mi5sZWdlbmQoZm9udHNpemU9OSwgZnJhbWVvbj1GYWxzZSwgbG9jPSJsb3dlciBsZWZ0IikKICAgIGF4Mi5zZXRfeXNjYWxlKCJsb2ciKQogICAgYXgyLnNldF94bGFiZWwoImRlcHJvamVjdGVkIHJhZGl1cyBbIiArIHVuaXRfbGFiZWwgKyAiXSIpCiAgICBheDIuc2V0X3lsYWJlbCgic3VyZmFjZSBkZW5zaXR5IFtuIHBlciAiICsgdW5pdF9sYWJlbCArICJeMl0iKQogICAgYXgyLmdyaWQoYWxwaGE9MC4xNSwgbHc9MC42KQogICAgYXgyLnNldF90aXRsZSgiTiA9ICIgKyBzdHIoaW50KHJlc3VsdFsiY291bnRzIl0uc3VtKCkpKSwgZm9udHNpemU9MTAsCiAgICAgICAgICAgICAgICAgIGxvYz0icmlnaHQiKQogICAgYXgyLnNldF90aXRsZSgiaG93IG1hbnkgcGVyIHVuaXQgYXJlYSwgYWdhaW5zdCByYWRpdXMiLCBmb250c2l6ZT0xMSwKICAgICAgICAgICAgICAgICAgbG9jPSJsZWZ0IikKCiAgICBmaWcuc3VwdGl0bGUoInt9IC0ge30iLmZvcm1hdCh0aXRsZSwgdHJlYXRtZW50WyJsYWJlbCJdKSwgZm9udHNpemU9MTMpCiAgICBmaWcudGV4dCgwLjA5LCAwLjAxNSwKICAgICAgICAgICAgIGNocigxMCkuam9pbih0ZXh0d3JhcC53cmFwKHRyZWF0bWVudFsibm90ZSJdLCB3aWR0aD0xNTApKSwKICAgICAgICAgICAgIGZvbnRzaXplPTksIGNvbG9yPSIjNGE0YTRhIiwgdmE9ImJvdHRvbSIpCiAgICBmaWcuc3VicGxvdHNfYWRqdXN0KGJvdHRvbT0wLjE3LCB0b3A9MC45MCkKICAgIHJldHVybiBmaWcKCmRlZiBvdmVybGF5X2ZpZ3VyZSh0cmVhdG1lbnRzLCByZXN1bHRzLCB1bml0X2xhYmVsLCB0aXRsZSwgc2NhbGU9MS4wKToKICAgICIiIlRoZSB0cmVhdG1lbnRzIG9uIG9uZSBwYWlyIG9mIGF4ZXMsIG9uIGEgc2hlZXQgb2YgaXRzIG93bi4iIiIKICAgIGZpZywgYXggPSBwbHQuc3VicGxvdHMoZmlnc2l6ZT0oMTMuMCwgNi4yKSkKICAgIGZvciBpLCAodCwgcmVzKSBpbiBlbnVtZXJhdGUoemlwKHRyZWF0bWVudHMsIHJlc3VsdHMpKToKICAgICAgICBnID0gcmVzWyJjb21wbGV0ZSJdCiAgICAgICAgaWYgZy5hbnkoKToKICAgICAgICAgICAgc3RlcF9wcm9maWxlKGF4LCByZXNbImVkZ2VzIl0gKiBzY2FsZSwKICAgICAgICAgICAgICAgICAgICAgICAgIG5wLndoZXJlKGcsIHJlc1siZGVuc2l0eSJdIC8gc2NhbGUgKiogMiwgbnAubmFuKSwKICAgICAgICAgICAgICAgICAgICAgICAgIENPTE9VUlNbaSAlIGxlbihDT0xPVVJTKV0sIGxhYmVsPXRbImxhYmVsIl0pCiAgICBheC5zZXRfeXNjYWxlKCJsb2ciKQogICAgYXguc2V0X3hsYWJlbCgiZGVwcm9qZWN0ZWQgcmFkaXVzIFsiICsgdW5pdF9sYWJlbCArICJdIikKICAgIGF4LnNldF95bGFiZWwoInN1cmZhY2UgZGVuc2l0eSBbbiBwZXIgIiArIHVuaXRfbGFiZWwgKyAiXjJdIikKICAgIGF4LmdyaWQoYWxwaGE9MC4xOCwgbHc9MC42KQogICAgYXgubGVnZW5kKGZvbnRzaXplPTEwLCBmcmFtZW9uPUZhbHNlKQogICAgYXguc2V0X3RpdGxlKCJ7fSAtIGFsbCB0cmVhdG1lbnRzIG92ZXJsYWlkOyB0aGUgc3ByZWFkIGJldHdlZW4gdGhlbSBpcyB0aGUgIgogICAgICAgICAgICAgICAgICJ1bmNlcnRhaW50eSBvbiB0aGUgcmFkaWkiLmZvcm1hdCh0aXRsZSksIGZvbnRzaXplPTEyLCBsb2M9ImxlZnQiKQogICAgZmlnLnRpZ2h0X2xheW91dCgpCiAgICByZXR1cm4gZmlnCgpkZWYgcnVuX3Byb2ZpbGVzKHgsIHksIGNlbnRyZSwgZnJhbWUsIG91dF9wcmVmaXgsIHRpdGxlLAogICAgICAgICAgICAgICAgIGdhbGF4eT1Ob25lLCBnZW9tPU5vbmUsIG5vcnRoX2FuZ2xlPTkwLjAsIG1pcnJvcmVkPUZhbHNlLAogICAgICAgICAgICAgICAgIG1hbnVhbF9pbmM9Tm9uZSwgbWFudWFsX3BhX2ltYWdlPU5vbmUsIG1hbnVhbF9wYV9za3k9Tm9uZSwKICAgICAgICAgICAgICAgICBzZWN0b3JfbWV0aG9kPUZhbHNlLCBzZWN0b3JzPTIwLCBzZWN0b3Jfc3RlcD01LjAsCiAgICAgICAgICAgICAgICAgcmluZ3M9MjAsIHNjYWxlPTEuMCwgdW5pdD0icHgiLCB2ZXJib3NlPVRydWUpOgogICAgIiIiRGVwcm9qZWN0LCBiaW4sIHBsb3QgYW5kIHNhdmUuIFJldHVybnMgKHRyZWF0bWVudHMsIHJlc3VsdHMsIHBhdGhzKS4KCiAgICBvdXRfcHJlZml4IGlzIGEgcGF0aCBzdGVtOyB0aGlzIHdyaXRlcyA8cHJlZml4Pl9wcm9maWxlcy5jc3YgYW5kCiAgICA8cHJlZml4Pl9wcm9maWxlcy5wbmcgbmV4dCB0byBpdC4KICAgICIiIgogICAgeCA9IG5wLmFzYXJyYXkoeCwgZHR5cGU9ZmxvYXQpCiAgICB5ID0gbnAuYXNhcnJheSh5LCBkdHlwZT1mbG9hdCkKCiAgICAjIFRvbyBtYW55IHJpbmdzIGZvciB0b28gZmV3IG9iamVjdHMgZG9lcyBub3QgYWRkIGRldGFpbCwgaXQgbWFudWZhY3R1cmVzCiAgICAjIG5vaXNlOiByaW5ncyBob2xkaW5nIG9uZSBvYmplY3Qgb3Igbm9uZSwgZXJyb3IgYmFycyB0YWxsZXIgdGhhbiB0aGUgcHJvZmlsZSwKICAgICMgYW5kIGEgamFnZ2VkIGxpbmUgdGhhdCBpbnZpdGVzIHJlYWRpbmcgc3RydWN0dXJlIGludG8gY291bnRpbmcgc3RhdGlzdGljcy4KICAgICMgU2VlbiBvbiBhIHJlYWwgcnVuIC0gMTgxIGJsdWUga25vdHMgc3ByZWFkIG92ZXIgMzAgcmluZ3MsIHdob2xlIHJpbmdzIGVtcHR5LgogICAgaWYgcmluZ3MgYW5kIGxlbih4KSBhbmQgbGVuKHgpIC8gZmxvYXQocmluZ3MpIDwgOC4wOgogICAgICAgIHN1Z2dlc3RlZCA9IG1heCg0LCBpbnQobGVuKHgpIC8gMTIpKQogICAgICAgIHByaW50KCIgIG5vdGU6IHt9IG9iamVjdHMgb3ZlciB7fSByaW5ncyBpcyBhYm91dCB7Oi4xZn0gcGVyIHJpbmcsIHNvIHRoZSAiCiAgICAgICAgICAgICAgInByb2ZpbGUgYmVsb3ciLmZvcm1hdChsZW4oeCksIHJpbmdzLCBsZW4oeCkgLyBmbG9hdChyaW5ncykpKQogICAgICAgIHByaW50KCIgICAgICAgIHdpbGwgYmUgZG9taW5hdGVkIGJ5IGNvdW50aW5nIG5vaXNlLiBBYm91dCB7fSByaW5ncyB3b3VsZCAiCiAgICAgICAgICAgICAgImJlIHJlYWRhYmxlLiIuZm9ybWF0KHN1Z2dlc3RlZCkpCgogICAgaWYgZ2VvbSBpcyBOb25lIGFuZCBnYWxheHk6CiAgICAgICAgdHJ5OgogICAgICAgICAgICBkZXNjcmliZSA9IGRlc2NyaWJlX19nYWxheHlfY2F0YWxvZ3VlICAjIGZyb20gZ2FsYXh5X2NhdGFsb2d1ZSwgaW5saW5lZCBhYm92ZQogICAgICAgICAgICBnZW9tID0gaHlwZXJsZWRhX2dlb21ldHJ5KGdhbGF4eSkKICAgICAgICAgICAgaWYgdmVyYm9zZToKICAgICAgICAgICAgICAgIHByaW50KCJcbkh5cGVyTEVEQSBnZW9tZXRyeSBmb3IgIiArIGdhbGF4eSArICI6IikKICAgICAgICAgICAgICAgIHByaW50KGRlc2NyaWJlKGdlb20pKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOgogICAgICAgICAgICBpZiB2ZXJib3NlOgogICAgICAgICAgICAgICAgcHJpbnQoIlxuQ2F0YWxvZ3VlIGxvb2t1cCBmYWlsZWQgKHt9OiB7fSkuIENvbnRpbnVpbmcgd2l0aG91dCBpdC4iCiAgICAgICAgICAgICAgICAgICAgICAuZm9ybWF0KHR5cGUoZXhjKS5fX25hbWVfXywgZXhjKSkKCiAgICB0cmVhdG1lbnRzID0gW3RyZWF0bWVudF9ub25lKCldCiAgICB0ID0gdHJlYXRtZW50X2NhdGFsb2d1ZShnZW9tLCBub3J0aF9hbmdsZSwgbWlycm9yZWQpCiAgICBpZiB0OgogICAgICAgIHRyZWF0bWVudHMuYXBwZW5kKHQpCiAgICB0ID0gdHJlYXRtZW50X21hbnVhbChtYW51YWxfaW5jLCBtYW51YWxfcGFfaW1hZ2UsIG1hbnVhbF9wYV9za3ksCiAgICAgICAgICAgICAgICAgICAgICAgICBub3J0aF9hbmdsZSwgbWlycm9yZWQpCiAgICBpZiB0OgogICAgICAgIHRyZWF0bWVudHMuYXBwZW5kKHQpCiAgICBpZiBzZWN0b3JfbWV0aG9kOgogICAgICAgIHRyZWF0bWVudHMuYXBwZW5kKHRyZWF0bWVudF9zZWN0b3IoeCAtIGNlbnRyZVswXSwgeSAtIGNlbnRyZVsxXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHNlY3RvcnMsIHNlY3Rvcl9zdGVwKSkKCiAgICByZXN1bHRzID0gY29tcHV0ZV9wcm9maWxlcyh4LCB5LCBjZW50cmUsIGZyYW1lLCB0cmVhdG1lbnRzLCByaW5ncz1yaW5ncykKCiAgICBpZiB2ZXJib3NlOgogICAgICAgIHByaW50KCJcblRyZWF0bWVudHM6IikKICAgICAgICBmb3IgdCwgcmVzIGluIHppcCh0cmVhdG1lbnRzLCByZXN1bHRzKToKICAgICAgICAgICAgcHJpbnQoIiAgLSB7fTogcl9tYXggPSB7Oi4xZn0ge30sIHt9IG9mIHt9IHJpbmdzIGNsaXBwZWQgYnkgdGhlIGZyYW1lIgogICAgICAgICAgICAgICAgICAuZm9ybWF0KHRbImxhYmVsIl0sIHJlc1sicl9tYXgiXSAqIHNjYWxlLCB1bml0LAogICAgICAgICAgICAgICAgICAgICAgICAgIGludCgofnJlc1siY29tcGxldGUiXSkuc3VtKCkpLCByaW5ncykpCiAgICAgICAgICAgIHByaW50KCIgICAgICAiICsgdFsibm90ZSJdKQoKICAgIGNzdl9wYXRoID0gb3V0X3ByZWZpeCArICJfcHJvZmlsZXMuY3N2IgogICAgcG5nX3BhdGggPSBvdXRfcHJlZml4ICsgIl9wcm9maWxlcy5wbmciCiAgICB3cml0ZV9wcm9maWxlc19jc3YoY3N2X3BhdGgsIHRyZWF0bWVudHMsIHJlc3VsdHMsIHNjYWxlLCB1bml0KQogICAgZmlnID0gcHJvZmlsZV9maWd1cmUodHJlYXRtZW50cywgcmVzdWx0cywgdW5pdCwgdGl0bGUsIHNjYWxlKQogICAgZmlnLnNhdmVmaWcocG5nX3BhdGgsIGRwaT0xNTAsIGJib3hfaW5jaGVzPSJ0aWdodCIpCiAgICBwbHQuY2xvc2UoZmlnKQogICAgY29tYmluZWRfYm9va3MgPSB3cml0ZV9vdmVybGF5X3dvcmtib29rKAogICAgICAgIG9zLnBhdGguc3BsaXRleHQocG5nX3BhdGgpWzBdICsgIi5jc3YiLCB0cmVhdG1lbnRzLCByZXN1bHRzLCBzY2FsZSwgdW5pdCkKCiAgICBpZiB2ZXJib3NlOgogICAgICAgIHByaW50KCJcbldyb3RlICIgKyBjc3ZfcGF0aCkKICAgICAgICBwcmludCgiV3JvdGUgIiArIHBuZ19wYXRoKQogICAgIyBBbmQgZWFjaCBwYXJ0IGFnYWluIG9uIGEgc2hlZXQgb2YgaXRzIG93biwgZnVsbCBzaXplLiBUaGUgY29tYmluZWQgb25lCiAgICAjIGFib3ZlIGlzIHVudG91Y2hlZDsgdGhlc2UgYXJlIGluIGFkZGl0aW9uIHRvIGl0LgogICAgZXh0cmFfcGF0aHMgPSBbXQogICAgZm9yIGksICh0cmVhdCwgcmVzKSBpbiBlbnVtZXJhdGUoemlwKHRyZWF0bWVudHMsIHJlc3VsdHMpKToKICAgICAgICBzbHVnID0gJycuam9pbihjIGlmIGMuaXNhbG51bSgpIGVsc2UgJ18nIGZvciBjIGluIHRyZWF0WydsYWJlbCddKQogICAgICAgIHNsdWcgPSAnXycuam9pbihwIGZvciBwIGluIHNsdWcuc3BsaXQoJ18nKSBpZiBwKS5sb3dlcigpCiAgICAgICAgb25lX3BhdGggPSAne31fcHJvZmlsZV97fS5wbmcnLmZvcm1hdChvdXRfcHJlZml4LCBzbHVnKQogICAgICAgIG9uZSA9IG9uZV90cmVhdG1lbnRfZmlndXJlKHRyZWF0LCByZXMsIHVuaXQsIHRpdGxlLCBzY2FsZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjb2xvdXJfaW5kZXg9aSkKICAgICAgICBvbmUuc2F2ZWZpZyhvbmVfcGF0aCwgZHBpPTE1MCwgYmJveF9pbmNoZXM9J3RpZ2h0JykKICAgICAgICBwbHQuY2xvc2Uob25lKQogICAgICAgIGV4dHJhX3BhdGhzLmFwcGVuZChvbmVfcGF0aCkKICAgICAgICBleHRyYV9wYXRocy5leHRlbmQod3JpdGVfdHJlYXRtZW50X3dvcmtib29rKAogICAgICAgICAgICBvbmVfcGF0aFs6LTRdICsgJy5jc3YnLCB0cmVhdCwgcmVzLCBzY2FsZSwgdW5pdCkpCiAgICBvdmVyX3BhdGggPSBvdXRfcHJlZml4ICsgJ19wcm9maWxlX2FsbF9vdmVybGFpZC5wbmcnCiAgICBvdmVyID0gb3ZlcmxheV9maWd1cmUodHJlYXRtZW50cywgcmVzdWx0cywgdW5pdCwgdGl0bGUsIHNjYWxlKQogICAgb3Zlci5zYXZlZmlnKG92ZXJfcGF0aCwgZHBpPTE1MCwgYmJveF9pbmNoZXM9J3RpZ2h0JykKICAgIHBsdC5jbG9zZShvdmVyKQogICAgZXh0cmFfcGF0aHMuYXBwZW5kKG92ZXJfcGF0aCkKICAgIGV4dHJhX3BhdGhzLmV4dGVuZCh3cml0ZV9vdmVybGF5X3dvcmtib29rKAogICAgICAgIG92ZXJfcGF0aFs6LTRdICsgJy5jc3YnLCB0cmVhdG1lbnRzLCByZXN1bHRzLCBzY2FsZSwgdW5pdCkpCiAgICBleHRyYV9wYXRoc1s6MF0gPSBjb21iaW5lZF9ib29rcwogICAgaWYgdmVyYm9zZToKICAgICAgICBmb3IgcCBpbiBleHRyYV9wYXRoczoKICAgICAgICAgICAgcHJpbnQoJ1dyb3RlICcgKyBwKQogICAgcmV0dXJuIHRyZWF0bWVudHMsIHJlc3VsdHMsIHR1cGxlKFtjc3ZfcGF0aCwgcG5nX3BhdGhdICsgZXh0cmFfcGF0aHMpCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIGJvcnJvd2VkIGZyb20gSElJX0Zyb21fSW1hZ2VzLnB5CiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KCm1hdHBsb3RsaWIudXNlKCJBZ2ciKQoKd2FybmluZ3MuZmlsdGVyd2FybmluZ3MoImlnbm9yZSIpCgpwYXNzICAjIGZyb20gcmFkaWFsX3Byb2ZpbGVzLCBpbmxpbmVkIGFib3ZlCgpwYXNzICAjIGZyb20gaW50ZXJhY3RpdmVfaW5wdXQsIGlubGluZWQgYWJvdmUKCnBsYW5fZm9sZGVyID0gcGxhbjsgZGVzY3JpYmVfZm9sZGVyID0gZGVzY3JpYmVfX2ZyYW1lX2ludmVudG9yeSAgIyBmcm9tIGZyYW1lX2ludmVudG9yeSwgaW5saW5lZCBhYm92ZQoKcGxhbl9mb2xkZXIgPSBwbGFuOyBkZXNjcmliZV9mb2xkZXIgPSBkZXNjcmliZV9fZnJhbWVfaW52ZW50b3J5ICAjIGZyb20gZnJhbWVfaW52ZW50b3J5LCBpbmxpbmVkIGFib3ZlCgpwYXNzICAjIGZyb20gZ2FsYXh5X2NhdGFsb2d1ZSwgaW5saW5lZCBhYm92ZQoKcGFzcyAgIyBmcm9tIGRlcHJvamVjdGlvbiwgaW5saW5lZCBhYm92ZQoKcGFzcyAgIyBmcm9tIGFuY2hvcl9yZWdpc3RyYXRpb24sIGlubGluZWQgYWJvdmUKCnBhc3MgICMgZnJvbSBnYWxheHlfd2luZG93LCBpbmxpbmVkIGFib3ZlCgpkZXNjcmliZV93aW5kb3cgPSBkZXNjcmliZV9fZ2FsYXh5X3dpbmRvdyAgIyBmcm9tIGdhbGF4eV93aW5kb3csIGlubGluZWQgYWJvdmUKCnBhc3MgICMgZnJvbSBnYWxheHlfZXh0ZW50LCBpbmxpbmVkIGFib3ZlCgpkZXNjcmliZV9leHRlbnQgPSBkZXNjcmliZV9fZ2FsYXh5X2V4dGVudCAgIyBmcm9tIGdhbGF4eV9leHRlbnQsIGlubGluZWQgYWJvdmUKCmRlc2NyaWJlX2FuY2hvcnMgPSBkZXNjcmliZV9fYW5jaG9yX3JlZ2lzdHJhdGlvbiAgIyBmcm9tIGFuY2hvcl9yZWdpc3RyYXRpb24sIGlubGluZWQgYWJvdmUKCmRlZiBkb25vcl9iZXNpZGUocGF0aCwgcm9sZSk6CiAgICAiIiJBIHNvbHZlZCBleHBvc3VyZSBpbiB0aGUgc2FtZSBmb2xkZXIgYXMgYHBhdGhgLCB3aGVuIGBwYXRoYCBoYXMgbm8gc29sdXRpb24uCgogICAgUmV0dXJucyBpdHMgcGF0aCwgb3IgTm9uZSB3aGVuIGBwYXRoYCBpcyBzb2x2ZWQgYWxyZWFkeSBvciBub3RoaW5nIGJlc2lkZSBpdAogICAgaXMuIFNhbWUgZmlsdGVyIGZpcnN0LCB0aGVuIGFueSBzb2x2ZWQgZnJhbWUgb2YgdGhlIGZpZWxkLgogICAgIiIiCiAgICB0cnk6CiAgICAgICAgcmVjID0gaW5zcGVjdChwYXRoKQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICByZXR1cm4gTm9uZQogICAgaWYgcmVjLmdldCgid2NzIik6CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIHJlY1sicm9sZSJdID0gcmVjLmdldCgicm9sZSIpIG9yIHJvbGUKICAgIGhlcmUgPSBvcy5wYXRoLmFic3BhdGgocGF0aCkKICAgIG90aGVycyA9IFtyIGZvciByIGluIHNjYW5fZm9sZGVyKG9zLnBhdGguZGlybmFtZShoZXJlKSwgcmVjdXJzaXZlPUZhbHNlKQogICAgICAgICAgICAgIGlmIG9zLnBhdGguYWJzcGF0aChyWyJwYXRoIl0pICE9IGhlcmUgYW5kIG5vdCByLmdldCgiZXJyb3IiKV0KICAgIGRvbm9yID0gZG9ub3JfZm9yKHJlYywgb3RoZXJzKQogICAgcmV0dXJuIGRvbm9yWyJwYXRoIl0gaWYgZG9ub3IgZWxzZSBOb25lCgpkZWYgZnJhbWVzX2Zyb21fZm9sZGVyKGZvbGRlciwgcm9sZXMpOgogICAgIiIiV29yayBvdXQgd2hpY2ggZmlsZSBpbiBhIGZvbGRlciBwbGF5cyBlYWNoIHBhcnQsIGFuZCBzYXkgc28gYmVmb3JlIHN0YXJ0aW5nLgoKICAgIFJldHVybnMgdGhyZWUgdGhpbmdzIHBlciByb2xlOiB0aGUgY2hvc2VuIGZyYW1lLCB0aGUgc29sdmVkIGV4cG9zdXJlIHRoYXQgd2lsbAogICAgbGVuZCBpdCBhIHBsYXRlIHNvbHV0aW9uIGlmIGl0IGhhcyBub25lLCBhbmQgdGhlIGxpc3Qgb2YgZXhwb3N1cmVzIHRvIGNvbWJpbmUKICAgIHdoZXJlIHRoZSBmb2xkZXIgaGVsZCBzZXZlcmFsIGFuZCBub3RoaW5nIGNvbWJpbmVkLgogICAgIiIiCiAgICBkZWYgcGljayhyb2xlLCBvcHRpb25zKToKICAgICAgICBwcmludCgiIikKICAgICAgICBwcmludCgiTW9yZSB0aGFuIG9uZSBmcmFtZSBjb3VsZCBiZSB0aGUgIiArIHJvbGUgKyAiOiIpCiAgICAgICAgZm9yIGksIG9wdCBpbiBlbnVtZXJhdGUob3B0aW9ucywgMSk6CiAgICAgICAgICAgIHByaW50KCIgIHt9LiB7fSIuZm9ybWF0KGksIG9wdFsibmFtZSJdKSkKICAgICAgICB3aGlsZSBUcnVlOgogICAgICAgICAgICBhbnN3ZXIgPSBhc2soIldoaWNoIG9uZSIsICIxIiwgaW50KQogICAgICAgICAgICBpZiAxIDw9IGFuc3dlciA8PSBsZW4ob3B0aW9ucyk6CiAgICAgICAgICAgICAgICByZXR1cm4gb3B0aW9uc1thbnN3ZXIgLSAxXQogICAgICAgICAgICBwcmludCgiICAgIHBpY2sgYSBudW1iZXIgZnJvbSB0aGUgbGlzdCIpCgogICAgcmVjb3JkcywgY2hvc2VuLCBtZW1iZXJzLCBkb25vcnMsIG5vdGVzID0gcGxhbl9mb2xkZXIoZm9sZGVyLCByb2xlcz1yb2xlcywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFzaz1waWNrKQogICAgcHJpbnQoZGVzY3JpYmVfZm9sZGVyKHJlY29yZHMsIGNob3Nlbiwgbm90ZXMpKQoKICAgIG1pc3NpbmcgPSBbciBmb3IgciBpbiByb2xlcyBpZiByIG5vdCBpbiBjaG9zZW5dCiAgICBpZiBtaXNzaW5nOgogICAgICAgIHJhaXNlIFN5c3RlbUV4aXQoIk5vdGhpbmcgaW4gdGhlIGZvbGRlciBpcyB0aGUgIiArICIgb3IgdGhlICIuam9pbihtaXNzaW5nKSkKCiAgICAjIFdoZXJlIHRoZSBmb2xkZXIgaG9sZHMgc2V2ZXJhbCBleHBvc3VyZXMgb2YgYSBmaWx0ZXIgYW5kIG5vdGhpbmcgY29tYmluZWQsCiAgICAjIHNheSB3aGF0IGNvbWJpbmluZyB0aGVtIGlzIHdvcnRoIGFuZCBsZXQgdGhlIHBlcnNvbiBkZWNpZGUuIFRoZSBnYWluIGlzIGEKICAgICMgcmVhbCBudW1iZXIsIHNvIGl0IGlzIHB1dCBvbiB0aGUgc2NyZWVuIHJhdGhlciB0aGFuIGhpZGRlbiBiZWhpbmQgYSBmbGFnCiAgICAjIG5vYm9keSB3aWxsIHJlbWVtYmVyIGEgeWVhciBmcm9tIG5vdy4KICAgIHRvX2NvbWJpbmUgPSB7fQogICAgZm9yIHJvbGUgaW4gcm9sZXM6CiAgICAgICAgZ3JvdXAgPSBtZW1iZXJzLmdldChyb2xlKSBvciBbXQogICAgICAgIGlmIGxlbihncm91cCkgPCAyOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGlmIGFza195ZXNfbm8oIlRoZXJlIGFyZSB7fSB7fSBleHBvc3VyZXMgYW5kIG5vIGNvbWJpbmVkIGZyYW1lLiBDb21iaW5pbmcgIgogICAgICAgICAgICAgICAgICAgICAgInRoZW0gbG93ZXJzIHRoZSBub2lzZSBieSBhYm91dCBhIGZhY3RvciBvZiB7Oi4xZn0uICIKICAgICAgICAgICAgICAgICAgICAgICJDb21iaW5lPyIuZm9ybWF0KGxlbihncm91cCksIHJvbGUsIGxlbihncm91cCkgKiogMC41KSwKICAgICAgICAgICAgICAgICAgICAgIFRydWUpOgogICAgICAgICAgICB0b19jb21iaW5lW3JvbGVdID0gW3JbInBhdGgiXSBmb3IgciBpbiBncm91cF0KICAgICAgICBlbHNlOgogICAgICAgICAgICBwcmludCgiICB1c2luZyB7fSBvbiBpdHMgb3duIi5mb3JtYXQoZ3JvdXBbMF1bIm5hbWUiXSkpCgogICAgbmFtZSwgbmFtZV9zb3VyY2UgPSBnYWxheHlfZnJvbV9yZWNvcmRzKHJlY29yZHMsIGZvbGRlcikKICAgIGlmIG5hbWU6CiAgICAgICAgcHJpbnQoIkdhbGF4eToge30gICAoZnJvbSB7fSkiLmZvcm1hdChuYW1lLCBuYW1lX3NvdXJjZSkpCgogICAgcmV0dXJuICh7cm9sZTogY2hvc2VuW3JvbGVdWyJwYXRoIl0gZm9yIHJvbGUgaW4gcm9sZXN9LAogICAgICAgICAgICB7cm9sZTogKGRvbm9yc1tyb2xlXVsicGF0aCJdIGlmIGRvbm9ycy5nZXQocm9sZSkgZWxzZSBOb25lKQogICAgICAgICAgICAgZm9yIHJvbGUgaW4gcm9sZXN9LAogICAgICAgICAgICB0b19jb21iaW5lLCBuYW1lKQoKZGVmIGhlYWRlcl9udW1iZXIoaGVhZGVyLCBrZXlzKToKICAgICIiIlRoZSBmaXJzdCBvZiB0aGVzZSBoZWFkZXIga2V5d29yZHMgdGhhdCBob2xkcyBhIHBvc2l0aXZlIG51bWJlciwgb3IgTm9uZS4iIiIKICAgIGZvciBrZXkgaW4ga2V5czoKICAgICAgICB0cnk6CiAgICAgICAgICAgIHZhbHVlID0gZmxvYXQoaGVhZGVyLmdldChrZXkpKQogICAgICAgIGV4Y2VwdCAoVHlwZUVycm9yLCBWYWx1ZUVycm9yKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICBpZiB2YWx1ZSA9PSB2YWx1ZSBhbmQgdmFsdWUgPiAwOgogICAgICAgICAgICByZXR1cm4gdmFsdWUKICAgIHJldHVybiBOb25lCgpkZWYgbG9hZF9mcmFtZShwYXRoLCBkb25vcl9wYXRoPU5vbmUpOgogICAgIiIiT3BlbiBhIEZJVFMgZmlsZSBhbmQgcmV0dXJuIHRoZSBzY2llbmNlIGltYWdlLCBpdHMgV0NTIGFuZCBhIGJhZC1waXhlbCBtYXNrLgoKICAgIEEgZnJhbWUgdGhhdCBjYW1lIG91dCBvZiBhIG1lZGlhbiBjb21iaW5lIHVzdWFsbHkgaGFzIG5vIHBsYXRlIHNvbHV0aW9uIGxlZnQKICAgIGluIGl0LCBiZWNhdXNlIHRoZSBjb21iaW5pbmcgc3RlcCB3cm90ZSBhIG5ldyBmaWxlIGFuZCBrZXB0IGFsbW9zdCBub3RoaW5nLgogICAgV2hlbiBgZG9ub3JfcGF0aGAgbmFtZXMgYSBzb2x2ZWQgZXhwb3N1cmUgb2YgdGhlIHNhbWUgZmllbGQsIGEgc29sdXRpb24gaXMKICAgIGZpdHRlZCBmb3IgdGhpcyBmcmFtZSBieSBtYXRjaGluZyBzdGFyIHBhdHRlcm5zIGFnYWluc3QgdGhhdCBvbmUsIGFuZCB0aGUKICAgIHJlc2lkdWFsIG9mIHRoYXQgZml0IGlzIHByaW50ZWQgc28gaXQgY2FuIGJlIGp1ZGdlZCByYXRoZXIgdGhhbiB0cnVzdGVkLgogICAgIiIiCiAgICBmcm9tIGFzdHJvcHkuaW8gaW1wb3J0IGZpdHMKICAgIGZyb20gYXN0cm9weS53Y3MgaW1wb3J0IFdDUwoKICAgIGhkdWwgPSBmaXRzLm9wZW4ocGF0aCkKICAgIHNjaSA9IE5vbmUKICAgIGZvciBoZHUgaW4gaGR1bDoKICAgICAgICBpZiBoZHUuZGF0YSBpcyBub3QgTm9uZSBhbmQgZ2V0YXR0cihoZHUuZGF0YSwgIm5kaW0iLCAwKSA9PSAyOgogICAgICAgICAgICBzY2kgPSBoZHUKICAgICAgICAgICAgYnJlYWsKICAgIGlmIHNjaSBpcyBOb25lOgogICAgICAgIGhkdWwuY2xvc2UoKQogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IocGF0aCArICI6IG5vIDItRCBpbWFnZSBmb3VuZCIpCgogICAgIyBTaW5nbGUgcHJlY2lzaW9uIG9uIHB1cnBvc2UuIENhbWVyYSBkYXRhIGFycml2ZXMgYXMgMTYtYml0IGludGVnZXJzLCBzbwogICAgIyBkb3VibGUgcHJlY2lzaW9uIGRvdWJsZXMgdGhlIG1lbW9yeSBmb3Igbm8gaW5mb3JtYXRpb24gYXQgYWxsOiBzZXZlbgogICAgIyBzaWduaWZpY2FudCBmaWd1cmVzIGFnYWluc3QgcGl4ZWwgbm9pc2Ugb2YgdGVucyBvZiBjb3VudHMuIE9uIGEgNjEtbWVnYXBpeGVsCiAgICAjIGZyYW1lIHRoZSBkaWZmZXJlbmNlIGlzIDI0NSBNQiBhZ2FpbnN0IDQ4OSwgYW5kIGZvdXIgZmlsdGVycyBvZiB0aG9zZSB3YXMKICAgICMgd2hhdCBkcm92ZSB0aGlzIG1hY2hpbmUgaW50byBzd2FwcGluZy4KICAgIGRhdGEgPSBucC5hc2FycmF5KHNjaS5kYXRhLCBkdHlwZT1ucC5mbG9hdDMyKQogICAgaGVhZGVyID0gc2NpLmhlYWRlcgoKICAgIGJwbSA9IE5vbmUKICAgIGZvciBoZHUgaW4gaGR1bDoKICAgICAgICBpZiBoZHUubmFtZS51cHBlcigpID09ICJCUE0iIGFuZCBoZHUuZGF0YSBpcyBub3QgTm9uZToKICAgICAgICAgICAgYnBtID0gbnAuYXNhcnJheShoZHUuZGF0YSkgPiAwCiAgICAgICAgICAgIGJyZWFrCgogICAgd2NzID0gTm9uZQogICAgdHJ5OgogICAgICAgIHcgPSBXQ1MoaGVhZGVyKQogICAgICAgIGlmIHcuaGFzX2NlbGVzdGlhbDoKICAgICAgICAgICAgd2NzID0gdy5jZWxlc3RpYWwKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcGFzcwoKICAgIGluZm8gPSB7CiAgICAgICAgImZpbHRlciI6IHN0cihoZWFkZXIuZ2V0KCJGSUxURVIiLCAiPyIpKSwKICAgICAgICAiZXhwdGltZSI6IGZsb2F0KGhlYWRlci5nZXQoIkVYUFRJTUUiLCAxLjApIG9yIDEuMCksCiAgICAgICAgInBpeGVsX3NjYWxlIjogaGVhZGVyLmdldCgiUElYU0NBTEUiLCBOb25lKSwKICAgICAgICAib2JqZWN0Ijogc3RyKGhlYWRlci5nZXQoIk9CSkVDVCIsICI/IikpLAogICAgICAgICMgV2hhdCB0aGUgcGhvdG9tZXRyeSBuZWVkcyB0byBrbm93IGFib3V0IHRoZSBkZXRlY3RvciwgY2FycmllZCBhbG9uZwogICAgICAgICMgc28gdGhhdCBhIGZyYW1lIHRoaXMgcGlwZWxpbmUgd3JpdGVzIHN0aWxsIHNheXMgaXQuCiAgICAgICAgInNhdHVyYXRlIjogaGVhZGVyX251bWJlcihoZWFkZXIsICgiU0FUVVJBVEUiLCAiTUFYTElOIiwgIlNBVExFVkVMIikpLAogICAgICAgICMgRWxlY3Ryb25zIHBlciBBRFUuIEVHQUlOIGZpcnN0OiBvbiBzb21lIGNhbWVyYXMgR0FJTiBpcyB0aGUgZ2FpbgogICAgICAgICMgU0VUVElORyAoNTEgb24gdGhlIEtpbm5lcmV0IGNvbG91ciBjYW1lcmEpLCBub3QgZWxlY3Ryb25zIHBlciBBRFUuCiAgICAgICAgImdhaW4iOiBuZXh0KChnIGZvciBnIGluIChoZWFkZXJfbnVtYmVyKGhlYWRlciwgKGssKSkgZm9yIGsgaW4KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICgiRUdBSU4iLCAiR0FJTiIpKSBpZiBnIGFuZCBnIDwgMjApLCBOb25lKSwKICAgICAgICAicmRub2lzZSI6IGhlYWRlcl9udW1iZXIoaGVhZGVyLCAoIlJETk9JU0UiLCAiUkVBRE5PSVMiLCAiUk9OIikpLAogICAgICAgICJuY29tYmluZSI6IGludChoZWFkZXJfbnVtYmVyKGhlYWRlciwgKCJOQ09NQklORSIsKSkgb3IgMSksCiAgICAgICAgImNvbWJ0eXBlIjogc3RyKGhlYWRlci5nZXQoIkNPTUJUWVBFIiwgIiIpKSwKICAgICAgICAiZGF0ZSI6IHN0cihoZWFkZXIuZ2V0KCJEQVRFLU9CUyIsICIiKSksCiAgICB9CiAgICBoZHVsLmNsb3NlKCkKCiAgICBpZiB3Y3MgaXMgTm9uZSBhbmQgZG9ub3JfcGF0aDoKICAgICAgICByZWZlcmVuY2UsIHJlZl93Y3MsIF8sIHJlZl9pbmZvID0gbG9hZF9mcmFtZShkb25vcl9wYXRoKQogICAgICAgIGlmIHJlZl93Y3MgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHdjcywgZGlhZyA9IHdjc19mcm9tX3JlZmVyZW5jZShkYXRhLCByZWZlcmVuY2UsIHJlZl93Y3MpCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOgogICAgICAgICAgICAgICAgIyBhc3Ryb2FsaWduIGZpbmRzIG5vIHN0YXIgcGF0dGVybiBpbiBjb21tb24gLSB0eXBpY2FsbHkgYSBmcmFtZQogICAgICAgICAgICAgICAgIyB3aXRoIHRoZSBzdGFycyB0YWtlbiBvdXQsIHN1Y2ggYXMgSC1hbHBoYSBtaW51cyBjb250aW51dW0uCiAgICAgICAgICAgICAgICBwcmludCgiICB7fTogY291bGQgbm90IGZpdCBhIHBsYXRlIHNvbHV0aW9uIGFnYWluc3Qge30gKHt9OiB7fSkiLmZvcm1hdCgKICAgICAgICAgICAgICAgICAgICBvcy5wYXRoLmJhc2VuYW1lKHBhdGgpLCBvcy5wYXRoLmJhc2VuYW1lKGRvbm9yX3BhdGgpLAogICAgICAgICAgICAgICAgICAgIHR5cGUoZXhjKS5fX25hbWVfXywgZXhjKSwgZmx1c2g9VHJ1ZSkKICAgICAgICAgICAgICAgIHJldHVybiBkYXRhLCBOb25lLCBicG0sIGluZm8KICAgICAgICAgICAgcHJpbnQoZGVzY3JpYmVfd2NzX2ZpdChvcy5wYXRoLmJhc2VuYW1lKHBhdGgpLCBkaWFnKSkKICAgICAgICAgICAgaWYgaW5mb1sicGl4ZWxfc2NhbGUiXSBpcyBOb25lOgogICAgICAgICAgICAgICAgaW5mb1sicGl4ZWxfc2NhbGUiXSA9IHJlZl9pbmZvWyJwaXhlbF9zY2FsZSJdCiAgICAgICAgICAgIGlmIGluZm9bImZpbHRlciJdID09ICI/IjoKICAgICAgICAgICAgICAgIGluZm9bImZpbHRlciJdID0gcmVmX2luZm9bImZpbHRlciJdCiAgICAgICAgICAgICMgQSBjb21iaW5lZCBmcmFtZSB3cml0dGVuIGJ5IGFub3RoZXIgcHJvZ3JhbSBvZnRlbiBrZWVwcyBubyBoZWFkZXIKICAgICAgICAgICAgIyBhdCBhbGwuIFRoZSBjYW1lcmEgYW5kIHRoZSBleHBvc3VyZSBsZW5ndGggYXJlIHRob3NlIG9mIHRoZSBzb2x2ZWQKICAgICAgICAgICAgIyBleHBvc3VyZSBpdCBjYW1lIGZyb20sIHNvIHRoZXkgYXJlIHRha2VuIGZyb20gdGhlcmUsIGFuZCBzYWlkIHNvLgogICAgICAgICAgICBib3Jyb3dlZCA9IFtdCiAgICAgICAgICAgIGZvciBrZXksIGNhcmQgaW4gKCgiZXhwdGltZSIsICJFWFBUSU1FIiksICgic2F0dXJhdGUiLCAiU0FUVVJBVEUiKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKCJnYWluIiwgIkdBSU4iKSwgKCJyZG5vaXNlIiwgIlJETk9JU0UiKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKCJkYXRlIiwgIkRBVEUtT0JTIiksICgib2JqZWN0IiwgIk9CSkVDVCIpKToKICAgICAgICAgICAgICAgIGlmIGNhcmQgaW4gaGVhZGVyIG9yIChrZXkgaW4gKCJzYXR1cmF0ZSIsICJnYWluIiwgInJkbm9pc2UiKQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFuZCBpbmZvW2tleV0gaXMgbm90IE5vbmUpOgogICAgICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgICAgICBpZiBrZXkgPT0gIm9iamVjdCIgYW5kIGluZm9ba2V5XSAhPSAiPyI6CiAgICAgICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgICAgIGlmIHJlZl9pbmZvLmdldChrZXkpIG5vdCBpbiAoTm9uZSwgIiIsICI/Iik6CiAgICAgICAgICAgICAgICAgICAgaW5mb1trZXldID0gcmVmX2luZm9ba2V5XQogICAgICAgICAgICAgICAgICAgIGJvcnJvd2VkLmFwcGVuZChjYXJkKQogICAgICAgICAgICBpZiBib3Jyb3dlZDoKICAgICAgICAgICAgICAgIHByaW50KCIgIHt9OiBubyB7fSBpbiBpdHMgaGVhZGVyOyB0YWtlbiBmcm9tIHt9Ii5mb3JtYXQoCiAgICAgICAgICAgICAgICAgICAgb3MucGF0aC5iYXNlbmFtZShwYXRoKSwgIiwgIi5qb2luKGJvcnJvd2VkKSwKICAgICAgICAgICAgICAgICAgICBvcy5wYXRoLmJhc2VuYW1lKGRvbm9yX3BhdGgpKSwgZmx1c2g9VHJ1ZSkKCiAgICByZXR1cm4gZGF0YSwgd2NzLCBicG0sIGluZm8KCmRlZiBfY2VudHJvaWRfbG9jYWxseShkYXRhLCBwb3NpdGlvbnMsIGJveCk6CiAgICAiIiJDZW50cmUgb2YgbGlnaHQgaW4gYSBzbWFsbCBib3gsIHdpdGggdGhhdCBib3gncyBvd24gc2t5IHRha2VuIG9mZiBmaXJzdC4KCiAgICBUaGUgb25lIHBsYWNlIHRoaXMgaXMgZG9uZSwgYmVjYXVzZSBnZXR0aW5nIGl0IHdyb25nIGlzIGludmlzaWJsZS4gVGhlIHNreSBpcwogICAgcGFydCBvZiB0aGUgd2VpZ2h0IGluIGEgY2VudHJlIG9mIGxpZ2h0LCBhbmQgaW4gYSBib3ggb2YgYSBjb3VwbGUgb2YgaHVuZHJlZAogICAgcGl4ZWxzIGl0IGlzIG1vc3Qgb2YgdGhlIHdlaWdodCwgc28gbGVhdmluZyBpdCBpbiByZXR1cm5zIHRoZSBtaWRkbGUgb2YgdGhlIGJveAogICAgZHJlc3NlZCB1cCBhcyBhIG1lYXN1cmVtZW50LiBFdmVyeSBjYWxsZXIgaGVyZSB3YW50cyB0aGUgc3Rhciwgbm90IHRoZSBib3guCgogICAgUmV0dXJucyBhbiBhcnJheSB0aGUgc2FtZSBsZW5ndGggYXMgYHBvc2l0aW9uc2AsIHdpdGggTmFOIHdoZXJlIGEgcG9zaXRpb24gc2l0cwogICAgdG9vIG5lYXIgYW4gZWRnZSBvciB0aGUgc3RhbXAgaG9sZHMgbm90aGluZyBhYm92ZSBpdHMgb3duIHNreS4KICAgICIiIgogICAgZGF0YSA9IG5wLmFzYXJyYXkoZGF0YSkKICAgIHBvcyA9IG5wLmFzYXJyYXkocG9zaXRpb25zLCBkdHlwZT1mbG9hdCkKICAgIGhhbGYgPSBib3ggLy8gMgogICAgbnksIG54ID0gZGF0YS5zaGFwZQogICAgZ3JpZF95LCBncmlkX3ggPSBucC5tZ3JpZFswOmJveCwgMDpib3hdCiAgICBvdXQgPSBucC5mdWxsKHBvcy5zaGFwZSwgbnAubmFuKQogICAgZm9yIGksICh4YywgeWMpIGluIGVudW1lcmF0ZShwb3MpOgogICAgICAgIGlmIG5vdCAobnAuaXNmaW5pdGUoeGMpIGFuZCBucC5pc2Zpbml0ZSh5YykpOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHhpLCB5aSA9IGludChyb3VuZCh4YykpLCBpbnQocm91bmQoeWMpKQogICAgICAgIGlmIG5vdCAoaGFsZiA8PSB4aSA8IG54IC0gaGFsZiBhbmQgaGFsZiA8PSB5aSA8IG55IC0gaGFsZik6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgc3RhbXAgPSBucC5uYW5fdG9fbnVtKAogICAgICAgICAgICBkYXRhW3lpIC0gaGFsZjp5aSArIGhhbGYgKyAxLCB4aSAtIGhhbGY6eGkgKyBoYWxmICsgMV0pLmFzdHlwZShmbG9hdCkKICAgICAgICBpZiBzdGFtcC5zaGFwZSAhPSAoYm94LCBib3gpOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHN0YW1wID0gc3RhbXAgLSBucC5tZWRpYW4oc3RhbXApCiAgICAgICAgc3RhbXBbc3RhbXAgPCAwLjBdID0gMC4wICAgICAgICAjIG5lZ2F0aXZlIG5vaXNlIG11c3Qgbm90IHB1bGwgdGhlIGNlbnRyZQogICAgICAgIHRvdGFsID0gc3RhbXAuc3VtKCkKICAgICAgICBpZiB0b3RhbCA8PSAwLjA6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgb3V0W2ldID0gW3hpIC0gaGFsZiArIChzdGFtcCAqIGdyaWRfeCkuc3VtKCkgLyB0b3RhbCwKICAgICAgICAgICAgICAgICAgeWkgLSBoYWxmICsgKHN0YW1wICogZ3JpZF95KS5zdW0oKSAvIHRvdGFsXQogICAgcmV0dXJuIG91dAoKZGVmIGNlbnRyZV9vbl9zdGFycyhkYXRhLCBwb3NpdGlvbnMsIGJveD0xNSwgbWF4X21vdmVfcHg9Ni4wKToKICAgICIiIk1vdmUgY2F0YWxvZ3VlIHBvc2l0aW9ucyBvbnRvIHRoZSBzdGFycyBhcyB0aGlzIGZyYW1lIGFjdHVhbGx5IHNob3dzIHRoZW0uCgogICAgV0hZIFRISVMgSVMgTkVFREVECiAgICBBIGNhdGFsb2d1ZSBwb3NpdGlvbiBhbmQgYSBzdGFyIGluIGEgZnJhbWUgYXJlIG5vdCB0aGUgc2FtZSB0aGluZyB0byBiZXR0ZXIKICAgIHRoYW4gYSBwaXhlbCBvciB0d28uIFRoZSBwbGF0ZSBzb2x1dGlvbiBoYXMgaXRzIG93biBlcnJvciwgdGhlIGNhdGFsb2d1ZSBpcyBhdAogICAgYSBkaWZmZXJlbnQgZXBvY2gsIGFuZCBhIGNvbWJpbmVkIGZyYW1lIGNhcnJpZXMgd2hhdGV2ZXIgaXRzIHJlZ2lzdHJhdGlvbiBsZWZ0CiAgICBiZWhpbmQuIE1lYXN1cmVkIG9uIGFuIE0zMyBSIGZyYW1lOiB0aGUgcmVhbCBzdGFycyBzYXQgYSBtZWRpYW4gb2YgMi40IHBpeGVscwogICAgYXdheSBmcm9tIHdoZXJlIEdhaWEgc2FpZCB0aGV5IHdlcmUsIGFuZCBub3QgYnkgYSBjb25zdGFudCBhbW91bnQgLSB0aGUgc3ByZWFkCiAgICB3YXMgMi4yIHBpeGVscy4KCiAgICBUaGF0IG1hdHRlcnMgYmVjYXVzZSBtb3N0IG9mIHdoYXQgaXMgZG9uZSB3aXRoIHRoZXNlIHBvc2l0aW9ucyBpcyBhcGVydHVyZQogICAgcGhvdG9tZXRyeSwgYW5kIGFuIGFwZXJ0dXJlIGlzIG5vdCBtb3ZlZCBieSBiZWluZyB3cm9uZy4gQSBjaXJjbGUgb2YgcmFkaXVzIDYKICAgIHBsYWNlZCAyLjQgcGl4ZWxzIG9mZiBjZW50cmUgbWlzc2VzIGEgZ29vZCBwYXJ0IG9mIHRoZSBzdGFyLCBhbmQgbWlzc2VzIGEKICAgIGRpZmZlcmVudCBwYXJ0IG9mIGV2ZXJ5IHN0YXIuIFRoZSBzdWJ0cmFjdGlvbiBmYWN0b3IgYnVpbHQgZnJvbSB0aG9zZSBmbHV4ZXMKICAgIGlzIHdyb25nLCBhbmQgdGhlIHN0YXIgd2lkdGhzIG1lYXN1cmVkIGFyb3VuZCB0aG9zZSBwb2ludHMgYXJlIG5vdCB0aGUgd2lkdGhzCiAgICBvZiBzdGFycy4gQm90aCBvZiB0aG9zZSB3ZXJlIHdyb25nIGhlcmUsIGFuZCBib3RoIHNob3dlZCBpbiB0aGUgcGljdHVyZS4KCiAgICBUaGUgb25lIHRoaW5nIHRoYXQgd2FzIHJpZ2h0IHdhcyB0aGUgZnJhbWUgcmVnaXN0cmF0aW9uLCBhbmQgb25seSBiZWNhdXNlIGl0CiAgICBjZW50cm9pZHMgaW5zaWRlIGEgYm94IGJlZm9yZSBmaXR0aW5nIC0gd2hpY2ggaXMgZXhhY3RseSB3aGF0IHRoaXMgZG9lcywgb25jZSwKICAgIGZvciBldmVyeW9uZSBlbHNlLgoKICAgIFdIWSBUSEUgU0tZIElTIFRBS0VOIE9GRiBGSVJTVAogICAgQSBjZW50cmUgb2YgbGlnaHQgaXMgYSB3ZWlnaHRlZCBhdmVyYWdlLCBhbmQgdGhlIHNreSBpcyBwYXJ0IG9mIHRoZSB3ZWlnaHQuIEluCiAgICBhIDE1IHBpeGVsIGJveCBvbiBhIGZyYW1lIHdoZXJlIHRoZSBza3kgc2l0cyBhdCBhIGZldyBodW5kcmVkIGNvdW50cywgdGhlIHR3bwogICAgaHVuZHJlZCBza3kgcGl4ZWxzIG91dHdlaWdoIHRoZSBzdGFyLCBhbmQgdGhlaXIgY2VudHJlIG9mIGxpZ2h0IGlzIHRoZSBtaWRkbGUKICAgIG9mIHRoZSBib3guIFNvIHRoZSBhbnN3ZXIgY29tZXMgYmFjayBhcyB2ZXJ5IG5lYXJseSB0aGUgcG9zaXRpb24gaXQgd2FzIGdpdmVuLAogICAgd2hhdGV2ZXIgdGhlIHN0YXIgaXMgZG9pbmcgLSBpdCBsb29rcyBsaWtlIGEgc3VjY2Vzc2Z1bCBtZWFzdXJlbWVudCBhbmQgaXMgbm90CiAgICBvbmUuIE1lYXN1cmVkIG9uIE0zMzogY2VudHJvaWRpbmcgdGhlIHJhdyBzdGFtcCBtb3ZlZCB0aGUgcG9zaXRpb25zIDAuNiBweCBhbmQKICAgIGxlZnQgdGhlIHN0YWNrZWQgc3RhciBmbGF0LXRvcHBlZCwgaXRzIGJyaWdodGVzdCByaW5nIGF0IHJhZGl1cyA0OyB0YWtpbmcgdGhlCiAgICBsb2NhbCBza3kgb2ZmIHRoZSBzdGFtcCBmaXJzdCBtb3ZlZCB0aGVtIDQuMyBweCBhbmQgZ2F2ZSBhIHN0YXIgdGhhdCBmYWxscyBmcm9tCiAgICAxMDIgY291bnRzIGF0IHRoZSBjZW50cmUgdG8gMTUgYXQgcmFkaXVzIDcsIHdoaWNoIGlzIHdoYXQgYSBzdGFyIGxvb2tzIGxpa2UuCgogICAgVGhlIHN0YW1wIGlzIGZsb29yZWQgYXQgemVybyBhZnRlciB0aGUgc2t5IGNvbWVzIG9mZiBzbyB0aGF0IG5lZ2F0aXZlIG5vaXNlCiAgICBjYW5ub3QgcHVsbCB0aGUgY2VudHJlIGFyb3VuZC4KCiAgICBQb3NpdGlvbnMgdGhhdCBjYW5ub3QgYmUgY2VudHJlZCwgb3IgdGhhdCBtb3ZlIGZ1cnRoZXIgdGhhbiBhIHBsYXRlIHNvbHV0aW9uCiAgICBwbGF1c2libHkgZXJycywgYXJlIGRyb3BwZWQgcmF0aGVyIHRoYW4gdHJ1c3RlZC4KICAgICIiIgogICAgcG9zaXRpb25zID0gbnAuYXNhcnJheShwb3NpdGlvbnMsIGR0eXBlPWZsb2F0KQogICAgaWYgbm90IGxlbihwb3NpdGlvbnMpOgogICAgICAgIHJldHVybiBwb3NpdGlvbnMsIHsibl9pbiI6IDAsICJuX291dCI6IDB9CgogICAgbnksIG54ID0gZGF0YS5zaGFwZQogICAgaGFsZiA9IGJveCAvLyAyCiAgICBlZGdlID0gYm94CiAgICBpbnNpZGUgPSAoKHBvc2l0aW9uc1s6LCAwXSA+IGVkZ2UpICYgKHBvc2l0aW9uc1s6LCAwXSA8IG54IC0gZWRnZSkgJgogICAgICAgICAgICAgIChwb3NpdGlvbnNbOiwgMV0gPiBlZGdlKSAmIChwb3NpdGlvbnNbOiwgMV0gPCBueSAtIGVkZ2UpKQogICAgcCA9IHBvc2l0aW9uc1tpbnNpZGVdCiAgICBpZiBub3QgbGVuKHApOgogICAgICAgIHJldHVybiBucC5lbXB0eSgoMCwgMikpLCB7Im5faW4iOiBsZW4ocG9zaXRpb25zKSwgIm5fb3V0IjogMCwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICJ3aHkiOiAibm9uZSBvZiB0aGVtIGZhbGwgaW5zaWRlIHRoZSBmcmFtZSJ9CgogICAgZm91bmQgPSBfY2VudHJvaWRfbG9jYWxseShkYXRhLCBwLCBib3gpCgogICAgZHgsIGR5ID0gZm91bmRbOiwgMF0gLSBwWzosIDBdLCBmb3VuZFs6LCAxXSAtIHBbOiwgMV0KICAgIG1vdmVkID0gbnAuaHlwb3QoZHgsIGR5KQogICAgZ29vZCA9IG5wLmlzZmluaXRlKG1vdmVkKSAmIChtb3ZlZCA8IG1heF9tb3ZlX3B4KQogICAgb3V0ID0gZm91bmRbZ29vZF0KICAgIHJldHVybiBvdXQsIHsibl9pbiI6IGludChsZW4ocG9zaXRpb25zKSksICJuX291dCI6IGludChnb29kLnN1bSgpKSwKICAgICAgICAgICAgICAgICAibWVkaWFuX21vdmUiOiBmbG9hdChucC5tZWRpYW4obW92ZWRbZ29vZF0pKSBpZiBnb29kLmFueSgpIGVsc2UgMC4wLAogICAgICAgICAgICAgICAgICJtZWRpYW5fZHgiOiBmbG9hdChucC5tZWRpYW4oZHhbZ29vZF0pKSBpZiBnb29kLmFueSgpIGVsc2UgMC4wLAogICAgICAgICAgICAgICAgICJtZWRpYW5fZHkiOiBmbG9hdChucC5tZWRpYW4oZHlbZ29vZF0pKSBpZiBnb29kLmFueSgpIGVsc2UgMC4wfQoKZGVmIGRlc2NyaWJlX2NlbnRyaW5nKGRpYWcpOgogICAgIiIiV2hhdCB0aGUgY2VudHJpbmcgbW92ZWQsIGluIGEgbGluZS4iIiIKICAgIGlmIGRpYWcuZ2V0KCJ3aHkiKToKICAgICAgICByZXR1cm4gIiAgY2F0YWxvZ3VlIHBvc2l0aW9ucyBsZWZ0IGFzIHRoZXkgYXJlICh7fSkiLmZvcm1hdChkaWFnWyJ3aHkiXSkKICAgIHJldHVybiAoIiAge30gb2Yge30gY2F0YWxvZ3VlIHN0YXJzIGZvdW5kIGFuZCBjZW50cmVkOyB0aGV5IHNhdCBhIG1lZGlhbiBvZiAiCiAgICAgICAgICAgICJ7Oi4yZn0gcHggZnJvbSB0aGUgY2F0YWxvZ3VlIHBvc2l0aW9uICh7OisuMmZ9LCB7OisuMmZ9KSIuZm9ybWF0KAogICAgICAgICAgICAgICAgZGlhZ1sibl9vdXQiXSwgZGlhZ1sibl9pbiJdLCBkaWFnWyJtZWRpYW5fbW92ZSJdLAogICAgICAgICAgICAgICAgZGlhZ1sibWVkaWFuX2R4Il0sIGRpYWdbIm1lZGlhbl9keSJdKSkKCmRlZiBjb21iaW5lX2V4cG9zdXJlcyhwYXRocywgZG9ub3JfcGF0aD1Ob25lLCBsYWJlbD0iIiwgYmFuZF9yb3dzPTUxMiwKICAgICAgICAgICAgICAgICAgICAgIHBvc2l0aW9ucz1Ob25lKToKICAgICIiIlB1dCBzZXZlcmFsIGV4cG9zdXJlcyBvZiBvbmUgZmlsdGVyIG9udG8gb25lIGdyaWQgYW5kIHRha2UgdGhlaXIgbWVkaWFuLgoKICAgIFdIWSBDT01CSU5FIEFUIEFMTAogICAgQSBuaWdodCdzIG9ic2VydmluZyBvZiBvbmUgZmlsdGVyIGlzIHNldmVyYWwgZXhwb3N1cmVzLiBUYWtpbmcgb25lIGFuZAogICAgZGlzY2FyZGluZyB0aGUgcmVzdCB0aHJvd3MgYXdheSBtb3N0IG9mIHdoYXQgd2FzIGdhdGhlcmVkOiBzaXggZnJhbWVzIGNvbWJpbmVkCiAgICBoYXZlIG5vaXNlIGxvd2VyIGJ5IHJvdWdobHkgdGhlIHNxdWFyZSByb290IG9mIHNpeCwgYSBmYWN0b3Igb2YgMi40LCBhbmQgaXQgaXMKICAgIHRoZSBmYWludGVzdCByZWdpb25zIC0gdGhlIG9uZXMgbmVhciB0aGUgZGV0ZWN0aW9uIGxpbWl0LCB3aGVyZSB0aGUgYW5zd2VyIGlzCiAgICBhY3R1YWxseSBkZWNpZGVkIC0gdGhhdCBhcHBlYXIgb3IgdmFuaXNoIG9uIGV4YWN0bHkgdGhhdCBkaWZmZXJlbmNlLgoKICAgIFdIWSBBIE1FRElBTgogICAgQmVjYXVzZSBpdCBpcyByb2J1c3QsIGNoZWFwLCBhbmQgdGhlIGFsdGVybmF0aXZlIGlzIG5vdCB3b3J0aCB3aGF0IGl0IGNvc3RzLgogICAgTWVhc3VyZWQgb24gdGhpcyBkYXRhOgoKICAgICAgICBzaXggZnJhbWVzICAgICAgICBub2lzZSAgICB0aW1lICAgICB3b3JraW5nIG1lbW9yeQogICAgICAgIG1lZGlhbiAgICAgICAgICAgICAxLjE2ICAgIDAuNDUgcyAgICAgMS40eCB0aGUgc3RhY2sKICAgICAgICBjbGlwcGVkIG1lYW4gICAgICAgMS4xMCAgICAwLjk3IHMgICAgIDMuM3ggdGhlIHN0YWNrCiAgICAgICAgcGxhaW4gYXZlcmFnZSAgICAgIDIuNjAqICAgMC4wMiBzICAgICAwLjJ4IHRoZSBzdGFjawogICAgICAgICAgICAgICAgICAgICAgICAgICAoKiB3aXRoIG9uZSBjb3NtaWMgcmF5IGluIGEgaHVuZHJlZCBwaXhlbHMpCgogICAgVGhlIGNsaXBwZWQgbWVhbiBidXlzIGZpdmUgdG8gc2V2ZW4gcGVyY2VudCBpbiBub2lzZSAtIGFib3V0IHdoYXQgYSBzZXZlbnRoCiAgICBleHBvc3VyZSB3b3VsZCBnaXZlIC0gZm9yIHR3aWNlIHRoZSB0aW1lIGFuZCB0d28gYW5kIGEgaGFsZiB0aW1lcyB0aGUgbWVtb3J5LgogICAgQSBwbGFpbiBhdmVyYWdlIGlzIG5vdCBhbiBvcHRpb24gYXQgYWxsOiBvbmUgY29zbWljIHJheSBydWlucyB0aGUgcGl4ZWwuIFNvCiAgICB0aGUgbWVkaWFuIGl0IGlzLCBhbmQgdGhlcmUgaXMgbm8gc3dpdGNoLCBiZWNhdXNlIHRoZXJlIGlzIG5vIGNhc2UgaGVyZSB3aGVyZQogICAgdGhlIG90aGVyIGFuc3dlciBpcyB0aGUgcmlnaHQgb25lLgoKICAgIERvbmUgYSBiYW5kIG9mIHJvd3MgYXQgYSB0aW1lLCBzbyB0aGUgbWVtb3J5IG5lZWRlZCBkb2VzIG5vdCBncm93IHdpdGggdGhlCiAgICBudW1iZXIgb2YgZXhwb3N1cmVzOiB0d2VudHkgZnJhbWVzIGNvbWJpbmUgaW4gdGhlIHNhbWUgc3BhY2UgYXMgdGhyZWUuCgogICAgVGhlIGZpcnN0IGV4cG9zdXJlIHNldHMgdGhlIGdyaWQ7IHRoZSByZXN0IGFyZSBicm91Z2h0IHRvIGl0IHRocm91Z2ggdGhlaXIKICAgIHBsYXRlIHNvbHV0aW9ucyBhbmQgdGhlbiBoYXZlIHRoZSBsZWZ0b3ZlciBzdWItcGl4ZWwgc2hpZnQgbWVhc3VyZWQgZnJvbSB0aGUKICAgIHN0YXJzIGFuZCB0YWtlbiBvdXQsIGV4YWN0bHkgYXMgdGhlIHNlcGFyYXRlIGZpbHRlcnMgYXJlLgogICAgIiIiCiAgICBpbXBvcnQgbnVtcHkgYXMgbnAKCiAgICByZWZlcmVuY2UsIHJlZl93Y3MsIHJlZl9icG0sIHJlZl9pbmZvID0gbG9hZF9mcmFtZShwYXRoc1swXSwgZG9ub3JfcGF0aCkKICAgIGlmIGxlbihwYXRocykgPT0gMToKICAgICAgICByZXR1cm4gcmVmZXJlbmNlLCByZWZfd2NzLCByZWZfYnBtLCByZWZfaW5mbwoKICAgIHByaW50KCJDb21iaW5pbmcge30ge30gZXhwb3N1cmVzIi5mb3JtYXQobGVuKHBhdGhzKSwgbGFiZWwgb3IgImZyYW1lcyIpKQogICAgIyBFdmVyeSBleHBvc3VyZSBpcyBzY2FsZWQgdG8gdGhlIGV4cG9zdXJlIHRpbWUgb2YgdGhlIGZpcnN0IGJlZm9yZSB0aGUKICAgICMgbWVkaWFuIGlzIHRha2VuLiBBIG1lZGlhbiBvZiBhIDYwIHMgYW5kIGEgMTIwIHMgZnJhbWUgdGhhdCB3ZXJlIG5vdAogICAgIyBzY2FsZWQgaXMgbmVpdGhlciBvZiB0aGVtOiBlYWNoIHBpeGVsIHRha2VzIHdoaWNoZXZlciB2YWx1ZSBoYXBwZW5zIHRvCiAgICAjIGZhbGwgaW4gdGhlIG1pZGRsZS4gQmVmb3JlIHNjYWxpbmcsIHBpeGVscyBhdCB0aGUgZGV0ZWN0b3IncyBzYXR1cmF0aW9uCiAgICAjIGxldmVsIGFyZSBibGFua2VkLCBzbyB0aGF0IGEgc3RhciBzYXR1cmF0ZWQgaW4gdGhlIGxvbmcgZXhwb3N1cmVzIGlzCiAgICAjIG1lYXN1cmVkIGZyb20gdGhlIHNob3J0IG9uZXMuCiAgICByZWZfZXhwdGltZSA9IGZsb2F0KHJlZl9pbmZvLmdldCgiZXhwdGltZSIpIG9yIDEuMCkKICAgIHNhdF9sZXZlbHMsIHNhdF9tYXNrcyA9IFtdLCBbXQoKICAgIGRlZiBzY2FsZWQoZnJhbWUsIGluZm8sIG5hbWUpOgogICAgICAgIGV4cHRpbWUgPSBmbG9hdChpbmZvLmdldCgiZXhwdGltZSIpIG9yIHJlZl9leHB0aW1lKQogICAgICAgIHJhdGlvID0gcmVmX2V4cHRpbWUgLyBleHB0aW1lCiAgICAgICAgc2F0ID0gaW5mby5nZXQoInNhdHVyYXRlIikKICAgICAgICBpZiBzYXQ6CiAgICAgICAgICAgIGZyYW1lID0gbnAuYXJyYXkoZnJhbWUsIGR0eXBlPW5wLmZsb2F0MzIsIGNvcHk9VHJ1ZSkKICAgICAgICAgICAgd2l0aCBucC5lcnJzdGF0ZShpbnZhbGlkPSJpZ25vcmUiKToKICAgICAgICAgICAgICAgIGhpdCA9IGZyYW1lID49IFNBVFVSQVRJT05fRlJBQ1RJT04gKiBzYXQKICAgICAgICAgICAgZnJhbWVbaGl0XSA9IG5wLm5hbgogICAgICAgICAgICBzYXRfbWFza3MuYXBwZW5kKGhpdCkKICAgICAgICAgICAgc2F0X2xldmVscy5hcHBlbmQoc2F0ICogcmF0aW8pCiAgICAgICAgZWxzZToKICAgICAgICAgICAgc2F0X2xldmVscy5hcHBlbmQoTm9uZSkKICAgICAgICBpZiBhYnMocmF0aW8gLSAxLjApID4gMWUtNjoKICAgICAgICAgICAgcHJpbnQoIiAge306IHs6LjBmfSBzLCBzY2FsZWQgYnkgezouM2Z9IHRvIHs6LjBmfSBzIi5mb3JtYXQoCiAgICAgICAgICAgICAgICBuYW1lLCBleHB0aW1lLCByYXRpbywgcmVmX2V4cHRpbWUpKQogICAgICAgICAgICBmcmFtZSA9IGZyYW1lICogbnAuZmxvYXQzMihyYXRpbykKICAgICAgICByZXR1cm4gZnJhbWUKCiAgICBzdGFjayA9IFtzY2FsZWQocmVmZXJlbmNlLCByZWZfaW5mbywgb3MucGF0aC5iYXNlbmFtZShwYXRoc1swXSkpXQogICAgdG90YWxfZXhwdGltZSA9IGZsb2F0KHJlZl9pbmZvLmdldCgiZXhwdGltZSIpIG9yIDAuMCkKCiAgICBmb3IgcGF0aCBpbiBwYXRoc1sxOl06CiAgICAgICAgZGF0YSwgd2NzLCBfLCBpbmZvID0gbG9hZF9mcmFtZShwYXRoLCBkb25vcl9wYXRoKQogICAgICAgIG5hbWUgPSBvcy5wYXRoLmJhc2VuYW1lKHBhdGgpCgogICAgICAgICMgRm91ciBicmlnaHQsIHdpZGVseSBzZXBhcmF0ZWQgc3RhcnMgZml4IHRoZSBzaGlmdCwgdGhlIHJvdGF0aW9uIGFuZCB0aGUKICAgICAgICAjIHNjYWxlIGJldHdlZW4gdHdvIGZyYW1lcyBvZiB0aGUgc2FtZSBmaWVsZC4gR29pbmcgdGhyb3VnaCBib3RoIHBsYXRlCiAgICAgICAgIyBzb2x1dGlvbnMgaW5zdGVhZCBjb3N0cyBhIG1pbnV0ZSBhIGZyYW1lIGFuZCBhcnJpdmVzIGF0IHRoZSBzYW1lIHBsYWNlLgogICAgICAgIHBsYWNlZCA9IEZhbHNlCiAgICAgICAgaWYgcG9zaXRpb25zIGlzIG5vdCBOb25lIGFuZCBsZW4ocG9zaXRpb25zKToKICAgICAgICAgICAgdHJhbnNmb3JtLCBhZGlhZyA9IGFuY2hvcl90cmFuc2Zvcm0ocmVmZXJlbmNlLCBkYXRhLCBwb3NpdGlvbnMpCiAgICAgICAgICAgIGlmIHRyYW5zZm9ybSBpcyBub3QgTm9uZToKICAgICAgICAgICAgICAgIGRhdGEgPSBhcHBseV90cmFuc2Zvcm0oZGF0YSwgdHJhbnNmb3JtLCByZWZlcmVuY2Uuc2hhcGUpCiAgICAgICAgICAgICAgICBwcmludCgiICAiICsgbmFtZSkKICAgICAgICAgICAgICAgIHByaW50KCIgICIgKyBkZXNjcmliZV9hbmNob3JzKGFkaWFnKSkKICAgICAgICAgICAgICAgIHBsYWNlZCA9IFRydWUKCiAgICAgICAgaWYgbm90IHBsYWNlZDoKICAgICAgICAgICAgaWYgd2NzIGlzIG5vdCBOb25lIGFuZCByZWZfd2NzIGlzIG5vdCBOb25lOgogICAgICAgICAgICAgICAgZGF0YSA9IGFsaWduX3RvKHJlZl93Y3MsIHJlZmVyZW5jZS5zaGFwZSwgZGF0YSwgd2NzKQogICAgICAgICAgICBlbGlmIGRhdGEuc2hhcGUgIT0gcmVmZXJlbmNlLnNoYXBlOgogICAgICAgICAgICAgICAgcHJpbnQoIiAgc2tpcHBpbmcge30gLSBubyBwbGF0ZSBzb2x1dGlvbiBhbmQgYSBkaWZmZXJlbnQgIgogICAgICAgICAgICAgICAgICAgICAgInNoYXBlIi5mb3JtYXQobmFtZSkpCiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICBkYXRhID0gcmVmaW5lX2FsaWdubWVudChyZWZlcmVuY2UsIGRhdGEsICIgICIgKyBuYW1lLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBwb3NpdGlvbnM9cG9zaXRpb25zKQoKICAgICAgICBzdGFjay5hcHBlbmQoc2NhbGVkKGRhdGEsIGluZm8sIG5hbWUpKQogICAgICAgIHRvdGFsX2V4cHRpbWUgKz0gZmxvYXQoaW5mby5nZXQoImV4cHRpbWUiKSBvciAwLjApCgogICAgbnksIG54ID0gcmVmZXJlbmNlLnNoYXBlCiAgICBjb21iaW5lZCA9IG5wLmVtcHR5KChueSwgbngpLCBkdHlwZT1ucC5mbG9hdDMyKQogICAgZm9yIHkwIGluIHJhbmdlKDAsIG55LCBiYW5kX3Jvd3MpOgogICAgICAgIHkxID0gbWluKHkwICsgYmFuZF9yb3dzLCBueSkKICAgICAgICBiYW5kID0gbnAuc3RhY2soW2ZyYW1lW3kwOnkxXSBmb3IgZnJhbWUgaW4gc3RhY2tdLCBheGlzPTApCiAgICAgICAgd2l0aCB3YXJuaW5ncy5jYXRjaF93YXJuaW5ncygpOgogICAgICAgICAgICB3YXJuaW5ncy5zaW1wbGVmaWx0ZXIoImlnbm9yZSIpCiAgICAgICAgICAgIGNvbWJpbmVkW3kwOnkxXSA9IG5wLm5hbm1lZGlhbihiYW5kLCBheGlzPTApCiAgICAgICAgICAgICMgQSBwaXhlbCBzYXR1cmF0ZWQgaW4gZXZlcnkgZXhwb3N1cmUgaGFzIG5vdGhpbmcgbGVmdCB0byB0YWtlIHRoZQogICAgICAgICAgICAjIG1lZGlhbiBvZi4gSXQgaXMgc2V0IHRvIHRoZSBzYXR1cmF0aW9uIGxldmVsLCBzbyB0aGF0IGl0IHN0YXlzIGEKICAgICAgICAgICAgIyBicmlnaHQgcGl4ZWwgdGhlIHBob3RvbWV0cnkgZmxhZ3MgYXMgc2F0dXJhdGVkLCBub3QgYSBob2xlLgogICAgICAgICAgICBpZiBzYXRfbWFza3M6CiAgICAgICAgICAgICAgICBob2xlID0gfm5wLmlzZmluaXRlKGNvbWJpbmVkW3kwOnkxXSkKICAgICAgICAgICAgICAgIGhvbGUgJj0gbnAuYW55KG5wLnN0YWNrKFttW3kwOnkxXSBmb3IgbSBpbiBzYXRfbWFza3NdLCBheGlzPTApLCBheGlzPTApCiAgICAgICAgICAgICAgICBpZiBob2xlLmFueSgpOgogICAgICAgICAgICAgICAgICAgIGNvbWJpbmVkW3kwOnkxXVtob2xlXSA9IG1heChzIGZvciBzIGluIHNhdF9sZXZlbHMgaWYgcykKCiAgICBwcmludCgiICBtZWRpYW4gb2Yge30gZXhwb3N1cmVzLCB0b3RhbCBleHBvc3VyZSB7Oi4wZn1zLCBub2lzZSBsb3dlciBieSBhYm91dCAiCiAgICAgICAgICAiYSBmYWN0b3Igb2YgezouMWZ9Ii5mb3JtYXQobGVuKHN0YWNrKSwgdG90YWxfZXhwdGltZSwgbGVuKHN0YWNrKSAqKiAwLjUpKQoKICAgICMgVGhlIG1lZGlhbiBzaXRzIGF0IHRoZSBsZXZlbCBvZiBPTkUgZXhwb3N1cmUgb2YgdGhlIGZpcnN0IGZyYW1lJ3MgbGVuZ3RoLAogICAgIyBzbyB0aGF0IGlzIHdoYXQgRVhQVElNRSBzYXlzLiBJdCB1c2VkIHRvIHNheSB0aGUgdG90YWwsIHdoaWNoIG1hZGUgdGhlCiAgICAjIGhlYWRlciBjbGFpbSBzZXZlcmFsIHRpbWVzIHRoZSBleHBvc3VyZSB0aGUgcGl4ZWwgdmFsdWVzIGhhdmUuIFRoZSB0b3RhbAogICAgIyBpcyBrZXB0IGluIFRPVEVYUCwgYW5kIE5DT01CSU5FIHNheXMgaG93IG1hbnkgZnJhbWVzIHdlbnQgaW4gLSB0aGUgbm9pc2UKICAgICMgb2YgYSBtZWRpYW4gb2YgTiBmcmFtZXMgaXMgdGhhdCBvZiBvbmUgZnJhbWUgd2l0aCBOIHggMi9waSB0aW1lcyB0aGUgZ2Fpbi4KICAgIGluZm8gPSBkaWN0KHJlZl9pbmZvKQogICAgaW5mb1siZXhwdGltZSJdID0gcmVmX2V4cHRpbWUKICAgIGluZm9bInRvdGFsX2V4cHRpbWUiXSA9IHRvdGFsX2V4cHRpbWUKICAgIGluZm9bIm5jb21iaW5lIl0gPSBsZW4oc3RhY2spCiAgICBpbmZvWyJjb21idHlwZSJdID0gIm1lZGlhbiIKICAgIGlmIGFsbChzIGZvciBzIGluIHNhdF9sZXZlbHMpOgogICAgICAgICMgUGl4ZWxzIGJlbG93IHRoZSBsZXZlbCBvZiB0aGUgc2hvcnRlc3QgZXhwb3N1cmUgY2FtZSBmcm9tIGF0IGxlYXN0CiAgICAgICAgIyBvbmUgdW5zYXR1cmF0ZWQgZnJhbWUuCiAgICAgICAgaW5mb1sic2F0dXJhdGUiXSA9IG1heChzYXRfbGV2ZWxzKQogICAgZWxzZToKICAgICAgICBpbmZvWyJzYXR1cmF0ZSJdID0gTm9uZQogICAgIyBTY2FsaW5nIGEgbG9uZ2VyIGV4cG9zdXJlIGRvd24gbG93ZXJzIGl0cyBub2lzZSBhbG9uZyB3aXRoIGl0cyBjb3VudHMsCiAgICAjIGFzIGEgaGlnaGVyIGdhaW4gd291bGQuIFdyaXR0ZW4gc28gdGhhdCBHQUlOIHggTkNPTUJJTkUgaXMgdGhlIGdhaW4gb2YgdGhlCiAgICAjIHRvdGFsIGV4cG9zdXJlIGV4cHJlc3NlZCBhdCB0aGlzIGZyYW1lJ3MgbGV2ZWwuCiAgICBpZiBpbmZvLmdldCgiZ2FpbiIpOgogICAgICAgIGluZm9bImdhaW4iXSA9IGZsb2F0KGluZm9bImdhaW4iXSkgKiB0b3RhbF9leHB0aW1lIC8gcmVmX2V4cHRpbWUgLyBsZW4oc3RhY2spCiAgICByZXR1cm4gY29tYmluZWQsIHJlZl93Y3MsIHJlZl9icG0sIGluZm8KClNBVFVSQVRJT05fRlJBQ1RJT04gPSAwLjggICAjIGEgcGl4ZWwgYWJvdmUgdGhpcyBmcmFjdGlvbiBvZiBTQVRVUkFURSBjb3VudHMgYXMgc2F0dXJhdGVkCgpkZWYgbmV3X3J1bl9mb2xkZXIoYmFzZSwgdG9vbCwgZ2FsYXh5KToKICAgICIiIkEgZm9sZGVyIG9mIGl0cyBvd24gZm9yIHRoaXMgcnVuOiByZXN1bHRfPHRvb2w+XzxnYWxheHk+XzxkYXRlPl88dGltZT4uCgogICAgRXZlcnkgcnVuIHVzZWQgdG8gd3JpdGUgaW50byB0aGUgc2FtZSBwbGFjZSB1bmRlciB0aGUgc2FtZSBuYW1lcywgc28gYQogICAgc2Vjb25kIHJ1biBvdmVyd3JvdGUgdGhlIGZpcnN0LCBhbmQgYSBydW4gdGhhdCBzdG9wcGVkIGhhbGYgd2F5IGxlZnQgdGhlCiAgICBwcmV2aW91cyBydW4ncyBmaWxlcyBzaXR0aW5nIGJlc2lkZSBpdHMgb3duLCBsb29raW5nIGN1cnJlbnQuIEEgbmV3IGZvbGRlcgogICAgZWFjaCB0aW1lIG1lYW5zIG5vdGhpbmcgYSBydW4gcmVhZHMgb3Igc2hvd3MgY2FuIGJlIGxlZnQgb3ZlciBmcm9tIGFub3RoZXIuCiAgICAiIiIKICAgIGltcG9ydCBkYXRldGltZQogICAgbmFtZSA9IHJlLnN1YihyIlteQS1aYS16MC05Xy4tXSsiLCAiXyIsIChnYWxheHkgb3IgImdhbGF4eSIpLnN0cmlwKCkpIG9yICJnYWxheHkiCiAgICBzdGFtcCA9IGRhdGV0aW1lLmRhdGV0aW1lLm5vdygpLnN0cmZ0aW1lKCIlWSVtJWRfJUglTSVTIikKICAgIHBhdGggPSBvcy5wYXRoLmpvaW4oYmFzZSBvciBvcy5nZXRjd2QoKSwgInJlc3VsdF97fV97fV97fSIuZm9ybWF0KHRvb2wsIG5hbWUsIHN0YW1wKSkKICAgIG9zLm1ha2VkaXJzKHBhdGgsIGV4aXN0X29rPVRydWUpCiAgICByZXR1cm4gcGF0aAoKZGVmIG91dHB1dF9wcmVmaXgoZ2l2ZW4sIGZhbGxiYWNrX2RpciwgZ2FsYXh5KToKICAgICIiIlR1cm4gd2hhdCB0aGUgcGVyc29uIHR5cGVkIGludG8gYSBzdGVtIGZvciB0aGUgb3V0cHV0IGZpbGVzLgoKICAgIEEgZm9sZGVyIGlzIGFjY2VwdGVkIGFzIHdlbGwgYXMgYSBzdGVtLCBiZWNhdXNlIGEgZm9sZGVyIGlzIHdoYXQgcGVvcGxlIHR5cGU6CiAgICBhc2tlZCB3aGVyZSB0byBwdXQgdGhlIG91dHB1dCwgdGhyZWUgcnVucyBpbiBhIHJvdyB3ZXJlIGdpdmVuIHRoZSBwYXRoIG9mIHRoZQogICAgZnJhbWVzJyBvd24gZm9sZGVyLCBhbmQgdGhlIGZpbGVzIGxhbmRlZCBiZXNpZGUgaXQgbmFtZWQgYWZ0ZXIgaXQuIElmIHdoYXQgd2FzCiAgICBnaXZlbiBpcyBhIGRpcmVjdG9yeSwgdGhlIGZpbGVzIGdvIGluc2lkZSBpdCwgbmFtZWQgYWZ0ZXIgdGhlIGdhbGF4eS4KICAgICIiIgogICAgbmFtZSA9IChnYWxheHkgb3IgImdhbGF4eSIpLnJlcGxhY2UoIiAiLCAiXyIpCiAgICBpZiBnaXZlbjoKICAgICAgICBnaXZlbiA9IHN0cihnaXZlbikuc3RyaXAoKS5zdHJpcCgnIicpLnN0cmlwKCInIikKICAgICAgICBpZiBvcy5wYXRoLmlzZGlyKGdpdmVuKToKICAgICAgICAgICAgcmV0dXJuIG9zLnBhdGguam9pbihnaXZlbiwgbmFtZSkKICAgICAgICBwYXJlbnQgPSBvcy5wYXRoLmRpcm5hbWUoZ2l2ZW4pCiAgICAgICAgaWYgcGFyZW50IGFuZCBub3Qgb3MucGF0aC5pc2RpcihwYXJlbnQpOgogICAgICAgICAgICBvcy5tYWtlZGlycyhwYXJlbnQsIGV4aXN0X29rPVRydWUpCiAgICAgICAgcmV0dXJuIGdpdmVuCiAgICByZXR1cm4gb3MucGF0aC5qb2luKGZhbGxiYWNrX2RpciBvciAiLiIsIG5hbWUpCgpkZWYgc2F2ZV9mcmFtZShwYXRoLCBkYXRhLCB3Y3M9Tm9uZSwgaW5mbz1Ob25lLCBoaXN0b3J5PSgpKToKICAgICIiIldyaXRlIGEgZnJhbWUgdGhpcyB0b29sIG1hZGUsIHdpdGggYSBoZWFkZXIgd29ydGggaGF2aW5nLgoKICAgIFRoZSBjb250aW51dW0tc3VidHJhY3RlZCBpbWFnZSBpcyB0aGUgb25lIHRoZSB3aG9sZSBtZWFzdXJlbWVudCByZXN0cyBvbiwgYW5kCiAgICB1bnRpbCBub3cgaXQgZXhpc3RlZCBvbmx5IGluc2lkZSBhIHBsb3QuIEFueW9uZSB3YW50aW5nIHRvIGNoZWNrIHRoZQogICAgc3VidHJhY3Rpb24gYnkgZXllIC0gd2hpY2ggaXMgZXhhY3RseSB3aGF0IHRoZSB0b29sIGFza3MgZm9yIHdoZW4gdGhlIHN0YXJzCiAgICBkaXNhZ3JlZSBhYm91dCB0aGUgZmFjdG9yIC0gaGFkIG5vIGZpbGUgdG8gb3Blbi4gSXQgY2FycmllcyB0aGUgcGxhdGUgc29sdXRpb24KICAgIHNvIGl0IGxpbmVzIHVwIHdpdGggdGhlIG9yaWdpbmFscyBpbiBhbnkgdmlld2VyLCBhbmQgYSBoaXN0b3J5IHNheWluZyB3aGF0IGl0CiAgICBpcyBhbmQgaG93IGl0IHdhcyBtYWRlLgogICAgV0hBVCBJUyBET05FIFRPIFRIRSBQSVhFTFMgQkVGT1JFIFRIRVkgQVJFIFdSSVRURU4KICAgIEJsYW5rIHBpeGVscyBhcmUgd3JpdHRlbiBhcyB6ZXJvIHJhdGhlciB0aGFuIGFzIE5hTi4gTmFOIGlzIGxlZ2FsIEZJVFMgYW5kCiAgICBhc3Ryb3B5IHJlYWRzIGl0IGhhcHBpbHksIGJ1dCBpdCBpcyBhIGZsb2F0aW5nLXBvaW50IHZhbHVlIHRoYXQgb2xkZXIgcmVhZGVycwogICAgcHJlZGF0ZTogQUlQNFdpbiB3b3VsZCBub3Qgb3BlbiBhIGZyYW1lIHdyaXR0ZW4gZnJvbSBoZXJlLCBhbmQgYSBzdWJ0cmFjdGVkCiAgICBNODMgZnJhbWUgY2FycmllZCBuaW5lIHRob3VzYW5kIG9mIHRoZW0sIGF0IHRoZSBlZGdlcyB0aGUgcmVzYW1wbGluZyBjb3VsZCBub3QKICAgIHJlYWNoLiBBIGZpbGUgdGhhdCBvbmx5IHRoaXMgcGlwZWxpbmUgY2FuIG9wZW4gaXMgbm90IG11Y2ggb2YgYW4gb3V0cHV0LgoKICAgIFdIWSBUSEUgUExBVEUgU09MVVRJT04gSVMgV1JJVFRFTiBUV0lDRQogICAgYXN0cm9weSB3cml0ZXMgYSByb3RhdGlvbiBhcyBhIFBDIG1hdHJpeCB3aXRoIGEgc2VwYXJhdGUgQ0RFTFQgc2NhbGUuIFRoYXQgaXMKICAgIHRoZSBjdXJyZW50IGNvbnZlbnRpb247IHRoZSBvbGRlciBvbmUgaXMgYSBzaW5nbGUgQ0QgbWF0cml4LCBhbmQgYSByZWFkZXIgdGhhdAogICAga25vd3Mgb25seSBDRCBzZWVzIGEgZnJhbWUgd2l0aCBubyB1c2FibGUgY29vcmRpbmF0ZXMuIEJvdGggYXJlIHdyaXR0ZW4gLSB0aGV5CiAgICBzYXkgdGhlIHNhbWUgdGhpbmcsIGFuZCBDRCA9IFBDIHggQ0RFTFQgaXMgZXhhY3QgLSBzbyBlaXRoZXIga2luZCBvZiByZWFkZXIKICAgIGZpbmRzIHdoYXQgaXQgZXhwZWN0cy4KICAgICIiIgogICAgZnJvbSBhc3Ryb3B5LmlvIGltcG9ydCBmaXRzCgogICAgaGVhZGVyID0gd2NzLnRvX2hlYWRlcigpIGlmIHdjcyBpcyBub3QgTm9uZSBlbHNlIGZpdHMuSGVhZGVyKCkKICAgIGlmIHdjcyBpcyBub3QgTm9uZToKICAgICAgICB0cnk6CiAgICAgICAgICAgIGNkID0gbnAuYXNhcnJheSh3Y3MucGl4ZWxfc2NhbGVfbWF0cml4LCBkdHlwZT1mbG9hdCkKICAgICAgICAgICAgZm9yIGkgaW4gKDEsIDIpOgogICAgICAgICAgICAgICAgZm9yIGogaW4gKDEsIDIpOgogICAgICAgICAgICAgICAgICAgIGhlYWRlclsiQ0R7fV97fSIuZm9ybWF0KGksIGopXSA9IGZsb2F0KGNkW2kgLSAxLCBqIC0gMV0pCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgcGFzcwogICAgaWYgaW5mbzoKICAgICAgICBpZiBpbmZvLmdldCgiZmlsdGVyIik6CiAgICAgICAgICAgIGhlYWRlclsiRklMVEVSIl0gPSBpbmZvWyJmaWx0ZXIiXQogICAgICAgIGlmIGluZm8uZ2V0KCJleHB0aW1lIik6CiAgICAgICAgICAgIGhlYWRlclsiRVhQVElNRSJdID0gZmxvYXQoaW5mb1siZXhwdGltZSJdKQogICAgICAgIGlmIGluZm8uZ2V0KCJvYmplY3QiKToKICAgICAgICAgICAgaGVhZGVyWyJPQkpFQ1QiXSA9IGluZm9bIm9iamVjdCJdCiAgICAgICAgaWYgaW5mby5nZXQoInBpeGVsX3NjYWxlIik6CiAgICAgICAgICAgIGhlYWRlclsiUElYU0NBTEUiXSA9IGZsb2F0KGluZm9bInBpeGVsX3NjYWxlIl0pCiAgICAgICAgZm9yIGtleSwgd29yZCBpbiAoKCJzYXR1cmF0ZSIsICJTQVRVUkFURSIpLCAoImdhaW4iLCAiR0FJTiIpLAogICAgICAgICAgICAgICAgICAgICAgICAgICgicmRub2lzZSIsICJSRE5PSVNFIiksICgidG90YWxfZXhwdGltZSIsICJUT1RFWFAiKSk6CiAgICAgICAgICAgIGlmIGluZm8uZ2V0KGtleSk6CiAgICAgICAgICAgICAgICBoZWFkZXJbd29yZF0gPSBmbG9hdChpbmZvW2tleV0pCiAgICAgICAgaWYgKGluZm8uZ2V0KCJuY29tYmluZSIpIG9yIDEpID4gMToKICAgICAgICAgICAgaGVhZGVyWyJOQ09NQklORSJdID0gaW50KGluZm9bIm5jb21iaW5lIl0pCiAgICAgICAgICAgIGhlYWRlclsiQ09NQlRZUEUiXSA9IHN0cihpbmZvLmdldCgiY29tYnR5cGUiKSBvciAibWVkaWFuIikKICAgICAgICBpZiBpbmZvLmdldCgiZGF0ZSIpOgogICAgICAgICAgICBoZWFkZXJbIkRBVEUtT0JTIl0gPSBzdHIoaW5mb1siZGF0ZSJdKQogICAgcGl4ZWxzID0gbnAuYXNhcnJheShkYXRhLCBkdHlwZT0iZmxvYXQzMiIpCiAgICBibGFuayA9IH5ucC5pc2Zpbml0ZShwaXhlbHMpCiAgICBpZiBibGFuay5hbnkoKToKICAgICAgICBwaXhlbHMgPSBucC53aGVyZShibGFuaywgbnAuZmxvYXQzMigwLjApLCBwaXhlbHMpCiAgICAgICAgaGVhZGVyWyJISVNUT1JZIl0gPSAoInt9IGJsYW5rIHBpeGVscyB3cml0dGVuIGFzIHplcm8sIG5vdCBOYU4sIHNvIHRoYXQgIgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICJvbGRlciByZWFkZXJzIGNhbiBvcGVuIHRoaXMiLmZvcm1hdChpbnQoYmxhbmsuc3VtKCkpKSkKICAgIGZvciBsaW5lIGluIGhpc3Rvcnk6CiAgICAgICAgaGVhZGVyWyJISVNUT1JZIl0gPSBsaW5lCgogICAgZml0cy5QcmltYXJ5SERVKGRhdGE9cGl4ZWxzLCBoZWFkZXI9aGVhZGVyKS53cml0ZXRvKHBhdGgsIG92ZXJ3cml0ZT1UcnVlKQogICAgcmV0dXJuIHBhdGgKCmRlZiBpbWFnZV9vcmllbnRhdGlvbih3Y3MpOgogICAgIiIiQW5nbGUgb2YgTm9ydGggaW4gdGhlIGltYWdlLCBhbmQgd2hldGhlciB0aGUgcGFyaXR5IGlzIGZsaXBwZWQuCgogICAgUmVhZGluZyB0aGlzIGZyb20gdGhlIFdDUyByZW1vdmVzIGEgd2hvbGUgY2xhc3Mgb2Ygc2lsZW50IGVycm9yLiBBc3N1bWluZwogICAgbm9ydGggaXMgdXAgd2hlbiBpdCBpcyBub3Qgcm90YXRlcyB0aGUgZGVwcm9qZWN0aW9uIGJ5IGFuIHVua25vd24gYW1vdW50LCBhbmQKICAgIG5vdGhpbmcgaW4gdGhlIG91dHB1dCBsb29rcyB3cm9uZy4KICAgICIiIgogICAgY2QgPSB3Y3MucGl4ZWxfc2NhbGVfbWF0cml4CiAgICBub3J0aCA9IG1hdGguZGVncmVlcyhtYXRoLmF0YW4yKGNkWzEsIDFdLCBjZFswLCAxXSkpICUgMzYwLjAKICAgIGVhc3QgPSBtYXRoLmRlZ3JlZXMobWF0aC5hdGFuMihjZFsxLCAwXSwgY2RbMCwgMF0pKSAlIDM2MC4wCiAgICBtaXJyb3JlZCA9ICgoZWFzdCAtIG5vcnRoKSAlIDM2MC4wKSA+IDE4MC4wCiAgICByZXR1cm4gbm9ydGgsIG1pcnJvcmVkCgpkZWYgZnJhbWVfZ2VvbWV0cnkocGF0aCwgZG9ub3JfcGF0aD1Ob25lKToKICAgICIiIldoZXJlIGEgZnJhbWUgcG9pbnRzIGFuZCBob3cgYmlnIGl0IGlzLCB3aXRob3V0IHJlYWRpbmcgaXRzIHBpeGVscy4KCiAgICBUaGUgY2F0YWxvZ3VlIGxvb2t1cHMgYWxsIG5lZWQgdGhpcyAtIHRoZSBnYWxheHkncyBjZW50cmUgaW4gdGhpcyBmcmFtZSwgdGhlCiAgICBzdGFycyBpbiBmcm9udCBvZiBpdCwgdGhlIGlzb3Bob3RhbCByYWRpdXMgaW4gcGl4ZWxzIC0gYW5kIHRoZXkgYXJlIG5lZWRlZAogICAgYmVmb3JlIHRoZSBmcmFtZXMgYXJlIGNvbWJpbmVkLCBub3QgYWZ0ZXIuIFJlYWRpbmcgYSBoZWFkZXIgaXMgaW5zdGFudCwgd2hpbGUKICAgIHJlYWRpbmcgYSA2MS1tZWdhcGl4ZWwgaW1hZ2UgaXMgbm90LCBzbyB0aGUgaGVhZGVyIGlzIHRyaWVkIGZpcnN0LiBBIGZyYW1lCiAgICB0aGF0IGxvc3QgaXRzIHBsYXRlIHNvbHV0aW9uIGluIGEgY29tYmluaW5nIHN0ZXAgaGFzIHRvIGJlIG9wZW5lZCBwcm9wZXJseSwKICAgIHNpbmNlIGl0cyBzb2x1dGlvbiBpcyBmaXR0ZWQgZnJvbSBhIGRvbm9yIGV4cG9zdXJlLgoKICAgIFJldHVybnMgKHdjcywgc2hhcGUsIGluZm8pLgogICAgIiIiCiAgICBmcm9tIGFzdHJvcHkuaW8gaW1wb3J0IGZpdHMKICAgIGZyb20gYXN0cm9weS53Y3MgaW1wb3J0IFdDUwoKICAgIHRyeToKICAgICAgICB3aXRoIGZpdHMub3BlbihwYXRoKSBhcyBoZHVsOgogICAgICAgICAgICBmb3IgaGR1IGluIGhkdWw6CiAgICAgICAgICAgICAgICBoZWFkZXIgPSBoZHUuaGVhZGVyCiAgICAgICAgICAgICAgICBpZiBoZWFkZXIuZ2V0KCJOQVhJUyIpID09IDI6CiAgICAgICAgICAgICAgICAgICAgdyA9IFdDUyhoZWFkZXIpCiAgICAgICAgICAgICAgICAgICAgaWYgdy5oYXNfY2VsZXN0aWFsOgogICAgICAgICAgICAgICAgICAgICAgICByZXR1cm4gKHcuY2VsZXN0aWFsLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIChoZWFkZXJbIk5BWElTMiJdLCBoZWFkZXJbIk5BWElTMSJdKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7InBpeGVsX3NjYWxlIjogaGVhZGVyLmdldCgiUElYU0NBTEUiLCBOb25lKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgImZpbHRlciI6IHN0cihoZWFkZXIuZ2V0KCJGSUxURVIiLCAiPyIpKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIm9iamVjdCI6IHN0cihoZWFkZXIuZ2V0KCJPQkpFQ1QiLCAiPyIpKX0pCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHBhc3MKCiAgICBkYXRhLCB3Y3MsIF8sIGluZm8gPSBsb2FkX2ZyYW1lKHBhdGgsIGRvbm9yX3BhdGgpCiAgICByZXR1cm4gd2NzLCBkYXRhLnNoYXBlLCBpbmZvCgpkZWYgc2FtZV9ncmlkKHdjc19hLCB3Y3NfYiwgc2hhcGUsIHRvbGVyYW5jZV9weD0xLjUpOgogICAgIiIiVHJ1ZSB3aGVuIHR3byBwbGF0ZSBzb2x1dGlvbnMgcHV0IHRoZSBza3kgaW4gdGhlIHNhbWUgcGl4ZWxzIGFscmVhZHkuCgogICAgV0hZIEFTSwogICAgUmVzYW1wbGluZyBhIGZyYW1lIGlzIHRoZSBtb3N0IGV4cGVuc2l2ZSB0aGluZyB0aGlzIHRvb2wgZG9lcyB0byBpdDogdHdvCiAgICBhc3Ryb21ldHJpYyB0cmFuc2Zvcm1zIG9mIGV2ZXJ5IHBpeGVsIHBvc2l0aW9uLCB0ZW5zIG9mIG1pbGxpb25zIG9mIHRoZW0uIEl0CiAgICBpcyB3b3J0aCBpdCB3aGVuIHRoZSBmcmFtZXMgcmVhbGx5IHNpdCBvbiBkaWZmZXJlbnQgZ3JpZHMuIEl0IGlzIHB1cmUgd2FzdGUKICAgIHdoZW4gdGhleSBkbyBub3QgLSBhbmQgYmV0d2VlbiBjb25zZWN1dGl2ZSBleHBvc3VyZXMgb2Ygb25lIGZpbHRlciwgdGhleQogICAgYWxtb3N0IG5ldmVyIGRvLiBNZWFzdXJlZCBvbiBhIG5pZ2h0IG9mIE0zMzogdGhlIGV4cG9zdXJlcyBvZiBhIGZpbHRlciBzYXQKICAgIHdpdGhpbiAwLjA1IHRvIDAuNyBvZiBhIHBpeGVsIG9mIGVhY2ggb3RoZXIuCgogICAgU28gdGhlIHF1ZXN0aW9uIGlzIGFza2VkIGJlZm9yZSB0aGUgd29yayBpcyBkb25lLiBBIGZldyBjb3JuZXIgYW5kIGNlbnRyZQogICAgcG9pbnRzIGFyZSBwdXNoZWQgdGhyb3VnaCBib3RoIHNvbHV0aW9uczsgaWYgbm9uZSBvZiB0aGVtIG1vdmVzIGJ5IG1vcmUgdGhhbgogICAgYWJvdXQgYSBwaXhlbCwgdGhlIGdyaWRzIGFyZSB0aGUgc2FtZSBhbmQgYSBzdWItcGl4ZWwgc2hpZnQgaXMgYWxsIHRoYXQgaXMKICAgIG5lZWRlZC4gVGhlIGZ1bGwgcmVzYW1wbGluZyBzdGF5cyBmb3IgdGhlIGNhc2UgaXQgd2FzIHdyaXR0ZW4gZm9yLgogICAgIiIiCiAgICBpbXBvcnQgbnVtcHkgYXMgbnAKCiAgICBpZiB3Y3NfYSBpcyBOb25lIG9yIHdjc19iIGlzIE5vbmU6CiAgICAgICAgcmV0dXJuIEZhbHNlCiAgICBueSwgbnggPSBzaGFwZQogICAgeHMgPSBucC5hcnJheShbMCwgbnggLSAxLCAwLCBueCAtIDEsIG54IC8vIDIsIG54IC8vIDQsIDMgKiBueCAvLyA0XSwKICAgICAgICAgICAgICAgICAgZHR5cGU9ZmxvYXQpCiAgICB5cyA9IG5wLmFycmF5KFswLCAwLCBueSAtIDEsIG55IC0gMSwgbnkgLy8gMiwgMyAqIG55IC8vIDQsIG55IC8vIDRdLAogICAgICAgICAgICAgICAgICBkdHlwZT1mbG9hdCkKICAgIHRyeToKICAgICAgICBza3kgPSB3Y3NfYS5waXhlbF90b193b3JsZCh4cywgeXMpCiAgICAgICAgYngsIGJ5ID0gd2NzX2Iud29ybGRfdG9fcGl4ZWwoc2t5KQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICByZXR1cm4gRmFsc2UKICAgIG1vdmVkID0gbnAuaHlwb3QoYnggLSB4cywgYnkgLSB5cykKICAgIHJldHVybiBib29sKG5wLmFsbChucC5pc2Zpbml0ZShtb3ZlZCkpIGFuZCBtb3ZlZC5tYXgoKSA8PSB0b2xlcmFuY2VfcHgpCgpkZWYgYWxpZ25fdG8ocmVmX3djcywgcmVmX3NoYXBlLCBkYXRhLCB3Y3MsIGJhbmRfcm93cz01MTIpOgogICAgIiIiUmVzYW1wbGUgb25lIGZyYW1lIG9udG8gYW5vdGhlciBmcmFtZSdzIHBpeGVsIGdyaWQgdXNpbmcgYm90aCBXQ1Mgc29sdXRpb25zLgoKICAgIEJvdGggZnJhbWVzIGFyZSBwbGF0ZS1zb2x2ZWQsIHNvIGV2ZXJ5IG91dHB1dCBwaXhlbCBjYW4gYmUgbWFwcGVkIHRvIGl0cyB0cnVlCiAgICBza3kgcG9zaXRpb24gYW5kIHRoZSBpbnB1dCBmcmFtZSByZWFkIHRoZXJlLiBFeGFjdCwgYW5kIGl0IG5lZWRzIG5vIG1hdGNoaW5nIG9mCiAgICBzdGFycyBiZXR3ZWVuIHRoZSB0d28gaW1hZ2VzLgoKICAgIERvbmUgYSBmZXcgaHVuZHJlZCByb3dzIGF0IGEgdGltZS4gVGhlIHdob2xlLWZyYW1lIHZlcnNpb24gYnVpbHQgZml2ZSBhcnJheXMKICAgIHRoZSBzaXplIG9mIHRoZSBpbWFnZSB0byBob2xkIHRoZSBjb29yZGluYXRlcyAtIGFib3V0IDIuNSBHQiBvbiBhIDYxLW1lZ2FwaXhlbAogICAgZnJhbWUsIGZvciBldmVyeSBleHBvc3VyZSAtIHdoaWNoIGlzIG1lbW9yeSBzcGVudCBvbiBzY3JhdGNoIHBhcGVyLiBUaGUKICAgIGFyaXRobWV0aWMgaXMgaWRlbnRpY2FsOyBvbmx5IHRoZSBvcmRlciBjaGFuZ2VkLgogICAgIiIiCiAgICBmcm9tIHNjaXB5Lm5kaW1hZ2UgaW1wb3J0IG1hcF9jb29yZGluYXRlcwoKICAgIG55LCBueCA9IHJlZl9zaGFwZQogICAgb3V0ID0gbnAuZW1wdHkoKG55LCBueCksIGR0eXBlPW5wLmZsb2F0MzIpCiAgICBmb3IgeTAgaW4gcmFuZ2UoMCwgbnksIGJhbmRfcm93cyk6CiAgICAgICAgeTEgPSBtaW4oeTAgKyBiYW5kX3Jvd3MsIG55KQogICAgICAgIHl5LCB4eCA9IG5wLm1ncmlkW3kwOnkxLCAwOm54XQogICAgICAgIHNreSA9IHJlZl93Y3MucGl4ZWxfdG9fd29ybGQoeHgsIHl5KQogICAgICAgIHN4LCBzeSA9IHdjcy53b3JsZF90b19waXhlbChza3kpCiAgICAgICAgb3V0W3kwOnkxXSA9IG1hcF9jb29yZGluYXRlcyhkYXRhLCBbc3ksIHN4XSwgb3JkZXI9MSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1vZGU9ImNvbnN0YW50IiwgY3ZhbD1ucC5uYW4pCiAgICByZXR1cm4gb3V0CgpkZWYgbWVhc3VyZV9md2htKGRhdGEsIG1hc2s9Tm9uZSwgbWF4X3NvdXJjZXM9MjAwLCBwb3NpdGlvbnM9Tm9uZSk6CiAgICAiIiJXaWR0aCBvZiB0aGUgbWVkaWFuIHN0YXIgaW4gYSBmcmFtZSwgaW4gcGl4ZWxzLCBhdCBoYWxmIGl0cyBoZWlnaHQuCgogICAgV0hZIFRIRSBNRURJQU4gU1RBUiBBTkQgTk9UIFRIRSBNRURJQU4gT0YgVEhFIFNUQVJTCiAgICBCZWNhdXNlIGEgc2luZ2xlIGZhaW50IHN0YXIgaXMgbW9zdGx5IG5vaXNlLCBhbmQgZXZlcnkgd2F5IG9mIHJlZHVjaW5nIG9uZSB0bwogICAgYSBudW1iZXIgaXMgcHVsbGVkIGFib3V0IGJ5IHRoYXQgbm9pc2UuIFN0YWNraW5nIHR3byBodW5kcmVkIG9mIHRoZW0gZmlyc3QKICAgIGdpdmVzIG9uZSBjbGVhbiBzdGFyIHRvIG1lYXN1cmUsIGFuZCBpdHMgd2lkdGggaXMgdGhlIHdpZHRoIHRoZSBmcmFtZSBoYXMuCgogICAgV0hZIE5PVCBUSEUgU0VDT05EIE1PTUVOVAogICAgVGhhdCBpcyB3aGF0IHRoaXMgZGlkIGJlZm9yZSwgYW5kIGl0IGdhdmUgdGhlIHdyb25nIGFuc3dlciBpbiB0aGUgd2F5IHRoYXQKICAgIG1hdHRlcnMgbW9zdDogaXQgZ290IHRoZSBvcmRlciBvZiB0aGUgdHdvIGZpbHRlcnMgYmFja3dhcmRzLiBBIHNlY29uZCBtb21lbnQKICAgIHdlaWdodHMgZWFjaCBwaXhlbCBieSB0aGUgc3F1YXJlIG9mIGl0cyBkaXN0YW5jZSwgc28gdGhlIG91dGVyIHJpbmcgb2YgYSAxOQogICAgcGl4ZWwgc3RhbXAgY291bnRzIGFib3V0IGEgaHVuZHJlZCBhbmQgc2V2ZW50eSB0aW1lcyBhcyBoZWF2aWx5IGFzIHRoZSBjZW50cmUuCiAgICBXaGF0ZXZlciBpcyBvdXQgdGhlcmUgdGhhdCBpcyBub3QgdGhlIHN0YXIgLSB0aGUgZ2FsYXh5J3Mgb3duIGxpZ2h0IHVuZGVyIGl0LAogICAgYSBuZWlnaGJvdXIsIHNreSBsZWZ0IG92ZXIgYmVjYXVzZSBvbmx5IHRoZSBmcmFtZSdzIGdsb2JhbCBsZXZlbCBoYWQgYmVlbgogICAgdGFrZW4gb2ZmIC0gaXMgdGhlcmVmb3JlIG1vc3Qgb2YgdGhlIGFuc3dlci4gT24gTTMzIHRoYXQgbWFkZSB0aGUgSC1hbHBoYQogICAgZnJhbWUgbWVhc3VyZSA1LjUzIHB4IGFnYWluc3QgUidzIDUuNzcsIHNvIHRoZSB0b29sIGJsdXJyZWQgSC1hbHBoYTsgc3RhY2tpbmcKICAgIHRoZSBzYW1lIHN0YXJzIHNob3dzIEgtYWxwaGEgYXQgNC45MSBhbmQgUiBhdCAzLjE0LCBzbyBILWFscGhhIHdhcyBhbHJlYWR5IHRoZQogICAgYnJvYWRlciBvZiB0aGUgdHdvIGFuZCBibHVycmluZyBpdCBtYWRlIHRoZSBtaXNtYXRjaCB3b3JzZS4gRXZlcnkgc3RhciB0aGVuCiAgICBzdWJ0cmFjdGVkIHRvIGEgYmxhY2sgY29yZSBpbnNpZGUgYSBicmlnaHQgcmluZywgd2hpY2ggaXMgZXhhY3RseSB3aGF0IHRoZQogICAgZnJhbWVzIGxvb2tlZCBsaWtlLgoKICAgIEhhbGYtbWF4aW11bSBpcyByZWFkIHN0cmFpZ2h0IG9mZiB0aGUgcmFkaWFsIHByb2ZpbGUgb2YgdGhlIHN0YWNrLiBJdCBpcyBhCiAgICBwcm9wZXJ0eSBvZiB0aGUgY29yZSwgd2hlcmUgdGhlIHN0YXIgaXMsIGFuZCBpdCBkb2VzIG5vdCBjYXJlIHdoYXQgdGhlIHN0YW1wJ3MKICAgIG91dHNraXJ0cyBjb250YWluLgoKICAgIFRIUkVFIFRISU5HUyBUSEUgU1RBQ0sgTkVFRFMKICAgIFRoZSBsb2NhbCBza3kgY29tZXMgb2ZmIGVhY2ggc3RhbXAgc2VwYXJhdGVseSwgdGFrZW4gZnJvbSBpdHMgYm9yZGVyIHJpbmcsIHNvCiAgICBhIHN0YXIgb24gYSBicmlnaHQgYXJtIGFuZCBhIHN0YXIgb24gZW1wdHkgc2t5IGNvbnRyaWJ1dGUgdGhlIHNhbWUgc3Rhci4KICAgIEVhY2ggc3RhbXAgaXMgc2hpZnRlZCBvbnRvIGEgY29tbW9uIGNlbnRyZSBiZWZvcmUgc3RhY2tpbmcsIHRvIGEgZnJhY3Rpb24gb2YgYQogICAgcGl4ZWwsIG9yIHRoZSBzdGFjayB3b3VsZCBiZSBzbWVhcmVkIGJ5IHVwIHRvIGhhbGYgYSBwaXhlbCBvZiByb3VuZGluZyAtIGFuZAogICAgc21lYXJlZCBkaWZmZXJlbnRseSBpbiBlYWNoIGZpbHRlciwgd2hpY2ggaXMgdGhlIGVycm9yIGJlaW5nIGxvb2tlZCBmb3IuCiAgICBTYXR1cmF0ZWQgc3RhcnMgYXJlIGRyb3BwZWQ6IGEgZmxhdCB0b3AgaXMgd2lkZXIgdGhhbiB0aGUgc3RhciB0aGF0IG1hZGUgaXQsCiAgICBhbmQgaXQgaXMgdGhlIGJyaWdodGVzdCBzdGFycyB0aGF0IHNhdHVyYXRlLCBzbyBrZWVwaW5nIHRoZW0gd291bGQgYmlhcyB0aGUKICAgIGZyYW1lIHdpdGggdGhlIGRlZXBlciBleHBvc3VyZS4gQSBjb3JlIHdpdGggdGhyZWUgb3IgbW9yZSBwaXhlbHMgd2l0aGluIHR3bwogICAgcGVyY2VudCBvZiBpdHMgcGVhayBpcyBmbGF0LCBhbmQgYSBzdGFyJ3MgY29yZSBpcyBub3QuCiAgICAiIiIKICAgIGZyb20gYXN0cm9weS5zdGF0cyBpbXBvcnQgc2lnbWFfY2xpcHBlZF9zdGF0cwogICAgZnJvbSBzY2lweS5uZGltYWdlIGltcG9ydCBzaGlmdCBhcyBfc2hpZnQKCiAgICBoYWxmID0gMTAKICAgIG55LCBueCA9IGRhdGEuc2hhcGUKICAgIHl5LCB4eCA9IG5wLm1ncmlkWy1oYWxmOmhhbGYgKyAxLCAtaGFsZjpoYWxmICsgMV0KICAgIHJyID0gbnAuaHlwb3QoeHgsIHl5KQogICAgYm9yZGVyID0gcnIgPiBoYWxmIC0gMS41CgogICAgaWYgcG9zaXRpb25zIGlzIG5vdCBOb25lIGFuZCBsZW4ocG9zaXRpb25zKSA+PSAxMDoKICAgICAgICBjYW5kaWRhdGVzID0gbnAuYXNhcnJheShwb3NpdGlvbnMsIGR0eXBlPWZsb2F0KQogICAgZWxzZToKICAgICAgICBmcm9tIHBob3R1dGlscy5kZXRlY3Rpb24gaW1wb3J0IERBT1N0YXJGaW5kZXIKICAgICAgICBtZWRpYW4sIHN0ZCA9IHNpZ21hX2NsaXBwZWRfc3RhdHMoZGF0YSwgc2lnbWE9My4wLCBtYXNrPW1hc2spWzE6XQogICAgICAgIGZpbmRlciA9IERBT1N0YXJGaW5kZXIoZndobT00LjAsIHRocmVzaG9sZD0yMC4wICogc3RkKQogICAgICAgIHRibCA9IGZpbmRlcihkYXRhIC0gbWVkaWFuLCBtYXNrPW1hc2spCiAgICAgICAgaWYgdGJsIGlzIE5vbmUgb3IgbGVuKHRibCkgPT0gMDoKICAgICAgICAgICAgcmV0dXJuIE5vbmUKICAgICAgICB0Ymwuc29ydCgiZmx1eCIpCiAgICAgICAgdGJsID0gdGJsWzo6LTFdWzptYXhfc291cmNlc10KICAgICAgICBjYW5kaWRhdGVzID0gbnAuY29sdW1uX3N0YWNrKFt0YmxbInhjZW50cm9pZCJdLCB0YmxbInljZW50cm9pZCJdXSkKCiAgICBmb3VuZCA9IFtdCiAgICBmb3IgeCwgeSBpbiBjYW5kaWRhdGVzOgogICAgICAgIHhpLCB5aSA9IGludChyb3VuZCh4KSksIGludChyb3VuZCh5KSkKICAgICAgICBpZiBub3QgKGhhbGYgPCB4aSA8IG54IC0gaGFsZiBhbmQgaGFsZiA8IHlpIDwgbnkgLSBoYWxmKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICBzdGFtcCA9IG5wLm5hbl90b19udW0oCiAgICAgICAgICAgIGRhdGFbeWkgLSBoYWxmOnlpICsgaGFsZiArIDEsIHhpIC0gaGFsZjp4aSArIGhhbGYgKyAxXSkuYXN0eXBlKGZsb2F0KQogICAgICAgIHN0YW1wID0gc3RhbXAgLSBucC5tZWRpYW4oc3RhbXBbYm9yZGVyXSkKICAgICAgICBjb3JlID0gc3RhbXBbaGFsZiAtIDE6aGFsZiArIDIsIGhhbGYgLSAxOmhhbGYgKyAyXQogICAgICAgIHBlYWsgPSBmbG9hdChjb3JlLm1heCgpKQogICAgICAgIGlmIHBlYWsgPD0gMCBvciAoY29yZSA+PSAwLjk4ICogcGVhaykuc3VtKCkgPj0gMzoKICAgICAgICAgICAgY29udGludWUKICAgICAgICBmb3VuZC5hcHBlbmQoKHBlYWssIF9zaGlmdChzdGFtcCwgKHlpIC0geSwgeGkgLSB4KSwgb3JkZXI9MSwgbW9kZT0ibmVhcmVzdCIpKSkKICAgIGlmIGxlbihmb3VuZCkgPCAxMDoKICAgICAgICByZXR1cm4gTm9uZQoKICAgIGZvdW5kLnNvcnQoa2V5PWxhbWJkYSBwYWlyOiAtcGFpclswXSkKICAgIHN0YWNrZWQgPSBucC5tZWRpYW4obnAuc3RhY2soW3MgZm9yIF8sIHMgaW4gZm91bmRbOm1heF9zb3VyY2VzXV0pLCBheGlzPTApCiAgICBwcm9maWxlID0gbnAuYXJyYXkoW3N0YWNrZWRbKHJyID49IHEpICYgKHJyIDwgcSArIDEpXS5tZWFuKCkKICAgICAgICAgICAgICAgICAgICAgICAgZm9yIHEgaW4gcmFuZ2UoaGFsZiArIDEpXSkKICAgIGlmIHByb2ZpbGVbMF0gPD0gMDoKICAgICAgICByZXR1cm4gTm9uZQogICAgZm9yIHEgaW4gcmFuZ2UoMSwgaGFsZiArIDEpOgogICAgICAgIGlmIHByb2ZpbGVbcV0gPCAwLjUgKiBwcm9maWxlWzBdOgogICAgICAgICAgICBiZWxvdywgYWJvdmUgPSBwcm9maWxlW3EgLSAxXSwgcHJvZmlsZVtxXQogICAgICAgICAgICByZXR1cm4gZmxvYXQoMi4wICogKHEgLSAxICsgKDAuNSAqIHByb2ZpbGVbMF0gLSBiZWxvdykgLyAoYWJvdmUgLSBiZWxvdykpKQogICAgcmV0dXJuIE5vbmUKCmRlZiBtYXRjaF9wc2Yoc2hhcnAsIGZ3aG1fc2hhcnAsIGZ3aG1fYmx1bnQpOgogICAgIiIiQmx1ciB0aGUgc2hhcnBlciBmcmFtZSBzbyBib3RoIGZyYW1lcyBoYXZlIHRoZSBzYW1lIHN0YXIgc2l6ZS4KCiAgICBTdWJ0cmFjdGluZyBmcmFtZXMgb2YgZGlmZmVyZW50IHNoYXJwbmVzcyBsZWF2ZXMgYSByaW5nIGFyb3VuZCBldmVyeSBzdGFyIC0gYQogICAgYnJpZ2h0IGNvcmUgd2l0aCBhIGRhcmsgaGFsbywgb3IgdGhlIHJldmVyc2UgLSBhbmQgdGhvc2UgcmluZ3MgYXJlIHRoZW4gZGV0ZWN0ZWQKICAgIGFzIEhJSSByZWdpb25zLiBNYXRjaGluZyB0aGUgd2lkdGhzIGZpcnN0IGlzIHdoYXQgbWFrZXMgc3RhcnMgZGlzYXBwZWFyIGNsZWFubHkuCiAgICAiIiIKICAgIGZyb20gc2NpcHkubmRpbWFnZSBpbXBvcnQgZ2F1c3NpYW5fZmlsdGVyCgogICAgaWYgZndobV9ibHVudCBpcyBOb25lIG9yIGZ3aG1fc2hhcnAgaXMgTm9uZSBvciBmd2htX2JsdW50IDw9IGZ3aG1fc2hhcnA6CiAgICAgICAgcmV0dXJuIHNoYXJwLCAwLjAKICAgIHNpZ21hID0gbWF0aC5zcXJ0KG1heChmd2htX2JsdW50ICoqIDIgLSBmd2htX3NoYXJwICoqIDIsIDAuMCkpIC8gMi4zNTQ4CiAgICByZXR1cm4gZ2F1c3NpYW5fZmlsdGVyKHNoYXJwLCBzaWdtYSksIHNpZ21hCgpkZWYgX3N0YWNrX3N0YXJzKGRhdGEsIHBvc2l0aW9ucywgbWF4X3NvdXJjZXM9MjAwLCBoYWxmPTEwKToKICAgICIiIlRoZSBtZWRpYW4gc3RhciBvZiBhIGZyYW1lOiBsb2NhbCBza3kgb2ZmLCBzdWItcGl4ZWwgYWxpZ25lZCwgbm8gc2F0dXJhdGlvbi4KCiAgICBPbmUgc3RhciBpcyBtb3N0bHkgbm9pc2UuIFR3byBodW5kcmVkIG9mIHRoZW0gc3RhY2tlZCBpcyBvbmUgY2xlYW4gc3RhciwgYW5kCiAgICBldmVyeSBxdWVzdGlvbiBhYm91dCB0aGUgc2hhcGUgb2YgdGhpcyBmcmFtZSdzIHN0YXJzIGlzIGFza2VkIG9mIHRoYXQuCiAgICAiIiIKICAgIGZyb20gc2NpcHkubmRpbWFnZSBpbXBvcnQgc2hpZnQgYXMgX3NoaWZ0CgogICAgZGF0YSA9IG5wLmFzYXJyYXkoZGF0YSkKICAgIHBvcyA9IG5wLmFzYXJyYXkocG9zaXRpb25zLCBkdHlwZT1mbG9hdCkKICAgIG55LCBueCA9IGRhdGEuc2hhcGUKICAgIHl5LCB4eCA9IG5wLm1ncmlkWy1oYWxmOmhhbGYgKyAxLCAtaGFsZjpoYWxmICsgMV0KICAgIGJvcmRlciA9IG5wLmh5cG90KHh4LCB5eSkgPiBoYWxmIC0gMS41CgogICAgZm91bmQgPSBbXQogICAgZm9yIHgsIHkgaW4gcG9zOgogICAgICAgIHhpLCB5aSA9IGludChyb3VuZCh4KSksIGludChyb3VuZCh5KSkKICAgICAgICBpZiBub3QgKGhhbGYgPCB4aSA8IG54IC0gaGFsZiBhbmQgaGFsZiA8IHlpIDwgbnkgLSBoYWxmKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICBzdGFtcCA9IG5wLm5hbl90b19udW0oCiAgICAgICAgICAgIGRhdGFbeWkgLSBoYWxmOnlpICsgaGFsZiArIDEsIHhpIC0gaGFsZjp4aSArIGhhbGYgKyAxXSkuYXN0eXBlKGZsb2F0KQogICAgICAgIHN0YW1wID0gc3RhbXAgLSBucC5tZWRpYW4oc3RhbXBbYm9yZGVyXSkKICAgICAgICBjb3JlID0gc3RhbXBbaGFsZiAtIDE6aGFsZiArIDIsIGhhbGYgLSAxOmhhbGYgKyAyXQogICAgICAgIHBlYWsgPSBmbG9hdChjb3JlLm1heCgpKQogICAgICAgIGlmIHBlYWsgPD0gMCBvciAoY29yZSA+PSAwLjk4ICogcGVhaykuc3VtKCkgPj0gMzoKICAgICAgICAgICAgY29udGludWUKICAgICAgICBmb3VuZC5hcHBlbmQoKHBlYWssIF9zaGlmdChzdGFtcCwgKHlpIC0geSwgeGkgLSB4KSwgb3JkZXI9MSwgbW9kZT0ibmVhcmVzdCIpKSkKICAgIGlmIGxlbihmb3VuZCkgPCAxMDoKICAgICAgICByZXR1cm4gTm9uZQogICAgZm91bmQuc29ydChrZXk9bGFtYmRhIHBhaXI6IC1wYWlyWzBdKQogICAgcmV0dXJuIG5wLm1lZGlhbihucC5zdGFjayhbcyBmb3IgXywgcyBpbiBmb3VuZFs6bWF4X3NvdXJjZXNdXSksIGF4aXM9MCkKCmRlZiBtYXRjaF9wc2Zfb25fc3RhcnMoaGEsIHJlZCwgcG9zaXRpb25zLCBtYXhfc2lnbWE9My4wLCBzdGVwPTAuMDIsIGhhbGY9MTApOgogICAgIiIiQmx1ciBvbmUgZnJhbWUgdW50aWwgaXRzIHN0YXJzIGhhdmUgdGhlIHNhbWUgc2hhcGUgYXMgdGhlIG90aGVyJ3MuCgogICAgV0hZIE5PVCBNRUFTVVJFIFRXTyBXSURUSFMgQU5EIFNVQlRSQUNUIFRIRU0gSU4gUVVBRFJBVFVSRQogICAgQmVjYXVzZSB0aGF0IGlzIHdoYXQgbWF0Y2hfcHNmIGRvZXMgYW5kIGl0IGRvZXMgbm90IHdvcmsgaGVyZSwgZm9yIHR3byByZWFzb25zCiAgICB0aGF0IGNvbXBvdW5kLiBBIHdpZHRoIGlzIG9uZSBudW1iZXIgc3F1ZWV6ZWQgb3V0IG9mIGEgd2hvbGUgcHJvZmlsZSwgc28gaXQKICAgIHRocm93cyBhd2F5IGV4YWN0bHkgdGhlIHBhcnQgdGhhdCBtYXR0ZXJzIC0gYSBzdGFyIGNhbiBoYXZlIHRoZSBzYW1lIGhhbGYtd2lkdGgKICAgIGFzIGFub3RoZXIgYW5kIGEgcXVpdGUgZGlmZmVyZW50IGNvcmUgYW5kIHNraXJ0LiBBbmQgdGhlIHF1YWRyYXR1cmUgZm9ybXVsYSBpcwogICAgb25seSBleGFjdCBmb3IgR2F1c3NpYW5zOyBhIHJlYWwgc3RhciBoYXMgYnJvYWRlciB3aW5ncyB0aGFuIGEgR2F1c3NpYW4sIHNvIHRoZQogICAgYmx1ciBpdCBwcmVzY3JpYmVzIGlzIG5vdCB0aGUgYmx1ciB0aGF0IG1hdGNoZXMgdGhlIHR3by4KCiAgICBNZWFzdXJlZCBvbiBNODM6IHRoZSB3aWR0aHMgY2FtZSBvdXQgSC1hbHBoYSAzLjc4IHB4IGFnYWluc3QgUiA0LjIyLCB3aGljaAogICAgcHJlc2NyaWJlZCBibHVycmluZyBILWFscGhhIGJ5IHNpZ21hIDAuODAuIEFmdGVyd2FyZHMgdGhlIHN0YWNrZWQgc3RhciBsZWZ0IGEKICAgIG5lZ2F0aXZlIGNvcmUgaW5zaWRlIGEgcG9zaXRpdmUgcmluZyAtIHRoZSBzaWduYXR1cmUgb2YgSC1hbHBoYSBub3cgYmVpbmcgdGhlCiAgICBicm9hZGVyIG9mIHRoZSB0d28uIFRoZSBwcmVzY3JpcHRpb24gaGFkIHRoZSBzaWduIG9mIHRoZSBhbnN3ZXIgd3JvbmcuCgogICAgV0hBVCBUSElTIERPRVMgSU5TVEVBRAogICAgSXQgYXNrcyB0aGUgcXVlc3Rpb24gdGhlIHBpY3R1cmUgYXNrcy4gQm90aCBmcmFtZXMnIHN0YXJzIGFyZSBzdGFja2VkLCBlYWNoCiAgICBwcm9maWxlIGlzIGRpdmlkZWQgYnkgaXRzIG93biBsaWdodCBpbnNpZGUgc2l4IHBpeGVscyBzbyB0aGF0IGJyaWdodG5lc3MgZHJvcHMKICAgIG91dCBvZiB0aGUgY29tcGFyaXNvbiBhbmQgb25seSBzaGFwZSBpcyBsZWZ0LCBhbmQgdGhlbiBvbmUgZnJhbWUgaXMgYmx1cnJlZCBieQogICAgZXZlcnkgc2lnbWEgaW4gYSByYW5nZSB3aGlsZSB0aGUgdHdvIHByb2ZpbGVzIGFyZSBjb21wYXJlZCByaW5nIGJ5IHJpbmcuIFRoZQogICAgc2lnbWEgdGhhdCBtYWtlcyB0aGVtIGFncmVlIGlzIHRoZSBhbnN3ZXIsIGFuZCBpdCBpcyBhbiBhbnN3ZXIgYWJvdXQgdGhlIHdob2xlCiAgICBwcm9maWxlIHJhdGhlciB0aGFuIGFib3V0IG9uZSBwb2ludCBvbiBpdC4KCiAgICBBIHBvc2l0aXZlIHNpZ21hIGJsdXJzIEgtYWxwaGEsIGEgbmVnYXRpdmUgb25lIGJsdXJzIFI7IHRoZSBzZWFyY2ggY292ZXJzIGJvdGgKICAgIHNvIGl0IGNhbm5vdCBnZXQgdGhlIGRpcmVjdGlvbiB3cm9uZy4KCiAgICBSZXR1cm5zIChoYSwgcmVkLCBzaWdtYSwgZGlhZ25vc3RpY3MpLgogICAgIiIiCiAgICBmcm9tIHNjaXB5Lm5kaW1hZ2UgaW1wb3J0IGdhdXNzaWFuX2ZpbHRlcgoKICAgIGlmIHBvc2l0aW9ucyBpcyBOb25lIG9yIGxlbihwb3NpdGlvbnMpIDwgMTA6CiAgICAgICAgcmV0dXJuIGhhLCByZWQsIDAuMCwgeyJ3aHkiOiAibm8gc3RhciBwb3NpdGlvbnMgdG8gbWVhc3VyZSBvbiJ9CiAgICBBMCA9IF9zdGFja19zdGFycyhoYSwgcG9zaXRpb25zLCBoYWxmPWhhbGYpCiAgICBSMCA9IF9zdGFja19zdGFycyhyZWQsIHBvc2l0aW9ucywgaGFsZj1oYWxmKQogICAgaWYgQTAgaXMgTm9uZSBvciBSMCBpcyBOb25lOgogICAgICAgIHJldHVybiBoYSwgcmVkLCAwLjAsIHsid2h5IjogIm5vdCBlbm91Z2ggdXNhYmxlIHN0YXJzIHRvIHN0YWNrIn0KCiAgICB5eSwgeHggPSBucC5tZ3JpZFstaGFsZjpoYWxmICsgMSwgLWhhbGY6aGFsZiArIDFdCiAgICByciA9IG5wLmh5cG90KHh4LCB5eSkKICAgIHJpbmdzID0gWyhyciA+PSBxKSAmIChyciA8IHEgKyAxKSBmb3IgcSBpbiByYW5nZShoYWxmKV0KICAgIGlubmVyID0gcnIgPCA2CgogICAgZGVmIHNoYXBlX29mKHN0YW1wKToKICAgICAgICAiIiJUaGUgcmFkaWFsIHByb2ZpbGUsIGRpdmlkZWQgYnkgdGhlIGxpZ2h0IGluIHRoZSBjb3JlIC0gc2hhcGUgb25seS4iIiIKICAgICAgICB0b3RhbCA9IHN0YW1wW2lubmVyXS5zdW0oKQogICAgICAgIGlmIHRvdGFsIDw9IDA6CiAgICAgICAgICAgIHJldHVybiBOb25lCiAgICAgICAgcmV0dXJuIG5wLmFycmF5KFtzdGFtcFttXS5tZWFuKCkgZm9yIG0gaW4gcmluZ3NdKSAvIHRvdGFsCgogICAgZGVmIG1pc21hdGNoKHNpZ21hKToKICAgICAgICBhID0gZ2F1c3NpYW5fZmlsdGVyKEEwLCBzaWdtYSkgaWYgc2lnbWEgPiAwIGVsc2UgQTAKICAgICAgICByID0gZ2F1c3NpYW5fZmlsdGVyKFIwLCAtc2lnbWEpIGlmIHNpZ21hIDwgMCBlbHNlIFIwCiAgICAgICAgc2EsIHNyID0gc2hhcGVfb2YoYSksIHNoYXBlX29mKHIpCiAgICAgICAgaWYgc2EgaXMgTm9uZSBvciBzciBpcyBOb25lOgogICAgICAgICAgICByZXR1cm4gbnAuaW5mCiAgICAgICAgcmV0dXJuIGZsb2F0KCgoc2EgLSBzcikgKiogMikuc3VtKCkpCgogICAgZ3JpZCA9IG5wLmFyYW5nZSgtbWF4X3NpZ21hLCBtYXhfc2lnbWEgKyBzdGVwLCBzdGVwKQogICAgc2NvcmVzID0gbnAuYXJyYXkoW21pc21hdGNoKHMpIGZvciBzIGluIGdyaWRdKQogICAgYmVzdCA9IGZsb2F0KGdyaWRbaW50KG5wLmFyZ21pbihzY29yZXMpKV0pCiAgICBpZiBhYnMoYmVzdCkgPCBzdGVwOgogICAgICAgIGJlc3QgPSAwLjAKCiAgICBkaWFnID0geyJzaWdtYSI6IGJlc3QsCiAgICAgICAgICAgICJtaXNtYXRjaF9iZWZvcmUiOiBtaXNtYXRjaCgwLjApLAogICAgICAgICAgICAibWlzbWF0Y2hfYWZ0ZXIiOiBtaXNtYXRjaChiZXN0KSwKICAgICAgICAgICAgImJsdXJyZWQiOiAoIkgtYWxwaGEiIGlmIGJlc3QgPiAwIGVsc2UgIlIiIGlmIGJlc3QgPCAwIGVsc2UgIm5laXRoZXIiKX0KICAgIGlmIGJlc3QgPiAwOgogICAgICAgIGhhID0gZ2F1c3NpYW5fZmlsdGVyKGhhLCBiZXN0KQogICAgZWxpZiBiZXN0IDwgMDoKICAgICAgICByZWQgPSBnYXVzc2lhbl9maWx0ZXIocmVkLCAtYmVzdCkKICAgIHJldHVybiBoYSwgcmVkLCBhYnMoYmVzdCksIGRpYWcKCmRlZiBkZXNjcmliZV9wc2ZfbWF0Y2goZGlhZyk6CiAgICAiIiJXaGF0IHRoZSBzZWFyY2ggc2V0dGxlZCBvbiwgYW5kIGJ5IGhvdyBtdWNoIGl0IGltcHJvdmVkIHRoaW5ncy4iIiIKICAgIGlmIGRpYWcuZ2V0KCJ3aHkiKToKICAgICAgICByZXR1cm4gIiAgcG9pbnQgc3ByZWFkIGZ1bmN0aW9ucyBsZWZ0IGFsb25lICh7fSkiLmZvcm1hdChkaWFnWyJ3aHkiXSkKICAgIGlmIGRpYWdbImJsdXJyZWQiXSA9PSAibmVpdGhlciI6CiAgICAgICAgcmV0dXJuICIgIHRoZSBzdGFycyBhbHJlYWR5IGhhdmUgdGhlIHNhbWUgc2hhcGUgaW4gYm90aCBmcmFtZXMiCiAgICBiZWZvcmUsIGFmdGVyID0gZGlhZ1sibWlzbWF0Y2hfYmVmb3JlIl0sIGRpYWdbIm1pc21hdGNoX2FmdGVyIl0KICAgIGxpbmUgPSAoIiAgYmx1cnJlZCB0aGUge30gZnJhbWUgYnkgc2lnbWEgPSB7Oi4yZn0gcHgsIGNob3NlbiBieSBjb21wYXJpbmcgdGhlICIKICAgICAgICAgICAgInN0YWNrZWQgc3RhciByaW5nIGJ5IHJpbmciLmZvcm1hdChkaWFnWyJibHVycmVkIl0sIGFicyhkaWFnWyJzaWdtYSJdKSkpCiAgICBpZiBiZWZvcmUgPiAwOgogICAgICAgIGxpbmUgKz0gKCJcbiAgICB0aGUgZGlzYWdyZWVtZW50IGJldHdlZW4gdGhlIHR3byBwcm9maWxlcyBmZWxsIHRvIHs6LjBmfSUgIgogICAgICAgICAgICAgICAgICJvZiB3aGF0IGl0IHdhcyIuZm9ybWF0KDEwMC4wICogYWZ0ZXIgLyBiZWZvcmUpKQogICAgcmV0dXJuIGxpbmUKCmRlZiBjb250aW51dW1fZmFjdG9yKGhhLCByZWQsIG1hc2s9Tm9uZSwgbl9zdGFycz0yMDAsIHBvc2l0aW9ucz1Ob25lKToKICAgICIiIkhvdyBtdWNoIHRvIGJyaWdodGVuIHRoZSBuYXJyb3cgZnJhbWUgYmVmb3JlIHN1YnRyYWN0aW5nIHRoZSBicm9hZCBvbmUuCgogICAgQSBmb3JlZ3JvdW5kIHN0YXIgZW1pdHMgbm8gSC1hbHBoYSBsaW5lIHdvcnRoIHNwZWFraW5nIG9mLCBzbyB3aGF0ZXZlciBpdCBwdXRzCiAgICBpbnRvIHRoZSBuYXJyb3cgZnJhbWUgaXMgcHVyZSBjb250aW51dW0sIGFuZCB0aGUgcmF0aW8gb2YgaXRzIGJyaWdodG5lc3MgaW4gdGhlCiAgICB0d28gZnJhbWVzIGlzIHRoZSBmYWN0b3Igd2Ugd2FudC4KCiAgICAiRm9yZWdyb3VuZCIgaXMgdGhlIHdvcmQgdGhhdCBtYXR0ZXJzLCBhbmQgaXQgaXMgd2h5IGBwb3NpdGlvbnNgIGV4aXN0cy4gUG9pbnQKICAgIHNvdXJjZXMgZm91bmQgaW4gdGhlIGltYWdlIGFyZSBmb3JlZ3JvdW5kIG9ubHkgd2hlbiB0aGUgZ2FsYXh5IGlzIGZhciBlbm91Z2gKICAgIGF3YXkgZm9yIGl0cyBvd24gc3RhcnMgbm90IHRvIGJlIHJlc29sdmVkLiBJbiBhIG5lYXJieSBnYWxheHkgdGhleSBhcmUsIGluCiAgICB0aGVpciB0aG91c2FuZHMsIGFuZCB0aGV5IGFyZSBub3QgcHVyZSBjb250aW51dW0gLSBtZWFzdXJlZCBvbiBNMzMgdGhlIG1peHR1cmUKICAgIGdhdmUgYSBzdGFyLXRvLXN0YXIgc2NhdHRlciBvZiA2NiBwZXJjZW50LCB3aGljaCBpcyBubyBmYWN0b3IgYXQgYWxsLiBQYXNzaW5nCiAgICBjYXRhbG9ndWUgcG9zaXRpb25zIG9mIHN0YXJzIHdpdGggYSBtZWFzdXJhYmxlIHBhcmFsbGF4IHNldHRsZXMgdGhlIHF1ZXN0aW9uCiAgICBwaHlzaWNhbGx5IGluc3RlYWQgb2YgaG9waW5nIGRpc3RhbmNlIHNldHRsZXMgaXQuCgogICAgVGhlIG1lZGlhbiBvZiB0aGUgcGVyLXN0YXIgcmF0aW8gaXMgdXNlZCByYXRoZXIgdGhhbiBhIGZpdCwgYmVjYXVzZSB0aGUgbWVkaWFuCiAgICBpcyBub3QgZHJhZ2dlZCBhcm91bmQgYnkgdGhlIGZldyBzdGFycyB0aGF0IGRvIGhhdmUgbGluZSBlbWlzc2lvbiwgb3IgYXJlCiAgICBzYXR1cmF0ZWQsIG9yIHNpdCBvbiBhIGJyaWdodCBwaWVjZSBvZiB0aGUgZ2FsYXh5LgoKICAgIFJldHVybnMgdGhlIGZhY3RvciBhbmQgZGlhZ25vc3RpY3MgLSBpbiBwYXJ0aWN1bGFyIHRoZSBzdGFyLXRvLXN0YXIgc2NhdHRlciwKICAgIHdoaWNoIGlzIHRoZSBob25lc3QgdW5jZXJ0YWludHkgb24gdGhlIHN1YnRyYWN0aW9uLgogICAgIiIiCiAgICBmcm9tIHBob3R1dGlscy5kZXRlY3Rpb24gaW1wb3J0IERBT1N0YXJGaW5kZXIKICAgIGZyb20gcGhvdHV0aWxzLmFwZXJ0dXJlIGltcG9ydCBDaXJjdWxhckFwZXJ0dXJlLCBDaXJjdWxhckFubnVsdXMsIGFwZXJ0dXJlX3Bob3RvbWV0cnkKICAgIGZyb20gYXN0cm9weS5zdGF0cyBpbXBvcnQgc2lnbWFfY2xpcHBlZF9zdGF0cwoKICAgIGlmIHBvc2l0aW9ucyBpcyBub3QgTm9uZSBhbmQgbGVuKHBvc2l0aW9ucykgPj0gMTA6CiAgICAgICAgIyBBIGdlbmVyb3VzIHNsaWNlIG9mIHRoZSBjYXRhbG9ndWUsIHRoZW4gdGhlIGNob2ljZSBpcyBtYWRlIG9uIHdoYXQgdGhlc2UKICAgICAgICAjIGZyYW1lcyBhY3R1YWxseSByZWNvcmRlZCByYXRoZXIgdGhhbiBvbiB3aGF0IHRoZSBjYXRhbG9ndWUgc2F5cy4gR2FpYSdzCiAgICAgICAgIyBHIG1hZ25pdHVkZSBpcyBub3QgdGhpcyB0ZWxlc2NvcGUncyBSLCBhbmQgYSBzdGFyIHRoYXQgaXMgcmVzcGVjdGFibGUgaW4KICAgICAgICAjIHRoZSBjYXRhbG9ndWUgY2FuIGJlIGEgaGFuZGZ1bCBvZiBjb3VudHMgdGhyb3VnaCBhIG5hcnJvdyBmaWx0ZXIgLSB3aGVyZQogICAgICAgICMgdGhlIHJhdGlvIGl0IGdpdmVzIGlzIG5vdCBhIG1lYXN1cmVtZW50IGJ1dCBhIGRpdmlzaW9uIGJ5IG5vaXNlLgogICAgICAgIHBvcyA9IG5wLmFzYXJyYXkocG9zaXRpb25zLCBkdHlwZT1mbG9hdClbOjQgKiBuX3N0YXJzXQogICAgZWxzZToKICAgICAgICBtZWFuX3IsIG1lZF9yLCBzdGRfciA9IHNpZ21hX2NsaXBwZWRfc3RhdHMocmVkLCBzaWdtYT0zLjAsIG1hc2s9bWFzaykKICAgICAgICBmaW5kZXIgPSBEQU9TdGFyRmluZGVyKGZ3aG09NC4wLCB0aHJlc2hvbGQ9MjUuMCAqIHN0ZF9yKQogICAgICAgIHN0YXJzID0gZmluZGVyKHJlZCAtIG1lZF9yLCBtYXNrPW1hc2spCiAgICAgICAgaWYgc3RhcnMgaXMgTm9uZSBvciBsZW4oc3RhcnMpIDwgNToKICAgICAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCJ0b28gZmV3IHN0YXJzIGZvdW5kIHRvIHNldCB0aGUgY29udGludXVtIGZhY3RvciIpCgogICAgICAgIHN0YXJzLnNvcnQoImZsdXgiKQogICAgICAgIHN0YXJzID0gc3RhcnNbOjotMV1bOm5fc3RhcnNdCiAgICAgICAgcG9zID0gbnAudHJhbnNwb3NlKChzdGFyc1sieGNlbnRyb2lkIl0sIHN0YXJzWyJ5Y2VudHJvaWQiXSkpCgogICAgYXAgPSBDaXJjdWxhckFwZXJ0dXJlKHBvcywgcj02LjApCiAgICBhbm4gPSBDaXJjdWxhckFubnVsdXMocG9zLCByX2luPTEwLjAsIHJfb3V0PTE2LjApCgogICAgZGVmIG5ldF9mbHV4KGltZyk6CiAgICAgICAgcGhvdCA9IGFwZXJ0dXJlX3Bob3RvbWV0cnkoaW1nLCBbYXAsIGFubl0pCiAgICAgICAgYmtnX21lYW4gPSBwaG90WyJhcGVydHVyZV9zdW1fMSJdIC8gYW5uLmFyZWEKICAgICAgICByZXR1cm4gbnAuYXNhcnJheShwaG90WyJhcGVydHVyZV9zdW1fMCJdIC0gYmtnX21lYW4gKiBhcC5hcmVhLCBkdHlwZT1mbG9hdCkKCiAgICBmX2hhLCBmX3IgPSBuZXRfZmx1eChoYSksIG5ldF9mbHV4KHJlZCkKCiAgICBvayA9IG5wLmlzZmluaXRlKGZfaGEpICYgbnAuaXNmaW5pdGUoZl9yKSAmIChmX2hhID4gMCkgJiAoZl9yID4gMCkKICAgIGlmIG9rLnN1bSgpIDwgNToKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoIm5vIHVzYWJsZSBzdGFycyBmb3IgdGhlIGNvbnRpbnV1bSBmYWN0b3IiKQoKICAgICMgS2VlcCB0aGUgYnJpZ2h0ZXN0LCBtZWFzdXJlZCBoZXJlLiBXaXRoIGEgY2F0YWxvZ3VlIGxpc3QgdGhpcyBpcyB3aGF0IHR1cm5zCiAgICAjIGEgc2V0IG9mIHN0YXJzIHRoYXQgYXJlIG1lcmVseSByZWFsIGludG8gYSBzZXQgdGhhdCBpcyB3b3J0aCBtZWFzdXJpbmcuCiAgICBpZiBvay5zdW0oKSA+IG5fc3RhcnM6CiAgICAgICAgb3JkZXIgPSBucC5hcmdzb3J0KG5wLndoZXJlKG9rLCBmX3IsIC1ucC5pbmYpKVs6Oi0xXVs6bl9zdGFyc10KICAgICAgICBrZWVwX2JyaWdodCA9IG5wLnplcm9zKG9rLnNoYXBlLCBkdHlwZT1ib29sKQogICAgICAgIGtlZXBfYnJpZ2h0W29yZGVyXSA9IFRydWUKICAgICAgICBvayAmPSBrZWVwX2JyaWdodAoKICAgIHJhdGlvID0gZl9yW29rXSAvIGZfaGFbb2tdCiAgICBtZWQgPSBmbG9hdChucC5tZWRpYW4ocmF0aW8pKQogICAgc3ByZWFkID0gMS40ODI2ICogbnAubWVkaWFuKG5wLmFicyhyYXRpbyAtIG1lZCkpCiAgICBpZiBzcHJlYWQgPiAwOgogICAgICAgIHJhdGlvID0gcmF0aW9bbnAuYWJzKHJhdGlvIC0gbWVkKSA8IDMuMCAqIHNwcmVhZF0KCiAgICBmYWN0b3IgPSBmbG9hdChucC5tZWRpYW4ocmF0aW8pKQogICAgc2NhdHRlciA9IGZsb2F0KDEuNDgyNiAqIG5wLm1lZGlhbihucC5hYnMocmF0aW8gLSBmYWN0b3IpKSkKICAgIHJldHVybiBmYWN0b3IsIHsKICAgICAgICAibl9zdGFycyI6IGludChyYXRpby5zaXplKSwKICAgICAgICAic2NhdHRlciI6IHNjYXR0ZXIsCiAgICAgICAgInJlbGF0aXZlX3NjYXR0ZXIiOiBzY2F0dGVyIC8gZmFjdG9yIGlmIGZhY3RvciBlbHNlIGZsb2F0KCJuYW4iKSwKICAgIH0KCmRlZiBiYWNrZ3JvdW5kX2JveChzaGFwZSwgcmVxdWVzdGVkPU5vbmUsIHIyNV9weD1Ob25lKToKICAgICIiIkhvdyBsYXJnZSBhIHNxdWFyZSB0byBlc3RpbWF0ZSB0aGUgYmFja2dyb3VuZCBpbi4KCiAgICBUaGlzIGlzIG5vdCBhIGZyZWUgcGFyYW1ldGVyLiBUaGUgYmFja2dyb3VuZCBtYXAgaXMgbWVhbnQgdG8gZm9sbG93IHRoZSBza3kKICAgIGFuZCB0aGUgZ2FsYXh5J3Mgb3duIHNtb290aCBsaWdodCwgYW5kIHRvIGxlYXZlIHRoZSByZWdpb25zIHNpdHRpbmcgb24gdG9wIG9mCiAgICBpdCBhbG9uZS4gQSBib3ggdGhlIHNpemUgb2YgYSBzcGlyYWwgYXJtIGRvZXMgdGhlIG9wcG9zaXRlOiBpdCB0cmVhdHMgdGhlIGFybQogICAgYXMgYmFja2dyb3VuZCBhbmQgc3VidHJhY3RzIGF3YXkgdGhlIHZlcnkgdGhpbmcgdGhlIGZhaW50IHJlZ2lvbnMgc3RhbmQgb24uCiAgICBNZWFzdXJlZCBvbiB0aGUgTTgzIGZyYW1lcywgZ29pbmcgZnJvbSBhIDY0LXBpeGVsIGJveCB0byBhIDIwMC1waXhlbCBvbmUgZm91bmQKICAgIDc5IG1vcmUgcmVnaW9ucyBpbnNpZGUgdGhlIGdhbGF4eSB3aGlsZSB0aGUgY291bnQgb3V0IGluIHRoZSBlbXB0eSBmaWVsZCwgd2hlcmUKICAgIG5vdGhpbmcgcmVhbCBjYW4gYmUsIGRpZCBub3QgbW92ZSBhdCBhbGwgLSBzbyB0aG9zZSA3OSB3ZXJlIHJlYWwgYW5kIHdlcmUgYmVpbmcKICAgIGVyYXNlZC4KCiAgICBBIHR3ZWxmdGggb2YgdGhlIGZyYW1lIGlzIHdpZGUgZW5vdWdoIHRvIGJlIHdlbGwgY2xlYXIgb2YgYW55IHNpbmdsZSByZWdpb24gYW5kCiAgICBzbWFsbCBlbm91Z2ggdG8gc3RpbGwgZm9sbG93IGEgZ3JhZGllbnQgYWNyb3NzIHRoZSBmaWVsZC4KICAgICIiIgogICAgaWYgcmVxdWVzdGVkOgogICAgICAgIHJldHVybiBpbnQocmVxdWVzdGVkKQogICAgIyBUaWVkIHRvIHRoZSBnYWxheHksIG5vdCB0byB0aGUgZnJhbWUuIEEgYm94IHNldCBhcyBhIGZyYWN0aW9uIG9mIHRoZSBpbWFnZQogICAgIyBtYWtlcyB0aGUgYW5zd2VyIGRlcGVuZCBvbiBob3cgbXVjaCBlbXB0eSBza3kgaGFwcGVuZWQgdG8gc3Vycm91bmQgdGhlCiAgICAjIGdhbGF4eSB0aGF0IG5pZ2h0IC0gYW5kIGl0IHNob3dlZDogY3JvcHBpbmcgdGhlIGZyYW1lIHRvIHRoZSBnYWxheHksIHdoaWNoCiAgICAjIHNob3VsZCBoYXZlIGNoYW5nZWQgbm90aGluZyBidXQgdGhlIHJ1bm5pbmcgdGltZSwgbW92ZWQgdGhlIHJlZ2lvbiBjb3VudCBvbgogICAgIyBNODMgZnJvbSAzNDggdG8gMjgyIHB1cmVseSBiZWNhdXNlIHRoZSBib3ggc2hyYW5rIHdpdGggdGhlIGZyYW1lLgogICAgIwogICAgIyBBIHRoaXJkIG9mIFIyNSBpcyB3aGF0IHRoZSBtZWFzdXJlbWVudCBwb2ludGVkIHRvLiBPbiBNODMsIFIyNSBpcyA1NjAgcHggYW5kCiAgICAjIHRoZSBib3ggdGhhdCB3b3JrZWQgYmVzdCB3YXMgMjAwLgogICAgaWYgcjI1X3B4OgogICAgICAgIHJldHVybiBpbnQobWluKG1heChyMjVfcHggLyAzLjAsIDY0KSwgNDAwKSkKICAgIHJldHVybiBpbnQobWluKG1heChtaW4oc2hhcGUpIC8vIDEyLCA2NCksIDI1NikpCgpkZWYgZGV0ZWN0X3JlZ2lvbnMobmV0LCB0aHJlc2hvbGRfc2lnbWE9NC4wLCBtaW5fcGl4ZWxzPTYsIG1hc2s9Tm9uZSwgZGVibGVuZD1UcnVlLAogICAgICAgICAgICAgICAgICAgYm94X3NpemU9Tm9uZSwgbmxldmVscz02NCwgY29udHJhc3Q9MWUtNCwgcjI1X3B4PU5vbmUpOgogICAgIiIiRmluZCBISUkgcmVnaW9ucyBpbiB0aGUgY29udGludXVtLXN1YnRyYWN0ZWQgZnJhbWUuCgogICAgSElJIHJlZ2lvbnMgYXJlIGV4dGVuZGVkIGJsb2JzLCBub3QgcG9pbnRzLCBzbyBjb25uZWN0ZWQgZ3JvdXBzIG9mIHBpeGVscyBhYm92ZQogICAgYSB0aHJlc2hvbGQgYXJlIGdyb3duIHJhdGhlciB0aGFuIHN0ZWxsYXIgcHJvZmlsZXMgZml0dGVkLiBUb3VjaGluZyBibG9icyBhcmUKICAgIHNlcGFyYXRlZCBieSBkZWJsZW5kaW5nLCB3aGljaCBpcyB3aGF0IHN0b3BzIG9uZSBicmlnaHQgY29tcGxleCBiZWluZyBjb3VudGVkCiAgICBvbmNlIHdoZXJlIGEgcGVyc29uIGNsaWNraW5nIGJ5IGV5ZSB3b3VsZCBoYXZlIGNvdW50ZWQgdGhyZWUuCgogICAgdGhyZXNob2xkX3NpZ21hIGlzIGluIHVuaXRzIG9mIHRoZSBsb2NhbCBiYWNrZ3JvdW5kIG5vaXNlLCBzbyB0aGUgc2FtZSBudW1iZXIKICAgIG1lYW5zIHRoZSBzYW1lIHRoaW5nIG9uIGFueSBmcmFtZS4gSXQgaXMgdGhlIG1vc3QgY29uc2VxdWVudGlhbCBjaG9pY2UgaW4gdGhlCiAgICB3aG9sZSBwaXBlbGluZSwgd2hpY2ggaXMgd2h5IGl0IGlzIHdyaXR0ZW4gaW50byBldmVyeSBvdXRwdXQgZmlsZS4KICAgICIiIgogICAgZnJvbSBwaG90dXRpbHMuYmFja2dyb3VuZCBpbXBvcnQgQmFja2dyb3VuZDJELCBNZWRpYW5CYWNrZ3JvdW5kCiAgICBmcm9tIHBob3R1dGlscy5zZWdtZW50YXRpb24gaW1wb3J0IGRldGVjdF9zb3VyY2VzLCBkZWJsZW5kX3NvdXJjZXMsIFNvdXJjZUNhdGFsb2cKICAgIGZyb20gYXN0cm9weS5jb252b2x1dGlvbiBpbXBvcnQgR2F1c3NpYW4yREtlcm5lbCwgY29udm9sdmUKCiAgICBib3ggPSBiYWNrZ3JvdW5kX2JveChuZXQuc2hhcGUsIGJveF9zaXplLCByMjVfcHgpCiAgICBia2cgPSBCYWNrZ3JvdW5kMkQobmV0LCBib3hfc2l6ZT1ib3gsIGZpbHRlcl9zaXplPTMsCiAgICAgICAgICAgICAgICAgICAgICAgYmtnX2VzdGltYXRvcj1NZWRpYW5CYWNrZ3JvdW5kKCksIG1hc2s9bWFzaykKICAgIHJlc2lkID0gbmV0IC0gYmtnLmJhY2tncm91bmQKCiAgICBrZXJuZWwgPSBHYXVzc2lhbjJES2VybmVsKDIuMCkKICAgIHNtb290aGVkID0gY29udm9sdmUocmVzaWQsIGtlcm5lbCwgbWFzaz1tYXNrKQoKICAgICMgVGhlIHRocmVzaG9sZCBoYXMgdG8gYmUgbWVhc3VyZWQgb24gdGhlIGltYWdlIGl0IGlzIGFwcGxpZWQgdG8uIFNtb290aGluZwogICAgIyBhdmVyYWdlcyBuZWlnaGJvdXJpbmcgcGl4ZWxzLCBzbyBpdCBsb3dlcnMgdGhlIG5vaXNlIGJ5IGFib3V0IGEgZmFjdG9yIG9mCiAgICAjIHNldmVuIGZvciB0aGlzIGtlcm5lbDsgY29tcGFyaW5nIHRoZSBzbW9vdGhlZCBpbWFnZSBhZ2FpbnN0IHRoZSB1bnNtb290aGVkCiAgICAjIG5vaXNlIC0gd2hpY2ggaXMgd2hhdCB0aGUgZmlyc3QgdmVyc2lvbiBvZiB0aGlzIGRpZCAtIG1hZGUgYSBub21pbmFsICIzCiAgICAjIHNpZ21hIiBiZWhhdmUgbGlrZSB0d2VudHksIGFuZCBxdWlldGx5IHRocmV3IGF3YXkgbW9zdCBvZiB0aGUgZmFpbnQgcmVnaW9ucy4KICAgIHNtb290aF9ia2cgPSBCYWNrZ3JvdW5kMkQoc21vb3RoZWQsIGJveF9zaXplPWJveCwgZmlsdGVyX3NpemU9MywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYmtnX2VzdGltYXRvcj1NZWRpYW5CYWNrZ3JvdW5kKCksIG1hc2s9bWFzaykKICAgIHRocmVzaG9sZCA9IHRocmVzaG9sZF9zaWdtYSAqIHNtb290aF9ia2cuYmFja2dyb3VuZF9ybXMKCiAgICBzZWcgPSBkZXRlY3Rfc291cmNlcyhzbW9vdGhlZCwgdGhyZXNob2xkLCBucGl4ZWxzPW1pbl9waXhlbHMsIG1hc2s9bWFzaykKICAgIGlmIHNlZyBpcyBOb25lOgogICAgICAgIHJldHVybiBOb25lLCBia2csIHJlc2lkCgogICAgaWYgZGVibGVuZDoKICAgICAgICAjIFNwbGl0IHJhdGhlciB0aGFuIGx1bXAuIEEgbGFyZ2UgY29tcGxleCBpcyB1c3VhbGx5IHNldmVyYWwgcmVnaW9ucyB0aGF0CiAgICAgICAgIyBoYXBwZW4gdG8gdG91Y2gsIGFuZCBhdCB0aGlzIGRpc3RhbmNlIG9uZSBibG9iIGNhbiBiZSBodW5kcmVkcyBvZiBwYXJzZWNzCiAgICAgICAgIyBhY3Jvc3MuIFNwbGl0dGluZyBpdCBjaGFuZ2VzIHRoZSBjb3VudCBhbmQgdGhlIHNpemUgZGlzdHJpYnV0aW9uOyBpdAogICAgICAgICMgYmFyZWx5IG1vdmVzIHRoZSByYWRpYWwgcHJvZmlsZSwgYmVjYXVzZSB0aGUgcGllY2VzIHNpdCB3aGVyZSB0aGUgd2hvbGUKICAgICAgICAjIHNhdC4gT24gdGhlIE04MyBmcmFtZXMgdGhlc2Ugc2V0dGluZ3MgdHVybmVkIDI3NyBibG9icyBpbnRvIDM2NCB3aXRob3V0CiAgICAgICAgIyBhZGRpbmcgYW55dGhpbmcgdG8gdGhlIGVtcHR5IGZpZWxkLgogICAgICAgIHNlZyA9IGRlYmxlbmRfc291cmNlcyhzbW9vdGhlZCwgc2VnLCBucGl4ZWxzPW1pbl9waXhlbHMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5sZXZlbHM9bmxldmVscywgY29udHJhc3Q9Y29udHJhc3QsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHByb2dyZXNzX2Jhcj1GYWxzZSkKCiAgICBjYXQgPSBTb3VyY2VDYXRhbG9nKHJlc2lkLCBzZWcsIG1hc2s9bWFzaykKICAgIHJldHVybiBjYXQsIGJrZywgcmVzaWQKCmRlZiBvZmZzZXRfZnJvbV9jYXRhbG9ndWUocmVmZXJlbmNlLCBpbWFnZSwgcG9zaXRpb25zLCBib3g9MTEsIG1heF9zaGlmdF9weD00LjApOgogICAgIiIiVGhlIHNoaWZ0IGJldHdlZW4gdHdvIGZyYW1lcywgbWVhc3VyZWQgb24gb25lIGFncmVlZCBsaXN0IG9mIHN0YXJzLgoKICAgIFdIWSBOT1QgSlVTVCBERVRFQ1QgU1RBUlMgSU4gQk9USAogICAgQmVjYXVzZSB0aGUgYnJpZ2h0ZXN0IHRocmVlIGh1bmRyZWQgc291cmNlcyBpbiBhIG5hcnJvdy1iYW5kIGZyYW1lIGFyZSBub3QgdGhlCiAgICBicmlnaHRlc3QgdGhyZWUgaHVuZHJlZCBpbiBhIGJyb2FkLWJhbmQgb25lIC0gZGlmZmVyZW50IGZpbHRlcnMgc2VlIGRpZmZlcmVudAogICAgc3RhcnMgLSBzbyB0d28gaW5kZXBlbmRlbnRseSBkZXRlY3RlZCBsaXN0cyBvdmVybGFwIHBvb3JseSBhbmQgdGhlIHBhaXJzIHRoYXQKICAgIGRvIG1hdGNoIGFyZSB0aGUgb25lcyB0aGF0IGhhcHBlbmVkIHRvIHN1cnZpdmUuIE9uIGFuIE0zMyBwYWlyIHRoaXMgbGVmdCAyMQogICAgdXNhYmxlIHN0YXJzIG91dCBvZiBzaXggaHVuZHJlZCwgd2l0aCBhIHNjYXR0ZXIgb2YgdHdvIHBpeGVscy4KCiAgICBHaXZlbiBhIGxpc3Qgb2YgcG9zaXRpb25zIGZyb20gYSBjYXRhbG9ndWUsIHRoZSBxdWVzdGlvbiBkaXNhcHBlYXJzOiB0aGUgc2FtZQogICAgc3RhciBpcyBjZW50cm9pZGVkIGluIGJvdGggZnJhbWVzLCBhbmQgZXZlcnkgb25lIG9mIHRoZW0gY29udHJpYnV0ZXMuCgogICAgV0hZIFRIRSBTS1kgQ09NRVMgT0ZGIEVBQ0ggU1RBTVAgRklSU1QKICAgIFNhbWUgcmVhc29uIGFzIGluIGNlbnRyZV9vbl9zdGFycywgYW5kIHRoZSBjb25zZXF1ZW5jZSBoZXJlIGlzIHdvcnNlLiBBIGNlbnRyZQogICAgb2YgbGlnaHQgY29tcHV0ZWQgd2l0aCB0aGUgc2t5IHN0aWxsIGluIHRoZSBzdGFtcCBpcyBkb21pbmF0ZWQgYnkgdGhlIHNreSwgc28KICAgIGl0IGxhbmRzIG5lYXIgdGhlIG1pZGRsZSBvZiB0aGUgYm94IHdoYXRldmVyIHRoZSBzdGFyIGRvZXMgLSBpbiBib3RoIGZyYW1lcy4KICAgIFRoZSB0d28gYW5zd2VycyB0aGVuIGFncmVlIGJlYXV0aWZ1bGx5IGFuZCB0aGVpciBkaWZmZXJlbmNlIGlzIGNsb3NlIHRvIHplcm8sCiAgICBzbyB0aGlzIGZ1bmN0aW9uIHJlcG9ydGVkICJhbHJlYWR5IGFsaWduZWQiIG9uIGEgcGFpciB0aGF0IHdhcyBhIGZ1bGwgcGl4ZWwKICAgIGFwYXJ0LiBPbiBNODMgdGhhdCBwaXhlbCBpcyB3aGF0IGxlZnQgZXZlcnkgc3RhciBhcyBhIGRhcmsgY29yZSBiZXNpZGUgYSBicmlnaHQKICAgIGNyZXNjZW50IGluIHRoZSBzdWJ0cmFjdGVkIGZyYW1lLgogICAgIiIiCiAgICBpZiBwb3NpdGlvbnMgaXMgTm9uZSBvciBsZW4ocG9zaXRpb25zKSA8IDEwOgogICAgICAgIHJldHVybiBOb25lCgogICAgcG9zID0gbnAuYXNhcnJheShwb3NpdGlvbnMsIGR0eXBlPWZsb2F0KQogICAgY2EgPSBfY2VudHJvaWRfbG9jYWxseShyZWZlcmVuY2UsIHBvcywgYm94KQogICAgY2IgPSBfY2VudHJvaWRfbG9jYWxseShpbWFnZSwgcG9zLCBib3gpCiAgICBpZiBjYSBpcyBOb25lIG9yIGNiIGlzIE5vbmU6CiAgICAgICAgcmV0dXJuIE5vbmUKCiAgICBkeCwgZHkgPSBjYls6LCAwXSAtIGNhWzosIDBdLCBjYls6LCAxXSAtIGNhWzosIDFdCiAgICBvayA9IChucC5pc2Zpbml0ZShkeCkgJiBucC5pc2Zpbml0ZShkeSkKICAgICAgICAgICYgKG5wLmFicyhkeCkgPCBtYXhfc2hpZnRfcHgpICYgKG5wLmFicyhkeSkgPCBtYXhfc2hpZnRfcHgpKQogICAgaWYgb2suc3VtKCkgPCAxMDoKICAgICAgICByZXR1cm4gTm9uZQoKICAgIG14LCBteSA9IGZsb2F0KG5wLm1lZGlhbihkeFtva10pKSwgZmxvYXQobnAubWVkaWFuKGR5W29rXSkpCiAgICByZXR1cm4gbXgsIG15LCB7CiAgICAgICAgIm5fcGFpcnMiOiBpbnQob2suc3VtKCkpLAogICAgICAgICJzY2F0dGVyX3giOiBmbG9hdCgxLjQ4MjYgKiBucC5tZWRpYW4obnAuYWJzKGR4W29rXSAtIG14KSkpLAogICAgICAgICJzY2F0dGVyX3kiOiBmbG9hdCgxLjQ4MjYgKiBucC5tZWRpYW4obnAuYWJzKGR5W29rXSAtIG15KSkpLAogICAgICAgICJmcm9tX2NhdGFsb2d1ZSI6IFRydWUsCiAgICB9CgpkZWYgbWVhc3VyZV9vZmZzZXQocmVmZXJlbmNlLCBpbWFnZSwgZndobT01LjUsIHRocmVzaG9sZF9zaWdtYT0yMC4wLCBuX3N0YXJzPTMwMCwKICAgICAgICAgICAgICAgICAgIG1heF9tYXRjaF9weD00LjApOgogICAgIiIiVGhlIHNoaWZ0IHN0aWxsIGxlZnQgYmV0d2VlbiB0d28gZnJhbWVzIGFmdGVyIHRoZXkgd2VyZSBwdXQgb24gb25lIGdyaWQuCgogICAgV0hZIFRISVMgSVMgTkVFREVEIEFGVEVSIEEgV0NTIEFMSUdOTUVOVAogICAgUmVzYW1wbGluZyB0aHJvdWdoIHR3byBwbGF0ZSBzb2x1dGlvbnMgaXMgb25seSBhcyBnb29kIGFzIHRob3NlIHNvbHV0aW9ucy4gRWFjaAogICAgd2FzIGZpdHRlZCBzZXBhcmF0ZWx5LCBvbiBhIGRpZmZlcmVudCBleHBvc3VyZSwgYW5kIGVhY2ggY2FycmllcyBpdHMgb3duIGVycm9yCiAgICBvZiBhIGZyYWN0aW9uIG9mIGFuIGFyY3NlY29uZC4gV2hhdCBzdXJ2aXZlcyBpcyBhIHNtYWxsIHN5c3RlbWF0aWMgc2hpZnQgLSBvbgogICAgdGhlIGZyYW1lcyB0aGlzIHdhcyB3cml0dGVuIGZvciwgMC42IG9mIGEgcGl4ZWwgLSBhbmQgaXQgaXMgdmlzaWJsZTogZXZlcnkgc3RhcgogICAgbGVhdmVzIGEgZGFyay1hbmQtYnJpZ2h0IHBhaXIgaW4gdGhlIHN1YnRyYWN0ZWQgaW1hZ2UgaW5zdGVhZCBvZiB2YW5pc2hpbmcsIGFuZAogICAgdGhlIGRldGVjdG9yIHJlcG9ydHMgdGhvc2UgcGFpcnMgYXMgcmVnaW9ucy4KCiAgICBBIHNoaWZ0IGlzIG1lYXN1cmFibGUgdG8gZmFyIGJldHRlciB0aGFuIGEgcGl4ZWwgYmVjYXVzZSBpdCBpcyBjb21tb24gdG8gZXZlcnkKICAgIHN0YXIgaW4gdGhlIGZyYW1lLiBIZXJlIGl0IGlzIHRoZSBtZWRpYW4gb3ZlciBhIGZldyBodW5kcmVkIG9mIHRoZW0sIGFuZCB0aGUKICAgIHNjYXR0ZXIgYWJvdXQgdGhhdCBtZWRpYW4gc2F5cyB3aGV0aGVyIGl0IHJlYWxseSBpcyBvbmUgcmlnaWQgc2hpZnQgb3Igd2hldGhlcgogICAgc29tZXRoaW5nIGVsc2UgaXMgd3JvbmcuCgogICAgUmV0dXJucyAoZHgsIGR5LCBkaWFnbm9zdGljcykuIGR4LCBkeSBhcmUgaG93IGZhciBgaW1hZ2VgIHNpdHMgZnJvbQogICAgYHJlZmVyZW5jZWAsIHNvIHNoaWZ0aW5nIGl0IGJ5IHRoZSBuZWdhdGl2ZSBvZiB0aGVtIGJyaW5ncyB0aGUgdHdvIHRvZ2V0aGVyLgogICAgIiIiCiAgICBpbXBvcnQgbnVtcHkgYXMgbnAKICAgIGZyb20gcGhvdHV0aWxzLmRldGVjdGlvbiBpbXBvcnQgREFPU3RhckZpbmRlcgogICAgZnJvbSBhc3Ryb3B5LnN0YXRzIGltcG9ydCBzaWdtYV9jbGlwcGVkX3N0YXRzCiAgICBmcm9tIHNjaXB5LnNwYXRpYWwgaW1wb3J0IGNLRFRyZWUKCiAgICBkZWYgYnJpZ2h0X3N0YXJzKGltZyk6CiAgICAgICAgXywgbWVkLCBzZCA9IHNpZ21hX2NsaXBwZWRfc3RhdHMoaW1nLCBzaWdtYT0zLjApCiAgICAgICAgdGFibGUgPSBEQU9TdGFyRmluZGVyKGZ3aG09ZndobSwgdGhyZXNob2xkPXRocmVzaG9sZF9zaWdtYSAqIHNkKShpbWcgLSBtZWQpCiAgICAgICAgaWYgdGFibGUgaXMgTm9uZSBvciBsZW4odGFibGUpID09IDA6CiAgICAgICAgICAgIHJldHVybiBucC5lbXB0eSgoMCwgMikpCiAgICAgICAgdGFibGUuc29ydCgiZmx1eCIpCiAgICAgICAgdGFibGUgPSB0YWJsZVs6Oi0xXVs6bl9zdGFyc10KICAgICAgICByZXR1cm4gbnAudHJhbnNwb3NlKChucC5hc2FycmF5KHRhYmxlWyJ4Y2VudHJvaWQiXSwgZHR5cGU9ZmxvYXQpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5wLmFzYXJyYXkodGFibGVbInljZW50cm9pZCJdLCBkdHlwZT1mbG9hdCkpKQoKICAgIGEsIGIgPSBicmlnaHRfc3RhcnMocmVmZXJlbmNlKSwgYnJpZ2h0X3N0YXJzKGltYWdlKQogICAgaWYgbGVuKGEpIDwgMTAgb3IgbGVuKGIpIDwgMTA6CiAgICAgICAgcmV0dXJuIDAuMCwgMC4wLCB7Im5fcGFpcnMiOiAwLAogICAgICAgICAgICAgICAgICAgICAgICAgICJ3aHkiOiAidG9vIGZldyBzdGFycyB0byBtZWFzdXJlIGEgc2hpZnQifQoKICAgIGRpc3RhbmNlLCBpbmRleCA9IGNLRFRyZWUoYikucXVlcnkoYSwgaz0xKQogICAgb2sgPSBkaXN0YW5jZSA8IG1heF9tYXRjaF9weAogICAgaWYgb2suc3VtKCkgPCAxMDoKICAgICAgICByZXR1cm4gMC4wLCAwLjAsIHsibl9wYWlycyI6IGludChvay5zdW0oKSksCiAgICAgICAgICAgICAgICAgICAgICAgICAgIndoeSI6ICJ0b28gZmV3IHN0YXJzIGNsb3NlIGVub3VnaCB0byBwYWlyIHVwIn0KCiAgICBkeCA9IGJbaW5kZXhbb2tdLCAwXSAtIGFbb2ssIDBdCiAgICBkeSA9IGJbaW5kZXhbb2tdLCAxXSAtIGFbb2ssIDFdCiAgICBteCwgbXkgPSBmbG9hdChucC5tZWRpYW4oZHgpKSwgZmxvYXQobnAubWVkaWFuKGR5KSkKICAgIHJldHVybiBteCwgbXksIHsKICAgICAgICAibl9wYWlycyI6IGludChvay5zdW0oKSksCiAgICAgICAgInNjYXR0ZXJfeCI6IGZsb2F0KDEuNDgyNiAqIG5wLm1lZGlhbihucC5hYnMoZHggLSBteCkpKSwKICAgICAgICAic2NhdHRlcl95IjogZmxvYXQoMS40ODI2ICogbnAubWVkaWFuKG5wLmFicyhkeSAtIG15KSkpLAogICAgfQoKZGVmIHNoaWZ0X2ZyYW1lKGltYWdlLCBkeCwgZHkpOgogICAgIiIiTW92ZSBhIGZyYW1lIGJ5IGEgZnJhY3Rpb24gb2YgYSBwaXhlbCwgdG8gdW5kbyBhIG1lYXN1cmVkIG9mZnNldC4KCiAgICBUaGUgYmxhbmsgbWFyZ2luIGxlZnQgYnkgYW4gZWFybGllciByZXNhbXBsaW5nIGlzIG5vdCBhIG51bWJlciwgYW5kIGEgY3ViaWMKICAgIHNwbGluZSBydW4gb3ZlciBpdCB3b3VsZCBzcHJlYWQgdGhhdCBub3RoaW5nbmVzcyBhY3Jvc3MgdGhlIHdob2xlIGZyYW1lIC0gdGhlCiAgICBmaXJzdCB2ZXJzaW9uIG9mIHRoaXMgZGlkIGV4YWN0bHkgdGhhdCwgYW5kIGV2ZXJ5IG1lYXN1cmVtZW50IGRvd25zdHJlYW0gY2FtZQogICAgYmFjayBlbXB0eS4gU28gdGhlIGdhcHMgYXJlIGZpbGxlZCBiZWZvcmUgdGhlIHNoaWZ0IGFuZCBwdXQgYmFjayBhZnRlciwgd2hpY2gKICAgIGtlZXBzIHRoZW0gZXhhY3RseSB3aGVyZSB0aGV5IHdlcmUuCiAgICAiIiIKICAgIGltcG9ydCBudW1weSBhcyBucAogICAgZnJvbSBzY2lweS5uZGltYWdlIGltcG9ydCBzaGlmdCBhcyBuZHNoaWZ0CgogICAga25vd24gPSBucC5pc2Zpbml0ZShpbWFnZSkKICAgIG1vdmVkID0gbmRzaGlmdChucC53aGVyZShrbm93biwgaW1hZ2UsIDAuMCksICgtZHksIC1keCksIG9yZGVyPTMsIG1vZGU9Im5lYXJlc3QiKQogICAgc3RpbGxfa25vd24gPSBuZHNoaWZ0KGtub3duLmFzdHlwZShmbG9hdCksICgtZHksIC1keCksIG9yZGVyPTEsIG1vZGU9Im5lYXJlc3QiKQogICAgbW92ZWRbc3RpbGxfa25vd24gPCAwLjVdID0gbnAubmFuCiAgICByZXR1cm4gbW92ZWQKCmRlZiByZWZpbmVfYWxpZ25tZW50KHJlZmVyZW5jZSwgaW1hZ2UsIGxhYmVsLCBtaW5fc2hpZnRfcHg9MC4wNSwgcG9zaXRpb25zPU5vbmUpOgogICAgIiIiTWVhc3VyZSB0aGUgbGVmdG92ZXIgc2hpZnQgYWdhaW5zdCBhIHJlZmVyZW5jZSBmcmFtZSBhbmQgdGFrZSBpdCBvdXQuCgogICAgV2hlbiBhIGxpc3Qgb2YgY2F0YWxvZ3VlIHN0YXIgcG9zaXRpb25zIGlzIGF2YWlsYWJsZSB0aGUgc2hpZnQgaXMgbWVhc3VyZWQgb24KICAgIHRob3NlLCB0aGUgc2FtZSBzdGFycyBpbiBib3RoIGZyYW1lcy4gT3RoZXJ3aXNlIHRoZSBmcmFtZXMgYXJlIGFza2VkIHRvIHN1cHBseQogICAgdGhlaXIgb3duIHN0YXJzLCB3aGljaCB3b3JrcyBiZXR3ZWVuIGV4cG9zdXJlcyB0aHJvdWdoIG9uZSBmaWx0ZXIgYW5kIHdvcmtzCiAgICBwb29ybHkgYmV0d2VlbiB0d28gZGlmZmVyZW50IG9uZXMuCiAgICAiIiIKICAgIGZyb21fY2F0YWxvZ3VlID0gb2Zmc2V0X2Zyb21fY2F0YWxvZ3VlKHJlZmVyZW5jZSwgaW1hZ2UsIHBvc2l0aW9ucykKICAgIGlmIGZyb21fY2F0YWxvZ3VlIGlzIG5vdCBOb25lOgogICAgICAgIGR4LCBkeSwgZGlhZyA9IGZyb21fY2F0YWxvZ3VlCiAgICBlbHNlOgogICAgICAgIGR4LCBkeSwgZGlhZyA9IG1lYXN1cmVfb2Zmc2V0KHJlZmVyZW5jZSwgaW1hZ2UpCiAgICBpZiBkaWFnLmdldCgibl9wYWlycyIsIDApIDwgMTA6CiAgICAgICAgcHJpbnQoIiAge306IGNvdWxkIG5vdCBjaGVjayB0aGUgYWxpZ25tZW50IC0ge30iLmZvcm1hdCgKICAgICAgICAgICAgbGFiZWwsIGRpYWcuZ2V0KCJ3aHkiLCAibm90IGVub3VnaCBzdGFycyIpKSkKICAgICAgICByZXR1cm4gaW1hZ2UKICAgIHNpemUgPSAoZHggKiBkeCArIGR5ICogZHkpICoqIDAuNQogICAgaWYgc2l6ZSA8IG1pbl9zaGlmdF9weDoKICAgICAgICBwcmludCgiICB7fTogYWxyZWFkeSBhbGlnbmVkIHRvIHs6LjJmfSBweCwgbm90aGluZyB0byBjb3JyZWN0Ii5mb3JtYXQoCiAgICAgICAgICAgIGxhYmVsLCBzaXplKSkKICAgICAgICByZXR1cm4gaW1hZ2UKICAgIHByaW50KCIgIHt9OiB7fXt9IHN0YXJzIHNheSBpdCBzaXRzICh7OisuMmZ9LCB7OisuMmZ9KSBweCBmcm9tIHRoZSByZWZlcmVuY2UgIgogICAgICAgICAgIihzY2F0dGVyIHs6LjJmfSwgezouMmZ9KTsgc2hpZnRlZCBiYWNrIi5mb3JtYXQoCiAgICAgICAgICAgICAgbGFiZWwsICJjYXRhbG9ndWUgIiBpZiBkaWFnLmdldCgiZnJvbV9jYXRhbG9ndWUiKSBlbHNlICIiLAogICAgICAgICAgICAgIGRpYWdbIm5fcGFpcnMiXSwgZHgsIGR5LAogICAgICAgICAgICAgIGRpYWdbInNjYXR0ZXJfeCJdLCBkaWFnWyJzY2F0dGVyX3kiXSkpCiAgICByZXR1cm4gc2hpZnRfZnJhbWUoaW1hZ2UsIGR4LCBkeSkKCmRlZiBzdGFyX3Bvc2l0aW9ucyhyZWQsIG1hc2s9Tm9uZSwgbl9zdGFycz00MDAsIHRocmVzaG9sZF9zaWdtYT0xNS4wKToKICAgICIiIlBvc2l0aW9ucyBvZiB0aGUgYnJpZ2h0IHBvaW50IHNvdXJjZXMgaW4gdGhlIGJyb2FkLWJhbmQgZnJhbWUuCgogICAgRXZlbiBhZnRlciBtYXRjaGluZyBhbmQgc3VidHJhY3Rpb24gYSBzdGFyIHJhcmVseSB2YW5pc2hlcyBwZXJmZWN0bHk6IGEgc21hbGwKICAgIGVycm9yIGluIHRoZSBmYWN0b3Igb3IgdGhlIGJsdXJyaW5nIGxlYXZlcyBhIHJlc2lkdWFsIGV4YWN0bHkgd2hlcmUgdGhlIHN0YXIKICAgIHdhcy4gVGhvc2UgcmVzaWR1YWxzIGFyZSBjb21wYWN0IGFuZCBicmlnaHQsIHNvIHRoZSBkZXRlY3RvciByZXBvcnRzIHRoZW0gYXMKICAgIEhJSSByZWdpb25zLiBLbm93aW5nIHdoZXJlIHRoZSBzdGFycyB3ZXJlIGxldHMgdXMgdGhyb3cgdGhlbSBvdXQuCiAgICAiIiIKICAgIGZyb20gcGhvdHV0aWxzLmRldGVjdGlvbiBpbXBvcnQgREFPU3RhckZpbmRlcgogICAgZnJvbSBhc3Ryb3B5LnN0YXRzIGltcG9ydCBzaWdtYV9jbGlwcGVkX3N0YXRzCgogICAgbWVhbiwgbWVkaWFuLCBzdGQgPSBzaWdtYV9jbGlwcGVkX3N0YXRzKHJlZCwgc2lnbWE9My4wLCBtYXNrPW1hc2spCiAgICBmaW5kZXIgPSBEQU9TdGFyRmluZGVyKGZ3aG09NC4wLCB0aHJlc2hvbGQ9dGhyZXNob2xkX3NpZ21hICogc3RkKQogICAgdGJsID0gZmluZGVyKHJlZCAtIG1lZGlhbiwgbWFzaz1tYXNrKQogICAgaWYgdGJsIGlzIE5vbmUgb3IgbGVuKHRibCkgPT0gMDoKICAgICAgICByZXR1cm4gbnAuZW1wdHkoKDAsIDIpKQogICAgdGJsLnNvcnQoImZsdXgiKQogICAgdGJsID0gdGJsWzo6LTFdWzpuX3N0YXJzXQogICAgcmV0dXJuIG5wLnRyYW5zcG9zZSgobnAuYXNhcnJheSh0YmxbInhjZW50cm9pZCJdLCBkdHlwZT1mbG9hdCksCiAgICAgICAgICAgICAgICAgICAgICAgICBucC5hc2FycmF5KHRibFsieWNlbnRyb2lkIl0sIGR0eXBlPWZsb2F0KSkpCgpkZWYgc3VidHJhY3RfY29udGludXVtKGhhLCByZWQsIGZhY3RvciwgbWFzaz1Ob25lKToKICAgICIiImZhY3RvciB4IEgtYWxwaGEgLSBSLCB3aXRoIGVhY2ggZnJhbWUncyBvd24gc2t5IHRha2VuIG9mZiBmaXJzdC4KCiAgICBXSFkgVEhFIFNLWSBDT01FUyBPRkYgQkVGT1JFIFRIRSBNVUxUSVBMSUNBVElPTgogICAgQmVjYXVzZSB0aGUgZmFjdG9yIGlzIG1lYW50IHRvIHNjYWxlIHRoZSBzdGFybGlnaHQsIGFuZCB0aGUgZnJhbWUgaG9sZHMgdHdvCiAgICB0aGluZ3M6IHN0YXJsaWdodCBhbmQgc2t5LiBNdWx0aXBseWluZyB0aGUgd2hvbGUgZnJhbWUgbXVsdGlwbGllcyB0aGUgc2t5IHdpdGgKICAgIGl0LiBPbiBhbiBNMzMgcGFpciB0aGUgSC1hbHBoYSBza3kgd2FzIDEyLjYgY291bnRzIGFuZCB0aGUgUiBza3kgOTUuNCwgYW5kIHRoZQogICAgZmFjdG9yIDI1LjcgLSBzbyB0aGUgcmVzdWx0IHNhdCBvbiBhIHBlZGVzdGFsIG9mIDI1LjcgeCAxMi42IC0gOTUuNCA9IDIyOQogICAgY291bnRzLCBzZXZlbnRlZW4gdGltZXMgdGhlIG5vaXNlIG9mIHRoZSBpbWFnZS4KCiAgICBEZXRlY3Rpb24gc3Vydml2ZWQgdGhhdCwgYmVjYXVzZSBpdCBlc3RpbWF0ZXMgYSBiYWNrZ3JvdW5kIGFuZCByZW1vdmVzIGl0CiAgICBiZWZvcmUgbG9va2luZyBmb3IgYW55dGhpbmcuIFdoYXQgZGlkIG5vdCBzdXJ2aXZlIHdhcyB0aGUgcGljdHVyZTogdGhlIHdob2xlCiAgICBkeW5hbWljIHJhbmdlIHdlbnQgaW50byB0aGUgcGVkZXN0YWwsIGZhaW50IGVtaXNzaW9uIHZhbmlzaGVkIGludG8gYSBmbGF0IGdyZXksCiAgICBhbmQgYW55IGRpZmZlcmVuY2UgaW4gZmxhdC1maWVsZGluZyBiZXR3ZWVuIHRoZSB0d28gZmlsdGVycyB3YXMgYW1wbGlmaWVkIGJ5CiAgICB0aGUgZmFjdG9yIGFsb25nIHdpdGggZXZlcnl0aGluZyBlbHNlLiBBbnlvbmUganVkZ2luZyB0aGUgc3VidHJhY3Rpb24gYnkgZXllIC0KICAgIHdoaWNoIGlzIHdoYXQgdGhpcyB0b29sIGFza3MgZm9yIHdoZW5ldmVyIHRoZSBmYWN0b3IgbG9va3MgZG91YnRmdWwgLSB3YXMKICAgIGxvb2tpbmcgYXQgYW4gaW1hZ2UgcnVpbmVkIGJlZm9yZSB0aGV5IHNhdyBpdC4KCiAgICBTa3kgaXMgbWVhc3VyZWQgd2l0aCBjbGlwcGluZywgc28gdGhlIGdhbGF4eSBhbmQgdGhlIHN0YXJzIGRvIG5vdCBkcmFnIGl0LgogICAgIiIiCiAgICBmcm9tIGFzdHJvcHkuc3RhdHMgaW1wb3J0IHNpZ21hX2NsaXBwZWRfc3RhdHMKCiAgICBza3lfaGEgPSBmbG9hdChzaWdtYV9jbGlwcGVkX3N0YXRzKGhhLCBzaWdtYT0zLjAsIG1hc2s9bWFzaylbMV0pCiAgICBza3lfcmVkID0gZmxvYXQoc2lnbWFfY2xpcHBlZF9zdGF0cyhyZWQsIHNpZ21hPTMuMCwgbWFzaz1tYXNrKVsxXSkKICAgIG5ldCA9IGZhY3RvciAqIChoYSAtIHNreV9oYSkgLSAocmVkIC0gc2t5X3JlZCkKICAgIHJldHVybiBuZXQsIHsic2t5X2hhIjogc2t5X2hhLCAic2t5X3JlZCI6IHNreV9yZWQsCiAgICAgICAgICAgICAgICAgInBlZGVzdGFsX2F2b2lkZWQiOiBmYWN0b3IgKiBza3lfaGEgLSBza3lfcmVkfQoKZGVmIGxpbmVfZXhjZXNzKG5ldCwgcmVkLCB4LCB5LCByYWRpdXNfcHg9NS4wKToKICAgICIiIkhvdyBtdWNoIGxpbmUgZW1pc3Npb24gZWFjaCBkZXRlY3Rpb24gaGFzLCBuZXh0IHRvIGl0cyBvd24gY29udGludXVtLgoKICAgIFRoZSByYXRpbyBvZiB3aGF0IGlzIGxlZnQgYWZ0ZXIgdGhlIHN1YnRyYWN0aW9uIHRvIHdoYXQgdGhlIGNvbnRpbnV1bSBmcmFtZQogICAgaG9sZHMgYXQgdGhlIHNhbWUgcGxhY2UuIEEgZm9yZWdyb3VuZCBzdGFyIGhhcyBub3RoaW5nIGxlZnQgb25jZSB0aGUKICAgIHN1YnRyYWN0aW9uIGlzIHJpZ2h0LCBzbyBpdHMgcmF0aW8gc2l0cyBuZWFyIHplcm8uIEFuIEhJSSByZWdpb24gaXMgZW1pc3Npb24KICAgIHdpdGggb25seSBhIG1vZGVzdCBjb250aW51dW0gdW5kZXIgaXQsIHNvIGl0cyByYXRpbyBpcyBvZiBvcmRlciBvbmUgb3IgbW9yZS4KICAgICIiIgogICAgZnJvbSBwaG90dXRpbHMuYXBlcnR1cmUgaW1wb3J0IENpcmN1bGFyQXBlcnR1cmUsIGFwZXJ0dXJlX3Bob3RvbWV0cnkKCiAgICBwb3NpdGlvbnMgPSBucC5jb2x1bW5fc3RhY2soW25wLmFzYXJyYXkoeCwgZHR5cGU9ZmxvYXQpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBucC5hc2FycmF5KHksIGR0eXBlPWZsb2F0KV0pCiAgICBpZiBub3QgbGVuKHBvc2l0aW9ucyk6CiAgICAgICAgcmV0dXJuIG5wLnplcm9zKDApCiAgICBhcCA9IENpcmN1bGFyQXBlcnR1cmUocG9zaXRpb25zLCByPXJhZGl1c19weCkKICAgIGZfbmV0ID0gbnAuYXNhcnJheShhcGVydHVyZV9waG90b21ldHJ5KG5wLm5hbl90b19udW0obmV0KSwgYXApWyJhcGVydHVyZV9zdW0iXSwKICAgICAgICAgICAgICAgICAgICAgICBkdHlwZT1mbG9hdCkKICAgIGZfcmVkID0gbnAuYXNhcnJheShhcGVydHVyZV9waG90b21ldHJ5KG5wLm5hbl90b19udW0ocmVkKSwgYXApWyJhcGVydHVyZV9zdW0iXSwKICAgICAgICAgICAgICAgICAgICAgICBkdHlwZT1mbG9hdCkKICAgIHdpdGggbnAuZXJyc3RhdGUoZGl2aWRlPSJpZ25vcmUiLCBpbnZhbGlkPSJpZ25vcmUiKToKICAgICAgICByZXR1cm4gZl9uZXQgLyBucC53aGVyZShucC5hYnMoZl9yZWQpID4gMS4wLCBucC5hYnMoZl9yZWQpLCBucC5uYW4pCgpkZWYgZmlsdGVyX2RldGVjdGlvbnMoeCwgeSwgY2VudHJlLCByX21heF9weD1Ob25lLCBzdGFyX3h5PU5vbmUsIHN0YXJfcmFkaXVzX3B4PTYuMCwKICAgICAgICAgICAgICAgICAgICAgIGV4Y2Vzcz1Ob25lLCBtaW5fZXhjZXNzPTAuNSwKICAgICAgICAgICAgICAgICAgICAgIG1pbm9yX2F4aXM9Tm9uZSwgZWxvbmdhdGlvbj1Ob25lLAogICAgICAgICAgICAgICAgICAgICAgbWluX3dpZHRoX3B4PTEuNSwgbWF4X2Vsb25nYXRpb249Mi41KToKICAgICIiIkRyb3AgZGV0ZWN0aW9ucyB0aGF0IGNhbm5vdCBiZSBISUkgcmVnaW9ucyBiZWxvbmdpbmcgdG8gdGhpcyBnYWxheHkuCgogICAgVHdvIGN1dHMsIGJvdGggYmx1bnQgYW5kIGJvdGggbmVjZXNzYXJ5LgoKICAgIE91dHNpZGUgdGhlIGdhbGF4eTogSElJIHJlZ2lvbnMgbGl2ZSBpbiB0aGUgZGlzaywgYSBmcmFtZSBpcyB1c3VhbGx5IG11Y2ggd2lkZXIKICAgIHRoYW4gdGhlIGdhbGF4eSwgYW5kIGV2ZXJ5dGhpbmcgZm91bmQgb3V0IGluIHRoZSBlbXB0eSBjb3JuZXJzIGlzIG5vaXNlIG9yIGFuCiAgICBhcnRlZmFjdC4gQ3V0dGluZyBhdCB0aGUgaXNvcGhvdGFsIHJhZGl1cyBSMjUgcmVtb3ZlcyB0aGVtIHdpdGhvdXQgdG91Y2hpbmcgdGhlCiAgICBkaXNrLCBhbmQgUjI1IGlzIGEgcHVibGlzaGVkIG51bWJlciByYXRoZXIgdGhhbiBhIGp1ZGdlbWVudCBjYWxsLgoKICAgIE9uIHRvcCBvZiBhIHN0YXI6IHN1YnRyYWN0aW9uIHJlc2lkdWFscyBzaXQgZXhhY3RseSB3aGVyZSB0aGUgc3RhcnMgd2VyZS4KCiAgICBXSFkgVEhFIFNUQVIgQ1VUIE5FRURTIFRIRSBTRUNPTkQgVEVTVAogICAgQSBicmlnaHQgSElJIHJlZ2lvbiBjb250YWlucyB0aGUgeW91bmcgY2x1c3RlciB0aGF0IGxpZ2h0cyBpdCwgYW5kIGEgeW91bmcKICAgIGNsdXN0ZXIgaXMgYSBwb2ludCBzb3VyY2UgaW4gdGhlIGNvbnRpbnV1bSBmcmFtZSAtIHNvIGEgY3V0IHRoYXQgcmVtb3ZlcwogICAgZXZlcnl0aGluZyBzaXR0aW5nIG9uIGEgcG9pbnQgc291cmNlIHJlbW92ZXMgdGhlIGJyaWdodGVzdCByZWdpb25zIGluIHRoZQogICAgZ2FsYXh5IGFsb25nIHdpdGggdGhlIGZvcmVncm91bmQgc3RhcnMuIE9uIHRoZSBNODMgZnJhbWVzIHRoaXMgd2FzIG1lYXN1cmVkOgogICAgb2YgZm9ydHkgZGV0ZWN0aW9ucyB0aGUgc3RhciBjdXQgdGhyZXcgYXdheSBpbnNpZGUgdGhlIGdhbGF4eSwgdHdlbnR5LW5pbmUgaGFkCiAgICByZWFsIGxpbmUgZW1pc3Npb24uIFRoZWlyIGVtaXNzaW9uIGlzIHdoYXQgc2VwYXJhdGVzIHRoZW0gLSB0aGUgcmVnaW9ucyBrZXB0CiAgICBieSB0aGUgcmVzdCBvZiB0aGUgcGlwZWxpbmUgc2hvd2VkIGEgcmF0aW8gYXJvdW5kIDIuMywgYW5kIHRoZSBwb2ludCBzb3VyY2VzCiAgICBvdXQgaW4gdGhlIGVtcHR5IGZpZWxkLCB3aGljaCByZWFsbHkgYXJlIHN0YXJzLCBhcm91bmQgMC4xNy4gU28gYSBkZXRlY3Rpb24KICAgIHNpdHRpbmcgb24gYSBwb2ludCBzb3VyY2UgaXMgc3RpbGwgZHJvcHBlZCwgdW5sZXNzIGl0IGlzIGVtaXR0aW5nLgogICAgIiIiCiAgICB4ID0gbnAuYXNhcnJheSh4LCBkdHlwZT1mbG9hdCkKICAgIHkgPSBucC5hc2FycmF5KHksIGR0eXBlPWZsb2F0KQogICAga2VlcCA9IG5wLm9uZXMoeC5zaXplLCBkdHlwZT1ib29sKQogICAgcmVwb3J0ID0ge30KCiAgICAjIFNoYXBlLiBBbiBISUkgcmVnaW9uIGlzIGEgYm9keSBvZiBnYXMsIHNvIGl0IGhhcyB3aWR0aDsgd2hhdCBpcyBsZWZ0IG92ZXIKICAgICMgZnJvbSBhbiBpbXBlcmZlY3Qgc3VidHJhY3Rpb24gaXMgYSB0aHJlYWQgLSBhIHBpeGVsIG9yIHR3byBhY3Jvc3MsIG9mdGVuCiAgICAjIGN1cnZlZCwgZm9sbG93aW5nIHRoZSBlZGdlIG9mIHdoYXRldmVyIGRpZCBub3QgY2FuY2VsLiBNZWFzdXJlZCBvbiB0aGUgTTgzCiAgICAjIGZyYW1lcywgYmxvYnMgaW5zaWRlIHRoZSBnYWxheHkgYXJlIHR5cGljYWxseSAxLjkgcHggd2lkZSBhbmQgYmxvYnMgb3V0IGluCiAgICAjIHRoZSBlbXB0eSBmaWVsZCwgd2hlcmUgbm90aGluZyByZWFsIGNhbiBiZSwgMC41IHB4LiBUaGF0IG9uZSBudW1iZXIgZG9lcwogICAgIyBtb3JlIHRvIGNsZWFuIHRoZSBjYXRhbG9ndWUgdGhhbiBhbnkgYnJpZ2h0bmVzcyBjdXQuCiAgICBpZiBtaW5vcl9heGlzIGlzIG5vdCBOb25lOgogICAgICAgIHdpZGVfZW5vdWdoID0gbnAuaXNmaW5pdGUobWlub3JfYXhpcykgJiAobnAuYXNhcnJheShtaW5vcl9heGlzKSA+PSBtaW5fd2lkdGhfcHgpCiAgICAgICAgcmVwb3J0WyJ0b28gdGhpbiB0byBiZSBhIHJlZ2lvbiJdID0gaW50KCh+d2lkZV9lbm91Z2ggJiBrZWVwKS5zdW0oKSkKICAgICAgICBrZWVwICY9IHdpZGVfZW5vdWdoCgogICAgaWYgZWxvbmdhdGlvbiBpcyBub3QgTm9uZToKICAgICAgICByb3VuZF9lbm91Z2ggPSBucC5pc2Zpbml0ZShlbG9uZ2F0aW9uKSAmIChucC5hc2FycmF5KGVsb25nYXRpb24pIDw9IG1heF9lbG9uZ2F0aW9uKQogICAgICAgIHJlcG9ydFsiYSBzdHJlYWssIG5vdCBhIHJlZ2lvbiJdID0gaW50KCh+cm91bmRfZW5vdWdoICYga2VlcCkuc3VtKCkpCiAgICAgICAga2VlcCAmPSByb3VuZF9lbm91Z2gKCiAgICBpZiByX21heF9weCBpcyBub3QgTm9uZToKICAgICAgICBpbnNpZGUgPSBucC5oeXBvdCh4IC0gY2VudHJlWzBdLCB5IC0gY2VudHJlWzFdKSA8PSByX21heF9weAogICAgICAgIHJlcG9ydFsib3V0c2lkZSB0aGUgZ2FsYXh5Il0gPSBpbnQoKH5pbnNpZGUgJiBrZWVwKS5zdW0oKSkKICAgICAgICBrZWVwICY9IGluc2lkZQoKICAgIGlmIHN0YXJfeHkgaXMgbm90IE5vbmUgYW5kIGxlbihzdGFyX3h5KToKICAgICAgICBkID0gbnAuaHlwb3QoeFs6LCBOb25lXSAtIHN0YXJfeHlbTm9uZSwgOiwgMF0sIHlbOiwgTm9uZV0gLSBzdGFyX3h5W05vbmUsIDosIDFdKQogICAgICAgIG9uX3N0YXIgPSBkLm1pbihheGlzPTEpIDwgc3Rhcl9yYWRpdXNfcHgKICAgICAgICBpZiBleGNlc3MgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIGVtaXR0aW5nID0gbnAuaXNmaW5pdGUoZXhjZXNzKSAmIChleGNlc3MgPiBtaW5fZXhjZXNzKQogICAgICAgICAgICByZXNjdWVkID0gaW50KChvbl9zdGFyICYgZW1pdHRpbmcgJiBrZWVwKS5zdW0oKSkKICAgICAgICAgICAgaWYgcmVzY3VlZDoKICAgICAgICAgICAgICAgIHJlcG9ydFsib24gYSBzdGFyIGJ1dCBlbWl0dGluZywga2VwdCJdID0gcmVzY3VlZAogICAgICAgICAgICBvbl9zdGFyID0gb25fc3RhciAmIH5lbWl0dGluZwogICAgICAgIHJlcG9ydFsib24gYSBzdGFyIl0gPSBpbnQoKG9uX3N0YXIgJiBrZWVwKS5zdW0oKSkKICAgICAgICBrZWVwICY9IH5vbl9zdGFyCgogICAgcmV0dXJuIGtlZXAsIHJlcG9ydAoKZGVmIGNvbXBsZXRlbmVzc192c19yYWRpdXMocmVzaWQsIGNlbnRyZSwgZGV0ZWN0X2t3YXJncywgcl9lZGdlcywKICAgICAgICAgICAgICAgICAgICAgICAgICAgbl9wZXJfcmluZz0zMCwgcGVha19jb3VudHM9MS4wLCBmd2htX3B4PTYuMCwgcm5nX3NlZWQ9MTEpOgogICAgIiIiV2hhdCBmcmFjdGlvbiBvZiBpbmplY3RlZCByZWdpb25zIGNvbWVzIGJhY2ssIHJpbmcgYnkgcmluZy4KCiAgICBXSFkgQk9USEVSCiAgICBBIHJhZGlhbCBwcm9maWxlIGlzIHRoZSBtZWFzdXJlbWVudCBtb3N0IGVhc2lseSBmYWtlZCBieSBhIGRldGVjdGlvbiBsaW1pdCB0aGF0CiAgICBjaGFuZ2VzIHdpdGggcmFkaXVzLiBUaGUgY2VudHJlIG9mIGEgZ2FsYXh5IGlzIGJyaWdodCBhbmQgY3Jvd2RlZCwgdGhlIG91dHNraXJ0cwogICAgZGFyayBhbmQgZW1wdHksIHNvIHRoZSBmYWludGVzdCB0aGluZyBzdGlsbCB2aXNpYmxlIGlzIG5vdCB0aGUgc2FtZSBpbiBib3RoLiBJZgogICAgZGV0ZWN0aW9uIGdldHMgaGFyZGVyIGlud2FyZHMsIHRoZSBwcm9maWxlIGdyb3dzIGEgY2VudHJhbCBob2xlIHRoYXQgYmVsb25ncyB0bwogICAgdGhlIHNlYXJjaCwgbm90IHRvIHRoZSBnYWxheHkuCgogICAgSE9XCiAgICBEcm9wIGFydGlmaWNpYWwgcmVnaW9ucyBvZiBrbm93biBwZWFrIGJyaWdodG5lc3MgaW50byB0aGUgcmVhbCBpbWFnZSwgcnVuCiAgICBleGFjdGx5IHRoZSBzYW1lIGRldGVjdGlvbiBvbiBpdCwgYW5kIGNvdW50IGhvdyBtYW55IGNvbWUgYmFjaywgcmluZyBieSByaW5nLgoKICAgIHBlYWtfY291bnRzIGlzIHRoZSBwZWFrIGhlaWdodCBvZiB0aGUgaW5qZWN0ZWQgcmVnaW9uIGFuZCBzaG91bGQgYmUgc2V0IGluIHVuaXRzCiAgICBvZiB0aGUgYmFja2dyb3VuZCBub2lzZS4gU2NhbGluZyBpbmplY3RlZCByZWdpb25zIHRvIHRoZSBicmlnaHRuZXNzIG9mIGFuCiAgICBhdmVyYWdlIHJlYWwgb25lIHB1dHMgdGhlbSBodW5kcmVkcyBvZiBzaWdtYSBhYm92ZSB0aGUgdGhyZXNob2xkLCB3aGVyZQogICAgZXZlcnl0aGluZyBpcyBmb3VuZCBldmVyeXdoZXJlIGFuZCB0aGUgdGVzdCBtZWFzdXJlcyBub3RoaW5nLgogICAgIiIiCiAgICBmcm9tIHNjaXB5Lm5kaW1hZ2UgaW1wb3J0IGdhdXNzaWFuX2ZpbHRlcgoKICAgIHJuZyA9IG5wLnJhbmRvbS5kZWZhdWx0X3JuZyhybmdfc2VlZCkKICAgIHNpZ21hID0gZndobV9weCAvIDIuMzU0OAogICAgbnksIG54ID0gcmVzaWQuc2hhcGUKICAgIHJlY292ZXJlZCA9IG5wLnplcm9zKGxlbihyX2VkZ2VzKSAtIDEpCiAgICBpbmplY3RlZCA9IG5wLnplcm9zKGxlbihyX2VkZ2VzKSAtIDEpCgogICAgZm9yIGsgaW4gcmFuZ2UobGVuKHJfZWRnZXMpIC0gMSk6CiAgICAgICAgcl9sbywgcl9oaSA9IHJfZWRnZXNba10sIHJfZWRnZXNbayArIDFdCiAgICAgICAgZmFrZSA9IG5wLnplcm9zX2xpa2UocmVzaWQpCiAgICAgICAgdHJ1dGggPSBbXQogICAgICAgIGZvciBfIGluIHJhbmdlKG5fcGVyX3JpbmcpOgogICAgICAgICAgICByYWQgPSBtYXRoLnNxcnQocm5nLnVuaWZvcm0ocl9sbyAqKiAyLCByX2hpICoqIDIpKQogICAgICAgICAgICB0aCA9IHJuZy51bmlmb3JtKDAuMCwgMi4wICogbWF0aC5waSkKICAgICAgICAgICAgcHggPSBjZW50cmVbMF0gKyByYWQgKiBtYXRoLmNvcyh0aCkKICAgICAgICAgICAgcHkgPSBjZW50cmVbMV0gKyByYWQgKiBtYXRoLnNpbih0aCkKICAgICAgICAgICAgaWYgbm90ICg4IDwgcHggPCBueCAtIDggYW5kIDggPCBweSA8IG55IC0gOCk6CiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICBmYWtlW2ludChyb3VuZChweSkpLCBpbnQocm91bmQocHgpKV0gKz0gcGVha19jb3VudHMKICAgICAgICAgICAgdHJ1dGguYXBwZW5kKChweCwgcHkpKQogICAgICAgIGlmIG5vdCB0cnV0aDoKICAgICAgICAgICAgY29udGludWUKCiAgICAgICAgIyBBIGRlbHRhIHNtb290aGVkIGJ5IGEgR2F1c3NpYW4gaGFzIHBlYWsgYW1wbGl0dWRlLygyIHBpIHNpZ21hXjIpOwogICAgICAgICMgbXVsdGlwbHlpbmcgaXQgYmFjayBnaXZlcyBhIHJlZ2lvbiB3aG9zZSBQRUFLIGlzIHBlYWtfY291bnRzLgogICAgICAgIGZha2UgPSBnYXVzc2lhbl9maWx0ZXIoZmFrZSwgc2lnbWEpICogKDIuMCAqIG1hdGgucGkgKiBzaWdtYSAqKiAyKQogICAgICAgIGNhdCwgXywgXyA9IGRldGVjdF9yZWdpb25zKHJlc2lkICsgZmFrZSwgKipkZXRlY3Rfa3dhcmdzKQoKICAgICAgICBpZiBjYXQgaXMgTm9uZToKICAgICAgICAgICAgaW5qZWN0ZWRba10gKz0gbGVuKHRydXRoKQogICAgICAgICAgICBjb250aW51ZQoKICAgICAgICBmeCA9IG5wLmFzYXJyYXkoY2F0LnhjZW50cm9pZCwgZHR5cGU9ZmxvYXQpCiAgICAgICAgZnkgPSBucC5hc2FycmF5KGNhdC55Y2VudHJvaWQsIGR0eXBlPWZsb2F0KQogICAgICAgICMgU2VnbWVudHMgdG91Y2hpbmcgdGhlIG1hc2sgY29tZSBiYWNrIHdpdGggYSBjZW50cm9pZCBvZiBOYU4uIExlZnQgaW4sIGEKICAgICAgICAjIHNpbmdsZSBOYU4gbWFrZXMgZXZlcnkgZGlzdGFuY2UgY29tcGFyaXNvbiBiZWxvdyBmYWxzZSBhbmQgdGhlIG1lYXN1cmVkCiAgICAgICAgIyBjb21wbGV0ZW5lc3MgY29sbGFwc2VzIHRvIHplcm8gZXZlcnl3aGVyZS4KICAgICAgICBmaW5pdGUgPSBucC5pc2Zpbml0ZShmeCkgJiBucC5pc2Zpbml0ZShmeSkKICAgICAgICBmeCwgZnkgPSBmeFtmaW5pdGVdLCBmeVtmaW5pdGVdCgogICAgICAgIGZvciAodHgsIHR5KSBpbiB0cnV0aDoKICAgICAgICAgICAgaW5qZWN0ZWRba10gKz0gMQogICAgICAgICAgICBpZiBmeC5zaXplIGFuZCBucC5taW4obnAuaHlwb3QoZnggLSB0eCwgZnkgLSB0eSkpIDwgMy4wOgogICAgICAgICAgICAgICAgcmVjb3ZlcmVkW2tdICs9IDEKCiAgICB3aXRoIG5wLmVycnN0YXRlKGludmFsaWQ9Imlnbm9yZSIsIGRpdmlkZT0iaWdub3JlIik6CiAgICAgICAgZnJhYyA9IG5wLndoZXJlKGluamVjdGVkID4gMCwgcmVjb3ZlcmVkIC8gaW5qZWN0ZWQsIG5wLm5hbikKICAgIHJldHVybiBmcmFjLCBpbmplY3RlZCwgcmVjb3ZlcmVkCgpkZWYgY2hvb3NlX3N1YnRyYWN0aW9uKGhhLCByZWQsIHBvc2l0aW9ucywgcHJlZml4LCBjZW50cmU9Tm9uZSwgcjI1X3B4PU5vbmUsCiAgICAgICAgICAgICAgICAgICAgICAgZmFjdG9ycz1Ob25lLCBibHVycz1Ob25lLCBhc2s9Tm9uZSk6CiAgICAiIiJPZmZlciB0aHJlZSBzdWJ0cmFjdGlvbnMgYXMgcGljdHVyZXMgYW5kIGxldCB0aGUgcGVyc29uIHBpY2sgb25lLgoKICAgIFdIWSBBIFBFUlNPTiBERUNJREVTIFRISVMKICAgIEJlY2F1c2UgZXZlcnkgYXV0b21hdGljIGNyaXRlcmlvbiB0cmllZCBoZXJlIGRpc2FncmVlZCB3aXRoIHRoZSBleWUsIGFuZCB0aGUKICAgIGV5ZSB3YXMgcmlnaHQuIE1hdGNoaW5nIHN0YXIgd2lkdGhzIGJ5IHRoZWlyIG1lYXN1cmVkIEZXSE0gcHJlc2NyaWJlZCBhIGJsdXIKICAgIHdpdGggdGhlIHNpZ24gd3JvbmcuIENob29zaW5nIHRoZSBmYWN0b3IgYXMgdGhlIG1lYW4gb3ZlciBzdGFycyBsZWZ0IHR3aWNlIGFzCiAgICBtYW55IHZpc2libGUgc3RhciByZXNpZHVhbHMgYXMgYSBoYW5kLXR1bmVkIHZhbHVlIGEgZmV3IHBlcmNlbnQgbG93ZXIuIFRoZQogICAgcXVhbnRpdHkgdGhhdCBhY3R1YWxseSBtYXR0ZXJzIC0gd2hldGhlciBhIHN0YXIgaGFzIHZhbmlzaGVkIGludG8gdGhlIHNreSBvcgogICAgc3RpbGwgc2hvd3MgYXMgYSBzcG90IHRoYXQgYSBkZXRlY3RvciB3aWxsIGNhbGwgYW4gSElJIHJlZ2lvbiAtIGlzIHNvbWV0aGluZyBhCiAgICBwZXJzb24gcmVhZHMgb2ZmIGEgZnJhbWUgaW4gYSBzZWNvbmQgYW5kIG5vIHNpbmdsZSBudW1iZXIgaGVyZSBoYXMgY2FwdHVyZWQuCgogICAgV0hZIFRIUkVFIEFORCBOT1QgQSBOVU1CRVIgT04gVEhFIFNDUkVFTgogICAgQSBudW1iZXIgY2Fubm90IGJlIGp1ZGdlZCB3aXRob3V0IGtub3dpbmcgd2hhdCBpdCBsb29rcyBsaWtlLiBUaHJlZSBmcmFtZXMKICAgIHdyaXR0ZW4gdG8gZGlzaywgZGlmZmVyaW5nIGVub3VnaCB0byB0ZWxsIGFwYXJ0LCBhbmQgb25lIHF1ZXN0aW9uLgoKICAgIEhPVyBUSEUgVEhSRUUgQVJFIFBJQ0tFRAogICAgVGhlIHdob2xlIGdyaWQgb2YgZmFjdG9yIGFuZCBibHVyIGlzIHNjb3JlZCBvbiBzdGFycyB0aGF0IGxpZSBvdXRzaWRlIHRoZQogICAgZ2FsYXh5LCB3aGVyZSBub3RoaW5nIGJ1dCBza3kgYW5kIHRoZSBzdGFyIGlzIGluIHRoZSBhcGVydHVyZS4gVGhlIHNjb3JlIGlzIGhvdwogICAgZmxhdCB3aGF0IGlzIGxlZnQgb2YgdGhlIG1lZGlhbiBzdGFyIGlzIC0gYSBnb29kIHN1YnRyYWN0aW9uIGxlYXZlcyBubyBjb3JlIGFuZAogICAgbm8gcmluZy4gVGhlIGJlc3QgaXMgb2ZmZXJlZCwgd2l0aCBvbmUgY2FuZGlkYXRlIGVpdGhlciBzaWRlIG9mIGl0IGluIGZhY3RvciwKICAgIHNvIHRoYXQgdGhlIHRocmVlIHNwYW4gdmlzaWJseSBsZXNzIGFuZCB2aXNpYmx5IG1vcmUgc3VidHJhY3Rpb24gcmF0aGVyIHRoYW4KICAgIHRocmVlIHZlcnNpb25zIG9mIHRoZSBzYW1lIGFuc3dlci4KCiAgICBSZXR1cm5zIChoYSwgcmVkLCBmYWN0b3IsIGRpYWdub3N0aWNzKS4gV2l0aG91dCBhIHF1ZXN0aW9uIHRvIGFzayAtIGEKICAgIG5vbi1pbnRlcmFjdGl2ZSBydW4gLSB0aGUgbWlkZGxlIG9uZSBpcyB0YWtlbiBhbmQgc2FpZCBzby4KICAgICIiIgogICAgZnJvbSBzY2lweS5uZGltYWdlIGltcG9ydCBnYXVzc2lhbl9maWx0ZXIKICAgIGZyb20gYXN0cm9weS5zdGF0cyBpbXBvcnQgc2lnbWFfY2xpcHBlZF9zdGF0cwoKICAgIGRpYWcgPSB7fQogICAgaWYgcG9zaXRpb25zIGlzIE5vbmUgb3IgbGVuKHBvc2l0aW9ucykgPCAzMDoKICAgICAgICByZXR1cm4gaGEsIHJlZCwgTm9uZSwgeyJ3aHkiOiAidG9vIGZldyBzdGFycyB0byBqdWRnZSBhIHN1YnRyYWN0aW9uIG9uIn0KCiAgICBwb3MgPSBucC5hc2FycmF5KHBvc2l0aW9ucywgZHR5cGU9ZmxvYXQpCiAgICBpZiBjZW50cmUgaXMgbm90IE5vbmUgYW5kIHIyNV9weDoKICAgICAgICBvdXRzaWRlID0gbnAuaHlwb3QocG9zWzosIDBdIC0gY2VudHJlWzBdLCBwb3NbOiwgMV0gLSBjZW50cmVbMV0pID4gMS4zICogcjI1X3B4CiAgICAgICAgaWYgb3V0c2lkZS5zdW0oKSA+PSAzMDoKICAgICAgICAgICAgcG9zID0gcG9zW291dHNpZGVdCiAgICBkaWFnWyJuX3N0YXJzIl0gPSBpbnQobGVuKHBvcykpCgogICAgIyBXSEVSRSBUSEUgUkFOR0UgQ09NRVMgRlJPTQogICAgIyBGcm9tIHRoaXMgcGFpciBvZiBmcmFtZXMsIG5ldmVyIGZyb20gYSBudW1iZXIgdHlwZWQgaW4uIFRoZSBmYWN0b3IgdGhhdAogICAgIyBjYW5jZWxzIGEgc3RhciBpcyBtZWFzdXJlZCBvbiB0aGUgc3RhcnMgdGhlbXNlbHZlcyBhbmQgdGhlIGdyaWQgaXMgbGFpZCBvdXQKICAgICMgYXJvdW5kIGl0LCBzbyBhIGdhbGF4eSB3aG9zZSBmYWN0b3IgaXMgdHdlbnR5IGlzIHNlYXJjaGVkIG5lYXIgdHdlbnR5IGFuZCBvbmUKICAgICMgd2hvc2UgZmFjdG9yIGlzIHR3ZWx2ZSBpcyBzZWFyY2hlZCBuZWFyIHR3ZWx2ZS4gQSBncmlkIHdyaXR0ZW4gYnkgaGFuZCB3b3VsZAogICAgIyBoYXZlIGJlZW4gdGhlIHJhbmdlIG9mIHdoaWNoZXZlciBnYWxheHkgaXQgd2FzIHdyaXR0ZW4gZm9yLgogICAgaWYgZmFjdG9ycyBpcyBOb25lOgogICAgICAgIHBhc3MgICMgZnJvbSBjb250aW51dW1fc3VidHJhY3Rpb24sIGlubGluZWQgYWJvdmUKICAgICAgICB0cnk6CiAgICAgICAgICAgIGNlbnRyZV9mYWN0b3IgPSBmbG9hdChjb25zdGFudF9mYWN0b3IoaGEsIHJlZCwgcG9zaXRpb25zPXBvcylbMF0pCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgY2VudHJlX2ZhY3RvciA9IE5vbmUKICAgICAgICBpZiBub3QgY2VudHJlX2ZhY3RvciBvciBub3QgbnAuaXNmaW5pdGUoY2VudHJlX2ZhY3Rvcikgb3IgY2VudHJlX2ZhY3RvciA8PSAwOgogICAgICAgICAgICByZXR1cm4gaGEsIHJlZCwgTm9uZSwgeyJ3aHkiOiAiY291bGQgbm90IG1lYXN1cmUgYSBmYWN0b3IgdG8gc2VhcmNoIGFyb3VuZCJ9CiAgICAgICAgZGlhZ1sibWVhc3VyZWRfZmFjdG9yIl0gPSBjZW50cmVfZmFjdG9yCgogICAgaCA9IDgKICAgIHl5LCB4eCA9IG5wLm1ncmlkWy1oOmggKyAxLCAtaDpoICsgMV0KICAgIHJyID0gbnAuaHlwb3QoeHgsIHl5KQogICAgZWRnZSA9IHJyID4gaCAtIDEuNQogICAgbnksIG54ID0gaGEuc2hhcGUKICAgIHNlbCA9IFsoaW50KHJvdW5kKHgpKSwgaW50KHJvdW5kKHkpKSkgZm9yIHgsIHkgaW4gcG9zCiAgICAgICAgICAgaWYgaCA8IGludChyb3VuZCh4KSkgPCBueCAtIGggYW5kIGggPCBpbnQocm91bmQoeSkpIDwgbnkgLSBoXQogICAgaWYgbGVuKHNlbCkgPCAzMDoKICAgICAgICByZXR1cm4gaGEsIHJlZCwgTm9uZSwgeyJ3aHkiOiAidGhlIHN0YXJzIGFsbCBmYWxsIHRvbyBuZWFyIHRoZSBmcmFtZSBlZGdlIn0KCiAgICBkZWYgc3RhbXBzKGltZyk6CiAgICAgICAgb3V0ID0gbnAuZW1wdHkoKGxlbihzZWwpLCAyICogaCArIDEsIDIgKiBoICsgMSksIGR0eXBlPW5wLmZsb2F0MzIpCiAgICAgICAgZm9yIGksICh4LCB5KSBpbiBlbnVtZXJhdGUoc2VsKToKICAgICAgICAgICAgcyA9IG5wLm5hbl90b19udW0oaW1nW3kgLSBoOnkgKyBoICsgMSwgeCAtIGg6eCArIGggKyAxXSkKICAgICAgICAgICAgb3V0W2ldID0gcyAtIG5wLm1lZGlhbihzW2VkZ2VdKQogICAgICAgIHJldHVybiBvdXQKCiAgICAjIFRIRVJFIElTIE5PIFJBTkdFCiAgICAjIEFuIHVwcGVyIGFuZCBsb3dlciBib3VuZCBvbiB0aGUgc2VhcmNoIGlzIGEgZ3Vlc3MgYWJvdXQgdGhlIGFuc3dlciwgYW5kIHRoZQogICAgIyBhbnN3ZXIgaXMgd2hhdCBpcyBiZWluZyBsb29rZWQgZm9yLiBTbyB0aGUgc2VhcmNoIHN0YXJ0cyBzbWFsbCBhbmQgZ3Jvd3M6IGFzCiAgICAjIGxvbmcgYXMgdGhlIGJlc3QgcG9pbnQgaXQgaGFzIGZvdW5kIHNpdHMgb24gYW4gZWRnZSBvZiB3aGF0IGhhcyBiZWVuIHRyaWVkLAogICAgIyB0aGF0IGVkZ2UgbW92ZXMgb3V0IGFuZCBpdCBsb29rcyBhZ2Fpbi4gSXQgc3RvcHMgd2hlbiB0aGUgYmVzdCBwb2ludCBoYXMKICAgICMgbmVpZ2hib3VycyBvbiBhbGwgc2lkZXMsIHdoaWNoIGlzIHRoZSBvbmx5IGV2aWRlbmNlIHRoYXQgaXQgaXMgYSBtaW5pbXVtCiAgICAjIHJhdGhlciB0aGFuIHRoZSBlbmQgb2YgdGhlIHBhcGVyLgogICAgZl9zdGVwID0gMC4wMjUgKiBjZW50cmVfZmFjdG9yIGlmIGZhY3RvcnMgaXMgTm9uZSBlbHNlIE5vbmUKICAgIGJfc3RlcCA9IDAuMjUKICAgIGlmIGZhY3RvcnMgaXMgTm9uZToKICAgICAgICBmX2xvLCBmX2hpID0gLTIsIDIgICAgICAgICAgICAgICAgICAjIGluIHN0ZXBzLCByZWxhdGl2ZSB0byB0aGUgbWVhc3VyZWQgZmFjdG9yCiAgICBlbHNlOgogICAgICAgIGZfbG8gPSBmX2hpID0gTm9uZQogICAgYl9sbywgYl9oaSA9IC0yLCAyCgogICAgY2FjaGUgPSB7fQogICAgc3RhbXBfY2FjaGUgPSB7fQoKICAgIGRlZiBzY29yZShmaSwgYmkpOgogICAgICAgIGtleSA9IChmaSwgYmkpCiAgICAgICAgaWYga2V5IGluIGNhY2hlOgogICAgICAgICAgICByZXR1cm4gY2FjaGVba2V5XQogICAgICAgIGJsdXIgPSBiaSAqIGJfc3RlcAogICAgICAgIGlmIGJpIG5vdCBpbiBzdGFtcF9jYWNoZToKICAgICAgICAgICAgQSA9IGdhdXNzaWFuX2ZpbHRlcihoYSwgYmx1cikgaWYgYmx1ciA+IDAgZWxzZSBoYQogICAgICAgICAgICBSID0gZ2F1c3NpYW5fZmlsdGVyKHJlZCwgLWJsdXIpIGlmIGJsdXIgPCAwIGVsc2UgcmVkCiAgICAgICAgICAgIHN0YW1wX2NhY2hlW2JpXSA9IChzdGFtcHMoQSksIHN0YW1wcyhSKSkKICAgICAgICBTQSwgU1IgPSBzdGFtcF9jYWNoZVtiaV0KICAgICAgICBmID0gKGNlbnRyZV9mYWN0b3IgKyBmaSAqIGZfc3RlcCkgaWYgZmFjdG9ycyBpcyBOb25lIGVsc2UgZmxvYXQoZmFjdG9yc1tmaV0pCiAgICAgICAgTSA9IG5wLm1lZGlhbihmICogU0EgLSBTUiwgYXhpcz0wKQogICAgICAgIHByb2YgPSBbTVsocnIgPj0gcSkgJiAocnIgPCBxICsgMSldLm1lYW4oKSBmb3IgcSBpbiByYW5nZSg2KV0KICAgICAgICBjYWNoZVtrZXldID0gKGZsb2F0KG5wLmFicyhwcm9mKS5tYXgoKSksIGZsb2F0KGYpLCBmbG9hdChibHVyKSkKICAgICAgICByZXR1cm4gY2FjaGVba2V5XQoKICAgIGlmIGZhY3RvcnMgaXMgbm90IE5vbmU6CiAgICAgICAgZl9sbywgZl9oaSA9IDAsIGxlbihmYWN0b3JzKSAtIDEKICAgIGlmIGJsdXJzIGlzIG5vdCBOb25lOgogICAgICAgIGJfc3RlcCA9IDEuMAogICAgICAgIGJfbG8sIGJfaGkgPSAwLCBsZW4oYmx1cnMpIC0gMQoKICAgIGZvciBfcm91bmQgaW4gcmFuZ2UoNDApOgogICAgICAgIGdyaWQgPSBbc2NvcmUoZmksIGJpKQogICAgICAgICAgICAgICAgZm9yIGZpIGluIHJhbmdlKGZfbG8sIGZfaGkgKyAxKQogICAgICAgICAgICAgICAgZm9yIGJpIGluIHJhbmdlKGJfbG8sIGJfaGkgKyAxKV0KICAgICAgICBiZXN0X2xlZnQsIGJlc3RfZiwgYmVzdF9ibHVyID0gbWluKGdyaWQpCiAgICAgICAgYmlfYmVzdCA9IGludChyb3VuZChiZXN0X2JsdXIgLyBiX3N0ZXApKSBpZiBiX3N0ZXAgZWxzZSAwCiAgICAgICAgZmlfYmVzdCA9IChpbnQocm91bmQoKGJlc3RfZiAtIGNlbnRyZV9mYWN0b3IpIC8gZl9zdGVwKSkKICAgICAgICAgICAgICAgICAgIGlmIGZhY3RvcnMgaXMgTm9uZSBlbHNlIE5vbmUpCiAgICAgICAgZ3JldyA9IEZhbHNlCiAgICAgICAgaWYgZmFjdG9ycyBpcyBOb25lOgogICAgICAgICAgICBpZiBmaV9iZXN0IDw9IGZfbG8gYW5kIGNlbnRyZV9mYWN0b3IgKyAoZl9sbyAtIDEpICogZl9zdGVwID4gMDoKICAgICAgICAgICAgICAgIGZfbG8gLT0gMTsgZ3JldyA9IFRydWUKICAgICAgICAgICAgaWYgZmlfYmVzdCA+PSBmX2hpOgogICAgICAgICAgICAgICAgZl9oaSArPSAxOyBncmV3ID0gVHJ1ZQogICAgICAgIGlmIGJpX2Jlc3QgPD0gYl9sbzoKICAgICAgICAgICAgYl9sbyAtPSAxOyBncmV3ID0gVHJ1ZQogICAgICAgIGlmIGJpX2Jlc3QgPj0gYl9oaToKICAgICAgICAgICAgYl9oaSArPSAxOyBncmV3ID0gVHJ1ZQogICAgICAgIGlmIG5vdCBncmV3OgogICAgICAgICAgICBicmVhawoKICAgIHNjb3JlZCA9IHNvcnRlZChjYWNoZS52YWx1ZXMoKSkKICAgIGJlc3RfbGVmdCwgYmVzdF9mLCBiZXN0X2JsdXIgPSBzY29yZWRbMF0KICAgIGRpYWdbImJlc3QiXSA9IChiZXN0X2YsIGJlc3RfYmx1ciwgYmVzdF9sZWZ0KQogICAgZGlhZ1sic2VhcmNoZWQiXSA9IHsKICAgICAgICAiZmFjdG9yIjogKG1pbihzWzFdIGZvciBzIGluIHNjb3JlZCksIG1heChzWzFdIGZvciBzIGluIHNjb3JlZCkpLAogICAgICAgICJibHVyIjogKG1pbihzWzJdIGZvciBzIGluIHNjb3JlZCksIG1heChzWzJdIGZvciBzIGluIHNjb3JlZCkpLAogICAgICAgICJwb2ludHMiOiBsZW4oc2NvcmVkKX0KICAgIHByaW50KCIgICAgc2VhcmNoZWQgZmFjdG9yIHs6LjJmfSB0byB7Oi4yZn0gYW5kIGJsdXIgezorLjJmfSB0byB7OisuMmZ9IHB4ICIKICAgICAgICAgICJvdmVyIHt9IGNvbWJpbmF0aW9ucywgdW50aWwgdGhlIGJlc3QgaGFkIG5laWdoYm91cnMgb24gZXZlcnkgc2lkZSIKICAgICAgICAgIC5mb3JtYXQoZGlhZ1sic2VhcmNoZWQiXVsiZmFjdG9yIl1bMF0sIGRpYWdbInNlYXJjaGVkIl1bImZhY3RvciJdWzFdLAogICAgICAgICAgICAgICAgICBkaWFnWyJzZWFyY2hlZCJdWyJibHVyIl1bMF0sIGRpYWdbInNlYXJjaGVkIl1bImJsdXIiXVsxXSwKICAgICAgICAgICAgICAgICAgbGVuKHNjb3JlZCkpKQoKICAgICMgT25lIGNhbmRpZGF0ZSBlaXRoZXIgc2lkZSBpbiBmYWN0b3IsIGVhY2ggd2l0aCBpdHMgb3duIGJlc3QgYmx1ciwgc28gdGhlCiAgICAjIHRocmVlIGRpZmZlciBieSBzb21ldGhpbmcgdGhlIGV5ZSBjYW4gc2VlLgogICAgZGVmIGJlc3RfYXQoZik6CiAgICAgICAgaGVyZSA9IFtzIGZvciBzIGluIHNjb3JlZCBpZiBhYnMoc1sxXSAtIGYpIDwgMWUtNl0KICAgICAgICByZXR1cm4gbWluKGhlcmUpIGlmIGhlcmUgZWxzZSBOb25lCgogICAgb3RoZXJzID0gc29ydGVkKHtmIGZvciBfLCBmLCBfIGluIHNjb3JlZH0pCiAgICBiZWxvdyA9IFtmIGZvciBmIGluIG90aGVycyBpZiBmIDwgYmVzdF9mXQogICAgYWJvdmUgPSBbZiBmb3IgZiBpbiBvdGhlcnMgaWYgZiA+IGJlc3RfZl0KICAgIHBpY2tzID0gW2Jlc3RfYXQoYmVsb3dbLTFdKSBpZiBiZWxvdyBlbHNlIE5vbmUsCiAgICAgICAgICAgICAoYmVzdF9sZWZ0LCBiZXN0X2YsIGJlc3RfYmx1ciksCiAgICAgICAgICAgICBiZXN0X2F0KGFib3ZlWzBdKSBpZiBhYm92ZSBlbHNlIE5vbmVdCiAgICBwaWNrcyA9IFtwIGZvciBwIGluIHBpY2tzIGlmIHAgaXMgbm90IE5vbmVdCiAgICBpZiBsZW4ocGlja3MpIDwgMzogICAgICAgICAgICAgICAgICAgICAgICMgdGhlIGJlc3Qgc2F0IGF0IGFuIGVuZCBvZiB0aGUgcmFuZ2UKICAgICAgICBleHRyYV9wb29sID0gW3MgZm9yIHMgaW4gc2NvcmVkIGlmIHNbMV0gbm90IGluIHtwWzFdIGZvciBwIGluIHBpY2tzfV0KICAgICAgICBwaWNrcyArPSBzb3J0ZWQoZXh0cmFfcG9vbClbOjMgLSBsZW4ocGlja3MpXQogICAgcGlja3Muc29ydChrZXk9bGFtYmRhIHA6IHBbMV0pCgogICAgcmV0dXJuIF93cml0ZV9jaG9pY2VzKGhhLCByZWQsIHBpY2tzLCBwcmVmaXgsIGRpYWcsIGFzaykKCmRlZiBzaG93X3BpY3R1cmVzKHBhdGhzLCB0aXRsZT1Ob25lKToKICAgICIiIlB1dCBwaWN0dXJlcyBhbHJlYWR5IHdyaXR0ZW4gdG8gZGlzayBvbiB0aGUgc2NyZWVuLCB3aXRob3V0IGJsb2NraW5nLgoKICAgIFdIWSBJVCBTSE9XUyBUSEUgU0FWRUQgRklMRSBBTkQgTk9UIFRIRSBGSUdVUkUKICAgIEJlY2F1c2UgdGhlIGZpZ3VyZSB3YXMgZHJhd24gdW5kZXIgQWdnLCB0aGUgYmFja2VuZCB0aGF0IHdyaXRlcyB0byBhIGZpbGUgYW5kCiAgICBoYXMgbm8gd2luZG93IHRvIGF0dGFjaCB0bywgYW5kIG1vdmluZyBhIGxpdmUgZmlndXJlIGJldHdlZW4gYmFja2VuZHMgaXMgbm90CiAgICBzb21ldGhpbmcgbWF0cGxvdGxpYiBzdXBwb3J0cy4gUmVhZGluZyBiYWNrIHRoZSBQTkcgdGhhdCB3YXMganVzdCB3cml0dGVuCiAgICBzaWRlc3RlcHMgdGhhdCBlbnRpcmVseSwgYW5kIGhhcyB0aGUgYWR2YW50YWdlIHRoYXQgd2hhdCBhcHBlYXJzIG9uIHRoZSBzY3JlZW4KICAgIGlzIGV4YWN0bHkgdGhlIGZpbGUgLSBpZiB0aGV5IGV2ZXIgZGlzYWdyZWVkIGl0IHdvdWxkIGJlIHRoZSBmaWxlIHRoYXQgaXMgd3JvbmcsCiAgICBhbmQgdGhlIGZpbGUgaXMgd2hhdCBnZXRzIGtlcHQuCgogICAgQW4gaW50ZXJhY3RpdmUgYmFja2VuZCBpcyBib3Jyb3dlZCBhbmQgZ2l2ZW4gYmFjay4gT24gYSBtYWNoaW5lIHdpdGggbm8gZGlzcGxheQogICAgbm90aGluZyBpcyBzaG93biBhbmQgbm90aGluZyBmYWlsczsgdGhlIGZpbGVzIGFyZSBzdGlsbCB0aGVyZS4KICAgICIiIgogICAgaW1wb3J0IG1hdHBsb3RsaWIKICAgIGltcG9ydCBtYXRwbG90bGliLnB5cGxvdCBhcyBwbHQKCiAgICBwYXRocyA9IFtwIGZvciBwIGluIHBhdGhzIGlmIHAgYW5kIG9zLnBhdGguZXhpc3RzKHApXQogICAgaWYgbm90IHBhdGhzOgogICAgICAgIHJldHVybiBGYWxzZQoKICAgIHN0YXJ0ZWQgPSBtYXRwbG90bGliLmdldF9iYWNrZW5kKCkKICAgIGlmIHN0YXJ0ZWQubG93ZXIoKSA9PSAiYWdnIjoKICAgICAgICBmb3IgY2FuZGlkYXRlIGluICgiVGtBZ2ciLCAiUXRBZ2ciLCAiUXQ1QWdnIiwgIk1hY09TWCIpOgogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBwbHQuc3dpdGNoX2JhY2tlbmQoY2FuZGlkYXRlKQogICAgICAgICAgICAgICAgYnJlYWsKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICBpZiBtYXRwbG90bGliLmdldF9iYWNrZW5kKCkubG93ZXIoKSA9PSAiYWdnIjoKICAgICAgICByZXR1cm4gRmFsc2UKCiAgICB0cnk6CiAgICAgICAgaW1wb3J0IG1hdHBsb3RsaWIuaW1hZ2UgYXMgbXBpbWcKICAgICAgICBmb3IgcGF0aCBpbiBwYXRoczoKICAgICAgICAgICAgaW1hZ2UgPSBtcGltZy5pbXJlYWQocGF0aCkKICAgICAgICAgICAgaGVpZ2h0LCB3aWR0aCA9IGltYWdlLnNoYXBlWzBdLCBpbWFnZS5zaGFwZVsxXQogICAgICAgICAgICBzaWRlID0gOS41CiAgICAgICAgICAgIGZpZyA9IHBsdC5maWd1cmUoZmlnc2l6ZT0oc2lkZSwgc2lkZSAqIGhlaWdodCAvIGZsb2F0KHdpZHRoKSkpCiAgICAgICAgICAgIGF4ID0gZmlnLmFkZF9heGVzKFswLCAwLCAxLCAxXSkKICAgICAgICAgICAgYXguaW1zaG93KGltYWdlKQogICAgICAgICAgICBheC5zZXRfYXhpc19vZmYoKQogICAgICAgICAgICBmaWcuY2FudmFzLm1hbmFnZXIuc2V0X3dpbmRvd190aXRsZSgKICAgICAgICAgICAgICAgIHRpdGxlIG9yIG9zLnBhdGguYmFzZW5hbWUocGF0aCkpCiAgICAgICAgcGx0LnNob3coYmxvY2s9RmFsc2UpCiAgICAgICAgcGx0LnBhdXNlKDAuNSkKICAgICAgICByZXR1cm4gVHJ1ZQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICByZXR1cm4gRmFsc2UKICAgIGZpbmFsbHk6CiAgICAgICAgdHJ5OgogICAgICAgICAgICBwbHQuc3dpdGNoX2JhY2tlbmQoc3RhcnRlZCkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBwYXNzCgpkZWYgX3Nob3dfY2hvaWNlc193aW5kb3coZnJhbWVzLCBueiwgc3RlcCk6CiAgICAiIiJQdXQgdGhlIHRocmVlIGNhbmRpZGF0ZXMgc2lkZSBieSBzaWRlIGluIG9uZSB3aW5kb3cgYW5kIGxlYXZlIGl0IG9wZW4uCgogICAgV0hZIEEgV0lORE9XIEFORCBOT1QgVEhSRUUgRklMRVMKICAgIEJlY2F1c2UgdGhlIGp1ZGdlbWVudCBpcyBhIGNvbXBhcmlzb24uIFRocmVlIGZpbGVzIG9wZW5lZCBvbmUgYWZ0ZXIgYW5vdGhlciBpbgogICAgYSBwaWN0dXJlIHZpZXdlciBhcmUgdGhyZWUgc2VwYXJhdGUgaW1wcmVzc2lvbnMgYSBmZXcgc2Vjb25kcyBhcGFydDsgc2lkZSBieQogICAgc2lkZSB0aGV5IGFyZSBvbmUgbG9vaywgYW5kIHRoZSBkaWZmZXJlbmNlIGJldHdlZW4gdGhlbSAtIHdoaWNoIGlzIHNtYWxsLCBhIGZldwogICAgcGVyY2VudCBpbiB0aGUgZmFjdG9yIC0gaXMgb25seSB2aXNpYmxlIHRoYXQgd2F5LgoKICAgIFRoZSBiYWNrZW5kIGlzIHRoZSBkaWZmaWN1bHR5LiBCb3RoIHRvb2xzIHNldCBBZ2cgYXQgaW1wb3J0LCB3aGljaCBkcmF3cyB0byBhCiAgICBmaWxlIGFuZCBuZXZlciB0byBhIHNjcmVlbiwgYW5kIHRoYXQgaXMgdGhlIHJpZ2h0IGRlZmF1bHQgZm9yIGEgcnVuIHRoYXQKICAgIHByb2R1Y2VzIGZpZ3VyZXMgdW5hdHRlbmRlZC4gU28gYW4gaW50ZXJhY3RpdmUgYmFja2VuZCBpcyBib3Jyb3dlZCBmb3IgdGhpcyBvbmUKICAgIHdpbmRvdyBhbmQgZ2l2ZW4gYmFjayBhZnRlcndhcmRzLCBhbmQgaWYgbm9uZSBvZiB0aGVtIGNhbiBiZSBoYWQgLSBhIG1hY2hpbmUKICAgIHdpdGggbm8gZGlzcGxheSwgYSBzZXNzaW9uIG92ZXIgc3NoIC0gbm90aGluZyBpcyBzaG93biBhbmQgdGhlIGZpbGVzIHdyaXR0ZW4KICAgIG5leHQgdG8gdGhlIG91dHB1dCBhcmUgdGhlIGZhbGxiYWNrLgoKICAgIFRoZSB3aW5kb3cgaXMgb3BlbmVkIHdpdGhvdXQgYmxvY2tpbmcsIHNvIHRoZSBxdWVzdGlvbiBjYW4gYmUgYW5zd2VyZWQgaW4gdGhlCiAgICB0ZXJtaW5hbCB3aGlsZSBpdCBpcyBzdGlsbCBvbiBzY3JlZW4uCgogICAgUmV0dXJucyBUcnVlIGlmIGEgd2luZG93IGFjdHVhbGx5IGFwcGVhcmVkLgogICAgIiIiCiAgICBpbXBvcnQgbWF0cGxvdGxpYgogICAgaW1wb3J0IG1hdHBsb3RsaWIucHlwbG90IGFzIHBsdAoKICAgIGRlZiBibG9ja19tZWFuKGEsIG4pOgogICAgICAgIGswID0gKGEuc2hhcGVbMF0gLy8gbikgKiBuCiAgICAgICAgazEgPSAoYS5zaGFwZVsxXSAvLyBuKSAqIG4KICAgICAgICByZXR1cm4gYVs6azAsIDprMV0ucmVzaGFwZShrMCAvLyBuLCBuLCBrMSAvLyBuLCBuKS5tZWFuKGF4aXM9KDEsIDMpKQoKICAgIHN0YXJ0ZWQgPSBtYXRwbG90bGliLmdldF9iYWNrZW5kKCkKICAgIGlmIHN0YXJ0ZWQubG93ZXIoKSA9PSAiYWdnIjoKICAgICAgICBmb3IgY2FuZGlkYXRlIGluICgiVGtBZ2ciLCAiUXRBZ2ciLCAiUXQ1QWdnIiwgIk1hY09TWCIpOgogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBwbHQuc3dpdGNoX2JhY2tlbmQoY2FuZGlkYXRlKQogICAgICAgICAgICAgICAgYnJlYWsKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICBpZiBtYXRwbG90bGliLmdldF9iYWNrZW5kKCkubG93ZXIoKSA9PSAiYWdnIjoKICAgICAgICByZXR1cm4gRmFsc2UgICAgICAgICAgICAgICAgICAgICAjIG5vdGhpbmcgb24gdGhpcyBtYWNoaW5lIGNhbiBzaG93IGEgd2luZG93CgogICAgdHJ5OgogICAgICAgIGZpZywgYXhlcyA9IHBsdC5zdWJwbG90cygxLCBsZW4oZnJhbWVzKSwgZmlnc2l6ZT0oNS4yICogbGVuKGZyYW1lcyksIDUuOCkpCiAgICAgICAgaWYgbGVuKGZyYW1lcykgPT0gMToKICAgICAgICAgICAgYXhlcyA9IFtheGVzXQogICAgICAgIGZvciBheCwgKGksIChmLCBibHVyLCBuZXQpKSBpbiB6aXAoYXhlcywgZW51bWVyYXRlKGZyYW1lcywgMSkpOgogICAgICAgICAgICBzbWFsbCA9IGJsb2NrX21lYW4obmV0LCBzdGVwKSBpZiBzdGVwID4gMSBlbHNlIG5ldAogICAgICAgICAgICBheC5pbXNob3coc21hbGwsIG9yaWdpbj0ibG93ZXIiLCBjbWFwPSJncmF5X3IiLAogICAgICAgICAgICAgICAgICAgICAgdm1pbj0tMiAqIG56LCB2bWF4PTggKiBueikKICAgICAgICAgICAgYXguc2V0X3h0aWNrcyhbXSk7IGF4LnNldF95dGlja3MoW10pCiAgICAgICAgICAgIGF4LnNldF90aXRsZSgiY2hvaWNlIHt9XG5mYWN0b3IgezouMmZ9ICAge30iLmZvcm1hdCgKICAgICAgICAgICAgICAgIGksIGYsIF9ibHVyX3dvcmRzKGJsdXIpKSwgZm9udHNpemU9MTIpCiAgICAgICAgZmlnLnN1cHRpdGxlKCJXaGljaCBzdWJ0cmFjdGlvbiBpcyBiZXN0PyBMb29rIGF0IHRoZSBzdGFycyBhbmQgYXQgdGhlICIKICAgICAgICAgICAgICAgICAgICAgImVtcHR5IHNreSwgdGhlbiBhbnN3ZXIgaW4gdGhlIHRlcm1pbmFsLiIsIGZvbnRzaXplPTEzKQogICAgICAgIGZpZy50aWdodF9sYXlvdXQoKQogICAgICAgIHBsdC5zaG93KGJsb2NrPUZhbHNlKQogICAgICAgIHBsdC5wYXVzZSgwLjUpCiAgICAgICAgcmV0dXJuIFRydWUKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcmV0dXJuIEZhbHNlCiAgICBmaW5hbGx5OgogICAgICAgIHRyeToKICAgICAgICAgICAgcGx0LnN3aXRjaF9iYWNrZW5kKHN0YXJ0ZWQpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgcGFzcwoKZGVmIF9ibHVyX3dvcmRzKGJsdXIpOgogICAgIiIiVGhlIGJsdXIgYXMgYSBwaHJhc2UsIHNvIHRoZSBwaWN0dXJlIHNheXMgd2hpY2ggZnJhbWUgd2FzIGJsdXJyZWQuIiIiCiAgICBpZiBibHVyID4gMDoKICAgICAgICByZXR1cm4gIkgtYWxwaGEgYmx1cnJlZCBieSB7Oi4yZn0gcHgiLmZvcm1hdChibHVyKQogICAgaWYgYmx1ciA8IDA6CiAgICAgICAgcmV0dXJuICJSIGJsdXJyZWQgYnkgezouMmZ9IHB4Ii5mb3JtYXQoLWJsdXIpCiAgICByZXR1cm4gIm5laXRoZXIgZnJhbWUgYmx1cnJlZCIKCmRlZiBfd3JpdGVfY2hvaWNlcyhoYSwgcmVkLCBwaWNrcywgcHJlZml4LCBkaWFnLCBhc2spOgogICAgIiIiV3JpdGUgb25lIHBpY3R1cmUgcGVyIGNhbmRpZGF0ZSwgYXNrLCBhbmQgYXBwbHkgdGhlIGFuc3dlci4iIiIKICAgIGltcG9ydCBtYXRwbG90bGliCiAgICBtYXRwbG90bGliLnVzZSgiQWdnIikKICAgIGltcG9ydCBtYXRwbG90bGliLnB5cGxvdCBhcyBwbHQKICAgIGZyb20gc2NpcHkubmRpbWFnZSBpbXBvcnQgZ2F1c3NpYW5fZmlsdGVyCiAgICBmcm9tIGFzdHJvcHkuc3RhdHMgaW1wb3J0IHNpZ21hX2NsaXBwZWRfc3RhdHMKCiAgICBkZWYgYmxvY2tfbWVhbihhLCBuKToKICAgICAgICBrMCA9IChhLnNoYXBlWzBdIC8vIG4pICogbgogICAgICAgIGsxID0gKGEuc2hhcGVbMV0gLy8gbikgKiBuCiAgICAgICAgcmV0dXJuIGFbOmswLCA6azFdLnJlc2hhcGUoazAgLy8gbiwgbiwgazEgLy8gbiwgbikubWVhbihheGlzPSgxLCAzKSkKCiAgICBzdGVwID0gbWF4KDEsIGludChyb3VuZChtYXgoaGEuc2hhcGUpIC8gMTQwMC4wKSkpCiAgICBmcmFtZXMsIHBhdGhzID0gW10sIFtdCiAgICBmb3IgaSwgKGxlZnQsIGYsIGJsdXIpIGluIGVudW1lcmF0ZShwaWNrcywgMSk6CiAgICAgICAgQSA9IGdhdXNzaWFuX2ZpbHRlcihoYSwgYmx1cikgaWYgYmx1ciA+IDAgZWxzZSBoYQogICAgICAgIFIgPSBnYXVzc2lhbl9maWx0ZXIocmVkLCAtYmx1cikgaWYgYmx1ciA8IDAgZWxzZSByZWQKICAgICAgICBmcmFtZXMuYXBwZW5kKChmLCBibHVyLCBmICogQSAtIFIpKQoKICAgICMgT25lIHN0cmV0Y2ggZm9yIGFsbCB0aHJlZSwgb3IgdGhlIGV5ZSBjb21wYXJlcyB0aHJlZSBkaWZmZXJlbnQgc3RyZXRjaGVzLgogICAgbWlkZGxlID0gZnJhbWVzW2xlbihmcmFtZXMpIC8vIDJdWzJdCiAgICBueiA9IGZsb2F0KHNpZ21hX2NsaXBwZWRfc3RhdHMobWlkZGxlLCBzaWdtYT0zLjApWzJdKQoKICAgIGZvciBpLCAoZiwgYmx1ciwgbmV0KSBpbiBlbnVtZXJhdGUoZnJhbWVzLCAxKToKICAgICAgICBzbWFsbCA9IGJsb2NrX21lYW4obmV0LCBzdGVwKSBpZiBzdGVwID4gMSBlbHNlIG5ldAogICAgICAgIGZpZywgYXggPSBwbHQuc3VicGxvdHMoZmlnc2l6ZT0oOSwgOSkpCiAgICAgICAgIyBEYXJrIG9uIGxpZ2h0LCB0aGUgd2F5IHRoaXMgaXMgbG9va2VkIGF0IGJ5IGV5ZS4KICAgICAgICBheC5pbXNob3coc21hbGwsIG9yaWdpbj0ibG93ZXIiLCBjbWFwPSJncmF5X3IiLCB2bWluPS0yICogbnosIHZtYXg9OCAqIG56KQogICAgICAgIGF4LnNldF94dGlja3MoW10pOyBheC5zZXRfeXRpY2tzKFtdKQogICAgICAgICMgVGhlIG51bWJlcnMgYmVsb25nIG9uIHRoZSBwaWN0dXJlLiBBIGNob2ljZSBtYWRlIGZyb20gdGhyZWUgdW5sYWJlbGxlZAogICAgICAgICMgZnJhbWVzIGNhbm5vdCBiZSB3cml0dGVuIGRvd24sIHJlcGVhdGVkLCBvciBhcmd1ZWQgd2l0aCBhZnRlcndhcmRzLgogICAgICAgIGF4LnNldF90aXRsZSgiY2hvaWNlIHt9XG5mYWN0b3IgezouMmZ9ICAge30iLmZvcm1hdChpLCBmLCBfYmx1cl93b3JkcyhibHVyKSksCiAgICAgICAgICAgICAgICAgICAgIGZvbnRzaXplPTE0KQogICAgICAgIGZpZy50aWdodF9sYXlvdXQoKQogICAgICAgIHBhdGggPSAie31fY2hvaWNlX3t9LnBuZyIuZm9ybWF0KHByZWZpeCwgaSkKICAgICAgICBmaWcuc2F2ZWZpZyhwYXRoLCBkcGk9MTAwLCBiYm94X2luY2hlcz0idGlnaHQiKQogICAgICAgIHBsdC5jbG9zZShmaWcpCiAgICAgICAgcGF0aHMuYXBwZW5kKHBhdGgpCiAgICBkaWFnWyJwYXRocyJdID0gcGF0aHMKICAgIGRpYWdbInBpY2tzIl0gPSBbKGYsIGJsdXIpIGZvciBmLCBibHVyLCBfIGluIGZyYW1lc10KCiAgICAjIFdIWSBUSEVSRSBJUyBOTyBXSU5ET1cKICAgICMgVGhlcmUgd2FzIG9uZSwgYW5kIGl0IGRpZCBub3Qgd29yay4gQSBmaWd1cmUgb3BlbmVkIHdpdGhvdXQgYmxvY2tpbmcsIHNvCiAgICAjIHRoYXQgdGhlIHF1ZXN0aW9uIGNvdWxkIHN0aWxsIGJlIGFuc3dlcmVkIGluIHRoZSB0ZXJtaW5hbCwgaXMgYSB3aW5kb3cgd2l0aAogICAgIyBub3RoaW5nIGRyaXZpbmcgaXRzIGV2ZW50IGxvb3A6IGl0IGFwcGVhcnMgYXMgYW4gZW1wdHkgZ3JleSByZWN0YW5nbGUgYW5kCiAgICAjIG9ubHkgZHJhd3MgaXRzZWxmIG9uY2UgaXQgaXMgY2xvc2VkLCB3aGljaCBpcyBhZnRlciB0aGUgYW5zd2VyIGhhcyBiZWVuCiAgICAjIGdpdmVuLiBBIHBpY3R1cmUgdGhhdCBjYW4gb25seSBiZSBzZWVuIGFmdGVyIGl0IGlzIG5vIGxvbmdlciBuZWVkZWQgaXMgd29yc2UKICAgICMgdGhhbiBubyBwaWN0dXJlLCBiZWNhdXNlIGl0IGFsc28gZnJlZXplcyB0aGUgc2NyZWVuLgogICAgIwogICAgIyBTbyB0aGUgdGhyZWUgYXJlIHdyaXR0ZW4gdG8gZGlzayBhbmQgdGhlaXIgbmFtZXMgYXJlIHByaW50ZWQuIE9wZW4gdGhlbSBpbgogICAgIyB3aGF0ZXZlciB5b3Ugbm9ybWFsbHkgbG9vayBhdCBpbWFnZXMgd2l0aCwgb3IgZG8gbm90IC0gdGhlIG51bWJlcnMgYXJlIG9uCiAgICAjIHRoZSBzY3JlZW4gZWl0aGVyIHdheSwgYW5kIHRoZSBtaWRkbGUgb25lIGlzIHRoZSBkZWZhdWx0LgogICAgc2hvd24gPSBGYWxzZQoKICAgIHByaW50KCIiKQogICAgcHJpbnQoIiAgVGhyZWUgc3VidHJhY3Rpb25zIG9mIHRoZSBzYW1lIGZyYW1lcywgd3JpdHRlbiBuZXh0IHRvIHRoZSBvdGhlciIpCiAgICBwcmludCgiICBvdXRwdXQuIExvb2sgYXQgdGhlIHN0YXJzIGFuZCBhdCB0aGUgZW1wdHkgc2t5IGluIHRoZW0sIHRoZW4gc2F5IikKICAgIHByaW50KCIgIHdoaWNoIG9uZSBpcyBiZXN0LiBFbnRlciB0YWtlcyB0aGUgbWlkZGxlIG9uZS4iKQogICAgcHJpbnQoIiAgICBpbiAiICsgKG9zLnBhdGguZGlybmFtZShwYXRoc1swXSkgb3IgIi4iKSkKICAgIGZvciBpLCAocCwgKGYsIGJsdXIsIF8pKSBpbiBlbnVtZXJhdGUoemlwKHBhdGhzLCBmcmFtZXMpLCAxKToKICAgICAgICBwcmludCgiICAgIHt9LiB7OjwyNH0gZmFjdG9yIHs6LjJmfSwge30iLmZvcm1hdCgKICAgICAgICAgICAgaSwgb3MucGF0aC5iYXNlbmFtZShwKSwgZiwgX2JsdXJfd29yZHMoYmx1cikpKQoKICAgIGFuc3dlciA9IDIgaWYgbGVuKHBhdGhzKSA+PSAyIGVsc2UgMQogICAgaWYgYXNrIGlzIG5vdCBOb25lOgogICAgICAgIHdoaWxlIFRydWU6CiAgICAgICAgICAgIGdvdCA9IGFzaygiV2hpY2ggb25lIiwgc3RyKGFuc3dlciksIGludCkKICAgICAgICAgICAgaWYgMSA8PSBnb3QgPD0gbGVuKHBhdGhzKToKICAgICAgICAgICAgICAgIGFuc3dlciA9IGdvdAogICAgICAgICAgICAgICAgYnJlYWsKICAgICAgICAgICAgcHJpbnQoIiAgICBhIG51bWJlciBmcm9tIHRoZSBsaXN0LCBwbGVhc2UiKQogICAgZWxzZToKICAgICAgICBwcmludCgiICAgIG5vdGhpbmcgdG8gYXNrIHdpdGgsIHNvIHt9IGlzIHVzZWQiLmZvcm1hdChhbnN3ZXIpKQoKICAgIGlmIHNob3duOgogICAgICAgIHRyeToKICAgICAgICAgICAgaW1wb3J0IG1hdHBsb3RsaWIucHlwbG90IGFzIHBsdAogICAgICAgICAgICBwbHQuY2xvc2UoImFsbCIpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgcGFzcwoKICAgIGYsIGJsdXIsIF8gPSBmcmFtZXNbYW5zd2VyIC0gMV0KICAgIGlmIGJsdXIgPiAwOgogICAgICAgIGhhID0gZ2F1c3NpYW5fZmlsdGVyKGhhLCBibHVyKQogICAgZWxpZiBibHVyIDwgMDoKICAgICAgICByZWQgPSBnYXVzc2lhbl9maWx0ZXIocmVkLCAtYmx1cikKICAgIGRpYWdbImNob3NlbiJdID0gYW5zd2VyCiAgICBkaWFnWyJmYWN0b3IiXSA9IGYKICAgIGRpYWdbImJsdXIiXSA9IGJsdXIKICAgIHJldHVybiBoYSwgcmVkLCBmLCBkaWFnCgpkZWYgZGVzY3JpYmVfY2hvaWNlKGRpYWcpOgogICAgIiIiV2hhdCB3YXMgY2hvc2VuLCBpbiBhIGxpbmUuIiIiCiAgICBpZiBkaWFnLmdldCgid2h5Iik6CiAgICAgICAgcmV0dXJuICIgIG5vIGNob2ljZSBvZmZlcmVkICh7fSkiLmZvcm1hdChkaWFnWyJ3aHkiXSkKICAgIGYsIGJsdXIgPSBkaWFnWyJmYWN0b3IiXSwgZGlhZ1siYmx1ciJdCiAgICB3aGF0ID0gKCJILWFscGhhIGJsdXJyZWQgYnkgezouMmZ9IHB4Ii5mb3JtYXQoYmx1cikgaWYgYmx1ciA+IDAgZWxzZQogICAgICAgICAgICAiUiBibHVycmVkIGJ5IHs6LjJmfSBweCIuZm9ybWF0KC1ibHVyKSBpZiBibHVyIDwgMCBlbHNlCiAgICAgICAgICAgICJuZWl0aGVyIGZyYW1lIGJsdXJyZWQiKQogICAgcmV0dXJuICgiICBjaG9pY2Uge30gdGFrZW46IGZhY3RvciB7Oi4yZn0sIHt9XG4iCiAgICAgICAgICAgICIgICAganVkZ2VkIG9uIHt9IHN0YXJzIG91dHNpZGUgdGhlIGdhbGF4eSIKICAgICAgICAgICAgLmZvcm1hdChkaWFnWyJjaG9zZW4iXSwgZiwgd2hhdCwgZGlhZ1sibl9zdGFycyJdKSkKCgojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCiMgQmx1ZV9DbHVzdGVyc19Gcm9tX0ltYWdlcy5weQojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCgpmcm9tIHNjaXB5LnNwYXRpYWwgaW1wb3J0IGNLRFRyZWUKZnJvbSBwaG90dXRpbHMuYXBlcnR1cmUgaW1wb3J0IEFwZXJ0dXJlU3RhdHMKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLSBQYXJhbWV0ZXJzIC0tLS0tLS0tLS0tLS0tLS0tLS0KIyBNZWFzdXJpbmcgdGhlIHNvdXJjZXMKQVBFUlRVUkVfU0NBTEUgPSAyICAgICAgICAgICAgICAgIyBhcGVydHVyZSByYWRpdXMgPSB0aGlzIHggdGhlIHR5cGljYWwgc3RhciBGV0hNIG9mIHRoZQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAjIGZyYW1lcyAtIE9ORSByYWRpdXMgZm9yIGV2ZXJ5IHNvdXJjZSwgaW4gYm90aCBiYW5kcwpBTk5VTFVTX0lOTkVSX1NDQUxFID0gMi41ICAgICAgICAjIHRoZSBza3kgcmluZyBhcm91bmQgZWFjaCBzb3VyY2UsIGluIHVuaXRzIG9mIHRoZQpBTk5VTFVTX09VVEVSX1NDQUxFID0gMyAgICAgICAgICAjIGFwZXJ0dXJlIHJhZGl1cwpGV0hNX1dJTkRPV19TSVpFID0gMTAgICAgICAgICAgICAjIGhhbGYtc2l6ZSBvZiB0aGUgYm94IGEgc291cmNlJ3Mgd2lkdGggaXMgbWVhc3VyZWQgaW4KREFPRklORF9GV0hNID0gTm9uZSAgICAgICAgICAgICAgIyBOb25lID0gbWVhc3VyZSBpdCBmcm9tIHRoZSBkYXRhIChyZWNvbW1lbmRlZCkKU0VFSU5HX0ZBTExCQUNLID0gNC4wICAgICAgICAgICAgIyBVc2VkIG9ubHkgaWYgdGhlIG1lYXN1cmVtZW50IGNhbm5vdCBiZSBtYWRlClNFRUlOR19NSU4sIFNFRUlOR19NQVggPSAxLjUsIDE1LjAgICAjIFNhbml0eSByYW5nZSBmb3IgdGhlIG1lYXN1cmVkIHZhbHVlLCBwaXhlbHMKU0lHTUFfQ0xJUCA9IDMuMCAgICAgICAgICAgICAgICAgIyBTaWdtYSBjbGlwcGluZyBsZXZlbCBmb3IgYmFja2dyb3VuZCBzdGF0aXN0aWNzCkRFVEVDVElPTl9USFJFU0hPTERfU0lHTUEgPSAzLjAgICMgRGV0ZWN0aW9uIHRocmVzaG9sZCBpbiB1bml0cyBvZiBiYWNrZ3JvdW5kIHNpZ21hClBFQUtfTUlOX1NURCA9IDMuMCAgICAgICAgICAgICAgICMgTWluaW11bSBwZWFrIHNpZ25hbC10by1ub2lzZSByYXRpbyBmb3IgYSBkZXRlY3Rpb24KQkFDS0dST1VORF9CT1hfU0laRSA9IDQ4ICAgICAgICAgIyBHcmlkIGNlbGwgZm9yIHRoZSBsb2NhbCBiYWNrZ3JvdW5kLCBwaXhlbHMKQkFDS0dST1VORF9GSUxURVJfU0laRSA9IDMgICAgICAgIyBNZWRpYW4gZmlsdGVyIGFwcGxpZWQgYWNyb3NzIHRoZSBncmlkClNFRUlOR19NRUFTVVJFRCA9IHt9CkJHX1NVQl9GVU5DID0gbnAubmFubWVkaWFuICAgICAgICMgRnVuY3Rpb24gdXNlZCB0byBjb21wdXRlIGFuZCBzdWJ0cmFjdCBiYWNrZ3JvdW5kIGxldmVsCkFTVFJPQUxJR05fTUFYX1NDQUxFX0VSUk9SID0gMC4wNSAgIyA1JQpHQUlOX0VfUEVSX0FEVSA9IDEuMCAgICAgICAgICAgICAjIGVsZWN0cm9ucyBwZXIgQURVLCB1c2VkIG9ubHkgaWYgdGhlIGhlYWRlciBoYXMgbm9uZQpSRUFEX05PSVNFX0UgPSA1LjAgICAgICAgICAgICAgICAjIHJlYWQgbm9pc2UgaW4gZWxlY3Ryb25zLCB1c2VkIG9ubHkgaWYgdGhlIGhlYWRlciBoYXMgbm9uZQpFREdFX01BUkdJTl9QWCA9IDMwCgojIENhbGlicmF0aW9uCkNBTElCX0NBVEFMT0dVRSA9ICJBUEFTUyIgICAgICAgICMgZGVmYXVsdDsgIkdBSUEiIGZvciBHYWlhIERSMyBzeW50aGV0aWMgQiBhbmQgVgpDQUxJQl9NQVRDSF9SQURJVVMgPSAzLjAgICAgICAgICAjIGFyY3NlYzogYSBjYXRhbG9ndWUgc3RhciBhbmQgYSBzb3VyY2UgdGhpcyBjbG9zZSBhcmUgb25lClhZX01BVENIX1RPTEVSQU5DRSA9IDMuMCAgICAgICAgICMgcGl4ZWxzLCBvbiB0aGUgViBncmlkLCBhZnRlciB0aGUgcGxhdGUgc29sdXRpb24gaXMgY2hlY2tlZApDQUxJQl9NSU5fU1RBUlMgPSA1ICAgICAgICAgICAgICAjIGZld2VyIGNhbGlicmF0aW9uIHN0YXJzIHRoYW4gdGhpcyBhbmQgdGhlIHJ1biBzdG9wcwpDQUxJQl9NQVhfSU5TVF9FUlIgPSAwLjA1ICAgICAgICAjIG1hZzsgb25seSB3ZWxsLW1lYXN1cmVkIHN0YXJzIHNldCB0aGUgY2FsaWJyYXRpb24KQ0FMSUJfTUFYX0NBVF9FUlIgPSAwLjEwICAgICAgICAgIyBtYWc7IGFuZCBvbmx5IHdlbGwtbWVhc3VyZWQgY2F0YWxvZ3VlIGVudHJpZXMKQ0FMSUJfQ0xJUF9TSUdNQSA9IDMuMCAgICAgICAgICAgIyBzdGFycyBmdXJ0aGVyIHRoYW4gdGhpcyBmcm9tIHRoZSBmaXQgYXJlIGxlZnQgb3V0CkNBTElCX01JTl9PVVRTSURFID0gMTAgICAgICAgICAgICMgdXNlIG9ubHkgc3RhcnMgYmV5b25kIFIyNSB3aGVuIGF0IGxlYXN0IHRoaXMgbWFueSBhcmUKV0NTX0NIRUNLX01JTl9GUkFDVElPTiA9IDAuNSAgICAgIyBzaGFyZSBvZiBicmlnaHQgc3RhcnMgdGhhdCBtdXN0IGhhdmUgYSBHYWlhIHN0YXIgbmVhcmJ5CgojIFRoZSBibHVlIGtub3RzCkNNRF9DT0xPUl9NSU4gPSAtMC41ICAgICAgICAgICAgICMgYSBibHVlIGtub3QgaGFzIChCLVYpMCBiZXR3ZWVuIHRoZXNlIHR3byB2YWx1ZXMKQ01EX0NPTE9SX01BWCA9IDAuNQpLTk9UX01BWF9CVl9FUlIgPSAwLjIgICAgICAgICAgICAjIGFuZCBhIEItViBlcnJvciBubyBsYXJnZXIgdGhhbiB0aGlzLCBpbiBtYWcKQ01EX01BWF9CVl9FUlIgPSAwLjMgICAgICAgICAgICAgIyBzb3VyY2VzIG5vaXNpZXIgdGhhbiB0aGlzIGFyZSBsZWZ0IG91dCBvZiB0aGUgZGlhZ3JhbXMKRk9SRUdST1VORF9UT0wgPSAzLjAgICAgICAgICAgICAgIyBwaXhlbHM6IGEgc291cmNlIHRoaXMgY2xvc2UgdG8gYSBHYWlhIGZvcmVncm91bmQgc3RhciBpcyBpdApTdGVwX3NpemUgPSAxMCAgICAgICAgICAgICAgICAgICAjIFJlc29sdXRpb24gb2YgVGhlIEJsdWUgS25vdHMgRGVuc2l0eSBEaWFncmFtCgpSX1YgPSAzLjEgICAgICAgICAgICAgICAgICAgICAgICAjIEFfViAvIEUoQi1WKSBmb3IgZHVzdCBpbiBvdXIgb3duIGdhbGF4eQoKQVJHUyA9IE5vbmUgICAgICAgICAgICAgICAgICAgICAgIyB0aGUgY29tbWFuZC1saW5lIGFyZ3VtZW50cywgaWYgYW55CgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tIFF1ZXN0aW9ucyAtLS0tLS0tLS0tLS0tLS0tLS0tCgpkZWYgX2FjY2VwdF9kZWZhdWx0cygpOgogICAgcmV0dXJuIEFSR1MgaXMgbm90IE5vbmUgYW5kIGdldGF0dHIoQVJHUywgInllcyIsIEZhbHNlKQoKCmRlZiBfYXNrKHByb21wdCwgZGVmYXVsdCwgY2FzdD1zdHIsIGNob2ljZXM9Tm9uZSk6CiAgICAiIiJBc2sgb25lIHF1ZXN0aW9uLiBFbnRlciBhY2NlcHRzIHRoZSBkZWZhdWx0IHNob3duIGluIGJyYWNrZXRzLgoKICAgIE9ubHkgYW4gZW1wdHkgYW5zd2VyIG1lYW5zICJ0aGUgZGVmYXVsdCIuIEFuIGFuc3dlciB0aGF0IGNhbm5vdCBiZSByZWFkIGlzCiAgICByZWZ1c2VkIGFuZCB0aGUgcXVlc3Rpb24gYXNrZWQgYWdhaW4gLSBpdCB1c2VkIHRvIGJlIHJlcGxhY2VkIGJ5IHRoZQogICAgZGVmYXVsdCB3aXRob3V0IGEgd29yZCwgc28gYSBtaXN0eXBlZCBkaXN0YW5jZSBiZWNhbWUgMTAgTXBjLiBBIGRlZmF1bHQgb2YKICAgIE5vbmUgbWVhbnMgdGhlcmUgaXMgbm9uZTogYW4gYW5zd2VyIGlzIG5lZWRlZC4KICAgICIiIgogICAgaWYgX2FjY2VwdF9kZWZhdWx0cygpOgogICAgICAgIGlmIGRlZmF1bHQgaXMgTm9uZToKICAgICAgICAgICAgc3lzLmV4aXQoZiJbaW5wdXRdICd7cHJvbXB0fScgaGFzIG5vIGRlZmF1bHQsIHNvIC0teWVzIGNhbm5vdCBhbnN3ZXIgIgogICAgICAgICAgICAgICAgICAgICBmIml0IC0gZ2l2ZSBpdCBvbiB0aGUgY29tbWFuZCBsaW5lIikKICAgICAgICBwcmludChmIntwcm9tcHR9IFt7ZGVmYXVsdH1dOiAoYWNjZXB0ZWQsIC0teWVzKSIpCiAgICAgICAgcmV0dXJuIGRlZmF1bHQKICAgIHNob3duID0gIiIgaWYgZGVmYXVsdCBpcyBOb25lIGVsc2UgZiIgW3tkZWZhdWx0fV0iCiAgICB3aGlsZSBUcnVlOgogICAgICAgIHMgPSBpbnB1dChmIntwcm9tcHR9e3Nob3dufTogIikuc3RyaXAoKQogICAgICAgIGlmIHMgPT0gIiI6CiAgICAgICAgICAgIGlmIGRlZmF1bHQgaXMgTm9uZToKICAgICAgICAgICAgICAgIHByaW50KCIgICBhbiBhbnN3ZXIgaXMgbmVlZGVkIGhlcmUiKQogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgcmV0dXJuIGRlZmF1bHQKICAgICAgICB0cnk6CiAgICAgICAgICAgIHYgPSBjYXN0KHMpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgcHJpbnQoZiIgICBjb3VsZCBub3QgcmVhZCAne3N9JyAtIHRyeSBhZ2FpbiIpCiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgaWYgY2hvaWNlcyBhbmQgdiBub3QgaW4gY2hvaWNlczoKICAgICAgICAgICAgcHJpbnQoZiIgICBhbnN3ZXIgb25lIG9mOiB7JywgJy5qb2luKGNob2ljZXMpfSIpCiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgcmV0dXJuIHYKCgpkZWYgX25vcm1fcGF0aChwOiBzdHIpIC0+IHN0cjoKICAgICIiIlJlbW92ZSBleHRyYSBxdW90ZXMgYW5kIG5vcm1hbGl6ZSBmaWxlc3lzdGVtIHBhdGguIiIiCiAgICBwID0gcC5zdHJpcCgpLnN0cmlwKCciJykuc3RyaXAoIiciKQogICAgcmV0dXJuIG9zLnBhdGgubm9ybXBhdGgocCkgaWYgcCBlbHNlICIiCgoKZGVmIG9mZmVyZWQobGFiZWwsIHZhbHVlLCBzb3VyY2UsIHVuaXQ9IiIsIG92ZXJyaWRlPU5vbmUpOgogICAgIiIiQSBjYXRhbG9ndWUgdmFsdWUsIHNob3duIHdpdGggd2hlcmUgaXQgY2FtZSBmcm9tLCBhbmQgYWNjZXB0ZWQgd2l0aAogICAgRW50ZXIgb3IgcmVwbGFjZWQgYnkgdHlwaW5nIGFub3RoZXIuIEEgdmFsdWUgZ2l2ZW4gb24gdGhlIGNvbW1hbmQgbGluZQogICAgaXMgdXNlZCBhcyBpdCBpcy4iIiIKICAgIGlmIG92ZXJyaWRlIGlzIG5vdCBOb25lOgogICAgICAgIHByaW50KGYie2xhYmVsfToge292ZXJyaWRlfXt1bml0fSAoZnJvbSB0aGUgY29tbWFuZCBsaW5lKSIpCiAgICAgICAgcmV0dXJuIGZsb2F0KG92ZXJyaWRlKQogICAgaWYgdmFsdWUgaXMgTm9uZToKICAgICAgICBwcmludChmIltsb29rdXBdIG5vIHtsYWJlbC5sb3dlcigpfSBmb3VuZCBpbiB0aGUgY2F0YWxvZ3VlIC0gcGxlYXNlIGdpdmUgb25lIikKICAgICAgICByZXR1cm4gZmxvYXQoX2FzayhsYWJlbCwgTm9uZSwgZmxvYXQpKQogICAgcHJpbnQoZiJbbG9va3VwXSB7bGFiZWx9OiB7dmFsdWU6Z317dW5pdH0gKHtzb3VyY2V9KSIpCiAgICByZXR1cm4gZmxvYXQoX2FzayhmIntsYWJlbH0gLSBFbnRlciB0byBhY2NlcHQsIG9yIHR5cGUgYW5vdGhlciIsIHZhbHVlLCBmbG9hdCkpCgoKZGVmIHN0b3AobWVzc2FnZSk6CiAgICAiIiJTdG9wIHRoZSBydW4gd2l0aCBhIG1lc3NhZ2UgdGhhdCBzYXlzIHdoYXQgdG8gZG8uIiIiCiAgICBwcmludCgiIikKICAgIHByaW50KCIqIiAqIDc4KQogICAgcHJpbnQoIlNUT1BQRUQ6ICIgKyBtZXNzYWdlKQogICAgcHJpbnQoIioiICogNzgpCiAgICByYWlzZSBTeXN0ZW1FeGl0KDEpCgoKZGVmIGxvdWQobWVzc2FnZSk6CiAgICBwcmludCgiIikKICAgIHByaW50KCIhIiAqIDc4KQogICAgcHJpbnQobWVzc2FnZSkKICAgIHByaW50KCIhIiAqIDc4KQogICAgcHJpbnQoIiIpCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tIFJlYWRpbmcgZnJhbWVzIGFuZCB0aGUgZGV0ZWN0b3IgLS0tLS0tLS0tLS0tLS0tLS0tLQoKZGVmIHJlYWRfaW1hZ2UocGF0aCk6CiAgICAiIiIoZGF0YSBhcyBmbG9hdDMyLCBoZWFkZXIpIGZyb20gdGhlIGZpcnN0IEhEVSB0aGF0IGhvbGRzIGEgMi1EIGltYWdlLgoKICAgIFRoZSBmaXJzdCBIRFUgaXMgbm90IGFsd2F5cyBpdDogYW4gTENPIC5maXRzLmZ6IGZyYW1lIGtlZXBzIGl0cyBpbWFnZSBpbgogICAgdGhlIHNlY29uZCwgYW5kIHJlYWRpbmcgdGhlIGZpcnN0IHJldHVybmVkIG5vdGhpbmcuCiAgICAiIiIKICAgIHdpdGggZml0cy5vcGVuKHBhdGgpIGFzIGhkdWw6CiAgICAgICAgZm9yIGhkdSBpbiBoZHVsOgogICAgICAgICAgICBpZiBoZHUuZGF0YSBpcyBub3QgTm9uZSBhbmQgZ2V0YXR0cihoZHUuZGF0YSwgIm5kaW0iLCAwKSA9PSAyOgogICAgICAgICAgICAgICAgaGVhZGVyID0gaGR1LmhlYWRlci5jb3B5KCkKICAgICAgICAgICAgICAgIGZvciBrZXkgaW4gKCJYVEVOU0lPTiIsICJQQ09VTlQiLCAiR0NPVU5UIiwgIkJaRVJPIiwgIkJTQ0FMRSIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAiQkxBTksiLCAiWklNQUdFIiwgIlpCSVRQSVgiLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIlpOQVhJUyIsICJaTkFYSVMxIiwgIlpOQVhJUzIiLCAiWlRJTEUxIiwgIlpUSUxFMiIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAiWkNNUFRZUEUiLCAiWlFVQU5USVoiLCAiWkRJVEhFUjAiLCAiWlNJTVBMRSIsICJaRVhURU5EIik6CiAgICAgICAgICAgICAgICAgICAgaGVhZGVyLnJlbW92ZShrZXksIGlnbm9yZV9taXNzaW5nPVRydWUsIHJlbW92ZV9hbGw9VHJ1ZSkKICAgICAgICAgICAgICAgIHJldHVybiBucC5hc2FycmF5KGhkdS5kYXRhLCBkdHlwZT1ucC5mbG9hdDMyKSwgaGVhZGVyCiAgICByYWlzZSBSdW50aW1lRXJyb3IoZiJubyAyLUQgaW1hZ2UgaW4ge3BhdGh9IikKCgpkZWYgY29tYmluZWRfY291bnQoaGVhZGVyKToKICAgICIiIkhvdyBtYW55IGZyYW1lcyB3ZW50IGludG8gdGhpcyBmaWxlLCBhbmQgaG93IHRoZXkgd2VyZSBjb21iaW5lZC4iIiIKICAgIG4gPSBoZWFkZXIuZ2V0KCJOQ09NQklORSIpCiAgICBob3cgPSBzdHIoaGVhZGVyLmdldCgiQ09NQlRZUEUiLCAibWVkaWFuIikpLmxvd2VyKCkKICAgIHRyeToKICAgICAgICBpZiBuIGFuZCBpbnQobikgPiAxOgogICAgICAgICAgICByZXR1cm4gaW50KG4pLCBfY29tYmluZV9raW5kKGhvdykKICAgIGV4Y2VwdCAoVHlwZUVycm9yLCBWYWx1ZUVycm9yKToKICAgICAgICBwYXNzCiAgICBmb3IgbGluZSBpbiBoZWFkZXIuZ2V0KCJISVNUT1JZIiwgW10pOgogICAgICAgIG0gPSByZS5zZWFyY2gociIobWVkaWFufGF2ZXJhZ2V8bWVhbnxzdW0pXHMrb2ZccysoXGQrKVxzKyIKICAgICAgICAgICAgICAgICAgICAgIHIiKD86aW1hZ2VzfGV4cG9zdXJlc3xmcmFtZXMpIiwgc3RyKGxpbmUpLCByZS5JKQogICAgICAgIGlmIG06CiAgICAgICAgICAgIHJldHVybiBpbnQobS5ncm91cCgyKSksIF9jb21iaW5lX2tpbmQobS5ncm91cCgxKSkKICAgIHJldHVybiAxLCAic2luZ2xlIgoKCmRlZiBfY29tYmluZV9raW5kKHdvcmQpOgogICAgd29yZCA9IHN0cih3b3JkKS5sb3dlcigpCiAgICByZXR1cm4gIm1lZGlhbiIgaWYgIm1lZCIgaW4gd29yZCBlbHNlICJzdW0iIGlmICJzdW0iIGluIHdvcmQgZWxzZSAibWVhbiIKCgpkZWYgZGV0ZWN0b3JfZ2FpbihoZWFkZXIpOgogICAgIiIiRWxlY3Ryb25zIHBlciBBRFUgb2YgVEhJUyBmaWxlLCBhbmQgd2hlcmUgdGhlIG51bWJlciBjYW1lIGZyb20uCgogICAgRUdBSU4gaXMgcHJlZmVycmVkOiBvbiBzb21lIGNhbWVyYXMgR0FJTiBpcyB0aGUgZ2FpbiBTRVRUSU5HICg1MSBvbiB0aGUKICAgIEtpbm5lcmV0IGNvbG91ciBjYW1lcmEpLCBub3QgZWxlY3Ryb25zIHBlciBBRFUuIEEgbWVkaWFuIG9mIE4gZnJhbWVzIGhhcwogICAgdGhlIG5vaXNlIG9mIG9uZSBmcmFtZSB3aXRoIE4geCAyL3BpIHRpbWVzIHRoZSBnYWluLCBhIG1lYW4gb2YgTiBmcmFtZXMKICAgIE4gdGltZXM7IGEgc3VtIGtlZXBzIHRoZSBnYWluIG9mIG9uZSBmcmFtZS4KICAgICIiIgogICAgZ2Fpbiwgc291cmNlID0gTm9uZSwgTm9uZQogICAgZm9yIGtleSBpbiAoIkVHQUlOIiwgIkdBSU4iKToKICAgICAgICB0cnk6CiAgICAgICAgICAgIHYgPSBmbG9hdChoZWFkZXIuZ2V0KGtleSkpCiAgICAgICAgZXhjZXB0IChUeXBlRXJyb3IsIFZhbHVlRXJyb3IpOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGlmIDAgPCB2IDwgMjA6CiAgICAgICAgICAgIGdhaW4sIHNvdXJjZSA9IHYsIGtleQogICAgICAgICAgICBicmVhawogICAgaWYgZ2FpbiBpcyBOb25lOgogICAgICAgIGdhaW4sIHNvdXJjZSA9IEdBSU5fRV9QRVJfQURVLCAiZGVmYXVsdCBjb25zdGFudCIKICAgIG4sIGhvdyA9IGNvbWJpbmVkX2NvdW50KGhlYWRlcikKICAgIGlmIG4gPiAxIGFuZCBob3cgPT0gInN1bSI6CiAgICAgICAgcmV0dXJuIGdhaW4sIGYie3NvdXJjZX0gKHN1bSBvZiB7bn0gZnJhbWVzKSIKICAgIGlmIG4gPiAxOgogICAgICAgIGZhY3RvciA9IG4gKiAoMi4wIC8gbnAucGkgaWYgaG93ID09ICJtZWRpYW4iIGVsc2UgMS4wKQogICAgICAgIHJldHVybiBnYWluICogZmFjdG9yLCBmIntzb3VyY2V9IHgge2ZhY3RvcjouMWZ9ICh7aG93fSBvZiB7bn0gZnJhbWVzKSIKICAgIHJldHVybiBnYWluLCBzb3VyY2UKCgpkZWYgcmVhZF9ub2lzZShoZWFkZXIpOgogICAgZm9yIGtleSBpbiAoIlJETk9JU0UiLCAiUkVBRE5PSVMiLCAiUkROT0lTRTEiLCAiUk9OIik6CiAgICAgICAgdHJ5OgogICAgICAgICAgICB2ID0gZmxvYXQoaGVhZGVyLmdldChrZXkpKQogICAgICAgICAgICBpZiB2ID4gMDoKICAgICAgICAgICAgICAgIHJldHVybiB2LCBrZXkKICAgICAgICBleGNlcHQgKFR5cGVFcnJvciwgVmFsdWVFcnJvcik6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICByZXR1cm4gUkVBRF9OT0lTRV9FLCAiZGVmYXVsdCBjb25zdGFudCIKCgpkZWYgc2F0dXJhdGlvbl9sZXZlbChyYXcsIGhlYWRlcik6CiAgICAiIiJUaGUgcGl4ZWwgdmFsdWUgYWJvdmUgd2hpY2ggYSBzb3VyY2UgY291bnRzIGFzIHNhdHVyYXRlZCwgYW5kIGhvdyBpdAogICAgd2FzIGZvdW5kOiBmcm9tIFNBVFVSQVRFIChvciBNQVhMSU4pIGluIHRoZSBoZWFkZXIgd2hlbiB0aGVyZSBpcyBvbmUsCiAgICBvdGhlcndpc2UgZnJvbSB0aGUgcGlsZS11cCBvZiBzYXR1cmF0ZWQgcGl4ZWxzIGF0IHRoZSB0b3Agb2YgdGhlIGltYWdlJ3MKICAgIGhpc3RvZ3JhbS4iIiIKICAgIGZvciBrZXkgaW4gKCJTQVRVUkFURSIsICJNQVhMSU4iLCAiU0FUTEVWRUwiKToKICAgICAgICB0cnk6CiAgICAgICAgICAgIHYgPSBmbG9hdChoZWFkZXIuZ2V0KGtleSkpCiAgICAgICAgZXhjZXB0IChUeXBlRXJyb3IsIFZhbHVlRXJyb3IpOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGlmIHYgPiAwOgogICAgICAgICAgICByZXR1cm4gU0FUVVJBVElPTl9GUkFDVElPTiAqIHYsIGYie1NBVFVSQVRJT05fRlJBQ1RJT046Z30geCB7a2V5fSAoe3Y6Z30pIgogICAgbSA9IEVER0VfTUFSR0lOX1BYCiAgICBpbm5lciA9IHJhd1ttOi1tLCBtOi1tXSBpZiBtaW4ocmF3LnNoYXBlKSA+IDQgKiBtIGVsc2UgcmF3CiAgICBpbm5lciA9IGlubmVyW25wLmlzZmluaXRlKGlubmVyKV0KICAgIHRvcCA9IGZsb2F0KG5wLnNvcnQoaW5uZXIpWy0yMF0pCiAgICBoaWdoID0gaW5uZXJbaW5uZXIgPiAwLjQgKiB0b3BdCiAgICBsZXZlbCwgaG93ID0gdG9wLCAidGhlIGltYWdlJ3MgbWF4aW11bSIKICAgIGlmIGhpZ2guc2l6ZSA+IDIwMDoKICAgICAgICBjb3VudHMsIGVkZ2VzID0gbnAuaGlzdG9ncmFtKGhpZ2gsIGJpbnM9NjAsIHJhbmdlPSgwLjQgKiB0b3AsIHRvcCkpCiAgICAgICAgZm9yIGkgaW4gcmFuZ2UoMTAsIGxlbihjb3VudHMpKToKICAgICAgICAgICAgYmVmb3JlID0gbnAubWVkaWFuKGNvdW50c1tpIC0gMTA6aV0pCiAgICAgICAgICAgIGlmIGNvdW50c1tpXSA+PSAyMCBhbmQgY291bnRzW2ldID4gMi4wICogbWF4KGJlZm9yZSwgMS4wKToKICAgICAgICAgICAgICAgIGxldmVsID0gZmxvYXQoZWRnZXNbaV0pCiAgICAgICAgICAgICAgICBob3cgPSAidGhlIHBpbGUtdXAgb2Ygc2F0dXJhdGVkIHBpeGVscyBpbiB0aGUgaGlzdG9ncmFtIgogICAgICAgICAgICAgICAgYnJlYWsKICAgIHJldHVybiBTQVRVUkFUSU9OX0ZSQUNUSU9OICogbGV2ZWwsIGYie1NBVFVSQVRJT05fRlJBQ1RJT046Z30geCB7aG93fSAoe2xldmVsOi4wZn0pIgoKCmRlZiBzdWJ0cmFjdF9iYWNrZ3JvdW5kX2FuZF9zYXZlKHBhdGgsIG91dF9kaXIpOgogICAgIiIiU3VidHJhY3QgdGhlIGZyYW1lJ3MgbWVkaWFuIGxldmVsIGFuZCBzYXZlIHRoZSByZXN1bHQgaW4gb3V0X2Rpci4KCiAgICBSZXR1cm5zIChwYXRoIHdyaXR0ZW4sIHJhdyBwaXhlbHMsIGJhY2tncm91bmQtc3VidHJhY3RlZCBwaXhlbHMsIGhlYWRlcikuCiAgICAiIiIKICAgIHJhdywgaGRyID0gcmVhZF9pbWFnZShwYXRoKQogICAgYmdfdmFsID0gZmxvYXQoQkdfU1VCX0ZVTkMocmF3KSkKICAgIGRhdGFfc3ViID0gcmF3IC0gbnAuZmxvYXQzMihiZ192YWwpCiAgICBzdGVtID0gb3MucGF0aC5iYXNlbmFtZShwYXRoKQogICAgZm9yIHN1ZmZpeCBpbiAoIi5maXRzLmZ6IiwgIi5maXRzIiwgIi5maXQiLCAiLmZ0cyIpOgogICAgICAgIGlmIHN0ZW0ubG93ZXIoKS5lbmRzd2l0aChzdWZmaXgpOgogICAgICAgICAgICBzdGVtID0gc3RlbVs6LWxlbihzdWZmaXgpXQogICAgICAgICAgICBicmVhawogICAgb3V0X3BhdGggPSBvcy5wYXRoLmpvaW4ob3V0X2Rpciwgc3RlbSArICJfYmdzdWIuZml0cyIpCiAgICBmaXRzLlByaW1hcnlIRFUoZGF0YT1kYXRhX3N1YiwgaGVhZGVyPWhkcikud3JpdGV0byhvdXRfcGF0aCwgb3ZlcndyaXRlPVRydWUpCiAgICBwcmludChmIltiZ3N1Yl0gd3JvdGUge291dF9wYXRofSAoYmc9e2JnX3ZhbDouM2Z9KSIpCiAgICByZXR1cm4gb3V0X3BhdGgsIHJhdywgZGF0YV9zdWIsIGhkcgoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLSBGaW5kaW5nIGFuZCBtZWFzdXJpbmcgdGhlIHNvdXJjZXMgLS0tLS0tLS0tLS0tLS0tLS0tLQoKZGVmIGxvY2FsX2JhY2tncm91bmQoZGF0YSk6CiAgICAiIiJCYWNrZ3JvdW5kIGFuZCBub2lzZSBvbiBhIGNvYXJzZSBncmlkLCBzbyBhIGdhbGF4eSdzIG93biBsaWdodCBkb2VzIG5vdAogICAgc2V0IHRoZSBkZXRlY3Rpb24gdGhyZXNob2xkIGZvciB0aGUgZW1wdHkgc2t5IGFyb3VuZCBpdC4iIiIKICAgIHJldHVybiBCYWNrZ3JvdW5kMkQoZGF0YSwgKEJBQ0tHUk9VTkRfQk9YX1NJWkUsIEJBQ0tHUk9VTkRfQk9YX1NJWkUpLAogICAgICAgICAgICAgICAgICAgICAgICBmaWx0ZXJfc2l6ZT0oQkFDS0dST1VORF9GSUxURVJfU0laRSwgQkFDS0dST1VORF9GSUxURVJfU0laRSksCiAgICAgICAgICAgICAgICAgICAgICAgIHNpZ21hX2NsaXA9U2lnbWFDbGlwKHNpZ21hPVNJR01BX0NMSVApLAogICAgICAgICAgICAgICAgICAgICAgICBia2dfZXN0aW1hdG9yPU1lZGlhbkJhY2tncm91bmQoKSkKCgpkZWYgc3Rhcl93aWR0aChkYXRhLCB4LCB5LCBoYWxmPTEyLCBza3k9MC4wKToKICAgICIiIldpZHRoIGF0IGhhbGYgbWF4aW11bSBvZiBvbmUgc291cmNlLCBpbiBwaXhlbHMsIG1lYXN1cmVkIG9uIHRoZSBkYXRhIGFzCiAgICBpdCBpcywgYWZ0ZXIgaXRzIGxvY2FsIHNreSBpcyB0YWtlbiBvZmYuCgogICAgQSB0d28tZGltZW5zaW9uYWwgR2F1c3NpYW4gaXMgZml0dGVkIHRvIHRoZSBzdGFtcCwgYW5kIHRoZSBGV0hNIGlzCiAgICAyLjM1NSB4IHNxcnQoc2lnbWFfeCBzaWdtYV95KS4gQ291bnRpbmcgdGhlIHBpeGVscyBhYm92ZSBoYWxmIHRoZSBwZWFrLCBhcwogICAgdGhpcyBkaWQgYmVmb3JlLCBjYW4gb25seSByZXR1cm4gd2hvbGUgYW5kIGhhbGYgcGl4ZWxzOiBhIG1lZGlhbiBvdmVyIGEKICAgIGh1bmRyZWQgc3RhcnMgdGhlbiBzaXRzIG9uIDQuMDAgd2hldGhlciB0aGUgc2VlaW5nIGlzIDMuNiBvciA0LjQgcHguIFRoZQogICAgcGl4ZWwgY291bnQgaXMgc3RpbGwgdGhlIGFuc3dlciB3aGVuIHRoZSBmaXQgZmFpbHMuCiAgICAiIiIKICAgIHgsIHkgPSBpbnQocm91bmQoeCkpLCBpbnQocm91bmQoeSkpCiAgICBpZiB4IC0gaGFsZiA8IDAgb3IgeSAtIGhhbGYgPCAwIG9yIHggKyBoYWxmID49IGRhdGEuc2hhcGVbMV0gb3IgeSArIGhhbGYgPj0gZGF0YS5zaGFwZVswXToKICAgICAgICByZXR1cm4gbnAubmFuCiAgICBjdXQgPSBucC5hc2FycmF5KGRhdGFbeSAtIGhhbGY6eSArIGhhbGYsIHggLSBoYWxmOnggKyBoYWxmXSwgZmxvYXQpIC0gc2t5CiAgICBwZWFrID0gY3V0Lm1heCgpCiAgICBpZiBub3QgbnAuaXNmaW5pdGUocGVhaykgb3IgcGVhayA8PSAwOgogICAgICAgIHJldHVybiBucC5uYW4KICAgIGlkeCA9IG5wLmFyZ3doZXJlKGN1dCA+IHBlYWsgLyAyKQogICAgaWYgbm90IGxlbihpZHgpOgogICAgICAgIHJldHVybiBucC5uYW4KICAgIGNvYXJzZSA9IG5wLm1lYW4oW25wLnB0cChpZHhbOiwgMV0pICsgMSwgbnAucHRwKGlkeFs6LCAwXSkgKyAxXSkKICAgIHRyeToKICAgICAgICBmcm9tIGFzdHJvcHkubW9kZWxpbmcgaW1wb3J0IG1vZGVscywgZml0dGluZwogICAgICAgIHl5LCB4eCA9IG5wLm1ncmlkWzA6Y3V0LnNoYXBlWzBdLCAwOmN1dC5zaGFwZVsxXV0KICAgICAgICBweSwgcHggPSBucC51bnJhdmVsX2luZGV4KG5wLmFyZ21heChjdXQpLCBjdXQuc2hhcGUpCiAgICAgICAgczAgPSBtYXgoY29hcnNlIC8gMi4zNTUsIDAuNSkKICAgICAgICBnID0gbW9kZWxzLkdhdXNzaWFuMkQocGVhaywgcHgsIHB5LCBzMCwgczAsIDAuMCkgKyBtb2RlbHMuQ29uc3QyRCgwLjApCiAgICAgICAgd2l0aCB3YXJuaW5ncy5jYXRjaF93YXJuaW5ncygpOgogICAgICAgICAgICB3YXJuaW5ncy5zaW1wbGVmaWx0ZXIoImlnbm9yZSIpCiAgICAgICAgICAgIGZpdCA9IGZpdHRpbmcuTGV2TWFyTFNRRml0dGVyKCkoZywgeHgsIHl5LCBucC5uYW5fdG9fbnVtKGN1dCkpCiAgICAgICAgc3gsIHN5ID0gYWJzKGZpdC54X3N0ZGRldl8wLnZhbHVlKSwgYWJzKGZpdC55X3N0ZGRldl8wLnZhbHVlKQogICAgICAgIGZ3aG0gPSAyLjM1NDggKiBucC5zcXJ0KHN4ICogc3kpCiAgICAgICAgaWYgKG5wLmlzZmluaXRlKGZ3aG0pIGFuZCBmaXQuYW1wbGl0dWRlXzAudmFsdWUgPiAwIGFuZCAwLjUgPCBmd2htIDwgMiAqIGhhbGYKICAgICAgICAgICAgICAgIGFuZCBhYnMoZml0LnhfbWVhbl8wLnZhbHVlIC0gcHgpIDwgMiBhbmQgYWJzKGZpdC55X21lYW5fMC52YWx1ZSAtIHB5KSA8IDIpOgogICAgICAgICAgICByZXR1cm4gZmxvYXQoZndobSkKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcGFzcwogICAgcmV0dXJuIGZsb2F0KGNvYXJzZSkKCgpkZWYgX3h5X29mKHRhYmxlKToKICAgICIiInggYW5kIHkgb2YgYSBEQU9TdGFyRmluZGVyIHRhYmxlLiBwaG90dXRpbHMgMy4wIHJlbmFtZWQgeGNlbnRyb2lkIGFuZAogICAgeWNlbnRyb2lkIHRvIHhfY2VudHJvaWQgYW5kIHlfY2VudHJvaWQ7IGJvdGggYXJlIGFjY2VwdGVkLiIiIgogICAgbmFtZXMgPSB0YWJsZS5jb2xuYW1lcwogICAgeHMgPSAieF9jZW50cm9pZCIgaWYgInhfY2VudHJvaWQiIGluIG5hbWVzIGVsc2UgInhjZW50cm9pZCIKICAgIHlzID0gInlfY2VudHJvaWQiIGlmICJ5X2NlbnRyb2lkIiBpbiBuYW1lcyBlbHNlICJ5Y2VudHJvaWQiCiAgICByZXR1cm4gbnAuYXNhcnJheSh0YWJsZVt4c10sIGZsb2F0KSwgbnAuYXNhcnJheSh0YWJsZVt5c10sIGZsb2F0KQoKCmRlZiBlc3RpbWF0ZV9zZWVpbmcoZGF0YSwgYmFja2dyb3VuZCwgcm1zLCBiYW5kLCByYXc9Tm9uZSwgc2F0X2xldmVsPU5vbmUpOgogICAgIiIiSG93IHdpZGUgYSBzdGFyIGlzIGluIHRoaXMgZnJhbWU6IHRoZSBtZWRpYW4gd2lkdGggb2YgYnJpZ2h0IHN0YXJzIHRoYXQKICAgIGFyZSBOT1Qgc2F0dXJhdGVkLiBBIHNhdHVyYXRlZCBzdGFyIGhhcyBhIGZsYXQgdG9wLCBzbyBpdHMgd2lkdGggYXQgaGFsZgogICAgdGhlIGZsYXQgdG9wIHNheXMgbm90aGluZyBhYm91dCB0aGUgc2VlaW5nIC0gYW5kIHRoZSBhcGVydHVyZSBvZiBldmVyeQogICAgc291cmNlIGlzIHNpemVkIGZyb20gdGhpcyBudW1iZXIuIiIiCiAgICB0cnk6CiAgICAgICAgZm91bmQgPSBEQU9TdGFyRmluZGVyKGZ3aG09U0VFSU5HX0ZBTExCQUNLLCB0aHJlc2hvbGQ9MjAuMCAqIHJtcykoZGF0YSAtIGJhY2tncm91bmQpCiAgICAgICAgaWYgZm91bmQgaXMgbm90IE5vbmUgYW5kIGxlbihmb3VuZCkgPj0gNToKICAgICAgICAgICAgZngsIGZ5ID0gX3h5X29mKGZvdW5kKQogICAgICAgICAgICBmbHV4ID0gbnAuYXNhcnJheShmb3VuZFsnZmx1eCddLCBmbG9hdCkKICAgICAgICAgICAgaWYgcmF3IGlzIG5vdCBOb25lIGFuZCBzYXRfbGV2ZWwgaXMgbm90IE5vbmU6CiAgICAgICAgICAgICAgICBwZWFrcyA9IG5wLmFycmF5KFtucC5uYW5tYXgocmF3W21heChpbnQocm91bmQoeSkpIC0gMSwgMCk6aW50KHJvdW5kKHkpKSArIDIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1heChpbnQocm91bmQoeCkpIC0gMSwgMCk6aW50KHJvdW5kKHgpKSArIDJdKQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZm9yIHgsIHkgaW4gemlwKGZ4LCBmeSldKQogICAgICAgICAgICAgICAgb2sgPSBwZWFrcyA8IHNhdF9sZXZlbAogICAgICAgICAgICAgICAgZngsIGZ5LCBmbHV4ID0gZnhbb2tdLCBmeVtva10sIGZsdXhbb2tdCiAgICAgICAgICAgIGlmIGxlbihmeCkgPj0gNToKICAgICAgICAgICAgICAgIG9yZGVyID0gbnAuYXJnc29ydChmbHV4KVstMjAwOl0KICAgICAgICAgICAgICAgIHdpZHRocyA9IG5wLmFzYXJyYXkoW3N0YXJfd2lkdGgoZGF0YSAtIGJhY2tncm91bmQsIGZ4W2ldLCBmeVtpXSkKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZvciBpIGluIG9yZGVyXSwgZHR5cGU9ZmxvYXQpCiAgICAgICAgICAgICAgICBzZWVpbmcgPSBucC5uYW5tZWRpYW4od2lkdGhzKQogICAgICAgICAgICAgICAgaWYgbnAuaXNmaW5pdGUoc2VlaW5nKSBhbmQgU0VFSU5HX01JTiA8PSBzZWVpbmcgPD0gU0VFSU5HX01BWDoKICAgICAgICAgICAgICAgICAgICBwcmludChmIiAgIHtiYW5kfTogc3RhcnMgYXJlIHtzZWVpbmc6LjJmfSBweCB3aWRlIChtZWRpYW4gb2YgIgogICAgICAgICAgICAgICAgICAgICAgICAgIGYie2ludChucC5pc2Zpbml0ZSh3aWR0aHMpLnN1bSgpKX0gYnJpZ2h0IHVuc2F0dXJhdGVkIHN0YXJzKSIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgZmx1c2g9VHJ1ZSkKICAgICAgICAgICAgICAgICAgICByZXR1cm4gZmxvYXQoc2VlaW5nKQogICAgICAgIHByaW50KGYiICAge2JhbmR9OiBjb3VsZCBub3QgbWVhc3VyZSB0aGUgc3RhciB3aWR0aCwgdXNpbmciCiAgICAgICAgICAgICAgZiIge1NFRUlOR19GQUxMQkFDS30gcHgiLCBmbHVzaD1UcnVlKQogICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBleGM6CiAgICAgICAgcHJpbnQoZiIgICB7YmFuZH06IHN0YXIgd2lkdGggbWVhc3VyZW1lbnQgZmFpbGVkICh7ZXhjfSksIHVzaW5nIgogICAgICAgICAgICAgIGYiIHtTRUVJTkdfRkFMTEJBQ0t9IHB4IiwgZmx1c2g9VHJ1ZSkKICAgIHJldHVybiBmbG9hdChTRUVJTkdfRkFMTEJBQ0spCgoKZGVmIGRldGVjdF9zb3VyY2VzKGRhdGEsIHJhdywgYmFuZCwgc2F0X2xldmVsLCB3aW5kb3c9Tm9uZSk6CiAgICAiIiJGaW5kIHRoZSBzb3VyY2VzIGluIG9uZSBmcmFtZS4gUmV0dXJucyAoeCwgeSwgZmx1eCBlc3RpbWF0ZSkgaW4gdGhlCiAgICBjb29yZGluYXRlcyBvZiB0aGUgd2hvbGUgZnJhbWUsIGFuZCB0aGUgZnJhbWUncyB0eXBpY2FsIHN0YXIgd2lkdGguCgogICAgYHdpbmRvd2AgaXMgKHkwLCB5MSwgeDAsIHgxKTogdGhlIHBhcnQgb2YgdGhlIGZyYW1lIHRoZSBnYWxheHkgb2NjdXBpZXMuCiAgICBGaW5kaW5nIHNvdXJjZXMgaXMgcGFpZCBmb3IgYnkgdGhlIHBpeGVsLCBzbyB3b3JraW5nIG91dHNpZGUgdGhlIGdhbGF4eQogICAgY29zdHMgdGltZSBhbmQgcmV0dXJucyBub3RoaW5nLgogICAgIiIiCiAgICB4X29mZnNldCA9IHlfb2Zmc2V0ID0gMAogICAgd29yaywgd29ya19yYXcgPSBkYXRhLCByYXcKICAgIGlmIHdpbmRvdyBpcyBub3QgTm9uZToKICAgICAgICB5MCwgeTEsIHgwLCB4MSA9IHdpbmRvdwogICAgICAgIHdvcmssIHdvcmtfcmF3ID0gZGF0YVt5MDp5MSwgeDA6eDFdLCByYXdbeTA6eTEsIHgwOngxXQogICAgICAgIHhfb2Zmc2V0LCB5X29mZnNldCA9IHgwLCB5MAogICAgICAgIHByaW50KGYiICAge2JhbmR9OiB3b3JraW5nIG9uIHt4MSAtIHgwfSB4IHt5MSAtIHkwfSBwaXhlbHMgYXJvdW5kIHRoZSAiCiAgICAgICAgICAgICAgZiJnYWxheHkiLCBmbHVzaD1UcnVlKQoKICAgIHByaW50KGYiRmluZGluZyBzb3VyY2VzIGluIHRoZSB7YmFuZH0gaW1hZ2UuLi4iLCBmbHVzaD1UcnVlKQogICAgYmtnID0gbG9jYWxfYmFja2dyb3VuZCh3b3JrKQogICAgcm1zID0gZmxvYXQobnAubWVkaWFuKGJrZy5iYWNrZ3JvdW5kX3JtcykpCiAgICBwcmludChmIiAgIHtiYW5kfTogbG9jYWwgc2t5IHJ1bnMgZnJvbSB7YmtnLmJhY2tncm91bmQubWluKCk6LjFmfSIKICAgICAgICAgIGYiIHRvIHtia2cuYmFja2dyb3VuZC5tYXgoKTouMWZ9IGNvdW50cywgbm9pc2Uge3JtczouMmZ9IiwgZmx1c2g9VHJ1ZSkKCiAgICBmd2htID0gREFPRklORF9GV0hNIGlmIERBT0ZJTkRfRldITSBlbHNlIGVzdGltYXRlX3NlZWluZygKICAgICAgICB3b3JrLCBia2cuYmFja2dyb3VuZCwgcm1zLCBiYW5kLCB3b3JrX3Jhdywgc2F0X2xldmVsKQogICAgU0VFSU5HX01FQVNVUkVEW2JhbmRdID0gZndobQoKICAgIHNvdXJjZXMgPSBEQU9TdGFyRmluZGVyKGZ3aG09ZndobSwgdGhyZXNob2xkPURFVEVDVElPTl9USFJFU0hPTERfU0lHTUEgKiBybXMpKAogICAgICAgIHdvcmsgLSBia2cuYmFja2dyb3VuZCkKICAgIGlmIHNvdXJjZXMgaXMgTm9uZSBvciBsZW4oc291cmNlcykgPT0gMDoKICAgICAgICBwcmludChmIiAgIHtiYW5kfTogbm8gc291cmNlcyBmb3VuZCIsIGZsdXNoPVRydWUpCiAgICAgICAgcmV0dXJuIG5wLmVtcHR5KDApLCBucC5lbXB0eSgwKSwgbnAuZW1wdHkoMCksIGZ3aG0KICAgIHNvdXJjZXMgPSBzb3VyY2VzW3NvdXJjZXNbJ3BlYWsnXSA+IFBFQUtfTUlOX1NURCAqIHJtc10KICAgIHgsIHkgPSBfeHlfb2Yoc291cmNlcykKICAgIGYgPSBucC5hc2FycmF5KHNvdXJjZXNbImZsdXgiXSwgZmxvYXQpCiAgICBvcmRlciA9IG5wLmFyZ3NvcnQoZilbOjotMV0KICAgIHgsIHksIGYgPSB4W29yZGVyXSArIHhfb2Zmc2V0LCB5W29yZGVyXSArIHlfb2Zmc2V0LCBmW29yZGVyXQoKICAgICMgT25lIHNvdXJjZSBmb3VuZCB0d28gb3IgdGhyZWUgdGltZXMgb3ZlciAtIGEgc2F0dXJhdGVkIHN0YXIgd2l0aCBhIGZsYXQKICAgICMgdG9wLCBvciBhIGtub3Qgd2l0aCB0d28gcGVha3MgaW5zaWRlIG9uZSBzZWVpbmcgZGlzYyAtIHdvdWxkIG90aGVyd2lzZSBiZQogICAgIyBtZWFzdXJlZCBhcyBzZXZlcmFsIHNvdXJjZXMgc2hhcmluZyBvbmUgYXBlcnR1cmUuIFR3byBzb3VyY2VzIGNsb3NlciB0aGFuCiAgICAjIG9uZSBGV0hNIGNhbm5vdCBiZSBtZWFzdXJlZCBhcGFydCwgc28gb25seSB0aGUgYnJpZ2h0ZXN0IGlzIGtlcHQuCiAgICBrZWVwID0gbnAub25lcyhsZW4oeCksIGR0eXBlPWJvb2wpCiAgICB0cmVlID0gY0tEVHJlZShucC5jb2x1bW5fc3RhY2soW3gsIHldKSkKICAgIGZvciBpIGluIHJhbmdlKGxlbih4KSk6CiAgICAgICAgaWYga2VlcFtpXToKICAgICAgICAgICAgZm9yIGogaW4gdHJlZS5xdWVyeV9iYWxsX3BvaW50KFt4W2ldLCB5W2ldXSwgcj1md2htKToKICAgICAgICAgICAgICAgIGlmIGogPiBpOgogICAgICAgICAgICAgICAgICAgIGtlZXBbal0gPSBGYWxzZQogICAgcHJpbnQoZiIgICB7YmFuZH06IHtsZW4oeCk6LH0gZGV0ZWN0aW9ucywge2ludCgofmtlZXApLnN1bSgpKTosfSBvZiB0aGVtICIKICAgICAgICAgIGYid2l0aGluIG9uZSBGV0hNIG9mIGEgYnJpZ2h0ZXIgb25lIGFuZCBkcm9wcGVkIiwgZmx1c2g9VHJ1ZSkKICAgIHJldHVybiB4W2tlZXBdLCB5W2tlZXBdLCBmW2tlZXBdLCBmd2htCgoKZGVmIHJlZ2lzdGVyX0Jfb250b19WKEJfaW1nLCBWX2ltZywgZml0c19maWxlX0IsIGZpdHNfZmlsZV9WKToKICAgICIiIlRoZSBtYXBwaW5nIGJldHdlZW4gQidzIHBpeGVsIGdyaWQgYW5kIFYncywgYm90aCB3YXlzLgoKICAgIEIgYW5kIFYgYXJlIHNlcGFyYXRlIGV4cG9zdXJlcyBhbmQgbm90aGluZyBndWFyYW50ZWVzIHRoZWlyIGdyaWRzCiAgICBjb2luY2lkZS4gYXN0cm9hbGlnbiBmaW5kcyB0aGUgcmVnaXN0cmF0aW9uIGZyb20gdGhlIHN0YXIgcGF0dGVybnMgb2YKICAgIHRoZXNlIHR3byBpbWFnZXM7IGVhY2ggZnJhbWUncyBvd24gcGxhdGUgc29sdXRpb24gaXMgdGhlIGZhbGxiYWNrLgoKICAgIFJldHVybnMgKGJfdG9fdiwgdl90b19iLCBkZXNjcmlwdGlvbik7IGVhY2ggdGFrZXMgYW4gKE4sIDIpIGFycmF5LgogICAgIiIiCiAgICB0cnk6CiAgICAgICAgdHJhbnNmLCAoc3JjX2xpc3QsIF90Z3RfbGlzdCkgPSBhYS5maW5kX3RyYW5zZm9ybSgKICAgICAgICAgICAgbnAubmFuX3RvX251bShCX2ltZyksIG5wLm5hbl90b19udW0oVl9pbWcpKQogICAgICAgIGlmIGFicyh0cmFuc2Yuc2NhbGUgLSAxLjApID4gQVNUUk9BTElHTl9NQVhfU0NBTEVfRVJST1I6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJmaXR0ZWQgc2NhbGUge3RyYW5zZi5zY2FsZTouM2Z9IGlzIG5vdCBwbGF1c2libGUgIgogICAgICAgICAgICAgICAgICAgICAgICAgICAgIGYiZm9yIHR3byBmcmFtZXMgZnJvbSB0aGUgc2FtZSB0ZWxlc2NvcGUiKQogICAgICAgIHRleHQgPSAoZiJmcm9tIHtsZW4oc3JjX2xpc3QpfSBtYXRjaGVkIHN0YXIgcGFpcnMgKGFzdHJvYWxpZ24pOiBzaGlmdCAiCiAgICAgICAgICAgICAgICBmIih7dHJhbnNmLnRyYW5zbGF0aW9uWzBdOisuMWZ9LCB7dHJhbnNmLnRyYW5zbGF0aW9uWzFdOisuMWZ9KSBweCwgIgogICAgICAgICAgICAgICAgZiJyb3RhdGlvbiB7bnAuZGVncmVlcyh0cmFuc2Yucm90YXRpb24pOisuMmZ9IGRlZywgc2NhbGUgIgogICAgICAgICAgICAgICAgZiJ7dHJhbnNmLnNjYWxlOi40Zn0iKQogICAgICAgIHByaW50KCIgICByZWdpc3RlcmVkIEIgb250byBWICIgKyB0ZXh0LCBmbHVzaD1UcnVlKQogICAgICAgIGludmVyc2UgPSB0cmFuc2YuaW52ZXJzZQogICAgICAgIHJldHVybiAobGFtYmRhIHh5OiB0cmFuc2YobnAuYXNhcnJheSh4eSwgZmxvYXQpKSwKICAgICAgICAgICAgICAgIGxhbWJkYSB4eTogaW52ZXJzZShucC5hc2FycmF5KHh5LCBmbG9hdCkpLCB0ZXh0KQogICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgICAgIHByaW50KGYiICAgYXN0cm9hbGlnbiByZWdpc3RyYXRpb24gZmFpbGVkICh7ZX0pIC0gIgogICAgICAgICAgICAgIGYiZmFsbGluZyBiYWNrIHRvIGVhY2ggZnJhbWUncyBvd24gV0NTIiwgZmx1c2g9VHJ1ZSkKICAgICAgICB3Y3NfQiA9IFdDUyhmaXRzLmdldGhlYWRlcihmaXRzX2ZpbGVfQikpLmNlbGVzdGlhbAogICAgICAgIHdjc19WID0gV0NTKGZpdHMuZ2V0aGVhZGVyKGZpdHNfZmlsZV9WKSkuY2VsZXN0aWFsCgogICAgICAgIGRlZiBiX3RvX3YoeHkpOgogICAgICAgICAgICB4eSA9IG5wLmFzYXJyYXkoeHksIGZsb2F0KQogICAgICAgICAgICByYSwgZGVjID0gd2NzX0IuYWxsX3BpeDJ3b3JsZCh4eVs6LCAwXSwgeHlbOiwgMV0sIDApCiAgICAgICAgICAgIHJldHVybiBucC5jb2x1bW5fc3RhY2sod2NzX1YuYWxsX3dvcmxkMnBpeChyYSwgZGVjLCAwKSkKCiAgICAgICAgZGVmIHZfdG9fYih4eSk6CiAgICAgICAgICAgIHh5ID0gbnAuYXNhcnJheSh4eSwgZmxvYXQpCiAgICAgICAgICAgIHJhLCBkZWMgPSB3Y3NfVi5hbGxfcGl4MndvcmxkKHh5WzosIDBdLCB4eVs6LCAxXSwgMCkKICAgICAgICAgICAgcmV0dXJuIG5wLmNvbHVtbl9zdGFjayh3Y3NfQi5hbGxfd29ybGQycGl4KHJhLCBkZWMsIDApKQogICAgICAgIHJldHVybiBiX3RvX3YsIHZfdG9fYiwgInRocm91Z2ggdGhlIHR3byBwbGF0ZSBzb2x1dGlvbnMiCgoKZGVmIGFwZXJ0dXJlX3Bob3RvbWV0cnlfYXQoZGF0YSwgcmF3LCB4eSwgcl9hcCwgcl9pbiwgcl9vdXQsIGdhaW4sIHJuKToKICAgICIiIkZsdXgsIGl0cyBlcnJvciwgdGhlIGJyaWdodGVzdCByYXcgcGl4ZWwsIGFuZCB0aGUgbG9jYWwgc2t5LCBmb3IgZXZlcnkKICAgIHBvc2l0aW9uIGluIHh5IC0gYWxsIHRocm91Z2ggb25lIGFwZXJ0dXJlIHJhZGl1cy4KCiAgICBUaGUgc2t5IGlzIHRoZSBzaWdtYS1jbGlwcGVkIG1lZGlhbiBvZiBhIHJpbmcgYXJvdW5kIHRoZSBzb3VyY2UsIHNvIHRoZQogICAgc21vb3RoIGxpZ2h0IG9mIHRoZSBnYWxheHkgdW5kZXIgYSBrbm90IGlzIHRha2VuIG9mZiB3aXRoIGl0LiBUaGUgZXJyb3IKICAgIGNvdW50cyB0aGUgcGhvdG9ucyBvZiB0aGUgc291cmNlLCB0aGUgc2t5IG5vaXNlIG1lYXN1cmVkIGluIHRoZSByaW5nIChvbmNlCiAgICBwZXIgYXBlcnR1cmUgcGl4ZWwsIGFuZCBvbmNlIG1vcmUgdGhyb3VnaCB0aGUgdW5jZXJ0YWludHkgb2YgdGhlIHNreSBsZXZlbAogICAgaXRzZWxmKSwgd2l0aCB0aGUgaGVhZGVyJ3MgcmVhZCBub2lzZSBhcyBhIGZsb29yLgogICAgIiIiCiAgICB4eSA9IG5wLmFzYXJyYXkoeHksIGZsb2F0KQogICAgYXAgPSBDaXJjdWxhckFwZXJ0dXJlKHh5LCByPXJfYXApCiAgICBhbm4gPSBDaXJjdWxhckFubnVsdXMoeHksIHJfaW49cl9pbiwgcl9vdXQ9cl9vdXQpCiAgICBza3kgPSBBcGVydHVyZVN0YXRzKGRhdGEsIGFubiwgc2lnbWFfY2xpcD1TaWdtYUNsaXAoc2lnbWE9My4wLCBtYXhpdGVycz01KSkKICAgIHNreV9tZWQgPSBucC5hdGxlYXN0XzFkKG5wLmFzYXJyYXkoc2t5Lm1lZGlhbiwgZmxvYXQpKQogICAgc2t5X3N0ZCA9IG5wLmF0bGVhc3RfMWQobnAuYXNhcnJheShza3kuc3RkLCBmbG9hdCkpCiAgICBuX3NreSA9IG5wLmF0bGVhc3RfMWQobnAuYXNhcnJheShza3kuc3VtX2FwZXJfYXJlYS52YWx1ZSwgZmxvYXQpKQogICAgdG90YWwgPSBucC5hc2FycmF5KGFwZXJ0dXJlX3Bob3RvbWV0cnkoZGF0YSwgYXApWyJhcGVydHVyZV9zdW0iXSwgZmxvYXQpCiAgICBuX2FwID0gZmxvYXQoYXAuYXJlYSkKICAgIGZsdXggPSB0b3RhbCAtIHNreV9tZWQgKiBuX2FwCiAgICBwaXhfdmFyID0gbnAubWF4aW11bShza3lfc3RkICoqIDIsIChybiAvIGdhaW4pICoqIDIpCiAgICB2YXIgPSAobnAuY2xpcChmbHV4LCAwLCBOb25lKSAvIGdhaW4KICAgICAgICAgICArIG5fYXAgKiBwaXhfdmFyICogKDEuMCArIG5fYXAgLyBucC5tYXhpbXVtKG5fc2t5LCAxKSkpCiAgICBwZWFrID0gbnAuYXRsZWFzdF8xZChucC5hc2FycmF5KAogICAgICAgIEFwZXJ0dXJlU3RhdHMocmF3LCBDaXJjdWxhckFwZXJ0dXJlKHh5LCByPW1heCgxLjUsIDAuNSAqIHJfYXApKSkubWF4LCBmbG9hdCkpCiAgICByZXR1cm4gZmx1eCwgbnAuc3FydCh2YXIpLCBwZWFrLCBza3lfbWVkCgoKZGVmIGluc3RydW1lbnRhbChmbHV4LCBlcnIpOgogICAgd2l0aCBucC5lcnJzdGF0ZShkaXZpZGU9Imlnbm9yZSIsIGludmFsaWQ9Imlnbm9yZSIpOgogICAgICAgIG1hZyA9IG5wLndoZXJlKGZsdXggPiAwLCAtMi41ICogbnAubG9nMTAoZmx1eCksIG5wLm5hbikKICAgICAgICBtYWdfZXJyID0gbnAud2hlcmUoZmx1eCA+IDAsIDEuMDg1NyAqIGVyciAvIGZsdXgsIG5wLm5hbikKICAgIHJldHVybiBtYWcsIG1hZ19lcnIKCgojIC0tLS0tLS0tLS0tLS0tLS0tLS0gVGhlIHBsYXRlIHNvbHV0aW9uLCBjaGVja2VkIGFnYWluc3QgR2FpYSAtLS0tLS0tLS0tLS0tLS0tLS0tCgpkZWYgZmllbGRfY2VudHJlX2FuZF9yYWRpdXMod2NzLCBzaGFwZSk6CiAgICBueSwgbnggPSBzaGFwZQogICAgY2VudHJlID0gd2NzLnBpeGVsX3RvX3dvcmxkKG54IC8gMiwgbnkgLyAyKQogICAgY29ybmVycyA9IHdjcy5waXhlbF90b193b3JsZChucC5hcnJheShbMCwgbngsIDAsIG54XSksIG5wLmFycmF5KFswLCAwLCBueSwgbnldKSkKICAgIHJldHVybiBjZW50cmUsIGZsb2F0KG5wLm1heChjZW50cmUuc2VwYXJhdGlvbihjb3JuZXJzKS5kZWcpKQoKCmRlZiB2ZXJpZnlfcGxhdGVfc29sdXRpb24oeHlfYnJpZ2h0LCBhbGxfeHksIHdjcywgc2hhcGUpOgogICAgIiIiVGhlIHBsYXRlIHNvbHV0aW9uIG9mIHRoZSBWIGZyYW1lLCBjaGVja2VkIGFnYWluc3QgR2FpYSBhbmQgcmVwYWlyZWQgaWYKICAgIGl0IGRvZXMgbm90IGZpdC4KCiAgICBFdmVyeXRoaW5nIHRoYXQgdXNlcyB0aGUgc2t5IGdvZXMgdGhyb3VnaCBpdDogdGhlIGNhbGlicmF0aW9uIHN0YXJzLCB0aGUKICAgIGZvcmVncm91bmQgc3RhcnMsIHRoZSBnYWxheHkncyBjZW50cmUuIFNvZnR3YXJlIHRoYXQgY29tYmluZXMgb3IgY3JvcHMKICAgIGZyYW1lcyBzb21ldGltZXMga2VlcHMgYSBzb2x1dGlvbiB0aGF0IGJlbG9uZ3MgdG8gYW5vdGhlciBmcmFtZSAtIGluIHRoZQogICAgQ2x1c3Rlcl9DTUQgdGVzdHMgYSBzcGxpdC1jaGFubmVsIGZpbGUgY2FycmllZCBvbmUgNDUgcHggb2ZmLCBhbmQgb25seSA0IG9mCiAgICAyLDAwMCBjYXRhbG9ndWUgc3RhcnMgbWF0Y2hlZC4KCiAgICBUaGUgMTAwIGJyaWdodGVzdCB1bnNhdHVyYXRlZCBzb3VyY2VzIGFyZSBwdXQgdGhyb3VnaCB0aGUgc29sdXRpb24gYW5kCiAgICBsb29rZWQgdXAgaW4gR2FpYSBEUjMuIElmIGF0IGxlYXN0IGhhbGYgaGF2ZSBhIEdhaWEgc3RhciB3aXRoaW4KICAgIENBTElCX01BVENIX1JBRElVUyB0aGUgc29sdXRpb24gaXMga2VwdC4gT3RoZXJ3aXNlIHRoZSBvZmZzZXQgYmV0d2VlbiB0aGUKICAgIEdhaWEgc3RhcnMgYW5kIHRoZSBzb3VyY2VzIGlzIGZvdW5kIChvciB0aGUgc3RhciBwYXR0ZXJucyBtYXRjaGVkLCB3aXRoCiAgICB0aGUgbWlycm9yIGltYWdlIHRyaWVkIHRvbykgYW5kIGEgbmV3IHNvbHV0aW9uIGZpdHRlZC4KICAgICIiIgogICAgZnJvbSBhc3Ryb3B5Lndjcy51dGlscyBpbXBvcnQgZml0X3djc19mcm9tX3BvaW50cwogICAgbnksIG54ID0gc2hhcGUKICAgIGNlbnRyZSwgcmFkaXVzID0gZmllbGRfY2VudHJlX2FuZF9yYWRpdXMod2NzLCBzaGFwZSkKICAgIHRyeToKICAgICAgICByZXMgPSBWaXppZXIoY29sdW1ucz1bIlJBX0lDUlMiLCAiREVfSUNSUyIsICJHbWFnIl0sCiAgICAgICAgICAgICAgICAgICAgIGNvbHVtbl9maWx0ZXJzPXsiR21hZyI6ICI8MTcifSwgcm93X2xpbWl0PS0xKS5xdWVyeV9yZWdpb24oCiAgICAgICAgICAgIFNreUNvb3JkKGNlbnRyZS5yYSwgY2VudHJlLmRlYyksIHJhZGl1cz0xLjMgKiByYWRpdXMgKiB1LmRlZywKICAgICAgICAgICAgY2F0YWxvZz0iSS8zNTUvZ2FpYWRyMyIpCiAgICAgICAgZ2FpYSA9IHJlc1swXQogICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBleGM6CiAgICAgICAgcHJpbnQoZiJbd2NzXSBHYWlhIHVuYXZhaWxhYmxlICh7ZXhjfSkgLSB0aGUgcGxhdGUgc29sdXRpb24gaXMgbm90IGNoZWNrZWQiKQogICAgICAgIHJldHVybiB3Y3MsICJub3QgY2hlY2tlZCAoR2FpYSB1bmF2YWlsYWJsZSkiCiAgICBnYWlhLnNvcnQoIkdtYWciKQogICAgZ3NreSA9IFNreUNvb3JkKG5wLmFzYXJyYXkoZ2FpYVsiUkFfSUNSUyJdLCBmbG9hdCkgKiB1LmRlZywKICAgICAgICAgICAgICAgICAgICBucC5hc2FycmF5KGdhaWFbIkRFX0lDUlMiXSwgZmxvYXQpICogdS5kZWcpCgogICAgZGVmIGZyYWN0aW9uKHcsIG49MTAwKToKICAgICAgICByYSwgZGVjID0gdy5hbGxfcGl4MndvcmxkKHh5X2JyaWdodFs6biwgMF0sIHh5X2JyaWdodFs6biwgMV0sIDApCiAgICAgICAgX2ksIHNlcCwgXyA9IFNreUNvb3JkKHJhICogdS5kZWcsIGRlYyAqIHUuZGVnKS5tYXRjaF90b19jYXRhbG9nX3NreShnc2t5KQogICAgICAgIG9rID0gc2VwLmFyY3NlYyA8IENBTElCX01BVENIX1JBRElVUwogICAgICAgIHJldHVybiBvay5tZWFuKCksIChucC5tZWRpYW4oc2VwLmFyY3NlY1tva10pIGlmIG9rLmFueSgpIGVsc2UgbnAubmFuKQoKICAgIGYwLCBzMCA9IGZyYWN0aW9uKHdjcykKICAgIHByaW50KGYiW3djc10gezEwMCAqIGYwOi4wZn0lIG9mIHRoZSAxMDAgYnJpZ2h0ZXN0IHNvdXJjZXMgaGF2ZSBhIEdhaWEgc3RhciAiCiAgICAgICAgICBmIndpdGhpbiB7Q0FMSUJfTUFUQ0hfUkFESVVTOmd9IGFyY3NlYyAobWVkaWFuIHtzMDouMmZ9IGFyY3NlYykiLCBmbHVzaD1UcnVlKQogICAgaWYgZjAgPj0gV0NTX0NIRUNLX01JTl9GUkFDVElPTjoKICAgICAgICByZXR1cm4gd2NzLCBmImNoZWNrZWQgYWdhaW5zdCBHYWlhOiB7MTAwICogZjA6LjBmfSUgb2YgYnJpZ2h0IHNvdXJjZXMgbWF0Y2giCgogICAgcHJpbnQoIlt3Y3NdIHRoZSBwbGF0ZSBzb2x1dGlvbiBpbiB0aGUgaGVhZGVyIGRvZXMgbm90IGZpdCB0aGlzIGltYWdlIC0gIgogICAgICAgICAgInJlcGFpcmluZyBpdCBmcm9tIHRoZSBzdGFyIHBhdHRlcm4iLCBmbHVzaD1UcnVlKQogICAgZ3gsIGd5ID0gd2NzLmFsbF93b3JsZDJwaXgoZ3NreS5yYS5kZWcsIGdza3kuZGVjLmRlZywgMCkKICAgIG5lYXIgPSAoZ3ggPiAtMC4wNSAqIG54KSAmIChneCA8IDEuMDUgKiBueCkgJiAoZ3kgPiAtMC4wNSAqIG55KSAmIChneSA8IDEuMDUgKiBueSkKICAgIHNyY19hbGwgPSBucC5jb2x1bW5fc3RhY2soW2d4LCBneV0pCiAgICBkZXRfdHJlZSA9IGNLRFRyZWUoeHlfYnJpZ2h0Wzo2MDBdKQogICAgbWF4X29mZiA9IDAuMjUgKiBtYXgobngsIG55KQogICAgYmVzdCA9IE5vbmUKICAgIGZvciBwYXJpdHkgaW4gKDEsIC0xKToKICAgICAgICBzcmMgPSBzcmNfYWxsLmNvcHkoKQogICAgICAgIGlmIHBhcml0eSA8IDA6CiAgICAgICAgICAgIHNyY1s6LCAwXSA9IChueCAtIDEpIC0gc3JjWzosIDBdCiAgICAgICAgY2FuZCA9IHNyY1tuZWFyXVs6NjAwXQogICAgICAgIGR4eSA9IFtdCiAgICAgICAgZm9yIGssIG5iIGluIGVudW1lcmF0ZShkZXRfdHJlZS5xdWVyeV9iYWxsX3BvaW50KGNhbmQsIHI9bWF4X29mZikpOgogICAgICAgICAgICBpZiBuYjoKICAgICAgICAgICAgICAgIGR4eS5hcHBlbmQoeHlfYnJpZ2h0Wzo2MDBdW25iXSAtIGNhbmRba10pCiAgICAgICAgaWYgZHh5OgogICAgICAgICAgICBkeHkgPSBucC52c3RhY2soZHh5KQogICAgICAgICAgICBiaW5zID0gbnAuYXJhbmdlKC1tYXhfb2ZmLCBtYXhfb2ZmICsgNCwgNC4wKQogICAgICAgICAgICBILCBleCwgZXkgPSBucC5oaXN0b2dyYW0yZChkeHlbOiwgMF0sIGR4eVs6LCAxXSwgYmlucz1bYmlucywgYmluc10pCiAgICAgICAgICAgIGksIGogPSBucC51bnJhdmVsX2luZGV4KG5wLmFyZ21heChIKSwgSC5zaGFwZSkKICAgICAgICAgICAgcGVhayA9IEhbaSwgal0KICAgICAgICAgICAgaWYgcGVhayA+PSAxMCBhbmQgcGVhayA+IDggKiBtYXgobnAubWVkaWFuKEhbSCA+IDBdKSwgMSk6CiAgICAgICAgICAgICAgICBzaGlmdCA9IG5wLmFycmF5KFswLjUgKiAoZXhbaV0gKyBleFtpICsgMV0pLCAwLjUgKiAoZXlbal0gKyBleVtqICsgMV0pXSkKICAgICAgICAgICAgICAgIFQgPSBsYW1iZGEgcCwgc189c2hpZnQ6IHAgKyBzXwogICAgICAgICAgICAgICAgaWYgYmVzdCBpcyBOb25lIG9yIHBlYWsgPiBiZXN0WzJdOgogICAgICAgICAgICAgICAgICAgIGJlc3QgPSAocGFyaXR5LCBULCBpbnQocGVhaykpCiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgIGZvciBuX2NhdCwgbl9kZXQgaW4gKCgxNTAsIDE1MCksICgyMDAsIDIwMCksICgzMDAsIDMwMCkpOgogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBULCAocywgX3QpID0gYWEuZmluZF90cmFuc2Zvcm0oY2FuZFs6bl9jYXRdLCB4eV9icmlnaHRbOm5fZGV0XSkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgIGlmIGFicyhULnNjYWxlIC0gMS4wKSA+IDAuMToKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgIGlmIGJlc3QgaXMgTm9uZSBvciBsZW4ocykgPiBiZXN0WzJdOgogICAgICAgICAgICAgICAgYmVzdCA9IChwYXJpdHksIFQsIGxlbihzKSkKICAgICAgICAgICAgYnJlYWsKICAgIGlmIGJlc3QgaXMgTm9uZToKICAgICAgICBzdG9wKCJQTEFURSBTT0xVVElPTjogdGhlIGhlYWRlcidzIHNvbHV0aW9uIGRvZXMgbm90IGZpdCB0aGUgViBpbWFnZSBhbmQgIgogICAgICAgICAgICAgImNvdWxkIG5vdCBiZSByZXBhaXJlZC4gUGxhdGUtc29sdmUgdGhlIFYgZnJhbWUgYWdhaW4gKGZvciBleGFtcGxlICIKICAgICAgICAgICAgICJ3aXRoIGFzdHJvbWV0cnkubmV0KSBhbmQgcmVydW4uIikKICAgIHBhcml0eSwgVCwgbmN0bCA9IGJlc3QKICAgIHNyYyA9IHNyY19hbGwuY29weSgpCiAgICBpZiBwYXJpdHkgPCAwOgogICAgICAgIHNyY1s6LCAwXSA9IChueCAtIDEpIC0gc3JjWzosIDBdCiAgICBndWVzcyA9IFQoc3JjKQogICAgdHJlZSA9IGNLRFRyZWUoYWxsX3h5KQogICAgbmV3ID0gd2NzCiAgICB0b2wgPSA1LjAKICAgIG9rID0gbnAuemVyb3MobGVuKGd1ZXNzKSwgYm9vbCkKICAgIGZvciBfIGluIHJhbmdlKDMpOgogICAgICAgIGQsIGogPSB0cmVlLnF1ZXJ5KGd1ZXNzLCBkaXN0YW5jZV91cHBlcl9ib3VuZD10b2wpCiAgICAgICAgb2sgPSBucC5pc2Zpbml0ZShkKQogICAgICAgIGlmIG9rLnN1bSgpIDwgMTA6CiAgICAgICAgICAgIGJyZWFrCiAgICAgICAgcHggPSBhbGxfeHlbaltva11dCiAgICAgICAgbmV3ID0gZml0X3djc19mcm9tX3BvaW50cygocHhbOiwgMF0sIHB4WzosIDFdKSwgZ3NreVtva10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBwcm9qX3BvaW50PSJjZW50ZXIiLCBwcm9qZWN0aW9uPSJUQU4iKQogICAgICAgIGd4MiwgZ3kyID0gbmV3LmFsbF93b3JsZDJwaXgoZ3NreS5yYS5kZWcsIGdza3kuZGVjLmRlZywgMCkKICAgICAgICBndWVzcyA9IG5wLmNvbHVtbl9zdGFjayhbZ3gyLCBneTJdKQogICAgICAgIHRvbCA9IFhZX01BVENIX1RPTEVSQU5DRQogICAgZjEsIHMxID0gZnJhY3Rpb24obmV3KQogICAgc2hpZnQgPSBucC5oeXBvdCgqKG5wLmFycmF5KG5ldy5hbGxfd29ybGQycGl4KGNlbnRyZS5yYS5kZWcsIGNlbnRyZS5kZWMuZGVnLCAwKSkKICAgICAgICAgICAgICAgICAgICAgICAtIG5wLmFycmF5KFtueCAvIDIsIG55IC8gMl0pKSkKICAgIHByaW50KGYiW3djc10gcmVwYWlyZWQgKHsnbWlycm9yZWQsICcgaWYgcGFyaXR5IDwgMCBlbHNlICcnfXtuY3RsfSBwYXR0ZXJuICIKICAgICAgICAgIGYic3RhcnMsIHtpbnQob2suc3VtKCkpfSBpbiB0aGUgZml0KTogdGhlIGZpZWxkIGNlbnRyZSBtb3ZlZCB7c2hpZnQ6LjBmfSBweDsgIgogICAgICAgICAgZiJub3cgezEwMCAqIGYxOi4wZn0lIG9mIGJyaWdodCBzb3VyY2VzIG1hdGNoIEdhaWEgKG1lZGlhbiB7czE6LjJmfSBhcmNzZWMpIiwKICAgICAgICAgIGZsdXNoPVRydWUpCiAgICBpZiBmMSA8IFdDU19DSEVDS19NSU5fRlJBQ1RJT046CiAgICAgICAgc3RvcCgiUExBVEUgU09MVVRJT046IHRoZSByZXBhaXJlZCBzb2x1dGlvbiBzdGlsbCBkb2VzIG5vdCBmaXQuICIKICAgICAgICAgICAgICJQbGF0ZS1zb2x2ZSB0aGUgViBmcmFtZSBhZ2FpbiBhbmQgcmVydW4uIikKICAgIHJldHVybiBuZXcsIChmIlJFUEFJUkVEIGZyb20gR2FpYTogaGVhZGVyIHNvbHV0aW9uIG1hdGNoZWQgezEwMCAqIGYwOi4wZn0lIG9mICIKICAgICAgICAgICAgICAgICBmImJyaWdodCBzb3VyY2VzLCByZXBhaXJlZCB7MTAwICogZjE6LjBmfSUsIGNlbnRyZSBtb3ZlZCB7c2hpZnQ6LjBmfSBweCIpCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tIENhbGlicmF0aW9uIC0tLS0tLS0tLS0tLS0tLS0tLS0KCmRlZiBxdWVyeV9jYWxpYnJhdGlvbl9jYXRhbG9ndWUod2hpY2gsIGNlbnRyZSwgcmFkaXVzX2RlZyk6CiAgICAiIiJSZWZlcmVuY2Ugc3RhcnMgd2l0aCBKb2huc29uIEIgYW5kIFY6IFJBLCBERUMsIE1hZ19CLCBNYWdfViwgZV9CLCBlX1YuCgogICAgQVBBU1MgRFI5IChJSS8zMzYvYXBhc3M5KSBpcyB0aGUgZGVmYXVsdC4gR2FpYSBEUjMgc3ludGhldGljIHBob3RvbWV0cnkKICAgIChJLzM2MC9zeW50cGhvdCkgaXMgY29tcHV0ZWQgZnJvbSBHYWlhJ3Mgb3duIGxvdy1yZXNvbHV0aW9uIHNwZWN0cmEgYW5kCiAgICBzdGFuZGFyZGlzZWQgdG8gdGhlIEpvaG5zb24tS3Jvbi1Db3VzaW5zIHN5c3RlbTogZGVlcGVyLCB1bmlmb3JtIGFjcm9zcwogICAgdGhlIHNreSwgYW5kIHJlc29sdmVkIGRvd24gdG8gYWJvdXQgYW4gYXJjc2Vjb25kLiBPbmx5IGVudHJpZXMgR2FpYSBmbGFncwogICAgYXMgZ29vZCBhcmUgdXNlZC4KCiAgICByb3dfbGltaXQgaGFzIHRvIGJlIHBhc3NlZCB0byB0aGUgY29uc3RydWN0b3I6IHNldCBvbiB0aGUgY2xhc3MgYWZ0ZXJ3YXJkcwogICAgaXQgbGVmdCBhc3Ryb3F1ZXJ5J3MgZGVmYXVsdCBvZiA1MCByb3dzLCBzaWxlbnRseS4KICAgICIiIgogICAgY29vcmQgPSBTa3lDb29yZChjZW50cmUucmEsIGNlbnRyZS5kZWMpCiAgICBlbXB0eSA9IHBkLkRhdGFGcmFtZShjb2x1bW5zPVsiUkEiLCAiREVDIiwgIk1hZ19CIiwgIk1hZ19WIiwgImVfQiIsICJlX1YiXSkKICAgIGlmIHdoaWNoID09ICJBUEFTUyI6CiAgICAgICAgY29scyA9IFsiUkFKMjAwMCIsICJERUoyMDAwIiwgIkJtYWciLCAiVm1hZyIsICJlX0JtYWciLCAiZV9WbWFnIl0KICAgICAgICByZXMgPSBWaXppZXIoY29sdW1ucz1jb2xzLCByb3dfbGltaXQ9LTEpLnF1ZXJ5X3JlZ2lvbigKICAgICAgICAgICAgY29vcmQsIHJhZGl1cz1yYWRpdXNfZGVnICogdS5kZWcsIGNhdGFsb2c9IklJLzMzNi9hcGFzczkiKQogICAgICAgIGlmIGxlbihyZXMpID09IDA6CiAgICAgICAgICAgIHJldHVybiBlbXB0eQogICAgICAgIHQgPSByZXNbMF0udG9fcGFuZGFzKCkKICAgICAgICBvdXQgPSBwZC5EYXRhRnJhbWUoeyJSQSI6IHRbIlJBSjIwMDAiXSwgIkRFQyI6IHRbIkRFSjIwMDAiXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICJNYWdfQiI6IHRbIkJtYWciXSwgIk1hZ19WIjogdFsiVm1hZyJdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgImVfQiI6IHRbImVfQm1hZyJdLCAiZV9WIjogdFsiZV9WbWFnIl19KQogICAgZWxzZToKICAgICAgICBjb2xzID0gWyJTb3VyY2UiLCAiUkFfSUNSUyIsICJERV9JQ1JTIiwgIkJtYWciLCAiVm1hZyIsICJGQiIsICJlX0ZCIiwKICAgICAgICAgICAgICAgICJGViIsICJlX0ZWIiwgIkJGbGFnIiwgIlZGbGFnIl0KICAgICAgICByZXMgPSBWaXppZXIoY29sdW1ucz1jb2xzLCByb3dfbGltaXQ9LTEpLnF1ZXJ5X3JlZ2lvbigKICAgICAgICAgICAgY29vcmQsIHJhZGl1cz1yYWRpdXNfZGVnICogdS5kZWcsIGNhdGFsb2c9IkkvMzYwL3N5bnRwaG90IikKICAgICAgICBpZiBsZW4ocmVzKSA9PSAwOgogICAgICAgICAgICByZXR1cm4gZW1wdHkKICAgICAgICB0ID0gcmVzWzBdLnRvX3BhbmRhcygpCiAgICAgICAgdCA9IHRbKHRbIkJGbGFnIl0gPT0gMSkgJiAodFsiVkZsYWciXSA9PSAxKV0KICAgICAgICBvdXQgPSBwZC5EYXRhRnJhbWUoeyJSQSI6IHRbIlJBX0lDUlMiXSwgIkRFQyI6IHRbIkRFX0lDUlMiXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICJNYWdfQiI6IHRbIkJtYWciXSwgIk1hZ19WIjogdFsiVm1hZyJdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgImVfQiI6IDEuMDg1NyAqIHRbImVfRkIiXSAvIHRbIkZCIl0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAiZV9WIjogMS4wODU3ICogdFsiZV9GViJdIC8gdFsiRlYiXX0pCiAgICByZXR1cm4gb3V0LnJlc2V0X2luZGV4KGRyb3A9VHJ1ZSkKCgpkZWYgbWF0Y2hfY2F0YWxvZ3VlKGRmLCB3Y3MsIGNhdCwgc2hhcGUpOgogICAgIiIiRWFjaCBjYXRhbG9ndWUgc3RhciBtYXRjaGVkIHRvIHRoZSBjbG9zZXN0IHNvdXJjZSwgb25jZS4KCiAgICBBIG1hdGNoIG11c3QgYmUgd2l0aGluIFhZX01BVENIX1RPTEVSQU5DRSBvbiB0aGUgViBncmlkIGFuZAogICAgQ0FMSUJfTUFUQ0hfUkFESVVTIG9uIHRoZSBza3kuIEVhY2ggc291cmNlIGlzIHVzZWQgZm9yIGF0IG1vc3Qgb25lCiAgICBjYXRhbG9ndWUgc3RhciAodGhlIGNsb3Nlc3QpIC0gYSBicmlnaHQgc3RhciBmb3VuZCBzZXZlcmFsIHRpbWVzIG92ZXIKICAgIHVzZWQgdG8gYmUgY291bnRlZCBvbmNlIHBlciBkZXRlY3Rpb24sIGFuZCBvbmUgc3RhciB0aGVuIG91dHdlaWdoZWQgdGhlCiAgICByZXN0IG9mIHRoZSBjYWxpYnJhdGlvbi4KICAgICIiIgogICAgY2F0ID0gY2F0W25wLmlzZmluaXRlKGNhdFsiTWFnX0IiXSkgJiBucC5pc2Zpbml0ZShjYXRbIk1hZ19WIl0pXS5yZXNldF9pbmRleChkcm9wPVRydWUpCiAgICBpZiBub3QgbGVuKGNhdCkgb3Igbm90IGxlbihkZik6CiAgICAgICAgcmV0dXJuIHBkLkRhdGFGcmFtZSgpCiAgICBjeCwgY3kgPSB3Y3MuYWxsX3dvcmxkMnBpeChjYXRbIlJBIl0udmFsdWVzLCBjYXRbIkRFQyJdLnZhbHVlcywgMCkKICAgIG55LCBueCA9IHNoYXBlCiAgICBpbnNpZGUgPSAoY3ggPiAwKSAmIChjeCA8IG54KSAmIChjeSA+IDApICYgKGN5IDwgbnkpCiAgICBjYXQgPSBjYXRbaW5zaWRlXS5yZXNldF9pbmRleChkcm9wPVRydWUpCiAgICBjeCwgY3kgPSBjeFtpbnNpZGVdLCBjeVtpbnNpZGVdCiAgICBpZiBub3QgbGVuKGNhdCk6CiAgICAgICAgcmV0dXJuIHBkLkRhdGFGcmFtZSgpCiAgICBkaXN0LCBpZHggPSBjS0RUcmVlKGRmW1siWF9WIiwgIllfViJdXS52YWx1ZXMpLnF1ZXJ5KAogICAgICAgIG5wLmNvbHVtbl9zdGFjayhbY3gsIGN5XSksIGRpc3RhbmNlX3VwcGVyX2JvdW5kPVhZX01BVENIX1RPTEVSQU5DRSkKICAgIGJlc3QgPSB7fQogICAgZm9yIGkgaW4gcmFuZ2UobGVuKGNhdCkpOgogICAgICAgIGlmIG5vdCBucC5pc2Zpbml0ZShkaXN0W2ldKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICBqID0gaW50KGlkeFtpXSkKICAgICAgICBpZiBqIG5vdCBpbiBiZXN0IG9yIGRpc3RbaV0gPCBiZXN0W2pdWzFdOgogICAgICAgICAgICBiZXN0W2pdID0gKGksIGRpc3RbaV0pCiAgICByb3dzID0gW10KICAgIGZvciBqLCAoaSwgZCkgaW4gYmVzdC5pdGVtcygpOgogICAgICAgIHNyYyA9IGRmLmlsb2Nbal0KICAgICAgICBzZXAgPSBTa3lDb29yZChzcmNbIlJBIl0gKiB1LmRlZywgc3JjWyJERUMiXSAqIHUuZGVnKS5zZXBhcmF0aW9uKAogICAgICAgICAgICBTa3lDb29yZChjYXQubG9jW2ksICJSQSJdICogdS5kZWcsIGNhdC5sb2NbaSwgIkRFQyJdICogdS5kZWcpKS5hcmNzZWMKICAgICAgICBpZiBzZXAgPiBDQUxJQl9NQVRDSF9SQURJVVM6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgciA9IHNyYy50b19kaWN0KCkKICAgICAgICByLnVwZGF0ZSh7InJvdyI6IGosICJDYXRfUkEiOiBjYXQubG9jW2ksICJSQSJdLCAiQ2F0X0RFQyI6IGNhdC5sb2NbaSwgIkRFQyJdLAogICAgICAgICAgICAgICAgICAiQ2F0X0IiOiBjYXQubG9jW2ksICJNYWdfQiJdLCAiQ2F0X1YiOiBjYXQubG9jW2ksICJNYWdfViJdLAogICAgICAgICAgICAgICAgICAiQ2F0X2VfQiI6IGNhdC5sb2NbaSwgImVfQiJdLCAiQ2F0X2VfViI6IGNhdC5sb2NbaSwgImVfViJdLAogICAgICAgICAgICAgICAgICAiU2VwX2FyY3NlYyI6IHNlcCwgIlNlcF9weCI6IGR9KQogICAgICAgIHJvd3MuYXBwZW5kKHIpCiAgICByZXR1cm4gcGQuRGF0YUZyYW1lKHJvd3MpCgoKZGVmIGNsaXBwZWRfbGluZSh4LCB5LCBuc2lnPUNBTElCX0NMSVBfU0lHTUEsIG1heGl0ZXI9MTApOgogICAgIiIiU3RyYWlnaHQtbGluZSBmaXQgeSA9IGludGVyY2VwdCArIHNsb3BlIHgsIHdpdGggcG9pbnRzIGZ1cnRoZXIgdGhhbgogICAgbnNpZyByb2J1c3Qgc2lnbWFzIGZyb20gdGhlIGxpbmUgbGVmdCBvdXQsIHJlcGVhdGVkIHVudGlsIG5vdGhpbmcgY2hhbmdlcy4KICAgIFJldHVybnMgKGludGVyY2VwdCwgc2xvcGUsIG1hc2tfdXNlZCwgcm1zX29mX3VzZWQpLiIiIgogICAgeCA9IG5wLmFzYXJyYXkoeCwgZmxvYXQpCiAgICB5ID0gbnAuYXNhcnJheSh5LCBmbG9hdCkKICAgIHVzZSA9IG5wLmlzZmluaXRlKHgpICYgbnAuaXNmaW5pdGUoeSkKICAgIGZvciBfIGluIHJhbmdlKG1heGl0ZXIpOgogICAgICAgIGlmIHVzZS5zdW0oKSA8IDM6CiAgICAgICAgICAgIGJyZWFrCiAgICAgICAgc2xvcGUsIGludGVyY2VwdCA9IG5wLnBvbHlmaXQoeFt1c2VdLCB5W3VzZV0sIDEpCiAgICAgICAgcmVzaWQgPSB5IC0gKGludGVyY2VwdCArIHNsb3BlICogeCkKICAgICAgICBzaWcgPSBtYXgoMS40ODI2ICogbnAubWVkaWFuKG5wLmFicyhyZXNpZFt1c2VdIC0gbnAubWVkaWFuKHJlc2lkW3VzZV0pKSksIDFlLTMpCiAgICAgICAgbmV3ID0gbnAuaXNmaW5pdGUocmVzaWQpICYgKG5wLmFicyhyZXNpZCkgPCBuc2lnICogc2lnKQogICAgICAgIGlmIG5wLmFycmF5X2VxdWFsKG5ldywgdXNlKToKICAgICAgICAgICAgYnJlYWsKICAgICAgICB1c2UgPSBuZXcKICAgIHNsb3BlLCBpbnRlcmNlcHQgPSBucC5wb2x5Zml0KHhbdXNlXSwgeVt1c2VdLCAxKQogICAgcmVzaWQgPSB5IC0gKGludGVyY2VwdCArIHNsb3BlICogeCkKICAgIHJldHVybiBpbnRlcmNlcHQsIHNsb3BlLCB1c2UsIGZsb2F0KG5wLnN0ZChyZXNpZFt1c2VdKSkKCgpkZWYgY2FsaWJyYXRpb25fY2FuZGlkYXRlcyhkZiwgd2NzLCBzaGFwZSwgd2hpY2gsIGNlbnRyZV9weD1Ob25lLCByMjVfcHg9Tm9uZSk6CiAgICAiIiJDYXRhbG9ndWUgc3RhcnMgbWF0Y2hlZCB0byBzb3VyY2VzIHRoYXQgY2FuIHNldCBhIGNhbGlicmF0aW9uOgogICAgdW5zYXR1cmF0ZWQgaW4gYm90aCBiYW5kcywgd2l0aCBubyBvdGhlciBzb3VyY2UgaW5zaWRlIHRoZSBhcGVydHVyZSwgd2VsbAogICAgbWVhc3VyZWQsIHdpdGggZmluaXRlIGFuZCB3ZWxsLW1lYXN1cmVkIGNhdGFsb2d1ZSBtYWduaXR1ZGVzLiBXaGVyZSBlbm91Z2gKICAgIG9mIHRoZW0gbGllIGJleW9uZCBSMjUgb25seSB0aG9zZSBhcmUgdXNlZDogdGhlIHNreSByaW5nIG9mIGEgc3RhciBvbiB0aGUKICAgIGRpc2MgaXMgbWVhc3VyaW5nIHRoZSBkaXNjLiIiIgogICAgY2VudHJlLCByYWRpdXMgPSBmaWVsZF9jZW50cmVfYW5kX3JhZGl1cyh3Y3MsIHNoYXBlKQogICAgbmFtZSA9ICJBUEFTUyBEUjkiIGlmIHdoaWNoID09ICJBUEFTUyIgZWxzZSAiR2FpYSBEUjMgc3ludGhldGljIHBob3RvbWV0cnkiCiAgICBwcmludChmIltjYWxpYl0gcXVlcnlpbmcge25hbWV9IHdpdGhpbiB7cmFkaXVzICogNjA6LjBmfSBhcmNtaW4gb2YgdGhlIGZpZWxkICIKICAgICAgICAgIGYiY2VudHJlLi4uIiwgZmx1c2g9VHJ1ZSkKICAgIGNhdCA9IHF1ZXJ5X2NhbGlicmF0aW9uX2NhdGFsb2d1ZSh3aGljaCwgY2VudHJlLCByYWRpdXMpCiAgICBwcmludChmIiAgIHtsZW4oY2F0KTosfSBjYXRhbG9ndWUgc3RhcnMgaW4gdGhlIHJlZ2lvbiIsIGZsdXNoPVRydWUpCiAgICBtID0gbWF0Y2hfY2F0YWxvZ3VlKGRmLCB3Y3MsIGNhdCwgc2hhcGUpCiAgICBzdGF0cyA9IHsiY2F0YWxvZ3VlIjogbGVuKGNhdCksICJtYXRjaGVkIjogbGVuKG0pfQogICAgaWYgbm90IGxlbihtKToKICAgICAgICByZXR1cm4gbSwgc3RhdHMKICAgIGtlZXAgPSB+bVsiU2F0dXJhdGVkIl0uYXN0eXBlKGJvb2wpCiAgICBjcm93ZGVkID0gbVsiTmVhcmVzdF9weCJdIDwgbVsiQXBlcnR1cmVfUmFkaXVzIl0KICAgIGdvb2RfZXJyID0gKChtWyJNYWdfaW5zdF9CX2VyciJdIDwgQ0FMSUJfTUFYX0lOU1RfRVJSKQogICAgICAgICAgICAgICAgJiAobVsiTWFnX2luc3RfVl9lcnIiXSA8IENBTElCX01BWF9JTlNUX0VSUikpCiAgICBjYXRfb2sgPSB+KChtWyJDYXRfZV9CIl0gPiBDQUxJQl9NQVhfQ0FUX0VSUikgfCAobVsiQ2F0X2VfViJdID4gQ0FMSUJfTUFYX0NBVF9FUlIpKQogICAga2VlcCAmPSB+Y3Jvd2RlZCAmIGdvb2RfZXJyICYgY2F0X29rCiAgICBzdGF0cy51cGRhdGUoeyJzYXR1cmF0ZWQiOiBpbnQobVsiU2F0dXJhdGVkIl0uYXN0eXBlKGJvb2wpLnN1bSgpKSwKICAgICAgICAgICAgICAgICAgImNyb3dkZWQiOiBpbnQoY3Jvd2RlZC5zdW0oKSksICJub2lzeSI6IGludCgofmdvb2RfZXJyKS5zdW0oKSksCiAgICAgICAgICAgICAgICAgICJiYWRfY2F0YWxvZ3VlX2Vycm9yIjogaW50KCh+Y2F0X29rKS5zdW0oKSksCiAgICAgICAgICAgICAgICAgICJtZWRpYW5fc2VwX2FyY3NlYyI6IGZsb2F0KG5wLm1lZGlhbihtWyJTZXBfYXJjc2VjIl0pKX0pCiAgICBpZiBjZW50cmVfcHggaXMgbm90IE5vbmUgYW5kIHIyNV9weDoKICAgICAgICBvdXRzaWRlID0gbnAuaHlwb3QobVsiWF9WIl0gLSBjZW50cmVfcHhbMF0sIG1bIllfViJdIC0gY2VudHJlX3B4WzFdKSA+IHIyNV9weAogICAgICAgIGlmIGludCgoa2VlcCAmIG91dHNpZGUpLnN1bSgpKSA+PSBDQUxJQl9NSU5fT1VUU0lERToKICAgICAgICAgICAgc3RhdHNbImluc2lkZV9nYWxheHlfbGVmdF9vdXQiXSA9IGludCgoa2VlcCAmIH5vdXRzaWRlKS5zdW0oKSkKICAgICAgICAgICAga2VlcCAmPSBvdXRzaWRlCiAgICBtWyJDYW5kaWRhdGUiXSA9IGtlZXAudmFsdWVzCiAgICByZXR1cm4gbSwgc3RhdHMKCgpkZWYgZml0X2NhbGlicmF0aW9uKGNhbmQpOgogICAgIiIiWmVybyBwb2ludHMgYW5kIGNvbG91ciB0ZXJtcyBmcm9tIHRoZSBjYW5kaWRhdGUgc3RhcnMuCgogICAgICAgIChiLXYpX2luc3QgICAgICA9IGFscGhhICsgYmV0YSAqIChCLVYpX2NhdCAgICAgICh0aGUgY29sb3VyIHRyYW5zZm9ybWF0aW9uKQogICAgICAgICAgICBzbyAoQi1WKSAgICA9IGEgKyBjICogKGItdilfaW5zdCAgIHdpdGggYyA9IDEvYmV0YSwgYSA9IC1hbHBoYS9iZXRhCiAgICAgICAgVl9jYXQgLSB2X2luc3QgID0gWlBfViArIGtfViAqIChCLVYpX2NhdAogICAgICAgIEJfY2F0IC0gYl9pbnN0ICA9IFpQX0IgKyBrX0IgKiAoQi1WKV9jYXQgICAoc2hvd24sIGFuZCBhIGNyb3NzLWNoZWNrKQoKICAgIFRoZSBjb2xvdXIgaXMgZml0dGVkIHRoaXMgd2F5IHJvdW5kIGJlY2F1c2UgYSBzdHJhaWdodC1saW5lIGZpdCB0cmVhdHMgeAogICAgYXMgZXhhY3Q6IHRoZSBjYXRhbG9ndWUgY29sb3VyIGlzIHRoZSBwcmVjaXNlIG9uZS4gRXZlcnkgZml0IGlzCiAgICBzaWdtYS1jbGlwcGVkOyBhIHN0YXIgY2xpcHBlZCBmcm9tIHRoZSBjb2xvdXIgb3IgdGhlIFYgZml0IGlzIG1hcmtlZCBhcwogICAgbm90IHVzZWQuIFRoaXMgcmVwbGFjZXMgb25lIHplcm8gcG9pbnQgcGVyIGJhbmQgZnJvbSB0aGUgYnJpZ2h0ZXN0IHN0YXJzLAogICAgd2hpY2ggbGVmdCB0aGUgY29sb3VyIG9uIHRoZSBjYW1lcmEncyBvd24gc2NhbGUgLSB3aXRoIGZpbHRlcnMgdGhhdCBhcmUKICAgIG5vdCBleGFjdGx5IEpvaG5zb24gQiBhbmQgViwgZXZlcnkgQi1WIHdhcyBvZmYgYnkgYSBmcmFjdGlvbiBvZiBpdHNlbGYuCiAgICAiIiIKICAgIGMgPSBjYW5kW2NhbmRbIkNhbmRpZGF0ZSJdXS5jb3B5KCkKICAgIGJ2X2luc3QgPSBjWyJNYWdfaW5zdF9CIl0gLSBjWyJNYWdfaW5zdF9WIl0KICAgIGJ2X2NhdCA9IGNbIkNhdF9CIl0gLSBjWyJDYXRfViJdCiAgICBhbHBoYSwgYmV0YSwgdXNlX2MsIF8gPSBjbGlwcGVkX2xpbmUoYnZfY2F0LCBidl9pbnN0KQogICAgY3Rlcm0sIGEgPSAxLjAgLyBiZXRhLCAtYWxwaGEgLyBiZXRhCiAgICBybXNfYyA9IGZsb2F0KG5wLnN0ZCgoYSArIGN0ZXJtICogYnZfaW5zdCAtIGJ2X2NhdClbdXNlX2NdKSkKICAgIHpwdiwga3YsIHVzZV92LCBybXNfdiA9IGNsaXBwZWRfbGluZShidl9jYXQsIGNbIkNhdF9WIl0gLSBjWyJNYWdfaW5zdF9WIl0pCiAgICB6cGIsIGtiLCB1c2VfYiwgcm1zX2IgPSBjbGlwcGVkX2xpbmUoYnZfY2F0LCBjWyJDYXRfQiJdIC0gY1siTWFnX2luc3RfQiJdKQogICAgdXNlZCA9IHVzZV9jICYgdXNlX3YKICAgIGNvcnIgPSBmbG9hdChucC5jb3JyY29lZihidl9pbnN0LCBidl9jYXQpWzAsIDFdKSBpZiBsZW4oYykgPiAyIGVsc2UgbnAubmFuCiAgICByZXR1cm4geyJhIjogYSwgImMiOiBjdGVybSwgIlpQX1YiOiB6cHYsICJrX1YiOiBrdiwgIlpQX0IiOiB6cGIsICJrX0IiOiBrYiwKICAgICAgICAgICAgInJtc19jb2xvdXJfZml0Ijogcm1zX2MsICJybXNfVl9maXQiOiBybXNfdiwgInJtc19CX2ZpdCI6IHJtc19iLAogICAgICAgICAgICAiTl9jYW5kaWRhdGVzIjogbGVuKGMpLCAiTl91c2VkIjogaW50KHVzZWQuc3VtKCkpLAogICAgICAgICAgICAiY29ycmVsYXRpb24iOiBjb3JyLCAidXNlZF9pbmRleCI6IGMuaW5kZXhbdXNlZF19CgoKZGVmIGFwcGx5X2NhbGlicmF0aW9uKGRmLCBjYWwpOgogICAgIiIiQ2FsaWJyYXRlZCBCLCBWLCBCLVYgYW5kIHRoZWlyIGVycm9ycyBmb3IgZXZlcnkgc291cmNlIGluIGRmLiIiIgogICAgYiwgdiA9IGRmWyJNYWdfaW5zdF9CIl0udG9fbnVtcHkoZmxvYXQpLCBkZlsiTWFnX2luc3RfViJdLnRvX251bXB5KGZsb2F0KQogICAgZWIsIGV2ID0gZGZbIk1hZ19pbnN0X0JfZXJyIl0udG9fbnVtcHkoZmxvYXQpLCBkZlsiTWFnX2luc3RfVl9lcnIiXS50b19udW1weShmbG9hdCkKICAgIGEsIGMsIHpwLCBrID0gY2FsWyJhIl0sIGNhbFsiYyJdLCBjYWxbIlpQX1YiXSwgY2FsWyJrX1YiXQogICAgYnYgPSBhICsgYyAqIChiIC0gdikKICAgIFYgPSB2ICsgenAgKyBrICogYnYKICAgIEIgPSBWICsgYnYKICAgICMgViA9ICgxIC0gayBjKSB2ICsgayBjIGIgKyBjb25zdCA7IEIgPSAoMSAtICgxK2spIGMpIHYgKyAoMStrKSBjIGIgKyBjb25zdAogICAgb3V0ID0gZGYuY29weSgpCiAgICBvdXRbIk1hZ19CX29icyJdLCBvdXRbIk1hZ19WX29icyJdLCBvdXRbIkJWX29icyJdID0gQiwgViwgYnYKICAgIG91dFsiTWFnX0JfZXJyIl0gPSBucC5oeXBvdCgoMSAtICgxICsgaykgKiBjKSAqIGV2LCAoMSArIGspICogYyAqIGViKQogICAgb3V0WyJNYWdfVl9lcnIiXSA9IG5wLmh5cG90KCgxIC0gayAqIGMpICogZXYsIGsgKiBjICogZWIpCiAgICBvdXRbIkJWX2VyciJdID0gbnAuYWJzKGMpICogbnAuaHlwb3QoZWIsIGV2KQogICAgcmV0dXJuIG91dAoKCmRlZiBjYWxpYnJhdGUoZGYsIHdjcywgc2hhcGUsIHdoaWNoLCBjZW50cmVfcHgsIHIyNV9weCwgb3V0X2Rpciwgc2FmZSk6CiAgICAiIiJSdW4gdGhlIHdob2xlIGNhbGlicmF0aW9uLCB3aXRoIEdhaWEgYXMgdGhlIGZhbGxiYWNrIGNhdGFsb2d1ZSwgdGhlCiAgICBjaGVja3MgdGhhdCBzdG9wIGEgcnVuIHRoYXQgY2Fubm90IGJlIHJpZ2h0LCBhbmQgdGhlIGNhbGlicmF0aW9uIHRhYmxlCiAgICB3cml0dGVuIG91dC4gUmV0dXJucyAoY2FsLCBjYXRhbG9ndWUgdXNlZCwgc3RhdHMsIG1hdGNoZWQgdGFibGUpLiIiIgogICAgY2FuZCwgc3RhdHMgPSBjYWxpYnJhdGlvbl9jYW5kaWRhdGVzKGRmLCB3Y3MsIHNoYXBlLCB3aGljaCwgY2VudHJlX3B4LCByMjVfcHgpCiAgICBuX2NhbmQgPSBpbnQoY2FuZFsiQ2FuZGlkYXRlIl0uc3VtKCkpIGlmIGxlbihjYW5kKSBlbHNlIDAKICAgIGlmIG5fY2FuZCA8IENBTElCX01JTl9TVEFSUyBhbmQgd2hpY2ggPT0gIkFQQVNTIjoKICAgICAgICBwcmludChmIltjYWxpYl0gb25seSB7bl9jYW5kfSB1c2FibGUgQVBBU1Mgc3RhcnMgLSB0cnlpbmcgR2FpYSBEUjMgc3ludGhldGljICIKICAgICAgICAgICAgICBmInBob3RvbWV0cnkgaW5zdGVhZCIsIGZsdXNoPVRydWUpCiAgICAgICAgd2hpY2ggPSAiR0FJQSIKICAgICAgICBjYW5kLCBzdGF0cyA9IGNhbGlicmF0aW9uX2NhbmRpZGF0ZXMoZGYsIHdjcywgc2hhcGUsIHdoaWNoLCBjZW50cmVfcHgsIHIyNV9weCkKICAgICAgICBuX2NhbmQgPSBpbnQoY2FuZFsiQ2FuZGlkYXRlIl0uc3VtKCkpIGlmIGxlbihjYW5kKSBlbHNlIDAKICAgIGlmIG5fY2FuZCA8IENBTElCX01JTl9TVEFSUzoKICAgICAgICBzdG9wKGYiQ0FMSUJSQVRJT046IG9ubHkge25fY2FuZH0gdXNhYmxlIHJlZmVyZW5jZSBzdGFycyAoYXQgbGVhc3QgIgogICAgICAgICAgICAgZiJ7Q0FMSUJfTUlOX1NUQVJTfSBhcmUgbmVlZGVkKS4gTWF0Y2hlZCB7c3RhdHMuZ2V0KCdtYXRjaGVkJywgMCl9LCBvZiAiCiAgICAgICAgICAgICBmIndoaWNoIHNhdHVyYXRlZCB7c3RhdHMuZ2V0KCdzYXR1cmF0ZWQnLCAwKX0sIGNyb3dkZWQgIgogICAgICAgICAgICAgZiJ7c3RhdHMuZ2V0KCdjcm93ZGVkJywgMCl9LCB0b28gbm9pc3kge3N0YXRzLmdldCgnbm9pc3knLCAwKX0uIikKICAgIHByaW50KGYiW2NhbGliXSB7c3RhdHNbJ21hdGNoZWQnXX0gY2F0YWxvZ3VlIHN0YXJzIG1hdGNoZWQgKG1lZGlhbiBzZXBhcmF0aW9uICIKICAgICAgICAgIGYie3N0YXRzWydtZWRpYW5fc2VwX2FyY3NlYyddOi4yZn0gYXJjc2VjKTsgbGVmdCBvdXQ6ICIKICAgICAgICAgIGYie3N0YXRzWydzYXR1cmF0ZWQnXX0gc2F0dXJhdGVkLCB7c3RhdHNbJ2Nyb3dkZWQnXX0gY3Jvd2RlZCwgIgogICAgICAgICAgZiJ7c3RhdHNbJ25vaXN5J119IHRvbyBub2lzeSwge3N0YXRzWydiYWRfY2F0YWxvZ3VlX2Vycm9yJ119IHdpdGggbGFyZ2UgIgogICAgICAgICAgZiJjYXRhbG9ndWUgZXJyb3JzIgogICAgICAgICAgKyAoZiIsIHtzdGF0c1snaW5zaWRlX2dhbGF4eV9sZWZ0X291dCddfSBvbiB0aGUgZ2FsYXh5J3MgZGlzYyIKICAgICAgICAgICAgIGlmIHN0YXRzLmdldCgiaW5zaWRlX2dhbGF4eV9sZWZ0X291dCIpIGVsc2UgIiIpLCBmbHVzaD1UcnVlKQogICAgY2FsID0gZml0X2NhbGlicmF0aW9uKGNhbmQpCiAgICBpZiBjYWxbIk5fdXNlZCJdIDwgQ0FMSUJfTUlOX1NUQVJTOgogICAgICAgIHN0b3AoZiJDQUxJQlJBVElPTjogb25seSB7Y2FsWydOX3VzZWQnXX0gc3RhcnMgc3Vydml2ZWQgdGhlIG91dGxpZXIgIgogICAgICAgICAgICAgZiJyZWplY3Rpb24gKGF0IGxlYXN0IHtDQUxJQl9NSU5fU1RBUlN9IGFyZSBuZWVkZWQpLiIpCgogICAgIyBUaGUgY29sb3VyIGNoZWNrLiBUaGUgaW5zdHJ1bWVudGFsIGNvbG91ciBtdXN0IHJpc2Ugd2l0aCB0aGUgY2F0YWxvZ3VlCiAgICAjIGNvbG91cjsgaWYgaXQgZmFsbHMsIHRoZSBmcmFtZSBsYWJlbGxlZCBCIGlzIG5vdCB0aGUgYmx1ZSBvbmUuIE5vdGhpbmcKICAgICMgZWxzZSBpbiB0aGUgb3V0cHV0IHdvdWxkIHNob3cgaXQ6IHRoZSBkaWFncmFtIHdvdWxkIGxvb2sgbm9ybWFsLCB3aXRoCiAgICAjIGV2ZXJ5IGNvbG91ciByZXZlcnNlZC4KICAgIGlmIG5vdCAoY2FsWyJjb3JyZWxhdGlvbiJdID4gMCk6CiAgICAgICAgc3RvcChmIkNPTE9VUiBDSEVDSzogdGhlIGluc3RydW1lbnRhbCBjb2xvdXIgYi12IEZBTExTIGFzIHRoZSBjYXRhbG9ndWUgIgogICAgICAgICAgICAgZiJCLVYgcmlzZXMgKGNvcnJlbGF0aW9uIHtjYWxbJ2NvcnJlbGF0aW9uJ106Ky4yZn0gb3ZlciAiCiAgICAgICAgICAgICBmIntjYWxbJ05fY2FuZGlkYXRlcyddfSBzdGFycykuIFRoZSBmcmFtZSB1c2VkIGFzIEIgaXMgcHJvYmFibHkgdGhlICIKICAgICAgICAgICAgIGYiViAob3IgUikgZnJhbWUgYW5kIHRoZSBvdGhlciB3YXkgcm91bmQgLSBjaGVjayB0aGUgRklMVEVSIGtleXdvcmRzICIKICAgICAgICAgICAgIGYiYW5kIHRoZSBmaWxlIG5hbWVzLiIpCiAgICBpZiBjYWxbImNvcnJlbGF0aW9uIl0gPCAwLjU6CiAgICAgICAgbG91ZChmIldBUk5JTkc6IHRoZSBpbnN0cnVtZW50YWwgYW5kIGNhdGFsb2d1ZSBjb2xvdXJzIGFncmVlIG9ubHkgd2Vha2x5ICIKICAgICAgICAgICAgIGYiKGNvcnJlbGF0aW9uIHtjYWxbJ2NvcnJlbGF0aW9uJ106LjJmfSkuIFRoZSBjb2xvdXJzIGJlbG93IGFyZSAiCiAgICAgICAgICAgICBmInVuY2VydGFpbi4iKQogICAgcHJpbnQoZiJbY2FsaWJdIHtjYWxbJ05fdXNlZCddfSBvZiB7Y2FsWydOX2NhbmRpZGF0ZXMnXX0gc3RhcnMgdXNlZCAiCiAgICAgICAgICBmIih7J0FQQVNTIERSOScgaWYgd2hpY2ggPT0gJ0FQQVNTJyBlbHNlICdHYWlhIERSMyBzeW50aGV0aWMnfSk6IiwgZmx1c2g9VHJ1ZSkKICAgIHByaW50KGYiICAgQi1WID0ge2NhbFsnYSddOisuM2Z9ICsge2NhbFsnYyddOi4zZn0geCAoYi12KSAgICAgIHNjYXR0ZXIgIgogICAgICAgICAgZiJ7Y2FsWydybXNfY29sb3VyX2ZpdCddOi4zZn0gbWFnIiwgZmx1c2g9VHJ1ZSkKICAgIHByaW50KGYiICAgViAgID0gdiArIHtjYWxbJ1pQX1YnXTouM2Z9IHtjYWxbJ2tfViddOisuM2Z9IHggKEItVikgICBzY2F0dGVyICIKICAgICAgICAgIGYie2NhbFsncm1zX1ZfZml0J106LjNmfSBtYWciLCBmbHVzaD1UcnVlKQogICAgcHJpbnQoZiIgICBCICAgPSBiICsge2NhbFsnWlBfQiddOi4zZn0ge2NhbFsna19CJ106Ky4zZn0geCAoQi1WKSAgIHNjYXR0ZXIgIgogICAgICAgICAgZiJ7Y2FsWydybXNfQl9maXQnXTouM2Z9IG1hZyIsIGZsdXNoPVRydWUpCiAgICBwcmludChmIiAgIGNvbG91ciBjb3JyZWxhdGlvbiB7Y2FsWydjb3JyZWxhdGlvbiddOisuMmZ9IiwgZmx1c2g9VHJ1ZSkKCiAgICB0YWJsZSA9IGNhbmQuY29weSgpCiAgICB0YWJsZVsiVXNlZCJdID0gRmFsc2UKICAgIHRhYmxlLmxvY1tjYWxbInVzZWRfaW5kZXgiXSwgIlVzZWQiXSA9IFRydWUKICAgIGJ2X2ZpdCA9IGNhbFsiYSJdICsgY2FsWyJjIl0gKiAodGFibGVbIk1hZ19pbnN0X0IiXSAtIHRhYmxlWyJNYWdfaW5zdF9WIl0pCiAgICB0YWJsZVsiUmVzaWRfQlYiXSA9IGJ2X2ZpdCAtICh0YWJsZVsiQ2F0X0IiXSAtIHRhYmxlWyJDYXRfViJdKQogICAgdGFibGVbIlJlc2lkX1YiXSA9ICh0YWJsZVsiTWFnX2luc3RfViJdICsgY2FsWyJaUF9WIl0gKyBjYWxbImtfViJdICogYnZfZml0CiAgICAgICAgICAgICAgICAgICAgICAgIC0gdGFibGVbIkNhdF9WIl0pCiAgICBwYXRoID0gb3MucGF0aC5qb2luKG91dF9kaXIsIGYie3NhZmV9X3JlZmVyZW5jZV9zdGFycy5jc3YiKQogICAgdGFibGUudG9fY3N2KHBhdGgsIGluZGV4PUZhbHNlKQogICAgcHJpbnQoZiJSZWZlcmVuY2Ugc3RhcnMsIHdpdGggZWFjaCBvbmUncyByZXNpZHVhbCwgc2F2ZWQgdG8ge3BhdGh9IikKICAgIHJldHVybiBjYWwsIHdoaWNoLCBzdGF0cywgY2FuZAoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLSBEaWFncmFtcyAtLS0tLS0tLS0tLS0tLS0tLS0tCgpkZWYgbWFrZV9jbWQoZGYsIGJhc2VfZmlsZSwgbGFiZWwsIG5hbWUsIG11PU5vbmUsIHdpbmRvdz1Ob25lKToKICAgICIiIkNvbG91ci1tYWduaXR1ZGUgZGlhZ3JhbSBvZiB0aGUgc291cmNlcyBpbiBkZiwgc2F2ZWQgbmV4dCB0byBiYXNlX2ZpbGUuCgogICAgVGhlIGNvbG91ciBheGlzIGZvbGxvd3MgdGhlIGRhdGEgcmF0aGVyIHRoYW4gYSBmaXhlZCByYW5nZSAtIHRoZSBvbGQgZml4ZWQKICAgIC0wLjUgdG8gMC41IGN1dCBldmVyeSByZWQgc291cmNlIG91dCBvZiB0aGUgZGlhZ3JhbSBvZiBhbGwgc291cmNlcywgd2hpY2gKICAgIGlzIHRoZSBkaWFncmFtIHRoYXQgc2hvd3Mgd2hlcmUgdGhlIGJsdWUga25vdHMgc3RhbmQgYWdhaW5zdCB0aGUgcmVzdC4KICAgIFNhdHVyYXRlZCBzb3VyY2VzIGFuZCBzb3VyY2VzIHdpdGggYSBCLVYgZXJyb3IgYWJvdmUgQ01EX01BWF9CVl9FUlIgYXJlIG5vdAogICAgZHJhd24gKHRoZXkgc3RheSBpbiB0aGUgdGFibGVzKS4gV2l0aCBhIGRpc3RhbmNlLCB0aGUgcmlnaHQtaGFuZCBheGlzIGlzCiAgICB0aGUgYWJzb2x1dGUgbWFnbml0dWRlIE1fVi4gQmFycyBvbiB0aGUgbGVmdCBzaG93IHRoZSB0eXBpY2FsIGVycm9yIGF0CiAgICBlYWNoIG1hZ25pdHVkZS4KICAgICIiIgogICAgaWYgbm90IHsiTWFnX0IiLCAiTWFnX1YifSA8PSBzZXQoZGYuY29sdW1ucyk6CiAgICAgICAgcHJpbnQoZiJNYWdfQiBvciBNYWdfViBub3QgZm91bmQgaW4ge2xhYmVsfSBjYXRhbG9nLiBDTUQgbm90IGNyZWF0ZWQuIikKICAgICAgICByZXR1cm4KICAgIGRmID0gZGYuY29weSgpCiAgICBkZlsiQi1WIl0gPSBkZlsiTWFnX0IiXSAtIGRmWyJNYWdfViJdCiAgICBjb2xzID0gW2MgZm9yIGMgaW4gZGYuY29sdW1ucyBpZiBjIG5vdCBpbiAoIkItViIsICJNYWdfViIpXSArIFsiQi1WIiwgIk1hZ19WIl0KICAgIGRmID0gZGZbY29sc10KICAgIG91dF9maWxlX2NvbG9yID0gYmFzZV9maWxlLnJlcGxhY2UoIi5jc3YiLCAiX3dpdGhfY29sb3IuY3N2IikKICAgIGRmLnRvX2NzdihvdXRfZmlsZV9jb2xvciwgaW5kZXg9RmFsc2UpCiAgICBwcmludChmIkZpbGUgd2l0aCBjb2xvciBpbmRleCBzYXZlZCB0byB7b3V0X2ZpbGVfY29sb3J9IikKCiAgICBzaG93ID0gbnAuaXNmaW5pdGUoZGZbIkItViJdKSAmIG5wLmlzZmluaXRlKGRmWyJNYWdfViJdKQogICAgaWYgIlNhdHVyYXRlZCIgaW4gZGYuY29sdW1uczoKICAgICAgICBzaG93ICY9IH5kZlsiU2F0dXJhdGVkIl0uYXN0eXBlKGJvb2wpCiAgICBpZiAiQlZfZXJyIiBpbiBkZi5jb2x1bW5zOgogICAgICAgIHNob3cgJj0gZGZbIkJWX2VyciJdIDw9IENNRF9NQVhfQlZfRVJSCiAgICBkID0gZGZbc2hvd10KICAgIGlmIG5vdCBsZW4oZCk6CiAgICAgICAgcHJpbnQoZiJObyBzb3VyY2VzIHRvIGRyYXcgaW4gdGhlIHtsYWJlbH0gZGlhZ3JhbS4iKQogICAgICAgIHJldHVybgoKICAgIGZpZywgYXggPSBwbHQuc3VicGxvdHMoZmlnc2l6ZT0oOCwgMTApKQogICAgYXguc2NhdHRlcihkWyJCLVYiXSwgZFsiTWFnX1YiXSwgcz0xOCwgZWRnZWNvbG9yPSJibGFjayIsIGZhY2Vjb2xvcj0iY3lhbiIsCiAgICAgICAgICAgICAgIGFscGhhPTAuNywgbGluZXdpZHRocz0wLjQpCiAgICBsbywgaGkgPSBucC5wZXJjZW50aWxlKGRbIkItViJdLCBbMSwgOTldKQogICAgbG8sIGhpID0gbWluKGxvLCB3aW5kb3dbMF0gaWYgd2luZG93IGVsc2UgbG8pLCBtYXgoaGksIHdpbmRvd1sxXSBpZiB3aW5kb3cgZWxzZSBoaSkKICAgIHBhZCA9IDAuMSAqIG1heChoaSAtIGxvLCAwLjUpCiAgICBheC5zZXRfeGxpbShsbyAtIHBhZCwgaGkgKyBwYWQpCiAgICB5bWluLCB5bWF4ID0gbnAubmFubWluKGRbIk1hZ19WIl0pLCBucC5uYW5tYXgoZFsiTWFnX1YiXSkKICAgIGF4LnNldF95bGltKHltYXggKyAwLjMsIHltaW4gLSAwLjMpCiAgICBpZiB3aW5kb3c6CiAgICAgICAgZm9yIHh3IGluIHdpbmRvdzoKICAgICAgICAgICAgYXguYXh2bGluZSh4dywgY29sb3I9InRhYjpibHVlIiwgbHM9Ii0tIiwgbHc9MSkKICAgIGlmIHsiTWFnX1ZfZXJyIiwgIkJWX2VyciJ9IDw9IHNldChkLmNvbHVtbnMpOgogICAgICAgIHhfYmFyID0gbG8gLSAwLjUgKiBwYWQKICAgICAgICBmb3IgbTAgaW4gbnAuYXJhbmdlKG5wLmZsb29yKHltaW4pLCBucC5jZWlsKHltYXgpLCAxLjApOgogICAgICAgICAgICBzZWwgPSAoZFsiTWFnX1YiXSA+PSBtMCkgJiAoZFsiTWFnX1YiXSA8IG0wICsgMSkKICAgICAgICAgICAgaWYgc2VsLnN1bSgpID49IDM6CiAgICAgICAgICAgICAgICBheC5lcnJvcmJhcih4X2JhciwgbTAgKyAwLjUsIHhlcnI9bnAubWVkaWFuKGQubG9jW3NlbCwgIkJWX2VyciJdKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgIHllcnI9bnAubWVkaWFuKGQubG9jW3NlbCwgIk1hZ19WX2VyciJdKSwgY29sb3I9IjAuMyIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBjYXBzaXplPTIsIGx3PTEpCiAgICBheC5zZXRfeGxhYmVsKCIoQiAtIFYpJF8wJCAgW21hZ10sIGNvcnJlY3RlZCBmb3IgR2FsYWN0aWMgcmVkZGVuaW5nIikKICAgIGF4LnNldF95bGFiZWwoIlYkXzAkICBbbWFnXSwgYXBwYXJlbnQsIGNvcnJlY3RlZCBmb3IgR2FsYWN0aWMgZXh0aW5jdGlvbiIpCiAgICBpZiBtdSBpcyBub3QgTm9uZSBhbmQgbnAuaXNmaW5pdGUobXUpOgogICAgICAgIHNlYyA9IGF4LnNlY29uZGFyeV95YXhpcygicmlnaHQiLCBmdW5jdGlvbnM9KGxhbWJkYSBtOiBtIC0gbXUsIGxhbWJkYSBNOiBNICsgbXUpKQogICAgICAgIHNlYy5zZXRfeWxhYmVsKGYiTSRfViQgIFttYWddICAgKGRpc3RhbmNlIG1vZHVsdXMge211Oi4yZn0pIikKICAgIGxlZnRfb3V0ID0gaW50KGxlbihkZikgLSBsZW4oZCkpCiAgICBheC5zZXRfdGl0bGUoZiJDTUQgb2Yge25hbWV9IHtsYWJlbH1cbntsZW4oZCl9IG9mIHtsZW4oZGYpfSBzb3VyY2VzIGRyYXduIgogICAgICAgICAgICAgICAgICsgKGYiICh7bGVmdF9vdXR9IHNhdHVyYXRlZCBvciB3aXRoIGEgQi1WIGVycm9yIGFib3ZlICIKICAgICAgICAgICAgICAgICAgICBmIntDTURfTUFYX0JWX0VSUjpnfSBtYWcgbGVmdCBvdXQpIiBpZiBsZWZ0X291dCBlbHNlICIiKQogICAgICAgICAgICAgICAgICsgKGYiOyBkYXNoZWQgbGluZXM6IHRoZSBibHVlLWtub3QgY29sb3VyIHdpbmRvdyIgaWYgd2luZG93IGVsc2UgIiIpKQogICAgb3V0X2NtZCA9IGJhc2VfZmlsZS5yZXBsYWNlKCIuY3N2IiwgIl9DTUQucG5nIikKICAgIGZpZy5zYXZlZmlnKG91dF9jbWQsIGRwaT0xNTAsIGJib3hfaW5jaGVzPSJ0aWdodCIpCiAgICBwbHQuY2xvc2UoZmlnKQogICAgcHJpbnQoZiJDTUQgZGlhZ3JhbSBzYXZlZCB0byB7b3V0X2NtZH0iKQoKCmRlZiBzb3VyY2VfbWFwKGRhdGEsIHgsIHksIHRpdGxlLCBwYXRoLCBjb2xvdXI9ImdyZWVuIiwgbGFiZWw9Ik1lYXN1cmVkIHNvdXJjZXMiKToKICAgIHBsdC5maWd1cmUoZmlnc2l6ZT0oMTAsIDEwKSkKICAgIGZpbml0ZSA9IGRhdGFbbnAuaXNmaW5pdGUoZGF0YSldCiAgICBwbHQuaW1zaG93KGRhdGEsIGNtYXA9ImdyYXkiLCBvcmlnaW49Imxvd2VyIiwKICAgICAgICAgICAgICAgdm1pbj1ucC5wZXJjZW50aWxlKGZpbml0ZSwgNSksIHZtYXg9bnAucGVyY2VudGlsZShmaW5pdGUsIDk5KSkKICAgIHBsdC5jb2xvcmJhcihsYWJlbD0iQ291bnRzIikKICAgIHBsdC5zY2F0dGVyKHgsIHksIHM9NDAsIGVkZ2Vjb2xvcj1jb2xvdXIsIGZhY2Vjb2xvcj0ibm9uZSIsIGxhYmVsPWxhYmVsKQogICAgcGx0LnRpdGxlKHRpdGxlKQogICAgcGx0LnhsYWJlbCgiWCBbcGl4ZWxzXSIpCiAgICBwbHQueWxhYmVsKCJZIFtwaXhlbHNdIikKICAgIHBsdC5sZWdlbmQoKQogICAgcGx0LnNhdmVmaWcocGF0aCwgZHBpPTE1MCkKICAgIHBsdC5jbG9zZSgpCiAgICBwcmludChmIlNvdXJjZSBtYXAgc2F2ZWQgdG8ge3BhdGh9IikKCgpkZWYgZ2FsYXh5X2NlbnRyZV9waXhlbChuYW1lLCB3Y3MsIHNoYXBlKToKICAgICIiIldoZXJlIHRoZSBnYWxheHkgc2l0cyBpbiB0aGlzIGZyYW1lLCBpbiBwaXhlbHMsIGZyb20gaXRzIGNhdGFsb2d1ZWQKICAgIHBvc2l0aW9uIHRocm91Z2ggdGhlIChjaGVja2VkKSBwbGF0ZSBzb2x1dGlvbjsgTm9uZSB3aGVuIHVua25vd24gb3IKICAgIG91dHNpZGUgdGhlIGZyYW1lLiIiIgogICAgdHJ5OgogICAgICAgIHgsIHkgPSB3Y3Mud29ybGRfdG9fcGl4ZWwoU2t5Q29vcmQuZnJvbV9uYW1lKG5hbWUpKQogICAgICAgIHgsIHkgPSBmbG9hdCh4KSwgZmxvYXQoeSkKICAgICAgICBpZiAwIDw9IHggPCBzaGFwZVsxXSBhbmQgMCA8PSB5IDwgc2hhcGVbMF06CiAgICAgICAgICAgIHByaW50KGYiW2xvb2t1cF0ge25hbWV9OiBjZW50cmVkIGF0IHBpeGVsICh7eDouMGZ9LCB7eTouMGZ9KSIsIGZsdXNoPVRydWUpCiAgICAgICAgICAgIHJldHVybiB4LCB5CiAgICAgICAgcHJpbnQoZiJbbG9va3VwXSB7bmFtZX0ncyBjYXRhbG9ndWVkIHBvc2l0aW9uIGZhbGxzIG91dHNpZGUgdGhpcyBmcmFtZSIsCiAgICAgICAgICAgICAgZmx1c2g9VHJ1ZSkKICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOgogICAgICAgIHByaW50KGYiW2xvb2t1cF0gZ2FsYXh5IGNlbnRyZSB1bmF2YWlsYWJsZSAoe2V4Y30pIiwgZmx1c2g9VHJ1ZSkKICAgIHJldHVybiBOb25lCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tIFRoZSBydW4gLS0tLS0tLS0tLS0tLS0tLS0tLQoKZGVmIHBhcnNlX2FyZ3MoYXJndj1Ob25lKToKICAgIGFwID0gYXJncGFyc2UuQXJndW1lbnRQYXJzZXIoCiAgICAgICAgZGVzY3JpcHRpb249IkJsdWUga25vdHMgb2YgYSBnYWxheHkgZnJvbSBhIEIgYW5kIGEgViBmcmFtZTogY2FsaWJyYXRlZCAiCiAgICAgICAgICAgICAgICAgICAgInBob3RvbWV0cnksIGNvbG91ci1tYWduaXR1ZGUgZGlhZ3JhbSwgcmFkaWFsIHByb2ZpbGUuICIKICAgICAgICAgICAgICAgICAgICAiU3RhcnRlZCB3aXRob3V0IGFyZ3VtZW50cyBpdCBhc2tzIGl0cyBxdWVzdGlvbnMgaW5zdGVhZC4iKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLWZvbGRlciIsIGhlbHA9ImZvbGRlciBob2xkaW5nIHRoZSBmcmFtZXM7IHdoaWNoIGlzIEIgYW5kIHdoaWNoICIKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICJpcyBWIGlzIHJlYWQgZnJvbSB0aGUgaGVhZGVycyIpCiAgICBhcC5hZGRfYXJndW1lbnQoIi0tYiIsIGhlbHA9InRoZSBCIGZyYW1lLCB3aGVuIG5hbWluZyB0aGUgZmlsZXMgeW91cnNlbGYiKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLXYiLCBoZWxwPSJ0aGUgViBmcmFtZSIpCiAgICBhcC5hZGRfYXJndW1lbnQoIi0tZ2FsYXh5IiwgaGVscD0iZ2FsYXh5IG5hbWUgKG90aGVyd2lzZSBmcm9tIHRoZSBPQkpFQ1Qga2V5d29yZCkiKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLWRpc3RhbmNlLW1wYyIsIHR5cGU9ZmxvYXQsIGhlbHA9ImRpc3RhbmNlOyBvdGhlcndpc2UgbG9va2VkIHVwIikKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1lYnYiLCB0eXBlPWZsb2F0LCBoZWxwPSJHYWxhY3RpYyBFKEItVik7IG90aGVyd2lzZSBsb29rZWQgdXAiKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLWNhdGFsb2ciLCBjaG9pY2VzPVsiQVBBU1MiLCAiR0FJQSJdLCBkZWZhdWx0PUNBTElCX0NBVEFMT0dVRSwKICAgICAgICAgICAgICAgICAgICBoZWxwPSJyZWZlcmVuY2Ugc3RhcnMgZm9yIHRoZSBjYWxpYnJhdGlvbiAoZGVmYXVsdCBBUEFTUzsgR2FpYSBEUjMgIgogICAgICAgICAgICAgICAgICAgICAgICAgInN5bnRoZXRpYyBwaG90b21ldHJ5IGlzIGFsc28gdGhlIGF1dG9tYXRpYyBmYWxsYmFjaykiKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLW1heC1idi1lcnIiLCB0eXBlPWZsb2F0LCBkZWZhdWx0PUtOT1RfTUFYX0JWX0VSUiwKICAgICAgICAgICAgICAgICAgICBoZWxwPSJsYXJnZXN0IEItViBlcnJvciBhIGJsdWUga25vdCBtYXkgaGF2ZSwgaW4gbWFnIikKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1vdXQtZGlyIiwgaGVscD0id3JpdGUgZXZlcnl0aGluZyBpbnRvIHRoaXMgZm9sZGVyIChkZWZhdWx0OiBhICIKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAibmV3IGZvbGRlciByZXN1bHRfQ01EXzxnYWxheHk+XzxkYXRlPl88dGltZT4gIgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICJiZXNpZGUgdGhlIGZyYW1lcykiKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLXllcyIsIGFjdGlvbj0ic3RvcmVfdHJ1ZSIsCiAgICAgICAgICAgICAgICAgICAgaGVscD0iYWNjZXB0IGV2ZXJ5IGRlZmF1bHQgd2l0aG91dCBhc2tpbmciKQogICAgcmV0dXJuIGFwLnBhcnNlX2FyZ3MoYXJndikKCgpkZWYgbWFpbihhcmd2PU5vbmUpOgogICAgZ2xvYmFsIEFSR1MsIGFzaywgYXNrX3llc19ubwogICAgQVJHUyA9IHBhcnNlX2FyZ3MoYXJndikKICAgIGlmIEFSR1MueWVzOgogICAgICAgICMgVGhlIHNoYXJlZCBmb2xkZXIgcXVlc3Rpb25zICh3aGljaCBmcmFtZSwgd2hldGhlciB0byBjb21iaW5lKSB0YWtlCiAgICAgICAgIyB0aGVpciBkZWZhdWx0cyB0b28uCiAgICAgICAgYXNrID0gbGFtYmRhIHByb21wdCwgZGVmYXVsdD1Ob25lLCBjYXN0PXN0cjogY2FzdChkZWZhdWx0KQogICAgICAgIGFza195ZXNfbm8gPSBsYW1iZGEgcHJvbXB0LCBkZWZhdWx0PUZhbHNlOiBkZWZhdWx0CgogICAgcHJpbnQoIj09PSBQaG90b21ldHJ5IElucHV0ID09PSIpCiAgICBmb2xkZXIgPSBBUkdTLmZvbGRlcgogICAgaWYgbm90IGZvbGRlciBhbmQgbm90IChBUkdTLmIgYW5kIEFSR1Mudik6CiAgICAgICAgZm9sZGVyID0gX25vcm1fcGF0aChfYXNrKAogICAgICAgICAgICAiUGF0aCB0byB0aGUgZm9sZGVyIGhvbGRpbmcgdGhlIGZyYW1lcyAoRW50ZXIgdG8gbmFtZSB0aGUgdHdvIGZpbGVzIGluc3RlYWQpIiwKICAgICAgICAgICAgIiIsIHN0cikpCiAgICBmaXRzX2ZpbGVfQiwgZml0c19maWxlX1YgPSBBUkdTLmIsIEFSR1MudgogICAgb2JqX25hbWUgPSBBUkdTLmdhbGF4eQogICAgZ3JvdXBzLCBkb25vcnMgPSB7fSwge30KICAgIGlmIGZvbGRlciBhbmQgb3MucGF0aC5pc2Rpcihmb2xkZXIpOgogICAgICAgIGNob3NlbiwgZG9ub3JzLCBncm91cHMsIGZvdW5kID0gZnJhbWVzX2Zyb21fZm9sZGVyKGZvbGRlciwgKCJCIiwgIlYiKSkKICAgICAgICBmaXRzX2ZpbGVfQiwgZml0c19maWxlX1YgPSBjaG9zZW5bIkIiXSwgY2hvc2VuWyJWIl0KICAgICAgICBvYmpfbmFtZSA9IG9ial9uYW1lIG9yIGZvdW5kCiAgICBlbGlmIGZvbGRlcjoKICAgICAgICBwcmludCgiW2lucHV0XSB0aGVyZSBpcyBubyBmb2xkZXIgYXQgIiArIGZvbGRlciArICIgLSBuYW1pbmcgdGhlIHR3byBmaWxlcyAiCiAgICAgICAgICAgICAgImluc3RlYWQiLCBmbHVzaD1UcnVlKQogICAgICAgIGZvbGRlciA9IE5vbmUKICAgIGlmIG5vdCBmaXRzX2ZpbGVfQjoKICAgICAgICBmaXRzX2ZpbGVfQiA9IF9ub3JtX3BhdGgoX2FzaygiUGF0aCB0byBCLWJhbmQgRklUUyIsIE5vbmUsIHN0cikpCiAgICBpZiBub3QgZml0c19maWxlX1Y6CiAgICAgICAgZml0c19maWxlX1YgPSBfbm9ybV9wYXRoKF9hc2soIlBhdGggdG8gVi1iYW5kIEZJVFMiLCBOb25lLCBzdHIpKQogICAgZm9yIHAgaW4gKGZpdHNfZmlsZV9CLCBmaXRzX2ZpbGVfVik6CiAgICAgICAgaWYgbm90IG9zLnBhdGguaXNmaWxlKHApOgogICAgICAgICAgICBzdG9wKGYidGhlcmUgaXMgbm8gZmlsZSBhdCB7cH0iKQogICAgaWYgbm90IG9ial9uYW1lOgogICAgICAgIHRyeToKICAgICAgICAgICAgb2JqX25hbWUgPSBzdHIoZml0cy5nZXRoZWFkZXIoZml0c19maWxlX1YpLmdldCgiT0JKRUNUIiwgIiIpKS5zdHJpcCgpIG9yIE5vbmUKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBvYmpfbmFtZSA9IE5vbmUKICAgIGlmIG5vdCBvYmpfbmFtZToKICAgICAgICBvYmpfbmFtZSA9IF9hc2soIkdhbGF4eSBuYW1lIiwgTm9uZSwgc3RyKQogICAgc2FmZSA9IHJlLnN1YihyIlteQS1aYS16MC05Xy4tXSsiLCAiXyIsIG9ial9uYW1lLnN0cmlwKCkpCgogICAgZGF0YWRpciA9IGZvbGRlciBvciBvcy5wYXRoLmRpcm5hbWUob3MucGF0aC5hYnNwYXRoKGZpdHNfZmlsZV9WKSkKICAgIGlmIEFSR1Mub3V0X2RpcjoKICAgICAgICBvdXRfZGlyID0gb3MucGF0aC5hYnNwYXRoKEFSR1Mub3V0X2RpcikKICAgICAgICBvcy5tYWtlZGlycyhvdXRfZGlyLCBleGlzdF9vaz1UcnVlKQogICAgZWxzZToKICAgICAgICBvdXRfZGlyID0gbmV3X3J1bl9mb2xkZXIoZGF0YWRpciwgIkNNRCIsIG9ial9uYW1lKQogICAgcHJpbnQoZiJbb3V0XSB3cml0aW5nIGV2ZXJ5dGhpbmcgdG8ge291dF9kaXJ9IikKCiAgICBjb21iaW5lZF9ub3RlID0gW10KICAgIGZvciByb2xlIGluICgiQiIsICJWIik6CiAgICAgICAgcGF0aHMgPSBncm91cHMuZ2V0KHJvbGUpCiAgICAgICAgaWYgbm90IHBhdGhzIGFuZCBkb25vcnMuZ2V0KHJvbGUpOgogICAgICAgICAgICAjIE9uZSBleHBvc3VyZSB3aXRoIG5vIHBsYXRlIHNvbHV0aW9uOiBpdCBib3Jyb3dzIG9uZSBmcm9tIGEgc29sdmVkCiAgICAgICAgICAgICMgZXhwb3N1cmUgb2YgdGhlIHNhbWUgZmllbGQsIGFuZCB0aGUgc29sdmVkIGNvcHkgaXMgd2hhdCBpcyBtZWFzdXJlZC4KICAgICAgICAgICAgcGF0aCA9IGZpdHNfZmlsZV9CIGlmIHJvbGUgPT0gIkIiIGVsc2UgZml0c19maWxlX1YKICAgICAgICAgICAgZGF0YV9jLCB3Y3NfYywgX2JwbSwgaW5mb19jID0gbG9hZF9mcmFtZShwYXRoLCBkb25vcnNbcm9sZV0pCiAgICAgICAgICAgIG91dCA9IG9zLnBhdGguam9pbihvdXRfZGlyLCBmIntzYWZlfV97cm9sZX1fc29sdmVkLmZpdHMiKQogICAgICAgICAgICBzYXZlX2ZyYW1lKG91dCwgZGF0YV9jLCB3Y3NfYywgaW5mb19jLCBoaXN0b3J5PSgKICAgICAgICAgICAgICAgICJQbGF0ZSBzb2x1dGlvbiBmaXR0ZWQgYWdhaW5zdCAiICsgb3MucGF0aC5iYXNlbmFtZShkb25vcnNbcm9sZV0pLCkpCiAgICAgICAgICAgIGNvbWJpbmVkX25vdGUuYXBwZW5kKGYie3JvbGV9OiBwbGF0ZSBzb2x1dGlvbiBib3Jyb3dlZCBmcm9tICIKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZiJ7b3MucGF0aC5iYXNlbmFtZShkb25vcnNbcm9sZV0pfSIpCiAgICAgICAgICAgIGlmIHJvbGUgPT0gIkIiOgogICAgICAgICAgICAgICAgZml0c19maWxlX0IgPSBvdXQKICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgIGZpdHNfZmlsZV9WID0gb3V0CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgaWYgbm90IHBhdGhzOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGRhdGFfYywgd2NzX2MsIF9icG0sIGluZm9fYyA9IGNvbWJpbmVfZXhwb3N1cmVzKHBhdGhzLCBkb25vcnMuZ2V0KHJvbGUpLCByb2xlKQogICAgICAgIG91dCA9IG9zLnBhdGguam9pbihvdXRfZGlyLCBmIntzYWZlfV97cm9sZX1fY29tYmluZWQuZml0cyIpCiAgICAgICAgc2F2ZV9mcmFtZShvdXQsIGRhdGFfYywgd2NzX2MsIGluZm9fYywKICAgICAgICAgICAgICAgICAgIGhpc3Rvcnk9KGYiTWVkaWFuIG9mIHtsZW4ocGF0aHMpfSBleHBvc3VyZXMiLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIkNvbWJpbmVkIGJ5IEJsdWVfQ2x1c3RlcnNfRnJvbV9JbWFnZXMucHkiKSkKICAgICAgICBjb21iaW5lZF9ub3RlLmFwcGVuZChmIntyb2xlfTogbWVkaWFuIG9mIHtsZW4ocGF0aHMpfSBleHBvc3VyZXMiKQogICAgICAgIGlmIHJvbGUgPT0gIkIiOgogICAgICAgICAgICBmaXRzX2ZpbGVfQiA9IG91dAogICAgICAgIGVsc2U6CiAgICAgICAgICAgIGZpdHNfZmlsZV9WID0gb3V0CiAgICAgICAgZGVsIGRhdGFfYwoKICAgICMgQSBmaWxlIG5hbWVkIG9uIHRoZSBjb21tYW5kIGxpbmUgdGhhdCBoYXMgbm8gcGxhdGUgc29sdXRpb24gLSB0eXBpY2FsbHkgYQogICAgIyBtZWRpYW4gd3JpdHRlbiBieSBhbm90aGVyIHByb2dyYW0sIHdoaWNoIGtlcHQgbm8gaGVhZGVyIC0gYm9ycm93cyBvbmUgZnJvbQogICAgIyBhIHNvbHZlZCBleHBvc3VyZSBpbiBpdHMgb3duIGZvbGRlciwgZXhhY3RseSBhcyBpbiB0aGUgZm9sZGVyIG1vZGUuCiAgICBmb3Igcm9sZSBpbiAoIkIiLCAiViIpOgogICAgICAgIHBhdGggPSBmaXRzX2ZpbGVfQiBpZiByb2xlID09ICJCIiBlbHNlIGZpdHNfZmlsZV9WCiAgICAgICAgaWYgZ3JvdXBzLmdldChyb2xlKSBvciBkb25vcnMuZ2V0KHJvbGUpOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGRvbm9yID0gZG9ub3JfYmVzaWRlKHBhdGgsIHJvbGUpCiAgICAgICAgaWYgZG9ub3IgaXMgTm9uZToKICAgICAgICAgICAgY29udGludWUKICAgICAgICBwcmludChmIltpbnB1dF0ge3JvbGV9OiB7b3MucGF0aC5iYXNlbmFtZShwYXRoKX0gaGFzIG5vIHBsYXRlIHNvbHV0aW9uOyAiCiAgICAgICAgICAgICAgZiJmaXR0aW5nIG9uZSBhZ2FpbnN0IHtvcy5wYXRoLmJhc2VuYW1lKGRvbm9yKX0iLCBmbHVzaD1UcnVlKQogICAgICAgIGRhdGFfYywgd2NzX2MsIF9icG0sIGluZm9fYyA9IGxvYWRfZnJhbWUocGF0aCwgZG9ub3IpCiAgICAgICAgaWYgd2NzX2MgaXMgTm9uZToKICAgICAgICAgICAgY29udGludWUKICAgICAgICAjIFRoZSBudW1iZXIgb2YgZnJhbWVzIGNvbWJpbmVkIGJlbG9uZ3MgdG8gdGhpcyBmaWxlLCBub3QgdG8gdGhlIGRvbm9yOwogICAgICAgICMgaXQgaXMga25vd24gb25seSBpZiB0aGlzIGZpbGUncyBvd24gaGVhZGVyIHNheXMgaXQuCiAgICAgICAgbl9jb21iLCBob3cgPSBjb21iaW5lZF9jb3VudChmaXRzLmdldGhlYWRlcihwYXRoKSkKICAgICAgICBpbmZvX2NbIm5jb21iaW5lIl0sIGluZm9fY1siY29tYnR5cGUiXSA9IG5fY29tYiwgKCIiIGlmIG5fY29tYiA9PSAxIGVsc2UgaG93KQogICAgICAgIG91dCA9IG9zLnBhdGguam9pbihvdXRfZGlyLCBmIntzYWZlfV97cm9sZX1fc29sdmVkLmZpdHMiKQogICAgICAgIHNhdmVfZnJhbWUob3V0LCBkYXRhX2MsIHdjc19jLCBpbmZvX2MsIGhpc3Rvcnk9KAogICAgICAgICAgICAiUGxhdGUgc29sdXRpb24gZml0dGVkIGFnYWluc3QgIiArIG9zLnBhdGguYmFzZW5hbWUoZG9ub3IpLCkpCiAgICAgICAgY29tYmluZWRfbm90ZS5hcHBlbmQoZiJ7cm9sZX06IHBsYXRlIHNvbHV0aW9uIGJvcnJvd2VkIGZyb20gIgogICAgICAgICAgICAgICAgICAgICAgICAgICAgIGYie29zLnBhdGguYmFzZW5hbWUoZG9ub3IpfSIpCiAgICAgICAgaWYgcm9sZSA9PSAiQiI6CiAgICAgICAgICAgIGZpdHNfZmlsZV9CID0gb3V0CiAgICAgICAgZWxzZToKICAgICAgICAgICAgZml0c19maWxlX1YgPSBvdXQKICAgICAgICBkZWwgZGF0YV9jCgogICAgIyAtLS0tIHRoZSBwdWJsaXNoZWQgbnVtYmVycywgc2hvd24gYW5kIGNvbmZpcm1lZCBiZWZvcmUgdXNlIC0tLS0tLS0tLS0tLQogICAgcHJpbnQoIkxvb2tpbmcgdGhlIGdhbGF4eSB1cC4uLiIsIGZsdXNoPVRydWUpCiAgICBkaXN0YW5jZSA9IG9mZmVyZWQoIkRpc3RhbmNlIChNcGMpIiwgbG9va3VwX2Rpc3RhbmNlKG9ial9uYW1lKSwKICAgICAgICAgICAgICAgICAgICAgICAiQ29zbWljZmxvd3MtMywgVHVsbHkgZXQgYWwuIDIwMTYiLCAiIE1wYyIsIEFSR1MuZGlzdGFuY2VfbXBjKQogICAgRV9CViA9IG9mZmVyZWQoIkdhbGFjdGljIGNvbG91ciBleGNlc3MgRShCLVYpIiwgbG9va3VwX2V4dGluY3Rpb24ob2JqX25hbWUpLAogICAgICAgICAgICAgICAgICAgIlNjaGxhZmx5ICYgRmlua2JlaW5lciAyMDExLCB2aWEgSVJTQSIsICIiLCBBUkdTLmVidikKICAgIGlmIGRpc3RhbmNlIDw9IDAgb3IgRV9CViA8IDA6CiAgICAgICAgc3RvcCgidGhlIGRpc3RhbmNlIG11c3QgYmUgcG9zaXRpdmUgYW5kIEUoQi1WKSBub3QgbmVnYXRpdmUiKQogICAgQV9WID0gUl9WICogRV9CVgogICAgbXUgPSA1LjAgKiBucC5sb2cxMChkaXN0YW5jZSAqIDEuMGU2KSAtIDUuMAogICAgcHJpbnQoZiJVc2luZyBkaXN0YW5jZSB7ZGlzdGFuY2U6LjJmfSBNcGMgKGRpc3RhbmNlIG1vZHVsdXMge211Oi4yZn0pLCIKICAgICAgICAgIGYiIEUoQi1WKSA9IHtFX0JWOi40Zn0sIEFfViA9IHtBX1Y6LjNmfSIsIGZsdXNoPVRydWUpCiAgICBmb3IgbGluZSBpbiBjb21iaW5lZF9ub3RlOgogICAgICAgIHByaW50KCJbaW5wdXRdICIgKyBsaW5lLCBmbHVzaD1UcnVlKQoKICAgICMgLS0tLSBmcmFtZXMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KICAgIHBhdGhfQiwgcmF3X0IsIGRhdGFfQiwgaGRyX0IgPSBzdWJ0cmFjdF9iYWNrZ3JvdW5kX2FuZF9zYXZlKGZpdHNfZmlsZV9CLCBvdXRfZGlyKQogICAgcGF0aF9WLCByYXdfViwgZGF0YV9WLCBoZHJfViA9IHN1YnRyYWN0X2JhY2tncm91bmRfYW5kX3NhdmUoZml0c19maWxlX1YsIG91dF9kaXIpCiAgICBzaGFwZV9WID0gZGF0YV9WLnNoYXBlCiAgICB3Y3NfViA9IFdDUyhoZHJfVikuY2VsZXN0aWFsCiAgICBpZiBub3Qgd2NzX1YuaGFzX2NlbGVzdGlhbDoKICAgICAgICBzdG9wKCJ0aGUgViBmcmFtZSBoYXMgbm8gcGxhdGUgc29sdXRpb247IHBsYXRlLXNvbHZlIGl0IGFuZCByZXJ1biIpCiAgICBzY2FsZV9hcmNzZWMgPSAzNjAwLjAgKiBmbG9hdChucC5zcXJ0KGFicyhucC5saW5hbGcuZGV0KHdjc19WLnBpeGVsX3NjYWxlX21hdHJpeCkpKSkKCiAgICBpbmZvID0ge30KICAgIGZvciBiYW5kLCByYXcsIGhkciBpbiAoKCJCIiwgcmF3X0IsIGhkcl9CKSwgKCJWIiwgcmF3X1YsIGhkcl9WKSk6CiAgICAgICAgc2F0LCBzYXRfc3JjID0gc2F0dXJhdGlvbl9sZXZlbChyYXcsIGhkcikKICAgICAgICBnYWluLCBnYWluX3NyYyA9IGRldGVjdG9yX2dhaW4oaGRyKQogICAgICAgIHJuLCBybl9zcmMgPSByZWFkX25vaXNlKGhkcikKICAgICAgICBpbmZvW2JhbmRdID0geyJzYXQiOiBzYXQsICJnYWluIjogZ2FpbiwgInJuIjogcm59CiAgICAgICAgcHJpbnQoZiIgICB7YmFuZH06IHNhdHVyYXRpb24gbGV2ZWwge3NhdDouMGZ9IEFEVSAoe3NhdF9zcmN9KTsgZ2FpbiAiCiAgICAgICAgICAgICAgZiJ7Z2FpbjouMmZ9IGUtL0FEVSAoe2dhaW5fc3JjfSk7IHJlYWQgbm9pc2Uge3JuOmd9IGUtICh7cm5fc3JjfSkiLAogICAgICAgICAgICAgIGZsdXNoPVRydWUpCgogICAgIyBXaGVyZSB0aGUgZ2FsYXh5IGlzIGFuZCBob3cgZmFyIGl0IHJlYWNoZXMsIGZvciB0aGUgZGV0ZWN0aW9uIHdpbmRvdywgdGhlCiAgICAjIGNhbGlicmF0aW9uIHN0YXJzIGFuZCB0aGUgdHJpbW1pbmcgb2YgdGhlIGNhdGFsb2d1ZS4KICAgIGNlbnRyZV9weCA9IGdhbGF4eV9jZW50cmVfcGl4ZWwob2JqX25hbWUsIHdjc19WLCBzaGFwZV9WKQogICAgcjI1ID0gTm9uZQogICAgZ2VvbSA9IE5vbmUKICAgIHdpbmRvdyA9IE5vbmUKICAgIHRyeToKICAgICAgICBnZW9tID0gaHlwZXJsZWRhX2dlb21ldHJ5KG9ial9uYW1lKQogICAgICAgIGlmIGdlb20gYW5kIGdlb20uZ2V0KCJkMjVfYXJjbWluIik6CiAgICAgICAgICAgIHIyNSA9IDAuNSAqIGdlb21bImQyNV9hcmNtaW4iXSAqIDYwLjAgLyBzY2FsZV9hcmNzZWMKICAgICAgICAgICAgaWYgY2VudHJlX3B4IGlzIG5vdCBOb25lOgogICAgICAgICAgICAgICAgd2luZG93ID0gZ2FsYXh5X3dpbmRvdyhzaGFwZV9WLCBjZW50cmVfcHgsIHIyNSwgbWFyZ2luPTEuNSkKICAgICAgICAgICAgICAgIGlmIHdpbmRvdyBpcyBOb25lOgogICAgICAgICAgICAgICAgICAgIHByaW50KCJbd2luZG93XSB0aGUgZ2FsYXh5IGZpbGxzIHRoaXMgZnJhbWUgLSB3b3JraW5nIG9uIGFsbCBvZiBpdCIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgZmx1c2g9VHJ1ZSkKICAgICAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICAgICAgeTAsIHkxLCB4MCwgeDEgPSB3aW5kb3cKICAgICAgICAgICAgICAgICAgICBwcmludChmIlt3aW5kb3ddIHRoZSBnYWxheHkgb2NjdXBpZXMge3gxIC0geDB9IHgge3kxIC0geTB9IG9mICIKICAgICAgICAgICAgICAgICAgICAgICAgICBmIntzaGFwZV9WWzFdfSB4IHtzaGFwZV9WWzBdfSBwaXhlbHM7IHRoZSByZXN0IGlzIHNreSIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgZmx1c2g9VHJ1ZSkKICAgICAgICBlbHNlOgogICAgICAgICAgICBwcmludCgiW3dpbmRvd10gbm8gaXNvcGhvdGFsIHJhZGl1cyBhdmFpbGFibGUgLSB3b3JraW5nIG9uIHRoZSB3aG9sZSBmcmFtZSIsCiAgICAgICAgICAgICAgICAgIGZsdXNoPVRydWUpCiAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGV4YzoKICAgICAgICBwcmludChmIlt3aW5kb3ddIGNvdWxkIG5vdCB3b3JrIG91dCB3aGVyZSB0aGUgZ2FsYXh5IGVuZHMgKHtleGN9KSAtIHdvcmtpbmcgIgogICAgICAgICAgICAgIGYib24gdGhlIHdob2xlIGZyYW1lIiwgZmx1c2g9VHJ1ZSkKCiAgICAjIC0tLS0gZGV0ZWN0aW9uIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KICAgIHByaW50KCJSZWdpc3RlcmluZyBCIG9udG8gVi4uLiIsIGZsdXNoPVRydWUpCiAgICBiX3RvX3YsIHZfdG9fYiwgcmVnX3RleHQgPSByZWdpc3Rlcl9CX29udG9fVihkYXRhX0IsIGRhdGFfViwgcGF0aF9CLCBwYXRoX1YpCiAgICB3aW5kb3dfQiA9IE5vbmUKICAgIGlmIHdpbmRvdyBpcyBub3QgTm9uZToKICAgICAgICAjIFRoZSBzYW1lIHBhcnQgb2YgdGhlIHNreSBpbiBCJ3Mgb3duIGdyaWQsIHdpdGggYSBtYXJnaW4gZm9yIHJvdGF0aW9uLgogICAgICAgIHkwLCB5MSwgeDAsIHgxID0gd2luZG93CiAgICAgICAgY29ybmVycyA9IHZfdG9fYihucC5hcnJheShbW3gwLCB5MF0sIFt4MSwgeTBdLCBbeDAsIHkxXSwgW3gxLCB5MV1dLCBmbG9hdCkpCiAgICAgICAgYngwLCBieTAgPSBucC5mbG9vcihjb3JuZXJzLm1pbihheGlzPTApKS5hc3R5cGUoaW50KSAtIDIwCiAgICAgICAgYngxLCBieTEgPSBucC5jZWlsKGNvcm5lcnMubWF4KGF4aXM9MCkpLmFzdHlwZShpbnQpICsgMjAKICAgICAgICB3aW5kb3dfQiA9IChtYXgoYnkwLCAwKSwgbWluKGJ5MSwgZGF0YV9CLnNoYXBlWzBdKSwKICAgICAgICAgICAgICAgICAgICBtYXgoYngwLCAwKSwgbWluKGJ4MSwgZGF0YV9CLnNoYXBlWzFdKSkKICAgIHhCLCB5QiwgZkIsIGZ3aG1fQiA9IGRldGVjdF9zb3VyY2VzKGRhdGFfQiwgcmF3X0IsICJCIiwgaW5mb1siQiJdWyJzYXQiXSwgd2luZG93X0IpCiAgICB4ViwgeVYsIGZWLCBmd2htX1YgPSBkZXRlY3Rfc291cmNlcyhkYXRhX1YsIHJhd19WLCAiViIsIGluZm9bIlYiXVsic2F0Il0sIHdpbmRvdykKICAgIGZ3aG0gPSBtYXgoZndobV9CLCBmd2htX1YpCgogICAgIyBPbmUgbGlzdCBvZiBzb3VyY2VzIG9uIHRoZSBWIGdyaWQ6IGV2ZXJ5dGhpbmcgZm91bmQgaW4gViwgYW5kIHdoYXRldmVyIHdhcwogICAgIyBmb3VuZCBpbiBCIHdpdGggbm90aGluZyBpbiBWIHdpdGhpbiBvbmUgRldITSAtIGEgdmVyeSBibHVlIGtub3QgY2FuIGJlCiAgICAjIGNsZWFyIGluIEIgYW5kIGxvc3QgaW4gVi4gRWFjaCBpcyB0aGVuIG1lYXN1cmVkIGluIGJvdGggZnJhbWVzIGF0IHRoZQogICAgIyBzYW1lIHBsYWNlIG9uIHRoZSBza3kgdGhyb3VnaCB0aGUgc2FtZSBhcGVydHVyZS4gQmVmb3JlLCBlYWNoIGJhbmQgd2FzCiAgICAjIG1lYXN1cmVkIHdoZXJlIGl0IGZvdW5kIHRoZSBzb3VyY2UsIHRocm91Z2ggYW4gYXBlcnR1cmUgc2l6ZWQgZnJvbSB0aGF0CiAgICAjIHNvdXJjZSdzIHdpZHRoIGluIHRoYXQgYmFuZDogdGhlIEIgYW5kIFYgYXBlcnR1cmVzIG9mIG9uZSBrbm90IGRpZmZlcmVkLAogICAgIyBhbmQgc28gaXRzIGNvbG91ciBkZXBlbmRlZCBvbiBpdHMgc2l6ZSBhcyBtdWNoIGFzIG9uIGl0cyBsaWdodC4KICAgIHh5Ql9vbl9WID0gYl90b192KG5wLmNvbHVtbl9zdGFjayhbeEIsIHlCXSkpIGlmIGxlbih4QikgZWxzZSBucC5lbXB0eSgoMCwgMikpCiAgICB4eVYgPSBucC5jb2x1bW5fc3RhY2soW3hWLCB5Vl0pIGlmIGxlbih4VikgZWxzZSBucC5lbXB0eSgoMCwgMikpCiAgICBpZiBsZW4oeHlWKSBhbmQgbGVuKHh5Ql9vbl9WKToKICAgICAgICBkLCBfID0gY0tEVHJlZSh4eVYpLnF1ZXJ5KHh5Ql9vbl9WLCBrPTEpCiAgICAgICAgYl9vbmx5ID0gZCA+IGZ3aG0KICAgIGVsc2U6CiAgICAgICAgYl9vbmx5ID0gbnAub25lcyhsZW4oeHlCX29uX1YpLCBkdHlwZT1ib29sKQogICAgeHkgPSBucC52c3RhY2soW3h5ViwgeHlCX29uX1ZbYl9vbmx5XV0pCiAgICBmb3VuZF9pbiA9IG5wLmFycmF5KFsiViJdICogbGVuKHh5VikgKyBbIkIiXSAqIGludChiX29ubHkuc3VtKCkpLCBkdHlwZT0iPFUyIikKICAgIGlmIGxlbih4eVYpIGFuZCBsZW4oeHlCX29uX1YpOgogICAgICAgIGRWLCBfID0gY0tEVHJlZSh4eUJfb25fVikucXVlcnkoeHlWLCBrPTEpCiAgICAgICAgZm91bmRfaW5bOmxlbih4eVYpXVtkViA8PSBmd2htXSA9ICJCViIKCiAgICByX2FwID0gQVBFUlRVUkVfU0NBTEUgKiBmd2htCiAgICByX2luLCByX291dCA9IEFOTlVMVVNfSU5ORVJfU0NBTEUgKiByX2FwLCBBTk5VTFVTX09VVEVSX1NDQUxFICogcl9hcAogICAgeHlfaW5fQiA9IHZfdG9fYih4eSkgaWYgbGVuKHh5KSBlbHNlIHh5CiAgICBlZGdlID0gcl9vdXQgKyAyCiAgICBpbnNpZGUgPSAoKHh5WzosIDBdID4gZWRnZSkgJiAoeHlbOiwgMF0gPCBzaGFwZV9WWzFdIC0gZWRnZSkKICAgICAgICAgICAgICAmICh4eVs6LCAxXSA+IGVkZ2UpICYgKHh5WzosIDFdIDwgc2hhcGVfVlswXSAtIGVkZ2UpCiAgICAgICAgICAgICAgJiAoeHlfaW5fQls6LCAwXSA+IGVkZ2UpICYgKHh5X2luX0JbOiwgMF0gPCBkYXRhX0Iuc2hhcGVbMV0gLSBlZGdlKQogICAgICAgICAgICAgICYgKHh5X2luX0JbOiwgMV0gPiBlZGdlKSAmICh4eV9pbl9CWzosIDFdIDwgZGF0YV9CLnNoYXBlWzBdIC0gZWRnZSkpCiAgICB4eSwgeHlfaW5fQiwgZm91bmRfaW4gPSB4eVtpbnNpZGVdLCB4eV9pbl9CW2luc2lkZV0sIGZvdW5kX2luW2luc2lkZV0KICAgIGlmIG5vdCBsZW4oeHkpOgogICAgICAgIHN0b3AoIm5vIHNvdXJjZXMgd2VyZSBmb3VuZCBpbiBib3RoIGZyYW1lcyIpCiAgICBwcmludChmIk1lYXN1cmluZyB7bGVuKHh5KTosfSBzb3VyY2VzIGluIGJvdGggYmFuZHMgdGhyb3VnaCBvbmUgYXBlcnR1cmUgb2YgIgogICAgICAgICAgZiJ7cl9hcDouMWZ9IHB4ICh7QVBFUlRVUkVfU0NBTEU6Z30geCB0aGUgRldITSBvZiB7ZndobTouMmZ9IHB4OyBza3kgcmluZyAiCiAgICAgICAgICBmIntyX2luOi4xZn0te3Jfb3V0Oi4xZn0gcHgpICBbZm91bmQgaW4gYm90aDoge2ludCgoZm91bmRfaW4gPT0gJ0JWJykuc3VtKCkpOix9LCIKICAgICAgICAgIGYiIFYgb25seToge2ludCgoZm91bmRfaW4gPT0gJ1YnKS5zdW0oKSk6LH0sIEIgb25seTogIgogICAgICAgICAgZiJ7aW50KChmb3VuZF9pbiA9PSAnQicpLnN1bSgpKTosfV0iLCBmbHVzaD1UcnVlKQoKICAgIGZsdXhWLCBlcnJWLCBwZWFrViwgc2t5ViA9IGFwZXJ0dXJlX3Bob3RvbWV0cnlfYXQoCiAgICAgICAgZGF0YV9WLCByYXdfViwgeHksIHJfYXAsIHJfaW4sIHJfb3V0LCBpbmZvWyJWIl1bImdhaW4iXSwgaW5mb1siViJdWyJybiJdKQogICAgZmx1eEIsIGVyckIsIHBlYWtCLCBza3lCID0gYXBlcnR1cmVfcGhvdG9tZXRyeV9hdCgKICAgICAgICBkYXRhX0IsIHJhd19CLCB4eV9pbl9CLCByX2FwLCByX2luLCByX291dCwgaW5mb1siQiJdWyJnYWluIl0sIGluZm9bIkIiXVsicm4iXSkKICAgIG1hZ1YsIG1hZ1ZfZXJyID0gaW5zdHJ1bWVudGFsKGZsdXhWLCBlcnJWKQogICAgbWFnQiwgbWFnQl9lcnIgPSBpbnN0cnVtZW50YWwoZmx1eEIsIGVyckIpCiAgICB3aWR0aFYgPSBucC5hcnJheShbc3Rhcl93aWR0aChkYXRhX1YsIHgsIHksIEZXSE1fV0lORE9XX1NJWkUsIHMpCiAgICAgICAgICAgICAgICAgICAgICAgZm9yICh4LCB5KSwgcyBpbiB6aXAoeHksIHNreVYpXSkKICAgIHdpZHRoQiA9IG5wLmFycmF5KFtzdGFyX3dpZHRoKGRhdGFfQiwgeCwgeSwgRldITV9XSU5ET1dfU0laRSwgcykKICAgICAgICAgICAgICAgICAgICAgICBmb3IgKHgsIHkpLCBzIGluIHppcCh4eV9pbl9CLCBza3lCKV0pCiAgICByYSwgZGVjID0gd2NzX1YuYWxsX3BpeDJ3b3JsZCh4eVs6LCAwXSwgeHlbOiwgMV0sIDApCgogICAgZGYgPSBwZC5EYXRhRnJhbWUoewogICAgICAgICJYX0IiOiB4eV9pbl9CWzosIDBdLCAiWV9CIjogeHlfaW5fQls6LCAxXSwgIkZXSE1fQiI6IHdpZHRoQiwKICAgICAgICAiRmx1eF9CIjogZmx1eEIsICJGbHV4X0JfZXJyIjogZXJyQiwKICAgICAgICAiWF9WIjogeHlbOiwgMF0sICJZX1YiOiB4eVs6LCAxXSwgIkZXSE1fViI6IHdpZHRoViwKICAgICAgICAiRmx1eF9WIjogZmx1eFYsICJGbHV4X1ZfZXJyIjogZXJyViwKICAgICAgICAiTWFnX2luc3RfQiI6IG1hZ0IsICJNYWdfaW5zdF9CX2VyciI6IG1hZ0JfZXJyLAogICAgICAgICJNYWdfaW5zdF9WIjogbWFnViwgIk1hZ19pbnN0X1ZfZXJyIjogbWFnVl9lcnIsCiAgICAgICAgIlBlYWtfQiI6IHBlYWtCLCAiUGVha19WIjogcGVha1YsCiAgICAgICAgIlNhdHVyYXRlZCI6IChwZWFrQiA+IGluZm9bIkIiXVsic2F0Il0pIHwgKHBlYWtWID4gaW5mb1siViJdWyJzYXQiXSksCiAgICAgICAgIkZvdW5kX2luIjogZm91bmRfaW4sICJBcGVydHVyZV9SYWRpdXMiOiByX2FwLAogICAgfSkKICAgIG5fYWxsID0gbGVuKGRmKQogICAgZGYgPSBkZlsoZGZbIkZsdXhfQiJdID4gMCkgJiAoZGZbIkZsdXhfViJdID4gMCldLnJlc2V0X2luZGV4KGRyb3A9VHJ1ZSkKICAgIHByaW50KGYiICAge2xlbihkZik6LH0gb2Yge25fYWxsOix9IGhhdmUgcG9zaXRpdmUgZmx1eCBpbiBib3RoIGJhbmRzOyAiCiAgICAgICAgICBmIntpbnQoZGZbJ1NhdHVyYXRlZCddLnN1bSgpKTosfSBvZiB0aGVtIHNhdHVyYXRlZCIsIGZsdXNoPVRydWUpCiAgICBpZiBsZW4oZGYpIDwgMjoKICAgICAgICBzdG9wKCJmZXdlciB0aGFuIHR3byBzb3VyY2VzIGhhdmUgcG9zaXRpdmUgZmx1eCBpbiBib3RoIGJhbmRzIikKICAgIG5lYXJlc3QgPSBjS0RUcmVlKGRmW1siWF9WIiwgIllfViJdXS52YWx1ZXMpLnF1ZXJ5KGRmW1siWF9WIiwgIllfViJdXS52YWx1ZXMsIGs9MilbMF1bOiwgMV0KICAgIGRmWyJOZWFyZXN0X3B4Il0gPSBuZWFyZXN0CgogICAgIyAtLS0tIHRoZSBwbGF0ZSBzb2x1dGlvbiwgY2hlY2tlZCwgdGhlbiB0aGUgc2t5IHBvc2l0aW9ucyAtLS0tLS0tLS0tLS0tLS0KICAgIGJyaWdodCA9IGRmW35kZlsiU2F0dXJhdGVkIl1dLnNvcnRfdmFsdWVzKCJGbHV4X1YiLCBhc2NlbmRpbmc9RmFsc2UpCiAgICB3Y3NfViwgd2NzX25vdGUgPSB2ZXJpZnlfcGxhdGVfc29sdXRpb24oYnJpZ2h0W1siWF9WIiwgIllfViJdXS50b19udW1weShmbG9hdCksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZGZbWyJYX1YiLCAiWV9WIl1dLnRvX251bXB5KGZsb2F0KSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3Y3NfViwgc2hhcGVfVikKICAgIHJhLCBkZWMgPSB3Y3NfVi5hbGxfcGl4MndvcmxkKGRmWyJYX1YiXS52YWx1ZXMsIGRmWyJZX1YiXS52YWx1ZXMsIDApCiAgICBkZlsiUkEiXSwgZGZbIkRFQyJdID0gcmEsIGRlYwogICAgaWYgIlJFUEFJUkVEIiBpbiB3Y3Nfbm90ZToKICAgICAgICAjIFRoZSBnYWxheHkncyBjZW50cmUgd2FzIHBsYWNlZCB0aHJvdWdoIHRoZSBvbGQgc29sdXRpb24uCiAgICAgICAgY2VudHJlX3B4ID0gZ2FsYXh5X2NlbnRyZV9waXhlbChvYmpfbmFtZSwgd2NzX1YsIHNoYXBlX1YpCiAgICAgICAgIyBBbmQgdGhlIFYgZnJhbWUgd3JpdHRlbiB0byB0aGUgb3V0cHV0IGNhcnJpZXMgdGhlIHJlcGFpcmVkIHNvbHV0aW9uLCBzbwogICAgICAgICMgdGhhdCBhIHRvb2wgcmVhZGluZyBwb3NpdGlvbnMgdGhyb3VnaCBpdCBsYXRlciAodGhlIGNvbXBhcmlzb24gd2l0aCB0aGUKICAgICAgICAjIEhJSSByZWdpb25zKSBwdXRzIHRoZW0gd2hlcmUgdGhleSBhcmUuCiAgICAgICAgd2l0aCBmaXRzLm9wZW4ocGF0aF9WLCBtb2RlPSJ1cGRhdGUiKSBhcyBoZHVsOgogICAgICAgICAgICBoID0gaGR1bFswXS5oZWFkZXIKICAgICAgICAgICAgZm9yIGtleSBpbiBsaXN0KGgua2V5cygpKToKICAgICAgICAgICAgICAgIGlmIHJlLm1hdGNoKHIiXihDVFlQRXxDUlZBTHxDUlBJWHxDREVMVHxDVU5JVHxDRFxkX1xkfFBDXGRfXGR8IgogICAgICAgICAgICAgICAgICAgICAgICAgICAgciJQVlxkX1xkK3xBX3xCX3xBUF98QlBffExPTlBPTEV8TEFUUE9MRXxXQ1NBWEVTfCIKICAgICAgICAgICAgICAgICAgICAgICAgICAgIHIiUkFERVNZU3xFUVVJTk9YKSIsIGtleSk6CiAgICAgICAgICAgICAgICAgICAgaC5yZW1vdmUoa2V5LCBpZ25vcmVfbWlzc2luZz1UcnVlLCByZW1vdmVfYWxsPVRydWUpCiAgICAgICAgICAgIGgudXBkYXRlKHdjc19WLnRvX2hlYWRlcihyZWxheD1UcnVlKSkKICAgICAgICAgICAgaFsiSElTVE9SWSJdID0gIlBsYXRlIHNvbHV0aW9uIHJlcGFpcmVkIGFnYWluc3QgR2FpYSBEUjMgYnkgQmx1ZV9DbHVzdGVycyIKCiAgICBjc3ZfZmlsZW5hbWUgPSBvcy5wYXRoLmpvaW4ob3V0X2RpciwgZiJ7c2FmZX1fcGhvdG9tZXRyeV9yZXN1bHRzLmNzdiIpCiAgICBkZi50b19jc3YoY3N2X2ZpbGVuYW1lLCBpbmRleD1GYWxzZSkKICAgIHByaW50KGYiSW5zdHJ1bWVudGFsIHBob3RvbWV0cnkgc2F2ZWQgdG8ge2Nzdl9maWxlbmFtZX0iKQoKICAgICMgLS0tLSBjYWxpYnJhdGlvbiAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQogICAgY2FsLCBjYXRhbG9ndWUsIGNhbF9zdGF0cywgbWF0Y2hlZCA9IGNhbGlicmF0ZSgKICAgICAgICBkZiwgd2NzX1YsIHNoYXBlX1YsIEFSR1MuY2F0YWxvZywgY2VudHJlX3B4LCByMjUsIG91dF9kaXIsIHNhZmUpCiAgICBkZiA9IGFwcGx5X2NhbGlicmF0aW9uKGRmLCBjYWwpCgogICAgIyBUaGUgZHVzdCBvZiBvdXIgb3duIGdhbGF4eSB0YWtlbiBvdXQuIEFfViBkaW1zIFY7IEIgaXMgZGltbWVkIGJ5CiAgICAjIEFfViArIEUoQi1WKSwgd2hpY2ggaXMgd2hhdCB0aGUgY29sb3VyIGV4Y2VzcyBtZWFucy4KICAgIGRmWyJNYWdfQiJdID0gZGZbIk1hZ19CX29icyJdIC0gKEFfViArIEVfQlYpCiAgICBkZlsiTWFnX1YiXSA9IGRmWyJNYWdfVl9vYnMiXSAtIEFfVgogICAgZGZbIk1fViJdID0gZGZbIk1hZ19WIl0gLSBtdQogICAgZGZbIlpQX1ZfdXNlZCJdLCBkZlsia19WX3VzZWQiXSA9IGNhbFsiWlBfViJdLCBjYWxbImtfViJdCiAgICBkZlsiQ29sb3VyX2FfdXNlZCJdLCBkZlsiQ29sb3VyX2NfdXNlZCJdID0gY2FsWyJhIl0sIGNhbFsiYyJdCiAgICBkZlsiTl9yZWZfdXNlZCJdID0gY2FsWyJOX3VzZWQiXQogICAgZGZbIkNhbGlicmF0aW9uX2NhdGFsb2d1ZSJdID0gY2F0YWxvZ3VlCiAgICBkZlsiQV9WX3VzZWQiXSwgZGZbIkVfQlZfdXNlZCJdID0gQV9WLCBFX0JWCiAgICBkZlsiRGlzdGFuY2VfTXBjX3VzZWQiXSwgZGZbIkRpc3RhbmNlX21vZHVsdXNfdXNlZCJdID0gZGlzdGFuY2UsIG11CiAgICBkZlsiRldITV9tZWFzdXJlZF9CIl0sIGRmWyJGV0hNX21lYXN1cmVkX1YiXSA9IGZ3aG1fQiwgZndobV9WCgogICAgY2FsaWJfZmlsZSA9IG9zLnBhdGguam9pbihvdXRfZGlyLCBmIntzYWZlfV9jYWxpYnJhdGVkX3Bob3RvbWV0cnkuY3N2IikKICAgIGRmLnRvX2NzdihjYWxpYl9maWxlLCBpbmRleD1GYWxzZSkKICAgIHByaW50KGYiQ2FsaWJyYXRlZCBwaG90b21ldHJ5IHNhdmVkIHRvIHtjYWxpYl9maWxlfSIpCgogICAgc291cmNlX21hcChkYXRhX1YsIHh5WzosIDBdLCB4eVs6LCAxXSwgZiJ7b2JqX25hbWV9IC0gViBiYW5kIHdpdGggbWVhc3VyZWQgc291cmNlcyIsCiAgICAgICAgICAgICAgIG9zLnBhdGguam9pbihvdXRfZGlyLCBmIntzYWZlfV9WX3NvdXJjZXMucG5nIikpCiAgICBzb3VyY2VfbWFwKGRhdGFfQiwgeHlfaW5fQls6LCAwXSwgeHlfaW5fQls6LCAxXSwKICAgICAgICAgICAgICAgZiJ7b2JqX25hbWV9IC0gQiBiYW5kIHdpdGggbWVhc3VyZWQgc291cmNlcyIsCiAgICAgICAgICAgICAgIG9zLnBhdGguam9pbihvdXRfZGlyLCBmIntzYWZlfV9CX3NvdXJjZXMucG5nIiksIGNvbG91cj0iYmx1ZSIpCgogICAgIyAtLS0tIGZvcmVncm91bmQgc3RhcnMgb3V0IC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQogICAgIyBFdmVyeSBzb3VyY2UgdGhhdCBpcyBhIGNhdGFsb2d1ZSBzdGFyIChtYXRjaGVkIHdpdGhpbiBDQUxJQl9NQVRDSF9SQURJVVMpLAogICAgIyBhbmQgZXZlcnkgc291cmNlIG9uIGEgR2FpYSBzdGFyIHdpdGggYSBzaWduaWZpY2FudCBwYXJhbGxheCBvciBwcm9wZXIKICAgICMgbW90aW9uLCBpcyBpbiBmcm9udCBvZiB0aGUgZ2FsYXh5LgogICAgaXNfc3RhciA9IG5wLnplcm9zKGxlbihkZiksIGR0eXBlPWJvb2wpCiAgICBpZiBsZW4obWF0Y2hlZCk6CiAgICAgICAgaXNfc3RhclttYXRjaGVkWyJyb3ciXS5hc3R5cGUoaW50KS52YWx1ZXNdID0gVHJ1ZQogICAgbl9jYXRfc3RhcnMgPSBpbnQoaXNfc3Rhci5zdW0oKSkKICAgIG5fZ2FpYSA9IDAKICAgIHRyeToKICAgICAgICBmZywgX2ZnZGlhZyA9IGdhaWFfZm9yZWdyb3VuZF9zdGFycyh3Y3NfViwgc2hhcGVfVikKICAgICAgICBpZiBmZyBpcyBub3QgTm9uZSBhbmQgbGVuKGZnKToKICAgICAgICAgICAgZCwgXyA9IGNLRFRyZWUobnAuYXNhcnJheShmZywgZmxvYXQpKS5xdWVyeShkZltbIlhfViIsICJZX1YiXV0udmFsdWVzLCBrPTEpCiAgICAgICAgICAgIG9uX2ZnID0gKGQgPD0gRk9SRUdST1VORF9UT0wpICYgfmlzX3N0YXIKICAgICAgICAgICAgbl9nYWlhID0gaW50KG9uX2ZnLnN1bSgpKQogICAgICAgICAgICBpc19zdGFyIHw9IG9uX2ZnCiAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGV4YzoKICAgICAgICBwcmludChmIiAgICBjb3VsZCBub3QgY2hlY2sgYWdhaW5zdCBHYWlhICh7ZXhjfSk7IGZvcmVncm91bmQgc3RhcnMgb3RoZXIgdGhhbiAiCiAgICAgICAgICAgICAgZiJ0aGUgY2F0YWxvZ3VlIHN0YXJzIHJlbWFpbiBpbiB0aGUgY2F0YWxvZ3VlIiwgZmx1c2g9VHJ1ZSkKICAgIGRmX2NsZWFuID0gZGZbfmlzX3N0YXJdLnJlc2V0X2luZGV4KGRyb3A9VHJ1ZSkKICAgIHByaW50KGYiICAgIHtuX2NhdF9zdGFyc30gc291cmNlcyBhcmUgY2F0YWxvZ3VlIHN0YXJzIGFuZCB7bl9nYWlhfSBtb3JlIHNpdCBvbiBhICIKICAgICAgICAgIGYiR2FpYSBmb3JlZ3JvdW5kIHN0YXIgKHBhcmFsbGF4IG9yIHByb3BlciBtb3Rpb24gc2lnbmlmaWNhbnQpOyAiCiAgICAgICAgICBmIntsZW4oZGZfY2xlYW4pfSBsZWZ0IiwgZmx1c2g9VHJ1ZSkKICAgIG5fb3V0c2lkZSA9IDAKICAgIGlmIHIyNSBhbmQgY2VudHJlX3B4IGlzIG5vdCBOb25lOgogICAgICAgIGQgPSBucC5oeXBvdChkZl9jbGVhblsiWF9WIl0gLSBjZW50cmVfcHhbMF0sIGRmX2NsZWFuWyJZX1YiXSAtIGNlbnRyZV9weFsxXSkKICAgICAgICBuX291dHNpZGUgPSBpbnQoKGQgPiByMjUpLnN1bSgpKQogICAgICAgIGRmX2NsZWFuID0gZGZfY2xlYW5bKGQgPD0gcjI1KS52YWx1ZXNdLnJlc2V0X2luZGV4KGRyb3A9VHJ1ZSkKICAgICAgICBwcmludChmIiAgICB7bl9vdXRzaWRlfSBzb3VyY2VzIGxpZSBiZXlvbmQgUjI1ICh7cjI1Oi4wZn0gcHgpLCBvdXRzaWRlIHRoZSAiCiAgICAgICAgICAgICAgZiJnYWxheHksIGFuZCBhcmUgZHJvcHBlZDsge2xlbihkZl9jbGVhbil9IGxlZnQiLCBmbHVzaD1UcnVlKQogICAgZWxzZToKICAgICAgICBwcmludCgiICAgIG5vIGdhbGF4eSByYWRpdXMgYXZhaWxhYmxlOyB0aGUgY2F0YWxvZ3VlIGlzIG5vdCB0cmltbWVkIHRvIHRoZSAiCiAgICAgICAgICAgICAgImdhbGF4eSIsIGZsdXNoPVRydWUpCgogICAgb3V0X2ZpbGUgPSBjYWxpYl9maWxlLnJlcGxhY2UoIl9jYWxpYnJhdGVkX3Bob3RvbWV0cnkuY3N2IiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICJfY2FsaWJyYXRlZF9waG90b21ldHJ5X25vX3N0YXJzLmNzdiIpCiAgICBkZl9jbGVhbi50b19jc3Yob3V0X2ZpbGUsIGluZGV4PUZhbHNlKQogICAgcHJpbnQoZiJDbGVhbmVkIGZpbGUgc2F2ZWQgdG8ge291dF9maWxlfSIpCiAgICBzb3VyY2VfbWFwKGRhdGFfViwgZGZfY2xlYW5bIlhfViJdLCBkZl9jbGVhblsiWV9WIl0sCiAgICAgICAgICAgICAgIGYie29ial9uYW1lfSAtIFYgYmFuZCB3aXRoIGRldGVjdGVkIHNvdXJjZXMgKG5vIHN0YXJzKSIsCiAgICAgICAgICAgICAgIG9zLnBhdGguam9pbihvdXRfZGlyLCBmIntzYWZlfV9WX3NvdXJjZXNfbm9fc3RhcnMucG5nIiksCiAgICAgICAgICAgICAgIGxhYmVsPSJDbGVhbmVkIHNvdXJjZXMgKG5vIHN0YXJzKSIpCgogICAgbWFrZV9jbWQoZGYsIGNhbGliX2ZpbGUsICIoYWxsIHNvdXJjZXMpIiwgb2JqX25hbWUsIG11KQogICAgbWFrZV9jbWQoZGZfY2xlYW4sIG91dF9maWxlLCAiKG5vIHN0YXJzKSIsIG9ial9uYW1lLCBtdSwKICAgICAgICAgICAgIHdpbmRvdz0oQ01EX0NPTE9SX01JTiwgQ01EX0NPTE9SX01BWCkpCgogICAgIyAtLS0tIHRoZSBibHVlIGtub3RzIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCiAgICBidjAgPSBkZl9jbGVhblsiTWFnX0IiXSAtIGRmX2NsZWFuWyJNYWdfViJdCiAgICBpbl93aW5kb3cgPSAoYnYwID49IENNRF9DT0xPUl9NSU4pICYgKGJ2MCA8PSBDTURfQ09MT1JfTUFYKQogICAgcHJlY2lzZSA9IGRmX2NsZWFuWyJCVl9lcnIiXSA8PSBBUkdTLm1heF9idl9lcnIKICAgIHVuc2F0ID0gfmRmX2NsZWFuWyJTYXR1cmF0ZWQiXS5hc3R5cGUoYm9vbCkKICAgIGRmX2NvbG9yX2ZpbHRlcmVkID0gZGZfY2xlYW5baW5fd2luZG93ICYgcHJlY2lzZSAmIHVuc2F0XS5jb3B5KCkKICAgIGRmX2NvbG9yX2ZpbHRlcmVkWyJCLVYiXSA9IGJ2MFtpbl93aW5kb3cgJiBwcmVjaXNlICYgdW5zYXRdCiAgICBkZl9jb2xvcl9maWx0ZXJlZCA9IGRmX2NvbG9yX2ZpbHRlcmVkLnJlc2V0X2luZGV4KGRyb3A9VHJ1ZSkKICAgIHByaW50KGYiQmx1ZSBrbm90czoge2xlbihkZl9jb2xvcl9maWx0ZXJlZCl9IHNvdXJjZXMgd2l0aCB7Q01EX0NPTE9SX01JTn0gPD0gIgogICAgICAgICAgZiIoQi1WKTAgPD0ge0NNRF9DT0xPUl9NQVh9IGFuZCBhIEItViBlcnJvciBvZiBhdCBtb3N0IHtBUkdTLm1heF9idl9lcnJ9IG1hZyIKICAgICAgICAgIGYiICAoe2ludCgoaW5fd2luZG93ICYgfnByZWNpc2UpLnN1bSgpKX0gbW9yZSBpbiB0aGUgY29sb3VyIHdpbmRvdyB3ZXJlIHRvbyAiCiAgICAgICAgICBmIm5vaXN5IHRvIGNsYXNzaWZ5LCB7aW50KChpbl93aW5kb3cgJiB+dW5zYXQpLnN1bSgpKX0gc2F0dXJhdGVkKSIsIGZsdXNoPVRydWUpCiAgICBvdXRfY3N2X2NvbG9yID0gb3MucGF0aC5qb2luKG91dF9kaXIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGYie3NhZmV9X2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFyc19jb2xvcl9maWx0ZXJlZC5jc3YiKQogICAgZGZfY29sb3JfZmlsdGVyZWQudG9fY3N2KG91dF9jc3ZfY29sb3IsIGluZGV4PUZhbHNlKQogICAgcHJpbnQoZiJDb2xvci1maWx0ZXJlZCBwaG90b21ldHJ5IHNhdmVkIHRvIHtvdXRfY3N2X2NvbG9yfSIpCiAgICBzb3VyY2VfbWFwKGRhdGFfViwgZGZfY29sb3JfZmlsdGVyZWRbIlhfViJdLCBkZl9jb2xvcl9maWx0ZXJlZFsiWV9WIl0sCiAgICAgICAgICAgICAgIGYie29ial9uYW1lfSAtIFYgYmFuZCB3aXRoIHRoZSBibHVlIGtub3RzIiwKICAgICAgICAgICAgICAgb3MucGF0aC5qb2luKG91dF9kaXIsIGYie3NhZmV9X1Zfc291cmNlc19jb2xvcl9maWx0ZXJlZC5wbmciKSwKICAgICAgICAgICAgICAgbGFiZWw9ZiJCbHVlIGtub3RzICh7Q01EX0NPTE9SX01JTn0gPD0gKEItVikwIDw9IHtDTURfQ09MT1JfTUFYfSkiKQogICAgbWFrZV9jbWQoZGZfY29sb3JfZmlsdGVyZWQsIG91dF9jc3ZfY29sb3IsICIoYmx1ZSBrbm90cykiLCBvYmpfbmFtZSwgbXUsCiAgICAgICAgICAgICB3aW5kb3c9KENNRF9DT0xPUl9NSU4sIENNRF9DT0xPUl9NQVgpKQoKICAgIHdyaXRlX3N1bW1hcnkob3MucGF0aC5qb2luKG91dF9kaXIsICJydW5fc3VtbWFyeS50eHQiKSwgbG9jYWxzKCkpCgogICAgaWYgZGZfY29sb3JfZmlsdGVyZWQuZW1wdHk6CiAgICAgICAgcHJpbnQoIk5vIGNvbG9yLWZpbHRlcmVkIHNvdXJjZXMgYXZhaWxhYmxlIHRvIGNvbXB1dGUgcmFkaWFsIGRpc3RhbmNlcy4iKQogICAgICAgIHJldHVybiAwCgogICAgIyAtLS0tIHJhZGlhbCBwcm9maWxlIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCiAgICBpZiBjZW50cmVfcHggaXMgTm9uZToKICAgICAgICBjZW50cmVfcHggPSAoc2hhcGVfVlsxXSAvIDIuMCwgc2hhcGVfVlswXSAvIDIuMCkKICAgICAgICBwcmludCgiW3Byb2ZpbGVdIHRoZSBnYWxheHkncyBjZW50cmUgaXMgdW5rbm93biAtIHVzaW5nIHRoZSBmcmFtZSdzIGNlbnRyZSIpCiAgICB4X2NlbnRlciwgeV9jZW50ZXIgPSBjZW50cmVfcHgKICAgIGRmX2NvbG9yX2ZpbHRlcmVkWyJSYWRpYWxfRGlzdGFuY2VfcHgiXSA9IG5wLmh5cG90KAogICAgICAgIGRmX2NvbG9yX2ZpbHRlcmVkWyJYX1YiXSAtIHhfY2VudGVyLCBkZl9jb2xvcl9maWx0ZXJlZFsiWV9WIl0gLSB5X2NlbnRlcikKICAgIHB4X3RvX3BjID0gbnAuZGVnMnJhZChzY2FsZV9hcmNzZWMgLyAzNjAwLjApICogZGlzdGFuY2UgKiAxLjBlNgogICAgZGZfY29sb3JfZmlsdGVyZWRbIlJhZGlhbF9EaXN0YW5jZV9wYyJdID0gZGZfY29sb3JfZmlsdGVyZWRbIlJhZGlhbF9EaXN0YW5jZV9weCJdICogcHhfdG9fcGMKICAgIGRmX2NvbG9yX2ZpbHRlcmVkID0gZGZfY29sb3JfZmlsdGVyZWQuc29ydF92YWx1ZXMoIlJhZGlhbF9EaXN0YW5jZV9weCIpLnJlc2V0X2luZGV4KGRyb3A9VHJ1ZSkKICAgIG91dF9jc3ZfY29sb3JfcmFkID0gb3MucGF0aC5qb2luKAogICAgICAgIG91dF9kaXIsIGYie3NhZmV9X2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFyc19jb2xvcl9maWx0ZXJlZF93aXRoX3JhZGl1cy5jc3YiKQogICAgZGZfY29sb3JfZmlsdGVyZWQudG9fY3N2KG91dF9jc3ZfY29sb3JfcmFkLCBpbmRleD1GYWxzZSkKICAgIHByaW50KGYiQ29sb3ItZmlsdGVyZWQgcGhvdG9tZXRyeSB3aXRoIHJhZGlhbCBkaXN0YW5jZXMgc2F2ZWQgdG8ge291dF9jc3ZfY29sb3JfcmFkfSIpCgogICAgZm9yIHVuaXQsIHNjYWxlX3UsIGNvbG91ciBpbiAoKCJweCIsIDEuMCwgImJsdWUiKSwgKCJwYyIsIHB4X3RvX3BjLCAiZ3JlZW4iKSk6CiAgICAgICAgcmFkaWkgPSBkZl9jb2xvcl9maWx0ZXJlZFsiUmFkaWFsX0Rpc3RhbmNlX3B4Il0gKiBzY2FsZV91CiAgICAgICAgd2lkdGggPSBTdGVwX3NpemUgKiBzY2FsZV91CiAgICAgICAgZWRnZXMgPSBucC5hcmFuZ2UoMCwgcmFkaWkubWF4KCkgKyB3aWR0aCwgd2lkdGgpCiAgICAgICAgY291bnRzLCBlZGdlcyA9IG5wLmhpc3RvZ3JhbShyYWRpaSwgYmlucz1lZGdlcykKICAgICAgICBhcmVhcyA9IG5wLnBpICogKGVkZ2VzWzE6XSAqKiAyIC0gZWRnZXNbOi0xXSAqKiAyKQogICAgICAgIGRlbnMgPSBjb3VudHMgLyBhcmVhcwogICAgICAgIHN1ZmZpeCA9ICIiIGlmIHVuaXQgPT0gInB4IiBlbHNlICJfcGMiCiAgICAgICAgcGQuRGF0YUZyYW1lKHtmIlJfaW5uZXJfe3VuaXR9IjogZWRnZXNbOi0xXSwgZiJSX291dGVyX3t1bml0fSI6IGVkZ2VzWzE6XSwKICAgICAgICAgICAgICAgICAgICAgIGYiUl9jZW50ZXJfe3VuaXR9IjogMC41ICogKGVkZ2VzWzE6XSArIGVkZ2VzWzotMV0pLAogICAgICAgICAgICAgICAgICAgICAgIk5fc291cmNlcyI6IGNvdW50cywgZiJBbm51bHVzX2FyZWFfe3VuaXR9MiI6IGFyZWFzLAogICAgICAgICAgICAgICAgICAgICAgZiJEZW5zaXR5X3Blcl97dW5pdH0yIjogZGVuc30pLnRvX2NzdigKICAgICAgICAgICAgb3MucGF0aC5qb2luKG91dF9kaXIsIGYie3NhZmV9X3JhZGlhbF9kZW5zaXR5X3Byb2ZpbGV7c3VmZml4fS5jc3YiKSwgaW5kZXg9RmFsc2UpCiAgICAgICAgc3RlcF9yID0gbnAuY29sdW1uX3N0YWNrKFtlZGdlc1s6LTFdLCBlZGdlc1sxOl1dKS5yYXZlbCgpCiAgICAgICAgc3RlcF9kID0gbnAucmVwZWF0KGRlbnMsIDIpCiAgICAgICAgc3RlcF9uID0gbnAucmVwZWF0KGNvdW50cywgMikKICAgICAgICBwZC5EYXRhRnJhbWUoe2YiUl9zdGVwX3t1bml0fSI6IHN0ZXBfciwgZiJEZW5zaXR5X3N0ZXBfcGVyX3t1bml0fTIiOiBzdGVwX2QsCiAgICAgICAgICAgICAgICAgICAgICAiTl9zb3VyY2VzX3N0ZXAiOiBzdGVwX259KS50b19jc3YoCiAgICAgICAgICAgIG9zLnBhdGguam9pbihvdXRfZGlyLCBmIntzYWZlfV9yYWRpYWxfZGVuc2l0eV9wcm9maWxlX3N0ZXB7c3VmZml4fS5jc3YiKSwKICAgICAgICAgICAgaW5kZXg9RmFsc2UpCiAgICAgICAgcGx0LmZpZ3VyZShmaWdzaXplPSg4LCA2KSkKICAgICAgICBwbHQuc3RlcChzdGVwX3IsIHN0ZXBfZCwgd2hlcmU9InBvc3QiLCBjb2xvcj1jb2xvdXIsIGxpbmV3aWR0aD0yKQogICAgICAgIHBsdC54bGFiZWwoIlJhZGlhbCBkaXN0YW5jZSBbcGl4ZWxzXSIgaWYgdW5pdCA9PSAicHgiIGVsc2UgIlJhZGlhbCBkaXN0YW5jZSBbcGNdIikKICAgICAgICBwbHQueWxhYmVsKCJTb3VyY2UgZGVuc2l0eSBbMS9waXhlbMKyXSIgaWYgdW5pdCA9PSAicHgiIGVsc2UgIlNvdXJjZSBkZW5zaXR5IFsxL3BjwrJdIikKICAgICAgICBwbHQudGl0bGUoZiJSYWRpYWwgZGVuc2l0eSBzdGVwIHByb2ZpbGUgb2Yge29ial9uYW1lfSAiCiAgICAgICAgICAgICAgICAgIGYiKHsncGl4ZWxzJyBpZiB1bml0ID09ICdweCcgZWxzZSAncGMnfSkiKQogICAgICAgIHBuZyA9IG9zLnBhdGguam9pbihvdXRfZGlyLCBmIntzYWZlfV9yYWRpYWxfZGVuc2l0eV9wcm9maWxlX3N0ZXB7c3VmZml4fS5wbmciKQogICAgICAgIHBsdC5zYXZlZmlnKHBuZywgZHBpPTE1MCkKICAgICAgICBwbHQuY2xvc2UoKQogICAgICAgIHByaW50KGYiUmFkaWFsIGRlbnNpdHkgc3RlcCBwcm9maWxlICh7dW5pdH0pIHNhdmVkIHRvIHtwbmd9IikKCiAgICAjID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CiAgICAjIFN0YWdlIDYgLSBkZXByb2plY3Rpb24gYW5kIHRoZSBzaGFyZWQgcmFkaWFsIGFuYWx5c2lzCiAgICAjID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CiAgICAjCiAgICAjIEEgc3BpcmFsIGdhbGF4eSBpcyBhIGZsYXQgZGlzayBzZWVuIGF0IGFuIGFuZ2xlLCBzbyBhIGRpc3RhbmNlIG1lYXN1cmVkIGluCiAgICAjIHRoZSBpbWFnZSBpcyBub3QgdGhlIGRpc3RhbmNlIGluc2lkZSB0aGUgZ2FsYXh5LiBUaGlzIHN0YWdlIHN0cmV0Y2hlcyB0aGUKICAgICMgc3F1YXNoZWQgZGlyZWN0aW9uIGJhY2sgb3V0IGFuZCByZWNvbXB1dGVzIHRoZSBwcm9maWxlIHNldmVyYWwgd2F5cyAtCiAgICAjIHVuY29ycmVjdGVkLCBmcm9tIHRoZSBIeXBlckxFREEgY2F0YWxvZ3VlLCBhbmQgZnJvbSB0aGUga25vdHMgdGhlbXNlbHZlcyAtCiAgICAjIGJlY2F1c2Ugbm8gZGVwcm9qZWN0aW9uIG1ldGhvZCBoYXMgYmVlbiBmb3VuZCByZWxpYWJseSBiZXR0ZXIgdGhhbiB0aGUKICAgICMgb3RoZXJzLiBUaGUgc3ByZWFkIGJldHdlZW4gdGhlIGN1cnZlcyBpcyB0aGUgdW5jZXJ0YWludHkgb24gdGhlIHJhZGlpLiBJdAogICAgIyBhbHNvIHVzZXMgdGhlIHJpbmcgYXJlYSBhY3R1YWxseSBjb3ZlcmVkIGJ5IHRoZSBmcmFtZS4KICAgIHRyeToKICAgICAgICBub3J0aCwgbWlycm9yZWQgPSBpbWFnZV9vcmllbnRhdGlvbih3Y3NfVikKICAgICAgICBwcmludChmIltkZXByb2plY3RdIG9yaWVudGF0aW9uIGZyb20gV0NTOiBub3J0aCB7bm9ydGg6LjFmfSBkZWcgZnJvbSAreCwgIgogICAgICAgICAgICAgIGYieydtaXJyb3JlZCcgaWYgbWlycm9yZWQgZWxzZSAnbm90IG1pcnJvcmVkJ30iKQogICAgICAgIHByaW50KCIiKQogICAgICAgIHByaW50KCJEZXByb2plY3RpbmcgYW5kIHJlYnVpbGRpbmcgdGhlIHJhZGlhbCBwcm9maWxlLi4uIikKICAgICAgICBydW5fcHJvZmlsZXMoCiAgICAgICAgICAgIGRmX2NvbG9yX2ZpbHRlcmVkWyJYX1YiXS50b19udW1weShkdHlwZT1mbG9hdCksCiAgICAgICAgICAgIGRmX2NvbG9yX2ZpbHRlcmVkWyJZX1YiXS50b19udW1weShkdHlwZT1mbG9hdCksCiAgICAgICAgICAgICh4X2NlbnRlciwgeV9jZW50ZXIpLCAoMC4wLCBmbG9hdChzaGFwZV9WWzFdKSwgMC4wLCBmbG9hdChzaGFwZV9WWzBdKSksCiAgICAgICAgICAgIG9zLnBhdGguam9pbihvdXRfZGlyLCBmIntzYWZlfV9kZXByb2plY3RlZCIpLAogICAgICAgICAgICBmIkJsdWUga25vdHMgLSB7b2JqX25hbWV9IiwgZ2FsYXh5PW9ial9uYW1lLAogICAgICAgICAgICBub3J0aF9hbmdsZT1ub3J0aCwgbWlycm9yZWQ9bWlycm9yZWQsIHNlY3Rvcl9tZXRob2Q9VHJ1ZSwKICAgICAgICAgICAgcmluZ3M9MjAsIHNjYWxlPXB4X3RvX3BjLCB1bml0PSJwYyIpCiAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGV4YzoKICAgICAgICBwcmludChmIltkZXByb2plY3RdIHRoZSBkZXByb2plY3Rpb24gc3RhZ2UgZmFpbGVkICh7dHlwZShleGMpLl9fbmFtZV9ffTogIgogICAgICAgICAgICAgIGYie2V4Y30pLiBFdmVyeXRoaW5nIGFib3ZlIHdhcyBzdGlsbCB3cml0dGVuLiIpCiAgICByZXR1cm4gMAoKCmRlZiB3cml0ZV9zdW1tYXJ5KHBhdGgsIHYpOgogICAgIiIiRXZlcnkgbnVtYmVyIHRoaXMgcnVuIHVzZWQgYW5kIGZvdW5kLCBpbiBvbmUgZmlsZSBiZXNpZGUgaXRzIG91dHB1dC4iIiIKICAgIGNhbCwgc3QgPSB2WyJjYWwiXSwgdlsiY2FsX3N0YXRzIl0KICAgIGxpbmVzID0gWwogICAgICAgIGYiQmx1ZV9DbHVzdGVyc19Gcm9tX0ltYWdlcy5weSBydW4gc3VtbWFyeSIsCiAgICAgICAgZiJnYWxheHkgICAgICAgICAgICAgIHt2WydvYmpfbmFtZSddfSIsCiAgICAgICAgZiJCIGZyYW1lICAgICAgICAgICAgIHt2WydmaXRzX2ZpbGVfQiddfSIsCiAgICAgICAgZiJWIGZyYW1lICAgICAgICAgICAgIHt2WydmaXRzX2ZpbGVfViddfSIsCiAgICBdICsgW2YiaW5wdXQgICAgICAgICAgICAgICB7bn0iIGZvciBuIGluIHZbImNvbWJpbmVkX25vdGUiXV0gKyBbCiAgICAgICAgZiJkaXN0YW5jZSAgICAgICAgICAgIHt2WydkaXN0YW5jZSddOi4zZn0gTXBjIChkaXN0YW5jZSBtb2R1bHVzIHt2WydtdSddOi4zZn0pIiwKICAgICAgICBmIkUoQi1WKSAgICAgICAgICAgICAge3ZbJ0VfQlYnXTouNGZ9ICAgQV9WID0ge3ZbJ0FfViddOi4zZn0iLAogICAgICAgIGYicGxhdGUgc29sdXRpb24gICAgICB7dlsnd2NzX25vdGUnXX0iLAogICAgICAgIGYicmVnaXN0cmF0aW9uICAgICAgICBCIG9udG8gViB7dlsncmVnX3RleHQnXX0iLAogICAgICAgIGYic3RhciBGV0hNICAgICAgICAgICBCIHt2Wydmd2htX0InXTouMmZ9IHB4LCBWIHt2Wydmd2htX1YnXTouMmZ9IHB4IiwKICAgICAgICBmImFwZXJ0dXJlICAgICAgICAgICAge3ZbJ3JfYXAnXTouMmZ9IHB4IHJhZGl1cywgc2t5IHJpbmcge3ZbJ3JfaW4nXTouMWZ9LSIKICAgICAgICBmInt2WydyX291dCddOi4xZn0gcHgsIHRoZSBzYW1lIGluIGJvdGggYmFuZHMiLAogICAgICAgIGYic2F0dXJhdGlvbiAgICAgICAgICBCIHt2WydpbmZvJ11bJ0InXVsnc2F0J106LjBmfSBBRFUsIFYge3ZbJ2luZm8nXVsnViddWydzYXQnXTouMGZ9IEFEVSIsCiAgICAgICAgZiJnYWluICAgICAgICAgICAgICAgIEIge3ZbJ2luZm8nXVsnQiddWydnYWluJ106LjJmfSwgViB7dlsnaW5mbyddWydWJ11bJ2dhaW4nXTouMmZ9IGUtL0FEVSIsCiAgICAgICAgZiJzb3VyY2VzIG1lYXN1cmVkICAgIHtsZW4odlsnZGYnXSl9IChwb3NpdGl2ZSBmbHV4IGluIGJvdGggYmFuZHMpIiwKICAgICAgICBmImNhbGlicmF0aW9uICAgICAgICAge3ZbJ2NhdGFsb2d1ZSddfToge2NhbFsnTl91c2VkJ119IG9mIHtjYWxbJ05fY2FuZGlkYXRlcyddfSAiCiAgICAgICAgZiJzdGFycyB1c2VkICh7c3QuZ2V0KCdtYXRjaGVkJywgMCl9IG1hdGNoZWQpIiwKICAgICAgICBmIiAgIEItViA9IHtjYWxbJ2EnXTorLjRmfSArIHtjYWxbJ2MnXTouNGZ9IHggKGItdikgICBzY2F0dGVyIHtjYWxbJ3Jtc19jb2xvdXJfZml0J106LjNmfSIsCiAgICAgICAgZiIgICBWICAgPSB2ICsge2NhbFsnWlBfViddOi40Zn0ge2NhbFsna19WJ106Ky40Zn0geCAoQi1WKSAgIHNjYXR0ZXIge2NhbFsncm1zX1ZfZml0J106LjNmfSIsCiAgICAgICAgZiIgICBCICAgPSBiICsge2NhbFsnWlBfQiddOi40Zn0ge2NhbFsna19CJ106Ky40Zn0geCAoQi1WKSAgIHNjYXR0ZXIge2NhbFsncm1zX0JfZml0J106LjNmfSIsCiAgICAgICAgZiIgICBjb2xvdXIgY29ycmVsYXRpb24ge2NhbFsnY29ycmVsYXRpb24nXTorLjNmfSIsCiAgICAgICAgZiJmb3JlZ3JvdW5kIHJlbW92ZWQgIHt2WyduX2NhdF9zdGFycyddfSBjYXRhbG9ndWUgc3RhcnMsIHt2WyduX2dhaWEnXX0gR2FpYSAiCiAgICAgICAgZiJmb3JlZ3JvdW5kIHN0YXJzLCB7dlsnbl9vdXRzaWRlJ119IGJleW9uZCBSMjUiLAogICAgICAgIGYic291cmNlcyBsZWZ0ICAgICAgICB7bGVuKHZbJ2RmX2NsZWFuJ10pfSIsCiAgICAgICAgZiJibHVlIGtub3RzICAgICAgICAgIHtsZW4odlsnZGZfY29sb3JfZmlsdGVyZWQnXSl9ICAoe0NNRF9DT0xPUl9NSU59IDw9IChCLVYpMCA8PSAiCiAgICAgICAgZiJ7Q01EX0NPTE9SX01BWH0sIEItViBlcnJvciA8PSB7QVJHUy5tYXhfYnZfZXJyfSwgbm90IHNhdHVyYXRlZCkiLAogICAgXQogICAgd2l0aCBvcGVuKHBhdGgsICJ3IiwgZW5jb2Rpbmc9InV0Zi04IikgYXMgZmg6CiAgICAgICAgZmgud3JpdGUoIlxuIi5qb2luKGxpbmVzKSArICJcbiIpCiAgICBwcmludChmIlJ1biBzdW1tYXJ5IHNhdmVkIHRvIHtwYXRofSIpCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIHN5cy5leGl0KG1haW4oKSkK"


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
