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

_CLUSTER_TOOL_SOURCE = "IyAtKi0gY29kaW5nOiB1dGYtOCAtKi0KIiIiCkNyZWF0ZWQgYnk6IERyLiBCb2F6IFJvbiBab2hhcgpodHRwczovL2dpdGh1Yi5jb20vQm9helJvblpvaGFyL09ic2VydmF0aW9uYWxBc3Ryb25vbXlFZHVjYXRpb25Ub29scy9ibG9iL21haW4vR2FsYXh5X0FuYWx5c2lzL0JsdWVfQ2x1c3RlcnNfRnJvbV9JbWFnZXMucHkKQWZmaWxpYXRpb246IEtpbm5lcmV0IE9ic2VydmF0b3J5Ck1lbWJlciBvZiB0aGUgTENPIEdsb2JhbCBTa3kgUGFydG5lcnMgcHJvZ3JhbW1lCkRhdGU6IFNlcHRlbWJlciAyMDI1CgpXcml0dGVuIGZvciBzdHVkZW50IHByb2plY3RzIG9uIHNwaXJhbCBnYWxheGllcy4KCldIQVQgSVQgRE9FUwoKVGFrZXMgb25lIEItYmFuZCBhbmQgb25lIFYtYmFuZCBpbWFnZSBvZiBhIGdhbGF4eSBhbmQgcHJvZHVjZXMgaXRzCmNvbG91ci1tYWduaXR1ZGUgZGlhZ3JhbSwgdG9nZXRoZXIgd2l0aCB0aGUgcmFkaWFsIGRlbnNpdHkgcHJvZmlsZSBvZiB0aGUKYmx1ZSBrbm90cyAtIHRoZSB5b3VuZyBzdGFyLWZvcm1pbmcgcmVnaW9ucyBpbiB0aGUgYXJtcy4KClRoZSBjb2xvdXIgb2YgYSBzb3VyY2Ugc2F5cyBob3cgaG90IGFuZCBob3cgeW91bmcgaXQgaXMuIFBsb3R0aW5nIGNvbG91cgphZ2FpbnN0IGJyaWdodG5lc3MgZm9yIGV2ZXJ5IHNvdXJjZSBpbiB0aGUgZ2FsYXh5IHNlcGFyYXRlcyB0aGUgYmx1ZSBrbm90cwpmcm9tIHRoZSBvbGRlciBwb3B1bGF0aW9uLCBhbmQgdGhlIHJhZGlhbCBwcm9maWxlIHRoZW4gc2hvd3MgaG93IHRoZXkgYXJlCmRpc3RyaWJ1dGVkIHdpdGggZGlzdGFuY2UgZnJvbSB0aGUgY2VudHJlIC0gd2hpY2ggaXMgd2hhdCBhIHNwaXJhbCBhcm0KbG9va3MgbGlrZSBpbiBudW1iZXJzLgoKV0hFUkUgVEhFIE1FVEhPRCBDT01FUyBGUk9NCgpJZGVudGlmeWluZyBzdGFyLWZvcm1pbmcgcmVnaW9ucyBieSB0aGVpciBjb2xvdXIgaW4gb3JkaW5hcnkgYnJvYWQtYmFuZAppbWFnZXMsIGFuZCB0aGVuIHRyZWF0aW5nIHRoZWlyIGRpc3RyaWJ1dGlvbiBhcyBhIG1lYXN1cmFibGUgcHJvcGVydHkgb2YgdGhlCmdhbGF4eSwgZm9sbG93cwoKICAgIEJyb3NjaCwgTi4gKDE5OTIpLiBTdGFyIGZvcm1hdGlvbiBzeXN0ZW1hdGljcyBmcm9tIGNvbG91ciBpbWFnZXMuCiAgICBBc3Ryb3BoeXNpY3MgYW5kIFNwYWNlIFNjaWVuY2UsIDE4OCwgMjg5LTI5OC4KICAgIGRvaToxMC4xMDA3L0JGMDA2NDQ5MTYKClRoZSBhcHBlYWwgZm9yIHRlYWNoaW5nIGlzIHRoYXQgaXQgYXNrcyBub3RoaW5nIGV4b3RpYy4gVHdvIGJyb2FkLWJhbmQgZnJhbWVzCm9mIGEgZ2FsYXh5LCBCIGFuZCBWLCBhcmUgd2l0aGluIHJlYWNoIG9mIGEgc2Nob29sLWFjY2Vzc2libGUgdGVsZXNjb3BlLCBhbmQKdGhlIHlvdW5nIHJlZ2lvbnMgc2VwYXJhdGUgb3V0IG9uIGNvbG91ciBhbG9uZSAtIG5vIHNwZWN0cm9zY29weSwgbm8gbmFycm93LQpiYW5kIGZpbHRlci4gQSBzdHVkZW50IHdpdGggb25lIG5pZ2h0IG9mIGRhdGEgY2FuIGFzayBhIHJlYWwgcXVlc3Rpb24gYWJvdXQKd2hlcmUgYSBnYWxheHkgaXMgZm9ybWluZyBzdGFycywgYW5kIGFuc3dlciBpdCB3aXRoIGEgbnVtYmVyLgoKVGhpcyBzY3JpcHQgaXMgdGhlIHBpcGVsaW5lIGZvciB0aGF0LCBhbmQgaGFzIGJlZW4gdXNlZCBhY3Jvc3MgbWFueSBzdHVkZW50CnByb2plY3RzLgoKV0hBVCBZT1UgTkVFRAoKICBhIEIgYW5kIGEgViBmcmFtZSBvZiB0aGUgc2FtZSBnYWxheHkgKG9yIGEgZm9sZGVyIGhvbGRpbmcgc2V2ZXJhbCBvZiBlYWNoKSwKICBwbGF0ZS1zb2x2ZWQ6IHRoZSByZWZlcmVuY2Ugc3RhcnMgYXJlIHB1bGxlZCBmcm9tIGEgY2F0YWxvZ3VlIGJ5IHNreQogIHBvc2l0aW9uLiBUaGUgcGxhdGUgc29sdXRpb24gaXMgY2hlY2tlZCBhZ2FpbnN0IEdhaWEgYW5kIHJlcGFpcmVkIGlmIGl0CiAgZG9lcyBub3QgZml0IHRoZSBpbWFnZS4KCldIQVQgSVQgQVNLUyBZT1UKCiAgdGhlIGZvbGRlciBob2xkaW5nIHRoZSBmcmFtZXMgIHdoaWNoIGZpbGUgaXMgQiBhbmQgd2hpY2ggaXMgViBpcyByZWFkIGZyb20KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhlIEZJTFRFUiBrZXl3b3JkIG9yIHRoZSBmaWxlIG5hbWUsIGFuZCB0aGUKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZ2FsYXh5J3MgbmFtZSBmcm9tIE9CSkVDVAogIGRpc3RhbmNlIGluIE1wYyAgICAgICAgICAgICAgICBzaG93biBmcm9tIENvc21pY2Zsb3dzLTMgKFR1bGx5KyAyMDE2KTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgRW50ZXIgYWNjZXB0cyBpdC4gQ29udmVydHMgcGl4ZWxzIHRvIHBhcnNlY3MKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYW5kIGdpdmVzIHRoZSBhYnNvbHV0ZSBtYWduaXR1ZGUgTV9WCiAgY29sb3VyIGV4Y2VzcyBFKEItVikgICAgICAgICAgIHNob3duIGZyb20gdGhlIFNjaGxhZmx5ICYgRmlua2JlaW5lciAoMjAxMSkKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZHVzdCBtYXBzIHZpYSBJUlNBOyBFbnRlciBhY2NlcHRzIGl0LgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBBX1YgaXMgMy4xIHggRShCLVYpCgpFdmVyeXRoaW5nIGNhbiBhbHNvIGJlIGdpdmVuIG9uIHRoZSBjb21tYW5kIGxpbmUsIGZvciBleGFtcGxlCgogIHB5dGhvbiBCbHVlX0NsdXN0ZXJzX0Zyb21fSW1hZ2VzLnB5IC0tZm9sZGVyICJEOi9NODMiIC0teWVzCiAgcHl0aG9uIEJsdWVfQ2x1c3RlcnNfRnJvbV9JbWFnZXMucHkgLS1iIEIuZml0cyAtLXYgVi5maXRzIC0tZ2FsYXh5IE04MwogICAgICAtLWRpc3RhbmNlLW1wYyA0LjcgLS1lYnYgMC4wNiAtLWNhdGFsb2cgR0FJQQoKSE9XIElUIFdPUktTCgogIDEuIHN1YnRyYWN0cyB0aGUgYmFja2dyb3VuZCBmcm9tIGVhY2ggZnJhbWUsIG1lYXN1cmVzIGhvdyB3aWRlIGEgc3RhciBpcwogICAgIGluIGVhY2ggKGZyb20gYnJpZ2h0IHN0YXJzIHRoYXQgYXJlIG5vdCBzYXR1cmF0ZWQpLCBhbmQgZmluZHMgdGhlCiAgICAgc291cmNlcyBpbiBib3RoCiAgMi4gcmVnaXN0ZXJzIEIgb250byBWLCBhbmQgbWFrZXMgb25lIGxpc3Qgb2Ygc291cmNlcyBvbiB0aGUgViBncmlkCiAgMy4gbWVhc3VyZXMgZXZlcnkgc291cmNlIGluIGJvdGggYmFuZHMgYXQgdGhlIHNhbWUgcGxhY2Ugb24gdGhlIHNreSwKICAgICB0aHJvdWdoIE9ORSBhcGVydHVyZTogdHdpY2UgdGhlIEZXSE0gb2YgdGhlIHdpZGVyIG9mIHRoZSB0d28gZnJhbWVzLgogICAgIEVhY2ggZmx1eCBnZXRzIGFuIGVycm9yIGZyb20gdGhlIHBob3RvbiBub2lzZSBhbmQgdGhlIG1lYXN1cmVkIHNreQogICAgIG5vaXNlLCBhbmQgZWFjaCBzb3VyY2UgYSBzYXR1cmF0aW9uIGZsYWcKICA0LiBjaGVja3MgdGhlIHBsYXRlIHNvbHV0aW9uIGFnYWluc3QgR2FpYSwgdGhlbiBjYWxpYnJhdGVzIGFnYWluc3QgZXZlcnkKICAgICBjYXRhbG9ndWUgc3RhciB0aGF0IGlzIG1hdGNoZWQgb25jZSwgbm90IHNhdHVyYXRlZCwgbm90IGNyb3dkZWQgYW5kCiAgICAgd2VsbCBtZWFzdXJlZCAoQVBBU1M5LCBvciBHYWlhIERSMyBzeW50aGV0aWMgcGhvdG9tZXRyeSwgd2hpY2ggaXMgYWxzbwogICAgIHRoZSBmYWxsYmFjayk6IGEgc3RyYWlnaHQtbGluZSBmaXQgd2l0aCBhIGNvbG91ciB0ZXJtIGFuZCBvdXRsaWVyCiAgICAgcmVqZWN0aW9uLAogICAgICAgICBCLVYgPSBhICsgYyB4IChiLXYpICAgICAgICBWID0gdiArIFpQICsgayB4IChCLVYpCiAgICAgVGhlIGZpdCBpcyBwcmludGVkIHdpdGggaXRzIHNjYXR0ZXIsIGFuZCB0aGUgcnVuIHN0b3BzIGlmIGZld2VyIHRoYW4KICAgICA1IHN0YXJzIHJlbWFpbiBvciBpZiB0aGUgY29sb3VycyBydW4gdGhlIHdyb25nIHdheSAoQiBhbmQgViBzd2FwcGVkKQogIDUuIHJlbW92ZXMgdGhlIGR1c3Qgb2Ygb3VyIG93biBnYWxheHksIGFuZCB0aGUgZm9yZWdyb3VuZCBzdGFycwogIDYuIGtlZXBzIGFzIGJsdWUga25vdHMgdGhlIHNvdXJjZXMgd2l0aCAtMC41IDw9IChCLVYpMCA8PSAwLjUgYW5kIGEgQi1WCiAgICAgZXJyb3Igb2YgYXQgbW9zdCAwLjIgbWFnLCBhbmQgcGxvdHMgdGhlIGNvbG91ci1tYWduaXR1ZGUgZGlhZ3JhbXMgYW5kCiAgICAgdGhlIHJhZGlhbCBkZW5zaXR5IHByb2ZpbGUKCldIQVQgWU9VIEdFVAoKRXZlcnkgcnVuIHdyaXRlcyBhIG5ldyBmb2xkZXIgYmVzaWRlIHRoZSBmcmFtZXMsCnJlc3VsdF9DTURfPGdhbGF4eT5fPGRhdGU+Xzx0aW1lPiwgc28gYSBzZWNvbmQgcnVuIG5ldmVyIG92ZXJ3cml0ZXMgdGhlIGZpcnN0CmFuZCBub3RoaW5nIGZyb20gYW4gZWFybGllciBydW4gY2FuIGJlIG1pc3Rha2VuIGZvciB0aGlzIG9uZS4gVGhlIGZvbGRlcgpob2xkaW5nIHlvdXIgZGF0YSBpcyBvdGhlcndpc2UgbGVmdCBleGFjdGx5IGFzIGl0IHdhcy4KClRoYXQgaXMgYSBsb3Qgb2YgZmlsZXMsIHNvIGhlcmUgaXMgd2hhdCBlYWNoIG9uZSBpcyBhbmQgd2hpY2ggeW91IGFjdHVhbGx5CndhbnQgdG8gbG9vayBhdC4gQmVsb3csIE5BTUUgc3RhbmRzIGZvciB0aGUgZ2FsYXh5IG5hbWUuCgogIHJ1bl9zdW1tYXJ5LnR4dCAgICAgICAgICAgICAgICAgZXZlcnkgbnVtYmVyIHRoZSBydW4gdXNlZCBhbmQgZm91bmQKCiAgQnktcHJvZHVjdHMsIGtlcHQgYmVjYXVzZSB0aGV5IGFyZSB3aGF0IGV2ZXJ5dGhpbmcgaXMgbWVhc3VyZWQgb246CgogICAgPGlucHV0Pl9iZ3N1Yi5maXRzICAgICAgICAgICAgZWFjaCBpbnB1dCBmcmFtZSB3aXRoIGl0cyBtZWRpYW4gbGV2ZWwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHN1YnRyYWN0ZWQKICAgIE5BTUVfQl9jb21iaW5lZC5maXRzICAgICAgICAgIHdoZW4gdGhlIGZvbGRlciBoZWxkIHNldmVyYWwgZXhwb3N1cmVzCiAgICBOQU1FX1ZfY29tYmluZWQuZml0cyAgICAgICAgICBvZiBhIGZpbHRlcjogdGhlaXIgbWVkaWFuCgogIFN0YWdlIDEgLSBldmVyeXRoaW5nIHRoYXQgd2FzIG1lYXN1cmVkCgogICAgTkFNRV9waG90b21ldHJ5X3Jlc3VsdHMuY3N2ICAgZXZlcnkgc291cmNlIG1lYXN1cmVkIGluIGJvdGggYmFuZHM6CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBwb3NpdGlvbiBpbiBlYWNoIGZyYW1lLCBSQSBhbmQgRGVjLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZmx1eGVzIGFuZCBpbnN0cnVtZW50YWwgbWFnbml0dWRlcyB3aXRoCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBlcnJvcnMsIHBlYWsgcGl4ZWwsIHNhdHVyYXRpb24gZmxhZwogICAgTkFNRV9yZWZlcmVuY2Vfc3RhcnMuY3N2ICAgICAgdGhlIGNhdGFsb2d1ZSBzdGFycyBtYXRjaGVkIHRvIHNvdXJjZXMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3aGV0aGVyIGVhY2ggd2FzIHVzZWQgaW4gdGhlIGNhbGlicmF0aW9uLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYW5kIGl0cyByZXNpZHVhbC4gSWYgdGhlIGNhbGlicmF0aW9uIGxvb2tzCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3cm9uZywgbG9vayBoZXJlIGZpcnN0CgogIFN0YWdlIDIgLSBjYWxpYnJhdGVkLCBzdGlsbCB3aXRoIGZvcmVncm91bmQgc3RhcnMgaW4gaXQKCiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeS5jc3YgICAgICAgICAgICBCLCBWLCBCLVYgd2l0aCBlcnJvcnMsIGJlZm9yZQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKF9vYnMpIGFuZCBhZnRlciB0aGUgR2FsYWN0aWMKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGR1c3QgY29ycmVjdGlvbiwgYW5kIE1fVgogICAgTkFNRV9jYWxpYnJhdGVkX3Bob3RvbWV0cnlfd2l0aF9jb2xvci5jc3YgdGhlIHNhbWUsIHdpdGggQi1WIGFkZGVkCiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9DTUQucG5nICAgICAgICBjb2xvdXItbWFnbml0dWRlIGRpYWdyYW0KICAgIE5BTUVfVl9zb3VyY2VzLnBuZyAgICAgICAgICAgICAgICAgICAgICAgIHRoZSBWIGltYWdlIHdpdGggZXZlcnkgbWVhc3VyZWQKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHNvdXJjZSBjaXJjbGVkCiAgICBOQU1FX0Jfc291cmNlcy5wbmcgICAgICAgICAgICAgICAgICAgICAgICB0aGUgc2FtZSBmb3IgQiwgYXQgdGhlIHNhbWUKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHNvdXJjZXMKCiAgU3RhZ2UgMyAtIGZvcmVncm91bmQgc3RhcnMgcmVtb3ZlZAoKICAgIENhdGFsb2d1ZSBzdGFycywgR2FpYSBzdGFycyB3aXRoIGEgc2lnbmlmaWNhbnQgcGFyYWxsYXggb3IgcHJvcGVyIG1vdGlvbiwKICAgIGFuZCBldmVyeXRoaW5nIGJleW9uZCBSMjUgYXJlIHRha2VuIG91dC4KCiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFycy5jc3YKICAgIE5BTUVfY2FsaWJyYXRlZF9waG90b21ldHJ5X25vX3N0YXJzX3dpdGhfY29sb3IuY3N2CiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFyc19DTUQucG5nCiAgICBOQU1FX1Zfc291cmNlc19ub19zdGFycy5wbmcKCiAgU3RhZ2UgNCAtIGJsdWUga25vdHMgb25seQoKICAgIFRISVMgSVMgVEhFIFJFU1VMVCB0aGUgcHJvamVjdHMgYXJlIHVzdWFsbHkgYWZ0ZXIuCgogICAgTkFNRV9jYWxpYnJhdGVkX3Bob3RvbWV0cnlfbm9fc3RhcnNfY29sb3JfZmlsdGVyZWQuY3N2CiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFyc19jb2xvcl9maWx0ZXJlZF93aXRoX2NvbG9yLmNzdgogICAgTkFNRV9jYWxpYnJhdGVkX3Bob3RvbWV0cnlfbm9fc3RhcnNfY29sb3JfZmlsdGVyZWRfQ01ELnBuZwogICAgTkFNRV9WX3NvdXJjZXNfY29sb3JfZmlsdGVyZWQucG5nCiAgICBOQU1FX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFyc19jb2xvcl9maWx0ZXJlZF93aXRoX3JhZGl1cy5jc3YKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhlIHNhbWUsIHBsdXMgZWFjaCBrbm90J3MgZGlzdGFuY2UKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZnJvbSB0aGUgY2VudHJlIG9mIHRoZSBnYWxheHkKCiAgU3RhZ2UgNSAtIGhvdyB0aGUga25vdHMgYXJlIGRpc3RyaWJ1dGVkCgogICAgTkFNRV9yYWRpYWxfZGVuc2l0eV9wcm9maWxlLmNzdiAgICAgICBrbm90cyBwZXIgc3F1YXJlIHBpeGVsLCBieSByYWRpdXMKICAgIE5BTUVfcmFkaWFsX2RlbnNpdHlfcHJvZmlsZV9wYy5jc3YgICAgdGhlIHNhbWUgaW4gcGFyc2VjcwogICAgTkFNRV9yYWRpYWxfZGVuc2l0eV9wcm9maWxlX3N0ZXAuY3N2ICAgICB0aGUgc2FtZSBhcyBhIHN0ZXAgZnVuY3Rpb24KICAgIE5BTUVfcmFkaWFsX2RlbnNpdHlfcHJvZmlsZV9zdGVwX3BjLmNzdiAgdGhlIHN0ZXAgZnVuY3Rpb24gaW4gcGFyc2VjcwogICAgTkFNRV9yYWRpYWxfZGVuc2l0eV9wcm9maWxlX3N0ZXAucG5nICAgICAgIHRoZSBwbG90LCBpbiBwaXhlbHMKICAgIE5BTUVfcmFkaWFsX2RlbnNpdHlfcHJvZmlsZV9zdGVwX3BjLnBuZyAgICB0aGUgcGxvdCwgaW4gcGFyc2VjcwogICAgTkFNRV9kZXByb2plY3RlZF8uLi4gICAgICAgICAgICAgICAgICAgICAgIHRoZSBwcm9maWxlIGRlcHJvamVjdGVkIHNldmVyYWwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3YXlzLCB3aXRoIHRoZSBmcmFtZSdzIGNvdmVyYWdlCgogIElmIHlvdSBsb29rIGF0IGZvdXIgZmlsZXMsIGxvb2sgYXQgdGhlc2U6CgogICAgcnVuX3N1bW1hcnkudHh0ICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRpZCB0aGUgY2FsaWJyYXRpb24gd29yawogICAgTkFNRV9WX3NvdXJjZXNfY29sb3JfZmlsdGVyZWQucG5nICAgICAgICAgIGRpZCBpdCBmaW5kIHRoZSBhcm1zCiAgICBOQU1FXy4uLl9jb2xvcl9maWx0ZXJlZF9DTUQucG5nICAgICAgICAgICAgdGhlIGRpYWdyYW0KICAgIE5BTUVfcmFkaWFsX2RlbnNpdHlfcHJvZmlsZV9zdGVwX3BjLnBuZyAgICB0aGUgYXJtcywgaW4gbnVtYmVycwoKICBJbiB0aGUgZGlhZ3JhbXMsIHNhdHVyYXRlZCBzb3VyY2VzIGFuZCBzb3VyY2VzIHdpdGggYSBCLVYgZXJyb3IgYWJvdmUgMC4zCiAgbWFnIGFyZSBub3QgZHJhd247IHRoZXkgc3RheSBpbiB0aGUgdGFibGVzLgoKVGhlIHBhcmFtZXRlcnMgYXQgdGhlIHRvcCBvZiB0aGUgQmx1ZV9DbHVzdGVycyBzZWN0aW9uIG9mIHRoaXMgZmlsZSBhcmUKdHVuZWQgZm9yIHRoZSBLaW5uZXJldCBhbmQgTENPIGZyYW1lcyB0aGVzZSBwcm9qZWN0cyB1c2UuIE9uIHZlcnkgZGlmZmVyZW50CmRhdGEgLSBhIG11Y2ggc21hbGxlciB0ZWxlc2NvcGUsIGEgbXVjaCBmYWludGVyIGdhbGF4eSAtIHRoZSBkZXRlY3Rpb24KdGhyZXNob2xkIGlzIHRoZSBmaXJzdCB0aGluZyB0byBsb29rIGF0LgoKVXNhZ2U6IHJ1biBpdCwgYW5kIGFuc3dlciB0aGUgcXVlc3Rpb25zOyBFbnRlciBhY2NlcHRzIHdoYXQgaXMgc2hvd24uCiIiIgoKCgpmcm9tIF9fZnV0dXJlX18gaW1wb3J0IGFubm90YXRpb25zCgppbXBvcnQgb3MKaW1wb3J0IHN5cwppbXBvcnQgcmUKaW1wb3J0IG51bXB5IGFzIG5wCmltcG9ydCBtYXRoCmltcG9ydCB0ZXh0d3JhcAppbXBvcnQgbWF0cGxvdGxpYgppbXBvcnQgbWF0cGxvdGxpYi5weXBsb3QgYXMgcGx0CmltcG9ydCBhcmdwYXJzZQppbXBvcnQgd2FybmluZ3MKaW1wb3J0IHBhbmRhcyBhcyBwZApmcm9tIGFzdHJvcHkuaW8gaW1wb3J0IGZpdHMKZnJvbSBwaG90dXRpbHMuZGV0ZWN0aW9uIGltcG9ydCBEQU9TdGFyRmluZGVyCmZyb20gcGhvdHV0aWxzLmFwZXJ0dXJlIGltcG9ydCBDaXJjdWxhckFwZXJ0dXJlLCBDaXJjdWxhckFubnVsdXMsIGFwZXJ0dXJlX3Bob3RvbWV0cnkKZnJvbSBwaG90dXRpbHMuYmFja2dyb3VuZCBpbXBvcnQgQmFja2dyb3VuZDJELCBNZWRpYW5CYWNrZ3JvdW5kCmZyb20gYXN0cm9weS5zdGF0cyBpbXBvcnQgc2lnbWFfY2xpcHBlZF9zdGF0cywgU2lnbWFDbGlwCmZyb20gc2NpcHkubmRpbWFnZSBpbXBvcnQgZ2F1c3NpYW5fZmlsdGVyCmZyb20gYXN0cm9weS53Y3MgaW1wb3J0IFdDUwpmcm9tIGFzdHJvcXVlcnkudml6aWVyIGltcG9ydCBWaXppZXIKZnJvbSBhc3Ryb3B5LmNvb3JkaW5hdGVzIGltcG9ydCBTa3lDb29yZAppbXBvcnQgYXN0cm9weS51bml0cyBhcyB1CmltcG9ydCBhc3Ryb2FsaWduIGFzIGFhCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIHNoYXJlZCBtb2R1bGUgaW50ZXJhY3RpdmVfaW5wdXQucHkKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQoKZGVmIG5vdGhpbmdfd2FzX3Bhc3NlZChhcmd2PU5vbmUpOgogICAgIiIiVHJ1ZSB3aGVuIHRoZSB0b29sIHdhcyBzdGFydGVkIHdpdGggbm8gYXJndW1lbnRzIGF0IGFsbC4KCiAgICBUaGF0IGlzIHRoZSBvbmx5IGNhc2Ugd2hlcmUgcXVlc3Rpb25zIGFyZSB0aGUgcmlnaHQgcmVzcG9uc2UuIEEgdG9vbCBzdGFydGVkCiAgICB3aXRoIHNvbWUgYXJndW1lbnRzIGJ1dCBub3QgYWxsIG9mIHRoZW0gaGFzIGEgbWlzdGFrZSBpbiBpdCwgYW5kIGFyZ3BhcnNlCiAgICBzYXlpbmcgd2hpY2ggb25lIGlzIG1pc3NpbmcgaXMgbW9yZSB1c2VmdWwgdGhhbiBhIHF1ZXN0aW9ubmFpcmUuCiAgICAiIiIKICAgIGlmIGFyZ3YgaXMgTm9uZToKICAgICAgICByZXR1cm4gbGVuKHN5cy5hcmd2KSA8PSAxCiAgICByZXR1cm4gbGVuKGFyZ3YpID09IDAKCmRlZiBub3JtYWxpc2VfcGF0aCh0ZXh0KToKICAgICIiIkEgcGF0aCBhcyBhIHBlcnNvbiBhY3R1YWxseSBwYXN0ZXMgaXQuCgogICAgV2luZG93cyBFeHBsb3JlcidzICJDb3B5IGFzIHBhdGgiIHdyYXBzIHRoZSBwYXRoIGluIGRvdWJsZSBxdW90ZXMsIGFuZCBhIHBhdGgKICAgIGRyYWdnZWQgaW50byBhIHRlcm1pbmFsIG9mdGVuIGFycml2ZXMgd2l0aCBhIHNwYWNlIG9uIHRoZSBlbmQuIEJvdGggYXJlIG1lYW50CiAgICBhcyB0aGUgc2FtZSBwYXRoLCBzbyBib3RoIGFyZSBhY2NlcHRlZC4KICAgICIiIgogICAgcmV0dXJuIG9zLnBhdGgubm9ybXBhdGgodGV4dC5zdHJpcCgpLnN0cmlwKCciJykuc3RyaXAoIiciKSkKCmRlZiBhc2socHJvbXB0LCBkZWZhdWx0PU5vbmUsIGNhc3Q9c3RyKToKICAgICIiIk9uZSBxdWVzdGlvbi4gVGhlIGRlZmF1bHQgaXMgc2hvd24gaW4gYnJhY2tldHMgYW5kIGEgYmFyZSBFbnRlciB0YWtlcyBpdC4KCiAgICBBIHF1ZXN0aW9uIHdpdGggbm8gZGVmYXVsdCBpcyBhc2tlZCBhZ2FpbiB1bnRpbCBpdCBpcyBhbnN3ZXJlZCwgYW5kIGFuIGFuc3dlcgogICAgdGhhdCB3aWxsIG5vdCBjb252ZXJ0IGlzIHJlZnVzZWQgb24gdGhlIHNwb3QgcmF0aGVyIHRoYW4gdGhyZWUgc3RlcHMgbGF0ZXIsCiAgICB3aGVyZSB0aGUgcmVhc29uIHdvdWxkIG5vIGxvbmdlciBiZSBvYnZpb3VzLgogICAgIiIiCiAgICB3aGlsZSBUcnVlOgogICAgICAgIHNob3duID0gKCJ7fSBbe31dOiAiLmZvcm1hdChwcm9tcHQsIGRlZmF1bHQpIGlmIGRlZmF1bHQgaXMgbm90IE5vbmUKICAgICAgICAgICAgICAgICBlbHNlICJ7fTogIi5mb3JtYXQocHJvbXB0KSkKICAgICAgICB0ZXh0ID0gaW5wdXQoc2hvd24pLnN0cmlwKCkKICAgICAgICBpZiBub3QgdGV4dDoKICAgICAgICAgICAgaWYgZGVmYXVsdCBpcyBOb25lOgogICAgICAgICAgICAgICAgcHJpbnQoIiAgICB0aGlzIG9uZSBoYXMgbm8gZGVmYXVsdCAtIHBsZWFzZSBhbnN3ZXIgaXQiKQogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgIyBUaGUgZGVmYXVsdCBnb2VzIHRocm91Z2ggdGhlIHNhbWUgY29udmVyc2lvbiBhcyBhIHR5cGVkIGFuc3dlci4KICAgICAgICAgICAgIyBSZXR1cm5pbmcgaXQgcmF3IGxvb2tzIGhhcm1sZXNzIGJlY2F1c2UgbW9zdCBkZWZhdWx0cyBhcmUgc3RyaW5ncywKICAgICAgICAgICAgIyBidXQgYXNrKCJXaGljaCBvbmUiLCAiMSIsIGludCkgdGhlbiBoYW5kcyBiYWNrICIxIiBhbmQgdGhlIGNhbGxlcgogICAgICAgICAgICAjIGNvbXBhcmVzIGEgc3RyaW5nIHdpdGggYSBudW1iZXIgc2V2ZXJhbCBsaW5lcyBsYXRlciAtIHdoaWNoIGlzCiAgICAgICAgICAgICMgZXhhY3RseSBob3cgYSBiYXJlIEVudGVyIG9uIHRoZSBmcmFtZS1jaG9pY2UgcHJvbXB0IGNyYXNoZWQgYSBydW4uCiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHJldHVybiBjYXN0KGRlZmF1bHQpCiAgICAgICAgICAgIGV4Y2VwdCAoVHlwZUVycm9yLCBWYWx1ZUVycm9yKToKICAgICAgICAgICAgICAgIHJldHVybiBkZWZhdWx0CiAgICAgICAgdHJ5OgogICAgICAgICAgICByZXR1cm4gY2FzdCh0ZXh0KQogICAgICAgIGV4Y2VwdCAoVHlwZUVycm9yLCBWYWx1ZUVycm9yKToKICAgICAgICAgICAgcHJpbnQoIiAgICB0aGF0IGlzIG5vdCBhIHt9IC0gdHJ5IGFnYWluIi5mb3JtYXQoCiAgICAgICAgICAgICAgICBnZXRhdHRyKGNhc3QsICJfX25hbWVfXyIsICJ2YWxpZCB2YWx1ZSIpKSkKCmRlZiBhc2tfZmlsZShwcm9tcHQpOgogICAgIiIiQSBwYXRoIHRvIGEgZmlsZSB0aGF0IGV4aXN0cywgYXNrZWQgYWdhaW4gdW50aWwgaXQgZG9lcy4KCiAgICBDaGVja2luZyBoZXJlIGlzIHdvcnRoIHRoZSB0d28gbGluZXM6IHRoZSBhbHRlcm5hdGl2ZSBpcyBhIHJ1biB0aGF0IHJlYWRzIHRocmVlCiAgICBmcmFtZXMsIHNwZW5kcyBhIG1pbnV0ZSBvbiB0aGUgZm91cnRoLCBhbmQgdGhlbiBkaWVzIG9uIGEgdHlwbyBpbiB0aGUgZmlyc3QuCiAgICAiIiIKICAgIHdoaWxlIFRydWU6CiAgICAgICAgcGF0aCA9IG5vcm1hbGlzZV9wYXRoKGFzayhwcm9tcHQpKQogICAgICAgIGlmIG9zLnBhdGguaXNmaWxlKHBhdGgpOgogICAgICAgICAgICByZXR1cm4gcGF0aAogICAgICAgIHByaW50KCIgICAgdGhlcmUgaXMgbm8gZmlsZSBhdCAiICsgcGF0aCkKCmRlZiBhc2tfZm9sZGVyKHByb21wdCk6CiAgICAiIiJBIHBhdGggdG8gYSBmb2xkZXIgdGhhdCBleGlzdHMsIGFza2VkIGFnYWluIHVudGlsIGl0IGRvZXMuIiIiCiAgICB3aGlsZSBUcnVlOgogICAgICAgIHBhdGggPSBub3JtYWxpc2VfcGF0aChhc2socHJvbXB0KSkKICAgICAgICBpZiBvcy5wYXRoLmlzZGlyKHBhdGgpOgogICAgICAgICAgICByZXR1cm4gcGF0aAogICAgICAgIHByaW50KCIgICAgdGhlcmUgaXMgbm8gZm9sZGVyIGF0ICIgKyBwYXRoKQoKZGVmIGFza19vcHRpb25hbF9mbG9hdChwcm9tcHQsIGJsYW5rX25vdGU9ImJsYW5rIHRvIHNraXAiKToKICAgICIiIkEgbnVtYmVyLCBvciBub3RoaW5nLiBSZXR1cm5zIE5vbmUgd2hlbiB0aGUgYW5zd2VyIGlzIGxlZnQgYmxhbmsuCgogICAgV2hhdCBibGFuayBtZWFucyBpcyB0aGUgY2FsbGVyJ3MgYnVzaW5lc3MsIHNvIHRoZSBjYWxsZXIgc2F5cyBpdC4gQmxhbmsgdXNlZAogICAgdG8gbWVhbiAiZG8gd2l0aG91dCIsIGFuZCBub3cgaXQgdXN1YWxseSBtZWFucyAiZ28gYW5kIGxvb2sgaXQgdXAiIC0gYSBwcm9tcHQKICAgIHRoYXQgc3RpbGwgcHJvbWlzZWQgdGhlIGZpcnN0IHdvdWxkIGJlIGx5aW5nIGFib3V0IHdoYXQgdGhlIHRvb2wgZG9lcy4KICAgICIiIgogICAgd2hpbGUgVHJ1ZToKICAgICAgICB0ZXh0ID0gaW5wdXQoInt9IFt7fV06ICIuZm9ybWF0KHByb21wdCwgYmxhbmtfbm90ZSkpLnN0cmlwKCkKICAgICAgICBpZiBub3QgdGV4dDoKICAgICAgICAgICAgcmV0dXJuIE5vbmUKICAgICAgICB0cnk6CiAgICAgICAgICAgIHJldHVybiBmbG9hdCh0ZXh0KQogICAgICAgIGV4Y2VwdCBWYWx1ZUVycm9yOgogICAgICAgICAgICBwcmludCgiICAgIHRoYXQgaXMgbm90IGEgbnVtYmVyIC0gdHJ5IGFnYWluLCBvciBwcmVzcyBFbnRlciB0byBza2lwIikKCmRlZiBhc2tfeWVzX25vKHByb21wdCwgZGVmYXVsdD1GYWxzZSk6CiAgICAiIiJBIHllcyBvciBubyBxdWVzdGlvbi4gQW55dGhpbmcgc3RhcnRpbmcgd2l0aCB5IG9yIG4gY291bnRzLiIiIgogICAgaGludCA9ICJZL24iIGlmIGRlZmF1bHQgZWxzZSAieS9OIgogICAgd2hpbGUgVHJ1ZToKICAgICAgICB0ZXh0ID0gaW5wdXQoInt9IFt7fV06ICIuZm9ybWF0KHByb21wdCwgaGludCkpLnN0cmlwKCkubG93ZXIoKQogICAgICAgIGlmIG5vdCB0ZXh0OgogICAgICAgICAgICByZXR1cm4gZGVmYXVsdAogICAgICAgIGlmIHRleHRbMF0gaW4gInluIjoKICAgICAgICAgICAgcmV0dXJuIHRleHRbMF0gPT0gInkiCiAgICAgICAgcHJpbnQoIiAgICBwbGVhc2UgYW5zd2VyIHllcyBvciBubyIpCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIHNoYXJlZCBtb2R1bGUgZnJhbWVfaW52ZW50b3J5LnB5CiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KClJPTEVTID0gKCJCIiwgIlYiLCAiUiIsICJIYSIpCgpST0xFX01FQU5JTkcgPSB7CiAgICAiQiI6ICJibHVlIiwKICAgICJWIjogInZpc3VhbCAtIHRoZSByZWZlcmVuY2UgZ3JpZCIsCiAgICAiUiI6ICJyZWQgY29udGludXVtIiwKICAgICJIYSI6ICJILWFscGhhIC0gdGhlIGVtaXNzaW9uIGxpbmUiLAp9CgpfUEFUVEVSTlMgPSB7CiAgICAiSGEiOiByZS5jb21waWxlKHIiKD88IVthLXowLTldKWhbXHNfLV0qYWwoPzpwaGF8ZmEpKD8hW2EtejAtOV0pIgogICAgICAgICAgICAgICAgICAgICByInwoPzwhW2EtejAtOV0paGEoPyFbYS16MC05XSkiKSwKICAgICJCIjogcmUuY29tcGlsZShyIig/PCFbYS16MC05XSkoPzpifGJlc3NlbGxbXHNfLV0qYnxicCkoPyFbYS16MC05XSkiKSwKICAgICJWIjogcmUuY29tcGlsZShyIig/PCFbYS16MC05XSkoPzp2fGJlc3NlbGxbXHNfLV0qdikoPyFbYS16MC05XSkiKSwKICAgICJSIjogcmUuY29tcGlsZShyIig/PCFbYS16MC05XSkoPzpyfHJwfHJjfHJwcmltZXxzbG9hbltcc18tXSpyfGJlc3NlbGxbXHNfLV0qcikiCiAgICAgICAgICAgICAgICAgICAgciIoPyFbYS16MC05XSkiKSwKfQoKX0NPTUJJTkVEX1dPUkRTID0gcmUuY29tcGlsZShyIlxiKG1lZGlhbnxhdmVyYWdlfG1lYW58c3VtKVxzK29mXGJ8XGJjb21iaW58XGJzdGFjayIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmUuSSkKCkZJVFNfU1VGRklYRVMgPSAoIi5maXRzIiwgIi5maXQiLCAiLmZ0cyIsICIuZml0cy5meiIpCgpPVVRQVVRfRk9MREVSUyA9IHsicmVzdWx0X2NtZCIsICJyZXN1bHQiLCAicmVzdWx0cyIsICJvdXRwdXQiLCAib3V0In0KCmRlZiByb2xlc19pbl90ZXh0KHRleHQpOgogICAgIiIiRXZlcnkgZmlsdGVyIHJvbGUgbmFtZWQgaW4gYSBwaWVjZSBvZiB0ZXh0LiIiIgogICAgaWYgbm90IHRleHQ6CiAgICAgICAgcmV0dXJuIHNldCgpCiAgICBsb3cgPSBzdHIodGV4dCkubG93ZXIoKQogICAgcmV0dXJuIHtyb2xlIGZvciByb2xlLCBwYXR0ZXJuIGluIF9QQVRURVJOUy5pdGVtcygpIGlmIHBhdHRlcm4uc2VhcmNoKGxvdyl9CgpfRElGRkVSRU5DRSA9IHJlLmNvbXBpbGUoCiAgICByIig/OmhbXHNfLV0qYWwoPzpwaGF8ZmEpfCg/PCFbYS16MC05XSloYSlbXi9cXF0qPy1ccyooPzpyfHJwfHJjKSg/IVthLXowLTldKSIKICAgIHIifG1pbnVzfGNvbnRbXHNfLV0qc3ViIiwgcmUuSSkKCmRlZiBpc19kaWZmZXJlbmNlX25hbWUobmFtZSk6CiAgICAiIiJUcnVlIHdoZW4gYSBmaWxlIG5hbWUgc2F5cyB0aGUgZnJhbWUgaXMgb25lIGZpbHRlciBtaW51cyBhbm90aGVyLCBzdWNoCiAgICBhcyBILWFscGhhIHdpdGggdGhlIFIgY29udGludXVtIHRha2VuIG9mZiAoIk04M19IYVgxMi1SLmZ0cyIpLiIiIgogICAgcmV0dXJuIGJvb2woX0RJRkZFUkVOQ0Uuc2VhcmNoKHN0cihuYW1lKSkpCgpfUFJPRFVDVF9ISVNUT1JZID0gcmUuY29tcGlsZSgKICAgIHIibXVsdGlwbGllZFxzK2J5fHN1YnRyYWN0ZWRcYi4qXC4oPzpmaXRzP3xmdHMpXGIiLCByZS5JKQoKZGVmIGlzX3Byb2R1Y3RfaGVhZGVyKGhlYWRlcik6CiAgICAiIiJUcnVlIHdoZW4gYSBISVNUT1JZIGxpbmUgc2F5cyB0aGUgZnJhbWUgd2FzIHNjYWxlZCBieSBhIGZhY3RvciBvciBoYWQKICAgIGFub3RoZXIgRklUUyBmaWxlIHN1YnRyYWN0ZWQgZnJvbSBpdC4gQSBjb25zdGFudCBza3kgKCJTdWJ0cmFjdGVkIDkuMCIpIGRvZXMKICAgIG5vdCBjb3VudC4iIiIKICAgIHJldHVybiBhbnkoX1BST0RVQ1RfSElTVE9SWS5zZWFyY2gobGluZSkgZm9yIGxpbmUgaW4gaGlzdG9yeV9saW5lcyhoZWFkZXIpKQoKZGVmIHJvbGVfZnJvbV9oZWFkZXIoaGVhZGVyKToKICAgICIiIlRoZSByb2xlIG5hbWVkIGJ5IHRoZSBmaWx0ZXIga2V5d29yZCwgb3IgTm9uZSB3aGVuIHRoZXJlIGlzIG5vIHN1Y2gga2V5d29yZC4iIiIKICAgIGZvciBrZXkgaW4gKCJGSUxURVIiLCAiRklMVEVSMSIsICJGSUxUTkFNRSIsICJJTlNGSUxURSIpOgogICAgICAgIHZhbHVlID0gaGVhZGVyLmdldChrZXkpCiAgICAgICAgaWYgdmFsdWUgaXMgTm9uZSBvciBzdHIodmFsdWUpLnN0cmlwKCkudXBwZXIoKSBpbiAoIiIsICJOT1RQUkVTRU5UIiwgIk5PTkUiKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICBmb3VuZCA9IHJvbGVzX2luX3RleHQodmFsdWUpCiAgICAgICAgaWYgbGVuKGZvdW5kKSA9PSAxOgogICAgICAgICAgICByZXR1cm4gZm91bmQucG9wKCkKICAgIHJldHVybiBOb25lCgpkZWYgaGlzdG9yeV9saW5lcyhoZWFkZXIpOgogICAgIiIiVGhlIGhpc3RvcnkgbGluZXMgb2YgYSBoZWFkZXIsIGFzIHBsYWluIHN0cmluZ3MuIiIiCiAgICB0cnk6CiAgICAgICAgcmV0dXJuIFtzdHIodikgZm9yIHYgaW4gaGVhZGVyWyJISVNUT1JZIl1dCiAgICBleGNlcHQgKEtleUVycm9yLCBUeXBlRXJyb3IpOgogICAgICAgIHJldHVybiBbXQoKZGVmIGlzX2NvbWJpbmVkKGhlYWRlcik6CiAgICAiIiJUcnVlIHdoZW4gdGhlIGhlYWRlciBzYXlzIHRoaXMgZnJhbWUgaXMgc2V2ZXJhbCBmcmFtZXMgY29tYmluZWQuIiIiCiAgICByZXR1cm4gYW55KF9DT01CSU5FRF9XT1JEUy5zZWFyY2gobGluZSkgZm9yIGxpbmUgaW4gaGlzdG9yeV9saW5lcyhoZWFkZXIpKQoKZGVmIGNvbWJpbmVfbm90ZShoZWFkZXIpOgogICAgIiIiVGhlIGhpc3RvcnkgbGluZSB0aGF0IHNheXMgaG93IHRoZSBmcmFtZSB3YXMgY29tYmluZWQsIGZvciB0aGUgc3VtbWFyeS4iIiIKICAgIGZvciBsaW5lIGluIGhpc3RvcnlfbGluZXMoaGVhZGVyKToKICAgICAgICBpZiBfQ09NQklORURfV09SRFMuc2VhcmNoKGxpbmUpOgogICAgICAgICAgICByZXR1cm4gbGluZS5zdHJpcCgpCiAgICByZXR1cm4gIiIKCmRlZiBoYXNfd2NzKGhlYWRlcik6CiAgICAiIiJUcnVlIHdoZW4gdGhlIGhlYWRlciBjYXJyaWVzIGEgcGxhdGUgc29sdXRpb24gdGhlIHRvb2xzIGNhbiB1c2UuIiIiCiAgICB0cnk6CiAgICAgICAgZnJvbSBhc3Ryb3B5LndjcyBpbXBvcnQgV0NTCiAgICAgICAgcmV0dXJuIFdDUyhoZWFkZXIpLmhhc19jZWxlc3RpYWwKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcmV0dXJuIEZhbHNlCgpkZWYgaW5zcGVjdChwYXRoKToKICAgICIiIkV2ZXJ5dGhpbmcgdGhlIHRvb2xzIG5lZWQgdG8ga25vdyBhYm91dCBvbmUgZmlsZS4iIiIKICAgIGZyb20gYXN0cm9weS5pbyBpbXBvcnQgZml0cwoKICAgIGhlYWRlciA9IGZpdHMuZ2V0aGVhZGVyKHBhdGgpCiAgICByb2xlID0gcm9sZV9mcm9tX2hlYWRlcihoZWFkZXIpCiAgICBzb3VyY2UgPSAidGhlIGZpbHRlciBrZXl3b3JkIGluIHRoZSBoZWFkZXIiCgogICAgaWYgaXNfcHJvZHVjdF9oZWFkZXIoaGVhZGVyKToKICAgICAgICAjIFRoZSBoZWFkZXIgc2F5cyBpdCBpcyB0aGlzIGZyYW1lIHNjYWxlZCwgb3IgYW5vdGhlciBmcmFtZSB0YWtlbiBvZmYKICAgICAgICAjIGl0ICgiTXVsdGlwbGllZCBieSAxMS41IiwgIlN1YnRyYWN0ZWQgLi4uUl9TVUJfTWVkaWFuX29mXzMiKTogbm90IGEKICAgICAgICAjIGZyYW1lIHRocm91Z2ggb25lIGZpbHRlciwgd2hhdGV2ZXIgaXRzIG5hbWUgb3IgRklMVEVSIGtleXdvcmQgc2F5cy4KICAgICAgICByb2xlID0gTm9uZQogICAgICAgIHNvdXJjZSA9ICJpdHMgaGlzdG9yeSAoYSBzY2FsZWQgb3Igc3VidHJhY3RlZCBwcm9kdWN0IC0gbm90IHVzZWQpIgogICAgZWxpZiByb2xlIGlzIE5vbmUgYW5kIGlzX2RpZmZlcmVuY2VfbmFtZShvcy5wYXRoLmJhc2VuYW1lKHBhdGgpKToKICAgICAgICAjICJIYVgxMi1SIiBpcyBILWFscGhhIHdpdGggMTIgeCBSIHRha2VuIG9mZjogYSBwcm9kdWN0LCBub3QgYSBmcmFtZQogICAgICAgICMgdGhyb3VnaCBlaXRoZXIgZmlsdGVyLiBOYW1pbmcgaXQgUiB3b3VsZCBwdXQgaXQgaW4gYXMgdGhlIGNvbnRpbnV1bS4KICAgICAgICBzb3VyY2UgPSAidGhlIGZpbGUgbmFtZSAoYSBkaWZmZXJlbmNlIG9mIHR3byBmcmFtZXMgLSBub3QgdXNlZCkiCiAgICBlbGlmIHJvbGUgaXMgTm9uZToKICAgICAgICBmb3VuZCA9IHJvbGVzX2luX3RleHQob3MucGF0aC5iYXNlbmFtZShwYXRoKSkKICAgICAgICBzb3VyY2UgPSAidGhlIGZpbGUgbmFtZSIKICAgICAgICBpZiBsZW4oZm91bmQpID09IDE6CiAgICAgICAgICAgIHJvbGUgPSBuZXh0KGl0ZXIoZm91bmQpKQoKICAgIHJldHVybiB7CiAgICAgICAgInBhdGgiOiBwYXRoLAogICAgICAgICJuYW1lIjogb3MucGF0aC5iYXNlbmFtZShwYXRoKSwKICAgICAgICAicm9sZSI6IHJvbGUsCiAgICAgICAgInJvbGVfc291cmNlIjogc291cmNlIGlmIHJvbGUgZWxzZSAiIiwKICAgICAgICAiY29tYmluZWQiOiBpc19jb21iaW5lZChoZWFkZXIpLAogICAgICAgICJjb21iaW5lX25vdGUiOiBjb21iaW5lX25vdGUoaGVhZGVyKSwKICAgICAgICAid2NzIjogaGFzX3djcyhoZWFkZXIpLAogICAgICAgICJleHB0aW1lIjogaGVhZGVyLmdldCgiRVhQVElNRSIpLAogICAgICAgICJvYmplY3QiOiAoc3RyKGhlYWRlci5nZXQoIk9CSkVDVCIpKS5zdHJpcCgpCiAgICAgICAgICAgICAgICAgICBpZiBoZWFkZXIuZ2V0KCJPQkpFQ1QiKSBlbHNlICIiKSwKICAgICAgICAic2hhcGUiOiAoaGVhZGVyLmdldCgiTkFYSVMyIiksIGhlYWRlci5nZXQoIk5BWElTMSIpKSwKICAgIH0KCmRlZiBzY2FuX2ZvbGRlcihmb2xkZXIsIHJlY3Vyc2l2ZT1UcnVlKToKICAgICIiIkluc3BlY3QgZXZlcnkgRklUUyBmaWxlIGluIGEgZm9sZGVyLiBSZXR1cm5zIGEgbGlzdCBvZiByZWNvcmRzLiIiIgogICAgZm91bmQgPSBbXQogICAgZm9yIHJvb3QsIGRpcnMsIGZpbGVzIGluIG9zLndhbGsoZm9sZGVyKToKICAgICAgICAjIERvIG5vdCB3YWxrIGludG8gd2hhdCBhbiBlYXJsaWVyIHJ1biB3cm90ZS4KICAgICAgICAjIEV2ZXJ5IHJ1biBub3cgd3JpdGVzIGEgZm9sZGVyIG9mIGl0cyBvd24gLSByZXN1bHRfQ01EXzxnYWxheHk+XzxkYXRlPiwKICAgICAgICAjIHJlc3VsdF9ISUlfLi4uLCByZXN1bHRfU3RhckZvcm1hdGlvbl8uLi4gLSBzbyBhbnl0aGluZyB3aG9zZSBuYW1lCiAgICAgICAgIyBzdGFydHMgd2l0aCAicmVzdWx0IiBpcyBvdXRwdXQsIG5vdCBkYXRhLgogICAgICAgIGRpcnNbOl0gPSBbZCBmb3IgZCBpbiBkaXJzIGlmIGQubG93ZXIoKSBub3QgaW4gT1VUUFVUX0ZPTERFUlMKICAgICAgICAgICAgICAgICAgIGFuZCBub3QgZC5sb3dlcigpLnN0YXJ0c3dpdGgoInJlc3VsdCIpXQogICAgICAgIGZvciBuYW1lIGluIHNvcnRlZChmaWxlcyk6CiAgICAgICAgICAgIGlmIG5vdCBuYW1lLmxvd2VyKCkuZW5kc3dpdGgoRklUU19TVUZGSVhFUyk6CiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICBwYXRoID0gb3MucGF0aC5qb2luKHJvb3QsIG5hbWUpCiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIGZvdW5kLmFwcGVuZChpbnNwZWN0KHBhdGgpKQogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGV4YzoKICAgICAgICAgICAgICAgIGZvdW5kLmFwcGVuZCh7InBhdGgiOiBwYXRoLCAibmFtZSI6IG5hbWUsICJyb2xlIjogTm9uZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgInJvbGVfc291cmNlIjogIiIsICJjb21iaW5lZCI6IEZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAiY29tYmluZV9ub3RlIjogIiIsICJ3Y3MiOiBGYWxzZSwgImV4cHRpbWUiOiBOb25lLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAib2JqZWN0IjogIiIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICJzaGFwZSI6IChOb25lLCBOb25lKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgImVycm9yIjogInt9OiB7fSIuZm9ybWF0KHR5cGUoZXhjKS5fX25hbWVfXywgZXhjKX0pCiAgICAgICAgaWYgbm90IHJlY3Vyc2l2ZToKICAgICAgICAgICAgYnJlYWsKICAgIHJldHVybiBmb3VuZAoKZGVmIGdhbGF4eV9mcm9tX3JlY29yZHMocmVjb3JkcywgZm9sZGVyPU5vbmUpOgogICAgIiIiVGhlIGdhbGF4eSdzIG5hbWUsIHRha2VuIGZyb20gdGhlIGZyYW1lcyB0aGVtc2VsdmVzLgoKICAgIFRoZSB0ZWxlc2NvcGUgd3JvdGUgaXQgaW50byBldmVyeSBleHBvc3VyZSBpdCB0b29rLCBzbyB0aGVyZSBpcyBubyByZWFzb24gdG8KICAgIGFzayBmb3IgaXQgLSBhbmQgbm8gcmVhc29uIHRvIHJlYWQgaXQgb2ZmIGEgZm9sZGVyIG5hbWUsIHdoaWNoIGlzIHdyaXR0ZW4gYnkgYQogICAgcGVyc29uIGFuZCBjYXJyaWVzIHdoYXRldmVyIGVsc2UgdGhleSB3YW50ZWQgdG8gcmVtZW1iZXIgdGhhdCBkYXkuIEEgY29tYmluZWQKICAgIGZyYW1lIGhhcyB1c3VhbGx5IGxvc3QgdGhlIGtleXdvcmQgYWxvbmcgd2l0aCB0aGUgcmVzdCBvZiBpdHMgaGVhZGVyLCBidXQgaXQKICAgIG9ubHkgdGFrZXMgb25lIGV4cG9zdXJlIGluIHRoZSBmb2xkZXIgdGhhdCBzdGlsbCBoYXMgaXQuCgogICAgUmV0dXJucyAobmFtZSwgd2hlcmUgaXQgY2FtZSBmcm9tKS4gVGhlIGZvbGRlciBuYW1lIGlzIHRoZSBmYWxsYmFjaywgY2xlYW5lZCBvZgogICAgdGhlIG5vdGVzIHBlb3BsZSBhcHBlbmQgdG8gaXQuCiAgICAiIiIKICAgIGltcG9ydCBjb2xsZWN0aW9ucwoKICAgIHZvdGVzID0gY29sbGVjdGlvbnMuQ291bnRlcihyLmdldCgib2JqZWN0IiwgIiIpIGZvciByIGluIHJlY29yZHMKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiByLmdldCgib2JqZWN0IikpCiAgICBpZiB2b3RlczoKICAgICAgICBuYW1lLCBuID0gdm90ZXMubW9zdF9jb21tb24oMSlbMF0KICAgICAgICByZXR1cm4gbmFtZSwgInRoZSBPQkpFQ1Qga2V5d29yZCwgaW4ge30gb2Yge30gZnJhbWVzIi5mb3JtYXQobiwgbGVuKHJlY29yZHMpKQoKICAgIGlmIGZvbGRlcjoKICAgICAgICBzdGVtID0gb3MucGF0aC5iYXNlbmFtZShvcy5wYXRoLm5vcm1wYXRoKGZvbGRlcikpCiAgICAgICAgIyAiTTMzIDAuNCBtZXRlciIgaXMgYSBub3RlIHRvIHNlbGYsIG5vdCBhIGNhdGFsb2d1ZSBuYW1lOyB0aGUgbmFtZSBpcyB0aGUKICAgICAgICAjIGZpcnN0IHdvcmQgdGhhdCBsb29rcyBsaWtlIG9uZS4KICAgICAgICBtID0gcmUubWF0Y2gociJccyooW0EtWmEtel17MSw0fVxzKlxkK1tBLVphLXpdPykiLCBzdGVtKQogICAgICAgIGlmIG06CiAgICAgICAgICAgIHJldHVybiByZS5zdWIociJccysiLCAiICIsIG0uZ3JvdXAoMSkpLnN0cmlwKCksICJ0aGUgZm9sZGVyIG5hbWUiCiAgICAgICAgcmV0dXJuIHN0ZW0sICJ0aGUgZm9sZGVyIG5hbWUiCgogICAgcmV0dXJuIE5vbmUsICIiCgpkZWYgY2hvb3NlKHJlY29yZHMsIHJvbGVzPVJPTEVTLCBhc2s9Tm9uZSk6CiAgICAiIiJQaWNrIG9uZSBmcmFtZSBwZXIgcm9sZTogdGhlIGNvbWJpbmVkIG9uZSwgd2hlbiB0aGVyZSBpcyBvbmUuCgogICAgQSBjb21iaW5lZCBmcmFtZSBpcyBkZWVwZXIgdGhhbiBhbnkgc2luZ2xlIGV4cG9zdXJlLCB3aGljaCBpcyB0aGUgcmVhc29uIGl0CiAgICB3YXMgbWFkZSwgc28gaXQgd2lucyB3aGVuZXZlciBvbmUgZXhpc3RzLiBXaGVuIHRoZXJlIGlzIG5vbmUgYnV0IHNldmVyYWwKICAgIGV4cG9zdXJlcywgYWxsIG9mIHRoZW0gYXJlIHJldHVybmVkIHRvIGJlIGNvbWJpbmVkIHJhdGhlciB0aGFuIG9uZSBiZWluZwogICAgcGlja2VkIC0gdGhhdCBpcyB3aGF0IHRoZSBvYnNlcnZlciB3ZW50IG91dCBhbmQgdG9vay4KCiAgICBSZXR1cm5zIChjaG9zZW4sIG1lbWJlcnMsIG5vdGVzKTogdGhlIHJlcHJlc2VudGF0aXZlIGZyYW1lIHBlciByb2xlLCB0aGUKICAgIGV4cG9zdXJlcyB0byBjb21iaW5lIHdoZXJlIHRoZXJlIGFyZSBzZXZlcmFsLCBhbmQgYW55dGhpbmcgd29ydGggc2F5aW5nLgogICAgIiIiCiAgICBjaG9zZW4sIG1lbWJlcnMsIG5vdGVzID0ge30sIHt9LCBbXQoKICAgIGZvciByb2xlIGluIHJvbGVzOgogICAgICAgIG1pbmUgPSBbciBmb3IgciBpbiByZWNvcmRzIGlmIHIuZ2V0KCJyb2xlIikgPT0gcm9sZSBhbmQgbm90IHIuZ2V0KCJlcnJvciIpXQogICAgICAgIGlmIG5vdCBtaW5lOgogICAgICAgICAgICBub3Rlcy5hcHBlbmQoInt9OiBub3RoaW5nIGluIHRoZSBmb2xkZXIgaWRlbnRpZmllcyBpdHNlbGYgYXMge30iLmZvcm1hdCgKICAgICAgICAgICAgICAgIHJvbGUsIFJPTEVfTUVBTklOR1tyb2xlXSkpCiAgICAgICAgICAgIGNvbnRpbnVlCgogICAgICAgIGNvbWJpbmVkID0gW3IgZm9yIHIgaW4gbWluZSBpZiByWyJjb21iaW5lZCJdXQogICAgICAgIGlmIGxlbihjb21iaW5lZCkgPT0gMToKICAgICAgICAgICAgY2hvc2VuW3JvbGVdID0gY29tYmluZWRbMF0KICAgICAgICAgICAgY29udGludWUKICAgICAgICBpZiBsZW4oY29tYmluZWQpID4gMToKICAgICAgICAgICAgaWYgYXNrIGlzIE5vbmU6CiAgICAgICAgICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoInt9OiB7fSBjb21iaW5lZCBmcmFtZXMgdG8gY2hvb3NlIGZyb20iLmZvcm1hdCgKICAgICAgICAgICAgICAgICAgICByb2xlLCBsZW4oY29tYmluZWQpKSkKICAgICAgICAgICAgY2hvc2VuW3JvbGVdID0gYXNrKHJvbGUsIGNvbWJpbmVkKQogICAgICAgICAgICBjb250aW51ZQoKICAgICAgICBzaW5nbGVzID0gW3IgZm9yIHIgaW4gbWluZSBpZiByWyJ3Y3MiXV0gb3IgbWluZQogICAgICAgIGlmIGxlbihzaW5nbGVzKSA9PSAxOgogICAgICAgICAgICBjaG9zZW5bcm9sZV0gPSBzaW5nbGVzWzBdCiAgICAgICAgICAgIG5vdGVzLmFwcGVuZCgie306IG5vIGNvbWJpbmVkIGZyYW1lLCB1c2luZyB0aGUgb25seSBleHBvc3VyZSIuZm9ybWF0KHJvbGUpKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgICMgU2V2ZXJhbCBleHBvc3VyZXMgYW5kIG5vdGhpbmcgY29tYmluZWQuIFBpY2tpbmcgb25lIG9mIHRoZW0gYW5kCiAgICAgICAgICAgICMgZGlzY2FyZGluZyB0aGUgcmVzdCB0aHJvd3MgYXdheSBtb3N0IG9mIHRoZSBvYnNlcnZhdGlvbiAtIHdpdGggc2l4CiAgICAgICAgICAgICMgSC1hbHBoYSBmcmFtZXMgaXQgY29zdHMgYSBmYWN0b3Igb2YgMi40IGluIG5vaXNlIC0gc28gdGhleSBhcmUgYWxsCiAgICAgICAgICAgICMga2VwdCBhbmQgdGhlIGNhbGxlciBjb21iaW5lcyB0aGVtLiBUaGUgZmlyc3QgaXMgbmFtZWQgYXMgdGhlCiAgICAgICAgICAgICMgcmVwcmVzZW50YXRpdmUgYmVjYXVzZSB0aGUgb3V0cHV0IGhhcyB0byBiZSBsYWJlbGxlZCB3aXRoIHNvbWV0aGluZy4KICAgICAgICAgICAgY2hvc2VuW3JvbGVdID0gc2luZ2xlc1swXQogICAgICAgICAgICBtZW1iZXJzW3JvbGVdID0gc2luZ2xlcwogICAgICAgICAgICBub3Rlcy5hcHBlbmQoInt9OiBubyBjb21iaW5lZCBmcmFtZTsge30gZXhwb3N1cmVzIHRvIGNvbWJpbmUiLmZvcm1hdCgKICAgICAgICAgICAgICAgIHJvbGUsIGxlbihzaW5nbGVzKSkpCgogICAgcmV0dXJuIGNob3NlbiwgbWVtYmVycywgbm90ZXMKCmRlZiBkZXNjcmliZShyZWNvcmRzLCBjaG9zZW4sIG5vdGVzKToKICAgICIiIlRoZSBzdW1tYXJ5IGEgcGVyc29uIHJlYWRzIGJlZm9yZSBhZ3JlZWluZyB0byB0aGUgcnVuLiIiIgogICAgbGluZXMgPSBbIiIsICJXaGF0IHRoZSBmb2xkZXIgaG9sZHM6Il0KICAgIGJ5X3JvbGUgPSB7fQogICAgZm9yIHIgaW4gcmVjb3JkczoKICAgICAgICBieV9yb2xlLnNldGRlZmF1bHQoci5nZXQoInJvbGUiKSBvciAiPyIsIFtdKS5hcHBlbmQocikKCiAgICBmb3Igcm9sZSBpbiBsaXN0KFJPTEVTKSArIFsiPyJdOgogICAgICAgIGdyb3VwID0gYnlfcm9sZS5nZXQocm9sZSkKICAgICAgICBpZiBub3QgZ3JvdXA6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgbGluZXMuYXBwZW5kKCIgICIgKyAoInt9ICh7fSkiLmZvcm1hdChyb2xlLCBST0xFX01FQU5JTkdbcm9sZV0pCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgcm9sZSBpbiBST0xFX01FQU5JTkcgZWxzZSAibm90IGlkZW50aWZpZWQiKSkKICAgICAgICBmb3IgciBpbiBzb3J0ZWQoZ3JvdXAsIGtleT1sYW1iZGEgeDogeFsibmFtZSJdKToKICAgICAgICAgICAgbWFyayA9ICItPiIgaWYgY2hvc2VuLmdldChyb2xlKSBpcyByIGVsc2UgIiAgIgogICAgICAgICAgICBpZiByLmdldCgiZXJyb3IiKToKICAgICAgICAgICAgICAgIGJpdHMgPSBbInVucmVhZGFibGU6ICIgKyByWyJlcnJvciJdXQogICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgYml0cyA9IFsiY29tYmluZWQiIGlmIHJbImNvbWJpbmVkIl0gZWxzZSAic2luZ2xlIiwKICAgICAgICAgICAgICAgICAgICAgICAgInNvbHZlZCIgaWYgclsid2NzIl0gZWxzZSAiTk8gV0NTIl0KICAgICAgICAgICAgbGluZXMuYXBwZW5kKCIgICAge30gezo8NDJ9IHt9Ii5mb3JtYXQobWFyaywgclsibmFtZSJdLCAiLCAiLmpvaW4oYml0cykpKQoKICAgIGxpbmVzLmFwcGVuZCgiIikKICAgIGxpbmVzLmFwcGVuZCgiQ2hvc2VuIGZvciB0aGUgbWVhc3VyZW1lbnQ6IikKICAgIGZvciByb2xlIGluIFJPTEVTOgogICAgICAgIHIgPSBjaG9zZW4uZ2V0KHJvbGUpCiAgICAgICAgbGluZXMuYXBwZW5kKCIgICAgezo8M30ge30iLmZvcm1hdChyb2xlLCByWyJuYW1lIl0gaWYgciBlbHNlICJub3RoaW5nIikpCiAgICBmb3IgbiBpbiBub3RlczoKICAgICAgICBsaW5lcy5hcHBlbmQoIiAgICBub3RlOiAiICsgbikKICAgIGxpbmVzLmFwcGVuZCgiIikKICAgIHJldHVybiAiXG4iLmpvaW4obGluZXMpCgpkZWYgd2NzX2Zyb21fcmVmZXJlbmNlKGltYWdlLCByZWZlcmVuY2VfaW1hZ2UsIHJlZmVyZW5jZV93Y3MsCiAgICAgICAgICAgICAgICAgICAgICAgZGV0ZWN0aW9uX3NpZ21hPTYuMCwgbWF4X2NvbnRyb2xfcG9pbnRzPTYwKToKICAgICIiIkZpdCBhIHBsYXRlIHNvbHV0aW9uIGZvciBhbiB1bnNvbHZlZCBmcmFtZSBieSBtYXRjaGluZyBpdHMgc3RhcnMuCgogICAgQSBjb21iaW5lZCBmcmFtZSBhbmQgdGhlIHNpbmdsZSBleHBvc3VyZXMgaXQgd2FzIGJ1aWx0IGZyb20gc2hvdyB0aGUgc2FtZQogICAgc3RhcnMgaW4gYWxtb3N0IHRoZSBzYW1lIHBsYWNlcywgc28gYSBzb2x2ZWQgZXhwb3N1cmUgb2YgdGhlIHNhbWUgZmllbGQgY2FuCiAgICBsZW5kIGl0cyBzb2x1dGlvbjogbWF0Y2ggdGhlIHN0YXIgcGF0dGVybnMsIHJlYWQgZWFjaCBtYXRjaGVkIHN0YXIncyBza3kKICAgIHBvc2l0aW9uIGZyb20gdGhlIHNvbHZlZCBmcmFtZSwgYW5kIGZpdCBhIHNvbHV0aW9uIHRvIHRob3NlIHBvc2l0aW9ucyBpbiB0aGUKICAgIHVuc29sdmVkIGZyYW1lJ3Mgb3duIHBpeGVscy4KCiAgICBUaGlzIGlzIG5vdCBjb3B5aW5nIGEgaGVhZGVyIGFjcm9zcy4gVGhlIHJlc3VsdCBpcyBmaXR0ZWQgdG8gbWF0Y2hlZCBzdGFycyBhbmQKICAgIGNvbWVzIHdpdGggdGhlIHJlc2lkdWFsIG9mIHRoYXQgZml0LCB3aGljaCBpcyB0aGUgaG9uZXN0IG1lYXN1cmUgb2YgaG93IHdlbGwKICAgIGl0IGRpZDogYSBmZXcgaHVuZHJlZHRocyBvZiBhIHBpeGVsIG1lYW5zIHRoZSB0d28gZnJhbWVzIHJlYWxseSBhcmUgb24gdGhlCiAgICBzYW1lIGdyaWQsIGFuZCBhIHJlc2lkdWFsIGFwcHJvYWNoaW5nIGEgcGl4ZWwgbWVhbnMgc29tZXRoaW5nIG1vdmVkLgoKICAgIFJldHVybnMgKHdjcywgZGlhZ25vc3RpY3MpLgogICAgIiIiCiAgICBpbXBvcnQgbnVtcHkgYXMgbnAKICAgIGltcG9ydCBhc3Ryb2FsaWduIGFzIGFhCiAgICBmcm9tIGFzdHJvcHkud2NzLnV0aWxzIGltcG9ydCBmaXRfd2NzX2Zyb21fcG9pbnRzCgogICAgdHJhbnNmb3JtLCAoc3JjLCBkc3QpID0gYWEuZmluZF90cmFuc2Zvcm0oCiAgICAgICAgaW1hZ2UsIHJlZmVyZW5jZV9pbWFnZSwKICAgICAgICBtYXhfY29udHJvbF9wb2ludHM9bWF4X2NvbnRyb2xfcG9pbnRzLAogICAgICAgIGRldGVjdGlvbl9zaWdtYT1kZXRlY3Rpb25fc2lnbWEpCgogICAgc2t5ID0gcmVmZXJlbmNlX3djcy5waXhlbF90b193b3JsZChkc3RbOiwgMF0sIGRzdFs6LCAxXSkKICAgIGZpdHRlZCA9IGZpdF93Y3NfZnJvbV9wb2ludHMoKHNyY1s6LCAwXSwgc3JjWzosIDFdKSwgc2t5LCBwcm9qX3BvaW50PSJjZW50ZXIiKQoKICAgIGJhY2tfeCwgYmFja195ID0gZml0dGVkLndvcmxkX3RvX3BpeGVsKHNreSkKICAgIHJlc2lkdWFsID0gbnAuaHlwb3QoYmFja194IC0gc3JjWzosIDBdLCBiYWNrX3kgLSBzcmNbOiwgMV0pCgogICAgcmV0dXJuIGZpdHRlZCwgewogICAgICAgICJuX3N0YXJzIjogaW50KGxlbihzcmMpKSwKICAgICAgICAibWVkaWFuX3Jlc2lkdWFsX3B4IjogZmxvYXQobnAubWVkaWFuKHJlc2lkdWFsKSksCiAgICAgICAgIndvcnN0X3Jlc2lkdWFsX3B4IjogZmxvYXQocmVzaWR1YWwubWF4KCkpLAogICAgICAgICJzaGlmdF9weCI6IChmbG9hdCh0cmFuc2Zvcm0udHJhbnNsYXRpb25bMF0pLCBmbG9hdCh0cmFuc2Zvcm0udHJhbnNsYXRpb25bMV0pKSwKICAgICAgICAicm90YXRpb25fZGVnIjogZmxvYXQobnAuZGVncmVlcyh0cmFuc2Zvcm0ucm90YXRpb24pKSwKICAgICAgICAic2NhbGUiOiBmbG9hdCh0cmFuc2Zvcm0uc2NhbGUpLAogICAgfQoKZGVmIGRlc2NyaWJlX3djc19maXQobmFtZSwgZGlhZyk6CiAgICAiIiJPbmUgcmVhZGFibGUgcGFyYWdyYXBoIGFib3V0IGEgc29sdXRpb24gbGVudCBmcm9tIGFub3RoZXIgZnJhbWUuIiIiCiAgICByZXR1cm4gKAogICAgICAgICIgIHt9OiBzb2x2ZWQgYnkgbWF0Y2hpbmcge30gc3RhcnMgYWdhaW5zdCBhbiBleHBvc3VyZSB0aGF0IGhhcyBhIHNvbHV0aW9uXG4iCiAgICAgICAgIiAgICBzaGlmdCAoezorLjFmfSwgezorLjFmfSkgcHgsIHJvdGF0aW9uIHs6Ky4zZn0gZGVnLCBzY2FsZSB7Oi41Zn1cbiIKICAgICAgICAiICAgIGZpdCByZXNpZHVhbDogbWVkaWFuIHs6LjJmfSBweCwgd29yc3QgezouMmZ9IHB4Ii5mb3JtYXQoCiAgICAgICAgICAgIG5hbWUsIGRpYWdbIm5fc3RhcnMiXSwgZGlhZ1sic2hpZnRfcHgiXVswXSwgZGlhZ1sic2hpZnRfcHgiXVsxXSwKICAgICAgICAgICAgZGlhZ1sicm90YXRpb25fZGVnIl0sIGRpYWdbInNjYWxlIl0sCiAgICAgICAgICAgIGRpYWdbIm1lZGlhbl9yZXNpZHVhbF9weCJdLCBkaWFnWyJ3b3JzdF9yZXNpZHVhbF9weCJdKSkKCmRlZiBkb25vcl9mb3IocmVjb3JkLCByZWNvcmRzKToKICAgICIiIkEgc29sdmVkIGV4cG9zdXJlIHRoYXQgY2FuIGxlbmQgaXRzIHBsYXRlIHNvbHV0aW9uIHRvIGFuIHVuc29sdmVkIGZyYW1lLgoKICAgIEFuIGV4cG9zdXJlIHRocm91Z2ggdGhlIHNhbWUgZmlsdGVyIGlzIHByZWZlcnJlZCAtIHNhbWUgc3RhcnMsIHNhbWUgZGVwdGgsCiAgICB0aGUgY2xvc2VzdCB0aGluZyB0byB0aGUgZnJhbWUgdGhhdCBuZWVkcyBzb2x2aW5nIC0gYnV0IGFueSBzb2x2ZWQgZXhwb3N1cmUKICAgIG9mIHRoZSBmaWVsZCB3aWxsIGRvLCBzaW5jZSB0aGUgbWF0Y2ggaXMgbWFkZSBvbiBzdGFyIHBvc2l0aW9ucyBhbmQgdGhvc2UgZG8KICAgIG5vdCBjYXJlIHdoaWNoIGZpbHRlciB0aGV5IHdlcmUgc2VlbiB0aHJvdWdoLgogICAgIiIiCiAgICBpZiByZWNvcmQgaXMgTm9uZSBvciByZWNvcmQuZ2V0KCJ3Y3MiKToKICAgICAgICByZXR1cm4gTm9uZQogICAgc2FtZV9maWx0ZXIgPSBbciBmb3IgciBpbiByZWNvcmRzCiAgICAgICAgICAgICAgICAgICBpZiByLmdldCgid2NzIikgYW5kIHIuZ2V0KCJyb2xlIikgPT0gcmVjb3JkLmdldCgicm9sZSIpXQogICAgaWYgc2FtZV9maWx0ZXI6CiAgICAgICAgcmV0dXJuIHNhbWVfZmlsdGVyWzBdCiAgICBzb2x2ZWQgPSBbciBmb3IgciBpbiByZWNvcmRzIGlmIHIuZ2V0KCJ3Y3MiKV0KICAgIHJldHVybiBzb2x2ZWRbMF0gaWYgc29sdmVkIGVsc2UgTm9uZQoKZGVmIHBsYW4oZm9sZGVyLCByb2xlcz1ST0xFUywgYXNrPU5vbmUsIHJlY3Vyc2l2ZT1UcnVlKToKICAgICIiIkV2ZXJ5dGhpbmcgYSB0b29sIG5lZWRzIGZyb20gb25lIGZvbGRlciBwYXRoLgoKICAgIFJldHVybnMgKHJlY29yZHMsIGNob3NlbiwgbWVtYmVycywgZG9ub3JzLCBub3Rlcyk6IHdoYXQgaXMgaW4gdGhlIGZvbGRlciwgdGhlCiAgICBmcmFtZSBjaG9zZW4gZm9yIGVhY2ggcm9sZSwgdGhlIHNldmVyYWwgZXhwb3N1cmVzIHRvIGNvbWJpbmUgd2hlbiB0aGVyZSB3YXMgbm8KICAgIGNvbWJpbmVkIGZyYW1lLCB3aGljaCBzb2x2ZWQgZXhwb3N1cmUgd2lsbCBsZW5kIGEgcGxhdGUgc29sdXRpb24gdG8gYSBmcmFtZQogICAgdGhhdCBsYWNrcyBvbmUsIGFuZCBhbnl0aGluZyB3b3J0aCBzYXlpbmcgb3V0IGxvdWQuCiAgICAiIiIKICAgIHJlY29yZHMgPSBzY2FuX2ZvbGRlcihmb2xkZXIsIHJlY3Vyc2l2ZT1yZWN1cnNpdmUpCiAgICBpZiBub3QgcmVjb3JkczoKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoIm5vIEZJVFMgZmlsZXMgaW4gIiArIHN0cihmb2xkZXIpKQogICAgY2hvc2VuLCBtZW1iZXJzLCBub3RlcyA9IGNob29zZShyZWNvcmRzLCByb2xlcz1yb2xlcywgYXNrPWFzaykKICAgIGRvbm9ycyA9IHtyb2xlOiBkb25vcl9mb3IocmVjLCByZWNvcmRzKSBmb3Igcm9sZSwgcmVjIGluIGNob3Nlbi5pdGVtcygpfQogICAgZm9yIHJvbGUsIGRvbm9yIGluIGRvbm9ycy5pdGVtcygpOgogICAgICAgIGlmIGRvbm9yIGlzIG5vdCBOb25lOgogICAgICAgICAgICBub3Rlcy5hcHBlbmQoInt9OiBoYXMgbm8gcGxhdGUgc29sdXRpb247IHdpbGwgYm9ycm93IG9uZSBmcm9tIHt9Ii5mb3JtYXQoCiAgICAgICAgICAgICAgICByb2xlLCBkb25vclsibmFtZSJdKSkKICAgICAgICBlbGlmIHJvbGUgaW4gY2hvc2VuIGFuZCBub3QgY2hvc2VuW3JvbGVdWyJ3Y3MiXToKICAgICAgICAgICAgbm90ZXMuYXBwZW5kKCJ7fTogaGFzIG5vIHBsYXRlIHNvbHV0aW9uIGFuZCB0aGVyZSBpcyBubyBzb2x2ZWQgIgogICAgICAgICAgICAgICAgICAgICAgICAgImV4cG9zdXJlIGluIHRoZSBmb2xkZXIgdG8gdGFrZSBvbmUgZnJvbSIuZm9ybWF0KHJvbGUpKQogICAgcmV0dXJuIHJlY29yZHMsIGNob3NlbiwgbWVtYmVycywgZG9ub3JzLCBub3RlcwoKZGVzY3JpYmVfX2ZyYW1lX2ludmVudG9yeSA9IGRlc2NyaWJlICAgIyB0aGlzIG1vZHVsZSdzIG93bjsgdGhlIHBsYWluIG5hbWUgZ2V0cyBzaGFkb3dlZCBiZWxvdwoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KIyBzaGFyZWQgbW9kdWxlIGFuY2hvcl9yZWdpc3RyYXRpb24ucHkKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQoKZGVmIHBpY2tfYW5jaG9ycyhwb3NpdGlvbnMsIHNoYXBlLCBuX2FuY2hvcnM9NCwgZWRnZV9tYXJnaW49NDApOgogICAgIiIiQ2hvb3NlIGEgZmV3IHdpZGVseSBzZXBhcmF0ZWQgc3RhcnMgZnJvbSBhIGxpc3Qgb2YgY2FuZGlkYXRlIHBvc2l0aW9ucy4KCiAgICBTcHJlYWQgaXMgdGhlIHdob2xlIHBvaW50LiBUaGUgcm90YXRpb24gYW5nbGUgaXMgbWVhc3VyZWQgYXMgYSBkaWZmZXJlbmNlIG9mCiAgICBkaXJlY3Rpb25zIGJldHdlZW4gdHdvIHN0YXJzLCBzbyBpdHMgcHJlY2lzaW9uIGlzIHRoZSBjZW50cmluZyBwcmVjaXNpb24KICAgIGRpdmlkZWQgYnkgdGhlIGRpc3RhbmNlIGJldHdlZW4gdGhlbTogYW5jaG9ycyBjbG9zZSB0b2dldGhlciBtZWFzdXJlIHRoZQogICAgYW5nbGUgYmFkbHkgbm8gbWF0dGVyIGhvdyB3ZWxsIGVhY2ggb25lIGlzIGNlbnRyZWQuCgogICAgVGhlIGZyYW1lIGlzIGRpdmlkZWQgaW50byBxdWFkcmFudHMgYW5kIG9uZSBzdGFyIGlzIHRha2VuIGZyb20gZWFjaCwgYXMgZmFyCiAgICBmcm9tIHRoZSBjZW50cmUgYXMgaXQgY2FuIGJlIGZvdW5kLiBUaGF0IGd1YXJhbnRlZXMgdGhlIHNwcmVhZCBpbnN0ZWFkIG9mCiAgICBob3BpbmcgZm9yIGl0LgogICAgIiIiCiAgICBwb3NpdGlvbnMgPSBucC5hc2FycmF5KHBvc2l0aW9ucywgZHR5cGU9ZmxvYXQpCiAgICBpZiBsZW4ocG9zaXRpb25zKSA9PSAwOgogICAgICAgIHJldHVybiBwb3NpdGlvbnMKCiAgICBueSwgbnggPSBzaGFwZQogICAgaW5zaWRlID0gKChwb3NpdGlvbnNbOiwgMF0gPiBlZGdlX21hcmdpbikgJiAocG9zaXRpb25zWzosIDBdIDwgbnggLSBlZGdlX21hcmdpbikKICAgICAgICAgICAgICAmIChwb3NpdGlvbnNbOiwgMV0gPiBlZGdlX21hcmdpbikgJiAocG9zaXRpb25zWzosIDFdIDwgbnkgLSBlZGdlX21hcmdpbikpCiAgICBwb3NpdGlvbnMgPSBwb3NpdGlvbnNbaW5zaWRlXQogICAgaWYgbGVuKHBvc2l0aW9ucykgPD0gbl9hbmNob3JzOgogICAgICAgIHJldHVybiBwb3NpdGlvbnMKCiAgICBjeCwgY3kgPSBueCAvIDIuMCwgbnkgLyAyLjAKICAgIGR4LCBkeSA9IHBvc2l0aW9uc1s6LCAwXSAtIGN4LCBwb3NpdGlvbnNbOiwgMV0gLSBjeQogICAgcmFkaXVzID0gbnAuaHlwb3QoZHgsIGR5KQoKICAgIGNob3NlbiA9IFtdCiAgICBxdWFkcmFudHMgPSAoKGR4ID49IDApICYgKGR5ID49IDApLCAoZHggPCAwKSAmIChkeSA+PSAwKSwKICAgICAgICAgICAgICAgICAoZHggPCAwKSAmIChkeSA8IDApLCAoZHggPj0gMCkgJiAoZHkgPCAwKSkKICAgIGZvciBxIGluIHF1YWRyYW50czoKICAgICAgICBpZiBxLmFueSgpOgogICAgICAgICAgICBjaG9zZW4uYXBwZW5kKHBvc2l0aW9uc1tucC53aGVyZShxKVswXVtucC5hcmdtYXgocmFkaXVzW3FdKV1dKQoKICAgICMgQSBmaWVsZCB3aXRoIGFuIGVtcHR5IHF1YWRyYW50IHN0aWxsIGdldHMgaXRzIGZvdXIsIHRha2VuIGZyb20gd2hlcmV2ZXIgdGhlCiAgICAjIHN0YXJzIGFjdHVhbGx5IGFyZS4KICAgIGlmIGxlbihjaG9zZW4pIDwgbl9hbmNob3JzOgogICAgICAgIG9yZGVyID0gbnAuYXJnc29ydChyYWRpdXMpWzo6LTFdCiAgICAgICAgZm9yIGkgaW4gb3JkZXI6CiAgICAgICAgICAgIHAgPSBwb3NpdGlvbnNbaV0KICAgICAgICAgICAgaWYgbm90IGFueShucC5hbGxjbG9zZShwLCBjKSBmb3IgYyBpbiBjaG9zZW4pOgogICAgICAgICAgICAgICAgY2hvc2VuLmFwcGVuZChwKQogICAgICAgICAgICBpZiBsZW4oY2hvc2VuKSA+PSBuX2FuY2hvcnM6CiAgICAgICAgICAgICAgICBicmVhawoKICAgIHJldHVybiBucC5hcnJheShjaG9zZW5bOm5fYW5jaG9yc10pCgpkZWYgY2VudHJlX29uKGltYWdlLCBwb3NpdGlvbnMsIGJveD0xNSk6CiAgICAiIiJXaGVyZSBlYWNoIHN0YXIgYWN0dWFsbHkgc2l0cyBpbiB0aGlzIGZyYW1lLCB0byBhIGZyYWN0aW9uIG9mIGEgcGl4ZWwuIiIiCiAgICBmcm9tIHBob3R1dGlscy5jZW50cm9pZHMgaW1wb3J0IGNlbnRyb2lkX3NvdXJjZXMsIGNlbnRyb2lkX2NvbQoKICAgIHBvc2l0aW9ucyA9IG5wLmFzYXJyYXkocG9zaXRpb25zLCBkdHlwZT1mbG9hdCkKICAgIGlmIG5vdCBsZW4ocG9zaXRpb25zKToKICAgICAgICByZXR1cm4gbnAuZW1wdHkoKDAsIDIpKQogICAgdHJ5OgogICAgICAgIHgsIHkgPSBjZW50cm9pZF9zb3VyY2VzKG5wLm5hbl90b19udW0oaW1hZ2UpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHBvc2l0aW9uc1s6LCAwXSwgcG9zaXRpb25zWzosIDFdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGJveF9zaXplPWJveCwgY2VudHJvaWRfZnVuYz1jZW50cm9pZF9jb20pCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHJldHVybiBucC5lbXB0eSgoMCwgMikpCiAgICByZXR1cm4gbnAuY29sdW1uX3N0YWNrKFt4LCB5XSkKCmRlZiBhbmNob3JfdHJhbnNmb3JtKHJlZmVyZW5jZSwgaW1hZ2UsIHBvc2l0aW9ucywgYm94PTE1LCBtYXhfbW92ZV9weD04LjAsCiAgICAgICAgICAgICAgICAgICAgIG5fYW5jaG9ycz00KToKICAgICIiIkZpdCBzaGlmdCwgcm90YXRpb24gYW5kIHNjYWxlIGZyb20gYSBmZXcgYW5jaG9yIHN0YXJzLgoKICAgIFJldHVybnMgKHRyYW5zZm9ybSwgZGlhZ25vc3RpY3MpIHdoZXJlIHRyYW5zZm9ybSBtYXBzIGEgcG9zaXRpb24gaW4gYGltYWdlYAogICAgdG8gdGhlIG1hdGNoaW5nIHBvc2l0aW9uIGluIGByZWZlcmVuY2VgLCBvciAoTm9uZSwgZGlhZ25vc3RpY3MpIHdoZW4gdGhlCiAgICBhbmNob3JzIGNvdWxkIG5vdCBiZSBtZWFzdXJlZCB3ZWxsIGVub3VnaCB0byB0cnVzdC4KCiAgICBUaGUgb2RkIG9uZSBvdXQgaXMgZHJvcHBlZDogZWFjaCBhbmNob3IncyByZXNpZHVhbCBhZ2FpbnN0IHRoZSBmaXQgaXMKICAgIG1lYXN1cmVkLCBhbmQgaWYgb25lIHN0YW5kcyBmYXIgYWJvdmUgdGhlIHJlc3QgaXQgaXMgcmVtb3ZlZCBhbmQgdGhlIGZpdAogICAgcmVkb25lLiBUaGF0IGlzIHdoYXQgdGhlIGZvdXJ0aCBhbmNob3IgaXMgZm9yLgogICAgIiIiCiAgICBhbmNob3JzID0gcGlja19hbmNob3JzKHBvc2l0aW9ucywgcmVmZXJlbmNlLnNoYXBlLCBuX2FuY2hvcnM9bl9hbmNob3JzKQogICAgZGlhZyA9IHsibl9hbmNob3JzIjogbGVuKGFuY2hvcnMpLCAiZHJvcHBlZCI6IDB9CiAgICBpZiBsZW4oYW5jaG9ycykgPCAyOgogICAgICAgIGRpYWdbIndoeSJdID0gImZld2VyIHRoYW4gdHdvIHVzYWJsZSBhbmNob3Igc3RhcnMiCiAgICAgICAgcmV0dXJuIE5vbmUsIGRpYWcKCiAgICBhID0gY2VudHJlX29uKHJlZmVyZW5jZSwgYW5jaG9ycywgYm94PWJveCkKICAgIGIgPSBjZW50cmVfb24oaW1hZ2UsIGFuY2hvcnMsIGJveD1ib3gpCiAgICBpZiBsZW4oYSkgIT0gbGVuKGIpIG9yIG5vdCBsZW4oYSk6CiAgICAgICAgZGlhZ1sid2h5Il0gPSAidGhlIGFuY2hvcnMgY291bGQgbm90IGJlIGNlbnRyZWQiCiAgICAgICAgcmV0dXJuIE5vbmUsIGRpYWcKCiAgICBvayA9IChucC5pc2Zpbml0ZShhKS5hbGwoYXhpcz0xKSAmIG5wLmlzZmluaXRlKGIpLmFsbChheGlzPTEpCiAgICAgICAgICAmIChucC5oeXBvdCgqKGIgLSBhKS5UKSA8IG1heF9tb3ZlX3B4KSkKICAgIGEsIGIgPSBhW29rXSwgYltva10KICAgIGlmIGxlbihhKSA8IDI6CiAgICAgICAgZGlhZ1sid2h5Il0gPSAidGhlIGFuY2hvcnMgbW92ZWQgZnVydGhlciB0aGFuIGEgZnJhbWUgbGlrZSB0aGlzIHNob3VsZCIKICAgICAgICByZXR1cm4gTm9uZSwgZGlhZwoKICAgIGRlZiBmaXQocCwgcSk6CiAgICAgICAgIiIiTGVhc3Qtc3F1YXJlcyBzaW1pbGFyaXR5IHRyYW5zZm9ybSB0YWtpbmcgcSBvbnRvIHAuIiIiCiAgICAgICAgcGMsIHFjID0gcC5tZWFuKGF4aXM9MCksIHEubWVhbihheGlzPTApCiAgICAgICAgcDAsIHEwID0gcCAtIHBjLCBxIC0gcWMKICAgICAgICBkZW5vbSA9IChxMCAqKiAyKS5zdW0oKQogICAgICAgIGlmIGRlbm9tIDw9IDA6CiAgICAgICAgICAgIHJldHVybiBOb25lCiAgICAgICAgIyBUaGUgcm90YXRpb24tYW5kLXNjYWxlIHBhcnQsIGFzIG9uZSBjb21wbGV4IG11bHRpcGxpY2F0aW9uLgogICAgICAgIG51bSA9IGNvbXBsZXgoKHAwWzosIDBdICogcTBbOiwgMF0gKyBwMFs6LCAxXSAqIHEwWzosIDFdKS5zdW0oKSwKICAgICAgICAgICAgICAgICAgICAgIChwMFs6LCAxXSAqIHEwWzosIDBdIC0gcDBbOiwgMF0gKiBxMFs6LCAxXSkuc3VtKCkpCiAgICAgICAgeiA9IG51bSAvIGRlbm9tCiAgICAgICAgcmV0dXJuIHosIHBjLCBxYwoKICAgIGdvdCA9IGZpdChhLCBiKQogICAgaWYgZ290IGlzIE5vbmU6CiAgICAgICAgZGlhZ1sid2h5Il0gPSAidGhlIGFuY2hvcnMgYXJlIG9uIHRvcCBvZiBlYWNoIG90aGVyIgogICAgICAgIHJldHVybiBOb25lLCBkaWFnCiAgICB6LCBwYywgcWMgPSBnb3QKCiAgICBpZiBsZW4oYSkgPj0gNDoKICAgICAgICBwcmVkID0gX2FwcGx5KGIsIHosIHBjLCBxYykKICAgICAgICByZXNpZCA9IG5wLmh5cG90KCoocHJlZCAtIGEpLlQpCiAgICAgICAgd29yc3QgPSBpbnQobnAuYXJnbWF4KHJlc2lkKSkKICAgICAgICBvdGhlcnMgPSBucC5kZWxldGUocmVzaWQsIHdvcnN0KQogICAgICAgIGlmIHJlc2lkW3dvcnN0XSA+IG1heCgzLjAgKiBucC5tZWRpYW4ob3RoZXJzKSwgMC41KToKICAgICAgICAgICAgYSwgYiA9IG5wLmRlbGV0ZShhLCB3b3JzdCwgMCksIG5wLmRlbGV0ZShiLCB3b3JzdCwgMCkKICAgICAgICAgICAgZGlhZ1siZHJvcHBlZCJdID0gMQogICAgICAgICAgICB6LCBwYywgcWMgPSBmaXQoYSwgYikKCiAgICBwcmVkID0gX2FwcGx5KGIsIHosIHBjLCBxYykKICAgIHJlc2lkID0gbnAuaHlwb3QoKihwcmVkIC0gYSkuVCkKICAgIGRpYWcudXBkYXRlKHsKICAgICAgICAibl91c2VkIjogbGVuKGEpLAogICAgICAgICJzaGlmdF9weCI6IChmbG9hdChwY1swXSAtIHFjWzBdICogei5yZWFsICsgcWNbMV0gKiB6LmltYWcpLAogICAgICAgICAgICAgICAgICAgICBmbG9hdChwY1sxXSAtIHFjWzBdICogei5pbWFnIC0gcWNbMV0gKiB6LnJlYWwpKSwKICAgICAgICAicm90YXRpb25fZGVnIjogZmxvYXQobnAuZGVncmVlcyhucC5hcmN0YW4yKHouaW1hZywgei5yZWFsKSkpLAogICAgICAgICJzY2FsZSI6IGZsb2F0KGFicyh6KSksCiAgICAgICAgInJlc2lkdWFsX3B4IjogZmxvYXQobnAubWF4KHJlc2lkKSksCiAgICAgICAgInNlcGFyYXRpb25fcHgiOiBmbG9hdChucC5oeXBvdCgqKGEubWF4KGF4aXM9MCkgLSBhLm1pbihheGlzPTApKSkpLAogICAgfSkKICAgIHJldHVybiAoeiwgcGMsIHFjKSwgZGlhZwoKZGVmIF9hcHBseShwb2ludHMsIHosIHBjLCBxYyk6CiAgICAiIiJNYXAgcG9pbnRzIHRocm91Z2ggYSBmaXR0ZWQgc2ltaWxhcml0eSB0cmFuc2Zvcm0uIiIiCiAgICBxMCA9IG5wLmFzYXJyYXkocG9pbnRzLCBkdHlwZT1mbG9hdCkgLSBxYwogICAgeCA9IHEwWzosIDBdICogei5yZWFsIC0gcTBbOiwgMV0gKiB6LmltYWcgKyBwY1swXQogICAgeSA9IHEwWzosIDBdICogei5pbWFnICsgcTBbOiwgMV0gKiB6LnJlYWwgKyBwY1sxXQogICAgcmV0dXJuIG5wLmNvbHVtbl9zdGFjayhbeCwgeV0pCgpkZWYgYXBwbHlfdHJhbnNmb3JtKGltYWdlLCB0cmFuc2Zvcm0sIHNoYXBlPU5vbmUpOgogICAgIiIiUmVzYW1wbGUgYSBmcmFtZSB0aHJvdWdoIGEgdHJhbnNmb3JtIGZpdHRlZCBmcm9tIHRoZSBhbmNob3JzLgoKICAgIE9uZSBhZmZpbmUgaW50ZXJwb2xhdGlvbiwgYW5kIG5vIGFzdHJvbWV0cmljIGFyaXRobWV0aWMgYXQgYWxsLiBUaGF0IGlzIHdoZXJlCiAgICB0aGUgdGltZSBnb2VzOiB0aGUgdmVyc2lvbiB0aGF0IHdlbnQgdGhyb3VnaCBib3RoIHBsYXRlIHNvbHV0aW9ucyBoYWQgdG8gdHVybgogICAgc2l4dHktb25lIG1pbGxpb24gcGl4ZWwgcG9zaXRpb25zIGludG8gc2t5IHBvc2l0aW9ucyBhbmQgYmFjayBhZ2FpbiBiZWZvcmUgaXQKICAgIGNvdWxkIGludGVycG9sYXRlIGFueXRoaW5nLgogICAgIiIiCiAgICBmcm9tIHNjaXB5Lm5kaW1hZ2UgaW1wb3J0IGFmZmluZV90cmFuc2Zvcm0KCiAgICB6LCBwYywgcWMgPSB0cmFuc2Zvcm0KICAgIHNoYXBlID0gc2hhcGUgb3IgaW1hZ2Uuc2hhcGUKCiAgICAjIGFmZmluZV90cmFuc2Zvcm0gcHVsbHM6IGZvciBlYWNoIG91dHB1dCBwaXhlbCBpdCBhc2tzIHdoZXJlIGluIHRoZSBpbnB1dCB0bwogICAgIyBsb29rLCBzbyB0aGUgaW52ZXJzZSBvZiB0aGUgZml0dGVkIG1hcCBpcyB3aGF0IGl0IG5lZWRzLCB3cml0dGVuIGluIHRoZQogICAgIyAocm93LCBjb2x1bW4pIG9yZGVyIGl0IGV4cGVjdHMuCiAgICBpbnYgPSAxLjAgLyB6CiAgICBtID0gbnAuYXJyYXkoW1tpbnYucmVhbCwgaW52LmltYWddLCBbLWludi5pbWFnLCBpbnYucmVhbF1dLCBkdHlwZT1mbG9hdCkKICAgIG9mZnNldCA9IG5wLmFycmF5KFtxY1sxXSwgcWNbMF1dKSAtIG0uZG90KG5wLmFycmF5KFtwY1sxXSwgcGNbMF1dKSkKCiAgICBrbm93biA9IG5wLmlzZmluaXRlKGltYWdlKQogICAgZmlsbGVkID0gbnAud2hlcmUoa25vd24sIGltYWdlLCAwLjApLmFzdHlwZShucC5mbG9hdDMyKQogICAgb3V0ID0gYWZmaW5lX3RyYW5zZm9ybShmaWxsZWQsIG0sIG9mZnNldD1vZmZzZXQsIG91dHB1dF9zaGFwZT1zaGFwZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgb3JkZXI9MSwgbW9kZT0iY29uc3RhbnQiLCBjdmFsPW5wLm5hbikKICAgIHNlZW4gPSBhZmZpbmVfdHJhbnNmb3JtKGtub3duLmFzdHlwZShucC5mbG9hdDMyKSwgbSwgb2Zmc2V0PW9mZnNldCwKICAgICAgICAgICAgICAgICAgICAgICAgICAgIG91dHB1dF9zaGFwZT1zaGFwZSwgb3JkZXI9MSwgbW9kZT0iY29uc3RhbnQiLCBjdmFsPTAuMCkKICAgIG91dFtzZWVuIDwgMC41XSA9IG5wLm5hbgogICAgcmV0dXJuIG91dAoKZGVmIGRlc2NyaWJlKGRpYWcpOgogICAgIiIiVGhlIHJlZ2lzdHJhdGlvbiwgaW4gYSBsaW5lIHdvcnRoIHByaW50aW5nLiIiIgogICAgaWYgZGlhZy5nZXQoIndoeSIpOgogICAgICAgIHJldHVybiAiICBhbmNob3JzIHVudXNhYmxlICh7fSk7IGZhbGxpbmcgYmFjayIuZm9ybWF0KGRpYWdbIndoeSJdKQogICAgcmV0dXJuICgiICB7fSBhbmNob3Igc3RhcnN7fSwgezouMGZ9IHB4IGFwYXJ0OiBzaGlmdCAoezorLjJmfSwgezorLjJmfSkgcHgsICIKICAgICAgICAgICAgInJvdGF0aW9uIHs6Ky40Zn0gZGVnLCBzY2FsZSB7Oi42Zn07IHdvcnN0IGFuY2hvciBvZmYgYnkgezouMmZ9IHB4IgogICAgICAgICAgICAuZm9ybWF0KGRpYWdbIm5fdXNlZCJdLAogICAgICAgICAgICAgICAgICAgICIgKG9uZSBkcm9wcGVkKSIgaWYgZGlhZy5nZXQoImRyb3BwZWQiKSBlbHNlICIiLAogICAgICAgICAgICAgICAgICAgIGRpYWdbInNlcGFyYXRpb25fcHgiXSwgZGlhZ1sic2hpZnRfcHgiXVswXSwgZGlhZ1sic2hpZnRfcHgiXVsxXSwKICAgICAgICAgICAgICAgICAgICBkaWFnWyJyb3RhdGlvbl9kZWciXSwgZGlhZ1sic2NhbGUiXSwgZGlhZ1sicmVzaWR1YWxfcHgiXSkpCgpkZXNjcmliZV9fYW5jaG9yX3JlZ2lzdHJhdGlvbiA9IGRlc2NyaWJlICAgIyB0aGlzIG1vZHVsZSdzIG93bjsgdGhlIHBsYWluIG5hbWUgZ2V0cyBzaGFkb3dlZCBiZWxvdwoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KIyBzaGFyZWQgbW9kdWxlIGdhbGF4eV93aW5kb3cucHkKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQoKZGVmIGdhbGF4eV93aW5kb3coc2hhcGUsIGNlbnRyZSwgcjI1X3B4LCBtYXJnaW49MS41KToKICAgICIiIlRoZSBzbGljZSBvZiBhIGZyYW1lIHdvcnRoIHdvcmtpbmcgb24sIGFzICh5MCwgeTEsIHgwLCB4MSkuCgogICAgUmV0dXJucyBOb25lIHdoZW4gdGhlcmUgaXMgbm8gcmFkaXVzIHRvIHdvcmsgZnJvbSwgb3Igd2hlbiB0aGUgd2luZG93IHdvdWxkCiAgICBjb3ZlciBlc3NlbnRpYWxseSB0aGUgd2hvbGUgZnJhbWUgYW55d2F5IC0gaW4gd2hpY2ggY2FzZSBjcm9wcGluZyB3b3VsZCBhZGQgYQogICAgYm9va2tlZXBpbmcgc3RlcCBhbmQgc2F2ZSBub3RoaW5nLgogICAgIiIiCiAgICBpZiBub3QgcjI1X3B4IG9yIHIyNV9weCA8PSAwOgogICAgICAgIHJldHVybiBOb25lCgogICAgbnksIG54ID0gc2hhcGUKICAgIHJlYWNoID0gbWFyZ2luICogcjI1X3B4CiAgICB4MCA9IGludChtYXgoMCwgbnAuZmxvb3IoY2VudHJlWzBdIC0gcmVhY2gpKSkKICAgIHgxID0gaW50KG1pbihueCwgbnAuY2VpbChjZW50cmVbMF0gKyByZWFjaCkpKQogICAgeTAgPSBpbnQobWF4KDAsIG5wLmZsb29yKGNlbnRyZVsxXSAtIHJlYWNoKSkpCiAgICB5MSA9IGludChtaW4obnksIG5wLmNlaWwoY2VudHJlWzFdICsgcmVhY2gpKSkKCiAgICBpZiB4MSAtIHgwIDwgNjQgb3IgeTEgLSB5MCA8IDY0OgogICAgICAgIHJldHVybiBOb25lCiAgICBpZiAoeDEgLSB4MCkgKiAoeTEgLSB5MCkgPiAwLjkgKiBueCAqIG55OgogICAgICAgIHJldHVybiBOb25lCiAgICByZXR1cm4gKHkwLCB5MSwgeDAsIHgxKQoKZGVmIGN1dChkYXRhLCB3aW5kb3cpOgogICAgIiIiVGhlIHdpbmRvdydzIHdvcnRoIG9mIGEgZnJhbWUuIiIiCiAgICBpZiB3aW5kb3cgaXMgTm9uZSBvciBkYXRhIGlzIE5vbmU6CiAgICAgICAgcmV0dXJuIGRhdGEKICAgIHkwLCB5MSwgeDAsIHgxID0gd2luZG93CiAgICByZXR1cm4gZGF0YVt5MDp5MSwgeDA6eDFdCgpkZWYgc2hpZnRfY2VudHJlKGNlbnRyZSwgd2luZG93KToKICAgICIiIldoZXJlIHRoZSBnYWxheHkncyBjZW50cmUgc2l0cyBvbmNlIHRoZSBmcmFtZSBoYXMgYmVlbiBjdXQuIiIiCiAgICBpZiB3aW5kb3cgaXMgTm9uZToKICAgICAgICByZXR1cm4gY2VudHJlCiAgICB5MCwgXywgeDAsIF8gPSB3aW5kb3cKICAgIHJldHVybiAoY2VudHJlWzBdIC0geDAsIGNlbnRyZVsxXSAtIHkwKQoKZGVmIHNoaWZ0X3Bvc2l0aW9ucyhwb3NpdGlvbnMsIHdpbmRvdywgc2hhcGU9Tm9uZSk6CiAgICAiIiJTdGFyIHBvc2l0aW9ucyBtb3ZlZCBpbnRvIHRoZSBjdXQgZnJhbWUsIGtlZXBpbmcgb25seSB0aG9zZSBpbnNpZGUgaXQuIiIiCiAgICBpZiB3aW5kb3cgaXMgTm9uZSBvciBwb3NpdGlvbnMgaXMgTm9uZSBvciBub3QgbGVuKHBvc2l0aW9ucyk6CiAgICAgICAgcmV0dXJuIHBvc2l0aW9ucwogICAgeTAsIHkxLCB4MCwgeDEgPSB3aW5kb3cKICAgIHAgPSBucC5hc2FycmF5KHBvc2l0aW9ucywgZHR5cGU9ZmxvYXQpCiAgICBpbnNpZGUgPSAoKHBbOiwgMF0gPj0geDApICYgKHBbOiwgMF0gPCB4MSkgJiAocFs6LCAxXSA+PSB5MCkgJiAocFs6LCAxXSA8IHkxKSkKICAgIHJldHVybiBucC5jb2x1bW5fc3RhY2soW3BbaW5zaWRlLCAwXSAtIHgwLCBwW2luc2lkZSwgMV0gLSB5MF0pCgpkZWYgcmVzdG9yZV9wb3NpdGlvbnMoeCwgeSwgd2luZG93KToKICAgICIiIlBvc2l0aW9ucyBtZWFzdXJlZCBpbiB0aGUgY3V0IGZyYW1lLCBwdXQgYmFjayB3aGVyZSB0aGV5IGJlbG9uZy4KCiAgICBFdmVyeXRoaW5nIHRoZSB0b29sIHJlcG9ydHMgLSBjYXRhbG9ndWVzLCBwbG90cywgdGhlIHBhaXJpbmcgd2l0aCB0aGUgYmx1ZQogICAga25vdHMgLSBpcyBpbiB0aGUgY29vcmRpbmF0ZXMgb2YgdGhlIG9yaWdpbmFsIGZyYW1lLCBzbyBtZWFzdXJlbWVudHMgbWFkZSBpbgogICAgdGhlIHdpbmRvdyBoYXZlIHRvIGNvbWUgYmFjayBvdXQgb2YgaXQuIEZvcmdldHRpbmcgdGhpcyB3b3VsZCBzaGlmdCBldmVyeQogICAgcG9zaXRpb24gYnkgdGhlIHNpemUgb2YgdGhlIGNyb3AsIHdoaWNoIGlzIHRoZSBraW5kIG9mIGVycm9yIHRoYXQgbG9va3MgbGlrZQogICAgYSBkaXNjb3ZlcnkuCiAgICAiIiIKICAgIGlmIHdpbmRvdyBpcyBOb25lOgogICAgICAgIHJldHVybiB4LCB5CiAgICB5MCwgXywgeDAsIF8gPSB3aW5kb3cKICAgIHJldHVybiBucC5hc2FycmF5KHgsIGR0eXBlPWZsb2F0KSArIHgwLCBucC5hc2FycmF5KHksIGR0eXBlPWZsb2F0KSArIHkwCgpkZWYgZGVzY3JpYmUod2luZG93LCBzaGFwZSwgbWFyZ2luKToKICAgICIiIldoYXQgd2FzIGN1dCwgYW5kIHdoYXQgaXQgc2F2ZWQuIiIiCiAgICBpZiB3aW5kb3cgaXMgTm9uZToKICAgICAgICByZXR1cm4gKCIgIHdvcmtpbmcgb24gdGhlIHdob2xlIGZyYW1lIC0gZWl0aGVyIG5vIGlzb3Bob3RhbCByYWRpdXMgaXMgIgogICAgICAgICAgICAgICAgImtub3duLCBvciB0aGUgZ2FsYXh5IGZpbGxzIGl0IikKICAgIHkwLCB5MSwgeDAsIHgxID0gd2luZG93CiAgICBrZXB0ID0gKHkxIC0geTApICogKHgxIC0geDApCiAgICB3aG9sZSA9IHNoYXBlWzBdICogc2hhcGVbMV0KICAgIHJldHVybiAoIiAgd29ya2luZyBvbiB7fSB4IHt9IHBpeGVscyBhcm91bmQgdGhlIGdhbGF4eSwgezouMGZ9JSBvZiB0aGUgZnJhbWUgIgogICAgICAgICAgICAiKHs6LjFmfSB4IFIyNSk7IHRoZSByZXN0IGlzIHNreSB0aGF0IHdvdWxkIGJlIGRpc2NhcmRlZCBhbnl3YXkiCiAgICAgICAgICAgIC5mb3JtYXQoeDEgLSB4MCwgeTEgLSB5MCwgMTAwLjAgKiBrZXB0IC8gd2hvbGUsIG1hcmdpbikpCgpkZXNjcmliZV9fZ2FsYXh5X3dpbmRvdyA9IGRlc2NyaWJlICAgIyB0aGlzIG1vZHVsZSdzIG93bjsgdGhlIHBsYWluIG5hbWUgZ2V0cyBzaGFkb3dlZCBiZWxvdwoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KIyBzaGFyZWQgbW9kdWxlIGdhbGF4eV9leHRlbnQucHkKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQoKZGVmIHN1cmZhY2VfYnJpZ2h0bmVzc19wcm9maWxlKGRhdGEsIGNlbnRyZSwgcl9tYXg9Tm9uZSwgbl9yaW5ncz00MCwgc3RlcD00LAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcGVyY2VudGlsZT05MC4wKToKICAgICIiIkhvdyBtdWNoIGxpZ2h0IGVhY2ggcmluZyBob2xkcywgYWJvdmUgd2hhdCBhbiBlbXB0eSByaW5nIGhvbGRzLgoKICAgIFdIWSBBIEhJR0ggUEVSQ0VOVElMRSBBTkQgTk9UIFRIRSBNRURJQU4KICAgIEJlY2F1c2Ugb2Ygd2hhdCBpcyBiZWluZyBtZWFzdXJlZC4gQSBzdGVsbGFyIGRpc2sgaXMgc21vb3RoLCBhbmQgaXRzIG1lZGlhbgogICAgZGVzY3JpYmVzIGl0LiBMaW5lIGVtaXNzaW9uIGlzIG5vdCBzbW9vdGg6IGl0IGxpdmVzIGluIGRpc2NyZXRlIHJlZ2lvbnMgd2l0aAogICAgZW1wdHkgc2t5IGJldHdlZW4gdGhlbSwgc28gYSByaW5nIGhvbGRpbmcgYSBkb3plbiBicmlnaHQgSElJIHJlZ2lvbnMgYW5kCiAgICB0d2VudHkgdGhvdXNhbmQgZW1wdHkgcGl4ZWxzIGhhcyBhIG1lZGlhbiBvZiB6ZXJvLiBNZWFzdXJlZCBvbiBNODMsIHRoZSBtZWRpYW4KICAgIG9mIHRoZSBzdWJ0cmFjdGVkIGZyYW1lIHJlYWNoZWQgdGhlIG5vaXNlIGF0IDI1NiBweCB3aGlsZSB0aGUgcmVnaW9ucwogICAgdGhlbXNlbHZlcyB3ZW50IG9uIHRvIDU1NiAtIGFuZCB1c2luZyB0aGUgbWVkaWFuIGN1dCB0aGUgY2F0YWxvZ3VlIGZyb20gMzM1CiAgICByZWdpb25zIHRvIDEyMi4KCiAgICBUaGUgbmluZXRpZXRoIHBlcmNlbnRpbGUgYXNrcyB0aGUgdXNlZnVsIHF1ZXN0aW9uIGluc3RlYWQ6IGlzIHRoZXJlIGFueXRoaW5nCiAgICBicmlnaHQgaW4gdGhpcyByaW5nLiBJdCB3b3JrcyBmb3IgYSBzbW9vdGggdHJhY2VyIHRvbywgd2hlcmUgaXQgc2ltcGx5IHRyYWNrcwogICAgdGhlIG1lZGlhbiB3aXRoIGFuIG9mZnNldC4KCiAgICBXSFkgVEhFIEZBUiBGSUVMRCBJUyBTVUJUUkFDVEVECiAgICBBIGhpZ2ggcGVyY2VudGlsZSBvZiBwdXJlIG5vaXNlIGlzIG5vdCB6ZXJvIC0gaXQgaXMgYWJvdXQgMS4zIHRpbWVzIHRoZSBub2lzZSwKICAgIGJlY2F1c2UgaXQgaXMgcGlja2luZyB0aGUgYnJpZ2h0ZXN0IHRlbnRoIG9mIHRoZSBzYW1wbGUuIENvbXBhcmluZyBhZ2FpbnN0CiAgICB6ZXJvIHdvdWxkIGNhbGwgZW1wdHkgc2t5ICJzdGlsbCBnYWxheHkiIGZvciBldmVyLiBTbyB0aGUgc2FtZSBzdGF0aXN0aWMgaXMKICAgIG1lYXN1cmVkIGZhciBvdXQsIHdoZXJlIHRoZXJlIGlzIG5vdGhpbmcsIGFuZCB0aGF0IGZsb29yIGlzIHRha2VuIG9mZi4KCiAgICBTYW1wbGVkIGV2ZXJ5IGZldyBwaXhlbHMgcmF0aGVyIHRoYW4gZXZlcnkgb25lOiB0aGlzIGlzIGEgc2hhcGUgbWVhc3VyZW1lbnQKICAgIG92ZXIgdGhvdXNhbmRzIG9mIHBpeGVscyBhIHJpbmcsIGFuZCByZWFkaW5nIGEgcXVhcnRlciBvZiB0aGVtIGNoYW5nZXMgbm90aGluZwogICAgd2hpbGUgbWFraW5nIGl0IHF1aWNrIGVub3VnaCBmb3IgYSA2MS1tZWdhcGl4ZWwgZnJhbWUuCiAgICAiIiIKICAgIGZyb20gYXN0cm9weS5zdGF0cyBpbXBvcnQgc2lnbWFfY2xpcHBlZF9zdGF0cwoKICAgIG55LCBueCA9IGRhdGEuc2hhcGUKICAgIHl5LCB4eCA9IG5wLm1ncmlkWzA6bnk6c3RlcCwgMDpueDpzdGVwXQogICAgciA9IG5wLmh5cG90KHh4IC0gY2VudHJlWzBdLCB5eSAtIGNlbnRyZVsxXSkucmF2ZWwoKQogICAgdiA9IG5wLmFzYXJyYXkoZGF0YVs6OnN0ZXAsIDo6c3RlcF0sIGR0eXBlPWZsb2F0KS5yYXZlbCgpCiAgICBvayA9IG5wLmlzZmluaXRlKHYpCiAgICByLCB2ID0gcltva10sIHZbb2tdCiAgICBpZiBub3QgbGVuKHIpOgogICAgICAgIHJldHVybiBOb25lCgogICAgaWYgcl9tYXggaXMgTm9uZToKICAgICAgICByX21heCA9IGZsb2F0KG5wLnBlcmNlbnRpbGUociwgOTkpKQogICAgZmFyID0gciA+IDAuOCAqIHIubWF4KCkKICAgIGlmIGZhci5zdW0oKSA8IDUwMDoKICAgICAgICBmYXIgPSByID4gbnAucGVyY2VudGlsZShyLCA4MCkKICAgIHNreSA9IGZsb2F0KG5wLm1lZGlhbih2W2Zhcl0pKQogICAgbm9pc2UgPSBmbG9hdChzaWdtYV9jbGlwcGVkX3N0YXRzKHZbZmFyXSwgc2lnbWE9My4wKVsyXSkKICAgIGlmIG5vdCBucC5pc2Zpbml0ZShub2lzZSkgb3Igbm9pc2UgPD0gMDoKICAgICAgICByZXR1cm4gTm9uZQoKICAgICMgV2hhdCB0aGlzIHN0YXRpc3RpYyByZWFkcyBvbiBlbXB0eSBza3kuIEV2ZXJ5dGhpbmcgaXMgbWVhc3VyZWQgYWdhaW5zdCBpdC4KICAgIGZsb29yID0gZmxvYXQobnAucGVyY2VudGlsZSh2W2Zhcl0sIHBlcmNlbnRpbGUpKQoKICAgIGVkZ2VzID0gbnAubGluc3BhY2UoMC4wLCByX21heCwgbl9yaW5ncyArIDEpCiAgICBjZW50cmVzLCBleGNlc3MgPSBbXSwgW10KICAgIGZvciBhLCBiIGluIHppcChlZGdlc1s6LTFdLCBlZGdlc1sxOl0pOgogICAgICAgIG0gPSAociA+PSBhKSAmIChyIDwgYikKICAgICAgICBpZiBtLnN1bSgpIDwgNTA6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgY2VudHJlcy5hcHBlbmQoMC41ICogKGEgKyBiKSkKICAgICAgICBleGNlc3MuYXBwZW5kKChmbG9hdChucC5wZXJjZW50aWxlKHZbbV0sIHBlcmNlbnRpbGUpKSAtIGZsb29yKSAvIG5vaXNlKQogICAgcmV0dXJuIG5wLmFycmF5KGNlbnRyZXMpLCBucC5hcnJheShleGNlc3MpLCBza3ksIG5vaXNlCgpkZWYgdmlzaWJsZV9yYWRpdXMoZGF0YSwgY2VudHJlLCB0aHJlc2hvbGQ9MC41LCByX21heD1Ob25lLCBwZXJjZW50aWxlPTkwLjApOgogICAgIiIiV2hlcmUgdGhlIGdhbGF4eSdzIGxpZ2h0IGZhbGxzIGJlbG93IGEgZnJhY3Rpb24gb2YgdGhlIHNreSBub2lzZS4KCiAgICBgdGhyZXNob2xkYCBpcyBpbiB1bml0cyBvZiB0aGF0IG5vaXNlLCBhbmQgaXQgaXMgdGhlIG9uZSByZWFsIGNob2ljZSBoZXJlLgogICAgSGFsZiBhIHNpZ21hIGlzIHRoZSBkZWZhdWx0IGJlY2F1c2UgdGhlIHR3byBlcnJvcnMgYXJlIG5vdCBzeW1tZXRyaWM6IHRvbwogICAgbGFyZ2UgYSByYWRpdXMgbGV0cyBlbXB0eSBza3kgaW50byB0aGUgbWVhc3VyZW1lbnQgYW5kIGRpbHV0ZXMgaXQsIHdoaWNoIGlzCiAgICB3aGF0IHdhcyBoYXBwZW5pbmcsIHdoaWxlIHRvbyBzbWFsbCBhIHJhZGl1cyBkZWxldGVzIHRoZSBvdXRlciByZWdpb25zIC0gYW5kCiAgICB0aGUgb3V0c2tpcnRzIGFyZSB3aGVyZSBhIHJhZGlhbCBwcm9maWxlIGlzIG1vc3Qgc2Vuc2l0aXZlLCBzaW5jZSB0aGUgZGVuc2l0eQogICAgaXMgbG93ZXN0IHRoZXJlLiBIYWxmIGEgc2lnbWEgc2l0cyB3aGVyZSB0aGUgcHJvZmlsZSBoYXMgY2xlYXJseSBmbGF0dGVuZWQgYnV0CiAgICBoYXMgbm90IHlldCByZWFjaGVkIHplcm8uCgogICAgUmV0dXJucyAocmFkaXVzIGluIHBpeGVscywgZGlhZ25vc3RpY3MpLCBvciAoTm9uZSwgZGlhZ25vc3RpY3MpIHdoZW4gdGhlCiAgICBwcm9maWxlIG5ldmVyIHJpc2VzIGFib3ZlIHRoZSB0aHJlc2hvbGQgYXQgYWxsIC0gYSBmcmFtZSB3aXRoIG5vIGdhbGF4eSBpbiBpdCwKICAgIG9yIGEgY2VudHJlIGluIHRoZSB3cm9uZyBwbGFjZS4KICAgICIiIgogICAgZ290ID0gc3VyZmFjZV9icmlnaHRuZXNzX3Byb2ZpbGUoZGF0YSwgY2VudHJlLCByX21heD1yX21heCwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHBlcmNlbnRpbGU9cGVyY2VudGlsZSkKICAgIGlmIGdvdCBpcyBOb25lOgogICAgICAgIHJldHVybiBOb25lLCB7IndoeSI6ICJjb3VsZCBub3QgbWVhc3VyZSBhIGJhY2tncm91bmQgdG8gY29tcGFyZSBhZ2FpbnN0In0KICAgIHJhZGlpLCBleGNlc3MsIHNreSwgbm9pc2UgPSBnb3QKCiAgICBhYm92ZSA9IGV4Y2VzcyA+IHRocmVzaG9sZAogICAgaWYgbm90IGFib3ZlLmFueSgpOgogICAgICAgIHJldHVybiBOb25lLCB7IndoeSI6ICJub3RoaW5nIHJpc2VzIGFib3ZlIHs6LjFmfSBzaWdtYSBvZiB0aGUgc2t5IG5vaXNlICIKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAiYW55d2hlcmUiLmZvcm1hdCh0aHJlc2hvbGQpLAogICAgICAgICAgICAgICAgICAgICAgInNreSI6IHNreSwgIm5vaXNlIjogbm9pc2V9CgogICAgIyBUaGUgbGFzdCByaW5nIGFib3ZlIHRoZSB0aHJlc2hvbGQsIHdhbGtpbmcgb3V0d2FyZHMgLSBub3QgdGhlIGZpcnN0IG9uZQogICAgIyBiZWxvdyBpdCwgc28gdGhhdCBhIHNpbmdsZSBub2lzeSByaW5nIGluIHRoZSBtaWRkbGUgZG9lcyBub3QgY3V0IHRoZSBnYWxheHkKICAgICMgaW4gaGFsZi4KICAgIGxhc3QgPSBpbnQobnAud2hlcmUoYWJvdmUpWzBdLm1heCgpKQogICAgcmFkaXVzID0gZmxvYXQocmFkaWlbbGFzdF0pCiAgICByZXR1cm4gcmFkaXVzLCB7InNreSI6IHNreSwgIm5vaXNlIjogbm9pc2UsICJ0aHJlc2hvbGQiOiB0aHJlc2hvbGQsCiAgICAgICAgICAgICAgICAgICAgImNlbnRyYWxfZXhjZXNzIjogZmxvYXQoZXhjZXNzWzBdKSBpZiBsZW4oZXhjZXNzKSBlbHNlIGZsb2F0KCJuYW4iKSwKICAgICAgICAgICAgICAgICAgICAicmFkaWkiOiByYWRpaSwgImV4Y2VzcyI6IGV4Y2Vzc30KCmRlZiBkZXNjcmliZShyYWRpdXMsIGRpYWcsIHIyNV9weD1Ob25lLCBsYWJlbD0iIik6CiAgICAiIiJXaGF0IHdhcyBtZWFzdXJlZCwgbmV4dCB0byB3aGF0IHRoZSBjYXRhbG9ndWUgc2F5cy4iIiIKICAgIGlmIHJhZGl1cyBpcyBOb25lOgogICAgICAgIHJldHVybiAiICB7fTogY291bGQgbm90IG1lYXN1cmUgYW4gZXh0ZW50ICh7fSkiLmZvcm1hdCgKICAgICAgICAgICAgbGFiZWwsIGRpYWcuZ2V0KCJ3aHkiLCAidW5rbm93biIpKQogICAgbGluZSA9ICgiICB7fTogdGhlIGxpZ2h0IGZhbGxzIGJlbG93IHs6LjFmfSB4IHRoZSBza3kgbm9pc2UgYXQgezouMGZ9IHB4IgogICAgICAgICAgICAuZm9ybWF0KGxhYmVsLCBkaWFnLmdldCgidGhyZXNob2xkIiwgMC41KSwgcmFkaXVzKSkKICAgIGlmIHIyNV9weDoKICAgICAgICBsaW5lICs9ICI7IHRoZSBjYXRhbG9ndWUncyBSMjUgaXMgezouMGZ9IHB4Ii5mb3JtYXQocjI1X3B4KQogICAgICAgIGlmIHJhZGl1cyA8IDAuOCAqIHIyNV9weDoKICAgICAgICAgICAgbGluZSArPSAoIlxuICAgIHRoZSBleHBvc3VyZSBkb2VzIG5vdCByZWFjaCB0aGUgY2F0YWxvZ3VlIHJhZGl1cywgc28gIgogICAgICAgICAgICAgICAgICAgICAidGhlIG1lYXN1cmVkIG9uZSBpcyB1c2VkIikKICAgICAgICBlbGlmIHJhZGl1cyA+IDEuMiAqIHIyNV9weDoKICAgICAgICAgICAgbGluZSArPSAoIlxuICAgIHRoZSBsaWdodCByZWFjaGVzIHBhc3QgdGhlIGNhdGFsb2d1ZSByYWRpdXM7IHRoZSAiCiAgICAgICAgICAgICAgICAgICAgICJjYXRhbG9ndWUgb25lIGlzIHVzZWQsIGJlaW5nIHRoZSBtb3JlIHN0YW5kYXJkIikKICAgIHJldHVybiBsaW5lCgpkZWYgd29ya2luZ19yYWRpdXMobWVhc3VyZWQsIHIyNV9weCk6CiAgICAiIiJUaGUgcmFkaXVzIHRvIGFjdHVhbGx5IHVzZTogdGhlIHNtYWxsZXIgb2Ygd2hhdCBpcyB0aGVyZSBhbmQgd2hhdCBpcyBwdWJsaXNoZWQuCgogICAgU21hbGxlciBvZiB0aGUgdHdvIG9uIHB1cnBvc2UuIElmIHRoZSBleHBvc3VyZSBkb2VzIG5vdCByZWFjaCBSMjUgdGhlbiB0aGUKICAgIHNwYWNlIGJldHdlZW4gaXMgZW1wdHkgc2t5IGFuZCBpbmNsdWRpbmcgaXQgb25seSBhZGRzIG5vaXNlLiBJZiB0aGUgbGlnaHQKICAgIHJlYWNoZXMgcGFzdCBSMjUgLSB3aGljaCBoYXBwZW5zIGZvciBhIGJyaWdodCBpbm5lciBkaXNrIG1lYXN1cmVkIGdlbmVyb3VzbHkgLQogICAgdGhlbiBSMjUgaXMgdGhlIHN0YW5kYXJkIGRlZmluaXRpb24gYW5kIHRoZXJlIGlzIG5vIHJlYXNvbiB0byBpbnZlbnQgYSB3aWRlcgogICAgb25lLgogICAgIiIiCiAgICBpZiBtZWFzdXJlZCBhbmQgcjI1X3B4OgogICAgICAgIHJldHVybiBtaW4obWVhc3VyZWQsIHIyNV9weCkKICAgIHJldHVybiBtZWFzdXJlZCBvciByMjVfcHgKCmRlc2NyaWJlX19nYWxheHlfZXh0ZW50ID0gZGVzY3JpYmUgICAjIHRoaXMgbW9kdWxlJ3Mgb3duOyB0aGUgcGxhaW4gbmFtZSBnZXRzIHNoYWRvd2VkIGJlbG93CgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIHNoYXJlZCBtb2R1bGUgZGVwcm9qZWN0aW9uLnB5CiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KCklOQ19NSU5fREVHID0gMzAuMAoKSU5DX01BWF9ERUcgPSA3MC4wCgpRMF9ERUZBVUxUID0gMC4xMwoKUTBfQllfVFlQRSA9IHsiZWFybHkiOiAwLjIwLCAibWlkIjogMC4xMywgImxhdGUiOiAwLjEwfQoKZGVmIHEwX2Zvcl90eXBlKG10eXBlOiBzdHIgfCBOb25lKSAtPiBmbG9hdDoKICAgICIiIlBpY2sgYSBkaXNrLXRoaWNrbmVzcyBjb25zdGFudCBmcm9tIHRoZSBtb3JwaG9sb2dpY2FsIHR5cGUgc3RyaW5nLgoKICAgIEZhbGxzIGJhY2sgdG8gdGhlIG1pZC10eXBlIHZhbHVlIHdoZW4gdGhlIHR5cGUgaXMgdW5rbm93biBvciB1bnBhcnNhYmxlLAogICAgd2hpY2ggaXMgdGhlIHNhZmUgY2hvaWNlOiBpdCBzaXRzIGJldHdlZW4gdGhlIHR3byBleHRyZW1lcy4KICAgICIiIgogICAgaWYgbm90IG10eXBlOgogICAgICAgIHJldHVybiBRMF9ERUZBVUxUCiAgICB0ID0gc3RyKG10eXBlKS5zdHJpcCgpLnVwcGVyKCkKICAgIGlmIHQuc3RhcnRzd2l0aCgoIlMwIiwgIlNBIiwgIkUiKSk6CiAgICAgICAgcmV0dXJuIFEwX0JZX1RZUEVbImVhcmx5Il0KICAgIGlmIHQuc3RhcnRzd2l0aCgoIlNEIiwgIlNNIiwgIklNIiwgIkkiKSk6CiAgICAgICAgcmV0dXJuIFEwX0JZX1RZUEVbImxhdGUiXQogICAgcmV0dXJuIFEwX0JZX1RZUEVbIm1pZCJdCgpkZWYgaW5jbGluYXRpb25fZnJvbV9heGlzX3JhdGlvKGJfb3Zlcl9hOiBmbG9hdCwgcTA6IGZsb2F0ID0gUTBfREVGQVVMVCkgLT4gZmxvYXQ6CiAgICAiIiJJbmNsaW5hdGlvbiBpbiBkZWdyZWVzIGZyb20gdGhlIG9ic2VydmVkIG1pbm9yL21ham9yIGF4aXMgcmF0aW8uCgogICAgVXNlcyB0aGUgSHViYmxlIGZvcm11bGEsIHdoaWNoIGFjY291bnRzIGZvciB0aGUgZGlzayBoYXZpbmcgcmVhbCB0aGlja25lc3M6CgogICAgICAgIGNvc14yKGkpID0gKHFeMiAtIHEwXjIpIC8gKDEgLSBxMF4yKQoKICAgIEF0IG1vZGVyYXRlIGluY2xpbmF0aW9ucyB0aGlzIGJhcmVseSBkaWZmZXJzIGZyb20gdGhlIG5haXZlIGFjb3MoYi9hKS4gQXQgaGlnaAogICAgaW5jbGluYXRpb24gaXQgbWF0dGVycyBhIGxvdDogZm9yIHEgPSAwLjIgdGhlIG5haXZlIHZhbHVlIGlzIDc4LjUgZGVnIHdoaWxlIHRoaXMKICAgIGdpdmVzIDgxLjIgZGVnLCB3aGljaCBjaGFuZ2VzIHRoZSBzdHJldGNoIGZhY3RvciBmcm9tIDUuMCB0byA2LjUuCgogICAgUmV0dXJucyA5MC4wIHdoZW4gdGhlIGdhbGF4eSBpcyB0aGlubmVyIHRoYW4gcTAsIGkuZS4gc2VlbiBlZGdlLW9uLgogICAgIiIiCiAgICBxID0gZmxvYXQoYl9vdmVyX2EpCiAgICBpZiBxIDw9IHEwOgogICAgICAgIHJldHVybiA5MC4wCiAgICBjb3MyID0gKHEgKiBxIC0gcTAgKiBxMCkgLyAoMS4wIC0gcTAgKiBxMCkKICAgIGNvczIgPSBtaW4obWF4KGNvczIsIDAuMCksIDEuMCkKICAgIHJldHVybiBtYXRoLmRlZ3JlZXMobWF0aC5hY29zKG1hdGguc3FydChjb3MyKSkpCgpkZWYgc3RyZXRjaF9mYWN0b3IoaW5jX2RlZzogZmxvYXQpIC0+IGZsb2F0OgogICAgIiIiSG93IG11Y2ggdGhlIG1pbm9yLWF4aXMgZGlyZWN0aW9uIGhhcyB0byBiZSBzdHJldGNoZWQ6IDEgLyBjb3MoaSkuIiIiCiAgICByZXR1cm4gMS4wIC8gbWF0aC5jb3MobWF0aC5yYWRpYW5zKGluY19kZWcpKQoKZGVmIGluY2xpbmF0aW9uX2lzX3VzYWJsZShpbmNfZGVnOiBmbG9hdCkgLT4gdHVwbGVbYm9vbCwgc3RyXToKICAgICIiIkRlY2lkZSB3aGV0aGVyIGRlcHJvamVjdGluZyBhdCB0aGlzIGluY2xpbmF0aW9uIGlzIHdvcnRoIGRvaW5nLgoKICAgIFJldHVybnMgKG9rLCByZWFzb24pLiBUaGUgcmVhc29uIGlzIG1lYW50IHRvIGJlIHByaW50ZWQgYW5kIHRvIGVuZCB1cCBpbiB0aGUKICAgIHBsb3QsIHNvIHRoYXQgYSBza2lwcGVkIGNvcnJlY3Rpb24gaXMgdmlzaWJsZSByYXRoZXIgdGhhbiBzaWxlbnQuCiAgICAiIiIKICAgIGlmIGluY19kZWcgPCBJTkNfTUlOX0RFRzoKICAgICAgICByZXR1cm4gRmFsc2UsICgKICAgICAgICAgICAgZiJpbmNsaW5hdGlvbiB7aW5jX2RlZzouMWZ9IGRlZyBpcyBiZWxvdyB7SU5DX01JTl9ERUc6LjBmfSBkZWcgLSB0aGUgZ2FsYXh5IGlzICIKICAgICAgICAgICAgImNsb3NlIHRvIGZhY2Utb24sIHRoZSBjb3JyZWN0aW9uIHdvdWxkIGJlIHNtYWxsZXIgdGhhbiBpdHMgb3duIHVuY2VydGFpbnR5LCAiCiAgICAgICAgICAgICJhbmQgdGhlIHBvc2l0aW9uIGFuZ2xlIGlzIHBvb3JseSBkZWZpbmVkLiBOb3QgY29ycmVjdGVkLiIKICAgICAgICApCiAgICBpZiBpbmNfZGVnID4gSU5DX01BWF9ERUc6CiAgICAgICAgcmV0dXJuIEZhbHNlLCAoCiAgICAgICAgICAgIGYiaW5jbGluYXRpb24ge2luY19kZWc6LjFmfSBkZWcgaXMgYWJvdmUge0lOQ19NQVhfREVHOi4wZn0gZGVnIC0gdGhlIGNvcnJlY3Rpb24gIgogICAgICAgICAgICAid291bGQgYmUgbGFyZ2UgYnV0IHVucmVsaWFibGUsIGFuZCBkdXN0IGJpYXNlcyBpdCBzeXN0ZW1hdGljYWxseS4gTm90IGNvcnJlY3RlZC4iCiAgICAgICAgKQogICAgcmV0dXJuIFRydWUsIGYiaW5jbGluYXRpb24ge2luY19kZWc6LjFmfSBkZWcgaXMgaW5zaWRlIHRoZSB1c2FibGUgd2luZG93LiIKCmRlZiBza3lfcGFfdG9faW1hZ2VfYW5nbGUocGFfc2t5X2RlZzogZmxvYXQsCiAgICAgICAgICAgICAgICAgICAgICAgICAgbm9ydGhfYW5nbGVfZGVnOiBmbG9hdCA9IDkwLjAsCiAgICAgICAgICAgICAgICAgICAgICAgICAgbWlycm9yZWQ6IGJvb2wgPSBGYWxzZSkgLT4gZmxvYXQ6CiAgICAiIiJDb252ZXJ0IGEgY2F0YWxvZ3VlIHBvc2l0aW9uIGFuZ2xlIGludG8gYW4gYW5nbGUgaW4gaW1hZ2UgcGl4ZWwgY29vcmRpbmF0ZXMuCgogICAgQ2F0YWxvZ3VlcyBnaXZlIHRoZSBwb3NpdGlvbiBhbmdsZSBtZWFzdXJlZCBvbiB0aGUgc2t5LCBmcm9tIE5vcnRoIHRvd2FyZHMgRWFzdC4KICAgIE91ciBvYmplY3QgcG9zaXRpb25zIGFyZSBpbiBpbWFnZSBwaXhlbHMsIHdoZXJlIHRoZSBtYWpvciBheGlzIGhhcyBzb21lIGFuZ2xlCiAgICBtZWFzdXJlZCBmcm9tIHRoZSAreCBheGlzLiBUaGVzZSBhcmUgbm90IHRoZSBzYW1lIG51bWJlciBhbmQgY29uZnVzaW5nIHRoZW0KICAgIHJvdGF0ZXMgdGhlIHdob2xlIGRlcHJvamVjdGlvbi4KCiAgICAgICAgbm9ydGhfYW5nbGVfZGVnICBhbmdsZSBvZiBOb3J0aCBpbiB0aGUgaW1hZ2UsIGNvdW50ZXJjbG9ja3dpc2UgZnJvbSAreC4KICAgICAgICAgICAgICAgICAgICAgICAgIDkwIGZvciB0aGUgdXN1YWwgIm5vcnRoIHVwIiBvcmllbnRhdGlvbi4KICAgICAgICBtaXJyb3JlZCAgICAgICAgIFRydWUgaWYgdGhlIGltYWdlIHBhcml0eSBpcyBmbGlwcGVkLCBzbyB0aGF0IEVhc3QgcnVucwogICAgICAgICAgICAgICAgICAgICAgICAgY2xvY2t3aXNlIGZyb20gTm9ydGggaW5zdGVhZCBvZiBjb3VudGVyY2xvY2t3aXNlLgoKICAgIElmIHlvdSBhbHJlYWR5IGtub3cgdGhlIG1ham9yLWF4aXMgYW5nbGUgZGlyZWN0bHkgaW4gaW1hZ2UgY29vcmRpbmF0ZXMsIHNraXAKICAgIHRoaXMgZnVuY3Rpb24gYW5kIHBhc3MgdGhhdCBhbmdsZSBzdHJhaWdodCB0byBkZXByb2plY3QoKS4KICAgICIiIgogICAgc2lnbiA9IC0xLjAgaWYgbWlycm9yZWQgZWxzZSAxLjAKICAgIHJldHVybiAobm9ydGhfYW5nbGVfZGVnICsgc2lnbiAqIHBhX3NreV9kZWcpICUgMTgwLjAKCmRlZiBkZXByb2plY3QoZHgsIGR5LCBtYWpvcl9heGlzX2FuZ2xlX2RlZzogZmxvYXQsIGluY19kZWc6IGZsb2F0KToKICAgICIiIlN0cmV0Y2ggY2VudHJlZCBwb3NpdGlvbnMgZnJvbSB0aGUgcGxhbmUgb2YgdGhlIHNreSBpbnRvIHRoZSBnYWxheHkgcGxhbmUuCgogICAgZHgsIGR5IG11c3QgYWxyZWFkeSBiZSBtZWFzdXJlZCBGUk9NIFRIRSBHQUxBWFkgQ0VOVFJFLiBUaGlzIG1hdHRlcnM6IHRoZQogICAgdHJhbnNmb3JtIGlzIGxpbmVhciwgc28gYXBwbHlpbmcgaXQgdG8gdW5jZW50cmVkIGNvb3JkaW5hdGVzIGxlYXZlcyB0aGUgc2hhcGUKICAgIGNvcnJlY3QgYnV0IG1vdmVzIHRoZSBjZW50cmUsIGFuZCBldmVyeSByYWRpdXMgbWVhc3VyZWQgZnJvbSB0aGUgb2xkIGNlbnRyZQogICAgdGhlbiBjb21lcyBvdXQgd3JvbmcuCgogICAgbWFqb3JfYXhpc19hbmdsZV9kZWcgaXMgbWVhc3VyZWQgaW4gdGhlIHNhbWUgZnJhbWUgYXMgZHgsIGR5IC0gY291bnRlcmNsb2Nrd2lzZQogICAgZnJvbSB0aGUgK3ggYXhpcy4gVXNlIHNreV9wYV90b19pbWFnZV9hbmdsZSgpIGlmIHlvdSBhcmUgc3RhcnRpbmcgZnJvbSBhCiAgICBjYXRhbG9ndWUgcG9zaXRpb24gYW5nbGUuCgogICAgVGhlIHRyYW5zZm9ybSByb3RhdGVzIHRoZSBtYWpvciBheGlzIG9udG8geCwgc3RyZXRjaGVzIHRoZSBwZXJwZW5kaWN1bGFyCiAgICBkaXJlY3Rpb24gYnkgMS9jb3MoaSksIGFuZCByb3RhdGVzIGJhY2suIFRoZSByZXN1bHRpbmcgbWF0cml4IGlzIHN5bW1ldHJpYywKICAgIGFzIGEgcHVyZSBzdHJldGNoIGFsb25nIGFuIGF4aXMgbXVzdCBiZS4KICAgICIiIgogICAgZHggPSBucC5hc2FycmF5KGR4LCBkdHlwZT1mbG9hdCkKICAgIGR5ID0gbnAuYXNhcnJheShkeSwgZHR5cGU9ZmxvYXQpCgogICAgdGggPSBtYXRoLnJhZGlhbnMobWFqb3JfYXhpc19hbmdsZV9kZWcpCiAgICBjLCBzID0gbWF0aC5jb3ModGgpLCBtYXRoLnNpbih0aCkKICAgIGsgPSBzdHJldGNoX2ZhY3RvcihpbmNfZGVnKQoKICAgIG0xMSA9IGMgKiBjICsgayAqIHMgKiBzCiAgICBtMTIgPSAoMS4wIC0gaykgKiBzICogYwogICAgbTIyID0gcyAqIHMgKyBrICogYyAqIGMKCiAgICByZXR1cm4gZHggKiBtMTEgKyBkeSAqIG0xMiwgZHggKiBtMTIgKyBkeSAqIG0yMgoKZGVmIGVmZmVjdGl2ZV9yaW5nX2FyZWFzKGVkZ2VzLAogICAgICAgICAgICAgICAgICAgICAgICAgZnJhbWVfY29ybmVycywKICAgICAgICAgICAgICAgICAgICAgICAgIGNlbnRyZV94eSwKICAgICAgICAgICAgICAgICAgICAgICAgIG1ham9yX2F4aXNfYW5nbGVfZGVnOiBmbG9hdCB8IE5vbmUsCiAgICAgICAgICAgICAgICAgICAgICAgICBpbmNfZGVnOiBmbG9hdCB8IE5vbmUsCiAgICAgICAgICAgICAgICAgICAgICAgICBuX3NhbXBsZXM6IGludCA9IDQwMF8wMDAsCiAgICAgICAgICAgICAgICAgICAgICAgICBybmdfc2VlZDogaW50ID0gMTIzNDUpOgogICAgIiIiQXJlYSBvZiBlYWNoIGFubnVsdXMgdGhhdCBhY3R1YWxseSBmYWxscyBpbnNpZGUgdGhlIGltYWdlZCBmcmFtZS4KCiAgICBXSFkgVEhJUyBFWElTVFMKICAgIEEgcmluZyBkcmF3biBhcm91bmQgdGhlIGdhbGF4eSBjZW50cmUgaXMgb25seSBwYXJ0bHkgY292ZXJlZCBieSB0aGUgaW1hZ2U6IGF0CiAgICBsYXJnZSByYWRpaSB0aGUgY29ybmVycyBvZiB0aGUgZnJhbWUgY3V0IGl0LiBEaXZpZGluZyBjb3VudHMgYnkgdGhlIGZ1bGwKICAgIGdlb21ldHJpYyByaW5nIGFyZWEgdGhlcmVmb3JlIG1ha2VzIHRoZSBvdXRlciBiaW5zIGxvb2sgZW1wdGllciB0aGFuIHRoZXkgYXJlLAogICAgYW5kIHByb2R1Y2VzIGEgcmFkaWFsIGRlY2xpbmUgdGhhdCBpcyBwdXJlIGdlb21ldHJ5LgoKICAgIFRoaXMgaXMgbm90IGh5cG90aGV0aWNhbC4gSW4gTTMzIGEgY2F0YWxvZ3VlIGNvdmVyaW5nIG9ubHkgdGhlIGNlbnRyYWwgMjMneDIzJwogICAgcHJvZHVjZWQgImFuIGFydGlmaWNpYWxseSBsYXJnZSBkZWNyZWFzZSBhdCByYWRpaSBsYXJnZXIgdGhhbiB0aGUgbGFyZ2VzdCBjaXJjbGUKICAgIGluc2NyaWJlZCBpbiB0aGUgYXJlYSIsIGFuZCB0aGUgcHVibGlzaGVkIGV4cG9uZW50aWFsIHNsb3BlIHdhcyB3cm9uZyBieSBhCiAgICBmYWN0b3Igb2YgdGhyZWUgdW50aWwgYSBzaXplIGN1dCBhbmQgYSByYWRpYWwgdHJ1bmNhdGlvbiB3ZXJlIGFwcGxpZWQuCgogICAgTUVUSE9ECiAgICBTY2F0dGVyIHJhbmRvbSBwb2ludHMgdW5pZm9ybWx5IG92ZXIgdGhlIGZyYW1lLCBwdXQgdGhlbSB0aHJvdWdoIGV4YWN0bHkgdGhlCiAgICBzYW1lIGRlcHJvamVjdGlvbiBhcyB0aGUgcmVhbCBvYmplY3RzLCBhbmQgaGlzdG9ncmFtIHRoZWlyIHJhZGlpLiBUaGUgZnJhY3Rpb24KICAgIGxhbmRpbmcgaW4gZWFjaCByaW5nLCB0aW1lcyB0aGUgZnJhbWUgYXJlYSwgaXMgdGhhdCByaW5nJ3MgZWZmZWN0aXZlIGFyZWEuCiAgICBUaGlzIGhhbmRsZXMgYW55IGZyYW1lIHNoYXBlIGFuZCBhbnkgZGVwcm9qZWN0aW9uIHdpdGhvdXQgc3BlY2lhbC1jYXNpbmcuCgogICAgZnJhbWVfY29ybmVycyBpcyAoeG1pbiwgeG1heCwgeW1pbiwgeW1heCkgaW4gdGhlIHNhbWUgcGl4ZWwgdW5pdHMgYXMgdGhlIGRhdGEuCiAgICBQYXNzIG1ham9yX2F4aXNfYW5nbGVfZGVnPU5vbmUgdG8gc2tpcCBkZXByb2plY3Rpb24gKHRoZSB1bmNvcnJlY3RlZCBjYXNlKS4KICAgICIiIgogICAgeG1pbiwgeG1heCwgeW1pbiwgeW1heCA9IGZyYW1lX2Nvcm5lcnMKICAgIHJuZyA9IG5wLnJhbmRvbS5kZWZhdWx0X3JuZyhybmdfc2VlZCkKCiAgICBzeCA9IHJuZy51bmlmb3JtKHhtaW4sIHhtYXgsIG5fc2FtcGxlcykKICAgIHN5ID0gcm5nLnVuaWZvcm0oeW1pbiwgeW1heCwgbl9zYW1wbGVzKQogICAgZnJhbWVfYXJlYSA9ICh4bWF4IC0geG1pbikgKiAoeW1heCAtIHltaW4pCgogICAgc2R4ID0gc3ggLSBjZW50cmVfeHlbMF0KICAgIHNkeSA9IHN5IC0gY2VudHJlX3h5WzFdCiAgICBpZiBtYWpvcl9heGlzX2FuZ2xlX2RlZyBpcyBub3QgTm9uZSBhbmQgaW5jX2RlZyBpcyBub3QgTm9uZToKICAgICAgICBzZHgsIHNkeSA9IGRlcHJvamVjdChzZHgsIHNkeSwgbWFqb3JfYXhpc19hbmdsZV9kZWcsIGluY19kZWcpCgogICAgc3IgPSBucC5oeXBvdChzZHgsIHNkeSkKICAgIGNvdW50cywgXyA9IG5wLmhpc3RvZ3JhbShzciwgYmlucz1lZGdlcykKICAgIHJldHVybiBjb3VudHMgLyBmbG9hdChuX3NhbXBsZXMpICogZnJhbWVfYXJlYQoKZGVmIHJhZGlhbF9kZW5zaXR5KGR4LCBkeSwKICAgICAgICAgICAgICAgICAgIG5fcmluZ3M6IGludCA9IDIwLAogICAgICAgICAgICAgICAgICAgcl9tYXg6IGZsb2F0IHwgTm9uZSA9IE5vbmUsCiAgICAgICAgICAgICAgICAgICBmcmFtZV9jb3JuZXJzPU5vbmUsCiAgICAgICAgICAgICAgICAgICBjZW50cmVfeHk9Tm9uZSwKICAgICAgICAgICAgICAgICAgIG1ham9yX2F4aXNfYW5nbGVfZGVnOiBmbG9hdCB8IE5vbmUgPSBOb25lLAogICAgICAgICAgICAgICAgICAgaW5jX2RlZzogZmxvYXQgfCBOb25lID0gTm9uZSwKICAgICAgICAgICAgICAgICAgIG1pbl9hcmVhX2ZyYWN0aW9uOiBmbG9hdCA9IDAuMTUpOgogICAgIiIiU3VyZmFjZSBkZW5zaXR5IG9mIG9iamVjdHMgaW4gZXF1YWwtd2lkdGggY29uY2VudHJpYyByaW5ncy4KCiAgICBFcXVhbC13aWR0aCByaW5ncyBvdXQgdG8gdGhlIG91dGVybW9zdCBvYmplY3QsIGZvbGxvd2luZyB0aGUgc3RhbmRhcmQgcHJhY3RpY2UKICAgIGluIHRoaXMgbGl0ZXJhdHVyZSAoMjAgcmluZ3MsIGVhY2ggMC4wNSBvZiB0aGUgbWF4aW11bSByYWRpdXMpLgoKICAgIFJpbmdzIHdob3NlIGVmZmVjdGl2ZSBhcmVhIGlzIGEgc21hbGwgZnJhY3Rpb24gb2YgdGhlaXIgZnVsbCBnZW9tZXRyaWMgYXJlYSBhcmUKICAgIG1hcmtlZCBpbmNvbXBsZXRlIHJhdGhlciB0aGFuIHBsb3R0ZWQsIGJlY2F1c2UgdGhlaXIgZGVuc2l0eSBpcyBkb21pbmF0ZWQgYnkgaG93CiAgICB0aGUgZnJhbWUgaGFwcGVucyB0byBjdXQgdGhlbS4gbWluX2FyZWFfZnJhY3Rpb24gc2V0cyB0aGF0IHRocmVzaG9sZC4KCiAgICBSZXR1cm5zIGEgZGljdCB3aXRoIHJpbmcgY2VudHJlcywgY291bnRzLCBhcmVhcywgZGVuc2l0aWVzIGFuZCBQb2lzc29uIGVycm9ycy4KICAgICIiIgogICAgZHggPSBucC5hc2FycmF5KGR4LCBkdHlwZT1mbG9hdCkKICAgIGR5ID0gbnAuYXNhcnJheShkeSwgZHR5cGU9ZmxvYXQpCiAgICByID0gbnAuaHlwb3QoZHgsIGR5KQoKICAgIGlmIHJfbWF4IGlzIE5vbmU6CiAgICAgICAgcl9tYXggPSBmbG9hdChyLm1heCgpKSBpZiByLnNpemUgZWxzZSAxLjAKICAgIGVkZ2VzID0gbnAubGluc3BhY2UoMC4wLCByX21heCwgbl9yaW5ncyArIDEpCiAgICBjZW50cmVzID0gMC41ICogKGVkZ2VzWzotMV0gKyBlZGdlc1sxOl0pCgogICAgY291bnRzLCBfID0gbnAuaGlzdG9ncmFtKHIsIGJpbnM9ZWRnZXMpCiAgICBnZW9tZXRyaWMgPSBtYXRoLnBpICogKGVkZ2VzWzE6XSAqKiAyIC0gZWRnZXNbOi0xXSAqKiAyKQoKICAgIGlmIGZyYW1lX2Nvcm5lcnMgaXMgbm90IE5vbmUgYW5kIGNlbnRyZV94eSBpcyBub3QgTm9uZToKICAgICAgICBhcmVhID0gZWZmZWN0aXZlX3JpbmdfYXJlYXMoZWRnZXMsIGZyYW1lX2Nvcm5lcnMsIGNlbnRyZV94eSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbWFqb3JfYXhpc19hbmdsZV9kZWcsIGluY19kZWcpCiAgICBlbHNlOgogICAgICAgIGFyZWEgPSBnZW9tZXRyaWMKCiAgICB3aXRoIG5wLmVycnN0YXRlKGRpdmlkZT0iaWdub3JlIiwgaW52YWxpZD0iaWdub3JlIik6CiAgICAgICAgZGVuc2l0eSA9IG5wLndoZXJlKGFyZWEgPiAwLCBjb3VudHMgLyBhcmVhLCBucC5uYW4pCiAgICAgICAgZXJyID0gbnAud2hlcmUoYXJlYSA+IDAsIG5wLnNxcnQobnAubWF4aW11bShjb3VudHMsIDApKSAvIGFyZWEsIG5wLm5hbikKICAgICAgICBmcmFjID0gbnAud2hlcmUoZ2VvbWV0cmljID4gMCwgYXJlYSAvIGdlb21ldHJpYywgMC4wKQoKICAgIGNvbXBsZXRlID0gZnJhYyA+PSBtaW5fYXJlYV9mcmFjdGlvbgoKICAgIHJldHVybiB7CiAgICAgICAgImVkZ2VzIjogZWRnZXMsCiAgICAgICAgInIiOiBjZW50cmVzLAogICAgICAgICJjb3VudHMiOiBjb3VudHMsCiAgICAgICAgImFyZWEiOiBhcmVhLAogICAgICAgICJhcmVhX2ZyYWN0aW9uIjogZnJhYywKICAgICAgICAiZGVuc2l0eSI6IGRlbnNpdHksCiAgICAgICAgImRlbnNpdHlfZXJyIjogZXJyLAogICAgICAgICJjb21wbGV0ZSI6IGNvbXBsZXRlLAogICAgICAgICJyX21heCI6IHJfbWF4LAogICAgfQoKZGVmIHNlY3Rvcl9kaXNwZXJzaW9uKGR4LCBkeSwgbWFqb3JfYXhpc19hbmdsZV9kZWcsIGluY19kZWcsIG5fc2VjdG9ycz0yMCk6CiAgICAiIiJIb3cgdW5ldmVuIHRoZSBvYmplY3QgY291bnRzIGFyZSBvbmNlIHRoZSBmaWVsZCBpcyBjdXQgaW50byBlcXVhbCB3ZWRnZXMuCgogICAgVEhFIElERUEKICAgIFNsaWNlIHRoZSBnYWxheHkgaW50byBlcXVhbCBzZWN0b3JzLCBsaWtlIHNsaWNlcyBvZiBhIHBpZS4gSWYgdGhlIGRpc2sgcmVhbGx5IGlzCiAgICByb3VuZCBhbmQgd2UgYXJlIGxvb2tpbmcgc3RyYWlnaHQgZG93biBvbiBpdCwgZXZlcnkgc2xpY2UgaG9sZHMgcm91Z2hseSB0aGUgc2FtZQogICAgbnVtYmVyIG9mIG9iamVjdHMuIFNlZW4gYXQgYW4gYW5nbGUgaXQgZG9lcyBub3QuIFNvIHRoZSB0aWx0IHRoYXQgZXZlbnMgdGhlCiAgICBzbGljZXMgb3V0IGlzIHRoZSB0aWx0IHRoYXQgdW5kb2VzIHRoZSBwcm9qZWN0aW9uLgoKICAgIE9wcG9zaXRlIHNsaWNlcyBhcmUgYWRkZWQgdG9nZXRoZXIgYmVmb3JlIHRoZSBkaXNwZXJzaW9uIGlzIG1lYXN1cmVkLiBBIGdhbGF4eQogICAgdGhhdCBpcyBsb3BzaWRlZCAtIGEgYnJpZ2h0ZXIgYXJtIG9uIG9uZSBzaWRlLCBhIGNvbXBhbmlvbiBwdWxsaW5nIGF0IGl0IC0gd291bGQKICAgIG90aGVyd2lzZSBwdXNoIHRoZSBhbnN3ZXIgYXJvdW5kOyBhZGRpbmcgYSBzbGljZSB0byB0aGUgb25lIGZhY2luZyBpdCBjYW5jZWxzCiAgICB0aGF0IGtpbmQgb2YgYXN5bW1ldHJ5LgoKICAgIFJldHVybnMgc3RkIC8gbWVhbiBvZiB0aGUgZm9sZGVkIGNvdW50cy4gU21hbGxlciBpcyBiZXR0ZXIuCiAgICAiIiIKICAgIGR4cCwgZHlwID0gZGVwcm9qZWN0KGR4LCBkeSwgbWFqb3JfYXhpc19hbmdsZV9kZWcsIGluY19kZWcpCiAgICB0aGV0YSA9IG5wLmFyY3RhbjIoZHlwLCBkeHApCiAgICBpZHggPSBucC5mbG9vcigodGhldGEgJSAoMi4wICogbWF0aC5waSkpIC8gKDIuMCAqIG1hdGgucGkpICogbl9zZWN0b3JzKS5hc3R5cGUoaW50KQogICAgaWR4ID0gbnAuY2xpcChpZHgsIDAsIG5fc2VjdG9ycyAtIDEpCiAgICBjb3VudHMgPSBucC5iaW5jb3VudChpZHgsIG1pbmxlbmd0aD1uX3NlY3RvcnMpLmFzdHlwZShmbG9hdCkKCiAgICBoYWxmID0gbl9zZWN0b3JzIC8vIDIKICAgIGZvbGRlZCA9IGNvdW50c1s6aGFsZl0gKyBjb3VudHNbaGFsZjpdCiAgICBtZWFuID0gZm9sZGVkLm1lYW4oKQogICAgaWYgbWVhbiA8PSAwOgogICAgICAgIHJldHVybiBucC5pbmYKICAgIHJldHVybiBmbG9hdChmb2xkZWQuc3RkKCkgLyBtZWFuKQoKZGVmIGZpdF9zZWN0b3JfbWV0aG9kKGR4LCBkeSwKICAgICAgICAgICAgICAgICAgICAgIG5fc2VjdG9ycz0yMCwKICAgICAgICAgICAgICAgICAgICAgIGFuZ2xlX3N0ZXA9NS4wLAogICAgICAgICAgICAgICAgICAgICAgaW5jX3N0ZXA9NS4wLAogICAgICAgICAgICAgICAgICAgICAgaW5jX3JhbmdlPSgyMC4wLCA4MC4wKSwKICAgICAgICAgICAgICAgICAgICAgIHJfbWF4PU5vbmUpOgogICAgIiIiU2VhcmNoIGZvciB0aGUgZGVwcm9qZWN0aW9uIGFuZ2xlcyB0aGF0IG1ha2UgdGhlIG9iamVjdCBjb3VudHMgbW9zdCBldmVuLgoKICAgIFNjYW5zIGEgZ3JpZCBvZiBwb3NpdGlvbiBhbmdsZSBhbmQgaW5jbGluYXRpb24gYW5kIHJldHVybnMgdGhlIHBhaXIgdGhhdAogICAgbWluaW1pc2VzIHNlY3Rvcl9kaXNwZXJzaW9uKCkuIFRoaXMgaXMgdGhlIHNlY29uZCBvZiB0aGUgdHdvIG1ldGhvZHMgaW50cm9kdWNlZAogICAgYnkgR2FyY2lhLUdvbWV6IGFuZCBBdGhhbmFzc291bGEsIGFuZCBpbiB0aGVpciBjb21wYXJpc29uIGl0IGlzIHRoZSBvbmUgdGhhdAogICAgYWdyZWVzIG1vc3QgY2xvc2VseSB3aXRoIGtpbmVtYXRpYyBkZXRlcm1pbmF0aW9ucyAtIDAuOTYgaW4gcG9zaXRpb24gYW5nbGUgYW5kCiAgICAwLjkwIGluIGNvcyhpbmNsaW5hdGlvbiksIGJldHRlciB0aGFuIHRoZWlyIEZvdXJpZXIgbWV0aG9kIGFuZCBiZXR0ZXIgdGhhbgogICAgcGhvdG9tZXRyeS4KCiAgICBXSEVOIElUIEZBSUxTIC0gcmVhZCB0aGlzIGJlZm9yZSB0cnVzdGluZyB0aGUgYW5zd2VyCiAgICAgICogQmVsb3cgcm91Z2hseSAxNTAtMjAwIG9iamVjdHMgdGhlIHN1cmZhY2UgYmVpbmcgbWluaW1pc2VkIG9mdGVuIGhhcyBzZXZlcmFsCiAgICAgICAgbWluaW1hIHJhdGhlciB0aGFuIG9uZSwgYW5kIHRoZSBkZWVwZXN0IGlzIG5vdCBhbHdheXMgdGhlIHJpZ2h0IG9uZS4gVGhlCiAgICAgICAgcmV0dXJuZWQgZGljdCByZXBvcnRzIHRoZSBydW5uZXJzLXVwIHNvIHlvdSBjYW4gbG9vay4KICAgICAgKiBJZiB0aGUgb2JqZWN0cyB0cmFjZSB0aGUgc3BpcmFsIGFybXMgb3IgYSBiYXIgaW5zdGVhZCBvZiBmaWxsaW5nIHRoZSBkaXNrLAogICAgICAgIHRoZSBtZXRob2QgY2lyY3VsYXJpc2VzIHRoZSBhcm1zIG9yIHRoZSBiYXIgcmF0aGVyIHRoYW4gdGhlIGRpc2suIEluIHRoZQogICAgICAgIHB1Ymxpc2hlZCBub3RlcyB0aGlzIGlzIGV4YWN0bHkgd2hhdCB3ZW50IHdyb25nIGZvciBzZXZlcmFsIGJhcnJlZCBnYWxheGllcy4KICAgICAgKiBBIHJpbmcgb2Ygb2JqZWN0cyBpcyBjaXJjdWxhcmlzZWQgYXMgaWYgaXQgd2VyZSB0aGUgZGlzay4gQW4gb3V0ZXIgcmluZyB3aXRoCiAgICAgICAgYW4gaW50cmluc2ljIGF4aXMgcmF0aW8gb2YgMC44NyBmYWtlcyBhbiBpbmNsaW5hdGlvbiBvZiBhYm91dCAxMiBkZWdyZWVzIGluIGEKICAgICAgICBnYWxheHkgdGhhdCBpcyByZWFsbHkgZmFjZS1vbi4KICAgICAgKiBJdCB3b3JrcyBiZXR0ZXIgdGhlIG1vcmUgaW5jbGluZWQgdGhlIGdhbGF4eSBpcywgYW5kIHBvb3JseSBuZWFyIGZhY2Utb24sCiAgICAgICAgd2hlcmUgdGhlIHBvc2l0aW9uIGFuZ2xlIGlzIGJhcmVseSBkZWZpbmVkIGluIHRoZSBmaXJzdCBwbGFjZS4KCiAgICByX21heCBvcHRpb25hbGx5IHJlc3RyaWN0cyB0aGUgZml0IHRvIG9iamVjdHMgaW5zaWRlIHRoYXQgcmFkaXVzLCB3aGljaCBoZWxwcwogICAgd2hlbiBhIGZldyBmYXItZmx1bmcgb2JqZWN0cyBkb21pbmF0ZSB0aGUgb3V0ZXIgc2VjdG9ycy4KICAgICIiIgogICAgZHggPSBucC5hc2FycmF5KGR4LCBkdHlwZT1mbG9hdCkKICAgIGR5ID0gbnAuYXNhcnJheShkeSwgZHR5cGU9ZmxvYXQpCgogICAgaWYgcl9tYXggaXMgbm90IE5vbmU6CiAgICAgICAga2VlcCA9IG5wLmh5cG90KGR4LCBkeSkgPD0gcl9tYXgKICAgICAgICBkeCwgZHkgPSBkeFtrZWVwXSwgZHlba2VlcF0KCiAgICBhbmdsZXMgPSBucC5hcmFuZ2UoMC4wLCAxODAuMCwgYW5nbGVfc3RlcCkKICAgIGluY3MgPSBucC5hcmFuZ2UoaW5jX3JhbmdlWzBdLCBpbmNfcmFuZ2VbMV0gKyAxZS05LCBpbmNfc3RlcCkKCiAgICBncmlkID0gbnAuZnVsbCgobGVuKGluY3MpLCBsZW4oYW5nbGVzKSksIG5wLmluZikKICAgIGZvciBpLCBpbmMgaW4gZW51bWVyYXRlKGluY3MpOgogICAgICAgIGZvciBqLCBhbmcgaW4gZW51bWVyYXRlKGFuZ2xlcyk6CiAgICAgICAgICAgIGdyaWRbaSwgal0gPSBzZWN0b3JfZGlzcGVyc2lvbihkeCwgZHksIGZsb2F0KGFuZyksIGZsb2F0KGluYyksIG5fc2VjdG9ycykKCiAgICBmbGF0ID0gbnAuYXJnc29ydChncmlkLCBheGlzPU5vbmUpCiAgICBiZXN0X2ksIGJlc3RfaiA9IG5wLnVucmF2ZWxfaW5kZXgoZmxhdFswXSwgZ3JpZC5zaGFwZSkKCiAgICAjIFJlcG9ydCBkaXN0aW5jdCBydW5uZXJzLXVwOiBsb2NhbCBtaW5pbWEgd2VsbCBzZXBhcmF0ZWQgZnJvbSB0aGUgYmVzdCBvbmUsIHNvCiAgICAjIHRoYXQgYSBzaGFsbG93IGdsb2JhbCBtaW5pbXVtIGRvZXMgbm90IGhpZGUgYSBjb21wZXRpbmcgc29sdXRpb24uCiAgICBydW5uZXJzID0gW10KICAgIGZvciBrIGluIGZsYXRbMTpdOgogICAgICAgIGksIGogPSBucC51bnJhdmVsX2luZGV4KGssIGdyaWQuc2hhcGUpCiAgICAgICAgc2VwX2FuZyA9IG1pbihhYnMoYW5nbGVzW2pdIC0gYW5nbGVzW2Jlc3Rfal0pLAogICAgICAgICAgICAgICAgICAgICAgMTgwLjAgLSBhYnMoYW5nbGVzW2pdIC0gYW5nbGVzW2Jlc3Rfal0pKQogICAgICAgIGlmIHNlcF9hbmcgPCAyMC4wIGFuZCBhYnMoaW5jc1tpXSAtIGluY3NbYmVzdF9pXSkgPCAxNS4wOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGlmIGFueShtaW4oYWJzKGFuZ2xlc1tqXSAtIGEpLCAxODAuMCAtIGFicyhhbmdsZXNbal0gLSBhKSkgPCAyMC4wCiAgICAgICAgICAgICAgIGFuZCBhYnMoaW5jc1tpXSAtIGMpIDwgMTUuMCBmb3IgYSwgYywgXyBpbiBydW5uZXJzKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICBydW5uZXJzLmFwcGVuZCgoZmxvYXQoYW5nbGVzW2pdKSwgZmxvYXQoaW5jc1tpXSksIGZsb2F0KGdyaWRbaSwgal0pKSkKICAgICAgICBpZiBsZW4ocnVubmVycykgPj0gMzoKICAgICAgICAgICAgYnJlYWsKCiAgICByZXR1cm4gewogICAgICAgICJhbmdsZSI6IGZsb2F0KGFuZ2xlc1tiZXN0X2pdKSwKICAgICAgICAiaW5jIjogZmxvYXQoaW5jc1tiZXN0X2ldKSwKICAgICAgICAiZGlzcGVyc2lvbiI6IGZsb2F0KGdyaWRbYmVzdF9pLCBiZXN0X2pdKSwKICAgICAgICAibl9vYmplY3RzIjogaW50KGR4LnNpemUpLAogICAgICAgICJuX3NlY3RvcnMiOiBuX3NlY3RvcnMsCiAgICAgICAgImFuZ2xlcyI6IGFuZ2xlcywKICAgICAgICAiaW5jcyI6IGluY3MsCiAgICAgICAgImdyaWQiOiBncmlkLAogICAgICAgICJydW5uZXJzX3VwIjogcnVubmVycywKICAgICAgICAiZmV3X29iamVjdHMiOiBib29sKGR4LnNpemUgPCAyMDApLAogICAgfQoKZGVmIGRpc3RyaWJ1dGlvbl9zaGFwZShkeCwgZHkpOgogICAgIiIiVGhlIGF4aXMgcmF0aW8gYW5kIHBvc2l0aW9uIGFuZ2xlIG9mIGEgc2V0IG9mIHBvc2l0aW9ucy4KCiAgICBGcm9tIHRoZSBzZWNvbmQgbW9tZW50czogdGhlIHNoYXBlIG9mIHRoZSBjbG91ZCBvZiBwb2ludHMsIG5vdCBvZiBhbnkgb25lIG9mCiAgICB0aGVtLiBSZXR1cm5zIChheGlzX3JhdGlvLCBhbmdsZV9kZWcpLCB0aGUgYW5nbGUgbWVhc3VyZWQgZnJvbSAreCB0aGUgc2FtZSB3YXkKICAgIHRoZSBkZXByb2plY3Rpb24gbWVhc3VyZXMgaXQuCiAgICAiIiIKICAgIGR4ID0gbnAuYXNhcnJheShkeCwgZHR5cGU9ZmxvYXQpCiAgICBkeSA9IG5wLmFzYXJyYXkoZHksIGR0eXBlPWZsb2F0KQogICAgb2sgPSBucC5pc2Zpbml0ZShkeCkgJiBucC5pc2Zpbml0ZShkeSkKICAgIGlmIG9rLnN1bSgpIDwgMjA6CiAgICAgICAgcmV0dXJuIGZsb2F0KCJuYW4iKSwgZmxvYXQoIm5hbiIpCiAgICBjb3YgPSBucC5jb3YoZHhbb2tdLCBkeVtva10pCiAgICB2YWx1ZXMsIHZlY3RvcnMgPSBucC5saW5hbGcuZWlnaChjb3YpCiAgICBpZiB2YWx1ZXNbMV0gPD0gMDoKICAgICAgICByZXR1cm4gZmxvYXQoIm5hbiIpLCBmbG9hdCgibmFuIikKICAgIHJhdGlvID0gZmxvYXQobnAuc3FydChtYXgodmFsdWVzWzBdLCAwLjApIC8gdmFsdWVzWzFdKSkKICAgIGFuZ2xlID0gZmxvYXQobnAuZGVncmVlcyhucC5hcmN0YW4yKHZlY3RvcnNbMSwgMV0sIHZlY3RvcnNbMCwgMV0pKSAlIDE4MC4wKQogICAgcmV0dXJuIHJhdGlvLCBhbmdsZQoKZGVmIGNoZWNrX2FnYWluc3RfY2F0YWxvZ3VlKGR4LCBkeSwgZXhwZWN0ZWRfcmF0aW8sIGV4cGVjdGVkX2FuZ2xlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgcmF0aW9fdG9sZXJhbmNlPTAuMTUsIGFuZ2xlX3RvbGVyYW5jZT0yNS4wKToKICAgICIiIkNvbXBhcmUgdGhlIHNoYXBlIG9mIHdoYXQgd2FzIGRldGVjdGVkIHdpdGggd2hhdCB0aGUgY2F0YWxvZ3VlIHByZWRpY3RzLgoKICAgIFdIWSBUSElTIElTIFdPUlRIIERPSU5HCiAgICBBIHRpbHRlZCBkaXNrIHNlZW4gZnJvbSBoZXJlIGlzIGFuIGVsbGlwc2UsIGFuZCB0aGUgY2F0YWxvZ3VlIHNheXMgd2hpY2gKICAgIGVsbGlwc2UuIFNvIHRoZSB0cmFjZXJzIGRldGVjdGVkIGluIGl0IGhhdmUgYSBzaGFwZSB0aGF0IGNhbiBiZSBwcmVkaWN0ZWQKICAgIGJlZm9yZSB0aGV5IGFyZSBtZWFzdXJlZCAtIGFuZCBjb21wYXJpbmcgdGhlIHR3byBjb3N0cyBub3RoaW5nIGFuZCBuZWVkcyBubwogICAgZXh0cmEgZGF0YS4KCiAgICBJdCBpcyBhIHJlYWwgdGVzdCBiZWNhdXNlIGl0IGNhbiBmYWlsLiBPbiBhIHNldCBvZiBNMzMgZnJhbWVzIHRoZSB0b29sCiAgICBkZXRlY3RlZCAyODQ0IHJlZ2lvbnMgYW5kIGRyZXcgYSBzbW9vdGgsIGVudGlyZWx5IHBsYXVzaWJsZSByYWRpYWwgcHJvZmlsZTsKICAgIHRoZSBkZXRlY3Rpb25zIGNhbWUgb3V0IHJvdW5kLCBheGlzIHJhdGlvIDAuOTYsIHdoaWxlIHRoZSBjYXRhbG9ndWUgc2F5cyAwLjU5LgogICAgVGhleSB3ZXJlIG5vdCB0cmFjaW5nIHRoZSBnYWxheHkgYXQgYWxsIC0gdGhleSB3ZXJlIHN1YnRyYWN0aW9uIHJlc2lkdWFscwogICAgc3ByZWFkIGV2ZW5seSBvdmVyIHRoZSBmaWVsZCAtIGFuZCBub3RoaW5nIGVsc2UgaW4gdGhlIG91dHB1dCBzaG93ZWQgaXQuCgogICAgV2hlbiB0aGUgdHdvIGFncmVlIGl0IGlzIGV2aWRlbmNlIHRoZSBkZXRlY3Rpb24gaXMgZmluZGluZyB0aGUgZ2FsYXh5LiBXaGVuCiAgICB0aGV5IGRpc2FncmVlLCBzb21ldGhpbmcgaXMgd3JvbmcgdXBzdHJlYW0gYW5kIHRoZSBwcm9maWxlIGJlbG93IHNob3VsZCBub3QgYmUKICAgIGJlbGlldmVkLiBUaGUgY2hlY2sgZG9lcyBub3Qgc2F5IHdoaWNoIG9mIHRoZSB0d28gaXMgYXQgZmF1bHQuCgogICAgQSBnYWxheHkgdG9vIHJvdW5kIGZvciBpdHMgb3duIHBvc2l0aW9uIGFuZ2xlIHRvIGJlIGRlZmluZWQgaXMgbm90IHRlc3RlZCBvbgogICAgYW5nbGUsIGZvciB0aGUgc2FtZSByZWFzb24gdGhlIGNhdGFsb2d1ZSBkZWNsaW5lcyB0byBwdWJsaXNoIG9uZS4KICAgICIiIgogICAgcmF0aW8sIGFuZ2xlID0gZGlzdHJpYnV0aW9uX3NoYXBlKGR4LCBkeSkKICAgIGlmIG5vdCBucC5pc2Zpbml0ZShyYXRpbyk6CiAgICAgICAgcmV0dXJuIHsib2siOiBOb25lLCAicmVhc29uIjogInRvbyBmZXcgcG9zaXRpb25zIHRvIG1lYXN1cmUgYSBzaGFwZSJ9CgogICAgb3V0ID0geyJtZWFzdXJlZF9yYXRpbyI6IHJhdGlvLCAibWVhc3VyZWRfYW5nbGUiOiBhbmdsZSwKICAgICAgICAgICAiZXhwZWN0ZWRfcmF0aW8iOiBleHBlY3RlZF9yYXRpbywgImV4cGVjdGVkX2FuZ2xlIjogZXhwZWN0ZWRfYW5nbGV9CgogICAgcHJvYmxlbXMgPSBbXQogICAgaWYgZXhwZWN0ZWRfcmF0aW8gaXMgbm90IE5vbmUgYW5kIG5wLmlzZmluaXRlKGV4cGVjdGVkX3JhdGlvKToKICAgICAgICBpZiBhYnMocmF0aW8gLSBleHBlY3RlZF9yYXRpbykgPiByYXRpb190b2xlcmFuY2U6CiAgICAgICAgICAgIHByb2JsZW1zLmFwcGVuZCgKICAgICAgICAgICAgICAgICJ0aGUgZGV0ZWN0aW9ucyBhcmUgc2hhcGVkIHs6LjJmfSB3aGVyZSB0aGUgY2F0YWxvZ3VlIHNheXMgezouMmZ9IgogICAgICAgICAgICAgICAgLmZvcm1hdChyYXRpbywgZXhwZWN0ZWRfcmF0aW8pKQogICAgIyBBIHJvdW5kIGdhbGF4eSBoYXMgbm8gbWVhbmluZ2Z1bCBtYWpvciBheGlzLCBzbyBpdHMgYW5nbGUgaXMgbm90IHRlc3RlZC4KICAgIGlmIChleHBlY3RlZF9hbmdsZSBpcyBub3QgTm9uZSBhbmQgbnAuaXNmaW5pdGUoZXhwZWN0ZWRfYW5nbGUpCiAgICAgICAgICAgIGFuZCBleHBlY3RlZF9yYXRpbyBpcyBub3QgTm9uZSBhbmQgZXhwZWN0ZWRfcmF0aW8gPCAwLjg1KToKICAgICAgICBnYXAgPSBhYnMoKGFuZ2xlIC0gZXhwZWN0ZWRfYW5nbGUgKyA5MC4wKSAlIDE4MC4wIC0gOTAuMCkKICAgICAgICBvdXRbImFuZ2xlX2dhcCJdID0gZ2FwCiAgICAgICAgaWYgZ2FwID4gYW5nbGVfdG9sZXJhbmNlOgogICAgICAgICAgICBwcm9ibGVtcy5hcHBlbmQoCiAgICAgICAgICAgICAgICAidGhlaXIgbG9uZyBheGlzIGxpZXMgYXQgezouMGZ9IGRlZyB3aGVyZSB0aGUgY2F0YWxvZ3VlIHNheXMgezouMGZ9IgogICAgICAgICAgICAgICAgLmZvcm1hdChhbmdsZSwgZXhwZWN0ZWRfYW5nbGUpKQoKICAgIG91dFsib2siXSA9IG5vdCBwcm9ibGVtcwogICAgb3V0WyJwcm9ibGVtcyJdID0gcHJvYmxlbXMKICAgIHJldHVybiBvdXQKCmRlZiBkZXNjcmliZV9jaGVjayhyZXN1bHQpOgogICAgIiIiVGhlIHNlbGYtY2hlY2ssIGluIGEgZm9ybSB3b3J0aCBwcmludGluZy4iIiIKICAgIGlmIHJlc3VsdC5nZXQoIm9rIikgaXMgTm9uZToKICAgICAgICByZXR1cm4gIiAgc2hhcGUgY2hlY2s6ICIgKyByZXN1bHQuZ2V0KCJyZWFzb24iLCAibm90IGRvbmUiKQogICAgaGVhZCA9ICgiICBzaGFwZSBjaGVjazogdGhlIGRldGVjdGlvbnMgYXJlIHNoYXBlZCB7Oi4yZn0gd2l0aCB0aGVpciBsb25nIGF4aXMgIgogICAgICAgICAgICAiYXQgezouMGZ9IGRlZyIuZm9ybWF0KHJlc3VsdFsibWVhc3VyZWRfcmF0aW8iXSwgcmVzdWx0WyJtZWFzdXJlZF9hbmdsZSJdKSkKICAgIGlmIHJlc3VsdC5nZXQoImV4cGVjdGVkX3JhdGlvIikgaXMgbm90IE5vbmU6CiAgICAgICAgaGVhZCArPSAiOyB0aGUgY2F0YWxvZ3VlIHByZWRpY3RzIHs6LjJmfSIuZm9ybWF0KHJlc3VsdFsiZXhwZWN0ZWRfcmF0aW8iXSkKICAgICAgICBpZiByZXN1bHQuZ2V0KCJleHBlY3RlZF9hbmdsZSIpIGlzIG5vdCBOb25lOgogICAgICAgICAgICBoZWFkICs9ICIgYXQgezouMGZ9IGRlZyIuZm9ybWF0KHJlc3VsdFsiZXhwZWN0ZWRfYW5nbGUiXSkKICAgIGlmIHJlc3VsdFsib2siXToKICAgICAgICByZXR1cm4gaGVhZCArICJcbiAgICB0aGV5IGFncmVlIC0gdGhlIGRldGVjdGlvbnMgZG8gdHJhY2UgdGhpcyBnYWxheHkiCiAgICByZXR1cm4gKGhlYWQgKyAiXG4gICAgVEhFWSBESVNBR1JFRTogIiArICI7ICIuam9pbihyZXN1bHRbInByb2JsZW1zIl0pICsKICAgICAgICAgICAgIlxuICAgIFNvbWV0aGluZyB1cHN0cmVhbSBpcyB3cm9uZy4gRGV0ZWN0aW9ucyB0aGF0IGRvIG5vdCBmb2xsb3cgdGhlICIKICAgICAgICAgICAgImdhbGF4eSdzXG4gICAgc2hhcGUgYXJlIHVzdWFsbHkgc3VidHJhY3Rpb24gcmVzaWR1YWxzIHNwcmVhZCBvdmVyIHRoZSAiCiAgICAgICAgICAgICJ3aG9sZSBmaWVsZCwgYW5kXG4gICAgdGhlIHJhZGlhbCBwcm9maWxlIGJlbG93IHdpbGwgbG9vayBzbW9vdGggYW5kICIKICAgICAgICAgICAgIm1lYW4gbm90aGluZy4iKQoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KIyBzaGFyZWQgbW9kdWxlIGdhbGF4eV9jYXRhbG9ndWUucHkKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQoKcGFzcyAgIyBmcm9tIGRlcHJvamVjdGlvbiwgaW5saW5lZCBhYm92ZQoKSFlQRVJMRURBX1ZJWklFUl9JRCA9ICJWSUkvMjM3IgoKZGVmIGh5cGVybGVkYV9nZW9tZXRyeShnYWxheHlfbmFtZTogc3RyLCBzZWFyY2hfcmFkaXVzX2FyY21pbjogZmxvYXQgPSAwLjUpIC0+IGRpY3Q6CiAgICAiIiJGZXRjaCBwb3NpdGlvbiBhbmdsZSwgYXhpcyByYXRpbyBhbmQgaW5jbGluYXRpb24gZm9yIG9uZSBnYWxheHkuCgogICAgUmV0dXJucyBhIGRpY3Qgd2l0aDoKICAgICAgICBuYW1lX3F1ZXJ5LCBwZ2MsIG10eXBlICAtIGlkZW50aWZpY2F0aW9uCiAgICAgICAgbG9nX3IyNSwgYl9vdmVyX2EgICAgICAgLSB0aGUgb2JzZXJ2ZWQgYXhpcyByYXRpbwogICAgICAgIHBhX3NreSAgICAgICAgICAgICAgICAgIC0gcG9zaXRpb24gYW5nbGUgb24gdGhlIHNreSwgb3IgTm9uZSBpZiBub3QgcHVibGlzaGVkCiAgICAgICAgaW5jX25haXZlICAgICAgICAgICAgICAgLSBhY29zKGIvYSksIGlnbm9yaW5nIGRpc2sgdGhpY2tuZXNzCiAgICAgICAgaW5jICAgICAgICAgICAgICAgICAgICAgLSB0aGUgSHViYmxlLWNvcnJlY3RlZCBpbmNsaW5hdGlvbiB3ZSBhY3R1YWxseSB1c2UKICAgICAgICBxMCAgICAgICAgICAgICAgICAgICAgICAtIHRoZSB0aGlja25lc3MgY29uc3RhbnQgY2hvc2VuIGZvciB0aGlzIHR5cGUKICAgICAgICBkMjVfYXJjbWluICAgICAgICAgICAgICAtIGlzb3Bob3RhbCBkaWFtZXRlciwgdXNlZnVsIGZvciBub3JtYWxpc2luZyByYWRpaQogICAgICAgIHNvdXJjZSAgICAgICAgICAgICAgICAgIC0gaHVtYW4tcmVhZGFibGUgcHJvdmVuYW5jZQoKICAgIFJhaXNlcyBMb29rdXBFcnJvciB3aGVuIHRoZSBuYW1lIHJlc29sdmVzIHRvIG5vdGhpbmcsIHNvIHRoZSBjYWxsZXIgY2FuIGZhbGwKICAgIGJhY2sgdG8gbWFudWFsbHkgc3VwcGxpZWQgYW5nbGVzIGluc3RlYWQgb2Ygc2lsZW50bHkgY29udGludWluZy4KICAgICIiIgogICAgZnJvbSBhc3Ryb3F1ZXJ5LnZpemllciBpbXBvcnQgVml6aWVyCiAgICBpbXBvcnQgYXN0cm9weS51bml0cyBhcyB1CgogICAgdiA9IFZpemllcihjYXRhbG9nPUhZUEVSTEVEQV9WSVpJRVJfSUQsIHJvd19saW1pdD0tMSkKICAgIHJlc3VsdCA9IHYucXVlcnlfb2JqZWN0KGdhbGF4eV9uYW1lLCByYWRpdXM9c2VhcmNoX3JhZGl1c19hcmNtaW4gKiB1LmFyY21pbikKICAgIGlmIG5vdCByZXN1bHQgb3IgbGVuKHJlc3VsdFswXSkgPT0gMDoKICAgICAgICByYWlzZSBMb29rdXBFcnJvcigKICAgICAgICAgICAgZiJIeXBlckxFREEgcmV0dXJuZWQgbm8gcm93IGZvciAne2dhbGF4eV9uYW1lfScuIENoZWNrIHRoZSBuYW1lLCBvciAiCiAgICAgICAgICAgICJzdXBwbHkgdGhlIGFuZ2xlcyBtYW51YWxseS4iCiAgICAgICAgKQoKICAgIHJvdyA9IHJlc3VsdFswXVswXQoKICAgIGRlZiBfZ2V0KGNvbCk6CiAgICAgICAgdHJ5OgogICAgICAgICAgICB2YWwgPSByb3dbY29sXQogICAgICAgIGV4Y2VwdCBLZXlFcnJvcjoKICAgICAgICAgICAgcmV0dXJuIE5vbmUKICAgICAgICB0cnk6CiAgICAgICAgICAgIGlmIGhhc2F0dHIodmFsLCAibWFzayIpIGFuZCB2YWwubWFzazoKICAgICAgICAgICAgICAgIHJldHVybiBOb25lCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgcGFzcwogICAgICAgIHRyeToKICAgICAgICAgICAgZiA9IGZsb2F0KHZhbCkKICAgICAgICBleGNlcHQgKFR5cGVFcnJvciwgVmFsdWVFcnJvcik6CiAgICAgICAgICAgIHJldHVybiBOb25lCiAgICAgICAgcmV0dXJuIE5vbmUgaWYgbWF0aC5pc25hbihmKSBlbHNlIGYKCiAgICBsb2dfcjI1ID0gX2dldCgibG9nUjI1IikKICAgIGlmIGxvZ19yMjUgaXMgTm9uZToKICAgICAgICByYWlzZSBMb29rdXBFcnJvcigKICAgICAgICAgICAgZiJIeXBlckxFREEgaGFzIGEgcm93IGZvciAne2dhbGF4eV9uYW1lfScgYnV0IG5vIGF4aXMgcmF0aW8gKGxvZ1IyNSksICIKICAgICAgICAgICAgInNvIHRoZSBpbmNsaW5hdGlvbiBjYW5ub3QgYmUgZGVyaXZlZC4gU3VwcGx5IHRoZSBhbmdsZXMgbWFudWFsbHkuIgogICAgICAgICkKCiAgICBiX292ZXJfYSA9IDEwLjAgKiogKC1sb2dfcjI1KQogICAgbXR5cGUgPSBOb25lCiAgICB0cnk6CiAgICAgICAgbXR5cGUgPSBzdHIocm93WyJNVHlwZSJdKS5zdHJpcCgpIG9yIE5vbmUKICAgIGV4Y2VwdCBLZXlFcnJvcjoKICAgICAgICBwYXNzCgogICAgcTAgPSBxMF9mb3JfdHlwZShtdHlwZSkKICAgIGxvZ19kMjUgPSBfZ2V0KCJsb2dEMjUiKQoKICAgIHJldHVybiB7CiAgICAgICAgIm5hbWVfcXVlcnkiOiBnYWxheHlfbmFtZSwKICAgICAgICAicGdjIjogX2dldCgiUEdDIiksCiAgICAgICAgIm10eXBlIjogbXR5cGUsCiAgICAgICAgImxvZ19yMjUiOiBsb2dfcjI1LAogICAgICAgICJiX292ZXJfYSI6IGJfb3Zlcl9hLAogICAgICAgICJwYV9za3kiOiBfZ2V0KCJQQSIpLAogICAgICAgICJpbmNfbmFpdmUiOiBtYXRoLmRlZ3JlZXMobWF0aC5hY29zKG1pbihtYXgoYl9vdmVyX2EsIDAuMCksIDEuMCkpKSwKICAgICAgICAiaW5jIjogaW5jbGluYXRpb25fZnJvbV9heGlzX3JhdGlvKGJfb3Zlcl9hLCBxMCksCiAgICAgICAgInEwIjogcTAsCiAgICAgICAgImQyNV9hcmNtaW4iOiAoMC4xICogMTAuMCAqKiBsb2dfZDI1KSBpZiBsb2dfZDI1IGlzIG5vdCBOb25lIGVsc2UgTm9uZSwKICAgICAgICAic291cmNlIjogZiJIeXBlckxFREEgdmlhIFZpemllUiB7SFlQRVJMRURBX1ZJWklFUl9JRH0iLAogICAgfQoKZGVmIGRlc2NyaWJlKGdlb206IGRpY3QpIC0+IHN0cjoKICAgICIiIk9uZSByZWFkYWJsZSBibG9jayBzdW1tYXJpc2luZyB3aGF0IHRoZSBjYXRhbG9ndWUgZ2F2ZSB1cy4iIiIKICAgIGxpbmVzID0gWwogICAgICAgIGYiICBjYXRhbG9ndWUgICAgICA6IHtnZW9tWydzb3VyY2UnXX0iLAogICAgICAgIGYiICBQR0MgICAgICAgICAgICA6IHtnZW9tWydwZ2MnXX0iLAogICAgICAgIGYiICB0eXBlICAgICAgICAgICA6IHtnZW9tWydtdHlwZSddfSIsCiAgICAgICAgZiIgIGIvYSAgICAgICAgICAgIDoge2dlb21bJ2Jfb3Zlcl9hJ106LjNmfSAgIChsb2dSMjUgPSB7Z2VvbVsnbG9nX3IyNSddOi4zZn0pIiwKICAgICAgICBmIiAgaW5jbGluYXRpb24gICAgOiB7Z2VvbVsnaW5jJ106LjFmfSBkZWcgICAiCiAgICAgICAgZiIobmFpdmUgYWNvcyhiL2EpID0ge2dlb21bJ2luY19uYWl2ZSddOi4xZn0gZGVnLCBxMCA9IHtnZW9tWydxMCddOi4yZn0pIiwKICAgIF0KICAgIGlmIGdlb21bInBhX3NreSJdIGlzIE5vbmU6CiAgICAgICAgbGluZXMuYXBwZW5kKCIgIHBvc2l0aW9uIGFuZ2xlIDogTk9UIFBVQkxJU0hFRCAtIHRoZSBnYWxheHkgaXMgdG9vIHJvdW5kIGZvciB0aGUgIgogICAgICAgICAgICAgICAgICAgICAibWFqb3IgYXhpcyB0byBiZSBkZWZpbmVkIikKICAgIGVsc2U6CiAgICAgICAgbGluZXMuYXBwZW5kKGYiICBwb3NpdGlvbiBhbmdsZSA6IHtnZW9tWydwYV9za3knXTouMWZ9IGRlZyBvbiB0aGUgc2t5IChOIHRocm91Z2ggRSkiKQogICAgaWYgZ2VvbVsiZDI1X2FyY21pbiJdOgogICAgICAgIGxpbmVzLmFwcGVuZChmIiAgRDI1ICAgICAgICAgICAgOiB7Z2VvbVsnZDI1X2FyY21pbiddOi4xZn0gYXJjbWluIikKICAgIHJldHVybiAiXG4iLmpvaW4obGluZXMpCgpSX1YgPSAzLjEgICAgICAgICAgICAgICAgICAgICAgICAgICMgQV9WIC8gRShCLVYpIGZvciBkdXN0IGluIG91ciBvd24gZ2FsYXh5CgpESVNUQU5DRV9DQVRBTE9HID0gIkovQUovMTUyLzUwIiAgICMgQ29zbWljZmxvd3MtMywgVHVsbHkgZXQgYWwuIDIwMTYKCmRlZiBsb29rdXBfZGlzdGFuY2UobmFtZSk6CiAgICAiIiJSZWRzaGlmdC1pbmRlcGVuZGVudCBkaXN0YW5jZSBpbiBNcGMgZnJvbSBDb3NtaWNmbG93cy0zLCBvciBOb25lLgoKICAgIEV2ZXJ5IGxlbmd0aCB0aGlzIHRvb2wgcmVwb3J0cyBpcyB0aGlzIG51bWJlciB0aW1lcyBhbiBhbmdsZSwgc28gYSB3cm9uZwogICAgZGlzdGFuY2UgcmVzY2FsZXMgdGhlIHdob2xlIGFuc3dlciBhbmQgbm90aGluZyBpbiB0aGUgb3V0cHV0IGxvb2tzIGFtaXNzLgogICAgUmVkc2hpZnQgaXMgbm8gaGVscCBmb3IgYSBnYWxheHkgdGhpcyBjbG9zZSAtIHRoZSByZWNlc3Npb24gdmVsb2NpdHkgb2YgYQogICAgZmV3LU1wYyBnYWxheHkgaXMgZG9taW5hdGVkIGJ5IGl0cyBvd24gbW90aW9uIHRocm91Z2ggdGhlIGxvY2FsIGdyb3VwLCBhbmQKICAgIE0xMDEncyBwdXRzIGl0IGF0IGFib3V0IGhhbGYgaXRzIHRydWUgZGlzdGFuY2UgLSBzbyBhIGNhdGFsb2d1ZSBvZiBkaXJlY3QKICAgIG1lYXN1cmVtZW50cyBpcyB3aGF0IGlzIHdhbnRlZC4KCiAgICBJdCBpcyBvZmZlcmVkLCBuZXZlciBpbXBvc2VkOiB0aGUgY2FsbGVyIGlzIGV4cGVjdGVkIHRvIHNob3cgaXQgYW5kIGxldCB0aGUKICAgIG9ic2VydmVyIG92ZXJydWxlIGl0LgogICAgIiIiCiAgICB0cnk6CiAgICAgICAgZnJvbSBhc3Ryb3B5LmNvb3JkaW5hdGVzIGltcG9ydCBTa3lDb29yZAogICAgICAgIGZyb20gYXN0cm9xdWVyeS52aXppZXIgaW1wb3J0IFZpemllcgogICAgICAgIGltcG9ydCBhc3Ryb3B5LnVuaXRzIGFzIHUKCiAgICAgICAgY29vcmQgPSBTa3lDb29yZC5mcm9tX25hbWUobmFtZSkKICAgICAgICAjICJfciIgaXMgVml6aWVSJ3Mgb3duIGRpc3RhbmNlIG9mIGVhY2ggcm93IGZyb20gdGhlIHBvc2l0aW9uIGFza2VkCiAgICAgICAgIyBhYm91dC4gVGhlIHJvd3MgY29tZSBiYWNrIGluIGNhdGFsb2d1ZSBvcmRlciwgbm90IG5lYXJlc3QgZmlyc3QsIHNvCiAgICAgICAgIyB3aXRob3V0IGl0IHRoZSBmaXJzdCByb3cgY291bGQgYmUgYSBuZWlnaGJvdXJpbmcgZ2FsYXh5IGluc2lkZSB0aGUKICAgICAgICAjIHNlYXJjaCBjaXJjbGUgYW5kIGl0cyBkaXN0YW5jZSB3b3VsZCBiZSB1c2VkIHdpdGhvdXQgYSB3b3JkLgogICAgICAgIGhpdCA9IFZpemllcihjb2x1bW5zPVsiKioiLCAiX3IiXSwgcm93X2xpbWl0PTIwKS5xdWVyeV9yZWdpb24oCiAgICAgICAgICAgIGNvb3JkLCByYWRpdXM9MiAqIHUuYXJjbWluLCBjYXRhbG9nPURJU1RBTkNFX0NBVEFMT0cpCiAgICAgICAgaWYgbm90IGhpdDoKICAgICAgICAgICAgcmV0dXJuIE5vbmUKICAgICAgICB0YWJsZSA9IGhpdFswXQogICAgICAgIGlmICJfciIgaW4gdGFibGUuY29sbmFtZXMgYW5kIGxlbih0YWJsZSkgPiAxOgogICAgICAgICAgICB0YWJsZSA9IHRhYmxlW25wLmFyZ3NvcnQobnAuYXNhcnJheSh0YWJsZVsiX3IiXSwgZHR5cGU9ZmxvYXQpKV0KICAgICAgICBmb3IgY29sIGluICgiRGlzdCIsICI8RGlzdD4iKToKICAgICAgICAgICAgaWYgY29sIGluIHRhYmxlLmNvbG5hbWVzOgogICAgICAgICAgICAgICAgdmFsdWUgPSBmbG9hdCh0YWJsZVtjb2xdWzBdKQogICAgICAgICAgICAgICAgaWYgdmFsdWUgPT0gdmFsdWUgYW5kIHZhbHVlID4gMDoKICAgICAgICAgICAgICAgICAgICByZXR1cm4gdmFsdWUKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIHJldHVybiBOb25lCgpkZWYgbG9va3VwX2V4dGluY3Rpb24obmFtZSk6CiAgICAiIiJHYWxhY3RpYyByZWRkZW5pbmcgRShCLVYpIHRvd2FyZHMgdGhlIGdhbGF4eSwgb3IgTm9uZS4KCiAgICBUaGUgU2NobGFmbHkgJiBGaW5rYmVpbmVyICgyMDExKSByZWNhbGlicmF0aW9uIG9mIHRoZSBTY2hsZWdlbCBkdXN0IG1hcHMsCiAgICBzZXJ2ZWQgYnkgSVJTQS4gTmVlZGVkIHdoZXJldmVyIGEgY29sb3VyIGlzIHR1cm5lZCBpbnRvIGEgc3RhdGVtZW50IGFib3V0IGEKICAgIHN0ZWxsYXIgcG9wdWxhdGlvbjogZHVzdCBpbiBvdXIgb3duIGdhbGF4eSByZWRkZW5zIGV2ZXJ5dGhpbmcgYmVoaW5kIGl0LCBhbmQKICAgIHdpdGhvdXQgdGhlIGNvcnJlY3Rpb24gZXZlcnkgc291cmNlIGlzIHNoaWZ0ZWQgdG93YXJkcyB0aGUgcmVkLCB3aGljaCBtb3ZlcwogICAgdGhlIGxpbmUgYmV0d2VlbiBhIHlvdW5nIGJsdWUgY2x1c3RlciBhbmQgYW4gb3JkaW5hcnkgc3Rhci4KCiAgICBXaGVyZSBpdCBjYW5jZWxzLCBhbmQgdGhpcyBpcyB3b3J0aCBrbm93aW5nIHNvIGl0IGRvZXMgbm90IGdldCAiZml4ZWQiIGxhdGVyOgogICAgZml0dGluZyB0aGUgY29udGludXVtIHN1YnRyYWN0aW9uIGZhY3RvciBhZ2FpbnN0IGFuIGluc3RydW1lbnRhbCBjb2xvdXIKICAgIG1lYXN1cmVkIGZyb20gdGhlIHNhbWUgcGFpciBvZiBmcmFtZXMuIFJlZGRlbmluZyBzaGlmdHMgZXZlcnkgc3RhciBieSB0aGUgc2FtZQogICAgYW1vdW50IHRoZXJlLCB0aGUgZml0dGVkIGxpbmUgc2hpZnRzIHdpdGggdGhlbSwgYW5kIHRoZSB2YWx1ZSByZWFkIG9mZiBpdCBmb3IKICAgIGFueSBnaXZlbiBwaXhlbCBpcyB1bmNoYW5nZWQuCiAgICAiIiIKICAgIHRyeToKICAgICAgICBmcm9tIGFzdHJvcXVlcnkuaXBhYy5pcnNhLmlyc2FfZHVzdCBpbXBvcnQgSXJzYUR1c3QKICAgICAgICB0YWJsZSA9IElyc2FEdXN0LmdldF9xdWVyeV90YWJsZShuYW1lLCBzZWN0aW9uPSJlYnYiKQogICAgICAgIHZhbHVlID0gZmxvYXQodGFibGVbImV4dCBTYW5kRiBtZWFuIl1bMF0pCiAgICAgICAgaWYgdmFsdWUgPT0gdmFsdWUgYW5kIHZhbHVlID49IDA6CiAgICAgICAgICAgIHJldHVybiB2YWx1ZQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICByZXR1cm4gTm9uZQogICAgcmV0dXJuIE5vbmUKCmRlZiBkZXNjcmliZV9sb29rdXBzKG5hbWUsIGRpc3RhbmNlX21wYywgZWJ2LCBkaXN0YW5jZV9zb3VyY2U9IiIsIGVidl9zb3VyY2U9IiIpOgogICAgIiIiV2hhdCB3YXMgbG9va2VkIHVwLCBzaG93biBiZWZvcmUgaXQgaXMgdXNlZC4iIiIKICAgIGxpbmVzID0gW10KICAgIGlmIGRpc3RhbmNlX21wYzoKICAgICAgICBsaW5lcy5hcHBlbmQoIiAgZGlzdGFuY2UgICAgezouMmZ9IE1wYyAgIHt9Ii5mb3JtYXQoCiAgICAgICAgICAgIGRpc3RhbmNlX21wYywgZGlzdGFuY2Vfc291cmNlIG9yICJmcm9tIENvc21pY2Zsb3dzLTMgKFR1bGx5IGV0IGFsLiAyMDE2KSIpKQogICAgICAgIGxpbmVzLmFwcGVuZCgiICAgICAgICAgICAgICBldmVyeSBsZW5ndGggYmVsb3cgc2NhbGVzIHdpdGggdGhpcyBudW1iZXIiKQogICAgZWxzZToKICAgICAgICBsaW5lcy5hcHBlbmQoIiAgZGlzdGFuY2UgICAgbm90IGZvdW5kIC0gcmFkaWkgd2lsbCBiZSBpbiBhcmNzZWMsIG5vdCBwYXJzZWNzIikKICAgIGlmIGVidiBpcyBub3QgTm9uZToKICAgICAgICBsaW5lcy5hcHBlbmQoIiAgcmVkZGVuaW5nICAgRShCLVYpID0gezouNGZ9LCBzbyBBX1YgPSB7Oi4zZn0gICB7fSIuZm9ybWF0KAogICAgICAgICAgICBlYnYsIFJfViAqIGVidiwKICAgICAgICAgICAgZWJ2X3NvdXJjZSBvciAiZnJvbSBTY2hsYWZseSAmIEZpbmtiZWluZXIgKDIwMTEpIikpCiAgICByZXR1cm4gIlxuIi5qb2luKGxpbmVzKQoKZGVmIGdhaWFfZm9yZWdyb3VuZF9zdGFycyh3Y3MsIHNoYXBlLCBnX2xpbWl0PTE5LjAsIHNpZ25pZmljYW5jZT0zLjAsCiAgICAgICAgICAgICAgICAgICAgICAgICAgbWF4X3Jvd3M9NTAwMDApOgogICAgIiIiUG9zaXRpb25zLCBpbiBwaXhlbHMsIG9mIHN0YXJzIHRoYXQgYXJlIGluIGZyb250IG9mIHRoZSBnYWxheHkuCgogICAgV0hZIEEgQ0FUQUxPR1VFIEFORCBOT1QgVEhFIElNQUdFCiAgICBFdmVyeXRoaW5nIGRvd25zdHJlYW0gdGhhdCBuZWVkcyAiYSBzdGFyIiBuZWVkcyBhIGZvcmVncm91bmQgc3Rhcjogc29tZXRoaW5nCiAgICB3aG9zZSBsaWdodCBpcyBwdXJlIGNvbnRpbnV1bSwgc28gdGhhdCBzY2FsaW5nIG9uZSBmaWx0ZXIgYWdhaW5zdCBhbm90aGVyIGFuZAogICAgc3VidHJhY3RpbmcgbWFrZXMgaXQgdmFuaXNoLiBEZXRlY3RpbmcgcG9pbnQgc291cmNlcyBpbiB0aGUgaW1hZ2UgZmluZHMgdGhvc2UsCiAgICBidXQgaW4gYSBuZWFyYnkgZ2FsYXh5IGl0IGFsc28gZmluZHMgdGhlIGdhbGF4eSdzIG93biBzdGFycyAtIGFuZCB0aG9zZSBhcmUgbm90CiAgICBwdXJlIGNvbnRpbnV1bS4gU29tZSBhcmUgcmVkIGdpYW50cywgc29tZSBhcmUgaG90IHlvdW5nIHN0YXJzIHNpdHRpbmcgaW4gdGhlCiAgICBnYXMgdGhhdCBpcyBiZWluZyBtZWFzdXJlZC4KCiAgICBIb3cgYmFkbHkgdGhpcyBtYXR0ZXJzIGRlcGVuZHMgb24gZGlzdGFuY2UsIHdoaWNoIGlzIHdoeSBpdCB3ZW50IHVubm90aWNlZCBmb3IKICAgIGEgbG9uZyB0aW1lLiBBdCBhIGZldyBNcGMgYSBnYWxheHkncyBvd24gc3RhcnMgYXJlIG5vdCByZXNvbHZlZCBhbmQgZXZlcnkgcG9pbnQKICAgIHNvdXJjZSByZWFsbHkgaXMgZm9yZWdyb3VuZC4gQXQgTTMzJ3MgMC45IE1wYyB0aGV5IGFyZSByZXNvbHZlZCBpbiB0aGVpcgogICAgdGhvdXNhbmRzOiBtZWFzdXJlZCBpbiB0aGlzIGZpZWxkLCBvZiA0MjA5IEdhaWEgc291cmNlcyBicmlnaHRlciB0aGFuIEc9MTksCiAgICBvbmx5IDE1OTAgYXJlIGZvcmVncm91bmQuIFNldHRpbmcgdGhlIHN1YnRyYWN0aW9uIGZhY3RvciBmcm9tIHRoZSBtaXh0dXJlIGdhdmUKICAgIGEgc3Rhci10by1zdGFyIHNjYXR0ZXIgb2YgNjYgcGVyY2VudCAtIHdoaWNoIGlzIHRvIHNheSBubyBmYWN0b3IgYXQgYWxsLgoKICAgIFRIRSBURVNUCiAgICBQYXJhbGxheCBhbmQgcHJvcGVyIG1vdGlvbi4gQSBzdGFyIGEgZmV3IGh1bmRyZWQgcGFyc2VjcyBhd2F5IGluIG91ciBvd24KICAgIGdhbGF4eSBoYXMgYm90aCwgbWVhc3VyYWJseS4gQSBzdGFyIGF0IDAuOSBNcGMgaGFzIG5laXRoZXIsIGF0IGFueSBwcmVjaXNpb24KICAgIEdhaWEgY2FuIHJlYWNoLiBTbyBhIHNpZ25pZmljYW50IHBhcmFsbGF4LCBvciBhIHNpZ25pZmljYW50IHByb3BlciBtb3Rpb24sIGlzCiAgICBhIHBoeXNpY2FsIHN0YXRlbWVudCB0aGF0IHRoZSBzdGFyIGlzIGluIGZyb250IC0gbm90IGEgYnJpZ2h0bmVzcyBjdXQgb3IgYQogICAgc2hhcGUgY3V0IHRoYXQgaGFwcGVucyB0byBjb3JyZWxhdGUgd2l0aCBpdC4KCiAgICBSZXR1cm5zIChwb3NpdGlvbnMsIGRpYWdub3N0aWNzKS4gUG9zaXRpb25zIGlzIGFuIChOLCAyKSBhcnJheSBvZiB4LCB5IGluIHRoZQogICAgZnJhbWU7IGFuIGVtcHR5IGFycmF5IG1lYW5zIHRoZSBxdWVyeSBmYWlsZWQgb3IgZm91bmQgbm90aGluZywgYW5kIHRoZSBjYWxsZXIKICAgIGlzIGV4cGVjdGVkIHRvIGZhbGwgYmFjayBvbiBkZXRlY3RpbmcgcG9pbnQgc291cmNlcy4KICAgICIiIgogICAgaW1wb3J0IG51bXB5IGFzIG5wCgogICAgZGlhZyA9IHsic291cmNlIjogIkdhaWEgRFIzIiwgIm5fY2F0YWxvZ3VlIjogMCwgIm5fZm9yZWdyb3VuZCI6IDAsCiAgICAgICAgICAgICJuX2luX2ZyYW1lIjogMCwgIndoeSI6ICIifQogICAgaWYgd2NzIGlzIE5vbmU6CiAgICAgICAgZGlhZ1sid2h5Il0gPSAibm8gcGxhdGUgc29sdXRpb24sIHNvIHRoZSBjYXRhbG9ndWUgY2Fubm90IGJlIHBsYWNlZCIKICAgICAgICByZXR1cm4gbnAuZW1wdHkoKDAsIDIpKSwgZGlhZwoKICAgIHRyeToKICAgICAgICBmcm9tIGFzdHJvcXVlcnkuZ2FpYSBpbXBvcnQgR2FpYQogICAgICAgIGltcG9ydCBhc3Ryb3B5LnVuaXRzIGFzIHUKCiAgICAgICAgbnksIG54ID0gc2hhcGVbMF0sIHNoYXBlWzFdCiAgICAgICAgY2VudHJlID0gd2NzLnBpeGVsX3RvX3dvcmxkKG54IC8gMi4wLCBueSAvIDIuMCkKICAgICAgICBjb3JuZXIgPSB3Y3MucGl4ZWxfdG9fd29ybGQoMC4wLCAwLjApCiAgICAgICAgcmFkaXVzID0gZmxvYXQoY2VudHJlLnNlcGFyYXRpb24oY29ybmVyKS50byh1LmRlZykudmFsdWUpICogMS4wNQoKICAgICAgICBxdWVyeSA9ICgKICAgICAgICAgICAgIlNFTEVDVCByYSwgZGVjLCBwaG90X2dfbWVhbl9tYWcgRlJPTSBnYWlhZHIzLmdhaWFfc291cmNlICIKICAgICAgICAgICAgIldIRVJFIDE9Q09OVEFJTlMoUE9JTlQocmEsIGRlYyksIENJUkNMRSh7Oi42Zn0sIHs6LjZmfSwgezouNGZ9KSkgIgogICAgICAgICAgICAiQU5EIHBob3RfZ19tZWFuX21hZyA8IHs6LjJmfSAiCiAgICAgICAgICAgICJBTkQgKHBhcmFsbGF4L3BhcmFsbGF4X2Vycm9yID4gezouMWZ9IE9SICIKICAgICAgICAgICAgIiAgICAgc3FydChwbXJhKnBtcmEgKyBwbWRlYypwbWRlYykgLyAiCiAgICAgICAgICAgICIgICAgIHNxcnQocG1yYV9lcnJvcipwbXJhX2Vycm9yICsgcG1kZWNfZXJyb3IqcG1kZWNfZXJyb3IpID4gezouMWZ9KSAiCiAgICAgICAgICAgICJPUkRFUiBCWSBwaG90X2dfbWVhbl9tYWciCiAgICAgICAgKS5mb3JtYXQoY2VudHJlLnJhLmRlZywgY2VudHJlLmRlYy5kZWcsIHJhZGl1cywgZ19saW1pdCwKICAgICAgICAgICAgICAgICBzaWduaWZpY2FuY2UsIHNpZ25pZmljYW5jZSkKCiAgICAgICAgam9iID0gR2FpYS5sYXVuY2hfam9iX2FzeW5jKHF1ZXJ5KSBpZiBtYXhfcm93cyA+IDIwMDAgZWxzZSBHYWlhLmxhdW5jaF9qb2IocXVlcnkpCiAgICAgICAgdGFibGUgPSBqb2IuZ2V0X3Jlc3VsdHMoKQogICAgICAgIGRpYWdbIm5fZm9yZWdyb3VuZCJdID0gbGVuKHRhYmxlKQogICAgICAgIGlmIG5vdCBsZW4odGFibGUpOgogICAgICAgICAgICBkaWFnWyJ3aHkiXSA9ICJubyBmb3JlZ3JvdW5kIHN0YXJzIGluIHRoZSBmaWVsZCIKICAgICAgICAgICAgcmV0dXJuIG5wLmVtcHR5KCgwLCAyKSksIGRpYWcKCiAgICAgICAgeCwgeSA9IHdjcy53b3JsZF90b19waXhlbF92YWx1ZXMobnAuYXNhcnJheSh0YWJsZVsicmEiXSwgZHR5cGU9ZmxvYXQpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5wLmFzYXJyYXkodGFibGVbImRlYyJdLCBkdHlwZT1mbG9hdCkpCiAgICAgICAgaW5zaWRlID0gKHggPiA1KSAmICh4IDwgbnggLSA2KSAmICh5ID4gNSkgJiAoeSA8IG55IC0gNikKICAgICAgICBkaWFnWyJuX2luX2ZyYW1lIl0gPSBpbnQoaW5zaWRlLnN1bSgpKQogICAgICAgICMgQnJpZ2h0ZXN0IGZpcnN0LiBXaG9ldmVyIHRha2VzICJ0aGUgZmlyc3QgdHdvIGh1bmRyZWQiIG9mIHRoaXMgbGlzdCBtdXN0CiAgICAgICAgIyBnZXQgdGhlIHR3byBodW5kcmVkIHdvcnRoIG1lYXN1cmluZzogYSBzdGFyIG5lYXIgdGhlIGNhdGFsb2d1ZSBsaW1pdCBpcwogICAgICAgICMgYSBmZXcgY291bnRzIHRocm91Z2ggYSBuYXJyb3cgZmlsdGVyLCBhbmQgYSByYXRpbyBidWlsdCBvbiBpdCBpcyBub2lzZS4KICAgICAgICAjIEhhbmRpbmcgYmFjayBhbiB1bm9yZGVyZWQgbGlzdCBjb3N0IGEgZmFjdG9yIG9mIHR3byBpbiB0aGUgc2NhdHRlciBvZiB0aGUKICAgICAgICAjIHN1YnRyYWN0aW9uIGZhY3RvciBiZWZvcmUgdGhpcyBsaW5lIGV4aXN0ZWQuCiAgICAgICAgZGlhZ1siZ19yYW5nZSJdID0gKGZsb2F0KG5wLm5hbm1pbih0YWJsZVsicGhvdF9nX21lYW5fbWFnIl0pKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgZmxvYXQobnAubmFubWF4KHRhYmxlWyJwaG90X2dfbWVhbl9tYWciXSkpKQogICAgICAgIHJldHVybiBucC5jb2x1bW5fc3RhY2soW3hbaW5zaWRlXSwgeVtpbnNpZGVdXSksIGRpYWcKCiAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGV4YzoKICAgICAgICBkaWFnWyJ3aHkiXSA9ICJ7fToge30iLmZvcm1hdCh0eXBlKGV4YykuX19uYW1lX18sIGV4YykKICAgICAgICByZXR1cm4gbnAuZW1wdHkoKDAsIDIpKSwgZGlhZwoKZGVmIGRlc2NyaWJlX2ZvcmVncm91bmQoZGlhZyk6CiAgICAiIiJXaGF0IHRoZSBjYXRhbG9ndWUgbG9va3VwIGZvdW5kLCBpbiBhIGxpbmUuIiIiCiAgICBpZiBkaWFnLmdldCgibl9pbl9mcmFtZSIpOgogICAgICAgIHJldHVybiAoIiAge30gZm9yZWdyb3VuZCBzdGFycyBmcm9tIHt9IChwYXJhbGxheCBvciBwcm9wZXIgbW90aW9uICIKICAgICAgICAgICAgICAgICJzaWduaWZpY2FudCk7IHRoZSBnYWxheHkncyBvd24gc3RhcnMgYXJlIG5vdCBhbW9uZyB0aGVtIi5mb3JtYXQoCiAgICAgICAgICAgICAgICAgICAgZGlhZ1sibl9pbl9mcmFtZSJdLCBkaWFnWyJzb3VyY2UiXSkpCiAgICByZXR1cm4gKCIgIG5vIGNhdGFsb2d1ZSBzdGFycyBhdmFpbGFibGUgKHt9KSAtIGZhbGxpbmcgYmFjayBvbiBwb2ludCBzb3VyY2VzICIKICAgICAgICAgICAgImZvdW5kIGluIHRoZSBpbWFnZSwgd2hpY2ggaW4gYSBuZWFyYnkgZ2FsYXh5IGluY2x1ZGVzIGl0cyBvd24gIgogICAgICAgICAgICAic3RhcnMiLmZvcm1hdChkaWFnLmdldCgid2h5Iikgb3IgInVua25vd24iKSkKCmRlc2NyaWJlX19nYWxheHlfY2F0YWxvZ3VlID0gZGVzY3JpYmUgICAjIHRoaXMgbW9kdWxlJ3Mgb3duOyB0aGUgcGxhaW4gbmFtZSBnZXRzIHNoYWRvd2VkIGJlbG93CgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIHNoYXJlZCBtb2R1bGUgY29udGludXVtX3N1YnRyYWN0aW9uLnB5CiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KCmRlZiBfYXBlcnR1cmVfZmx1eGVzKGZyYW1lcywgcG9zaXRpb25zLCByPTYuMCwgcl9pbj0xMC4wLCByX291dD0xNi4wKToKICAgICIiIkJhY2tncm91bmQtc3VidHJhY3RlZCBhcGVydHVyZSBmbHV4IGF0IHRoZSBzYW1lIHBvc2l0aW9ucyBpbiBzZXZlcmFsIGZyYW1lcy4iIiIKICAgIGZyb20gcGhvdHV0aWxzLmFwZXJ0dXJlIGltcG9ydCAoQ2lyY3VsYXJBcGVydHVyZSwgQ2lyY3VsYXJBbm51bHVzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBhcGVydHVyZV9waG90b21ldHJ5KQoKICAgIGFwID0gQ2lyY3VsYXJBcGVydHVyZShwb3NpdGlvbnMsIHI9cikKICAgIGFubiA9IENpcmN1bGFyQW5udWx1cyhwb3NpdGlvbnMsIHJfaW49cl9pbiwgcl9vdXQ9cl9vdXQpCiAgICBvdXQgPSBbXQogICAgZm9yIGltZyBpbiBmcmFtZXM6CiAgICAgICAgcGhvdCA9IGFwZXJ0dXJlX3Bob3RvbWV0cnkoaW1nLCBbYXAsIGFubl0pCiAgICAgICAgYmtnX21lYW4gPSBwaG90WyJhcGVydHVyZV9zdW1fMSJdIC8gYW5uLmFyZWEKICAgICAgICBvdXQuYXBwZW5kKG5wLmFzYXJyYXkocGhvdFsiYXBlcnR1cmVfc3VtXzAiXSAtIGJrZ19tZWFuICogYXAuYXJlYSwgZHR5cGU9ZmxvYXQpKQogICAgcmV0dXJuIG91dAoKZGVmIGZpbmRfcmVmZXJlbmNlX3N0YXJzKGZyYW1lLCBtYXNrPU5vbmUsIG5fc3RhcnM9MzAwLCB0aHJlc2hvbGRfc2lnbWE9MjUuMCk6CiAgICAiIiJCcmlnaHQsIGlzb2xhdGVkIHBvaW50IHNvdXJjZXMgdG8gY2FsaWJyYXRlIGFnYWluc3QuIiIiCiAgICBmcm9tIHBob3R1dGlscy5kZXRlY3Rpb24gaW1wb3J0IERBT1N0YXJGaW5kZXIKICAgIGZyb20gYXN0cm9weS5zdGF0cyBpbXBvcnQgc2lnbWFfY2xpcHBlZF9zdGF0cwoKICAgIG1lYW4sIG1lZGlhbiwgc3RkID0gc2lnbWFfY2xpcHBlZF9zdGF0cyhmcmFtZSwgc2lnbWE9My4wLCBtYXNrPW1hc2spCiAgICBmaW5kZXIgPSBEQU9TdGFyRmluZGVyKGZ3aG09NC4wLCB0aHJlc2hvbGQ9dGhyZXNob2xkX3NpZ21hICogc3RkKQogICAgdGJsID0gZmluZGVyKGZyYW1lIC0gbWVkaWFuLCBtYXNrPW1hc2spCiAgICBpZiB0YmwgaXMgTm9uZSBvciBsZW4odGJsKSA9PSAwOgogICAgICAgIHJldHVybiBucC5lbXB0eSgoMCwgMikpCiAgICB0Ymwuc29ydCgiZmx1eCIpCiAgICB0YmwgPSB0YmxbOjotMV1bOm5fc3RhcnNdCiAgICByZXR1cm4gbnAudHJhbnNwb3NlKChucC5hc2FycmF5KHRibFsieGNlbnRyb2lkIl0sIGR0eXBlPWZsb2F0KSwKICAgICAgICAgICAgICAgICAgICAgICAgIG5wLmFzYXJyYXkodGJsWyJ5Y2VudHJvaWQiXSwgZHR5cGU9ZmxvYXQpKSkKCmRlZiBjb25zdGFudF9mYWN0b3IoaGEsIHJlZCwgbWFzaz1Ob25lLCBuX3N0YXJzPTIwMCwgcG9zaXRpb25zPU5vbmUpOgogICAgIiIiT25lIGZhY3RvciBmb3IgdGhlIHdob2xlIGZyYW1lLCB0YWtlbiBmcm9tIHRoZSBzdGFycy4KCiAgICBSZXR1cm5zIHRoZSBmYWN0b3IgdG8gbXVsdGlwbHkgSC1hbHBoYSBieSBiZWZvcmUgc3VidHJhY3RpbmcgUiwgYW5kIHRoZQogICAgc3Rhci10by1zdGFyIHNjYXR0ZXIsIHdoaWNoIGlzIHRoZSBob25lc3QgdW5jZXJ0YWludHkgb24gaXQuCiAgICAiIiIKICAgIHBvcyA9IChucC5hc2FycmF5KHBvc2l0aW9ucywgZHR5cGU9ZmxvYXQpWzpuX3N0YXJzXQogICAgICAgICAgIGlmIHBvc2l0aW9ucyBpcyBub3QgTm9uZSBhbmQgbGVuKHBvc2l0aW9ucykgPj0gMTAKICAgICAgICAgICBlbHNlIGZpbmRfcmVmZXJlbmNlX3N0YXJzKHJlZCwgbWFzaz1tYXNrLCBuX3N0YXJzPW5fc3RhcnMpKQogICAgaWYgbGVuKHBvcykgPCA1OgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigidG9vIGZldyBzdGFycyBmb3VuZCB0byBzZXQgdGhlIGNvbnRpbnV1bSBmYWN0b3IiKQoKICAgIGZfaGEsIGZfciA9IF9hcGVydHVyZV9mbHV4ZXMoW2hhLCByZWRdLCBwb3MpCiAgICBvayA9IG5wLmlzZmluaXRlKGZfaGEpICYgbnAuaXNmaW5pdGUoZl9yKSAmIChmX2hhID4gMCkKICAgIGlmIG9rLnN1bSgpIDwgNToKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoIm5vIHVzYWJsZSBzdGFycyBmb3IgdGhlIGNvbnRpbnV1bSBmYWN0b3IiKQoKICAgIHJhdGlvID0gZl9yW29rXSAvIGZfaGFbb2tdCiAgICBtZWQgPSBmbG9hdChucC5tZWRpYW4ocmF0aW8pKQogICAgc3ByZWFkID0gMS40ODI2ICogbnAubWVkaWFuKG5wLmFicyhyYXRpbyAtIG1lZCkpCiAgICBpZiBzcHJlYWQgPiAwOgogICAgICAgIHJhdGlvID0gcmF0aW9bbnAuYWJzKHJhdGlvIC0gbWVkKSA8IDMuMCAqIHNwcmVhZF0KCiAgICBmYWN0b3IgPSBmbG9hdChucC5tZWRpYW4ocmF0aW8pKQogICAgc2NhdHRlciA9IGZsb2F0KDEuNDgyNiAqIG5wLm1lZGlhbihucC5hYnMocmF0aW8gLSBmYWN0b3IpKSkKICAgIHJldHVybiBmYWN0b3IsIHsKICAgICAgICAibl9zdGFycyI6IGludChyYXRpby5zaXplKSwKICAgICAgICAic2NhdHRlciI6IHNjYXR0ZXIsCiAgICAgICAgInJlbGF0aXZlX3NjYXR0ZXIiOiBzY2F0dGVyIC8gZmFjdG9yIGlmIGZhY3RvciBlbHNlIGZsb2F0KCJuYW4iKSwKICAgICAgICAibWV0aG9kIjogIm9uZSBjb25zdGFudCBmb3IgdGhlIHdob2xlIGZyYW1lIiwKICAgIH0KCmRlZiBjb2xvdXJfbWFwKGIsIHYsIHNtb290aF9weD02LjAsIG1pbl9zaWduYWw9Tm9uZSk6CiAgICAiIiJBbiBpbnN0cnVtZW50YWwgQiBtaW51cyBWIHZhbHVlIGZvciBldmVyeSBwaXhlbCwgc21vb3RoZWQgdG8gYmUgdXNhYmxlLgoKICAgIENvbG91ciBpcyBhIHJhdGlvIG9mIHR3byBmYWludCB0aGluZ3MsIHNvIGl0IGlzIGZhciBub2lzaWVyIHRoYW4gZWl0aGVyIGZyYW1lCiAgICBvbiBpdHMgb3duLiBCb3RoIGZyYW1lcyBhcmUgc21vb3RoZWQgYmVmb3JlIHRoZSByYXRpbyBpcyB0YWtlbiAtIHRoZSBjb2xvdXIgb2YKICAgIGEgc3Rhci1mb3JtaW5nIGNvbXBsZXggdmFyaWVzIG9uIHNjYWxlcyBtdWNoIGxhcmdlciB0aGFuIGEgcGl4ZWwsIHNvIHRoaXMgY29zdHMKICAgIG5vdGhpbmcgcmVhbCBhbmQgbWFrZXMgdGhlIG1hcCB1c2FibGUuCgogICAgUmV0dXJucyB0aGUgY29sb3VyIG1hcCBhbmQgYSBib29sZWFuIGFycmF5IG1hcmtpbmcgd2hlcmUgaXQgY2FuIGJlIHRydXN0ZWQuCiAgICAiIiIKICAgIGZyb20gc2NpcHkubmRpbWFnZSBpbXBvcnQgZ2F1c3NpYW5fZmlsdGVyCiAgICBmcm9tIGFzdHJvcHkuc3RhdHMgaW1wb3J0IHNpZ21hX2NsaXBwZWRfc3RhdHMKCiAgICBicyA9IGdhdXNzaWFuX2ZpbHRlcihucC5uYW5fdG9fbnVtKGIsIG5hbj0wLjApLCBzbW9vdGhfcHgpCiAgICB2cyA9IGdhdXNzaWFuX2ZpbHRlcihucC5uYW5fdG9fbnVtKHYsIG5hbj0wLjApLCBzbW9vdGhfcHgpCgogICAgaWYgbWluX3NpZ25hbCBpcyBOb25lOgogICAgICAgIF8sIF8sIHN0ZF92ID0gc2lnbWFfY2xpcHBlZF9zdGF0cyh2LCBzaWdtYT0zLjApCiAgICAgICAgbWluX3NpZ25hbCA9IDMuMCAqIHN0ZF92IC8gbWF4KHNtb290aF9weCwgMS4wKQoKICAgIHVzYWJsZSA9IChicyA+IG1pbl9zaWduYWwpICYgKHZzID4gbWluX3NpZ25hbCkgJiBucC5pc2Zpbml0ZShicykgJiBucC5pc2Zpbml0ZSh2cykKICAgIHdpdGggbnAuZXJyc3RhdGUoZGl2aWRlPSJpZ25vcmUiLCBpbnZhbGlkPSJpZ25vcmUiKToKICAgICAgICBjb2xvdXIgPSBucC53aGVyZSh1c2FibGUsIC0yLjUgKiBucC5sb2cxMChucC53aGVyZSh1c2FibGUsIGJzIC8gdnMsIDEuMCkpLCBucC5uYW4pCiAgICByZXR1cm4gY29sb3VyLCB1c2FibGUKCmRlZiBjb2xvdXJfZGVwZW5kZW50X2ZhY3RvcihoYSwgcmVkLCBiLCB2LCBtYXNrPU5vbmUsIG5fc3RhcnM9MzAwLCBzbW9vdGhfcHg9Ni4wLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgcG9zaXRpb25zPU5vbmUpOgogICAgIiIiQSBmYWN0b3IgdGhhdCB2YXJpZXMgYWNyb3NzIHRoZSBmcmFtZSB3aXRoIHRoZSBjb2xvdXIgb2Ygd2hhdCBpcyB0aGVyZS4KCiAgICBGb2xsb3dzIHRoZSByZWNvbW1lbmRhdGlvbiBvZiBTcGVjdG9yLCBGaW5rZWxtYW4gYW5kIEJyb3NjaCAoMjAxMik6IG1lYXN1cmUgdGhlCiAgICBmYWN0b3Igc3RhciBieSBzdGFyLCBtZWFzdXJlIGVhY2ggc3RhcidzIGNvbG91ciwgZml0IGEgc3RyYWlnaHQgbGluZSB0aHJvdWdoCiAgICB0aGVtLCBhbmQgdGhlbiByZWFkIHRoZSBmYWN0b3Igb2ZmIHRoYXQgbGluZSBmb3IgZXZlcnkgcGFydCBvZiB0aGUgaW1hZ2UKICAgIGFjY29yZGluZyB0byBpdHMgb3duIGNvbG91ci4KCiAgICBXSEVSRSBJVCBGQUxMUyBCQUNLCiAgICBUaGUgY29sb3VyIG9mIGEgcGl4ZWwgaXMgb25seSBtZWFzdXJhYmxlIHdoZXJlIGJvdGggYnJvYWQgYmFuZHMgaGF2ZSBzaWduYWwuCiAgICBJbiB0aGUgZW1wdHkgc2t5IGJldHdlZW4gdGhlIGFybXMgdGhleSBkbyBub3QsIGFuZCB0aGVyZSB0aGUgY29uc3RhbnQgZmFjdG9yIGlzCiAgICB1c2VkIGluc3RlYWQuIFRoZSBkaWFnbm9zdGljcyByZXBvcnQgd2hhdCBmcmFjdGlvbiBvZiB0aGUgZnJhbWUgZ290IHdoaWNoLCBzbyBhCiAgICBydW4gd2hlcmUgdGhlIGNvbG91ciBtYXAgYmFyZWx5IGFwcGxpZWQgZG9lcyBub3QgbG9vayBsaWtlIG9uZSB3aGVyZSBpdCBkaWQuCgogICAgUmV0dXJucyAoZmFjdG9yX21hcCwgZGlhZ25vc3RpY3MpLiBmYWN0b3JfbWFwIGhhcyB0aGUgc2FtZSBzaGFwZSBhcyB0aGUgZnJhbWVzLgogICAgIiIiCiAgICAjIENhdGFsb2d1ZSBwb3NpdGlvbnMgd2hlbiB0aGV5IGFyZSBhdmFpbGFibGU6IHRoZSByZWxhdGlvbiBiZWluZyBmaXR0ZWQgaXMKICAgICMgdGhlIG9uZSBmb3JlZ3JvdW5kIHN0YXJzIGZvbGxvdywgYW5kIGEgbmVhcmJ5IGdhbGF4eSdzIG93biBzdGFycyBzaXR0aW5nIGluCiAgICAjIHRoZSBzYW1wbGUgYXJlIHdoYXQgbWFrZXMgdGhlIGZpdCBmYWlsIHJhdGhlciB0aGFuIG1lcmVseSBzY2F0dGVyLgogICAgcG9zID0gKG5wLmFzYXJyYXkocG9zaXRpb25zLCBkdHlwZT1mbG9hdClbOm5fc3RhcnNdCiAgICAgICAgICAgaWYgcG9zaXRpb25zIGlzIG5vdCBOb25lIGFuZCBsZW4ocG9zaXRpb25zKSA+PSAxMAogICAgICAgICAgIGVsc2UgZmluZF9yZWZlcmVuY2Vfc3RhcnModiwgbWFzaz1tYXNrLCBuX3N0YXJzPW5fc3RhcnMpKQogICAgaWYgbGVuKHBvcykgPCA4OgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigidG9vIGZldyBzdGFycyB0byBmaXQgYSBjb2xvdXIgcmVsYXRpb24iKQoKICAgIGZfaGEsIGZfciwgZl9iLCBmX3YgPSBfYXBlcnR1cmVfZmx1eGVzKFtoYSwgcmVkLCBiLCB2XSwgcG9zKQogICAgb2sgPSAobnAuaXNmaW5pdGUoZl9oYSkgJiBucC5pc2Zpbml0ZShmX3IpICYgbnAuaXNmaW5pdGUoZl9iKSAmIG5wLmlzZmluaXRlKGZfdikKICAgICAgICAgICYgKGZfaGEgPiAwKSAmIChmX2IgPiAwKSAmIChmX3YgPiAwKSkKICAgIGlmIG9rLnN1bSgpIDwgODoKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoInRvbyBmZXcgc3RhcnMgd2l0aCBnb29kIGZsdXggaW4gYWxsIGZvdXIgZnJhbWVzIikKCiAgICByYXRpbyA9IGZfcltva10gLyBmX2hhW29rXQogICAgY29sb3VyID0gLTIuNSAqIG5wLmxvZzEwKGZfYltva10gLyBmX3Zbb2tdKQoKICAgICMgQ2xpcCBvYnZpb3VzIG91dGxpZXJzIGluIGJvdGggZGlyZWN0aW9ucyBiZWZvcmUgZml0dGluZzogc2F0dXJhdGVkIHN0YXJzLCBhbmQKICAgICMgYW55IHN0YXIgdGhhdCBkb2VzIGhhdmUgbGluZSBlbWlzc2lvbiBhbmQgc28gaXMgbm90IHB1cmUgY29udGludXVtLgogICAgZ29vZCA9IG5wLmlzZmluaXRlKHJhdGlvKSAmIG5wLmlzZmluaXRlKGNvbG91cikKICAgIGZvciBfIGluIHJhbmdlKDMpOgogICAgICAgIGlmIGdvb2Quc3VtKCkgPCA2OgogICAgICAgICAgICBicmVhawogICAgICAgIG1lZF9yID0gbnAubWVkaWFuKHJhdGlvW2dvb2RdKQogICAgICAgIHNwcmVhZCA9IDEuNDgyNiAqIG5wLm1lZGlhbihucC5hYnMocmF0aW9bZ29vZF0gLSBtZWRfcikpCiAgICAgICAgaWYgc3ByZWFkIDw9IDA6CiAgICAgICAgICAgIGJyZWFrCiAgICAgICAgZ29vZCAmPSBucC5hYnMocmF0aW8gLSBtZWRfcikgPCAzLjAgKiBzcHJlYWQKCiAgICBpZiBnb29kLnN1bSgpIDwgNjoKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoIm5vdCBlbm91Z2ggY2xlYW4gc3RhcnMgbGVmdCB0byBmaXQgdGhlIGNvbG91ciByZWxhdGlvbiIpCgogICAgc2xvcGUsIGludGVyY2VwdCA9IG5wLnBvbHlmaXQoY29sb3VyW2dvb2RdLCByYXRpb1tnb29kXSwgMSkKICAgIHByZWRpY3RlZCA9IG5wLnBvbHl2YWwoW3Nsb3BlLCBpbnRlcmNlcHRdLCBjb2xvdXJbZ29vZF0pCiAgICByZXNpZCA9IHJhdGlvW2dvb2RdIC0gcHJlZGljdGVkCiAgICBzc190b3QgPSBmbG9hdChucC5zdW0oKHJhdGlvW2dvb2RdIC0gcmF0aW9bZ29vZF0ubWVhbigpKSAqKiAyKSkKICAgIHIyID0gZmxvYXQoMS4wIC0gbnAuc3VtKHJlc2lkICoqIDIpIC8gc3NfdG90KSBpZiBzc190b3QgPiAwIGVsc2UgZmxvYXQoIm5hbiIpCgogICAgY29uc3QgPSBmbG9hdChucC5tZWRpYW4ocmF0aW9bZ29vZF0pKQogICAgY29uc3Rfc2NhdHRlciA9IGZsb2F0KDEuNDgyNiAqIG5wLm1lZGlhbihucC5hYnMocmF0aW9bZ29vZF0gLSBjb25zdCkpKQogICAgZml0X3NjYXR0ZXIgPSBmbG9hdCgxLjQ4MjYgKiBucC5tZWRpYW4obnAuYWJzKHJlc2lkKSkpCgogICAgY19sbywgY19oaSA9IG5wLnBlcmNlbnRpbGUoY29sb3VyW2dvb2RdLCBbMiwgOThdKQoKICAgICMgT25seSB1c2UgdGhlIGZpdHRlZCByZWxhdGlvbiBpZiBpdCBhY3R1YWxseSBkZXNjcmliZXMgdGhlIHN0YXJzIGJldHRlciB0aGFuIGEKICAgICMgc2luZ2xlIG51bWJlciBkb2VzLiBGaXR0aW5nIGEgc2xvcGUgdGhyb3VnaCBhIGhhbmRmdWwgb2Ygc3RhcnMgc3Bhbm5pbmcgYQogICAgIyBuYXJyb3cgcmFuZ2Ugb2YgY29sb3VyIGNhbiBlYXNpbHkgY29tZSBvdXQgd29yc2UgdGhhbiB0aGUgY29uc3RhbnQgaXQgd2FzCiAgICAjIG1lYW50IHRvIGltcHJvdmUgb24sIGFuZCBhIG1ldGhvZCB0aGF0IGlzIGJldHRlciBpbiBwcmluY2lwbGUgYnV0IG1lYXN1cmFibHkKICAgICMgd29yc2Ugb24gdGhpcyBkYXRhIHNob3VsZCBub3QgYmUgdGhlIG9uZSB0aGF0IHJ1bnMuCiAgICBpbXByb3ZlcyA9IChmaXRfc2NhdHRlciA8IGNvbnN0X3NjYXR0ZXIpIGFuZCAocjIgPj0gMC4zMCkgYW5kIChnb29kLnN1bSgpID49IDEyKQogICAgcmVhc29uID0gIiIKICAgIGlmIG5vdCBpbXByb3ZlczoKICAgICAgICBiaXRzID0gW10KICAgICAgICBpZiBnb29kLnN1bSgpIDwgMTI6CiAgICAgICAgICAgIGJpdHMuYXBwZW5kKCJvbmx5IHt9IGNsZWFuIHN0YXJzIi5mb3JtYXQoaW50KGdvb2Quc3VtKCkpKSkKICAgICAgICBpZiByMiA8IDAuMzA6CiAgICAgICAgICAgIGJpdHMuYXBwZW5kKCJ0aGUgZml0IGV4cGxhaW5zIGxpdHRsZSBvZiB0aGUgc2NhdHRlciAoUjIgPSB7Oi4yZn0pIi5mb3JtYXQocjIpKQogICAgICAgIGlmIGZpdF9zY2F0dGVyID49IGNvbnN0X3NjYXR0ZXI6CiAgICAgICAgICAgIGJpdHMuYXBwZW5kKCJ0aGUgZml0dGVkIGxpbmUgc2NhdHRlcnMgbW9yZSB0aGFuIGEgY29uc3RhbnQgIgogICAgICAgICAgICAgICAgICAgICAgICAiKHs6LjRmfSBhZ2FpbnN0IHs6LjRmfSkiLmZvcm1hdChmaXRfc2NhdHRlciwgY29uc3Rfc2NhdHRlcikpCiAgICAgICAgaWYgY19oaSAtIGNfbG8gPCAwLjU6CiAgICAgICAgICAgIGJpdHMuYXBwZW5kKCJ0aGUgc3RhcnMgc3BhbiBvbmx5IHs6LjJmfSBtYWcgb2YgY29sb3VyIi5mb3JtYXQoY19oaSAtIGNfbG8pKQogICAgICAgIHJlYXNvbiA9ICI7ICIuam9pbihiaXRzKQoKICAgIGNtYXAsIHVzYWJsZSA9IGNvbG91cl9tYXAoYiwgdiwgc21vb3RoX3B4PXNtb290aF9weCkKICAgIGNfY2xpcHBlZCA9IG5wLmNsaXAoY21hcCwgY19sbywgY19oaSkKCiAgICBmYWN0b3JfbWFwID0gbnAuZnVsbChoYS5zaGFwZSwgY29uc3QsIGR0eXBlPWZsb2F0KQogICAgYXBwbGllZCA9IG5wLnplcm9zKGhhLnNoYXBlLCBkdHlwZT1ib29sKQogICAgaWYgaW1wcm92ZXM6CiAgICAgICAgIyBOZXZlciBleHRyYXBvbGF0ZSB0aGUgcmVsYXRpb24gYmV5b25kIHRoZSBjb2xvdXJzIHRoZSBzdGFycyBhY3R1YWxseQogICAgICAgICMgc3Bhbm5lZDsgb3V0c2lkZSB0aGF0IHJhbmdlIHRoZSBzdHJhaWdodCBsaW5lIGlzIGEgZ3Vlc3MsIG5vdCBhIG1lYXN1cmVtZW50LgogICAgICAgIGFwcGxpZWQgPSB1c2FibGUgJiBucC5pc2Zpbml0ZShjX2NsaXBwZWQpCiAgICAgICAgZmFjdG9yX21hcFthcHBsaWVkXSA9IG5wLnBvbHl2YWwoW3Nsb3BlLCBpbnRlcmNlcHRdLCBjX2NsaXBwZWRbYXBwbGllZF0pCgogICAgcmV0dXJuIGZhY3Rvcl9tYXAsIHsKICAgICAgICAibWV0aG9kIjogKCJmYWN0b3IgZml0dGVkIGFnYWluc3QgY29sb3VyLCBwZXIgU3BlY3RvciBldCBhbC4gMjAxMiIKICAgICAgICAgICAgICAgICAgIGlmIGltcHJvdmVzIGVsc2UKICAgICAgICAgICAgICAgICAgICJvbmUgY29uc3RhbnQgLSB0aGUgY29sb3VyIGZpdCB3YXMgdHJpZWQgYW5kIHJlamVjdGVkIiksCiAgICAgICAgInVzZWRfY29sb3VyIjogYm9vbChpbXByb3ZlcyksCiAgICAgICAgInJlamVjdGVkX2JlY2F1c2UiOiByZWFzb24sCiAgICAgICAgIm5fc3RhcnMiOiBpbnQoZ29vZC5zdW0oKSksCiAgICAgICAgInNsb3BlIjogZmxvYXQoc2xvcGUpLAogICAgICAgICJpbnRlcmNlcHQiOiBmbG9hdChpbnRlcmNlcHQpLAogICAgICAgICJmaXRfcjIiOiByMiwKICAgICAgICAiY29sb3VyX3JhbmdlIjogKGZsb2F0KGNfbG8pLCBmbG9hdChjX2hpKSksCiAgICAgICAgImNvbnN0YW50X2ZhY3RvciI6IGNvbnN0LAogICAgICAgICJjb25zdGFudF9zY2F0dGVyIjogY29uc3Rfc2NhdHRlciwKICAgICAgICAicmVsYXRpdmVfc2NhdHRlciI6IGNvbnN0X3NjYXR0ZXIgLyBjb25zdCBpZiBjb25zdCBlbHNlIGZsb2F0KCJuYW4iKSwKICAgICAgICAiZml0X3NjYXR0ZXIiOiBmaXRfc2NhdHRlciwKICAgICAgICAic2NhdHRlcl9yZWR1Y3Rpb24iOiAoY29uc3Rfc2NhdHRlciAvIGZpdF9zY2F0dGVyKSBpZiBmaXRfc2NhdHRlciA+IDAgZWxzZSBmbG9hdCgibmFuIiksCiAgICAgICAgImZyYWN0aW9uX2NvbG91cl9hcHBsaWVkIjogZmxvYXQoYXBwbGllZC5tZWFuKCkpLAogICAgICAgICJmYWN0b3JfbWluIjogZmxvYXQobnAubmFubWluKGZhY3Rvcl9tYXApKSwKICAgICAgICAiZmFjdG9yX21heCI6IGZsb2F0KG5wLm5hbm1heChmYWN0b3JfbWFwKSksCiAgICB9CgpkZWYgZGVzY3JpYmUoZGlhZyk6CiAgICAiIiJBIHJlYWRhYmxlIHN1bW1hcnkgb2Ygd2hpY2hldmVyIG1ldGhvZCB3YXMgdXNlZC4iIiIKICAgIGxpbmVzID0gWyIgIG1ldGhvZDogIiArIGRpYWcuZ2V0KCJtZXRob2QiLCAiPyIpLAogICAgICAgICAgICAgIiAgc3RhcnMgdXNlZDoge30iLmZvcm1hdChkaWFnLmdldCgibl9zdGFycyIpKV0KICAgIGlmIGRpYWcuZ2V0KCJyZWplY3RlZF9iZWNhdXNlIik6CiAgICAgICAgbGluZXMuYXBwZW5kKCIgIHRoZSBjb2xvdXIgZml0IHdhcyByZWplY3RlZCBiZWNhdXNlICIgKyBkaWFnWyJyZWplY3RlZF9iZWNhdXNlIl0pCiAgICAgICAgbGluZXMuYXBwZW5kKCIgIGZhbGxpbmcgYmFjayB0byBhIGNvbnN0YW50IGZhY3RvciBvZiB7Oi4zZn0iLmZvcm1hdCgKICAgICAgICAgICAgZGlhZ1siY29uc3RhbnRfZmFjdG9yIl0pKQogICAgaWYgInNsb3BlIiBpbiBkaWFnOgogICAgICAgIGxpbmVzICs9IFsKICAgICAgICAgICAgIiAgZmFjdG9yID0gezouNGZ9ICsgezouNGZ9IHggKEItViBpbnN0cnVtZW50YWwpLCAgUjIgPSB7Oi4yZn0iLmZvcm1hdCgKICAgICAgICAgICAgICAgIGRpYWdbImludGVyY2VwdCJdLCBkaWFnWyJzbG9wZSJdLCBkaWFnWyJmaXRfcjIiXSksCiAgICAgICAgICAgICIgIGNvbG91cnMgc3Bhbm5lZCBieSB0aGUgc3RhcnM6IHs6LjJmfSB0byB7Oi4yZn0iLmZvcm1hdCgqZGlhZ1siY29sb3VyX3JhbmdlIl0pLAogICAgICAgICAgICAiICBmYWN0b3IgdmFyaWVzIGFjcm9zcyB0aGUgZnJhbWUgZnJvbSB7Oi4zZn0gdG8gezouM2Z9Ii5mb3JtYXQoCiAgICAgICAgICAgICAgICBkaWFnWyJmYWN0b3JfbWluIl0sIGRpYWdbImZhY3Rvcl9tYXgiXSksCiAgICAgICAgICAgICIgIGEgc2luZ2xlIGNvbnN0YW50IHdvdWxkIGhhdmUgYmVlbiB7Oi4zZn0iLmZvcm1hdChkaWFnWyJjb25zdGFudF9mYWN0b3IiXSksCiAgICAgICAgICAgICIgIHNjYXR0ZXIgYWJvdXQgYSBjb25zdGFudCB7Oi40Zn0sIGFib3V0IHRoZSBmaXR0ZWQgbGluZSB7Oi40Zn0gIgogICAgICAgICAgICAiLSB0aGUgbGluZSBpcyB7Oi4wJX0ge30iLmZvcm1hdCgKICAgICAgICAgICAgICAgIGRpYWdbImNvbnN0YW50X3NjYXR0ZXIiXSwgZGlhZ1siZml0X3NjYXR0ZXIiXSwKICAgICAgICAgICAgICAgIGFicyhkaWFnWyJmaXRfc2NhdHRlciJdIC8gZGlhZ1siY29uc3RhbnRfc2NhdHRlciJdIC0gMS4wKQogICAgICAgICAgICAgICAgaWYgZGlhZ1siY29uc3RhbnRfc2NhdHRlciJdIGVsc2UgZmxvYXQoIm5hbiIpLAogICAgICAgICAgICAgICAgInRpZ2h0ZXIiIGlmIGRpYWdbImZpdF9zY2F0dGVyIl0gPCBkaWFnWyJjb25zdGFudF9zY2F0dGVyIl0gZWxzZSAid2lkZXIiKSwKICAgICAgICAgICAgIiAgdGhlIGNvbG91ciByZWxhdGlvbiB3YXMgYXBwbGllZCB0byB7Oi4wJX0gb2YgdGhlIGZyYW1lOyB0aGUgcmVzdCBmZWxsICIKICAgICAgICAgICAgImJhY2sgdG8gdGhlIGNvbnN0YW50Ii5mb3JtYXQoZGlhZ1siZnJhY3Rpb25fY29sb3VyX2FwcGxpZWQiXSksCiAgICAgICAgXQogICAgZWxzZToKICAgICAgICBsaW5lcy5hcHBlbmQoIiAgZmFjdG9yIHs6LjRmfSwgc3Rhci10by1zdGFyIHNjYXR0ZXIgezouMSV9Ii5mb3JtYXQoCiAgICAgICAgICAgIGRpYWcuZ2V0KCJjb25zdGFudF9mYWN0b3IiLCBmbG9hdCgibmFuIikpLAogICAgICAgICAgICBkaWFnLmdldCgicmVsYXRpdmVfc2NhdHRlciIsIGZsb2F0KCJuYW4iKSkpKQogICAgcmV0dXJuICJcbiIuam9pbihsaW5lcykKCmRlc2NyaWJlX19jb250aW51dW1fc3VidHJhY3Rpb24gPSBkZXNjcmliZSAgICMgdGhpcyBtb2R1bGUncyBvd247IHRoZSBwbGFpbiBuYW1lIGdldHMgc2hhZG93ZWQgYmVsb3cKCgojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCiMgc2hhcmVkIG1vZHVsZSByYWRpYWxfcHJvZmlsZXMucHkKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQoKbWF0cGxvdGxpYi51c2UoIkFnZyIpCgpwYXNzICAjIGZyb20gZGVwcm9qZWN0aW9uLCBpbmxpbmVkIGFib3ZlCgpDT0xPVVJTID0gWyIjNDQ0NDQ0IiwgIiNhMzI4M2MiLCAiIzJiNWE4OSIsICIjM2Y3ZDUyIiwgIiM4YTVhMmIiXQoKZGVmIHRyZWF0bWVudF9ub25lKCk6CiAgICByZXR1cm4geyJrZXkiOiAibm9uZSIsICJsYWJlbCI6ICJBcyBtZWFzdXJlZCwgbm8gdGlsdCBjb3JyZWN0aW9uIiwgImFuZ2xlIjogTm9uZSwgImluYyI6IE5vbmUsCiAgICAgICAgICAgICJub3RlIjogIlBvc2l0aW9ucyBleGFjdGx5IGFzIG1lYXN1cmVkIG9uIHRoZSBpbWFnZS4gQ29udHJvbCBwYW5lbC4ifQoKZGVmIHRyZWF0bWVudF9jYXRhbG9ndWUoZ2VvbSwgbm9ydGhfYW5nbGU9OTAuMCwgbWlycm9yZWQ9RmFsc2UpOgogICAgIiIiRGVwcm9qZWN0aW9uIHVzaW5nIHRoZSBwb3NpdGlvbiBhbmdsZSBhbmQgYXhpcyByYXRpbyBmcm9tIEh5cGVyTEVEQS4iIiIKICAgIGlmIGdlb20gaXMgTm9uZToKICAgICAgICByZXR1cm4gTm9uZQogICAgcGFfc2t5LCBpbmMgPSBnZW9tLmdldCgicGFfc2t5IiksIGdlb20uZ2V0KCJpbmMiKQogICAgaWYgcGFfc2t5IGlzIE5vbmU6CiAgICAgICAgcmV0dXJuIHsia2V5IjogImxlZGEiLCAibGFiZWwiOiAiVGlsdCBjb3JyZWN0aW9uIG5vdCBwb3NzaWJsZSIsICJhbmdsZSI6IE5vbmUsICJpbmMiOiBOb25lLAogICAgICAgICAgICAgICAgIm5vdGUiOiAiSHlwZXJMRURBIHB1Ymxpc2hlcyBubyBwb3NpdGlvbiBhbmdsZSBmb3IgdGhpcyBnYWxheHkgLSBpdCBpcyAiCiAgICAgICAgICAgICAgICAgICAgICAgICJ0b28gcm91bmQgZm9yIGEgbWFqb3IgYXhpcyB0byBiZSBkZWZpbmVkLiBOb3QgY29ycmVjdGVkLiJ9CiAgICBvaywgcmVhc29uID0gaW5jbGluYXRpb25faXNfdXNhYmxlKGluYykKICAgIGlmIG5vdCBvazoKICAgICAgICByZXR1cm4geyJrZXkiOiAibGVkYSIsICJsYWJlbCI6ICJUaWx0IGNvcnJlY3Rpb24gbm90IHBvc3NpYmxlIiwgImFuZ2xlIjogTm9uZSwKICAgICAgICAgICAgICAgICJpbmMiOiBOb25lLCAibm90ZSI6IHJlYXNvbn0KICAgIGFuZyA9IHNreV9wYV90b19pbWFnZV9hbmdsZShwYV9za3ksIG5vcnRoX2FuZ2xlLCBtaXJyb3JlZCkKICAgIHJldHVybiB7CiAgICAgICAgImtleSI6ICJsZWRhIiwgImxhYmVsIjogIkNvcnJlY3RlZCBmb3IgYSB0aWx0IG9mIHs6LjBmfSBkZWcgKEh5cGVyTEVEQSkiLmZvcm1hdChpbmMpLAogICAgICAgICJhbmdsZSI6IGFuZywgImluYyI6IGluYywKICAgICAgICAibm90ZSI6ICgiUEEgezouMWZ9IGRlZyBvbiBza3kgLT4gezouMWZ9IGRlZyBpbiBpbWFnZSAobm9ydGggYXQgezouMGZ9IGRlZ3t9KS4gIgogICAgICAgICAgICAgICAgICJTdHJldGNoIHs6LjJmfXggYWxvbmcgdGhlIG1pbm9yIGF4aXMuIikuZm9ybWF0KAogICAgICAgICAgICAgICAgICAgICBwYV9za3ksIGFuZywgbm9ydGhfYW5nbGUsICIsIG1pcnJvcmVkIiBpZiBtaXJyb3JlZCBlbHNlICIiLAogICAgICAgICAgICAgICAgICAgICBzdHJldGNoX2ZhY3RvcihpbmMpKSwKICAgIH0KCmRlZiB0cmVhdG1lbnRfbWFudWFsKGluYywgcGFfaW1hZ2U9Tm9uZSwgcGFfc2t5PU5vbmUsIG5vcnRoX2FuZ2xlPTkwLjAsIG1pcnJvcmVkPUZhbHNlKToKICAgICIiIkRlcHJvamVjdGlvbiB1c2luZyBhbmdsZXMgdGhlIHVzZXIgc3VwcGxpZWQsIGZyb20gdGhlIGxpdGVyYXR1cmUgb3IgYnkgZXllLiIiIgogICAgaWYgaW5jIGlzIE5vbmU6CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIG9rLCByZWFzb24gPSBpbmNsaW5hdGlvbl9pc191c2FibGUoaW5jKQogICAgaWYgcGFfaW1hZ2UgaXMgbm90IE5vbmU6CiAgICAgICAgYW5nLCBzcmMgPSBwYV9pbWFnZSAlIDE4MC4wLCAiZ2l2ZW4gZGlyZWN0bHkgaW4gaW1hZ2UgY29vcmRpbmF0ZXMiCiAgICBlbGlmIHBhX3NreSBpcyBub3QgTm9uZToKICAgICAgICBhbmcgPSBza3lfcGFfdG9faW1hZ2VfYW5nbGUocGFfc2t5LCBub3J0aF9hbmdsZSwgbWlycm9yZWQpCiAgICAgICAgc3JjID0gImZyb20gc2t5IFBBIHs6LjFmfSBkZWciLmZvcm1hdChwYV9za3kpCiAgICBlbHNlOgogICAgICAgIHJldHVybiB7ImtleSI6ICJtYW51YWwiLCAibGFiZWwiOiAiVGlsdCBjb3JyZWN0aW9uIGluY29tcGxldGUiLCAiYW5nbGUiOiBOb25lLAogICAgICAgICAgICAgICAgImluYyI6IE5vbmUsICJub3RlIjogIkFuIGluY2xpbmF0aW9uIHdhcyBnaXZlbiBidXQgbm8gcG9zaXRpb24gYW5nbGUuIn0KICAgIGlmIG5vdCBvazoKICAgICAgICByZXR1cm4geyJrZXkiOiAibWFudWFsIiwgImxhYmVsIjogIlRpbHQgY29ycmVjdGlvbiBub3QgcG9zc2libGUiLCAiYW5nbGUiOiBOb25lLAogICAgICAgICAgICAgICAgImluYyI6IE5vbmUsICJub3RlIjogcmVhc29ufQogICAgcmV0dXJuIHsia2V5IjogIm1hbnVhbCIsICJsYWJlbCI6ICJDb3JyZWN0ZWQgZm9yIGEgdGlsdCBvZiB7Oi4wZn0gZGVnICh5b3VycykiLmZvcm1hdChpbmMpLAogICAgICAgICAgICAiYW5nbGUiOiBhbmcsICJpbmMiOiBpbmMsCiAgICAgICAgICAgICJub3RlIjogKCJVc2VyLXN1cHBsaWVkIGdlb21ldHJ5LCB7fS4gTWFqb3IgYXhpcyBhdCB7Oi4xZn0gZGVnIGluIHRoZSAiCiAgICAgICAgICAgICAgICAgICAgICJpbWFnZS4gU3RyZXRjaCB7Oi4yZn14LiIpLmZvcm1hdChzcmMsIGFuZywgc3RyZXRjaF9mYWN0b3IoaW5jKSl9CgpkZWYgdHJlYXRtZW50X3NlY3RvcihkeCwgZHksIG5fc2VjdG9ycz0yMCwgc3RlcD01LjApOgogICAgIiIiRGVwcm9qZWN0aW9uIGRlcml2ZWQgZnJvbSB0aGUgb2JqZWN0cyB0aGVtc2VsdmVzLCB3aXRoIGl0cyBjYXZlYXRzIGF0dGFjaGVkLiIiIgogICAgZml0ID0gZml0X3NlY3Rvcl9tZXRob2QoZHgsIGR5LCBuX3NlY3RvcnM9bl9zZWN0b3JzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgYW5nbGVfc3RlcD1zdGVwLCBpbmNfc3RlcD1zdGVwKQogICAgb2ssIHJlYXNvbiA9IGluY2xpbmF0aW9uX2lzX3VzYWJsZShmaXRbImluYyJdKQoKICAgIHdhcm4gPSAiIgogICAgaWYgZml0WyJmZXdfb2JqZWN0cyJdOgogICAgICAgIHdhcm4gKz0gKCIgT25seSB7fSBvYmplY3RzIC0gYmVsb3cgdGhlIDE1MCB0byAyMDAgbmVlZGVkIGZvciBhIHNpbmdsZSBjbGVhbiAiCiAgICAgICAgICAgICAgICAgIm1pbmltdW0sIHNvIHRyZWF0IHRoaXMgd2l0aCBzdXNwaWNpb24uIi5mb3JtYXQoZml0WyJuX29iamVjdHMiXSkpCiAgICBpZiBmaXRbInJ1bm5lcnNfdXAiXToKICAgICAgICBhLCBjLCBkID0gZml0WyJydW5uZXJzX3VwIl1bMF0KICAgICAgICBtYXJnaW4gPSAoZCAtIGZpdFsiZGlzcGVyc2lvbiJdKSAvIG1heChmaXRbImRpc3BlcnNpb24iXSwgMWUtOSkKICAgICAgICB3YXJuICs9ICgiIENsb3Nlc3Qgcml2YWwgc29sdXRpb246IFBBIHs6LjBmfSBkZWcsIGkgezouMGZ9IGRlZywgd29yc2UgYnkgb25seSAiCiAgICAgICAgICAgICAgICAgIns6LjAlfS4iLmZvcm1hdChhLCBjLCBtYXJnaW4pKQoKICAgIGlmIG5vdCBvazoKICAgICAgICByZXR1cm4geyJrZXkiOiAic2VjdG9yIiwgImxhYmVsIjogIlRpbHQgZnJvbSB0aGUgZGF0YSAtIG5vdCBwb3NzaWJsZSIsICJhbmdsZSI6IE5vbmUsCiAgICAgICAgICAgICAgICAiaW5jIjogTm9uZSwKICAgICAgICAgICAgICAgICJub3RlIjogKCJTZWN0b3IgbWV0aG9kIGZvdW5kIFBBIHs6LjBmfSBkZWcsIGkgezouMGZ9IGRlZywgYnV0ICIKICAgICAgICAgICAgICAgICAgICAgICAgIC5mb3JtYXQoZml0WyJhbmdsZSJdLCBmaXRbImluYyJdKSArIHJlYXNvbiArIHdhcm4pfQogICAgcmV0dXJuIHsia2V5IjogInNlY3RvciIsICJsYWJlbCI6ICJDb3JyZWN0ZWQgZm9yIGEgdGlsdCBvZiB7Oi4wZn0gZGVnIChtZWFzdXJlZCBoZXJlKSIuZm9ybWF0KGZpdFsiaW5jIl0pLAogICAgICAgICAgICAiYW5nbGUiOiBmaXRbImFuZ2xlIl0sICJpbmMiOiBmaXRbImluYyJdLAogICAgICAgICAgICAibm90ZSI6ICgiQW5nbGVzIGRlcml2ZWQgZnJvbSB0aGUgb2JqZWN0cyB0aGVtc2VsdmVzIGJ5IGV2ZW5pbmcgb3V0IHRoZSAiCiAgICAgICAgICAgICAgICAgICAgICJjb3VudHMgaW4ge30gd2VkZ2VzOiBQQSB7Oi4wZn0gZGVnIGluIGltYWdlLCBpIHs6LjBmfSBkZWcsICIKICAgICAgICAgICAgICAgICAgICAgInN0cmV0Y2ggezouMmZ9eC4iKS5mb3JtYXQobl9zZWN0b3JzLCBmaXRbImFuZ2xlIl0sIGZpdFsiaW5jIl0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHN0cmV0Y2hfZmFjdG9yKGZpdFsiaW5jIl0pKSArIHdhcm59CgpkZWYgY29tcHV0ZV9wcm9maWxlcyh4LCB5LCBjZW50cmUsIGZyYW1lLCB0cmVhdG1lbnRzLCByaW5ncz0yMCk6CiAgICAiIiJEZXByb2plY3QgYW5kIGJpbiB0aGUgb2JqZWN0cyBvbmNlIHBlciB0cmVhdG1lbnQuIiIiCiAgICBkeDAsIGR5MCA9IG5wLmFzYXJyYXkoeCwgZmxvYXQpIC0gY2VudHJlWzBdLCBucC5hc2FycmF5KHksIGZsb2F0KSAtIGNlbnRyZVsxXQogICAgcmVzdWx0cyA9IFtdCiAgICBmb3IgdCBpbiB0cmVhdG1lbnRzOgogICAgICAgIGlmIHRbImFuZ2xlIl0gaXMgTm9uZToKICAgICAgICAgICAgZHhwLCBkeXAgPSBkeDAsIGR5MAogICAgICAgIGVsc2U6CiAgICAgICAgICAgIGR4cCwgZHlwID0gZGVwcm9qZWN0KGR4MCwgZHkwLCB0WyJhbmdsZSJdLCB0WyJpbmMiXSkKICAgICAgICByZXMgPSByYWRpYWxfZGVuc2l0eShkeHAsIGR5cCwgbl9yaW5ncz1yaW5ncywgZnJhbWVfY29ybmVycz1mcmFtZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjZW50cmVfeHk9Y2VudHJlLCBtYWpvcl9heGlzX2FuZ2xlX2RlZz10WyJhbmdsZSJdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIGluY19kZWc9dFsiaW5jIl0pCiAgICAgICAgcmVzWyJkeCJdLCByZXNbImR5Il0gPSBkeHAsIGR5cAogICAgICAgIHJlc3VsdHMuYXBwZW5kKHJlcykKICAgIHJldHVybiByZXN1bHRzCgpkZWYgZ2VvbWV0cmljX2FyZWFzKGVkZ2VzKToKICAgICIiIlRoZSBmdWxsIG1hdGhlbWF0aWNhbCByaW5nIGFyZWFzLCBmb3IgY29tcGFyaXNvbiB3aXRoIHRoZSBvYnNlcnZlZCBvbmVzLiIiIgogICAgcmV0dXJuIG1hdGgucGkgKiAobnAuYXNhcnJheShlZGdlcylbMTpdICoqIDIgLSBucC5hc2FycmF5KGVkZ2VzKVs6LTFdICoqIDIpCgpkZWYgd3JpdGVfcHJvZmlsZXNfY3N2KHBhdGgsIHRyZWF0bWVudHMsIHJlc3VsdHMsIHNjYWxlPTEuMCwgdW5pdD0icHgiKToKICAgICIiIlNhdmUgZXZlcnkgcHJvZmlsZSwgaW5jbHVkaW5nIGJvdGggdGhlIG9ic2VydmVkIGFuZCB0aGUgZnVsbCByaW5nIGFyZWFzLgoKICAgIEtlZXBpbmcgdGhlIGZ1bGwgZ2VvbWV0cmljIGFyZWEgYWxvbmdzaWRlIHRoZSBvYnNlcnZlZCBvbmUgbGV0cyBhbnlvbmUgY29tcGFyZQogICAgYWdhaW5zdCBvbGRlciBhbmFseXNlcywgd2hpY2ggZGl2aWRlZCBieSB0aGUgZnVsbCBhcmVhIGFuZCBzbyB1bmRlci1yZXBvcnRlZCB0aGUKICAgIGRlbnNpdHkgb2YgYW55IHJpbmcgdGhlIGZyYW1lIGhhZCBjbGlwcGVkLgogICAgIiIiCiAgICB3aXRoIG9wZW4ocGF0aCwgInciLCBlbmNvZGluZz0idXRmLTgiKSBhcyBmaDoKICAgICAgICBmaC53cml0ZSgidHJlYXRtZW50LG1ham9yX2F4aXNfYW5nbGVfZGVnLGluY2xpbmF0aW9uX2RlZywiCiAgICAgICAgICAgICAgICAgInJfaW5uZXJfe3V9LHJfb3V0ZXJfe3V9LHJfY2VudHJlX3t1fSxuX29iamVjdHMsIgogICAgICAgICAgICAgICAgICJvYnNlcnZlZF9hcmVhX3t1fTIsZnVsbF9yaW5nX2FyZWFfe3V9MixhcmVhX2ZyYWN0aW9uLCIKICAgICAgICAgICAgICAgICAiZGVuc2l0eV9wZXJfe3V9MixkZW5zaXR5X2VycixyaW5nX2NvbXBsZXRlXG4iLmZvcm1hdCh1PXVuaXQpKQogICAgICAgIGZvciB0LCByZXMgaW4gemlwKHRyZWF0bWVudHMsIHJlc3VsdHMpOgogICAgICAgICAgICBnZW8gPSBnZW9tZXRyaWNfYXJlYXMocmVzWyJlZGdlcyJdKQogICAgICAgICAgICBmb3IgayBpbiByYW5nZShsZW4ocmVzWyJyIl0pKToKICAgICAgICAgICAgICAgIGZoLndyaXRlKCJ7fSx7fSx7fSx7Oi40Zn0sezouNGZ9LHs6LjRmfSx7fSx7Oi40Zn0sezouNGZ9LHs6LjRmfSwiCiAgICAgICAgICAgICAgICAgICAgICAgICAiezouNmV9LHs6LjZlfSx7fVxuIi5mb3JtYXQoCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdFsibGFiZWwiXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAiIiBpZiB0WyJhbmdsZSJdIGlzIE5vbmUgZWxzZSAiezouMWZ9Ii5mb3JtYXQodFsiYW5nbGUiXSksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIiIgaWYgdFsiaW5jIl0gaXMgTm9uZSBlbHNlICJ7Oi4xZn0iLmZvcm1hdCh0WyJpbmMiXSksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVzWyJlZGdlcyJdW2tdICogc2NhbGUsIHJlc1siZWRnZXMiXVtrICsgMV0gKiBzY2FsZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXNbInIiXVtrXSAqIHNjYWxlLCBpbnQocmVzWyJjb3VudHMiXVtrXSksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVzWyJhcmVhIl1ba10gKiBzY2FsZSAqKiAyLCBnZW9ba10gKiBzY2FsZSAqKiAyLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJlc1siYXJlYV9mcmFjdGlvbiJdW2tdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJlc1siZGVuc2l0eSJdW2tdIC8gc2NhbGUgKiogMiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXNbImRlbnNpdHlfZXJyIl1ba10gLyBzY2FsZSAqKiAyLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIGludChyZXNbImNvbXBsZXRlIl1ba10pKSkKCmRlZiBzdGVwX3Byb2ZpbGUoYXgsIGVkZ2VzLCB2YWx1ZXMsIGNvbG91ciwgbHc9MS43LCBhbHBoYT0xLjAsIGxzPSItIiwgbGFiZWw9Tm9uZSk6CiAgICAiIiJEcmF3IGEgZGVuc2l0eSBwcm9maWxlIGFzIHN0ZXBzLCBmbGF0IGFjcm9zcyBlYWNoIHJpbmcuCgogICAgQSBzdXJmYWNlIGRlbnNpdHkgaXMgbm90IGEgbWVhc3VyZW1lbnQgYXQgYSBwb2ludC4gSXQgaXMgb25lIG51bWJlciBmb3IgYSB3aG9sZQogICAgcmluZyAtIHRoZSBjb3VudCBpbiBpdCBkaXZpZGVkIGJ5IGl0cyBhcmVhIC0gYW5kIGl0IHNheXMgbm90aGluZyBhYm91dCB3aGVyZQogICAgaW5zaWRlIHRoYXQgcmluZyB0aGUgb2JqZWN0cyBzYXQuIERyYXdpbmcgaXQgYXMgYSBkb3QgYXQgdGhlIHJpbmcncyBjZW50cmUKICAgIGpvaW5lZCB0byB0aGUgbmV4dCBkb3QgYnkgYSBzbG9waW5nIGxpbmUgaW52aXRlcyBleGFjdGx5IHRoZSByZWFkaW5nIHRoZSBkYXRhCiAgICBkb2VzIG5vdCBzdXBwb3J0OiB0aGF0IHRoZSBkZW5zaXR5IHZhcmllcyBzbW9vdGhseSBhbmQgdGhhdCB0aGUgdmFsdWUgYXQgdGhlCiAgICBjZW50cmUgb2YgdGhlIHJpbmcgaXMgc29tZWhvdyBiZXR0ZXIgbWVhc3VyZWQgdGhhbiBhdCBpdHMgZWRnZS4KCiAgICBTdGVwcyBzYXkgd2hhdCB3YXMgYWN0dWFsbHkgbWVhc3VyZWQsIGFuZCBpdCBpcyBob3cgdGhlIHRvb2wgaW4gdGhlIHJlcG9zaXRvcnkKICAgIGhhcyBhbHdheXMgZHJhd24gdGhpcywgc28gdGhlIHR3byBjYW4gYmUgbGFpZCBzaWRlIGJ5IHNpZGUuCiAgICAiIiIKICAgIHkgPSBucC5hcHBlbmQodmFsdWVzLCB2YWx1ZXNbLTFdKQogICAgYXguc3RlcChlZGdlcywgeSwgd2hlcmU9InBvc3QiLCBjb2xvcj1jb2xvdXIsIGx3PWx3LCBhbHBoYT1hbHBoYSwgbHM9bHMsCiAgICAgICAgICAgIGxhYmVsPWxhYmVsKQoKZGVmIHByb2ZpbGVfZmlndXJlKHRyZWF0bWVudHMsIHJlc3VsdHMsIHVuaXRfbGFiZWwsIHRpdGxlLCBzY2FsZT0xLjApOgogICAgIiIiT25lIHJvdyBwZXIgdHJlYXRtZW50IC0gc2NhdHRlciBhbmQgcHJvZmlsZSAtIHRoZW4gYWxsIHByb2ZpbGVzIG92ZXJsYWlkLiIiIgogICAgbiA9IGxlbih0cmVhdG1lbnRzKQogICAgZmlnID0gcGx0LmZpZ3VyZShmaWdzaXplPSgxMS4wLCA0LjEgKiBuICsgMy40KSkKICAgIGdzID0gZmlnLmFkZF9ncmlkc3BlYyhuICsgMSwgMiwgaGVpZ2h0X3JhdGlvcz1bMV0gKiBuICsgWzEuMjVdLAogICAgICAgICAgICAgICAgICAgICAgICAgIGhzcGFjZT0wLjg1LCB3c3BhY2U9MC4zMCkKCiAgICBmb3IgaSwgKHQsIHJlcykgaW4gZW51bWVyYXRlKHppcCh0cmVhdG1lbnRzLCByZXN1bHRzKSk6CiAgICAgICAgY29sID0gQ09MT1VSU1tpICUgbGVuKENPTE9VUlMpXQogICAgICAgIGR4dSwgZHl1ID0gcmVzWyJkeCJdICogc2NhbGUsIHJlc1siZHkiXSAqIHNjYWxlCiAgICAgICAgcnUgPSByZXNbInIiXSAqIHNjYWxlCiAgICAgICAgZHUgPSByZXNbImRlbnNpdHkiXSAvIHNjYWxlICoqIDIKICAgICAgICBldSA9IHJlc1siZGVuc2l0eV9lcnIiXSAvIHNjYWxlICoqIDIKCiAgICAgICAgYXggPSBmaWcuYWRkX3N1YnBsb3QoZ3NbaSwgMF0pCiAgICAgICAgYXguc2NhdHRlcihkeHUsIGR5dSwgcz03LCBjPWNvbCwgYWxwaGE9MC42NSwgbGluZXdpZHRocz0wKQogICAgICAgIGF4LnBsb3QoMCwgMCwgIisiLCBjPSJrIiwgbXM9MTEsIG1ldz0xLjYpCiAgICAgICAgbGltID0gMS4wNiAqIG1heChucC5hYnMoZHh1KS5tYXgoKSwgbnAuYWJzKGR5dSkubWF4KCkpCiAgICAgICAgYXguc2V0X3hsaW0oLWxpbSwgbGltKTsgYXguc2V0X3lsaW0oLWxpbSwgbGltKQogICAgICAgIGF4LnNldF9hc3BlY3QoImVxdWFsIiwgYWRqdXN0YWJsZT0iYm94IikKICAgICAgICBheC5zZXRfdGl0bGUodFsibGFiZWwiXSwgZm9udHNpemU9MTAuNSwgbG9jPSJsZWZ0IikKICAgICAgICBheC5zZXRfeGxhYmVsKCJ4IGZyb20gY2VudHJlIFsiICsgdW5pdF9sYWJlbCArICJdIiwgZm9udHNpemU9OC41KQogICAgICAgIGF4LnNldF95bGFiZWwoInkgZnJvbSBjZW50cmUgWyIgKyB1bml0X2xhYmVsICsgIl0iLCBmb250c2l6ZT04LjUpCiAgICAgICAgYXgudGlja19wYXJhbXMobGFiZWxzaXplPTcuNSkKICAgICAgICBheC5ncmlkKGFscGhhPTAuMTUsIGx3PTAuNikKCiAgICAgICAgYXgyID0gZmlnLmFkZF9zdWJwbG90KGdzW2ksIDFdKQogICAgICAgIGcgPSByZXNbImNvbXBsZXRlIl0KICAgICAgICBlZGdlcyA9IHJlc1siZWRnZXMiXSAqIHNjYWxlCiAgICAgICAgIyBUaGUgc3RlcCBpcyB0aGUgbWVhc3VyZW1lbnQuIFRoZSBiYXJzIG9uIHRvcCBvZiBpdCBhcmUgdGhlIGNvdW50aW5nCiAgICAgICAgIyB1bmNlcnRhaW50eSwgZHJhd24gYXQgdGhlIG1pZGRsZSBvZiBlYWNoIHJpbmcgb25seSBiZWNhdXNlIHRoZXkgaGF2ZSB0bwogICAgICAgICMgYmUgZHJhd24gc29tZXdoZXJlLgogICAgICAgIHN0ZXBfcHJvZmlsZShheDIsIGVkZ2VzLCBucC53aGVyZShnLCBkdSwgbnAubmFuKSwgY29sKQogICAgICAgIGlmIGcuYW55KCk6CiAgICAgICAgICAgIGF4Mi5lcnJvcmJhcihydVtnXSwgZHVbZ10sIHllcnI9ZXVbZ10sIGZtdD0ibm9uZSIsIGVjb2xvcj1jb2wsCiAgICAgICAgICAgICAgICAgICAgICAgICBlbGluZXdpZHRoPTEuMCwgY2Fwc2l6ZT0yLCBhbHBoYT0wLjc1KQogICAgICAgIGlmICh+ZykuYW55KCk6CiAgICAgICAgICAgIHN0ZXBfcHJvZmlsZShheDIsIGVkZ2VzLCBucC53aGVyZSh+ZywgZHUsIG5wLm5hbiksIGNvbCwKICAgICAgICAgICAgICAgICAgICAgICAgIGx3PTEuMiwgYWxwaGE9MC40NSwgbHM9Ii0tIiwKICAgICAgICAgICAgICAgICAgICAgICAgIGxhYmVsPSJyaW5nIGNsaXBwZWQgYnkgdGhlIGZyYW1lIikKICAgICAgICAgICAgYXgyLmxlZ2VuZChmb250c2l6ZT03LCBmcmFtZW9uPUZhbHNlLCBsb2M9Imxvd2VyIGxlZnQiKQogICAgICAgIGF4Mi5zZXRfeXNjYWxlKCJsb2ciKQogICAgICAgIGF4Mi5zZXRfeGxhYmVsKCJkZXByb2plY3RlZCByYWRpdXMgWyIgKyB1bml0X2xhYmVsICsgIl0iLCBmb250c2l6ZT04LjUpCiAgICAgICAgYXgyLnNldF95bGFiZWwoInN1cmZhY2UgZGVuc2l0eSBbbiBwZXIgIiArIHVuaXRfbGFiZWwgKyAiXjJdIiwgZm9udHNpemU9OC41KQogICAgICAgIGF4Mi50aWNrX3BhcmFtcyhsYWJlbHNpemU9Ny41KQogICAgICAgIGF4Mi5ncmlkKGFscGhhPTAuMTUsIGx3PTAuNikKICAgICAgICBheDIuc2V0X3RpdGxlKCJOID0gIiArIHN0cihpbnQocmVzWyJjb3VudHMiXS5zdW0oKSkpLCBmb250c2l6ZT05LCBsb2M9InJpZ2h0IikKCiAgICAgICAgYXgudGV4dCgwLjAsIC0wLjMwLCAiXG4iLmpvaW4odGV4dHdyYXAud3JhcCh0WyJub3RlIl0sIHdpZHRoPTEyNSkpLAogICAgICAgICAgICAgICAgdHJhbnNmb3JtPWF4LnRyYW5zQXhlcywgZm9udHNpemU9Ny42LCBjb2xvcj0iIzRhNGE0YSIsCiAgICAgICAgICAgICAgICB2YT0idG9wIiwgaGE9ImxlZnQiLCBsaW5lc3BhY2luZz0xLjUpCgogICAgYXhjID0gZmlnLmFkZF9zdWJwbG90KGdzW24sIDpdKQogICAgZm9yIGksICh0LCByZXMpIGluIGVudW1lcmF0ZSh6aXAodHJlYXRtZW50cywgcmVzdWx0cykpOgogICAgICAgIGcgPSByZXNbImNvbXBsZXRlIl0KICAgICAgICBpZiBnLmFueSgpOgogICAgICAgICAgICBzdGVwX3Byb2ZpbGUoYXhjLCByZXNbImVkZ2VzIl0gKiBzY2FsZSwKICAgICAgICAgICAgICAgICAgICAgICAgIG5wLndoZXJlKGcsIHJlc1siZGVuc2l0eSJdIC8gc2NhbGUgKiogMiwgbnAubmFuKSwKICAgICAgICAgICAgICAgICAgICAgICAgIENPTE9VUlNbaSAlIGxlbihDT0xPVVJTKV0sIGxhYmVsPXRbImxhYmVsIl0pCiAgICBheGMuc2V0X3lzY2FsZSgibG9nIikKICAgIGF4Yy5zZXRfeGxhYmVsKCJkZXByb2plY3RlZCByYWRpdXMgWyIgKyB1bml0X2xhYmVsICsgIl0iKQogICAgYXhjLnNldF95bGFiZWwoInN1cmZhY2UgZGVuc2l0eSBbbiBwZXIgIiArIHVuaXRfbGFiZWwgKyAiXjJdIikKICAgIGF4Yy5ncmlkKGFscGhhPTAuMTgsIGx3PTAuNikKICAgIGF4Yy5sZWdlbmQoZm9udHNpemU9OC41LCBmcmFtZW9uPUZhbHNlKQogICAgYXhjLnNldF90aXRsZSgiQWxsIHRyZWF0bWVudHMgb3ZlcmxhaWQgLSB0aGUgc3ByZWFkIGJldHdlZW4gdGhlbSBpcyB0aGUgIgogICAgICAgICAgICAgICAgICAidW5jZXJ0YWludHkgb24gdGhlIHJhZGlpIiwgZm9udHNpemU9MTAsIGxvYz0ibGVmdCIpCgogICAgZmlnLnN1cHRpdGxlKHRpdGxlLCBmb250c2l6ZT0xMywgeT0wLjk5NSkKICAgIHJldHVybiBmaWcKCmRlZiBfcHJvZmlsZV9yb3dzKHJlc3VsdCwgc2NhbGUsIHVuaXQpOgogICAgIiIiVGhlIHByb2ZpbGUgZXhhY3RseSBhcyB0aGUgcGljdHVyZSBkcmF3cyBpdCwgcmluZyBieSByaW5nLiIiIgogICAgZWRnZXMgPSByZXN1bHRbImVkZ2VzIl0gKiBzY2FsZQogICAgaGVhZCA9IFsicmluZ19pbm5lcl8iICsgdW5pdCwgInJpbmdfb3V0ZXJfIiArIHVuaXQsICJyX2NlbnRyZV8iICsgdW5pdCwKICAgICAgICAgICAgIm5fb2JqZWN0cyIsICJkZW5zaXR5X3Blcl8iICsgdW5pdCArICIyIiwgImRlbnNpdHlfZXJyb3IiLAogICAgICAgICAgICAicmluZ19jb21wbGV0ZSJdCiAgICByb3dzID0gW10KICAgIGZvciBrIGluIHJhbmdlKGxlbihyZXN1bHRbInIiXSkpOgogICAgICAgIHJvd3MuYXBwZW5kKFtmbG9hdChlZGdlc1trXSksIGZsb2F0KGVkZ2VzW2sgKyAxXSksCiAgICAgICAgICAgICAgICAgICAgIGZsb2F0KHJlc3VsdFsiciJdW2tdICogc2NhbGUpLAogICAgICAgICAgICAgICAgICAgICBpbnQocmVzdWx0WyJjb3VudHMiXVtrXSksCiAgICAgICAgICAgICAgICAgICAgIGZsb2F0KHJlc3VsdFsiZGVuc2l0eSJdW2tdIC8gc2NhbGUgKiogMiksCiAgICAgICAgICAgICAgICAgICAgIGZsb2F0KHJlc3VsdFsiZGVuc2l0eV9lcnIiXVtrXSAvIHNjYWxlICoqIDIpLAogICAgICAgICAgICAgICAgICAgICBpbnQoYm9vbChyZXN1bHRbImNvbXBsZXRlIl1ba10pKV0pCiAgICByZXR1cm4gaGVhZCwgcm93cwoKZGVmIF9wb3NpdGlvbl9yb3dzKHJlc3VsdCwgc2NhbGUsIHVuaXQpOgogICAgIiIiRXZlcnkgb2JqZWN0LCB3aGVyZSB0aGUgcGljdHVyZSBwdXRzIGl0OiBkZXByb2plY3RlZCwgY2VudHJlZCwgaW4gYHVuaXRgLiIiIgogICAgaGVhZCA9IFsieF9mcm9tX2NlbnRyZV8iICsgdW5pdCwgInlfZnJvbV9jZW50cmVfIiArIHVuaXQsCiAgICAgICAgICAgICJyYWRpdXNfIiArIHVuaXRdCiAgICBkeCA9IG5wLmFzYXJyYXkocmVzdWx0WyJkeCJdLCBkdHlwZT1mbG9hdCkgKiBzY2FsZQogICAgZHkgPSBucC5hc2FycmF5KHJlc3VsdFsiZHkiXSwgZHR5cGU9ZmxvYXQpICogc2NhbGUKICAgIHJvd3MgPSBbW2Zsb2F0KGEpLCBmbG9hdChiKSwgZmxvYXQobnAuaHlwb3QoYSwgYikpXSBmb3IgYSwgYiBpbiB6aXAoZHgsIGR5KV0KICAgIHJldHVybiBoZWFkLCByb3dzCgpkZWYgX3dyaXRlX3RhYmxlcyhwYXRoLCBzaGVldHMpOgogICAgIiIiV3JpdGUgZWFjaCB0YWJsZSBhcyBpdHMgb3duIENTViwgbmFtZWQgYWZ0ZXIgdGhlIHBpY3R1cmUgaXQgYmVsb25ncyB0by4KCiAgICBXSFkgT05FIEZJTEUgUEVSIFRBQkxFIEFORCBOT1QgT05FIFBFUiBQSUNUVVJFCiAgICBCZWNhdXNlIHRoZSB0d28gdGFibGVzIGJlaGluZCBhIHBpY3R1cmUgaGF2ZSBkaWZmZXJlbnQgc2hhcGVzIC0gb25lIHJvdyBwZXIKICAgIHJpbmcgaW4gdGhlIHByb2ZpbGUsIG9uZSByb3cgcGVyIG9iamVjdCBpbiB0aGUgcG9zaXRpb25zIC0gYW5kIHB1dHRpbmcgdGhlbSBpbgogICAgb25lIGZpbGUgbWVhbnMgb25lIG9mIHRoZW0gc3RhcnRzIGhhbGZ3YXkgZG93biwgd2hpY2ggaXMgZXhhY3RseSB0aGUgbGF5b3V0CiAgICB0aGF0IG1ha2VzIGEgc3ByZWFkc2hlZXQgYXdrd2FyZCB0byBjaGFydCBmcm9tLiBUd28gZmlsZXMsIGVhY2ggYSBjbGVhbgogICAgcmVjdGFuZ2xlIHdpdGggYSBoZWFkZXIgcm93LCBjYW4gYmUgY2hhcnRlZCBieSBzZWxlY3RpbmcgdHdvIGNvbHVtbnMuCgogICAgQ1NWIHJhdGhlciB0aGFuIGEgc3ByZWFkc2hlZXQgZm9ybWF0IG9uIHB1cnBvc2U6IGl0IG9wZW5zIGluIEV4Y2VsLCBpdCBvcGVucwogICAgaW4gYW55dGhpbmcgZWxzZSwgYW5kIGl0IG5lZWRzIG5vIGxpYnJhcnkgdGhhdCBtaWdodCBub3QgYmUgaW5zdGFsbGVkIG9uIHRoZQogICAgbWFjaGluZSB0aGlzIGVuZHMgdXAgcnVubmluZyBvbi4KCiAgICBSZXR1cm5zIHRoZSBwYXRocyB3cml0dGVuLgogICAgIiIiCiAgICBpbXBvcnQgY3N2IGFzIF9jc3YKCiAgICBzdGVtID0gb3MucGF0aC5zcGxpdGV4dChwYXRoKVswXQogICAgd3JpdHRlbiA9IFtdCiAgICBmb3IgbmFtZSwgKGhlYWQsIHJvd3MpIGluIHNoZWV0cy5pdGVtcygpOgogICAgICAgIHNsdWcgPSAiXyIuam9pbihwIGZvciBwIGluCiAgICAgICAgICAgICAgICAgICAgICAgICIiLmpvaW4oYyBpZiBjLmlzYWxudW0oKSBlbHNlICJfIiBmb3IgYyBpbiBuYW1lKS5zcGxpdCgiXyIpCiAgICAgICAgICAgICAgICAgICAgICAgIGlmIHApLmxvd2VyKCkKICAgICAgICBvdXQgPSAie31fe30uY3N2Ii5mb3JtYXQoc3RlbSwgc2x1ZykKICAgICAgICB3aXRoIG9wZW4ob3V0LCAidyIsIG5ld2xpbmU9IiIsIGVuY29kaW5nPSJ1dGYtOCIpIGFzIGZoOgogICAgICAgICAgICB3cml0ZXIgPSBfY3N2LndyaXRlcihmaCkKICAgICAgICAgICAgd3JpdGVyLndyaXRlcm93KGhlYWQpCiAgICAgICAgICAgIHdyaXRlci53cml0ZXJvd3Mocm93cykKICAgICAgICB3cml0dGVuLmFwcGVuZChvdXQpCiAgICByZXR1cm4gd3JpdHRlbgoKZGVmIHdyaXRlX3RyZWF0bWVudF93b3JrYm9vayhwYXRoLCB0cmVhdG1lbnQsIHJlc3VsdCwgc2NhbGUsIHVuaXQpOgogICAgIiIiVGhlIHR3byB0YWJsZXMgYmVoaW5kIG9uZSB0cmVhdG1lbnQncyBwaWN0dXJlLiIiIgogICAgcmV0dXJuIF93cml0ZV90YWJsZXMocGF0aCwgewogICAgICAgICJwcm9maWxlIjogX3Byb2ZpbGVfcm93cyhyZXN1bHQsIHNjYWxlLCB1bml0KSwKICAgICAgICAicG9zaXRpb25zIjogX3Bvc2l0aW9uX3Jvd3MocmVzdWx0LCBzY2FsZSwgdW5pdCksCiAgICB9KQoKZGVmIHdyaXRlX292ZXJsYXlfd29ya2Jvb2socGF0aCwgdHJlYXRtZW50cywgcmVzdWx0cywgc2NhbGUsIHVuaXQpOgogICAgIiIiT25lIHRhYmxlIHBlciBjdXJ2ZSBpbiB0aGUgb3ZlcmxheS4iIiIKICAgIHNoZWV0cyA9IHt9CiAgICBmb3IgdHJlYXRtZW50LCByZXN1bHQgaW4gemlwKHRyZWF0bWVudHMsIHJlc3VsdHMpOgogICAgICAgIG5hbWUgPSAiIi5qb2luKGMgaWYgKGMuaXNhbG51bSgpIG9yIGMgPT0gIiAiKSBlbHNlICIgIgogICAgICAgICAgICAgICAgICAgICAgIGZvciBjIGluIHRyZWF0bWVudFsibGFiZWwiXSkuc3RyaXAoKQogICAgICAgIHNoZWV0c1tuYW1lWzozMV0gb3IgInRyZWF0bWVudCJdID0gX3Byb2ZpbGVfcm93cyhyZXN1bHQsIHNjYWxlLCB1bml0KQogICAgcmV0dXJuIF93cml0ZV90YWJsZXMocGF0aCwgc2hlZXRzKQoKZGVmIG9uZV90cmVhdG1lbnRfZmlndXJlKHRyZWF0bWVudCwgcmVzdWx0LCB1bml0X2xhYmVsLCB0aXRsZSwgc2NhbGU9MS4wLAogICAgICAgICAgICAgICAgICAgICAgICAgY29sb3VyX2luZGV4PTApOgogICAgIiIiT25lIHRyZWF0bWVudCBvbiBpdHMgb3duIHNoZWV0OiB3aGVyZSB0aGUgb2JqZWN0cyBhcmUsIGFuZCB0aGVpciBwcm9maWxlLgoKICAgIFdIWSBUSEUgU0FNRSBUSElORyBJUyBEUkFXTiBUV0lDRQogICAgVGhlIGNvbWJpbmVkIHNoZWV0IGlzIGZvciBjb21wYXJpbmcgdGhlIHRyZWF0bWVudHMgYWdhaW5zdCBlYWNoIG90aGVyLCB3aGljaAogICAgaXMgd2hhdCBpdCBpcyBmb3IgYW5kIHdoYXQgaXQgaXMgZ29vZCBhdC4gSXQgaXMgbm90IHdoYXQgeW91IHB1dCBpbiBmcm9udCBvZgogICAgc29tZWJvZHksIG9yIGludG8gYSByZXBvcnQ6IGF0IGZvdXIgaW5jaGVzIGhpZ2ggYSBwYW5lbCBpcyB0b28gc21hbGwgdG8gcmVhZCBhCiAgICByYWRpdXMgb2ZmIGl0LCBhbmQgdHdvIHRyZWF0bWVudHMgc2lkZSBieSBzaWRlIGNvbXBldGUgZm9yIHRoZSBleWUgd2hlbiBvbmx5CiAgICBvbmUgb2YgdGhlbSBpcyB1bmRlciBkaXNjdXNzaW9uLiBTbyBlYWNoIGlzIHdyaXR0ZW4gYWdhaW4gb24gaXRzIG93biwgZnVsbAogICAgc2l6ZSwgYW5kIHRoZSBvdmVybGF5IHRvby4gVGhlIGNvbWJpbmVkIHNoZWV0IGlzIHVuY2hhbmdlZDsgdGhlc2UgYXJlIGV4dHJhLgogICAgIiIiCiAgICBjb2wgPSBDT0xPVVJTW2NvbG91cl9pbmRleCAlIGxlbihDT0xPVVJTKV0KICAgIGZpZyA9IHBsdC5maWd1cmUoZmlnc2l6ZT0oMTMuMCwgNi4yKSkKICAgIGdzID0gZmlnLmFkZF9ncmlkc3BlYygxLCAyLCB3c3BhY2U9MC4yOCkKCiAgICBkeHUsIGR5dSA9IHJlc3VsdFsiZHgiXSAqIHNjYWxlLCByZXN1bHRbImR5Il0gKiBzY2FsZQogICAgcnUgPSByZXN1bHRbInIiXSAqIHNjYWxlCiAgICBkdSA9IHJlc3VsdFsiZGVuc2l0eSJdIC8gc2NhbGUgKiogMgogICAgZXUgPSByZXN1bHRbImRlbnNpdHlfZXJyIl0gLyBzY2FsZSAqKiAyCgogICAgYXggPSBmaWcuYWRkX3N1YnBsb3QoZ3NbMCwgMF0pCiAgICBheC5zY2F0dGVyKGR4dSwgZHl1LCBzPTE0LCBjPWNvbCwgYWxwaGE9MC43LCBsaW5ld2lkdGhzPTApCiAgICBheC5wbG90KDAsIDAsICIrIiwgYz0iayIsIG1zPTE0LCBtZXc9MS44KQogICAgbGltID0gMS4wNiAqIG1heChucC5hYnMoZHh1KS5tYXgoKSwgbnAuYWJzKGR5dSkubWF4KCkpCiAgICBheC5zZXRfeGxpbSgtbGltLCBsaW0pCiAgICBheC5zZXRfeWxpbSgtbGltLCBsaW0pCiAgICBheC5zZXRfYXNwZWN0KCJlcXVhbCIsIGFkanVzdGFibGU9ImJveCIpCiAgICBheC5zZXRfeGxhYmVsKCJ4IGZyb20gY2VudHJlIFsiICsgdW5pdF9sYWJlbCArICJdIikKICAgIGF4LnNldF95bGFiZWwoInkgZnJvbSBjZW50cmUgWyIgKyB1bml0X2xhYmVsICsgIl0iKQogICAgYXguZ3JpZChhbHBoYT0wLjE1LCBsdz0wLjYpCiAgICBheC5zZXRfdGl0bGUoIndoZXJlIHRoZXkgYXJlIiwgZm9udHNpemU9MTEsIGxvYz0ibGVmdCIpCgogICAgYXgyID0gZmlnLmFkZF9zdWJwbG90KGdzWzAsIDFdKQogICAgZyA9IHJlc3VsdFsiY29tcGxldGUiXQogICAgZWRnZXMgPSByZXN1bHRbImVkZ2VzIl0gKiBzY2FsZQogICAgc3RlcF9wcm9maWxlKGF4MiwgZWRnZXMsIG5wLndoZXJlKGcsIGR1LCBucC5uYW4pLCBjb2wpCiAgICBpZiBnLmFueSgpOgogICAgICAgIGF4Mi5lcnJvcmJhcihydVtnXSwgZHVbZ10sIHllcnI9ZXVbZ10sIGZtdD0ibm9uZSIsIGVjb2xvcj1jb2wsCiAgICAgICAgICAgICAgICAgICAgIGVsaW5ld2lkdGg9MS4yLCBjYXBzaXplPTMsIGFscGhhPTAuOCkKICAgIGlmICh+ZykuYW55KCk6CiAgICAgICAgc3RlcF9wcm9maWxlKGF4MiwgZWRnZXMsIG5wLndoZXJlKH5nLCBkdSwgbnAubmFuKSwgY29sLAogICAgICAgICAgICAgICAgICAgICBsdz0xLjMsIGFscGhhPTAuNDUsIGxzPSItLSIsCiAgICAgICAgICAgICAgICAgICAgIGxhYmVsPSJyaW5nIGNsaXBwZWQgYnkgdGhlIGZyYW1lIikKICAgICAgICBheDIubGVnZW5kKGZvbnRzaXplPTksIGZyYW1lb249RmFsc2UsIGxvYz0ibG93ZXIgbGVmdCIpCiAgICBheDIuc2V0X3lzY2FsZSgibG9nIikKICAgIGF4Mi5zZXRfeGxhYmVsKCJkZXByb2plY3RlZCByYWRpdXMgWyIgKyB1bml0X2xhYmVsICsgIl0iKQogICAgYXgyLnNldF95bGFiZWwoInN1cmZhY2UgZGVuc2l0eSBbbiBwZXIgIiArIHVuaXRfbGFiZWwgKyAiXjJdIikKICAgIGF4Mi5ncmlkKGFscGhhPTAuMTUsIGx3PTAuNikKICAgIGF4Mi5zZXRfdGl0bGUoIk4gPSAiICsgc3RyKGludChyZXN1bHRbImNvdW50cyJdLnN1bSgpKSksIGZvbnRzaXplPTEwLAogICAgICAgICAgICAgICAgICBsb2M9InJpZ2h0IikKICAgIGF4Mi5zZXRfdGl0bGUoImhvdyBtYW55IHBlciB1bml0IGFyZWEsIGFnYWluc3QgcmFkaXVzIiwgZm9udHNpemU9MTEsCiAgICAgICAgICAgICAgICAgIGxvYz0ibGVmdCIpCgogICAgZmlnLnN1cHRpdGxlKCJ7fSAtIHt9Ii5mb3JtYXQodGl0bGUsIHRyZWF0bWVudFsibGFiZWwiXSksIGZvbnRzaXplPTEzKQogICAgZmlnLnRleHQoMC4wOSwgMC4wMTUsCiAgICAgICAgICAgICBjaHIoMTApLmpvaW4odGV4dHdyYXAud3JhcCh0cmVhdG1lbnRbIm5vdGUiXSwgd2lkdGg9MTUwKSksCiAgICAgICAgICAgICBmb250c2l6ZT05LCBjb2xvcj0iIzRhNGE0YSIsIHZhPSJib3R0b20iKQogICAgZmlnLnN1YnBsb3RzX2FkanVzdChib3R0b209MC4xNywgdG9wPTAuOTApCiAgICByZXR1cm4gZmlnCgpkZWYgb3ZlcmxheV9maWd1cmUodHJlYXRtZW50cywgcmVzdWx0cywgdW5pdF9sYWJlbCwgdGl0bGUsIHNjYWxlPTEuMCk6CiAgICAiIiJUaGUgdHJlYXRtZW50cyBvbiBvbmUgcGFpciBvZiBheGVzLCBvbiBhIHNoZWV0IG9mIGl0cyBvd24uIiIiCiAgICBmaWcsIGF4ID0gcGx0LnN1YnBsb3RzKGZpZ3NpemU9KDEzLjAsIDYuMikpCiAgICBmb3IgaSwgKHQsIHJlcykgaW4gZW51bWVyYXRlKHppcCh0cmVhdG1lbnRzLCByZXN1bHRzKSk6CiAgICAgICAgZyA9IHJlc1siY29tcGxldGUiXQogICAgICAgIGlmIGcuYW55KCk6CiAgICAgICAgICAgIHN0ZXBfcHJvZmlsZShheCwgcmVzWyJlZGdlcyJdICogc2NhbGUsCiAgICAgICAgICAgICAgICAgICAgICAgICBucC53aGVyZShnLCByZXNbImRlbnNpdHkiXSAvIHNjYWxlICoqIDIsIG5wLm5hbiksCiAgICAgICAgICAgICAgICAgICAgICAgICBDT0xPVVJTW2kgJSBsZW4oQ09MT1VSUyldLCBsYWJlbD10WyJsYWJlbCJdKQogICAgYXguc2V0X3lzY2FsZSgibG9nIikKICAgIGF4LnNldF94bGFiZWwoImRlcHJvamVjdGVkIHJhZGl1cyBbIiArIHVuaXRfbGFiZWwgKyAiXSIpCiAgICBheC5zZXRfeWxhYmVsKCJzdXJmYWNlIGRlbnNpdHkgW24gcGVyICIgKyB1bml0X2xhYmVsICsgIl4yXSIpCiAgICBheC5ncmlkKGFscGhhPTAuMTgsIGx3PTAuNikKICAgIGF4LmxlZ2VuZChmb250c2l6ZT0xMCwgZnJhbWVvbj1GYWxzZSkKICAgIGF4LnNldF90aXRsZSgie30gLSBhbGwgdHJlYXRtZW50cyBvdmVybGFpZDsgdGhlIHNwcmVhZCBiZXR3ZWVuIHRoZW0gaXMgdGhlICIKICAgICAgICAgICAgICAgICAidW5jZXJ0YWludHkgb24gdGhlIHJhZGlpIi5mb3JtYXQodGl0bGUpLCBmb250c2l6ZT0xMiwgbG9jPSJsZWZ0IikKICAgIGZpZy50aWdodF9sYXlvdXQoKQogICAgcmV0dXJuIGZpZwoKZGVmIHJ1bl9wcm9maWxlcyh4LCB5LCBjZW50cmUsIGZyYW1lLCBvdXRfcHJlZml4LCB0aXRsZSwKICAgICAgICAgICAgICAgICBnYWxheHk9Tm9uZSwgZ2VvbT1Ob25lLCBub3J0aF9hbmdsZT05MC4wLCBtaXJyb3JlZD1GYWxzZSwKICAgICAgICAgICAgICAgICBtYW51YWxfaW5jPU5vbmUsIG1hbnVhbF9wYV9pbWFnZT1Ob25lLCBtYW51YWxfcGFfc2t5PU5vbmUsCiAgICAgICAgICAgICAgICAgc2VjdG9yX21ldGhvZD1GYWxzZSwgc2VjdG9ycz0yMCwgc2VjdG9yX3N0ZXA9NS4wLAogICAgICAgICAgICAgICAgIHJpbmdzPTIwLCBzY2FsZT0xLjAsIHVuaXQ9InB4IiwgdmVyYm9zZT1UcnVlKToKICAgICIiIkRlcHJvamVjdCwgYmluLCBwbG90IGFuZCBzYXZlLiBSZXR1cm5zICh0cmVhdG1lbnRzLCByZXN1bHRzLCBwYXRocykuCgogICAgb3V0X3ByZWZpeCBpcyBhIHBhdGggc3RlbTsgdGhpcyB3cml0ZXMgPHByZWZpeD5fcHJvZmlsZXMuY3N2IGFuZAogICAgPHByZWZpeD5fcHJvZmlsZXMucG5nIG5leHQgdG8gaXQuCiAgICAiIiIKICAgIHggPSBucC5hc2FycmF5KHgsIGR0eXBlPWZsb2F0KQogICAgeSA9IG5wLmFzYXJyYXkoeSwgZHR5cGU9ZmxvYXQpCgogICAgIyBUb28gbWFueSByaW5ncyBmb3IgdG9vIGZldyBvYmplY3RzIGRvZXMgbm90IGFkZCBkZXRhaWwsIGl0IG1hbnVmYWN0dXJlcwogICAgIyBub2lzZTogcmluZ3MgaG9sZGluZyBvbmUgb2JqZWN0IG9yIG5vbmUsIGVycm9yIGJhcnMgdGFsbGVyIHRoYW4gdGhlIHByb2ZpbGUsCiAgICAjIGFuZCBhIGphZ2dlZCBsaW5lIHRoYXQgaW52aXRlcyByZWFkaW5nIHN0cnVjdHVyZSBpbnRvIGNvdW50aW5nIHN0YXRpc3RpY3MuCiAgICAjIFNlZW4gb24gYSByZWFsIHJ1biAtIDE4MSBibHVlIGtub3RzIHNwcmVhZCBvdmVyIDMwIHJpbmdzLCB3aG9sZSByaW5ncyBlbXB0eS4KICAgIGlmIHJpbmdzIGFuZCBsZW4oeCkgYW5kIGxlbih4KSAvIGZsb2F0KHJpbmdzKSA8IDguMDoKICAgICAgICBzdWdnZXN0ZWQgPSBtYXgoNCwgaW50KGxlbih4KSAvIDEyKSkKICAgICAgICBwcmludCgiICBub3RlOiB7fSBvYmplY3RzIG92ZXIge30gcmluZ3MgaXMgYWJvdXQgezouMWZ9IHBlciByaW5nLCBzbyB0aGUgIgogICAgICAgICAgICAgICJwcm9maWxlIGJlbG93Ii5mb3JtYXQobGVuKHgpLCByaW5ncywgbGVuKHgpIC8gZmxvYXQocmluZ3MpKSkKICAgICAgICBwcmludCgiICAgICAgICB3aWxsIGJlIGRvbWluYXRlZCBieSBjb3VudGluZyBub2lzZS4gQWJvdXQge30gcmluZ3Mgd291bGQgIgogICAgICAgICAgICAgICJiZSByZWFkYWJsZS4iLmZvcm1hdChzdWdnZXN0ZWQpKQoKICAgIGlmIGdlb20gaXMgTm9uZSBhbmQgZ2FsYXh5OgogICAgICAgIHRyeToKICAgICAgICAgICAgZGVzY3JpYmUgPSBkZXNjcmliZV9fZ2FsYXh5X2NhdGFsb2d1ZSAgIyBmcm9tIGdhbGF4eV9jYXRhbG9ndWUsIGlubGluZWQgYWJvdmUKICAgICAgICAgICAgZ2VvbSA9IGh5cGVybGVkYV9nZW9tZXRyeShnYWxheHkpCiAgICAgICAgICAgIGlmIHZlcmJvc2U6CiAgICAgICAgICAgICAgICBwcmludCgiXG5IeXBlckxFREEgZ2VvbWV0cnkgZm9yICIgKyBnYWxheHkgKyAiOiIpCiAgICAgICAgICAgICAgICBwcmludChkZXNjcmliZShnZW9tKSkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGV4YzoKICAgICAgICAgICAgaWYgdmVyYm9zZToKICAgICAgICAgICAgICAgIHByaW50KCJcbkNhdGFsb2d1ZSBsb29rdXAgZmFpbGVkICh7fToge30pLiBDb250aW51aW5nIHdpdGhvdXQgaXQuIgogICAgICAgICAgICAgICAgICAgICAgLmZvcm1hdCh0eXBlKGV4YykuX19uYW1lX18sIGV4YykpCgogICAgdHJlYXRtZW50cyA9IFt0cmVhdG1lbnRfbm9uZSgpXQogICAgdCA9IHRyZWF0bWVudF9jYXRhbG9ndWUoZ2VvbSwgbm9ydGhfYW5nbGUsIG1pcnJvcmVkKQogICAgaWYgdDoKICAgICAgICB0cmVhdG1lbnRzLmFwcGVuZCh0KQogICAgdCA9IHRyZWF0bWVudF9tYW51YWwobWFudWFsX2luYywgbWFudWFsX3BhX2ltYWdlLCBtYW51YWxfcGFfc2t5LAogICAgICAgICAgICAgICAgICAgICAgICAgbm9ydGhfYW5nbGUsIG1pcnJvcmVkKQogICAgaWYgdDoKICAgICAgICB0cmVhdG1lbnRzLmFwcGVuZCh0KQogICAgaWYgc2VjdG9yX21ldGhvZDoKICAgICAgICB0cmVhdG1lbnRzLmFwcGVuZCh0cmVhdG1lbnRfc2VjdG9yKHggLSBjZW50cmVbMF0sIHkgLSBjZW50cmVbMV0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBzZWN0b3JzLCBzZWN0b3Jfc3RlcCkpCgogICAgcmVzdWx0cyA9IGNvbXB1dGVfcHJvZmlsZXMoeCwgeSwgY2VudHJlLCBmcmFtZSwgdHJlYXRtZW50cywgcmluZ3M9cmluZ3MpCgogICAgaWYgdmVyYm9zZToKICAgICAgICBwcmludCgiXG5UcmVhdG1lbnRzOiIpCiAgICAgICAgZm9yIHQsIHJlcyBpbiB6aXAodHJlYXRtZW50cywgcmVzdWx0cyk6CiAgICAgICAgICAgIHByaW50KCIgIC0ge306IHJfbWF4ID0gezouMWZ9IHt9LCB7fSBvZiB7fSByaW5ncyBjbGlwcGVkIGJ5IHRoZSBmcmFtZSIKICAgICAgICAgICAgICAgICAgLmZvcm1hdCh0WyJsYWJlbCJdLCByZXNbInJfbWF4Il0gKiBzY2FsZSwgdW5pdCwKICAgICAgICAgICAgICAgICAgICAgICAgICBpbnQoKH5yZXNbImNvbXBsZXRlIl0pLnN1bSgpKSwgcmluZ3MpKQogICAgICAgICAgICBwcmludCgiICAgICAgIiArIHRbIm5vdGUiXSkKCiAgICBjc3ZfcGF0aCA9IG91dF9wcmVmaXggKyAiX3Byb2ZpbGVzLmNzdiIKICAgIHBuZ19wYXRoID0gb3V0X3ByZWZpeCArICJfcHJvZmlsZXMucG5nIgogICAgd3JpdGVfcHJvZmlsZXNfY3N2KGNzdl9wYXRoLCB0cmVhdG1lbnRzLCByZXN1bHRzLCBzY2FsZSwgdW5pdCkKICAgIGZpZyA9IHByb2ZpbGVfZmlndXJlKHRyZWF0bWVudHMsIHJlc3VsdHMsIHVuaXQsIHRpdGxlLCBzY2FsZSkKICAgIGZpZy5zYXZlZmlnKHBuZ19wYXRoLCBkcGk9MTUwLCBiYm94X2luY2hlcz0idGlnaHQiKQogICAgcGx0LmNsb3NlKGZpZykKICAgIGNvbWJpbmVkX2Jvb2tzID0gd3JpdGVfb3ZlcmxheV93b3JrYm9vaygKICAgICAgICBvcy5wYXRoLnNwbGl0ZXh0KHBuZ19wYXRoKVswXSArICIuY3N2IiwgdHJlYXRtZW50cywgcmVzdWx0cywgc2NhbGUsIHVuaXQpCgogICAgaWYgdmVyYm9zZToKICAgICAgICBwcmludCgiXG5Xcm90ZSAiICsgY3N2X3BhdGgpCiAgICAgICAgcHJpbnQoIldyb3RlICIgKyBwbmdfcGF0aCkKICAgICMgQW5kIGVhY2ggcGFydCBhZ2FpbiBvbiBhIHNoZWV0IG9mIGl0cyBvd24sIGZ1bGwgc2l6ZS4gVGhlIGNvbWJpbmVkIG9uZQogICAgIyBhYm92ZSBpcyB1bnRvdWNoZWQ7IHRoZXNlIGFyZSBpbiBhZGRpdGlvbiB0byBpdC4KICAgIGV4dHJhX3BhdGhzID0gW10KICAgIGZvciBpLCAodHJlYXQsIHJlcykgaW4gZW51bWVyYXRlKHppcCh0cmVhdG1lbnRzLCByZXN1bHRzKSk6CiAgICAgICAgc2x1ZyA9ICcnLmpvaW4oYyBpZiBjLmlzYWxudW0oKSBlbHNlICdfJyBmb3IgYyBpbiB0cmVhdFsnbGFiZWwnXSkKICAgICAgICBzbHVnID0gJ18nLmpvaW4ocCBmb3IgcCBpbiBzbHVnLnNwbGl0KCdfJykgaWYgcCkubG93ZXIoKQogICAgICAgIG9uZV9wYXRoID0gJ3t9X3Byb2ZpbGVfe30ucG5nJy5mb3JtYXQob3V0X3ByZWZpeCwgc2x1ZykKICAgICAgICBvbmUgPSBvbmVfdHJlYXRtZW50X2ZpZ3VyZSh0cmVhdCwgcmVzLCB1bml0LCB0aXRsZSwgc2NhbGUsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY29sb3VyX2luZGV4PWkpCiAgICAgICAgb25lLnNhdmVmaWcob25lX3BhdGgsIGRwaT0xNTAsIGJib3hfaW5jaGVzPSd0aWdodCcpCiAgICAgICAgcGx0LmNsb3NlKG9uZSkKICAgICAgICBleHRyYV9wYXRocy5hcHBlbmQob25lX3BhdGgpCiAgICAgICAgZXh0cmFfcGF0aHMuZXh0ZW5kKHdyaXRlX3RyZWF0bWVudF93b3JrYm9vaygKICAgICAgICAgICAgb25lX3BhdGhbOi00XSArICcuY3N2JywgdHJlYXQsIHJlcywgc2NhbGUsIHVuaXQpKQogICAgb3Zlcl9wYXRoID0gb3V0X3ByZWZpeCArICdfcHJvZmlsZV9hbGxfb3ZlcmxhaWQucG5nJwogICAgb3ZlciA9IG92ZXJsYXlfZmlndXJlKHRyZWF0bWVudHMsIHJlc3VsdHMsIHVuaXQsIHRpdGxlLCBzY2FsZSkKICAgIG92ZXIuc2F2ZWZpZyhvdmVyX3BhdGgsIGRwaT0xNTAsIGJib3hfaW5jaGVzPSd0aWdodCcpCiAgICBwbHQuY2xvc2Uob3ZlcikKICAgIGV4dHJhX3BhdGhzLmFwcGVuZChvdmVyX3BhdGgpCiAgICBleHRyYV9wYXRocy5leHRlbmQod3JpdGVfb3ZlcmxheV93b3JrYm9vaygKICAgICAgICBvdmVyX3BhdGhbOi00XSArICcuY3N2JywgdHJlYXRtZW50cywgcmVzdWx0cywgc2NhbGUsIHVuaXQpKQogICAgZXh0cmFfcGF0aHNbOjBdID0gY29tYmluZWRfYm9va3MKICAgIGlmIHZlcmJvc2U6CiAgICAgICAgZm9yIHAgaW4gZXh0cmFfcGF0aHM6CiAgICAgICAgICAgIHByaW50KCdXcm90ZSAnICsgcCkKICAgIHJldHVybiB0cmVhdG1lbnRzLCByZXN1bHRzLCB0dXBsZShbY3N2X3BhdGgsIHBuZ19wYXRoXSArIGV4dHJhX3BhdGhzKQoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KIyBib3Jyb3dlZCBmcm9tIEhJSV9Gcm9tX0ltYWdlcy5weQojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCgptYXRwbG90bGliLnVzZSgiQWdnIikKCndhcm5pbmdzLmZpbHRlcndhcm5pbmdzKCJpZ25vcmUiKQoKcGFzcyAgIyBmcm9tIHJhZGlhbF9wcm9maWxlcywgaW5saW5lZCBhYm92ZQoKcGFzcyAgIyBmcm9tIGludGVyYWN0aXZlX2lucHV0LCBpbmxpbmVkIGFib3ZlCgpwbGFuX2ZvbGRlciA9IHBsYW47IGRlc2NyaWJlX2ZvbGRlciA9IGRlc2NyaWJlX19mcmFtZV9pbnZlbnRvcnkgICMgZnJvbSBmcmFtZV9pbnZlbnRvcnksIGlubGluZWQgYWJvdmUKCnBsYW5fZm9sZGVyID0gcGxhbjsgZGVzY3JpYmVfZm9sZGVyID0gZGVzY3JpYmVfX2ZyYW1lX2ludmVudG9yeSAgIyBmcm9tIGZyYW1lX2ludmVudG9yeSwgaW5saW5lZCBhYm92ZQoKcGFzcyAgIyBmcm9tIGdhbGF4eV9jYXRhbG9ndWUsIGlubGluZWQgYWJvdmUKCnBhc3MgICMgZnJvbSBkZXByb2plY3Rpb24sIGlubGluZWQgYWJvdmUKCnBhc3MgICMgZnJvbSBhbmNob3JfcmVnaXN0cmF0aW9uLCBpbmxpbmVkIGFib3ZlCgpwYXNzICAjIGZyb20gZ2FsYXh5X3dpbmRvdywgaW5saW5lZCBhYm92ZQoKZGVzY3JpYmVfd2luZG93ID0gZGVzY3JpYmVfX2dhbGF4eV93aW5kb3cgICMgZnJvbSBnYWxheHlfd2luZG93LCBpbmxpbmVkIGFib3ZlCgpwYXNzICAjIGZyb20gZ2FsYXh5X2V4dGVudCwgaW5saW5lZCBhYm92ZQoKZGVzY3JpYmVfZXh0ZW50ID0gZGVzY3JpYmVfX2dhbGF4eV9leHRlbnQgICMgZnJvbSBnYWxheHlfZXh0ZW50LCBpbmxpbmVkIGFib3ZlCgpkZXNjcmliZV9hbmNob3JzID0gZGVzY3JpYmVfX2FuY2hvcl9yZWdpc3RyYXRpb24gICMgZnJvbSBhbmNob3JfcmVnaXN0cmF0aW9uLCBpbmxpbmVkIGFib3ZlCgpkZWYgZG9ub3JfYmVzaWRlKHBhdGgsIHJvbGUpOgogICAgIiIiQSBzb2x2ZWQgZXhwb3N1cmUgaW4gdGhlIHNhbWUgZm9sZGVyIGFzIGBwYXRoYCwgd2hlbiBgcGF0aGAgaGFzIG5vIHNvbHV0aW9uLgoKICAgIFJldHVybnMgaXRzIHBhdGgsIG9yIE5vbmUgd2hlbiBgcGF0aGAgaXMgc29sdmVkIGFscmVhZHkgb3Igbm90aGluZyBiZXNpZGUgaXQKICAgIGlzLiBTYW1lIGZpbHRlciBmaXJzdCwgdGhlbiBhbnkgc29sdmVkIGZyYW1lIG9mIHRoZSBmaWVsZC4KICAgICIiIgogICAgdHJ5OgogICAgICAgIHJlYyA9IGluc3BlY3QocGF0aCkKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIGlmIHJlYy5nZXQoIndjcyIpOgogICAgICAgIHJldHVybiBOb25lCiAgICByZWNbInJvbGUiXSA9IHJlYy5nZXQoInJvbGUiKSBvciByb2xlCiAgICBoZXJlID0gb3MucGF0aC5hYnNwYXRoKHBhdGgpCiAgICBvdGhlcnMgPSBbciBmb3IgciBpbiBzY2FuX2ZvbGRlcihvcy5wYXRoLmRpcm5hbWUoaGVyZSksIHJlY3Vyc2l2ZT1GYWxzZSkKICAgICAgICAgICAgICBpZiBvcy5wYXRoLmFic3BhdGgoclsicGF0aCJdKSAhPSBoZXJlIGFuZCBub3Qgci5nZXQoImVycm9yIildCiAgICBkb25vciA9IGRvbm9yX2ZvcihyZWMsIG90aGVycykKICAgIHJldHVybiBkb25vclsicGF0aCJdIGlmIGRvbm9yIGVsc2UgTm9uZQoKZGVmIGZyYW1lc19mcm9tX2ZvbGRlcihmb2xkZXIsIHJvbGVzKToKICAgICIiIldvcmsgb3V0IHdoaWNoIGZpbGUgaW4gYSBmb2xkZXIgcGxheXMgZWFjaCBwYXJ0LCBhbmQgc2F5IHNvIGJlZm9yZSBzdGFydGluZy4KCiAgICBSZXR1cm5zIHRocmVlIHRoaW5ncyBwZXIgcm9sZTogdGhlIGNob3NlbiBmcmFtZSwgdGhlIHNvbHZlZCBleHBvc3VyZSB0aGF0IHdpbGwKICAgIGxlbmQgaXQgYSBwbGF0ZSBzb2x1dGlvbiBpZiBpdCBoYXMgbm9uZSwgYW5kIHRoZSBsaXN0IG9mIGV4cG9zdXJlcyB0byBjb21iaW5lCiAgICB3aGVyZSB0aGUgZm9sZGVyIGhlbGQgc2V2ZXJhbCBhbmQgbm90aGluZyBjb21iaW5lZC4KICAgICIiIgogICAgZGVmIHBpY2socm9sZSwgb3B0aW9ucyk6CiAgICAgICAgcHJpbnQoIiIpCiAgICAgICAgcHJpbnQoIk1vcmUgdGhhbiBvbmUgZnJhbWUgY291bGQgYmUgdGhlICIgKyByb2xlICsgIjoiKQogICAgICAgIGZvciBpLCBvcHQgaW4gZW51bWVyYXRlKG9wdGlvbnMsIDEpOgogICAgICAgICAgICBwcmludCgiICB7fS4ge30iLmZvcm1hdChpLCBvcHRbIm5hbWUiXSkpCiAgICAgICAgd2hpbGUgVHJ1ZToKICAgICAgICAgICAgYW5zd2VyID0gYXNrKCJXaGljaCBvbmUiLCAiMSIsIGludCkKICAgICAgICAgICAgaWYgMSA8PSBhbnN3ZXIgPD0gbGVuKG9wdGlvbnMpOgogICAgICAgICAgICAgICAgcmV0dXJuIG9wdGlvbnNbYW5zd2VyIC0gMV0KICAgICAgICAgICAgcHJpbnQoIiAgICBwaWNrIGEgbnVtYmVyIGZyb20gdGhlIGxpc3QiKQoKICAgIHJlY29yZHMsIGNob3NlbiwgbWVtYmVycywgZG9ub3JzLCBub3RlcyA9IHBsYW5fZm9sZGVyKGZvbGRlciwgcm9sZXM9cm9sZXMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBhc2s9cGljaykKICAgIHByaW50KGRlc2NyaWJlX2ZvbGRlcihyZWNvcmRzLCBjaG9zZW4sIG5vdGVzKSkKCiAgICBtaXNzaW5nID0gW3IgZm9yIHIgaW4gcm9sZXMgaWYgciBub3QgaW4gY2hvc2VuXQogICAgaWYgbWlzc2luZzoKICAgICAgICByYWlzZSBTeXN0ZW1FeGl0KCJOb3RoaW5nIGluIHRoZSBmb2xkZXIgaXMgdGhlICIgKyAiIG9yIHRoZSAiLmpvaW4obWlzc2luZykpCgogICAgIyBXaGVyZSB0aGUgZm9sZGVyIGhvbGRzIHNldmVyYWwgZXhwb3N1cmVzIG9mIGEgZmlsdGVyIGFuZCBub3RoaW5nIGNvbWJpbmVkLAogICAgIyBzYXkgd2hhdCBjb21iaW5pbmcgdGhlbSBpcyB3b3J0aCBhbmQgbGV0IHRoZSBwZXJzb24gZGVjaWRlLiBUaGUgZ2FpbiBpcyBhCiAgICAjIHJlYWwgbnVtYmVyLCBzbyBpdCBpcyBwdXQgb24gdGhlIHNjcmVlbiByYXRoZXIgdGhhbiBoaWRkZW4gYmVoaW5kIGEgZmxhZwogICAgIyBub2JvZHkgd2lsbCByZW1lbWJlciBhIHllYXIgZnJvbSBub3cuCiAgICB0b19jb21iaW5lID0ge30KICAgIGZvciByb2xlIGluIHJvbGVzOgogICAgICAgIGdyb3VwID0gbWVtYmVycy5nZXQocm9sZSkgb3IgW10KICAgICAgICBpZiBsZW4oZ3JvdXApIDwgMjoKICAgICAgICAgICAgY29udGludWUKICAgICAgICBpZiBhc2tfeWVzX25vKCJUaGVyZSBhcmUge30ge30gZXhwb3N1cmVzIGFuZCBubyBjb21iaW5lZCBmcmFtZS4gQ29tYmluaW5nICIKICAgICAgICAgICAgICAgICAgICAgICJ0aGVtIGxvd2VycyB0aGUgbm9pc2UgYnkgYWJvdXQgYSBmYWN0b3Igb2YgezouMWZ9LiAiCiAgICAgICAgICAgICAgICAgICAgICAiQ29tYmluZT8iLmZvcm1hdChsZW4oZ3JvdXApLCByb2xlLCBsZW4oZ3JvdXApICoqIDAuNSksCiAgICAgICAgICAgICAgICAgICAgICBUcnVlKToKICAgICAgICAgICAgdG9fY29tYmluZVtyb2xlXSA9IFtyWyJwYXRoIl0gZm9yIHIgaW4gZ3JvdXBdCiAgICAgICAgZWxzZToKICAgICAgICAgICAgcHJpbnQoIiAgdXNpbmcge30gb24gaXRzIG93biIuZm9ybWF0KGdyb3VwWzBdWyJuYW1lIl0pKQoKICAgIG5hbWUsIG5hbWVfc291cmNlID0gZ2FsYXh5X2Zyb21fcmVjb3JkcyhyZWNvcmRzLCBmb2xkZXIpCiAgICBpZiBuYW1lOgogICAgICAgIHByaW50KCJHYWxheHk6IHt9ICAgKGZyb20ge30pIi5mb3JtYXQobmFtZSwgbmFtZV9zb3VyY2UpKQoKICAgIHJldHVybiAoe3JvbGU6IGNob3Nlbltyb2xlXVsicGF0aCJdIGZvciByb2xlIGluIHJvbGVzfSwKICAgICAgICAgICAge3JvbGU6IChkb25vcnNbcm9sZV1bInBhdGgiXSBpZiBkb25vcnMuZ2V0KHJvbGUpIGVsc2UgTm9uZSkKICAgICAgICAgICAgIGZvciByb2xlIGluIHJvbGVzfSwKICAgICAgICAgICAgdG9fY29tYmluZSwgbmFtZSkKCmRlZiBoZWFkZXJfbnVtYmVyKGhlYWRlciwga2V5cyk6CiAgICAiIiJUaGUgZmlyc3Qgb2YgdGhlc2UgaGVhZGVyIGtleXdvcmRzIHRoYXQgaG9sZHMgYSBwb3NpdGl2ZSBudW1iZXIsIG9yIE5vbmUuIiIiCiAgICBmb3Iga2V5IGluIGtleXM6CiAgICAgICAgdHJ5OgogICAgICAgICAgICB2YWx1ZSA9IGZsb2F0KGhlYWRlci5nZXQoa2V5KSkKICAgICAgICBleGNlcHQgKFR5cGVFcnJvciwgVmFsdWVFcnJvcik6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgaWYgdmFsdWUgPT0gdmFsdWUgYW5kIHZhbHVlID4gMDoKICAgICAgICAgICAgcmV0dXJuIHZhbHVlCiAgICByZXR1cm4gTm9uZQoKZGVmIGxvYWRfZnJhbWUocGF0aCwgZG9ub3JfcGF0aD1Ob25lKToKICAgICIiIk9wZW4gYSBGSVRTIGZpbGUgYW5kIHJldHVybiB0aGUgc2NpZW5jZSBpbWFnZSwgaXRzIFdDUyBhbmQgYSBiYWQtcGl4ZWwgbWFzay4KCiAgICBBIGZyYW1lIHRoYXQgY2FtZSBvdXQgb2YgYSBtZWRpYW4gY29tYmluZSB1c3VhbGx5IGhhcyBubyBwbGF0ZSBzb2x1dGlvbiBsZWZ0CiAgICBpbiBpdCwgYmVjYXVzZSB0aGUgY29tYmluaW5nIHN0ZXAgd3JvdGUgYSBuZXcgZmlsZSBhbmQga2VwdCBhbG1vc3Qgbm90aGluZy4KICAgIFdoZW4gYGRvbm9yX3BhdGhgIG5hbWVzIGEgc29sdmVkIGV4cG9zdXJlIG9mIHRoZSBzYW1lIGZpZWxkLCBhIHNvbHV0aW9uIGlzCiAgICBmaXR0ZWQgZm9yIHRoaXMgZnJhbWUgYnkgbWF0Y2hpbmcgc3RhciBwYXR0ZXJucyBhZ2FpbnN0IHRoYXQgb25lLCBhbmQgdGhlCiAgICByZXNpZHVhbCBvZiB0aGF0IGZpdCBpcyBwcmludGVkIHNvIGl0IGNhbiBiZSBqdWRnZWQgcmF0aGVyIHRoYW4gdHJ1c3RlZC4KICAgICIiIgogICAgZnJvbSBhc3Ryb3B5LmlvIGltcG9ydCBmaXRzCiAgICBmcm9tIGFzdHJvcHkud2NzIGltcG9ydCBXQ1MKCiAgICBoZHVsID0gZml0cy5vcGVuKHBhdGgpCiAgICBzY2kgPSBOb25lCiAgICBmb3IgaGR1IGluIGhkdWw6CiAgICAgICAgaWYgaGR1LmRhdGEgaXMgbm90IE5vbmUgYW5kIGdldGF0dHIoaGR1LmRhdGEsICJuZGltIiwgMCkgPT0gMjoKICAgICAgICAgICAgc2NpID0gaGR1CiAgICAgICAgICAgIGJyZWFrCiAgICBpZiBzY2kgaXMgTm9uZToKICAgICAgICBoZHVsLmNsb3NlKCkKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKHBhdGggKyAiOiBubyAyLUQgaW1hZ2UgZm91bmQiKQoKICAgICMgU2luZ2xlIHByZWNpc2lvbiBvbiBwdXJwb3NlLiBDYW1lcmEgZGF0YSBhcnJpdmVzIGFzIDE2LWJpdCBpbnRlZ2Vycywgc28KICAgICMgZG91YmxlIHByZWNpc2lvbiBkb3VibGVzIHRoZSBtZW1vcnkgZm9yIG5vIGluZm9ybWF0aW9uIGF0IGFsbDogc2V2ZW4KICAgICMgc2lnbmlmaWNhbnQgZmlndXJlcyBhZ2FpbnN0IHBpeGVsIG5vaXNlIG9mIHRlbnMgb2YgY291bnRzLiBPbiBhIDYxLW1lZ2FwaXhlbAogICAgIyBmcmFtZSB0aGUgZGlmZmVyZW5jZSBpcyAyNDUgTUIgYWdhaW5zdCA0ODksIGFuZCBmb3VyIGZpbHRlcnMgb2YgdGhvc2Ugd2FzCiAgICAjIHdoYXQgZHJvdmUgdGhpcyBtYWNoaW5lIGludG8gc3dhcHBpbmcuCiAgICBkYXRhID0gbnAuYXNhcnJheShzY2kuZGF0YSwgZHR5cGU9bnAuZmxvYXQzMikKICAgIGhlYWRlciA9IHNjaS5oZWFkZXIKCiAgICBicG0gPSBOb25lCiAgICBmb3IgaGR1IGluIGhkdWw6CiAgICAgICAgaWYgaGR1Lm5hbWUudXBwZXIoKSA9PSAiQlBNIiBhbmQgaGR1LmRhdGEgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIGJwbSA9IG5wLmFzYXJyYXkoaGR1LmRhdGEpID4gMAogICAgICAgICAgICBicmVhawoKICAgIHdjcyA9IE5vbmUKICAgIHRyeToKICAgICAgICB3ID0gV0NTKGhlYWRlcikKICAgICAgICBpZiB3Lmhhc19jZWxlc3RpYWw6CiAgICAgICAgICAgIHdjcyA9IHcuY2VsZXN0aWFsCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHBhc3MKCiAgICBpbmZvID0gewogICAgICAgICJmaWx0ZXIiOiBzdHIoaGVhZGVyLmdldCgiRklMVEVSIiwgIj8iKSksCiAgICAgICAgImV4cHRpbWUiOiBmbG9hdChoZWFkZXIuZ2V0KCJFWFBUSU1FIiwgMS4wKSBvciAxLjApLAogICAgICAgICJwaXhlbF9zY2FsZSI6IGhlYWRlci5nZXQoIlBJWFNDQUxFIiwgTm9uZSksCiAgICAgICAgIm9iamVjdCI6IHN0cihoZWFkZXIuZ2V0KCJPQkpFQ1QiLCAiPyIpKSwKICAgICAgICAjIFdoYXQgdGhlIHBob3RvbWV0cnkgbmVlZHMgdG8ga25vdyBhYm91dCB0aGUgZGV0ZWN0b3IsIGNhcnJpZWQgYWxvbmcKICAgICAgICAjIHNvIHRoYXQgYSBmcmFtZSB0aGlzIHBpcGVsaW5lIHdyaXRlcyBzdGlsbCBzYXlzIGl0LgogICAgICAgICJzYXR1cmF0ZSI6IGhlYWRlcl9udW1iZXIoaGVhZGVyLCAoIlNBVFVSQVRFIiwgIk1BWExJTiIsICJTQVRMRVZFTCIpKSwKICAgICAgICAjIEVsZWN0cm9ucyBwZXIgQURVLiBFR0FJTiBmaXJzdDogb24gc29tZSBjYW1lcmFzIEdBSU4gaXMgdGhlIGdhaW4KICAgICAgICAjIFNFVFRJTkcgKDUxIG9uIHRoZSBLaW5uZXJldCBjb2xvdXIgY2FtZXJhKSwgbm90IGVsZWN0cm9ucyBwZXIgQURVLgogICAgICAgICJnYWluIjogbmV4dCgoZyBmb3IgZyBpbiAoaGVhZGVyX251bWJlcihoZWFkZXIsIChrLCkpIGZvciBrIGluCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAoIkVHQUlOIiwgIkdBSU4iKSkgaWYgZyBhbmQgZyA8IDIwKSwgTm9uZSksCiAgICAgICAgInJkbm9pc2UiOiBoZWFkZXJfbnVtYmVyKGhlYWRlciwgKCJSRE5PSVNFIiwgIlJFQUROT0lTIiwgIlJPTiIpKSwKICAgICAgICAibmNvbWJpbmUiOiBpbnQoaGVhZGVyX251bWJlcihoZWFkZXIsICgiTkNPTUJJTkUiLCkpIG9yIDEpLAogICAgICAgICJjb21idHlwZSI6IHN0cihoZWFkZXIuZ2V0KCJDT01CVFlQRSIsICIiKSksCiAgICAgICAgImRhdGUiOiBzdHIoaGVhZGVyLmdldCgiREFURS1PQlMiLCAiIikpLAogICAgfQogICAgaGR1bC5jbG9zZSgpCgogICAgaWYgd2NzIGlzIE5vbmUgYW5kIGRvbm9yX3BhdGg6CiAgICAgICAgcmVmZXJlbmNlLCByZWZfd2NzLCBfLCByZWZfaW5mbyA9IGxvYWRfZnJhbWUoZG9ub3JfcGF0aCkKICAgICAgICBpZiByZWZfd2NzIGlzIG5vdCBOb25lOgogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICB3Y3MsIGRpYWcgPSB3Y3NfZnJvbV9yZWZlcmVuY2UoZGF0YSwgcmVmZXJlbmNlLCByZWZfd2NzKQogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGV4YzoKICAgICAgICAgICAgICAgICMgYXN0cm9hbGlnbiBmaW5kcyBubyBzdGFyIHBhdHRlcm4gaW4gY29tbW9uIC0gdHlwaWNhbGx5IGEgZnJhbWUKICAgICAgICAgICAgICAgICMgd2l0aCB0aGUgc3RhcnMgdGFrZW4gb3V0LCBzdWNoIGFzIEgtYWxwaGEgbWludXMgY29udGludXVtLgogICAgICAgICAgICAgICAgcHJpbnQoIiAge306IGNvdWxkIG5vdCBmaXQgYSBwbGF0ZSBzb2x1dGlvbiBhZ2FpbnN0IHt9ICh7fToge30pIi5mb3JtYXQoCiAgICAgICAgICAgICAgICAgICAgb3MucGF0aC5iYXNlbmFtZShwYXRoKSwgb3MucGF0aC5iYXNlbmFtZShkb25vcl9wYXRoKSwKICAgICAgICAgICAgICAgICAgICB0eXBlKGV4YykuX19uYW1lX18sIGV4YyksIGZsdXNoPVRydWUpCiAgICAgICAgICAgICAgICByZXR1cm4gZGF0YSwgTm9uZSwgYnBtLCBpbmZvCiAgICAgICAgICAgIHByaW50KGRlc2NyaWJlX3djc19maXQob3MucGF0aC5iYXNlbmFtZShwYXRoKSwgZGlhZykpCiAgICAgICAgICAgIGlmIGluZm9bInBpeGVsX3NjYWxlIl0gaXMgTm9uZToKICAgICAgICAgICAgICAgIGluZm9bInBpeGVsX3NjYWxlIl0gPSByZWZfaW5mb1sicGl4ZWxfc2NhbGUiXQogICAgICAgICAgICBpZiBpbmZvWyJmaWx0ZXIiXSA9PSAiPyI6CiAgICAgICAgICAgICAgICBpbmZvWyJmaWx0ZXIiXSA9IHJlZl9pbmZvWyJmaWx0ZXIiXQogICAgICAgICAgICAjIEEgY29tYmluZWQgZnJhbWUgd3JpdHRlbiBieSBhbm90aGVyIHByb2dyYW0gb2Z0ZW4ga2VlcHMgbm8gaGVhZGVyCiAgICAgICAgICAgICMgYXQgYWxsLiBUaGUgY2FtZXJhIGFuZCB0aGUgZXhwb3N1cmUgbGVuZ3RoIGFyZSB0aG9zZSBvZiB0aGUgc29sdmVkCiAgICAgICAgICAgICMgZXhwb3N1cmUgaXQgY2FtZSBmcm9tLCBzbyB0aGV5IGFyZSB0YWtlbiBmcm9tIHRoZXJlLCBhbmQgc2FpZCBzby4KICAgICAgICAgICAgYm9ycm93ZWQgPSBbXQogICAgICAgICAgICBmb3Iga2V5LCBjYXJkIGluICgoImV4cHRpbWUiLCAiRVhQVElNRSIpLCAoInNhdHVyYXRlIiwgIlNBVFVSQVRFIiksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICgiZ2FpbiIsICJHQUlOIiksICgicmRub2lzZSIsICJSRE5PSVNFIiksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICgiZGF0ZSIsICJEQVRFLU9CUyIpLCAoIm9iamVjdCIsICJPQkpFQ1QiKSk6CiAgICAgICAgICAgICAgICBpZiBjYXJkIGluIGhlYWRlciBvciAoa2V5IGluICgic2F0dXJhdGUiLCAiZ2FpbiIsICJyZG5vaXNlIikKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBhbmQgaW5mb1trZXldIGlzIG5vdCBOb25lKToKICAgICAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICAgICAgaWYga2V5ID09ICJvYmplY3QiIGFuZCBpbmZvW2tleV0gIT0gIj8iOgogICAgICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgICAgICBpZiByZWZfaW5mby5nZXQoa2V5KSBub3QgaW4gKE5vbmUsICIiLCAiPyIpOgogICAgICAgICAgICAgICAgICAgIGluZm9ba2V5XSA9IHJlZl9pbmZvW2tleV0KICAgICAgICAgICAgICAgICAgICBib3Jyb3dlZC5hcHBlbmQoY2FyZCkKICAgICAgICAgICAgaWYgYm9ycm93ZWQ6CiAgICAgICAgICAgICAgICBwcmludCgiICB7fTogbm8ge30gaW4gaXRzIGhlYWRlcjsgdGFrZW4gZnJvbSB7fSIuZm9ybWF0KAogICAgICAgICAgICAgICAgICAgIG9zLnBhdGguYmFzZW5hbWUocGF0aCksICIsICIuam9pbihib3Jyb3dlZCksCiAgICAgICAgICAgICAgICAgICAgb3MucGF0aC5iYXNlbmFtZShkb25vcl9wYXRoKSksIGZsdXNoPVRydWUpCgogICAgcmV0dXJuIGRhdGEsIHdjcywgYnBtLCBpbmZvCgpkZWYgX2NlbnRyb2lkX2xvY2FsbHkoZGF0YSwgcG9zaXRpb25zLCBib3gpOgogICAgIiIiQ2VudHJlIG9mIGxpZ2h0IGluIGEgc21hbGwgYm94LCB3aXRoIHRoYXQgYm94J3Mgb3duIHNreSB0YWtlbiBvZmYgZmlyc3QuCgogICAgVGhlIG9uZSBwbGFjZSB0aGlzIGlzIGRvbmUsIGJlY2F1c2UgZ2V0dGluZyBpdCB3cm9uZyBpcyBpbnZpc2libGUuIFRoZSBza3kgaXMKICAgIHBhcnQgb2YgdGhlIHdlaWdodCBpbiBhIGNlbnRyZSBvZiBsaWdodCwgYW5kIGluIGEgYm94IG9mIGEgY291cGxlIG9mIGh1bmRyZWQKICAgIHBpeGVscyBpdCBpcyBtb3N0IG9mIHRoZSB3ZWlnaHQsIHNvIGxlYXZpbmcgaXQgaW4gcmV0dXJucyB0aGUgbWlkZGxlIG9mIHRoZSBib3gKICAgIGRyZXNzZWQgdXAgYXMgYSBtZWFzdXJlbWVudC4gRXZlcnkgY2FsbGVyIGhlcmUgd2FudHMgdGhlIHN0YXIsIG5vdCB0aGUgYm94LgoKICAgIFJldHVybnMgYW4gYXJyYXkgdGhlIHNhbWUgbGVuZ3RoIGFzIGBwb3NpdGlvbnNgLCB3aXRoIE5hTiB3aGVyZSBhIHBvc2l0aW9uIHNpdHMKICAgIHRvbyBuZWFyIGFuIGVkZ2Ugb3IgdGhlIHN0YW1wIGhvbGRzIG5vdGhpbmcgYWJvdmUgaXRzIG93biBza3kuCiAgICAiIiIKICAgIGRhdGEgPSBucC5hc2FycmF5KGRhdGEpCiAgICBwb3MgPSBucC5hc2FycmF5KHBvc2l0aW9ucywgZHR5cGU9ZmxvYXQpCiAgICBoYWxmID0gYm94IC8vIDIKICAgIG55LCBueCA9IGRhdGEuc2hhcGUKICAgIGdyaWRfeSwgZ3JpZF94ID0gbnAubWdyaWRbMDpib3gsIDA6Ym94XQogICAgb3V0ID0gbnAuZnVsbChwb3Muc2hhcGUsIG5wLm5hbikKICAgIGZvciBpLCAoeGMsIHljKSBpbiBlbnVtZXJhdGUocG9zKToKICAgICAgICBpZiBub3QgKG5wLmlzZmluaXRlKHhjKSBhbmQgbnAuaXNmaW5pdGUoeWMpKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICB4aSwgeWkgPSBpbnQocm91bmQoeGMpKSwgaW50KHJvdW5kKHljKSkKICAgICAgICBpZiBub3QgKGhhbGYgPD0geGkgPCBueCAtIGhhbGYgYW5kIGhhbGYgPD0geWkgPCBueSAtIGhhbGYpOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHN0YW1wID0gbnAubmFuX3RvX251bSgKICAgICAgICAgICAgZGF0YVt5aSAtIGhhbGY6eWkgKyBoYWxmICsgMSwgeGkgLSBoYWxmOnhpICsgaGFsZiArIDFdKS5hc3R5cGUoZmxvYXQpCiAgICAgICAgaWYgc3RhbXAuc2hhcGUgIT0gKGJveCwgYm94KToKICAgICAgICAgICAgY29udGludWUKICAgICAgICBzdGFtcCA9IHN0YW1wIC0gbnAubWVkaWFuKHN0YW1wKQogICAgICAgIHN0YW1wW3N0YW1wIDwgMC4wXSA9IDAuMCAgICAgICAgIyBuZWdhdGl2ZSBub2lzZSBtdXN0IG5vdCBwdWxsIHRoZSBjZW50cmUKICAgICAgICB0b3RhbCA9IHN0YW1wLnN1bSgpCiAgICAgICAgaWYgdG90YWwgPD0gMC4wOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIG91dFtpXSA9IFt4aSAtIGhhbGYgKyAoc3RhbXAgKiBncmlkX3gpLnN1bSgpIC8gdG90YWwsCiAgICAgICAgICAgICAgICAgIHlpIC0gaGFsZiArIChzdGFtcCAqIGdyaWRfeSkuc3VtKCkgLyB0b3RhbF0KICAgIHJldHVybiBvdXQKCmRlZiBjZW50cmVfb25fc3RhcnMoZGF0YSwgcG9zaXRpb25zLCBib3g9MTUsIG1heF9tb3ZlX3B4PTYuMCk6CiAgICAiIiJNb3ZlIGNhdGFsb2d1ZSBwb3NpdGlvbnMgb250byB0aGUgc3RhcnMgYXMgdGhpcyBmcmFtZSBhY3R1YWxseSBzaG93cyB0aGVtLgoKICAgIFdIWSBUSElTIElTIE5FRURFRAogICAgQSBjYXRhbG9ndWUgcG9zaXRpb24gYW5kIGEgc3RhciBpbiBhIGZyYW1lIGFyZSBub3QgdGhlIHNhbWUgdGhpbmcgdG8gYmV0dGVyCiAgICB0aGFuIGEgcGl4ZWwgb3IgdHdvLiBUaGUgcGxhdGUgc29sdXRpb24gaGFzIGl0cyBvd24gZXJyb3IsIHRoZSBjYXRhbG9ndWUgaXMgYXQKICAgIGEgZGlmZmVyZW50IGVwb2NoLCBhbmQgYSBjb21iaW5lZCBmcmFtZSBjYXJyaWVzIHdoYXRldmVyIGl0cyByZWdpc3RyYXRpb24gbGVmdAogICAgYmVoaW5kLiBNZWFzdXJlZCBvbiBhbiBNMzMgUiBmcmFtZTogdGhlIHJlYWwgc3RhcnMgc2F0IGEgbWVkaWFuIG9mIDIuNCBwaXhlbHMKICAgIGF3YXkgZnJvbSB3aGVyZSBHYWlhIHNhaWQgdGhleSB3ZXJlLCBhbmQgbm90IGJ5IGEgY29uc3RhbnQgYW1vdW50IC0gdGhlIHNwcmVhZAogICAgd2FzIDIuMiBwaXhlbHMuCgogICAgVGhhdCBtYXR0ZXJzIGJlY2F1c2UgbW9zdCBvZiB3aGF0IGlzIGRvbmUgd2l0aCB0aGVzZSBwb3NpdGlvbnMgaXMgYXBlcnR1cmUKICAgIHBob3RvbWV0cnksIGFuZCBhbiBhcGVydHVyZSBpcyBub3QgbW92ZWQgYnkgYmVpbmcgd3JvbmcuIEEgY2lyY2xlIG9mIHJhZGl1cyA2CiAgICBwbGFjZWQgMi40IHBpeGVscyBvZmYgY2VudHJlIG1pc3NlcyBhIGdvb2QgcGFydCBvZiB0aGUgc3RhciwgYW5kIG1pc3NlcyBhCiAgICBkaWZmZXJlbnQgcGFydCBvZiBldmVyeSBzdGFyLiBUaGUgc3VidHJhY3Rpb24gZmFjdG9yIGJ1aWx0IGZyb20gdGhvc2UgZmx1eGVzCiAgICBpcyB3cm9uZywgYW5kIHRoZSBzdGFyIHdpZHRocyBtZWFzdXJlZCBhcm91bmQgdGhvc2UgcG9pbnRzIGFyZSBub3QgdGhlIHdpZHRocwogICAgb2Ygc3RhcnMuIEJvdGggb2YgdGhvc2Ugd2VyZSB3cm9uZyBoZXJlLCBhbmQgYm90aCBzaG93ZWQgaW4gdGhlIHBpY3R1cmUuCgogICAgVGhlIG9uZSB0aGluZyB0aGF0IHdhcyByaWdodCB3YXMgdGhlIGZyYW1lIHJlZ2lzdHJhdGlvbiwgYW5kIG9ubHkgYmVjYXVzZSBpdAogICAgY2VudHJvaWRzIGluc2lkZSBhIGJveCBiZWZvcmUgZml0dGluZyAtIHdoaWNoIGlzIGV4YWN0bHkgd2hhdCB0aGlzIGRvZXMsIG9uY2UsCiAgICBmb3IgZXZlcnlvbmUgZWxzZS4KCiAgICBXSFkgVEhFIFNLWSBJUyBUQUtFTiBPRkYgRklSU1QKICAgIEEgY2VudHJlIG9mIGxpZ2h0IGlzIGEgd2VpZ2h0ZWQgYXZlcmFnZSwgYW5kIHRoZSBza3kgaXMgcGFydCBvZiB0aGUgd2VpZ2h0LiBJbgogICAgYSAxNSBwaXhlbCBib3ggb24gYSBmcmFtZSB3aGVyZSB0aGUgc2t5IHNpdHMgYXQgYSBmZXcgaHVuZHJlZCBjb3VudHMsIHRoZSB0d28KICAgIGh1bmRyZWQgc2t5IHBpeGVscyBvdXR3ZWlnaCB0aGUgc3RhciwgYW5kIHRoZWlyIGNlbnRyZSBvZiBsaWdodCBpcyB0aGUgbWlkZGxlCiAgICBvZiB0aGUgYm94LiBTbyB0aGUgYW5zd2VyIGNvbWVzIGJhY2sgYXMgdmVyeSBuZWFybHkgdGhlIHBvc2l0aW9uIGl0IHdhcyBnaXZlbiwKICAgIHdoYXRldmVyIHRoZSBzdGFyIGlzIGRvaW5nIC0gaXQgbG9va3MgbGlrZSBhIHN1Y2Nlc3NmdWwgbWVhc3VyZW1lbnQgYW5kIGlzIG5vdAogICAgb25lLiBNZWFzdXJlZCBvbiBNMzM6IGNlbnRyb2lkaW5nIHRoZSByYXcgc3RhbXAgbW92ZWQgdGhlIHBvc2l0aW9ucyAwLjYgcHggYW5kCiAgICBsZWZ0IHRoZSBzdGFja2VkIHN0YXIgZmxhdC10b3BwZWQsIGl0cyBicmlnaHRlc3QgcmluZyBhdCByYWRpdXMgNDsgdGFraW5nIHRoZQogICAgbG9jYWwgc2t5IG9mZiB0aGUgc3RhbXAgZmlyc3QgbW92ZWQgdGhlbSA0LjMgcHggYW5kIGdhdmUgYSBzdGFyIHRoYXQgZmFsbHMgZnJvbQogICAgMTAyIGNvdW50cyBhdCB0aGUgY2VudHJlIHRvIDE1IGF0IHJhZGl1cyA3LCB3aGljaCBpcyB3aGF0IGEgc3RhciBsb29rcyBsaWtlLgoKICAgIFRoZSBzdGFtcCBpcyBmbG9vcmVkIGF0IHplcm8gYWZ0ZXIgdGhlIHNreSBjb21lcyBvZmYgc28gdGhhdCBuZWdhdGl2ZSBub2lzZQogICAgY2Fubm90IHB1bGwgdGhlIGNlbnRyZSBhcm91bmQuCgogICAgUG9zaXRpb25zIHRoYXQgY2Fubm90IGJlIGNlbnRyZWQsIG9yIHRoYXQgbW92ZSBmdXJ0aGVyIHRoYW4gYSBwbGF0ZSBzb2x1dGlvbgogICAgcGxhdXNpYmx5IGVycnMsIGFyZSBkcm9wcGVkIHJhdGhlciB0aGFuIHRydXN0ZWQuCiAgICAiIiIKICAgIHBvc2l0aW9ucyA9IG5wLmFzYXJyYXkocG9zaXRpb25zLCBkdHlwZT1mbG9hdCkKICAgIGlmIG5vdCBsZW4ocG9zaXRpb25zKToKICAgICAgICByZXR1cm4gcG9zaXRpb25zLCB7Im5faW4iOiAwLCAibl9vdXQiOiAwfQoKICAgIG55LCBueCA9IGRhdGEuc2hhcGUKICAgIGhhbGYgPSBib3ggLy8gMgogICAgZWRnZSA9IGJveAogICAgaW5zaWRlID0gKChwb3NpdGlvbnNbOiwgMF0gPiBlZGdlKSAmIChwb3NpdGlvbnNbOiwgMF0gPCBueCAtIGVkZ2UpICYKICAgICAgICAgICAgICAocG9zaXRpb25zWzosIDFdID4gZWRnZSkgJiAocG9zaXRpb25zWzosIDFdIDwgbnkgLSBlZGdlKSkKICAgIHAgPSBwb3NpdGlvbnNbaW5zaWRlXQogICAgaWYgbm90IGxlbihwKToKICAgICAgICByZXR1cm4gbnAuZW1wdHkoKDAsIDIpKSwgeyJuX2luIjogbGVuKHBvc2l0aW9ucyksICJuX291dCI6IDAsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAid2h5IjogIm5vbmUgb2YgdGhlbSBmYWxsIGluc2lkZSB0aGUgZnJhbWUifQoKICAgIGZvdW5kID0gX2NlbnRyb2lkX2xvY2FsbHkoZGF0YSwgcCwgYm94KQoKICAgIGR4LCBkeSA9IGZvdW5kWzosIDBdIC0gcFs6LCAwXSwgZm91bmRbOiwgMV0gLSBwWzosIDFdCiAgICBtb3ZlZCA9IG5wLmh5cG90KGR4LCBkeSkKICAgIGdvb2QgPSBucC5pc2Zpbml0ZShtb3ZlZCkgJiAobW92ZWQgPCBtYXhfbW92ZV9weCkKICAgIG91dCA9IGZvdW5kW2dvb2RdCiAgICByZXR1cm4gb3V0LCB7Im5faW4iOiBpbnQobGVuKHBvc2l0aW9ucykpLCAibl9vdXQiOiBpbnQoZ29vZC5zdW0oKSksCiAgICAgICAgICAgICAgICAgIm1lZGlhbl9tb3ZlIjogZmxvYXQobnAubWVkaWFuKG1vdmVkW2dvb2RdKSkgaWYgZ29vZC5hbnkoKSBlbHNlIDAuMCwKICAgICAgICAgICAgICAgICAibWVkaWFuX2R4IjogZmxvYXQobnAubWVkaWFuKGR4W2dvb2RdKSkgaWYgZ29vZC5hbnkoKSBlbHNlIDAuMCwKICAgICAgICAgICAgICAgICAibWVkaWFuX2R5IjogZmxvYXQobnAubWVkaWFuKGR5W2dvb2RdKSkgaWYgZ29vZC5hbnkoKSBlbHNlIDAuMH0KCmRlZiBkZXNjcmliZV9jZW50cmluZyhkaWFnKToKICAgICIiIldoYXQgdGhlIGNlbnRyaW5nIG1vdmVkLCBpbiBhIGxpbmUuIiIiCiAgICBpZiBkaWFnLmdldCgid2h5Iik6CiAgICAgICAgcmV0dXJuICIgIGNhdGFsb2d1ZSBwb3NpdGlvbnMgbGVmdCBhcyB0aGV5IGFyZSAoe30pIi5mb3JtYXQoZGlhZ1sid2h5Il0pCiAgICByZXR1cm4gKCIgIHt9IG9mIHt9IGNhdGFsb2d1ZSBzdGFycyBmb3VuZCBhbmQgY2VudHJlZDsgdGhleSBzYXQgYSBtZWRpYW4gb2YgIgogICAgICAgICAgICAiezouMmZ9IHB4IGZyb20gdGhlIGNhdGFsb2d1ZSBwb3NpdGlvbiAoezorLjJmfSwgezorLjJmfSkiLmZvcm1hdCgKICAgICAgICAgICAgICAgIGRpYWdbIm5fb3V0Il0sIGRpYWdbIm5faW4iXSwgZGlhZ1sibWVkaWFuX21vdmUiXSwKICAgICAgICAgICAgICAgIGRpYWdbIm1lZGlhbl9keCJdLCBkaWFnWyJtZWRpYW5fZHkiXSkpCgpkZWYgY29tYmluZV9leHBvc3VyZXMocGF0aHMsIGRvbm9yX3BhdGg9Tm9uZSwgbGFiZWw9IiIsIGJhbmRfcm93cz01MTIsCiAgICAgICAgICAgICAgICAgICAgICBwb3NpdGlvbnM9Tm9uZSk6CiAgICAiIiJQdXQgc2V2ZXJhbCBleHBvc3VyZXMgb2Ygb25lIGZpbHRlciBvbnRvIG9uZSBncmlkIGFuZCB0YWtlIHRoZWlyIG1lZGlhbi4KCiAgICBXSFkgQ09NQklORSBBVCBBTEwKICAgIEEgbmlnaHQncyBvYnNlcnZpbmcgb2Ygb25lIGZpbHRlciBpcyBzZXZlcmFsIGV4cG9zdXJlcy4gVGFraW5nIG9uZSBhbmQKICAgIGRpc2NhcmRpbmcgdGhlIHJlc3QgdGhyb3dzIGF3YXkgbW9zdCBvZiB3aGF0IHdhcyBnYXRoZXJlZDogc2l4IGZyYW1lcyBjb21iaW5lZAogICAgaGF2ZSBub2lzZSBsb3dlciBieSByb3VnaGx5IHRoZSBzcXVhcmUgcm9vdCBvZiBzaXgsIGEgZmFjdG9yIG9mIDIuNCwgYW5kIGl0IGlzCiAgICB0aGUgZmFpbnRlc3QgcmVnaW9ucyAtIHRoZSBvbmVzIG5lYXIgdGhlIGRldGVjdGlvbiBsaW1pdCwgd2hlcmUgdGhlIGFuc3dlciBpcwogICAgYWN0dWFsbHkgZGVjaWRlZCAtIHRoYXQgYXBwZWFyIG9yIHZhbmlzaCBvbiBleGFjdGx5IHRoYXQgZGlmZmVyZW5jZS4KCiAgICBXSFkgQSBNRURJQU4KICAgIEJlY2F1c2UgaXQgaXMgcm9idXN0LCBjaGVhcCwgYW5kIHRoZSBhbHRlcm5hdGl2ZSBpcyBub3Qgd29ydGggd2hhdCBpdCBjb3N0cy4KICAgIE1lYXN1cmVkIG9uIHRoaXMgZGF0YToKCiAgICAgICAgc2l4IGZyYW1lcyAgICAgICAgbm9pc2UgICAgdGltZSAgICAgd29ya2luZyBtZW1vcnkKICAgICAgICBtZWRpYW4gICAgICAgICAgICAgMS4xNiAgICAwLjQ1IHMgICAgIDEuNHggdGhlIHN0YWNrCiAgICAgICAgY2xpcHBlZCBtZWFuICAgICAgIDEuMTAgICAgMC45NyBzICAgICAzLjN4IHRoZSBzdGFjawogICAgICAgIHBsYWluIGF2ZXJhZ2UgICAgICAyLjYwKiAgIDAuMDIgcyAgICAgMC4yeCB0aGUgc3RhY2sKICAgICAgICAgICAgICAgICAgICAgICAgICAgKCogd2l0aCBvbmUgY29zbWljIHJheSBpbiBhIGh1bmRyZWQgcGl4ZWxzKQoKICAgIFRoZSBjbGlwcGVkIG1lYW4gYnV5cyBmaXZlIHRvIHNldmVuIHBlcmNlbnQgaW4gbm9pc2UgLSBhYm91dCB3aGF0IGEgc2V2ZW50aAogICAgZXhwb3N1cmUgd291bGQgZ2l2ZSAtIGZvciB0d2ljZSB0aGUgdGltZSBhbmQgdHdvIGFuZCBhIGhhbGYgdGltZXMgdGhlIG1lbW9yeS4KICAgIEEgcGxhaW4gYXZlcmFnZSBpcyBub3QgYW4gb3B0aW9uIGF0IGFsbDogb25lIGNvc21pYyByYXkgcnVpbnMgdGhlIHBpeGVsLiBTbwogICAgdGhlIG1lZGlhbiBpdCBpcywgYW5kIHRoZXJlIGlzIG5vIHN3aXRjaCwgYmVjYXVzZSB0aGVyZSBpcyBubyBjYXNlIGhlcmUgd2hlcmUKICAgIHRoZSBvdGhlciBhbnN3ZXIgaXMgdGhlIHJpZ2h0IG9uZS4KCiAgICBEb25lIGEgYmFuZCBvZiByb3dzIGF0IGEgdGltZSwgc28gdGhlIG1lbW9yeSBuZWVkZWQgZG9lcyBub3QgZ3JvdyB3aXRoIHRoZQogICAgbnVtYmVyIG9mIGV4cG9zdXJlczogdHdlbnR5IGZyYW1lcyBjb21iaW5lIGluIHRoZSBzYW1lIHNwYWNlIGFzIHRocmVlLgoKICAgIFRoZSBmaXJzdCBleHBvc3VyZSBzZXRzIHRoZSBncmlkOyB0aGUgcmVzdCBhcmUgYnJvdWdodCB0byBpdCB0aHJvdWdoIHRoZWlyCiAgICBwbGF0ZSBzb2x1dGlvbnMgYW5kIHRoZW4gaGF2ZSB0aGUgbGVmdG92ZXIgc3ViLXBpeGVsIHNoaWZ0IG1lYXN1cmVkIGZyb20gdGhlCiAgICBzdGFycyBhbmQgdGFrZW4gb3V0LCBleGFjdGx5IGFzIHRoZSBzZXBhcmF0ZSBmaWx0ZXJzIGFyZS4KICAgICIiIgogICAgaW1wb3J0IG51bXB5IGFzIG5wCgogICAgcmVmZXJlbmNlLCByZWZfd2NzLCByZWZfYnBtLCByZWZfaW5mbyA9IGxvYWRfZnJhbWUocGF0aHNbMF0sIGRvbm9yX3BhdGgpCiAgICBpZiBsZW4ocGF0aHMpID09IDE6CiAgICAgICAgcmV0dXJuIHJlZmVyZW5jZSwgcmVmX3djcywgcmVmX2JwbSwgcmVmX2luZm8KCiAgICBwcmludCgiQ29tYmluaW5nIHt9IHt9IGV4cG9zdXJlcyIuZm9ybWF0KGxlbihwYXRocyksIGxhYmVsIG9yICJmcmFtZXMiKSkKICAgICMgRXZlcnkgZXhwb3N1cmUgaXMgc2NhbGVkIHRvIHRoZSBleHBvc3VyZSB0aW1lIG9mIHRoZSBmaXJzdCBiZWZvcmUgdGhlCiAgICAjIG1lZGlhbiBpcyB0YWtlbi4gQSBtZWRpYW4gb2YgYSA2MCBzIGFuZCBhIDEyMCBzIGZyYW1lIHRoYXQgd2VyZSBub3QKICAgICMgc2NhbGVkIGlzIG5laXRoZXIgb2YgdGhlbTogZWFjaCBwaXhlbCB0YWtlcyB3aGljaGV2ZXIgdmFsdWUgaGFwcGVucyB0bwogICAgIyBmYWxsIGluIHRoZSBtaWRkbGUuIEJlZm9yZSBzY2FsaW5nLCBwaXhlbHMgYXQgdGhlIGRldGVjdG9yJ3Mgc2F0dXJhdGlvbgogICAgIyBsZXZlbCBhcmUgYmxhbmtlZCwgc28gdGhhdCBhIHN0YXIgc2F0dXJhdGVkIGluIHRoZSBsb25nIGV4cG9zdXJlcyBpcwogICAgIyBtZWFzdXJlZCBmcm9tIHRoZSBzaG9ydCBvbmVzLgogICAgcmVmX2V4cHRpbWUgPSBmbG9hdChyZWZfaW5mby5nZXQoImV4cHRpbWUiKSBvciAxLjApCiAgICBzYXRfbGV2ZWxzLCBzYXRfbWFza3MgPSBbXSwgW10KCiAgICBkZWYgc2NhbGVkKGZyYW1lLCBpbmZvLCBuYW1lKToKICAgICAgICBleHB0aW1lID0gZmxvYXQoaW5mby5nZXQoImV4cHRpbWUiKSBvciByZWZfZXhwdGltZSkKICAgICAgICByYXRpbyA9IHJlZl9leHB0aW1lIC8gZXhwdGltZQogICAgICAgIHNhdCA9IGluZm8uZ2V0KCJzYXR1cmF0ZSIpCiAgICAgICAgaWYgc2F0OgogICAgICAgICAgICBmcmFtZSA9IG5wLmFycmF5KGZyYW1lLCBkdHlwZT1ucC5mbG9hdDMyLCBjb3B5PVRydWUpCiAgICAgICAgICAgIHdpdGggbnAuZXJyc3RhdGUoaW52YWxpZD0iaWdub3JlIik6CiAgICAgICAgICAgICAgICBoaXQgPSBmcmFtZSA+PSBTQVRVUkFUSU9OX0ZSQUNUSU9OICogc2F0CiAgICAgICAgICAgIGZyYW1lW2hpdF0gPSBucC5uYW4KICAgICAgICAgICAgc2F0X21hc2tzLmFwcGVuZChoaXQpCiAgICAgICAgICAgIHNhdF9sZXZlbHMuYXBwZW5kKHNhdCAqIHJhdGlvKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIHNhdF9sZXZlbHMuYXBwZW5kKE5vbmUpCiAgICAgICAgaWYgYWJzKHJhdGlvIC0gMS4wKSA+IDFlLTY6CiAgICAgICAgICAgIHByaW50KCIgIHt9OiB7Oi4wZn0gcywgc2NhbGVkIGJ5IHs6LjNmfSB0byB7Oi4wZn0gcyIuZm9ybWF0KAogICAgICAgICAgICAgICAgbmFtZSwgZXhwdGltZSwgcmF0aW8sIHJlZl9leHB0aW1lKSkKICAgICAgICAgICAgZnJhbWUgPSBmcmFtZSAqIG5wLmZsb2F0MzIocmF0aW8pCiAgICAgICAgcmV0dXJuIGZyYW1lCgogICAgc3RhY2sgPSBbc2NhbGVkKHJlZmVyZW5jZSwgcmVmX2luZm8sIG9zLnBhdGguYmFzZW5hbWUocGF0aHNbMF0pKV0KICAgIHRvdGFsX2V4cHRpbWUgPSBmbG9hdChyZWZfaW5mby5nZXQoImV4cHRpbWUiKSBvciAwLjApCgogICAgZm9yIHBhdGggaW4gcGF0aHNbMTpdOgogICAgICAgIGRhdGEsIHdjcywgXywgaW5mbyA9IGxvYWRfZnJhbWUocGF0aCwgZG9ub3JfcGF0aCkKICAgICAgICBuYW1lID0gb3MucGF0aC5iYXNlbmFtZShwYXRoKQoKICAgICAgICAjIEZvdXIgYnJpZ2h0LCB3aWRlbHkgc2VwYXJhdGVkIHN0YXJzIGZpeCB0aGUgc2hpZnQsIHRoZSByb3RhdGlvbiBhbmQgdGhlCiAgICAgICAgIyBzY2FsZSBiZXR3ZWVuIHR3byBmcmFtZXMgb2YgdGhlIHNhbWUgZmllbGQuIEdvaW5nIHRocm91Z2ggYm90aCBwbGF0ZQogICAgICAgICMgc29sdXRpb25zIGluc3RlYWQgY29zdHMgYSBtaW51dGUgYSBmcmFtZSBhbmQgYXJyaXZlcyBhdCB0aGUgc2FtZSBwbGFjZS4KICAgICAgICBwbGFjZWQgPSBGYWxzZQogICAgICAgIGlmIHBvc2l0aW9ucyBpcyBub3QgTm9uZSBhbmQgbGVuKHBvc2l0aW9ucyk6CiAgICAgICAgICAgIHRyYW5zZm9ybSwgYWRpYWcgPSBhbmNob3JfdHJhbnNmb3JtKHJlZmVyZW5jZSwgZGF0YSwgcG9zaXRpb25zKQogICAgICAgICAgICBpZiB0cmFuc2Zvcm0gaXMgbm90IE5vbmU6CiAgICAgICAgICAgICAgICBkYXRhID0gYXBwbHlfdHJhbnNmb3JtKGRhdGEsIHRyYW5zZm9ybSwgcmVmZXJlbmNlLnNoYXBlKQogICAgICAgICAgICAgICAgcHJpbnQoIiAgIiArIG5hbWUpCiAgICAgICAgICAgICAgICBwcmludCgiICAiICsgZGVzY3JpYmVfYW5jaG9ycyhhZGlhZykpCiAgICAgICAgICAgICAgICBwbGFjZWQgPSBUcnVlCgogICAgICAgIGlmIG5vdCBwbGFjZWQ6CiAgICAgICAgICAgIGlmIHdjcyBpcyBub3QgTm9uZSBhbmQgcmVmX3djcyBpcyBub3QgTm9uZToKICAgICAgICAgICAgICAgIGRhdGEgPSBhbGlnbl90byhyZWZfd2NzLCByZWZlcmVuY2Uuc2hhcGUsIGRhdGEsIHdjcykKICAgICAgICAgICAgZWxpZiBkYXRhLnNoYXBlICE9IHJlZmVyZW5jZS5zaGFwZToKICAgICAgICAgICAgICAgIHByaW50KCIgIHNraXBwaW5nIHt9IC0gbm8gcGxhdGUgc29sdXRpb24gYW5kIGEgZGlmZmVyZW50ICIKICAgICAgICAgICAgICAgICAgICAgICJzaGFwZSIuZm9ybWF0KG5hbWUpKQogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgZGF0YSA9IHJlZmluZV9hbGlnbm1lbnQocmVmZXJlbmNlLCBkYXRhLCAiICAiICsgbmFtZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcG9zaXRpb25zPXBvc2l0aW9ucykKCiAgICAgICAgc3RhY2suYXBwZW5kKHNjYWxlZChkYXRhLCBpbmZvLCBuYW1lKSkKICAgICAgICB0b3RhbF9leHB0aW1lICs9IGZsb2F0KGluZm8uZ2V0KCJleHB0aW1lIikgb3IgMC4wKQoKICAgIG55LCBueCA9IHJlZmVyZW5jZS5zaGFwZQogICAgY29tYmluZWQgPSBucC5lbXB0eSgobnksIG54KSwgZHR5cGU9bnAuZmxvYXQzMikKICAgIGZvciB5MCBpbiByYW5nZSgwLCBueSwgYmFuZF9yb3dzKToKICAgICAgICB5MSA9IG1pbih5MCArIGJhbmRfcm93cywgbnkpCiAgICAgICAgYmFuZCA9IG5wLnN0YWNrKFtmcmFtZVt5MDp5MV0gZm9yIGZyYW1lIGluIHN0YWNrXSwgYXhpcz0wKQogICAgICAgIHdpdGggd2FybmluZ3MuY2F0Y2hfd2FybmluZ3MoKToKICAgICAgICAgICAgd2FybmluZ3Muc2ltcGxlZmlsdGVyKCJpZ25vcmUiKQogICAgICAgICAgICBjb21iaW5lZFt5MDp5MV0gPSBucC5uYW5tZWRpYW4oYmFuZCwgYXhpcz0wKQogICAgICAgICAgICAjIEEgcGl4ZWwgc2F0dXJhdGVkIGluIGV2ZXJ5IGV4cG9zdXJlIGhhcyBub3RoaW5nIGxlZnQgdG8gdGFrZSB0aGUKICAgICAgICAgICAgIyBtZWRpYW4gb2YuIEl0IGlzIHNldCB0byB0aGUgc2F0dXJhdGlvbiBsZXZlbCwgc28gdGhhdCBpdCBzdGF5cyBhCiAgICAgICAgICAgICMgYnJpZ2h0IHBpeGVsIHRoZSBwaG90b21ldHJ5IGZsYWdzIGFzIHNhdHVyYXRlZCwgbm90IGEgaG9sZS4KICAgICAgICAgICAgaWYgc2F0X21hc2tzOgogICAgICAgICAgICAgICAgaG9sZSA9IH5ucC5pc2Zpbml0ZShjb21iaW5lZFt5MDp5MV0pCiAgICAgICAgICAgICAgICBob2xlICY9IG5wLmFueShucC5zdGFjayhbbVt5MDp5MV0gZm9yIG0gaW4gc2F0X21hc2tzXSwgYXhpcz0wKSwgYXhpcz0wKQogICAgICAgICAgICAgICAgaWYgaG9sZS5hbnkoKToKICAgICAgICAgICAgICAgICAgICBjb21iaW5lZFt5MDp5MV1baG9sZV0gPSBtYXgocyBmb3IgcyBpbiBzYXRfbGV2ZWxzIGlmIHMpCgogICAgcHJpbnQoIiAgbWVkaWFuIG9mIHt9IGV4cG9zdXJlcywgdG90YWwgZXhwb3N1cmUgezouMGZ9cywgbm9pc2UgbG93ZXIgYnkgYWJvdXQgIgogICAgICAgICAgImEgZmFjdG9yIG9mIHs6LjFmfSIuZm9ybWF0KGxlbihzdGFjayksIHRvdGFsX2V4cHRpbWUsIGxlbihzdGFjaykgKiogMC41KSkKCiAgICAjIFRoZSBtZWRpYW4gc2l0cyBhdCB0aGUgbGV2ZWwgb2YgT05FIGV4cG9zdXJlIG9mIHRoZSBmaXJzdCBmcmFtZSdzIGxlbmd0aCwKICAgICMgc28gdGhhdCBpcyB3aGF0IEVYUFRJTUUgc2F5cy4gSXQgdXNlZCB0byBzYXkgdGhlIHRvdGFsLCB3aGljaCBtYWRlIHRoZQogICAgIyBoZWFkZXIgY2xhaW0gc2V2ZXJhbCB0aW1lcyB0aGUgZXhwb3N1cmUgdGhlIHBpeGVsIHZhbHVlcyBoYXZlLiBUaGUgdG90YWwKICAgICMgaXMga2VwdCBpbiBUT1RFWFAsIGFuZCBOQ09NQklORSBzYXlzIGhvdyBtYW55IGZyYW1lcyB3ZW50IGluIC0gdGhlIG5vaXNlCiAgICAjIG9mIGEgbWVkaWFuIG9mIE4gZnJhbWVzIGlzIHRoYXQgb2Ygb25lIGZyYW1lIHdpdGggTiB4IDIvcGkgdGltZXMgdGhlIGdhaW4uCiAgICBpbmZvID0gZGljdChyZWZfaW5mbykKICAgIGluZm9bImV4cHRpbWUiXSA9IHJlZl9leHB0aW1lCiAgICBpbmZvWyJ0b3RhbF9leHB0aW1lIl0gPSB0b3RhbF9leHB0aW1lCiAgICBpbmZvWyJuY29tYmluZSJdID0gbGVuKHN0YWNrKQogICAgaW5mb1siY29tYnR5cGUiXSA9ICJtZWRpYW4iCiAgICBpZiBhbGwocyBmb3IgcyBpbiBzYXRfbGV2ZWxzKToKICAgICAgICAjIFBpeGVscyBiZWxvdyB0aGUgbGV2ZWwgb2YgdGhlIHNob3J0ZXN0IGV4cG9zdXJlIGNhbWUgZnJvbSBhdCBsZWFzdAogICAgICAgICMgb25lIHVuc2F0dXJhdGVkIGZyYW1lLgogICAgICAgIGluZm9bInNhdHVyYXRlIl0gPSBtYXgoc2F0X2xldmVscykKICAgIGVsc2U6CiAgICAgICAgaW5mb1sic2F0dXJhdGUiXSA9IE5vbmUKICAgICMgU2NhbGluZyBhIGxvbmdlciBleHBvc3VyZSBkb3duIGxvd2VycyBpdHMgbm9pc2UgYWxvbmcgd2l0aCBpdHMgY291bnRzLAogICAgIyBhcyBhIGhpZ2hlciBnYWluIHdvdWxkLiBXcml0dGVuIHNvIHRoYXQgR0FJTiB4IE5DT01CSU5FIGlzIHRoZSBnYWluIG9mIHRoZQogICAgIyB0b3RhbCBleHBvc3VyZSBleHByZXNzZWQgYXQgdGhpcyBmcmFtZSdzIGxldmVsLgogICAgaWYgaW5mby5nZXQoImdhaW4iKToKICAgICAgICBpbmZvWyJnYWluIl0gPSBmbG9hdChpbmZvWyJnYWluIl0pICogdG90YWxfZXhwdGltZSAvIHJlZl9leHB0aW1lIC8gbGVuKHN0YWNrKQogICAgcmV0dXJuIGNvbWJpbmVkLCByZWZfd2NzLCByZWZfYnBtLCBpbmZvCgpTQVRVUkFUSU9OX0ZSQUNUSU9OID0gMC44ICAgIyBhIHBpeGVsIGFib3ZlIHRoaXMgZnJhY3Rpb24gb2YgU0FUVVJBVEUgY291bnRzIGFzIHNhdHVyYXRlZAoKZGVmIG5ld19ydW5fZm9sZGVyKGJhc2UsIHRvb2wsIGdhbGF4eSk6CiAgICAiIiJBIGZvbGRlciBvZiBpdHMgb3duIGZvciB0aGlzIHJ1bjogcmVzdWx0Xzx0b29sPl88Z2FsYXh5Pl88ZGF0ZT5fPHRpbWU+LgoKICAgIEV2ZXJ5IHJ1biB1c2VkIHRvIHdyaXRlIGludG8gdGhlIHNhbWUgcGxhY2UgdW5kZXIgdGhlIHNhbWUgbmFtZXMsIHNvIGEKICAgIHNlY29uZCBydW4gb3Zlcndyb3RlIHRoZSBmaXJzdCwgYW5kIGEgcnVuIHRoYXQgc3RvcHBlZCBoYWxmIHdheSBsZWZ0IHRoZQogICAgcHJldmlvdXMgcnVuJ3MgZmlsZXMgc2l0dGluZyBiZXNpZGUgaXRzIG93biwgbG9va2luZyBjdXJyZW50LiBBIG5ldyBmb2xkZXIKICAgIGVhY2ggdGltZSBtZWFucyBub3RoaW5nIGEgcnVuIHJlYWRzIG9yIHNob3dzIGNhbiBiZSBsZWZ0IG92ZXIgZnJvbSBhbm90aGVyLgogICAgIiIiCiAgICBpbXBvcnQgZGF0ZXRpbWUKICAgIG5hbWUgPSByZS5zdWIociJbXkEtWmEtejAtOV8uLV0rIiwgIl8iLCAoZ2FsYXh5IG9yICJnYWxheHkiKS5zdHJpcCgpKSBvciAiZ2FsYXh5IgogICAgc3RhbXAgPSBkYXRldGltZS5kYXRldGltZS5ub3coKS5zdHJmdGltZSgiJVklbSVkXyVIJU0lUyIpCiAgICBwYXRoID0gb3MucGF0aC5qb2luKGJhc2Ugb3Igb3MuZ2V0Y3dkKCksICJyZXN1bHRfe31fe31fe30iLmZvcm1hdCh0b29sLCBuYW1lLCBzdGFtcCkpCiAgICBvcy5tYWtlZGlycyhwYXRoLCBleGlzdF9vaz1UcnVlKQogICAgcmV0dXJuIHBhdGgKCmRlZiBvdXRwdXRfcHJlZml4KGdpdmVuLCBmYWxsYmFja19kaXIsIGdhbGF4eSk6CiAgICAiIiJUdXJuIHdoYXQgdGhlIHBlcnNvbiB0eXBlZCBpbnRvIGEgc3RlbSBmb3IgdGhlIG91dHB1dCBmaWxlcy4KCiAgICBBIGZvbGRlciBpcyBhY2NlcHRlZCBhcyB3ZWxsIGFzIGEgc3RlbSwgYmVjYXVzZSBhIGZvbGRlciBpcyB3aGF0IHBlb3BsZSB0eXBlOgogICAgYXNrZWQgd2hlcmUgdG8gcHV0IHRoZSBvdXRwdXQsIHRocmVlIHJ1bnMgaW4gYSByb3cgd2VyZSBnaXZlbiB0aGUgcGF0aCBvZiB0aGUKICAgIGZyYW1lcycgb3duIGZvbGRlciwgYW5kIHRoZSBmaWxlcyBsYW5kZWQgYmVzaWRlIGl0IG5hbWVkIGFmdGVyIGl0LiBJZiB3aGF0IHdhcwogICAgZ2l2ZW4gaXMgYSBkaXJlY3RvcnksIHRoZSBmaWxlcyBnbyBpbnNpZGUgaXQsIG5hbWVkIGFmdGVyIHRoZSBnYWxheHkuCiAgICAiIiIKICAgIG5hbWUgPSAoZ2FsYXh5IG9yICJnYWxheHkiKS5yZXBsYWNlKCIgIiwgIl8iKQogICAgaWYgZ2l2ZW46CiAgICAgICAgZ2l2ZW4gPSBzdHIoZ2l2ZW4pLnN0cmlwKCkuc3RyaXAoJyInKS5zdHJpcCgiJyIpCiAgICAgICAgaWYgb3MucGF0aC5pc2RpcihnaXZlbik6CiAgICAgICAgICAgIHJldHVybiBvcy5wYXRoLmpvaW4oZ2l2ZW4sIG5hbWUpCiAgICAgICAgcGFyZW50ID0gb3MucGF0aC5kaXJuYW1lKGdpdmVuKQogICAgICAgIGlmIHBhcmVudCBhbmQgbm90IG9zLnBhdGguaXNkaXIocGFyZW50KToKICAgICAgICAgICAgb3MubWFrZWRpcnMocGFyZW50LCBleGlzdF9vaz1UcnVlKQogICAgICAgIHJldHVybiBnaXZlbgogICAgcmV0dXJuIG9zLnBhdGguam9pbihmYWxsYmFja19kaXIgb3IgIi4iLCBuYW1lKQoKZGVmIHNhdmVfZnJhbWUocGF0aCwgZGF0YSwgd2NzPU5vbmUsIGluZm89Tm9uZSwgaGlzdG9yeT0oKSk6CiAgICAiIiJXcml0ZSBhIGZyYW1lIHRoaXMgdG9vbCBtYWRlLCB3aXRoIGEgaGVhZGVyIHdvcnRoIGhhdmluZy4KCiAgICBUaGUgY29udGludXVtLXN1YnRyYWN0ZWQgaW1hZ2UgaXMgdGhlIG9uZSB0aGUgd2hvbGUgbWVhc3VyZW1lbnQgcmVzdHMgb24sIGFuZAogICAgdW50aWwgbm93IGl0IGV4aXN0ZWQgb25seSBpbnNpZGUgYSBwbG90LiBBbnlvbmUgd2FudGluZyB0byBjaGVjayB0aGUKICAgIHN1YnRyYWN0aW9uIGJ5IGV5ZSAtIHdoaWNoIGlzIGV4YWN0bHkgd2hhdCB0aGUgdG9vbCBhc2tzIGZvciB3aGVuIHRoZSBzdGFycwogICAgZGlzYWdyZWUgYWJvdXQgdGhlIGZhY3RvciAtIGhhZCBubyBmaWxlIHRvIG9wZW4uIEl0IGNhcnJpZXMgdGhlIHBsYXRlIHNvbHV0aW9uCiAgICBzbyBpdCBsaW5lcyB1cCB3aXRoIHRoZSBvcmlnaW5hbHMgaW4gYW55IHZpZXdlciwgYW5kIGEgaGlzdG9yeSBzYXlpbmcgd2hhdCBpdAogICAgaXMgYW5kIGhvdyBpdCB3YXMgbWFkZS4KICAgIFdIQVQgSVMgRE9ORSBUTyBUSEUgUElYRUxTIEJFRk9SRSBUSEVZIEFSRSBXUklUVEVOCiAgICBCbGFuayBwaXhlbHMgYXJlIHdyaXR0ZW4gYXMgemVybyByYXRoZXIgdGhhbiBhcyBOYU4uIE5hTiBpcyBsZWdhbCBGSVRTIGFuZAogICAgYXN0cm9weSByZWFkcyBpdCBoYXBwaWx5LCBidXQgaXQgaXMgYSBmbG9hdGluZy1wb2ludCB2YWx1ZSB0aGF0IG9sZGVyIHJlYWRlcnMKICAgIHByZWRhdGU6IEFJUDRXaW4gd291bGQgbm90IG9wZW4gYSBmcmFtZSB3cml0dGVuIGZyb20gaGVyZSwgYW5kIGEgc3VidHJhY3RlZAogICAgTTgzIGZyYW1lIGNhcnJpZWQgbmluZSB0aG91c2FuZCBvZiB0aGVtLCBhdCB0aGUgZWRnZXMgdGhlIHJlc2FtcGxpbmcgY291bGQgbm90CiAgICByZWFjaC4gQSBmaWxlIHRoYXQgb25seSB0aGlzIHBpcGVsaW5lIGNhbiBvcGVuIGlzIG5vdCBtdWNoIG9mIGFuIG91dHB1dC4KCiAgICBXSFkgVEhFIFBMQVRFIFNPTFVUSU9OIElTIFdSSVRURU4gVFdJQ0UKICAgIGFzdHJvcHkgd3JpdGVzIGEgcm90YXRpb24gYXMgYSBQQyBtYXRyaXggd2l0aCBhIHNlcGFyYXRlIENERUxUIHNjYWxlLiBUaGF0IGlzCiAgICB0aGUgY3VycmVudCBjb252ZW50aW9uOyB0aGUgb2xkZXIgb25lIGlzIGEgc2luZ2xlIENEIG1hdHJpeCwgYW5kIGEgcmVhZGVyIHRoYXQKICAgIGtub3dzIG9ubHkgQ0Qgc2VlcyBhIGZyYW1lIHdpdGggbm8gdXNhYmxlIGNvb3JkaW5hdGVzLiBCb3RoIGFyZSB3cml0dGVuIC0gdGhleQogICAgc2F5IHRoZSBzYW1lIHRoaW5nLCBhbmQgQ0QgPSBQQyB4IENERUxUIGlzIGV4YWN0IC0gc28gZWl0aGVyIGtpbmQgb2YgcmVhZGVyCiAgICBmaW5kcyB3aGF0IGl0IGV4cGVjdHMuCiAgICAiIiIKICAgIGZyb20gYXN0cm9weS5pbyBpbXBvcnQgZml0cwoKICAgIGhlYWRlciA9IHdjcy50b19oZWFkZXIoKSBpZiB3Y3MgaXMgbm90IE5vbmUgZWxzZSBmaXRzLkhlYWRlcigpCiAgICBpZiB3Y3MgaXMgbm90IE5vbmU6CiAgICAgICAgdHJ5OgogICAgICAgICAgICBjZCA9IG5wLmFzYXJyYXkod2NzLnBpeGVsX3NjYWxlX21hdHJpeCwgZHR5cGU9ZmxvYXQpCiAgICAgICAgICAgIGZvciBpIGluICgxLCAyKToKICAgICAgICAgICAgICAgIGZvciBqIGluICgxLCAyKToKICAgICAgICAgICAgICAgICAgICBoZWFkZXJbIkNEe31fe30iLmZvcm1hdChpLCBqKV0gPSBmbG9hdChjZFtpIC0gMSwgaiAtIDFdKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHBhc3MKICAgIGlmIGluZm86CiAgICAgICAgaWYgaW5mby5nZXQoImZpbHRlciIpOgogICAgICAgICAgICBoZWFkZXJbIkZJTFRFUiJdID0gaW5mb1siZmlsdGVyIl0KICAgICAgICBpZiBpbmZvLmdldCgiZXhwdGltZSIpOgogICAgICAgICAgICBoZWFkZXJbIkVYUFRJTUUiXSA9IGZsb2F0KGluZm9bImV4cHRpbWUiXSkKICAgICAgICBpZiBpbmZvLmdldCgib2JqZWN0Iik6CiAgICAgICAgICAgIGhlYWRlclsiT0JKRUNUIl0gPSBpbmZvWyJvYmplY3QiXQogICAgICAgIGlmIGluZm8uZ2V0KCJwaXhlbF9zY2FsZSIpOgogICAgICAgICAgICBoZWFkZXJbIlBJWFNDQUxFIl0gPSBmbG9hdChpbmZvWyJwaXhlbF9zY2FsZSJdKQogICAgICAgIGZvciBrZXksIHdvcmQgaW4gKCgic2F0dXJhdGUiLCAiU0FUVVJBVEUiKSwgKCJnYWluIiwgIkdBSU4iKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAoInJkbm9pc2UiLCAiUkROT0lTRSIpLCAoInRvdGFsX2V4cHRpbWUiLCAiVE9URVhQIikpOgogICAgICAgICAgICBpZiBpbmZvLmdldChrZXkpOgogICAgICAgICAgICAgICAgaGVhZGVyW3dvcmRdID0gZmxvYXQoaW5mb1trZXldKQogICAgICAgIGlmIChpbmZvLmdldCgibmNvbWJpbmUiKSBvciAxKSA+IDE6CiAgICAgICAgICAgIGhlYWRlclsiTkNPTUJJTkUiXSA9IGludChpbmZvWyJuY29tYmluZSJdKQogICAgICAgICAgICBoZWFkZXJbIkNPTUJUWVBFIl0gPSBzdHIoaW5mby5nZXQoImNvbWJ0eXBlIikgb3IgIm1lZGlhbiIpCiAgICAgICAgaWYgaW5mby5nZXQoImRhdGUiKToKICAgICAgICAgICAgaGVhZGVyWyJEQVRFLU9CUyJdID0gc3RyKGluZm9bImRhdGUiXSkKICAgIHBpeGVscyA9IG5wLmFzYXJyYXkoZGF0YSwgZHR5cGU9ImZsb2F0MzIiKQogICAgYmxhbmsgPSB+bnAuaXNmaW5pdGUocGl4ZWxzKQogICAgaWYgYmxhbmsuYW55KCk6CiAgICAgICAgcGl4ZWxzID0gbnAud2hlcmUoYmxhbmssIG5wLmZsb2F0MzIoMC4wKSwgcGl4ZWxzKQogICAgICAgIGhlYWRlclsiSElTVE9SWSJdID0gKCJ7fSBibGFuayBwaXhlbHMgd3JpdHRlbiBhcyB6ZXJvLCBub3QgTmFOLCBzbyB0aGF0ICIKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAib2xkZXIgcmVhZGVycyBjYW4gb3BlbiB0aGlzIi5mb3JtYXQoaW50KGJsYW5rLnN1bSgpKSkpCiAgICBmb3IgbGluZSBpbiBoaXN0b3J5OgogICAgICAgIGhlYWRlclsiSElTVE9SWSJdID0gbGluZQoKICAgIGZpdHMuUHJpbWFyeUhEVShkYXRhPXBpeGVscywgaGVhZGVyPWhlYWRlcikud3JpdGV0byhwYXRoLCBvdmVyd3JpdGU9VHJ1ZSkKICAgIHJldHVybiBwYXRoCgpkZWYgaW1hZ2Vfb3JpZW50YXRpb24od2NzKToKICAgICIiIkFuZ2xlIG9mIE5vcnRoIGluIHRoZSBpbWFnZSwgYW5kIHdoZXRoZXIgdGhlIHBhcml0eSBpcyBmbGlwcGVkLgoKICAgIFJlYWRpbmcgdGhpcyBmcm9tIHRoZSBXQ1MgcmVtb3ZlcyBhIHdob2xlIGNsYXNzIG9mIHNpbGVudCBlcnJvci4gQXNzdW1pbmcKICAgIG5vcnRoIGlzIHVwIHdoZW4gaXQgaXMgbm90IHJvdGF0ZXMgdGhlIGRlcHJvamVjdGlvbiBieSBhbiB1bmtub3duIGFtb3VudCwgYW5kCiAgICBub3RoaW5nIGluIHRoZSBvdXRwdXQgbG9va3Mgd3JvbmcuCiAgICAiIiIKICAgIGNkID0gd2NzLnBpeGVsX3NjYWxlX21hdHJpeAogICAgbm9ydGggPSBtYXRoLmRlZ3JlZXMobWF0aC5hdGFuMihjZFsxLCAxXSwgY2RbMCwgMV0pKSAlIDM2MC4wCiAgICBlYXN0ID0gbWF0aC5kZWdyZWVzKG1hdGguYXRhbjIoY2RbMSwgMF0sIGNkWzAsIDBdKSkgJSAzNjAuMAogICAgbWlycm9yZWQgPSAoKGVhc3QgLSBub3J0aCkgJSAzNjAuMCkgPiAxODAuMAogICAgcmV0dXJuIG5vcnRoLCBtaXJyb3JlZAoKZGVmIGZyYW1lX2dlb21ldHJ5KHBhdGgsIGRvbm9yX3BhdGg9Tm9uZSk6CiAgICAiIiJXaGVyZSBhIGZyYW1lIHBvaW50cyBhbmQgaG93IGJpZyBpdCBpcywgd2l0aG91dCByZWFkaW5nIGl0cyBwaXhlbHMuCgogICAgVGhlIGNhdGFsb2d1ZSBsb29rdXBzIGFsbCBuZWVkIHRoaXMgLSB0aGUgZ2FsYXh5J3MgY2VudHJlIGluIHRoaXMgZnJhbWUsIHRoZQogICAgc3RhcnMgaW4gZnJvbnQgb2YgaXQsIHRoZSBpc29waG90YWwgcmFkaXVzIGluIHBpeGVscyAtIGFuZCB0aGV5IGFyZSBuZWVkZWQKICAgIGJlZm9yZSB0aGUgZnJhbWVzIGFyZSBjb21iaW5lZCwgbm90IGFmdGVyLiBSZWFkaW5nIGEgaGVhZGVyIGlzIGluc3RhbnQsIHdoaWxlCiAgICByZWFkaW5nIGEgNjEtbWVnYXBpeGVsIGltYWdlIGlzIG5vdCwgc28gdGhlIGhlYWRlciBpcyB0cmllZCBmaXJzdC4gQSBmcmFtZQogICAgdGhhdCBsb3N0IGl0cyBwbGF0ZSBzb2x1dGlvbiBpbiBhIGNvbWJpbmluZyBzdGVwIGhhcyB0byBiZSBvcGVuZWQgcHJvcGVybHksCiAgICBzaW5jZSBpdHMgc29sdXRpb24gaXMgZml0dGVkIGZyb20gYSBkb25vciBleHBvc3VyZS4KCiAgICBSZXR1cm5zICh3Y3MsIHNoYXBlLCBpbmZvKS4KICAgICIiIgogICAgZnJvbSBhc3Ryb3B5LmlvIGltcG9ydCBmaXRzCiAgICBmcm9tIGFzdHJvcHkud2NzIGltcG9ydCBXQ1MKCiAgICB0cnk6CiAgICAgICAgd2l0aCBmaXRzLm9wZW4ocGF0aCkgYXMgaGR1bDoKICAgICAgICAgICAgZm9yIGhkdSBpbiBoZHVsOgogICAgICAgICAgICAgICAgaGVhZGVyID0gaGR1LmhlYWRlcgogICAgICAgICAgICAgICAgaWYgaGVhZGVyLmdldCgiTkFYSVMiKSA9PSAyOgogICAgICAgICAgICAgICAgICAgIHcgPSBXQ1MoaGVhZGVyKQogICAgICAgICAgICAgICAgICAgIGlmIHcuaGFzX2NlbGVzdGlhbDoKICAgICAgICAgICAgICAgICAgICAgICAgcmV0dXJuICh3LmNlbGVzdGlhbCwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAoaGVhZGVyWyJOQVhJUzIiXSwgaGVhZGVyWyJOQVhJUzEiXSksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgeyJwaXhlbF9zY2FsZSI6IGhlYWRlci5nZXQoIlBJWFNDQUxFIiwgTm9uZSksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICJmaWx0ZXIiOiBzdHIoaGVhZGVyLmdldCgiRklMVEVSIiwgIj8iKSksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICJvYmplY3QiOiBzdHIoaGVhZGVyLmdldCgiT0JKRUNUIiwgIj8iKSl9KQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICBwYXNzCgogICAgZGF0YSwgd2NzLCBfLCBpbmZvID0gbG9hZF9mcmFtZShwYXRoLCBkb25vcl9wYXRoKQogICAgcmV0dXJuIHdjcywgZGF0YS5zaGFwZSwgaW5mbwoKZGVmIHNhbWVfZ3JpZCh3Y3NfYSwgd2NzX2IsIHNoYXBlLCB0b2xlcmFuY2VfcHg9MS41KToKICAgICIiIlRydWUgd2hlbiB0d28gcGxhdGUgc29sdXRpb25zIHB1dCB0aGUgc2t5IGluIHRoZSBzYW1lIHBpeGVscyBhbHJlYWR5LgoKICAgIFdIWSBBU0sKICAgIFJlc2FtcGxpbmcgYSBmcmFtZSBpcyB0aGUgbW9zdCBleHBlbnNpdmUgdGhpbmcgdGhpcyB0b29sIGRvZXMgdG8gaXQ6IHR3bwogICAgYXN0cm9tZXRyaWMgdHJhbnNmb3JtcyBvZiBldmVyeSBwaXhlbCBwb3NpdGlvbiwgdGVucyBvZiBtaWxsaW9ucyBvZiB0aGVtLiBJdAogICAgaXMgd29ydGggaXQgd2hlbiB0aGUgZnJhbWVzIHJlYWxseSBzaXQgb24gZGlmZmVyZW50IGdyaWRzLiBJdCBpcyBwdXJlIHdhc3RlCiAgICB3aGVuIHRoZXkgZG8gbm90IC0gYW5kIGJldHdlZW4gY29uc2VjdXRpdmUgZXhwb3N1cmVzIG9mIG9uZSBmaWx0ZXIsIHRoZXkKICAgIGFsbW9zdCBuZXZlciBkby4gTWVhc3VyZWQgb24gYSBuaWdodCBvZiBNMzM6IHRoZSBleHBvc3VyZXMgb2YgYSBmaWx0ZXIgc2F0CiAgICB3aXRoaW4gMC4wNSB0byAwLjcgb2YgYSBwaXhlbCBvZiBlYWNoIG90aGVyLgoKICAgIFNvIHRoZSBxdWVzdGlvbiBpcyBhc2tlZCBiZWZvcmUgdGhlIHdvcmsgaXMgZG9uZS4gQSBmZXcgY29ybmVyIGFuZCBjZW50cmUKICAgIHBvaW50cyBhcmUgcHVzaGVkIHRocm91Z2ggYm90aCBzb2x1dGlvbnM7IGlmIG5vbmUgb2YgdGhlbSBtb3ZlcyBieSBtb3JlIHRoYW4KICAgIGFib3V0IGEgcGl4ZWwsIHRoZSBncmlkcyBhcmUgdGhlIHNhbWUgYW5kIGEgc3ViLXBpeGVsIHNoaWZ0IGlzIGFsbCB0aGF0IGlzCiAgICBuZWVkZWQuIFRoZSBmdWxsIHJlc2FtcGxpbmcgc3RheXMgZm9yIHRoZSBjYXNlIGl0IHdhcyB3cml0dGVuIGZvci4KICAgICIiIgogICAgaW1wb3J0IG51bXB5IGFzIG5wCgogICAgaWYgd2NzX2EgaXMgTm9uZSBvciB3Y3NfYiBpcyBOb25lOgogICAgICAgIHJldHVybiBGYWxzZQogICAgbnksIG54ID0gc2hhcGUKICAgIHhzID0gbnAuYXJyYXkoWzAsIG54IC0gMSwgMCwgbnggLSAxLCBueCAvLyAyLCBueCAvLyA0LCAzICogbnggLy8gNF0sCiAgICAgICAgICAgICAgICAgIGR0eXBlPWZsb2F0KQogICAgeXMgPSBucC5hcnJheShbMCwgMCwgbnkgLSAxLCBueSAtIDEsIG55IC8vIDIsIDMgKiBueSAvLyA0LCBueSAvLyA0XSwKICAgICAgICAgICAgICAgICAgZHR5cGU9ZmxvYXQpCiAgICB0cnk6CiAgICAgICAgc2t5ID0gd2NzX2EucGl4ZWxfdG9fd29ybGQoeHMsIHlzKQogICAgICAgIGJ4LCBieSA9IHdjc19iLndvcmxkX3RvX3BpeGVsKHNreSkKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcmV0dXJuIEZhbHNlCiAgICBtb3ZlZCA9IG5wLmh5cG90KGJ4IC0geHMsIGJ5IC0geXMpCiAgICByZXR1cm4gYm9vbChucC5hbGwobnAuaXNmaW5pdGUobW92ZWQpKSBhbmQgbW92ZWQubWF4KCkgPD0gdG9sZXJhbmNlX3B4KQoKZGVmIGFsaWduX3RvKHJlZl93Y3MsIHJlZl9zaGFwZSwgZGF0YSwgd2NzLCBiYW5kX3Jvd3M9NTEyKToKICAgICIiIlJlc2FtcGxlIG9uZSBmcmFtZSBvbnRvIGFub3RoZXIgZnJhbWUncyBwaXhlbCBncmlkIHVzaW5nIGJvdGggV0NTIHNvbHV0aW9ucy4KCiAgICBCb3RoIGZyYW1lcyBhcmUgcGxhdGUtc29sdmVkLCBzbyBldmVyeSBvdXRwdXQgcGl4ZWwgY2FuIGJlIG1hcHBlZCB0byBpdHMgdHJ1ZQogICAgc2t5IHBvc2l0aW9uIGFuZCB0aGUgaW5wdXQgZnJhbWUgcmVhZCB0aGVyZS4gRXhhY3QsIGFuZCBpdCBuZWVkcyBubyBtYXRjaGluZyBvZgogICAgc3RhcnMgYmV0d2VlbiB0aGUgdHdvIGltYWdlcy4KCiAgICBEb25lIGEgZmV3IGh1bmRyZWQgcm93cyBhdCBhIHRpbWUuIFRoZSB3aG9sZS1mcmFtZSB2ZXJzaW9uIGJ1aWx0IGZpdmUgYXJyYXlzCiAgICB0aGUgc2l6ZSBvZiB0aGUgaW1hZ2UgdG8gaG9sZCB0aGUgY29vcmRpbmF0ZXMgLSBhYm91dCAyLjUgR0Igb24gYSA2MS1tZWdhcGl4ZWwKICAgIGZyYW1lLCBmb3IgZXZlcnkgZXhwb3N1cmUgLSB3aGljaCBpcyBtZW1vcnkgc3BlbnQgb24gc2NyYXRjaCBwYXBlci4gVGhlCiAgICBhcml0aG1ldGljIGlzIGlkZW50aWNhbDsgb25seSB0aGUgb3JkZXIgY2hhbmdlZC4KICAgICIiIgogICAgZnJvbSBzY2lweS5uZGltYWdlIGltcG9ydCBtYXBfY29vcmRpbmF0ZXMKCiAgICBueSwgbnggPSByZWZfc2hhcGUKICAgIG91dCA9IG5wLmVtcHR5KChueSwgbngpLCBkdHlwZT1ucC5mbG9hdDMyKQogICAgZm9yIHkwIGluIHJhbmdlKDAsIG55LCBiYW5kX3Jvd3MpOgogICAgICAgIHkxID0gbWluKHkwICsgYmFuZF9yb3dzLCBueSkKICAgICAgICB5eSwgeHggPSBucC5tZ3JpZFt5MDp5MSwgMDpueF0KICAgICAgICBza3kgPSByZWZfd2NzLnBpeGVsX3RvX3dvcmxkKHh4LCB5eSkKICAgICAgICBzeCwgc3kgPSB3Y3Mud29ybGRfdG9fcGl4ZWwoc2t5KQogICAgICAgIG91dFt5MDp5MV0gPSBtYXBfY29vcmRpbmF0ZXMoZGF0YSwgW3N5LCBzeF0sIG9yZGVyPTEsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtb2RlPSJjb25zdGFudCIsIGN2YWw9bnAubmFuKQogICAgcmV0dXJuIG91dAoKZGVmIG1lYXN1cmVfZndobShkYXRhLCBtYXNrPU5vbmUsIG1heF9zb3VyY2VzPTIwMCwgcG9zaXRpb25zPU5vbmUpOgogICAgIiIiV2lkdGggb2YgdGhlIG1lZGlhbiBzdGFyIGluIGEgZnJhbWUsIGluIHBpeGVscywgYXQgaGFsZiBpdHMgaGVpZ2h0LgoKICAgIFdIWSBUSEUgTUVESUFOIFNUQVIgQU5EIE5PVCBUSEUgTUVESUFOIE9GIFRIRSBTVEFSUwogICAgQmVjYXVzZSBhIHNpbmdsZSBmYWludCBzdGFyIGlzIG1vc3RseSBub2lzZSwgYW5kIGV2ZXJ5IHdheSBvZiByZWR1Y2luZyBvbmUgdG8KICAgIGEgbnVtYmVyIGlzIHB1bGxlZCBhYm91dCBieSB0aGF0IG5vaXNlLiBTdGFja2luZyB0d28gaHVuZHJlZCBvZiB0aGVtIGZpcnN0CiAgICBnaXZlcyBvbmUgY2xlYW4gc3RhciB0byBtZWFzdXJlLCBhbmQgaXRzIHdpZHRoIGlzIHRoZSB3aWR0aCB0aGUgZnJhbWUgaGFzLgoKICAgIFdIWSBOT1QgVEhFIFNFQ09ORCBNT01FTlQKICAgIFRoYXQgaXMgd2hhdCB0aGlzIGRpZCBiZWZvcmUsIGFuZCBpdCBnYXZlIHRoZSB3cm9uZyBhbnN3ZXIgaW4gdGhlIHdheSB0aGF0CiAgICBtYXR0ZXJzIG1vc3Q6IGl0IGdvdCB0aGUgb3JkZXIgb2YgdGhlIHR3byBmaWx0ZXJzIGJhY2t3YXJkcy4gQSBzZWNvbmQgbW9tZW50CiAgICB3ZWlnaHRzIGVhY2ggcGl4ZWwgYnkgdGhlIHNxdWFyZSBvZiBpdHMgZGlzdGFuY2UsIHNvIHRoZSBvdXRlciByaW5nIG9mIGEgMTkKICAgIHBpeGVsIHN0YW1wIGNvdW50cyBhYm91dCBhIGh1bmRyZWQgYW5kIHNldmVudHkgdGltZXMgYXMgaGVhdmlseSBhcyB0aGUgY2VudHJlLgogICAgV2hhdGV2ZXIgaXMgb3V0IHRoZXJlIHRoYXQgaXMgbm90IHRoZSBzdGFyIC0gdGhlIGdhbGF4eSdzIG93biBsaWdodCB1bmRlciBpdCwKICAgIGEgbmVpZ2hib3VyLCBza3kgbGVmdCBvdmVyIGJlY2F1c2Ugb25seSB0aGUgZnJhbWUncyBnbG9iYWwgbGV2ZWwgaGFkIGJlZW4KICAgIHRha2VuIG9mZiAtIGlzIHRoZXJlZm9yZSBtb3N0IG9mIHRoZSBhbnN3ZXIuIE9uIE0zMyB0aGF0IG1hZGUgdGhlIEgtYWxwaGEKICAgIGZyYW1lIG1lYXN1cmUgNS41MyBweCBhZ2FpbnN0IFIncyA1Ljc3LCBzbyB0aGUgdG9vbCBibHVycmVkIEgtYWxwaGE7IHN0YWNraW5nCiAgICB0aGUgc2FtZSBzdGFycyBzaG93cyBILWFscGhhIGF0IDQuOTEgYW5kIFIgYXQgMy4xNCwgc28gSC1hbHBoYSB3YXMgYWxyZWFkeSB0aGUKICAgIGJyb2FkZXIgb2YgdGhlIHR3byBhbmQgYmx1cnJpbmcgaXQgbWFkZSB0aGUgbWlzbWF0Y2ggd29yc2UuIEV2ZXJ5IHN0YXIgdGhlbgogICAgc3VidHJhY3RlZCB0byBhIGJsYWNrIGNvcmUgaW5zaWRlIGEgYnJpZ2h0IHJpbmcsIHdoaWNoIGlzIGV4YWN0bHkgd2hhdCB0aGUKICAgIGZyYW1lcyBsb29rZWQgbGlrZS4KCiAgICBIYWxmLW1heGltdW0gaXMgcmVhZCBzdHJhaWdodCBvZmYgdGhlIHJhZGlhbCBwcm9maWxlIG9mIHRoZSBzdGFjay4gSXQgaXMgYQogICAgcHJvcGVydHkgb2YgdGhlIGNvcmUsIHdoZXJlIHRoZSBzdGFyIGlzLCBhbmQgaXQgZG9lcyBub3QgY2FyZSB3aGF0IHRoZSBzdGFtcCdzCiAgICBvdXRza2lydHMgY29udGFpbi4KCiAgICBUSFJFRSBUSElOR1MgVEhFIFNUQUNLIE5FRURTCiAgICBUaGUgbG9jYWwgc2t5IGNvbWVzIG9mZiBlYWNoIHN0YW1wIHNlcGFyYXRlbHksIHRha2VuIGZyb20gaXRzIGJvcmRlciByaW5nLCBzbwogICAgYSBzdGFyIG9uIGEgYnJpZ2h0IGFybSBhbmQgYSBzdGFyIG9uIGVtcHR5IHNreSBjb250cmlidXRlIHRoZSBzYW1lIHN0YXIuCiAgICBFYWNoIHN0YW1wIGlzIHNoaWZ0ZWQgb250byBhIGNvbW1vbiBjZW50cmUgYmVmb3JlIHN0YWNraW5nLCB0byBhIGZyYWN0aW9uIG9mIGEKICAgIHBpeGVsLCBvciB0aGUgc3RhY2sgd291bGQgYmUgc21lYXJlZCBieSB1cCB0byBoYWxmIGEgcGl4ZWwgb2Ygcm91bmRpbmcgLSBhbmQKICAgIHNtZWFyZWQgZGlmZmVyZW50bHkgaW4gZWFjaCBmaWx0ZXIsIHdoaWNoIGlzIHRoZSBlcnJvciBiZWluZyBsb29rZWQgZm9yLgogICAgU2F0dXJhdGVkIHN0YXJzIGFyZSBkcm9wcGVkOiBhIGZsYXQgdG9wIGlzIHdpZGVyIHRoYW4gdGhlIHN0YXIgdGhhdCBtYWRlIGl0LAogICAgYW5kIGl0IGlzIHRoZSBicmlnaHRlc3Qgc3RhcnMgdGhhdCBzYXR1cmF0ZSwgc28ga2VlcGluZyB0aGVtIHdvdWxkIGJpYXMgdGhlCiAgICBmcmFtZSB3aXRoIHRoZSBkZWVwZXIgZXhwb3N1cmUuIEEgY29yZSB3aXRoIHRocmVlIG9yIG1vcmUgcGl4ZWxzIHdpdGhpbiB0d28KICAgIHBlcmNlbnQgb2YgaXRzIHBlYWsgaXMgZmxhdCwgYW5kIGEgc3RhcidzIGNvcmUgaXMgbm90LgogICAgIiIiCiAgICBmcm9tIGFzdHJvcHkuc3RhdHMgaW1wb3J0IHNpZ21hX2NsaXBwZWRfc3RhdHMKICAgIGZyb20gc2NpcHkubmRpbWFnZSBpbXBvcnQgc2hpZnQgYXMgX3NoaWZ0CgogICAgaGFsZiA9IDEwCiAgICBueSwgbnggPSBkYXRhLnNoYXBlCiAgICB5eSwgeHggPSBucC5tZ3JpZFstaGFsZjpoYWxmICsgMSwgLWhhbGY6aGFsZiArIDFdCiAgICByciA9IG5wLmh5cG90KHh4LCB5eSkKICAgIGJvcmRlciA9IHJyID4gaGFsZiAtIDEuNQoKICAgIGlmIHBvc2l0aW9ucyBpcyBub3QgTm9uZSBhbmQgbGVuKHBvc2l0aW9ucykgPj0gMTA6CiAgICAgICAgY2FuZGlkYXRlcyA9IG5wLmFzYXJyYXkocG9zaXRpb25zLCBkdHlwZT1mbG9hdCkKICAgIGVsc2U6CiAgICAgICAgZnJvbSBwaG90dXRpbHMuZGV0ZWN0aW9uIGltcG9ydCBEQU9TdGFyRmluZGVyCiAgICAgICAgbWVkaWFuLCBzdGQgPSBzaWdtYV9jbGlwcGVkX3N0YXRzKGRhdGEsIHNpZ21hPTMuMCwgbWFzaz1tYXNrKVsxOl0KICAgICAgICBmaW5kZXIgPSBEQU9TdGFyRmluZGVyKGZ3aG09NC4wLCB0aHJlc2hvbGQ9MjAuMCAqIHN0ZCkKICAgICAgICB0YmwgPSBmaW5kZXIoZGF0YSAtIG1lZGlhbiwgbWFzaz1tYXNrKQogICAgICAgIGlmIHRibCBpcyBOb25lIG9yIGxlbih0YmwpID09IDA6CiAgICAgICAgICAgIHJldHVybiBOb25lCiAgICAgICAgdGJsLnNvcnQoImZsdXgiKQogICAgICAgIHRibCA9IHRibFs6Oi0xXVs6bWF4X3NvdXJjZXNdCiAgICAgICAgY2FuZGlkYXRlcyA9IG5wLmNvbHVtbl9zdGFjayhbdGJsWyJ4Y2VudHJvaWQiXSwgdGJsWyJ5Y2VudHJvaWQiXV0pCgogICAgZm91bmQgPSBbXQogICAgZm9yIHgsIHkgaW4gY2FuZGlkYXRlczoKICAgICAgICB4aSwgeWkgPSBpbnQocm91bmQoeCkpLCBpbnQocm91bmQoeSkpCiAgICAgICAgaWYgbm90IChoYWxmIDwgeGkgPCBueCAtIGhhbGYgYW5kIGhhbGYgPCB5aSA8IG55IC0gaGFsZik6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgc3RhbXAgPSBucC5uYW5fdG9fbnVtKAogICAgICAgICAgICBkYXRhW3lpIC0gaGFsZjp5aSArIGhhbGYgKyAxLCB4aSAtIGhhbGY6eGkgKyBoYWxmICsgMV0pLmFzdHlwZShmbG9hdCkKICAgICAgICBzdGFtcCA9IHN0YW1wIC0gbnAubWVkaWFuKHN0YW1wW2JvcmRlcl0pCiAgICAgICAgY29yZSA9IHN0YW1wW2hhbGYgLSAxOmhhbGYgKyAyLCBoYWxmIC0gMTpoYWxmICsgMl0KICAgICAgICBwZWFrID0gZmxvYXQoY29yZS5tYXgoKSkKICAgICAgICBpZiBwZWFrIDw9IDAgb3IgKGNvcmUgPj0gMC45OCAqIHBlYWspLnN1bSgpID49IDM6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgZm91bmQuYXBwZW5kKChwZWFrLCBfc2hpZnQoc3RhbXAsICh5aSAtIHksIHhpIC0geCksIG9yZGVyPTEsIG1vZGU9Im5lYXJlc3QiKSkpCiAgICBpZiBsZW4oZm91bmQpIDwgMTA6CiAgICAgICAgcmV0dXJuIE5vbmUKCiAgICBmb3VuZC5zb3J0KGtleT1sYW1iZGEgcGFpcjogLXBhaXJbMF0pCiAgICBzdGFja2VkID0gbnAubWVkaWFuKG5wLnN0YWNrKFtzIGZvciBfLCBzIGluIGZvdW5kWzptYXhfc291cmNlc11dKSwgYXhpcz0wKQogICAgcHJvZmlsZSA9IG5wLmFycmF5KFtzdGFja2VkWyhyciA+PSBxKSAmIChyciA8IHEgKyAxKV0ubWVhbigpCiAgICAgICAgICAgICAgICAgICAgICAgIGZvciBxIGluIHJhbmdlKGhhbGYgKyAxKV0pCiAgICBpZiBwcm9maWxlWzBdIDw9IDA6CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIGZvciBxIGluIHJhbmdlKDEsIGhhbGYgKyAxKToKICAgICAgICBpZiBwcm9maWxlW3FdIDwgMC41ICogcHJvZmlsZVswXToKICAgICAgICAgICAgYmVsb3csIGFib3ZlID0gcHJvZmlsZVtxIC0gMV0sIHByb2ZpbGVbcV0KICAgICAgICAgICAgcmV0dXJuIGZsb2F0KDIuMCAqIChxIC0gMSArICgwLjUgKiBwcm9maWxlWzBdIC0gYmVsb3cpIC8gKGFib3ZlIC0gYmVsb3cpKSkKICAgIHJldHVybiBOb25lCgpkZWYgbWF0Y2hfcHNmKHNoYXJwLCBmd2htX3NoYXJwLCBmd2htX2JsdW50KToKICAgICIiIkJsdXIgdGhlIHNoYXJwZXIgZnJhbWUgc28gYm90aCBmcmFtZXMgaGF2ZSB0aGUgc2FtZSBzdGFyIHNpemUuCgogICAgU3VidHJhY3RpbmcgZnJhbWVzIG9mIGRpZmZlcmVudCBzaGFycG5lc3MgbGVhdmVzIGEgcmluZyBhcm91bmQgZXZlcnkgc3RhciAtIGEKICAgIGJyaWdodCBjb3JlIHdpdGggYSBkYXJrIGhhbG8sIG9yIHRoZSByZXZlcnNlIC0gYW5kIHRob3NlIHJpbmdzIGFyZSB0aGVuIGRldGVjdGVkCiAgICBhcyBISUkgcmVnaW9ucy4gTWF0Y2hpbmcgdGhlIHdpZHRocyBmaXJzdCBpcyB3aGF0IG1ha2VzIHN0YXJzIGRpc2FwcGVhciBjbGVhbmx5LgogICAgIiIiCiAgICBmcm9tIHNjaXB5Lm5kaW1hZ2UgaW1wb3J0IGdhdXNzaWFuX2ZpbHRlcgoKICAgIGlmIGZ3aG1fYmx1bnQgaXMgTm9uZSBvciBmd2htX3NoYXJwIGlzIE5vbmUgb3IgZndobV9ibHVudCA8PSBmd2htX3NoYXJwOgogICAgICAgIHJldHVybiBzaGFycCwgMC4wCiAgICBzaWdtYSA9IG1hdGguc3FydChtYXgoZndobV9ibHVudCAqKiAyIC0gZndobV9zaGFycCAqKiAyLCAwLjApKSAvIDIuMzU0OAogICAgcmV0dXJuIGdhdXNzaWFuX2ZpbHRlcihzaGFycCwgc2lnbWEpLCBzaWdtYQoKZGVmIF9zdGFja19zdGFycyhkYXRhLCBwb3NpdGlvbnMsIG1heF9zb3VyY2VzPTIwMCwgaGFsZj0xMCk6CiAgICAiIiJUaGUgbWVkaWFuIHN0YXIgb2YgYSBmcmFtZTogbG9jYWwgc2t5IG9mZiwgc3ViLXBpeGVsIGFsaWduZWQsIG5vIHNhdHVyYXRpb24uCgogICAgT25lIHN0YXIgaXMgbW9zdGx5IG5vaXNlLiBUd28gaHVuZHJlZCBvZiB0aGVtIHN0YWNrZWQgaXMgb25lIGNsZWFuIHN0YXIsIGFuZAogICAgZXZlcnkgcXVlc3Rpb24gYWJvdXQgdGhlIHNoYXBlIG9mIHRoaXMgZnJhbWUncyBzdGFycyBpcyBhc2tlZCBvZiB0aGF0LgogICAgIiIiCiAgICBmcm9tIHNjaXB5Lm5kaW1hZ2UgaW1wb3J0IHNoaWZ0IGFzIF9zaGlmdAoKICAgIGRhdGEgPSBucC5hc2FycmF5KGRhdGEpCiAgICBwb3MgPSBucC5hc2FycmF5KHBvc2l0aW9ucywgZHR5cGU9ZmxvYXQpCiAgICBueSwgbnggPSBkYXRhLnNoYXBlCiAgICB5eSwgeHggPSBucC5tZ3JpZFstaGFsZjpoYWxmICsgMSwgLWhhbGY6aGFsZiArIDFdCiAgICBib3JkZXIgPSBucC5oeXBvdCh4eCwgeXkpID4gaGFsZiAtIDEuNQoKICAgIGZvdW5kID0gW10KICAgIGZvciB4LCB5IGluIHBvczoKICAgICAgICB4aSwgeWkgPSBpbnQocm91bmQoeCkpLCBpbnQocm91bmQoeSkpCiAgICAgICAgaWYgbm90IChoYWxmIDwgeGkgPCBueCAtIGhhbGYgYW5kIGhhbGYgPCB5aSA8IG55IC0gaGFsZik6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgc3RhbXAgPSBucC5uYW5fdG9fbnVtKAogICAgICAgICAgICBkYXRhW3lpIC0gaGFsZjp5aSArIGhhbGYgKyAxLCB4aSAtIGhhbGY6eGkgKyBoYWxmICsgMV0pLmFzdHlwZShmbG9hdCkKICAgICAgICBzdGFtcCA9IHN0YW1wIC0gbnAubWVkaWFuKHN0YW1wW2JvcmRlcl0pCiAgICAgICAgY29yZSA9IHN0YW1wW2hhbGYgLSAxOmhhbGYgKyAyLCBoYWxmIC0gMTpoYWxmICsgMl0KICAgICAgICBwZWFrID0gZmxvYXQoY29yZS5tYXgoKSkKICAgICAgICBpZiBwZWFrIDw9IDAgb3IgKGNvcmUgPj0gMC45OCAqIHBlYWspLnN1bSgpID49IDM6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgZm91bmQuYXBwZW5kKChwZWFrLCBfc2hpZnQoc3RhbXAsICh5aSAtIHksIHhpIC0geCksIG9yZGVyPTEsIG1vZGU9Im5lYXJlc3QiKSkpCiAgICBpZiBsZW4oZm91bmQpIDwgMTA6CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIGZvdW5kLnNvcnQoa2V5PWxhbWJkYSBwYWlyOiAtcGFpclswXSkKICAgIHJldHVybiBucC5tZWRpYW4obnAuc3RhY2soW3MgZm9yIF8sIHMgaW4gZm91bmRbOm1heF9zb3VyY2VzXV0pLCBheGlzPTApCgpkZWYgbWF0Y2hfcHNmX29uX3N0YXJzKGhhLCByZWQsIHBvc2l0aW9ucywgbWF4X3NpZ21hPTMuMCwgc3RlcD0wLjAyLCBoYWxmPTEwKToKICAgICIiIkJsdXIgb25lIGZyYW1lIHVudGlsIGl0cyBzdGFycyBoYXZlIHRoZSBzYW1lIHNoYXBlIGFzIHRoZSBvdGhlcidzLgoKICAgIFdIWSBOT1QgTUVBU1VSRSBUV08gV0lEVEhTIEFORCBTVUJUUkFDVCBUSEVNIElOIFFVQURSQVRVUkUKICAgIEJlY2F1c2UgdGhhdCBpcyB3aGF0IG1hdGNoX3BzZiBkb2VzIGFuZCBpdCBkb2VzIG5vdCB3b3JrIGhlcmUsIGZvciB0d28gcmVhc29ucwogICAgdGhhdCBjb21wb3VuZC4gQSB3aWR0aCBpcyBvbmUgbnVtYmVyIHNxdWVlemVkIG91dCBvZiBhIHdob2xlIHByb2ZpbGUsIHNvIGl0CiAgICB0aHJvd3MgYXdheSBleGFjdGx5IHRoZSBwYXJ0IHRoYXQgbWF0dGVycyAtIGEgc3RhciBjYW4gaGF2ZSB0aGUgc2FtZSBoYWxmLXdpZHRoCiAgICBhcyBhbm90aGVyIGFuZCBhIHF1aXRlIGRpZmZlcmVudCBjb3JlIGFuZCBza2lydC4gQW5kIHRoZSBxdWFkcmF0dXJlIGZvcm11bGEgaXMKICAgIG9ubHkgZXhhY3QgZm9yIEdhdXNzaWFuczsgYSByZWFsIHN0YXIgaGFzIGJyb2FkZXIgd2luZ3MgdGhhbiBhIEdhdXNzaWFuLCBzbyB0aGUKICAgIGJsdXIgaXQgcHJlc2NyaWJlcyBpcyBub3QgdGhlIGJsdXIgdGhhdCBtYXRjaGVzIHRoZSB0d28uCgogICAgTWVhc3VyZWQgb24gTTgzOiB0aGUgd2lkdGhzIGNhbWUgb3V0IEgtYWxwaGEgMy43OCBweCBhZ2FpbnN0IFIgNC4yMiwgd2hpY2gKICAgIHByZXNjcmliZWQgYmx1cnJpbmcgSC1hbHBoYSBieSBzaWdtYSAwLjgwLiBBZnRlcndhcmRzIHRoZSBzdGFja2VkIHN0YXIgbGVmdCBhCiAgICBuZWdhdGl2ZSBjb3JlIGluc2lkZSBhIHBvc2l0aXZlIHJpbmcgLSB0aGUgc2lnbmF0dXJlIG9mIEgtYWxwaGEgbm93IGJlaW5nIHRoZQogICAgYnJvYWRlciBvZiB0aGUgdHdvLiBUaGUgcHJlc2NyaXB0aW9uIGhhZCB0aGUgc2lnbiBvZiB0aGUgYW5zd2VyIHdyb25nLgoKICAgIFdIQVQgVEhJUyBET0VTIElOU1RFQUQKICAgIEl0IGFza3MgdGhlIHF1ZXN0aW9uIHRoZSBwaWN0dXJlIGFza3MuIEJvdGggZnJhbWVzJyBzdGFycyBhcmUgc3RhY2tlZCwgZWFjaAogICAgcHJvZmlsZSBpcyBkaXZpZGVkIGJ5IGl0cyBvd24gbGlnaHQgaW5zaWRlIHNpeCBwaXhlbHMgc28gdGhhdCBicmlnaHRuZXNzIGRyb3BzCiAgICBvdXQgb2YgdGhlIGNvbXBhcmlzb24gYW5kIG9ubHkgc2hhcGUgaXMgbGVmdCwgYW5kIHRoZW4gb25lIGZyYW1lIGlzIGJsdXJyZWQgYnkKICAgIGV2ZXJ5IHNpZ21hIGluIGEgcmFuZ2Ugd2hpbGUgdGhlIHR3byBwcm9maWxlcyBhcmUgY29tcGFyZWQgcmluZyBieSByaW5nLiBUaGUKICAgIHNpZ21hIHRoYXQgbWFrZXMgdGhlbSBhZ3JlZSBpcyB0aGUgYW5zd2VyLCBhbmQgaXQgaXMgYW4gYW5zd2VyIGFib3V0IHRoZSB3aG9sZQogICAgcHJvZmlsZSByYXRoZXIgdGhhbiBhYm91dCBvbmUgcG9pbnQgb24gaXQuCgogICAgQSBwb3NpdGl2ZSBzaWdtYSBibHVycyBILWFscGhhLCBhIG5lZ2F0aXZlIG9uZSBibHVycyBSOyB0aGUgc2VhcmNoIGNvdmVycyBib3RoCiAgICBzbyBpdCBjYW5ub3QgZ2V0IHRoZSBkaXJlY3Rpb24gd3JvbmcuCgogICAgUmV0dXJucyAoaGEsIHJlZCwgc2lnbWEsIGRpYWdub3N0aWNzKS4KICAgICIiIgogICAgZnJvbSBzY2lweS5uZGltYWdlIGltcG9ydCBnYXVzc2lhbl9maWx0ZXIKCiAgICBpZiBwb3NpdGlvbnMgaXMgTm9uZSBvciBsZW4ocG9zaXRpb25zKSA8IDEwOgogICAgICAgIHJldHVybiBoYSwgcmVkLCAwLjAsIHsid2h5IjogIm5vIHN0YXIgcG9zaXRpb25zIHRvIG1lYXN1cmUgb24ifQogICAgQTAgPSBfc3RhY2tfc3RhcnMoaGEsIHBvc2l0aW9ucywgaGFsZj1oYWxmKQogICAgUjAgPSBfc3RhY2tfc3RhcnMocmVkLCBwb3NpdGlvbnMsIGhhbGY9aGFsZikKICAgIGlmIEEwIGlzIE5vbmUgb3IgUjAgaXMgTm9uZToKICAgICAgICByZXR1cm4gaGEsIHJlZCwgMC4wLCB7IndoeSI6ICJub3QgZW5vdWdoIHVzYWJsZSBzdGFycyB0byBzdGFjayJ9CgogICAgeXksIHh4ID0gbnAubWdyaWRbLWhhbGY6aGFsZiArIDEsIC1oYWxmOmhhbGYgKyAxXQogICAgcnIgPSBucC5oeXBvdCh4eCwgeXkpCiAgICByaW5ncyA9IFsocnIgPj0gcSkgJiAocnIgPCBxICsgMSkgZm9yIHEgaW4gcmFuZ2UoaGFsZildCiAgICBpbm5lciA9IHJyIDwgNgoKICAgIGRlZiBzaGFwZV9vZihzdGFtcCk6CiAgICAgICAgIiIiVGhlIHJhZGlhbCBwcm9maWxlLCBkaXZpZGVkIGJ5IHRoZSBsaWdodCBpbiB0aGUgY29yZSAtIHNoYXBlIG9ubHkuIiIiCiAgICAgICAgdG90YWwgPSBzdGFtcFtpbm5lcl0uc3VtKCkKICAgICAgICBpZiB0b3RhbCA8PSAwOgogICAgICAgICAgICByZXR1cm4gTm9uZQogICAgICAgIHJldHVybiBucC5hcnJheShbc3RhbXBbbV0ubWVhbigpIGZvciBtIGluIHJpbmdzXSkgLyB0b3RhbAoKICAgIGRlZiBtaXNtYXRjaChzaWdtYSk6CiAgICAgICAgYSA9IGdhdXNzaWFuX2ZpbHRlcihBMCwgc2lnbWEpIGlmIHNpZ21hID4gMCBlbHNlIEEwCiAgICAgICAgciA9IGdhdXNzaWFuX2ZpbHRlcihSMCwgLXNpZ21hKSBpZiBzaWdtYSA8IDAgZWxzZSBSMAogICAgICAgIHNhLCBzciA9IHNoYXBlX29mKGEpLCBzaGFwZV9vZihyKQogICAgICAgIGlmIHNhIGlzIE5vbmUgb3Igc3IgaXMgTm9uZToKICAgICAgICAgICAgcmV0dXJuIG5wLmluZgogICAgICAgIHJldHVybiBmbG9hdCgoKHNhIC0gc3IpICoqIDIpLnN1bSgpKQoKICAgIGdyaWQgPSBucC5hcmFuZ2UoLW1heF9zaWdtYSwgbWF4X3NpZ21hICsgc3RlcCwgc3RlcCkKICAgIHNjb3JlcyA9IG5wLmFycmF5KFttaXNtYXRjaChzKSBmb3IgcyBpbiBncmlkXSkKICAgIGJlc3QgPSBmbG9hdChncmlkW2ludChucC5hcmdtaW4oc2NvcmVzKSldKQogICAgaWYgYWJzKGJlc3QpIDwgc3RlcDoKICAgICAgICBiZXN0ID0gMC4wCgogICAgZGlhZyA9IHsic2lnbWEiOiBiZXN0LAogICAgICAgICAgICAibWlzbWF0Y2hfYmVmb3JlIjogbWlzbWF0Y2goMC4wKSwKICAgICAgICAgICAgIm1pc21hdGNoX2FmdGVyIjogbWlzbWF0Y2goYmVzdCksCiAgICAgICAgICAgICJibHVycmVkIjogKCJILWFscGhhIiBpZiBiZXN0ID4gMCBlbHNlICJSIiBpZiBiZXN0IDwgMCBlbHNlICJuZWl0aGVyIil9CiAgICBpZiBiZXN0ID4gMDoKICAgICAgICBoYSA9IGdhdXNzaWFuX2ZpbHRlcihoYSwgYmVzdCkKICAgIGVsaWYgYmVzdCA8IDA6CiAgICAgICAgcmVkID0gZ2F1c3NpYW5fZmlsdGVyKHJlZCwgLWJlc3QpCiAgICByZXR1cm4gaGEsIHJlZCwgYWJzKGJlc3QpLCBkaWFnCgpkZWYgZGVzY3JpYmVfcHNmX21hdGNoKGRpYWcpOgogICAgIiIiV2hhdCB0aGUgc2VhcmNoIHNldHRsZWQgb24sIGFuZCBieSBob3cgbXVjaCBpdCBpbXByb3ZlZCB0aGluZ3MuIiIiCiAgICBpZiBkaWFnLmdldCgid2h5Iik6CiAgICAgICAgcmV0dXJuICIgIHBvaW50IHNwcmVhZCBmdW5jdGlvbnMgbGVmdCBhbG9uZSAoe30pIi5mb3JtYXQoZGlhZ1sid2h5Il0pCiAgICBpZiBkaWFnWyJibHVycmVkIl0gPT0gIm5laXRoZXIiOgogICAgICAgIHJldHVybiAiICB0aGUgc3RhcnMgYWxyZWFkeSBoYXZlIHRoZSBzYW1lIHNoYXBlIGluIGJvdGggZnJhbWVzIgogICAgYmVmb3JlLCBhZnRlciA9IGRpYWdbIm1pc21hdGNoX2JlZm9yZSJdLCBkaWFnWyJtaXNtYXRjaF9hZnRlciJdCiAgICBsaW5lID0gKCIgIGJsdXJyZWQgdGhlIHt9IGZyYW1lIGJ5IHNpZ21hID0gezouMmZ9IHB4LCBjaG9zZW4gYnkgY29tcGFyaW5nIHRoZSAiCiAgICAgICAgICAgICJzdGFja2VkIHN0YXIgcmluZyBieSByaW5nIi5mb3JtYXQoZGlhZ1siYmx1cnJlZCJdLCBhYnMoZGlhZ1sic2lnbWEiXSkpKQogICAgaWYgYmVmb3JlID4gMDoKICAgICAgICBsaW5lICs9ICgiXG4gICAgdGhlIGRpc2FncmVlbWVudCBiZXR3ZWVuIHRoZSB0d28gcHJvZmlsZXMgZmVsbCB0byB7Oi4wZn0lICIKICAgICAgICAgICAgICAgICAib2Ygd2hhdCBpdCB3YXMiLmZvcm1hdCgxMDAuMCAqIGFmdGVyIC8gYmVmb3JlKSkKICAgIHJldHVybiBsaW5lCgpkZWYgY29udGludXVtX2ZhY3RvcihoYSwgcmVkLCBtYXNrPU5vbmUsIG5fc3RhcnM9MjAwLCBwb3NpdGlvbnM9Tm9uZSk6CiAgICAiIiJIb3cgbXVjaCB0byBicmlnaHRlbiB0aGUgbmFycm93IGZyYW1lIGJlZm9yZSBzdWJ0cmFjdGluZyB0aGUgYnJvYWQgb25lLgoKICAgIEEgZm9yZWdyb3VuZCBzdGFyIGVtaXRzIG5vIEgtYWxwaGEgbGluZSB3b3J0aCBzcGVha2luZyBvZiwgc28gd2hhdGV2ZXIgaXQgcHV0cwogICAgaW50byB0aGUgbmFycm93IGZyYW1lIGlzIHB1cmUgY29udGludXVtLCBhbmQgdGhlIHJhdGlvIG9mIGl0cyBicmlnaHRuZXNzIGluIHRoZQogICAgdHdvIGZyYW1lcyBpcyB0aGUgZmFjdG9yIHdlIHdhbnQuCgogICAgIkZvcmVncm91bmQiIGlzIHRoZSB3b3JkIHRoYXQgbWF0dGVycywgYW5kIGl0IGlzIHdoeSBgcG9zaXRpb25zYCBleGlzdHMuIFBvaW50CiAgICBzb3VyY2VzIGZvdW5kIGluIHRoZSBpbWFnZSBhcmUgZm9yZWdyb3VuZCBvbmx5IHdoZW4gdGhlIGdhbGF4eSBpcyBmYXIgZW5vdWdoCiAgICBhd2F5IGZvciBpdHMgb3duIHN0YXJzIG5vdCB0byBiZSByZXNvbHZlZC4gSW4gYSBuZWFyYnkgZ2FsYXh5IHRoZXkgYXJlLCBpbgogICAgdGhlaXIgdGhvdXNhbmRzLCBhbmQgdGhleSBhcmUgbm90IHB1cmUgY29udGludXVtIC0gbWVhc3VyZWQgb24gTTMzIHRoZSBtaXh0dXJlCiAgICBnYXZlIGEgc3Rhci10by1zdGFyIHNjYXR0ZXIgb2YgNjYgcGVyY2VudCwgd2hpY2ggaXMgbm8gZmFjdG9yIGF0IGFsbC4gUGFzc2luZwogICAgY2F0YWxvZ3VlIHBvc2l0aW9ucyBvZiBzdGFycyB3aXRoIGEgbWVhc3VyYWJsZSBwYXJhbGxheCBzZXR0bGVzIHRoZSBxdWVzdGlvbgogICAgcGh5c2ljYWxseSBpbnN0ZWFkIG9mIGhvcGluZyBkaXN0YW5jZSBzZXR0bGVzIGl0LgoKICAgIFRoZSBtZWRpYW4gb2YgdGhlIHBlci1zdGFyIHJhdGlvIGlzIHVzZWQgcmF0aGVyIHRoYW4gYSBmaXQsIGJlY2F1c2UgdGhlIG1lZGlhbgogICAgaXMgbm90IGRyYWdnZWQgYXJvdW5kIGJ5IHRoZSBmZXcgc3RhcnMgdGhhdCBkbyBoYXZlIGxpbmUgZW1pc3Npb24sIG9yIGFyZQogICAgc2F0dXJhdGVkLCBvciBzaXQgb24gYSBicmlnaHQgcGllY2Ugb2YgdGhlIGdhbGF4eS4KCiAgICBSZXR1cm5zIHRoZSBmYWN0b3IgYW5kIGRpYWdub3N0aWNzIC0gaW4gcGFydGljdWxhciB0aGUgc3Rhci10by1zdGFyIHNjYXR0ZXIsCiAgICB3aGljaCBpcyB0aGUgaG9uZXN0IHVuY2VydGFpbnR5IG9uIHRoZSBzdWJ0cmFjdGlvbi4KICAgICIiIgogICAgZnJvbSBwaG90dXRpbHMuZGV0ZWN0aW9uIGltcG9ydCBEQU9TdGFyRmluZGVyCiAgICBmcm9tIHBob3R1dGlscy5hcGVydHVyZSBpbXBvcnQgQ2lyY3VsYXJBcGVydHVyZSwgQ2lyY3VsYXJBbm51bHVzLCBhcGVydHVyZV9waG90b21ldHJ5CiAgICBmcm9tIGFzdHJvcHkuc3RhdHMgaW1wb3J0IHNpZ21hX2NsaXBwZWRfc3RhdHMKCiAgICBpZiBwb3NpdGlvbnMgaXMgbm90IE5vbmUgYW5kIGxlbihwb3NpdGlvbnMpID49IDEwOgogICAgICAgICMgQSBnZW5lcm91cyBzbGljZSBvZiB0aGUgY2F0YWxvZ3VlLCB0aGVuIHRoZSBjaG9pY2UgaXMgbWFkZSBvbiB3aGF0IHRoZXNlCiAgICAgICAgIyBmcmFtZXMgYWN0dWFsbHkgcmVjb3JkZWQgcmF0aGVyIHRoYW4gb24gd2hhdCB0aGUgY2F0YWxvZ3VlIHNheXMuIEdhaWEncwogICAgICAgICMgRyBtYWduaXR1ZGUgaXMgbm90IHRoaXMgdGVsZXNjb3BlJ3MgUiwgYW5kIGEgc3RhciB0aGF0IGlzIHJlc3BlY3RhYmxlIGluCiAgICAgICAgIyB0aGUgY2F0YWxvZ3VlIGNhbiBiZSBhIGhhbmRmdWwgb2YgY291bnRzIHRocm91Z2ggYSBuYXJyb3cgZmlsdGVyIC0gd2hlcmUKICAgICAgICAjIHRoZSByYXRpbyBpdCBnaXZlcyBpcyBub3QgYSBtZWFzdXJlbWVudCBidXQgYSBkaXZpc2lvbiBieSBub2lzZS4KICAgICAgICBwb3MgPSBucC5hc2FycmF5KHBvc2l0aW9ucywgZHR5cGU9ZmxvYXQpWzo0ICogbl9zdGFyc10KICAgIGVsc2U6CiAgICAgICAgbWVhbl9yLCBtZWRfciwgc3RkX3IgPSBzaWdtYV9jbGlwcGVkX3N0YXRzKHJlZCwgc2lnbWE9My4wLCBtYXNrPW1hc2spCiAgICAgICAgZmluZGVyID0gREFPU3RhckZpbmRlcihmd2htPTQuMCwgdGhyZXNob2xkPTI1LjAgKiBzdGRfcikKICAgICAgICBzdGFycyA9IGZpbmRlcihyZWQgLSBtZWRfciwgbWFzaz1tYXNrKQogICAgICAgIGlmIHN0YXJzIGlzIE5vbmUgb3IgbGVuKHN0YXJzKSA8IDU6CiAgICAgICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigidG9vIGZldyBzdGFycyBmb3VuZCB0byBzZXQgdGhlIGNvbnRpbnV1bSBmYWN0b3IiKQoKICAgICAgICBzdGFycy5zb3J0KCJmbHV4IikKICAgICAgICBzdGFycyA9IHN0YXJzWzo6LTFdWzpuX3N0YXJzXQogICAgICAgIHBvcyA9IG5wLnRyYW5zcG9zZSgoc3RhcnNbInhjZW50cm9pZCJdLCBzdGFyc1sieWNlbnRyb2lkIl0pKQoKICAgIGFwID0gQ2lyY3VsYXJBcGVydHVyZShwb3MsIHI9Ni4wKQogICAgYW5uID0gQ2lyY3VsYXJBbm51bHVzKHBvcywgcl9pbj0xMC4wLCByX291dD0xNi4wKQoKICAgIGRlZiBuZXRfZmx1eChpbWcpOgogICAgICAgIHBob3QgPSBhcGVydHVyZV9waG90b21ldHJ5KGltZywgW2FwLCBhbm5dKQogICAgICAgIGJrZ19tZWFuID0gcGhvdFsiYXBlcnR1cmVfc3VtXzEiXSAvIGFubi5hcmVhCiAgICAgICAgcmV0dXJuIG5wLmFzYXJyYXkocGhvdFsiYXBlcnR1cmVfc3VtXzAiXSAtIGJrZ19tZWFuICogYXAuYXJlYSwgZHR5cGU9ZmxvYXQpCgogICAgZl9oYSwgZl9yID0gbmV0X2ZsdXgoaGEpLCBuZXRfZmx1eChyZWQpCgogICAgb2sgPSBucC5pc2Zpbml0ZShmX2hhKSAmIG5wLmlzZmluaXRlKGZfcikgJiAoZl9oYSA+IDApICYgKGZfciA+IDApCiAgICBpZiBvay5zdW0oKSA8IDU6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCJubyB1c2FibGUgc3RhcnMgZm9yIHRoZSBjb250aW51dW0gZmFjdG9yIikKCiAgICAjIEtlZXAgdGhlIGJyaWdodGVzdCwgbWVhc3VyZWQgaGVyZS4gV2l0aCBhIGNhdGFsb2d1ZSBsaXN0IHRoaXMgaXMgd2hhdCB0dXJucwogICAgIyBhIHNldCBvZiBzdGFycyB0aGF0IGFyZSBtZXJlbHkgcmVhbCBpbnRvIGEgc2V0IHRoYXQgaXMgd29ydGggbWVhc3VyaW5nLgogICAgaWYgb2suc3VtKCkgPiBuX3N0YXJzOgogICAgICAgIG9yZGVyID0gbnAuYXJnc29ydChucC53aGVyZShvaywgZl9yLCAtbnAuaW5mKSlbOjotMV1bOm5fc3RhcnNdCiAgICAgICAga2VlcF9icmlnaHQgPSBucC56ZXJvcyhvay5zaGFwZSwgZHR5cGU9Ym9vbCkKICAgICAgICBrZWVwX2JyaWdodFtvcmRlcl0gPSBUcnVlCiAgICAgICAgb2sgJj0ga2VlcF9icmlnaHQKCiAgICByYXRpbyA9IGZfcltva10gLyBmX2hhW29rXQogICAgbWVkID0gZmxvYXQobnAubWVkaWFuKHJhdGlvKSkKICAgIHNwcmVhZCA9IDEuNDgyNiAqIG5wLm1lZGlhbihucC5hYnMocmF0aW8gLSBtZWQpKQogICAgaWYgc3ByZWFkID4gMDoKICAgICAgICByYXRpbyA9IHJhdGlvW25wLmFicyhyYXRpbyAtIG1lZCkgPCAzLjAgKiBzcHJlYWRdCgogICAgZmFjdG9yID0gZmxvYXQobnAubWVkaWFuKHJhdGlvKSkKICAgIHNjYXR0ZXIgPSBmbG9hdCgxLjQ4MjYgKiBucC5tZWRpYW4obnAuYWJzKHJhdGlvIC0gZmFjdG9yKSkpCiAgICByZXR1cm4gZmFjdG9yLCB7CiAgICAgICAgIm5fc3RhcnMiOiBpbnQocmF0aW8uc2l6ZSksCiAgICAgICAgInNjYXR0ZXIiOiBzY2F0dGVyLAogICAgICAgICJyZWxhdGl2ZV9zY2F0dGVyIjogc2NhdHRlciAvIGZhY3RvciBpZiBmYWN0b3IgZWxzZSBmbG9hdCgibmFuIiksCiAgICB9CgpkZWYgYmFja2dyb3VuZF9ib3goc2hhcGUsIHJlcXVlc3RlZD1Ob25lLCByMjVfcHg9Tm9uZSk6CiAgICAiIiJIb3cgbGFyZ2UgYSBzcXVhcmUgdG8gZXN0aW1hdGUgdGhlIGJhY2tncm91bmQgaW4uCgogICAgVGhpcyBpcyBub3QgYSBmcmVlIHBhcmFtZXRlci4gVGhlIGJhY2tncm91bmQgbWFwIGlzIG1lYW50IHRvIGZvbGxvdyB0aGUgc2t5CiAgICBhbmQgdGhlIGdhbGF4eSdzIG93biBzbW9vdGggbGlnaHQsIGFuZCB0byBsZWF2ZSB0aGUgcmVnaW9ucyBzaXR0aW5nIG9uIHRvcCBvZgogICAgaXQgYWxvbmUuIEEgYm94IHRoZSBzaXplIG9mIGEgc3BpcmFsIGFybSBkb2VzIHRoZSBvcHBvc2l0ZTogaXQgdHJlYXRzIHRoZSBhcm0KICAgIGFzIGJhY2tncm91bmQgYW5kIHN1YnRyYWN0cyBhd2F5IHRoZSB2ZXJ5IHRoaW5nIHRoZSBmYWludCByZWdpb25zIHN0YW5kIG9uLgogICAgTWVhc3VyZWQgb24gdGhlIE04MyBmcmFtZXMsIGdvaW5nIGZyb20gYSA2NC1waXhlbCBib3ggdG8gYSAyMDAtcGl4ZWwgb25lIGZvdW5kCiAgICA3OSBtb3JlIHJlZ2lvbnMgaW5zaWRlIHRoZSBnYWxheHkgd2hpbGUgdGhlIGNvdW50IG91dCBpbiB0aGUgZW1wdHkgZmllbGQsIHdoZXJlCiAgICBub3RoaW5nIHJlYWwgY2FuIGJlLCBkaWQgbm90IG1vdmUgYXQgYWxsIC0gc28gdGhvc2UgNzkgd2VyZSByZWFsIGFuZCB3ZXJlIGJlaW5nCiAgICBlcmFzZWQuCgogICAgQSB0d2VsZnRoIG9mIHRoZSBmcmFtZSBpcyB3aWRlIGVub3VnaCB0byBiZSB3ZWxsIGNsZWFyIG9mIGFueSBzaW5nbGUgcmVnaW9uIGFuZAogICAgc21hbGwgZW5vdWdoIHRvIHN0aWxsIGZvbGxvdyBhIGdyYWRpZW50IGFjcm9zcyB0aGUgZmllbGQuCiAgICAiIiIKICAgIGlmIHJlcXVlc3RlZDoKICAgICAgICByZXR1cm4gaW50KHJlcXVlc3RlZCkKICAgICMgVGllZCB0byB0aGUgZ2FsYXh5LCBub3QgdG8gdGhlIGZyYW1lLiBBIGJveCBzZXQgYXMgYSBmcmFjdGlvbiBvZiB0aGUgaW1hZ2UKICAgICMgbWFrZXMgdGhlIGFuc3dlciBkZXBlbmQgb24gaG93IG11Y2ggZW1wdHkgc2t5IGhhcHBlbmVkIHRvIHN1cnJvdW5kIHRoZQogICAgIyBnYWxheHkgdGhhdCBuaWdodCAtIGFuZCBpdCBzaG93ZWQ6IGNyb3BwaW5nIHRoZSBmcmFtZSB0byB0aGUgZ2FsYXh5LCB3aGljaAogICAgIyBzaG91bGQgaGF2ZSBjaGFuZ2VkIG5vdGhpbmcgYnV0IHRoZSBydW5uaW5nIHRpbWUsIG1vdmVkIHRoZSByZWdpb24gY291bnQgb24KICAgICMgTTgzIGZyb20gMzQ4IHRvIDI4MiBwdXJlbHkgYmVjYXVzZSB0aGUgYm94IHNocmFuayB3aXRoIHRoZSBmcmFtZS4KICAgICMKICAgICMgQSB0aGlyZCBvZiBSMjUgaXMgd2hhdCB0aGUgbWVhc3VyZW1lbnQgcG9pbnRlZCB0by4gT24gTTgzLCBSMjUgaXMgNTYwIHB4IGFuZAogICAgIyB0aGUgYm94IHRoYXQgd29ya2VkIGJlc3Qgd2FzIDIwMC4KICAgIGlmIHIyNV9weDoKICAgICAgICByZXR1cm4gaW50KG1pbihtYXgocjI1X3B4IC8gMy4wLCA2NCksIDQwMCkpCiAgICByZXR1cm4gaW50KG1pbihtYXgobWluKHNoYXBlKSAvLyAxMiwgNjQpLCAyNTYpKQoKZGVmIGRldGVjdF9yZWdpb25zKG5ldCwgdGhyZXNob2xkX3NpZ21hPTQuMCwgbWluX3BpeGVscz02LCBtYXNrPU5vbmUsIGRlYmxlbmQ9VHJ1ZSwKICAgICAgICAgICAgICAgICAgIGJveF9zaXplPU5vbmUsIG5sZXZlbHM9NjQsIGNvbnRyYXN0PTFlLTQsIHIyNV9weD1Ob25lKToKICAgICIiIkZpbmQgSElJIHJlZ2lvbnMgaW4gdGhlIGNvbnRpbnV1bS1zdWJ0cmFjdGVkIGZyYW1lLgoKICAgIEhJSSByZWdpb25zIGFyZSBleHRlbmRlZCBibG9icywgbm90IHBvaW50cywgc28gY29ubmVjdGVkIGdyb3VwcyBvZiBwaXhlbHMgYWJvdmUKICAgIGEgdGhyZXNob2xkIGFyZSBncm93biByYXRoZXIgdGhhbiBzdGVsbGFyIHByb2ZpbGVzIGZpdHRlZC4gVG91Y2hpbmcgYmxvYnMgYXJlCiAgICBzZXBhcmF0ZWQgYnkgZGVibGVuZGluZywgd2hpY2ggaXMgd2hhdCBzdG9wcyBvbmUgYnJpZ2h0IGNvbXBsZXggYmVpbmcgY291bnRlZAogICAgb25jZSB3aGVyZSBhIHBlcnNvbiBjbGlja2luZyBieSBleWUgd291bGQgaGF2ZSBjb3VudGVkIHRocmVlLgoKICAgIHRocmVzaG9sZF9zaWdtYSBpcyBpbiB1bml0cyBvZiB0aGUgbG9jYWwgYmFja2dyb3VuZCBub2lzZSwgc28gdGhlIHNhbWUgbnVtYmVyCiAgICBtZWFucyB0aGUgc2FtZSB0aGluZyBvbiBhbnkgZnJhbWUuIEl0IGlzIHRoZSBtb3N0IGNvbnNlcXVlbnRpYWwgY2hvaWNlIGluIHRoZQogICAgd2hvbGUgcGlwZWxpbmUsIHdoaWNoIGlzIHdoeSBpdCBpcyB3cml0dGVuIGludG8gZXZlcnkgb3V0cHV0IGZpbGUuCiAgICAiIiIKICAgIGZyb20gcGhvdHV0aWxzLmJhY2tncm91bmQgaW1wb3J0IEJhY2tncm91bmQyRCwgTWVkaWFuQmFja2dyb3VuZAogICAgZnJvbSBwaG90dXRpbHMuc2VnbWVudGF0aW9uIGltcG9ydCBkZXRlY3Rfc291cmNlcywgZGVibGVuZF9zb3VyY2VzLCBTb3VyY2VDYXRhbG9nCiAgICBmcm9tIGFzdHJvcHkuY29udm9sdXRpb24gaW1wb3J0IEdhdXNzaWFuMkRLZXJuZWwsIGNvbnZvbHZlCgogICAgYm94ID0gYmFja2dyb3VuZF9ib3gobmV0LnNoYXBlLCBib3hfc2l6ZSwgcjI1X3B4KQogICAgYmtnID0gQmFja2dyb3VuZDJEKG5ldCwgYm94X3NpemU9Ym94LCBmaWx0ZXJfc2l6ZT0zLAogICAgICAgICAgICAgICAgICAgICAgIGJrZ19lc3RpbWF0b3I9TWVkaWFuQmFja2dyb3VuZCgpLCBtYXNrPW1hc2spCiAgICByZXNpZCA9IG5ldCAtIGJrZy5iYWNrZ3JvdW5kCgogICAga2VybmVsID0gR2F1c3NpYW4yREtlcm5lbCgyLjApCiAgICBzbW9vdGhlZCA9IGNvbnZvbHZlKHJlc2lkLCBrZXJuZWwsIG1hc2s9bWFzaykKCiAgICAjIFRoZSB0aHJlc2hvbGQgaGFzIHRvIGJlIG1lYXN1cmVkIG9uIHRoZSBpbWFnZSBpdCBpcyBhcHBsaWVkIHRvLiBTbW9vdGhpbmcKICAgICMgYXZlcmFnZXMgbmVpZ2hib3VyaW5nIHBpeGVscywgc28gaXQgbG93ZXJzIHRoZSBub2lzZSBieSBhYm91dCBhIGZhY3RvciBvZgogICAgIyBzZXZlbiBmb3IgdGhpcyBrZXJuZWw7IGNvbXBhcmluZyB0aGUgc21vb3RoZWQgaW1hZ2UgYWdhaW5zdCB0aGUgdW5zbW9vdGhlZAogICAgIyBub2lzZSAtIHdoaWNoIGlzIHdoYXQgdGhlIGZpcnN0IHZlcnNpb24gb2YgdGhpcyBkaWQgLSBtYWRlIGEgbm9taW5hbCAiMwogICAgIyBzaWdtYSIgYmVoYXZlIGxpa2UgdHdlbnR5LCBhbmQgcXVpZXRseSB0aHJldyBhd2F5IG1vc3Qgb2YgdGhlIGZhaW50IHJlZ2lvbnMuCiAgICBzbW9vdGhfYmtnID0gQmFja2dyb3VuZDJEKHNtb290aGVkLCBib3hfc2l6ZT1ib3gsIGZpbHRlcl9zaXplPTMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGJrZ19lc3RpbWF0b3I9TWVkaWFuQmFja2dyb3VuZCgpLCBtYXNrPW1hc2spCiAgICB0aHJlc2hvbGQgPSB0aHJlc2hvbGRfc2lnbWEgKiBzbW9vdGhfYmtnLmJhY2tncm91bmRfcm1zCgogICAgc2VnID0gZGV0ZWN0X3NvdXJjZXMoc21vb3RoZWQsIHRocmVzaG9sZCwgbnBpeGVscz1taW5fcGl4ZWxzLCBtYXNrPW1hc2spCiAgICBpZiBzZWcgaXMgTm9uZToKICAgICAgICByZXR1cm4gTm9uZSwgYmtnLCByZXNpZAoKICAgIGlmIGRlYmxlbmQ6CiAgICAgICAgIyBTcGxpdCByYXRoZXIgdGhhbiBsdW1wLiBBIGxhcmdlIGNvbXBsZXggaXMgdXN1YWxseSBzZXZlcmFsIHJlZ2lvbnMgdGhhdAogICAgICAgICMgaGFwcGVuIHRvIHRvdWNoLCBhbmQgYXQgdGhpcyBkaXN0YW5jZSBvbmUgYmxvYiBjYW4gYmUgaHVuZHJlZHMgb2YgcGFyc2VjcwogICAgICAgICMgYWNyb3NzLiBTcGxpdHRpbmcgaXQgY2hhbmdlcyB0aGUgY291bnQgYW5kIHRoZSBzaXplIGRpc3RyaWJ1dGlvbjsgaXQKICAgICAgICAjIGJhcmVseSBtb3ZlcyB0aGUgcmFkaWFsIHByb2ZpbGUsIGJlY2F1c2UgdGhlIHBpZWNlcyBzaXQgd2hlcmUgdGhlIHdob2xlCiAgICAgICAgIyBzYXQuIE9uIHRoZSBNODMgZnJhbWVzIHRoZXNlIHNldHRpbmdzIHR1cm5lZCAyNzcgYmxvYnMgaW50byAzNjQgd2l0aG91dAogICAgICAgICMgYWRkaW5nIGFueXRoaW5nIHRvIHRoZSBlbXB0eSBmaWVsZC4KICAgICAgICBzZWcgPSBkZWJsZW5kX3NvdXJjZXMoc21vb3RoZWQsIHNlZywgbnBpeGVscz1taW5fcGl4ZWxzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICBubGV2ZWxzPW5sZXZlbHMsIGNvbnRyYXN0PWNvbnRyYXN0LAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICBwcm9ncmVzc19iYXI9RmFsc2UpCgogICAgY2F0ID0gU291cmNlQ2F0YWxvZyhyZXNpZCwgc2VnLCBtYXNrPW1hc2spCiAgICByZXR1cm4gY2F0LCBia2csIHJlc2lkCgpkZWYgb2Zmc2V0X2Zyb21fY2F0YWxvZ3VlKHJlZmVyZW5jZSwgaW1hZ2UsIHBvc2l0aW9ucywgYm94PTExLCBtYXhfc2hpZnRfcHg9NC4wKToKICAgICIiIlRoZSBzaGlmdCBiZXR3ZWVuIHR3byBmcmFtZXMsIG1lYXN1cmVkIG9uIG9uZSBhZ3JlZWQgbGlzdCBvZiBzdGFycy4KCiAgICBXSFkgTk9UIEpVU1QgREVURUNUIFNUQVJTIElOIEJPVEgKICAgIEJlY2F1c2UgdGhlIGJyaWdodGVzdCB0aHJlZSBodW5kcmVkIHNvdXJjZXMgaW4gYSBuYXJyb3ctYmFuZCBmcmFtZSBhcmUgbm90IHRoZQogICAgYnJpZ2h0ZXN0IHRocmVlIGh1bmRyZWQgaW4gYSBicm9hZC1iYW5kIG9uZSAtIGRpZmZlcmVudCBmaWx0ZXJzIHNlZSBkaWZmZXJlbnQKICAgIHN0YXJzIC0gc28gdHdvIGluZGVwZW5kZW50bHkgZGV0ZWN0ZWQgbGlzdHMgb3ZlcmxhcCBwb29ybHkgYW5kIHRoZSBwYWlycyB0aGF0CiAgICBkbyBtYXRjaCBhcmUgdGhlIG9uZXMgdGhhdCBoYXBwZW5lZCB0byBzdXJ2aXZlLiBPbiBhbiBNMzMgcGFpciB0aGlzIGxlZnQgMjEKICAgIHVzYWJsZSBzdGFycyBvdXQgb2Ygc2l4IGh1bmRyZWQsIHdpdGggYSBzY2F0dGVyIG9mIHR3byBwaXhlbHMuCgogICAgR2l2ZW4gYSBsaXN0IG9mIHBvc2l0aW9ucyBmcm9tIGEgY2F0YWxvZ3VlLCB0aGUgcXVlc3Rpb24gZGlzYXBwZWFyczogdGhlIHNhbWUKICAgIHN0YXIgaXMgY2VudHJvaWRlZCBpbiBib3RoIGZyYW1lcywgYW5kIGV2ZXJ5IG9uZSBvZiB0aGVtIGNvbnRyaWJ1dGVzLgoKICAgIFdIWSBUSEUgU0tZIENPTUVTIE9GRiBFQUNIIFNUQU1QIEZJUlNUCiAgICBTYW1lIHJlYXNvbiBhcyBpbiBjZW50cmVfb25fc3RhcnMsIGFuZCB0aGUgY29uc2VxdWVuY2UgaGVyZSBpcyB3b3JzZS4gQSBjZW50cmUKICAgIG9mIGxpZ2h0IGNvbXB1dGVkIHdpdGggdGhlIHNreSBzdGlsbCBpbiB0aGUgc3RhbXAgaXMgZG9taW5hdGVkIGJ5IHRoZSBza3ksIHNvCiAgICBpdCBsYW5kcyBuZWFyIHRoZSBtaWRkbGUgb2YgdGhlIGJveCB3aGF0ZXZlciB0aGUgc3RhciBkb2VzIC0gaW4gYm90aCBmcmFtZXMuCiAgICBUaGUgdHdvIGFuc3dlcnMgdGhlbiBhZ3JlZSBiZWF1dGlmdWxseSBhbmQgdGhlaXIgZGlmZmVyZW5jZSBpcyBjbG9zZSB0byB6ZXJvLAogICAgc28gdGhpcyBmdW5jdGlvbiByZXBvcnRlZCAiYWxyZWFkeSBhbGlnbmVkIiBvbiBhIHBhaXIgdGhhdCB3YXMgYSBmdWxsIHBpeGVsCiAgICBhcGFydC4gT24gTTgzIHRoYXQgcGl4ZWwgaXMgd2hhdCBsZWZ0IGV2ZXJ5IHN0YXIgYXMgYSBkYXJrIGNvcmUgYmVzaWRlIGEgYnJpZ2h0CiAgICBjcmVzY2VudCBpbiB0aGUgc3VidHJhY3RlZCBmcmFtZS4KICAgICIiIgogICAgaWYgcG9zaXRpb25zIGlzIE5vbmUgb3IgbGVuKHBvc2l0aW9ucykgPCAxMDoKICAgICAgICByZXR1cm4gTm9uZQoKICAgIHBvcyA9IG5wLmFzYXJyYXkocG9zaXRpb25zLCBkdHlwZT1mbG9hdCkKICAgIGNhID0gX2NlbnRyb2lkX2xvY2FsbHkocmVmZXJlbmNlLCBwb3MsIGJveCkKICAgIGNiID0gX2NlbnRyb2lkX2xvY2FsbHkoaW1hZ2UsIHBvcywgYm94KQogICAgaWYgY2EgaXMgTm9uZSBvciBjYiBpcyBOb25lOgogICAgICAgIHJldHVybiBOb25lCgogICAgZHgsIGR5ID0gY2JbOiwgMF0gLSBjYVs6LCAwXSwgY2JbOiwgMV0gLSBjYVs6LCAxXQogICAgb2sgPSAobnAuaXNmaW5pdGUoZHgpICYgbnAuaXNmaW5pdGUoZHkpCiAgICAgICAgICAmIChucC5hYnMoZHgpIDwgbWF4X3NoaWZ0X3B4KSAmIChucC5hYnMoZHkpIDwgbWF4X3NoaWZ0X3B4KSkKICAgIGlmIG9rLnN1bSgpIDwgMTA6CiAgICAgICAgcmV0dXJuIE5vbmUKCiAgICBteCwgbXkgPSBmbG9hdChucC5tZWRpYW4oZHhbb2tdKSksIGZsb2F0KG5wLm1lZGlhbihkeVtva10pKQogICAgcmV0dXJuIG14LCBteSwgewogICAgICAgICJuX3BhaXJzIjogaW50KG9rLnN1bSgpKSwKICAgICAgICAic2NhdHRlcl94IjogZmxvYXQoMS40ODI2ICogbnAubWVkaWFuKG5wLmFicyhkeFtva10gLSBteCkpKSwKICAgICAgICAic2NhdHRlcl95IjogZmxvYXQoMS40ODI2ICogbnAubWVkaWFuKG5wLmFicyhkeVtva10gLSBteSkpKSwKICAgICAgICAiZnJvbV9jYXRhbG9ndWUiOiBUcnVlLAogICAgfQoKZGVmIG1lYXN1cmVfb2Zmc2V0KHJlZmVyZW5jZSwgaW1hZ2UsIGZ3aG09NS41LCB0aHJlc2hvbGRfc2lnbWE9MjAuMCwgbl9zdGFycz0zMDAsCiAgICAgICAgICAgICAgICAgICBtYXhfbWF0Y2hfcHg9NC4wKToKICAgICIiIlRoZSBzaGlmdCBzdGlsbCBsZWZ0IGJldHdlZW4gdHdvIGZyYW1lcyBhZnRlciB0aGV5IHdlcmUgcHV0IG9uIG9uZSBncmlkLgoKICAgIFdIWSBUSElTIElTIE5FRURFRCBBRlRFUiBBIFdDUyBBTElHTk1FTlQKICAgIFJlc2FtcGxpbmcgdGhyb3VnaCB0d28gcGxhdGUgc29sdXRpb25zIGlzIG9ubHkgYXMgZ29vZCBhcyB0aG9zZSBzb2x1dGlvbnMuIEVhY2gKICAgIHdhcyBmaXR0ZWQgc2VwYXJhdGVseSwgb24gYSBkaWZmZXJlbnQgZXhwb3N1cmUsIGFuZCBlYWNoIGNhcnJpZXMgaXRzIG93biBlcnJvcgogICAgb2YgYSBmcmFjdGlvbiBvZiBhbiBhcmNzZWNvbmQuIFdoYXQgc3Vydml2ZXMgaXMgYSBzbWFsbCBzeXN0ZW1hdGljIHNoaWZ0IC0gb24KICAgIHRoZSBmcmFtZXMgdGhpcyB3YXMgd3JpdHRlbiBmb3IsIDAuNiBvZiBhIHBpeGVsIC0gYW5kIGl0IGlzIHZpc2libGU6IGV2ZXJ5IHN0YXIKICAgIGxlYXZlcyBhIGRhcmstYW5kLWJyaWdodCBwYWlyIGluIHRoZSBzdWJ0cmFjdGVkIGltYWdlIGluc3RlYWQgb2YgdmFuaXNoaW5nLCBhbmQKICAgIHRoZSBkZXRlY3RvciByZXBvcnRzIHRob3NlIHBhaXJzIGFzIHJlZ2lvbnMuCgogICAgQSBzaGlmdCBpcyBtZWFzdXJhYmxlIHRvIGZhciBiZXR0ZXIgdGhhbiBhIHBpeGVsIGJlY2F1c2UgaXQgaXMgY29tbW9uIHRvIGV2ZXJ5CiAgICBzdGFyIGluIHRoZSBmcmFtZS4gSGVyZSBpdCBpcyB0aGUgbWVkaWFuIG92ZXIgYSBmZXcgaHVuZHJlZCBvZiB0aGVtLCBhbmQgdGhlCiAgICBzY2F0dGVyIGFib3V0IHRoYXQgbWVkaWFuIHNheXMgd2hldGhlciBpdCByZWFsbHkgaXMgb25lIHJpZ2lkIHNoaWZ0IG9yIHdoZXRoZXIKICAgIHNvbWV0aGluZyBlbHNlIGlzIHdyb25nLgoKICAgIFJldHVybnMgKGR4LCBkeSwgZGlhZ25vc3RpY3MpLiBkeCwgZHkgYXJlIGhvdyBmYXIgYGltYWdlYCBzaXRzIGZyb20KICAgIGByZWZlcmVuY2VgLCBzbyBzaGlmdGluZyBpdCBieSB0aGUgbmVnYXRpdmUgb2YgdGhlbSBicmluZ3MgdGhlIHR3byB0b2dldGhlci4KICAgICIiIgogICAgaW1wb3J0IG51bXB5IGFzIG5wCiAgICBmcm9tIHBob3R1dGlscy5kZXRlY3Rpb24gaW1wb3J0IERBT1N0YXJGaW5kZXIKICAgIGZyb20gYXN0cm9weS5zdGF0cyBpbXBvcnQgc2lnbWFfY2xpcHBlZF9zdGF0cwogICAgZnJvbSBzY2lweS5zcGF0aWFsIGltcG9ydCBjS0RUcmVlCgogICAgZGVmIGJyaWdodF9zdGFycyhpbWcpOgogICAgICAgIF8sIG1lZCwgc2QgPSBzaWdtYV9jbGlwcGVkX3N0YXRzKGltZywgc2lnbWE9My4wKQogICAgICAgIHRhYmxlID0gREFPU3RhckZpbmRlcihmd2htPWZ3aG0sIHRocmVzaG9sZD10aHJlc2hvbGRfc2lnbWEgKiBzZCkoaW1nIC0gbWVkKQogICAgICAgIGlmIHRhYmxlIGlzIE5vbmUgb3IgbGVuKHRhYmxlKSA9PSAwOgogICAgICAgICAgICByZXR1cm4gbnAuZW1wdHkoKDAsIDIpKQogICAgICAgIHRhYmxlLnNvcnQoImZsdXgiKQogICAgICAgIHRhYmxlID0gdGFibGVbOjotMV1bOm5fc3RhcnNdCiAgICAgICAgcmV0dXJuIG5wLnRyYW5zcG9zZSgobnAuYXNhcnJheSh0YWJsZVsieGNlbnRyb2lkIl0sIGR0eXBlPWZsb2F0KSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICBucC5hc2FycmF5KHRhYmxlWyJ5Y2VudHJvaWQiXSwgZHR5cGU9ZmxvYXQpKSkKCiAgICBhLCBiID0gYnJpZ2h0X3N0YXJzKHJlZmVyZW5jZSksIGJyaWdodF9zdGFycyhpbWFnZSkKICAgIGlmIGxlbihhKSA8IDEwIG9yIGxlbihiKSA8IDEwOgogICAgICAgIHJldHVybiAwLjAsIDAuMCwgeyJuX3BhaXJzIjogMCwKICAgICAgICAgICAgICAgICAgICAgICAgICAid2h5IjogInRvbyBmZXcgc3RhcnMgdG8gbWVhc3VyZSBhIHNoaWZ0In0KCiAgICBkaXN0YW5jZSwgaW5kZXggPSBjS0RUcmVlKGIpLnF1ZXJ5KGEsIGs9MSkKICAgIG9rID0gZGlzdGFuY2UgPCBtYXhfbWF0Y2hfcHgKICAgIGlmIG9rLnN1bSgpIDwgMTA6CiAgICAgICAgcmV0dXJuIDAuMCwgMC4wLCB7Im5fcGFpcnMiOiBpbnQob2suc3VtKCkpLAogICAgICAgICAgICAgICAgICAgICAgICAgICJ3aHkiOiAidG9vIGZldyBzdGFycyBjbG9zZSBlbm91Z2ggdG8gcGFpciB1cCJ9CgogICAgZHggPSBiW2luZGV4W29rXSwgMF0gLSBhW29rLCAwXQogICAgZHkgPSBiW2luZGV4W29rXSwgMV0gLSBhW29rLCAxXQogICAgbXgsIG15ID0gZmxvYXQobnAubWVkaWFuKGR4KSksIGZsb2F0KG5wLm1lZGlhbihkeSkpCiAgICByZXR1cm4gbXgsIG15LCB7CiAgICAgICAgIm5fcGFpcnMiOiBpbnQob2suc3VtKCkpLAogICAgICAgICJzY2F0dGVyX3giOiBmbG9hdCgxLjQ4MjYgKiBucC5tZWRpYW4obnAuYWJzKGR4IC0gbXgpKSksCiAgICAgICAgInNjYXR0ZXJfeSI6IGZsb2F0KDEuNDgyNiAqIG5wLm1lZGlhbihucC5hYnMoZHkgLSBteSkpKSwKICAgIH0KCmRlZiBzaGlmdF9mcmFtZShpbWFnZSwgZHgsIGR5KToKICAgICIiIk1vdmUgYSBmcmFtZSBieSBhIGZyYWN0aW9uIG9mIGEgcGl4ZWwsIHRvIHVuZG8gYSBtZWFzdXJlZCBvZmZzZXQuCgogICAgVGhlIGJsYW5rIG1hcmdpbiBsZWZ0IGJ5IGFuIGVhcmxpZXIgcmVzYW1wbGluZyBpcyBub3QgYSBudW1iZXIsIGFuZCBhIGN1YmljCiAgICBzcGxpbmUgcnVuIG92ZXIgaXQgd291bGQgc3ByZWFkIHRoYXQgbm90aGluZ25lc3MgYWNyb3NzIHRoZSB3aG9sZSBmcmFtZSAtIHRoZQogICAgZmlyc3QgdmVyc2lvbiBvZiB0aGlzIGRpZCBleGFjdGx5IHRoYXQsIGFuZCBldmVyeSBtZWFzdXJlbWVudCBkb3duc3RyZWFtIGNhbWUKICAgIGJhY2sgZW1wdHkuIFNvIHRoZSBnYXBzIGFyZSBmaWxsZWQgYmVmb3JlIHRoZSBzaGlmdCBhbmQgcHV0IGJhY2sgYWZ0ZXIsIHdoaWNoCiAgICBrZWVwcyB0aGVtIGV4YWN0bHkgd2hlcmUgdGhleSB3ZXJlLgogICAgIiIiCiAgICBpbXBvcnQgbnVtcHkgYXMgbnAKICAgIGZyb20gc2NpcHkubmRpbWFnZSBpbXBvcnQgc2hpZnQgYXMgbmRzaGlmdAoKICAgIGtub3duID0gbnAuaXNmaW5pdGUoaW1hZ2UpCiAgICBtb3ZlZCA9IG5kc2hpZnQobnAud2hlcmUoa25vd24sIGltYWdlLCAwLjApLCAoLWR5LCAtZHgpLCBvcmRlcj0zLCBtb2RlPSJuZWFyZXN0IikKICAgIHN0aWxsX2tub3duID0gbmRzaGlmdChrbm93bi5hc3R5cGUoZmxvYXQpLCAoLWR5LCAtZHgpLCBvcmRlcj0xLCBtb2RlPSJuZWFyZXN0IikKICAgIG1vdmVkW3N0aWxsX2tub3duIDwgMC41XSA9IG5wLm5hbgogICAgcmV0dXJuIG1vdmVkCgpkZWYgcmVmaW5lX2FsaWdubWVudChyZWZlcmVuY2UsIGltYWdlLCBsYWJlbCwgbWluX3NoaWZ0X3B4PTAuMDUsIHBvc2l0aW9ucz1Ob25lKToKICAgICIiIk1lYXN1cmUgdGhlIGxlZnRvdmVyIHNoaWZ0IGFnYWluc3QgYSByZWZlcmVuY2UgZnJhbWUgYW5kIHRha2UgaXQgb3V0LgoKICAgIFdoZW4gYSBsaXN0IG9mIGNhdGFsb2d1ZSBzdGFyIHBvc2l0aW9ucyBpcyBhdmFpbGFibGUgdGhlIHNoaWZ0IGlzIG1lYXN1cmVkIG9uCiAgICB0aG9zZSwgdGhlIHNhbWUgc3RhcnMgaW4gYm90aCBmcmFtZXMuIE90aGVyd2lzZSB0aGUgZnJhbWVzIGFyZSBhc2tlZCB0byBzdXBwbHkKICAgIHRoZWlyIG93biBzdGFycywgd2hpY2ggd29ya3MgYmV0d2VlbiBleHBvc3VyZXMgdGhyb3VnaCBvbmUgZmlsdGVyIGFuZCB3b3JrcwogICAgcG9vcmx5IGJldHdlZW4gdHdvIGRpZmZlcmVudCBvbmVzLgogICAgIiIiCiAgICBmcm9tX2NhdGFsb2d1ZSA9IG9mZnNldF9mcm9tX2NhdGFsb2d1ZShyZWZlcmVuY2UsIGltYWdlLCBwb3NpdGlvbnMpCiAgICBpZiBmcm9tX2NhdGFsb2d1ZSBpcyBub3QgTm9uZToKICAgICAgICBkeCwgZHksIGRpYWcgPSBmcm9tX2NhdGFsb2d1ZQogICAgZWxzZToKICAgICAgICBkeCwgZHksIGRpYWcgPSBtZWFzdXJlX29mZnNldChyZWZlcmVuY2UsIGltYWdlKQogICAgaWYgZGlhZy5nZXQoIm5fcGFpcnMiLCAwKSA8IDEwOgogICAgICAgIHByaW50KCIgIHt9OiBjb3VsZCBub3QgY2hlY2sgdGhlIGFsaWdubWVudCAtIHt9Ii5mb3JtYXQoCiAgICAgICAgICAgIGxhYmVsLCBkaWFnLmdldCgid2h5IiwgIm5vdCBlbm91Z2ggc3RhcnMiKSkpCiAgICAgICAgcmV0dXJuIGltYWdlCiAgICBzaXplID0gKGR4ICogZHggKyBkeSAqIGR5KSAqKiAwLjUKICAgIGlmIHNpemUgPCBtaW5fc2hpZnRfcHg6CiAgICAgICAgcHJpbnQoIiAge306IGFscmVhZHkgYWxpZ25lZCB0byB7Oi4yZn0gcHgsIG5vdGhpbmcgdG8gY29ycmVjdCIuZm9ybWF0KAogICAgICAgICAgICBsYWJlbCwgc2l6ZSkpCiAgICAgICAgcmV0dXJuIGltYWdlCiAgICBwcmludCgiICB7fToge317fSBzdGFycyBzYXkgaXQgc2l0cyAoezorLjJmfSwgezorLjJmfSkgcHggZnJvbSB0aGUgcmVmZXJlbmNlICIKICAgICAgICAgICIoc2NhdHRlciB7Oi4yZn0sIHs6LjJmfSk7IHNoaWZ0ZWQgYmFjayIuZm9ybWF0KAogICAgICAgICAgICAgIGxhYmVsLCAiY2F0YWxvZ3VlICIgaWYgZGlhZy5nZXQoImZyb21fY2F0YWxvZ3VlIikgZWxzZSAiIiwKICAgICAgICAgICAgICBkaWFnWyJuX3BhaXJzIl0sIGR4LCBkeSwKICAgICAgICAgICAgICBkaWFnWyJzY2F0dGVyX3giXSwgZGlhZ1sic2NhdHRlcl95Il0pKQogICAgcmV0dXJuIHNoaWZ0X2ZyYW1lKGltYWdlLCBkeCwgZHkpCgpkZWYgc3Rhcl9wb3NpdGlvbnMocmVkLCBtYXNrPU5vbmUsIG5fc3RhcnM9NDAwLCB0aHJlc2hvbGRfc2lnbWE9MTUuMCk6CiAgICAiIiJQb3NpdGlvbnMgb2YgdGhlIGJyaWdodCBwb2ludCBzb3VyY2VzIGluIHRoZSBicm9hZC1iYW5kIGZyYW1lLgoKICAgIEV2ZW4gYWZ0ZXIgbWF0Y2hpbmcgYW5kIHN1YnRyYWN0aW9uIGEgc3RhciByYXJlbHkgdmFuaXNoZXMgcGVyZmVjdGx5OiBhIHNtYWxsCiAgICBlcnJvciBpbiB0aGUgZmFjdG9yIG9yIHRoZSBibHVycmluZyBsZWF2ZXMgYSByZXNpZHVhbCBleGFjdGx5IHdoZXJlIHRoZSBzdGFyCiAgICB3YXMuIFRob3NlIHJlc2lkdWFscyBhcmUgY29tcGFjdCBhbmQgYnJpZ2h0LCBzbyB0aGUgZGV0ZWN0b3IgcmVwb3J0cyB0aGVtIGFzCiAgICBISUkgcmVnaW9ucy4gS25vd2luZyB3aGVyZSB0aGUgc3RhcnMgd2VyZSBsZXRzIHVzIHRocm93IHRoZW0gb3V0LgogICAgIiIiCiAgICBmcm9tIHBob3R1dGlscy5kZXRlY3Rpb24gaW1wb3J0IERBT1N0YXJGaW5kZXIKICAgIGZyb20gYXN0cm9weS5zdGF0cyBpbXBvcnQgc2lnbWFfY2xpcHBlZF9zdGF0cwoKICAgIG1lYW4sIG1lZGlhbiwgc3RkID0gc2lnbWFfY2xpcHBlZF9zdGF0cyhyZWQsIHNpZ21hPTMuMCwgbWFzaz1tYXNrKQogICAgZmluZGVyID0gREFPU3RhckZpbmRlcihmd2htPTQuMCwgdGhyZXNob2xkPXRocmVzaG9sZF9zaWdtYSAqIHN0ZCkKICAgIHRibCA9IGZpbmRlcihyZWQgLSBtZWRpYW4sIG1hc2s9bWFzaykKICAgIGlmIHRibCBpcyBOb25lIG9yIGxlbih0YmwpID09IDA6CiAgICAgICAgcmV0dXJuIG5wLmVtcHR5KCgwLCAyKSkKICAgIHRibC5zb3J0KCJmbHV4IikKICAgIHRibCA9IHRibFs6Oi0xXVs6bl9zdGFyc10KICAgIHJldHVybiBucC50cmFuc3Bvc2UoKG5wLmFzYXJyYXkodGJsWyJ4Y2VudHJvaWQiXSwgZHR5cGU9ZmxvYXQpLAogICAgICAgICAgICAgICAgICAgICAgICAgbnAuYXNhcnJheSh0YmxbInljZW50cm9pZCJdLCBkdHlwZT1mbG9hdCkpKQoKZGVmIHN1YnRyYWN0X2NvbnRpbnV1bShoYSwgcmVkLCBmYWN0b3IsIG1hc2s9Tm9uZSk6CiAgICAiIiJmYWN0b3IgeCBILWFscGhhIC0gUiwgd2l0aCBlYWNoIGZyYW1lJ3Mgb3duIHNreSB0YWtlbiBvZmYgZmlyc3QuCgogICAgV0hZIFRIRSBTS1kgQ09NRVMgT0ZGIEJFRk9SRSBUSEUgTVVMVElQTElDQVRJT04KICAgIEJlY2F1c2UgdGhlIGZhY3RvciBpcyBtZWFudCB0byBzY2FsZSB0aGUgc3RhcmxpZ2h0LCBhbmQgdGhlIGZyYW1lIGhvbGRzIHR3bwogICAgdGhpbmdzOiBzdGFybGlnaHQgYW5kIHNreS4gTXVsdGlwbHlpbmcgdGhlIHdob2xlIGZyYW1lIG11bHRpcGxpZXMgdGhlIHNreSB3aXRoCiAgICBpdC4gT24gYW4gTTMzIHBhaXIgdGhlIEgtYWxwaGEgc2t5IHdhcyAxMi42IGNvdW50cyBhbmQgdGhlIFIgc2t5IDk1LjQsIGFuZCB0aGUKICAgIGZhY3RvciAyNS43IC0gc28gdGhlIHJlc3VsdCBzYXQgb24gYSBwZWRlc3RhbCBvZiAyNS43IHggMTIuNiAtIDk1LjQgPSAyMjkKICAgIGNvdW50cywgc2V2ZW50ZWVuIHRpbWVzIHRoZSBub2lzZSBvZiB0aGUgaW1hZ2UuCgogICAgRGV0ZWN0aW9uIHN1cnZpdmVkIHRoYXQsIGJlY2F1c2UgaXQgZXN0aW1hdGVzIGEgYmFja2dyb3VuZCBhbmQgcmVtb3ZlcyBpdAogICAgYmVmb3JlIGxvb2tpbmcgZm9yIGFueXRoaW5nLiBXaGF0IGRpZCBub3Qgc3Vydml2ZSB3YXMgdGhlIHBpY3R1cmU6IHRoZSB3aG9sZQogICAgZHluYW1pYyByYW5nZSB3ZW50IGludG8gdGhlIHBlZGVzdGFsLCBmYWludCBlbWlzc2lvbiB2YW5pc2hlZCBpbnRvIGEgZmxhdCBncmV5LAogICAgYW5kIGFueSBkaWZmZXJlbmNlIGluIGZsYXQtZmllbGRpbmcgYmV0d2VlbiB0aGUgdHdvIGZpbHRlcnMgd2FzIGFtcGxpZmllZCBieQogICAgdGhlIGZhY3RvciBhbG9uZyB3aXRoIGV2ZXJ5dGhpbmcgZWxzZS4gQW55b25lIGp1ZGdpbmcgdGhlIHN1YnRyYWN0aW9uIGJ5IGV5ZSAtCiAgICB3aGljaCBpcyB3aGF0IHRoaXMgdG9vbCBhc2tzIGZvciB3aGVuZXZlciB0aGUgZmFjdG9yIGxvb2tzIGRvdWJ0ZnVsIC0gd2FzCiAgICBsb29raW5nIGF0IGFuIGltYWdlIHJ1aW5lZCBiZWZvcmUgdGhleSBzYXcgaXQuCgogICAgU2t5IGlzIG1lYXN1cmVkIHdpdGggY2xpcHBpbmcsIHNvIHRoZSBnYWxheHkgYW5kIHRoZSBzdGFycyBkbyBub3QgZHJhZyBpdC4KICAgICIiIgogICAgZnJvbSBhc3Ryb3B5LnN0YXRzIGltcG9ydCBzaWdtYV9jbGlwcGVkX3N0YXRzCgogICAgc2t5X2hhID0gZmxvYXQoc2lnbWFfY2xpcHBlZF9zdGF0cyhoYSwgc2lnbWE9My4wLCBtYXNrPW1hc2spWzFdKQogICAgc2t5X3JlZCA9IGZsb2F0KHNpZ21hX2NsaXBwZWRfc3RhdHMocmVkLCBzaWdtYT0zLjAsIG1hc2s9bWFzaylbMV0pCiAgICBuZXQgPSBmYWN0b3IgKiAoaGEgLSBza3lfaGEpIC0gKHJlZCAtIHNreV9yZWQpCiAgICByZXR1cm4gbmV0LCB7InNreV9oYSI6IHNreV9oYSwgInNreV9yZWQiOiBza3lfcmVkLAogICAgICAgICAgICAgICAgICJwZWRlc3RhbF9hdm9pZGVkIjogZmFjdG9yICogc2t5X2hhIC0gc2t5X3JlZH0KCmRlZiBsaW5lX2V4Y2VzcyhuZXQsIHJlZCwgeCwgeSwgcmFkaXVzX3B4PTUuMCk6CiAgICAiIiJIb3cgbXVjaCBsaW5lIGVtaXNzaW9uIGVhY2ggZGV0ZWN0aW9uIGhhcywgbmV4dCB0byBpdHMgb3duIGNvbnRpbnV1bS4KCiAgICBUaGUgcmF0aW8gb2Ygd2hhdCBpcyBsZWZ0IGFmdGVyIHRoZSBzdWJ0cmFjdGlvbiB0byB3aGF0IHRoZSBjb250aW51dW0gZnJhbWUKICAgIGhvbGRzIGF0IHRoZSBzYW1lIHBsYWNlLiBBIGZvcmVncm91bmQgc3RhciBoYXMgbm90aGluZyBsZWZ0IG9uY2UgdGhlCiAgICBzdWJ0cmFjdGlvbiBpcyByaWdodCwgc28gaXRzIHJhdGlvIHNpdHMgbmVhciB6ZXJvLiBBbiBISUkgcmVnaW9uIGlzIGVtaXNzaW9uCiAgICB3aXRoIG9ubHkgYSBtb2Rlc3QgY29udGludXVtIHVuZGVyIGl0LCBzbyBpdHMgcmF0aW8gaXMgb2Ygb3JkZXIgb25lIG9yIG1vcmUuCiAgICAiIiIKICAgIGZyb20gcGhvdHV0aWxzLmFwZXJ0dXJlIGltcG9ydCBDaXJjdWxhckFwZXJ0dXJlLCBhcGVydHVyZV9waG90b21ldHJ5CgogICAgcG9zaXRpb25zID0gbnAuY29sdW1uX3N0YWNrKFtucC5hc2FycmF5KHgsIGR0eXBlPWZsb2F0KSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbnAuYXNhcnJheSh5LCBkdHlwZT1mbG9hdCldKQogICAgaWYgbm90IGxlbihwb3NpdGlvbnMpOgogICAgICAgIHJldHVybiBucC56ZXJvcygwKQogICAgYXAgPSBDaXJjdWxhckFwZXJ0dXJlKHBvc2l0aW9ucywgcj1yYWRpdXNfcHgpCiAgICBmX25ldCA9IG5wLmFzYXJyYXkoYXBlcnR1cmVfcGhvdG9tZXRyeShucC5uYW5fdG9fbnVtKG5ldCksIGFwKVsiYXBlcnR1cmVfc3VtIl0sCiAgICAgICAgICAgICAgICAgICAgICAgZHR5cGU9ZmxvYXQpCiAgICBmX3JlZCA9IG5wLmFzYXJyYXkoYXBlcnR1cmVfcGhvdG9tZXRyeShucC5uYW5fdG9fbnVtKHJlZCksIGFwKVsiYXBlcnR1cmVfc3VtIl0sCiAgICAgICAgICAgICAgICAgICAgICAgZHR5cGU9ZmxvYXQpCiAgICB3aXRoIG5wLmVycnN0YXRlKGRpdmlkZT0iaWdub3JlIiwgaW52YWxpZD0iaWdub3JlIik6CiAgICAgICAgcmV0dXJuIGZfbmV0IC8gbnAud2hlcmUobnAuYWJzKGZfcmVkKSA+IDEuMCwgbnAuYWJzKGZfcmVkKSwgbnAubmFuKQoKZGVmIGZpbHRlcl9kZXRlY3Rpb25zKHgsIHksIGNlbnRyZSwgcl9tYXhfcHg9Tm9uZSwgc3Rhcl94eT1Ob25lLCBzdGFyX3JhZGl1c19weD02LjAsCiAgICAgICAgICAgICAgICAgICAgICBleGNlc3M9Tm9uZSwgbWluX2V4Y2Vzcz0wLjUsCiAgICAgICAgICAgICAgICAgICAgICBtaW5vcl9heGlzPU5vbmUsIGVsb25nYXRpb249Tm9uZSwKICAgICAgICAgICAgICAgICAgICAgIG1pbl93aWR0aF9weD0xLjUsIG1heF9lbG9uZ2F0aW9uPTIuNSk6CiAgICAiIiJEcm9wIGRldGVjdGlvbnMgdGhhdCBjYW5ub3QgYmUgSElJIHJlZ2lvbnMgYmVsb25naW5nIHRvIHRoaXMgZ2FsYXh5LgoKICAgIFR3byBjdXRzLCBib3RoIGJsdW50IGFuZCBib3RoIG5lY2Vzc2FyeS4KCiAgICBPdXRzaWRlIHRoZSBnYWxheHk6IEhJSSByZWdpb25zIGxpdmUgaW4gdGhlIGRpc2ssIGEgZnJhbWUgaXMgdXN1YWxseSBtdWNoIHdpZGVyCiAgICB0aGFuIHRoZSBnYWxheHksIGFuZCBldmVyeXRoaW5nIGZvdW5kIG91dCBpbiB0aGUgZW1wdHkgY29ybmVycyBpcyBub2lzZSBvciBhbgogICAgYXJ0ZWZhY3QuIEN1dHRpbmcgYXQgdGhlIGlzb3Bob3RhbCByYWRpdXMgUjI1IHJlbW92ZXMgdGhlbSB3aXRob3V0IHRvdWNoaW5nIHRoZQogICAgZGlzaywgYW5kIFIyNSBpcyBhIHB1Ymxpc2hlZCBudW1iZXIgcmF0aGVyIHRoYW4gYSBqdWRnZW1lbnQgY2FsbC4KCiAgICBPbiB0b3Agb2YgYSBzdGFyOiBzdWJ0cmFjdGlvbiByZXNpZHVhbHMgc2l0IGV4YWN0bHkgd2hlcmUgdGhlIHN0YXJzIHdlcmUuCgogICAgV0hZIFRIRSBTVEFSIENVVCBORUVEUyBUSEUgU0VDT05EIFRFU1QKICAgIEEgYnJpZ2h0IEhJSSByZWdpb24gY29udGFpbnMgdGhlIHlvdW5nIGNsdXN0ZXIgdGhhdCBsaWdodHMgaXQsIGFuZCBhIHlvdW5nCiAgICBjbHVzdGVyIGlzIGEgcG9pbnQgc291cmNlIGluIHRoZSBjb250aW51dW0gZnJhbWUgLSBzbyBhIGN1dCB0aGF0IHJlbW92ZXMKICAgIGV2ZXJ5dGhpbmcgc2l0dGluZyBvbiBhIHBvaW50IHNvdXJjZSByZW1vdmVzIHRoZSBicmlnaHRlc3QgcmVnaW9ucyBpbiB0aGUKICAgIGdhbGF4eSBhbG9uZyB3aXRoIHRoZSBmb3JlZ3JvdW5kIHN0YXJzLiBPbiB0aGUgTTgzIGZyYW1lcyB0aGlzIHdhcyBtZWFzdXJlZDoKICAgIG9mIGZvcnR5IGRldGVjdGlvbnMgdGhlIHN0YXIgY3V0IHRocmV3IGF3YXkgaW5zaWRlIHRoZSBnYWxheHksIHR3ZW50eS1uaW5lIGhhZAogICAgcmVhbCBsaW5lIGVtaXNzaW9uLiBUaGVpciBlbWlzc2lvbiBpcyB3aGF0IHNlcGFyYXRlcyB0aGVtIC0gdGhlIHJlZ2lvbnMga2VwdAogICAgYnkgdGhlIHJlc3Qgb2YgdGhlIHBpcGVsaW5lIHNob3dlZCBhIHJhdGlvIGFyb3VuZCAyLjMsIGFuZCB0aGUgcG9pbnQgc291cmNlcwogICAgb3V0IGluIHRoZSBlbXB0eSBmaWVsZCwgd2hpY2ggcmVhbGx5IGFyZSBzdGFycywgYXJvdW5kIDAuMTcuIFNvIGEgZGV0ZWN0aW9uCiAgICBzaXR0aW5nIG9uIGEgcG9pbnQgc291cmNlIGlzIHN0aWxsIGRyb3BwZWQsIHVubGVzcyBpdCBpcyBlbWl0dGluZy4KICAgICIiIgogICAgeCA9IG5wLmFzYXJyYXkoeCwgZHR5cGU9ZmxvYXQpCiAgICB5ID0gbnAuYXNhcnJheSh5LCBkdHlwZT1mbG9hdCkKICAgIGtlZXAgPSBucC5vbmVzKHguc2l6ZSwgZHR5cGU9Ym9vbCkKICAgIHJlcG9ydCA9IHt9CgogICAgIyBTaGFwZS4gQW4gSElJIHJlZ2lvbiBpcyBhIGJvZHkgb2YgZ2FzLCBzbyBpdCBoYXMgd2lkdGg7IHdoYXQgaXMgbGVmdCBvdmVyCiAgICAjIGZyb20gYW4gaW1wZXJmZWN0IHN1YnRyYWN0aW9uIGlzIGEgdGhyZWFkIC0gYSBwaXhlbCBvciB0d28gYWNyb3NzLCBvZnRlbgogICAgIyBjdXJ2ZWQsIGZvbGxvd2luZyB0aGUgZWRnZSBvZiB3aGF0ZXZlciBkaWQgbm90IGNhbmNlbC4gTWVhc3VyZWQgb24gdGhlIE04MwogICAgIyBmcmFtZXMsIGJsb2JzIGluc2lkZSB0aGUgZ2FsYXh5IGFyZSB0eXBpY2FsbHkgMS45IHB4IHdpZGUgYW5kIGJsb2JzIG91dCBpbgogICAgIyB0aGUgZW1wdHkgZmllbGQsIHdoZXJlIG5vdGhpbmcgcmVhbCBjYW4gYmUsIDAuNSBweC4gVGhhdCBvbmUgbnVtYmVyIGRvZXMKICAgICMgbW9yZSB0byBjbGVhbiB0aGUgY2F0YWxvZ3VlIHRoYW4gYW55IGJyaWdodG5lc3MgY3V0LgogICAgaWYgbWlub3JfYXhpcyBpcyBub3QgTm9uZToKICAgICAgICB3aWRlX2Vub3VnaCA9IG5wLmlzZmluaXRlKG1pbm9yX2F4aXMpICYgKG5wLmFzYXJyYXkobWlub3JfYXhpcykgPj0gbWluX3dpZHRoX3B4KQogICAgICAgIHJlcG9ydFsidG9vIHRoaW4gdG8gYmUgYSByZWdpb24iXSA9IGludCgofndpZGVfZW5vdWdoICYga2VlcCkuc3VtKCkpCiAgICAgICAga2VlcCAmPSB3aWRlX2Vub3VnaAoKICAgIGlmIGVsb25nYXRpb24gaXMgbm90IE5vbmU6CiAgICAgICAgcm91bmRfZW5vdWdoID0gbnAuaXNmaW5pdGUoZWxvbmdhdGlvbikgJiAobnAuYXNhcnJheShlbG9uZ2F0aW9uKSA8PSBtYXhfZWxvbmdhdGlvbikKICAgICAgICByZXBvcnRbImEgc3RyZWFrLCBub3QgYSByZWdpb24iXSA9IGludCgofnJvdW5kX2Vub3VnaCAmIGtlZXApLnN1bSgpKQogICAgICAgIGtlZXAgJj0gcm91bmRfZW5vdWdoCgogICAgaWYgcl9tYXhfcHggaXMgbm90IE5vbmU6CiAgICAgICAgaW5zaWRlID0gbnAuaHlwb3QoeCAtIGNlbnRyZVswXSwgeSAtIGNlbnRyZVsxXSkgPD0gcl9tYXhfcHgKICAgICAgICByZXBvcnRbIm91dHNpZGUgdGhlIGdhbGF4eSJdID0gaW50KCh+aW5zaWRlICYga2VlcCkuc3VtKCkpCiAgICAgICAga2VlcCAmPSBpbnNpZGUKCiAgICBpZiBzdGFyX3h5IGlzIG5vdCBOb25lIGFuZCBsZW4oc3Rhcl94eSk6CiAgICAgICAgZCA9IG5wLmh5cG90KHhbOiwgTm9uZV0gLSBzdGFyX3h5W05vbmUsIDosIDBdLCB5WzosIE5vbmVdIC0gc3Rhcl94eVtOb25lLCA6LCAxXSkKICAgICAgICBvbl9zdGFyID0gZC5taW4oYXhpcz0xKSA8IHN0YXJfcmFkaXVzX3B4CiAgICAgICAgaWYgZXhjZXNzIGlzIG5vdCBOb25lOgogICAgICAgICAgICBlbWl0dGluZyA9IG5wLmlzZmluaXRlKGV4Y2VzcykgJiAoZXhjZXNzID4gbWluX2V4Y2VzcykKICAgICAgICAgICAgcmVzY3VlZCA9IGludCgob25fc3RhciAmIGVtaXR0aW5nICYga2VlcCkuc3VtKCkpCiAgICAgICAgICAgIGlmIHJlc2N1ZWQ6CiAgICAgICAgICAgICAgICByZXBvcnRbIm9uIGEgc3RhciBidXQgZW1pdHRpbmcsIGtlcHQiXSA9IHJlc2N1ZWQKICAgICAgICAgICAgb25fc3RhciA9IG9uX3N0YXIgJiB+ZW1pdHRpbmcKICAgICAgICByZXBvcnRbIm9uIGEgc3RhciJdID0gaW50KChvbl9zdGFyICYga2VlcCkuc3VtKCkpCiAgICAgICAga2VlcCAmPSB+b25fc3RhcgoKICAgIHJldHVybiBrZWVwLCByZXBvcnQKCmRlZiBjb21wbGV0ZW5lc3NfdnNfcmFkaXVzKHJlc2lkLCBjZW50cmUsIGRldGVjdF9rd2FyZ3MsIHJfZWRnZXMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgIG5fcGVyX3Jpbmc9MzAsIHBlYWtfY291bnRzPTEuMCwgZndobV9weD02LjAsIHJuZ19zZWVkPTExKToKICAgICIiIldoYXQgZnJhY3Rpb24gb2YgaW5qZWN0ZWQgcmVnaW9ucyBjb21lcyBiYWNrLCByaW5nIGJ5IHJpbmcuCgogICAgV0hZIEJPVEhFUgogICAgQSByYWRpYWwgcHJvZmlsZSBpcyB0aGUgbWVhc3VyZW1lbnQgbW9zdCBlYXNpbHkgZmFrZWQgYnkgYSBkZXRlY3Rpb24gbGltaXQgdGhhdAogICAgY2hhbmdlcyB3aXRoIHJhZGl1cy4gVGhlIGNlbnRyZSBvZiBhIGdhbGF4eSBpcyBicmlnaHQgYW5kIGNyb3dkZWQsIHRoZSBvdXRza2lydHMKICAgIGRhcmsgYW5kIGVtcHR5LCBzbyB0aGUgZmFpbnRlc3QgdGhpbmcgc3RpbGwgdmlzaWJsZSBpcyBub3QgdGhlIHNhbWUgaW4gYm90aC4gSWYKICAgIGRldGVjdGlvbiBnZXRzIGhhcmRlciBpbndhcmRzLCB0aGUgcHJvZmlsZSBncm93cyBhIGNlbnRyYWwgaG9sZSB0aGF0IGJlbG9uZ3MgdG8KICAgIHRoZSBzZWFyY2gsIG5vdCB0byB0aGUgZ2FsYXh5LgoKICAgIEhPVwogICAgRHJvcCBhcnRpZmljaWFsIHJlZ2lvbnMgb2Yga25vd24gcGVhayBicmlnaHRuZXNzIGludG8gdGhlIHJlYWwgaW1hZ2UsIHJ1bgogICAgZXhhY3RseSB0aGUgc2FtZSBkZXRlY3Rpb24gb24gaXQsIGFuZCBjb3VudCBob3cgbWFueSBjb21lIGJhY2ssIHJpbmcgYnkgcmluZy4KCiAgICBwZWFrX2NvdW50cyBpcyB0aGUgcGVhayBoZWlnaHQgb2YgdGhlIGluamVjdGVkIHJlZ2lvbiBhbmQgc2hvdWxkIGJlIHNldCBpbiB1bml0cwogICAgb2YgdGhlIGJhY2tncm91bmQgbm9pc2UuIFNjYWxpbmcgaW5qZWN0ZWQgcmVnaW9ucyB0byB0aGUgYnJpZ2h0bmVzcyBvZiBhbgogICAgYXZlcmFnZSByZWFsIG9uZSBwdXRzIHRoZW0gaHVuZHJlZHMgb2Ygc2lnbWEgYWJvdmUgdGhlIHRocmVzaG9sZCwgd2hlcmUKICAgIGV2ZXJ5dGhpbmcgaXMgZm91bmQgZXZlcnl3aGVyZSBhbmQgdGhlIHRlc3QgbWVhc3VyZXMgbm90aGluZy4KICAgICIiIgogICAgZnJvbSBzY2lweS5uZGltYWdlIGltcG9ydCBnYXVzc2lhbl9maWx0ZXIKCiAgICBybmcgPSBucC5yYW5kb20uZGVmYXVsdF9ybmcocm5nX3NlZWQpCiAgICBzaWdtYSA9IGZ3aG1fcHggLyAyLjM1NDgKICAgIG55LCBueCA9IHJlc2lkLnNoYXBlCiAgICByZWNvdmVyZWQgPSBucC56ZXJvcyhsZW4ocl9lZGdlcykgLSAxKQogICAgaW5qZWN0ZWQgPSBucC56ZXJvcyhsZW4ocl9lZGdlcykgLSAxKQoKICAgIGZvciBrIGluIHJhbmdlKGxlbihyX2VkZ2VzKSAtIDEpOgogICAgICAgIHJfbG8sIHJfaGkgPSByX2VkZ2VzW2tdLCByX2VkZ2VzW2sgKyAxXQogICAgICAgIGZha2UgPSBucC56ZXJvc19saWtlKHJlc2lkKQogICAgICAgIHRydXRoID0gW10KICAgICAgICBmb3IgXyBpbiByYW5nZShuX3Blcl9yaW5nKToKICAgICAgICAgICAgcmFkID0gbWF0aC5zcXJ0KHJuZy51bmlmb3JtKHJfbG8gKiogMiwgcl9oaSAqKiAyKSkKICAgICAgICAgICAgdGggPSBybmcudW5pZm9ybSgwLjAsIDIuMCAqIG1hdGgucGkpCiAgICAgICAgICAgIHB4ID0gY2VudHJlWzBdICsgcmFkICogbWF0aC5jb3ModGgpCiAgICAgICAgICAgIHB5ID0gY2VudHJlWzFdICsgcmFkICogbWF0aC5zaW4odGgpCiAgICAgICAgICAgIGlmIG5vdCAoOCA8IHB4IDwgbnggLSA4IGFuZCA4IDwgcHkgPCBueSAtIDgpOgogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgZmFrZVtpbnQocm91bmQocHkpKSwgaW50KHJvdW5kKHB4KSldICs9IHBlYWtfY291bnRzCiAgICAgICAgICAgIHRydXRoLmFwcGVuZCgocHgsIHB5KSkKICAgICAgICBpZiBub3QgdHJ1dGg6CiAgICAgICAgICAgIGNvbnRpbnVlCgogICAgICAgICMgQSBkZWx0YSBzbW9vdGhlZCBieSBhIEdhdXNzaWFuIGhhcyBwZWFrIGFtcGxpdHVkZS8oMiBwaSBzaWdtYV4yKTsKICAgICAgICAjIG11bHRpcGx5aW5nIGl0IGJhY2sgZ2l2ZXMgYSByZWdpb24gd2hvc2UgUEVBSyBpcyBwZWFrX2NvdW50cy4KICAgICAgICBmYWtlID0gZ2F1c3NpYW5fZmlsdGVyKGZha2UsIHNpZ21hKSAqICgyLjAgKiBtYXRoLnBpICogc2lnbWEgKiogMikKICAgICAgICBjYXQsIF8sIF8gPSBkZXRlY3RfcmVnaW9ucyhyZXNpZCArIGZha2UsICoqZGV0ZWN0X2t3YXJncykKCiAgICAgICAgaWYgY2F0IGlzIE5vbmU6CiAgICAgICAgICAgIGluamVjdGVkW2tdICs9IGxlbih0cnV0aCkKICAgICAgICAgICAgY29udGludWUKCiAgICAgICAgZnggPSBucC5hc2FycmF5KGNhdC54Y2VudHJvaWQsIGR0eXBlPWZsb2F0KQogICAgICAgIGZ5ID0gbnAuYXNhcnJheShjYXQueWNlbnRyb2lkLCBkdHlwZT1mbG9hdCkKICAgICAgICAjIFNlZ21lbnRzIHRvdWNoaW5nIHRoZSBtYXNrIGNvbWUgYmFjayB3aXRoIGEgY2VudHJvaWQgb2YgTmFOLiBMZWZ0IGluLCBhCiAgICAgICAgIyBzaW5nbGUgTmFOIG1ha2VzIGV2ZXJ5IGRpc3RhbmNlIGNvbXBhcmlzb24gYmVsb3cgZmFsc2UgYW5kIHRoZSBtZWFzdXJlZAogICAgICAgICMgY29tcGxldGVuZXNzIGNvbGxhcHNlcyB0byB6ZXJvIGV2ZXJ5d2hlcmUuCiAgICAgICAgZmluaXRlID0gbnAuaXNmaW5pdGUoZngpICYgbnAuaXNmaW5pdGUoZnkpCiAgICAgICAgZngsIGZ5ID0gZnhbZmluaXRlXSwgZnlbZmluaXRlXQoKICAgICAgICBmb3IgKHR4LCB0eSkgaW4gdHJ1dGg6CiAgICAgICAgICAgIGluamVjdGVkW2tdICs9IDEKICAgICAgICAgICAgaWYgZnguc2l6ZSBhbmQgbnAubWluKG5wLmh5cG90KGZ4IC0gdHgsIGZ5IC0gdHkpKSA8IDMuMDoKICAgICAgICAgICAgICAgIHJlY292ZXJlZFtrXSArPSAxCgogICAgd2l0aCBucC5lcnJzdGF0ZShpbnZhbGlkPSJpZ25vcmUiLCBkaXZpZGU9Imlnbm9yZSIpOgogICAgICAgIGZyYWMgPSBucC53aGVyZShpbmplY3RlZCA+IDAsIHJlY292ZXJlZCAvIGluamVjdGVkLCBucC5uYW4pCiAgICByZXR1cm4gZnJhYywgaW5qZWN0ZWQsIHJlY292ZXJlZAoKZGVmIGNob29zZV9zdWJ0cmFjdGlvbihoYSwgcmVkLCBwb3NpdGlvbnMsIHByZWZpeCwgY2VudHJlPU5vbmUsIHIyNV9weD1Ob25lLAogICAgICAgICAgICAgICAgICAgICAgIGZhY3RvcnM9Tm9uZSwgYmx1cnM9Tm9uZSwgYXNrPU5vbmUpOgogICAgIiIiT2ZmZXIgdGhyZWUgc3VidHJhY3Rpb25zIGFzIHBpY3R1cmVzIGFuZCBsZXQgdGhlIHBlcnNvbiBwaWNrIG9uZS4KCiAgICBXSFkgQSBQRVJTT04gREVDSURFUyBUSElTCiAgICBCZWNhdXNlIGV2ZXJ5IGF1dG9tYXRpYyBjcml0ZXJpb24gdHJpZWQgaGVyZSBkaXNhZ3JlZWQgd2l0aCB0aGUgZXllLCBhbmQgdGhlCiAgICBleWUgd2FzIHJpZ2h0LiBNYXRjaGluZyBzdGFyIHdpZHRocyBieSB0aGVpciBtZWFzdXJlZCBGV0hNIHByZXNjcmliZWQgYSBibHVyCiAgICB3aXRoIHRoZSBzaWduIHdyb25nLiBDaG9vc2luZyB0aGUgZmFjdG9yIGFzIHRoZSBtZWFuIG92ZXIgc3RhcnMgbGVmdCB0d2ljZSBhcwogICAgbWFueSB2aXNpYmxlIHN0YXIgcmVzaWR1YWxzIGFzIGEgaGFuZC10dW5lZCB2YWx1ZSBhIGZldyBwZXJjZW50IGxvd2VyLiBUaGUKICAgIHF1YW50aXR5IHRoYXQgYWN0dWFsbHkgbWF0dGVycyAtIHdoZXRoZXIgYSBzdGFyIGhhcyB2YW5pc2hlZCBpbnRvIHRoZSBza3kgb3IKICAgIHN0aWxsIHNob3dzIGFzIGEgc3BvdCB0aGF0IGEgZGV0ZWN0b3Igd2lsbCBjYWxsIGFuIEhJSSByZWdpb24gLSBpcyBzb21ldGhpbmcgYQogICAgcGVyc29uIHJlYWRzIG9mZiBhIGZyYW1lIGluIGEgc2Vjb25kIGFuZCBubyBzaW5nbGUgbnVtYmVyIGhlcmUgaGFzIGNhcHR1cmVkLgoKICAgIFdIWSBUSFJFRSBBTkQgTk9UIEEgTlVNQkVSIE9OIFRIRSBTQ1JFRU4KICAgIEEgbnVtYmVyIGNhbm5vdCBiZSBqdWRnZWQgd2l0aG91dCBrbm93aW5nIHdoYXQgaXQgbG9va3MgbGlrZS4gVGhyZWUgZnJhbWVzCiAgICB3cml0dGVuIHRvIGRpc2ssIGRpZmZlcmluZyBlbm91Z2ggdG8gdGVsbCBhcGFydCwgYW5kIG9uZSBxdWVzdGlvbi4KCiAgICBIT1cgVEhFIFRIUkVFIEFSRSBQSUNLRUQKICAgIFRoZSB3aG9sZSBncmlkIG9mIGZhY3RvciBhbmQgYmx1ciBpcyBzY29yZWQgb24gc3RhcnMgdGhhdCBsaWUgb3V0c2lkZSB0aGUKICAgIGdhbGF4eSwgd2hlcmUgbm90aGluZyBidXQgc2t5IGFuZCB0aGUgc3RhciBpcyBpbiB0aGUgYXBlcnR1cmUuIFRoZSBzY29yZSBpcyBob3cKICAgIGZsYXQgd2hhdCBpcyBsZWZ0IG9mIHRoZSBtZWRpYW4gc3RhciBpcyAtIGEgZ29vZCBzdWJ0cmFjdGlvbiBsZWF2ZXMgbm8gY29yZSBhbmQKICAgIG5vIHJpbmcuIFRoZSBiZXN0IGlzIG9mZmVyZWQsIHdpdGggb25lIGNhbmRpZGF0ZSBlaXRoZXIgc2lkZSBvZiBpdCBpbiBmYWN0b3IsCiAgICBzbyB0aGF0IHRoZSB0aHJlZSBzcGFuIHZpc2libHkgbGVzcyBhbmQgdmlzaWJseSBtb3JlIHN1YnRyYWN0aW9uIHJhdGhlciB0aGFuCiAgICB0aHJlZSB2ZXJzaW9ucyBvZiB0aGUgc2FtZSBhbnN3ZXIuCgogICAgUmV0dXJucyAoaGEsIHJlZCwgZmFjdG9yLCBkaWFnbm9zdGljcykuIFdpdGhvdXQgYSBxdWVzdGlvbiB0byBhc2sgLSBhCiAgICBub24taW50ZXJhY3RpdmUgcnVuIC0gdGhlIG1pZGRsZSBvbmUgaXMgdGFrZW4gYW5kIHNhaWQgc28uCiAgICAiIiIKICAgIGZyb20gc2NpcHkubmRpbWFnZSBpbXBvcnQgZ2F1c3NpYW5fZmlsdGVyCiAgICBmcm9tIGFzdHJvcHkuc3RhdHMgaW1wb3J0IHNpZ21hX2NsaXBwZWRfc3RhdHMKCiAgICBkaWFnID0ge30KICAgIGlmIHBvc2l0aW9ucyBpcyBOb25lIG9yIGxlbihwb3NpdGlvbnMpIDwgMzA6CiAgICAgICAgcmV0dXJuIGhhLCByZWQsIE5vbmUsIHsid2h5IjogInRvbyBmZXcgc3RhcnMgdG8ganVkZ2UgYSBzdWJ0cmFjdGlvbiBvbiJ9CgogICAgcG9zID0gbnAuYXNhcnJheShwb3NpdGlvbnMsIGR0eXBlPWZsb2F0KQogICAgaWYgY2VudHJlIGlzIG5vdCBOb25lIGFuZCByMjVfcHg6CiAgICAgICAgb3V0c2lkZSA9IG5wLmh5cG90KHBvc1s6LCAwXSAtIGNlbnRyZVswXSwgcG9zWzosIDFdIC0gY2VudHJlWzFdKSA+IDEuMyAqIHIyNV9weAogICAgICAgIGlmIG91dHNpZGUuc3VtKCkgPj0gMzA6CiAgICAgICAgICAgIHBvcyA9IHBvc1tvdXRzaWRlXQogICAgZGlhZ1sibl9zdGFycyJdID0gaW50KGxlbihwb3MpKQoKICAgICMgV0hFUkUgVEhFIFJBTkdFIENPTUVTIEZST00KICAgICMgRnJvbSB0aGlzIHBhaXIgb2YgZnJhbWVzLCBuZXZlciBmcm9tIGEgbnVtYmVyIHR5cGVkIGluLiBUaGUgZmFjdG9yIHRoYXQKICAgICMgY2FuY2VscyBhIHN0YXIgaXMgbWVhc3VyZWQgb24gdGhlIHN0YXJzIHRoZW1zZWx2ZXMgYW5kIHRoZSBncmlkIGlzIGxhaWQgb3V0CiAgICAjIGFyb3VuZCBpdCwgc28gYSBnYWxheHkgd2hvc2UgZmFjdG9yIGlzIHR3ZW50eSBpcyBzZWFyY2hlZCBuZWFyIHR3ZW50eSBhbmQgb25lCiAgICAjIHdob3NlIGZhY3RvciBpcyB0d2VsdmUgaXMgc2VhcmNoZWQgbmVhciB0d2VsdmUuIEEgZ3JpZCB3cml0dGVuIGJ5IGhhbmQgd291bGQKICAgICMgaGF2ZSBiZWVuIHRoZSByYW5nZSBvZiB3aGljaGV2ZXIgZ2FsYXh5IGl0IHdhcyB3cml0dGVuIGZvci4KICAgIGlmIGZhY3RvcnMgaXMgTm9uZToKICAgICAgICBwYXNzICAjIGZyb20gY29udGludXVtX3N1YnRyYWN0aW9uLCBpbmxpbmVkIGFib3ZlCiAgICAgICAgdHJ5OgogICAgICAgICAgICBjZW50cmVfZmFjdG9yID0gZmxvYXQoY29uc3RhbnRfZmFjdG9yKGhhLCByZWQsIHBvc2l0aW9ucz1wb3MpWzBdKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIGNlbnRyZV9mYWN0b3IgPSBOb25lCiAgICAgICAgaWYgbm90IGNlbnRyZV9mYWN0b3Igb3Igbm90IG5wLmlzZmluaXRlKGNlbnRyZV9mYWN0b3IpIG9yIGNlbnRyZV9mYWN0b3IgPD0gMDoKICAgICAgICAgICAgcmV0dXJuIGhhLCByZWQsIE5vbmUsIHsid2h5IjogImNvdWxkIG5vdCBtZWFzdXJlIGEgZmFjdG9yIHRvIHNlYXJjaCBhcm91bmQifQogICAgICAgIGRpYWdbIm1lYXN1cmVkX2ZhY3RvciJdID0gY2VudHJlX2ZhY3RvcgoKICAgIGggPSA4CiAgICB5eSwgeHggPSBucC5tZ3JpZFstaDpoICsgMSwgLWg6aCArIDFdCiAgICByciA9IG5wLmh5cG90KHh4LCB5eSkKICAgIGVkZ2UgPSByciA+IGggLSAxLjUKICAgIG55LCBueCA9IGhhLnNoYXBlCiAgICBzZWwgPSBbKGludChyb3VuZCh4KSksIGludChyb3VuZCh5KSkpIGZvciB4LCB5IGluIHBvcwogICAgICAgICAgIGlmIGggPCBpbnQocm91bmQoeCkpIDwgbnggLSBoIGFuZCBoIDwgaW50KHJvdW5kKHkpKSA8IG55IC0gaF0KICAgIGlmIGxlbihzZWwpIDwgMzA6CiAgICAgICAgcmV0dXJuIGhhLCByZWQsIE5vbmUsIHsid2h5IjogInRoZSBzdGFycyBhbGwgZmFsbCB0b28gbmVhciB0aGUgZnJhbWUgZWRnZSJ9CgogICAgZGVmIHN0YW1wcyhpbWcpOgogICAgICAgIG91dCA9IG5wLmVtcHR5KChsZW4oc2VsKSwgMiAqIGggKyAxLCAyICogaCArIDEpLCBkdHlwZT1ucC5mbG9hdDMyKQogICAgICAgIGZvciBpLCAoeCwgeSkgaW4gZW51bWVyYXRlKHNlbCk6CiAgICAgICAgICAgIHMgPSBucC5uYW5fdG9fbnVtKGltZ1t5IC0gaDp5ICsgaCArIDEsIHggLSBoOnggKyBoICsgMV0pCiAgICAgICAgICAgIG91dFtpXSA9IHMgLSBucC5tZWRpYW4oc1tlZGdlXSkKICAgICAgICByZXR1cm4gb3V0CgogICAgIyBUSEVSRSBJUyBOTyBSQU5HRQogICAgIyBBbiB1cHBlciBhbmQgbG93ZXIgYm91bmQgb24gdGhlIHNlYXJjaCBpcyBhIGd1ZXNzIGFib3V0IHRoZSBhbnN3ZXIsIGFuZCB0aGUKICAgICMgYW5zd2VyIGlzIHdoYXQgaXMgYmVpbmcgbG9va2VkIGZvci4gU28gdGhlIHNlYXJjaCBzdGFydHMgc21hbGwgYW5kIGdyb3dzOiBhcwogICAgIyBsb25nIGFzIHRoZSBiZXN0IHBvaW50IGl0IGhhcyBmb3VuZCBzaXRzIG9uIGFuIGVkZ2Ugb2Ygd2hhdCBoYXMgYmVlbiB0cmllZCwKICAgICMgdGhhdCBlZGdlIG1vdmVzIG91dCBhbmQgaXQgbG9va3MgYWdhaW4uIEl0IHN0b3BzIHdoZW4gdGhlIGJlc3QgcG9pbnQgaGFzCiAgICAjIG5laWdoYm91cnMgb24gYWxsIHNpZGVzLCB3aGljaCBpcyB0aGUgb25seSBldmlkZW5jZSB0aGF0IGl0IGlzIGEgbWluaW11bQogICAgIyByYXRoZXIgdGhhbiB0aGUgZW5kIG9mIHRoZSBwYXBlci4KICAgIGZfc3RlcCA9IDAuMDI1ICogY2VudHJlX2ZhY3RvciBpZiBmYWN0b3JzIGlzIE5vbmUgZWxzZSBOb25lCiAgICBiX3N0ZXAgPSAwLjI1CiAgICBpZiBmYWN0b3JzIGlzIE5vbmU6CiAgICAgICAgZl9sbywgZl9oaSA9IC0yLCAyICAgICAgICAgICAgICAgICAgIyBpbiBzdGVwcywgcmVsYXRpdmUgdG8gdGhlIG1lYXN1cmVkIGZhY3RvcgogICAgZWxzZToKICAgICAgICBmX2xvID0gZl9oaSA9IE5vbmUKICAgIGJfbG8sIGJfaGkgPSAtMiwgMgoKICAgIGNhY2hlID0ge30KICAgIHN0YW1wX2NhY2hlID0ge30KCiAgICBkZWYgc2NvcmUoZmksIGJpKToKICAgICAgICBrZXkgPSAoZmksIGJpKQogICAgICAgIGlmIGtleSBpbiBjYWNoZToKICAgICAgICAgICAgcmV0dXJuIGNhY2hlW2tleV0KICAgICAgICBibHVyID0gYmkgKiBiX3N0ZXAKICAgICAgICBpZiBiaSBub3QgaW4gc3RhbXBfY2FjaGU6CiAgICAgICAgICAgIEEgPSBnYXVzc2lhbl9maWx0ZXIoaGEsIGJsdXIpIGlmIGJsdXIgPiAwIGVsc2UgaGEKICAgICAgICAgICAgUiA9IGdhdXNzaWFuX2ZpbHRlcihyZWQsIC1ibHVyKSBpZiBibHVyIDwgMCBlbHNlIHJlZAogICAgICAgICAgICBzdGFtcF9jYWNoZVtiaV0gPSAoc3RhbXBzKEEpLCBzdGFtcHMoUikpCiAgICAgICAgU0EsIFNSID0gc3RhbXBfY2FjaGVbYmldCiAgICAgICAgZiA9IChjZW50cmVfZmFjdG9yICsgZmkgKiBmX3N0ZXApIGlmIGZhY3RvcnMgaXMgTm9uZSBlbHNlIGZsb2F0KGZhY3RvcnNbZmldKQogICAgICAgIE0gPSBucC5tZWRpYW4oZiAqIFNBIC0gU1IsIGF4aXM9MCkKICAgICAgICBwcm9mID0gW01bKHJyID49IHEpICYgKHJyIDwgcSArIDEpXS5tZWFuKCkgZm9yIHEgaW4gcmFuZ2UoNildCiAgICAgICAgY2FjaGVba2V5XSA9IChmbG9hdChucC5hYnMocHJvZikubWF4KCkpLCBmbG9hdChmKSwgZmxvYXQoYmx1cikpCiAgICAgICAgcmV0dXJuIGNhY2hlW2tleV0KCiAgICBpZiBmYWN0b3JzIGlzIG5vdCBOb25lOgogICAgICAgIGZfbG8sIGZfaGkgPSAwLCBsZW4oZmFjdG9ycykgLSAxCiAgICBpZiBibHVycyBpcyBub3QgTm9uZToKICAgICAgICBiX3N0ZXAgPSAxLjAKICAgICAgICBiX2xvLCBiX2hpID0gMCwgbGVuKGJsdXJzKSAtIDEKCiAgICBmb3IgX3JvdW5kIGluIHJhbmdlKDQwKToKICAgICAgICBncmlkID0gW3Njb3JlKGZpLCBiaSkKICAgICAgICAgICAgICAgIGZvciBmaSBpbiByYW5nZShmX2xvLCBmX2hpICsgMSkKICAgICAgICAgICAgICAgIGZvciBiaSBpbiByYW5nZShiX2xvLCBiX2hpICsgMSldCiAgICAgICAgYmVzdF9sZWZ0LCBiZXN0X2YsIGJlc3RfYmx1ciA9IG1pbihncmlkKQogICAgICAgIGJpX2Jlc3QgPSBpbnQocm91bmQoYmVzdF9ibHVyIC8gYl9zdGVwKSkgaWYgYl9zdGVwIGVsc2UgMAogICAgICAgIGZpX2Jlc3QgPSAoaW50KHJvdW5kKChiZXN0X2YgLSBjZW50cmVfZmFjdG9yKSAvIGZfc3RlcCkpCiAgICAgICAgICAgICAgICAgICBpZiBmYWN0b3JzIGlzIE5vbmUgZWxzZSBOb25lKQogICAgICAgIGdyZXcgPSBGYWxzZQogICAgICAgIGlmIGZhY3RvcnMgaXMgTm9uZToKICAgICAgICAgICAgaWYgZmlfYmVzdCA8PSBmX2xvIGFuZCBjZW50cmVfZmFjdG9yICsgKGZfbG8gLSAxKSAqIGZfc3RlcCA+IDA6CiAgICAgICAgICAgICAgICBmX2xvIC09IDE7IGdyZXcgPSBUcnVlCiAgICAgICAgICAgIGlmIGZpX2Jlc3QgPj0gZl9oaToKICAgICAgICAgICAgICAgIGZfaGkgKz0gMTsgZ3JldyA9IFRydWUKICAgICAgICBpZiBiaV9iZXN0IDw9IGJfbG86CiAgICAgICAgICAgIGJfbG8gLT0gMTsgZ3JldyA9IFRydWUKICAgICAgICBpZiBiaV9iZXN0ID49IGJfaGk6CiAgICAgICAgICAgIGJfaGkgKz0gMTsgZ3JldyA9IFRydWUKICAgICAgICBpZiBub3QgZ3JldzoKICAgICAgICAgICAgYnJlYWsKCiAgICBzY29yZWQgPSBzb3J0ZWQoY2FjaGUudmFsdWVzKCkpCiAgICBiZXN0X2xlZnQsIGJlc3RfZiwgYmVzdF9ibHVyID0gc2NvcmVkWzBdCiAgICBkaWFnWyJiZXN0Il0gPSAoYmVzdF9mLCBiZXN0X2JsdXIsIGJlc3RfbGVmdCkKICAgIGRpYWdbInNlYXJjaGVkIl0gPSB7CiAgICAgICAgImZhY3RvciI6IChtaW4oc1sxXSBmb3IgcyBpbiBzY29yZWQpLCBtYXgoc1sxXSBmb3IgcyBpbiBzY29yZWQpKSwKICAgICAgICAiYmx1ciI6IChtaW4oc1syXSBmb3IgcyBpbiBzY29yZWQpLCBtYXgoc1syXSBmb3IgcyBpbiBzY29yZWQpKSwKICAgICAgICAicG9pbnRzIjogbGVuKHNjb3JlZCl9CiAgICBwcmludCgiICAgIHNlYXJjaGVkIGZhY3RvciB7Oi4yZn0gdG8gezouMmZ9IGFuZCBibHVyIHs6Ky4yZn0gdG8gezorLjJmfSBweCAiCiAgICAgICAgICAib3ZlciB7fSBjb21iaW5hdGlvbnMsIHVudGlsIHRoZSBiZXN0IGhhZCBuZWlnaGJvdXJzIG9uIGV2ZXJ5IHNpZGUiCiAgICAgICAgICAuZm9ybWF0KGRpYWdbInNlYXJjaGVkIl1bImZhY3RvciJdWzBdLCBkaWFnWyJzZWFyY2hlZCJdWyJmYWN0b3IiXVsxXSwKICAgICAgICAgICAgICAgICAgZGlhZ1sic2VhcmNoZWQiXVsiYmx1ciJdWzBdLCBkaWFnWyJzZWFyY2hlZCJdWyJibHVyIl1bMV0sCiAgICAgICAgICAgICAgICAgIGxlbihzY29yZWQpKSkKCiAgICAjIE9uZSBjYW5kaWRhdGUgZWl0aGVyIHNpZGUgaW4gZmFjdG9yLCBlYWNoIHdpdGggaXRzIG93biBiZXN0IGJsdXIsIHNvIHRoZQogICAgIyB0aHJlZSBkaWZmZXIgYnkgc29tZXRoaW5nIHRoZSBleWUgY2FuIHNlZS4KICAgIGRlZiBiZXN0X2F0KGYpOgogICAgICAgIGhlcmUgPSBbcyBmb3IgcyBpbiBzY29yZWQgaWYgYWJzKHNbMV0gLSBmKSA8IDFlLTZdCiAgICAgICAgcmV0dXJuIG1pbihoZXJlKSBpZiBoZXJlIGVsc2UgTm9uZQoKICAgIG90aGVycyA9IHNvcnRlZCh7ZiBmb3IgXywgZiwgXyBpbiBzY29yZWR9KQogICAgYmVsb3cgPSBbZiBmb3IgZiBpbiBvdGhlcnMgaWYgZiA8IGJlc3RfZl0KICAgIGFib3ZlID0gW2YgZm9yIGYgaW4gb3RoZXJzIGlmIGYgPiBiZXN0X2ZdCiAgICBwaWNrcyA9IFtiZXN0X2F0KGJlbG93Wy0xXSkgaWYgYmVsb3cgZWxzZSBOb25lLAogICAgICAgICAgICAgKGJlc3RfbGVmdCwgYmVzdF9mLCBiZXN0X2JsdXIpLAogICAgICAgICAgICAgYmVzdF9hdChhYm92ZVswXSkgaWYgYWJvdmUgZWxzZSBOb25lXQogICAgcGlja3MgPSBbcCBmb3IgcCBpbiBwaWNrcyBpZiBwIGlzIG5vdCBOb25lXQogICAgaWYgbGVuKHBpY2tzKSA8IDM6ICAgICAgICAgICAgICAgICAgICAgICAjIHRoZSBiZXN0IHNhdCBhdCBhbiBlbmQgb2YgdGhlIHJhbmdlCiAgICAgICAgZXh0cmFfcG9vbCA9IFtzIGZvciBzIGluIHNjb3JlZCBpZiBzWzFdIG5vdCBpbiB7cFsxXSBmb3IgcCBpbiBwaWNrc31dCiAgICAgICAgcGlja3MgKz0gc29ydGVkKGV4dHJhX3Bvb2wpWzozIC0gbGVuKHBpY2tzKV0KICAgIHBpY2tzLnNvcnQoa2V5PWxhbWJkYSBwOiBwWzFdKQoKICAgIHJldHVybiBfd3JpdGVfY2hvaWNlcyhoYSwgcmVkLCBwaWNrcywgcHJlZml4LCBkaWFnLCBhc2spCgpkZWYgc2hvd19waWN0dXJlcyhwYXRocywgdGl0bGU9Tm9uZSk6CiAgICAiIiJQdXQgcGljdHVyZXMgYWxyZWFkeSB3cml0dGVuIHRvIGRpc2sgb24gdGhlIHNjcmVlbiwgd2l0aG91dCBibG9ja2luZy4KCiAgICBXSFkgSVQgU0hPV1MgVEhFIFNBVkVEIEZJTEUgQU5EIE5PVCBUSEUgRklHVVJFCiAgICBCZWNhdXNlIHRoZSBmaWd1cmUgd2FzIGRyYXduIHVuZGVyIEFnZywgdGhlIGJhY2tlbmQgdGhhdCB3cml0ZXMgdG8gYSBmaWxlIGFuZAogICAgaGFzIG5vIHdpbmRvdyB0byBhdHRhY2ggdG8sIGFuZCBtb3ZpbmcgYSBsaXZlIGZpZ3VyZSBiZXR3ZWVuIGJhY2tlbmRzIGlzIG5vdAogICAgc29tZXRoaW5nIG1hdHBsb3RsaWIgc3VwcG9ydHMuIFJlYWRpbmcgYmFjayB0aGUgUE5HIHRoYXQgd2FzIGp1c3Qgd3JpdHRlbgogICAgc2lkZXN0ZXBzIHRoYXQgZW50aXJlbHksIGFuZCBoYXMgdGhlIGFkdmFudGFnZSB0aGF0IHdoYXQgYXBwZWFycyBvbiB0aGUgc2NyZWVuCiAgICBpcyBleGFjdGx5IHRoZSBmaWxlIC0gaWYgdGhleSBldmVyIGRpc2FncmVlZCBpdCB3b3VsZCBiZSB0aGUgZmlsZSB0aGF0IGlzIHdyb25nLAogICAgYW5kIHRoZSBmaWxlIGlzIHdoYXQgZ2V0cyBrZXB0LgoKICAgIEFuIGludGVyYWN0aXZlIGJhY2tlbmQgaXMgYm9ycm93ZWQgYW5kIGdpdmVuIGJhY2suIE9uIGEgbWFjaGluZSB3aXRoIG5vIGRpc3BsYXkKICAgIG5vdGhpbmcgaXMgc2hvd24gYW5kIG5vdGhpbmcgZmFpbHM7IHRoZSBmaWxlcyBhcmUgc3RpbGwgdGhlcmUuCiAgICAiIiIKICAgIGltcG9ydCBtYXRwbG90bGliCiAgICBpbXBvcnQgbWF0cGxvdGxpYi5weXBsb3QgYXMgcGx0CgogICAgcGF0aHMgPSBbcCBmb3IgcCBpbiBwYXRocyBpZiBwIGFuZCBvcy5wYXRoLmV4aXN0cyhwKV0KICAgIGlmIG5vdCBwYXRoczoKICAgICAgICByZXR1cm4gRmFsc2UKCiAgICBzdGFydGVkID0gbWF0cGxvdGxpYi5nZXRfYmFja2VuZCgpCiAgICBpZiBzdGFydGVkLmxvd2VyKCkgPT0gImFnZyI6CiAgICAgICAgZm9yIGNhbmRpZGF0ZSBpbiAoIlRrQWdnIiwgIlF0QWdnIiwgIlF0NUFnZyIsICJNYWNPU1giKToKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgcGx0LnN3aXRjaF9iYWNrZW5kKGNhbmRpZGF0ZSkKICAgICAgICAgICAgICAgIGJyZWFrCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgaWYgbWF0cGxvdGxpYi5nZXRfYmFja2VuZCgpLmxvd2VyKCkgPT0gImFnZyI6CiAgICAgICAgcmV0dXJuIEZhbHNlCgogICAgdHJ5OgogICAgICAgIGltcG9ydCBtYXRwbG90bGliLmltYWdlIGFzIG1waW1nCiAgICAgICAgZm9yIHBhdGggaW4gcGF0aHM6CiAgICAgICAgICAgIGltYWdlID0gbXBpbWcuaW1yZWFkKHBhdGgpCiAgICAgICAgICAgIGhlaWdodCwgd2lkdGggPSBpbWFnZS5zaGFwZVswXSwgaW1hZ2Uuc2hhcGVbMV0KICAgICAgICAgICAgc2lkZSA9IDkuNQogICAgICAgICAgICBmaWcgPSBwbHQuZmlndXJlKGZpZ3NpemU9KHNpZGUsIHNpZGUgKiBoZWlnaHQgLyBmbG9hdCh3aWR0aCkpKQogICAgICAgICAgICBheCA9IGZpZy5hZGRfYXhlcyhbMCwgMCwgMSwgMV0pCiAgICAgICAgICAgIGF4Lmltc2hvdyhpbWFnZSkKICAgICAgICAgICAgYXguc2V0X2F4aXNfb2ZmKCkKICAgICAgICAgICAgZmlnLmNhbnZhcy5tYW5hZ2VyLnNldF93aW5kb3dfdGl0bGUoCiAgICAgICAgICAgICAgICB0aXRsZSBvciBvcy5wYXRoLmJhc2VuYW1lKHBhdGgpKQogICAgICAgIHBsdC5zaG93KGJsb2NrPUZhbHNlKQogICAgICAgIHBsdC5wYXVzZSgwLjUpCiAgICAgICAgcmV0dXJuIFRydWUKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcmV0dXJuIEZhbHNlCiAgICBmaW5hbGx5OgogICAgICAgIHRyeToKICAgICAgICAgICAgcGx0LnN3aXRjaF9iYWNrZW5kKHN0YXJ0ZWQpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgcGFzcwoKZGVmIF9zaG93X2Nob2ljZXNfd2luZG93KGZyYW1lcywgbnosIHN0ZXApOgogICAgIiIiUHV0IHRoZSB0aHJlZSBjYW5kaWRhdGVzIHNpZGUgYnkgc2lkZSBpbiBvbmUgd2luZG93IGFuZCBsZWF2ZSBpdCBvcGVuLgoKICAgIFdIWSBBIFdJTkRPVyBBTkQgTk9UIFRIUkVFIEZJTEVTCiAgICBCZWNhdXNlIHRoZSBqdWRnZW1lbnQgaXMgYSBjb21wYXJpc29uLiBUaHJlZSBmaWxlcyBvcGVuZWQgb25lIGFmdGVyIGFub3RoZXIgaW4KICAgIGEgcGljdHVyZSB2aWV3ZXIgYXJlIHRocmVlIHNlcGFyYXRlIGltcHJlc3Npb25zIGEgZmV3IHNlY29uZHMgYXBhcnQ7IHNpZGUgYnkKICAgIHNpZGUgdGhleSBhcmUgb25lIGxvb2ssIGFuZCB0aGUgZGlmZmVyZW5jZSBiZXR3ZWVuIHRoZW0gLSB3aGljaCBpcyBzbWFsbCwgYSBmZXcKICAgIHBlcmNlbnQgaW4gdGhlIGZhY3RvciAtIGlzIG9ubHkgdmlzaWJsZSB0aGF0IHdheS4KCiAgICBUaGUgYmFja2VuZCBpcyB0aGUgZGlmZmljdWx0eS4gQm90aCB0b29scyBzZXQgQWdnIGF0IGltcG9ydCwgd2hpY2ggZHJhd3MgdG8gYQogICAgZmlsZSBhbmQgbmV2ZXIgdG8gYSBzY3JlZW4sIGFuZCB0aGF0IGlzIHRoZSByaWdodCBkZWZhdWx0IGZvciBhIHJ1biB0aGF0CiAgICBwcm9kdWNlcyBmaWd1cmVzIHVuYXR0ZW5kZWQuIFNvIGFuIGludGVyYWN0aXZlIGJhY2tlbmQgaXMgYm9ycm93ZWQgZm9yIHRoaXMgb25lCiAgICB3aW5kb3cgYW5kIGdpdmVuIGJhY2sgYWZ0ZXJ3YXJkcywgYW5kIGlmIG5vbmUgb2YgdGhlbSBjYW4gYmUgaGFkIC0gYSBtYWNoaW5lCiAgICB3aXRoIG5vIGRpc3BsYXksIGEgc2Vzc2lvbiBvdmVyIHNzaCAtIG5vdGhpbmcgaXMgc2hvd24gYW5kIHRoZSBmaWxlcyB3cml0dGVuCiAgICBuZXh0IHRvIHRoZSBvdXRwdXQgYXJlIHRoZSBmYWxsYmFjay4KCiAgICBUaGUgd2luZG93IGlzIG9wZW5lZCB3aXRob3V0IGJsb2NraW5nLCBzbyB0aGUgcXVlc3Rpb24gY2FuIGJlIGFuc3dlcmVkIGluIHRoZQogICAgdGVybWluYWwgd2hpbGUgaXQgaXMgc3RpbGwgb24gc2NyZWVuLgoKICAgIFJldHVybnMgVHJ1ZSBpZiBhIHdpbmRvdyBhY3R1YWxseSBhcHBlYXJlZC4KICAgICIiIgogICAgaW1wb3J0IG1hdHBsb3RsaWIKICAgIGltcG9ydCBtYXRwbG90bGliLnB5cGxvdCBhcyBwbHQKCiAgICBkZWYgYmxvY2tfbWVhbihhLCBuKToKICAgICAgICBrMCA9IChhLnNoYXBlWzBdIC8vIG4pICogbgogICAgICAgIGsxID0gKGEuc2hhcGVbMV0gLy8gbikgKiBuCiAgICAgICAgcmV0dXJuIGFbOmswLCA6azFdLnJlc2hhcGUoazAgLy8gbiwgbiwgazEgLy8gbiwgbikubWVhbihheGlzPSgxLCAzKSkKCiAgICBzdGFydGVkID0gbWF0cGxvdGxpYi5nZXRfYmFja2VuZCgpCiAgICBpZiBzdGFydGVkLmxvd2VyKCkgPT0gImFnZyI6CiAgICAgICAgZm9yIGNhbmRpZGF0ZSBpbiAoIlRrQWdnIiwgIlF0QWdnIiwgIlF0NUFnZyIsICJNYWNPU1giKToKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgcGx0LnN3aXRjaF9iYWNrZW5kKGNhbmRpZGF0ZSkKICAgICAgICAgICAgICAgIGJyZWFrCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgaWYgbWF0cGxvdGxpYi5nZXRfYmFja2VuZCgpLmxvd2VyKCkgPT0gImFnZyI6CiAgICAgICAgcmV0dXJuIEZhbHNlICAgICAgICAgICAgICAgICAgICAgIyBub3RoaW5nIG9uIHRoaXMgbWFjaGluZSBjYW4gc2hvdyBhIHdpbmRvdwoKICAgIHRyeToKICAgICAgICBmaWcsIGF4ZXMgPSBwbHQuc3VicGxvdHMoMSwgbGVuKGZyYW1lcyksIGZpZ3NpemU9KDUuMiAqIGxlbihmcmFtZXMpLCA1LjgpKQogICAgICAgIGlmIGxlbihmcmFtZXMpID09IDE6CiAgICAgICAgICAgIGF4ZXMgPSBbYXhlc10KICAgICAgICBmb3IgYXgsIChpLCAoZiwgYmx1ciwgbmV0KSkgaW4gemlwKGF4ZXMsIGVudW1lcmF0ZShmcmFtZXMsIDEpKToKICAgICAgICAgICAgc21hbGwgPSBibG9ja19tZWFuKG5ldCwgc3RlcCkgaWYgc3RlcCA+IDEgZWxzZSBuZXQKICAgICAgICAgICAgYXguaW1zaG93KHNtYWxsLCBvcmlnaW49Imxvd2VyIiwgY21hcD0iZ3JheV9yIiwKICAgICAgICAgICAgICAgICAgICAgIHZtaW49LTIgKiBueiwgdm1heD04ICogbnopCiAgICAgICAgICAgIGF4LnNldF94dGlja3MoW10pOyBheC5zZXRfeXRpY2tzKFtdKQogICAgICAgICAgICBheC5zZXRfdGl0bGUoImNob2ljZSB7fVxuZmFjdG9yIHs6LjJmfSAgIHt9Ii5mb3JtYXQoCiAgICAgICAgICAgICAgICBpLCBmLCBfYmx1cl93b3JkcyhibHVyKSksIGZvbnRzaXplPTEyKQogICAgICAgIGZpZy5zdXB0aXRsZSgiV2hpY2ggc3VidHJhY3Rpb24gaXMgYmVzdD8gTG9vayBhdCB0aGUgc3RhcnMgYW5kIGF0IHRoZSAiCiAgICAgICAgICAgICAgICAgICAgICJlbXB0eSBza3ksIHRoZW4gYW5zd2VyIGluIHRoZSB0ZXJtaW5hbC4iLCBmb250c2l6ZT0xMykKICAgICAgICBmaWcudGlnaHRfbGF5b3V0KCkKICAgICAgICBwbHQuc2hvdyhibG9jaz1GYWxzZSkKICAgICAgICBwbHQucGF1c2UoMC41KQogICAgICAgIHJldHVybiBUcnVlCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHJldHVybiBGYWxzZQogICAgZmluYWxseToKICAgICAgICB0cnk6CiAgICAgICAgICAgIHBsdC5zd2l0Y2hfYmFja2VuZChzdGFydGVkKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHBhc3MKCmRlZiBfYmx1cl93b3JkcyhibHVyKToKICAgICIiIlRoZSBibHVyIGFzIGEgcGhyYXNlLCBzbyB0aGUgcGljdHVyZSBzYXlzIHdoaWNoIGZyYW1lIHdhcyBibHVycmVkLiIiIgogICAgaWYgYmx1ciA+IDA6CiAgICAgICAgcmV0dXJuICJILWFscGhhIGJsdXJyZWQgYnkgezouMmZ9IHB4Ii5mb3JtYXQoYmx1cikKICAgIGlmIGJsdXIgPCAwOgogICAgICAgIHJldHVybiAiUiBibHVycmVkIGJ5IHs6LjJmfSBweCIuZm9ybWF0KC1ibHVyKQogICAgcmV0dXJuICJuZWl0aGVyIGZyYW1lIGJsdXJyZWQiCgpkZWYgX3dyaXRlX2Nob2ljZXMoaGEsIHJlZCwgcGlja3MsIHByZWZpeCwgZGlhZywgYXNrKToKICAgICIiIldyaXRlIG9uZSBwaWN0dXJlIHBlciBjYW5kaWRhdGUsIGFzaywgYW5kIGFwcGx5IHRoZSBhbnN3ZXIuIiIiCiAgICBpbXBvcnQgbWF0cGxvdGxpYgogICAgbWF0cGxvdGxpYi51c2UoIkFnZyIpCiAgICBpbXBvcnQgbWF0cGxvdGxpYi5weXBsb3QgYXMgcGx0CiAgICBmcm9tIHNjaXB5Lm5kaW1hZ2UgaW1wb3J0IGdhdXNzaWFuX2ZpbHRlcgogICAgZnJvbSBhc3Ryb3B5LnN0YXRzIGltcG9ydCBzaWdtYV9jbGlwcGVkX3N0YXRzCgogICAgZGVmIGJsb2NrX21lYW4oYSwgbik6CiAgICAgICAgazAgPSAoYS5zaGFwZVswXSAvLyBuKSAqIG4KICAgICAgICBrMSA9IChhLnNoYXBlWzFdIC8vIG4pICogbgogICAgICAgIHJldHVybiBhWzprMCwgOmsxXS5yZXNoYXBlKGswIC8vIG4sIG4sIGsxIC8vIG4sIG4pLm1lYW4oYXhpcz0oMSwgMykpCgogICAgc3RlcCA9IG1heCgxLCBpbnQocm91bmQobWF4KGhhLnNoYXBlKSAvIDE0MDAuMCkpKQogICAgZnJhbWVzLCBwYXRocyA9IFtdLCBbXQogICAgZm9yIGksIChsZWZ0LCBmLCBibHVyKSBpbiBlbnVtZXJhdGUocGlja3MsIDEpOgogICAgICAgIEEgPSBnYXVzc2lhbl9maWx0ZXIoaGEsIGJsdXIpIGlmIGJsdXIgPiAwIGVsc2UgaGEKICAgICAgICBSID0gZ2F1c3NpYW5fZmlsdGVyKHJlZCwgLWJsdXIpIGlmIGJsdXIgPCAwIGVsc2UgcmVkCiAgICAgICAgZnJhbWVzLmFwcGVuZCgoZiwgYmx1ciwgZiAqIEEgLSBSKSkKCiAgICAjIE9uZSBzdHJldGNoIGZvciBhbGwgdGhyZWUsIG9yIHRoZSBleWUgY29tcGFyZXMgdGhyZWUgZGlmZmVyZW50IHN0cmV0Y2hlcy4KICAgIG1pZGRsZSA9IGZyYW1lc1tsZW4oZnJhbWVzKSAvLyAyXVsyXQogICAgbnogPSBmbG9hdChzaWdtYV9jbGlwcGVkX3N0YXRzKG1pZGRsZSwgc2lnbWE9My4wKVsyXSkKCiAgICBmb3IgaSwgKGYsIGJsdXIsIG5ldCkgaW4gZW51bWVyYXRlKGZyYW1lcywgMSk6CiAgICAgICAgc21hbGwgPSBibG9ja19tZWFuKG5ldCwgc3RlcCkgaWYgc3RlcCA+IDEgZWxzZSBuZXQKICAgICAgICBmaWcsIGF4ID0gcGx0LnN1YnBsb3RzKGZpZ3NpemU9KDksIDkpKQogICAgICAgICMgRGFyayBvbiBsaWdodCwgdGhlIHdheSB0aGlzIGlzIGxvb2tlZCBhdCBieSBleWUuCiAgICAgICAgYXguaW1zaG93KHNtYWxsLCBvcmlnaW49Imxvd2VyIiwgY21hcD0iZ3JheV9yIiwgdm1pbj0tMiAqIG56LCB2bWF4PTggKiBueikKICAgICAgICBheC5zZXRfeHRpY2tzKFtdKTsgYXguc2V0X3l0aWNrcyhbXSkKICAgICAgICAjIFRoZSBudW1iZXJzIGJlbG9uZyBvbiB0aGUgcGljdHVyZS4gQSBjaG9pY2UgbWFkZSBmcm9tIHRocmVlIHVubGFiZWxsZWQKICAgICAgICAjIGZyYW1lcyBjYW5ub3QgYmUgd3JpdHRlbiBkb3duLCByZXBlYXRlZCwgb3IgYXJndWVkIHdpdGggYWZ0ZXJ3YXJkcy4KICAgICAgICBheC5zZXRfdGl0bGUoImNob2ljZSB7fVxuZmFjdG9yIHs6LjJmfSAgIHt9Ii5mb3JtYXQoaSwgZiwgX2JsdXJfd29yZHMoYmx1cikpLAogICAgICAgICAgICAgICAgICAgICBmb250c2l6ZT0xNCkKICAgICAgICBmaWcudGlnaHRfbGF5b3V0KCkKICAgICAgICBwYXRoID0gInt9X2Nob2ljZV97fS5wbmciLmZvcm1hdChwcmVmaXgsIGkpCiAgICAgICAgZmlnLnNhdmVmaWcocGF0aCwgZHBpPTEwMCwgYmJveF9pbmNoZXM9InRpZ2h0IikKICAgICAgICBwbHQuY2xvc2UoZmlnKQogICAgICAgIHBhdGhzLmFwcGVuZChwYXRoKQogICAgZGlhZ1sicGF0aHMiXSA9IHBhdGhzCiAgICBkaWFnWyJwaWNrcyJdID0gWyhmLCBibHVyKSBmb3IgZiwgYmx1ciwgXyBpbiBmcmFtZXNdCgogICAgIyBXSFkgVEhFUkUgSVMgTk8gV0lORE9XCiAgICAjIFRoZXJlIHdhcyBvbmUsIGFuZCBpdCBkaWQgbm90IHdvcmsuIEEgZmlndXJlIG9wZW5lZCB3aXRob3V0IGJsb2NraW5nLCBzbwogICAgIyB0aGF0IHRoZSBxdWVzdGlvbiBjb3VsZCBzdGlsbCBiZSBhbnN3ZXJlZCBpbiB0aGUgdGVybWluYWwsIGlzIGEgd2luZG93IHdpdGgKICAgICMgbm90aGluZyBkcml2aW5nIGl0cyBldmVudCBsb29wOiBpdCBhcHBlYXJzIGFzIGFuIGVtcHR5IGdyZXkgcmVjdGFuZ2xlIGFuZAogICAgIyBvbmx5IGRyYXdzIGl0c2VsZiBvbmNlIGl0IGlzIGNsb3NlZCwgd2hpY2ggaXMgYWZ0ZXIgdGhlIGFuc3dlciBoYXMgYmVlbgogICAgIyBnaXZlbi4gQSBwaWN0dXJlIHRoYXQgY2FuIG9ubHkgYmUgc2VlbiBhZnRlciBpdCBpcyBubyBsb25nZXIgbmVlZGVkIGlzIHdvcnNlCiAgICAjIHRoYW4gbm8gcGljdHVyZSwgYmVjYXVzZSBpdCBhbHNvIGZyZWV6ZXMgdGhlIHNjcmVlbi4KICAgICMKICAgICMgU28gdGhlIHRocmVlIGFyZSB3cml0dGVuIHRvIGRpc2sgYW5kIHRoZWlyIG5hbWVzIGFyZSBwcmludGVkLiBPcGVuIHRoZW0gaW4KICAgICMgd2hhdGV2ZXIgeW91IG5vcm1hbGx5IGxvb2sgYXQgaW1hZ2VzIHdpdGgsIG9yIGRvIG5vdCAtIHRoZSBudW1iZXJzIGFyZSBvbgogICAgIyB0aGUgc2NyZWVuIGVpdGhlciB3YXksIGFuZCB0aGUgbWlkZGxlIG9uZSBpcyB0aGUgZGVmYXVsdC4KICAgIHNob3duID0gRmFsc2UKCiAgICBwcmludCgiIikKICAgIHByaW50KCIgIFRocmVlIHN1YnRyYWN0aW9ucyBvZiB0aGUgc2FtZSBmcmFtZXMsIHdyaXR0ZW4gbmV4dCB0byB0aGUgb3RoZXIiKQogICAgcHJpbnQoIiAgb3V0cHV0LiBMb29rIGF0IHRoZSBzdGFycyBhbmQgYXQgdGhlIGVtcHR5IHNreSBpbiB0aGVtLCB0aGVuIHNheSIpCiAgICBwcmludCgiICB3aGljaCBvbmUgaXMgYmVzdC4gRW50ZXIgdGFrZXMgdGhlIG1pZGRsZSBvbmUuIikKICAgIHByaW50KCIgICAgaW4gIiArIChvcy5wYXRoLmRpcm5hbWUocGF0aHNbMF0pIG9yICIuIikpCiAgICBmb3IgaSwgKHAsIChmLCBibHVyLCBfKSkgaW4gZW51bWVyYXRlKHppcChwYXRocywgZnJhbWVzKSwgMSk6CiAgICAgICAgcHJpbnQoIiAgICB7fS4gezo8MjR9IGZhY3RvciB7Oi4yZn0sIHt9Ii5mb3JtYXQoCiAgICAgICAgICAgIGksIG9zLnBhdGguYmFzZW5hbWUocCksIGYsIF9ibHVyX3dvcmRzKGJsdXIpKSkKCiAgICBhbnN3ZXIgPSAyIGlmIGxlbihwYXRocykgPj0gMiBlbHNlIDEKICAgIGlmIGFzayBpcyBub3QgTm9uZToKICAgICAgICB3aGlsZSBUcnVlOgogICAgICAgICAgICBnb3QgPSBhc2soIldoaWNoIG9uZSIsIHN0cihhbnN3ZXIpLCBpbnQpCiAgICAgICAgICAgIGlmIDEgPD0gZ290IDw9IGxlbihwYXRocyk6CiAgICAgICAgICAgICAgICBhbnN3ZXIgPSBnb3QKICAgICAgICAgICAgICAgIGJyZWFrCiAgICAgICAgICAgIHByaW50KCIgICAgYSBudW1iZXIgZnJvbSB0aGUgbGlzdCwgcGxlYXNlIikKICAgIGVsc2U6CiAgICAgICAgcHJpbnQoIiAgICBub3RoaW5nIHRvIGFzayB3aXRoLCBzbyB7fSBpcyB1c2VkIi5mb3JtYXQoYW5zd2VyKSkKCiAgICBpZiBzaG93bjoKICAgICAgICB0cnk6CiAgICAgICAgICAgIGltcG9ydCBtYXRwbG90bGliLnB5cGxvdCBhcyBwbHQKICAgICAgICAgICAgcGx0LmNsb3NlKCJhbGwiKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHBhc3MKCiAgICBmLCBibHVyLCBfID0gZnJhbWVzW2Fuc3dlciAtIDFdCiAgICBpZiBibHVyID4gMDoKICAgICAgICBoYSA9IGdhdXNzaWFuX2ZpbHRlcihoYSwgYmx1cikKICAgIGVsaWYgYmx1ciA8IDA6CiAgICAgICAgcmVkID0gZ2F1c3NpYW5fZmlsdGVyKHJlZCwgLWJsdXIpCiAgICBkaWFnWyJjaG9zZW4iXSA9IGFuc3dlcgogICAgZGlhZ1siZmFjdG9yIl0gPSBmCiAgICBkaWFnWyJibHVyIl0gPSBibHVyCiAgICByZXR1cm4gaGEsIHJlZCwgZiwgZGlhZwoKZGVmIGRlc2NyaWJlX2Nob2ljZShkaWFnKToKICAgICIiIldoYXQgd2FzIGNob3NlbiwgaW4gYSBsaW5lLiIiIgogICAgaWYgZGlhZy5nZXQoIndoeSIpOgogICAgICAgIHJldHVybiAiICBubyBjaG9pY2Ugb2ZmZXJlZCAoe30pIi5mb3JtYXQoZGlhZ1sid2h5Il0pCiAgICBmLCBibHVyID0gZGlhZ1siZmFjdG9yIl0sIGRpYWdbImJsdXIiXQogICAgd2hhdCA9ICgiSC1hbHBoYSBibHVycmVkIGJ5IHs6LjJmfSBweCIuZm9ybWF0KGJsdXIpIGlmIGJsdXIgPiAwIGVsc2UKICAgICAgICAgICAgIlIgYmx1cnJlZCBieSB7Oi4yZn0gcHgiLmZvcm1hdCgtYmx1cikgaWYgYmx1ciA8IDAgZWxzZQogICAgICAgICAgICAibmVpdGhlciBmcmFtZSBibHVycmVkIikKICAgIHJldHVybiAoIiAgY2hvaWNlIHt9IHRha2VuOiBmYWN0b3IgezouMmZ9LCB7fVxuIgogICAgICAgICAgICAiICAgIGp1ZGdlZCBvbiB7fSBzdGFycyBvdXRzaWRlIHRoZSBnYWxheHkiCiAgICAgICAgICAgIC5mb3JtYXQoZGlhZ1siY2hvc2VuIl0sIGYsIHdoYXQsIGRpYWdbIm5fc3RhcnMiXSkpCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIEJsdWVfQ2x1c3RlcnNfRnJvbV9JbWFnZXMucHkKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQoKZnJvbSBzY2lweS5zcGF0aWFsIGltcG9ydCBjS0RUcmVlCmZyb20gcGhvdHV0aWxzLmFwZXJ0dXJlIGltcG9ydCBBcGVydHVyZVN0YXRzCgojIC0tLS0tLS0tLS0tLS0tLS0tLS0gUGFyYW1ldGVycyAtLS0tLS0tLS0tLS0tLS0tLS0tCiMgTWVhc3VyaW5nIHRoZSBzb3VyY2VzCkFQRVJUVVJFX1NDQUxFID0gMiAgICAgICAgICAgICAgICMgYXBlcnR1cmUgcmFkaXVzID0gdGhpcyB4IHRoZSB0eXBpY2FsIHN0YXIgRldITSBvZiB0aGUKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIyBmcmFtZXMgLSBPTkUgcmFkaXVzIGZvciBldmVyeSBzb3VyY2UsIGluIGJvdGggYmFuZHMKQU5OVUxVU19JTk5FUl9TQ0FMRSA9IDIuNSAgICAgICAgIyB0aGUgc2t5IHJpbmcgYXJvdW5kIGVhY2ggc291cmNlLCBpbiB1bml0cyBvZiB0aGUKQU5OVUxVU19PVVRFUl9TQ0FMRSA9IDMgICAgICAgICAgIyBhcGVydHVyZSByYWRpdXMKRldITV9XSU5ET1dfU0laRSA9IDEwICAgICAgICAgICAgIyBoYWxmLXNpemUgb2YgdGhlIGJveCBhIHNvdXJjZSdzIHdpZHRoIGlzIG1lYXN1cmVkIGluCkRBT0ZJTkRfRldITSA9IE5vbmUgICAgICAgICAgICAgICMgTm9uZSA9IG1lYXN1cmUgaXQgZnJvbSB0aGUgZGF0YSAocmVjb21tZW5kZWQpClNFRUlOR19GQUxMQkFDSyA9IDQuMCAgICAgICAgICAgICMgVXNlZCBvbmx5IGlmIHRoZSBtZWFzdXJlbWVudCBjYW5ub3QgYmUgbWFkZQpTRUVJTkdfTUlOLCBTRUVJTkdfTUFYID0gMS41LCAxNS4wICAgIyBTYW5pdHkgcmFuZ2UgZm9yIHRoZSBtZWFzdXJlZCB2YWx1ZSwgcGl4ZWxzClNJR01BX0NMSVAgPSAzLjAgICAgICAgICAgICAgICAgICMgU2lnbWEgY2xpcHBpbmcgbGV2ZWwgZm9yIGJhY2tncm91bmQgc3RhdGlzdGljcwpERVRFQ1RJT05fVEhSRVNIT0xEX1NJR01BID0gMy4wICAjIERldGVjdGlvbiB0aHJlc2hvbGQgaW4gdW5pdHMgb2YgYmFja2dyb3VuZCBzaWdtYQpQRUFLX01JTl9TVEQgPSAzLjAgICAgICAgICAgICAgICAjIE1pbmltdW0gcGVhayBzaWduYWwtdG8tbm9pc2UgcmF0aW8gZm9yIGEgZGV0ZWN0aW9uCkJBQ0tHUk9VTkRfQk9YX1NJWkUgPSA0OCAgICAgICAgICMgR3JpZCBjZWxsIGZvciB0aGUgbG9jYWwgYmFja2dyb3VuZCwgcGl4ZWxzCkJBQ0tHUk9VTkRfRklMVEVSX1NJWkUgPSAzICAgICAgICMgTWVkaWFuIGZpbHRlciBhcHBsaWVkIGFjcm9zcyB0aGUgZ3JpZApTRUVJTkdfTUVBU1VSRUQgPSB7fQpCR19TVUJfRlVOQyA9IG5wLm5hbm1lZGlhbiAgICAgICAjIEZ1bmN0aW9uIHVzZWQgdG8gY29tcHV0ZSBhbmQgc3VidHJhY3QgYmFja2dyb3VuZCBsZXZlbApBU1RST0FMSUdOX01BWF9TQ0FMRV9FUlJPUiA9IDAuMDUgICMgNSUKR0FJTl9FX1BFUl9BRFUgPSAxLjAgICAgICAgICAgICAgIyBlbGVjdHJvbnMgcGVyIEFEVSwgdXNlZCBvbmx5IGlmIHRoZSBoZWFkZXIgaGFzIG5vbmUKUkVBRF9OT0lTRV9FID0gNS4wICAgICAgICAgICAgICAgIyByZWFkIG5vaXNlIGluIGVsZWN0cm9ucywgdXNlZCBvbmx5IGlmIHRoZSBoZWFkZXIgaGFzIG5vbmUKRURHRV9NQVJHSU5fUFggPSAzMAoKIyBDYWxpYnJhdGlvbgpDQUxJQl9DQVRBTE9HVUUgPSAiQVBBU1MiICAgICAgICAjIGRlZmF1bHQ7ICJHQUlBIiBmb3IgR2FpYSBEUjMgc3ludGhldGljIEIgYW5kIFYKQ0FMSUJfTUFUQ0hfUkFESVVTID0gMy4wICAgICAgICAgIyBhcmNzZWM6IGEgY2F0YWxvZ3VlIHN0YXIgYW5kIGEgc291cmNlIHRoaXMgY2xvc2UgYXJlIG9uZQpYWV9NQVRDSF9UT0xFUkFOQ0UgPSAzLjAgICAgICAgICAjIHBpeGVscywgb24gdGhlIFYgZ3JpZCwgYWZ0ZXIgdGhlIHBsYXRlIHNvbHV0aW9uIGlzIGNoZWNrZWQKQ0FMSUJfTUlOX1NUQVJTID0gNSAgICAgICAgICAgICAgIyBmZXdlciBjYWxpYnJhdGlvbiBzdGFycyB0aGFuIHRoaXMgYW5kIHRoZSBydW4gc3RvcHMKQ0FMSUJfTUFYX0lOU1RfRVJSID0gMC4wNSAgICAgICAgIyBtYWc7IG9ubHkgd2VsbC1tZWFzdXJlZCBzdGFycyBzZXQgdGhlIGNhbGlicmF0aW9uCkNBTElCX01BWF9DQVRfRVJSID0gMC4xMCAgICAgICAgICMgbWFnOyBhbmQgb25seSB3ZWxsLW1lYXN1cmVkIGNhdGFsb2d1ZSBlbnRyaWVzCkNBTElCX0NMSVBfU0lHTUEgPSAzLjAgICAgICAgICAgICMgc3RhcnMgZnVydGhlciB0aGFuIHRoaXMgZnJvbSB0aGUgZml0IGFyZSBsZWZ0IG91dApDQUxJQl9NSU5fT1VUU0lERSA9IDEwICAgICAgICAgICAjIHVzZSBvbmx5IHN0YXJzIGJleW9uZCBSMjUgd2hlbiBhdCBsZWFzdCB0aGlzIG1hbnkgYXJlCldDU19DSEVDS19NSU5fRlJBQ1RJT04gPSAwLjUgICAgICMgc2hhcmUgb2YgYnJpZ2h0IHN0YXJzIHRoYXQgbXVzdCBoYXZlIGEgR2FpYSBzdGFyIG5lYXJieQoKIyBUaGUgYmx1ZSBrbm90cwpDTURfQ09MT1JfTUlOID0gLTAuNSAgICAgICAgICAgICAjIGEgYmx1ZSBrbm90IGhhcyAoQi1WKTAgYmV0d2VlbiB0aGVzZSB0d28gdmFsdWVzCkNNRF9DT0xPUl9NQVggPSAwLjUKS05PVF9NQVhfQlZfRVJSID0gMC4yICAgICAgICAgICAgIyBhbmQgYSBCLVYgZXJyb3Igbm8gbGFyZ2VyIHRoYW4gdGhpcywgaW4gbWFnCkNNRF9NQVhfQlZfRVJSID0gMC4zICAgICAgICAgICAgICMgc291cmNlcyBub2lzaWVyIHRoYW4gdGhpcyBhcmUgbGVmdCBvdXQgb2YgdGhlIGRpYWdyYW1zCkZPUkVHUk9VTkRfVE9MID0gMy4wICAgICAgICAgICAgICMgcGl4ZWxzOiBhIHNvdXJjZSB0aGlzIGNsb3NlIHRvIGEgR2FpYSBmb3JlZ3JvdW5kIHN0YXIgaXMgaXQKU3RlcF9zaXplID0gMTAgICAgICAgICAgICAgICAgICAgIyBSZXNvbHV0aW9uIG9mIFRoZSBCbHVlIEtub3RzIERlbnNpdHkgRGlhZ3JhbQoKUl9WID0gMy4xICAgICAgICAgICAgICAgICAgICAgICAgIyBBX1YgLyBFKEItVikgZm9yIGR1c3QgaW4gb3VyIG93biBnYWxheHkKCkFSR1MgPSBOb25lICAgICAgICAgICAgICAgICAgICAgICMgdGhlIGNvbW1hbmQtbGluZSBhcmd1bWVudHMsIGlmIGFueQoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLSBRdWVzdGlvbnMgLS0tLS0tLS0tLS0tLS0tLS0tLQoKZGVmIF9hY2NlcHRfZGVmYXVsdHMoKToKICAgIHJldHVybiBBUkdTIGlzIG5vdCBOb25lIGFuZCBnZXRhdHRyKEFSR1MsICJ5ZXMiLCBGYWxzZSkKCgpkZWYgX2Fzayhwcm9tcHQsIGRlZmF1bHQsIGNhc3Q9c3RyLCBjaG9pY2VzPU5vbmUpOgogICAgIiIiQXNrIG9uZSBxdWVzdGlvbi4gRW50ZXIgYWNjZXB0cyB0aGUgZGVmYXVsdCBzaG93biBpbiBicmFja2V0cy4KCiAgICBPbmx5IGFuIGVtcHR5IGFuc3dlciBtZWFucyAidGhlIGRlZmF1bHQiLiBBbiBhbnN3ZXIgdGhhdCBjYW5ub3QgYmUgcmVhZCBpcwogICAgcmVmdXNlZCBhbmQgdGhlIHF1ZXN0aW9uIGFza2VkIGFnYWluIC0gaXQgdXNlZCB0byBiZSByZXBsYWNlZCBieSB0aGUKICAgIGRlZmF1bHQgd2l0aG91dCBhIHdvcmQsIHNvIGEgbWlzdHlwZWQgZGlzdGFuY2UgYmVjYW1lIDEwIE1wYy4gQSBkZWZhdWx0IG9mCiAgICBOb25lIG1lYW5zIHRoZXJlIGlzIG5vbmU6IGFuIGFuc3dlciBpcyBuZWVkZWQuCiAgICAiIiIKICAgIGlmIF9hY2NlcHRfZGVmYXVsdHMoKToKICAgICAgICBpZiBkZWZhdWx0IGlzIE5vbmU6CiAgICAgICAgICAgIHN5cy5leGl0KGYiW2lucHV0XSAne3Byb21wdH0nIGhhcyBubyBkZWZhdWx0LCBzbyAtLXllcyBjYW5ub3QgYW5zd2VyICIKICAgICAgICAgICAgICAgICAgICAgZiJpdCAtIGdpdmUgaXQgb24gdGhlIGNvbW1hbmQgbGluZSIpCiAgICAgICAgcHJpbnQoZiJ7cHJvbXB0fSBbe2RlZmF1bHR9XTogKGFjY2VwdGVkLCAtLXllcykiKQogICAgICAgIHJldHVybiBkZWZhdWx0CiAgICBzaG93biA9ICIiIGlmIGRlZmF1bHQgaXMgTm9uZSBlbHNlIGYiIFt7ZGVmYXVsdH1dIgogICAgd2hpbGUgVHJ1ZToKICAgICAgICBzID0gaW5wdXQoZiJ7cHJvbXB0fXtzaG93bn06ICIpLnN0cmlwKCkKICAgICAgICBpZiBzID09ICIiOgogICAgICAgICAgICBpZiBkZWZhdWx0IGlzIE5vbmU6CiAgICAgICAgICAgICAgICBwcmludCgiICAgYW4gYW5zd2VyIGlzIG5lZWRlZCBoZXJlIikKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgIHJldHVybiBkZWZhdWx0CiAgICAgICAgdHJ5OgogICAgICAgICAgICB2ID0gY2FzdChzKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHByaW50KGYiICAgY291bGQgbm90IHJlYWQgJ3tzfScgLSB0cnkgYWdhaW4iKQogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGlmIGNob2ljZXMgYW5kIHYgbm90IGluIGNob2ljZXM6CiAgICAgICAgICAgIHByaW50KGYiICAgYW5zd2VyIG9uZSBvZjogeycsICcuam9pbihjaG9pY2VzKX0iKQogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHJldHVybiB2CgoKZGVmIF9ub3JtX3BhdGgocDogc3RyKSAtPiBzdHI6CiAgICAiIiJSZW1vdmUgZXh0cmEgcXVvdGVzIGFuZCBub3JtYWxpemUgZmlsZXN5c3RlbSBwYXRoLiIiIgogICAgcCA9IHAuc3RyaXAoKS5zdHJpcCgnIicpLnN0cmlwKCInIikKICAgIHJldHVybiBvcy5wYXRoLm5vcm1wYXRoKHApIGlmIHAgZWxzZSAiIgoKCmRlZiBvZmZlcmVkKGxhYmVsLCB2YWx1ZSwgc291cmNlLCB1bml0PSIiLCBvdmVycmlkZT1Ob25lKToKICAgICIiIkEgY2F0YWxvZ3VlIHZhbHVlLCBzaG93biB3aXRoIHdoZXJlIGl0IGNhbWUgZnJvbSwgYW5kIGFjY2VwdGVkIHdpdGgKICAgIEVudGVyIG9yIHJlcGxhY2VkIGJ5IHR5cGluZyBhbm90aGVyLiBBIHZhbHVlIGdpdmVuIG9uIHRoZSBjb21tYW5kIGxpbmUKICAgIGlzIHVzZWQgYXMgaXQgaXMuIiIiCiAgICBpZiBvdmVycmlkZSBpcyBub3QgTm9uZToKICAgICAgICBwcmludChmIntsYWJlbH06IHtvdmVycmlkZX17dW5pdH0gKGZyb20gdGhlIGNvbW1hbmQgbGluZSkiKQogICAgICAgIHJldHVybiBmbG9hdChvdmVycmlkZSkKICAgIGlmIHZhbHVlIGlzIE5vbmU6CiAgICAgICAgcHJpbnQoZiJbbG9va3VwXSBubyB7bGFiZWwubG93ZXIoKX0gZm91bmQgaW4gdGhlIGNhdGFsb2d1ZSAtIHBsZWFzZSBnaXZlIG9uZSIpCiAgICAgICAgcmV0dXJuIGZsb2F0KF9hc2sobGFiZWwsIE5vbmUsIGZsb2F0KSkKICAgIHByaW50KGYiW2xvb2t1cF0ge2xhYmVsfToge3ZhbHVlOmd9e3VuaXR9ICh7c291cmNlfSkiKQogICAgcmV0dXJuIGZsb2F0KF9hc2soZiJ7bGFiZWx9IC0gRW50ZXIgdG8gYWNjZXB0LCBvciB0eXBlIGFub3RoZXIiLCB2YWx1ZSwgZmxvYXQpKQoKCmRlZiBzdG9wKG1lc3NhZ2UpOgogICAgIiIiU3RvcCB0aGUgcnVuIHdpdGggYSBtZXNzYWdlIHRoYXQgc2F5cyB3aGF0IHRvIGRvLiIiIgogICAgcHJpbnQoIiIpCiAgICBwcmludCgiKiIgKiA3OCkKICAgIHByaW50KCJTVE9QUEVEOiAiICsgbWVzc2FnZSkKICAgIHByaW50KCIqIiAqIDc4KQogICAgcmFpc2UgU3lzdGVtRXhpdCgxKQoKCmRlZiBsb3VkKG1lc3NhZ2UpOgogICAgcHJpbnQoIiIpCiAgICBwcmludCgiISIgKiA3OCkKICAgIHByaW50KG1lc3NhZ2UpCiAgICBwcmludCgiISIgKiA3OCkKICAgIHByaW50KCIiKQoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLSBSZWFkaW5nIGZyYW1lcyBhbmQgdGhlIGRldGVjdG9yIC0tLS0tLS0tLS0tLS0tLS0tLS0KCmRlZiByZWFkX2ltYWdlKHBhdGgpOgogICAgIiIiKGRhdGEgYXMgZmxvYXQzMiwgaGVhZGVyKSBmcm9tIHRoZSBmaXJzdCBIRFUgdGhhdCBob2xkcyBhIDItRCBpbWFnZS4KCiAgICBUaGUgZmlyc3QgSERVIGlzIG5vdCBhbHdheXMgaXQ6IGFuIExDTyAuZml0cy5meiBmcmFtZSBrZWVwcyBpdHMgaW1hZ2UgaW4KICAgIHRoZSBzZWNvbmQsIGFuZCByZWFkaW5nIHRoZSBmaXJzdCByZXR1cm5lZCBub3RoaW5nLgogICAgIiIiCiAgICB3aXRoIGZpdHMub3BlbihwYXRoKSBhcyBoZHVsOgogICAgICAgIGZvciBoZHUgaW4gaGR1bDoKICAgICAgICAgICAgaWYgaGR1LmRhdGEgaXMgbm90IE5vbmUgYW5kIGdldGF0dHIoaGR1LmRhdGEsICJuZGltIiwgMCkgPT0gMjoKICAgICAgICAgICAgICAgIGhlYWRlciA9IGhkdS5oZWFkZXIuY29weSgpCiAgICAgICAgICAgICAgICBmb3Iga2V5IGluICgiWFRFTlNJT04iLCAiUENPVU5UIiwgIkdDT1VOVCIsICJCWkVSTyIsICJCU0NBTEUiLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIkJMQU5LIiwgIlpJTUFHRSIsICJaQklUUElYIiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICJaTkFYSVMiLCAiWk5BWElTMSIsICJaTkFYSVMyIiwgIlpUSUxFMSIsICJaVElMRTIiLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIlpDTVBUWVBFIiwgIlpRVUFOVElaIiwgIlpESVRIRVIwIiwgIlpTSU1QTEUiLCAiWkVYVEVORCIpOgogICAgICAgICAgICAgICAgICAgIGhlYWRlci5yZW1vdmUoa2V5LCBpZ25vcmVfbWlzc2luZz1UcnVlLCByZW1vdmVfYWxsPVRydWUpCiAgICAgICAgICAgICAgICByZXR1cm4gbnAuYXNhcnJheShoZHUuZGF0YSwgZHR5cGU9bnAuZmxvYXQzMiksIGhlYWRlcgogICAgcmFpc2UgUnVudGltZUVycm9yKGYibm8gMi1EIGltYWdlIGluIHtwYXRofSIpCgoKZGVmIGNvbWJpbmVkX2NvdW50KGhlYWRlcik6CiAgICAiIiJIb3cgbWFueSBmcmFtZXMgd2VudCBpbnRvIHRoaXMgZmlsZSwgYW5kIGhvdyB0aGV5IHdlcmUgY29tYmluZWQuIiIiCiAgICBuID0gaGVhZGVyLmdldCgiTkNPTUJJTkUiKQogICAgaG93ID0gc3RyKGhlYWRlci5nZXQoIkNPTUJUWVBFIiwgIm1lZGlhbiIpKS5sb3dlcigpCiAgICB0cnk6CiAgICAgICAgaWYgbiBhbmQgaW50KG4pID4gMToKICAgICAgICAgICAgcmV0dXJuIGludChuKSwgX2NvbWJpbmVfa2luZChob3cpCiAgICBleGNlcHQgKFR5cGVFcnJvciwgVmFsdWVFcnJvcik6CiAgICAgICAgcGFzcwogICAgZm9yIGxpbmUgaW4gaGVhZGVyLmdldCgiSElTVE9SWSIsIFtdKToKICAgICAgICBtID0gcmUuc2VhcmNoKHIiKG1lZGlhbnxhdmVyYWdlfG1lYW58c3VtKVxzK29mXHMrKFxkKylccysiCiAgICAgICAgICAgICAgICAgICAgICByIig/OmltYWdlc3xleHBvc3VyZXN8ZnJhbWVzKSIsIHN0cihsaW5lKSwgcmUuSSkKICAgICAgICBpZiBtOgogICAgICAgICAgICByZXR1cm4gaW50KG0uZ3JvdXAoMikpLCBfY29tYmluZV9raW5kKG0uZ3JvdXAoMSkpCiAgICByZXR1cm4gMSwgInNpbmdsZSIKCgpkZWYgX2NvbWJpbmVfa2luZCh3b3JkKToKICAgIHdvcmQgPSBzdHIod29yZCkubG93ZXIoKQogICAgcmV0dXJuICJtZWRpYW4iIGlmICJtZWQiIGluIHdvcmQgZWxzZSAic3VtIiBpZiAic3VtIiBpbiB3b3JkIGVsc2UgIm1lYW4iCgoKZGVmIGRldGVjdG9yX2dhaW4oaGVhZGVyKToKICAgICIiIkVsZWN0cm9ucyBwZXIgQURVIG9mIFRISVMgZmlsZSwgYW5kIHdoZXJlIHRoZSBudW1iZXIgY2FtZSBmcm9tLgoKICAgIEVHQUlOIGlzIHByZWZlcnJlZDogb24gc29tZSBjYW1lcmFzIEdBSU4gaXMgdGhlIGdhaW4gU0VUVElORyAoNTEgb24gdGhlCiAgICBLaW5uZXJldCBjb2xvdXIgY2FtZXJhKSwgbm90IGVsZWN0cm9ucyBwZXIgQURVLiBBIG1lZGlhbiBvZiBOIGZyYW1lcyBoYXMKICAgIHRoZSBub2lzZSBvZiBvbmUgZnJhbWUgd2l0aCBOIHggMi9waSB0aW1lcyB0aGUgZ2FpbiwgYSBtZWFuIG9mIE4gZnJhbWVzCiAgICBOIHRpbWVzOyBhIHN1bSBrZWVwcyB0aGUgZ2FpbiBvZiBvbmUgZnJhbWUuCiAgICAiIiIKICAgIGdhaW4sIHNvdXJjZSA9IE5vbmUsIE5vbmUKICAgIGZvciBrZXkgaW4gKCJFR0FJTiIsICJHQUlOIik6CiAgICAgICAgdHJ5OgogICAgICAgICAgICB2ID0gZmxvYXQoaGVhZGVyLmdldChrZXkpKQogICAgICAgIGV4Y2VwdCAoVHlwZUVycm9yLCBWYWx1ZUVycm9yKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICBpZiAwIDwgdiA8IDIwOgogICAgICAgICAgICBnYWluLCBzb3VyY2UgPSB2LCBrZXkKICAgICAgICAgICAgYnJlYWsKICAgIGlmIGdhaW4gaXMgTm9uZToKICAgICAgICBnYWluLCBzb3VyY2UgPSBHQUlOX0VfUEVSX0FEVSwgImRlZmF1bHQgY29uc3RhbnQiCiAgICBuLCBob3cgPSBjb21iaW5lZF9jb3VudChoZWFkZXIpCiAgICBpZiBuID4gMSBhbmQgaG93ID09ICJzdW0iOgogICAgICAgIHJldHVybiBnYWluLCBmIntzb3VyY2V9IChzdW0gb2Yge259IGZyYW1lcykiCiAgICBpZiBuID4gMToKICAgICAgICBmYWN0b3IgPSBuICogKDIuMCAvIG5wLnBpIGlmIGhvdyA9PSAibWVkaWFuIiBlbHNlIDEuMCkKICAgICAgICByZXR1cm4gZ2FpbiAqIGZhY3RvciwgZiJ7c291cmNlfSB4IHtmYWN0b3I6LjFmfSAoe2hvd30gb2Yge259IGZyYW1lcykiCiAgICByZXR1cm4gZ2Fpbiwgc291cmNlCgoKZGVmIHJlYWRfbm9pc2UoaGVhZGVyKToKICAgIGZvciBrZXkgaW4gKCJSRE5PSVNFIiwgIlJFQUROT0lTIiwgIlJETk9JU0UxIiwgIlJPTiIpOgogICAgICAgIHRyeToKICAgICAgICAgICAgdiA9IGZsb2F0KGhlYWRlci5nZXQoa2V5KSkKICAgICAgICAgICAgaWYgdiA+IDA6CiAgICAgICAgICAgICAgICByZXR1cm4gdiwga2V5CiAgICAgICAgZXhjZXB0IChUeXBlRXJyb3IsIFZhbHVlRXJyb3IpOgogICAgICAgICAgICBjb250aW51ZQogICAgcmV0dXJuIFJFQURfTk9JU0VfRSwgImRlZmF1bHQgY29uc3RhbnQiCgoKZGVmIHNhdHVyYXRpb25fbGV2ZWwocmF3LCBoZWFkZXIpOgogICAgIiIiVGhlIHBpeGVsIHZhbHVlIGFib3ZlIHdoaWNoIGEgc291cmNlIGNvdW50cyBhcyBzYXR1cmF0ZWQsIGFuZCBob3cgaXQKICAgIHdhcyBmb3VuZDogZnJvbSBTQVRVUkFURSAob3IgTUFYTElOKSBpbiB0aGUgaGVhZGVyIHdoZW4gdGhlcmUgaXMgb25lLAogICAgb3RoZXJ3aXNlIGZyb20gdGhlIHBpbGUtdXAgb2Ygc2F0dXJhdGVkIHBpeGVscyBhdCB0aGUgdG9wIG9mIHRoZSBpbWFnZSdzCiAgICBoaXN0b2dyYW0uIiIiCiAgICBmb3Iga2V5IGluICgiU0FUVVJBVEUiLCAiTUFYTElOIiwgIlNBVExFVkVMIik6CiAgICAgICAgdHJ5OgogICAgICAgICAgICB2ID0gZmxvYXQoaGVhZGVyLmdldChrZXkpKQogICAgICAgIGV4Y2VwdCAoVHlwZUVycm9yLCBWYWx1ZUVycm9yKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICBpZiB2ID4gMDoKICAgICAgICAgICAgcmV0dXJuIFNBVFVSQVRJT05fRlJBQ1RJT04gKiB2LCBmIntTQVRVUkFUSU9OX0ZSQUNUSU9OOmd9IHgge2tleX0gKHt2Omd9KSIKICAgIG0gPSBFREdFX01BUkdJTl9QWAogICAgaW5uZXIgPSByYXdbbTotbSwgbTotbV0gaWYgbWluKHJhdy5zaGFwZSkgPiA0ICogbSBlbHNlIHJhdwogICAgaW5uZXIgPSBpbm5lcltucC5pc2Zpbml0ZShpbm5lcildCiAgICB0b3AgPSBmbG9hdChucC5zb3J0KGlubmVyKVstMjBdKQogICAgaGlnaCA9IGlubmVyW2lubmVyID4gMC40ICogdG9wXQogICAgbGV2ZWwsIGhvdyA9IHRvcCwgInRoZSBpbWFnZSdzIG1heGltdW0iCiAgICBpZiBoaWdoLnNpemUgPiAyMDA6CiAgICAgICAgY291bnRzLCBlZGdlcyA9IG5wLmhpc3RvZ3JhbShoaWdoLCBiaW5zPTYwLCByYW5nZT0oMC40ICogdG9wLCB0b3ApKQogICAgICAgIGZvciBpIGluIHJhbmdlKDEwLCBsZW4oY291bnRzKSk6CiAgICAgICAgICAgIGJlZm9yZSA9IG5wLm1lZGlhbihjb3VudHNbaSAtIDEwOmldKQogICAgICAgICAgICBpZiBjb3VudHNbaV0gPj0gMjAgYW5kIGNvdW50c1tpXSA+IDIuMCAqIG1heChiZWZvcmUsIDEuMCk6CiAgICAgICAgICAgICAgICBsZXZlbCA9IGZsb2F0KGVkZ2VzW2ldKQogICAgICAgICAgICAgICAgaG93ID0gInRoZSBwaWxlLXVwIG9mIHNhdHVyYXRlZCBwaXhlbHMgaW4gdGhlIGhpc3RvZ3JhbSIKICAgICAgICAgICAgICAgIGJyZWFrCiAgICByZXR1cm4gU0FUVVJBVElPTl9GUkFDVElPTiAqIGxldmVsLCBmIntTQVRVUkFUSU9OX0ZSQUNUSU9OOmd9IHgge2hvd30gKHtsZXZlbDouMGZ9KSIKCgpkZWYgc3VidHJhY3RfYmFja2dyb3VuZF9hbmRfc2F2ZShwYXRoLCBvdXRfZGlyKToKICAgICIiIlN1YnRyYWN0IHRoZSBmcmFtZSdzIG1lZGlhbiBsZXZlbCBhbmQgc2F2ZSB0aGUgcmVzdWx0IGluIG91dF9kaXIuCgogICAgUmV0dXJucyAocGF0aCB3cml0dGVuLCByYXcgcGl4ZWxzLCBiYWNrZ3JvdW5kLXN1YnRyYWN0ZWQgcGl4ZWxzLCBoZWFkZXIpLgogICAgIiIiCiAgICByYXcsIGhkciA9IHJlYWRfaW1hZ2UocGF0aCkKICAgIGJnX3ZhbCA9IGZsb2F0KEJHX1NVQl9GVU5DKHJhdykpCiAgICBkYXRhX3N1YiA9IHJhdyAtIG5wLmZsb2F0MzIoYmdfdmFsKQogICAgc3RlbSA9IG9zLnBhdGguYmFzZW5hbWUocGF0aCkKICAgIGZvciBzdWZmaXggaW4gKCIuZml0cy5meiIsICIuZml0cyIsICIuZml0IiwgIi5mdHMiKToKICAgICAgICBpZiBzdGVtLmxvd2VyKCkuZW5kc3dpdGgoc3VmZml4KToKICAgICAgICAgICAgc3RlbSA9IHN0ZW1bOi1sZW4oc3VmZml4KV0KICAgICAgICAgICAgYnJlYWsKICAgIG91dF9wYXRoID0gb3MucGF0aC5qb2luKG91dF9kaXIsIHN0ZW0gKyAiX2Jnc3ViLmZpdHMiKQogICAgZml0cy5QcmltYXJ5SERVKGRhdGE9ZGF0YV9zdWIsIGhlYWRlcj1oZHIpLndyaXRldG8ob3V0X3BhdGgsIG92ZXJ3cml0ZT1UcnVlKQogICAgcHJpbnQoZiJbYmdzdWJdIHdyb3RlIHtvdXRfcGF0aH0gKGJnPXtiZ192YWw6LjNmfSkiKQogICAgcmV0dXJuIG91dF9wYXRoLCByYXcsIGRhdGFfc3ViLCBoZHIKCgojIC0tLS0tLS0tLS0tLS0tLS0tLS0gRmluZGluZyBhbmQgbWVhc3VyaW5nIHRoZSBzb3VyY2VzIC0tLS0tLS0tLS0tLS0tLS0tLS0KCmRlZiBsb2NhbF9iYWNrZ3JvdW5kKGRhdGEpOgogICAgIiIiQmFja2dyb3VuZCBhbmQgbm9pc2Ugb24gYSBjb2Fyc2UgZ3JpZCwgc28gYSBnYWxheHkncyBvd24gbGlnaHQgZG9lcyBub3QKICAgIHNldCB0aGUgZGV0ZWN0aW9uIHRocmVzaG9sZCBmb3IgdGhlIGVtcHR5IHNreSBhcm91bmQgaXQuIiIiCiAgICByZXR1cm4gQmFja2dyb3VuZDJEKGRhdGEsIChCQUNLR1JPVU5EX0JPWF9TSVpFLCBCQUNLR1JPVU5EX0JPWF9TSVpFKSwKICAgICAgICAgICAgICAgICAgICAgICAgZmlsdGVyX3NpemU9KEJBQ0tHUk9VTkRfRklMVEVSX1NJWkUsIEJBQ0tHUk9VTkRfRklMVEVSX1NJWkUpLAogICAgICAgICAgICAgICAgICAgICAgICBzaWdtYV9jbGlwPVNpZ21hQ2xpcChzaWdtYT1TSUdNQV9DTElQKSwKICAgICAgICAgICAgICAgICAgICAgICAgYmtnX2VzdGltYXRvcj1NZWRpYW5CYWNrZ3JvdW5kKCkpCgoKZGVmIHN0YXJfd2lkdGgoZGF0YSwgeCwgeSwgaGFsZj0xMiwgc2t5PTAuMCk6CiAgICAiIiJXaWR0aCBhdCBoYWxmIG1heGltdW0gb2Ygb25lIHNvdXJjZSwgaW4gcGl4ZWxzLCBtZWFzdXJlZCBvbiB0aGUgZGF0YSBhcwogICAgaXQgaXMsIGFmdGVyIGl0cyBsb2NhbCBza3kgaXMgdGFrZW4gb2ZmLgoKICAgIEEgdHdvLWRpbWVuc2lvbmFsIEdhdXNzaWFuIGlzIGZpdHRlZCB0byB0aGUgc3RhbXAsIGFuZCB0aGUgRldITSBpcwogICAgMi4zNTUgeCBzcXJ0KHNpZ21hX3ggc2lnbWFfeSkuIENvdW50aW5nIHRoZSBwaXhlbHMgYWJvdmUgaGFsZiB0aGUgcGVhaywgYXMKICAgIHRoaXMgZGlkIGJlZm9yZSwgY2FuIG9ubHkgcmV0dXJuIHdob2xlIGFuZCBoYWxmIHBpeGVsczogYSBtZWRpYW4gb3ZlciBhCiAgICBodW5kcmVkIHN0YXJzIHRoZW4gc2l0cyBvbiA0LjAwIHdoZXRoZXIgdGhlIHNlZWluZyBpcyAzLjYgb3IgNC40IHB4LiBUaGUKICAgIHBpeGVsIGNvdW50IGlzIHN0aWxsIHRoZSBhbnN3ZXIgd2hlbiB0aGUgZml0IGZhaWxzLgogICAgIiIiCiAgICB4LCB5ID0gaW50KHJvdW5kKHgpKSwgaW50KHJvdW5kKHkpKQogICAgaWYgeCAtIGhhbGYgPCAwIG9yIHkgLSBoYWxmIDwgMCBvciB4ICsgaGFsZiA+PSBkYXRhLnNoYXBlWzFdIG9yIHkgKyBoYWxmID49IGRhdGEuc2hhcGVbMF06CiAgICAgICAgcmV0dXJuIG5wLm5hbgogICAgY3V0ID0gbnAuYXNhcnJheShkYXRhW3kgLSBoYWxmOnkgKyBoYWxmLCB4IC0gaGFsZjp4ICsgaGFsZl0sIGZsb2F0KSAtIHNreQogICAgcGVhayA9IGN1dC5tYXgoKQogICAgaWYgbm90IG5wLmlzZmluaXRlKHBlYWspIG9yIHBlYWsgPD0gMDoKICAgICAgICByZXR1cm4gbnAubmFuCiAgICBpZHggPSBucC5hcmd3aGVyZShjdXQgPiBwZWFrIC8gMikKICAgIGlmIG5vdCBsZW4oaWR4KToKICAgICAgICByZXR1cm4gbnAubmFuCiAgICBjb2Fyc2UgPSBucC5tZWFuKFtucC5wdHAoaWR4WzosIDFdKSArIDEsIG5wLnB0cChpZHhbOiwgMF0pICsgMV0pCiAgICB0cnk6CiAgICAgICAgZnJvbSBhc3Ryb3B5Lm1vZGVsaW5nIGltcG9ydCBtb2RlbHMsIGZpdHRpbmcKICAgICAgICB5eSwgeHggPSBucC5tZ3JpZFswOmN1dC5zaGFwZVswXSwgMDpjdXQuc2hhcGVbMV1dCiAgICAgICAgcHksIHB4ID0gbnAudW5yYXZlbF9pbmRleChucC5hcmdtYXgoY3V0KSwgY3V0LnNoYXBlKQogICAgICAgIHMwID0gbWF4KGNvYXJzZSAvIDIuMzU1LCAwLjUpCiAgICAgICAgZyA9IG1vZGVscy5HYXVzc2lhbjJEKHBlYWssIHB4LCBweSwgczAsIHMwLCAwLjApICsgbW9kZWxzLkNvbnN0MkQoMC4wKQogICAgICAgIHdpdGggd2FybmluZ3MuY2F0Y2hfd2FybmluZ3MoKToKICAgICAgICAgICAgd2FybmluZ3Muc2ltcGxlZmlsdGVyKCJpZ25vcmUiKQogICAgICAgICAgICBmaXQgPSBmaXR0aW5nLkxldk1hckxTUUZpdHRlcigpKGcsIHh4LCB5eSwgbnAubmFuX3RvX251bShjdXQpKQogICAgICAgIHN4LCBzeSA9IGFicyhmaXQueF9zdGRkZXZfMC52YWx1ZSksIGFicyhmaXQueV9zdGRkZXZfMC52YWx1ZSkKICAgICAgICBmd2htID0gMi4zNTQ4ICogbnAuc3FydChzeCAqIHN5KQogICAgICAgIGlmIChucC5pc2Zpbml0ZShmd2htKSBhbmQgZml0LmFtcGxpdHVkZV8wLnZhbHVlID4gMCBhbmQgMC41IDwgZndobSA8IDIgKiBoYWxmCiAgICAgICAgICAgICAgICBhbmQgYWJzKGZpdC54X21lYW5fMC52YWx1ZSAtIHB4KSA8IDIgYW5kIGFicyhmaXQueV9tZWFuXzAudmFsdWUgLSBweSkgPCAyKToKICAgICAgICAgICAgcmV0dXJuIGZsb2F0KGZ3aG0pCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHBhc3MKICAgIHJldHVybiBmbG9hdChjb2Fyc2UpCgoKZGVmIF94eV9vZih0YWJsZSk6CiAgICAiIiJ4IGFuZCB5IG9mIGEgREFPU3RhckZpbmRlciB0YWJsZS4gcGhvdHV0aWxzIDMuMCByZW5hbWVkIHhjZW50cm9pZCBhbmQKICAgIHljZW50cm9pZCB0byB4X2NlbnRyb2lkIGFuZCB5X2NlbnRyb2lkOyBib3RoIGFyZSBhY2NlcHRlZC4iIiIKICAgIG5hbWVzID0gdGFibGUuY29sbmFtZXMKICAgIHhzID0gInhfY2VudHJvaWQiIGlmICJ4X2NlbnRyb2lkIiBpbiBuYW1lcyBlbHNlICJ4Y2VudHJvaWQiCiAgICB5cyA9ICJ5X2NlbnRyb2lkIiBpZiAieV9jZW50cm9pZCIgaW4gbmFtZXMgZWxzZSAieWNlbnRyb2lkIgogICAgcmV0dXJuIG5wLmFzYXJyYXkodGFibGVbeHNdLCBmbG9hdCksIG5wLmFzYXJyYXkodGFibGVbeXNdLCBmbG9hdCkKCgpkZWYgZXN0aW1hdGVfc2VlaW5nKGRhdGEsIGJhY2tncm91bmQsIHJtcywgYmFuZCwgcmF3PU5vbmUsIHNhdF9sZXZlbD1Ob25lKToKICAgICIiIkhvdyB3aWRlIGEgc3RhciBpcyBpbiB0aGlzIGZyYW1lOiB0aGUgbWVkaWFuIHdpZHRoIG9mIGJyaWdodCBzdGFycyB0aGF0CiAgICBhcmUgTk9UIHNhdHVyYXRlZC4gQSBzYXR1cmF0ZWQgc3RhciBoYXMgYSBmbGF0IHRvcCwgc28gaXRzIHdpZHRoIGF0IGhhbGYKICAgIHRoZSBmbGF0IHRvcCBzYXlzIG5vdGhpbmcgYWJvdXQgdGhlIHNlZWluZyAtIGFuZCB0aGUgYXBlcnR1cmUgb2YgZXZlcnkKICAgIHNvdXJjZSBpcyBzaXplZCBmcm9tIHRoaXMgbnVtYmVyLiIiIgogICAgdHJ5OgogICAgICAgIGZvdW5kID0gREFPU3RhckZpbmRlcihmd2htPVNFRUlOR19GQUxMQkFDSywgdGhyZXNob2xkPTIwLjAgKiBybXMpKGRhdGEgLSBiYWNrZ3JvdW5kKQogICAgICAgIGlmIGZvdW5kIGlzIG5vdCBOb25lIGFuZCBsZW4oZm91bmQpID49IDU6CiAgICAgICAgICAgIGZ4LCBmeSA9IF94eV9vZihmb3VuZCkKICAgICAgICAgICAgZmx1eCA9IG5wLmFzYXJyYXkoZm91bmRbJ2ZsdXgnXSwgZmxvYXQpCiAgICAgICAgICAgIGlmIHJhdyBpcyBub3QgTm9uZSBhbmQgc2F0X2xldmVsIGlzIG5vdCBOb25lOgogICAgICAgICAgICAgICAgcGVha3MgPSBucC5hcnJheShbbnAubmFubWF4KHJhd1ttYXgoaW50KHJvdW5kKHkpKSAtIDEsIDApOmludChyb3VuZCh5KSkgKyAyLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYXgoaW50KHJvdW5kKHgpKSAtIDEsIDApOmludChyb3VuZCh4KSkgKyAyXSkKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZvciB4LCB5IGluIHppcChmeCwgZnkpXSkKICAgICAgICAgICAgICAgIG9rID0gcGVha3MgPCBzYXRfbGV2ZWwKICAgICAgICAgICAgICAgIGZ4LCBmeSwgZmx1eCA9IGZ4W29rXSwgZnlbb2tdLCBmbHV4W29rXQogICAgICAgICAgICBpZiBsZW4oZngpID49IDU6CiAgICAgICAgICAgICAgICBvcmRlciA9IG5wLmFyZ3NvcnQoZmx1eClbLTIwMDpdCiAgICAgICAgICAgICAgICB3aWR0aHMgPSBucC5hc2FycmF5KFtzdGFyX3dpZHRoKGRhdGEgLSBiYWNrZ3JvdW5kLCBmeFtpXSwgZnlbaV0pCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBmb3IgaSBpbiBvcmRlcl0sIGR0eXBlPWZsb2F0KQogICAgICAgICAgICAgICAgc2VlaW5nID0gbnAubmFubWVkaWFuKHdpZHRocykKICAgICAgICAgICAgICAgIGlmIG5wLmlzZmluaXRlKHNlZWluZykgYW5kIFNFRUlOR19NSU4gPD0gc2VlaW5nIDw9IFNFRUlOR19NQVg6CiAgICAgICAgICAgICAgICAgICAgcHJpbnQoZiIgICB7YmFuZH06IHN0YXJzIGFyZSB7c2VlaW5nOi4yZn0gcHggd2lkZSAobWVkaWFuIG9mICIKICAgICAgICAgICAgICAgICAgICAgICAgICBmIntpbnQobnAuaXNmaW5pdGUod2lkdGhzKS5zdW0oKSl9IGJyaWdodCB1bnNhdHVyYXRlZCBzdGFycykiLAogICAgICAgICAgICAgICAgICAgICAgICAgIGZsdXNoPVRydWUpCiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIGZsb2F0KHNlZWluZykKICAgICAgICBwcmludChmIiAgIHtiYW5kfTogY291bGQgbm90IG1lYXN1cmUgdGhlIHN0YXIgd2lkdGgsIHVzaW5nIgogICAgICAgICAgICAgIGYiIHtTRUVJTkdfRkFMTEJBQ0t9IHB4IiwgZmx1c2g9VHJ1ZSkKICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOgogICAgICAgIHByaW50KGYiICAge2JhbmR9OiBzdGFyIHdpZHRoIG1lYXN1cmVtZW50IGZhaWxlZCAoe2V4Y30pLCB1c2luZyIKICAgICAgICAgICAgICBmIiB7U0VFSU5HX0ZBTExCQUNLfSBweCIsIGZsdXNoPVRydWUpCiAgICByZXR1cm4gZmxvYXQoU0VFSU5HX0ZBTExCQUNLKQoKCmRlZiBkZXRlY3Rfc291cmNlcyhkYXRhLCByYXcsIGJhbmQsIHNhdF9sZXZlbCwgd2luZG93PU5vbmUpOgogICAgIiIiRmluZCB0aGUgc291cmNlcyBpbiBvbmUgZnJhbWUuIFJldHVybnMgKHgsIHksIGZsdXggZXN0aW1hdGUpIGluIHRoZQogICAgY29vcmRpbmF0ZXMgb2YgdGhlIHdob2xlIGZyYW1lLCBhbmQgdGhlIGZyYW1lJ3MgdHlwaWNhbCBzdGFyIHdpZHRoLgoKICAgIGB3aW5kb3dgIGlzICh5MCwgeTEsIHgwLCB4MSk6IHRoZSBwYXJ0IG9mIHRoZSBmcmFtZSB0aGUgZ2FsYXh5IG9jY3VwaWVzLgogICAgRmluZGluZyBzb3VyY2VzIGlzIHBhaWQgZm9yIGJ5IHRoZSBwaXhlbCwgc28gd29ya2luZyBvdXRzaWRlIHRoZSBnYWxheHkKICAgIGNvc3RzIHRpbWUgYW5kIHJldHVybnMgbm90aGluZy4KICAgICIiIgogICAgeF9vZmZzZXQgPSB5X29mZnNldCA9IDAKICAgIHdvcmssIHdvcmtfcmF3ID0gZGF0YSwgcmF3CiAgICBpZiB3aW5kb3cgaXMgbm90IE5vbmU6CiAgICAgICAgeTAsIHkxLCB4MCwgeDEgPSB3aW5kb3cKICAgICAgICB3b3JrLCB3b3JrX3JhdyA9IGRhdGFbeTA6eTEsIHgwOngxXSwgcmF3W3kwOnkxLCB4MDp4MV0KICAgICAgICB4X29mZnNldCwgeV9vZmZzZXQgPSB4MCwgeTAKICAgICAgICBwcmludChmIiAgIHtiYW5kfTogd29ya2luZyBvbiB7eDEgLSB4MH0geCB7eTEgLSB5MH0gcGl4ZWxzIGFyb3VuZCB0aGUgIgogICAgICAgICAgICAgIGYiZ2FsYXh5IiwgZmx1c2g9VHJ1ZSkKCiAgICBwcmludChmIkZpbmRpbmcgc291cmNlcyBpbiB0aGUge2JhbmR9IGltYWdlLi4uIiwgZmx1c2g9VHJ1ZSkKICAgIGJrZyA9IGxvY2FsX2JhY2tncm91bmQod29yaykKICAgIHJtcyA9IGZsb2F0KG5wLm1lZGlhbihia2cuYmFja2dyb3VuZF9ybXMpKQogICAgcHJpbnQoZiIgICB7YmFuZH06IGxvY2FsIHNreSBydW5zIGZyb20ge2JrZy5iYWNrZ3JvdW5kLm1pbigpOi4xZn0iCiAgICAgICAgICBmIiB0byB7YmtnLmJhY2tncm91bmQubWF4KCk6LjFmfSBjb3VudHMsIG5vaXNlIHtybXM6LjJmfSIsIGZsdXNoPVRydWUpCgogICAgZndobSA9IERBT0ZJTkRfRldITSBpZiBEQU9GSU5EX0ZXSE0gZWxzZSBlc3RpbWF0ZV9zZWVpbmcoCiAgICAgICAgd29yaywgYmtnLmJhY2tncm91bmQsIHJtcywgYmFuZCwgd29ya19yYXcsIHNhdF9sZXZlbCkKICAgIFNFRUlOR19NRUFTVVJFRFtiYW5kXSA9IGZ3aG0KCiAgICBzb3VyY2VzID0gREFPU3RhckZpbmRlcihmd2htPWZ3aG0sIHRocmVzaG9sZD1ERVRFQ1RJT05fVEhSRVNIT0xEX1NJR01BICogcm1zKSgKICAgICAgICB3b3JrIC0gYmtnLmJhY2tncm91bmQpCiAgICBpZiBzb3VyY2VzIGlzIE5vbmUgb3IgbGVuKHNvdXJjZXMpID09IDA6CiAgICAgICAgcHJpbnQoZiIgICB7YmFuZH06IG5vIHNvdXJjZXMgZm91bmQiLCBmbHVzaD1UcnVlKQogICAgICAgIHJldHVybiBucC5lbXB0eSgwKSwgbnAuZW1wdHkoMCksIG5wLmVtcHR5KDApLCBmd2htCiAgICBzb3VyY2VzID0gc291cmNlc1tzb3VyY2VzWydwZWFrJ10gPiBQRUFLX01JTl9TVEQgKiBybXNdCiAgICB4LCB5ID0gX3h5X29mKHNvdXJjZXMpCiAgICBmID0gbnAuYXNhcnJheShzb3VyY2VzWyJmbHV4Il0sIGZsb2F0KQogICAgb3JkZXIgPSBucC5hcmdzb3J0KGYpWzo6LTFdCiAgICB4LCB5LCBmID0geFtvcmRlcl0gKyB4X29mZnNldCwgeVtvcmRlcl0gKyB5X29mZnNldCwgZltvcmRlcl0KCiAgICAjIE9uZSBzb3VyY2UgZm91bmQgdHdvIG9yIHRocmVlIHRpbWVzIG92ZXIgLSBhIHNhdHVyYXRlZCBzdGFyIHdpdGggYSBmbGF0CiAgICAjIHRvcCwgb3IgYSBrbm90IHdpdGggdHdvIHBlYWtzIGluc2lkZSBvbmUgc2VlaW5nIGRpc2MgLSB3b3VsZCBvdGhlcndpc2UgYmUKICAgICMgbWVhc3VyZWQgYXMgc2V2ZXJhbCBzb3VyY2VzIHNoYXJpbmcgb25lIGFwZXJ0dXJlLiBUd28gc291cmNlcyBjbG9zZXIgdGhhbgogICAgIyBvbmUgRldITSBjYW5ub3QgYmUgbWVhc3VyZWQgYXBhcnQsIHNvIG9ubHkgdGhlIGJyaWdodGVzdCBpcyBrZXB0LgogICAga2VlcCA9IG5wLm9uZXMobGVuKHgpLCBkdHlwZT1ib29sKQogICAgdHJlZSA9IGNLRFRyZWUobnAuY29sdW1uX3N0YWNrKFt4LCB5XSkpCiAgICBmb3IgaSBpbiByYW5nZShsZW4oeCkpOgogICAgICAgIGlmIGtlZXBbaV06CiAgICAgICAgICAgIGZvciBqIGluIHRyZWUucXVlcnlfYmFsbF9wb2ludChbeFtpXSwgeVtpXV0sIHI9ZndobSk6CiAgICAgICAgICAgICAgICBpZiBqID4gaToKICAgICAgICAgICAgICAgICAgICBrZWVwW2pdID0gRmFsc2UKICAgIHByaW50KGYiICAge2JhbmR9OiB7bGVuKHgpOix9IGRldGVjdGlvbnMsIHtpbnQoKH5rZWVwKS5zdW0oKSk6LH0gb2YgdGhlbSAiCiAgICAgICAgICBmIndpdGhpbiBvbmUgRldITSBvZiBhIGJyaWdodGVyIG9uZSBhbmQgZHJvcHBlZCIsIGZsdXNoPVRydWUpCiAgICByZXR1cm4geFtrZWVwXSwgeVtrZWVwXSwgZltrZWVwXSwgZndobQoKCmRlZiByZWdpc3Rlcl9CX29udG9fVihCX2ltZywgVl9pbWcsIGZpdHNfZmlsZV9CLCBmaXRzX2ZpbGVfVik6CiAgICAiIiJUaGUgbWFwcGluZyBiZXR3ZWVuIEIncyBwaXhlbCBncmlkIGFuZCBWJ3MsIGJvdGggd2F5cy4KCiAgICBCIGFuZCBWIGFyZSBzZXBhcmF0ZSBleHBvc3VyZXMgYW5kIG5vdGhpbmcgZ3VhcmFudGVlcyB0aGVpciBncmlkcwogICAgY29pbmNpZGUuIGFzdHJvYWxpZ24gZmluZHMgdGhlIHJlZ2lzdHJhdGlvbiBmcm9tIHRoZSBzdGFyIHBhdHRlcm5zIG9mCiAgICB0aGVzZSB0d28gaW1hZ2VzOyBlYWNoIGZyYW1lJ3Mgb3duIHBsYXRlIHNvbHV0aW9uIGlzIHRoZSBmYWxsYmFjay4KCiAgICBSZXR1cm5zIChiX3RvX3YsIHZfdG9fYiwgZGVzY3JpcHRpb24pOyBlYWNoIHRha2VzIGFuIChOLCAyKSBhcnJheS4KICAgICIiIgogICAgdHJ5OgogICAgICAgIHRyYW5zZiwgKHNyY19saXN0LCBfdGd0X2xpc3QpID0gYWEuZmluZF90cmFuc2Zvcm0oCiAgICAgICAgICAgIG5wLm5hbl90b19udW0oQl9pbWcpLCBucC5uYW5fdG9fbnVtKFZfaW1nKSkKICAgICAgICBpZiBhYnModHJhbnNmLnNjYWxlIC0gMS4wKSA+IEFTVFJPQUxJR05fTUFYX1NDQUxFX0VSUk9SOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYiZml0dGVkIHNjYWxlIHt0cmFuc2Yuc2NhbGU6LjNmfSBpcyBub3QgcGxhdXNpYmxlICIKICAgICAgICAgICAgICAgICAgICAgICAgICAgICBmImZvciB0d28gZnJhbWVzIGZyb20gdGhlIHNhbWUgdGVsZXNjb3BlIikKICAgICAgICB0ZXh0ID0gKGYiZnJvbSB7bGVuKHNyY19saXN0KX0gbWF0Y2hlZCBzdGFyIHBhaXJzIChhc3Ryb2FsaWduKTogc2hpZnQgIgogICAgICAgICAgICAgICAgZiIoe3RyYW5zZi50cmFuc2xhdGlvblswXTorLjFmfSwge3RyYW5zZi50cmFuc2xhdGlvblsxXTorLjFmfSkgcHgsICIKICAgICAgICAgICAgICAgIGYicm90YXRpb24ge25wLmRlZ3JlZXModHJhbnNmLnJvdGF0aW9uKTorLjJmfSBkZWcsIHNjYWxlICIKICAgICAgICAgICAgICAgIGYie3RyYW5zZi5zY2FsZTouNGZ9IikKICAgICAgICBwcmludCgiICAgcmVnaXN0ZXJlZCBCIG9udG8gViAiICsgdGV4dCwgZmx1c2g9VHJ1ZSkKICAgICAgICBpbnZlcnNlID0gdHJhbnNmLmludmVyc2UKICAgICAgICByZXR1cm4gKGxhbWJkYSB4eTogdHJhbnNmKG5wLmFzYXJyYXkoeHksIGZsb2F0KSksCiAgICAgICAgICAgICAgICBsYW1iZGEgeHk6IGludmVyc2UobnAuYXNhcnJheSh4eSwgZmxvYXQpKSwgdGV4dCkKICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZToKICAgICAgICBwcmludChmIiAgIGFzdHJvYWxpZ24gcmVnaXN0cmF0aW9uIGZhaWxlZCAoe2V9KSAtICIKICAgICAgICAgICAgICBmImZhbGxpbmcgYmFjayB0byBlYWNoIGZyYW1lJ3Mgb3duIFdDUyIsIGZsdXNoPVRydWUpCiAgICAgICAgd2NzX0IgPSBXQ1MoZml0cy5nZXRoZWFkZXIoZml0c19maWxlX0IpKS5jZWxlc3RpYWwKICAgICAgICB3Y3NfViA9IFdDUyhmaXRzLmdldGhlYWRlcihmaXRzX2ZpbGVfVikpLmNlbGVzdGlhbAoKICAgICAgICBkZWYgYl90b192KHh5KToKICAgICAgICAgICAgeHkgPSBucC5hc2FycmF5KHh5LCBmbG9hdCkKICAgICAgICAgICAgcmEsIGRlYyA9IHdjc19CLmFsbF9waXgyd29ybGQoeHlbOiwgMF0sIHh5WzosIDFdLCAwKQogICAgICAgICAgICByZXR1cm4gbnAuY29sdW1uX3N0YWNrKHdjc19WLmFsbF93b3JsZDJwaXgocmEsIGRlYywgMCkpCgogICAgICAgIGRlZiB2X3RvX2IoeHkpOgogICAgICAgICAgICB4eSA9IG5wLmFzYXJyYXkoeHksIGZsb2F0KQogICAgICAgICAgICByYSwgZGVjID0gd2NzX1YuYWxsX3BpeDJ3b3JsZCh4eVs6LCAwXSwgeHlbOiwgMV0sIDApCiAgICAgICAgICAgIHJldHVybiBucC5jb2x1bW5fc3RhY2sod2NzX0IuYWxsX3dvcmxkMnBpeChyYSwgZGVjLCAwKSkKICAgICAgICByZXR1cm4gYl90b192LCB2X3RvX2IsICJ0aHJvdWdoIHRoZSB0d28gcGxhdGUgc29sdXRpb25zIgoKCmRlZiBhcGVydHVyZV9waG90b21ldHJ5X2F0KGRhdGEsIHJhdywgeHksIHJfYXAsIHJfaW4sIHJfb3V0LCBnYWluLCBybik6CiAgICAiIiJGbHV4LCBpdHMgZXJyb3IsIHRoZSBicmlnaHRlc3QgcmF3IHBpeGVsLCBhbmQgdGhlIGxvY2FsIHNreSwgZm9yIGV2ZXJ5CiAgICBwb3NpdGlvbiBpbiB4eSAtIGFsbCB0aHJvdWdoIG9uZSBhcGVydHVyZSByYWRpdXMuCgogICAgVGhlIHNreSBpcyB0aGUgc2lnbWEtY2xpcHBlZCBtZWRpYW4gb2YgYSByaW5nIGFyb3VuZCB0aGUgc291cmNlLCBzbyB0aGUKICAgIHNtb290aCBsaWdodCBvZiB0aGUgZ2FsYXh5IHVuZGVyIGEga25vdCBpcyB0YWtlbiBvZmYgd2l0aCBpdC4gVGhlIGVycm9yCiAgICBjb3VudHMgdGhlIHBob3RvbnMgb2YgdGhlIHNvdXJjZSwgdGhlIHNreSBub2lzZSBtZWFzdXJlZCBpbiB0aGUgcmluZyAob25jZQogICAgcGVyIGFwZXJ0dXJlIHBpeGVsLCBhbmQgb25jZSBtb3JlIHRocm91Z2ggdGhlIHVuY2VydGFpbnR5IG9mIHRoZSBza3kgbGV2ZWwKICAgIGl0c2VsZiksIHdpdGggdGhlIGhlYWRlcidzIHJlYWQgbm9pc2UgYXMgYSBmbG9vci4KICAgICIiIgogICAgeHkgPSBucC5hc2FycmF5KHh5LCBmbG9hdCkKICAgIGFwID0gQ2lyY3VsYXJBcGVydHVyZSh4eSwgcj1yX2FwKQogICAgYW5uID0gQ2lyY3VsYXJBbm51bHVzKHh5LCByX2luPXJfaW4sIHJfb3V0PXJfb3V0KQogICAgc2t5ID0gQXBlcnR1cmVTdGF0cyhkYXRhLCBhbm4sIHNpZ21hX2NsaXA9U2lnbWFDbGlwKHNpZ21hPTMuMCwgbWF4aXRlcnM9NSkpCiAgICBza3lfbWVkID0gbnAuYXRsZWFzdF8xZChucC5hc2FycmF5KHNreS5tZWRpYW4sIGZsb2F0KSkKICAgIHNreV9zdGQgPSBucC5hdGxlYXN0XzFkKG5wLmFzYXJyYXkoc2t5LnN0ZCwgZmxvYXQpKQogICAgbl9za3kgPSBucC5hdGxlYXN0XzFkKG5wLmFzYXJyYXkoc2t5LnN1bV9hcGVyX2FyZWEudmFsdWUsIGZsb2F0KSkKICAgIHRvdGFsID0gbnAuYXNhcnJheShhcGVydHVyZV9waG90b21ldHJ5KGRhdGEsIGFwKVsiYXBlcnR1cmVfc3VtIl0sIGZsb2F0KQogICAgbl9hcCA9IGZsb2F0KGFwLmFyZWEpCiAgICBmbHV4ID0gdG90YWwgLSBza3lfbWVkICogbl9hcAogICAgcGl4X3ZhciA9IG5wLm1heGltdW0oc2t5X3N0ZCAqKiAyLCAocm4gLyBnYWluKSAqKiAyKQogICAgdmFyID0gKG5wLmNsaXAoZmx1eCwgMCwgTm9uZSkgLyBnYWluCiAgICAgICAgICAgKyBuX2FwICogcGl4X3ZhciAqICgxLjAgKyBuX2FwIC8gbnAubWF4aW11bShuX3NreSwgMSkpKQogICAgcGVhayA9IG5wLmF0bGVhc3RfMWQobnAuYXNhcnJheSgKICAgICAgICBBcGVydHVyZVN0YXRzKHJhdywgQ2lyY3VsYXJBcGVydHVyZSh4eSwgcj1tYXgoMS41LCAwLjUgKiByX2FwKSkpLm1heCwgZmxvYXQpKQogICAgcmV0dXJuIGZsdXgsIG5wLnNxcnQodmFyKSwgcGVhaywgc2t5X21lZAoKCmRlZiBpbnN0cnVtZW50YWwoZmx1eCwgZXJyKToKICAgIHdpdGggbnAuZXJyc3RhdGUoZGl2aWRlPSJpZ25vcmUiLCBpbnZhbGlkPSJpZ25vcmUiKToKICAgICAgICBtYWcgPSBucC53aGVyZShmbHV4ID4gMCwgLTIuNSAqIG5wLmxvZzEwKGZsdXgpLCBucC5uYW4pCiAgICAgICAgbWFnX2VyciA9IG5wLndoZXJlKGZsdXggPiAwLCAxLjA4NTcgKiBlcnIgLyBmbHV4LCBucC5uYW4pCiAgICByZXR1cm4gbWFnLCBtYWdfZXJyCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tIFRoZSBwbGF0ZSBzb2x1dGlvbiwgY2hlY2tlZCBhZ2FpbnN0IEdhaWEgLS0tLS0tLS0tLS0tLS0tLS0tLQoKZGVmIGZpZWxkX2NlbnRyZV9hbmRfcmFkaXVzKHdjcywgc2hhcGUpOgogICAgbnksIG54ID0gc2hhcGUKICAgIGNlbnRyZSA9IHdjcy5waXhlbF90b193b3JsZChueCAvIDIsIG55IC8gMikKICAgIGNvcm5lcnMgPSB3Y3MucGl4ZWxfdG9fd29ybGQobnAuYXJyYXkoWzAsIG54LCAwLCBueF0pLCBucC5hcnJheShbMCwgMCwgbnksIG55XSkpCiAgICByZXR1cm4gY2VudHJlLCBmbG9hdChucC5tYXgoY2VudHJlLnNlcGFyYXRpb24oY29ybmVycykuZGVnKSkKCgpkZWYgdmVyaWZ5X3BsYXRlX3NvbHV0aW9uKHh5X2JyaWdodCwgYWxsX3h5LCB3Y3MsIHNoYXBlKToKICAgICIiIlRoZSBwbGF0ZSBzb2x1dGlvbiBvZiB0aGUgViBmcmFtZSwgY2hlY2tlZCBhZ2FpbnN0IEdhaWEgYW5kIHJlcGFpcmVkIGlmCiAgICBpdCBkb2VzIG5vdCBmaXQuCgogICAgRXZlcnl0aGluZyB0aGF0IHVzZXMgdGhlIHNreSBnb2VzIHRocm91Z2ggaXQ6IHRoZSBjYWxpYnJhdGlvbiBzdGFycywgdGhlCiAgICBmb3JlZ3JvdW5kIHN0YXJzLCB0aGUgZ2FsYXh5J3MgY2VudHJlLiBTb2Z0d2FyZSB0aGF0IGNvbWJpbmVzIG9yIGNyb3BzCiAgICBmcmFtZXMgc29tZXRpbWVzIGtlZXBzIGEgc29sdXRpb24gdGhhdCBiZWxvbmdzIHRvIGFub3RoZXIgZnJhbWUgLSBpbiB0aGUKICAgIENsdXN0ZXJfQ01EIHRlc3RzIGEgc3BsaXQtY2hhbm5lbCBmaWxlIGNhcnJpZWQgb25lIDQ1IHB4IG9mZiwgYW5kIG9ubHkgNCBvZgogICAgMiwwMDAgY2F0YWxvZ3VlIHN0YXJzIG1hdGNoZWQuCgogICAgVGhlIDEwMCBicmlnaHRlc3QgdW5zYXR1cmF0ZWQgc291cmNlcyBhcmUgcHV0IHRocm91Z2ggdGhlIHNvbHV0aW9uIGFuZAogICAgbG9va2VkIHVwIGluIEdhaWEgRFIzLiBJZiBhdCBsZWFzdCBoYWxmIGhhdmUgYSBHYWlhIHN0YXIgd2l0aGluCiAgICBDQUxJQl9NQVRDSF9SQURJVVMgdGhlIHNvbHV0aW9uIGlzIGtlcHQuIE90aGVyd2lzZSB0aGUgb2Zmc2V0IGJldHdlZW4gdGhlCiAgICBHYWlhIHN0YXJzIGFuZCB0aGUgc291cmNlcyBpcyBmb3VuZCAob3IgdGhlIHN0YXIgcGF0dGVybnMgbWF0Y2hlZCwgd2l0aAogICAgdGhlIG1pcnJvciBpbWFnZSB0cmllZCB0b28pIGFuZCBhIG5ldyBzb2x1dGlvbiBmaXR0ZWQuCiAgICAiIiIKICAgIGZyb20gYXN0cm9weS53Y3MudXRpbHMgaW1wb3J0IGZpdF93Y3NfZnJvbV9wb2ludHMKICAgIG55LCBueCA9IHNoYXBlCiAgICBjZW50cmUsIHJhZGl1cyA9IGZpZWxkX2NlbnRyZV9hbmRfcmFkaXVzKHdjcywgc2hhcGUpCiAgICB0cnk6CiAgICAgICAgcmVzID0gVml6aWVyKGNvbHVtbnM9WyJSQV9JQ1JTIiwgIkRFX0lDUlMiLCAiR21hZyJdLAogICAgICAgICAgICAgICAgICAgICBjb2x1bW5fZmlsdGVycz17IkdtYWciOiAiPDE3In0sIHJvd19saW1pdD0tMSkucXVlcnlfcmVnaW9uKAogICAgICAgICAgICBTa3lDb29yZChjZW50cmUucmEsIGNlbnRyZS5kZWMpLCByYWRpdXM9MS4zICogcmFkaXVzICogdS5kZWcsCiAgICAgICAgICAgIGNhdGFsb2c9IkkvMzU1L2dhaWFkcjMiKQogICAgICAgIGdhaWEgPSByZXNbMF0KICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOgogICAgICAgIHByaW50KGYiW3djc10gR2FpYSB1bmF2YWlsYWJsZSAoe2V4Y30pIC0gdGhlIHBsYXRlIHNvbHV0aW9uIGlzIG5vdCBjaGVja2VkIikKICAgICAgICByZXR1cm4gd2NzLCAibm90IGNoZWNrZWQgKEdhaWEgdW5hdmFpbGFibGUpIgogICAgZ2FpYS5zb3J0KCJHbWFnIikKICAgIGdza3kgPSBTa3lDb29yZChucC5hc2FycmF5KGdhaWFbIlJBX0lDUlMiXSwgZmxvYXQpICogdS5kZWcsCiAgICAgICAgICAgICAgICAgICAgbnAuYXNhcnJheShnYWlhWyJERV9JQ1JTIl0sIGZsb2F0KSAqIHUuZGVnKQoKICAgIGRlZiBmcmFjdGlvbih3LCBuPTEwMCk6CiAgICAgICAgcmEsIGRlYyA9IHcuYWxsX3BpeDJ3b3JsZCh4eV9icmlnaHRbOm4sIDBdLCB4eV9icmlnaHRbOm4sIDFdLCAwKQogICAgICAgIF9pLCBzZXAsIF8gPSBTa3lDb29yZChyYSAqIHUuZGVnLCBkZWMgKiB1LmRlZykubWF0Y2hfdG9fY2F0YWxvZ19za3koZ3NreSkKICAgICAgICBvayA9IHNlcC5hcmNzZWMgPCBDQUxJQl9NQVRDSF9SQURJVVMKICAgICAgICByZXR1cm4gb2subWVhbigpLCAobnAubWVkaWFuKHNlcC5hcmNzZWNbb2tdKSBpZiBvay5hbnkoKSBlbHNlIG5wLm5hbikKCiAgICBmMCwgczAgPSBmcmFjdGlvbih3Y3MpCiAgICBwcmludChmIlt3Y3NdIHsxMDAgKiBmMDouMGZ9JSBvZiB0aGUgMTAwIGJyaWdodGVzdCBzb3VyY2VzIGhhdmUgYSBHYWlhIHN0YXIgIgogICAgICAgICAgZiJ3aXRoaW4ge0NBTElCX01BVENIX1JBRElVUzpnfSBhcmNzZWMgKG1lZGlhbiB7czA6LjJmfSBhcmNzZWMpIiwgZmx1c2g9VHJ1ZSkKICAgIGlmIGYwID49IFdDU19DSEVDS19NSU5fRlJBQ1RJT046CiAgICAgICAgcmV0dXJuIHdjcywgZiJjaGVja2VkIGFnYWluc3QgR2FpYTogezEwMCAqIGYwOi4wZn0lIG9mIGJyaWdodCBzb3VyY2VzIG1hdGNoIgoKICAgIHByaW50KCJbd2NzXSB0aGUgcGxhdGUgc29sdXRpb24gaW4gdGhlIGhlYWRlciBkb2VzIG5vdCBmaXQgdGhpcyBpbWFnZSAtICIKICAgICAgICAgICJyZXBhaXJpbmcgaXQgZnJvbSB0aGUgc3RhciBwYXR0ZXJuIiwgZmx1c2g9VHJ1ZSkKICAgIGd4LCBneSA9IHdjcy5hbGxfd29ybGQycGl4KGdza3kucmEuZGVnLCBnc2t5LmRlYy5kZWcsIDApCiAgICBuZWFyID0gKGd4ID4gLTAuMDUgKiBueCkgJiAoZ3ggPCAxLjA1ICogbngpICYgKGd5ID4gLTAuMDUgKiBueSkgJiAoZ3kgPCAxLjA1ICogbnkpCiAgICBzcmNfYWxsID0gbnAuY29sdW1uX3N0YWNrKFtneCwgZ3ldKQogICAgZGV0X3RyZWUgPSBjS0RUcmVlKHh5X2JyaWdodFs6NjAwXSkKICAgIG1heF9vZmYgPSAwLjI1ICogbWF4KG54LCBueSkKICAgIGJlc3QgPSBOb25lCiAgICBmb3IgcGFyaXR5IGluICgxLCAtMSk6CiAgICAgICAgc3JjID0gc3JjX2FsbC5jb3B5KCkKICAgICAgICBpZiBwYXJpdHkgPCAwOgogICAgICAgICAgICBzcmNbOiwgMF0gPSAobnggLSAxKSAtIHNyY1s6LCAwXQogICAgICAgIGNhbmQgPSBzcmNbbmVhcl1bOjYwMF0KICAgICAgICBkeHkgPSBbXQogICAgICAgIGZvciBrLCBuYiBpbiBlbnVtZXJhdGUoZGV0X3RyZWUucXVlcnlfYmFsbF9wb2ludChjYW5kLCByPW1heF9vZmYpKToKICAgICAgICAgICAgaWYgbmI6CiAgICAgICAgICAgICAgICBkeHkuYXBwZW5kKHh5X2JyaWdodFs6NjAwXVtuYl0gLSBjYW5kW2tdKQogICAgICAgIGlmIGR4eToKICAgICAgICAgICAgZHh5ID0gbnAudnN0YWNrKGR4eSkKICAgICAgICAgICAgYmlucyA9IG5wLmFyYW5nZSgtbWF4X29mZiwgbWF4X29mZiArIDQsIDQuMCkKICAgICAgICAgICAgSCwgZXgsIGV5ID0gbnAuaGlzdG9ncmFtMmQoZHh5WzosIDBdLCBkeHlbOiwgMV0sIGJpbnM9W2JpbnMsIGJpbnNdKQogICAgICAgICAgICBpLCBqID0gbnAudW5yYXZlbF9pbmRleChucC5hcmdtYXgoSCksIEguc2hhcGUpCiAgICAgICAgICAgIHBlYWsgPSBIW2ksIGpdCiAgICAgICAgICAgIGlmIHBlYWsgPj0gMTAgYW5kIHBlYWsgPiA4ICogbWF4KG5wLm1lZGlhbihIW0ggPiAwXSksIDEpOgogICAgICAgICAgICAgICAgc2hpZnQgPSBucC5hcnJheShbMC41ICogKGV4W2ldICsgZXhbaSArIDFdKSwgMC41ICogKGV5W2pdICsgZXlbaiArIDFdKV0pCiAgICAgICAgICAgICAgICBUID0gbGFtYmRhIHAsIHNfPXNoaWZ0OiBwICsgc18KICAgICAgICAgICAgICAgIGlmIGJlc3QgaXMgTm9uZSBvciBwZWFrID4gYmVzdFsyXToKICAgICAgICAgICAgICAgICAgICBiZXN0ID0gKHBhcml0eSwgVCwgaW50KHBlYWspKQogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICBmb3Igbl9jYXQsIG5fZGV0IGluICgoMTUwLCAxNTApLCAoMjAwLCAyMDApLCAoMzAwLCAzMDApKToKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgVCwgKHMsIF90KSA9IGFhLmZpbmRfdHJhbnNmb3JtKGNhbmRbOm5fY2F0XSwgeHlfYnJpZ2h0WzpuX2RldF0pCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICBpZiBhYnMoVC5zY2FsZSAtIDEuMCkgPiAwLjE6CiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICBpZiBiZXN0IGlzIE5vbmUgb3IgbGVuKHMpID4gYmVzdFsyXToKICAgICAgICAgICAgICAgIGJlc3QgPSAocGFyaXR5LCBULCBsZW4ocykpCiAgICAgICAgICAgIGJyZWFrCiAgICBpZiBiZXN0IGlzIE5vbmU6CiAgICAgICAgc3RvcCgiUExBVEUgU09MVVRJT046IHRoZSBoZWFkZXIncyBzb2x1dGlvbiBkb2VzIG5vdCBmaXQgdGhlIFYgaW1hZ2UgYW5kICIKICAgICAgICAgICAgICJjb3VsZCBub3QgYmUgcmVwYWlyZWQuIFBsYXRlLXNvbHZlIHRoZSBWIGZyYW1lIGFnYWluIChmb3IgZXhhbXBsZSAiCiAgICAgICAgICAgICAid2l0aCBhc3Ryb21ldHJ5Lm5ldCkgYW5kIHJlcnVuLiIpCiAgICBwYXJpdHksIFQsIG5jdGwgPSBiZXN0CiAgICBzcmMgPSBzcmNfYWxsLmNvcHkoKQogICAgaWYgcGFyaXR5IDwgMDoKICAgICAgICBzcmNbOiwgMF0gPSAobnggLSAxKSAtIHNyY1s6LCAwXQogICAgZ3Vlc3MgPSBUKHNyYykKICAgIHRyZWUgPSBjS0RUcmVlKGFsbF94eSkKICAgIG5ldyA9IHdjcwogICAgdG9sID0gNS4wCiAgICBvayA9IG5wLnplcm9zKGxlbihndWVzcyksIGJvb2wpCiAgICBmb3IgXyBpbiByYW5nZSgzKToKICAgICAgICBkLCBqID0gdHJlZS5xdWVyeShndWVzcywgZGlzdGFuY2VfdXBwZXJfYm91bmQ9dG9sKQogICAgICAgIG9rID0gbnAuaXNmaW5pdGUoZCkKICAgICAgICBpZiBvay5zdW0oKSA8IDEwOgogICAgICAgICAgICBicmVhawogICAgICAgIHB4ID0gYWxsX3h5W2pbb2tdXQogICAgICAgIG5ldyA9IGZpdF93Y3NfZnJvbV9wb2ludHMoKHB4WzosIDBdLCBweFs6LCAxXSksIGdza3lbb2tdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcHJval9wb2ludD0iY2VudGVyIiwgcHJvamVjdGlvbj0iVEFOIikKICAgICAgICBneDIsIGd5MiA9IG5ldy5hbGxfd29ybGQycGl4KGdza3kucmEuZGVnLCBnc2t5LmRlYy5kZWcsIDApCiAgICAgICAgZ3Vlc3MgPSBucC5jb2x1bW5fc3RhY2soW2d4MiwgZ3kyXSkKICAgICAgICB0b2wgPSBYWV9NQVRDSF9UT0xFUkFOQ0UKICAgIGYxLCBzMSA9IGZyYWN0aW9uKG5ldykKICAgIHNoaWZ0ID0gbnAuaHlwb3QoKihucC5hcnJheShuZXcuYWxsX3dvcmxkMnBpeChjZW50cmUucmEuZGVnLCBjZW50cmUuZGVjLmRlZywgMCkpCiAgICAgICAgICAgICAgICAgICAgICAgLSBucC5hcnJheShbbnggLyAyLCBueSAvIDJdKSkpCiAgICBwcmludChmIlt3Y3NdIHJlcGFpcmVkICh7J21pcnJvcmVkLCAnIGlmIHBhcml0eSA8IDAgZWxzZSAnJ317bmN0bH0gcGF0dGVybiAiCiAgICAgICAgICBmInN0YXJzLCB7aW50KG9rLnN1bSgpKX0gaW4gdGhlIGZpdCk6IHRoZSBmaWVsZCBjZW50cmUgbW92ZWQge3NoaWZ0Oi4wZn0gcHg7ICIKICAgICAgICAgIGYibm93IHsxMDAgKiBmMTouMGZ9JSBvZiBicmlnaHQgc291cmNlcyBtYXRjaCBHYWlhIChtZWRpYW4ge3MxOi4yZn0gYXJjc2VjKSIsCiAgICAgICAgICBmbHVzaD1UcnVlKQogICAgaWYgZjEgPCBXQ1NfQ0hFQ0tfTUlOX0ZSQUNUSU9OOgogICAgICAgIHN0b3AoIlBMQVRFIFNPTFVUSU9OOiB0aGUgcmVwYWlyZWQgc29sdXRpb24gc3RpbGwgZG9lcyBub3QgZml0LiAiCiAgICAgICAgICAgICAiUGxhdGUtc29sdmUgdGhlIFYgZnJhbWUgYWdhaW4gYW5kIHJlcnVuLiIpCiAgICByZXR1cm4gbmV3LCAoZiJSRVBBSVJFRCBmcm9tIEdhaWE6IGhlYWRlciBzb2x1dGlvbiBtYXRjaGVkIHsxMDAgKiBmMDouMGZ9JSBvZiAiCiAgICAgICAgICAgICAgICAgZiJicmlnaHQgc291cmNlcywgcmVwYWlyZWQgezEwMCAqIGYxOi4wZn0lLCBjZW50cmUgbW92ZWQge3NoaWZ0Oi4wZn0gcHgiKQoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLSBDYWxpYnJhdGlvbiAtLS0tLS0tLS0tLS0tLS0tLS0tCgpkZWYgcXVlcnlfY2FsaWJyYXRpb25fY2F0YWxvZ3VlKHdoaWNoLCBjZW50cmUsIHJhZGl1c19kZWcpOgogICAgIiIiUmVmZXJlbmNlIHN0YXJzIHdpdGggSm9obnNvbiBCIGFuZCBWOiBSQSwgREVDLCBNYWdfQiwgTWFnX1YsIGVfQiwgZV9WLgoKICAgIEFQQVNTIERSOSAoSUkvMzM2L2FwYXNzOSkgaXMgdGhlIGRlZmF1bHQuIEdhaWEgRFIzIHN5bnRoZXRpYyBwaG90b21ldHJ5CiAgICAoSS8zNjAvc3ludHBob3QpIGlzIGNvbXB1dGVkIGZyb20gR2FpYSdzIG93biBsb3ctcmVzb2x1dGlvbiBzcGVjdHJhIGFuZAogICAgc3RhbmRhcmRpc2VkIHRvIHRoZSBKb2huc29uLUtyb24tQ291c2lucyBzeXN0ZW06IGRlZXBlciwgdW5pZm9ybSBhY3Jvc3MKICAgIHRoZSBza3ksIGFuZCByZXNvbHZlZCBkb3duIHRvIGFib3V0IGFuIGFyY3NlY29uZC4gT25seSBlbnRyaWVzIEdhaWEgZmxhZ3MKICAgIGFzIGdvb2QgYXJlIHVzZWQuCgogICAgcm93X2xpbWl0IGhhcyB0byBiZSBwYXNzZWQgdG8gdGhlIGNvbnN0cnVjdG9yOiBzZXQgb24gdGhlIGNsYXNzIGFmdGVyd2FyZHMKICAgIGl0IGxlZnQgYXN0cm9xdWVyeSdzIGRlZmF1bHQgb2YgNTAgcm93cywgc2lsZW50bHkuCiAgICAiIiIKICAgIGNvb3JkID0gU2t5Q29vcmQoY2VudHJlLnJhLCBjZW50cmUuZGVjKQogICAgZW1wdHkgPSBwZC5EYXRhRnJhbWUoY29sdW1ucz1bIlJBIiwgIkRFQyIsICJNYWdfQiIsICJNYWdfViIsICJlX0IiLCAiZV9WIl0pCiAgICBpZiB3aGljaCA9PSAiQVBBU1MiOgogICAgICAgIGNvbHMgPSBbIlJBSjIwMDAiLCAiREVKMjAwMCIsICJCbWFnIiwgIlZtYWciLCAiZV9CbWFnIiwgImVfVm1hZyJdCiAgICAgICAgcmVzID0gVml6aWVyKGNvbHVtbnM9Y29scywgcm93X2xpbWl0PS0xKS5xdWVyeV9yZWdpb24oCiAgICAgICAgICAgIGNvb3JkLCByYWRpdXM9cmFkaXVzX2RlZyAqIHUuZGVnLCBjYXRhbG9nPSJJSS8zMzYvYXBhc3M5IikKICAgICAgICBpZiBsZW4ocmVzKSA9PSAwOgogICAgICAgICAgICByZXR1cm4gZW1wdHkKICAgICAgICB0ID0gcmVzWzBdLnRvX3BhbmRhcygpCiAgICAgICAgb3V0ID0gcGQuRGF0YUZyYW1lKHsiUkEiOiB0WyJSQUoyMDAwIl0sICJERUMiOiB0WyJERUoyMDAwIl0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAiTWFnX0IiOiB0WyJCbWFnIl0sICJNYWdfViI6IHRbIlZtYWciXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICJlX0IiOiB0WyJlX0JtYWciXSwgImVfViI6IHRbImVfVm1hZyJdfSkKICAgIGVsc2U6CiAgICAgICAgY29scyA9IFsiU291cmNlIiwgIlJBX0lDUlMiLCAiREVfSUNSUyIsICJCbWFnIiwgIlZtYWciLCAiRkIiLCAiZV9GQiIsCiAgICAgICAgICAgICAgICAiRlYiLCAiZV9GViIsICJCRmxhZyIsICJWRmxhZyJdCiAgICAgICAgcmVzID0gVml6aWVyKGNvbHVtbnM9Y29scywgcm93X2xpbWl0PS0xKS5xdWVyeV9yZWdpb24oCiAgICAgICAgICAgIGNvb3JkLCByYWRpdXM9cmFkaXVzX2RlZyAqIHUuZGVnLCBjYXRhbG9nPSJJLzM2MC9zeW50cGhvdCIpCiAgICAgICAgaWYgbGVuKHJlcykgPT0gMDoKICAgICAgICAgICAgcmV0dXJuIGVtcHR5CiAgICAgICAgdCA9IHJlc1swXS50b19wYW5kYXMoKQogICAgICAgIHQgPSB0Wyh0WyJCRmxhZyJdID09IDEpICYgKHRbIlZGbGFnIl0gPT0gMSldCiAgICAgICAgb3V0ID0gcGQuRGF0YUZyYW1lKHsiUkEiOiB0WyJSQV9JQ1JTIl0sICJERUMiOiB0WyJERV9JQ1JTIl0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAiTWFnX0IiOiB0WyJCbWFnIl0sICJNYWdfViI6IHRbIlZtYWciXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICJlX0IiOiAxLjA4NTcgKiB0WyJlX0ZCIl0gLyB0WyJGQiJdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgImVfViI6IDEuMDg1NyAqIHRbImVfRlYiXSAvIHRbIkZWIl19KQogICAgcmV0dXJuIG91dC5yZXNldF9pbmRleChkcm9wPVRydWUpCgoKZGVmIG1hdGNoX2NhdGFsb2d1ZShkZiwgd2NzLCBjYXQsIHNoYXBlKToKICAgICIiIkVhY2ggY2F0YWxvZ3VlIHN0YXIgbWF0Y2hlZCB0byB0aGUgY2xvc2VzdCBzb3VyY2UsIG9uY2UuCgogICAgQSBtYXRjaCBtdXN0IGJlIHdpdGhpbiBYWV9NQVRDSF9UT0xFUkFOQ0Ugb24gdGhlIFYgZ3JpZCBhbmQKICAgIENBTElCX01BVENIX1JBRElVUyBvbiB0aGUgc2t5LiBFYWNoIHNvdXJjZSBpcyB1c2VkIGZvciBhdCBtb3N0IG9uZQogICAgY2F0YWxvZ3VlIHN0YXIgKHRoZSBjbG9zZXN0KSAtIGEgYnJpZ2h0IHN0YXIgZm91bmQgc2V2ZXJhbCB0aW1lcyBvdmVyCiAgICB1c2VkIHRvIGJlIGNvdW50ZWQgb25jZSBwZXIgZGV0ZWN0aW9uLCBhbmQgb25lIHN0YXIgdGhlbiBvdXR3ZWlnaGVkIHRoZQogICAgcmVzdCBvZiB0aGUgY2FsaWJyYXRpb24uCiAgICAiIiIKICAgIGNhdCA9IGNhdFtucC5pc2Zpbml0ZShjYXRbIk1hZ19CIl0pICYgbnAuaXNmaW5pdGUoY2F0WyJNYWdfViJdKV0ucmVzZXRfaW5kZXgoZHJvcD1UcnVlKQogICAgaWYgbm90IGxlbihjYXQpIG9yIG5vdCBsZW4oZGYpOgogICAgICAgIHJldHVybiBwZC5EYXRhRnJhbWUoKQogICAgY3gsIGN5ID0gd2NzLmFsbF93b3JsZDJwaXgoY2F0WyJSQSJdLnZhbHVlcywgY2F0WyJERUMiXS52YWx1ZXMsIDApCiAgICBueSwgbnggPSBzaGFwZQogICAgaW5zaWRlID0gKGN4ID4gMCkgJiAoY3ggPCBueCkgJiAoY3kgPiAwKSAmIChjeSA8IG55KQogICAgY2F0ID0gY2F0W2luc2lkZV0ucmVzZXRfaW5kZXgoZHJvcD1UcnVlKQogICAgY3gsIGN5ID0gY3hbaW5zaWRlXSwgY3lbaW5zaWRlXQogICAgaWYgbm90IGxlbihjYXQpOgogICAgICAgIHJldHVybiBwZC5EYXRhRnJhbWUoKQogICAgZGlzdCwgaWR4ID0gY0tEVHJlZShkZltbIlhfViIsICJZX1YiXV0udmFsdWVzKS5xdWVyeSgKICAgICAgICBucC5jb2x1bW5fc3RhY2soW2N4LCBjeV0pLCBkaXN0YW5jZV91cHBlcl9ib3VuZD1YWV9NQVRDSF9UT0xFUkFOQ0UpCiAgICBiZXN0ID0ge30KICAgIGZvciBpIGluIHJhbmdlKGxlbihjYXQpKToKICAgICAgICBpZiBub3QgbnAuaXNmaW5pdGUoZGlzdFtpXSk6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgaiA9IGludChpZHhbaV0pCiAgICAgICAgaWYgaiBub3QgaW4gYmVzdCBvciBkaXN0W2ldIDwgYmVzdFtqXVsxXToKICAgICAgICAgICAgYmVzdFtqXSA9IChpLCBkaXN0W2ldKQogICAgcm93cyA9IFtdCiAgICBmb3IgaiwgKGksIGQpIGluIGJlc3QuaXRlbXMoKToKICAgICAgICBzcmMgPSBkZi5pbG9jW2pdCiAgICAgICAgc2VwID0gU2t5Q29vcmQoc3JjWyJSQSJdICogdS5kZWcsIHNyY1siREVDIl0gKiB1LmRlZykuc2VwYXJhdGlvbigKICAgICAgICAgICAgU2t5Q29vcmQoY2F0LmxvY1tpLCAiUkEiXSAqIHUuZGVnLCBjYXQubG9jW2ksICJERUMiXSAqIHUuZGVnKSkuYXJjc2VjCiAgICAgICAgaWYgc2VwID4gQ0FMSUJfTUFUQ0hfUkFESVVTOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHIgPSBzcmMudG9fZGljdCgpCiAgICAgICAgci51cGRhdGUoeyJyb3ciOiBqLCAiQ2F0X1JBIjogY2F0LmxvY1tpLCAiUkEiXSwgIkNhdF9ERUMiOiBjYXQubG9jW2ksICJERUMiXSwKICAgICAgICAgICAgICAgICAgIkNhdF9CIjogY2F0LmxvY1tpLCAiTWFnX0IiXSwgIkNhdF9WIjogY2F0LmxvY1tpLCAiTWFnX1YiXSwKICAgICAgICAgICAgICAgICAgIkNhdF9lX0IiOiBjYXQubG9jW2ksICJlX0IiXSwgIkNhdF9lX1YiOiBjYXQubG9jW2ksICJlX1YiXSwKICAgICAgICAgICAgICAgICAgIlNlcF9hcmNzZWMiOiBzZXAsICJTZXBfcHgiOiBkfSkKICAgICAgICByb3dzLmFwcGVuZChyKQogICAgcmV0dXJuIHBkLkRhdGFGcmFtZShyb3dzKQoKCmRlZiBjbGlwcGVkX2xpbmUoeCwgeSwgbnNpZz1DQUxJQl9DTElQX1NJR01BLCBtYXhpdGVyPTEwKToKICAgICIiIlN0cmFpZ2h0LWxpbmUgZml0IHkgPSBpbnRlcmNlcHQgKyBzbG9wZSB4LCB3aXRoIHBvaW50cyBmdXJ0aGVyIHRoYW4KICAgIG5zaWcgcm9idXN0IHNpZ21hcyBmcm9tIHRoZSBsaW5lIGxlZnQgb3V0LCByZXBlYXRlZCB1bnRpbCBub3RoaW5nIGNoYW5nZXMuCiAgICBSZXR1cm5zIChpbnRlcmNlcHQsIHNsb3BlLCBtYXNrX3VzZWQsIHJtc19vZl91c2VkKS4iIiIKICAgIHggPSBucC5hc2FycmF5KHgsIGZsb2F0KQogICAgeSA9IG5wLmFzYXJyYXkoeSwgZmxvYXQpCiAgICB1c2UgPSBucC5pc2Zpbml0ZSh4KSAmIG5wLmlzZmluaXRlKHkpCiAgICBmb3IgXyBpbiByYW5nZShtYXhpdGVyKToKICAgICAgICBpZiB1c2Uuc3VtKCkgPCAzOgogICAgICAgICAgICBicmVhawogICAgICAgIHNsb3BlLCBpbnRlcmNlcHQgPSBucC5wb2x5Zml0KHhbdXNlXSwgeVt1c2VdLCAxKQogICAgICAgIHJlc2lkID0geSAtIChpbnRlcmNlcHQgKyBzbG9wZSAqIHgpCiAgICAgICAgc2lnID0gbWF4KDEuNDgyNiAqIG5wLm1lZGlhbihucC5hYnMocmVzaWRbdXNlXSAtIG5wLm1lZGlhbihyZXNpZFt1c2VdKSkpLCAxZS0zKQogICAgICAgIG5ldyA9IG5wLmlzZmluaXRlKHJlc2lkKSAmIChucC5hYnMocmVzaWQpIDwgbnNpZyAqIHNpZykKICAgICAgICBpZiBucC5hcnJheV9lcXVhbChuZXcsIHVzZSk6CiAgICAgICAgICAgIGJyZWFrCiAgICAgICAgdXNlID0gbmV3CiAgICBzbG9wZSwgaW50ZXJjZXB0ID0gbnAucG9seWZpdCh4W3VzZV0sIHlbdXNlXSwgMSkKICAgIHJlc2lkID0geSAtIChpbnRlcmNlcHQgKyBzbG9wZSAqIHgpCiAgICByZXR1cm4gaW50ZXJjZXB0LCBzbG9wZSwgdXNlLCBmbG9hdChucC5zdGQocmVzaWRbdXNlXSkpCgoKZGVmIGNhbGlicmF0aW9uX2NhbmRpZGF0ZXMoZGYsIHdjcywgc2hhcGUsIHdoaWNoLCBjZW50cmVfcHg9Tm9uZSwgcjI1X3B4PU5vbmUpOgogICAgIiIiQ2F0YWxvZ3VlIHN0YXJzIG1hdGNoZWQgdG8gc291cmNlcyB0aGF0IGNhbiBzZXQgYSBjYWxpYnJhdGlvbjoKICAgIHVuc2F0dXJhdGVkIGluIGJvdGggYmFuZHMsIHdpdGggbm8gb3RoZXIgc291cmNlIGluc2lkZSB0aGUgYXBlcnR1cmUsIHdlbGwKICAgIG1lYXN1cmVkLCB3aXRoIGZpbml0ZSBhbmQgd2VsbC1tZWFzdXJlZCBjYXRhbG9ndWUgbWFnbml0dWRlcy4gV2hlcmUgZW5vdWdoCiAgICBvZiB0aGVtIGxpZSBiZXlvbmQgUjI1IG9ubHkgdGhvc2UgYXJlIHVzZWQ6IHRoZSBza3kgcmluZyBvZiBhIHN0YXIgb24gdGhlCiAgICBkaXNjIGlzIG1lYXN1cmluZyB0aGUgZGlzYy4iIiIKICAgIGNlbnRyZSwgcmFkaXVzID0gZmllbGRfY2VudHJlX2FuZF9yYWRpdXMod2NzLCBzaGFwZSkKICAgIG5hbWUgPSAiQVBBU1MgRFI5IiBpZiB3aGljaCA9PSAiQVBBU1MiIGVsc2UgIkdhaWEgRFIzIHN5bnRoZXRpYyBwaG90b21ldHJ5IgogICAgcHJpbnQoZiJbY2FsaWJdIHF1ZXJ5aW5nIHtuYW1lfSB3aXRoaW4ge3JhZGl1cyAqIDYwOi4wZn0gYXJjbWluIG9mIHRoZSBmaWVsZCAiCiAgICAgICAgICBmImNlbnRyZS4uLiIsIGZsdXNoPVRydWUpCiAgICBjYXQgPSBxdWVyeV9jYWxpYnJhdGlvbl9jYXRhbG9ndWUod2hpY2gsIGNlbnRyZSwgcmFkaXVzKQogICAgcHJpbnQoZiIgICB7bGVuKGNhdCk6LH0gY2F0YWxvZ3VlIHN0YXJzIGluIHRoZSByZWdpb24iLCBmbHVzaD1UcnVlKQogICAgbSA9IG1hdGNoX2NhdGFsb2d1ZShkZiwgd2NzLCBjYXQsIHNoYXBlKQogICAgc3RhdHMgPSB7ImNhdGFsb2d1ZSI6IGxlbihjYXQpLCAibWF0Y2hlZCI6IGxlbihtKX0KICAgIGlmIG5vdCBsZW4obSk6CiAgICAgICAgcmV0dXJuIG0sIHN0YXRzCiAgICBrZWVwID0gfm1bIlNhdHVyYXRlZCJdLmFzdHlwZShib29sKQogICAgY3Jvd2RlZCA9IG1bIk5lYXJlc3RfcHgiXSA8IG1bIkFwZXJ0dXJlX1JhZGl1cyJdCiAgICBnb29kX2VyciA9ICgobVsiTWFnX2luc3RfQl9lcnIiXSA8IENBTElCX01BWF9JTlNUX0VSUikKICAgICAgICAgICAgICAgICYgKG1bIk1hZ19pbnN0X1ZfZXJyIl0gPCBDQUxJQl9NQVhfSU5TVF9FUlIpKQogICAgY2F0X29rID0gfigobVsiQ2F0X2VfQiJdID4gQ0FMSUJfTUFYX0NBVF9FUlIpIHwgKG1bIkNhdF9lX1YiXSA+IENBTElCX01BWF9DQVRfRVJSKSkKICAgIGtlZXAgJj0gfmNyb3dkZWQgJiBnb29kX2VyciAmIGNhdF9vawogICAgc3RhdHMudXBkYXRlKHsic2F0dXJhdGVkIjogaW50KG1bIlNhdHVyYXRlZCJdLmFzdHlwZShib29sKS5zdW0oKSksCiAgICAgICAgICAgICAgICAgICJjcm93ZGVkIjogaW50KGNyb3dkZWQuc3VtKCkpLCAibm9pc3kiOiBpbnQoKH5nb29kX2Vycikuc3VtKCkpLAogICAgICAgICAgICAgICAgICAiYmFkX2NhdGFsb2d1ZV9lcnJvciI6IGludCgofmNhdF9vaykuc3VtKCkpLAogICAgICAgICAgICAgICAgICAibWVkaWFuX3NlcF9hcmNzZWMiOiBmbG9hdChucC5tZWRpYW4obVsiU2VwX2FyY3NlYyJdKSl9KQogICAgaWYgY2VudHJlX3B4IGlzIG5vdCBOb25lIGFuZCByMjVfcHg6CiAgICAgICAgb3V0c2lkZSA9IG5wLmh5cG90KG1bIlhfViJdIC0gY2VudHJlX3B4WzBdLCBtWyJZX1YiXSAtIGNlbnRyZV9weFsxXSkgPiByMjVfcHgKICAgICAgICBpZiBpbnQoKGtlZXAgJiBvdXRzaWRlKS5zdW0oKSkgPj0gQ0FMSUJfTUlOX09VVFNJREU6CiAgICAgICAgICAgIHN0YXRzWyJpbnNpZGVfZ2FsYXh5X2xlZnRfb3V0Il0gPSBpbnQoKGtlZXAgJiB+b3V0c2lkZSkuc3VtKCkpCiAgICAgICAgICAgIGtlZXAgJj0gb3V0c2lkZQogICAgbVsiQ2FuZGlkYXRlIl0gPSBrZWVwLnZhbHVlcwogICAgcmV0dXJuIG0sIHN0YXRzCgoKZGVmIGZpdF9jYWxpYnJhdGlvbihjYW5kKToKICAgICIiIlplcm8gcG9pbnRzIGFuZCBjb2xvdXIgdGVybXMgZnJvbSB0aGUgY2FuZGlkYXRlIHN0YXJzLgoKICAgICAgICAoYi12KV9pbnN0ICAgICAgPSBhbHBoYSArIGJldGEgKiAoQi1WKV9jYXQgICAgICAodGhlIGNvbG91ciB0cmFuc2Zvcm1hdGlvbikKICAgICAgICAgICAgc28gKEItVikgICAgPSBhICsgYyAqIChiLXYpX2luc3QgICB3aXRoIGMgPSAxL2JldGEsIGEgPSAtYWxwaGEvYmV0YQogICAgICAgIFZfY2F0IC0gdl9pbnN0ICA9IFpQX1YgKyBrX1YgKiAoQi1WKV9jYXQKICAgICAgICBCX2NhdCAtIGJfaW5zdCAgPSBaUF9CICsga19CICogKEItVilfY2F0ICAgKHNob3duLCBhbmQgYSBjcm9zcy1jaGVjaykKCiAgICBUaGUgY29sb3VyIGlzIGZpdHRlZCB0aGlzIHdheSByb3VuZCBiZWNhdXNlIGEgc3RyYWlnaHQtbGluZSBmaXQgdHJlYXRzIHgKICAgIGFzIGV4YWN0OiB0aGUgY2F0YWxvZ3VlIGNvbG91ciBpcyB0aGUgcHJlY2lzZSBvbmUuIEV2ZXJ5IGZpdCBpcwogICAgc2lnbWEtY2xpcHBlZDsgYSBzdGFyIGNsaXBwZWQgZnJvbSB0aGUgY29sb3VyIG9yIHRoZSBWIGZpdCBpcyBtYXJrZWQgYXMKICAgIG5vdCB1c2VkLiBUaGlzIHJlcGxhY2VzIG9uZSB6ZXJvIHBvaW50IHBlciBiYW5kIGZyb20gdGhlIGJyaWdodGVzdCBzdGFycywKICAgIHdoaWNoIGxlZnQgdGhlIGNvbG91ciBvbiB0aGUgY2FtZXJhJ3Mgb3duIHNjYWxlIC0gd2l0aCBmaWx0ZXJzIHRoYXQgYXJlCiAgICBub3QgZXhhY3RseSBKb2huc29uIEIgYW5kIFYsIGV2ZXJ5IEItViB3YXMgb2ZmIGJ5IGEgZnJhY3Rpb24gb2YgaXRzZWxmLgogICAgIiIiCiAgICBjID0gY2FuZFtjYW5kWyJDYW5kaWRhdGUiXV0uY29weSgpCiAgICBidl9pbnN0ID0gY1siTWFnX2luc3RfQiJdIC0gY1siTWFnX2luc3RfViJdCiAgICBidl9jYXQgPSBjWyJDYXRfQiJdIC0gY1siQ2F0X1YiXQogICAgYWxwaGEsIGJldGEsIHVzZV9jLCBfID0gY2xpcHBlZF9saW5lKGJ2X2NhdCwgYnZfaW5zdCkKICAgIGN0ZXJtLCBhID0gMS4wIC8gYmV0YSwgLWFscGhhIC8gYmV0YQogICAgcm1zX2MgPSBmbG9hdChucC5zdGQoKGEgKyBjdGVybSAqIGJ2X2luc3QgLSBidl9jYXQpW3VzZV9jXSkpCiAgICB6cHYsIGt2LCB1c2Vfdiwgcm1zX3YgPSBjbGlwcGVkX2xpbmUoYnZfY2F0LCBjWyJDYXRfViJdIC0gY1siTWFnX2luc3RfViJdKQogICAgenBiLCBrYiwgdXNlX2IsIHJtc19iID0gY2xpcHBlZF9saW5lKGJ2X2NhdCwgY1siQ2F0X0IiXSAtIGNbIk1hZ19pbnN0X0IiXSkKICAgIHVzZWQgPSB1c2VfYyAmIHVzZV92CiAgICBjb3JyID0gZmxvYXQobnAuY29ycmNvZWYoYnZfaW5zdCwgYnZfY2F0KVswLCAxXSkgaWYgbGVuKGMpID4gMiBlbHNlIG5wLm5hbgogICAgcmV0dXJuIHsiYSI6IGEsICJjIjogY3Rlcm0sICJaUF9WIjogenB2LCAia19WIjoga3YsICJaUF9CIjogenBiLCAia19CIjoga2IsCiAgICAgICAgICAgICJybXNfY29sb3VyX2ZpdCI6IHJtc19jLCAicm1zX1ZfZml0Ijogcm1zX3YsICJybXNfQl9maXQiOiBybXNfYiwKICAgICAgICAgICAgIk5fY2FuZGlkYXRlcyI6IGxlbihjKSwgIk5fdXNlZCI6IGludCh1c2VkLnN1bSgpKSwKICAgICAgICAgICAgImNvcnJlbGF0aW9uIjogY29yciwgInVzZWRfaW5kZXgiOiBjLmluZGV4W3VzZWRdfQoKCmRlZiBhcHBseV9jYWxpYnJhdGlvbihkZiwgY2FsKToKICAgICIiIkNhbGlicmF0ZWQgQiwgViwgQi1WIGFuZCB0aGVpciBlcnJvcnMgZm9yIGV2ZXJ5IHNvdXJjZSBpbiBkZi4iIiIKICAgIGIsIHYgPSBkZlsiTWFnX2luc3RfQiJdLnRvX251bXB5KGZsb2F0KSwgZGZbIk1hZ19pbnN0X1YiXS50b19udW1weShmbG9hdCkKICAgIGViLCBldiA9IGRmWyJNYWdfaW5zdF9CX2VyciJdLnRvX251bXB5KGZsb2F0KSwgZGZbIk1hZ19pbnN0X1ZfZXJyIl0udG9fbnVtcHkoZmxvYXQpCiAgICBhLCBjLCB6cCwgayA9IGNhbFsiYSJdLCBjYWxbImMiXSwgY2FsWyJaUF9WIl0sIGNhbFsia19WIl0KICAgIGJ2ID0gYSArIGMgKiAoYiAtIHYpCiAgICBWID0gdiArIHpwICsgayAqIGJ2CiAgICBCID0gViArIGJ2CiAgICAjIFYgPSAoMSAtIGsgYykgdiArIGsgYyBiICsgY29uc3QgOyBCID0gKDEgLSAoMStrKSBjKSB2ICsgKDEraykgYyBiICsgY29uc3QKICAgIG91dCA9IGRmLmNvcHkoKQogICAgb3V0WyJNYWdfQl9vYnMiXSwgb3V0WyJNYWdfVl9vYnMiXSwgb3V0WyJCVl9vYnMiXSA9IEIsIFYsIGJ2CiAgICBvdXRbIk1hZ19CX2VyciJdID0gbnAuaHlwb3QoKDEgLSAoMSArIGspICogYykgKiBldiwgKDEgKyBrKSAqIGMgKiBlYikKICAgIG91dFsiTWFnX1ZfZXJyIl0gPSBucC5oeXBvdCgoMSAtIGsgKiBjKSAqIGV2LCBrICogYyAqIGViKQogICAgb3V0WyJCVl9lcnIiXSA9IG5wLmFicyhjKSAqIG5wLmh5cG90KGViLCBldikKICAgIHJldHVybiBvdXQKCgpkZWYgY2FsaWJyYXRlKGRmLCB3Y3MsIHNoYXBlLCB3aGljaCwgY2VudHJlX3B4LCByMjVfcHgsIG91dF9kaXIsIHNhZmUpOgogICAgIiIiUnVuIHRoZSB3aG9sZSBjYWxpYnJhdGlvbiwgd2l0aCBHYWlhIGFzIHRoZSBmYWxsYmFjayBjYXRhbG9ndWUsIHRoZQogICAgY2hlY2tzIHRoYXQgc3RvcCBhIHJ1biB0aGF0IGNhbm5vdCBiZSByaWdodCwgYW5kIHRoZSBjYWxpYnJhdGlvbiB0YWJsZQogICAgd3JpdHRlbiBvdXQuIFJldHVybnMgKGNhbCwgY2F0YWxvZ3VlIHVzZWQsIHN0YXRzLCBtYXRjaGVkIHRhYmxlKS4iIiIKICAgIGNhbmQsIHN0YXRzID0gY2FsaWJyYXRpb25fY2FuZGlkYXRlcyhkZiwgd2NzLCBzaGFwZSwgd2hpY2gsIGNlbnRyZV9weCwgcjI1X3B4KQogICAgbl9jYW5kID0gaW50KGNhbmRbIkNhbmRpZGF0ZSJdLnN1bSgpKSBpZiBsZW4oY2FuZCkgZWxzZSAwCiAgICBpZiBuX2NhbmQgPCBDQUxJQl9NSU5fU1RBUlMgYW5kIHdoaWNoID09ICJBUEFTUyI6CiAgICAgICAgcHJpbnQoZiJbY2FsaWJdIG9ubHkge25fY2FuZH0gdXNhYmxlIEFQQVNTIHN0YXJzIC0gdHJ5aW5nIEdhaWEgRFIzIHN5bnRoZXRpYyAiCiAgICAgICAgICAgICAgZiJwaG90b21ldHJ5IGluc3RlYWQiLCBmbHVzaD1UcnVlKQogICAgICAgIHdoaWNoID0gIkdBSUEiCiAgICAgICAgY2FuZCwgc3RhdHMgPSBjYWxpYnJhdGlvbl9jYW5kaWRhdGVzKGRmLCB3Y3MsIHNoYXBlLCB3aGljaCwgY2VudHJlX3B4LCByMjVfcHgpCiAgICAgICAgbl9jYW5kID0gaW50KGNhbmRbIkNhbmRpZGF0ZSJdLnN1bSgpKSBpZiBsZW4oY2FuZCkgZWxzZSAwCiAgICBpZiBuX2NhbmQgPCBDQUxJQl9NSU5fU1RBUlM6CiAgICAgICAgc3RvcChmIkNBTElCUkFUSU9OOiBvbmx5IHtuX2NhbmR9IHVzYWJsZSByZWZlcmVuY2Ugc3RhcnMgKGF0IGxlYXN0ICIKICAgICAgICAgICAgIGYie0NBTElCX01JTl9TVEFSU30gYXJlIG5lZWRlZCkuIE1hdGNoZWQge3N0YXRzLmdldCgnbWF0Y2hlZCcsIDApfSwgb2YgIgogICAgICAgICAgICAgZiJ3aGljaCBzYXR1cmF0ZWQge3N0YXRzLmdldCgnc2F0dXJhdGVkJywgMCl9LCBjcm93ZGVkICIKICAgICAgICAgICAgIGYie3N0YXRzLmdldCgnY3Jvd2RlZCcsIDApfSwgdG9vIG5vaXN5IHtzdGF0cy5nZXQoJ25vaXN5JywgMCl9LiIpCiAgICBwcmludChmIltjYWxpYl0ge3N0YXRzWydtYXRjaGVkJ119IGNhdGFsb2d1ZSBzdGFycyBtYXRjaGVkIChtZWRpYW4gc2VwYXJhdGlvbiAiCiAgICAgICAgICBmIntzdGF0c1snbWVkaWFuX3NlcF9hcmNzZWMnXTouMmZ9IGFyY3NlYyk7IGxlZnQgb3V0OiAiCiAgICAgICAgICBmIntzdGF0c1snc2F0dXJhdGVkJ119IHNhdHVyYXRlZCwge3N0YXRzWydjcm93ZGVkJ119IGNyb3dkZWQsICIKICAgICAgICAgIGYie3N0YXRzWydub2lzeSddfSB0b28gbm9pc3ksIHtzdGF0c1snYmFkX2NhdGFsb2d1ZV9lcnJvciddfSB3aXRoIGxhcmdlICIKICAgICAgICAgIGYiY2F0YWxvZ3VlIGVycm9ycyIKICAgICAgICAgICsgKGYiLCB7c3RhdHNbJ2luc2lkZV9nYWxheHlfbGVmdF9vdXQnXX0gb24gdGhlIGdhbGF4eSdzIGRpc2MiCiAgICAgICAgICAgICBpZiBzdGF0cy5nZXQoImluc2lkZV9nYWxheHlfbGVmdF9vdXQiKSBlbHNlICIiKSwgZmx1c2g9VHJ1ZSkKICAgIGNhbCA9IGZpdF9jYWxpYnJhdGlvbihjYW5kKQogICAgaWYgY2FsWyJOX3VzZWQiXSA8IENBTElCX01JTl9TVEFSUzoKICAgICAgICBzdG9wKGYiQ0FMSUJSQVRJT046IG9ubHkge2NhbFsnTl91c2VkJ119IHN0YXJzIHN1cnZpdmVkIHRoZSBvdXRsaWVyICIKICAgICAgICAgICAgIGYicmVqZWN0aW9uIChhdCBsZWFzdCB7Q0FMSUJfTUlOX1NUQVJTfSBhcmUgbmVlZGVkKS4iKQoKICAgICMgVGhlIGNvbG91ciBjaGVjay4gVGhlIGluc3RydW1lbnRhbCBjb2xvdXIgbXVzdCByaXNlIHdpdGggdGhlIGNhdGFsb2d1ZQogICAgIyBjb2xvdXI7IGlmIGl0IGZhbGxzLCB0aGUgZnJhbWUgbGFiZWxsZWQgQiBpcyBub3QgdGhlIGJsdWUgb25lLiBOb3RoaW5nCiAgICAjIGVsc2UgaW4gdGhlIG91dHB1dCB3b3VsZCBzaG93IGl0OiB0aGUgZGlhZ3JhbSB3b3VsZCBsb29rIG5vcm1hbCwgd2l0aAogICAgIyBldmVyeSBjb2xvdXIgcmV2ZXJzZWQuCiAgICBpZiBub3QgKGNhbFsiY29ycmVsYXRpb24iXSA+IDApOgogICAgICAgIHN0b3AoZiJDT0xPVVIgQ0hFQ0s6IHRoZSBpbnN0cnVtZW50YWwgY29sb3VyIGItdiBGQUxMUyBhcyB0aGUgY2F0YWxvZ3VlICIKICAgICAgICAgICAgIGYiQi1WIHJpc2VzIChjb3JyZWxhdGlvbiB7Y2FsWydjb3JyZWxhdGlvbiddOisuMmZ9IG92ZXIgIgogICAgICAgICAgICAgZiJ7Y2FsWydOX2NhbmRpZGF0ZXMnXX0gc3RhcnMpLiBUaGUgZnJhbWUgdXNlZCBhcyBCIGlzIHByb2JhYmx5IHRoZSAiCiAgICAgICAgICAgICBmIlYgKG9yIFIpIGZyYW1lIGFuZCB0aGUgb3RoZXIgd2F5IHJvdW5kIC0gY2hlY2sgdGhlIEZJTFRFUiBrZXl3b3JkcyAiCiAgICAgICAgICAgICBmImFuZCB0aGUgZmlsZSBuYW1lcy4iKQogICAgaWYgY2FsWyJjb3JyZWxhdGlvbiJdIDwgMC41OgogICAgICAgIGxvdWQoZiJXQVJOSU5HOiB0aGUgaW5zdHJ1bWVudGFsIGFuZCBjYXRhbG9ndWUgY29sb3VycyBhZ3JlZSBvbmx5IHdlYWtseSAiCiAgICAgICAgICAgICBmIihjb3JyZWxhdGlvbiB7Y2FsWydjb3JyZWxhdGlvbiddOi4yZn0pLiBUaGUgY29sb3VycyBiZWxvdyBhcmUgIgogICAgICAgICAgICAgZiJ1bmNlcnRhaW4uIikKICAgIHByaW50KGYiW2NhbGliXSB7Y2FsWydOX3VzZWQnXX0gb2Yge2NhbFsnTl9jYW5kaWRhdGVzJ119IHN0YXJzIHVzZWQgIgogICAgICAgICAgZiIoeydBUEFTUyBEUjknIGlmIHdoaWNoID09ICdBUEFTUycgZWxzZSAnR2FpYSBEUjMgc3ludGhldGljJ30pOiIsIGZsdXNoPVRydWUpCiAgICBwcmludChmIiAgIEItViA9IHtjYWxbJ2EnXTorLjNmfSArIHtjYWxbJ2MnXTouM2Z9IHggKGItdikgICAgICBzY2F0dGVyICIKICAgICAgICAgIGYie2NhbFsncm1zX2NvbG91cl9maXQnXTouM2Z9IG1hZyIsIGZsdXNoPVRydWUpCiAgICBwcmludChmIiAgIFYgICA9IHYgKyB7Y2FsWydaUF9WJ106LjNmfSB7Y2FsWydrX1YnXTorLjNmfSB4IChCLVYpICAgc2NhdHRlciAiCiAgICAgICAgICBmIntjYWxbJ3Jtc19WX2ZpdCddOi4zZn0gbWFnIiwgZmx1c2g9VHJ1ZSkKICAgIHByaW50KGYiICAgQiAgID0gYiArIHtjYWxbJ1pQX0InXTouM2Z9IHtjYWxbJ2tfQiddOisuM2Z9IHggKEItVikgICBzY2F0dGVyICIKICAgICAgICAgIGYie2NhbFsncm1zX0JfZml0J106LjNmfSBtYWciLCBmbHVzaD1UcnVlKQogICAgcHJpbnQoZiIgICBjb2xvdXIgY29ycmVsYXRpb24ge2NhbFsnY29ycmVsYXRpb24nXTorLjJmfSIsIGZsdXNoPVRydWUpCgogICAgdGFibGUgPSBjYW5kLmNvcHkoKQogICAgdGFibGVbIlVzZWQiXSA9IEZhbHNlCiAgICB0YWJsZS5sb2NbY2FsWyJ1c2VkX2luZGV4Il0sICJVc2VkIl0gPSBUcnVlCiAgICBidl9maXQgPSBjYWxbImEiXSArIGNhbFsiYyJdICogKHRhYmxlWyJNYWdfaW5zdF9CIl0gLSB0YWJsZVsiTWFnX2luc3RfViJdKQogICAgdGFibGVbIlJlc2lkX0JWIl0gPSBidl9maXQgLSAodGFibGVbIkNhdF9CIl0gLSB0YWJsZVsiQ2F0X1YiXSkKICAgIHRhYmxlWyJSZXNpZF9WIl0gPSAodGFibGVbIk1hZ19pbnN0X1YiXSArIGNhbFsiWlBfViJdICsgY2FsWyJrX1YiXSAqIGJ2X2ZpdAogICAgICAgICAgICAgICAgICAgICAgICAtIHRhYmxlWyJDYXRfViJdKQogICAgcGF0aCA9IG9zLnBhdGguam9pbihvdXRfZGlyLCBmIntzYWZlfV9yZWZlcmVuY2Vfc3RhcnMuY3N2IikKICAgIHRhYmxlLnRvX2NzdihwYXRoLCBpbmRleD1GYWxzZSkKICAgIHByaW50KGYiUmVmZXJlbmNlIHN0YXJzLCB3aXRoIGVhY2ggb25lJ3MgcmVzaWR1YWwsIHNhdmVkIHRvIHtwYXRofSIpCiAgICByZXR1cm4gY2FsLCB3aGljaCwgc3RhdHMsIGNhbmQKCgojIC0tLS0tLS0tLS0tLS0tLS0tLS0gRGlhZ3JhbXMgLS0tLS0tLS0tLS0tLS0tLS0tLQoKZGVmIG1ha2VfY21kKGRmLCBiYXNlX2ZpbGUsIGxhYmVsLCBuYW1lLCBtdT1Ob25lLCB3aW5kb3c9Tm9uZSk6CiAgICAiIiJDb2xvdXItbWFnbml0dWRlIGRpYWdyYW0gb2YgdGhlIHNvdXJjZXMgaW4gZGYsIHNhdmVkIG5leHQgdG8gYmFzZV9maWxlLgoKICAgIFRoZSBjb2xvdXIgYXhpcyBmb2xsb3dzIHRoZSBkYXRhIHJhdGhlciB0aGFuIGEgZml4ZWQgcmFuZ2UgLSB0aGUgb2xkIGZpeGVkCiAgICAtMC41IHRvIDAuNSBjdXQgZXZlcnkgcmVkIHNvdXJjZSBvdXQgb2YgdGhlIGRpYWdyYW0gb2YgYWxsIHNvdXJjZXMsIHdoaWNoCiAgICBpcyB0aGUgZGlhZ3JhbSB0aGF0IHNob3dzIHdoZXJlIHRoZSBibHVlIGtub3RzIHN0YW5kIGFnYWluc3QgdGhlIHJlc3QuCiAgICBTYXR1cmF0ZWQgc291cmNlcyBhbmQgc291cmNlcyB3aXRoIGEgQi1WIGVycm9yIGFib3ZlIENNRF9NQVhfQlZfRVJSIGFyZSBub3QKICAgIGRyYXduICh0aGV5IHN0YXkgaW4gdGhlIHRhYmxlcykuIFdpdGggYSBkaXN0YW5jZSwgdGhlIHJpZ2h0LWhhbmQgYXhpcyBpcwogICAgdGhlIGFic29sdXRlIG1hZ25pdHVkZSBNX1YuIEJhcnMgb24gdGhlIGxlZnQgc2hvdyB0aGUgdHlwaWNhbCBlcnJvciBhdAogICAgZWFjaCBtYWduaXR1ZGUuCiAgICAiIiIKICAgIGlmIG5vdCB7Ik1hZ19CIiwgIk1hZ19WIn0gPD0gc2V0KGRmLmNvbHVtbnMpOgogICAgICAgIHByaW50KGYiTWFnX0Igb3IgTWFnX1Ygbm90IGZvdW5kIGluIHtsYWJlbH0gY2F0YWxvZy4gQ01EIG5vdCBjcmVhdGVkLiIpCiAgICAgICAgcmV0dXJuCiAgICBkZiA9IGRmLmNvcHkoKQogICAgZGZbIkItViJdID0gZGZbIk1hZ19CIl0gLSBkZlsiTWFnX1YiXQogICAgY29scyA9IFtjIGZvciBjIGluIGRmLmNvbHVtbnMgaWYgYyBub3QgaW4gKCJCLVYiLCAiTWFnX1YiKV0gKyBbIkItViIsICJNYWdfViJdCiAgICBkZiA9IGRmW2NvbHNdCiAgICBvdXRfZmlsZV9jb2xvciA9IGJhc2VfZmlsZS5yZXBsYWNlKCIuY3N2IiwgIl93aXRoX2NvbG9yLmNzdiIpCiAgICBkZi50b19jc3Yob3V0X2ZpbGVfY29sb3IsIGluZGV4PUZhbHNlKQogICAgcHJpbnQoZiJGaWxlIHdpdGggY29sb3IgaW5kZXggc2F2ZWQgdG8ge291dF9maWxlX2NvbG9yfSIpCgogICAgc2hvdyA9IG5wLmlzZmluaXRlKGRmWyJCLVYiXSkgJiBucC5pc2Zpbml0ZShkZlsiTWFnX1YiXSkKICAgIGlmICJTYXR1cmF0ZWQiIGluIGRmLmNvbHVtbnM6CiAgICAgICAgc2hvdyAmPSB+ZGZbIlNhdHVyYXRlZCJdLmFzdHlwZShib29sKQogICAgaWYgIkJWX2VyciIgaW4gZGYuY29sdW1uczoKICAgICAgICBzaG93ICY9IGRmWyJCVl9lcnIiXSA8PSBDTURfTUFYX0JWX0VSUgogICAgZCA9IGRmW3Nob3ddCiAgICBpZiBub3QgbGVuKGQpOgogICAgICAgIHByaW50KGYiTm8gc291cmNlcyB0byBkcmF3IGluIHRoZSB7bGFiZWx9IGRpYWdyYW0uIikKICAgICAgICByZXR1cm4KCiAgICBmaWcsIGF4ID0gcGx0LnN1YnBsb3RzKGZpZ3NpemU9KDgsIDEwKSkKICAgIGF4LnNjYXR0ZXIoZFsiQi1WIl0sIGRbIk1hZ19WIl0sIHM9MTgsIGVkZ2Vjb2xvcj0iYmxhY2siLCBmYWNlY29sb3I9ImN5YW4iLAogICAgICAgICAgICAgICBhbHBoYT0wLjcsIGxpbmV3aWR0aHM9MC40KQogICAgbG8sIGhpID0gbnAucGVyY2VudGlsZShkWyJCLVYiXSwgWzEsIDk5XSkKICAgIGxvLCBoaSA9IG1pbihsbywgd2luZG93WzBdIGlmIHdpbmRvdyBlbHNlIGxvKSwgbWF4KGhpLCB3aW5kb3dbMV0gaWYgd2luZG93IGVsc2UgaGkpCiAgICBwYWQgPSAwLjEgKiBtYXgoaGkgLSBsbywgMC41KQogICAgYXguc2V0X3hsaW0obG8gLSBwYWQsIGhpICsgcGFkKQogICAgeW1pbiwgeW1heCA9IG5wLm5hbm1pbihkWyJNYWdfViJdKSwgbnAubmFubWF4KGRbIk1hZ19WIl0pCiAgICBheC5zZXRfeWxpbSh5bWF4ICsgMC4zLCB5bWluIC0gMC4zKQogICAgaWYgd2luZG93OgogICAgICAgIGZvciB4dyBpbiB3aW5kb3c6CiAgICAgICAgICAgIGF4LmF4dmxpbmUoeHcsIGNvbG9yPSJ0YWI6Ymx1ZSIsIGxzPSItLSIsIGx3PTEpCiAgICBpZiB7Ik1hZ19WX2VyciIsICJCVl9lcnIifSA8PSBzZXQoZC5jb2x1bW5zKToKICAgICAgICB4X2JhciA9IGxvIC0gMC41ICogcGFkCiAgICAgICAgZm9yIG0wIGluIG5wLmFyYW5nZShucC5mbG9vcih5bWluKSwgbnAuY2VpbCh5bWF4KSwgMS4wKToKICAgICAgICAgICAgc2VsID0gKGRbIk1hZ19WIl0gPj0gbTApICYgKGRbIk1hZ19WIl0gPCBtMCArIDEpCiAgICAgICAgICAgIGlmIHNlbC5zdW0oKSA+PSAzOgogICAgICAgICAgICAgICAgYXguZXJyb3JiYXIoeF9iYXIsIG0wICsgMC41LCB4ZXJyPW5wLm1lZGlhbihkLmxvY1tzZWwsICJCVl9lcnIiXSksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICB5ZXJyPW5wLm1lZGlhbihkLmxvY1tzZWwsICJNYWdfVl9lcnIiXSksIGNvbG9yPSIwLjMiLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgY2Fwc2l6ZT0yLCBsdz0xKQogICAgYXguc2V0X3hsYWJlbCgiKEIgLSBWKSRfMCQgIFttYWddLCBjb3JyZWN0ZWQgZm9yIEdhbGFjdGljIHJlZGRlbmluZyIpCiAgICBheC5zZXRfeWxhYmVsKCJWJF8wJCAgW21hZ10sIGFwcGFyZW50LCBjb3JyZWN0ZWQgZm9yIEdhbGFjdGljIGV4dGluY3Rpb24iKQogICAgaWYgbXUgaXMgbm90IE5vbmUgYW5kIG5wLmlzZmluaXRlKG11KToKICAgICAgICBzZWMgPSBheC5zZWNvbmRhcnlfeWF4aXMoInJpZ2h0IiwgZnVuY3Rpb25zPShsYW1iZGEgbTogbSAtIG11LCBsYW1iZGEgTTogTSArIG11KSkKICAgICAgICBzZWMuc2V0X3lsYWJlbChmIk0kX1YkICBbbWFnXSAgIChkaXN0YW5jZSBtb2R1bHVzIHttdTouMmZ9KSIpCiAgICBsZWZ0X291dCA9IGludChsZW4oZGYpIC0gbGVuKGQpKQogICAgYXguc2V0X3RpdGxlKGYiQ01EIG9mIHtuYW1lfSB7bGFiZWx9XG57bGVuKGQpfSBvZiB7bGVuKGRmKX0gc291cmNlcyBkcmF3biIKICAgICAgICAgICAgICAgICArIChmIiAoe2xlZnRfb3V0fSBzYXR1cmF0ZWQgb3Igd2l0aCBhIEItViBlcnJvciBhYm92ZSAiCiAgICAgICAgICAgICAgICAgICAgZiJ7Q01EX01BWF9CVl9FUlI6Z30gbWFnIGxlZnQgb3V0KSIgaWYgbGVmdF9vdXQgZWxzZSAiIikKICAgICAgICAgICAgICAgICArIChmIjsgZGFzaGVkIGxpbmVzOiB0aGUgYmx1ZS1rbm90IGNvbG91ciB3aW5kb3ciIGlmIHdpbmRvdyBlbHNlICIiKSkKICAgIG91dF9jbWQgPSBiYXNlX2ZpbGUucmVwbGFjZSgiLmNzdiIsICJfQ01ELnBuZyIpCiAgICBmaWcuc2F2ZWZpZyhvdXRfY21kLCBkcGk9MTUwLCBiYm94X2luY2hlcz0idGlnaHQiKQogICAgcGx0LmNsb3NlKGZpZykKICAgIHByaW50KGYiQ01EIGRpYWdyYW0gc2F2ZWQgdG8ge291dF9jbWR9IikKCgpkZWYgc291cmNlX21hcChkYXRhLCB4LCB5LCB0aXRsZSwgcGF0aCwgY29sb3VyPSJncmVlbiIsIGxhYmVsPSJNZWFzdXJlZCBzb3VyY2VzIik6CiAgICBwbHQuZmlndXJlKGZpZ3NpemU9KDEwLCAxMCkpCiAgICBmaW5pdGUgPSBkYXRhW25wLmlzZmluaXRlKGRhdGEpXQogICAgcGx0Lmltc2hvdyhkYXRhLCBjbWFwPSJncmF5Iiwgb3JpZ2luPSJsb3dlciIsCiAgICAgICAgICAgICAgIHZtaW49bnAucGVyY2VudGlsZShmaW5pdGUsIDUpLCB2bWF4PW5wLnBlcmNlbnRpbGUoZmluaXRlLCA5OSkpCiAgICBwbHQuY29sb3JiYXIobGFiZWw9IkNvdW50cyIpCiAgICBwbHQuc2NhdHRlcih4LCB5LCBzPTQwLCBlZGdlY29sb3I9Y29sb3VyLCBmYWNlY29sb3I9Im5vbmUiLCBsYWJlbD1sYWJlbCkKICAgIHBsdC50aXRsZSh0aXRsZSkKICAgIHBsdC54bGFiZWwoIlggW3BpeGVsc10iKQogICAgcGx0LnlsYWJlbCgiWSBbcGl4ZWxzXSIpCiAgICBwbHQubGVnZW5kKCkKICAgIHBsdC5zYXZlZmlnKHBhdGgsIGRwaT0xNTApCiAgICBwbHQuY2xvc2UoKQogICAgcHJpbnQoZiJTb3VyY2UgbWFwIHNhdmVkIHRvIHtwYXRofSIpCgoKZGVmIGdhbGF4eV9jZW50cmVfcGl4ZWwobmFtZSwgd2NzLCBzaGFwZSk6CiAgICAiIiJXaGVyZSB0aGUgZ2FsYXh5IHNpdHMgaW4gdGhpcyBmcmFtZSwgaW4gcGl4ZWxzLCBmcm9tIGl0cyBjYXRhbG9ndWVkCiAgICBwb3NpdGlvbiB0aHJvdWdoIHRoZSAoY2hlY2tlZCkgcGxhdGUgc29sdXRpb247IE5vbmUgd2hlbiB1bmtub3duIG9yCiAgICBvdXRzaWRlIHRoZSBmcmFtZS4iIiIKICAgIHRyeToKICAgICAgICB4LCB5ID0gd2NzLndvcmxkX3RvX3BpeGVsKFNreUNvb3JkLmZyb21fbmFtZShuYW1lKSkKICAgICAgICB4LCB5ID0gZmxvYXQoeCksIGZsb2F0KHkpCiAgICAgICAgaWYgMCA8PSB4IDwgc2hhcGVbMV0gYW5kIDAgPD0geSA8IHNoYXBlWzBdOgogICAgICAgICAgICBwcmludChmIltsb29rdXBdIHtuYW1lfTogY2VudHJlZCBhdCBwaXhlbCAoe3g6LjBmfSwge3k6LjBmfSkiLCBmbHVzaD1UcnVlKQogICAgICAgICAgICByZXR1cm4geCwgeQogICAgICAgIHByaW50KGYiW2xvb2t1cF0ge25hbWV9J3MgY2F0YWxvZ3VlZCBwb3NpdGlvbiBmYWxscyBvdXRzaWRlIHRoaXMgZnJhbWUiLAogICAgICAgICAgICAgIGZsdXNoPVRydWUpCiAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGV4YzoKICAgICAgICBwcmludChmIltsb29rdXBdIGdhbGF4eSBjZW50cmUgdW5hdmFpbGFibGUgKHtleGN9KSIsIGZsdXNoPVRydWUpCiAgICByZXR1cm4gTm9uZQoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLSBUaGUgcnVuIC0tLS0tLS0tLS0tLS0tLS0tLS0KCmRlZiBwYXJzZV9hcmdzKGFyZ3Y9Tm9uZSk6CiAgICBhcCA9IGFyZ3BhcnNlLkFyZ3VtZW50UGFyc2VyKAogICAgICAgIGRlc2NyaXB0aW9uPSJCbHVlIGtub3RzIG9mIGEgZ2FsYXh5IGZyb20gYSBCIGFuZCBhIFYgZnJhbWU6IGNhbGlicmF0ZWQgIgogICAgICAgICAgICAgICAgICAgICJwaG90b21ldHJ5LCBjb2xvdXItbWFnbml0dWRlIGRpYWdyYW0sIHJhZGlhbCBwcm9maWxlLiAiCiAgICAgICAgICAgICAgICAgICAgIlN0YXJ0ZWQgd2l0aG91dCBhcmd1bWVudHMgaXQgYXNrcyBpdHMgcXVlc3Rpb25zIGluc3RlYWQuIikKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1mb2xkZXIiLCBoZWxwPSJmb2xkZXIgaG9sZGluZyB0aGUgZnJhbWVzOyB3aGljaCBpcyBCIGFuZCB3aGljaCAiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAiaXMgViBpcyByZWFkIGZyb20gdGhlIGhlYWRlcnMiKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLWIiLCBoZWxwPSJ0aGUgQiBmcmFtZSwgd2hlbiBuYW1pbmcgdGhlIGZpbGVzIHlvdXJzZWxmIikKICAgIGFwLmFkZF9hcmd1bWVudCgiLS12IiwgaGVscD0idGhlIFYgZnJhbWUiKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLWdhbGF4eSIsIGhlbHA9ImdhbGF4eSBuYW1lIChvdGhlcndpc2UgZnJvbSB0aGUgT0JKRUNUIGtleXdvcmQpIikKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1kaXN0YW5jZS1tcGMiLCB0eXBlPWZsb2F0LCBoZWxwPSJkaXN0YW5jZTsgb3RoZXJ3aXNlIGxvb2tlZCB1cCIpCiAgICBhcC5hZGRfYXJndW1lbnQoIi0tZWJ2IiwgdHlwZT1mbG9hdCwgaGVscD0iR2FsYWN0aWMgRShCLVYpOyBvdGhlcndpc2UgbG9va2VkIHVwIikKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1jYXRhbG9nIiwgY2hvaWNlcz1bIkFQQVNTIiwgIkdBSUEiXSwgZGVmYXVsdD1DQUxJQl9DQVRBTE9HVUUsCiAgICAgICAgICAgICAgICAgICAgaGVscD0icmVmZXJlbmNlIHN0YXJzIGZvciB0aGUgY2FsaWJyYXRpb24gKGRlZmF1bHQgQVBBU1M7IEdhaWEgRFIzICIKICAgICAgICAgICAgICAgICAgICAgICAgICJzeW50aGV0aWMgcGhvdG9tZXRyeSBpcyBhbHNvIHRoZSBhdXRvbWF0aWMgZmFsbGJhY2spIikKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1tYXgtYnYtZXJyIiwgdHlwZT1mbG9hdCwgZGVmYXVsdD1LTk9UX01BWF9CVl9FUlIsCiAgICAgICAgICAgICAgICAgICAgaGVscD0ibGFyZ2VzdCBCLVYgZXJyb3IgYSBibHVlIGtub3QgbWF5IGhhdmUsIGluIG1hZyIpCiAgICBhcC5hZGRfYXJndW1lbnQoIi0tb3V0LWRpciIsIGhlbHA9IndyaXRlIGV2ZXJ5dGhpbmcgaW50byB0aGlzIGZvbGRlciAoZGVmYXVsdDogYSAiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIm5ldyBmb2xkZXIgcmVzdWx0X0NNRF88Z2FsYXh5Pl88ZGF0ZT5fPHRpbWU+ICIKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAiYmVzaWRlIHRoZSBmcmFtZXMpIikKICAgIGFwLmFkZF9hcmd1bWVudCgiLS15ZXMiLCBhY3Rpb249InN0b3JlX3RydWUiLAogICAgICAgICAgICAgICAgICAgIGhlbHA9ImFjY2VwdCBldmVyeSBkZWZhdWx0IHdpdGhvdXQgYXNraW5nIikKICAgIHJldHVybiBhcC5wYXJzZV9hcmdzKGFyZ3YpCgoKZGVmIG1haW4oYXJndj1Ob25lKToKICAgIGdsb2JhbCBBUkdTLCBhc2ssIGFza195ZXNfbm8KICAgIEFSR1MgPSBwYXJzZV9hcmdzKGFyZ3YpCiAgICBpZiBBUkdTLnllczoKICAgICAgICAjIFRoZSBzaGFyZWQgZm9sZGVyIHF1ZXN0aW9ucyAod2hpY2ggZnJhbWUsIHdoZXRoZXIgdG8gY29tYmluZSkgdGFrZQogICAgICAgICMgdGhlaXIgZGVmYXVsdHMgdG9vLgogICAgICAgIGFzayA9IGxhbWJkYSBwcm9tcHQsIGRlZmF1bHQ9Tm9uZSwgY2FzdD1zdHI6IGNhc3QoZGVmYXVsdCkKICAgICAgICBhc2tfeWVzX25vID0gbGFtYmRhIHByb21wdCwgZGVmYXVsdD1GYWxzZTogZGVmYXVsdAoKICAgIHByaW50KCI9PT0gUGhvdG9tZXRyeSBJbnB1dCA9PT0iKQogICAgZm9sZGVyID0gQVJHUy5mb2xkZXIKICAgIGlmIG5vdCBmb2xkZXIgYW5kIG5vdCAoQVJHUy5iIGFuZCBBUkdTLnYpOgogICAgICAgIGZvbGRlciA9IF9ub3JtX3BhdGgoX2FzaygKICAgICAgICAgICAgIlBhdGggdG8gdGhlIGZvbGRlciBob2xkaW5nIHRoZSBmcmFtZXMgKEVudGVyIHRvIG5hbWUgdGhlIHR3byBmaWxlcyBpbnN0ZWFkKSIsCiAgICAgICAgICAgICIiLCBzdHIpKQogICAgZml0c19maWxlX0IsIGZpdHNfZmlsZV9WID0gQVJHUy5iLCBBUkdTLnYKICAgIG9ial9uYW1lID0gQVJHUy5nYWxheHkKICAgIGdyb3VwcywgZG9ub3JzID0ge30sIHt9CiAgICBpZiBmb2xkZXIgYW5kIG9zLnBhdGguaXNkaXIoZm9sZGVyKToKICAgICAgICBjaG9zZW4sIGRvbm9ycywgZ3JvdXBzLCBmb3VuZCA9IGZyYW1lc19mcm9tX2ZvbGRlcihmb2xkZXIsICgiQiIsICJWIikpCiAgICAgICAgZml0c19maWxlX0IsIGZpdHNfZmlsZV9WID0gY2hvc2VuWyJCIl0sIGNob3NlblsiViJdCiAgICAgICAgb2JqX25hbWUgPSBvYmpfbmFtZSBvciBmb3VuZAogICAgZWxpZiBmb2xkZXI6CiAgICAgICAgcHJpbnQoIltpbnB1dF0gdGhlcmUgaXMgbm8gZm9sZGVyIGF0ICIgKyBmb2xkZXIgKyAiIC0gbmFtaW5nIHRoZSB0d28gZmlsZXMgIgogICAgICAgICAgICAgICJpbnN0ZWFkIiwgZmx1c2g9VHJ1ZSkKICAgICAgICBmb2xkZXIgPSBOb25lCiAgICBpZiBub3QgZml0c19maWxlX0I6CiAgICAgICAgZml0c19maWxlX0IgPSBfbm9ybV9wYXRoKF9hc2soIlBhdGggdG8gQi1iYW5kIEZJVFMiLCBOb25lLCBzdHIpKQogICAgaWYgbm90IGZpdHNfZmlsZV9WOgogICAgICAgIGZpdHNfZmlsZV9WID0gX25vcm1fcGF0aChfYXNrKCJQYXRoIHRvIFYtYmFuZCBGSVRTIiwgTm9uZSwgc3RyKSkKICAgIGZvciBwIGluIChmaXRzX2ZpbGVfQiwgZml0c19maWxlX1YpOgogICAgICAgIGlmIG5vdCBvcy5wYXRoLmlzZmlsZShwKToKICAgICAgICAgICAgc3RvcChmInRoZXJlIGlzIG5vIGZpbGUgYXQge3B9IikKICAgIGlmIG5vdCBvYmpfbmFtZToKICAgICAgICB0cnk6CiAgICAgICAgICAgIG9ial9uYW1lID0gc3RyKGZpdHMuZ2V0aGVhZGVyKGZpdHNfZmlsZV9WKS5nZXQoIk9CSkVDVCIsICIiKSkuc3RyaXAoKSBvciBOb25lCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgb2JqX25hbWUgPSBOb25lCiAgICBpZiBub3Qgb2JqX25hbWU6CiAgICAgICAgb2JqX25hbWUgPSBfYXNrKCJHYWxheHkgbmFtZSIsIE5vbmUsIHN0cikKICAgIHNhZmUgPSByZS5zdWIociJbXkEtWmEtejAtOV8uLV0rIiwgIl8iLCBvYmpfbmFtZS5zdHJpcCgpKQoKICAgIGRhdGFkaXIgPSBmb2xkZXIgb3Igb3MucGF0aC5kaXJuYW1lKG9zLnBhdGguYWJzcGF0aChmaXRzX2ZpbGVfVikpCiAgICBpZiBBUkdTLm91dF9kaXI6CiAgICAgICAgb3V0X2RpciA9IG9zLnBhdGguYWJzcGF0aChBUkdTLm91dF9kaXIpCiAgICAgICAgb3MubWFrZWRpcnMob3V0X2RpciwgZXhpc3Rfb2s9VHJ1ZSkKICAgIGVsc2U6CiAgICAgICAgb3V0X2RpciA9IG5ld19ydW5fZm9sZGVyKGRhdGFkaXIsICJDTUQiLCBvYmpfbmFtZSkKICAgIHByaW50KGYiW291dF0gd3JpdGluZyBldmVyeXRoaW5nIHRvIHtvdXRfZGlyfSIpCgogICAgY29tYmluZWRfbm90ZSA9IFtdCiAgICBmb3Igcm9sZSBpbiAoIkIiLCAiViIpOgogICAgICAgIHBhdGhzID0gZ3JvdXBzLmdldChyb2xlKQogICAgICAgIGlmIG5vdCBwYXRocyBhbmQgZG9ub3JzLmdldChyb2xlKToKICAgICAgICAgICAgIyBPbmUgZXhwb3N1cmUgd2l0aCBubyBwbGF0ZSBzb2x1dGlvbjogaXQgYm9ycm93cyBvbmUgZnJvbSBhIHNvbHZlZAogICAgICAgICAgICAjIGV4cG9zdXJlIG9mIHRoZSBzYW1lIGZpZWxkLCBhbmQgdGhlIHNvbHZlZCBjb3B5IGlzIHdoYXQgaXMgbWVhc3VyZWQuCiAgICAgICAgICAgIHBhdGggPSBmaXRzX2ZpbGVfQiBpZiByb2xlID09ICJCIiBlbHNlIGZpdHNfZmlsZV9WCiAgICAgICAgICAgIGRhdGFfYywgd2NzX2MsIF9icG0sIGluZm9fYyA9IGxvYWRfZnJhbWUocGF0aCwgZG9ub3JzW3JvbGVdKQogICAgICAgICAgICBvdXQgPSBvcy5wYXRoLmpvaW4ob3V0X2RpciwgZiJ7c2FmZX1fe3JvbGV9X3NvbHZlZC5maXRzIikKICAgICAgICAgICAgc2F2ZV9mcmFtZShvdXQsIGRhdGFfYywgd2NzX2MsIGluZm9fYywgaGlzdG9yeT0oCiAgICAgICAgICAgICAgICAiUGxhdGUgc29sdXRpb24gZml0dGVkIGFnYWluc3QgIiArIG9zLnBhdGguYmFzZW5hbWUoZG9ub3JzW3JvbGVdKSwpKQogICAgICAgICAgICBjb21iaW5lZF9ub3RlLmFwcGVuZChmIntyb2xlfTogcGxhdGUgc29sdXRpb24gYm9ycm93ZWQgZnJvbSAiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGYie29zLnBhdGguYmFzZW5hbWUoZG9ub3JzW3JvbGVdKX0iKQogICAgICAgICAgICBpZiByb2xlID09ICJCIjoKICAgICAgICAgICAgICAgIGZpdHNfZmlsZV9CID0gb3V0CiAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICBmaXRzX2ZpbGVfViA9IG91dAogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGlmIG5vdCBwYXRoczoKICAgICAgICAgICAgY29udGludWUKICAgICAgICBkYXRhX2MsIHdjc19jLCBfYnBtLCBpbmZvX2MgPSBjb21iaW5lX2V4cG9zdXJlcyhwYXRocywgZG9ub3JzLmdldChyb2xlKSwgcm9sZSkKICAgICAgICBvdXQgPSBvcy5wYXRoLmpvaW4ob3V0X2RpciwgZiJ7c2FmZX1fe3JvbGV9X2NvbWJpbmVkLmZpdHMiKQogICAgICAgIHNhdmVfZnJhbWUob3V0LCBkYXRhX2MsIHdjc19jLCBpbmZvX2MsCiAgICAgICAgICAgICAgICAgICBoaXN0b3J5PShmIk1lZGlhbiBvZiB7bGVuKHBhdGhzKX0gZXhwb3N1cmVzIiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICJDb21iaW5lZCBieSBCbHVlX0NsdXN0ZXJzX0Zyb21fSW1hZ2VzLnB5IikpCiAgICAgICAgY29tYmluZWRfbm90ZS5hcHBlbmQoZiJ7cm9sZX06IG1lZGlhbiBvZiB7bGVuKHBhdGhzKX0gZXhwb3N1cmVzIikKICAgICAgICBpZiByb2xlID09ICJCIjoKICAgICAgICAgICAgZml0c19maWxlX0IgPSBvdXQKICAgICAgICBlbHNlOgogICAgICAgICAgICBmaXRzX2ZpbGVfViA9IG91dAogICAgICAgIGRlbCBkYXRhX2MKCiAgICAjIEEgZmlsZSBuYW1lZCBvbiB0aGUgY29tbWFuZCBsaW5lIHRoYXQgaGFzIG5vIHBsYXRlIHNvbHV0aW9uIC0gdHlwaWNhbGx5IGEKICAgICMgbWVkaWFuIHdyaXR0ZW4gYnkgYW5vdGhlciBwcm9ncmFtLCB3aGljaCBrZXB0IG5vIGhlYWRlciAtIGJvcnJvd3Mgb25lIGZyb20KICAgICMgYSBzb2x2ZWQgZXhwb3N1cmUgaW4gaXRzIG93biBmb2xkZXIsIGV4YWN0bHkgYXMgaW4gdGhlIGZvbGRlciBtb2RlLgogICAgZm9yIHJvbGUgaW4gKCJCIiwgIlYiKToKICAgICAgICBwYXRoID0gZml0c19maWxlX0IgaWYgcm9sZSA9PSAiQiIgZWxzZSBmaXRzX2ZpbGVfVgogICAgICAgIGlmIGdyb3Vwcy5nZXQocm9sZSkgb3IgZG9ub3JzLmdldChyb2xlKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICBkb25vciA9IGRvbm9yX2Jlc2lkZShwYXRoLCByb2xlKQogICAgICAgIGlmIGRvbm9yIGlzIE5vbmU6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgcHJpbnQoZiJbaW5wdXRdIHtyb2xlfToge29zLnBhdGguYmFzZW5hbWUocGF0aCl9IGhhcyBubyBwbGF0ZSBzb2x1dGlvbjsgIgogICAgICAgICAgICAgIGYiZml0dGluZyBvbmUgYWdhaW5zdCB7b3MucGF0aC5iYXNlbmFtZShkb25vcil9IiwgZmx1c2g9VHJ1ZSkKICAgICAgICBkYXRhX2MsIHdjc19jLCBfYnBtLCBpbmZvX2MgPSBsb2FkX2ZyYW1lKHBhdGgsIGRvbm9yKQogICAgICAgIGlmIHdjc19jIGlzIE5vbmU6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgIyBUaGUgbnVtYmVyIG9mIGZyYW1lcyBjb21iaW5lZCBiZWxvbmdzIHRvIHRoaXMgZmlsZSwgbm90IHRvIHRoZSBkb25vcjsKICAgICAgICAjIGl0IGlzIGtub3duIG9ubHkgaWYgdGhpcyBmaWxlJ3Mgb3duIGhlYWRlciBzYXlzIGl0LgogICAgICAgIG5fY29tYiwgaG93ID0gY29tYmluZWRfY291bnQoZml0cy5nZXRoZWFkZXIocGF0aCkpCiAgICAgICAgaW5mb19jWyJuY29tYmluZSJdLCBpbmZvX2NbImNvbWJ0eXBlIl0gPSBuX2NvbWIsICgiIiBpZiBuX2NvbWIgPT0gMSBlbHNlIGhvdykKICAgICAgICBvdXQgPSBvcy5wYXRoLmpvaW4ob3V0X2RpciwgZiJ7c2FmZX1fe3JvbGV9X3NvbHZlZC5maXRzIikKICAgICAgICBzYXZlX2ZyYW1lKG91dCwgZGF0YV9jLCB3Y3NfYywgaW5mb19jLCBoaXN0b3J5PSgKICAgICAgICAgICAgIlBsYXRlIHNvbHV0aW9uIGZpdHRlZCBhZ2FpbnN0ICIgKyBvcy5wYXRoLmJhc2VuYW1lKGRvbm9yKSwpKQogICAgICAgIGNvbWJpbmVkX25vdGUuYXBwZW5kKGYie3JvbGV9OiBwbGF0ZSBzb2x1dGlvbiBib3Jyb3dlZCBmcm9tICIKICAgICAgICAgICAgICAgICAgICAgICAgICAgICBmIntvcy5wYXRoLmJhc2VuYW1lKGRvbm9yKX0iKQogICAgICAgIGlmIHJvbGUgPT0gIkIiOgogICAgICAgICAgICBmaXRzX2ZpbGVfQiA9IG91dAogICAgICAgIGVsc2U6CiAgICAgICAgICAgIGZpdHNfZmlsZV9WID0gb3V0CiAgICAgICAgZGVsIGRhdGFfYwoKICAgICMgLS0tLSB0aGUgcHVibGlzaGVkIG51bWJlcnMsIHNob3duIGFuZCBjb25maXJtZWQgYmVmb3JlIHVzZSAtLS0tLS0tLS0tLS0KICAgIHByaW50KCJMb29raW5nIHRoZSBnYWxheHkgdXAuLi4iLCBmbHVzaD1UcnVlKQogICAgZGlzdGFuY2UgPSBvZmZlcmVkKCJEaXN0YW5jZSAoTXBjKSIsIGxvb2t1cF9kaXN0YW5jZShvYmpfbmFtZSksCiAgICAgICAgICAgICAgICAgICAgICAgIkNvc21pY2Zsb3dzLTMsIFR1bGx5IGV0IGFsLiAyMDE2IiwgIiBNcGMiLCBBUkdTLmRpc3RhbmNlX21wYykKICAgIEVfQlYgPSBvZmZlcmVkKCJHYWxhY3RpYyBjb2xvdXIgZXhjZXNzIEUoQi1WKSIsIGxvb2t1cF9leHRpbmN0aW9uKG9ial9uYW1lKSwKICAgICAgICAgICAgICAgICAgICJTY2hsYWZseSAmIEZpbmtiZWluZXIgMjAxMSwgdmlhIElSU0EiLCAiIiwgQVJHUy5lYnYpCiAgICBpZiBkaXN0YW5jZSA8PSAwIG9yIEVfQlYgPCAwOgogICAgICAgIHN0b3AoInRoZSBkaXN0YW5jZSBtdXN0IGJlIHBvc2l0aXZlIGFuZCBFKEItVikgbm90IG5lZ2F0aXZlIikKICAgIEFfViA9IFJfViAqIEVfQlYKICAgIG11ID0gNS4wICogbnAubG9nMTAoZGlzdGFuY2UgKiAxLjBlNikgLSA1LjAKICAgIHByaW50KGYiVXNpbmcgZGlzdGFuY2Uge2Rpc3RhbmNlOi4yZn0gTXBjIChkaXN0YW5jZSBtb2R1bHVzIHttdTouMmZ9KSwiCiAgICAgICAgICBmIiBFKEItVikgPSB7RV9CVjouNGZ9LCBBX1YgPSB7QV9WOi4zZn0iLCBmbHVzaD1UcnVlKQogICAgZm9yIGxpbmUgaW4gY29tYmluZWRfbm90ZToKICAgICAgICBwcmludCgiW2lucHV0XSAiICsgbGluZSwgZmx1c2g9VHJ1ZSkKCiAgICAjIC0tLS0gZnJhbWVzIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCiAgICBwYXRoX0IsIHJhd19CLCBkYXRhX0IsIGhkcl9CID0gc3VidHJhY3RfYmFja2dyb3VuZF9hbmRfc2F2ZShmaXRzX2ZpbGVfQiwgb3V0X2RpcikKICAgIHBhdGhfViwgcmF3X1YsIGRhdGFfViwgaGRyX1YgPSBzdWJ0cmFjdF9iYWNrZ3JvdW5kX2FuZF9zYXZlKGZpdHNfZmlsZV9WLCBvdXRfZGlyKQogICAgc2hhcGVfViA9IGRhdGFfVi5zaGFwZQogICAgd2NzX1YgPSBXQ1MoaGRyX1YpLmNlbGVzdGlhbAogICAgaWYgbm90IHdjc19WLmhhc19jZWxlc3RpYWw6CiAgICAgICAgc3RvcCgidGhlIFYgZnJhbWUgaGFzIG5vIHBsYXRlIHNvbHV0aW9uOyBwbGF0ZS1zb2x2ZSBpdCBhbmQgcmVydW4iKQogICAgc2NhbGVfYXJjc2VjID0gMzYwMC4wICogZmxvYXQobnAuc3FydChhYnMobnAubGluYWxnLmRldCh3Y3NfVi5waXhlbF9zY2FsZV9tYXRyaXgpKSkpCgogICAgaW5mbyA9IHt9CiAgICBmb3IgYmFuZCwgcmF3LCBoZHIgaW4gKCgiQiIsIHJhd19CLCBoZHJfQiksICgiViIsIHJhd19WLCBoZHJfVikpOgogICAgICAgIHNhdCwgc2F0X3NyYyA9IHNhdHVyYXRpb25fbGV2ZWwocmF3LCBoZHIpCiAgICAgICAgZ2FpbiwgZ2Fpbl9zcmMgPSBkZXRlY3Rvcl9nYWluKGhkcikKICAgICAgICBybiwgcm5fc3JjID0gcmVhZF9ub2lzZShoZHIpCiAgICAgICAgaW5mb1tiYW5kXSA9IHsic2F0Ijogc2F0LCAiZ2FpbiI6IGdhaW4sICJybiI6IHJufQogICAgICAgIHByaW50KGYiICAge2JhbmR9OiBzYXR1cmF0aW9uIGxldmVsIHtzYXQ6LjBmfSBBRFUgKHtzYXRfc3JjfSk7IGdhaW4gIgogICAgICAgICAgICAgIGYie2dhaW46LjJmfSBlLS9BRFUgKHtnYWluX3NyY30pOyByZWFkIG5vaXNlIHtybjpnfSBlLSAoe3JuX3NyY30pIiwKICAgICAgICAgICAgICBmbHVzaD1UcnVlKQoKICAgICMgV2hlcmUgdGhlIGdhbGF4eSBpcyBhbmQgaG93IGZhciBpdCByZWFjaGVzLCBmb3IgdGhlIGRldGVjdGlvbiB3aW5kb3csIHRoZQogICAgIyBjYWxpYnJhdGlvbiBzdGFycyBhbmQgdGhlIHRyaW1taW5nIG9mIHRoZSBjYXRhbG9ndWUuCiAgICBjZW50cmVfcHggPSBnYWxheHlfY2VudHJlX3BpeGVsKG9ial9uYW1lLCB3Y3NfViwgc2hhcGVfVikKICAgIHIyNSA9IE5vbmUKICAgIGdlb20gPSBOb25lCiAgICB3aW5kb3cgPSBOb25lCiAgICB0cnk6CiAgICAgICAgZ2VvbSA9IGh5cGVybGVkYV9nZW9tZXRyeShvYmpfbmFtZSkKICAgICAgICBpZiBnZW9tIGFuZCBnZW9tLmdldCgiZDI1X2FyY21pbiIpOgogICAgICAgICAgICByMjUgPSAwLjUgKiBnZW9tWyJkMjVfYXJjbWluIl0gKiA2MC4wIC8gc2NhbGVfYXJjc2VjCiAgICAgICAgICAgIGlmIGNlbnRyZV9weCBpcyBub3QgTm9uZToKICAgICAgICAgICAgICAgIHdpbmRvdyA9IGdhbGF4eV93aW5kb3coc2hhcGVfViwgY2VudHJlX3B4LCByMjUsIG1hcmdpbj0xLjUpCiAgICAgICAgICAgICAgICBpZiB3aW5kb3cgaXMgTm9uZToKICAgICAgICAgICAgICAgICAgICBwcmludCgiW3dpbmRvd10gdGhlIGdhbGF4eSBmaWxscyB0aGlzIGZyYW1lIC0gd29ya2luZyBvbiBhbGwgb2YgaXQiLAogICAgICAgICAgICAgICAgICAgICAgICAgIGZsdXNoPVRydWUpCiAgICAgICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgICAgIHkwLCB5MSwgeDAsIHgxID0gd2luZG93CiAgICAgICAgICAgICAgICAgICAgcHJpbnQoZiJbd2luZG93XSB0aGUgZ2FsYXh5IG9jY3VwaWVzIHt4MSAtIHgwfSB4IHt5MSAtIHkwfSBvZiAiCiAgICAgICAgICAgICAgICAgICAgICAgICAgZiJ7c2hhcGVfVlsxXX0geCB7c2hhcGVfVlswXX0gcGl4ZWxzOyB0aGUgcmVzdCBpcyBza3kiLAogICAgICAgICAgICAgICAgICAgICAgICAgIGZsdXNoPVRydWUpCiAgICAgICAgZWxzZToKICAgICAgICAgICAgcHJpbnQoIlt3aW5kb3ddIG5vIGlzb3Bob3RhbCByYWRpdXMgYXZhaWxhYmxlIC0gd29ya2luZyBvbiB0aGUgd2hvbGUgZnJhbWUiLAogICAgICAgICAgICAgICAgICBmbHVzaD1UcnVlKQogICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBleGM6CiAgICAgICAgcHJpbnQoZiJbd2luZG93XSBjb3VsZCBub3Qgd29yayBvdXQgd2hlcmUgdGhlIGdhbGF4eSBlbmRzICh7ZXhjfSkgLSB3b3JraW5nICIKICAgICAgICAgICAgICBmIm9uIHRoZSB3aG9sZSBmcmFtZSIsIGZsdXNoPVRydWUpCgogICAgIyAtLS0tIGRldGVjdGlvbiAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCiAgICBwcmludCgiUmVnaXN0ZXJpbmcgQiBvbnRvIFYuLi4iLCBmbHVzaD1UcnVlKQogICAgYl90b192LCB2X3RvX2IsIHJlZ190ZXh0ID0gcmVnaXN0ZXJfQl9vbnRvX1YoZGF0YV9CLCBkYXRhX1YsIHBhdGhfQiwgcGF0aF9WKQogICAgd2luZG93X0IgPSBOb25lCiAgICBpZiB3aW5kb3cgaXMgbm90IE5vbmU6CiAgICAgICAgIyBUaGUgc2FtZSBwYXJ0IG9mIHRoZSBza3kgaW4gQidzIG93biBncmlkLCB3aXRoIGEgbWFyZ2luIGZvciByb3RhdGlvbi4KICAgICAgICB5MCwgeTEsIHgwLCB4MSA9IHdpbmRvdwogICAgICAgIGNvcm5lcnMgPSB2X3RvX2IobnAuYXJyYXkoW1t4MCwgeTBdLCBbeDEsIHkwXSwgW3gwLCB5MV0sIFt4MSwgeTFdXSwgZmxvYXQpKQogICAgICAgIGJ4MCwgYnkwID0gbnAuZmxvb3IoY29ybmVycy5taW4oYXhpcz0wKSkuYXN0eXBlKGludCkgLSAyMAogICAgICAgIGJ4MSwgYnkxID0gbnAuY2VpbChjb3JuZXJzLm1heChheGlzPTApKS5hc3R5cGUoaW50KSArIDIwCiAgICAgICAgd2luZG93X0IgPSAobWF4KGJ5MCwgMCksIG1pbihieTEsIGRhdGFfQi5zaGFwZVswXSksCiAgICAgICAgICAgICAgICAgICAgbWF4KGJ4MCwgMCksIG1pbihieDEsIGRhdGFfQi5zaGFwZVsxXSkpCiAgICB4QiwgeUIsIGZCLCBmd2htX0IgPSBkZXRlY3Rfc291cmNlcyhkYXRhX0IsIHJhd19CLCAiQiIsIGluZm9bIkIiXVsic2F0Il0sIHdpbmRvd19CKQogICAgeFYsIHlWLCBmViwgZndobV9WID0gZGV0ZWN0X3NvdXJjZXMoZGF0YV9WLCByYXdfViwgIlYiLCBpbmZvWyJWIl1bInNhdCJdLCB3aW5kb3cpCiAgICBmd2htID0gbWF4KGZ3aG1fQiwgZndobV9WKQoKICAgICMgT25lIGxpc3Qgb2Ygc291cmNlcyBvbiB0aGUgViBncmlkOiBldmVyeXRoaW5nIGZvdW5kIGluIFYsIGFuZCB3aGF0ZXZlciB3YXMKICAgICMgZm91bmQgaW4gQiB3aXRoIG5vdGhpbmcgaW4gViB3aXRoaW4gb25lIEZXSE0gLSBhIHZlcnkgYmx1ZSBrbm90IGNhbiBiZQogICAgIyBjbGVhciBpbiBCIGFuZCBsb3N0IGluIFYuIEVhY2ggaXMgdGhlbiBtZWFzdXJlZCBpbiBib3RoIGZyYW1lcyBhdCB0aGUKICAgICMgc2FtZSBwbGFjZSBvbiB0aGUgc2t5IHRocm91Z2ggdGhlIHNhbWUgYXBlcnR1cmUuIEJlZm9yZSwgZWFjaCBiYW5kIHdhcwogICAgIyBtZWFzdXJlZCB3aGVyZSBpdCBmb3VuZCB0aGUgc291cmNlLCB0aHJvdWdoIGFuIGFwZXJ0dXJlIHNpemVkIGZyb20gdGhhdAogICAgIyBzb3VyY2UncyB3aWR0aCBpbiB0aGF0IGJhbmQ6IHRoZSBCIGFuZCBWIGFwZXJ0dXJlcyBvZiBvbmUga25vdCBkaWZmZXJlZCwKICAgICMgYW5kIHNvIGl0cyBjb2xvdXIgZGVwZW5kZWQgb24gaXRzIHNpemUgYXMgbXVjaCBhcyBvbiBpdHMgbGlnaHQuCiAgICB4eUJfb25fViA9IGJfdG9fdihucC5jb2x1bW5fc3RhY2soW3hCLCB5Ql0pKSBpZiBsZW4oeEIpIGVsc2UgbnAuZW1wdHkoKDAsIDIpKQogICAgeHlWID0gbnAuY29sdW1uX3N0YWNrKFt4ViwgeVZdKSBpZiBsZW4oeFYpIGVsc2UgbnAuZW1wdHkoKDAsIDIpKQogICAgaWYgbGVuKHh5VikgYW5kIGxlbih4eUJfb25fVik6CiAgICAgICAgZCwgXyA9IGNLRFRyZWUoeHlWKS5xdWVyeSh4eUJfb25fViwgaz0xKQogICAgICAgIGJfb25seSA9IGQgPiBmd2htCiAgICBlbHNlOgogICAgICAgIGJfb25seSA9IG5wLm9uZXMobGVuKHh5Ql9vbl9WKSwgZHR5cGU9Ym9vbCkKICAgIHh5ID0gbnAudnN0YWNrKFt4eVYsIHh5Ql9vbl9WW2Jfb25seV1dKQogICAgZm91bmRfaW4gPSBucC5hcnJheShbIlYiXSAqIGxlbih4eVYpICsgWyJCIl0gKiBpbnQoYl9vbmx5LnN1bSgpKSwgZHR5cGU9IjxVMiIpCiAgICBpZiBsZW4oeHlWKSBhbmQgbGVuKHh5Ql9vbl9WKToKICAgICAgICBkViwgXyA9IGNLRFRyZWUoeHlCX29uX1YpLnF1ZXJ5KHh5Viwgaz0xKQogICAgICAgIGZvdW5kX2luWzpsZW4oeHlWKV1bZFYgPD0gZndobV0gPSAiQlYiCgogICAgcl9hcCA9IEFQRVJUVVJFX1NDQUxFICogZndobQogICAgcl9pbiwgcl9vdXQgPSBBTk5VTFVTX0lOTkVSX1NDQUxFICogcl9hcCwgQU5OVUxVU19PVVRFUl9TQ0FMRSAqIHJfYXAKICAgIHh5X2luX0IgPSB2X3RvX2IoeHkpIGlmIGxlbih4eSkgZWxzZSB4eQogICAgZWRnZSA9IHJfb3V0ICsgMgogICAgaW5zaWRlID0gKCh4eVs6LCAwXSA+IGVkZ2UpICYgKHh5WzosIDBdIDwgc2hhcGVfVlsxXSAtIGVkZ2UpCiAgICAgICAgICAgICAgJiAoeHlbOiwgMV0gPiBlZGdlKSAmICh4eVs6LCAxXSA8IHNoYXBlX1ZbMF0gLSBlZGdlKQogICAgICAgICAgICAgICYgKHh5X2luX0JbOiwgMF0gPiBlZGdlKSAmICh4eV9pbl9CWzosIDBdIDwgZGF0YV9CLnNoYXBlWzFdIC0gZWRnZSkKICAgICAgICAgICAgICAmICh4eV9pbl9CWzosIDFdID4gZWRnZSkgJiAoeHlfaW5fQls6LCAxXSA8IGRhdGFfQi5zaGFwZVswXSAtIGVkZ2UpKQogICAgeHksIHh5X2luX0IsIGZvdW5kX2luID0geHlbaW5zaWRlXSwgeHlfaW5fQltpbnNpZGVdLCBmb3VuZF9pbltpbnNpZGVdCiAgICBpZiBub3QgbGVuKHh5KToKICAgICAgICBzdG9wKCJubyBzb3VyY2VzIHdlcmUgZm91bmQgaW4gYm90aCBmcmFtZXMiKQogICAgcHJpbnQoZiJNZWFzdXJpbmcge2xlbih4eSk6LH0gc291cmNlcyBpbiBib3RoIGJhbmRzIHRocm91Z2ggb25lIGFwZXJ0dXJlIG9mICIKICAgICAgICAgIGYie3JfYXA6LjFmfSBweCAoe0FQRVJUVVJFX1NDQUxFOmd9IHggdGhlIEZXSE0gb2Yge2Z3aG06LjJmfSBweDsgc2t5IHJpbmcgIgogICAgICAgICAgZiJ7cl9pbjouMWZ9LXtyX291dDouMWZ9IHB4KSAgW2ZvdW5kIGluIGJvdGg6IHtpbnQoKGZvdW5kX2luID09ICdCVicpLnN1bSgpKTosfSwiCiAgICAgICAgICBmIiBWIG9ubHk6IHtpbnQoKGZvdW5kX2luID09ICdWJykuc3VtKCkpOix9LCBCIG9ubHk6ICIKICAgICAgICAgIGYie2ludCgoZm91bmRfaW4gPT0gJ0InKS5zdW0oKSk6LH1dIiwgZmx1c2g9VHJ1ZSkKCiAgICBmbHV4ViwgZXJyViwgcGVha1YsIHNreVYgPSBhcGVydHVyZV9waG90b21ldHJ5X2F0KAogICAgICAgIGRhdGFfViwgcmF3X1YsIHh5LCByX2FwLCByX2luLCByX291dCwgaW5mb1siViJdWyJnYWluIl0sIGluZm9bIlYiXVsicm4iXSkKICAgIGZsdXhCLCBlcnJCLCBwZWFrQiwgc2t5QiA9IGFwZXJ0dXJlX3Bob3RvbWV0cnlfYXQoCiAgICAgICAgZGF0YV9CLCByYXdfQiwgeHlfaW5fQiwgcl9hcCwgcl9pbiwgcl9vdXQsIGluZm9bIkIiXVsiZ2FpbiJdLCBpbmZvWyJCIl1bInJuIl0pCiAgICBtYWdWLCBtYWdWX2VyciA9IGluc3RydW1lbnRhbChmbHV4ViwgZXJyVikKICAgIG1hZ0IsIG1hZ0JfZXJyID0gaW5zdHJ1bWVudGFsKGZsdXhCLCBlcnJCKQogICAgd2lkdGhWID0gbnAuYXJyYXkoW3N0YXJfd2lkdGgoZGF0YV9WLCB4LCB5LCBGV0hNX1dJTkRPV19TSVpFLCBzKQogICAgICAgICAgICAgICAgICAgICAgIGZvciAoeCwgeSksIHMgaW4gemlwKHh5LCBza3lWKV0pCiAgICB3aWR0aEIgPSBucC5hcnJheShbc3Rhcl93aWR0aChkYXRhX0IsIHgsIHksIEZXSE1fV0lORE9XX1NJWkUsIHMpCiAgICAgICAgICAgICAgICAgICAgICAgZm9yICh4LCB5KSwgcyBpbiB6aXAoeHlfaW5fQiwgc2t5QildKQogICAgcmEsIGRlYyA9IHdjc19WLmFsbF9waXgyd29ybGQoeHlbOiwgMF0sIHh5WzosIDFdLCAwKQoKICAgIGRmID0gcGQuRGF0YUZyYW1lKHsKICAgICAgICAiWF9CIjogeHlfaW5fQls6LCAwXSwgIllfQiI6IHh5X2luX0JbOiwgMV0sICJGV0hNX0IiOiB3aWR0aEIsCiAgICAgICAgIkZsdXhfQiI6IGZsdXhCLCAiRmx1eF9CX2VyciI6IGVyckIsCiAgICAgICAgIlhfViI6IHh5WzosIDBdLCAiWV9WIjogeHlbOiwgMV0sICJGV0hNX1YiOiB3aWR0aFYsCiAgICAgICAgIkZsdXhfViI6IGZsdXhWLCAiRmx1eF9WX2VyciI6IGVyclYsCiAgICAgICAgIk1hZ19pbnN0X0IiOiBtYWdCLCAiTWFnX2luc3RfQl9lcnIiOiBtYWdCX2VyciwKICAgICAgICAiTWFnX2luc3RfViI6IG1hZ1YsICJNYWdfaW5zdF9WX2VyciI6IG1hZ1ZfZXJyLAogICAgICAgICJQZWFrX0IiOiBwZWFrQiwgIlBlYWtfViI6IHBlYWtWLAogICAgICAgICJTYXR1cmF0ZWQiOiAocGVha0IgPiBpbmZvWyJCIl1bInNhdCJdKSB8IChwZWFrViA+IGluZm9bIlYiXVsic2F0Il0pLAogICAgICAgICJGb3VuZF9pbiI6IGZvdW5kX2luLCAiQXBlcnR1cmVfUmFkaXVzIjogcl9hcCwKICAgIH0pCiAgICBuX2FsbCA9IGxlbihkZikKICAgIGRmID0gZGZbKGRmWyJGbHV4X0IiXSA+IDApICYgKGRmWyJGbHV4X1YiXSA+IDApXS5yZXNldF9pbmRleChkcm9wPVRydWUpCiAgICBwcmludChmIiAgIHtsZW4oZGYpOix9IG9mIHtuX2FsbDosfSBoYXZlIHBvc2l0aXZlIGZsdXggaW4gYm90aCBiYW5kczsgIgogICAgICAgICAgZiJ7aW50KGRmWydTYXR1cmF0ZWQnXS5zdW0oKSk6LH0gb2YgdGhlbSBzYXR1cmF0ZWQiLCBmbHVzaD1UcnVlKQogICAgaWYgbGVuKGRmKSA8IDI6CiAgICAgICAgc3RvcCgiZmV3ZXIgdGhhbiB0d28gc291cmNlcyBoYXZlIHBvc2l0aXZlIGZsdXggaW4gYm90aCBiYW5kcyIpCiAgICBuZWFyZXN0ID0gY0tEVHJlZShkZltbIlhfViIsICJZX1YiXV0udmFsdWVzKS5xdWVyeShkZltbIlhfViIsICJZX1YiXV0udmFsdWVzLCBrPTIpWzBdWzosIDFdCiAgICBkZlsiTmVhcmVzdF9weCJdID0gbmVhcmVzdAoKICAgICMgLS0tLSB0aGUgcGxhdGUgc29sdXRpb24sIGNoZWNrZWQsIHRoZW4gdGhlIHNreSBwb3NpdGlvbnMgLS0tLS0tLS0tLS0tLS0tCiAgICBicmlnaHQgPSBkZlt+ZGZbIlNhdHVyYXRlZCJdXS5zb3J0X3ZhbHVlcygiRmx1eF9WIiwgYXNjZW5kaW5nPUZhbHNlKQogICAgd2NzX1YsIHdjc19ub3RlID0gdmVyaWZ5X3BsYXRlX3NvbHV0aW9uKGJyaWdodFtbIlhfViIsICJZX1YiXV0udG9fbnVtcHkoZmxvYXQpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRmW1siWF9WIiwgIllfViJdXS50b19udW1weShmbG9hdCksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgd2NzX1YsIHNoYXBlX1YpCiAgICByYSwgZGVjID0gd2NzX1YuYWxsX3BpeDJ3b3JsZChkZlsiWF9WIl0udmFsdWVzLCBkZlsiWV9WIl0udmFsdWVzLCAwKQogICAgZGZbIlJBIl0sIGRmWyJERUMiXSA9IHJhLCBkZWMKICAgIGlmICJSRVBBSVJFRCIgaW4gd2NzX25vdGU6CiAgICAgICAgIyBUaGUgZ2FsYXh5J3MgY2VudHJlIHdhcyBwbGFjZWQgdGhyb3VnaCB0aGUgb2xkIHNvbHV0aW9uLgogICAgICAgIGNlbnRyZV9weCA9IGdhbGF4eV9jZW50cmVfcGl4ZWwob2JqX25hbWUsIHdjc19WLCBzaGFwZV9WKQogICAgICAgICMgQW5kIHRoZSBWIGZyYW1lIHdyaXR0ZW4gdG8gdGhlIG91dHB1dCBjYXJyaWVzIHRoZSByZXBhaXJlZCBzb2x1dGlvbiwgc28KICAgICAgICAjIHRoYXQgYSB0b29sIHJlYWRpbmcgcG9zaXRpb25zIHRocm91Z2ggaXQgbGF0ZXIgKHRoZSBjb21wYXJpc29uIHdpdGggdGhlCiAgICAgICAgIyBISUkgcmVnaW9ucykgcHV0cyB0aGVtIHdoZXJlIHRoZXkgYXJlLgogICAgICAgIHdpdGggZml0cy5vcGVuKHBhdGhfViwgbW9kZT0idXBkYXRlIikgYXMgaGR1bDoKICAgICAgICAgICAgaCA9IGhkdWxbMF0uaGVhZGVyCiAgICAgICAgICAgIGZvciBrZXkgaW4gbGlzdChoLmtleXMoKSk6CiAgICAgICAgICAgICAgICBpZiByZS5tYXRjaChyIl4oQ1RZUEV8Q1JWQUx8Q1JQSVh8Q0RFTFR8Q1VOSVR8Q0RcZF9cZHxQQ1xkX1xkfCIKICAgICAgICAgICAgICAgICAgICAgICAgICAgIHIiUFZcZF9cZCt8QV98Ql98QVBffEJQX3xMT05QT0xFfExBVFBPTEV8V0NTQVhFU3wiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICByIlJBREVTWVN8RVFVSU5PWCkiLCBrZXkpOgogICAgICAgICAgICAgICAgICAgIGgucmVtb3ZlKGtleSwgaWdub3JlX21pc3Npbmc9VHJ1ZSwgcmVtb3ZlX2FsbD1UcnVlKQogICAgICAgICAgICBoLnVwZGF0ZSh3Y3NfVi50b19oZWFkZXIocmVsYXg9VHJ1ZSkpCiAgICAgICAgICAgIGhbIkhJU1RPUlkiXSA9ICJQbGF0ZSBzb2x1dGlvbiByZXBhaXJlZCBhZ2FpbnN0IEdhaWEgRFIzIGJ5IEJsdWVfQ2x1c3RlcnMiCgogICAgY3N2X2ZpbGVuYW1lID0gb3MucGF0aC5qb2luKG91dF9kaXIsIGYie3NhZmV9X3Bob3RvbWV0cnlfcmVzdWx0cy5jc3YiKQogICAgZGYudG9fY3N2KGNzdl9maWxlbmFtZSwgaW5kZXg9RmFsc2UpCiAgICBwcmludChmIkluc3RydW1lbnRhbCBwaG90b21ldHJ5IHNhdmVkIHRvIHtjc3ZfZmlsZW5hbWV9IikKCiAgICAjIC0tLS0gY2FsaWJyYXRpb24gLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KICAgIGNhbCwgY2F0YWxvZ3VlLCBjYWxfc3RhdHMsIG1hdGNoZWQgPSBjYWxpYnJhdGUoCiAgICAgICAgZGYsIHdjc19WLCBzaGFwZV9WLCBBUkdTLmNhdGFsb2csIGNlbnRyZV9weCwgcjI1LCBvdXRfZGlyLCBzYWZlKQogICAgZGYgPSBhcHBseV9jYWxpYnJhdGlvbihkZiwgY2FsKQoKICAgICMgVGhlIGR1c3Qgb2Ygb3VyIG93biBnYWxheHkgdGFrZW4gb3V0LiBBX1YgZGltcyBWOyBCIGlzIGRpbW1lZCBieQogICAgIyBBX1YgKyBFKEItViksIHdoaWNoIGlzIHdoYXQgdGhlIGNvbG91ciBleGNlc3MgbWVhbnMuCiAgICBkZlsiTWFnX0IiXSA9IGRmWyJNYWdfQl9vYnMiXSAtIChBX1YgKyBFX0JWKQogICAgZGZbIk1hZ19WIl0gPSBkZlsiTWFnX1Zfb2JzIl0gLSBBX1YKICAgIGRmWyJNX1YiXSA9IGRmWyJNYWdfViJdIC0gbXUKICAgIGRmWyJaUF9WX3VzZWQiXSwgZGZbImtfVl91c2VkIl0gPSBjYWxbIlpQX1YiXSwgY2FsWyJrX1YiXQogICAgZGZbIkNvbG91cl9hX3VzZWQiXSwgZGZbIkNvbG91cl9jX3VzZWQiXSA9IGNhbFsiYSJdLCBjYWxbImMiXQogICAgZGZbIk5fcmVmX3VzZWQiXSA9IGNhbFsiTl91c2VkIl0KICAgIGRmWyJDYWxpYnJhdGlvbl9jYXRhbG9ndWUiXSA9IGNhdGFsb2d1ZQogICAgZGZbIkFfVl91c2VkIl0sIGRmWyJFX0JWX3VzZWQiXSA9IEFfViwgRV9CVgogICAgZGZbIkRpc3RhbmNlX01wY191c2VkIl0sIGRmWyJEaXN0YW5jZV9tb2R1bHVzX3VzZWQiXSA9IGRpc3RhbmNlLCBtdQogICAgZGZbIkZXSE1fbWVhc3VyZWRfQiJdLCBkZlsiRldITV9tZWFzdXJlZF9WIl0gPSBmd2htX0IsIGZ3aG1fVgoKICAgIGNhbGliX2ZpbGUgPSBvcy5wYXRoLmpvaW4ob3V0X2RpciwgZiJ7c2FmZX1fY2FsaWJyYXRlZF9waG90b21ldHJ5LmNzdiIpCiAgICBkZi50b19jc3YoY2FsaWJfZmlsZSwgaW5kZXg9RmFsc2UpCiAgICBwcmludChmIkNhbGlicmF0ZWQgcGhvdG9tZXRyeSBzYXZlZCB0byB7Y2FsaWJfZmlsZX0iKQoKICAgIHNvdXJjZV9tYXAoZGF0YV9WLCB4eVs6LCAwXSwgeHlbOiwgMV0sIGYie29ial9uYW1lfSAtIFYgYmFuZCB3aXRoIG1lYXN1cmVkIHNvdXJjZXMiLAogICAgICAgICAgICAgICBvcy5wYXRoLmpvaW4ob3V0X2RpciwgZiJ7c2FmZX1fVl9zb3VyY2VzLnBuZyIpKQogICAgc291cmNlX21hcChkYXRhX0IsIHh5X2luX0JbOiwgMF0sIHh5X2luX0JbOiwgMV0sCiAgICAgICAgICAgICAgIGYie29ial9uYW1lfSAtIEIgYmFuZCB3aXRoIG1lYXN1cmVkIHNvdXJjZXMiLAogICAgICAgICAgICAgICBvcy5wYXRoLmpvaW4ob3V0X2RpciwgZiJ7c2FmZX1fQl9zb3VyY2VzLnBuZyIpLCBjb2xvdXI9ImJsdWUiKQoKICAgICMgLS0tLSBmb3JlZ3JvdW5kIHN0YXJzIG91dCAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KICAgICMgRXZlcnkgc291cmNlIHRoYXQgaXMgYSBjYXRhbG9ndWUgc3RhciAobWF0Y2hlZCB3aXRoaW4gQ0FMSUJfTUFUQ0hfUkFESVVTKSwKICAgICMgYW5kIGV2ZXJ5IHNvdXJjZSBvbiBhIEdhaWEgc3RhciB3aXRoIGEgc2lnbmlmaWNhbnQgcGFyYWxsYXggb3IgcHJvcGVyCiAgICAjIG1vdGlvbiwgaXMgaW4gZnJvbnQgb2YgdGhlIGdhbGF4eS4KICAgIGlzX3N0YXIgPSBucC56ZXJvcyhsZW4oZGYpLCBkdHlwZT1ib29sKQogICAgaWYgbGVuKG1hdGNoZWQpOgogICAgICAgIGlzX3N0YXJbbWF0Y2hlZFsicm93Il0uYXN0eXBlKGludCkudmFsdWVzXSA9IFRydWUKICAgIG5fY2F0X3N0YXJzID0gaW50KGlzX3N0YXIuc3VtKCkpCiAgICBuX2dhaWEgPSAwCiAgICB0cnk6CiAgICAgICAgZmcsIF9mZ2RpYWcgPSBnYWlhX2ZvcmVncm91bmRfc3RhcnMod2NzX1YsIHNoYXBlX1YpCiAgICAgICAgaWYgZmcgaXMgbm90IE5vbmUgYW5kIGxlbihmZyk6CiAgICAgICAgICAgIGQsIF8gPSBjS0RUcmVlKG5wLmFzYXJyYXkoZmcsIGZsb2F0KSkucXVlcnkoZGZbWyJYX1YiLCAiWV9WIl1dLnZhbHVlcywgaz0xKQogICAgICAgICAgICBvbl9mZyA9IChkIDw9IEZPUkVHUk9VTkRfVE9MKSAmIH5pc19zdGFyCiAgICAgICAgICAgIG5fZ2FpYSA9IGludChvbl9mZy5zdW0oKSkKICAgICAgICAgICAgaXNfc3RhciB8PSBvbl9mZwogICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBleGM6CiAgICAgICAgcHJpbnQoZiIgICAgY291bGQgbm90IGNoZWNrIGFnYWluc3QgR2FpYSAoe2V4Y30pOyBmb3JlZ3JvdW5kIHN0YXJzIG90aGVyIHRoYW4gIgogICAgICAgICAgICAgIGYidGhlIGNhdGFsb2d1ZSBzdGFycyByZW1haW4gaW4gdGhlIGNhdGFsb2d1ZSIsIGZsdXNoPVRydWUpCiAgICBkZl9jbGVhbiA9IGRmW35pc19zdGFyXS5yZXNldF9pbmRleChkcm9wPVRydWUpCiAgICBwcmludChmIiAgICB7bl9jYXRfc3RhcnN9IHNvdXJjZXMgYXJlIGNhdGFsb2d1ZSBzdGFycyBhbmQge25fZ2FpYX0gbW9yZSBzaXQgb24gYSAiCiAgICAgICAgICBmIkdhaWEgZm9yZWdyb3VuZCBzdGFyIChwYXJhbGxheCBvciBwcm9wZXIgbW90aW9uIHNpZ25pZmljYW50KTsgIgogICAgICAgICAgZiJ7bGVuKGRmX2NsZWFuKX0gbGVmdCIsIGZsdXNoPVRydWUpCiAgICBuX291dHNpZGUgPSAwCiAgICBpZiByMjUgYW5kIGNlbnRyZV9weCBpcyBub3QgTm9uZToKICAgICAgICBkID0gbnAuaHlwb3QoZGZfY2xlYW5bIlhfViJdIC0gY2VudHJlX3B4WzBdLCBkZl9jbGVhblsiWV9WIl0gLSBjZW50cmVfcHhbMV0pCiAgICAgICAgbl9vdXRzaWRlID0gaW50KChkID4gcjI1KS5zdW0oKSkKICAgICAgICBkZl9jbGVhbiA9IGRmX2NsZWFuWyhkIDw9IHIyNSkudmFsdWVzXS5yZXNldF9pbmRleChkcm9wPVRydWUpCiAgICAgICAgcHJpbnQoZiIgICAge25fb3V0c2lkZX0gc291cmNlcyBsaWUgYmV5b25kIFIyNSAoe3IyNTouMGZ9IHB4KSwgb3V0c2lkZSB0aGUgIgogICAgICAgICAgICAgIGYiZ2FsYXh5LCBhbmQgYXJlIGRyb3BwZWQ7IHtsZW4oZGZfY2xlYW4pfSBsZWZ0IiwgZmx1c2g9VHJ1ZSkKICAgIGVsc2U6CiAgICAgICAgcHJpbnQoIiAgICBubyBnYWxheHkgcmFkaXVzIGF2YWlsYWJsZTsgdGhlIGNhdGFsb2d1ZSBpcyBub3QgdHJpbW1lZCB0byB0aGUgIgogICAgICAgICAgICAgICJnYWxheHkiLCBmbHVzaD1UcnVlKQoKICAgIG91dF9maWxlID0gY2FsaWJfZmlsZS5yZXBsYWNlKCJfY2FsaWJyYXRlZF9waG90b21ldHJ5LmNzdiIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAiX2NhbGlicmF0ZWRfcGhvdG9tZXRyeV9ub19zdGFycy5jc3YiKQogICAgZGZfY2xlYW4udG9fY3N2KG91dF9maWxlLCBpbmRleD1GYWxzZSkKICAgIHByaW50KGYiQ2xlYW5lZCBmaWxlIHNhdmVkIHRvIHtvdXRfZmlsZX0iKQogICAgc291cmNlX21hcChkYXRhX1YsIGRmX2NsZWFuWyJYX1YiXSwgZGZfY2xlYW5bIllfViJdLAogICAgICAgICAgICAgICBmIntvYmpfbmFtZX0gLSBWIGJhbmQgd2l0aCBkZXRlY3RlZCBzb3VyY2VzIChubyBzdGFycykiLAogICAgICAgICAgICAgICBvcy5wYXRoLmpvaW4ob3V0X2RpciwgZiJ7c2FmZX1fVl9zb3VyY2VzX25vX3N0YXJzLnBuZyIpLAogICAgICAgICAgICAgICBsYWJlbD0iQ2xlYW5lZCBzb3VyY2VzIChubyBzdGFycykiKQoKICAgIG1ha2VfY21kKGRmLCBjYWxpYl9maWxlLCAiKGFsbCBzb3VyY2VzKSIsIG9ial9uYW1lLCBtdSkKICAgIG1ha2VfY21kKGRmX2NsZWFuLCBvdXRfZmlsZSwgIihubyBzdGFycykiLCBvYmpfbmFtZSwgbXUsCiAgICAgICAgICAgICB3aW5kb3c9KENNRF9DT0xPUl9NSU4sIENNRF9DT0xPUl9NQVgpKQoKICAgICMgLS0tLSB0aGUgYmx1ZSBrbm90cyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQogICAgYnYwID0gZGZfY2xlYW5bIk1hZ19CIl0gLSBkZl9jbGVhblsiTWFnX1YiXQogICAgaW5fd2luZG93ID0gKGJ2MCA+PSBDTURfQ09MT1JfTUlOKSAmIChidjAgPD0gQ01EX0NPTE9SX01BWCkKICAgIHByZWNpc2UgPSBkZl9jbGVhblsiQlZfZXJyIl0gPD0gQVJHUy5tYXhfYnZfZXJyCiAgICB1bnNhdCA9IH5kZl9jbGVhblsiU2F0dXJhdGVkIl0uYXN0eXBlKGJvb2wpCiAgICBkZl9jb2xvcl9maWx0ZXJlZCA9IGRmX2NsZWFuW2luX3dpbmRvdyAmIHByZWNpc2UgJiB1bnNhdF0uY29weSgpCiAgICBkZl9jb2xvcl9maWx0ZXJlZFsiQi1WIl0gPSBidjBbaW5fd2luZG93ICYgcHJlY2lzZSAmIHVuc2F0XQogICAgZGZfY29sb3JfZmlsdGVyZWQgPSBkZl9jb2xvcl9maWx0ZXJlZC5yZXNldF9pbmRleChkcm9wPVRydWUpCiAgICBwcmludChmIkJsdWUga25vdHM6IHtsZW4oZGZfY29sb3JfZmlsdGVyZWQpfSBzb3VyY2VzIHdpdGgge0NNRF9DT0xPUl9NSU59IDw9ICIKICAgICAgICAgIGYiKEItVikwIDw9IHtDTURfQ09MT1JfTUFYfSBhbmQgYSBCLVYgZXJyb3Igb2YgYXQgbW9zdCB7QVJHUy5tYXhfYnZfZXJyfSBtYWciCiAgICAgICAgICBmIiAgKHtpbnQoKGluX3dpbmRvdyAmIH5wcmVjaXNlKS5zdW0oKSl9IG1vcmUgaW4gdGhlIGNvbG91ciB3aW5kb3cgd2VyZSB0b28gIgogICAgICAgICAgZiJub2lzeSB0byBjbGFzc2lmeSwge2ludCgoaW5fd2luZG93ICYgfnVuc2F0KS5zdW0oKSl9IHNhdHVyYXRlZCkiLCBmbHVzaD1UcnVlKQogICAgb3V0X2Nzdl9jb2xvciA9IG9zLnBhdGguam9pbihvdXRfZGlyLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBmIntzYWZlfV9jYWxpYnJhdGVkX3Bob3RvbWV0cnlfbm9fc3RhcnNfY29sb3JfZmlsdGVyZWQuY3N2IikKICAgIGRmX2NvbG9yX2ZpbHRlcmVkLnRvX2NzdihvdXRfY3N2X2NvbG9yLCBpbmRleD1GYWxzZSkKICAgIHByaW50KGYiQ29sb3ItZmlsdGVyZWQgcGhvdG9tZXRyeSBzYXZlZCB0byB7b3V0X2Nzdl9jb2xvcn0iKQogICAgc291cmNlX21hcChkYXRhX1YsIGRmX2NvbG9yX2ZpbHRlcmVkWyJYX1YiXSwgZGZfY29sb3JfZmlsdGVyZWRbIllfViJdLAogICAgICAgICAgICAgICBmIntvYmpfbmFtZX0gLSBWIGJhbmQgd2l0aCB0aGUgYmx1ZSBrbm90cyIsCiAgICAgICAgICAgICAgIG9zLnBhdGguam9pbihvdXRfZGlyLCBmIntzYWZlfV9WX3NvdXJjZXNfY29sb3JfZmlsdGVyZWQucG5nIiksCiAgICAgICAgICAgICAgIGxhYmVsPWYiQmx1ZSBrbm90cyAoe0NNRF9DT0xPUl9NSU59IDw9IChCLVYpMCA8PSB7Q01EX0NPTE9SX01BWH0pIikKICAgIG1ha2VfY21kKGRmX2NvbG9yX2ZpbHRlcmVkLCBvdXRfY3N2X2NvbG9yLCAiKGJsdWUga25vdHMpIiwgb2JqX25hbWUsIG11LAogICAgICAgICAgICAgd2luZG93PShDTURfQ09MT1JfTUlOLCBDTURfQ09MT1JfTUFYKSkKCiAgICB3cml0ZV9zdW1tYXJ5KG9zLnBhdGguam9pbihvdXRfZGlyLCAicnVuX3N1bW1hcnkudHh0IiksIGxvY2FscygpKQoKICAgIGlmIGRmX2NvbG9yX2ZpbHRlcmVkLmVtcHR5OgogICAgICAgIHByaW50KCJObyBjb2xvci1maWx0ZXJlZCBzb3VyY2VzIGF2YWlsYWJsZSB0byBjb21wdXRlIHJhZGlhbCBkaXN0YW5jZXMuIikKICAgICAgICByZXR1cm4gMAoKICAgICMgLS0tLSByYWRpYWwgcHJvZmlsZSAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQogICAgaWYgY2VudHJlX3B4IGlzIE5vbmU6CiAgICAgICAgY2VudHJlX3B4ID0gKHNoYXBlX1ZbMV0gLyAyLjAsIHNoYXBlX1ZbMF0gLyAyLjApCiAgICAgICAgcHJpbnQoIltwcm9maWxlXSB0aGUgZ2FsYXh5J3MgY2VudHJlIGlzIHVua25vd24gLSB1c2luZyB0aGUgZnJhbWUncyBjZW50cmUiKQogICAgeF9jZW50ZXIsIHlfY2VudGVyID0gY2VudHJlX3B4CiAgICBkZl9jb2xvcl9maWx0ZXJlZFsiUmFkaWFsX0Rpc3RhbmNlX3B4Il0gPSBucC5oeXBvdCgKICAgICAgICBkZl9jb2xvcl9maWx0ZXJlZFsiWF9WIl0gLSB4X2NlbnRlciwgZGZfY29sb3JfZmlsdGVyZWRbIllfViJdIC0geV9jZW50ZXIpCiAgICBweF90b19wYyA9IG5wLmRlZzJyYWQoc2NhbGVfYXJjc2VjIC8gMzYwMC4wKSAqIGRpc3RhbmNlICogMS4wZTYKICAgIGRmX2NvbG9yX2ZpbHRlcmVkWyJSYWRpYWxfRGlzdGFuY2VfcGMiXSA9IGRmX2NvbG9yX2ZpbHRlcmVkWyJSYWRpYWxfRGlzdGFuY2VfcHgiXSAqIHB4X3RvX3BjCiAgICBkZl9jb2xvcl9maWx0ZXJlZCA9IGRmX2NvbG9yX2ZpbHRlcmVkLnNvcnRfdmFsdWVzKCJSYWRpYWxfRGlzdGFuY2VfcHgiKS5yZXNldF9pbmRleChkcm9wPVRydWUpCiAgICBvdXRfY3N2X2NvbG9yX3JhZCA9IG9zLnBhdGguam9pbigKICAgICAgICBvdXRfZGlyLCBmIntzYWZlfV9jYWxpYnJhdGVkX3Bob3RvbWV0cnlfbm9fc3RhcnNfY29sb3JfZmlsdGVyZWRfd2l0aF9yYWRpdXMuY3N2IikKICAgIGRmX2NvbG9yX2ZpbHRlcmVkLnRvX2NzdihvdXRfY3N2X2NvbG9yX3JhZCwgaW5kZXg9RmFsc2UpCiAgICBwcmludChmIkNvbG9yLWZpbHRlcmVkIHBob3RvbWV0cnkgd2l0aCByYWRpYWwgZGlzdGFuY2VzIHNhdmVkIHRvIHtvdXRfY3N2X2NvbG9yX3JhZH0iKQoKICAgIGZvciB1bml0LCBzY2FsZV91LCBjb2xvdXIgaW4gKCgicHgiLCAxLjAsICJibHVlIiksICgicGMiLCBweF90b19wYywgImdyZWVuIikpOgogICAgICAgIHJhZGlpID0gZGZfY29sb3JfZmlsdGVyZWRbIlJhZGlhbF9EaXN0YW5jZV9weCJdICogc2NhbGVfdQogICAgICAgIHdpZHRoID0gU3RlcF9zaXplICogc2NhbGVfdQogICAgICAgIGVkZ2VzID0gbnAuYXJhbmdlKDAsIHJhZGlpLm1heCgpICsgd2lkdGgsIHdpZHRoKQogICAgICAgIGNvdW50cywgZWRnZXMgPSBucC5oaXN0b2dyYW0ocmFkaWksIGJpbnM9ZWRnZXMpCiAgICAgICAgYXJlYXMgPSBucC5waSAqIChlZGdlc1sxOl0gKiogMiAtIGVkZ2VzWzotMV0gKiogMikKICAgICAgICBkZW5zID0gY291bnRzIC8gYXJlYXMKICAgICAgICBzdWZmaXggPSAiIiBpZiB1bml0ID09ICJweCIgZWxzZSAiX3BjIgogICAgICAgIHBkLkRhdGFGcmFtZSh7ZiJSX2lubmVyX3t1bml0fSI6IGVkZ2VzWzotMV0sIGYiUl9vdXRlcl97dW5pdH0iOiBlZGdlc1sxOl0sCiAgICAgICAgICAgICAgICAgICAgICBmIlJfY2VudGVyX3t1bml0fSI6IDAuNSAqIChlZGdlc1sxOl0gKyBlZGdlc1s6LTFdKSwKICAgICAgICAgICAgICAgICAgICAgICJOX3NvdXJjZXMiOiBjb3VudHMsIGYiQW5udWx1c19hcmVhX3t1bml0fTIiOiBhcmVhcywKICAgICAgICAgICAgICAgICAgICAgIGYiRGVuc2l0eV9wZXJfe3VuaXR9MiI6IGRlbnN9KS50b19jc3YoCiAgICAgICAgICAgIG9zLnBhdGguam9pbihvdXRfZGlyLCBmIntzYWZlfV9yYWRpYWxfZGVuc2l0eV9wcm9maWxle3N1ZmZpeH0uY3N2IiksIGluZGV4PUZhbHNlKQogICAgICAgIHN0ZXBfciA9IG5wLmNvbHVtbl9zdGFjayhbZWRnZXNbOi0xXSwgZWRnZXNbMTpdXSkucmF2ZWwoKQogICAgICAgIHN0ZXBfZCA9IG5wLnJlcGVhdChkZW5zLCAyKQogICAgICAgIHN0ZXBfbiA9IG5wLnJlcGVhdChjb3VudHMsIDIpCiAgICAgICAgcGQuRGF0YUZyYW1lKHtmIlJfc3RlcF97dW5pdH0iOiBzdGVwX3IsIGYiRGVuc2l0eV9zdGVwX3Blcl97dW5pdH0yIjogc3RlcF9kLAogICAgICAgICAgICAgICAgICAgICAgIk5fc291cmNlc19zdGVwIjogc3RlcF9ufSkudG9fY3N2KAogICAgICAgICAgICBvcy5wYXRoLmpvaW4ob3V0X2RpciwgZiJ7c2FmZX1fcmFkaWFsX2RlbnNpdHlfcHJvZmlsZV9zdGVwe3N1ZmZpeH0uY3N2IiksCiAgICAgICAgICAgIGluZGV4PUZhbHNlKQogICAgICAgIHBsdC5maWd1cmUoZmlnc2l6ZT0oOCwgNikpCiAgICAgICAgcGx0LnN0ZXAoc3RlcF9yLCBzdGVwX2QsIHdoZXJlPSJwb3N0IiwgY29sb3I9Y29sb3VyLCBsaW5ld2lkdGg9MikKICAgICAgICBwbHQueGxhYmVsKCJSYWRpYWwgZGlzdGFuY2UgW3BpeGVsc10iIGlmIHVuaXQgPT0gInB4IiBlbHNlICJSYWRpYWwgZGlzdGFuY2UgW3BjXSIpCiAgICAgICAgcGx0LnlsYWJlbCgiU291cmNlIGRlbnNpdHkgWzEvcGl4ZWzCsl0iIGlmIHVuaXQgPT0gInB4IiBlbHNlICJTb3VyY2UgZGVuc2l0eSBbMS9wY8KyXSIpCiAgICAgICAgcGx0LnRpdGxlKGYiUmFkaWFsIGRlbnNpdHkgc3RlcCBwcm9maWxlIG9mIHtvYmpfbmFtZX0gIgogICAgICAgICAgICAgICAgICBmIih7J3BpeGVscycgaWYgdW5pdCA9PSAncHgnIGVsc2UgJ3BjJ30pIikKICAgICAgICBwbmcgPSBvcy5wYXRoLmpvaW4ob3V0X2RpciwgZiJ7c2FmZX1fcmFkaWFsX2RlbnNpdHlfcHJvZmlsZV9zdGVwe3N1ZmZpeH0ucG5nIikKICAgICAgICBwbHQuc2F2ZWZpZyhwbmcsIGRwaT0xNTApCiAgICAgICAgcGx0LmNsb3NlKCkKICAgICAgICBwcmludChmIlJhZGlhbCBkZW5zaXR5IHN0ZXAgcHJvZmlsZSAoe3VuaXR9KSBzYXZlZCB0byB7cG5nfSIpCgogICAgIyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQogICAgIyBTdGFnZSA2IC0gZGVwcm9qZWN0aW9uIGFuZCB0aGUgc2hhcmVkIHJhZGlhbCBhbmFseXNpcwogICAgIyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQogICAgIwogICAgIyBBIHNwaXJhbCBnYWxheHkgaXMgYSBmbGF0IGRpc2sgc2VlbiBhdCBhbiBhbmdsZSwgc28gYSBkaXN0YW5jZSBtZWFzdXJlZCBpbgogICAgIyB0aGUgaW1hZ2UgaXMgbm90IHRoZSBkaXN0YW5jZSBpbnNpZGUgdGhlIGdhbGF4eS4gVGhpcyBzdGFnZSBzdHJldGNoZXMgdGhlCiAgICAjIHNxdWFzaGVkIGRpcmVjdGlvbiBiYWNrIG91dCBhbmQgcmVjb21wdXRlcyB0aGUgcHJvZmlsZSBzZXZlcmFsIHdheXMgLQogICAgIyB1bmNvcnJlY3RlZCwgZnJvbSB0aGUgSHlwZXJMRURBIGNhdGFsb2d1ZSwgYW5kIGZyb20gdGhlIGtub3RzIHRoZW1zZWx2ZXMgLQogICAgIyBiZWNhdXNlIG5vIGRlcHJvamVjdGlvbiBtZXRob2QgaGFzIGJlZW4gZm91bmQgcmVsaWFibHkgYmV0dGVyIHRoYW4gdGhlCiAgICAjIG90aGVycy4gVGhlIHNwcmVhZCBiZXR3ZWVuIHRoZSBjdXJ2ZXMgaXMgdGhlIHVuY2VydGFpbnR5IG9uIHRoZSByYWRpaS4gSXQKICAgICMgYWxzbyB1c2VzIHRoZSByaW5nIGFyZWEgYWN0dWFsbHkgY292ZXJlZCBieSB0aGUgZnJhbWUuCiAgICB0cnk6CiAgICAgICAgbm9ydGgsIG1pcnJvcmVkID0gaW1hZ2Vfb3JpZW50YXRpb24od2NzX1YpCiAgICAgICAgcHJpbnQoZiJbZGVwcm9qZWN0XSBvcmllbnRhdGlvbiBmcm9tIFdDUzogbm9ydGgge25vcnRoOi4xZn0gZGVnIGZyb20gK3gsICIKICAgICAgICAgICAgICBmInsnbWlycm9yZWQnIGlmIG1pcnJvcmVkIGVsc2UgJ25vdCBtaXJyb3JlZCd9IikKICAgICAgICBwcmludCgiIikKICAgICAgICBwcmludCgiRGVwcm9qZWN0aW5nIGFuZCByZWJ1aWxkaW5nIHRoZSByYWRpYWwgcHJvZmlsZS4uLiIpCiAgICAgICAgcnVuX3Byb2ZpbGVzKAogICAgICAgICAgICBkZl9jb2xvcl9maWx0ZXJlZFsiWF9WIl0udG9fbnVtcHkoZHR5cGU9ZmxvYXQpLAogICAgICAgICAgICBkZl9jb2xvcl9maWx0ZXJlZFsiWV9WIl0udG9fbnVtcHkoZHR5cGU9ZmxvYXQpLAogICAgICAgICAgICAoeF9jZW50ZXIsIHlfY2VudGVyKSwgKDAuMCwgZmxvYXQoc2hhcGVfVlsxXSksIDAuMCwgZmxvYXQoc2hhcGVfVlswXSkpLAogICAgICAgICAgICBvcy5wYXRoLmpvaW4ob3V0X2RpciwgZiJ7c2FmZX1fZGVwcm9qZWN0ZWQiKSwKICAgICAgICAgICAgZiJCbHVlIGtub3RzIC0ge29ial9uYW1lfSIsIGdhbGF4eT1vYmpfbmFtZSwKICAgICAgICAgICAgbm9ydGhfYW5nbGU9bm9ydGgsIG1pcnJvcmVkPW1pcnJvcmVkLCBzZWN0b3JfbWV0aG9kPVRydWUsCiAgICAgICAgICAgIHJpbmdzPTIwLCBzY2FsZT1weF90b19wYywgdW5pdD0icGMiKQogICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBleGM6CiAgICAgICAgcHJpbnQoZiJbZGVwcm9qZWN0XSB0aGUgZGVwcm9qZWN0aW9uIHN0YWdlIGZhaWxlZCAoe3R5cGUoZXhjKS5fX25hbWVfX306ICIKICAgICAgICAgICAgICBmIntleGN9KS4gRXZlcnl0aGluZyBhYm92ZSB3YXMgc3RpbGwgd3JpdHRlbi4iKQogICAgcmV0dXJuIDAKCgpkZWYgd3JpdGVfc3VtbWFyeShwYXRoLCB2KToKICAgICIiIkV2ZXJ5IG51bWJlciB0aGlzIHJ1biB1c2VkIGFuZCBmb3VuZCwgaW4gb25lIGZpbGUgYmVzaWRlIGl0cyBvdXRwdXQuIiIiCiAgICBjYWwsIHN0ID0gdlsiY2FsIl0sIHZbImNhbF9zdGF0cyJdCiAgICBsaW5lcyA9IFsKICAgICAgICBmIkJsdWVfQ2x1c3RlcnNfRnJvbV9JbWFnZXMucHkgcnVuIHN1bW1hcnkiLAogICAgICAgIGYiZ2FsYXh5ICAgICAgICAgICAgICB7dlsnb2JqX25hbWUnXX0iLAogICAgICAgIGYiQiBmcmFtZSAgICAgICAgICAgICB7dlsnZml0c19maWxlX0InXX0iLAogICAgICAgIGYiViBmcmFtZSAgICAgICAgICAgICB7dlsnZml0c19maWxlX1YnXX0iLAogICAgXSArIFtmImlucHV0ICAgICAgICAgICAgICAge259IiBmb3IgbiBpbiB2WyJjb21iaW5lZF9ub3RlIl1dICsgWwogICAgICAgIGYiZGlzdGFuY2UgICAgICAgICAgICB7dlsnZGlzdGFuY2UnXTouM2Z9IE1wYyAoZGlzdGFuY2UgbW9kdWx1cyB7dlsnbXUnXTouM2Z9KSIsCiAgICAgICAgZiJFKEItVikgICAgICAgICAgICAgIHt2WydFX0JWJ106LjRmfSAgIEFfViA9IHt2WydBX1YnXTouM2Z9IiwKICAgICAgICBmInBsYXRlIHNvbHV0aW9uICAgICAge3ZbJ3djc19ub3RlJ119IiwKICAgICAgICBmInJlZ2lzdHJhdGlvbiAgICAgICAgQiBvbnRvIFYge3ZbJ3JlZ190ZXh0J119IiwKICAgICAgICBmInN0YXIgRldITSAgICAgICAgICAgQiB7dlsnZndobV9CJ106LjJmfSBweCwgViB7dlsnZndobV9WJ106LjJmfSBweCIsCiAgICAgICAgZiJhcGVydHVyZSAgICAgICAgICAgIHt2WydyX2FwJ106LjJmfSBweCByYWRpdXMsIHNreSByaW5nIHt2WydyX2luJ106LjFmfS0iCiAgICAgICAgZiJ7dlsncl9vdXQnXTouMWZ9IHB4LCB0aGUgc2FtZSBpbiBib3RoIGJhbmRzIiwKICAgICAgICBmInNhdHVyYXRpb24gICAgICAgICAgQiB7dlsnaW5mbyddWydCJ11bJ3NhdCddOi4wZn0gQURVLCBWIHt2WydpbmZvJ11bJ1YnXVsnc2F0J106LjBmfSBBRFUiLAogICAgICAgIGYiZ2FpbiAgICAgICAgICAgICAgICBCIHt2WydpbmZvJ11bJ0InXVsnZ2FpbiddOi4yZn0sIFYge3ZbJ2luZm8nXVsnViddWydnYWluJ106LjJmfSBlLS9BRFUiLAogICAgICAgIGYic291cmNlcyBtZWFzdXJlZCAgICB7bGVuKHZbJ2RmJ10pfSAocG9zaXRpdmUgZmx1eCBpbiBib3RoIGJhbmRzKSIsCiAgICAgICAgZiJjYWxpYnJhdGlvbiAgICAgICAgIHt2WydjYXRhbG9ndWUnXX06IHtjYWxbJ05fdXNlZCddfSBvZiB7Y2FsWydOX2NhbmRpZGF0ZXMnXX0gIgogICAgICAgIGYic3RhcnMgdXNlZCAoe3N0LmdldCgnbWF0Y2hlZCcsIDApfSBtYXRjaGVkKSIsCiAgICAgICAgZiIgICBCLVYgPSB7Y2FsWydhJ106Ky40Zn0gKyB7Y2FsWydjJ106LjRmfSB4IChiLXYpICAgc2NhdHRlciB7Y2FsWydybXNfY29sb3VyX2ZpdCddOi4zZn0iLAogICAgICAgIGYiICAgViAgID0gdiArIHtjYWxbJ1pQX1YnXTouNGZ9IHtjYWxbJ2tfViddOisuNGZ9IHggKEItVikgICBzY2F0dGVyIHtjYWxbJ3Jtc19WX2ZpdCddOi4zZn0iLAogICAgICAgIGYiICAgQiAgID0gYiArIHtjYWxbJ1pQX0InXTouNGZ9IHtjYWxbJ2tfQiddOisuNGZ9IHggKEItVikgICBzY2F0dGVyIHtjYWxbJ3Jtc19CX2ZpdCddOi4zZn0iLAogICAgICAgIGYiICAgY29sb3VyIGNvcnJlbGF0aW9uIHtjYWxbJ2NvcnJlbGF0aW9uJ106Ky4zZn0iLAogICAgICAgIGYiZm9yZWdyb3VuZCByZW1vdmVkICB7dlsnbl9jYXRfc3RhcnMnXX0gY2F0YWxvZ3VlIHN0YXJzLCB7dlsnbl9nYWlhJ119IEdhaWEgIgogICAgICAgIGYiZm9yZWdyb3VuZCBzdGFycywge3ZbJ25fb3V0c2lkZSddfSBiZXlvbmQgUjI1IiwKICAgICAgICBmInNvdXJjZXMgbGVmdCAgICAgICAge2xlbih2WydkZl9jbGVhbiddKX0iLAogICAgICAgIGYiYmx1ZSBrbm90cyAgICAgICAgICB7bGVuKHZbJ2RmX2NvbG9yX2ZpbHRlcmVkJ10pfSAgKHtDTURfQ09MT1JfTUlOfSA8PSAoQi1WKTAgPD0gIgogICAgICAgIGYie0NNRF9DT0xPUl9NQVh9LCBCLVYgZXJyb3IgPD0ge0FSR1MubWF4X2J2X2Vycn0sIG5vdCBzYXR1cmF0ZWQpIiwKICAgIF0KICAgIHdpdGggb3BlbihwYXRoLCAidyIsIGVuY29kaW5nPSJ1dGYtOCIpIGFzIGZoOgogICAgICAgIGZoLndyaXRlKCJcbiIuam9pbihsaW5lcykgKyAiXG4iKQogICAgcHJpbnQoZiJSdW4gc3VtbWFyeSBzYXZlZCB0byB7cGF0aH0iKQoKCmlmIF9fbmFtZV9fID09ICJfX21haW5fXyI6CiAgICBzeXMuZXhpdChtYWluKCkpCg=="


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
