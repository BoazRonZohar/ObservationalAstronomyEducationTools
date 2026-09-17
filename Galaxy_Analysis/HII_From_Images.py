"""
HII_From_Images.py - find HII regions in a pair of R and H-alpha images.

WHAT IT DOES
Give it the narrow-band H-alpha frame and the broad-band R frame of the same
galaxy. It lines them up, removes the starlight that both filters see, finds what
is left - the ionised hydrogen - and writes out a catalogue of positions, a scatter
map and a radial density profile.

ORDER OF OPERATIONS
  1. read both frames
  2. put the R frame onto the H-alpha pixel grid, using both WCS solutions
  3. measure the FWHM of the stars in each frame
  4. blur whichever frame is sharper, so both have the same star size
  5. measure the stars in both frames; their brightness ratio is the factor
  6. subtract:  factor x H-alpha  -  R
  7. detect, filter, then the shared radial analysis

Step 4 comes before step 5 on purpose. A star is measured through an aperture of
fixed size, and if one frame is sharper than the other the aperture catches a
different share of the light in each. Matching the blur first makes that question
disappear instead of having to be corrected for.

WHY THE SUBTRACTION IS THE HARD PART
Through the narrow filter every pixel receives two things: the H-alpha line from
the HII regions, and ordinary red starlight passing through anyway. The R frame
sees mostly that same starlight. Brighten H-alpha until its starlight matches R,
subtract, and what survives is line emission. Stars in our own galaxy should
vanish; if they do not, the factor is wrong.

THE KNOWN WEAKNESS OF THIS TOOL, STATED UP FRONT
That factor is not one number for the whole image - it depends on the colour of
whatever is being measured. Treating it as a constant, which is all that two frames
allow, biases the measured line strength: published tests put the error at up to 30
percent when the factor is set from foreground stars and up to 40 percent when it
is set from parts of the galaxy itself. Use Star_Formation_All_In_One.py when you also have
B and V, because with a colour the factor can be fitted rather than assumed.

WHAT DECIDES HOW MANY REGIONS YOU FIND
Far more than the subtraction: the detection threshold. Two student teams working
from the same M74 frames catalogued 106 and 369 regions, differing only in how
carefully they clicked and at what contrast they viewed. So this tool states its
threshold in units of the background noise, writes it into every output file, and
can measure how the detection limit varies with distance from the centre - because
a limit that worsens towards the crowded centre invents a central hole.

USAGE
    python HII_From_Images.py --halpha ha.fits --red r.fits --galaxy "NGC 628"
    python HII_From_Images.py --halpha ha.fits --red r.fits --galaxy "NGC 4212"
        --distance-mpc 16.7 --sector-method --completeness
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
# HII_From_Images.py
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

def detection_figure(resid, x, y, centre, profile, comp, scale, unit, args,
                     factor, diag, title, shapes=None):
    """The subtracted frame with the regions marked, and the completeness curves."""
    import textwrap
    from astropy.visualization import ZScaleInterval

    rows = 2 if comp else 1
    fig = plt.figure(figsize=(11.5, 5.4 * rows + 1.6))
    gs = fig.add_gridspec(rows, 1, hspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    finite = np.isfinite(resid)
    lo, hi = ZScaleInterval().get_limits(resid[finite])
    ax.imshow(resid, origin="lower", cmap="gray_r", vmin=lo, vmax=hi)
    if shapes is not None:
        # The marker is the region itself, not a token standing in for it, so a
        # large complex gets a large outline and it is obvious at a glance whether
        # the detection covered the whole blob or only its brightest corner.
        from matplotlib.patches import Ellipse
        for xi, yi, ai, bi, ti in zip(x, y, *shapes):
            if not (np.isfinite(ai) and np.isfinite(bi)):
                continue
            ax.add_patch(Ellipse((xi, yi), width=4.0 * ai, height=4.0 * bi,
                                 angle=ti, fill=False, ec="#a3283c", lw=0.9))
    else:
        ax.plot(x, y, "o", ms=5, mfc="none", mec="#a3283c", mew=0.9)
    ax.plot(centre[0], centre[1], "+", c="#2b5a89", ms=14, mew=2)
    ax.set_title("continuum-subtracted frame, {} regions kept".format(len(x)),
                 fontsize=11, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])

    if comp:
        ax2 = fig.add_subplot(gs[1, 0])
        shades = ["#2b5a89", "#7aa6d4", "#a8c4de", "#c9dae9"]
        for i, (lev, frac) in enumerate(sorted(comp.items(), reverse=True)):
            ax2.plot(profile["r"] * scale, 100 * frac, "o-", ms=4, lw=1.4,
                     c=shades[i % len(shades)],
                     label="peak {:.0f} sigma".format(lev))
        ax2.axhline(50, ls="--", lw=1, c="#888")
        ax2.set_ylim(0, 105)
        ax2.set_xlabel("radius [" + unit + "]")
        ax2.set_ylabel("recovered [%]")
        ax2.grid(alpha=0.18, lw=0.6)
        ax2.legend(fontsize=8, frameon=False, loc="center right")
        ax2.set_title("completeness against radius - artificial regions injected "
                      "into the real frame and re-detected", fontsize=10, loc="left")

    note = ("Detection: {:.1f} sigma above the local background, at least {} "
            "connected pixels, deblending {}. Subtraction {:.3f} x H-alpha - R, "
            "factor from {} stars with {:.1%} star-to-star scatter. A single "
            "constant factor can bias line strengths by tens of percent.").format(
        args.threshold, args.min_pixels, "off" if args.no_deblend else "on",
        factor, diag["n_stars"], diag["relative_scatter"])
    fig.text(0.06, 0.045, "\n".join(textwrap.wrap(note, width=130)),
             fontsize=7.9, color="#4a4a4a", va="top")

    fig.suptitle(title, fontsize=13, y=0.99)
    return fig

def ask_for_arguments():
    """Ask for what the command line would have carried, and hand back that list.

    Only the answers that cannot be guessed are asked for. Everything else keeps
    its default and can still be set with a flag, so the questions stay short
    enough that a person actually reads them.
    """
    print("No arguments were given, so I will ask instead.")
    print("Press Enter to accept the value shown in brackets.")
    print("")

    argv = ["--folder", ask_folder("Path to the folder holding the frames")]

    # Not asked for: the frames name the galaxy themselves and the folder scan
    # reads it. Only a folder whose frames have lost that keyword needs --galaxy.

    distance = ask_optional_float(
        "Distance in Mpc", "blank and it is looked up")
    if distance is not None:
        argv += ["--distance-mpc", repr(distance)]

    argv += ["--threshold", repr(ask("Detection threshold, in units of the "
                                     "background noise", 4.0, float))]
    argv += ["--rings", repr(ask("Number of radial rings", 20, int))]

    if ask_yes_no("Measure how many regions the frame is actually deep enough to "
                  "show? (slow)", False):
        argv += ["--completeness"]

    prefix = ask("Where to write the output, name without extension "
                 "(blank to write beside the H-alpha frame)", "")
    if prefix:
        argv += ["--out-prefix", normalise_path(prefix)]

    print("")
    return argv

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Find HII regions from an R and H-alpha image pair.")
    ap.add_argument("--folder", default=None,
                    help="a folder of frames; the two files are worked out from "
                         "the headers, and this is the usual way to run the tool")
    ap.add_argument("--halpha", default=None,
                    help="a single file, when you would rather name it yourself")
    ap.add_argument("--red", default=None)
    ap.add_argument("--galaxy", default=None,
                    help="name; used for the centre and for the catalogue geometry")
    ap.add_argument("--centre", nargs=2, type=float, default=None,
                    metavar=("X", "Y"), help="galaxy centre in pixels, overrides the name")
    ap.add_argument("--threshold", type=float, default=4.0,
                    help="detection threshold in units of background noise")
    ap.add_argument("--min-pixels", type=int, default=6)
    ap.add_argument("--extent-sigma", type=float, default=0.5,
                    help="how far above the sky noise the galaxy's light must be "
                         "to count as still present; sets the radius measured "
                         "from this exposure rather than taken from the catalogue")
    ap.add_argument("--window-margin", type=float, default=1.5,
                    help="how far beyond R25 to look for regions, as a multiple; "
                         "the frame is cut to this before detection, which is "
                         "where most of the time goes. R25 is an isophote rather "
                         "than an edge, so the default is generous")
    ap.add_argument("--background-box", type=int, default=None,
                    help="size of the square the background is measured in, in "
                         "pixels; the default is a twelfth of the frame")
    ap.add_argument("--min-width", type=float, default=1.2,
                    help="narrowest a blob may be and still count as a region, in "
                         "pixels; below this it is a subtraction residual")
    ap.add_argument("--max-elongation", type=float, default=2.5,
                    help="how far from round a region may be before it is treated "
                         "as a streak")
    ap.add_argument("--no-deblend", action="store_true")
    ap.add_argument("--deblend-levels", type=int, default=64,
                    help="how finely a touching group is sliced apart; higher "
                         "splits more")
    ap.add_argument("--deblend-contrast", type=float, default=1e-4,
                    help="how faint a peak may be, as a fraction of its group, and "
                         "still become a region of its own; lower splits more")
    ap.add_argument("--max-radius-r25", type=float, default=1.0,
                    help="keep detections inside this multiple of R25 (0 disables)")
    ap.add_argument("--star-radius", type=float, default=6.0)
    ap.add_argument("--star-margin", type=float, default=1.2,
                    help="stars used to set the subtraction factor must lie "
                         "beyond this multiple of R25, clear of the galaxy's "
                         "light; R25 is an isophote, not an edge, so a margin "
                         "beyond it is needed")
    ap.add_argument("--no-star-cut", action="store_true")
    ap.add_argument("--rings", type=int, default=20)
    ap.add_argument("--distance-mpc", type=float, default=None)
    ap.add_argument("--sector-method", action="store_true")
    ap.add_argument("--sectors", type=int, default=20)
    ap.add_argument("--sector-step", type=float, default=5.0)
    ap.add_argument("--completeness", action="store_true")
    ap.add_argument("--completeness-levels", type=float, nargs="+",
                    default=[10.0, 5.0, 3.0],
                    help="peak brightness of injected regions, in units of the "
                         "background noise")
    ap.add_argument("--no-catalogue", action="store_true")
    ap.add_argument("--out-prefix", default=None)

    # Started with nothing at all - from Spyder, or by double-clicking the file.
    # That is not a mistake, it is someone who has not been told the flags yet.
    if nothing_was_passed(argv):
        argv = ask_for_arguments()
    args = ap.parse_args(argv)

    donors, groups = {}, {}
    if args.folder:
        chosen, donors, groups, found_name = frames_from_folder(args.folder,
                                                                ("R", "Ha"))
        args.halpha = chosen["Ha"]
        args.red = chosen["R"]
        if not args.galaxy and found_name:
            args.galaxy = found_name

    # The distance is published, and every length reported below is that number
    # times an angle. Look it up, show it, and let it be overruled - which is what
    # passing --distance-mpc does.
    if args.galaxy and not args.distance_mpc:
        found = lookup_distance(args.galaxy)
        if found:
            args.distance_mpc = found
            print("")
            print(describe_lookups(args.galaxy, found, None))
            print("")
    if not (args.halpha and args.red):
        ap.error("give either --folder, or both --halpha and --red")

    # ---- what the catalogues know, before any pixels are touched -----------
    # These used to be looked up in the middle of the run, after the frames had
    # been combined and the subtraction factor already set. That order was wrong
    # twice over: the combining wants a list of stars to measure its shifts on,
    # and the factor wants to know which stars sit clear of the galaxy's light.
    # Both answers come from the catalogues, so the catalogues come first.
    probe_wcs, probe_shape, probe_info = frame_geometry(
        (groups.get("Ha") or [args.halpha])[0], donors.get("Ha"))

    geom = None
    if args.galaxy and not args.no_catalogue:
        try:
            describe = describe__galaxy_catalogue  # from galaxy_catalogue, inlined above
            geom = hyperleda_geometry(args.galaxy)
            print("")
            print("HyperLEDA geometry for " + args.galaxy + ":")
            print(describe(geom))
        except Exception as exc:
            print("Catalogue lookup failed: {}".format(exc))

    centre = None
    if args.centre:
        centre = (args.centre[0], args.centre[1])
        print("Galaxy centre given as ({:.1f}, {:.1f})".format(*centre))
    elif args.galaxy and probe_wcs is not None:
        try:
            from astropy.coordinates import SkyCoord
            c = SkyCoord.from_name(args.galaxy)
            px, py = probe_wcs.world_to_pixel(c)
            centre = (float(px), float(py))
            print("Galaxy centre from the WCS and the name: ({:.1f}, {:.1f})".format(
                *centre))
        except Exception as exc:
            print("Could not resolve the centre from the name ({})".format(exc))
    if centre is None:
        centre = (probe_shape[1] / 2.0, probe_shape[0] / 2.0)
        print("WARNING: falling back to the image centre ({:.1f}, {:.1f}). Give "
              "--centre for a meaningful radial profile.".format(*centre))

    scale_probe = probe_info.get("pixel_scale")
    r25_px = None
    if geom and geom.get("d25_arcmin") and scale_probe:
        r25_px = 0.5 * geom["d25_arcmin"] * 60.0 / float(scale_probe)

    print("Looking up the foreground stars in this field")
    fg_stars, fg_diag = gaia_foreground_stars(probe_wcs, probe_shape)
    print(describe_foreground(fg_diag))
    _centre_ref = load_frame((groups.get("R") or [args.red])[0], donors.get("R"))[0]
    fg_stars, _cdiag = centre_on_stars(_centre_ref, fg_stars)
    print(describe_centring(_cdiag))
    del _centre_ref

    # Stars clear of the galaxy, for measuring the subtraction factor and the
    # star sizes. This is how the measurement is done by hand and for the same
    # reason: the sky annulus around a star that sits on an arm is measuring the
    # arm, not the sky, and the flux that comes out of it is wrong in a way that
    # scatters rather than cancels. On M33 the factor measured over the whole
    # field disagreed from star to star by 40 percent.
    clear_stars = fg_stars
    if len(fg_stars) and r25_px:
        outside = np.hypot(fg_stars[:, 0] - centre[0],
                           fg_stars[:, 1] - centre[1]) > args.star_margin * r25_px
        if outside.sum() >= 20:
            clear_stars = fg_stars[outside]
            print("  {} of them lie beyond {:.1f} x R25 and are clear of the "
                  "galaxy's light; those are the ones measured".format(
                      int(outside.sum()), args.star_margin))
        else:
            print("  only {} lie beyond {:.1f} x R25, too few to measure with, so "
                  "all of them are used - the galaxy fills this frame and the "
                  "subtraction factor will be poorer for it".format(
                      int(outside.sum()), args.star_margin))

    print("Reading frames")
    ha, ha_wcs, ha_bpm, ha_info = combine_exposures(
        groups.get("Ha") or [args.halpha], donors.get("Ha"), "H-alpha",
        positions=fg_stars if len(fg_stars) else None)
    red, red_wcs, red_bpm, red_info = combine_exposures(
        groups.get("R") or [args.red], donors.get("R"), "red",
        positions=fg_stars if len(fg_stars) else None)
    print("  H-alpha : {:<9} {:>6.0f}s  {}".format(
        ha_info["filter"], ha_info["exptime"], os.path.basename(args.halpha)))
    print("  red     : {:<9} {:>6.0f}s  {}".format(
        red_info["filter"], red_info["exptime"], os.path.basename(args.red)))

    pixel_scale = ha_info["pixel_scale"] or red_info["pixel_scale"]
    if pixel_scale:
        print("  pixel scale {:.4f} arcsec/px".format(float(pixel_scale)))

    mask = ha_bpm
    if red_bpm is not None:
        mask = red_bpm if mask is None else (mask | red_bpm)

    north_angle, mirrored = 90.0, False
    if ha_wcs is not None:
        north_angle, mirrored = image_orientation(ha_wcs)
        print("  orientation from WCS: north {:.1f} deg from +x, {}".format(
            north_angle, "mirrored" if mirrored else "not mirrored"))

    placed = False
    if len(fg_stars):
        _tf, _ad = anchor_transform(ha, red, fg_stars)
        if _tf is not None:
            print("Placing the red frame on the H-alpha grid from anchor stars")
            print(describe_anchors(_ad))
            red = apply_transform(red, _tf, ha.shape)
            placed = True

    if placed:
        pass
    elif ha_wcs is not None and red_wcs is not None:
        print("Aligning the red frame onto the H-alpha grid using both WCS solutions")
        red = align_to(ha_wcs, ha.shape, red, red_wcs)
        # Two plate solutions fitted separately do not agree perfectly, and what is
        # left shows up as a dark-and-bright pair at every star in the subtracted
        # frame. The stars themselves measure it.
        red = refine_alignment(ha, red, "red", positions=fg_stars)
    else:
        print("WARNING: at least one frame has no WCS. Assuming the frames already "
              "line up pixel for pixel, which is often false.")

    _fw_stars = fg_stars if len(fg_stars) else None
    f_ha = measure_fwhm(ha, mask=mask, positions=_fw_stars)
    f_r = measure_fwhm(red, mask=mask, positions=_fw_stars)
    print("  FWHM  H-alpha {:.2f} px   red {:.2f} px  (reported, not acted on)"
          .format(f_ha if f_ha else float("nan"), f_r if f_r else float("nan")))

    # The blur and the factor are settled together, and by eye. Measuring two
    # widths and subtracting them in quadrature got the sign of the answer wrong
    # on real frames, and the mean factor over the stars left twice as many
    # visible star residuals as a value three percent away from it.
    _stem_now = output_prefix(args.out_prefix,
                              os.path.dirname(os.path.abspath(args.halpha)),
                              args.galaxy)
    ha, red, _picked, _chdiag = choose_subtraction(
        ha, red, _fw_stars, _stem_now, centre=centre, r25_px=r25_px, ask=ask)
    if _picked is not None:
        print(describe_choice(_chdiag))

    print("Measuring the stars in both frames to set the subtraction factor")
    if _picked is not None:
        factor = float(_picked)
        diag = {"n_stars": _chdiag.get("n_stars", 0), "relative_scatter": 0.0}
        print("  factor {:.3f}, as chosen above".format(factor))
    else:
        factor, diag = continuum_factor(
            ha, red, mask=mask, positions=clear_stars if len(clear_stars) else None)
        print("  stars are {:.3f}x brighter in {} than in {}".format(
            factor, red_info["filter"], ha_info["filter"]))
        print("  so the subtraction is  {:.3f} x H-alpha  -  R".format(factor))
        print("  from {} stars, star-to-star scatter {:.1%}".format(
            diag["n_stars"], diag["relative_scatter"]))

    # The scatter is not decoration. When the stars disagree this badly the factor
    # is barely a measurement, every star in the field leaves a residual, and the
    # detector reports those residuals as regions - on one M33 set at 60 percent
    # scatter it found 2844 of them, none of which was real. Said loudly here
    # because everything downstream still looks perfectly normal.
    if diag["relative_scatter"] > 0.20:
        print("")
        print("  WARNING: the stars disagree about the factor by {:.0%}. The "
              "subtraction".format(diag["relative_scatter"]))
        print("  cannot be trusted at that level, and what is detected below may be")
        print("  residuals rather than emission. Look at the subtracted frame before")
        print("  believing any of it - a good subtraction makes stars vanish.")
        print("")

    net, sky_diag = subtract_continuum(ha, red, factor, mask=mask)
    print("  sky removed first: {:.1f} counts from H-alpha, {:.1f} from R"
          .format(sky_diag["sky_ha"], sky_diag["sky_red"]))
    print("  without that the result would sit on a pedestal of {:.0f} counts"
          .format(sky_diag["pedestal_avoided"]))

    # Save it. This is the image the whole measurement rests on, and until now it
    # existed only inside a plot - so the one thing the tool asks a person to do
    # when the subtraction looks doubtful, look at the subtracted frame, could not
    # be done.
    _stem = output_prefix(args.out_prefix,
                          os.path.dirname(os.path.abspath(args.halpha)),
                          args.galaxy)
    # Named for what it is. Both tools use the same name for the same thing.
    _sub_path = _stem + "_Ha_minus_continuum.fits"
    save_frame(_sub_path, net, ha_wcs, ha_info, history=(
        "H-alpha with the continuum subtracted: {:.5f} x H-alpha - R".format(factor),
        "Factor from {} foreground stars, scatter {:.1%}".format(
            diag["n_stars"], diag["relative_scatter"]),
        "Made by HII_From_Images.py"))
    print("Wrote the H-alpha frame with the continuum taken out:")
    print("  " + _sub_path)

    _look_path = _stem + "_Ha_minus_continuum.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        # Imported here, not relied on from the module: this file imports it
        # inside each function that needs it, and this block was written
        # assuming otherwise.
        from astropy.stats import sigma_clipped_stats as _sigma_stats
        _nz = float(_sigma_stats(net, sigma=3.0)[2])
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

    # How far the galaxy reaches in THIS exposure, measured on the image that
    # holds the tracer being looked for - here the subtracted frame, because it
    # is emission that is being counted, and emission stops further in than the
    # old starlight does. The catalogue radius is what a deep plate shows, not
    # what six minutes on a small telescope shows.
    # Reported, not applied. Measuring where the emission fades is genuinely
    # informative - it says how deep this exposure went - but it is not a boundary
    # for the measurement. Emission comes in discrete regions with empty sky
    # between them, so any surface-brightness statistic understates how far the
    # regions reach: on M83 it gave 296 px while real regions were found out to
    # 556, and using it as a cut threw away half the catalogue.
    seen_r, seen_diag = visible_radius(net, centre, threshold=args.extent_sigma)
    print(describe_extent(seen_r, seen_diag, r25_px, "H-alpha emission"))

    # Detection is the most expensive thing here and it costs by the pixel, so it
    # is given the part of the frame the galaxy is in.
    window = galaxy_window(net.shape, centre, r25_px, margin=args.window_margin)
    print(describe_window(window, net.shape, args.window_margin))
    net_w = cut(net, window)
    mask_w = cut(mask, window) if mask is not None else None
    centre_w = shift_centre(centre, window)

    print("Detecting regions at {:.1f} sigma, minimum {} pixels".format(
        args.threshold, args.min_pixels))
    det_kwargs = dict(threshold_sigma=args.threshold, min_pixels=args.min_pixels,
                      mask=mask_w, deblend=not args.no_deblend,
                      box_size=args.background_box,
                      nlevels=args.deblend_levels, contrast=args.deblend_contrast,
                      r25_px=r25_px)
    cat, bkg, resid_w = detect_regions(net_w, **det_kwargs)
    if cat is None or len(cat) == 0:
        print("No regions found. Try a lower --threshold.")
        return 1

    # Back into the coordinates of the original frame, straight away. Everything
    # downstream - the catalogue, the plots, the pairing - speaks that language.
    x, y = restore_positions(np.asarray(cat.xcentroid, dtype=float),
                             np.asarray(cat.ycentroid, dtype=float), window)
    # Shape, straight from the segmentation: how wide each blob is and how far
    # from round. A region has a body; a subtraction residual is a thread.
    _minor = np.asarray(cat.semiminor_sigma.value, dtype=float)
    _major = np.asarray(cat.semimajor_sigma.value, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        _elong = _major / np.where(_minor > 1e-6, _minor, np.nan)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    _minor, _elong = _minor[good], _elong[good]
    print("  found {} regions".format(len(x)))

    # Cut down to what can plausibly be an HII region in this galaxy, before any
    # profile is computed. Doing it afterwards would leave contaminants in the rings.
    r_max_px = None
    if args.max_radius_r25 and r25_px:
        r_max_px = args.max_radius_r25 * r25_px
        print("Keeping detections inside {:.2f} x R25 = {:.0f} px".format(
            args.max_radius_r25, r_max_px))
    elif args.max_radius_r25:
        print("No D25 available, so no radial cut is applied. Detections far from "
              "the galaxy will contaminate the outer rings.")

    # For the cut, the catalogue list is the right one: it removes what is really
    # a foreground star and leaves the galaxy's own bright clusters alone.
    if args.no_star_cut:
        star_xy = None
    elif len(fg_stars):
        star_xy = fg_stars
    else:
        star_xy = star_positions(red, mask=mask)
    excess = line_excess(net, red, x, y)
    keep, report = filter_detections(x, y, centre, r_max_px, star_xy,
                                     args.star_radius, excess=excess,
                                     minor_axis=_minor, elongation=_elong,
                                     min_width_px=args.min_width,
                                     max_elongation=args.max_elongation)
    n_before = x.size
    x, y = x[keep], y[keep]
    _area = np.asarray(cat.area, dtype=float)[good][keep]
    _major_k = np.asarray(cat.semimajor_sigma.value, dtype=float)[good][keep]
    _minor_k = np.asarray(cat.semiminor_sigma.value, dtype=float)[good][keep]
    _theta_k = np.asarray(cat.orientation.value, dtype=float)[good][keep]
    print("Filtered {} -> {} regions ({})".format(
        n_before, x.size,
        ", ".join("{} {}".format(v, k) for k, v in report.items()) or "nothing removed"))
    if x.size == 0:
        print("Everything was filtered out. Check --centre and --max-radius-r25.")
        return 1

    # A tilted disk is an ellipse, and the catalogue says which one - so the shape
    # of what was detected can be predicted before it is measured, and compared.
    # It costs nothing and it is the only test here that can catch a detection
    # that has gone wrong while still producing a smooth, believable profile.
    if geom and geom.get("b_over_a"):
        expected_angle = None
        if geom.get("pa_sky") is not None:
            expected_angle = sky_pa_to_image_angle(geom["pa_sky"], north_angle,
                                                   mirrored)
        check = check_against_catalogue(x - centre[0], y - centre[1],
                                        geom["b_over_a"], expected_angle)
        print(describe_check(check))

    scale, unit = 1.0, "px"
    if pixel_scale:
        scale, unit = float(pixel_scale), "arcsec"
        if args.distance_mpc:
            scale = float(pixel_scale) * args.distance_mpc * 1e6 / 206264.806
            unit = "pc"

    prefix = _stem

    cat_path = prefix + "_regions.csv"
    with open(cat_path, "w", encoding="utf-8") as fh:
        fh.write("# HII regions from {} and {}\n".format(
            os.path.basename(args.halpha), os.path.basename(args.red)))
        fh.write("# detection {:.1f} sigma, min {} px, deblend={}\n".format(
            args.threshold, args.min_pixels, not args.no_deblend))
        fh.write("# subtraction {:.5f} x H-alpha - R, factor from {} stars, "
                 "scatter {:.1%}\n".format(factor, diag["n_stars"],
                                           diag["relative_scatter"]))
        fh.write("# centre {:.3f} {:.3f}\n".format(centre[0], centre[1]))
        fh.write("# kept within {} px of the centre; detections on stars removed\n".format(
            "{:.0f}".format(r_max_px) if r_max_px else "no limit"))
        fh.write("# area is how many pixels the region covers; radius is that of "
                 "a circle of the same area\n")
        fh.write("x,y,area_px,radius_px,radius_{}\n".format(unit))
        for xi, yi, ai in zip(x, y, _area):
            rad = float(np.sqrt(ai / np.pi))
            fh.write("{:.3f},{:.3f},{:.0f},{:.2f},{:.2f}\n".format(
                xi, yi, ai, rad, rad * scale))
    print("")
    print("Wrote " + cat_path)

    title = "HII regions - " + (args.galaxy or "galaxy")
    frame = (0.0, float(ha.shape[1]), 0.0, float(ha.shape[0]))
    treatments, results, _ = run_profiles(
        x, y, centre, frame, prefix, title,
        geom=geom, north_angle=north_angle, mirrored=mirrored,
        sector_method=args.sector_method, sectors=args.sectors,
        sector_step=args.sector_step, rings=args.rings, scale=scale, unit=unit)

    comp = None
    if args.completeness:
        print("")
        print("Measuring completeness against radius by injecting regions "
              "(this takes a minute)")
        rms = float(np.nanmedian(bkg.background_rms))
        print("  background noise {:.1f} counts, detection threshold {:.1f} sigma"
              .format(rms, args.threshold))
        comp = {}
        for lev in args.completeness_levels:
            frac, inj, rec = completeness_vs_radius(
                resid_w, centre_w, det_kwargs, results[0]["edges"],
                n_per_ring=30, peak_counts=lev * rms, fwhm_px=(f_ha or 6.0))
            comp[lev] = frac
            print("  peak {:.0f} sigma:".format(lev))
            for k in range(len(frac)):
                if np.isfinite(frac[k]):
                    print("    ring {:2d}  r = {:6.1f} px  recovered {:3.0f}%".format(
                        k, results[0]["r"][k], 100 * frac[k]))

    # The picture is of the window that was worked on, so the positions drawn on
    # it have to be in the window's coordinates, not the frame's.
    _wx, _wy = x, y
    _wcentre = centre
    if window is not None:
        _y0, _, _x0, _ = window
        _wx, _wy = x - _x0, y - _y0
        _wcentre = centre_w
    fig = detection_figure(resid_w, _wx, _wy, _wcentre, results[0], comp, scale,
                           unit, args, factor, diag, title,
                           shapes=(_major_k, _minor_k, _theta_k))
    png = prefix + "_detection.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote " + png)
    return 0

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


if __name__ == "__main__":
    sys.exit(main())
