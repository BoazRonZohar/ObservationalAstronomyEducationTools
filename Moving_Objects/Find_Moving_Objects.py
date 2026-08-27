# -*- coding: utf-8 -*-
"""
Find_Moving_Objects.py
============================

Find faint moving objects by TRACKING them, not by detecting them.

The third of the moving-object tools, and the one built for objects too faint
to stand out in any single frame. The other two ask each frame "is there
something here?" and then try to join the answers. This one asks the whole
night at once.

Why the other two are not enough
--------------------------------

On thirteen 60 s frames from the Faulkes 2 m, an asteroid the observer found
by eye in 2011 sits at a signal-to-noise of 5 to 12 per frame. It is one of
about seven thousand marginal detections. There is no threshold that keeps it
and rejects the noise: at 4 sigma it is buried among 578 spurious detections
per frame, and at 6 sigma it is gone. Both earlier tools failed on it, in
both directions.

The method, and where it came from
----------------------------------

Every step below except the stacking was proposed by the observer, who found
these objects by hand in 2011 and knows how it is actually done.

  1. detect, with a LOCAL background
        A global threshold is set by the brightest part of the frame. Next to
        a saturated star it is far above anything faint - which is exactly
        where this asteroid sits, and why it was invisible to the first
        attempt. Subtracting a background measured on a 48 px grid makes the
        threshold mean the same thing everywhere.

  2. keep only things with the instrument's WIDTH
        A real object has a PSF: it is as wide as every star in the frame. A
        cosmic ray is one pixel wide. A satellite and a meteor are lines with
        no width at all. Measuring the SHORT axis and requiring it to match
        the stars removes nine of every ten detections in one step - and it is
        an absolute test against the instrument, not a shape ratio.

  3. three epochs, not two, and the middle one must agree
        Two points always define a line, so any early detection paired with
        any late one is a "candidate" and millions of them are nonsense. Three
        points test whether the motion is UNIFORM, which is a real property of
        a solar system object over twenty minutes and not a property of noise.
        With ~600 detections per frame the chance a random one lands within
        2 px of the predicted midpoint is about 1 in 500, so this removes
        99.8% of the accidental pairs before any real work is done. It also
        removes cosmic rays at the root: a cosmic ray exists in one frame, so
        it cannot take part in a triple at all.

  4. blink with a STRIDE
        Between neighbouring frames this asteroid moves 1.8 px, less than the
        3.2 px width of a star, so it does not separate from itself. Between
        frames four apart it moves 6 px and separates cleanly.

  5. THEN stack along the velocity the triple implies
        This is the part the earlier tools lacked. Instead of asking "is there
        light at the predicted place in each frame" - a question noise answers
        yes to often enough - shift all the frames by where the object should
        have travelled and add them. A real object piles up: signal-to-noise
        gains the square root of the number of frames, 3.6x here, so 6 becomes
        22. An accidental triple has nothing along the rest of its line, so
        its stack stays empty.

Steps 2 and 3 are what make step 5 affordable. Without them there would be
two thousand blind velocities to try; with them, only the few dozen the data
itself supports.

Known answer
------------

The 13 frames of 2P/Encke from 2011-07-05 (Faulkes North) contain an asteroid
the observer discovered by eye: 18.8 arcsec/hour, present in all 13 frames,
straight to 0.18 arcsec, at RA 343.8985 Dec -8.7898 at MJD 55747.50356. Any
change to this file must still find it.

What you get for each track
---------------------------

The rate and direction, how many frames it appeared in, how far it moved
against its own scatter, and its position at the first frame. Then the
catalogue answer: SkyBoT is asked what known solar system object was at that
place at that moment.

When it names one, the next line is a link straight to that object's entry in
the Minor Planet Center database - orbit, observation history, ephemeris. A
numbered object is linked by its number, which never changes; an unnumbered
one by its provisional designation.

When it names nothing, there is no link, and that is the interesting case.

----------------------------------------------------------------------
Created by: Dr. Boaz Ron Zohar
Affiliation: Kinneret Observatory
Member of the LCO Global Sky Partners programme
Part of Observational Astronomy Education Tools
https://github.com/BoazRonZohar/ObservationalAstronomyEducationTools
"""

import os
import re
import sys
import glob
import argparse
import warnings

import numpy as np

NL = chr(10)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from astropy.io import fits                                      # noqa: E402
from astropy.wcs import WCS, FITSFixedWarning                    # noqa: E402
from astropy.time import Time                                    # noqa: E402
from astropy.utils.exceptions import AstropyWarning              # noqa: E402
from astropy.stats import SigmaClip, sigma_clipped_stats         # noqa: E402
from photutils.background import Background2D, MedianBackground  # noqa: E402
from photutils.detection import DAOStarFinder                    # noqa: E402
from scipy.ndimage import shift as ndshift, gaussian_filter      # noqa: E402
from scipy.spatial import cKDTree                                # noqa: E402

warnings.filterwarnings("ignore", category=AstropyWarning)
warnings.filterwarnings("ignore", category=FITSFixedWarning)


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------

def image_and_header(path):
    """One 2-D image and its header, whatever the file holds.

    A three-plane colour frame gives up its green plane - half the pixels of
    the Bayer mosaic rather than a quarter, so the sharpest of the three. A
    frame that is already one plane is used as it stands."""
    with fits.open(path, memmap=False) as hd:
        h, d = hd[0].header, hd[0].data
        if d is None and len(hd) > 1:
            h, d = hd[1].header, hd[1].data
        if d is None:
            raise RuntimeError("no image data")
        img = np.asarray(d[1] if d.ndim == 3 else d, dtype=float)
    return img, h


def build_wcs(header, fix_crpix=True):
    """The plate solution, with the stale crop rows taken out of it.

    MaxIm crops YORGSUBF rows after PinPoint has solved the frame and leaves
    CRPIX2 on the uncropped centre. Sky coordinates read straight from such a
    header sit about 15 arcsec from the truth, which does not matter for
    finding a track - the error is the same in every frame - but matters
    completely for saying WHERE the object was, which is the whole point."""
    h = header.copy()
    for k in list(h):
        if k.startswith(("TR1_", "TR2_")):
            del h[k]
    h["NAXIS"] = 2
    for k in ("NAXIS3", "NAXIS4"):
        if k in h:
            del h[k]
    if fix_crpix:
        try:
            h["CRPIX2"] = float(h["CRPIX2"]) - float(h.get("YORGSUBF", 0) or 0)
        except (KeyError, TypeError, ValueError):
            pass
    return WCS(h)


def frame_mjd(h):
    for k in ("MJD-OBS", "MJD_OBS"):
        if k in h:
            try:
                return float(h[k])
            except (TypeError, ValueError):
                pass
    for k in ("DATE-OBS", "DATE_OBS"):
        if k in h:
            try:
                return float(Time(str(h[k]), format="isot", scale="utc").mjd)
            except Exception:
                pass
    return float("nan")


def object_name(h):
    """OBJECT, with any per-frame scheduling serial removed."""
    return re.sub(r"\s*:\s*\d+\s*$", "", str(h.get("OBJECT", "")).strip()).strip()


def scan_folder(folder):
    files = []
    for root, _d, _f in os.walk(folder):
        for pat in ("*.fits", "*.fts", "*.fit", "*.FITS", "*.FTS", "*.FIT"):
            files += glob.glob(os.path.join(root, pat))
    out = {}
    for p in sorted(set(files)):
        try:
            h = fits.getheader(p)
            if h.get("NAXIS", 0) == 0 and "CD1_1" not in h:
                h = fits.getheader(p, 1)
        except Exception:
            continue
        if not all(k in h for k in ("CD1_1", "CRVAL1", "CRPIX1")):
            continue
        obj = object_name(h)
        if not obj:
            continue
        out.setdefault(obj, []).append((p, str(h.get("DATE-OBS", ""))))
    # one moment appears once: colour planes of the same exposure are merged
    final = {}
    for obj, rows in out.items():
        best = {}
        for path, when in rows:
            key = when or path
            base = os.path.basename(path).lower()
            rank = 0 if "green" in base else (1 if not any(
                t in base for t in ("red_of", "blue_of")) else 2)
            if key not in best or rank < best[key][1]:
                best[key] = (path, rank)
        final[obj] = sorted(p for p, _r in best.values())
    return final


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def short_axis(im, x, y, box=6):
    """(long, short) axis of a source, as FWHM in pixels, from its moments."""
    xi, yi = int(round(x)), int(round(y))
    if xi - box < 0 or yi - box < 0 or xi + box >= im.shape[1] or yi + box >= im.shape[0]:
        return None
    c = np.clip(im[yi - box:yi + box + 1, xi - box:xi + box + 1], 0, None)
    s = c.sum()
    if s <= 0:
        return None
    gy, gx = np.mgrid[-box:box + 1, -box:box + 1]
    mx, my = (c * gx).sum() / s, (c * gy).sum() / s
    xx = (c * (gx - mx) ** 2).sum() / s
    yy = (c * (gy - my) ** 2).sum() / s
    xy = (c * (gx - mx) * (gy - my)).sum() / s
    tr, dt = xx + yy, xx * yy - xy * xy
    if dt <= 0 or tr <= 0:
        return None
    r = np.sqrt(max(tr * tr / 4 - dt, 0.0))
    a2, b2 = tr / 2 + r, tr / 2 - r
    if a2 <= 0 or b2 <= 0:
        return None
    return 2.3548 * np.sqrt(a2), 2.3548 * np.sqrt(b2)


def prepare(paths, psf_px, threshold, max_per_frame=400, min_usable=5,
            verbose=True):
    """Align every frame, remove its local background, and find its sources."""
    import tempfile
    ref_w, c0 = None, None
    store, mjds, det = None, [], []
    scratch = os.path.join(tempfile.gettempdir(),
                           f"movtrack_{os.getpid()}.dat")
    for n, p in enumerate(paths):
        try:
            img, h = image_and_header(p)
            w = build_wcs(h)
            mjd = frame_mjd(h)
            if not np.isfinite(mjd):
                continue
        except Exception as e:
            if verbose:
                print(f"    skipped {os.path.basename(p)}: {e}")
            continue
        if ref_w is None:
            ref_w = w
            c0 = w.all_pix2world([[img.shape[1] / 2, img.shape[0] / 2]], 0)[0]
        x, y = w.all_world2pix(c0[0], c0[1], 0)
        img = ndshift(img, (img.shape[0] / 2 - float(y), img.shape[1] / 2 - float(x)),
                      order=1, mode="nearest")
        # A background measured on a grid, not one number for the whole frame.
        # This is the step that made the difference: next to a saturated star a
        # global threshold sits far above anything faint.
        bkg = Background2D(img, (48, 48), filter_size=(3, 3),
                           sigma_clip=SigmaClip(sigma=3.0),
                           bkg_estimator=MedianBackground())
        clean = gaussian_filter(img - bkg.background, 1.0)
        # The threshold has to be measured on the image the finder actually
        # sees. Taking it from the background BEFORE smoothing made "4 sigma"
        # mean 13.5 sigma of the smoothed frame - the known asteroid peaks at
        # 11 sigma there and was thrown away by the very step meant to find it.
        rms = float(sigma_clipped_stats(clean, sigma=3.0)[2])
        # The finder's default shape cuts exist to reject things that are not
        # stars - which includes comets. A coma is exactly the sort of soft,
        # extended source `sharplo` throws away, so the cuts are opened here.
        src = DAOStarFinder(fwhm=psf_px, threshold=threshold * rms,
                            sharplo=0.05, sharphi=3.0,
                            roundlo=-1.5, roundhi=1.5,
                            exclude_border=True)(clean)
        # Saturated stars are rejected HERE, by name, and not left to the
        # finder's shape cuts. Those cuts used to remove them as a side effect
        # of removing anything that is not a neat point - and the same cuts
        # removed comets. Opening them for the comet let a single saturated
        # star flood the output with 178 fragments of itself. A bright star's
        # core and spikes are not a shape problem, they are a brightness
        # problem, so they are masked as one.
        hot = img > 0.85 * float(np.nanmax(img))
        bad_xy = np.argwhere(hot)
        bad_tree = cKDTree(bad_xy[:, ::-1]) if len(bad_xy) else None
        k = len(mjds)
        if src is not None:
            for q in src:
                sh = short_axis(clean, float(q["xcentroid"]), float(q["ycentroid"]))
                if sh is None:
                    continue
                L, W = sh
                # It must HAVE a width - that is what separates a real object
                # from a cosmic ray, a satellite or a meteor, all of which are
                # lines with no thickness. There is deliberately NO upper limit
                # on the width: an earlier version rejected anything wider than
                # 2.2 PSF, which is precisely what a comet is. The tool was
                # filtering out the kind of object it was meant to find.
                if W < 0.55 * psf_px or L / W > 3.0:
                    continue
                if bad_tree is not None and bad_tree.query_ball_point(
                        (float(q["xcentroid"]), float(q["ycentroid"])), 90.0):
                    continue                     # inside a saturated star
                det.append((k, float(q["xcentroid"]), float(q["ycentroid"]),
                            float(q["flux"])))
        if max_per_frame:
            mine = [d for d in det if d[0] == k]
            if len(mine) > max_per_frame:
                mine.sort(key=lambda d: -d[3])
                det = [d for d in det if d[0] != k] + mine[:max_per_frame]
        if store is None:
            store = np.memmap(scratch, dtype=np.float32, mode="w+",
                              shape=(len(paths),) + clean.shape)
        store[len(mjds)] = clean.astype(np.float32)
        mjds.append(mjd)
        if verbose:
            print(f"\r    {n + 1}/{len(paths)} frames", end="", flush=True)
    if verbose:
        print()
    if len(mjds) < min_usable:
        raise SystemExit("Too few usable frames.")
    frames = store[:len(mjds)]
    return (frames, np.array(mjds), np.array(det), ref_w, scratch)


def drop_static(det, n_frames, tol=2.5, min_frames=None):
    """Set aside every position the sky keeps returning to."""
    if min_frames is None:
        min_frames = max(3, int(0.35 * n_frames))
    xy = det[:, 1:3]
    tree = cKDTree(xy)
    pairs = tree.query_ball_tree(tree, r=tol)
    lab = -np.ones(len(xy), int)
    n = 0
    for i in range(len(xy)):
        if lab[i] >= 0:
            continue
        st = [i]
        while st:
            j = st.pop()
            if lab[j] >= 0:
                continue
            lab[j] = n
            st.extend(pairs[j])
        n += 1
    keep = np.ones(len(xy), bool)
    centres, spreads = [], []
    for k in range(n):
        m = lab == k
        if len(np.unique(det[m, 0])) < min_frames:
            continue
        # A star's detections sit on top of each other. A SLOW MOVER's do not:
        # this asteroid shifts 1.8 px between neighbouring frames, inside the
        # 2.5 px linking radius, so single-link clustering chained all thirteen
        # of its detections into one group that looked exactly like a star
        # present in every frame - and the star filter deleted the very object
        # the tool exists to find. A cluster is only a star if it is COMPACT.
        p = xy[m]
        if np.hypot(*(p.max(axis=0) - p.min(axis=0))) > 2.0 * tol:
            continue
        keep[m] = False
        centres.append(p.mean(axis=0))
        spreads.append(float(np.sqrt(np.mean(
            np.sum((p - p.mean(axis=0)) ** 2, axis=1)))))
    precision = float(np.median(spreads)) if spreads else float("nan")
    return (det[keep], np.array(centres) if centres else np.zeros((0, 2)),
            precision)


# --------------------------------------------------------------------------
# triples and stacking
# --------------------------------------------------------------------------

def triples(det, hours, n_frames, psf_px, max_rate_px_h, mid_tol=2.0,
            min_move_px=3.0, min_middles=2, verbose=True):
    """Velocities supported by three or more epochs moving at ONE rate.

    Two points define a line and prove nothing. A third has to land where the
    other two say it should - a real property of anything orbiting the Sun
    over twenty minutes, and one noise satisfies about once in five hundred
    tries.

    WHICH three is not fixed, and that matters. A first version asked for the
    first, middle and last frames; the known asteroid is detected in the first
    and missing from both of the others, so no triple could ever form and the
    search returned nothing. Detection at this brightness is intermittent by
    nature. So any early detection is paired with any late one, and the line
    between them is then offered to EVERY frame in between - the object has to
    be found in at least `min_middles` of them, not in two particular ones."""
    # ANY two detections far enough apart in time - not "one early, one late".
    # The known asteroid is detected in frames 0,1,2,3,4 and 7 and in none of
    # the last four, so an early-times-late scheme had nothing to pair it with
    # and could never find it however good the rest of the method was.
    t0, t1 = hours[0], hours[-1]
    early = det
    late = det
    if len(det) == 0:
        return []
    by_frame = {k: cKDTree(det[det[:, 0] == k][:, 1:3])
                for k in range(n_frames) if (det[:, 0] == k).any()}
    # frames actually used for the middle test: a dozen spread through the run
    probe = sorted(set(int(round(v)) for v in
                       np.linspace(0, n_frames - 1, min(14, n_frames))))
    probe = [k for k in probe if k in by_frame]
    need = max(2, int(round(min_middles * len(probe) / max(n_frames - 2, 1))))
    need = min(need, max(2, len(probe) - 2))
    out = []
    span_all = t1 - t0
    max_move = max_rate_px_h * span_all
    by_frame_pts = {k: det[det[:, 0] == k] for k in by_frame}
    for ai in range(len(early)):
        ka = int(early[ai, 0]); ax, ay = early[ai, 1], early[ai, 2]
        ta = hours[ka]
        for kc, tree_c in by_frame.items():
            tc = hours[kc]
            dt = tc - ta
            if dt <= 0.45 * span_all:
                continue
            pts = by_frame_pts[kc]
            near = tree_c.query_ball_point((ax, ay), max_rate_px_h * dt)
            for ci in near:
                cx, cy = pts[ci, 1], pts[ci, 2]
                move = np.hypot(cx - ax, cy - ay)
                if move < min_move_px:
                    continue
                vx, vy = (cx - ax) / dt, (cy - ay) / dt
                seen = 0
                for k in probe:
                    if k in (ka, kc):
                        continue
                    px = ax + vx * (hours[k] - ta)
                    py = ay + vy * (hours[k] - ta)
                    if by_frame[k].query_ball_point((px, py), mid_tol):
                        seen += 1
                if seen >= need:
                    out.append((vx, vy, ax, ay, ta, seen + 2))
    if verbose:
        print(f"    {len(out)} lines that at least {need + 2} of "
              f"{len(probe)} sampled epochs agree on")
    return out


def stack_along(frames, hours, vx, vy, x0, y0, t0, half=22):
    """Add the frames up along one velocity, and see if anything piles up.

    Only a small box is stacked: the velocity already says where the object
    should be in every frame, so there is nothing to gain from moving whole
    images around."""
    n, ny, nx = frames.shape
    acc = np.zeros((2 * half + 1, 2 * half + 1))
    used = 0
    for k in range(n):
        X = x0 + vx * (hours[k] - t0)
        Y = y0 + vy * (hours[k] - t0)
        xi, yi = int(round(X)), int(round(Y))
        if xi - half < 0 or yi - half < 0 or xi + half >= nx or yi + half >= ny:
            continue
        acc += frames[k][yi - half:yi + half + 1, xi - half:xi + half + 1]
        used += 1
    if used < 5:
        return None
    acc /= used
    _m, med, rms = sigma_clipped_stats(acc, sigma=3.0)
    c = acc[half - 3:half + 4, half - 3:half + 4]
    return (float(c.max() - med) / rms if rms > 0 else np.nan), used, acc



# --------------------------------------------------------------------------
# the quick look
# --------------------------------------------------------------------------

class Tee:
    """Write to the screen and to a file at the same time.

    The console is where the answer appears and also where it disappears: it
    scrolls, and a session that took an hour leaves nothing behind. Everything
    printed goes to run_log.txt as well, including the questions asked and the
    answers given, so a run can be read back exactly as it happened."""

    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, "a", encoding="utf-8")
        import datetime
        self.fh.write(chr(10) + "=" * 70 + chr(10) +
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
                      chr(10) + "=" * 70 + chr(10))

    def write(self, s):
        self.stream.write(s)
        try:
            self.fh.write(s)
            self.fh.flush()
        except Exception:
            pass
        return len(s)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


def identify(ra_deg, dec_deg, mjd, radius_arcsec=30.0, site="500", timeout=25):
    """Ask SkyBoT what known solar system object is at this place and time.

    IMCCE's SkyBoT holds the orbits of everything catalogued and will say what
    was in a given patch of sky at a given moment. That turns a track into a
    name - or, much more interestingly, fails to, which is the only way an
    unknown object announces itself.

    The default observer is the geocentre. For a main-belt asteroid that is
    wrong by a few arcseconds - Earth's radius over a couple of astronomical
    units - so the match radius allows for it. Give the observatory's MPC code
    as `site` and the residual drops to the measurement error.

    Never fatal: no internet, a slow service or an unexpected reply all return
    None, and the run carries on."""
    import urllib.request
    import urllib.parse
    url = "https://ssp.imcce.fr/webservices/skybot/api/conesearch.php?" + \
        urllib.parse.urlencode({
            "-ep": f"{mjd + 2400000.5:.6f}", "-ra": f"{ra_deg:.6f}",
            "-dec": f"{dec_deg:.6f}", "-rm": f"{max(radius_arcsec / 60.0, 1.0):.2f}",
            "-mime": "text", "-output": "object", "-loc": str(site),
            "-filter": "0", "-objFilter": "111", "-refsys": "EQJ2000",
            "-from": "FindMovingObjects"})
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
    except Exception as e:
        return {"error": str(e)}
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = [c.strip() for c in line.split("|")]
        if len(f) < 8 or f[0].lower().startswith("num"):
            continue
        try:
            # the separation from the search centre, in arcsec
            sep = min((float(v) for v in f[6:9]
                       if v.replace(".", "", 1).replace("-", "", 1).isdigit()),
                      default=None)
        except Exception:
            sep = None
        if sep is None:
            continue
        mag = None
        for v in f[4:8]:
            try:
                x = float(v)
                if 0 < x < 40:
                    mag = x
                    break
            except ValueError:
                continue
        rows.append({"number": f[0] or "-", "name": f[1], "sep": sep,
                     "mag": mag, "ra": f[2], "dec": f[3]})
    rows.sort(key=lambda z: z["sep"])
    return {"objects": rows}


def mpc_url(number, name):
    """The object's page in the Minor Planet Center database.

    The number is used when the object has one: a numbered object keeps its
    number for good, while names and provisional designations change. An
    unnumbered object has only its designation, so that is what goes in.

    Returns None when there is nothing to build a link from, rather than a
    link that leads nowhere.
    """
    import urllib.parse
    num = str(number or "").strip().strip("()").strip()
    ident = num if num not in ("-", "") else str(name or "").strip()
    if not ident:
        return None
    return ("https://www.minorplanetcenter.net/db_search/show_object?object_id="
            + urllib.parse.quote(ident))


def name_it(ra, dec, mjd, args):
    """What the catalogue thinks this is, as a list of lines to print.

    A list rather than one string because an identified object also gets a
    link to its MPC entry, and the callers indent each line themselves.
    """
    if getattr(args, "no_catalogue", False):
        return ["not checked against the catalogue"]
    r = identify(ra, dec, mjd, radius_arcsec=args.match_arcsec,
                 site=args.site_code)
    if "error" in r:
        return [f"could not reach SkyBoT ({r['error']}) - the position above "
                f"can be pasted into it by hand"]
    objs = r.get("objects", [])
    near = [o for o in objs if o["sep"] <= args.match_arcsec]
    if not near:
        n = len(objs)
        return [f"NO known solar system object within {args.match_arcsec:.0f} "
                f"arcsec ({n} catalogued further out) - worth a second look"]
    o = near[0]
    num = f"({o['number']}) " if o["number"] not in ("-", "") else ""
    mag = f", magnitude {o['mag']:.1f}" if o["mag"] else ""
    extra = ""
    if len(near) > 1:
        extra = f"  [{len(near) - 1} other catalogued object(s) also within " \
                f"{args.match_arcsec:.0f} arcsec]"
    lines = [f"identified: {num}{o['name']} - {o['sep']:.1f} arcsec from the "
             f"measured position{mag}{extra}"]
    link = mpc_url(o["number"], o["name"])
    if link:
        lines.append(f"MPC entry: {link}")
    return lines


def draw_objects(objs, frames, hours, px, stem, tag=""):
    """One picture per object: the same field, three times, circle moving.

    A window that follows the object shows a dot standing still and proves
    nothing. The stars have to be in the picture, and staying put, for the
    motion to be visible at all - which is also why the field is several times
    the length of the track rather than a tight crop."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  could not draw the cut-outs: {e}")
        return
    for k, o in enumerate(objs, 1):
        try:
            cx0 = int(round(0.5 * (o["X"].min() + o["X"].max())))
            cy0 = int(round(0.5 * (o["Y"].min() + o["Y"].max())))
            half = int(max(170, 4.0 * max(np.ptp(o["X"]), np.ptp(o["Y"]))))
            pick = [0, len(frames) // 2, len(frames) - 1]
            # Four panels. The first is the WHOLE frame with the object marked,
            # because a close-up on its own says what it does but never says
            # where it is - and where it is, is the first thing anyone asks.
            fig, ax = plt.subplots(1, 4, figsize=(21.5, 6.0))
            full = frames[0]
            flo, fhi = np.percentile(full[::4, ::4], [30, 99.5])
            ax[0].imshow(full, cmap="gray_r", vmin=flo, vmax=fhi, origin="lower")
            ax[0].plot(o["X"], o["Y"], "-", color="red", lw=1.2)
            ax[0].plot(o["x0"], o["y0"], "o", ms=30, mfc="none",
                       mec="red", mew=1.4)
            ax[0].add_patch(plt.Rectangle(
                (cx0 - half, cy0 - half), 2 * half, 2 * half,
                fill=False, ec="red", lw=0.8, ls="--"))
            ax[0].set_xticks([]); ax[0].set_yticks([])
            ax[0].set_title(f"the whole frame, {full.shape[1]}x{full.shape[0]} px"
                            f"\nobject at pixel {o['x0']:.0f},{o['y0']:.0f}",
                            fontsize=10)
            for n, i in enumerate(pick, start=1):
                y0, y1 = max(cy0 - half, 0), min(cy0 + half, frames.shape[1])
                x0, x1 = max(cx0 - half, 0), min(cx0 + half, frames.shape[2])
                c = frames[i][y0:y1, x0:x1]
                lo, hi = np.percentile(c, [15, 99.5]) if c.size else (0, 1)
                ax[n].imshow(c, cmap="gray_r", vmin=lo, vmax=hi,
                             origin="lower", extent=[x0, x1, y0, y1])
                xp = np.polyval([o["vx"], o["x0"]], hours[i])
                yp = np.polyval([o["vy"], o["y0"]], hours[i])
                ax[n].plot(xp, yp, "o", ms=26, mfc="none", mec="red", mew=1.4)
                ax[n].set_xticks([]); ax[n].set_yticks([])
                ax[n].set_title(f"{hours[i] * 60:.0f} min", fontsize=11)
            rate = float(np.hypot(o["vx"], o["vy"]) * px)
            fig.suptitle(f"object {k}{tag}  -  {rate:.1f} arcsec/hour  -  "
                         f"same field in the three close-ups, only the circle moves",
                         fontsize=12)
            fig.tight_layout()
            fig.savefig(f"{stem}_object{k}.png", dpi=110)
            plt.close(fig)
        except Exception as e:
            print(f"  could not draw object {k}: {e}")


def quick_look(paths, idx, args, stem=None, px_scale=1.0, verbose_depth=True):
    """A real search, on a handful of frames spread through the stretch.

    The first version of this read exactly three - first, middle, last - which
    is the classic blink and is right in principle. In practice it failed on
    the one night whose answer is known, and for a reason worth keeping in
    view: at this brightness a detection is intermittent. The 2011 asteroid is
    found in frames 0,1,2,3,4 and 7 of thirteen and missing from the rest, so
    no triple built from fixed positions contained it three times. Three frames
    are enough for an object that is plainly there in all of them, and this one
    is not.

    So a handful is read instead of three, and the full method runs on them.
    Seven frames out of a hundred and sixty is still a twentieth of the work,
    and it asks a fair question rather than a lucky one."""
    sub = [paths[i] for i in idx]
    # Progress is shown here too. Reading fourteen frames off an external disk
    # is a minute or two of silence, and silence is indistinguishable from a
    # hung run - which is exactly the question this tool kept being asked.
    print("")
    frames, mjds, det, ref_w, scratch = prepare(
        sub, args.psf_px, args.threshold, max_per_frame=args.max_per_frame,
        min_usable=3, verbose=True)
    try:
        # How deep each frame went, from how many sources it yielded. Spread
        # alone is a blind choice: it picks whichever frame falls on the mark,
        # cloud or not, and on the one night whose answer is known the blind
        # picks landed on frames where the asteroid was not detected at all.
        # The sample is read in PAIRS that are adjacent in time and the deeper
        # of each pair is kept - so the seven that go forward are both spread
        # across the night and the best of what was looked at.
        counts = np.array([int((det[:, 0] == k).sum()) for k in range(len(mjds))])
        if len(mjds) <= args.quick_frames:
            # Nothing to choose between: throwing away half of a short stretch
            # leaves too little to define a track at all, which is how a look
            # at the first seven frames of Encke came back empty when all seven
            # together hold the asteroid.
            keep = list(range(len(mjds)))
        else:
            keep = []
            for i in range(0, len(mjds) - 1, 2):
                keep.append(i if counts[i] >= counts[i + 1] else i + 1)
            if len(mjds) % 2:
                keep.append(len(mjds) - 1)
        keep = sorted(set(keep))
        if verbose_depth:
            print(f" [{len(mjds)} read, {len(keep)} kept: "
                  f"{', '.join(str(counts[k]) for k in keep)} sources]",
                  end="")
        remap = -np.ones(len(mjds), int)
        remap[keep] = np.arange(len(keep))
        m = np.isin(det[:, 0].astype(int), keep)
        sel = det[m].copy()
        sel[:, 0] = remap[sel[:, 0].astype(int)]
        objs = search_prepared(frames[keep], mjds[keep], sel,
                               psf_px=args.psf_px, min_snr=args.min_snr,
                               min_significance=args.min_significance,
                               max_wander=args.max_wander, px_scale=px_scale,
                               max_rate_arcsec_h=args.max_rate, verbose=False)
        hrs = hours_of(mjds[keep])
        # The pictures are drawn HERE, while the frames still exist. A quick
        # look used to leave nothing behind but a line on the screen, which is
        # the one thing that cannot be checked afterwards.
        if stem and objs:
            draw_objects(objs, frames[keep], hrs, px_scale, stem,
                         tag=" (quick look)")
        return objs, hrs, np.sort(mjds[keep]), ref_w
    finally:
        del frames
        try:
            os.remove(scratch)
        except OSError:
            pass


def quick_session(paths, args, px_scale, out_stem):
    """Look at a few frames, say what is there, and let the observer choose.

    A stretch that shows nothing may still hold an object that was only in the
    field for part of it, so the stretch is halved and looked at again - as
    many times as the observer wants, and no more."""
    lo, hi = 0, len(paths) - 1
    while True:
        n = min(2 * args.quick_frames, hi - lo + 1)
        idx = sorted(set(int(round(v)) for v in np.linspace(lo, hi, n)))
        print(f"\n  reading {len(idx)} frames of {len(paths)} "
              f"(numbers {idx[0] + 1} to {idx[-1] + 1}) ...", end="", flush=True)
        stem = f"{out_stem}_quick_{lo + 1}-{hi + 1}"
        try:
            objs, hours, mjds, ref_w = quick_look(paths, idx, args, stem=stem,
                                                  px_scale=px_scale)
        except SystemExit as e:
            print(f" {e}")
            return
        span = hours[-1] - hours[0]
        print(f" {span * 60:.0f} minutes apart")
        names = []
        if objs:
            print(f"  FOUND {len(objs)} thing(s) that moved:")
            for k, o in enumerate(objs, 1):
                rate = float(np.hypot(o["vx"], o["vy"]) * px_scale)
                sc = ref_w.all_pix2world(o["x0"], o["y0"], 0)
                print(f"    {k}. {rate:6.1f} arcsec/hour | seen in "
                      f"{o['n']}/{len(idx)} | moved {o['moved_px'] * px_scale:.1f}\" "
                      f"= {o['sig']:.0f}x its scatter | "
                      f"RA {float(sc[0]):.6f} Dec {float(sc[1]):+.6f}")
                names.append(name_it(float(sc[0]), float(sc[1]),
                                     float(mjds[0]), args))
                for _ln in names[-1]:
                    print(f"       {_ln}")
            with open(stem + "_tracks.txt", "w", encoding="utf-8") as fh:
                fh.write(f"quick look, frames {lo + 1} to {hi + 1} of "
                         f"{len(paths)}" + NL +
                         f"{len(idx)} read, {len(objs)} object(s), "
                         f"first MJD {mjds[0]:.6f}" + NL + NL)
                for k, o in enumerate(objs, 1):
                    rate = float(np.hypot(o["vx"], o["vy"]) * px_scale)
                    ang = float(np.degrees(np.arctan2(o["vx"], o["vy"])) % 360)
                    sc = ref_w.all_pix2world(o["x0"], o["y0"], 0)
                    fh.write(f"object {k}: {rate:.1f} arcsec/hour towards "
                             f"{ang:.1f} deg | seen in {o['n']} frames | "
                             f"moved {o['moved_px'] * px_scale:.1f} arcsec, "
                             f"{o['sig']:.0f}x its own "
                             f"{o['rms_px'] * px_scale:.2f} arcsec scatter" + NL +
                             f"          start RA {float(sc[0]):.6f} "
                             f"Dec {float(sc[1]):+.6f} at MJD {mjds[0]:.6f}"
                             f"   (pixel {o['x0']:.1f},{o['y0']:.1f})" + NL +
                             "".join("          " + _ln + NL
                                     for _ln in names[k - 1]) + NL)
            print(f"  -> {stem}_tracks.txt  and a picture per object")
            print("  (a sample of frames - run again and answer n to the quick "
                  "look to measure them properly)")
        else:
            print("  NOT FOUND in this sample")
        # Halving is the point, so it is allowed until a half is genuinely too
        # small to search. Requiring twice the sample size refused to halve a
        # thirteen frame night - and halving is exactly what that night needed,
        # because its asteroid is only detectable in the first third and no
        # pair inside that third spans enough of the WHOLE run to count.
        if hi - lo < 6:
            print("  the stretch is too short to halve again")
            return
        try:
            ans = input("  look at the first half, the second half, or stop? "
                        "[f/s/n]: ").strip().lower()
        except EOFError:
            return
        if ans.startswith("f"):
            hi = (lo + hi) // 2
        elif ans.startswith("s"):
            lo = (lo + hi) // 2
        else:
            return


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def pixel_scale(header):
    """Arcseconds per pixel, from the plate solution."""
    try:
        return float(np.hypot(float(header["CD1_1"]),
                              float(header["CD2_1"])) * 3600.0)
    except Exception:
        return 1.0


def search(paths, psf_px=3.2, threshold=4.0, min_snr=8.0,
           max_per_frame=400, min_significance=5.0, max_wander=5.0,
           max_rate_arcsec_h=400.0, verbose=True):
    frames, mjds, det, ref_w, scratch = prepare(paths, psf_px, threshold,
                                                max_per_frame=max_per_frame,
                                                verbose=verbose)
    try:
        h0 = fits.getheader(paths[0])
        if "CD1_1" not in h0:
            h0 = fits.getheader(paths[0], 1)
        px = pixel_scale(h0)
    except Exception:
        px = 1.0
    objs = search_prepared(frames, mjds, det, psf_px, min_snr,
                           min_significance, max_wander, px,
                           max_rate_arcsec_h, verbose)
    return objs, frames, hours_of(mjds), mjds[np.argsort(mjds)], ref_w, scratch


def hours_of(mjds):
    m = np.sort(np.asarray(mjds, float))
    return (m - m[0]) * 24.0


def search_prepared(frames, mjds, det, psf_px=3.2, min_snr=8.0,
                    min_significance=5.0, max_wander=5.0, px_scale=1.0,
                    max_rate_arcsec_h=400.0, verbose=True):
    """Everything after the frames have been read and their sources found."""
    order = np.argsort(mjds)
    frames, mjds = frames[order], mjds[order]
    remap = np.empty(len(order), int)
    remap[order] = np.arange(len(order))
    det[:, 0] = remap[det[:, 0].astype(int)]
    hours = (mjds - mjds[0]) * 24.0
    if verbose:
        print(f"    {len(det)} detections with a star's width, "
              f"{len(frames)} frames over {hours[-1]:.2f} h")

    left, stat, precision = drop_static(det, len(frames))
    if verbose:
        print(f"    {len(stat)} static stars, {len(left)} detections left over"
              + (f", positions repeat to {precision:.2f} px"
                 if np.isfinite(precision) else ""))

    span = hours[-1] - hours[0]
    # The ceiling is a RATE, not a distance. It used to be sixty pixels of
    # total travel however long the night was, which says a slow object on a
    # long run and a fast one on a short run are the same thing - they are not.
    # An object at 29.5 arcsec/hour crossed 83 px in two hours of WASP-52 and
    # was refused by that ceiling, while the same object inside a shorter
    # stretch passed. What is physical is the rate: a main-belt asteroid moves
    # around 30 arcsec/hour, and a near-Earth object a few hundred, so the
    # limit is set in those units and converted with the pixel scale.
    max_rate_px_h = max_rate_arcsec_h / max(px_scale, 1e-6)
    cands = triples(left, hours, len(frames), psf_px,
                    max_rate_px_h=max_rate_px_h, verbose=verbose)

    seen, hits = set(), []
    for vx, vy, x0, y0, t0, nep in cands:
        r = stack_along(frames, hours, vx, vy, x0, y0, t0)
        if r is None:
            continue
        snr, used, acc = r
        if not np.isfinite(snr) or snr < min_snr:
            continue
        key = (int(round(x0 / 12)), int(round(y0 / 12)),
               int(round(vx / 4)), int(round(vy / 4)))
        if key in seen:
            continue
        seen.add(key)
        hits.append(dict(snr=snr, vx=vx, vy=vy, x0=x0, y0=y0, t0=t0,
                         used=used, epochs=nep))
    hits.sort(key=lambda h: -h["snr"])
    _ = min_snr

    # One object arrives many times over, because many different pairs of its
    # own detections imply nearly the same line. Reporting eight lines that are
    # really two objects is how the comet came to be called "not found" when it
    # was sitting in the list twice. Two tracks are the same object if they
    # occupy the same place at the beginning, the middle and the end.
    def at(h, t):
        return h["x0"] + h["vx"] * (t - h["t0"]), h["y0"] + h["vy"] * (t - h["t0"])
    probes = (hours[0], hours[len(hours) // 2], hours[-1])
    objects = []
    for h in hits:
        for g in objects:
            if all(np.hypot(*(np.subtract(at(h, t), at(g[0], t)))) < 10.0
                   for t in probes):
                g.append(h)
                break
        else:
            objects.append([h])

    # measure each distinct object properly: follow it frame by frame from its
    # own predicted path, and fit the track to what is actually there
    out = []
    for g in objects:
        h = g[0]
        # Measured against the LINE, not against the previous frame. Chaining
        # frame to frame let the box wander: one frame locked onto a
        # neighbouring star and every later frame followed that star, which is
        # how this object came out at 28 arcsec/hour instead of 18.8. The line
        # is refitted twice, and each frame is re-measured within a few pixels
        # of where the line says the object is - so a single bad frame cannot
        # drag the rest of the track with it.
        cx = np.array([h["vx"], h["x0"] - h["vx"] * h["t0"]])
        cy = np.array([h["vy"], h["y0"] - h["vy"] * h["t0"]])
        X = Y = t = None
        for _pass in range(3):
            xs, ys, sn, ts = [], [], [], []
            for k in range(len(frames)):
                gx, gy = np.polyval(cx, hours[k]), np.polyval(cy, hours[k])
                xi, yi = int(round(gx)), int(round(gy))
                r = 5 if _pass else 9
                if (xi - r < 0 or yi - r < 0 or xi + r >= frames.shape[2]
                        or yi + r >= frames.shape[1]):
                    continue
                c = frames[k][yi - r:yi + r + 1, xi - r:xi + r + 1]
                j = np.unravel_index(np.argmax(c), c.shape)
                sl = np.clip(c[max(j[0] - 3, 0):j[0] + 4,
                               max(j[1] - 3, 0):j[1] + 4], 0, None)
                tot = sl.sum()
                if tot <= 0:
                    continue
                gy2, gx2 = np.mgrid[0:sl.shape[0], 0:sl.shape[1]]
                xs.append(xi - r + max(j[1] - 3, 0) + (sl * gx2).sum() / tot)
                ys.append(yi - r + max(j[0] - 3, 0) + (sl * gy2).sum() / tot)
                sn.append(float(c[j] / np.std(c)))
                ts.append(hours[k])
            if len(xs) < 4:
                break
            X, Y, t, SN = (np.array(xs), np.array(ys), np.array(ts), sn)
            cx, cy = np.polyfit(t, X, 1), np.polyfit(t, Y, 1)
        if X is None or len(X) < max(6, int(0.6 * len(frames))):
            continue
        res = np.hypot(X - np.polyval(cx, t), Y - np.polyval(cy, t))
        rms_px = float(np.sqrt(np.mean(res ** 2)))
        moved_px = float(np.hypot(cx[0], cy[0]) * (t[-1] - t[0]))
        if rms_px <= 0 or moved_px / rms_px < min_significance:
            continue
        # Five times, not three. A comet is not a point: its centroid is
        # measured less precisely than a star's however good the frames are,
        # and at three times the stellar scatter the 2011 comet was thrown out
        # while the asteroid beside it survived. The allowance has to fit the
        # kind of object being looked for, not only the instrument.
        if np.isfinite(precision) and rms_px > max_wander * max(precision, 0.15):
            continue                      # wanders further than the stars do
        out.append(dict(n=len(X), vx=cx[0], vy=cy[0], x0=np.polyval(cx, 0.0),
                        y0=np.polyval(cy, 0.0), rms_px=float(np.sqrt(np.mean(res ** 2))),
                        snr=h["snr"], snr_frame=float(np.median(SN)),
                        lines=len(g), X=X, Y=Y, T=t,
                        moved_px=moved_px, sig=moved_px / rms_px))
    # Merge AGAIN, now that every track has been refitted: two candidates that
    # started apart routinely converge onto the same object, and reporting it
    # twice is how a list of two objects looked like a list of three.
    t_a, t_b = hours[0], hours[-1]

    def ends(o):
        return (o["x0"] + o["vx"] * t_a, o["y0"] + o["vy"] * t_a,
                o["x0"] + o["vx"] * t_b, o["y0"] + o["vy"] * t_b)

    merged = []
    for o in sorted(out, key=lambda z: -z["sig"]):
        ax, ay, bx, by = ends(o)
        for m in merged:
            mx, my, nx_, ny_ = ends(m)
            if np.hypot(ax - mx, ay - my) < 9.0 and np.hypot(bx - nx_, by - ny_) < 9.0:
                break
        else:
            merged.append(o)
    return merged


def main():
    ap = argparse.ArgumentParser(
        description="Track faint moving objects by stacking along their motion.")
    ap.add_argument("--folder", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="look at a few frames first, and ask before doing more")
    ap.add_argument("--quick_frames", type=int, default=7,
                    help="how many frames the quick look reads")
    ap.add_argument("--psf_px", type=float, default=3.2)
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--no_catalogue", action="store_true",
                    help="skip the SkyBoT check (no internet, or not wanted)")
    ap.add_argument("--match_arcsec", type=float, default=30.0,
                    help="how close a catalogued object must be to count as "
                         "the same one")
    ap.add_argument("--site_code", default="500",
                    help="MPC observatory code. 500 is the geocentre, which "
                         "for a main-belt asteroid is a few arcsec out")
    ap.add_argument("--max_rate", type=float, default=400.0,
                    help="fastest motion looked for, arcsec/hour. A main-belt "
                         "asteroid moves about 30; a near-Earth object can "
                         "reach several hundred")
    ap.add_argument("--max_wander", type=float, default=5.0,
                    help="how many times the stars' own scatter a track may "
                         "wander off its line")
    ap.add_argument("--min_significance", type=float, default=5.0,
                    help="how many times its own scatter a track must travel")
    ap.add_argument("--max_per_frame", type=int, default=400,
                    help="keep only this many of the brightest per frame")
    ap.add_argument("--min_snr", type=float, default=8.0,
                    help="how strong the stacked object must be")
    args = ap.parse_args()

    print("=" * 70)
    print("Find_Moving_Objects - stack along the motion, then look")
    me = os.path.abspath(__file__)
    try:
        import datetime
        when = datetime.datetime.fromtimestamp(
            os.path.getmtime(me)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        when = "unknown"
    print(f"  running: {me}")
    print(f"  last changed: {when}")
    print("=" * 70)

    if args.folder is None:
        args.folder = input("Enter path to the folder with the frames: ").strip().strip('"')
    if args.output_dir is None:
        d = os.path.join(args.folder, "Moving_Objects")
        v = input(f"Enter path for the results [{d}]: ").strip().strip('"')
        args.output_dir = v or d
    # Everything this writes goes into one sub-folder called result. Pointing
    # the tool at the folder of frames is the natural thing to do, and without
    # this the text files and pictures land among the frames themselves.
    args.output_dir = os.path.join(args.output_dir, "result")
    os.makedirs(args.output_dir, exist_ok=True)
    # Everything the screen shows is written down as well. A console scrolls
    # away and cannot be gone back to, and the identification - the one line
    # that says what the object actually is - was only ever on the screen.
    log_path = os.path.join(args.output_dir, "run_log.txt")
    sys.stdout = Tee(sys.stdout, log_path)
    print(f"  results will be written to {os.path.abspath(args.output_dir)}")

    targets = scan_folder(args.folder)
    if not targets:
        raise SystemExit("No plate-solved frames with an OBJECT keyword found.")
    for t, ps in sorted(targets.items()):
        print(f"\n{t}  -  {len(ps)} frames")
        if len(ps) < 6:
            print("  too few frames"); continue
        # Asked, not switched. --quick was a command-line flag, and this is run
        # from an editor with no arguments, so the one control that saves an
        # hour of reading was unreachable in practice. Everything else this
        # family of tools needs is asked for; so is this.
        quick = args.quick
        if not quick and len(ps) > 2 * args.quick_frames:
            while True:
                try:
                    ans = input(f"  A quick look first, on {2 * args.quick_frames} "
                                f"frames instead of {len(ps)}? [Y/n]: ").strip().lower()
                except EOFError:
                    ans = ""
                if ans in ("", "y", "yes"):
                    quick = True
                    break
                if ans in ("n", "no"):
                    break
                print(f"    '{ans}' is not y or n - please answer y or n "
                      f"(check the keyboard language)")
        if quick:
            try:
                h0 = fits.getheader(ps[0])
                if "CD1_1" not in h0:
                    h0 = fits.getheader(ps[0], 1)
                pxs = float(np.hypot(float(h0["CD1_1"]), float(h0["CD2_1"])) * 3600)
            except Exception:
                pxs = 1.0
            os.makedirs(args.output_dir, exist_ok=True)
            quick_session(ps, args, pxs,
                          os.path.join(args.output_dir,
                                       re.sub(r"[^A-Za-z0-9_]+", "_", t)))
            continue
        objs, frames, hours, mjds, ref_w, scratch = search(
            ps, psf_px=args.psf_px, threshold=args.threshold,
            min_snr=args.min_snr, max_per_frame=args.max_per_frame,
            min_significance=args.min_significance, max_wander=args.max_wander,
            max_rate_arcsec_h=args.max_rate)
        os.makedirs(args.output_dir, exist_ok=True)
        stem = os.path.join(args.output_dir, re.sub(r"[^A-Za-z0-9_]+", "_", t))
        try:
            px = float(np.hypot(ref_w.wcs.cd[0][0], ref_w.wcs.cd[1][0])) * 3600.0
        except Exception:
            px = 1.0
        if not objs:
            print("  nothing moved in a straight line at a steady rate")
        with open(stem + "_tracks.txt", "w", encoding="utf-8") as fh:
            fh.write(f"{t}" + NL + f"{len(frames)} frames over "
                     f"{hours[-1]:.3f} h, first MJD {mjds[0]:.6f}" + NL + NL)
            for k, o in enumerate(objs, 1):
                rate = float(np.hypot(o["vx"], o["vy"]) * px)
                ang = float(np.degrees(np.arctan2(o["vx"], o["vy"])) % 360)
                sc = ref_w.all_pix2world(o["x0"], o["y0"], 0)
                line = (f"object {k}: {rate:5.1f} arcsec/hour towards {ang:5.1f} deg | "
                        f"seen in {o['n']}/{len(frames)} frames | "
                        f"moved {o['moved_px'] * px:.1f} arcsec, "
                        f"{o['sig']:.0f}x its own {o['rms_px'] * px:.2f} arcsec scatter | "
                        f"S/N {o['snr_frame']:.1f} per frame, {o['snr']:.0f} stacked")
                pos = (f"          start RA {float(sc[0]):.6f} Dec {float(sc[1]):+.6f} "
                       f"at MJD {mjds[0]:.6f}   (pixel {o['x0']:.1f},{o['y0']:.1f})")
                print("  " + line)
                print(pos)
                fh.write(line + NL + pos + NL)
                who = name_it(float(sc[0]), float(sc[1]), float(mjds[0]), args)
                for _ln in who:
                    print("  " + _ln)
                fh.write("".join("          " + _ln + NL for _ln in who) + NL)
        draw_objects(objs, frames, hours, px, stem)
        out = stem + "_tracks.txt"
        print(f"  -> {out}")
        del frames
        try:
            os.remove(scratch)
        except OSError:
            pass


if __name__ == "__main__":
    main()
