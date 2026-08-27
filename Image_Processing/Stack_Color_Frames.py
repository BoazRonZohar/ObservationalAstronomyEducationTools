#!/usr/bin/env python3
"""
Stack_Color_Frames.py
======================

Purpose: stack (integrate) tens to hundreds of colour FITS images that
have already been calibrated (bias + flat fielding) into one good image -
before further processing in PixInsight (stretch, DBE, deconvolution etc).

Memory-frugal architecture:
    Instead of holding every frame in RAM at once (which needs tens of GB
    on large sets), the script writes the aligned frames to a temporary
    memmap file on disk, and then combines them in bands of rows.
    RAM stays at a few hundred MB even with hundreds of frames.

Steps:
    1. First pass: load each file, measure "sharpness", keep the header. The
       data is released from memory at once. Frames whose pointing
       (OBJCTRA/OBJCTDEC) is well away from the rest are rejected (a slew fault).
    2. Choose a reference frame (the image with the most detail / sharpest
       stars, among the frames that passed the pointing check).
    3. Second pass: per-frame cosmetic correction (removing hot pixels and
       cosmic ray hits against the local neighbourhood), then align every
       frame to the reference and write it to the temporary memmap. Alignment
       is by stars (astroalign), and if that fails it falls back to phase
       cross-correlation, which does not need star detection at all.
    4. Combine with sigma clipping in bands of rows, ignoring NaN
       (areas left uncovered by the alignment shift).
    5. Crop to the area with full coverage, and copy the original header
       (WCS, OBJECT, FILTER, cumulative EXPTIME etc) to the output file.

Usage:
    python Stack_Color_Frames.py
    python Stack_Color_Frames.py --input "F:\\Lights" --output "F:\\stacked.fits"

Required dependencies:
    pip install astropy astroalign numpy scipy scikit-image

----------------------------------------------------------------------
Created by: Dr. Boaz Ron Zohar
Affiliation: Kinneret Observatory
Member of the LCO Global Sky Partners programme
Part of Observational Astronomy Education Tools
https://github.com/BoazRonZohar/ObservationalAstronomyEducationTools
"""

import argparse
import sys
import tempfile
import warnings
from pathlib import Path, PureWindowsPath

import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation

try:
    import astroalign as aa
except ImportError:
    print("Error: astroalign is not installed. Run: pip install astroalign", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------
# Path input cleaning - handles the common variations, above all "Copy as path" on Windows
# --------------------------------------------------------------------------

def clean_path_input(raw: str):
    """
    Clean and normalise a path the user pastes or types, coping with:
      - single or double quotes wrapping the path ("Copy as path" on Windows)
      - stray spaces at the start or the end
      - backslashes (\\) alongside forward slashes (/), even mixed
      - paths containing ~ (the user's home)
      - a network UNC path (\\\\server\\share\\...)
      - a URI-form path, for example file:///F:/folder
    """
    if raw is None:
        return None

    s = raw.strip()

    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()

    if not s:
        return None

    if s.lower().startswith("file:///"):
        s = s[8:]
    elif s.lower().startswith("file://"):
        s = s[7:]

    if not s:
        return None

    looks_like_windows = ("\\" in s) or (len(s) >= 2 and s[1] == ":" and s[0].isalpha())

    if looks_like_windows:
        win_path = PureWindowsPath(s)
        path = Path(*win_path.parts)
    else:
        path = Path(s)

    return path.expanduser().resolve()


def resolve_output_path(raw_output: str, input_dir: Path) -> Path:
    """
    Work out the output file path from what the user typed:
      1. If only a bare filename was typed with no folder, save it inside
         the input folder (not in Python's current working directory,
      2. If a known FITS extension is missing, add ".fits" automatically.
    """
    raw_stripped = raw_output.strip().strip('"').strip("'")
    is_bare_filename = ("\\" not in raw_stripped) and ("/" not in raw_stripped)

    if is_bare_filename:
        output_path = (input_dir / raw_stripped).resolve()
    else:
        output_path = clean_path_input(raw_output)

    if output_path.suffix.lower() not in (".fits", ".fit", ".fts"):
        output_path = output_path.with_name(output_path.name + ".fits")
        print(f"  (added the .fits extension automatically: {output_path})")

    return output_path


def prompt_for_path(prompt_text: str, must_exist: bool = True) -> Path:
    """Ask the user for a path, clean it, and check it is valid (if must_exist=True)."""
    while True:
        raw = input(prompt_text).strip()
        path = clean_path_input(raw)
        if path is None:
            print("  Empty path, try again.")
            continue
        if must_exist and not path.exists():
            print(f"  Path not found: {path}\n  Try again (with or without quotes).")
            continue
        return path


# --------------------------------------------------------------------------
# Loading FITS and working out the data layout
# --------------------------------------------------------------------------

def load_fits_as_rgb(filepath: Path):
    """
    Load a FITS file and return (rgb, header).
      - rgb: an (H, W, 3) array as float32. float32 and not float64 on purpose:
        it halves the memory, and the precision is more than enough for camera
        data (which arrives as 16-bit integers or float32 anyway).
      - header: the original header, for copying metadata to the output file.
    Supports the layouts (3, H, W), (H, W, 3) and monochrome (H, W).
    """
    with fits.open(filepath, memmap=False) as hdul:
        data = None
        header = None
        for hdu in hdul:
            if hdu.data is not None:
                data = hdu.data
                header = hdu.header
                break
        if data is None:
            raise ValueError(f"No valid data found in {filepath}")

        data = np.asarray(data, dtype=np.float32)

    if data.ndim == 2:
        rgb = np.stack([data, data, data], axis=-1)
    elif data.ndim == 3:
        if data.shape[0] == 3:
            rgb = np.ascontiguousarray(np.transpose(data, (1, 2, 0)))
        elif data.shape[-1] == 3:
            rgb = data
        else:
            raise ValueError(f"Unrecognised FITS layout in {filepath}: shape={data.shape}")
    else:
        raise ValueError(f"Unsupported number of dimensions in {filepath}: {data.ndim}")

    return rgb, header


# --------------------------------------------------------------------------
# Cosmetic correction - removing outlying pixels (hot pixels, cosmic rays)
# --------------------------------------------------------------------------

def cosmetic_correct(rgb: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """
    Replace single pixels that stand sharply out of their local 3x3
    neighbourhood, per channel, with the local median - the typical hot pixel

    Why this is needed on top of the between-frame sigma clipping at the
    combine step: after alignment every frame is shifted by a slightly
    different amount, so an outlying pixel in one frame does not necessarily
    "land" on the same pixel of the final result in every frame - and so it
    does not always look extreme against the others. Removing it per frame,

    Real stars are not harmed: the PSF of a real star (even undersampled)
    lifts its neighbouring pixels too, so the local median stays close to the
    central value and there is no sharp deviation. A single hot pixel or
    cosmic ray hit stands sharply against unlit neighbours, and is detected.
    """
    from scipy.ndimage import median_filter

    corrected = rgb.copy()
    for c in range(rgb.shape[-1]):
        channel = rgb[..., c]
        local_median = median_filter(channel, size=3)
        residual = channel - local_median
        robust_sigma = 1.4826 * (np.median(np.abs(residual)) + 1e-6)
        outliers = np.abs(residual) > sigma * robust_sigma
        corrected[outliers, c] = local_median[outliers]
    return corrected


# --------------------------------------------------------------------------
# Choosing the reference frame
# --------------------------------------------------------------------------

def estimate_sharpness(luminance: np.ndarray) -> float:
    """
    Estimate "sharpness" / amount of detail from the variance of the gradient.
    A sharp image with more stars -> a higher value.
    """
    gy, gx = np.gradient(luminance)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    return float(np.var(grad_mag))


# --------------------------------------------------------------------------
# Pointing consistency check - finding frames aimed at the wrong piece of sky
# --------------------------------------------------------------------------

def parse_pointing(header):
    """
    Pull (RA, Dec) in degrees out of the header, from the standard
    OBJCTRA/OBJCTDEC fields. Returns None if they are missing or unparseable.
    """
    if header is None:
        return None
    ra_raw = header.get("OBJCTRA")
    dec_raw = header.get("OBJCTDEC")
    if not ra_raw or not dec_raw:
        return None
    try:
        ra_deg = Angle(str(ra_raw), unit=u.hourangle).degree
        dec_deg = Angle(str(dec_raw), unit=u.deg).degree
        return ra_deg, dec_deg
    except Exception:
        return None


def find_pointing_outliers(headers, max_offset_deg):
    """
    Find frames pointing at a piece of sky significantly different from the
    median pointing of the rest - for example a frame caught during a slew,
    before the mount settled on the right target. Such a frame could (if it
    happens to be the "sharpest") be chosen as the reference by mistake and

    wreck the alignment of every other frame. Returns a set of indices (in
    header order) to skip. If fewer than half the frames carry valid pointing
    fields the whole check is skipped (not enough to judge an "outlier").
    """
    pointings = [parse_pointing(h) for h in headers]
    valid = [(i, p) for i, p in enumerate(pointings) if p is not None]
    if len(valid) < max(2, len(headers) // 2):
        return set()

    ras = np.array([p[0] for _, p in valid])
    decs = np.array([p[1] for _, p in valid])
    median_coord = SkyCoord(ra=np.median(ras) * u.deg, dec=np.median(decs) * u.deg)

    outliers = set()
    for i, (ra, dec) in valid:
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        if coord.separation(median_coord).degree > max_offset_deg:
            outliers.add(i)
    return outliers


# --------------------------------------------------------------------------
# Satellite trail detection - a long, thin, straight line across the frame
# --------------------------------------------------------------------------

def detect_trail(
    luminance: np.ndarray,
    downsample: int = 3,
    bg_sigma: float = 6.0,
    thresh_mad: float = 3.0,
    min_len_frac: float = 0.35,
    min_eccentricity: float = 0.98,
) -> bool:
    """
    Detect a frame with a satellite trail - a straight thin line crossing much of it.

    Method: shrink the image (block-max pooling, not a plain average or
    subsample - so a line a few pixels wide is not lost), subtract a smooth
    background (a large Gaussian, to cancel nebulosity and vignetting), and
    look for connected components in the mask above the threshold. Stars
    (even bright ones, even with blooming) are compact components - fairly low
    eccentricity, short. A real trail is unusually thin and long

    This does not catch stars "smeared" by a tracking fault within a single
    exposure (short streaks spread over the whole frame, not one long line) -
    a different problem, needing real star-shape measurement. Not done here.
    """
    from skimage.measure import block_reduce, label, regionprops
    from scipy.ndimage import gaussian_filter

    small = block_reduce(luminance, block_size=(downsample, downsample), func=np.max)
    bg = gaussian_filter(small, sigma=bg_sigma)
    residual = small - bg

    med = np.median(residual)
    mad = np.median(np.abs(residual - med)) + 1e-6
    threshold = med + thresh_mad * 1.4826 * mad
    mask = residual > threshold

    diag = np.hypot(*small.shape)
    for region in regionprops(label(mask, connectivity=2)):
        if (region.major_axis_length > min_len_frac * diag
                and region.eccentricity > min_eccentricity):
            return True
    return False


def find_trail_frames(luminances, **kwargs) -> set:
    """Run detect_trail over a list of luminance images and return the positive indices."""
    return {i for i, lum in enumerate(luminances) if detect_trail(lum, **kwargs)}


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

def register_via_cross_correlation(rgb: np.ndarray, reference_luminance: np.ndarray):
    """
    Alignment that does not depend on star detection at all: find the global
    shift (dy, dx) between the images with phase cross-correlation (FFT
    based), and shift every colour channel by exactly the same amount.

    Far more robust than triangle matching when the number or quality of the
    detected stars varies between frames - as long as the shift is mostly a
    translation (no significant rotation), the usual case on a tracking mount.
    """
    luminance = rgb.mean(axis=-1)
    try:
        shift_yx, _, _ = phase_cross_correlation(
            reference_luminance, luminance, upsample_factor=10
        )
    except Exception as e:
        print(f"    cross-correlation failed ({e})")
        return None

    aligned = np.empty_like(rgb)
    for c in range(3):
        # order=1 (linear) on purpose and not a higher-order spline: a spline
        # does recursive prefiltering that can smear a single NaN across the
        # whole image. Linear keeps the NaN confined to the shifted edges.
        aligned[..., c] = ndi_shift(
            rgb[..., c], shift=shift_yx, order=1, mode="constant", cval=np.nan
        )
    return aligned


def register_rgb_frame(rgb: np.ndarray, reference_luminance: np.ndarray):
    """
    Align an RGB image to the reference.
    Tries astroalign first (triangle matching - it handles rotation and flip),
    and falls back to cross-correlation (translation only, but more robust).
    Returns None if both methods failed.
    """
    luminance = rgb.mean(axis=-1)
    try:
        # find_transform (not register!) is the one that returns a transform
        # object. aa.register returns (aligned image, footprint) - using that
        # here was a bug that meant astroalign in fact never succeeded.
        transform, _ = aa.find_transform(
            luminance, reference_luminance, detection_sigma=5
        )
    except Exception as e:
        print(f"    astroalign failed ({e}) - falling back to cross-correlation")
        return register_via_cross_correlation(rgb, reference_luminance)

    try:
        aligned = np.empty_like(rgb)
        for c in range(3):
            aligned[..., c] = aa.apply_transform(
                transform, rgb[..., c], reference_luminance, fill_value=np.nan
            )[0]
        return aligned
    except Exception as e:
        print(f"    applying the transform failed ({e}) - falling back to cross-correlation")
        return register_via_cross_correlation(rgb, reference_luminance)


# --------------------------------------------------------------------------
# Stacking with sigma clipping - in bands, frugal with memory
# --------------------------------------------------------------------------

def stack_with_rejection_banded(
    mm: np.memmap, sigma: float = 3.0, band_rows: int = 128, combine: str = "mean"
) -> np.ndarray:
    """
    Combine the aligned frames (from the memmap on disk) with sigma clipping,
    per pixel and per channel, ignoring NaN (uncovered pixels).

    The work is done in bands of rows and not on the whole image at once, so
    RAM stays low even with hundreds of high-resolution frames.
    In principle the same as "Average with rejection" (combine="mean") or
    "Median" (combine="median") in PixInsight ImageIntegration.

    combine="median" is much more robust to isolated streaks (a satellite
    trail, a cosmic ray hit) that the sigma clipping did not fully reject - a
    median over tens of frames is barely affected by them, with no threshold
    to tune. The price: slightly more statistical noise than combine="mean"
    (a median "wastes" some of the information in the good frames, unlike a
    mean). The default stays "mean"; choose "median" explicitly.
    """
    n_frames, height, width, channels = mm.shape
    result = np.empty((height, width, channels), dtype=np.float32)

    for r0 in range(0, height, band_rows):
        r1 = min(r0 + band_rows, height)
        band = np.array(mm[:, r0:r1, :, :])  # (N, band_h, W, 3)

        masked = np.ma.masked_invalid(band)
        clipped = sigma_clip(masked, sigma=sigma, axis=0, masked=True)
        if combine == "median":
            band_combined = np.ma.median(clipped, axis=0)
        else:
            band_combined = np.ma.mean(clipped, axis=0)

        # Pixels rejected or NaN in every frame - filled with the median, and
        # if that is undefined too (no coverage at all) with zero, so no NaN is left.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fallback = np.nanmedian(band, axis=0)
        fallback = np.nan_to_num(fallback, nan=0.0)

        result[r0:r1] = np.ma.filled(band_combined, fallback).astype(np.float32)

        pct = 100.0 * r1 / height
        print(f"\r  combining... {pct:5.1f}%", end="", flush=True)

    print()
    return result


def crop_to_coverage(stacked_rgb: np.ndarray, coverage: np.ndarray):
    """
    Crop to the rectangle where every frame has full coverage (no NaN).
    This removes the "thinned" edges produced by the alignment shift, which
    otherwise show up as rectangles of a different brightness (a classic

    Returns (cropped, r0, c0) - r0/c0 are the offset of the start of the crop
    (row/column), needed to fix CRPIX1/CRPIX2 in the header (otherwise the WCS
    copied from the reference frame points at the wrong pixel after the crop -
    it refers to a place in the original frame, not in the cropped image).
    """
    if not coverage.any():
        print("  Warning: no area has full coverage from every frame - skipping the crop")
        return stacked_rgb, 0, 0

    rows = np.any(coverage, axis=1)
    cols = np.any(coverage, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    cropped = stacked_rgb[r0:r1 + 1, c0:c1 + 1, :]
    print(f"  cropped from {stacked_rgb.shape[:2]} to {cropped.shape[:2]} (full coverage only)")
    return cropped, int(r0), int(c0)


# --------------------------------------------------------------------------
# Saving as FITS ready for PixInsight, with the original header copied over
# --------------------------------------------------------------------------

def save_stacked_fits(
    stacked_rgb: np.ndarray,
    output_path: Path,
    n_frames: int,
    source_header=None,
    total_exptime=None,
    crop_offset=(0, 0),
):
    """
    Save as 32-bit float FITS in (3, H, W) layout - the format most astronomy
    software (PixInsight included) expects.

    If source_header is passed (normally the reference frame's), every field
    in it is copied: WCS/RA/DEC, OBJECT, FILTER, TELESCOP, INSTRUME,
    CCD-TEMP, DATE-OBS and so on - critical so that plate solving and
    astrometry in PixInsight can work and do not fail on an empty header.

    The data-layout fields (NAXIS*, BITPIX, SIMPLE, EXTEND, BSCALE, BZERO)
    are always set again from the actual data, so they cannot contradict it.

    crop_offset=(r0, c0): the offset of the start of the crop
    (crop_to_coverage) relative to the reference's original frame. If the
    source carries a real WCS (CRPIX1/2) it refers to a pixel in the original
    frame, not the cropped image - without the adjustment here the WCS would
    """
    data_chw = np.ascontiguousarray(
        np.transpose(stacked_rgb, (2, 0, 1)).astype(np.float32)
    )
    hdu = fits.PrimaryHDU(data=data_chw)

    if source_header is not None:
        structural_keys = {
            "SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3",
            "EXTEND", "BSCALE", "BZERO",
        }
        copied = 0
        for card in source_header.cards:
            key = card.keyword
            if not key or key in structural_keys:
                continue
            try:
                if key in ("HISTORY", "COMMENT"):
                    hdu.header[key] = card.value
                else:
                    hdu.header[key] = (card.value, card.comment)
                copied += 1
            except Exception:
                continue
        print(f"  copied {copied} header fields from the reference frame")

        r0, c0 = crop_offset
        if (r0 or c0) and "CRPIX1" in hdu.header and "CRPIX2" in hdu.header:
            hdu.header["CRPIX1"] = hdu.header["CRPIX1"] - c0
            hdu.header["CRPIX2"] = hdu.header["CRPIX2"] - r0
            print(f"  fixed CRPIX1/CRPIX2 for the crop offset (r0={r0}, c0={c0})")

    if total_exptime is not None:
        hdu.header["EXPTIME"] = (
            total_exptime, "Total exposure of all combined frames (s)"
        )

    hdu.header["NCOMBINE"] = (n_frames, "Number of frames combined")
    hdu.header["HISTORY"] = "Stacked with Stack_Color_Frames.py (sigma-clip average)"

    hdu.writeto(output_path, overwrite=True)


# --------------------------------------------------------------------------
# main pipeline
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stack calibrated colour FITS images."
    )
    parser.add_argument("--input", default=None,
                        help="folder holding the colour FITS files (lights)")
    parser.add_argument("--output", default=None, help="path to the output file (.fits)")
    parser.add_argument("--sigma", type=float, default=3.0,
                        help="sigma clipping threshold (default 3.0)")
    parser.add_argument("--band-rows", type=int, default=128,
                        help="rows per processing band. Smaller = less RAM (default 128)")
    parser.add_argument("--max-pointing-offset-deg", type=float, default=1.0,
                        help="maximum deviation (in degrees) from the median pointing of "
                             "all the frames (from OBJCTRA/OBJCTDEC); an outlying frame is "
                             "skipped before the reference is chosen (default 1.0)")
    parser.add_argument("--no-cosmetic-correction", action="store_true",
                        help="turn off per-frame hot pixel / cosmic ray removal "
                             "before alignment (default: on)")
    parser.add_argument("--combine", choices=("mean", "median"), default=None,
                        help="how to combine after sigma clipping: mean (slightly "
                             "better SNR) or median (much more robust to streaks and "
                             "satellite trails that were not fully rejected). If not "
                             "given you are asked (default on pressing Enter: mean)")
    parser.add_argument("--no-trail-detection", action="store_true",
                        help="turn off automatic detection of frames with a long "
                             "sharp line (meteor/satellite/aircraft) and their "
                             "exclusion from the combine (default: on)")
    args = parser.parse_args()

    # --- input folder ---
    if args.input:
        input_dir = clean_path_input(args.input)
        if input_dir is None or not input_dir.exists():
            print(f"The path given to --input was not found: {args.input}", file=sys.stderr)
            sys.exit(1)
    else:
        input_dir = prompt_for_path("Paste or type the path to the FITS (lights) folder: ")

    if not input_dir.is_dir():
        print(f"The path given is not a folder: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # --- output path ---
    raw_output = args.output or input(
        "Paste or type a path for the output file (for example stacked.fits, or "
        "just a filename, to save it inside the input folder): "
    )
    output_path = resolve_output_path(raw_output, input_dir)

    if not output_path.parent.exists():
        print(f"The destination folder does not exist: {output_path.parent}", file=sys.stderr)
        sys.exit(1)
    if output_path.is_dir():
        print(f"The output path given is an existing folder: {output_path}\n"
              f"Add a filename, for example: {output_path / 'stacked.fits'}",
              file=sys.stderr)
        sys.exit(1)

    # --- combine method (mean/median) - asked only if --combine was not given ---
    if args.combine is None:
        raw_combine = input(
            "Combine method - mean or median? median is more robust to satellite "
            "trails and streaks that were not fully rejected [Enter=mean]: "
        ).strip().lower()
        args.combine = raw_combine if raw_combine in ("mean", "median") else "mean"

    fits_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".fits", ".fit", ".fts")
        and p.resolve() != output_path.resolve()
    )
    if not fits_files:
        print(f"No FITS files found in the folder {input_dir}", file=sys.stderr)
        sys.exit(1)

    # ==================== pass 1: scan and choose the reference ====================
    print(f"Found {len(fits_files)} FITS files.")
    print("First pass: scanning and choosing a reference frame...")

    scores = []
    headers = []
    shapes = []
    valid_files = []
    trail_flags = []

    for i, f in enumerate(fits_files):
        try:
            rgb, header = load_fits_as_rgb(f)
        except Exception as e:
            print(f"  skipping {f.name}: {e}")
            continue
        luminance = rgb.mean(axis=-1)
        scores.append(estimate_sharpness(luminance))
        headers.append(header)
        shapes.append(rgb.shape)
        valid_files.append(f)
        trail_flags.append(
            False if args.no_trail_detection else detect_trail(luminance)
        )
        del rgb  # released at once - this is what makes hundreds of frames possible
        print(f"\r  scanned {i + 1}/{len(fits_files)}", end="", flush=True)

    print()
    if len(valid_files) < 2:
        print("At least 2 valid images are needed to combine.", file=sys.stderr)
        sys.exit(1)

    outlier_idxs = find_pointing_outliers(headers, args.max_pointing_offset_deg)
    if outlier_idxs:
        print(f"  Warning: {len(outlier_idxs)} frames point at a different piece of sky "
              f"from the rest (deviation > {args.max_pointing_offset_deg}°) - they will be skipped:")
        for i in sorted(outlier_idxs):
            print(f"    - {valid_files[i].name}")
        keep = [i for i in range(len(valid_files)) if i not in outlier_idxs]
        scores = [scores[i] for i in keep]
        headers = [headers[i] for i in keep]
        shapes = [shapes[i] for i in keep]
        trail_flags = [trail_flags[i] for i in keep]
        valid_files = [valid_files[i] for i in keep]

        if len(valid_files) < 2:
            print("At least 2 valid images (with consistent pointing) are needed.",
                  file=sys.stderr)
            sys.exit(1)

    trail_idxs = {i for i, flagged in enumerate(trail_flags) if flagged}
    if trail_idxs:
        print(f"  Warning: {len(trail_idxs)} frames carry a long sharp line "
              f"(most likely a meteor, satellite or aircraft) - they will be skipped:")
        for i in sorted(trail_idxs):
            print(f"    - {valid_files[i].name}")
        keep = [i for i in range(len(valid_files)) if i not in trail_idxs]
        scores = [scores[i] for i in keep]
        headers = [headers[i] for i in keep]
        shapes = [shapes[i] for i in keep]
        valid_files = [valid_files[i] for i in keep]

        if len(valid_files) < 2:
            print("At least 2 valid images (with no crossing lines) are needed.",
                  file=sys.stderr)
            sys.exit(1)

    ref_idx = int(np.argmax(scores))
    ref_shape = shapes[ref_idx]
    reference_header = headers[ref_idx]
    print(f"  chosen: {valid_files[ref_idx].name}  (size {ref_shape[0]}x{ref_shape[1]})")

    # Frames of a different size from the reference cannot be aligned - filtered out
    usable = [i for i, s in enumerate(shapes) if s == ref_shape]
    if len(usable) < len(valid_files):
        print(f"  Warning: {len(valid_files) - len(usable)} frames are a different "
              f"size from the reference - they will be skipped")

    height, width, _ = ref_shape
    n_usable = len(usable)

    est_gb = n_usable * height * width * 3 * 4 / (1024 ** 3)
    print(f"  Temporary disk space needed: about {est_gb:.2f} GB "
          f"(RAM stays low thanks to band processing)")

    # ==================== pass 2: align and write to the memmap ====================
    reference_rgb, _ = load_fits_as_rgb(valid_files[ref_idx])
    if not args.no_cosmetic_correction:
        reference_rgb = cosmetic_correct(reference_rgb)
    reference_luminance = reference_rgb.mean(axis=-1)

    # The temporary file is kept beside the output file (usually a drive with room), not on C:
    tmp_dir = output_path.parent
    tmp_file = tempfile.NamedTemporaryFile(
        prefix=".stack_tmp_", suffix=".dat", dir=str(tmp_dir), delete=False
    )
    tmp_path = Path(tmp_file.name)
    tmp_file.close()

    coverage = np.ones((height, width), dtype=bool)
    included_headers = []
    n_written = 0

    try:
        mm = np.memmap(tmp_path, dtype=np.float32, mode="w+",
                       shape=(n_usable, height, width, 3))

        print("Second pass: aligning and writing to disk...")
        for i in usable:
            f = valid_files[i]
            if i == ref_idx:
                aligned = reference_rgb
            else:
                try:
                    rgb, _ = load_fits_as_rgb(f)
                except Exception as e:
                    print(f"  [{f.name}] loading failed: {e}")
                    continue
                if not args.no_cosmetic_correction:
                    rgb = cosmetic_correct(rgb)
                aligned = register_rgb_frame(rgb, reference_luminance)
                del rgb
                if aligned is None:
                    print(f"  [{f.name}] alignment failed - skipping")
                    continue

            mm[n_written] = aligned
            coverage &= ~np.isnan(aligned).any(axis=-1)
            included_headers.append(headers[i])
            n_written += 1
            del aligned
            print(f"\r  aligned and written {n_written}/{n_usable}", end="", flush=True)

        print()
        if n_written < 2:
            print("Fewer than 2 frames aligned successfully - nothing to combine.", file=sys.stderr)
            sys.exit(1)

        del reference_rgb, reference_luminance
        mm.flush()

        # ==================== combine ====================
        active = mm[:n_written]
        print(f"Combining {n_written} frames with sigma clipping (sigma={args.sigma}, "
              f"combine={args.combine})...")
        stacked = stack_with_rejection_banded(
            active, sigma=args.sigma, band_rows=args.band_rows, combine=args.combine
        )

        print("Cropping to the area with full coverage...")
        stacked, crop_r0, crop_c0 = crop_to_coverage(stacked, coverage)

        del mm, active

    finally:
        try:
            tmp_path.unlink()
        except Exception:
            print(f"  (could not delete the temporary file: {tmp_path})")

    # ==================== save ====================
    total_exptime = None
    exptimes = [h.get("EXPTIME") for h in included_headers
                if h is not None and h.get("EXPTIME") is not None]
    if exptimes:
        total_exptime = float(sum(exptimes))
        print(f"  Cumulative exposure time: {total_exptime:.0f} seconds "
              f"({total_exptime / 60:.1f} minutes)")

    print(f"Saving the result to {output_path}...")
    save_stacked_fits(stacked, output_path, n_frames=n_written,
                      source_header=reference_header, total_exptime=total_exptime,
                      crop_offset=(crop_r0, crop_c0))

    print("Done. The file can be opened directly in PixInsight for further processing.")


if __name__ == "__main__":
    main()
