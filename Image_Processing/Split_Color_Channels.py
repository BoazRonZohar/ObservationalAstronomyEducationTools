# -*- coding: utf-8 -*-
"""
Split_Color_Channels.py

Writes the three colour planes of a one-shot-colour frame out as three
separate FITS files.

A colour frame holds its red, green and blue planes in one file. Some tools
want them apart: AIP4Win, PixInsight and most photometry packages measure one
plane at a time. This is the step you would otherwise do by hand in Fitswork,
done for a whole folder tree at once.

    NGC225-001.fts   ->   NGC225-001_R.fts
                          NGC225-001_G.fts
                          NGC225-001_B.fts

Each output keeps the original header, with CHANNEL set to R, G or B so you can
tell them apart later. Files that already end in _R, _G or _B are skipped, so
running it twice on the same folder does nothing the second time.

Both plane layouts are recognised, (3, H, W) and (H, W, 3), and anything that
is not a three-plane image is reported and passed over.

Note: you do NOT need this before the photometry tools in this collection.
Photometry_Transit_Eclipse_Color_Plate_Solved.py reads the three planes
straight out of the colour frame in memory. Use this when you want the
channels as files, to open in some other program.

Usage:
    python Split_Color_Channels.py
    python Split_Color_Channels.py --folder "D:\\Clusters\\NGC 225"

Every sub-folder below the one you give is searched.

----------------------------------------------------------------------
Created by: Dr. Boaz Ron Zohar
Affiliation: Kinneret Observatory
Member of the LCO Global Sky Partners programme
Part of Observational Astronomy Education Tools
https://github.com/BoazRonZohar/ObservationalAstronomyEducationTools
"""

import argparse
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

# One-shot-colour frames come out of different programs with different
# extensions; all of them are the same thing here.
FITS_SUFFIXES = (".fts", ".fits", ".fit")


def is_split_file(path: Path) -> bool:
    """Return True if file already looks like a split R/G/B file."""
    return path.stem.endswith(("_R", "_G", "_B"))


def split_single_fts(fts_path: Path):
    """Split one RGB frame into R, G, B files in the same folder."""
    if is_split_file(fts_path):
        return

    with fits.open(fts_path) as hdul:
        hdu = hdul[0]
        data = hdu.data
        header = hdu.header

    if data is None:
        print(f"Skipped (no data): {fts_path}")
        return

    if data.ndim != 3:
        print(f"Skipped (not 3D RGB): {fts_path}  shape={data.shape}")
        return

    if data.shape[0] == 3:
        r, g, b = data[0], data[1], data[2]
    elif data.shape[-1] == 3:
        r, g, b = data[..., 0], data[..., 1], data[..., 2]
    else:
        print(f"Skipped (no 3-channel axis): {fts_path}  shape={data.shape}")
        return

    stem = fts_path.stem
    suffix = fts_path.suffix
    r_path = fts_path.with_name(f"{stem}_R{suffix}")
    g_path = fts_path.with_name(f"{stem}_G{suffix}")
    b_path = fts_path.with_name(f"{stem}_B{suffix}")

    r_header = header.copy()
    g_header = header.copy()
    b_header = header.copy()

    r_header["CHANNEL"] = "R"
    g_header["CHANNEL"] = "G"
    b_header["CHANNEL"] = "B"

    fits.writeto(r_path, np.array(r, copy=False), header=r_header, overwrite=True)
    fits.writeto(g_path, np.array(g, copy=False), header=g_header, overwrite=True)
    fits.writeto(b_path, np.array(b, copy=False), header=b_header, overwrite=True)

    print(f"{fts_path.name} -> {r_path.name}, {g_path.name}, {b_path.name}")


def process_all_subfolders(root: Path):
    """Split every colour frame below `root`, in place."""
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)
        for filename in sorted(filenames):
            if not filename.lower().endswith(FITS_SUFFIXES):
                continue
            split_single_fts(dir_path / filename)
            n += 1
    return n


def clean_path(raw: str) -> Path:
    """Windows 'Copy as path' wraps the path in quotes; take them off."""
    s = (raw or "").strip().strip('"').strip("'").strip()
    return Path(s).expanduser() if s else None


def main():
    ap = argparse.ArgumentParser(
        description="Split one-shot-colour FITS frames into R, G and B files.")
    ap.add_argument("--folder", default=None,
                    help="folder with the colour frames; sub-folders are "
                         "searched too")
    args = ap.parse_args()

    folder = clean_path(args.folder) if args.folder else None
    while folder is None or not folder.is_dir():
        if folder is not None:
            print(f"Not a directory: {folder}")
        try:
            folder = clean_path(input("Paste or type the folder with the "
                                      "colour frames: "))
        except (EOFError, KeyboardInterrupt):
            print()
            return 1

    print(f"Searching {folder} and everything below it...")
    n = process_all_subfolders(folder)
    if n == 0:
        print("No FITS files found. Looked for: " + ", ".join(FITS_SUFFIXES))
    else:
        print(f"Done. {n} file(s) examined.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
