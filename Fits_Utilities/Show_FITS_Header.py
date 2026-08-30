# -*- coding: utf-8 -*-
"""
Show_FITS_Header.py
==============

Prints EVERYTHING in a FITS header.

AIP4Win's Header tab only shows the HISTORY lines - the keyword list is not
displayed there at all. This prints the whole thing: every keyword, its value
and its comment, plus the HISTORY at the end.

Run it and give it a file:

    python Show_FITS_Header.py

or straight from the command line:

    python Show_FITS_Header.py "F:\\path\\to\\frame.fts"

It also writes the same text to a file NEXT TO THIS SCRIPT (never into the
folder the frames live in), so it can be opened in Notepad instead of read in
the console.

----------------------------------------------------------------------
Created by: Dr. Boaz Ron Zohar
Affiliation: Kinneret Observatory
Member of the LCO Global Sky Partners programme
Part of Observational Astronomy Education Tools
https://github.com/BoazRonZohar/ObservationalAstronomyEducationTools
"""

import os
import sys
from astropy.io import fits


def dump(path):
    lines = []
    with fits.open(path) as hdul:
        for n, hdu in enumerate(hdul):
            h = hdu.header
            shape = getattr(hdu.data, "shape", None)
            lines.append("=" * 70)
            lines.append(f"HDU {n}   shape={shape}   {len(h)} keywords")
            lines.append("=" * 70)
            hist, com = [], []
            for card in h.cards:
                k = card.keyword
                if k == "HISTORY":
                    hist.append(str(card.value))
                elif k == "COMMENT":
                    com.append(str(card.value))
                elif k == "":
                    continue
                else:
                    c = f"   / {card.comment}" if card.comment else ""
                    lines.append(f"{k:<9}= {card.value}{c}")
            if com:
                lines.append("")
                lines.append("--- COMMENT ---")
                lines += ["   " + c for c in com]
            if hist:
                lines.append("")
                lines.append("--- HISTORY (this is the part AIP4Win shows) ---")
                lines += ["   " + x for x in hist]
    return "\n".join(lines)


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter path to the FITS file: ").strip().strip('"').strip("'")
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        raise SystemExit(f"File not found: {path}")

    text = dump(path)
    print(text)

    # deliberately NOT next to the FITS file: nothing should ever be written
    # into an observation folder
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, os.path.splitext(os.path.basename(path))[0] + "_header.txt")
    try:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("\n" + "=" * 70)
        print(f"also written to: {out}")
    except Exception as e:
        print(f"\n(could not write the text file: {e})")


if __name__ == "__main__":
    main()
