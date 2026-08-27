# Observational Astronomy Education Tools

[![Licence: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![For education](https://img.shields.io/badge/for-teaching%20and%20research-blue.svg)](#licence)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

Tools for teaching astronomy with real telescope data: planning what to observe,
processing the frames that come back, measuring stars in them, and turning those
measurements into light curves.

They were written for school and undergraduate projects at Kinneret Observatory
and on the Las Cumbres Observatory network, and are used on real observations —
eclipsing binaries, exoplanet transits, star clusters, galaxies and asteroids.
Nothing here is a simulation.

Everything works from a folder of FITS frames and a browser. There is no server,
no account, and no upload: the planning tools run entirely inside your browser,
and the Python scripts run on your own machine, on your own files.

---

## Start here, depending on what you have

| You have | Use |
|---|---|
| Nothing yet — you want to know what is worth observing tonight | [Observation planning](#observation-planning) |
| Frames from your own telescope, with a plate solution | `Photometry_Transit_Eclipse_Color_Plate_Solved.py` (colour) |
| Frames from LCO (BANZAI `-e91.fits`) | `Photometry_Transit_Eclipse_Mono_Plate_Solved.py` |
| Frames with no plate solution — stacked in AIP4Win, WCS lost | `Photometry_Transit_Eclipse_Mono_Star_List.py` |
| A night of frames and a suspicion something moved | `Find_Moving_Objects.py` |
| Colour frames that need combining before anything else | `Stack_Color_Frames.py` |

---

## Observation planning

Four calculators. **They need no installation at all** — download the file,
double-click it, and it opens in your browser. They work offline; the catalogues
are inside the files.

| Tool | What it answers |
|---|---|
| `Plan_Eclipsing_Binaries_Any_Telescope.html` | Which eclipsing binaries are in eclipse tonight, from *your* coordinates. 19,935 systems. |
| `Plan_Exoplanet_Transits_Any_Telescope.html` | Which exoplanets transit tonight, from your coordinates. 107 bright planets. |
| `Plan_Eclipsing_Binaries_LCO.html` | The same for the LCO network, plus ready-made observing windows and a downloadable request file. |
| `Plan_Exoplanet_Transits_LCO.html` | The same for transits. 4,477 transits. |

Enter your latitude, longitude and a date; every time is given in your own local
clock and in UTC. The two `LCO` tools also build the JSON request you submit to
the network, working out the exposure count, block length, cadence and baseline
from the window and the exposure time you choose.

Each result carries its **timing uncertainty**, recomputed for the date you asked
about. An ephemeris drifts, and a prediction without an error bar is a guess.
Systems whose error cannot be quantified were removed from the catalogue rather
than shown with a made-up number.

**Data sources:** [Gaia DR3](https://gea.esac.esa.int/archive/)
(`vari_eclipsing_binary`), [GCVS 5.1](http://www.sai.msu.su/gcvs/gcvs/),
[AAVSO VSX](https://vsx.aavso.org/),
[VarAstro](https://var.astro.cz/en/Stars/MinimaPredictions),
[TEPCat](https://www.astro.keele.ac.uk/jkt/tepcat/) and the
[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).

---

## Photometry

Differential photometry of a variable star against an ensemble of comparison
stars, across tens to hundreds of frames. Three tools, differing only in what
kind of frames they take and how they find the stars in them.

| Tool | Frames | Finds the stars by |
|---|---|---|
| `Photometry_Transit_Eclipse_Color_Plate_Solved.py` | one-shot colour, three planes | the plate solution in each header |
| `Photometry_Transit_Eclipse_Mono_Plate_Solved.py` | monochrome, single filter | the plate solution in each header |
| `Photometry_Transit_Eclipse_Mono_Star_List.py` | monochrome, **no plate solution needed** | a star list you supply |

All three size their apertures from the FWHM measured in the data, model the
noise with the CCD equation, reject comparison stars that turn out to be
variable or drifting, and write a CSV, an Excel sheet, a light curve and a plot
per comparison star.

The two plate-solved tools measure every frame **twice**, by two independent
positioning methods, and compare the answers. A run where the two disagree is
marked `CHECK` rather than quietly published.

The summary reports one number that decides whether there is anything there:
how much smooth variation the target shows *against how much the comparison
stars show in the same frames*. Photometric noise is not white — airmass and
transparency wander slowly, and a slow wander looks exactly like a shallow
eclipse. If the comparison stars wander with your target, it was the atmosphere.

`Photometry_Guide.txt` covers all of this in detail, including how to hand the
script exact coordinates for a close pair, and the two accepted star-list
formats: a four-column CSV — there is one to copy in `star_list_example.csv` —
or an AIP4Win *Star Data Tool* export, which needs no editing at all.

---

## Moving objects

`Find_Moving_Objects.py` finds faint moving objects by **tracking** them rather
than detecting them: it keeps only detections as wide as the instrument's PSF,
requires three epochs on a uniform line rather than two, and then stacks the
frames along the implied velocity. An object too faint to see in any single
frame piles up; an accidental alignment does not.

Each track is checked against [SkyBoT](https://ssp.imcce.fr/webservices/skybot/),
which says what known solar system object was at that place at that moment. When
it names one, you get a link straight to its
[Minor Planet Center](https://www.minorplanetcenter.net/) entry. When it names
nothing, that is the interesting case.

---

## Image processing

| Tool | |
|---|---|
| `Stack_Color_Frames.py` | Combines tens to hundreds of calibrated colour frames into one. Star alignment with a phase-correlation fallback, per-frame hot-pixel and cosmic-ray removal, automatic rejection of trailed and mis-pointed frames, sigma clipping. Works in bands off a disk memmap, so RAM stays in the hundreds of MB even with hundreds of frames. |
| `Split_Color_Channels.py` | Writes the three colour planes out as separate FITS files, headers and WCS intact — for when another program needs them apart. |
| `Subtract_Sky_Background.py` | Puts every frame in a folder on the same sky baseline. |

---

## Looking inside a frame

`Show_FITS_Header.py` prints a FITS header in full — every keyword, its value
and its comment, and the HISTORY at the end. AIP4Win's Header tab shows only the
HISTORY lines, so when you need to know what GAIN, SATURATE or CRPIX2 actually
say, this is how you find out. It writes the same text to a file beside itself,
never into the folder your frames live in.

---

## Star clusters and galaxies

| Tool | |
|---|---|
| `Cluster_CMD.py` | Photometry and colour-magnitude diagrams for open and globular clusters: calibration, extinction correction, membership selection. Cluster membership comes from Gaia astrometry, not from whatever happens to lie in the same direction. |
| `Galaxy_CMD.py` | The same for galaxies, and the radial density profile of the blue knots — the young star-forming regions in the arms. |
| `Open_Cluster_Name_Resolver.py` | Turns `M6` into `NGC_6405` — the name the catalogue actually uses. Run it before `Cluster_CMD.py`. |
| `List_Catalogue_Clusters.py` | Prints every cluster name in the catalogue, for when a name is being rejected. |

Finding star-forming regions by their colour in ordinary broad-band images,
and treating their distribution as a measurable property of the galaxy,
follows Brosch, N. (1992), *Star formation systematics from colour images*,
Astrophysics and Space Science 188, 289–298
([doi:10.1007/BF00644916](https://doi.org/10.1007/BF00644916)).

What makes it suit a classroom is that it asks for nothing exotic: two
broad-band frames, B and V, are within reach of a school-accessible
telescope, and the young regions separate out on colour alone — no
spectroscopy, no narrow-band filter. A student with one night of data can ask
where a galaxy is forming stars, and answer it with a number.

---

## Talking to LCO

| Tool | |
|---|---|
| `LCO_Submit_Request.py` | Validates a request against the API, shows you what it will cost in allocation time, and submits it only with `--submit` and only after you confirm at the keyboard. |
| `LCO_Check_Status.py` | Read-only. Allocation used, every request group, which windows are left, and what the scheduler actually placed. |

Both read your token from the `LCO_TOKEN` environment variable, or ask for it
without echoing it. Nothing is stored in the files.

No request files are included: a request carries a proposal code, target
coordinates and observing windows, all of which are yours. Build one with a
planning tool above and put it in `LCO_API/`.

---

## Installing

The HTML planning tools need nothing. For the Python:

```
pip install -r requirements.txt
```

Python 3.8 or newer. Every script runs the same way — press F5 in Spyder, or run
it from the command line; it asks for what it needs, and every question has a
sensible default.

---

## A note on the numbers

Where a tool tells you something, it is because it was measured. The zero points
in the LCO tools came from cross-matching a real frame against APASS DR9. The
saturation thresholds are the values observed on each individual camera. The
5.41 s per-frame overhead was calibrated against the portal. The claim that
88.8% of independent exoplanet ephemerides agree to within a quarter of an hour
is the result of checking them.

Where something could not be measured, it is left out or flagged, not guessed.

---

## Licence

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) —
Attribution, NonCommercial, ShareAlike.

Use these tools, change them and teach with them freely. Three conditions:

- **Attribution** — credit the author and link to the licence.
- **NonCommercial** — not for commercial use. These are made for teaching.
- **ShareAlike** — if you share a changed version, share it under this same
  licence, so it stays available to the next teacher.

The licence covers the tools. The catalogues and services they draw on belong
to others and carry their own terms; see `LICENSE` for the list.

Created by **Dr. Boaz Ron Zohar**
Kinneret Observatory · Member of the LCO Global Sky Partners programme
