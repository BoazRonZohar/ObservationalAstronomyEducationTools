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
| B and V frames of a star cluster | `Cluster_CMD.py` |
| B, V, R and H-alpha frames of a spiral galaxy | `Star_Formation_All_In_One.py` |

---

## Observation planning

Four calculators. **They need no installation at all** — download the file,
double-click it, and it opens in your browser. They work offline; the catalogues
are inside the files.

| Tool | What it answers |
|---|---|
| `Plan_Eclipsing_Binaries_Any_Telescope.html` | Which eclipsing binaries are in eclipse tonight, from *your* coordinates. 18,813 systems. |
| `Plan_Exoplanet_Transits_Any_Telescope.html` | Which exoplanets transit tonight, from your coordinates. 107 bright planets. |
| `Plan_Eclipsing_Binaries_LCO.html` | The same for the LCO network, plus ready-made observing windows and a downloadable request file. |
| `Plan_Exoplanet_Transits_LCO.html` | The same for transits. 4,477 planets. |

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

**Close companions are found and kept out.** A star finder never reports a
companion closer than about one star-width — the pair arrives as one stretched
star. The tools look at the light instead, the way an observer does by eye: one
star rises to a single peak, two stars give peak–dip–peak. When a companion is
found, the aperture *shrinks* as the seeing softens, instead of growing into
the companion; left to grow, it wrote 82 mmag of false variation into one light
curve. Neighbouring stars are kept out of the target's sky ring the same way.
On fields with neither, nothing changes.

**You choose the comparison stars after seeing what they did.** Every run
prints a table — each comparison star's brightness relative to the target and
how much it scattered, per channel — and saves it as `comparison_stars.txt`.
Run the tool again on the same folder and it offers to rebuild the light curve
with the stars you name, from the measurements already on disk: seconds, not
minutes, and the original run is kept. On a night with saturated exposures
this took one channel from 42 to 21 mmag of noise. No option to remember — the
tool asks.

`Photometry_Guide.txt` covers all of this in detail, including how to read the
comparison-star table, how to hand the script exact coordinates, and the two
accepted star-list formats: a four-column CSV — there is one to copy in
`star_list_example.csv` — or an AIP4Win *Star Data Tool* export, which needs no
editing at all.

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

**The quick look.** Reading a whole night off an external disk takes a long
time, so the tool first offers a quick look: 14 frames spread over the **whole**
run, all of them searched. If you ask for more, each pass doubles the sampling —
14, 27, 53, 105 … — until the sample is the entire run. The new frames fall
between the ones already looked at, and the time span never shrinks: how far a
track moved is what it is judged by, so the full arc is kept at every pass.

**Crowded fields.** Stars that stay put are set aside first, and *every*
remaining candidate is then searched — not only the brightest, because an
asteroid is usually among the faintest things left. In a rich Milky Way field
that means thousands of candidates per frame and a search of several minutes; a
counter shows how far it has got and roughly how long is left. `--max_per_frame`
can cap the number, but it keeps the brightest and throws away the faint end.

**Expect false alarms, and look.** The tool is tuned to miss as little as
possible, so a rich field can report a dozen or more "movers" crawling at a few
arcsec/hour in random directions — usually two close stars blended together,
whose combined centre drifts as the seeing changes. Every track comes with a
picture of the same field at the start, middle and end of the run: if the
circle moves and the dot under it does not, it is not an object.

Everything is written to a `result` folder: a `_tracks.txt` with each object's
rate, direction, position and identification, one picture per object, and a
`run_log.txt` of everything that appeared on the screen.

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
| `Galaxy_CMD.py` | The same for galaxies, and the radial density profile of the blue knots — the young star-forming regions in the arms. Kept for projects already built on it; for new work, see [Star formation in spiral galaxies](#star-formation-in-spiral-galaxies). |
| `Open_Cluster_Name_Resolver.py` | Turns `M6` into `NGC_6405` — the name the catalogue actually uses. `Cluster_CMD.py` now does this itself; useful on its own when a name needs checking. |
| `List_Catalogue_Clusters.py` | Prints every cluster name in the catalogue, for when a name is being rejected. |

`Cluster_CMD.py` asks for a folder. The filter of each frame is read from its
header, the cluster's name from `OBJECT`, whether it is open or globular from
SIMBAD, and its distance and reddening from the Harris catalogue for globulars
and the Dias catalogue for open clusters. Every number that is looked up is shown
with the catalogue it came from before it is used, and can be overridden with a
keystroke — a distance sets the whole vertical scale of the diagram, and a wrong
one moves every star together.

It draws each diagram twice. Aperture photometry is exact for a star on its own
and meaningless for two stars sharing the circle, which is what the centre of a
globular cluster is made of. So one diagram holds every measured source, and one
leaves out those with another source inside their own aperture; a map of the
frame beside each shows where the two differ. On M12 the second removes most of
the core; on M67 it removes a handful. Which one to believe is a judgement about
the cluster, and both are kept for it.

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

## Star formation in spiral galaxies

In `Galaxy_Analysis/`. Two tracers of star formation, and the link between them:
blue knots, which are young star clusters, and HII regions, the gas lit up around
stars that have only just switched on. They mark the same event about ten million
years apart, and the distance between a knot and the region it came from is the
cluster's drift made visible.

| Tool | |
|---|---|
| `Blue_Clusters_From_Images.py` | B and V frames → the blue knots, their colour-magnitude diagram, and their radial profile. |
| `HII_From_Images.py` | R and H-alpha frames → the H-alpha frame with the continuum taken out, the HII regions in it, and their radial profile. |
| `Star_Formation_All_In_One.py` | All four frames → both tracers on one grid, and the separation between each knot and its nearest region. |
| `Compare_Clusters_And_Regions.py` | Two output folders already written by the first two tools → the same comparison, without the frames. |

Each is one file that runs on its own. Given a folder, they read the filters from
the headers, combine the exposures, and look up the galaxy's distance, reddening
and geometry from published catalogues.

Two things set these apart from the older `Galaxy_CMD.py`. Foreground stars are
removed using Gaia parallaxes and proper motions rather than by shape: a cluster
several megaparsecs away is unresolved and looks exactly like a star, so anything
that removes star-shaped sources removes the clusters first. And the amount of
continuum to subtract from H-alpha, together with how much to blur one frame to
match the other, is chosen by looking: the tool writes three versions and asks
which is best, because every automatic rule tried disagreed with the eye.

The comparison tool works from finished results rather than frames, which makes
it useful for re-examining old projects. Its two inputs come from separate runs on
separate pixel grids, so it converts both catalogues to sky coordinates and
measures the separation as an angle; nothing is resampled and neither grid is
preferred.

Radial profiles from all of them are drawn with and without a correction for the
galaxy's tilt, each as its own figure, with the numbers behind every figure
written beside it as a CSV.

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
to others and carry their own terms; see `NOTICE` for the list.

Created by **Dr. Boaz Ron Zohar**
Kinneret Observatory · Member of the LCO Global Sky Partners programme
