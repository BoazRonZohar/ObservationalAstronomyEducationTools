isochrones_PARSEC_UBV.csv.gz -- PARSEC isochrones in Johnson B and V for Cluster_CMD
====================================================================================

What it is
----------
A gzip-compressed CSV table of theoretical isochrones (absolute V magnitude and
B-V colour as a function of initial mass) from the PARSEC v1.2S stellar
evolution models, for fitting observed cluster colour-magnitude diagrams.
It was built by make_isochrone_table.py from the raw output files of the
Padova CMD web interface.

Source and exact settings
-------------------------
Web interface : CMD 3.9, https://stev.oapd.inaf.it/cgi-bin/cmd
                (form last modified Tue Nov 25 09:52:10 2025)
Downloaded    : 2026-10-07 (CMD output timestamps 09:23-09:30 UTC)
Tracks        : PARSEC version 1.2S + COLIBRI S_37 (default COLIBRI choice;
                CMD uses S_37 + S_35 + PR16 depending on Z)
                n_inTPC = 10 (default); no post-AGB/WD (default)
Mass loss     : Reimers on the RGB, eta = 0.2
Photometry    : "UBVRIJHK (cf. Maiz-Apellaniz 2006 + Bessell 1990)", Vega mags,
                bolometric corrections "YBC + new Vega" (form default;
                Chen et al. 2019, Vega SED of Bohlin et al. 2020)
Circumstellar dust : none ("No dust" for both M and C stars). Note: the form's
                default is 60% silicate + 40% AlOx / 85% AMC + 15% SiC; dust
                only affects TP-AGB stars, which are not in this table anyway.
Extinction    : A_V = 0
LPV           : option 4 (default; not used here)
IMF           : Kroupa (2001, 2002) canonical, corrected for unresolved binaries
                (default; does not affect the columns kept here)
Output        : isochrone tables with evolutionary stage label

Grid
----
[M/H]     : -2.19, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2,
            0.0, +0.2, +0.4  (14 values; -2.19 corresponds to Z = 0.0001, the
            lowest value covered by the v1.2S tracks -- lower inputs are
            accepted by the form but are clamped to the same tracks)
log(age/yr): 6.6 to 7.9 in steps of 0.1, then 8.00 to 10.12 in steps of 0.02
            (121 values)
Total     : 1694 isochrones, 346256 rows (file size about 2.6 MB)

The form was queried with the [M/H] input (not Z), one request per [M/H] and
age range (28 requests). The raw outputs are kept in raw/ next to the script.

Columns
-------
Lines starting with '#' at the top are comments. Then a header row:
  MH,logAge,label,Mini,MV,BV
MH     [M/H], 3 decimals
logAge log10(age/yr), 3 decimals
label  PARSEC evolutionary stage: 0 pre-main sequence (see below),
       1 main sequence, 2 subgiant (Hertzsprung gap), 3 red giant branch,
       4 core He burning (1st part: HB / red clump), 5 core He burning
       (2nd part, blue loop), 6 core He burning (3rd part), 7 early AGB
Mini   initial mass in solar masses, 5 significant digits
MV     absolute V magnitude (Vmag), 3 decimals
BV     B - V (Bmag - Vmag), 3 decimals

Rows are grouped by isochrone (MH, then logAge, increasing) and keep the CMD's
own order inside each isochrone (increasing Mini).

Selection and thinning
----------------------
- Only labels 0 to 7 are kept (dropped: 8 TP-AGB, 9 post-AGB).
  Label 0 is kept on purpose. PARSEC calls a low-mass star "pre-main
  sequence" until it has burnt a little of its hydrogen, which for stars
  below the Sun's mass takes far longer than reaching the main sequence:
  at [M/H] = 0 and 1 Gyr everything below 0.75 solar masses is still
  label 0. Without it the younger isochrones (at [M/H] = 0, every one
  younger than about 4 Gyr) would have no lower main sequence.
- Only points with MV <= 12.
- Inside each isochrone, the first and last point of each run of equal label
  are always kept; any other point is kept only if it lies at least 0.01 mag
  from the last kept point in the (BV, MV) plane.

Spot checks
-----------
[M/H] = 0.00, logAge = 9.60 : turnoff (bluest label-1 point) MV = 3.336,
                              B-V = 0.503; median MV of label-4 points = 0.71
[M/H] = -2.19, logAge = 10.10: turnoff MV = 3.878, B-V = 0.367;
                              median MV of label-4 points = 0.46

Citations
---------
As listed by CMD 3.9 in the output header ("Basic references"):
  Bressan et al. (2012), MNRAS, 427, 127            (PARSEC)
  Chen et al. (2014), MNRAS, 444, 2525
  Chen et al. (2015), MNRAS, 452, 1068
  Tang et al. (2014), MNRAS, 445, 4287
  Marigo et al. (2017), ApJ, 835, 77
  Pastorelli et al. (2019), MNRAS, 485, 5666
  Pastorelli et al. (2020), MNRAS, 498, 3283   (listed by CMD as "in press")
For the YBC bolometric corrections:
  Chen et al. (2019), A&A, 632, A105
For the photometric system:
  Maiz-Apellaniz (2006), AJ, 131, 1184; Bessell (1990), PASP, 102, 1181

Regenerating
------------
  python make_isochrone_table.py [folder of raw files]
reads MH*_age*.dat(.gz) from that folder (default: raw/ next to the script)
and rewrites isochrones_PARSEC_UBV.csv.gz. The raw CMD outputs are not in the
repository; download them again from the CMD web interface with the settings
above, one file per [M/H] and age range, named MH<value>_age<range>.dat.
