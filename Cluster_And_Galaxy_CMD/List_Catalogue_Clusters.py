# -*- coding: utf-8 -*-
"""
List_Catalogue_Clusters.py

Prints every open cluster name in Cantat-Gaudin's catalogue (Vizier
J/A+A/640/A1) - the same list Cluster_CMD.py matches against.

Useful when a name is being rejected and you want to see what the catalogue
actually calls it. It downloads the full membership table, so give it a moment.

----------------------------------------------------------------------
Created by: Dr. Boaz Ron Zohar
Affiliation: Kinneret Observatory
Member of the LCO Global Sky Partners programme
Part of Observational Astronomy Education Tools
https://github.com/BoazRonZohar/ObservationalAstronomyEducationTools
"""

from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1
cat = Vizier.get_catalogs("J/A+A/640/A1")
members_table = cat[1]
print(set(members_table["Cluster"]))
