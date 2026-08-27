# -*- coding: utf-8 -*-
"""
Subtract_Sky_Background.py

Puts the sky background of every frame in a folder on the same footing.

Each frame is shifted so its median sits at exactly 5.0 ADU, with BZERO and
any PEDESTAL neutralised first so the header cannot skew the arithmetic. A
run of frames taken as the sky brightened or a gradient drifted then shares
one baseline, which is what a differential measurement across the run needs.

----------------------------------------------------------------------
Created by: Dr. Boaz Ron Zohar
Affiliation: Kinneret Observatory
Member of the LCO Global Sky Partners programme
Part of Observational Astronomy Education Tools
https://github.com/BoazRonZohar/ObservationalAstronomyEducationTools
"""

import os
import glob
import numpy as np
from astropy.io import fits

def deal_with(single_filename, output_directory):
	try:
		with fits.open(single_filename, do_not_scale_image_data=True) as hdul:
			target_hdu = hdul[0] if hdul[0].data is not None else hdul[1]
			data = np.array(target_hdu.data, dtype=np.float32, copy=True)
			
			# Neutralize any header interference
			bzero = target_hdu.header.get('BZERO', 0)
			pedestal = target_hdu.header.get('PEDESTAL', 0)
			data = data + bzero + pedestal
			
			# 1. Bring median to exactly 5.0
			current_median = np.median(data)
			adjustment = current_median - 5.0
			data = data - adjustment
			
			# 2. Force Clip to prevent software confusion
			data = np.clip(data, -100, 65000)

			# --- THE ADDED PART: Verification ---
			verified_median = np.median(data)
			# ------------------------------------

			# 3. Save with a fresh header that carries over everything from YOUR
			#    header except the keys describing how the pixels were encoded.
			#    Those must NOT be copied: the data is written as float32 here, so
			#    a stale BZERO/BSCALE/BITPIX would be re-applied on write and
			#    corrupt every value. Everything else is kept - in particular the
			#    PinPoint plate solution (CD matrix, CRVAL/CRPIX, PLTSOLVD), which
			#    lets the photometry stage place stars by RA/Dec instead of having
			#    to match an anchor pattern, plus OBJCTRA/OBJCTDEC, AIRMASS, FWHM
			#    and ZMAG.
			encoding_keys = {
				'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
				'EXTEND', 'BZERO', 'BSCALE', 'BLANK', 'PEDESTAL',
				'DATAMIN', 'DATAMAX',
			}

			new_header = fits.Header()
			new_header['BITPIX'] = -32
			new_header['NAXIS'] = 2
			new_header['NAXIS1'] = data.shape[1]
			new_header['NAXIS2'] = data.shape[0]
			new_header['PEDESTAL'] = 0

			for card in target_hdu.header.cards:
				key = card.keyword
				if not key or key in encoding_keys:
					continue
				try:
					if key in ('HISTORY', 'COMMENT'):
						new_header.add_history(str(card.value)) if key == 'HISTORY' \
							else new_header.add_comment(str(card.value))
					else:
						new_header[key] = (card.value, card.comment)
				except Exception:
					pass

			new_header.add_history('Sky subtracted: median forced to 5.0 ADU')

			base_name = os.path.basename(single_filename)
			new_path = os.path.join(output_directory, "Sub_" + base_name)
			fits.writeto(new_path, data, new_header, overwrite=True)
			
			print(f"File: {base_name} | Target: 5.0 | Python Verified: {verified_median:.4f}")

	except Exception as e:
		print(f"Error: {str(e)}")

# Execution logic
dir_path = input("Enter path: ").strip().strip('"')
output_dir = os.path.join(dir_path, "Reduced")
if not os.path.exists(output_dir): os.makedirs(output_dir)
files = sorted([f for f in glob.glob(os.path.join(dir_path, "*.fit*")) if "Sub_" not in f])

for f in files:
	deal_with(f, output_dir)