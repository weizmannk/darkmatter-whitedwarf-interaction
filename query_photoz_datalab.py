# Author: Igor Andreoni
##  pip install --upgrade noaodatalab

import os
import numpy as np
from tqdm.auto import tqdm

import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from dl import queryClient as qc



datapath = f"{os.path.dirname(os.path.realpath('__file__'))}/classified_unclassified_objet"

targetdir ='../ZTF'
if not os.path.isdir(targetdir):
    os.makedirs(targetdir)



def query_coords_ls(ra,dec,radius_arcsec=5,
                         radius_nuclear=1., catalog='dr8_north', datalab=True,
                         check_quality=True):
    '''Query the database to search for matches at the given coordinates'''

    #Crossmatch with photoz database
    if datalab is True:
        radius_deg = radius_arcsec / 3600.
        query = qc.query(sql=f"SELECT z_phot_median, z_phot_std, z_phot_l95, ra, dec, \
                             type, flux_z from ls_dr8.photo_z INNER JOIN ls_dr8.tractor \
                             ON ls_dr8.tractor.ls_id = ls_dr8.photo_z.ls_id \
                             where 't' = Q3C_RADIAL_QUERY(ra, dec, {ra}, {dec}, {radius_deg})")

        result0 = query.split('\n')
        result0 = [r.split(",") for r in result0][1:-1]

        ras = [float(r[3]) for r in result0]
        decs = [float(r[4]) for r in result0]
        result = []
        if len(ras) > 0:
            # Add separation
            gal_coords = SkyCoord(ra=ras*u.deg, dec=decs*u.deg)
            cand_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            sep = cand_coord.separation(gal_coords)
            for i in np.arange(len(ras)):
                result0[i].append(float(sep[i].arcsec))
                # Check that the separation is less than required
                if float(sep[i].arcsec) < radius_arcsec:
                    result.append(result0[i])
        # Print the results
        print("z_phot_median, z_phot_std, z_phot_l95, ra, dec, type, flux_z, sep_arcsec")
        for r in result:
            print(", ".join([str(x) for x in r])) 
        
        return result


if __name__ == '__main__':
    
    """
    import argparse

    parser = argparse.ArgumentParser(description='Legacy Survey phtotoz')
    parser.add_argument('radec', metavar='RA, Dec', type=str, nargs='+',
                        help='RA and Dec (degrees)')
    parser.add_argument('-r', dest='radius', type=float,
                        required=False, help='Search radius (arcsc)',
                        default=2)
    args = parser.parse_args()

    # RA and Dec
    ra, dec = float(args.radec[0]), float(args.radec[1])

    # Radius
    search_rad = args.radius
    query_coords_ls(ra, dec, radius_arcsec=search_rad)
    """


unclassified_data = pd.read_csv(f'{datapath}/unclassified_lc_gap_all.csv')[1:2]
unclassified_data.groupby('name').ngroups

ra  = unclassified_data.groupby(['name']).apply(lambda x: x['ra'].median())
dec = unclassified_data.groupby(['name']).apply(lambda x: x['dec'].median())
# RA and Dec
#ra, dec = float(args.radec[0]), float(args.radec[1])
# Radius
search_rad = 5

source_name   = []
z_phot_median = []
z_phot_std    = []
z_phot_l95    = []
ra_photz      = []
dec_photz     = []
type_data     = []
flux_z        = [] 
sep_arcsec    = []

with tqdm(total=unclassified_data.groupby('name').ngroups) as progress:
    for name in unclassified_data['name'].unique():
        # result output "z_phot_median, z_phot_std, z_phot_l95, ra, dec, type, flux_z, sep_arcsec"
        result = query_coords_ls(ra[name], dec[name], radius_arcsec=search_rad)[0]
        
        source_name.append(name)
        z_phot_median.append(result[0])
        z_phot_std.append(result[1])
        z_phot_l95.append(result[2])
        ra_photz.append(result[3])
        dec_photz.append(result[4])
        type_data.append(result[5])
        flux_z.append(result[6])
        sep_arcsec.append(result[7])
        
        