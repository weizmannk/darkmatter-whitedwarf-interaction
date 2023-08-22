
import pandas as pd

import glob
import os
from tqdm.auto import tqdm

targetdir ='./nmma_ZTF'
datapath = f"{os.path.dirname(os.path.realpath('__file__'))}/lc_gap_csv"

outdir = targetdir           
if not os.path.isdir(outdir):
    os.makedirs(outdir)


def data_filename(datapath):
    """[using recursively to find files]

    :param datapath: the filename directory
    :type datapath: [str]
    :return: all .csv files 
    :rtype: [list]
    """
    
    return glob.glob(datapath+'/**/*.csv', recursive=True)



classification_dir =  f"{os.path.dirname(os.path.realpath('__file__'))}/classified_unclassified_objet/classified_lc_gap_all.csv"
classified_data    = pd.read_csv(classification_dir) 

source_name        = classified_data['name'].unique()
#redshift = []

ztf_files = data_filename(datapath)

with tqdm(total=len(ztf_files)) as progress:
    for forcedfile in ztf_files:
    
        name = forcedfile.split('/')[-1].split('.')[0].split('_')[-1]
        
        if name in source_name:
            data = pd.read_csv(forcedfile)
            dict_file = {'jd' : data['jd'], 'mag' : data['magpsf'], 'mag_unc' : data['sigmapsf'], 'filters' : data['filter'], 'programid':data['programid']}

            df = pd.DataFrame(data=dict_file) 
            df.sort_values("jd", ascending=False)
            df.to_csv(f"{outdir}/lc_{name}" + '.csv')

            
            del name, data, dict_file , df

        
        progress.update()

        
        
            
