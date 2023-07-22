import numpy as np
import glob
import os
import pandas as pd
from astropy.table import Table

from tqdm.auto import tqdm
import time

datapath = f"{os.path.dirname(os.path.realpath('__file__'))}/lc_gap_csv"
classified_dir =  f"{os.path.dirname(os.path.realpath('__file__'))}/classified_unclassified_objet/ZTF_Gap_Trasien_classifications.dat"

targetdir ='../ZTF'
if not os.path.isdir(targetdir):
    os.makedirs(targetdir)


def data_filename(datapath):
    """[using recursively to find files]

    :param datapath: the filename directory
    :type datapath: [str]
    :return: all .csv files 
    :rtype: [list]
    """
    
    return glob.glob(datapath+'/**/*.csv', recursive=True)



print("Start : %s" % time.ctime())
start_time = time.time()


ztf_files = data_filename(datapath)
classified_source_name = Table.read(classified_dir, format='ascii.fast_tab')['source_name']

lcs_unclass = pd.DataFrame()
lcs_class   = pd.DataFrame()

unclass = 0
classified = 0

with tqdm(total=len(ztf_files)) as progress:
    for forcedfile in ztf_files:
        name = forcedfile.split('/')[-1].split('.')[0].split('_')[-1]
        data = pd.read_csv(forcedfile)
        data.sort_values(by='jd', ascending=True)
        if name in classified_source_name :
            lcs_class = pd.concat([lcs_class, data], ignore_index = True)
            classified+=1
        else:
            lcs_unclass = pd.concat([lcs_unclass, data], ignore_index = True)
            unclass+=1

        
        progress.update()
        
    lcs_class.sort_values(by='jd', ascending=True)
    lcs_class.to_csv(f'classified_lc_gap_all.csv', index=False)
    
    lcs_unclass.sort_values(by='jd', ascending=True)
    lcs_unclass.to_csv(f'unclassified_lc_gap_all.csv', index=False)
    
    print("=====================================================")
    print('unclassified  :', unclass)
    print("    ")
    print('classified  :', classified)
    print("=====================================================")


                                   
    print("End : %s" % time.ctime())
    #time.sleep(2)

    # determine the time take to run this, in the format (Hours,Minutes, and Secondes)
    hours, rem       =  divmod((time.time() - start_time), 3600)
    minutes, seconds =  divmod(rem, 60)


    print("    ")
    print("    ")

    print("+----------------------------------------------------+")
    print("    ")
    print("--- This run took {:0>2}:{:0>2}:{:05.2f} ---".format(int(hours),int(minutes),seconds))
    print("    ")     
    print("+----------------------------------------------------+")