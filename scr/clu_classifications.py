import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time

datapath = f"{os.path.dirname(os.path.realpath('__file__'))}/../data/classified_unclassified_objet"

targetdir ='../ZTF'
if not os.path.isdir(targetdir):
    os.makedirs(targetdir)


print("Start : %s" % time.ctime())
start_time = time.time()


classified_data   = pd.read_csv(f'{datapath}/classified_lc_gap_all.csv')
unclassified_data = pd.read_csv(f'{datapath}/unclassified_lc_gap_all.csv')

clu_data = pd.read_csv(f"{os.path.dirname(os.path.realpath('__file__'))}/../data/crossmatch_clu_dist20.0-200.0Mpc_projdist10-130.csv")
## remove the NaN in the redshift CLU 
no_nan_idx = np.where(clu_data['clu_z'] != ' nan')[0]
clu_data = clu_data.iloc[no_nan_idx]

lcs_unclass = pd.DataFrame()
lcs_class   = pd.DataFrame()

unclass = 0
classified = 0


with tqdm(total=clu_data.groupby('name').ngroups) as progress:
    for name  in clu_data['name'].unique():
        idx = np.where(clu_data['name'] == name)[0]
        lc = clu_data.iloc[idx]
        
        if name in list(classified_data['name']) :
            lcs_class = pd.concat([lcs_class, lc], ignore_index = True)
            classified+=1
        else:
            lcs_unclass = pd.concat([lcs_unclass, lc], ignore_index = True)
            unclass+=1

        
        progress.update()
    
    lcs_class.to_csv(f'{datapath}/crossmatch_clu_classified.csv', index=False)
    lcs_unclass.to_csv(f'{datapath}/crossmatch_clu_unclassified.csv', index=False)
    
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
