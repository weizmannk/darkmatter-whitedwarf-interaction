import os
from tqdm.auto import tqdm

from astropy.time import Time, TimeDelta
import pandas as pd
import numpy as np
import glob

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
import matplotlib.pyplot as plt

targetdir ='../ZTF'
datapath = f"{os.path.dirname(os.path.realpath('__file__'))}/lc_gap_csv"

ztf_files = glob.glob(datapath+'/**/*.csv', recursive=True)

figsdir = "%s/%s" % (targetdir, "figures")
if not os.path.isdir(figsdir):
    os.makedirs(figsdir)

program_ids = [1,2,3]

with tqdm(total=len(ztf_files)) as progress:
    for forcedfile in ztf_files:
        name = forcedfile.split('/')[-1].split('.')[0].split('_')[-1]
        data = pd.read_csv(forcedfile) 

        #data = data.sort_values("jd", ascending=True)

        #firstdate_ztf  = np.min(data['jd'])


        idxs = []
        for ii, programid in  enumerate(data['programid']):
            if programid in program_ids:
                idxs.append(ii)
        data = data.iloc[idxs]    

        # plotting code
        times = Time(data['jd'], format='jd')
        daysago = TimeDelta(times - Time(np.min(data['jd']), format='jd'), scale='tt')

        filts = []
        for ii, filt in enumerate(data['filter']):
            filts.append(filt.replace("'","").replace("b",""))
        filts = np.array(filts)
        g = np.where(filts == 'g')[0]
        r = np.where(filts == 'r')[0]
        i = np.where(filts == 'i')[0]

        fig = plt.figure(figsize=(8,6))
        plt.errorbar(daysago[r].value, data['magpsf'].iloc[r], yerr=data['sigmapsf'].iloc[r],
                        fmt='.', color='r', label='r-band')
        plt.errorbar(daysago[i].value, data['magpsf'].iloc[i], yerr=data['sigmapsf'].iloc[i],
                        fmt='.', color='goldenrod', label='i-band')
        plt.errorbar(daysago[g].value, data['magpsf'].iloc[g], yerr=data['sigmapsf'].iloc[g],
                        fmt='.', color='g', label='g-band')


        #plt.ylim(23.0, 18.0)
        #plt.xlim(200, 0)
        plt.title(name)
        plt.gca().invert_yaxis()
        plt.legend(loc='best')
        plt.grid(True)

        plt.xlabel('Time since first ZTF detection [days]')
        plt.ylabel('absolute magnitude (AB)')
        plotname = os.path.join(figsdir, '%s.png' % name)
        plt.savefig(plotname)
        plt.grid()
        plt.close()
        progress.update()
