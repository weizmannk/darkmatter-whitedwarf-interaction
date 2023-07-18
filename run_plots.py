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


def data_filename(datapath):
    """[using recursively to find files]

    :param datapath: the filename directory
    :type datapath: [str]
    :return: all .csv files 
    :rtype: [list]
    """
    
    return glob.glob(datapath+'/**/*.csv', recursive=True)

# =============================================================================
# linear regeression  using Max liklihood bayesain  model
# =============================================================================
def sufficient_statistics(t, mag, magerr):
    """[summary]

    :param t: [observation time delays  ]
    :type t: [float in jd time]
    :param mag: [magnitude of observation]
    :type mag: [float]
    :param magerr: [magnitude error]
    :type magerr: [float]
    :return: [ Matrix ]
    :rtype: [type]
    """
    x = t
    y = mag
    sigma = magerr

    # estimate of magerror
    w = 1.0 / sigma**2
    So = np.sum(w)
    Sx = np.sum(w * x)
    Sy = np.sum(w * y)
    Sxy = np.sum(w * x * y)
    Sxx = np.sum(w * x * x)
    Matrix = np.array([[Sxx, Sx], [Sx, So]])
    vector = np.array([Sxy, Sy])
    return Matrix, vector

def photometry(targetdir, name, program_ids, forcedfile):

    figsdir = "%s/%s" % (targetdir, "figures")
    if not os.path.isdir(figsdir):
        os.makedirs(figsdir)
        
    data = pd.read_csv(forcedfile) 
    
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
    
    ####################
    #####
    ###################
    
    fig = plt.figure(figsize=(8,6))
    
    for filt in np.unique(filts):
        if filt =='r':
            band = r
        elif filt == 'g':
            band = g
        else:
            band =i
            
        indx   = np.where(data['magpsf'].iloc[band] == np.min(data['magpsf'].iloc[band]))[0][0]
        t_peak = data['jd'][indx]
        y_peak =  np.min(data['magpsf'].iloc[band])
        
        tab = data.iloc[band][data['jd'].iloc[band] >= t_peak]

        if filt == 'r':
            plt.errorbar(daysago[r].value, data['magpsf'].iloc[r], yerr=data['sigmapsf'].iloc[r],
                    fmt='.', color='r', label='r-band')
        
        elif filt == 'g':
            plt.errorbar(daysago[g].value, data['magpsf'].iloc[g], yerr=data['sigmapsf'].iloc[g],
                    fmt='.', color='g', label='g-band')
        else:
            if len(i) != 0:
                plt.errorbar(daysago[i].value, data['magpsf'].iloc[i], yerr=data['sigmapsf'].iloc[i],
                             fmt='.', color='goldenrod', label='i-band')
        
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
    
    return fig
        




program_ids = [1,2,3]
ztf_files = data_filename(datapath)

with tqdm(total=len(ztf_files)) as progress:
    for forcedfile in ztf_files:
        name = forcedfile.split('/')[-1].split('.')[0].split('_')[-1]
        
        photometry(targetdir=targetdir, name=name, program_ids=program_ids, forcedfile=forcedfile)

        #data = data.sort_values("jd", ascending=True)
        #firstdate_ztf  = np.min(data['jd'])

        progress.update()
        
