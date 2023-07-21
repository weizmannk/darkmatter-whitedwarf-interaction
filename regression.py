import time
import os
from tqdm.auto import tqdm
import optparse
from astropy.time import Time, TimeDelta
import pandas as pd
import numpy as np
import glob
import corner
import pymultinest
from scipy import stats
from sklearn.linear_model import LinearRegression

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
# Read csv file and  create a base directory (folder) to save the plot 
# =============================================================================
def parse_commandline(csv):
    """[Loadind the data of cvs file ]

    :param folder: [output name to save the plots]
    :type folder: [str]
    :param csv: [ intput csv data ]
    :type csv: [type]
    :return: [return a dictionary contain plot directory, cvs file]
    :rtype: [optparse.Values]
    """

    parser = optparse.OptionParser()
    parser.add_option("--doPlots",  action="store_true", default=False)
    parser.add_option("-p", "--plotDir", default= "outputs")
    parser.add_option("-l", "--lightcurve", default=csv)

    opts, args = parser.parse_args()

    return opts


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
    w = 1.0 / sigma**2  
    So = np.sum(w)       # estimate of mageer
    Sx = np.sum(w * x)
    Sy = np.sum(w * y)
    Sxy = np.sum(w * x * y)
    Sxx = np.sum(w * x * x)
    Matrix = np.array([[Sxx, Sx], [Sx, So]])
    vector = np.array([Sxy, Sy])
    return Matrix, vector

# =============================================================================
# scipy  linear regeression model
# =============================================================================
def scipy_linear_model(t, a, b):
    """[linear regression ]

    :param t: [delay of observation]
    :type t: [array]
    :return: [the regression model for prediction]
    :rtype: [type]
    """
    return a*t + b

# =============================================================================
# sklearn linear regeression model
# =============================================================================
def sklearn_linear_model(t, mag):
    """[Linear Regression model using sklearn]

    :param t: [time of observation in days]
    :type t: [float]
    :param mag: [magnitude of observation]
    :type mag: [float]
    """
    sk_t = np.array(t).reshape((-1, 1))
    result = LinearRegression().fit(sk_t, mag)
    score = result.score(sk_t, mag)
    model = result.predict(sk_t)
    
    return model, score

# =============================================================================
# Regeression using the pymultinest (bayesain model)
# =============================================================================
def plaw(t, f0, alpha=1.0, t0=None):
    """
    Power law function:
    f / f0 = (t - t0) ** -alpha
    """
    return f0 * (t - t0) ** -alpha

def pmag(t, f0, alpha=1.0, t0=None):
    """
    Use indices from plaw fit to calculate magnitudes
    -2.5 log(f/f0) = 2.5 * alpha * log(dt)
    m = -2.5 log(f)
    m + 2.5 log(f0) = 2.5 * alpha * log(dt)
    """ 
    return 2.5 * alpha * np.log10(t - t0) - 2.5 * np.log10(f0)

def eflux(emag, flux):
    """
    Error propoagation:
    m - m0 = dm = 2.5 log(1 + df / f0)
    10.0 ** (dm / 2.5) = 1 + df / f0
    df = f0 * (10.0 ** (dm / 2.5) - 1)
    """
    return flux * (10.0 ** (emag / 2.5) - 1)

def myprior(cube, ndim, nparams, tmax, tmin):

    cube[0] = cube[0]*(tmax - tmin) + tmin
    cube[1] = cube[1]*3.0
    cube[2] = cube[2]*20.0 - 10.0

def myloglike(cube, ndim, nparams, t, y, dy):
    
    t0fit = cube[0]
    alpha = cube[1]
    f0 = 10**cube[2]

    mod = pmag(t, f0, alpha=alpha, t0=t0fit)
    
    idx1 = np.where(~np.isnan(y))[0]
    idx2 = np.where(np.isnan(y))[0]

    chisq = -(1/2)*np.sum(((y[idx1] - mod[idx1])**2.) /
                            (dy[idx1]**2.))/(len(t[idx1]) - nparams)

    #gaussprobvals = np.sum(np.log(1-scipy.stats.norm.cdf(dy[i
    # dx2], mod[idx2], 0.1)))

    return chisq




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
            
        if len(band) > 3:
            tmax = np.min(data['jd'].iloc[band])
            tmin = np.min(data['jd'].iloc[band])-3
                
            indx   = np.where(data['magpsf'].iloc[band] == np.min(data['magpsf'].iloc[band]))[0][0]
            t_peak = data['jd'][indx]
            y_peak =  np.min(data['magpsf'].iloc[band])
            
            tab = data.iloc[band][data['jd'].iloc[band] >= t_peak]
        
            plotDir = os.path.join(f"{figsdir}", f"{name}/{filt}")
            if not os.path.isdir(plotDir):
                os.makedirs(plotDir)
                
            
            t = data['jd'].iloc[band]
            y = data['magpsf'].iloc[band]
            dy = data['sigmapsf'].iloc[band]
            #dy = np.sqrt(dy**2 + 0.1**2)  # np.mea
            
            
            pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling=False, resume=True, verbose=True, sampling_efficiency='parameter',
                        n_live_points=n_live_points, outputfiles_basename='%s/2-' % plotDir, evidence_tolerance=evidence_tolerance, multimodal=False, max_iter=max_iter)

            multifile = os.path.join(plotDir, '2-post_equal_weights.dat')
            pym_data = np.loadtxt(multifile)
            
            
            
            # =============================================================================
            # plot corner and save the data in  plotName directory 
            # =============================================================================
            labels = [f"$t_0$", f"$\\alpha$", f"$\\log_{10} f_0$"]
            plotName = "%s/corner.pdf" % (plotDir)
            figure = corner.corner(pym_data[:, :-1], labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={
                                    "fontsize": 24}, label_kwargs={"fontsize": 24}, title_fmt=".2f", smooth=3,  color="coral")
            figure.set_size_inches(14.0, 14.0)
            plt.savefig(plotName)
            plt.close()
            # =============================================================================
            # Call for posterio  mag calculate by using bayesian methode with pymultinest
            # =============================================================================
            idx = np.argmax(data[:, -1])
            t_0, alpha, f0 = data[idx, :-1]
            tt = np.linspace(np.min(t), np.max(t), 1000)
            mod = pmag(tt, 10**f0, alpha=alpha, t0=t_0)

            # =============================================================================
            # scipy model
            # =============================================================================
            a, b, r_value, p_value, std = stats.linregress(t, y)
            scipy_model = scipy_linear_model(t, a, b)

            # =============================================================================
            # scikitlearn linear model regeression
            # =============================================================================
            sklearn_model, sk_score = sklearn_linear_model(t, y)

            # =============================================================================
            # sufficient statistics to get the likelihood maximun, bayesian approach 
            # =============================================================================
            M, v = sufficient_statistics(t, y, dy)
            x1, x2 = np.linalg.solve(M, v)
            max_likelihood = x1*t + x2

            # =============================================================================
            # plot the lightcurve and Regression models  
            # =============================================================================
            plotName = "%s/lightcurve.pdf" % (plotDir)
            plt.figure(filt)
            plt.errorbar(t-tmin, y, dy, fmt='o', c='k')
            plt.plot(tt-tmin, mod, 'r-', linewidth=2,
                        label="pymultinest model")
            plt.plot(t - tmin, sklearn_model, c='brown', linewidth=2,
                        label="sklearn linear model pred = "+np.str(np.round(sk_score*100, 2))+"%")
            plt.plot(t - tmin, scipy_model, 'b-', linewidth=2,
                        label="scipy linear model pred = "+np.str(np.round(r_value*100, 2))+"%")
            plt.plot(t - tmin, max_likelihood, 'g-',
                        linewidth=2, label="Max likelihood")

            plt.xlabel(r'Time [days] [t0 = %.5f]' % tmin, fontsize=18)
            plt.ylabel(r'Magnitude', fontsize=18)
            plt.grid()
            plt.legend(loc='best')
            plt.gca().invert_yaxis()
            plt.savefig(plotName)
            plt.close()

        else:
            print("******************************************************")
            print("Attention, the length of Data in this filter:",
                    filter, "is only", len(band))
            print("Please wait, we go on  for others filter ")

            time.sleep(5)


    #return fig
        

parameters = ["t0", "alpha", "f0"]
n_params = len(parameters)

n_live_points = 1000
evidence_tolerance = 0.1
max_iter = -1


program_ids = [1,2,3]
ztf_files = data_filename(datapath)[10:]

with tqdm(total=len(ztf_files)) as progress:
    for forcedfile in ztf_files:
        name = forcedfile.split('/')[-1].split('.')[0].split('_')[-1]
        
        photometry(targetdir=targetdir, name=name, program_ids=program_ids, forcedfile=forcedfile)

        #data = data.sort_values("jd", ascending=True)
        #firstdate_ztf  = np.min(data['jd'])

        progress.update()
        
