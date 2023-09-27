import pymultinest
import numpy 
import os
import pandas as pd
from astropy.time import Time

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import emcee
import corner
from pymultinest.solve import solve
import json
import scipy.stats
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable



parameters = ['alpha', 'k1', 'k2']
ndim = len(parameters)

infile = "/home/kiendrebeogo/weizmann-doc/Dark-matter/DarkMatter-WhiteDwarf/nmma_ZTF/lc_ZTF22aboisvs.csv"

data = pd.read_csv(infile)

data = data.sort_values('jd', ascending=True)

idx = numpy.where(data['mag_unc'] != numpy.inf)[0]
mjd_min = Time(data["jd"].iloc[idx[0]], format='jd').mjd

ztfr = data[data['filters'] =='ztfr']
ztfi = data[data['filters'] =='ztfi']



time_r = Time(ztfr['jd'], format='jd').mjd - mjd_min
time_i = Time(ztfi['jd'], format='jd').mjd - mjd_min

mag_r = ztfr['mag'].to_numpy
mag_i = ztfi['mag'].to_numpy

mag_err_r = ztfr['mag_unc'].to_numpy
mag_err_i = ztfr['mag_unc'].to_numpy


###########
#ZF
rbandtimes_LDT = time_r 
rbandmag_LDT = mag_r 
rbanderrs_LDT = mag_err_r 

ibandtimes_LDT = time_i 
ibandmag_LDT = mag_i 
ibanderrs_LDT = mag_err_i

## Read 

#LDT and LT data together

#rbandtimes_LDT = numpy.array([3.6, 9.5, 18.5, 21.5, 28.5, 36.53, 1.272039349998522, 1.308559349999996, 1.3958093499968527, 2.266309350001393, 2.2955893499965896, 3.2731893500022124, 3.283359349996317, 7.277719350000552, 7.2914793500021915, 8.273209350001707, 8.290449350002746, 10.288109349996375, 10.306029350002063, 11.271729350002715, 11.290619350002089, 14.287679349996324, 14.313049349999346, 18.297539350001898, 38.265369349996035, 1.05022335, 1.05371835, 1.10147035, 2.11955235, 3.00693735, 4.03068335, 5.07656435])
#rbandmag_LDT = numpy.array([20.44, 21.85, 22.71, 22.91, 23.55, 23.64, 18.85, 18.87, 19.03, 19.84, 19.84, 20.31, 20.33, 21.6, 21.44, 21.68, 21.91, 22.04, 22.04, 22.09, 22.09, 22.45, 22.35, 22.99, 24.06, 18.58, 18.55, 18.66, 19.77, 20.41, 20.83, 20.91])
#rbanderrs_LDT = numpy.array([0.04, 0.08, 0.06, 0.06, 0.15, 0.11, 0.02, 0.02, 0.04, 0.05, 0.05, 0.05, 0.09, 0.08, 0.08, 0.08, 0.13, 0.11, 0.1, 0.11, 0.1, 0.17, 0.15, 0.24, 0.32,0.06, 0.05, 0.08, 0.14, 0.15, 0.16, 0.21])                             

rbandhz = numpy.array([4.82e14] * len(rbandtimes_LDT))

#ibandtimes_LDT = numpy.array([3.6,4.4, 9.5, 17.4, 18.5, 21.5, 28.5, 36.53, 52.53, 1.2728993499986245, 1.309939350001514, 1.3966793500003405, 2.2678093499998795, 2.296639350002806, 3.275849350000499, 3.2843993499991484, 7.280389350002224, 7.28876935000153, 8.276559350000753, 8.28703934999794, 10.292159349999565, 10.302629350000643, 11.276709350000601, 11.287209349997283, 14.292659350001486, 14.309619350002322, 18.286279350002587, 36.275029349999386,1.04287735, 1.04447835, 1.04607935, 3.00313235, 4.02688035, 5.07656635, 6.09907835])
#ibandmag_LDT = numpy.array([19.37,19.78, 20.75, 21.71, 21.83, 21.82, 22.07, 22.45, 22.73, 17.77, 17.8, 17.93, 18.69, 18.73, 19.29, 19.28, 20.44, 20.37, 20.58, 20.69, 21.07, 20.92, 21.12, 21.17, 21.47, 21.35, 21.91, 22.23, 17.48, 17.49, 17.52, 19.20, 19.68, 19.85, 20.43])                            
#ibanderrs_LDT = numpy.array([0.01,0.02, 0.05, 0.05, 0.15, 0.02, 0.09, 0.05, 0.15, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.04, 0.05, 0.06, 0.05, 0.05, 0.06, 0.07, 0.09, 0.13, 0.11, 0.04, 0.03, 0.04, 0.14, 0.14, 0.19, 0.18])                             
ibandhz = numpy.array([3.93e14] * len(ibandtimes_LDT))



def magtoflux(mag):
    return 10**((mag + 48.6)/-2.5)

def fluxtomag(flux):
    return -2.5 * numpy.log10(flux) - 48.6

def magtofluxerr(mag, err):
    return numpy.log(10) * magtoflux(mag) * (err/2.5)

def fluxtomagerr(flux,err):
    return (2.5/numpy.log(10)) * (err / flux)


fluxr_LDT = magtoflux(rbandmag_LDT) 
fluxrerr_LDT =  magtofluxerr(rbandmag_LDT, rbanderrs_LDT)

fluxi_LDT = magtoflux(ibandmag_LDT) 
fluxierr_LDT =magtofluxerr(ibandmag_LDT, ibanderrs_LDT)



def my_prior_transform(cube, ndim, numpyarams):
    #alpha 
    cube[0] = cube[0] *2

    #k1 
    cube[1] = 10**(cube[1]*28/5 - 756/25)
    #k2
    cube[2] = 10**(cube[2]*28/5 - 756/25)
    
    #Gaussian priors for host galaxy fluxes
    mui = magtoflux(23.8)
    sigmai = magtofluxerr(23.8, 0.14)
    rvs = scipy.stats.norm(mui, sigmai)
    cube[4] = rvs.ppf(cube[4] )  
                      
    mur = magtoflux(24.88)
    sigmar = magtofluxerr(24.88, 0.08)    
    rvs = scipy.stats.norm(mur,sigmar)
    cube[3] = rvs.ppf(cube[3])    
    
    #log_f, comment if not including beta parameter
    cube[5] = cube[5]*5-4
    

def my_likelihood(cube, ndim, numpyarams):
    
    alpha = cube[0]
    k1= cube[1]
    k2 = cube[2]
    hostr = cube[3]
    hosti = cube[4]
    log_f = cube[5]
    m1 = k1*rbandtimes_LDT**(-alpha) + hostr
    m2 = k2*ibandtimes_LDT**(-alpha) + hosti
    
    #comment these three lines below if not including beta parameter
    sigma1 = fluxrerr_LDT**2 + m1**2 * numpy.exp(2*log_f)
    sigma2 = fluxierr_LDT**2 + m2**2 * numpy.exp(2*log_f)
    return -0.5*(numpy.sum(numpy.abs(fluxr_LDT-m1)**2/sigma1+ numpy.log(sigma1))  + numpy.sum(numpy.abs(fluxi_LDT-m2)**2/sigma2+numpy.log(sigma2)))
    
    #uncomment if not including beta parameter
    #return  -0.5*(numpy.sum(numpy.abs(fluxr_LDT-m1)**2/fluxrerr_LDT**2 + numpy.log(2*numpy.pi*fluxrerr_LDT**2))  + numpy.sum(numpy.abs(fluxi_LDT-m2)**2/fluxierr_LDT**2+numpy.log(2*numpy.pi*fluxierr_LDT**2)))
    

def modelfinal(t1, alpha, k1, k2, hostr, hosti):
    #k, alpha = params
    #k1, k2, alpha=params
    #alpha = params
    #return k*t**(-alpha)    
    return (k1*t1**(-alpha) + hostr,k2*t1**(-alpha) + hosti)



prefix = 'hump_'

parameters = ["alpha", 'k1', 'k2', 'host_r', 'host_i', 'log_f']
n_params = len(parameters)


pymultinest.run(my_likelihood, my_prior_transform, 6, outputfiles_basename= '_afterglowbestfitfinal_', resume = False, verbose = True, sampling_efficiency = 0.3, n_live_points = 100)
json.dump(parameters, open('_afterglowbestfitfinal_params.json', 'w')) # save parameter names



a = pymultinest.Analyzer(outputfiles_basename= '_afterglowbestfitfinal_', n_params = n_params)

a_lnZ = a.get_stats()['global evidence']

print('************************')
print('MAIN RESULT: Evidence Z ')
print( '************************')
print (' log Z for model with 1 line = %.1f' % (a_lnZ / numpy.log(10)))

print(a.get_best_fit()['log_likelihood'], a.get_best_fit()['parameters'])
print("median",a.get_stats()['marginals'][0]['median'], a.get_stats()['marginals'][1]['median'], a.get_stats()['marginals'][2]['median'], a.get_stats()['marginals'][3]['median'], a.get_stats()['marginals'][4]['median'])
print(a.get_stats()['marginals'][0]['1sigma'], a.get_stats()['marginals'][1]['1sigma'], a.get_stats()['marginals'][2]['1sigma'], a.get_stats()['marginals'][3]['1sigma'], a.get_stats()['marginals'][4]['1sigma'])








#Now do the inclusion of the SN model



def addmags(m1, m2):
    return -2.5 * numpy.log10(10**(-0.4*m1) + 10**(-0.4*m2))

#SN 1998bw data
times = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0])
rband = numpy.array([26.627169017689866, 26.147455553706884, 25.857326220581996, 25.6286032467831, 25.439779734562453, 25.2789812284108, 25.13895392202461, 25.014940287247356, 24.898780530800938, 24.793589389763543, 24.697695517142, 24.60958807661088, 24.52809699095576, 24.460803697391263, 24.43280103464396, 24.405502478087104, 24.37887348601524, 24.3528819977827, 24.327498201736518, 24.327669689982745, 24.36144462914782, 24.39630406120276, 24.432319948008534, 24.46957166052236, 24.508147032163112, 24.552942272563744, 24.602052618002087, 24.65349001050449, 24.707485996982147, 24.764308501259396, 24.824269867145794, 24.875195333850062, 24.92753526984947, 24.982526710083416, 25.040452704028723, 25.101644192911085, 25.16347730141206, 25.213325445857492, 25.265572804218174, 25.320462056499746, 25.378274673811475, 25.439339653358655, 25.489395376065, 25.52037923569024, 25.55227332661044, 25.585132752535795, 25.61901777759799, 25.653994491529957, 25.690135585601972, 25.727521262208562, 25.766240306719656, 25.80639135762146, 25.848084420663714, 25.89144268551984, 25.91715366038394, 25.94136869286008, 25.96613613374879, 25.991481776378073, 26.017433263880267, 26.04402027041479, 26.07127470514197, 26.099230942463866, 26.12792608269871, 26.157400248147663, 26.18769692048161, 26.21243375663926, 26.22765948879602, 26.243101778224883, 26.258766874152194, 26.27466130027163, 26.290791871056204, 26.307165709300303, 26.323790265004966, 26.34067333573153, 26.357823088562956, 26.375248083827636, 26.392957300758123, 26.405040679212647, 26.41646927148549, 26.428019443417266, 26.439693809625368, 26.451495069987324, 26.463426013388556, 26.47548952167842, 26.487688573848388, 26.500026250447746, 26.51250573825303, 26.525130335208914, 26.537903455659816, 26.550828635892874, 26.56390954001488, 26.577149966187548, 26.59055385324773, 26.60412528774134, 26.61786851140259, 26.63178792911259, 26.64588811737483, 26.66017383334824, 26.6746500244825, 26.68932183880446, 26.70419463590907, 26.71927399871368, 26.73456574604012, 26.750075946095578, 26.765810930930396, 26.781777311959004, 26.797981996639244, 26.81443220641533, 26.831135496041178, 26.848099774413452, 26.865333327058156, 26.882844840430863, 26.900643428208873, 26.918738659774597, 26.93714059111307, 26.95585979837321, 26.97490741437342, 26.99429516836691, 27.014035429422517, 27.03414125382264, 27.054626436933304, 27.075505570062294, 27.096794102892403, 27.118508412158832, 27.14066587733551, 27.163284964206625, 27.18638531733013, 27.209987862553326, 27.234114920921087, 27.25879033553055])
rband = addmags(rband, 24.88)
iband = numpy.array([25.94880341149537, 25.46908994571719, 25.17896061139238, 24.950237636849415, 24.76141412412225, 24.600615617603534, 24.460588310939105, 24.336574675943673, 24.19013346486844, 24.059262023173464, 23.94248210331075, 23.83705194350665, 23.740959502328685, 23.661173253921714, 23.621779459874148, 23.583765088250146, 23.54703679406318, 23.51151039798284, 23.477109724286493, 23.47119487065795, 23.502107309743437, 23.533925726977515, 23.56670483449377, 23.600504455256743, 23.635390180134582, 23.67622080944086, 23.72101885118507, 23.767745128806503, 23.816573131322258, 23.86770086917808, 23.921355724963536, 23.950387723233987, 23.9778987234375, 24.006124975703923, 24.035104666381528, 24.064879123964577, 24.09249550432264, 24.105721203736433, 24.119109998374697, 24.132665960993116, 24.146393318833404, 24.160296461536657, 24.18172145970012, 24.2143134686407, 24.247914193987725, 24.28258807457487, 24.31840592827209, 24.35544582278052, 24.393794100282555, 24.43354658976601, 24.474810049845303, 24.5177038967542, 24.562362287940196, 24.6089366528587, 24.63754300033763, 24.664732542685073, 24.69262050663756, 24.721243721213717, 24.75064200759098, 24.780858512278282, 24.81194008799283, 24.843937730679432, 24.87690708290568, 24.910909016110615, 24.946010307009743, 24.976231528971503, 24.997766577790486, 25.019737422183155, 25.04216206511209, 25.065059648679373, 25.088450552306657, 25.112356501726055, 25.136800690242474, 25.161807913963763, 25.187404722974875, 25.213619590766584, 25.240483104629277, 25.252896740192824, 25.263441601100425, 25.27408988079341, 25.2848436279414, 25.29570495269802, 25.306676029186164, 25.317759098109974, 25.328956469501634, 25.340270525611224, 25.35170372394876, 25.363258600488084, 25.37493777304315, 25.3867439448278, 25.398679908211275, 25.410748548682456, 25.422952849036896, 25.435295893801882, 25.447780873915935, 25.46041109168047, 25.473189966002927, 25.486121037952095, 25.4992079766483, 25.5124545855129, 25.525864808903783, 25.53944273916578, 25.553192624127533, 25.567118875079217, 25.581226075268564, 25.595518988956137, 25.610002571074652, 25.62468197754129, 25.639562576276784, 25.654649958990152, 25.669949953793815, 25.6854686387204, 25.701212356219564, 25.71718772872147, 25.733401675362447, 25.749861429978584, 25.76657456048434, 25.78354898976615, 25.80079301823538, 25.818315348201374, 25.836125110243664, 25.854231891783517, 25.87264576807857, 25.89137733589145, 25.910437750114063, 25.929838763664495, 25.949592771013833, 25.969712855746575, 25.990212842611626, 26.011107354582585])
iband = addmags(iband, 23.8)




#taking into account host




tckr= interpolate.splrep(times, magtoflux(numpy.array(rband)), s=0, k=3)
tcki = interpolate.splrep(times, magtoflux(numpy.array(iband)), s=0, k=3)
#isplinefit = interpolate.BSpline(*tck)(x_ne



parameters = ['fluxstretch', 'timestretch' , 'alpha', 'k1', 'k2']
ndim = len(parameters)
parameters = ['fluxstretch', 'timestretch' , 'alpha', 'k1', 'k2']
ndim = len(parameters)

def find_nearest(array, value):
    array = numpy.asarray(array)
    idx = (numpy.abs(array - value)).argmin()
    return idx

def chi(obs,exp,error):
    return (exp-obs)/(error)



#means of distribution of k and s values from literature, (flux and time stretching factors with respect to SN 1998bw) draw from these

means = [1.0004838709677422, 0.8806451612903226]
cov = [[0.21223392,0.02914301], [0.02914301, 0.03695957]]
#x, y = numpy.random.multivariate_normal(means, cov, 10000).T

a = numpy.linalg.inv(cov)
l, v = numpy.linalg.eigh(a)
rotation_matrix = numpy.dot(v, numpy.diag(1. / numpy.sqrt(l)))

def transform_correlated(quantiles):
    # sample a independent multivariate gaussian
    independent_gaussian = scipy.stats.norm.ppf(quantiles)
    # rotate and shift
    return means + numpy.einsum('ij,kj->ki', rotation_matrix, independent_gaussian)    

samples = transform_correlated(numpy.random.uniform(0, 1, size=(1000, 2)))
x = samples[:,0]
y = samples[:,1]

r = 0.876
sx = 0.46
sy = 0.20

#means = numpy.array([0.9, 1.25])
#means = numpy.array([0.8, 0.8])
means = numpy.array([1.02, 0.88])
cov = numpy.array([[sx**2, r*sx*sy], [r*sx*sy, sy**2]])

a = numpy.linalg.inv(cov)
l, v = numpy.linalg.eigh(a)
rotation_matrix = numpy.dot(v, numpy.diag(1. / numpy.sqrt(l)))

def tc(quantiles):
    # sample a independent multivariate gaussian
    independent_gaussian = scipy.stats.norm.ppf(quantiles)
    # rotate and shift
    return means + numpy.einsum('ij,kj->ki', rotation_matrix, independent_gaussian)
#point = numpy.random.uniform(0, 1, size=(1, 2))


def my_prior_transform(cube, ndim, numpyarams):
    #pulling from the bivariate normal distribution for k and s 
    
    
    point = numpy.empty((1,2))
    point[0,0] = cube[0]
    point[0,1] = cube[1]
    cubearray = (tc(point))
    cube[0] = cubearray[0][0]
    cube[1] = cubearray[0][1]
    #print(cube[0], cube[1])
    
    
    
   
    #power law index 
    #rvs = scipy.stats.norm(1.440, 0.005)
    #cube[2] = rvs.ppf(cube[2])
    
    cube[2] = cube[2] *2
    #constants of proportionality k1 and k2
    
    cube[3] = 10**(cube[3]*28/5 - 756/25)
    
    
    cube[4] = 10**(cube[4]*28/5 - 756/25)
    
    #host_galaxy fluxes with gaussian priors
    mui = magtoflux(23.8)
    sigmai = magtofluxerr(23.8, 0.14)
	
    mur = magtoflux(24.88)
    sigmar = magtofluxerr(24.88, 0.08)    
    
    rvs = scipy.stats.norm(mur,sigmar)
    cube[5] = rvs.ppf(cube[5])
    
    rvs = scipy.stats.norm(mui, sigmai)
    cube[6] = rvs.ppf(cube[6])    
    
    
    
    #log_f comment if not using beta parameter
    cube[7] = cube[7]*5 - 4
    
    
    

def my_likelihood(cube, ndim , numpyarams):
	fluxstretch, timestretch, alpha, k1, k2, hostr, hosti, log_f = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6], cube[7]
	#fluxstretch, timestretch = params
	time = numpy.linspace(1,70,10000)
	
	#for 98bw
	rspline =  interpolate.BSpline(*tckr)(time)
	ispline =  interpolate.BSpline(*tcki)(time)

	m1array = []
	m2array = []
	
	#SN1998bw
	#only have SN flux is interpolation gives positive value (after around day 8)
	
	for i in range(len(rbandtimes_LDT)):
		if interpolate.BSpline(*tckr)(rbandtimes_LDT[i]/timestretch) < 0:
		    m1array.append(0)
		else:
		    m1array.append(  float(interpolate.BSpline(*tckr)(rbandtimes_LDT[i]/timestretch)))



	for i in range(len(ibandtimes_LDT)):
		if interpolate.BSpline(*tcki)(ibandtimes_LDT[i]/timestretch) < 0:

		    m2array.append(0)
		else:
		    m2array.append( float(interpolate.BSpline(*tcki)(ibandtimes_LDT[i]/timestretch)))
	
	
	
	#add host galaxy contribution
	m1= fluxstretch * numpy.array(m1array) + k1*rbandtimes_LDT**(-alpha) + hostr 

	m2 = fluxstretch * numpy.array(m2array)  + k2*ibandtimes_LDT**(-alpha) + hosti
	
	#Comment three lines below if not using beta parameter
	sigma1 = fluxrerr_LDT**2 + m1**2 * numpy.exp(2*log_f)
	sigma2 = fluxierr_LDT**2 + m2**2 * numpy.exp(2*log_f)
	return -0.5*(numpy.sum(numpy.abs(fluxr_LDT-m1)**2/sigma1+ numpy.log(sigma1))  + numpy.sum(numpy.abs(fluxi_LDT-m2)**2/sigma2+numpy.log(sigma2)))	

        #uncomment if not using beta parameter
	#return   -0.5*(numpy.sum(numpy.abs(fluxr_LDT-m1)**2/fluxrerr_LDT**2 + numpy.log(2*numpy.pi*fluxrerr_LDT**2))  + numpy.sum(numpy.abs(fluxi_LDT-m2)**2/fluxierr_LDT**2+numpy.log(2*numpy.pi*fluxierr_LDT**2)))
	


def modelfinal(t, fluxstretch, timestretch, alpha, k1, k2, hostr, hosti):
	#fluxstretch, timestretch= params
	#fluxstretch, timestretch = params
	#return (fluxstretch * p_r((numpy.array(t)/timestretch)) +1.62e-27*t**(-1.465), fluxstretch * p_i((numpy.array(t)/timestretch)) + 4.43e-27*t**(-1.465))

	m1array = []
	m2array = []
	
	#98bw
	
	for i in range(len(t)):
		if interpolate.BSpline(*tckr)(t[i]/timestretch) < 0:
		    m1array.append(0)
		else:
		    m1array.append(  float(interpolate.BSpline(*tckr)(t[i]/timestretch)))

	for i in range(len(t)):
		if interpolate.BSpline(*tcki)(t[i]/timestretch) < 0:
		    m2array.append(0)
		else:
		    m2array.append( float(interpolate.BSpline(*tcki)(t[i]/timestretch)))


	
	m1= fluxstretch * numpy.array(m1array) + k1*t**(-alpha) + hostr  #10**-26.689*rbandtimes_LDT**(-1.49)
	m2 = fluxstretch * numpy.array(m2array)  + k2*t**(-alpha)   + hosti

	return (m1, m2)
from pymultinest.solve import solve


parameters = ["fluxstretch", "timestretch", "alpha", "k1", "k2", "hostr", "hosti"]
n_params = len(parameters)


from scipy.signal import savgol_filter


fig2, ax2 = plt.subplots(figsize=(6,6))
divider = make_axes_locatable(ax2)
ax2_histx = divider.append_axes("bottom", 1.2, pad=0.0, sharex=ax2)
ax2_histx.xaxis.set_tick_params(labeltop=False)


#plt.plot(x, ydata, '+ ', color='red', label='data')


pymultinest.run(my_likelihood, my_prior_transform, 8, outputfiles_basename= '_afterglow1998bwbestfitfinal_', resume = False, verbose = True,  sampling_efficiency = 0.3, n_live_points =100)
json.dump(parameters, open('_afterglow1998bwbestfitfinal_params.json', 'w')) # save parameter na

a = pymultinest.Analyzer(outputfiles_basename= '_afterglow1998bwbestfitfinal_', n_params = n_params)

a_lnZ = a.get_stats()['global evidence']

print('************************')
print('MAIN RESULT: Evidence Z ')
print( '************************')
print (' log Z for model with 1 line = %.1f' % (a_lnZ / numpy.log(10)))

print(a.get_best_fit()['log_likelihood'], a.get_best_fit()['parameters'])
print("median",a.get_stats()['marginals'][0]['median'], a.get_stats()['marginals'][1]['median'], a.get_stats()['marginals'][2]['median'], a.get_stats()['marginals'][3]['median'], a.get_stats()['marginals'][4]['median'],  a.get_stats()['marginals'][5]['median'],  a.get_stats()['marginals'][6]['median'], a.get_stats()['marginals'][7]['median'])
print(a.get_stats()['marginals'][0]['1sigma'], a.get_stats()['marginals'][1]['1sigma'], a.get_stats()['marginals'][2]['1sigma'], a.get_stats()['marginals'][3]['1sigma'], a.get_stats()['marginals'][4]['1sigma'],  a.get_stats()['marginals'][5]['1sigma'],  a.get_stats()['marginals'][6]['1sigma'], a.get_stats()['marginals'][7]['1sigma'])




json.dump(parameters, open('params.json', 'w'))




