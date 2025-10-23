"""
This python ...

authors: C. Granero-Belinchon (IMT- Atlantique).

date: 09/2021
"""
#%% Import Libraries

import numpy as np
import sys

from Increments import Incrs_anisotropic_generator2d

from scipy.stats import kurtosis, skew

#import h5py as h5 # for saving the results
import netCDF4

import matplotlib.pyplot as plt

import infomeasure as im # to compute information measures

#%%
dir = 'D:/IMT/3A/PRO_COM/PROCOM-2D-Turbulence/simulations/input/'
sim_n = '01'
path= dir + 'sim_' + sim_n + '/' + 'vars_k32_v2.nc'

file2read = netCDF4.Dataset(path,'r')
#print(file2read.variables.keys())

time = file2read.variables['time']  # Time
x = file2read.variables['x']  # X
y = file2read.variables['y']  # Y
ww = file2read.variables['q']  # vorticity


ww = np.squeeze(np.asarray(ww))
ls=12.56/ww.shape[1] # sampling distance
timei=900

WW=ww[timei,:,:]

# Standard deviation estimation
sigma=np.std(WW)

# Normalization
WW=(WW)/sigma

N=len(WW)

# Parameters definition
Nanalyse=2**10 # number of increments to analyse (512 / 1024 is a good compromise between statistical convergence and computation time)
Nreal=2 # number of realizations for the statistics
scaleth=20 # maximum scale to analyse

# Definition of scales
scales=np.arange(-scaleth,scaleth+1,1)

# Initialization of matrix
S2=np.zeros((Nreal, len(scales),len(scales)))

# Initialization of information measures

# Skewness, flatness
skewness = np.zeros((Nreal, len(scales), len(scales)))
flatness = np.zeros((Nreal, len(scales), len(scales)))

# Shanon entropy
entropy = np.zeros((Nreal, len(scales), len(scales)))

# Distance to Gaussian distribution (Kullback-Leiber divergence)
dist_gauss = np.zeros((Nreal, len(scales), len(scales)))

# Radius and angle of the increments
radius = np.zeros((Nreal, len(scales), len(scales)))
angle = np.zeros((Nreal, len(scales), len(scales)))

# Estimation of all information measures
for isx in range(len(scales)): #x dimension
    for isy in range(len(scales)): # y dimension
        print(isx,"/",len(scales), '----' , isy,"/",len(scales))
        
        scalex=scales[isx]
        scaley=scales[isy]
        
        if scalex!=0 or scaley !=0:
            # Generation of increments
            incrs = Incrs_anisotropic_generator2d(WW, scalex, scaley)

            # flatten incrs to get a vector
            incrs = incrs.flatten()

            incrs = np.random.permutation(incrs)[0:int(len(incrs)/Nanalyse)*Nanalyse].reshape(-1,Nanalyse)

            # create Gaussian realizations matching each increment-realization's mean and std
            means = np.mean(incrs, axis=1)
            stds = np.std(incrs, axis=1, ddof=0)
            stds[stds == 0] = 1e-12
            gauss = np.random.normal(loc=means[:, None], scale=stds[:, None], size=incrs.shape)

            # Estimation of information measures
            for ir in range(Nreal):
                S2[ir,isy,isx] = np.mean(incrs[ir,:]**2)
                skewness[ir,isy,isx] = skew(incrs[ir,:])
                flatness[ir,isy,isx] = kurtosis(incrs[ir,:])
                entropy[ir,isy,isx] = im.entropy(incrs[ir,:], approach="metric")
                dist_gauss[ir,isy,isx] = im.kullback_leiber_divergence(incrs[ir,:] , gauss[ir,:] ,approach="metric")

np.savez(dir + 'Vorticity_S2_Image_Nanalyse1024_scales1-100.npz', S2=S2, scalesx=scales, scalesy=scales, N=N)







# %%
