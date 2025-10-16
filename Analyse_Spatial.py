"""
This python ...

authors: C. Granero-Belinchon (IMT- Atlantique).

date: 09/2021
"""
#%% Import Libraries

import numpy as np
import sys

from Increments import Incrs_anisotropic_generator2d

#import h5py as h5 # for saving the results
import netCDF4

import matplotlib.pyplot as plt

#%%
dir = 'D:/IMT/3A/PRO_COM/transfer_10818383_files_96d3ed0a/'
path= dir + 'vars_k32_v2.nc'

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

# Definition of scales
scales=np.arange(-10,11,1)

# Parameters definition
Nanalyse=2**10 # number of increments to analyse (512 / 1024 is a good compromise between statistical convergence and computation time)
Nreal=2 # number of realizations for the statistics
scaleth=10 # Theiler to avoid correlation between increments

# Initialization of matrix
S2=np.zeros((Nreal, len(scales),len(scales)))
       
# Estimation of S2 for each combination of scales (scalex,scaley)
for isx in range(len(scales)): #x dimension
    for isy in range(len(scales)): # y dimension
        print(isx,"/",len(scales), '----' , isy,"/",len(scales))
        
        scalex=scales[isx]
        scaley=scales[isy]
        
        if scalex!=0 or scaley !=0:
            # Generation of increments
            incrs = Incrs_anisotropic_generator2d(WW, scalex, scaley)
            
            # Theiler + random permutation + reshape
            incrs=incrs[0:-1:scaleth,0:-1:scaleth].flatten()
            
            incrs = np.random.permutation(incrs)[0:int(len(incrs)/Nanalyse)*Nanalyse].reshape(-1,Nanalyse)
            #print(incrsx.shape)
            
            for ir in range(Nreal):
                S2[ir,isy,isx] = np.mean(incrs[ir,:]**2)

np.savez(dir + 'Vorticity_S2_Image_Nanalyse1024_scales1-100.npz', S2=S2, scalesx=scales, scalesy=scales, N=N)






