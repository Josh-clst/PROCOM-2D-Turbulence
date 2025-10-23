#%% Import Libraries

import numpy as np

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

print(ww.shape)

incr_scale = 10
time_increments = np.zeros((incr_scale-1, ww.shape[0]-incr_scale, ww.shape[1], ww.shape[2]))
print(time_increments.shape)

for scale in range(1,incr_scale):
    print("Processing scale ", scale, " out of ", incr_scale-1)
    time_increments[scale,:,:,:] = (ww[scale::,:,:]-ww[0:-scale,:,:])[0:scale-incr_scale,:,:]

# Theiler window to avoid temporal correlations (not implemented yet)



# %%
