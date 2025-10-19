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
print(ww.shape)

incr_scale = 10

time_increments = np.zeros((ww.shape[0]-incr_scale, ww.shape[2], ww.shape[3]))

for t in range(ww.shape[0]-incr_scale):
    print(f'Processing time step {t+1} / {ww.shape[0]-incr_scale}')
    frame_t = np.squeeze(np.asarray(ww[t,:,:]))
    frame_t_plus_scale = np.squeeze(np.asarray(ww[t + incr_scale - 1,:,:]))
    increments_t = frame_t_plus_scale - frame_t
    time_increments[t,:,:] = increments_t

np.savez(dir + 'Vorticity_Time_Increments_Scale' + str(incr_scale) + '.npz', time_increments=time_increments)
