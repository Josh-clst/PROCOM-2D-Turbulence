#!/usr/bin/env python3

#%% Import Libraries

import numpy as np
import matplotlib.pyplot as plt

#import h5py as h5 # for saving the results
import netCDF4


#%%
path='/Users/c20grane/Desktop/Research/ANR_SCALES/PhD_Causality/2D/Simus/vars_k4.nc'

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

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
fig.subplots_adjust(wspace=0.15, hspace=0.2)
img=ax.pcolormesh(x,y,WW,cmap='seismic', rasterized=True)
fig.colorbar(img)
#plt.savefig('Vorticity_k4.pdf',format='pdf',bbox_inches='tight')
plt.show()

