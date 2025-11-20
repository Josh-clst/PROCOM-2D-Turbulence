"""
This python ...

authors: C. Granero-Belinchon (IMT- Atlantique).

date: 09/2021
"""
#%% Import Libraries

import numpy as np
import sys
import os

from Increments import Incrs_anisotropic_generator2d

from dirs import dir, save_dir

from scipy.stats import kurtosis, skew

#import h5py as h5 # for saving the results
import netCDF4

import matplotlib.pyplot as plt

import infomeasure as im # to compute information measures

#%%
sim_n = '01'
path= dir + 'sim_' + sim_n + '/' + 'vars_k32_v2.nc'
save_dir = save_dir + 'sim_' + sim_n + '/'
# Ensure output directory exists to avoid "No such file or directory" when saving
os.makedirs(save_dir, exist_ok=True)

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
print("WW shape :", WW.shape)

# Standard deviation estimation
sigma=np.std(WW)

# Normalization
WW=(WW)/sigma

N=len(WW)

# Parameters definition
Nanalyse=2**10 # number of increments to analyse (512 / 1024 is a good compromise between statistical convergence and computation time)
Nreal=2 # number of realizations for the statistics
scaleth=40 # maximum scale to analyse


scale_dir = f'scales_1-{scaleth}/'
# Ensure scale-specific output directory exists
if not os.path.exists(save_dir + scale_dir):
    os.makedirs(save_dir + scale_dir, exist_ok=True)

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
radius = np.zeros((len(scales), len(scales)))
angle = np.zeros((len(scales), len(scales)))

#%%

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

            # radius[isy,isx] = np.sqrt(scalex**2 + scaley**2)
            # angle[isy,isx] = np.arctan2(scaley, scalex)

            # Estimation of information measures
            for ir in range(Nreal):
                S2[ir,isy,isx] = np.mean(incrs[ir,:]**2)
                skewness[ir,isy,isx] = skew(incrs[ir,:], bias=True)
                flatness[ir,isy,isx] = kurtosis(incrs[ir,:], bias=True)
                entropy[ir,isy,isx] = im.entropy(incrs[ir,:], approach="kl", k = 5)
                dist_gauss[ir,isy,isx] = im.kullback_leiber_divergence(incrs[ir,:] , gauss[ir,:] ,approach="kl", k = 5)

# %%
# Visualization of the information measures
save_graphs = True

fig1 = plt.figure()
plt.imshow(np.mean(S2,axis=0), extent=[-scaleth*ls, scaleth*ls, -scaleth*ls, scaleth*ls])
plt.colorbar()
plt.title('Second order structure function')
plt.xlabel('Scale x')
plt.ylabel('Scale y')
plt.show()

fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

im0 = axes[0, 0].imshow(np.mean(dist_gauss, axis=0),
                        extent=[-scaleth*ls, scaleth*ls, -scaleth*ls, scaleth*ls],
                        origin='lower')
axes[0, 0].set_title('Distance to Gaussian distribution')
axes[0, 0].set_xlabel('Scale x')
axes[0, 0].set_ylabel('Scale y')
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(np.mean(skewness, axis=0),
                        extent=[-scaleth*ls, scaleth*ls, -scaleth*ls, scaleth*ls],
                        origin='lower')
axes[0, 1].set_title('Skewness of increments')
axes[0, 1].set_xlabel('Scale x')
axes[0, 1].set_ylabel('Scale y')
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[1, 0].imshow(np.mean(flatness, axis=0),
                        extent=[-scaleth*ls, scaleth*ls, -scaleth*ls, scaleth*ls],
                        origin='lower')
axes[1, 0].set_title('Flatness of increments')
axes[1, 0].set_xlabel('Scale x')
axes[1, 0].set_ylabel('Scale y')
plt.colorbar(im2, ax=axes[1, 0])

im3 = axes[1, 1].imshow(np.mean(entropy, axis=0),
                        extent=[-scaleth*ls, scaleth*ls, -scaleth*ls, scaleth*ls],
                        origin='lower')
axes[1, 1].set_title('Shannon entropy of increments')
axes[1, 1].set_xlabel('Scale x')
axes[1, 1].set_ylabel('Scale y')
plt.colorbar(im3, ax=axes[1, 1])

plt.tight_layout()
plt.show()

# Histograms of the information measures
fig3, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(dist_gauss.flatten(), bins=50, color='blue', alpha=0.7)
axes[0, 0].set_title('Histogram of Distance to Gaussian distribution')
axes[0, 0].set_xlabel('Kullback-Leibler Divergence')
axes[0, 0].set_ylabel('Frequency')
axes[0, 1].hist(skewness.flatten(), bins=50, color='orange', alpha=0.7)
axes[0, 1].set_title('Histogram of Skewness')
axes[0, 1].set_xlabel('Skewness')
axes[0, 1].set_ylabel('Frequency')
axes[1, 0].hist(flatness.flatten(), bins=50, color='green', alpha=0.7)
axes[1, 0].set_title('Histogram of Flatness')
axes[1, 0].set_xlabel('Flatness')
axes[1, 0].set_ylabel('Frequency')
axes[1, 1].hist(entropy.flatten(), bins=50, color='red', alpha=0.7)
axes[1, 1].set_title('Histogram of Shannon Entropy')
axes[1, 1].set_xlabel('Entropy')
axes[1, 1].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

if save_graphs:
    # Saving the information measures
    fig1.savefig(save_dir + scale_dir + f'Vorticity_S2_Image_Nanalyse1024_scales1-{scaleth}.png', dpi=300)
    fig2.savefig(save_dir + scale_dir + f'Vorticity_Results_Nanalyse1024_scales1-{scaleth}.png', dpi=300)
    fig3.savefig(save_dir + scale_dir + f'Vorticity_Histograms_Nanalyse1024_scales1-{scaleth}.png', dpi=300)

# np.savez(save_dir + f'Vorticity_S2_Image_Nanalyse1024_scales1-{scaleth}.npz', S2=S2, scalesx=scales, scalesy=scales, N=N)


# %%
