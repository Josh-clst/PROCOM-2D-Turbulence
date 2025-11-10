#%% Import Libraries

import numpy as np

#import h5py as h5 # for saving the results
import netCDF4

import matplotlib.pyplot as plt

from dirs import dir, save_dir

import os

import infomeasure as im # to compute information measures

from scipy.stats import kurtosis, skew

#%%
sim_n = '01'
path= dir + 'sim_' + sim_n + '/' + 'vars_k32_v2.nc'
save_dir = save_dir + 'sim_' + sim_n + '/'
# Ensure output directory exists to avoid "No such file or directory" when saving
os.makedirs(save_dir, exist_ok=True)

file2read = netCDF4.Dataset(path, 'r')
#print(file2read.variables.keys())

time = file2read.variables['time']  # Time
x = file2read.variables['x']  # X
y = file2read.variables['y']  # Y
ww = file2read.variables['q']  # vorticity
ww = np.squeeze(np.asarray(ww))

print(ww.shape)

incr_scale = 5
time_increments = np.zeros((incr_scale, ww.shape[0]-incr_scale, ww.shape[1], ww.shape[2]))
print(time_increments.shape)

nb_im = time_increments.shape[1]

# %%

for scale in range(1,incr_scale+1):
    print("Processing scale ", scale, " out of ", incr_scale)
    time_increments[scale-1,:,:,:] = (ww[scale:,:,:]-ww[0:-scale,:,:])[0:nb_im,:,:]


# %%

# Theiler window to avoid temporal correlations (not implemented yet)
theiler_window = incr_scale / 2

# Initialization of matrix
S2=np.zeros((incr_scale,nb_im))

# Initialization of information measures

# Skewness, flatness
skewness = np.zeros((incr_scale))
flatness = np.zeros((incr_scale))

# Shanon entropy
entropy = np.zeros((incr_scale))
# Distance to Gaussian distribution (Kullback-Leiber divergence)
dist_gauss = np.zeros((incr_scale))

# Gaussian distribution for reference
gauss_size = time_increments[0,:,:,:].flatten().shape[0]
gauss = np.random.normal(0, 1, size=gauss_size)


for scale in range(1,incr_scale):
    print("Computing statistics for scale ", scale, " out of ", incr_scale-1)
    incrs = time_increments[scale,:,:,:].flatten()
        
    # Second order structure function
    S2[scale-1] = np.mean(incrs**2)
    
    # Skewness and flatness
    skewness[scale-1] = skew(incrs, bias=True)
    flatness[scale-1] = kurtosis(incrs, bias=True)
    
    # Shanon entropy    
    entropy[scale-1] = im.entropy(incrs, approach="kl", k = 5)
    
    # Distance to Gaussian distribution (Kullback-Leiber divergence)
    dist_gauss[scale-1] = im.kullback_leiber_divergence(incrs, gauss, approach="kl", k = 5)


# %%
save_graph = True

scales = np.arange(1, incr_scale)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].loglog(scales, S2, marker='o')
axs[0, 0].set_title('Second Order Structure Function')
axs[0, 0].set_xlabel('Scale (time steps)')
axs[0, 0].set_ylabel('S2')
axs[0, 0].grid(True)
axs[0, 1].loglog(scales, skewness, marker='o', color='orange')
axs[0, 1].set_title('Skewness')
axs[0, 1].set_xlabel('Scale (time steps)')
axs[0, 1].set_ylabel('Skewness')
axs[0, 1].grid(True)
axs[1, 0].loglog(scales, flatness, marker='o', color='green')
axs[1, 0].set_title('Flatness')
axs[1, 0].set_xlabel('Scale (time steps)')
axs[1, 0].set_ylabel('Flatness')
axs[1, 0].grid(True)
axs[1, 1].loglog(scales, entropy, marker='o', color='red')
axs[1, 1].set_title('Shannon Entropy')
axs[1, 1].set_xlabel('Scale (time steps)')
axs[1, 1].set_ylabel('Entropy')
axs[1, 1].grid(True)
plt.tight_layout()
if save_graph:
    plt.savefig(save_dir + 'Temporal_Analysis_Results.pdf', format='pdf', dpi=300)
plt.show()
# %%
