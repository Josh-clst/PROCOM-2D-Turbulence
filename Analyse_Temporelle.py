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

incr_scale = 1
time_increments = np.zeros((ww.shape[0]-incr_scale, ww.shape[1], ww.shape[2]))
print(time_increments.shape)

nb_im = time_increments.shape[0]

# %%

time_increments = (ww[incr_scale:,:,:]-ww[0:-incr_scale,:,:])[0:nb_im,:,:]


# %%

# Theiler window to avoid temporal correlations (not implemented yet)
theiler_window = incr_scale / 2

# Initialization of matrix
S2=np.zeros(nb_im)

# Initialization of information measures

# Skewness, flatness
skewness = np.zeros((nb_im))
flatness = np.zeros((nb_im))

# Shanon entropy
entropy = np.zeros((nb_im))
# Distance to Gaussian distribution (Kullback-Leiber divergence)
dist_gauss = np.zeros((nb_im))

# Gaussian distribution for reference
gauss_size = time_increments[0,:,:].flatten().shape[0]
gauss = np.random.normal(0, 1, size=gauss_size)

# 2d FFT
fft_2d = np.zeros((nb_im, ww.shape[1], ww.shape[2]), dtype=np.complex128)

# %%

for im_i in range(nb_im):
    print("Processing time increment ", im_i+1, " / ", nb_im)
    incr_data = time_increments[im_i,:,:].flatten()

    # Second order structure function
    S2[im_i] = np.mean(incr_data**2)

    # Skewness and flatness
    skewness[im_i] = skew(incr_data)
    flatness[im_i] = kurtosis(incr_data, fisher=False)

    # Shannon entropy
    entropy[im_i] = im.entropy(incr_data, approach="kl", k = 5)

    # Kullback-Leibler divergence to Gaussian
    dist_gauss[im_i] = im.kullback_leiber_divergence(incr_data , gauss ,approach="kl", k = 5)

    # 2d FFT
    fft_2d[im_i,:,:] = np.fft.fft2(time_increments[im_i,:,:])


# %%
save_graph = True

imgs = np.arange(nb_im)

fig1 = plt.figure(figsize=(8,6))
plt.plot(imgs, S2, label='S2')
plt.title('Second Order Structure Function')
plt.xlabel('Time Increments')
plt.ylabel('S2')
plt.grid(True)

# FFT Visualization
fig_fft, axs_fft = plt.subplots(1, 2, figsize=(12, 6))
magnitude_spectrum = np.abs(fft_2d[0,:,:])

axs_fft[0].imshow(np.fft.fftshift(magnitude_spectrum), cmap='viridis')
axs_fft[0].set_title('Magnitude Spectrum of First Time Increment')
axs_fft[0].axis('off')
phase_spectrum = np.angle(fft_2d[0,:,:])
axs_fft[1].imshow(np.fft.fftshift(phase_spectrum), cmap='twilight')
axs_fft[1].set_title('Phase Spectrum of First Time Increment')
axs_fft[1].axis('off')
plt.tight_layout()

# Data visualization of information measures via line plots
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 10))

axs2[0, 0].plot(imgs, dist_gauss, color='blue')
axs2[0, 0].set_title('Distance to Gaussian (K-L Divergence)')
axs2[0, 0].set_xlabel('Time Increments')
axs2[0, 0].set_ylabel('K-L Divergence')
axs2[0, 0].grid(True)
axs2[0, 1].plot(imgs, skewness, color='orange')
axs2[0, 1].set_title('Skewness')
axs2[0, 1].set_xlabel('Time Increments')
axs2[0, 1].set_ylabel('Skewness')
axs2[0, 1].grid(True)
axs2[1, 0].plot(imgs, flatness, color='green')
axs2[1, 0].set_title('Flatness')
axs2[1, 0].set_xlabel('Time Increments')
axs2[1, 0].set_ylabel('Flatness')
axs2[1, 0].grid(True)
axs2[1, 1].plot(imgs, entropy, color='red')
axs2[1, 1].set_title('Shannon Entropy')
axs2[1, 1].set_xlabel('Time Increments')
axs2[1, 1].set_ylabel('Entropy')
axs2[1, 1].grid(True)

# Data Visualization via histograms

fig3, axs3 = plt.subplots(2, 2, figsize=(12, 10))

axs3[0, 0].hist(dist_gauss, bins=30, color='blue', alpha=0.7)
axs3[0, 0].set_title('Histogram of Distance to Gaussian (K-L Divergence)')
axs3[0, 0].set_xlabel('K-L Divergence')
axs3[0, 0].set_ylabel('Frequency')
axs3[0, 1].hist(skewness, bins=30, color='orange', alpha=0.7)
axs3[0, 1].set_title('Histogram of Skewness')
axs3[0, 1].set_xlabel('Skewness')
axs3[0, 1].set_ylabel('Frequency')
axs3[1, 0].hist(flatness, bins=30, color='green', alpha=0.7)
axs3[1, 0].set_title('Histogram of Flatness')
axs3[1, 0].set_xlabel('Flatness')
axs3[1, 0].set_ylabel('Frequency')
axs3[1, 1].hist(entropy, bins=30, color='red', alpha=0.7)
axs3[1, 1].set_title('Histogram of Shannon Entropy')
axs3[1, 1].set_xlabel('Entropy')
axs3[1, 1].set_ylabel('Frequency')

# Save graphs if required
if save_graph:
    fig_fft.savefig(save_dir + 'Temporal_Analysis_FFT.png', format='png', dpi=300)
    fig1.savefig(save_dir + 'Temporal_Analysis_S2.png', format='png', dpi=300)
    fig2.savefig(save_dir + 'Temporal_Analysis_Info_Measures.png', format='png', dpi=300)
    fig3.savefig(save_dir + 'Temporal_Analysis_Histograms.png', format='png', dpi=300)

plt.show()
# %%
