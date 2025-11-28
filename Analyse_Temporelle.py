#%% 
# Import Libraries

import numpy as np

#import h5py as h5 # for saving the results
import netCDF4

import matplotlib.pyplot as plt

from dirs import dir, save_dir

import os

import infomeasure as im # to compute information measures

from scipy.stats import kurtosis, skew

sim_n = '01'
path= dir + 'sim_' + sim_n + '/' + 'vars_k32_v2.nc'
save_dir = save_dir + 'sim_' + sim_n + '/' + 'Temporal_Analysis/'
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

incr_scale = 100
time_increments = np.zeros((ww.shape[0]-incr_scale, ww.shape[1], ww.shape[2]))
print(time_increments.shape)

nb_im = time_increments.shape[0]

# Initialization of matrix
S2=np.zeros(incr_scale)

# Initialization of information measures

# Skewness, flatness
skewness = np.zeros((incr_scale))
flatness = np.zeros((incr_scale))

# Shanon entropy
entropy = np.zeros((incr_scale))

# Distance to Gaussian distribution (Kullback-Leiber divergence)
dist_gauss = np.zeros((incr_scale))

# FFT storage
fft = np.zeros((incr_scale, ww.shape[1], ww.shape[2]), dtype=np.complex128)

# %%
# Theiler window to avoid temporal correlations (not implemented yet)
theiler_window = 100

for scale_i in range(1,incr_scale+1):
    print(f'Analyzing scale {scale_i} / {incr_scale}')

    time_increments = (ww[scale_i:,:,:]-ww[:-scale_i,:,:])[0:nb_im,:,:]

    incr_data = time_increments.flatten()[::theiler_window]

    # Second order structure function
    S2[scale_i-1] = np.mean(incr_data**2)

    # Skewness and flatness
    skewness[scale_i-1] = skew(incr_data)
    flatness[scale_i-1] = kurtosis(incr_data, fisher=False)

    # Shannon entropy
    entropy[scale_i-1] = im.entropy(incr_data, approach="kl", k = 5)

    # Kullback-Leibler divergence to Gaussian
    dist_gauss[scale_i-1] = 1/2 * np.log(2 * np.pi * np.e * np.var(incr_data)) - entropy[scale_i-1]

# %%
# FFT computation
fft_scale = np.fft.fft(ww, axis=0)


# %%
save_values = True
load_values = False

# Saving the information measures
if save_values:
    np.savez(save_dir + f'Vorticity_Temporal_Information_Measures_incrscale{incr_scale}.npz',
            S2=S2,
            skewness=skewness,
            flatness=flatness,
            entropy=entropy,
            dist_gauss=dist_gauss,
            fft=fft_scale
            )
    
# Loading the information measures
if load_values:
    data = np.load(save_dir + f'Vorticity_Temporal_Information_Measures_incrscale{incr_scale}.npz')
    S2 = data['S2']
    skewness = data['skewness']
    flatness = data['flatness']
    entropy = data['entropy']
    dist_gauss = data['dist_gauss']
    fft_scale = data['fft']

# %%
save_graph = True

fig1 = plt.figure(figsize=(8,6))
plt.plot(S2, label='S2')
plt.title('Second Order Structure Function')
plt.xlabel('Scale')
plt.ylabel('S2')
plt.xscale('log')
plt.grid(True)

# FFT Visualization
freq = np.arange(fft_scale.shape[0]) - (fft_scale.shape[0] // 2)

fig_fft, axs_fft = plt.subplots(1, 2, figsize=(12, 6))

phase_spectrum = np.angle(fft_scale)

fft_scale_abs = np.abs(fft_scale)
magnitude_spectrum = np.mean(fft_scale_abs, axis=(1,2))

axs_fft[0].plot(freq, np.fft.fftshift(magnitude_spectrum))
axs_fft[0].set_title('Magnitude Spectrum')
axs_fft[0].set_xlabel('Frequency')
axs_fft[0].set_ylabel('Magnitude')
axs_fft[0].set_yscale('log')

phase_spectrum_plot = np.mean(phase_spectrum, axis=(1,2))
axs_fft[1].plot(freq, np.fft.fftshift(phase_spectrum_plot))
axs_fft[1].set_title('Phase Spectrum')
axs_fft[1].set_xlabel('Frequency')
axs_fft[1].set_ylabel('Phase (radians)')
plt.tight_layout()

# Data visualization of information measures via line plots
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 10))

axs2[0, 0].plot(dist_gauss, color='blue')
axs2[0, 0].set_title('Distance to Gaussian (K-L Divergence)')
axs2[0, 0].set_xlabel('Scale')
axs2[0, 0].set_ylabel('K-L Divergence')
axs2[0, 0].set_xscale('log')
axs2[0, 0].grid(True)
axs2[0, 1].plot(skewness, color='orange')
axs2[0, 1].set_title('Skewness')
axs2[0, 1].set_xlabel('Scale')
axs2[0, 1].set_ylabel('Skewness')
axs2[0, 1].set_xscale('log')
axs2[0, 1].grid(True)
axs2[1, 0].plot(flatness, color='green')
axs2[1, 0].set_title('Flatness')
axs2[1, 0].set_xlabel('Scale')
axs2[1, 0].set_ylabel('Flatness')
axs2[1, 0].grid(True)
axs2[1, 0].set_xscale('log')
axs2[1, 1].plot(entropy, color='red')
axs2[1, 1].set_title('Shannon Entropy')
axs2[1, 1].set_xlabel('Scale')
axs2[1, 1].set_ylabel('Entropy')
axs2[1, 1].set_xscale('log')
axs2[1, 1].grid(True)
plt.tight_layout()


# Save graphs if required
if save_graph:
    fig_fft.savefig(save_dir + 'Temporal_Analysis_FFT.png', format='png', dpi=300)
    fig1.savefig(save_dir + 'Temporal_Analysis_S2.png', format='png', dpi=300)
    fig2.savefig(save_dir + 'Temporal_Analysis_Info_Measures.png', format='png', dpi=300)

plt.show()
# %%
