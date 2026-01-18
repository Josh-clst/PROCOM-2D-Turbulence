# %%

import numpy as np
import sys
import os

from Increments import Incrs_anisotropic_generator2d

from scipy.stats import kurtosis, skew

#import h5py as h5 # for saving the results
import netCDF4

from skimage.draw import line

import matplotlib.pyplot as plt

import infomeasure as im # to compute information measures
from dirs import dir, save_dir

# Parameters definition
Nanalyse=2**10 # number of increments to analyse (512 / 1024 is a good compromise between statistical convergence and computation time)
scaleth= 50 # maximum scale to analyse
Nreal = 5 # number of realizations to average the information measures over
n_angles = 8

sim_n = '04'
path= dir + 'sim_' + sim_n + '/' + 'vars.nc'
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
timei= 9 * ww.shape[0] // 10 

WW=ww[timei,:,:]
print("WW shape :", WW.shape)

plt.imshow(WW, label='Vorticity field snapshot', cmap='jet')

# Standard deviation estimation
sigma=np.std(WW)

# Normalization
WW=(WW)/sigma

N=len(WW)

scale_dir = f'scales_1-{scaleth}/'
# Ensure scale-specific output directory exists
if not os.path.exists(save_dir + scale_dir):
    os.makedirs(save_dir + scale_dir, exist_ok=True)

# Definition of scales
scales=np.arange(-scaleth,scaleth+1,1)

# Initialization of matrix
S2=np.zeros((len(scales),len(scales),Nreal))

# Initialization of information measures

# Skewness, flatness
skewness = np.zeros((len(scales), len(scales), Nreal))
flatness = np.zeros((len(scales), len(scales), Nreal))

# Shanon entropy
entropy = np.zeros((len(scales), len(scales), Nreal))

# Distance to Gaussian distribution (Kullback-Leiber divergence)
dist_gauss = np.zeros((len(scales), len(scales), Nreal))

load_values = True
scales = np.arange(-scaleth,scaleth+1,1)
scale_dir = f'scales_1-{scaleth}/'

if load_values:
    data = np.load(save_dir + scale_dir + f'Vorticity_Information_Measures_Nanalyse{Nanalyse}_scales1-{scaleth}.npz')
    S2 = data['S2']
    skewness = data['skewness']
    flatness = data['flatness']
    entropy = data['entropy']
    dist_gauss = data['dist_gauss']
    scalesx = data['scalesx']
    scalesy = data['scalesy']

meshgrid = np.meshgrid(scales,scales)

cnt_row, cnt_col = scaleth, scaleth

radius_angle = []
skewness_angle = []
flatness_angle = []
entropy_angle = []
dist_gauss_angle = []

for k in range(n_angles):

    dx = np.cos(2*k/n_angles*np.pi)
    dy = np.sin(2*k/n_angles*np.pi)

    dx = 0 if np.abs(dx) < 1e-8 else dx
    dy = 0 if np.abs(dy) < 1e-8 else dy
    
    t_min = scaleth / max(np.abs(dx),np.abs(dy))

    epixel_x = int(round(t_min*dx,0)) + scaleth
    epixel_y = int(round(t_min*dy,0)) + scaleth

    rr, cc = line(cnt_row,cnt_col,epixel_x,epixel_y)

    rr, cc = rr[1:], cc[1:]

    radius_angle.append(ls * np.sqrt( (np.float64(rr-scaleth)**2 + np.float64(cc-scaleth)**2) ) / 12.56)

    skewness_angle.append(skewness[rr,cc])
    flatness_angle.append(flatness[rr,cc])
    entropy_angle.append(entropy[rr,cc])
    dist_gauss_angle.append(dist_gauss[rr,cc])

# Preparing the data for plotting

radius_angle = [np.log(elt) for elt in radius_angle]
flatness_angle = [np.log((elt+3)/3) for elt in flatness_angle]

# Visualization of the information measures
save_graphs = True

fig1 = plt.figure()
plt.imshow(S2, extent=[-scaleth*ls, scaleth*ls, -scaleth*ls, scaleth*ls])
plt.colorbar()
plt.title('Second order structure function')
plt.xlabel('Scale x')
plt.ylabel('Scale y')
plt.show()

fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

im0 = axes[0, 0].imshow(dist_gauss,
                        extent=[-scaleth*ls, scaleth*ls, -scaleth*ls, scaleth*ls],
                        origin='lower')
axes[0, 0].set_title('Distance to Gaussian distribution')
axes[0, 0].set_xlabel('Scale x')
axes[0, 0].set_ylabel('Scale y')
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(skewness,
                        extent=[-scaleth*ls, scaleth*ls, -scaleth*ls, scaleth*ls],
                        origin='lower')
axes[0, 1].set_title('Skewness of increments')
axes[0, 1].set_xlabel('Scale x')
axes[0, 1].set_ylabel('Scale y')
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[1, 0].imshow(flatness,
                        extent=[-scaleth*ls, scaleth*ls, -scaleth*ls, scaleth*ls],
                        origin='lower')
axes[1, 0].set_title('Flatness of increments')
axes[1, 0].set_xlabel('Scale x')
axes[1, 0].set_ylabel('Scale y')
plt.colorbar(im2, ax=axes[1, 0])

im3 = axes[1, 1].imshow(entropy,
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

# Plot angular profiles: 4 panels for skewness, flatness, entropy, distance-to-gauss
fig4, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
titles = ['Skewness', 'Flatness', 'Shannon entropy', 'Distance to Gaussian']
data_lists = [skewness_angle, flatness_angle, entropy_angle, dist_gauss_angle]

for i, ax in enumerate(axes):

    cutoff_radius = [-6, -5]
    n_cutoff = len(cutoff_radius)
    
    # Aggregate data from all angles
    r_all = np.concatenate(radius_angle)
    d_all = np.concatenate([data_lists[i][k] for k in range(n_angles)])
    
    if i == 2:  # Entropy plot
        for j in range(-1, n_cutoff):
            cutoff_1 = cutoff_radius[j] if j != -1 else -np.inf
            cutoff_2 = cutoff_radius[j+1] if j != n_cutoff-1 else np.inf

            mask = np.all([r_all > cutoff_1, r_all < cutoff_2], axis=0)

            r_fit = r_all[mask]
            d_fit = d_all[mask]

            if r_fit.size > 1:
                coeffs = np.polyfit(r_fit, d_fit, 1)
                fit_line = np.polyval(coeffs, r_fit) + 0.15
                colors = ['red', 'green', 'blue']
                ax.plot(r_fit, fit_line, linestyle='--', color=colors[j], label="Slope: {:.2f}".format(coeffs[0]))
    
    # Plot all angles
    for k in range(n_angles):
        r = radius_angle[k]
        d = data_lists[i][k]
        m = min(r.size, d.size)
        ax.plot(r[:m], d[:m], linestyle='-')

    ax.set_title(titles[i])
    ax.set_xlabel('radius (log scale)' if np.any(np.isfinite(radius_angle)) else 'radius')
    ax.set_ylabel(titles[i])
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

if save_graphs:
    # Saving the information measures
    fig1.savefig(save_dir + scale_dir + f'Vorticity_S2_Image_Nanalyse1024_scales1-{scaleth}.png', dpi=300)
    fig2.savefig(save_dir + scale_dir + f'Vorticity_Results_Nanalyse1024_scales1-{scaleth}.png', dpi=300)
    fig3.savefig(save_dir + scale_dir + f'Vorticity_Histograms_Nanalyse1024_scales1-{scaleth}.png', dpi=300)
    fig4.savefig(save_dir + scale_dir + f'Vorticity_Angular_Profiles_Nanalyse{Nanalyse}_scales1-{scaleth}.png', dpi=300)
# %%
