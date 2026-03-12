# %%

import numpy as np
import os

#import h5py as h5 # for saving the results
import netCDF4

from skimage.draw import line

import matplotlib.pyplot as plt

import infomeasure as im # to compute information measures
from dirs import dir, save_dir

# Parameters definition
sim_list = ['01', '02']
n_sim = len(sim_list)
scaleth_list= [75, 50]
cutoff_radius = [-6.2, -5.3, -4.6]

if "03" in sim_list:
    K = 4
else:
    K = 32

data_chunks = [[] for k in range(len(cutoff_radius)+1)]
coeffs_chunks = [[] for k in range(len(cutoff_radius)+1)]

n_angles = 8

# Plot angular profiles: 4 panels for skewness, flatness, entropy, distance-to-gauss
fig4, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
titles = ['Skewness', 'Flatness', 'Shannon entropy', 'Distance to Gaussian']

for j in range(n_sim):
    sim_n = sim_list[j]
    scaleth = scaleth_list[j]

    radius_angle = []
    skewness_angle = []
    flatness_angle = []
    entropy_angle = []
    dist_gauss_angle = []

    path= dir + 'sim_' + sim_n + '/' + 'vars.nc'
    out_dir = save_dir + 'sim_' + sim_n + '/'
    # Ensure output directory exists to avoid "No such file or directory" when saving
    os.makedirs(out_dir, exist_ok=True)

    file2read = netCDF4.Dataset(path,'r')
    #print(file2read.variables.keys())

    time = file2read.variables['time']  # Time
    x = file2read.variables['x']  # X
    y = file2read.variables['y']  # Y
    ww = file2read.variables['q']  # vorticity

    ww = np.squeeze(np.asarray(ww))
    box_size = 12.56
    ls= box_size / ww.shape[1] # sampling distance
    timei= 9 * ww.shape[0] // 10 

    WW=ww[timei,:,:]
    print("WW shape :", WW.shape)

    scale_dir = f'scales_1-{scaleth}/'
    # Ensure scale-specific output directory exists
    if not os.path.exists(out_dir + scale_dir):
        os.makedirs(out_dir + scale_dir, exist_ok=True)

    # Compute and plot energy spectrum
    fft_ww = np.fft.fft2(WW)
    fft_ww_shifted = np.fft.fftshift(fft_ww)
    energy_spectrum = np.abs(fft_ww_shifted)**2
    
    # Compute radial wavenumber
    ny, nx = WW.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, ls)
    ky = 2 * np.pi * np.fft.fftfreq(ny, ls)
    kx_shifted = np.fft.fftshift(kx)
    ky_shifted = np.fft.fftshift(ky)
    Kx, Ky = np.meshgrid(kx_shifted, ky_shifted)
    k_radial = np.sqrt(Kx**2 + Ky**2)
    
    # Bin the energy by wavenumber
    k_bins = np.linspace(0, np.max(k_radial), 100)
    E_k = np.zeros(len(k_bins)-1)
    for i in range(len(k_bins)-1):
        mask = (k_radial >= k_bins[i]) & (k_radial < k_bins[i+1])
        E_k[i] = np.mean(energy_spectrum[mask]) if np.any(mask) else 0
    
    k_centers = (k_bins[:-1] + k_bins[1:]) / 2

    # Plot energy spectrum
    fig_spectrum, ax_spectrum = plt.subplots(figsize=(10, 6))
    ax_spectrum.plot(np.log(k_centers), np.log(E_k))
    ax_spectrum.set_xlabel('Wavenumber k')
    ax_spectrum.set_ylabel('Energy E(k)')
    ax_spectrum.set_title(f'Energy Spectrum - Simulation {sim_n}')
    ax_spectrum.grid(True, which='both', alpha=0.3)
    ax_spectrum.axvline(x = np.log(K), color='black', linestyle='--', label='Energy Injection Radius')
    fig_spectrum.savefig(out_dir + scale_dir + f'Energy-spectrum-{scaleth}.png', dpi=300)

    # Definition of scales
    scales=np.arange(-scaleth,scaleth+1,1)

    # Initialization of information measures

    S2=np.zeros((len(scales),len(scales)))

    # Skewness, flatness
    skewness = np.zeros((len(scales), len(scales)))
    flatness = np.zeros((len(scales), len(scales)))

    # Shanon entropy
    entropy = np.zeros((len(scales), len(scales)))

    # Distance to Gaussian distribution (Kullback-Leiber divergence)
    dist_gauss = np.zeros((len(scales), len(scales)))

    data = np.load(out_dir + scale_dir + f'Vorticity_Information_Measures_Nanalyse1024_scales1-{scaleth}.npz')
    S2 = data['S2']
    skewness = data['skewness']
    flatness = data['flatness']
    entropy = data['entropy']
    dist_gauss = data['dist_gauss']
    scalesx = data['scalesx']
    scalesy = data['scalesy']

    meshgrid = np.meshgrid(scales,scales)

    cnt_row, cnt_col = scaleth, scaleth

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

        radius_angle.append(np.log(np.sqrt( (np.float64(rr-scaleth)**2 + np.float64(cc-scaleth)**2) ) / ww.shape[1]))

        skewness_angle.append(skewness[rr,cc])
        flatness_angle.append(np.log((flatness[rr,cc]+3)/3))
        entropy_angle.append(entropy[rr,cc])
        dist_gauss_angle.append(dist_gauss[rr,cc])



    data_lists = [skewness_angle, flatness_angle, entropy_angle, dist_gauss_angle]

    for i, ax in enumerate(axes):

        n_cutoff = len(cutoff_radius)
        
        # Aggregate data from all angles
        r_all = np.concatenate(radius_angle)
        d_all = np.concatenate([data_lists[i][k] for k in range(n_angles)])
        
        if i == 2:  # Regression for first simulation plotted foir the entropy graph
            for j in range(-1, n_cutoff):
                cutoff_1 = cutoff_radius[j] if j != -1 else -np.inf
                cutoff_2 = cutoff_radius[j+1] if j != n_cutoff-1 else np.inf

                mask = np.all([r_all > cutoff_1, r_all < cutoff_2], axis=0)

                r_fit = r_all[mask]
                d_fit = d_all[mask]

                if r_fit.size > 1:
                    coeffs = np.polyfit(r_fit, d_fit, 1)
                    
                    if coeffs[0] >= 0:
                        coeffs_chunks[j].append(coeffs)
                        data_chunks[j].append(r_fit)                        
        
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

ax = axes[2]
vertical_line_position =  np.log(1/K * 2/box_size)
ax.axvline(x=vertical_line_position, color='black', linestyle='--', label='Energy Injection Radius')
colors = ["C"+str(i) for i in range(len(data_chunks))]
for i, r_fit in enumerate(data_chunks):
    
    data = np.concatenate(r_fit)
    coeff_list = coeffs_chunks[i]
    coeffs = np.zeros((len(coeff_list),2))
    for j, elt in enumerate(coeff_list):
        coeffs[j,:] = elt
    coeffs = np.mean(coeffs, axis = 0)

    fit_line = np.polyval(coeffs, data) + 0.15
    ax.plot(data, fit_line, linestyle='--', color=colors[i], label="Slope: {:.2f}".format(coeffs[0]))
ax.legend()

plt.tight_layout()
plt.show()

out_dir = save_dir + 'angular_profiles/'

scale_name = ""
for scaleth in scaleth_list:
    scale_name += f'{scaleth}+'

scale_name = scale_name[:-1]

sim_dir = ''
for sim in sim_list:
    sim_dir += f'sim_{sim}_'

sim_dir = sim_dir[:-1] + '/'

# Ensure scale-specific output directory exists
if not os.path.exists(out_dir + sim_dir):
    os.makedirs(out_dir + sim_dir, exist_ok=True)

save_graph = True

if save_graph:
    fig4.savefig(out_dir + sim_dir + f'Vorticity_Angular_Profiles_Nanalyse1024_scales1-{scale_name}-Injection.png', dpi=300)

# %%
