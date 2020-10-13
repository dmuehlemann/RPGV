# -*- coding: utf-8 -*-
"""
Created on Sam Aug  7 11:50:05 2020

@author: Dirk
"""

import scipy.signal as signal
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr


#Define input and output data
data_folder = Path("../data/")
filename = data_folder / 'gph-daily-mean.nc'

data_out = data_folder / 'gph-daily-mean-lowpass_2_0-25.nc'
fig_out = data_folder / 'fig/gph-daily-mean-lowpass_2_0-25.png'


#Load data
z_all = xr.open_dataset(filename)


# First, design the Buterworth filter
N  = 2   # Filter order
Wn = 0.25 # Cutoff frequency
B, A = signal.butter(N, Wn, output='ba')


# temp = z_all.isel(latitude=10, longitude=10).z.loc["2000-01-01":"2005-01-01"]
# Second, apply the filter
z_allf = xr.apply_ufunc(
    signal.filtfilt, B, A, z_all,
    kwargs=dict(
        axis=0,
    )
)


# Make plots
fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.plot(z_all.z[10000:10200, 117, 145], 'b-')
plt.plot(z_allf.z[10000:10200, 117, 145], 'r-',)
plt.ylabel("Geopotential height")
plt.legend(['Original','Filtered'])
plt.title("4-day lowpass filtered geopotential height")
ax1.axes.get_xaxis().set_visible(False)

ax1 = fig.add_subplot(212)
plt.plot(z_all.z[10000:10200, 117, 145]-z_allf.z[10000:10200, 117, 145], 'b-')
plt.ylabel("Geopotential height")
plt.xlabel("Days")
plt.legend(['Residuals'])



#save results and plot
z_allf.to_netcdf(data_out)
fig.savefig(fig_out)


