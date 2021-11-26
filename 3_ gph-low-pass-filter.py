# -*- coding: utf-8 -*-
"""
Created on Sam Aug  7 11:50:05 2020

@author: Dirk

This scripts applies a 10day low pass filter to the ERA5 gph daily means

"""

import scipy.signal as signal
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr


#Define input and output data
data_folder = Path("../data/")
filename = data_folder / 'gph-daily-mean.nc'

data_out = data_folder / 'gph-daily-mean-lowpass_2_0-1.nc'
fig_out = data_folder / 'fig/gph-daily-mean-lowpass_2_0-1.png'


#Load data
z_all = xr.open_dataset(filename)


# First, design the Buterworth filter
N  = 2   # Filter order
Wn = 0.1 # Cutoff frequency
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
d = 10000
a=10150
b=100
c=150
for i in range(0,10):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    plt.plot(z_all.z[d:a, b, c], 'b-')
    plt.plot(z_allf.z[d:a, b, c], 'r-',)
    plt.ylabel("Geopotential height")
    plt.legend(['Original','Filtered'])
    plt.title("4-day lowpass filtered geopotential height")
    ax1.axes.get_xaxis().set_visible(False)
    
    ax1 = fig.add_subplot(212)
    plt.plot(z_all.z[d:a, b, c]-z_allf.z[d:a, b, c], 'b-')
    plt.ylabel("Geopotential height")
    plt.xlabel("Days")
    plt.legend(['Residuals'])
    name= 'fig/filter/gph-daily-mean-lowpass_2_0-25_150d'+str(i)+'.png'
    a = a +5
    b = b +5
    c = c+5
    d = d +5
    fig.savefig(data_folder / name)


#save results and plot
# z_allf.to_netcdf(data_out)
# fig.savefig(fig_out)


