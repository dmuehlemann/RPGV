# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:19:48 2020

@author: Dirk




Check duration of weather regime
"""



from pathlib import Path
import xarray as xr



######################Load Datasets#################

data_folder = Path("../data/")

file_wr = data_folder / 'wr_time-c7_std_30days_lowpass_2_0-25.nc'
wr = xr.open_dataset(file_wr)
f_out = data_folder / 'wr_time-c7_std_30days_lowpass_2_0-25_short.nc'


#######Remove all days where weather regime lasts shorter than
alone = 0
twodays = 0
threedays = 0
fourdays = 0
rest = 0

for i in range(0,len(wr.wr.values)-1):
    if wr.wr.values[i] != wr.wr.values[i-1] and wr.wr.values[i] != wr.wr.values[i+1]:
        # print(i)
        alone = alone +1
        wr.wr[i] = 7
    
for i in range(0,len(wr.wr.values)-2):
    if wr.wr.values[i] != wr.wr.values[i-1] and wr.wr.values[i] == wr.wr.values[i+1] and wr.wr.values[i]!=wr.wr.values[i+2]:
      # print(i)
      twodays = twodays +1 
      wr.wr[i] = 7
      wr.wr[i+1] = 7
      
for i in range(0,len(wr.wr.values)-3):
    if wr.wr.values[i] != wr.wr.values[i-1] and wr.wr.values[i] == wr.wr.values[i+1] and wr.wr.values[i]==wr.wr.values[i+2] and wr.wr.values[i]!=wr.wr.values[i+3]:
        # print(i)
        threedays = threedays +1
        wr.wr[i] = 7
        wr.wr[i+1] = 7
        wr.wr[i+2] = 7
      
# for i in range(0,len(wr.wr.values)-4):
#     if wr.wr.values[i] != wr.wr.values[i-1] and wr.wr.values[i] == wr.wr.values[i+1] and wr.wr.values[i]==wr.wr.values[i+2] and wr.wr.values[i]==wr.wr.values[i+3] and wr.wr.values[i]!=wr.wr.values[i+4]:
#         # print(i)
#         fourdays = fourdays +1
#         wr.wr[i] = 7
#         wr.wr[i+1] = 7
#         wr.wr[i+2] = 7
#         wr.wr[i+3] = 7
remove = alone + 2*twodays + 3*threedays + 4*fourdays   
rest = len(wr.wr.values) - alone - 2*twodays - 3*threedays - 4*fourdays

wr.to_netcdf(f_out)
