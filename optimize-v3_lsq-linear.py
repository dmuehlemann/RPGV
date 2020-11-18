# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:33:08 2020

@author: Dirk
"""


import numpy as np
from pathlib import Path
# import cartopy.crs as ccrs
# import matplotlib.pyplot as plt
import xarray as xr
# import calendar
import pandas as pd
# import matplotlib as mpl
# import geopandas as gpd
# from mapclassify import Quantiles, UserDefined
from scipy.optimize import lsq_linear


######################LOAD DATASET#################


data_folder = Path("../data/")


file_ninja = data_folder / 'ninja_and_wr_30days_lowpass_2_0-1_short10.nc'
ninja = xr.open_dataset(file_ninja)

ic_file = data_folder / 'source/installed_capacities_IRENA.csv'

ic = pd.read_csv(ic_file, header=0, parse_dates=[0], index_col=0, squeeze=True)



#####################END LOAD DATASET#####################################


######################CREATE NEEDED DATASETS#########################
###here no Brexit ;-)
ninja = ninja.rename_vars({'GB':'UK'})
ic = ic.rename(columns={'GB':'UK'})
ic = ic.to_xarray()


# ninja_tot = []
ninja_season= []
##calculate delata CF for each WR
mean_season = ninja.drop('wr').groupby('time.season').mean()
for i in range(0, int(ninja.wr.max()+1)):
    # ninja_tot.append(ninja.drop('wr').where(ninja.wr==i, drop=True).mean())# - ninja.drop('wr').mean())/ninja.drop('wr').mean())
    ninja_season.append(ninja.drop('wr').where(ninja.wr==i, drop=True).groupby('time.season').mean() \
                          - mean_season \
                          #/ ninja.drop('wr').groupby('time.season').mean()
                          )

#Create array (Matrix A) for DJF with all WR and countries
A = np.append([ninja_season[0].sel(season='DJF').to_array().values],[ninja_season[1].sel(season='DJF').to_array().values], axis=0)
for i in range(2,len(ninja_season)):
    A = np.append(A,[ninja_season[i].sel(season='DJF').to_array().values], axis=0)

#create array for all available IC 
b = 0
for i in ninja:
    for a in ic:
        if i==a:
            if b==0:
                ic_reduced = xr.DataArray(ic[a][-1])
                b = b +1
            else:
                ic_reduced = xr.merge([ic_reduced, ic[a][-1]])



######################END CREATE DATASET#########################


######################DEFINE CONSTRAINTS AND BOUNDS#########################                          
#Define lower and upper bound with already installed capacity   
lb = ic_reduced.to_array().values
ub = lb *5
lb_null = ic_reduced.to_array().values *0
ub_inf = np.array(np.ones(lb_null.shape) * np.inf)



#Define vector b with zeros for variability and at the end value for tot installed capcity
b = np.zeros(A.shape[0])
#Add tot IC constraint
A = np.append(A, [np.ones(A.shape[1])], 0)
b = np.append(b, ic_reduced.to_array().sum()*10)      


#Define total production
P = mean_season.sel(season='DJF').to_array() * ic_reduced.to_array()



######################END DEFINIDTIONn#########################      


###########calculate least sqaure with matrix A and vector b and evaluate result#########################      
#Calc LSQ
res = lsq_linear(A, b, bounds=(lb,ub))

dif = res.x -lb
ic_reduced.to_array()[np.where(dif>1)]
dif[np.where(dif>1)]




