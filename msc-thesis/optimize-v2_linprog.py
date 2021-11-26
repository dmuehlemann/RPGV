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
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import itertools as it
from scipy.optimize import linprog


######################Load Datasets#################
######################Load Datasets#################

data_folder = Path("../data/")


file_ninja = data_folder / 'ninja_and_wr_30days_lowpass_2_0-1_short10.nc'
ninja = xr.open_dataset(file_ninja)

ic_file = data_folder / 'source/installed_capacities_IRENA.csv'

ic = pd.read_csv(ic_file, header=0, parse_dates=[0], index_col=0, squeeze=True)




###here no Brexit ;-)
ninja = ninja.rename_vars({'GB':'UK'})
ic = ic.rename(columns={'GB':'UK'})
ic = ic.to_xarray()

####################################################


#########Create Testdata##########################

ninja_tot = []
ninja_season= []


##percentage deviation
mean_season = ninja.drop('wr').groupby('time.season').mean()
for i in range(0, int(ninja.wr.max()+1)):
    # ninja_tot.append(ninja.drop('wr').where(ninja.wr==i, drop=True).mean())# - ninja.drop('wr').mean())/ninja.drop('wr').mean())
    ninja_season.append(ninja.drop('wr').where(ninja.wr==i, drop=True).groupby('time.season').mean() \
                          - mean_season \
                          #/ ninja.drop('wr').groupby('time.season').mean()
                          )


tot = ninja.DE.groupby('time.season').mean().to_dataset().merge(ninja.ES.groupby('time.season').mean())
    
#create Testdata IC
ic_temp = ic.DE[9].to_dataset().merge(ic.ES[9])
ic_arr = ic_temp.to_array()




# # minimize_scalar(lambda ic_arr: func(ic_arr), bounds=(0,500), method='bounded', options={'maxiter': 10})
# test = np.append(ninja_season[2].sel(season='DJF').to_array().values,ninja_season[0].sel(season='DJF').to_array().values ).tolist()


# ##Hier müsste man wohl mal frequency rechnen
DE_sum = ninja_season[0]['DE']
ES_sum = ninja_season[0]['ES']
for i in range(1,len(ninja_season)):
    DE_sum = DE_sum + ninja_season[i]['DE']
    ES_sum = ES_sum + ninja_season[i]['ES']

c_1d = np.append(DE_sum.sel(season='DJF'), ES_sum.sel(season='DJF'))


c = np.append([ninja_season[0].sel(season='DJF').to_array().values],[ninja_season[1].sel(season='DJF').to_array().values], axis=0)
c_all = np.append([ninja_season[0].sel(season='DJF').to_array().values],[ninja_season[1].sel(season='DJF').to_array().values], axis=0)
for i in range(2,len(ninja_season)):
    c_all = np.append(c_all,[ninja_season[i].sel(season='DJF').to_array().values], axis=0)

# A = [[-3, 1], [1, 2]]
# b = [6, 4]

x0_bounds = (48960, 100000)
x1_bounds = (8761, 100000)
res = linprog(c_1d, bounds=[x0_bounds, x1_bounds])



