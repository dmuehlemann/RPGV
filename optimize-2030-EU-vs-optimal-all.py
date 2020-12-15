# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:59:12 2020

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
import geopandas as gpd


######################LOAD DATASET#################


data_folder = Path("../data/")


file_ninja = data_folder / 'ninja_and_wr_30days_lowpass_2_0-1_short10.nc'
ninja = xr.open_dataset(file_ninja)

ic_file = data_folder / 'source/installed_capacities_IRENA.csv'
ic_2030 = data_folder / 'source/planned-IC-2030-EU.xlsx'


ic = pd.read_csv(ic_file, header=0, parse_dates=[0], index_col=0, squeeze=True)
ic_2030 = pd.read_excel(ic_2030, header=0, parse_dates=[0], index_col=0, squeeze=True)
ic_2030 = ic_2030.transpose()



###here no Brexit ;-)
ninja = ninja.rename_vars({'GB':'UK'})
ic = ic.rename(columns={'GB':'UK'})
ic_2030 = ic_2030.rename(columns={'GB':'UK'})
ic = ic.to_xarray()
ic_2030 = ic_2030.to_xarray()


#Load data created with the uncomment part below
ninja_season= []
for i in range(0,8):
    filename = 'ninja_season_wr'+str(i)+'.nc'
    temp = xr.open_dataset(data_folder / filename, mask_and_scale=False, decode_times=False)
    ninja_season.append(temp)
mean_season = xr.open_dataset(data_folder / 'ninja_season_mean.nc')

# ninja_season= []
# ##calculate delata CF for each WR
# mean_season = ninja.drop('wr').groupby('time.season').mean()
# for i in range(0, int(ninja.wr.max()+1)):
#     # ninja_tot.append(ninja.drop('wr').where(ninja.wr==i, drop=True).mean())# - ninja.drop('wr').mean())/ninja.drop('wr').mean())
#     ninja_season.append(ninja.drop('wr').where(ninja.wr==i, drop=True).groupby('time.season').mean() \
#                           - mean_season \
#                           #/ ninja.drop('wr').groupby('time.season').mean()
#                           )



#####################END LOAD DATASET#####################################


######################CREATE NEEDED DATASETS#########################

#Create array (Matrix A) for DJF with all WR and countries

#create array for all available IC 
# b = 0
# for i in ninja:
#     for a in ic:
#         if i==a:
#             if b==0:
#                 ic_reduced = xr.DataArray(ic[a][-1])
#                 b = b +1
#             else:
#                 ic_reduced = xr.merge([ic_reduced, ic[a][-1]])

c = 0
for i in ninja:
    for a in ic_2030:
        if i==a:
            if c==0:
                ic_reduced_2030 = xr.DataArray(ic_2030.isel(index=10)[a]*1000)
                ic_reduced = xr.DataArray(ic[a][-1])
                c = c +1
            else:
                ic_reduced_2030 = xr.merge([ic_reduced_2030, ic_2030.isel(index=10)[a]*1000])
                ic_reduced = xr.merge([ic_reduced, ic[a][-1]])



country = []
for i in ic_reduced_2030:
    country.append(i)



A_DJF = np.append([ninja_season[0].sel(season='DJF')[country].to_array().values],[ninja_season[1].sel(season='DJF')[country].to_array().values], axis=0)
A_MAM = np.append([ninja_season[0].sel(season='MAM')[country].to_array().values],[ninja_season[1].sel(season='MAM')[country].to_array().values], axis=0)
A_JJA = np.append([ninja_season[0].sel(season='JJA')[country].to_array().values],[ninja_season[1].sel(season='JJA')[country].to_array().values], axis=0)
A_SON = np.append([ninja_season[0].sel(season='SON')[country].to_array().values],[ninja_season[1].sel(season='SON')[country].to_array().values], axis=0)
for i in range(2,len(ninja_season)):
    A_DJF = np.append(A_DJF,[ninja_season[i].sel(season='DJF')[country].to_array().values], axis=0)
    A_MAM = np.append(A_MAM,[ninja_season[i].sel(season='MAM')[country].to_array().values], axis=0)
    A_JJA = np.append(A_JJA,[ninja_season[i].sel(season='JJA')[country].to_array().values], axis=0)
    A_SON = np.append(A_SON,[ninja_season[i].sel(season='SON')[country].to_array().values], axis=0)

array_tuple = (A_DJF, A_MAM, A_JJA, A_SON)
A_all = np.vstack(array_tuple)



######################END CREATE DATASET#########################


######################DEFINE CONSTRAINTS AND BOUNDS#########################
"""
According to a recent 100% RES scenario of the Energy Watch Group, the EU needs 
to increase its PV capacity from 117 GW to over 630 GW by 2025 and 1.94 TW by 2050 
in order to cover 100% of its electricity needs by renewable energy.

"""                          
#Define lower and upper bound with already installed capacity   
lb = ic_reduced.to_array().values
#ub = ic_reduced_2030.to_array().values * 10
#ub = lb *30
lb_null = ic_reduced.to_array().values *0
lb_ones = ic_reduced.to_array().values/ic_reduced.to_array().values

#ub infinity
ub = np.array(np.ones(lb_null.shape) * np.inf)



#Define vector b with zeros for variability 
b = np.zeros(A_all.shape[0])


#####WITH TOT IC AS CONSTRAINT
#Add tot IC constraint
A_tot_IC = np.append(A_all, [np.ones(A_all.shape[1])], 0)

#IC 2030
target_IC = ic_reduced_2030.to_array().sum()
b_tot_IC = np.append(b, target_IC)      




#####WITH TOT PRODUCTION AS CONSTRAINT
#Define total production
P = (mean_season.mean().to_array() * ic_reduced_2030.to_array()).sum()
b_tot_P = np.append(b, P)
A_tot_P = np.append(A_all, [mean_season.mean()[country].to_array()], 0)


#####WITH TOT PRODUCTION and IC AS CONSTRAINT
#Define total production
b_both = np.append(b, target_IC)
b_both = np.append(b, P)
A_both = np.append(A_all, [np.ones(A_all.shape[1])], 0)
A_both = np.append(A_all, [mean_season.mean()[country].to_array()], 0)

######################END DEFINIDTIONn#########################      


###########calculate least sqaure with matrix A and vector b and evaluate result#########################      
#Calc LSQ
res_tot_IC = lsq_linear(A_tot_IC, b_tot_IC, bounds=(lb,ub))
res_tot_P = lsq_linear(A_tot_P, b_tot_P, bounds=(lb,ub))
res_both = lsq_linear(A_both, b_both, bounds=(lb,ub))

#Comparison
current_state = np.append((A_all * ic_reduced_2030.to_array().values).sum(axis=1), 0 )
print('Planed IC 2030:\n' + str(current_state))
print('With total IC (planed for 2030):\n' + str(res_tot_IC.fun))
print('With total production (planed for 2030):\n' + str(res_tot_P.fun))
print('With total IC and production (planed for 2030):\n' + str(res_both.fun))

data = np.c_[current_state, res_tot_IC.fun, res_tot_P.fun, res_both.fun]


df = pd.DataFrame((data), columns=['Planed IC 2030', 'With total IC (planed for 2030) as constraint', 'With total production (planed for 2030) as constraint', 'With total IC and production (planed for 2030) as constraint'])

for i in range(0,8):
    df = df.rename({i: 'WR'+str(i)}, axis='index')
    df = df.rename({i+8:'WR'+str(i)}, axis='index')
    df = df.rename({i+16:'WR'+str(i)}, axis='index')
    df = df.rename({i+24:'WR'+str(i)}, axis='index')
df = df.rename({32:'Tot IC/P'}, axis='index')


###########PLOT DATA#########################  
#Plot deviation per weather regime and season
ax = df.plot(kind='bar', figsize=(16,9))
#with total planed IC / power production for 2030 as constraint for every season
ax.set_title('IC distribution optimization (with planed IC for 2030 and for every season)', fontsize=20)
ax.set_ylabel('Deviation of PV power production from the seasonal mean in MW')
ax.set_xlabel('Weather regime / Total installed capacity or production')

fig = ax.get_figure()
fig.savefig(data_folder / 'fig/optimize_2030_EU_all.png')





#Plot optimizied IC distribution
#add newly calculated installed capacity into one array
country = []
for i in ic_reduced_2030:
    country.append(i)

data_ic = np.c_[ic_reduced.to_array().values, res_tot_IC.x, res_tot_P.x, res_both.x]
df_ic = pd.DataFrame((data_ic), columns=['Planed IC 2030', 'With total IC (planed for 2030) as constraint', 'With total production (planed for 2030) as constraint', 'With total IC and production (planed for 2030) as constraint'], index=country)
df_ic.to_excel(data_folder / 'ic_new_2030_EU.xlsx')







# #map infos for IC per country
# #Read shapefile using Geopandas
# shapefile = data_folder / 'map/NUTS_RG_01M_2021_4326.shp/NUTS_RG_01M_2021_4326.shp'
# eu = gpd.read_file(shapefile)[['NAME_LATN', 'CNTR_CODE', 'geometry']]
# #Rename columns.
# eu.columns = ['country', 'country_code', 'geometry']


# cf_plotting = []

# for i in range(0, len(df_ic.transpose())):
#     cf_plotting.append(eu.merge(df_ic.iloc[:,i], left_on = 'country_code', right_index=True))
#     temp = eu.merge(ninja_tot[i].to_dataframe().transpose(), left_on = 'country_code', right_index=True)
#     cf_plotting[i] = cf_plotting[i].assign(tot=temp[0])








