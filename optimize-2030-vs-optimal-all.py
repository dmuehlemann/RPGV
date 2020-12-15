# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:59:12 2020

@author: Dirk
"""

import numpy as np
from pathlib import Path
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
# import calendar
import pandas as pd
# import matplotlib as mpl
# import geopandas as gpd
# from mapclassify import Quantiles, UserDefined
from scipy.optimize import lsq_linear
import matplotlib as mpl
import geopandas as gpd
import yaml as yaml
import pycountry

######################LOAD DATASET#################


data_folder = Path("../data/")


file_ninja = data_folder / 'ninja_and_wr_30days_lowpass_2_0-1_short10.nc'
ninja = xr.open_dataset(file_ninja)

ic_file = data_folder / 'source/installed_capacities_IRENA.csv'
ic_2030 = data_folder / 'source/planned-IC-2030.xlsx'


ic = pd.read_csv(ic_file, header=0, parse_dates=[0], index_col=0, squeeze=True)
ic_2030 = pd.read_excel(ic_2030, header=0, parse_dates=[0], index_col=0, squeeze=True)
ic_2030 = ic_2030.transpose()



with open(data_folder / 'source/IC-potential.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    #ic_pot = yaml.load(file, Loader=yaml.FullLoader)
    ic_pot_file = yaml.safe_load(file)



#Load and edit pot country
countries = {}
for country in pycountry.countries:
    countries[country.alpha_3] = country.alpha_2
alpha2 = [countries.get(country, np.nan) for country in ic_pot_file['locations']]
ic_pot = {}
for i in ic_pot_file['locations']:
    ic_pot[countries[i]] = ic_pot_file['locations'][i]['techs']['roof_mounted_pv']['constraints']['energy_cap_max']*100 #yaml in 100,000 MW --> *100 to convert in GW
    
ic_pot_df = pd.DataFrame([[ic_pot.get(country, np.nan) for country in ic_2030]], columns=ic_2030.columns, index=['IC pot'])
ic_2030 = ic_2030.append(ic_pot_df)


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
A_DJF = np.append([ninja_season[0].sel(season='DJF').to_array().values],[ninja_season[1].sel(season='DJF').to_array().values], axis=0)
A_MAM = np.append([ninja_season[0].sel(season='MAM').to_array().values],[ninja_season[1].sel(season='MAM').to_array().values], axis=0)
A_JJA = np.append([ninja_season[0].sel(season='JJA').to_array().values],[ninja_season[1].sel(season='JJA').to_array().values], axis=0)
A_SON = np.append([ninja_season[0].sel(season='SON').to_array().values],[ninja_season[1].sel(season='SON').to_array().values], axis=0)
for i in range(2,len(ninja_season)):
    A_DJF = np.append(A_DJF,[ninja_season[i].sel(season='DJF').to_array().values], axis=0)
    A_MAM = np.append(A_MAM,[ninja_season[i].sel(season='MAM').to_array().values], axis=0)
    A_JJA = np.append(A_JJA,[ninja_season[i].sel(season='JJA').to_array().values], axis=0)
    A_SON = np.append(A_SON,[ninja_season[i].sel(season='SON').to_array().values], axis=0)

array_tuple = (A_DJF, A_MAM, A_JJA, A_SON)
A_all = np.vstack(array_tuple)

#create array for all available IC 
b = 0
for i in ninja:
    for a in ic:
        if i==a:
            if b==0:
                ic_reduced = xr.DataArray(ic[a][-1])
                ic_reduced_2030 = xr.DataArray(ic_2030.sel(index=2030)[a]*1000)
                ic_reduced_pot = xr.DataArray(ic_2030.sel(index='IC pot')[a]*1000)
                b = b +1
            else:
                ic_reduced = xr.merge([ic_reduced, ic[a][-1]])
                ic_reduced_2030 = xr.merge([ic_reduced_2030, ic_2030.sel(index=2030)[a]*1000])
                ic_reduced_pot = xr.merge([ic_reduced_pot, ic_2030.sel(index='IC pot')[a]*1000])


#ic_reduced_pot --> add IC 2030 * 5 if na
for i in ic_reduced_pot:
    if np.isnan(ic_reduced_pot[i]):
        ic_reduced_pot[i] = ic_reduced_2030[i]*5


######################END CREATE DATASET#########################


######################DEFINE CONSTRAINTS AND BOUNDS#########################
"""
According to a recent 100% RES scenario of the Energy Watch Group, the EU needs 
to increase its PV capacity from 117 GW to over 630 GW by 2025 and 1.94 TW by 2050 
in order to cover 100% of its electricity needs by renewable energy.

"""                          
#Define lower and upper bound with already installed capacity   
lb = ic_reduced.to_array().values
ub = ic_reduced_pot.to_array().values
#ub = lb *30
lb_null = ic_reduced.to_array().values *0
lb_ones = ic_reduced.to_array().values/ic_reduced.to_array().values


#upper bound infinitiy
# ub = np.array(np.ones(lb_null.shape) * np.inf)



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
A_tot_P = np.append(A_all, [mean_season.mean().to_array()], 0)


#####WITH TOT PRODUCTION and IC AS CONSTRAINT
#Define total production
b_both = np.append(b, target_IC)
b_both = np.append(b_both, P)
A_both = np.append(A_all, [np.ones(A_all.shape[1])], 0)
A_both = np.append(A_both, [mean_season.mean().to_array()], 0)

######################END DEFINIDTIONn#########################      


###########calculate least sqaure with matrix A and vector b and evaluate result#########################      
#Calc LSQ
res_tot_IC = lsq_linear(A_tot_IC, b_tot_IC, bounds=(lb,ub))
res_tot_P = lsq_linear(A_tot_P, b_tot_P, bounds=(lb,ub))
res_both = lsq_linear(A_both, b_both, bounds=(lb,ub))

#Comparison
current_state = (A_all * ic_reduced_2030.to_array().values).sum(axis=1)
print('Planed IC 2030:\n' + str(current_state))
print('With total IC (planed for 2030):\n' + str(res_tot_IC.fun))
print('With total production (planed for 2030):\n' + str(res_tot_P.fun))
print('With total IC and production (planed for 2030):\n' + str(res_both.fun))

data = np.c_[current_state, res_tot_IC.fun[:-1], res_tot_P.fun[:-1], res_both.fun[:-2]]


df = pd.DataFrame((data), columns=['Planed IC 2030', 'With total IC (planed for 2030) as constraint', 'With total production (planed for 2030) as constraint', 'With total IC and production (planed for 2030) as constraint'])

for i in range(0,8):
    df = df.rename({i: 'WR'+str(i)}, axis='index')
    df = df.rename({i+8:'WR'+str(i)}, axis='index')
    df = df.rename({i+16:'WR'+str(i)}, axis='index')
    df = df.rename({i+24:'WR'+str(i)}, axis='index')
# df = df.rename({32:'Tot IC/P'}, axis='index')


###########PLOT DATA#########################  
#Plot deviation per weather regime and season
ax = df.plot(kind='bar', figsize=(16,9))
#with total planed IC / power production for 2030 as constraint for every season
ax.set_title('IC distribution optimization (with planed IC for 2030 and for every season)', fontsize=20)
ax.set_ylabel('Deviation of PV power production from the seasonal mean in MW')
ax.set_xlabel('Weather regime / Total installed capacity or production')

fig = ax.get_figure()
fig.savefig(data_folder / 'fig/optimize_2030_all_lbpot.png')


#Total production
tot_P = []
tot_P.append((mean_season.mean().to_array() * ic_reduced_2030.to_array()).sum())
tot_P.append((mean_season.mean().to_array() * res_tot_IC.x).sum())
tot_P.append((mean_season.mean().to_array() * res_tot_P.x).sum())
tot_P.append((mean_season.mean().to_array() * res_both.x).sum())


#Total installed capacity
tot_IC = []
tot_IC.append(ic_reduced_2030.to_array().values.sum())
tot_IC.append(res_tot_IC.x.sum())
tot_IC.append(res_tot_P.x.sum())
tot_IC.append(res_both.x.sum())


#Plot optimizied IC distribution
#add newly calculated installed capacity into one array
country = []
for i in ic_reduced_2030:
    country.append(i)

data_ic = np.c_[ic_reduced_2030.to_array().values, res_tot_IC.x, res_tot_P.x, res_both.x]
df_ic = pd.DataFrame((data_ic), columns=['Planed IC 2030', 'With total IC (planed for 2030) as constraint', 'With total production (planed for 2030) as constraint', 'With total IC and production (planed for 2030) as constraint'], index=country)
df_ic.to_excel(data_folder / 'ic_new_2030_lbpot.xlsx')

#ic minus lower bound
df_ic_lb = (df_ic.transpose() -lb).transpose()





#map infos for IC per country
#Read shapefile using Geopandas
shapefile_old = data_folder / 'map/NUTS_RG_01M_2021_4326.shp/NUTS_RG_01M_2021_4326.shp'

#shapefile = data_folder / 'map/NUTS_RG_01M_2021_4326_LEVL_0/NUTS_RG_01M_2021_4326_LEVL_0.shp'
shapefile = data_folder / 'map/CNTR_RG_01M_2020_4326/CNTR_RG_01M_2020_4326.shp'


# eu = gpd.read_file(shapefile)[['NAME_LATN', 'CNTR_CODE', 'geometry']]
eu = gpd.read_file(shapefile)[['CNTR_NAME', 'CNTR_ID', 'geometry']]
#Rename columns.
eu.columns = ['country', 'country_code', 'geometry']


#Rename Greece EL --> GR
eu.country_code[60] = 'GR'
#eu.country_code[9] = 'GR'


ic_plotting = []
ic_lb_plotting = []
for i in range(0, len(df_ic.transpose())):
    ic_plotting.append(eu.merge(df_ic.iloc[:,i], left_on = 'country_code', right_index=True))
    ic_lb_plotting.append(eu.merge(df_ic_lb.iloc[:,i], left_on = 'country_code', right_index=True))
    # temp = eu.merge(ninja_tot[i].to_dataframe().transpose(), left_on = 'country_code', right_index=True)
    # cf_plotting[i] = cf_plotting[i].assign(tot=temp[0])

plt.close("all")
f, ax = plt.subplots(
    ncols=4,
    nrows=2,
    
    figsize=(30, 20),
)
cmap = mpl.cm.get_cmap("Reds")
vmin = 0
vmax = 100000
a = 'Planed IC 2030'
c=0
for i in ic_plotting:
    i.dropna().plot(ax=ax[0,c], column=i.columns[3], cmap=cmap,
                              vmax=vmax, vmin=vmin,
                              legend=True, 
                              legend_kwds={'label': "Installed capacity (in MW)",
                              'orientation': "horizontal",}
                                                              
                              )
    
    ax[0,c].set_xlim(left=-13, right=35)
    ax[0,c].set_ylim(bottom=35, top=70) 
    ax[0,c].set_title(i.columns[3] + '\n Total IC (GW): ' + str(tot_IC[c].round(0)/1000) + '\n Total Production (GW): ' + str(tot_P[c].values.round(0)/1000), fontsize=15)
    ax[0,c].set_axis_off()
    c = c +1
    
c=0
for i in ic_lb_plotting:
    i.dropna().plot(ax=ax[1,c], column=i.columns[3], cmap=cmap,
                              vmax=vmax, vmin=vmin,
                              legend=True, 
                              legend_kwds={'label': "Installed capacity (in MW)",
                              'orientation': "horizontal",}
                                                              
                              )
    
    ax[1,c].set_xlim(left=-13, right=35)
    ax[1,c].set_ylim(bottom=35, top=70) 
    ax[1,c].set_title(i.columns[3] + '\n Total IC (GW): ' + str(tot_IC[c].round(0)/1000) + '\n Total Production (GW): ' + str(tot_P[c].values.round(0)/1000), fontsize=15)
    ax[1,c].set_axis_off()
    c = c +1
    
f.savefig(data_folder / 'fig/new_ic_destribution.png')

