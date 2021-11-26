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
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

######################LOAD DATASET#############################################
data_folder = Path("../data/")
file_ninja = data_folder / 'ninja_and_wr_30days_lowpass_2_0-1_short10.nc'
ninja = xr.open_dataset(file_ninja)

ic_file = data_folder / 'source/installed_capacities_IRENA.csv'
ic_2030 = data_folder / 'source/planned-IC-2030.xlsx'
ic = pd.read_csv(ic_file, header=0, parse_dates=[0], index_col=0, squeeze=True)
ic_2030 = pd.read_excel(ic_2030, header=0, parse_dates=[0], index_col=0, squeeze=True)
ic_2030 = ic_2030.transpose()

#Load and prepare potential IC (roof mounted)
with open(data_folder / 'source/IC-potential.yaml') as file:
    ic_pot_file = yaml.safe_load(file)
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

#####################END LOAD DATASET##########################################


#####################CREATE NEEDED DATASETS FOR LSQ############################

#Create array (Matrix A) for all seasons with all WR and countries
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

#create array for different IC configuration
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


######################END CREATE DATASET#######################################   


######################DEFINE CONSTRAINTS AND BOUNDS############################   
"""
According to a recent 100% RES scenario of the Energy Watch Group, the EU needs 
to increase its PV capacity from 117 GW to over 630 GW by 2025 and 1.94 TW by 2050 
in order to cover 100% of its electricity needs by renewable energy.

"""     

#Define project name which is used for filesaving
project = 'optimization_2030_minimize-cost'
                     
#Define lower and upper bound with already installed capacity   
lb = ic_reduced.to_array().values
ub = ic_reduced_pot.to_array().values


#Define vector b with zeros for variability reduction
b = np.zeros(A_all.shape[0])


#####WITH TOT IC AS CONSTRAINT
#Add tot IC constraint
A_tot_IC = np.append(A_all, [np.ones(A_all.shape[1])], 0)

#IC 2030
target_IC = ic_reduced_2030.to_array().sum()
b_tot_IC = np.append(b, target_IC)      


#####WITH TOT PRODUCTION AS CONSTRAINT
#Define total production
P = (mean_season.mean().to_array() * ic_reduced_2030.to_array()).sum()*10
b_tot_P = np.append(b, P)
A_tot_P = np.append(A_all, [mean_season.mean().to_array()], 0)


#####WITH TOT PRODUCTION and IC AS CONSTRAINT
#Define total production
b_both = np.append(b, target_IC)
b_both = np.append(b_both, P)
A_both = np.append(A_all, [np.ones(A_all.shape[1])], 0)
A_both = np.append(A_both, [mean_season.mean().to_array()], 0)

######################END DEFINIDTIONn#########################################       


###############CALCULATE LSQ AND PREPARE RESULTS###############################
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

data_var = np.c_[current_state, res_tot_IC.fun[:-1], res_tot_P.fun[:-1], res_both.fun[:-2]]

df_var = pd.DataFrame((data_var), columns=['Planed IC 2030', 'With total IC (planed for 2030) as constraint', 'With total production (planed for 2030) as constraint', 'With total IC and production (planed for 2030) as constraint'])

for i in range(0,8):
    df_var = df_var.rename({i: 'WR'+str(i)}, axis='index')
    df_var = df_var.rename({i+8:'WR'+str(i)}, axis='index')
    df_var = df_var.rename({i+16:'WR'+str(i)}, axis='index')
    df_var = df_var.rename({i+24:'WR'+str(i)}, axis='index')



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


###############END CALCULATE LSQ AND PREPARE RESULTS###########################


##########################PLOT RESULTS#########################################

#Plot deviation per weather regime and season
ax_var = df_var.plot(kind='bar', figsize=(16,9))
#with total planed IC / power production for 2030 as constraint for every season
ax_var.set_title('Optimized IC distribution (with planed IC for 2030)', fontsize=20, pad=20)
ax_var.set_ylabel('Deviation of PV power production from the seasonal mean in MW', fontsize=15)
ax_var.set_xlabel('Weather regime', fontsize=15, labelpad=10)
ax_var.axvspan(0,7.5, facecolor='w', alpha=0.2, label='Winter')
ax_var.text(3, ax_var.yaxis.get_data_interval().min(), 'Winter', fontsize=15)
ax_var.axvspan(7.5,15.5, facecolor='g', alpha=0.2, label='Spring')
ax_var.text(11, ax_var.yaxis.get_data_interval().min(), 'Spring', fontsize=15)
ax_var.axvspan(15.5,23.5, facecolor='r', alpha=0.2, label='Summer')
ax_var.text(19, ax_var.yaxis.get_data_interval().min(), 'Summer', fontsize=15)
ax_var.axvspan(23.5,32, facecolor='b', alpha=0.2, label='Autumn')
ax_var.text(27, ax_var.yaxis.get_data_interval().min(), 'Autumn', fontsize=15)
fig = ax_var.get_figure()
fig.savefig(data_folder / str('fig/' + project +'_variability.png'))


#Plot optimizied IC distribution


#add newly calculated installed capacity into one dataframe
country = []
for i in ic_reduced_2030:
    country.append(i)

data_ic = np.c_[ic_reduced_2030.to_array().values, res_tot_IC.x, res_tot_P.x, res_both.x]
df_ic = pd.DataFrame((data_ic), columns=['Planed IC 2030', 'With total IC (planed for 2030) as constraint', 'With total production (planed for 2030) as constraint', 'With total IC and production (planed for 2030) as constraint'], index=country)
df_ic.to_excel(data_folder / str(project + 'new-ic.xlsx'))
#ic minus lower bound
df_ic_lb = round((df_ic.transpose() -lb).transpose())


#Read shapefile using Geopandas
#shapefile_old = data_folder / 'map/NUTS_RG_01M_2021_4326.shp/NUTS_RG_01M_2021_4326.shp'
#shapefile = data_folder / 'map/NUTS_RG_01M_2021_4326_LEVL_0/NUTS_RG_01M_2021_4326_LEVL_0.shp'
shapefile = data_folder / 'map/CNTR_RG_01M_2020_4326/CNTR_RG_01M_2020_4326.shp'
# eu = gpd.read_file(shapefile)[['NAME_LATN', 'CNTR_CODE', 'geometry']]
eu = gpd.read_file(shapefile)[['CNTR_NAME', 'CNTR_ID', 'geometry']]
#Rename columns.
eu.columns = ['country', 'country_code', 'geometry']
#Rename Greece EL --> GR
eu.country_code[60] = 'GR'
#eu.country_code[9] = 'GR'

#Merge results into geopandas
ic_plotting = []
ic_lb_plotting = []
for i in range(0, len(df_ic.transpose())):
    ic_plotting.append(eu.merge(df_ic.iloc[:,i], left_on = 'country_code', right_index=True))
    ic_lb_plotting.append(eu.merge(df_ic_lb.iloc[:,i], left_on = 'country_code', right_index=True))


f, ax = plt.subplots(
    ncols=4,
    nrows=2,
    figsize=(30, 15),
)

cmap = mpl.cm.get_cmap("Reds")
cmap.set_under(color='black')
vmin = 0.001
vmax = 100000

#Plot absolut data
c=0
for i in ic_plotting:
    if c == 0:
        # divider = make_axes_locatable(ax[0,c])
        # cax = divider.append_axes("bottom", size="5%", pad=0.4)
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
    else:
        i.dropna().plot(ax=ax[0,c], column=i.columns[3], cmap=cmap,
                                  vmax=vmax, vmin=vmin,
                                  )
        
        ax[0,c].set_xlim(left=-13, right=35)
        ax[0,c].set_ylim(bottom=35, top=70) 
        ax[0,c].set_title(i.columns[3] + '\n Total IC (GW): ' + str(tot_IC[c].round(0)/1000) + '\n Total Production (GW): ' + str(tot_P[c].values.round(0)/1000), fontsize=15)
        ax[0,c].set_axis_off()
        c = c +1

#Plot data minus lower bound
c=0
for i in ic_lb_plotting:
    if c == 0:
        # divider = make_axes_locatable(ax[1,c])
        # cax = divider.append_axes("bottom", size="5%", pad=0.4)
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
    else:
        i.dropna().plot(ax=ax[1,c], column=i.columns[3], cmap=cmap,
                                  vmax=vmax, vmin=vmin,
                                  )
        
        ax[1,c].set_xlim(left=-13, right=35)
        ax[1,c].set_ylim(bottom=35, top=70) 
        ax[1,c].set_title(i.columns[3] + '\n Total IC (GW): ' + str(tot_IC[c].round(0)/1000) + '\n Total Production (GW): ' + str(tot_P[c].values.round(0)/1000), fontsize=15)
        ax[1,c].set_axis_off()
        c = c +1


# Move legend to rigt place
leg1 = ax[1,0].get_figure().get_axes()[8]
leg2 = ax[1,0].get_figure().get_axes()[9]
leg1.set_position([0.37,0.45,0.3,0.1])
leg2.set_position([0.37,0.05,0.3,0.1])

#move subplot
pos2 = [ax[1,0].get_position().x0, ax[0,1].get_position().y0, ax[0,1].get_position().width, ax[0,1].get_position().height]
ax[0,0].set_position(pos2)
pos4 = [ax[0,0].get_position().x0, ax[1,1].get_position().y0, ax[1,1].get_position().width, ax[1,1].get_position().height]
ax[1,0].set_position(pos4)
    
f.suptitle('IC distribution', fontsize=30)
f.savefig(data_folder / str('fig/' + project + '_ic-distribution.png'))

