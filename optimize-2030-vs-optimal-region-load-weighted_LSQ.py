# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:27:19 2020

@author: Dirk
"""

import numpy as np
from pathlib import Path
# import cartopy.crs as ccrs
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
# import matplotlib.patches as mpatches
# from mpl_toolkits.axes_grid1 import make_axes_locatable

######################LOAD DATASETS#############################################
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





#Load data ninja season dataset
ninja_season= []
for i in range(0,8):
    filename = 'ninja_season_wr'+str(i)+'.nc'
    temp = xr.open_dataset(data_folder / filename, mask_and_scale=False, decode_times=False)
    ninja_season.append(temp)
mean_season = xr.open_dataset(data_folder / 'ninja_season_mean.nc')


#Load load/consumption dataset 
country_load_file = data_folder / 'source/opsd-time_series-2020-10-06/time_series_60min_singleindex.csv'
country_load = pd.read_csv(country_load_file, header=0, parse_dates=[0], index_col=0, squeeze=True)
mask = country_load.columns.str.contains('load_actual_entsoe_transparency')
country_load_year = country_load.loc[:,mask].resample('Y').sum(min_count=365*24)
country_load_year.columns = country_load_year.columns.str.replace('_load_actual_entsoe_transparency', '')

#eurostat for missing countires
eurostat_country_load_file = data_folder / 'source/eurostat_load.xlsx'
eurostat_country_load = pd.read_excel(eurostat_country_load_file, header=0, parse_dates=[0], index_col=0, squeeze=True)
eurostat_country_load = eurostat_country_load.transpose()

###Rename GB with UK --> good idea??
ninja = ninja.rename_vars({'GB':'UK'})
ic = ic.rename(columns={'GB':'UK'})
ic_2030 = ic_2030.rename(columns={'GB':'UK'})
country_load_year = country_load_year.rename(columns={'GB_UKM':'UK'})

#create xarray for IC
ic_xr = ic.to_xarray()
ic_2030_xr = ic_2030.to_xarray()

#Remove unneeded data and add need data from eurostat
country_load_year = country_load_year[country_load_year.columns.drop(list(country_load_year.filter(regex='_')))]


#missing countries in open power data --> #'MT', 'AL','MK', 'BA'#'MD'
country_load_year['MT'] = eurostat_country_load['Malta'].values
country_load_year['AL'] = eurostat_country_load['Albania'].values
country_load_year['MK'] = eurostat_country_load['North Macedonia'].values
country_load_year['BA'] = eurostat_country_load['Bosnia and Herzegovina'].values
country_load_year['MD'] = eurostat_country_load['Moldova'].values

country_load_year = country_load_year.reindex(sorted(country_load_year.columns), axis=1)


last_value = country_load_year.notna()[::-1].idxmax()

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
    for a in ic_xr:
        if i==a:
            if b==0:
                ic_reduced = xr.DataArray(ic_xr[a][-1])
                ic_reduced_2030 = xr.DataArray(ic_2030_xr.sel(index=2030)[a]*1000)
                ic_reduced_pot = xr.DataArray(ic_2030_xr.sel(index='IC pot')[a]*1000)
                b = b +1
            else:
                ic_reduced = xr.merge([ic_reduced, ic_xr[a][-1]])
                ic_reduced_2030 = xr.merge([ic_reduced_2030, ic_2030_xr.sel(index=2030)[a]*1000])
                ic_reduced_pot = xr.merge([ic_reduced_pot, ic_2030_xr.sel(index='IC pot')[a]*1000])


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
project = 'optimization_2030_regions'
                     
#Define lower and upper bound with already installed capacity   
lb = ic_reduced.to_array().values
ub = ic_reduced_pot.to_array().values


#Define vector b with zeros for variability reduction
b = np.zeros(A_all.shape[0])


#####WITH TOT IC AS CONSTRAINT
#split in regions
countries = {}
for i in ic_reduced:
    countries[i]= 0
    


#Define regions
regions = ['east','central','south_east','iberia','british','northern']

east = ['HR','HU', 'SK', 'PL', 'CZ', 'SI', 'RS','MD'] #
central = ['FR', 'DE', 'LU', 'NL','BE', 'DK', 'CH', 'AT']
south_east = ['IT', 'GR', 'CY', 'BG', 'RO', 'ME','MT', 'AL','MK', 'BA'] #
iberia = ['PT', 'ES']
british = ['UK', 'IE']
northern=['NO','SE','FI', 'EE', 'LV', 'LT'] #'IS'

east_load = country_load_year.mean()[east].sum()
central_load = country_load_year.mean()[central].sum()
south_east_load = country_load_year.mean()[south_east].sum() 
iberia_load = country_load_year.mean()[iberia].sum() 
british_load = country_load_year.mean()[british].sum() 
northern_load = country_load_year.mean()[northern].sum() 
tot_load = east_load + central_load + south_east_load + iberia_load + british_load + northern_load


 
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


#####WITH TOT PRODUCTION PER REGIONS and IC AS CONSTRAINT
#Define total production
b_both = np.append(b, target_IC)
b_both = np.append(b_both, P)
A_both = np.append(A_all, [np.ones(A_all.shape[1])], 0)
A_both = np.append(A_both, [mean_season.mean().to_array()], 0)


#ADD RELATIVE LOAD per regions AS CONSTRAINT
#create dataframe to add as poduction constrain to matrix for regions
A_load_regions =  pd.DataFrame(columns=regions, index=np.array(list(countries.keys())))
A_load_regions[:] = 0
A_load_regions['east'].loc[east] = 1
A_load_regions['central'].loc[central] = 1
A_load_regions['south_east'].loc[south_east] = 1
A_load_regions['iberia'].loc[iberia] = 1
A_load_regions['british'].loc[british] = 1
A_load_regions['northern'].loc[northern] = 1

# A_regions = A_both.copy()
# b_regions = b_both.copy()

A_regions = A_all.copy()
b_regions = b.copy()

for i in A_load_regions:
    A_regions = np.append(A_regions, [mean_season.mean().to_array()*A_load_regions[i].to_numpy(dtype='float64')], 0)
    b_regions = np.append(b_regions, P * (globals()[i + '_load']/tot_load)) 

 

######################END DEFINIDTIONn#########################################       


###############CALCULATE LSQ AND PREPARE RESULTS###############################


#add weighting
W_tot_IC = np.ones(b_tot_IC.shape)
W_tot_P = np.ones(b_tot_P.shape)
W_both = np.ones(b_both.shape)
W_both[-2]=1
W_both[-1]=1
W_regions = np.ones(b_regions.shape)
# W_regions[-8]=1
# W_regions[-7]=1
W_regions[-6:] = 10





A_tot_IC = A_tot_IC * np.sqrt(W_tot_IC[:,np.newaxis])
A_tot_P = A_tot_P * np.sqrt(W_tot_P[:,np.newaxis])
A_both = A_both * np.sqrt(W_both[:,np.newaxis])
A_regions = A_regions * np.sqrt(W_regions[:,np.newaxis])

b_tot_IC = b_tot_IC * np.sqrt(W_tot_IC)
b_tot_P = b_tot_P * np.sqrt(W_tot_P)
b_both = b_both * np.sqrt(W_both)
b_regions = b_regions * np.sqrt(W_regions)

# W = np.array([1,2,3,4,5])
# Aw = A * np.sqrt(W[:,np.newaxis])
# Bw = B * np.sqrt(W)
# X = np.linalg.lstsq(Aw, Bw)


#Calc LSQ
res_tot_IC = lsq_linear(A_tot_IC, b_tot_IC, bounds=(lb,ub))
res_tot_P = lsq_linear(A_tot_P, b_tot_P, bounds=(lb,ub))
res_both = lsq_linear(A_both, b_both, bounds=(lb,ub))
res_regions = lsq_linear(A_regions, b_regions, bounds=(lb,ub))


#Comparison

#muesch vilicht für alli mache
current_state = (A_all * ic_reduced_2030.to_array().values).sum(axis=1)
var_tot_IC = (A_all * res_tot_IC.x).sum(axis=1)
var_tot_P = (A_all * res_tot_P.x).sum(axis=1)
var_both = (A_all * res_both.x).sum(axis=1)
var_regions = (A_all * res_regions.x).sum(axis=1)


# print('Planed IC 2030:\n' + str(current_state))
# print('With total IC (planed for 2030):\n' + str(res_tot_IC.fun))
# print('With total production (planed for 2030):\n' + str(res_tot_P.fun))
# print('With total IC and production (planed for 2030):\n' + str(res_both.fun))

# data_var = np.c_[current_state,var_tot_IC, var_tot_P,var_both]
data_var = np.c_[current_state,var_regions]

# df_var = pd.DataFrame((data_var), columns=['Planed IC 2030', 'With total IC (planed for 2030) as constraint', 'With total production (planed for 2030) as constraint', 'With total IC and production (planed for 2030) as constraint'])
df_var = pd.DataFrame((data_var), columns=['Planed IC 2030', 'With loads per region as constraint'])
for i in range(0,8):
    df_var = df_var.rename({i: 'WR'+str(i)}, axis='index')
    df_var = df_var.rename({i+8:'WR'+str(i)}, axis='index')
    df_var = df_var.rename({i+16:'WR'+str(i)}, axis='index')
    df_var = df_var.rename({i+24:'WR'+str(i)}, axis='index')


#Calc Variance per season

var_winter = []
var_spring = []
var_summer = []
var_autumn = []
for i in range(0,8):
    var_winter.extend(abs(df_var[:].to_numpy()[i+1:8] - df_var[:].to_numpy()[i]))
    var_spring.extend(abs(df_var[:].to_numpy()[i+9:16] - df_var[:].to_numpy()[i+8]))
    var_summer.extend(abs(df_var[:].to_numpy()[i+17:24] - df_var[:].to_numpy()[i+16]))
    var_autumn.extend(abs(df_var[:].to_numpy()[i+25:32] - df_var[:].to_numpy()[i+24]))
    
tot_var_winter = sum(var_winter)
tot_var_spring = sum(var_spring)
tot_var_summer = sum(var_summer)
tot_var_autumn = sum(var_autumn)
tot_var = tot_var_winter + tot_var_spring + tot_var_summer + tot_var_autumn

df_tot_var = pd.DataFrame([tot_var_winter, tot_var_spring, tot_var_summer, tot_var_autumn, tot_var], columns=df_var.columns, index=['winter', 'spring', 'summer', 'autumn', 'total'])
ax_tot_var = df_tot_var.plot(kind='bar', figsize=(16,12))
fig_tot_var = ax_tot_var.get_figure()
fig_tot_var.savefig(data_folder / str('fig/' + project +'_tot_variability.png'))
    
#Total production
tot_P = []
tot_P.append((mean_season.mean().to_array() * ic_reduced_2030.to_array()).sum())
# tot_P.append((mean_season.mean().to_array() * res_tot_IC.x).sum())
# tot_P.append((mean_season.mean().to_array() * res_tot_P.x).sum())
# tot_P.append((mean_season.mean().to_array() * res_both.x).sum())
tot_P.append((mean_season.mean().to_array() * res_regions.x).sum())

#Total installed capacity
tot_IC = []
tot_IC.append(ic_reduced_2030.to_array().values.sum())
# tot_IC.append(res_tot_IC.x.sum())
# tot_IC.append(res_tot_P.x.sum())
# tot_IC.append(res_both.x.sum())
tot_IC.append(res_regions.x.sum())

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

# data_ic = np.c_[ic_reduced_2030.to_array().values, res_tot_IC.x, res_tot_P.x, res_both.x]
data_ic = np.c_[ic_reduced_2030.to_array().values, res_regions.x]
df_ic = pd.DataFrame((data_ic), columns=['Planed IC 2030', 'With loads per region as constraint'], index=country)
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
    ncols=2,
    nrows=2,
    figsize=(20, 18),
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
leg1 = ax[1,0].get_figure().get_axes()[4]
leg2 = ax[1,0].get_figure().get_axes()[5]
leg1.set_position([0.37,0.45,0.3,0.1])
leg2.set_position([0.37,0.05,0.3,0.1])

#move subplot
pos2 = [ax[1,0].get_position().x0, ax[0,1].get_position().y0, ax[0,1].get_position().width, ax[0,1].get_position().height]
ax[0,0].set_position(pos2)
pos4 = [ax[0,0].get_position().x0, ax[1,1].get_position().y0, ax[1,1].get_position().width, ax[1,1].get_position().height]
ax[1,0].set_position(pos4)
    
f.suptitle('IC distribution', fontsize=30)
f.savefig(data_folder / str('fig/' + project + '_ic-distribution.png'))

