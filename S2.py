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


mpl.rc('font', serif='times new roman')
plt.rcParams["font.family"] = "Times New Roman"
######################LOAD DATASETS#############################################
data_folder = Path("../data/")
file_ninja = data_folder / 'ninja_and_wr_30days_lowpass_2_0-1_short3.nc'
ninja = xr.open_dataset(file_ninja)

file_wr = data_folder / 'wr_time-c7_std_30days_lowpass_2_0-1_short3.nc'
wr = xr.open_dataset(file_wr)

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
project = 'Scenario_1'
                     
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
P = (mean_season.mean().to_array() * ic_reduced_2030.to_array()).sum()
b_tot_P = np.append(b, P)
A_tot_P = np.append(A_all, [mean_season.mean().to_array()], 0)


#####WITH TOT PRODUCTION PER REGIONS and IC AS CONSTRAINT
#Define total production
b_S1 = np.append(b, target_IC)
b_S1 = np.append(b_S1, P)
A_S1 = np.append(A_all, [np.ones(A_all.shape[1])], 0)
A_S1 = np.append(A_S1, [mean_season.mean().to_array()], 0)


######################END DEFINIDTIONn#########################################       


###############CALCULATE LSQ AND PREPARE RESULTS###############################


#add weighting
#Variability
W_S1 = np.ones(b_S1.shape)

#Installed capacity
W_S1[-2]=0.01

#Production
W_S1[-1]=1


#add weightning to coefficent matrix A and target vector b
A_S1 = A_S1 * np.sqrt(W_S1[:,np.newaxis])
b_S1 = b_S1 * np.sqrt(W_S1)


#Calc LSQ
res_S1 = lsq_linear(A_S1, b_S1, bounds=(lb,ub))

#Comparison

#calculate variability for each wr season and specific IC
current_state= (A_all * ic_reduced.to_array().values).sum(axis=1)
state_2030 = (A_all * ic_reduced_2030.to_array().values).sum(axis=1)
var_S1 = (A_all * res_S1.x).sum(axis=1)

# data_var = np.c_[current_state,var_tot_IC, var_tot_P,var_S1]
data_var = np.c_[current_state, state_2030,var_S1]

# df_var = pd.DataFrame((data_var), columns=['Planned IC 2030', 'With total IC (planned for 2030) as constraint', 'With total production (Planned for 2030) as constraint', 'With total IC and production (Planned for 2030) as constraint'])
df_var = pd.DataFrame((data_var), columns=['Variability with installed PV capacity 2019', 'Variability with installed PV capacity planned for 2030', 'Variability with installed PV capacity and production (2030) as constraint (S1)'])
for i in range(0,8):
    df_var = df_var.rename({i: 'WR'+str(i)}, axis='index')
    df_var = df_var.rename({i+8:'WR'+str(i)}, axis='index')
    df_var = df_var.rename({i+16:'WR'+str(i)}, axis='index')
    df_var = df_var.rename({i+24:'WR'+str(i)}, axis='index')
df_var = df_var.rename({'WR7':'no regime'})



#calculate frequencies per wr
frequency = (wr.where(wr.wr == 0).groupby('time.season').count() / wr.groupby('time.season').count()).to_dataframe().rename(columns={'wr':'WR0'})
for i in range(1,wr.wr.max().values+1):
    col = str('WR')+str(i)
    frequency[col] = (wr.where(wr.wr == i).groupby('time.season').count() / wr.groupby('time.season').count()).wr

tot_frequency = (wr.where(wr.wr == 0).count() / wr.count()).expand_dims(dim='tot').to_dataframe().rename(columns={'wr':'WR0'})
for i in range(1,wr.wr.max().values+1):
    col = str('WR')+str(i)
    tot_frequency[col] = (wr.where(wr.wr == i).count() / wr.count()).expand_dims(dim='tot').wr

frequency = frequency.append(tot_frequency).rename({0: 'tot'})
frequency = frequency.rename(columns={'WR7':'no regime'})


#calcualte frequency times #days of wr per season --> for toal variability
temp_fre_DJF = np.transpose(np.array([frequency.loc['DJF']]*3))
temp_fre_MAM = np.transpose(np.array([frequency.loc['MAM']]*3))
temp_fre_JJA = np.transpose(np.array([frequency.loc['JJA']]*3))
temp_fre_SON = np.transpose(np.array([frequency.loc['SON']]*3))
temp_fre = np.concatenate((temp_fre_DJF, temp_fre_MAM, temp_fre_JJA, temp_fre_SON))
data_var_fre = data_var * temp_fre


# df_var_frequency = pd.DataFrame((data_var_fre), columns=['Variability with installed PV capacity 2019', 'Variability with installed PV capacity planned for 2030', 'Variability with installed PV capacity and production (2030) as constraint (S1)'])
# for i in range(0,8):
#     df_var_frequency = df_var_frequency.rename({i: 'WR'+str(i)}, axis='index')
#     df_var_frequency = df_var_frequency.rename({i+8:'WR'+str(i)}, axis='index')
#     df_var_frequency = df_var_frequency.rename({i+16:'WR'+str(i)}, axis='index')
#     df_var_frequency = df_var_frequency.rename({i+24:'WR'+str(i)}, axis='index')
# df_var_frequency = df_var_frequency.rename({'WR7':'no regime'})

var_winter = abs(df_var[:].to_numpy()[0+1:8] - df_var[:].to_numpy()[0])
var_spring = abs(df_var[:].to_numpy()[0+9:16] - df_var[:].to_numpy()[0+8])
var_summer = abs(df_var[:].to_numpy()[0+17:24] - df_var[:].to_numpy()[0+16])
var_autumn = abs(df_var[:].to_numpy()[0+25:32] - df_var[:].to_numpy()[0+24])
for i in range(1,8):
    var_winter= np.append(var_winter, abs(df_var[:].to_numpy()[i+1:8] - df_var[:].to_numpy()[i]), axis=0)
    var_spring = np.append(var_spring, abs(df_var[:].to_numpy()[i+9:16] - df_var[:].to_numpy()[i+8]), axis=0)
    var_summer = np.append(var_summer, abs(df_var[:].to_numpy()[i+17:24] - df_var[:].to_numpy()[i+16]), axis=0)
    var_autumn = np.append(var_autumn, abs(df_var[:].to_numpy()[i+25:32] - df_var[:].to_numpy()[i+24]), axis=0)

var_all = np.vstack((var_winter, var_spring, var_summer, var_autumn))



var_winter_fre = abs((df_var[:].to_numpy()[0+1:8] - df_var[:].to_numpy()[0]) * temp_fre_DJF[0] * temp_fre_DJF[0+1:8])
var_spring_fre = abs((df_var[:].to_numpy()[0+9:16] - df_var[:].to_numpy()[0+8]) * temp_fre_MAM[0] * temp_fre_MAM[0+1:8])
var_summer_fre = abs((df_var[:].to_numpy()[0+17:24] - df_var[:].to_numpy()[0+16]) * temp_fre_JJA[0] * temp_fre_JJA[0+1:8])
var_autumn_fre = abs((df_var[:].to_numpy()[0+25:32] - df_var[:].to_numpy()[0+24]) * temp_fre_SON[0] * temp_fre_SON[0+1:8])
for i in range(1,8):
    var_winter_fre= np.append(var_winter_fre, abs((df_var[:].to_numpy()[i+1:8] - df_var[:].to_numpy()[i])* temp_fre_DJF[i] * temp_fre_DJF[i+1:8]), axis=0)
    var_spring_fre = np.append(var_spring_fre, abs((df_var[:].to_numpy()[i+9:16] - df_var[:].to_numpy()[i+8])* temp_fre_MAM[i] * temp_fre_MAM[i+1:8]), axis=0)
    var_summer_fre = np.append(var_summer_fre, abs((df_var[:].to_numpy()[i+17:24] - df_var[:].to_numpy()[i+16])* temp_fre_JJA[i] * temp_fre_JJA[i+1:8]), axis=0)
    var_autumn_fre = np.append(var_autumn_fre, abs((df_var[:].to_numpy()[i+25:32] - df_var[:].to_numpy()[i+24])* temp_fre_SON[i] * temp_fre_SON[i+1:8]), axis=0)

var_all_fre = np.vstack((var_winter_fre, var_spring_fre, var_summer_fre, var_autumn_fre))


# #max variaility --> difference between max ele. prod. WR and min ele. prod. WR
# var_winter_max = abs(df_var_frequency[:].to_numpy()[0:8].max(axis=0) - df_var_frequency[:].to_numpy()[0:8].min(axis=0))
# var_spring_max = abs(df_var_frequency[:].to_numpy()[8:16].max(axis=0) - df_var_frequency[:].to_numpy()[8:16].min(axis=0))
# var_summer_max = abs(df_var_frequency[:].to_numpy()[16:24].max(axis=0) - df_var_frequency[:].to_numpy()[16:24].min(axis=0))
# var_autumn_max = abs(df_var_frequency[:].to_numpy()[24:32].max(axis=0) - df_var_frequency[:].to_numpy()[24:32].min(axis=0))


   
#calc total variability
tot_var_winter = sum(var_winter_fre)#/len(var_winter)
tot_var_spring = sum(var_spring_fre)#/len(var_spring)
tot_var_summer = sum(var_summer_fre)#/len(var_summer)
tot_var_autumn = sum(var_autumn_fre)#/len(var_autumn)
days_seasons = wr.groupby('time.season').count()
tot_var = ((days_seasons.sel(season='DJF').to_array()*tot_var_winter + days_seasons.sel(season='MAM').to_array()*tot_var_spring + days_seasons.sel(season='JJA').to_array()*tot_var_summer + days_seasons.sel(season='SON').to_array()*tot_var_autumn)/    days_seasons.sum().to_array().values).values
tot_var_std = np.array([[var_winter.std(axis=0)[0], var_spring.std(axis=0)[0], var_summer.std(axis=0)[0], var_autumn.std(axis=0)[0], var_all.std(axis=0)[0]],
                        [var_winter.std(axis=0)[1], var_spring.std(axis=0)[1] ,var_summer.std(axis=0)[1], var_autumn.std(axis=0)[1], var_all.std(axis=0)[1]], 
                        [var_winter.std(axis=0)[2], var_spring.std(axis=0)[2] ,var_summer.std(axis=0)[2], var_autumn.std(axis=0)[2], var_all.std(axis=0)[2]]])




#Total production
tot_P = []
tot_P.append((mean_season.mean().to_array() * ic_reduced.to_array()).sum())
tot_P.append((mean_season.mean().to_array() * ic_reduced_2030.to_array()).sum())
# tot_P.append((mean_season.mean().to_array() * res_tot_IC.x).sum())
# tot_P.append((mean_season.mean().to_array() * res_tot_P.x).sum())
# tot_P.append((mean_season.mean().to_array() * res_S1.x).sum())
tot_P.append((mean_season.mean().to_array() * res_S1.x).sum())

#Total installed capacity
tot_IC = []
tot_IC.append(ic_reduced.to_array().values.sum())
tot_IC.append(ic_reduced_2030.to_array().values.sum())
# tot_IC.append(res_tot_IC.x.sum())
# tot_IC.append(res_tot_P.x.sum())
# tot_IC.append(res_S1.x.sum())
tot_IC.append(res_S1.x.sum())

###############END CALCULATE LSQ AND PREPARE RESULTS###########################


##########################PLOT RESULTS#########################################


#######Plot total variability per season#######
df_tot_var = pd.DataFrame([tot_var_winter/1000, tot_var_spring/1000, tot_var_summer/1000, tot_var_autumn/1000, tot_var/1000], columns=df_var.columns, index=['Winter', 'Spring', 'Summer', 'Autumn', 'Total'])
ax_tot_var = df_tot_var.plot(kind='bar', figsize=(20,10), rot=0, fontsize=14) #yerr=tot_var_std/1000, error_kw=dict(capsize=4,)
ax_tot_var.set_ylabel("Variability in GW", fontsize=14, fontfamily='times new roman')
ax_tot_var.legend(loc=6, bbox_to_anchor=(0.0, 0.8), fontsize=12)

# create a list for tmin and max
s = 0
min = []
max = []
for i in range(4):
    min.append(var_all[s:len(var_summer)+s].min(axis=0)[0]/1000)
    max.append(var_all[s:len(var_summer)+s].max(axis=0)[0]/1000)
    s = s +len(var_summer)
min.append(var_all[:].min(axis=0)[0]/1000)
max.append(var_all[:].max(axis=0)[0]/1000)

s=0
for i in range(4):
    min.append(var_all[s:len(var_summer)+s].min(axis=0)[1]/1000)
    max.append(var_all[s:len(var_summer)+s].max(axis=0)[1]/1000)
    s = s +len(var_summer)
min.append(var_all[:].min(axis=0)[1]/1000)
max.append(var_all[:].max(axis=0)[1]/1000)

s=0
for i in range(4):
    min.append(var_all[s:len(var_summer)+s].min(axis=0)[2]/1000)
    max.append(var_all[s:len(var_summer)+s].max(axis=0)[2]/1000)
    s = s +len(var_summer)
min.append(var_all[:].min(axis=0)[2]/1000)
max.append(var_all[:].max(axis=0)[2]/1000)

m=0
n=1
for p in ax_tot_var.patches:
    x = p.get_x()  # get the bottom left x corner of the bar
    w = p.get_width()  # get width of bar
    h = p.get_height()  # get height of bar
    # line = ax_tot_var.vlines(x+w/2, min[m], max[m], color='k', linewidth =1, linestyles ='solid',)  # draw a vertical line
    # ax_tot_var.plot(x+w/2, min[m], linestyle="", markersize=10, 
    #      marker="*", color="k", label="Min", markeredgecolor="k")
    ax_tot_var.plot(x+w/2, max[m], linestyle="", markersize=8, 
          marker="D", color="k", label="Max", markeredgecolor="k", )
    # ax_tot_var.text(x+w/2, max[m], 'Max', ha='center')
    # ax_tot_var.plot(x+w/2, df_tot_var.to_numpy().transpose().flatten()[m] , linestyle="", markersize=6, 
    #      marker="o", color="k", label="Max", markeredgecolor="k" )
    # print(min[m], max[m])
    m=m+1
   
ax_tot_var.axvspan(-0.5,0.5, facecolor='w', alpha=0.2, label='Winter')
ax_tot_var.axvspan(0.5,1.5, facecolor='g', alpha=0.2, label='Spring')
ax_tot_var.axvspan(1.5,2.5, facecolor='r', alpha=0.2, label='Summer')
ax_tot_var.axvspan(2.5,3.5, facecolor='b', alpha=0.2, label='Autumn')
ax_tot_var.axvspan(3.5,4.5, facecolor='k', alpha=0.2, label='total')

fig_tot_var = ax_tot_var.get_figure()
fig_tot_var.savefig(data_folder / str('fig/' + project +'_tot_variability.png'))

#######END Plot total variability per season#######

#Plot deviation per weather regime and season
ax_var = (df_var/1000).plot(kind='bar', figsize=(20,10), fontsize=14)
#with total Planned IC / power production for 2030 as constraint for every season
# ax_var.set_title('Optimized IC distribution (with installed PV capacity planned for 2030)', fontsize=20, pad=20)
ax_var.legend(loc=2, fontsize=12)
ax_var.set_ylabel('Deviation of PV power production from the seasonal mean in GW', fontsize=14)
ax_var.set_xlabel('Weather regimes', fontsize=14, labelpad=10)
ax_var.axvspan(0,7.5, facecolor='w', alpha=0.2, label='Winter')
ax_var.text(2.5, ax_var.yaxis.get_data_interval().min(), 'Winter', fontsize=14)
ax_var.axvspan(7.5,15.5, facecolor='g', alpha=0.2, label='Spring')
ax_var.text(10.5, ax_var.yaxis.get_data_interval().min(), 'Spring', fontsize=14)
ax_var.axvspan(15.5,23.5, facecolor='r', alpha=0.2, label='Summer')
ax_var.text(18.5, ax_var.yaxis.get_data_interval().min(), 'Summer', fontsize=14)
ax_var.axvspan(23.5,32, facecolor='b', alpha=0.2, label='Autumn')
ax_var.text(26.5, ax_var.yaxis.get_data_interval().min(), 'Autumn', fontsize=14)
fig = ax_var.get_figure()
fig.savefig(data_folder / str('fig/' + project +'_variability.png'))


#Plot optimizied IC distribution


#add newly calculated installed capacity into one dataframe
country = []
for i in ic_reduced_2030:
    country.append(i)

# data_ic = np.c_[ic_reduced_2030.to_array().values, res_tot_IC.x, res_tot_P.x, res_S1.x]
data_ic = np.c_[ic_reduced.to_array().values, ic_reduced_2030.to_array().values, res_S1.x]
df_ic = pd.DataFrame((data_ic), columns=['Installed PV capacity 2019', 'Installed PV capacity planned for 2030', 'Installed PV capacity and production (2030) as constraint (S1)'], index=country)
df_ic.to_excel(data_folder / str(project + 'new-ic.xlsx'))
#ic minus lower bound
df_ic_lb = round((df_ic[['Installed PV capacity planned for 2030', 'Installed PV capacity and production (2030) as constraint (S1)']].transpose() -lb).transpose())


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
    ic_plotting.append(eu.merge(df_ic.iloc[:,i]/1000, left_on = 'country_code', right_index=True))

for i in range(0, len(df_ic_lb.transpose())):
    ic_lb_plotting.append(eu.merge(df_ic_lb.iloc[:,i]/1000, left_on = 'country_code', right_index=True))


f, ax = plt.subplots(
    ncols=3,
    nrows=1,
    figsize=(22, 6),
)

cmap = mpl.cm.get_cmap("Reds")
cmap.set_under(color='white')
vmin = 0.01
vmax = 100

#Plot absolut data
c=0
for i in ic_plotting:
    if c == 0:
        # divider = make_axes_locatable(ax[0,c])
        # cax = divider.append_axes("bottom", size="5%", pad=0.4)
        i.dropna().plot(ax=ax[c], column=i.columns[3], cmap=cmap,edgecolor='black',
                                  vmax=vmax, vmin=vmin,
                                  legend=True, 
                                  legend_kwds={'orientation': "horizontal", }
                                                                  
                                  )
        
        ax[c].set_xlim(left=-13, right=35)
        ax[c].set_ylim(bottom=35, top=70) 
        ax[c].set_title(r"$\bf{" + i.columns[3].replace(' ', '~') + "}$" +
                        '\n Total installed PV capacity (GW): ' + str((tot_IC[c]/1000).round(1)) + 
                        '\n Total mean PV production (GW): ' + str((tot_P[c].values/1000).round(1)) +
                        '\n Total mean variability (GW): ' + str((tot_var[c]/1000).round(1))+ 
                        '\n Total max variability (GW): ' + str((max[c+4*(c+1)]).round(1)), 
                        fontsize=14)
        ax[c].set_axis_off()
        c = c +1
    else:
        i.dropna().plot(ax=ax[c], column=i.columns[3], cmap=cmap,edgecolor='black',
                                  vmax=vmax, vmin=vmin,
                                  )
        
        ax[c].set_xlim(left=-13, right=35)
        ax[c].set_ylim(bottom=35, top=70) 
        ax[c].set_title(r"$\bf{" + i.columns[3].replace(' ', '~') + "}$" +
                        '\n Total installed PV capacity (GW): ' + str((tot_IC[c]/1000).round(1)) + 
                        '\n Total mean PV production (GW): ' + str((tot_P[c].values/1000).round(1)) +
                        '\n Total mean variability (GW): ' + str((tot_var[c]/1000).round(1))+
                        '\n Total max variability (GW): ' + str((max[c+4*(c+1)]).round(1)), 
                        fontsize=14)
        ax[c].set_axis_off()
        c = c +1
# Move legend to rigt place
leg1 = ax[0].get_figure().get_axes()[3]
leg1.set_position([0.37,0.00,0.3,0.1])
leg1.set_xlabel('Total installed PV capacity (in GW)', fontsize=14)
#move subplot
pos2 = [ax[0].get_position().x0, ax[1].get_position().y0, ax[1].get_position().width, ax[1].get_position().height]
ax[0].set_position(pos2)
pos4 = [ax[0].get_position().x0, ax[1].get_position().y0, ax[1].get_position().width, ax[1].get_position().height]
ax[0].set_position(pos4)
    
# f.suptitle('Distribution of installed PV capacity', fontsize=30)
f.savefig(data_folder / str('fig/' + project + '_ic-distribution_absolut.png'))





#Plot data minus lower bound
f, ax = plt.subplots(
    ncols=2,
    nrows=1,
    figsize=(20, 10),
)

cmap = mpl.cm.get_cmap("Reds")
cmap.set_under(color='white')
vmin = 0.01
vmax = 100

c=0
for i in ic_lb_plotting:
    if c == 0:
        # divider = make_axes_locatable(ax[1,c])
        # cax = divider.append_axes("bottom", size="5%", pad=0.4)
        i.dropna().plot(ax=ax[c], column=i.columns[3], cmap=cmap, edgecolor='black',
                                  vmax=vmax, vmin=vmin,
                                  legend=True,
                                  legend_kwds={'orientation': "horizontal",}
                                                                  
                                  )
        
        ax[c].set_xlim(left=-13, right=35)
        ax[c].set_ylim(bottom=35, top=70) 
        ax[c].set_title(r"$\bf{" + i.columns[3].replace(' ', '~') +'~(only~additional)' + "}$" +
                        '\n Additional installed PV capacity (GW): ' + str(((tot_IC[c+1]-tot_IC[0])/1000).round(1)) + 
                        '\n Additional mean PV production (GW): ' + str(((tot_P[c+1]-tot_P[0]).values/1000).round(1)) +
                        '\n Total mean variability (GW): ' + str((tot_var[c+1]/1000).round(1)) +
                        '\n Total max variability (GW): ' + str((max[c+4+1+4*(c+1)]).round(1)), 
                        fontsize=14)
        ax[c].set_axis_off()
        c = c +1
    else:
        i.dropna().plot(ax=ax[c], column=i.columns[3], cmap=cmap, edgecolor='black',
                                  vmax=vmax, vmin=vmin,
                                  )
        
        ax[c].set_xlim(left=-13, right=35)
        ax[c].set_ylim(bottom=35, top=70) 
        ax[c].set_title(r"$\bf{" + i.columns[3].replace(' ', '~') +'~(only~additional)' + "}$" +
                       '\n Additional installed PV capacity (GW): ' + str(((tot_IC[c+1]-tot_IC[0])/1000).round(1)) + 
                        '\n Additional mean PV production (GW): ' + str(((tot_P[c+1]-tot_P[0]).values/1000).round(1)) +
                        '\n Total mean variability (GW): ' + str((tot_var[c+1]/1000).round(1)) +
                        '\n Total max variability (GW): ' + str((max[c+4+1+4*(c+1)]).round(1)), 
                        fontsize=14)
        ax[c].set_axis_off()
        c = c +1
    


# Move legend to rigt place
leg1 = ax[0].get_figure().get_axes()[2]
leg1.set_position([0.37,0.1,0.3,0.1])
leg1.set_xlabel('Additional installed PV capacity (in GW)', fontsize=14)


#move subplot
pos2 = [ax[0].get_position().x0, ax[1].get_position().y0, ax[1].get_position().width, ax[1].get_position().height]
ax[0].set_position(pos2)
pos4 = [ax[0].get_position().x0, ax[1].get_position().y0, ax[1].get_position().width, ax[1].get_position().height]
ax[0].set_position(pos4)
    
# f.suptitle('Distribution of installed PV capacity', fontsize=30)
f.savefig(data_folder / str('fig/' + project + '_ic-distribution_additional.png'))

