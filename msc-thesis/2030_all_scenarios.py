# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:27:19 2020

@author: Dirk Mühlemann
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from scipy.optimize import lsq_linear
import matplotlib as mpl
import geopandas as gpd
import yaml as yaml
import pycountry


######################LOAD DATASETS#############################################

#Load dataset with WR and CF data
data_folder = Path("../data/")
file_ninja = data_folder / 'ninja_and_wr_30days_lowpass_2_0-1_short3.nc'
ninja = xr.open_dataset(file_ninja)


#Load dataset with WR infos
file_wr = data_folder / 'wr_time-c7_std_30days_lowpass_2_0-1_short3.nc'
wr = xr.open_dataset(file_wr)


#Load installed capacity data
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

#Load load/consumption from Eurostat dataset for missing countires
eurostat_country_load_file = data_folder / 'source/eurostat_load.xlsx'
eurostat_country_load = pd.read_excel(eurostat_country_load_file, header=0, parse_dates=[0], index_col=0, squeeze=True)
eurostat_country_load = eurostat_country_load.transpose()

###Rename GB with UK
ninja = ninja.rename_vars({'GB':'UK'})
ic = ic.rename(columns={'GB':'UK'})
ic_2030 = ic_2030.rename(columns={'GB':'UK'})
country_load_year = country_load_year.rename(columns={'GB_UKM':'UK'})

#Create xarray for IC
ic_xr = ic.to_xarray()
ic_2030_xr = ic_2030.to_xarray()

#Remove unneeded data and add need data from eurostat
country_load_year = country_load_year[country_load_year.columns.drop(list(country_load_year.filter(regex='_')))]


#Missing countries in open power data --> #'MT', 'AL','MK', 'BA'#'MD'
country_load_year['MT'] = eurostat_country_load['Malta'].values
country_load_year['AL'] = eurostat_country_load['Albania'].values
country_load_year['MK'] = eurostat_country_load['North Macedonia'].values
country_load_year['BA'] = eurostat_country_load['Bosnia and Herzegovina'].values
country_load_year['MD'] = eurostat_country_load['Moldova'].values
country_load_year = country_load_year.reindex(sorted(country_load_year.columns), axis=1)
last_value = country_load_year.notna()[::-1].idxmax()

#####################END LOAD DATASETS##########################################


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

#Create array for different IC configuration
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

for i in ic_reduced_pot:
    if np.isnan(ic_reduced_pot[i]):
        ic_reduced_pot[i] = ic_reduced_2030[i]* ic_reduced_pot.to_array().sum() / ic_reduced.to_array().sum()

load = ic_reduced.copy()
cf = ninja.mean().drop_vars(wr)
last_value = country_load_year.notna()[::-1].idxmax()

for i in country_load_year:
    load[i] = country_load_year[i][last_value[i]]


#Calculate 10% of load which is used as lb in scenario autarky
ic_min_load = (load.drop_vars('UA')/ (365*24)) / cf *0.1



######################END CREATE DATASET#######################################   


######################DEFINE CONSTRAINTS AND BOUNDS############################   
  
#Define project name which is used for filesaving
project = '2030'

#Naming for variability plots
var_2030 = 'NECPs 2030'
var_prod = 'Variability only'
var_cost = 'Variability & Costs'
var_autarky = 'Variability & Autarky'

#Naming for distribution plots
IC_2030_name = 'NECPs 2030'
IC_prod_name = 'Variability only'
IC_cost_name = 'Variability & Costs'
IC_autarky_name = 'Variability & Autarky'                     

#Define lower and upper bound with already installed capacity   
lb = ic_reduced.to_array().values
ub = ic_reduced_pot.to_array().values


#Define lower bound for scenario autarky (10% of load --> see above)
lb_autarky = ic_min_load.to_array().values


#Norway and MD needs to be adjusted becuase lb is higher than ub
a=0
for i in lb_autarky:
    if lb_autarky[a]-ub[a]>0:
        lb_autarky[a]=ub[a]-0.00000001
    a=a+1

#Define vector b with zeros for variability reduction
b = np.zeros(A_all.shape[0])


#Add tot IC constraint
A_tot_IC = np.append(A_all, [np.ones(A_all.shape[1])], 0)


#IC 2030
target_IC = ic_reduced_2030.to_array().sum()
target_IC_cost = ic_reduced_2030.to_array().sum()*0
b_tot_IC = np.append(b, target_IC)      
b_tot_IC_cost = np.append(b, target_IC_cost) 

#Define total production
P = (mean_season.mean().to_array() * ic_reduced_2030.to_array()).sum()
P_cost = (mean_season.mean().to_array() * ic_reduced_2030.to_array()).sum()*1.5


#Sceanrio variability
b_prod = np.append(b, target_IC)
b_prod = np.append(b_prod, P)
A_prod = np.append(A_all, [np.ones(A_all.shape[1])], 0)
A_prod = np.append(A_prod, [mean_season.mean().to_array()], 0)


#Sceanrio cost
b_cost = np.append(b, target_IC_cost)
b_cost = np.append(b_cost, P_cost)
A_cost = np.append(A_all, [np.ones(A_all.shape[1])], 0)
A_cost = np.append(A_cost, [mean_season.mean().to_array()], 0)


#Sceanrio autarky
b_autarky = np.append(b, target_IC)
b_autarky = np.append(b_autarky, P)
A_autarky = np.append(A_all, [np.ones(A_all.shape[1])], 0)
A_autarky = np.append(A_autarky, [mean_season.mean().to_array()], 0)

######################END DEFINIDTIONn#########################################       


###############CALCULATE LSQ AND PREPARE RESULTS###############################


#Weighting for variability
W_prod = np.ones(b_prod.shape)
W_cost = np.ones(b_cost.shape)*10
W_autarky = np.ones(b_autarky.shape)

#Weighting for Installed capacity
W_prod[-2]=0
W_cost[-2]=0.1
W_autarky[-2]=0

#Weighting for Production
W_prod[-1]=10
W_cost[-1]=8.7
W_autarky[-1]=10

#Add weights to coefficent matrix A and target vector b
A_prod = A_prod * np.sqrt(W_prod[:,np.newaxis])
b_prod = b_prod * np.sqrt(W_prod)
A_cost = A_cost * np.sqrt(W_cost[:,np.newaxis])
b_cost = b_cost * np.sqrt(W_cost)
A_autarky = A_autarky * np.sqrt(W_autarky[:,np.newaxis])
b_autarky = b_autarky * np.sqrt(W_autarky)


#Calc LSQ
res_prod = lsq_linear(A_prod, b_prod, bounds=(lb,ub))
res_cost = lsq_linear(A_cost, b_cost, bounds=(lb,ub))
res_autarky = lsq_linear(A_autarky, b_autarky, bounds=(lb_autarky,ub))


#Calculate variability for each wr season and specific IC
state_2030 = (A_all * ic_reduced_2030.to_array().values).sum(axis=1)
state_prod = (A_all * res_prod.x).sum(axis=1)
state_cost = (A_all * res_cost.x).sum(axis=1)
state_autarky = (A_all * res_autarky.x).sum(axis=1)


#Create Dataframe
data_var = np.c_[state_2030, state_prod, state_cost, state_autarky]
df_var = pd.DataFrame((data_var), columns=[var_2030, var_prod, var_cost, var_autarky])
for i in range(0,8):
    df_var = df_var.rename({i: 'WR'+str(i)}, axis='index')
    df_var = df_var.rename({i+8:'WR'+str(i)}, axis='index')
    df_var = df_var.rename({i+16:'WR'+str(i)}, axis='index')
    df_var = df_var.rename({i+24:'WR'+str(i)}, axis='index')
df_var = df_var.rename({'WR7':'no regime'})



#Calculate frequencies per wr
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


#Calculate transition from one wr to next wr
df_transition_fre = []
for i in ('DJF', 'MAM', 'JJA', 'SON'):
    # print(i)
    df_transition = pd.DataFrame(np.zeros((8,8)),
                   columns=[0,1,2,3,4,5,6,7])
    wr_list = wr.wr[wr.groupby('time.season').groups[i]].values.tolist() 
    for i, j in enumerate(wr_list[:-1]):
        if j  != wr_list[i+1]: 
            # print(wr_list[i])
            # print(wr_list[i+1])
            df_transition[wr_list[i]][wr_list[i+1]] = df_transition[wr_list[i]][wr_list[i+1]] + 1
          
    df_transition_fre.append(df_transition / df_transition.to_numpy().flatten().sum())

#Calcualte frequency times #days of wr per season --> for toal variability
temp_fre_DJF = np.transpose(np.array([frequency.loc['DJF']]*3))
temp_fre_MAM = np.transpose(np.array([frequency.loc['MAM']]*3))
temp_fre_JJA = np.transpose(np.array([frequency.loc['JJA']]*3))
temp_fre_SON = np.transpose(np.array([frequency.loc['SON']]*3))
temp_fre = np.concatenate((temp_fre_DJF, temp_fre_MAM, temp_fre_JJA, temp_fre_SON))


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


var_winter_fre = abs((df_var[:].to_numpy()[0] - df_var[:].to_numpy()[0:8])) * (np.transpose(np.array([df_transition_fre[0][0]+df_transition_fre[0].loc[0]]*4)))
var_spring_fre = abs((df_var[:].to_numpy()[0+8] - df_var[:].to_numpy()[8:16])) * (np.transpose(np.array([df_transition_fre[1][0]+df_transition_fre[1].loc[0]]*4)))
var_summer_fre = abs((df_var[:].to_numpy()[0+16] - df_var[:].to_numpy()[16:24])) * (np.transpose(np.array([df_transition_fre[2][0]+df_transition_fre[2].loc[0]]*4)))
var_autumn_fre = abs((df_var[:].to_numpy()[0+24] - df_var[:].to_numpy()[24:32])) * (np.transpose(np.array([df_transition_fre[3][0]+df_transition_fre[3].loc[0]]*4)))
for i in range(1,8):
    var_winter_fre= np.append(var_winter_fre, abs((df_var[:].to_numpy()[i] - df_var[:].to_numpy()[0:8])) * (np.transpose(np.array([df_transition_fre[0][i]+df_transition_fre[0].loc[i]]*4))), axis=0)
    var_spring_fre = np.append(var_spring_fre,  abs((df_var[:].to_numpy()[i+8] - df_var[:].to_numpy()[8:16])) * (np.transpose(np.array([df_transition_fre[1][i]+df_transition_fre[1].loc[i]]*4))), axis=0)
    var_summer_fre = np.append(var_summer_fre, abs((df_var[:].to_numpy()[i+16] - df_var[:].to_numpy()[16:24])) * (np.transpose(np.array([df_transition_fre[2][i]+df_transition_fre[2].loc[i]]*4))), axis=0)
    var_autumn_fre = np.append(var_autumn_fre, abs((df_var[:].to_numpy()[i+24] - df_var[:].to_numpy()[24:32])) * (np.transpose(np.array([df_transition_fre[3][i]+df_transition_fre[3].loc[i]]*4))), axis=0)

var_winter_fre = np.array([list(dict.fromkeys(var_winter_fre.transpose().tolist()[0]))[1:],
                         list(dict.fromkeys(var_winter_fre.transpose().tolist()[1]))[1:],
                         list(dict.fromkeys(var_winter_fre.transpose().tolist()[2]))[1:],
                         list(dict.fromkeys(var_winter_fre.transpose().tolist()[3]))[1:]]).transpose()

var_spring_fre = np.array([list(dict.fromkeys(var_spring_fre.transpose().tolist()[0]))[1:],
                         list(dict.fromkeys(var_spring_fre.transpose().tolist()[1]))[1:],
                         list(dict.fromkeys(var_spring_fre.transpose().tolist()[2]))[1:],
                         list(dict.fromkeys(var_spring_fre.transpose().tolist()[3]))[1:]]).transpose()

var_summer_fre = np.array([list(dict.fromkeys(var_summer_fre.transpose().tolist()[0]))[1:],
                         list(dict.fromkeys(var_summer_fre.transpose().tolist()[1]))[1:],
                         list(dict.fromkeys(var_summer_fre.transpose().tolist()[2]))[1:],
                         list(dict.fromkeys(var_summer_fre.transpose().tolist()[3]))[1:]]).transpose()

var_autumn_fre = np.array([list(dict.fromkeys(var_autumn_fre.transpose().tolist()[0]))[1:],
                         list(dict.fromkeys(var_autumn_fre.transpose().tolist()[1]))[1:],
                         list(dict.fromkeys(var_autumn_fre.transpose().tolist()[2]))[1:],
                         list(dict.fromkeys(var_autumn_fre.transpose().tolist()[3]))[1:]]).transpose()

var_all_fre = np.vstack((var_winter_fre, var_spring_fre, var_summer_fre, var_autumn_fre))


   
#Calc total variability
tot_var_winter = sum(var_winter_fre)#/len(var_winter)
tot_var_spring = sum(var_spring_fre)#/len(var_spring)
tot_var_summer = sum(var_summer_fre)#/len(var_summer)
tot_var_autumn = sum(var_autumn_fre)#/len(var_autumn)
days_seasons = wr.groupby('time.season').count()
tot_var = ((days_seasons.sel(season='DJF').to_array()*tot_var_winter + days_seasons.sel(season='MAM').to_array()*tot_var_spring + days_seasons.sel(season='JJA').to_array()*tot_var_summer + days_seasons.sel(season='SON').to_array()*tot_var_autumn)/    days_seasons.sum().to_array().values).values
tot_var_std = np.array([[var_winter.std(axis=0)[0], var_spring.std(axis=0)[0], var_summer.std(axis=0)[0], var_autumn.std(axis=0)[0], var_all.std(axis=0)[0]],
                        [var_winter.std(axis=0)[1], var_spring.std(axis=0)[1] ,var_summer.std(axis=0)[1], var_autumn.std(axis=0)[1], var_all.std(axis=0)[1]],
                        [var_winter.std(axis=0)[2], var_spring.std(axis=0)[2] ,var_summer.std(axis=0)[2], var_autumn.std(axis=0)[2], var_all.std(axis=0)[2]],
                        [var_winter.std(axis=0)[3], var_spring.std(axis=0)[3] ,var_summer.std(axis=0)[3], var_autumn.std(axis=0)[3], var_all.std(axis=0)[3]]])




#Total production
tot_P = []
tot_P.append((mean_season.mean().to_array() * ic_reduced.to_array()).sum())
tot_P.append((mean_season.mean().to_array() * ic_reduced_2030.to_array()).sum())
tot_P.append((mean_season.mean().to_array() * res_prod.x).sum())
tot_P.append((mean_season.mean().to_array() * res_cost.x).sum())
tot_P.append((mean_season.mean().to_array() * res_autarky.x).sum())

#Total installed capacity
tot_IC = []
tot_IC.append(ic_reduced.to_array().values.sum())
tot_IC.append(ic_reduced_2030.to_array().values.sum())
tot_IC.append(res_prod.x.sum())
tot_IC.append(res_cost.x.sum())
tot_IC.append(res_autarky.x.sum())

###############END CALCULATE LSQ AND PREPARE RESULTS###########################


##########################PLOT RESULTS#########################################


#######Plot total variability per season#######
df_tot_var = pd.DataFrame([tot_var_winter/1000, tot_var_spring/1000, tot_var_summer/1000, tot_var_autumn/1000, tot_var/1000], columns=df_var.columns, index=['Winter', 'Spring', 'Summer', 'Autumn', 'Total'])
my_colors = ['dimgrey', 'purple', '#DC5D4B', 'gold', 'k']
ax_tot_var = df_tot_var.plot(kind='bar', figsize=(8.27,4.135), rot=0, fontsize=9, color=my_colors)
ax_tot_var.set_ylabel("Variability in GW", fontsize=9)


#Create a list for min and max variability
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

s=0
for i in range(4):
    min.append(var_all[s:len(var_summer)+s].min(axis=0)[3]/1000)
    max.append(var_all[s:len(var_summer)+s].max(axis=0)[3]/1000)
    s = s +len(var_summer)
min.append(var_all[:].min(axis=0)[3]/1000)
max.append(var_all[:].max(axis=0)[3]/1000)


#Add max var as marker to plot
m=0
n=1
for p in ax_tot_var.patches:
    x = p.get_x()  # get the bottom left x corner of the bar
    w = p.get_width()  # get width of bar
    h = p.get_height()  # get height of bar
    if m ==0:
        ax_tot_var.plot(x+w/2, max[m], linestyle="", markersize=3, 
          marker="D", color="k", label="Max var", markeredgecolor="k", )
    else:
        ax_tot_var.plot(x+w/2, max[m], linestyle="", markersize=3, 
          marker="D", color="k", label="_Max var", markeredgecolor="k", )
    m=m+1

leg = ax_tot_var.legend(loc=6, bbox_to_anchor=(0.0, 0.75), fontsize=9,)    
   
ax_tot_var.axvspan(-0.5,0.5, facecolor='w', alpha=0.08, label='Winter')
ax_tot_var.axvspan(0.5,1.5, facecolor='g', alpha=0.08, label='Spring')
ax_tot_var.axvspan(1.5,2.5, facecolor='r', alpha=0.08, label='Summer')
ax_tot_var.axvspan(2.5,3.5, facecolor='b', alpha=0.08, label='Autumn')
ax_tot_var.axvspan(3.5,4.5, facecolor='k', alpha=0.08, label='total')


fig_tot_var = ax_tot_var.get_figure()
fig_tot_var.savefig(data_folder / str('fig/' + project +'_tot_variability.tiff'), dpi=300)

#######END Plot total variability per season#######



#######Plot deviation per weather regime and season#######
ax_var = (df_var/1000).plot(kind='bar', figsize=(8.27,4.135), fontsize=9, color=my_colors)
ax_var.legend(loc=2, fontsize=9)
ax_var.set_ylabel('Deviation of PV power production from the seasonal mean in GW', fontsize=9)
ax_var.set_xlabel('Weather regimes', fontsize=9, labelpad=10)
ax_var.axvspan(0,7.5, facecolor='w', alpha=0.08, label='Winter')
ax_var.text(2.5, ax_var.yaxis.get_data_interval().min(), 'Winter', fontsize=9)
ax_var.axvspan(7.5,15.5, facecolor='g', alpha=0.08, label='Spring')
ax_var.text(10.5, ax_var.yaxis.get_data_interval().min(), 'Spring', fontsize=9)
ax_var.axvspan(15.5,23.5, facecolor='r', alpha=0.08, label='Summer')
ax_var.text(18.5, ax_var.yaxis.get_data_interval().min(), 'Summer', fontsize=9)
ax_var.axvspan(23.5,32, facecolor='b', alpha=0.08, label='Autumn')
ax_var.text(26.5, ax_var.yaxis.get_data_interval().min(), 'Autumn', fontsize=9)
fig = ax_var.get_figure()
fig.savefig(data_folder / str('fig/' + project +'_variability.tiff'),dpi=300)
#######END Plot deviation per weather regime and season#######




#######Plot optimizied IC distribution#######
#Add newly calculated installed capacity into dataframe (absolut IC and only addtional IC)
country = []
for i in ic_reduced_2030:
    country.append(i)
data_ic = np.c_[ic_reduced_2030.to_array().values, res_prod.x, res_cost.x, res_autarky.x]
df_ic = pd.DataFrame((data_ic), columns=[IC_2030_name, IC_prod_name, IC_cost_name, IC_autarky_name], index=country)
df_ic.to_excel(data_folder / str(project + 'new-ic.xlsx'))
df_ic_lb = round((df_ic[[IC_2030_name, IC_prod_name, IC_cost_name, IC_autarky_name]].transpose() -lb).transpose())
df_ic_ub = round((df_ic[[IC_2030_name, IC_prod_name, IC_cost_name, IC_autarky_name]].transpose() -ub).transpose())


#Read and adjust shapefile using Geopandas
shapefile = data_folder / 'map/CNTR_RG_01M_2020_4326/CNTR_RG_01M_2020_4326.shp'
eu = gpd.read_file(shapefile)[['CNTR_NAME', 'CNTR_ID', 'geometry']]
eu.columns = ['country', 'country_code', 'geometry']
eu.country_code[60] = 'GR'


#Merge results into geopandas
ic_plotting = []
ic_lb_plotting = []
for i in range(0, len(df_ic.transpose())):
    ic_plotting.append(eu.merge(df_ic.iloc[:,i]/1000, left_on = 'country_code', right_index=True))

for i in range(0, len(df_ic_lb.transpose())):
    ic_lb_plotting.append(eu.merge(df_ic_lb.iloc[:,i]/1000, left_on = 'country_code', right_index=True))



#######Plot distirbution for TOTAL/ABSOLUT installed capacity#######
f, ax = plt.subplots(
    ncols=2,
    nrows=2,
    figsize=(7.48, 7),
)

cmap = mpl.cm.get_cmap("Reds")
cmap.set_under(color='white')
vmin = 0.01
vmax = 100
labels = ["a)", "b)", 'c)', 'd)']

n=0
for i in ic_plotting:
    ma = i.dropna().plot(ax=ax.reshape(-1)[n], column=i.columns[3], cmap=cmap, edgecolor='black', linewidth=0.1,
                              vmax=vmax, vmin=vmin,
                              )
    #Adjust map    
    ax.reshape(-1)[n].set_xlim(left=-13, right=35)
    ax.reshape(-1)[n].set_ylim(bottom=34, top=70) 
    ax.reshape(-1)[n].set_title(labels[n] + " " + i.columns[3],pad=-10, loc='left', fontsize=9)
    
    #Add table with IC and variability    
    col = ('Capacity', 'Mean output', 'Mean var', 'Max var')
    cell_text = [[str((tot_IC[n+1]/1000).round(1)) + ' GW', 
                  str(((tot_P[n+1]).values/1000).round(1)) + ' GW',
                  str((tot_var[n]/1000).round(1)) + ' GW', 
                  str((max[5*(n+1)-1]).round(1)) + ' GW']]
    table = ax.reshape(-1)[n].table(cellText=cell_text,
                  colLabels=col,
                  cellLoc = 'center',
                  loc = 'bottom',
                  edges = 'open',
                  )
    
    #Format table borders
    for i in range(4):
        if i==3:
            table.get_celld()[0,i].visible_edges = 'B'
        else:
            table.get_celld()[0,i].visible_edges = 'BR'
            table.get_celld()[1,i].visible_edges = 'R'
    ax.reshape(-1)[n].set_axis_off()
    n = n +1

#Remove space between supplots
f.subplots_adjust(wspace=0.1, hspace=0.1)  

#Add colorbar which is not straight forward with geopandas and mubiple subplots
cax = f.add_axes([0.3,0.07,0.4,0.02])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
#Fake up the array of the scalar mappable. Urgh...
sm._A = []
cbar = f.colorbar(sm, cax=cax, orientation='horizontal')
cbar.set_label('Additional installed PV capacity (in GW)', fontsize=9)
f.savefig(data_folder / str('fig/' + project + '_ic-distribution_absolut.tiff'), dpi=300)


#######END Plot distirbution for TOTAL installed capacity#######


#######Plot distirbution for ADDITIONAL installed capacity#######
f, ax = plt.subplots(
    ncols=2,
    nrows=2,
    figsize=(7.48, 7),
)

cmap = mpl.cm.get_cmap("Reds")
cmap.set_under(color='white')
vmin = 0.01
vmax = 100
n=0

for i in ic_lb_plotting:
    ma = i.dropna().plot(ax=ax.reshape(-1)[n], column=i.columns[3], cmap=cmap, edgecolor='black', linewidth=0.1,
                              vmax=vmax, vmin=vmin,
                              )
    #Adjust map    
    ax.reshape(-1)[n].set_xlim(left=-13, right=35)
    ax.reshape(-1)[n].set_ylim(bottom=34, top=70) 
    ax.reshape(-1)[n].set_title(labels[n] + " " + i.columns[3],pad=-10, loc='left', fontsize=9)
    
    #Add table with IC and variability    
    col = ('Capacity', 'Mean output', 'Mean var', 'Max var')
    cell_text = [[str((tot_IC[n+1]/1000).round(1)) + ' GW', 
                  str(((tot_P[n+1]).values/1000).round(1)) + ' GW',
                  str((tot_var[n]/1000).round(1)) + ' GW', 
                  str((max[5*(n+1)-1]).round(1)) + ' GW']]
    table = ax.reshape(-1)[n].table(cellText=cell_text,
                  colLabels=col,
                  cellLoc = 'center',
                  # bbox = [0, 0.9, 0.5, 0.2],
                  loc = 'bottom',
                  edges = 'open',
                  )
        
    #Format table borders
    for i in range(4):
        if i==3:
            table.get_celld()[0,i].visible_edges = 'B'
        else:
            table.get_celld()[0,i].visible_edges = 'BR'
            table.get_celld()[1,i].visible_edges = 'R'
    ax.reshape(-1)[n].set_axis_off()
    n = n +1

#remove space between supplots
f.subplots_adjust(wspace=0.1, hspace=0.1)  

#Add colorbar which is not straight forward with geopandas and mubiple subplots
cax = f.add_axes([0.3,0.07,0.4,0.02])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
# fake up the array of the scalar mappable. Urgh...
sm._A = []
cbar = f.colorbar(sm, cax=cax, orientation='horizontal')
cbar.set_label('Additional installed PV capacity (in GW)', fontsize=9)


f.savefig(data_folder / str('fig/' + project + '_ic-distribution_additional.tiff'), dpi=300)
#######END Plot distirbution for ADDTIONAL installed capacity#######