# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:33:56 2020

@author: Dirk
"""


import numpy as np
from pathlib import Path
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
# import calendar
import pandas as pd
import matplotlib as mpl
import geopandas as gpd
# from mapclassify import Quantiles, UserDefined


######################Load Datasets#################

data_folder = Path("../data/")

filename_std_ano = data_folder / 'z_all_std_ano.nc'
filename = data_folder / 'gph-daily-mean.nc'
z_all_std_ano = xr.open_dataset(filename_std_ano)['z']
z_all = xr.open_dataset(filename)['z']

file_wr = data_folder / 'wr_time-c7_std.nc'
wr = xr.open_dataset(file_wr)

file_cf_DJF = data_folder / 'results/relative_mean_wr_country_c7_DJF.csv'
relative_mean_wr_country_DJF = pd.read_csv(file_cf_DJF, index_col=0)
file_cf_MAM = data_folder / 'results/relative_mean_wr_country_c7_MAM.csv'
relative_mean_wr_country_MAM = pd.read_csv(file_cf_MAM, index_col=0)
file_cf_JJA = data_folder / 'results/relative_mean_wr_country_c7_JJA.csv'
relative_mean_wr_country_JJA = pd.read_csv(file_cf_JJA, index_col=0)
file_cf_SON = data_folder / 'results/relative_mean_wr_country_c7_SON.csv'
relative_mean_wr_country_SON = pd.read_csv(file_cf_SON, index_col=0)

######################Plot results#################

#Rows and colums
r = 5
c = wr.wr.max().values+1

#Create subplots and colorbar for wr
cmap = mpl.cm.get_cmap("RdBu_r")
plt.close("all")
f, ax = plt.subplots(
    ncols=c,
    nrows=r,
    subplot_kw={"projection": ccrs.Orthographic(central_longitude=-20, central_latitude=60)},
    figsize=(30, 20),
)
cbar_ax = f.add_axes([0.3, .93, 0.4, 0.02])

#map infos for relative capacity factors per country
#Read shapefile using Geopandas
shapefile = data_folder / 'map/NUTS_RG_01M_2021_4326.shp/NUTS_RG_01M_2021_4326.shp'
eu = gpd.read_file(shapefile)[['NAME_LATN', 'CNTR_CODE', 'geometry']]
#Rename columns.
eu.columns = ['country', 'country_code', 'geometry']
cf_plotting_DJF = eu.merge(relative_mean_wr_country_DJF*100, left_on = 'country_code', right_index=True)
cf_plotting_MAM = eu.merge(relative_mean_wr_country_MAM*100, left_on = 'country_code', right_index=True)
cf_plotting_JJA = eu.merge(relative_mean_wr_country_JJA*100, left_on = 'country_code', right_index=True)
cf_plotting_SON = eu.merge(relative_mean_wr_country_SON*100, left_on = 'country_code', right_index=True)
cf_plotting = [cf_plotting_DJF, cf_plotting_MAM, cf_plotting_JJA, cf_plotting_SON]

season = {1: 'winter', 2: 'spring', 3: 'summer', 4: 'autumn'}




vmax_std_ano = 1.5
vmin_std_ano = -1.5

vmax_cf = 15
vmin_cf = -15

for i in range(0,wr.wr.max().values+1):
    mean_wr_std_ano = z_all_std_ano[np.where(wr.wr==i)[0][:]].mean(axis=0)
    frequency = len(np.where(wr.wr == i)[0]) / len(wr.wr)
 
    if i != 0:

        #standard anomalie height plot

        title= 'WR' + str(i) + ' ' +  str(np.round(frequency * 100, decimals=1)) + "%"
        ax[0, i].coastlines()
        ax[0, i].set_global()
        mean_wr_std_ano.plot.contourf(ax=ax[0, i], cmap=cmap, vmin=vmin_std_ano, vmax=vmax_std_ano,
                                  transform=ccrs.PlateCarree(), add_colorbar=False)
        ax[0, i].set_title(title, fontsize=16)
        
        
        #Plot CF
        for s in range(1,len(cf_plotting)+1):
            ax[s, i] = plt.subplot(r, c, c *s + i +1)  # override the GeoAxes object
            cf_plotting[s-1].dropna().plot(ax = ax[s,i], column='WR'+str(i), cmap=cmap,
                                      vmax=vmax_cf, vmin=vmin_cf,
                                      legend=False,)
            #add title to the map
            ax[s,i].set_title('CF during WR'+str(i)+' in '+ str(season[s]), fontdict= 
                        {'fontsize':15})
            #remove axes
            ax[s,i].set_axis_off()
            #only plot relevant part of map
            ax[s,i].set_xlim(left=-20, right=40)
            ax[s,i].set_ylim(bottom=30, top=80)
        
        #move subplot
        # pos1 = ax[1,i].get_position()
        # pos2 = [pos1.x0, ax[1,0].get_position().y0, pos1.width, pos1.height]
        # ax[1,i].set_position(pos2)
        
        
        
        
    else:
  
        #standard anomalie height plot
        # vmax_std_ano = mean_wr_std_ano.max()
        # vmin_std_ano = mean_wr_std_ano.min()
        title= 'WR' + str(i) + ' ' +  str(np.round(frequency * 100, decimals=1)) + "%"
        ax[0, i].coastlines()
        ax[0, i].set_global()
        mean_wr_std_ano.plot.contourf(ax=ax[0, i], vmin=vmin_std_ano, vmax=vmax_std_ano, cmap=cmap,
                                  transform=ccrs.PlateCarree(), add_colorbar=True, 
                                  cbar_kwargs={'label': "Standardized anomalies of geoptential height at 500hPa","orientation": "horizontal"}, 
                                  cbar_ax=cbar_ax)
        ax[0, i].set_title(title, fontsize=16) 
        
        
        
        #Plot CF
        for s in range(1,len(cf_plotting)+1):
            if s==len(cf_plotting):
    
                ax[s, i] = plt.subplot(r, c, c * s +i + 1)  # override the GeoAxes object
                cf_plotting[s-1].dropna().plot(ax = ax[s,i], column='WR'+str(i), cmap=cmap,
                                          vmax=vmax_cf, vmin=vmin_cf,
                                          legend=True, 
                                          legend_kwds={'label': "Deviation from mean capacity factor per country in %",
                                          'orientation': "horizontal",}
                                                                          
                                          )
                #add title to the map
                ax[s,i].set_title('CF during WR'+str(i) +' in '+ str(season[s]), fontdict= 
                            {'fontsize':15})
                #remove axes
                ax[s,i].set_axis_off()
          
                #adjust EU plot --> exclude "far away" regions :-)
                ax[s,i].set_xlim(left=-20, right=40)
                ax[s,i].set_ylim(bottom=30, top=80)
           
            else:
                ax[s, i] = plt.subplot(r, c, c * s +i + 1)  # override the GeoAxes object
                cf_plotting[s-1].dropna().plot(ax = ax[s,i], column='WR'+str(i), cmap=cmap,
                                          vmax=vmax_cf, vmin=vmin_cf,
                                          legend=False, 
                                          )
                #add title to the map
                ax[s,i].set_title('CF during WR'+str(i) +' in '+ str(season[s]), fontdict= 
                            {'fontsize':15})
                #remove axes
                ax[s,i].set_axis_off()
          
                #adjust EU plot --> exclude "far away" regions :-)
                ax[s,i].set_xlim(left=-20, right=40)
                ax[s,i].set_ylim(bottom=30, top=80) 

               
        
     
# Move CF legend to rigt place
leg = ax[1,i].get_figure().get_axes()[12]
leg.set_position([0.3,0.1,0.4,0.02])
#move subplot
pos1 = ax[4,0].get_position()
pos2 = [pos1.x0, ax[4,1].get_position().y0, ax[4,1].get_position().width, ax[4,1].get_position().height]
ax[4,0].set_position(pos2)

    
    #Plot monthly frequency of weather regime    
    # monthly_frequency = wr.where(wr==i).dropna(dim='time').groupby('time.month').count()
    # ax[2, i] = plt.subplot(2, c, c + 1 + i)  # override the GeoAxes object
    # monthly_frequency = pd.Series(data=monthly_frequency.wr, index=calendar.month_abbr[1:13])
    
    # monthly_frequency.plot.bar(ax=ax[2,i])

#¶plt.subplots_adjust(left=0.05, right=0.92, bottom=0.25)

plt.suptitle("Mean weather regime fields (standardized anomalies) and its country specific capacity factor deviation", fontsize=20)
plt.savefig("../data/fig/cf_and_wr_plot_seasons.png")


