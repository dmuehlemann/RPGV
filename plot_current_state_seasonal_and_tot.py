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

filename_std_ano = data_folder / 'z_all_std_ano_30days_lowpass_2_0-25.nc'
z_all_std_ano = xr.open_dataset(filename_std_ano)['z']

file_wr = data_folder / 'wr_time-c7_std_14days_lowpass_2_0-25.nc'
wr = xr.open_dataset(file_wr)

file_ninja = data_folder / 'ninja_and_wr_30days_lowpass_2_0-25.nc'
ninja = xr.open_dataset(file_ninja)



#####CF calculations##########
ninja.where(ninja.wr==0, drop=True)


ninja_tot = []
ninja_season= []

for i in range(0, int(wr.wr.max())+1):
    ninja_tot.append((ninja.drop('wr').where(ninja.wr==i, drop=True).mean() - ninja.drop('wr').mean())/ninja.drop('wr').mean())
    ninja_season.append((ninja.drop('wr').where(ninja.wr==i, drop=True).groupby('time.season').mean() - \
                          ninja.drop('wr').groupby('time.season').mean()) / \
                          ninja.drop('wr').groupby('time.season').mean()
                          )
    ninja_tot[i] = ninja_tot[i].expand_dims('CF')



######################Plot results#################


#map infos for relative capacity factors per country
#Read shapefile using Geopandas
shapefile = data_folder / 'map/NUTS_RG_01M_2021_4326.shp/NUTS_RG_01M_2021_4326.shp'
eu = gpd.read_file(shapefile)[['NAME_LATN', 'CNTR_CODE', 'geometry']]
#Rename columns.
eu.columns = ['country', 'country_code', 'geometry']


cf_plotting = []

for i in range(0, int(wr.wr.max())+1):
    cf_plotting.append(eu.merge(ninja_season[i].to_dataframe().transpose()*100, left_on = 'country_code', right_index=True))
    temp = eu.merge(ninja_tot[i].to_dataframe().transpose()*100, left_on = 'country_code', right_index=True)
    cf_plotting[i] = cf_plotting[i].assign(tot=temp[0])



season = {0: 'DJF', 1: 'MAM', 2: 'JJA', 3: 'SON', 4: 'tot'}


#Rows and colums
r = 6
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
        s=1
        for a in season.values():
            #ax[s, i] = plt.subplot(r, c, c *s + i +1)  # override the GeoAxes object
            cf_plotting[i].dropna().plot(ax = ax[s,i], column=a, cmap=cmap,
                                      vmax=vmax_cf, vmin=vmin_cf,
                                      legend=False, 
                                      )
            #add title to the map
            ax[s,i].set_title('CF during WR'+str(i) + ' ' + str(a), fontdict= 
                        {'fontsize':15})
            #remove axes
            ax[s,i].set_axis_off()
      
            #adjust EU plot --> exclude "far away" regions :-)
            ax[s,i].set_xlim(left=-20, right=40)
            ax[s,i].set_ylim(bottom=30, top=80) 
            s=s+1
        
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
        s = 1
        for a in season.values():
            if a==len(season.values()):
    
                #ax[s, i] = plt.subplot(r, c, c * s +i + 1)  # override the GeoAxes object
                cf_plotting[i].dropna().plot(ax = ax[s,i], column=a, cmap=cmap,
                                          vmax=vmax_cf, vmin=vmin_cf,
                                          legend=True, 
                                          legend_kwds={'label': "Deviation from mean capacity factor per country in %",
                                          'orientation': "horizontal",}
                                                                          
                                          )
                #add title to the map
                ax[s,i].set_title('CF during WR'+str(i) +' ' + str(a), fontdict= 
                            {'fontsize':15})
                #remove axes
                ax[s,i].set_axis_off()
          
                #adjust EU plot --> exclude "far away" regions :-)
                ax[s,i].set_xlim(left=-20, right=40)
                ax[s,i].set_ylim(bottom=30, top=80)
                s = s +1
           
            else:
                #ax[s, i] = plt.subplot(r, c, c * s +i + 1)  # override the GeoAxes object
                cf_plotting[i].dropna().plot(ax = ax[s,i], column=a, cmap=cmap,
                                          vmax=vmax_cf, vmin=vmin_cf,
                                          legend=False, 
                                          )
                #add title to the map
                ax[s,i].set_title('CF during WR'+str(i) + ' ' + str(a), fontdict= 
                            {'fontsize':15})
                #remove axes
                ax[s,i].set_axis_off()
          
                #adjust EU plot --> exclude "far away" regions :-)
                ax[s,i].set_xlim(left=-20, right=40)
                ax[s,i].set_ylim(bottom=30, top=80) 
                s=s+1

               
        
     
# Move CF legend to rigt place
leg = ax[1,i].get_figure().get_axes()[42]
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
plt.savefig("../data/fig/cf_and_wr_plot_v2.png")


