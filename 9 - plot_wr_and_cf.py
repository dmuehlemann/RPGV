# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:33:56 2020

@author: Dirk

This scripts creates a plot with all weather regimes and the 
related seasonal capacity factors


"""


import numpy as np
from pathlib import Path
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib as mpl
import geopandas as gpd


######################Load Datasets#################

data_folder = Path("../data/")

filename_std_ano = data_folder / 'z_all_std_ano_30days_lowpass_2_0-1.nc'
z_all_std_ano = xr.open_dataset(filename_std_ano)['z']

file_wr = data_folder / 'wr_time-c7_std_30days_lowpass_2_0-1_short3.nc'
wr = xr.open_dataset(file_wr)

file_ninja = data_folder / 'ninja_and_wr_30days_lowpass_2_0-1_short3.nc'
ninja = xr.open_dataset(file_ninja)



###here no Brexit ;-)
ninja = ninja.rename_vars({'GB':'UK'})
ninja = ninja.rename_vars({'GR':'EL'})

#####CF calculations##########


ninja_tot = []
ninja_season= []

##absolut anomalie
for i in range(0, int(wr.wr.max())+1):
    ninja_tot.append((ninja.drop('wr').where(ninja.wr==i, drop=True).mean() - ninja.drop('wr').mean()))
    ninja_season.append((ninja.drop('wr').where(ninja.wr==i, drop=True).groupby('time.season').mean() - \
                          ninja.drop('wr').groupby('time.season').mean()))
    ninja_tot[i] = ninja_tot[i].expand_dims('CF')



######################Plot results#################


#Map infos for relative capacity factors per country
#Read shapefile using Geopandas
shapefile = data_folder / 'map/CNTR_RG_01M_2020_4326/CNTR_RG_01M_2020_4326.shp'
eu = gpd.read_file(shapefile)[['CNTR_NAME', 'CNTR_ID', 'geometry']]
#Rename columns
eu.columns = ['country', 'country_code', 'geometry']


cf_plotting = []

for i in range(0, int(wr.wr.max())+1):
    cf_plotting.append(eu.merge(ninja_season[i].to_dataframe().transpose(), left_on = 'country_code', right_index=True))
    temp = eu.merge(ninja_tot[i].to_dataframe().transpose(), left_on = 'country_code', right_index=True)
    cf_plotting[i] = cf_plotting[i].assign(tot=temp[0])



season = {0: 'DJF', 1: 'MAM', 2: 'JJA', 3: 'SON'}


#Rows and colums
r = 5
c = wr.wr.max().values+1

#Create subplots and colorbar for wr
cmap = mpl.cm.get_cmap("RdGy_r")
cmap_wr = mpl.cm.get_cmap("RdYlBu_r")

plt.close("all")
f, ax = plt.subplots(
    ncols=c,
    nrows=r,
    subplot_kw={"projection": ccrs.Orthographic(central_longitude=-20, central_latitude=60)},
    figsize=(11.69,7.2),
)
cbar_ax = f.add_axes([0.3, 0.94, 0.4, 0.02])



vmax_std_ano = 2.1
vmin_std_ano = -2.1

vmax_cf = 0.02
vmin_cf = -0.02

for i in range(0,wr.wr.max().values+1):
    mean_wr_std_ano = z_all_std_ano[np.where(wr.wr==i)[0][:]].mean(axis=0)
    frequency = len(np.where(wr.wr == i)[0]) / len(wr.wr)
 
    if i != 0:

        #standard anomalie height plot
        if i==wr.wr.max().values:
            title= 'no regime ' +  str(np.round(frequency * 100, decimals=1)) + "%"
        else:
            title= 'WR' + str(i+1) + ' ' +  str(np.round(frequency * 100, decimals=1)) + "%"
        ax[0, i].coastlines()
        ax[0, i].set_global()
        mean_wr_std_ano.plot.imshow(ax=ax[0,i], cmap=cmap_wr, transform=ccrs.PlateCarree(), add_colorbar=False)
        ax[0, i].set_title(title, fontsize=11,)
        
        
        #Plot CF
        s=1
        for a in season.values():
            ax[s, i] = plt.subplot(r, c, c * s + i + 1)  # override the GeoAxes object
            cf_plotting[i].dropna().plot(ax = ax[s,i], column=a, cmap=cmap, edgecolor='black', linewidth=0.05,
                                      vmax=vmax_cf, vmin=vmin_cf,
                                      legend=False, 
                                      )
            ax[s,i].set_axis_off()
      
            #adjust EU plot --> exclude "far away" regions :-)
            ax[s,i].set_xlim(left=-12, right=35)
            ax[s,i].set_ylim(bottom=32, top=72) 
            s=s+1
            
     
        
        
    else:
  
        #standard anomalie height plot
        title= 'WR' + str(i+1) + ' ' +  str(np.round(frequency * 100, decimals=1)) + "%"
        ax[0, i].coastlines()
        ax[0, i].set_global()
        con = mean_wr_std_ano.plot.imshow(ax=ax[0,i],  cmap=cmap_wr, transform=ccrs.PlateCarree(), add_colorbar=False) 
        cb = plt.colorbar(con, cax=cbar_ax, orientation='horizontal',)
        cb.ax.set_title(label='Standardized anomalies of geoptential height at 500hPa [unitless]',size=9,)
        ax[0, i].set_title(title, fontsize=11,) 
        
        
        
        #Plot CF
        s = 1
        for a in season.values():
            if a=='SON':
    
                ax[s, i] = plt.subplot(r, c, c * s + i + 1)  # override the GeoAxes object
                plt.subplots_adjust(wspace=0.05, hspace=0.001)   
                cf_plotting[i].dropna().plot(ax = ax[s,i], column=a, cmap=cmap, edgecolor='black', linewidth=0.05,
                                          vmax=vmax_cf, vmin=vmin_cf,
                                          legend=True, 
                                          legend_kwds={'label': "Capacity factor anomaly [unitless]",
                                          'orientation': "horizontal",}
                                                                          
                                          )
                #add title to the map
                # ax[s,i].set_title('CF during WR'+str(i) +' ' + str(a), fontsize=16, **csfont)
                #remove axes
                ax[s,i].set_axis_off()
          
                #adjust EU plot --> exclude "far away" regions :-)
                ax[s,i].set_xlim(left=-12, right=35)
                ax[s,i].set_ylim(bottom=32, top=72) 
                s = s +1
           
            else:
                ax[s, i] = plt.subplot(r, c, c * s + i + 1)  # override the GeoAxes object
                plt.subplots_adjust(wspace=0.05, hspace=0.001)   
                cf_plotting[i].dropna().plot(ax = ax[s,i], column=a, cmap=cmap, edgecolor='black', linewidth=0.1,
                                          vmax=vmax_cf, vmin=vmin_cf,
                                          legend=False, 
                                          )
                #remove axes
                ax[s,i].set_axis_off()
          
                #adjust EU plot --> exclude "far away" regions :-)
                ax[s,i].set_xlim(left=-12, right=35)
                ax[s,i].set_ylim(bottom=32, top=72) 
                s=s+1

f.subplots_adjust(wspace=0.05, hspace=-0.2)               
        
     
# Move CF legend to rigt place
leg = ax[1,i].get_figure().get_axes()[13]
leg.set_position([0.3,0.12,0.4,0.02])
leg.set_xlabel("Capacity factor anomaly [unitless]", fontsize=9,)
#Move subplot
pos1 = ax[4,0].get_position()
pos2 = [ax[3,0].get_position().x0, ax[4,1].get_position().y0, ax[4,1].get_position().width, ax[4,1].get_position().height]
ax[4,0].set_position(pos2)


#Some cosmetic adjustments to the plot
rect = plt.Rectangle((-14,31),50*8, 45,
                     clip_on=False,
                     facecolor='w',alpha=0.08, label='Winter')
ax[1,0].add_patch(rect)
ax[1,0].text(-21, 42, 'Winter', fontsize=11, rotation='vertical')

rect = plt.Rectangle((-14,31),50*8, 45,
                     clip_on=False,
                     facecolor='g',alpha=0.08, label='Spring')
ax[2,0].add_patch(rect)
ax[2,0].text(-21, 42, 'Spring', fontsize=11, rotation='vertical')


rect = plt.Rectangle((-14,31),50*8, 45,
                     clip_on=False,
                     facecolor='r',alpha=0.08, label='Summer')
ax[3,0].add_patch(rect)
ax[3,0].text(-21, 42, 'Summer', fontsize=11, rotation='vertical')


rect = plt.Rectangle((-14,31),50*8, 45,
                     clip_on=False,
                     facecolor='b',alpha=0.08, label='Autumn')
ax[4,0].add_patch(rect)
ax[4,0].text(-21, 42, 'Autumn', fontsize=11, rotation='vertical')


plt.savefig("../data/fig/wr_and_cf.tiff", dpi=300)


