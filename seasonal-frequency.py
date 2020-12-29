# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:52:17 2020

@author: Dirk
"""

import numpy as np
from pathlib import Path
import xarray as xr
import pandas as pd
import calendar
import itertools
import matplotlib.pyplot as plt


######################Load Datasets#################

data_folder = Path("../data/")

filename_std_ano = data_folder / 'z_all_std_ano_30days_lowpass_2_0-1.nc'
z_all_std_ano = xr.open_dataset(filename_std_ano)['z']

file_wr = data_folder / 'wr_time-c7_std_30days_lowpass_2_0-1_short10.nc'
wr = xr.open_dataset(file_wr)

fig_out = data_folder / 'fig/cumulative__frequency.png'

######################Calculate frequencies#################

seasonal_count = pd.concat([pd.DataFrame([z_all_std_ano.time[np.where(wr.wr==i)[0][:]].groupby('time.season').count().values], columns=z_all_std_ano.time.groupby('time.season').count().season) for i in range(0,wr.wr.max().values+1)],
      ignore_index=True)

seasonal_frequency = pd.concat([pd.DataFrame([seasonal_count.loc[i].values / seasonal_count.sum(axis=0).values], columns=z_all_std_ano.time.groupby('time.season').count().season) for i in range(0,wr.wr.max().values+1)],
      ignore_index=True)


month_count = pd.concat([pd.DataFrame([z_all_std_ano.time[np.where(wr.wr==i)[0][:]].groupby('time.month').count().values], columns=z_all_std_ano.time.groupby('time.month').count().month) for i in range(0,wr.wr.max().values+1)],
      ignore_index=True)

month_frequency = pd.concat([pd.DataFrame([month_count.loc[i].values / month_count.sum(axis=0).values*100], columns=z_all_std_ano.time.groupby('time.month').count().month) for i in range(0,wr.wr.max().values+1)],
      ignore_index=True)

year_frequency = pd.concat([pd.DataFrame([z_all_std_ano.time[np.where(wr.wr==i)[0][:]].count().values], columns=['year']) for i in range(0,wr.wr.max().values+1)],ignore_index=True)



month_frequency['Year']=year_frequency.year.values/year_frequency.year.values.sum()*100


for i in z_all_std_ano.time.groupby('time.month').count().month.values:
    month_frequency = month_frequency.rename(columns={i: calendar.month_abbr[i]})
    
for i in range(0,wr.wr.max().values+1):
    month_frequency = month_frequency.rename(index={i: 'WR'+str(i)})    

######################Plot results#################



ax = month_frequency.transpose().plot.bar(stacked=True, width=1, edgecolor='grey',colormap='tab20c', figsize=(10,8), fontsize=11)
plt.xlim(-0.5,len(month_frequency.transpose())-.5)
plt.ylim(0,100)
plt.xlabel(None)
plt.ylabel('WR cumulative frequency (%)', fontsize=11)
handles, labels = ax.get_legend_handles_labels()
handles.reverse()
labels.reverse()
ax.legend(handles, labels, bbox_to_anchor=(1, 1), labelspacing=4.3, frameon=False, fontsize=11)

plt.savefig(fig_out)






