# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:35:33 2020

@author: Dirk
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:33:08 2020

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
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import lsq_linear
import itertools as it
from scipy.optimize import linprog


######################Load Datasets#################
######################Load Datasets#################

data_folder = Path("../data/")


file_ninja = data_folder / 'ninja_and_wr_30days_lowpass_2_0-1_short10.nc'
ninja = xr.open_dataset(file_ninja)

ic_file = data_folder / 'source/installed_capacities_IRENA.csv'

ic = pd.read_csv(ic_file, header=0, parse_dates=[0], index_col=0, squeeze=True)




###here no Brexit ;-)
ninja = ninja.rename_vars({'GB':'UK'})
ic = ic.rename(columns={'GB':'UK'})
ic = ic.to_xarray()

####################################################


#########Create Testdata##########################

ninja_season= []
#testdata delta_cf per weather regime and season
mean_season_DE = ninja.DE.groupby('time.season').mean()
mean_season_ES = ninja.ES.groupby('time.season').mean()
for i in range(0, int(ninja.wr.max()+1)):
    DE = (ninja.DE.where(ninja.wr==i, drop=True).groupby('time.season').mean()- mean_season_DE)#/ninja.DE.groupby('time.season').mean()
    ES = (ninja.ES.where(ninja.wr==i, drop=True).groupby('time.season').mean()- mean_season_ES)#/ninja.ES.groupby('time.season').mean()
    ninja_season.append(DE.to_dataset().merge(ES.to_dataset()))


tot = ninja.DE.groupby('time.season').mean().to_dataset().merge(ninja.ES.groupby('time.season').mean())
    
#create Testdata IC
ic_temp = ic.DE[9].to_dataset().merge(ic.ES[9])
ic_arr = ic_temp.to_array()



c = np.append([ninja_season[0].sel(season='DJF').to_array().values],[ninja_season[1].sel(season='DJF').to_array().values], axis=0)
c_all = np.append([ninja_season[0].sel(season='DJF').to_array().values],[ninja_season[1].sel(season='DJF').to_array().values], axis=0)
for i in range(2,len(ninja_season)):
    c_all = np.append(c_all,[ninja_season[i].sel(season='DJF').to_array().values], axis=0)





lb =ic_arr.values
ub = ic_arr.values *5
lb_null = ic_arr.values *0
ub_inf = np.array([np.inf, np.inf])

###########TEST


#Test capacity factors
c_test = np.array([[-0.01,  0.02],[0.02, -0.04],[1,1]])


#Test with lsq
b0 = np.array([0,  0, 173163])
b1 = (lb * c_test).sum(axis=0)
b2 = np.array([2566,  946])



res_bounds = lsq_linear(c_test, b0), bounds=(lb, ub))
res_P = lsq_linear(c_test, b1, bounds=(lb_null, ub_inf))
res_P2 = lsq_linear(c_test, b2, bounds=(lb_null, ub_inf))
res_both = lsq_linear(c_test, b1, bounds=(lb, ub))
res_both2 = lsq_linear(c_test, b2, bounds=(lb, ub))


print('With bounds and b = 0: ' + str(res_bounds.x))
print('With bounds from 0 to inf and b = delta_CF * IC: ' + str(res_P.x))
print('With bounds from 0 to inf and b = CF * IC : ' + str(res_P2.x))
print('With bounds and b = delta_CF * IC: ' + str(res_both.x))
print('With bounds and b = CF * IC: ' + str(res_both2.x))



#Comparison with COBYLA 
def fun(x0):
    return abs((c_test * x0).sum())



x1_bounds = (lb[0], ub[0])
x2_bounds = (lb[1], ub[1])
bnds = (x1_bounds, x2_bounds)

cons = ({'type': 'eq', 'fun': lambda x0:  x0*tot.sel(season='DJF').to_array()-tot.sel(season='DJF').to_array() * lb})
cons2 = ({'type': 'ineq', 'fun': lambda x0:  x0*tot.sel(season='DJF').to_array()-tot.sel(season='DJF').to_array() * lb})

x0 =ic_arr.values
res_min = minimize(fun, x0, method='SLSQP', bounds=bnds)
res_min1 = minimize(fun, x0, method='SLSQP', constraints=cons)
res_min2 = minimize(fun, x0, method='SLSQP', constraints=cons2)
res_min3 = minimize(fun, x0, method='SLSQP', constraints=cons, bounds=bnds)
res_min4 = minimize(fun, x0, method='SLSQP', constraints=cons2, bounds=bnds)

print('With bounds and no constraints: ' + str(res_min.x))
print('Without bounds and with constraints (equal production): ' + str(res_min1.x))
print('Without bounds and with constraints (greater production): ' + str(res_min2.x))
print('With bounds and constraints (equal):' + str(res_min3.x))
print('With bounds and constraints (greater):' + str(res_min4.x))

#Comparison with linprog --> wie mach ich das es gegen 0 geht!?

c_test2 = abs(c_test.sum(axis=0))


x0_bounds = (None, None)

res_linprog = linprog(c_test2, bounds=[x1_bounds, x2_bounds])











