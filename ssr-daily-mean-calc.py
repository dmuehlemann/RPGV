# -*- coding: utf-8 -*-
"""
Created on Sat May 16 13:25:51 2020

@author: Dirk
"""

"""
Compute and plot the leading EOF of geopotential height on the 500 hPa
pressure surface over the European/Atlantic sector during winter time.

This example uses the plain numpy interface.

Additional requirements for this example:

    * netCDF4 (http://unidata.github.io/netcdf4-python/)
    * matplotlib (http://matplotlib.org/)
    * cartopy (http://scitools.org.uk/cartopy/)

"""

import time, sys
from datetime import datetime, timedelta
from netCDF4 import Dataset, date2num, num2date
import numpy as np
from pathlib import Path
# import xarray as xr 


# Read geopotential height data using the netCDF4 module. The file contains
# December-February averages of geopotential height at 500 hPa for the
# European/Atlantic domain (80W-40E, 20-90N).

data_folder = Path("../data/")
f_in = data_folder / 'gph-djf-all.nc'
f_out = data_folder / 'gph-mean.nc'
"ncin = Dataset(filename, 'r')"



def dailymean(d):
    #day = 20170102
    #d = datetime.strptime(str(day), '%Y%m%d')
    
    
    time_needed = []
    for i in range(1, 25):
        time_needed.append(d + timedelta(hours = i))
        
        
    with Dataset(f_in) as ds_src:
        var_time = ds_src.variables['time']
        time_avail = num2date(var_time[:], var_time.units,
                calendar = var_time.calendar)
     
        indices = []
        tot=0
        for tm in time_needed:
            tot = tot + 1 
            a = np.where(time_avail == tm)[0]
            if len(a) == 0:
                tot = tot -1
                sys.stderr.write('Error: data is missing/incomplete - %s!\n'
                        % tm.strftime('%Y%m%d %H:%M:%S'))
                #sys.exit(200)
                # tm = tm - timedelta(hours = 1)
                # a = np.where(time_avail == tm)[0]
                # if len(a) == 0:
                #     sys.stderr.write('Error: data is missing/incomplete AGAIN (minus 1 hours did not help)!!! - %s!\n'
                #             % tm.strftime('%Y%m%d %H:%M:%S'))
                
                # else:
                #     print('Use minus 1 hours- %s!\n'
                #             % tm.strftime('%Y%m%d %H:%M:%S'))
                #     indices.append(a[0])
                #sys.exit(200)
            else:
                #print('Found %s' % tm.strftime('%Y%m%d %H:%M:%S'))
                indices.append(a[0])
     
        var_gph = ds_src.variables['z']
        gph_values_set = False
        for idx in indices:
            if not gph_values_set:
                data = var_gph[idx, :, :]
                gph_values_set = True
            else:
                data += var_gph[idx, :, :]
        data = np.true_divide(data, tot)
        print(tot)
        print(data[1,1])
        return data
        
        
def createdata(data):
    with Dataset(f_in) as ds_src:
        var_gph = ds_src.variables['z']
        #var_time = ds_src.variables['time'][:]
        #dates = num2date(var_time, ds_src.variables['time'].units)
        with Dataset(f_out, mode = 'w', format = 'NETCDF3_64BIT_OFFSET') as ds_dest:
            # Dimensions
            for name in ['latitude', 'longitude']:
                dim_src = ds_src.dimensions[name]
                ds_dest.createDimension(name, dim_src.size)
                var_src = ds_src.variables[name]
                var_dest = ds_dest.createVariable(name, var_src.datatype, (name,))
                var_dest[:] = var_src[:]
                var_dest.setncattr('units', var_src.units)
                var_dest.setncattr('long_name', var_src.long_name)
         
            dim_time = len(data[0,0,:])
            ds_dest.createDimension('time', dim_time)
            var = ds_dest.createVariable('time', np.int32, ('time',))
            time_units = 'hours since 1900-01-01 00:00:00'
            time_cal = 'gregorian'
            var[:] = date2num([d[:dim_time]], units = time_units, calendar = time_cal)
            var.setncattr('units', time_units)
            var.setncattr('long_name', 'time')
            var.setncattr('calendar', time_cal)
         
            # Variables
            var = ds_dest.createVariable(var_gph.name, np.double, var_gph.dimensions)
            var[:, :, :] = np.einsum('abc->cab', data)
            var.setncattr('units', var_gph.units)
            var.setncattr('long_name', var_gph.long_name)
         
            # Attributes
            ds_dest.setncattr('Conventions', 'CF-1.6')
            ds_dest.setncattr('history', '%s %s'
                    % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    ' '.join(time.tzname)))
         
            print('Done! Daily mean geopotential height saved in %s' % f_out)
            



# data_gph = Dataset(f_in,'r')
# var_time = data_gph['time']

with Dataset(f_in) as ds_src:
    var_time = ds_src.variables['time'][:]
    dates = num2date(var_time, ds_src.variables['time'].units)



#dates_avail = dates[:].strftime('%Y%m%d%H')
#num_dates = [int(d.strftime('%Y%m%d%H')) for d in dates]
day = [int(d.strftime('%Y%m%d')) for d in dates]
day = list(dict.fromkeys(day))
d = [datetime.strptime(str(d), '%Y%m%d') for d in day]



data0 = dailymean(d[0])
data = data0[..., np.newaxis]
for i in d[1:]:
     datatemp = dailymean(i)
     data = np.append(data, np.atleast_3d(datatemp), axis=2)
     print(i)



createdata(data)


  
     


# dailymean(20170101)

