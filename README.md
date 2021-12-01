# RPGV
How to Distribute New Solar Systems in Europe to Reduce Power Generation Variability (RPGV)

## Input Data

Different source data are needed to run the scripts and recreate the results and plots. Here you can find an overview of these data with a short description of what was used within this work and where you can download them.

### ERA5
We use [ERA5 hourly data on pressure levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form) from Januray 1979 to June 2020. The geographical area is 80°W to 40°E, 30°N to 90°N. The data are downloaded per season and are defined in the first scripts '1_gph-daily-mean-calc.py' the following:
- file1 = data_folder / 'gph-djf-all.nc'
- file2 = data_folder / 'gph-mam-all.nc'
- file3 = data_folder / 'gph-jja-all.nc'
- file4 = data_folder / 'gph-son-all.nc'


### Renewables.ninja
Country-level PV power generation is taken from [Renewables.ninja](https://www.renewables.ninja/downloads). The explicit used dataset can be found [here](https://www.renewables.ninja/static/downloads/ninja_europe_pv_v1.1.zip). We use European country-specific capacity factors based on the reanalyse dataset MERRA-2 covering 1985-2016.
The dataset is first used in the script '7_wr-ninja-combi.py' the following
filename = data_folder / 'ninja/ninja_europe_pv_v1.1/ninja_pv_europe_v1.1_merra2.csv'





## Figure overview

| Figure | Filename | Creating python script |
|---|---|---|
Figure 1 | approach-overview.jpg | -
Figure 2 | wr_and_cf.tiff | 9_plot-wr-and-cf.py
Figure 3 | 2030_ic-distribution_additional.tiff | 10_2030-all-scenarios.py
Figure 4 | 2030_tot_variability.tiff | 10_2030-all-scenarios.py
Figure 5 | 2050_ic-distribution_additional.tiff | 11_2050-all-scenarios.py
Figure 6 | 2050_tot_variability.tiff | 11_2050-all-scenarios.py
Supplementary material Figure 1 | 2030_ic-distribution_absolut.tiff | 10_2030-all-scenarios.py
Supplementary material Figure 2 | 2030_variability.tiff | 10_2030-all-scenarios.py
Supplementary material Figure 3 | 2050_ic-distribution_absolut.tiff | 11_2050-all-scenarios.py
Supplementary material Figure 4 | 2050_variability.tiff | 11_2050-all-scenarios.py


