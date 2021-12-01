# RPGV
How to Distribute New Solar Systems in Europe to Reduce Power Generation Variability (RPGV)

## Input Data

Different source data are needed to run the scripts and recreate the results and plots. Here you can find an overview of these data with a short description of what was used within this work and where you can download it.

### ERA5
We use [ERA5 hourly data on pressure levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form) from Januray 1979 to June 2020. The geographical area is 80°W to 40°E, 30°N to 90°N. The data are downloaded per season and are defined in the first scripts '1_gph-daily-mean-calc.py' the following:
- file1 = data_folder / 'gph-djf-all.nc'
- file2 = data_folder / 'gph-mam-all.nc'
- file3 = data_folder / 'gph-jja-all.nc'
- file4 = data_folder / 'gph-son-all.nc'


### Renewables.ninja
Country-level PV power generation is taken from [Renewables.ninja](https://www.renewables.ninja/downloads). The explicit used dataset can be found [here](https://www.renewables.ninja/static/downloads/ninja_europe_pv_v1.1.zip). We use European country-specific capacity factors based on the reanalyse dataset MERRA-2 covering 1985-2016.
The dataset is first used in the script '7_wr-ninja-combi.py':
- filename = data_folder / 'ninja/ninja_europe_pv_v1.1/ninja_pv_europe_v1.1_merra2.csv'

### Installed PV capacities
To compute actual national PV power generation from current capacity factors, we use installed capacities from the [International Renewable Energy Agency](https://irena.org/publications/2020/Mar/Renewable-Capacity-Statistics-2020). Since these numbers are listed in a PDF we provide the used/created csv file in the folder sources with the name ['installed_capacities_IRENA.csv'](https://github.com/dmuehlemann/RPGV/blob/master/sources/installed_capacities_IRENA.csv).
The dataset is used in the scripts '10_2030-all-scenarios.py' and '11_2050-all-scenarios.py':
- ic_file = data_folder / 'source/installed_capacities_IRENA.csv'


### NECPs
To assess future configurations, we use the [National Energy and Climate Plans (NECPs)](https://ec.europa.eu/energy/topics/energy-strategy/national-energy-climate-plans_en) in which countries define capacity targets until 2030. The actual source to get the planed installed capacities are taken from [SolarPower Europe](https://www.solarpowereurope.org/solar-map-of-eu-countries/) where the NECPs are nicely summarized in an interactive map. When NECPs are not available we consider individual national plans or, as a last resort, apply the average PV installed capacity growth rate until the year 2030 from all EU countries to the currently installed PV capacities. The used dataset can be found in the folder sources with the name ['planned-IC-2030.xlsx'](https://github.com/dmuehlemann/RPGV/blob/master/sources/planned-IC-2030.xlsx).
The dataset is used in the scripts '10_2030-all-scenarios.py' and '11_2050-all-scenarios.py':
- ic_2030 = data_folder / 'source/planned-IC-2030.xlsx'


### Electricity consumption data
We use hourly electricity consumption data from  [Open Power System Data](https://doi.org/10.25832/time_series/2020-10-06) and fill gaps with data from the [statistical office of the European Union](https://ec.europa.eu/eurostat/databrowser/view/nrg_cb_e/default/table?lang=en). 
The datasets are used in the scripts '10_2030-all-scenarios.py' and '11_2050-all-scenarios.py':
- country_load_file = data_folder / 'source/opsd-time_series-2020-10-06/time_series_60min_singleindex.csv'
- eurostat_country_load_file = data_folder / 'source/eurostat_load.xlsx'

### Installed capacity potential
The upper bound which is used in the linear least-square problems is always set to the roof-top mounted PV potential per country. The data for the roof-top mounted PV potential is taken by [Tröndle et al. (2019)](https://doi.org/10.1016/j.esr.2019.100388).
The dataset is used in the scripts '10_2030-all-scenarios.py' and '11_2050-all-scenarios.py':
'''
with open(data_folder / 'source/IC-potential.yaml') as file:
    ic_pot_file = yaml.safe_load(file)
'''

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


