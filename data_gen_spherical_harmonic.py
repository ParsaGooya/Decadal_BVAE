import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import dask
import xarray as xr
from pathlib import Path
from data_utils.datahandling import combine_observations
import pyshtools as pysh

R_earth_km = 6371


def normal_samples_within_bounds(mean, cov, num_samples, bounds=(-1, 1)):
    samples = []
    while len(samples) < num_samples:
        # Generate a sample from a multivariate normal distribution
        sample = np.random.multivariate_normal(mean, cov)
        # Check if the sample lies within the specified bounds
        if np.all(sample >= bounds[0]) and np.all(sample <= bounds[1]):
            samples.append(sample)
    return np.array(samples)

def uniform_samples_with_exceptions(num_samples, range = (0,1), exceptions = [], no_replacement = True):
    samples = []
    while len(samples) < num_samples:
        # Generate a sample from a multivariate normal distribution
        sample = int(np.random.uniform(*range))
                # Check if the sample lies within the specified bounds
        if no_replacement:
            if sample not in samples:
                if sample not in exceptions:
                    samples.append(sample)
        else:
            if sample not in exceptions:
                    samples.append(sample)
    return np.array(samples)


def sample_gen(time, freq_coeffs, harmonic_wavenumbers, harmonic_m):
    assert len(harmonic_wavenumbers) == len(freq_coeffs) == len(harmonic_m)
    # s = normal_samples_within_bounds(np.zeros(len(freq_coeffs)), np.identity(len(freq_coeffs)), 1, bounds=(-np.inf, np.inf)).squeeze()
    s = np.random.uniform(-1,+1,(2))
    ds_ls = []
    for t in time:
        sin_phase =  np.sin((t * 2 * np.pi / freq_coeffs) +  np.pi * s)
        combined_coeffs = pysh.SHCoeffs.from_zeros(90)
        for ind, hv in enumerate(harmonic_wavenumbers):
            
            coeffs = pysh.SHCoeffs.from_zeros(90)
            coeffs.coeffs[0, hv, harmonic_m[ind]] = sin_phase[ind] 
            combined_coeffs += coeffs

        combined_grid = combined_coeffs.expand(grid='DH')
        ds_ls.append(xr.DataArray(
                                combined_grid.data,
                                dims=["lat", "lon"],  # Define the dimensions
                                coords={"lat": combined_grid.lats() , "lon": combined_grid.lons()},
                                name = 'fgco2'))
    return xr.concat(ds_ls, dim = 'time')

    # cos_phase = np.cos((time[:,None] * 2 * np.pi / freq_coeffs[None,])[None,] + 2*np.pi * s) 
    
def rewrite_monthly(ds):
    return xr.concat([ds[i:i+12,].rename({'time':'lead_time'}).assign_coords(lead_time = np.arange(1,13)) for i in range(0,len(ds.time),12)], dim = 'year')  



# LOC_FORECASTS_fgco2 = '/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/data/fgco2/simulation'
# latent_dim = 2
# num_samples = 50
# freq_coeffs = uniform_samples_with_exceptions(latent_dim, (12,120), exceptions  = [0], no_replacement=True)
# harmonic_wavenumbers = uniform_samples_with_exceptions(latent_dim, (1,10), exceptions  = [0], no_replacement=True) 
# harmonic_m = uniform_samples_with_exceptions(latent_dim, (0,3), no_replacement=True)


# print(f'random time scales at {freq_coeffs} months')
# print(f'Spherical harmonic wavenumbers {harmonic_wavenumbers} deg')
# print(f'Spherical harmonic order m {harmonic_m}')

# ds_in = xr.open_mfdataset(str(Path(LOC_FORECASTS_fgco2, "*.nc")), combine='nested', concat_dim='year').mean('ensembles').load()['fgco2'].sel(year = slice(1950,None))
# time = np.arange(len(ds_in.year) * 12)
# years = ds_in.year.values
# del ds_in
# ls = []
# for member in tqdm(range(num_samples)):
#     ds = sample_gen(time, freq_coeffs, harmonic_wavenumbers, harmonic_m) 
#     # ds = ds.interp(lat = np.arange(-89.5,90,1), lon = np.arange(0.5,360,1))
#     ls.append(rewrite_monthly(ds).assign_coords(year = years))

# fake_ds = xr.concat(ls, dim = 'ensembles').assign_coords(ensembles = np.arange(num_samples)).transpose('year','ensembles','lead_time',...)
# fake_ds.attrs['time_scales'] = freq_coeffs
# fake_ds.attrs['Spherical_harmonic_wavenumbers'] =  harmonic_wavenumbers 
# fake_ds.attrs['Spherical_harmonic_spatial_scale'] =  2 * np.pi * R_earth_km / harmonic_wavenumbers 
# fake_ds.attrs['Spherical_harmonic_order'] = harmonic_m
# fake_ds.attrs['units'] = 'month, deg, km'
# fake_ds.to_netcdf(LOC_FORECASTS_fgco2 + '/fake_test/fake_historical_ensembles_19501-201412_1x1_LE_pi.nc')



ds_fake = xr.open_dataset('/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/data/fgco2/simulation/fake_test/fake_historical_ensembles_19501-201412_1x1_LE_pi.nc')['fgco2']

LOC_FORECASTS_fgco2 = '/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/data/fgco2/simulation'
latent_dim = 2
num_samples = 1000
freq_coeffs = ds_fake.attrs['time_scales']
harmonic_wavenumbers = ds_fake.attrs['Spherical_harmonic_wavenumbers']
harmonic_m = ds_fake.attrs['Spherical_harmonic_order'] 
print(f'random time scales at {freq_coeffs} months')
print(f'Spherical harmonic wavenumbers {harmonic_wavenumbers} deg')
print(f'Spherical harmonic order m {harmonic_m}')

ds_in = xr.open_mfdataset(str(Path(LOC_FORECASTS_fgco2, "*.nc")), combine='nested', concat_dim='year').mean('ensembles').load()['fgco2'].sel(year = slice(1950,None))
time = np.arange(len(ds_in.year) * 12)[-24:]
years = ds_in.year.values[-2:]
del ds_in, ds_fake

ls = []
for member in tqdm(range(num_samples)):
    ds = sample_gen(time, freq_coeffs, harmonic_wavenumbers, harmonic_m) 
    # ds = ds.interp(lat = np.arange(-89.5,90,1), lon = np.arange(0.5,360,1))
    ls.append(rewrite_monthly(ds).assign_coords(year = years))

fake_ds = xr.concat(ls, dim = 'ensembles').assign_coords(ensembles = np.arange(num_samples)).transpose('year','ensembles','lead_time',...)
fake_ds.attrs['time_scales'] = freq_coeffs
fake_ds.attrs['Spherical_harmonic_wavenumbers'] =  harmonic_wavenumbers 
fake_ds.attrs['Spherical_harmonic_spatial_scale'] =  2 * np.pi * R_earth_km / harmonic_wavenumbers 
fake_ds.attrs['Spherical_harmonic_order'] = harmonic_m
fake_ds.attrs['units'] = 'month, deg, km'
fake_ds.assign_coords(ensembles = np.arange(num_samples)).to_netcdf(LOC_FORECASTS_fgco2 + '/fake_test/fake_historical_ensembles_2013-2014_1x1_LE_pi.nc')