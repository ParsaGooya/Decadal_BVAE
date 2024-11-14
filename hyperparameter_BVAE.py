import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import dask
import xarray as xr
from pathlib import Path
import random

import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from models.autoencoder import Autoencoder, Autoencoder_decoupled, Autoencoder_decoder
from models.unet import UNet
from models.cnn import CNN
from losses import WeightedMSE,WeightedMSESignLoss, GlobalLoss, WeightedMSEKLD, WeightedMSESignLossKLD, VAEloss
from data_utils.datahandling import combine_observations
from preprocessing import align_data_and_targets, get_coordinate_indices, create_train_mask, reshape_obs_to_data, config_grid
from preprocessing import AnomaliesScaler_v1, AnomaliesScaler_v2, Detrender, Standardizer, Normalizer, PreprocessingPipeline, Spatialnanremove
from torch_datasets import XArrayDataset
from subregions import subregions
from data_locations import LOC_FORECASTS_fgco2_2pi, LOC_FORECASTS_fgco2_pi, LOC_OBSERVATIONS_fgco2_v2023, LOC_FORECASTS_fgco2_simple, LOC_FORECASTS_fgco2
import gc
# specify data directories
data_dir_forecast = LOC_FORECASTS_fgco2
data_dir_obs = LOC_OBSERVATIONS_fgco2_v2023
unit_change = 60*60*24*365 * 1000 /12 * -1 ## Change units for ESM data to mol m-2 yr-1


def HP_congif(params, data_dir_obs, lead_years):

    if params['BVAE'] is not None:
        assert type(params['BVAE']) == int, 'Input the size of output ensemble as BVAE ...'
    else:
        params['BVAE'] = 1

    if params['BVAE'] is not None:
        params['ensemble_mode'] = 'LE'
        assert params['ensemble_list'] is not None, 'for the BVAE model you need to specify the ensemble size ...'
        assert type(params['BVAE']) == int, 'Input the size of output ensemble as VAE ...'
     
    ### load data
    print("Start training")


    if params['ensemble_list'] is not None: ## PG: calculate the mean if ensemble mean is none
        print("Load forecasts")
        ds_in = xr.open_dataset(data_dir_forecast).sel(ensembles = params['ensemble_list']).load()['fgco2']
        if params['ensemble_mode'] == 'Mean': ##
            ds_in = ds_in.mean('ensembles') ##
        else:
            print(f'Warning: ensemble_mode is {params["ensemble_mode"]}. Training for large ensemble ...')
    else:    ## Load specified members
        print("Load forecasts") 
        ds_in = xr.open_dataset(data_dir_forecast).mean('ensembles').load()['fgco2']
    
    print("Load observations")
    if params['correction']:
        obs_in = combine_observations(data_dir_obs, two_dim=True) # 1961.01 - 2021.12
    else:
        obs_in = ds_in.mean('ensembles')[:,:12].rename({'lead_time' : 'month'})

    obs_in = obs_in.expand_dims('channels', axis=2)

    if 'ensembles' in ds_in.dims: ### PG: add channels dimention to the correct axis based on whether we have ensembles or not
        ds_in = ds_in.expand_dims('channels', axis=3).sortby('ensembles')
    else:
        ds_in = ds_in.expand_dims('channels', axis=2) 

    ds_raw, obs_raw = align_data_and_targets(ds_in, obs_in, lead_years)  # extract valid lead times and usable years

    if 'ensembles' in ds_raw.dims: ## PG: reorder dimensions in you have ensembles
        ds_raw_ensemble_mean = ds_raw.transpose('year','lead_time','ensembles',...)* unit_change
    else:
        ds_raw_ensemble_mean = ds_raw.transpose('year','lead_time',...)* unit_change
    
    obs_raw = reshape_obs_to_data(obs_raw, ds_raw_ensemble_mean, return_xarray=True)

    if not ds_raw_ensemble_mean.year.equals(obs_raw.year):
            
            ds_raw_ensemble_mean = ds_raw_ensemble_mean.sel(year = obs_raw.year)
    #######################################################################################################################################
    nanremover = Spatialnanremove()## PG: Get an instance of the class
    if params['correction']:
        nanremover.fit(ds_raw_ensemble_mean[:,:12,...], obs_raw[:,:12,...]) ## PG:extract the commong grid points between training and obs data
    else:
        nanremover.fit(ds_raw_ensemble_mean[:,:12,...], ds_raw_ensemble_mean[:,:12,...]) ## PG:extract the commong grid points between training and obs data
    ds_raw_ensemble_mean = nanremover.to_map(nanremover.sample(ds_raw_ensemble_mean)) ## PG: flatten and sample training data at those locations
    obs_raw = nanremover.to_map(nanremover.sample(obs_raw)) ## PG: flatten and sample obs data at those locations    
    #######################################################################################################################################
    params['nanremover'] = nanremover
    del ds_raw
    del ds_in, obs_in
    gc.collect()
    return  ds_raw_ensemble_mean, obs_raw, params

def smooth_curve(list, factor = 0.9):
    smoothed_list = []
    for point in list:
        if smoothed_list:
            previous = smoothed_list[-1]
            smoothed_list.append(previous* factor + point * (1- factor))
        else:
            smoothed_list.append(point)
    return smoothed_list



def training_hp(hyperparamater_grid: dict, params:dict, ds_raw_ensemble_mean: XArrayDataset ,obs_raw: XArrayDataset , test_year, lead_time = None, n_runs=1, results_dir=None, numpy_seed=None, torch_seed=None):
    
    if params['remove_ensemble_mean']:
        assert not params['correction'], 'With correction set to True, cannot train with ensemble anomlalies.'
    
    # if not params['non_random_decoder_initialization']:
    #    assert any([all([params['time_features'] is not None, params['append_mode'] != 1]), params['condition_embedding_size'] is not None]), 'For random decoder initializaiton, conditions should be provided to the decoder ...' 
    

    if params['model'] == UNet:
            params["arch"] = '_default'

    assert params['version'] in [0, 1,2,3]
    if params['version'] == 1:

        params['forecast_preprocessing_steps'] = [
        ('anomalies', AnomaliesScaler_v1(axis=0)),
        ('standardize', Standardizer(axis = (0,1,2)))]
        params['observations_preprocessing_steps'] = [
        ('anomalies', AnomaliesScaler_v1(axis=0))  ]

    elif params['version'] == 2:

        params['forecast_preprocessing_steps'] = [
        ('anomalies', AnomaliesScaler_v2(axis=0)),
        ('standardize', Standardizer(axis = (0,1,2)))]
        params['observations_preprocessing_steps'] = [
        ('anomalies', AnomaliesScaler_v2(axis=0))  ]

    elif params['version'] == 3:

        params['forecast_preprocessing_steps'] = [
        ('anomalies', AnomaliesScaler_v1(axis=0)),
        ('standardize', Standardizer(axis = (0,1,2)))]
        params['observations_preprocessing_steps'] = [
        ('anomalies', AnomaliesScaler_v2(axis=0))  ]

    else:
        params['forecast_preprocessing_steps'] = []
        params['observations_preprocessing_steps'] = []

    for key, value in hyperparamater_grid.items():
            params[key] = value 
    if 'hidden_dims' in hyperparamater_grid.keys():
        params['arch'] = None
    if params["arch"] == 3:
        params["hidden_dims"] = [[1500, 720, 360, 180, 90, 30], [90, 180, 360, 720, 1500]]
    if params['arch'] == 2:
        params["hidden_dims"] = [[1500, 720, 360, 180, 90], [180, 360, 720, 1500]]
    if params['arch'] == 1:
        params["hidden_dims"] = [[ 720, 360, 180, 90,30], [90,180, 360, 720]]
    if params['condition_embedding_size'] == 'encoder':
        params['condition_embedding_size'] = params["hidden_dims"][0]

    if params['lr_scheduler']:
        start_factor = params['start_factor']
        end_factor = params['end_factor']
        total_iters = params['total_iters']

    ##### PG: Ensemble members to load 
    ensemble_list = params['ensemble_list']
    ###### PG: Add ensemble features to training features
    ensemble_mode = params['ensemble_mode'] ##
    
    if 'atm_co2' in params['extra_predictors']:
        atm_co2 = xr.open_dataset('/home/rpg002/CMIP6_ssp245_xCO2atm_1982_2029.nc').ssp245
        atm_co2 = reshape_obs_to_data(atm_co2, ds_raw_ensemble_mean, return_xarray=True).rename({'month' : 'lead_time'})
        try:
            extra_predictors = atm_co2.sel(year = ds_raw_ensemble_mean.year).expand_dims('channels', axis=2)
        except:
            raise ValueError("Extra predictors not available at the same years as the predictors.")
    else:
         extra_predictors = None     

    if extra_predictors is not None:
        if params["model"] in [Autoencoder_decoupled, Autoencoder]:
            weights = np.cos(np.ones_like(extra_predictors.lon) * (np.deg2rad(extra_predictors.lat.to_numpy()))[..., None])  # Moved this up
            weights = xr.DataArray(weights, dims = extra_predictors.dims[-2:], name = 'weights').assign_coords({'lat': extra_predictors.lat, 'lon' : extra_predictors.lon}) 
            extra_predictors = (extra_predictors * weights).sum(['lat','lon'])/weights.sum(['lat','lon'])
        else:  
            if not all(['ensembles' not in extra_predictors.dims, 'ensembles' in ds_raw_ensemble_mean.dims]): 
                
                ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean, extra_predictors], dim = 'channels')
            else:
                ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean, extra_predictors.expand_dims(ensembles = ds_raw_ensemble_mean['ensembles'], axis = 2) ], dim = 'channels')
            extra_predictors = None
 
    hyperparam = params['hyperparam']
    reg_scale = params["reg_scale"]
    model = params["model"]
    hidden_dims = params["hidden_dims"]
    time_features = params["time_features"].copy() if params["time_features"] is not None else None
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    batch_size_correction = params['batch_size_correction']
    batch_normalization = params["batch_normalization"]
    dropout_rate = params["dropout_rate"]
    batch_shuffle = params["batch_shuffle"]
    optimizer = params["optimizer"]
    lr = params["lr"]

    forecast_preprocessing_steps = params["forecast_preprocessing_steps"]
    observations_preprocessing_steps = params["observations_preprocessing_steps"]

    loss_region = params["loss_region"]
    subset_dimensions = params["subset_dimensions"]
    Biome = params['Biome']
    size_val_years = params['size_val_years']
    l2_reg = params["L2_reg"]
    conditional_embedding = True if params['condition_embedding_size'] is not None else False

    test_years = test_year

    if n_runs > 1:
        numpy_seed = None
        torch_seed = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    print(f"Start run for test year {test_year}...")
    if params['correction']:
        train_years = ds_raw_ensemble_mean.year[ds_raw_ensemble_mean.year < test_year-size_val_years].to_numpy()
    else:
        train_years = ds_raw_ensemble_mean.year[ds_raw_ensemble_mean.year < test_year].to_numpy()

    n_train = len(train_years)
    train_mask = create_train_mask(ds_raw_ensemble_mean[:n_train,...])
    if not params['correction']:
        train_mask = np.full(train_mask.shape, False, dtype=bool)
        indices = np.random.choice(train_mask.size, int(size_val_years * train_mask.size), replace=False)
        train_mask.flat[indices] = True

    ds_baseline = ds_raw_ensemble_mean[:n_train,...]
    obs_baseline = obs_raw[:n_train,...]

    if 'ensembles' in ds_raw_ensemble_mean.dims: ## PG: Broadcast the mask to the correct shape if you have an ensembles dim.
        preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None,None], ds_baseline.shape)
    else:
        preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None], ds_baseline.shape)
    preprocessing_mask_obs = np.broadcast_to(train_mask[...,None,None,None], obs_baseline.shape)

    if subset_dimensions is not None:
        xmin, xmax, ymin, ymax = get_coordinate_indices(ds_raw_ensemble_mean, subset_dimensions)
        ds = ds_raw_ensemble_mean[..., xmin:xmax+1, ymin:ymax+1]
        obs = obs_raw[..., xmin:xmax+1, ymin:ymax+1]

    if numpy_seed is not None:
        np.random.seed(numpy_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    # Data preprocessing
    
    ds_pipeline = PreprocessingPipeline(forecast_preprocessing_steps).fit(ds_baseline, mask=preprocessing_mask_fct)
    ds = ds_pipeline.transform(ds_raw_ensemble_mean)

    obs_pipeline = PreprocessingPipeline(observations_preprocessing_steps).fit(obs_baseline, mask=preprocessing_mask_obs)
    if 'standardize' in ds_pipeline.steps:
        obs_pipeline.add_fitted_preprocessor(ds_pipeline.get_preprocessors('standardize'), 'standardize')
    obs = obs_pipeline.transform(obs_raw)
    del ds_baseline, obs_baseline, preprocessing_mask_obs, preprocessing_mask_fct
    gc.collect()
    if params['correction']:
        year_max = ds[:n_train + size_val_years+1].year[-1].values
    else:
        year_max = ds[:n_train + 1].year[-1].values

    if params['remove_ensemble_mean']:
            ds_em = ds.mean('ensembles')
            ds = ds - ds_em
            
    # TRAIN MODEL


    ds_train = ds[:n_train,...]
    obs_train = obs[:n_train,...]
    if params['correction']:
        ds_validation = ds[n_train:n_train + size_val_years,...]
        obs_validation = obs[n_train:n_train + size_val_years,...]
    else:
        ds_validation = ds[:n_train,...]
        obs_validation = obs[:n_train,...]

    if params['remove_ensemble_mean']:
        ds_em_train = ds_em.sel(year = ds_train.year)
        del ds_em

    ##### PG: The ocean carbon flux has NaN values over land in both forecast and obs data and these are not necessarily in the excat same grid points. ###
    ##### We need to extract the common grid points where both obs and model data exist. That said, we need to flatten both the training and target data
    ##### I defined a Nanremover class. See preprocessing.py.
        
    weights = np.cos(np.ones_like(ds_train.lon) * (np.deg2rad(ds_train.lat.to_numpy()))[..., None])  # Moved this up
    weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon}) # Create an DataArray to pass to Spatialnanremove() 
    weights_val = weights.copy()

    #### Set weights to 1!!:
    # weights = xr.ones_like(weights)
    ### Increase the weight over some biome :
    if Biome is not None:
        if type(Biome) == dict:
            for ind, sc in Biome.items():
                    weights = weights + (sc-1) * weights.where(biomes == ind).fillna(0)  
        elif type(Biome) == list:
            for ind in Biome:
                    weights = weights + weights.where(biomes == ind).fillna(0) 
        else:
                weights = weights + weights.where(biomes == Biome).fillna(0)
##########################################            
    # nanremover = Spatialnanremove() ## PG: Get an instance of the class
    # nanremover.fit(ds_train[:,:12,...], obs_train[:,:12,...]) ## PG:extract the commong grid points between training and obs data
    nanremover = params['nanremover']
    ds_train = nanremover.sample(ds_train) ## PG: flatten and sample training data at those locations
    obs_train = nanremover.sample(obs_train) ## PG: flatten and sample obs data at those locations
    ########################################################################

    if model in [UNet , CNN]: ## PG: If the model starts with a nn.Conv2d write back the flattened data to maps.

        ds_train = nanremover.to_map(ds_train).fillna(0.0) ## PG: fill NaN values with 0.0 for training
        obs_train = nanremover.to_map(obs_train).fillna(0.0) ## PG: fill NaN values with 0.0 for training

        img_dim = ds_train.shape[-2] * ds_train.shape[-1] 
        if loss_region is not None:
            loss_region_indices, loss_area = get_coordinate_indices(ds_raw_ensemble_mean, loss_region)
        
        else:
            loss_region_indices = None
    
    else: ## PG: If you have a dense first layer keep the data flattened.
        
        weights = nanremover.sample(weights) ## PG: flatten and sample weighs at those locations
        weights_val = nanremover.sample(weights_val)
        img_dim = ds_train.shape[-1] ## PG: The input dim is now the length of the flattened dimention.
        if loss_region is not None:
    
            loss_region_indices = nanremover.extract_indices(subregions[loss_region]) ## PG: We need 1D index list rather than 2D lat:lon.

        else:
            loss_region_indices = None

    weights = weights.values
    weights_val = weights_val.values
    del ds, obs
    gc.collect()
    torch.cuda.empty_cache() 
    torch.cuda.synchronize() 

    if time_features is None:

            add_feature_dim = 0
    else:

            add_feature_dim = len(time_features)
    
    if extra_predictors is not None:
            add_feature_dim = add_feature_dim + len(params['extra_predictors'])



    if model == Autoencoder:
        net = model(img_dim, hidden_dims[0], hidden_dims[1], added_features_dim=add_feature_dim, append_mode=params['append_mode'], batch_normalization=batch_normalization, dropout_rate=dropout_rate, VAE = params['BVAE'], condition_embedding_dims = params['condition_embedding_size'], device = device)
        if params['correction']:
            net_decoder = Autoencoder_decoder(img_dim, hidden_dims[0][-1], hidden_dims[1], added_features_dim=add_feature_dim, append_mode=params['append_mode'], batch_normalization=batch_normalization, dropout_rate=dropout_rate,  device = device)             
    elif model == Autoencoder_decoupled:
        net = model(img_dim, hidden_dims[0], hidden_dims[1], added_features_dim=add_feature_dim, append_mode=params['append_mode'], batch_normalization=batch_normalization, dropout_rate=dropout_rate)


    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = l2_reg)
    if params['lr_scheduler']:
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)

    ## PG: XArrayDataset now needs to know if we are adding ensemble features. The outputs are datasets that are maps or flattened in space depending on the model.
    train_set = XArrayDataset(ds_train, obs_train, mask=train_mask,lead_time_mask = params['lead_time_mask'], extra_predictors=extra_predictors, in_memory=True, lead_time=lead_time, time_features=time_features, aligned = True, year_max=year_max, conditional_embedding = conditional_embedding, cross_member_training = params['cross_member_training']) 
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=batch_shuffle)

    if conditional_embedding:
        if params['remove_ensemble_mean']:
            ds_train_conds = nanremover.sample(ds_em_train).stack(flattened=('year','lead_time')).transpose('flattened',...)[~train_mask.flatten()]
            del ds_em_train
        else:
            ds_train_conds = ds_train.stack(flattened=('year','lead_time')).transpose('flattened',...)[~train_mask.flatten()].mean('ensembles')
        if lead_time is not None:
            ds_train_conds = ds_train_conds.where((ds_train_conds.lead_time >=  (lead_time - 1) * 12 + 1) & (ds_train_conds.lead_time < (lead_time *12 )+1), drop = True)
        ds_train_conds = torch.from_numpy(ds_train_conds.to_numpy())
        
    
    criterion = WeightedMSESignLossKLD(weights=weights, device=device, hyperparam=hyperparam, reduction=params['loss_reduction'], loss_area=loss_region_indices,  scale = reg_scale)
  # criterion = VAEloss(weights = weights , device = device)

    # EVALUATE MODEL
    ##################################################################################################################################
    ds_validation = nanremover.sample(ds_validation, mode = 'Eval')  ## PG: Sample the test data at the common locations
    obs_validation = nanremover.sample(obs_validation)
    if model == UNet:
        ds_validation = nanremover.to_map(ds_validation).fillna(0.0)  ## PG: Write back to map if the model starts with a nn.Conv2D
        obs_validation = nanremover.to_map(obs_validation).fillna(0.0)

    # val_batch_size = int(np.ceil(batch_size/len(ensemble_list)))
    val_batch_size = batch_size
    if params['correction']:
        val_mask = create_train_mask(ds_validation)
    else:
        val_mask = ~train_mask
    validation_set = XArrayDataset(ds_validation, obs_validation, mask=val_mask,lead_time = lead_time, extra_predictors=extra_predictors, lead_time_mask = params['lead_time_mask'], time_features=time_features, in_memory=False, aligned = True, year_max=year_max, BVAE= params['BVAE'], cross_member_training = params['cross_member_training']) 
    dataloader_val = DataLoader(validation_set, batch_size=val_batch_size, shuffle=True)   
 

    criterion_eval = WeightedMSESignLossKLD(weights=weights_val, device=device, hyperparam=1, reduction=params['loss_reduction'], loss_area=loss_region_indices, scale = reg_scale)
    criterion_eval_mean = WeightedMSE(weights=weights_val, device='cpu', hyperparam=1, reduction=params['loss_reduction'], loss_area=loss_region_indices)
    ##################################################################################################################################
    del ds_train, obs_train, ds_validation, obs_validation
    gc.collect()
    epoch_loss_train = []
    epoch_loss_train_KL = []
    epoch_loss_train_MSE = []
    epoch_loss_val = []
    epoch_loss_val_KL = []
    epoch_loss_val_ens_mean = []
    epoch_loss_val_mean = []

    for epoch in tqdm.tqdm(range(epochs)):

        net.train()
        batch_loss = 0
        batch_loss_MSE = 0
        batch_loss_KLD = 0
        if type(params['beta']) == dict:
            beta = params['beta']['start'] + (params['beta']['end'] - params['beta']['start']) * min(epoch/params['beta']['epochs'], 1)
        else:
            beta = params['beta']

        if all([params['cross_member_training'], epoch>=1]):
                train_set, shuffle_idx = train_set.shuffle_target_ensembles( return_shuffled_idx = True)
                dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
                validation_set = validation_set.shuffle_target_ensembles(shuffle_idx = shuffle_idx)  
                dataloader_val = DataLoader(validation_set, batch_size=val_batch_size, shuffle=True)   

        for batch, (x, y) in enumerate(dataloader):
            if conditional_embedding:
                cond_idx = x[-1]
                x = [x[i] for i in range(len(x) - 1)] if len(x) >2 else x[0]
                cond = ds_train_conds[cond_idx].float().to(device)
            else:
                cond = None
            if (type(x) == list) or (type(x) == tuple):
                x = (x[0].to(device), x[1].to(device))
            else:
                x = x.to(device)
            if (type(y) == list) or (type(y) == tuple):
                y, m = (y[0].to(device), y[1].to(device))
            else:
                y = y.to(device)
                m  = None
            optimizer.zero_grad()
            if not params['cross_member_training']:
                y = x[0].to(device) if (type(x) == list) or (type(x) == tuple) else x.to(device)
                adjusted_forecast, mu, log_var = net(x, condition = cond, sample_size = params['training_sample_size'])
                adjusted_forecast = adjusted_forecast.mean(0)
                # adjusted_forecast = torch.flatten(adjusted_forecast, start_dim= 0 , end_dim=1)
            
            else:
                ### first generate mu and sigma then sample from z
                pass
            loss, MSEL, KLDL = criterion(adjusted_forecast, y ,mu, log_var, beta = beta, mask = m, return_ind_loss=True, print_loss = True)

            batch_loss += loss.item()
            batch_loss_MSE += MSEL.item()
            batch_loss_KLD += KLDL.item()
            loss.backward()
            optimizer.step()
            
        epoch_loss_train.append(batch_loss / len(dataloader))
        epoch_loss_train_KL.append(batch_loss_KLD / len(dataloader))
        epoch_loss_train_MSE.append(batch_loss_MSE / len(dataloader))

        if params['lr_scheduler']:
            scheduler.step()
        net.eval()
        val_loss = 0
        val_loss_KL = 0
        val_loss_ens_mean = 0
        val_loss_mean = 0
        
        for batch, (x, target) in enumerate(dataloader_val):         
            with torch.no_grad():            
                if (type(x) == list) or (type(x) == tuple):
                    test_raw = (x[0].to(device), x[1].to(device))
                    test_raw = (torch.flatten(test_raw[0], start_dim= 0 , end_dim=1), torch.flatten( test_raw[1].unsqueeze(1).expand(test_raw[1].shape[0], test_raw[0].shape[1], *test_raw[1].shape[1:]), start_dim= 0 , end_dim=1))
                else:
                    test_raw = x.to(device)
                    test_raw = torch.flatten(test_raw, start_dim= 0 , end_dim=1)
                if (type(target) == list) or (type(target) == tuple):
                    test_obs, m = (target[0].to(device), target[1].to(device))
                else:
                    test_obs = target.to(device)
                    m = None
                if conditional_embedding:
                        cond = x[0].mean(axis = -3).to(device) if (type(x) == list) or (type(x) == tuple) else x.mean(axis = -3).to(device)        
                else:
                        cond = None
                if params['cross_member_training']:
                    test_obs = torch.flatten(test_obs, start_dim= 0 , end_dim=1)
                else:
                    test_obs = test_raw[0]  if (type(x) == list) or (type(x) == tuple) else test_raw

                # if not params['non_random_decoder_initialization']:
                #     sample_size = test_raw[0].shape[0] if (type(test_raw) == list) or (type(test_raw) == tuple) else test_raw.shape[0]
                #     z =  Normal(torch.zeros(sample_size, net.latent_size), torch.ones((sample_size, net.latent_size))).rsample(sample_shape=(params['BVAE'],)).to(device)
                #     if all([params['time_features'] is not None, params['append_mode'] != 1]):
                #         z = torch.cat([z, test_raw[1].expand((1, *test_raw[1].shape))], dim=-1)
                #     if conditional_embedding:
                #         cond_embedded = net.embedding(cond.to(device).flatten(start_dim=1))
                #         cond_embedded = cond_embedded.unsqueeze(-2).expand(cond_embedded.shape[0], int(sample_size/cond_embedded.shape[0]), net.embedding_size)
                #         cond_embedded = torch.flatten(cond_embedded, start_dim = 0, end_dim = 1)
                #         cond_embedded = cond_embedded.unsqueeze(0).expand((z.shape[0], z.shape[1], net.embedding_size))
                #         z = torch.cat([z, cond_embedded], dim=-1)     

                #         out_shape = z.shape
                #         out = net.decoder(torch.flatten(z, start_dim = 0, end_dim = 1))
                #         out = torch.unflatten(out, dim = 0 , sizes = out_shape[0:2])
    
                #         test_adjusted =   out.view((1, *test_raw[0].shape) ) if (type(x) == list) or (type(x) == tuple) else out.view((1, *test_raw.shape) )

                # else:
                test_adjusted, mu_val, log_var_val  =  net(test_raw, condition = cond, sample_size =  params['training_sample_size'])
                test_adjusted = test_adjusted.mean(0)
                # test_adjusted = torch.flatten(test_adjusted, start_dim= 0 , end_dim=1)

                loss_, loss_mean, loss_KL = criterion_eval(test_adjusted, test_obs ,mu_val, log_var_val, beta = beta, mask = m, return_ind_loss=True, print_loss = False)

                val_loss += loss_.item()
                val_loss_KL += loss_KL.item()
                val_loss_mean += loss_mean.item()
                
                test_obs = torch.mean(x[0], 1)  if (type(x) == list) or (type(x) == tuple) else  torch.mean(x, 1)
                sample_size = x[0].shape[0] if (type(x) == list) or (type(x) == tuple) else x.shape[0]
                if params['non_random_decoder_initialization']:
                    ad, mu, log_var = net(test_raw, condition = cond, sample_size = 1)
                    mu = torch.unflatten(mu, dim = 0, sizes = (-1,len(ensemble_list)))
                    var = torch.exp(torch.unflatten(log_var, dim = 0, sizes = (-1,len(ensemble_list)))) + 1e-4
                    z =  Normal(torch.mean(mu, -2), torch.std(mu, -2)).rsample(sample_shape=(params['BVAE'] * len(ensemble_list),)).to(device)
                    # z = torch.transpose(z, 1,0)
                else:
                    z =  Normal(torch.zeros((sample_size, net.latent_size)), torch.ones((sample_size, net.latent_size))).rsample(sample_shape=(params['BVAE'] * len(ensemble_list),)).to(device)

                if all([params['time_features'] is not None, params['append_mode'] != 1]):
                        z = torch.cat([z, test_raw[1].unsqueeze(0).expand((params['BVAE'], *test_raw[1].shape))], dim=-1)
                if conditional_embedding:
                        cond_embedded = net.embedding(torch.flatten(cond, start_dim = 1).to(device))
                        cond_embedded = cond_embedded.unsqueeze(-3).expand(params['BVAE'] * len(ensemble_list), *cond_embedded.shape)
                        z = torch.cat([z, cond_embedded], dim=-1)
                z = torch.transpose(z,1,0)            
                out_shape = z.shape
                test_adjusted = net.decoder(torch.flatten(z, start_dim = 0, end_dim = 1))
                test_adjusted = torch.unflatten(test_adjusted, dim = 0 , sizes = out_shape[0:2]).mean(-2).unsqueeze(-2)
                loss_val_ensemble = criterion_eval_mean(test_adjusted.detach().to('cpu'), test_obs)

                val_loss_ens_mean += loss_val_ensemble.item()

        epoch_loss_val.append(val_loss / len(dataloader_val))
        epoch_loss_val_ens_mean.append(val_loss_ens_mean / len(dataloader_val))
        epoch_loss_val_KL.append(val_loss_KL / len(dataloader_val))
        epoch_loss_val_mean.append(val_loss_mean / len(dataloader_val))
        # Store results as NetCDF            
    del adjusted_forecast, y, train_set,validation_set, dataloader, dataloader_val, x , m, test_raw, test_obs,  target, test_adjusted, net
    gc.collect()
    torch.cuda.empty_cache() 
    torch.cuda.synchronize() 
    epoch_loss_val = smooth_curve(epoch_loss_val)
    epoch_loss_val_KL = smooth_curve(epoch_loss_val_KL)
    epoch_loss_val_mean = smooth_curve(epoch_loss_val_mean)
    epoch_loss_val_ens_mean = smooth_curve(epoch_loss_val_ens_mean)

    plt.figure(figsize = (8,10))
    plt.subplot(2,1,1)
    plt.plot(np.arange(2,epochs+1), epoch_loss_train[1:], label = 'Train')
    plt.plot(np.arange(2,epochs+1), epoch_loss_val[1:], label = 'Validation')
    plt.title(f'{hyperparamater_grid}')
    plt.legend(loc='upper right')
    plt.ylabel('MSE + KLD')
    # plt.twinx()
    plt.plot(np.arange(2,epochs+1), epoch_loss_train_MSE[1:], label = 'Train MSE', alpha = 0.5)
    plt.plot(np.arange(2,epochs+1), epoch_loss_val_mean[1:], label = 'Validation MSE', alpha = 0.5)
    # plt.ylabel('MSE')
    # plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    plt.subplot(2,1,2)
    plt.plot(np.arange(2,epochs+1), epoch_loss_train_KL[1:], label = 'Train KLD')
    plt.plot(np.arange(2,epochs+1), epoch_loss_val_KL[1:], label = 'Validation KLD')
    plt.legend(loc='upper right')
    plt.ylabel('KLD')
    plt.twinx()
    plt.plot(np.arange(2,epochs+1), epoch_loss_val_ens_mean[1:], label = 'Validation Ensemble mean MSE',  color = 'k', alpha = 0.5)
    plt.ylabel('MSE')
    plt.legend(loc='lower right')
    plt.xlabel('Epoch') 
    plt.grid()
    plt.show()
    plt.savefig(results_dir+f'/val-train_loss_1982-{test_year}-{hyperparamater_grid}.png')
    plt.close()


    with open(Path(results_dir, "Hyperparameter_training.txt"), 'a') as f:
        f.write(
  
            f"{hyperparamater_grid} ---> val_loss at best epoch: {epoch_loss_val[np.argmin(epoch_loss_val_mean)]} at {np.argmin(epoch_loss_val_mean)+1}  (MSE : {np.min(epoch_loss_val_mean)})\n" +  ## PG: The scale to be passed to Signloss regularization
            f"-------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n" 
        )
    return np.min(epoch_loss_val_mean), epoch_loss_val, epoch_loss_val_KL, epoch_loss_val_mean, epoch_loss_val_ens_mean, epoch_loss_train

                                 #########         ##########

    
def run_hp_tunning( ds_raw_ensemble_mean: XArrayDataset ,obs_raw: XArrayDataset,  hyperparameterspace: list, params:dict, y_start: int, y_end:int, out_dir_x , lead_time = None, n_runs=1, numpy_seed=None, torch_seed=None ):

    

    val_losses = np.zeros([y_end - y_start + 1 , len(hyperparameterspace), params['epochs']]) ####
    val_losses_KL = np.zeros([y_end - y_start + 1 , len(hyperparameterspace), params['epochs']]) ####
    val_losses_mean = np.zeros([y_end - y_start + 1 , len(hyperparameterspace), params['epochs']]) ####
    val_losses_ens_mean = np.zeros([y_end - y_start + 1 , len(hyperparameterspace), params['epochs']]) ####
    train_losses = np.zeros([y_end - y_start + 1, len(hyperparameterspace), params['epochs']]) ####

    for ind_, test_year in enumerate(range(y_start,y_end+1)):
    
        if len(params["extra_predictors"]) > 0:
            out_dir_x = out_dir_x + '_extra_predictors'

        out_dir    = f'{out_dir_x}/_{test_year}'    
        
        
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        if  params['lr_scheduler']:
            start_factor = params['start_factor']
            end_factor = params['end_factor']
            total_iters = params['total_iters']

        with open(Path(out_dir, "Hyperparameter_training.txt"), 'w') as f:
            f.write(
                f"model\t{params['model']}\n" +
                "default set-up:\n" + 
                f"hidden_dims\t{params['hidden_dims']}\n" +
                f"time_features\t{params['time_features']}\n" +
                f"extra_predictors\t{params['extra_predictors']}\n" +
                f"append_mode\t{params['append_mode']}\n" +
                f"ensemble_list\t{params['ensemble_list']}\n" + ## PG: Ensemble list
                f"batch_normalization\t{params['batch_normalization']}\n" +
                f"dropout_rate\t{params['dropout_rate']}\n" +
                f"lr\t{params['lr']}\n" +
                f"lr_scheduler\t{params['lr_scheduler']}: {start_factor} --> {end_factor} in {total_iters} epochs\n" + 
                f"L2_reg\t{params['L2_reg']}\n" +
                f"Lead time mask\t{params['lead_time_mask']}\n" +
                f"BVAE_default\t1\n" +
                f"training_sample_size_default\t{params['training_sample_size']}\n" +
                f"loss_reduction\t{params['loss_reduction']}\n"  + 
                f"Correction\t{params['correction']}\n" +
                f"batch_size_correction\t{params['batch_size_correction']}\n\n\n"
                ' ----------------------------------------------------------------------------------\n'
            )
        

        
        losses = np.zeros(len(hyperparameterspace))
 

       
        for ind, dic in enumerate(hyperparameterspace):
            try:
                print(f'Training for {dic}')
                losses[ind], val_losses[ind_, ind, :], val_losses_KL[ind_, ind, :], val_losses_mean[ind_, ind, :],  val_losses_ens_mean[ind_, ind, :], train_losses[ind_, ind, :] = training_hp( ds_raw_ensemble_mean =  ds_raw_ensemble_mean,obs_raw = obs_raw , hyperparamater_grid= dic, lead_time = lead_time , params = params ,test_year=test_year, n_runs=n_runs, results_dir=out_dir, numpy_seed=numpy_seed, torch_seed=torch_seed)
            except:
                with open(Path(out_dir, "Hyperparameter_training.txt"), 'a') as f:
                    f.write(
                    f"{dic} ---> Non trainable! \n" +  ## PG: The scale to be passed to Signloss regularization
                    f"-------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n" 
                        ) 

        
        with open(Path(out_dir, "Hyperparameter_training.txt"), 'a') as f:
            f.write(
    
                f"Best MSE: {min(losses)} --> {hyperparameterspace[np.argmin(losses)]} \n" +  ## PG: The scale to be passed to Signloss regularization
                f"--------------------------------------------------------------------------------------------------------\n" 
            )

        print(f"Best loss: {min(losses)} --> {hyperparameterspace[np.argmin(losses)]}")
        print(f'Output dir: {out_dir}')
        print('Training done.')

    coords = []
    for item in hyperparameterspace:
        coords.append(str(tuple(item.values())))

    ds_val = xr.DataArray(val_losses, dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
    ds_val.attrs['hyperparameters'] = list(config_dict.keys())
    ds_val.to_netcdf(out_dir_x + '/validation_losses.nc')

    ds_val_KL = xr.DataArray(val_losses_KL, dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
    ds_val_KL.attrs['hyperparameters'] = list(config_dict.keys())
    ds_val_KL.to_netcdf(out_dir_x + '/validation_losses_KL.nc')

    ds_val_mean = xr.DataArray(val_losses_mean, dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
    ds_val_mean.attrs['hyperparameters'] = list(config_dict.keys())
    ds_val_mean.to_netcdf(out_dir_x + '/validation_losses_mean.nc')

    ds_val_ens_mean = xr.DataArray(val_losses_ens_mean, dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
    ds_val_ens_mean.attrs['hyperparameters'] = list(config_dict.keys())
    ds_val_ens_mean.to_netcdf(out_dir_x + '/validation_losses_ens_mean.nc')

    ds_train = xr.DataArray(train_losses, dims = ['test_years', 'hyperparameters','epochs'], name = 'Train_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1), 'hyperparameters': coords})
    ds_train.attrs['hyperparameters'] = list(config_dict.keys())
    ds_train.to_netcdf(out_dir_x + '/train_losses.nc')

    

    

if __name__ == "__main__":

    fake_data = 'pi'
    # test_year =  2019 # last n years to test consecutively
    lead_years = 1
    n_runs = 1  # number of training runs
    lead_time = None
    params = {
        "model": Autoencoder,
        "hidden_dims":  [],# [3600, 1800, 900, 1800, 3600],
        "time_features":  None, #['month_sin', 'month_cos', 'lead_time'],
        "extra_predictors" : [],
        'ensemble_list' : None, ## PG
        'ensemble_mode' : 'LE',
        "epochs": 150,
        "batch_size": 50,
        "reg_scale" : 0,
        'beta' : 1,
        'hyperparam' : 1,
        "batch_normalization": False,
        "dropout_rate": 0,
        "append_mode": 1,
        "optimizer": torch.optim.Adam,
        "lr": 0.0001,
        "batch_shuffle" : True,
        "loss_region": None,
        "subset_dimensions": None,
        "lead_time_mask" : None,
        "L2_reg" : 0,
        'lr_scheduler' : True,
        'BVAE' : 10,
        'training_sample_size' : 1,
        'non_random_decoder_initialization' : True,
        'condition_embedding_size' : None,
        'cross_member_training' : False,
        'remove_ensemble_mean' : False,
        'correction' : False,
        'batch_size_correction' : 64,
        'loss_reduction' : 'sum' # mean or sum
    }


    if params["model"] not in [Autoencoder, Autoencoder_decoupled]:
        params["append_mode"] = None


    params['ensemble_list'] = np.arange(1,21)#[f'r{e}i1p2f1' for e in range(1,21,1)] ## PG

    biomes = xr.open_dataset('/home/rpg002/fgco2_decadal_forecast_adjustment/Time_Varying_Biomes.nc').MeanBiomes.transpose()
    params['Biome'] = None

    if fake_data is not None:
        if fake_data == '2pi':
            data_dir_forecast = LOC_FORECASTS_fgco2_2pi
        if fake_data == 'pi':
            data_dir_forecast = LOC_FORECASTS_fgco2_pi
        else:
            data_dir_forecast = LOC_FORECASTS_fgco2_simple
        unit_change = 1

    #######################################  Don't touch the following #######################################################

    ds_raw_ensemble_mean, obs_raw, params = HP_congif(params, data_dir_obs, lead_years)


    ################################################ specifics #################################################################

    y_start = 2014
    y_end = 2014
    # y_end = ds_raw_ensemble_mean.year[-1].values +1 
    params['size_val_years'] = 0.005

    # config_dict = {'L2_reg' : [0.00001, 0.0001] , 'dropout_rate' : [0.2, 0.1], 'batch_size' :[8,16,64] }
    # config_dict = { 'hidden_dims' : [ [[1500, 1500,1500,1500, 2], [1500,1500, 1500, 1500]], [[1500, 1500,1500,1500,1500, 2], [1500,1500,1500, 1500, 1500]],
    #                                  [[1500, 1500,1500,1500,1500,1500, 2], [1500,1500,1500, 1500, 1500,1500]],  [[1500,1500,1500, 1500,1500,1500,1500, 2], [1500,1500,1500,1500,1500, 1500, 1500]]],
    #                'beta' : [1,2, 5],
    #                'training_sample_size' : [1,100],
    #                }

    config_dict = { 'hidden_dims' : [ [[1500, 1500,1500,1500,1500,1500,1500,1500, 2], [1500,1500,1500,1500,1500,1500, 1500, 1500]],  [[1500,1500,1500, 1500,1500,1500,1500,1500, 1500,2], [1500,1500,1500, 1500,1500,1500,1500,1500, 1500]]],
                   'beta' : [0.25,  0.01,0.005],
                    'batch_size' : [100,200],
                    'drop_out' : [0,0.2]}

    # config_dict = { 'hidden_dims' : [ [[1500, 1500,1500,1500,1500,1500, 2], [1500,1500,1500,1500, 1500, 1500]], [[750, 750,750,750,750,750, 2], [750,750,750, 750, 750,750]], [[750, 750,750,750,750, 2], [750,750,750, 750, 750]]],
    #                'beta' : [1,1.5],
    #                'training_sample_size' : [1,100]
    #                }
    hyperparameterspace = config_grid(config_dict).full_grid()

    params['interp_nan'] = None #'lat' or 'lon'
    params['version'] = 0  ## 1,2,3
    params['arch'] = None

    ##############################################################################################################################
    
    out_dir_x  = f'/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/fgco2_ems/SOM-FFN/results/{params["model"].__name__}/run_set_8_toy/Model_tunning/arch{params["arch"]}/'
    out_dir_x = out_dir_x + f'NV{params["size_val_years"]}_batch_lr_reg_tunning_v{params["version"]}'

    
    if params['lr_scheduler']:
        out_dir_x = out_dir_x + '_lr_scheduler'
        params['start_factor'] = 1.0
        params['end_factor'] = 0.1
        params['total_iters'] = 50
    
    if type(params['beta']) == dict:
        out_dir_x = out_dir_x + '_Bannealing'


    if any([all([params['time_features'] is not None, params['append_mode'] != 1]), params['condition_embedding_size'] is not None]):
        if params['condition_embedding_size'] is not None:
            out_dir_x = out_dir_x + f'_cEBVAE'
        else:
            out_dir_x = out_dir_x + f'_cBVAE'
    else:
        out_dir_x = out_dir_x + f'_BVAE'
    

    if params['non_random_decoder_initialization']:
        out_dir_x = out_dir_x + f'_ValNnRandDecodInit'

    if params['cross_member_training'] : 
        out_dir_x = out_dir_x + f'_CrMmbrTr'

    if params['remove_ensemble_mean'] : 
        out_dir_x = out_dir_x + f'_RmEnsMn'

    if params['correction']:
        out_dir_x = out_dir_x + f'_Correction'

    if params['loss_reduction'] == 'sum':
        out_dir_x = out_dir_x + f'_MSESUM'
    
    if lead_time is not None:
        out_dir_x = out_dir_x + f'_LY{lead_time}'
    
    if fake_data is not None:
        if fake_data == '2pi':
            out_dir_x = out_dir_x + f'_fake_data_2pi' 
        elif fake_data == 'pi':
            out_dir_x = out_dir_x + f'_fake_data_pi_test' 
        else:
            out_dir_x = out_dir_x + f'_fake_data_simple' 

    # out_dir_x = out_dir_x + f'_no_seed'
    out_dir_x = out_dir_x + f"_TSE{len(params['ensemble_list'])}" 
    
    run_hp_tunning(ds_raw_ensemble_mean = ds_raw_ensemble_mean ,obs_raw = obs_raw,  hyperparameterspace = hyperparameterspace, params = params, y_start = y_start ,y_end = y_end, lead_time = lead_time, out_dir_x = out_dir_x, n_runs=1, numpy_seed=1, torch_seed=1 )