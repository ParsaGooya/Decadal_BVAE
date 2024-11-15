import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import dask
import xarray as xr
from pathlib import Path
from torch.distributions import Normal
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
# from models.AE_13Nov_2024 import Autoencoder, Autoencoder_decoupled, MAF  ## !!!!!!! for versions older than 11 Nov 2024 use thise line !!!!!!!!!! ##
from models.autoencoder import Autoencoder, Autoencoder_decoupled, MAF  
from models.unet import UNet
from models.cnn import CNN, SCNN
from losses import WeightedMSE, WeightedMSESignLoss, WeightedMSEKLD, WeightedMSESignLossKLD, VAEloss
from data_utils.datahandling import combine_observations
from preprocessing import align_data_and_targets, get_coordinate_indices, create_train_mask, reshape_obs_to_data
from preprocessing import AnomaliesScaler_v1, AnomaliesScaler_v2, Standardizer, PreprocessingPipeline, Spatialnanremove, calculate_climatology
from torch_datasets import XArrayDataset
from subregions import subregions
from data_locations import LOC_FORECASTS_fgco2_2pi, LOC_FORECASTS_fgco2_pi, LOC_OBSERVATIONS_fgco2_v2023, LOC_FORECASTS_fgco2_simple, LOC_FORECASTS_fgco2
import gc
import glob
# specify data directories
data_dir_forecast = LOC_FORECASTS_fgco2
data_dir_obs = LOC_OBSERVATIONS_fgco2_v2023
unit_change = 60*60*24*365 * 1000 /12 * -1 ## Change units for ESM data to mol m-2 yr-1


def predict(params, test_years, lead_years,model_year = None,  results_dir=None, num_stds = 1 ):

   
    if 'CrMmbrTr' in  results_dir:
        params['cross_member_training'] = True 
    else:
        params['cross_member_training'] = False 

    if 'RmEnsMn' in  results_dir:
        params['remove_ensemble_mean'] = True 
    else:
        params['remove_ensemble_mean'] = False 
     
    if 'MSESUM' in  results_dir:
        params['loss_reduction'] = 'sum'
    else:
        params['loss_reduction'] = 'mean'

    if 'LY' in results_dir:
        lead_time = int(out_dir.split('LY')[1][0])
    else:
        lead_time = None

    if 'fxpvar' not in  results_dir:
        params['fixed_posterior_variance'] =  None
    else:
        params['fixed_posterior_variance'] =  np.array(params['fixed_posterior_variance'])

    if 'cEFullBVAE' in  results_dir:
        params['full_conditioning'] = True 
    else:
        params['full_conditioning'] = False 

    if params['prior_flow'] is not None:
        dics = {}
        if len((params['prior_flow'].split('args'))) > 0:
            pass
        
        dics['num_layers'] = eval((params['prior_flow'].split('num_layers'))[-1].split('}')[0].split(':')[-1])   
        dics['type'] = eval(results_dir.split('prior')[0].split('_')[-1])
        params['prior_flow'] = dics

    if 'latentdependant' in  results_dir:
        params['condition_dependant_latent'] = True
        assert params['condition_embedding_size'] is not None
        assert params['non_random_decoder_initialization'] != 'histogram_based_sampling'
        params['full_conditioning'] = True
        if params['prior_flow'] is not None:
            assert params['non_random_decoder_initialization'] is False, 'non_random_decoder_initialization should be False for condition dependant flow based prior ...'
    else: 
        params['condition_dependant_latent'] = False

    try:
        params['arch'] = int(out_dir.split('arch')[1][0])
    except:
        params['arch'] = None
    
    assert params['version'] in [0, 1,2,3]

    if params['BVAE'] is not None:
        assert type(params['BVAE']) == int, 'Input the size of output ensemble as BVAE ...'
    else:
        params['BVAE'] = 1
    
    if params['BVAE'] is not None:
        params['ensemble_mode'] = 'LE'
        assert params['ensemble_list'] is not None, 'for the BVAE model you need to specify the ensemble size ...'
    
    if not params['non_random_decoder_initialization']:
       if not params['remove_ensemble_mean']:
            assert any([all([params['time_features'] is not None, params['append_mode'] != 1]), params['condition_embedding_size'] is not None]), 'For random decoder initializaiton, conditions should be provided to the decoder ...' 
    

    if params["model"] not in [Autoencoder, Autoencoder_decoupled]:
        params["append_mode"] = None

    if params['version'] == 1:

        params['forecast_preprocessing_steps'] = [
        ('standardize', Standardizer())]
        params['observations_preprocessing_steps'] = []

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
    

    print("Start training")


    ##### PG: Ensemble members to load 
    ensemble_list = params['ensemble_list']
    ###### PG: Add ensemble features to training features
    ensemble_mode = params['ensemble_mode'] ##
    
    if params["arch"] == 3:
        params["hidden_dims"] = [[1500, 720, 360, 180, 90, 30], [90, 180, 360, 720, 1500]]
    if params['arch'] == 2:
        params["hidden_dims"] = [[1500, 720, 360, 180, 90], [180, 360, 720, 1500]]
    if params['arch'] == 1:
        params["hidden_dims"] = [[ 720, 360, 180, 90,30], [90,180, 360, 720]]
    if params['condition_embedding_size'] == 'encoder':
        params['condition_embedding_size'] = params["hidden_dims"][0]



    if ensemble_list is not None: ## PG: calculate the mean if ensemble mean is none
        print("Load forecasts")
        ds_in = xr.open_dataset(data_dir_forecast).sel(ensembles = ensemble_list).load()['fgco2']
        if ensemble_mode == 'Mean': ##
            ds_in = ds_in.mean('ensembles') ##
        else:
            print(f'Warning: ensemble_mode is {ensemble_mode}. Training for large ensemble ...')

    else:    ## Load specified members
        print("Load forecasts") 
        ds_in = xr.open_dataset(data_dir_forecast).mean('ensembles').load()['fgco2']

    print("Load observations")

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
    
    del ds_in, obs_in
    gc.collect()
    ##### PG: The ocean carbon flux has NaN values over land in both forecast and obs data and these are not necessarily in the excat same grid points. ###
    ##### We need to extract the common grid points where both obs and model data exist. That said, we need to flatten both the training and target data
    ##### I defined a Nanremover class. See preprocessing.py.
    
    nanremover = Spatialnanremove()## PG: Get an instance of the class
    nanremover.fit(ds_raw_ensemble_mean[:,:12,...], ds_raw_ensemble_mean[:,:12,...]) ## PG:extract the commong grid points between training and obs data
    ds_raw_ensemble_mean = nanremover.to_map(nanremover.sample(ds_raw_ensemble_mean)) ## PG: flatten and sample training data at those locations
    obs_raw = nanremover.to_map(nanremover.sample(obs_raw)) ## PG: flatten and sample obs data at those locations    
    #######################################################################################################################################

    if 'atm_co2' in params['extra_predictors']:
        atm_co2 = xr.open_dataset('/home/rpg002/CMIP6_ssp245_xCO2atm_1982_2029.nc').ssp245
        atm_co2 = reshape_obs_to_data(atm_co2, ds_raw_ensemble_mean, return_xarray=True).rename({'month' : 'lead_time'})
        try:
            extra_predictors = atm_co2.sel(year = ds_raw_ensemble_mean.year).expand_dims('channels', axis=2)
        except:
            raise ValueError("Extra predictors not available at the same years as the predictors.")
        del atm_co2
        gc.collect()
    else:
           extra_predictors = None     

    if extra_predictors is not None:
        if params["model"] in [ Autoencoder]:
            weights = np.cos(np.ones_like(extra_predictors.lon) * (np.deg2rad(extra_predictors.lat.to_numpy()))[..., None])  # Moved this up
            weights = xr.DataArray(weights, dims = extra_predictors.dims[-2:], name = 'weights').assign_coords({'lat': extra_predictors.lat, 'lon' : extra_predictors.lon}) 
            extra_predictors = (extra_predictors * weights).sum(['lat','lon'])/weights.sum(['lat','lon'])
        else:  
            if not all(['ensembles' not in extra_predictors.dims, 'ensembles' in ds_raw_ensemble_mean.dims]): 
                
                ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean, extra_predictors], dim = 'channels')
            else:
                ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean, extra_predictors.expand_dims(ensembles = ds_raw_ensemble_mean['ensembles'], axis = 2) ], dim = 'channels')
            extra_predictors = None



    hyperparam = params["hyperparam"]
    reg_scale = params["reg_scale"]
    model = params["model"]
    hidden_dims = params["hidden_dims"]
    time_features = params["time_features"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    batch_normalization = params["batch_normalization"]
    dropout_rate = params["dropout_rate"]
    condition_embedding_size = params['condition_embedding_size']
    lr = params["lr"]
    l2_reg = params["L2_reg"]
    
    forecast_preprocessing_steps = params["forecast_preprocessing_steps"]
    observations_preprocessing_steps = params["observations_preprocessing_steps"]

    loss_region = params["loss_region"]
    subset_dimensions = params["subset_dimensions"]

    conditional_embedding = True if condition_embedding_size is not None else False
   
    del ds_raw
    gc.collect()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for y_idx, test_year in enumerate(test_years):

        print(f"Start run for test year {test_year}...")

        if model_year is None:
            model_year_ = np.min([test_year - 1, ds_raw_ensemble_mean.year[-1].values])
        else:
            model_year_ = model_year
        ####### time inclusion
        # if params['correction']:
        train_years = ds_raw_ensemble_mean.year[ds_raw_ensemble_mean.year <= model_year_].to_numpy()
        # else:
        #     train_years = ds_raw_ensemble_mean.year[ds_raw_ensemble_mean.year <= test_year].to_numpy()
        n_train = len(train_years)
        train_mask = create_train_mask(ds_raw_ensemble_mean[:n_train,...])
        # if not params['correction']:
        #     train_mask = np.full(train_mask.shape, False, dtype=bool)

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


        # Data preprocessing
        
        ds_pipeline = PreprocessingPipeline(forecast_preprocessing_steps).fit(ds_baseline, mask=preprocessing_mask_fct)
        ds = ds_pipeline.transform(ds_raw_ensemble_mean)

        obs_pipeline = PreprocessingPipeline(observations_preprocessing_steps).fit(obs_baseline, mask=preprocessing_mask_obs)
        if 'standardize' in ds_pipeline.steps:
            obs_pipeline.add_fitted_preprocessor(ds_pipeline.get_preprocessors('standardize'), 'standardize')
        obs = obs_pipeline.transform(obs_raw)

        if params['remove_ensemble_mean']:
            ds_em = ds.mean('ensembles')
            ds = ds - ds_em
        
        # if params['correction']:
        year_max = ds[:n_train + 1].year[-1].values 
        # else:
        # year_max = ds[:n_train].year[-1].values 

        del ds_baseline, obs_baseline, preprocessing_mask_obs, preprocessing_mask_fct
        gc.collect()
        # TRAIN MODEL
        ####### time inclusion
        
        ds_train = ds[:n_train,...]
        obs_train = obs[:n_train,...]

        # if params['correction']:
        
        ds_test = ds.sel(year = slice(test_year, test_year))
        obs_test = obs.sel(year = slice(test_year, test_year))
        # else:
        #     ds_test = ds[n_train - 1:n_train ,...]
        #     obs_test = obs[n_train -1:n_train ,...]

        if params['remove_ensemble_mean']:
            ds_em_train = ds_em.sel(year = ds_train.year)
            ds_em_test = ds_em.sel(year = ds_test.year)
            del ds_em

        weights = np.cos(np.ones_like(ds_train.lon) * (np.deg2rad(ds_train.lat.to_numpy()))[..., None])  # Moved this up
        weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon}) # Create an DataArray to pass to Spatialnanremove() 
        weights_ = weights.copy()
        
        ### Increase the weight over some biome :
        # if Biome is not None:
        #     if type(Biome) == dict:
        #         for ind, scale in Biome.items():
        #                 weights = weights + (scale-1) * weights.where(biomes == ind).fillna(0)       
        #         else:
        #                 weights = weights + weights.where(biomes == Biome).fillna(0)

        ########################################################################

        if model in [UNet , CNN]: ## PG: If the model starts with a nn.Conv2d write back the flattened data to maps.

            ds_train = ds_train.fillna(0.0) ## PG: fill NaN values with 0.0 for training
            obs_train = obs_train.fillna(0.0) ## PG: fill NaN values with 0.0 for training

            img_dim = ds_train.shape[-2] * ds_train.shape[-1] 
            if loss_region is not None:
                loss_region_indices, loss_area = get_coordinate_indices(ds_raw_ensemble_mean, loss_region)
            
            else:
                loss_region_indices = None
        
        else: ## PG: If you have a dense first layer keep the data flattened.

            ds_train = nanremover.sample(ds_train) ## PG: flatten and sample training data at those locations
            obs_train = nanremover.sample(obs_train) ## PG: flatten and sample obs data at those locations   
            weights = nanremover.sample(weights) ## PG: flatten and sample weighs at those locations
            weights_ = nanremover.sample(weights_)

            img_dim = ds_train.shape[-1] ## PG: The input dim is now the length of the flattened dimention.
            if loss_region is not None:
        
                loss_region_indices = nanremover.extract_indices(subregions[loss_region]) ## PG: We need 1D index list rather than 2D lat:lon.

            else:
                loss_region_indices = None

        del ds, obs
        gc.collect()
        try:
            torch.cuda.empty_cache() 
            torch.cuda.synchronize() 
        except:
            pass
        weights = weights.values
        weights_ = weights_.values

        if time_features is None:
                add_feature_dim = 0
        else:
                add_feature_dim = len(time_features)
        if extra_predictors is not None:
            add_feature_dim = add_feature_dim + len(params['extra_predictors'])



        if model == Autoencoder:
            net = model(img_dim, hidden_dims[0], hidden_dims[1], added_features_dim=add_feature_dim, append_mode=params['append_mode'], batch_normalization=batch_normalization, dropout_rate=dropout_rate, VAE = params['BVAE'], condition_embedding_dims = params['condition_embedding_size'], full_conditioning = params["full_conditioning"], 
                        condition_dependant_latent = params["condition_dependant_latent"], fixed_posterior_variance = params['fixed_posterior_variance'], prior_flow = params['prior_flow'], condemb_to_decoder = params['condemb_to_decoder'], device = device)
        elif model == Autoencoder_decoupled:
            net = model(img_dim, hidden_dims[0], hidden_dims[1], added_features_dim=add_feature_dim, append_mode=params['append_mode'], batch_normalization=batch_normalization, dropout_rate=dropout_rate)

        print('Loading model ....')
        net.load_state_dict(torch.load(glob.glob(results_dir+ '/Saved_models' + f'/*-{model_year_}*.pth')[0], map_location=torch.device('cpu'))) 

        net.to(device)
        ## PG: XArrayDataset now needs to know if we are adding ensemble features. The outputs are datasets that are maps or flattened in space depending on the model.
        if params['non_random_decoder_initialization'] == 'histogram_based_sampling':
            train_set = XArrayDataset(ds_train, obs_train, mask=train_mask, lead_time = lead_time, extra_predictors= extra_predictors,lead_time_mask = params['lead_time_mask'], in_memory=False, time_features=time_features, aligned = True, year_max = year_max, conditional_embedding = conditional_embedding, cross_member_training = params['cross_member_training']) 
        # dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        if conditional_embedding:
            if params['remove_ensemble_mean']:
                ds_train_conds = nanremover.sample(ds_em_train).stack(flattened=('year','lead_time')).transpose('flattened',...)[~train_mask.flatten()]
                del ds_em_train
            else:
                ds_train_conds = ds_train.stack(flattened=('year','lead_time')).transpose('flattened',...)[~train_mask.flatten()].mean('ensembles')
            if lead_time is not None:
                ds_train_conds = ds_train_conds.where((ds_train_conds.lead_time >=  (lead_time - 1) * 12 + 1) & (ds_train_conds.lead_time < (lead_time *12 )+1), drop = True)
            condition_standardizer = Standardizer()
            condition_standardizer.fit(ds_train_conds)
            ds_train_conds = condition_standardizer.transform(ds_train_conds)
            ds_train_conds = torch.from_numpy(ds_train_conds.to_numpy())

        # if reg_scale is None: ## PG: if no penalizing for negative anomalies
        # criterion = WeightedMSESignLossKLD(weights=weights, device=device, hyperparam=hyperparam, reduction=params['loss_reduction'], loss_area=loss_region_indices, scale=reg_scale, Beta = params['beta'])
        # else:
        #         criterion = WeightedMSESignLossKLD(weights=weights, device=device, hyperparam=hyperparam, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, min_val=0, max_val=None)
        # criterion = VAEloss(weights = weights , device = device)

        # epoch_loss = []
        # epoch_MSE = []
        # net.train()
        # num_batches = len(dataloader)
        # for epoch in tqdm.tqdm(range(epochs)):
        #     #### You can comment this section if you don't want shuffling with each epoch ###
        #     if all([params['cross_member_training'], epoch>=1]):
        #         train_set = train_set.shuffle_target_ensembles()
        #         dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        #     #############################################################################
        #     batch_loss = 0
        #     batch_loss_MSE = 0
        #     for batch, (x, y) in enumerate(dataloader):
        #         if conditional_embedding:
        #             cond_idx = x[-1]
        #             x = [x[i] for i in range(len(x) - 1)] if len(x) >2 else x[0]
        #             cond = ds_train_conds[cond_idx].float().to(device)
        #         else:
        #             cond = None
        #         if (type(x) == list) or (type(x) == tuple):
        #             x = (x[0].to(device), x[1].to(device))
        #         else:
        #             x = x.to(device)
        #         if (type(y) == list) or (type(y) == tuple):
        #             y, m = (y[0].to(device), y[1].to(device))
        #         else:
        #             y = y.to(device)
        #             m  = None

        #         optimizer.zero_grad()
        #         # if not params['cross_member_training']:
        #         y = x[0] if (type(x) == list) or (type(x) == tuple) else x
        #         adjusted_forecast, mu, log_var = net(x, condition = cond,  sample_size = params['training_sample_size'])
        #         adjusted_forecast = adjusted_forecast.mean(0)
        #             # adjusted_forecast = torch.flatten(adjusted_forecast, start_dim= 0 , end_dim=1)
        #         # else:
        #         #     ## first generate mu and sigma then sample from z
        #         #     pass
        #         loss, MSE, KLD = criterion(adjusted_forecast, y ,mu, log_var, mask = m, return_ind_loss=True)

        #         batch_loss += loss.item()
        #         batch_loss_MSE += MSE.item()
        #         loss.backward()
        #         optimizer.step()
        #     epoch_loss.append(batch_loss / num_batches)
        #     epoch_MSE.append(batch_loss_MSE / num_batches)

        #     if params['lr_scheduler']:
        #         scheduler.step()
        # del train_set, dataloader, adjusted_forecast, x, y , m, criterion, loss
        # gc.collect()

        # EVALUATE MODEL
        ##################################################################################################################################
        ####### time inclusion
        ds_test = nanremover.sample(ds_test, mode = 'Eval')  ## PG: Sample the test data at the common locations
        obs_test = nanremover.sample(obs_test)
        if model in [UNet, CNN]:
            ds_test = nanremover.to_map(ds_test).fillna(0.0)  ## PG: Write back to map if the model starts with a nn.Conv2D
            obs_test = nanremover.to_map(obs_test).fillna(0.0)
        ##################################################################################################################################

        test_lead_time_list = np.arange(1, ds_test.shape[1] + 1)
        test_years_list = np.arange(1, ds_test.shape[0] + 1)  ## PG: Extract the number of years as well 
        test_set = XArrayDataset(ds_test, obs_test, lead_time = lead_time, extra_predictors= extra_predictors,lead_time_mask = params['lead_time_mask'], time_features=time_features,  in_memory=False, aligned = True, year_max = year_max, BVAE = params['BVAE'])
        # dataloader_test = DataLoader(test_set, batch_size=1, shuffle=False)
        criterion_test =  WeightedMSE(weights=weights_, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
        test_results = np.zeros_like(xr.concat([ds_test for _ in range(params['BVAE'])], dim = 'ensembles').values)
        ds_test = ds_test.rename({'ensembles' : 'batch'})

        if params['non_random_decoder_initialization'] == 'histogram_based_sampling':
            x_in = torch.from_numpy(train_set.data.to_numpy()).float().to(device)
            if conditional_embedding:
                cond = ds_train_conds[train_set.condition_target_ids].float().to(device)
            else:
                cond = None
            if all([params['time_features'] is not None, params['append_mode'] != 1]):
                x_in = (x_in , torch.from_numpy(train_set.time_features).float().to(device))
            with torch.no_grad():
                train_mu = net(x_in, condition = cond, sample_size = 1)[1]
                try:
                    train_mu = train_mu.numpy()
                except:
                    train_mu = train_mu.cpu().numpy()
            hist, edges = np.histogramdd(train_mu, bins=100 * net.latent_size )
            del x_in, cond

        if 'ensembles' in ds_test.dims:
            test_loss = np.zeros(shape=(ds_test.shape[0], ds_test.shape[1], ds_test.shape[2]))
        else:
            test_loss = np.zeros(shape=(ds_test.shape[0], ds_test.shape[1]))
            
        for i, (x, target) in enumerate(test_set):          
                year_idx, lead_time_list_idx = np.divmod(i, len(test_lead_time_list))
                lead_time_idx = test_lead_time_list[lead_time_list_idx] - 1
                net.eval()
                with torch.no_grad():
                    if (type(x) == list) or (type(x) == tuple):
                        test_raw = (x[0].to(device), x[1].to(device))
                        test_raw = (test_raw[0],test_raw[1].unsqueeze(-2).expand(test_raw[0].shape[0], test_raw[1].shape[-1]))
                    else:
                        test_raw = x.to(device)
                        # test_raw = torch.flatten(test_raw, start_dim= 0 , end_dim=1)
                    if (type(target) == list) or (type(target) == tuple):
                        test_obs, m = (target[0].to(device).unsqueeze(0), target[1].to(device).unsqueeze(0))
                    else:
                        test_obs = target.to(device).unsqueeze(0)
                        m = None
                    if conditional_embedding:
                            cond = test_raw[0].mean(axis = -3) if (type(test_raw) == list) or (type(test_raw) == tuple) else test_raw.mean(axis = -3)         
                    else:
                            cond = None

                    cond = (cond - condition_standardizer.mean)/condition_standardizer.std
                    sample_size = test_raw[0].shape[0] if (type(test_raw) == list) or (type(test_raw) == tuple) else test_raw.shape[0]
                    if params['non_random_decoder_initialization'] is False:
                        z =  Normal(torch.zeros(net.latent_size), torch.ones(( net.latent_size))).rsample(sample_shape=(params['BVAE']* sample_size,)).to(device)

                    else:
                        if params['condition_dependant_latent']:
                                _, _, _, cond_mu, cond_log_var = net(test_raw, condition = cond, sample_size = 1)
                                cond_var = torch.exp(cond_log_var) + 1e-4
                                z =  Normal(cond_mu, torch.sqrt(cond_var)).rsample(sample_shape=(params['BVAE'] * sample_size,)).to(device)
                    
                        if params['non_random_decoder_initialization'] == 'encoder_based_sampling':
                            _, mu, log_var = net(test_raw, condition = cond, sample_size = 1)[:3]
                            var = torch.exp(log_var) + 1e-4
                            z =  Normal(torch.mean(mu, 0), torch.std(mu, 0)).rsample(sample_shape=(params['BVAE'] * sample_size,)).to(device)
                            
                        elif params['non_random_decoder_initialization'] == 'histogram_based_sampling':
                            z = hist_sampling(hist, edges, net.latent_size , num_samples = params['BVAE'] * sample_size)
                            z = torch.from_numpy(z).float().to(device)
                        # z = torch.unflatten(z, dim = 0, sizes = (-1,len(ensemble_list)))
                    #### cut from above

                    if params['prior_flow'] is not None:
                            # shape = z.shape
                            # z,_ = net.flow(torch.flatten(z, start_dim = 0, end_dim = 1))
                            # z = torch.unflatten(z, dim = 0, sizes = shape[:2])
                            if params['condition_dependant_latent']:
                                cond_embedded = net.embedding(cond.to(device))
                                try:
                                    cond_embedded = net.condition_embedding(cond_embedded.flatten(start_dim=1)) ## for versions older than 11 Nov 2024 use thise line!
                                except:
                                    pass
                                cond_embedded = cond_embedded.expand((z.shape[0], net.embedding_size))
                            else:
                                cond_embedded = None
                            z,_ = net.flow.inverse(z, condition = cond_embedded)
                            
                    if all([params['time_features'] is not None, params['append_mode'] != 1]):
                        z = torch.unflatten(z, dim = 0, sizes = (-1,len(ensemble_list)))
                        z = torch.cat([z, test_raw[1].unsqueeze(0).expand((params['BVAE'], *test_raw[1].shape))], dim=-1)
                        z = torch.flatten(z, start_dim = 0, end_dim = 1)

                    if all([conditional_embedding is True,  params['condemb_to_decoder']]):
                        cond_embedded = net.embedding(cond.to(device))
                        if all([params['condition_dependant_latent'], params['flow'] is None]):
                            cond_embedded = net.condition_mu(cond_embedded.flatten(start_dim=1))
                            try:
                                cond_embedded = net.condition_embedding(cond_embedded.flatten(start_dim=1)) ## for versions older than 11 Nov 2024 use thise line!
                            except:
                                pass
                        # cond_embedded = cond_embedded.unsqueeze(-2).expand(cond_embedded.shape[0], int(sample_size/cond_embedded.shape[0]), net.embedding_size)
                        # cond_embedded = torch.flatten(cond_embedded, start_dim = 0, end_dim = 1)
                        # cond_embedded = cond_embedded.unsqueeze(0).expand((z.shape[0], z.shape[1], net.embedding_size))
                        cond_embedded = cond_embedded.expand((z.shape[0], net.embedding_size))
                        z = torch.cat([z, cond_embedded], dim=-1)

                    # out_shape = z.shape
                    # out = net.decoder(torch.flatten(z, start_dim = 0, end_dim = 1))
                    # out = torch.unflatten(out, dim = 0 , sizes = out_shape[0:2])

                    # test_adjusted =   out.view((params['BVAE'], *test_raw[0].shape) ) if (type(x) == list) or (type(x) == tuple) else out.view((params['BVAE'], *test_raw.shape) )
                    # test_adjusted = torch.unflatten(test_adjusted, dim = 1, sizes = (-1,len(ensemble_list)))
                    # test_adjusted = torch.flatten(torch.transpose(test_adjusted, 1,0), start_dim= 1 , end_dim=2)
                    target = torch.mean(test_raw[0], 0)  if (type(x) == list) or (type(x) == tuple) else  torch.mean(test_raw, 0)
                    #########
                    out = net.decoder(z)
                    test_adjusted =   out.unsqueeze(-2)                                                                                      
                    # loss = criterion_test(torch.mean(test_adjusted, 1), target.unsqueeze(-2) )
                    loss = criterion_test(torch.mean(test_adjusted, 0), target)
                    
                    test_results[year_idx,lead_time_idx, ] = test_adjusted.to(torch.device('cpu')).numpy()
                    test_loss[year_idx, lead_time_idx] = loss.item() 
        del  test_set , test_raw, test_obs, x, target, m,  test_adjusted , ds_test, obs_test,
        gc.collect()

        reverse_preprocessing_pipeline =  ds_pipeline
        ##################################################################################################################################
        if model in [UNet , CNN]:   ## PG: if the output is already a map
            test_results_untransformed = reverse_preprocessing_pipeline.inverse_transform(test_results)
            result = xr.DataArray(test_results_untransformed, ds_test.coords, ds_test.dims, name='nn_adjusted')
        else:  
            test_results_upsampled = nanremover.to_map(test_results)  ## PG: If the output is spatially flat, write back to maps
            if params['remove_ensemble_mean']:
                test_results_upsampled = test_results_upsampled + ds_em_test
                del ds_em_test
            test_results_untransformed = reverse_preprocessing_pipeline.inverse_transform(test_results_upsampled.values) ## PG: Check preprocessing.AnomaliesScaler for changes
            result = xr.DataArray(test_results_untransformed, test_results_upsampled.coords, test_results_upsampled.dims, name='nn_adjusted')
        
        ##################################################################################################################################
        # Store results as NetCDF            
        if params['non_random_decoder_initialization']:
            result.to_netcdf(path=Path(results_dir, f'tests/nn_adjusted_{test_year}_saved_model_{params["non_random_decoder_initialization"]}.nc', mode='w'))
        else:
            result.to_netcdf(path=Path(results_dir, f'tests/nn_adjusted_{test_year}_saved_model_{num_stds}stds.nc', mode='w'))

        del   test_results, test_results_untransformed
        del result, net
        gc.collect()
        try:
            torch.cuda.empty_cache() 
            torch.cuda.synchronize()
        except:
            pass   

def hist_sampling(hist, edges, latent_dim , num_samples = 200, bin_shrinkage = 0):

    hist_prob = hist / hist.sum()
    hist_flat = hist_prob.ravel()
    bin_indices = np.random.choice(len(hist_flat), p=hist_flat, size=num_samples)
    bin_coords = np.unravel_index(bin_indices, hist.shape)
    samples = np.zeros((num_samples, latent_dim)) 
    for dim in range(latent_dim):
        # Get the lower and upper edges for the current dimension's bins
        bin_lower_edges = edges[dim][bin_coords[dim]]
        bin_upper_edges = edges[dim][bin_coords[dim] + 1]

        # Uniformly sample within the bin's lower and upper edge for each dimension
        samples[:, dim] = np.random.uniform(bin_lower_edges * (1 - bin_shrinkage), bin_upper_edges * (1 - bin_shrinkage), size=num_samples)
    return samples

def extract_params(model_dir):
    params = {}
    path = glob.glob(model_dir + '/*.txt')[0]
    file = open(path)
    content=file.readlines()
    for line in content:
        key = line.split('\t')[0]
        try:
            value = line.split('\t')[1].split('\n')[0]
        except:
            value = line.split('\t')[1]
        try:    
            params[key] = eval(value)
        except:
            if key == 'ensemble_list':
                ls = []
                for item in value.split('[')[1].split(']')[0].split(' '):
                    try:
                        ls.append(eval(item))
                    except:
                        pass
                params[key] = ls
            else:
                params[key] = value
    return params

if __name__ == "__main__":

    fake_data = 'pi'
    out_dir_x  = f'/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/fgco2_ems/SOM-FFN/results/Autoencoder/run_set_8_toy'
    out_dir    = f'{out_dir_x}/N2_v1_Banealing_L0_archNone_batch100_e100_lr_scheduler_cBVAElatentdependant_cR_10-1_MAFprior_hist_fake_data_normal_pi_TSE20' 

    lead_years = int(len(xr.open_mfdataset(str(Path(out_dir , "*.nc")), combine='nested', concat_dim='year').lead_time)/12)

    params = extract_params(out_dir)
    print(f'loaded configuration: \n')
    for key, values in params.items():
        print(f'{key} : {values} \n')
    
    try:
        version = int(out_dir.split('/')[-1].split('_')[1][1])
    except:
        version = (out_dir.split('/')[-1].split('_')[1][1:])
      
    params["version"] = version
    print( f'Version: {version}')
    print(f'lead_years: {lead_years}')

    ### handles
    num_stds = 1
    params['non_random_decoder_initialization'] = False ## False,True, 'encoder_based_sampling', 'histogram_based_sampling'
    params['BVAE'] = 500
    test_years = [2013,2014] #np.arange(2005,2015) #
    

    Path(out_dir + '/tests').mkdir(parents=True, exist_ok=True)

    if fake_data is not None:
        if fake_data == '2pi':
            data_dir_forecast = LOC_FORECASTS_fgco2_2pi
        elif fake_data == 'pi':
            data_dir_forecast = LOC_FORECASTS_fgco2_pi
        else:
            data_dir_forecast = LOC_FORECASTS_fgco2_simple
        unit_change = 1

    predict(params, test_years=test_years,model_year = None, num_stds = num_stds , lead_years=lead_years,  results_dir=out_dir)
    print(f'Output dir: {out_dir}')
    print('Training done.')

