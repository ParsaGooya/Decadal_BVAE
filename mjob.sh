#!/bin/bash
#PBS -l select=1:ncpus=12:vntype=gpu:ngpus=1:mem=110G
#PBS -l place=scatter
#PBS -W umask=022
#PBS -l walltime=10:00:00
#PBS -o BashHelloJob.out
#PBS -e BashHelloJob.err

working_dir=/home/rpg002/fgco2_decadal_forecast_adjustment_BVAE_historical/
source .condainit  
conda activate rsaenv_neurips  
cd ${working_dir}
python run_training_BVAE.py  >> /home/rpg002/fgco2_decadal_forecast_adjustment_BVAE_historical/output.txt