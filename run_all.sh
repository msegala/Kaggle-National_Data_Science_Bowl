#!/bin/bash

echo "Running run_analysis.py"
sudo THEANO_FLAGS='cuda.root=/opt/cuda,device=gpu,floatX=float32' python2.7 run_analysis.py fit model-base.pickle

echo "Running run_analysis_GaussianNoiseLayer.py"
sudo THEANO_FLAGS='cuda.root=/opt/cuda,device=gpu,floatX=float32' python2.7 run_analysis_GaussianNoiseLayer.py fit model-base_GaussianNoiseLayer.pickle

echo "Running run_analysis_untieBiases.py"
sudo THEANO_FLAGS='cuda.root=/opt/cuda,device=gpu,floatX=float32' python2.7 run_analysis_untieBiases.py fit model-base_UntieBiases.pickle

echo "Running run_analysis_GaussianNoiseLayer_untieBiases.py"
sudo THEANO_FLAGS='cuda.root=/opt/cuda,device=gpu,floatX=float32' python2.7 run_analysis_GaussianNoiseLayer_untieBiases.py fit model-base_GaussianNoiseLayer_untieBiases.pickle

echo "Running run_analysis_more_filters.py"
sudo THEANO_FLAGS='cuda.root=/opt/cuda,device=gpu,floatX=float32' python2.7 run_analysis_more_filters.py fit model-base_more_filters.pickle

echo "Running run_analysis_more_filters_2.py"
sudo THEANO_FLAGS='cuda.root=/opt/cuda,device=gpu,floatX=float32' python2.7 run_analysis_more_filters_2.py fit model-base_more_filters_2.pickle

echo "Running run_analysis_swap_dropout.py"
sudo THEANO_FLAGS='cuda.root=/opt/cuda,device=gpu,floatX=float32' python2.7 run_analysis_swap_dropout.py fit model-base_swap_dropout.pickle

echo "Running run_analysis_increaseDropout.py"
sudo THEANO_FLAGS='cuda.root=/opt/cuda,device=gpu,floatX=float32' python2.7 run_analysis_increaseDropout.py fit model-base_increaseDropoutRate.pickle

