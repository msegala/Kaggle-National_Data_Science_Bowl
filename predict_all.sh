#!/bin/bash

echo "Running run_analysis.py"
sudo THEANO_FLAGS='cuda.root=/opt/cuda,device=gpu,floatX=float32' python2.7 run_predict.py predict model-base.pickle base

echo "Running run_analysis_GaussianNoiseLayer.py"
sudo THEANO_FLAGS='cuda.root=/opt/cuda,device=gpu,floatX=float32' python2.7 run_predict.py predict model-base_GaussianNoiseLayer.pickle GaussianNoiseLayer

echo "Running run_analysis_untieBiases.py"
sudo THEANO_FLAGS='cuda.root=/opt/cuda,device=gpu,floatX=float32' python2.7 run_predict.py predict model-base_UntieBiases.pickle UntieBiases

echo "Running run_analysis_GaussianNoiseLayer_untieBiases.py"
sudo THEANO_FLAGS='cuda.root=/opt/cuda,device=gpu,floatX=float32' python2.7 run_predict.py predict model-base_GaussianNoiseLayer_untieBiases.pickle GaussianNoiseLayer_untieBiases

echo "Running run_analysis_more_filters.py"
sudo THEANO_FLAGS='cuda.root=/opt/cuda,device=gpu,floatX=float32' python2.7 run_predict.py predict model-base_more_filters.pickle more_filters

echo "Running run_analysis_more_filters_2.py"
sudo THEANO_FLAGS='cuda.root=/opt/cuda,device=gpu,floatX=float32' python2.7 run_predict.py predict model-base_more_filters_2.pickle more_filters_2

echo "Running run_analysis_swap_dropout.py"
sudo THEANO_FLAGS='cuda.root=/opt/cuda,device=gpu,floatX=float32' python2.7 run_predict.py predict model-base_swap_dropout.pickle swap_dropout

echo "Running run_analysis_increaseDropout.py"
sudo THEANO_FLAGS='cuda.root=/opt/cuda,device=gpu,floatX=float32' python2.7 run_predict.py predict model-base_increaseDropoutRate.pickle increaseDropoutRate

