#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=$PATH:$(pwd)
export DATA_DIR=$(pwd)/data
export RESULTS_DIR=$(pwd)/tmp

THEANO_FLAGS=mode=FAST_RUN,device=opencl0:0,floatX=float32
export THEANO_FLAGS
