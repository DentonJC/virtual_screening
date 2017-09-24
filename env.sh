#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$HOME/virtual_sreening
export DATA_DIR=$SCRATCH/virtual_sreening/data
export RESULTS_DIR=$SCRATCH/virtual_sreening/tmp

THEANO_FLAGS=mode=FAST_RUN,device=opencl0:0,floatX=float32
export THEANO_FLAGS
