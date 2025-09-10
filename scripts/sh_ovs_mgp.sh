#!/bin/bash

current_path=$(pwd)
root_dir=$(dirname "$current_path")

export PYTHONPATH=$PYTHONPATH:$root_dir
cd ../ovs/
# export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=8 main_ovs_mgp.py --cfg-path $1
