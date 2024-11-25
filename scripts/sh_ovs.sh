#!/bin/bash

current_path=$(pwd)
root_dir=$(dirname "$current_path")

export PYTHONPATH=$PYTHONPATH:$root_dir
cd ../ovs/
CUDA_VISIBLE_DEVICES=3 python main_ovs.py --cfg-path $1