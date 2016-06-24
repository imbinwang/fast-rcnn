#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/smallnet_linemod_comp_2.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/wb_SmallNet/solver_linemod.prototxt \
  --iters 2000000 \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb linemod_bg_train \
  --cfg experiments/wb_cfgs/linemod_comp_2.yml
