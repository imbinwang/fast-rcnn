#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/caffenet_linemod_pose_reg.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/wb_CaffeNet/solver_linemod_pose_reg.prototxt \
  --iters 30000 \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb linemod_train \
  --cfg experiments/wb_cfgs/linemod_pose_reg.yml
