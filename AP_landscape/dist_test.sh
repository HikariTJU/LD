#!/usr/bin/env bash

CONFIG1=$1
CONFIG2=$2
CHECKPOINT1=$3
CHECKPOINT2=$4
GPUS=$5
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG1 $CONFIG2 $CHECKPOINT1 $CHECKPOINT2 --launcher pytorch ${@:4}
