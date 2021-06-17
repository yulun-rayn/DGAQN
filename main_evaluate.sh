#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate dgaqn-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name DGAQN_eval_1000"
# PYARGS="$PYARGS --use_cpu"
PYARGS="$PYARGS --gpu 0"
PYARGS="$PYARGS --data_path $DATA/src/dataset"
PYARGS="$PYARGS --warm_start_dataset NSP15_6W01_A_3_H.negonly_unique_30k.csv"
PYARGS="$PYARGS --artifact_path $DATA/artifact/dgaqn"
PYARGS="$PYARGS --reward_type dock" # options: logp, plogp, dock
PYARGS="$PYARGS --adt_tmp_dir 000"

# PYARGS="$PYARGS --greedy"
PYARGS="$PYARGS --model_path /path/to/trained/dgaqn.pth"
PYARGS="$PYARGS --nb_test 1000"

python src/main_evaluate.py $PYARGS