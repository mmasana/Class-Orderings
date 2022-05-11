#!/bin/bash

if [ "$1" != "" ]; then
    echo "Running on gpu: $1"
else
    echo "No gpu has been assigned."
fi

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd .. && pwd )"
if [ "$2" != "" ]; then
    PROJECT_DIR=$2
else
    echo "No FACIL dir is given. Default will be used."
fi
SRC_DIR="$PROJECT_DIR/src"
echo "Project dir: $PROJECT_DIR"
echo "Sources dir: $SRC_DIR"

RESULTS_DIR="$PROJECT_DIR/results"
if [ "$3" != "" ]; then
    RESULTS_DIR=$3
else
    echo "No results dir is given. Default will be used."
fi
echo "Results dir: $RESULTS_DIR"

for SEED in 0 1 2 3 4 5 6 7 8 9
do
      PYTHONPATH=$SRC_DIR python3 -u "$SRC_DIR"/main_incremental.py --exp-name base_${SEED} \
             --datasets cifar100_fixed --num-tasks 1 --network resnet32 --seed $SEED \
             --nepochs 200 --batch-size 128 --results-path "$RESULTS_DIR" \
             --gridsearch-tasks 1 --gridsearch-config gridsearch_config \
             --approach finetuning --gpu "$1" --save-models
done
