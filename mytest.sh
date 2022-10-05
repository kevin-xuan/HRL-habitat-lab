#/bin/bash

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

set -x
python habitat_baselines/run.py \
    --exp-config ../habitat-challenge/configs/methods/ddppo_monolithic.yaml \
    --run-type train \
    BASE_TASK_CONFIG_PATH ../habitat-challenge/configs/tasks/rearrange.local.rgbd.yaml \
    TASK_CONFIG.DATASET.SPLIT 'train' \
    TASK_CONFIG.TASK.TASK_SPEC_BASE_PATH ../habitat-challenge/configs/pddl/ \
    TENSORBOARD_DIR ./tb \
    CHECKPOINT_FOLDER ./checkpoints \
    LOG_FILE ./train.log