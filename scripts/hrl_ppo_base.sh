## !/bin/bash
## SBATCH --job-name=5_hrl_ppo
## SBATCH --output=5_hrl_ppo.out
## SBATCH --error=5_hrl_ppo.err
## SBATCH --gres=gpu:v100:3
## SBATCH --cpus-per-task 5
## SBATCH --nodes 1
## SBATCH --ntasks-per-node 3
## SBATCH --mem=30GB
## SBATCH --time=168:30:00
## SBATCH --constraint=[gpu]
## #SBATCH --partition=batch
## SBATCH --partition=batch

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet


# module load cudnn/7.5.0-cuda10.1.105
# module load nccl/2.4.8-cuda10.1
# source /home/hanp/anaconda3/bin/activate hrl_habitat
# cd /home/hanp/hanp/research/UESTC/xuan_rao/code/HRL-habitat-lab

set -x
python habitat_baselines/run.py \
    BASE_TASK_CONFIG_PATH ../habitat-challenge/configs/tasks/rearrange.local.rgbd.yaml \
    TASK_CONFIG.DATASET.SPLIT 'train' \
    TASK_CONFIG.TASK.TASK_SPEC_BASE_PATH ../habitat-challenge/configs/pddl/ \
    TASK_CONFIG.SIMULATOR.DEBUG_RENDER True \
    TENSORBOARD_DIR ./hierarchical_tb/ \
    CHECKPOINT_FOLDER ./hierarchical_checkpoints/ \
    LOG_FILE ./hierarchical_train.log 
