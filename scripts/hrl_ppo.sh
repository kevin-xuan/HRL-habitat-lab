#!/bin/bash
#SBATCH --job-name=5_hrl_ppo
#SBATCH --output=5_hrl_ppo.out
#SBATCH --error=5_hrl_ppo.err
#SBATCH --gres=gpu:v100:3
#SBATCH --cpus-per-task 5
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 3
#SBATCH --mem=30GB
#SBATCH --time=168:30:00
#SBATCH --constraint=[gpu]
#SBATCH --partition=batch
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet


module load cudnn/7.5.0-cuda10.1.105
module load nccl/2.4.8-cuda10.1
source /home/hanp/anaconda3/bin/activate hrl_habitat
cd /home/hanp/hanp/research/UESTC/xuan_rao/code/HRL-habitat-lab

set -x
srun python habitat_baselines/run.py \
    TENSORBOARD_DIR ./5_hrl_tb/ \
    CHECKPOINT_FOLDER ./5_hrl_checkpoints/ \
    LOG_FILE ./5_hrl_train.log 
