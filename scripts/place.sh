#!/bin/bash
#SBATCH --job-name=ddppo_place
#SBATCH --output=ddppo_place.out
#SBATCH --error=ddppo_place.err
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task 10
#SBATCH --nodes 1
#SBATCH --mem=120GB
#SBATCH --time=96:30:00
#SBATCH --constraint=[gpu]
#SBATCH --partition=batch

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet


module load cudnn/7.5.0-cuda10.1.105
module load nccl/2.4.8-cuda10.1
source /home/hanp/anaconda3/bin/activate rx_habitat
cd /home/hanp/hanp/research/UESTC/xuan_rao/code/habitat-lab

set -x
python habitat_baselines/run.py \
    --exp-config habitat_baselines/config/rearrange/ddppo_place.yaml \
    --run-type train \
    TENSORBOARD_DIR ./place_r3m_tb/ \
    CHECKPOINT_FOLDER ./place_r3m_checkpoints/ \
    LOG_FILE ./place_r3m_train.log