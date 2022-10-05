python habitat_baselines/run.py \
    --exp-config habitat_baselines/config/rearrange/ddppo_open_cab.yaml \
    --run-type train \
    TENSORBOARD_DIR ./open_cab_tb/ \
    CHECKPOINT_FOLDER ./open_cab_checkpoints/ \
    LOG_FILE ./open_cab_train.log