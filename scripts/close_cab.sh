python habitat_baselines/run.py \
    --exp-config habitat_baselines/config/rearrange/ddppo_close_cab.yaml \
    --run-type train \
    TENSORBOARD_DIR ./close_cab_tb/ \
    CHECKPOINT_FOLDER ./close_cab_checkpoints/ \
    LOG_FILE ./close_cab_train.log