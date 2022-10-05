python habitat_baselines/run.py \
    --exp-config habitat_baselines/config/rearrange/ddppo_pick.yaml \
    --run-type train \
    TENSORBOARD_DIR ./pick_r3m_tb/ \
    CHECKPOINT_FOLDER ./pick_r3m_checkpoints/ \
    LOG_FILE ./pick_r3m_train.log