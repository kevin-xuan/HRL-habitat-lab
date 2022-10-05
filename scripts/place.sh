python habitat_baselines/run.py \
    --exp-config habitat_baselines/config/rearrange/ddppo_place.yaml \
    --run-type train \
    TENSORBOARD_DIR ./place_r3m_tb/ \
    CHECKPOINT_FOLDER ./place_r3m_checkpoints/ \
    LOG_FILE ./place_r3m_train.log