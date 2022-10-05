python habitat_baselines/run.py \
    --exp-config habitat_baselines/config/rearrange/ddppo_nav_to_obj.yaml \
    --run-type train \
    TENSORBOARD_DIR ./nav_to_obj_r3m_tb/ \
    CHECKPOINT_FOLDER ./nav_to_obj_r3m_checkpoints/ \
    LOG_FILE ./nav_to_obj_r3m_train.log