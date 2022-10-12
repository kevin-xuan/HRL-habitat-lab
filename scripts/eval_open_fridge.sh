python habitat_baselines/run.py \
    --exp-config habitat_baselines/config/rearrange/ddppo_open_fridge.yaml \
    --run-type eval \
    EVAL_CKPT_PATH_DIR ./open_cab_checkpoints \
    LOG_FILE ./open_fridge_eval.log \
    NUM_ENVIRONMENTS 1 \
    VIDEO_OPTION [] \
    TEST_EPISODE_COUNT 100