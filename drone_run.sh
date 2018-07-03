# change this to be based on your directory
CONFIG_PATH=$RL_SRC_DIR/docs/drone_config.ini

# same as in config.ini under TRAIN_CONFIG
BASE_DIR=$RL_RESULT_DIR/drone_with_RL_results/
# clean the results
rm -rf $BASE_DIR
mkdir -p $BASE_DIR

ALGO=ddpg

python3 main.py --config-path $CONFIG_PATH --algo $ALGO
