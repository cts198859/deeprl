# change this to be based on your directory
BASE_RL_DIR=/Users/jenyapergament/Documents/Research/Betteroff/RL_code/deeprl
BASE_RL_RESULTS_DIR=/Users/jenyapergament/Documents/Research/Betteroff/RL_code/deeprl/results

CONFIG_PATH=$BASE_RL_DIR/docs/drone_config.ini

# same as in config.ini under TRAIN_CONFIG
BASE_DIR=$BASE_RL_RESULTS_DIR/drone_with_RL_results/

# clean the results
#rm -rf $BASE_DIR
#mkdir -p $BASE_DIR:

ALGO=ddpg

python3 main.py --mode 'evaluate' --config-path $CONFIG_PATH --algo $ALGO
