# change this to be based on your directory
CONFIG_PATH=/Users/csandeep/Documents/work/uhana/work/deeprl/config.ini

# same as in config.ini under TRAIN_CONFIG
BASE_DIR=/Users/csandeep/Documents/work/uhana/work/rl_test/

# clean the results
rm -rf $BASE_DIR
mkdir -p $BASE_DIR

python3 main.py --config-path $CONFIG_PATH

