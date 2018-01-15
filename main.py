import argparse
import configparser
import numpy as np
import os
import tensorflow as tf
import multiprocessing as mp
from agents.models import run_update
from agents.utils import GlobalCounter
from envs.wrapper import GymEnv
from train import run_explore
import time


def parse_args():
    default_config_path = '/Users/tchu/Documents/Uhana/remote/deeprl/config.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=False,
                        default=default_config_path, help="config path")
    return parser.parse_args()


def init_out_dir(base_dir):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    save_path = base_dir + '/model/'
    log_path = base_dir + '/log/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    return save_path, log_path


def init_shared_data(train_config):
    total_step = int(train_config.getfloat('MAX_STEP'))
    n_env = int(train_config.getfloat('NUM_ENV'))
    save_step = int(train_config.getfloat('SAVE_INTERVAL'))
    log_step = int(train_config.getfloat('LOG_INTERVAL'))
    global_counter = GlobalCounter(total_step, save_step, log_step)
    mp_dict = mp.Manager().dict()
    mp_list = []
    mp_dict['global_counter'] = global_counter
    for _ in range(n_env):
        # batch, wt, (cum_reward, step)
        cur_list = [mp.Queue(1), mp.Queue(1), mp.Queue(1)]
        mp_list.append(cur_list)
    return mp_dict, mp_list


def gym_env():
    # input paramters
    args = parse_args()
    parser = configparser.ConfigParser()
    parser.read(args.config_path)
    seed = parser.getint('TRAIN_CONFIG', 'SEED')
    n_env = parser.getint('TRAIN_CONFIG', 'NUM_ENV')
    env_name = parser.get('ENV_CONFIG', 'NAME')
    is_discrete = parser.getboolean('ENV_CONFIG', 'DISCRETE')
    total_step = int(parser.getfloat('TRAIN_CONFIG', 'MAX_STEP'))
    base_dir = parser.get('TRAIN_CONFIG', 'BASE_DIR')

    # initalize global agent and env
    env = GymEnv(env_name, is_discrete)
    env.seed(seed)
    n_a = env.n_a
    n_s = env.n_s
    save_path, log_path = init_out_dir(base_dir)

    tf.set_random_seed(seed)
    mp_dict, mp_list = init_shared_data(parser['TRAIN_CONFIG'])

    global_agent = mp.Process(target=run_update,
                              args=(n_s, n_a, total_step, parser['MODEL_CONFIG'], is_discrete,
                                    n_env, save_path, log_path, mp_dict, mp_list),
                              daemon=True)
    global_agent.start()
    local_agents = []
    for i in range(n_env):
        agent = mp.Process(target=run_explore,
                           args=(n_s, n_a, total_step, i, parser['MODEL_CONFIG'], is_discrete,
                                 env_name, seed, mp_dict, mp_list[i]))
        agent.start()
        local_agents.append(agent)
    try:
        global_agent.join()
    except KeyboardInterrupt:
        print('ctrl+C pressed ...')
        global_agent.join()
        # TODO: why join is executed correctly sometimes?
        time.sleep(2)
    except:
        global_agent.terminate()
        global_agent.join()

    for agent in local_agents:
        agent.terminate()
        agent.join()

if __name__ == '__main__':
    gym_env()
