#!/usr/bin/env python3
import argparse
import configparser
import numpy as np
import signal
import tensorflow as tf
import threading
import os

from agents.models import A2C, DDPG
from envs.wrapper import GymEnv
from envs.drone_wrapper import DroneEnv
from train import Trainer, AsyncTrainer, Evaluator
from utils import *

RL_RESULT_DIR = os.environ["RL_RESULT_DIR"]

def parse_args():
    default_config_path = '/Users/tchu/Documents/Uhana/remote/deeprl/config.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=False,
                        default=default_config_path, help="config path")
    parser.add_argument('--mode', type=str, required=False,
                        default='train', help="train or evaluate")
    parser.add_argument('--algo', type=str, required=False,
                        default='a2c', help="a2c, ddpg, ppo")
    return parser.parse_args()


def gym_train(parser, algo):
    seed = parser.getint('TRAIN_CONFIG', 'SEED')
    num_env = parser.getint('TRAIN_CONFIG', 'NUM_ENV')
    env_name = parser.get('ENV_CONFIG', 'NAME')
    is_discrete = parser.getboolean('ENV_CONFIG', 'DISCRETE')
    if parser.getboolean('ENV_CONFIG', 'ISDRONEENV'):
        env = DroneEnv(env_name, is_discrete)
    else:
        env = GymEnv(env_name, is_discrete)
    env.seed(seed)
    n_a = env.n_a
    n_s = env.n_s
    total_step = int(parser.getfloat('TRAIN_CONFIG', 'MAX_STEP'))
    base_dir = RL_RESULT_DIR + '/drone_with_RL_results/'
    save_step = int(parser.getfloat('TRAIN_CONFIG', 'SAVE_INTERVAL'))
    log_step = int(parser.getfloat('TRAIN_CONFIG', 'LOG_INTERVAL'))
    save_path, log_path = init_out_dir(base_dir, 'train')

    tf.set_random_seed(seed)
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    if algo == 'a2c':
        global_model = A2C(sess, n_s, n_a, total_step, model_config=parser['MODEL_CONFIG'],
                           discrete=is_discrete)
    elif algo == 'ddpg':
        assert(not is_discrete)
        global_model = DDPG(sess, n_s, n_a, total_step, model_config=parser['MODEL_CONFIG'])
    else:
        global_model = None
    global_counter = GlobalCounter(total_step, save_step, log_step)
    coord = tf.train.Coordinator()
    threads = []
    trainers = []
    model_summary = init_model_summary(global_model.name)

    if num_env == 1:
        # regular training
        summary_writer = tf.summary.FileWriter(log_path, sess.graph)
        trainer = Trainer(env, global_model, save_path, summary_writer, global_counter, model_summary)
        trainers.append(trainer)
    else:
        assert(algo == 'a2c')
        # asynchronous training
        lr_scheduler = global_model.lr_scheduler
        beta_scheduler = global_model.beta_scheduler
        optimizer = global_model.optimizer
        lr = global_model.lr
        models = []
        wt_summary = global_model.policy.summary
        reward_summary = None
        # initialize model to update graph
        for i in range(num_env):
            models.append(A2C(sess, n_s, n_a, total_step, i_thread=i, optimizer=optimizer,
                              lr=lr, model_config=parser['MODEL_CONFIG'], discrete=is_discrete))
        summary_writer = tf.summary.FileWriter(log_path, sess.graph)
        for i in range(num_env):
            env = GymEnv(env_name, is_discrete)
            env.seed(seed + i)
            trainer = AsyncTrainer(env, models[i], save_path, summary_writer, global_counter,
                                   i, lr_scheduler, beta_scheduler, model_summary, wt_summary,
                                   reward_summary=reward_summary)
            if i == 0:
                reward_summary = (trainer.reward_summary, trainer.total_reward)
            trainers.append(trainer)

    sess.run(tf.global_variables_initializer())
    global_model.init_train()
    saver = tf.train.Saver(max_to_keep=20)
    global_model.load(saver, save_path)

    def train_fn(i_thread):
        trainers[i_thread].run(sess, saver, coord)

    for i in range(num_env):
        thread = threading.Thread(target=train_fn, args=(i,))
        thread.start()
        threads.append(thread)
    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()
    coord.request_stop()
    coord.join(threads)
    save_flag = input('save final model? Y/N: ')
    if save_flag.lower().startswith('y'):
        print('saving model at step %d ...' % global_counter.cur_step)
        global_model.save(saver, save_path + 'checkpoint', global_counter.cur_step)


def gym_evaluate(parser, n_episode, algo):
    seed = parser.getint('TRAIN_CONFIG', 'SEED')
    env_name = parser.get('ENV_CONFIG', 'NAME')
    is_discrete = parser.getboolean('ENV_CONFIG', 'DISCRETE')
    if parser.getboolean('ENV_CONFIG', 'ISDRONEENV'):
        env = DroneEnv(env_name, is_discrete)
    else:
        env = GymEnv(env_name, is_discrete)
    env.seed(seed)
    n_a = env.n_a
    n_s = env.n_s
    sess = tf.Session()
    if algo == 'a2c':
        model = A2C(sess, n_s, n_a, -1, model_config=parser['MODEL_CONFIG'],
                    discrete=is_discrete)
    elif algo == 'ddpg':
        assert(not is_discrete)
        model = DDPG(sess, n_s, n_a, total_step, model_config=parser['MODEL_CONFIG'])
    else:
        model = None
    #base_dir = parser.get('TRAIN_CONFIG', 'BASE_DIR')
    base_dir = RL_RESULT_DIR + '/drone_with_RL_results/'
    save_path, log_path = init_out_dir(base_dir, 'evaluate')
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    model.load(saver, save_path)
    evaluator = Evaluator(env, model, log_path, n_episode)
    evaluator.run()

if __name__ == '__main__':
    args = parse_args()
    parser = configparser.ConfigParser()
    parser.read(args.config_path)
    if args.mode == 'train':
        gym_train(parser, args.algo)
    elif args.mode == 'evaluate':
        n_episode = int(input('evaluation episodes: '))
        gym_evaluate(parser, n_episode, args.algo)

