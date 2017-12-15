import argparse
import configparser
import itertools
import numpy as np
import os
import signal
import tensorflow as tf
import threading

from agents.models import A2C
from envs.wrapper import GymEnv
from train import Trainer, AsyncTrainer

class GlobalCounter:
    def __init__(self, total_step, save_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_save_step = 0
        self.total_step = total_step
        self.save_step = save_step
        self.log_step = log_step

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_save(self):
        save = False
        if (self.cur_step - self.cur_save_step) >= self.save_step:
            save = True
            self.cur_save_step = self.cur_step
        return save

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        return (self.cur_step >= self.total_step)


def parse_args():
    default_config_path = '/Users/tchu/Documents/Uhana/remote/deeprl_signal_control/config.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=False,
                        default=default_config_path, help="config path")
    return parser.parse_args()


def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')


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


def gym_env():
    args = parse_args()
    parser = configparser.ConfigParser()
    parser.read(args.config_path)
    seed = parser.getint('TRAIN_CONFIG', 'SEED')
    num_env = parser.getint('TRAIN_CONFIG', 'NUM_ENV')
    env_name = parser.get('ENV_CONFIG', 'NAME')
    env = GymEnv(env_name)
    env.seed(seed)
    n_a = env.n_a
    n_s = env.n_s
    total_step = int(parser.getfloat('TRAIN_CONFIG', 'MAX_STEP'))
    base_dir = parser.get('TRAIN_CONFIG', 'BASE_DIR')
    save_step = int(parser.getfloat('TRAIN_CONFIG', 'SAVE_INTERVAL'))
    log_step = int(parser.getfloat('TRAIN_CONFIG', 'LOG_INTERVAL'))
    save_path, log_path = init_out_dir(base_dir)

    tf.set_random_seed(seed)
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    sess = tf.Session(config=config)
    global_model = A2C(sess, n_s, n_a, total_step, model_config=parser['MODEL_CONFIG'])
    saver = tf.train.Saver(max_to_keep=20)
    global_model.load(saver, save_path)
    global_counter = GlobalCounter(total_step, save_step, log_step)
    coord = tf.train.Coordinator()
    threads = []
    trainers = []

    def train_fn(i_thread):
        trainers[i_thread].run(sess, saver, coord)

    if num_env == 1:
        # regular training
        summary_writer = tf.summary.FileWriter(log_path, sess.graph)
        trainer = Trainer(env, global_model, save_path, summary_writer, global_counter)
        trainers.append(trainer)
    else:
        # asynchronous training
        lr_scheduler = global_model.lr_scheduler
        beta_scheduler = global_model.beta_scheduler
        optimizer = global_model.optimizer
        models = []
        # initialize model to update graph
        for i in range(num_env):
            models.append(A2C(sess, n_s, n_a, total_step, i_thread=i, optimizer=optimizer,
                              model_config=parser['MODEL_CONFIG']))
        summary_writer = tf.summary.FileWriter(log_path, sess.graph)
        for i in range(num_env):
            env = GymEnv(env_name)
            env.seed(seed + i)
            trainer = AsyncTrainer(env, models[i], save_path, summary_writer, global_counter,
                                   i, lr_scheduler, beta_scheduler)
            trainers.append(trainer)

    sess.run(tf.global_variables_initializer())

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
        global_model.save(saver, save_path + 'step', global_counter.cur_step)


if __name__ == '__main__':
    gym_env()
