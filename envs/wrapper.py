import gym
import numpy as np
import sys, os

TASK_ROOT_DIR=os.environ['TASK_ROOT_DIR']
sys.path.append(TASK_ROOT_DIR + '/image-based-lqr/')

from modified_LQR_env import *

class GymEnv:
    def __init__(self, name, discrete=True):
        
        if name == 'LQR':
            env = LQRCarEnv({})
        else:
            env = gym.make(name)

        self.discrete = discrete
        if discrete:
            self.n_a = env.action_space.n
        else:
            self.n_a = env.action_space.shape[0]
            # only single dimension is allowed
            # assert(self.n_a == 1)
        self.n_s = env.observation_space.shape[0]
        s_min = np.array(env.observation_space.low)
        s_max = np.array(env.observation_space.high)
        s_mean, s_scale = .5 * (s_min + s_max), .5 * (s_max - s_min)
        if not discrete:
            a_min = np.array(env.action_space.low)
            a_max = np.array(env.action_space.high)
            a_mean, a_scale = .5 * (a_min + a_max), .5 * (a_max - a_min)

        def scale_ob(ob):
            return (np.array(ob) - s_mean) / s_scale

        def reset():
            return scale_ob(env.reset())

        def step(action):
            if not discrete:
                action = action * a_scale + a_mean
            ob, r, done, info = env.step(action)
            return scale_ob(np.ravel(ob)), r, done, action

        self.seed = env.seed
        self.step = step
        self.reset = reset
