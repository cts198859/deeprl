import gym
import numpy as np


class GymEnv:
    def __init__(self, name, discrete=True):
        env = gym.make(name)
        self.discrete = discrete
        if discrete:
            self.n_a = env.action_space.n
        else:
            self.n_a = env.action_space.shape[0]
            # only single dimension is allowed
            assert(self.n_a == 1)
        self.n_s = env.observation_space.shape[0]
        s_min = np.array(env.observation_space.low)
        s_max = np.array(env.observation_space.high)
        s_mean, s_scale = .5 * (s_min + s_max), .5 * (s_max - s_min)
        if not discrete:
            a_min = np.array(env.action_space.low)[0]
            a_max = np.array(env.action_space.high)[0]
            a_mean, a_scale = .5 * (a_min + a_max), .5 * (a_max - a_min)

        def sample_action(mu, std):
            # only for continuous control
            action = np.random.randn() * std + mu
            return np.clip(action, -1, 1)

        def scale_ob(ob):
            return (np.array(ob) - s_mean) / s_scale

        def reset():
            return scale_ob(env.reset())

        def step(action):
            if not discrete:
                action = [action * a_scale + a_mean]
            ob, r, done, info = env.step(action)
            return scale_ob(ob), r, done, info

        self.seed = env.seed
        self.step = step
        self.reset = reset
        self.sample_action = sample_action
