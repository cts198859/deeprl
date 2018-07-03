import os
import sys
import numpy as np

PATH_TO_DEEPCUT_CODE = os.environ["DEEPCUT_SRC_DIR"]
PATH_TO_DEEPCUT_RES = os.environ["DEEPCUT_RESULT_DIR"]

sys.path.insert(0, PATH_TO_DEEPCUT_CODE)
import drone_environment
import simple_rmse

BANDWIDTH_FILE_NAME = PATH_TO_DEEPCUT_RES + "/traces/bandwidth_trace_exp2.csv"
IMAGE_FILE_NAME = PATH_TO_DEEPCUT_RES +  "/traces/image_trace_exp2.csv"

class DroneEnv:
    def __init__(self, name, discrete=True):
        bg = drone_environment.BandwidthGenerator(BANDWIDTH_FILE_NAME)
        dg = drone_environment.DataGenerator(IMAGE_FILE_NAME)
        nn = simple_rmse.model()
        env = drone_environment.DroneEnvironment(bg, dg, nn, discrete)
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
            return scale_ob(ob), r, done, action

        def get_results_df():
            return env.results_df

        self.seed = env.seed
        self.step = step
        self.reset = reset
        self.get_results_df = get_results_df

