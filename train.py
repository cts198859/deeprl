import numpy as np
import tensorflow as tf
from envs.wrapper import GymEnv
from agents.models import A2C


def explore(env, model, prev_ob, prev_done, cum_reward, cum_actions,
            mp_dict):
    ob = prev_ob
    done = prev_done
    return_reward, return_step = -1, -1
    for _ in range(model.n_step):
        if env.discrete:
            policy, value = model.forward(ob, done)
            action = np.random.choice(np.arange(len(policy)), p=policy)
        else:
            mu, std, value = model.forward(ob, done)
            action = env.sample_action(mu, std)
            policy = [mu, std]
        next_ob, reward, done, _ = env.step(action)
        cum_actions.append(action)
        cum_reward += reward
        global_step = mp_dict['global_counter'].next()
        model.add_transition(ob, action, reward, value, done)
        # logging
        if mp_dict['global_counter'].should_log():
            tf.logging.info('''thread %d, global step %d, episode step %d,
                               ob: %s, a: %.2f, pi: %s, v: %.2f, r: %.2f, done: %r''' %
                            (model.i_thread, global_step, len(cum_actions),
                             str(ob), action, str(policy), value, reward, done))
        # termination
        if done:
            ob = env.reset()
            return_reward, return_step = cum_reward, global_step
            cum_reward = 0
            cum_actions = []
        else:
            ob = next_ob
    if done:
        R = 0
    else:
        R = model.forward(ob, False, 'v')
    return ob, done, R, cum_reward, cum_actions, return_reward, return_step


def run_explore(n_s, n_a, total_step, i, model_config, is_discrete,
                env_name, seed, mp_dict, mp_ques):
    model = A2C(n_s, n_a, total_step, i_thread=i, model_config=model_config,
                discrete=is_discrete)
    env = GymEnv(env_name, is_discrete)
    env.seed(seed + i)
    ob = env.reset()
    done = False
    cum_reward = 0
    cum_actions = []
    model.reset()
    tf.logging.set_verbosity(tf.logging.INFO)
    try:
        while True:
            global_wts = mp_dict['global_wt']
            if global_wts is None:
                continue
            model.sync_wt(global_wts)
            ob, done, R, cum_reward, cum_actions, return_reward, return_step = \
                explore(env, model, ob, done, cum_reward, cum_actions, mp_dict)
            batch = model.sample_transition(R)
            mp_ques[0].put(batch)
            mp_ques[1].put((return_reward, return_step))
    except KeyboardInterrupt:
        pass
