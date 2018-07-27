import numpy as np
import tensorflow as tf
import pandas as pd
from utils import plot_episode

class Trainer:
    def __init__(self, env, model, save_path, summary_writer, global_counter, 
                 model_summary, i_thread=-1):
        self.cur_step = 0
        self.i_thread = i_thread
        self.global_counter = global_counter
        self.save_path = save_path
        self.env = env
        self.model = model
        self.algo = self.model.name
        self.n_step = self.model.n_step
        self._init_env_summary()
        self._init_model_summary(model_summary)
        self.summary_writer = summary_writer

    def _init_model_summary(self, model_summary):
        self.model_summaries = model_summary[0]
        self.policy_loss = model_summary[1]
        self.value_loss = model_summary[2]
        self.total_loss = model_summary[3]
        self.lr = model_summary[4]
        self.gradnorm = model_summary[5]
        if self.algo in ['a2c', 'ppo']:
            self.entropy_loss = model_summary[6]
            self.beta = model_summary[7]
            if self.algo == 'ppo':
                self.policy_kl = model_summary[8]
                self.clip_rate = model_summary[9]
        elif self.algo == 'ddpg':
            self.gradnorm_v = model_summary[6]
        
    def _init_env_summary(self):
        self.total_reward = tf.placeholder(tf.float32, [])
        self.actions = tf.placeholder(tf.int32, [None])
        summaries = []
        summaries.append(tf.summary.scalar('total_reward', self.total_reward))
        summaries.append(tf.summary.histogram('explore', self.actions))
        self.env_summaries = tf.summary.merge(summaries)
        tf.logging.set_verbosity(tf.logging.INFO)

    def _add_env_summary(self, sess, cum_reward, cum_actions, global_step):
        summ = sess.run(self.env_summaries, {self.total_reward:cum_reward, self.actions:cum_actions})
        self.summary_writer.add_summary(summ, global_step=global_step)

    def _add_model_summary(self, sess, policy_loss, value_loss,
                           total_loss, gradnorm, cur_lr, global_step, extras=[]):
        if self.algo == 'a2c':
            summ = sess.run(self.model_summaries, {self.entropy_loss: extras[0],
                                                   self.policy_loss: policy_loss,
                                                   self.value_loss: value_loss,
                                                   self.total_loss: total_loss,
                                                   self.lr: cur_lr,
                                                   self.beta: extras[1],
                                                   self.gradnorm: gradnorm})
        elif self.algo == 'ppo':
            summ = sess.run(self.model_summaries, {self.entropy_loss: extras[0],
                                                   self.policy_loss: policy_loss,
                                                   self.value_loss: value_loss,
                                                   self.total_loss: total_loss,
                                                   self.lr: cur_lr,
                                                   self.beta: extras[1],
                                                   self.gradnorm: gradnorm,
                                                   self.policy_kl: extras[2],
                                                   self.clip_rate: extras[3]})
        elif self.algo == 'ddpg':
            summ = sess.run(self.model_summaries, {self.gradnorm_v: extras[0],
                                                   self.policy_loss: policy_loss,
                                                   self.value_loss: value_loss,
                                                   self.total_loss: total_loss,
                                                   self.lr: cur_lr,
                                                   self.gradnorm: gradnorm})
        self.summary_writer.add_summary(summ, global_step=global_step)

    def explore(self, sess, prev_ob, prev_done, cum_reward, cum_actions):
        ob = prev_ob
        done = prev_done
        for _ in range(self.n_step):
            if self.env.discrete:
                if self.algo == 'a2c':
                    action, policy, value = self.model.forward(ob, done)
                elif self.algo == 'ppo':
                    action, policy, value, logprob = self.model.forward(ob, done)
            else:
                if self.algo in ['a2c', 'ppo']:
                    # TODO: relax this to multiple actions
                    if self.algo == 'a2c':
                        action, mu, std, value = self.model.forward(ob, done)
                    else:
                        action, mu, std, value, logprob = self.model.forward(ob, done)
                    policy = [mu, std]
                elif self.algo == 'ddpg':
                    action = self.model.forward(ob)
                    policy = 'n/a'
                    value = np.nan
            next_ob, reward, done, _ = self.env.step(action)
            # TODO: remove action?
            cum_actions.append(action[0])
            cum_reward += reward
            global_step = self.global_counter.next()
            self.cur_step += 1
            if self.algo == 'a2c':
                self.model.add_transition(ob, action, reward, value, done)
            elif self.algo == 'ppo':
                self.model.add_transition(ob, action, reward, value, done, logprob)
            elif self.algo == 'ddpg':
                self.model.add_transition(ob, action, reward, next_ob, done)
            # logging
            if self.global_counter.should_log():
                tf.logging.info('''thread %d, global step %d, local step %d, episode step %d,
                                   ob: %s, a: %r, pi: %s, v: %.2f, r: %.2f, done: %r''' %
                                (self.i_thread, global_step, self.cur_step, len(cum_actions),
                                 str(ob), action, str(policy), value, reward, done))
            # termination
            if done:
                ob = self.env.reset()
                if self.algo == 'ddpg':
                    self.model.reset_noise()
                self._add_env_summary(sess, cum_reward, cum_actions, global_step)
                cum_reward = 0
                cum_actions = []
            else:
                ob = next_ob
        if done or (self.algo == 'ddpg'):
            R = 0
        else:
            R = self.model.forward(ob, False, 'v')
        return ob, done, R, cum_reward, cum_actions

    def run(self, sess, saver, coord):
        ob = self.env.reset()
        done = False
        cum_reward = 0
        cum_actions = []
        while not coord.should_stop():
            ob, done, R, cum_reward, cum_actions = self.explore(sess, ob, done, cum_reward, cum_actions)
            cur_lr = self.model.lr_scheduler.get(self.n_step)
            if self.algo == 'a2c':
                cur_beta = self.model.beta_scheduler.get(self.n_step)
                entropy_loss, policy_loss, value_loss, total_loss, gradnorm = \
                    self.model.backward(R, cur_lr, cur_beta)
                extras = [entropy_loss, cur_beta]
            elif self.algo == 'ppo':
                cur_beta = self.model.beta_scheduler.get(self.n_step)
                cur_clip = self.model.clip_scheduler.get(self.n_step)
                entropy_loss, policy_loss, value_loss, total_loss, gradnorm, policy_kl, clip_rate = \
                    self.model.backward(R, cur_lr, cur_beta, cur_clip)
                extras = [entropy_loss, cur_beta, policy_kl, clip_rate]
            elif self.algo == 'ddpg':
                value_loss, policy_loss, total_loss, gradnorm_v, gradnorm = \
                    self.model.backward(cur_lr)
                extras = [gradnorm_v]
            global_step = self.global_counter.cur_step
            self._add_model_summary(sess, policy_loss, value_loss, total_loss, gradnorm,
                                    cur_lr, global_step, extras=extras)
            # do not show wt details in tensorboard
            # summ = sess.run(self.model.policy.summary)
            # self.summary_writer.add_summary(summ, global_step=global_step)
            self.summary_writer.flush()
            # save model
            if self.global_counter.should_save():
                print('saving model at step %d ...' % global_step)
                self.model.save(saver, self.save_path + 'checkpoint', global_step)
            if self.global_counter.should_stop():
                coord.request_stop()
                print('max step reached, press Ctrl+C to end program ...')
                return


class AsyncTrainer(Trainer):
    def __init__(self, env, model, save_path, summary_writer, global_counter,
                 i_thread, lr_scheduler, beta_scheduler, model_summary, wt_summary,
                 reward_summary=None, clip_scheduler=None):
        self.cur_step = 0
        self.i_thread = i_thread
        self.global_counter = global_counter
        self.save_path = save_path
        self.env = env
        self.model = model
        self.algo = self.model.name
        self.n_step = self.model.n_step
        self.lr_scheduler = lr_scheduler
        self.beta_scheduler = beta_scheduler
        self.clip_scheduler = clip_scheduler
        self.summary_writer = summary_writer
        self._init_env_summary(reward_summary, i_thread)
        self._init_model_summary(model_summary)
        self.wt_summary = wt_summary

    def _init_env_summary(self, reward_summary, i_thread):
        if reward_summary is None:
            self.total_reward = tf.placeholder(tf.float32, [])
            self.reward_summary = tf.summary.scalar('total_reward', self.total_reward)
        else:
            self.reward_summary, self.total_reward = reward_summary
        self.actions = tf.placeholder(tf.int32, [None])
        self.action_summary = tf.summary.histogram('explore/' + str(i_thread), self.actions)
        tf.logging.set_verbosity(tf.logging.INFO)

    def _add_env_summary(self, sess, cum_reward, cum_actions, global_step):
        summ = sess.run(self.reward_summary, {self.total_reward:cum_reward})
        self.summary_writer.add_summary(summ, global_step=global_step)
        summ = sess.run(self.action_summary, {self.actions:cum_actions})
        self.summary_writer.add_summary(summ, global_step=global_step)

    def run(self, sess, saver, coord):
        ob = self.env.reset()
        done = False
        cum_reward = 0
        cum_actions = []
        while not coord.should_stop():
            sess.run(self.model.policy.sync_wt)
            ob, done, R, cum_reward, cum_actions = self.explore(sess, ob, done, cum_reward, cum_actions)
            cur_lr = self.lr_scheduler.get(self.n_step)
            cur_beta = self.beta_scheduler.get(self.n_step)
            if self.algo == 'a2c':
                entropy_loss, policy_loss, value_loss, total_loss, gradnorm = \
                    self.model.backward(R, cur_lr, cur_beta)
                extras = [entropy_loss, cur_beta]
            elif self.algo == 'ppo':
                cur_clip = self.clip_scheduler.get(self.n_step)
                entropy_loss, policy_loss, value_loss, total_loss, gradnorm, policy_kl, clip_rate = \
                    self.model.backward(R, cur_lr, cur_beta, cur_clip)
                extras = [entropy_loss, cur_beta, policy_kl, clip_rate]
            global_step = self.global_counter.cur_step
            self._add_model_summary(sess, policy_loss, value_loss,
                                    total_loss, gradnorm, cur_lr, global_step, extras=extras)
            # summ = sess.run(self.wt_summary)
            # self.summary_writer.add_summary(summ, global_step=global_step)
            self.summary_writer.flush()
            # save model
            if self.global_counter.should_save():
                print('saving model at step %d ...' % global_step)
                self.model.save(saver, self.save_path + 'checkpoint', global_step)
            if (self.global_counter.should_stop()) and (not coord.should_stop()):
                coord.request_stop()
                print('max step reached, press Ctrl+C to end program ...')
                return

class Evaluator:
    def __init__(self, env, model, log_path, n_episode):
        self.env = env
        self.model = model
        self.algo = self.model.name
        self.log_path = log_path
        self.n = n_episode

    def perform(self, run):
        ob = self.env.reset()
        done = False
        actions = []
        rewards = []
        states = []
        while True:
            states.append(ob)
            if self.env.discrete:
                _, policy = self.model.forward(ob, done, mode='p')
                action = np.argmax(policy)
            else:
                if self.algo in ['a2c', 'ppo']:
                    _, mu, _ = self.model.forward(ob, done, mode='p')
                    action = np.clip(mu, -1, 1)
                elif self.algo == 'ddpg':
                    action = self.model.forward(ob, mode='act')
            next_ob, reward, done, raw_action = self.env.step(action)
            actions.append(raw_action)
            rewards.append(reward)
            if done:
                break
            ob = next_ob
        actions = np.array(actions)
        rewards = np.array(rewards)
        states = np.array(states)
        plot_episode(actions, states, rewards, run, self.log_path)
        return states, actions, rewards

    def run(self):
        df_ls = []
        total_rewards = []
        for i in range(self.n):
            states, actions, rewards = self.perform(i)
            total_rewards.append(np.sum(rewards))
            cur_dict = {'action': str(actions), 'reward': rewards,
                        'run': np.ones(len(actions)) * i}
            for j in range(states.shape[1]):
                cur_dict['state_%d' % j] = states[:, j]
            df = pd.DataFrame(cur_dict)
            df_ls.append(df)
        total_rewards = np.array(total_rewards)
        print('total reward mean: %.2f, std: %.2f' %
              (np.mean(total_rewards), np.std(total_rewards)))
        df = pd.concat(df_ls)
        df.to_csv(self.log_path + '/data.csv')
