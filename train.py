import numpy as np
import tensorflow as tf


class Trainer:
    def __init__(self, env, model, save_path, log_path, global_counter, i_thread=-1):
        self.cur_step = 0
        self.i_thread = i_thread
        self.global_counter = global_counter
        self.save_path = save_path
        self.env = env
        self.model = model
        self.n_step = self.model.n_step
        self._init_summary()
        self.summary_writer = tf.summary.FileWriter(log_path)

    def _init_summary(self):
        self.total_reward = tf.placeholder(tf.float32, [])
        self.actions = tf.placeholder(tf.int32, [None])
        summaries = []
        summaries.append(tf.summary.scalar('explore/total_reward', self.total_reward))
        summaries.append(tf.summary.histogram('explore/action', self.actions))
        self.summaries = tf.summary.merge(summaries)
        tf.logging.set_verbosity(tf.logging.INFO)

    def _add_summary(self, sess, cum_reward, cum_actions):
        summ = sess.run(self.summaries, {self.total_reward:cum_reward, self.actions:cum_actions})
        self.summary_writer.add_summary(summ, global_step=self.cur_step)

    def explore(self, sess, prev_ob, prev_done, cum_reward, cum_actions):
        ob = prev_ob
        done = prev_done
        for _ in range(self.n_step):
            policy, value = self.model.forward(ob, done)
            action = np.random.choice(np.arange(len(policy)), p=policy)
            next_ob, reward, done, _ = self.env.step(action)
            cum_actions.append(action)
            cum_reward += reward
            global_step = self.global_counter.next()
            self.cur_step += 1
            self.model.add_transition(ob, action, reward, value, done)
            # logging
            if self.global_counter.should_log():
                tf.logging.info('''thread %d, global step %d, local step %d, episode step %d,
                                   ob: %s, a: %d, pi: %s, v: %.2f, r: %.2f, done: %r''' %
                                (self.i_thread, global_step, self.cur_step, len(cum_actions),
                                 str(ob), action, str(policy), value, reward, done))
            # termination
            if done:
                ob = self.env.reset()
                self._add_summary(sess, cum_reward, cum_actions)
                cum_reward = 0
                cum_actions = []
            else:
                ob = next_ob
        if done:
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
            summ = self.model.backward(R)
            global_step = self.global_counter.cur_step
            self.summary_writer.add_summary(summ, global_step=global_step)
            self.summary_writer.flush()
            # save model
            if self.global_counter.should_save():
                print('saving model at step %d ...' % global_step)
                self.model.save(saver, self.save_path + 'step', global_step)
            if self.global_counter.should_stop():
                coord.request_stop()
                print('max step reached, press Ctrl+C to end program ...')
                return


class AsyncTrainer(Trainer):
    def __init__(self, env, model, lr_scheduler, beta_scheduler, summary_writer,
                 i_thread, global_counter, save_path):
        self.env = env
        self.model = model
        self.cur_step = 0
        self.i_thread = i_thread
        self.global_counter = global_counter
        self.save_path = save_path
        self.n_step = self.model.n_step
        self._init_summary()
        self.summary_writer = summary_writer
        self.lr_scheduler = lr_scheduler
        self.beta_scheduler = beta_scheduler

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
            summ = self.model.backward(R, cur_lr=cur_lr, cur_beta=cur_beta)
            global_step = self.global_counter.cur_step
            self.summary_writer.add_summary(summ, global_step=global_step)
            self.summary_writer.flush()
            # save model
            if self.global_counter.should_save():
                print('saving model at step %d ...' % global_step)
                self.model.save(saver, self.save_path + 'step', global_step)
            if self.global_counter.should_stop():
                coord.request_stop()
                print('max step reached, press Ctrl+C to end program ...')
                return
