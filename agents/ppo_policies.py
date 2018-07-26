import numpy as np
import tensorflow as tf
from agents.utils import *
from agents.a2c_policies import A2CPolicy
import bisect

def PPOPolicy(A2CPolicy):
    def __init__(self, n_a, n_s, n_step, n_past, discrete, imp_clip):
        super().__init__(n_a, n_s, n_step, n_past, discrete)
        self.imp_clip = imp_clip

    def _discrete_policy_loss(self):
        A_sparse = tf.one_hot(self.A, self.n_a)
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)
        entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
        policy_loss = -tf.reduce_mean(tf.reduce_sum(log_pi * A_sparse, axis=1) * self.ADV)
        return policy_loss, entropy_loss

    def _continuous_policy_loss(self):
        a_norm_dist = tf.contrib.distributions.Normal(self.pi[0], self.pi[1])
        log_prob = a_norm_dist.log_prob(tf.squeeze(self.A, axis=1))
        entropy_loss = -tf.reduce_mean(a_norm_dist.entropy()) * self.entropy_coef
        policy_loss = -tf.reduce_mean(log_prob * self.ADV)
        return policy_loss, entropy_loss

    def prepare_loss(self, optimizer, lr, v_coef, max_grad_norm, alpha, epsilon):
        if not self.discrete:
            self.A = tf.placeholder(tf.float32, [self.n_step, self.n_a])
        else:
            self.A = tf.placeholder(tf.int32, [self.n_step])
        self.ADV = tf.placeholder(tf.float32, [self.n_step])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.entropy_coef = tf.placeholder(tf.float32, [])
        # old v and pi values for clipping
        self.OLDV = 
        if self.discrete:
            policy_loss, entropy_loss = self._discrete_policy_loss()
        else:
            policy_loss, entropy_loss = self._continuous_policy_loss()
        value_loss = tf.reduce_mean(tf.square(self.R - self.v)) * 0.5 * v_coef
        self.loss = policy_loss + value_loss + entropy_loss

        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        if optimizer is None:
            # global policy
            self.lr = tf.placeholder(tf.float32, [])
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha,
                                                       epsilon=epsilon)
            self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        else:
            # local policy
            self.lr = lr
            self.optimizer = None
            global_name = self.name.split('_')[0] + '_' + str(-1)
            global_wts = tf.trainable_variables(scope=global_name)
            self._train = optimizer.apply_gradients(list(zip(grads, global_wts)))
            self.sync_wt = self._sync_wt(global_wts, wts)
        self.train_out = [entropy_loss, policy_loss, value_loss, self.loss, self.grad_norm]
