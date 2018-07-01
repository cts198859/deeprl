import os
from agents.utils import *
from agents.a2c_policies import *
from agents.ddpg_policies import *


class A2C:
    #TODO: clean up A2C format
    def __init__(self, sess, n_s, n_a, total_step, i_thread=-1, optimizer=None, lr=None,
                 model_config=None, discrete=True):
        self.sess = sess
        policy = model_config.get('POLICY')
        v_coef = model_config.getfloat('VALUE_COEF')
        max_grad_norm = model_config.getfloat('MAX_GRAD_NORM')
        alpha = model_config.getfloat('RMSP_ALPHA')
        epsilon = model_config.getfloat('RMSP_EPSILON')
        lr_init = model_config.getfloat('LR_INIT')
        lr_min = model_config.getfloat('LR_MIN')
        lr_decay = model_config.get('LR_DECAY')
        beta_init = model_config.getfloat('ENTROPY_COEF_INIT')
        beta_min = model_config.getfloat('ENTROPY_COEF_MIN')
        beta_decay = model_config.get('ENTROPY_DECAY')
        beta_ratio = model_config.getfloat('ENTROPY_RATIO')
        gamma = model_config.getfloat('GAMMA')
        self.reward_norm = model_config.getfloat('REWARD_NORM')
        n_step = model_config.getint('NUM_STEP')
        n_past = model_config.getint('NUM_PAST')
        n_fc = model_config.get('NUM_FC').split(',')
        n_fc = [int(x) for x in n_fc]

        if policy == 'lstm':
            n_lstm = model_config.getint('NUM_LSTM')
            self.policy = A2CLstmPolicy(n_s, n_a, n_step, i_thread, n_past, n_fc=n_fc,
                                        n_lstm=n_lstm, discrete=discrete)
        elif policy == 'cnn1':
            n_filter = model_config.getint('NUM_FILTER')
            m_filter = model_config.getint('SIZE_FILTER')
            self.policy = A2CCnn1DPolicy(n_s, n_a, n_step, i_thread, n_past,
                                         n_fc=n_fc, n_filter=n_filter,
                                         m_filter=m_filter, discrete=discrete)
        self.name = 'a2c'
        self.policy.prepare_loss(optimizer, lr, v_coef, max_grad_norm, alpha, epsilon)

        if (i_thread == -1) and (total_step > 0):
            # global lr and entropy beta scheduler
            self.lr_scheduler = Scheduler(lr_init, lr_min, total_step, decay=lr_decay)
            self.beta_scheduler = Scheduler(beta_init, beta_min, total_step * beta_ratio,
                                            decay=beta_decay)
        self.trans_buffer = OnPolicyBuffer(gamma)
        self.n_step = n_step
        self.optimizer = self.policy.optimizer
        self.lr = self.policy.lr
        self.discrete = discrete
        self.i_thread = i_thread

    def init_train(self):
        pass

    def save(self, saver, model_dir, global_step):
        saver.save(self.sess, model_dir, global_step=global_step)

    def load(self, saver, model_dir, checkpoint=None):
        if self.i_thread == -1:
            save_file = None
            save_step = 0
            if os.path.exists(model_dir):
                if checkpoint is None:
                    for file in os.listdir(model_dir):
                        if file.startswith('checkpoint'):
                            prefix = file.split('.')[0]
                            tokens = prefix.split('-')
                            if len(tokens) != 2:
                                continue
                            cur_step = int(tokens[1])
                            if cur_step > save_step:
                                save_file = prefix
                                save_step = cur_step
                else:
                    save_file = 'checkpoint-' + str(int(checkpoint))
            if save_file is not None:
                saver.restore(self.sess, model_dir + save_file)
                print('checkpoint loaded: ', save_file)
            else:
                print('could not find old checkpoint')

    def backward(self, R, cur_lr, cur_beta):
        obs, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(R, self.discrete)
        return self.policy.backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta)

    def forward(self, ob, done, mode='pv'):
        return self.policy.forward(self.sess, ob, done, mode)

    def add_transition(self, ob, action, reward, value, done):
        if self.reward_norm:
            reward /= self.reward_norm
        self.trans_buffer.add_transition(ob, action, reward, value, done)


class DDPG(A2C):
    def __init__(self, sess, n_s, n_a, total_step, model_config=None, i_thread=-1):
        self.name = 'ddpg'
        self.reward_norm = model_config.getfloat('REWARD_NORM')
        self.v_coef = model_config.getfloat('VALUE_COEF')
        self.n_update = model_config.getint('NUM_UPDATE')
        self.n_warmup = model_config.getfloat('WARMUP_STEP')
        self.n_step = model_config.getint('NUM_STEP')
        self._init_policy(n_s, n_a, model_config)
        self.sess = sess
        self.total_step = total_step
        self.i_thread = i_thread
        if self.total_step > 0:
            self._init_scheduler(model_config)
            self._init_train(model_config)

    def add_transition(self, ob, action, reward, next_ob, done):
        if self.reward_norm:
            reward /= self.reward_norm
        self.trans_buffer.add_transition(ob, action, reward, next_ob, done)

    def backward(self, cur_lr):
        if self.trans_buffer.size < self.n_batch:
            return [np.nan] * self.policy.train_out_num
        if self.trans_buffer.size < self.n_warmup:
            warmup = True
        else:
            warmup = False
        lr_actor, lr_critic = cur_lr, cur_lr * self.v_coef
        obs, acts, next_obs, rs, dones = self.trans_buffer.sample_transition()
        # summary: loss_v, loss_p, loss, grad_norm_v, grad_norm_p
        summary = []
        for _ in range(self.n_update):
            val = self.policy.backward(self.sess, obs, acts, next_obs, dones, rs,
                                       lr_critic, lr_actor, warmup=warmup)
            summary.append(val)
        return list(np.mean(np.array(summary), axis=0))

    def forward(self, ob, mode='explore'):
        return self.policy.forward(self.sess, ob, mode=mode)

    def init_train(self):
        self.sess.run(self.policy.init_target)

    def reset_noise(self):
        self.policy.reset_noise(self.sess)

    def _init_policy(self, n_s, n_a, model_config):
        if ('OU_THETA' in model_config) and ('OU_SIGMA' in model_config):
            theta = model_config.getfloat('OU_THETA')
            sigma = model_config.getfloat('OU_SIGMA')
            noise_generator = OUNoise(theta=theta, sigma=sigma)
        else:
            noise_generator = OUNoise()
        policy_name = model_config.get('POLICY')
        n_batch = model_config.getint('BATCH_SIZE')
        n_fc = model_config.get('NUM_FC').split(',')
        n_fc = [int(x) for x in n_fc]
        self.n_batch = n_batch
        if policy_name == 'fc':
            self.policy = DDPGFCPolicy(n_s, n_a, n_batch, n_fc, noise_generator)
        else:
            self.policy = None

    def _init_scheduler(self, model_config):
        lr_init = model_config.getfloat('LR_INIT')
        lr_decay = model_config.get('LR_DECAY')
        if lr_decay == 'constant':
            self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        else:
            lr_min = model_config.getfloat('LR_MIN')
            self.lr_scheduler = Scheduler(lr_init, lr_min, self.total_step,
                                          decay=lr_decay)

    def _init_train(self, model_config):
        max_grad_norm = model_config.getfloat('MAX_GRAD_NORM')
        gamma = model_config.getfloat('GAMMA')
        tau = model_config.getfloat('TAU')
        if 'L2_ACTOR' in model_config:
            l2_actor = model_config.getfloat('L2_ACTOR')
        else:
            l2_actor = 0
        if 'L2_CRITIC' in model_config:
            l2_critic = model_config.getfloat('L2_CRITIC')
        else:
            l2_critic = 0
        self.policy.prepare_loss(self.v_coef, l2_actor, l2_critic, gamma, tau, max_grad_norm)
        buffer_size = model_config.getfloat('BUFFER_SIZE')
        self.trans_buffer = ReplayBuffer(buffer_size, self.n_batch)
