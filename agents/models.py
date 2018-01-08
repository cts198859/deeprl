import os
from agents.utils import *
from agents.policies import *


class A2C:
    def __init__(self, sess, n_s, n_a, total_step, i_thread=-1, optimizer=None, lr=None,
                 model_config=None, discrete=True):
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
        reward_norm = model_config.getfloat('REWARD_NORM')
        n_step = model_config.getint('NUM_STEP')
        n_past = model_config.getint('NUM_PAST')
        n_fc = model_config.get('NUM_FC').split(',')
        n_fc = [int(x) for x in n_fc]

        if policy == 'lstm':
            n_lstm = model_config.getint('NUM_LSTM')
            self.policy = LstmPolicy(n_s, n_a, n_step, i_thread, n_past, n_fc=n_fc,
                                     n_lstm=n_lstm, discrete=discrete)
        elif policy == 'cnn1':
            n_filter = model_config.getint('NUM_FILTER')
            m_filter = model_config.getint('SIZE_FILTER')
            self.policy = Cnn1DPolicy(n_s, n_a, n_step, i_thread, n_past,
                                      n_fc=n_fc, n_filter=n_filter,
                                      m_filter=m_filter, discrete=discrete)
        self.name = self.policy.name
        self.policy.prepare_loss(optimizer, lr, v_coef, max_grad_norm, alpha, epsilon)

        if (i_thread == -1) and (total_step > 0):
            # global lr and entropy beta scheduler
            self.lr_scheduler = Scheduler(lr_init, lr_min, total_step, decay=lr_decay)
            self.beta_scheduler = Scheduler(beta_init, beta_min, total_step * beta_ratio,
                                            decay=beta_decay)
        self.trans_buffer = OnPolicyBuffer(gamma)

        def save(saver, model_dir, global_step):
            saver.save(sess, model_dir, global_step=global_step)

        def load(saver, model_dir, checkpoint=None):
            if i_thread == -1:
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
                    saver.restore(sess, model_dir + save_file)
                    print('checkpoint loaded: ', save_file)
                else:
                    print('could not find old checkpoint')

        def backward(R, cur_lr, cur_beta):
            obs, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(R, discrete)
            return self.policy.backward(sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta)

        def forward(ob, done, out_type='pv'):
            return self.policy.forward(sess, ob, done, out_type)

        def add_transition(ob, action, reward, value, done):
            if reward_norm:
                reward /= reward_norm
            self.trans_buffer.add_transition(ob, action, reward, value, done)

        self.save = save
        self.load = load
        self.backward = backward
        self.forward = forward
        self.n_step = n_step
        self.optimizer = self.policy.optimizer
        self.lr = self.policy.lr
        self.add_transition = add_transition
