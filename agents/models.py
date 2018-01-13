import os
from agents.utils import *
from agents.policies import *


class A2C:
    def __init__(self, n_s, n_a, total_step, i_thread=-1, model_config=None, discrete=True):
        policy = model_config.get('POLICY')
        v_coef = model_config.getfloat('VALUE_COEF')
        max_grad_norm = model_config.getfloat('MAX_GRAD_NORM')
        alpha = model_config.getfloat('RMSP_ALPHA')
        epsilon = model_config.getfloat('RMSP_EPSILON')
        gamma = model_config.getfloat('GAMMA')
        reward_norm = model_config.getfloat('REWARD_NORM')
        n_step = model_config.getint('NUM_STEP')
        n_past = model_config.getint('NUM_PAST')
        n_fc = model_config.get('NUM_FC').split(',')
        n_fc = [int(x) for x in n_fc]
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

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
        # global agent updates wt while local agent updates observations
        if i_thread == -1:
            lr_init = model_config.getfloat('LR_INIT')
            lr_min = model_config.getfloat('LR_MIN')
            lr_decay = model_config.get('LR_DECAY')
            beta_init = model_config.getfloat('ENTROPY_COEF_INIT')
            beta_min = model_config.getfloat('ENTROPY_COEF_MIN')
            beta_decay = model_config.get('ENTROPY_DECAY')
            beta_ratio = model_config.getfloat('ENTROPY_RATIO')
            saver = tf.train.Saver(max_to_keep=20)
            self.lr_scheduler = Scheduler(lr_init, lr_min, total_step, n_step, decay=lr_decay)
            self.beta_scheduler = Scheduler(beta_init, beta_min, total_step * beta_ratio,
                                       n_step, decay=beta_decay)
            self.policy.prepare_loss(v_coef, max_grad_norm, alpha, epsilon)
            sess.run(tf.global_variables_initializer())
        else:
            trans_buffer = OnPolicyBuffer(gamma)
            self.policy.prepare_sync()

        def save(model_dir, global_step):
            assert(i_thread == -1)
            saver.save(sess, model_dir, global_step=global_step)

        def load(model_dir, checkpoint=None):
            assert(i_thread == -1)
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

        def forward(ob, done, out_type='pv'):
            return self.policy.forward(sess, ob, done, out_type)

        def backward(obs, A, dones, R, ADV, cur_lr, cur_beta, i):
            return self.policy.backward(sess, obs, A, dones, R, ADV,
                                        cur_lr, cur_beta, i=i)

        def sync_wt(wts):
            self.policy.sync_wt(sess, wts)

        def get_wt():
            return sess.run(self.policy.wts)

        def add_transition(ob, action, reward, value, done):
            if reward_norm:
                reward /= reward_norm
            trans_buffer.add_transition(ob, action, reward, value, done)

        def sample_transition(R):
            obs, A, dones, R, ADV = trans_buffer.sample_transition(R, discrete=discrete)
            batch = {}
            batch['obs'] = obs
            batch['A'] = A
            batch['dones'] = dones
            batch['R'] = R
            batch['ADV'] = ADV
            return batch

        self.i_thread = i_thread
        self.n_step = n_step
        self.save = save
        self.load = load
        self.sess = sess
        self.reset = self.policy.reset
        self.forward = forward
        self.backward = backward
        self.sync_wt = sync_wt
        self.get_wt = get_wt
        self.add_transition = add_transition
        self.sample_transition = sample_transition


def run_update(n_s, n_a, total_step, model_config, is_discrete,
               n_env, save_path, log_path, mp_dict, mp_list):
    model = A2C(n_s, n_a, total_step, model_config=model_config,
                discrete=is_discrete)
    model.load(save_path)
    model.policy.reset(n_env)
    mp_dict['global_wt'] = model.get_wt()
    summary_writer = tf.summary.FileWriter(log_path, model.sess.graph)
    total_reward = tf.placeholder(tf.float32, [])
    reward_summary = tf.summary.scalar('total_reward', total_reward)
    try:
        while True:
            global_counter = mp_dict['global_counter']
            for i in range(n_env):
                batch = mp_list[i][0].get()
                obs = batch['obs']
                dones = batch['dones']
                A = batch['A']
                ADV = batch['ADV']
                R = batch['R']
                cur_lr = model.lr_scheduler.get()
                cur_beta = model.beta_scheduler.get()
                summary = model.backward(obs, A, dones, R, ADV, cur_lr, cur_beta, i)
                for _ in range(len(A)):
                    global_counter.next()
                global_step = global_counter.cur_step
                summary_writer.add_summary(summary, global_step=global_step)
                reward, step = mp_list[i][1].get()
                if step > 0:
                    summ = model.sess.run(reward_summary, {total_reward:reward})
                    summary_writer.add_summary(summ, global_step=step)
                summary_writer.flush()
                mp_dict['global_wt'] = model.get_wt()
                if global_counter.should_save():
                    print('saving model at step %d ...' % global_step)
                    model.save(save_path + 'checkpoint', global_step)
            if global_counter.should_stop():
                print('max step reached ...')
                global_step = mp_dict['global_counter'].cur_step
                print('saving final model at step %d ...' % global_step)
                model.save(save_path + 'checkpoint', global_step)
                break
            mp_dict['global_counter'] = global_counter
    except KeyboardInterrupt:
        print('saving final model ...')
        model.save(save_path + 'checkpoint', total_step)
