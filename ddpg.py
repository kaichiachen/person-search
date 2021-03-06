import tensorflow as tf
import numpy as np
import os
import sys
import shutil
from env import Env
import time

np.random.seed(0)
tf.set_random_seed(0)

MAX_EPISODES = 1000
MAX_EP_STEPS = 5
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
GAMMA = 1.1  # reward discount
REPLACE_ITER_A = 1100
REPLACE_ITER_C = 1000
MEMORY_CAPACITY = 5000
KNOWLEDGE_EPOCH = 300
BATCH_SIZE = 16
VAR_MIN = 0.05
RENDER = False
LOAD = False

env = Env()
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
ACTION_BOUND = env.action_bound

with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')
with tf.name_scope('KA'):
    KA = tf.placeholder(tf.float32, shape=[None, 5], name='ka')
    
    
class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)
            
        self.loss = self.loss = tf.reduce_mean(tf.squared_difference(KA, self.a))
        self.train_k_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 1024, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 1024, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.sigmoid, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1
    def learn_knowledge(self, s, a):
        a = np.array(a)
        s = s[np.newaxis, :]
        a = a[np.newaxis, :]
        loss, _ = sess.run([self.loss, self.train_k_op], feed_dict={S:s, KA:a})
        return loss

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))
            
            
class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 1024
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 1024, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1
        return loss
        
        
class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

sess = tf.Session()
sess.__enter__()

actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

saver = tf.train.Saver()
path = './output/models'

if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())
    
def train():
    var = 1  # control exploration
    max_step = 5.
    last_epoch = 0
    
    import shutil
    shutil.rmtree('./logs/ddpg')
    shutil.rmtree('./output/imgs')
    shutil.rmtree('./output/models')
    os.mkdir('./logs/ddpg')
    os.mkdir('./output/imgs')
    os.mkdir('./output/models')
    writer = tf.summary.FileWriter("./logs/ddpg", sess.graph)
    index = 0
    while True:
        summary = tf.Summary()
        index += 1
        max_step = max([max_step*.99999, MAX_EP_STEPS])
        epoch, s = env.reset(int(max_step))
#         if epoch > MAX_EPISODES:
#             break
        ep_reward = 0
        actions = [[] for _ in range(int(max_step))]
        start = time.time()
        for t in range(int(max_step)):
            
            if epoch <= KNOWLEDGE_EPOCH:
                a = env.continuous_action_knowledge()
                loss = actor.learn_knowledge(s, a)
                actor_loss_value = summary.value.add()
                actor_loss_value.simple_value = loss
                actor_loss_value.tag = 'actor_loss
                aa = actor.choose_action(s)
            else:
                a = actor.choose_action(s)
                a = np.clip(np.random.normal(a, var), *ACTION_BOUND)
                
            s_, r, done, info = env.step(a, t+1)
            iou = info['iou']
            actions[t] = list(a)
            
            M.store_transition(s, a, r, s_)

            if M.pointer > MEMORY_CAPACITY: # start learning after 50 epoch
                var = max([var*.9999, VAR_MIN])    # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]

                loss = critic.learn(b_s, b_a, b_r, b_s_)
                critic_loss_value = summary.value.add()
                critic_loss_value.simple_value = loss
                critic_loss_value.tag = 'critic_loss'
                if epoch > KNOWLEDGE_EPOCH:
                    actor.learn(b_s)

            s = s_
            ep_reward += r*GAMMA
            
            #actions[t] = [info['x'],info['y'],info['width'],info['height'], iou]
            result = '| done' if done else '| ----'
            if epoch <= KNOWLEDGE_EPOCH:
                a = aa
            #sys.stdout.write('\r' + 
            print('' + 
                  'Epoch:'+ str(epoch) +
                  ' (%d/%d)' % (index, len(env.random_index)) +
                  result +
                  '| Step: %i' % (t+1) +
                  '| R: %.2f' % ep_reward +
                  '| IOU: %.3f' % iou +
                  '| INFO: %s, %s, %s, %s, %f' % (str(info['x']),str(info['y']),str(info['width']),str(info['height']),a[-1]) +          
                  '| Time: %.2f' % (time.time()-start) + 
                  '| Zero State: %.2f' % (len(np.where(s==0)[0])/(7*7*512))
                  )
            
            if t == int(max_step)-1 or done:
                if iou > 0.5:
                    env.save()
                break
        if RENDER:
            env.render()
        
        if epoch != last_epoch:
            index = 0
            last_epoch = epoch
            if epoch % 100 == 0:
                ckpt_path = os.path.join(path, 'DDPG_epoch_%d.ckpt' % epoch)
                save_path = saver.save(sess, ckpt_path, write_meta_graph=True)
                print("\nSave Model %s\n" % save_path)
                
        for i, action in enumerate(actions):
            if len(action) == 0:
                break
            for ii, a in enumerate(action):
                action_value = summary.value.add()
                action_value.simple_value = a
                action_value.tag = '%d_%d_action' % (i, ii)
        reward_value = summary.value.add()
        reward_value.simple_value = ep_reward
        reward_value.tag = 'reward'
        iou_value = summary.value.add()
        iou_value.simple_value = iou
        iou_value.tag = 'iou'
        iou_diff_value = summary.value.add()
        iou_diff_value.simple_value = iou - env.init_iou
        iou_diff_value.tag = 'iou_diff'
        step_value = summary.value.add()
        step_value.simple_value = t+1
        step_value.tag = 'step'
        var_value = summary.value.add()
        var_value.simple_value = var
        var_value.tag = 'var'
        epoch_value = summary.value.add()
        epoch_value.simple_value = epoch
        epoch_value.tag = 'epoch'
        writer.add_summary(summary, len(env.random_index)*epoch + index)


def eval():
    env.set_fps(30)
    s = env.reset()
    while True:
        if RENDER:
            env.render()
        a = actor.choose_action(s)
        s_, r, done = env.step(a)
        s = s_

if __name__ == '__main__':
    if LOAD:
        eval()
    else:
        train() 
