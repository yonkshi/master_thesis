import tensorflow as tf
from model import *

EPOCHS = 100
def build_train_model(obs_t0, action, obs_t1, reward):

    # Building the model
    M = build_maneuver_network(action)
    O = build_symbol_net(obs_t0)
    O_t1, reward_out = build_simulator(O, M)

    # Observation at T1
    O_truth_t1 = build_symbol_net(obs_t1)


    # Building the loss
    L_symbol = tf.nn.sigmoid_cross_entropy_with_logits(O_truth_t1, O_t1)
    L_reward = tf.nn.l2_loss(reward - reward_out)

    loss = L_symbol + L_reward

    optimizer = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(loss)

    return optimizer, L_symbol, L_reward, loss

def train():

    obs_t0 = np.random.randn((200,200, 3))
    obs_t1 = np.random.randn()
    reward = np.random.uniform(-1, 1)
    action = np.random.shuffle([0,0,0,1,0])

    optmizer, L_symbol, L_reward, loss = build_train_model(obs_t0, action, obs_t1, reward)


    for i in range(EPOCHS):
        pass
