import tensorflow as tf

from modules import multihead_attention, ff


def state_attention():
    pass

def simulator():
    pass

def decoder():
    pass


class SymbolicSimulator:
    def __init__(self, hp):
        self.hp = hp

    def encode_states(self, s_tilda, training=True):
        '''
        Attention module that encodes low-level-high-dim states into low-dim symbolic states
        :param s_tilda: batched continous states
        :param training:
        :return:
        '''
        ## Blocks
        for i in range(self.hp.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # self-attention
                s_tilda = multihead_attention(queries=s_tilda,
                                          keys=s_tilda,
                                          values=s_tilda,
                                          num_heads=self.hp.num_heads,
                                          dropout_rate=self.hp.dropout_rate,
                                          training=training,
                                          causality=False)
                # feed forward
                s_tilda = ff(s_tilda, num_units=[self.hp.d_states_ff, self.hp.d_states])

        return s_tilda

    def encode_actions(self, a_tilda, training=True):
        '''
        Attention module that encodes low-level-high-dim actions into low-dim symbolic action
        :param a_tilda: batched continous states
        :param training:
        :return:
        '''
        ## Blocks
        for i in range(self.hp.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # self-attention
                a_tilda = multihead_attention(queries=a_tilda,
                                          keys=a_tilda,
                                          values=a_tilda,
                                          num_heads=self.hp.num_heads,
                                          dropout_rate=self.hp.dropout_rate,
                                          training=training,
                                          causality=False)
                # feed forward
                a_tilda = ff(a_tilda, num_units=[self.hp.d_actions_ff, self.hp.d_actions])

        return a_tilda

    def simulator(self, a_tilda, training=True):
        '''
        Main NN for simulation. Actions (Semantic) in, State & Rewards (Semantic) out
        :param a_tilda:
        :param training:
        :return:
        '''
        a_star = self.encode_actions(a_tilda)


        # Inner layer
        a_star = tf.layers.dense(a_star, self.hp.d_actions, activation=tf.nn.relu)
        # Outer layer
        s_star = tf.layers.dense(a_star, self.hp.d_states)

        return s_star


    def train(self, s_tilda, a_tilda, rewards,):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        s_star = self.encode_states(s_tilda)
        s_hat = self.simulator(self.encode_actions(a_tilda))

        # train scheme
        loss = tf.nn.l2_loss(s_star-s_hat)

        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(self.hp.lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', global_step)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries