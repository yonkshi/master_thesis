import tensorflow as tf

from hparams import Hparams
from models import SymbolicSimulator

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()


sim = SymbolicSimulator(hp)

states = tf.zeros([hp.batch_size, 30, hp.d_states])
actions = tf.zeros([hp.batch_size, 30, hp.d_actions])
sim.train(states, actions, 100)
print("done!")