import model
from data import x_train, y_train, vocab_size
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, help='Embedding size', default=16)
parser.add_argument('--lr', type=float, help='Learning Rate', default=0.1)
parser.add_argument('--iters', type=int, help='Number of interations', default=100)
option = parser.parse_args()

EMB = option.size
VOC = vocab_size

w2v = model.Word2Vec(VOC, EMB, True, x_train, y_train)
sess = tf.Session()

w2v.train(option.iters, option.lr, sess)

