from model.model import Word2Vec
from data.data import generate_batch
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--vocaSize', type=int, help='Vocabulary size', default=50000)
parser.add_argument('--size', type=int, help='Embedding size', default=128)
parser.add_argument('--lr', type=float, help='Learning Rate', default=0.1)
parser.add_argument('--iters', type=int, help='Number of interations', default=100000)
parser.add_argument('--checkpoint', type=int, help='Checkpoint', default=1000)
parser.add_argument('--resume', type=bool, help='Resume', default=False)
option = parser.parse_args()
EMB = option.size
VOC = option.vocaSize

w2v = Word2Vec(VOC, EMB, True, _getBatch=generate_batch, _resume=option.resume, _checkpoint=option.checkpoint)
sess = tf.Session()

w2v.train(option.iters, option.lr, sess)
