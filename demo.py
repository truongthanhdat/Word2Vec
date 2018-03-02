import data
import model
import numpy as np
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, help='Embedding size', default=16)
parser.add_argument('-k', type=int, help='Number of retrieval', default=10)
parser.add_argument('--input', type=str, help='Word', default='he')
option = parser.parse_args()

weights = np.load('weights.npy').item()

w2v = model.Word2Vec(data.vocab_size, option.size, _isTrain=False, _weights=weights)

voca = option.input.lower()

vec = np.zeros((1, data.vocab_size))
vec[0][data.word2int[voca]] = 1

sess = tf.Session()
sess.run(tf.global_variables_initializer())

predict = sess.run(w2v.predict, feed_dict={w2v.input: vec})

rank = np.argsort(predict[0])

for i in rank[:option.k]:
    print data.int2word[i]

