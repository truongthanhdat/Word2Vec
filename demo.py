import data
import model
import numpy as np
import tensorflow as tf

weights = np.load('weights.npy').item()

w2v = model.Word2Vec(data.vocab_size, 5, _isTrain=False, _weights=weights)

voca = 'he'

vec = np.zeros((1, data.vocab_size))
vec[0][data.word2int[voca]] = 1

sess = tf.Session()
sess.run(tf.global_variables_initializer())

predict = sess.run(w2v.predict, feed_dict={w2v.input: vec})

rank = np.argsort(predict[0])

for i in rank:
    print data.int2word[i]

