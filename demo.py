from data.data import dictionary, reverse_dictionary
from model.model import Word2Vec
import numpy as np
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--vocaSize', type=int, help='Vocabulary size', default=50000)
parser.add_argument('--embSize', type=int, help='Embedding size', default=128)
parser.add_argument('-k', type=int, help='Number of retrieval', default=10)
parser.add_argument('--word', type=str, help='Word', default='he')
parser.add_argument('--weights', type=str, help='Weights path', default='weights/weights_final.npy')
option = parser.parse_args()

BATCH_SIZE = 128

def dist(u, v):
    return np.sum((u - v) ** 2)

def find(word):
    weights = np.load(option.weights).item()
    w2v = Word2Vec(option.vocaSize, option.embSize, _isTrain=False, _weights=weights)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    sentinal = np.zeros((1, option.vocaSize), dtype=np.float32)
    sentinal[0][dictionary[word]] = 1
    out_sentinal = sess.run(w2v.e1, feed_dict={w2v.input: sentinal})

    first = 0
    last = BATCH_SIZE
    distance = np.zeros(option.vocaSize, dtype=np.float32)
    while (first < option.vocaSize):
        input = np.zeros((BATCH_SIZE, option.vocaSize), dtype=np.float32)
        for i in xrange(first, last):
            input[i - first][i] = 1
        output = sess.run(w2v.e1, feed_dict={w2v.input: input})

        for i in xrange(first, last):
            distance[i] = dist(out_sentinal[0], output[i - first])

        first = first + BATCH_SIZE
        last = last + BATCH_SIZE
        if last > option.vocaSize:
            last = option.vocaSize
    return distance

if __name__ == '__main__':
    distance = find(option.word)
    rank = np.argsort(distance)
    print 'Top %d words similar with %s' % (option.k, option.word)
    for i in rank[:option.k]:
        print reverse_dictionary[i]

