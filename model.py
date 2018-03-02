import tensorflow as tf
import numpy as np

class Word2Vec:
    def __init__(self, _sizeVoc, _sizeEmb, _isTrain, _x=None, _y=None, _weights=None):
        self.sizeVoc = _sizeVoc
        self.sizeEmb = _sizeEmb
        self.isTrain = _isTrain
        self.weights = _weights

        self.x = _x
        self.y = _y

        self.build()

    def createWeights(self, name, shape):
        if (self.isTrain):
            return tf.Variable(tf.random_normal(shape=shape, name=name))
        else:
            return tf.constant(self.weights[name], name=name)


    def getWeigts(self, name):
        return tf.get_default_graph().get_tensor_by_name(name + ':0')

    def build(self):
        #Input and Output
        self.input = tf.placeholder(tf.float32, shape=[None, self.sizeVoc])
        self.label = tf.placeholder(tf.float32, shape=[None, self.sizeVoc])

        #Embedding
        self.w1 = self.createWeights('w1', [self.sizeVoc, self.sizeEmb])
        self.b1 = self.createWeights('b1', [self.sizeEmb])
        self.e1 = tf.add(tf.matmul(self.input, self.w1), self.b1)

        self.w2 = self.createWeights('w2', [self.sizeEmb, self.sizeVoc])
        self.b2 = self.createWeights('b2', [self.sizeVoc])
        self.e2 = tf.add(tf.matmul(self.e1, self.w2), self.b2)

        #Prediction
        self.predict = tf.nn.softmax(self.e2)

        #Loss
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(self.predict), reduction_indices=[1]))

    def train(self, numIters, sess):
        opt = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
        sess.run(tf.global_variables_initializer())
        for idx in xrange(numIters):
            sess.run(opt, feed_dict={self.input: self.x, self.label: self.y})
            print 'Step', idx, '. Loss:', sess.run(self.loss, feed_dict={self.input: self.x, self.label: self.y})

        self.save(sess)

    def save(self, sess):
        dict = {}
        dict['w1'] = sess.run(self.w1)
        dict['b1'] = sess.run(self.b1)
        dict['w2'] = sess.run(self.w2)
        dict['b2'] = sess.run(self.b2)
        np.save('weights.npy', dict)





