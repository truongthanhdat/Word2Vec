import tensorflow as tf
import numpy as np
import os

class Word2Vec:
    def __init__(self, _sizeVoc, _sizeEmb, _isTrain, _getBatch=None, _weights=None, _resume=False, _checkpoint=10000):
        self.sizeVoc = _sizeVoc
        self.sizeEmb = _sizeEmb
        self.isTrain = _isTrain
        self.weights = _weights
        self.getBatch = _getBatch
        self.resume = _resume
        self.checkpoint = _checkpoint
        self.start = 0

        if self.resume:
            self.start = self.findStart()
            if self.start > 0:
                self.weights = np.load('weights/weights_%d.npy' % self.start).item()
            else:
                self.resume = False

        self.build()

        if self.isTrain:
            tf.summary.scalar("Loss", self.loss)
            self.summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter('logs/', graph=tf.get_default_graph())

    def createWeights(self, name, shape):
        if self.isTrain:
            if self.resume:
                return tf.Variable(self.weights[name], name=name)
            else:
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
        #self.loss = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(self.predict), reduction_indices=[1]))
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.label, logits=self.e2)

    def findStart(self):
        if self.resume:
            listDir = os.listdir('weights')
            if len(listDir) == 0:
                return 0
            else:
                start = 0
                for file in listDir:
                    chk = file[8:][:-4]
                    if chk == 'final':
                        return -1
                    else:
                        if int(chk) > start:
                            start = int(chk)
                return start
        else:
            return 0

    def train(self, numIters, lr, sess):
        opt = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)
        sess.run(tf.global_variables_initializer())

        for idx in xrange(self.start, numIters):
            x, y = self.getBatch(batch_size=128, num_skips=2, skip_window=1)
            _, summary, loss = sess.run([opt, self.summary_op, self.loss], feed_dict={self.input: x, self.label: y})
            self.summary_writer.add_summary(summary, idx)
            print 'Step', idx, '. Loss:', loss
            if ((idx + 1) % self.checkpoint) == 0:
                self.save(sess, idx)
        self.save(sess)

    def save(self, sess, idx=None):
        dict = {}
        dict['w1'] = sess.run(self.w1)
        dict['b1'] = sess.run(self.b1)
        dict['w2'] = sess.run(self.w2)
        dict['b2'] = sess.run(self.b2)
        if idx:
            np.save('weights/weights_%d.npy' % (idx + 1), dict)
        else:
            np.save('weights/weights_final.npy', dict)






