import model
from data import x_train, y_train, vocab_size
import tensorflow as tf

EMB = 5
VOC = vocab_size

w2v = model.Word2Vec(VOC, EMB, True, x_train, y_train)
sess = tf.Session()

w2v.train(100000, sess)

