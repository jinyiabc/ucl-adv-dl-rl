# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


#The central unit of data in TensorFlow is the tensor. A tensor consists of a set of primitive values shaped into an array of any number of dimensions. A tensor's rank is its number of dimensions, while its shape is a tuple of integers specifying the array's length along each dimension. Here are some examples of tensor values:



#3. # a rank 0 tensor; a scalar with shape [],
#[1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
#[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
#[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]


#a = tf.constant(3.0, dtype=tf.float32)
#b = tf.constant(4.0) # also tf.float32 implicitly
#total = a + b
#print(a)
#print(b)
#print(total)

#writer = tf.summary.FileWriter('.')
#writer.add_graph(tf.get_default_graph())
#writer.flush()

#sess = tf.Session()
#print(sess.run({'ab':(a, b), 'total':total}))


#vec = tf.random_uniform(shape=(3,))
#out1 = vec + 1
#out2 = vec + 2
#print(sess.run(vec))
#print(sess.run(vec))
#print(sess.run((out1, out2)))


#x = tf.placeholder(tf.float32)
#y = tf.placeholder(tf.float32)
#z = x + y
#
#
#print(sess.run(z, feed_dict={x: 3, y: 4.5}))
#print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))



#my_data = [
#    [0, 1,],
#    [2, 3,],
#    [4, 5,],
#    [6, 7,],
#]
#slices = tf.data.Dataset.from_tensor_slices(my_data)
#next_item = slices.make_one_shot_iterator().get_next()

#x = tf.placeholder(tf.float32, shape=[None, 3])
#linear_model = tf.layers.Dense(units=1)
#y = linear_model(x)
#
#init = tf.global_variables_initializer()
#sess.run(init)
#
#print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))

x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)

#print(sess.run(y_pred))

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
#print(sess.run(loss))
for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)

print(sess.run(y_pred))

a = tf.constant(np.array([[.1, .3, .5, .9]]))
print(sess.run(tf.nn.softmax(a)))
##[[ 0.16838508  0.205666    0.25120102  0.37474789]]
#sm = tf.nn.softmax(a)
#ce = cross_entropy(sm)

