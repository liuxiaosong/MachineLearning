#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'xiaosong Liu'

import tensorflow as tf
import numpy as np
import functools

def lazy_property(function):
	attribute = '_' + function.__name__

	@property
	@functools.wraps(function)
	def wrapper(self):
		if not hasattr(self, attribute):
			setattr(self, attribute, function(self))
		return getattr(self, attribute)

	return wrapper

class LR_Classifier:
	def __init__(self,
				learning_rate = 0.01,
				batch_size = 100,
				training_epoch = 1000,
				display_step = 100):

		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.training_epoch = training_epoch
		self.display_step = display_step

	def __Preprocessing(self, train_X, train_Y):
		self.row = train_X.shape[0]
		self.col = train_X.shape[1]
		self.num_class = train_Y.shape[1]

		self.X = tf.placeholder(shape=[None, self.col], dtype=tf.float32)
		self.Y = tf.placeholder(shape=[None, self.num_class], dtype=tf.float32)
		self.A = tf.Variable(tf.random_normal(shape=[self.col, self.num_class]))
		self.b = tf.Variable(tf.random_normal(shape=[1,1]))
		self.model_output = tf.add(tf.matmul(self.X, self.A), self.b)

	def __dense_to_one_hot(self, labels_dense):
		num_classes = len(np.unique(labels_dense))
		rows_labels = labels_dense.shape[0]
		index_offset = np.arange(rows_labels) * num_classes
		labels_one_hot = np.zeros((rows_labels, num_classes))
		labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

		return labels_one_hot

	@lazy_property
	def Cost(self):
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.model_output, labels=self.Y))
		return loss

	@lazy_property
	def Prediction(self):
		prediction = tf.argmax(tf.sigmoid(self.model_output),1)
		return prediction

	def fit(self, train_X, train_Y):
		train_Y = self.__dense_to_one_hot(train_Y)
		self.sess = tf.InteractiveSession()
		self.__Preprocessing(train_X, train_Y)
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.Cost)
		self.sess.run(tf.global_variables_initializer())

		for epoch in range(self.training_epoch):
			self.sess.run(self.optimizer, feed_dict={self.X:train_X, self.Y:train_Y})
			if epoch % self.display_step == 0:
				loss = self.sess.run(self.Cost, feed_dict={self.X:train_X, self.Y:train_Y})
				print ('epoch= ', epoch, ' loss= ',loss)

	def pred(self, test_X):
		prediction = self.sess.run(self.Prediction, feed_dict={self.X:test_X})
		return prediction