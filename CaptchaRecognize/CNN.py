#!/usr/bin/env python
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

class CNN_Classifier:
	def __init__(self,
				batch_size = 100,
				learning_rate = 0.005,
				epoches = 1000,
				display_step = 100):
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.enpoch = enpoches
		self.display_step = display_step

	def __preprocessing(self,Train_X, Train_Y):
		self.img_width = Train_X[0].shape[0]
		self.img_height = Train_X[0].shape[1]
		self.target_size = max(Train_Y) + 1
		self.channels = 1
		self.conv1_n = 25
		self.conv2_n = 50
		self.filter = 4
		self.max_pool = 2
		self.full_connect_size = 100
		
		x_input_shape = (None, self.img_width, self.img_height, self.channels)
		self.x_input = tf.placeholder(shape=x_input_shape, dtype=tf.float32)
		self.y_target = tf.placeholder(shape=None, dtype=tf.int32)

		self.conv1_filter = tf.Variable(tf.truncated_normal(shape=[self.filter,self.filter,self.channels,self.conv1_n], stddev=0.1, dtype=tf.float32))
		self.conv1_bias = tf.Variable(tf.zeros(shape=[self.conv1_n], dtype=tf.float32))
		self.conv2_filter = tf.Variable(tf.truncated_normal(shape=[self.filter,self.filter,self.conv1_n, self.conv2_n], stddev=0.1, dtype=tf.float32))
		self.conv2_bias = tf.Variable(tf.zeros(shape=[self.conv2_n], dtype=tf.float32))

		output_width = self.img_width // (self.max_pool * self.max_pool)
		output_height = self.img_height // (self.max_pool * self.max_pool)
		full_input_size = output_width * output_height * self.conv2_n

		self.full1_weight = tf.Variable(tf.truncated_normal(shape=[full_input_size, self.full_connect_size],
															stddev=0.1, dtype=tf.float32))
		self.full1_bias = tf.Variable(tf.truncated_normal(shape=[self.full_connect_size], stddev=0.1, dtype=tf.float32))

		self.full2_weight = tf.Variable(tf.truncated_normal(shape=[self.full_connect_size, self.target_size],
															stddev=0.1, dtype=tf.float32))
		self.full2_bias = tf.Variable(tf.truncated_normal(shape=[self.target_size], stddev=0.1, dtype=tf.float32))

	#没有传入tensor参数，时返回值是method类型，传入后变成tensor类型？
	#@lazy_property
	def conv_layer(self, input_layer, filter_, strides_=[1,1,1,1], padding_='SAME'):
		conv_out = tf.nn.conv2d(input_layer, filter=filter_, strides=strides_, padding=padding_)

		return conv_out

	#@lazy_property
	def relu_activation(self, input_layer, bias):
		relu_out = tf.nn.relu(tf.nn.bias_add(input_layer, bias))

		return relu_out

	#@lazy_property
	def max_pooling_layer(self, input_layer, ksize_=[1,2,2,1],
								strides_=[1,2,2,1], padding_='SAME'):
		max_pool_out = tf.nn.max_pool(input_layer, ksize=ksize_, strides=strides_, padding=padding_)

		return max_pool_out

	#@lazy_property
	def fully_connected_layer(self, input_layer, full_weight, full_bias):
		full_net_out = tf.add(tf.matmul(input_layer, full_weight), full_bias)

		return full_net_out

	@lazy_property
	def model_output(self):
		conv_layer1 = self.conv_layer(self.x_input, self.conv1_filter)
		conv_layer1 = self.relu_activation(conv_layer1, self.conv1_bias)
		max_pool_layer1 = self.max_pooling_layer(conv_layer1)

		conv_layer2 = self.conv_layer(max_pool_layer1, self.conv2_filter)
		conv_layer2 = self.relu_activation(conv_layer2, self.conv2_bias)
		max_pool_layer2 = self.max_pooling_layer(conv_layer2)

		# Transform Output into a 1xN layer for next fully connected layer
		final_conv_shape = max_pool_layer2.get_shape().as_list()
		final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
		final_output = tf.reshape(max_pool_layer2, [-1, final_shape])

		fully_connected_layer1 = self.fully_connected_layer(final_output, self.full1_weight, self.full1_bias)
		fully_connected_layer1 = tf.nn.relu(fully_connected_layer1)
		model_output = self.fully_connected_layer(fully_connected_layer1, self.full2_weight, self.full2_bias)

		return model_output

	@lazy_property
	def cost(self):
		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.model_output, labels=self.y_target))
		
		return loss

	@lazy_property
	def prediction(self):
		prediction_ = tf.argmax(tf.nn.softmax(self.model_output),axis=1)

		return prediction_

	def fit(self, Train_X, Train_Y):
		self.sess = tf.InteractiveSession()
		self.__preprocessing(Train_X, Train_Y)
		self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.cost)
		self.sess.run(tf.global_variables_initializer())
		self.loss_vec = []
		for i in range(self.enpoch):
			rand_index = np.random.choice(len(Train_X), size=self.batch_size)
			rand_x = Train_X[rand_index]
			rand_x = np.expand_dims(rand_x, 3)
			rand_y = Train_Y[rand_index]
			train_dict = {self.x_input: rand_x, self.y_target: rand_y}
			self.sess.run(self.optimizer, feed_dict=train_dict)
			temp_train_loss = self.sess.run(self.cost, feed_dict=train_dict)
			if (i+1) % self.display_step == 0:
				self.loss_vec.append(temp_train_loss)
				print('loss = ' + str(temp_train_loss))

	def pred(self, x_test):
		x_test = np.expand_dims(x_test, 3)
		pred_ = self.sess.run(self.prediction, feed_dict={self.x_input:x_test})

		return pred_

	def test(self, Train_X, Train_Y):
		self.sess = tf.InteractiveSession()
		self.__preprocessing(Train_X, Train_Y)
		self.sess.run(tf.global_variables_initializer())

		print(self.sess.run(tf.shape(self.conv1_filter)))

		# after first convolutional layer
		conv_layer1 = self.conv_layer(self.x_input, self.conv1_filter)
		result = self.sess.run(conv_layer1, feed_dict={self.x_input:Train_X, self.y_target:Train_Y})
		print("conv_layer1 = " + str(result.shape))

		# after first relu activaiton
		conv_layer1 = self.relu_activation(conv_layer1, self.conv1_bias)
		result = self.sess.run(conv_layer1, feed_dict={self.x_input:Train_X, self.y_target:Train_Y})
		print("relu_layer1 = " + str(result.shape))

		# after first max_pooling layer
		max_pool_layer1 = self.max_pooling_layer(conv_layer1)
		result = self.sess.run(max_pool_layer1, feed_dict={self.x_input:Train_X, self.y_target:Train_Y})
		print("max_pool_layer1 = " + str(result.shape))

		# after second convolutional layer
		conv_layer2 = self.conv_layer(max_pool_layer1, self.conv2_filter)
		result = self.sess.run(conv_layer2, feed_dict={self.x_input:Train_X, self.y_target:Train_Y})
		print("conv_layer2 = " + str(result.shape))

		# after second relu activation
		conv_layer2 = self.relu_activation(conv_layer2, self.conv2_bias)
		result = self.sess.run(conv_layer2, feed_dict={self.x_input:Train_X, self.y_target:Train_Y})
		print("relu_layer2 = " + str(result.shape))
		
		# after second max_pooling layer
		max_pool_layer2 = self.max_pooling_layer(conv_layer2)
		result = self.sess.run(max_pool_layer2, feed_dict={self.x_input:Train_X, self.y_target:Train_Y})
		print("max_pool_layer2 = " + str(result.shape))

		final_conv_shape = max_pool_layer2.get_shape().as_list()
		final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
		final_output = tf.reshape(max_pool_layer2, [-1, final_shape])
		print(final_output.shape)


if __name__ == '__main__':
	model = CNN_Classifier()
	model.test()



