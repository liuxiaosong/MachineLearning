#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'xiaosong Liu'

import tensorflow as tf
import numpy as np

model_file_name = './model/lstm-model'

class LSTM_Model:
	def __init__(self, batch_size=100, learning_rate=0.001, rnn_size=30, sequence_len=30,
					embedding_size=30, vocab_size=5000, epoches=50, keep_prob=0.8):
		self.batch_size = batch_size
		self.rnn_size = rnn_size
		self.learning_rate = learning_rate
		self.sequence_len = sequence_len
		self.embedding_size = embedding_size
		self.vocab_size = vocab_size
		self.epoches = epoches
		self.keep_prob = keep_prob

		self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
		self.initial_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)

		self.x_data = tf.placeholder(tf.int32, [self.batch_size, self.sequence_len])
		self.y_output = tf.placeholder(tf.int32, [self.batch_size])

		self.W = tf.Variable(tf.truncated_normal([self.rnn_size, 2], stddev=0.1), name='W')
		self.b = tf.Variable(tf.constant(0.1, shape=[2]), name='b')

		self.sess = tf.Session()

	def model_output(self):
		#定义词嵌套
		with tf.variable_scope('logits_output', reuse=tf.AUTO_REUSE) as scope:
			embedding_mat = tf.get_variable('embedding_mat', [self.vocab_size, self.embedding_size],
											tf.float32, tf.random_normal_initializer())

			embedding_output = tf.nn.embedding_lookup(embedding_mat, self.x_data)
			output, state = tf.nn.dynamic_rnn(self.lstm_cell, embedding_output, dtype=tf.float32)
			output = tf.nn.dropout(output, self.keep_prob)

			output = tf.transpose(output, [1, 0, 2]) #转换为[train_sequence_len, batch_size, embedding_size]
			last_output = tf.gather(output, int(output.get_shape()[0]) - 1) #获取序列的最后输出结果
			logits_output = tf.nn.bias_add(tf.matmul(last_output, self.W), self.b)

			return logits_output

	def cost(self):
		losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.model_output(), labels=self.y_output)
		loss = tf.reduce_mean(losses)

		return loss

	def prediction(self):
		preds = tf.nn.softmax(self.model_output())

		return tf.argmax(preds, 1)

	def generate_batch(self, x_data, y_data):
		shuffled_index = np.random.permutation(np.arange(len(x_data)))
		x_data = x_data[shuffled_index]
		y_data = y_data[shuffled_index]

		num_batches = int(len(x_data)/self.batch_size)

		for i in range(num_batches):
			min_index = i * self.batch_size
			max_index = np.min([len(x_data), ((i+1) * self.batch_size)])
			x_train_batch = x_data[min_index:max_index]
			y_train_batch = y_data[min_index:max_index]
			yield x_train_batch, y_train_batch

	def train(self, x_train, y_train):
		loss = self.cost()
		optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)

		init = tf.global_variables_initializer()
		self.sess.run(init)

		self.saver = tf.train.Saver()

		print("start training....")

		for epoch in range(self.epoches):
			
			for x_train_batch, y_train_batch in self.generate_batch(x_train, y_train):
				train_dict = {self.x_data:x_train_batch, self.y_output:y_train_batch}
				self.sess.run(optimizer, feed_dict=train_dict)

			temp_loss = self.sess.run(loss, feed_dict=train_dict)
			self.saver.save(self.sess, model_file_name, global_step = epoch)

			print('epoch %d : loss = %f' % (epoch, temp_loss))


	def pred(self, x_test, y_test):
		accu_list = []
		for x_test_batch, y_test_batch in self.generate_batch(x_test, y_test):
			temp = self.sess.run(self.prediction(), feed_dict={self.x_data:x_test_batch})
			accu = np.sum((np.array(temp) == y_test_batch).astype(int))/len(y_test_batch)
			print('test_accuracy= {0:.4f}'.format(accu))
			accu_list.append(temp)

		accuracy = np.mean(accu_list)

		return pred_list, accuracy
