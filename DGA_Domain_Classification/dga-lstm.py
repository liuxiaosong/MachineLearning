#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'xiaosong Liu'

import pandas as pd
import os
import numpy as np
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from lstm_model import LSTM_Model

MAX_FEATURE_LEN = 10000
VECTORIZER_FILE = './tmp/vectorizer-10k.pkl'
SEQUENCE_LEN = 15

def load_examples():
	'''
		加载正负样本集

	'''
	negetive_file = './alexa_100k.txt'
	postive_file = './360_dga_100k.csv'
	negetive_df = pd.read_csv(negetive_file, header=None)
	negetive_df.columns = ['domain']
	postive_df = pd.read_csv(postive_file, header=None, index_col=0)
	postive_df.columns = ['domain']
	return negetive_df, postive_df

def data_clean(df):
	'''
		清洗异常数据或空值
	'''
	df.dropna(axis=0)
	[df['domain'].str.len() < 3]== None
	df.dropna(axis=0)
	
	return df

def bulid_vectorizer(df, max_features_len):
	'''
		使用2-gram创建词袋生成器
	'''
	print('创建词袋生成器')
	vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2), max_features=max_features_len)
	vectorizer.fit(df['domain'].astype(str))
	#print(vectorizer.vocabulary_)
	joblib.dump(vectorizer, VECTORIZER_FILE)

	return vectorizer

def ngram_tokenier(str, n):
	'''
		创建2-gram 分词器
	'''
	for ix, char in enumerate(str):
		start_ix = ix
		end_ix = ix + n
		if end_ix <= len(str):
			yield ix, str[start_ix:end_ix]

def feature_vectorization(vectorizer, df, seq_len):
	'''
		将文本型特征转换为数值型特征
	'''
	print('特征向量化')
	#feature_matrix = vectorizer.transform(df['domain'].astype(str))
	x_data = np.zeros((len(df['domain']), seq_len))

	vocab_dict = vectorizer.vocabulary_
	# 生成序列长为seq_len的索引数组， 不够补0，超过的截断
	for row_ix, domain in enumerate(df['domain'].astype(str)):
		for (col_ix, gram) in ngram_tokenier(domain, 2):
			if (gram in vocab_dict) and (col_ix<seq_len):
				x_data[row_ix, col_ix] = vocab_dict[gram]
	return x_data

def main():
	neg_df, pos_df = load_examples()
	neg_df = data_clean(neg_df)
	neg_df['class'] = 0
	pos_df['class'] = 1
	neg_pos_df = pd.concat([neg_df, pos_df], axis=0)

	if os.path.exists(VECTORIZER_FILE):
		vectorizer = joblib.load(VECTORIZER_FILE)
	else:
		vectorizer = bulid_vectorizer(neg_pos_df, max_features_len=MAX_FEATURE_LEN)

	vocab_size = len(vectorizer.vocabulary_)

	x_data = feature_vectorization(vectorizer, neg_pos_df, SEQUENCE_LEN)
	
	shuffled_index = np.random.permutation(np.arange(len(x_data)))
	print(vocab_size)
	print(x_data[:10])

	#获取矩阵中每行向量中值为1的索引，组成词索引列表
	#x_index = [[i for i,v in enumerate(x) if v==1] for x in x_matrix]

	#x_data = np.zeros((len(x_index), SEQUENCE_LEN))
	# 生成序列长为30的索引数组， 不够补0，超过的截断
	#for i, x in enumerate(x_index):
	#	for j, v in enumerate(x[:SEQUENCE_LEN]):
	#		x_data[i,j] = v

	y_data = neg_pos_df['class'].as_matrix()

	#return x_data, y_data, vocab_size


	#x_data, y_data, VOCAB_SIZE = data_transform()

	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

	#创建lstm模型
	model = LSTM_Model(batch_size=1000, learning_rate=0.001, sequence_len=SEQUENCE_LEN, rnn_size=64,
				   vocab_size=vocab_size, epoches=5, embedding_size=64, keep_prob=0.75)
	model.train(x_train, y_train)
	y_pred, accuracy = model.pred(x_test, y_test)
	print(accuracy)

if __name__ == '__main__':
	main()



