#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'xiaosong Liu'

import csv
import numpy as np
from hmmlearn import hmm
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import tldextract

#处理域名的最小长度
MIN_LEN = 5
#hmm模型
FILE_MODEL = 'hmm-white-900.m'

def load_alexa(filename):
	domain_list = []
	csv_reader = csv.reader(open(filename))
	for row in csv_reader:
		domain = row[1]
		if len(domain) >= MIN_LEN:
			domain_list.append(domain)
	return domain_list

def load_dga(filename):
	domain_list = []
	with open(filename,'r') as dga_file:
		for line in dga_file.readlines():
			domain = line.split(',')[0]
			if len(domain) >= MIN_LEN:
				domain_list.append(domain)
	return domain_list

def load_dga1(filename):
	domain_list = []
	with open(filename,'r') as dga_file:
		for domain in dga_file.readlines():
			if len(domain.strip('\n')) >= MIN_LEN:
				domain_list.append(domain.strip('\n'))
	return domain_list

def domain2vector(domain):
	vector = []
	ext = tldextract.extract(domain)
	if len(ext.domain) >= MIN_LEN:
		for char in ext.domain:
			if ord(char) >= ord('a') and ord(char) <= ord('z'):
				if char in 'aeiou':
					vector.append([1])
				else:
					vector.append([2])
			elif ord(char) >= ord('0') and ord(char) <= ord('9'):
				vector.append([3])
			else:
				vector.append([4])
	return np.array(vector) 

def train_hmm(domain_list):
	X = [[0]]
	X_lens = [1]
	for domain in domain_list:
		vector = domain2vector(domain)
		X = np.concatenate([X,vector])
		X_lens.append(len(vector))
		#print(X)
		#print(X_lens)
	remodel = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
	remodel.fit(X,X_lens)
	joblib.dump(remodel, FILE_MODEL)

def test_alexa(remodel,filename):
	x=[]
	y=[]
	alexa_list = load_alexa(filename)
	for domain in alexa_list:
		print(domain)
		vector = domain2vector(domain)
		if len(vector) :
			pro = remodel.score(vector)
			x.append(len(domain))
			y.append(pro)
	return x,y

def test_dga(remodel,filename):
	x=[]
	y=[]
	dga_list = load_dga1(filename)
	for domain in dga_list:
		vector = domain2vector(domain)
		if len(vector) :
			pro = remodel.score(vector)
			#x.append(len(domain))
			#y.append(pro)
			if pro >= 0 and pro <= 50 and len(domain)<= 32:
				x.append(domain)
				y.append(pro)
			#elif len(domain) <= 35:
			#x1.append(domain)
	#print(len(x))
	#print()	#y.append(pro)
	return x,y
def get_uniq_charnum(domain_list):
	x = []
	y = []
	for domain in domain_list:
		#x.append(len(domain))
		count = len(set(domain))
		count = (0.0+count)/len(domain)
		if count > 0.5 and len(domain) <= 32:
			x.append(domain)
		else:
			y.append(domain)
		#y.append(count)
	return x,y

def get_digit_freq(domain_list):
	x = []
	y = []
	for domain in domain_list:
		digit_count = 0
		for char in domain:
			if ord(char) >= ord('0') and ord(char) <= ord('9'):
				digit_count += 1
		digit_freq = (0.0+digit_count)/len(domain)
		x.append(len(domain))
		y.append(digit_freq)
	return x,y

def write_file(filename,dga_list):
	with open(filename,'a') as f:
		for domain in dga_list:
			f.write(domain+'\n')


def show_hmm():
	#remodel = joblib.load(FILE_MODEL)
	#x_1,y_1 = test_alexa(remodel,'alexa-top-1000.csv')
	#x_2,y_2 = test_dga(remodel,'dga-cryptolocke-1000.txt')
	#x_3,y_3 = test_dga(remodel,'dga-post-tovar-goz-1000.txt')
	alexa_list = load_alexa('alexa-top-1000.csv')
	dga_list = load_dga1('dga-post-tovar-goz-1000-p.txt')
	dga_list1 = load_dga1('dga_22.txt')
	#x_2,y_2 = get_uniq_charnum(domain_list)
	#x_2,y_2 = test_dga(remodel,'dga-domain.txt')
	x_1,y_1 = get_digit_freq(alexa_list)
	x_2,y_2 = get_digit_freq(dga_list)
	x_3,y_3 = get_digit_freq(dga_list1)
	fig,ax = plt.subplots()
	ax.set_xlabel('Domain Length')
	ax.set_ylabel('Digit Freq')
	ax.scatter(x_1,y_1,color='g',label='alexa-doamin',marker='o')
	ax.scatter(x_2,y_2,color='r',label='dga-domain',marker='*')
	ax.scatter(x_3,y_3,color='b',label='dga-domain',marker='^')
	ax.legend(loc='best')
	plt.show()

if __name__ == '__main__':
	#print(len(load_alexa('alexa-top-1000.csv')))
	#print(len(load_dga('dga-cryptolocke-1000.txt')))
	#print(domain2vector('alexa-top-1/000.csv'))
	#domain_list = load_alexa('alexa-top-1000.csv')
	dga_list = load_dga1('dga-post-tovar-goz-1000-p.txt')
	print(dga_list[0])
	#train_hmm(domain_list)
	#show_hmm()
	#dga_list = load_dga('dga-post-tovar-goz-1000.txt')
	#x,y = get_digit_freq(['baidu12.com','1we2vs45vd.com'])
	#print(x,y)
	#remodel = joblib.load(FILE_MODEL)
	#x_2,y_2 = test_dga(remodel,'dga-domain.txt')
	#domain_list = load_dga1('dga_2.txt')
	#x_2,y_2 = get_uniq_charnum(domain_list)

	#write_file('dga-post-tovar-goz-1000-p.txt',dga_list)
	#write_file('pro_22.txt',y_2)
