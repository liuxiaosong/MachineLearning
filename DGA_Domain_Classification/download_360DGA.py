#!/usr/bin/python 
# -*- coding:utf-8 -*-

__author__ = 'xiaosong_liu'

import requests
import os
import pandas as pd


dga_url = 'http://data.netlab.360.com/feeds/dga/dga.txt'
save_file_name = './360_dga.txt'
save_csv_name = './360_dga_100w.csv'

def download_file(url, filename):
	res = requests.get(url, stream=True)

	with open(filename, 'wb') as f:
		for chunk in res.iter_content(chunk_size=512):
			if chunk:
				f.write(chunk)

def load_file(filename):
	dga_df = pd.read_csv(filename,
						encoding='utf-8',
						header=None,
						sep='\t',
						names=['dga_class', 'dga_domain', 'start_time', 'end_time'],
						skiprows=18,
						skipfooter=1,
						index_col=None,
						engine='python')
	return dga_df

def dump_file(filename, df):
	df.to_csv(filename, encoding='utf-8')


#df = load_file(save_file_name)
#dump_file(save_csv_name, df[['dga_class', 'dga_domain']])

df = pd.read_csv(save_csv_name)
df_dga = df['dga_domain'].sample(n=100000)
df_dga.to_csv('360_dga_100k.csv')