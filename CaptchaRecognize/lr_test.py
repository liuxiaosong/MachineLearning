#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'xiaosong Liu'


import numpy as np
from sklearn import datasets
from LogisticRegression import LR_Classifier

iris = datasets.load_iris()
x_vals = iris.data
y_vals = iris.target


model = LR_Classifier(learning_rate=0.01, training_epoch=2000, display_step=200)
model.fit(x_vals,y_vals)
predictions = model.pred(x_vals)

Accuracy = np.mean(np.equal(predictions,y_vals).astype(np.float32))
print('Accuracy = %.3f' % Accuracy)