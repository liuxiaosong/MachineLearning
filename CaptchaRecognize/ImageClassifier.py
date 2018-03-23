#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'xiaosong Liu'

import os
import numpy as np
import matplotlib.pyplot as plt
from LogisticRegression import LR_Classifier
from sklearn.cross_validation import train_test_split

from CNN import CNN_Classifier
from ImagePreprocessing import load_images,image_show,image_expansion,image_grayscale


def image_aspect_ratio(image):
    img_height = float(image.shape[0])
    img_width = float(image.shape[1])

    return img_width / img_height

def image_pixel_sum(image):
    img_height = float(image.shape[0])
    img_width = float(image.shape[1])

    return img_width * img_height

def image_binarization_ratio(image):
    black_pixel_count = float(np.sum(image==0))
    pixel_count = float(image.shape[0] * image.shape[1])
    
    return  black_pixel_count / pixel_count

def load_train_samples():
    first_path = 'split_pic_temp'
    secend_path = '0123456789abcdefghijklmnopqrstuvwxyz'

    sample_vec = [(image, index) for index, path in enumerate(secend_path) 
                                 for image in load_images(os.path.join(first_path, path))]
    print("sample_count = %d" % len(sample_vec))                           
    return sample_vec


def data_preprocessing(images):
    img_expansion = [image_expansion(image_grayscale(image)) for image in images]
    #print(img_expansion[0].shape)
    x_vals = np.array([image.ravel() for image in img_expansion])
    # data normalization
    x_vals = np.array([np.where(x==0,255,0) for x in x_vals])

    #print(x_vals.shape)
    #print(y_vals.shape)
    return x_vals, img_expansion

def train_model(x_vals, y_vals):
    samples = load_train_samples()
    images, labels = zip(*samples)
    x_vals, _ = data_preprocessing(images,labels)
    y_vals = np.array(labels)

    model = LR_Classifier()
    model = LR_Classifier(learning_rate=0.01, training_epoch=3000, display_step=300)
    model.fit(x_vals,y_vals)

    return model

def prediction(model,test_images):
    test_vals, test_img_exp = data_preprocessing(test_images)
    predictions = model.pred(test_vals)

    return predictions

def accuracy(model, x_test, y_test):
    predictions = model.pred(x_test)
    accuracy_ = np.mean(np.equal(predictions,y_test).astype(np.float32))

    return accuracy_

def visualization_loss(x_vals, y_vals):
    plt.plot(x_vals, y_vals, 'k--', label='loss-values')
    plt.title('train-loss')
    plt.xlabel('training epoch counts')
    plt.ylabel('training loss values')
    plt.legend(loc='upper right')
    plt.show()

def pick_hyperparam(s,e):
    m = np.log10(s)
    n = np.log10(e)
    r = np.random.rand()
    r = m + (n-m)*r
    r = np.power(10, r)

    return r

def main():
    samples = load_train_samples()
    images, labels = zip(*samples)
    x_vals, _ = data_preprocessing(images)
    y_vals = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(x_vals, y_vals, test_size=0.2, random_state=0)

    #alpha, lambd = 0.01, 0.64
    for i in range(10):
        alpha = pick_hyperparam(0.009, 0.02)
        lambd = pick_hyperparam(0.5, 0.7)
        print('alpha = %0.4f , lambda = %0.4f' % (alpha, lambd))
        model = CNN_Classifier(learning_rate=alpha, epoches=50)
        #model = LR_Classifier(learning_rate=alpha, training_epoch=1200, display_step=240, regularization_term=lambd)
        #flag = model.load_model()
        flag = False
        model.fit(x_train,y_train,load_flag=flag)

        #评估模型对测试样本的泛化能力
        accuracy_ = accuracy(model, x_test, y_test)
        print('Accuracy = %.3f' % accuracy_)

    #可视化训练集的loss值
    #epoch, loss = zip(*model.loss_recoding)
    #visualization_loss(epoch, loss)

    #test_images = load_images('test_temp')
    #test_labels = 
    #predictions = prediction(model, x_test)
    #label = '0123456789abcdefghijklmnopqrstuvwxyz'
    #for image,index in zip(test_images,predictions):
    #    print(label[index], end='\t')
    #    image_show(image)


if __name__ == '__main__':
    main()   
#samples = load_samples()
#for image,label in samples[10:12]:
#    print(label)
#    print(image.shape)
#    image_show(image)

'''
img_vec = load_images('split_pic_temp')

img_aspect_ratio_vec = []
img_pixel_count_vec = []
img_binarization_ratio_vec = []

for image in img_vec:
    img_aspect_ratio = image_aspect_ratio(image)
    img_aspect_ratio_vec.append(img_aspect_ratio)
    img_pixel_count = image_pixel_sum(image)
    img_pixel_count_vec.append(img_pixel_count)
    img_binarization_ratio = image_binarization_ratio(image)
    img_binarization_ratio_vec.append(img_binarization_ratio)

plt.plot(img_pixel_count_vec, img_binarization_ratio_vec, 'ko', label='picel count vs aspect ratio ')
#plt.plot(class2_x, class2_y, 'kx', label='Class -1')
plt.title('image feature')
plt.xlabel('picel count')
plt.ylabel('aspect ratio')
plt.legend(loc='lower right')
plt.show()

'''

