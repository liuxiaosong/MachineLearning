#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'该模块生成n-gram'

__author__ = 'xiaosong Liu'


def bigrams(words):
    wprev = None
    for w in words:
        if not wprev==None:
            yield ''.join((wprev, w))
        wprev = w

def trigrams(words):
    wprev1 = None
    wprev2 = None
    for w in words:
        if not (wprev1==None or wprev2==None):
            yield ''.join((wprev1,wprev2, w))
        wprev1 = wprev2
        wprev2 = w

if __name__ == '__main__':
    pass
    #for i in trigrams('baidu.com'):
    #    print(i)

