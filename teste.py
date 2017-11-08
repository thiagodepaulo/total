#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:15:43 2017

@author: thiagodepaulo
"""
from util import Loader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import check_X_y
import numpy as np

s_dataset = '/exp/datasets/docs_rotulados/SyskillWebert-Parsed'
# Load datasets
l = Loader()
d = l.from_files(s_dataset)

count = CountVectorizer()
corpus = count.fit_transform(d['corpus'])

y = np.array(d['class_index'])

print(corpus.shape)