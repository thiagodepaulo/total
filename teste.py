#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:15:43 2017

@author: thiagodepaulo
"""

from util import Loader
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from imbhn import IMBHN
from sklearn.model_selection import cross_val_score


s_dataset = '/exp/datasets/docs_rotulados/SyskillWebert-Parsed'
# Load datasets
l = Loader()
d = l.from_files(s_dataset)
count = CountVectorizer()
corpus = count.fit_transform(d['corpus'])
y = np.array(d['class_index'])

clf = IMBHN(max_itr=20)
#clf.fit(corpus,y)
print("oi")
print(clf.get_params())
scores = cross_val_score(clf, corpus, y, cv=10)
print(scores)

