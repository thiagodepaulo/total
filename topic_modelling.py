#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 15:41:51 2017

@author: thiagodepaulo
"""

from sklearn.pipeline import Pipeline
from preprocessor import Preprocessor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tempfile import mkdtemp
from sklearn.externals.joblib import Memory
from util import Loader

def create_pipes(cache=False):
    steps = [('preprocessor',Preprocessor()), ('countvectorizer',CountVectorizer()), ('clf',LatentDirichletAllocation())]
    memory=None
    if cache:
        cachedir = mkdtemp()
        memory = Memory(cachedir=cachedir, verbose=10)        
    return Pipeline(steps, memory=memory) 

def get_top_words(model, feature_names, n_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return topics


#parameters
s_dataset = '/exp/datasets/Discursos/discurso_*'
l = Loader()
d = l.from_files_2(s_dataset)

preproc = Preprocessor(lang='portuguese')
d = preproc.transform(d)

vect = CountVectorizer()
X = vect.fit_transform(d)

lda = LatentDirichletAllocation(n_components=100, learning_method='batch', max_iter=50)
lda.fit(X)
feature_names = vect.get_feature_names()
topics = get_top_words(lda, feature_names, 10)
print(topics)








