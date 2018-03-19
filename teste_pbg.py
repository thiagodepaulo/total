#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:57:32 2018

@author: thiagodepaulo
"""
from pbg import PBG
from util import Loader
from sklearn.pipeline import Pipeline
from preprocessor import Preprocessor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# carrega matrix
s_dataset = '/exp/datasets/docs_rotulados/SyskillWebert-Parsed'
l = Loader()
d = l.from_files(s_dataset)

cvect = CountVectorizer()
steps = [('preprocessor',Preprocessor()), ('countvectorizer',cvect)]
pipe = Pipeline(steps) 
pipe.fit(d['corpus'])
M = pipe.transform(d['corpus'])

r_pbg=True
model=None

if r_pbg :
    pbg = PBG(50, alpha=0.5, beta=0.001, local_max_itr=10, 
              global_max_itr=10, local_threshold = 1e-6, global_threshold = 1e-6, 
              max_time=18000, save_interval=-1, out_dir='.', out_A='A', out_B='B', calc_q=False, debug=False)
    pbg.fit(M)
    pbg.transform(M)
    model=pbg
else:
     lda = LatentDirichletAllocation(n_components=50, learning_method='batch')
     lda.fit(M)
     lda.transform(M)
     model = lda
     

def get_top_words(model, feature_names, n_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return topics

topics = get_top_words(model, cvect.get_feature_names(), 10)
for topic in topics:
    print(', '.join(topic))





#pbg.bgp(M, Mcsc, W, D, A, B)

	