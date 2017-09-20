#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 19:14:42 2017

@author: thiagodepaulo
"""

#
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV    
from preprocessor import Preprocessor
from util import Loader
import logging
import sys

# create pipeline 
def pipe(ops):
    logging.info("creating pipeline")
    s_count_vec = 'count_vec'
    s_hash_vec = 'hash_vec'
    s_tfidf_vec = 'tfidf_vec'
    s_my_preproc = 'my_preproc'
    s_tfidf_trans = 'tfidf_trans'
    s_mnb_cls = 'mnb_cls'
    
    pipe_dic = {}    
    ####  Pré-Processing ##########################         
    pipe_dic[s_my_preproc] = Preprocessor() if s_my_preproc in ops else None
    pipe_dic[s_count_vec] = CountVectorizer() if s_count_vec in ops else None
    # n_features=2**20 (default value)
    pipe_dic[s_hash_vec] = HashingVectorizer() if s_hash_vec in ops else None
    pipe_dic[s_tfidf_vec] = TfidfVectorizer() if s_tfidf_vec in ops else None
    
    # transformer    
    pipe_dic[s_tfidf_trans] = TfidfTransformer() if s_tfidf_trans in ops else None

    ### Classification Algorithms #################
    pipe_dic[s_mnb_cls] = MultinomialNB() if s_mnb_cls in ops else None    
    
    return Pipeline([(key, pipe_dic[key]) for key in ops if pipe_dic[key] != None ])
        
logger = logging.getLogger()
logger.setLevel(logging.INFO)
#logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load datasets
logging.info("loading dataset")
s_dataset = '/exp/datasets/docs_rotulados/SyskillWebert-Parsed'        
l = Loader()
d = l.from_files(s_dataset)
logging.info("done")

# create pipeline
text_clf = pipe(['my_preproc', 'tfidf_vec', 'mnb_cls'])

parameters = {}

logging.info("intializing processing")    
# Grid Search
gs_clf = GridSearchCV(text_clf, parameters, cv=10, n_jobs=-1)
gs_clf = gs_clf.fit(d['corpus'], d['class_index'])
print(gs_clf.cv_results_)

    
        
    