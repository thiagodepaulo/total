#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 19:14:42 2017

@author: thiagodepaulo
"""

from sklearn.model_selection import GridSearchCV    
from util import Loader
import logging
from pipes import pipe

## parameters and inputs
pipes_ops = ['my_preproc', 'tfidf_vec', 'mnb_cls']
s_dataset = '/exp/datasets/docs_rotulados/SyskillWebert-Parsed'        
parameters = {}

     
logger = logging.getLogger()
logger.setLevel(logging.INFO)
#logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load datasets
logging.info("loading dataset")
l = Loader()
d = l.from_files(s_dataset)
logging.info("done")

# create pipeline
text_clf = pipe(pipes_ops)

logging.info("intializing processing")    
# Grid Search
gs_clf = GridSearchCV(text_clf, parameters, cv=10, n_jobs=-1)
gs_clf = gs_clf.fit(d['corpus'], d['class_index'])
print(gs_clf.cv_results_)

    
        
