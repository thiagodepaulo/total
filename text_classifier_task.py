#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 19:14:42 2017

@author: thiagodepaulo
"""

from sklearn.pipeline import Pipeline
from preprocessor import Preprocessor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from tempfile import mkdtemp
from sklearn.externals.joblib import Memory
from sklearn.model_selection import GridSearchCV    
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from util import Loader
import pandas as pd
import logging


def create_pipes(cache=False):
    steps = [('preprocessor',Preprocessor()), ('countvectorizer',CountVectorizer()), ('tfidftransformer',TfidfTransformer()), 
             ('clf',MultinomialNB())]
    memory=None
    if cache:
        cachedir = mkdtemp()
        memory = Memory(cachedir=cachedir, verbose=10)        
    return Pipeline(steps, memory=memory) 

def experiment(dataset, pipe, parameters, logger=logging.getLogger(), cv=10, out_file_csv='out.csv', 
               scoring = ['accuracy', 'f1_micro', 'f1_macro', 'f1_weighted']):
    logging.info("intializing processing")
    # Grid Search    
    gs_clf = GridSearchCV(pipe, parameters, cv=cv, n_jobs=-1, return_train_score=False, refit='accuracy', scoring=scoring)
    gs_clf = gs_clf.fit(d['corpus'], d['class_index'])
    #print(gs_clf.cv_results_)
    df = pd.DataFrame(gs_clf.cv_results_)
    df.to_csv(out_file_csv)
    print(df)            

#parameters
s_dataset = '/exp/datasets/docs_rotulados/SyskillWebert-Parsed'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
#logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load datasets
logging.info("loading dataset")
l = Loader()
d = l.from_files(s_dataset)
logging.info("done")    
    
parameters = { 'preprocessor':[None, Preprocessor(lang='english')],  'clf':[SVC(), MultinomialNB()] }

rcv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
cv = StratifiedKFold(n_splits=10, random_state=0)

experiment(d,create_pipes(), parameters, logger=logger, cv=cv)




    
        
    