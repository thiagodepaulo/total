#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:41:19 2017

@author: thiagodepaulo
"""
from text_classifier_task import experiment, create_pipes
from preprocessor import Preprocessor
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from util import Loader
from imbhn import IMBHN
import logging

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
    
parameters = { 'preprocessor':[None, Preprocessor(lang='english')],  'clf':[SVC(), MultinomialNB(), 
                               MultinomialNB(alpha=0.01), BernoulliNB(alpha=.01), IMBHN()] }

rcv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
cv = StratifiedKFold(n_splits=10, random_state=0)

experiment(d,create_pipes(cache=True), parameters, logger=logger, cv=cv, scoring = ['accuracy'])
