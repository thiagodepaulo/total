#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 15:49:32 2017

@author: thiagodepaulo
"""

from preprocessor import Preprocessor
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
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

