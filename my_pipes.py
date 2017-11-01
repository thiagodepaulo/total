#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 10:24:48 2017

@author: thiagodepaulo
"""
from sklearn.pipeline import Pipeline
from preprocessor import Preprocessor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from tempfile import mkdtemp
from sklearn.externals.joblib import Memory

def preprocessor_steps():
    return [('preprocessor',Preprocessor()), ('countvectorizer',CountVectorizer()), ('tfidftransformer',TfidfTransformer())]

def mnb_pipes(cache=False):    
    memory=None
    if cache:
        cachedir = mkdtemp()
        memory = Memory(cachedir=cachedir, verbose=10)
    steps = preprocessor_steps()
    steps.append(('multinomialnb',MultinomialNB()))
    return Pipeline(steps, memory=memory)


    