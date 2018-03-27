#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 17:13:30 2018

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

