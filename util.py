#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:46:22 2017

@author: thiagodepaulo
"""
import re
import glob
import os.path
import codecs 


class Loader:
    
    def __init__(self):
        pass
    
    def from_files(self, path):
        dirs = glob.glob(os.path.join(path,'*'));
        class_names = []
        class_idx = []
        cid = -1
        corpus = []
        for _dir in dirs:
            cid+= 1
            class_names.append(os.path.basename(_dir))
            arqs = glob.glob(os.path.join(_dir,'*'))
            for arq in arqs:                
                with codecs.open(arq, "r", "ISO-8859-1") as myfile:
                    data=myfile.read().replace('\n', '')
                corpus.append(data)
                class_idx.append(cid)
        result = {'corpus':corpus, 'class_index': class_idx, 'class_names':class_names}
        return result
    
    
    def from_text_line_by_line(self, arq):
        doc = []
        for line in open(arq):
            doc.append(line)
        return doc
    
    def _str_to_list(self, s):
        _s = re.split(',|{|}',s)
        return [ x for x in _s if len(x) > 0]
    
    def _str_to_date(self, s):
        pass
    
    def _convert(self, x, i, attr_list):        
        if attr_list[i][1] == self.attr_numeric[1]:
            return float(x)
        elif attr_list[i][1] == self.attr_numeric[2]:
            return int(x)
        elif attr_list[i][1] == self.attr_string[0]:
            return x.replace("'","").replace('\'',"").replace('\"',"")
        else:
            return x.replace("'","").replace('\'',"").replace('\"',"")
            
    
    def from_arff(self, arq, delimiter=','):
        relation_name = ''
        attr_count = 0
        attr_list = []
        data = []
        self.attr_numeric = ['numeric', 'real', 'integer']
        self.attr_string = ['string']
        self.attr_date = ['date']
        read_data = False
        for line in open(arq):                        
            line = line.lower().strip()            
            if line.startswith('#'): continue
            if read_data:
                vdata = line.split(delimiter)                                
                data.append([ self._convert(x,i,attr_list) for i,x in enumerate(vdata) ])
            elif not line.startswith('#'):
                if line.startswith('@relation'):
                    relation_name = line.split()[1]
                elif line.startswith('@attribute'):
                    attr_count += 1
                    attr = line.split()
                    attr_type = attr[2]
                    if attr_type in self.attr_numeric or attr_type in self.attr_string:
                        attr_list.append((attr[1], attr[2]))
                    elif attr_type in self.attr_date:
                        attr_list.append((attr[1], self._str_to_date(attr[2])))
                    else:
                        attr_list.append((attr[1], self._str_to_list(''.join(attr[2:]))))
                elif line.startswith('@data'):
                    read_data = True 
                    continue
        d = dict()
        d['attributes'] = attr_list
        d['data'] = data
        d['relation'] = relation_name
        return d
    
    def from_sparse_arff(self,arq, delimiter=','):
        pass
        
#        
#l = Loader()
##d = l.from_arff('datasets/SyskillWebert.arff')
#d = l.from_files('/exp/datasets/docs_rotulados/SyskillWebert-Parsed')
#
#import preprocessor
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.datasets import fetch_20newsgroups
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.linear_model import SGDClassifier
#from sklearn.pipeline import Pipeline
#import numpy as np
#from sklearn import metrics
#from sklearn.model_selection import GridSearchCV
#
##text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), 
##                     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),])
#
##parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3),}
#
#text_clf = Pipeline([('text_preproc',preprocessor.Preprocessor()), ('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), 
#                     ('clf', MultinomialNB()),])
#
#parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),}
#
#
#gs_clf = GridSearchCV(text_clf, parameters,  cv=10, n_jobs=-1)
#gs_clf = gs_clf.fit(d['corpus'], d['class_index'])
#print(gs_clf.cv_results_)
#
