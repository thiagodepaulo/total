#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:35:15 2017

@author: thiago
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
#from scipy.sparse import csr_matrix

class IMBHN(BaseEstimator, ClassifierMixin):
                
    def __init__(self, eta=0.1, max_itr=1000, min_sqr_error=0.005):
        self.eta = eta
        self.max_itr = max_itr
        self.min_sqr_error = min_sqr_error
                    
    def stop_analysis(self, mean_error, num_iterations):
        #if mean_error - self.mean_error == 0:            
         #   return True
        if mean_error < self.min_sqr_error:
            return True
        print(self.max_itr)
        if num_iterations > self.max_itr:
            return True
        self.mean_error = mean_error
        self.niterations = num_iterations
        return False
    
    def init_dataset(self, X,y):
        self.ndocs, self.nterms = X.shape
        self.nclass = len(set(y))
        self.W = X  # document-term matrix
        self.D = range(self.ndocs)       # set of documents
        self.C = range(self.nclass)      # set of classes
        self.T = range(self.nterms)      # set of terms        
        self.Y = y                  # class labels        
        self.current_doc_index = -1
        self.terms_by_doc = []
        self.small_float = 0.000001
        self.mean_error = float("-inf")
        print("oi"+str(self.max_itr))
        
    def get_terms_by_doc(self, d):
        if self.current_doc_index == d:
            return self.terms_by_doc
        self.current_doc_index = d
        self.terms_by_doc = self.W.indices[self.W.indptr[d]:self.W.indptr[d+1]]
        return self.terms_by_doc

    def classify(self, d):
        out = np.zeros(self.nclass)
        for c in self.C: 
            cw = 0 # class_weight
            for t in self.set_of_terms_by_doc(d):
                cw += self.F[t,c] * self.W[d,t]        
            out[c] = cw
        return out
    
    def classify_hard(self, d):                
        _max, _max_c = float("-inf"), -1        
        for c in self.C: 
            cw = 0 # class_weight
            for t in self.get_terms_by_doc(d):
                cw += self.F[t,c] * self.W[d,t]                    
            if _max < cw: _max, _max_c = cw, c                        
        out = np.zeros(self.nclass)
        if _max > self.small_float: out[_max_c] = 1
        return out
        

    def fit(self, X, y):
        print("running")
        self.init_dataset(X,y)
        _exit = False
        self.F = np.zeros((self.nterms, self.nclass))
        num_it = 0
        
        while(_exit == False):            
            mean_error = 0.0
            for d in self.D:
                estimated_classes = self.classify_hard(d)
                for c in self.C:
                    error = (1 if self.Y[d] == c else 0) - estimated_classes[c]
                    mean_error += ((error*error)/2.0)
                    for t in self.get_terms_by_doc(d):
                        current_weight = self.F[t,c]
                        new_weight = current_weight + (self.eta * self.W[d,t] * error)
                        self.F[t,c] = new_weight
            num_it += 1
            mean_error = mean_error/self.ndocs
            _exit = self.stop_analysis(mean_error, num_it)
            print(num_it)
            print(mean_error)            
        return self
    
    def transform(self, X):
        return None
    
    def predict(self, X):
        ndocs = X.shape[0]
        result = np.zeros(ndocs)
        for d in range(ndocs):
            _max, _max_c = float("-inf"), -1
            for c in self.C: 
                cw = 0 # class_weight
                for t in X.indices[X.indptr[d]:X.indptr[d+1]]:
                    cw += self.F[t,c] * X[d,t]            
                if _max < cw: _max, _max_c = cw, c
            result[d] = _max_c
        return result

            