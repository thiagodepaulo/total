#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:35:15 2017

@author: thiago
"""
import numpy as np
#from scipy.sparse import csr_matrix

class IMBHN:
                
    def __init__(self, eta=0.1, num_max_itr=100, min_sqr_error=0.005):
        self.eta = eta
        self.max_itr = num_max_itr
        self.min_sqr_error = min_sqr_error
        
    
    def weight_initialization(self):
        pass
    
    def stop_analysis(self):
        pass

    def class_weight(self, d, c):
        cw = 0 # class_weight
        for t in self.T:
            if self.W[d,t]: cw += self.F[t,c] * self.W[d,t]
        return cw
    
    def weight_corretion(self, d, c, error):
        for t in self.T:
            if error==1 and self.W[d,t]>0:
                current_weight = self.F[t,c]
                new_weight = current_weight + (self.eta * self.W[d,t] * error)
                self.F[t,c] = new_weight
                
    def _set(self, X,y):
        ndocs, nterms = X.shape
        self.W = X
        self.D = range(ndocs)
        self.C = range(len(y))
        self.T = range(nterms)
        self.Y = y
        self.classOut = np.zeros(len(D))
        

    def fit(self, X, y):
        stop_criterion = False
        self.weight_initialization()
        self.num_it = 0

        while(stop_criterion):
            squared_error_acm = 0
            for d in self.D:
                out = np.zeros(len(self.C))
                for c in self.C: out[c] = self.class_weight(d,c)
                self.classOut[d] = np.argmax(out)
                    
                for c in self.C:
                    error = 0 if self.classOut[d] == self.Y[d] else 1
                    squared_error_acm = float(error**2)/2.0
                    self.weight_corretion(d, c, error)
        
            self.mean_sqr_error = squared_error_acm/len(self.D)
            self.num_it += 1
            stop_criterion = self.stop_analysis()

            