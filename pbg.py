#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 18:01:50 2018

@author: thiagodepaulo
"""

import numpy as np
import time
from sklearn.base import BaseEstimator, ClassifierMixin
from os.path import join
from util import RandMatrices
#from scipy.sparse import csr_matrix

class PBG(BaseEstimator, ClassifierMixin):

    def __init__(self, n_components, alpha=0.05, beta=0.0001, local_max_itr=50,
              global_max_itr=50, local_threshold = 1e-6, global_threshold = 1e-6,
              max_time=18000, save_interval=-1, out_dir='.', out_A='A', out_B='B',
              calc_q=False, debug=False):
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.local_max_itr = local_max_itr
        self.global_max_itr = global_max_itr
        self.local_threshold = local_threshold
        self.global_threshold = global_threshold
        self.max_time = max_time    # seconds
        self.save_interval = save_interval
        self.out_dir = out_dir
        self.out_A = out_A
        self.out_B = out_B
        self.calc_q = calc_q
        self.debug = debug

    def normalizedbycolumn_map(self, B):
        n = len(B.values()[0])
        col_sum = np.zeros(n)
        for key in B:
            vet = B[key]
            for i in range(n):
                col_sum[i] += vet[i]
        for key in B:
            vet = B[key]
            for i in range(n):
                vet[i] /= col_sum[i]
                vet[i] = self.beta + vet[i]
        return B

    def normalizebycolumn_plus_beta(self, B):
        if isinstance(B, dict):
            return self.normalizedbycolumn_map(B)
        nrow, ncol = B.shape
        for i in range(ncol):
            B[:,i] /= B[:,i].sum()
        return self.beta + B

    def Q2(self, X, D, A, B, alpha):
        CONST = 0.0000001
        _sum = 0
        for d_j in D:
            for w_i, f_ji in zip(X.indices[X.indptr[d_j]:X.indptr[d_j+1]],
                       X.data[X.indptr[d_j]:X.indptr[d_j+1]]):
                AB_ji = A[d_j]*B[w_i]
                C_ji = (AB_ji / AB_ji.sum())
                _sum += sum((f_ji * C_ji) * (np.log((AB_ji + CONST) / (C_ji + CONST))))
            _sum -= sum((alpha - A[d_j]) * np.log(A[d_j] + CONST) - A[d_j]*(np.log(A[d_j]  + CONST ) - 1))
        return _sum

    def Q(self, X, D, A, B):
        _sum = 0
        for d_j in D:
            for w_i, f_ji in zip(X.indices[X.indptr[d_j]:X.indptr[d_j+1]],
                       X.data[X.indptr[d_j]:X.indptr[d_j+1]]):
                sumAjBi = sum(A[d_j]*B[w_i])
                _sum += f_ji * np.log( f_ji / (sumAjBi)) - f_ji + (sumAjBi)
        return _sum

    def global_propag(self, Xcrc, W, A, B):
        for w_i in W:
            nB_i = np.zeros(self.n_components)
            for d_j, f_ji in zip(Xcrc.indices[Xcrc.indptr[w_i]:Xcrc.indptr[w_i+1]],
                        Xcrc.data[Xcrc.indptr[w_i]:Xcrc.indptr[w_i+1]]):
                H = (A[d_j] * B[w_i] )
                nB_i += f_ji * (H / H.sum())
            B[w_i] = nB_i
        # B = self.beta + self.normalizebycolumn(B)
        return self.normalizebycolumn_plus_beta(B)


    def local_propag(self, X, d_j, A_j, B):
        nA_j = np.zeros(len(A_j))
        for w_i, f_ji in zip(X.indices[X.indptr[d_j]:X.indptr[d_j+1]],
                       X.data[X.indptr[d_j]:X.indptr[d_j+1]]):
            H = (A_j * B[w_i])
            nA_j += f_ji * (H / H.sum())
        nA_j += self.alpha
        return nA_j

    def suppress(self, A_j, y_j):
        aux = A_j[y_j]
        A_j.fill(0)
        A_j[y_j] = aux

    def bgp(self, X, Xcrc, W, D, A, B, labelled=None):        
        global_niter = 0

        t0 = time.time()
        while global_niter <= self.global_max_itr :
            global_niter += 1

            if time.time() - t0 > self.max_time:
                break
            if self.save_interval!= None and self.save_interval % global_niter == 0 :
                self.save_matrices(A,B,global_niter)

            for d_j in D:
                local_niter = 0
                if self.debug and global_niter % 10 == 0: print(d_j)
                while local_niter <= self.local_max_itr:
                    local_niter += 1
                    oldA_j = np.array(A[d_j])
                    A[d_j] = self.local_propag(X, d_j, A[d_j], B)
                    mean_change = np.mean(abs(A[d_j] - oldA_j))
                    if mean_change <= self.local_threshold:
                        #if self.debug: print('convergiu itr %s' %local_niter)
                        break
                if (labelled is not  None) and (labelled[d_j] != -1):
                    self.suppress(A[d_j], labelled[d_j])
            self.global_propag(Xcrc, W, A, B)
            if self.calc_q:
                q = self.Q2(X, D, A, B, self.alpha)
                if self.debug: print('itr %s Q %s' % (global_niter, q))
            #if abs(q - oldq) <= self.GLOBAL_CONV_THRESHOLD:
            #    print '\t\t **GLOBAL convergiu em %s iteracoes' %global_niter
            #    break
            #oldq = q

    def fit(self, X, y=None):        
        rand = RandMatrices()
        #D -> set of documents indices, W-> set of word indices, K-> number of topics
        D, W, K = range(X.shape[0]), range(X.shape[1]), self.n_components
        A,B = rand.create_rand_matrices(D ,W ,K )
        # convert matriz
        Xcsc = X.tocsc()
        self.bgp(X, Xcsc, W, D, A, B, labelled=y)
        self.components_ = B.transpose()
        if y is not None:
            # label construction
            #truct a categorical distribution for classification only
            classes = np.unique(y)
            classes = (classes[classes != -1])
            self.classes_ = classes
            # atribui índices de classes aos exemplos não rotulados
            self.transduction_ = self.classes_[np.argmax(A, axis=1)].ravel()
            # cria distribuição dos rótulos (normaliza a matriz A)
            normalizer = np.atleast_2d(np.sum(A, axis=1)).T
            self.label_distributions_ = A / normalizer
        return self

    def transform(self, X):
        return None


    def save_matrices(self, A, B, global_niter):
        np.save(join(self.out_dir, self.out_A+'_'+str(global_niter)), A)
        np.save(join(self.out_dir, self.out_B+'_'+str(global_niter)), B)

#normalizer = np.atleast_2d(np.sum(probabilities, axis=1)).T
 #       probabilities /= normalizer
#        
