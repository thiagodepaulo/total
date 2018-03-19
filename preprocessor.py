#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:17:31 2017

@author: thiagodepaulo
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:33:39 2017
@author: thiagodepaulo
"""
import re
import nltk
from unidecode import unidecode
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from sklearn.base import TransformerMixin, BaseEstimator
import logging


class Preprocessor(TransformerMixin, BaseEstimator):
    
    def __init__(self, lang='english', stop_words=True, stem=True):
        self.lang = lang                
        self.stop_words = stop_words
        self.stem = stem        
        pass
    
    def cleaning(self, corpus):
        return [self.strip_accents_nonalpha(text) for text in corpus]
        
    # remove accents and numeric characteres
    def strip_accents_nonalpha(self, text):
        text = text.lower()
        #if not isinstance(text, unicode):
        #    text = unicode(text, 'utf-8')
        t = unidecode(text)
        t.encode("ascii")  #works fine, because all non-ASCII from s are replaced with their equivalents
        t = re.sub(r'[^a-z]', ' ', t)       
        t = ' '.join(t.strip().split())
        return t
    
    def remove_stopwords(self, corpus, remove_accents=False):
        stopwords = self.l_stopwords
        if remove_accents:
            stopwords = [self.strip_accents_nonalpha(w) for w in stopwords]
        pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
        return [pattern.sub('', text) for text in corpus]
    
    def remove_words(self, corpus, l_words):                
        pattern = re.compile(r'\b(' + r'|'.join(l_words) + r')\b\s*')
        return [pattern.sub('', text) for text in corpus]
    
    def stemmer(self, corpus):
        stemmer = SnowballStemmer(self.lang)
        self.len_min_word = 1
        self.stem_map = {}
        
        def stem(word):
            stemmed = stemmer.stem(word)
            if stemmed not in self.stem_map:
                self.stem_map[stemmed] = []
            self.stem_map[stemmed].append(word)
            return stemmed
            
        corpus = [[stem(s) for s in x.split() if len(s) > self.len_min_word] for x in corpus]
        return [' '.join(x) for x in corpus]

    def re_stemmer(self, word):
        if not hasattr(self, 'stem_counters'):
            self.stem_counters = { w:Counter(self.stem_map[w]) for w in self.stem_map}        
        return self.stem_counters[word].most_common(1)[0][0]
    
    def preprocess(self, corpus, stop_words=True, stemmer=True):        
        self.l_stopwords = nltk.corpus.stopwords.words(self.lang)
        self.len_min_word = 1
        def do_preprocessing(l_docs):
            logging.info("preprocessing "+str(len(l_docs))+" documents")
            if stop_words:
                l_docs = self.remove_stopwords(l_docs, remove_accents=False)
            if stemmer:
                l_docs = self.stemmer(l_docs)
            l_docs = self.cleaning(l_docs)
            logging.info("done preprocessing")
            return l_docs
        
        if type(corpus) is dict:
            corpus = {k: do_preprocessing(corpus[k]) for k in corpus.keys()}
        else:
            corpus = do_preprocessing(corpus)        
        return corpus
    
    def fit(self, X, y=None):                
        return self
    
    def transform(self, X, *_):
        return self.preprocess(X, stop_words=self.stop_words, stemmer=self.stem)
            
    

#l = ['oi, como vai você?','Eu vou bem, obrigado#$% obrigadinho obrigadão falar falarão falará ']

#p = Preprocessor()
#print(p.preprocess(l))
    
    
    
