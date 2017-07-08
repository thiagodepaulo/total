#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:46:22 2017

@author: thiagodepaulo
"""
import re

class Loader:
    
    def __init__(self):
        pass
    
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
        
        
        
l = Loader()
d = l.from_arff('datasets/SyskillWebert.arff')
print(d)

