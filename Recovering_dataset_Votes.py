#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:17:45 2022

@author: aghm
"""

import numpy as np
import pandas as pd


label=pd.read_csv('Votes_label_server.csv') 
label=label.loc[:, ~label.columns.str.contains('^Unnamed')]
print(label.columns)
fbp_array=[0.1,0.4,0.7,0.9]
for fbp in fbp_array:
    mat=[]
    mat2=[]
    for att in label.columns:    
        if(att=='?'):
            att='AA'
        ######
        aux2=[]
        aux2=pd.read_csv('Results-VAE/Votes4D'+att+'_flopprob_'+str(fbp)+'_peratt_1000_probabilities_centroid.csv')
        aux2=aux2.loc[:, ~aux2.columns.str.contains('^Unnamed')]
        mat2.append(np.array(aux2.idxmax(axis=1)))
    ########
    b=[]
    b=np.array(mat2)    
    b=np.transpose(np.array(mat2))
    out=[]
    out=pd.DataFrame(b,columns=[label.columns])
    out.to_csv('csv_eval/VAE_Votes_FULLSPACE_flopprob_'+str(fbp)+'_.csv')

###################################