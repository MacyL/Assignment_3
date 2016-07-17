# /usr/bin/env python
import os, sys
import glob
import re
import nltk
import pandas as pd
import numpy as np
# input dir
pathPos = "/home/admin1/Documents/Machine_learning/txt_sentoken/official/unigram/pos/*.txt"
pathNeg = "/home/admin1/Documents/Machine_learning/txt_sentoken/official/unigram/neg/*.txt"
filesPos = glob.glob(pathPos) 
filesNeg = glob.glob(pathNeg)

# read in unigram vocab file as reference.
Reference=pd.read_csv('/home/admin1/Documents/Machine_learning/txt_sentoken/official/vocab.txt')
#ReferenceSort=Reference.sort_values(['Count'], ascending=False)
# This is only for bigram
#Reference=ReferenceSort[ReferenceSort.Vocabulary.str.startswith("('")]
ReferenceToken5=Reference[Reference['Count']>=5]
# Just save it in system in case something bad happen. 
ReferenceBackUp=ReferenceToken5
# read in files and merge with reference by left join. 
Reference=ReferenceToken5
myCol=['R_count','Vocabulary']
# This is for Positive
for files in filesPos: 	
	myArticle=pd.read_csv(files)
	Reference=pd.merge(left=Reference,right=myArticle,how='left',on=['Vocabulary'])
	head, tail = os.path.split(files)
	# This is to create a list of column name for later 
	myCol +=[tail]
Reference.columns=myCol
Reference.to_csv('Unigram_Pos_full.csv')
# Now for Negative
Reference=ReferenceToken5
myCol=['R_count','Vocabulary']
# read in files and merge with reference by left join
for files in filesNeg:
	myArticle=pd.read_csv(files)
	Reference=pd.merge(left=Reference,right=myArticle,how='left',on=['Vocabulary'])
	head, tail = os.path.split(files)
	myCol +=[tail]
Reference.columns=myCol
Reference.to_csv('Unigram_Neg_full.csv')



