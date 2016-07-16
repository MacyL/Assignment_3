# /usr/bin/env python
import os, sys
import glob
import re
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from collections import Counter
# input dir
pathPos = "/home/admin1/Documents/Machine_learning/txt_sentoken/pos/*.txt"
pathNeg = "/home/admin1/Documents/Machine_learning/txt_sentoken/neg/*.txt"
filesPos = glob.glob(pathPos) 
filesNeg = glob.glob(pathNeg)
# output dir
pathPosOut = "/home/admin1/Documents/Machine_learning/txt_sentoken/official/unigram/pos/"
pathNegOut = "/home/admin1/Documents/Machine_learning/txt_sentoken/official/unigram/neg/"
pathBigramPosOut= "/home/admin1/Documents/Machine_learning/txt_sentoken/official/bigram/pos/"
pathBigramNegOut= "/home/admin1/Documents/Machine_learning/txt_sentoken/official/bigram/neg/"
myFullList=[]
myFullBigramList=[]
# tokenize Positive list and put them in myFullList
for files in filesPos: 	
	myFilePos=open(files,'r').read()
	tokenizer= RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
	tokensPos = tokenizer.tokenize(myFilePos)
	tokenBigramPos =list(nltk.bigrams(tokensPos))
	#prepare the output file name 	
	head, tail = os.path.split(files)
	#output to unigram file
	myTokensFile=open(os.path.join(pathPosOut, tail), 'w')
	myTokensFile.write(str(tokensPos)+'\n')
	myTokensFile.close()
	#output to bigram file
	myBiTokensFile=open(os.path.join(pathBigramPosOut, tail), 'w')
	myBiTokensFile.write(str(tokenBigramPos)+'\n')
	myBiTokensFile.close()
	# put unigram token in a list 
	myFullList += tokensPos
	# put bigram token in a list 
	myFullBigramList += tokenBigramPos

# tokenize Negtive list and put them in myFullList
for files in filesNeg: 
	myFileNeg=open(files,'r').read()
	tokenizer= RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
	tokensNeg = tokenizer.tokenize(myFileNeg)
	tokenBigramNeg =list(nltk.bigrams(tokensNeg))
	#prepare the output file name	
	head, tail = os.path.split(files)
	#output to unigram file
	myTokensFile=open(os.path.join(pathNegOut, tail), 'w')
	myTokensFile.write(str(tokensPos)+'\n')
	myTokensFile.close()
	#output to bigram file
	myBiTokensFile=open(os.path.join(pathBigramNegOut, tail), 'w')
	myBiTokensFile.write(str(tokenBigramNeg)+'\n')
	myBiTokensFile.close()
	# put bigram token in a list
	myFullBigramList += tokensNeg
# an initial report. 
print('process has been done, the total unigram token is :',len(myFullList))
print('process has been done, the total bigram token is :',len(myFullBigramList))
#create vocabulary list with counts. name as vocab.txt
myToken=Counter(myFullList).keys()
myTokenCount=Counter(myFullList).values()
test=Counter(myFullList)
testItem=test.items()
myVocab = pd.DataFrame({'Count' : list(myTokenCount),
'Vocabulary' : list(myToken),
  })
myBiToken=Counter(myFullBigramList).keys()
myBiTokenCount=Counter(myFullBigramList).values()
#testfile=open('test.txt','w')
#testfile.write(str(testItem)+'\n')
#testfile.close()
myBigramVocab = pd.DataFrame({'Count' : list(myBiTokenCount),
'Vocabulary' : list(myBiToken),
  })
myVocab.to_csv('/home/admin1/Documents/Machine_learning/txt_sentoken/official/vocab.txt')
myBigramVocab.to_csv('/home/admin1/Documents/Machine_learning/txt_sentoken/official/bigram_vocab.txt')


#print(myVocab)
#outfile=open("/home/admin1/Documents/Machine_learning/txt_sentoken/official/vocab.txt","w",encoding="UTF-8")
#outfile.write(str(myVocab)+'\n')
#outfile.close()
#outfile=open("/home/admin1/Documents/Machine_learning/txt_sentoken/official/bigram_vocab.txt","w",encoding="UTF-8")
#outfile.write(str(myBigramVocab)+'\n')
#outfile.close()
print('The tokens and counts has been calculated. The file is saved as : vocab.txt and bigram_vocab.txt')
