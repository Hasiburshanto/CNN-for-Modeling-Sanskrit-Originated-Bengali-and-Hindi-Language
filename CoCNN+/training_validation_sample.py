import numpy as geek
from keras.preprocessing.text import Tokenizer
import pickle
from random import random
from Variables import *

"""
****  Input: Dataset (a text file containing a full sentence per line) and the output files obtained from
             running data.py file.
****  Output: Training and validation samples.



=========================================================================================================

path :: Dataset (a text file containing a full sentence per line) path
dic  :: The folder path where all data.py output pickle files have been saved  

"""
#This variable is defined in Variables.py file 
#sentence_topWord_percentage=85
path="/content/drive/My Drive/AAAI Code With Dataset/Dataset/DatasetsProthomAlo.txt"
dic="/content/drive/My Drive/AAAI Code With Dataset/"
dic=""

# Loading the necessary dictionary mappings 
with open(dic+'dict_Of_All_Words_Index.pickle', 'rb') as handle:
    dict_Of_All_Words_Index = pickle.load(handle)

with open(dic+'dict_Of_TOP_Words_index.pickle', 'rb') as handle:
    dict_Of_TOP_Words_index = pickle.load(handle)

with open(dic+'dict_Of_index_All_Words.pickle', 'rb') as handle:
    dict_Of_index_All_Words = pickle.load(handle)


## Load all sentences into "Sentences" variable
Sentences=[]
with open(path,'r',encoding = 'utf-8') as f:
  for line in f:
      Sentences.append(line)

## Replaces each word of the sentences with corresponding word index 
unique_Sentance_number_represent=[] 
for i in Sentences:
  x=i.strip()
  x=x.split()
  
  d=[]
  for j in x:

    d.append(dict_Of_All_Words_Index[j])
  if len(d)>1:
    unique_Sentance_number_represent.append(d)

print(len(unique_Sentance_number_represent))

## Keep only those sentences which have at least sentence_topWord_percentage portion of top k words  
temp_dataset=[]

for i in unique_Sentance_number_represent:
      UNK,WORD=0,0
      for j in i:
        y=dict_Of_index_All_Words[j] 
        if y not in dict_Of_TOP_Words_index:
          UNK=UNK+1
        else: 
          WORD=WORD+1
      
      if  (WORD/len(i))*100  >=sentence_topWord_percentage:
        temp_dataset.append(i)

print("New Size of dataset:: ",len(temp_dataset),"Old Size of dataset:: ",len(unique_Sentance_number_represent) )


## Make samples and Split into Training and validation set.
# Each sample contains some input words and the next word as output 

train=[]
valid=[]
for i in temp_dataset:
  tm=[]
  tm.append(dict_Of_All_Words_Index["কককককককককক"]) ## default start word index
  for j in i:
    tm.append(j)
  #print(len(tm),tm)
  # minimum sentence length 4: 1st word is a default symbol, last word is output. We are assuming that
  # input should be at least 4-2 = 2 words for next word prediction 
  for k in range(4,len(tm)+1): 
    tem1=[]
    tem1=tm[0:k]  
    le=len(dict_Of_index_All_Words[tem1[-1]])
    x=dict_Of_index_All_Words[tem1[-1]]
    flag=0

    if x not in dict_Of_TOP_Words_index:
      flag=1
    
    ## Model will not predict any numeric value and UNK word (Not in top k word list).
    ## All numeric values should be replaced by "1111111111" in your dataset 
    if dict_Of_index_All_Words[tem1[-1]]!="1111111111"  and flag==0 :
      ## You can put all samples in one file if you want. You would want to do that if you have a separate 
      # test dataset 
      if(random()<=.90):  ## we are performing a 90-10 split. You can change this 
        
        train.append(tm[0:k])
      else:
        valid.append(tm[0:k])
      
  
print("Training sample size :: ",len(train))  
print("Validation sample size :: ",len(valid))  

##Saved Training and validation Data.
with open(dic+'Train.data', 'wb') as filehandle:
    pickle.dump(train, filehandle)

with open(dic+'Valid.data', 'wb') as filehandle:
    pickle.dump(valid, filehandle)