import pickle
import numpy as geek
from random import random
from keras.preprocessing.text import Tokenizer
from Variables import *

"""
****  Input: Dataset (a text file containing a full sentence per line).
****  Output: (1) unique characters, top words, unique words, unique word to index mapping, index to unique
                  word mapping, top word to index mapping, index to top word mapping 



=========================================================================================================

path :: Dataset path. The dataset contains a sentence in each line 
dic :: This variable defines the folder path where the outputs will be stored as pickle files 
"""
#These two variables are defined in Variables.py file 
#top_word_unique_char_no=25000
#coverage_of_words=0.90
path="/content/drive/MyDrive/AAAI codes and Datasets/Bangla/sample data.txt"
dic="/content/drive/My Drive/AAAI Code With Dataset/"
dic=""
# loading the sentences 
Sentences=[]
with open(path,'r',encoding = 'utf-8') as f:
  for line in f:
    Sentences.append(line)

## Appending all words to the All_words list. Same word will appear multiple times 
All_words=[]
for i in Sentences:
  words=i.split()
  for j in words:
      All_words.append(j)    
All_words.append("কককককককককক") ## "কককককককককক" is used as beginning  of sentence.


### Used for merging two lists taken as parameter
def append_to_list(All_unique, Part_of_Unique_words):
  temp=[]
  for i in All_unique:
    temp.append(i)
  for i in Part_of_Unique_words:
    temp.append(i)

  return temp

### returns all unique words from a word list taken as parameter.
def Find_Unique_Words(word_list):
  index=0
  words=[]
  All_unique=[]
  for i in word_list:
    index=index+1
    words.append(i)

    if index%5000000==0 and index!=0 : # It is hard to load all words into array and find out unique word using built in libary. 
                                        ## That is why we load a part of all words into a list and find out unique word from that list.
                                        ## Then free that space for reusing it.
      out_arr = geek.asarray(words) 
      Part_of_Unique_words = geek.unique(out_arr, axis=0)
      All_unique=append_to_list(All_unique,Part_of_Unique_words) # appending unique word chunks
      words=[]

  if len(words)>0 :  # number of words in word_list may not be a multiple of 5000000. 
    out_arr = geek.asarray(words) 
    Part_of_Unique_words = geek.unique(out_arr, axis=0)
    All_unique=append_to_list(All_unique,Part_of_Unique_words)

  All_unique = geek.asarray(All_unique)  
  # Since we are appending unique word chunks, the merged list may containing redundancy. 
  All_unique = geek.unique(All_unique, axis=0)
  return All_unique


Unique_Words=Find_Unique_Words(All_words)

print("Total Unique words::  ",len(Unique_Words))


## frq_new will contain each unique word and its frequency in corpus --> [frequncy, word]
frq_new=[]
for i in Unique_Words:
  data=[]
  data.append(0)  # initialize each word frequeny with 0
  data.append(i)
  frq_new.append(data)

## word to index mapping
dict_Of_All_Words_Index = {  Unique_Words[i] : i for i in range(0, len(Unique_Words) ) }
## index to word mapping  
dict_Of_index_All_Words = { i : Unique_Words[i] for i in range(0, len(Unique_Words) ) }

## calculating frequncy of all unique words and updating frq_new with those frequencies
for i in Sentences:
  words=i.split()
  for j in words:     
        frq_new[dict_Of_All_Words_Index[j]][0]=frq_new[dict_Of_All_Words_Index[j]][0]+1

##Sorting the unique words in descending order of frequency.
demo=sorted(frq_new, key=lambda x: x[0])
ch= demo[::-1]


## Based on the defined coverage_of_words and word frequency, we are finding out top k words (top_k_words).
top_k_words=[]
count=0
fr=0
cover_fre=coverage_of_words * len(All_words)
for i in range(0, len(ch)):
  if(fr<cover_fre+1):
    top_k_words.append(ch[i][1])
  fr=fr+ch[i][0]

print("Top Words size::  ",len(top_k_words))


## For Charecter level tokenizer 
tokenizer = Tokenizer(
    char_level=True, ##-->>If True, a character will be treated as a token.

    filters=None, ##-->> A string where each element is a character that will be filtered from the texts.
                  ## The default is all punctuation, plus tabs and line breaks, minus the ' character.

    lower=False ##-->>Boolean. Whether to convert the texts to lowercase.
)

if len(All_words)<top_word_unique_char_no:
  print("top_word_unique_char_no is ",top_word_unique_char_no,"and that is  greater than total unique word count(",len(All_words),")")
  print("So,Now top_word_unique_char_no is ",len(top_k_words))
  top_word_unique_char_no=len(top_k_words)
## Fit the charecter level tokenizer on defined top_word_unique_char_no number of top words
for i in range (0,top_word_unique_char_no+1):
  
  tokenizer.fit_on_texts(ch[i][1])
tokenizer.fit_on_texts("$#ক") ### "#" and "$" are used to mark beginning and end of a word, respectively. ??


## index to top k word mapping 
dict_Of_index_Top_Words= { i : top_k_words[i] for i in range(0, len(top_k_words) ) }
## top k word to index mapping 
dict_Of_TOP_Words_index = {  top_k_words[i] : i for i in range(0, len(top_k_words) ) }



## Save all outputs in files
with open(dic+'tokenizer.pickle', 'wb') as handle:  ### -->> Total Unique character in dataset, saved as tokenizer.
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL) 

with open(dic+'top_k_word.pickle', 'wb') as handle:  ###-->> top words :: Based on the coverage.
    pickle.dump(top_k_words, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(dic+'unique word.pickle', 'wb') as handle: ###-->> unique word in dataset.
    pickle.dump(Unique_Words, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(dic+'dict_Of_index_All_Words.pickle', 'wb') as handle:  ###-->> index to word map dictionary
    pickle.dump(dict_Of_index_All_Words, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(dic+'dict_Of_All_Words_Index.pickle', 'wb') as handle:  ###-->> word to index map dictionary
    pickle.dump(dict_Of_All_Words_Index, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(dic+'dict_Of_index_Top_Words.pickle', 'wb') as handle:  ###-->> index to top word map dictionary
    pickle.dump(dict_Of_index_Top_Words, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(dic+'dict_Of_TOP_Words_index.pickle', 'wb') as handle:  ###-->> top word to index mapping 
    pickle.dump(dict_Of_TOP_Words_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
