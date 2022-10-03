import pickle
import numpy as np
from random import random
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from random import random, choice
from random import *
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, Conv1D, GlobalMaxPooling1D, Multiply,BatchNormalization,Activation,Flatten
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam ,SGD# SGD with a learning rate of 0.001 can be a good choice here
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from Variables import *
##Saved data path
dic="/content/drive/My Drive/Next Word Prediction Data/"
dic=""

##Load Saved model
root=""
model = keras.models.load_model(root+'Saved_Model')
 
##Loading all saved data 
with open(dic+'dict_Of_index_All_Words.pickle', 'rb') as handle:
    dict_Of_index_All_Words = pickle.load(handle)
with open(dic+'dict_Of_All_Words_Index.pickle', 'rb') as handle:
    dict_Of_All_Words_Index = pickle.load(handle)

with open(dic+'dict_Of_index_Top_Words.pickle', 'rb') as handle:
    dict_Of_index_Top_Words = pickle.load(handle)
with open(dic+'dict_Of_TOP_Words_index.pickle', 'rb') as handle:
    dict_Of_TOP_Words_index = pickle.load(handle)

## provide your validation/ test sample file path here 
with open(dic+'Valid.data', 'rb') as handle:
    Valid_data = pickle.load(handle)

with open(dic+'tokenizer.pickle', 'rb') as handle:
    Tokenizer = pickle.load(handle) 
with open(dic+'top_k_word.pickle', 'rb') as handle:
    top_k_word = pickle.load(handle)

Valid_data=np.array(Valid_data)

top_word_size=len(top_k_word)


# same as in DataGenerator_new class of generator.py 
def padding_word(dem):
  
  dem=dem.strip()
  word=[]
  word.append(Tokenizer.word_index.get('#'))
  for y in dem:
    value_check=Tokenizer.word_index.get(y)
    if(value_check==None):
      value_check=len(Tokenizer.word_index)+1
    word.append(value_check)
  word.append(Tokenizer.word_index.get('$'))
  flag=0
  out=[]
  if(len(word)<word_char_size):
    word_len=len(word)
    dif=word_char_size-word_len
    flag=1
    for c in range(0,math.ceil(dif/2)):
      out.append(0)
    for c in word:
      out.append(c)
    for c in range(0,math.floor(dif/2)):
      out.append(0)

  if(len(word)>=word_char_size and flag==0):
    
    out=word[0:word_char_size]
    out[11]=Tokenizer.word_index.get('$')
  return out

# same as in DataGenerator_new class of generator.py
def padding_sentance(dem): 
  if(len(dem)<sen_size-1):
    x=len(dem)
    pad_sen=[]
    dif=sen_size-1-x
    for i in range(0,dif):
      pad_sen.append([0]*word_char_size)
    for c in dem:
      pad_sen.append(c)
  else:
    pad_sen=dem[0:sen_size-1]
  return pad_sen

# same as in DataGenerator_new class of generator.py
def mask_sen(pad_sen):
  pad_zero=[]
  pad_zero=[0.0]*CNN_Vec_size

  pad_one=[]
  pad_one=[1.0]*CNN_Vec_size

  full_sen=[]
  for i in pad_sen:
    all_zero=0
    for j in i:
       if(j==0):
         all_zero=all_zero+1
    if (all_zero==len(i)):
      full_sen.append(pad_zero)
    else:
      full_sen.append(pad_one)
  
  return full_sen

# same as in DataGenerator_new class of generator.py
def x_y_genarator_model_2(one_sentence):
    input1=[]
    length=len(one_sentence)
    for i in range(0,length-1):
      j=one_sentence[i]
      d=dict_Of_index_All_Words[j]
      input1.append(padding_word(d))
    input11=padding_sentance(input1)
    mask=mask_sen(input11)
    
    temp2=one_sentence[-1]

  
    x=dict_Of_index_All_Words[temp2]
    output=dict_Of_TOP_Words_index[x]
    
    return input11,mask,output



output=[]
in_1=[]
mask_1=[]
# p is probability threshold required during PPL calculation 
p = 1/ (len(top_k_word))
all_p=0
total=0

for i in range (0,len(Valid_data)):
    # Generate input sentence, mask input and ground truth output 
    total=1+total
    sen=Valid_data[i] 
    a,b,c=x_y_genarator_model_2(sen)
    in_1.append(a) 
    mask_1.append(b)
    output.append(c)
    
    ## RAM space may overflow if we want to perform parallel computation on all samples at once 
    if total%35000==0 and total!=0: 
      in_1=np.array(in_1)
      mask_1=np.array(mask_1)
      output=np.array(output)
      # model predicted probability vector
      xx=model.predict([in_1,mask_1], batch_size=32, verbose=0, steps=None, callbacks=None, max_queue_size=10,
        workers=6, use_multiprocessing=False
          )
      
      ## PPL calculation:
      # exp( -1/n * (log(P1)+log(P2)+...+log(Pn)) )
      # here, n is total number of samples. Pi is the model predicted probability of the ground truth output word 
      # of input i no. input sample 
      for k in range(0,35000):
          d=output[k]
          pi=xx[k][d]
          # There may exist noisy samples in a dataset. The probability for those sample ground truth outputs
          # may appear to be absurdly small, which will adversely affect the PPL, no matter how large the
          # the probability values are for most other samples. We assume the probability threshold to be 
          # (1/top_word_no), assuming random uniform prediction by the model at worst case.  
          if(pi<p):
            pi=p
          all_p=all_p+np.log(pi)
      output=[]
      in_1=[]
      mask_1=[]
# Sample no. may not be exact multiple of 35000. probability calculation for rest of the samples 
in_1=np.array(in_1)
mask_1=np.array(mask_1)
output=np.array(output)
xx=model.predict([in_1,mask_1], batch_size=32, verbose=0, steps=None, callbacks=None, max_queue_size=10,
      workers=6, use_multiprocessing=False
        )

for k in range(0,len(output)):
  d=output[k]
  pi=xx[k][d]
  if(pi<p):
    pi=p

  all_p=all_p+np.log(pi)
sum= -(1/(total))*all_p
print(np.exp(sum))