# Building and saving our proposed Coordinated_CNN model with randomly initialized weights
import pickle
import numpy as np
from random import random
import keras
from keras.preprocessing.text import Tokenizer
import math 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Bidirectional, LSTM, Activation, dot, Embedding,BatchNormalization
from tensorflow.keras.utils import plot_model
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from Variables import  *

## The folder path where all data.py output pickle files have been saved and also where you want to save your model
dic="/content/drive/My Drive/AAAI Code With Dataset/"
dic=""
with open(dic+'tokenizer.pickle', 'rb') as handle:
    Tokenizer = pickle.load(handle) 
with open(dic+'top_k_word.pickle', 'rb') as handle:
    top_k_word = pickle.load(handle)
    
character_size=len(Tokenizer.word_index)+1
unique_words=len(top_k_word)
top_word_size=unique_words

### Building the vocabulary learner sub-model 
visible1 = layers.Input(shape=(sen_size-1, word_char_size,))
embedding_layer_1=Embedding(character_size+1,
              output_dim=char_vec_size)(visible1) 
 
x = TimeDistributed(Conv1D(64,2,padding='same',strides=1))(embedding_layer_1)
x = TimeDistributed( BatchNormalization())(x)
x = TimeDistributed(Activation("relu"))(x)

x = TimeDistributed(Conv1D(64,3,padding='same', strides=1))(x)
x = TimeDistributed( BatchNormalization())(x)
x = TimeDistributed(Activation("relu"))(x)

x = TimeDistributed(Conv1D(128,3,padding='same',strides=1))(x)
x = TimeDistributed( BatchNormalization())(x)
x = TimeDistributed(Activation("relu"))(x)

x = TimeDistributed(Conv1D(128,3,padding='same',strides=1))(x)
x = TimeDistributed( BatchNormalization())(x)
x = TimeDistributed(Activation("relu"))(x)

x = TimeDistributed(Conv1D(CNN_Vec_size,4,padding='same',strides=1))(x) 
x = TimeDistributed( BatchNormalization())(x)
x = TimeDistributed(Activation("relu"))(x)

x = TimeDistributed(GlobalMaxPooling1D())(x) ## get CNNvecs for all input words

## Masking pad words with 0's and valid words with 1's 
mask_input1 = layers.Input(shape=(sen_size-1, CNN_Vec_size,))
mask = layers.Multiply()([x, mask_input1])

### Building the  terminal coordinator sub-model
x = Conv1D(64,2,padding='same',name="CNN2",strides=1)(mask)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = Conv1D(92,3,padding='same',name="CNN3",strides=1)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = Conv1D(128,3,padding='same',name="CNN4",strides=1)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = Conv1D(128,3,padding='same',name="CNN5",strides=1)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = Conv1D(192,4,padding='same',name="CNN6",strides=1)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = layers.Flatten()(x) # Coordination vector

d1 = Dense(128, activation='relu')(x)
drop1=Dropout(Dropout_Rate)(d1)
d2 = Dense(192, activation='relu')(drop1)
drop2=Dropout(Dropout_Rate)(d2)
d3 = Dense(256, activation='relu')(drop2)

output = Dense(unique_words, activation='softmax')(d3)
model = keras.Model(inputs=[visible1,mask_input1], outputs=output)

opt = keras.optimizers.SGD(learning_rate=learning_rate,clipnorm=clipnorm)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['categorical_crossentropy']
              )

### Viewing model diagram
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.save(dic+"Saved_Model")