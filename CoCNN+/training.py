
import genaretor ##Class

import pickle
import numpy as np
from tensorflow import keras
from Variables import *
##dic:: folder path of where you want to save your trained model 
dic="/content/drive/My Drive/AAAI Code With Dataset/"
dic=""
epoch_number=1
Batch_size=64
##root:: Saved model path
root="/content/drive/My Drive/AAAI Code With Dataset/"
root=""
## Already created these files and saved.
with open(dic+'dict_Of_index_All_Words.pickle', 'rb') as handle:
    dict_Of_index_All_Words = pickle.load(handle)
with open(dic+'dict_Of_All_Words_Index.pickle', 'rb') as handle:
    dict_Of_All_Words_Index = pickle.load(handle)

with open(dic+'dict_Of_index_Top_Words.pickle', 'rb') as handle:
    dict_Of_index_Top_Words = pickle.load(handle)
with open(dic+'dict_Of_TOP_Words_index.pickle', 'rb') as handle:
    dict_Of_TOP_Words_index = pickle.load(handle)
with open(dic+'Train.data', 'rb') as handle:
    Train_data = pickle.load(handle)
with open(dic+'Valid.data', 'rb') as handle:
    Valid_data = pickle.load(handle)

with open(dic+'tokenizer.pickle', 'rb') as handle:
    Tokenizer = pickle.load(handle) 

with open(dic+'top_k_word.pickle', 'rb') as handle:
    top_k_word = pickle.load(handle)

Train_data=np.array(Train_data) 
train_data_length=len(Train_data)

Valid_data=np.array(Valid_data)
valid_data_length=len(Valid_data)


top_word_size=len(top_k_word)

training_generator = genaretor.DataGenerator_new(Train_data,train_data_length, Batch_size,top_word_size,
                                                dict_Of_index_All_Words,dict_Of_All_Words_Index,
                                                dict_Of_index_Top_Words,dict_Of_TOP_Words_index,
                                                Tokenizer,sen_size,word_char_size,CNN_Vec_size)

valid_generator = genaretor.DataGenerator_new(Valid_data,valid_data_length, Batch_size,top_word_size,
                                                dict_Of_index_All_Words,dict_Of_All_Words_Index,
                                                dict_Of_index_Top_Words,dict_Of_TOP_Words_index,
                                                Tokenizer,sen_size,word_char_size,CNN_Vec_size)

print(top_word_size)


model = keras.models.load_model(root+"Saved_Model")

model.fit(training_generator,validation_data=valid_generator, epochs=epoch_number, verbose=1,
                    workers=6)

model.save(root+"Saved_Model")