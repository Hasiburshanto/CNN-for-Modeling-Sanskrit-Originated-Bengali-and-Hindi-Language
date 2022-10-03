# You can change the following. These will affect your model training and validation 
word_char_size=10+2 ## Maximum 10 characters in one word. The extra two are the beginning and end marker

sen_size=15 ## Maximum word number in a sentence.

# top_word_unique_char_no:: We only consider those characters which are in the top top_word_unique_char_no words
#                           according to frequncy.
top_word_unique_char_no=25000 

# coverage_of_words :: We take those words in our top word list to be predicted by our model which cover
#                      coverage_of_words portion of our entire corpus
coverage_of_words=0.90

#sentence_topWord_percentage :: We shall only take those sentences as samples, where there are at least
#                               sentence_topWord_percentage portion of top words.
sentence_topWord_percentage=85 
CNN_Vec_size=256
Dropout_Rate=0.3
char_vec_size=40  # char2vec
learning_rate=0.001 
clipnorm=5.0  # gradient clipping threshold to avoid gradient exploding problem 