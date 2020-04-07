#!/usr/bin/env python
# coding: utf-8
import numpy as np
from keras.layers.embeddings import Embedding
from sklearn.model_selection import StratifiedKFold


#Load the embedding matrix
emb_matrix = np.load('emb_128_so_tfidf.npy')
emb_matrix.shape

def pretrained_embedding_layer(emb_matrix):    
    
    vocab_len, emb_dim = emb_matrix.shape
    
    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer =  Embedding(vocab_len, emb_dim, trainable = False)

    # Build the embedding layer, it is required before setting the weights of the embedding layer.
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

embedding_layer = pretrained_embedding_layer(emb_matrix)
embedding_layer.get_config()


#Define the model
from keras.models import Model, load_model
from keras.layers import Dense, Input, Dropout, GRU, LSTM, Activation, Conv1D, Bidirectional
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.optimizers import Adam

def auto_tagging(input_shape, emb_matrix, y_dim):
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors
    embedding_layer = pretrained_embedding_layer(emb_matrix)
    
    embeddings = embedding_layer(sentence_indices)    
    X = Bidirectional(LSTM(256, return_sequences=False))(embeddings)
    X = Dropout(0.5)(X)
    X = Dense(y_dim, activation='softmax')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, output=X)
        
    return model

opt = Adam(lr = 0.001, beta_1=0.9, beta_2=0.999, decay = 0.01) # default


#Stratified K-fold cross-validation
x = np.load('x_all.npy')
y = np.load('y_all.npy')

kf = StratifiedKFold(n_splits=10) #DO NOT set shuffle to True.
kf.get_n_splits(x, y)

#For stratified K-fold, the split() method need to be fed by a 1-D array of y.
index_list = []
for idx in range(y.shape[0]):
    if np.nonzero(y[idx,:])[0].size == 0:
        index_list.append(y.shape[1])
    else:
        index_list.append(np.argmax(y[idx,:]))


#K-fold cv
training_records = []
test_records = []
cnt = 1
for e in kf.split(x, np.asarray(index_list)):
    print('Training round',cnt)
    
    x_training_set = x[e[0]]
    y_training_set = y[e[0]]
    x_test_set = x[e[1]]
    y_test_set = y[e[1]]
    
    y_dim = y_training_set.shape[1] # for softmax
    x_maxlen = len(x_training_set[0])
    #y_dim = 1 # for binary
    model = auto_tagging((x_maxlen,), emb_matrix, y_dim)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['categorical_accuracy'])
    
    hist = model.fit(x_training_set, y_training_set, epochs = 11, batch_size = 1024, shuffle=True).history
    training_records.append(hist['categorical_accuracy'][-1])
    print()
    loss, acc = model.evaluate(x_test_set, y_test_set)
    print("Test accuracy = ", acc)
    test_records.append(acc)
    print()
    #saving arraies and indies
    if cnt == kf.get_n_splits(x, y):
        np.save('training_index.npy', e[0])
        np.save('test_index.npy', e[1])
        np.save('x_training_set.npy', x_training_set)
        np.save('y_training_set.npy', y_training_set)
        np.save('x_test_set.npy', x_test_set)
        np.save('y_test_set.npy', y_test_set)
    cnt = cnt + 1

model.save('auto_tagging_emb_128_so_tfidf_0730.h5')
