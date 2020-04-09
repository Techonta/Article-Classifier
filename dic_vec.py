#!/usr/bin/env python
# coding: utf-8
from gensim.models import word2vec
from gensim import models
import numpy as np
import pandas as pd
from data_processing_util import *
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import logging
from logging.handlers import RotatingFileHandler
import time

def init_log(log_file_name):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')
    logFile = log_file_name+'_'+timestr+'.log'
    handler = RotatingFileHandler(logFile, mode='a', maxBytes=5*1024*1024, 
                                     backupCount=2, encoding=None, delay=0)
    handler.setFormatter(log_formatter)
    handler.setLevel(logging.INFO)
    app_log = logging.getLogger('root')
    app_log.setLevel(logging.DEBUG)
    app_log.addHandler(handler)
    return app_log

def create_dictionary(corpus_Series, tokenizer_name=None, to_disk=True):
    token = Tokenizer(oov_token='unknow_word')
    token.fit_on_texts(list(corpus_Series))
    print('Total words:', len(token.word_index))
    if to_disk:
        if tokenizer_name==None:
            print('Can not save the tokenizer. Please provide a valid file name of tokenizer_name.')
            return
        with open(tokenizer_name, 'wb') as handle:
            pickle.dump(token, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(tokenizer_name, 'saved.')
    return token.word_index

def categorial_tfidf_generater(catalog_dic, corpus_dataframe, words_idx, nor = False):
    #modify the dictionary to fit the usage of scikit-learn which specified the index of dictionary should start from 0.
    words_idx['unknow_word'] = 0
    tfidf_ary = np.array([]).reshape(0,len(words_idx))
    for c in catalog_dic.keys():
        if c in set(corpus_dataframe['Catalog_Y']):
            df = corpus_dataframe.query("Catalog_Y == @c")
            vectorizer_w = TfidfVectorizer(analyzer = 'word', vocabulary = words_idx)
            ary_word = vectorizer_w.fit_transform(df['context_seg']).toarray()
            tfidf_sum = np.sum(ary_word, axis=0).reshape((1, len(words_idx)))   #sum tfidf scores for each word
            if nor:
                tfidf_nor = tfidf_sum/df.count()[0]                             #divide sum of the tfidf scores by documents as the vectors to represent of the category
                tfidf_ary = np.vstack([tfidf_ary, tfidf_nor])                   #stack all 
            else:
                tfidf_ary = np.vstack([tfidf_ary, tfidf_sum])                   #stack all the summed-only for tfidf scores
        else:
            print(c, 'is not in catalogs.')

    
    #shift the first column which is tfidf scores of 'unknow_word' to the last.
    #As the tfidf scores of 'unknow_word' are zeros, let's delete the first column and add a zeros-column to the last.
    tfidf_ary_n = np.delete(tfidf_ary, 0, axis = 1)
    zero=np.zeros((tfidf_ary.shape[0],1))
    tfidf_ary_n = np.hstack((tfidf_ary_n, zero))
    if nor:
        np.save('tfidf_weight_averaged.npy', tfidf_ary_n)      #normalization
        print('Normalized tfidf array output.')
    else:
        np.save('tfidf_weight_summed_only.npy', tfidf_ary_n)   #sum only
        print('Summed-only (Unnormalized) tfidf array output.')
    return tfidf_ary_n

def word2vec_training(corpusfile, emb_dim, to_disk=True):
    sentences = word2vec.LineSentence(corpusfile)
    model = word2vec.Word2Vec(sentences, window=5, size=emb_dim, min_count=2, sg=1, negative=10, compute_loss=True)
    print('loss:', model.get_latest_training_loss())
    if to_disk:
        model.save("word2vec.model")
    return model 

def create_mapping_dic(gensim_Word2Vec_model): 
    word_to_vec_map = {}
    vocab = gensim_Word2Vec_model.wv.vocab
    for word in vocab.keys():
        word_to_vec_map[word] = gensim_Word2Vec_model[word]
    return word_to_vec_map

def get_embedding_matrix(word_to_vec_map, word_to_index, gensim_Word2Vec_model):    
    app_log = init_log('word2vec_training')
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    e = gensim_Word2Vec_model.wv.index2word[0]
    emb_dim = word_to_vec_map[e].shape[0]               # define dimensionality of GloVe word vectors
    
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        try:
            emb_matrix[index, :] = word_to_vec_map[word]
        except:
            app_log.info('%s is not in word_to_vec_map.', word)
            continue
    
    return emb_matrix

def word2vec_embedding(Series, emb_dim, words_dic, model_to_disk=True):
    save_list_to_file(list(Series), 'segments.txt')
    gensim_Word2Vec_model = word2vec_training('segments.txt', emb_dim, model_to_disk)
    #delete temp file
    if os.path.exists('segments.txt'): os.remove('segments.txt')
    #create dictionary {'word': vector}
    word_to_vec_map = create_mapping_dic(gensim_Word2Vec_model)
    emb_matrix = get_embedding_matrix(word_to_vec_map, words_dic, gensim_Word2Vec_model)
    print('Embedding matrix generated with shape', emb_matrix.shape)
    return emb_matrix

def tfidf_concat(tfidf_weights, emb_matrix, to_disk=True):
    tfidf_weights_trp = tfidf_weights.transpose() #(categories, words) --> (words, cateogries)
    zero = np.zeros((1,tfidf_weights.shape[0]))
    # adding zeros to 1st row to fit Keras embedding (requirement)
    tfidf_weights_trp = np.vstack([zero, tfidf_weights_trp])
    emb_tfidf = np.concatenate((emb_matrix, tfidf_weights_trp), axis=1)
    if to_disk:
        timestr = time.strftime("%Y%m%d_%H%M%S")
        np.save('emb_tfidf_'+timestr+'.npy', emb_tfidf)
        print('emb_tfidf_'+timestr+'.npy saved.')
    return emb_tfidf


#Create tokens by Keras, saving by pickle
df_all = pd.read_csv('data/new_corpus_20180809_renewed.csv')
catalog_dic = get_catalog_dic(drop_else = False)
words_dic = create_dictionary(df_all['context_seg'],
                  tokenizer_name='data/tokenizer_20180809.pickle',
                  to_disk=True
                 )

tfidf_ary = categorial_tfidf_generater(catalog_dic, df_all, words_dic, nor = False)
emb_matrix = word2vec_embedding(df_all['context_seg'], 128, words_dic, model_to_disk=True)
emb_tfidf = tfidf_concat(tfidf_ary, emb_matrix, to_disk=True)






