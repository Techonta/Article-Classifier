#!/usr/bin/env python
# coding: utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
import data_processing_util

catalog_dic = get_catalog_dic(drop_else = False)
df_allcorpus = pd.read_csv('data/new_corpus_20180808_renewed.csv')
with open('data/tokenizer.pickle', 'rb') as handle:
    words_idx = pickle.load(handle).word_index
    print('Total words:', len(words_idx))


#modify the dictionary to fit the usage of scikit-learn which specified the index of dictionary should start from 0.
words_idx['unknow_word'] = 0


def category_tfidf_generater(catalog_dic, corpus_dataframe, words_idx, nor = False):
    tfidf_ary = np.array([]).reshape(0,len(words_idx))
    for c in catalog_dic.keys():
        if c in set(corpus_dataframe['Catalog_Y']):
            df = corpus_dataframe.query("Catalog_Y == @c")
            vectorizer_w = TfidfVectorizer(analyzer = 'word', vocabulary = words_idx)
            ary_word = vectorizer_w.fit_transform(df['context_seg']).toarray()  #iteratively input the corpus by category and return is the matrix of documents * words.
            tfidf_sum = np.sum(ary_word, axis=0).reshape((1, len(words_idx)))   #sum tfidf scores for each word
            tfidf_count = np.count_nonzero(ary_word, axis=0).reshape((1, len(words_idx))) #count how many tfidf scores
            if nor:
                tfidf_nor = tfidf_sum/tfidf_count                               #average the tfidf scores
                tfidf_ary = np.vstack([tfidf_ary, tfidf_nor])                   #stack all 
            else:
                tfidf_ary = np.vstack([tfidf_ary, tfidf_sum])                   #stack all the summed-only for tfidf scores
        else:
            print(c, 'is not in catalogs.')
    
    #shift the first column which is tfidf scores of 'unknow_word' to the last
    tfidf_ary_n = np.delete(tfidf_ary, 0, axis = 1)
    tfidf_ary_n.shape
    
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

tfidf_ary = category_tfidf_generater(catalog_dic, df_allcorpus, words_idx, nor = True)
