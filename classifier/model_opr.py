#!/usr/bin/env python
# coding: utf-8
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Model, load_model
from data_processing_util import *
import pandas as pd
from new_cut_sentence import *
from statistics import mode

model = load_model('<model_name>')

inv_catalogs_dic = get_catalog_dic(inversed = True, drop_else = True)

def predict(input_string, show_segments = None):
    sample = jieba_segmentation([input_string], stopword_set)
    if show_segments:
        print('Jieba segmentaion:', sample)
    total_sequence = tokenizer.texts_to_sequences(sample)
    sequences = pad_sequences(
        total_sequence,
        maxlen = 25,
        padding = 'post',
        truncating = 'post')
    prediction = model.predict(sequences)
    return prediction.max(), inv_catalogs_dic[np.argmax(prediction)]

corpus = sentences()
def corpus_analysis(corpus):
    category_list = []
    percentage_list = []
    result = {}
    for sentence in corpus:
        print(sentence, predict(sentence))
        percentage, category = predict(sentence)
        if percentage > 0.5: #collect all categories which probability > 0.5
            category_list.append(category)
            percentage_list.append(percentage)
    if category_list == []: return 'ELSE'
    try:
        corpus_category = mode(category_list) #use the mode of collected categories as the category of the article
        return corpus_category
    except Exception as e:
        print(e)
        for category in set(category_list): #if there is no unique mode, compare the sum of probabilities of each category and choose the maximium
            category_percentage = []
            for idx in (np.where(np.asarray(category_list)==category)[0]): #get every score for the category
                category_percentage.append(percentage_list[idx])
            result[sum(category_percentage)] = category
        return result[max(result.keys())]
