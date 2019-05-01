# coding: utf-8
# - This job is to cleanse the repeated article, doen't rearrange the order of the sentences in the article. The premise of this job is that the article need to be structured on its correct begining.

import math
from statistics import mode
import pandas as pd
import pymysql.cursors
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import math
import time
import sys



def fine_repeat_unit(input_str):
    #define basic parameters
    fraction = 2 #Determines how to split the article. If fraction = 2, the article would be split by the dichotomy.
    sL = 3       #the shortest length of the finally spilited string
    max_split = int(math.log((len(input_str)/sL),fraction)) #The maximum number of times for the spliting.
    
    #Precheck the length of input_str is >= the threshold length so that prevent the max_split to be 0.
    """
    If the max_split need to be >= 1, then math.log(len(input_str)/3,2) need to be >= 1
    and len(input_str)/3 need to be >=2, len(input_str) need to be >=6.
    """
    length_limit = fraction * sL
    if len(input_str) < length_limit:
        return input_str
    
    #define the help function for sub string extraction
    def sub_str(target_string, fraction):
        idx_fraction = int((len(target_string)/fraction))
        sub_string = target_string[0:idx_fraction]
        return sub_string
    
    #define the help function for chosing small mode
    def take_samll_mode(dic):
        max_rpt = 0
        elm_same_rpt = []
        rpt_list = list(set(dic.values()))
        rpt_list.sort(reverse=False)
        ary = np.asarray(list(dic.values()))
        #find the mode of the repeat times by thier frequence. The max_rpt means the max frequence of those repeat times
        for n in rpt_list:
            if len(np.where(ary == n)[0].tolist()) > max_rpt:
                max_rpt = len(np.where(ary == n)[0].tolist())
        #find the repeat times with the frequence equal to max_rpt and append them into the list of elm_same_rpt.
        for n in rpt_list:
            if len(np.where(ary == n)[0].tolist()) == max_rpt:
                elm_same_rpt.append(n)
        #use the minimum of those repeat times
        #return min(elm_same_rpt)
        elm_same_rpt.sort()
        if elm_same_rpt[0] == 1:
            return elm_same_rpt[1]
        else:
            return elm_same_rpt[0]
    
    """
    Fine the possible repeat counts with the format, {fraction, repeat frequence under the fraction}, 
    ex: if an article with structure ABAB, which A and B are paragraphs in article, is splited 2 parts
    and the sub string is shown twice, one of the record would be {1/2: 2}
    """
    repeat_counts = {}
    
    for split_cnt in range(max_split):   #If the max_split is 0, the repeat_counts would be empty.
        sub_string = sub_str(input_str, fraction)
        if sub_string == '':    #prevent the sub-string go into ''
            break
        #print(sub_string)
        repeat_counts['1/'+str(fraction)] = input_str.count(sub_string)
        fraction = fraction*2
        
    #print(repeat_counts)
    
    #use the mode as the possible repeat counts
    try:
        mode_repeats = mode(list(repeat_counts.values()))
    except Exception as e:
        mode_repeats = take_samll_mode(repeat_counts)
    
    #get the index of smallist unit repeated by mode_repeats times
    temp_list = list(repeat_counts.values())
    temp_list.reverse()
    #print(len(list(repeat_counts.items())),temp_list.index(mode_repeats))
    a = len(list(repeat_counts.items()))-temp_list.index(mode_repeats)-2
    repeated_unit_frac = int(list(repeat_counts.items())[a][0].split('/')[1])

    #sample the entity
    repeated_unit = sub_str(input_str, repeated_unit_frac)
    return repeated_unit



def search_article(raw, repeated_unit):
    """
    Find all of the starts for repeated_unit, and caculate their index (strats from 0) distance as cardinal.
    """
    index = 0
    index_pre = 0
    cardinal = 1 #any positive integer
    backward_distances = {}
    while index < len(raw):
        #print('start to find string at', index)
        index = raw.find(repeated_unit, index)
        distance = index-index_pre
        backward_distances[distance] = index
        #print('entity found at', index, 'pre:', index_pre)

        if index == -1: #can't find the string in the following text
            #print('backward_distances:', backward_distances)
            max_dist = max((list(backward_distances.keys())))
            end = backward_distances[max_dist]
            if end == 0:
                return raw
            else:
                if len(raw[end:])<=len(raw[end-max_dist:end]):
                    return raw[end-max_dist:end].replace(u'\xa0', u' ')
                else:
                    return raw[end:].replace(u'\xa0', u' ')
            #print(max_dist)
            break

        index_pre = index
        index += len(repeated_unit)
        if (index % len(raw) == 0):
            index = index - 1
        


def get_repeated_segment_index(article,window): #return the earlist index of repeated segments
    #define helper function
    def window_collect(window, text):
        strides = 1
        index = 0
        segments_list=[]
        while index+window < len(text):
            segments_list.append(text[index:index+window])
            index = index + strides
        segments_list.append(text[index:])
        segments_ary = np.asarray(segments_list)
        return segments_ary
    
    segments_index = {}
    segments_array = window_collect(window, article)
    for segment in list(set(segments_array)):
        if len(np.where(segments_array == segment)[0]) > 1: #only collects the index of repeated segment
            segments_index[segment] = article.find(segment)
    try:
        return min(segments_index.values()) #return the earlist index of repeated segments
    except:
        return -1


def internal_deduplicate(text, windowForRepeatedSengment):
    string_list = []
    idx = 0
    
    #define helper function
    def get_initial_string(text, repeate_seg_index, windowForRepeatedSengment):
        if repeate_seg_index != 0:
            return text[:repeate_seg_index]
        else:
            seg = text[repeate_seg_index:repeate_seg_index+windowForRepeatedSengment]
            next_idx = text.find(seg, repeate_seg_index+1)
            return text[next_idx:next_idx+windowForRepeatedSengment]
    
    while idx != -1:
        idx = get_repeated_segment_index(text, windowForRepeatedSengment) #minimum index where repeated segment was found
        #print('idx:', idx)
        if idx == -1: break
        initial_string = get_initial_string(text, idx, windowForRepeatedSengment) #collect the initial_string based on repeated_segment_index. initial_string is the initial of the string not repeated.
        #print('initial_string:', initial_string)
        string_list.append(initial_string)
        #print('text[len(initial_string):]',text[len(initial_string):])
        #print('fine_repeat_unit(text[len(initial_string):])', fine_repeat_unit(text[len(initial_string):]))
        #check if the length of output from fine_repeat_unit() is > 1. If not, it could be a last piece of the text and appended into the list
        if len(fine_repeat_unit(text[len(initial_string):])) == 1:
            string_list.append(fine_repeat_unit(text[len(initial_string):]))
            break
        text = search_article(text[len(initial_string):], fine_repeat_unit(text[len(initial_string):]))
        #print('text:', text)
        #print('Round end.\n')
    t = ''
    for seg in string_list:
        t = t + seg
    if text in t:
        return t
    else:
        return t+text



def update_article(raw_article):
    repeat_unit = fine_repeat_unit(raw_article)
    article = search_article(raw_article, repeat_unit)
    window = int(len(article)*0.07)
    while (get_repeated_segment_index(article, window) != (-1)):
        #print('internally repeated structure detected.')
        article = internal_deduplicate(article, window)
    return article



if __name__ == '__main__':
    with open(sys.argv[1], 'r') as file:
        raw_article = file.read().replace('\n', '')
    print()
    print('raw article:')
    print(raw_article)
    cleanArticle = update_article(raw_article)
    print()
    print('cleaned article:')
    print(cleanArticle)
    print()

