# coding: utf-8
import jieba
import pandas as pd
import numpy as np
import ast

def load_jieba_fine_tune(filepath, list_type=None):
    if list_type==None:
        print("Please provide the Fine-tune list type, 'split' or 'join'.")
        return
    if list_type == 'split':
        tuplized = []
        temp = read_list(filepath)
        for entity in temp:
            tree = ast.parse(entity, mode='eval')
            clause = compile(tree, '<AST>', 'eval')
            #x = eval(clause)
            tuplized.append(eval(clause))
        return tuplized
    if list_type == 'join':
        temp = read_list(filepath)
        return temp

def jieba_config(stop_words_path, customized_dic_path):
    # load stopwords set
    stopword_set = set()
    with open(stop_words_path,'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))

    # load dictionary
    jieba.load_userdict(customized_dic_path)
    
    return stopword_set


def data_preparing_fromcsv(corpusSource_csv):
    df = pd.read_csv(corpusSource_csv) #use the Pandas read_csv() to open and check file format.
    if 'context' in list(df):
        context_df = df['context']
        context_list = context_df.values.tolist()
        if context_list[0] == 'id,context,label\n': del context_list[0]
    return context_list



# Generate segments at Python list
def jieba_segmentation(corpusSource_list, stopword_set):
    totalsamplelines=len(corpusSource_list)
    samplelines_seg=[]
    for i in range(totalsamplelines):
        """# This part should be put into the data_preparing()
        if isSplitted == False:
            replaced = corpusSource_list[i].split(',')[0]+','+corpusSource_list[i].split(',')[1]+','
            text=corpusSource_list[i].replace(replaced, "").strip('\n').lower()
        """
        text = corpusSource_list[i].strip('\n').lower()
        words = jieba.cut(text,cut_all=False)
        segment = " "
        for word in words:
            #word = word.strip(',')
            if word not in stopword_set:
                segment = segment + word + ' '
        samplelines_seg.append(segment)
    return samplelines_seg


#Serach every segment in given DataFrame. Mapping and Fetching the tag when the segment exists in the key word column.
def tags(samplelines_seg_list, mapping_dataframe):
    tags_dic = {}
    untagged = 0
    num = len(samplelines_seg_list)
    for i in range(num):
        #print(samplelines_seg_list[i])
        tags = set()
        for seg in samplelines_seg_list[i].split():
            #print(seg)
            #if seg == '翻譯年糕': continue
            index_tuple = np.where(mapping_dataframe.values == seg)
            #print(index_tuple)
            if len(index_tuple[0]) == 0: continue
            if mapping_dataframe.values[index_tuple[0][0]][1] == '0':
                tag = mapping_dataframe.values[index_tuple[0][0]][0]
            else:
                tag = mapping_dataframe.values[index_tuple[0][0]][1]
            tags.add(tag)
        #print(tags)
        if len(tags) == 0: untagged = untagged + 1
        tags_dic[i] = tags
    print('',
          'Untagged sentences:', untagged, '\n',
          'tagged sentences:', len(samplelines_seg_list)-untagged, '\n',
          'Total sentences:', len(samplelines_seg_list)
         )
    return tags_dic


def get_mapping_df():
    df = pd.read_csv('data/personal_interest_category_keyword.csv')
    mapping_df = df[['interest_category_2', 'interest_category_3', 'key_word']]
    return mapping_df


def get_csv_miss_labeled(filename, context_list, tags_dic):
    csv_miss_cnt = 0
    #creat csv file with specified catalog name
    catalog_name = filename.split('_')[-1].split('.')[0]
    file = open(catalog_name + '_recheck.csv','w')
    file.write('line_index,context,csv_label,new_label,check')
    file.write('\n')
    file.flush()
    #miss-labeled data in csv (jieba_positive - csv_positive)
    csv_positive_df = pd.read_csv(filename).query('label == 1')['context']
    for i in range(len(context_list)):
        output_str = ''
        if catalog_name in tags_dic[i]:
            if context_list[i] not in csv_positive_df.values:
                csv_miss_cnt = csv_miss_cnt + 1
                output_str = str(i+2) + ',' + context_list[i] + ',0' + ',1'
                #print(output_str)
                file.write(output_str)
                file.write('\n')
                file.flush()
    file.close()
    #print(csv_miss_cnt)

#miss-labeled by Jieba (csv_positive - jieba_positive)
def jieba_miss_label(filename, stopword_set, tags_dic, context_list):
    jieba_miss_cnt = 0
    jieba_missed_lines_seg = []
    jieba_positive_list = []
    csv_positive_df = pd.read_csv(filename).query('label == 1')['context']
    tagged_from_db = csv_positive_df.values.tolist()
    catalog_name = filename.split('_')[-1].split('.')[0]
    for i in range(len(tags_dic)):
        if catalog_name in tags_dic[i]:
            jieba_positive_list.append(context_list[i])
    for line in tagged_from_db: 
        if line not in jieba_positive_list:
            jieba_miss_cnt = jieba_miss_cnt + 1
            words = jieba.cut(line,cut_all=False)
            segment = ""
            for word in words:
                if word not in stopword_set:
                    segment = segment + word + ' '
            else:
                jieba_missed_lines_seg.append(segment)
                print(line, '--->', segment)
    print('Jieba missed:', jieba_miss_cnt)
    return jieba_missed_lines_seg

#key words in database check
def show_key_words_indatabase(query_str, mapping_dataframe):
    df_testlist = mapping_dataframe['key_word'].values.tolist()
    for kw in df_testlist:
        if kw in query_str:
            print(kw)

#Jieba fine-tune
def jieba_fine_tune(jion_list, split_list):
    for j_item in jion_list:
        jieba.suggest_freq(j_item, True)
    
    for s_item in split_list:
        jieba.suggest_freq(s_item, True)


def save_list_to_file(list_name, file_name, delimiter=None):
    if (delimiter==None): delimiter=';;;'
    file_path = file_name
    thefile = open(file_path, 'w')
    for item in list_name:
        thefile.write(str(item)+delimiter)
    thefile.close()
    print(file_name, 'saved.')

def jieba_fine_tune_list_renew(split_list, join_list):
    save_list_to_file(split_list ,'data/jieba/jieba_split_list.txt')
    save_list_to_file(join_list ,'data/jieba/jieba_join_list.txt')
    print('Jieba fine-tune renew is done.')    

    
def read_list(filepath):
    join_list= []
    text_file = open(filepath, 'r')
    text_file.close
    temp = str(text_file.readlines()[0]).split(';;;')
    if temp[-1] == '': del temp[-1]
    return temp

def get_catalog_dic(inversed=False, drop_else=True):
    #get mapping DataFrame
    mapping_df = get_mapping_df()
    #put catalogs into Set to deduplicate
    catalogs_set = set()
    for line in mapping_df.values:
        if line[1] == '0':
            catalogs_set.add(line[0])
        else:
            catalogs_set.add(line[1])
    #Add 'ELSE' to catalogs or not.
    if (drop_else == False):
        catalogs_set.add('ELSE')
    catalogs_list = list(catalogs_set)
    catalogs_list.sort() #sort() prevents the ordering issue of the set().
    #put catalogs into dictionary with specified index
    idx = 0
    catalogs_dic = {}
    for item in catalogs_list:
        catalogs_dic[catalogs_list[idx]] = idx
        idx = idx + 1
    #inversion or not
    if inversed == False:
        return catalogs_dic
    else:
        inv_catalogs_dic = {v: k for k, v in catalogs_dic.items()}
        return inv_catalogs_dic
