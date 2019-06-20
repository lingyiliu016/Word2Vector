import os
import re
import jieba
import numpy as np


absolute_file_name_list = []
sentence_list =[]
word_segment_list = []
vocabulary_list = []
word_to_index_map = {}
index_to_word_map = {}
train_data_set_list = []
# 测试产物，后可删除
temp_list = []

def get_file_list(data_path):
    file_list = os.listdir(data_path)
    # absolute_file_name_list = []
    for file_name in file_list:
        absolute_file_name = data_path + file_name
        absolute_file_name_list.append(absolute_file_name)


def get_sentence_from_file(file_name):
    with open(file_name,'rt') as f:
        data = f.read()
        temp_list = re.split(r"？”|！”|。”|\?”|!”|\.“|。|！|？|\.|!|\?|……", data)
        for sentence in temp_list:
            if sentence != '':
                sentence_list.append(sentence.replace('\n','').replace(u'\u3000',u''))

def word_segment(sentence):
    sentence_word_list = []
    temp = jieba.cut(sentence, cut_all=False)
    for word in temp:
        sentence_word_list.append(word)
    word_segment_list.append(sentence_word_list)

def get_vocabulary():
    for sentence_segment in word_segment_list:
        for word in sentence_segment:
            if word not in vocabulary_list:
                vocabulary_list.append(word)

def map_word_and_index():
    for index, word in enumerate(vocabulary_list):
        word_to_index_map[word] = index
        index_to_word_map[index] = word

def get_skip_gram_train_data_set(windows_size):
    for sentence in word_segment_list:
        for index, word in enumerate(sentence):
            for i in range(index - windows_size, index + windows_size + 1):
                if i >= 0 and i < len(sentence) and i != index:
                    target_index = word_to_index_map[word]
                    context_index = word_to_index_map[sentence[i]]
                    train_data_set_list.append((target_index, context_index))
                    # 测试产物，可删除
                    temp_list.append((word,sentence[i]))


def prepare_data(windows_size):
    data_path = "/Users/cly/PycharmProjects/Word2Vec/data/"
    get_file_list(data_path)

    for file_name in absolute_file_name_list:
        get_sentence_from_file(file_name)

    for sentence in sentence_list:
        word_segment(sentence)

    get_vocabulary()

    map_word_and_index()

    get_skip_gram_train_data_set(windows_size)

    # for sentence_segement in word_segment_list:
    #     print(sentence_segement)
    #
    # for (key, value) in word_to_index_map.items():
    #     print("word:", key, " index:", value)
    #
    # for (key,value) in index_to_word_map.items():
    #     print("key:", key, " value:", value)
    #
    # for element in temp_list:
    #     print(element)
    #
    # for element in train_data_set_list:
    #     print(element)




def generate_batch(batch_size):
    input = np.ndarray(shape=(batch_size), dtype=np.int32)
    label = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    train_data_length = len(train_data_set_list)
    # print("训练集长度：",train_data_length)
    random_list = np.random.randint(train_data_length,size=batch_size)
    # print("random_list:", random_list)
    for i, index in enumerate(random_list):
        input[i]   = train_data_set_list[index][0]
        label[i,0] = train_data_set_list[index][1]

    return input, label



if __name__ == "__main__":
    prepare_data(windows_size=3)
    input, label = generate_batch(batch_size=128)
    for index in range(128):
        print("input:",input[index], "\t label:",label[index])