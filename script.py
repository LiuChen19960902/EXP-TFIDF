import time
import re
import jieba
import random
from collections import Counter
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def tf(word, count):
    return count[word] / sum(count.values())


def n_containing(word, count_list):
    return sum(1 for count in count_list if word in count)


def idf(word, count_list):
    return math.log(len(count_list) / (1 + n_containing(word, count_list)))


def df(word, count_list):
    return math.log(1 + n_containing(word, count_list))


def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)


def tfdf(word, count, count_list):
    return tf(word, count) * df(word, count_list)


def get_count_list(sentence_list):
    count_list = []
    for i in range(len(sentence_list)):
        count = Counter(sentence_list[i])
        count_list.append(count)
    return count_list


def pad_sentence(sentence, max_len):
    """
    :param sentence: list
    :param max_len:
    :return:
    """
    while len(sentence) < max_len:
        sentence.append(0)
    return sentence


def get_cosine_similarity(vec1, vec2):
    result = np.squeeze(cosine_similarity(vec1, vec2)).tolist()
    return result


def sklearn_tfidf(corpus):
    c = [' '.join(i) for i in corpus]
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(c))
    weight = tfidf.toarray()
    vec = []
    for index, s in enumerate(c):
        vec.append((c[index], weight[index][np.newaxis, :]))
    return vec, len(vectorizer.get_feature_names())


def stopwordlist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def get_feature(input_file, stopword_file, sample=False):
    f = open(input_file, 'r', encoding='utf-8')
    stopwords = stopwordlist(stopword_file)
    vocab = []
    sentence_list = []
    # sentence_index_list = []
    sentence_len = []
    all_sentences = f.readlines()

    if sample:
        sample_span = list(range(len(all_sentences)))
        sample_sentence = random.sample(sample_span, 10000)
        for sample in sample_sentence:
            line = all_sentences[sample].strip('\n').split('\t')
            row = line[1]
            pattern = '[\u4e00-\u9fa5]'
            result = re.compile(pattern).findall(row)
            if len(result) != 0:
                # new_text.write(str(n) + '\t' + row + '\n')
                jieba_cut = ' '.join(jieba.cut(row)).split(' ')
                cut = []
                for word in jieba_cut:
                    if word not in vocab:
                        vocab.append(word)
                    if word not in stopwords:
                        cut.append(word)
                # print(cut)
                sentence_list.append(cut)
                # sentence_index_list.append(n)
                # n += 1
                if len(cut) not in sentence_len:
                    sentence_len.append(len(cut))
        return sentence_list
    else:
        for i in all_sentences:
            line = i.strip('\n').split('\t')
            row = line[1]
            pattern = '[\u4e00-\u9fa5]'
            result = re.compile(pattern).findall(row)
            if len(result) != 0:
                # new_text.write(str(n)+'\t'+row+'\n')
                jieba_cut = ' '.join(jieba.cut(row)).split(' ')
                cut = []
                for word in jieba_cut:
                    if word not in vocab:
                        vocab.append(word)
                    if word not in stopwords:
                        cut.append(word)

                sentence_list.append(cut)
                # sentence_index_list.append(n)
                # n += 1
                if len(cut) not in sentence_len:
                    sentence_len.append(len(cut))
        return sentence_list


def cal_cosine(outfile, sklearn_vec, mode=None):
    """
    :param outfile:
    :param sklearn_vec:
    :param mode: [avg, cosine]
    :return:
    """
    outfile2 = open(outfile, 'a', encoding='gbk')
    if mode == 'cosine':
        outfile2.write('句子1,相似度' + '\n')
        for sent_index2, (sent1_, vec1_) in enumerate(sklearn_vec):
            start = time.perf_counter()
            init_cosine = 0
            for (sent2_, vec2_) in sklearn_vec[sent_index2+1:]:
                cosine_ = get_cosine_similarity(vec1_, vec2_)
                init_cosine += cosine_
            end = time.perf_counter()
            print(sent_index2, '/', len(sklearn_vec), 'time is: ', end-start, 's')
            outfile2.write(' '.join(sent1_).replace(' ', '') + ',' + str(init_cosine)+'\n')
    else:
        outfile2.write('句子1,均值' + '\n')
        repeat = []
        for sent_index2, (sent1_, vec1_) in enumerate(sklearn_vec):
            print(sent_index2, '/', len(sklearn_vec))
            # mean = np.sum(np.squeeze(vec1_)) / np.log(len(sent1_.split(' '))+1)  # tfdf
            mean = np.sum(np.squeeze(vec1_)) / math.pow(len(sent1_.split(' ')), 1/3)  # 1_2, 1_3, 1_4 开方

            s = sent1_.replace(' ', '')
            if sent1_ not in repeat:
                repeat.append(sent1_)
                try:
                    outfile2.write(s+','+str(mean)+'\n')
                except UnicodeEncodeError:
                    pass


if __name__ == '__main__':
    files = ['测试', '前端开发', '人工智能', '移动开发', '游戏', '数据']
    # files = ['后端开发']
    for file in files:
        print(file)
        sentence_list = get_feature(input_file='data/'+file+'.txt', stopword_file='stopwords.txt', sample=False)
        sklearn_vec, dim = sklearn_tfidf(sentence_list)
        print('dim is {}'.format(dim))
        cal_cosine('data/'+file+'1_3.csv', sklearn_vec, 'avg')