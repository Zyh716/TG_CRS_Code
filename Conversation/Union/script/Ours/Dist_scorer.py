import argparse
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import json
import ipdb
import re
import jieba

word2index = json.load(open("../../data/tf_bpe2index.json", encoding='utf-8'))
jieba.load_userdict('../../data/dict.txt')


def back_process(sen):
    sentences = ''
    new_sen = []
    for word in sen:
        if word == 'ï¼Œ':
            s1 = ' '.join(new_sen)
            new_sen = []
            if s1 in sentences:
                break
            else:
                if sentences != '':
                    sentences += ' ï¼Œ ' + s1
                else:
                    sentences += s1
        else:
            new_sen.append(word)

    sentences2nd = sentences.split()
    try:
        final_sentence = [sentences2nd[0]]
    except:
        return []
    for i in range(1, len(sentences2nd)):
        if sentences2nd[i] == sentences2nd[i - 1]:
            break
        else:
            final_sentence.append(sentences2nd[i])
    return final_sentence


def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1) * np.dot(v2, v2)) + 1e-4)


def sen2id(sen):
    candidates = []
    for word in sen:
        #if word not in stopwords and word in word2index:
        if word in word2index:
            candidates.append(word2index[word])
    #return list(set(candidates))
    return candidates


def dist_cal(tokenized_gen, tokenized_tar):
    scores = []
    dis1 = 0
    dis_set1 = set()
    dis2 = 0
    dis_set2 = set()
    dis3 = 0
    dis_set3 = set()
    dis4 = 0
    dis_set4 = set()
    output = []
    for sen, tar in zip(tokenized_gen, tokenized_tar):
        sen = sen[0]
        tar = tar[0]

        sen = re.sub(r'《(.*)》', '<movie>', sen)
        sen_split_by_movie = list(sen.split('<movie>'))
        sen = []
        for i, sen_split in enumerate(sen_split_by_movie):
            for segment in jieba.cut(sen_split):
                sen.append(segment)
            if i != len(sen_split_by_movie) - 1:
                sen.append('<movie>')

        tar = re.sub(r'《(.*)》', '<movie>', tar)
        tar_split_by_movie = list(tar.split('<movie>'))
        tar = []
        for i, tar_split in enumerate(tar_split_by_movie):
            for segment in jieba.cut(tar_split):
                tar.append(segment)
            if i != len(tar_split_by_movie) - 1:
                tar.append('<movie>')

        golden = tar
        prediction = sen

        for word in prediction:
            dis_set1.add(word)
            dis1 += 1
        for i in range(1, len(prediction)):
            dis_set2.add(prediction[i - 1] + ' ' + prediction[i])
            dis2 += 1
        for i in range(2, len(prediction)):
            dis_set3.add(prediction[i - 2] + ' ' + prediction[i])
            dis3 += 1
        for i in range(3, len(prediction)):
            dis_set4.add(prediction[i - 3] + ' ' + prediction[i])
            dis4 += 1
        prediction = sen2id(prediction)
        golden = sen2id(golden)

    print(
        len(dis_set1) / dis1,
        len(dis_set2) / dis2,
        len(dis_set3) / dis3,
        len(dis_set4) / dis4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args = parser.parse_args()
    with open(args.input) as f:
        lines = f.read().strip().split('\n')
        tokenized_gen = [
            line.split(': ')[1:] for line in lines if 'Generated' in line
        ]
        tokenized_tar = [
            line.split(': ')[1:] for line in lines if 'GroundTruth' in line
        ]
        # print(tokenized_tar[:5])
        # print(tokenized_gen[:5])
        # print(tokenized_gen[-5:])

    dist_cal(tokenized_gen, tokenized_tar)
