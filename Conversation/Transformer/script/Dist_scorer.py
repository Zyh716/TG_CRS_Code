import argparse
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import json
import ipdb
import re
import jieba
word2index=json.load(open("../../data/tf_bpe2index.json",encoding='utf-8'))

def back_process(sen):
    sentences=''
    new_sen=[]
    for word in sen:
        if word=='ï¼Œ':
            s1=' '.join(new_sen)
            new_sen=[]
            if s1 in sentences:
                break
            else:
                if sentences!='':
                    sentences+=' ï¼Œ '+s1
                else:
                    sentences+=s1
        else:
            new_sen.append(word)

    sentences2nd=sentences.split()
    try:
        final_sentence=[sentences2nd[0]]
    except:
        return []
    for i in range(1,len(sentences2nd)):
        if sentences2nd[i]==sentences2nd[i-1]:
            break
        else:
            final_sentence.append(sentences2nd[i])
    return final_sentence

def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1) * np.dot(v2, v2)) + 1e-4)

def sen2id(sen):
    candidates=[]
    for word in sen:
        #if word not in stopwords and word in word2index:
        if word in word2index:
            candidates.append(word2index[word])
    #return list(set(candidates))
    return candidates

def cal_calculate(tokenized_gen, tokenized_tar):
    # data=json.load(open(name,encoding='utf-8'))
    scores=[]
    dis1=0
    dis_set1=set()
    dis2=0
    dis_set2=set()
    output=[]
    for sen, tar in zip(tokenized_gen, tokenized_tar):
        # context=line['sample'][:-1]
        # prediction=line['pre_answer'][0]
        # golden=line['sample'][-1]
        # output.append('context: '+'\t'.join(context)+'\n')
        # prediction=back_process(prediction)
        # output.append('result: '+' '.join(prediction)+'\n')
        golden = tar
        prediction = sen
        # print(golden)
        # print(prediction)
        # ipdb.set_trace()
        ### 已经获得了prediction/golden， string
        for word in prediction:
            dis_set1.add(word)
            dis1+=1
        for i in range(1,len(prediction)):
            dis_set2.add(prediction[i-1]+' '+prediction[i])
            dis2+=1
        prediction=sen2id(prediction)
        golden=sen2id(golden)
    print(len(dis_set1)/dis1, len(dis_set2)/dis2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args = parser.parse_args()
    with open(args.input) as f:
        lines = f.read().strip().split('\n')
        # tokenized_gen = [line.split(': ')[1:] for line in lines if 'Generated' in line]
        # tokenized_tar = [line.split(': ')[1:] for line in lines if 'Recommender' in line]
        tokenized_gen = []
        tokenized_tar = []
        for line in lines:
            if 'Generated' in line:
                sen = line.split('] ')[1]
                new_sen = []
                for word in sen.split():
                    if word == '_split_':
                        break
                    if word != '<movie>':
                        new_sen.append(word)
                    else:
                        new_sen.append(word)
                tokenized_gen.append(new_sen)
            if 'GroundTruth' in line:
                sen = line.split('] ')[1]
                new_sen = []
                for word in sen.split():
                    if word == '_split_':
                        break
                    if word != '<movie>':
                        new_sen.append(word)
                    else:
                        new_sen.append(word)
                tokenized_tar.append(new_sen)
        print(tokenized_tar[:5])
        print(tokenized_gen[:5])
        # print(tokenized_gen[-5:])

    cal_calculate(tokenized_gen, tokenized_tar)
