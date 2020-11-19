import argparse
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import ipdb
import re
import jieba

# jieba.load_userdict('/home/zhouyuanhang/project/CRS/ConvModel/GPT2_2v2/data/dict.txt')
# print("Use dictionary just generate")

def bleu_cal(sen1, tar1):
    bleu1 = sentence_bleu([tar1], sen1, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu([tar1], sen1, weights=(0, 1, 0, 0))
    bleu3 = sentence_bleu([tar1], sen1, weights=(0, 0, 1, 0))
    bleu4 = sentence_bleu([tar1], sen1, weights=(0, 0, 0, 1))
    return bleu1, bleu2, bleu3, bleu4

def bleu(tokenized_gen, tokenized_tar):
    bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum, count = 0, 0, 0, 0, 0
    for sen, tar in zip(tokenized_gen, tokenized_tar):        
        # sen: word list
        # sen = [word for word in sen]
        # tar = [word for word in tar]
        # print(sen)
        # print(tar)
        # break
        bleu1, bleu2, bleu3, bleu4=bleu_cal(sen, tar)
        bleu1_sum+=bleu1
        bleu2_sum+=bleu2
        bleu3_sum+=bleu3
        bleu4_sum+=bleu4
        count+=1
    
    return bleu1_sum/count, bleu2_sum/count, bleu3_sum/count, bleu4_sum/count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args = parser.parse_args()
    with open(args.input) as f:
        lines = f.read().strip().split('\n')
        # element is string: ? ? ? ?? ?? <movie> ? ?? ? ?? ~ ~ ???? ? ?? ?
        # sen: string: ? ? ? ?? ?? <movie> ? ?? ? ?? ~ ~ ???? ? ?? ?
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

    scores = bleu(tokenized_gen, tokenized_tar)
    for n, score in enumerate(scores):
        print(f'BLEU {n+1} = ({score:.4f})')