import argparse
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import ipdb
import re
import jieba

jieba.load_userdict('../../data/dict.txt')


def bleu_cal(sen1, tar1):
    bleu1 = sentence_bleu([tar1], sen1, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu([tar1], sen1, weights=(0, 1, 0, 0))
    bleu3 = sentence_bleu([tar1], sen1, weights=(0, 0, 1, 0))
    bleu4 = sentence_bleu([tar1], sen1, weights=(0, 0, 0, 1))
    return bleu1, bleu2, bleu3, bleu4


def bleu(tokenized_gen, tokenized_tar):
    bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum, count = 0, 0, 0, 0, 0
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
        bleu1, bleu2, bleu3, bleu4 = bleu_cal(sen, tar)
        bleu1_sum += bleu1
        bleu2_sum += bleu2
        bleu3_sum += bleu3
        bleu4_sum += bleu4
        count += 1

    return bleu1_sum / count, bleu2_sum / count, bleu3_sum / count, bleu4_sum / count


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
        print(tokenized_tar[:5])
        print(tokenized_gen[:5])
        # print(tokenized_gen[-5:])

    scores = bleu(tokenized_gen, tokenized_tar)
    for n, score in enumerate(scores):
        print(f'BLEU {n+1} = ({score:.4f})')
