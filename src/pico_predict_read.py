import json
import os
import random
import re
import math
from collections import Counter
from os.path import join as pjoin

from others.logging import logger
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-raw_path", default="/home/qianqian/scibert/data/pico/ebmnlp/cord.txt", type=str)
parser.add_argument("-save_path", default="/home/qianqian/covid-bert/token_data/", type=str)
parser.add_argument("-predict_path", default="/home/qianqian/scibert/out.txt", type=str)
parser.add_argument("-corpus", default="cord-19", type=str)

args = parser.parse_args()
raw_path = os.path.abspath(args.raw_path)
save_path = os.path.abspath(args.save_path)
predict_path = os.path.abspath(args.predict_path)

# make directories for saving data if they don't already exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

if args.corpus =='pubmed':
    test_path = os.path.join(raw_path, 'test_pubmed')
    val_path = os.path.join(raw_path, 'val_pubmed')
    train_path = os.path.join(raw_path, 'train_pubmed')
    test_corpora = sorted([os.path.join(raw_path, f) for f in os.listdir(test_path)
                           if not f.startswith('.') and not f.endswith('.abs.txt.json')])
    val_corpora = sorted([os.path.join(raw_path, f) for f in os.listdir(val_path)
                          if not f.startswith('.') and not f.endswith('.abs.txt.json')])
    train_corpora = sorted([os.path.join(raw_path, f) for f in os.listdir(train_path)
                            if not f.startswith('.') and not f.endswith('.abs.txt.json')])
    test_len = len(test_corpora)
    val_len = len(val_corpora)
    train_len = len(train_corpora)

predict_json = []
with open(predict_path, "r") as f:
    for line in f:
        content = json.loads(line)
        predict_json.append(content)
        tag_leng = len(content['tags'])
        words_leng = len(content['words'])
        try:
            assert tag_leng == words_leng, (content['tags'], content['words'])
        except Exception:
            print (words_leng - tag_leng)
            print ([key for key, val in dict(Counter(content['words'])).items() if val == (words_leng - tag_leng)])
tags = [item for v in predict_json for item in v['tags']]
words = [item for v in predict_json for item in v['words']]
#is_divider = False
print (len(tags), len(words))
count = 0
doc_id = 0
with open(raw_path, "r") as data_file:
    temp_tags = []
    for line in tqdm(data_file):
        if line.strip() == '':
            continue
        else:
            first_token = line.strip().split()[0]
            if first_token == "-DOCSTART-":
                if temp_tags:
                    if args.corpus == 'pubmed':
                        if doc_id <= train_len-1:
                            save_path=os.path.join(save_path,'train_pubmed')
                        elif doc_id <=train_len+val_len-1:
                            save_path = os.path.join(save_path, 'val_pubmed')
                            doc_id = doc_id - train_len+1
                        elif doc_id <= train_len + val_len+test_len - 1:
                            save_path = os.path.join(save_path, 'test_pubmed')
                            doc_id = doc_id - train_len - test_len + 1
                    tpath = os.path.join(save_path, '{}.tag.json'.format(doc_id))
                    with open(tpath, 'w') as f:
                        f.write(json.dumps(temp_tags))
                    # Save tags to json
                temp_tags = []
                doc_id = line.strip().split()[1][1:-1]
            else:
                fields = line.strip().split()
                if len(fields) == 4:
                    word = fields[0]
                else:
                    word = " ".join([w for w in fields[:-3]])
                pred_word = words[count]
                #print(word, pred_word)
                try:
                    assert word==pred_word
                except AssertionError as e:
                    print (word, pred_word)
                temp_tags.append((tags[count],pred_word))
                count += 1
    if args.corpus == 'pubmed':
        save_path = os.path.join(save_path, 'test_pubmed')
        doc_id = doc_id - train_len - test_len + 1
    tpath = os.path.join(save_path, '{}.tag.json'.format(doc_id))
    with open(tpath, 'w') as f:
        f.write(json.dumps(temp_tags))








