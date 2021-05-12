import json
import os
import random
import re
import math
from collections import Counter
from os.path import join as pjoin
import argparse

from others.logging import logger
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-raw_path", default="./token_data/", type=str)
parser.add_argument("-save_path", default="/home/qianqian/scibert/data/pico/ebmnlp/", type=str)

corpora = sorted([os.path.join(raw_path, f) for f in os.listdir(raw_path)
                      if not f.startswith('.') and not f.endswith('.abs.txt.json')])
print('... Packing tokenized data into pico txt...')
print('Converting files count: {}'.format(len(corpora)))
with tqdm(total=len(corpora)) as pbar:
    with open(os.path.join(save_path, 'cord.txt'), 'w') as f_new:
        i=0
        for f_main in corpora:
            paper_id = os.path.basename(f_main).split('.')[0]
            if i==0:
                print(f_main, paper_id)
            count = 0
            f_new.write(f'-DOCSTART- ({paper_id})')
            f_new.write('\n\n')
            with open(f_main, 'r') as f:
                json_main = json.load(f)
                for sent in json_main['sentences']:
                    j = 0
                    newline = False
                    for token in sent['tokens']:
                        newline = False
                        #print("word:", token['word'])
                        f_new.write(' '.join([token['word'], 'NN', 'O', 'O']))
                        f_new.write('\n')
                        j += 1
                        overlong_sequence = j >= 250
                        end_token = token['word'] == '.'
                        newline = overlong_sequence or end_token

                        if overlong_sequence:
                            count += 1
                            print("too long sentences:")
                        if newline:
                            f_new.write('\n')
                            j = 0
                    if not newline:
                        f_new.write('\n')
              
            f_new.write('\n\n')
            if count != 0:
                print("id, count:", i, count)
            i += 1
            #if i == 60:
            #    break
            pbar.update()
    pbar.close()




