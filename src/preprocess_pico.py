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

raw_path = "./token_data/"
save_path = "~/scibert/data/pico/ebmnlp/"
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
            f_new.write(f'-DOCSTART- ({paper_id})')
            f_new.write('\n\n')
            with open(f_main, 'r') as f:
                json_main = json.load(f)
                for sent in json_main['sentences']:
                    for token in sent['tokens']:
                        #print("word:", token['word'])
                        f_new.write(' '.join([token['word'], 'NN', 'O', 'O']))
                        f_new.write('\n')
                        if token['word'] == '.':
                            f_new.write('\n')
            f_new.write('\n\n')
            i += 1
            pbar.update()
    pbar.close()




