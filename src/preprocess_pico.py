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
save_path = "./scibert/data/pico/ebmnlp/"
corpora = sorted([os.path.join(raw_path, f) for f in os.listdir(raw_path)
                      if not f.startswith('.')])
print('... Packing tokenized data into pico txt...')
print('Converting files count: {}'.format(len(corpora)))
with tqdm(total=len(corpora)) as pbar:
    with open(os.path.join(save_path, 'cord.txt'), 'w') as f_new:
        for f_main in corpora:
            paper_id = os.path.basename(f_main).split('.')[0]
            f_new.write(f'-DOCSTART- ({paper_id})')
            f_new.write('\n\n')
            with open(f_main, 'r') as f:
                json_main = json.load(f)
                for sent in json_main['sentences']:
                    for token in sent['tokens']:
                        for word in token:
                            f_new.write(' '.join([token, 'NN', 'O', 'O']))
                            f_new.write('\n')
                            if word == '.':
                                f_new.write('\n')
            f_new.write('\n\n')
    pbar.close()




