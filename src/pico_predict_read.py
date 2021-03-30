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

raw_path = "/home/qianqian/scibert/data/pico/ebmnlp/cord.txt"
predict_path = "/home/qianqian/scibert/out_gpu"
save_path = "/home/qianqian/covid-bert/token_data/"

predict_json = []
with open(predict_path, "r") as f:
    for line in f:
        predict_json.append(json.loads(line))
tags = [item for item in v['tags'] for v in predict]
words = [item for item in v['words'] for v in predict]
is_divider = False
count = 0
with open(raw_path, "r") as data_file:
    temp_tags = []
    for line in data_file:
        if line.strip() == '':
            continue
        else:
            first_token = line.strip().split()[0]
            if first_token == "-DOCSTART-":
                if temp_tags:
                    tpath = os.path.join(save_path, '{}.tag.json'.format(doc_id))
                    with open(tpath, 'w') as f:
                        f.write(json.dumps(temp_tags))
                    # Save tags to json
                temp_tags = []
                doc_id = line.strip().split()[1][1:-1]
            else:
                word = line.strip().split()[0]
                pred_word = words[count]
                assert word==pred_word
                temp_tags.append((tags[count],pred_word))
                count += 1








