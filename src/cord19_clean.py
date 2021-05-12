import argparse
import json
import os
import re
from tqdm import tqdm

import csv
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument('-R', '--root', dest='root_dir', default='./raw_data/',
                    help='Directory to CORD-19 downloaded')


def clean_json(json_dict):
    #
    # how about bib? they also indicate what the paper is about in general
    #
    title = json_dict['metadata']['title']
    body = json_dict['body_text']
    text = []

    for p in body:
        if p['section'] == 'Pre-publication history':
            continue
        p_text = p['text'].strip()
        p_text = re.sub('\[[\d\s,]+?\]', '', p_text) # matches references e.g. [12]
        p_text = re.sub('\(Table \d+?\)', '', p_text) # matches table references e.g. (Table 1)
        p_text = re.sub('\(Fig. \d+?\)', '', p_text) # matches fig references e.g. (Fig. 1)
        text.append(p_text)

    return {'title': title, 'text': text}


if __name__ == '__main__':
    args = parser.parse_args()

    # dirname = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(args.root_dir)
    meta_path = os.path.join(root_dir, 'metadata.csv')
    new_meta_path = os.path.join(root_dir, 'PMC.csv')
    pmc_path = os.path.join(root_dir, 'document_parses', 'pmc_json')
    post_path = os.path.join(root_dir, 'document_parses', 'post_json')

    with open(meta_path, 'r') as f:
        df = pd.read_csv(meta_path, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
    
    print('Length of csv before removing papers without abstract: {}'.format(df.shape[0]))
    # skip papers without abstract
    df = df[~pd.isnull(df.abstract)]
    print('Length of csv after removing papers without abstract: {}'.format(df.shape[0]))
    # drop duplicates
    df['title_lower'] = df.title.str.lower()
    df= df.drop_duplicates(subset='title_lower').drop(columns='title_lower')
    print('Length of csv once articles deduplicated: \t{}'.format(df.shape[0]))


    # pandas with tqdm requires manual update for now
    # df_len = df.shape[0]
    # df_len = 10
    no_path_counter = 0
    pmc_files = 0

    if not os.path.isdir(post_path):
        raise ValueError('{} is not a directory'.format(post_path))

    write_head = False
    with open(new_meta_path, 'w') as f:
        w = csv.writer(f)

        print('Cleaning & saving off only pubmed files to {}...'.format(post_path))
        for i, row in tqdm(df.iterrows(),total=df.shape[0]):

            # JB: is there a reason we only want pubmed articles rather than other articles?
            fpath = os.path.join(pmc_path, '{}.xml.json'.format(row['pmcid'])) 
            if not os.path.isfile(fpath):
                no_path_counter +=1
                continue

            # before the script was only reading and wasn't writing out files (only 'r' param)
            with open(fpath, 'r') as fi: 
                json_dict = json.load(fi)

            # clean data
            cleaned_dict = clean_json(json_dict)

            # include abstract from paper for gold summary
            cleaned_dict['abstract'] = row['abstract']

            # writing out cleaned version
            outpath = os.path.join(post_path, '{}.xml.json'.format(row['pmcid'])) 
            with open(outpath,'w') as fi:
                json.dump(cleaned_dict,fi)

            if not write_head:
                w.writerow(cleaned_dict.keys())
                write_head = True
            
            w.writerow(cleaned_dict.values())
            pmc_files+=1
                   
    print('After preprocessing - total with no path: \t{}'.format(no_path_counter)) # 70963
    print('After preprocessing - total saved: \t{}'.format(pmc_files)) # 42021

    # Check length of dataframe written out is same as number of paths written out
    new_metadata_df = pd.read_csv(new_meta_path)
    if not len(new_metadata_df)==pmc_files:
        print('Length of csv ({}) different to number of files written out ({})'.format(len(new_metadata_df),pmc_files))

