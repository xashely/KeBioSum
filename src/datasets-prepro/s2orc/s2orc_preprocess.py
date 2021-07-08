import json
import pandas as pd
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-raw_path_metadata", default="/Users/User/Documents/repos/covid-bert/sample_data/s2orc/sample_metadata.jsonl", type=str)

args = parser.parse_args()
raw_path = os.path.abspath(args.raw_path_metadata)

df = pd.read_json(path_or_buf=raw_path, lines=True)
df = df.dropna(subset=['has_pdf_parse','mag_field_of_study'])
df = df[df['mag_field_of_study'].apply(lambda x: True if 'Medicine' in x else False)]
df = df[df['has_pdf_parse']==True]
df = df[df['has_pdf_parsed_abstract']==1]
df = df[df['has_pdf_parsed_body_text']==1]
df = df[df['has_pdf_parsed_ref_entries']==1]
print(f"Total amount of data: {len(df)}")  

print(df.iloc[0])