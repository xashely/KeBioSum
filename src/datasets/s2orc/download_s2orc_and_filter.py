"""


Example of how one would download & process a single batch of S2ORC to filter to specific field of study.
Can be useful for those who can't store the full dataset onto disk easily.
Please adapt this to your own field of study.


Creates directory structure:

|-- metadata/
    |-- raw/
        |-- metadata_0.jsonl.gz      << input; deleted after processed
    |-- medicine/
        |-- metadata_0.jsonl         << output
|-- pdf_parses/
    |-- raw/
        |-- pdf_parses_0.jsonl.gz    << input; deleted after processed
    |-- medicine/
        |-- pdf_parses_0.jsonl       << output

"""


import os
import subprocess
import gzip
import io
import json
from tqdm import tqdm
import pandas as pd
import argparse


# process single batch
def process_batch(batch: dict):
    # this downloads both the metadata & full text files for a particular shard
    cmd = ["wget", "-O", batch['input_metadata_path'], batch['input_metadata_url']]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    cmd = ["wget", "-O", batch['input_pdf_parses_path'], batch['input_pdf_parses_url']]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    # first, let's filter metadata JSONL to only papers with a particular field of study.
    # we also want to remember which paper IDs to keep, so that we can get their full text later.
    paper_ids_to_keep = dict()
    with gzip.open(batch['input_metadata_path'], 'rb') as gz, open(batch['output_metadata_path'], 'wb') as f_out:
        f = io.BufferedReader(gz)
        all_papers_count = 0
        keep_papers_count = 0
        for line in tqdm(f.readlines()):
            all_papers_count +=1
            metadata_dict = json.loads(line)
            paper_id = metadata_dict['paper_id']
            mag_field_of_study = metadata_dict['mag_field_of_study']
            pdf_parse = metadata_dict['has_pdf_parse']
            abstract = metadata_dict['abstract']
            pid = metadata_dict['pmc_id']
            try:
                body_text = metadata_dict['has_pdf_parsed_body_text']
                ref_text = metadata_dict['has_pdf_parsed_ref_entries']
            except: 
                continue

            if ((mag_field_of_study and 'Medicine' in mag_field_of_study) and (pdf_parse) and (abstract) and (body_text) and (ref_text) and (pid)):  
                paper_ids_to_keep.update({paper_id:pid})
                f_out.write(line)
                keep_papers_count +=1

    # now, we get those papers' full text
    with gzip.open(batch['input_pdf_parses_path'], 'rb') as gz:
        f = io.BufferedReader(gz)
        for line in tqdm(f.readlines()):
            metadata_dict = json.loads(line)
            paper_id = metadata_dict['paper_id']
            if paper_id in paper_ids_to_keep.keys():
                pmc_id = paper_ids_to_keep[paper_id]
                with open(os.path.join(batch['output_pdf_parses_path'],f"{pmc_id}.xml.json"), 'wb') as f_out:
                    f_out.write(line)

    # write results into csv format
    metadata_csv_path = f"{batch['output_metadata_path'].split('_')[0]}.csv"
    df_to_append = pd.read_json(batch['output_metadata_path'],lines=True)
    df_to_append = df_to_append.rename(columns={'pmc_id':'pmcid'}) # rename to be consistent with cord dataset
    if os.path.exists(metadata_csv_path):
        df = pd.read_csv(metadata_csv_path)
        df = df.append(df_to_append).reset_index(drop=True)
    else:
        df = df_to_append
    df.to_csv(metadata_csv_path,index=False)

    # now delete the raw files to clear up space for other shards
    os.remove(batch['input_metadata_path'])
    os.remove(batch['input_pdf_parses_path'])
    print(f"Kept {keep_papers_count} papers out of total of {all_papers_count}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-links_path", default="./src/other_datasets/s2orc/download_links.txt", type=str)
    parser.add_argument("-save_path", default="../data/s2orc/raw_data", type=str)

    args = parser.parse_args()
    links_path = os.path.abspath(args.links_path)
    save_path = os.path.abspath(args.save_path)

    METADATA_INPUT_DIR = os.path.join(save_path,'metadata/raw/')
    METADATA_OUTPUT_DIR = save_path
    PDF_PARSES_INPUT_DIR = os.path.join(save_path,'document_parses/raw/')
    PDF_PARSES_OUTPUT_DIR = os.path.join(save_path,'document_parses/pmc_json/')

    os.makedirs(METADATA_INPUT_DIR, exist_ok=True)
    os.makedirs(METADATA_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PDF_PARSES_INPUT_DIR, exist_ok=True)
    os.makedirs(PDF_PARSES_OUTPUT_DIR, exist_ok=True)

    with open(links_path,'r') as f:
        download_linkss = json.load(f)
    
    print(f"Number of tar.gz files to download and process: {len(download_linkss)}")

    # turn these into batches of work
    # TODO: feel free to come up with your own naming convention for 'input_{metadata|pdf_parses}_path'
    batches = [{
        'input_metadata_url': download_links['metadata'],
        'input_metadata_path': os.path.join(METADATA_INPUT_DIR,
                                            os.path.basename(download_links['metadata'].split('?')[0])),
        'output_metadata_path': os.path.join(METADATA_OUTPUT_DIR,
                                             os.path.basename(download_links['metadata'].split('.gz?')[0])),
        'input_pdf_parses_url': download_links['pdf_parses'],
        'input_pdf_parses_path': os.path.join(PDF_PARSES_INPUT_DIR,
                                              os.path.basename(download_links['pdf_parses'].split('?')[0])),
        'output_pdf_parses_path': PDF_PARSES_OUTPUT_DIR,
    } for download_links in download_linkss]

    batches = batches[0:2]
    
    for idx, batch in enumerate(batches):
        print(f"processing batch no: {idx}")
        process_batch(batch=batch)
    
    print("saving off metadata csv")
    metadata_csv_path = f"{batch['output_metadata_path'].split('_')[0]}.csv"
    df = pd.read_csv(metadata_csv_path)
    csv_save_path = os.path.join(save_path,'metadata.csv')
    df.to_csv(csv_save_path,index=False)
    os.remove(metadata_csv_path)

    
