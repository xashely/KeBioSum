1. Examples of original files for different data sets are in ./raw directory
2. After running `preprocess.py` which converts txt files to tokenized files, example in : ./token_data
3. After running `src/preprocess_pico.py`, this converts data into format in ./pico_preprocess/cord.txt. Running scibert `predict` to PICO prediction, the output file is the same format as this but includes tags of PICO elements. 
4. Afer running `src/preprocess.py -mode format_to_line` this converts data into format in ./json_data, where 'src' is the article and 'tgt' is the abstractive summary and 'tag' is the pico predictions of the 'src' so will be the same dimension as src. 
5. Next steps are to convert to pytorch files with `src/preprocess.py -mode format_to_bert`. This generates pytorch `.pt` files containing an array of articles indictionary format for training the extractive summarisation model. Keys in the dictionaries are: 

```
{
    'src': tokenized src text (not truncated), 
    'tgt': tokenized abstractive original target (i.e. abstract of article), 
    'src_sent_labels': an array of labels to indicated whether a sentence should be included in the target extractive summary. This is generated using greedy selection., 
    'segs': attention mask?,
    'clss': starting index of token in each sentence , '
    'src_txt': raw source text, 
    'tgt_txt': raw target abstract
} 

```
