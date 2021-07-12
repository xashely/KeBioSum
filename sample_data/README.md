1. Examples of original files for different data sets are in ./raw directory
2. After running `preprocess.py` which converts txt files to tokenized files, example in : ./token_data
3. After running `src/preprocess_pico.py`, this converts data into format in ./pico_preprocess/cord.txt. Running scibert `predict` to PICO prediction, the output file is the same format as this but includes tags of PICO elements. 
4. Afer running `src/preprocess.py -mode format_to_line` this converts data into format in ./json_data
5. Next steps are to convert to pytorch files 
