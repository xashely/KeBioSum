# covid-bert

**This code is for the extractive summarization for the covid-19 dataset based on pretrained language model and domain knowledge including pico and terms**

Results on the CORD-19 dataset (June 27, 2020):


<table class="tg">
  <tr>
    <th class="tg-0pky">Models</th>
    <th class="tg-0pky">ROUGE-1</th>
    <th class="tg-0pky">ROUGE-2</th>
    <th class="tg-0pky">ROUGE-L</th>
  </tr>
  <tr>
    <td class="tg-0pky">covid-bert</td>
    <td class="tg-0pky">30.11</td>
    <td class="tg-0pky">9.96</td>
    <td class="tg-0pky">26.96</td>
  </tr>
</table>

**Python version**: This code is in Python3.6

**Package Requirements**: torch==1.1.0 pytorch_transformers tensorboardX multiprocess pyrouge


Some codes are borrowed from PreSumm (https://github.com/nlpyang/PreSumm)

#### Step 1 Download CORD-19 dataset
Download and unzip the `CORD-19` directories from [here](https://allenai.org/data/cord-19). Put all files in the directory `./raw_data`

####  Step 2. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
```
export CLASSPATH=/home/qianqian/stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar
```
replacing `/path/to/` with the path to where you saved the `stanford-corenlp-4.2.0` directory. 

####  Step 3. Clean the Data
```
python src/cord19_clean.py
```

####  Step 4. Sentence Splitting and Tokenization

```
python src/preprocess.py -mode tokenize -raw_path ./raw_data/ -save_path ./token_data/
```

* `RAW_PATH` is the directory containing story files, `JSON_PATH` is the target directory to save the generated json files

####  Step 5. PICO Prediction

Using scibert (https://github.com/allenai/scibert) trained on the EBM-NLP dataset (https://github.com/bepnye/EBM-NLP):

1. Preprocess the tokenized data into the pico input data on the trained scibert:
```
python src/preprocess_pico.py
```
2. Training pico extraction model

Install the allennlp: 
```
git clone https://github.com/ibeltagy/allennlp.git
```
```
git check out branch fp_16_and_others
```
```
pip install --editable .
```

Download scibert model with the link (https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar)

Export scibert in the bash script:
```
export BERT_VOCAB=/home/qianqian/scibert/model/vocab.txt
```
```
export BERT_WEIGHTS=/home/qianqian/scibert/model/weights.tar.gz
```

```
bash scripts/train_allennlp_local.sh wotune_model/
```

3. Predicting pico for cord-19
```
python -m allennlp.run predict --output-file=out.txt --include-package=scibert --predictor=sentence-tagger --use-dataset-reader --cuda-device=0 --batch-size=256 --silent ./wotune_model/model.tar.gz  ./data/pico/ebmnlp/cord.txt
```
4. Format the predicted pico to Json Files
```
python src/pico_predict_read.py
```

####  Step 6. Format to Simpler Json Files
 
```
python src/preprocess.py -mode format_to_lines -raw_path ./token_data/ -save_path ./json_data
```

* `RAW_PATH` is the directory containing tokenized files, `JSON_PATH` is the target directory to save the generated json files

####  Step 7. Format to PyTorch Files
```
python src/preprocess.py -mode format_to_bert -raw_path ./json_data/ -save_path ./bert_data/  -lower -n_cpus 1 -log_file ./logs/preprocess.log
```

* `JSON_PATH` is the directory containing json files, `BERT_DATA_PATH` is the target directory to save the generated binary files

#### step 8. Format pico json to input files
```
python src/preprocess.py -mode format_to_pico_adapter -raw_path ./json_data/ -save_path ./pico_adapter_data/
```

## Pico Adapter


## Model Training

**First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use ``-visible_gpus -1``, after downloading, you could kill the process and rerun the code with multi-GPUs.**

```
python src/train.py -task ext -mode train -bert_data_path ./bert_data/ -ext_dropout 0.1 -model_path ./models/ -lr 2e-3 -visible_gpus 0 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -train_steps 50000 -accum_count 2 -log_file ./logs/ext_bert_covid -use_interval true -warmup_steps 10000 -max_pos 512
```


## Model Evaluation
```
python src/train.py -task ext -mode validate -batch_size 3000 -test_batch_size 500 -bert_data_path ./bert_data/ -log_file ./logs/val_ext_bert_covid -model_path ./models/ -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -result_path ./results/ext_bert_covid -test_all True
```
```
python src/train.py -task ext -mode test -batch_size 3000 -test_batch_size 500 -bert_data_path ./bert_data/ -log_file ./logs/test_ext_bert_covid -test_from ./models/model_step_9000.pt -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -result_path ./results/ext_bert_covid 
```
* `-mode` can be {`validate, test`}, where `validate` will inspect the model directory and evaluate the model for each saved checkpoint, `test` need to be used with `-test_from`, indicating the checkpoint you want to use (choose the top checkpoint on the validation dataset)
* `MODEL_PATH` is the directory of saved checkpoints
* use `-mode valiadte` with `-test_all`, the system will load all saved checkpoints and select the top ones to generate summaries
