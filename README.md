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

### Step 1. Download datasets 
#### CORD-19 dataset
Download and unzip the `CORD-19` directories from [here](https://allenai.org/data/cord-19). Put all files in the directory `./raw_data`
#### PubMed dataset
Download zip file from [here] (https://drive.google.com/file/d/1lvsqvsFi3W-pE1SqNZI0s8NR9rC1tsja/view). Put all files in directory `./raw`
####  S2ORC dataset
Details of the dataset can be found [here] (https://github.com/allenai/s2orc). To prepare, follow instructions [here] (src/datasets/s2orc/README.md))

###  Step 2. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile (`/.bashrc` file):
```
 for file in `find /home/qianqian/stanford-corenlp-4.2.1  -name "*.jar"`; do export CLASSPATH="$CLASSPATH:`realpath $file`"; done
```
replacing `/path/to/` with the path to where you saved the `stanford-corenlp-4.2.0` directory. 

###  Step 3. Cleaning data and and Tokenization

For CORD19 or S2ORC data (both from allenai), use the following command to preprocess the data, the raw data files should be in a folder ./raw_data/pmc_data/document_parses/pmc_json/ with the associated metadata csv file at ./raw_data/pmc_data/document_parses/metadata.csv

```
python src/preprocess.py -mode tokenize_allenai_datasets -raw_path ./raw_data/ -save_path ./token_data/ -log ./tokenize_allenai.log
```

For the PubMed dataset. 
```
python src/preprocess.py -mode tokenize_pubmed_dataset -raw_path ./raw/ -save_path ./token_data/ -log ./tokenize_pubmed.log
```

* `RAW_PATH` is the directory containing story files, `JSON_PATH` is the target directory to save the generated json files

###  Step 4. PICO Prediction

Using scibert (https://github.com/allenai/scibert) trained on the EBM-NLP dataset (https://github.com/bepnye/EBM-NLP):

1. Preprocess the tokenized data into the pico input data on the trained scibert:
```
python src/preprocess_pico.py -raw_path .=/token_data/ -save_path ..output_data/pico_preprocess/
```
2. Training pico extraction model

Install the allennlp: 
```
git clone https://github.com/ibeltagy/allennlp.git
```
```
git checkout fp16_and_others
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
export CUDA_VISIBLE_DEVICES=0 
export PICO_MODE='PREDICT'
cd ./scibert
python -m allennlp.run predict --output-file=/mnt/disk/jenny/pubmed-dataset/pico_preprocess/test/output.txt --include-package=scibert --predictor=sentence-tagger --use-dataset-reader --cuda-device=0 --batch-size=32 --silent /home/qianqian/scratch/scibert/wotune_model/model.tar.gz  /mnt/disk/jenny/pubmed-dataset/pico_preprocess/test/cord.txt
```

4. Format the predicted pico to Json Files
```
python src/pico_predict_read.py -raw_path out.txt -save_path .=/token_data/ -predict_path  ./data/pico/ebmnlp/cord.txt
```

###  Step 5. Format to Simpler Json Files
 
```
python src/preprocess.py -mode format_to_lines -raw_path ./token_data/ -save_path ./json_data -log ./tokenize.log
```

* `RAW_PATH` is the directory containing tokenized files, `JSON_PATH` is the target directory to save the generated json files

###  Step 6. Format to PyTorch Files
```
python src/preprocess.py -mode format_to_bert -raw_path ./json_data/ -save_path ./bert_data/  -lower -n_cpus 1 -log_file ./logs/preprocess.log 
```

* `JSON_PATH` is the directory containing json files, `BERT_DATA_PATH` is the target directory to save the generated binary files

### Step 7. Format pico json to input files
```
python src/preprocess.py -mode format_to_pico_adapter -raw_path ./json_data/ -save_path ./pico_adapter_data/ -log_file ./pico_adapter_robert.log
```
Note depending on model type you want to use, you can change `format_to_bert` to `format_to_pubmed_bert` or `format_to_robert`

### Step 8. Pico Adapter - train PICO adapter model which will be included as an adapter in model training in the next step
```
CUDA_VISIBLE_DEVICES=0 python src/pico_adapter.py -model robert -path /data/xieqianqian/covid-bert/data/pico_roberta_data -output ./pico_adapter_output 
```
* -model can be [bert, robert, pubmed]

### Step 9. Model Training

**First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use ``-visible_gpus -1``, after downloading, you could kill the process and rerun the code with multi-GPUs.**

```
python src/train.py -task ext -mode train -bert_data_path /data/xieqianqian/covid-bert/data/pubmed_data/ -ext_dropout 0.4 -model_path /data/xieqianqian/covid-bert/models_2/ -lr 2e-3 -visible_gpus 2 -report_every 50 -save_checkpoint_steps 1000 -batch_size 12000 -train_steps 20000 -accum_count 2 -log_file /data/xieqianqian/covid-bert/logs/ext_bert_covid -use_interval true -warmup_steps 5000 -model pubmed -adapter_training_strategy discriminative -adapter_path_pubmed_discriminative /home/jenny/data/covid/pico_adapter_model_outputs_pubmed/adapter/final_pubmed_adapter
```



### Step 10. Model Evaluation
```
python src/train.py -task ext -mode validate -batch_size 12000 -test_batch_size 12000 -bert_data_path ./bert_data/ -log_file ./logs/val_ext_bert_covid -model_path ./models/ -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -result_path ./results/ext_bert_covid -test_all True -model bert
```
```
python src/train.py -task ext -mode test -batch_size 3000 -test_batch_size 500 -bert_data_path ./bert_data/ -log_file ./logs/test_ext_bert_covid -test_from ./models/model_step_9000.pt -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -result_path ./results/ext_bert_covid -model bert
```
* `-mode` can be {`validate, test`}, where `validate` will inspect the model directory and evaluate the model for each saved checkpoint, `test` need to be used with `-test_from`, indicating the checkpoint you want to use (choose the top checkpoint on the validation dataset)
* `MODEL_PATH` is the directory of saved checkpoints
* use `-mode valiadte` with `-test_all`, the system will load all saved checkpoints and select the top ones to generate summaries

