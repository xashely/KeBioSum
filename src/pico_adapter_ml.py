import logging
import os
import sys
import torch
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, RobertaConfig, \
    RobertaModelWithHeads, AutoModelWithHeads, AutoModelForMaskedLM
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
from transformers import AdapterType
from datasets import ClassLabel, load_dataset, load_metric
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import glob
import transformers
import random
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("-model", default='robert', type=str)
parser.add_argument("-path", default='~/covid-bert/pico_adapter_data', type=str)
args = parser.parse_args()

from transformers import (
    AdapterConfig,
    AdapterType,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    MultiLingAdapterArguments,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

# device = torch.cuda.is_available()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
pico_adapter_data_path = args.path
label_list = ['O', "I-INT", "I-PAR", "I-OUT"]
batch_size = 24
task = 'ner'

metric = load_metric("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != 0]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != 0]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def load_dataset(corpus_type, model, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        print(pt_file)
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        # print(len(dataset))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = glob.glob(pico_adapter_data_path + '/' + corpus_type + '.padpter.pt')
    print(pts)
    # print(pico_adapter_data_path)
    # print(pico_adapter_data_path + '/' + corpus_type + '.[0-9]*.padapter.pt')
    if pts:
        src, label, mask, type_id = [], [], [], []

        dataset = _lazy_dataset_loader(pts[0], corpus_type)
        for data in dataset:
            src.append(data['src'])
            label.append(data['src_orig'])
            mask.append(data['mask'])
            if model == "bert" or model == 'pubmed':
                type_id.append(data['token_type_ids'])
        if model == "bert" or model == 'pubmed':
            return src, label, mask, type_id
        else:
            return src, label, mask


class PicoDataset(torch.utils.data.Dataset):
    def __init__(self, src_idx, labels, mask):
        self.input_ids = src_idx
        self.token_type_ids = [0] * len(self.input_ids)
        self.labels = labels
        #self.attention_mask = np.ones((len(mask), len(mask[0])))  # [1]* len(self.input_ids)

    def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx]) for key, val in self.input_ids.items()}
        item = {}
        # print(self.labels[idx],type(self.labels[idx]))
        item['labels'] = self.labels[idx]
        item['input_ids'] = self.input_ids[idx]
        item['attention_mask'] = self.attention_mask[idx]
        # print(item['input_ids'])
        # print(len(item['input_ids']))
        return item

    def __len__(self):
        return len(self.labels)


class PicoBertDataset(torch.utils.data.Dataset):
    def __init__(self, src_idx, labels, mask, type_ids):
        self.input_ids = src_idx
        self.token_type_ids = type_ids
        self.labels = labels
        # print(len(mask))
        # self.attention_mask = np.ones((len(mask), len(mask[0])))  # mask

    def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx]) for key, val in self.input_ids.items()}
        item = {}
        # print(self.labels[idx],type(self.labels[idx]))
        item['labels'] = self.labels[idx]
        item['input_ids'] = self.input_ids[idx]
        item['attention_mask'] = self.attention_mask[idx]
        item['token_type_ids'] = self.token_type_ids[idx]

        # print(item['input_ids'])
        # print(item['labels'])
        # print(item['attention_mask'])
        return item

    def __len__(self):
        return len(self.labels)


def main():
    args = parser.parse_args()
    if args.model == "robert":
        train_src, train_labels, train_mask = load_dataset('train', args.model, shuffle=True)
        val_src, val_labels, val_mask = load_dataset('valid', args.model, shuffle=False)
        #test_src, test_labels, test_mask = load_dataset('test', args.model, shuffle=False)
        print(train_src[0], train_src[1])
        print(train_labels[0], train_labels[1])
        print(train_mask[0], train_mask[1])
        train_dataset = PicoDataset(train_src, train_labels, train_mask)
        val_dataset = PicoDataset(val_src, val_labels, val_mask)
        #test_dataset = PicoDataset(test_src, test_labels, test_mask)
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        # tokenizer.save_pretrained('./save_pretrained/')
    else:
        train_src, train_labels, train_mask, train_type_id = load_dataset('train', args.model, shuffle=True)
        val_src, val_labels, val_mask, val_type_id = load_dataset('valid', args.model, shuffle=False)
        #test_src, test_labels, test_mask, test_type_id = load_dataset('test', args.model, shuffle=False)
        print(len(train_type_id))
        train_dataset = PicoBertDataset(train_src, train_labels, train_mask, train_type_id)
        val_dataset = PicoBertDataset(val_src, val_labels, val_mask, val_type_id)
        #test_dataset = PicoBertDataset(test_src, test_labels, test_mask, test_type_id)
        if args.model == 'bert':
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        # tokenizer.save_pretrained('./save_pretrained/')

    if args.model == "robert":
        model = AutoModelForMaskedLM.from_pretrained("roberta-base")
    if args.model == 'bert':
        model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
    if args.model == 'pubmed':
        model = AutoModelForMaskedLM.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
    model.resize_token_embeddings(len(tokenizer))
    model.train_adapter(task)
    model.set_active_adapters(task)
    arg = TrainingArguments(
        # f"test-{task}",
        output_dir='/data/xieqianqian/covid-bert/results/',
        evaluation_strategy="epoch",
        warmup_steps=500,
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=12,
        save_strategy="no",
        save_total_limit=1,
        load_best_model_at_end=True,
        weight_decay=0.001,
        logging_dir='/data/xieqianqian/covid-bert/logs/',
    )
    data_collator =  DataCollatorForLanguageModeling(tokenizer=tokenizer)
    # metric = load_metric("seqeval")
    trainer = Trainer(
        model=model,
        args=arg,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        do_save_full_model=False,
        do_save_adapters=False,
    )

    train_result=trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(eval_dataset)
    perplexity = math.exp(metrics["eval_loss"])
    metrics["perplexity"] = perplexity
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    if args.model == "robert":
        model.save_adapter("/data/xieqianqian/covid-bert/adapter/mlm_adapter", "mlm")
    if args.model == "bert":
        model.save_adapter("/data/xieqianqian/covid-bert/adapter/mlm_bert_adapter", "mlm")
    if args.model == "pubmed":
        model.save_adapter("/data/xieqianqian/covid-bert/adapter/mlm_pubmed_adapter", "mlm")

if __name__ == "__main__":
    main()
