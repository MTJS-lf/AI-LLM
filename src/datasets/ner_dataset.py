import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple
import json

import datasets
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer,AutoTokenizer

max_int = torch.iinfo(torch.int64).max

class TrainDatasetForNer(Dataset):
    def __init__(
            self,
            data_path,
            tokenizer: PreTrainedTokenizer,
            max_seq_len=128,
            data_type="train"
    ):

        self.dataset = []
        data_file_path = os.path.join(data_path, data_type+".json")
        with open(data_file_path, 'r') as fp:
            for line in fp:
                item = json.loads(line)
                self.dataset.append(item)
        label_file_path = os.path.join(data_path, "ent2id.json")
        self.label2ids = {}
        with open(label_file_path, 'r') as fp:
            self.label2ids = json.load(fp)
        self.id2label = {} 
        for k, v in self.label2ids.items():
            self.id2label[v] = k
        self.tokenizer = tokenizer
        self.total_len = len(self.dataset)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.total_len

    def get_id2labels(self):
        return self.id2label         

    def __getitem__(self, item):
        assert isinstance(self.dataset[item]['text'], str)
        assert isinstance(self.dataset[item]['labels'], list)
        text = self.dataset[item]['text']
        tokens = [i for i in text]
        features = self.tokenizer.encode_plus(text=tokens,
                                    max_length=self.max_seq_len,
                                    padding="max_length",
                                    truncation='longest_first')
        label_ids = [0] * len(tokens)
        count = (features["attention_mask"]).count(1)
        entities = self.dataset[item]['labels']
        for entity in entities:
            ent_type = entity[1]
            ent_start = entity[2]
            ent_end = ent_start + len(entity[4]) - 1
            if ent_start == ent_end:
                label_ids[ent_start] = self.label2ids['B-' + ent_type]
            else:
                label_ids[ent_start] = self.label2ids['B-' + ent_type]
                label_ids[ent_end] = self.label2ids['I-' + ent_type]
                for i in range(ent_start + 1, ent_end):
                    label_ids[i] = self.label2ids['I-' + ent_type]
        if len(label_ids) > self.max_seq_len - 2:
            label_ids = label_ids[:self.max_seq_len - 2]
        label_ids = [0] + label_ids + [0]
        if len(label_ids) < self.max_seq_len:
            pad_length = self.max_seq_len - len(label_ids)
            label_ids = label_ids + [0] * pad_length
        features["labels"] = label_ids
        return features

if __name__ == "__main__":
   tokenizer = AutoTokenizer.from_pretrained("/data/BaseModel/chinese-roberta-wwm-ext/")   
   data_path = "/data/ner_data/"
   dataset = TrainDatasetForNer(data_path, tokenizer, data_type="train")
   for data in dataset:
       print(data)
