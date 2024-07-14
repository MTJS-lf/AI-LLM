import os
import sys
import random
from typing import Any, Sequence, cast
import datasets
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizer,AutoTokenizer

class EmbeddingDataCollator:
    def __init__(self, tokenizer,query_max_length=64, doc_max_length=512):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
     
    def __call__(self, samples):
        data_type = samples[0]["type"]
        neg_text_input_ids = None
        neg_text_attention_mask = None
        if  data_type == "pair_score":
            texts = [ sample['query'] for sample in samples ] 
            pair_texts = [ sample['pair'] for sample in samples ]

            text_ids = self.tokenizer(texts, padding=True, max_length=self.query_max_length, truncation=True, return_tensors='pt')
            text_input_ids = text_ids["input_ids"]
            text_attention_mask = text_ids["attention_mask"]

            pair_text_ids = self.tokenizer(pair_texts, padding=True, max_length=self.doc_max_length, truncation=True, return_tensors='pt')
            pair_text_input_ids = pair_text_ids["input_ids"]
            pair_text_attention_mask = pair_text_ids["attention_mask"]
            labels = [ sample['label'] for sample in samples ]
            labels = torch.tensor(labels, dtype=torch.float32)
            return {
                'text_input_ids': cast(torch.Tensor, text_input_ids),
                'text_attention_mask': cast(torch.Tensor, text_attention_mask),
                'pair_text_input_ids': cast(torch.Tensor, pair_text_input_ids),
                'pair_text_attention_mask': cast(torch.Tensor, pair_text_attention_mask),
                'labels': labels,
                'type': data_type
            }
        elif data_type == "pair_retri_constrast":
            texts = [ sample['query'] for sample in samples ] 
            pair_texts = [ sample['pair'] for sample in samples ]
            neg_texts = []
            neg_text_index = []
            for i, sample in enumerate(samples):
                if 'neg' in sample:
                    for neg in sample["neg"]:
                        neg_texts.append(neg)
                        neg_text_index.append(i)
            text_ids = self.tokenizer(texts, padding=True, max_length=self.query_max_length, truncation=True, return_tensors='pt')
            text_input_ids = text_ids["input_ids"]
            text_attention_mask = text_ids["attention_mask"]
            pair_text_ids = self.tokenizer(pair_texts, padding=True, max_length=self.doc_max_length, truncation=True, return_tensors='pt')
            pair_text_input_ids = pair_text_ids["input_ids"]
            pair_text_attention_mask = pair_text_ids["attention_mask"]
            if len(neg_texts) > 0:
                neg_text_ids = self.tokenizer(neg_texts, padding=True, max_length=self.doc_max_length, truncation=True, return_tensors='pt')
                neg_text_input_ids = neg_text_ids["input_ids"]
                neg_text_attention_mask = neg_text_ids["attention_mask"]

            return {
                'text_input_ids': cast(torch.Tensor, text_input_ids),
                'text_attention_mask': cast(torch.Tensor, text_attention_mask),
                'pair_text_input_ids': cast(torch.Tensor, pair_text_input_ids),
                'pair_text_attention_mask': cast(torch.Tensor, pair_text_attention_mask),
                'neg_text_input_ids': cast(torch.Tensor, neg_text_input_ids),
                'neg_text_attention_mask': cast(torch.Tensor, neg_text_attention_mask),
                'neg_text_index': cast(torch.Tensor, neg_text_index),
                'type': data_type
            }
        else:
            texts = [ sample['query'] for sample in samples ] 
            pair_texts = [ sample['pair'] for sample in samples ]
            neg_texts = []
            for i, sample in enumerate(samples):
                if 'neg' in sample:
                    for neg in sample["neg"]:
                        neg_texts.append(neg)
            text_ids = self.tokenizer(texts, padding=True, max_length=self.query_max_length, truncation=True, return_tensors='pt')
            text_input_ids = text_ids["input_ids"]
            text_attention_mask = text_ids["attention_mask"]
            pair_text_ids = self.tokenizer(pair_texts, padding=True, max_length=self.doc_max_length, truncation=True, return_tensors='pt')
            pair_text_input_ids = pair_text_ids["input_ids"]
            pair_text_attention_mask = pair_text_ids["attention_mask"]
            if len(neg_texts) > 0:
                neg_text_ids = self.tokenizer(neg_texts, padding=True, max_length=self.doc_max_length, truncation=True, return_tensors='pt')
                neg_text_input_ids = neg_text_ids["input_ids"]
                neg_text_attention_mask = neg_text_ids["attention_mask"]
            return {
                'text_input_ids': cast(torch.Tensor, text_input_ids),
                'text_attention_mask': cast(torch.Tensor, text_attention_mask),
                'pair_text_input_ids': cast(torch.Tensor, pair_text_input_ids),
                'pair_text_attention_mask': cast(torch.Tensor, pair_text_attention_mask),
                'neg_text_input_ids': cast(torch.Tensor, neg_text_input_ids),
                'neg_text_attention_mask': cast(torch.Tensor, neg_text_attention_mask),
                'type': data_type,
            }
             

class EmbeddingDataSet(Dataset):
    def __init__(self, data_path,
            query_instruction=None,
            passage_instruction=None,
            train_group_size=5,
            ):
        self.dataset = datasets.load_dataset('json', data_files=data_path, split='train')
        print(self.dataset)
        self.total_len = len(self.dataset)
        self.query_instruction = query_instruction
        self.passage_instruction = passage_instruction
        self.train_group_size = train_group_size

    def __len__(self):
        return self.total_len
     
    def __getitem__(self, item):
        query = self.dataset[item]['query']
        if self.query_instruction is not None:
            query = self.query_instruction + query
        assert isinstance(self.dataset[item]['pos'], list)
        pair = random.choice(self.dataset[item]['pos'])
        dataset_type = 'pair_retri_constrast' 
        neg_passages = []
        if "neg" in self.dataset[item]:
            if len(self.dataset[item]['neg']) < self.train_group_size - 1:
                num = math.ceil((self.train_group_size, - 1) / len(self.dataset[item]['neg']))
                negs = random.sample(self.dataset[item]['neg'] * num, self.train_group_size - 1)
            else:
                negs = random.sample(self.dataset[item]['neg'], self.train_group_size - 1)
            neg_passages.extend(negs)

        if self.passage_instruction is not None:
            pair = self.passage_instruction + pair
            neg_passages = [self.passage_instruction+p for p in neg_passages]
        label = None
        if 'label' in self.dataset[item]:
            label = self.dataset[item]["label"]
            dataset_type = "pair_score"
        return {
            'query': query,
            'pair':pair,
            'neg': neg_passages,
            'label': label,
            'type': dataset_type
        }
 
if __name__ == "__main__":
   tokenizer = AutoTokenizer.from_pretrained("/mnt/data1/AIModel/chinese-roberta-wwm-ext/")   
   data_path = "/mnt/data1/Data/EncodeData/train_15neg/split/STS-B_neg.jsonl"
   dataset = EmbeddingDataSet(data_path)
   my_collate = EmbeddingDataCollator(tokenizer, query_max_length=12, doc_max_length=20)
   data_loader = DataLoader(dataset=dataset, batch_size=2, collate_fn=my_collate)
   for data in data_loader:
       print(data)
