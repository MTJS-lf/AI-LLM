# -*- coding: utf-8 -*-

import os
import sys
import json

from typing import Dict, Optional

import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer


class SupervisedChatDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length=128
    ):
        super(SupervisedChatDataset, self).__init__()
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = [self.tokenizer.get_command(f"<|user|>")] 
        self.assistant_tokens = [self.tokenizer.get_command(f"<|assistant|>")] 
        self.ignore_index = -100
        item = self.preprocessing(self.data[0])
        print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        for id_ in item["labels"]:
            if id_ == -100:
                continue

            labels.append(id_)
        print("label:", self.tokenizer.decode(labels))

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        input_ids = [self.tokenizer.get_command('[gMASK]'), self.tokenizer.get_command('sop')]
        label_ids = [self.ignore_index, self.ignore_index]

        for message in example["conversations"]:
            role = message["role"]
            content = message["content"]
            content_ids = self.tokenizer.encode(content, add_special_tokens=False)
            if role == "user" or role == "system":
                input_ids += self.user_tokens + content_ids
                label_ids += [self.ignore_index] * len(self.user_tokens) + [
                    self.ignore_index
                ] * len(content_ids)
            else:
                input_ids += self.assistant_tokens + content_ids
                label_ids += (
                    [self.ignore_index] * len(self.assistant_tokens)
                    + content_ids
                )
        input_ids.append(self.tokenizer.eos_token_id)
        label_ids.append(self.tokenizer.eos_token_id)
        # truncate to max len
        input_ids = input_ids[: self.model_max_length]
        label_ids = label_ids[: self.model_max_length]
        attention_mask = [1] * len(input_ids)
        # pad to max len
        input_ids += [self.tokenizer.eos_token_id] * (
            self.model_max_length - len(input_ids)
        )
        label_ids += [self.ignore_index] * (self.model_max_length - len(label_ids))
        attention_mask += [0] * (self.model_max_length - len(attention_mask))
        # convert to pt tensor
        input_ids = torch.LongTensor(input_ids)
        label_ids = torch.LongTensor(label_ids)
        attention_mask = torch.LongTensor(attention_mask)
        return {
            "input_ids": input_ids,
            "labels": label_ids,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])



if __name__ == "__main__":
   tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
   print(tokenizer)
   data_path = sys.argv[2]
   #print(data_path)
   chat_dataset = SupervisedChatDataset(data_path, tokenizer,) 
   print("load data")

   

