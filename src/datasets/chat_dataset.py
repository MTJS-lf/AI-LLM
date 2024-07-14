# -*- coding: utf-8 -*-

import os
import sys
import json

from typing import Dict, Optional

import torch
import datasets
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


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

class GLM3ChatDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128, data_type="train"):
        super(GLM3ChatDataset, self).__init__()
        self.dataset = datasets.load_dataset('json', data_dir=data_path)
        self.data = [example["conversations"] for example in self.dataset[data_type]]
        self.tokenizer = tokenizer
        self.model_max_length = int(max_len)
        self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.preprocess(self.data[i])

    def preprocess(
        self,
        source,
        system_message: str = "You are a helpful assistant."
    ) -> Dict:
        roles = {"user": "<|user|>", "assistant": "<|assistant|>", "system": "<|system|>"}
        gmask_ids = self.tokenizer.get_command('[gMASK]') 
        sop_ids = self.tokenizer.get_command('sop')
    
        # Apply prompt templates
        input_ids, loss_masks = [gmask_ids,sop_ids], [False, False]
        for j, sentence in enumerate(source):
            role = sentence["from"]
            loss_mask_val = True
            if role in ("system", "user"):
                loss_mask_val = False
            new_input_ids = self.tokenizer.build_single_message(sentence["from"],'',  sentence["content"])
            new_loss_masks = [loss_mask_val] * len(new_input_ids)
            input_ids += new_input_ids
            loss_masks += new_loss_masks
        input_ids.append(self.tokenizer.eos_token_id)
        loss_masks = [False, *loss_masks]
        assert len(input_ids) == len(loss_masks)
        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100)
        attention_mask = [1] * len(input_ids)
        # pad to max len
        if len(input_ids) < self.model_max_length:
            input_ids += [self.tokenizer.eos_token_id] * (
                self.model_max_length - len(input_ids)
            )
            labels += [self.ignore_index] * (self.model_max_length - len(labels))
            attention_mask += [0] * (self.model_max_length - len(attention_mask))
        # convert to pt tensor
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class QwenSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path, tokenizer, max_len=256, data_type="train"):
        super(QwenSupervisedDataset, self).__init__()
        
        self.dataset = datasets.load_dataset('json', data_dir=data_path)
        self.data = [example["conversations"] for example in self.dataset[data_type]]
        self.tokenizer = tokenizer
        self.max_len = int(max_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.preprocess(self.data[i])

    def preprocess(
        self,
        source,
        system_message: str = "You are a helpful assistant."
    ) -> Dict:
        roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
    
        im_start = self.tokenizer('<|im_start|>').input_ids[0]
        im_end = self.tokenizer('<|im_end|>').input_ids[0]
        nl_tokens = self.tokenizer('\n').input_ids
        _system = self.tokenizer('system').input_ids + nl_tokens
        _user = self.tokenizer('user').input_ids + nl_tokens
        _assistant = self.tokenizer('assistant').input_ids + nl_tokens
    
        # Apply prompt templates
        input_ids, targets = [], []
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]
    
        input_id, target = [], []
        system = [im_start] + _system + self.tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = self.tokenizer(role).input_ids + nl_tokens + \
                self.tokenizer(sentence["content"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(self.tokenizer(role).input_ids) + \
                    _input_id[len(self.tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        if len(input_id) < self.max_len:
            input_id += [self.tokenizer.pad_token_id] * (self.max_len - len(input_id))
            target += [IGNORE_TOKEN_ID] * (self.max_len - len(target))
        input_id = input_id[:self.max_len]
        target = target[:self.max_len]
        input_id = torch.tensor(input_id, dtype=torch.int)
        target = torch.tensor(target, dtype=torch.int)
        attention_mask = input_id.ne(self.tokenizer.pad_token_id)
        return {
            "input_ids": input_id,
            "labels": target,
            "attention_mask": attention_mask,
        }
    

if __name__ == "__main__":
   tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], use_fast=False, trust_remote_code=True)
   print(tokenizer)
   print(tokenizer.pad_token_id)
   data_path = sys.argv[2]
   #print(data_path)
   chat_dataset = GLM3ChatDataset(data_path, tokenizer, data_type="test") 
   for data in chat_dataset:
       print(data)
   print("load data")

   

