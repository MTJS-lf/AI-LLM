from typing import Dict, Union, List, Tuple
from collections import OrderedDict
import inspect

import torch
from torch import nn, Tensor
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import transformers


class BaseExportModel(nn.Sequential):

    def __init__(self, model, model_name, device=None, max_seq_len=None, max_query_length=None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._base_config = AutoConfig.from_pretrained(model_name)
        self.to(device)  # to device
        if max_seq_len is None:
            self._max_len = self._base_config.max_position_embeddings
        else:
            self._max_len = max_seq_len
        self._do_lower_case = self.tokenizer.do_lower_case
        signature = inspect.signature(self.model.forward)
        print(list(signature.parameters.keys()))
        input_names = list(signature.parameters.keys())
        self._input_names = ["input_ids", "token_type_ids", "attention_mask"]
        if max_query_length is not None:
            if "query_seg_ids" in input_names:
                self._input_names.append("query_seg_ids")
            if "title_seg_ids" in input_names:
                self._input_names.append("title_seg_ids")
        self._tokenizer = self.tokenizer
        self.max_query_length = max_query_length

    @property
    def max_seq_len(self):
        return self._max_len

    @property
    def do_lower_case(self):
        return self._do_lower_case

    @property
    def preprocessor_kwargs(self):
        return {'max_seq_len': self.max_seq_len, 'do_lower_case': self.do_lower_case}

    @property
    def config(self):
        return self._base_config

    @property
    def input_names(self):
        return self._input_names

    @property
    def output_names(self):
        return ['logits']

    @property
    def input_axes(self):
        dynamic_axes = {}
        for name in self.input_names:
            dynamic_axes[name] = {0: 'batch_size', 1: 'max_seq_len'}
        return dynamic_axes

    @property
    def output_axes(self):
        dynamic_axes = {}
        dynamic_axes['logits'] = {0: 'batch_size', 1: 'max_seq_len'}
        return dynamic_axes

    def save(self, save_path):
        self._tokenizer.save_pretrained(save_path)
        self._base_config.save_pretrained(save_path)

    def get_dummy_inputs(self, dummy=None, batch_size=1, device='cpu', return_tensors="pt"):
        text = dummy if dummy is not None else (" ".join([self._tokenizer.unk_token]) * self._max_len)
        dummy_input = [text] * batch_size
        features = self.tokenize(dummy_input)
        inputs = {}
        for name in self.input_names:
            if name in features:
                if return_tensors == "pt":
                    inputs[name] = features[name].to(device)
                else:
                    inputs[name] = features[name].cpu().numpy()
        if "query_seg_ids" in self.input_names:
             inputs["query_seg_ids"] = inputs["input_ids"][:, 1:self.max_query_length + 1]
        if "title_seg_ids" in self.input_names:
             inputs["title_seg_ids"] = inputs["input_ids"][:, self.max_query_length + 1:self._max_len - 2]
        print("inputs= ", inputs)
        return inputs

    def tokenize(self, texts: List[str]):
        if self._do_lower_case:
            texts = [s.lower() for s in texts]
        return self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=self._max_len)

    def forward(self, input_ids: Tensor=None, token_type_ids: Tensor=None, attention_mask: Tensor=None, positions_ids: Tensor=None, query_seg_ids: Tensor=None, title_seg_ids: Tensor=None, *args, **kwargs):
        inputs = {}
        if 'input_ids' in self.input_names:
            inputs['input_ids'] = input_ids
        if 'attention_mask' in self.input_names:
            inputs['attention_mask'] = attention_mask
        if 'token_type_ids' in self.input_names:
            inputs['token_type_ids'] = token_type_ids
        if 'positions_ids' in self.input_names:
            inputs['positions_ids'] = positions_ids
        if "query_seg_ids" in self.input_names:
            inputs['query_seg_ids'] = query_seg_ids
        if "title_seg_ids" in self.input_names:
            inputs['title_seg_ids'] = title_seg_ids
        outputs = self.model(**inputs)
        ret = OrderedDict()
        for name in self.output_names:
            ret[name] = outputs[name].detach()
        return ret

    def encode(self, texts: List[str]):
        if isinstance(texts, str):
            texts = [texts]
        features = self.tokenize(texts)
        features = {k:v.to(self._device) for k,v in features.items()}
