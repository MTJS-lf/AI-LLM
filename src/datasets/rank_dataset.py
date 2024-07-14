import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple
import jieba

import datasets
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer,AutoTokenizer

max_int = torch.iinfo(torch.int64).max

class TrainDatasetForEmbedding(Dataset):
    def __init__(
            self,
            data_path,
            train_group_size,
            tokenizer: PreTrainedTokenizer,
            query_instruction=None,
            passage_instruction=None,
            data_type="train"
    ):

        print(data_path)
        self.dataset = datasets.load_dataset('json', data_dir=data_path)
        print(self.dataset)
        print(self.dataset[data_type][0])
        self.dataset = self.dataset[data_type]

        self.tokenizer = tokenizer
        self.total_len = len(self.dataset)
        self.query_instruction = query_instruction
        self.passage_instruction = passage_instruction
        self.train_group_size = train_group_size

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]['query']
        if self.query_instruction is not None:
            query = self.query_instruction + query

        passages = []

        assert isinstance(self.dataset[item]['pair'], str)
        pos = self.dataset[item]['pair']
        passages.append(pos)

        if "neg" in self.dataset[item]:
            if len(self.dataset[item]['neg']) < self.train_group_size - 1:
                num = math.ceil((self.train_group_size, - 1) / len(self.dataset[item]['neg']))
                negs = random.sample(self.dataset[item]['neg'] * num, self.train_group_size - 1)
            else:
                negs = random.sample(self.dataset[item]['neg'], self.train_group_size - 1)
            passages.extend(negs)

        if self.passage_instruction is not None:
            passages = [self.passage_instruction+p for p in passages]
        return query, passages

@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "passage": d_collated}


class TrainDatasetForCrossModel(Dataset):
    def __init__(
            self,
            data_path,
            tokenizer: PreTrainedTokenizer,
            model_max_length=128,
            problem_type="single_label_classification",
            data_type="train"
    ):
        self.dataset = datasets.load_dataset('csv', data_dir=data_path, sep=',')
        print(data_path)
        print(self.dataset)
        print(self.dataset[data_type][0])
        self.dataset = self.dataset[data_type]
        self.total_len = len(self.dataset)
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.problem_type = problem_type

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        item = self.tokenizer.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation=True,
            max_length=self.model_max_length,
            padding="max_length",
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):

        assert isinstance(self.dataset[item]['sentence1'], str)
        assert isinstance(self.dataset[item]['sentence2'], str)
        sentence1 = self.dataset[item]['sentence1']
        sentence2 = self.dataset[item]['sentence2']
        features = self.create_one_example(sentence1, sentence2)
        label = 1
        import random
        if random.random() > 0.5:
            label = 0
        if "label" in self.dataset[item]:
            if self.problem_type == "single_label_classification" or self.problem_type == "multi_label_classification":
                label = int(self.dataset[item]['label'])
            else:
                label = float(self.dataset[item]['label'])
        features["labels"] = label
        return features

class TrainDatasetMatchForCrossModel(Dataset):
    def __init__(
            self,
            data_path,
            tokenizer: PreTrainedTokenizer,
            query_max_length=20,
            model_max_length=64,
            problem_type="single_label_classification",
            data_type="train"
    ):
        self.dataset = datasets.load_dataset('csv', data_dir=data_path, sep=',')
        #self.dataset = datasets.load_dataset('json', data_dir=data_path)
        print(data_path)
        print(self.dataset)
        print(self.dataset[data_type][0])
        self.dataset = self.dataset[data_type]
        self.total_len = len(self.dataset)
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        #self.title_max_length = title_max_length
        self.problem_type = problem_type
        self.model_max_length = model_max_length
        self.title_max_length = self.model_max_length - self.query_max_length - 3
        self.data_type = data_type

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        item = self.tokenizer.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation=True,
            max_length=self.model_max_length,
            padding="max_length",
        )
        return item

    def create_text(self, text, begin=0, pos_begin=0, type='q'):
        max_len = self.query_max_length
        if type == 'd':
            max_len = self.title_max_length - 1
        token_dict = self.tokenizer(text)
        input_ids = token_dict["input_ids"]
        token_type_ids = token_dict["token_type_ids"]
        attention_mask = token_dict["attention_mask"]
        input_ids = input_ids[begin:-1]
        token_type_ids = token_type_ids[begin:-1]
        attention_mask = attention_mask[begin:-1]
        position_ids = [ pos_begin + i for i in range(len(input_ids)) ]
        if len(position_ids) > 0:
            max_position_ids = position_ids[-1]
        else:
            max_position_ids = 0
        seq_len = len(input_ids)
        input_ids_size = len(input_ids)
        if input_ids_size > max_len + 1:
            input_ids = input_ids[0:max_len+1]
            token_type_ids = token_type_ids[0:max_len+1]
            attention_mask = attention_mask[0:max_len+1]
            position_ids = position_ids[0:max_len+1]
            max_position_ids = position_ids[-1]
        else:
            left_size = max_len + 1 - input_ids_size
            left_ids = [ 0 for i in range(left_size) ]
            input_ids.extend(left_ids)
            token_type_ids.extend(left_ids)
            attention_mask.extend(left_ids)
            position_ids.extend(left_ids)
        input_ids.extend([self.tokenizer.sep_token_id])
        token_type_ids.extend([0])
        attention_mask.extend([1])
        position_ids.extend([max_position_ids+1])
        if type == 'd':
            token_type_ids = [ 1 for i in range(len(input_ids)) ]
        seg_ids = []
        seg_list = jieba.cut(text)
        all_seg = []
        for seg in seg_list:
            seg_id = 0
            s_size = len(seg)
            all_seg.append(seg)
            if s_size > 1:
                for s in seg:
                    char_id = self.tokenizer.convert_tokens_to_ids(s)
                    seg_id = seg_id % max_int
                    seg_id = seg_id * 3 +  5 * (char_id % 10) + 3 * (char_id % 20) + 2 * (char_id % 30)

            else:
                char_id = self.tokenizer.convert_tokens_to_ids(seg)
                seg_id = char_id
            seg_id = seg_id % max_int
            seg_ids.extend([ seg_id for i in range(s_size) ])
        if type == 'd':
            max_len = self.title_max_length
        if len(seg_ids) > max_len:
            seg_ids = seg_ids[:max_len]
        else:
            left_size = max_len - len(seg_ids)
            left_ids = [ 100 for i in range(left_size) ]
            seg_ids.extend(left_ids)
        return {"input_ids": input_ids, "token_type_ids":token_type_ids, "attention_mask":attention_mask, "position_ids":position_ids, "seg_ids":seg_ids, "seq_len": seq_len}

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):

        assert isinstance(self.dataset[item]['sentence1'], str)
        assert isinstance(self.dataset[item]['sentence2'], str)
        sentence1 = self.dataset[item]['sentence1']
        sentence2 = self.dataset[item]['sentence2']
        query_ids = self.create_text(sentence1)
        query_seq_len = query_ids["seq_len"]
        title_ids = self.create_text(sentence2, begin=1, pos_begin=query_seq_len+1, type='d')
        input_ids = query_ids["input_ids"] + title_ids["input_ids"]
        token_type_ids = query_ids["token_type_ids"] + title_ids["token_type_ids"]
        attention_mask = query_ids["attention_mask"] + title_ids["attention_mask"]
        position_ids = query_ids["position_ids"] + title_ids["position_ids"]
        query_seg_ids = query_ids["seg_ids"]
        title_seg_ids = title_ids["seg_ids"]
        features = {"input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "query_seg_ids": query_seg_ids,
                    "title_seg_ids": title_seg_ids
                }
        label = 0
        if "label" in self.dataset[item]:
            if self.problem_type == "single_label_classification" or self.problem_type == "multi_label_classification":
                label = int(self.dataset[item]['label'])
            else:
                label_id = int(self.dataset[item]['label'])
                if self.data_type == "test":
                    if label_id  == 0:
                        label = 0.0
                    elif label_id == 1:
                        label = 0.5
                    elif label_id == 2:
                        label = 1
                else:
                    if label_id  == 0:
                        label = 0.0
                    elif label_id == 1:
                        label = 0.2
                    elif label_id == 2:
                        label = 0.45
                    elif label_id == 3:
                        label = 0.85
                    elif label_id == 4:
                        label = 1

        features["labels"] = label
        print("query = ", sentence1, "title = ", sentence2, "labele = ", label)
        return features

if __name__ == "__main__":
   tokenizer = AutoTokenizer.from_pretrained("/data/BaseModel/dienstag/chinese-roberta-wwm-ext/")
   data_path = "/data/relevamce_title_data/test"
   dataset = TrainDatasetMatchForCrossModel(data_path, tokenizer, data_type="test")
   for data in dataset:
       print(data)
