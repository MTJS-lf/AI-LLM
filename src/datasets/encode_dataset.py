import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer


class TrainDatasetForEmbedding(Dataset):
    def __init__(
            self,
            data_path,
            train_group_size,
            tokenizer: PreTrainedTokenizer,
            query_instruction=None,
            passage_instruction=None
    ):
        self.dataset = datasets.load_dataset('json', data_files=data_path, split='train')

        self.tokenizer = tokenizer
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]['query']
        #if query_instruction is not None:
        #    query = query_instruction + query

        passages = []

        assert isinstance(self.dataset[item]['pos'], list)
        pos = random.choice(self.dataset[item]['pos'])
        passages.append(pos)

        #if len(self.dataset[item]['neg']) < train_group_size - 1:
        #    num = math.ceil((train_group_size, - 1) / len(self.dataset[item]['neg']))
        #    negs = random.sample(self.dataset[item]['neg'] * num, train_group_size, - 1)
        #else:
        #    negs = random.sample(self.dataset[item]['neg'], train_group_size, - 1)
        #passages.extend(negs)

        #if passage_instruction is not None:
        #    passages = [passage_instruction+p for p in passages]
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
