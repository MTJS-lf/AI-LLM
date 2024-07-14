# !/usr/bin/python
# -*- coding: utf-8 -*-

import math
from abc import ABC
from typing import Union, Dict

import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler

class TorchDataSet(Dataset):
    def __init__(self, config, mode: str):
        """

        Args:
            config(OmegaConf): 配置文件
            mode(str): 当前状态

        """
        super().__init__()
        self.config = config
        self.mode = mode
        self.logger = init_logger(__name__)

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError


class TorchIterableDataSet(IterableDataset, ABC):
    def __init__(self, config, mode: str):
        """

        Args:
            config(OmegaConf): 配置文件
            mode(str): 当前状态

        """
        super().__init__()
        self.config = config
        self.mode = mode
        self.file_path = self.config.data_path[f"{mode}_file"]
        self.start = 0
        self.end = self._get_file_line_num(self.file_path)

        self.rank = config.distributed.rank if torch.distributed.is_initialized() else 0
        self.world_size = config.distributed.world_size if torch.distributed.is_initialized() else 1

        self.per_worker = int(math.floor((self.end - self.start) / float(self.world_size)))
        self.iter_start = self.start + self.rank * self.per_worker
        self.iter_end = min(self.iter_start + self.per_worker, self.end)

        self.logger = init_logger(__name__)

    def _get_file_line_num(self, file_path):
        end = 0
        with open(file_path, 'r') as fin:
            for _ in enumerate(fin):
                end += 1
        return end

    def __iter__(self):
        sample_iterator = self._sample_generator(self.iter_start, self.iter_end)
        return sample_iterator

    def _sample_generator(self, start, end):
        with open(self.file_path, 'r') as fin:
            for i, line in enumerate(fin):
                if i < start:
                    continue
                if i >= end:
                    break
                record = self._build_record(line)
                yield record

    def _build_record(self, line):
        raise NotImplementedError

    def __len__(self):
        return self.end - self.start


def dataloader_builder(datasets: Dict, config: Union[DictConfig, ListConfig], is_distributed: bool):
    """
    构造生成四个 dataloader (train, eval, test, predict)，支持分布式训练

    """
    dataloaders = {}
    for mode in ['train', 'eval', 'test', 'predict']:
        if mode not in datasets:
            continue
        if is_distributed:
            batch_size = config.data["per_gpu_%s_batch_size" % mode]
        else:
            batch_size = config.data["per_gpu_%s_batch_size" % mode] * max(1, len(config.gpu_list))
        dataset = datasets[mode]
        if is_distributed and mode in config.distributed.distributed_step and not isinstance(dataset, IterableDataset):
            sampler = DistributedSampler(dataset)
        else:
            sampler = None
        num_workers = 0 if isinstance(dataset, IterableDataset) else config.data.num_dataloader_workers
        dataloaders[mode] = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=config.data.pin_memory,
            drop_last=config.data.drop_last,
            shuffle=(not is_distributed and mode == "train" and not isinstance(dataset, IterableDataset))
        )
    return dataloaders

