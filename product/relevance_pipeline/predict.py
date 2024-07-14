import logging
import os
import sys
import yaml
from copy import deepcopy
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
    Trainer,
    default_data_collator,
)

sys.path.append('..')
sys.path.append('../../')
from src.models.ClassificationModel.modeling import ClassificationMatchModel, ClassificationModel
from src.datasets.rank_dataset import *
from src.utils.arguments import ModelArguments, DataArguments, TrainingArguments
from src.export_model.base_export_model import BaseExportModel

logger = logging.getLogger(__name__)


def main():
    config_path = "conf.yml"
    with open(config_path, "r", encoding="utf8") as fc:
        conf = yaml.safe_load(fc)
    
    train_args = deepcopy(conf["train_args"])  
    training_args = TrainingArguments(
        **train_args,
    )
    model_conf = deepcopy(conf["model_args"])
    model_args = ModelArguments(
        **model_conf,
    )
    data_conf = deepcopy(conf["data_args"])
    data_args = DataArguments(
        **data_conf,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    model = ClassificationModel(model_name=model_args.model_name_or_path,
                           problem_type="regression"
                           )

    model.load_state_dict(torch.load("models/pytorch_model.bin"))

    #train_dataset = TrainDatasetMatchForCrossModel(data_args.eval_path, tokenizer=tokenizer, problem_type="regression", data_type="test")
    model.training = False
    export_model = BaseExportModel(model, model_args.model_name_or_path)
    tensor_inputs = export_model.get_dummy_inputs(dummy=None, batch_size=1, return_tensors="pt")
    print(tensor_inputs)
    #data_loader = DataLoader(train_dataset, batch_size = 1, collate_fn=default_data_collator)
    #for data in data_loader:
    #    #print(data)
    print(model(**tensor_inputs))

if __name__ == "__main__":
    main()

