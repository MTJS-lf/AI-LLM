import logging
import os
import sys
import yaml
from copy import deepcopy
from pathlib import Path

import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
    default_data_collator,
)

sys.path.append('..')
sys.path.append('../../')
from src.trainer.trainer import BaseTrainer
from src.models.ClassificationModel.modeling import ClassificationModel
from src.datasets.rank_dataset import *
from src.utils.arguments import ModelArguments, DataArguments, TrainingArguments
from src.utils.util import model_exporter
from src.export_model.transformer_seq_cls_model import TransformerSequenceClassificationModel
from src.export_model.base_export_model import BaseExportModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)

transformers.logging.set_verbosity_error()


def main():
    model = AutoModelForSequenceClassification.from_pretrained(sys.argv[1])
    export_model = BaseExportModel(model, sys.argv[1])
    #export_model= TransformerSequenceClassificationModel(sys.argv[1])
    onnx_inputs, onnx_outputs = model_exporter(export_model, "models_test_v1")
    print('onnx inputs', onnx_inputs)
    print('onnx outputs', onnx_outputs)    

if __name__ == "__main__":
    main()
