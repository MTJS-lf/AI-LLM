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
from src.models.ClassificationModel.modeling import ClassificationModel, ClassificationMatchModel
from src.datasets.rank_dataset import TrainDatasetForCrossModel, TrainDatasetMatchForCrossModel
from src.utils.arguments import ModelArguments, DataArguments, TrainingArguments
from src.utils.util import model_exporter
from src.export_model.transformer_seq_cls_model import TransformerSequenceClassificationModel
from src.export_model.base_export_model import BaseExportModel

logger = logging.getLogger(__name__)

transformers.logging.set_verbosity_error()


def main():
    config_path = "conf.yml"
    with open(config_path, "r", encoding="utf8") as fc:
        conf = yaml.safe_load(fc)
    
    train_args = deepcopy(conf["train_args"])  
    training_args = TrainingArguments(
        **train_args,
    )
    print(training_args)
    model_conf = deepcopy(conf["model_args"])
    model_args = ModelArguments(
        **model_conf,
    )
    data_conf = deepcopy(conf["data_args"])
    data_args = DataArguments(
        **data_conf,
    )

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    model = ClassificationMatchModel(model_name=model_args.model_name_or_path, problem_type="regression")

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False

    train_dataset = TrainDatasetMatchForCrossModel(data_args.train_path,
                                                   tokenizer=tokenizer,
                                                   query_max_length=20,
                                                   model_max_length=model_args.model_max_length,
                                                   problem_type="regression")
    eval_dataset = TrainDatasetMatchForCrossModel(data_args.eval_path,
                                                   tokenizer=tokenizer,
                                                   query_max_length=20,
                                                   model_max_length=model_args.model_max_length,
                                                   data_type="test",
                                                   problem_type="regression")

    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)

    if training_args.export_onnx:
        export_model = BaseExportModel(model, model_args.model_name_or_path, max_seq_len=model_args.model_max_length, max_query_length=20)
        onnx_inputs, onnx_outputs = model_exporter(export_model, training_args.export_path)
        print('onnx inputs', onnx_inputs)
        print('onnx outputs', onnx_outputs)    

if __name__ == "__main__":
    main()
