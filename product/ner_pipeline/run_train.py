import logging
import os
import sys
import yaml
from copy import deepcopy
from pathlib import Path
import numpy as np
from seqeval.metrics import accuracy_score, classification_report

import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
    default_data_collator,
)

sys.path.append('..')
sys.path.append('../../')
from src.trainer.trainer import NerTrainer, BaseTrainer
from src.models.Seq2Seq.modeling import TransformerNerModel
from src.datasets.ner_dataset import TrainDatasetForNer
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

    train_dataset = TrainDatasetForNer(data_args.train_path, tokenizer=tokenizer, data_type="train")
    eval_dataset = TrainDatasetForNer(data_args.eval_path, tokenizer=tokenizer, data_type="test")
    id2label = train_dataset.id2label
    print(id2label, len(id2label))

    num_labels = len(id2label)
    model = TransformerNerModel(model_name=model_args.model_name_or_path,
                                use_lstm=model_args.use_lstm, 
                                use_crf=model_args.use_crf, 
                                num_labels=num_labels,
                                max_seq_len=model_args.model_max_length)
    print(model)

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False

    def compute_ner_metrics(p):
        predictions, labels = p
        print(predictions.shape, labels.shape)
        bs, seq_len = labels.shape 
        predictions = predictions.reshape((bs, seq_len)) 
        #predictions = np.argmax(predictions, axis=2)
        # Remove ignored index (special tokens)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
       
        report = classification_report(true_labels, true_predictions, output_dict=True) 
        
        report.pop('weighted avg')
        macro_score = report.pop('macro avg')
        micro_score = report.pop('micro avg')
    
        scores = {}
        scores["precision"] = micro_score['precision']
        scores["recall"] = micro_score['recall']
        scores["f1"] = micro_score['f1-score']
        scores["accuracy"] = accuracy_score(y_true=true_labels, y_pred=true_predictions)
        for tp, tp_score in report.items():
            scores[tp] = {}
            scores[tp]["precision"] = round(tp_score['precision'], 4)
            scores[tp]["recall"] = round(tp_score['recall'], 4)
            scores[tp]["f1"] = round(tp_score['f1-score'], 4)
            scores[tp]['support'] = tp_score['support']
        return scores

    trainer = NerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_ner_metrics,        
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
        export_model = BaseExportModel(model, model_args.model_name_or_path, max_seq_len=128)
        onnx_inputs, onnx_outputs = model_exporter(export_model, training_args.export_path)
        print('onnx inputs', onnx_inputs)
        print('onnx outputs', onnx_outputs)    

if __name__ == "__main__":
    main()
