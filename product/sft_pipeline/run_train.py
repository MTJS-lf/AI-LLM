# -*- coding: utf-8 -*-

import os
import math
import logging
import json
import sys
import yaml
from copy import deepcopy
from pathlib import Path

import transformers
import torch
from packaging import version

from typing import Optional
from functools import partial
from dataclasses import dataclass, field

from transformers.utils import add_start_docstrings
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import torch_distributed_zero_first
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    LlamaTokenizer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

sys.path.append('../../')
from src.datasets.chat_dataset import QwenSupervisedDataset
from src.utils.arguments import ModelArguments, DataArguments, TrainingArguments

from transformers import Trainer

logger = logging.getLogger(__name__)

def print_rank_0(msg, log_file, rank=0):
    if rank <= 0:
        with open(log_file, "a") as f:
            print(msg)
            f.write(msg + "\n")


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

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = torch.distributed.get_rank()
    log_file = os.path.join(training_args.output_dir, "print_log.txt")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, fp16-bits training: {training_args.fp16}, bf16-bits training: {training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    training_args._frozen = False
    training_args.data_seed = training_args.seed

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
    )
    print(model)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="right", use_fast=False, trust_remote_code=True, pad_token='<|endoftext|>')
    tokenizer.padding_side = "left"  # Allow batched inference

    print_rank_0(
        "tokenizer.eos_token_id = {}".format(tokenizer.eos_token_id),
        log_file,
        global_rank,
    )
    print_rank_0(
        "tokenizer.pad_token_id = {}".format(tokenizer.pad_token_id),
        log_file,
        global_rank,
    )
    print_rank_0(
        "tokenizer.bos_token_id = {}".format(tokenizer.bos_token_id),
        log_file,
        global_rank,
    )

    # peft model
    if training_args.use_lora:
        print_rank_0(
            "Loading lora config from {}".format(training_args.lora_config),
            log_file,
            global_rank,
        )
        lora_config = json.load(open(training_args.lora_config))
        print_rank_0("Lora config: {}".format(lora_config), log_file, global_rank)
        if training_args.use_int8_training:
            print_rank_0(
                "training_args.use_int8_training!!! (int8 is not compatible with DeepSpeed)",
                log_file,
                global_rank,
            )
            model = prepare_model_for_int8_training(model)
        config = LoraConfig(
            r=lora_config["lora_r"],
            lora_alpha=lora_config["lora_alpha"],
            target_modules=lora_config["lora_target_modules"],
            lora_dropout=lora_config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        # "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    print(model)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # model.is_parallelizable = True
    # model.model_parallel = True

    assert os.path.exists(data_args.train_path), "{} file not exists".format(data_args.train_path)

    with torch_distributed_zero_first(global_rank):
        train_data = QwenSupervisedDataset(data_args.train_path, tokenizer, data_type="train")
        val_data = QwenSupervisedDataset(data_args.eval_path, tokenizer, data_type="test")

    for i in range(2):
        print_rank_0(
            "Eval tokenized example: {}".format(val_data[i]), log_file, global_rank
        )
    for i in range(2):
        print_rank_0(
            "Train tokenized example: {}".format(train_data[i]), log_file, global_rank
        )

    training_nums = len(train_data)
    num_gpus = torch.cuda.device_count()

    batch_size = (
        training_args.per_device_train_batch_size
        * training_args.world_size
        * training_args.gradient_accumulation_steps
    )
    # train steps
    t_total = math.ceil(training_nums / batch_size) * training_args.num_train_epochs
    # eval steps
    training_args.eval_steps = max(t_total // (training_args.num_train_epochs * 4), 5)
    # save steps
    training_args.save_steps = training_args.eval_steps
    training_args.warmup_steps = (
        int(t_total * training_args.warmup_ratio)
        if training_args.warmup_ratio > 0.0
        else training_args.warmup_steps
    )
    print_rank_0(
        "num_gpus = {}, training_nums = {}, t_total = {}, warmup_steps = {}, eval_steps = {}, save_steps = {}".format(
            num_gpus,
            training_nums,
            t_total,
            training_args.warmup_steps,
            training_args.eval_steps,
            training_args.save_steps,
        ),
        log_file,
        global_rank,
    )
    print_rank_0(
        "val data nums = {}, training_nums = {}, batch_size = {}".format(
            len(val_data), training_nums, batch_size
        ),
        log_file,
        global_rank,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    print_rank_0(
        f"Using {training_args.half_precision_backend} half precision backend",
        log_file,
        global_rank,
    )
    # Train!
    len_dataloader = len(trainer.get_train_dataloader())
    num_update_steps_per_epoch = (
        len_dataloader // training_args.gradient_accumulation_steps
    )

    total_train_batch_size = (
        training_args.train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    num_examples = trainer.num_examples(trainer.get_train_dataloader())
    num_train_samples = num_examples * training_args.num_train_epochs
    max_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
    print_rank_0("***** Running training *****", log_file, global_rank)
    print_rank_0(f"  Num examples = {num_examples}", log_file, global_rank)
    print_rank_0(f"  Num train samples = {num_train_samples}", log_file, global_rank)
    print_rank_0(f"  world_size = {world_size}", log_file, global_rank)
    print_rank_0(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}",
        log_file,
        global_rank,
    )
    print_rank_0(
        f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}",
        log_file,
        global_rank,
    )
    print_rank_0(f"  Total optimization steps = {max_steps}", log_file, global_rank)

    # https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958/3
    model.config.use_cache = False

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(training_args.output_dir)
    print_rank_0(
        "\n Training completed!!! If there's a warning about missing keys above, please disregard :)",
        log_file,
        global_rank,
    )


if __name__ == "__main__":
    main()

