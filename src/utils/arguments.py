import os
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments
from transformers.utils import add_start_docstrings


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_type: Optional[str] = field(default=None,  metadata={"help": "model type"})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    use_flash_attention: bool = field(
        default=False, metadata={"help": ("Whether to use memory efficient attention.")}
    )
    model_outputs: Optional[List[str]] = field(default=None, metadata={"help": "The list of keys for model output"})
    use_lstm: bool = field(default=False, metadata={"help": ("Whether to use lstm layer for ner model.")})
    use_crf: bool = field(default=False, metadata={"help": ("Whether to use crf layer for ner model.")})
    model_max_length: int = field(default=128, metadata={"help": "model max length for text"})
    query_max_length: int = field(default=64, metadata={"help": "query max length for relevance model"})


@dataclass
class DataArguments:
    knowledge_distillation: bool = field(
        default=False, metadata={"help": "Use knowledge distillation when `pos_scores` and `neg_scores` are in features of training data"}
    )
    train_path: Optional[str] = field(
        default=None, metadata={"help": "One or more paths to training data"}
    )
    eval_path: Optional[str] = field(
        default=None, metadata={"help": "One or more paths to eval data"}
    )
    cache_path: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the cached data"}
    )
    train_group_size: int = field(default=8)

    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_example_num_per_dataset: int = field(
        default=None, metadata={"help": "the max number of examples for each dataset"}
    )

    query_instruction_for_retrieval: str= field(
        default=None, metadata={"help": "instruction for query"}
    )
    passage_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "instruction for passage"}
    )
    
    same_task_within_batch: bool = field(
            default=False, metadata={"help": "All samples in the same batch comes from the same task."}
    )
    shuffle_ratio: float = field(
            default=0.0, metadata={"help": "The ratio of shuffling the text"}
    )
    
    small_threshold: int = field(
            default=0, metadata={"help": "The threshold of small dataset. All small dataset in the same directory will be merged into one dataset."}
    )
    drop_threshold: int = field(
            default=0, metadata={"help": "The threshold for dropping merged small dataset. If the number of examples in the merged small dataset is less than this threshold, it will be dropped."}
    )
    data_type: str = field(default="json", metadata={"help": "input data file type"})

@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class TrainingArguments(TrainingArguments):
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA."})
    use_int8_training: bool = field(
        default=False, metadata={"help": "Whether to use int8 training."}
    )
    lora_config: Optional[str] = field(
        default=None,
        metadata={"help": "LoRA config file."},
    )
    ddp_find_unused_parameters: bool = field(
        default=False, metadata={"help": "ddp_find_unused_parameters"}
    )
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "gradient_checkpointing"}
    )
    # https://discuss.huggingface.co/t/wandb-does-not-display-train-eval-loss-except-for-last-one/9170
    evaluation_strategy: str = field(
        default="steps", metadata={"help": "The evaluation strategy to use."}
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints"
            )
        },
    )
    report_to: str = field(
        default="wandb",
        metadata={
            "help": "The list of integrations to report the results and logs to."
        },
    )
    deepspeed: str = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) or an already"
                " loaded json file as a dict"
            )
        },
    )
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    export_onnx: bool = field(default=False, metadata={"help": "Whether to run training."})
    export_path: str = field(default=None, metadata={"help": "Path to export model for serving "})
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(default=False, metadata={"help": "Freeze the parameters of position embeddings"})
    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method, should be cls or mean"})
    normlized: bool = field(default=True)
    use_inbatch_neg: bool = field(default=True, metadata={"help": "use passages in the same batch as negatives"})
    add_dense_layer: bool = field(default=False, metadata={"help": "encode model add dense layer"})
    embedding_dim: int = field(default=128,metadata={"help": "if save_dense is True, should set embedding dim"})
