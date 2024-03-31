import os
import torch
import argparse

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_dir', type=str, help='')
    parser.add_argument('--model_dir', type=str, help='')

    return parser.parse_args()


if __name__ == "__main__":
    args = set_args()
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("base NODEL")
    print(base_model)
    print("base NODEL")
    model = PeftModel.from_pretrained(base_model, args.model_dir, torch_dtype=torch.float16)
    print(model)

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.model_dir)
    print(merged_model)
