# -*- coding: utf-8 -*-

from peft import PeftModel

from transformers.trainer import *


class MyTrianer(Trainer):
    def save_model(self, output_dir: Optional[str]=None):
        return
