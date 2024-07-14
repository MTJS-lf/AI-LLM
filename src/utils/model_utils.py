from dataclasses import dataclass
from typing import Dict, Optional

from torch import Tensor
from transformers.file_utils import ModelOutput

@dataclass
class ClassificationModelOutput(ModelOutput):
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None
    scores: Optional[Tensor] = None

@dataclass
class EncoderModelOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

@dataclass
class Seq2SeqModelOutput(ModelOutput):
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None
