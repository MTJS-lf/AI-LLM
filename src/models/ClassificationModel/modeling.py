import logging

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, Tensor
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from src.utils.model_utils import ClassificationModelOutput


logger = logging.getLogger(__name__)

class ClassificationModel(nn.Module):

    def __init__(self, model_name, problem_type="single_label_classification", num_labels=1):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.config = self.model.config
        self.problem_type = problem_type
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        if self.problem_type == "regression":
            self.loss_fct = nn.MSELoss()
        elif self.problem_type == "multi_label_classification":
            self.loss_fct = nn.BCEWithLogitsLoss()
        else:
            self.loss_fct = nn.BCEWithLogitsLoss()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def compute_loss(self, logits, labels):
        if self.problem_type == "regression":
            loss = self.loss_fct(logits.squeeze(), labels.squeeze())
        elif  self.problem_type == "multi_label_classification":
            loss = self.loss_fct(logits, labels)
        else:
            loss = self.loss_fct(logits.view(-1), labels.float().view(-1))
        return loss

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, labels=None, return_dict=True, **kw_inputs):
        base_model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=return_dict)
        base_pooled_output = base_model_output[1]
        logits = self.classifier(base_pooled_output)
        model_outputs = {"logits": logits}
        if labels is not None:
            loss = self.compute_loss(logits, labels)
            model_outputs["loss"] = loss
            return model_outputs
        else:
            return model_outputs


    def save_pretrained(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)


class ClassificationMatchModel(ClassificationModel):
    def __init__(self, 
            model_name, 
            problem_type="single_label_classification", 
            num_labels=1,
            max_query_len=20,
            max_title_len=41, 
            use_match=True,
            is_training=True):

        super(ClassificationMatchModel, self).__init__(model_name, problem_type, num_labels)

        self.max_query_len = max_query_len
        self.max_title_len = max_title_len
        self.max_len = self.max_query_len + self.max_title_len + 3
        self.output_channels = 3
        self.pool_dim = 32
        self.pool_layer = nn.Linear(768, self.pool_dim)
        self.query_classifier = nn.Linear(768, 1)
        self.kernal_size = 2
        self.conv = torch.nn.Conv2d(3, self.output_channels, (self.kernal_size, self.kernal_size))
        self.padding_height = ((self.max_query_len - 1) * 1 + self.kernal_size - self.max_query_len) % self.kernal_size
        self.padding_width = ((self.max_title_len - 1) * 1 + self.kernal_size - self.max_title_len) % self.kernal_size
        self.feature_pool = nn.MaxPool2d(kernel_size=(1, self.max_title_len))
        self.use_match = use_match
        classifier_input_dim = self.pool_dim
        if self.use_match:
            classifier_input_dim = self.pool_dim + self.max_query_len * self.output_channels
        self.classifier = nn.Linear(classifier_input_dim, 1)
        self.is_training = is_training
        self.use_match = use_match

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def compute_loss(self, logits, labels):
        if self.problem_type == "regression":
            loss = self.loss_fct(logits.squeeze(), labels.squeeze())
        elif  self.problem_type == "multi_label_classification":
            loss = self.loss_fct(logits, labels)
        else:
            loss = self.loss_fct(logits.view(-1), labels.float().view(-1))
        return loss


    def forward(self,
            input_ids,
            attention_mask=None,
            token_type_ids=None, 
            postion_ids=None,
            query_seg_ids=None,
            title_seg_ids=None,
            labels=None,
            return_dict=True,
            **kw_inputs):

        base_model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=postion_ids, return_dict=return_dict)
        base_pooled_output = base_model_output[1]
        pooled_output = self.pool_layer(base_pooled_output)
        logits = None
        if self.use_match:
           sequence_output = base_model_output[0]
           query_sequence_output = sequence_output[:, 1:self.max_query_len + 1, :]
           query_sequence_output = torch.nn.functional.normalize(query_sequence_output, dim=-1)
           title_sequence_output = sequence_output[:, self.max_query_len + 1:self.max_len-2, :]
           title_sequence_output = torch.nn.functional.normalize(title_sequence_output, dim=-1)
           query_weights = self.query_classifier(query_sequence_output)

           ## char match
           query_char_ids_pad = input_ids[:, 1:self.max_query_len + 1]
           title_char_ids_pad = input_ids[:, self.max_query_len + 1:self.max_len-2]
           query_char_ids = torch.where(query_char_ids_pad == 100, -100, query_char_ids_pad)
           title_char_ids = torch.where(title_char_ids_pad == 100, -101, title_char_ids_pad)
           query_char_ids_tensor = query_char_ids.view(-1, self.max_query_len, 1)
           title_char_ids_tensor = title_char_ids.view(-1, 1, self.max_title_len)
           qt_char_match = torch.eq(query_char_ids_tensor, title_char_ids_tensor)
           qt_char_match = qt_char_match.view(-1, 1, self.max_query_len, self.max_title_len)

           query_mask_ids = torch.where(query_char_ids_pad == 100, 0, 1)
           query_mask_ids_tensor = query_mask_ids.view(-1, self.max_query_len, 1)
           query_weights = torch.mul(query_mask_ids_tensor, query_weights)

           title_mask_ids = torch.where(title_char_ids_pad == 100, 0, 1)
           title_mask_ids_tensor = title_mask_ids.view(-1, self.max_title_len, 1)

           # sem match
           query_sequence_output = torch.mul(query_sequence_output, query_mask_ids_tensor)
           title_sequence_output = torch.mul(title_sequence_output, title_mask_ids_tensor)
           title_sequence_output = torch.transpose(title_sequence_output, 2, 1)
           qt_sem_match = torch.matmul(query_sequence_output, title_sequence_output)
           qt_sem_match = qt_sem_match.view(-1, 1, self.max_query_len, self.max_title_len)

           # segment match 
           query_seg_ids = torch.where(query_seg_ids == 100, -100, query_seg_ids)
           title_seg_ids = torch.where(title_seg_ids == 100, -101, title_seg_ids)
           query_seg_ids_tensor = query_seg_ids.view(-1, self.max_query_len, 1)
           title_seg_ids_tensor = title_seg_ids.view(-1, 1, self.max_title_len)
           qt_seg_match = torch.eq(query_seg_ids_tensor, title_seg_ids_tensor)
           qt_seg_match = qt_seg_match.view(-1, 1, self.max_query_len, self.max_title_len)

           # concat match
           qt_match = torch.concat((qt_sem_match, qt_char_match, qt_seg_match), dim=1)
           qt_match_pad = nn.functional.pad(qt_match, (self.padding_height, self.padding_height - 1, self.padding_width, self.padding_width - 1))
           feature_map = self.conv(qt_match_pad)
           pool_out = self.feature_pool(feature_map)
           pool_out = pool_out.view(-1, self.max_query_len, self.output_channels)

           weight_match_layer = torch.mul(query_weights, pool_out)
           match_layer_flat = weight_match_layer.view(-1, self.output_channels * self.max_query_len)
           qt_match = torch.concat((pooled_output, match_layer_flat), dim=-1)
           logits = self.classifier(qt_match)
        else:
           logits = self.classifier(pooled_output)

        #model_output = ClassificationModelOutput(
        #        logits=logits,
        #        scores=logits,
        #        )
        model_output = {
            "logits": logits,
            "scores": logits
        }
        if labels is not None:
            loss = self.compute_loss(logits, labels)
            model_output["loss"] = loss
            print("model:", model_output)
            return model_output
        elif self.training:
            raise ValueError('training stage input should set labels.')
        else:
            return model_output


if __name__ == "__main__":
    import sys     
    model_dir = "/data/BaseModel/dienstag/chinese-roberta-wwm-ext"
    model = ClassificationMatchModel(model_dir, is_training=False)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    print(tokenizer)
