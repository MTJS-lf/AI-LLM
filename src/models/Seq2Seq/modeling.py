import sys
import torch
import torch.nn as nn
from src.layers.CRF import CRF
from transformers import AutoModel
from src.utils.model_utils import Seq2SeqModelOutput

class TransformerNerModel(nn.Module):
    def __init__(self,
            model_name,
            use_lstm=False,
            use_crf=True,
            num_layers=2,
            lstm_hidden=256,
            dropout=0.1, 
            num_labels=7,
            max_seq_len=128,
            **kwargs,
            ):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.config = self.base_model.config
        self.num_layers = num_layers
        self.lstm_hidden = lstm_hidden
        self.num_labels = num_labels
        self.use_lstm = use_lstm
        self.use_crf = use_crf
        self.config = self.base_model.config
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        hidden_dims = self.config.hidden_size

        if self.use_lstm:
            self.lstm = nn.LSTM(hidden_dims, self.lstm_hidden, self.num_layers, bidirectional=True, batch_first=True, dropout=self.dropout)
            self.linear = nn.Linear(self.lstm_hidden * 2, self.num_labels)
            init_blocks = [self.linear]
        else:
            mid_linear_dims = kwargs.pop('mid_linear_dims', 128)
            self.mid_linear = nn.Sequential(
                nn.Linear(hidden_dims, mid_linear_dims),
                nn.ReLU(),
                nn.Dropout(self.dropout))
            hidden_dims = mid_linear_dims
            self.classifier = nn.Linear(hidden_dims, self.num_labels)
            init_blocks = [self.mid_linear, self.classifier]

        self._init_weights(init_blocks, initializer_range=self.config.initializer_range)
        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=True)

        self.loss_fct = nn.CrossEntropyLoss()

    def _init_weights(self, blocks, **kwargs):
        """
        init_weights
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def init_hidden(self, batch_size):
        h0 = torch.randn(2 * self.num_layers, batch_size, self.lstm_hidden, requires_grad=True).to(self.device)
        c0 = torch.randn(2 * self.num_layers, batch_size, self.lstm_hidden, requires_grad=True).to(self.device)
        return h0, c0

    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable(**kwargs)

    def forward(self,
               input_ids,
               token_type_ids=None,
               attention_mask=None,
               postion_ids=None,
               labels=None,
               return_dict=True,
               **kw_inputs):
        base_outputs = self.base_model(
           input_ids=input_ids,
           attention_mask=attention_mask,
           token_type_ids=token_type_ids
        )

        seq_out = base_outputs[0]
        batch_size = seq_out.size(0)

        if self.use_lstm:
            seq_out, _ = self.lstm(seq_out)
            seq_out = seq_out.contiguous().view(-1, self.lstm_hidden * 2)
            seq_out = self.linear(seq_out)
            seq_out = seq_out.contiguous().view(batch_size, self.max_seq_len, -1) #[batch_size, max_len, num_labels]:
        else:
            seq_out = self.mid_linear(seq_out)  # [batch_size, max_len, 128]
            seq_out = self.classifier(seq_out)  # [24, max_len, num_labels]
        #model_output = Seq2SeqModelOutput()
        model_output = {}
        if self.use_crf:
            logits = self.crf.decode(seq_out)
            #model_output.logits = torch.tensor(logits)
            model_output['logits'] = torch.tensor(logits)
            if labels is None:
                return model_output
            loss = -self.crf(seq_out, labels, mask=attention_mask, reduction='mean')
            model_output['loss'] = loss
            return model_output
        else:
            logits = seq_out
            model_output['logits'] = logits
            if labels is None:
                return model_output
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.size()[2])[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.loss_fct(active_logits, active_labels)
            model_output['loss'] = loss
            return model_output
