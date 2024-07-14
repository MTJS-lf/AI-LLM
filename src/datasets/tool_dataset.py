# -*- coding: utf-8 -*-

import os
import sys

from base_dataset import BaseDataset

class ToolDataset(BaseDataset):
    def __init__(self, config, tokenizer):
        super(ToolDataset, self).__init__(config)
        self.data_path = config["data_path"]
        self.tokenizer = tokenizer
        self.user_tokens = [self.tokenizer.get_command(f"<|user|>")]
        self.assistant_tokens = [self.tokenizer.get_command(f"<|assistant|>")]
        self.system_tokens = [self.tokenizer.get_command(f"<|system|>")]
        self.ingore_indec = -100
        self.data = json.load(open(self.data_pat))
        self.system_prompt = "You are a helpful assistant."
        self.prompt_ids = self.system_tokens + self.tokenizer.encode(self.system_prompt, add_special_tokens=False)
        self.tools_prompt = '''
          Use the following format:
          Question: the input question you must answer
          Thought: you should always think about what to do
          Action: the action to take, should be one of [google_search, code_interpreter]
          Action Input: the input to the action
          Observation: the result of the action
          ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
          Thought: I now know the final answer
          Final Answer: the final answer to the original input question
          Begin!
        '''
        self.tool_prompt_ids = self.tokenizer.encode(self.tool_prompt, add_special_tokens=False)
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])

    def preprocessing(self, item):
        input_ids = [self.tokenizer.get_command('[gMASK]'), self.tokenizer.get_command('sop')]
        label_ids = [self.ignore_index, self.ignore_index]
        input_ids += self.prompt_ids
        label_ids += [self.ignore_index] * len(self.self.prompt_ids)
        if "functions" in item["functions"]:
            input_ids += self.user_tokens
            label_ids = [self.ignore_index]
            for message in item["functions"]:
                fuction_content = json.dumps(message)
                function_ids += self.tokenizer.encode(fuction_content, add_special_tokens=False)
                input_ids += function_ids
                label_ids +=  [self.ignore_index] * len(self.function_ids)
            input_ids += self.tool_prompt_ids
            label_ids += [self.ignore_index] * len(self.self.tool_prompt_ids)
            

                 
           

            



