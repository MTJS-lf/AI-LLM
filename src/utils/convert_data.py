# -*- coding: utf-8 -*-

import os
import sys
import json
import random

def process(raw_data):
    data_dict = json.loads(raw_data)
    id = data_dict["kind"]
    conversations = []
    use_message = {"from": "user", "content": data_dict["input"]}
    assistant_message = {"from": "assistant", "content": data_dict["target"]}
    conversations.append(use_message)
    conversations.append(assistant_message)
    messages = {"id": id, "conversations": conversations}
    return messages
        
if __name__ == "__main__":
    input_file = sys.argv[1]
    output_train_file = sys.argv[2]
    output_test_file = sys.argv[3]
    output_train_fp = open(output_train_file, 'w+')
    output_test_fp = open(output_test_file, 'w+')
    with open(input_file, 'r') as fp:
        for line in fp:
            messages = process(line.strip("\n"))
            if random.random() < 0.001:
                output_test_fp.write(json.dumps(messages, ensure_ascii=False)) 
                output_test_fp.write("\n")
            else:
                output_train_fp.write(json.dumps(messages, ensure_ascii=False)) 
                output_train_fp.write("\n")
    output_train_fp.close()
    output_test_fp.close()
      
