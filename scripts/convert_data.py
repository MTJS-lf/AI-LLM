# -*- coding: utf-8 -*-
import os
import json
import argparse

def process(content):
    content_json = json.loads(content)
    #system_message = {"role": "system", "content": content_json.get("instruction", "")}
    conv_messages = []
    instruction = content_json.get("instruction", "")
    for instance in content_json.get("instances", []):
        content = instance.get("input", "")
        user_content = instruction + "," + content
        if instruction == "":
            user_content = content
        if content == "":
            user_content = instruction
        user_message = {"role": "user", "content": user_content}
        assistant_message = {"role": "assistant", "content": instance.get("output", "")}
        conv_messages.append(user_message)
        conv_messages.append(assistant_message)
    messages = {"conversations": conv_messages}
    return messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help="convert file")
    parser.add_argument('--output_path', default="train_message.json", type=str, help="save path")
    args = parser.parse_args()
    messages = []
    with open(args.input_file , 'r') as fp:
        for line in fp:
            message = process(line)
            print(message)
            messages.append(message)
    with open(args.output_path, 'w', encoding='utf-8') as fw:
        json.dump(messages, fw, ensure_ascii=False)
        
