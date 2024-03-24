# -*- coding: utf-8 -*-
import os
import json
import argparse

def process_chat(content):
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

def process_encode(content):
    content_json = json.loads(content)
    #system_message = {"role": "system", "content": content_json.get("instruction", "")}
    instruction = content_json.get("instruction", "")
    query = instruction
    content = "" 
    for instance in content_json.get("instances", []):
        content = instance.get("input", "")
        user_content = instruction + "," + content
        if instruction == "":
            user_content = content
        if content == "":
            user_content = instruction
        content = instance.get("output", "")
        break
    messages = {"query": query, "pos": [content]}
    return messages

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help="convert file")
    parser.add_argument('--output_path', default="train_message.json", type=str, help="save path")
    parser.add_argument('--type', default="chat", type=str, help="data type")
    args = parser.parse_args()
    if args.type == "emb":
        fw = open(args.output_path, 'w', encoding='utf-8') 
        with open(args.input_file , 'r') as fp:
            for line in fp:
                message = process_encode(line)
                print(message)
                fw.write(json.dumps(message, ensure_ascii=False))
                fw.write("\n")
        fw.close()
    else:
        messages = []
        with open(args.input_file , 'r') as fp:
            for line in fp:
                message = process_chat(line)
                print(message)
                messages.append(message)
        with open(args.output_path, 'w', encoding='utf-8') as fw:
            json.dump(messages, fw, ensure_ascii=False)
            
