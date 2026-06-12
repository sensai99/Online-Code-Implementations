import os
import json

def set_env_vars(file_path):
    dict_env_vars = {}
    with open(file_path, "r") as f:
        for line in f.readlines():
            key, value = line.strip().split("=")
            dict_env_vars[key] = value
    return dict_env_vars

def save_jsonl(context, file_path):
    with open(file_path, "a") as f:
        for item in context:
            f.write(json.dumps(item) + "\n")

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f.readlines()]