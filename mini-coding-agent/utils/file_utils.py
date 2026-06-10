import json

def save_jsonl(context, file_path):
    with open(file_path, "a") as f:
        for item in context:
            f.write(json.dumps(item) + "\n")

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f.readlines()]