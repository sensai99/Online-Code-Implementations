from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 1,
        "lr": 1e-4,
        "d_model": 512,
        "src_seq_len": 350,
        "tgt_seq_len": 350,
        "src_lang": "en",
        "tgt_lang": "it",
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/model"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_filename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def get_latest_weights_file_path(config):
    model_folder = config['model_folder']
    model_filename = config['model_basename']
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])