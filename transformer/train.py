import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config

from pathlib import Path
from tqdm import tqdm


def get_sentences(dataset, lang):
    for item in dataset:
        yield item['translation'][lang]

def build_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_sentences(dataset, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def build_dataset(config):
    raw_dataset = load_dataset('opus_books', f'{config['src_lang']}-{config['tgt_lang']}', split = 'train')

    # Tokenizers for source and target language
    src_tokenizer = build_tokenizer(config, raw_dataset, config['src_lang'])
    tgt_tokenizer = build_tokenizer(config, raw_dataset, config['tgt_lang'])

    # Split the dataset to val
    train_dataset_size = int(0.9 * len(raw_dataset))
    val_dataset_size = len(raw_dataset) - train_dataset_size
    raw_train_dataset, raw_val_dataset = random_split(raw_dataset, [train_dataset_size, val_dataset_size])

    train_dataset = BilingualDataset(raw_train_dataset, src_tokenizer, config['src_lang'], config['src_seq_len'], tgt_tokenizer, config['tgt_lang'], config['tgt_seq_len'])
    val_dataset = BilingualDataset(raw_val_dataset, src_tokenizer, config['src_lang'], config['src_seq_len'], tgt_tokenizer, config['tgt_lang'], config['tgt_seq_len'])

    src_max_len, tgt_max_len = 0, 0
    for item in raw_dataset:
        src_ids = src_tokenizer.encode(item['translation'][config['src_lang']])
        tgt_ids = tgt_tokenizer.encode(item['translation'][config['tgt_lang']])

        src_max_len = max(src_max_len, len(src_ids))
        tgt_max_len = max(src_max_len, len(tgt_ids))

    print(f'Max length of source sentence: {src_max_len}')
    print(f'Max length of target sentence: {tgt_max_len}')

    train_dataloader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = config['batch_size'])
    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer

def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(src_vocab_size, tgt_vocab_size, config['src_seq_len'], config['tgt_seq_len'], config['d_model'])
    return model

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    # Create model folder to store the weights during training
    Path(config['model_folder']).mkdir(parents = True, exist_ok = True)

    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = build_dataset(config)
    
    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size())

    # TODO: Tensorboard

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        # TODO: Need to pick the latest epoch?
        model_filename = get_weights_file_path(config, '')
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # TODO: label_smoothing?
    loss_fn = nn.CrossEntropyLoss(ignore_index = src_tokenizer.token_to_id('[PAD]'), label_smoothing = 0.1)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f"Processing epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (batch_size, src_seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch_size, tgt_seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1,  src_seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, tgt_seq_len, tgt_seq_len)
            optimizer.zero_grad()

            # Call the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, src_seq_len, d_model)
            decoder_output = model.decode(decoder_input, decoder_mask, encoder_input, encoder_mask) # (batch_size, tgt_seq_len, d_model)
            predictions = model.project(decoder_output) # (batch_size, tgt_seq_len, tgt_vocab_size)

            target = batch['target'].to(device) # (batch_size, tgt_seq_len)

            # (batch_size, tgt_seq_len, tgt_vocab_size) -> (batch_size * tgt_seq_len, tgt_vocab_size)
            loss = loss_fn(predictions.view(-1, tgt_tokenizer.get_vocab_size()), target.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            loss.backward()
            optimizer.step()
            global_step += 1
        
        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

    return

if __name__ == "__main__":
    config = get_config()
    train_model(config)