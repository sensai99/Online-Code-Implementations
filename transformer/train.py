from torch.utils.data import random_split, DataLoader

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset import BilingualDataset, causal_mask

from pathlib import Path


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
    src_tokenizer = build_dataset(config, raw_dataset, config['src_lang'])
    tgt_tokenizer = build_dataset(config, raw_dataset, config['tgt_lang'])

    # Split the dataset to val
    train_dataset_size = int(0.9 * len(raw_dataset))
    val_dataset_size = len(raw_dataset) - train_dataset_size
    raw_train_dataset, raw_val_dataset = random_split(raw_dataset, [train_dataset_size, val_dataset_size])

    train_dataset = BilingualDataset(raw_train_dataset, src_tokenizer, tgt_tokenizer, config['src_lang'], config['tgt_lang'], config['src_seq_len'])
    val_dataset = BilingualDataset(raw_val_dataset, src_tokenizer, tgt_tokenizer, config['src_lang'], config['tgt_lang'], config['src_seq_len'])

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