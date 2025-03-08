import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, hugging_face_dataset, src_tokenizer, src_lang, src_seq_len, tgt_tokenizer, tgt_lang, tgt_seq_len) -> None:
        super().__init__()

        self.dataset = hugging_face_dataset
        self.src_tokenizer = src_tokenizer
        self.src_lang = src_lang
        self.src_seq_len = src_seq_len
        
        self.tgt_tokenizer = tgt_tokenizer
        self.tgt_lang = tgt_lang
        self.tgt_seq_len = tgt_seq_len

        self.src_sos_token = torch.tensor([src_tokenizer.token_to_id('[SOS]')], dtype = torch.int64)
        self.src_eos_token = torch.tensor([src_tokenizer.token_to_id('[EOS]')], dtype = torch.int64)
        self.src_pad_token = torch.tensor([src_tokenizer.token_to_id('[PAD]')], dtype = torch.int64)
        
        self.tgt_sos_token = torch.tensor([tgt_tokenizer.token_to_id('[SOS]')], dtype = torch.int64)
        self.tgt_eos_token = torch.tensor([tgt_tokenizer.token_to_id('[EOS]')], dtype = torch.int64)
        self.tgt_pad_token = torch.tensor([tgt_tokenizer.token_to_id('[PAD]')], dtype = torch.int64)
        return
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        src_target_pair = self.dataset[idx]
        src_txt = src_target_pair['translation'][self.src_lang]
        tgt_txt = src_target_pair['translation'][self.tgt_lang]

        src_input_tokens = self.src_tokenizer.encode(src_txt).ids
        tgt_input_tokens = self.src_tokenizer.encode(tgt_txt).ids

        src_num_padding_tokens = self.src_seq_len - len(src_input_tokens) - 2
        tgt_num_padding_tokens = self.tgt_seq_len - len(tgt_input_tokens) - 1

        if src_num_padding_tokens < 0 or tgt_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
    
        # Building the encoder input -> <SOS> {... INPUT TOKENS} <EOS> {...<PAD> tokens if needed}
        encoder_input = torch.cat(
            [
                self.src_sos_token,
                torch.tensor(src_input_tokens, dtype = torch.int64),
                self.src_eos_token,
                torch.tensor([self.src_pad_token] * src_num_padding_tokens, dtype = torch.int64)
            ]
        )

        # Building the decoder input -> <SOS> {... INPUT TOKENS} {...<PAD> tokens if needed}
        decoder_input = torch.cat(
            [
                self.tgt_sos_token,
                torch.tensor(tgt_input_tokens, dtype = torch.int64),
                torch.tensor([self.tgt_pad_token] * tgt_num_padding_tokens, dtype = torch.int64)
            ]
        )

        # Building the target -> {... INPUT TOKENS} <EOS> {...<PAD> tokens if needed}
        target = torch.cat(
            [
                torch.tensor(tgt_input_tokens, dtype = torch.int64),
                self.tgt_eos_token,
                torch.tensor([self.tgt_pad_token] * tgt_num_padding_tokens, dtype = torch.int64)
            ]
        )

        assert encoder_input.shape[0] == self.src_seq_len
        assert decoder_input.shape[0] == self.tgt_seq_len
        assert target.shape[0] == self.tgt_seq_len

        # Need to get clarity on the shapes of these tensors
        return {
            "encoder_input": encoder_input, # (src_seq_len)
            "encoder_mask": (encoder_input != self.src_pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, src_seq_len)
            "decoder_input": decoder_input, # (tgt_seq_len)
            "decoder_mask": (decoder_input != self.tgt_pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # Didn't understand this?
            "target": target,
            "src_txt": src_txt,
            "tgt_txt": tgt_txt
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0



