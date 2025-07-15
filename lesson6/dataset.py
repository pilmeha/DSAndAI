import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from tokenizers import Tokenizer

import os
from tqdm import tqdm

class WMTDataset(Dataset):
    def __init__(self, split='train', max_length=128, max_samples=None):
        self.src_lang = 'de'
        self.tgt_lang = 'en'

        self.dataset = load_dataset("wmt14", f'{self.src_lang}-{self.tgt_lang}')[split]
        self.tokenizer: Tokenizer = Tokenizer.from_file('C:\\Users\\garma\\DSAndAI\\lesson6\\tokenizer.json')
        self.tokenizer.add_special_tokens(['<pad>', '<s>', '</s>'])
        self.max_length = max_length

        if max_samples:
            
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

        def filter_function(sample):
            translation = sample['translation']
            if self.src_lang in translation and self.tgt_lang in translation:
                src_ids = self.tokenizer.encode(translation[self.src_lang]).ids
                tgt_ids = self.tokenizer.encode(translation[self.tgt_lang]).ids
                return (len(src_ids) + 2) <= self.max_length and (len(tgt_ids) + 2) <= self.max_length
            return False
        
        valid_indices = []
        for i in tqdm(range(len(self.dataset)), desc='Filtering'):
            if filter_function(self.dataset[i]):
                valid_indices.append(i)

        self.dataset = self.dataset.select(valid_indices)
        print(f"Dataset len: {len(self.dataset)}")

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()
    
    def get_pad_token_id(self):
        return self.tokenizer.token_to_id('<pad>')

    def get_bos_token_id(self):
        return self.tokenizer.token_to_id('<s>')

    def get_eos_token_id(self):
        return self.tokenizer.token_to_id('</s>')

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset[index]

        bos_token = self.get_bos_token_id()
        eos_token = self.get_eos_token_id()

        src_text = item['translation'][self.src_lang]
        tgt_text = item['translation'][self.tgt_lang]

        src_ids = [bos_token] + self.tokenizer.encode(src_text).ids + [eos_token]
        tgt_ids = [bos_token] + self.tokenizer.encode(tgt_text).ids + [eos_token]

        return {
            'src_ids': torch.Tensor(src_ids),
            'tgt_ids': torch.Tensor(tgt_ids)
        }
    
class Collator:
    def __init__(self, pad_token: int):
        self.pad_token = pad_token

    def __call__(self, batch: list):
        src_ids = [item['src_ids'] for item in batch]
        tgt_ids = [item['tgt_ids'] for item in batch]

        src_ids = pad_sequence(src_ids, batch_first=True, padding_value=self.pad_token)
        tgt_ids = pad_sequence(tgt_ids, batch_first=True, padding_value=self.pad_token)

        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids
        }
    
def create_dataloaders(
    batch_size = 32,
    max_length = 128,
    max_train_samples = None,
    max_val_samples = None,
    max_workers = 0
):
    
    train_dataset = WMTDataset(
        split = 'train',
        max_length=max_length,
        max_samples=max_train_samples
    )

    val_dataset = WMTDataset(
        split='validation',
        max_length=max_length,
        max_samples=max_val_samples
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=Collator(train_dataset.get_pad_token_id()),
        num_workers=max_workers
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=Collator(train_dataset.get_pad_token_id()),
        num_workers=max_workers
    )

    return train_dataset, train_dataloader, val_dataloader