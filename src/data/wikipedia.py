"""
Wikipedia dataset loading, caching, tokenization and MLM collation.
"""

import os

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class WikipediaDatasetManager:
    """
    Load, split and tokenize the HuggingFace Wikipedia dataset for MLM.
    Results are cached to disk to avoid redundant downloads / processing.
    """

    def __init__(self,
                 dataset_name: str = "wikimedia/wikipedia",
                 dataset_config: str = "20231101.en",
                 cache_dir: str = "./data_cache"):
        self.dataset_name   = dataset_name
        self.dataset_config = dataset_config
        self.cache_dir      = cache_dir

        print("Initialisation du tokenizer BERT...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def load_and_split(self, train_size: int = 1_000_000,
                       val_size: int = 100_000, seed: int = 45) -> DatasetDict:
        cache_path = os.path.join(
            self.cache_dir, f"wiki_train{train_size}_val{val_size}_seed{seed}"
        )
        if os.path.exists(cache_path):
            print(f"✓ Chargement depuis le cache: {cache_path}")
            return load_from_disk(cache_path)

        print(f"Téléchargement du dataset {self.dataset_name}...")
        dataset = load_dataset(self.dataset_name, self.dataset_config, split='train')
        total   = train_size + val_size
        shuffled = dataset.shuffle(seed=seed).select(range(total))

        ds = DatasetDict({
            'train':      shuffled.select(range(train_size)),
            'validation': shuffled.select(range(train_size, total)),
        })
        os.makedirs(self.cache_dir, exist_ok=True)
        ds.save_to_disk(cache_path)
        print(f"✓ Train: {len(ds['train'])} | Val: {len(ds['validation'])}")
        return ds

    def create_dataloaders(self, dataset_dict: DatasetDict, batch_size: int = 256,
                           max_length: int = 128, mlm_probability: float = 0.15,
                           num_workers: int = 0):
        tokenized_cache = os.path.join(self.cache_dir, f"tokenized_maxlen{max_length}")

        if os.path.exists(tokenized_cache):
            print(f"✓ Chargement des tokens depuis: {tokenized_cache}")
            tok = load_from_disk(tokenized_cache)
            train_ds, val_ds = tok['train'], tok['validation']
        else:
            def tokenize_fn(examples):
                return self.tokenizer(
                    examples['text'], truncation=True,
                    padding='max_length', max_length=max_length,
                    return_tensors=None,
                )

            train_ds = dataset_dict['train'].map(
                tokenize_fn, batched=True,
                remove_columns=dataset_dict['train'].column_names,
                desc="Tokenizing train",
            )
            val_ds = dataset_dict['validation'].map(
                tokenize_fn, batched=True,
                remove_columns=dataset_dict['validation'].column_names,
                desc="Tokenizing val",
            )
            os.makedirs(self.cache_dir, exist_ok=True)
            DatasetDict({'train': train_ds, 'validation': val_ds}).save_to_disk(tokenized_cache)

        train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        val_ds.set_format(type='torch',   columns=['input_ids', 'attention_mask'])

        collator = MLMCollator(self.tokenizer, mlm_probability)
        pin = torch.cuda.is_available()

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=collator, num_workers=num_workers, pin_memory=pin)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                  collate_fn=collator, num_workers=num_workers, pin_memory=pin)

        print(f"✓ {len(train_loader)} train batches | {len(val_loader)} val batches")
        return train_loader, val_loader


class MLMCollator:
    """
    Dynamic MLM masking following the original BERT 80/10/10 strategy.
    Applied on-the-fly so each epoch sees different masks.
    """

    def __init__(self, tokenizer: BertTokenizer, mlm_probability: float = 0.15):
        self.tokenizer       = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_id   = tokenizer.mask_token_id
        self.pad_token_id    = tokenizer.pad_token_id
        self.cls_token_id    = tokenizer.cls_token_id
        self.sep_token_id    = tokenizer.sep_token_id
        self.vocab_size      = tokenizer.vocab_size

    def __call__(self, batch):
        input_ids      = torch.stack([item['input_ids']      for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels         = input_ids.clone()
        input_ids, labels = self._mask_tokens(input_ids, labels)
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}

    def _mask_tokens(self, input_ids: torch.Tensor, labels: torch.Tensor):
        prob_matrix = torch.full(input_ids.shape, self.mlm_probability)
        special = (
            (input_ids == self.pad_token_id) |
            (input_ids == self.cls_token_id) |
            (input_ids == self.sep_token_id)
        )
        prob_matrix.masked_fill_(special, 0.0)
        masked = torch.bernoulli(prob_matrix).bool()
        labels[~masked] = -100

        # 80 % → [MASK]
        replace = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked
        input_ids[replace] = self.mask_token_id

        # 10 % → random token
        randomize = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked & ~replace
        )
        input_ids[randomize] = torch.randint(self.vocab_size, input_ids.shape)[randomize]

        return input_ids, labels
