from pathlib import Path

from transformers import PreTrainedTokenizerBase

from datasets import Value, load_dataset


def get_dataset(name, tokenizer, max_length, padding='max_length', truncation=True, dataset_root=None):
    if dataset_root is None:
        dataset_root = Path('datasets')
    else:
        assert isinstance(dataset_root, str, Path)
        dataset_root = Path(dataset_root)
    assert dataset_root.exists()

    if name == 'e2e':
        data_files = {
            split: str(dataset_root / 'e2e' / f'{split}.txt')
            for split in ['train', 'val', 'test']
        }
        dataset = load_dataset('text', data_files=data_files)
    elif name == 'roc':
        data_files = {
            split: str(dataset_root / 'roc' / f'{split}.json')
            for split in ['train', 'val']
        }
        dataset = load_dataset('text', data_files=data_files)
    elif name == 'sst':
        dataset = load_dataset('sst')
        dataset['val'] = dataset.pop('validation')
    elif name == 'ag_news':
        dataset = load_dataset('ag_news')
        train_val = dataset['train'].train_test_split(test_size=1000, seed=42)
        dataset['train'] = train_val['train']
        dataset['val'] = train_val['test']
    else:
        raise NotImplementedError(f'Dataset {name} not implemented')

    dataset = preprocess_dataset(dataset, name, tokenizer, max_length, padding, truncation)

    return dataset


def preprocess_dataset(dataset, name, tokenizer, max_length, padding, truncation):
    if name == 'e2e':
        def preprocess_e2e(example):
            meta, text = example['text'].split('||')
            text = PreTrainedTokenizerBase.clean_up_tokenization(text.strip())
            label = meta.split('|')
            new_example = {
                'text': text,
                'label': label,
            }
            return new_example

        dataset = dataset.map(
            preprocess_e2e,
            load_from_cache_file=False,
        )
    elif name == 'roc':
        def preprocess_roc(example):
            assert example['text'][:2] == '["'
            assert example['text'][-2:] == '"]'
            text = example['text'][2:-2]
            new_example = {
                'text': text,
            }
            return new_example

        dataset = dataset.map(
            preprocess_roc,
            load_from_cache_file=False,
        )

        # Hold out some validation samples for testing
        dataset = dataset.shuffle(seed=42)
        val_test = dataset['val'].train_test_split(train_size=1000, shuffle=False)
        dataset['val'] = val_test['train']
        dataset['test'] = val_test['test']
    elif name == 'sst':
        def preprocess_sst(example):
            text = PreTrainedTokenizerBase.clean_up_tokenization(example['sentence'].strip())
            label = 0 if example['label'] < 0.5 else 1
            new_example = {
                'text': text,
                'label': label,
            }
            return new_example

        dataset = dataset.map(
            preprocess_sst,
            load_from_cache_file=False,
            remove_columns=['sentence', 'tokens', 'tree'],
        ).cast_column('label', Value(dtype='int64', id=None))
    elif name == 'ag_news':
        def preprocess_ag_news(example):
            text = PreTrainedTokenizerBase.clean_up_tokenization(example['text'].strip())
            label = example['label']
            assert label in {0, 1, 2, 3}
            new_example = {
                'text': text,
                'label': label,
            }
            return new_example

        dataset = dataset.map(
            preprocess_ag_news,
            load_from_cache_file=False,
        )
    else:
        raise NotImplementedError(f'Dataset {name} not implemented')

    def tokenization(example):
        text = example['text']
        new_example = tokenizer(text, padding=padding, truncation=truncation, max_length=max_length)
        return new_example

    # Remain `text` column for evaluation
    dataset = dataset.map(
        tokenization,
        # remove_columns=['text'],
    )

    return dataset
