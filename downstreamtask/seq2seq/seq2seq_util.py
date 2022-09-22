import random
import logging
import numpy as np

from datasets import Dataset, DatasetDict
from transformers import is_torch_available

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def set_seed(seed: int = 42):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, and/or ``torch``
    (if installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8-sig") as rf:
        for line in rf:
            line = line.strip()
            data.append(line)
    return data


def make_dataset(raw):
    dic = {"transliteration": []}

    for idx in range(len(raw)):
        src = raw[idx].split("\t")[0]
        tgt = raw[idx].split("\t")[-1]
        # dic["transliteration"].append({args.src_lang: src, args.tgt_lang: tgt})
        dic["transliteration"].append({"en": src, "ko": tgt})
    dataset = Dataset.from_dict(dic)

    return dataset


def split_data(data_list, ratio):
    """ratio: 0 ~ 1"""
    set_seed(42)
    random.shuffle(data_list)
    pivot = int(len(data_list) * ratio)
    ldata = data_list[:pivot]
    rdata = data_list[pivot:]

    train_dataset = make_dataset(ldata)
    validation_dataset = make_dataset(rdata)

    raw = DatasetDict({"train": train_dataset, "validation": validation_dataset})

    return raw


def preprocess_batch_for_hf_dataset(dataset, tokenizer, args):
    tokenizer.src_lang = args.src_lang
    tokenizer.tgt_lang = args.tgt_lang
    
    if tokenizer in ["mt5-small", "mt5-base", "mt5-large", "mt5-3b", "mt5-11b"]:
        prefix = "transliterate English to Korean: "
    else:
        prefix = ""

    inputs = [prefix + data[args.src_lang] for data in dataset["transliteration"]]
    targets = [data[args.tgt_lang] for data in dataset["transliteration"]]
    model_inputs = tokenizer(
        inputs,
        max_length=args.max_seq_length,
        truncation=True,
    )

    # Setup the TOKENIZER for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=args.max_seq_length,
            truncation=True,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_hf_dataset(data, tokenizer, args):
    dataset = data.map(
        lambda x: preprocess_batch_for_hf_dataset(x, tokenizer, args),
        batched=True,
    )

    return dataset
