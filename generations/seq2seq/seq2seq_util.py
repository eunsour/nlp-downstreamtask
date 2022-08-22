import random
import logging
import numpy as np

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, is_torch_available

logger = logging.getLogger(__name__) # pylint: disable=invalid-name

# pretrained_model_name = "google/mt5-base"
pretrained_model_name = "facebook/m2m100_418M"

source_lang = "en"
target_lang = "ko"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name,
    do_lower_case=True,
    src_lang = source_lang,
    tgt_lang = target_lang
)

model_checkpoint = pretrained_model_name.split("/")[-1]

if model_checkpoint in ["mt5-small", "mt5-base", "mt5-larg", "mt5-3b", "mt5-11b"]:
    prefix = "translate English to Korean: "
else:
    prefix = ""


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
    dic = {
        "transliteration": []
        }

    for idx in range(len(raw)):
        src = raw[idx].split("\t")[0]
        tgt = raw[idx].split("\t")[-1]
        dic["transliteration"].append(
            {
                source_lang: src, 
                target_lang: tgt
            }
        )

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

    raw = DatasetDict(
        {
            "train": train_dataset, 
            "validation": validation_dataset
        }
    )

    return raw

def preprocess_batch_for_hf_dataset(dataset, src_lang, tgt_lang):
    inputs = [prefix + data[src_lang] for data in dataset["transliteration"]]
    targets = [data[tgt_lang] for data in dataset["transliteration"]]
    model_inputs = tokenizer(inputs, max_length=32, truncation=True)

    # Setup the TOKENIZER for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=32, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_hf_dataset(dataset):
    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(
            x,
            source_lang,
            target_lang
        ),
        batched=True,
    )
    
    return dataset


def set_logger(args):
    import torch
    if torch.cuda.is_available():
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(levelname)s:%(name)s:%(message)s",
        )
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    logger.info("Training/evaluation parameters %s", args)

