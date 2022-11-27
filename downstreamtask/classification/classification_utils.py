import os
import torch
import logging

from tqdm.auto import tqdm
from datasets import Dataset
from multiprocessing import Pool


logger = logging.getLogger(__name__)


def build_classification_dataset(data, tokenizer, args, mode, multi_label, output_mode, no_cache):
    # cached_features_file = os.path.join(
    #     args.cache_dir,
    #     "cached_{}_{}_{}_{}_{}".format(
    #         mode,
    #         args.model_type,
    #         args.max_seq_length,
    #         len(args.labels_list),
    #         len(data),
    #     ),
    # )

    # if os.path.exists(cached_features_file) and (
    #     (not args.reprocess_input_data and not args.no_cache)
    #     or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
    # ):
    #     data = torch.load(cached_features_file)
    #     logger.info(f" Features loaded from cache at {cached_features_file}")
    #     examples, labels = data
    # else:
    logger.info(" Converting to features started. Cache is not used.")

    if len(data) == 3:
        # Sentence pair task
        text_a, text_b, labels = data
    else:
        text_a, labels = data
        text_b = None

    # If labels_map is defined, then labels need to be replaced with ints
    if args.labels_map and not args.regression:
        if multi_label:
            labels = [[args.labels_map[l] for l in label] for label in labels]
        else:
            labels = [args.labels_map[label] for label in labels]

        # if (mode == "train" and args.use_multiprocessing) or (
        #     mode == "dev" and args.use_multiprocessing_for_evaluation
        # ):
        #     if args.multiprocessing_chunksize == -1:
        #         chunksize = max(len(data) // (args.process_count * 2), 500)
        #     else:
        #         chunksize = args.multiprocessing_chunksize

        #     if text_b is not None:
        #         data = [
        #             (
        #                 text_a[i : i + chunksize],
        #                 text_b[i : i + chunksize],
        #                 tokenizer,
        #                 args.max_seq_length,
        #             )
        #             for i in range(0, len(text_a), chunksize)
        #         ]
        #     else:
        #         data = [
        #             (text_a[i : i + chunksize], None, tokenizer, args.max_seq_length)
        #             for i in range(0, len(text_a), chunksize)
        #         ]

        #     with Pool(args.process_count) as p:
        #         examples = list(
        #             tqdm(
        #                 p.imap(preprocess_data_multiprocessing, data),
        #                 total=len(text_a),
        #                 disable=args.silent,
        #             )
        #         )

        examples = {key: torch.cat([example[key] for example in examples]) for key in examples[0]}
    else:
        examples = preprocess_data(text_a, text_b, labels, tokenizer, args.max_seq_length)

    if output_mode == "classification":
        labels = torch.tensor(labels, dtype=torch.long)
    elif output_mode == "regression":
        labels = torch.tensor(labels, dtype=torch.float)

    data = (examples, labels)

    # if not args.no_cache and not no_cache:
    #     logger.info(" Saving features into cached file %s", cached_features_file)
    #     torch.save(data, cached_features_file)

    return (examples, labels)


def preprocess_data(text_a, text_b, labels, tokenizer, max_seq_length):
    return tokenizer(
        text=text_a,
        text_pair=text_b,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt",
    )


def preprocess_data_multiprocessing(data):
    text_a, text_b, tokenizer, max_seq_length = data

    examples = tokenizer(
        text=text_a,
        text_pair=text_b,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    return examples


class ClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, args, mode, multi_label, output_mode, no_cache):
        self.examples, self.labels = build_classification_dataset(
            data, tokenizer, args, mode, multi_label, output_mode, no_cache
        )

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        return (
            {key: self.examples[key][index] for key in self.examples},
            self.labels[index],
        )


def load_hf_dataset(dataset, tokenizer, args, multi_label):
    if args.labels_map and not args.regression:
        dataset = dataset.map(lambda x: map_labels_to_numeric(x, multi_label, args))

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(x, tokenizer=tokenizer, max_seq_length=args.max_seq_length),
        batched=True,
    )

    if args.model_type in ["bert", "xlnet", "albert", "layoutlm", "layoutlmv2"]:
        dataset.set_format(
            type="pt",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
    else:
        dataset.set_format(type="pt", columns=["input_ids", "attention_mask", "label"])

    if isinstance(dataset, str):
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        return dataset["train"]
    else:
        return dataset


def preprocess_batch_for_hf_dataset(dataset, tokenizer, max_seq_length):
    if "text_b" in dataset:
        return tokenizer(
            text=dataset["text_a"],
            text_pair=dataset["text_b"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )
    else:
        return tokenizer(
            text=dataset["document"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )


def map_labels_to_numeric(example, multi_label, args):
    if multi_label:
        example["label"] = [args.labels_map[label] for label in example["label"]]
    else:
        example["label"] = args.labels_map[example["label"]]

    return example
