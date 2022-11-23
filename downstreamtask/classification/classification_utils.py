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
