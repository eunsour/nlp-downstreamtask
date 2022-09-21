import pandas as pd
import pytest
from generations.config.model_args import Seq2SeqTrainingArguments

from generations.seq2seq import Seq2SeqModel


@pytest.mark.parametrize(
    "model_type, model_name",
    [
        ("mt5", "google/mt5-base"),
        ("m2m", "facebook/m2m100_418M"),
        # ("reformer", "google/reformer-crime-and-punishment"),
        # ("xlnet", "xlnet-base-cased"),
        # ("xlm", "xlm-mlm-17-1280"),
        # ("roberta", "roberta-base"),
        # ("distilbert", "distilbert-base-uncased"),
        # ("albert", "albert-base-v1"),
        # ("camembert", "camembert-base"),
        # ("xlmroberta", "xlm-roberta-base"),
        # ("flaubert", "flaubert-base-cased"),
    ],
)

def test_seq2seq(model_type, model_name):
    train_data = [
        ["transliterate English to Korean", "transformer", "트랜스포머"],
        ["transliterate English to Korean", "huggingface", "허깅페이스"],
    ]

    train_df = pd.DataFrame(train_data, columns=["prefix", "input_text", "target_text"])

    eval_data = [
        ["transliterate English to Korean", "Factory", "팩토리"],
        ["transliterate English to Korean", "Pytorch", "파이토치"],
    ]

    eval_df = pd.DataFrame(eval_data, columns=["prefix", "input_text", "target_text"])

    model_args = Seq2SeqTrainingArguments(
        output_dir="./outputs",
        generation_num_beams=1,
        generation_max_length=24,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        num_train_epochs=3,
        no_cuda=False,
        auto_find_batch_size=True,
    )
    
    # Create Seq2Seq Model
    model = Seq2SeqModel(model_type, model_name, args=model_args)

    # Train Seq2Seq Model on new task
    model.train_model(train_df)

    # Evaluate Seq2Seq Model on new task
    model.eval_model(eval_df)

    # Predict with trained Seq2Seq model
    print(model.predict(["machine learning", "deep learning"]))

    # Load test
    model = Seq2SeqModel(model_type, model_name, args=model_args)

    # Evaluate Seq2Seq Model on new task
    model.eval_model(eval_df)

    # Predict with trained Seq2Seq model
    print(model.predict(["machine learning", "deep learning"]))
