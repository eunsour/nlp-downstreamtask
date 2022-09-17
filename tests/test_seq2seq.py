import pandas as pd
import pytest

from generations.seq2seq.seq2seq_model import Seq2SeqModel


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


def test_t5(model_type, model_name):
    train_data = [
        ["transliterate English to Korean", "transformer", "트랜스포머"],
        ["transliterate English to Korean", "huggingface", "허깅페이스"],
    ]

    train_df = pd.DataFrame(train_data, columns=["prefix", "input_text", "target_text"])

    eval_data = [
        ["transliterate English to Korean", "three", "쓰리"],
        ["transliterate English to Korean", "four", "포"],
    ]

    eval_df = pd.DataFrame(eval_data, columns=["prefix", "input_text", "target_text"])

    eval_df = train_df.copy()

    model_args = {
        "output_dir":'./outputs',
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": 10,
        "train_batch_size": 2,
        "num_train_epochs": 2,
        "save_model_every_epoch": False,
        "max_length": 20,
        "num_beams": 1,
        "no_cuda": True
    }

    # Create T5 Model
    model = Seq2SeqModel(model_type, model_name, args=model_args)

    # Train T5 Model on new task
    model.train_model(train_df)

    # Evaluate T5 Model on new task
    model.eval_model(eval_df)

    # Predict with trained T5 model
    model.predict(["convert: four", "convert: five"])

    # Load test
    model = Seq2SeqModel("t5", "outputs", args=model_args)

    # Evaluate T5 Model on new task
    model.eval_model(eval_df)

    # Predict with trained T5 model
    model.predict(["convert: four", "convert: five"])

if __name__ == "__main__":
    test_t5()
