import pandas as pd
import pytest

from downstreamtask.classification import ClassificationModel


@pytest.mark.parametrize(
    "model_type, model_name",
    [
        ("bert", "bert-base-uncased"),
        ("bigbird", "google/bigbird-roberta-base"),
        # ("longformer", "allenai/longformer-base-4096"),
        # ("electra", "google/electra-small-discriminator"),
        # ("mobilebert", "google/mobilebert-uncased"),
        # ("bertweet", "vinai/bertweet-base"),
        # ("deberta", "microsoft/deberta-base"),
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
def test_binary_classification(model_type, model_name):
    return


@pytest.mark.parametrize(
    "model_type, model_name",
    [
        # ("bert", "bert-base-uncased"),
        # ("xlnet", "xlnet-base-cased"),
        ("bigbird", "google/bigbird-roberta-base"),
        # ("xlm", "xlm-mlm-17-1280"),
        ("roberta", "roberta-base"),
        # ("distilbert", "distilbert-base-uncased"),
        # ("albert", "albert-base-v1"),
        # ("camembert", "camembert-base"),
        # ("xlmroberta", "xlm-roberta-base"),
        # ("flaubert", "flaubert-base-cased"),
    ],
)
def test_multiclass_classification(model_type, model_name):
    return
