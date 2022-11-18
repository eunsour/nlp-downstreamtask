import pytest

from downstreamtask.seq2seq import Seq2SeqModel
from downstreamtask.seq2seq.seq2seq_util import split_data
from downstreamtask.config.model_args import Seq2SeqTrainingArguments


@pytest.mark.parametrize(
    "model_type, model_name",
    [
        ("mt5", "google/mt5-base"),
        ("m2m", "facebook/m2m100_418M"),
        ("mbart", "facebook/mbart-large-50"),
    ],
)
def test_seq2seq(model_type, model_name):
    data = [
        "micwhite\t마이화이트",
        "hydralight\t하이드라라이트",
        "characdoor\t케락도어",
        "acevan\t에이스밴",
        "worldtec\t월드텍",
        "unievall\t유니발",
        "psddr\t피에스디알",
        "padishah\t파디샤",
        "unlook\t언룩",
        "keratherm\t케라섬",
    ]

    raw = split_data(data, ratio=0.8)

    model_args = Seq2SeqTrainingArguments(
        auto_find_batch_size=True,
        generation_max_length=24,
        no_cuda=True,
        num_train_epochs=1,
        overwrite_output_dir=True,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
    )

    # Create Seq2Seq Model
    model = Seq2SeqModel(model_type, model_name, args=model_args)

    # Train Seq2Seq Model on new task
    model.train_model(raw)

    # Evaluate Seq2Seq Model on new task
    model.eval_model(raw)

    # Predict with trained Seq2Seq model
    print(model.predict(["machine learning", "deep learning"]))

    # Load test
    model = Seq2SeqModel(model_type, model_name, args=model_args)

    # Evaluate Seq2Seq Model on new task
    model.eval_model(raw)

    # Predict with trained Seq2Seq model
    print(model.predict(["machine learning", "deep learning"]))
