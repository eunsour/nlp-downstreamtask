import pytest

from datasets import load_dataset

from downstreamtask.classification import ClassificationModel
from downstreamtask.config.model_args import ClassificationArgs


@pytest.mark.parametrize(
    "model_type, model_name",
    [
        ("bert", "bert-base-uncased"),
        ("albert", "albert-base-v2"),
        ("roberta", "roberta-base"),
        ("electra", "kykim/electra-kor-base")
    ],
)
def test_sentiment_analysis(model_type, model_name):

    nsmc_train = load_dataset("nsmc", split="train[:20]+train[-20:]")
    nsmc_test = load_dataset("nsmc", split="test[:5]+test[-5:]")
    nsmc_val = load_dataset("nsmc", split="test[5:10]+test[-10:-50]")

    training_args = ClassificationArgs(
        do_eval=True,
        do_train=True,
        evaluation_strategy="epoch",
        fp16=False,
        # load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=50,
        logging_strategy="steps",
        num_train_epochs=1,
        output_dir=f"outputs/{model_name}-finetuned-sentiment-analysis",
        per_device_eval_batch_size=16,
        per_device_train_batch_size=16,
        report_to="wandb",
        save_strategy="epoch",
        warmup_steps=500,
        weight_decay=0.01,
        wandb_project="sentiment-analysis",
        wandb_kwargs={"name": f"{model_name}-finetuned"},
        seed=42,
    )

    # Create a ClassificationModel
    model = ClassificationModel(model_type, model_name, args=training_args, num_labels=2)

    # Train the model
    model.train_model(train_df=nsmc_train, eval_df=nsmc_val)

    # # Evaluate the model
    model.eval_model(nsmc_test)

    # predictions, raw_outputs = model.predict(["이도 저도 아닌 영화입니다."])
