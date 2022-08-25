
from generations.seq2seq.seq2seq_util import *
from generations.seq2seq.seq2seq_model import Seq2SeqModel
from generations.config.model_args import Seq2SeqTrainingArguments

data_path = "./dataset/data.txt"

raw = load_data(data_path)[:1000]
raw = split_data(raw, ratio=0.9)
dataset = load_hf_dataset(raw)

model_args = Seq2SeqTrainingArguments(
    output_dir='./outputs',
    generation_num_beams=1,
    generation_max_length=24,
    evaluation_strategy='steps',
    
    save_strategy="steps",
    save_steps=250,
    eval_steps=1000,

    per_device_train_batch_size=18,
    per_device_eval_batch_size=18,
    weight_decay=0.01,
    # save_total_limit=3,
    num_train_epochs=50,
    fp16=False,
    no_cuda=True,
    auto_find_batch_size=True,
    predict_with_generate=True,
    overwrite_output_dir = True,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False, 

    wandb_project = "RTE - Hyperparameter Optimization",
    wandb_kwargs = {"name": "vanilla"},
    # run_name="ut_del_three_per_each_ver2_early_stop_4"  
    # wandb_project='markview-test'

    save_best_model = True,
    best_model_dir='outputs/best_model',
    # output_dir=f"{model_type}-finetuned-{source_lang}-to-{target_lang}",
)

# set_logger(model_args)
# model = Seq2SeqModel("mt5", './literation/checkpoint-500', args=model_args)
# model = Seq2SeqModel("m2m", "./finetuned_model/m2m", args=model_args)
model = Seq2SeqModel("m2m", "facebook/m2m100_418M", args=model_args)

# Train the model
model.train_model(dataset['train'], eval_data=dataset['validation'])

# Optional: Evaluate the model. We'll test it properly anyway.
results = model.eval_model(dataset['validation'], verbose=True)
print(model.predict(["transformer", "markcloud"]))
