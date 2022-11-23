import os
import wandb
import torch
import logging

from dataclasses import asdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, RandomSampler
from downstreamtask.config.model_args import ClassificationArgs
from downstreamtask.classification.classification_utils import (
    load_hf_dataset,
)

from transformers import (
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    ElectraConfig,
    ElectraForSequenceClassification,
    ElectraTokenizerFast,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    Trainer,
)

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)


class ClassificationModel:
    def __init__(
        self,
        model_type,
        model_name,
        num_labels=None,
        weight=None,
        # sliding_window=False,
        args=None,
        use_cuda=True,
    ):

        MODEL_CLASSES = {
            "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
            "bert": (BertConfig, BertForSequenceClassification, BertTokenizerFast),
            "electra": (
                ElectraConfig,
                ElectraForSequenceClassification,
                ElectraTokenizerFast,
            ),
            "roberta": (
                RobertaConfig,
                RobertaForSequenceClassification,
                RobertaTokenizerFast,
            ),
        }

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, ClassificationArgs):
            self.args = args

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

        if num_labels:
            self.config = config_class.from_pretrained(model_name, num_labels=num_labels)
            self.num_labels = num_labels
        else:
            self.config = config_class.from_pretrained(model_name)
            self.num_labels = self.config.num_labels

        self.tokenizer = tokenizer_class.from_pretrained(model_name)
        self.num_labels = num_labels
        self.weight = weight
        # self.sliding_window = sliding_window

        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable. Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        if self.weight:
            self.model = model_class.from_pretrained(
                model_name,
                config=self.config,
                weight=torch.Tensor(self.weight).to(self.device),
                # sliding_window=self.sliding_window,
            )
        else:
            self.model = model_class.from_pretrained(model_name, config=self.config)

        if not use_cuda:
            self.args["fp16"] = False

    def train_model(
        self,
        train_df,
        multi_label=False,
        output_dir=None,
        # show_running_loss=True,
        args=None,
        eval_df=None,
        verbose=True,
        **kwargs,
    ):

        if args:
            self.args.update_from_dict(args)
        else:
            args = self.args

        train_dataset = load_hf_dataset(train_df, self.tokenizer, self.args, multi_label=multi_label)
        eval_dataset = load_hf_dataset(eval_df, self.tokenizer, self.args, multi_label=multi_label)

        # train_dataset = RandomSampler(train_dataset)
        # eval_dataset = RandomSampler(eval_dataset)

        # train_dataloader = DataLoader(
        #     train_dataset,
        #     sampler=train_sampler,
        #     batch_size=self.args.per_device_train_batch_size,
        #     num_workers=self.args.dataloader_num_workers,
        # )

        # os.makedirs(output_dir, exist_ok=True)

        # global_step, training_details = self.train(
        #     train_dataloader,
        #     output_dir,
        #     multi_label=multi_label,
        #     # show_running_loss=show_running_loss,
        #     eval_df=eval_df,
        #     verbose=verbose,
        #     **kwargs,
        # )

        if args.wandb_project:
            if not wandb.setup().settings.sweep_id:
                logger.info(" Initializing WandB run for training.")
                wandb.init(
                    project=args.wandb_project,
                    config={**asdict(args)},
                    **args.wandb_kwargs,
                )
                wandb.run._label(repo="downstreamtask")
                self.wandb_run_id = wandb.run.id
            wandb.watch(self.model)

        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        trainer.evaluate()

        self.save_model_args(args.best_model_dir, trainer, self.tokenizer)

        if verbose:
            logger.info(" Training of {} model complete. Saved to {}.".format(self.args.model_type, output_dir))

    def save_model_args(self, best_model_dir, trainer, tokenizer):
        os.makedirs(best_model_dir, exist_ok=True)
        trainer.save_model(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)

    def eval_model(self, eval_data, output_dir=None, verbose=True, silent=False, **kwargs):
        model = self.model
        # self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=model)

        eval_data = load_hf_dataset(eval_data, self.tokenizer, self.args, multi_label=False)

        self.trainer = Trainer(
            model,
            self.args,
            tokenizer=self.tokenizer,
            # data_collator=self.data_collator,
            eval_dataset=eval_data,
        )

        self.trainer.evaluate()

    def evaluate():
        ...

    def tune_model():
        ...

    def predict(self, to_predict, multi_label=False):
        ...

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)

        return {"Accuracy": acc, "F1": f1, "Precision": precision, "Recall": recall}

    def _load_model_args(self, input_dir):
        args = ClassificationArgs()
        args.load(input_dir)
        return args
