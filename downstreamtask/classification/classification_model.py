import os
import wandb
import torch
import logging
import warnings
import numpy as np

from dataclasses import asdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, RandomSampler

from downstreamtask.config.utils import sweep_config_to_sweep_values
from downstreamtask.config.model_args import ClassificationArgs
from downstreamtask.classification.classification_utils import load_hf_dataset, ClassificationDataset

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
        **kwargs,
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

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

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

        # self.tokenizer = tokenizer_class.from_pretrained(
        #     tokenizer_name, do_lower_case=self.args.do_lower_case, **kwargs
        # )

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

        self.args.model_name = model_name
        self.args.model_type = model_type

        if not use_cuda:
            self.args["fp16"] = False

        if self.args.wandb_project and not wandb_available:
            warnings.warn(
                "wandb_project specified but wandb is not available. Wandb disabled."
            )
            self.args.wandb_project = None

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

        if not output_dir:
            output_dir = self.args.output_dir

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set overwrite_output_dir: True to automatically overwrite.".format(output_dir)
            )

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

        if self.args.use_hf_datasets:
            train_dataset = load_hf_dataset(train_df, self.tokenizer, self.args, multi_label=multi_label)

        else:
            warnings.warn(
                "Dataframe headers not specified. Falling back to using column 0 as text and column 1 as labels."
            )
            train_examples = (
                train_df.iloc[:, 0].astype(str).tolist(),
                train_df.iloc[:, 1].tolist(),
            )
            train_dataset = self.load_and_cache_examples(train_examples, verbose=verbose)

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

        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=train_dataset,
            # eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        trainer.evaluate()

        self.save_model_args(args.best_model_dir, trainer, self.tokenizer)

        if verbose:
            logger.info(" Training of {} model complete. Saved to {}.".format(self.args.model_type, output_dir))

        return trainer

    def load_and_cache_examples(
        self,
        examples,
        evaluate=False,
        no_cache=False,
        multi_label=False,
        verbose=True,
        silent=False,
    ):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        args = self.args

        if not multi_label and args.regression:
            output_mode = "regression"
        else:
            output_mode = "classification"

        mode = "dev" if evaluate else "train"

        dataset = ClassificationDataset(
            examples,
            self.tokenizer,
            self.args,
            mode=mode,
            multi_label=multi_label,
            output_mode=output_mode,
            no_cache=no_cache,
        )

        return dataset

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
        model = self.model
        args = self.args

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = np.empty((len(to_predict), self.num_labels))

        if multi_label:
            out_label_ids = np.empty((len(to_predict), self.num_labels))
        else:
            out_label_ids = np.empty((len(to_predict)))

        if not multi_label and self.args.onnx:
            model_inputs = self.tokenizer.batch_encode_plus(
                to_predict, return_tensors="pt", padding=True, truncation=True
            )

            if self.args.model_type in [
                "bert",
                "xlnet",
                "albert",
                "layoutlm",
                "layoutlmv2",
            ]:
                for i, (input_ids, attention_mask, token_type_ids) in enumerate(
                    zip(
                        model_inputs["input_ids"],
                        model_inputs["attention_mask"],
                        model_inputs["token_type_ids"],
                    )
                ):
                    input_ids = input_ids.unsqueeze(0).detach().cpu().numpy()
                    attention_mask = attention_mask.unsqueeze(0).detach().cpu().numpy()
                    token_type_ids = token_type_ids.unsqueeze(0).detach().cpu().numpy()
                    inputs_onnx = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids,
                    }

                    # Run the model (None = get all the outputs)
                    output = self.model.run(None, inputs_onnx)

                    preds[i] = output[0]

            else:
                for i, (input_ids, attention_mask) in enumerate(
                    zip(model_inputs["input_ids"], model_inputs["attention_mask"])
                ):
                    input_ids = input_ids.unsqueeze(0).detach().cpu().numpy()
                    attention_mask = attention_mask.unsqueeze(0).detach().cpu().numpy()
                    inputs_onnx = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    }

                    # Run the model (None = get all the outputs)
                    output = self.model.run(None, inputs_onnx)

                    preds[i] = output[0]

            model_outputs = preds
            preds = np.argmax(preds, axis=1)

    def _move_model_to_device(self):
        self.model.to(self.device)

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
