from __future__ import absolute_import, division, print_function
import collections
import logging
import math
import os
import random
import warnings
from dataclasses import asdict
from multiprocessing import cpu_count
import tempfile
from pathlib import Path

from collections import Counter
import numpy as np
import pandas as pd
import torch
from scipy.stats import mode, pearsonr
from scipy.special import softmax
from sklearn.metrics import (
    confusion_matrix,
    label_ranking_average_precision_score,
    matthews_corrcoef,
    mean_squared_error,
    roc_curve,
    auc,
    average_precision_score,
)
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from tqdm.contrib import tenumerate
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from torch.optim import AdamW
from transformers.optimization import Adafactor
from transformers import (
    AlbertConfig,
    AlbertTokenizer,
    AlbertForSequenceClassification,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertTokenizerFast,
    BertForSequenceClassification,
    BertweetTokenizer,
    BigBirdConfig,
    BigBirdTokenizer,
    BigBirdForSequenceClassification,
    CamembertConfig,
    CamembertTokenizerFast,
    CamembertForSequenceClassification,
    DebertaConfig,
    DebertaForSequenceClassification,
    DebertaTokenizer,
    DebertaV2Config,
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer,
    DistilBertConfig,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    ElectraConfig,
    ElectraTokenizerFast,
    ElectraForSequenceClassification,
    FlaubertConfig,
    FlaubertTokenizer,
    FlaubertForSequenceClassification,
    HerbertTokenizerFast,
    LayoutLMConfig,
    LayoutLMTokenizerFast,
    LayoutLMForSequenceClassification,
    LayoutLMv2Config,
    LayoutLMv2TokenizerFast,
    LayoutLMv2ForSequenceClassification,
    LongformerConfig,
    LongformerTokenizerFast,
    LongformerForSequenceClassification,
    MPNetConfig,
    MPNetForSequenceClassification,
    MPNetTokenizerFast,
    MobileBertConfig,
    MobileBertTokenizerFast,
    MobileBertForSequenceClassification,
    NystromformerConfig,
    # NystromformerTokenizer,
    NystromformerForSequenceClassification,
    RemBertConfig,
    RemBertTokenizerFast,
    RemBertForSequenceClassification,
    RobertaConfig,
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    SqueezeBertConfig,
    SqueezeBertForSequenceClassification,
    SqueezeBertTokenizerFast,
    WEIGHTS_NAME,
    XLMConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizerFast,
    XLMRobertaForSequenceClassification,
    XLMTokenizer,
    XLMForSequenceClassification,
    XLNetConfig,
    XLNetTokenizerFast,
    XLNetForSequenceClassification,
)
from transformers.convert_graph_to_onnx import convert, quantize

from downstreamtask.config.model_args import ClassificationArgs
from downstreamtask.config.utils import sweep_config_to_sweep_values
from downstreamtask.classification.classification_utils import *

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "auto": (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer),
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizerFast),
    "bertweet": (
        RobertaConfig,
        RobertaForSequenceClassification,
        BertweetTokenizer,
    ),
    "bigbird": (
        BigBirdConfig,
        BigBirdForSequenceClassification,
        BigBirdTokenizer,
    ),
    "camembert": (
        CamembertConfig,
        CamembertForSequenceClassification,
        CamembertTokenizerFast,
    ),
    "deberta": (
        DebertaConfig,
        DebertaForSequenceClassification,
        DebertaTokenizer,
    ),
    "debertav2": (
        DebertaV2Config,
        DebertaV2ForSequenceClassification,
        DebertaV2Tokenizer,
    ),
    "distilbert": (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    ),
    "electra": (
        ElectraConfig,
        ElectraForSequenceClassification,
        ElectraTokenizerFast,
    ),
    "flaubert": (
        FlaubertConfig,
        FlaubertForSequenceClassification,
        FlaubertTokenizer,
    ),
    "herbert": (
        BertConfig,
        BertForSequenceClassification,
        HerbertTokenizerFast,
    ),
    "layoutlm": (
        LayoutLMConfig,
        LayoutLMForSequenceClassification,
        LayoutLMTokenizerFast,
    ),
    "layoutlmv2": (
        LayoutLMv2Config,
        LayoutLMv2ForSequenceClassification,
        LayoutLMv2TokenizerFast,
    ),
    "longformer": (
        LongformerConfig,
        LongformerForSequenceClassification,
        LongformerTokenizerFast,
    ),
    "mobilebert": (
        MobileBertConfig,
        MobileBertForSequenceClassification,
        MobileBertTokenizerFast,
    ),
    "mpnet": (MPNetConfig, MPNetForSequenceClassification, MPNetTokenizerFast),
    "nystromformer": (
        NystromformerConfig,
        NystromformerForSequenceClassification,
        BigBirdTokenizer,
    ),
    "rembert": (
        RemBertConfig,
        RemBertForSequenceClassification,
        RemBertTokenizerFast,
    ),
    "roberta": (
        RobertaConfig,
        RobertaForSequenceClassification,
        RobertaTokenizerFast,
    ),
    "squeezebert": (
        SqueezeBertConfig,
        SqueezeBertForSequenceClassification,
        SqueezeBertTokenizerFast,
    ),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "xlmroberta": (
        XLMRobertaConfig,
        XLMRobertaForSequenceClassification,
        XLMRobertaTokenizerFast,
    ),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizerFast),
}


class ClassificationModel:
    def __init__(
        self,
        model_type,
        model_name,
        tokenizer_type=None,
        tokenizer_name=None,
        num_labels=None,
        weight=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        onnx_execution_provider=None,
        **kwargs,
    ):

        """
        Initializes a ClassificationModel model.

        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            tokenizer_type: The type of tokenizer (auto, bert, xlnet, xlm, roberta, distilbert, etc.) to use. If a string is passed, Simple Transformers will try to initialize a tokenizer class from the available MODEL_CLASSES.
                                Alternatively, a Tokenizer class (subclassed from PreTrainedTokenizer) can be passed.
            tokenizer_name: The name/path to the tokenizer. If the tokenizer_type is not specified, the model_type will be used to determine the type of the tokenizer.
            num_labels (optional): The number of labels or classes in the dataset.
            weight (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            onnx_execution_provider (optional): ExecutionProvider to use with ONNX Runtime. Will use CUDA (if use_cuda) or CPU (if use_cuda is False) by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, ClassificationArgs):
            self.args = args

        if self.args.thread_count:
            torch.set_num_threads(self.args.thread_count)

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        if self.args.seed:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.seed)

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

        if tokenizer_type is not None:
            if isinstance(tokenizer_type, str):
                _, _, tokenizer_class = MODEL_CLASSES[tokenizer_type]
            else:
                tokenizer_class = tokenizer_type

        if num_labels:
            self.config = config_class.from_pretrained(model_name, num_labels=num_labels, **self.args.config)
            self.num_labels = num_labels
        else:
            self.config = config_class.from_pretrained(model_name, **self.args.config)
            self.num_labels = self.config.num_labels

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        if self.args.onnx:
            from onnxruntime import InferenceSession, SessionOptions

            if not onnx_execution_provider:
                onnx_execution_provider = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"

            options = SessionOptions()

            if self.args.dynamic_quantize:
                model_path = quantize(Path(os.path.join(model_name, "onnx_model.onnx")))
                self.model = InferenceSession(model_path.as_posix(), options, providers=[onnx_execution_provider])
            else:
                model_path = os.path.join(model_name, "onnx_model.onnx")
                self.model = InferenceSession(model_path, options, providers=[onnx_execution_provider])
        else:
            if not self.args.quantized_model:
                self.model = model_class.from_pretrained(model_name, config=self.config, **kwargs)
            else:
                quantized_weights = torch.load(os.path.join(model_name, "pytorch_model.bin"))

                self.model = model_class.from_pretrained(None, config=self.config, state_dict=quantized_weights)

            if self.args.dynamic_quantize:
                self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            if self.args.quantized_model:
                self.model.load_state_dict(quantized_weights)
            if self.args.dynamic_quantize:
                self.args.quantized_model = True

        self.results = {}

        if not use_cuda:
            self.args.fp16 = False

        if self.args.fp16:
            try:
                from torch.cuda import amp
            except AttributeError:
                raise AttributeError("fp16 requires Pytorch >= 1.6. Please update Pytorch or turn off fp16.")

        if tokenizer_name is None:
            tokenizer_name = model_name

        if tokenizer_name in [
            "vinai/bertweet-base",
            "vinai/bertweet-covid19-base-cased",
            "vinai/bertweet-covid19-base-uncased",
        ]:
            self.tokenizer = tokenizer_class.from_pretrained(
                tokenizer_name,
                do_lower_case=self.args.do_lower_case,
                normalization=True,
                **kwargs,
            )
        else:
            self.tokenizer = tokenizer_class.from_pretrained(
                tokenizer_name, do_lower_case=self.args.do_lower_case, **kwargs
            )

        if self.args.special_tokens_list:
            self.tokenizer.add_tokens(self.args.special_tokens_list, special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.args.model_name = model_name
        self.args.model_type = model_type
        self.args.tokenizer_name = tokenizer_name
        self.args.tokenizer_type = tokenizer_type

        if model_type in ["camembert", "xlmroberta"]:
            warnings.warn(
                f"use_multiprocessing automatically disabled as {model_type}"
                " fails when using multiprocessing for feature conversion."
            )
            self.args.use_multiprocessing = False

        if self.args.wandb_project and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args.wandb_project = None

    def train_model(
        self,
        train_df,
        multi_label=False,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_df=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model using 'train_df'

        Args:
            train_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be trained on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_df (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        if args:
            self.args.update_from_dict(args)

        self.trainer = Trainer(
            model,
            args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset[validation_key],
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )

        self.trainer.train()

    def compute_metrics(
        self,
        preds,
        model_outputs,
        labels,
        eval_examples=None,
        multi_label=False,
        **kwargs,
    ):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            preds: Model predictions
            model_outputs: Model outputs
            labels: Ground truth labels
            eval_examples: List of examples on which evaluation was performed
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results.
            For non-binary classification, the dictionary format is: (Matthews correlation coefficient, tp, tn, fp, fn).
            For binary classification, the dictionary format is: (Matthews correlation coefficient, tp, tn, fp, fn, AUROC, AUPRC).
            wrong: List of InputExample objects corresponding to each incorrect prediction by the model
        """  # noqa: ignore flake8"

        assert len(preds) == len(labels)

        extra_metrics = {}
        for metric, func in kwargs.items():
            if metric.startswith("prob_"):
                extra_metrics[metric] = func(labels, model_outputs)
            else:
                extra_metrics[metric] = func(labels, preds)

        if multi_label:
            threshold_values = self.args.threshold if self.args.threshold else 0.5
            if isinstance(threshold_values, list):
                mismatched = labels != [
                    [self._threshold(pred, threshold_values[i]) for i, pred in enumerate(example)] for example in preds
                ]
            else:
                mismatched = labels != [
                    [self._threshold(pred, threshold_values) for pred in example] for example in preds
                ]
        else:
            mismatched = labels != preds

        if eval_examples:
            if not isinstance(eval_examples[0], InputExample):
                if len(eval_examples) == 2:
                    # Single sentence task
                    eval_examples = [
                        InputExample(
                            guid=i,
                            text_a=example,
                            text_b=None,
                            label=label,
                        )
                        for i, (example, label) in enumerate(zip(eval_examples[0], eval_examples[1]))
                    ]
                elif len(eval_examples) == 3:
                    # Sentence pair task
                    eval_examples = [
                        InputExample(
                            guid=i,
                            text_a=example_a,
                            text_b=example_b,
                            label=label,
                        )
                        for i, (example_a, example_b, label) in enumerate(
                            zip(eval_examples[0], eval_examples[1], eval_examples[2])
                        )
                    ]
            wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        else:
            wrong = ["NA"]

        if multi_label:
            label_ranking_score = label_ranking_average_precision_score(labels, preds)
            return {**{"LRAP": label_ranking_score}, **extra_metrics}, wrong
        elif self.args.regression:
            return {**extra_metrics}, wrong

        mcc = matthews_corrcoef(labels, preds)
        if self.model.num_labels == 2:
            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
            if self.args.sliding_window:
                return (
                    {
                        **{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn},
                        **extra_metrics,
                    },
                    wrong,
                )
            else:
                scores = np.array([softmax(element)[1] for element in model_outputs])
                fpr, tpr, thresholds = roc_curve(labels, scores)
                auroc = auc(fpr, tpr)
                auprc = average_precision_score(labels, scores)
                return (
                    {
                        **{
                            "mcc": mcc,
                            "tp": tp,
                            "tn": tn,
                            "fp": fp,
                            "fn": fn,
                            "auroc": auroc,
                            "auprc": auprc,
                        },
                        **extra_metrics,
                    },
                    wrong,
                )
        else:
            return {**{"mcc": mcc}, **extra_metrics}, wrong
