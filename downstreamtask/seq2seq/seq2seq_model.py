import os
import torch
import logging
from dataclasses import asdict

import warnings

warnings.filterwarnings("ignore")

from transformers import (
    AutoConfig,
    AutoTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    MBartConfig,
    MBartForConditionalGeneration,
    M2M100Config,
    M2M100ForConditionalGeneration,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    MT5Config,
    MT5ForConditionalGeneration,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoModelForSeq2SeqLM,
)

from downstreamtask.seq2seq.seq2seq_util import *
from downstreamtask.config.utils import sweep_config_to_sweep_values
from downstreamtask.config.model_args import Seq2SeqTrainingArguments

from tqdm.auto import tqdm
from multiprocessing import Pool


try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "auto": (AutoConfig, AutoModelForSeq2SeqLM),
    "bart": (BartConfig, BartForConditionalGeneration),
    "mbart": (MBartConfig, MBartForConditionalGeneration),
    "t5": (T5Config, T5ForConditionalGeneration),
    "mt5": (MT5Config, MT5ForConditionalGeneration),
    "m2m": (M2M100Config, M2M100ForConditionalGeneration),
}


def update_from_dict(self, new_values):
    if isinstance(new_values, dict):
        for key, value in new_values.items():
            setattr(self, key, value)
    else:
        raise (TypeError(f"{new_values} is not a Python dict."))


class Seq2SeqModel:
    def __init__(
        self,
        model_type,
        model_name,
        args=None,
        tokenizer=None,
        cuda_device=-1,
        **kwargs,
    ):

        """
        Initializes a T5Model model.

        Args:
            model_type: The type of model (t5, mt5, byt5)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            no_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, Seq2SeqTrainingArguments):
            self.args = args

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

        if not args.no_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'no_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set `no_cuda=False`."
                )
        else:
            self.device = "cpu"

        self.results = {}

        config_class, model_class = MODEL_CLASSES[model_type]

        if model_name is None:
            self.config = self.args.config
            self.model = model_class(config=self.config)
        else:
            self.config = config_class.from_pretrained(model_name, **self.args.config)
            self.model = model_class.from_pretrained(model_name, config=self.config)

        if isinstance(tokenizer, T5Tokenizer):
            self.tokenizer = tokenizer
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                truncate=True,
                do_lower_case=True,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
            )

        if self.args.dynamic_quantize:
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)

        if self.args.special_tokens_list:
            self.tokenizer.add_tokens(self.args.special_tokens_list, special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.args.model_type = model_type

        if model_name is None:
            self.args.model_name = "T5_from_scratch"
        else:
            self.args.model_name = model_name

        if self.args.wandb_project and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args.wandb_project = None

    def train_model(
        self,
        train_data,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_data=None,
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
                " Set args.overwrite_output_dir = True to overcome.".format(output_dir)
            )

        model = self.model
        self._move_model_to_device()
        train_dataset = self.load_and_cache_examples(train_data, verbose=verbose)

        if args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                config={**asdict(args)},
                **args.wandb_kwargs,
            )
            wandb.run._label(repo="downstreamtask")
            wandb.watch(self.model)
            self.wandb_run_id = wandb.run.id

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=model)

        self.trainer = Seq2SeqTrainer(
            model,
            self.args,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            train_dataset=train_dataset["train"],
            eval_dataset=train_dataset["validation"],
        )

        self.trainer.train()

    def eval_model(self, eval_data, output_dir=None, verbose=True, silent=False, **kwargs):
        model = self.model
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=model)

        self.trainer = Seq2SeqTrainer(
            model,
            self.args,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            eval_dataset=eval_data,
        )

        self.trainer.evaluate()

    def predict(self, to_predict):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction. Note that the prefix should be prepended to the text.

        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        self._move_model_to_device()
        # self.use_pretrained_model()

        all_outputs = []

        # Batching
        for batch in tqdm(
            [
                to_predict[i : i + self.args.per_device_eval_batch_size]
                for i in range(0, len(to_predict), self.args.per_device_eval_batch_size)
            ],
            desc="Generating outputs",
            disable=self.args.silent,
        ):
            input_batch = self.tokenizer.prepare_seq2seq_batch(
                src_texts=batch,
                max_length=self.args.generation_max_length,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )

            input_ids = input_batch["input_ids"]
            attention_mask = input_batch["attention_mask"]

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            outputs = self.model.generate(
                input_ids=input_ids,
                num_beams=self.args.generation_num_beams,
                max_length=self.args.generation_max_length,
                length_penalty=self.args.length_penalty,
                early_stopping=self.args.early_stopping,
                repetition_penalty=self.args.repetition_penalty,
                do_sample=self.args.do_sample,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                num_return_sequences=self.args.num_return_sequences,
            )

            all_outputs.extend(outputs.cpu().numpy())

        if self.args.use_multiprocessed_decoding:
            if self.args.multiprocessing_chunksize == -1:
                chunksize = max(len(all_outputs) // (self.args.process_count * 2), 500)
            else:
                chunksize = self.args.multiprocessing_chunksize

            self.model.to("cpu")
            with Pool(self.args.process_count) as p:
                outputs = list(
                    tqdm(
                        p.imap(self._decode, all_outputs, chunksize=chunksize),
                        total=len(all_outputs),
                        desc="Decoding outputs",
                        disable=self.args.silent,
                    )
                )
            self._move_model_to_device()

        else:
            outputs = [
                self.tokenizer.decode(
                    output_id,
                    skip_special_tokens=self.args.skip_special_tokens,
                    clean_up_tokenization_spaces=True,
                )
                for output_id in all_outputs
            ]

        if self.args.num_return_sequences > 1:
            return [
                outputs[i : i + self.args.num_return_sequences]
                for i in range(0, len(outputs), self.args.num_return_sequences)
            ]
        else:
            return outputs

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _decode(self, output_id):
        return self.tokenizer.decode(
            output_id,
            skip_special_tokens=self.args.skip_special_tokens,
            clean_up_tokenization_spaces=True,
        )

    def load_and_cache_examples(self, data, evaluate=False, no_cache=False, verbose=True, silent=False):
        """
        Creates a Seq2SeqDataset from data.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"

        if self.args.use_hf_datasets:
            dataset = load_hf_dataset(data, tokenizer, self.args)
            return dataset
        elif args.dataset_class:
            CustomDataset = args.dataset_class
            return CustomDataset(tokenizer, args, data, mode)

    def save_model(self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            self.save_model_args(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = Seq2SeqTrainingArguments()
        args.load(input_dir)
        return args
