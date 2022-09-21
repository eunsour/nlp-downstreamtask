# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import logging
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Optional
from multiprocessing import cpu_count

from transformers import TrainingArguments
from transformers.utils import add_start_docstrings

logger = logging.getLogger(__name__)

def get_default_process_count():
    process_count = cpu_count() - 2 if cpu_count() > 2 else 1
    if sys.platform == "win32":
        process_count = min(process_count, 61)

    return process_count

@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Args:
        sortish_sampler (`bool`, *optional*, defaults to `False`):
            Whether to use a *sortish sampler* or not. Only possible if the underlying datasets are *Seq2SeqDataset*
            for now but will become generally available in the near future.

            It sorts the inputs according to lengths in order to minimize the padding size, with a bit of randomness
            for the training set.
        predict_with_generate (`bool`, *optional*, defaults to `False`):
            Whether to use generate to calculate generative metrics (ROUGE, BLEU).
        generation_max_length (`int`, *optional*):
            The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default to the
            `max_length` value of the model configuration.
        generation_num_beams (`int`, *optional*):
            The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default to the
            `num_beams` value of the model configuration.
    """

    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to use SortishSampler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `max_length` value of the model configuration."
        },
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `num_beams` value of the model configuration."
        },
    )

    
    # customized arguments
    best_model_dir: str = "outputs/best_model"
    cache_dir: str = "cache_dir/"
    config: dict = field(default_factory=dict)
    dataset_class: Dataset = None
    do_sample: bool = False
    dynamic_quantize: Optional[bool] = field(default=False)
    early_stopping: bool = True
    length_penalty: Optional[int] = 2.0
    max_seq_length: int = 128
    model_type: Optional[str] = None
    multiprocessing_chunksize: Optional[int] = -1
    n_gpu: int = 1
    no_cache: bool = False
    num_return_sequences: int = 1
    process_count: int = field(default_factory=get_default_process_count)
    repetition_penalty: float = 1.0
    save_best_model: bool = True
    should_log: bool = False
    silent: bool = False
    skip_special_tokens: bool = True
    special_tokens_list: list = field(default_factory=list)
    src_lang = "en"
    tgt_lang = "ko"
    top_k: float = None
    top_p: float = None
    use_hf_datasets: bool = True
    use_multiprocessed_decoding: bool = False
    wandb_kwargs: dict = field(default_factory=dict)
    wandb_project: str = None

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))
            