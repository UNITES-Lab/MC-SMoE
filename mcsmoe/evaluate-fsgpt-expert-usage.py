# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2023/8/6

import os
import resource

import torch
import transformers.utils.logging as hf_logging
from accelerate.utils import set_seed
from datasets import load_dataset
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer
)

from mcsmoe.data import (
    CausalZeroShotDataPreProcessor,
    tokenize_causal_zero_shot,
    TASK_MAPPING_DATASET_ARGUMENTS,
    DataCollatorForLanguageModeling
)
from mcsmoe.models import FSGPTMoEForCausalLM

set_seed(42)
hf_logging.set_verbosity_info()
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def _evaluate_downstream_usage(
        model,
        tokenizer,
        task: str = None,
        eval_batch_size: int = 32,
):
    save_dir = f"./visualization/fsgpt/{task}/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        keys_to_ignore=["answer_idx", "choice_idx", "idx"]
    )

    raw_dataset = load_dataset(*TASK_MAPPING_DATASET_ARGUMENTS[task])
    dataset = raw_dataset["train"].select(range(min(1024, len(raw_dataset["train"]))))
    dataset = dataset.map(
        CausalZeroShotDataPreProcessor(benchmark=task),
        batched=True,
        num_proc=8,
        remove_columns=dataset.column_names,
        load_from_cache_file=False
    )
    tokenized_dataset = dataset.map(
        lambda x: tokenize_causal_zero_shot(tokenizer=tokenizer, batch=x),
        num_proc=8,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False
    )
    dataloader = DataLoader(
        tokenized_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=eval_batch_size,
        num_workers=8
    )
    expert_usage = {}

    for layer_idx in tqdm(range(1, 12, 2), desc="Initializing"):
        ffn_name = f"decoder.layers.{layer_idx}.ffn"
        expert_usage[ffn_name] = torch.zeros(model.config.num_experts)

    model.eval()
    for batch in tqdm(dataloader, desc=f"Evaluating experts on {task}..."):
        with torch.no_grad():
            with torch.autocast("cuda"):
                for key in ["answer_idx", "choice_idx", "idx"]:
                    if key in batch:
                        batch.pop(key)
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(**batch)
                for layer_idx in range(1, 12, 2):
                    ffn_name = f"decoder.layers.{layer_idx}.ffn"
                    decoder_router_probs = outputs.router_logits[layer_idx][1]
                    decoder_router_expert_index = torch.argsort(decoder_router_probs)[:, -2:].reshape(-1)
                    for idx in decoder_router_expert_index:
                        expert_usage[ffn_name][idx] += 1

    for key, item in expert_usage.items():
        expert_usage[key] = item / item.sum()

    torch.save(expert_usage, os.path.join(save_dir, "expert-usage.pt"))


def main(
        checkpoint: str,
        eval_batch_size: int = 32,
):
    task_list = ["mrpc", "sst2"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = FSGPTMoEForCausalLM.from_pretrained(
        checkpoint,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        moe_eval_capacity_token_fraction=1.0,
    )
    for task in task_list:
        print(f"Evaluating {task}...")
        _evaluate_downstream_usage(
            model=model,
            tokenizer=tokenizer,
            task=task,
            eval_batch_size=eval_batch_size,
        )


if __name__ == '__main__':
    Fire(main)
