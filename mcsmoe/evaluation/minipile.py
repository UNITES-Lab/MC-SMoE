# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/18
from itertools import chain
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling
)


def get_minipile_dataloder(
        tokenizer: PreTrainedTokenizer,
        block_size: int,
        batch_size: int,
        subset_ratio: Optional[float] = 1.0,
) -> DataLoader:
    dataset = load_dataset("JeanKaddour/minipile", split="validation")
    column_names = dataset.column_names

    dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=False),
        batched=True,
        num_proc=8,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(
        group_texts,
        batched=True,
        num_proc=8,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    dataset = dataset.shuffle()
    dataset = dataset.select(range(int(len(dataset) * subset_ratio)))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    return data_loader


def evaluate_minipile_perplexity(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        log: Optional[bool] = True,
        block_size: Optional[int] = 512,
        batch_size: Optional[int] = 1,
) -> float:
    data_loader = get_minipile_dataloder(
        tokenizer=tokenizer,
        block_size=block_size,
        batch_size=batch_size,
    )

    loss_list = []
    for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Evaluating")):
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, output_router_logits=False)
        loss_list.append(outputs.loss.item())

    ppl = torch.exp(torch.tensor(loss_list).mean()).item()

    if log:
        print(f"Perplexity of MiniPile: {ppl:.2f}")

    return ppl
