import os
import os.path
from typing import Optional, Union, List

import torch
from accelerate.logging import get_logger
from datasets import load_dataset
from evaluate import load
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from transformers import logging as hf_logging

from mcsmoe.data import (
    CasualZeroShotDataPreProcessor,
    tokenize_casual_zero_shot,
    TASK_MAPPING_DATASET_ARGUMENTS,
    DataCollatorForLanguageModeling,
    build_index_for_dataset,
    gather_predictions_references_by_casual_lm_loss
)
from mcsmoe.merging.utils import load_merged_fsgpt_moe_from_checkpoint
from mcsmoe.pruning.losparse import (
    LoSparseConfig,
    substitute_fsgpt_moe_expert_weights,
    LoSparsePrunerForFSGPT
)

logger = get_logger(__name__)
logger.setLevel(20)
hf_logging.set_verbosity_error()


def sanitize_pruning_layers(layers: Union[str, List, int]):
    if layers is None:
        layers = list(range(1, 12, 2))
    elif isinstance(layers, str) and len(layers) > 0:
        layers = [int(x) for x in layers.split(",")]
    elif isinstance(layers, str) and len(layers) == 0:
        layers = []
    elif isinstance(layers, int):
        layers = [layers]
    return layers


def post_merging_losparse(
        checkpoint: str,
        output_dir: str,
        task: str = "mrpc",
        final_threshold: Optional[float] = 0.01,
        low_rank_factor: Optional[int] = 16,
        warmup_steps: Optional[int] = 4,
        samples_batch_size: Optional[int] = 32,
        eval_batch_size: Optional[int] = 32,
        compression_layers: Optional[Union[str, List, int]] = None,
):
    if output_dir is None:
        raise ValueError("output_dir must be specified")
    else:
        output_dir = os.path.join(output_dir, f"rank-{low_rank_factor}")
        os.makedirs(output_dir, exist_ok=True)

    compression_layers = sanitize_pruning_layers(compression_layers)
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(checkpoint)
    except OSError:
        tokenizer = GPT2TokenizerFast.from_pretrained("Phando/fairseq-moe-15b-bf16")
    config = LoSparseConfig(
        final_threshold=final_threshold,
        warmup_steps=warmup_steps
    )
    # load merge
    model = load_merged_fsgpt_moe_from_checkpoint(
        checkpoint=checkpoint,
    )
    d_model = model.config.hidden_size
    d_ff = model.config.intermediate_size
    parameter_ratio = (d_model + d_ff) * low_rank_factor / (d_model * d_ff)
    model = substitute_fsgpt_moe_expert_weights(
        model=model,
        parameter_ratio=parameter_ratio,
        has_sparse=True,
        use_svd=True,
        compression_layers=compression_layers
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        keys_to_ignore=["answer_idx", "choice_idx", "idx"]
    )
    raw_dataset = load_dataset(*TASK_MAPPING_DATASET_ARGUMENTS[task])
    dataset = raw_dataset['train'].select(range(512))
    dataset = build_index_for_dataset(dataset)
    dataset = dataset.map(
        CasualZeroShotDataPreProcessor(benchmark=task),
        batched=True,
        num_proc=8,
        remove_columns=dataset.column_names,
        load_from_cache_file=False
    )
    tokenized_dataset = dataset.map(
        lambda x: tokenize_casual_zero_shot(tokenizer=tokenizer, batch=x, keep_input_only=True),
        num_proc=8,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False
    )
    dataloader = DataLoader(
        tokenized_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=samples_batch_size,
        num_workers=8
    )

    eval_dataset = raw_dataset["validation"]
    eval_dataset = build_index_for_dataset(eval_dataset)
    eval_dataset = eval_dataset.map(
        CasualZeroShotDataPreProcessor(benchmark=task),
        batched=True,
        num_proc=8,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False
    )
    tokenized_eval_dataset = eval_dataset.map(
        lambda x: tokenize_casual_zero_shot(tokenizer=tokenizer, batch=x),
        num_proc=8,
        batched=True,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False
    )
    eval_dataloader = DataLoader(
        tokenized_eval_dataset,
        shuffle=False,
        batch_size=eval_batch_size,
        num_workers=4,
        collate_fn=data_collator
    )

    pruner = LoSparsePrunerForFSGPT(
        config=config,
        total_step=len(dataloader),
        mask_param_names=['sparse']
    )

    complete_steps = 0
    model.eval()
    for step, batch in enumerate(tqdm(dataloader, desc="LoSparse Pruning")):
        with torch.autocast("cuda"):
            # fake training, no optim
            for key in batch:
                if key in ["answer_idx", "choice_idx", "idx"]:
                    batch.pop(key)
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            threshold, mask_threshold = pruner.update_and_pruning(
                named_parameters_caller=model.named_parameters,
                global_step=complete_steps
            )
            model.zero_grad()
            print(
                f"Step {complete_steps} threshold: {threshold} mask_threshold: {mask_threshold} num-param: {sum([p.count_nonzero() for p in model.parameters()])}")
        complete_steps += 1
    print(f"LoSparse pruned model num params: {model.num_parameters()}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # eval
    ids_list = []
    answer_ids_list = []
    choice_ids_list = []
    losses_list = []
    try:
        metric = load(*TASK_MAPPING_DATASET_ARGUMENTS[task])
    except FileNotFoundError:
        print(f"[Evaluation warning] No metric found for task {task}, using accuracy instead.")
        metric = load("accuracy")
    for eval_step, eval_batch in tqdm(enumerate(eval_dataloader),
                                      total=len(eval_dataloader), desc="Evaluating"):
        ids_list += eval_batch.pop("idx")
        answer_ids_list += eval_batch.pop("answer_idx")
        choice_ids_list += eval_batch.pop("choice_idx")
        eval_batch = {k: v.cuda() for k, v in eval_batch.items()}
        with torch.no_grad():
            with torch.autocast("cuda"):
                logits = model(**eval_batch).logits
            # eval probs
            loss_fct = torch.nn.CrossEntropyLoss()
            labels = eval_batch["labels"]
            shift_labels = labels.new_zeros(labels.shape)
            shift_labels[:, :-1] = labels[:, 1:].clone()
            shift_labels[:, -1] = model.config.pad_token_id
            losses = torch.stack([
                loss_fct(logits[i], shift_labels[i].to(logits.device)) for i in range(logits.shape[0])
            ])
            losses_list += losses.tolist()
    predictions_references = gather_predictions_references_by_casual_lm_loss(
        ids_list=ids_list,
        answer_ids_list=answer_ids_list,
        choice_ids_list=choice_ids_list,
        losses_list=losses_list,
    )
    eval_res = metric.compute(predictions=predictions_references["predictions"],
                              references=predictions_references["references"])
    print(f"Task-specific {task} evaluation result: {eval_res}")


if __name__ == "__main__":
    Fire(post_merging_losparse)
