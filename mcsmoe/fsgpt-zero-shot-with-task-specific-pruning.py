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
    CausalZeroShotDataPreProcessor,
    tokenize_causal_zero_shot,
    TASK_MAPPING_DATASET_ARGUMENTS,
    DataCollatorForLanguageModeling,
    build_index_for_dataset,
    gather_predictions_references_by_causal_lm_loss
)
from mcsmoe.merging.grouping_fsgpt import (
    ExpertUsageFrequencyTrackerForFSGPT
)
from mcsmoe.models import (
    FSGPTMoEForCausalLM,
    FSGPTMoEConfig
)
from mcsmoe.pruning import (
    TaskSpecificExpertPrunerForFSGPT,
    transform_fsgpt_model_for_task_specific_pruning
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


def forward_only_with_task_specific_pruning(
        checkpoint: str,
        output_dir: str,
        task: str = "mrpc",
        num_samples_for_pruning: Optional[int] = 512,
        samples_batch_size: Optional[int] = 32,
        eval_batch_size: Optional[int] = 32,
        threshold_beta: Optional[float] = 0.1,
        minimum_num_experts: Optional[int] = 8,
        pruning_layers: Optional[Union[str, List, int]] = None,
        num_pruning_times: Optional[int] = 4,
        debug: Optional[bool] = False,
):
    if output_dir is None:
        raise ValueError("output_dir must be specified")
    else:
        output_dir = os.path.join(output_dir, "task-specific-pruning")
        os.makedirs(output_dir, exist_ok=True)

    pruning_layers = sanitize_pruning_layers(pruning_layers)
    tokenizer = GPT2TokenizerFast.from_pretrained(checkpoint)
    if debug:
        config = FSGPTMoEConfig.from_pretrained(
            checkpoint,
            moe_eval_capacity_token_fraction=1.0,
        )
        model = FSGPTMoEForCausalLM(config).bfloat16().cuda()
        print("[Merging]Debug mode, using fake model")
    else:
        model = FSGPTMoEForCausalLM.from_pretrained(
            checkpoint,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            moe_eval_capacity_token_fraction=1.0,
        )
    model = transform_fsgpt_model_for_task_specific_pruning(model)
    tokenizer = GPT2TokenizerFast.from_pretrained(checkpoint)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        keys_to_ignore=["answer_idx", "choice_idx", "idx"]
    )
    raw_dataset = load_dataset(*TASK_MAPPING_DATASET_ARGUMENTS[task])
    dataset = raw_dataset["train"].select(range(num_samples_for_pruning))
    dataset = build_index_for_dataset(dataset)
    dataset = dataset.map(
        CausalZeroShotDataPreProcessor(benchmark=task),
        batched=True,
        num_proc=8,
        remove_columns=dataset.column_names,
        load_from_cache_file=False
    )
    tokenized_dataset = dataset.map(
        lambda x: tokenize_causal_zero_shot(tokenizer=tokenizer, batch=x, keep_input_only=True),
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
        CausalZeroShotDataPreProcessor(benchmark=task),
        batched=True,
        num_proc=8,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False
    )
    tokenized_eval_dataset = eval_dataset.map(
        lambda x: tokenize_causal_zero_shot(tokenizer=tokenizer, batch=x),
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

    usage_tracker = ExpertUsageFrequencyTrackerForFSGPT(
        named_parameters_caller=model.named_parameters,
        compute_every_n_steps=1,
        beta=0.6,
        device='cuda'
    )
    task_specifc = TaskSpecificExpertPrunerForFSGPT(
        model_config=model.config,
        threshold_beta=threshold_beta,
        minimum_num_experts=minimum_num_experts,
        pruning_layers=pruning_layers
    )

    complete_steps = 0
    model.eval()
    max_train_steps = len(dataloader)
    pruning_interval_steps = int(max_train_steps / num_pruning_times)
    for step, batch in enumerate(tqdm(dataloader, desc="Task-specific Pruning")):
        with torch.no_grad(), torch.autocast("cuda"):
            # fake training, no optim
            for key in batch:
                if key in ["answer_idx", "choice_idx", "idx"]:
                    batch.pop(key)
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            usage_tracker.step(
                model_outputs=outputs,
                global_step=complete_steps
            )
            if complete_steps % pruning_interval_steps == 0:
                task_specifc.execute_pruning(
                    model=model,
                    usage_frequency_state_dict=usage_tracker.usage_frequency_state_dict(),
                    prune_to_minimum=False
                )
        complete_steps += 1
    task_specifc.execute_pruning(
        model=model,
        usage_frequency_state_dict=usage_tracker.usage_frequency_state_dict(),
        prune_to_minimum=True
    )
    print(f"Task-specific pruned model num params: {model.num_parameters()}")

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
    predictions_references = gather_predictions_references_by_causal_lm_loss(
        ids_list=ids_list,
        answer_ids_list=answer_ids_list,
        choice_ids_list=choice_ids_list,
        losses_list=losses_list,
    )
    eval_res = metric.compute(predictions=predictions_references["predictions"],
                              references=predictions_references["references"])
    print(f"Task-specific {task} evaluation result: {eval_res}")


if __name__ == "__main__":
    Fire(forward_only_with_task_specific_pruning)
