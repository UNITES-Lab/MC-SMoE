import os
import resource
from typing import Optional, Union, List

import torch
import transformers.utils.logging as hf_logging
from accelerate.utils import set_seed
from datasets import load_dataset
from evaluate import load
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer
)

from mcsmoe.data import (
    CasualZeroShotDataPreProcessor,
    tokenize_casual_zero_shot,
    TASK_MAPPING_DATASET_ARGUMENTS,
    DataCollatorForLanguageModeling,
    build_index_for_dataset,
    gather_predictions_references_by_casual_lm_loss
)
from mcsmoe.merging.grouping_fsgpt import (
    ExpertsGrouperForFSGPT,
    merge_by_groups_with_usage_frequency_weighting,
)
from mcsmoe.models import FSGPTMoEForCausalLM, FSGPTMoEConfig

set_seed(42)
hf_logging.set_verbosity_info()
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def _sanitize_merging_layers(layers: Union[str, List, int]):
    if layers is None:
        layers = list(range(1, 12, 2))
    elif isinstance(layers, str) and len(layers) > 0:
        layers = [int(x) for x in layers.split(",")]
    elif isinstance(layers, str) and len(layers) == 0:
        layers = []
    elif isinstance(layers, int):
        layers = [layers]
    return layers


def merge_fsgpt_by_usage_frequency_weighting(
        output_dir: Optional[str] = None,
        checkpoint: Optional[str] = None,
        task: Optional[str] = "sst2",
        num_samples_for_merging: Optional[int] = 512,
        samples_batch_size: Optional[int] = 32,
        eval_batch_size: Optional[int] = 32,
        similarity_base: Optional[str] = "router-logits",
        similarity_fn: Optional[str] = "cosine",
        num_groups: Optional[int] = 16,
        merging_layers: Optional[Union[str, List, int]] = None,
        debug: Optional[bool] = False,
):
    if output_dir is None:
        raise ValueError("output_dir must be specified")
    else:
        output_dir = os.path.join(output_dir, similarity_base)
        os.makedirs(output_dir, exist_ok=True)
    merging_layers = _sanitize_merging_layers(merging_layers)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
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
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        keys_to_ignore=["answer_idx", "choice_idx", "idx"]
    )
    raw_dataset = load_dataset(*TASK_MAPPING_DATASET_ARGUMENTS[task])
    dataset = raw_dataset["train"].select(range(min(num_samples_for_merging, len(raw_dataset["train"]))))
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

    grouper = ExpertsGrouperForFSGPT(
        config=model.config,
        similarity_base=similarity_base,
        similarity_fn=similarity_fn,
    )
    grouper.compute_all_similarities(
        model=model,
        dataloader=dataloader,
    )
    grouper.compute_all_usages(
        model=model,
        dataloader=dataloader,
    )
    core_experts = grouper.group_experts_into_clusters_by_routing_guided_globally(
        average_num_groups=num_groups,
        merging_layers=merging_layers,
    )
    print(f"[Merging]Core experts: {core_experts}")
    torch.save(grouper.group_state_dict(), os.path.join(output_dir, "group_state_dict.pt"))
    torch.save(core_experts, os.path.join(output_dir, "core_experts.pt"))
    print(f"[Merging]Number of parameters before merging: {model.num_parameters()}")
    model = merge_by_groups_with_usage_frequency_weighting(
        model=model,
        grouper=grouper,
        merging_layers=merging_layers,
    )
    print(f"[Merging]Number of parameters after merging: {model.num_parameters()}")
    model.save_pretrained(output_dir)

    # ========= evaluate downstream zero-shot =========
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

    model.eval()
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
    print(f"{task} evaluation result: {eval_res}")


if __name__ == '__main__':
    Fire(merge_fsgpt_by_usage_frequency_weighting)
