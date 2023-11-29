import resource
from typing import Optional, List, Union, Tuple

import torch
from accelerate.utils import set_seed
from datasets import load_dataset
from evaluate import load
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
)

from mcsmoe.data import (
    CausalZeroShotDataPreProcessor,
    tokenize_causal_zero_shot,
    TASK_MAPPING_DATASET_ARGUMENTS,
    DataCollatorForLanguageModeling,
    build_index_for_dataset,
    gather_predictions_references_by_causal_lm_loss
)
from mcsmoe.models import FSGPTForCausalLM, FSGPTMoEForCausalLM

set_seed(42)
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def prepare_model(
        checkpoint: str,
) -> Union[FSGPTMoEForCausalLM, FSGPTForCausalLM]:
    if checkpoint is not None and "moe" in checkpoint:
        model = FSGPTMoEForCausalLM.from_pretrained(
            checkpoint,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            moe_eval_capacity_token_fraction=1.0,
        )
    else:
        checkpoint = "Phando/fairseq-dense-125m" if checkpoint is None else checkpoint
        model = FSGPTForCausalLM.from_pretrained(
            checkpoint,
            device_map='auto',
            torch_dtype=torch.bfloat16
        )
    return model


def evaluate_downstream_zero_shot(
        tasks: Union[str, Tuple, List],
        checkpoint: Optional[str] = None,
        eval_batch_size: int = 64,
):
    model = prepare_model(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # tokenizer.model_max_length = 1024 # for memory issue
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        keys_to_ignore=["answer_idx", "choice_idx", "idx"]
    )

    task_list = tasks.replace(" ", "").split(",") if isinstance(tasks, str) else list(tasks)
    eval_res_dict = {}

    for task in task_list:
        raw_dataset = load_dataset(*TASK_MAPPING_DATASET_ARGUMENTS[task])
        eval_dataset = raw_dataset["validation"] if task != "mnli" else raw_dataset["validation_matched"]

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
        predictions_references = gather_predictions_references_by_causal_lm_loss(
            ids_list=ids_list,
            answer_ids_list=answer_ids_list,
            choice_ids_list=choice_ids_list,
            losses_list=losses_list,
        )
        eval_res = metric.compute(predictions=predictions_references["predictions"],
                                  references=predictions_references["references"])
        print(f"{task} evaluation result: {eval_res}")
        eval_res_dict[task] = eval_res
    print(f"Full evaluation result: {eval_res_dict}")


if __name__ == '__main__':
    Fire(evaluate_downstream_zero_shot)
