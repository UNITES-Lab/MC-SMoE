import math
import os
import os.path
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    get_scheduler,
)

from mcsmoe.data import (
    Seq2SeqDataPreProcessor,
    tokenize_seq2seq,
    TASK_MAPPING_DATASET_ARGUMENTS,
    DataCollatorForSeq2Seq,
    get_evaluate_fn,
    EXTRA_KEYS_FOR_EVAL,
    keep_only_supporting_facts_in_context_for_hotpotqa
)

logger = get_logger(__name__)
logger.setLevel(20)


@dataclass
class T5FineTuningConfig:
    task: Optional[str] = "cola"
    num_epochs: Optional[int] = 5
    batch_size: Optional[int] = 32
    adam_epsilon: Optional[float] = 1e-6
    adam_betas: Optional[Tuple[float, float]] = (0.9, 0.98)
    lr: Optional[float] = 5e-4
    warmup_steps: Optional[int] = 16
    weight_decay: Optional[float] = 0.01
    dropout: Optional[float] = 0.1


def finetune_on_downstream_task(
        output_dir: Optional[str] = None,
        task: Optional[str] = "cola",
        no_eval_until_epochs: Optional[int] = 0,
        per_device_train_batch_size: Optional[int] = 8,
        per_device_eval_batch_size: Optional[int] = 8,
        num_epochs: Optional[int] = 10,
        report: Optional[bool] = True,
        log_steps: Optional[int] = 1,
        weight_decay: Optional[float] = 0.01,
        learning_rate: Optional[float] = 5e-4,
        gradient_accumulation_steps: Optional[int] = 32,
        warmup_steps: Optional[int] = 16,
        save_each_epoch: Optional[bool] = False,
        preprocessing_num_workers: Optional[int] = None,
):
    if output_dir is None:
        raise ValueError("output_dir must be specified")
    if task not in TASK_MAPPING_DATASET_ARGUMENTS:
        raise ValueError(f"task must be one of {list(TASK_MAPPING_DATASET_ARGUMENTS.keys())}")
    if task not in output_dir:
        output_dir = os.path.join(output_dir, task)

    training_config = T5FineTuningConfig(
        task=task,
        num_epochs=num_epochs,
        adam_epsilon=1e-8,
        adam_betas=(0.9, 0.98),
        lr=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        dropout=0.1,
    )

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5TokenizerFast.from_pretrained("t5-base")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           max_length=tokenizer.model_max_length,
                                           return_tensors='pt',
                                           keys_to_ignore=EXTRA_KEYS_FOR_EVAL)

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    set_seed(42)

    if accelerator.is_local_main_process:
        print(f"Number of parameters in T5: {model.num_parameters()}")
        if report:
            wandb.init(project="mc-smoe",
                       config={**model.config.__dict__, **training_config.__dict__},
                       name=f"t5-{task}-{per_device_train_batch_size}-bs-{learning_rate}-lr")

    raw_dataset = load_dataset(*TASK_MAPPING_DATASET_ARGUMENTS[task])

    train_dataset = raw_dataset["train"]
    eval_dataset = raw_dataset["validation"] if task != "mnli" else (
        raw_dataset["validation_matched"], raw_dataset["validation_mismatched"]  # TODO
    )

    with accelerator.main_process_first():
        if task == "hotpotqa":
            train_dataset = train_dataset.map(
                keep_only_supporting_facts_in_context_for_hotpotqa,
                batched=False,
                num_proc=preprocessing_num_workers
            )
            eval_dataset = eval_dataset.map(
                keep_only_supporting_facts_in_context_for_hotpotqa,
                batched=False,
                num_proc=preprocessing_num_workers
            )
        train_dataset = train_dataset.map(
            Seq2SeqDataPreProcessor(benchmark=task),
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=train_dataset.column_names
        )
        eval_dataset = eval_dataset.map(
            Seq2SeqDataPreProcessor(benchmark=task, keep_specific_keys=EXTRA_KEYS_FOR_EVAL),
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=eval_dataset.column_names
        )

        tokenized_train_dataset = train_dataset.map(
            lambda x: tokenize_seq2seq(tokenizer=tokenizer, batch=x, keep_other_keys=False),
            num_proc=preprocessing_num_workers,
            batched=True,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=False
        )
        tokenized_eval_dataset = eval_dataset.map(
            lambda x: tokenize_seq2seq(tokenizer=tokenizer, batch=x, keep_other_keys=True),
            num_proc=preprocessing_num_workers,
            batched=True,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=False
        )

    logger.info(f"Number of training examples: {len(tokenized_train_dataset)}")
    logger.info(f"Number of validation examples: {len(tokenized_eval_dataset)}")

    train_dataloader = DataLoader(
        tokenized_train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=per_device_train_batch_size,
        num_workers=5
    )
    eval_dataloader = DataLoader(
        tokenized_eval_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=per_device_eval_batch_size,
        num_workers=5
    )

    # optimizer
    no_decay = ["bias", "layer_norm.weight", "LayerNorm", "layernorm", "layer_norm", "ln"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                  lr=learning_rate,
                                  eps=training_config.adam_epsilon,
                                  betas=training_config.adam_betas,
                                  weight_decay=weight_decay)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch
    no_eval_until_steps = no_eval_until_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps
    )
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    evaluate_fn = get_evaluate_fn(
        task=task,
        tokenizer=tokenizer,
        raw_eval_dataset=raw_dataset["validation"]
    )

    num_eval_steps = num_update_steps_per_epoch
    total_batch_size = (
            per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    ) if accelerator.num_processes is not None else (
            per_device_train_batch_size * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Num Samples = {len(train_dataloader)}")
    logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_eval = 0

    # Train!
    for epoch in range(num_epochs):
        # model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_local_main_process and completed_steps % log_steps == 0 and (
                    accelerator.sync_gradients
            ):
                logger.info(f"epoch {epoch}, step {step}: loss {loss.item()}")
                wandb.log({"train_loss": loss.item(),
                           "epoch": completed_steps / num_update_steps_per_epoch,
                           "learning_rate": lr_scheduler.get_lr()}, step=completed_steps)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
            if (completed_steps % num_eval_steps == 0 and completed_steps >= no_eval_until_steps and
                accelerator.sync_gradients) or (completed_steps <= 1 and step == 0 and no_eval_until_steps == 0
            ):
                model.eval()
                losses = []
                output_labels = []
                output_predictions = []
                output_ids = [] if task in ["squad", "copa", "multirc", "squad_v2", "hotpotqa"] else None
                for eval_step, eval_batch in enumerate(eval_dataloader):
                    extra_keys_eval_batch = {}
                    for key in list(eval_batch.keys()):
                        if key in EXTRA_KEYS_FOR_EVAL:
                            extra_keys_eval_batch[key] = eval_batch.pop(key)
                    with torch.no_grad():
                        outputs = model(**eval_batch)
                    eval_labels = accelerator.gather(eval_batch['labels'])
                    output_labels += torch.cat([
                        eval_labels,
                        torch.ones(eval_labels.shape[0], tokenizer.model_max_length - eval_labels.shape[1],
                                   dtype=eval_labels.dtype,
                                   device=eval_labels.device) * -100
                    ], dim=-1)
                    eval_logits = accelerator.gather(outputs.logits)
                    output_predictions += eval_logits.argmax(dim=-1).tolist()
                    if task in ["squad", "squad_v2", "hotpotqa"]:
                        output_ids += extra_keys_eval_batch["id"]
                    elif task == "copa" or task == "multirc":
                        output_ids += extra_keys_eval_batch["idx"]
                    losses.append(accelerator.gather_for_metrics(outputs["loss"]))
                losses = torch.cat(losses)
                eval_loss = torch.mean(losses)
                output_labels = torch.stack(output_labels, dim=0)
                eval_res = evaluate_fn(predictions=output_predictions, labels=output_labels, ids=output_ids)
                metric_key = list(eval_res.keys())[0]
                eval_res["loss"] = eval_loss.item()

                if eval_res[metric_key] > best_eval:
                    best_eval = eval_res[metric_key]
                    accelerator.wait_for_everyone()
                    wandb.summary["best_" + metric_key] = best_eval
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(os.path.join(output_dir, "best"),
                                                    is_main_process=accelerator.is_local_main_process,
                                                    save_function=accelerator.save)
                    if accelerator.is_local_main_process:
                        tokenizer.save_pretrained(os.path.join(output_dir, "best"))
                        print(f"Best model saved with best evaluation {metric_key}: {eval_res[metric_key]}")
                        print(f"All evaluation results: {eval_res}")

                if accelerator.is_local_main_process:
                    print(f"Step {completed_steps}: eval loss {eval_res['loss']}")
                    eval_res = {("eval_" + k): v for k, v in eval_res.items()}
                    wandb.log(eval_res, step=completed_steps)

                # model.train()

        if save_each_epoch:
            save_dir = os.path.join(output_dir, f"epoch-{epoch + 1}")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(save_dir,
                                            is_main_process=accelerator.is_local_main_process,
                                            save_function=accelerator.save)
            if accelerator.is_local_main_process:
                tokenizer.save_pretrained(save_dir)

    # Finish Training!
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(os.path.join(output_dir, "latest"),
                                    is_main_process=accelerator.is_local_main_process,
                                    save_function=accelerator.save)

    if accelerator.is_local_main_process:
        tokenizer.save_pretrained(os.path.join(output_dir, "latest"))

    if accelerator.is_local_main_process and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    Fire(finetune_on_downstream_task)
