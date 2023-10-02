from typing import Optional, Union

import torch
from accelerate.utils import set_seed
from datasets import load_dataset
from fire import Fire
from torch.utils.data import DataLoader
from transformers import (
    GPT2TokenizerFast,
)

from mcsmoe.calflops import calculate_flops
from mcsmoe.data import (
    Seq2SeqDataPreProcessor,
    tokenize_seq2se2_to_casual_lm,
    TASK_MAPPING_DATASET_ARGUMENTS,
    DataCollatorForLanguageModeling,
)
from mcsmoe.merging.utils import load_merged_fsgpt_moe_from_checkpoint
from mcsmoe.models import (
    FSGPTForCausalLM,
    FSGPTMoEForCausalLM,
)
from mcsmoe.pruning.losparse import substitute_fsgpt_moe_expert_weights

set_seed(233)


class ModelWrapper(torch.nn.Module):
    def __init__(self, model: Union[FSGPTForCausalLM, FSGPTMoEForCausalLM]):
        super().__init__()
        self.model = model

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.FloatTensor,
            labels: torch.LongTensor,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).logits


def prepare_dataloader(task: str, tokenizer: GPT2TokenizerFast, batch_size: int):
    raw_dataset = load_dataset(*TASK_MAPPING_DATASET_ARGUMENTS[task])
    dataset = raw_dataset["train"]
    dataset = dataset.map(
        Seq2SeqDataPreProcessor(benchmark=task),
        batched=True,
        num_proc=8,
        remove_columns=dataset.column_names,
        load_from_cache_file=False
    )
    dataset = dataset.map(
        lambda x: tokenize_seq2se2_to_casual_lm(tokenizer=tokenizer, batch=x),
        num_proc=8,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        keys_to_ignore=["answer_idx", "choice_idx", "idx"]
    )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=batch_size,
        num_workers=8
    )
    return dataloader


def profile_inference_flops(
        checkpoint: str,
        task: str,
        batch_size: Optional[int] = 256,
        losparse_checkpoint: Optional[str] = None,
):
    if "dense" in checkpoint:
        model = FSGPTForCausalLM.from_pretrained(
            checkpoint,
            device_map="auto",
        )
    elif "merging" in checkpoint:
        model = load_merged_fsgpt_moe_from_checkpoint(checkpoint)
        if losparse_checkpoint is not None:
            d_model = model.config.d_model
            d_ff = model.config.d_ff
            low_rank_factor = 32
            parameter_ratio = (d_model + d_ff) * low_rank_factor / (d_model * d_ff)
            model = substitute_fsgpt_moe_expert_weights(
                model,
                use_svd=False,
                parameter_ratio=parameter_ratio,
                encoder_compression_layers=list(range(3, 12, 2)),
                decoder_compression_layers=list(range(1, 12, 2)),
            )
            model.load_state_dict(torch.load(os.path.join(losparse_checkpoint, "pytorch_model.bin")))
        for layer_idx in range(1, 12, 2):
            model.encoder.block[layer_idx].layer[-1].mlp.router.expert_capacity = 2048
            model.decoder.block[layer_idx].layer[-1].mlp.router.expert_capacity = 2048
    else:
        model = FSGPTMoEForCausalLM.from_pretrained(
            checkpoint,
            device_map="auto",
            moe_eval_capacity_token_fraction=1.0
        )
    tokenizer = GPT2TokenizerFast.from_pretrained(checkpoint)
    dataloader = prepare_dataloader(task, tokenizer, batch_size)
    batch = next(iter(dataloader))
    batch = {k: v.cuda() for k, v in batch.items()}

    print("Start profiling...")
    model = model.bfloat16()
    model = ModelWrapper(model)

    with torch.no_grad(), torch.autocast("cuda"):
        flops, macs, params = calculate_flops(
            model=model,
            kwargs=batch,
            print_detailed=True,
            include_backPropagation=False,
            is_sparse=True
        )
    print(f"Task: {task}.")
    print(f"Checkpoint: {checkpoint}.")
    print(f"Batch size: {batch_size}.")
    print(f"Input sequence length: {batch['input_ids'].shape[1]}")
    print(f"Output sequence length: {batch['labels'].shape[1]}")
    print(f"FLOPs: {flops}. MACs: {macs}. Params: {params}.")


if __name__ == '__main__':
    Fire(profile_inference_flops)
