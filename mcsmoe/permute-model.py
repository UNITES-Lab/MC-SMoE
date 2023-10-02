import os
from typing import Optional

from accelerate.utils import set_seed
from datasets import load_dataset
from fire import Fire
from torch.utils.data import DataLoader
from transformers import (
    SwitchTransformersForConditionalGeneration as HFSwitch,
    AutoTokenizer
)

from mcsmoe.data import (
    Seq2SeqDataPreProcessor,
    tokenize_seq2seq,
    TASK_MAPPING_DATASET_ARGUMENTS,
    DataCollatorForSeq2Seq
)
from mcsmoe.merging.permutation import (
    align_switch_permutation_for_all_experts_by_weight_matching,
    align_switch_permutation_for_all_experts_by_activation_matching
)

set_seed(2333)


def align_switch_model_by_weight_matching_permutation(
        checkpoint: str,
        save_dir: str,
        include_wo: Optional[bool] = True,
):
    model = HFSwitch.from_pretrained(checkpoint).cuda()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = align_switch_permutation_for_all_experts_by_weight_matching(model, include_wo=include_wo).cpu()
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def align_switch_model_by_activation_matching_permutation(
        checkpoint: str,
        save_dir: str,
        task: str,
        num_samples: int,
        batch_size: Optional[int] = 128,
):
    if batch_size > num_samples:
        print(f"batch_size {batch_size} is larger than num_samples {num_samples}, "
              f"set batch_size to num_samples")
        batch_size = num_samples

    model = HFSwitch.from_pretrained(checkpoint).cuda()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           max_length=tokenizer.model_max_length,
                                           return_tensors='pt')
    raw_dataset = load_dataset(*TASK_MAPPING_DATASET_ARGUMENTS[task])
    train_dataset = raw_dataset["train"]
    train_dataset = train_dataset.map(
        Seq2SeqDataPreProcessor(benchmark=task),
        batched=True,
        num_proc=8,
        remove_columns=train_dataset.column_names
    )
    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenize_seq2seq(tokenizer=tokenizer, batch=x, keep_other_keys=False),
        num_proc=8,
        batched=True,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False
    )
    dataset_for_permutation = tokenized_train_dataset.shuffle(seed=2333).select(range(num_samples))
    dataloader_for_permutation = DataLoader(
        dataset_for_permutation,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=batch_size,
        num_workers=3
    )
    model = align_switch_permutation_for_all_experts_by_activation_matching(model, dataloader_for_permutation).cpu()
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def main(
        checkpoint: str,
        save_dir: str,
        strategy: str,
        task: Optional[str] = None,
        num_samples: Optional[int] = None,
        batch_size: Optional[int] = 128,
        include_wo: Optional[bool] = True
):
    if os.path.isdir(save_dir):
        raise ValueError(f"save_dir {save_dir} already exists")
    if strategy == 'weight':
        align_switch_model_by_weight_matching_permutation(checkpoint, save_dir, include_wo)
    elif strategy == 'activation':
        align_switch_model_by_activation_matching_permutation(checkpoint, save_dir, task, num_samples, batch_size)
    else:
        raise ValueError(f"strategy {strategy} is not supported")


if __name__ == '__main__':
    Fire(main)
