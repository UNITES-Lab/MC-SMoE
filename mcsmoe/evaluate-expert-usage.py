import torch
import transformers.utils.logging as hf_logging
from accelerate.utils import set_seed
from datasets import load_dataset
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    T5TokenizerFast,
    SwitchTransformersForConditionalGeneration as HFSwitch
)

from mcsmoe.data import (
    Seq2SeqDataPreProcessor,
    tokenize_seq2seq,
    TASK_MAPPING_DATASET_ARGUMENTS,
    DataCollatorForSeq2Seq
)

set_seed(42)
hf_logging.set_verbosity_info()


def evaluate_downstream_usage(
        checkpoint: str = None,
        task: str = None,
):

    tokenizer = T5TokenizerFast.from_pretrained("google/switch-base-32")
    model = HFSwitch.from_pretrained(checkpoint)

    raw_dataset = load_dataset(*TASK_MAPPING_DATASET_ARGUMENTS[task])
    dataset = raw_dataset["train"].select(range(min(256, len(raw_dataset["train"]))))
    dataset = dataset.map(
        Seq2SeqDataPreProcessor(benchmark=task),
        batched=True,
        num_proc=8,
        remove_columns=dataset.column_names
    )
    tokenized_dataset = dataset.map(
        lambda x: tokenize_seq2seq(tokenizer=tokenizer, batch=x, keep_other_keys=False),
        num_proc=8,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           max_length=tokenizer.model_max_length,
                                           return_tensors='pt')
    dataloader = DataLoader(
        tokenized_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=8,
        num_workers=8
    )
    expert_usage = {}

    for layer_idx in tqdm(range(1, 12, 2), desc="Registering forward hook..."):
        encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
        decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
        expert_usage[encoder_mlp_name] = torch.zeros(32)
        expert_usage[decoder_mlp_name] = torch.zeros(32)

    model.eval()
    model.cuda()
    for batch in tqdm(dataloader, desc="Evaluating experts..."):
        with torch.no_grad():
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            for layer_idx in range(1, 12, 2):
                encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
                decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
                encoder_router_expert_index = outputs.encoder_router_logits[layer_idx][1].reshape(-1)
                decoder_router_expert_index = outputs.decoder_router_logits[layer_idx][1].reshape(-1)
                for idx in encoder_router_expert_index:
                    expert_usage[encoder_mlp_name][idx] += 1
                for idx in decoder_router_expert_index:
                    expert_usage[decoder_mlp_name][idx] += 1

    for key, item in expert_usage.items():
        expert_usage[key] = item / item.sum()

    torch.save(expert_usage, f"./visualization/{task}/expert-usage.pt")


def main():
    task_to_checkpoint = {
        "sst2": "vanilla-32e-acc95.75",
        "mrpc": "vanilla-32e-acc90.20",
        "copa": "vanilla-32e-acc68.00",
        "multirc": "vanilla-32e-f1a76.19",
        "squad": "vanilla-32e-em65.39",
        "wikiqa": "vanilla-32e-acc96.45",
        "winogrande": "vanilla-32e-acc61.80"
    }
    for task, checkpoint in task_to_checkpoint.items():
        print(f"Evaluating {task}...")
        evaluate_downstream_usage(
            checkpoint=f"./results/{task}/{checkpoint}",
            task=task
        )


if __name__ == '__main__':
    Fire(main)
