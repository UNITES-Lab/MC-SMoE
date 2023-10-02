from typing import Optional

import torch
from accelerate.utils import set_seed
from fire import Fire
from transformers import (
    AutoTokenizer
)

from mcsmoe.merging.permutation import (
    align_fsgpt_permutation_for_all_experts_by_weight_matching
)
from mcsmoe.models import (
    FSGPTMoEForCausalLM
)

set_seed(2333)


def align_fsgpt_model_by_weight_matching_permutation(
        checkpoint: str,
        save_dir: str,
        include_wo: Optional[bool] = True,
):
    model = FSGPTMoEForCausalLM.from_pretrained(
        checkpoint,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        moe_eval_capacity_token_fraction=1.0,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = align_fsgpt_permutation_for_all_experts_by_weight_matching(model, include_wo=include_wo).cpu()
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == '__main__':
    Fire(align_fsgpt_model_by_weight_matching_permutation)
