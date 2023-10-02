import os
import random
from typing import Optional

import torch
from tqdm import tqdm
from transformers import (
    SwitchTransformersForConditionalGeneration,
    SwitchTransformersSparseMLP
)

from mcsmoe.models.fsgpt_moe import (
    FSGPTMoESparseMLP,
    FSGPTMoEForCausalLM
)


def _merge_switch_expert_module(
        mlp: SwitchTransformersSparseMLP,
        group_labels: torch.Tensor,
) -> SwitchTransformersSparseMLP:
    for label in group_labels.unique():
        indices = torch.where(group_labels == label)[0]
        if len(indices) == 1:
            continue
        # merge experts
        for idx in indices[1:]:
            mlp.experts[f"expert_{idx}"] = mlp.experts[f"expert_{indices[0]}"]
            # Despite only one expert within a group has the right weights,
            # this is supposed to be correct since the nn.ModuleDict always keep the largest index as the true expert.
            # (i.e. The equal sign is not that normal function here.)
            # In this way, as long as we assign the largest index in the end, we are good.
    return mlp


def load_merged_switch_transformers_from_checkpoint(
        checkpoint: str,
        group_state_checkpoint: Optional[str] = None,
        sanity_check: Optional[bool] = False,
) -> SwitchTransformersForConditionalGeneration:
    if group_state_checkpoint is None:
        group_state_checkpoint = os.path.join(checkpoint, "group_state_dict.pt")
    model = SwitchTransformersForConditionalGeneration.from_pretrained(checkpoint)
    state_dict = model.state_dict()
    group_state_dict = torch.load(group_state_checkpoint, map_location="cpu")

    # merge experts
    config = model.config
    sparse_layer_indices = list(range(1, config.num_layers, config.encoder_sparse_step))
    for layer_idx in tqdm(sparse_layer_indices, desc="Merging expert modules"):
        # Merge encoder
        encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
        encoder_mlp = model.encoder.block[layer_idx].layer[-1].mlp
        group_labels = group_state_dict[encoder_mlp_name]
        model.encoder.block[layer_idx].layer[-1].mlp = _merge_switch_expert_module(encoder_mlp, group_labels)
        # Merge decoder
        decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
        decoder_mlp = model.decoder.block[layer_idx].layer[-1].mlp
        group_labels = group_state_dict[decoder_mlp_name]
        model.decoder.block[layer_idx].layer[-1].mlp = _merge_switch_expert_module(decoder_mlp, group_labels)

    if sanity_check:
        for name, param in tqdm(state_dict.items(), desc="Sanity checking"):
            if not torch.allclose(param, model.state_dict()[name]):
                raise ValueError(f"Parameter {name} is not merged correctly.")

    model.group_state_dict = group_state_dict
    return model


def _merge_fsgpt_expert_module(
        ffn: FSGPTMoESparseMLP,
        group_labels: torch.Tensor,
) -> FSGPTMoESparseMLP:
    for label in group_labels.unique():
        indices = torch.where(group_labels == label)[0]
        if len(indices) == 1:
            continue
        # merge experts
        for idx in indices[1:]:
            ffn.experts[f"expert_{idx}"] = ffn.experts[f"expert_{indices[0]}"]
            # Despite only one expert within a group has the right weights,
            # this is supposed to be correct since the nn.ModuleDict always keep the largest index as the true expert.
            # (i.e. The equal sign is not that normal function here.)
            # In this way, as long as we assign the largest index in the end, we are good.
    return ffn


def load_merged_fsgpt_moe_from_checkpoint(
        checkpoint: str,
        group_state_checkpoint: Optional[str] = None,
        sanity_check: Optional[bool] = False,
) -> FSGPTMoEForCausalLM:
    if group_state_checkpoint is None:
        group_state_checkpoint = os.path.join(checkpoint, "group_state_dict.pt")
    model = FSGPTMoEForCausalLM.from_pretrained(
        checkpoint,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    state_dict = model.state_dict() if sanity_check else None
    group_state_dict = torch.load(group_state_checkpoint, map_location="cpu")

    # merge experts
    config = model.config
    sparse_layer_indices = list(range(1, config.num_layers, config.sparse_step))
    for layer_idx in tqdm(sparse_layer_indices, desc="Merging expert modules"):
        ffn_name = f"decoder.layers.{layer_idx}.ffn"
        ffn = model.decoder.layers[layer_idx].ffn
        group_labels = group_state_dict[ffn_name]
        model.decoder.layers[layer_idx].ffn = _merge_fsgpt_expert_module(ffn, group_labels)

    if sanity_check:
        for name, param in tqdm(state_dict.items(), desc="Sanity checking"):
            if not torch.allclose(param, model.state_dict()[name]):
                raise ValueError(f"Parameter {name} is not merged correctly.")

    model.group_state_dict = group_state_dict
    return model


def generate_random_group_labels(
        num_experts: int,
        num_groups: int,
) -> torch.Tensor:
    """
    Assign random group labels to experts, with each group has at least one expert.

    Examples
    --------
    >>> generate_random_group_labels(10, 3).unique().sort()[0]
    tensor([0, 1, 2])
    """
    group_labels = torch.zeros(num_experts, dtype=torch.long)
    for i in range(num_groups):
        group_labels[i] = i
    for i in range(num_groups, num_experts):
        group_labels[i] = random.randint(0, num_groups - 1)
    return group_labels[torch.randperm(num_experts)]
