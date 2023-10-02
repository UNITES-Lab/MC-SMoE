import os
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, Optional, List, Union, Tuple, Callable, Iterator

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    PretrainedConfig
)
from transformers.modeling_outputs import MoECausalLMOutputWithPast

from mcsmoe.merging.permutation import (
    permute_fsgpt_ffn_dense_expert_,
    compute_fsgpt_permutation_by_weight_matching,
    compute_switch_permutation_by_activation_matching
)
from mcsmoe.models.fsgpt_moe import (
    FSGPTMoEConfig,
    FSGPTMoEForCausalLM,
    FSGPTMoESparseMLP,
    FSGPTMoETop2Router
)
from mcsmoe.utils.constants import BF16_EPS
from .grouping import SIMILARITY_MAPPING_FUNCTION, LEGAL_SIMILARITY_BASES
from .utils import generate_random_group_labels


class ExpertsGrouperForFSGPT(object):
    def __init__(
            self,
            config: Union[FSGPTMoEConfig, PretrainedConfig],
            similarity_fn: str = "cosine",
            similarity_base: str = "router-logits",
    ):
        if similarity_fn not in SIMILARITY_MAPPING_FUNCTION:
            raise ValueError(
                f"[Merging]similarity_fn should be one of {SIMILARITY_MAPPING_FUNCTION.keys()}, got {similarity_fn} instead.")
        if similarity_base not in LEGAL_SIMILARITY_BASES:
            raise ValueError(
                f"[Merging]similarity_base should be one of {LEGAL_SIMILARITY_BASES}, got {similarity_base} instead.")

        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.sparse_layer_indices = list(range(1, config.num_layers, config.sparse_step))
        self.similarity_fn = SIMILARITY_MAPPING_FUNCTION[similarity_fn]
        self.similarity_base = similarity_base
        self._group_state_dict = None
        self._similarity_state_dict = None
        self._usage_frequency_state_dict = None
        self.reset_all()

    def reset_all(self):
        if self.similarity_base == "mse":
            self.similarity_fn = SIMILARITY_MAPPING_FUNCTION["mse"]
            print("[Merging]Set similarity_fn to mse for mse similarity_base.")
        self._group_state_dict = dict()
        self._similarity_state_dict = dict()
        self._usage_frequency_state_dict = dict()
        # Similarity range: [0, 1]
        for layer_idx in self.sparse_layer_indices:
            ffn_name = f"decoder.layers.{layer_idx}.ffn"
            self._group_state_dict[ffn_name] = torch.arange(self.num_experts,
                                                            device="cuda")
            self._similarity_state_dict[ffn_name] = torch.zeros(
                (self.num_experts, self.num_experts), device="cuda"
            ) + torch.eye(self.num_experts, device="cuda")
            self._usage_frequency_state_dict[ffn_name] = torch.ones(self.num_experts, device="cuda") / self.num_experts

    def similarity_state_dict(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self._similarity_state_dict)

    def group_state_dict(self) -> Dict[str, torch.LongTensor]:
        return deepcopy(self._group_state_dict)

    def usage_frequency_state_dict(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self._usage_frequency_state_dict)

    def save_similarity(self, ffn_name: str, i: int, j: int, similarity: float):
        self._similarity_state_dict[ffn_name][i, j] = similarity
        self._similarity_state_dict[ffn_name][j, i] = similarity

    def get_similarity(self, ffn_name: str, i: int, j: int) -> float:
        return self._similarity_state_dict[ffn_name][i, j].item()

    def get_similarity_matrix(self, ffn_name: str) -> torch.Tensor:
        return deepcopy(self._similarity_state_dict[ffn_name])

    def save_group_state_dict(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._group_state_dict, os.path.join(save_dir, "group_state_dict.pt"))

    def load_group_state_dict(self, load_dir: str):
        self._group_state_dict = torch.load(os.path.join(load_dir, "group_state_dict.pt"))

    def _assign_num_groups_per_layer(
            self,
            average_num_groups: int,
            merging_layers: List[int],
    ) -> Dict[str, int]:
        num_grouping_layers = len(merging_layers)
        total_num_groups = average_num_groups * num_grouping_layers + self.num_experts * (
                len(self.sparse_layer_indices) - num_grouping_layers
        )
        all_usage_frequency = []
        usage_frequency_dict = deepcopy(self._usage_frequency_state_dict)
        for i, layer_idx in enumerate(self.sparse_layer_indices):
            # 1. Experts in the excluded layers are always not merged.
            if layer_idx not in merging_layers:
                usage_frequency_dict[f"decoder.layers.{layer_idx}.ffn"] = torch.ones_like(
                    usage_frequency_dict[f"decoder.layers.{layer_idx}.ffn"]
                )
            # 2. Each layer must have at least one group, set the most used expert in a layer to frequency 1.
            max_usage_index = torch.argmax(usage_frequency_dict[f"decoder.layers.{layer_idx}.ffn"])
            usage_frequency_dict[f"decoder.layers.{layer_idx}.ffn"][max_usage_index] = 1.0
            # 3. Collect all usage frequency.
            all_usage_frequency.append(usage_frequency_dict[f"decoder.layers.{layer_idx}.ffn"])
        all_usage_frequency = torch.cat(all_usage_frequency, dim=0)
        sorted_usage_frequency, sorted_indices = torch.sort(all_usage_frequency, descending=True)
        num_groups_per_layer = dict()

        # Note: When threshold is 0.0, the actual number of groups is smaller than total_num_groups.
        if average_num_groups == self.num_experts:
            total_num_groups = total_num_groups - 1
        frequency_threshold = sorted_usage_frequency[total_num_groups]

        print(f"[Merging]Frequency threshold: {frequency_threshold}")
        if frequency_threshold == 1.0:
            raise ValueError("[Merging]The number of groups is too large, please reduce the number of groups.")
        for i, layer_idx in enumerate(self.sparse_layer_indices):
            num_groups_per_layer[f"decoder.layers.{layer_idx}.ffn"] = torch.sum(
                (usage_frequency_dict[f"decoder.layers.{layer_idx}.ffn"] > frequency_threshold).long()
            ).item()
        return num_groups_per_layer

    def group_experts_into_clusters_by_routing_guided_globally(
            self,
            average_num_groups: int,
            merging_layers: List[int],
            group_capacity: Optional[int] = None,
    ) -> Dict[str, List[int]]:
        """
        Globally group experts into clusters by routing-guided clustering, each layer will have different number of
         clusters. The total number of clusters is determined by average_num_groups.

        Parameters
        ----------
        average_num_groups: int
            The average number of clusters for all layers.
        merging_layers: List[int]
            The layers that are excluded from merging.
        group_capacity: Optional[int]
            The maximum number of experts in each group. If None, the number of experts in each group is not limited.

        Returns
        -------
        core_experts: Dict[str, List[int]]
            The core experts of each cluster
        """
        # 1. Assign num_groups respectively for each layer according to average_num_groups
        group_capacity = group_capacity if group_capacity is not None else self.num_experts
        num_groups_per_layer = self._assign_num_groups_per_layer(
            average_num_groups, merging_layers
        )
        print(f"[Merging]Number of groups of each layer: {num_groups_per_layer}")
        # 2. Group experts into clusters for each layer
        core_experts = dict()
        for layer_idx in tqdm(self.sparse_layer_indices,
                              desc=f"Globally routing-guided clustering experts into average {average_num_groups} clusters"):
            ffn_name = f"decoder.layers.{layer_idx}.ffn"
            num_groups = num_groups_per_layer[ffn_name]
            group_member_count = torch.zeros(num_groups)
            indices_sorted_by_usage = torch.argsort(self._usage_frequency_state_dict[ffn_name], descending=True)
            # 1. Assign top-K most-used experts with label 0 to K-1 respectively
            core_expert_indices = indices_sorted_by_usage[:num_groups]
            core_experts[ffn_name] = core_expert_indices.tolist()
            for i in range(num_groups):
                self._group_state_dict[ffn_name][indices_sorted_by_usage[i]] = i
                group_member_count[i] += 1
            # 2. Assign left unassigned experts to the cluster with the most similar core
            similarity_matrix = self.get_similarity_matrix(ffn_name)
            for i in range(num_groups, self.num_experts):
                # Find the most similar core
                most_similar_core = core_expert_indices[
                    torch.argmax(similarity_matrix[i, core_expert_indices])
                ]
                most_similar_group_label = self._group_state_dict[ffn_name][most_similar_core]
                self._group_state_dict[ffn_name][i] = most_similar_group_label
                if group_member_count[self._group_state_dict[ffn_name][i]] >= group_capacity:
                    if len(core_expert_indices) == 1:
                        raise ValueError(
                            f"[Merging]The number of groups at Decoder layer {layer_idx} is too small!"
                        )
                    # Kick out the filled group as well as its core, by pop the core from core_experts
                    core_index = torch.argmax(similarity_matrix[i, core_expert_indices])
                    core_expert_indices = torch.cat(
                        [core_expert_indices[:core_index], core_expert_indices[core_index + 1:]]
                    )
        return core_experts

    def group_experts_into_clusters_by_routing_guided(
            self,
            num_groups: int,
    ) -> Dict[str, List[int]]:
        """
        1. Select top-K most-used experts, assign them with label 0 to K-1 respectively.
            These K experts are the cores of each cluster
        2. For each left unassigned expert, assign it to the cluster with the most similar core.

        Returns
        -------
        core_experts: Dict[str, List[int]]
            The core experts of each cluster
        """
        core_experts = {}
        if num_groups > self.num_experts:
            raise ValueError(
                f"[Merging]num_groups should be less than or equal to {self.num_experts}, got {num_groups} instead.")

        for layer_idx in tqdm(self.sparse_layer_indices,
                              desc=f"Routing-guided clustering experts into {num_groups} clusters"):
            # Decoder
            ffn_name = f"decoder.layers.{layer_idx}.ffn"
            indices_sorted_by_usage = torch.argsort(self._usage_frequency_state_dict[ffn_name], descending=True)
            # 1. Assign top-K most-used experts with label 0 to K-1 respectively
            core_experts[ffn_name] = indices_sorted_by_usage[:num_groups].tolist()
            for i in range(num_groups):
                self._group_state_dict[ffn_name][indices_sorted_by_usage[i]] = i
            # 2. Assign left unassigned experts to the cluster with the most similar core
            similarity_matrix = self.get_similarity_matrix(ffn_name)
            for i in range(num_groups, self.num_experts):
                # Find the most similar core
                most_similar_core = indices_sorted_by_usage[
                    torch.argmax(similarity_matrix[i, indices_sorted_by_usage[:num_groups]])
                ]
                self._group_state_dict[ffn_name][i] = self._group_state_dict[ffn_name][
                    most_similar_core]
        return core_experts

    def compute_all_usages(
            self,
            model: FSGPTMoEForCausalLM,
            dataloader: DataLoader,
    ):
        model.eval()
        # reset all usage frequency to 0
        self._usage_frequency_state_dict = {
            k: torch.zeros_like(v) for k, v in self._usage_frequency_state_dict.items()
        }
        for batch in tqdm(dataloader, desc="[Merging]Computing all usages..."):
            with torch.no_grad():
                with torch.autocast("cuda"):
                    batch = {k: v.cuda() for k, v in batch.items()}
                    outputs = model(**batch)
                    for layer_idx in self.sparse_layer_indices:
                        ffn_name = f"decoder.layers.{layer_idx}.ffn"
                        decoder_router_probs = outputs.router_logits[layer_idx][1]
                        decoder_router_expert_index = torch.argsort(decoder_router_probs)[:, -2:].reshape(-1)
                        for idx in decoder_router_expert_index:
                            self._usage_frequency_state_dict[ffn_name][idx] += 1
        self._usage_frequency_state_dict = {
            k: v / torch.sum(v) for k, v in self._usage_frequency_state_dict.items()
        }

    def compute_all_similarities(
            self,
            model: FSGPTMoEForCausalLM,
            dataloader: DataLoader = None,
    ):
        if self.similarity_base not in ["weight", "router-weight"] and dataloader is None:
            raise ValueError(
                "[Merging]Dataloader should be provided when similarity_base is not 'weight' or 'router-weight'")

        model = model.eval()
        if self.similarity_base == 'router-logits':
            self._compute_all_similarities_by_router_logits(model, dataloader)
        elif self.similarity_base == 'random':
            self._compute_all_similarities_by_random()
        else:
            raise NotImplementedError

    def _compute_all_similarities_by_random(self):
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[Merging]Computing similarities randomly..."):
            ffn_name = f"decoder.layers.{layer_idx}.ffn"
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    self.save_similarity(ffn_name, i, j, torch.rand(1).item())

    def _compute_all_similarities_by_router_logits(
            self, model: FSGPTMoEForCausalLM, dataloader: DataLoader
    ):
        router_logits_outputs_list = []
        for batch in dataloader:
            with torch.no_grad():
                with torch.autocast("cuda"):
                    batch = {k: v.cuda() for k, v in batch.items()}
                    outputs = model(**batch)
                    router_logits_outputs_list.append(outputs.router_logits)
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[Merging]Computing similarities by router logits..."):
            ffn_name = f"decoder.layers.{layer_idx}.ffn"
            router_logits_list = [
                router_logits[layer_idx][0].reshape(
                    -1, self.num_experts) for router_logits in router_logits_outputs_list
            ]
            router_logits = torch.cat(router_logits_list, dim=0)
            with torch.no_grad():
                for i in range(self.num_experts):
                    for j in range(i + 1, self.num_experts):
                        # Decoder
                        i_flat = router_logits[:, i].flatten()
                        j_flat = router_logits[:, j].flatten()
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(ffn_name, i, j, similarity)

    def group_experts_randomly(
            self,
            num_groups: int,
    ):
        for layer_idx in tqdm(self.sparse_layer_indices,
                              desc=f"Randomly merging experts into {num_groups} clusters"):
            ffn_name = f"decoder.layers.{layer_idx}.ffn"
            group_labels = generate_random_group_labels(self.num_experts, num_groups)
            self._group_state_dict[ffn_name] = group_labels


def _merge_ffn_experts_by_usage_frequency_weighting(
        ffn: FSGPTMoESparseMLP,
        group_labels: torch.LongTensor,
        usage_frequencies: torch.Tensor,
) -> FSGPTMoESparseMLP:
    assert len(group_labels) == len(usage_frequencies) == len(ffn.experts)
    for label in group_labels.unique():
        expert_indices = torch.where(group_labels == label)[0]
        with torch.no_grad():
            fc1_weight_list = torch.stack(
                [ffn.experts[f"expert_{expert_idx}"].fc1.weight * usage_frequencies[expert_idx] for expert_idx in
                 expert_indices], dim=0
            )
            fc2_weight_list = torch.stack(
                [ffn.experts[f"expert_{expert_idx}"].fc2.weight * usage_frequencies[expert_idx] for expert_idx in
                 expert_indices], dim=0
            )
            fc1_bias_list = torch.stack(
                [ffn.experts[f"expert_{expert_idx}"].fc1.bias * usage_frequencies[expert_idx] for expert_idx in
                 expert_indices], dim=0
            )
            fc2_bias_list = torch.stack(
                [ffn.experts[f"expert_{expert_idx}"].fc2.bias * usage_frequencies[expert_idx] for expert_idx in
                 expert_indices], dim=0
            )
            fc1_weight = torch.sum(fc1_weight_list, dim=0) / (
                    torch.sum(usage_frequencies[expert_indices], dim=0) + BF16_EPS)
            fc1_bias = torch.sum(fc1_bias_list, dim=0) / (
                    torch.sum(usage_frequencies[expert_indices], dim=0) + BF16_EPS)
            fc2_weight = torch.sum(fc2_weight_list, dim=0) / (
                    torch.sum(usage_frequencies[expert_indices], dim=0) + BF16_EPS)
            fc2_bias = torch.sum(fc2_bias_list, dim=0) / (
                    torch.sum(usage_frequencies[expert_indices], dim=0) + BF16_EPS)
            ffn.experts[f"expert_{expert_indices[0]}"].fc1.weight.copy_(fc1_weight)
            ffn.experts[f"expert_{expert_indices[0]}"].fc1.bias.copy_(fc1_bias)
            ffn.experts[f"expert_{expert_indices[0]}"].fc2.weight.copy_(fc2_weight)
            ffn.experts[f"expert_{expert_indices[0]}"].fc2.bias.copy_(fc2_bias)
            for expert_idx in expert_indices[1:]:
                # Binding merged experts to the first of them
                ffn.experts[f"expert_{expert_idx}"] = ffn.experts[f"expert_{expert_indices[0]}"]
    return ffn


def merge_by_groups_with_usage_frequency_weighting(
        model: FSGPTMoEForCausalLM,
        grouper: ExpertsGrouperForFSGPT,
        merging_layers: Optional[List[int]] = None,
) -> FSGPTMoEForCausalLM:
    usage_frequency_dict = grouper.usage_frequency_state_dict()
    for layer_idx in tqdm(
            grouper.sparse_layer_indices,
            desc=f"[Merging]Merging experts with usage-frequency-weighted averaging..."
    ):
        if merging_layers is None or layer_idx in merging_layers:
            ffn_name = f"decoder.layers.{layer_idx}.ffn"
            group_labels = grouper.group_state_dict()[ffn_name]
            usage_frequencies = usage_frequency_dict[ffn_name]
            model.decoder.layers[layer_idx].ffn = _merge_ffn_experts_by_usage_frequency_weighting(
                ffn=model.decoder.layers[layer_idx].ffn,
                group_labels=group_labels,
                usage_frequencies=usage_frequencies,
            )
    return model


def _merge_ffn_experts_by_averaging(
        ffn: FSGPTMoESparseMLP,
        group_labels: torch.LongTensor,
        permute: bool,
        permute_strategy: str,
        forwarded_hidden_states: Optional[Tuple[torch.Tensor]] = None,
) -> FSGPTMoESparseMLP:
    for label in group_labels.unique():
        expert_indices = torch.where(group_labels == label)[0]
        if permute and permute_strategy == "weight-matching":
            for expert_idx in expert_indices[1:]:
                perm = compute_fsgpt_permutation_by_weight_matching(
                    reference_ffn=ffn.experts[f"expert_{expert_indices[0]}"],
                    target_ffn=ffn.experts[f"expert_{expert_idx}"],
                    include_wo=True
                )
                ffn.experts[f"expert_{expert_idx}"] = permute_fsgpt_ffn_dense_expert_(
                    ffn.experts[f"expert_{expert_idx}"], perm
                )
        elif permute and permute_strategy == "activation-matching":
            group_forwarded_hidden_states = torch.cat([
                forwarded_hidden_states[expert_idx] for expert_idx in expert_indices
            ], dim=0)
            for expert_idx in expert_indices[1:]:
                perm = compute_switch_permutation_by_activation_matching(
                    reference_mlp=ffn.experts[f"expert_{expert_indices[0]}"],
                    target_mlp=ffn.experts[f"expert_{expert_idx}"],
                    forwarded_hidden_states=group_forwarded_hidden_states,
                )
                ffn.experts[f"expert_{expert_idx}"] = permute_fsgpt_ffn_dense_expert_(
                    ffn.experts[f"expert_{expert_idx}"], perm
                )
        elif permute:
            raise ValueError(f"Unknown permute strategy: {permute_strategy}")

        with torch.no_grad():
            fc1_weight = torch.mean(
                torch.stack([ffn.experts[f"expert_{expert_idx}"].fc1.weight for expert_idx in expert_indices]),
                dim=0
            )
            fc2_weight = torch.mean(
                torch.stack([ffn.experts[f"expert_{expert_idx}"].fc2.weight for expert_idx in expert_indices]),
                dim=0
            )
            fc1_bias = torch.mean(
                torch.stack([ffn.experts[f"expert_{expert_idx}"].fc1.bias for expert_idx in expert_indices]),
                dim=0
            )
            fc2_bias = torch.mean(
                torch.stack([ffn.experts[f"expert_{expert_idx}"].fc2.bias for expert_idx in expert_indices]),
                dim=0
            )
            ffn.experts[f"expert_{expert_indices[0]}"].fc1.weight.copy_(fc1_weight)
            ffn.experts[f"expert_{expert_indices[0]}"].fc2.weight.copy_(fc2_weight)
            ffn.experts[f"expert_{expert_indices[0]}"].fc1.bias.copy_(fc1_bias)
            ffn.experts[f"expert_{expert_indices[0]}"].fc2.bias.copy_(fc2_bias)
            for expert_idx in expert_indices[1:]:
                # Binding merged experts to the first of them
                ffn.experts[f"expert_{expert_idx}"] = ffn.experts[f"expert_{expert_indices[0]}"]
    return ffn


def merge_by_groups(
        model: FSGPTMoEForCausalLM,
        grouper: ExpertsGrouperForFSGPT,
        merging_layers: Optional[List[int]] = None,
        permute: Optional[bool] = False,
        permute_strategy: Optional[str] = "weight-matching",
        dataloader: Optional[DataLoader] = None,
) -> FSGPTMoEForCausalLM:
    forwarded_hidden_states = dict()
    router_indices = dict()
    if permute_strategy == "activation-matching":
        model.eval().cuda()
        handles = []

        def _get_activation_hook(name):
            def hook(module, input, output):
                forwarded_hidden_states[name].append(input[0].detach().reshape(-1, input[0].shape[-1]))

            return hook

        for layer_idx in tqdm(
                grouper.sparse_layer_indices,
                desc=f"[Merging]Registering forward hook..."
        ):
            fff_name = f"decoder.layers.{layer_idx}.ffn"
            forwarded_hidden_states[fff_name] = []
            handles.append(model.decoder.layers[layer_idx].ffn.register_forward_hook(
                _get_activation_hook(fff_name))
            )

        # {name: values}, values will be of shape (len(dataloader), batch_size * seq_len)
        router_indices = {name: [] for name in forwarded_hidden_states.keys()}
        with torch.no_grad(), torch.autocast("cuda"):
            for batch in tqdm(dataloader, desc="[Merging]Computing activations..."):
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(**batch)
                for layer_idx in grouper.sparse_layer_indices:
                    router_indices[f"decoder.layers.{layer_idx}.ffn"].append(
                        outputs.router_logits[layer_idx][2].reshape(-1)
                    )

        for handle in handles:
            handle.remove()

    num_experts = grouper.num_experts

    if permute:
        print(f"[Merging]Permutation is enabled, will permute experts by {permute_strategy} in the same group.")
    for layer_idx in tqdm(grouper.sparse_layer_indices,
                          desc="[Merging]Merging experts with averaging..."):
        if merging_layers is None or layer_idx in merging_layers:
            fff_name = f"decoder.layers.{layer_idx}.ffn"
            group_labels = grouper.group_state_dict()[fff_name]
            if permute_strategy == "activation-matching":
                layer_forwarded_hidden_states = tuple()
                for expert_idx in range(num_experts):
                    layer_forwarded_hidden_states += (
                        torch.cat(
                            [forwarded_hidden_states[fff_name][i][
                                 router_indices[fff_name][i] == expert_idx]
                             for i in range(len(dataloader))], dim=0),
                    )
                model.decoder.layers[layer_idx].ffn = _merge_ffn_experts_by_averaging(
                    ffn=model.decoder.layers[layer_idx].ffn,
                    group_labels=group_labels,
                    permute=permute,
                    permute_strategy=permute_strategy,
                    forwarded_hidden_states=layer_forwarded_hidden_states
                )
            else:
                model.decoder.layers[layer_idx].ffn = _merge_ffn_experts_by_averaging(
                    ffn=model.decoder.layers[layer_idx].ffn,
                    group_labels=group_labels,
                    permute=permute,
                    permute_strategy=permute_strategy,
                )
    return model


def _prune_ffn_experts_by_dropping_non_core(
        ffn: FSGPTMoESparseMLP,
        core_expert_indices: List[int],
) -> FSGPTMoESparseMLP:
    non_core_expert_mask = torch.ones(len(ffn.experts), dtype=torch.bool)
    non_core_expert_mask[core_expert_indices] = False
    non_core_expert_indices = torch.where(non_core_expert_mask)[0]
    bf16_mask_value = -1e4

    def _custom_forward(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.LongTensor] = None) -> Tuple:
        self.input_dtype = hidden_states.dtype
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape((batch_size * sequence_length), hidden_dim)
        hidden_states = hidden_states.to(self.dtype)
        self._cast_classifier()
        router_logits = self.classifier(hidden_states)

        router_logits = router_logits.masked_fill(non_core_expert_mask.to(router_logits.device), bf16_mask_value)

        top_1_mask, router_probs = self.route_tokens(router_logits, self.input_dtype, padding_mask)
        return top_1_mask, router_probs, router_logits

    ffn.router.forward = _custom_forward.__get__(ffn.router, FSGPTMoETop2Router)

    for expert_idx in non_core_expert_indices:
        # fake expert, this will raise error if used by mistake
        ffn.experts[f"expert_{expert_idx}"] = torch.nn.Linear(ffn.router.classifier.in_features, 1)

    return ffn


def prune_non_core_experts_by_groups(
        model: FSGPTMoEForCausalLM,
        grouper: ExpertsGrouperForFSGPT,
        core_experts: Dict[str, List[int]],
        merging_layers: Optional[List[int]] = None,
) -> FSGPTMoEForCausalLM:
    for layer_idx in tqdm(grouper.sparse_layer_indices, desc="[Merging]Pruning non-core experts by dropping..."):
        if merging_layers is None or layer_idx in merging_layers:
            ffn_name = f"decoder.layers.{layer_idx}.ffn"
            model.decoder.layers[layer_idx].ffn = _prune_ffn_experts_by_dropping_non_core(
                ffn=model.decoder.layers[layer_idx].ffn,
                core_expert_indices=core_experts[ffn_name]
            )
    return model


class ExpertUsageFrequencyTrackerForFSGPT(object):
    """
    A class to track the usage frequencies of experts in the model during the training process.
    """

    def __init__(
            self,
            named_parameters_caller: Callable[[], Iterator[Tuple[str, torch.nn.Parameter]]],
            beta: Optional[float] = 0.9,
            compute_every_n_steps: Optional[int] = 10,
            device: Optional[str] = 'cpu',
    ):
        """

        Parameters
        ----------
        named_parameters_caller: Callable[[], Iterator[Tuple[str, torch.nn.Parameter]]]
            Normally, it is `model.named_parameters`
        beta: Optional[float]
            The beta parameter in exponential moving average
        compute_every_n_steps: Optional[int]
            Compute usage EMA every n steps
        device: Optional[str]
            The device to store the usage frequency state dict
        """
        self.beta = beta
        self.compute_fisher_every_n_steps = compute_every_n_steps
        self.device = device
        self.exp_expert_usage_frequency_dict = None
        self.last_error = -1

        self._init_usage_frequency_dict(named_parameters_caller)
        self.num_layers = None

    def usage_frequency_state_dict(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self.exp_expert_usage_frequency_dict)

    def _init_usage_frequency_dict(self,
                                   named_parameters_caller: Callable[[], Iterator[Tuple[str, torch.nn.Parameter]]]):
        self.exp_expert_usage_frequency_dict = OrderedDict()
        for name, _ in named_parameters_caller():
            if "router.classifier" in name:
                mlp_name = name.split(".router")[0]
                self.exp_expert_usage_frequency_dict[mlp_name] = []

        for name, _ in named_parameters_caller():
            if "expert_" in name and "fc1.weight" in name:
                mlp_name = name.split(".experts.expert_")[0]
                self.exp_expert_usage_frequency_dict[mlp_name].append(1.0)

        for key, value in self.exp_expert_usage_frequency_dict.items():
            value = torch.tensor(value, device=self.device)
            # init as average used frequency
            self.exp_expert_usage_frequency_dict[key] = value / torch.sum(value)
        self.num_layers = len(self.exp_expert_usage_frequency_dict) // 2

    def _update_exp_usage_frequency_state(
            self,
            model_outputs: Union[List[MoECausalLMOutputWithPast], MoECausalLMOutputWithPast]
    ):
        usage_frequency_state_dict = OrderedDict()
        is_first_step = self.last_error < 0
        for name in self.exp_expert_usage_frequency_dict.keys():
            usage_frequency_state_dict[name] = torch.zeros_like(self.exp_expert_usage_frequency_dict[name])
            # Capture the layer index from the name
            layer_idx = int(name.split(".layers.")[1].split(".ffn")[0])
            if isinstance(model_outputs, list):
                router_expert_index = [
                    torch.argsort(outputs.router_logits[layer_idx][1])[:, -2:].reshape(-1) for outputs in model_outputs]
                router_expert_index = torch.concat(router_expert_index)
            else:
                router_expert_index = torch.argsort(model_outputs.router_logits[layer_idx][1])[:, -2:].reshape(-1)
            for idx in router_expert_index:
                usage_frequency_state_dict[name][idx] += 1
            usage_frequency_state_dict[name] = usage_frequency_state_dict[name] / torch.sum(
                usage_frequency_state_dict[name])
        self.last_error = sum(
            [torch.sum(torch.abs(usage_frequency_state_dict[name] - self.exp_expert_usage_frequency_dict[name]))
             for name in self.exp_expert_usage_frequency_dict.keys()]
        )
        for name in self.exp_expert_usage_frequency_dict.keys():
            if is_first_step:
                self.exp_expert_usage_frequency_dict[name] = usage_frequency_state_dict[name]
            else:
                self.exp_expert_usage_frequency_dict[name] = self.beta * self.exp_expert_usage_frequency_dict[
                    name] + (1 - self.beta) * usage_frequency_state_dict[name]
        # normalize
        for name in self.exp_expert_usage_frequency_dict.keys():
            self.exp_expert_usage_frequency_dict[name] = self.exp_expert_usage_frequency_dict[name] / torch.sum(
                self.exp_expert_usage_frequency_dict[name])

    def step(self, model_outputs: Union[List[MoECausalLMOutputWithPast], MoECausalLMOutputWithPast],
             global_step: int) -> float:
        """
        Update the usage frequency state of experts and return the last error of the state which is L1 norm

        Parameters
        ----------
        model_outputs: Union[List[MoECausalLMOutputWithPast], MoECausalLMOutputWithPast]
            The outputs of the model on dataset for merging
        global_step: int
            The global step of the training process

        Returns
        -------
        float
            The last error of the state which is L1 norm
        """
        if global_step % self.compute_fisher_every_n_steps == 0:
            self._update_exp_usage_frequency_state(model_outputs)
        return self.last_error
