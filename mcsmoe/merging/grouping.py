import os
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, Optional, List, Union, Callable, Iterator, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    SwitchTransformersConfig,
    SwitchTransformersForConditionalGeneration,
    SwitchTransformersSparseMLP,
    SwitchTransformersTop1Router,
    PretrainedConfig
)
from transformers.modeling_outputs import Seq2SeqMoEOutput
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersDenseActDense

from .permutation import (
    permute_switch_mlp_dense_expert_,
    compute_switch_permutation_by_weight_matching,
    compute_switch_permutation_by_activation_matching,
    merge_switch_mlp_by_activation_matching_within_and_across_models,
    merge_switch_mlp_by_weight_matching_within_and_across_models
)
from .utils import generate_random_group_labels
from ..utils.constants import FP32_EPS

__all__ = [
    'ExpertsGrouperForSwitch',
    'merge_by_groups',
    'LEGAL_SIMILARITY_BASES',
    'SIMILARITY_MAPPING_FUNCTION',
    'ExpertUsageFrequencyTracker',
    'prune_non_core_experts_by_groups',
    'merge_by_groups_with_usage_frequency_weighting',
    'merge_non_core_by_groups_with_usage_frequency_weighting',
    'merge_all_non_core_to_one_with_usage_frequency_weighting',
    'merge_by_groups_within_and_across_models',
    'merge_non_core_by_groups_within_and_across_models',
    'merge_all_non_core_to_one_within_and_across_models',
]

SIMILARITY_MAPPING_FUNCTION = {
    "cosine": lambda x, y: (F.cosine_similarity(x, y, dim=-1, eps=FP32_EPS) + 1) / 2,
    "mse": lambda x, y: 1 / (1 + 0.1 * torch.log(F.mse_loss(x, y, reduction="sum"))),
}
LEGAL_SIMILARITY_BASES = ["weight", "feature", "feature.abs", "weight-feature", "gradient", "weight-gradient",
                          "router-logits", "router-weight", "router-weight-feature", "mse", "random",
                          "feature-correlation.lsa", "feature-correlation.max"]


class ExpertsGrouperForSwitch(object):
    def __init__(
            self,
            config: Union[SwitchTransformersConfig, PretrainedConfig],
            similarity_fn: str = "cosine",
            similarity_base: str = "weight",
    ):
        if similarity_fn not in SIMILARITY_MAPPING_FUNCTION:
            raise ValueError(
                f"[Merging]similarity_fn should be one of {SIMILARITY_MAPPING_FUNCTION.keys()}, got {similarity_fn} instead.")
        if similarity_base not in LEGAL_SIMILARITY_BASES:
            raise ValueError(
                f"[Merging]similarity_base should be one of {LEGAL_SIMILARITY_BASES}, got {similarity_base} instead.")

        self.num_experts = config.num_experts
        self.d_model = config.d_model
        self.sparse_layer_indices = list(range(1, config.num_layers, config.encoder_sparse_step))
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
        # Similarity range: [0, 2]
        for layer_idx in self.sparse_layer_indices:
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            self._group_state_dict[encoder_mlp_name] = torch.arange(self.num_experts,
                                                                    device="cuda")
            self._group_state_dict[decoder_mlp_name] = torch.arange(self.num_experts,
                                                                    device="cuda")
            self._similarity_state_dict[encoder_mlp_name] = torch.zeros(
                (self.num_experts, self.num_experts), device="cuda"
            ) + torch.eye(self.num_experts, device="cuda")
            self._similarity_state_dict[decoder_mlp_name] = torch.zeros(
                (self.num_experts, self.num_experts), device="cuda"
            ) + torch.eye(self.num_experts, device="cuda")
            self._usage_frequency_state_dict[encoder_mlp_name] = torch.zeros(self.num_experts, device="cuda")
            self._usage_frequency_state_dict[decoder_mlp_name] = torch.zeros(self.num_experts, device="cuda")

    def similarity_state_dict(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self._similarity_state_dict)

    def group_state_dict(self) -> Dict[str, torch.LongTensor]:
        return deepcopy(self._group_state_dict)

    def usage_frequency_state_dict(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self._usage_frequency_state_dict)

    def save_similarity(self, mlp_name: str, i: int, j: int, similarity: float):
        self._similarity_state_dict[mlp_name][i, j] = similarity
        self._similarity_state_dict[mlp_name][j, i] = similarity

    def get_similarity(self, mlp_name: str, i: int, j: int) -> float:
        return self._similarity_state_dict[mlp_name][i, j].item()

    def get_similarity_matrix(self, mlp_name: str) -> torch.Tensor:
        return deepcopy(self._similarity_state_dict[mlp_name])

    def save_group_state_dict(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._group_state_dict, os.path.join(save_dir, "group_state_dict.pt"))

    def load_group_state_dict(self, load_dir: str):
        self._group_state_dict = torch.load(os.path.join(load_dir, "group_state_dict.pt"))

    @staticmethod
    def defend_core_experts_fisher(
            fisher_state_dict: Dict[str, torch.Tensor],
            core_experts: Dict[str, List[int]],
    ) -> Dict[str, torch.Tensor]:
        """TODO: This is a unfinished solution to avoid zero fisher in core experts."""
        for mlp_name, core_expert_list in core_experts.items():
            for i in core_expert_list:
                expert_name = f"{mlp_name}.experts.expert_{i}.wi.weight"
                if fisher_state_dict[expert_name].abs().sum() < FP32_EPS:
                    fisher_state_dict[expert_name] = torch.ones_like(fisher_state_dict[expert_name]) * FP32_EPS
                expert_name = f"{mlp_name}.experts.expert_{i}.wo.weight"
                if fisher_state_dict[expert_name].abs().sum() < FP32_EPS:
                    fisher_state_dict[expert_name] = torch.ones_like(fisher_state_dict[expert_name]) * FP32_EPS
        return fisher_state_dict

    def _assign_num_groups_per_layer(
            self,
            average_num_groups: int,
            merging_encoder_layers: List[int],
            merging_decoder_layers: List[int],
    ) -> Dict[str, int]:
        num_grouping_layers = len(merging_encoder_layers) + len(merging_decoder_layers)
        total_num_groups = average_num_groups * num_grouping_layers + self.num_experts * (
                len(self.sparse_layer_indices) * 2 - num_grouping_layers
        )
        all_usage_frequency = []
        usage_frequency_dict = deepcopy(self._usage_frequency_state_dict)
        for i, layer_idx in enumerate(self.sparse_layer_indices):
            # 1. Experts in the excluded layers are always not merged.
            if layer_idx not in merging_encoder_layers:
                usage_frequency_dict[f"encoder.block.{layer_idx}.layer.1.mlp"] = torch.ones_like(
                    usage_frequency_dict[f"encoder.block.{layer_idx}.layer.1.mlp"]
                )
            if layer_idx not in merging_decoder_layers:
                usage_frequency_dict[f"decoder.block.{layer_idx}.layer.2.mlp"] = torch.ones_like(
                    usage_frequency_dict[f"decoder.block.{layer_idx}.layer.2.mlp"]
                )
            # 2. Each layer must have at least one group, set the most used expert in a layer to frequency 1.
            max_usage_index = torch.argmax(usage_frequency_dict[f"encoder.block.{layer_idx}.layer.1.mlp"])
            usage_frequency_dict[f"encoder.block.{layer_idx}.layer.1.mlp"][max_usage_index] = 1.0
            max_usage_index = torch.argmax(usage_frequency_dict[f"decoder.block.{layer_idx}.layer.2.mlp"])
            usage_frequency_dict[f"decoder.block.{layer_idx}.layer.2.mlp"][max_usage_index] = 1.0
            # 3. Collect all usage frequency.
            all_usage_frequency.append(usage_frequency_dict[f"encoder.block.{layer_idx}.layer.1.mlp"])
            all_usage_frequency.append(usage_frequency_dict[f"decoder.block.{layer_idx}.layer.2.mlp"])
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
            num_groups_per_layer[f"encoder.block.{layer_idx}.layer.1.mlp"] = torch.sum(
                (usage_frequency_dict[f"encoder.block.{layer_idx}.layer.1.mlp"] > frequency_threshold).long()
            ).item()
            num_groups_per_layer[f"decoder.block.{layer_idx}.layer.2.mlp"] = torch.sum(
                (usage_frequency_dict[f"decoder.block.{layer_idx}.layer.2.mlp"] > frequency_threshold).long()
            ).item()
        return num_groups_per_layer

    def group_experts_randomly(
            self,
            num_groups: int,
    ):
        for layer_idx in tqdm(self.sparse_layer_indices,
                              desc=f"Randomly merging experts into {num_groups} clusters"):
            # 1. Encoder
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            group_labels = generate_random_group_labels(self.num_experts, num_groups)
            self._group_state_dict[encoder_mlp_name] = group_labels
            # 2. Decoder
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            group_labels = generate_random_group_labels(self.num_experts, num_groups)
            self._group_state_dict[decoder_mlp_name] = group_labels

    def group_experts_into_clusters_by_routing_guided_globally(
            self,
            average_num_groups: int,
            merging_encoder_layers: List[int],
            merging_decoder_layers: List[int],
            encoder_group_capacity: Optional[int] = None,
            decoder_group_capacity: Optional[int] = None,
    ) -> Dict[str, List[int]]:
        """
        Globally group experts into clusters by routing-guided clustering, each layer will have different number of
         clusters. The total number of clusters is determined by average_num_groups.

        Parameters
        ----------
        average_num_groups: int
            The average number of clusters for all layers.
        merging_encoder_layers: List[int]
            The layers of encoder that are excluded from merging.
        merging_decoder_layers: List[int]
            The layers of decoder that are excluded from merging.
        encoder_group_capacity: Optional[int]
            The maximum number of experts in each group in Encoder layers. If None, the number of experts in each group is not limited.
        decoder_group_capacity: Optional[int]
            The maximum number of experts in each group in Decoder layers. If None, the number of experts in each group is not limited.

        Returns
        -------
        core_experts: Dict[str, List[int]]
            The core experts of each cluster
        """
        # By default, the first layer of encoder is excluded.
        # 1. Assign num_groups respectively for each layer according to average_num_groups
        encoder_group_capacity = encoder_group_capacity if encoder_group_capacity is not None else self.num_experts
        decoder_group_capacity = decoder_group_capacity if decoder_group_capacity is not None else self.num_experts
        num_groups_per_layer = self._assign_num_groups_per_layer(
            average_num_groups, merging_encoder_layers, merging_decoder_layers
        )
        print(f"[Merging]Number of groups of each layer: {num_groups_per_layer}")
        # 2. Group experts into clusters for each layer
        core_experts = dict()
        for layer_idx in tqdm(self.sparse_layer_indices,
                              desc=f"Globally routing-guided clustering experts into average {average_num_groups} clusters"):
            # 1. Encoder
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            num_groups = num_groups_per_layer[encoder_mlp_name]
            group_member_count = torch.zeros(num_groups)
            indices_sorted_by_usage = torch.argsort(self._usage_frequency_state_dict[encoder_mlp_name], descending=True)
            # 1.1 Assign top-K most-used experts with label 0 to K-1 respectively
            core_expert_indices = indices_sorted_by_usage[:num_groups]
            core_experts[encoder_mlp_name] = core_expert_indices.tolist()
            for i in range(num_groups):
                self._group_state_dict[encoder_mlp_name][core_expert_indices[i]] = i
                group_member_count[i] += 1
            # 1.2 Assign left unassigned experts to the cluster with the most similar core
            similarity_matrix = self.get_similarity_matrix(encoder_mlp_name)
            for i in range(num_groups, self.num_experts):
                # Find the most similar core
                most_similar_core = core_expert_indices[
                    torch.argmax(similarity_matrix[i, core_expert_indices])
                ]
                most_similar_group_label = self._group_state_dict[encoder_mlp_name][most_similar_core]
                self._group_state_dict[encoder_mlp_name][i] = most_similar_group_label
                group_member_count[most_similar_group_label] += 1
                if group_member_count[self._group_state_dict[encoder_mlp_name][i]] >= encoder_group_capacity:
                    if len(core_expert_indices) == 1:
                        raise ValueError(
                            f"[Merging]The number of groups at Encoder layer {layer_idx} is too small!"
                        )
                    # Kick out the filled group as well as its core, by pop the core from core_experts
                    core_index = torch.argmax(similarity_matrix[i, core_expert_indices])
                    core_expert_indices = torch.cat(
                        [core_expert_indices[:core_index], core_expert_indices[core_index + 1:]]
                    )

            # 2. Decoder
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            num_groups = num_groups_per_layer[decoder_mlp_name]
            group_member_count = torch.zeros(num_groups)
            indices_sorted_by_usage = torch.argsort(self._usage_frequency_state_dict[decoder_mlp_name], descending=True)
            # 2.1 Assign top-K most-used experts with label 0 to K-1 respectively
            core_expert_indices = indices_sorted_by_usage[:num_groups]
            core_experts[decoder_mlp_name] = core_expert_indices.tolist()
            for i in range(num_groups):
                self._group_state_dict[decoder_mlp_name][indices_sorted_by_usage[i]] = i
                group_member_count[i] += 1
            # 2.2 Assign left unassigned experts to the cluster with the most similar core
            similarity_matrix = self.get_similarity_matrix(decoder_mlp_name)
            for i in range(num_groups, self.num_experts):
                # Find the most similar core
                most_similar_core = core_expert_indices[
                    torch.argmax(similarity_matrix[i, core_expert_indices])
                ]
                most_similar_group_label = self._group_state_dict[decoder_mlp_name][most_similar_core]
                self._group_state_dict[decoder_mlp_name][i] = most_similar_group_label
                if group_member_count[self._group_state_dict[decoder_mlp_name][i]] >= decoder_group_capacity:
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
            # 1. Encoder
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            indices_sorted_by_usage = torch.argsort(self._usage_frequency_state_dict[encoder_mlp_name], descending=True)
            # 1.1 Assign top-K most-used experts with label 0 to K-1 respectively
            core_experts[encoder_mlp_name] = indices_sorted_by_usage[:num_groups].tolist()
            for i in range(num_groups):
                self._group_state_dict[encoder_mlp_name][indices_sorted_by_usage[i]] = i
            # 1.2 Assign left unassigned experts to the cluster with the most similar core
            similarity_matrix = self.get_similarity_matrix(encoder_mlp_name)
            for i in range(num_groups, self.num_experts):
                # Find the most similar core
                most_similar_core = indices_sorted_by_usage[
                    torch.argmax(similarity_matrix[i, indices_sorted_by_usage[:num_groups]])
                ]
                self._group_state_dict[encoder_mlp_name][i] = self._group_state_dict[encoder_mlp_name][
                    most_similar_core]

            # 2. Decoder
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            indices_sorted_by_usage = torch.argsort(self._usage_frequency_state_dict[decoder_mlp_name], descending=True)
            # 2.1 Assign top-K most-used experts with label 0 to K-1 respectively
            core_experts[decoder_mlp_name] = indices_sorted_by_usage[:num_groups].tolist()
            for i in range(num_groups):
                self._group_state_dict[decoder_mlp_name][indices_sorted_by_usage[i]] = i
            # 2.2 Assign left unassigned experts to the cluster with the most similar core
            similarity_matrix = self.get_similarity_matrix(decoder_mlp_name)
            for i in range(num_groups, self.num_experts):
                # Find the most similar core
                most_similar_core = indices_sorted_by_usage[
                    torch.argmax(similarity_matrix[i, indices_sorted_by_usage[:num_groups]])
                ]
                self._group_state_dict[decoder_mlp_name][i] = self._group_state_dict[decoder_mlp_name][
                    most_similar_core]
        return core_experts

    def group_experts_into_clusters_by_spectral_clustering(
            self,
            num_groups: int
    ):
        """
        Group experts by spectral clustering.
        """
        if num_groups > self.num_experts:
            raise ValueError(
                f"[Merging]num_groups should be less than or equal to {self.num_experts}, got {num_groups} instead.")
        for layer_idx in tqdm(self.sparse_layer_indices,
                              desc=f"Spectral clustering experts into {num_groups} clusters..."):
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"

            # 1. Encoder layer_idx MLP
            # 1.1 Compute grouping labels
            similarity_matrix = self.get_similarity_matrix(encoder_mlp_name).cpu().numpy()
            clustering = SpectralClustering(
                n_clusters=num_groups,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=0
            ).fit(similarity_matrix)
            self._group_state_dict[encoder_mlp_name] = torch.tensor(clustering.labels_, device="cuda")

            # 2. Decoder layer_idx MLP
            similarity_matrix = self.get_similarity_matrix(decoder_mlp_name).cpu().numpy()
            clustering = SpectralClustering(
                n_clusters=num_groups,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=0
            ).fit(similarity_matrix)
            self._group_state_dict[decoder_mlp_name] = torch.tensor(clustering.labels_, device="cuda")

    def compute_all_usages(
            self,
            model: SwitchTransformersForConditionalGeneration,
            batch: Dict[str, torch.Tensor],
            mini_batch_size: Optional[int] = 128,
    ):
        model.cuda()
        model.eval()
        total_batch_size = batch["input_ids"].shape[0]
        if mini_batch_size > total_batch_size:
            mini_batch_size = total_batch_size
        num_batches = total_batch_size // mini_batch_size
        for i in tqdm(range(num_batches), desc="[Merging]Computing all usages..."):
            with torch.no_grad():
                mini_batch = {k: v[i * mini_batch_size: (i + 1) * mini_batch_size] for k, v in batch.items()}
                mini_batch = {k: v.cuda() for k, v in mini_batch.items()}
                outputs = model(**mini_batch)
                for layer_idx in self.sparse_layer_indices:
                    encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
                    encoder_router_expert_index = outputs.encoder_router_logits[layer_idx][1].reshape(-1)
                    for idx in encoder_router_expert_index:
                        self._usage_frequency_state_dict[encoder_mlp_name][idx] += 1
                    decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
                    decoder_router_expert_index = outputs.decoder_router_logits[layer_idx][1].reshape(-1)
                    for idx in decoder_router_expert_index:
                        self._usage_frequency_state_dict[decoder_mlp_name][idx] += 1
        self._usage_frequency_state_dict = {
            k: v / torch.sum(v) for k, v in self._usage_frequency_state_dict.items()
        }

    def reverse_all_similarities(self):
        print("[Merging]Reversing all similarities...")
        for key in self._similarity_state_dict.keys():
            self._similarity_state_dict[key] = 1 - self._similarity_state_dict[key]

    def compute_all_similarities(
            self,
            model: SwitchTransformersForConditionalGeneration,
            batch: Dict[str, torch.Tensor] = None
    ):
        if self.similarity_base not in ["weight", "router-weight"] and batch is None:
            raise ValueError(
                "[Merging]batch should be provided when similarity_base is not 'weight' or 'router-weight'")

        model = model.cuda()
        model = model.eval()
        if self.similarity_base == "weight":
            self._compute_all_similarities_by_weight(model.state_dict())
        elif self.similarity_base == 'router-weight':
            self._compute_all_similarities_by_router_weight(model.state_dict())
        elif self.similarity_base == 'router-logits':
            batch = {k: v.cuda() for k, v in batch.items()}
            self._compute_all_similarities_by_router_logits(model, batch)
        elif self.similarity_base == 'gradient':
            model = self._accumulate_gradient(model, batch)
            self._compute_all_similarities_by_gradient(model)
            model.zero_grad()
        elif self.similarity_base == 'weight-gradient':
            model = self._accumulate_gradient(model, batch)
            self._compute_all_similarities_by_weight_gradient(model)
            model.zero_grad()
        elif self.similarity_base == 'feature':
            batch = {k: v.cuda() for k, v in batch.items()}
            self._compute_all_similarities_by_feature(model, batch)
        elif self.similarity_base == 'feature.abs':
            batch = {k: v.cuda() for k, v in batch.items()}
            self._compute_all_similarities_by_feature_abs(model, batch)
        elif self.similarity_base == 'weight-feature':
            batch = {k: v.cuda() for k, v in batch.items()}
            self._compute_all_similarities_by_weight_feature(model, batch)
        elif self.similarity_base == 'router-weight-feature':
            batch = {k: v.cuda() for k, v in batch.items()}
            self._compute_all_similarities_by_router_weight_feature(model, batch)
        elif self.similarity_base == 'mse':
            batch = {k: v.cuda() for k, v in batch.items()}
            self._compute_all_similarities_by_mse(model, batch)
        elif self.similarity_base == 'random':
            self._compute_all_similarities_by_random()
        elif self.similarity_base == 'feature-correlation.lsa':
            batch = {k: v.cuda() for k, v in batch.items()}
            self._compute_all_similarities_by_average_feature_correlation(model, batch, lsa=True)
        elif self.similarity_base == 'feature-correlation.max':
            batch = {k: v.cuda() for k, v in batch.items()}
            self._compute_all_similarities_by_average_feature_correlation(model, batch, lsa=False)
        else:
            raise NotImplementedError

    def _compute_all_similarities_by_weight(self, state_dict: Dict[str, torch.Tensor]):
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[Merging]Computing similarities by weight..."):
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    # 1. Encoder
                    i_flat = torch.cat(
                        [state_dict[f"{encoder_mlp_name}.experts.expert_{i}.wi.weight"].flatten(),
                         state_dict[f"{encoder_mlp_name}.experts.expert_{i}.wo.weight"].flatten()],
                        dim=0
                    )
                    j_flat = torch.cat(
                        [state_dict[f"{encoder_mlp_name}.experts.expert_{j}.wi.weight"].flatten(),
                         state_dict[f"{encoder_mlp_name}.experts.expert_{j}.wo.weight"].flatten()],
                        dim=0
                    )
                    similarity = self.similarity_fn(i_flat, j_flat)
                    self.save_similarity(encoder_mlp_name, i, j, similarity)

                    # 2. Decoder
                    i_flat = torch.cat(
                        [state_dict[f"{decoder_mlp_name}.experts.expert_{i}.wi.weight"].flatten(),
                         state_dict[f"{decoder_mlp_name}.experts.expert_{i}.wo.weight"].flatten()],
                        dim=0
                    )
                    j_flat = torch.cat(
                        [state_dict[f"{decoder_mlp_name}.experts.expert_{j}.wi.weight"].flatten(),
                         state_dict[f"{decoder_mlp_name}.experts.expert_{j}.wo.weight"].flatten()],
                        dim=0
                    )
                    similarity = self.similarity_fn(i_flat, j_flat)
                    self.save_similarity(decoder_mlp_name, i, j, similarity)

    @staticmethod
    def _accumulate_gradient(
            model: SwitchTransformersForConditionalGeneration,
            batch: Dict[str, torch.Tensor],
            mini_batch_size: Optional[int] = 16,
    ) -> SwitchTransformersForConditionalGeneration:
        batch_size = batch["input_ids"].shape[0]
        num_batches = batch_size // mini_batch_size
        for i in tqdm(range(num_batches), desc="[Merging]Accumulating gradient..."):
            mini_batch = {k: v[i * mini_batch_size: (i + 1) * mini_batch_size] for k, v in batch.items()}
            mini_batch = {k: v.cuda() for k, v in mini_batch.items()}
            loss = model(**mini_batch).loss
            loss /= num_batches
            loss.backward(retain_graph=True)
        return model

    def _compute_all_similarities_by_gradient(
            self, model: SwitchTransformersForConditionalGeneration
    ):
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[Merging]Computing similarities by gradient..."):
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            encoder_mlp = model.encoder.block[layer_idx].layer[-1].mlp
            decoder_mlp = model.decoder.block[layer_idx].layer[-1].mlp
            with torch.no_grad():
                for i in range(self.num_experts):
                    for j in range(i + 1, self.num_experts):
                        # 1. Encoder
                        i_flat = torch.cat(
                            [encoder_mlp.experts[f"expert_{i}"].wi.weight.grad.data.flatten(),
                             encoder_mlp.experts[f"expert_{i}"].wo.weight.grad.data.flatten()],
                            dim=0
                        )
                        j_flat = torch.cat(
                            [encoder_mlp.experts[f"expert_{j}"].wi.weight.grad.data.flatten(),
                             encoder_mlp.experts[f"expert_{j}"].wo.weight.grad.data.flatten()],
                            dim=0
                        )
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(encoder_mlp_name, i, j, similarity)

                        # 2. Decoder
                        i_flat = torch.cat(
                            [decoder_mlp.experts[f"expert_{i}"].wi.weight.grad.data.flatten(),
                             decoder_mlp.experts[f"expert_{i}"].wo.weight.grad.data.flatten()],
                            dim=0
                        )
                        j_flat = torch.cat(
                            [decoder_mlp.experts[f"expert_{j}"].wi.weight.grad.data.flatten(),
                             decoder_mlp.experts[f"expert_{j}"].wo.weight.grad.data.flatten()],
                            dim=0
                        )
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(decoder_mlp_name, i, j, similarity)

    def _compute_all_similarities_by_weight_gradient(
            self, model: SwitchTransformersForConditionalGeneration
    ):
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[Merging]Computing similarities by weight*gradient..."):
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            encoder_mlp = model.encoder.block[layer_idx].layer[-1].mlp
            decoder_mlp = model.decoder.block[layer_idx].layer[-1].mlp
            with torch.no_grad():
                for i in range(self.num_experts):
                    for j in range(i + 1, self.num_experts):
                        # 1. Encoder
                        i_flat = torch.cat(
                            [encoder_mlp.experts[f"expert_{i}"].wi.weight.grad.data.flatten(),
                             encoder_mlp.experts[f"expert_{i}"].wo.weight.grad.data.flatten()],
                            dim=0
                        ) * torch.cat(
                            [encoder_mlp.experts[f"expert_{i}"].wi.weight.data.flatten(),
                             encoder_mlp.experts[f"expert_{i}"].wo.weight.data.flatten()],
                            dim=0
                        )
                        j_flat = torch.cat(
                            [encoder_mlp.experts[f"expert_{j}"].wi.weight.grad.data.flatten(),
                             encoder_mlp.experts[f"expert_{j}"].wo.weight.grad.data.flatten()],
                            dim=0
                        ) * torch.cat(
                            [encoder_mlp.experts[f"expert_{j}"].wi.weight.data.flatten(),
                             encoder_mlp.experts[f"expert_{j}"].wo.weight.data.flatten()],
                            dim=0
                        )
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(encoder_mlp_name, i, j, similarity)

                        # 2. Decoder
                        i_flat = torch.cat(
                            [decoder_mlp.experts[f"expert_{i}"].wi.weight.grad.data.flatten(),
                             decoder_mlp.experts[f"expert_{i}"].wo.weight.grad.data.flatten()],
                            dim=0
                        ) * torch.cat(
                            [decoder_mlp.experts[f"expert_{i}"].wi.weight.data.flatten(),
                             decoder_mlp.experts[f"expert_{i}"].wo.weight.data.flatten()],
                            dim=0
                        )
                        j_flat = torch.cat(
                            [decoder_mlp.experts[f"expert_{j}"].wi.weight.grad.data.flatten(),
                             decoder_mlp.experts[f"expert_{j}"].wo.weight.grad.data.flatten()],
                            dim=0
                        ) * torch.cat(
                            [decoder_mlp.experts[f"expert_{j}"].wi.weight.data.flatten(),
                             decoder_mlp.experts[f"expert_{j}"].wo.weight.data.flatten()],
                            dim=0
                        )
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(decoder_mlp_name, i, j, similarity)

    def _compute_all_similarities_by_router_weight(
            self, state_dict: Dict[str, torch.Tensor]
    ):
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[Merging]Computing similarities by router rows..."):
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    # 1. Encoder
                    i_flat = state_dict[f"{encoder_mlp_name}.router.classifier.weight"][i]
                    j_flat = state_dict[f"{encoder_mlp_name}.router.classifier.weight"][j]
                    similarity = self.similarity_fn(i_flat, j_flat)
                    self.save_similarity(encoder_mlp_name, i, j, similarity)

                    # 2. Decoder
                    i_flat = state_dict[f"{decoder_mlp_name}.router.classifier.weight"][i]
                    j_flat = state_dict[f"{decoder_mlp_name}.router.classifier.weight"][j]
                    similarity = self.similarity_fn(i_flat, j_flat)
                    self.save_similarity(decoder_mlp_name, i, j, similarity)

    def _compute_all_similarities_by_router_logits(
            self, model: SwitchTransformersForConditionalGeneration, batch: Dict[str, torch.Tensor]
    ):
        with torch.no_grad():
            outputs = model(**batch)
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[Merging]Computing similarities by router logits..."):
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            encoder_router_logits = outputs.encoder_router_logits[layer_idx][0].reshape(-1, self.num_experts)
            decoder_router_logits = outputs.decoder_router_logits[layer_idx][0].reshape(-1, self.num_experts)
            with torch.no_grad():
                for i in range(self.num_experts):
                    for j in range(i + 1, self.num_experts):
                        # 1. Encoder
                        i_flat = encoder_router_logits[:, i].flatten()
                        j_flat = encoder_router_logits[:, j].flatten()
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(encoder_mlp_name, i, j, similarity)

                        # 2. Decoder
                        i_flat = decoder_router_logits[:, i].flatten()
                        j_flat = decoder_router_logits[:, j].flatten()
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(decoder_mlp_name, i, j, similarity)

    def _compute_all_similarities_by_router_weight_feature(
            self, model: SwitchTransformersForConditionalGeneration, batch: Dict[str, torch.Tensor]
    ):
        self.activations = {}
        handles = []
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[Merging]Registering forward hook..."):
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            handles.append(model.encoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
                self._get_mlp_activation(encoder_mlp_name)
            ))
            handles.append(model.decoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
                self._get_mlp_activation(decoder_mlp_name)
            ))
        with torch.no_grad():
            model(**batch)
        state_dict = model.state_dict()
        for layer_idx in tqdm(self.sparse_layer_indices,
                              desc="[Merging]Computing similarities by router weight*feature..."):
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            # features of shape (batch_size*sequence_length, hidden_size)
            encoder_features = self.activations[encoder_mlp_name]
            decoder_features = self.activations[decoder_mlp_name]
            with torch.no_grad():
                for i in range(self.num_experts):
                    for j in range(i + 1, self.num_experts):
                        # 1. Encoder
                        features = torch.linalg.norm(
                            encoder_features,
                            dim=0
                        )
                        i_flat = state_dict[f"{encoder_mlp_name}.router.classifier.weight"][i] * features
                        j_flat = state_dict[f"{encoder_mlp_name}.router.classifier.weight"][j] * features
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(encoder_mlp_name, i, j, similarity)

                        # 2. Decoder
                        features = torch.linalg.norm(
                            decoder_features,
                            dim=0
                        )
                        i_flat = state_dict[f"{decoder_mlp_name}.router.classifier.weight"][i] * features
                        j_flat = state_dict[f"{decoder_mlp_name}.router.classifier.weight"][j] * features
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(decoder_mlp_name, i, j, similarity)
        for handle in handles:
            handle.remove()

    def _get_mlp_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = input[0].detach().reshape(
                -1, self.d_model)  # of shape (batch_size*sequence_length, hidden_size)

        return hook

    def _compute_all_similarities_by_feature(
            self, model: SwitchTransformersForConditionalGeneration, batch: Dict[str, torch.Tensor]
    ):
        self.activations = {}
        handles = []
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[Merging]Registering forward hook..."):
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            handles.append(model.encoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
                self._get_mlp_activation(encoder_mlp_name)
            ))
            handles.append(model.decoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
                self._get_mlp_activation(decoder_mlp_name)
            ))
        with torch.no_grad():
            outputs = model(**batch)
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[Merging]Computing similarities by feature..."):
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            # router index of shape (batch_size*sequence_length)
            encoder_router_expert_index = outputs.encoder_router_logits[layer_idx][1].reshape(-1)
            decoder_router_expert_index = outputs.decoder_router_logits[layer_idx][1].reshape(-1)
            # features of shape (batch_size*sequence_length, hidden_size)
            encoder_features = self.activations[encoder_mlp_name]
            decoder_features = self.activations[decoder_mlp_name]
            with torch.no_grad():
                for i in range(self.num_experts):
                    for j in range(i + 1, self.num_experts):
                        # 1. Encoder
                        i_flat = torch.mean(
                            encoder_features[encoder_router_expert_index == i],
                            dim=0
                        )
                        if torch.isnan(i_flat).any():
                            i_flat = torch.zeros(self.d_model, device='cuda')
                        j_flat = torch.mean(
                            encoder_features[encoder_router_expert_index == j],
                            dim=0
                        )
                        if torch.isnan(j_flat).any():
                            j_flat = torch.zeros(self.d_model, device='cuda')
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(encoder_mlp_name, i, j, similarity)

                        # 2. Decoder
                        i_flat = torch.mean(
                            decoder_features[decoder_router_expert_index == i],
                            dim=0
                        )
                        if torch.isnan(i_flat).any():
                            i_flat = torch.zeros(self.d_model, device='cuda')
                        j_flat = torch.mean(
                            decoder_features[decoder_router_expert_index == j],
                            dim=0
                        )
                        if torch.isnan(j_flat).any():
                            j_flat = torch.zeros(self.d_model, device='cuda')
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(decoder_mlp_name, i, j, similarity)
        for handle in handles:
            handle.remove()

    def _compute_all_similarities_by_feature_abs(
            self, model: SwitchTransformersForConditionalGeneration, batch: Dict[str, torch.Tensor]
    ):
        self.activations = {}
        handles = []
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[Merging]Registering forward hook..."):
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            handles.append(model.encoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
                self._get_mlp_activation(encoder_mlp_name))
            )
            handles.append(model.decoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
                self._get_mlp_activation(decoder_mlp_name))
            )
        with torch.no_grad():
            outputs = model(**batch)
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[Merging]Computing similarities by feature.abs..."):
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            # router index of shape (batch_size*sequence_length)
            encoder_router_expert_index = outputs.encoder_router_logits[layer_idx][1].reshape(-1)
            decoder_router_expert_index = outputs.decoder_router_logits[layer_idx][1].reshape(-1)
            # features of shape (batch_size*sequence_length, hidden_size)
            encoder_features = self.activations[encoder_mlp_name]
            decoder_features = self.activations[decoder_mlp_name]
            with torch.no_grad():
                for i in range(self.num_experts):
                    for j in range(i + 1, self.num_experts):
                        # 1. Encoder
                        i_flat = torch.mean(
                            encoder_features[encoder_router_expert_index == i].abs(),
                            dim=0
                        )
                        if torch.isnan(i_flat).any():
                            i_flat = torch.zeros(self.d_model, device='cuda')
                        j_flat = torch.mean(
                            encoder_features[encoder_router_expert_index == j].abs(),
                            dim=0
                        )
                        if torch.isnan(j_flat).any():
                            j_flat = torch.zeros(self.d_model, device='cuda')
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(encoder_mlp_name, i, j, similarity)

                        # 2. Decoder
                        i_flat = torch.mean(
                            decoder_features[decoder_router_expert_index == i].abs(),
                            dim=0
                        )
                        if torch.isnan(i_flat).any():
                            i_flat = torch.zeros(self.d_model, device='cuda')
                        j_flat = torch.mean(
                            decoder_features[decoder_router_expert_index == j].abs(),
                            dim=0
                        )
                        if torch.isnan(j_flat).any():
                            j_flat = torch.zeros(self.d_model, device='cuda')
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(decoder_mlp_name, i, j, similarity)
        for handle in handles:
            handle.remove()

    def _compute_all_similarities_by_weight_feature(
            self, model: SwitchTransformersForConditionalGeneration, batch: Dict[str, torch.Tensor]
    ):
        self.activations = {}
        handles = []
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[Merging]Registering forward hook..."):
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            handles.append(model.encoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
                self._get_mlp_activation(encoder_mlp_name)
            ))
            handles.append(model.decoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
                self._get_mlp_activation(decoder_mlp_name)
            ))
        with torch.no_grad():
            outputs = model(**batch)
        state_dict = model.state_dict()
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[Merging]Computing similarities by weight*feature..."):
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            # router index of shape (batch_size*sequence_length)
            encoder_router_expert_index = outputs.encoder_router_logits[layer_idx][1].reshape(-1)
            decoder_router_expert_index = outputs.decoder_router_logits[layer_idx][1].reshape(-1)
            # features of shape (batch_size*sequence_length, hidden_size)
            encoder_features = self.activations[encoder_mlp_name]
            decoder_features = self.activations[decoder_mlp_name]
            with torch.no_grad():
                for i in range(self.num_experts):
                    for j in range(i + 1, self.num_experts):
                        # 1. Encoder
                        i_feature = torch.linalg.norm(
                            encoder_features[encoder_router_expert_index == i],
                            dim=0
                        )
                        if torch.isnan(i_feature).any():
                            i_feature = torch.zeros(self.d_model, device='cuda')
                        i_flat = (state_dict[f"{encoder_mlp_name}.experts.expert_{i}.wi.weight"] * i_feature).flatten()
                        j_feature = torch.linalg.norm(
                            encoder_features[encoder_router_expert_index == j],
                            dim=0
                        )
                        if torch.isnan(j_feature).any():
                            j_feature = torch.zeros(self.d_model, device='cuda')
                        j_flat = (state_dict[f"{encoder_mlp_name}.experts.expert_{j}.wi.weight"] * j_feature).flatten()
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(encoder_mlp_name, i, j, similarity)

                        # 2. Decoder
                        i_feature = torch.linalg.norm(
                            decoder_features[decoder_router_expert_index == i],
                            dim=0
                        )
                        if torch.isnan(i_feature).any():
                            i_feature = torch.zeros(self.d_model, device='cuda')
                        i_flat = (state_dict[f"{decoder_mlp_name}.experts.expert_{i}.wi.weight"] * i_feature).flatten()
                        j_feature = torch.linalg.norm(
                            decoder_features[decoder_router_expert_index == j],
                            dim=0
                        )
                        if torch.isnan(j_feature).any():
                            j_feature = torch.zeros(self.d_model, device='cuda')
                        j_flat = (state_dict[f"{decoder_mlp_name}.experts.expert_{j}.wi.weight"] * j_feature).flatten()
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(decoder_mlp_name, i, j, similarity)

        for handle in handles:
            handle.remove()

    def _compute_all_similarities_by_mse(
            self,
            model: SwitchTransformersForConditionalGeneration,
            batch: Dict[str, torch.Tensor]
    ):
        self.activations = {}
        handles = []
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[Merging]Registering forward hook..."):
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            handles.append(model.encoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
                self._get_mlp_activation(encoder_mlp_name)
            ))
            handles.append(model.decoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
                self._get_mlp_activation(decoder_mlp_name)
            ))
        with torch.no_grad():
            outputs = model(**batch)
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[Merging]Computing similarities by MSE..."):
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            # router index of shape (batch_size*sequence_length)
            encoder_router_expert_index = outputs.encoder_router_logits[layer_idx][1].reshape(-1)
            decoder_router_expert_index = outputs.decoder_router_logits[layer_idx][1].reshape(-1)
            # features of shape (batch_size*sequence_length, hidden_size)
            encoder_features = self.activations[encoder_mlp_name]
            decoder_features = self.activations[decoder_mlp_name]
            with torch.no_grad():
                for i in range(self.num_experts):
                    encoder_i = model.encoder.block[layer_idx].layer[-1].mlp.experts[f"expert_{i}"]
                    encoder_i_inputs = encoder_features[encoder_router_expert_index == i]
                    encoder_i_outputs = encoder_i(encoder_i_inputs)
                    decoder_i = model.decoder.block[layer_idx].layer[-1].mlp.experts[f"expert_{i}"]
                    decoder_i_inputs = decoder_features[decoder_router_expert_index == i]
                    decoder_i_outputs = decoder_i(decoder_i_inputs)
                    for j in range(i + 1, self.num_experts):
                        encoder_j = model.encoder.block[layer_idx].layer[-1].mlp.experts[f"expert_{j}"]
                        encoder_j_inputs = encoder_features[encoder_router_expert_index == j]
                        encoder_j_outputs = encoder_j(encoder_j_inputs)
                        merged_new_mlp = SwitchTransformersDenseActDense(model.config).cuda()

                        # 1. Encoder
                        merged_new_mlp.wi.weight.copy_((encoder_i.wi.weight.data + encoder_j.wi.weight.data) / 2)
                        merged_new_mlp.wo.weight.copy_((encoder_i.wo.weight.data + encoder_j.wo.weight.data) / 2)
                        merged_new_outputs_from_i = merged_new_mlp(encoder_i_inputs)
                        merged_new_outputs_from_j = merged_new_mlp(encoder_j_inputs)
                        if len(encoder_i_inputs) == 0 and len(encoder_j_inputs) == 0:
                            # When both empty, similarity is 1
                            similarity = 1.0
                        else:
                            merged_new_outputs = torch.cat(
                                [merged_new_outputs_from_i, merged_new_outputs_from_j], dim=0
                            )
                            encoder_outputs = torch.cat([encoder_i_outputs, encoder_j_outputs], dim=0)
                            similarity = self.similarity_fn(merged_new_outputs, encoder_outputs)
                        self.save_similarity(encoder_mlp_name, i, j, similarity)

                        # 2. Decoder
                        decoder_j = model.decoder.block[layer_idx].layer[-1].mlp.experts[f"expert_{j}"]
                        decoder_j_inputs = decoder_features[decoder_router_expert_index == j]
                        decoder_j_outputs = decoder_j(decoder_j_inputs)
                        merged_new_mlp.wi.weight.copy_((decoder_i.wi.weight.data + decoder_j.wi.weight.data) / 2)
                        merged_new_mlp.wo.weight.copy_((decoder_i.wo.weight.data + decoder_j.wo.weight.data) / 2)
                        merged_new_outputs_from_i = merged_new_mlp(decoder_i_inputs)
                        merged_new_outputs_from_j = merged_new_mlp(decoder_j_inputs)
                        if len(decoder_i_inputs) == 0 and len(decoder_j_inputs) == 0:
                            # When both empty, similarity is 1
                            similarity = 1.0
                        else:
                            merged_new_outputs = torch.cat(
                                [merged_new_outputs_from_i, merged_new_outputs_from_j], dim=0
                            )
                            decoder_outputs = torch.cat(
                                [decoder_i_outputs, decoder_j_outputs], dim=0
                            )
                            similarity = self.similarity_fn(merged_new_outputs, decoder_outputs)
                        self.save_similarity(decoder_mlp_name, i, j, similarity)
        for handle in handles:
            handle.remove()

    def _compute_all_similarities_by_random(self):
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[Merging]Computing similarities randomly..."):
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    self.save_similarity(encoder_mlp_name, i, j, torch.rand(1).item())
                    self.save_similarity(decoder_mlp_name, i, j, torch.rand(1).item())

    @staticmethod
    def _compute_similarity_by_average_correlation(
            first_mlp: SwitchTransformersSparseMLP,
            second_mlp: SwitchTransformersSparseMLP,
            forwarded_hidden_states: torch.Tensor,
            lsa: bool
    ) -> float:
        if len(forwarded_hidden_states) <= 1:
            # TODO: What to do when both experts are never selected?
            # For now, return 1.0 to just merge them
            return 1.0

        first_activations = []
        second_activations = []

        def _first_hook(model, input, output):
            first_activations.append(input[0].detach().reshape(-1, input[0].shape[-1]))

        def _second_hook(model, input, output):
            second_activations.append(input[0].detach().reshape(-1, input[0].shape[-1]))

        first_handle = first_mlp.wo.register_forward_hook(_first_hook)
        second_handle = second_mlp.wo.register_forward_hook(_second_hook)
        with torch.no_grad():
            first_mlp(forwarded_hidden_states)
            second_mlp(forwarded_hidden_states)
        first_handle.remove()
        second_handle.remove()
        first_activations = torch.cat(first_activations, dim=0)
        second_activations = torch.cat(second_activations, dim=0)

        # Compute correlation
        mean_first = torch.mean(first_activations, dim=0, keepdim=True)  # (1, d_ff)
        mean_second = torch.mean(second_activations, dim=0, keepdim=True)  # (1, d_ff)
        std_first = torch.std(first_activations, dim=0, keepdim=True)  # (1, d_ff)
        std_second = torch.std(second_activations, dim=0, keepdim=True)  # (1, d_ff)
        covar = torch.mm(
            (first_activations - mean_first).t(),
            (second_activations - mean_second)
        ) / (len(forwarded_hidden_states) - 1)  # (d_ff, d_ff)
        corr = covar / (std_first.t() * std_second + FP32_EPS)  # (d_ff, d_ff)

        # Compute linear sum assignment to find maximum correlation
        if lsa:
            corr = corr.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(corr, maximize=True)
            # Normalize to [0, 1]
            return (corr[row_ind, col_ind] + 1).sum() / corr.shape[0] / 2
        else:
            return (corr.max(dim=0).values + 1).sum() / corr.shape[0] / 2

    def _compute_all_similarities_by_average_feature_correlation(
            self,
            model: SwitchTransformersForConditionalGeneration,
            batch: Dict[str, torch.Tensor],
            lsa: bool
    ):
        self.activations = {}
        handles = []
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[Merging]Registering forward hook..."):
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            handles.append(model.encoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
                self._get_mlp_activation(encoder_mlp_name)
            ))
            handles.append(model.decoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
                self._get_mlp_activation(decoder_mlp_name)
            ))
        model = model.eval()
        encoder_router_indices = {l: None for l in self.sparse_layer_indices}
        decoder_router_indices = {l: None for l in self.sparse_layer_indices}
        with torch.no_grad():
            outputs = model(**batch)
            for layer_idx in self.sparse_layer_indices:
                encoder_router_indices[layer_idx] = outputs.encoder_router_logits[layer_idx][1]
                decoder_router_indices[layer_idx] = outputs.decoder_router_logits[layer_idx][1]
        del outputs
        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()

        progress_bar = tqdm(
            range(len(self.sparse_layer_indices) * self.num_experts * (self.num_experts - 1) // 2),
            desc="[Merging]Computing similarities by average feature correlation..."
        )

        for layer_idx in self.sparse_layer_indices:
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            encoder_router_expert_index = encoder_router_indices[layer_idx].reshape(-1)
            decoder_router_expert_index = decoder_router_indices[layer_idx].reshape(-1)
            encoder_hidden_states = self.activations[encoder_mlp_name]
            decoder_hidden_states = self.activations[decoder_mlp_name]
            with torch.no_grad():
                for i in range(self.num_experts):
                    for j in range(i + 1, self.num_experts):
                        similarity = self._compute_similarity_by_average_correlation(
                            model.encoder.block[layer_idx].layer[-1].mlp.experts[f"expert_{i}"],
                            model.encoder.block[layer_idx].layer[-1].mlp.experts[f"expert_{j}"],
                            torch.cat([
                                encoder_hidden_states[encoder_router_expert_index == i],
                                encoder_hidden_states[encoder_router_expert_index == j]
                            ], dim=0),
                            lsa=lsa
                        )
                        self.save_similarity(encoder_mlp_name, i, j, similarity)
                        similarity = self._compute_similarity_by_average_correlation(
                            model.decoder.block[layer_idx].layer[-1].mlp.experts[f"expert_{i}"],
                            model.decoder.block[layer_idx].layer[-1].mlp.experts[f"expert_{j}"],
                            torch.cat([
                                decoder_hidden_states[decoder_router_expert_index == i],
                                decoder_hidden_states[decoder_router_expert_index == j]
                            ], dim=0),
                            lsa=lsa
                        )
                        self.save_similarity(decoder_mlp_name, i, j, similarity)
                        progress_bar.update(1)


def _merge_mlp_experts_by_averaging(
        mlp: SwitchTransformersSparseMLP,
        group_labels: torch.LongTensor,
        permute: bool,
        permute_strategy: str,
        forwarded_hidden_states: Optional[Tuple[torch.Tensor]] = None,
) -> SwitchTransformersSparseMLP:
    for label in group_labels.unique():
        expert_indices = torch.where(group_labels == label)[0]
        if permute and permute_strategy == "weight-matching":
            for expert_idx in expert_indices[1:]:
                perm = compute_switch_permutation_by_weight_matching(
                    reference_mlp=mlp.experts[f"expert_{expert_indices[0]}"],
                    target_mlp=mlp.experts[f"expert_{expert_idx}"],
                    include_wo=True
                )
                mlp.experts[f"expert_{expert_idx}"] = permute_switch_mlp_dense_expert_(
                    mlp.experts[f"expert_{expert_idx}"], perm
                )
        elif permute and permute_strategy == "activation-matching":
            group_forwarded_hidden_states = torch.cat([
                forwarded_hidden_states[expert_idx] for expert_idx in expert_indices
            ], dim=0)
            for expert_idx in expert_indices[1:]:
                perm = compute_switch_permutation_by_activation_matching(
                    reference_mlp=mlp.experts[f"expert_{expert_indices[0]}"],
                    target_mlp=mlp.experts[f"expert_{expert_idx}"],
                    forwarded_hidden_states=group_forwarded_hidden_states,
                )
                mlp.experts[f"expert_{expert_idx}"] = permute_switch_mlp_dense_expert_(
                    mlp.experts[f"expert_{expert_idx}"], perm
                )
        elif permute:
            raise ValueError(f"Unknown permute strategy: {permute_strategy}")

        with torch.no_grad():
            wi_weight = torch.mean(
                torch.stack([mlp.experts[f"expert_{expert_idx}"].wi.weight for expert_idx in expert_indices]),
                dim=0
            )
            wo_weight = torch.mean(
                torch.stack([mlp.experts[f"expert_{expert_idx}"].wo.weight for expert_idx in expert_indices]),
                dim=0
            )
            mlp.experts[f"expert_{expert_indices[0]}"].wi.weight.copy_(wi_weight)
            mlp.experts[f"expert_{expert_indices[0]}"].wo.weight.copy_(wo_weight)
            for expert_idx in expert_indices[1:]:
                # Binding merged experts to the first of them
                mlp.experts[f"expert_{expert_idx}"] = mlp.experts[f"expert_{expert_indices[0]}"]
    return mlp


def _merge_mlp_experts_by_fisher_weighted_averaging(
        mlp: SwitchTransformersSparseMLP,
        group_labels: torch.LongTensor,
        name_prefix: str,
        experts_fisher_state_dict: Dict[str, torch.Tensor],
        permute: Optional[bool] = False
) -> SwitchTransformersSparseMLP:
    for label in group_labels.unique():
        expert_indices = torch.where(group_labels == label)[0]
        if permute:
            for expert_idx in expert_indices[1:]:
                perm = compute_switch_permutation_by_weight_matching(
                    reference_mlp=mlp.experts[f"expert_{expert_indices[0]}"],
                    target_mlp=mlp.experts[f"expert_{expert_idx}"],
                    include_wo=True
                )
                mlp.experts[f"expert_{expert_idx}"] = permute_switch_mlp_dense_expert_(
                    mlp.experts[f"expert_{expert_idx}"], perm
                )
        with torch.no_grad():
            wi_weight_list = torch.stack(
                [mlp.experts[f"expert_{expert_idx}"].wi.weight for expert_idx in expert_indices], dim=0
            )
            wo_weight_list = torch.stack(
                [mlp.experts[f"expert_{expert_idx}"].wo.weight for expert_idx in expert_indices], dim=0
            )
            wi_fisher_list = torch.stack(
                [experts_fisher_state_dict[f"{name_prefix}.expert_{expert_idx}.wi.weight"]
                 for expert_idx in expert_indices], dim=0
            )
            wo_fisher_list = torch.stack(
                [experts_fisher_state_dict[f"{name_prefix}.expert_{expert_idx}.wo.weight"]
                 for expert_idx in expert_indices], dim=0
            )

            wi_weight = torch.sum(wi_weight_list * wi_fisher_list, dim=0) / (
                    torch.sum(wi_fisher_list, dim=0) + FP32_EPS)
            wo_weight = torch.sum(wo_weight_list * wo_fisher_list, dim=0) / (
                    torch.sum(wo_fisher_list, dim=0) + FP32_EPS)

            mlp.experts[f"expert_{expert_indices[0]}"].wi.weight.copy_(wi_weight)
            mlp.experts[f"expert_{expert_indices[0]}"].wo.weight.copy_(wo_weight)
            for expert_idx in expert_indices[1:]:
                # Binding merged experts to the first of them
                mlp.experts[f"expert_{expert_idx}"] = mlp.experts[f"expert_{expert_indices[0]}"]
    return mlp


def merge_by_groups(
        model: SwitchTransformersForConditionalGeneration,
        grouper: ExpertsGrouperForSwitch,
        strategy: str = "average",
        fisher_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        encoder_merging_layers: Optional[List[int]] = None,
        decoder_merging_layers: Optional[List[int]] = None,
        permute: Optional[bool] = False,
        permute_strategy: Optional[str] = "weight-matching",
        dataloader: Optional[DataLoader] = None,
) -> SwitchTransformersForConditionalGeneration:
    """
    Merge experts by groups, strategies include:
        1. "average": simply average the weights of experts in the same group.
        2. "fisher": average the weights of experts in the same group weighted by (diagnonal) fisher information.
    Parameters
    ----------
    model: SwitchTransformersForConditionalGeneration
        The model to merge experts.
    grouper: ExpertsGrouperForSwitch
        The grouper to group experts, supposed to have been called `grouper.compute_all_usages()` and
            one of `grouper.group_experts()` (i.e. have grouped labels).
    fisher_state_dict: Optional[Dict[str, torch.Tensor]]
        The fisher state dict, if None, can only use (vanilla )averaging strategy, else use fisher averaging strategy.
    strategy: str
        The strategy to merge experts, one of ["replace", "drop"].
    encoder_merging_layers: Optional[List[int]]
        The encoder layers to merge experts, if None, merge all layers.
    decoder_merging_layers: Optional[List[int]]
        The decoder layers to merge experts, if None, merge all layers.
    permute: Optional[bool]
        Whether to permute the experts in the same group.
    permute_strategy: Optional[str]
        The strategy to permute experts, one of ["weight-matching", "random"].
    dataloader: Optional[DataLoader]
        The dataloader to compute activations, only used when `strategy` is "activation-matching".
    """
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
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            forwarded_hidden_states[encoder_mlp_name] = []
            forwarded_hidden_states[decoder_mlp_name] = []
            handles.append(model.encoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
                _get_activation_hook(encoder_mlp_name))
            )
            handles.append(model.decoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
                _get_activation_hook(decoder_mlp_name))
            )

        # {name: values}, values will be of shape (len(dataloader), batch_size * seq_len)
        router_indices = {name: [] for name in forwarded_hidden_states.keys()}
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="[Merging]Computing activations..."):
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(**batch)
                for layer_idx in grouper.sparse_layer_indices:
                    router_indices[f"encoder.block.{layer_idx}.layer.1.mlp"].append(
                        outputs.encoder_router_logits[layer_idx][1].reshape(-1)
                    )
                    router_indices[f"decoder.block.{layer_idx}.layer.2.mlp"].append(
                        outputs.decoder_router_logits[layer_idx][1].reshape(-1)
                    )

        for handle in handles:
            handle.remove()

    num_experts = grouper.num_experts

    if permute:
        print(f"[Merging]Permutation is enabled, will permute experts by {permute_strategy} in the same group.")
    if strategy == "average":
        for layer_idx in tqdm(grouper.sparse_layer_indices,
                              desc="[Merging]Merging experts with averaging..."):
            if encoder_merging_layers is None or layer_idx in encoder_merging_layers:
                encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
                encoder_group_labels = grouper.group_state_dict()[encoder_mlp_name]
                if permute_strategy == "activation-matching":
                    layer_forwarded_hidden_states = tuple()
                    for expert_idx in range(num_experts):
                        layer_forwarded_hidden_states += (
                            torch.cat(
                                [forwarded_hidden_states[encoder_mlp_name][i][
                                     router_indices[encoder_mlp_name][i] == expert_idx]
                                 for i in range(len(dataloader))], dim=0),
                        )
                    model.encoder.block[layer_idx].layer[-1].mlp = _merge_mlp_experts_by_averaging(
                        mlp=model.encoder.block[layer_idx].layer[-1].mlp,
                        group_labels=encoder_group_labels,
                        permute=permute,
                        permute_strategy=permute_strategy,
                        forwarded_hidden_states=layer_forwarded_hidden_states
                    )
                else:
                    model.encoder.block[layer_idx].layer[-1].mlp = _merge_mlp_experts_by_averaging(
                        mlp=model.encoder.block[layer_idx].layer[-1].mlp,
                        group_labels=encoder_group_labels,
                        permute=permute,
                        permute_strategy=permute_strategy,
                    )
            if decoder_merging_layers is None or layer_idx in decoder_merging_layers:
                decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
                decoder_group_labels = grouper.group_state_dict()[decoder_mlp_name]
                if permute_strategy == "activation-matching":
                    layer_forwarded_hidden_states = tuple()
                    for expert_idx in range(num_experts):
                        layer_forwarded_hidden_states += (
                            torch.cat(
                                [forwarded_hidden_states[decoder_mlp_name][i][
                                     router_indices[decoder_mlp_name][i] == expert_idx]
                                 for i in range(len(dataloader))], dim=0),
                        )
                    model.decoder.block[layer_idx].layer[-1].mlp = _merge_mlp_experts_by_averaging(
                        mlp=model.decoder.block[layer_idx].layer[-1].mlp,
                        group_labels=decoder_group_labels,
                        permute=permute,
                        permute_strategy=permute_strategy,
                        forwarded_hidden_states=layer_forwarded_hidden_states
                    )
                else:
                    model.decoder.block[layer_idx].layer[-1].mlp = _merge_mlp_experts_by_averaging(
                        mlp=model.decoder.block[layer_idx].layer[-1].mlp,
                        group_labels=decoder_group_labels,
                        permute=permute,
                        permute_strategy=permute_strategy,
                    )
    elif strategy == "fisher":
        if fisher_state_dict is None:
            raise ValueError("[Merging]Fisher averaging strategy requires fisher_state_dict")
        fisher_state_dict = {k: v.cuda() for k, v in fisher_state_dict.items()}
        for layer_idx in tqdm(grouper.sparse_layer_indices, desc="[Merging]Merging experts with fisher..."):
            if encoder_merging_layers is None or layer_idx in encoder_merging_layers:
                encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
                encoder_group_labels = grouper._group_state_dict[encoder_mlp_name]
                model.encoder.block[layer_idx].layer[-1].mlp = _merge_mlp_experts_by_fisher_weighted_averaging(
                    mlp=model.encoder.block[layer_idx].layer[-1].mlp,
                    group_labels=encoder_group_labels,
                    name_prefix=encoder_mlp_name + ".experts",
                    experts_fisher_state_dict=fisher_state_dict,
                    permute=permute
                )
            if decoder_merging_layers is None or layer_idx in decoder_merging_layers:
                decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
                decoder_group_labels = grouper._group_state_dict[decoder_mlp_name]
                model.decoder.block[layer_idx].layer[-1].mlp = _merge_mlp_experts_by_fisher_weighted_averaging(
                    mlp=model.decoder.block[layer_idx].layer[-1].mlp,
                    group_labels=decoder_group_labels,
                    name_prefix=decoder_mlp_name + ".experts",
                    experts_fisher_state_dict=fisher_state_dict,
                    permute=permute
                )
    return model


def _prune_mlp_experts_by_replacing_non_core(
        mlp: SwitchTransformersSparseMLP,
        group_labels: torch.LongTensor,
        core_expert_indices: List[int],
) -> SwitchTransformersSparseMLP:
    """
    Prune non-core experts by replacing them with core experts
    """
    for core_idx in core_expert_indices:
        with torch.no_grad():
            experts_indices = torch.where(group_labels == group_labels[core_idx])[0]
            core_expert = mlp.experts[f"expert_{core_idx}"]
            for expert_idx in experts_indices:
                if expert_idx == core_idx:
                    continue
                mlp.experts[f"expert_{expert_idx}"] = core_expert

    return mlp


def _prune_mlp_experts_by_dropping_non_core(
        mlp: SwitchTransformersSparseMLP,
        core_expert_indices: List[int],
) -> SwitchTransformersSparseMLP:
    non_core_expert_mask = torch.ones(len(mlp.experts), dtype=torch.bool)
    non_core_expert_mask[core_expert_indices] = False
    non_core_expert_indices = torch.where(non_core_expert_mask)[0]
    mask_value = -1e6

    def _custom_forward(self, hidden_states: torch.Tensor) -> Tuple:
        _, router_logits = self._compute_router_probabilities(hidden_states)

        router_logits = router_logits.masked_fill(non_core_expert_mask.to(router_logits.device), mask_value)
        router_probs = F.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)

        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = F.one_hot(expert_index, num_classes=self.num_experts)
        # Since experts are pruned, no need to do max-capacity dropping
        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits

    mlp.router.forward = _custom_forward.__get__(mlp.router, SwitchTransformersTop1Router)

    for expert_idx in non_core_expert_indices:
        # fake expert, this will raise error if used by mistake
        mlp.experts[f"expert_{expert_idx}"] = torch.nn.Linear(mlp.router.classifier.in_features, 1)

    return mlp


def prune_non_core_experts_by_groups(
        model: SwitchTransformersForConditionalGeneration,
        grouper: ExpertsGrouperForSwitch,
        core_experts: Dict[str, List[int]],
        strategy: str = "replace",
        encoder_merging_layers: Optional[List[int]] = None,
        decoder_merging_layers: Optional[List[int]] = None,
) -> SwitchTransformersForConditionalGeneration:
    """
    Simply prune non-core experts by replacing them with core experts, or dropping them.
    (The difference between replacing and dropping is whether to route to the pruned experts and redirect to core experts)

    Parameters
    ----------
    model: SwitchTransformersForConditionalGeneration
        The model to merge experts
    grouper: ExpertsGrouperForSwitch
        The grouper to group experts, supposed to have been called `grouper.compute_all_usages()` and
            one of `grouper.group_experts()` (i.e. have grouped labels)
    core_experts: Dict[str, List[int]]
        The core experts dict, normally returned by `grouper.group_experts_into_clusters_by_routing_guided_globally()`
    strategy: str
        The strategy to merge experts, one of ["replace", "drop"]
    encoder_merging_layers: Optional[List[int]]
        The encoder layers to merge experts, if None, merge all layers
    decoder_merging_layers: Optional[List[int]]
        The decoder layers to merge experts, if None, merge all layers
    """
    if strategy == "replace":
        for layer_idx in tqdm(grouper.sparse_layer_indices, desc="[Merging]Pruning non-core experts by replacing..."):
            if encoder_merging_layers is None or layer_idx in encoder_merging_layers:
                encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
                encoder_group_labels = grouper.group_state_dict()[encoder_mlp_name]
                model.encoder.block[layer_idx].layer[-1].mlp = _prune_mlp_experts_by_replacing_non_core(
                    mlp=model.encoder.block[layer_idx].layer[-1].mlp,
                    group_labels=encoder_group_labels,
                    core_expert_indices=core_experts[encoder_mlp_name]
                )
            if decoder_merging_layers is None or layer_idx in decoder_merging_layers:
                decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
                decoder_group_labels = grouper.group_state_dict()[decoder_mlp_name]
                model.decoder.block[layer_idx].layer[-1].mlp = _prune_mlp_experts_by_replacing_non_core(
                    mlp=model.decoder.block[layer_idx].layer[-1].mlp,
                    group_labels=decoder_group_labels,
                    core_expert_indices=core_experts[decoder_mlp_name]
                )
    elif strategy == "drop":
        for layer_idx in tqdm(grouper.sparse_layer_indices, desc="[Merging]Pruning non-core experts by dropping..."):
            if encoder_merging_layers is None or layer_idx in encoder_merging_layers:
                encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
                model.encoder.block[layer_idx].layer[-1].mlp = _prune_mlp_experts_by_dropping_non_core(
                    mlp=model.encoder.block[layer_idx].layer[-1].mlp,
                    core_expert_indices=core_experts[encoder_mlp_name]
                )
            if decoder_merging_layers is None or layer_idx in decoder_merging_layers:
                decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
                model.decoder.block[layer_idx].layer[-1].mlp = _prune_mlp_experts_by_dropping_non_core(
                    mlp=model.decoder.block[layer_idx].layer[-1].mlp,
                    core_expert_indices=core_experts[decoder_mlp_name]
                )
    return model


def _merge_mlp_experts_by_usage_frequency_weighting(
        mlp: SwitchTransformersSparseMLP,
        group_labels: torch.LongTensor,
        usage_frequencies: torch.Tensor,
        permute: bool,
) -> SwitchTransformersSparseMLP:
    for label in group_labels.unique():
        expert_indices = torch.where(group_labels == label)[0]
        if permute:
            for expert_idx in expert_indices[1:]:
                perm = compute_switch_permutation_by_weight_matching(
                    reference_mlp=mlp.experts[f"expert_{expert_indices[0]}"],
                    target_mlp=mlp.experts[f"expert_{expert_idx}"],
                    include_wo=True
                )
                mlp.experts[f"expert_{expert_idx}"] = permute_switch_mlp_dense_expert_(
                    mlp.experts[f"expert_{expert_idx}"], perm
                )
        with torch.no_grad():
            wi_weight_list = torch.stack(
                [mlp.experts[f"expert_{expert_idx}"].wi.weight * usage_frequencies[expert_idx] for expert_idx in
                 expert_indices], dim=0
            )
            wo_weight_list = torch.stack(
                [mlp.experts[f"expert_{expert_idx}"].wo.weight * usage_frequencies[expert_idx] for expert_idx in
                 expert_indices], dim=0
            )
            wi_weight = torch.sum(wi_weight_list, dim=0) / (
                    torch.sum(usage_frequencies[expert_indices], dim=0) + FP32_EPS)
            wo_weight = torch.sum(wo_weight_list, dim=0) / (
                    torch.sum(usage_frequencies[expert_indices], dim=0) + FP32_EPS)
            mlp.experts[f"expert_{expert_indices[0]}"].wi.weight.copy_(wi_weight)
            mlp.experts[f"expert_{expert_indices[0]}"].wo.weight.copy_(wo_weight)
            for expert_idx in expert_indices[1:]:
                # Binding merged experts to the first of them
                mlp.experts[f"expert_{expert_idx}"] = mlp.experts[f"expert_{expert_indices[0]}"]
    return mlp


def _merge_mlp_experts_by_usage_frequency_weighting_within_and_across_models(
        mlp: SwitchTransformersSparseMLP,
        group_labels: torch.LongTensor,
        usage_frequencies: torch.Tensor
) -> SwitchTransformersSparseMLP:
    for label in group_labels.unique():
        expert_indices = torch.where(group_labels == label)[0]
        with torch.no_grad():
            merged_mlp = merge_switch_mlp_by_weight_matching_within_and_across_models(
                mlp_list=[mlp.experts[f"expert_{expert_idx}"] for expert_idx in expert_indices],
                average_coefs=usage_frequencies[expert_indices].tolist()
            )
            mlp.experts[f"expert_{expert_indices[0]}"].wi.weight.copy_(merged_mlp.wi.weight)
            mlp.experts[f"expert_{expert_indices[0]}"].wo.weight.copy_(merged_mlp.wo.weight)
            del merged_mlp
            torch.cuda.empty_cache()
            for expert_idx in expert_indices[1:]:
                # Binding merged experts to the first of them
                mlp.experts[f"expert_{expert_idx}"] = mlp.experts[f"expert_{expert_indices[0]}"]
    return mlp


def merge_by_groups_with_usage_frequency_weighting(
        model: SwitchTransformersForConditionalGeneration,
        grouper: ExpertsGrouperForSwitch,
        strategy: str = "normal",
        encoder_merging_layers: Optional[List[int]] = None,
        decoder_merging_layers: Optional[List[int]] = None,
        permute: Optional[bool] = False,
        within_and_across_models: Optional[bool] = False,
) -> SwitchTransformersForConditionalGeneration:
    """
    Merge experts by usage-frequency-weighted averaging, strategies include:
        1. normal: merge experts in each group by usage-frequency-weighted averaging.
        2. reversed: reverse usage frequencies by 1 - usage_frequency and merge experts in each group by
                        usage-frequency-weighted averaging.
        3. random: randomly initialize usage frequencies and merge experts in each group by
                        usage-frequency-weighted averaging.

    Parameters
    ----------
    model: SwitchTransformersForConditionalGeneration
        The model to merge experts
    grouper: ExpertsGrouperForSwitch
        The grouper to group experts, supposed to have been called `grouper.compute_all_usages()` and
            one of `grouper.group_experts()` (i.e. have grouped labels)
    strategy: str
        The strategy to merge experts, one of ["normal", "reversed", "random"]
    encoder_merging_layers: Optional[List[int]]
        The encoder layers to merge experts, if None, merge all layers
    decoder_merging_layers: Optional[List[int]]
        The decoder layers to merge experts, if None, merge all layers
    permute: Optional[bool]
        Whether to permute the experts in the same group, only availabel when `within_and_across_models` is False.
    within_and_across_models: Optional[bool]
        Whether to merge experts within and across models.
    """
    if permute:
        print("[Merging]Permutation is enabled, will permute experts in the same group.")
    usage_frequency_dict = grouper.usage_frequency_state_dict()
    if strategy == "reversed":
        for key, value in usage_frequency_dict.items():
            usage_frequency_dict[key] = 1 - value
    elif strategy == "random":
        for key, value in usage_frequency_dict.items():
            usage_frequency_dict[key] = torch.rand_like(value)
    elif strategy != "normal":
        raise ValueError(f"[Merging]Unknown strategy {strategy}")

    for layer_idx in tqdm(
            grouper.sparse_layer_indices,
            desc=f"[Merging]Merging experts with {strategy} usage-frequency-weighted averaging..."
    ):
        if encoder_merging_layers is None or layer_idx in encoder_merging_layers:
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            encoder_group_labels = grouper.group_state_dict()[encoder_mlp_name]
            usage_frequencies = usage_frequency_dict[encoder_mlp_name]
            model.encoder.block[layer_idx].layer[-1].mlp = _merge_mlp_experts_by_usage_frequency_weighting(
                mlp=model.encoder.block[layer_idx].layer[-1].mlp,
                group_labels=encoder_group_labels,
                usage_frequencies=usage_frequencies,
                permute=permute
            ) if not within_and_across_models else _merge_mlp_experts_by_usage_frequency_weighting_within_and_across_models(
                mlp=model.encoder.block[layer_idx].layer[-1].mlp,
                group_labels=encoder_group_labels,
                usage_frequencies=usage_frequencies,
            )
        if decoder_merging_layers is None or layer_idx in decoder_merging_layers:
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            decoder_group_labels = grouper.group_state_dict()[decoder_mlp_name]
            usage_frequencies = usage_frequency_dict[decoder_mlp_name]
            model.decoder.block[layer_idx].layer[-1].mlp = _merge_mlp_experts_by_usage_frequency_weighting(
                mlp=model.decoder.block[layer_idx].layer[-1].mlp,
                group_labels=decoder_group_labels,
                usage_frequencies=usage_frequencies,
                permute=permute
            ) if not within_and_across_models else _merge_mlp_experts_by_usage_frequency_weighting_within_and_across_models(
                mlp=model.decoder.block[layer_idx].layer[-1].mlp,
                group_labels=decoder_group_labels,
                usage_frequencies=usage_frequencies,
            )
    return model


def _merge_mlp_experts_within_and_across_models(
        mlp: SwitchTransformersSparseMLP,
        group_labels: torch.LongTensor,
        forwarded_hidden_states: Tuple[torch.Tensor],
        dominant_alone: bool,
        core_expert_indices: Optional[List[int]] = None,
        usage_frequencies: Optional[torch.Tensor] = None,
) -> SwitchTransformersSparseMLP:
    """
    Merge grouped experts within and across models.

    Parameters
    ----------
    mlp: SwitchTransformersSparseMLP
        The mlp to merge experts.
    group_labels: torch.LongTensor
        The group labels of experts.
    forwarded_hidden_states: Tuple[torch.Tensor]
        The forwarded hidden states of each expert, should be of length num_experts
    dominant_alone: bool
        Whether to merge the dominant expert alone.
        If True, the merging process in a group will be done in two steps:
            1. Merge all experts except the dominant one.
            2. Merge the dominant expert with the merged expert in step 1.
    core_expert_indices: List[int]

    Returns
    -------
    mlp: SwitchTransformersSparseMLP
        The merged mlp.
    """
    if dominant_alone and core_expert_indices is None:
        raise ValueError("[Merging]dominant_alone is True, but core_expert_indices is None")

    for label in group_labels.unique():
        expert_indices = torch.where(group_labels == label)[0]
        with torch.no_grad():
            if dominant_alone:
                group_core_expert_indices = torch.stack([
                    idx for idx in expert_indices if idx in core_expert_indices])
                to_skip = False
                if len(group_core_expert_indices) == len(expert_indices):
                    merged_expert = mlp.experts[f"expert_{expert_indices[0]}"]
                    to_skip = True
                elif usage_frequencies is not None and len(group_core_expert_indices) == 1:
                    non_core_usage_sum = torch.sum(
                        usage_frequencies[[expert_idx.item() for expert_idx in
                                           expert_indices if expert_idx not in group_core_expert_indices]]).item()
                    if non_core_usage_sum == 0:
                        merged_expert = mlp.experts[f"expert_{group_core_expert_indices[0]}"]
                        to_skip = True
                    else:
                        to_skip = False
                if not to_skip:
                    # Stage 1: merge all experts except the dominant one
                    group_forwarded_hidden_states = torch.cat([
                        forwarded_hidden_states[expert_idx] for expert_idx in expert_indices if
                        expert_idx not in group_core_expert_indices
                    ], dim=0)
                    if usage_frequencies is not None:
                        non_core_usages = usage_frequencies[[expert_idx.item() for expert_idx in expert_indices if
                                                             expert_idx not in group_core_expert_indices]]
                    merged_expert = merge_switch_mlp_by_activation_matching_within_and_across_models(
                        mlp_list=[mlp.experts[f"expert_{expert_idx}"] for expert_idx in expert_indices if
                                  expert_idx not in group_core_expert_indices],
                        forwarded_hidden_states=group_forwarded_hidden_states,
                        average_coefs=non_core_usages.tolist() if usage_frequencies is not None else None
                    )
                    # Stage 2: merge the dominant expert with the merged expert in stage 1
                    group_forwarded_hidden_states = torch.cat([
                        forwarded_hidden_states[expert_idx] for expert_idx in expert_indices
                    ], dim=0)
                    if usage_frequencies is not None:
                        core_usages = usage_frequencies[group_core_expert_indices]
                        non_core_usage_sum = torch.sum(non_core_usages).item()
                    merged_expert = merge_switch_mlp_by_activation_matching_within_and_across_models(
                        mlp_list=[merged_expert] + [mlp.experts[f"expert_{expert_idx}"] for expert_idx in
                                                    group_core_expert_indices],
                        forwarded_hidden_states=group_forwarded_hidden_states,
                        average_coefs=[non_core_usage_sum] + core_usages.tolist(
                        ) if usage_frequencies is not None else None
                    )
            else:
                # Merge all experts in the group
                group_forwarded_hidden_states = torch.cat([
                    forwarded_hidden_states[expert_idx] for expert_idx in expert_indices
                ], dim=0)
                merged_expert = merge_switch_mlp_by_activation_matching_within_and_across_models(
                    mlp_list=[mlp.experts[f"expert_{expert_idx}"] for expert_idx in expert_indices],
                    forwarded_hidden_states=group_forwarded_hidden_states,
                    average_coefs=usage_frequencies[expert_indices].tolist() if usage_frequencies is not None else None
                )
            mlp.experts[f"expert_{expert_indices[0]}"].wi.weight.copy_(merged_expert.wi.weight)
            mlp.experts[f"expert_{expert_indices[0]}"].wo.weight.copy_(merged_expert.wo.weight)
            for expert_idx in expert_indices[1:]:
                # Binding merged experts to the first of them
                mlp.experts[f"expert_{expert_idx}"] = mlp.experts[f"expert_{expert_indices[0]}"]

    return mlp


def merge_by_groups_within_and_across_models(
        switch_model: SwitchTransformersForConditionalGeneration,
        grouper: ExpertsGrouperForSwitch,
        dataloader: DataLoader,
        encoder_merging_layers: Optional[List[int]] = None,
        decoder_merging_layers: Optional[List[int]] = None,
        dominant_alone: Optional[bool] = False,
        core_experts: Optional[Dict[str, List[int]]] = None,
        usage_weighted: Optional[bool] = False,
) -> SwitchTransformersForConditionalGeneration:
    # {name: values}, values  will be of shape (len(dataloader), batch_size * seq_len, d_ff)
    forwarded_hidden_states = dict()

    usage_frequencies = grouper.usage_frequency_state_dict()

    switch_model.eval().cuda()
    handles = []

    def _get_activation_hook(name):
        def hook(module, input, output):
            forwarded_hidden_states[name].append(input[0].detach().reshape(-1, input[0].shape[-1]))

        return hook

    for layer_idx in tqdm(
            grouper.sparse_layer_indices,
            desc=f"[Merging]Registering forward hook..."
    ):
        encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
        decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
        forwarded_hidden_states[encoder_mlp_name] = []
        forwarded_hidden_states[decoder_mlp_name] = []
        handles.append(switch_model.encoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
            _get_activation_hook(encoder_mlp_name))
        )
        handles.append(switch_model.decoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
            _get_activation_hook(decoder_mlp_name))
        )

    # {name: values}, values will be of shape (len(dataloader), batch_size * seq_len)
    router_indices = {name: [] for name in forwarded_hidden_states.keys()}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Merging]Computing activations..."):
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = switch_model(**batch)
            for layer_idx in grouper.sparse_layer_indices:
                router_indices[f"encoder.block.{layer_idx}.layer.1.mlp"].append(
                    outputs.encoder_router_logits[layer_idx][1].reshape(-1)
                )
                router_indices[f"decoder.block.{layer_idx}.layer.2.mlp"].append(
                    outputs.decoder_router_logits[layer_idx][1].reshape(-1)
                )

    for handle in handles:
        handle.remove()

    num_experts = grouper.num_experts
    for layer_idx in tqdm(
            grouper.sparse_layer_indices,
            desc=f"[Merging]Merging by groups within and across experts..."
    ):
        if encoder_merging_layers is None or layer_idx in encoder_merging_layers:
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            encoder_group_labels = grouper.group_state_dict()[encoder_mlp_name]
            layer_forwarded_hidden_states = tuple()
            for expert_idx in range(num_experts):
                layer_forwarded_hidden_states += (
                    torch.cat(
                        [forwarded_hidden_states[encoder_mlp_name][i][router_indices[encoder_mlp_name][i] == expert_idx]
                         for i in range(len(dataloader))], dim=0),
                )
            switch_model.encoder.block[layer_idx].layer[
                -1].mlp = _merge_mlp_experts_within_and_across_models(
                mlp=switch_model.encoder.block[layer_idx].layer[-1].mlp,
                group_labels=encoder_group_labels,
                forwarded_hidden_states=layer_forwarded_hidden_states,
                dominant_alone=dominant_alone,
                core_expert_indices=core_experts[encoder_mlp_name] if core_experts is not None else None,
                usage_frequencies=usage_frequencies[encoder_mlp_name] if usage_weighted else None,
            )
        if decoder_merging_layers is None or layer_idx in decoder_merging_layers:
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            decoder_group_labels = grouper.group_state_dict()[decoder_mlp_name]
            layer_forwarded_hidden_states = tuple()
            for expert_idx in range(num_experts):
                layer_forwarded_hidden_states += (
                    torch.cat(
                        [forwarded_hidden_states[decoder_mlp_name][i][router_indices[decoder_mlp_name][i] == expert_idx]
                         for i in range(len(dataloader))], dim=0),
                )
            switch_model.decoder.block[layer_idx].layer[
                -1].mlp = _merge_mlp_experts_within_and_across_models(
                mlp=switch_model.decoder.block[layer_idx].layer[-1].mlp,
                group_labels=decoder_group_labels,
                forwarded_hidden_states=layer_forwarded_hidden_states,
                dominant_alone=dominant_alone,
                core_expert_indices=core_experts[decoder_mlp_name] if core_experts is not None else None,
                usage_frequencies=usage_frequencies[decoder_mlp_name] if usage_weighted else None,
            )

    del forwarded_hidden_states, router_indices
    torch.cuda.empty_cache()
    return switch_model


def _merge_mlp_non_core_experts_by_usage_frequency_weighting(
        mlp: SwitchTransformersSparseMLP,
        group_labels: torch.LongTensor,
        usage_frequencies: torch.Tensor,
        core_experts: List[int],
) -> SwitchTransformersSparseMLP:
    for label in group_labels.unique():
        expert_indices = torch.where(group_labels == label)[0]
        # Only merge non-core experts
        expert_indices = torch.tensor(
            [expert_idx.item() for expert_idx in expert_indices if expert_idx not in core_experts]
        )
        if len(expert_indices) == 0:
            continue
        with torch.no_grad():
            wi_weight_list = torch.stack(
                [mlp.experts[f"expert_{expert_idx}"].wi.weight * usage_frequencies[expert_idx] for expert_idx in
                 expert_indices], dim=0
            )
            wo_weight_list = torch.stack(
                [mlp.experts[f"expert_{expert_idx}"].wo.weight * usage_frequencies[expert_idx] for expert_idx in
                 expert_indices], dim=0
            )
            wi_weight = torch.sum(wi_weight_list, dim=0) / (
                    torch.sum(usage_frequencies[expert_indices], dim=0) + FP32_EPS)
            wo_weight = torch.sum(wo_weight_list, dim=0) / (
                    torch.sum(usage_frequencies[expert_indices], dim=0) + FP32_EPS)
            mlp.experts[f"expert_{expert_indices[0]}"].wi.weight.copy_(wi_weight)
            mlp.experts[f"expert_{expert_indices[0]}"].wo.weight.copy_(wo_weight)
            for expert_idx in expert_indices[1:]:
                # Binding merged experts to the first of them
                mlp.experts[f"expert_{expert_idx}"] = mlp.experts[f"expert_{expert_indices[0]}"]
    return mlp


def merge_non_core_by_groups_with_usage_frequency_weighting(
        model: SwitchTransformersForConditionalGeneration,
        grouper: ExpertsGrouperForSwitch,
        core_experts: Dict[str, List[int]],
        encoder_merging_layers: Optional[List[int]] = None,
        decoder_merging_layers: Optional[List[int]] = None,
) -> SwitchTransformersForConditionalGeneration:
    """
    Similar to `merge_by_groups_with_usage_frequency_weighting`, but only merge non-core experts.

    Parameters
    ----------
    model: SwitchTransformersForConditionalGeneration
        The model to merge experts
    grouper: ExpertsGrouperForSwitch
        The grouper to group experts, supposed to have been called `grouper.compute_all_usages()` and
            one of `grouper.group_experts()` (i.e. have grouped labels)
    core_experts: Dict[str, List[int]]
        The core experts dict, normally returned by `grouper.group_experts_into_clusters_by_routing_guided_globally()`
    encoder_merging_layers: Optional[List[int]]
        The encoder layers to merge experts, if None, merge all layers
    decoder_merging_layers: Optional[List[int]]
        The decoder layers to merge experts, if None, merge all layers

    Returns
    -------
    model: SwitchTransformersForConditionalGeneration
        The model with merged non-core experts
    """
    usage_frequency_dict = grouper.usage_frequency_state_dict()
    for layer_idx in tqdm(
            grouper.sparse_layer_indices,
            desc=f"[Merging]Merging only non-core experts with usage-frequency-weighted averaging..."
    ):
        if encoder_merging_layers is None or layer_idx in encoder_merging_layers:
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            encoder_group_labels = grouper.group_state_dict()[encoder_mlp_name]
            usage_frequencies = usage_frequency_dict[encoder_mlp_name]
            model.encoder.block[layer_idx].layer[-1].mlp = _merge_mlp_non_core_experts_by_usage_frequency_weighting(
                mlp=model.encoder.block[layer_idx].layer[-1].mlp,
                group_labels=encoder_group_labels,
                usage_frequencies=usage_frequencies,
                core_experts=core_experts[encoder_mlp_name]
            )
        if decoder_merging_layers is None or layer_idx in decoder_merging_layers:
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            decoder_group_labels = grouper.group_state_dict()[decoder_mlp_name]
            usage_frequencies = usage_frequency_dict[decoder_mlp_name]
            model.decoder.block[layer_idx].layer[-1].mlp = _merge_mlp_non_core_experts_by_usage_frequency_weighting(
                mlp=model.decoder.block[layer_idx].layer[-1].mlp,
                group_labels=decoder_group_labels,
                usage_frequencies=usage_frequencies,
                core_experts=core_experts[decoder_mlp_name]
            )
    return model


def _merge_mlp_non_core_experts_within_and_across_models(
        mlp: SwitchTransformersSparseMLP,
        group_labels: torch.LongTensor,
        forwarded_hidden_states: Tuple[torch.Tensor],
        core_expert_indices: Optional[List[int]] = None,
) -> SwitchTransformersSparseMLP:
    """
    Merge grouped experts within and across models, only merge non-core experts. (N groups 2N experts)

    Parameters
    ----------
    mlp: SwitchTransformersSparseMLP
        The mlp to merge experts.
    group_labels: torch.LongTensor
        The group labels of experts.
    forwarded_hidden_states: Tuple[torch.Tensor]
        The forwarded hidden states of each expert, should be of length num_experts
    core_expert_indices: List[int]

    Returns
    -------
    mlp: SwitchTransformersSparseMLP
        The merged mlp.
    """
    for label in group_labels.unique():
        expert_indices = torch.where(group_labels == label)[0]
        # Only merge non-core experts
        expert_indices = torch.tensor(
            [expert_idx.item() for expert_idx in expert_indices if expert_idx not in core_expert_indices]
        )
        with torch.no_grad():
            if len(expert_indices) == 0:
                continue
            # Merge all experts except the dominant one
            group_forwarded_hidden_states = torch.cat([
                forwarded_hidden_states[expert_idx] for expert_idx in expert_indices
            ], dim=0)
            merged_expert = merge_switch_mlp_by_activation_matching_within_and_across_models(
                mlp_list=[mlp.experts[f"expert_{expert_idx}"] for expert_idx in expert_indices],
                forwarded_hidden_states=group_forwarded_hidden_states,
            )
            mlp.experts[f"expert_{expert_indices[0]}"].wi.weight.copy_(merged_expert.wi.weight)
            mlp.experts[f"expert_{expert_indices[0]}"].wo.weight.copy_(merged_expert.wo.weight)
            for expert_idx in expert_indices[1:]:
                # Binding merged experts to the first of them
                mlp.experts[f"expert_{expert_idx}"] = mlp.experts[f"expert_{expert_indices[0]}"]

    return mlp


def merge_non_core_by_groups_within_and_across_models(
        switch_model: SwitchTransformersForConditionalGeneration,
        grouper: ExpertsGrouperForSwitch,
        dataloader: DataLoader,
        encoder_merging_layers: Optional[List[int]] = None,
        decoder_merging_layers: Optional[List[int]] = None,
        core_experts: Optional[Dict[str, List[int]]] = None,
) -> SwitchTransformersForConditionalGeneration:
    # {name: values}, values  will be of shape (len(dataloader), batch_size * seq_len, d_ff)
    forwarded_hidden_states = dict()

    switch_model.eval()
    handles = []

    def _get_activation_hook(name):
        def hook(module, input, output):
            forwarded_hidden_states[name].append(input[0].detach().reshape(-1, input[0].shape[-1]))

        return hook

    for layer_idx in tqdm(
            grouper.sparse_layer_indices,
            desc=f"[Merging]Registering forward hook..."
    ):
        encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
        decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
        forwarded_hidden_states[encoder_mlp_name] = []
        forwarded_hidden_states[decoder_mlp_name] = []
        handles.append(switch_model.encoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
            _get_activation_hook(encoder_mlp_name))
        )
        handles.append(switch_model.decoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
            _get_activation_hook(decoder_mlp_name))
        )

    # {name: values}, values will be of shape (len(dataloader), batch_size * seq_len)
    router_indices = {name: [] for name in forwarded_hidden_states.keys()}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Merging]Computing activations..."):
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = switch_model(**batch)
            for layer_idx in grouper.sparse_layer_indices:
                router_indices[f"encoder.block.{layer_idx}.layer.1.mlp"].append(
                    outputs.encoder_router_logits[layer_idx][1].reshape(-1)
                )
                router_indices[f"decoder.block.{layer_idx}.layer.2.mlp"].append(
                    outputs.decoder_router_logits[layer_idx][1].reshape(-1)
                )

    for handle in handles:
        handle.remove()

    num_experts = grouper.num_experts
    for layer_idx in tqdm(
            grouper.sparse_layer_indices,
            desc=f"[Merging]Merging non core by groups within and across experts..."
    ):
        if encoder_merging_layers is None or layer_idx in encoder_merging_layers:
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            encoder_group_labels = grouper.group_state_dict()[encoder_mlp_name]
            layer_forwarded_hidden_states = tuple()
            for expert_idx in range(num_experts):
                layer_forwarded_hidden_states += (
                    torch.cat(
                        [forwarded_hidden_states[encoder_mlp_name][i][router_indices[encoder_mlp_name][i] == expert_idx]
                         for i in range(len(dataloader))], dim=0),
                )
            switch_model.encoder.block[layer_idx].layer[
                -1].mlp = _merge_mlp_non_core_experts_within_and_across_models(
                mlp=switch_model.encoder.block[layer_idx].layer[-1].mlp,
                group_labels=encoder_group_labels,
                forwarded_hidden_states=layer_forwarded_hidden_states,
                core_expert_indices=core_experts[encoder_mlp_name],
            )
        if decoder_merging_layers is None or layer_idx in decoder_merging_layers:
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            decoder_group_labels = grouper.group_state_dict()[decoder_mlp_name]
            layer_forwarded_hidden_states = tuple()
            for expert_idx in range(num_experts):
                layer_forwarded_hidden_states += (
                    torch.cat(
                        [forwarded_hidden_states[decoder_mlp_name][i][router_indices[decoder_mlp_name][i] == expert_idx]
                         for i in range(len(dataloader))], dim=0),
                )
            switch_model.decoder.block[layer_idx].layer[
                -1].mlp = _merge_mlp_non_core_experts_within_and_across_models(
                mlp=switch_model.decoder.block[layer_idx].layer[-1].mlp,
                group_labels=decoder_group_labels,
                forwarded_hidden_states=layer_forwarded_hidden_states,
                core_expert_indices=core_experts[decoder_mlp_name],
            )

    del forwarded_hidden_states, router_indices
    torch.cuda.empty_cache()
    return switch_model


def _merge_mlp_all_non_core_experts_to_one_with_frequency_weighting(
        mlp: SwitchTransformersSparseMLP,
        usage_frequencies: torch.Tensor,
        core_experts: List[int],
) -> SwitchTransformersSparseMLP:
    non_core_expert_indices = torch.tensor(
        [expert_idx for expert_idx in range(len(mlp.experts)) if expert_idx not in core_experts]
    )
    if len(non_core_expert_indices) == 0:
        return mlp
    with torch.no_grad():
        wi_weight_list = torch.stack(
            [mlp.experts[f"expert_{expert_idx}"].wi.weight * usage_frequencies[expert_idx] for expert_idx in
             non_core_expert_indices], dim=0
        )
        wo_weight_list = torch.stack(
            [mlp.experts[f"expert_{expert_idx}"].wo.weight * usage_frequencies[expert_idx] for expert_idx in
             non_core_expert_indices], dim=0
        )
        wi_weight = torch.sum(wi_weight_list, dim=0) / (
                torch.sum(usage_frequencies[non_core_expert_indices], dim=0) + FP32_EPS)
        wo_weight = torch.sum(wo_weight_list, dim=0) / (
                torch.sum(usage_frequencies[non_core_expert_indices], dim=0) + FP32_EPS)
        mlp.experts[f"expert_{non_core_expert_indices[0]}"].wi.weight.copy_(wi_weight)
        mlp.experts[f"expert_{non_core_expert_indices[0]}"].wo.weight.copy_(wo_weight)
        for expert_idx in non_core_expert_indices[1:]:
            # Binding merged experts to the first of them
            mlp.experts[f"expert_{expert_idx}"] = mlp.experts[f"expert_{non_core_expert_indices[0]}"]

    return mlp


def merge_all_non_core_to_one_with_usage_frequency_weighting(
        model: SwitchTransformersForConditionalGeneration,
        grouper: ExpertsGrouperForSwitch,
        core_experts: Dict[str, List[int]],
        encoder_merging_layers: Optional[List[int]] = None,
        decoder_merging_layers: Optional[List[int]] = None,
) -> SwitchTransformersForConditionalGeneration:
    """
    Merge all non-core experts to one, and leave core experts alone

    Parameters
    ----------
    model: SwitchTransformersForConditionalGeneration
        The model to merge experts
    grouper: ExpertsGrouperForSwitch
        The grouper to group experts, supposed to have been called `grouper.compute_all_usages()` only
            (i.e. no need to have grouped labels)
    core_experts: Dict[str, List[int]]
        The core experts dict, normally returned by `grouper.group_experts_into_clusters_by_routing_guided_globally()`
    encoder_merging_layers: Optional[List[int]]
        The encoder layers to merge experts, if None, merge all layers
    decoder_merging_layers: Optional[List[int]]
        The decoder layers to merge experts, if None, merge all layers

    Returns
    -------
    model: SwitchTransformersForConditionalGeneration
        The model with merged non-core experts
    """
    usage_frequency_dict = grouper.usage_frequency_state_dict()
    for layer_idx in tqdm(
            grouper.sparse_layer_indices,
            desc=f"[Merging]Merging all non-core experts to one with usage-frequency-weighted averaging..."
    ):
        if encoder_merging_layers is None or layer_idx in encoder_merging_layers:
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            usage_frequencies = usage_frequency_dict[encoder_mlp_name]
            model.encoder.block[layer_idx].layer[
                -1].mlp = _merge_mlp_all_non_core_experts_to_one_with_frequency_weighting(
                mlp=model.encoder.block[layer_idx].layer[-1].mlp,
                usage_frequencies=usage_frequencies,
                core_experts=core_experts[encoder_mlp_name]
            )
        if decoder_merging_layers is None or layer_idx in decoder_merging_layers:
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            usage_frequencies = usage_frequency_dict[decoder_mlp_name]
            model.decoder.block[layer_idx].layer[
                -1].mlp = _merge_mlp_all_non_core_experts_to_one_with_frequency_weighting(
                mlp=model.decoder.block[layer_idx].layer[-1].mlp,
                usage_frequencies=usage_frequencies,
                core_experts=core_experts[decoder_mlp_name]
            )
    return model


def _merge_mlp_all_non_core_experts_to_one_within_and_across_models(
        mlp: SwitchTransformersSparseMLP,
        forwarded_hidden_states: Tuple[torch.Tensor],
        core_experts: List[int],
) -> SwitchTransformersSparseMLP:
    non_core_expert_indices = torch.tensor(
        [expert_idx for expert_idx in range(len(mlp.experts)) if expert_idx not in core_experts]
    )
    if len(non_core_expert_indices) == 0:
        return mlp
    with torch.no_grad():
        non_core_forwarded_hidden_states = torch.cat([
            forwarded_hidden_states[expert_idx] for expert_idx in non_core_expert_indices
        ], dim=0)
        merged_expert = merge_switch_mlp_by_activation_matching_within_and_across_models(
            mlp_list=[mlp.experts[f"expert_{expert_idx}"] for expert_idx in non_core_expert_indices],
            forwarded_hidden_states=non_core_forwarded_hidden_states,
        )
        mlp.experts[f"expert_{non_core_expert_indices[0]}"].wi.weight.copy_(merged_expert.wi.weight)
        mlp.experts[f"expert_{non_core_expert_indices[0]}"].wo.weight.copy_(merged_expert.wo.weight)
        for expert_idx in non_core_expert_indices[1:]:
            # Binding merged experts to the first of them
            mlp.experts[f"expert_{expert_idx}"] = mlp.experts[f"expert_{non_core_expert_indices[0]}"]

    return mlp


def merge_all_non_core_to_one_within_and_across_models(
        switch_model: SwitchTransformersForConditionalGeneration,
        grouper: ExpertsGrouperForSwitch,
        dataloader: DataLoader,
        core_experts: Dict[str, List[int]],
        encoder_merging_layers: Optional[List[int]] = None,
        decoder_merging_layers: Optional[List[int]] = None,
) -> SwitchTransformersForConditionalGeneration:
    # {name: values}, values  will be of shape (len(dataloader), batch_size * seq_len, d_ff)
    forwarded_hidden_states = dict()

    switch_model.eval()
    handles = []

    def _get_activation_hook(name):
        def hook(module, input, output):
            forwarded_hidden_states[name].append(input[0].detach().reshape(-1, input[0].shape[-1]))

        return hook

    for layer_idx in tqdm(
            grouper.sparse_layer_indices,
            desc=f"[Merging]Registering forward hook..."
    ):
        encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
        decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
        forwarded_hidden_states[encoder_mlp_name] = []
        forwarded_hidden_states[decoder_mlp_name] = []
        handles.append(switch_model.encoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
            _get_activation_hook(encoder_mlp_name))
        )
        handles.append(switch_model.decoder.block[layer_idx].layer[-1].mlp.register_forward_hook(
            _get_activation_hook(decoder_mlp_name))
        )

    # {name: values}, values will be of shape (len(dataloader), batch_size * seq_len)
    router_indices = {name: [] for name in forwarded_hidden_states.keys()}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Merging]Computing activations..."):
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = switch_model(**batch)
            for layer_idx in grouper.sparse_layer_indices:
                router_indices[f"encoder.block.{layer_idx}.layer.1.mlp"].append(
                    outputs.encoder_router_logits[layer_idx][1].reshape(-1)
                )
                router_indices[f"decoder.block.{layer_idx}.layer.2.mlp"].append(
                    outputs.decoder_router_logits[layer_idx][1].reshape(-1)
                )

    for handle in handles:
        handle.remove()

    num_experts = grouper.num_experts

    for layer_idx in tqdm(
            grouper.sparse_layer_indices,
            desc=f"[Merging]Merging all non-core experts to one within and across experts..."
    ):
        if encoder_merging_layers is None or layer_idx in encoder_merging_layers:
            encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            layer_forwarded_hidden_states = tuple()
            for expert_idx in range(num_experts):
                layer_forwarded_hidden_states += (
                    torch.cat(
                        [forwarded_hidden_states[encoder_mlp_name][i][router_indices[encoder_mlp_name][i] == expert_idx]
                         for i in range(len(dataloader))], dim=0),
                )
            switch_model.encoder.block[layer_idx].layer[
                -1].mlp = _merge_mlp_all_non_core_experts_to_one_within_and_across_models(
                mlp=switch_model.encoder.block[layer_idx].layer[-1].mlp,
                forwarded_hidden_states=layer_forwarded_hidden_states,
                core_experts=core_experts[encoder_mlp_name],
            )
        if decoder_merging_layers is None or layer_idx in decoder_merging_layers:
            decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            layer_forwarded_hidden_states = tuple()
            for expert_idx in range(num_experts):
                layer_forwarded_hidden_states += (
                    torch.cat(
                        [forwarded_hidden_states[decoder_mlp_name][i][router_indices[decoder_mlp_name][i] == expert_idx]
                         for i in range(len(dataloader))], dim=0),
                )
            switch_model.decoder.block[layer_idx].layer[
                -1].mlp = _merge_mlp_all_non_core_experts_to_one_within_and_across_models(
                mlp=switch_model.decoder.block[layer_idx].layer[-1].mlp,
                forwarded_hidden_states=layer_forwarded_hidden_states,
                core_experts=core_experts[decoder_mlp_name],
            )

    return switch_model


class ExpertUsageFrequencyTracker(object):
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
            if "expert_" in name and "wi" in name:
                mlp_name = name.split(".experts.expert_")[0]
                self.exp_expert_usage_frequency_dict[mlp_name].append(1.0)

        for key, value in self.exp_expert_usage_frequency_dict.items():
            value = torch.tensor(value, device=self.device)
            # init as average used frequency
            self.exp_expert_usage_frequency_dict[key] = value / torch.sum(value)
        self.num_layers = len(self.exp_expert_usage_frequency_dict) // 2

    def _update_exp_usage_frequency_state(
            self,
            model_outputs: Union[List[Seq2SeqMoEOutput], Seq2SeqMoEOutput]
    ):
        usage_frequency_state_dict = OrderedDict()
        is_first_step = self.last_error < 0
        for name in self.exp_expert_usage_frequency_dict.keys():
            usage_frequency_state_dict[name] = torch.zeros_like(self.exp_expert_usage_frequency_dict[name])
            # `name` is like "encoder.block.1.layer.1.mlp" or "decoder.block.3.layer.2.mlp" or etc.,
            # Capture the layer index from the name
            layer_idx = int(name.split(".block.")[1].split(".layer.")[0])
            is_encoder = "encoder" in name
            if is_encoder:
                if isinstance(model_outputs, list):
                    router_expert_index = [op.encoder_router_logits[layer_idx][1].reshape(-1) for op in model_outputs]
                    router_expert_index = torch.concat(router_expert_index)
                else:
                    router_expert_index = model_outputs.encoder_router_logits[layer_idx][1].reshape(-1)
            else:
                if isinstance(model_outputs, list):
                    router_expert_index = [op.decoder_router_logits[layer_idx][1].reshape(-1) for op in model_outputs]
                    router_expert_index = torch.concat(router_expert_index)
                else:
                    router_expert_index = model_outputs.decoder_router_logits[layer_idx][1].reshape(-1)
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

    def step(self, model_outputs: Union[List[Seq2SeqMoEOutput], Seq2SeqMoEOutput], global_step: int) -> float:
        """
        Update the usage frequency state of experts and return the last error of the state which is L1 norm

        Parameters
        ----------
        model_outputs: Union[List[Seq2SeqMoEOutput], Seq2SeqMoEOutput]
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
