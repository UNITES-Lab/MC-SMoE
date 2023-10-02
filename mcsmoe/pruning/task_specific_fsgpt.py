import re
from collections import OrderedDict
from typing import Optional, Tuple, Dict, List

import torch

from mcsmoe.models.fsgpt_moe import (
    FSGPTMoEForCausalLM,
    FSGPTMoEConfig,
    FSGPTMoESparseMLP,
    FSGPTMoETop2Router
)


class TaskSpecificExpertPrunerForFSGPT(object):
    """
    Reproduce "Task-Speciﬁc Expert Pruning for Sparse Mixture-of-Experts" paper
    """

    def __init__(
            self,
            model_config: FSGPTMoEConfig,
            threshold_beta: Optional[float] = 0.1,
            minimum_num_experts: Optional[int] = 8,
            pruning_layers: Optional[List[int]] = None,
    ):
        self.num_experts = model_config.num_experts
        self.sparse_layer_indices = list(range(1, model_config.num_layers, model_config.sparse_step))

        self.experts_alive = None
        self.threshold_beta = threshold_beta
        self.minimum_num_experts = minimum_num_experts

        self._reset_all(pruning_layers)

    def _reset_all(
            self,
            pruning_layers: Optional[List[int]],
    ):
        self.experts_alive = OrderedDict()
        for layer_idx in self.sparse_layer_indices:
            if pruning_layers is None or layer_idx in pruning_layers:
                ffn_name = f"decoder.layers.{layer_idx}.ffn"
                self.experts_alive[ffn_name] = [True] * self.num_experts

    def execute_pruning(
            self,
            model: FSGPTMoEForCausalLM,
            usage_frequency_state_dict: Dict[str, torch.Tensor],
            prune_to_minimum: Optional[bool] = False,
    ) -> FSGPTMoEForCausalLM:
        print("[Task-Specific Pruning]Start!")
        if prune_to_minimum:
            print(f"[Task-Specific Pruning]Prune to minimum number of experts {self.minimum_num_experts}")
        # 1. Update experts alive
        for name in self.experts_alive:
            num_alive = sum(self.experts_alive[name])
            usage_frequency = usage_frequency_state_dict[name]
            # normalize usage frequency in alive experts and set dead experts' usage frequency to 0
            usage_frequency = usage_frequency / usage_frequency[self.experts_alive[name]].sum()
            usage_frequency[~torch.tensor(self.experts_alive[name], dtype=torch.bool)] = 0.0
            if num_alive < self.minimum_num_experts:
                raise ValueError(
                    f"Number of alive experts {num_alive} is less than minimum number of experts {self.minimum_num_experts}")
            if num_alive == self.minimum_num_experts:
                continue
            threshold = self.threshold_beta / num_alive
            if (usage_frequency >= threshold).sum() < self.minimum_num_experts or prune_to_minimum:
                # select top-m experts
                _, top_m_indices = torch.topk(usage_frequency, self.minimum_num_experts)
                self.experts_alive[name] = [True if i in top_m_indices else False for i in range(self.num_experts)]
            else:
                # select experts with usage frequency > threshold
                self.experts_alive[name] = [True if f > threshold else False for f in usage_frequency]
            print(f"Number of alive experts in {name}: {sum(self.experts_alive[name])} / {self.num_experts}")

        # 2. Prune experts
        for name in self.experts_alive:
            layer_idx = int(re.findall(r"\d+", name)[0])
            model.decoder.layers[layer_idx].ffn = self._prune_ffn_experts(
                self.experts_alive[name],
                model.decoder.layers[layer_idx].ffn
            )

        return model

    @staticmethod
    def _prune_ffn_experts(
            experts_alive: List[bool],
            ffn: FSGPTMoESparseMLP
    ) -> FSGPTMoESparseMLP:
        """
        mlp.router should have member `experts_mask` of tensor
        """
        dead_experts_mask = ~torch.tensor(experts_alive, dtype=torch.bool)
        with torch.no_grad(), torch.autocast("cuda"):
            ffn.router.experts_mask.data[dead_experts_mask] = -1e9

        return ffn


def transform_fsgpt_model_for_task_specific_pruning(
        model: FSGPTMoEForCausalLM
) -> FSGPTMoEForCausalLM:
    """
    Add expert mask to MLP.
    """
    config = model.config
    sparse_layer_indices = list(range(1, config.num_layers, config.sparse_step))

    def _custom_forward(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.LongTensor] = None) -> Tuple:
        self.input_dtype = hidden_states.dtype
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape((batch_size * sequence_length), hidden_dim)
        hidden_states = hidden_states.to(self.dtype)
        self._cast_classifier()
        router_logits = self.classifier(hidden_states)

        router_logits = router_logits * self.experts_mask

        top_1_mask, router_probs = self.route_tokens(router_logits, self.input_dtype, padding_mask)
        return top_1_mask, router_probs, router_logits

    for layer_idx in sparse_layer_indices:
        device = model.decoder.layers[layer_idx].ffn.router.classifier.weight.device
        model.decoder.layers[layer_idx].ffn.router.experts_mask = torch.nn.Parameter(torch.tensor(
            [1.0] * model.decoder.layers[layer_idx].ffn.router.num_experts, device=device), requires_grad=False)
        model.decoder.layers[layer_idx].ffn.router.forward = _custom_forward.__get__(
            model.decoder.layers[layer_idx].ffn.router, FSGPTMoETop2Router
        )

    return model
