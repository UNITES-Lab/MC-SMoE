import re
from collections import OrderedDict
from typing import Optional, Tuple, Dict, List

import torch
from torch.nn import functional as F
from transformers import (
    SwitchTransformersForConditionalGeneration,
    SwitchTransformersConfig,
    SwitchTransformersSparseMLP,
    SwitchTransformersTop1Router
)


class TaskSpecificExpertPruner(object):
    """
    Reproduce "Task-Speciﬁc Expert Pruning for Sparse Mixture-of-Experts" paper
    """

    def __init__(
            self,
            model_config: SwitchTransformersConfig,
            threshold_beta: Optional[float] = 0.1,
            minimum_num_experts: Optional[int] = 8,
            encoder_pruning_layers: Optional[List[int]] = None,
            decoder_pruning_layers: Optional[List[int]] = None,
    ):
        self.num_experts = model_config.num_experts
        self.sparse_layer_indices = list(range(1, model_config.num_layers, model_config.encoder_sparse_step))

        self.experts_alive = None
        self.threshold_beta = threshold_beta
        self.minimum_num_experts = minimum_num_experts

        self._reset_all(encoder_pruning_layers, decoder_pruning_layers)

    def _reset_all(
            self,
            encoder_pruning_layers: Optional[List[int]],
            decoder_pruning_layers: Optional[List[int]],
    ):
        self.experts_alive = OrderedDict()
        for layer_idx in self.sparse_layer_indices:
            if encoder_pruning_layers is None or layer_idx in encoder_pruning_layers:
                encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
                self.experts_alive[encoder_mlp_name] = [True] * self.num_experts
            if decoder_pruning_layers is None or layer_idx in decoder_pruning_layers:
                decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"
                self.experts_alive[decoder_mlp_name] = [True] * self.num_experts

    def execute_pruning(
            self,
            model: SwitchTransformersForConditionalGeneration,
            usage_frequency_state_dict: Dict[str, torch.Tensor],
            prune_to_minimum: Optional[bool] = False,
    ) -> SwitchTransformersForConditionalGeneration:
        """
        Execute pruning based on current EMA usage frequency state dict

        Parameters
        ----------
        model: SwitchTransformersForConditionalGeneration
            Model to be pruned.
        usage_frequency_state_dict: Dict[str, torch.Tensor]
            EMA usage frequency state dict.
        prune_to_minimum: Optional[bool]
            Whether to prune to minimum number of experts.

        Returns
        -------
        SwitchTransformersForConditionalGeneration
            Pruned model.

        """
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
            encoder_or_decoder = name.split(".")[0]
            if encoder_or_decoder == "encoder":
                model.encoder.block[layer_idx].layer[-1].mlp = self._prune_mlp_experts(
                    self.experts_alive[name],
                    model.encoder.block[layer_idx].layer[-1].mlp
                )
            elif encoder_or_decoder == "decoder":
                model.decoder.block[layer_idx].layer[-1].mlp = self._prune_mlp_experts(
                    self.experts_alive[name],
                    model.decoder.block[layer_idx].layer[-1].mlp
                )
            else:
                raise ValueError(f"Unknown encoder_or_decoder {encoder_or_decoder}")

        return model

    @staticmethod
    def _prune_mlp_experts(
            experts_alive: List[bool],
            mlp: SwitchTransformersSparseMLP
    ) -> SwitchTransformersSparseMLP:
        """
        mlp.router should have member `experts_mask` of tensor
        """
        dead_experts_mask = ~torch.tensor(experts_alive, dtype=torch.bool)
        with torch.no_grad():
            mlp.router.experts_mask.data[dead_experts_mask] = -1e6

        return mlp


def transform_switch_model_for_task_specific_pruning(
        model: SwitchTransformersForConditionalGeneration
) -> SwitchTransformersForConditionalGeneration:
    """
    Add expert mask to MLP.
    """
    config = model.config
    sparse_layer_indices = list(range(1, config.num_layers, config.encoder_sparse_step))

    def _custom_forward(self, hidden_states: torch.Tensor) -> Tuple:
        _, router_logits = self._compute_router_probabilities(hidden_states)

        router_logits = router_logits * self.experts_mask
        router_probs = F.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)

        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = F.one_hot(expert_index, num_classes=self.num_experts)
        # Since experts are pruned, no need to do max-capacity dropping
        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits

    for layer_idx in sparse_layer_indices:
        model.encoder.block[layer_idx].layer[-1].mlp.router.experts_mask = torch.nn.Parameter(torch.tensor(
            [1.0] * model.encoder.block[layer_idx].layer[-1].mlp.router.num_experts), requires_grad=False)
        model.encoder.block[layer_idx].layer[-1].mlp.router.forward = _custom_forward.__get__(
            model.encoder.block[layer_idx].layer[-1].mlp.router, SwitchTransformersTop1Router
        )
        model.decoder.block[layer_idx].layer[-1].mlp.router.experts_mask = torch.nn.Parameter(torch.tensor(
            [1.0] * model.decoder.block[layer_idx].layer[-1].mlp.router.num_experts), requires_grad=False)
        model.decoder.block[layer_idx].layer[-1].mlp.router.forward = _custom_forward.__get__(
            model.decoder.block[layer_idx].layer[-1].mlp.router, SwitchTransformersTop1Router
        )

    return model
