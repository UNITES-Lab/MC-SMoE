from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.modeling_outputs import ModelOutput

__all__ = [
    "MoEModelOutputWithPastAndCrossAttentions",
    "MoECausalLMOutputWithPastAndCrossAttentions",
    "MoECasualLMDistillationOutputWithPastAndCrossAttention",
    "MoEDistillationOutput",
]


@dataclass
class MoEModelOutputWithPastAndCrossAttentions(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    router_probs: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MoECausalLMOutputWithPastAndCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    z_loss: torch.FloatTensor = None
    aux_loss: torch.FloatTensor = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None


class MoECasualLMDistillationOutputWithPastAndCrossAttention(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    z_loss: torch.FloatTensor = None
    aux_loss: torch.FloatTensor = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    kd_loss: torch.FloatTensor = None
    lm_loss: torch.FloatTensor = None


class MoEDistillationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    kd_loss: torch.FloatTensor = None
    task_loss: torch.FloatTensor = None
    hd_loss: torch.FloatTensor = None
    norm_loss: torch.FloatTensor = None
    sing_reg_loss: torch.FloatTensor = None
    orth_reg_loss: torch.FloatTensor = None
    student_logits: torch.FloatTensor = None
    teacher_logits: torch.FloatTensor = None
