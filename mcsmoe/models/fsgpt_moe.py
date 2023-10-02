"""
Reproduce the Fairseq GPT model.

Description from Fairseq:
We differ from Brown et al. (2020) in two ways:
(1) we use only dense attention, while they alternate between dense and locally banded sparse attention;
(2) we train our models with sinusoidal positional embeddings, following Shortformer (Press et al., 2020).


"""
import math
from typing import Any, Optional
from typing import Tuple, Union, Dict

import torch
import torch.onnx.operators
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    MoEModelOutputWithPastAndCrossAttentions,
    MoECausalLMOutputWithPast
)

from .fsgpt import FSGPTConfig

__all__ = [
    "FSGPTMoEConfig",
    "FSGPTMoETop2Router",
    "FSGPTMoEDenseActDense",
    "FSGPTMoESparseMLP",
    "FSGPTMoEBlock",
    "FSGPTMoEModel",
    "FSGPTMoEForCausalLM",
]


def rename_fairseq_state_dict(
        state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    convert_state_dict = dict()
    for name, param in state_dict.items():
        if name == 'decoder.version':
            continue
        name = name.replace('decoder.output_projection', 'lm_head')
        name = name.replace('fc1', 'ffn.fc1')
        name = name.replace('fc2', 'ffn.fc2')
        convert_state_dict[name] = param

    return convert_state_dict


class FSGPTMoEConfig(FSGPTConfig):
    def __init__(
            self,
            sparse_step=2,
            router_bias=False,
            num_experts=512,
            expert_capacity=None,
            router_ignore_padding_tokens=True,
            router_dtype="float32",
            second_expert_policy="all",
            normalize_router_prob_before_dropping=False,
            batch_prioritized_routing=False,
            moe_eval_capacity_token_fraction=-1.0,
            moe_token_dropout=0.2,
            output_router_logits=True,
            **kwargs,
    ):
        self.sparse_step = sparse_step
        self.router_bias = router_bias
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.router_ignore_padding_tokens = router_ignore_padding_tokens
        self.router_dtype = router_dtype
        self.second_expert_policy = second_expert_policy
        self.normalize_router_prob_before_dropping = normalize_router_prob_before_dropping
        self.batch_prioritized_routing = batch_prioritized_routing
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction
        self.moe_token_dropout = moe_token_dropout
        self.output_router_logits = output_router_logits

        super().__init__(**kwargs)


def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def make_positions(tensor, padding_idx: int, offset: int):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + offset - 1


class FSGPTMoESinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, num_positions=1024):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.weights = FSGPTMoESinusoidalPositionalEmbedding.get_embedding(
            num_positions + self.offset, embedding_dim, self.offset
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_positions = int(1e5)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(
            num_embeddings: int, embedding_dim: int, offset: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if offset is not None:
            emb[offset - 1, :] = 0
        return emb

    def forward(
            self,
            input,
            incremental_state: Optional[Any] = None,
            timestep: Optional[Tensor] = None,
            positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.offset + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = FSGPTMoESinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.offset
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    self.weights.index_select(index=self.offset - 1 + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return self.weights[self.offset - 1 + pos, :].expand(bsz, 1, -1)

        positions = make_positions(
            input, self.padding_idx, self.offset
        )
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )


class FSGPTMoESelfAttention(nn.Module):
    def __init__(self, config: FSGPTMoEConfig):
        super().__init__()

        max_positions = config.max_position_embeddings

        bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
            1, 1, max_positions, max_positions
        )

        self.register_buffer("bias", bias, persistent=False)
        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)

        self.num_heads = config.num_heads
        self.embed_dim = config.hidden_size
        self.head_dim = self.embed_dim // self.num_heads

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(float(config.resid_dropout))

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.query_scaled = math.sqrt(self.head_dim)

    @staticmethod
    def _split_heads(tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    @staticmethod
    def _merge_heads(tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            layer_past=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        query = self.q_proj(hidden_states) / self.query_scaled
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class FSGPTMoETop2Router(nn.Module):
    """
    Adapted form hugginfgace NLLB-MoE

    Router using tokens choose top-2 experts assignment.

    This router uses the same mechanism as in NLLB-MoE from the fairseq repository. Items are sorted by router_probs
    and then routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee
    that each token is processed by an expert**, or that each expert receives at least one token.

    The router combining weights are also returned to make sure that the states that are not updated will be masked.

    """

    def __init__(self, config: FSGPTMoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        self.router_ignore_padding_tokens = config.router_ignore_padding_tokens
        self.dtype = getattr(torch, config.router_dtype)

        self.second_expert_policy = config.second_expert_policy
        self.normalize_router_prob_before_dropping = config.normalize_router_prob_before_dropping
        self.batch_prioritized_routing = config.batch_prioritized_routing
        self.moe_eval_capacity_token_fraction = config.moe_eval_capacity_token_fraction

    def _cast_classifier(self):
        r"""
        `bitsandbytes` `Linear8bitLt` layers does not support manual casting Therefore we need to check if they are an
        instance of the `Linear8bitLt` class by checking special attributes.
        """
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            self.classifier = self.classifier.to(self.dtype)

    def normalize_router_probabilities(self, router_probs, top_1_mask, top_2_mask):
        top_1_max_probs = (router_probs * top_1_mask).sum(dim=1)
        top_2_max_probs = (router_probs * top_2_mask).sum(dim=1)
        denom_s = torch.clamp(top_1_max_probs + top_2_max_probs, min=torch.finfo(router_probs.dtype).eps)
        top_1_max_probs = top_1_max_probs / denom_s
        top_2_max_probs = top_2_max_probs / denom_s
        return top_1_max_probs, top_2_max_probs

    def route_tokens(
            self,
            router_logits: torch.Tensor,
            input_dtype: torch.dtype = torch.float32,
            padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple:
        """
        Computes the `dispatch_mask` and the `dispatch_weights` for each experts. The masks are adapted to the expert
        capacity.
        """
        nb_tokens = router_logits.shape[0]
        # Apply Softmax and cast back to the original `dtype`
        router_probs = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(input_dtype)
        top_1_expert_index = torch.argmax(router_probs, dim=-1)
        top_1_mask = torch.nn.functional.one_hot(top_1_expert_index, num_classes=self.num_experts)

        if self.second_expert_policy == "sampling":
            gumbel = torch.distributions.gumbel.Gumbel(0, 1).rsample
            router_logits += gumbel(router_logits.shape).to(router_logits.device)

        # replace top_1_expert_index with min values
        logits_except_top_1 = router_logits.masked_fill(top_1_mask.bool(), float("-inf"))
        top_2_expert_index = torch.argmax(logits_except_top_1, dim=-1)
        top_2_mask = torch.nn.functional.one_hot(top_2_expert_index, num_classes=self.num_experts)

        if self.normalize_router_prob_before_dropping:
            top_1_max_probs, top_2_max_probs = self.normalize_router_probabilities(
                router_probs, top_1_mask, top_2_mask
            )

        if self.second_expert_policy == "random":
            top_2_max_probs = (router_probs * top_2_mask).sum(dim=1)
            sampled = (2 * top_2_max_probs) > torch.rand_like(top_2_max_probs.float())
            top_2_mask = top_2_mask * sampled.repeat(self.num_experts, 1).transpose(1, 0)

        if padding_mask is not None and not self.router_ignore_padding_tokens:
            if len(padding_mask.shape) == 4:
                # only get the last causal mask
                padding_mask = padding_mask[:, :, -1, :].reshape(-1)[-nb_tokens:]
            non_padding = ~padding_mask.bool()
            top_1_mask = top_1_mask * non_padding.unsqueeze(-1).to(top_1_mask.dtype)
            top_2_mask = top_2_mask * non_padding.unsqueeze(-1).to(top_1_mask.dtype)

        if self.batch_prioritized_routing:
            # sort tokens based on their routing probability
            # to make sure important tokens are routed, first
            importance_scores = -1 * router_probs.max(dim=1)[0]
            sorted_top_1_mask = top_1_mask[importance_scores.argsort(dim=0)]
            sorted_cumsum1 = (torch.cumsum(sorted_top_1_mask, dim=0) - 1) * sorted_top_1_mask
            locations1 = sorted_cumsum1[importance_scores.argsort(dim=0).argsort(dim=0)]

            sorted_top_2_mask = top_2_mask[importance_scores.argsort(dim=0)]
            sorted_cumsum2 = (torch.cumsum(sorted_top_2_mask, dim=0) - 1) * sorted_top_2_mask
            locations2 = sorted_cumsum2[importance_scores.argsort(dim=0).argsort(dim=0)]
            # Update 2nd's location by accounting for locations of 1st
            locations2 += torch.sum(top_1_mask, dim=0, keepdim=True)

        else:
            locations1 = torch.cumsum(top_1_mask, dim=0) - 1
            locations2 = torch.cumsum(top_2_mask, dim=0) - 1
            # Update 2nd's location by accounting for locations of 1st
            locations2 += torch.sum(top_1_mask, dim=0, keepdim=True)

        if not self.training and self.moe_eval_capacity_token_fraction > 0:
            self.expert_capacity = math.ceil(self.moe_eval_capacity_token_fraction * nb_tokens)
        else:
            capacity = 2 * math.ceil(nb_tokens / self.num_experts)
            self.expert_capacity = capacity if self.expert_capacity is None else self.expert_capacity

        # Remove locations outside capacity from ( cumsum < capacity = False will not be routed)
        top_1_mask = top_1_mask * torch.lt(locations1, self.expert_capacity)
        top_2_mask = top_2_mask * torch.lt(locations2, self.expert_capacity)

        if not self.normalize_router_prob_before_dropping:
            top_1_max_probs, top_2_max_probs = self.normalize_router_probabilities(
                router_probs, top_1_mask, top_2_mask
            )

        # Calculate combine_weights and dispatch_mask
        gates1 = top_1_max_probs[:, None] * top_1_mask
        gates2 = top_2_max_probs[:, None] * top_2_mask
        router_probs = gates1 + gates2

        return top_1_mask, router_probs

    def forward(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.LongTensor] = None) -> Tuple:
        r"""
        The hidden states are reshaped to simplify the computation of the router probabilities (combining weights for
        each experts.)

        Args:
            hidden_states (`torch.Tensor`):
                (batch_size, sequence_length, hidden_dim) from which router probabilities are computed.
        Returns:
            top_1_mask (`torch.Tensor` of shape (batch_size, sequence_length)):
                Index tensor of shape [batch_size, sequence_length] corresponding to the expert selected for each token
                using the top1 probabilities of the router.
            router_probabilities (`torch.Tensor` of shape (batch_size, sequence_length, nump_experts)):
                Tensor of shape (batch_size, sequence_length, num_experts) corresponding to the probabilities for each
                token and expert. Used for routing tokens to experts.
            router_logits (`torch.Tensor` of shape (batch_size, sequence_length))):
                Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding to raw router logits.
                This is used later for computing router z-loss.
        """
        self.input_dtype = hidden_states.dtype
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape((batch_size * sequence_length), hidden_dim)
        hidden_states = hidden_states.to(self.dtype)
        self._cast_classifier()
        router_logits = self.classifier(hidden_states)
        top_1_mask, router_probs = self.route_tokens(router_logits, self.input_dtype, padding_mask)
        return top_1_mask, router_probs, router_logits


class FSGPTMoEDenseActDense(nn.Module):
    def __init__(self, config: FSGPTMoEConfig, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, config.hidden_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
                isinstance(self.fc2.weight, torch.Tensor)
                and hidden_states.dtype != self.fc2.weight.dtype
                and self.fc2.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.fc2.weight.dtype)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class FSGPTMoESparseMLP(nn.Module):
    r"""
    Adapted from huggingface NLLB-MoE
    """

    def __init__(self, config: FSGPTMoEConfig, ffn_dim: int, expert_class: nn.Module = FSGPTMoEDenseActDense):
        super().__init__()
        self.router = FSGPTMoETop2Router(config)
        self.moe_token_dropout = config.moe_token_dropout
        self.token_dropout = nn.Dropout(self.moe_token_dropout)
        self.num_experts = config.num_experts

        self.experts = nn.ModuleDict()
        for idx in range(self.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config, ffn_dim)

    def forward(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.Tensor] = False):
        r"""
        The goal of this forward pass is to have the same number of operation as the equivalent `NllbMoeDenseActDense`
        (mlp) layer. This means that all of the hidden states should be processed at most twice ( since we are using a
        top_2 gating mecanism). This means that we keep the complexity to O(batch_size x sequence_length x hidden_dim)
        instead of O(num_experts x batch_size x sequence_length x hidden_dim).

        1- Get the `router_probs` from the `router`. The shape of the `router_mask` is `(batch_size X sequence_length,
        num_expert)` and corresponds to the boolean version of the `router_probs`. The inputs are masked using the
        `router_mask`.

        2- Dispatch the hidden_states to its associated experts. The router probabilities are used to weight the
        contribution of each experts when updating the masked hidden states.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_dim)`):
                The hidden states
            padding_mask (`torch.Tensor`, *optional*, defaults to `False`):
                Attention mask. Can be in the causal form or not.

        Returns:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_dim)`):
                Updated hidden states
            router_logits (`torch.Tensor` of shape `(batch_size, sequence_length, num_experts)`):
                Needed for computing the loss

        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        top_1_mask, router_probs, router_logits = self.router(hidden_states, padding_mask)
        router_mask = router_probs.bool()
        hidden_states = hidden_states.reshape((batch_size * sequence_length), hidden_dim)
        masked_hidden_states = torch.einsum("bm,be->ebm", hidden_states, router_mask)
        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, idx]
            combining_weights = router_probs[token_indices, idx]
            if isinstance(expert, torch.nn.Linear):
                # After pruning,there will be fake experts with no parameters which are nn.Linear
                # need to move them to the same device as the hidden states
                expert = expert.to(masked_hidden_states.device)
            expert_output = expert(masked_hidden_states[idx, token_indices])
            if self.moe_token_dropout > 0:
                if self.training:
                    expert_output = self.token_dropout(expert_output)
                else:
                    expert_output *= 1 - self.moe_token_dropout
            masked_hidden_states[idx, token_indices] = torch.einsum("b,be->be", combining_weights, expert_output)
        hidden_states = masked_hidden_states.sum(dim=0).reshape(batch_size, sequence_length, hidden_dim)

        top_1_expert_index = torch.argmax(top_1_mask, dim=-1)
        return hidden_states, (router_logits, router_probs, top_1_expert_index)


class FSGPTMoEBlock(nn.Module):
    def __init__(self, config: FSGPTMoEConfig, is_sparse: bool = False):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * hidden_size
        self.is_sparse = is_sparse
        self.self_attn = FSGPTMoESelfAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)
        if self.is_sparse:
            self.ffn = FSGPTMoESparseMLP(config, inner_dim)
        else:
            self.ffn = FSGPTMoEDenseActDense(config, inner_dim)
        self.final_layer_norm = nn.LayerNorm(hidden_size)

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
            output_router_logits=False,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.is_sparse:
            hidden_states, router_states = self.ffn(hidden_states, attention_mask)
        else:
            hidden_states, router_states = self.ffn(hidden_states), None

        # residual connection
        hidden_states = residual + hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        if output_router_logits:
            outputs = outputs + (router_states,)

        return outputs  # hidden_states, present, (attentions, cross_attentions), router_logits


class FSGPTMoEPreTrainedModel(PreTrainedModel):
    config_class = FSGPTMoEConfig
    base_model_prefix = "decoder"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FSGPTMoEBlock"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear,)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.hidden_size ** -0.5)
            if module.weight.data.shape[0] == self.config.vocab_size:
                module.weight.data[self.config.pad_token_id].zero_()
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.hidden_size ** -0.5)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GPTNeoModel):
            module.gradient_checkpointing = value


class FSGPTMoEModel(FSGPTMoEPreTrainedModel):
    def __init__(self, config: FSGPTMoEConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, self.embed_dim, padding_idx=config.pad_token_id)
        self.embed_positions = FSGPTMoESinusoidalPositionalEmbedding(
            embedding_dim=self.embed_dim,
            padding_idx=config.pad_token_id,
            num_positions=config.max_position_embeddings,
        )
        self.drop = nn.Dropout(float(config.embed_dropout))

        self.embed_scale = (
            1.0 if config.no_scale_embedding else math.sqrt(self.embed_dim)
        )
        self.layers = nn.ModuleList()
        sparse_step = config.sparse_step
        for i in range(config.num_layers):
            is_sparse = (i + 1) % sparse_step == 0 if sparse_step > 0 else False
            self.layers.append(FSGPTMoEBlock(config, is_sparse=is_sparse))
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MoEModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.layers))
        else:
            past_length = past_key_values[0][0].size(-2)

        # if position_ids is None:
        #     position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        #     position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        position_embeds = self.embed_positions(input_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.embed_tokens(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_router_probs = () if output_router_logits else None
        for i, (block, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

            if output_router_logits:
                all_router_probs += (outputs[-1],)

        hidden_states = self.layer_norm(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in
                [hidden_states,
                 presents,
                 all_hidden_states,
                 all_self_attentions,
                 all_router_probs]
                if v is not None
            )

        return MoEModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            router_probs=all_router_probs
        )


class FSGPTMoEForCausalLM(FSGPTMoEPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: FSGPTMoEConfig):
        super().__init__(config)
        self.decoder = FSGPTMoEModel(config)
        # Share input output embeddings
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.decoder.embed_tokens.weight

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        return model_inputs

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MoECausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        transformer_outputs = self.decoder(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Compute loss in fp32 to match with mesh-tf version
            # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MoECausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            router_logits=transformer_outputs.router_probs,
        )

    @staticmethod
    def _reorder_cache(
            past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
