""" Adapted from modeling_switch_transformers.py in HuggingFace transformers repository """

import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Any
from typing import Optional, Union
from typing import Tuple
from warnings import warn

import numpy as np
import torch
import torch.nn.functional as F
from torch import linalg as LA
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    MoEModelOutput,
    MoEModelOutputWithPastAndCrossAttentions,
    Seq2SeqMoEModelOutput,
    Seq2SeqMoEOutput
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.switch_transformers import SwitchTransformersConfig
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    is_torch_fx_proxy,
    logging,
)

from .modeling_outputs import MoEDistillationOutput

logger = logging.get_logger(__name__)


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

    [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence

    >>> DataCollatorForT5MLM.compute_input_and_target_lengths(512, 0.15, 3.0)
    (568, 114)
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length

class SwitchTransformersWrapperForDistillation(nn.Module):
    EXCEPTED_CONFIG_KEYS = [
        "freeze_router",
        "sparse_encoder_layer_list",
        "sparse_decoder_layer_list",
        "expert_norm_coef",
        "router_z_loss_coef",
        "router_aux_loss_coef",
        "only_task_vectors_for_pruning",
        "keep_singular_values",
    ]

    def __init__(
            self,
            teacher=None,
            student=None,
            mlm_lambda: float = 1.0,
            kd_lambda: float = 1.0,
            kd_temperature: float = 1.0,
            model_max_length: int = 512,
            reg_lambda: float = 0.0,
            hd_lambda: float = 0.0,
            orth_reg_lambda: float = 0.0,
            sing_reg_lambda: float = 0.0,
            hd_cos_sim: bool = False,
            next_rank: int = None,
    ):
        super().__init__()
        self.kd_lambda = kd_lambda
        self.kd_temperature = kd_temperature
        self.model_max_length = model_max_length
        self.mlm_lambda = mlm_lambda
        self.reg_lambda = reg_lambda
        self.hd_cos_sim = hd_cos_sim
        self.hd_lambda = hd_lambda
        self.orth_reg_lambda = orth_reg_lambda
        self.sing_reg_lambda = sing_reg_lambda

        if next_rank is None and hasattr(student.config, "low_rank_factor"):
            self.next_rank = student.config.low_rank_factor
        elif next_rank is None:
            self.next_rank = student.config.d_model
        else:
            self.next_rank = next_rank

        self.teacher = teacher
        self.teacher.router_z_loss_coef = 0.0
        self.teacher.router_aux_loss_coef = 0.0

        self.student = student
        self.student.router_z_loss_coef = 0.0
        self.student.router_aux_loss_coef = 0.0

        for key, student_value in self.student.config.__dict__.items():
            if key not in self.EXCEPTED_CONFIG_KEYS and self.teacher.config.__dict__[key] != student_value:
                warn(f"Teacher and student have different config: {key}, "
                     f"at teacher: {self.teacher.config.__dict__[key]} and at student: {student_value}")

    def forward(
            self,
            *args,
            **kwargs
    ):
        # Shut down dropout and layer norm for accurate distillation
        self.teacher.eval()
        self.student.eval()

        kwargs["output_hidden_states"] = True

        with torch.no_grad():
            teacher_outputs = self.teacher(*args, **kwargs)
        teacher_encoder_hidden_states = teacher_outputs.encoder_hidden_states[1:]  # skip embedding hidden states
        teacher_decoder_hidden_states = teacher_outputs.decoder_hidden_states[1:]
        teacher_encoder_hidden_states = torch.stack(teacher_encoder_hidden_states, dim=0)
        teacher_decoder_hidden_states = torch.stack(teacher_decoder_hidden_states, dim=0)
        teacher_logits = teacher_outputs.logits

        student_outputs = self.student(*args, **kwargs)
        student_encoder_hidden_states = student_outputs.encoder_hidden_states[1:]
        student_decoder_hidden_states = student_outputs.decoder_hidden_states[1:]
        student_encoder_hidden_states = torch.stack(student_encoder_hidden_states, dim=0)
        student_decoder_hidden_states = torch.stack(student_decoder_hidden_states, dim=0)
        student_logits = student_outputs.logits

        # 1. kl divergence loss
        kd_criteria = nn.KLDivLoss(reduction="batchmean")
        kd_loss = kd_criteria(
            F.log_softmax(student_logits / self.kd_temperature, dim=-1),
            F.softmax(teacher_logits / self.kd_temperature, dim=-1)
        )
        kd_loss = kd_loss * self.kd_lambda

        # 2. language modeling loss
        task_loss = student_outputs.loss * self.mlm_lambda

        # 3. expert U & V norm loss
        device = task_loss.device
        if self.reg_lambda == 0:
            norm_loss = torch.tensor(0.0, device=device)
        elif self.next_rank != self.student.config.low_rank_factor:
            norm_loss = self.compute_u_v_reg_for_next_rank(device=device)
            norm_loss = norm_loss * self.reg_lambda
        else:
            norm_loss = torch.tensor(0.0, device=device)

        # 4. hidden states loss
        if self.hd_cos_sim:
            hd_criteria = nn.CosineSimilarity(dim=-1)
        else:
            hd_criteria = nn.MSELoss(reduction="none")

        encoder_hidden_states_loss = hd_criteria(student_encoder_hidden_states, teacher_encoder_hidden_states).mean()
        decoder_hidden_states_loss = hd_criteria(student_decoder_hidden_states, teacher_decoder_hidden_states).mean()
        hidden_states_loss = (encoder_hidden_states_loss + decoder_hidden_states_loss) / 2

        if self.hd_lambda == 0:
            hd_loss = torch.tensor(0.0, device=device)
        elif self.hd_cos_sim:
            # Cosine similarity is in [-1, 1], so we need to convert it to [0, 1]
            hd_loss = (1 - hidden_states_loss) * self.hd_lambda / 2
        else:
            hd_loss = torch.log(hidden_states_loss) * self.hd_lambda

        # 5. orthogonality regularization loss
        if self.orth_reg_lambda == 0:
            orth_reg_loss = torch.tensor(0.0, device=device)
        else:
            orth_reg_loss = self.compute_orthogonality_reg(device=device) * self.orth_reg_lambda

        # 6. singular value sparsity regularization loss
        if self.sing_reg_lambda == 0:
            sing_reg_loss = torch.tensor(0.0, device=device)
        else:
            sing_reg_loss = self.compute_singular_sparsity_reg(device=device) * self.sing_reg_lambda

        # total loss
        loss = kd_loss + task_loss + norm_loss + hd_loss + orth_reg_loss + sing_reg_loss

        return MoEDistillationOutput(
            loss=loss,
            student_logits=student_logits,
            kd_loss=kd_loss,
            hd_loss=hd_loss,
            task_loss=task_loss,
            norm_loss=norm_loss,
            sing_reg_loss=sing_reg_loss,
            orth_reg_loss=orth_reg_loss,
        )

    def save_pretrained(self, *args, **kwargs):
        self.student.save_pretrained(*args, **kwargs)

    def compute_u_v_reg_for_next_rank(self, device) -> torch.Tensor:
        """
        Intuitively, this computes the average singular values of components to be pruned next iter.

        Given next_rank, we compute the average singular values of components by
         U[:, next_rank:] @ V[:, next_rank:].T
        """
        reg_loss = torch.tensor(0.0, device=device)
        next_rank = self.next_rank
        for layer_idx in self.student.config.sparse_encoder_layer_list:
            encoder_mlp = self.student.encoder.block[layer_idx].layer[-1].mlp
            decoder_mlp = self.student.decoder.block[layer_idx].layer[-1].mlp
            for expert_idx in range(self.student.config.num_experts):
                u_norm_loss = LA.norm(
                    encoder_mlp.experts[f"expert_{expert_idx}"].wi_u,
                    ord=2,
                    dim=0
                )[next_rank:]
                v_norm_loss = LA.norm(
                    encoder_mlp.experts[f"expert_{expert_idx}"].wi_v,
                    ord=2,
                    dim=1
                )[next_rank:]
                reg_loss += (u_norm_loss * v_norm_loss).mean()

                u_norm_loss = LA.norm(
                    encoder_mlp.experts[f"expert_{expert_idx}"].wo_u,
                    ord=2,
                    dim=0
                )[next_rank:]
                v_norm_loss = LA.norm(
                    encoder_mlp.experts[f"expert_{expert_idx}"].wo_v,
                    ord=2,
                    dim=1
                )[next_rank:]
                reg_loss += (u_norm_loss * v_norm_loss).mean()

                u_norm_loss = LA.norm(
                    decoder_mlp.experts[f"expert_{expert_idx}"].wi_u,
                    ord=2,
                    dim=0
                )[next_rank:]
                v_norm_loss = LA.norm(
                    decoder_mlp.experts[f"expert_{expert_idx}"].wi_v,
                    ord=2,
                    dim=1
                )[next_rank:]
                reg_loss += (u_norm_loss * v_norm_loss).mean()

                u_norm_loss = LA.norm(
                    decoder_mlp.experts[f"expert_{expert_idx}"].wo_u,
                    ord=2,
                    dim=0
                )[next_rank:]
                v_norm_loss = LA.norm(
                    decoder_mlp.experts[f"expert_{expert_idx}"].wo_v,
                    ord=2,
                    dim=1
                )[next_rank:]
                reg_loss += (u_norm_loss * v_norm_loss).mean()

        reg_loss /= len(self.student.config.sparse_encoder_layer_list) * self.student.config.num_experts * 4

        return reg_loss

    def compute_singular_sparsity_reg(self, device) -> torch.Tensor:
        """
        Sparsity-inducing regularization loss for S matrix in USV decomposition.

        Given W=USV, we compute the sparsity-inducing regularization loss of W by
            reg_loss = ||W||_1 / ||W||_2
        """
        if not hasattr(self.student.config, "keep_singular_values") or not self.student.config.keep_singular_values:
            raise RuntimeError(f"Student model must be of form USV^T, but keep_singular_values is not set to True")

        reg_loss = torch.tensor(0.0, device=device)
        for layer_idx in self.student.config.sparse_encoder_layer_list:
            encoder_mlp = self.student.encoder.block[layer_idx].layer[-1].mlp
            decoder_mlp = self.student.decoder.block[layer_idx].layer[-1].mlp
            for expert_idx in range(self.student.config.num_experts):
                # Be careful with NaN
                reg_loss += LA.norm(encoder_mlp.experts[f"expert_{expert_idx}"].wi_mat_s, ord=1) / (LA.norm(
                    encoder_mlp.experts[f"expert_{expert_idx}"].wi_mat_s, ord=2) + 1e-5)
                reg_loss += LA.norm(encoder_mlp.experts[f"expert_{expert_idx}"].wo_mat_s, ord=1) / (LA.norm(
                    encoder_mlp.experts[f"expert_{expert_idx}"].wo_mat_s, ord=2) + 1e-5)
                reg_loss += LA.norm(decoder_mlp.experts[f"expert_{expert_idx}"].wi_mat_s, ord=1) / (LA.norm(
                    decoder_mlp.experts[f"expert_{expert_idx}"].wi_mat_s, ord=2) + 1e-5)
                reg_loss += LA.norm(decoder_mlp.experts[f"expert_{expert_idx}"].wo_mat_s, ord=1) / (LA.norm(
                    decoder_mlp.experts[f"expert_{expert_idx}"].wo_mat_s, ord=2) + 1e-5)

        reg_loss /= len(self.student.config.sparse_encoder_layer_list) * self.student.config.num_experts * 4

        return reg_loss

    def compute_orthogonality_reg(self, device) -> torch.Tensor:
        """
        Orthogonality inducer for U and V matrices in USV decomposition.

        Given W=USV, we compute the orthogonality-inducing regularization loss of U and V by
            reg_loss = ( ||U^T U - I||_F^2 + ||V^T V - I||_F^2 ) / 2

        """

        if not hasattr(self.student.config, "keep_singular_values") or not self.student.config.keep_singular_values:
            raise RuntimeError(f"Student model must be of form USV^T, but keep_singular_values is not set to True")

        reg_loss = torch.tensor(0.0, device=device)
        for layer_idx in self.student.config.sparse_encoder_layer_list:
            encoder_mlp = self.student.encoder.block[layer_idx].layer[-1].mlp
            decoder_mlp = self.student.decoder.block[layer_idx].layer[-1].mlp
            for expert_idx in range(self.student.config.num_experts):
                reg_loss += LA.norm(
                    encoder_mlp.experts[f"expert_{expert_idx}"].wi_mat_u.T @ encoder_mlp.experts[
                        f"expert_{expert_idx}"].wi_mat_u - torch.eye(self.student.config.low_rank_factor,
                                                                     device=device, requires_grad=False),
                    ord="fro"
                )
                reg_loss += LA.norm(
                    encoder_mlp.experts[f"expert_{expert_idx}"].wi_mat_v @ encoder_mlp.experts[
                        f"expert_{expert_idx}"].wi_mat_v.T - torch.eye(self.student.config.low_rank_factor,
                                                                       device=device, requires_grad=False),
                    ord="fro"
                )
                reg_loss += LA.norm(
                    encoder_mlp.experts[f"expert_{expert_idx}"].wo_mat_u.T @ encoder_mlp.experts[
                        f"expert_{expert_idx}"].wo_mat_u - torch.eye(self.student.config.low_rank_factor,
                                                                     device=device, requires_grad=False),
                    ord="fro"
                )
                reg_loss += LA.norm(
                    encoder_mlp.experts[f"expert_{expert_idx}"].wo_mat_v @ encoder_mlp.experts[
                        f"expert_{expert_idx}"].wo_mat_v.T - torch.eye(self.student.config.low_rank_factor,
                                                                       device=device, requires_grad=False),
                    ord="fro"
                )

                reg_loss += LA.norm(
                    decoder_mlp.experts[f"expert_{expert_idx}"].wi_mat_u.T @ decoder_mlp.experts[
                        f"expert_{expert_idx}"].wi_mat_u - torch.eye(self.student.config.low_rank_factor,
                                                                     device=device, requires_grad=False),
                    ord="fro"
                )
                reg_loss += LA.norm(
                    decoder_mlp.experts[f"expert_{expert_idx}"].wi_mat_v @ decoder_mlp.experts[
                        f"expert_{expert_idx}"].wi_mat_v.T - torch.eye(self.student.config.low_rank_factor,
                                                                       device=device, requires_grad=False),
                    ord="fro"
                )
                reg_loss += LA.norm(
                    decoder_mlp.experts[f"expert_{expert_idx}"].wo_mat_u.T @ decoder_mlp.experts[
                        f"expert_{expert_idx}"].wo_mat_u - torch.eye(self.student.config.low_rank_factor,
                                                                     device=device, requires_grad=False),
                    ord="fro"
                )
                reg_loss += LA.norm(
                    decoder_mlp.experts[f"expert_{expert_idx}"].wo_mat_v @ decoder_mlp.experts[
                        f"expert_{expert_idx}"].wo_mat_v.T - torch.eye(self.student.config.low_rank_factor,
                                                                       device=device, requires_grad=False),
                    ord="fro"
                )

        return reg_loss / (len(self.student.config.sparse_encoder_layer_list) * self.student.config.num_experts * 8)
