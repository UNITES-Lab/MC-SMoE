""" Adapted from LoSparse (arXiv 2306.11222)  """
import math
import random
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable, Set, Union

import torch
from deepspeed.utils import safe_get_full_grad, safe_get_full_fp32_param
from torch import nn
from tqdm import tqdm
from transformers import (
    SwitchTransformersForConditionalGeneration,
    SwitchTransformersSparseMLP
)

from mcsmoe.models.fsgpt_moe import (
    FSGPTMoEForCausalLM,
    FSGPTMoESparseMLP
)
from mcsmoe.models.sparse import LoSparseLinear

__all__ = [
    'LoSparseConfig',
    'substitute_switch_transformers_expert_weights',
    'substitute_fsgpt_moe_expert_weights',
    'LoSparsePruner',
    'LoSparsePrunerForFSGPT',
]


@dataclass
class LoSparseConfig:
    beta1: Optional[float] = 0.85
    beta2: Optional[int] = 1  # disable uncertainty when beta2 = 1
    deltaT: Optional[int] = 1
    initial_threshold: Optional[float] = 1.0
    final_threshold: Optional[float] = 0.25
    initial_warmup: Optional[int] = 1
    final_warmup: Optional[int] = 4
    warmup_steps: Optional[int] = 100


def low_rank_decomposition(
        weight: torch.Tensor,
        rank_ratio: Optional[float] = 0.1,
        parameter_ratio: Optional[float] = 0.15,
        remove_criteria: Optional[str] = 'max_eigenvalue',
        return_dict: Optional[bool] = False,
):
    """
    Parameters
    ----------
    weight: torch.Tensor
        The matrix to decompose, of shape (H, W)
    rank_ratio: float, optional, default 0.1
        The ratio of the reduced rank to the original rank:
            rank_of_decomposed_matrix / rank_of_input_weight
    parameter_ratio: float, optional, default 0.15
        The ratio of the number of parameters of the decomposed matrix to the original matrix:
            parameter_num_of_decomposed_matrix / (H * W).
        If specify, override rank_ratio
    remove_criteria: str, optional, default 'max_eigenvalue'
        The criteria to remove the small eigenvalues, of ['max_eigenvalue', 'random', 'min_eigenvalue']
    return_dict: bool, optional, default False
        Return a dict if True, else return a tuple (L, R)
    debug: bool, optional, default False
        Print debug information if True
    """
    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"
    height, width = weight.size()

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    dtype = weight.dtype
    weight = weight.float()
    u_, s_, v_ = torch.linalg.svd(weight, full_matrices=False)
    u_ = u_.to(dtype)
    s_ = s_.to(dtype)
    v_ = v_.to(dtype)
    rank = torch.count_nonzero(s_)

    if parameter_ratio is not None:
        reduced_rank = math.ceil(parameter_ratio * (height * width) / (height + width))
    else:
        reduced_rank = math.ceil(rank * rank_ratio)

    if remove_criteria == 'max_eigenvalue':
        l_ = u_ @ (torch.sqrt(torch.diag(s_)[:, 0:reduced_rank]))
        r_ = torch.sqrt(torch.diag(s_)[0:reduced_rank, :]) @ v_
    elif remove_criteria == 'random':
        selected_index = random.choices(range(len(s_)), k=reduced_rank)
        l_ = u_ @ (torch.sqrt(torch.diag(s_)[:, selected_index]))
        r_ = torch.sqrt(torch.diag(s_)[selected_index, :]) @ v_
    elif remove_criteria == 'min_eigenvalue':
        len_s = len(s_)
        l_ = u_ @ (torch.sqrt(torch.diag(s_)[:, len_s - reduced_rank:]))
        r_ = torch.sqrt(torch.diag(s_)[len_s - reduced_rank:, :]) @ v_
    else:
        raise NameError("remove criteria not support")

    if return_dict:
        return {"L": l_, "R": r_, "U": u_, "S": s_, "Vh": v_, 'reduced_rank': reduced_rank}
    else:
        return l_, r_


def prune(module):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == LoSparseLinear:
            print("====================================================")
            print(attr_str, target_attr)
            target_attr.prune_sparse()
    for name, immediate_child_module in module.named_children():
        prune(immediate_child_module)


def _substitute_single_linear_weight(
        module: nn.Module,
        parameter_ratio: float,
        has_sparse: bool,
        use_svd: bool,
        **kwargs
) -> nn.Module:
    """
    Substitute a single Linear weight with to LinearLoSparse

    Examples
    --------
    >>> linear = nn.Linear(16, 32)
    >>> linear = _substitute_single_linear_weight(linear, parameter_ratio=0.15, has_sparse=True, use_svd=True)
    Reduced Rank: 2 | Num Parameters: 96
    >>> linear
    LinearLoSparse(
      (right): Linear(in_features=16, out_features=2, bias=False)
      (left): Linear(in_features=2, out_features=32, bias=False)
      (sparse): Linear(in_features=16, out_features=32, bias=False)
    )
    """
    has_bias = module.bias is not None

    if use_svd:
        # Decompose a matrix by SVD
        device = module.weight.device
        output = low_rank_decomposition(module.weight.cuda(), parameter_ratio=parameter_ratio,
                                        return_dict=True, **kwargs)
        l_, r_, reduced_rank = output['L'], output['R'], output['reduced_rank']
        s_ = module.weight.cuda() - torch.mm(l_, r_)

        # Create a nn.Module and assign decomposed weights to the parameters
        linear_loras = LoSparseLinear(module.in_features, module.out_features, reduced_rank,
                                      has_bias=has_bias, has_sparse=has_sparse)
        linear_loras.initialize_weight(l_.to(device), r_.to(device), s_.to(device), module.bias)

    else:
        height, width = module.weight.shape
        reduced_rank = math.ceil(parameter_ratio * (height * width) / (height + width))
        l_ = torch.zeros(height, reduced_rank, requires_grad=True)
        r_ = torch.zeros(reduced_rank, width, requires_grad=True)
        s_ = torch.zeros(height, width, requires_grad=True)

        # Create a nn.Module and assign decomposed weights to the parameters
        linear_loras = LoSparseLinear(module.in_features, module.out_features, reduced_rank,
                                      has_bias=has_bias, has_sparse=has_sparse)

        linear_loras.initialize_weight(l_, r_, s_, module.bias)

    return linear_loras


def _get_moe_mlp_real_expert_names(
        mlp: Union[SwitchTransformersSparseMLP, FSGPTMoESparseMLP]
) -> Set[str]:
    names = set()
    for name, _ in mlp.named_parameters():
        # capture "expert_?" in the name
        match = re.search(r"expert_(\d+)", name)
        if match:
            names.add(match.group(0))
    return names


def substitute_switch_transformers_expert_weights(
        model: SwitchTransformersForConditionalGeneration,
        parameter_ratio: Optional[float] = 0.15,
        has_sparse: Optional[bool] = True,
        use_svd: Optional[bool] = True,
        encoder_compression_layers: Optional[List[int]] = None,
        decoder_compression_layers: Optional[List[int]] = None,
        **kwargs
) -> SwitchTransformersForConditionalGeneration:
    """
    Substitute the model's MLP weights with LoSparse

    Parameters
    ----------
    model: SwitchTransformersForConditionalGeneration
        The model to substitute
    parameter_ratio: float, optional, default 0.15
        The ratio of the number of parameters of the decomposed matrix to the original matrix:
            low rank matrix parameter / original matrix parameter
    has_sparse: bool, optional, default True.
        Use LoRaS if True, else use Low Rank only
    use_svd: bool, optional, default True.
        Use SVD if True, else use Low Rank only
    encoder_compression_layers: List[int], optional, default None
        The layers to substitute in the encoder. If None, substitute all layers
    decoder_compression_layers: List[int], optional, default None
        The layers to substitute in the decoder. If None, substitute all layers

    Returns
    -------
    model: SwitchTransformersForConditionalGeneration
        The model with LoSparse weights
    """
    print(f"[LoSparse] Before substituting: {model.num_parameters()}")
    if encoder_compression_layers is not None:
        print(f"[LoSparse] Substituting partial encoder layers: {encoder_compression_layers}")
    if decoder_compression_layers is not None:
        print(f"[LoSparse] Substituting partial decoder layers: {decoder_compression_layers}")
    config = model.config
    num_layers = config.num_layers
    if hasattr(config, 'sparse_encoder_layer_list'):
        sparse_layer_list = config.sparse_encoder_layer_list
    else:
        sparse_layer_list = [i for i in range(1, num_layers, config.encoder_sparse_step)]

    for layer_idx in tqdm(sparse_layer_list, desc="Substituting MLP weights"):
        if encoder_compression_layers is None or layer_idx in encoder_compression_layers:
            encoder_mlp = model.encoder.block[layer_idx].layer[-1].mlp
            for expert_name in _get_moe_mlp_real_expert_names(encoder_mlp):
                encoder_mlp.experts[expert_name].wi = _substitute_single_linear_weight(
                    module=encoder_mlp.experts[expert_name].wi,
                    parameter_ratio=parameter_ratio,
                    has_sparse=has_sparse,
                    use_svd=use_svd,
                    **kwargs
                )
                encoder_mlp.experts[expert_name].wo = _substitute_single_linear_weight(
                    module=encoder_mlp.experts[expert_name].wo,
                    parameter_ratio=parameter_ratio,
                    has_sparse=has_sparse,
                    use_svd=use_svd,
                    **kwargs
                )
        if decoder_compression_layers is None or layer_idx in decoder_compression_layers:
            decoder_mlp = model.decoder.block[layer_idx].layer[-1].mlp
            for expert_name in _get_moe_mlp_real_expert_names(decoder_mlp):
                decoder_mlp.experts[expert_name].wi = _substitute_single_linear_weight(
                    module=decoder_mlp.experts[expert_name].wi,
                    parameter_ratio=parameter_ratio,
                    has_sparse=has_sparse,
                    use_svd=use_svd,
                    **kwargs
                )
                decoder_mlp.experts[expert_name].wo = _substitute_single_linear_weight(
                    module=decoder_mlp.experts[expert_name].wo,
                    parameter_ratio=parameter_ratio,
                    has_sparse=has_sparse,
                    use_svd=use_svd,
                    **kwargs
                )

    print(f"[LoSparse] After substituting: {model.num_parameters()}")
    return model


def substitute_fsgpt_moe_expert_weights(
        model: FSGPTMoEForCausalLM,
        parameter_ratio: Optional[float] = 0.15,
        has_sparse: Optional[bool] = True,
        use_svd: Optional[bool] = True,
        compression_layers: Optional[List[int]] = None,
        **kwargs
) -> FSGPTMoEForCausalLM:
    print(f"[LoSparse] Before substituting: {model.num_parameters()}")
    if compression_layers is not None:
        print(f"[LoSparse] Substituting partial decoder layers: {compression_layers}")
    config = model.config
    num_layers = config.num_layers
    sparse_layer_list = [i for i in range(1, num_layers, config.sparse_step)]

    for layer_idx in tqdm(sparse_layer_list, desc="Substituting MLP weights"):
        if compression_layers is None or layer_idx in compression_layers:
            ffn = model.decoder.layers[layer_idx].ffn
            for expert_name in _get_moe_mlp_real_expert_names(ffn):
                ffn.experts[expert_name].fc1 = _substitute_single_linear_weight(
                    module=ffn.experts[expert_name].fc1,
                    parameter_ratio=parameter_ratio,
                    has_sparse=has_sparse,
                    use_svd=use_svd,
                    **kwargs
                )
                ffn.experts[expert_name].fc2 = _substitute_single_linear_weight(
                    module=ffn.experts[expert_name].fc2,
                    parameter_ratio=parameter_ratio,
                    has_sparse=has_sparse,
                    use_svd=use_svd,
                    **kwargs
                )

    print(f"[LoSparse] After substituting: {model.num_parameters()}")
    return model


class LoSparsePruner(object):
    def __init__(
            self,
            config: LoSparseConfig,
            total_step: int,
            mask_param_names: Optional[List[str]] = None,
            pruner_name: Optional[str] = 'PLATON',
            structured_method: Optional[str] = 'mean',
            structured_direction: Optional[str] = 'row'
    ):
        print(f"[LoSparse] config: {config}")
        if mask_param_names is None:
            mask_param_names = ['sparse']
        self.config = config
        self.importance = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.mask_param_names = mask_param_names
        self.total_step = total_step
        self.pruner_name = pruner_name
        self.beta1 = self.config.beta1
        self.beta2 = self.config.beta2
        self.deltaT = self.config.deltaT
        self.structured_method = structured_method
        self.structured_direction = structured_direction

    def whether_mask_para(
            self,
            name: str
    ):
        return any(mask_name in name for mask_name in self.mask_param_names)

    def structured_prune(
            self,
            is_dict_mat: torch.Tensor
    ) -> torch.Tensor:
        num_row, num_col = is_dict_mat.shape
        if self.structured_direction == 'row_col':
            if self.structured_method == "mean":
                return torch.mean(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "sum":
                return torch.sum(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "max":
                return torch.max(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "min":
                return torch.min(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            else:
                raise ValueError("Unimplemented Structured Method: %s" % self.structured_method)
        elif self.structured_direction == 'row':
            if self.structured_method == "mean":
                return torch.mean(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
            elif self.structured_method == "sum":
                return torch.sum(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
            elif self.structured_method == "max":
                return torch.max(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
            elif self.structured_method == "min":
                return torch.min(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
            else:
                raise ValueError("Unimplemented Structured Method: %s" % self.structured_method)
        elif self.structured_direction == 'col':
            if self.structured_method == "mean":
                return torch.mean(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "sum":
                return torch.sum(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "max":
                return torch.max(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "min":
                return torch.min(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            else:
                raise ValueError("Unimplemented Structured Method: %s" % self.structured_method)
        else:
            raise ValueError("Unimplemented Structured Direction: %s" % self.structured_direction)

    def schedule_threshold_comb(
            self,
            step: int
    ) -> Tuple[float, bool]:
        # Schedule the remaining ratio
        total_step = self.total_step
        initial_threshold = self.config.initial_threshold
        final_threshold = self.config.final_threshold
        initial_warmup = self.config.initial_warmup
        final_warmup = self.config.final_warmup
        warmup_steps = self.config.warmup_steps
        if step <= initial_warmup * warmup_steps:
            threshold = initial_threshold
            mask_ind = False
        elif step > (total_step - final_warmup * warmup_steps):
            threshold = final_threshold
            mask_ind = True
        else:
            spars_warmup_steps = initial_warmup * warmup_steps
            spars_schedule_steps = (final_warmup + initial_warmup) * warmup_steps
            mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedule_steps)
            threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)
            mask_ind = True if step % self.deltaT == 0 else False
        return threshold, mask_ind

    def update_ipt_with_local_window(
            self,
            named_parameters_caller: Callable,
            global_step: int
    ):
        # Calculate the sensitivity and uncertainty
        for name, param in named_parameters_caller():
            if self.whether_mask_para(name):
                if name not in self.exp_avg_ipt:
                    self.exp_avg_ipt[name] = torch.zeros_like(safe_get_full_fp32_param(param))
                    self.importance[name] = torch.zeros_like(safe_get_full_fp32_param(param))
                    if self.beta2 > 0 and self.beta2 != 1:
                        self.exp_avg_unc[name] = torch.zeros_like(safe_get_full_fp32_param(param))
                if self.pruner_name == 'Magnitude':
                    # Calculate the score of magnitude pruning
                    self.importance[name] = safe_get_full_fp32_param(param).abs().detach()
                elif self.pruner_name == 'PLATON':
                    local_step = global_step % self.deltaT
                    update_step = global_step // self.deltaT
                    if local_step == 0:
                        self.exp_avg_ipt[name] = self.beta1 * self.exp_avg_ipt[name] + (
                                1 - self.beta1) * self.importance[name]
                        if 0 < self.beta2 < 1:
                            self.exp_avg_unc[name] = (
                                    self.beta2 * self.exp_avg_unc[name] + (1 - self.beta2) * (
                                    self.importance[name] - self.exp_avg_ipt[name]).abs()
                            )
                        elif self.beta2 == 2.:
                            self.exp_avg_unc[name] = (
                                    (update_step * self.exp_avg_unc[name] + (
                                            self.importance[name] - self.exp_avg_ipt[name]) ** 2) / (
                                            update_step + 1)
                            )
                        self.importance[name] = (
                                safe_get_full_fp32_param(param) * safe_get_full_grad(param)
                        ).abs().detach()
                    else:
                        self.importance[name] = (self.importance[name] * local_step + (
                                safe_get_full_fp32_param(param) * safe_get_full_grad(param)).abs().detach()) / (
                                                        local_step + 1)
                else:
                    raise ValueError("Incorrect Pruner Name.")

    def mask_with_threshold(
            self,
            named_parameters_caller: Callable,
            threshold: float
    ) -> float:
        # Calculate the final importance score
        is_dict = dict()
        for name, param in named_parameters_caller():
            if self.whether_mask_para(name):
                if self.pruner_name == 'Magnitude':
                    is_dict[name] = self.importance[name]
                elif self.pruner_name == 'PLATON':
                    if 0 < self.beta2 < 1:
                        is_dict[name] = self.exp_avg_ipt[name] * self.exp_avg_unc[name]
                    elif self.beta2 == 1.:
                        is_dict[name] = self.exp_avg_ipt[name]
                    elif self.beta2 == 2.:
                        is_dict[name] = self.exp_avg_ipt[name] * self.exp_avg_unc.sqrt()
                    else:
                        # Handling the unaccepted beta2 to default setting
                        is_dict[name] = self.exp_avg_ipt[name] * (
                                self.importance[name] - self.exp_avg_ipt[name]).abs()
                else:
                    raise ValueError("Incorrect Pruner Name.")
                if self.structured_method is not None and len(is_dict[name].shape) == 2:
                    is_dict[name] = self.structured_prune(is_dict[name])
        # Calculate the mask threshold
        all_is = torch.cat([is_dict[n].view(-1) for n in is_dict.keys()])
        mask_threshold = torch.kthvalue(
            all_is.to(torch.float32),
            int(all_is.shape[0] * (1 - threshold))
        )[0].item()
        # Mask weights whose importance lower than threshold
        for name, param in named_parameters_caller():
            if self.whether_mask_para(name):
                param.data.masked_fill_(is_dict[name] < mask_threshold, 0.0)
        return mask_threshold

    def update_and_pruning(
            self,
            named_parameters_caller: Callable,
            global_step: int
    ) -> Tuple[float, float]:
        # Update importance score after optimizer stepping
        self.update_ipt_with_local_window(named_parameters_caller, global_step)
        # Get the remaining ratio
        threshold, mask_ind = self.schedule_threshold_comb(global_step)
        if mask_ind:
            # Mask weights during masking horizon
            mask_threshold = self.mask_with_threshold(named_parameters_caller, threshold)
        else:
            mask_threshold = None
        return threshold, mask_threshold


class LoSparsePrunerForFSGPT(object):
    def __init__(
            self,
            config: LoSparseConfig,
            total_step: int,
            mask_param_names: Optional[List[str]] = None,
            pruner_name: Optional[str] = 'PLATON',
            structured_method: Optional[str] = 'mean',
            structured_direction: Optional[str] = 'row'
    ):
        print(f"[LoSparse] config: {config}")
        if mask_param_names is None:
            mask_param_names = ['sparse']
        self.config = config
        self.importance = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.mask_param_names = mask_param_names
        self.total_step = total_step
        self.pruner_name = pruner_name
        self.beta1 = self.config.beta1
        self.beta2 = self.config.beta2
        self.deltaT = self.config.deltaT
        self.structured_method = structured_method
        self.structured_direction = structured_direction

    def whether_mask_para(
            self,
            name: str
    ):
        return any(mask_name in name for mask_name in self.mask_param_names)

    def structured_prune(
            self,
            is_dict_mat: torch.Tensor
    ) -> torch.Tensor:
        num_row, num_col = is_dict_mat.shape
        if self.structured_direction == 'row_col':
            if self.structured_method == "mean":
                return torch.mean(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "sum":
                return torch.sum(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "max":
                return torch.max(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "min":
                return torch.min(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            else:
                raise ValueError("Unimplemented Structured Method: %s" % self.structured_method)
        elif self.structured_direction == 'row':
            if self.structured_method == "mean":
                return torch.mean(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
            elif self.structured_method == "sum":
                return torch.sum(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
            elif self.structured_method == "max":
                return torch.max(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
            elif self.structured_method == "min":
                return torch.min(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
            else:
                raise ValueError("Unimplemented Structured Method: %s" % self.structured_method)
        elif self.structured_direction == 'col':
            if self.structured_method == "mean":
                return torch.mean(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "sum":
                return torch.sum(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "max":
                return torch.max(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "min":
                return torch.min(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            else:
                raise ValueError("Unimplemented Structured Method: %s" % self.structured_method)
        else:
            raise ValueError("Unimplemented Structured Direction: %s" % self.structured_direction)

    def schedule_threshold_comb(
            self,
            step: int
    ) -> Tuple[float, bool]:
        # Schedule the remaining ratio
        total_step = self.total_step
        initial_threshold = self.config.initial_threshold
        final_threshold = self.config.final_threshold
        initial_warmup = self.config.initial_warmup
        final_warmup = self.config.final_warmup
        warmup_steps = self.config.warmup_steps
        if step <= initial_warmup * warmup_steps:
            threshold = initial_threshold
            mask_ind = False
        elif step > (total_step - final_warmup * warmup_steps):
            threshold = final_threshold
            mask_ind = True
        else:
            spars_warmup_steps = initial_warmup * warmup_steps
            spars_schedule_steps = (final_warmup + initial_warmup) * warmup_steps
            mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedule_steps)
            threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)
            mask_ind = True if step % self.deltaT == 0 else False
        return threshold, mask_ind

    def update_ipt_with_local_window(
            self,
            named_parameters_caller: Callable,
            global_step: int
    ):
        # Calculate the sensitivity and uncertainty
        for name, param in named_parameters_caller():
            if self.whether_mask_para(name):
                if name not in self.exp_avg_ipt:
                    self.exp_avg_ipt[name] = torch.zeros_like(param)
                    self.importance[name] = torch.zeros_like(param)
                    if self.beta2 > 0 and self.beta2 != 1:
                        self.exp_avg_unc[name] = torch.zeros_like(param)
                if self.pruner_name == 'Magnitude':
                    # Calculate the score of magnitude pruning
                    self.importance[name] = param.abs().detach()
                elif self.pruner_name == 'PLATON':
                    local_step = global_step % self.deltaT
                    update_step = global_step // self.deltaT
                    if local_step == 0:
                        self.exp_avg_ipt[name] = self.beta1 * self.exp_avg_ipt[name] + (
                                1 - self.beta1) * self.importance[name]
                        if 0 < self.beta2 < 1:
                            self.exp_avg_unc[name] = (
                                    self.beta2 * self.exp_avg_unc[name] + (1 - self.beta2) * (
                                    self.importance[name] - self.exp_avg_ipt[name]).abs()
                            )
                        elif self.beta2 == 2.:
                            self.exp_avg_unc[name] = (
                                    (update_step * self.exp_avg_unc[name] + (
                                            self.importance[name] - self.exp_avg_ipt[name]) ** 2) / (
                                            update_step + 1)
                            )
                        self.importance[name] = (
                                param * param.grad
                        ).abs().detach()
                    else:
                        self.importance[name] = (self.importance[name] * local_step + (
                                param * param.grad).abs().detach()) / (local_step + 1)
                else:
                    raise ValueError("Incorrect Pruner Name.")

    def mask_with_threshold(
            self,
            named_parameters_caller: Callable,
            threshold: float
    ) -> float:
        # Calculate the final importance score
        is_dict = dict()
        for name, param in named_parameters_caller():
            if self.whether_mask_para(name):
                if self.pruner_name == 'Magnitude':
                    is_dict[name] = self.importance[name]
                elif self.pruner_name == 'PLATON':
                    if 0 < self.beta2 < 1:
                        is_dict[name] = self.exp_avg_ipt[name] * self.exp_avg_unc[name]
                    elif self.beta2 == 1.:
                        is_dict[name] = self.exp_avg_ipt[name]
                    elif self.beta2 == 2.:
                        is_dict[name] = self.exp_avg_ipt[name] * self.exp_avg_unc.sqrt()
                    else:
                        # Handling the unaccepted beta2 to default setting
                        is_dict[name] = self.exp_avg_ipt[name] * (
                                self.importance[name] - self.exp_avg_ipt[name]).abs()
                else:
                    raise ValueError("Incorrect Pruner Name.")
                if self.structured_method is not None and len(is_dict[name].shape) == 2:
                    is_dict[name] = self.structured_prune(is_dict[name])
        # Calculate the mask threshold
        all_is = torch.cat([is_dict[n].view(-1) for n in is_dict.keys()])
        mask_threshold = torch.kthvalue(
            all_is.to(torch.float32),
            int(all_is.shape[0] * (1 - threshold))
        )[0].item()
        # Mask weights whose importance lower than threshold
        for name, param in named_parameters_caller():
            if self.whether_mask_para(name):
                param.data.masked_fill_(is_dict[name] < mask_threshold, 0.0)
        return mask_threshold

    def update_and_pruning(
            self,
            named_parameters_caller: Callable,
            global_step: int
    ) -> Tuple[float, float]:
        # Update importance score after optimizer stepping
        self.update_ipt_with_local_window(named_parameters_caller, global_step)
        # Get the remaining ratio
        threshold, mask_ind = self.schedule_threshold_comb(global_step)
        if mask_ind:
            # Mask weights during masking horizon
            mask_threshold = self.mask_with_threshold(named_parameters_caller, threshold)
        else:
            mask_threshold = None
        return threshold, mask_threshold
