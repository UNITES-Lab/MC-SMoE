from copy import deepcopy
from typing import Optional, List

import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersDenseActDense,
    SwitchTransformersForConditionalGeneration
)

from mcsmoe.models.fsgpt_moe import (
    FSGPTMoEDenseActDense,
    FSGPTMoEForCausalLM,
    FSGPTMoEConfig
)
from mcsmoe.utils.constants import FP32_EPS

__all__ = [
    "permute_switch_mlp_dense_expert_",
    "compute_switch_permutation_by_weight_matching",
    "align_switch_permutation_for_all_experts_by_weight_matching",
    "align_switch_permutation_for_all_experts_by_activation_matching",
    "align_fsgpt_permutation_for_all_experts_by_weight_matching",
    "compute_switch_permutation_by_activation_matching",
    "merge_switch_mlp_by_activation_matching_within_and_across_models",
    "merge_switch_mlp_by_weight_matching_within_and_across_models",
    "permute_fsgpt_ffn_dense_expert_",
    "compute_fsgpt_permutation_by_weight_matching",
]


def permute_switch_mlp_dense_expert_(
        dense_mlp: SwitchTransformersDenseActDense,
        perm: torch.Tensor,
) -> SwitchTransformersDenseActDense:
    """
    Permute the weights of MLP according to the given permutation, the behavior of the MLP is still the same as before.

    The original weights of the MLP are Wi and Wo, the permutation matrix is P, the input is x, then the output of the
    MLP is Wo @ act(Wi @ x^T), where act is the activation function. After the permutation, the output of the MLP is
    (Wo @ P) @ act((P @ Wi) @ x^T), which is the same as the original output.

    So the permutation of the MLP is (Wi, Wo) -> (P @ Wi, Wo @ P).

    Parameters
    ----------
    dense_mlp: SwitchTransformersDenseActDense
        The MLP to be permuted.
    perm: torch.Tensor, of shape (d_ff, )
        The permutation vector (not permutation matrix) to be applied to the MLP.

    Returns
    -------
    mlp: SwitchTransformersDenseActDense
        The permuted MLP, from in-place operation.

    Examples
    --------
    >>> from transformers import SwitchTransformersConfig
    >>> mlp = SwitchTransformersDenseActDense(SwitchTransformersConfig(d_model=16, d_ff=32, dropout_rate=0))
    >>> dummy_input = torch.randn(4, 16)
    >>> permuted_mlp = permute_switch_mlp_dense_expert_(deepcopy(mlp), torch.randperm(32))
    >>> torch.allclose(mlp(dummy_input), permuted_mlp(dummy_input))
    True
    """
    d_ff = dense_mlp.wi.out_features

    # Check the permutation vector
    if perm.shape != (d_ff,):
        raise ValueError(f"The shape of the permutation vector should be (d_ff, ), but got {perm.shape}.")
    if not torch.allclose(perm.sort()[0], torch.arange(d_ff, device=perm.device)):
        raise ValueError("The permutation vector should be a permutation.")

    # Permute the weights of the MLP
    with torch.no_grad():
        dense_mlp.wi.weight.data = dense_mlp.wi.weight.data[perm, :]
        dense_mlp.wo.weight.data = dense_mlp.wo.weight.data[:, perm]

    return dense_mlp


def compute_switch_permutation_by_weight_matching(
        reference_mlp: SwitchTransformersDenseActDense,
        target_mlp: SwitchTransformersDenseActDense,
        include_wo: bool,
) -> torch.Tensor:
    """
    Compute the permutation vector by weight match that can permute the weights of the target MLP
     to match the weights of the reference MLP.

    Specifically, find a permutation of the target MLP such that the summation of L2 error between each row of
     reference MLP and target MLP is minimal.

    This can be formulated as linear sum assignment problem:
        argmin{||vec(Wr) - vec(Wt @ P)||^2} = argmax{vec(Wr) * vec(Wt @ P)}

    Parameters
    ----------
    reference_mlp: SwitchTransformersDenseActDense
        The reference MLP.
    target_mlp: SwitchTransformersDenseActDense
        The target MLP.
    include_wo: bool, default False
        Whether to include the (each column) weights of the second layer of MLP in the weight matching.

    Returns
    -------
    perm: torch.Tensor, of shape (d_ff, )
        The permutation vector that can permute the weights of the target MLP to match the weights of the reference MLP.

    Examples
    --------
    >>> from transformers import SwitchTransformersConfig
    >>> mlp = SwitchTransformersDenseActDense(SwitchTransformersConfig(d_model=16, d_ff=32, dropout_rate=0))
    >>> perm = compute_switch_permutation_by_weight_matching(mlp, mlp)
    >>> torch.allclose(perm, torch.arange(32))
    True
    """
    with torch.no_grad():
        lsa_cost_matrix = torch.mm(
            reference_mlp.wi.weight.data, target_mlp.wi.weight.data.t()
        )
        if include_wo:
            lsa_cost_matrix += torch.mm(
                reference_mlp.wo.weight.data.t(), target_mlp.wo.weight.data
            )
    _, perm = linear_sum_assignment(lsa_cost_matrix.cpu().numpy(), maximize=True)
    return torch.from_numpy(perm).to(lsa_cost_matrix.device)


def align_switch_permutation_for_all_experts_by_weight_matching(
        switch_model: SwitchTransformersForConditionalGeneration,
        include_wo: bool,
) -> SwitchTransformersForConditionalGeneration:
    """
    Align the permutation of all experts in the switch model by weight matching.

    Parameters
    ----------
    switch_model: SwitchTransformersForConditionalGeneration
        The switch model to be aligned.
    include_wo: bool, default False
        Whether to include the (each column) weights of the second layer of MLP in the weight matching.

    Returns
    -------
    switch_model: SwitchTransformersForConditionalGeneration
        The aligned switch model, from in-place operation.
    """
    config = switch_model.config
    sparse_layer_indices = list(range(1, config.num_layers, config.encoder_sparse_step))
    num_experts = config.num_experts
    for layer_idx in tqdm(sparse_layer_indices,
                          desc=f"[Permutation]Aligning permutation {'with' if include_wo else 'without'} Wo"):
        encoder_mlp = switch_model.encoder.block[layer_idx].layer[-1].mlp
        decoder_mlp = switch_model.decoder.block[layer_idx].layer[-1].mlp
        for expert_idx in range(1, num_experts):
            # Permute encoder
            perm = compute_switch_permutation_by_weight_matching(
                encoder_mlp.experts["expert_0"],
                encoder_mlp.experts[f"expert_{expert_idx}"],
                include_wo=include_wo,
            )
            encoder_mlp.experts[f"expert_{expert_idx}"] = permute_switch_mlp_dense_expert_(
                encoder_mlp.experts[f"expert_{expert_idx}"], perm
            )
            # Permute decoder
            perm = compute_switch_permutation_by_weight_matching(
                decoder_mlp.experts["expert_0"],
                decoder_mlp.experts[f"expert_{expert_idx}"],
                include_wo=include_wo,
            )
            decoder_mlp.experts[f"expert_{expert_idx}"] = permute_switch_mlp_dense_expert_(
                decoder_mlp.experts[f"expert_{expert_idx}"], perm
            )

    return switch_model


def permute_fsgpt_ffn_dense_expert_(
        dense_ffn: FSGPTMoEDenseActDense,
        perm: torch.Tensor,
) -> FSGPTMoEDenseActDense:
    """
    Similar to permute_switch_mlp_dense_expert, but for fsgpt

    Examples
    --------
    >>> ffn = FSGPTMoEDenseActDense(FSGPTMoEConfig(hidden_size=8), intermediate_size=32).eval()
    >>> perm = torch.randperm(32)
    >>> permuted_ffn = permute_fsgpt_ffn_dense_expert_(deepcopy(ffn), perm)
    >>> dummy_input = torch.arange(32).reshape(4,8).float()
    >>> torch.allclose(ffn(dummy_input), permuted_ffn(dummy_input))
    True
    """
    hidden_size = dense_ffn.fc1.out_features

    # Check the permutation vector
    if perm.shape != (hidden_size,):
        raise ValueError(f"The shape of the permutation vector should be (d_ff, ), but got {perm.shape}.")
    if not torch.allclose(perm.sort()[0], torch.arange(hidden_size, device=perm.device)):
        raise ValueError("The permutation vector should be a permutation.")

    # Permute the weights of the FFN
    with torch.no_grad():
        dense_ffn.fc1.weight.data = dense_ffn.fc1.weight.data[perm, :]
        dense_ffn.fc1.bias.data = dense_ffn.fc1.bias.data[perm]
        dense_ffn.fc2.weight.data = dense_ffn.fc2.weight.data[:, perm]

    return dense_ffn


def compute_fsgpt_permutation_by_weight_matching(
        reference_ffn: FSGPTMoEDenseActDense,
        target_ffn: FSGPTMoEDenseActDense,
        include_wo: Optional[bool] = True,
) -> torch.Tensor:
    """
    Similar to compute_switch_permutation_by_weight_matching, but for fsgpt

    Will transform BF16 to FP16 for computation
    Examples
    --------
    >>> ffn = FSGPTMoEDenseActDense(FSGPTMoEConfig(hidden_size=8), intermediate_size=32)
    >>> perm = compute_fsgpt_permutation_by_weight_matching(ffn, ffn)
    >>> torch.allclose(perm, torch.arange(32))
    True
    """
    with torch.no_grad():
        lsa_cost_matrix = torch.mm(
            reference_ffn.fc1.weight.data.float(), target_ffn.fc1.weight.data.t().float()
        )
        if include_wo:
            lsa_cost_matrix += torch.mm(
                reference_ffn.fc2.weight.data.t().float(), target_ffn.fc2.weight.data.float()
            )
    _, perm = linear_sum_assignment(lsa_cost_matrix.cpu().numpy(), maximize=True)
    return torch.from_numpy(perm).to(lsa_cost_matrix.device)


def align_fsgpt_permutation_for_all_experts_by_weight_matching(
        fsgpt_model: FSGPTMoEForCausalLM,
        include_wo: bool = True,
) -> FSGPTMoEForCausalLM:
    config = fsgpt_model.config
    sparse_layer_indices = list(range(1, config.num_layers, config.sparse_step))
    num_experts = config.num_experts
    progress_bar = tqdm(range(len(sparse_layer_indices) * (num_experts - 1)),
                        desc=f"[Permutation]Aligning permutation {'with' if include_wo else 'without'} Wo")
    for layer_idx in sparse_layer_indices:
        ffn = fsgpt_model.decoder.layers[layer_idx].ffn
        for expert_idx in range(1, num_experts):
            perm = compute_fsgpt_permutation_by_weight_matching(
                ffn.experts["expert_0"],
                ffn.experts[f"expert_{expert_idx}"],
                include_wo=include_wo,
            )
            ffn.experts[f"expert_{expert_idx}"] = permute_fsgpt_ffn_dense_expert_(
                ffn.experts[f"expert_{expert_idx}"], perm
            )
            progress_bar.update(1)

    return fsgpt_model


def compute_switch_permutation_by_activation_matching(
        reference_mlp: SwitchTransformersDenseActDense,
        target_mlp: SwitchTransformersDenseActDense,
        forwarded_hidden_states: torch.Tensor,
        mini_batch_size: Optional[int] = None,
) -> torch.Tensor:
    """
    In particular, the forwarded_hidden_states is supposed to be gathered from tokens routed to both
    reference_mlp and target_mlp, and the permutation is computed by matching the activation of the
    forwarded_hidden_states.

    Parameters
    ----------
    reference_mlp: SwitchTransformersDenseActDense
        The reference MLP.
    target_mlp: SwitchTransformersDenseActDense
        The target MLP.
    forwarded_hidden_states: torch.Tensor, of shape (batch_size * seq_len, d_ff)
        The hidden states that are forwarded to both reference_mlp and target_mlp.
    mini_batch_size: int, default None
        The mini batch size for the activation matching. If None, the mini batch size is set to be the batch size.

    Returns
    -------
    perm: torch.Tensor, of shape (d_ff, )
        The permutation vector that can permute the weights of the target MLP to match the weights of the reference MLP.

    Examples
    --------
    >>> from transformers import SwitchTransformersConfig
    >>> from copy import deepcopy
    >>> mlp = SwitchTransformersDenseActDense(SwitchTransformersConfig(d_model=16, d_ff=32, dropout_rate=0))
    >>> perm = compute_switch_permutation_by_activation_matching(mlp, deepcopy(mlp), torch.randn(64, 16))
    >>> torch.allclose(perm, torch.arange(32))
    True
    """
    if len(forwarded_hidden_states) == 0 or len(forwarded_hidden_states) == 1:
        return torch.arange(reference_mlp.wi.out_features, device=forwarded_hidden_states.device)

    if forwarded_hidden_states.shape[-1] != reference_mlp.wi.in_features:
        raise ValueError(
            f"The last dimension of forwarded_hidden_states should be {reference_mlp.wi.in_features}, "
            f"but got {forwarded_hidden_states.shape[-1]}."
        )
    if mini_batch_size is None:
        mini_batch_size = forwarded_hidden_states.shape[0]
    reference_activations = []
    target_activations = []

    def _ref_activation_hook(module, input, output):
        reference_activations.append(input[0].detach().reshape(-1, input[0].shape[-1]))

    def _target_activation_hook(module, input, output):
        target_activations.append(input[0].detach().reshape(-1, input[0].shape[-1]))

    reference_handle = reference_mlp.wo.register_forward_hook(_ref_activation_hook)
    target_handle = target_mlp.wo.register_forward_hook(_target_activation_hook)
    with torch.no_grad():
        for i in range(0, forwarded_hidden_states.shape[0], mini_batch_size):
            reference_mlp(forwarded_hidden_states[i:i + mini_batch_size])
            target_mlp(forwarded_hidden_states[i:i + mini_batch_size])

    reference_activations = torch.cat(reference_activations, dim=0)  # (batch_size * seq_len, d_ff)
    target_activations = torch.cat(target_activations, dim=0)  # (batch_size * seq_len, d_ff)

    # Compute the correlation matrix as the cost matrix
    mean_ref = reference_activations.mean(dim=0, keepdim=True)  # (1, d_ff)
    mean_target = target_activations.mean(dim=0, keepdim=True)  # (1, d_ff)
    std_ref = reference_activations.std(dim=0, keepdim=True)  # (1, d_ff)
    std_target = target_activations.std(dim=0, keepdim=True)  # (1, d_ff)
    covar = torch.mm(
        (reference_activations - mean_ref).t(),
        (target_activations - mean_target)
    ) / (reference_activations.shape[0] - 1)  # (d_ff, d_ff)
    cost_matrix = covar / (std_ref.t() * std_target + FP32_EPS)  # (d_ff, d_ff)

    _, perm = linear_sum_assignment(cost_matrix.cpu().numpy(), maximize=True)
    reference_handle.remove()
    target_handle.remove()
    return torch.from_numpy(perm).to(cost_matrix.device)


def align_switch_permutation_for_all_experts_by_activation_matching(
        switch_model: SwitchTransformersForConditionalGeneration,
        dataloader: DataLoader,
) -> SwitchTransformersForConditionalGeneration:
    """
    Align the permutation of all experts in the switch model by activation matching.

    In practice, the hidden states of the tokens routed to both reference_mlp and target_mlp are gathered
        from the dataloader, and the permutation is computed by matching the activation of the hidden states.

    Parameters
    ----------
    switch_model: SwitchTransformersForConditionalGeneration
        The switch model to be aligned.
    dataloader: DataLoader
        The dataloader to be used to gather the hidden states for activation matching.

    Returns
    -------
    switch_model: SwitchTransformersForConditionalGeneration
        The aligned switch model, from in-place operation.

    """
    # {name: values}, values  will be of shape (len(dataloader), batch_size * seq_len, d_ff)
    forwarded_hidden_states = dict()

    switch_model.eval()
    config = switch_model.config
    sparse_layer_indices = list(range(1, config.num_layers, config.encoder_sparse_step))

    # Register the activation hook for all experts
    handles = []

    def _get_activation_hook(name):
        def hook(module, input, output):
            forwarded_hidden_states[name].append(input[0].detach().reshape(-1, input[0].shape[-1]))

        return hook

    for layer_idx in tqdm(sparse_layer_indices, desc="[Permutation]Registering forward hook..."):
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
        for batch in tqdm(dataloader, desc="[Permutation]Computing activations..."):
            batch = {k: v.cuda() for k, v in batch.items()}
            output = switch_model(**batch)
            for layer_idx in sparse_layer_indices:
                router_indices[f"encoder.block.{layer_idx}.layer.1.mlp"].append(
                    output.encoder_router_logits[layer_idx][1].reshape(-1)
                )
                router_indices[f"decoder.block.{layer_idx}.layer.2.mlp"].append(
                    output.decoder_router_logits[layer_idx][1].reshape(-1)
                )

    # Compute the permutation for all experts
    num_experts = config.num_experts
    progress_bar = tqdm(range(len(sparse_layer_indices) * (num_experts - 1)),
                        desc="[Permutation]Aligning permutation by activation matching")
    for layer_idx in sparse_layer_indices:
        encoder_mlp = switch_model.encoder.block[layer_idx].layer[-1].mlp
        encoder_mlp_name = f"encoder.block.{layer_idx}.layer.1.mlp"
        decoder_mlp = switch_model.decoder.block[layer_idx].layer[-1].mlp
        decoder_mlp_name = f"decoder.block.{layer_idx}.layer.2.mlp"

        expert0_encoder_hidden_states = torch.cat([
            forwarded_hidden_states[encoder_mlp_name][i][router_indices[encoder_mlp_name][i] == 0] for i in
            range(len(forwarded_hidden_states[encoder_mlp_name]))
        ])
        expert0_decoder_hidden_states = torch.cat([
            forwarded_hidden_states[decoder_mlp_name][i][router_indices[decoder_mlp_name][i] == 0] for i in
            range(len(forwarded_hidden_states[decoder_mlp_name]))
        ])
        for expert_idx in range(1, num_experts):
            # Permute encoder
            expert_encoder_hidden_states = torch.cat([
                forwarded_hidden_states[encoder_mlp_name][i][router_indices[encoder_mlp_name][i] == expert_idx]
                for i in range(len(forwarded_hidden_states[encoder_mlp_name]))
            ])
            perm = compute_switch_permutation_by_activation_matching(
                reference_mlp=encoder_mlp.experts["expert_0"],
                target_mlp=encoder_mlp.experts[f"expert_{expert_idx}"],
                forwarded_hidden_states=torch.cat([expert0_encoder_hidden_states, expert_encoder_hidden_states], dim=0)
            )
            encoder_mlp.experts[f"expert_{expert_idx}"] = permute_switch_mlp_dense_expert_(
                encoder_mlp.experts[f"expert_{expert_idx}"], perm
            )
            # Permute decoder
            expert_decoder_hidden_states = torch.cat([
                forwarded_hidden_states[decoder_mlp_name][i][router_indices[decoder_mlp_name][i] == expert_idx]
                for i in range(len(forwarded_hidden_states[decoder_mlp_name]))
            ])
            perm = compute_switch_permutation_by_activation_matching(
                reference_mlp=decoder_mlp.experts["expert_0"],
                target_mlp=decoder_mlp.experts[f"expert_{expert_idx}"],
                forwarded_hidden_states=torch.cat([expert0_decoder_hidden_states, expert_decoder_hidden_states], dim=0)
            )
            decoder_mlp.experts[f"expert_{expert_idx}"] = permute_switch_mlp_dense_expert_(
                decoder_mlp.experts[f"expert_{expert_idx}"], perm
            )

            progress_bar.update(1)

    for handle in handles:
        handle.remove()
    return switch_model


def merge_switch_mlp_by_activation_matching_within_and_across_models(
        mlp_list: List[SwitchTransformersDenseActDense],
        forwarded_hidden_states: torch.Tensor,
        mini_batch_size: Optional[int] = None,
        alpha_for_repeated_merging: Optional[float] = 0.1,
        average_coefs: Optional[List[float]] = None,
) -> SwitchTransformersDenseActDense:
    """
    Merge the MLPs in the mlp_list by activation matching. Specifically, the MLPs in the mlp_list are merged
    by matching the activation of the forwarded_hidden_states, within and across models.


    Parameters
    ----------
    mlp_list: List[SwitchTransformersDenseActDense]
        The MLPs to be merged.
    forwarded_hidden_states: torch.Tensor, of shape (batch_size * seq_len, d_ff)
        The hidden states that are forwarded to all MLPs in the mlp_list.
    mini_batch_size: int, default None
        The mini batch size for the activation matching. If None, the mini batch size is set to be the batch size.
    alpha_for_repeated_merging: float, default 0.1
        The alpha for repeated merging, which is used to update the correlation matrix after each merging.
        In particular, the correlation between the merged feature and the other features is set to be the minimum
            of the old features' correlation weighted.
    average_coefs: Optional[List[float]], default None
        The average coefficients for the features in the MLPs. If None, the average coefficients are set to be 1.0.

    Returns
    -------
    merged_mlp: SwitchTransformersDenseActDense
        The merged MLP.

    Examples
    --------
    >>> from transformers import SwitchTransformersConfig
    >>> from copy import deepcopy
    >>> mlp = SwitchTransformersDenseActDense(SwitchTransformersConfig(d_model=16, d_ff=32, dropout_rate=0))
    >>> mlp_merged = merge_switch_mlp_by_activation_matching_within_and_across_models([mlp, deepcopy(mlp)], torch.randn(64, 16))
    >>> dummy_input = torch.randn(4, 16)
    >>> torch.allclose(mlp_merged(dummy_input), mlp(dummy_input))
    True
    """
    mlp_list = [mlp.eval() for mlp in mlp_list]
    concat_mlp = deepcopy(mlp_list[0])
    d_ff, d_model = concat_mlp.wi.out_features, concat_mlp.wi.in_features
    if average_coefs is None:
        average_coefs = [1.0] * len(mlp_list) * d_ff
    elif len(average_coefs) == len(mlp_list):
        average_coefs = [coef for coef in average_coefs for _ in range(d_ff)]
    elif len(average_coefs) != len(mlp_list) * d_ff:
        raise ValueError(
            f"The length of average_coefs should be either {len(mlp_list)} or {len(mlp_list) * d_ff}, "
            f"but got {len(average_coefs)}."
        )
    num_mlp = len(mlp_list)
    if len(forwarded_hidden_states) == 0 or len(forwarded_hidden_states) == 1:
        return concat_mlp
    if mini_batch_size is None:
        mini_batch_size = forwarded_hidden_states.shape[0]

    mlp_all_wi = torch.cat([
        mlp.wi.weight.data for mlp in mlp_list
    ], dim=0)
    mlp_all_wo = torch.cat([
        mlp.wo.weight.data for mlp in mlp_list
    ], dim=1)
    concat_mlp.wi = torch.nn.Linear(d_model, d_ff * num_mlp, bias=False)
    concat_mlp.wo = torch.nn.Linear(d_ff * num_mlp, d_model, bias=False)
    with torch.no_grad():
        concat_mlp.wi.weight.data = mlp_all_wi  # (d_ff * num_mlp, d_model)
        concat_mlp.wo.weight.data = mlp_all_wo  # (d_model, d_ff * num_mlp)
    concat_mlp = concat_mlp.eval().to(forwarded_hidden_states.device)

    activations = []

    def _activation_hook(module, input, output):
        activations.append(input[0].detach().reshape(-1, input[0].shape[-1]))

    handle = concat_mlp.wo.register_forward_hook(_activation_hook)
    with torch.no_grad():
        for i in range(0, forwarded_hidden_states.shape[0], mini_batch_size):
            concat_mlp(forwarded_hidden_states[i:i + mini_batch_size])

    activations = torch.cat(activations, dim=0)  # (batch_size * seq_len, d_ff * num_mlp)

    # Initialize the correlation matrix
    mean = activations.mean(dim=0, keepdim=True)  # (1, d_ff * num_mlp)
    std = activations.std(dim=0, keepdim=True)  # (1, d_ff * num_mlp)
    covar = torch.mm(
        (activations - mean).t(),
        (activations - mean)
    ) / (activations.shape[0] - 1)  # (d_ff * num_mlp, d_ff * num_mlp)
    corr_matrix = covar / (std.t() * std + FP32_EPS)  # (d_ff * num_mlp, d_ff * num_mlp)

    del activations, covar, std, mean
    torch.cuda.empty_cache()

    corr_matrix[torch.arange(d_ff * num_mlp), torch.arange(d_ff * num_mlp)] = -1  # Remove self-correlation

    # Greedy Merging!
    while mlp_all_wi.shape[0] > d_ff:
        # Select the most correlated pair
        max_index = torch.argmax(corr_matrix)
        max_i, max_j = max_index // corr_matrix.shape[0], max_index % corr_matrix.shape[0]

        # Merge the most correlated pair, replace the first feature with the merged one
        i_coef, j_coef = average_coefs[max_i], average_coefs[max_j]
        mlp_all_wi[max_i] = (i_coef * mlp_all_wi[max_i] + j_coef * mlp_all_wi[max_j]) / (i_coef + j_coef + FP32_EPS)
        mlp_all_wo[:, max_i] = (i_coef * mlp_all_wo[:, max_i] + j_coef * mlp_all_wo[:, max_j]) / (
                i_coef + j_coef + FP32_EPS)

        # Remove the second feature
        mlp_all_wi = torch.cat([
            mlp_all_wi[:max_j],
            mlp_all_wi[max_j + 1:]
        ], dim=0)
        mlp_all_wo = torch.cat([
            mlp_all_wo[:, :max_j],
            mlp_all_wo[:, max_j + 1:]
        ], dim=1)

        # Update the correlation matrix
        updated_corr_vec = alpha_for_repeated_merging * torch.min(
            torch.stack([corr_matrix[max_i], corr_matrix[max_j]]), dim=0
        ).values
        corr_matrix[max_i] = updated_corr_vec
        corr_matrix[:, max_i] = updated_corr_vec
        corr_matrix[max_i, max_i] = -1  # Remove self-correlation

        # Remove the second feature from the correlation matrix
        corr_matrix = torch.cat([
            corr_matrix[:, :max_j],
            corr_matrix[:, max_j + 1:]
        ], dim=1)
        corr_matrix = torch.cat([
            corr_matrix[:max_j],
            corr_matrix[max_j + 1:]
        ], dim=0)

        # Update the average coefs
        average_coefs[max_i] += average_coefs[max_j]
        average_coefs = average_coefs[:max_j] + average_coefs[max_j + 1:]

    handle.remove()
    merged_mlp = deepcopy(mlp_list[0])
    with torch.no_grad():
        merged_mlp.wi.weight.data = mlp_all_wi
        merged_mlp.wo.weight.data = mlp_all_wo

    return merged_mlp


def merge_switch_mlp_by_weight_matching_within_and_across_models(
        mlp_list: List[SwitchTransformersDenseActDense],
        include_wo: Optional[bool] = True,
        average_coefs: Optional[List[float]] = None,
) -> SwitchTransformersDenseActDense:
    """
    Merge the MLPs in the mlp_list by weight matching. Specifically, the MLPs in the mlp_list are merged

    Parameters
    ----------
    mlp_list: List[SwitchTransformersDenseActDense]
        The MLPs to be merged.
    include_wo: bool, default True
        Whether to include the (each column) weights of the second layer of MLP in the weight matching.
    average_coefs: List[float], default None
        The average coefs for each MLP in the mlp_list. If None, the average coefs are set to be 1.0 for all MLPs.

    Returns
    -------
    merged_mlp: SwitchTransformersDenseActDense
        The merged MLP.

    Examples
    --------
    >>> from transformers import SwitchTransformersConfig
    >>> from copy import deepcopy
    >>> mlp = SwitchTransformersDenseActDense(SwitchTransformersConfig(d_model=16, d_ff=32, dropout_rate=0))
    >>> mlp_merged = merge_switch_mlp_by_activation_matching_within_and_across_models([mlp, deepcopy(mlp)], torch.randn(64, 16))
    >>> dummy_input = torch.randn(4, 16)
    >>> torch.allclose(mlp_merged(dummy_input), mlp(dummy_input))
    True
    """
    d_ff, d_model = mlp_list[0].wi.out_features, mlp_list[0].wi.in_features
    if average_coefs is None:
        average_coefs = [1.0] * len(mlp_list) * d_ff
    elif len(average_coefs) == len(mlp_list):
        average_coefs = [coef for coef in average_coefs for _ in range(d_ff)]
    elif len(average_coefs) != len(mlp_list) * d_ff:
        raise ValueError(
            f"The length of average_coefs should be either {len(mlp_list)} or {len(mlp_list) * d_ff}, "
            f"but got {len(average_coefs)}."
        )
    concat_wi = torch.cat([
        mlp.wi.weight.data for mlp in mlp_list
    ], dim=0)  # of shape (d_ff * num_mlp, d_model)
    concat_wo = torch.cat([
        mlp.wo.weight.data.t() for mlp in mlp_list
    ], dim=0)  # of shape (d_ff * num_mlp, d_model)
    if include_wo:
        cost_matrix = torch.mm(concat_wi, concat_wi.t()) + torch.mm(concat_wo, concat_wo.t())
    else:
        cost_matrix = torch.mm(concat_wi, concat_wi.t())

    # Remove self-correlation
    dtype_min = torch.finfo(cost_matrix.dtype).min
    cost_matrix[torch.arange(d_ff * len(mlp_list)), torch.arange(d_ff * len(mlp_list))] = dtype_min

    # Greedy Merging!
    while concat_wi.shape[0] > d_ff:
        # Select the most correlated pair
        max_index = torch.argmax(cost_matrix)
        max_i, max_j = max_index // cost_matrix.shape[0], max_index % cost_matrix.shape[0]

        # Merge the most correlated pair, replace the first feature with the merged one
        i_coef = average_coefs[max_i]
        j_coef = average_coefs[max_j]
        concat_wi[max_i] = (i_coef * concat_wi[max_i] + j_coef * concat_wi[max_j]) / (i_coef + j_coef + FP32_EPS)
        concat_wo[max_i] = (i_coef * concat_wo[max_i] + j_coef * concat_wo[max_j]) / (i_coef + j_coef + FP32_EPS)

        # Update the cost matrix
        updated_cost_vec = torch.mm(
            concat_wi[max_i].reshape(1, -1),
            concat_wi.t()
        ) + torch.mm(
            concat_wo[max_i].reshape(1, -1),
            concat_wo.t()
        )
        cost_matrix[max_i] = updated_cost_vec
        cost_matrix[:, max_i] = updated_cost_vec
        cost_matrix[max_i, max_i] = dtype_min  # Remove self-correlation

        # Remove the second feature
        concat_wi = torch.cat([
            concat_wi[:max_j],
            concat_wi[max_j + 1:]
        ], dim=0)
        concat_wo = torch.cat([
            concat_wo[:max_j],
            concat_wo[max_j + 1:]
        ], dim=0)

        # Remove the second feature from the correlation matrix
        cost_matrix = torch.cat([
            cost_matrix[:, :max_j],
            cost_matrix[:, max_j + 1:]
        ], dim=1)
        cost_matrix = torch.cat([
            cost_matrix[:max_j],
            cost_matrix[max_j + 1:]
        ], dim=0)

        # Update the average coefs
        average_coefs[max_i] += average_coefs[max_j]
        average_coefs = average_coefs[:max_j] + average_coefs[max_j + 1:]

    merged_mlp = deepcopy(mlp_list[0])
    with torch.no_grad():
        merged_mlp.wi.weight.data = concat_wi
        merged_mlp.wo.weight.data = concat_wo.t()

    return merged_mlp
