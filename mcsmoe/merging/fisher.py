from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Any, Tuple, Iterator, Callable, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersDenseActDense as HFDenseActDense,
    SwitchTransformersSparseMLP,
    SwitchTransformersConfig
)

__all__ = [
    "compute_experts_exact_fisher_matrix_for_classification",
    "compute_experts_empirical_fisher_matrix",
    "fisher_merge_layer_experts",
    "ExpertFisherTracker",
]


def compute_experts_exact_fisher_matrix_for_classification(
        model: PreTrainedModel,
        dataloader: DataLoader,
        classification_token_index: int,
        classification_label_token_ids: List[int],
        no_fill: bool,
) -> Dict[str, Any]:
    """
    Given a model and a dataloader, compute the Fisher information matrix of the model.
    Note:
        1. Here we only estimate the diagonal of the Fisher matrix
        2. Since T5 is a seq2seq model, we specially design the prompt by keeping only one classification token
            different in the output sequence. Typical examples are "yes" vs "no", "A" vs "B" vs "C", etc.
        3. Thus, the classification probability of the model is the softmax of the logits of the prediction at the
            classification token index.
    """
    if no_fill:
        return {}
    if dataloader.batch_size != 1:
        raise ValueError("The batch size of the dataloader must be 1")
    if classification_label_token_ids is None:
        raise ValueError("The classification label token ids list must be provided")

    model.eval()
    model.cuda()
    num_labels = len(classification_label_token_ids)
    print(
        f"Computing the Fisher information matrix of the model with {num_labels} labels and {len(dataloader)} samples"
    )
    experts_fisher_state_dict = {}
    for name, param in model.named_parameters():
        if 'expert' in name:
            experts_fisher_state_dict[name] = torch.zeros_like(param.data, requires_grad=False, device='cuda')

    for batch in tqdm(dataloader, desc="[Fisher]Running through the dataloader"):
        # forward pass
        model.zero_grad()
        batch = {k: v.to('cuda') for k, v in batch.items()}
        logits = model(**batch).logits  # shape (1, seq_len, vocab_size)
        logits = logits.squeeze(dim=0)[classification_token_index, classification_label_token_ids]  # shape (num_labels)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        # compute the diagonal of the Fisher matrix
        for i in range(num_labels):
            log_probs[i].backward(retain_graph=True)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'expert_' in name:
                        experts_fisher_state_dict[name] += ((param.grad.data ** 2) * probs[i])
            model.zero_grad()

    experts_fisher_state_dict = {name: fisher.cpu() / len(dataloader)
                                 for name, fisher in experts_fisher_state_dict.items()}
    model.cpu()
    return {"experts_fisher_state_dict": experts_fisher_state_dict}


def compute_experts_empirical_fisher_matrix(
        model: PreTrainedModel,
        dataloader: DataLoader,
        no_fill: bool,
) -> Dict[str, Any]:
    """
    Similar to `compute_experts_fisher_matrix_for_classification`, but we compute the empirical Fisher matrix.
    This is useful for NLG/QA tasks which have an exponential explosion of the number of classes.
    """
    if no_fill:
        return {}
    if dataloader.batch_size != 1:
        raise ValueError("The batch size of the dataloader must be 1")

    model.eval()
    model.cuda()
    print(
        f"Computing the Empirical Fisher information matrix of the model with {len(dataloader)} samples"
    )
    experts_fisher_state_dict = {}
    for name, param in model.named_parameters():
        if 'expert' in name:
            experts_fisher_state_dict[name] = torch.zeros_like(param.data, requires_grad=False, device='cuda')

    for batch in tqdm(dataloader, desc=f"[Empirical Fisher]Running through {len(dataloader)} samples"):
        # forward pass
        model.zero_grad()
        batch = {k: v.to('cuda') for k, v in batch.items()}
        outputs = model(**batch)
        outputs.loss.backward(retain_graph=True)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'expert_' in name:
                    experts_fisher_state_dict[name] += (param.grad.data ** 2)

    experts_fisher_state_dict = {name: fisher.cpu() / len(dataloader)
                                 for name, fisher in experts_fisher_state_dict.items()}
    model.cpu()
    return {"experts_fisher_state_dict": experts_fisher_state_dict}


def fisher_merge_layer_experts(
        sparse_mlp: SwitchTransformersSparseMLP,
        config: SwitchTransformersConfig,
        name_prefix: str,
        experts_fisher_state_dict: Dict[str, torch.Tensor] = None,
        no_fill: bool = False,
) -> HFDenseActDense:
    """
    Merge experts in a SwitchTransformersSparseMLP into one MLP of SwitchTransformersDenseActDense
     by fisher information matrix, requires the modules to retain gradients
    """
    dense = HFDenseActDense(config)
    if no_fill:
        return dense

    wi_weight_list = []
    wo_weight_list = []
    wi_fisher_list = []
    wo_fisher_list = []
    num_experts = config.num_experts
    for expert_idx in range(num_experts):
        wi_fisher_list.append(experts_fisher_state_dict[f"{name_prefix}.expert_{expert_idx}.wi.weight"])
        wo_fisher_list.append(experts_fisher_state_dict[f"{name_prefix}.expert_{expert_idx}.wo.weight"])
        wi_weight_list.append(sparse_mlp.experts[f"expert_{expert_idx}"].wi.weight)
        wo_weight_list.append(sparse_mlp.experts[f"expert_{expert_idx}"].wo.weight)
    wi_fisher_list = torch.stack(wi_fisher_list, dim=0)
    wo_fisher_list = torch.stack(wo_fisher_list, dim=0)
    wi_weight_list = torch.stack(wi_weight_list, dim=0)
    wo_weight_list = torch.stack(wo_weight_list, dim=0)

    # nan check
    wi_weight = torch.sum(wi_weight_list * wi_fisher_list, dim=0) / (torch.sum(wi_fisher_list, dim=0) + 1e-6)
    wo_weight = torch.sum(wo_weight_list * wo_fisher_list, dim=0) / (torch.sum(wo_fisher_list, dim=0) + 1e-6)

    with torch.no_grad():
        dense.wi.weight.copy_(wi_weight)
        dense.wo.weight.copy_(wo_weight)
    return dense


class ExpertFisherTracker(object):
    """
    A class to track the Empirical Fisher information matrix of the experts during training,
        using exponential moving average


    Examples
    --------
    # 1. define the tracker
    >>> tracker = ExpertFisherTracker(named_parameters_caller=model.named_parameters,
    >>>                               begin_step=0,
    >>>                               end_step=len(train_dataloader),
    >>>                               compute_fisher_every_n_steps=1,
    >>>                               beta=0.9,
    >>>                               device='cpu')
    # --- inside the training loop ---
    # 2. compute the fisher matrix at the step `complete_steps`
    >>> fisher_state_dict = compute_experts_empirical_fisher_matrix(model, fisher_dataloader)
    >>> tracker.step(fisher_state_dict, complete_steps)
    """

    def __init__(
            self,
            named_parameters_caller: Callable[[], Iterator[Tuple[str, torch.nn.Parameter]]],
            begin_step: int,
            end_step: int,
            compute_fisher_every_n_steps: Optional[int] = 1,
            beta: Optional[float] = 0.9,
            device: Optional[str] = 'cpu',
    ):
        self.beta = beta
        self.begin_step = begin_step
        self.end_step = end_step
        self.device = device
        self.compute_fisher_every_n_steps = compute_fisher_every_n_steps
        self.exp_experts_fisher_dict = None
        self.last_error = -1

        self._init_fisher_state(named_parameters_caller)

    def fisher_state_dict(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self.exp_experts_fisher_dict)

    def _init_fisher_state(self, named_parameters_caller: Callable[[], Iterator[Tuple[str, torch.nn.Parameter]]]):
        self.exp_experts_fisher_dict = OrderedDict()
        for name, param in named_parameters_caller():
            if "expert_" in name:
                self.exp_experts_fisher_dict[name] = torch.zeros_like(param.data,
                                                                      requires_grad=False, device=self.device)

    def _update_exp_fisher_state(self, fisher_state_dict: Dict[str, torch.Tensor]):
        is_first_step = self.last_error < 0
        fisher_state_dict = {k: v.to(self.device) for k, v in fisher_state_dict.items()}
        self.last_error = sum(
            [torch.sum(torch.abs(self.exp_experts_fisher_dict[name] - fisher_state_dict[name])).item()
             for name in self.exp_experts_fisher_dict.keys()]
        )
        for name in self.exp_experts_fisher_dict.keys():
            if is_first_step:
                self.exp_experts_fisher_dict[name] = fisher_state_dict[name].detach()
            else:
                self.exp_experts_fisher_dict[name] = self.beta * self.exp_experts_fisher_dict[name] + \
                                                     (1 - self.beta) * fisher_state_dict[name]

    def step(self, fisher_state_dict: Dict[str, torch.Tensor], global_step: int) -> float:
        if (self.begin_step <= global_step <= self.end_step) and (
                global_step % self.compute_fisher_every_n_steps == 0):
            self._update_exp_fisher_state(fisher_state_dict)

        return self.last_error
