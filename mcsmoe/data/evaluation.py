import random
from typing import Callable, Union, List, Optional, Dict

import torch
from datasets.arrow_dataset import Dataset
from evaluate import load
from promptsource.templates import DatasetTemplates
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .preprocessors import TASK_MAPPING_DATASET_ARGUMENTS, TASK_MAPPING_PROMPT_KEY, get_label_mapping_id

__all__ = [
    "get_evaluate_fn",
    "get_classification_label_index_and_token_ids",
    "EXTRA_KEYS_FOR_EVAL",
]

EXTRA_KEYS_FOR_EVAL = ["id", "idx"]


def get_evaluate_fn(
        task: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        raw_eval_dataset: Optional[Dataset] = None
) -> Callable:
    if task in ["squad", "squad_v2", "hotpotqa"]:
        return get_squad_evaluate_fn(tokenizer)
    elif task == "openbookqa":
        return get_openbookqa_evaluate_fn(tokenizer)
    elif task == "copa":
        return get_copa_evaluate_fn(tokenizer, raw_eval_dataset)
    elif task == "multirc":
        return get_multirc_evaluate_fn(tokenizer)
    elif task == "stsb":
        return get_stsb_evaluate_fn(tokenizer)
    else:
        # including other GLUE tasks, WinoGrande, WikiQA
        return get_cls_evaluate_fn(task, tokenizer)


def get_classification_label_index_and_token_ids(
        task: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> tuple:
    prompt = DatasetTemplates(*TASK_MAPPING_DATASET_ARGUMENTS[task])[TASK_MAPPING_PROMPT_KEY[task]]
    if task in ["sst2", "mrpc"]:
        classification_token_index = 0
        choices = " ".join(prompt.get_fixed_answer_choices_list())
        classification_label_token_ids = tokenizer.encode(choices, add_special_tokens=False)
    else:
        return None, None
    return classification_token_index, classification_label_token_ids


def get_cls_evaluate_fn(
        task: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> Callable:
    """
    Get the evaluate function for GLUE tasks.
    """

    def evaluate_fn(predictions: Union[torch.Tensor, List[int], List[torch.LongTensor]], labels: torch.Tensor,
                    **kwargs):
        try:
            metric = load(*TASK_MAPPING_DATASET_ARGUMENTS[task])
        except FileNotFoundError:
            print(f"[Evaluation warning] No metric found for task {task}, using accuracy instead.")
            metric = load("accuracy")
        label_mapping_id = get_label_mapping_id(task)
        eos_token_id = tokenizer.eos_token_id
        predictions = [p[:p.index(eos_token_id) + 1] if eos_token_id in p else p for
                       p in predictions]
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)  # string lengths batch
        not_even_wrong = 0
        not_even_wrong_predictions = []
        for p in predictions:
            if p not in label_mapping_id:
                not_even_wrong += 1
                not_even_wrong_predictions.append(p)
        if not_even_wrong > 0:
            print(
                f"[Evaluation warning] {not_even_wrong} among total {len(predictions)} predictions are not in the label mapping.")
            print(f"[Evaluation warning] Some of the not even wrong predictions are: {not_even_wrong_predictions[:16]}")
        print(predictions[:32])
        predictions = [label_mapping_id[p] if p in label_mapping_id else random.choice(list(label_mapping_id.values()))
                       for p in predictions]
        references = labels.long().masked_fill(labels == -100, tokenizer.pad_token_id).tolist()
        references = tokenizer.batch_decode(references, skip_special_tokens=True)
        print(references[:32])
        references = [label_mapping_id[r]
                      for r in references]
        return metric.compute(predictions=predictions, references=references)

    return evaluate_fn


def get_stsb_evaluate_fn(
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> Callable:
    """
    Get the evaluate function for GLUE tasks.
    """

    def _is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def evaluate_fn(predictions: Union[torch.Tensor, List[int], List[torch.LongTensor]], labels: torch.Tensor,
                    **kwargs):
        metric = load(*TASK_MAPPING_DATASET_ARGUMENTS["stsb"])
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)  # string lengths batch
        references = labels.long().masked_fill(labels == -100, tokenizer.pad_token_id).tolist()
        references = tokenizer.batch_decode(references, skip_special_tokens=True)
        not_even_wrong = 0
        not_even_wrong_predictions = []
        not_even_wrong_references = []
        for i, p in enumerate(predictions):
            if not _is_float(p):
                not_even_wrong += 1
                not_even_wrong_predictions.append(p)
                not_even_wrong_references.append(references[i])
        if not_even_wrong > 0:
            print(
                f"[Evaluation warning] {not_even_wrong} among total {len(predictions)} predictions are not in the label mapping.")
        predictions = [float(p) if _is_float(p) else round(random.uniform(0, 25)) / 5 for p in predictions]

        references = [float(r) for r in references]
        return metric.compute(predictions=predictions, references=references)

    return evaluate_fn


def get_multirc_evaluate_fn(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> Callable:
    """
    Get the evaluate function for SuperGLUE-MultiRC task.
    """

    def evaluate_fn(predictions: Union[torch.Tensor, List[int], List[torch.LongTensor]], labels: torch.Tensor,
                    ids: List[Dict[str, int]], **kwargs):
        metric = load(*TASK_MAPPING_DATASET_ARGUMENTS['multirc'])
        label_mapping_id = get_label_mapping_id('multirc')
        # cut up tokens after the first [EOS] token, predictions is a list of token id list
        eos_token_id = tokenizer.eos_token_id
        predictions = [p[:p.index(eos_token_id) + 1] if eos_token_id in p else p for
                       p in predictions]
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)  # string lengths batch
        not_even_wrong = 0
        not_even_wrong_preds = []
        for p in predictions:
            if p not in label_mapping_id:
                not_even_wrong += 1
                not_even_wrong_preds.append(p)
        if not_even_wrong > 0:
            print(
                f"[Evaluation warning] {not_even_wrong} among total {len(predictions)} predictions are not in the label mapping.")
            print(f"[Evaluation warning] Some of the not even wrong predictions are: {not_even_wrong_preds[:16]}")
        predictions = [label_mapping_id[p] if p in label_mapping_id else random.choice([0, 1])
                       for p in predictions]
        references = labels.long().masked_fill(labels == -100, tokenizer.pad_token_id).tolist()
        references = tokenizer.batch_decode(references, skip_special_tokens=True)
        predictions = [
            {'prediction': p, 'idx': id} for p, id in zip(predictions, ids)
        ]
        references = [label_mapping_id[r] for r in references]
        results = metric.compute(predictions=predictions, references=references)
        return {
            'f1_a': results['f1_a'],
            'f1_m': results['f1_m'],
            'exact_match': results['exact_match']
        }

    return evaluate_fn


def get_squad_evaluate_fn(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> Callable:
    """
    Get the evaluate function for SQuAD tasks.
    """

    def evaluate_fn(predictions: Union[torch.Tensor, List[int], List[torch.LongTensor]], labels: torch.Tensor,
                    ids: List[str], **kwargs):
        metric = load(*TASK_MAPPING_DATASET_ARGUMENTS['squad'])
        # cut up tokens after the first [EOS] token, predictions is a list of token id list
        eos_token_id = tokenizer.eos_token_id
        predictions = [p[:p.index(eos_token_id) + 1] if eos_token_id in p else p for
                       p in predictions]
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)  # string lengths batch
        references = labels.long().masked_fill(labels == -100, tokenizer.pad_token_id).tolist()
        references = tokenizer.batch_decode(references, skip_special_tokens=True)
        for i in range(8):
            print(predictions[i], references[i])
        predictions = [
            {'prediction_text': p, 'id': id} for p, id in zip(predictions, ids)
        ]
        references = [
            # answer_start is not used in the evaluation, so fake it
            {'answers': {'text': [reference], 'answer_start': [2333]}, 'id': id} for reference, id in
            zip(references, ids)
        ]
        return metric.compute(predictions=predictions, references=references)

    return evaluate_fn


def get_openbookqa_evaluate_fn(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> Callable:
    """
    Get the evaluate function for OpenBookQA tasks.
    """

    def evaluate_fn(predictions: Union[torch.Tensor, List[int], List[torch.LongTensor]], labels: torch.Tensor,
                    **kwargs):
        metric = load("accuracy")
        label_mapping_id = get_label_mapping_id('openbookqa')
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)  # string lengths batch
        predictions = [label_mapping_id[p] if p in label_mapping_id else -1
                       for p in predictions]
        references = labels.long().masked_fill(labels == -100, tokenizer.pad_token_id).tolist()
        references = tokenizer.batch_decode(references, skip_special_tokens=True)
        references = [label_mapping_id[reference]
                      for reference in references]
        return metric.compute(predictions=predictions, references=references)

    return evaluate_fn


def get_copa_evaluate_fn(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                         raw_eval_dataset: Dataset) -> Callable:
    """
    Get the evaluate function for SuperGLUE-COPA task.
    """
    id2choices = {
        item['idx']: [item['choice1'], item['choice2']] for item in raw_eval_dataset
    }
    id2references = {
        item['idx']: item['label'] for item in raw_eval_dataset
    }

    def evaluate_fn(predictions: Union[torch.Tensor, List[int], List[torch.LongTensor]], ids: List[int], **kwargs):
        metric = load('super_glue', 'copa')
        # cut up tokens after the first [EOS] token, predictions is a list of token id list
        eos_token_id = tokenizer.eos_token_id
        predictions = [p[:p.index(eos_token_id) + 1] if eos_token_id in p else p for
                       p in predictions]
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)  # string lengths batch
        predictions = [id2choices[idx].index(p) if p in id2choices[idx] else -1 for idx, p in zip(ids, predictions)]
        references = [id2references[idx] for idx in ids]
        return metric.compute(predictions=predictions, references=references)

    return evaluate_fn


def gather_predictions_references_by_causal_lm_loss(
        ids_list: List[int],
        answer_ids_list: List[int],
        choice_ids_list: List[int],
        losses_list: List[float],
) -> Dict[str, List[int]]:
    assert len(ids_list) == len(answer_ids_list) == len(choice_ids_list) == len(losses_list)
    num_choices = max(choice_ids_list) + 1
    idx_choice_idx_to_loss = {}
    idx_to_answer_idx = {}
    for idx, choice_idx, answer_idx, loss in zip(ids_list, choice_ids_list, answer_ids_list, losses_list):
        idx_choice_idx_to_loss[f"{idx}-{choice_idx}"] = loss
        if idx not in idx_to_answer_idx:
            idx_to_answer_idx[idx] = answer_idx
        else:
            assert idx_to_answer_idx[idx] == answer_idx
    predictions = []
    references = []
    for idx in sorted(idx_to_answer_idx.keys()):
        idx_loss_list = []
        for choice_idx in range(num_choices):
            idx_loss_list.append(idx_choice_idx_to_loss[f"{idx}-{choice_idx}"])
        best_choice_idx = idx_loss_list.index(min(idx_loss_list))
        predictions.append(best_choice_idx)
        references.append(idx_to_answer_idx[idx])
    return {"predictions": predictions, "references": references}
