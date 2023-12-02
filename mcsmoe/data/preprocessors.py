"""
Functionalities for pre-processing and post-processing of GLUE datasets.
"""
from typing import Dict, List, Union, Any, Optional

import torch
from datasets.arrow_dataset import Dataset
from promptsource.templates import DatasetTemplates
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

TASK_MAPPING_DATASET_ARGUMENTS = {
    "cola": ["glue", "cola"],
    "stsb": ["glue", "stsb"],
    "rte": ["glue", "rte"],
    "mnli": ["glue", "mnli"],
    "sst2": ["glue", "sst2"],
    "qqp": ["glue", "qqp"],
    "qnli": ["glue", "qnli"],
    "mrpc": ["glue", "mrpc"],
    "multirc": ["super_glue", "multirc"],
    "openbookqa": ["openbookqa", "main"],
    "squad": ["squad"],
    "squad_v2": ["squad_v2"],
    "copa": ["super_glue", "copa"],
    "winogrande": ["winogrande", "winogrande_l"],
    "wikiqa": ["wiki_qa"],
    "triviaqa": ["trivia_qa", "unfiltered"],
    "hotpotqa": ["hotpot_qa", "distractor"],
    "hellaswag": ["hellaswag"]
}

TASK_MAPPING_PROMPT_KEY = {
    "multirc": "found_this_answer",
    # "openbookqa": "pick_using_id",
    "openbookqa": "pick_answer_with_options",
    "sst2": "positive negative after",
    "mrpc": "paraphrase",
    "rte": "does the claim… follow the fact…",
    "squad": "answer_given_context_and_question",
    "squad_v2": "Questions with Context",
    # "copa": "cause_effect",
    "copa": "best_option",
    "qqp": "answer",
    "cola": "Make sense yes no",
    "stsb": "examples",
    "qnli": "based only on",
    "winogrande": "True or False",
    "wikiqa": "Decide_good_answer",
    "hotpotqa": "generate_answer_affirmative",
    "mnli": "GPT-3 style",
    "hellaswag": "Appropriate continuation - Yes or No",
}


def get_label_mapping_id(task: str) -> Dict[str, int]:
    """
    Examples
    --------
    >>> get_label_mapping_id("multirc")
    {'No': 0, 'Yes': 1}
    """
    prompt = DatasetTemplates(*TASK_MAPPING_DATASET_ARGUMENTS[task])[TASK_MAPPING_PROMPT_KEY[task]]
    choices = prompt.get_fixed_answer_choices_list()
    if task == "winogrande":
        # to minuscule for more balanced `input_ids` length
        choices = [choice.lower() for choice in choices]
    return {choice: i for i, choice in enumerate(choices)}


class Seq2SeqDataPreProcessor:
    """
    Examples
    --------
    >>> from datasets import load_dataset
    >>> proc = Seq2SeqDataPreProcessor("multirc")
    >>> dataset = load_dataset("super_glue", "multirc", split="train[:4]")
    >>> proc(dataset[0]).keys()
    dict_keys(['inputs', 'targets'])
    >>> len(proc(dataset[:2])['inputs'])
    2
    """

    def __init__(self, benchmark: str, keep_specific_keys: List[str] = None):
        self.benchmark = benchmark
        available_prompts = DatasetTemplates(*TASK_MAPPING_DATASET_ARGUMENTS[self.benchmark])
        self.prompt = available_prompts[TASK_MAPPING_PROMPT_KEY[self.benchmark]]
        self.keep_specific_keys = keep_specific_keys if keep_specific_keys else []

    def __call__(self, examples: Dict[str, Union[List, Any]], batched: Optional[bool] = True) -> Dict[str, List]:
        first_key = list(examples.keys())[0]
        if isinstance(examples[first_key], list) or batched:
            batch_size = len(examples["label"]) if "label" in examples else len(examples[first_key])
            ret = {'inputs': [], 'targets': []}
            for i in range(batch_size):
                result = self.prompt.apply({k: v[i] for k, v in examples.items()})
                ret['inputs'].append(result[0])
                if self.benchmark == "winogrande":
                    ret['targets'].append(result[1].lower())
                else:
                    ret['targets'].append(result[1])
        else:
            result = self.prompt.apply(examples)
            ret = {
                'inputs': result[0],
                'targets': result[1] if self.benchmark != "winogrande" else result[1].lower()
            }
        for key in examples:
            if key not in ret and key in self.keep_specific_keys:
                ret[key] = examples[key]
        return ret


class Seq2SeqZeroShotDataPreProcessor:
    """
    Examples
    --------
    >>> from datasets import load_dataset
    >>> proc = Seq2SeqZeroShotDataPreProcessor("winogrande")
    >>> dataset = load_dataset("winogrande", "winogrande_l", split="train[:4]")
    >>> proc(dataset[:2], batched=True).keys()
    dict_keys(['inputs', 'candidates', 'answer_ids'])
    >>> len(proc(dataset[:2])['inputs'])
    2
    """

    def __init__(self, benchmark: str, keep_specific_keys: List[str] = None):
        self.benchmark = benchmark
        available_prompts = DatasetTemplates(*TASK_MAPPING_DATASET_ARGUMENTS[self.benchmark])
        self.prompt = available_prompts[TASK_MAPPING_PROMPT_KEY[self.benchmark]]
        self.keep_specific_keys = keep_specific_keys if keep_specific_keys else []

    def __call__(self, examples: Dict[str, Union[List, Any]], batched: Optional[bool] = True) -> Dict[str, List]:
        first_key = list(examples.keys())[0]
        if isinstance(examples[first_key], list) or batched:
            batch_size = len(examples["label"]) if "label" in examples else len(examples[first_key])
            ret = {'inputs': [], 'candidates': [], 'answer_ids': []}
            for i in range(batch_size):
                sample = {k: v[i] for k, v in examples.items()}
                prompt, answer = self.prompt.apply(sample)
                choices_list = self.prompt.get_answer_choices_list(sample)
                answer_idx = choices_list.index(answer)
                ret['inputs'].append(prompt)
                ret['candidates'].append(choices_list)
                ret['answer_ids'].append(answer_idx)
        else:
            raise NotImplementedError("Seq2SeqZeroShotDataPreProcessor only supports batched=True")
        for key in examples:
            if key not in ret and key in self.keep_specific_keys:
                ret[key] = examples[key]
        return ret


class CausalZeroShotDataPreProcessor:
    """
    Examples
    --------
    >>> from datasets import load_dataset
    >>> proc = CausalZeroShotDataPreProcessor("winogrande")
    >>> dataset = load_dataset("winogrande", "winogrande_l", split="train[:4]")
    >>> proc(dataset[:2], batched=True).keys()
    dict_keys(['candidates', 'answer_ids'])
    >>> len(proc(dataset[:2], batched=True)['answer_ids'])
    2
    """

    def __init__(self, benchmark: str, keep_specific_keys: List[str] = None,
                 prompt_key: Optional[str] = None):
        self.benchmark = benchmark
        available_prompts = DatasetTemplates(*TASK_MAPPING_DATASET_ARGUMENTS[self.benchmark])
        self.prompt = available_prompts[TASK_MAPPING_PROMPT_KEY[self.benchmark]] if prompt_key is None \
            else available_prompts[prompt_key]
        self.keep_specific_keys = keep_specific_keys if keep_specific_keys else []

    def __call__(self, examples: Dict[str, Union[List, Any]], batched: Optional[bool] = True) -> Dict[str, List]:
        assert "idx" in examples, "Call build_index_for_dataset() first."
        if isinstance(examples["idx"], list) or batched:
            batch_size = len(examples["label"]) if "label" in examples else len(examples["idx"])
            ret = {'text': [], 'answer_idx': [], "choice_idx": [], "idx": []}
            for key in examples:
                if key not in ret and key in self.keep_specific_keys and key != "idx":
                    ret[key] = []
            for i in range(batch_size):
                sample = {k: v[i] for k, v in examples.items()}
                prompt, answer = self.prompt.apply(sample)
                choices_list = self.prompt.get_answer_choices_list(sample)
                if choices_list is None:
                    ret['text'].append(prompt + "\n" + answer)
                else:
                    answer_idx = choices_list.index(answer)
                    for j, choice in enumerate(choices_list):
                        ret['text'].append(prompt + "\n" + choice)
                        ret['answer_idx'].append(answer_idx)
                        ret['choice_idx'].append(j)
                        ret['idx'].append(sample['idx'])
                        for key in sample:
                            if key not in ret and key in self.keep_specific_keys and key != "idx":
                                ret[key].append(sample[key])
        else:
            raise NotImplementedError("ZeroShotCausalDataPreProcessor only supports batched=True")
        return ret


class CausalOneShotDataPreProcessor:
    """
    Examples
    --------
    >>> from datasets import load_dataset
    >>> proc = CausalZeroShotDataPreProcessor("winogrande")
    >>> dataset = load_dataset("winogrande", "winogrande_l", split="train[:4]")
    >>> proc(dataset[:2], batched=True).keys()
    dict_keys(['candidates', 'answer_ids'])
    >>> len(proc(dataset[:2], batched=True)['answer_ids'])
    2
    """

    def __init__(
            self,
            benchmark: str,
            example: Dict[str, Any],
            keep_specific_keys: List[str] = None,
            prompt_key: Optional[str] = None
    ):
        self.benchmark = benchmark
        available_prompts = DatasetTemplates(*TASK_MAPPING_DATASET_ARGUMENTS[self.benchmark])
        self.prompt = available_prompts[TASK_MAPPING_PROMPT_KEY[self.benchmark]] if prompt_key is None \
            else available_prompts[prompt_key]
        self.keep_specific_keys = keep_specific_keys if keep_specific_keys else []
        self.example = "\n".join(self.prompt.apply(example))

    def __call__(self, examples: Dict[str, Union[List, Any]], batched: Optional[bool] = True) -> Dict[str, List]:
        assert "idx" in examples, "Call build_index_for_dataset() first."
        if isinstance(examples["idx"], list) or batched:
            batch_size = len(examples["label"]) if "label" in examples else len(examples["idx"])
            ret = {'text': [], 'answer_idx': [], "choice_idx": [], "idx": []}
            for key in examples:
                if key not in ret and key in self.keep_specific_keys and key != "idx":
                    ret[key] = []
            for i in range(batch_size):
                sample = {k: v[i] for k, v in examples.items()}
                prompt, answer = self.prompt.apply(sample)
                choices_list = self.prompt.get_answer_choices_list(sample)
                answer_idx = choices_list.index(answer)
                for j, choice in enumerate(choices_list):
                    ret['text'].append(self.example + "\n\n" + prompt + "\n" + choice)
                    ret['answer_idx'].append(answer_idx)
                    ret['choice_idx'].append(j)
                    ret['idx'].append(sample['idx'])
                    for key in sample:
                        if key not in ret and key in self.keep_specific_keys and key != "idx":
                            ret[key].append(sample[key])
        else:
            raise NotImplementedError("ZeroShotCausalDataPreProcessor only supports batched=True")
        return ret


def keep_only_supporting_facts_in_context_for_hotpotqa(examples: Dict[str, Any]):
    """ This is for fxxking long context in HotpotQA. Now keep only supporting facts in context. ^^ """
    new_context = {
        'title': [],
        'sentences': []
    }
    sup_facts = examples['supporting_facts']
    for title, sent_ids in zip(sup_facts['title'], sup_facts['sent_id']):
        vanilla_index = examples['context']['title'].index(title)
        vanilla_sentences = examples['context']['sentences'][vanilla_index]
        if len(vanilla_sentences) <= sent_ids:
            continue
        if title not in new_context['title']:
            new_context['title'].append(title)
            new_context['sentences'].append([vanilla_sentences[sent_ids]])
        else:
            new_context['sentences'][new_context['title'].index(title)].append(
                vanilla_sentences[sent_ids])
    examples['context'] = new_context
    return examples


def tokenize_seq2seq(
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        batch: Dict[str, List],
        keep_other_keys=False
) -> Dict[str, List]:
    inputs = tokenizer(batch.pop("inputs"), truncation=True, return_attention_mask=True)
    targets = tokenizer(batch.pop("targets"), truncation=True, padding=False, return_attention_mask=False)
    labels = targets["input_ids"]
    # Replace pad_token_id 0 to -100 in labels
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels]
    ret = {"input_ids": inputs["input_ids"],
           "attention_mask": inputs["attention_mask"],
           "labels": labels}
    # This is for some dataset evaluation like "idx" in MultiRC, "id" in SQuAD
    if keep_other_keys:
        for key in batch:
            ret[key] = batch[key]
    return ret


def tokenize_seq2se2_to_causal_lm(
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        batch: Dict[str, List],
) -> Dict[str, List]:
    inputs_list = batch.pop("inputs")  # of shape (batch_size)
    targets_list = batch.pop("targets")  # of shape (batch_size)
    model_inputs = tokenizer(
        [_replace_new_line_with_eos(inp + "\n" + tar, tokenizer.eos_token) for inp, tar in
         zip(inputs_list, targets_list)],
        truncation=True, return_attention_mask=True
    )
    ret = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
    }
    return ret


def tokenize_causal_generation(
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        batch: Dict[str, List],
        for_eval: Optional[bool] = False,
) -> Dict[str, List]:
    """
    For training dataset:
        Training samples are supposed to be input to model(**batch) for loss calculation.
        - input_ids: "inputs" + "\n" + "targets" + "eos_token"

    For evaluation dataset:
        Evaluation samples are supposed to be input to model.generate() for prediction.
        - inputs: "inputs" + "\n"
        - references: "targets" + "eos_token"

    """
    if for_eval:
        inputs = []
        references = []
        for inp, tar in zip(batch.pop("inputs"), batch.pop("targets")):
            inp += "\n"
            inputs.append(inp)
            references.append(tar + tokenizer.eos_token)
        inputs = tokenizer(inputs, truncation=True, return_attention_mask=True)
        references = tokenizer(references, truncation=True, padding=False, return_attention_mask=False)
        ret = {"input_ids": inputs["input_ids"],
               "attention_mask": inputs["attention_mask"],
               "references": references["input_ids"]}
        if for_eval:
            for key in batch:
                ret[key] = batch[key]
        return ret
    else:
        inputs_targets = []
        for inp, tar in zip(batch.pop("inputs"), batch.pop("targets")):
            inp += "\n"
            inputs_targets.append(inp + tar + tokenizer.eos_token)
        inputs_targets = tokenizer(inputs_targets, truncation=True, padding=False, return_attention_mask=True)
        ret = {"input_ids": inputs_targets["input_ids"],
               "attention_mask": inputs_targets["attention_mask"]}
        return ret


def tokenize_seq2seq_zero_shot(
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        batch: Dict[str, List],
) -> Dict[str, List[Union[Dict[str, List[int]], int]]]:
    """
    Returns
    -------
    Dict:
        - "candidates": List[Dict]
            - "input_ids"
            - "attention_mask"
        - "answer_ids": List[int]
    """
    inputs_list = batch.pop("inputs")  # of shape (batch_size)
    candidates_list = batch.pop("candidates")  # of shape (batch_size, num_choices)
    answer_ids_list = batch.pop("answer_ids")  # of shape (batch_size)
    batch_size = len(inputs_list)
    num_choices = len(candidates_list[0])
    candidate_texts = [candidates_list[i // num_choices][i % num_choices] for i in range(batch_size * num_choices)]
    ret = {"input_ids": [],
           "attention_mask": [],
           "candidates": [],
           "answer_ids": answer_ids_list}
    for i in range(batch_size):
        inputs = tokenizer(
            inputs_list[i],
            truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        candidates = tokenizer(
            candidate_texts[i * num_choices: (i + 1) * num_choices],
            truncation=True, padding=True, return_attention_mask=False, return_tensors=None
        )['input_ids']
        candidates = [[-100 if token == tokenizer.pad_token_id else token for token in candidate]
                      for candidate in candidates]
        ret["candidates"].append(torch.tensor(candidates))
        ret["input_ids"].append(inputs['input_ids'].repeat(num_choices, 1))
        ret["attention_mask"].append(inputs['attention_mask'].repeat(num_choices, 1))
    for key in batch:
        # keep other keys
        ret[key] = batch[key]
    return ret


def _replace_new_line_with_eos(text: str, eos_token: str) -> str:
    return text.replace("\n", eos_token)


def tokenize_causal_zero_shot(
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        batch: Dict[str, List],
        replace_new_line_with_eos: Optional[bool] = True,
        keep_input_only: Optional[bool] = False,
) -> Dict[str, List[Union[Dict[str, List[int]], int]]]:
    texts = batch.pop("text")
    if replace_new_line_with_eos:
        texts = [_replace_new_line_with_eos(text, tokenizer.eos_token) + tokenizer.eos_token for text in texts]
    else:
        texts = [text + tokenizer.eos_token for text in texts]
    texts = tokenizer(texts, truncation=True, return_attention_mask=True)
    ret = {
        "input_ids": texts["input_ids"],
        "attention_mask": texts["attention_mask"],
    }
    if not keep_input_only:
        for key in batch:
            ret[key] = batch[key]
    return ret


def convert_hotpot_to_squad_format(
        json_dict, gold_paras_only=True, combine_context=True
):
    new_dict = {"data": []}
    count = 0
    for example in json_dict:
        raw_contexts = example["context"]
        if gold_paras_only:
            support = {
                para_title: line_num
                for para_title, line_num in example["supporting_facts"].items()
            }
            raw_contexts = [lst for lst in raw_contexts.items() if lst[0] in support]
        contexts = ["".join(lst[1]) for lst in raw_contexts]
        if combine_context:
            contexts = [" ".join(contexts)]
        answer = example["answer"]
        for context in contexts:
            context = add_yes_no(context)
            answer_start = context.index(answer) if answer in context else -1
            new_dict["data"].append(
                create_para_dict(
                    create_example_dict(
                        context=context,
                        answer_start=answer_start,
                        answer=answer,
                        id=str(count),  # SquadExample.__repr__ only accepts type==str
                        is_impossible=(answer_start == -1),
                        question=example["question"],
                    )
                )
            )
            count += 1
    return new_dict


def build_index_for_dataset(
        dataset: Dataset
):
    """ add a key 'idx' to each example in dataset """
    if "idx" in dataset.column_names:
        return dataset
    ids = list(range(len(dataset)))
    dataset = dataset.add_column("idx", ids)
    return dataset
