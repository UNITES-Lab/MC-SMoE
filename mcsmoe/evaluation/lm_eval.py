# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/18

import json
from pathlib import Path
from typing import Optional

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks
from lm_eval.utils import make_table

TASK_TO_NUM_FEWSHOT = {
    "arc_challenge": 25,
    "hellaswag": 10,
    "truthfulqa": 0,
    "mmlu": 5,
    "winogrande": 5,
    "gsm8k": 5
}


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def evaluate_fewshot(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        task: str,
        num_fewshot: int,
        log: Optional[bool] = True,
        output_path: Optional[str] = None,
):
    initialize_tasks(verbosity="WARNING")
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=1,
    )
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task,
        num_fewshot=num_fewshot,
        batch_size=1,
    )

    if output_path:
        path = Path(output_path)
        # check if file or 'dir/results.json' exists
        if path.is_file() or Path(output_path).joinpath("results.json").is_file():
            print(
                f"File already exists at {path}. Results will be overwritten."
            )
            output_path_file = path.joinpath("results.json")
            assert not path.is_file(), "File already exists"
        # if path json then get parent dir
        elif path.suffix in (".json", ".jsonl"):
            output_path_file = path
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.parent
        else:
            path.mkdir(parents=True, exist_ok=True)
            output_path_file = path.joinpath("results.json")
    else:
        output_path_file = None

    dumped = json.dumps(
        results, indent=2, default=_handle_non_serializable, ensure_ascii=False
    )

    if output_path:
        output_path_file.open("w").write(dumped)

    if log:
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

    return results
