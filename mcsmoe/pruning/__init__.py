from .losparse import (
    LoSparsePruner,
    LoSparsePrunerForFSGPT,
    substitute_fsgpt_moe_expert_weights,
    substitute_switch_transformers_expert_weights
)
from .task_specific import (
    TaskSpecificExpertPruner,
    transform_switch_model_for_task_specific_pruning
)
from .task_specific_fsgpt import (
    TaskSpecificExpertPrunerForFSGPT,
    transform_fsgpt_model_for_task_specific_pruning
)

