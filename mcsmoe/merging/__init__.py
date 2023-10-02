from .fisher import (
    compute_experts_exact_fisher_matrix_for_classification,
    compute_experts_empirical_fisher_matrix,
    fisher_merge_layer_experts,
    ExpertFisherTracker
)
from .grouping import (
    ExpertsGrouperForSwitch,
    merge_by_groups,
    LEGAL_SIMILARITY_BASES,
    ExpertUsageFrequencyTracker,
    prune_non_core_experts_by_groups,
    merge_by_groups_with_usage_frequency_weighting,
    merge_non_core_by_groups_with_usage_frequency_weighting,
    merge_all_non_core_to_one_with_usage_frequency_weighting,
    merge_by_groups_within_and_across_models,
    merge_non_core_by_groups_within_and_across_models,
    merge_all_non_core_to_one_within_and_across_models
)
from .utils import load_merged_switch_transformers_from_checkpoint
