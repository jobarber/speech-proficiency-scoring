"""Feature-group conventions.

Every feature is namespaced `"{group}__{name}"` so the modeling layer can select
the feature group that drives each analytic head (delivery/accuracy/prosody) and
so explanations can be grouped by construct.
"""

from __future__ import annotations

GROUPS = ("delivery", "accuracy", "prosody")


def ns(group: str, name: str) -> str:
    assert group in GROUPS, f"unknown feature group {group!r}"
    return f"{group}__{name}"


def group_of(feature_name: str) -> str:
    return feature_name.split("__", 1)[0]


def features_in_group(feature_names, group: str) -> list[str]:
    return [f for f in feature_names if group_of(f) == group]


# Direction each feature is allowed to push its score (+1 up, -1 down) in the shipped
# model. These encode the rubric, not the data: more pausing can never raise fluency,
# a better alignment score can never lower accuracy. Features whose direction is
# genuinely ambiguous for proficiency (pitch spread, rhythm variability, phonation
# ratio, run length in short sentences) are left unconstrained.
FEATURE_SIGNS = {
    "delivery__speech_rate_wps": 1,
    "delivery__speech_rate_sps": 1,
    "delivery__articulation_rate_wps": 1,
    "delivery__articulation_rate_sps": 1,
    "delivery__silence_ratio": -1,
    "delivery__n_pauses": -1,
    "delivery__pause_rate_per_min": -1,
    "delivery__pause_total_s": -1,
    "delivery__pause_mean_s": -1,
    "delivery__pause_sd_s": -1,
    "delivery__n_long_pauses": -1,
    "delivery__long_pause_ratio": -1,
    "delivery__n_runs": -1,
    "delivery__run_len_sd": -1,
    "delivery__interword_interval_sd": -1,
    "delivery__repetition_count": -1,
    "delivery__repetition_rate": -1,
    "delivery__leading_silence_s": -1,
    "delivery__mean_word_duration_s": -1,
    "accuracy__wer": -1,
    "accuracy__cer": -1,
    "accuracy__sub_rate": -1,
    "accuracy__del_rate": -1,
    "accuracy__ins_rate": -1,
    "accuracy__mean_word_prob": 1,
    "accuracy__min_word_prob": 1,
    "accuracy__std_word_prob": -1,
    "accuracy__low_conf_frac": -1,
    "accuracy__med_conf_frac": -1,
    "accuracy__gop_mean": 1,
    "accuracy__gop_min": 1,
    "accuracy__gop_std": -1,
    "accuracy__gop_low_frac": -1,
    "prosody__energy_sd_db": 1,
    "prosody__energy_dynamic_range_db": 1,
    "prosody__run_rate_sd": -1,
    "prosody__internal_long_pause_count": -1,
}
