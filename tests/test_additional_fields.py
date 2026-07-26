"""Unit coverage for the ``additional_fields`` mechanism on Analyzer.

The extraction half of this mechanism is currently dormant on the dftracer
path: the native C++ indexer emits a fixed schema and never calls
``extra_columns_fn``. The configuration, validation, merge and aggregation
plumbing is exercised here directly so it stays correct until the indexer can
surface ``args.*`` columns.
"""

import pytest
from dftracer.analyzer.analyzer import Analyzer
from dftracer.analyzer.config import (
    AdditionalFieldConfig,
    AnalyzerPresetConfigAgent,
    AnalyzerPresetConfigPOSIX,
)


@pytest.mark.smoke
def test_merge_prefers_analyzer_level_over_preset_level():
    merged = Analyzer._merge_additional_fields(
        preset_additional_fields={
            "tokens": AdditionalFieldConfig(source="args.tokens", dtype="float64", agg="sum"),
            "kept": AdditionalFieldConfig(source="args.kept", dtype="string", agg="unique_set"),
        },
        override_additional_fields={
            "tokens": AdditionalFieldConfig(source="args.total_tokens", dtype="float64", agg="max"),
        },
    )
    assert merged["tokens"].source == "args.total_tokens"
    assert merged["tokens"].agg == "max"
    assert merged["kept"].source == "args.kept"


@pytest.mark.smoke
def test_merge_accepts_plain_dicts_and_lowercases_agg():
    merged = Analyzer._merge_additional_fields(
        preset_additional_fields={"n": {"source": "args.n", "dtype": "Int64", "agg": "SUM"}},
        override_additional_fields=None,
    )
    assert merged["n"].agg == "sum"


@pytest.mark.smoke
@pytest.mark.parametrize(
    "field_cfg, message",
    [
        (AdditionalFieldConfig(source="", dtype="float64", agg="sum"), "source must be defined"),
        (AdditionalFieldConfig(source="args.x", dtype="float64", agg="median"), "must be one of"),
        (AdditionalFieldConfig(source="args.x", dtype="not-a-dtype", agg="sum"), "not a valid pandas dtype"),
        (AdditionalFieldConfig(source="args.x", dtype="string", agg="sum"), "requires a numeric dtype"),
    ],
)
def test_invalid_additional_field_is_rejected(field_cfg, message):
    with pytest.raises(ValueError, match=message):
        Analyzer._merge_additional_fields({"x": field_cfg}, None)


@pytest.mark.smoke
def test_source_resolution_walks_dot_path_and_tolerates_absence():
    event = {"args": {"completion_tokens": "340", "nested": {"deep": 1}}}
    assert Analyzer._resolve_additional_field_source(event, "args.completion_tokens") == "340"
    assert Analyzer._resolve_additional_field_source(event, "args.nested.deep") == 1
    assert Analyzer._resolve_additional_field_source(event, "args.missing") is None
    assert Analyzer._resolve_additional_field_source(event, "nope.at.all") is None
    # A dot-path that runs into a scalar must not raise.
    assert Analyzer._resolve_additional_field_source(event, "args.completion_tokens.more") is None


@pytest.mark.smoke
def test_value_coercion_matches_declared_dtype():
    assert Analyzer._coerce_additional_field_value("340", "float64") == 340.0
    assert Analyzer._coerce_additional_field_value(340.0, "Int64") == 340
    assert Analyzer._coerce_additional_field_value(7, "string") == "7"
    assert Analyzer._coerce_additional_field_value("yes", "bool") is True
    assert Analyzer._coerce_additional_field_value(None, "float64") is None


class _Probe(Analyzer):
    """Minimal concrete Analyzer; the abstract read/postread hooks are unused here."""

    def __init__(self, preset, **kwargs):
        # Bypass Analyzer.__init__'s Dask client requirement.
        self.preset = preset
        self.additional_fields = self._merge_additional_fields(
            getattr(preset, "additional_fields", None), kwargs.get("additional_fields")
        )
        self._additional_field_names = tuple(sorted(self.additional_fields.keys(), key=len, reverse=True))
        self._additional_unique_set_fields = {
            name for name, cfg in self.additional_fields.items() if cfg.agg == "unique_set"
        }

    def read_trace(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError

    def postread_trace(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError


@pytest.mark.smoke
def test_extractor_emits_only_present_fields_with_coerced_types():
    probe = _Probe(AnalyzerPresetConfigAgent())
    extract = probe._build_additional_field_extractor()

    values = extract(
        {
            "cat": "llm",
            "args": {"completion_tokens": "340", "tool_name": "read_file", "step": 2},
        }
    )
    assert values == {"completion_tokens": 340.0, "tool_name": "read_file", "step": 2.0}
    # Absent args produce no keys at all rather than nulls.
    assert extract({"cat": "posix", "args": {}}) == {}

    columns = probe._build_additional_field_columns()
    assert columns["completion_tokens"] == "float64"
    assert columns["tool_name"] == "string"


@pytest.mark.smoke
def test_rollup_aggregation_routes_additional_fields_by_agg():
    probe = _Probe(AnalyzerPresetConfigAgent())
    suffixes = ["file_name", "host_name"]

    # Numeric additional field, plain and layer-prefixed, rolls up with its agg.
    assert probe._get_rollup_aggregation("completion_tokens", suffixes) == "sum"
    assert probe._get_rollup_aggregation("llm_completion_tokens", suffixes) == "sum"
    # unique_set fields flatten instead of summing.
    assert probe._get_rollup_aggregation("tool_name", suffixes) != "sum"
    assert probe._get_rollup_aggregation("llm_tool_name", suffixes) != "sum"
    # View-type suffixes still win over additional-field matching.
    assert probe._get_rollup_aggregation("posix_read_file_name", suffixes) != "sum"
    # Per-call stat columns keep their min/max semantics.
    assert probe._get_rollup_aggregation("time_call_min", suffixes) == "min"
    assert probe._get_rollup_aggregation("time_call_max", suffixes) == "max"
    assert probe._get_rollup_aggregation("time_sq", suffixes) == "sum"
    # Anything unrecognized sums, as before.
    assert probe._get_rollup_aggregation("posix_read_count", suffixes) == "sum"


@pytest.mark.smoke
def test_hlm_additional_aggregations_skip_groupby_fields():
    probe = _Probe(AnalyzerPresetConfigAgent())
    hlm_agg = probe._get_hlm_additional_aggregations()
    # `step` is in the agent preset's hlm_fields, so it must not be aggregated as well.
    assert "step" not in hlm_agg
    assert hlm_agg["completion_tokens"] == "sum"
    assert hlm_agg["tool_name"] != "sum"


@pytest.mark.smoke
def test_analyzer_level_additional_fields_can_be_set_from_the_cli():
    """Hydra must accept whole-dict overrides and let them win over the preset."""
    from hydra import compose, initialize
    from dftracer.analyzer.config import init_hydra_config_store

    with initialize(version_base=None, config_path=None):
        init_hydra_config_store()
        cfg = compose(
            config_name="config",
            overrides=[
                "analyzer/preset=agent",
                "+analyzer.additional_fields.event_id={source:id,dtype:Int64,agg:max}",
                "+analyzer.additional_fields.completion_tokens={source:args.ct,dtype:Int64,agg:max}",
            ],
        )

    merged = Analyzer._merge_additional_fields(
        cfg.analyzer.preset.additional_fields, cfg.analyzer.additional_fields
    )
    # New field from the CLI.
    assert merged["event_id"] == AdditionalFieldConfig(source="id", dtype="Int64", agg="max")
    # Analyzer level overrides the preset's own entry...
    assert merged["completion_tokens"].source == "args.ct"
    assert merged["completion_tokens"].agg == "max"
    # ...while untouched preset fields survive.
    assert merged["tool_name"].source == "args.tool_name"


@pytest.mark.smoke
def test_preset_without_additional_fields_is_inert():
    probe = _Probe(AnalyzerPresetConfigPOSIX())
    assert probe.additional_fields == {}
    assert probe._get_hlm_additional_aggregations() == {}
    assert probe._match_additional_field("posix_read_count") is None
    assert probe._get_rollup_aggregation("posix_read_count", ["file_name"]) == "sum"
