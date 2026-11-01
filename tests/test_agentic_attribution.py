"""Aggregated rows must not be counted in more than one agent step."""

import pandas as pd
from dftracer.analyzer.agentic import UNASSIGNED_STEP, attribute_steps


def row(cat, start, end, step='', workflow='', agent=''):
    return {
        'cat': cat,
        'time_start': start,
        'time_end': end,
        'step': step,
        'workflow_id': workflow,
        'agent_id': agent,
        'trace_id': '',
        'host_hash': '',
        'count': 1,
    }


def test_child_process_io_and_missing_span_scope_are_inherited():
    frame = pd.DataFrame(
        [
            row('step', 0, 20, '0', 'run', 'a'),
            row('step', 20, 40, '0', 'run', 'a'),
            row('llm', 2, 12, '0', 'run'),
            row('posix', 25, 30),
            row('posix', 40, 42),
        ]
    )
    frame['pid'] = [1, 1, 1, 999, 999]
    events, _ = attribute_steps(frame)
    assert list(events['step']) == ['0', '0', '0', '0', UNASSIGNED_STEP]
    assert list(events.loc[[2, 3], 'agent_id']) == ['a', 'a']
    assert events.loc[3, 'workflow_id'] == 'run'
    assert events['count'].sum() == frame['count'].sum()


def test_overlapping_workflows_require_a_unique_scope_match():
    frame = pd.DataFrame(
        [
            row('step', 0, 40, '0', 'one', 'a'),
            row('step', 0, 40, '0', 'two', 'b'),
            row('posix', 10, 20),
            row('posix', 10, 20, workflow='two'),
            row('llm', 10, 20, '0', 'one'),
            row('llm', 10, 20, '0'),
        ]
    )
    events, _ = attribute_steps(frame)
    assert events.loc[0, 'step'] == 'one/a/0'
    assert events.loc[1, 'step'] == 'two/b/0'
    assert events.loc[2, 'step'] == UNASSIGNED_STEP
    assert events.loc[3, 'step'] == 'two/b/0'
    assert events.loc[4, 'step'] == 'one/a/0'
    assert events.loc[5, 'step'] == UNASSIGNED_STEP
    assert events['count'].sum() == 6


def test_aggregate_straddling_step_boundary_is_not_split_or_duplicated():
    frame = pd.DataFrame(
        [
            row('step', 0, 20, '0'),
            row('step', 20, 40, '1'),
            row('posix', 15, 25),
            row('posix', 20, 22),
        ]
    )
    frame.loc[2, 'count'] = 7
    events, _ = attribute_steps(frame)
    assert events.loc[2, 'step'] == UNASSIGNED_STEP
    assert events.loc[2, 'count'] == 7
    assert events.loc[3, 'step'] == '1'


def test_no_step_spans_preserves_explicit_ids_and_leaves_io_unassigned():
    events, _ = attribute_steps(
        pd.DataFrame(
            [
                row('llm', 0, 10, '0'),
                row('posix', 2, 5),
            ]
        )
    )
    assert list(events['step']) == ['0', UNASSIGNED_STEP]


def test_profiles_use_the_same_global_windows():
    spans = pd.DataFrame([row('step', 0, 40, '0')])
    profiles = pd.DataFrame([row('posix', 10, 20), row('posix', 10, 50)])
    _, profiles = attribute_steps(spans, profiles)
    assert list(profiles['step']) == ['0', UNASSIGNED_STEP]


def test_view_fragments_are_combined_per_process_before_taking_maximum():
    from dftracer.analyzer.agentic import coalesce_process_records

    frame = pd.DataFrame(
        {
            'step': ['0', '1', '0'],
            'time_range': [0, 0, 0],
            'proc_name': ['p1', 'p1', 'p2'],
            'time': [0.08, 0.04, 0.10],
            'time_sq': [0.0064, 0.0016, 0.01],
            'time_call_min': [0.08, 0.04, 0.10],
        }
    ).set_index(['step', 'time_range', 'proc_name'])
    grouped = coalesce_process_records(frame, 'time_range')
    assert grouped['time'].max() == 0.12
    assert grouped.loc[(0, 'p1'), 'time_sq'] == 0.008
    assert grouped.loc[(0, 'p1'), 'time_call_min'] == 0.04


def test_zero_token_ratios_are_unavailable_and_cache_keys_follow_settings():
    from dftracer.analyzer import init_with_hydra

    app = init_with_hydra(
        hydra_overrides=[
            'analyzer/preset=agentic',
            'cluster=none',
            'trace_path=unused',
        ]
    )
    try:
        fields = {
            'posix_read_size_sum': 4096.0,
            'posix_write_size_sum': 0.0,
            'posix_count_sum': 2.0,
            'posix_metadata_count_sum': 1.0,
            'posix_time_sum': 0.001,
            'posix_data_size_sum': 4096.0,
            'posix_data_count_sum': 1.0,
            'llm_completion_tokens_sum': 0.0,
        }
        view = app.analyzer._set_additional_metrics(pd.DataFrame([fields]), ('step',))
        for col in (
            'bytes_read_per_output_token',
            'bytes_written_per_output_token',
            'io_ops_per_output_token',
            'metadata_ops_per_output_token',
            'io_time_per_output_token',
        ):
            assert pd.isna(view.loc[0, col])
        assert view.loc[0, 'bytes_per_io_op'] > 0
        app.analyzer.time_granularity = 0.01
        first = app.analyzer.get_checkpoint_name('hlm')
        app.analyzer.time_granularity = 0.02
        assert app.analyzer.get_checkpoint_name('hlm') != first
        app.analyzer.time_granularity = 0.01
        assert app.analyzer.get_checkpoint_name('hlm') == first
        app.analyzer.preset.correlate_steps = False
        assert app.analyzer.get_checkpoint_name('hlm') != first
    finally:
        app.shutdown()
