"""Agent spans and separate-process I/O through the native indexed read."""

import gzip
import pathlib

import pytest
from dftracer.analyzer import init_with_hydra
from dftracer.analyzer.utils.preset_utils import classify_categories
from pandas.testing import assert_frame_equal

FIXTURE = pathlib.Path(__file__).parent / 'data' / 'agentic'
SEMANTIC_COLS = [
    'llm_count_sum',
    'llm_time_sum',
    'llm_prompt_tokens_sum',
    'llm_completion_tokens_sum',
    'llm_total_tokens_sum',
    'posix_count_sum',
    'posix_time_sum',
    'posix_read_size_sum',
    'posix_write_size_sum',
    'posix_metadata_count_sum',
    'bytes_read_per_output_token',
    'bytes_written_per_output_token',
    'io_ops_per_output_token',
    'metadata_ops_per_output_token',
    'io_time_per_output_token',
]


def overrides(trace, checkpoint_dir, cluster='none', explicit=True, checkpoint=False):
    result = [
        f'trace_path={trace}',
        f'cluster={cluster}',
        f'analyzer.checkpoint={checkpoint}',
        f'analyzer.checkpoint_dir={checkpoint_dir}',
        'view_types=[time_range]',
        'analyzer.profile_time_granularity=1',
    ]
    if explicit:
        result.append('analyzer/preset=agentic')
    return result


def analyze(args):
    app = init_with_hydra(hydra_overrides=args)
    try:
        assert app.analyzer.preset.name == 'agentic'
        return app.analyze_trace()
    finally:
        app.shutdown()


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp('agentic')
    trace = root / 'trace'
    trace.mkdir()
    for source in FIXTURE.glob("*.pfw"):
        with gzip.open(trace / (source.name + ".gz"), "wb") as out:
            out.write(source.read_bytes())
    args = overrides(trace, root / 'checkpoints', checkpoint=True)
    return analyze(args), trace, args


def test_detection_priorities():
    assert classify_categories(['POSIX', 'llm', 'step']) == 'agentic'
    assert classify_categories(['workflow']) == 'agentic'
    assert classify_categories(['llm', 'ai_root']) == 'ai'
    assert classify_categories(['llm', 'data_loader']) == 'dlio'


def assert_step_view_totals(run):
    result, _, _ = run
    view = result.get_flat_view('step').sort_index()
    assert list(view.index) == ['0', '1']
    assert list(view['llm_count_sum']) == [1, 1]
    assert list(view['llm_prompt_tokens_sum']) == [100, 200]
    assert list(view['llm_completion_tokens_sum']) == [20, 40]
    assert list(view['llm_total_tokens_sum']) == [120, 240]
    assert view.loc['0', 'posix_read_size_sum'] == 12288
    assert view.loc['1', 'posix_write_size_sum'] == 1024
    assert list(view['posix_count_sum']) == [3, 2]
    assert list(view['posix_metadata_count_sum']) == [1, 1]
    assert view.loc['0', 'posix_time_sum'] == pytest.approx(330 / 1e6)
    assert view.loc['1', 'posix_time_sum'] == pytest.approx(340 / 1e6)
    assert view.loc['0', 'llm_time_call_frac_step'] == pytest.approx(0.08 / 0.15)
    # The two read calls were folded together natively; the power sum must
    # still describe 100 us and 200 us, rather than one 300 us call.
    assert view.loc['0', 'posix_read_time_call_std'] == pytest.approx(50 / 1e6)


def assert_normalized_metrics(run):
    view = run[0].get_flat_view('step')
    assert view.loc['0', 'bytes_read_per_output_token'] == pytest.approx(614.4)
    assert view.loc['1', 'bytes_written_per_output_token'] == pytest.approx(25.6)
    assert list(view['io_ops_per_output_token']) == pytest.approx([3 / 20, 2 / 40])
    assert list(view['metadata_ops_per_output_token']) == pytest.approx([1 / 20, 1 / 40])
    assert view.loc['0', 'io_time_per_output_token'] == pytest.approx(330 / 1e6 / 20)


def assert_unassigned_io(run):
    view = run[0].get_flat_view('time_range')
    assert view['posix_count_sum'].sum() == 10
    assert view['posix_read_size_sum'].sum() == 12288 + 256 + 4096
    assert view['workflow_count_sum'].sum() == 1
    assert view['judge_count_sum'].sum() == 1
    # Both LLM calls share one process and bucket: compare processes only
    # after combining their step contributions (0.08 + 0.04 seconds).
    assert view['llm_time_proc_max'].sum() == pytest.approx(0.12)


def assert_cached_run(run):
    result, _, args = run
    cached = analyze(args)
    assert_frame_equal(result.get_flat_view('step'), cached.get_flat_view('step'))


def assert_distributed_run(run, tmp_path):
    from dask.distributed import LocalCluster

    result, trace, _ = run
    with LocalCluster(n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None) as cluster:
        args = overrides(trace, tmp_path / 'checkpoints', cluster='external', explicit=False)
        args += [f'cluster.scheduler_address={cluster.scheduler_address}', 'cluster.restart_on_connect=False']
        distributed = analyze(args)
    assert_frame_equal(
        result.get_flat_view('step')[SEMANTIC_COLS].sort_index(),
        distributed.get_flat_view('step')[SEMANTIC_COLS].sort_index(),
        check_exact=False,
        rtol=1e-9,
    )


def test_agentic_e2e_with_cache_and_distributed_detection(run, tmp_path):
    from dftracer.analyzer.output import ConsoleOutput

    assert_step_view_totals(run)
    assert_normalized_metrics(run)
    assert_unassigned_io(run)
    ConsoleOutput(compact=True).handle_result(run[0])
    assert_cached_run(run)
    assert_distributed_run(run, tmp_path)
