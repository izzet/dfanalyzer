# Performance tooling

Scratch tooling for the view-computation optimisation. Not part of the package.

## Where the time goes

Profiled on `develop` @ `d8f741a`, one worker, `view_types=[time_range,proc_name]`:

| workload | events | layers | views | total | `_compute_view` | rows per view |
|---|---|---|---|---|---|---|
| `dftracer-ai` | 125,669 | 14 | 28 | 41.2 s | 31.8 s (77%) | 0–135 |
| `cm1` (48 files) | 284,041 | 1 | 2 | 9.7 s | 6.0 s (62%) | 778 |

Runtime tracks **layers × view types**, not trace size: the 125k-event trace takes
4× longer than the 284k-event one because it has 14 layers instead of 1.

Inside `_compute_view` the cost is Dask orchestration, not computation:

```
graph build   18.0 s  (44% of total)
persist()     13.8 s  (33% of total)
```

That is not upstream work being pulled through. Materialising a view's input --
which forces all upstream computation -- costs 0.69 s for `cm1`, while
`_compute_view` then spends a further 4.46 s. The same aggregation on the same
778-row frame takes **170 ms in plain pandas versus 2,462 ms through Dask**.

## Measuring size without triggering a compute

Only ask the scheduler about collections that are **already persisted**.
Measured on a lazy frame hanging off a persisted parent:

```
len(lazy_child)              -> 4 partition executions
memory_usage_per_partition() -> 4 partition executions
nbytes(persisted parent)     -> 0 partition executions
```

So `main_view` (persisted in `_compute_main_view`) can be sized for free via
`client.nbytes(keys=[f.key for f in futures_of(main_view)])`. Any size check on
a lazy frame executes the chain, which is fine on small traces and ruinous on
large ones.

## Usage

```bash
# phase attribution: how much of a run is per-view Dask overhead
python scripts/perf/view_breakdown.py tests/data/extracted/dftracer-ai ai

# equivalence: prove an optimisation changed nothing observable
python scripts/perf/equivalence.py capture tests/data/extracted/dftracer-ai ai /tmp/base
#   ... make a change ...
python scripts/perf/equivalence.py capture tests/data/extracted/dftracer-ai ai /tmp/new
python scripts/perf/equivalence.py compare /tmp/base /tmp/new
```

`compare` checks every flat view for exact equality including dtypes, plus
`raw_stats`, `layers` and `view_types`. It exits non-zero on any difference.
