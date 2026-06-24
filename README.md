# Data Flow Analyzer

![Build and Test](https://github.com/LLNL/dfanalyzer/actions/workflows/ci.yml/badge.svg)
![PyPI - Version](https://img.shields.io/pypi/v/dftracer-analyzer?label=PyPI)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/dftracer-analyzer?label=Wheel)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dftracer-analyzer?label=Python)

## Overview

DFAnalyzer is an open-source tool for analyzing performance data from large-scale workflows on distributed systems. It presents a hierarchical, layer-by-layer summary of an application's execution, from high-level application events down to low-level POSIX calls. For each layer, DFAnalyzer quantifies time, operation counts, and data volume, and calculates key performance metrics like bandwidth and operations per second. It also visualizes the overlap between different layers, helping to characterize and understand complex I/O and compute patterns.

## Installation

To install DFAnalyzer through `pip` (recommended for most users):

```bash
# Ensure runtime dependencies for optional features (e.g., Darshan, Recorder) are installed.
# This might involve using your system's package manager or a tool like Spack.
# Example using Spack to prepare the environment:
# spack -e tools install
pip install dftracer-analyzer
```

To install DFAnalyzer from source (for developers or custom builds):

```bash
# 1. Install system dependencies:
#    Refer to the "Install system dependencies" step in .github/workflows/ci.yml
#    (e.g., build-essential, cmake, libarrow-dev, libhdf5-dev, ninja-build, etc.).
#    Alternatively, tools like Spack can help manage these:
#    # spack -e tools install
module load ninja

# 2. Install Python build dependencies:
python -m pip install --upgrade pip meson-python setuptools wheel

# 3. Install DFAnalyzer from the root of this repository:
#    The following command includes optional C++ components (tests and tools).
#    The --prefix argument is optional and specifies the installation location.
pip install -e . \
  -Csetup-args="--prefix=$HOME/.local" \
  -Csetup-args="-Denable_tests=true" \
  -Csetup-args="-Denable_tools=true"

# (Optional) Install dependencies for running tests if you plan to contribute or run local tests:
# pip install -r tests/requirements.txt
```

## Usage

Here's an example of how to run DFAnalyzer using sample data included in the repository:

```bash
# Before running, ensure the sample data is extracted.
# For example, to extract the 'dftracer-dlio' sample used below:
# mkdir -p tests/data/extracted
# tar -xzf tests/data/dftracer-dlio.tar.gz -C tests/data/extracted
dfanalyzer analyzer/preset=dlio trace_path=tests/data/extracted/dftracer-dlio view_types=[time_range]
```

This command analyzes the traces and prints a high-level summary of the application's execution. Below is a sample of the "Time Period Summary" output:

```bash
                                                  Time Period Summary
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                                                                    ┃ Unit             ┃                 Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ Job Time                                                                  │ seconds          │                56.695 │
│ Total Count                                                               │ count            │                18,039 │
│ Total Files                                                               │ count            │                   166 │
│ Total Nodes                                                               │ count            │                     1 │
│ Total Processes                                                           │ count            │                     8 │
│ App Count                                                                 │ count            │                     8 │
│ Training Count                                                            │ count            │                     8 │
│ Epoch Count                                                               │ count            │                    40 │
│ Compute Count                                                             │ count            │                   200 │
│ Fetch Data Count                                                          │ count            │                   160 │
│ Checkpoint Count                                                          │ count            │                     8 │
│ Data Loader Count                                                         │ count            │                   816 │
│ Data Loader Fork Count                                                    │ count            │                    96 │
│ Reader Count                                                              │ count            │                 3,200 │
│ POSIX - All Count                                                         │ count            │                10,581 │
│ POSIX - All Size                                                          │ MB               │            111833.172 │
│ POSIX - All Bandwidth                                                     │ MB/s             │              6048.367 │
│ POSIX - All Avg Transfer Size                                             │ MB               │                10.569 │
│ POSIX - Reader Count                                                      │ count            │                10,432 │
│ POSIX - Reader Size                                                       │ MB               │            111833.161 │
│ POSIX - Reader Bandwidth                                                  │ MB/s             │              6095.909 │
│ POSIX - Reader Avg Transfer Size                                          │ MB               │                10.720 │
│ POSIX - Checkpoint Count                                                  │ count            │                    45 │
│ POSIX - Checkpoint Size                                                   │ MB               │                 0.011 │
│ POSIX - Checkpoint Bandwidth                                              │ MB/s             │                 2.525 │
│ POSIX - Checkpoint Avg Transfer Size                                      │ MB               │                 0.000 │
└───────────────────────────────────────────────────────────────────────────┴──────────────────┴───────────────────────┘
```

DFAnalyzer also provides a detailed breakdown of performance metrics for each layer of the application. Here is a snippet of the "Layer Breakdown" section from the same run, which includes the percentage of time each layer overlaps with its parent layer:

```bash
                                             Layer Breakdown (w/ overlap %)
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer                  ┃         Time (s) ┃             Ops ┃     Ops/sec ┃            Size (MB) ┃  Bandwidth (MB/s) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ App                    │    55.246 (----) │        8 (----) │       0.145 │                    - │                 - │
│ Training               │    55.246 (----) │        8 (----) │       0.145 │                    - │                 - │
│ Epoch                  │    54.937 (----) │       40 (----) │       0.728 │                    - │                 - │
│ Compute                │    40.854 (----) │      200 (----) │       4.895 │                    - │                 - │
│ Fetch Data             │    16.889 (----) │      160 (----) │       9.474 │                    - │                 - │
│ Checkpoint             │     0.005 (----) │        8 (----) │    1762.503 │                    - │                 - │
│ Data Loader            │    21.871 ( 54%) │      816 ( 57%) │      37.310 │                    - │                 - │
│ Data Loader Fork       │     0.181 (  0%) │       96 (  0%) │     530.903 │                    - │                 - │
│ Reader                 │    21.480 ( 55%) │    3,200 ( 67%) │     148.979 │                    - │                 - │
│ POSIX - All            │    18.490 ( 54%) │   10,581 ( 59%) │     572.261 │    111833.172 ( 59%) │          6048.367 │
│ POSIX - Reader         │    18.346 ( 55%) │   10,432 ( 60%) │     568.637 │    111833.161 ( 59%) │          6095.909 │
│ POSIX - Checkpoint     │     0.004 (----) │       45 (----) │   10433.573 │         0.011 (----) │             2.525 │
└────────────────────────┴──────────────────┴─────────────────┴─────────────┴──────────────────────┴───────────────────┘
```

## Analysis facts (DFDiagnoser integration)

Beyond the human-readable summary, DFAnalyzer can emit **analysis facts** — compact,
machine-readable bottleneck signals (`analyzer.fact-envelope.v1`) that
[DFDiagnoser](https://github.com/LLNL/dfdiagnoser) turns into longitudinal findings
and [DFOptimizer](https://github.com/LLNL/dfoptimizer) turns into tuning actions. Facts
are **opt-in** and additive: with `facts.enabled=false` (the default) the analysis
output is unchanged.

A fact is produced per view per analysis window by either builder:

- **rule** (`facts.eval_mode=rule`) — YAML conditions over view metrics
  (`facts.eval_rule_file=<rules.yaml>`), e.g. *fetch time dominates compute*.
- **metric** (`facts.eval_mode=metric`) — WISIO-style slope detection: an entity whose
  share of time is disproportionate to its share of operations.

Each fact carries a continuous `severity` in [0,1], a two-level `scope`
(`layer:view` aggregate or `layer:view:entity` detail), and `opportunity_tags`.

### Producing facts to a bundle (offline)

`output=file` writes the deliverable bundle — `facts.jsonl` (one envelope per window),
`detail_view_*.parquet`, and `raw_stats.json` — that `dfdiagnoser input=file` consumes:

```bash
dfanalyzer analyzer/preset=dlio trace_path=tests/data/extracted/dftracer-dlio \
    view_types=[time_range] \
    facts.enabled=true facts.eval_mode=rule \
    facts.eval_rule_file=python/dftracer/analyzer/configs/fact_rules/dlio.yaml \
    output=file output.path=/tmp/bundle
```

```text
[info ] file_output.facts   path=/tmp/bundle/facts.jsonl
$ ls /tmp/bundle
facts.jsonl  detail_view_proc_name.parquet  detail_view_time_range.parquet  raw_stats.json
```

### Full offline chain (analyzer → diagnoser → optimizer)

```bash
# 1. analyze -> fact bundle
dfanalyzer analyzer/preset=dlio trace_path=tests/data/extracted/dftracer-dlio \
    view_types=[time_range] facts.enabled=true output=file output.path=/tmp/bundle

# 2. diagnose -> longitudinal findings
dfdiagnoser input=file input.path=/tmp/bundle output=console

# 3. optimize -> ActionPlans (offline replay of the findings)
python -m dfoptimizer --transport file --findings-file findings.jsonl
```

The temporal axis for longitudinal facts is `time_range` offline (or `window` when
streaming over ZMQ/Mofka). Spatial views (`file_name`/`proc_name`) yield one-shot facts.

### Facts configuration

| key | default | meaning |
|---|---|---|
| `facts.enabled` | `false` | master switch; off = analysis output unchanged |
| `facts.eval_mode` | `rule` | `rule` (YAML conditions) or `metric` (slope) |
| `facts.eval_rule_file` | `""` | rule YAML (when `eval_mode=rule`) |
| `facts.emit_mode` | `aggregate` | `aggregate` (per-view rollup) or `detail` (per-entity) |
| `facts.emit_flat_views` | `true` | also write the detail views into the bundle |

## Further Information

For more details, to report issues, or to contribute to DFAnalyzer, please refer to the following resources:

- **[Official DFAnalyzer Documentation](https://dfanalyzer.readthedocs.io/)**: For detailed usage, configuration options, and information about analyzers.
- **[Issue Tracker](https://github.com/LLNL/dfanalyzer/issues)**: To report bugs or suggest new features.
- **[Contributing Guidelines](./CONTRIBUTING.md)**: For information on how to contribute to the project, including setting up a development environment and coding standards.
- **[Citation File](./CITATION.cff)**: If you use DFAnalyzer in your research, please cite it using the information in this file.

## Acknowledgments

This work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344. This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research under the DOE Early Career Research Program (LLNL-CONF-862440). Also, this research is supported in part by the National Science Foundation (NSF) under Grants OAC-2104013, OAC-2313154, and OAC-2411318.
