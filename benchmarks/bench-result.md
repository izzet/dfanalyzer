# Benchmark Results

Run date: 2026-03-21
Platform: Linux 4.18.0, Python 3.11.5
All throughput results: pandas-only, 1 warmup + 5 measured runs (mean +/- std)

## Eval 1: Throughput

### Profile Expansion

Pure pandas `_expand_profile_buckets` at 5x expansion (5s buckets -> 1s buckets).

| Scale | Input | Output | Time | Peak Mem | Throughput |
|-------|-------|--------|------|----------|------------|
| 1K | 1,000 rows | 5,000 rows | 0.006s +/- 0.001s | 2.3 MB | 179,494 rows/s |
| 10K | 10,000 rows | 50,000 rows | 0.014s +/- 0.002s | 22.5 MB | 702,405 rows/s |
| 100K | 100,000 rows | 500,000 rows | 0.124s +/- 0.047s | 224.2 MB | 804,188 rows/s |
| 1M | 1,000,000 rows | 5,000,000 rows | 2.998s +/- 0.195s | 2,242.1 MB | 333,520 rows/s |

### Resolution Matching (expand + reconcile)

Pure pandas: expand profiles 5x then reconcile with trace HLM (dedup, merge, groupby/agg).

| Total Events | Profiles | Expanded | Traces | Time | Peak Mem | Throughput |
|--------------|----------|----------|--------|------|----------|------------|
| 10K | 5,000 | 25,000 | 5,000 | 0.033s +/- 0.001s | 13.2 MB | 301,945 events/s |
| 100K | 50,000 | 250,000 | 50,000 | 0.220s +/- 0.004s | 128.2 MB | 454,213 events/s |
| 1M | 500,000 | 2,500,000 | 500,000 | 2.298s +/- 0.063s | 1,231.7 MB | 435,164 events/s |

## Eval 2: Bottleneck Accuracy

Dataset: UNet3D DLIO traces (worker files only, non-worker processes filtered out)
Comparison: dft-normal (full traces, ground truth) vs dft-agg-selective (hybrid: traces + counter profiles)
Distribution: uniform (g_a = g_p = 5s, no expansion needed)

### Bottleneck Detection Summary

| Worker Files | Normal Events | Hybrid Events | Bottlenecks (N) | Bottlenecks (H) | Shared | Recall | Precision | Exact Agr. | Within-1 Agr. | MAE |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 5,902,935 | 5,933,545 | 20,870 | 20,861 | 20,855 | 99.9% | 100.0% | 72.5% | 97.7% | 0.289 |
| 8 | 11,800,304 | 11,887,876 | 20,876 | 20,876 | 20,876 | 100.0% | 100.0% | 74.8% | 98.3% | 0.260 |
| 16 | 23,754,949 | 23,710,120 | 20,926 | 20,926 | 20,926 | 100.0% | 100.0% | 76.4% | 97.9% | 0.251 |

Hybrid analysis achieves **100% bottleneck recall and precision** at 8+ worker files. Event counts between normal and hybrid match closely (<0.2% difference), confirming that `dft_cnt`-based profile accounting is accurate. Exact severity agreement improves with scale (72.5% -> 76.4%), while within-1 agreement stays above 97% across all scales. MAE decreases from 0.289 to 0.251 with more data.

### Per-Metric Severity Agreement (16 worker files)

Severity slopes (0-3 scale) compared per time-range bucket across 837 time windows.

| Metric Category | Exact Agreement | Within-1 Agreement | MAE | Spearman rho |
|-----------------|----------------:|-------------------:|----:|-------------:|
| Intensity (read, data, posix) | 100.0% | 100.0% | 0.00 | -- |
| Data/read ops slope | 83.6% | 97.3% | 0.19 | 0.73 |
| General ops slope (posix, reader_posix) | 84.6% | 97.7% | 0.18 | 0.74 |
| Stat ops slope | 84.9% | 98.7% | 0.17 | 0.66 |
| Metadata ops slope | 58.2% | 97.8% | 0.44 | 0.12 |
| Seek ops slope | 54.7% | 99.5% | 0.46 | 0.10 |
| Open ops slope | 55.8% | 96.7% | 0.48 | 0.02 |
| Close ops slope | 73.8% | 91.0% | 0.38 | 0.07 |
| Dataloader item ops slope | 52.1% | 99.2% | 0.49 | 0.06 |
| Dataloader ops slope | 50.7% | 98.7% | 0.51 | 0.03 |

## Running the benchmarks

```bash
# Expansion + resolution matching (pure pandas, ~30s)
pytest benchmarks/eval_throughput.py -v -s -k "expansion or resolution_matching"

# Full pipeline (synthetic, ~20 min)
pytest benchmarks/eval_throughput.py -v -s -k "synthetic_pipeline"

# Real data with N files (default 4)
BENCH_N_FILES=4 pytest benchmarks/eval_throughput.py -v -s -k "real_data"

# Accuracy at multiple scales (default 4,8,16)
BENCH_SCALES="4,8,16" pytest benchmarks/eval_accuracy.py -v -s -k "bottleneck_accuracy"

# Full dataset (all worker files)
BENCH_N_FILES=0 pytest benchmarks/ -v -s

# Control number of measured runs (default 5)
BENCH_RUNS=3 pytest benchmarks/eval_throughput.py -v -s
```
