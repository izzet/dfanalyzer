# Resolution Matching Reconciler: Algorithm Description

## Problem Statement

Hybrid profiling traces combine two distinct event representations:
- **Duration events** (`ph="X"`): individual I/O operations with precise timestamps, durations, and sizes.
- **Counter events** (`ph="C"`): pre-aggregated statistics covering fixed-width time buckets (e.g., 5-second windows), summarizing counts, durations, and sizes for each (function, file, category) combination.

The Reconciler unifies these heterogeneous representations into a single High-Level Metric (HLM) table suitable for downstream bottleneck analysis. The key challenge is that the analysis time granularity may differ from the profile bucket width, requiring resolution matching to align the two sources before merging.

## Definitions

| Symbol | Description |
|--------|-------------|
| $T$ | Set of duration trace events |
| $P$ | Set of counter/profile events |
| $g_a$ | Analysis time granularity (seconds) |
| $g_p$ | Profile bucket width (seconds) |
| $K$ | HLM groupby key: (view_types $\cup$ {cat, io_cat, acc_pat, func_name}) |
| $M$ | HLM measure columns: {count, time, size} |

## Algorithm

### Phase 1: Profile Standardization

Counter events are parsed into a canonical profile schema during trace reading. Each profile row carries:
- Identity columns: `proc_name`, `file_name`, `cat`, `func_name`, `io_cat`
- Temporal columns: `time_range`, `time_start`, `time_end`
- Aggregate measures: `count` (from `dft_cnt`), `time` (from `dur_sum`), `size` (from `ret_sum`)
- Bound statistics: `time_min`, `time_max`, `size_min`, `size_max`

Duplicate canonical keys `(proc_name, time_range, cat, func_name, io_cat, file_name)` are coalesced by summing measures and taking min/max of bounds.

### Phase 2: Resolution Matching (Expansion)

Given analysis granularity $g_a$ and profile bucket width $g_p$:

**Case 1: $g_a = g_p$** — No expansion needed. Profile rows map directly to HLM time ranges.

**Case 2: $g_a > g_p$ (coarser analysis)** — Simple roll-up via standard HLM aggregation. Multiple profile buckets fall into a single analysis bucket.

**Case 3: $g_a < g_p$ (finer analysis)** — Resolution matching is required. Each profile row is expanded into $k = g_p / g_a$ sub-bucket rows.

```
EXPAND-PROFILE-BUCKETS(P, k, g_a, distribution)
  Input:  Profile rows P, expansion factor k, sub-bucket width g_a, distribution strategy
  Output: Expanded profile rows P' with |P'| = k * |P|

  for each row r in P:
      for i = 0 to k-1:
          r_i = copy(r)
          r_i.time_start = r.time_start + i * g_a * time_resolution
          r_i.time_end   = r_i.time_start + g_a * time_resolution
          r_i.time_range = r_i.time_start / (g_a * time_resolution)

          if distribution = "uniform":
              r_i.time  = r.time / k
              r_i.count = floor(r.count / k) + (1 if i < r.count mod k else 0)
              r_i.size  = r.size / k

          else if distribution = "weighted":
              w_i = time_min + (time_max - time_min) * i / (k - 1)
              W   = sum of all w_j for j = 0..k-1
              r_i.time  = r.time * (w_i / W)
              r_i.count = floor(r.count / k) + (1 if i < r.count mod k else 0)
              r_i.size  = r.size / k

          emit r_i

  Invariants:
    sum(r_i.time)  = r.time       (for all sub-buckets of each original row)
    sum(r_i.count) = r.count
    sum(r_i.size)  = r.size
```

The *uniform* strategy distributes measures equally across sub-buckets. The *weighted* strategy uses `time_min` and `time_max` bounds to create a linear ramp, giving proportionally more duration to sub-buckets where per-event latency is expected to be higher.

### Phase 3: HLM Construction

Both trace events and (expanded) profile rows are independently aggregated into HLM tables using the same groupby-aggregate operation:

```
COMPUTE-HLM(events, K)
  Input:  Event rows, groupby key K
  Output: HLM table

  return events.groupby(K).agg(
      time  = sum,
      count = sum,
      size  = sum
  )
```

This produces two HLM tables: `trace_hlm` from duration events $T$, and `profile_hlm` from (expanded) profile events $P'$.

### Phase 4: Per-Layer Reconciliation

Reconciliation is performed independently for each analysis layer (e.g., POSIX, data, compute, dataloader). Each layer defines a filter predicate over `cat` and `func_name`.

```
RECONCILE-HLM(trace_hlm_L, profile_hlm_L, K)
  Input:  Layer-filtered trace HLM, layer-filtered profile HLM, groupby key K
  Output: Reconciled HLM for layer L

  // Step 1: Identify HLM keys present in trace data
  trace_keys = unique K-projections of trace_hlm_L

  // Step 2: Find profile-only rows (no trace coverage)
  for each row r in profile_hlm_L:
      if r[K] in trace_keys:
          mark r as overlapping       // trace data takes precedence
      else:
          mark r as profile-only

  // Step 3: Merge — trace wins on overlap
  combined = trace_hlm_L  UNION  profile_only(profile_hlm_L)

  // Step 4: Re-aggregate (handles any residual key collisions)
  return combined.groupby(K).agg(sum M)
```

The **trace-wins** policy ensures that when both trace and profile data cover the same (function, time range, file, process) combination, the higher-fidelity trace data is used. Profile data fills in coverage for functions/time ranges that were aggregated away in the hybrid trace.

### Reconciliation Properties

1. **Completeness**: Every function observed in either traces or profiles appears in the reconciled HLM.
2. **No double-counting**: For any HLM key present in both sources, only the trace-derived row is retained.
3. **Measure conservation**: For profile-only keys, the sum of measures over all sub-buckets equals the original profile aggregate.
4. **Layer isolation**: Reconciliation operates independently per layer, so layer-specific filter predicates are applied before merging.

## Overall Pipeline

```
HYBRID-ANALYSIS(trace_file)
  // Read
  (T, P, metadata) = read_trace(trace_file)

  // Expand profiles if needed
  if g_a < g_p:
      P' = EXPAND-PROFILE-BUCKETS(P, g_p / g_a, g_a, distribution)
  else:
      P' = P

  // Build HLMs independently
  trace_hlm  = COMPUTE-HLM(T, K)
  profile_hlm = COMPUTE-HLM(P', K)

  // Per-layer reconciliation and analysis
  for each layer L in layer_definitions:
      trace_hlm_L   = filter(trace_hlm, L.predicate)
      profile_hlm_L = filter(profile_hlm, L.predicate)
      hlm_L = RECONCILE-HLM(trace_hlm_L, profile_hlm_L, K)
      main_view_L = aggregate(hlm_L)      // standard analysis path
      views_L = compute_views(main_view_L) // bottleneck detection
```

## Complexity

Let $n_T$ = number of trace events, $n_P$ = number of profile rows, $k$ = expansion factor.

| Phase | Time | Space |
|-------|------|-------|
| Profile expansion | $O(k \cdot n_P)$ | $O(k \cdot n_P)$ |
| HLM construction (trace) | $O(n_T \log n_T)$ | $O(n_T)$ |
| HLM construction (profile) | $O(k \cdot n_P \log(k \cdot n_P))$ | $O(k \cdot n_P)$ |
| Reconciliation (per layer) | $O(\|H_T\| + \|H_P\|)$ | $O(\|H_T\| + \|H_P\|)$ |

where $\|H_T\|$ and $\|H_P\|$ are the number of unique HLM keys in the trace and profile tables respectively (typically much smaller than $n_T$ and $n_P$).


--- 

- put all on slides, and algorithms
- include bandwidth for eval1 
- provide scripts for bottleneck accuracy  
