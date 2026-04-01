Here's the structure and reasoning behind the algorithm description:                                   
                                 
  4 phases, matching the code flow:                                                                      
                                                                                                         
  1. Profile Standardization (Phase 1) — This is what dftracer.py's load_objects_dict() and the          
  canonicalization in read_trace() do. Counter events (ph="C") arrive with raw fields like dft_cnt,      
  dur_sum, ret_sum and get mapped into the same schema as HLM input: count, time, size, plus identity
  columns. Duplicates are coalesced. This is the "parsing" step.
  2. Resolution Matching / Expansion (Phase 2) — This is _expand_profile_buckets and
  _validate_and_expand_profiles. The key insight here is the three cases based on how $g_a$ (analysis
  granularity) relates to $g_p$ (profile bucket width). The interesting case is $g_a < g_p$ (e.g.,
  analyzing at 1s when profiles are 5s buckets), where each profile row is split into $k$ sub-buckets. I
  described both distribution strategies — uniform and weighted (the linear ramp using
  time_min/time_max). The invariants guarantee no measures are lost or created.
  3. HLM Construction (Phase 3) — _compute_high_level_metrics and _compute_profile_hlm. Both sources go
  through the exact same groupby-sum pipeline independently, producing two HLM tables.
  4. Per-Layer Reconciliation (Phase 4) — _reconcile_hlm called from _analyze_hlm's layer loop. The
  trace-wins policy is the core design choice: when both sources have data for the same HLM key (function
   + time range + file + process), we keep the trace-derived row since it's higher fidelity. Profile rows
   only fill gaps.

  Design choices I highlighted:
  - Per-layer reconciliation (not global) — this was a deliberate design decision noted in PLAN.md for
  dft-agg-full support
  - The 4 formal properties (completeness, no double-counting, measure conservation, layer isolation)
  - Complexity section showing reconciliation itself is linear in HLM size, not event count

  Things I'm unsure about for a paper context: The weighted distribution strategy — is the linear ramp
  from time_min to time_max the right framing? And the ~33% error in Eval 2 — should the algorithm
  description acknowledge that selective aggregation intentionally drops coverage, or is that a dataset
  property rather than an algorithm property?

  What are your thoughts? Anything you'd want to change, reframe, or add for the paper?