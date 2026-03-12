Think about what are the key aspects for internal evaluations next.
My main thoughts

Algorithmic performance scalability
Accuracy with hybrid in compared traces


Then for each app we want to say what we found common between the traces and selective and what was missed or new. Also highlight trace processing performance with smaller hybrid traces.

THis si the overall need from the paper.

for design, can we have a algorithm which describes the Resolution Matching.
This would be one of the analysis evaluations which show the performance of the matching with different order of events. Basically if we have 10000 events half of each what is the cost of resolving. similarly for 100K and 1M and 10M events. I want throughput as the key metric.

Another evaluation i want to do is critical bottleneck found from traces vs critical bottleneck found from hybrid traces as a measure of accuracy in %.

These two should cover the aspect of algorithm and its effectiveness for hybrid data

So three things

A algorithm that describes at a highlevel (that i will add to paper eventually) the Reconciler. 
Evaluations
1. how the performance of the matching with different order of events. Basically if we have 10000 events half of each what is the cost of resolving. similarly for 100K and 1M and 10M events. I want throughput as the key metric.
2. i want to do is critical bottleneck found from traces vs critical bottleneck found from hybrid traces as a measure of accuracy in %.

DATASET:

/p/lustre5/iopp/rayandrew/dfprofiler/results/unet3d

dft-agg-full is just counter events (using ph="C" inside perfetto trace). all events will be aggregated per 5s. there is several metric such as counting (we call it "dft_cnt" so not conflict with "count" parameter of POSIX/STDIO API), dur_min, dur_max, etc
dft-agg-selective is more selective such that we don't aggregate all events. e.g. in this case the time window is still 5s (same to agg-full) but we only aggregate lseek64, __fxstat64, close and open64 since those events are small but does not contribute too much to full pipeline time
no-dft is running without dftracer so I do not think you need to analyze this. this is just a baseline
dft-normal is tracing mode (the usual dftracer), so you will see AI Logging there

---

Conversation notes to resume later

Current idea discussed:
- Goal is for `ph="C"` aggregated events to eventually produce a signature compatible with DFAnalyzer analysis, but the best reconcile point is likely at HLM level rather than `main_view`.

Raw trace findings from `/p/lustre5/iopp/rayandrew/dfprofiler/results/unet3d`:
- `dft-agg-selective` is mixed trace data: it contains normal `ph="X"` duration events for retained calls and `ph="C"` counter events for aggregated calls.
- `dft-agg-full` is mostly lifecycle `ph="X"` events such as start/end plus POSIX `ph="C"` counter events.
- Example `ph="C"` args include fields such as `dft_cnt`, `dur_sum`, `dur_min`, `dur_max`, `ret_sum`, `ret_min`, `ret_max`, `offset_sum`, `offset_min`, `offset_max`, and still include `fhash` / `hhash`.

Agreed parsing direction:
- In `python/dftracer/analyzer/dftracer.py`, `load_objects_dict()` should explicitly check `ph == "C"` and assign a new `final_dict["type"]` value not currently used by metadata or normal events.
- Those counter/profile rows should be kept separate from normal duration traces in `_handle_metadata()`, likely on analyzer state such as `self._profiles` / `self.counters`.
- Counter rows still need metadata enrichment similar to normal traces, especially `file_name`, `host_name`, and enough fields to derive `proc_name`, `io_cat`, and `time_range`.

Important design correction:
- Do not change `read_trace()` to return `(traces, profiles)` unless necessary.
- Reason: `Analyzer.analyze_trace()` currently assumes a dataframe return, and changing the signature would force plumbing changes through `DFTracerAnalyzer`, `RecorderAnalyzer`, `DarshanAnalyzer`, and Darshan's custom `analyze_trace()` path.
- Lower-friction option is to keep `read_trace()` returning normal traces while storing profiles on analyzer instance state.

Recommended flow for implementation:
1. Parse `ph="C"` records into profile rows with a distinct type.
2. In `_handle_metadata()`, split raw rows into:
   - normal duration traces
   - metadata tables
   - profile/counter rows
3. Normalize profile rows so they use the same timestamp origin and compatible columns:
   - `func_name`
   - `cat`
   - `proc_name`
   - `file_name`
   - `io_cat`
   - `time_range`
   - aggregate metric columns derived from `dft_cnt`, `dur_sum`, `ret_sum`, etc
4. Compute normal HLM from duration traces.
5. Compute a profile HLM from counters.
6. Reconcile `trace_hlm` with `profile_hlm`.
7. Feed reconciled HLM into the existing `_analyze_hlm()` path so the rest of analysis stays unchanged.

Why HLM reconciliation is preferred over `main_view` reconciliation:
- `main_view` is not a single structure; it is built later per layer from HLM.
- Counter data more naturally maps to HLM semantics: `time`, `count`, `size`, `cat`, `func_name`, `io_cat`, plus the view keys.
- Reconciling after `main_view` would require synthesizing already-layered derived metrics, which is more brittle.

Open issues for later:
- `read_stats()` should likely become profile-aware if `total_event_count` needs to reflect `dft_cnt` rather than only row counts.
- Time normalization must use one shared minimum timestamp for both normal traces and counters or `time_range` alignment will drift.
- Bottleneck accuracy evaluation is intentionally deferred for a later discussion.

---

Later design update after code exploration

API direction now agreed:
- `read_trace()` should return a structured `ReadTraceResult`, not a bare dataframe and not a raw tuple.
- `AnalysisResult` should carry `_read_result` internally and expose `result.traces` / `result.profiles`.
- Profiles are intended to become native analyzer input, but we should move step by step and not add extra hooks such as `postread_profiles()` yet.

Counter trace findings that affect design:
- The profile bucket width in these datasets is 5 seconds (`ts` advances in `5_000_000` microsecond increments).
- In `dft-agg-selective`, the `ph="C"` rows are effectively already aggregated at a grain very close to HLM input once metadata is attached.
- Over a selective sample, `(cat, name, pid, tid, ts, fhash, hhash)` had no duplicates, but dropping `cat` / `name` introduced heavy collisions.
- This means the counter rows are not `main_view`-like yet; they are closer to HLM rows because they still carry function/category identity.
- In `dft-agg-full`, many counters are not file-based at all, and some rows still duplicate even when keyed by `(cat, name, pid, tid, ts, fhash, hhash, epoch, step)`, so a DFTracer-side coalescing/canonicalization step is still required there.

Important correction to earlier thinking:
- We should think of selective counter rows as canonical HLM-input-like rows at source bucket granularity, not as raw events and not as `main_view`.
- `main_view` is already one level later than HLM and collapses over fields such as `cat`, `func_name`, and `io_cat`.
- Therefore reconciliation/union should happen at HLM, not at `main_view`.

Implication for `dft-agg-selective` first:
- Start with the same-granularity case first: analysis time granularity = profile bucket granularity = 5s.
- In that case there is no true resolution matching problem yet.
- `dftracer.read_trace()` should standardize `profiles` into a canonical schema that is already close to HLM input:
  - source bucket identity (`ts` or derived `time_range`)
  - `proc_name`
  - nullable `file_name`
  - `cat`
  - `func_name`
  - `io_cat`
  - aggregate metrics (`count`, `time`, `size`, etc) derived from counter fields such as `dft_cnt`, `dur_sum`, and `ret_sum`
- For selective traces at 5s granularity, we should build `profile_hlm` directly from those canonical profile rows and union/reconcile with `trace_hlm`.
- We do not need to synthesize pseudo-events for this first path.

Cases to keep in mind later:
- If analysis granularity equals profile granularity, direct profile-to-HLM mapping is enough.
- If analysis granularity is coarser than profile granularity, simple roll-up is enough.
- If analysis granularity is finer than profile granularity, that is the real resolution-matching problem.

Immediate follow-up notes after walking the merge path:
- Enforce profile-compatible analysis granularity strictly. The current hybrid path is only exact when the analysis granularity is `5s` or an integer multiple of `5s`; non-aligned values such as `6s` or `7s` need real rebucketing/resolution matching and should be rejected for now.
- Add an explicit HLM overlap diagnostic during reconcile. The current reconcile policy is "trace wins on exact HLM-key collisions", so we should log how many exact overlaps exist whenever hybrid HLM is built.
- Deferred on purpose: add a profile-side equivalent of `postread_trace()` filtering. Right now traces go through ignored-file / ignored-function filtering, but profiles do not. Park this until after the current granularity and overlap diagnostics are in place.

Follow-up design note for `dft-agg-full`:
- The current global `trace_hlm` + `profile_hlm` reconcile is acceptable for `dft-agg-selective` because the profile side is effectively just POSIX.
- For `dft-agg-full`, profiles contribute rows to multiple layers (`posix`, `data`, `compute`, `dataloader`, `device`, `comm`), so reconciliation should move into the per-layer `layer_hlm` loop rather than happen once above all layers.
- Separate concern, explicitly deferred for now: if `file_name` is included in `view_types`, fileless profile rows are lost during HLM grouping before the layer loop runs. This is not unique to hybrid profiles, and for now we accept it as a view-selection limitation rather than widening the current implementation scope.
