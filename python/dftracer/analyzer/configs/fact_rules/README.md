# Fact rule sets: which are threshold-free, which are not, and why

Standing directive (2026-08-31): **no hand-set domain thresholds in fact rules.** Diagnosis emits
the raw per-domain times and the planner decides. SC pap685 reviewers R1/R3/R4 all attacked the
thresholds; the rebuttal defended them as "conservative critical-path criteria, not tuned values",
backed it with an offline sensitivity analysis, conceded that learning them was future work, and
the scores did not move.

## Threshold-free (use these)

| file | workload / boundary | domains |
|---|---|---|
| `dlio-signal.yaml` | DLIO, cross-layer | input (reader_posix), compute |
| `dlio-signal-app.yaml` | DLIO, app-only boundary | fetch, compute |
| `stormer-signal.yaml` | StorMer, cross-layer | input, decode, comm, compute |
| `halo3d.yaml` | halo3d three-way | storage, decode, comm |

All four have the same shape: one rule, `emit_mode: window`, constant severity, and a `when` of
`active_time > 0`, which is a did-a-window-happen guard and not a domain gate.

**They require `DFOPTIMIZER_ALGO=clbo` or `cbandit`.** A rule that fires every window is only safe
with a planner that can decide from raw metrics and can choose to do nothing. The default rule
`Planner` acts on every finding it receives, so pairing it with an unconditional finding actuates
constantly.

## Deliberately gated — DO NOT convert these

| file | why it must keep its thresholds |
|---|---|
| `halo3d_gated.yaml` | The **ablation control** for `halo3d.yaml`. It carries the stormer.yaml values on purpose, to show magic numbers transplanted across workloads. Converting it deletes the experiment that tests our own "no magic numbers" claim. |
| `posix.yaml`, `dlio-posix-corona.yaml`, `dlio-posix-lustre3.yaml` | Single-layer arms. See below: a POSIX-only observer **cannot** be threshold-free. |
| `dlio.yaml`, `dlio-all.yaml`, `dlio-app.yaml`, `stormer.yaml` | The pre-directive rule sets, retained so old runs stay reproducible and so the gated-vs-free comparison has a baseline. New work should not use them. |

## Why a single-layer observer cannot be threshold-free

This falls out of the formulation rather than being a limitation of any implementation, and it is
worth stating in the paper because it turns the reviewers' threshold objection into an argument
*for* cross-layer observation.

Threshold-free decision-making replaces "is this metric above a constant" with "which domain holds
the most time". That is a **comparison**, and a comparison needs at least two things to compare.
Checked directly against `cbandit._finalize_window`:

    POSIX-only metrics -> {'reader_posix_time_proc_max': 4.0}
      active_time available? False          <- no app-level wall exists at this boundary
      domains with any time: {'input': 4.0} <- exactly one
      unattributed: 0.0

With one domain the argmax is degenerate: the observer always picks `input` and therefore always
acts. To decide *whether* to act at all it has to ask "is 4.0 s a lot", which is exactly the
question a threshold answers and a comparison cannot.

So the single-layer arms keep their gates, and that is the honest result: **thresholds are what a
single-layer observer needs in order to substitute for the comparison it cannot make.**

## What is still a constant

The planner carries `alpha` (significance) and `effect_floor` (minimum interesting effect) in
`clbo.accept_decision`. Those are standard statistical parameters whose meaning transfers across
workloads. "600 MB/s on Corona" does not. Replacing a magic number with a decision rule is the
goal; claiming there are no constants would be false.
