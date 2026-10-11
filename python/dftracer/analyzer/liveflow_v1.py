"""Option-A attribution with declared families; canonical family slopes stay unchanged.

Service-share attribution is proportional, not causal. Independently reduced
process maxima need not belong to the same rank. The presence/count convention
for instrumented zero durations is inherited from the sealed halo3d rule.
"""
import math
import os
from typing import Dict, List

import pandas as pd
import structlog

logger = structlog.get_logger()


def declaration(workload=None, tf_decode=None):
    workload = workload or os.environ.get('LIVEFLOW_WORKLOAD', '')
    if workload not in ('halo3d', 'stormer', 'dlio-pytorch', 'dlio-tf', 'sfno'):
        raise ValueError('liveflow-v1 requires LIVEFLOW_WORKLOAD=halo3d|stormer|dlio-pytorch|dlio-tf|sfno')
    if tf_decode is None:
        tf_decode = os.environ.get('DFOPTIMIZER_REGISTER_COMPUTATION_THREADS', '0') == '1'
    families = ['storage']
    if workload == 'halo3d' or (workload == 'dlio-tf' and tf_decode):
        families.append('decode')
    # DELIVERY: the worker -> trainer sample handoff (liveflow_delivery.py) is a declared input family
    # for the two PyTorch DataLoader workloads whose samples are large. Its wait is measured exactly
    # in the trainer's main thread (stall.delivery), so the storage/delivery split is a measurement,
    # not an estimate. Declared, never inferred from a missing column: an unmeasured declared family
    # leaves the window invalid and nothing is attributed.
    # Cross-layer ablation (LIVEFLOW_SFNO_APP_ONLY=1): an application-level-only detector declares ONE input
    # family, storage; the trainer then emits its whole fetch wait as stall.storage (the DLIO-style undivided
    # next(loader) total) and the delivery family is not declared.
    app_only = workload == 'sfno' and os.environ.get('LIVEFLOW_SFNO_APP_ONLY', '0') == '1'
    if workload in ('sfno', 'stormer') and not app_only:
        families.append('delivery')
    # JOB-LEVEL LAYER (LIVEFLOW_JOB_LEVEL=1, DLIO job input sync): barrier wait that a peer's input stall explains
    # is attributed to the input families.  Off in the per-node ablation, which still receives the same job
    # columns (provenance) but never moves time between families.
    job_level = os.environ.get('LIVEFLOW_JOB_LEVEL', '0') == '1'
    return dict(workload=workload, input_families=families, comm=True, job_level=job_level,
                split_wait=len(families) == 2,
                wait_per_step=workload in ('halo3d', 'stormer'),
                compute_includes_comm=workload != 'halo3d')


# Service-time layer used to apportion the fetch wait only when the direct exposure split is absent.
_SERVICE_LAYER = {'storage': 'reader_posix_read', 'decode': 'reader_decode', 'delivery': 'delivery'}


# Layers that carry a family's service per workload preset (halo3d's storage layer is `reader_posix`, DLIO's `reader_posix_read`).
_SERVICE_LAYERS = {'storage': ('reader_posix_read', 'reader_posix'), 'decode': ('reader_decode',), 'delivery': ('delivery',)}


def _finite(row, name):
    try:
        value = float(row[name])
    except (KeyError, ValueError, TypeError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _measured(row, family):
    # fastpath deliberately encodes a measured exact-zero duration as NaN.
    # A positive event count distinguishes that case from an absent instrument.
    if family + '_time_proc_max' not in row.index:
        return None
    value = _finite(row, family + '_time_proc_max')
    count = _finite(row, family + '_count_sum')
    if value is None and count is not None and count > 0:
        return 0.0
    return value


# Window-level job columns (metrics.attach_input_job_step_max_sum_metrics): totals in seconds.
JOB_CRITICAL = 'domain_input_job_step_max_sum_time'
JOB_OWN = 'domain_input_job_own_mean_time'
JOB_VALUE_KEYS = ('lf_v1_job_available', 'lf_v1_job_level', 'lf_v1_job_critical_input', 'lf_v1_job_own_input',
                  'lf_v1_job_peer_skew', 'lf_v1_job_comm_wait_before', 'lf_v1_job_attributed')


def job_attribution(critical, own, comm, shares):
    """Peer-skew attribution of barrier wait to the input families (all window totals, seconds).

        S = max(critical_input - own_input, 0)   input wait a typical rank paid because a PEER stalled
        W = comm                                 barrier/comm wait this node measured
        A = min(S, W)                            the part of W that peer input explains

    No constant: S and W are differences/values of measurements from the same window and A is their
    minimum, so the attribution never exceeds either measurement (a wait that is larger than the skew
    keeps its remainder as comm; a skew larger than the wait cannot create wait that was not paid).
    Returns None when the measurement is unavailable or the family split is unknown: the window is then
    left exactly as the per-node view reads it (suppressed, never invented).
    """
    if critical is None or own is None or comm is None or not shares:
        return None
    skew = max(critical - own, 0.0)
    attributed = min(skew, max(comm, 0.0))
    return dict(skew=skew, attributed=attributed, split={f: attributed * x for f, x in shares.items()})


def _service_shares(row, families, waits):
    """Split of an attributed input wait across the declared families: the existing service-share rule.

    One family takes all of it.  With two, service time of each family's layer (the same quantity that
    apportions the undivided fetch wait); if that is absent the window's own measured waits.  Unknown
    split -> None (no attribution).
    """
    if len(families) == 1:
        return {families[0]: 1.0}
    service = [_measured(row, _SERVICE_LAYER[f]) for f in families]
    if all(v is not None for v in service) and sum(service) > 0:
        return {f: v / sum(service) for f, v in zip(families, service)}
    total = sum(waits.get(f, 0.0) for f in families)
    if total > 0:
        return {f: waits.get(f, 0.0) / total for f in families}
    return None


# Per-stage SERVICE used by the Little's-law sizing (dfoptimizer/planner/sizing.py): the busy seconds one rank's stage spends per
# training step (sum of the layer's operation durations over all threads of the rank, divided by the window's steps).  Per-step
# columns, forwarded to the planner like the job-level provenance.  `_critical` is 1.0 when the value is the maximum over the
# node's rank processes (rank membership from the window's rank pids), 0.0 when it is the node total divided by the rank count
# (forked worker processes cannot be mapped to a rank from their names).
SERVICE_VALUE_KEYS = tuple(k for f in ('storage', 'decode', 'delivery')
                           for k in ('lf_v1_svc_' + f, 'lf_v1_svc_' + f + '_critical'))

# FactEngine.evaluate invokes prepare_views before evaluating declared-family
# rules. These columns therefore belong to its window interface even though
# they are not raw aggregates. Invalid/missing measurements still fail closed
# in prepare_views; advertising this interface must not invent measurements.
PREPARED_WINDOW_METRICS = frozenset((
    'lf_v1_valid', 'lf_v1_storage_declared', 'lf_v1_decode_declared',
    'lf_v1_delivery_declared', 'lf_v1_comm_declared',
    'lf_v1_storage_wait', 'lf_v1_decode_wait', 'lf_v1_delivery_wait',
    'lf_v1_comm_wait', 'lf_v1_compute_time', 'lf_v1_active_time',
    'lf_v1_per_step', 'xlayer_source_is_instrument', 'xlayer_storage_share',
    'xlayer_decode_share', 'xlayer_delivery_share', 'observation_only',
    'liveflow_v1_attribution',
    'reader_posix_read_ops_slope', 'reader_decode_ops_slope',
    'delivery_ops_slope', 'comm_ops_slope',
)) | frozenset(JOB_VALUE_KEYS) | frozenset(SERVICE_VALUE_KEYS)


def _proc_pid(name):
    parts = str(name).split('#')
    if len(parts) >= 3 and parts[0] == 'app':
        try:
            return int(parts[2])
        except ValueError:
            return None
    return None


def _stage_service(row, proc_rows, layer, n, n_ranks, rank_pids):
    """(busy seconds per step of the critical rank's stage, critical flag) or None when the layer was not measured."""
    col = layer + '_time_sum'
    total = _finite(row, col)
    count = _finite(row, layer + '_count_sum')
    if total is None and count is not None and count > 0:
        total = 0.0          # a measured exact-zero duration (fastpath encodes it as NaN), as in _measured
    if total is None or n is None or n < 1:
        return None
    if proc_rows is not None and rank_pids and col in proc_rows.columns:
        names = (proc_rows.index.get_level_values('proc_name') if isinstance(proc_rows.index, pd.MultiIndex)
                 and 'proc_name' in proc_rows.index.names else proc_rows.index)
        times = pd.to_numeric(proc_rows[col], errors='coerce').fillna(0.0).to_numpy()
        by_rank, off_rank = {}, 0.0
        for name, t in zip(names, times):
            pid = _proc_pid(name)
            if pid in rank_pids:
                by_rank[pid] = by_rank.get(pid, 0.0) + float(t)
            else:
                off_rank += float(t)
        if by_rank and off_rank <= 0.0 and max(by_rank.values()) > 0.0:
            return max(by_rank.values()) / n, 1.0
    if not n_ranks:
        return None
    return total / n_ranks / n, 0.0


def window_trace_integrity(traces, rank_pids, step_evidence, dropped_records=None, *, global_steps=True):
    """Check every control rank against the application's completed step range.

    Cumulative SHM loss cannot be assigned to an exact window. Once any loss
    occurs, fail closed for the rest of this stream, including worker-only loss.
    The series header counts steps even when its timing arrays are binned.
    """
    import re
    from .metrics import authoritative_control_rows, _thread_ids

    result = {"lf_v1_trace_complete": 0.0}
    if dropped_records is not None:
        result["lf_v1_trace_dropped"] = float(dropped_records)
        if dropped_records != 0:
            return result
    match = re.match(r"^v1;first=(\d+);n=(\d+);b=(\d+);", str(step_evidence.get("step_series", "")))
    if not match or not rank_pids or traces is None:
        return result
    first, count, bins = map(int, match.groups())
    if count < 1 or bins < 1 or not {"pid", "step"}.issubset(traces.columns):
        return result
    end = step_evidence.get("completed_steps")
    if end is not None and int(end) != first + count - 1:
        return result
    expected = set(range(first, first + count))
    mask = authoritative_control_rows(traces, rank_pids)
    tids = _thread_ids(traces)
    if tids is not None:
        mask &= tids == pd.to_numeric(traces["pid"], errors="coerce")
    rows = traces.loc[mask].copy()
    rows["pid"] = pd.to_numeric(rows["pid"], errors="coerce")
    rows["step"] = pd.to_numeric(rows["step"], errors="coerce")
    # Boundary/counter events are not per-step workload observations.
    if "cat" in rows:
        rows = rows.loc[~rows["cat"].isin(("window", "control"))]
    local_range = None
    for pid in rank_pids:
        observed = set(rows.loc[rows["pid"] == int(pid), "step"].dropna())
        if global_steps:
            if not expected.issubset(observed):
                return result
        else:
            # Training presets stamp epoch-local steps; the control series is
            # cumulative. Count and contiguous coverage must agree on EVERY
            # rank, without guessing an offset from a surviving fragment.
            if len(observed) != count or max(observed) - min(observed) + 1 != count:
                return result
            bounds = (min(observed), max(observed))
            if local_range is not None and bounds != local_range:
                return result
            local_range = bounds
    result["lf_v1_trace_complete"] = 1.0
    return result


def prepare_views(flat_views, declared, context=None):
    """Adapt real analyzer views to the shared rule's seconds-per-step interface.

    Undeclared families are absent by construction. A missing metric needed for
    a declared family leaves an invalid window; no attributed finding is emitted.
    """
    view = flat_views.get(('window',))
    if view is None:
        return flat_views
    frame = view.copy()
    proc = flat_views.get(('proc_name',))
    for index, row in view.iterrows():
        families = list(declared['input_families'])
        values = {'lf_v1_valid': 0.0, 'lf_v1_storage_declared': float('storage' in families),
                  'lf_v1_decode_declared': float('decode' in families),
                  'lf_v1_delivery_declared': float('delivery' in families),
                  'lf_v1_comm_declared': float(declared['comm'])}
        x = _measured(row, 'fetch_iter')
        comm = _measured(row, 'comm') if declared['comm'] else 0.0
        compute = _measured(row, 'compute')
        n = _finite(row, 'step_nunique')
        source = None
        waits = {}
        if len(families) == 1:
            # DLIO's stall.storage is an undivided next(loader) window total.
            # StorMer's native account uses the shared per-step-average emitter.
            measured = _measured(row, 'stall_storage')
            if measured is not None:
                x = measured * n if declared['wait_per_step'] and n is not None else measured
                source = 1.0
            else:
                source = 0.0
            if x is not None:
                waits[families[0]] = x
        else:
            counts = [_finite(row, 'stall_' + family + '_count_sum') for family in families]
            direct = declared['split_wait'] and all(c is not None and c > 0 for c in counts)
            if direct and n is not None:
                raw = [_measured(row, 'stall_' + family) for family in families]
                if all(v is not None for v in raw):
                    scale = n if declared['wait_per_step'] else 1.0
                    waits = dict(zip(families, (v * scale for v in raw)))
                    source = 1.0
            elif x is not None:
                service = [_measured(row, _SERVICE_LAYER[f]) for f in families]
                if all(v is not None for v in service) and sum(service) > 0:
                    waits = dict(zip(families, (x * v / sum(service) for v in service)))
                    source = 0.0
        valid = (all(f in waits for f in families) and comm is not None and compute is not None
                 and n is not None and n >= 1)
        integrity = (context or {}).get('trace_integrity', {})
        if integrity or (context or {}).get('native_window') is not None:
            valid = valid and integrity.get('lf_v1_trace_complete') == 1.0
        values.update(integrity)
        job = {k: 0.0 for k in JOB_VALUE_KEYS}
        job['lf_v1_job_level'] = float(bool(declared.get('job_level')))
        if valid:
            critical, own = _finite(row, JOB_CRITICAL), _finite(row, JOB_OWN)
            if critical is not None and own is not None:
                job.update(lf_v1_job_available=1.0, lf_v1_job_critical_input=critical,
                           lf_v1_job_own_input=own, lf_v1_job_peer_skew=max(critical - own, 0.0),
                           lf_v1_job_comm_wait_before=comm)
                if declared.get('job_level'):
                    got = job_attribution(critical, own, comm, _service_shares(row, families, waits))
                    if got is not None:
                        for family, moved in got['split'].items():
                            waits[family] = waits.get(family, 0.0) + moved
                        comm -= got['attributed']
                        job['lf_v1_job_attributed'] = got['attributed']
        svc = {k: float('nan') for k in SERVICE_VALUE_KEYS}
        if valid:
            peers_for_svc = None
            if proc is not None:
                if isinstance(proc.index, pd.MultiIndex) and 'window' in proc.index.names:
                    peers_for_svc = proc.loc[proc.index.get_level_values('window') == index]
                elif len(view) == 1:
                    peers_for_svc = proc
            rank_pids = {int(p) for p in ((context or {}).get('rank_pids') or ())}
            fcount = _finite(row, 'fetch_iter_count_sum')
            n_ranks = len(rank_pids) or (int(round(fcount / n)) if fcount and n else None)
            for family in families:
                got = None
                for layer in _SERVICE_LAYERS[family]:      # first layer this workload's preset measures
                    got = _stage_service(row, peers_for_svc, layer, n, n_ranks, rank_pids)
                    if got is not None:
                        break
                if got is not None:
                    svc['lf_v1_svc_' + family], svc['lf_v1_svc_' + family + '_critical'] = got
        if valid:
            storage, decode = waits.get('storage', 0.0), waits.get('decode', 0.0)
            delivery = waits.get('delivery', 0.0)
            # compute excludes the comm the node MEASURED; moving a peer-explained part of that wait to input
            # does not make it compute.
            compute = (max(compute - job['lf_v1_job_comm_wait_before'] if job['lf_v1_job_available'] else compute - comm, 0.0)
                       if declared['compute_includes_comm'] else compute)
            storage, decode, delivery, comm, compute = (
                value / n for value in (storage, decode, delivery, comm, compute))
            for key in ('lf_v1_job_critical_input', 'lf_v1_job_own_input', 'lf_v1_job_peer_skew',
                        'lf_v1_job_comm_wait_before', 'lf_v1_job_attributed'):
                job[key] = job[key] / n
            total = storage + decode + delivery + comm + compute
            exposed = storage + decode + delivery
            values.update(lf_v1_valid=1.0, lf_v1_storage_wait=storage, lf_v1_decode_wait=decode,
                          lf_v1_delivery_wait=delivery,
                          lf_v1_comm_wait=comm, lf_v1_compute_time=compute,
                          lf_v1_active_time=total, xlayer_source_is_instrument=source,
                          xlayer_storage_share=storage / exposed if exposed else 0.0,
                          xlayer_decode_share=decode / exposed if exposed else 0.0,
                          xlayer_delivery_share=delivery / exposed if exposed else 0.0,
                          observation_only=float(exposed + comm == 0.0),
                          liveflow_v1_attribution=1.0, lf_v1_per_step=1.0)
        else:
            logger.info('facts.liveflow_v1.missing_declared_metrics', window=str(index),
                        workload=declared['workload'], input_families=families)
        values.update(job)
        values.update(svc)
        # Detector 1 is reduced within ONE proc_name operation family. No
        # comparison or division between raw slopes from different families.
        for prefix in ('reader_posix_read', 'reader_decode', 'delivery', 'comm'):
            key = prefix + '_ops_slope'
            slope = float('nan')
            if proc is not None and key in proc.columns:
                peers = proc
                if isinstance(proc.index, pd.MultiIndex) and 'window' in proc.index.names:
                    peers = proc.loc[proc.index.get_level_values('window') == index]
                elif len(view) != 1:
                    peers = proc.iloc[:0]  # No guessed window join for a multi-window view.
                numeric = pd.to_numeric(peers[key], errors='coerce')
                good = numeric[numeric.map(lambda v: pd.notna(v) and math.isfinite(v) and v > 0)]
                if len(good):
                    slope = float(good.min())
            values[key] = slope
        for key, value in values.items():
            frame.loc[index, key] = value
    return {**flat_views, ('window',): frame}


# Per-step evidence of one window (liveflow-v1): the string series the application's Window put on its
# window.stop (dfoptimizer/step_series.py) and the node's memory-pressure counters at the window close.
STEP_EVIDENCE_COUNTERS = ("vm_allocstall", "vm_pgscan_direct", "vm_compact_stall", "mem_anon_kb")
STEP_EVIDENCE_FIELDS = ("step_series", *STEP_EVIDENCE_COUNTERS, "mem_avail_kb")


def _series_wall_total(text) -> float:
    """Sum of the `wall=` series of a v1 step_series string (10 microsecond units); -1 if unparseable."""
    try:
        for part in str(text).split(";"):
            if part.startswith("wall="):
                return float(sum(int(v) for v in part[5:].split(",")))
    except ValueError:
        pass
    return -1.0


def window_step_evidence(candidates: List[dict]) -> Dict[str, object]:
    """Per-step series and node counters for the window, from the window.stop records of its ranks.

    The series is the CRITICAL rank's (the one with the largest total step wall in the window), so the
    wall and the per-family exposures it carries are one coherent vector. Node counters are node-wide
    and identical across the node's ranks up to read skew: the largest is taken (cumulative counters and
    anonymous memory), the smallest for available memory (the conservative one).
    """
    out: Dict[str, object] = {}
    series = [(_series_wall_total(c["step_series"]), c["step_series"]) for c in candidates
              if isinstance(c.get("step_series"), str)]
    series = [t for t in series if t[0] >= 0.0]
    if series:
        out["step_series"] = max(series, key=lambda t: t[0])[1]
    for name in STEP_EVIDENCE_COUNTERS + ("mem_avail_kb",):
        vals = []
        for c in candidates:
            try:
                vals.append(int(c[name]))
            except (KeyError, TypeError, ValueError):
                continue
        if vals:
            out[name] = min(vals) if name == "mem_avail_kb" else max(vals)
    return out
