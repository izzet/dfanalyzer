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


def prepare_views(flat_views, declared):
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
