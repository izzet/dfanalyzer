"""Option-A attribution with declared families; canonical family slopes stay unchanged.

Service-share attribution is proportional, not causal. Independently reduced
process maxima need not belong to the same rank. The presence/count convention
for instrumented zero durations is inherited from the sealed halo3d rule.
"""
import math
import os

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
    if workload in ('sfno', 'stormer'):
        families.append('delivery')
    return dict(workload=workload, input_families=families, comm=True,
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
        if valid:
            storage, decode = waits.get('storage', 0.0), waits.get('decode', 0.0)
            delivery = waits.get('delivery', 0.0)
            compute = max(compute - comm, 0.0) if declared['compute_includes_comm'] else compute
            storage, decode, delivery, comm, compute = (
                value / n for value in (storage, decode, delivery, comm, compute))
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
