import dataclasses as dc
from typing import Dict, List, Optional, Set

import structlog


logger = structlog.get_logger()


class WindowBuffer:
    """
    Buffer events by window using configurable start/end markers.

    Each emitted event is tagged with ``window`` (the analysis window number =
    the count of start markers seen for that pid) -- the longitudinal axis. It
    does NOT overwrite ``epoch``/``step`` with the window number: those are
    workload attributes, not the window. See docs/window-as-longitudinal-axis.md.
    """

    def __init__(
        self,
        window_start_name: str = "epoch.start",
        window_end_name: str = "epoch.block",
        process_key: str = "pid",
        *,
        epoch_start_name: Optional[str] = None,
        epoch_end_name: Optional[str] = None,
    ):
        if epoch_start_name is not None:
            window_start_name = epoch_start_name
        if epoch_end_name is not None:
            window_end_name = epoch_end_name
        self.window_start_name = window_start_name
        self.window_end_name = window_end_name
        self.process_key = process_key
        self.window_numbers: Dict[int, int] = {}
        self.seen_pids_by_window: Dict[int, set] = {}
        self.ended_pids_by_window: Dict[int, set] = {}
        self.buffer_by_window: Dict[int, List[dict]] = {}

    def push(self, event: dict) -> Optional[List[dict]]:
        if event.get("name") == self.window_start_name:
            pid_start = event.get(self.process_key)
            if pid_start is None:
                logger.error("window.start.no_pid")
                raise ValueError("Missing pid in window start event")
            self.window_numbers[pid_start] = self.window_numbers.get(pid_start, 0) + 1
            window_for_pid = self.window_numbers[pid_start]
            self.seen_pids_by_window.setdefault(window_for_pid, set()).add(pid_start)
            logger.info("window.start", pid=pid_start, window=window_for_pid)

        pid = event.get(self.process_key)
        if pid is None:
            logger.error("event.no_pid", name=event.get("name"))
            raise ValueError(f"Missing pid on event: {event.get('name')}")

        window_for_pid = self.window_numbers.get(pid, 0)
        if window_for_pid > 0:
            # `window` is the analysis window (the longitudinal axis). epoch/step
            # are workload attributes, NOT the window: we don't blindly copy the
            # window number into them. They keep whatever real value the event
            # already carries (or stay absent); the window count is not an epoch
            # or a step. See docs/window-as-longitudinal-axis.md.
            event["window"] = window_for_pid
            self.seen_pids_by_window.setdefault(window_for_pid, set()).add(pid)
            self.buffer_by_window.setdefault(window_for_pid, []).append(event)

        if event.get("name") == self.window_end_name:
            candidate_window = None
            for window in sorted(self.seen_pids_by_window.keys()):
                if pid in self.seen_pids_by_window.get(window, set()):
                    ended_set = self.ended_pids_by_window.get(window, set())
                    if pid not in ended_set:
                        candidate_window = window
                        break
            if candidate_window is None:
                logger.error("window.end.no_seen_window", pid=pid)
                raise ValueError(f"No seen window for pid {pid}")

            self.ended_pids_by_window.setdefault(candidate_window, set()).add(pid)
            logger.debug("window.end.from_pid", pid=pid, window=candidate_window)
            seen = self.seen_pids_by_window.get(candidate_window, set())
            ended = self.ended_pids_by_window.get(candidate_window, set())
            if seen and ended >= seen:
                data_to_emit = self.buffer_by_window.get(candidate_window, [])
                self.buffer_by_window.pop(candidate_window, None)
                return data_to_emit
        return None


EpochBuffer = WindowBuffer


@dc.dataclass
class CompletedWindowBoundary:
    window_index: int
    boundary_ts_ns: int
    ranks_received: int
    events_written_by_pid: Dict[int, int] = dc.field(default_factory=dict)
    start_events_written_by_pid: Dict[int, int] = dc.field(default_factory=dict)


@dc.dataclass
class _PendingWindowBoundary:
    window_index: int
    boundary_ts_ns: int = 0
    ranks_seen: Set[int] = dc.field(default_factory=set)
    events_written_by_pid: Dict[int, int] = dc.field(default_factory=dict)
    start_events_written_by_pid: Dict[int, int] = dc.field(default_factory=dict)


class WindowTracker:
    """
    Track control boundaries by per-rank window sequence.

    This prevents fast ranks from having window N+1 collapsed into the same
    analyzer window as slower ranks still finishing window N.
    """

    def __init__(self, num_ranks: int, require_explicit_start: bool = False):
        self.num_ranks = num_ranks
        self.require_explicit_start = require_explicit_start
        self.active_window_by_pid: Dict[int, int] = {}
        self.completed_window_by_pid: Dict[int, int] = {}
        self.pending_by_window: Dict[int, _PendingWindowBoundary] = {}
        self.next_window_to_emit = 1

    def current_window(self, pid: int) -> Optional[int]:
        active_window = self.active_window_by_pid.get(pid)
        if active_window is not None:
            return active_window
        if self.require_explicit_start:
            return None
        return self.completed_window_by_pid.get(pid, 0) + 1 or 1

    def next_window(self, pid: int) -> int:
        completed_window = self.completed_window_by_pid.get(pid, 0)
        active_window = self.active_window_by_pid.get(pid)
        if active_window is None:
            return completed_window + 1 if completed_window > 0 else 1
        if active_window <= completed_window:
            return completed_window + 1
        return active_window

    def observe_start_boundary(self, pid: int, events_written: Optional[int] = None) -> int:
        next_window = self.completed_window_by_pid.get(pid, 0) + 1
        active_window = self.active_window_by_pid.get(pid)
        if active_window is None or active_window < next_window:
            self.active_window_by_pid[pid] = next_window
        elif active_window > next_window:
            logger.warning(
                "window.start.out_of_order",
                pid=pid,
                active_window=active_window,
                next_window=next_window,
            )
        else:
            logger.debug("window.start.duplicate", pid=pid, window=active_window)
        window_index = self.active_window_by_pid[pid]
        pending = self.pending_by_window.setdefault(
            window_index,
            _PendingWindowBoundary(window_index=window_index),
        )
        if events_written is not None and events_written > 0:
            current_start = pending.start_events_written_by_pid.get(pid)
            if current_start is None:
                pending.start_events_written_by_pid[pid] = events_written
        return window_index

    def observe_end_boundary(
        self,
        pid: int,
        boundary_ts_ns: int,
        events_written: Optional[int] = None,
    ) -> List[CompletedWindowBoundary]:
        window_index = self.active_window_by_pid.get(pid)
        if window_index is None:
            if self.require_explicit_start:
                logger.warning(
                    "window.boundary.without_start",
                    pid=pid,
                    boundary_ts_ns=boundary_ts_ns,
                )
                return []
            window_index = self.completed_window_by_pid.get(pid, 0) + 1
            self.active_window_by_pid[pid] = window_index

        completed_window = self.completed_window_by_pid.get(pid, 0)
        if window_index <= completed_window:
            logger.warning(
                "window.boundary.out_of_order",
                pid=pid,
                window=window_index,
                completed_window=completed_window,
            )
            return []

        self.completed_window_by_pid[pid] = window_index
        # An end boundary closes the active window for this pid. Clearing the
        # active assignment allows end-only control streams to advance to the
        # next window even when no explicit start boundary is emitted.
        self.active_window_by_pid.pop(pid, None)

        pending = self.pending_by_window.setdefault(
            window_index,
            _PendingWindowBoundary(window_index=window_index),
        )
        if pid in pending.ranks_seen:
            logger.warning("window.boundary.duplicate_pid", pid=pid, window=window_index)
        else:
            pending.ranks_seen.add(pid)
        if boundary_ts_ns > pending.boundary_ts_ns:
            pending.boundary_ts_ns = boundary_ts_ns
        if events_written is not None and events_written > 0:
            pending.events_written_by_pid[pid] = events_written

        completed: List[CompletedWindowBoundary] = []
        while True:
            next_pending = self.pending_by_window.get(self.next_window_to_emit)
            if next_pending is None or len(next_pending.ranks_seen) < self.num_ranks:
                break
            completed.append(
                CompletedWindowBoundary(
                    window_index=next_pending.window_index,
                    boundary_ts_ns=next_pending.boundary_ts_ns,
                    ranks_received=len(next_pending.ranks_seen),
                    events_written_by_pid=dict(next_pending.events_written_by_pid),
                    start_events_written_by_pid=dict(next_pending.start_events_written_by_pid),
                )
            )
            self.pending_by_window.pop(self.next_window_to_emit, None)
            self.next_window_to_emit += 1
        return completed


# Backward compatibility alias
WindowBoundaryTracker = WindowTracker
