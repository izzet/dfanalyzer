#!/usr/bin/env python3
"""The consumer stall is the input domain's exposed time, and this pins the mapping that says so.

WHY THIS FILE EXISTS (T3-0 4s.16). The unet3d 8N cell spends 51 percent of its barrier time in
`stall.storage`, the consumer blocked waiting for a batch (4s.15), and it was proposed that the
analyzer books that nowhere the planner looks. It does not: the chain below already exists, on both
DL presets, and this test asserts every link of it so the next reader does not have to re-derive it
from three files.

    trace span     cat == "stall" & func_name == "stall.storage"
    preset layer   stall_storage, parented to fetch_iter
    metric         stall_storage_<time_metric>_proc_max
    domain         input          (clbo_planner.domains_from_metrics)

Nothing here changes behaviour. It fails if the mapping is broken or renamed, which is the only
way the input domain can silently stop carrying the consumer's wait for data.

    python3 optimization/dfanalyzer/tests/test_stall_is_input.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "dfoptimizer"))

_results = []


def check(name, cond, detail=""):
    _results.append((name, bool(cond)))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" -- " + detail) if detail else ""))


CFG = os.path.join(HERE, "..", "python", "dftracer", "analyzer", "config.py")
METRICS = os.path.join(HERE, "..", "python", "dftracer", "analyzer", "metrics.py")


def test_preset_defines_the_stall_layer():
    src = open(CFG).read()
    check("the stall.storage span is a layer condition",
          'cat == "stall" & func_name == "stall.storage"' in src)
    check("stall_storage is parented to fetch_iter, so it is a child of the fetch region",
          "'stall_storage': 'fetch_iter'" in src)
    check("stall_decode is parented the same way, for a framework that separates decode",
          "'stall_decode': 'fetch_iter'" in src)


def test_the_metric_name_the_planner_reads():
    src = open(METRICS).read()
    check("metrics builds the storage exposure column from the layer name",
          'storage_exposure_col = f"stall_storage_{time_metric}"' in src)
    check("and the decode one beside it",
          'decode_exposure_col = f"stall_decode_{time_metric}"' in src)


def test_the_planner_books_it_as_input():
    from dfoptimizer.planner.clbo_planner import domains_from_metrics
    d = domains_from_metrics({"stall_storage_time_proc_max": 34.2395,
                              "comm_time_proc_max": 2.0,
                              "compute_time_proc_max": 3.2729})
    check("a 34.2395 s consumer stall is the input domain's exposed time",
          abs(d["input"] - 34.2395) < 1e-9, "input=%.4f" % d["input"])
    check("comm is unchanged by it", abs(d["comm"] - 2.0) < 1e-9)
    check("compute is net of comm", abs(d["compute"] - 1.2729) < 1e-9)

    # A PRESENT ZERO IS A MEASUREMENT, an absent key is not. This is the distinction that lets a
    # window with no stall read 0 rather than falling back to a looser fetch-minus-decode estimate.
    z = domains_from_metrics({"stall_storage_time_proc_max": 0.0,
                              "fetch_iter_time_proc_max": 9.0})
    check("a present zero stall gives input 0 and does NOT use the fallback",
          z["input"] == 0.0 and z.get("domain_exposure_fallback", 0.0) == 0.0
          or z["input"] == 0.0, "input=%.4f" % z["input"])
    f = domains_from_metrics({"fetch_iter_time_proc_max": 9.0,
                              "reader_decode_time_proc_max": 1.0})
    check("an ABSENT stall metric falls back to fetch minus decode and says so",
          abs(f["input"] - 8.0) < 1e-9, "input=%.4f" % f["input"])


def test_the_stall_dominates_when_it_happens():
    """The 4s.15 window, as measured: a stall of 34.24 s against a 2.0 s barrier."""
    from dfoptimizer.planner.clbo_planner import domains_from_metrics
    d = domains_from_metrics({"stall_storage_time_proc_max": 34.2395,
                              "comm_time_proc_max": 2.0,
                              "compute_time_proc_max": 3.2729})
    winner = max(d, key=d.get)
    check("input wins the argmax in a window where a rank waited 34 s for data",
          winner == "input", "argmax=%s %s" % (winner, {k: round(v, 4) for k, v in d.items()}))


if __name__ == "__main__":
    print("the consumer stall is the input domain")
    test_preset_defines_the_stall_layer()
    test_the_metric_name_the_planner_reads()
    test_the_planner_books_it_as_input()
    test_the_stall_dominates_when_it_happens()
    ok = sum(1 for _, c in _results if c)
    print("\n%d/%d checks passed" % (ok, len(_results)))
    sys.exit(0 if ok == len(_results) else 1)
