from types import MappingProxyType, SimpleNamespace

from dftracer.analyzer.dftracer import DFTracerAnalyzer, io_function, load_objects_dict


def test_io_function_ignores_invalid_numeric_fields_without_dropping_event():
    row = io_function(
        {
            "name": "pread64",
            "cat": "POSIX",
            "args": {
                "ret": "not-an-int",
                "offset": "bad-offset",
                "fhash": "abc123",
            },
        }
    )

    assert row["file_hash"] == "abc123"
    assert row["io_cat"] == 1
    assert "size" not in row
    assert "offset" not in row


def test_load_objects_dict_ignores_invalid_step_value_instead_of_dropping_record():
    rows = list(
        load_objects_dict(
            {
                "name": "call",
                "cat": "tool",
                "ph": "X",
                "pid": 1,
                "tid": 1,
                "ts": "10",
                "dur": "5",
                "args": {
                    "step": "not-a-number",
                    "tool_name": "write_file",
                },
            },
            time_approximate=True,
            extra_columns=None,
            extra_columns_fn=None,
        )
    )

    assert len(rows) == 1
    assert rows[0]["cat"] == "tool"
    assert rows[0]["ts"] == 10
    assert rows[0]["dur"] == 5
    assert "step" not in rows[0]


def test_load_objects_dict_handles_read_only_mapping_without_mutation():
    event = MappingProxyType(
        {
            "name": "call",
            "cat": "tool",
            "ph": "X",
            "pid": 1,
            "tid": 1,
            "ts": "10",
            "dur": "5",
            "args": MappingProxyType(
                {
                    "step": "2",
                    "tool_name": "write_file",
                }
            ),
        }
    )

    rows = list(
        load_objects_dict(
            event,
            time_approximate=False,
            extra_columns=None,
            extra_columns_fn=None,
        )
    )

    assert len(rows) == 1
    assert rows[0]["ts"] == 10
    assert rows[0]["dur"] == 5
    assert rows[0]["te"] == 15
    assert rows[0]["tinterval"] is not None


def test_get_columns_uses_wide_enough_dtype_for_deep_agent_levels():
    dummy = SimpleNamespace(time_approximate=True)

    columns = DFTracerAnalyzer._get_columns(dummy, None)

    assert columns["level"] == "Int16"
