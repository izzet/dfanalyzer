from dftracer.analyzer.constants import IOCategory
from dftracer.analyzer.dftracer import get_io_cat, io_function


def test_get_io_cat_recognizes_pread64_and_pwrite64_aliases():
    assert get_io_cat("pread64") == IOCategory.READ.value
    assert get_io_cat("pwrite64") == IOCategory.WRITE.value


def test_io_function_counts_pread64_return_value_as_read_size():
    row = io_function(
        {
            "name": "pread64",
            "cat": "POSIX",
            "args": {
                "ret": 512,
                "offset": 128,
                "fhash": "abc123",
            },
        }
    )

    assert row["io_cat"] == IOCategory.READ.value
    assert row["size"] == 512
    assert row["offset"] == 128
