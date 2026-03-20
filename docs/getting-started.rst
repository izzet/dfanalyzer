Getting Started
===============

Installation
------------

To install DFAnalyzer through ``pip`` (recommended for most users):

.. code-block:: bash

   # Ensure runtime dependencies for optional features (e.g., Darshan, Recorder) are installed.
   # This might involve using your system's package manager or a tool like Spack.
   # Example using Spack to prepare the environment:
   # spack -e tools install
   pip install dftracer-analyzer

To install DFAnalyzer from source (for developers or custom builds):

.. code-block:: bash

   # 1. Install system dependencies:
   #    Refer to the "Install system dependencies" step in .github/workflows/ci.yml
   #    (e.g., build-essential, cmake, libarrow-dev, libhdf5-dev, ninja-build, etc.).
   #    Alternatively, tools like Spack can help manage these:
   #    # spack -e tools install
   module load ninja

   # 2. Install Python build dependencies:
   python -m pip install --upgrade pip meson-python setuptools wheel

   # 3. Install DFAnalyzer from the root of this repository:
   #    The following command includes optional C++ components (tests and tools).
   #    The --prefix argument is optional and specifies the installation location.
   pip install -e . \
     -Csetup-args="--prefix=$HOME/.local" \
     -Csetup-args="-Denable_tests=true" \
     -Csetup-args="-Denable_tools=true"

   # (Optional) Install dependencies for running tests if you plan to contribute or run local tests:
   # pip install -r tests/requirements.txt

Usage
-----

Here's an example of how to run DFAnalyzer with the ``recorder`` analyzer using sample data included in the repository:

.. code-block:: bash

   # Before running, ensure the sample data is extracted.
   # For example, to extract the 'dftracer-dlio' sample used below:
   # mkdir -p tests/data/extracted
   # tar -xzf tests/data/dftracer-dlio.tar.gz -C tests/data/extracted
   dfanalyzer analyzer/preset=dlio trace_path=tests/data/extracted/dftracer-dlio view_types=[time_range]

This command analyzes the traces and prints a high-level summary of the application's execution. Below is a sample of the "Time Period Summary" output:

.. code-block:: none

                                                     Time Period Summary
   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
   ┃ Metric                                                                    ┃ Unit             ┃                 Value ┃
   ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
   │ Job Time                                                                  │ seconds          │                56.695 │
   │ Total Count                                                               │ count            │                18,039 │
   │ Total Files                                                               │ count            │                   166 │
   │ Total Nodes                                                               │ count            │                     1 │
   │ Total Processes                                                           │ count            │                     8 │
   │ App Count                                                                 │ count            │                     8 │
   │ Training Count                                                            │ count            │                     8 │
   │ Epoch Count                                                               │ count            │                    40 │
   │ Compute Count                                                             │ count            │                   200 │
   │ Fetch Data Count                                                          │ count            │                   160 │
   │ Checkpoint Count                                                          │ count            │                     8 │
   │ Data Loader Count                                                         │ count            │                   816 │
   │ Data Loader Fork Count                                                    │ count            │                    96 │
   │ Reader Count                                                              │ count            │                 3,200 │
   │ POSIX - All Count                                                         │ count            │                10,581 │
   │ POSIX - All Size                                                          │ MB               │            111833.172 │
   │ POSIX - All Bandwidth                                                     │ MB/s             │              6048.367 │
   │ POSIX - All Avg Transfer Size                                             │ MB               │                10.569 │
   │ POSIX - Reader Count                                                      │ count            │                10,432 │
   │ POSIX - Reader Size                                                       │ MB               │            111833.161 │
   │ POSIX - Reader Bandwidth                                                  │ MB/s             │              6095.909 │
   │ POSIX - Reader Avg Transfer Size                                          │ MB               │                10.720 │
   │ POSIX - Checkpoint Count                                                  │ count            │                    45 │
   │ POSIX - Checkpoint Size                                                   │ MB               │                 0.011 │
   │ POSIX - Checkpoint Bandwidth                                              │ MB/s             │                 2.525 │
   │ POSIX - Checkpoint Avg Transfer Size                                      │ MB               │                 0.000 │
   └───────────────────────────────────────────────────────────────────────────┴──────────────────┴───────────────────────┘

DFAnalyzer also provides a detailed breakdown of performance metrics for each layer of the application. Here is a snippet of the "Layer Breakdown" section from the same run, which includes the percentage of time each layer overlaps with its parent layer:

.. code-block:: none

                                                Layer Breakdown (w/ overlap %)
   ┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
   ┃ Layer                  ┃         Time (s) ┃             Ops ┃     Ops/sec ┃            Size (MB) ┃  Bandwidth (MB/s) ┃
   ┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
   │ App                    │    55.246 (----) │        8 (----) │       0.145 │                    - │                 - │
   │ Training               │    55.246 (----) │        8 (----) │       0.145 │                    - │                 - │
   │ Epoch                  │    54.937 (----) │       40 (----) │       0.728 │                    - │                 - │
   │ Compute                │    40.854 (----) │      200 (----) │       4.895 │                    - │                 - │
   │ Fetch Data             │    16.889 (----) │      160 (----) │       9.474 │                    - │                 - │
   │ Checkpoint             │     0.005 (----) │        8 (----) │    1762.503 │                    - │                 - │
   │ Data Loader            │    21.871 ( 54%) │      816 ( 57%) │      37.310 │                    - │                 - │
   │ Data Loader Fork       │     0.181 (  0%) │       96 (  0%) │     530.903 │                    - │                 - │
   │ Reader                 │    21.480 ( 55%) │    3,200 ( 67%) │     148.979 │                    - │                 - │
   │ POSIX - All            │    18.490 ( 54%) │   10,581 ( 59%) │     572.261 │    111833.172 ( 59%) │          6048.367 │
   │ POSIX - Reader         │    18.346 ( 55%) │   10,432 ( 60%) │     568.637 │    111833.161 ( 59%) │          6095.909 │
   │ POSIX - Checkpoint     │     0.004 (----) │       45 (----) │   10433.573 │         0.011 (----) │             2.525 │
   └────────────────────────┴──────────────────┴─────────────────┴─────────────┴──────────────────────┴───────────────────┘
