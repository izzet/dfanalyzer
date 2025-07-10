.. _interactive-analysis:

Interactive Analysis
====================

DFAnalyzer provides a Python API for interactive analysis, allowing for detailed exploration of I/O traces within environments like Jupyter notebooks. This guide walks through a typical interactive analysis workflow.

.. contents::
   :local:

Prepare Environment
-------------------

First, ensure DFAnalyzer is installed in your Python environment. For detailed instructions, please refer to the :doc:`getting-started` guide.

Prepare Trace Data
------------------

Next, ensure your trace data is accessible. You can use the sample datasets located in the ``tests/data`` directory. For this example, we extract a sample trace archive.

.. code-block:: bash

   !mkdir -p ./data
   !tar -xzf ../../tests/data/dftracer-dlio.tar.gz -C ./data

Run Analysis
------------

With the environment and data ready, you can run the analysis.

Initialize DFAnalyzer
~~~~~~~~~~~~~~~~~~~~~

Initialize DFAnalyzer using ``init_with_hydra``, providing configuration overrides as needed. This sets up the analyzer, such as ``dftracer`` with a specific preset like ``dlio``.

.. code-block:: python

   from dfanalyzer import init_with_hydra

   percentile = 0.9
   run_dir = f"./unet3d_v100_hdf5"
   time_granularity = 5e6  # 5 seconds
   trace_path = f"./data/dftracer-dlio"
   view_types = ["time_range", "proc_name"]

   dfa = init_with_hydra(
       hydra_overrides=[
           'analyzer=dftracer',
           'analyzer/preset=dlio',
           'analyzer.checkpoint=False',
           f"analyzer.time_granularity={time_granularity}",
           f"hydra.run.dir={run_dir}",
           f"percentile={percentile}",
           f"trace_path={trace_path}",
       ]
   )

You can inspect the Dask client and the preset configuration:

.. code-block:: python

   # Access the Dask client
   dfa.client

   # View the preset configuration
   dict(dfa.analyzer.preset.layer_defs)


Execute the Analysis
~~~~~~~~~~~~~~~~~~~~

Run the trace analysis using the ``analyze_trace`` method.

.. code-block:: python

   result = dfa.analyze_trace(percentile=percentile, view_types=view_types)

The results can then be passed to the output handler to display a summary.

.. code-block:: python

   dfa.output.handle_result(result)


Result Exploration
------------------

The ``result`` object contains detailed views of the analyzed data, which you can explore using pandas DataFrames.

.. code-block:: python

   # Raw trace data
   result._traces.head()

   # High-level metrics 
   result._hlms['reader_posix_lustre'].head()

   # Layer-based characteristics
   result._main_views['reader_posix_lustre'].head()

   # Time-range views
   result.views['reader_posix_lustre'][('time_range',)].head()
