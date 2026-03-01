py.install_sources(
    [
        ver_py,
        '__init__.py',
        '__main__.py',
        'analysis.py',
        'analysis_utils.py',
        'analyzer.py',
        'cluster.py',
        'config.py',
        'constants.py',
        'darshan.py',
        'dftracer.py',
        'fact_engine.py',
        'fact_rules.py',
        'input.py',
        'metrics.py',
        'output.py',
        'recorder.py',
        'types.py',
    ],
    subdir: 'dftracer/analyzer',
)

install_data(
    'configs/fact_rules/dlio.yaml',
    install_dir: py.get_install_dir(subdir: 'dftracer/analyzer/configs/fact_rules'),
)

install_data(
    'configs/schemas/analyzer.fact-envelope.v1.schema.json',
    install_dir: py.get_install_dir(subdir: 'dftracer/analyzer/configs/schemas'),
)

subdir('streaming')
subdir('utils')
