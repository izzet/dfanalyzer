project(
    'dftracer-analyzer',
    ['c', 'cpp'],
    meson_version: '>= 1.5.0',
    version: run_command(['scripts/versioning/scm_version.py', 'version'], check: true).stdout().strip(),
    default_options: ['buildtype=debugoptimized', 'cpp_std=c++17', 'warning_level=3'],
)

#### VERSIONING FILE ####

py_mod = import('python')
py = py_mod.find_installation('python3')

# Run helper to get version, version_tuple, and commit_id
ver_out = run_command(py, files('scripts/versioning/scm_version.py'),
                      check: false).stdout().strip()

version = meson.project_version()
version_tuple_lit = '(0, 0, 0)'
commit_id = ''

if ver_out != ''
  lines = ver_out.split('\n')
  if lines.length() >= 1 and lines[0] != ''
    version = lines[0]
  endif
  if lines.length() >= 2 and lines[1] != ''
    version_tuple_lit = lines[1]
  endif
  if lines.length() >= 3
    commit_id = lines[2]
  endif
endif

# Write VERSION.txt to scripts/versioning/ before creating dist
# We need to pass the version explicitly since .git won't be available
meson.add_dist_script(
  'sh', '-c',
  'cd "$MESON_DIST_ROOT" && echo "' + version + '" > scripts/versioning/VERSION.txt'
)

conf = configuration_data()
conf.set('VERSION', version)
conf.set('VERSION_TUPLE', version_tuple_lit)
conf.set('COMMIT_ID', commit_id)

ver_py = configure_file(
  input: 'scripts/versioning/_version.py.in',
  output: '_version.py',
  configuration: conf
)

####################

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

subdir('python/dftracer/analyzer')

if get_option('enable_tools')
    subdir('tools')
endif
