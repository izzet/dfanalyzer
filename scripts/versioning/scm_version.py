#!/usr/bin/env python3
import os, subprocess, sys

def sh(*args):
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""

def repo_root(explicit):
    if explicit:
        return explicit
    return sh('git', 'rev-parse', '--show-toplevel')

def parse_describe(desc):
    # Accept TAG-N-gHASH[-dirty]
    if not desc:
        return None, None
    core = desc[:-6] if desc.endswith('-dirty') else desc
    try:
        tag_raw, n_str, _ = core.rsplit('-', 2)
        distance = int(n_str)
    except Exception:
        return None, None
    tag = tag_raw[1:] if tag_raw.startswith('v') else tag_raw
    return tag, distance

def compute_version(repo):
    if not repo:
        return '0.0.0'
    # 1) Prefer git describe (if tags exist)
    desc = sh('git', '-C', repo, 'describe', '--tags', '--long', '--dirty')
    tag, distance = parse_describe(desc)
    if tag is not None:
        return ('%s.dev%d' % (tag, distance)) if (distance and distance > 0) else tag
    # 2) No tags → total commit count
    total = sh('git', '-C', repo, 'rev-list', '--count', 'HEAD')
    try:
        n = int(total)
    except Exception:
        n = 0
    return '0.0' if n == 0 else '0.0.dev%d' % n

def tuple_lit_from_version(ver):
    if '.dev' in ver:
        base, dev = ver.split('.dev', 1)
        ints = [int(x) for x in base.split('.')]
        return '(' + ', '.join([str(x) for x in ints]) + ", 'dev" + dev + "'" + ')'
    ints = [int(x) for x in ver.split('.')]
    return '(' + ', '.join([str(x) for x in ints]) + ')'

def main():
    # Usage: scm_version.py [version|tuple|commit|write|all] [repo]
    mode = 'all'
    repo = None
    if len(sys.argv) >= 2 and sys.argv[1] in ('version','tuple','commit','write','all'):
        mode = sys.argv[1]
        repo = sys.argv[2] if len(sys.argv) >= 3 else os.getenv('REPO_ROOT')
    elif len(sys.argv) >= 2:
        repo = sys.argv[1]

    repo = repo_root(repo)
    ver = compute_version(repo)
    vt  = tuple_lit_from_version(ver)

    if mode == 'version':
        print(ver)
    elif mode == 'tuple':
        print(vt)
    elif mode == 'commit':
        print('')  # you don’t want +hash
    elif mode == 'write':
        out = sys.argv[3] if len(sys.argv) >= 4 else 'scripts/versioning/VERSION.txt'
        with open(out, 'w', encoding='utf-8') as f:
            f.write(ver + '\n')
    else:
        print(ver); print(vt); print('')

if __name__ == '__main__':
    main()
