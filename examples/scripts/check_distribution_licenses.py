# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Verify the license declarations and actual contents of wheel and sdist.

Usage: python scripts/check_distribution_licenses.py DIST_DIR
No Newton/GPU/runtime dependencies are imported.
"""
from email.parser import BytesParser
from collections import Counter
import argparse
from pathlib import Path
import ast
import hashlib
import json
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = json.loads((ROOT / 'third_party_inventory.json').read_text())
EXPECTED = {p.relative_to(ROOT).as_posix() for p in ROOT.glob('shelf/**/scene.json')}
BINARIES = {e['binary_payload']['path'] for e in INVENTORY['bundled_scenes']}
ADDITIONAL = {e['path']: e for e in INVENTORY.get('additional_bundled_assets', [])}
NOTICES = ['LICENSE', 'NOTICE', 'THIRD_PARTY.md', 'third_party_inventory.json'] + INVENTORY.get('bundled_component_notices', []) + [
    p.relative_to(ROOT).as_posix() for p in sorted((ROOT / 'licenses').rglob('*.txt'))
]


class AuditError(Exception):
    """A declared license, notice or asset does not match the tree or a distribution."""


def require(condition, message='license audit check failed'):
    """Explicit check that still runs under ``python -O``."""
    if not condition:
        raise AuditError(message)


def asset_calls(source):
    """Keep source positions for navigation, but review the semantic call signature."""
    found = []
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else '')
        if name in {'get_asset', 'get_asset_directory', 'get_kamino_basics_asset',
                    'download_asset', 'download_external_git_folder'}:
            found.append({
                'call': name,
                'function': ast.unparse(node.func),
                'argument': ast.unparse(node.args[0]) if node.args else '',
                'arguments': [ast.unparse(arg) for arg in node.args],
                'keywords': sorted(({'name': kw.arg, 'value': ast.unparse(kw.value)} for kw in node.keywords),
                                   key=lambda kw: kw['name'] or ''),
                'line': node.lineno,
            })
            # Directory helpers carry no filename; include the enclosing join so
            # changing bear.usd to another file cannot silently pass review.
            parent = parents.get(node)
            if name == 'get_asset_directory' and isinstance(parent, ast.Call):
                found[-1]['path_expression'] = ast.unparse(parent)
    return found


def call_sites():
    return {p.relative_to(ROOT).as_posix(): calls
            for p in sorted(p for directory in ('shelf', 'common')
                            for p in (ROOT / directory).glob('**/*.py'))
            if (calls := asset_calls(p.read_text()))}


def call_site_signatures(sites):
    # Ignore line numbers and traversal order, while preserving duplicate calls.
    return {path: Counter(json.dumps({k: v for k, v in call.items() if k != 'line'}, sort_keys=True)
                          for call in calls)
            for path, calls in sites.items()}


def check_sources():
    require(EXPECTED == {s['path'] for s in INVENTORY['bundled_scenes']}, 'Update the scene inventory')
    require(call_site_signatures(call_sites()) == call_site_signatures(INVENTORY['runtime_call_sites']),
            'Review new/changed runtime asset references; after review run '
            'python scripts/check_distribution_licenses.py --update-call-sites')
    for entry in INVENTORY['runtime_assets']:
        require(hashlib.sha256((ROOT / entry['notice']).read_bytes()).hexdigest() == entry['notice_sha256'],
                (entry['notice'], 'runtime notice hash differs'))
    for name, entry in ADDITIONAL.items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == entry['sha256'], (name, 'hash differs'))
        require((ROOT / entry['notice']).is_file(), (name, 'missing notice'))
    for entry in INVENTORY['documentation_media']:
        require(hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest() == entry['sha256'],
                (entry['path'], 'hash differs'))
    for entry in INVENTORY['bundled_scenes']:
        binary = entry['binary_payload']
        require(hashlib.sha256((ROOT / binary['path']).read_bytes()).hexdigest() == binary['sha256'],
                (binary['path'], 'hash differs'))
        data = (ROOT / entry['path']).read_bytes()
        require(hashlib.sha256(data).hexdigest() == entry['sha256'], (entry['path'], 'hash differs'))
        scene = json.loads(data)
        require((scene.get('info', {}).get('robot') or 'none') == entry['robot'], (entry['path'], 'robot differs'))
        for name in entry['notices']:
            require(name in NOTICES, (entry['path'], 'undeclared notice', name))
        if 'model_evidence' in entry:
            expected = entry['model_evidence']['mass_prefix']
            actual = scene['bodies']['mass'][:len(expected)]
            require(len(actual) == len(expected) and all(abs(a - b) < 1e-6 for a, b in zip(actual, expected)),
                    (entry['path'], 'model masses differ'))
    ports = [p for p in ROOT.glob('shelf/**/presets.toml') if 'source = "steel-webgpu' in p.read_text()]
    require(len(ports) == 27, ('expected 27 steel-webgpu ports', len(ports)))
    require(all('# Direct source: steel-webgpu;' in p.with_name('example.py').read_text() for p in ports),
            'a steel-webgpu port lacks its direct-source header')
    source = INVENTORY['steel_webgpu_source']
    notice = (ROOT / source['notice']).read_bytes()
    require(hashlib.sha256(notice).hexdigest() == source['notice_sha256'], 'steel-webgpu notice hash differs')
    require(all(line in notice.decode() for line in source['copyright']), 'steel-webgpu copyright lines missing')
    require(all(p.with_name('example.py').read_text().startswith(source['port_header'] + '\n') for p in ports),
            'a steel-webgpu port lacks its copyright header')
    for name in ['nut_bolt_hydro', 'nut_bolt_sdf']:
        path = next(ROOT.glob(f'shelf/**/{name}/example.py'))
        tree = ast.parse(path.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == 'download_external_git_folder']
        require(len(calls) == 1 and any(k.arg == 'ref' for k in calls[0].keywords), (name, 'download is not pinned'))
        require('aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3' in path.read_text(), (name, 'unexpected download revision'))


def check(path):
    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            files = {n: archive.read(n) for n in archive.namelist() if not n.endswith('/')}
        metadata_name = next(n for n in files if n.endswith('.dist-info/METADATA'))
    else:
        with tarfile.open(path) as archive:
            files = {m.name.split('/', 1)[1]: archive.extractfile(m).read()
                     for m in archive.getmembers() if m.isfile()}
        metadata_name = 'PKG-INFO'
    metadata = BytesParser().parsebytes(files[metadata_name])
    require(metadata['License-Expression'] == INVENTORY['package_license_expression'],
            (path, 'license expression differs'))
    declared = metadata.get_all('License-File', [])
    require(set(declared) == set(NOTICES), (path, 'license metadata differs', set(declared) ^ set(NOTICES)))
    for name in NOTICES:
        matches = [v for k, v in files.items() if k == name or k.endswith('/licenses/' + name)]
        require(matches and all(v == (ROOT / name).read_bytes() for v in matches),
                (path, 'missing/altered notice', name))
    json_files = {n for n in files if n.endswith('.json')}
    known_json = EXPECTED | set(ADDITIONAL) | {'third_party_inventory.json', 'docs/steel_tuning/evidence.json'}
    require(all(n in known_json or n.endswith('/licenses/third_party_inventory.json') for n in json_files),
            (path, 'unreviewed JSON data'))
    scenes = {n for n in files if n.startswith('shelf/') and n.endswith('/scene.json')}
    require(scenes == EXPECTED, (path, 'scene inventory differs', scenes ^ EXPECTED))
    require(all(files[n] == (ROOT / n).read_bytes() for n in scenes), (path, 'scene bytes differ'))
    actual_bins = {n for n in files if n.endswith('.bin')}
    require(actual_bins == BINARIES, (path, 'binary scene inventory differs', actual_bins ^ BINARIES))
    require(all(files[n] == (ROOT / n).read_bytes() for n in BINARIES), (path, 'binary scene bytes differ'))
    for name in ADDITIONAL:
        require(name in files and files[name] == (ROOT / name).read_bytes(),
                (path, 'missing/altered additional asset', name))
    forbidden = { '.obj', '.stl', '.urdf', '.xml', '.mjcf', '.usd', '.usda', '.usdc', '.usdz',
                 '.onnx', '.pt', '.npz', '.npy', '.glb', '.gltf', '.ply', '.fbx'}
    require(not any(Path(n).suffix.lower() in forbidden for n in files if n not in ADDITIONAL),
            (path, 'external model/policy was bundled'))
    require(not any(n.startswith('docs/') and Path(n).suffix.lower() in {'.png', '.gif', '.webp'} for n in files),
            (path, 'documentation image was bundled'))
    for name, data in files.items():
        if name in BINARIES or name in ADDITIONAL:
            continue
        try:
            data.decode('utf-8')
        except UnicodeDecodeError:
            raise AuditError((path, 'unreviewed binary file', name)) from None
    print(f'{path.name}: {len(scenes)} exact JSON/binary pairs; {len(NOTICES)} complete license/inventory files; metadata and distribution boundaries passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dist_dir', nargs='?')
    parser.add_argument('--update-call-sites', action='store_true',
                        help='Refresh reviewed call signatures and diagnostic line numbers only')
    args = parser.parse_args()
    if args.update_call_sites:
        INVENTORY['runtime_call_sites'] = call_sites()
        (ROOT / 'third_party_inventory.json').write_text(json.dumps(INVENTORY, indent=2) + '\n')
        print('Updated runtime call-site inventory; model, notice and media declarations unchanged')
        sys.exit(0)
    if not args.dist_dir:
        parser.error('Provide a distribution directory, or --update-call-sites')
    check_sources()
    directory = Path(args.dist_dir)
    wheels = list(directory.glob('*.whl'))
    sdists = list(directory.glob('*.tar.gz'))
    require(wheels and sdists, 'Build both a wheel and sdist first')
    for artifact in sorted(wheels + sdists):
        check(artifact)
