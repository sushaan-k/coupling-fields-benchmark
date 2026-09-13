"""Download public counts and retain the frozen donor/marker selection."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

import h5py
import numpy as np


def digest(path, algorithm='sha256'):
    value = hashlib.new(algorithm)
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def column(node):
    if isinstance(node, h5py.Dataset):
        return node.asstr()[:] if node.dtype.kind in 'OSU' else node[:]
    codes = node['codes'][:]
    assert np.all(codes >= 0)
    return column(node['categories'])[codes]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--public-commit', required=True)
    args = parser.parse_args()
    here, work = Path(__file__).resolve().parent, args.work
    work.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((here / 'freeze.json').read_text())
    for name in ('PLAN.md', 'selection.json', 'acquire.py'):
        assert digest(here / name) == frozen['sha256'][name]
    base = f'https://raw.githubusercontent.com/sushaan-k/coupling-fields-benchmark/{args.public_commit}/analysis/colon_generalization/'
    with urllib.request.urlopen(base + 'freeze.json', timeout=60) as response:
        assert response.read() == (here / 'freeze.json').read_bytes()
    access = work / 'count_access.json'
    if not access.exists():
        with access.open('x') as handle:
            json.dump({'public_commit': args.public_commit,
                       'started_utc': datetime.now(timezone.utc).isoformat()}, handle, indent=2)
    selection = json.loads((here / 'selection.json').read_text())
    path = work / 'colon_counts.h5ad'
    if not path.exists():
        temporary = path.with_suffix('.partial')
        with urllib.request.urlopen(selection['url'], timeout=120) as response, temporary.open('wb') as out:
            for block in iter(lambda: response.read(8*1024*1024), b''):
                out.write(block)
        temporary.rename(path)
    assert path.stat().st_size == 3786136022
    assert digest(path, 'md5') == 'd0dbb0fc4c95fbd5ce8e72cf418aad21'
    rows = np.array([p['selected_rows'] for p in selection['patients']])
    wanted = np.array(selection['axes']['RNA'] + selection['axes']['ADT'])
    with h5py.File(path, 'r') as h:
        names = column(h['var/_index'])
        for assay in ('RNA', 'ADT'):
            assert names[selection['axes'][assay]].tolist() == selection['selected_feature_names'][assay]
        donors, kind = column(h['obs/CoLabs_patient']), column(h['obs/LIBRARY.TYPE'])
        assert len({donors[p[0]] for p in rows}) == len(rows)
        for r in rows:
            assert len(set(donors[r])) == 1 and np.all(kind[r] == 'GEX_CITE')
        matrix = h['layers/counts']
        assert matrix.attrs['encoding-type'] == 'csr_matrix'
        assert list(matrix.attrs['shape']) == selection['shape']
        ptr = matrix['indptr'][:]
        assert len(ptr) == selection['shape'][0]+1 and np.all(np.diff(ptr) >= 0)
        assert ptr[-1] == len(matrix['indices']) == len(matrix['data'])
        output = np.zeros((len(rows)*512, 18))
        ordered = sorted((int(row), i) for i, row in enumerate(rows.ravel()))
        lookup = {int(c): j for j, c in enumerate(wanted)}
        for row, position in ordered:
            columns = matrix['indices'][ptr[row]:ptr[row+1]]
            values = matrix['data'][ptr[row]:ptr[row+1]]
            assert np.isfinite(values).all() and np.all(values >= 0)
            assert np.all(values == np.floor(values))
            for col, val in zip(columns, values):
                if int(col) in lookup:
                    output[position, lookup[int(col)]] += val
        output = output.reshape(len(rows), 512, 18)
    for name, segment in [('adaptation', slice(0, 256)), ('scoring', slice(256, 512))]:
        np.savez_compressed(work / f'{name}_counts.npz', rna=output[:, segment, :9],
                            protein=output[:, segment, 9:])
    record = {'source_md5': digest(path, 'md5'), 'public_commit': args.public_commit,
              'shape': list(output.shape), 'retained_files': {
              name: digest(work / name) for name in ('adaptation_counts.npz', 'scoring_counts.npz')}}
    (work / 'acquisition.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
