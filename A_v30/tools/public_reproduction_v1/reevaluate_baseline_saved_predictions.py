"""Independent saved-baseline inference and lineage review; Python stdlib only.

MIT; copyright (c) 2026 Salih Colakoglu and Mustafa Onat.
No fitting, optimizer updates, MATLAB calls, private admissions or network I/O.
This version requires prospective independent review before its first source run.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
import sys
import time

TOL = 2e-11
SEEDS = (101, 211, 307, 401, 503)
METHODS = ('PROTO', 'NN1_FULL', 'LIN_SEQ', 'LIN_JOINT_GLOBAL')
PAIRS = ((1, 2), (1, 3), (1, 4))
COUNTS = {'source_rows': 104, 'normalizers': 5, 'linear_jobs': 175,
          'aliases': 300, 'deterministic_states': 15, 'predictions': 6552}


def load(path, limit=16*1024*1024):
    data = Path(path).read_bytes()
    if len(data) > limit:
        raise ValueError('Input byte limit exceeded: ' + Path(path).name)
    return json.loads(data.decode('utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def checked_path(root, spec):
    relative = Path(spec['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Input mapping must be package-relative.')
    path = (root / relative).resolve()
    if not inside(path, root) or not path.is_file():
        raise ValueError('Missing/out-of-package input: ' + str(relative))
    if path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
        raise ValueError('Mapped public file does not match its pin: ' + str(relative))
    return path


class Review:
    def __init__(self):
        self.checks = collections.Counter()
        self.failures = []
        self.failure_count = 0
        self.max_absolute_difference = 0.0

    def check(self, condition, group, detail):
        self.checks[group] += 1
        if not condition:
            self.failure_count += 1
            if len(self.failures) < 50:
                self.failures.append({'group': group, 'detail': str(detail)})
        return bool(condition)

    def close(self, actual, saved, group, detail):
        self.checks[group] += 1
        if saved is None:
            ok = actual is None
        elif (isinstance(saved, (int, float)) and not isinstance(saved, bool)
              and isinstance(actual, (int, float)) and not isinstance(actual, bool)
              and math.isfinite(saved) and math.isfinite(actual)):
            delta = abs(actual-saved)
            self.max_absolute_difference = max(self.max_absolute_difference, delta)
            ok = delta <= TOL*max(1.0, abs(saved))
        else:
            ok = False
        if not ok:
            self.failure_count += 1
            if len(self.failures) < 50:
                self.failures.append({'group': group, 'detail': str(detail),
                                      'actual': actual, 'saved': saved})
        return ok

    def vector(self, actual, saved, group, detail):
        if not self.check(isinstance(saved, list) and len(actual) == len(saved), group, detail+' shape'):
            return
        for i, (a, b) in enumerate(zip(actual, saved)):
            if isinstance(a, list):
                self.vector(a, b, group, detail+'/'+str(i))
            else:
                self.close(a, b, group, detail+'/'+str(i))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized(row, norm):
    return [(row['raw'][j]-norm['mean'][j])/norm['sd'][j] for j in range(2)]


def prediction_key(p):
    # Null task means class-incremental context, not a fabricated task label.
    return (p['fold'], p['stream'], p['stage'], p['method'], p['seed'],
            p['task'], p['parentrow'])


def expected_job(fold, stream, stage, method, seed):
    prefix = f'J_f{fold}_q{seed}_'
    if stage == 1:
        return prefix+'common_s1'
    return prefix+(f'seq_{stream}_s{stage}' if method == 'LIN_SEQ'
                   else f'joint_global_s{stage}')


def decode(method, state, z, eligible):
    scores, logits, distances = [None]*4, [None]*4, [None]*4
    if method.startswith('LIN_'):
        raw = [z[0]*state['W'][0][k] + z[1]*state['W'][1][k] + state['b'][k]
               for k in range(4)]
        mx = max(raw[k-1] for k in eligible)
        ex = [math.exp(raw[k-1]-mx) for k in eligible]
        denominator = math.fsum(ex)
        for k, value in zip(eligible, ex):
            logits[k-1] = raw[k-1]
            scores[k-1] = value/denominator
    else:
        for k in eligible:
            if method == 'PROTO':
                centre = [state['sumS'][k-1][j]/state['countn'][k-1] for j in range(2)]
                distance = (z[0]-centre[0])**2+(z[1]-centre[1])**2
            else:
                refs = [v for v, label in zip(state['X'], state['y']) if label == k]
                require(bool(refs), 'Missing eligible nearest-reference class.')
                distance = min((z[0]-v[0])**2+(z[1]-v[1])**2 for v in refs)
            distances[k-1] = distance
            scores[k-1] = -distance
    values = [scores[k-1] for k in eligible]
    require(all(math.isfinite(v) for v in values), 'Nonfinite inference output.')
    best = max(values)
    tied = [k for k in eligible if scores[k-1] == best]
    ordered = sorted(values, reverse=True)
    return {'scores': scores, 'logits': logits, 'distances': distances,
            'gap': ordered[0]-ordered[1], 'prediction': min(tied),
            'tie': len(tied) > 1, 'status': 'valid', 'failure': ''}


def read_sources(paths):
    with paths['source_csv'].open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == COUNTS['source_rows'], 'Wrong raw-source row count.')
    source = {}
    for r in rows:
        identity = r['row_id']
        require(identity not in source, 'Duplicate source-row identity.')
        row = {'id': identity, 'label': int(r['label']), 'group': r['condition_group'],
               'group_order': int(r['group_order']),
               'raw': [float(r['E_MPa']), float(r['Hardness_ShoreD'])]}
        row['fold'] = row['group_order'] % 5 + 1
        require(row['label'] in (1, 2, 3, 4) and all(math.isfinite(v) for v in row['raw']),
                'Invalid raw label or measured input.')
        source[identity] = row
    ordered = sorted(source, key=lambda k: (source[k]['group_order'], source[k]['label']))
    groups = collections.defaultdict(list)
    for row in source.values():
        groups[row['group_order']].append(row)
    require(set(groups) == set(range(26)), 'Incomplete condition-group order.')
    require(all(sorted(r['label'] for r in g) == [1, 2, 3, 4]
                and len({r['group'] for r in g}) == 1 for g in groups.values()),
            'A group must contain exactly four contexts.')
    result = load(paths['baseline_result'])
    require(result['schema'] == 'A29_BASELINE_RESULT_V1', 'Wrong saved baseline schema.')
    return source, ordered, result


def evaluate(paths, export=False):
    review = Review()
    source, ordered, result = read_sources(paths)
    protocol = load(paths['baseline_protocol'])
    admission = load(paths['source_admission'])
    require(protocol['schema'] == 'A29_SIMPLE_BASELINE_PROTOCOL_V2'
            and protocol['id'] == 'A29_BASELINE_GROUP5_FIXEDORDER_V2', 'Wrong baseline protocol identity.')
    require(protocol['features']['ordered_columns'] == ['E_MPa', 'Hardness_ShoreD']
            and protocol['linear_training']['seeds'] == list(SEEDS), 'Unsupported predictor/seed contract.')
    require(admission['schema'] == 'A29_SOURCE_ADMISSION_AND_OUTER_FOLD_DESIGN_V1', 'Wrong source-admission identity.')
    require(protocol['source']['sha256'] == admission['source']['sha256'] == digest(paths['source_csv']),
            'Protocol/admission raw-source pin mismatch.')
    admitted = {}
    for group in admission['whole_group_assignments']:
        for parent in group['parent_records']:
            require(parent['row_id'] not in admitted, 'Duplicate admitted parent.')
            admitted[parent['row_id']] = (group['condition_group'], group['group_order_source_0based'],
                                          group['outer_fold'], parent['label'])
    require(set(admitted) == set(source), 'Admission/raw-source membership differs.')
    for identity, row in source.items():
        review.check(admitted[identity] == (row['group'], row['group_order'], row['fold'], row['label']),
                     'source_admission_links', identity)
    for key in ('normalizers', 'linear_jobs', 'aliases', 'deterministic_states', 'predictions'):
        require(len(result[key]) == COUNTS[key], 'Wrong saved population: '+key)
    catalogue = result['source_record_catalog']
    require([r['parentrow'] for r in catalogue] == ordered, 'Catalogue ordering/identity mismatch.')
    for r in catalogue:
        s = source[r['parentrow']]
        review.check((r['group'], r['group_order'], r['label'], r['outer_fold']) ==
                     (s['group'], s['group_order'], s['label'], s['fold']), 'raw_links', r['parentrow'])
    norms = {n['fold']: n for n in result['normalizers']}
    require(set(norms) == set(range(1, 6)), 'Normalizer fold keys.')
    for fold, n in norms.items():
        ids = [k for k in ordered if source[k]['fold'] != fold and source[k]['label'] <= 2]
        review.check(n['train_row_ids'] == ids and n['n'] == len(ids), 'normalizers', f'fold{fold} TRAIN membership')
        require(len(n['mean']) == len(n['sd']) == 2 and all(math.isfinite(v) for v in n['mean']+n['sd'])
                and all(v > 0 for v in n['sd']), 'Invalid frozen normalizer.')
        mean = [math.fsum(source[k]['raw'][j] for k in ids)/len(ids) for j in range(2)]
        sd = [math.sqrt(math.fsum((source[k]['raw'][j]-mean[j])**2 for k in ids)/(len(ids)-1)) for j in range(2)]
        review.vector(mean, n['mean'], 'normalizers', f'fold{fold} mean')
        review.vector(sd, n['sd'], 'normalizers', f'fold{fold} ddof1 SD')
    jobs = {j['job_id']: j for j in result['linear_jobs']}
    require(len(jobs) == 175, 'Duplicate linear job identity.')
    expected_aliases = {(f, stream, st, method, seed): expected_job(f, stream, st, method, seed)
                        for f, stream, st, method, seed in itertools.product(
                            range(1, 6), ('TIL', 'CIL'), range(1, 4), METHODS[2:], SEEDS)}
    aliases = {(r['fold'], r['stream'], r['stage'], r['method'], r['seed']): r['job_id']
               for r in result['aliases']}
    require(len(aliases) == 300, 'Duplicate logical alias key.')
    review.check(aliases == expected_aliases, 'aliases', 'Complete300 logical alias mapping')
    require(set(jobs) == set(expected_aliases.values()), 'Unused/missing unique saved linear job.')
    for identity, job in jobs.items():
        f, st, seed, kind = job['fold'], job['stage'], job['seed'], job['kind']
        require(f in norms and seed in SEEDS and st in (1, 2, 3), 'Invalid saved linear job key.')
        if st == 1:
            expected_id, expected_kind, prior, classes = f'J_f{f}_q{seed}_common_s1', 'common_stage1', '', [1, 2]
        elif kind == 'joint_global':
            expected_id, expected_kind, prior, classes = f'J_f{f}_q{seed}_joint_global_s{st}', kind, '', list(range(1, st+2))
        else:
            require(kind in ('seq_TIL', 'seq_CIL'), 'Unknown linear job kind.')
            stream = kind[4:]
            expected_id, expected_kind = f'J_f{f}_q{seed}_{kind}_s{st}', kind
            prior = expected_job(f, stream, st-1, 'LIN_SEQ', seed)
            classes = list(PAIRS[st-1]) if stream == 'TIL' else [st+1]
        eligible = list(PAIRS[st-1]) if kind == 'seq_TIL' else list(range(1, st+2))
        ids = [k for k in ordered if source[k]['fold'] != f and source[k]['label'] in classes]
        review.check(identity == expected_id and kind == expected_kind and job['start_job_id'] == prior,
                     'state_chain', identity+' linkage')
        review.check(job['training_row_ids'] == ids and job['eligible'] == eligible,
                     'state_chain', identity+' TRAIN/mask')
        common = jobs[f'J_f{f}_q{seed}_common_s1']
        review.check(job['initial_model'] == common['initial_model'], 'state_chain', identity+' original init')
        if prior:
            review.check(job['start_state'] == jobs[prior]['end_state'], 'state_chain', identity+' carry exact state')
        else:
            review.check(all(job['start_state'][k] == job['initial_model'][k] for k in ('W', 'b')),
                         'state_chain', identity+' fresh W/b')
            start = job['start_state']
            review.check(start['t'] == 0 and start['mW'] == start['vW'] == [[0]*4, [0]*4]
                         and start['mb'] == start['vb'] == [0]*4,
                         'state_chain', identity+' fresh optimizer reset')
        state = job['end_state']
        review.check(state['t'] == job['start_state']['t']+600,
                     'state_chain', identity+' recorded optimizer count (no updates recomputed)')
        require(len(state['W']) == 2 and all(len(v) == 4 for v in state['W']) and len(state['b']) == 4,
                'Linear saved-parameter orientation.')
        require(all(math.isfinite(v) for row in state['W'] for v in row)
                and all(math.isfinite(v) for v in state['b']), 'Nonfinite saved linear parameters.')
    det = {r['state_id']: r for r in result['deterministic_states']}
    require(set(det) == {f'D_f{f}_s{st}' for f in range(1, 6) for st in range(1, 4)}, 'Deterministic state identities.')
    for identity, state in det.items():
        f, st = state['fold'], state['stage']
        require(f in norms and st in (1, 2, 3) and identity == f'D_f{f}_s{st}',
                'Deterministic state key/contents mismatch.')
        ids = [k for k in ordered if source[k]['fold'] != f and source[k]['label'] <= st+1]
        review.check(state['normalizer_fold'] == f and state['admitted_row_ids'] == ids,
                     'deterministic_raw_links', identity+' admitted sources')
        proto, nn = state['proto'], state['nn']
        active = [k <= st+1 for k in range(1, 5)]
        review.check(proto['active'] == nn['active'] == active, 'deterministic_raw_links', identity+' active classes')
        review.check(nn['row_ids'] == ids and nn['y'] == [source[k]['label'] for k in ids],
                     'deterministic_raw_links', identity+' nearest reference identities')
        require(len(nn['X']) == len(ids) and all(len(row) == 2 for row in nn['X']), 'Nearest reference shape.')
        for k, z in zip(ids, nn['X']):
            review.vector(normalized(source[k], norms[f]), z, 'deterministic_raw_links', identity+'/'+k)
        for k in range(1, 5):
            members = [r for r in ids if source[r]['label'] == k]
            total = [math.fsum(normalized(source[r], norms[f])[j] for r in members) for j in range(2)]
            review.check(proto['countn'][k-1] == len(members), 'deterministic_raw_links', identity+f' count{k}')
            review.vector(total, proto['sumS'][k-1], 'deterministic_raw_links', identity+f' sum{k}')
            centre = [v/len(members) for v in total] if members else [None, None]
            review.vector(centre, state['batch_centroid'][k-1], 'deterministic_raw_links', identity+f' centroid{k}')
    expected_keys = set()
    for f, stream, st, method in itertools.product(range(1, 6), ('TIL', 'CIL'), range(1, 4), METHODS):
        seeds = (None,) if method in METHODS[:2] else SEEDS
        tasks = range(1, st+1) if stream == 'TIL' else (None,)
        for seed, task in itertools.product(seeds, tasks):
            classes = PAIRS[task-1] if task is not None else range(1, st+2)
            for k in ordered:
                if source[k]['fold'] == f and source[k]['label'] in classes:
                    expected_keys.add((f, stream, st, method, seed, task, k))
    require(len(expected_keys) == 6552, 'Prospective query population count.')
    actual_keys, prediction_ids, output = set(), set(), []
    counts = collections.Counter()
    for p in result['predictions']:
        key, identity = prediction_key(p), p['prediction_id']
        require(key not in actual_keys and identity not in prediction_ids, 'Duplicate query/prediction identity.')
        actual_keys.add(key); prediction_ids.add(identity)
        require(key in expected_keys, 'Unadmitted query population key.')
        f, stream, st, method, seed, task, parent = key
        row = source[parent]
        eligible = list(PAIRS[task-1]) if task is not None else list(range(1, st+2))
        review.check(p['eligible'] == eligible and p['truth'] == row['label']
                     and (p['group'], p['group_order']) == (row['group'], row['group_order']),
                     'query_links', identity+' truth/metadata/mask')
        state_id = f'D_f{f}_s{st}' if method in METHODS[:2] else aliases[(f, stream, st, method, seed)]
        review.check(p['state_id'] == state_id, 'query_links', identity+' saved state alias')
        z = normalized(row, norms[f])
        state = det[state_id]['proto' if method == 'PROTO' else 'nn'] if method in METHODS[:2] else jobs[state_id]['end_state']
        values = decode(method, state, z, eligible)
        for name in ('scores', 'logits', 'distances'):
            review.vector(values[name], p[name], 'prediction_numeric', identity+'/'+name)
        review.close(values['gap'], p['gap'], 'prediction_numeric', identity+'/gap')
        for name in ('prediction', 'tie', 'status', 'failure'):
            review.check(values[name] == p[name], 'prediction_exact', identity+'/'+name)
        counts[(method, stream)] += 1
        if export:
            output.append(dict(prediction_id=identity, fold=f, stream=stream, stage=st,
                               method=method, seed=seed, task=task, parentrow=parent,
                               truth=row['label'], eligible=eligible, state_id=state_id,
                               raw_input=row['raw'], normalized_input=z, **values))
    review.check(actual_keys == expected_keys, 'query_population', 'Exact6552 independent query keys')
    summary = {'schema': 'A30_PORTABLE_BASELINE_SAVED_INFERENCE_REVIEW_V1',
               'status': 'PASS' if not review.failure_count else 'MISMATCH',
               'predictions': len(actual_keys), 'source_rows': len(source),
               'linear_jobs': len(jobs), 'logical_aliases': len(aliases),
               'deterministic_snapshots': len(det), 'normalizers': len(norms),
               'checks': sum(review.checks.values()), 'check_groups': dict(review.checks),
               'failure_count': review.failure_count, 'first_failures': review.failures,
               'max_absolute_numeric_difference': review.max_absolute_difference,
               'tolerance': TOL, 'tolerance_rule': 'abs(actual-saved)<=2e-11*max(1,abs(saved)); labels/ties/status exact',
               'counts_by_method_stream': [{'method': m, 'stream': s, 'predictions': n}
                                          for (m, s), n in sorted(counts.items())],
               'scope': 'Raw-input saved-state baseline inference and lineage only. No optimization, gradients, old benchmark rerun, physical validation or significance inference.',
               'new_training': 0, 'optimizer_updates': 0, 'physical_validation': 0}
    return summary, output


def fresh_output(path, package_root):
    p = Path(path).resolve()
    if p.exists() or inside(p, package_root):
        raise ValueError('Output must be a fresh file outside the sealed package.')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', required=True)
    parser.add_argument('--input-map', type=Path, default=Path(__file__).with_name('PUBLIC_REPRODUCTION_INPUTS_v1.json'))
    parser.add_argument('--report', type=Path)
    parser.add_argument('--expected-jsonl', type=Path, help='Optional independently computed6552-row export for later Simulink comparison.')
    args = parser.parse_args()
    root = Path(args.package_root).resolve()
    report = fresh_output(args.report, root) if args.report else None
    export = fresh_output(args.expected_jsonl, root) if args.expected_jsonl else None
    require(report is None or export is None or report != export, 'Use distinct report/export paths.')
    started = time.monotonic()
    summary = {'schema': 'A30_PORTABLE_BASELINE_SAVED_INFERENCE_REVIEW_V1', 'status': 'HOLD', 'new_training': 0}
    paths = {}
    try:
        mapping = load(args.input_map, 1024*1024)
        require(mapping['schema'] == 'A30_PUBLIC_REPRODUCTION_INPUTS_V1', 'Wrong input-map schema.')
        paths = {role: checked_path(root, spec) for role, spec in mapping['runtime_inputs'].items()}
        require({'source_csv', 'baseline_result', 'baseline_protocol', 'source_admission'} <= set(paths), 'Missing public input role.')
        before = {role: digest(p) for role, p in paths.items()}
        summary, rows = evaluate(paths, export=export is not None)
        require(all(digest(p) == before[role] for role, p in paths.items()), 'Input changed during review.')
        summary['input_map_sha256'] = digest(args.input_map)
        summary['input_pins'] = mapping['runtime_inputs']
        summary['source_files_unchanged'] = True
        if export and summary['status'] == 'PASS':
            payload = ''.join(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n' for row in rows)
            require(len(payload.encode('utf-8')) <= 16*1024*1024, 'Expected export exceeds16MiB bound.')
            with export.open('x', encoding='utf-8', newline='\n') as handle:
                handle.write(payload)
            summary['expected_export'] = {'file': export.name, 'rows': len(rows), 'sha256': digest(export)}
    except Exception as error:
        summary.update(status='HOLD', error_type=type(error).__name__, error=str(error))
    summary['elapsed_seconds'] = time.monotonic()-started
    text = json.dumps(summary, indent=2, allow_nan=False)
    if report:
        with report.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(text+'\n')
    print(text)
    return 0 if summary['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
