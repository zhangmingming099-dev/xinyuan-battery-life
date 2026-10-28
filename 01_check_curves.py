"""MATR充电曲线只读检查。先核验字段和单位，不训练、不生成最终归一化张量。"""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

FILES = [
    '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
    '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
]
FIELDS = ('V', 'I', 'T', 't', 'Qc', 'Qd')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'没有可写记录：{path.name}')
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def stat(path):
    s = path.stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector(file, group, key, index):
    if key not in group:
        raise ValueError(f'缺少字段{key}')
    refs = group[key]
    if refs.ndim != 2 or refs.shape[1] != 1 or index >= len(refs):
        raise ValueError(f'{key}循环引用维度或范围异常')
    target = file[refs[index, 0]]
    if np.asarray(target.attrs.get('MATLAB_empty', 0)).any():
        return np.array([], dtype=float)
    a = np.asarray(target[()], dtype=float)
    if a.ndim != 2 or 1 not in a.shape:
        raise ValueError(f'{key}不是向量：{a.shape}')
    return a.reshape(-1)


def latest(root, required):
    choices = sorted((p for p in root.glob('run_*')
                      if all((p / k).is_file() for k in required)), reverse=True)
    if not choices:
        raise FileNotFoundError(f'找不到包含{required}的结果目录：{root}')
    return choices[0]


def empty_row(record):
    return dict(battery_id=record['battery_id'], cycle=int(record['cumulative_cycle']),
                source_record_id=record['source_record_id'],
                source_cycle=int(record['source_cycle']), status='', raw_points=0,
                finite_points=0, candidate_charge_points=0, negative_time_steps=0,
                candidate_q_decreases=0, candidate_q_duplicates=0,
                candidate_q_min='', candidate_q_max='', raw_current_min='',
                raw_current_max='', raw_time_span='', raw_qc_span='',
                integral_ratio_if_A_minutes='', integral_ratio_if_Crate_minutes='',
                integral_ratio_if_A_seconds='', integral_ratio_if_Crate_seconds='',
                unit_candidate_only='', best_integral_relative_error='', flags='')


def inspect(file, group, record):
    r = empty_row(record)
    a = {k: vector(file, group, k, int(record['source_cycle']) - 1) for k in FIELDS}
    lengths = {len(x) for x in a.values()}
    if lengths == {0}:
        r.update(status='empty_placeholder', flags='原始空循环，不向后借循环')
        return r, None
    if len(lengths) != 1 or 0 in lengths:
        r.update(status='length_mismatch', flags=str({k: len(v) for k, v in a.items()}))
        return r, None
    r['raw_points'] = len(a['V'])
    finite = np.logical_and.reduce([np.isfinite(a[k]) for k in FIELDS])
    r['finite_points'] = int(finite.sum())
    if finite.sum() < 3:
        r.update(status='insufficient_finite_points')
        return r, None
    flags = []
    r['negative_time_steps'] = int(np.sum(np.diff(a['t']) < 0))
    if r['negative_time_steps']:
        flags.append('源时间回退，禁止按该序列推断单位')
    if not finite.all():
        flags.append('含非有限测量')
    r['raw_current_min'] = float(np.min(a['I'][finite]))
    r['raw_current_max'] = float(np.max(a['I'][finite]))
    r['raw_time_span'] = float(np.ptp(a['t'][finite]))
    qcspan = float(np.ptp(a['Qc'][finite]))
    r['raw_qc_span'] = qcspan
    # 整循环正电流积分仅作单位候选诊断；可能受脉冲和记录处理影响，不能视为授权单位元数据。
    if finite.all() and not r['negative_time_steps'] and qcspan > .05:
        integral = float(np.trapezoid(np.maximum(a['I'], 0), a['t']))
        hypotheses = {
            'A_minutes': integral / 60 / qcspan,
            'Crate_minutes': integral * 1.1 / 60 / qcspan,
            'A_seconds': integral / 3600 / qcspan,
            'Crate_seconds': integral * 1.1 / 3600 / qcspan,
        }
        for key, value in hypotheses.items():
            r['integral_ratio_if_' + key] = value
        best = min(hypotheses, key=lambda key: abs(hypotheses[key] - 1))
        error = abs(hypotheses[best] - 1)
        r['best_integral_relative_error'] = error
        r['unit_candidate_only'] = best if error <= .03 else 'unresolved'
        if error > .03:
            flags.append('电流时间容量关系未能通过3%诊断容差')
    # 初筛：主放电量超过0.01之前、正源电流且充电量为正。
    # 阈值用于可见的候选段诊断，不作为论文充电段截取的最终定义。
    discharge = np.flatnonzero(np.isfinite(a['Qd']) & (a['Qd'] > .01))
    end = int(discharge[0]) if len(discharge) else len(a['Qd'])
    candidate = finite & (np.arange(len(finite)) < end) & (a['I'] > .02) & (a['Qc'] > 1e-6)
    n = int(candidate.sum())
    r['candidate_charge_points'] = n
    if n:
        q = a['Qc'][candidate]
        r['candidate_q_min'], r['candidate_q_max'] = float(q.min()), float(q.max())
        r['candidate_q_decreases'] = int(np.sum(np.diff(q) < -1e-6))
        r['candidate_q_duplicates'] = int(np.sum(np.abs(np.diff(q)) <= 1e-6))
    else:
        flags.append('未识别充电候选点')
    if r['candidate_q_decreases']:
        flags.append('候选充电容量存在回退，后续需明确处理')
    r['status'] = ('candidate_for_review' if n >= 20 and not r['negative_time_steps']
                   else 'needs_review')
    r['flags'] = ';'.join(flags)
    # 不删点、不插值、不归一化；候选数据只用于诊断图。
    return r, {k: v[candidate] for k, v in a.items()}


def draw(path, examples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for label, a in examples:
        for ax, key in zip(axes, ['V', 'I', 'T']):
            ax.plot(a['Qc'], a[key], label=label, linewidth=1)
    for ax, ylabel in zip(axes, ['源电压 V', '源电流 单位待核验', '源温度 单位待核验']):
        ax.set_xlabel('源充电容量 Qc'); ax.set_ylabel(ylabel); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('MATR充电候选段抽查  未归一化  不代表最终截段规则')
    fig.tight_layout()
    fig.savefig(path / 'charge_check.png', dpi=300)
    fig.savefig(path / 'charge_check.svg')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, default=Path('D:/LvShuangJianBing'))
    ap.add_argument('--feature-dir', type=Path)
    ap.add_argument('--summary-dir', type=Path)
    ap.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parent / 'results')
    ap.add_argument('--limit', type=int, default=0, help='只检查前几颗；0检查全部基准电芯')
    ap.add_argument('--cycles', default='all', help='all或如2,10,100；不改变实际循环编号')
    args = ap.parse_args()
    if args.limit < 0:
        raise ValueError('--limit不能小于0')
    wanted = set(range(1, 101)) if args.cycles == 'all' else {int(x) for x in args.cycles.split(',')}
    if not wanted or not wanted <= set(range(1, 101)):
        raise ValueError('只允许实际第1至100循环')
    root = args.project_root.resolve()
    feature = args.feature_dir or latest(root / 'data/features/MATR', ['labels.csv', 'SUCCESS.json'])
    summary = args.summary_dir or latest(root / 'data/processed/MATR', ['cycle_summary.csv', 'SUCCESS.json'])
    labels_path, cycle_path = feature / 'labels.csv', summary / 'cycle_summary.csv'
    labels = [r for r in read_csv(labels_path) if r['training_eligible'] == '1']
    if len({r['battery_id'] for r in labels}) != len(labels) or not labels:
        raise ValueError('标签表为空或身份重复')
    labels.sort(key=lambda r: tuple(map(int, re.fullmatch(r'b(\d+)c(\d+)', r['battery_id']).groups())))
    if args.limit:
        labels = labels[:args.limit]
    ids = {r['battery_id'] for r in labels}
    records = [r for r in read_csv(cycle_path)
               if r['battery_id'] in ids and int(r['cumulative_cycle']) in wanted]
    mapping = {(r['battery_id'], int(r['cumulative_cycle'])): r for r in records}
    if len(mapping) != len(records) or len(mapping) != len(ids) * len(wanted):
        raise ValueError('前100循环映射缺失或重复；停止，不猜测来源')
    paths = [root / 'data/raw' / x for x in FILES]
    before = {str(p): stat(p) for p in paths}
    small_hashes = {str(p): digest(p) for p in [labels_path, cycle_path]}
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = args.output_root / ('check_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    write_json(out / 'config.json', dict(project_root=str(root), feature_dir=str(feature),
               summary_dir=str(summary), selected_cells=[r['battery_id'] for r in labels],
               actual_cycles=sorted(wanted), source_stat=before, input_sha256=small_hashes,
               scope='diagnostic_only_not_training', nominal_Ah_for_hypothesis=1.1))
    rows, cell_rows, examples = [], [], []
    with ExitStack() as stack:
        files = {name: stack.enter_context(h5py.File(p, 'r')) for name, p in zip(FILES, paths)}
        for count, label in enumerate(labels, 1):
            cell = label['battery_id']; rr = []
            for cycle in sorted(wanted):
                record = mapping[(cell, cycle)]
                filename = Path(record['source_file']).name
                if filename not in files:
                    raise ValueError(f'不认识源文件：{filename}')
                f = files[filename]
                index = int(record['source_index_0based'])
                group = f[f['batch']['cycles'][index, 0]]
                try:
                    row, a = inspect(f, group, record)
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    row = empty_row(record); row.update(status='read_error', flags=str(exc))
                    a = None
                rr.append(row); rows.append(row)
                if count == 1 and cycle in {2, 10, 100} and a is not None and len(a['V']):
                    examples.append((f'{cell} 第{cycle}循环', a))
            statuses = Counter(r['status'] for r in rr)
            units = Counter(r['unit_candidate_only'] for r in rr if r['unit_candidate_only'])
            cell_rows.append(dict(battery_id=cell, inspected_cycles=len(rr),
                candidate_cycles=statuses['candidate_for_review'],
                empty_cycles=statuses['empty_placeholder'],
                other_review_cycles=sum(v for k, v in statuses.items() if k not in {'candidate_for_review', 'empty_placeholder'}),
                read_error_cycles=statuses['read_error'],
                unit_hypotheses=json.dumps(dict(units), ensure_ascii=False),
                author_cycle_life=label['author_cycle_life'],
                label_protocol=label['label_protocol']))
            print(f'{count}/{len(labels)} {cell}：候选{statuses["candidate_for_review"]}，'
                  f'空循环{statuses["empty_placeholder"]}，其他需复核{cell_rows[-1]["other_review_cycles"]}', flush=True)
    after = {str(p): stat(p) for p in paths}
    if before != after or any(digest(Path(p)) != h for p, h in small_hashes.items()):
        raise RuntimeError('检查期间源文件发生变化，请停止并核对')
    write_csv(out / 'cycle_check.csv', rows)
    write_csv(out / 'cell_check.csv', cell_rows)
    units = Counter(r['unit_candidate_only'] for r in rows if r['unit_candidate_only'])
    report = dict(status='MATR_curve_audit_complete', cells=len(labels), cycle_records=len(rows),
        status_counts=dict(Counter(r['status'] for r in rows)),
        unit_hypotheses_only=dict(units),
        source_open_mode='read_only', source_size_mtime_unchanged=True,
        input_csv_sha256_unchanged=True, original_mat_sha256_checked=False,
        training_started=False, normalized_tensor_created=False,
        limitations=['单位结果是数值一致性候选，需要与原始资料核对',
                     '候选充电段不是论文最终截段规则',
                     '本次不确认119颗论文名单，也不删除5颗电芯',
                     '诊断阈值不等于训练排除或部署拒判阈值'])
    write_json(out / 'summary.json', report)
    if examples:
        draw(out, examples)
    write_json(out / 'SUCCESS.json', report)
    print('\n检查完成：' + str(out.resolve()))
    print('本次只检查MATR；未训练模型，未修改原始数据或HUST。')
    print('请查看cell_check.csv、cycle_check.csv、summary.json和charge_check.png。')
    print('下一步依据检查结果确认单位与截段规则，再生成曲线张量。')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'\n检查未完成：{exc}', file=sys.stderr)
        raise
