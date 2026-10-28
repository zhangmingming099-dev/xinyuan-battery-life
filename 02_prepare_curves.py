"""MATR曲线输入准备：论文3.1.2的方法重实现。仅生成未归一化张量，不训练。"""
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

NAMES = ['2017-05-12_batchdata_updated_struct_errorcorrect.mat',
         '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
         '2018-04-12_batchdata_updated_struct_errorcorrect.mat']
FIELDS = ('V', 'I', 'T', 'Qc', 'Qd', 't')


def csv_read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    if not rows:
        raise ValueError(f'没有记录：{path}')
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def json_write(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_stat(path):
    s = path.stat(); return dict(size=s.st_size, mtime_ns=s.st_mtime_ns)


def find_run(parent, pattern, required, status=None):
    for p in sorted(parent.glob(pattern), reverse=True):
        if not all((p / f).is_file() for f in required):
            continue
        if status is not None:
            with (p / 'SUCCESS.json').open(encoding='utf-8') as f:
                if json.load(f).get('status') != status:
                    continue
        return p
    raise FileNotFoundError(f'没有匹配结果：{parent}/{pattern}')


def read_vector(f, group, field, index):
    refs = group[field]
    if refs.ndim != 2 or refs.shape[1] != 1 or index >= len(refs):
        raise ValueError(f'{field}循环引用异常')
    obj = f[refs[index, 0]]
    if np.asarray(obj.attrs.get('MATLAB_empty', 0)).any():
        return np.array([], dtype=float)
    a = np.asarray(obj[()], dtype=float)
    if a.ndim != 2 or 1 not in a.shape:
        raise ValueError(f'{field}不是向量')
    return a.reshape(-1)


def time_quality(t):
    """区分浮点精度级负差和明显负差，不修改原时间，不排序。"""
    if not np.isfinite(t).all():
        return dict(time_tolerance='', tiny_negative_steps=0, real_negative_steps=0,
                    time_nonfinite=True)
    tolerance = max(1e-10, 64 * np.finfo(float).eps * max(1., float(np.max(np.abs(t)))))
    dt = np.diff(t)
    return dict(time_tolerance=tolerance,
                tiny_negative_steps=int(np.sum((dt < 0) & (dt >= -tolerance))),
                real_negative_steps=int(np.sum(dt < -tolerance)), time_nonfinite=False)


def nearest_indices(q, grid):
    """q严格递增；相等距离时选左点。超范围位置在调用方以mask屏蔽。"""
    right = np.searchsorted(q, grid, side='left').clip(0, len(q) - 1)
    left = (right - 1).clip(0, len(q) - 1)
    return np.where(np.abs(q[right] - grid) < np.abs(q[left] - grid), right, left)


def prepare_cycle(a, grid, max_q_gap=.03):
    n = len(grid)
    x = np.full((3, n), np.nan, dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    info = dict(status='', source_points=len(a['V']), finite_candidate_points=0,
                unique_charge_points=0, charge_q_min='', charge_q_max='',
                true_q_decreases=0, removed_exact_q_duplicates=0,
                valid_grid_points=0, grid_coverage=0., unsupported_internal_gap_points=0, time_tolerance='',
                tiny_negative_steps=0, real_negative_steps=0, time_nonfinite=False)
    if {len(v) for v in a.values()} == {0}:
        info['status'] = 'empty_placeholder'; return x, mask, info
    if len({len(v) for v in a.values()}) != 1 or not len(a['V']):
        info['status'] = 'field_length_mismatch'; return x, mask, info
    info.update(time_quality(a['t']))
    if info['real_negative_steps']:
        info['status'] = 'time_reversal_needs_review'; return x, mask, info
    if info['time_nonfinite']:
        info['status'] = 'time_nonfinite_needs_review'; return x, mask, info
    finite = np.logical_and.reduce([np.isfinite(a[k]) for k in FIELDS])
    # 明示的重实现规则：主放电量超过0.01前，源正电流>0.02且Qc>0。
    # 不声称该阈值就是论文作者的截段定义；保留相应清单供复查。
    dis = np.flatnonzero(np.isfinite(a['Qd']) & (a['Qd'] > .01))
    end = int(dis[0]) if len(dis) else len(a['V'])
    use = finite & (np.arange(len(finite)) < end) & (a['I'] > .02) & (a['Qc'] > 0)
    q = a['Qc'][use]
    info['finite_candidate_points'] = int(len(q))
    if len(q) < 20 or np.ptp(q) < .05:
        info['status'] = 'insufficient_charge_points'; return x, mask, info
    info['true_q_decreases'] = int(np.sum(np.diff(q) < -1e-6))
    if info['true_q_decreases']:
        info['status'] = 'charge_q_reversal_needs_review'; return x, mask, info
    # 小于容量容差的回退按源记录保留到质量记录；不排序乱序轨迹。
    # 仅取严格增长的首个观测，重复或极微小回退不产生重复插值坐标。
    keep = []
    previous = -np.inf
    for i, value in enumerate(q):
        if value > previous:
            keep.append(i); previous = float(value)
    q_keep = q[keep]
    info['removed_exact_q_duplicates'] = int(np.sum(np.diff(q) == 0))
    info['omitted_nonincreasing_q_points'] = len(q) - len(keep)
    info['unique_charge_points'] = len(keep)
    if len(keep) < 20:
        info['status'] = 'insufficient_unique_charge_points'; return x, mask, info
    channels = np.stack([a[k][use][keep] for k in ['V', 'I', 'T']])
    info['charge_q_min'], info['charge_q_max'] = float(q_keep[0]), float(q_keep[-1])
    mask = (grid >= q_keep[0]) & (grid <= q_keep[-1])
    right = np.searchsorted(q_keep, grid, side='left').clip(0, len(q_keep)-1)
    left = (right-1).clip(0, len(q_keep)-1)
    large_gap = ((q_keep[right]-q_keep[left] > max_q_gap)
                 & (grid > q_keep[left]) & (grid < q_keep[right]))
    info['unsupported_internal_gap_points'] = int(np.sum(mask & large_gap))
    mask &= ~large_gap
    nearest = nearest_indices(q_keep, grid[mask])
    x[:, mask] = channels[:, nearest].astype(np.float32)
    info['valid_grid_points'] = int(mask.sum())
    info['grid_coverage'] = float(mask.mean())
    info['status'] = 'aligned' if mask.any() else 'no_supported_grid'
    return x, mask, info


def verified_folds(path, all_ids, protocol=False):
    rows = csv_read(path)
    ids = {r['battery_id'] for r in rows}
    if not ids <= all_ids:
        raise ValueError(f'划分包含标签表以外电芯：{path}')
    folds = sorted({int(r['fold']) for r in rows})
    if folds != [1, 2, 3, 4, 5]:
        raise ValueError('外层划分不是五折')
    test_count = Counter()
    for fold in folds:
        rr = [r for r in rows if int(r['fold']) == fold]
        if len(rr) != len(ids) or {r['battery_id'] for r in rr} != ids:
            raise ValueError('某折身份缺失或重复')
        groups = {role: set() for role in ['train', 'calibration', 'test']}
        for r in rr:
            if r['role'] not in groups:
                raise ValueError('不认识的数据角色')
            if r['role'] == 'test': test_count[r['battery_id']] += 1
            if protocol: groups[r['role']].add(r['protocol_group'])
        if protocol and any(groups[a] & groups[b] for a, b in
                            [('train', 'calibration'), ('train', 'test'), ('calibration', 'test')]):
            raise ValueError('协议组跨角色泄漏')
    if any(test_count[i] != 1 for i in ids):
        raise ValueError('不是每颗电芯恰好测试一次')
    return rows


def plot_preview(path, X, valid, grid, ids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, c, title in zip(axes, range(3), ['源电压', '源电流 单位待核验', '源温度']):
        values = np.ma.array(X[0, c], mask=~valid[0])
        im = ax.imshow(values, origin='lower', aspect='auto',
                       extent=[float(grid[0]), float(grid[-1]), .5, 100.5])
        ax.set_title(title); ax.set_xlabel('充电容量轴 Qc'); ax.set_ylabel('实际循环')
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle(f'{ids[0]} 前100循环曲线输入  空白为未观测  未归一化')
    fig.tight_layout(); fig.savefig(path / 'curves.png', dpi=300)
    fig.savefig(path / 'curves.svg'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, default=Path('D:/LvShuangJianBing'))
    ap.add_argument('--feature-dir', type=Path)
    ap.add_argument('--summary-dir', type=Path)
    ap.add_argument('--random-run', type=Path)
    ap.add_argument('--protocol-run', type=Path)
    ap.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parent / 'results')
    ap.add_argument('--points', type=int, default=1024)
    ap.add_argument('--q-min', type=float, default=.01)
    ap.add_argument('--q-max', type=float, default=1.)
    ap.add_argument('--max-q-gap', type=float, default=.03)
    args = ap.parse_args()
    if args.points < 16 or not (0 <= args.q_min < args.q_max <= 1.5) or not (0 < args.max_q_gap <= 1):
        raise ValueError('网格参数不合法')
    root = args.project_root.resolve()
    feature = args.feature_dir or find_run(root / 'data/features/MATR', 'run_*', ['labels.csv', 'SUCCESS.json'])
    summary = args.summary_dir or find_run(root / 'data/processed/MATR', 'run_*', ['cycle_summary.csv', 'SUCCESS.json'])
    random_run = args.random_run or find_run(root / 'models/MATR', 'yanzheng_*',
                    ['fenzu.csv', 'SUCCESS.json'], 'MATR_cv_interval_risk_complete')
    protocol_run = args.protocol_run or find_run(root / 'models/MATR', 'xieyi_*',
                    ['fenzu.csv', 'SUCCESS.json'], 'MATR_protocol_holdout_complete')
    labels_path, summary_path = feature / 'labels.csv', summary / 'cycle_summary.csv'
    labels = [r for r in csv_read(labels_path) if r['training_eligible'] == '1']
    labels.sort(key=lambda r: tuple(map(int, re.fullmatch(r'b(\d+)c(\d+)', r['battery_id']).groups())))
    ids = [r['battery_id'] for r in labels]
    if not ids or len(set(ids)) != len(ids): raise ValueError('电芯身份为空或重复')
    random_rows = verified_folds(random_run / 'fenzu.csv', set(ids))
    protocol_rows = verified_folds(protocol_run / 'fenzu.csv', set(ids), protocol=True)
    if {r['battery_id'] for r in random_rows} != set(ids):
        raise ValueError('随机划分与当前基准电芯不一致')
    source_paths = [labels_path, summary_path, random_run / 'fenzu.csv', protocol_run / 'fenzu.csv']
    hashes = {str(p): fingerprint(p) for p in source_paths}
    all_records = csv_read(summary_path)
    records = [r for r in all_records if r['battery_id'] in set(ids) and 1 <= int(r['cumulative_cycle']) <= 100]
    mapping = {(r['battery_id'], int(r['cumulative_cycle'])): r for r in records}
    if len(mapping) != len(records) or len(records) != len(ids) * 100:
        raise ValueError('实际前100循环映射缺失或重复，停止，不向后借循环')
    raw_paths = {name: root / 'data/raw' / name for name in NAMES}
    raw_before = {name: source_stat(p) for name, p in raw_paths.items()}
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = args.output_root / ('curves_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    grid = np.linspace(args.q_min, args.q_max, args.points, dtype=np.float64)
    config = dict(paper_section='3.1.2 Input Data Generation', source_paper='10.3390/sym17030351',
        project_root=str(root), feature_dir=str(feature), summary_dir=str(summary),
        random_run=str(random_run), protocol_run=str(protocol_run),
        shape=[len(ids), 3, 100, args.points], channels=['V_source', 'I_source', 'T_source'],
        grid=dict(axis='source_Qc', source_capacity_convention='Ah per existing summary; no rescaling',
                  min=args.q_min, max=args.q_max, points=args.points, interpolation='nearest',
                  choice='fixed a priori; not fitted using test data'),
        current_unit='source values preserved; C-rate hypothesis not promoted to confirmed A',
        time_rule='tiny negative dt within max(1e-10,64*eps*scale) recorded; real reversal blocks cycle',
        phase_rule='before first Qd>0.01; I_source>0.02; Qc>0; finite V/I/T/Qc/Qd/t',
        duplicate_rule='retain first strictly increasing Qc observations; record omitted points',
        missing_rule='NaN plus observed mask; no extrapolation, no missing-cycle fill',
        max_q_gap=args.max_q_gap,
        normalization='none; future training must fit scaling and imputation inside training fold',
        paper_deviations=['119-cell list and author split not reproduced',
            'explicit fixed absolute Qc grid 0.01..1.00 by default; exact paper endpoints unspecified',
            'missingness is explicit; no global normalization or guessed V/I/T fusion',
            'charging-stage thresholds are implementation choices pending source clarification'],
        input_sha256=hashes, raw_stat=raw_before, started_at=datetime.now().isoformat())
    json_write(out / 'config.json', config)
    X = np.full((len(ids), 3, 100, args.points), np.nan, dtype=np.float32)
    observed = np.zeros((len(ids), 100, args.points), dtype=bool)
    cycle_rows, cell_rows = [], []
    with ExitStack() as stack:
        files = {name: stack.enter_context(h5py.File(path, 'r')) for name, path in raw_paths.items()}
        for i, cell in enumerate(ids):
            rr = []
            for cycle in range(1, 101):
                rec = mapping[(cell, cycle)]
                name = Path(rec['source_file']).name
                if name not in files: raise ValueError(f'未知源文件{name}')
                f = files[name]; group = f[f['batch/cycles'][int(rec['source_index_0based']), 0]]
                a = {k: read_vector(f, group, k, int(rec['source_cycle']) - 1) for k in FIELDS}
                x, mask, info = prepare_cycle(a, grid, args.max_q_gap)
                # 固定CSV列，失败循环也保留完整质量记录。
                info.setdefault('omitted_nonincreasing_q_points', 0)
                info = {k: info[k] for k in sorted(info)}
                row = dict(battery_id=cell, cycle=cycle, source_record_id=rec['source_record_id'],
                           source_cycle=rec['source_cycle'], **info)
                cycle_rows.append(row); rr.append(row)
                X[i, :, cycle - 1, :] = x; observed[i, cycle - 1] = mask
            count = Counter(r['status'] for r in rr)
            cell_rows.append(dict(tensor_index=i, battery_id=cell, aligned_cycles=count['aligned'],
                empty_cycles=count['empty_placeholder'], review_cycles=100-count['aligned']-count['empty_placeholder'],
                tiny_negative_time_steps=sum(r['tiny_negative_steps'] for r in rr),
                real_negative_time_steps=sum(r['real_negative_steps'] for r in rr),
                observed_grid_fraction=float(observed[i].mean())))
            print(f'{i+1}/{len(ids)} {cell}：对齐{count["aligned"]}循环，'
                  f'网格有效{observed[i].mean():.1%}，需复核{cell_rows[-1]["review_cycles"]}', flush=True)
    # NaN位置和观测mask必须完全一致。
    if not np.array_equal(np.isfinite(X), np.broadcast_to(observed[:, None], X.shape)):
        raise RuntimeError('观测mask与张量有限值不一致')
    cl = np.array([float(r['author_cycle_life']) for r in labels], dtype=np.float64)
    if not np.isfinite(cl).all() or np.any(cl <= 100): raise ValueError('寿命标签异常')
    np.savez_compressed(out / 'curves.npz', X=X, observed=observed, q_grid=grid,
                        battery_id=np.array(ids, dtype='U20'), cycles=np.arange(1, 101),
                        channels=np.array(config['channels'], dtype='U20'),
                        author_CL=cl, author_RUL100=cl-100)
    csv_write(out / 'labels.csv', labels)
    csv_write(out / 'source_map.csv', sorted(records, key=lambda r: (ids.index(r['battery_id']), int(r['cumulative_cycle']))))
    csv_write(out / 'cycle_quality.csv', cycle_rows)
    csv_write(out / 'cell_quality.csv', cell_rows)
    csv_write(out / 'folds_random.csv', random_rows)
    csv_write(out / 'folds_protocol.csv', protocol_rows)
    if any(source_stat(p) != raw_before[name] for name, p in raw_paths.items()):
        raise RuntimeError('检查期间源MAT文件发生变化')
    if any(fingerprint(Path(p)) != sha for p, sha in hashes.items()):
        raise RuntimeError('检查期间输入CSV发生变化')
    result = dict(status='MATR_curve_tensor_complete', cells=len(ids), tensor_shape=list(X.shape),
        status_counts=dict(Counter(r['status'] for r in cycle_rows)),
        tiny_negative_time_steps=sum(r['tiny_negative_steps'] for r in cycle_rows),
        real_negative_time_steps=sum(r['real_negative_steps'] for r in cycle_rows),
        protocol_cell_count=len({r['battery_id'] for r in protocol_rows}),
        protocol_not_included=sorted(set(ids)-{r['battery_id'] for r in protocol_rows}),
        observed_grid_fraction=float(observed.mean()), source_stat_unchanged=True,
        source_csv_sha256_unchanged=True, original_mat_sha256_checked=False,
        tensor_sha256=fingerprint(out / 'curves.npz'), normalization_fitted=False,
        CTK_labels_created=False, neural_network_trained=False,
        note='数据准备完成；缺失处理和缩放须训练折内拟合；尚非精确论文复现')
    plot_preview(out, X, observed, grid, ids)
    json_write(out / 'summary.json', result); json_write(out / 'SUCCESS.json', result)
    print('\nMATR曲线输入准备完成：' + str(out.resolve()))
    print(f'张量{tuple(X.shape)}；浮点精度级负差{result["tiny_negative_time_steps"]}；明显时间回退{result["real_negative_time_steps"]}')
    print('已有缺失保留NaN及mask；未全局归一化，未训练，未改动原始MAT、旧模型或HUST。')
    print('论文位置：3.1.2输入数据生成；下一模块为3.1.3膝点标注与3.2模型训练。')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'\n处理未完成：{exc}', file=sys.stderr)
        raise
