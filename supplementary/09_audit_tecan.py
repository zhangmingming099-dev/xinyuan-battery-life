"""Read-only model/input audit. No optimizer steps, no held-out evaluation."""
import argparse
import importlib.util
import json
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import time


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def module(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def check_structure(torch, old):
    """Synthetic invariance and per-convolution causality checks."""
    torch.manual_seed(91)
    net = old.build_network(torch, True, 8).eval()
    x = torch.randn(2, 4, 8, 30)
    with torch.no_grad():
        p = net(x)
        delta = float((p-net(x.flip(2))).abs().max())
    causal = []
    for block in net.blocks:
        layer = block.branch[0]
        a = torch.randn(1, layer.conv.in_channels, 8, 30)
        b = a.clone(); b[..., 15:] += 3
        with torch.no_grad():
            error = float((layer(a)[..., :15]-layer(b)[..., :15]).abs().max())
        causal.append(error)
    assert max(causal) < 1e-6, '卷积的左侧填充检查失败'
    assert delta < 1e-5, '当前容量轴排列不变性与预期不一致'
    return dict(causal_convolution_future_change_max=causal,
                common_capacity_permutation_output_max=delta,
                note='ECA沿容量与循环维全局池化；卷积因果性不代表中间注意力对前缀严格因果。固定前100循环预测可使用窗口内全部信息。')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, help='08的warm_tecan结果目录；默认最近完成目录')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    import numpy as np
    import torch
    torch.set_num_threads(max(1, args.threads))
    old = module(root/'03_knee_train.py', 'audit_old03')
    warm = module(root/'08_warm_tecan.py', 'audit_warm08')
    structure = check_structure(torch, old)
    if args.self_test:
        print('PASS：合成数据卷积因果性与容量轴排列检查。'); return
    if args.run is None:
        runs = [p for p in (root/'results').glob('warm_tecan_*')
                if (p/'SUCCESS.json').exists()]
        if not runs:
            raise RuntimeError('没有完成的08结果，请先把本代码放到D:\\xinyuanzhice，与03和08放在一起。')
        args.run = max(runs, key=lambda p: p.name)
    run = args.run.resolve()
    if not (run/'SUCCESS.json').exists():
        raise RuntimeError('指定08实验尚未完成')
    cfg = read(run/'config.json')
    if digest(root/'03_knee_train.py') != cfg['original_model_sha256']:
        raise RuntimeError('03代码已变化，不能用不同结构解释已有权重')
    if digest(root/'08_warm_tecan.py') != cfg['script_sha256']:
        raise RuntimeError('08代码与该次实验不一致，请保留生成该结果的原版08代码')
    source = Path(cfg['source'])
    data = Path(read(source/'config.json')['data'])
    if digest(data/'curves.npz') != cfg['tensor_sha256']:
        raise RuntimeError('曲线数据指纹已变化')
    inner = cfg['roles']
    with (source/'outer_splits.csv').open(encoding='utf-8-sig', newline='') as f:
        outer = [r for r in csv.DictReader(f) if int(r['repeat']) == cfg['repeat']]
    warm.checked_roles(inner, outer)
    fit_names = [r['battery_id'] for r in inner if r['role'] == 'fit']
    if len(fit_names) != 79:
        raise RuntimeError('预期79颗内部训练电芯')
    print('读取已有数据和权重；只检查79颗内部训练电芯，梯度探针取前8颗。', flush=True)
    begin = time.perf_counter()
    with np.load(data/'curves.npz', allow_pickle=False) as z:
        lookup = {s:i for i,s in enumerate(z['battery_id'].tolist())}
        ii = [lookup[s] for s in fit_names]
        raw = z['X'][ii].copy(); obs = z['observed'][ii].copy()
        target = z['author_CL'][ii].astype(float)
    saved = read(run/'preprocessing.json')
    fitted = old.fit_preprocessing(raw, np.arange(len(raw)))
    prep_error = max(float(np.max(np.abs(np.array(fitted[k])-saved['input'][k]))) for k in fitted)
    if prep_error > 1e-6:
        raise RuntimeError('重新计算的训练集预处理参数与08保存值不一致')
    bins = int(cfg['capacity_bins'])
    values = old.transform(raw, obs, saved['input'], bins)
    if not np.isfinite(values).all():
        raise RuntimeError('预处理后存在NaN或无穷')
    channel = []
    block = raw.shape[-1]//bins
    for c, name in enumerate(['V', 'I_source_unit', 'T']):
        a = raw[:, c]
        med = saved['input']['median'][c]
        filled = np.where(np.isfinite(a), a, med)
        reconstructed = np.repeat(filled.reshape(len(raw), 100, bins, block).mean(-1), block, axis=-1)
        valid = np.isfinite(a)
        mse = float(np.mean((reconstructed[valid]-a[valid])**2))
        variance = float(np.var(a[valid]))
        lo = saved['input']['min'][c]; scale = saved['input']['scale'][c]
        channel.append(dict(channel=name, finite_fraction=float(valid.mean()),
            raw_min=float(np.nanmin(a)), raw_max=float(np.nanmax(a)),
            scaled_min=float(np.nanmin((a-lo)/scale)), scaled_max=float(np.nanmax((a-lo)/scale)),
            binning_RMSE_source_unit=float(np.sqrt(mse)),
            binning_MSE_over_total_variance=mse/max(variance, 1e-12)))
    n = min(8, len(values)); x = torch.from_numpy(values[:n].copy())
    scaling = saved['target']
    y = torch.tensor(((target[:n]-scaling['mean'])/scaling['std']).astype('float32'))
    base_state = torch.load(run/'pretrain'/'latest.pt', map_location='cpu', weights_only=True)['best_state']
    reports = []; gates = []; gradients = []
    def make_base():
        net = old.build_network(torch, False, bins); net.load_state_dict(base_state); return net
    base = make_base().eval()
    identity = warm.warm_network(torch, old, bins, base_state).eval()
    with torch.no_grad():
        initial_difference = float((base(x[:2])-identity(x[:2])).abs().max())
    assert initial_difference < 1e-6
    del identity, base
    for stage, attention in [('pretrain', False), ('tcn_control', False), ('identity_tecan', True)]:
        print('检查 '+stage+'：前向、输入敏感性、梯度。', flush=True)
        net = warm.warm_network(torch, old, bins, base_state) if attention else make_base()
        ckpt = torch.load(run/stage/'latest.pt', map_location='cpu', weights_only=True)
        net.load_state_dict(ckpt['best_state']); net.eval()
        with torch.no_grad():
            p = net(x)
            qperm = net(x.flip(2))
            reverse = net(x.flip(3))
            probes = {}
            for c, name in enumerate(['V', 'I', 'T', 'mask']):
                altered = x.clone(); altered[:, c] = x[:, c].mean()
                probes[name] = float((net(altered)-p).abs().mean()*scaling['std'])
            if attention:
                for i, b in enumerate(net.blocks):
                    # Capture actual pre-attention feature maps, without retaining large tensors.
                    def hook(mod, inp, out, index=i):
                        score = inp[0].mean((2,3)).unsqueeze(1)
                        gate = 2*torch.sigmoid(mod.conv(score))
                        gates.append(dict(block=index, min=float(gate.min()), max=float(gate.max()),
                            mean=float(gate.mean()), std=float(gate.std(unbiased=False)),
                            near_extreme_fraction=float(((gate<.02)|(gate>1.98)).float().mean())))
                    b.attention.register_forward_hook(hook)
                net(x)
                # Disable attention only for a counterfactual forward pass, without changing files.
                original = [b.attention for b in net.blocks]
                for b in net.blocks: b.attention = torch.nn.Identity()
                ablated = net(x)
                for b, a in zip(net.blocks, original): b.attention = a
                for a in original: a._forward_hooks.clear()
                bypass = float((p-ablated).abs().mean()*scaling['std'])
            else: bypass = None
        net.zero_grad(set_to_none=True)
        loss = ((net(x[:2])-y[:2])**2).mean()
        loss.backward()
        for name, parameter in net.named_parameters():
            g = parameter.grad
            gradients.append(dict(stage=stage, parameter=name, has_gradient=g is not None,
                gradient_finite=None if g is None else bool(torch.isfinite(g).all()),
                gradient_norm=None if g is None else float(g.norm())))
        reports.append(dict(stage=stage, best_epoch=int(ckpt['best_epoch']),
            fit_probe_n=n, fit_probe_MAE=float((p*scaling['std']+scaling['mean']-torch.tensor(target[:n])).abs().mean()),
            prediction_span_cycles=float((p.max()-p.min())*scaling['std']),
            capacity_reversal_change_max_cycles=float((p-qperm).abs().max()*scaling['std']),
            cycle_reversal_change_mean_cycles=float((p-reverse).abs().mean()*scaling['std']),
            replace_channel_with_mean_change_cycles=probes,
            bypass_attention_change_mean_cycles=bypass,
            finite_probe_loss=bool(torch.isfinite(loss))))
        del net, ckpt
    out = root/'results'/('audit_tecan_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    report = dict(source=str(run), input_shape=list(values.shape), checked_cells=fit_names,
        preprocessing_max_error=prep_error, channel_audit=channel, structure=structure,
        identity_initial_output_max_error=initial_difference, models=reports, gates=gates,
        elapsed_seconds=time.perf_counter()-begin,
        trained=False, outer_test_evaluated=False, validation_evaluated=False,
        limitations=['训练探针敏感性不是特征因果贡献或最终预测精度。',
          '容量轴排列不变是当前结构特征，不能单凭此断言造成精度下降。',
          '梯度非零只能证明反向传播可达，不能证明训练或泛化充分。',
          '固定Qc对齐不直接保留原始充电时间；本脚本不补充时间输入。',
          '未核验作者完整实现；当前模型无完整空间分支，不属于论文精确复现。'])
    (out/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for filename, rows in [('gradients.csv', gradients), ('channels.csv', channel)]:
        with (out/filename).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ['模型与输入专项检查完成（没有训练、没有使用外层测试集）。',
             f'训练集预处理重算最大误差：{prep_error:.8g}',
             f'恒等注意力初始输出差：{initial_difference:.8g}',
             '容量轴排列检查：见audit.json；当前结构预期对共同排列基本不敏感。']
    for r in reports:
        lines.append(f"{r['stage']}：训练探针预测跨度{r['prediction_span_cycles']:.2f}循环，倒序循环输出平均变化{r['cycle_reversal_change_mean_cycles']:.2f}循环")
    bad = [g for g in gradients if g['gradient_finite'] is False or not g['has_gradient']]
    lines += [f'缺失或非有限梯度参数数：{len(bad)}（具体参数见gradients.csv）',
              '这些检查不产生新的最终精度，也不能据此保证TECAN超过TCN。',
              '结果目录：'+str(out), '请把本段汇总和audit.json发回来。']
    text = '\n'.join(lines)
    (out/'summary.txt').write_text(text, encoding='utf-8-sig')
    print('\n'+text, flush=True)


if __name__ == '__main__':
    main()
