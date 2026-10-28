"""Exploratory inner-only matched-update comparison of capacity-axis residual convolutions."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from datetime import datetime


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def shape_network(torch, old, bins, state):
    class ShapeNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = old.build_network(torch, False, bins)
            self.trunk.load_state_dict(state)
            self.local = torch.nn.Sequential(
                torch.nn.Conv2d(4, 8, (5, 1), padding=(2, 0)),
                torch.nn.LeakyReLU(.1),
                torch.nn.Conv2d(8, 4, (5, 1), padding=(2, 0)))
            torch.nn.init.zeros_(self.local[-1].weight)
            torch.nn.init.zeros_(self.local[-1].bias)
        def forward(self, x):
            # Residual shape features may take any real value; they are not a new observed mask.
            return self.trunk(x + self.local(x))
    return ShapeNet()


def self_test(torch, old):
    torch.manual_seed(104)
    base = old.build_network(torch, False, 12).eval()
    state = {k:v.clone() for k,v in base.state_dict().items()}
    shape = shape_network(torch, old, 12, state).eval()
    x = torch.randn(3, 4, 12, 20)
    with torch.no_grad():
        torch.testing.assert_close(base(x), shape(x), rtol=0, atol=0)
    before = shape.local[-1].weight.detach().clone()
    opt = torch.optim.Adam(shape.parameters(), lr=.001)
    for _ in range(2):
        opt.zero_grad(); loss = (shape(x)-torch.tensor([-1., .5, 2.])).square().mean()
        loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in shape.parameters())
        opt.step()
    assert not torch.equal(before, shape.local[-1].weight)
    assert float(shape.local[0].weight.grad.abs().sum()) > 0
    shape.eval()
    # Shuffle capacity locations; the newly trained local operator can distinguish adjacency.
    with torch.no_grad():
        delta = float((shape(x)-shape(x[:, :, torch.randperm(12), :])).abs().max())
    assert delta > 1e-7, '局部模块未表现出容量轴邻接敏感性'
    # No temporal mixing in the new operator.
    other = x.clone(); other[..., 10:] += 10
    with torch.no_grad():
        torch.testing.assert_close(shape.local(x)[..., :10], shape.local(other)[..., :10], rtol=0, atol=0)
    print('PASS：恒等起点、局部模块梯度和更新、容量邻接敏感性、局部模块无跨循环混合。', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, help='已完成的08结果，默认最近完成目录')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--lr', type=float, default=.0001)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.epochs < 1 or args.threads < 1 or args.lr <= 0: ap.error('轮数、线程和学习率必须为正')
    import numpy as np
    import torch
    root = Path(__file__).resolve().parent
    old = load_module(root/'03_knee_train.py', 'shape03')
    warm = load_module(root/'08_warm_tecan.py', 'shape08')
    torch.set_num_threads(args.threads)
    if args.self_test: self_test(torch, old); return
    if args.run is None:
        runs = [p for p in (root/'results').glob('warm_tecan_*') if (p/'SUCCESS.json').exists()]
        if not runs: raise RuntimeError('未找到完成的08实验，请与原有03、08代码放在D:\\xinyuanzhice同一层')
        args.run = max(runs, key=lambda p:p.name)
    source = args.run.resolve()
    if not (source/'SUCCESS.json').exists(): raise RuntimeError('08实验尚未完成')
    cfg = warm.json_read(source/'config.json')
    for name, key in [('03_knee_train.py', 'original_model_sha256'), ('08_warm_tecan.py', 'script_sha256')]:
        if warm.sha(root/name) != cfg[key]: raise RuntimeError(name+'与生成08结果时的代码不一致')
    original = Path(cfg['source']); repeat = int(cfg['repeat'])
    outer = [r for r in warm.csv_read(original/'outer_splits.csv') if int(r['repeat']) == repeat]
    roles = cfg['roles']; warm.checked_roles(roles, outer)
    data = Path(warm.json_read(original/'config.json')['data'])
    if warm.sha(data/'curves.npz') != cfg['tensor_sha256']: raise RuntimeError('曲线数据指纹变化')
    print('读取内部拟合79颗、验证20颗；不使用外层测试标签。', flush=True)
    with np.load(data/'curves.npz', allow_pickle=False) as z:
        lookup = {s:i for i,s in enumerate(z['battery_id'].tolist())}
        ii = [lookup[r['battery_id']] for r in roles]
        raw = z['X'][ii].copy(); observed = z['observed'][ii].copy()
        target = z['author_CL'][ii].astype(float)
    fit = np.array([i for i,r in enumerate(roles) if r['role']=='fit'])
    val = np.array([i for i,r in enumerate(roles) if r['role']=='validation'])
    if (len(fit),len(val)) != (79,20): raise RuntimeError('内部样本数量异常')
    prep = old.fit_preprocessing(raw, fit)
    saved = warm.json_read(source/'preprocessing.json')
    if any(not np.allclose(prep[k], saved['input'][k], atol=1e-7, rtol=0) for k in prep):
        raise RuntimeError('预处理参数不一致')
    bins = int(cfg['capacity_bins']); values = old.transform(raw, observed, prep, bins)
    del raw, observed
    if not np.isfinite(values).all(): raise RuntimeError('输入存在非有限值')
    scaling = dict(mean=float(target[fit].mean()), std=max(float(target[fit].std()),1.))
    if not np.allclose([scaling['mean'],scaling['std']], [saved['target']['mean'],saved['target']['std']]):
        raise RuntimeError('目标缩放不一致')
    x = torch.from_numpy(values)
    y = torch.from_numpy(((target-scaling['mean'])/scaling['std']).astype('float32'))
    checkpoint = source/'tcn_control'/'latest.pt'
    base_state = torch.load(checkpoint, map_location='cpu', weights_only=True)['best_state']
    def base_maker():
        net = old.build_network(torch, False, bins); net.load_state_dict(base_state); return net
    def shape_maker(): return shape_network(torch, old, bins, base_state)
    base = base_maker().eval(); shaped = shape_maker().eval()
    with torch.no_grad(): torch.testing.assert_close(base(x[:2]),shaped(x[:2]),rtol=0,atol=0)
    baseline_score, _ = warm.evaluate(torch,base,x,y,val,target,scaling)
    del base,shaped
    config = dict(source=str(source), checkpoint_sha256=warm.sha(checkpoint),
        source03_sha256=warm.sha(root/'03_knee_train.py'),source08_sha256=warm.sha(root/'08_warm_tecan.py'),
        script_sha256=warm.sha(Path(__file__)), tensor_sha256=cfg['tensor_sha256'],
        roles=roles, bins=bins, epochs=args.epochs, lr=args.lr, threads=args.threads,
        seed=20261022+repeat, torch_version=str(torch.__version__), numpy_version=np.__version__,
        model='TCN plus zero-output residual capacity-axis conv4-8-4 kernels5x1',
        selection='minimum internal normalized lifetime MSE, including epoch0',
        limitation='reused internal validation; exploratory, not independent performance; equal update counts, unequal parameter counts')
    signature = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    matches = [p for p in (root/'results').glob('capacity_shape_*')
               if (p/'config.json').exists() and warm.json_read(p/'config.json').get('signature')==signature]
    out = max(matches,key=lambda p:p.name) if matches else root/'results'/('capacity_shape_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=True)
    lock = warm.lock_run(out)
    warm.json_write(out/'config.json',dict(signature=signature,**config))
    warm.json_write(out/'preprocessing.json',dict(input=prep,target=scaling))
    if (out/'SUCCESS.json').exists():
        print((out/'summary.txt').read_text(encoding='utf-8-sig')); return
    print(f'恒等起点检查通过。两个模型各追加{args.epochs}轮；每轮保存，可关闭后重新双击继续。', flush=True)
    print('结果目录：'+str(out), flush=True)
    phases=[dict(name='全部参数联合训练',epochs=args.epochs,freeze_trunk=False,trunk_lr=args.lr,attention_lr=args.lr)]
    rows=[]; predictions=[]
    for name,maker in [('tcn_control',base_maker),('capacity_shape_tcn',shape_maker)]:
        print('\n训练 '+name,flush=True)
        net,result=warm.run_stage(torch,old,out/name,maker,phases,x,y,target,fit,val,scaling,
                                  config['seed'],signature)
        fs,_=warm.evaluate(torch,net,x,y,fit,target,scaling)
        vs,p=warm.evaluate(torch,net,x,y,val,target,scaling)
        rows.append(dict(model=name,fit_MAE=fs['mae'],validation_MAE=vs['mae'],validation_MSE=vs['mse'],
                         best_epoch=result['best_epoch'],epochs_run=result['epochs_run'],
                         parameters=sum(t.numel() for t in net.parameters())))
        predictions.extend(dict(model=name,battery_id=roles[i]['battery_id'],role='inner_validation',
                                true_CL=float(target[i]),prediction_CL=float(v)) for i,v in zip(val,p))
        del net
    better=rows[1]['validation_MSE']<rows[0]['validation_MSE']-1e-6
    report=dict(status='exploratory_inner_shape_comparison_complete',summary=rows,
        starting_validation=baseline_score,shape_better_by_validation_MSE=better,
        median_validation_MAE=float(np.abs(target[val]-np.median(target[fit])).mean()),
        outer_test_evaluated=False,independent_validation=False,
        warning='Not final five-repeat accuracy. Neither attention benefit nor exact paper reproduction is established.')
    warm.csv_write(out/'summary.csv',rows);warm.csv_write(out/'inner_predictions.csv',predictions)
    lines=['容量轴局部形状内部验证完成（不是最终五次测试精度）。',
           f"共同起点：验证MAE={baseline_score['mae']:.2f}，验证MSE={baseline_score['mse']:.5f}"]
    for r in rows:
        lines.append(f"{r['model']}：拟合MAE={r['fit_MAE']:.2f}，验证MAE={r['validation_MAE']:.2f}，验证MSE={r['validation_MSE']:.5f}，最佳轮={r['best_epoch']}")
    lines += ['容量模块优于同轮数TCN（按内部验证MSE）：'+str(better),
              '最佳轮0表示未超过共同起点。既有内部验证集被重复使用，结果仅作探索。',
              '没有自动启动五次实验，没有覆盖原模型。','结果目录：'+str(out),
              '请发送末尾汇总；若有改善，再决定独立评估方案。']
    text='\n'.join(lines)
    (out/'summary.txt').write_text(text,encoding='utf-8-sig')
    warm.json_write(out/'SUCCESS.json',report)
    print('\n'+text,flush=True)


if __name__=='__main__': main()
