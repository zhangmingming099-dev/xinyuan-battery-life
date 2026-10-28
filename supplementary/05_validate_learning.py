"""Choose a learning rate on the saved first-fold inner split only."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import time
from datetime import datetime
from pathlib import Path


def read_json(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def read_csv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_json(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description='只使用第一折内部验证对比学习率，不评价外层测试集')
    ap.add_argument('--run', type=Path)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    if min(args.epochs, args.patience, args.threads) < 1:
        ap.error('参数必须为正数')
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    root = Path(__file__).resolve().parent
    source = root/'03_knee_train.py'
    spec = importlib.util.spec_from_file_location('curve03', source)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    if args.run is None:
        candidates = [p for p in (root/'models').glob('train_*')
                      if (p/'SUCCESS.json').exists() and (p/'random_fold1/inner_roles.csv').exists()]
        if not candidates:
            raise ValueError('未找到完成的03随机划分结果，请使用--run指定目录')
        args.run = max(candidates, key=lambda p:p.name)
    config = read_json(args.run/'config.json')
    data = Path(config['data_directory'])
    print('正在读取已有曲线；不重新生成膝点、不跑五折。', flush=True)
    if old.sha(data/'curves.npz') != read_json(data/'SUCCESS.json').get('tensor_sha256'):
        raise ValueError('曲线张量指纹不一致')
    inner_path = args.run/'random_fold1/inner_roles.csv'
    outer = read_csv(args.run/'random_fold1/outer_roles.csv')
    inner = read_csv(inner_path)
    allowed = {r['battery_id'] for r in outer if r['role']=='train'}
    if (len({r['battery_id'] for r in inner}) != len(inner)
            or {r['battery_id'] for r in inner} != allowed
            or any(r['role'] not in ('fit','validation') for r in inner)):
        raise ValueError('内部角色重复、缺失或包含外层非训练电芯')
    # Slice first: preprocessing and every target-dependent calculation below
    # use only saved outer-training identities.
    with np.load(data/'curves.npz', allow_pickle=False) as z:
        lookup = {s:i for i,s in enumerate(z['battery_id'].tolist())}
        names = [r['battery_id'] for r in inner]
        ix = [lookup[s] for s in names]
        X = z['X'][ix].copy()
        observed = z['observed'][ix].copy()
        target = z['author_CL'][ix].astype(np.float32)
    fit = np.array([i for i,r in enumerate(inner) if r['role']=='fit'], dtype=int)
    val = np.array([i for i,r in enumerate(inner) if r['role']=='validation'], dtype=int)
    if len(fit)<8 or len(val)<3 or not np.isfinite(target).all():
        raise ValueError('内部样本不足或标签非法')
    bins = int(config['arguments']['charge_bins'])
    prep = old.fit_preprocessing(X,fit)
    values = old.transform(X,observed,prep,bins)
    if not np.isfinite(values).all():
        raise ValueError('预处理后含非法值')
    mean = float(target[fit].mean())
    sd = max(float(target[fit].std()),1.)
    tx = torch.from_numpy(values)
    ty = torch.from_numpy(((target-mean)/sd).astype(np.float32))
    out = root/'results'/('validate_learning_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    seed = int(config['arguments']['seed'])+1
    cfg = dict(source_run=str(args.run), data=str(data), device='cpu', capacity_bins=bins,
        model='TECAN direct CL; same architecture/dropout/clipping as 03',
        learning_rates=[1e-4,3e-4,1e-3], seed=seed, batch_size=8,
        epochs=args.epochs, patience=args.patience, threads=args.threads,
        split=inner, preprocessing_fit='inner fit only', selection='minimum inner validation MSE',
        target_mean=mean,target_std=sd,source03_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        inner_roles_sha256=old.sha(inner_path),tensor_sha256=old.sha(data/'curves.npz'),
        warning='Exploratory hyperparameter diagnostic; no outer calibration/test metrics. Not an unbiased performance estimate.')
    save_json(out/'config.json',cfg)
    save_json(out/'preprocessing.json',prep)
    baseline_median=float(np.median(target[fit]))
    baseline_mae=float(np.abs(target[val]-baseline_median).mean())
    baseline_mean_mse=float(np.mean(((target[val]-mean)/sd)**2))
    print(f'内部拟合{len(fit)}颗，内部验证{len(val)}颗；容量轴{bins}点。',flush=True)
    print(f'训练中位数基线：内部验证MAE={baseline_mae:.2f}循环；训练均值基线标准化MSE={baseline_mean_mse:.5f}',flush=True)
    print('共3个学习率，每组最多150轮（默认），30轮无改善会提前停止；每轮显示耗时。',flush=True)
    results=[]
    for lr in cfg['learning_rates']:
        old.seed_all(torch,seed)
        net=old.build_network(torch,True,bins)
        optimizer=torch.optim.Adam(net.parameters(),lr=lr)
        rng=np.random.default_rng(seed)
        best=float('inf'); stale=0; best_epoch=0
        run_dir=out/f'lr_{lr:g}'
        run_dir.mkdir()
        start=time.perf_counter()
        print(f'\n开始学习率 {lr:g}',flush=True)
        with (run_dir/'loss.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['epoch','elapsed_seconds','train_mse_with_dropout',
                'fit_mse_eval','validation_mse','validation_mae_cycles','validation_prediction_range_cycles',
                'max_gradient_norm_before_clipping'])
            writer.writeheader()
            for epoch in range(1,args.epochs+1):
                net.train();total=0.;count=0;max_grad=0.
                for batch in np.array_split(rng.permutation(fit),math.ceil(len(fit)/8)):
                    optimizer.zero_grad(set_to_none=True)
                    loss=((net(tx[batch])-ty[batch])**2).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError('损失非有限，请发送日志')
                    loss.backward()
                    gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                    max_grad=max(max_grad,float(gn))
                    optimizer.step()
                    total+=float(loss.detach())*len(batch);count+=len(batch)
                net.eval()
                def evaluate(indices):
                    with torch.no_grad():
                        return torch.cat([net(tx[b]) for b in np.array_split(indices,math.ceil(len(indices)/8))])
                pf=evaluate(fit);pv=evaluate(val)
                fm=float(((pf-ty[fit])**2).mean())
                vm=float(((pv-ty[val])**2).mean())
                physical=pv.numpy()*sd+mean
                mae=float(np.abs(physical-target[val]).mean())
                if not all(math.isfinite(t) for t in (fm,vm,mae)):
                    raise RuntimeError('验证损失非有限，请发送日志')
                if vm<best-1e-6:
                    best=vm;best_epoch=epoch;stale=0
                    best_predictions=physical.copy()
                    best_fit_mse=fm
                    torch.save(dict(state_dict=net.state_dict(),epoch=epoch,lr=lr,
                        target_mean=mean,target_std=sd,capacity_bins=bins,
                        purpose='inner-fit checkpoint only; not deployment model'),run_dir/'best_inner.pt')
                else:
                    stale+=1
                elapsed=time.perf_counter()-start
                writer.writerow(dict(epoch=epoch,elapsed_seconds=elapsed,
                    train_mse_with_dropout=total/count,fit_mse_eval=fm,validation_mse=vm,
                    validation_mae_cycles=mae,validation_prediction_range_cycles=float(np.ptp(physical)),
                    max_gradient_norm_before_clipping=max_grad));f.flush()
                print(f'  {epoch}/{args.epochs}轮 拟合MSE={fm:.4f} 验证MSE={vm:.4f} '
                      f'验证MAE={mae:.2f} 最佳轮={best_epoch} 已用{elapsed/60:.1f}分钟',flush=True)
                if stale>=args.patience:
                    print('  内部验证提前停止。',flush=True)
                    break
        result=dict(lr=lr,best_epoch=best_epoch,epochs_run=epoch,seconds=elapsed,
            fit_mse_at_best_epoch=best_fit_mse,best_validation_mse=best,
            validation_mae_cycles=float(np.abs(best_predictions-target[val]).mean()),
            validation_prediction_range_cycles=float(np.ptp(best_predictions)))
        results.append(result)
        with (run_dir/'inner_validation_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['battery_id','true_CL','prediction_CL_at_best_mse'])
            w.writerows((names[i],float(target[i]),float(p)) for i,p in zip(val,best_predictions))
        save_json(out/'partial_summary.json',results)
        del net,optimizer
    selected=min(results,key=lambda r:r['best_validation_mse'])
    report=dict(status='inner_learning_rate_check_complete',results=results,selected=selected,
        baseline_validation_mae_cycles=baseline_mae,baseline_mean_validation_mse=baseline_mean_mse,
        selection_metric='inner validation MSE; MAE is descriptive only',
        outer_test_evaluated=False,outer_calibration_evaluated=False,
        note='Do not treat the winning validation score as final performance or transfer this rate automatically to CTK models.')
    save_json(out/'SUCCESS.json',report)
    print('\n第一折内部验证汇总（不是五折最终精度）：',flush=True)
    print(f'训练中位数基线：MAE={baseline_mae:.2f}循环',flush=True)
    for r in results:
        print(f"lr={r['lr']:g}：最佳轮{r['best_epoch']}，拟合MSE={r['fit_mse_at_best_epoch']:.4f}，"
              f"验证MSE={r['best_validation_mse']:.4f}，验证MAE={r['validation_mae_cycles']:.2f}循环",flush=True)
    print(f"按内部验证MSE选择：lr={selected['lr']:g}",flush=True)
    print('结果目录：'+str(out),flush=True)
    print('把末尾汇总发来；暂不自动启动五折或覆盖03结果。',flush=True)


if __name__=='__main__':
    main()
