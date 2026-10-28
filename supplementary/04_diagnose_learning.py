"""Small training-only diagnostic, not a model-performance experiment."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import time
from datetime import datetime
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    ap = argparse.ArgumentParser(description='第一折训练子集学习能力检查；不进行测试集评价')
    ap.add_argument('--run', type=Path, default=None, help='03已完成的训练目录')
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--seconds', type=float, default=120, help='每个学习率的软时间上限')
    ap.add_argument('--bins', type=int, default=32)
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    if args.steps < 1 or args.seconds <= 0 or args.bins < 1 or args.threads < 1:
        ap.error('参数必须为正数')
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    root = Path(__file__).resolve().parent
    source = root / '03_knee_train.py'
    if not source.exists():
        raise FileNotFoundError('请把04与03_knee_train.py放在同一文件夹')
    spec = importlib.util.spec_from_file_location('curve03', source)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    if args.run is None:
        runs = [p for p in (root/'models').glob('train_*')
                if (p/'SUCCESS.json').exists() and (p/'random_fold1/outer_roles.csv').exists()]
        if not runs:
            raise ValueError('没有找到已完成的random训练，请用--run指定目录')
        args.run = max(runs, key=lambda p: p.name)
    config = read_json(args.run/'config.json')
    data = Path(config['data_directory'])
    print('读取已有张量；不重新截曲线、不生成膝点、不跑五折。', flush=True)
    expected = read_json(data/'SUCCESS.json').get('tensor_sha256')
    if not expected or old.sha(data/'curves.npz') != expected:
        raise ValueError('张量指纹不一致，停止检查')
    with (args.run/'random_fold1/outer_roles.csv').open(encoding='utf-8-sig', newline='') as f:
        roles = list(csv.DictReader(f))
    train_ids = sorted(r['battery_id'] for r in roles if r['role'] == 'train')
    if len(train_ids) < 8 or len(set(train_ids)) != len(train_ids):
        raise ValueError('训练身份不足或重复')
    with np.load(data/'curves.npz', allow_pickle=False) as z:
        ids = z['battery_id'].tolist()
        lookup = {s:i for i,s in enumerate(ids)}
        ix = [lookup[s] for s in train_ids]
        # 只取训练角色；不读取留出集标签用于诊断。
        X = z['X'][ix].copy()
        observed = z['observed'][ix].copy()
        labels = z['author_CL'][ix].copy()
    if X.shape[-1] % args.bins:
        raise ValueError('bins需要整除1024')
    if not np.isfinite(labels).all():
        raise ValueError('训练寿命标签含非法数值')
    order = np.argsort(labels)
    chosen = order[np.linspace(0, len(order)-1, 8).astype(int)]
    # 选择覆盖训练寿命范围的8颗电芯；这只是记忆能力检查，不估计泛化。
    raw = X[chosen]
    obs = observed[chosen]
    target = labels[chosen]
    if float(target.std()) < 1:
        raise ValueError('训练子集寿命差异不足')
    prep = old.fit_preprocessing(raw, np.arange(8))
    values = old.transform(raw, obs, prep, args.bins)
    if not np.isfinite(values).all():
        raise ValueError('预处理后出现NaN或无穷')
    out = root/'results'/('diagnose_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    selected_ids = [train_ids[i] for i in chosen]
    cfg = dict(source_run=str(args.run), source_data=str(data), device='cpu',
               selected_training_ids=selected_ids, original_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               arguments=vars(args)|{'run':str(args.run)},
               purpose='training-only memorization diagnostic, no held-out evaluation',
               deviations=['8 training cells chosen by training label range',
                           '32 capacity bins by default, versus previous 128',
                           'dropout disabled to check memorization; same TECAN architecture otherwise',
                           'full batch; no gradient clipping so raw gradient is visible'],
               warning='Passing does not prove predictive accuracy; failure does not identify a unique cause.')
    (out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'只使用第一折训练中的8颗电芯：{selected_ids}',flush=True)
    print(f'容量轴{args.bins}点；每个学习率最多{args.steps}步／约{args.seconds:g}秒，完成当前步后检查时间。',flush=True)
    print('两次学习率对照均禁用Dropout，仅诊断能否记住训练样本。',flush=True)
    print('预处理后各测量通道的样本间平均标准差：',
          [round(float(values[:,c].std(axis=0).mean()),6) for c in range(3)],flush=True)
    tx = torch.from_numpy(values)
    ty = torch.from_numpy(((target-target.mean())/target.std()).astype(np.float32))
    outcomes=[]
    for lr in (1e-5,1e-3):
        old.seed_all(torch,2026)
        net=old.build_network(torch,True,args.bins)
        # eval disables dropout; gradients remain enabled during optimization.
        net.eval()
        opt=torch.optim.Adam(net.parameters(),lr=lr)
        with torch.no_grad():
            initial=float(((net(tx)-ty)**2).mean())
        start=time.perf_counter()
        timed_out=False
        print(f'\n学习率 {lr:g}：初始标准化MSE={initial:.6f}',flush=True)
        path=out/f'learning_lr_{lr:g}.csv'
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['step','elapsed_seconds','loss_before_update',
                'loss_after_update','gradient_norm','prediction_std_cycles','prediction_range_cycles'])
            writer.writeheader()
            for step in range(1,args.steps+1):
                opt.zero_grad(set_to_none=True)
                pred=net(tx)
                loss=((pred-ty)**2).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('损失非有限，停止；请发送已有日志')
                loss.backward()
                grads=[p.grad for p in net.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads):
                    raise RuntimeError('梯度缺失或非有限，停止；请发送已有日志')
                norm=math.sqrt(sum(float(g.square().sum()) for g in grads))
                opt.step()
                with torch.no_grad():
                    current=net(tx)
                    after=float(((current-ty)**2).mean())
                    pc=current.numpy()*float(target.std())+float(target.mean())
                if not math.isfinite(after):
                    raise RuntimeError('参数更新后损失非有限')
                elapsed=time.perf_counter()-start
                row=dict(step=step,elapsed_seconds=elapsed,loss_before_update=float(loss.detach()),
                    loss_after_update=after,gradient_norm=norm,
                    prediction_std_cycles=float(pc.std()),prediction_range_cycles=float(np.ptp(pc)))
                writer.writerow(row);f.flush()
                print(f'  {step}/{args.steps}步 MSE={after:.5f} 梯度={norm:.3g} '
                      f'预测跨度={np.ptp(pc):.2f}循环 已用{elapsed:.1f}秒',flush=True)
                if elapsed>=args.seconds:
                    timed_out=True
                    print('  已到本组时间上限，保存结果并结束本组。',flush=True)
                    break
        outcome=dict(lr=lr,steps=step,seconds=elapsed,initial_mse=initial,final_mse=after,
            reduction_fraction=(initial-after)/max(initial,1e-12),
            prediction_range_cycles=float(np.ptp(pc)),true_range_cycles=float(np.ptp(target)),
            timed_out=timed_out,
            diagnostic_only_mse_below_0_1=after<.1)
        outcomes.append(outcome)
        with (out/f'predictions_lr_{lr:g}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['training_cell','true_CL','diagnostic_prediction_CL'])
            w.writerows(zip(selected_ids,target.tolist(),pc.tolist()))
        (out/'summary.json').write_text(json.dumps(outcomes,ensure_ascii=False,indent=2),encoding='utf-8')
        del net,opt
    print('\n诊断完成；这不是五折评测，不能用作模型最终精度。',flush=True)
    for r in outcomes:
        print(f"lr={r['lr']:g}：MSE {r['initial_mse']:.5f} → {r['final_mse']:.5f}，"
              f"预测跨度{r['prediction_range_cycles']:.2f}循环，完成{r['steps']}步",flush=True)
    print('结果目录：'+str(out),flush=True)
    print('把末尾汇总发来；先不要重新跑五折。',flush=True)


if __name__=='__main__':
    main()
