"""MATR膝点标注与曲线网络比较。依据论文方法重实现，不是作者源码或精确复现。"""
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import sys
from importlib.metadata import version
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from threadpoolctl import threadpool_limits

PAPER = '10.3390/sym17030351'
NOMINAL_AH = 1.1
MODELS = ['tcn', 'tecan', 'model2']
NAMES = {'tcn': 'TCN直接寿命', 'tecan': 'TECAN直接寿命',
         'model2': 'TECAN膝点辅助', 'median': '训练中位数'}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def serial(obj):
    if isinstance(obj, dict): return {str(k): serial(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [serial(v) for v in obj]
    if isinstance(obj, np.ndarray): return serial(obj.tolist())
    if isinstance(obj, np.generic): return serial(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj): return None
    return obj


def write_json(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(serial(obj), f, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows):
    if not rows: return
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        w.writerows(serial(rows))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def latest_curves(root):
    for p in sorted(root.glob('curves_*'), reverse=True):
        s = p / 'SUCCESS.json'
        if s.is_file() and json.loads(s.read_text(encoding='utf-8')).get('status') == 'MATR_curve_tensor_complete':
            return p
    raise FileNotFoundError(f'未找到02生成的完整曲线目录：{root}')


def bacon(params, x):
    a0, a1, a2, x0 = params
    z = x-x0
    return a0 + a1*z + a2*z*np.tanh(z/1e-8)


def fit_knee(cycles, capacity, author_cl):
    """单电芯离线标签。x归一到[0,1]，Q除1.1；坐标选择与质量界限均显式记录。"""
    row = dict(status='unfitted', CTK='', n_points=0, polynomial_rmse_Ah='',
               bacon_rmse_Ah='', knot_cycle='', initial_slope='', final_slope='',
               valid_root_count=0, all_root_cycles='', reason='')
    cc, qq = np.asarray(cycles, float), np.asarray(capacity, float)
    ok = np.isfinite(cc) & np.isfinite(qq) & (qq > 0) & (cc <= author_cl)
    cc, qq = cc[ok], qq[ok]
    order = np.argsort(cc); cc, qq = cc[order], qq[order]
    row['n_points'] = len(cc)
    if len(cc) < 100 or np.any(np.diff(cc) <= 0) or cc[-1]-cc[0] < 100:
        row.update(status='invalid', reason='轨迹太短或循环编号重复'); return row, None
    x0, span = float(cc[0]), float(cc[-1]-cc[0])
    x = (cc-x0)/span; y = qq/NOMINAL_AH
    poly = np.polynomial.Polynomial.fit(x, y, 5).convert()
    poly_rmse = float(np.sqrt(np.mean((poly(x)-y)**2))*NOMINAL_AH)
    row['polynomial_rmse_Ah'] = poly_rmse
    # 多初值只优化该电芯的拟合损失，不依据CL-CTK相关性选择结果。
    fits = []
    for knot in [.2, .35, .5, .65, .8, .9]:
        start = [float(np.clip(poly(knot),-.999,1.999)), -.1, -.1, knot]
        try:
            fit = least_squares(lambda p: bacon(p, x)-y, start,
                    bounds=([-1., -10., -10., .05], [2., 10., 0., .95]),
                    loss='soft_l1', f_scale=.005, max_nfev=1500)
        except (ValueError, FloatingPointError):
            continue
        if fit.success and np.isfinite(fit.x).all():
            fits.append(fit)
    if not fits:
        row.update(status='invalid', reason='Bacon-Watts拟合不收敛'); return row, None
    fit = min(fits, key=lambda f: f.cost)
    a0, a1, a2, knot = map(float, fit.x)
    beta1, beta2 = a1-a2, a1+a2
    bw_rmse = float(np.sqrt(np.mean((bacon(fit.x, x)-y)**2))*NOMINAL_AH)
    row.update(bacon_rmse_Ah=bw_rmse, knot_cycle=x0+knot*span,
               initial_slope=beta1, final_slope=beta2)
    # 论文式12的数值稳定等价式，避免beta1+beta2接近0时相消。
    A, S = 1-beta1*beta2, beta1+beta2
    denominator = A+math.sqrt(A*A+S*S)
    if abs(denominator) < 1e-12:
        row.update(status='invalid', reason='角平分线数值退化'); return row, None
    k = -S/denominator
    roots = (poly-np.polynomial.Polynomial([a0-k*knot, k])).roots()
    rr = sorted(float(z.real) for z in roots if abs(z.imag) < 1e-7 and 0 < z.real < 1)
    row['valid_root_count'] = len(rr)
    row['all_root_cycles'] = ';'.join(f'{x0+r*span:.6f}' for r in rr)
    problems = []
    if poly_rmse > .03: problems.append('五次拟合RMSE超过0.03Ah')
    if bw_rmse > .03: problems.append('Bacon-Watts拟合RMSE超过0.03Ah')
    if beta2 >= beta1-.01 or beta2 >= -.02: problems.append('未识别明确的后期加速衰减')
    if knot <= .051 or knot >= .949: problems.append('转折参数贴近拟合边界')
    if len(rr) != 1: problems.append('角平分线与五次曲线有效交点不是唯一')
    info = dict(cycles=cc, capacity=qq, x=x, polynomial=poly.coef,
                bacon_params=fit.x, coordinate_origin=x0, coordinate_span=span,
                bisector_slope=k, bisector_intercept=a0-k*knot)
    if problems:
        row.update(status='needs_review', reason=';'.join(problems)); return row, info
    ctk = x0+rr[0]*span
    if not 0 < ctk < author_cl:
        row.update(status='needs_review', reason='CTK不在寿命范围内'); return row, info
    row.update(status='valid_by_rule', CTK=float(ctk), reason='自动规则通过，仍需人工图检')
    return row, info


def make_knees(data, out, ids, cls, config):
    source = Path(config['summary_dir']) / 'cycle_summary.csv'
    feature_success = Path(config['feature_dir']) / 'SUCCESS.json'
    if not source.is_file() or not feature_success.is_file():
        raise FileNotFoundError('原有汇总或特征SUCCESS缺失；不能重建膝点')
    old_hash = config.get('input_sha256', {}).get(str(source))
    if old_hash and sha(source) != old_hash: raise ValueError('源汇总与02记录的指纹不一致')
    raw_hash = {str(p): sha(p) for p in [source, feature_success]}
    masks = {tuple(x) for x in json.loads(feature_success.read_text(encoding='utf-8')).get('masked_capacity_points', [])}
    traces = {i: [] for i in ids}
    masked = Counter()
    for r in read_csv(source):
        cell = r['battery_id']
        if cell not in traces: continue
        cycle = int(r['cumulative_cycle'])
        if (cell, cycle) in masks:
            masked[cell] += 1; continue
        value = r['QDischarge_clean_Ah']
        if value != '': traces[cell].append((cycle, float(value)))
    rows, details = [], {}
    plot_dir = out/'knee_plots'; plot_dir.mkdir()
    for j, (cell, cl) in enumerate(zip(ids, cls), 1):
        pairs = traces[cell]
        with threadpool_limits(limits=1):
            row, detail = fit_knee([p[0] for p in pairs], [p[1] for p in pairs], float(cl))
        rows.append(dict(battery_id=cell, author_CL=float(cl), known_masked_points=masked[cell], **row))
        if detail is not None:
            details[cell] = {k: v for k, v in detail.items() if k not in {'cycles', 'capacity', 'x'}}
        plot_knee(plot_dir/f'{cell}.png', cell, row, detail)
        print(f'膝点 {j}/{len(ids)} {cell}：{row["status"]}', flush=True)
    if any(sha(Path(p)) != h for p, h in raw_hash.items()): raise RuntimeError('膝点处理期间源输入发生变化')
    write_csv(out/'knee_labels.csv', rows)
    write_json(out/'knee_fits.json', details)
    write_json(out/'knee_label_config.json', dict(source_hashes=raw_hash,
        coordinates='x=(cycle-first_valid)/(last_valid-first_valid); y=Q/1.1',
        polynomial_degree=5, bacon_eta=1e-8, bacon_loss='soft_l1', f_scale=.005,
        fit_rmse_limit_Ah=.03, require_unique_root=True, label_is_algorithmic_proxy=True,
        full_trajectory_used_for_label_only=True, known_masks=sorted(masks),
        not_paper_exact_CTK=True))
    return rows


def plot_knee(path, cell, row, info):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(7.5,4))
    if info is not None:
        cc, xx = info['cycles'], info['x']
        ax.scatter(cc, info['capacity'], s=2, alpha=.4, label='历史容量')
        ax.plot(cc, np.polynomial.polynomial.polyval(xx, info['polynomial'])*1.1, label='五次拟合')
        ax.plot(cc, bacon(info['bacon_params'], xx)*1.1, label='Bacon–Watts拟合')
        if row['CTK'] != '': ax.axvline(row['CTK'], color='r', linestyle='--', label='算法膝点标签')
        ax.legend(fontsize=8)
    ax.set_title(f'{cell}  {row["status"]}'); ax.set_xlabel('实际循环'); ax.set_ylabel('放电容量 Ah')
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)


def train_mapping(ctk, cl):
    ctk, cl = np.asarray(ctk), np.asarray(cl)
    if len(ctk) < 3 or np.std(ctk) < 1: raise ValueError('训练CTK不足或无变化')
    a, b = np.linalg.lstsq(np.column_stack([ctk, np.ones(len(ctk))]), cl, rcond=None)[0]
    if not np.isfinite([a,b]).all() or a <= 0: raise ValueError('训练集CTK到CL映射非正，停止MODEL2')
    return float(a), float(b)


def fit_preprocessing(X, indices):
    # 仅训练索引拟合，每通道标量插补及0-1缩放，不利用全生命周期或测试统计。
    a = X[indices]
    flat = a.transpose(1,0,2,3).reshape(3,-1)
    lo, hi, med = np.nanmin(flat,axis=1), np.nanmax(flat,axis=1), np.nanmedian(flat,axis=1)
    if not np.isfinite([lo,hi,med]).all(): raise ValueError('训练通道全缺失或非法')
    scale = np.where(hi-lo > 1e-8, hi-lo, 1.)
    return dict(min=lo.tolist(), scale=scale.tolist(), median=med.tolist())


def transform(X, observed, prep, bins):
    if X.shape[-1] % bins: raise ValueError('charge-bins必须整除容量采样点数')
    med=np.array(prep['median'],np.float32)[None,:,None,None]
    lo=np.array(prep['min'],np.float32)[None,:,None,None]
    scale=np.array(prep['scale'],np.float32)[None,:,None,None]
    z=(np.where(np.isfinite(X),X,med)-lo)/scale  # 不裁剪测试越界值。
    b,c,n,p=z.shape
    z=z.reshape(b,c,n,bins,p//bins).mean(-1).transpose(0,1,3,2)
    quality=observed.reshape(b,n,bins,p//bins).mean(-1).transpose(0,2,1)[:,None]
    return np.concatenate([z,quality],axis=1).astype(np.float32)


def build_network(torch, attention, bins):
    nn=torch.nn
    class Causal(nn.Module):
        def __init__(self,cin,cout,d):
            super().__init__(); self.pad=7*d
            self.conv=nn.utils.parametrizations.weight_norm(nn.Conv2d(cin,cout,(1,8),dilation=(1,d)))
        def forward(self,x):
            return self.conv(torch.nn.functional.pad(x,(self.pad,0,0,0)))
    class ECA(nn.Module):
        def __init__(self):
            super().__init__(); self.conv=nn.Conv1d(1,1,5,padding=2,bias=False)
        def forward(self,x):
            score=x.mean((2,3)).unsqueeze(1)
            weights=torch.sigmoid(self.conv(score)).squeeze(1)[:,:,None,None]
            return x*weights
    class Block(nn.Module):
        def __init__(self,cin,cout,d):
            super().__init__()
            self.branch=nn.Sequential(Causal(cin,cout,d),nn.LeakyReLU(.1),nn.Dropout(.1),
                                      Causal(cout,cout,d),nn.LeakyReLU(.1),nn.Dropout(.1))
            self.attention=ECA() if attention else nn.Identity()
            self.skip=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
            self.act=nn.LeakyReLU(.1)
        def forward(self,x): return self.act(self.attention(self.branch(x))+self.skip(x))
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            # 三个测量通道加质量mask，经可学习1×1融合为论文时间分支的单通道。
            # 具体融合及mask属于显式实现差异，不冒充作者源码。
            self.fusion=nn.Conv2d(4,1,1)
            self.blocks=nn.Sequential(Block(1,16,1),Block(16,32,2),Block(32,64,5))
            self.head=nn.Linear(64,1)
        def forward(self,x):
            z=self.blocks(self.fusion(x))
            return self.head(z[:,:,:,-1].mean(2)).squeeze(1)
    return Net()


def seed_all(torch, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def fit_network(torch, values, target, train, validation, attention, args,
                device, seed, epochs, phase):
    seed_all(torch,seed)
    net=build_network(torch,attention,args.charge_bins).to(device)
    optimizer=torch.optim.Adam(net.parameters(),lr=args.lr)
    mean, sd=float(target[train].mean()),float(target[train].std())
    sd=max(sd,1.)
    x=torch.from_numpy(values); y=torch.from_numpy(((target-mean)/sd).astype(np.float32))
    rng=np.random.default_rng(seed); logs=[]; best=float('inf'); best_epoch=1; stale=0
    for epoch in range(1,epochs+1):
        net.train(); total=0.; count=0
        for batch in np.array_split(rng.permutation(train), math.ceil(len(train)/args.batch_size)):
            bx=x[batch].to(device); by=y[batch].to(device)
            optimizer.zero_grad(set_to_none=True); loss=torch.mean((net(bx)-by)**2)
            if not torch.isfinite(loss): raise RuntimeError('训练损失非有限，停止本次运行')
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            optimizer.step(); total+=float(loss.detach().cpu())*len(batch);count+=len(batch)
        val=None
        if validation is not None:
            net.eval()
            with torch.no_grad():
                preds=torch.cat([net(x[b].to(device)).cpu() for b in
                    np.array_split(validation,math.ceil(len(validation)/args.batch_size))])
                val=float(torch.mean((preds-y[validation])**2))
            if not math.isfinite(val): raise RuntimeError('内部验证损失非有限')
            if val < best-1e-6: best=val;best_epoch=epoch;stale=0
            else: stale+=1
        logs.append(dict(phase=phase,epoch=epoch,train_mse_normalized=total/count,
                         inner_validation_mse_normalized=val))
        if epoch==1 or epoch%25==0:
            print(f'  {phase} {epoch}/{epochs} 训练MSE={total/count:.4f}'+
                  (f' 内部验证MSE={val:.4f}' if val is not None else ''),flush=True)
        if validation is not None and stale>=args.patience: break
    return net,dict(target_mean=mean,target_std=sd),best_epoch,logs


def predict(torch, net, values, indices, scaling, device, batch_size):
    net.eval(); result=[]
    with torch.no_grad():
        for batch in np.array_split(indices,math.ceil(len(indices)/batch_size)):
            v=net(torch.from_numpy(values[batch]).to(device)).cpu().numpy()
            result.extend((v*scaling['target_std']+scaling['target_mean']).tolist())
    result=np.asarray(result)
    if len(result)!=len(indices) or not np.isfinite(result).all():
        raise RuntimeError('预测长度异常或出现非有限值，停止而不生成虚假指标')
    return result


def inner_split(train, cells, folds, fold, experiment, seed):
    if experiment=='protocol':
        group_lookup={r['battery_id']:r['protocol_group'] for r in folds if int(r['fold'])==fold}
        groups=[group_lookup[cells[i]] for i in train]
        if len(set(groups))<3: raise ValueError('训练协议组不足，不能独立内部验证')
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(train,groups=groups))
        fit,val=train[a],train[b]
    else:
        fit,val=train_test_split(train,test_size=.2,random_state=seed)
    if len(fit)<8 or len(val)<3: raise ValueError('内部训练或验证样本不足，先复核膝点标签')
    return np.asarray(fit),np.asarray(val)


def interval(cal_truth, cal_prediction, truth, prediction):
    k=math.ceil((len(cal_truth)+1)*.9)
    if k>len(cal_truth):
        return [None]*len(truth),[None]*len(truth),None
    radius=float(np.sort(np.abs(cal_truth-cal_prediction))[k-1])
    return np.maximum(100,prediction-radius).tolist(),(prediction+radius).tolist(),radius


def evaluate_rows(experiment,fold,model,indices,ids,cl,ctk,pred,raw,lower,upper,ctk_pred=None):
    result=[]
    for j,i in enumerate(indices):
        covered=(bool(lower[j]<=cl[i]<=upper[j]) if lower[j] is not None else None)
        result.append(dict(experiment=experiment,fold=fold,model=model,battery_id=ids[i],
            true_CL=float(cl[i]),true_RUL100=float(cl[i]-100),
            algorithmic_true_CTK=(float(ctk[i]) if np.isfinite(ctk[i]) else None),
            prediction_CL=float(pred[j]),prediction_RUL100=float(pred[j]-100),
            prediction_CL_unclipped=float(raw[j]),prediction_CTK=(float(ctk_pred[j]) if ctk_pred is not None else None),
            error_cycles=float(pred[j]-cl[i]),lower_CL=lower[j],upper_CL=upper[j],covered=covered))
    return result


def summarize(rows):
    result=[]
    for exp,model in sorted({(r['experiment'],r['model']) for r in rows}):
        rr=[r for r in rows if r['experiment']==exp and r['model']==model]
        errors=np.array([r['error_cycles'] for r in rr]); ci=[r for r in rr if r['covered'] is not None]
        kk=[r for r in rr if r['prediction_CTK'] is not None and r['algorithmic_true_CTK'] is not None]
        result.append(dict(experiment=exp,model=model,n_predictions=len(rr),
            n_unique_cells=len({r['battery_id'] for r in rr}),
            MAE_cycles=float(np.mean(np.abs(errors))),RMSE_cycles=float(np.sqrt(np.mean(errors**2))),
            bias_cycles=float(errors.mean()),
            CL_MAPE_percent=float(np.mean([abs(r['error_cycles'])/r['true_CL'] for r in rr])*100),
            interval_evaluated_cells=len(ci),
            SCP_coverage=(float(np.mean([r['covered'] for r in ci])) if ci else None),
            SCP_mean_width=(float(np.mean([r['upper_CL']-r['lower_CL'] for r in ci])) if ci else None),
            CTK_evaluable_cells=len(kk),
            CTK_MAE=(float(np.mean([abs(r['prediction_CTK']-r['algorithmic_true_CTK']) for r in kk])) if kk else None)))
    return result


def plot_results(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    for exp in sorted({r['experiment'] for r in rows}):
        models=sorted({r['model'] for r in rows if r['experiment']==exp and r['model']!='median'})
        fig,axes=plt.subplots(1,len(models),figsize=(5*len(models),4.5),squeeze=False)
        for ax,model in zip(axes[0],models):
            rr=[r for r in rows if r['experiment']==exp and r['model']==model]
            ax.scatter([r['true_CL'] for r in rr],[r['prediction_CL'] for r in rr],s=15)
            top=max([r['true_CL'] for r in rr]+[r['prediction_CL'] for r in rr])*1.05
            ax.plot([0,top],[0,top],'--',color='gray');ax.set_xlim(0,top);ax.set_ylim(0,top)
            ax.set_title(NAMES[model]);ax.set_xlabel('作者基准总寿命 循环');ax.set_ylabel('折外预测总寿命 循环');ax.grid(alpha=.2)
        fig.suptitle(f'{exp} 相同训练标签子集  全部留出测试电芯  方法重实现')
        fig.tight_layout();fig.savefig(out/f'prediction_{exp}.png',dpi=300)
        fig.savefig(out/f'prediction_{exp}.svg');plt.close(fig)


def self_test():
    cc=np.arange(1,1001,dtype=float);xx=(cc-1)/999
    cap=bacon([.91,-.15,-.13,.65],xx)*1.1
    with threadpool_limits(limits=1):
        row,_=fit_knee(cc,cap,1001.)
    assert row['status']=='valid_by_rule',row
    assert 500<row['CTK']<800
    bad,_=fit_knee(np.arange(10),np.ones(10),500.)
    assert bad['status']=='invalid'
    a,b=train_mapping(np.array([100.,200.,300.]),np.array([180.,330.,480.]))
    assert np.allclose([a,b],[1.5,30.])
    X=np.ones((3,3,4,8),np.float32);X[0]*=2;X[1]*=3;X[2]*=1000
    mask=np.ones((3,4,8),bool);X[0,:,0,:]=np.nan;mask[0,0,:]=False
    prep=fit_preprocessing(X,[0,1]); assert np.allclose(prep['min'],[2,2,2])
    z=transform(X,mask,prep,4); assert z.shape==(3,4,4,4) and np.isfinite(z).all()
    assert z[2,:3].max()>1 # 测试越界不裁剪；测试值不参与缩放拟合。
    lo,hi,rad=interval(np.arange(8),np.zeros(8),np.array([5.]),np.array([5.]))
    assert rad is None and lo==[None]
    print('PASS：人工轨迹膝点、训练内映射、训练内插补缩放、越界保留、小校准集处理。')
    try:
        import torch
    except ImportError:
        print('未安装PyTorch：网络前向／反向尚未检查。');return
    torch.set_num_threads(2)
    for attention in [False,True]:
        net=build_network(torch,attention,4)
        output=net(torch.randn(2,4,4,20));assert output.shape==(2,)
        output.square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    print('PASS：TCN和TECAN人工张量前向／反向。此检查不训练真实电池数据。')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path)
    ap.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parent/'models')
    ap.add_argument('--experiment',choices=['random','protocol','all'],default='random')
    ap.add_argument('--models',default='tcn,tecan,model2')
    ap.add_argument('--folds',default='1,2,3,4,5')
    ap.add_argument('--epochs',type=int,default=450)
    ap.add_argument('--patience',type=int,default=50)
    ap.add_argument('--lr',type=float,default=1e-5)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--charge-bins',type=int,default=128)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--labels-only',action='store_true')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test: self_test();return
    if min(args.epochs,args.patience,args.batch_size,args.charge_bins,args.threads)<1 or args.lr<=0:
        raise ValueError('训练参数必须为正数')
    models=args.models.split(','); folds_to_run=[int(x) for x in args.folds.split(',')]
    if not set(models)<=set(MODELS) or not models or len(set(models))!=len(models): raise ValueError('模型参数非法')
    if not set(folds_to_run)<={1,2,3,4,5} or len(set(folds_to_run))!=len(folds_to_run):raise ValueError('折参数非法')
    torch=None
    if not args.labels_only:
        try: import torch
        except ImportError: raise RuntimeError('先安装PyTorch，或用--labels-only仅生成膝点标签。参见配套说明。')
        torch.set_num_threads(args.threads)
        if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('当前PyTorch无法使用CUDA')
        device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
        print(f'训练设备：{device}；容量轴汇聚到{args.charge_bins}点；最大{args.epochs}轮',flush=True)
        # 在长任务前用人工张量检查本机PyTorch接口；不使用真实数据。
        for attention in [False,True]:
            check=build_network(torch,attention,4)
            check(torch.randn(2,4,4,20)).square().mean().backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in check.parameters()):
                raise RuntimeError('网络预检梯度异常')
            del check
        print('本机网络前向／反向预检通过。',flush=True)
    data=args.data or latest_curves(Path(__file__).resolve().parent/'results')
    success=json.loads((data/'SUCCESS.json').read_text(encoding='utf-8'))
    if success.get('status')!='MATR_curve_tensor_complete':raise ValueError('不是02的完整结果目录')
    config=json.loads((data/'config.json').read_text(encoding='utf-8'))
    inputs={str(data/f):sha(data/f) for f in ['curves.npz','labels.csv','folds_random.csv','folds_protocol.csv','config.json','SUCCESS.json']}
    if success.get('tensor_sha256')!=inputs[str(data/'curves.npz')]:raise ValueError('曲线张量指纹不一致')
    with np.load(data/'curves.npz',allow_pickle=False) as z:
        X=z['X'];obs=z['observed'];ids=z['battery_id'].tolist();cl=z['author_CL']
    if X.shape[:3]!=(len(ids),3,100) or obs.shape!=(len(ids),100,X.shape[-1]):raise ValueError('曲线张量维度异常')
    if len(set(ids))!=len(ids) or not np.array_equal(np.isfinite(X),np.broadcast_to(obs[:,None],X.shape)):
        raise ValueError('身份重复或缺失mask不一致')
    if X.shape[-1]%args.charge_bins:raise ValueError('charge-bins必须整除1024')
    out=args.output_root/('train_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.json',dict(source_paper=PAPER,data_directory=str(data),arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        python=platform.python_version(),numpy=np.__version__,torch=(torch.__version__ if torch else None),
        dependencies={k:version(k) for k in ['numpy','scipy','scikit-learn','matplotlib']},
        input_sha256=inputs,selection_uses_outer_calibration=False,selection_uses_outer_test=False,
        method='paper-inspired reimplementation; no MCCA spatial branch',
        deviations=['normalized-coordinate robust CTK algorithm with explicit failure rules',
                    'source units retained; train-only scalar minmax and median imputation',
                    'fixed capacity-axis mean pooling before encoder; default128, optional1024',
                    'observed fraction extra channel and learnable 4-to-1 fusion',
                    'LeakyReLU .1, dropout .1, last-time mean-over-capacity head',
                    'internal early stopping then fresh outer-train refit; training-only CL mapping'],
        scope='exploratory; previous outer tests already seen; common CTK-qualified training subset; unfiltered outer calibration and test'))
    knee_rows=make_knees(data,out,ids,cl,config)
    status_counts=dict(Counter(r['status'] for r in knee_rows))
    print('\n膝点标签汇总：'+str(status_counts),flush=True)
    write_json(out/'LABELS_DONE.json',dict(status='algorithmic_knee_labels_complete',counts=status_counts,
        needs_visual_review=True,not_independently_measured_knee=True))
    if args.labels_only:
        print('仅标注完成：'+str(out.resolve()));return
    # 为隔离膝点辅助作用，各网络使用相同的CTK可标注训练子集。
    # 校准／测试电芯全部保留，禁止根据未来CTK标注质量筛选评测对象。
    good=np.array([r['status']=='valid_by_rule' for r in knee_rows])
    ctk=np.array([float(r['CTK']) if r['CTK']!='' else np.nan for r in knee_rows])
    lookup={cell:i for i,cell in enumerate(ids)}
    write_csv(out/'knee_label_review.csv',[r for r in knee_rows if r['status']!='valid_by_rule'])
    experiments=['random','protocol'] if args.experiment=='all' else [args.experiment]
    all_results=[];fold_metrics=[]
    for experiment in experiments:
        original=read_csv(data/f'folds_{experiment}.csv')
        for fold in folds_to_run:
            source_rows=[r for r in original if int(r['fold'])==fold]
            if len({r['battery_id'] for r in source_rows})!=len(source_rows):raise ValueError('划分身份重复')
            kept=[r for r in source_rows if r['role']!='train' or good[lookup[r['battery_id']]]]
            roles={role:np.array([lookup[r['battery_id']] for r in kept if r['role']==role],dtype=int)
                   for role in ['train','calibration','test']}
            train,cal,test=roles['train'],roles['calibration'],roles['test']
            if len(train)<12 or len(cal)<1 or len(test)<1:raise ValueError('膝点有效子集不足，已保存标签和图，请先复核；不自动放宽规则')
            if set(train)&set(cal) or set(train)&set(test) or set(cal)&set(test):raise ValueError('角色泄漏')
            if experiment=='protocol':
                groups={role:{r['protocol_group'] for r in kept if r['role']==role} for role in roles}
                if groups['train']&groups['calibration'] or groups['train']&groups['test'] or groups['calibration']&groups['test']:
                    raise ValueError('协议组跨角色')
            fit,val=inner_split(train,ids,original,fold,experiment,args.seed+fold)
            fold_dir=out/f'{experiment}_fold{fold}';fold_dir.mkdir()
            write_csv(fold_dir/'outer_roles.csv',kept)
            write_csv(fold_dir/'inner_roles.csv',[dict(battery_id=ids[i],role=role) for role,ii in [('fit',fit),('validation',val)] for i in ii])
            prep_inner=fit_preprocessing(X,fit);prep_outer=fit_preprocessing(X,train)
            inner_values=transform(X,obs,prep_inner,args.charge_bins)
            final_values=transform(X,obs,prep_outer,args.charge_bins)
            write_json(fold_dir/'preprocessing.json',dict(inner=prep_inner,outer=prep_outer))
            a,b=train_mapping(ctk[train],cl[train])
            write_json(fold_dir/'ctk_to_cl.json',dict(a=a,b=b,fitted_battery_ids=[ids[i] for i in train]))
            baseline=np.full(len(test),np.median(cl[train]));baseline_cal=np.full(len(cal),np.median(cl[train]))
            lo,hi,rad=interval(cl[cal],baseline_cal,cl[test],baseline)
            fold_rows=evaluate_rows(experiment,fold,'median',test,ids,cl,ctk,baseline,baseline,lo,hi)
            print(f'\n{experiment} 第{fold}折：共同标签子集训练{len(train)}／完整校准{len(cal)}／完整测试{len(test)}',flush=True)
            for model in models:
                target=ctk if model=='model2' else cl
                attention=model!='tcn'
                print(f'开始 {NAMES[model]}，仅内部验证选择轮数',flush=True)
                net,_,best,logs=fit_network(torch,inner_values,target,fit,val,attention,args,
                    device,args.seed+fold,args.epochs,'内部验证')
                del net
                write_csv(fold_dir/f'{model}_inner_loss.csv',logs)
                write_json(fold_dir/f'{model}_selection.json',dict(selected_epochs=best,
                    rule='minimum inner validation MSE with early stopping',outer_test_used=False))
                net,scale,_,logs=fit_network(torch,final_values,target,train,None,attention,args,
                    device,args.seed+fold+10000,best,'训练折重训')
                write_csv(fold_dir/f'{model}_refit_loss.csv',logs)
                pred0=predict(torch,net,final_values,test,scale,device,args.batch_size)
                cal0=predict(torch,net,final_values,cal,scale,device,args.batch_size)
                if model=='model2':
                    knee_prediction=np.maximum(0,pred0)
                    raw=a*knee_prediction+b;raw_cal=a*np.maximum(0,cal0)+b
                else: knee_prediction=None;raw=pred0;raw_cal=cal0
                pred=np.maximum(100,raw);cal_pred=np.maximum(100,raw_cal)
                lower,upper,radius=interval(cl[cal],cal_pred,cl[test],pred)
                rr=evaluate_rows(experiment,fold,model,test,ids,cl,ctk,pred,raw,lower,upper,knee_prediction)
                fold_rows.extend(rr)
                checkpoint=dict(state_dict={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                    parameter_count=sum(p.numel() for p in net.parameters()),
                    architecture=dict(attention=attention,charge_bins=args.charge_bins,channels=[16,32,64],
                                      dilation=[1,2,5],kernel=[1,8],fusion_inputs=4),
                    target_scaling=scale,preprocessing=prep_outer,model=model,
                    ctk_mapping=dict(a=a,b=b) if model=='model2' else None,
                    input_window=100,selected_epochs=best)
                torch.save(checkpoint,fold_dir/f'{model}.pt')
                write_json(fold_dir/f'{model}_interval.json',dict(method='SCP',target_coverage=.9,
                    calibration_cells=len(cal),radius=radius,
                    available=radius is not None,reason_if_unavailable='90%有限半径至少需要9颗校准电芯'))
                print(f'完成 {NAMES[model]}：测试MAE={np.mean(np.abs(pred-cl[test])):.2f}循环，内部选择{best}轮',flush=True)
                del net
            write_csv(fold_dir/'predictions.csv',fold_rows)
            all_results.extend(fold_rows);fold_metrics.extend([dict(fold=fold,**r) for r in summarize(fold_rows)])
            del inner_values,final_values
    write_csv(out/'predictions.csv',all_results);write_csv(out/'fold_metrics.csv',fold_metrics)
    metrics=summarize(all_results);write_csv(out/'summary.csv',metrics)
    plot_results(out,all_results)
    if any(sha(Path(p))!=value for p,value in inputs.items()):raise RuntimeError('训练期间曲线输入发生变化')
    write_json(out/'SUCCESS.json',dict(status='MATR_curve_network_comparison_complete',
        evaluated_experiments=experiments,evaluated_folds=folds_to_run,models=models,
        is_full_five_fold=sorted(folds_to_run)==[1,2,3,4,5],original_cells=len(ids),
        common_knee_qualified_cells=int(good.sum()),knee_label_status_counts=status_counts,
        source_unchanged=True,metrics=metrics,CTK_is_algorithmic_proxy=True,
        risk_rejection_evaluated=False,independent_blind_test=False))
    print('\n汇总（相同训练标签子集，完整留出测试，单位循环）：')
    for r in metrics:print(f'{r["experiment"]}/{NAMES[r["model"]]}：电芯{r["n_unique_cells"]}，MAE={r["MAE_cycles"]:.2f}，RMSE={r["RMSE_cycles"]:.2f}')
    print('结果目录：'+str(out.resolve()))
    print('论文位置：3.1.3膝点标注与3.2模型方法重实现；已附新模型SCP，风险复核尚未重新评价。')


if __name__=='__main__':
    main()
