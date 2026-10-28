"""TCN training/calibration/test development experiment; never reuse a 99-cell checkpoint."""
import argparse
import atexit
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time


def read_json(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_json(path,data):
    p=path.with_suffix('.json.tmp');p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(p,path)
def write_csv(path,rows):
    if not rows:return
    p=path.with_suffix('.csv.tmp')
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    os.replace(p,path)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def load_module(path):
    s=importlib.util.spec_from_file_location('trust03',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def clone(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
def atomic_torch(torch,path,value):
    p=path.with_suffix('.pt.tmp');torch.save(value,p);os.replace(p,path)
def lock(directory):
    f=(directory/'RUNNING.lock').open('a+b');f.seek(0,2)
    if not f.tell():f.write(b'0');f.flush()
    f.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:f.close();raise RuntimeError('相同实验已在其他窗口运行')
    atexit.register(f.close);return f


def conformal_radius(np,residual,alpha):
    a=np.asarray(residual,float)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all() or np.any(a<0):raise ValueError('校准残差非法')
    k=math.ceil((len(a)+1)*(1-alpha))
    if not 1<=k<=len(a):raise ValueError('该校准样本数不足以给出所选名义覆盖率的有限区间')
    return float(np.sort(a)[k-1]),k


def acceptance_threshold(np,scores,rate):
    if not 0<rate<=1:raise ValueError('接受率非法')
    if rate==1:return None
    k=math.ceil(len(scores)*rate)
    return float(np.sort(scores)[k-1])


def role_check(train,cal,test,fit,val):
    t,c,e,f,v=map(set,[train,cal,test,fit,val])
    if t&c or t&e or c&e or f&v or f|v!=t:raise ValueError('训练、校准、测试或内部身份交叉')
    if any(len(a)!=len(set(a)) for a in [train,cal,test,fit,val]):raise ValueError('身份重复')


def fit_score_model(np,values,proper):
    features=score_features(np,values)
    center=np.median(features[proper],axis=0)
    scale=np.maximum(1.4826*np.median(np.abs(features[proper]-center),axis=0),.001)
    return dict(center=center.tolist(),scale=scale.tolist(),
        feature_definition='per V/I/T: first10 mean, last10 mean, std of 100 cycle means, range of cycle means',
        formula='log1p(robust_distance)+5*missing_fraction+5*outside_range_fraction',
        fitted_on='74 proper training cells only',not_a_probability=True)


def score_features(np,values):
    # values: B x 4 x capacity bins x cycles; no labels used.
    a=values[:,:3].mean(axis=2)
    return np.stack([a[:,:,:10].mean(2),a[:,:,-10:].mean(2),a.std(2),np.ptp(a,axis=2)],axis=2).reshape(len(a),12)


def review_scores(np,values,raw,obs,prep,model):
    features=score_features(np,values)
    z=(features-np.array(model['center']))/np.array(model['scale'])
    distance=np.sqrt(np.mean(z*z,axis=1));missing=1-obs.mean(axis=(1,2));outside=[]
    for a in raw:
        n=0;den=0
        for c in range(3):
            finite=np.isfinite(a[c]);v=a[c][finite];lo=prep['min'][c];hi=lo+prep['scale'][c]
            n+=int(((v<lo)|(v>hi)).sum());den+=len(v)
        outside.append(n/max(den,1))
    outside=np.array(outside)
    score=np.log1p(distance)+5*missing+5*outside
    if not np.isfinite(score).all():raise ValueError('复核分数非有限')
    return score,distance,missing,outside


def train(torch,np,old,out,values,target,fit,val,epochs,patience,lr,seed,signature,stop_after=None):
    """Inner selection or exact-epoch proper refit, with per-epoch resumable state."""
    out.mkdir(exist_ok=True);last=out/'latest.pt';done=out/'DONE.json'
    old.seed_all(torch,seed);net=old.build_network(torch,False,values.shape[2])
    optimizer=torch.optim.Adam(net.parameters(),lr=lr);rng=np.random.default_rng(seed)
    scale=dict(target_mean=float(target[fit].mean()),target_std=max(float(target[fit].std()),1.))
    x=torch.from_numpy(values);y=torch.from_numpy(((target-scale['target_mean'])/scale['target_std']).astype('float32'))
    completed=0;best=None;best_epoch=0;best_state=None;stale=0;history=[];elapsed=0.
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=True)
        if ck['signature']!=signature:raise ValueError('训练断点配置不一致')
        net.load_state_dict(ck['state']);optimizer.load_state_dict(ck['optimizer'])
        rng.bit_generator.state=ck['shuffle_state'];torch.set_rng_state(ck['torch_rng'])
        completed=ck['epoch'];best=ck['best'];best_epoch=ck['best_epoch'];best_state=ck['best_state']
        stale=ck['stale'];history=ck['history'];elapsed=ck['elapsed_seconds']
        if done.exists():
            if val is not None:net.load_state_dict(best_state)
            return net,scale,read_json(done)
        print(f'恢复 {out.name}，已完成{completed}轮。',flush=True)
    start=time.perf_counter();previous=elapsed
    for epoch in range(completed+1,epochs+1):
        if val is not None and stale>=patience:break
        net.train();total=0.;count=0
        for batch in np.array_split(rng.permutation(fit),math.ceil(len(fit)/8)):
            optimizer.zero_grad(set_to_none=True);loss=(net(x[batch])-y[batch]).square().mean()
            if not bool(torch.isfinite(loss)):raise RuntimeError('训练损失非有限')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();total+=float(loss.detach())*len(batch);count+=len(batch)
        vm=None;mae=None
        if val is not None:
            net.eval()
            with torch.no_grad():
                vp=torch.cat([net(x[b]) for b in np.array_split(val,math.ceil(len(val)/8))])
                vm=float((vp-y[val]).square().mean())
                physical=vp.numpy()*scale['target_std']+scale['target_mean']
                mae=float(np.abs(physical-target[val]).mean())
            if not math.isfinite(vm):raise RuntimeError('内部验证损失非有限')
            if best is None or vm<best-1e-6:best=vm;best_epoch=epoch;best_state=clone(net);stale=0
            else:stale+=1
        else:best_epoch=epoch
        elapsed=previous+time.perf_counter()-start
        history.append(dict(epoch=epoch,train_mse=total/count,inner_validation_mse=vm,
                            inner_validation_MAE=mae,best_epoch=best_epoch,elapsed_seconds=elapsed))
        ck=dict(signature=signature,state=clone(net),optimizer=optimizer.state_dict(),
                shuffle_state=rng.bit_generator.state,torch_rng=torch.get_rng_state(),epoch=epoch,
                best=best,best_epoch=best_epoch,best_state=best_state,stale=stale,
                history=history,elapsed_seconds=elapsed)
        atomic_torch(torch,last,ck);write_csv(out/'loss.csv',history)
        seconds=(time.perf_counter()-start)/(epoch-completed)
        line=f"{out.name} {epoch}/{epochs}轮 训练MSE={total/count:.4f}"
        if vm is not None:line+=f' 内部验证MSE={vm:.4f} 最佳轮={best_epoch}'
        print(line+f' 已用{elapsed/60:.1f}分钟，该阶段按上限估计剩余{seconds*(epochs-epoch)/60:.1f}分钟',flush=True)
        if stop_after is not None and epoch>=stop_after:return None,None,None
    if val is not None:net.load_state_dict(best_state)
    result=dict(selected_epochs=best_epoch,epochs_run=len(history),best_inner_mse=best,
                elapsed_seconds=elapsed,selection_uses_calibration_or_test=False)
    write_json(done,result);return net,scale,result


def mean_or_none(np,a):return float(np.mean(a)) if len(a) else None


def evaluate_review(np,cal_scores,test_scores,error,covered,lower,upper):
    rows=[];rng=np.random.default_rng(20261022)
    for rate in [.5,.6,.7,.8,.9,1.]:
        threshold=acceptance_threshold(np,cal_scores,rate)
        keep=np.ones(len(error),bool) if threshold is None else test_scores<=threshold
        n=int(keep.sum());random_mae=[]
        if n:
            random_mae=[float(error[rng.choice(len(error),n,replace=False)].mean()) for _ in range(1000)]
        rows.append(dict(target_calibration_acceptance=rate,score_threshold=threshold,
            test_accepted=n,test_total=len(error),actual_test_acceptance=n/len(error),
            accepted_MAE=mean_or_none(np,error[keep]),reviewed_MAE=mean_or_none(np,error[~keep]),
            all_test_MAE=float(error.mean()),random_same_count_MAE_mean=mean_or_none(np,random_mae),
            random_same_count_MAE_std=float(np.std(random_mae,ddof=1)) if random_mae else None,
            accepted_interval90_coverage=mean_or_none(np,covered[keep]),
            accepted_interval90_mean_width=mean_or_none(np,(upper-lower)[keep])))
    return rows


def make_plots(np,out,truth,pred,lower,upper,keep,review):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=np.argsort(truth);x=np.arange(1,len(truth)+1)
    fig,ax=plt.subplots(figsize=(10,5.5))
    ax.errorbar(x,pred[order],yerr=np.stack([pred[order]-lower[order],upper[order]-pred[order]]),
        fmt='o',markersize=4,color='#5375cc',ecolor='#b9c9e8',capsize=3,label='TCN + nominal 90% interval')
    ax.scatter(x,truth[order],marker='x',color='#d76c50',label='Observed lifetime',zorder=4)
    rejected=~keep[order]
    ax.scatter(x[rejected],pred[order][rejected],facecolors='none',edgecolors='#d19933',s=110,label='Suggested review')
    ax.set(xlabel='Test cells sorted by observed lifetime (display only)',ylabel='Cycle life / cycles',
        title='Development split: 74 train / 25 calibration / 25 test')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'test_intervals.png',dpi=180);plt.close(fig)
    rows=[r for r in review if r['accepted_MAE'] is not None]
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot([100*r['actual_test_acceptance'] for r in rows],[r['accepted_MAE'] for r in rows],'-o',color='#5375cc',label='Input-score acceptance')
    ax.plot([100*r['actual_test_acceptance'] for r in rows],[r['random_same_count_MAE_mean'] for r in rows],'--',color='#38aa9e',label='Random same-count subset mean')
    ax.axhline(review[-1]['all_test_MAE'],color='#9d88ba',linestyle=':',label='All test cells')
    ax.set(xlabel='Actual retained test cells / %',ylabel='Retained MAE / cycles',
           title='Thresholds fixed from calibration scores; not selected on test error')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'review_acceptance.png',dpi=180);plt.close(fig)


def self_test(torch,np,old,root):
    import tempfile
    torch.set_num_threads(2)
    q,k=conformal_radius(np,np.arange(1.,26.),.1);assert (q,k)==(24.,24)
    assert conformal_radius(np,np.arange(1.,26.),.05)==(25.,25)
    try:conformal_radius(np,np.array([1.]),.01)
    except ValueError:pass
    else:raise AssertionError('有限校准样本检查失败')
    assert acceptance_threshold(np,np.arange(25.),.8)==19.
    try:role_check([0,1],[1],[2],[0],[1])
    except ValueError:pass
    else:raise AssertionError('身份交叉未拦截')
    role_check([0,1],[2],[3],[0],[1])
    values=np.random.default_rng(2).normal(size=(8,4,4,20)).astype('float32')
    target=np.arange(8.)*100+200;fit=np.arange(6);val=np.arange(6,8)
    sm=fit_score_model(np,values,fit);changed=values.copy();changed[val]*=100
    assert sm==fit_score_model(np,changed,fit), '复核评分统计混入留出身份'
    with tempfile.TemporaryDirectory(dir=root) as td:
        p=Path(td)
        for validation in [val,None]:
            suffix='select' if validation is not None else 'refit'
            kw=dict(torch=torch,np=np,old=old,values=values,target=target,fit=fit,val=validation,
                    epochs=3,patience=3,lr=.001,seed=34,signature=suffix)
            a,_,ra=train(out=p/('full_'+suffix),**kw)
            train(out=p/('resume_'+suffix),stop_after=1,**kw)
            b,_,rb=train(out=p/('resume_'+suffix),**kw)
            for name,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[name],rtol=0,atol=0)
            assert ra['selected_epochs']==rb['selected_epochs']
    print('PASS：有限样本区间分位数、固定80%阈值、身份隔离、选轮与重训断点一致性。',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path);ap.add_argument('--epochs',type=int,default=150)
    ap.add_argument('--patience',type=int,default=30);ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--self-test',action='store_true');ap.add_argument('--fresh',action='store_true')
    args=ap.parse_args()
    if min(args.epochs,args.patience,args.threads)<1:ap.error('轮数、耐心和线程须为正')
    import numpy as np
    import torch
    from sklearn.model_selection import StratifiedShuffleSplit
    root=Path(__file__).resolve().parent;old=load_module(root/'03_knee_train.py')
    if args.self_test:self_test(torch,np,old,root);return
    torch.set_num_threads(args.threads)
    if args.source is None:
        candidates=[p for p in (root/'models').glob('stratified_*') if (p/'SUCCESS.json').exists() and (p/'outer_splits.csv').exists()]
        if not candidates:raise ValueError('未找到完成的07分层实验；请把代码放在D:\\xinyuanzhice')
        args.source=max(candidates,key=lambda p:p.name)
    original=read_json(args.source/'config.json');data=Path(original['data'])
    if sha(root/'03_knee_train.py')!=original['source03_sha256']:raise ValueError('03代码与原分层实验不一致')
    tensor_sha=sha(data/'curves.npz')
    if tensor_sha!=original['tensor_sha256']:raise ValueError('曲线张量指纹不一致')
    original_rows=[r for r in read_csv(args.source/'outer_splits.csv') if int(r['repeat'])==1]
    with np.load(data/'curves.npz',allow_pickle=False) as z:
        ids=z['battery_id'].astype(str).tolist();raw=z['X'].copy();obs=z['observed'].copy();target=z['author_CL'].astype(float)
    lookup={s:i for i,s in enumerate(ids)};groups={r['battery_id']:r['stratum'] for r in original_rows}
    if len(ids)!=124 or set(ids)!=set(groups) or not np.isfinite(target).all() or np.any(target<=100):
        raise ValueError('电芯身份或寿命标签异常')
    pool=np.array([lookup[r['battery_id']] for r in original_rows if r['role']=='train'])
    test=np.array([lookup[r['battery_id']] for r in original_rows if r['role']=='test'])
    group=np.array([groups[s] for s in ids]);seed=20261022
    proper_i,cal_i=next(StratifiedShuffleSplit(n_splits=1,test_size=25,random_state=seed).split(pool,group[pool]))
    proper=pool[proper_i];cal=pool[cal_i]
    fit_i,val_i=next(StratifiedShuffleSplit(n_splits=1,test_size=15,random_state=seed+1).split(proper,group[proper]))
    fit=proper[fit_i];val=proper[val_i];role_check(proper,cal,test,fit,val)
    if tuple(map(len,[proper,cal,test,fit,val]))!=(74,25,25,59,15):raise ValueError('固定划分数量异常')
    cfg=dict(source=str(args.source),data=str(data),tensor_sha256=tensor_sha,source03_sha256=sha(root/'03_knee_train.py'),
        script_sha256=sha(Path(__file__)),outer_splits_sha256=sha(args.source/'outer_splits.csv'),
        seed=seed,proper_train_ids=[ids[i] for i in proper],calibration_ids=[ids[i] for i in cal],
        test_ids=[ids[i] for i in test],inner_fit_ids=[ids[i] for i in fit],inner_validation_ids=[ids[i] for i in val],
        bins=128,lr=.001,max_epochs=args.epochs,patience=args.patience,threads=args.threads,
        nominal_coverage=[.9,.95],primary_review_target_acceptance=.8,
        review_formula='log1p(robust_feature_distance)+5*missing_fraction+5*outside_training_range_fraction',
        torch_version=str(torch.__version__),numpy_version=np.__version__,
        development_data_previously_viewed=True,uses_pretrained99_checkpoint=False,
        limitation='Separated roles within this run; not fresh external validation, no conditional or safety guarantee.')
    signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    parent=root/'models';matches=[]
    if not args.fresh:
        matches=[p for p in parent.glob('trust_tcn_*') if (p/'config.json').exists() and read_json(p/'config.json').get('signature')==signature]
    out=max(matches,key=lambda p:p.name) if matches else parent/('trust_tcn_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=True);run_lock=lock(out);write_json(out/'config.json',dict(signature=signature,**cfg))
    proper_set,cal_set,test_set,fit_set=map(set,[proper,cal,test,fit])
    roles=[dict(battery_id=s,stratum=group[i],role='train' if i in proper_set else 'calibration' if i in cal_set else 'test',
                inner_role=('fit' if i in fit_set else 'validation') if i in proper_set else '') for i,s in enumerate(ids)]
    write_csv(out/'roles.csv',roles)
    if (out/'SUCCESS.json').exists():print((out/'summary.txt').read_text(encoding='utf-8-sig'));return
    print('可信分析：74训练 / 25校准 / 25测试；内部59拟合 / 15验证。',flush=True)
    print('只训练一个TCN；不复用99颗模型、不覆盖原实验。每轮保存，关闭后双击同一入口可续跑。',flush=True)
    print('结果目录：'+str(out),flush=True)
    inner_prep=old.fit_preprocessing(raw,fit);inner_values=old.transform(raw,obs,inner_prep,128)
    net,_,selection=train(torch,np,old,out/'inner_selection',inner_values,target,fit,val,
        args.epochs,args.patience,.001,seed+2,signature)
    del net,inner_values
    epochs=selection['selected_epochs'];write_json(out/'selection.json',selection)
    print(f'内部选择{epochs}轮；重新初始化后用74颗训练，不接触校准或测试标签。',flush=True)
    prep=old.fit_preprocessing(raw,proper);values=old.transform(raw,obs,prep,128)
    if not np.isfinite(values).all():raise ValueError('模型输入非有限')
    net,scaling,refit=train(torch,np,old,out/'proper_refit',values,target,proper,None,
        epochs,args.patience,.001,seed+3,signature+':refit:'+str(epochs))
    atomic_torch(torch,out/'tcn.pt',dict(state_dict=clone(net),attention=False,capacity_bins=128,input_window=100,
        preprocessing=prep,target_scaling=scaling,training_cells=[ids[i] for i in proper],
        warning='74-cell development model; interval calibration is stored separately'))
    score_model=fit_score_model(np,values,proper)
    score,distance,missing,outside=review_scores(np,values,raw,obs,prep,score_model)
    write_json(out/'review_score_model.json',score_model);write_json(out/'preprocessing.json',prep)
    # Calibration quantities and all input-score thresholds are fixed before any test error is computed.
    cal_pred=np.maximum(old.predict(torch,net,values,cal,scaling,'cpu',8),100.)
    residual=np.abs(target[cal]-cal_pred);radii=[]
    for alpha in [.1,.05]:
        q,k=conformal_radius(np,residual,alpha);radii.append(dict(nominal_coverage=1-alpha,alpha=alpha,radius=q,order_statistic=k,n_calibration=25))
    thresholds=[dict(target_acceptance=a,threshold=acceptance_threshold(np,score[cal],a)) for a in [.5,.6,.7,.8,.9,1.]]
    calibration=dict(method='absolute-residual split conformal, finite-sample order statistic',intervals=radii,
        review_thresholds=thresholds,primary_acceptance=.8,calibration_cells=[ids[i] for i in cal],
        model_sha256=sha(out/'tcn.pt'),review_is_heuristic=True,
        warning='Constant residual radius; do not use interval width or clipping-induced width changes as a risk ranking.')
    write_json(out/'calibration.json',calibration)
    write_csv(out/'calibration_predictions.csv',[dict(battery_id=ids[i],true_CL=float(target[i]),prediction_CL=float(p),
        absolute_residual=float(e),review_score=float(score[i])) for i,p,e in zip(cal,cal_pred,residual)])
    test_raw=old.predict(torch,net,values,test,scaling,'cpu',8);pred=np.maximum(test_raw,100.);truth=target[test]
    error=np.abs(pred-truth);metrics=[];intervals=[]
    for r in radii:
        lower=np.maximum(100.,pred-r['radius']);upper=pred+r['radius'];covered=(truth>=lower)&(truth<=upper)
        intervals.append((lower,upper,covered))
        metrics.append(dict(nominal_coverage=r['nominal_coverage'],n_test=25,covered_count=int(covered.sum()),
            observed_coverage=float(covered.mean()),mean_interval_width=float((upper-lower).mean()),radius_cycles=r['radius']))
    lower,upper,covered=intervals[0]
    review=evaluate_review(np,score[cal],score[test],error,covered,lower,upper)
    primary=next(r for r in review if r['target_calibration_acceptance']==.8)
    gate=primary['score_threshold'];keep=score[test]<=gate
    prediction_rows=[]
    for j,i in enumerate(test):
        lo95,hi95,cv95=intervals[1]
        prediction_rows.append(dict(battery_id=ids[i],stratum=group[i],role='test',true_CL=float(truth[j]),
            raw_prediction_CL=float(test_raw[j]),prediction_CL=float(pred[j]),prediction_RUL100=float(pred[j]-100),
            absolute_error=float(error[j]),lower90=float(lower[j]),upper90=float(upper[j]),covered90=bool(covered[j]),
            lower95=float(lo95[j]),upper95=float(hi95[j]),covered95=bool(cv95[j]),
            review_score=float(score[i]),robust_feature_distance=float(distance[i]),missing_fraction=float(missing[i]),
            outside_training_range_fraction=float(outside[i]),accepted_at_cal80=bool(keep[j]),
            suggested_review_at_cal80=not bool(keep[j])))
    subgroup=[]
    for g in ['fast','medium','slow']:
        m=group[test]==g;n=int(m.sum())
        subgroup.append(dict(stratum=g,n_test=n,MAE=mean_or_none(np,error[m]),
            interval90_coverage=mean_or_none(np,covered[m]),accepted_at_cal80=int((m&keep).sum())))
    write_csv(out/'test_predictions.csv',prediction_rows);write_csv(out/'interval_metrics.csv',metrics)
    write_csv(out/'review_curve.csv',review);write_csv(out/'stratum_metrics.csv',subgroup)
    baseline=float(np.abs(truth-np.median(target[proper])).mean())
    plot_status='generated'
    try:make_plots(np,out,truth,pred,lower,upper,keep,review)
    except ImportError:plot_status='matplotlib unavailable; CSV results complete'
    report=dict(status='TCN_development_trust_analysis_complete',n_train=74,n_calibration=25,n_test=25,
        TCN_MAE=float(error.mean()),TCN_RMSE=float(np.sqrt(np.mean(error**2))),median_MAE=baseline,
        intervals=metrics,primary_review=primary,
        primary_review_reduces_observed_MAE=None if primary['accepted_MAE'] is None else primary['accepted_MAE']<float(error.mean()),
        plot_status=plot_status,fresh_external_validation=False,conditional_coverage_guaranteed=False,risk_threshold_validated_for_deployment=False,
        note='Prior development data were viewed. 25 test cells give 4 percentage point coverage resolution; small subgroups are descriptive only.')
    def f(v):return '无样本' if v is None else f'{v:.2f}'
    lines=['TCN可信分析完成（开发数据补充验证，不是外部新数据验证）。',
        '划分：74训练 / 25校准 / 25测试。',
        f"TCN测试MAE={report['TCN_MAE']:.2f}循环，RMSE={report['TCN_RMSE']:.2f}循环；训练中位数MAE={baseline:.2f}。"]
    for r in metrics:lines.append(f"名义{r['nominal_coverage']:.0%}区间：测试覆盖{r['covered_count']}/25={r['observed_coverage']:.1%}，平均宽度={r['mean_interval_width']:.2f}循环。")
    lines += [f"校准集预设80%接受规则：测试实际保留{primary['test_accepted']}/25，保留MAE={f(primary['accepted_MAE'])}，复核部分MAE={f(primary['reviewed_MAE'])}。",
        '保留MAE低于全部测试MAE：'+str(report['primary_review_reduces_observed_MAE']),
        '接受后区间覆盖率不继承全体覆盖保证；复核分数不是安全概率，阈值尚不能用于部署。',
        '本次74颗模型与原99颗模型不同，不能直接替换原五次实验成绩。',
        '结果目录：'+str(out),'请发送summary.txt、interval_metrics.csv和review_curve.csv。']
    text='\n'.join(lines);(out/'summary.txt').write_text(text,encoding='utf-8-sig')
    write_json(out/'SUCCESS.json',report);print('\n'+text,flush=True)


if __name__=='__main__':main()
