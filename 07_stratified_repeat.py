"""Five repeated nominal-charge-time-stratified 99/25 splits.

Protocol inspired by the supplied screenshot, not an exact TECAN reproduction.
No conformal calibration or deployment risk thresholds are fitted here.
"""
import argparse
import csv
import importlib.util
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def policy_time(text):
    # A trailing C may be missing in source descriptors, e.g. 4C(31%)-5.
    # Accept only a complete numeric two-step policy, and disclose this repair.
    number=r'(\d+(?:\.\d+)?)'
    cleaned=text.strip().replace(' ','').removesuffix('-newstructure')
    m=re.fullmatch(number+r'C\('+number+r'%\)-'+number+r'(C?)',cleaned)
    if not m:
        raise ValueError('不能解析充电协议，停止而不猜测：'+text)
    c1,q1,c2=map(float,m.groups()[:3]);q1/=100.
    if min(c1,c2)<=0 or not 0<=q1<=.8:
        raise ValueError('充电协议参数超范围：'+text)
    # C-rate is fractional nominal capacity per hour, so result is minutes.
    minutes=60.*(q1/c1+(.8-q1)/c2)
    return minutes,(m.group(4)=='')


def time_group(minutes):
    return 'fast' if minutes<10.5 else ('medium' if minutes<=11.7 else 'slow')


def strata_from_source(data,ids):
    import numpy as np
    rows=read_csv(data/'source_map.csv')
    by_id={s:[] for s in ids}
    for r in rows:
        if r['battery_id'] in by_id and 1<=int(r['cumulative_cycle'])<=100:
            by_id[r['battery_id']].append(r)
    result=[]
    for cell in ids:
        rr=by_id[cell]
        policies={r['policy_raw'].strip() for r in rr}
        if len(policies)!=1:
            raise ValueError(f'{cell}前100循环协议缺失或不一致')
        policy=next(iter(policies));minutes,repaired=policy_time(policy)
        measured=[]
        for r in rr:
            try:
                value=float(r['chargetime_clean_source_unit_pending'])
            except (ValueError,KeyError):
                continue
            if np.isfinite(value) and value>0:
                measured.append(value)
        if len(measured)<80:
            raise ValueError(f'{cell}早期有效充电时间不足80条；先复核单位与原始记录')
        median=float(np.median(measured));ratio=median/minutes
        # Sanity check of units/order of magnitude, not a metadata certification.
        if not .8<=ratio<=1.2:
            raise ValueError(f'{cell}实测源数值/协议分钟数={ratio:.3f}，不自动换算；请核实')
        result.append(dict(battery_id=cell,policy_raw=policy,
            nominal_time_to_80_percent_minutes=minutes,stratum=time_group(minutes),
            missing_trailing_C_accepted=repaired,source_newstructure_suffix=text_suffix(policy),early_measured_source_median=median,
            measured_to_nominal_ratio=ratio,early_measured_n=len(measured)))
    return result


def text_suffix(policy):
    return policy.strip().endswith('-newstructure')


def make_splits(groups):
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit
    groups=np.asarray(groups)
    # Frozen seeds; never search for a favorable test split.
    outer=StratifiedShuffleSplit(n_splits=5,train_size=99,test_size=25,random_state=20261021)
    splits=[]
    for repeat,(train,test) in enumerate(outer.split(np.zeros(len(groups)),groups),1):
        inside=StratifiedShuffleSplit(n_splits=1,test_size=20,random_state=20261021+repeat)
        fi,vi=next(inside.split(np.zeros(len(train)),groups[train]))
        splits.append((repeat,train,test,train[fi],train[vi]))
    return splits


def self_test():
    import numpy as np
    np.testing.assert_allclose(policy_time('3.6C(80%)-3.6C')[0],13+1/3)
    np.testing.assert_allclose(policy_time('4C(31%)-5')[0],60*(.31/4+.49/5))
    assert policy_time('4C(31%)-5')[1]
    np.testing.assert_allclose(policy_time('5C(67%)-4C-newstructure')[0],60*(.67/5+.13/4))
    assert [time_group(v) for v in (10.49,10.5,11.7,11.71)]==['fast','medium','medium','slow']
    g=['fast']*40+['medium']*40+['slow']*44
    for _,tr,te,fit,val in make_splits(g):
        assert (len(tr),len(te),len(fit),len(val))==(99,25,79,20)
        assert not set(tr)&set(te) and set(fit)|set(val)==set(tr) and not set(fit)&set(val)
        assert {g[i] for i in tr}=={g[i] for i in te}==set(g)
    print('PASS：协议时长、缺C标记、分层边界、五次99/25及内部79/20身份隔离。',flush=True)


def fit_net(torch,old,values,target,train,validation,attention,lr,epochs,patience,seed,bins,label):
    import numpy as np
    old.seed_all(torch,seed)
    net=old.build_network(torch,attention,bins)
    opt=torch.optim.Adam(net.parameters(),lr=lr)
    mean=float(target[train].mean());sd=max(float(target[train].std()),1.)
    x=torch.from_numpy(values)
    y=torch.from_numpy(((target-mean)/sd).astype(np.float32))
    rng=np.random.default_rng(seed);best=float('inf');best_epoch=0;stale=0;history=[];best_state=None
    begin=time.perf_counter()
    for epoch in range(1,epochs+1):
        net.train();total=0.;count=0
        for batch in np.array_split(rng.permutation(train),math.ceil(len(train)/8)):
            opt.zero_grad(set_to_none=True)
            loss=((net(x[batch])-y[batch])**2).mean()
            if not torch.isfinite(loss):raise RuntimeError('损失非有限')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            opt.step();total+=float(loss.detach())*len(batch);count+=len(batch)
        vm=None
        if validation is not None:
            net.eval()
            with torch.no_grad():
                vp=torch.cat([net(x[b]) for b in np.array_split(validation,math.ceil(len(validation)/8))])
                vm=float(((vp-y[validation])**2).mean())
            if not math.isfinite(vm):raise RuntimeError('验证损失非有限')
            if vm<best-1e-6:
                best=vm;best_epoch=epoch;stale=0
                best_state={k:v.detach().clone() for k,v in net.state_dict().items()}
            else:stale+=1
        history.append(dict(epoch=epoch,train_mse_normalized=total/count,inner_validation_mse=vm,
                            elapsed_seconds=time.perf_counter()-begin))
        print(f'    {label} {epoch}/{epochs}轮 拟合MSE={total/count:.4f}'+
              (f' 验证MSE={vm:.4f} 最佳轮{best_epoch}' if vm is not None else '')+
              f' 已用{(time.perf_counter()-begin)/60:.1f}分钟',flush=True)
        if validation is not None and stale>=patience:break
    if validation is not None:
        net.load_state_dict(best_state)
    return net,dict(target_mean=mean,target_std=sd),best_epoch,best,history


def main():
    ap=argparse.ArgumentParser(description='按标称充电时间分层，重复五次99训练/25测试；不生成区间')
    ap.add_argument('--run',type=Path,help='03已完成结果，复用曲线路径及膝点标签')
    ap.add_argument('--repeats',default='1,2,3,4,5')
    ap.add_argument('--models',default='tcn,tecan,model2')
    ap.add_argument('--epochs',type=int,default=150)
    ap.add_argument('--patience',type=int,default=30)
    ap.add_argument('--bins',type=int,default=128)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test();return
    if min(args.epochs,args.patience,args.bins,args.threads)<1:ap.error('数值参数必须为正')
    repeats=[int(s) for s in args.repeats.split(',')];models=args.models.split(',')
    if not set(repeats)<={1,2,3,4,5} or len(set(repeats))!=len(repeats):ap.error('重复编号应为1至5，不能重复')
    if not set(models)<={'tcn','tecan','model2'} or len(set(models))!=len(models):ap.error('模型名错误或重复')
    import numpy as np
    root=Path(__file__).resolve().parent
    source=root/'03_knee_train.py'
    if not source.exists():raise FileNotFoundError('07与03_knee_train.py需放同一文件夹')
    spec=importlib.util.spec_from_file_location('curve03',source)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    if args.run is None:
        choices=[p for p in (root/'models').glob('train_*')
                 if (p/'SUCCESS.json').exists() and (p/'knee_labels.csv').exists()]
        if not choices:raise ValueError('没有可复用的03完整结果，请用--run指定')
        args.run=max(choices,key=lambda p:p.name)
    original=load_json(args.run/'config.json');data=Path(original['data_directory'])
    print('读取124颗电芯的已有曲线和膝点；先核对标称充电时间。',flush=True)
    tensor_sha=old.sha(data/'curves.npz')
    if tensor_sha!=load_json(data/'SUCCESS.json').get('tensor_sha256'):raise ValueError('张量指纹不一致')
    with np.load(data/'curves.npz',allow_pickle=False) as z:
        X=z['X'];obs=z['observed'];ids=z['battery_id'].tolist();cl=z['author_CL'].astype(float)
    if len(ids)!=124 or len(set(ids))!=124 or not np.isfinite(cl).all():raise ValueError('需要124颗唯一且标签完整的电芯')
    if X.shape[-1]%args.bins:raise ValueError('bins需整除容量轴点数')
    strata=strata_from_source(data,ids)
    groups=[r['stratum'] for r in strata];splits=make_splits(groups)
    knees={r['battery_id']:r for r in read_csv(args.run/'knee_labels.csv')}
    if 'model2' in models and any(knees.get(s,{}).get('status')!='valid_by_rule' for s in ids):
        raise ValueError('有膝点未通过规则检查，不为凑124颗强行填充；可用--models tcn,tecan运行直接寿命模型')
    ctk=np.array([float(knees[s]['CTK']) if knees.get(s,{}).get('status')=='valid_by_rule' else np.nan for s in ids])
    out=root/'models'/('stratified_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    save_csv(out/'strata.csv',strata)
    memberships=[]
    for repeat,train,test,fit,val in splits:
        for role,indices in [('train',train),('test',test)]:
            memberships.extend(dict(repeat=repeat,battery_id=ids[i],role=role,stratum=groups[i]) for i in indices)
    save_csv(out/'outer_splits.csv',memberships)
    cfg=dict(protocol='screenshot-inspired 5 repeated stratified 99/25 splits, not fivefold CV',
        source_run=str(args.run),data=str(data),tensor_sha256=tensor_sha,
        source_map_sha256=old.sha(data/'source_map.csv'),source03_sha256=old.sha(source),
        source_knee_sha256=old.sha(args.run/'knee_labels.csv'),script_sha256=old.sha(Path(__file__)),
        seed=20261021,learning_rates=[1e-4,1e-3],batch_size=8,device='cpu',
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        stratum_rule='nominal 0-80% time=60*(q1/C1+(.8-q1)/C2); fast<10.5; medium10.5..11.7; slow>11.7',
        unit_check='early measured source median/nominal minutes must be in [.8,1.2]; sanity check only',
        inner='79 fit /20 validation stratified within each 99 training cells; fit-only preprocessing',
        outer='select lr and epoch internally, fresh 99-cell refit, then 25-cell test',
        scope='exploratory known dataset; test identities may recur across repeats; no interval or deployment claim',
        deviations=['exact author split IDs and seeds unavailable',
                    'explicit boundary convention: 10.5 and11.7 included in medium',
                    'TECAN-inspired architecture unchanged from03; 128 bins, mask, fusion and head deviations retained',
                    '150epoch cap,30patience and two lr candidates; not exact paper hyperparameters',
                    'algorithmic CTK proxy reused; no independent knee ground truth'])
    save_json(out/'config.json',cfg)
    for group in ['fast','medium','slow']:
        print(f'分层 {group}：{groups.count(group)}颗',flush=True)
    print('已保存全部五次固定划分；各次测试25颗可能重复，不是五折。',flush=True)
    if args.prepare_only:
        print('仅准备完成，未训练：'+str(out),flush=True);return
    import torch
    torch.set_num_threads(args.threads)
    print('开始CPU训练；每轮显示进度。这是完整实验，明显比04/06检查久。',flush=True)
    all_rows=[];metrics=[]
    for repeat,train,test,fit,val in splits:
        if repeat not in repeats:continue
        folder=out/f'repeat_{repeat}';folder.mkdir()
        save_csv(folder/'inner_roles.csv',[dict(battery_id=ids[i],role=role,stratum=groups[i])
            for role,indices in [('fit',fit),('validation',val)] for i in indices])
        prep=old.fit_preprocessing(X,fit);values=old.transform(X,obs,prep,args.bins)
        save_json(folder/'inner_preprocessing.json',prep)
        predictions={'median':np.full(len(test),np.median(cl[train]))}
        # No outer-test metric is printed until all models' inner selection and refit finish.
        for model in models:
            print(f'\n重复{repeat}/5 {old.NAMES[model]}：内部拟合79、验证20，最终训练99、测试25。',flush=True)
            target=ctk if model=='model2' else cl
            candidates=[]
            for lr in cfg['learning_rates']:
                net,scaling,best_epoch,score,history=fit_net(torch,old,values,target,fit,val,model!='tcn',lr,
                    args.epochs,args.patience,20261021+repeat,args.bins,f'lr={lr:g}')
                save_csv(folder/f'{model}_lr{lr:g}_inner_loss.csv',history)
                candidates.append(dict(lr=lr,selected_epochs=best_epoch,inner_target_mse=score))
                del net
            selected=min(candidates,key=lambda r:r['inner_target_mse'])
            save_json(folder/f'{model}_selection.json',dict(candidates=candidates,selected=selected,
                outer_test_used=False,criterion='minimum standardized target MSE on inner validation'))
            print(f"  选择lr={selected['lr']:g}，使用全部99颗重训{selected['selected_epochs']}轮。",flush=True)
            outer_prep=old.fit_preprocessing(X,train)
            final_values=old.transform(X,obs,outer_prep,args.bins)
            net,scaling,_,_,history=fit_net(torch,old,final_values,target,train,None,model!='tcn',selected['lr'],
                selected['selected_epochs'],args.patience,20261021+repeat,args.bins,'99颗重训')
            save_csv(folder/f'{model}_refit_loss.csv',history)
            mapping=None
            pred=old.predict(torch,net,final_values,test,scaling,'cpu',8)
            if model=='model2':
                a,b=old.train_mapping(ctk[train],cl[train]);mapping=dict(a=a,b=b,fitted_on='99 training cells only')
                pred=np.maximum(pred,0)*a+b
            predictions[model]=np.maximum(pred,100.)
            save_json(folder/f'{model}_preprocessing.json',outer_prep)
            torch.save(dict(state_dict=net.state_dict(),architecture='03 build_network',attention=model!='tcn',
                capacity_bins=args.bins,target_scaling=scaling,ctk_to_cl=mapping,preprocessing=outer_prep,
                selected=selected,training_cells=[ids[i] for i in train],
                input_window=100,warning='experimental point predictor; no interval/risk thresholds'),folder/f'{model}.pt')
            del net,final_values
        fold_rows=[]
        for model,pred in predictions.items():
            err=pred-cl[test]
            metric=dict(repeat=repeat,model=model,n_test=25,MAE_cycles=float(np.abs(err).mean()),
                RMSE_cycles=float(np.sqrt(np.mean(err**2))),prediction_range_cycles=float(np.ptp(pred)))
            metrics.append(metric)
            for i,p in zip(test,pred):
                fold_rows.append(dict(repeat=repeat,model=model,battery_id=ids[i],stratum=groups[i],
                    true_CL=float(cl[i]),prediction_CL=float(p),error_cycles=float(p-cl[i])))
            print(f"重复{repeat} {model}：MAE={metric['MAE_cycles']:.2f} RMSE={metric['RMSE_cycles']:.2f}",flush=True)
        all_rows.extend(fold_rows)
        save_csv(folder/'predictions.csv',fold_rows)
        save_csv(out/'repeat_metrics.csv',metrics);save_csv(out/'predictions.csv',all_rows)
    aggregate=[]
    for model in sorted({r['model'] for r in metrics}):
        rr=[r for r in metrics if r['model']==model]
        mae=np.array([r['MAE_cycles'] for r in rr]);rmse=np.array([r['RMSE_cycles'] for r in rr])
        aggregate.append(dict(model=model,n_repeats=len(rr),MAE_mean=float(mae.mean()),
            MAE_sample_std=float(mae.std(ddof=1)) if len(rr)>1 else None,
            RMSE_mean=float(rmse.mean()),RMSE_sample_std=float(rmse.std(ddof=1)) if len(rr)>1 else None))
    save_csv(out/'summary.csv',aggregate)
    save_json(out/'SUCCESS.json',dict(status='stratified_repeated_point_prediction_complete',
        completed_repeats=repeats,summary=aggregate,full_five_repeats=set(repeats)=={1,2,3,4,5},
        caution='sample standard deviations describe repeated splits, not independent-trial confidence intervals'))
    print('\n分层重复划分汇总（各次99训练/25测试；测试对象可重复）：',flush=True)
    for r in aggregate:
        sd=f"{r['MAE_sample_std']:.2f}" if r['MAE_sample_std'] is not None else '未计算'
        print(f"{r['model']}：{r['n_repeats']}次，MAE均值={r['MAE_mean']:.2f}，标准差={sd}，RMSE均值={r['RMSE_mean']:.2f}",flush=True)
    print('结果目录：'+str(out),flush=True)


if __name__=='__main__':
    main()
