"""Inner-fit-only TCN warm start and identity-initialized ECA experiment.
No outer-test prediction. Checkpoints contain state dictionaries, not modules.
"""
import argparse
import atexit
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time
from datetime import datetime


def json_read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def csv_read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def json_write(path,data):
    temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)


def csv_write(path,rows):
    if not rows:return
    temp=path.with_suffix('.csv.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    os.replace(temp,path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def import_model(path):
    spec=importlib.util.spec_from_file_location('curve03',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def clone_state(net):
    return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}


def save_torch(torch,path,data):
    temp=path.with_suffix('.pt.tmp');torch.save(data,temp);os.replace(temp,path)


def lock_run(directory):
    handle=(directory/'RUNNING.lock').open('a+b')
    handle.seek(0,2)
    if handle.tell()==0:handle.write(b'0');handle.flush()
    handle.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close();raise RuntimeError('相同实验已在另一个窗口运行，请保留一个训练窗口')
    atexit.register(handle.close)
    return handle


def warm_network(torch,old,bins,base_state):
    nn=torch.nn
    class IdentityECA(nn.Module):
        def __init__(self):
            super().__init__();self.conv=nn.Conv1d(1,1,5,padding=2,bias=False)
            nn.init.zeros_(self.conv.weight)
        def forward(self,x):
            score=x.mean((2,3)).unsqueeze(1)
            gate=2*torch.sigmoid(self.conv(score)).squeeze(1)[:,:,None,None]
            return x*gate
    net=old.build_network(torch,True,bins)
    for block in net.blocks:block.attention=IdentityECA()
    result=net.load_state_dict(base_state,strict=False)
    expected={f'blocks.{i}.attention.conv.weight' for i in range(3)}
    if set(result.missing_keys)!=expected or result.unexpected_keys:
        raise RuntimeError('TCN到TECAN的参数映射不完整')
    return net


def checked_roles(rows,outer):
    allowed={r['battery_id'] for r in outer if r['role']=='train'}
    held={r['battery_id'] for r in outer if r['role']!='train'}
    names=[r['battery_id'] for r in rows]
    if len(names)!=len(set(names)) or set(names)!=allowed or allowed&held:
        raise ValueError('内部身份重复、缺失或混入外层测试身份')
    if any(r['role'] not in ('fit','validation') for r in rows):raise ValueError('内部角色非法')


def stage_optimizer(torch,net,phase):
    attention=[];trunk=[]
    for name,param in net.named_parameters():
        is_attention='.attention.' in name
        param.requires_grad_(is_attention if phase['freeze_trunk'] else True)
        if param.requires_grad:(attention if is_attention else trunk).append(param)
    groups=[]
    if trunk:groups.append(dict(params=trunk,lr=phase['trunk_lr']))
    if attention:groups.append(dict(params=attention,lr=phase['attention_lr']))
    if not groups:raise RuntimeError('没有可训练参数')
    return torch.optim.Adam(groups)


def evaluate(torch,net,x,y,indices,target,scaling):
    import numpy as np
    net.eval()
    with torch.no_grad():
        predictions=torch.cat([net(x[b]) for b in np.array_split(indices,math.ceil(len(indices)/8))])
        mse=float(((predictions-y[indices])**2).mean())
    physical=predictions.numpy()*scaling['std']+scaling['mean']
    if not math.isfinite(mse) or not np.isfinite(physical).all():raise RuntimeError('验证输出非有限')
    return dict(mse=mse,mae=float(np.abs(physical-target[indices]).mean()),
                prediction_range=float(np.ptp(physical))),physical


def run_stage(torch,old,directory,maker,phases,x,y,target,fit,val,scaling,seed,signature,
              reset_dropout=True,stop_after=None):
    """Atomic per-epoch state + optimizer/shuffle/dropout RNG, supporting resume."""
    import numpy as np
    directory.mkdir(parents=True,exist_ok=True)
    last=directory/'latest.pt';done=directory/'DONE.json'
    old.seed_all(torch,seed)
    net=maker()
    if reset_dropout:torch.manual_seed(seed+10000)
    shuffle=np.random.default_rng(seed+10000 if reset_dropout else seed)
    loaded=None
    if last.exists():
        loaded=torch.load(last,map_location='cpu',weights_only=True)
        if loaded['signature']!=signature:raise ValueError('检查点与本次配置不匹配')
        net.load_state_dict(loaded['state'])
        if done.exists():
            net.load_state_dict(loaded['best_state'])
            return net,json_read(done)
        completed=loaded['epoch'];best=loaded['best_score'];best_state=loaded['best_state']
        best_epoch=loaded['best_epoch'];history=loaded['history'];accumulated=loaded['elapsed_seconds']
        shuffle.bit_generator.state=loaded['shuffle_state'];torch.set_rng_state(loaded['torch_rng'])
        print(f'恢复 {directory.name}：已完成{completed}轮。',flush=True)
    else:
        completed=0;history=[];accumulated=0.
        best,_=evaluate(torch,net,x,y,val,target,scaling)
        best_state=clone_state(net);best_epoch=0
    total_epochs=sum(p['epochs'] for p in phases)
    begin=time.perf_counter();elapsed=accumulated;opt=None;previous_phase=None
    for epoch in range(completed+1,total_epochs+1):
        cursor=0
        for phase_index,phase in enumerate(phases):
            cursor+=phase['epochs']
            if epoch<=cursor:break
        if previous_phase!=phase_index:
            opt=stage_optimizer(torch,net,phase)
            if loaded is not None and loaded['phase_index']==phase_index:
                opt.load_state_dict(loaded['optimizer'])
            previous_phase=phase_index
        net.train();total_loss=0.;count=0;gradient_max=0.
        for batch in np.array_split(shuffle.permutation(fit),math.ceil(len(fit)/8)):
            net.zero_grad(set_to_none=True)
            loss=((net(x[batch])-y[batch])**2).mean()
            if not torch.isfinite(loss):raise RuntimeError('训练损失非有限，请保留日志')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            gradient_max=max(gradient_max,float(norm));opt.step()
            total_loss+=float(loss.detach())*len(batch);count+=len(batch)
        score,_=evaluate(torch,net,x,y,val,target,scaling)
        if score['mse']<best['mse']-1e-6:
            best=score;best_state=clone_state(net);best_epoch=epoch
        elapsed=accumulated+time.perf_counter()-begin
        history.append(dict(epoch=epoch,phase=phase['name'],train_mse=total_loss/count,
            validation_mse=score['mse'],validation_mae_cycles=score['mae'],
            prediction_range_cycles=score['prediction_range'],best_epoch=best_epoch,
            gradient_norm_max=gradient_max,elapsed_seconds=elapsed))
        state=dict(signature=signature,state=clone_state(net),best_state=best_state,
            best_score=best,best_epoch=best_epoch,epoch=epoch,phase_index=phase_index,
            optimizer=opt.state_dict(),shuffle_state=shuffle.bit_generator.state,
            torch_rng=torch.get_rng_state(),history=history,elapsed_seconds=elapsed)
        save_torch(torch,last,state)
        csv_write(directory/'loss.csv',history)
        mean_seconds=(time.perf_counter()-begin)/(epoch-completed)
        print(f'  {directory.name} {epoch}/{total_epochs}轮 {phase["name"]} '
              f'训练MSE={total_loss/count:.4f} 验证MAE={score["mae"]:.2f} '
              f'验证MSE={score["mse"]:.4f} 最佳轮={best_epoch} '
              f'已用{elapsed/60:.1f}分钟 该段预计剩余{mean_seconds*(total_epochs-epoch)/60:.1f}分钟',flush=True)
        if stop_after is not None and epoch>=stop_after:return None,None
    net.load_state_dict(best_state)
    csv_write(directory/'loss.csv',history)
    result=dict(best_epoch=best_epoch,epochs_run=total_epochs,elapsed_seconds=elapsed,
        validation_mse=best['mse'],validation_mae_cycles=best['mae'],
        prediction_range_cycles=best['prediction_range'],epoch_zero_eligible=True)
    json_write(done,result)
    return net,result


def self_test(old,torch,root):
    import numpy as np
    torch.set_num_threads(2);old.seed_all(torch,2026)
    base=old.build_network(torch,False,4);base.eval()
    adapted=warm_network(torch,old,4,clone_state(base));adapted.eval()
    x=torch.randn(6,4,4,20);target=np.array([100.,200.,400.,600.,800.,900.])
    y=torch.tensor(((target-300)/200).astype('float32'))
    with torch.no_grad():torch.testing.assert_close(base(x),adapted(x),rtol=0,atol=0)
    initial=clone_state(adapted);opt=stage_optimizer(torch,adapted,
        dict(freeze_trunk=True,trunk_lr=0.,attention_lr=.001))
    adapted.train();opt.zero_grad();((adapted(x[:4])-y[:4])**2).mean().backward();opt.step()
    after=clone_state(adapted)
    assert all(torch.equal(initial[k],after[k]) for k in initial if '.attention.' not in k)
    assert any(not torch.equal(initial[k],after[k]) for k in initial if '.attention.' in k)
    try:checked_roles([dict(battery_id='bad',role='fit')],[dict(battery_id='bad',role='test')])
    except ValueError:pass
    else:raise AssertionError('角色隔离失败')
    phases=[dict(name='test1',epochs=2,freeze_trunk=False,trunk_lr=.001,attention_lr=.001),
            dict(name='test2',epochs=2,freeze_trunk=False,trunk_lr=.0001,attention_lr=.0001)]
    with tempfile.TemporaryDirectory(dir=root) as td:
        p=Path(td)
        maker=lambda:old.build_network(torch,False,4)
        kw=dict(torch=torch,old=old,maker=maker,phases=phases,x=x,y=y,target=target,
            fit=np.arange(4),val=np.arange(4,6),scaling=dict(mean=300.,std=200.),seed=77,signature='synthetic')
        a,ra=run_stage(directory=p/'straight',**kw)
        run_stage(directory=p/'resumed',stop_after=2,**kw)
        b,rb=run_stage(directory=p/'resumed',**kw)
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        assert ra['validation_mse']==rb['validation_mse']
    print('PASS：恒等注意力输出完全一致、冻结主干不变、注意力参数更新、外层角色拦截、断点与连续训练一致。',flush=True)


def main():
    ap=argparse.ArgumentParser(description='内部TCN预训练、等步数TCN对照与恒等ECA微调；不评价外层测试')
    ap.add_argument('--source',type=Path)
    ap.add_argument('--repeat',type=int,default=1,choices=range(1,6))
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--freeze-epochs',type=int,default=10)
    ap.add_argument('--joint-epochs',type=int,default=30)
    ap.add_argument('--pretrain-epochs',type=int,default=None)
    ap.add_argument('--fresh',action='store_true',help='主动创建独立实验，不复用已有诊断')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if min(args.threads,args.freeze_epochs,args.joint_epochs)<1:ap.error('轮数和线程数必须为正')
    import numpy as np
    import torch
    root=Path(__file__).resolve().parent
    old=import_model(root/'03_knee_train.py')
    if args.self_test:self_test(old,torch,root);return
    torch.set_num_threads(args.threads)
    if args.source is None:
        candidates=[p for p in (root/'models').glob('stratified_*') if (p/'SUCCESS.json').exists()]
        if not candidates:raise ValueError('未找到完整分层实验，请用--source指定')
        args.source=max(candidates,key=lambda p:p.name)
    source_config=json_read(args.source/'config.json')
    fold=args.source/f'repeat_{args.repeat}'
    inner=csv_read(fold/'inner_roles.csv')
    outer=[r for r in csv_read(args.source/'outer_splits.csv') if int(r['repeat'])==args.repeat]
    checked_roles(inner,outer)
    selected=json_read(fold/'tcn_selection.json')['selected']
    pre_epochs=args.pretrain_epochs if args.pretrain_epochs is not None else int(selected['selected_epochs'])
    if pre_epochs<1:raise ValueError('预训练轮数必须大于0')
    data=Path(source_config['data']);tensor_hash=sha(data/'curves.npz')
    if tensor_hash!=json_read(data/'SUCCESS.json').get('tensor_sha256'):raise ValueError('曲线指纹不一致')
    names=[r['battery_id'] for r in inner]
    print('读取第一组训练内部数据；不使用外层25颗的标签，不重做曲线或膝点。',flush=True)
    with np.load(data/'curves.npz',allow_pickle=False) as z:
        lookup={s:i for i,s in enumerate(z['battery_id'].tolist())}
        ii=[lookup[s] for s in names]
        raw=z['X'][ii].copy();obs=z['observed'][ii].copy();target=z['author_CL'][ii].astype(float)
    fit=np.array([i for i,r in enumerate(inner) if r['role']=='fit'],dtype=int)
    val=np.array([i for i,r in enumerate(inner) if r['role']=='validation'],dtype=int)
    if (len(fit),len(val))!=(79,20) or not np.isfinite(target).all():raise ValueError('内部样本或标签异常')
    bins=int(source_config['arguments']['bins'])
    prep=old.fit_preprocessing(raw,fit);values=old.transform(raw,obs,prep,bins)
    del raw,obs
    if not np.isfinite(values).all():raise ValueError('预处理后出现非法值')
    scaling=dict(mean=float(target[fit].mean()),std=max(float(target[fit].std()),1.))
    x=torch.from_numpy(values);y=torch.from_numpy(((target-scaling['mean'])/scaling['std']).astype(np.float32))
    seed=20261021+args.repeat
    cfg=dict(source=str(args.source),repeat=args.repeat,roles=inner,tensor_sha256=tensor_hash,
        original_model_sha256=sha(root/'03_knee_train.py'),script_sha256=sha(Path(__file__)),
        inner_roles_sha256=sha(fold/'inner_roles.csv'),selection_sha256=sha(fold/'tcn_selection.json'),
        capacity_bins=bins,pretrain_lr=float(selected['lr']),pretrain_epochs=pre_epochs,
        freeze_epochs=args.freeze_epochs,joint_epochs=args.joint_epochs,seed=seed,threads=args.threads,
        torch_version=str(torch.__version__),numpy_version=np.__version__,
        joint_trunk_lr=.0001,joint_attention_lr=.0003,frozen_attention_lr=.001,
        selection='minimum inner validation normalized CL MSE, including epoch0',
        control='same extra minibatch-update count, not equal trainable parameters or identical computation cost',
        limitation='exploratory repeated use of an existing inner split; not independent final accuracy',
        method='TCN warm start + zero-initialized 2*sigmoid ECA; experimental change from paper')
    signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    parent=root/'results';parent.mkdir(exist_ok=True)
    matches=[]
    if not args.fresh:
        for directory in parent.glob('warm_tecan_*'):
            if (directory/'config.json').exists() and json_read(directory/'config.json').get('signature')==signature:
                matches.append(directory)
    out=max(matches,key=lambda p:p.name) if matches else parent/('warm_tecan_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=True)
    _run_lock=lock_run(out)
    json_write(out/'config.json',dict(signature=signature,**cfg));json_write(out/'preprocessing.json',dict(input=prep,target=scaling))
    if (out/'SUCCESS.json').exists():
        print('此配置已完成，不重复训练。',flush=True)
        print_summary(json_read(out/'SUCCESS.json'),out);return
    print(f'内部拟合79颗、内部验证20颗；容量轴{bins}点。',flush=True)
    print(f'预训练{pre_epochs}轮；两条对照各追加{args.freeze_epochs+args.joint_epochs}轮；CPU。',flush=True)
    print('每轮保存断点；关闭后重新双击同一启动文件可继续。',flush=True)
    print('结果目录：'+str(out),flush=True)
    pre_phase=[dict(name='TCN预训练',epochs=pre_epochs,freeze_trunk=False,trunk_lr=cfg['pretrain_lr'],attention_lr=cfg['pretrain_lr'])]
    print('\n第1/3段：只用79颗内部拟合电芯预训练TCN。',flush=True)
    base,base_result=run_stage(torch,old,out/'pretrain',lambda:old.build_network(torch,False,bins),
        pre_phase,x,y,target,fit,val,scaling,seed,signature,reset_dropout=False)
    base_state=clone_state(base)
    adapted=warm_network(torch,old,bins,base_state);adapted.eval();base.eval()
    with torch.no_grad():torch.testing.assert_close(base(x[val[:2]]),adapted(x[val[:2]]),rtol=0,atol=0)
    print('已确认：加入注意力后的初始输出与预训练TCN完全一致。',flush=True)
    del adapted
    def control_maker():
        model=old.build_network(torch,False,bins);model.load_state_dict(base_state);return model
    control_phases=[dict(name='TCN继续训练A',epochs=args.freeze_epochs,freeze_trunk=False,trunk_lr=.0001,attention_lr=.0001),
                    dict(name='TCN继续训练B',epochs=args.joint_epochs,freeze_trunk=False,trunk_lr=.0001,attention_lr=.0001)]
    print('\n第2/3段：TCN等额外训练步数对照。',flush=True)
    control,control_result=run_stage(torch,old,out/'tcn_control',control_maker,control_phases,
        x,y,target,fit,val,scaling,seed,signature)
    warm_phases=[dict(name='冻结主干仅训练ECA',epochs=args.freeze_epochs,freeze_trunk=True,trunk_lr=0.,attention_lr=.001),
                 dict(name='主干与ECA联合微调',epochs=args.joint_epochs,freeze_trunk=False,trunk_lr=.0001,attention_lr=.0003)]
    print('\n第3/3段：恒等初始化注意力微调。',flush=True)
    warm,warm_result=run_stage(torch,old,out/'identity_tecan',lambda:warm_network(torch,old,bins,base_state),
        warm_phases,x,y,target,fit,val,scaling,seed,signature)
    rows=[];pred_rows=[]
    for name,model,result in [('TCN预训练起点',base,base_result),('TCN等步数继续训练',control,control_result),
                              ('恒等初始化TECAN微调',warm,warm_result)]:
        fit_score,_=evaluate(torch,model,x,y,fit,target,scaling)
        val_score,pred=evaluate(torch,model,x,y,val,target,scaling)
        rows.append(dict(model=name,fit_MAE=fit_score['mae'],validation_MAE=val_score['mae'],
            validation_MSE=val_score['mse'],prediction_range_cycles=val_score['prediction_range'],best_epoch=result['best_epoch']))
        pred_rows.extend(dict(model=name,battery_id=names[i],role='inner_validation',true_CL=float(target[i]),prediction_CL=float(p))
                         for i,p in zip(val,pred))
    baseline=float(np.abs(target[val]-np.median(target[fit])).mean())
    report=dict(status='inner_warm_tecan_diagnostic_complete',summary=rows,
        median_baseline_validation_MAE=baseline,selection_metric='inner validation MSE',
        identity_tecan_better_than_continued_tcn=rows[2]['validation_MSE']<rows[1]['validation_MSE']-1e-6,
        outer_test_evaluated=False,independent_validation=False,
        note='Best epoch0 means no post-initialization improvement. This is not five-repeat test performance.')
    csv_write(out/'summary.csv',rows);csv_write(out/'inner_predictions.csv',pred_rows)
    json_write(out/'SUCCESS.json',report);print_summary(report,out)


def print_summary(report,out):
    print('\n内部验证汇总（不是最终五次测试精度）：',flush=True)
    print(f"训练中位数基线：验证MAE={report['median_baseline_validation_MAE']:.2f}循环",flush=True)
    for r in report['summary']:
        print(f"{r['model']}：拟合MAE={r['fit_MAE']:.2f}，验证MAE={r['validation_MAE']:.2f}，"
              f"验证MSE={r['validation_MSE']:.5f}，最佳轮={r['best_epoch']}",flush=True)
    print('注意力微调的内部验证MSE优于继续训练TCN：'+str(report['identity_tecan_better_than_continued_tcn']),flush=True)
    print('结果目录：'+str(out),flush=True)
    print('请发送末尾汇总；不自动启动五次外层测试。',flush=True)


if __name__=='__main__':main()
