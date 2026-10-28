"""Local TCN inference. No training or interval calibration."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''): h.update(part)
    return h.hexdigest()


def read_json(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def read_curves(source):
    # Uploaded data are arrays only. No pickle/object-array deserialization.
    with zipfile.ZipFile(source) as z:
        if sum(i.file_size for i in z.infolist()) > 256*1024*1024:
            raise ValueError('解压后数据超过256MB，请拆分电芯文件')
    if hasattr(source,'seek'): source.seek(0)
    with np.load(source,allow_pickle=False) as z:
        required={'X','observed','q_grid','cycles','channels','battery_id'}
        if not required.issubset(z.files): raise ValueError('请使用02生成的曲线NPZ；缺少字段：'+','.join(sorted(required-set(z.files))))
        x=z['X']; observed=z['observed']; ids=z['battery_id']; grid=z['q_grid']; cycles=z['cycles']; channels=z['channels']
    if x.ndim!=4 or x.shape[1:]!=(3,100,1024) or not 1<=len(x)<=124:
        raise ValueError('X必须为N×3×100×1024，N在1至124之间')
    if x.dtype.kind!='f' or np.isinf(x).any(): raise ValueError('曲线必须为浮点数组；缺失用NaN，不能使用无穷值')
    if observed.dtype.kind!='b' or observed.shape!=(len(x),100,1024): raise ValueError('observed必须为N×100×1024布尔数组')
    if ids.shape!=(len(x),) or ids.dtype.kind not in ('U','S'): raise ValueError('battery_id必须为字符串数组')
    ids=ids.astype(str).tolist()
    if len(ids)!=len(set(ids)) or any(not s or len(s)>80 for s in ids): raise ValueError('电芯编号为空、重复或过长')
    if grid.shape!=(1024,) or not np.allclose(grid,np.linspace(.01,1.,1024),atol=1e-8,rtol=0):
        raise ValueError('容量网格必须为0.01至1.00的1024点固定网格')
    if not np.array_equal(cycles,np.arange(1,101)): raise ValueError('循环必须为1至100，不能把后续循环前移填补')
    if channels.astype(str).tolist()!=['V_source','I_source','T_source']: raise ValueError('通道顺序应为V_source、I_source、T_source')
    if not np.array_equal(np.isfinite(x),np.broadcast_to(observed[:,None],x.shape)):
        raise ValueError('观测mask与V/I/T有限值不一致；请使用02脚本重新核对')
    if not observed.reshape(len(x),-1).any(axis=1).all(): raise ValueError('存在完全没有有效观测的电芯')
    return dict(X=x.astype('float32',copy=False),observed=observed,ids=ids,grid=grid)


class Workbench:
    def __init__(self,root,run=None):
        self.root=Path(root)
        torch.set_num_threads(2)
        spec=importlib.util.spec_from_file_location('workbench03',self.root/'inference.py')
        self.old=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.old)
        if run is None:
            candidates=[p for p in (self.root/'models').glob('stratified_*') if (p/'SUCCESS.json').exists()]
            if not candidates: raise ValueError('未找到已完成的stratified实验')
            run=max(candidates,key=lambda p:p.name)
        self.run=Path(run); self.config=read_json(self.run/'config.json')
        if sha(self.root/'03_knee_train.py')!=self.config['source03_sha256']:
            raise ValueError('03模型定义与该次训练时不一致')
        # Fixed repeat1 as a demonstration checkpoint; no selection by test performance.
        self.repeat=1; self.folder=self.run/'repeat_1'; self.path=self.folder/'tcn.pt'
        ck=torch.load(self.path,map_location='cpu',weights_only=True)
        if ck['attention'] or ck['input_window']!=100: raise ValueError('当前页面仅支持前100循环TCN模型')
        self.bins=int(ck['capacity_bins']);self.prep=ck['preprocessing'];self.scaling=ck['target_scaling']
        self.net=self.old.build_network(torch,False,self.bins); self.net.load_state_dict(ck['state_dict']);self.net.eval()
        self.training=set(ck['training_cells']); self.model_sha=sha(self.path)
        self.saved={r['battery_id']:r for r in rows(self.folder/'predictions.csv') if r['model']=='tcn'}
        if len(self.saved)!=25 or self.training & set(self.saved): raise ValueError('训练与留出身份异常')
        self.summary=rows(self.run/'summary.csv')
        self.data_path=self.root/Path(self.config['data'])/'curves.npz'
        if sha(self.data_path)!=self.config['tensor_sha256']: raise ValueError('曲线文件指纹与训练记录不同')
        self.curves=read_curves(self.data_path)

    def infer(self,curves,index,mode='upload'):
        a=curves['X'][index:index+1]; mask=curves['observed'][index:index+1]
        values=self.old.transform(a,mask,self.prep,self.bins)
        raw=float(self.old.predict(torch,self.net,values,np.array([0]),self.scaling,'cpu',8)[0])
        cl=max(raw,100.); name=curves['ids'][index]
        reasons=[]; channel=[]
        for c,label in enumerate(['电压','源单位电流','温度']):
            v=a[0,c]; valid=np.isfinite(v); lo=self.prep['min'][c]; hi=lo+self.prep['scale'][c]
            outside=float(((v[valid]<lo)|(v[valid]>hi)).mean())
            channel.append(dict(channel=label,minimum=float(v[valid].min()),maximum=float(v[valid].max()),outside_training_range_fraction=outside))
            if outside>0: reasons.append(label+'存在超出训练缩放范围的观测')
        empty=int((~mask[0].any(axis=1)).sum()); coverage=float(mask.mean())
        if coverage<1: reasons.append(f'部分曲线点缺失，已按训练参数插补；有效覆盖率{coverage:.2%}')
        if empty: reasons.append(f'有{empty}个循环没有有效曲线')
        if raw<100: reasons.append('原始寿命输出低于已观测窗口，按既有实验规则截到100循环')
        if mode=='upload': reasons.append('新数据的化学体系、单位及采集工况需要人工核对，系统未完成跨域有效性验证')
        result=dict(battery_id=name,mode=mode,model='TCN',repeat=1,model_sha256=self.model_sha,
            input_window_cycles=100,raw_CL_cycles=raw,predicted_CL_cycles=cl,predicted_RUL100_cycles=max(cl-100,0.),
            observed_fraction=coverage,empty_cycles=empty,channels=channel,
            review_status='建议复核' if reasons else '基础输入检查通过，适用性仍需复核',
            review_reasons=reasons,interval=None,interval_status='尚未独立校准',
            risk_probability=None,assessment_type='输入质量与训练范围检查；不是经验证的风险拒判阈值')
        if mode=='replay':
            if name not in self.saved: raise ValueError('回放仅支持固定模型留出的25颗电芯')
            saved=self.saved[name]
            result.update(actual_CL_cycles=float(saved['true_CL']),saved_prediction_CL_cycles=float(saved['prediction_CL']),
                          reproduction_absolute_difference=abs(cl-float(saved['prediction_CL'])))
        # Downsample only for display; prediction always uses full 1024-point input and saved preprocessing.
        plots=[]
        for cycle in [1,10,50,100]:
            plots.append(dict(cycle=cycle,points=[dict(q=float(curves['grid'][j]),
                V=None if not np.isfinite(a[0,0,cycle-1,j]) else float(a[0,0,cycle-1,j]),
                I=None if not np.isfinite(a[0,1,cycle-1,j]) else float(a[0,1,cycle-1,j]),
                T=None if not np.isfinite(a[0,2,cycle-1,j]) else float(a[0,2,cycle-1,j])) for j in range(0,1024,8)]))
        result['curve_preview']=plots
        return result

    def verify(self):
        ids=list(self.saved); lookup={s:i for i,s in enumerate(self.curves['ids'])}
        indices=np.array([lookup[s] for s in ids])
        a=self.curves['X'][indices]; mask=self.curves['observed'][indices]
        values=self.old.transform(a,mask,self.prep,self.bins)
        pred=np.maximum(self.old.predict(torch,self.net,values,np.arange(25),self.scaling,'cpu',8),100.)
        expected=np.array([float(self.saved[s]['prediction_CL']) for s in ids])
        delta=float(np.max(np.abs(pred-expected)))
        if delta>.01: raise ValueError(f'页面推理与保存预测不一致：最大差{delta:.6f}循环')
        return dict(status='PASS',checked_heldout_cells=25,max_prediction_difference_cycles=delta,
                    trained=False,checkpoint=str(self.path),model_sha256=self.model_sha)

    def info(self):
        return dict(ids=list(self.saved),summary=self.summary,model_sha256=self.model_sha,
                    training_cells=len(self.training),heldout_cells=25,repeat=1,
                    input='前100循环充电V/I/T曲线',interval_status='尚未独立校准',
                    catalog_cells=len(self.curves['ids']),observed_fraction=float(self.curves['observed'].mean()),
                    replay_records=[dict(battery_id=k,true_CL=float(v['true_CL']),prediction_CL=float(v['prediction_CL']))
                                    for k,v in self.saved.items()])

    def batch(self,curves,indices,mode):
        result=[]
        for i in indices:
            row=self.infer(curves,i,mode)
            row.pop('curve_preview',None)
            result.append(row)
        return result

    def stability(self,name):
        if name not in self.saved:raise ValueError('压力测试仅支持当前25颗留出电芯')
        i=self.curves['ids'].index(name)
        original=self.curves['X'][i:i+1].copy();observed=self.curves['observed'][i:i+1].copy()
        def prediction(a,m,clip=True):
            v=self.old.transform(a,m,self.prep,self.bins)
            raw=float(self.old.predict(torch,self.net,v,np.array([0]),self.scaling,'cpu',8)[0])
            return max(raw,100.) if clip else raw
        baseline=prediction(original,observed);result=[]
        specs=[('noise',.005),('noise',.01),('noise',.02),('missing',.01),('missing',.05),('missing',.1),('tail',5),('tail',10)]
        for kind,level in specs:
            values=[];raw_values=[]
            for repeat in range(1 if kind=='tail' else 3):
                rng=np.random.default_rng(20261022+repeat)
                a=original.copy();m=observed.copy()
                if kind=='noise':
                    scale=np.array(self.prep['scale'],dtype='float32')[None,:,None,None]
                    a=a+rng.normal(size=a.shape).astype('float32')*scale*level
                    label=f'通道噪声 σ={level*100:g}%训练缩放跨度'
                elif kind=='missing':
                    keep=rng.random(size=m.shape)>=level;m=m&keep
                    a=np.where(m[:,None],a,np.nan)
                    label=f'随机缺失{level*100:g}%网格点'
                else:
                    m[:,-int(level):]=False;a[:,:,-int(level):]=np.nan
                    label=f'末尾{int(level)}循环曲线缺失'
                raw_value=prediction(a,m,False);raw_values.append(raw_value);values.append(max(raw_value,100.))
            shifts=np.array(values)-baseline
            result.append(dict(scenario=label,repeats=len(values),mean_prediction=float(np.mean(values)),
                mean_signed_change=float(shifts.mean()),mean_absolute_change=float(np.abs(shifts).mean()),
                minimum_prediction=float(min(values)),maximum_prediction=float(max(values)),
                raw_mean_prediction=float(np.mean(raw_values)),
                below_window_fraction=float((np.array(raw_values)<100.).mean())))
        return dict(battery_id=name,baseline_CL_cycles=baseline,scenarios=result,seed=20261022,
                    model_sha256=self.model_sha,
                    warning='固定模型上的输入扰动敏感性实验。寿命按原规则截到100循环，原始输出与截断比例单独记录。不是置信区间、误差上界、安全概率或全工况可靠性证明。')
