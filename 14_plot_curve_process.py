"""Publication figure from one real MATR cycle; reuse 02 preprocessing and verify saved tensor."""
import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


COLORS=dict(blue='#466F87',cyan='#78B6C8',yellow='#F2D98E',coral='#E48578',green='#98AF1E',ink='#263A47',gray='#A8B2B8')


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for a in iter(lambda:f.read(1048576),b''):h.update(a)
    return h.hexdigest()
def write_csv(path,headers,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(headers);w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',type=Path,default=Path(r'D:\xinyuanzhice'))
    ap.add_argument('--data',type=Path,help='02曲线准备目录；默认最近完成的1024点目录')
    ap.add_argument('--battery',default='b2c1')
    ap.add_argument('--cycle',type=int,default=100)
    ap.add_argument('--dpi',type=int,default=600)
    args=ap.parse_args()
    if not 1<=args.cycle<=100:ap.error('循环须在1至100之间')
    if not 300<=args.dpi<=1200:ap.error('PNG分辨率请设为300至1200dpi')
    import numpy as np
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    here=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('curve02',here/'02_prepare_curves.py')
    prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
    if args.data is None:
        dirs=[p for p in (args.project/'results').glob('curves_*')
              if all((p/n).exists() for n in ['SUCCESS.json','config.json','source_map.csv','curves.npz'])]
        if not dirs:raise ValueError('未找到02结果，请用--data指定曲线准备目录')
        args.data=max(dirs,key=lambda p:p.name)
    cfg=json.loads((args.data/'config.json').read_text(encoding='utf-8-sig'))
    rec=[r for r in read_csv(args.data/'source_map.csv') if r['battery_id']==args.battery and int(r['cumulative_cycle'])==args.cycle]
    if len(rec)!=1:raise ValueError('该电芯/循环没有唯一来源记录')
    rec=rec[0];source=Path(rec['source_file'])
    if not source.is_file():source=Path(cfg['project_root'])/'data'/'raw'/source.name
    if not source.is_file():raise FileNotFoundError('找不到原始MAT：'+str(source))
    stat_before=dict(size=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns)
    print(f'只读取一条真实循环：{args.battery}，第{args.cycle}循环。',flush=True)
    with h5py.File(source,'r') as f:
        group=f[f['batch/cycles'][int(rec['source_index_0based']),0]]
        a={name:prep.read_vector(f,group,name,int(rec['source_cycle'])-1) for name in prep.FIELDS}
    with np.load(args.data/'curves.npz',allow_pickle=False) as z:
        ids=z['battery_id'].astype(str).tolist();i=ids.index(args.battery)
        grid=z['q_grid'].copy();saved=z['X'][i,:,args.cycle-1].copy();saved_mask=z['observed'][i,args.cycle-1].copy()
    if len(grid)!=1024:raise ValueError('本图要求已保存的1024点容量网格')
    aligned,mask,quality=prep.prepare_cycle(a,grid,float(cfg['max_q_gap']))
    if quality['status']!='aligned':raise ValueError('当前循环不能对齐：'+quality['status'])
    if not np.array_equal(mask,saved_mask):raise ValueError('重算mask与已保存张量不一致，停止绘图')
    np.testing.assert_allclose(aligned,saved,rtol=0,atol=0,equal_nan=True,
        err_msg='重算V/I/T与已保存张量不一致，停止绘图')
    finite=np.logical_and.reduce([np.isfinite(a[k]) for k in prep.FIELDS])
    dis=np.flatnonzero(np.isfinite(a['Qd'])&(a['Qd']>.01));end=int(dis[0]) if len(dis) else len(a['V'])
    candidate=finite&(np.arange(len(finite))<end)&(a['I']>.02)&(a['Qc']>0)
    raw_indices=np.flatnonzero(candidate);q=a['Qc'][candidate]
    selected=[];previous=-np.inf
    for j,v in enumerate(q):
        if v>previous:selected.append(j);previous=float(v)
    raw_indices=raw_indices[np.array(selected)];q=a['Qc'][raw_indices]
    channels=np.stack([a[k][raw_indices] for k in ['V','I','T']])
    if not len(raw_indices):raise ValueError('没有选中的真实观测')
    # Keep source units explicit: source time and current have not been independently certified.
    time_source=a['t']-float(a['t'][0]);font_name=None
    for name in ['Microsoft YaHei','SimHei','Noto Sans CJK SC','SimSun']:
        try:font_manager.findfont(font_manager.FontProperties(family=name),fallback_to_default=False);font_name=name;break
        except ValueError:pass
    if font_name is None:raise RuntimeError('未找到中文字体，请在Windows电脑运行，或安装思源黑体后重试')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[font_name],
        'axes.unicode_minus':False,'font.size':10.5,'axes.labelsize':10.5,
        'xtick.labelsize':9,'ytick.labelsize':9,'axes.linewidth':.8,
        'text.color':COLORS['ink'],'axes.labelcolor':COLORS['ink'],
        'xtick.color':COLORS['ink'],'ytick.color':COLORS['ink'],
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','savefig.facecolor':'white'})
    fig=plt.figure(figsize=(13.2,8.4),facecolor='white')
    gs=fig.add_gridspec(4,3,height_ratios=[1,1,1,.22],hspace=.64,wspace=.40,
        left=.067,right=.978,bottom=.17,top=.845)
    axes=[[fig.add_subplot(gs[row,col]) for col in range(3)] for row in range(3)]
    ylabels=['电压 / V','电流 / 源单位','温度 / °C']
    missing=np.flatnonzero(np.diff(np.r_[False,~mask,False].astype(int)))
    ranges=[];step=float(grid[1]-grid[0])
    for left,right in zip(missing[::2],missing[1::2]):
        ranges.append((max(float(grid[0]),float(grid[left]-step/2)),min(float(grid[-1]),float(grid[right-1]+step/2))))
    for row,(field,ylabel) in enumerate(zip(['V','I','T'],ylabels)):
        for col in range(3):
            ax=axes[row][col];ax.spines[['top','right']].set_visible(False)
            ax.spines[['left','bottom']].set_color('#BAC5CB');ax.grid(axis='y',color='#E8EDF0',linewidth=.65)
            ax.set_axisbelow(True);ax.set_ylabel(ylabel,labelpad=5)
        ax=axes[row][0]
        ax.plot(time_source,a[field],color=COLORS['blue'],lw=1.35)
        ymin,ymax=ax.get_ylim()
        ax.fill_between(time_source,ymin,ymax,where=candidate,color=COLORS['yellow'],alpha=.5,linewidth=0)
        ax.set_ylim(ymin,ymax)
        if end<len(time_source):ax.axvline(time_source[end],color=COLORS['coral'],lw=1,ls='--')
        ax=axes[row][1]
        ax.plot(q,channels[row],color=COLORS['cyan'],lw=1.1)
        ax.scatter(q,channels[row],s=6,color=COLORS['cyan'],edgecolors='none',zorder=3)
        ax.set_xlim(float(grid[0])-.01,float(grid[-1])+.01)
        ax=axes[row][2]
        ax.plot(grid,aligned[row],color=COLORS['blue'],lw=1.3)
        ax.plot(grid[mask][::18],aligned[row,mask][::18],ls='none',marker='o',ms=2.6,
            markerfacecolor='white',markeredgewidth=.7,color=COLORS['blue'])
        for lo,hi in ranges:ax.axvspan(lo,hi,color=COLORS['coral'],alpha=.28,lw=0)
        ax.set_xlim(float(grid[0])-.01,float(grid[-1])+.01)
        # Match vertical scales within each row for valid visual comparison.
        values=np.concatenate([a[field][np.isfinite(a[field])],channels[row]])
        lo,hi=float(values.min()),float(values.max());margin=max((hi-lo)*.10,.01)
        for col in range(3):axes[row][col].set_ylim(lo-margin,hi+margin)
        if row<2:
            for col in range(3):axes[row][col].tick_params(labelbottom=False)
    axes[2][0].set_xlabel('相对时间 t / 源单位')
    for col in [1,2]:axes[2][col].set_xlabel('充电容量 Qc / Ah')
    # Show genuine nearest-neighbour mapping at a supported, fixed capacity location.
    good=grid[mask];mid=float(good[np.argmin(np.abs(good-.42))]);width=.045
    in_grid=mask&(grid>=mid-width/2)&(grid<=mid+width/2)
    in_raw=(q>=mid-width/2)&(q<=mid+width/2)
    inset=axes[0][2].inset_axes([.48,.12,.49,.48])
    inset.set_facecolor('#FCFCFC')
    inset.scatter(q[in_raw],channels[0,in_raw],s=11,facecolors='white',edgecolors=COLORS['blue'],lw=.75,zorder=3)
    inset.plot(grid[in_grid],aligned[0,in_grid],color=COLORS['coral'],lw=.8,marker='x',ms=3,markeredgewidth=.6)
    inset.set_xlim(mid-width/2,mid+width/2);inset.tick_params(labelsize=6,pad=1)
    inset.set_title('最近邻局部放大',fontsize=7,pad=2)
    inset.grid(color='#E7ECEF',lw=.4)
    for spine in inset.spines.values():spine.set_color('#CBD4D9');spine.set_linewidth(.6)
    for col,text in enumerate(['A  原始循环与充电截段','B  选取的充电观测','C  固定1024点容量网格']):
        pos=axes[0][col].get_position();fig.text(pos.x0,pos.y1+.033,text,fontsize=12.3,fontweight='bold',color=COLORS['blue'])
    for col in [0,1]:
        note=fig.add_subplot(gs[3,col]);note.axis('off')
        text=(f'黄色：充电候选观测；保留源记录顺序' if col==0 else
              f'原始有效容量坐标 {len(q)} 个；最近邻对齐')
        note.text(0,.5,text,fontsize=8.6,color=COLORS['ink'],va='center')
    support=fig.add_subplot(gs[3,2]);support.imshow(mask[None,:].astype(int),aspect='auto',
        extent=[float(grid[0]),float(grid[-1]),0,1],cmap=ListedColormap([COLORS['coral'],COLORS['green']]),vmin=0,vmax=1,interpolation='nearest')
    support.set_yticks([]);support.set_xticks([.01,.5,1.]);support.tick_params(labelsize=7,pad=1,length=2)
    for s in support.spines.values():s.set_visible(False)
    support.set_title(f'观测mask：有效{int(mask.sum())} / 1024',fontsize=8.5,pad=3)
    fig.text(.5,.974,'早期充电曲线截取与容量轴对齐',ha='center',fontsize=18,fontweight='bold',color=COLORS['ink'])
    fig.text(.5,.929,f'真实电芯 {args.battery}  ·  第{args.cycle}循环  ·  V / I / T未归一化  ·  有效网格 {mask.mean():.2%}',
             ha='center',fontsize=11,color=COLORS['blue'])
    handles=[Line2D([],[],color=COLORS['blue'],lw=1.5,label='原始记录 / 对齐结果'),
             Patch(facecolor=COLORS['yellow'],alpha=.65,label='选取的充电观测'),
             Line2D([],[],color=COLORS['cyan'],marker='o',ms=3,lw=1,label='原始充电点'),
             Patch(facecolor=COLORS['coral'],alpha=.5,label='无观测支持网格'),
             Patch(facecolor=COLORS['green'],label='有效观测mask')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.072),ncol=5,frameon=False,fontsize=9.2,handlelength=1.8,columnspacing=1.4)
    note=(f'缺失来自真实记录：{int((~mask).sum())}个无支持网格，不外推、不插补。' if (~mask).any() else
          '该真实循环的1024个网格全部有观测支持，未人为添加缺失。')
    fig.text(.5,.04,note+' 电流与时间保留源单位；主图标记稀疏显示，实际网格仍为1024点。',ha='center',fontsize=9,color=COLORS['ink'])
    out=here/'results'/('curve_process_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(parents=True)
    stem=f'charging_curve_process_{args.battery}_cycle{args.cycle}'
    print('与已保存曲线核对通过，正在输出高清和矢量文件……',flush=True)
    fig.savefig(out/(stem+'.png'),dpi=args.dpi)
    fig.savefig(out/(stem+'.pdf'))
    fig.savefig(out/(stem+'.svg'))
    fig.savefig(out/'preview.png',dpi=160)
    plt.close(fig)
    write_csv(out/'selected_charge_points.csv',['raw_index_0based','Qc','V','I_source','T','t_source'],
        [[int(j),float(a['Qc'][j]),float(a['V'][j]),float(a['I'][j]),float(a['T'][j]),float(a['t'][j])] for j in raw_indices])
    write_csv(out/'aligned_1024_points.csv',['Qc','V','I_source','T','observed'],
        [[float(grid[j])]+[float(v) if np.isfinite(v) else '' for v in aligned[:,j]]+[int(mask[j])] for j in range(len(grid))])
    after=dict(size=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns)
    if stat_before!=after:raise RuntimeError('读取期间源文件状态变化，请检查')
    metadata=dict(battery_id=args.battery,cycle=args.cycle,source_file=str(source),source_record=rec,
        source_stat_unchanged=True,raw_MAT_full_hash_computed=False,
        preprocessing_script_sha256=sha(here/'02_prepare_curves.py'),data_directory=str(args.data),
        matches_saved_tensor_exactly=True,quality=quality,palette=COLORS,png_dpi=args.dpi,
        figure_size_inches=[13.2,8.4],artificial_missingness_added=False,
        selected_example_reason='Fixed b2c1/cycle100 illustrates its real terminal unsupported grid; not selected by prediction error.')
    (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    caption=(f'图 早期充电曲线截取与容量轴对齐。以MATR电芯{args.battery}第{args.cycle}循环为例，'
             'A为原始V/I/T及充电观测选择，B为沿Qc组织的有效充电观测，C为固定1024点最近邻对齐结果。'
             f'观测mask中有效{int(mask.sum())}点，无支持{int((~mask).sum())}点；未外推或插补。'
             '电流和时间保留源单位，局部图用于展示最近邻映射。')
    (out/'图注.txt').write_text(caption,encoding='utf-8-sig')
    print('\n绘图完成：'+str(out),flush=True)
    print(f'原始点{len(a["V"])}，有效充电坐标{len(q)}，有效网格{int(mask.sum())}，无支持网格{int((~mask).sum())}。',flush=True)
    print(f'PNG {args.dpi}dpi；PDF/SVG为矢量图。没有训练模型或修改原数据。',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        print('绘图失败：'+str(e),flush=True);sys.exit(1)
