# 芯源智测：最终源代码与复现实验

## 直接体验
无需开发环境时，请使用交付目录04的免安装压缩包，全部解压后运行START.bat。
本目录是可编辑源代码，不是另一份免安装运行环境。

## 从源码启动
Windows 64位，建议Python 3.13；CPU即可。先创建独立环境，再安装推理依赖：
```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements_inference.txt
python prepare_paths.py
python -B tcn_workbench_blue/server.py
```
访问 http://127.0.0.1:8504 ，保留启动窗口。关闭窗口或Ctrl+C停止服务。
环境已配置时可双击START_SOURCE.bat；VERIFY_SOURCE.bat只做25颗回放一致性核验。
运行包只包含推理依赖；它不是训练依赖安装包。

## 文件说明
- 01_check_curves.py：原始充电曲线检查。
- 02_prepare_curves.py：容量轴对齐、充电阶段筛选、缺失标记、NPZ构建。
- 03_knee_train.py：模型定义、训练内预处理及膝点候选生成。按训练时原字节保留，不要随意修改后套用旧权重。
- 07_stratified_repeat.py：最终五次分层重复留出主实验，固定种子20261021。5次并非五折。
- 13_trust_tcn.py：74训练/25校准/25测试的可信分析补充实验。
- inference.py：从03原函数抽取的推理实现，供网页减少训练依赖；不改变网络和预处理算法。
- tcn_workbench_blue/：最终蓝紫版网页与本地服务源码。
- results/curves_*：已处理曲线、质量记录、标签及来源映射。
- models/stratified_*：五次主实验权重、配置、划分、训练曲线与逐颗预测。
- models/trust_tcn_*：补充实验模型、校准参数、复核规则及结果。
- models/train_*：为最终实验复用的历史膝点标签与来源记录，不作为最终成绩。
- supplementary/：报告中用过的内部训练诊断脚本，运行时应复制到项目根目录；内部验证成绩不作为最终测试精度。

## 复现主实验和可信分析
安装完整训练依赖 `python -m pip install -r requirements_training.txt`。
首次运行先执行 `python prepare_paths.py`，它仅替换当前副本中的本地路径，并将原记录保存为config.original.json。
已有曲线与膝点标签可直接复用，不必下载原始MAT或重跑03。
```
python 07_stratified_repeat.py --run models/train_20261020_194327_170397 --epochs 150 --patience 30 --threads 2
python 13_trust_tcn.py --source models/stratified_20261021_095330_972431 --epochs 150 --patience 30 --threads 2
```
CPU全量重训可能耗时数小时；不在打开工作台时自动训练。新结果生成新目录，不覆盖原权重。
训练可能受软件版本与数值实现影响，保存结果与权重核验是准确复核既有成绩的入口。
历史环境见requirements_recorded_versions.txt；可安装依赖列表与历史环境记录分开保留。

## 功能与结果边界
网页固定第1次划分99颗训练的TCN模型，提供点预测、质量提示及扰动敏感性分析。
独立区间和正式复核分数来自另一个74颗训练模型，仅在补充实验中报告，未接入网页。
网页的“区间未校准”表示当前网页模型尚未接入校准，并非没有开展补充实验。
