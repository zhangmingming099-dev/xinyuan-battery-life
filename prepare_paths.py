from pathlib import Path
import json, shutil
ROOT=Path(__file__).resolve().parent
data=ROOT/'results'/'curves_20261020_183314_791763'
main=ROOT/'models'/'stratified_20261021_095330_972431'
old=ROOT/'models'/'train_20261020_194327_170397'
trust=ROOT/'models'/'trust_tcn_20261022_145654_589873'
changes=[(old/'config.json',{'data_directory':str(data)}),
         (main/'config.json',{'data':str(data),'source_run':str(old)}),
         (trust/'config.json',{'data':str(data),'source':str(main)})]
for p,patch in changes:
    backup=p.with_name('config.original.json')
    if not backup.exists():shutil.copy2(p,backup)
    obj=json.loads(p.read_text(encoding='utf-8-sig'));obj.update(patch)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Local paths prepared. Original records saved as config.original.json.')
