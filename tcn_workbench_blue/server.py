"""Dependency-free localhost dashboard; model math uses the existing numpy/torch environment."""
import argparse
import csv
import io
import json
from pathlib import Path
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
import webbrowser
import zipfile
import numpy as np
from core import Workbench, read_curves


def create_server(workbench,port):
    token=secrets.token_urlsafe(32); assets=Path(__file__).resolve().parent
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,content,mime='application/json; charset=utf-8',filename=None):
            data=content if isinstance(content,bytes) else (json.dumps(content,ensure_ascii=False,allow_nan=False).encode() if not isinstance(content,str) else content.encode())
            self.send_response(status);self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            if filename:self.send_header('Content-Disposition','attachment; filename="'+filename+'"')
            self.end_headers();self.wfile.write(data)
        def allowed_host(self):
            return self.headers.get('Host') in (f'127.0.0.1:{port}',f'localhost:{port}')
        def do_GET(self):
            if not self.allowed_host():self.reply(403,dict(error='仅允许本机访问'));return
            path=urlparse(self.path).path
            if path=='/': self.reply(200,(assets/'index.html').read_text(encoding='utf-8').replace('__TOKEN__',token),'text/html; charset=utf-8')
            elif path=='/api/info': self.reply(200,workbench.info())
            elif path=='/style.css': self.reply(200,(assets/'style.css').read_text(encoding='utf-8'),'text/css; charset=utf-8')
            elif path=='/app.js': self.reply(200,(assets/'app.js').read_text(encoding='utf-8').replace('__TOKEN__',token),'text/javascript; charset=utf-8')
            elif path=='/favicon.ico': self.reply(204,b'','image/x-icon')
            else:self.reply(404,dict(error='页面不存在'))
        def do_POST(self):
            if not self.allowed_host() or self.headers.get('X-Workbench-Token')!=token:
                self.reply(403,dict(error='请从本机工作台页面操作'));return
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=200*1024*1024: raise ValueError('文件或请求为空，或超过200MB')
                body=self.rfile.read(n)
                if len(body)!=n: raise ValueError('数据传输不完整')
                path=urlparse(self.path).path
                if path=='/api/replay':
                    name=json.loads(body)['battery_id']
                    if name not in workbench.saved: raise ValueError('请选择留出电芯')
                    result=workbench.infer(workbench.curves,workbench.curves['ids'].index(name),'replay')
                    self.reply(200,result)
                elif path=='/api/upload':
                    curves=read_curves(io.BytesIO(body))
                    if len(curves['ids'])>20: raise ValueError('单次上传最多20颗电芯，请拆分文件')
                    result=([workbench.infer(curves,0,'upload')] if len(curves['ids'])==1 else
                            workbench.batch(curves,range(len(curves['ids'])),'upload'))
                    self.reply(200,dict(results=result))
                elif path=='/api/batch':
                    names=json.loads(body).get('ids',list(workbench.saved))
                    if not isinstance(names,list) or not 1<=len(names)<=25 or len(set(names))!=len(names):
                        raise ValueError('请选择1至25颗不同电芯')
                    if any(n not in workbench.saved for n in names):raise ValueError('批量回放仅支持留出电芯')
                    indices=[workbench.curves['ids'].index(n) for n in names]
                    self.reply(200,dict(results=workbench.batch(workbench.curves,indices,'replay')))
                elif path=='/api/stability':
                    self.reply(200,workbench.stability(json.loads(body)['battery_id']))
                elif path=='/api/sample':
                    name=json.loads(body)['battery_id']
                    if name not in workbench.saved: raise ValueError('请选择留出电芯')
                    i=workbench.curves['ids'].index(name);buf=io.BytesIO()
                    np.savez_compressed(buf,X=workbench.curves['X'][i:i+1],observed=workbench.curves['observed'][i:i+1],
                        q_grid=workbench.curves['grid'],cycles=np.arange(1,101),
                        channels=np.array(['V_source','I_source','T_source']),battery_id=np.array([name]))
                    self.reply(200,buf.getvalue(),'application/octet-stream','curve_sample.npz')
                else:self.reply(404,dict(error='接口不存在'))
            except (ValueError,KeyError,TypeError,IndexError,OSError,zipfile.BadZipFile) as e:self.reply(400,dict(error=str(e)))
            except Exception as e:
                print('推理错误：'+repr(e),flush=True);self.reply(500,dict(error='处理失败，请查看启动窗口中的错误信息'))
        def log_message(self,format,*args): pass
    return HTTPServer(('127.0.0.1',port),Handler)


def self_test():
    x=np.ones((1,3,100,1024),dtype=np.float32);mask=np.ones((1,100,1024),dtype=bool)
    def pack(a=x,m=mask,grid=None):
        b=io.BytesIO();np.savez_compressed(b,X=a,observed=m,q_grid=np.linspace(.01,1,1024) if grid is None else grid,
            cycles=np.arange(1,101),channels=np.array(['V_source','I_source','T_source']),battery_id=np.array(['test']))
        b.seek(0);return b
    assert read_curves(pack())['X'].shape==x.shape
    invalid=x.copy();invalid[0,0,0,0]=np.nan
    for payload in [pack(a=invalid),pack(grid=np.linspace(0,1,1024))]:
        try:read_curves(payload)
        except ValueError:pass
        else:raise AssertionError('非法数据未被拦截')
    print('PASS：曲线格式、mask一致性和容量网格验证。')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    ap.add_argument('--run',type=Path);ap.add_argument('--port',type=int,default=8504)
    ap.add_argument('--verify-only',action='store_true');ap.add_argument('--no-browser',action='store_true');ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args()
    if a.self_test:self_test();return
    print('加载已有TCN与本地曲线，并核对25颗留出电芯的历史预测。不会训练模型。',flush=True)
    w=Workbench(a.root,a.run);print(json.dumps(w.verify(),ensure_ascii=False,indent=2),flush=True)
    if a.verify_only:return
    httpd=create_server(w,a.port);url=f'http://127.0.0.1:{a.port}'
    print('\n芯源智测已启动：'+url+'\n请保留此窗口；关闭窗口后网页服务结束。Ctrl+C停止。',flush=True)
    if not a.no_browser:threading.Timer(1,lambda:webbrowser.open(url)).start()
    try:httpd.serve_forever()
    except KeyboardInterrupt:pass
    finally:httpd.server_close()


if __name__=='__main__':
    try:main()
    except Exception as e:
        print('启动失败：'+str(e),flush=True);sys.exit(1)
