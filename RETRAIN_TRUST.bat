@echo off
setlocal
cd /d "%~dp0"
python prepare_paths.py
if errorlevel 1 goto done
python -u 13_trust_tcn.py --source models\stratified_20261021_095330_972431 --epochs 150 --patience 30 --threads 2
:done
pause
