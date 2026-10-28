@echo off
setlocal
cd /d "%~dp0"
python prepare_paths.py
if errorlevel 1 goto done
python -u 07_stratified_repeat.py --run models\train_20261020_194327_170397 --epochs 150 --patience 30 --threads 2
:done
pause
