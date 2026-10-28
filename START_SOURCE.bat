@echo off
setlocal
cd /d "%~dp0"
python prepare_paths.py
if errorlevel 1 goto done
python -B tcn_workbench_blue\server.py
:done
pause
