@echo off
REM TBScope launcher - uses the existing CUDA venv (torch 2.11.0+cu128).
REM Nothing to install; this interpreter already has everything.

set PY=C:\Users\OM\Om\Projects\wake_engine\.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8

if not exist "%PY%" (
    echo ERROR: CUDA interpreter not found at:
    echo   %PY%
    echo Edit run.cmd and point PY at a python with torch installed.
    exit /b 1
)

if "%1"=="" goto app
if "%1"=="app"     goto app
if "%1"=="gpu"     goto gpu
if "%1"=="data"    goto data
if "%1"=="train"   goto train
if "%1"=="predict" goto predict
goto usage

:app
echo Starting TBScope on http://localhost:8502
"%PY%" -m streamlit run "%~dp0app.py" --server.port 8502
goto :eof

:gpu
"%PY%" -c "import torch;print('torch',torch.__version__);print('cuda',torch.cuda.is_available());print('device',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY')"
goto :eof

:data
"%PY%" "%~dp0scripts\fetch_data.py"
"%PY%" "%~dp0scripts\fetch_shenzhen_subset.py" --per-class 340 --workers 14
"%PY%" "%~dp0scripts\prepare.py"
"%PY%" "%~dp0scripts\fetch_demo_xrays.py"
goto :eof

:train
"%PY%" "%~dp0scripts\train.py" --split split --epochs 14 --batch 32
"%PY%" "%~dp0scripts\train.py" --split cross_site --epochs 14 --batch 32
"%PY%" "%~dp0scripts\calibration_analysis.py"
goto :eof

:predict
"%PY%" "%~dp0scripts\predict_all.py" --tag "" --split split
goto :eof

:usage
echo Usage: run.cmd [app^|gpu^|data^|train^|predict]
echo   app      start the Streamlit app (default)
echo   gpu      check CUDA is working
echo   data     download and prepare all radiographs
echo   train    train pooled + cross-site, run calibration analysis
echo   predict  batch inference + Grad-CAM overlays
exit /b 1
