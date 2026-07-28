@echo off
setlocal

cd /d "%~dp0.." || exit /b 1

where conda >nul 2>nul
if errorlevel 1 (
  echo ERROR: conda was not found. Run this from Anaconda Prompt.
  exit /b 1
)

call conda create -n santapp-ruler python=3.11 -y || exit /b 1
call conda activate santapp-ruler || exit /b 1
python -m pip install --upgrade pip setuptools wheel || exit /b 1
python -m pip install -r requirements-torch-cu128.txt || exit /b 1
python -m pip install -r requirements.txt || exit /b 1
python -m santapp_ruler doctor
