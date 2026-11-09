@echo off
echo ===============================================================================
echo EXECUTING REPRODUCIBLE AI/ML RESEARCH PIPELINE
echo Target Directory: D:\AI_Research_Paper
echo ===============================================================================

cd /d D:\AI_Research_Paper

if not exist ".venv" (
    echo [*] Creating local Python virtual environment (.venv)...
    python -m venv .venv
)

echo [*] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [*] Verifying and installing required Python packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo [*] Launching Master Research Pipeline...
python src\research_pipeline.py

echo ===============================================================================
echo Pipeline execution complete. Press any key to exit.
echo ===============================================================================
pause
