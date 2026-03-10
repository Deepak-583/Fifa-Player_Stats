@echo off
cd /d "%~dp0"
echo Installing dependencies (if needed)...
pip install -r requirements.txt -q
echo.
echo Starting FIFA Player Stats App...
python -m streamlit run app.py
pause
