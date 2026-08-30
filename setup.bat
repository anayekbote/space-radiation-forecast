@echo off
echo =========================================================
echo Space Radiation Hazard Forecast: Environment Setup (Win10)
echo =========================================================
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo [✓] Environment configured. Launch with run.bat
pause
