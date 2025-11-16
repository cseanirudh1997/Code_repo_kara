@echo off
title AI Text-to-SQL Application
color 0A

echo.
echo   🤖 AI TEXT-TO-SQL APPLICATION
echo   ============================
echo.

echo 📦 Installing required packages...
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Package installation failed!
    echo 💡 Try: pip install --user -r requirements.txt
    pause
    exit /b 1
)

echo ✅ Packages installed successfully!
echo.
echo 🚀 Starting application...
echo 📱 Open browser to: http://127.0.0.1:8000
echo 🛑 Press Ctrl+C to stop
echo.

python complete_text_to_sql.py

pause