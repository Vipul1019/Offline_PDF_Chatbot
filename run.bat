@echo off
echo ================================================
echo   Offline PDF Chatbot
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Install dependencies if not already installed
echo Checking dependencies...
pip show gradio >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies (first run only)...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies.
        pause
        exit /b 1
    )
)

echo.
echo Starting chatbot at http://127.0.0.1:7860
echo Make sure Ollama is running in another window: ollama serve
echo.

python app.py

pause
