@echo off
echo ============================================
echo   AutoDefect AI - Starting Server...
echo ============================================
echo.

call conda activate genai

echo Loading all models and starting server on http://localhost:5000
echo Press Ctrl+C to stop the server.
echo.

python src/app/backend.py --fold 4 --port 5000
