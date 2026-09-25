@echo off
setlocal

set "ROOT=C:\Users\hp\OneDrive\Desktop\MaBs"

echo.
echo ==========================================
echo LeadScout MABS v3 Update
echo ==========================================
echo.

if not exist "%ROOT%\main.py" (
    echo ERROR: MABS folder not found:
    echo %ROOT%
    pause
    exit /b 1
)

if not exist "%ROOT%\web" mkdir "%ROOT%\web"

echo [1/4] Backing up current files...
copy /Y "%ROOT%\main.py" "%ROOT%\main_before_v3_backup.py" >nul
if exist "%ROOT%\web\index.html" copy /Y "%ROOT%\web\index.html" "%ROOT%\web\index_before_v3_backup.html" >nul

echo [2/4] Installing new main.py...
copy /Y "%~dp0main.py" "%ROOT%\main.py" >nul

echo [3/4] Installing new web UI...
copy /Y "%~dp0web\index.html" "%ROOT%\web\index.html" >nul

echo [4/4] Checking Python syntax...
python -m py_compile "%ROOT%\main.py"
if errorlevel 1 (
    echo.
    echo ERROR: main.py syntax check failed.
    echo Your backup is still available:
    echo %ROOT%\main_before_v3_backup.py
    pause
    exit /b 1
)

echo.
echo ==========================================
echo UPDATE COMPLETE
echo ==========================================
echo.
echo Installed:
echo   Multi-location search
echo   Parallel location searches
echo   Parallel detail extraction
echo   Client-side filtering
echo   Column sorting
echo   Phone-only filter
echo   Minimum rating filter
echo   Dark/Light mode
echo   Search location in results
echo.
echo Backup:
echo   %ROOT%\main_before_v3_backup.py
echo   %ROOT%\web\index_before_v3_backup.html
echo.
pause
