@echo off
setlocal enabledelayedexpansion

rem === Parametres ===
set "CSV=payloads\device_sensor_index_pairs.csv"
if not "%~1"=="" set "CSV=%~1"
set "START=2026-01-01 00:00:00"
set "END=2026-09-09 00:00:00"
set "LOG=logs\batch_run.log"

if not exist logs mkdir logs

echo ===== Batch run started %date% %time% ===== >> "%LOG%"

for /F "usebackq skip=1 tokens=1,2 delims=," %%A in ("%CSV%") do (
    echo [%%A / %%B] start >> "%LOG%"
    .\venv\Scripts\python.exe .\src\main.py --device_id %%A --sensor_id %%B --start "%START%" --end "%END%" >> "%LOG%" 2>&1
    echo [%%A / %%B] exit code: !errorlevel! >> "%LOG%"
)

echo ===== Batch run finished %date% %time% ===== >> "%LOG%"
echo Termine. Voir %LOG% pour le detail.
