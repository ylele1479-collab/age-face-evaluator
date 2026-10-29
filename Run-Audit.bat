@echo off
cd /d "%~dp0"
"%~dp0runtime\python.exe" "%~dp0bootstrap.py" audit %*
if errorlevel 1 pause
