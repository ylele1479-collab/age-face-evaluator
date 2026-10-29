@echo off
cd /d "%~dp0"
"%~dp0runtime\python.exe" "%~dp0bootstrap.py" selfcheck %*
if errorlevel 1 pause
